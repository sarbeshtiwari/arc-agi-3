from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from arcengine import (
    ARCBaseGame,
    ActionInput,
    Camera,
    GameAction,
    Level,
    Sprite,
)
from arcengine.interfaces import RenderableUserDisplay
from gymnasium import spaces


@dataclass
class GameState:
    text_observation: str
    image_observation: Optional[bytes]
    valid_actions: Optional[List[str]]
    turn: int
    metadata: dict = field(default_factory=dict)


@dataclass
class StepResult:
    state: GameState
    reward: float
    done: bool
    info: dict = field(default_factory=dict)


ARC_PALETTE = np.array(
    [
        [255, 255, 255],
        [204, 204, 204],
        [153, 153, 153],
        [102, 102, 102],
        [51, 51, 51],
        [0, 0, 0],
        [229, 58, 163],
        [255, 123, 204],
        [249, 60, 49],
        [30, 147, 255],
        [136, 216, 241],
        [255, 220, 0],
        [255, 133, 27],
        [146, 18, 49],
        [79, 204, 48],
        [163, 86, 208],
    ],
    dtype=np.uint8,
)

MAX_LIVES = 3
GW = 14
GH = 16
OFF = -5

C_BG = 0
C_ROAD = 5
C_SAFE = 9
C_HOME = 3
C_FROG = 11
C_CAR1 = 2
C_CAR2 = 6
C_BUS1 = 7
C_BUS2 = 8
C_TRUCK = 14
C_WALL = 1
C_LIVES_ON = 8
C_LIVES_OFF = 4
C_MOVES_ON = 3
C_MOVES_OFF = 0

FROG_START_X = GW // 2
FROG_START_Y = 13

_MAX_VEH_PER_LANE = 6
_MAX_LANES = 8

_TILE_CHAR: Dict[int, str] = {
    C_BG: " ",
    C_ROAD: ".",
    C_SAFE: "_",
    C_HOME: "H",
    C_WALL: "#",
    C_CAR1: "c",
    C_CAR2: "c",
    C_BUS1: "B",
    C_BUS2: "B",
    C_TRUCK: "T",
    C_FROG: "@",
}


class FrHUD(RenderableUserDisplay):
    def __init__(self) -> None:
        self.lives: int = MAX_LIVES
        self.moves_left: int = 1
        self.moves_max: int = 1
        self.flash_timer: int = 0

    def set_lives(self, v: int) -> None:
        self.lives = max(0, min(MAX_LIVES, v))

    def set_moves(self, left: int, total: int) -> None:
        self.moves_left = max(0, left)
        self.moves_max = max(1, total)

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        w = frame.shape[1]
        h = frame.shape[0]

        filled = round(w * self.moves_left / self.moves_max)
        frame[0, :filled] = C_MOVES_ON
        frame[0, filled:] = C_MOVES_OFF

        frame[1, :] = C_LIVES_OFF
        for i in range(MAX_LIVES):
            lx = w - (MAX_LIVES - i) * 3
            c = C_LIVES_ON if i < self.lives else C_LIVES_OFF
            if 0 <= lx < w:
                frame[1, lx] = c
            if 0 <= lx + 1 < w:
                frame[1, lx + 1] = c

        if self.flash_timer > 0:
            frame[0, :] = C_LIVES_ON
            frame[h - 1, :] = C_LIVES_ON
            frame[:, 0] = C_LIVES_ON
            frame[:, w - 1] = C_LIVES_ON

        return frame


def _sp(
    color: int,
    name: str,
    layer: int = 1,
    collidable: bool = False,
    visible: bool = True,
    tags: Optional[List[str]] = None,
) -> Sprite:
    return Sprite(
        pixels=[[color]],
        name=name,
        visible=visible,
        collidable=collidable,
        layer=layer,
        tags=tags or [name],
    )


def _lane(
    y: int,
    direction: int,
    color: int,
    width: int,
    speed: int,
    gap: int,
    starts: List[int],
) -> Dict:
    return {
        "y": y,
        "direction": direction,
        "color": color,
        "width": width,
        "speed": speed,
        "gap": gap,
        "starts": starts,
    }


_LEVEL_SPECS = [
    {
        "name": "Level 1 — Slow Traffic",
        "moves_limit": 60,
        "lanes": [
            _lane(3, +1, C_CAR1, 1, 1, 3, [1, 6, 11]),
            _lane(4, -1, C_CAR2, 1, 1, 4, [3, 8]),
            _lane(8, +1, C_BUS1, 2, 1, 3, [0, 6]),
            _lane(9, -1, C_CAR2, 1, 1, 3, [2, 7, 12]),
        ],
        "inactive_lanes": [5, 6, 10, 11],
    },
    {
        "name": "Level 2 — Mixed Speed",
        "moves_limit": 70,
        "lanes": [
            _lane(3, +1, C_CAR1, 1, 1, 2, [0, 4, 9]),
            _lane(4, -1, C_CAR2, 1, 2, 3, [2, 7]),
            _lane(5, +1, C_BUS1, 2, 1, 4, [1, 8]),
            _lane(6, -1, C_CAR1, 1, 1, 3, [3, 9]),
            _lane(8, +1, C_CAR2, 1, 2, 2, [0, 5, 10]),
            _lane(9, -1, C_BUS2, 2, 1, 3, [2, 9]),
            _lane(10, +1, C_CAR1, 1, 1, 3, [4, 10]),
            _lane(11, -1, C_CAR2, 1, 2, 4, [1, 8]),
        ],
        "inactive_lanes": [],
    },
    {
        "name": "Level 3 — Trucks and Buses",
        "moves_limit": 80,
        "lanes": [
            _lane(3, +1, C_TRUCK, 3, 1, 2, [0, 6]),
            _lane(4, -1, C_CAR2, 1, 2, 2, [1, 5, 9]),
            _lane(5, +1, C_BUS1, 2, 2, 2, [0, 7]),
            _lane(6, -1, C_TRUCK, 3, 1, 2, [2, 8]),
            _lane(8, +1, C_CAR1, 1, 2, 2, [0, 4, 9]),
            _lane(9, -1, C_BUS2, 2, 2, 3, [1, 8]),
            _lane(10, +1, C_TRUCK, 3, 1, 2, [0, 7]),
            _lane(11, -1, C_CAR2, 1, 2, 2, [2, 6, 11]),
        ],
        "inactive_lanes": [],
    },
    {
        "name": "Level 4 — Chaos Mode",
        "moves_limit": 90,
        "lanes": [
            _lane(3, +1, C_CAR1, 1, 2, 3, [0, 5, 10]),
            _lane(4, -1, C_TRUCK, 2, 1, 3, [1, 7]),
            _lane(5, +1, C_BUS1, 2, 2, 3, [0, 7]),
            _lane(6, -1, C_CAR2, 1, 2, 2, [2, 6, 11]),
            _lane(8, +1, C_TRUCK, 2, 2, 3, [0, 6]),
            _lane(9, -1, C_CAR1, 1, 2, 3, [1, 6, 11]),
            _lane(10, +1, C_BUS2, 2, 2, 3, [2, 9]),
            _lane(11, -1, C_TRUCK, 2, 1, 3, [0, 6]),
        ],
        "inactive_lanes": [],
    },
]


def _build_level(spec: Dict) -> Level:
    sps: List[Sprite] = []

    for y in range(GH):
        for x in range(GW):
            if y == 0 or y == GH - 1:
                col = C_WALL
                tag = "wall"
            elif y == 1:
                col = C_HOME
                tag = "home"
            elif y in (2, 7, 12, 13, 14):
                col = C_SAFE
                tag = "safe"
            else:
                col = C_ROAD
                tag = "road"
            sps.append(
                _sp(col, tag, layer=0, collidable=False, tags=[tag]).set_position(x, y)
            )

    for y in range(1, GH - 1):
        sps.append(
            _sp(C_WALL, "wall", layer=1, collidable=True, tags=["wall"]).set_position(
                0, y
            )
        )
        sps.append(
            _sp(C_WALL, "wall", layer=1, collidable=True, tags=["wall"]).set_position(
                GW - 1, y
            )
        )

    sps.append(
        _sp(C_FROG, "frog", layer=5, collidable=False, tags=["frog"]).set_position(
            FROG_START_X, FROG_START_Y
        )
    )

    for lane_i in range(_MAX_LANES):
        for veh_i in range(_MAX_VEH_PER_LANE):
            if lane_i < len(spec["lanes"]):
                lane = spec["lanes"][lane_i]
                col = lane["color"]
            else:
                col = C_CAR1
            sps.append(
                _sp(
                    col,
                    f"veh_{lane_i}_{veh_i}",
                    layer=3,
                    collidable=False,
                    visible=False,
                    tags=["vehicle", f"lane_{lane_i}", f"veh_{lane_i}_{veh_i}"],
                ).set_position(OFF, 1)
            )

    return Level(
        sprites=sps,
        grid_size=(GW, GH),
        data=spec,
        name=spec["name"],
    )


class Vehicle:
    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        direction: int,
        speed: int,
        sprites: List[Sprite],
    ) -> None:
        self.x = x
        self.y = y
        self.width = width
        self.direction = direction
        self.speed = speed
        self.sprites = sprites

    def cells(self) -> List[int]:
        return [self.x + i for i in range(self.width)]

    def update(self) -> None:
        self.x += self.direction * self.speed
        if self.direction == +1:
            if self.x >= GW - 1:
                self.x = 1 - (self.width - 1)
        else:
            if self.x + self.width - 1 < 1:
                self.x = GW - 1
        for i, sp in enumerate(self.sprites):
            cx = self.x + i
            if 0 < cx < GW - 1:
                sp.set_position(cx, self.y)
                sp.set_visible(True)
            else:
                sp.set_position(OFF, 1)
                sp.set_visible(False)


class Fr01(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)
        self._lives: int = MAX_LIVES
        self._action_count: int = 0
        self._game_won: bool = False
        self.is_game_over: bool = False
        self._history: List[Dict] = []

        self._levels = [_build_level(s) for s in _LEVEL_SPECS]
        self.total_levels: int = len(self._levels)
        self.current_level_index: int = 0

        self._hud = FrHUD()

        cam = Camera(
            x=0,
            y=0,
            width=GW,
            height=GH,
            background=C_BG,
            letter_box=C_BG,
            interfaces=[self._hud],
        )

        super().__init__(
            game_id="fr01",
            levels=self._levels,
            camera=cam,
            available_actions=[0, 1, 2, 3, 4, 7],
        )

    def on_set_level(self, level: Level) -> None:
        for i, lv in enumerate(self._levels):
            if lv is level:
                self.current_level_index = i
                break

        spec = _LEVEL_SPECS[self.current_level_index]
        self._moves_max = spec["moves_limit"]
        self._moves_left = self._moves_max

        self._lives = MAX_LIVES
        self._action_count = 0
        self.is_game_over = False
        self._history = []

        self._frog_spr: Sprite = self.current_level.get_sprites_by_tag("frog")[0]
        self._fx = FROG_START_X
        self._fy = FROG_START_Y
        self._frog_spr.set_position(self._fx, self._fy)
        self._frog_spr.set_visible(True)

        self._vehicles: List[Vehicle] = []
        for lane_i, lane in enumerate(spec["lanes"]):
            veh_sprs_all = self.current_level.get_sprites_by_tag(f"lane_{lane_i}")
            veh_pool = list(veh_sprs_all)
            w = lane["width"]
            veh_pool_idx = 0
            for start_x in lane["starts"]:
                sprites = []
                for wi in range(w):
                    if veh_pool_idx < len(veh_pool):
                        sp = veh_pool[veh_pool_idx]
                        sp.color_remap(None, lane["color"])
                        sprites.append(sp)
                        veh_pool_idx += 1
                if len(sprites) < w:
                    continue
                veh = Vehicle(
                    start_x, lane["y"], w, lane["direction"], lane["speed"], sprites
                )
                for i, sp in enumerate(sprites):
                    cx = start_x + i
                    if 0 < cx < GW - 1:
                        sp.set_position(cx, lane["y"])
                        sp.set_visible(True)
                    else:
                        sp.set_position(OFF, 1)
                        sp.set_visible(False)
                self._vehicles.append(veh)

        self._grace = 0

        self._hud.set_lives(self._lives)
        self._hud.set_moves(self._moves_left, self._moves_max)
        self._hud.flash_timer = 0

    def _frog_hit(self) -> bool:
        if self._grace > 0:
            return False
        for veh in self._vehicles:
            if veh.y == self._fy and self._fx in veh.cells():
                return True
        return False

    def _respawn(self) -> None:
        self._lives -= 1
        self._hud.set_lives(self._lives)
        self._hud.flash_timer = 2
        self._fx = FROG_START_X
        self._fy = FROG_START_Y
        self._frog_spr.set_position(self._fx, self._fy)
        self._grace = 3

    def _save_state(self) -> None:
        self._history.append(
            {
                "fx": self._fx,
                "fy": self._fy,
                "grace": self._grace,
                "moves_left": self._moves_left,
                "vehicle_xs": [veh.x for veh in self._vehicles],
            }
        )

    def _undo(self) -> None:
        if not self._history:
            return
        snap = self._history.pop()
        self._fx = snap["fx"]
        self._fy = snap["fy"]
        self._grace = snap["grace"]
        self._moves_left = snap["moves_left"]
        self._frog_spr.set_position(self._fx, self._fy)
        for veh, vx in zip(self._vehicles, snap["vehicle_xs"]):
            veh.x = vx
            veh.update()

    def handle_reset(self) -> None:
        self.is_game_over = False
        self._lives = MAX_LIVES
        self._action_count = 0
        self._history = []
        self._hud.set_lives(self._lives)
        self._hud.flash_timer = 0
        self._fx = FROG_START_X
        self._fy = FROG_START_Y
        self._frog_spr.set_position(self._fx, self._fy)
        self._frog_spr.set_visible(True)
        self._grace = 0
        spec = _LEVEL_SPECS[self.current_level_index]
        self._moves_max = spec["moves_limit"]
        self._moves_left = self._moves_max
        self._hud.set_moves(self._moves_left, self._moves_max)
        for veh in self._vehicles:
            veh.update()

    def step(self) -> None:
        if self.action.id == GameAction.RESET:
            self.handle_reset()
            self.complete_action()
            return

        if self._hud.flash_timer > 0:
            self._hud.flash_timer -= 1
            self.complete_action()
            return

        self._action_count += 1
        aid = self.action.id

        if aid == GameAction.ACTION7:
            self._moves_left = max(0, self._moves_left - 1)
            self._hud.set_moves(self._moves_left, self._moves_max)
            if self._moves_left <= 0:
                self.is_game_over = True
                self.complete_action()
                self.lose()
                return
            self._undo()
            self.complete_action()
            return

        self._save_state()

        dx, dy = 0, 0
        if aid == GameAction.ACTION1:
            dy = -1
        elif aid == GameAction.ACTION2:
            dy = +1
        elif aid == GameAction.ACTION3:
            dx = -1
        elif aid == GameAction.ACTION4:
            dx = +1

        nx = self._fx + dx
        ny = self._fy + dy

        if nx <= 0 or nx >= GW - 1:
            nx = self._fx

        if ny <= 0:
            ny = self._fy
        if ny >= GH - 1:
            ny = self._fy

        self._fx = nx
        self._fy = ny
        self._frog_spr.set_position(self._fx, self._fy)

        if self._grace > 0:
            self._grace -= 1

        if self._fy == 1:
            self._lives = MAX_LIVES
            self._hud.set_lives(self._lives)
            if self.current_level_index >= self.total_levels - 1:
                self._game_won = True
            self.next_level()
            self.complete_action()
            return

        for veh in self._vehicles:
            veh.update()

        if self._frog_hit():
            if self._lives <= 1:
                self._lives -= 1
                self._hud.set_lives(self._lives)
                self._hud.flash_timer = 2
                self.is_game_over = True
                self.complete_action()
                self.lose()
                return
            self._respawn()
            self._hud.set_moves(self._moves_left, self._moves_max)
            self.complete_action()
            return

        self._moves_left = max(0, self._moves_left - 1)
        self._hud.set_moves(self._moves_left, self._moves_max)

        if self._moves_left <= 0:
            self.is_game_over = True
            self.complete_action()
            self.lose()
            return

        self.complete_action()


class PuzzleEnvironment:
    _ACTION_MAP: Dict[str, GameAction] = {
        "reset": GameAction.RESET,
        "up": GameAction.ACTION1,
        "down": GameAction.ACTION2,
        "left": GameAction.ACTION3,
        "right": GameAction.ACTION4,
        "undo": GameAction.ACTION7,
    }
    _VALID_ACTIONS = list(_ACTION_MAP.keys())

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self._engine = Fr01(seed=seed)
        self._total_turns = 0
        self._done = False
        self._last_action_was_reset = False

    def _episode_terminal(self) -> bool:
        e = self._engine
        return (
            e._game_won
            or e.is_game_over
            or e.current_level_index >= e.total_levels
            or e._lives <= 0
        )

    def reset(self) -> GameState:
        e = self._engine
        self._total_turns = 0
        self._done = False
        game_won = e._game_won or e.current_level_index >= e.total_levels
        if game_won or e._action_count == 0 or self._last_action_was_reset:
            self._engine = Fr01(seed=self._seed)
            e = self._engine
        else:
            e.perform_action(ActionInput(id=GameAction.RESET))
        self._last_action_was_reset = True
        return self._build_game_state()

    def get_actions(self) -> List[str]:
        if self._episode_terminal():
            return ["reset"]
        return list(self._VALID_ACTIONS)

    def is_done(self) -> bool:
        return self._done or self._episode_terminal()

    def step(self, action: str) -> StepResult:
        e = self._engine

        if action == "reset":
            game_won = e._game_won or e.current_level_index >= e.total_levels
            full_restart = (
                game_won
                or e.is_game_over
                or e._action_count == 0
                or self._last_action_was_reset
            )
            state = self.reset()
            return StepResult(
                state=state,
                reward=0.0,
                done=False,
                info={"action": "reset", "full_restart": full_restart},
            )

        if action not in self._ACTION_MAP:
            raise ValueError(
                f"Invalid action '{action}'. "
                f"Must be one of {list(self._ACTION_MAP.keys())}"
            )

        self._last_action_was_reset = False
        self._total_turns += 1

        game_action = self._ACTION_MAP[action]
        info: Dict[str, Any] = {"action": action}

        level_before = e.current_level_index
        lives_before = e._lives

        action_input = ActionInput(id=game_action)
        frame = e.perform_action(action_input, raw=True)

        state_name = frame.state.name if frame and frame.state else ""
        game_won = state_name == "WIN" or e._game_won
        game_over = state_name == "GAME_OVER" or e.is_game_over

        level_reward = 1.0 / e.total_levels

        if game_won:
            self._done = True
            info["reason"] = "game_complete"
            return StepResult(
                state=self._build_game_state(done=True),
                reward=level_reward,
                done=True,
                info=info,
            )

        if game_over:
            self._done = True
            info["reason"] = "game_over"
            return StepResult(
                state=self._build_game_state(done=True),
                reward=0.0,
                done=True,
                info=info,
            )

        reward = 0.0
        if e.current_level_index != level_before:
            reward = level_reward
            info["reason"] = "level_complete"

        if e._lives < lives_before:
            info["life_lost"] = True

        return StepResult(
            state=self._build_game_state(done=False),
            reward=reward,
            done=False,
            info=info,
        )

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError(f"Unsupported render mode: {mode}")
        e = self._engine
        index_grid = e.camera.render(e.current_level.get_sprites())
        h, w = index_grid.shape[:2]
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for idx in range(len(ARC_PALETTE)):
            mask = index_grid == idx
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            rgb[mask] = ARC_PALETTE[idx]
        out_size = 64
        if h != out_size or w != out_size:
            scale_y = out_size / h
            scale_x = out_size / w
            ys = (np.arange(out_size) / scale_y).astype(int)
            xs = (np.arange(out_size) / scale_x).astype(int)
            ys = np.clip(ys, 0, h - 1)
            xs = np.clip(xs, 0, w - 1)
            rgb = rgb[np.ix_(ys, xs)]
        return rgb

    def close(self) -> None:
        self._engine = None

    def _build_text_observation(self) -> str:
        e = self._engine
        level_idx = e.current_level_index
        spec = _LEVEL_SPECS[level_idx]
        lines: List[str] = []

        lines.append(f'Level {level_idx + 1}/{len(_LEVEL_SPECS)} "{spec["name"]}"')
        lines.append(
            f"Lives: {e._lives}/{MAX_LIVES} | Moves: {e._moves_left}/{e._moves_max}"
        )
        lines.append(f"Frog: ({e._fx},{e._fy})")
        lines.append("")

        veh_cells: Dict[Tuple[int, int], int] = {}
        for veh in e._vehicles:
            for cx in veh.cells():
                if 0 < cx < GW - 1:
                    veh_cells[(cx, veh.y)] = (
                        veh.sprites[0].pixels[0][0] if veh.sprites else C_CAR1
                    )

        lines.append("   " + " ".join(f"{c:2d}" for c in range(GW)))
        for y in range(GH):
            cells: List[str] = []
            for x in range(GW):
                if x == e._fx and y == e._fy:
                    cells.append(" @")
                elif (x, y) in veh_cells:
                    ch = _TILE_CHAR.get(veh_cells[(x, y)], "v")
                    cells.append(f" {ch}")
                elif y == 0 or y == GH - 1:
                    cells.append(" #")
                elif x == 0 or x == GW - 1:
                    cells.append(" #")
                elif y == 1:
                    cells.append(" H")
                elif y in (2, 7, 12, 13, 14):
                    cells.append(" _")
                else:
                    cells.append(" .")
            lines.append(f"{y:2d}" + "".join(cells))

        return "\n".join(lines)

    def _build_game_state(self, done: bool = False) -> GameState:
        e = self._engine
        valid_actions = self.get_actions() if not done else None
        return GameState(
            text_observation=self._build_text_observation(),
            image_observation=None,
            valid_actions=valid_actions,
            turn=self._total_turns,
            metadata={
                "total_levels": e.total_levels,
                "level_index": e.current_level_index,
                "levels_completed": getattr(e, "_score", 0),
                "game_over": e.is_game_over,
                "done": done,
                "info": {},
            },
        )


class ArcGameEnv(gym.Env):
    metadata: Dict[str, Any] = {
        "render_modes": ["rgb_array"],
        "render_fps": 5,
    }

    ACTION_LIST: List[str] = [
        "reset",
        "up",
        "down",
        "left",
        "right",
        "undo",
    ]

    OBS_HEIGHT: int = 64
    OBS_WIDTH: int = 64

    def __init__(
        self,
        seed: int = 0,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()

        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"Unsupported render_mode '{render_mode}'. "
                f"Supported: {self.metadata['render_modes']}"
            )
        self.render_mode: Optional[str] = render_mode

        self._action_to_string: Dict[int, str] = {
            i: a for i, a in enumerate(self.ACTION_LIST)
        }
        self._string_to_action: Dict[str, int] = {
            a: i for i, a in enumerate(self.ACTION_LIST)
        }

        self.observation_space: spaces.Space = spaces.Box(
            low=0,
            high=255,
            shape=(self.OBS_HEIGHT, self.OBS_WIDTH, 3),
            dtype=np.uint8,
        )
        self.action_space: spaces.Space = spaces.Discrete(len(self.ACTION_LIST))

        self._seed: int = seed
        self._env: Any = None

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        if seed is not None:
            self._seed = seed

        self._env = PuzzleEnvironment(seed=self._seed)
        state = self._env.reset()

        return self._get_obs(), self._build_info(state)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action_str: str = self._action_to_string[int(action)]
        result: StepResult = self._env.step(action_str)

        obs = self._get_obs()
        reward: float = result.reward
        terminated: bool = result.done
        truncated: bool = False
        info = self._build_info(result.state, result.info)

        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode == "rgb_array":
            return self._get_obs()
        return None

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None

    def action_mask(self) -> np.ndarray:
        mask = np.zeros(len(self.ACTION_LIST), dtype=np.int8)
        if self._env is not None:
            for a in self._env.get_actions():
                idx = self._string_to_action.get(a)
                if idx is not None:
                    mask[idx] = 1
        return mask

    def _get_obs(self) -> np.ndarray:
        frame = self._env.render(mode="rgb_array")
        if frame.shape[0] != self.OBS_HEIGHT or frame.shape[1] != self.OBS_WIDTH:
            frame = self._resize_nearest(frame, self.OBS_HEIGHT, self.OBS_WIDTH)
        return frame

    @staticmethod
    def _resize_nearest(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        src_h, src_w = img.shape[0], img.shape[1]
        row_idx = (np.arange(target_h) * src_h // target_h).astype(int)
        col_idx = (np.arange(target_w) * src_w // target_w).astype(int)
        return img[np.ix_(row_idx, col_idx)].astype(np.uint8)

    def _build_info(
        self,
        state: GameState,
        step_info: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "text_observation": state.text_observation,
            "valid_actions": state.valid_actions,
            "turn": state.turn,
            "game_metadata": state.metadata,
        }
        if step_info:
            info["step_info"] = step_info
        return info


if __name__ == "__main__":
    from gymnasium.utils.env_checker import check_env

    env = ArcGameEnv(seed=0, render_mode="rgb_array")
    check_env(env.unwrapped, skip_render_check=True)
    obs, info = env.reset()
    mask = env.action_mask()
    valid_indices = np.where(mask == 1)[0]
    if len(valid_indices) > 0:
        obs, reward, term, trunc, info = env.step(valid_indices[0])
    env.close()
