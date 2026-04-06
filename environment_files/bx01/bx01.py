import random
import struct
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from arcengine import (
    ARCBaseGame,
    ActionInput,
    Camera,
    GameAction,
    GameState as EngineState,
    Level,
    Sprite,
)


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


STANDING = 0
LYING_X = 1
LYING_Y = 2

VOID = "."
FLOOR = "F"
GOAL = "G"
PRESS_SW = "P"
PREC_SW = "Q"
BRIDGE_ON = "B"
BRIDGE_OFF = "b"

BG_COLOR = 5
FLOOR_CLR = 0
GOAL_CLR = 14
BLOCK_CLR = 4
PRESS_CLR = 7
PREC_CLR = 11
BR_ON_CLR = 3
BR_OFF_CLR = 13
BAR_FILLED = 6
BAR_EMPTY = 8
LIFE_ON_CLR = 14
LIFE_OFF_CLR = 4
MAX_LIVES = 3

DIR_UP = (0, -1)
DIR_DOWN = (0, 1)
DIR_LEFT = (-1, 0)
DIR_RIGHT = (1, 0)

_tile = Sprite(
    pixels=[[1]],
    name="tile",
    visible=True,
    collidable=True,
)


def _move(orient, x, y, dx, dy):
    if orient == STANDING:
        if dy == -1:
            return (LYING_Y, x, y - 2)
        elif dy == 1:
            return (LYING_Y, x, y + 1)
        elif dx == -1:
            return (LYING_X, x - 2, y)
        elif dx == 1:
            return (LYING_X, x + 1, y)

    elif orient == LYING_X:
        if dy == -1:
            return (LYING_X, x, y - 1)
        elif dy == 1:
            return (LYING_X, x, y + 1)
        elif dx == -1:
            return (STANDING, x - 1, y)
        elif dx == 1:
            return (STANDING, x + 2, y)

    elif orient == LYING_Y:
        if dy == -1:
            return (STANDING, x, y - 1)
        elif dy == 1:
            return (STANDING, x, y + 2)
        elif dx == -1:
            return (LYING_Y, x - 1, y)
        elif dx == 1:
            return (LYING_Y, x + 1, y)

    raise ValueError(f"Invalid orientation or direction: {orient}, ({dx}, {dy})")


def _get_cells(orient, x, y):
    if orient == STANDING:
        return [(x, y)]
    elif orient == LYING_X:
        return [(x, y), (x + 1, y)]
    elif orient == LYING_Y:
        return [(x, y), (x, y + 1)]
    raise ValueError(f"Invalid orientation: {orient}")


_LEVEL_3 = {
    "map": [
        "............",
        ".FFFFF.FFFF.",
        ".FFFFF.FFFF.",
        ".FFFFF.FFFF.",
        ".FFPFFbbFGF.",
        ".FFFFF.FFFF.",
        ".FFFFF.FFFF.",
        "............",
    ],
    "start": (1, 1, STANDING),
    "goal": (9, 4),
    "switches": {
        (3, 4): [(6, 4), (7, 4)],
    },
    "par": 8,
}

_LEVEL_4 = {
    "map": [
        "..............",
        ".FFFFFFFF.....",
        ".FFFFFFFF.....",
        ".FFFQFFFF.....",
        ".FFFFFFFF.....",
        ".FFFFFFFFbbFFF",
        "..........FFFF",
        "..........FGFF",
        "..........FFFF",
        "..............",
    ],
    "start": (1, 1, STANDING),
    "goal": (11, 7),
    "switches": {
        (4, 3): [(9, 5), (10, 5)],
    },
    "par": 16,
}

_LEVEL_5 = {
    "map": [
        ".............",
        ".FFFFF.FFFF..",
        ".FFPFFbbFFFF.",
        ".FFFFF.FQFF..",
        ".FFFFF.FFFF..",
        ".........b...",
        ".........b...",
        ".......FFFFF.",
        ".......FFFFF.",
        ".......FFGFF.",
        ".......FFFFF.",
        ".............",
    ],
    "start": (1, 1, STANDING),
    "goal": (9, 9),
    "switches": {
        (3, 2): [(6, 2), (7, 2)],
        (8, 3): [(9, 5), (9, 6)],
    },
    "par": 18,
}

_LEVEL_6 = {
    "map": [
        "..............",
        ".FFFF..FFFFF..",
        ".FFFF..FFFFF..",
        ".FFFF..FFQFF..",
        ".FPFFbbFFFFF..",
        ".FFFF..FFFFF..",
        "..........b...",
        "..........b...",
        "..FFFFFFFFFFF.",
        "..FFFFFFFFFFF.",
        "..FFFFF.......",
        "..FFPFF..FGFF.",
        "..FFFFFbbFFFF.",
        "..............",
    ],
    "start": (1, 1, STANDING),
    "goal": (10, 11),
    "switches": {
        (2, 4): [(5, 4), (6, 4)],
        (9, 3): [(10, 6), (10, 7)],
        (4, 11): [(7, 12), (8, 12)],
    },
    "par": 20,
}


ALL_LEVELS = [_LEVEL_3, _LEVEL_4, _LEVEL_5, _LEVEL_6]

BLOCK_START_POSITIONS = [
    [(1, 1), (3, 1), (5, 1), (1, 5)],
    [(1, 1), (4, 1), (7, 1), (1, 4)],
    [(1, 1), (3, 1), (1, 3), (3, 3)],
    [(1, 1), (3, 1), (1, 3), (3, 3)],
]


class Bx01(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:

        self._seed = seed
        self._rng = random.Random(seed)

        self._orient: int = STANDING
        self._bx: int = 0
        self._by: int = 0

        self._tiles: dict[tuple[int, int], str] = {}
        self._map_w: int = 0
        self._map_h: int = 0
        self._goal: tuple[int, int] = (0, 0)
        self._switches: dict[tuple[int, int], list[tuple[int, int]]] = {}
        self._precision_switches: set[tuple[int, int]] = set()
        self._bridge_states: dict[tuple[int, int], bool] = {}

        self._lives: int = MAX_LIVES

        self._sprite_at: dict[tuple[int, int], Sprite] = {}
        self._moves: int = 0
        self._history: List[Dict] = []

        game_levels = [
            Level(sprites=[], grid_size=(16, 16), name=f"Level {i + 1}")
            for i in range(len(ALL_LEVELS))
        ]

        camera = Camera(
            background=BG_COLOR,
            letter_box=BG_COLOR,
            width=16,
            height=16,
        )

        super().__init__(
            game_id="bx01",
            levels=game_levels,
            camera=camera,
            available_actions=[0, 1, 2, 3, 4, 7],
            seed=seed,
        )

    def _place(self, x: int, y: int, color: int) -> None:
        sprite = _tile.clone().color_remap(None, color).set_position(x, y)
        self.current_level.add_sprite(sprite)
        self._sprite_at[(x, y)] = sprite

    def _remove_sprite(self, x: int, y: int) -> None:
        pos = (x, y)
        if pos in self._sprite_at:
            self.current_level.remove_sprite(self._sprite_at[pos])
            del self._sprite_at[pos]

    def _get_budget_base(self) -> int:
        if 0 <= self.level_index < len(ALL_LEVELS):
            return ALL_LEVELS[self.level_index]["par"]
        return 1

    def _render_progress_bar(self) -> None:
        bar_y = 15
        bar_len = 10

        par = self._get_budget_base()
        budget = 3 * par

        filled = min(bar_len, self._moves * bar_len // max(budget, 1))

        for i in range(bar_len):
            bx = 1 + i
            self._remove_sprite(bx, bar_y)
            if i < filled:
                self._place(bx, bar_y, BAR_FILLED)
            else:
                self._place(bx, bar_y, BAR_EMPTY)

    def _render_lives(self) -> None:
        lives_y = 15
        for i in range(MAX_LIVES):
            lx = 12 + i
            self._remove_sprite(lx, lives_y)
            clr = LIFE_ON_CLR if i < self._lives else LIFE_OFF_CLR
            self._place(lx, lives_y, clr)

    def _tile_at(self, x: int, y: int) -> str:
        return self._tiles.get((x, y), VOID)

    def _is_passable(self, x: int, y: int) -> bool:
        ch = self._tile_at(x, y)
        if ch == VOID:
            return False
        if ch in (FLOOR, GOAL, PRESS_SW, PREC_SW):
            return True
        if ch in (BRIDGE_ON, BRIDGE_OFF):
            return self._bridge_states.get((x, y), False)
        return False

    def _get_tile_color(self, x: int, y: int) -> int:
        ch = self._tile_at(x, y)
        if ch == VOID:
            return BG_COLOR
        elif ch == FLOOR:
            return FLOOR_CLR
        elif ch == GOAL:
            return GOAL_CLR
        elif ch == PRESS_SW:
            return PRESS_CLR
        elif ch == PREC_SW:
            return PREC_CLR
        elif ch in (BRIDGE_ON, BRIDGE_OFF):
            if self._bridge_states.get((x, y), False):
                return BR_ON_CLR
            else:
                return BR_OFF_CLR
        return BG_COLOR

    def _render_all(self) -> None:
        self.current_level.remove_all_sprites()
        self._sprite_at.clear()

        ox = (16 - self._map_w) // 2
        oy = (15 - self._map_h) // 2
        if oy < 0:
            oy = 0
        if ox < 0:
            ox = 0

        for (mx, my), ch in self._tiles.items():
            sx, sy = mx + ox, my + oy
            if 0 <= sx < 16 and 0 <= sy < 15:
                color = self._get_tile_color(mx, my)
                if color != BG_COLOR:
                    self._place(sx, sy, color)

        block_cells = _get_cells(self._orient, self._bx, self._by)
        block_color = BLOCK_CLR
        for cx, cy in block_cells:
            sx, sy = cx + ox, cy + oy
            if 0 <= sx < 16 and 0 <= sy < 15:
                self._remove_sprite(sx, sy)
                self._place(sx, sy, block_color)

        self._render_progress_bar()
        self._render_lives()

    def _activate_switches(self) -> None:
        cells = _get_cells(self._orient, self._bx, self._by)
        for cx, cy in cells:
            if (cx, cy) in self._switches:
                if (cx, cy) in self._precision_switches:
                    if self._orient != STANDING:
                        continue
                for bx_s, by_s in self._switches[(cx, cy)]:
                    self._bridge_states[(bx_s, by_s)] = not self._bridge_states.get(
                        (bx_s, by_s), False
                    )

    def _build_level(self, idx: int) -> None:
        data = ALL_LEVELS[idx]
        raw_map = data["map"]

        self._tiles = {}
        self._bridge_states = {}
        self._map_h = len(raw_map)
        self._map_w = max(len(row) for row in raw_map)

        for y, row in enumerate(raw_map):
            for x, ch in enumerate(row):
                self._tiles[(x, y)] = ch
                if ch == BRIDGE_ON:
                    self._bridge_states[(x, y)] = True
                elif ch == BRIDGE_OFF:
                    self._bridge_states[(x, y)] = False

        sx, sy, so = data["start"]
        positions = BLOCK_START_POSITIONS[idx]
        start_x, start_y = self._rng.choice(positions)
        self._bx = start_x
        self._by = start_y
        self._orient = so
        self._goal = data["goal"]

        self._switches = data["switches"]
        self._precision_switches = set()
        for sw_x, sw_y in self._switches:
            if self._tiles.get((sw_x, sw_y)) == PREC_SW:
                self._precision_switches.add((sw_x, sw_y))

        self._moves = 0
        self._history = []
        self._render_all()

    def on_set_level(self, level: Level) -> None:
        level.remove_all_sprites()
        self._sprite_at.clear()
        self._build_level(self.level_index)

    def handle_reset(self) -> None:
        self._rng = random.Random(self._seed)
        self._lives = MAX_LIVES
        self._history = []
        super().handle_reset()

    def _save_state(self) -> None:
        self._history.append(
            {
                "orient": self._orient,
                "bx": self._bx,
                "by": self._by,
                "moves": self._moves,
                "bridge_states": dict(self._bridge_states),
            }
        )

    def _undo(self) -> None:
        if not self._history:
            return
        snap = self._history.pop()
        self._orient = snap["orient"]
        self._bx = snap["bx"]
        self._by = snap["by"]
        self._moves = snap["moves"]
        self._bridge_states = snap["bridge_states"]
        self._render_all()

    def _handle_move(self, dx: int, dy: int) -> None:
        new_orient, nx, ny = _move(self._orient, self._bx, self._by, dx, dy)
        new_cells = _get_cells(new_orient, nx, ny)

        falls = False
        for cx, cy in new_cells:
            if not self._is_passable(cx, cy):
                falls = True
                break
        if falls:
            self._lose_life()
            return

        self._save_state()
        self._orient = new_orient
        self._bx = nx
        self._by = ny
        self._moves += 1

        self._activate_switches()

        self._render_all()

        gx, gy = self._goal
        if self._orient == STANDING and self._bx == gx and self._by == gy:
            self.next_level()
            return

        budget = 3 * self._get_budget_base()
        if self._moves >= budget:
            self._lose_life()

    def _lose_life(self) -> None:
        self._lives -= 1
        if self._lives <= 0:
            self.lose()
        else:
            self._reset_level_state()

    def _reset_level_state(self) -> None:
        self._build_level(self.level_index)

    def step(self) -> None:
        action_id = self.action.id

        if action_id == GameAction.ACTION7:
            self._undo()
            self.complete_action()
            return

        if action_id == GameAction.ACTION1:
            self._handle_move(0, -1)
        elif action_id == GameAction.ACTION2:
            self._handle_move(0, 1)
        elif action_id == GameAction.ACTION3:
            self._handle_move(-1, 0)
        elif action_id == GameAction.ACTION4:
            self._handle_move(1, 0)

        self.complete_action()


class PuzzleEnvironment:
    ARC_PALETTE = [
        (255, 255, 255),
        (204, 204, 204),
        (153, 153, 153),
        (102, 102, 102),
        (51, 51, 51),
        (0, 0, 0),
        (229, 58, 163),
        (255, 123, 204),
        (249, 60, 49),
        (30, 147, 255),
        (136, 216, 241),
        (255, 220, 0),
        (255, 133, 27),
        (146, 18, 49),
        (79, 204, 48),
        (163, 86, 208),
    ]

    ACTION_MAP = {
        "up": 1,
        "down": 2,
        "left": 3,
        "right": 4,
        "reset": 0,
        "undo": 7,
    }

    def __init__(self, seed: int = 0) -> None:
        self._engine = Bx01(seed=seed)
        self._total_turns = 0
        self._last_action_was_reset = False
        self._prev_score = 0

    def _build_text_obs(self) -> str:
        e = self._engine
        ox = (16 - e._map_w) // 2
        oy = (15 - e._map_h) // 2
        if oy < 0:
            oy = 0
        if ox < 0:
            ox = 0

        grid = [["." for _ in range(e._map_w)] for _ in range(e._map_h)]

        for (mx, my), ch in e._tiles.items():
            if 0 <= mx < e._map_w and 0 <= my < e._map_h:
                if ch == VOID:
                    grid[my][mx] = "."
                elif ch == FLOOR:
                    grid[my][mx] = "F"
                elif ch == GOAL:
                    grid[my][mx] = "G"
                elif ch == PRESS_SW:
                    grid[my][mx] = "P"
                elif ch == PREC_SW:
                    grid[my][mx] = "Q"
                elif ch in (BRIDGE_ON, BRIDGE_OFF):
                    if e._bridge_states.get((mx, my), False):
                        grid[my][mx] = "B"
                    else:
                        grid[my][mx] = "b"

        block_cells = _get_cells(e._orient, e._bx, e._by)
        for cx, cy in block_cells:
            if 0 <= cx < e._map_w and 0 <= cy < e._map_h:
                grid[cy][cx] = "@"

        orient_names = {STANDING: "STANDING", LYING_X: "LYING_X", LYING_Y: "LYING_Y"}
        par = ALL_LEVELS[e.level_index]["par"] if e.level_index < len(ALL_LEVELS) else 1
        budget = 3 * par

        header = (
            f"Level:{e.level_index + 1}/{len(ALL_LEVELS)} "
            f"Lives:{e._lives} "
            f"Moves:{e._moves}/{budget} "
            f"Orient:{orient_names.get(e._orient, '?')}"
        )

        grid_text = "\n".join("".join(row) for row in grid)
        return header + "\n" + grid_text

    @staticmethod
    def _encode_png(rgb: np.ndarray) -> bytes:
        h, w, _ = rgb.shape
        raw_rows = []
        for y in range(h):
            raw_rows.append(b"\x00" + rgb[y].tobytes())
        raw_data = b"".join(raw_rows)
        compressed = zlib.compress(raw_data)

        def _chunk(ctype: bytes, data: bytes) -> bytes:
            c = ctype + data
            return (
                struct.pack(">I", len(data))
                + c
                + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
            )

        ihdr_data = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
        png = b"\x89PNG\r\n\x1a\n"
        png += _chunk(b"IHDR", ihdr_data)
        png += _chunk(b"IDAT", compressed)
        png += _chunk(b"IEND", b"")
        return png

    def _build_image_bytes(self) -> Optional[bytes]:
        e = self._engine
        index_grid = e.camera.render(e.current_level.get_sprites())
        if index_grid is None or index_grid.size == 0:
            return None
        h, w = index_grid.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for idx, color in enumerate(self.ARC_PALETTE):
            mask = index_grid == idx
            rgb[mask] = color
        return self._encode_png(rgb)

    def _build_game_state(
        self, done: bool = False, info: Optional[Dict] = None
    ) -> GameState:
        e = self._engine
        valid_actions = self.get_actions() if not done else None
        return GameState(
            text_observation=self._build_text_obs(),
            image_observation=self._build_image_bytes(),
            valid_actions=valid_actions,
            turn=self._total_turns,
            metadata={
                "total_levels": len(e._levels),
                "level_index": e.level_index,
                "levels_completed": getattr(e, "_score", 0),
                "lives": e._lives,
                "moves": e._moves,
                "orient": e._orient,
                "game_over": getattr(e._state, "name", "") == "GAME_OVER",
                "done": done,
                "info": info or {},
            },
        )

    def reset(self) -> GameState:
        e = self._engine
        reset_input = ActionInput(id=GameAction.RESET)
        e.perform_action(reset_input)
        self._total_turns = 0
        self._prev_score = getattr(e, "_score", 0)
        self._last_action_was_reset = True
        return self._build_game_state()

    def get_actions(self) -> List[str]:
        e = self._engine
        state_name = getattr(e._state, "name", "")
        if state_name in ("WIN", "GAME_OVER"):
            return ["reset"]
        return ["reset", "up", "down", "left", "right", "undo"]

    def is_done(self) -> bool:
        e = self._engine
        state_name = getattr(e._state, "name", "")
        return state_name == "WIN"

    def step(self, action: str) -> StepResult:
        e = self._engine

        if action == "reset":
            state = self.reset()
            return StepResult(
                state=state, reward=0.0, done=False, info={"action": "reset"}
            )

        base_action = action.split()[0] if action and action.split() else ""
        if base_action not in self.ACTION_MAP:
            return StepResult(
                state=self._build_game_state(),
                reward=0.0,
                done=False,
                info={"action": action, "error": "invalid_action"},
            )

        self._last_action_was_reset = False
        self._total_turns += 1

        game_action_id = self.ACTION_MAP[base_action]
        action_map = {
            1: GameAction.ACTION1,
            2: GameAction.ACTION2,
            3: GameAction.ACTION3,
            4: GameAction.ACTION4,
            7: GameAction.ACTION7,
        }
        game_action = action_map[game_action_id]
        action_input = ActionInput(id=game_action)

        prev_score = self._prev_score
        frame = e.perform_action(action_input, raw=True)
        new_score = frame.levels_completed
        levels_advanced = new_score - prev_score
        self._prev_score = new_score

        total_levels = len(e._levels)
        reward = (
            (levels_advanced / total_levels)
            if (total_levels > 0 and levels_advanced > 0)
            else 0.0
        )
        done = frame.state.name == "WIN"

        info: Dict = {"action": action}
        if done:
            info["reason"] = (
                "game_complete" if frame.state.name == "WIN" else "game_over"
            )

        return StepResult(
            state=self._build_game_state(done=done, info=info),
            reward=reward,
            done=done,
            info=info,
        )

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError(f"Unsupported render mode: {mode}")
        e = self._engine
        index_grid = e.camera.render(e.current_level.get_sprites())
        h, w = index_grid.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for idx, color in enumerate(self.ARC_PALETTE):
            mask = index_grid == idx
            rgb[mask] = color
        if h != 64 or w != 64:
            row_idx = (np.arange(64) * h // 64).astype(int)
            col_idx = (np.arange(64) * w // 64).astype(int)
            rgb = rgb[np.ix_(row_idx, col_idx)]
        return rgb

    def close(self) -> None:
        self._engine = None


class ArcGameEnv(gym.Env):
    metadata: Dict[str, Any] = {
        "render_modes": ["rgb_array"],
        "render_fps": 10,
    }

    ACTION_LIST: List[str] = ["reset", "up", "down", "left", "right", "undo"]

    OBS_HEIGHT: int = 64
    OBS_WIDTH: int = 64

    def __init__(self, seed: int = 0, render_mode: Optional[str] = None) -> None:
        super().__init__()

        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"Unsupported render_mode '{render_mode}'. Supported: {self.metadata['render_modes']}"
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
        self._env: Optional[PuzzleEnvironment] = None

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
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
        terminated: bool = result.done
        truncated: bool = False
        return (
            obs,
            result.reward,
            terminated,
            truncated,
            self._build_info(result.state, result.info),
        )

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
        self, state: GameState, step_info: Optional[Dict] = None
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
