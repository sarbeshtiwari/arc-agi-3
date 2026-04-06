import random
import struct
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import gymnasium as gym
from gymnasium import spaces

from arcengine import (
    ARCBaseGame,
    ActionInput,
    BlockingMode,
    Camera,
    GameAction,
    Level,
    RenderableUserDisplay,
    Sprite,
)


BG = 0
WALL = 5
SPARK = 8
WATER = 10
GHOST = 3

LIFE_ON = 7
LIFE_OFF = 5
MOVE_ON = 12
MOVE_OFF = 6


LEVELS = [
    {
        "name": "Split Merge",
        "budget": 60,
        "grid": [
            "###########",
            "###...#####",
            "###.F.#####",
            "###..F..###",
            "#####WF.###",
            "#####.....#",
            "#######.F.#",
            "#######...#",
            "###########",
        ],
    },
    {
        "name": "Winding Halls",
        "budget": 180,
        "grid": [
            "#############",
            "#.....#.#..##",
            "##.#..###F#.#",
            "#.#.....#.F.#",
            "#......##.#.#",
            "#..#..#.....#",
            "#......W.#W.#",
            "#W..####.#..#",
            "#....##.F..##",
            "#...#F..#...#",
            "#############",
        ],
    },
    {
        "name": "Ghost Hunt",
        "budget": 260,
        "grid": [
            "#############",
            "##.#...##...#",
            "##..F.......#",
            "#.#.##....#.#",
            "#.##.##..F..#",
            "#..G.#.#..###",
            "#.........#.#",
            "#F...#W.....#",
            "#..W#.......#",
            "#...###....##",
            "#############",
        ],
    },
    {
        "name": "Double Haunting",
        "budget": 200,
        "grid": [
            "###############",
            "#...........#.#",
            "#..#.....G.#..#",
            "#F##W.....#.###",
            "#.##.F..##.#..#",
            "##....#....#..#",
            "#.#..##..FF...#",
            "##.WG.#.#.###.#",
            "#......#......#",
            "#.......###.W.#",
            "###############",
        ],
    },
]

SPARK_START_POSITIONS = [
    [
        [(4, 2), (5, 3), (6, 4), (8, 6)],
        [(4, 1), (5, 1), (6, 3), (8, 5)],
        [(5, 2), (7, 3), (7, 4), (9, 6)],
    ],
    [
        [(9, 2), (10, 3), (8, 8), (5, 9)],
        [(9, 1), (10, 3), (8, 5), (5, 9)],
        [(9, 2), (11, 3), (10, 8), (7, 9)],
        [(9, 5), (10, 3), (8, 8), (5, 9)],
    ],
    [
        [(4, 2), (9, 4), (1, 7)],
        [(4, 1), (9, 1), (1, 3)],
        [(11, 2), (11, 4), (4, 7)],
        [(2, 2), (7, 4), (1, 7)],
    ],
    [
        [(1, 3), (5, 4), (9, 6), (10, 6)],
        [(1, 1), (5, 1), (9, 5), (10, 4)],
        [(1, 4), (5, 5), (9, 8), (10, 6)],
    ],
]


class FireHUD(RenderableUserDisplay):
    def __init__(self, game):
        self.game = game

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        total = self.game.TOTAL_LIVES
        current = self.game._lives

        for i in range(total):
            x = 1 + (i * 2)
            y = 1
            if y < len(frame) and x < len(frame[0]):
                if i < current:
                    frame[y][x] = LIFE_ON
                else:
                    frame[y][x] = LIFE_OFF

        width = len(frame[0])
        moves = self.game.moves_left
        max_moves = self.game.max_moves

        bar_start = (self.game.TOTAL_LIVES * 2) + 2
        bar_width = width - bar_start

        if max_moves > 0:
            filled = int(bar_width * moves / max_moves)
        else:
            filled = 0

        for i in range(bar_width):
            x = bar_start + i
            if i < filled:
                frame[0][x] = MOVE_ON
            else:
                frame[0][x] = MOVE_OFF

        return frame


class Av01(ARCBaseGame):
    TOTAL_LIVES = 3

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)
        self.sparks = []
        self.wall_set = set()
        self.water_set = set()

        self.ghosts = []

        self.GW = 0
        self.GH = 0

        self._lives = self.TOTAL_LIVES
        self.max_moves = 0
        self.moves_left = 0
        self.reset_cycle = 0
        self._history: List[Dict] = []

        self.board_sprite = None
        self.hud = FireHUD(self)

        levels = []
        for i, cfg in enumerate(LEVELS):
            gh = len(cfg["grid"])
            gw = len(cfg["grid"][0])
            levels.append(
                Level(
                    sprites=[],
                    grid_size=(gw, gh),
                    name=f"Level {i + 1} - {cfg['name']}",
                )
            )

        gw0 = len(LEVELS[0]["grid"][0])
        gh0 = len(LEVELS[0]["grid"])

        camera = Camera(
            x=0,
            y=0,
            background=BG,
            width=gw0,
            height=gh0,
            letter_box=BG,
            interfaces=[self.hud],
        )

        super().__init__(
            game_id="av01",
            levels=levels,
            camera=camera,
            available_actions=[0, 1, 2, 3, 4, 7],
        )

    def on_set_level(self, level: Level) -> None:
        cfg = LEVELS[self.level_index]
        rows = cfg["grid"]

        self.GH = len(rows)
        self.GW = len(rows[0])

        self.camera.width = self.GW
        self.camera.height = self.GH

        self.max_moves = cfg["budget"]
        self.moves_left = self.max_moves

        self.sparks = []
        self.wall_set = set()
        self.water_set = set()
        self.ghosts = []
        self._history = []

        for y, row in enumerate(rows):
            for x, ch in enumerate(row):
                if ch == "#":
                    self.wall_set.add((x, y))
                elif ch == "W":
                    self.water_set.add((x, y))
                elif ch == "G":
                    self.ghosts.append({"x": x, "y": y, "alive": True})

        spark_config = self._rng.choice(SPARK_START_POSITIONS[self.level_index])
        for x, y in spark_config:
            self.sparks.append({"x": x, "y": y})

        self.board_sprite = self._make_board()
        self.current_level._sprites.clear()
        self.current_level._sprites.append(self.board_sprite)

    def _save_state(self) -> dict:
        return {
            "sparks": [{"x": s["x"], "y": s["y"]} for s in self.sparks],
            "ghosts": [
                {"x": g["x"], "y": g["y"], "alive": g["alive"]} for g in self.ghosts
            ],
            "moves_left": self.moves_left,
        }

    def _restore_state(self, state: dict) -> None:
        self.sparks = [{"x": s["x"], "y": s["y"]} for s in state["sparks"]]
        self.ghosts = [
            {"x": g["x"], "y": g["y"], "alive": g["alive"]} for g in state["ghosts"]
        ]
        self.moves_left = state["moves_left"]
        self._rebuild_board()

    def step(self) -> None:
        action = self.action.id

        if action == GameAction.RESET:
            self.complete_action()
            return

        if action == GameAction.ACTION7:
            if self._history:
                prev = self._history.pop()
                self._restore_state(prev)
            else:
                self.moves_left -= 1
            if self.moves_left < 0:
                self._lose_life()
            self._rebuild_board()
            self.complete_action()
            return

        if action in (
            GameAction.ACTION1,
            GameAction.ACTION2,
            GameAction.ACTION3,
            GameAction.ACTION4,
        ):
            self._history.append(self._save_state())
            self.reset_cycle = 0
            self.moves_left -= 1

        if self.moves_left < 0:
            self._lose_life()
            self.complete_action()
            return

        dx, dy = 0, 0
        if action == GameAction.ACTION1:
            dy = -1
        elif action == GameAction.ACTION2:
            dy = 1
        elif action == GameAction.ACTION3:
            dx = -1
        elif action == GameAction.ACTION4:
            dx = 1

        self._slide_sparks(dx, dy)
        self._slide_ghosts(dx, dy)

        if self._check_water():
            self.complete_action()
            return

        self._check_ghost_water()

        if self._check_ghost_fire_collision():
            self.complete_action()
            return

        if self._check_win():
            self._lives = self.TOTAL_LIVES
            self.next_level()
            self.complete_action()
            return

        self._rebuild_board()
        self.complete_action()

    def _slide_sparks(self, dx, dy):
        if dx == 0 and dy == 0:
            return
        for s in self.sparks:
            x, y = s["x"], s["y"]
            while True:
                nx, ny = x + dx, y + dy
                if nx < 0 or nx >= self.GW or ny < 0 or ny >= self.GH:
                    break
                if (nx, ny) in self.wall_set:
                    break
                x, y = nx, ny
                if (x, y) in self.water_set:
                    break
            s["x"], s["y"] = x, y

    def _slide_ghosts(self, dx, dy):
        if dx == 0 and dy == 0:
            return
        for g in self.ghosts:
            if not g["alive"]:
                continue
            x, y = g["x"], g["y"]
            while True:
                nx, ny = x + dx, y + dy
                if nx < 0 or nx >= self.GW or ny < 0 or ny >= self.GH:
                    break
                if (nx, ny) in self.wall_set:
                    break
                x, y = nx, ny
                if (x, y) in self.water_set:
                    break
            g["x"], g["y"] = x, y

    def _check_ghost_water(self):
        for g in self.ghosts:
            if g["alive"] and (g["x"], g["y"]) in self.water_set:
                g["alive"] = False

    def _check_water(self):
        for s in self.sparks:
            if (s["x"], s["y"]) in self.water_set:
                self._lose_life()
                return True
        return False

    def _check_ghost_fire_collision(self):
        ghost_positions = {(g["x"], g["y"]) for g in self.ghosts if g["alive"]}
        for s in self.sparks:
            if (s["x"], s["y"]) in ghost_positions:
                self._lose_life()
                return True
        return False

    def _check_win(self) -> bool:
        alive_ghosts = [g for g in self.ghosts if g["alive"]]

        if not self.sparks:
            return False

        fire_positions = {(s["x"], s["y"]) for s in self.sparks}

        if len(fire_positions) == 1 and len(alive_ghosts) == 0:
            return True
        return False

    def _lose_life(self):
        self._lives -= 1
        self._history = []
        if self._lives <= 0:
            self.lose()
        else:
            self.set_level(self.level_index)

    def handle_reset(self):
        self._lives = self.TOTAL_LIVES
        if self.reset_cycle == 0:
            self.reset_cycle = 1
            super().handle_reset()
        else:
            self.reset_cycle = 0
            self.set_level(0)

    def _render(self):
        grid = [[BG] * self.GW for _ in range(self.GH)]

        for x, y in self.wall_set:
            grid[y][x] = WALL

        for x, y in self.water_set:
            grid[y][x] = WATER

        for g in self.ghosts:
            if g["alive"]:
                grid[g["y"]][g["x"]] = GHOST

        for s in self.sparks:
            grid[s["y"]][s["x"]] = SPARK

        return grid

    def _make_board(self):
        pixels = np.array(self._render())
        return Sprite(
            pixels=pixels,
            name="board",
            x=0,
            y=0,
            layer=0,
            blocking=BlockingMode.NOT_BLOCKED,
            collidable=False,
        )

    def _rebuild_board(self):
        pixels = self._render()
        for y in range(self.GH):
            for x in range(self.GW):
                self.board_sprite.pixels[y, x] = pixels[y][x]


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

CELL_CHARS = {
    BG: ".",
    WALL: "#",
    SPARK: "F",
    WATER: "W",
    GHOST: "G",
}


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


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    chunk = chunk_type + data
    return (
        struct.pack(">I", len(data))
        + chunk
        + struct.pack(">I", zlib.crc32(chunk) & 0xFFFFFFFF)
    )


def _encode_png(rgb: np.ndarray) -> bytes:
    h, w = rgb.shape[0], rgb.shape[1]
    raw = b""
    for y in range(h):
        raw += b"\x00" + rgb[y].tobytes()
    compressed = zlib.compress(raw)
    out = b"\x89PNG\r\n\x1a\n"
    out += _png_chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    out += _png_chunk(b"IDAT", compressed)
    out += _png_chunk(b"IEND", b"")
    return out


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
        self._engine = Av01(seed=seed)
        self._total_turns = 0
        self._done = False
        self._game_won = False
        self._last_action_was_reset = False

    def reset(self) -> GameState:
        e = self._engine
        if self._game_won or self._last_action_was_reset:
            e.perform_action(ActionInput(id=GameAction.RESET))
            e.perform_action(ActionInput(id=GameAction.RESET))
        else:
            e.perform_action(ActionInput(id=GameAction.RESET))
        self._last_action_was_reset = True
        self._done = False
        self._game_won = False
        self._total_turns = 0
        return self._build_game_state()

    def get_actions(self) -> List[str]:
        if self._done:
            return ["reset"]
        return list(self._VALID_ACTIONS)

    def step(self, action: str) -> StepResult:
        e = self._engine

        if action == "reset":
            state = self.reset()
            return StepResult(
                state=state, reward=0.0, done=False, info={"action": "reset"}
            )

        if action not in self._ACTION_MAP:
            raise ValueError(
                f"Invalid action '{action}'. Must be one of {list(self._ACTION_MAP.keys())}"
            )

        self._last_action_was_reset = False
        self._total_turns += 1

        lives_before = e._lives
        level_before = e.level_index

        game_action = self._ACTION_MAP[action]
        info: Dict = {"action": action}

        action_input = ActionInput(id=game_action)
        frame = e.perform_action(action_input, raw=True)

        state_name = frame.state.name if frame and frame.state else ""
        game_won = state_name == "WIN"
        game_over = state_name == "GAME_OVER"

        total_levels = len(e._levels)
        level_reward = 1.0 / total_levels

        info["lives"] = e._lives
        info["level"] = e.level_index + 1
        info["moves_left"] = e.moves_left
        info["max_moves"] = e.max_moves

        if game_won:
            self._done = True
            self._game_won = True
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
        if e._lives < lives_before:
            info["reason"] = "life_lost"
        elif e.level_index != level_before:
            reward = level_reward
            info["reason"] = "level_complete"

        return StepResult(
            state=self._build_game_state(done=False),
            reward=reward,
            done=False,
            info=info,
        )

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError(f"Unsupported render mode: {mode}")
        index_grid = self._engine.camera.render(
            self._engine.current_level.get_sprites()
        )
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

    def is_done(self) -> bool:
        return self._done

    def _build_text_observation(self) -> str:
        engine = self._engine
        level_idx = engine.level_index
        level_name = LEVELS[level_idx]["name"] if level_idx < len(LEVELS) else "Unknown"
        total_levels = len(engine._levels)

        lines = []
        lines.append(
            f"Level {level_idx + 1}/{total_levels} - {level_name}"
            f" | Lives: {engine._lives}/{engine.TOTAL_LIVES}"
            f" | Moves: {engine.moves_left}/{engine.max_moves}"
            f" | Turn: {self._total_turns}"
        )
        lines.append("")

        grid = engine._render()
        for row in grid:
            row_str = ""
            for cell in row:
                row_str += CELL_CHARS.get(cell, "?")
            lines.append(row_str)

        lines.append("")
        lines.append(f"Sparks: {len(engine.sparks)}")
        alive_ghosts = sum(1 for g in engine.ghosts if g["alive"])
        if engine.ghosts:
            lines.append(f"Ghosts: {alive_ghosts}")
        lines.append("")
        lines.append(f"Actions: {', '.join(self._VALID_ACTIONS)}")

        return "\n".join(lines)

    def _build_game_state(self, done: bool = False) -> GameState:
        engine = self._engine
        level_idx = engine.level_index
        level_name = LEVELS[level_idx]["name"] if level_idx < len(LEVELS) else "Unknown"
        total_levels = len(engine._levels)

        valid_actions = self.get_actions() if not done else None
        image_bytes = _encode_png(self.render())
        alive_ghosts = sum(1 for g in engine.ghosts if g["alive"])

        return GameState(
            text_observation=self._build_text_observation(),
            image_observation=image_bytes,
            valid_actions=valid_actions,
            turn=self._total_turns,
            metadata={
                "total_levels": total_levels,
                "level_index": level_idx,
                "levels_completed": getattr(self._engine, "_score", 0),
                "game_over": getattr(getattr(self._engine, "_state", None), "name", "")
                == "GAME_OVER",
                "done": done,
                "info": {},
                "level": level_idx + 1,
                "level_name": level_name,
                "lives": engine._lives,
                "moves_left": engine.moves_left,
                "max_moves": engine.max_moves,
                "sparks": len(engine.sparks),
                "alive_ghosts": alive_ghosts,
            },
        )


class ArcGameEnv(gym.Env):
    metadata: Dict[str, Any] = {
        "render_modes": ["rgb_array"],
        "render_fps": 5,
    }

    ACTION_LIST: List[str] = ["reset", "up", "down", "left", "right", "undo"]

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
        self._env: Optional[PuzzleEnvironment] = None

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
