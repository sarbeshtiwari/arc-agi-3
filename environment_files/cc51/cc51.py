import io
import random
import struct
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, List

import gymnasium as gym
import numpy as np
from arcengine import (
    ActionInput,
    ARCBaseGame,
    Camera,
    GameAction,
    GameState as EngineState,
    Level,
    RenderableUserDisplay,
    Sprite,
)
from gymnasium import spaces


@dataclass
class GameState:
    text_observation: str
    image_observation: bytes | None
    valid_actions: list[str] | None
    turn: int
    metadata: dict = field(default_factory=dict)


@dataclass
class StepResult:
    state: GameState
    reward: float
    done: bool
    info: dict = field(default_factory=dict)


C_UNPAINTED = 0
C_PAINTED = 4
C_BG = 5
C_PLAYER = 14
C_ENEMY = 8
C_BORDER = 12
C_LEVEL_ON = 12
C_LEVEL_OFF = 5
C_TRAIL = 9

MAX_LIVES = 3


LEVEL_VARIANTS = [
    [
        {"grid_rows": 6, "grid_cols": 6, "enemies": [(5, 5)]},
        {"grid_rows": 6, "grid_cols": 6, "enemies": [(5, 0)]},
        {"grid_rows": 6, "grid_cols": 6, "enemies": [(0, 5)]},
        {"grid_rows": 6, "grid_cols": 6, "enemies": [(3, 3)]},
    ],
    [
        {"grid_rows": 8, "grid_cols": 8, "enemies": [(7, 7)]},
        {"grid_rows": 8, "grid_cols": 8, "enemies": [(7, 0)]},
        {"grid_rows": 8, "grid_cols": 8, "enemies": [(0, 7)]},
        {"grid_rows": 8, "grid_cols": 8, "enemies": [(4, 4)]},
    ],
    [
        {"grid_rows": 10, "grid_cols": 10, "enemies": [(9, 9)]},
        {"grid_rows": 10, "grid_cols": 10, "enemies": [(9, 0)]},
        {"grid_rows": 10, "grid_cols": 10, "enemies": [(0, 9)]},
        {"grid_rows": 10, "grid_cols": 10, "enemies": [(5, 5)]},
    ],
    [
        {"grid_rows": 12, "grid_cols": 12, "enemies": [(11, 11), (0, 11)]},
        {"grid_rows": 12, "grid_cols": 12, "enemies": [(11, 0), (0, 11)]},
        {"grid_rows": 12, "grid_cols": 12, "enemies": [(11, 11), (6, 0)]},
        {"grid_rows": 12, "grid_cols": 12, "enemies": [(6, 11), (11, 0)]},
    ],
    [
        {"grid_rows": 16, "grid_cols": 16, "enemies": [(15, 15), (0, 15)]},
        {"grid_rows": 16, "grid_cols": 16, "enemies": [(15, 0), (0, 15)]},
        {"grid_rows": 16, "grid_cols": 16, "enemies": [(15, 15), (8, 0)]},
        {"grid_rows": 16, "grid_cols": 16, "enemies": [(8, 15), (15, 0)]},
    ],
]


def _make_level(ldef):
    return Level(
        sprites=[],
        grid_size=(64, 64),
        data={
            "grid_rows": ldef["grid_rows"],
            "grid_cols": ldef["grid_cols"],
            "enemies": [list(e) for e in ldef["enemies"]],
        },
    )


levels = [_make_level(vs[0]) for vs in LEVEL_VARIANTS]


class GridPainterDisplay(RenderableUserDisplay):
    def __init__(self):
        self._grid = None
        self._rows = 6
        self._cols = 6
        self._player_row = 0
        self._player_col = 0
        self._enemies = []
        self._level_index = 0
        self._total_levels = 5
        self._cell_size = 4
        self._gap = 1
        self._ox = 0
        self._oy = 0

    def reset(self, grid, rows, cols, player_pos, enemies, level_index):
        self._grid = grid
        self._rows = rows
        self._cols = cols
        self._player_row, self._player_col = player_pos
        self._enemies = enemies
        self._level_index = level_index

        max_pixels = 58

        for cs in [4, 3, 2]:
            gap = 1
            needed = cols * (cs + gap) - gap + 2
            needed_h = rows * (cs + gap) - gap + 2
            if needed <= 62 and needed_h <= 58:
                self._cell_size = cs
                self._gap = gap
                break

        step = self._cell_size + self._gap
        grid_w = cols * step - self._gap
        grid_h = rows * step - self._gap

        self._ox = (64 - grid_w) // 2
        self._oy = 6 + (58 - grid_h) // 2

    @staticmethod
    def _set_pixel(frame, fy, fx, color):
        if 0 <= fy < 64 and 0 <= fx < 64:
            frame[fy, fx] = color

    def _draw_cell(self, frame, py, px, color, size):
        for dy in range(size):
            for dx in range(size):
                self._set_pixel(frame, py + dy, px + dx, color)

    def _cell_screen_pos(self, r, c):
        step = self._cell_size + self._gap
        px = self._ox + c * step
        py = self._oy + r * step
        return py, px

    def _draw_border(self, frame):
        step = self._cell_size + self._gap
        w = self._cols * step - self._gap
        h = self._rows * step - self._gap
        ox, oy = self._ox, self._oy

        for dx in range(-1, w + 1):
            self._set_pixel(frame, oy - 1, ox + dx, C_BORDER)
            self._set_pixel(frame, oy + h, ox + dx, C_BORDER)

        for dy in range(h):
            self._set_pixel(frame, oy + dy, ox - 1, C_BORDER)
            self._set_pixel(frame, oy + dy, ox + w, C_BORDER)

    def _draw_level_indicator(self, frame):

        total_w = self._total_levels * 3 - 1
        start_x = (64 - total_w) // 2
        fy = 2
        for i in range(self._total_levels):
            color = C_LEVEL_ON if i <= self._level_index else C_LEVEL_OFF
            fx = start_x + i * 3
            self._set_pixel(frame, fy, fx, color)
            self._set_pixel(frame, fy, fx + 1, color)
            self._set_pixel(frame, fy + 1, fx, color)
            self._set_pixel(frame, fy + 1, fx + 1, color)

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        if self._grid is None:
            return frame

        cs = self._cell_size

        self._draw_level_indicator(frame)

        self._draw_border(frame)

        for r in range(self._rows):
            for c in range(self._cols):
                py, px = self._cell_screen_pos(r, c)
                self._draw_cell(frame, py, px, self._grid[r][c], cs)

        for er, ec in self._enemies:
            py, px = self._cell_screen_pos(er, ec)
            self._draw_cell(frame, py, px, C_ENEMY, cs)

        py, px = self._cell_screen_pos(self._player_row, self._player_col)
        self._draw_cell(frame, py, px, C_PLAYER, cs)

        return frame


class LivesDisplay(RenderableUserDisplay):
    def __init__(self):
        self.lives = MAX_LIVES

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        for i in range(MAX_LIVES):
            color = C_ENEMY if i < self.lives else C_BG

            x = 61 - i * 3
            frame[2, x] = color
            frame[2, x - 1] = color
            frame[3, x] = color
            frame[3, x - 1] = color
        return frame


class Cc51(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._display = GridPainterDisplay()
        self._ld = LivesDisplay()
        self._lives = MAX_LIVES
        self._mid_level_reset = False
        self._rng = random.Random(seed)
        self._engine_snapshot = None
        self._engine_can_undo = False
        super().__init__(
            "cc51",
            levels,
            Camera(0, 0, 64, 64, C_BG, C_BG, [self._display, self._ld]),
            available_actions=[0, 1, 2, 3, 4, 5, 6, 7],
        )
        self._tick = 0

    def set_level(self, index: int) -> None:
        variant = self._rng.choice(LEVEL_VARIANTS[index])
        new_lv = _make_level(variant)
        self._levels[index] = new_lv.clone()
        self._clean_levels[index] = new_lv.clone()
        super().set_level(index)

    def on_set_level(self, level: Level) -> None:
        self._rows = self.current_level.get_data("grid_rows")
        self._cols = self.current_level.get_data("grid_cols")
        enemy_data = self.current_level.get_data("enemies")

        self._grid = [[C_UNPAINTED] * self._cols for _ in range(self._rows)]

        self._player_row = 0
        self._player_col = 0
        self._grid[0][0] = C_PAINTED

        self._enemies = [list(e) for e in enemy_data]

        self._tick = 0

        self._engine_can_undo = False
        self._engine_snapshot = None

        if not self._mid_level_reset:
            self._lives = MAX_LIVES
        self._mid_level_reset = False
        self._ld.lives = self._lives

        self._display.reset(
            self._grid,
            self._rows,
            self._cols,
            (self._player_row, self._player_col),
            self._enemies,
            self._current_level_index,
        )

    def handle_reset(self) -> None:
        state = getattr(self, "_state", None)
        if state in (EngineState.GAME_OVER, EngineState.WIN):
            self._lives = MAX_LIVES
            self._ld.lives = MAX_LIVES
            super().handle_reset()
        else:
            super().handle_reset()

    def _lose_life(self):

        self._lives -= 1
        self._ld.lives = self._lives
        if self._lives <= 0:
            self.lose()
        else:
            self._mid_level_reset = True
            self.level_reset()

    def _move_player(self, dr, dc):
        nr = self._player_row + dr
        nc = self._player_col + dc
        if 0 <= nr < self._rows and 0 <= nc < self._cols:
            self._player_row = nr
            self._player_col = nc
            self._grid[nr][nc] = C_PAINTED

    def _move_enemies(self):
        for enemy in self._enemies:
            er, ec = enemy
            pr, pc = self._player_row, self._player_col
            dr = pr - er
            dc = pc - ec

            if abs(dr) >= abs(dc):
                if dr != 0:
                    step_r = 1 if dr > 0 else -1
                    new_r = er + step_r
                    if 0 <= new_r < self._rows:
                        enemy[0] = new_r
                elif dc != 0:
                    step_c = 1 if dc > 0 else -1
                    new_c = ec + step_c
                    if 0 <= new_c < self._cols:
                        enemy[1] = new_c
            else:
                if dc != 0:
                    step_c = 1 if dc > 0 else -1
                    new_c = ec + step_c
                    if 0 <= new_c < self._cols:
                        enemy[1] = new_c
                elif dr != 0:
                    step_r = 1 if dr > 0 else -1
                    new_r = er + step_r
                    if 0 <= new_r < self._rows:
                        enemy[0] = new_r

            self._grid[enemy[0]][enemy[1]] = C_UNPAINTED

    def _check_caught(self) -> bool:

        for er, ec in self._enemies:
            if er == self._player_row and ec == self._player_col:
                return True
        return False

    def _check_win(self) -> bool:

        enemy_positions = set()
        for er, ec in self._enemies:
            enemy_positions.add((er, ec))

        for r in range(self._rows):
            for c in range(self._cols):
                if (r, c) in enemy_positions:
                    continue
                if self._grid[r][c] != C_PAINTED:
                    return False
        return True

    def _engine_save_snapshot(self) -> None:
        self._engine_snapshot = {
            "player_row": self._player_row,
            "player_col": self._player_col,
            "enemies": [[er, ec] for er, ec in self._enemies],
            "grid": [row[:] for row in self._grid],
            "tick": self._tick,
        }

    def _engine_restore_snapshot(self) -> None:
        snap = self._engine_snapshot
        self._player_row = snap["player_row"]
        self._player_col = snap["player_col"]
        self._enemies = [[er, ec] for er, ec in snap["enemies"]]
        self._grid = [row[:] for row in snap["grid"]]
        self._display._player_row = self._player_row
        self._display._player_col = self._player_col
        self._display._enemies = self._enemies
        self._display._grid = self._grid

    def step(self) -> None:
        aid = self.action.id

        if aid == GameAction.ACTION7:
            if self._engine_can_undo and self._engine_snapshot is not None:
                self._engine_restore_snapshot()
                self._tick += 1
                self._engine_can_undo = False
                self._engine_snapshot = None
            self.complete_action()
            return

        moved = False
        self._engine_save_snapshot()

        if aid == GameAction.ACTION1:
            self._move_player(-1, 0)
            moved = True
        elif aid == GameAction.ACTION2:
            self._move_player(1, 0)
            moved = True
        elif aid == GameAction.ACTION3:
            self._move_player(0, -1)
            moved = True
        elif aid == GameAction.ACTION4:
            self._move_player(0, 1)
            moved = True
        elif aid == GameAction.ACTION5:
            moved = True
        elif aid == GameAction.ACTION6:
            raw_x = self.action.data.get("x", -1)
            raw_y = self.action.data.get("y", -1)
            coords = self.camera.display_to_grid(raw_x, raw_y)
            if coords:
                gx, gy = coords
                step_size = self._display._cell_size + self._display._gap
                ox = self._display._ox
                oy = self._display._oy
                rel_x = gx - ox
                rel_y = gy - oy
                if rel_x >= 0 and rel_y >= 0:
                    col = rel_x // step_size
                    row = rel_y // step_size
                    in_cell_x = rel_x % step_size
                    in_cell_y = rel_y % step_size
                    if (
                        0 <= row < self._rows
                        and 0 <= col < self._cols
                        and in_cell_x < self._display._cell_size
                        and in_cell_y < self._display._cell_size
                    ):
                        dr = row - self._player_row
                        dc = col - self._player_col
                        if abs(dr) >= abs(dc):
                            if dr != 0:
                                self._move_player(1 if dr > 0 else -1, 0)
                            elif dc != 0:
                                self._move_player(0, 1 if dc > 0 else -1)
                        else:
                            if dc != 0:
                                self._move_player(0, 1 if dc > 0 else -1)
                            elif dr != 0:
                                self._move_player(1 if dr > 0 else -1, 0)
                        moved = True

        if moved and self._check_caught():
            self._engine_can_undo = False
            self._engine_snapshot = None
            self._lose_life()
            self.complete_action()
            return

        if moved and self._check_win():
            self._engine_can_undo = False
            self._engine_snapshot = None
            self.next_level()
            self.complete_action()
            return

        if moved:
            self._tick += 1
            if self._tick % 3 == 0:
                self._move_enemies()

        if moved and self._check_caught():
            self._engine_can_undo = False
            self._engine_snapshot = None
            self._lose_life()
            self.complete_action()
            return

        self._display._player_row = self._player_row
        self._display._player_col = self._player_col
        self._display._enemies = self._enemies
        self._display._grid = self._grid

        if moved:
            self._engine_can_undo = True

        self.complete_action()

    def _grid_to_text(self) -> str:
        painted = self._count_painted()
        total = self._total_cells()
        header = (
            f"Level: {self._current_level_index + 1}/5 | "
            f"Lives: {self._lives}/{MAX_LIVES} | "
            f"Painted: {painted}/{total}"
        )
        rows = [header]
        for r in range(self._rows):
            row_chars = []
            for c in range(self._cols):
                if r == self._player_row and c == self._player_col:
                    row_chars.append("P")
                elif any(er == r and ec == c for er, ec in self._enemies):
                    row_chars.append("E")
                elif self._grid[r][c] == C_PAINTED:
                    row_chars.append("#")
                else:
                    row_chars.append(".")
            rows.append(" ".join(row_chars))
        return "\n".join(rows)

    def _count_painted(self) -> int:
        count = 0
        for r in range(self._rows):
            for c in range(self._cols):
                if self._grid[r][c] == C_PAINTED:
                    count += 1
        return count

    def _total_cells(self) -> int:
        return self._rows * self._cols


class PuzzleEnvironment:
    ARC_PALETTE = [
        (0, 0, 0),
        (0, 116, 217),
        (255, 65, 54),
        (46, 204, 64),
        (255, 220, 0),
        (170, 170, 170),
        (240, 18, 190),
        (255, 133, 27),
        (127, 219, 255),
        (135, 12, 37),
        (0, 48, 73),
        (106, 76, 48),
        (255, 182, 193),
        (80, 80, 80),
        (50, 205, 50),
        (128, 0, 128),
    ]

    _ACTION_MAP: Dict[str, GameAction] = {
        "reset": GameAction.RESET,
        "up": GameAction.ACTION1,
        "down": GameAction.ACTION2,
        "left": GameAction.ACTION3,
        "right": GameAction.ACTION4,
        "select": GameAction.ACTION5,
        "click": GameAction.ACTION6,
        "undo": GameAction.ACTION7,
    }

    _VALID_ACTIONS: List[str] = [
        "up",
        "down",
        "left",
        "right",
        "select",
        "click",
        "undo",
    ]

    def __init__(self, seed: int = 0) -> None:
        self._engine = Cc51(seed=seed)
        self._done = False
        self._last_action_was_reset = False
        self._total_turns = 0

    def _render_frame(self) -> np.ndarray:
        e = self._engine
        return e.camera.render(e.current_level.get_sprites())

    def _frame_to_png(self, frame: np.ndarray) -> bytes:
        h, w = frame.shape[:2]
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for idx, color in enumerate(self.ARC_PALETTE):
            mask = frame == idx
            rgb[mask] = color
        buf = io.BytesIO()
        buf.write(b"\x89PNG\r\n\x1a\n")

        def _chunk(ctype: bytes, data: bytes) -> None:
            buf.write(struct.pack(">I", len(data)))
            buf.write(ctype)
            buf.write(data)
            buf.write(struct.pack(">I", zlib.crc32(ctype + data) & 0xFFFFFFFF))

        _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        raw = b""
        for y in range(h):
            raw += b"\x00" + rgb[y].tobytes()
        _chunk(b"IDAT", zlib.compress(raw))
        _chunk(b"IEND", b"")
        return buf.getvalue()

    def _build_state(self) -> GameState:
        e = self._engine
        frame = self._render_frame()
        png_bytes = self._frame_to_png(frame)

        return GameState(
            text_observation=e._grid_to_text(),
            image_observation=png_bytes,
            valid_actions=self.get_actions(),
            turn=self._total_turns,
            metadata={
                "total_levels": len(self._engine._levels),
                "level": e._current_level_index,
                "lives": e._lives,
                "painted": e._count_painted(),
                "total_cells": e._total_cells(),
                "player_pos": [e._player_row, e._player_col],
                "enemy_positions": [list(ep) for ep in e._enemies],
                "grid_rows": e._rows,
                "grid_cols": e._cols,
            },
        )

    def reset(self) -> GameState:
        e = self._engine
        if self._done and e._state == EngineState.WIN:
            e.full_reset()
        elif self._last_action_was_reset:
            e.full_reset()
        else:
            e.perform_action(ActionInput(id=GameAction.RESET))
        self._done = False
        self._last_action_was_reset = True
        self._total_turns = 0
        return self._build_state()

    def get_actions(self) -> List[str]:
        if self._done:
            return ["reset"]
        return list(self._VALID_ACTIONS)

    def step(self, action: str) -> StepResult:
        parts = action.split()
        action_key = parts[0] if parts else action

        if action_key not in self._ACTION_MAP:
            raise ValueError(
                f"Invalid action '{action_key}'. Must be one of {list(self._ACTION_MAP.keys())}"
            )

        e = self._engine

        if action_key == "reset":
            state = self.reset()
            return StepResult(
                state=state, reward=0.0, done=False, info={"action": "reset"}
            )

        self._last_action_was_reset = False

        game_action = self._ACTION_MAP[action_key]

        if game_action == GameAction.ACTION6:
            if len(parts) < 3:
                parts = ["click", "32", "32"]
            cx = int(parts[1])
            cy = int(parts[2])
            action_input = ActionInput(id=game_action, data={"x": cx, "y": cy})
        else:
            action_input = ActionInput(id=game_action)

        level_before = e._current_level_index

        result = e.perform_action(action_input)

        self._total_turns += 1

        done = False
        reward = 0.0
        info: Dict[str, Any] = {"action": action}

        if result.state == EngineState.WIN:
            done = True
            self._done = True
            reward = 1.0 / len(e._levels)
            info["reason"] = "game_complete"
        elif result.state == EngineState.GAME_OVER:
            done = True
            self._done = True
            info["reason"] = "game_over"
        elif e._current_level_index > level_before:
            reward = 1.0 / len(e._levels)
            info["reason"] = "level_complete"

        return StepResult(
            state=self._build_state(),
            reward=reward,
            done=done,
            info=info,
        )

    def is_done(self) -> bool:
        return self._done

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError(f"Unsupported render mode: {mode}")
        frame = self._render_frame()
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        for idx, color in enumerate(self.ARC_PALETTE):
            mask = frame == idx
            rgb[mask] = color
        return rgb

    def close(self) -> None:
        self._engine = None


class ArcGameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 5}

    ACTION_LIST: list[str] = [
        "reset",
        "up",
        "down",
        "left",
        "right",
        "select",
        "click",
        "undo",
    ]
    OBS_HEIGHT: int = 64
    OBS_WIDTH: int = 64

    def __init__(self, seed: int = 0, render_mode: str | None = None) -> None:
        super().__init__()

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.OBS_HEIGHT, self.OBS_WIDTH, 3),
            dtype=np.uint8,
        )
        self.action_space = spaces.Discrete(len(self.ACTION_LIST))

        self._action_to_string: dict[int, str] = {
            i: a for i, a in enumerate(self.ACTION_LIST)
        }
        self._string_to_action: dict[str, int] = {
            a: i for i, a in enumerate(self.ACTION_LIST)
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self._seed = seed
        self._env: PuzzleEnvironment | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        if seed is not None:
            self._seed = seed
        self._env = PuzzleEnvironment(seed=self._seed)

        state: GameState = self._env.reset()
        obs = self._get_obs()
        info = self._build_info(state)
        return obs, info

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self._env is not None, "Call reset() before step()"

        action_str: str = self._action_to_string[int(action)]

        if action_str == "click":
            action_str = "click 32 32"

        result: StepResult = self._env.step(action_str)

        obs = self._get_obs()
        terminated: bool = result.done
        truncated: bool = False
        info = self._build_info(result.state, step_info=result.info)

        return obs, result.reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode == "rgb_array":
            return self._get_obs()
        return None

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None

    @staticmethod
    def _resize_nearest(img: np.ndarray, h: int, w: int) -> np.ndarray:
        src_h, src_w = img.shape[:2]
        row_idx = (np.arange(h) * src_h) // h
        col_idx = (np.arange(w) * src_w) // w
        return img[np.ix_(row_idx, col_idx)]

    def _get_obs(self) -> np.ndarray:
        assert self._env is not None
        frame = self._env.render(mode="rgb_array")
        h, w = frame.shape[:2]
        if h != self.OBS_HEIGHT or w != self.OBS_WIDTH:
            frame = self._resize_nearest(frame, self.OBS_HEIGHT, self.OBS_WIDTH)
        return frame

    def _build_info(self, state: GameState, step_info: dict | None = None) -> dict:
        info: dict = {
            "text_observation": state.text_observation,
            "valid_actions": state.valid_actions,
            "turn": state.turn,
            "game_metadata": state.metadata,
        }
        if step_info is not None:
            info["step_info"] = step_info
        return info
