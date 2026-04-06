from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from arcengine import (
    ActionInput,
    ARCBaseGame,
    Camera,
    GameAction,
    Level,
    RenderableUserDisplay,
    Sprite,
)
from gymnasium import spaces

WHITE = 0
LIGHT_GREY = 2
GREY = 3
DARK_GREY = 4
BLACK = 5
RED = 8
BLUE = 9
YELLOW = 11
ORANGE = 12
GREEN = 14
PURPLE = 15

CELL_COLORS = {
    0: WHITE,
    1: BLUE,
    2: PURPLE,
    3: RED,
}

COL_BORDER = DARK_GREY
COL_CURSOR = YELLOW
COL_CURSOR_DOT = ORANGE
COL_WIN = GREEN
COL_PREVIEW = LIGHT_GREY

TILE_PX_MAP = {5: 10, 6: 9}
BAR_IW = 42

MODE_ROW_COL = "row_col"
MODE_DIAG = "diag"
MODE_PARITY = "parity"
MODE_RING = "ring"
MODE_MIXED = "mixed"

TILE_CIRCLE = 0
TILE_SQUARE = 1
TILE_DIAMOND = 2

_EFF_DIAG = "_diag"

CURSOR_DIRECTIONS = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
PULSE_ACTION = 5
UNDO_ACTION = 7

MAX_LIVES = 3

LEVEL_CONFIGS = [
    {
        "grid_size": 5,
        "pulse_mode": MODE_ROW_COL,
        "mod": 2,
        "seeds": [(1, 2, 1), (3, 3, 1)],
        "max_moves": 48,
    },
    {
        "grid_size": 5,
        "pulse_mode": MODE_DIAG,
        "mod": 2,
        "seeds": [(0, 2, 1), (2, 0, 1), (4, 3, 1)],
        "max_moves": 84,
    },
    {
        "grid_size": 5,
        "pulse_mode": MODE_RING,
        "mod": 2,
        "seeds": [(1, 1, 1), (1, 3, 1), (3, 1, 1), (3, 3, 1)],
        "max_moves": 72,
    },
    {
        "grid_size": 6,
        "pulse_mode": MODE_MIXED,
        "mod": 4,
        "seeds": [
            (0, 0, 3),
            (1, 0, 3),
            (2, 0, 3),
            (3, 3, 3),
            (4, 1, 3),
            (0, 4, 3),
            (5, 2, 3),
        ],
        "max_moves": 126,
    },
]

NUM_LEVELS = len(LEVEL_CONFIGS)


def _make_tile_types(mode: str, gs: int):
    if mode == MODE_PARITY:
        return [[(r + c) % 2 for c in range(gs)] for r in range(gs)]
    if mode == MODE_MIXED:
        return [[(r + c) % 3 for c in range(gs)] for r in range(gs)]
    return None


def _resolve_pulse_type(mode, r, c, ttypes):
    if mode == MODE_DIAG:
        return _EFF_DIAG
    if mode == MODE_PARITY:
        return MODE_ROW_COL if (r + c) % 2 == 0 else _EFF_DIAG
    if mode == MODE_MIXED and ttypes:
        t = ttypes[r][c]
        if t == TILE_CIRCLE:
            return MODE_ROW_COL
        if t == TILE_SQUARE:
            return _EFF_DIAG
        if t == TILE_DIAMOND:
            return MODE_RING
    return mode


def _affected_cells(r, c, gs, mode, ttypes):
    out = set()
    effective_pulse = _resolve_pulse_type(mode, r, c, ttypes)

    if effective_pulse == MODE_ROW_COL:
        for i in range(gs):
            out.add((r, i))
        for i in range(gs):
            out.add((i, c))
    elif effective_pulse == _EFF_DIAG:
        diff = r - c
        for i in range(gs):
            j = i - diff
            if 0 <= j < gs:
                out.add((i, j))
        total = r + c
        for i in range(gs):
            j = total - i
            if 0 <= j < gs:
                out.add((i, j))
    elif effective_pulse == MODE_RING:
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                if max(abs(dr), abs(dc)) == 2 and not (abs(dr) == 2 and abs(dc) == 2):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < gs and 0 <= nc < gs:
                        out.add((nr, nc))
    return out


def apply_pulse(grid, r, c, gs, mode, ttypes, mod):
    for nr, nc in _affected_cells(r, c, gs, mode, ttypes):
        grid[nr][nc] = (grid[nr][nc] + 1) % mod


def _check_all_zero(grid, gs):
    for r in range(gs):
        for c in range(gs):
            if grid[r][c] != 0:
                return False
    return True


def _generate_start_grid(idx):
    cfg = LEVEL_CONFIGS[idx]
    gs = cfg["grid_size"]
    mode = cfg["pulse_mode"]
    mod = cfg["mod"]
    ttypes = _make_tile_types(mode, gs)
    grid = [[0] * gs for _ in range(gs)]
    for r, c, times in cfg["seeds"]:
        for _ in range(times):
            apply_pulse(grid, r, c, gs, mode, ttypes, mod)
    return grid


ALL_STARTS: List[List[List[int]]] = []
for _i in range(NUM_LEVELS):
    _g = _generate_start_grid(_i)
    if _check_all_zero(_g, LEVEL_CONFIGS[_i]["grid_size"]):
        raise ValueError("Level {} is trivially solved!".format(_i + 1))
    ALL_STARTS.append(_g)


_marker = Sprite(
    pixels=[[BLACK]],
    name="marker",
    visible=False,
    collidable=False,
    tags=["marker"],
    layer=0,
)

levels = []
for _i in range(NUM_LEVELS):
    levels.append(
        Level(
            sprites=[_marker.clone().set_position(0, 0)],
            grid_size=(64, 64),
            data={"idx": _i},
            name="Level " + str(_i + 1),
        )
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


ARC_PALETTE = np.array(
    [
        [255, 255, 255],
        [204, 204, 204],
        [153, 153, 153],
        [102, 102, 102],
        [ 51,  51,  51],
        [  0,   0,   0],
        [229,  58, 163],
        [255, 123, 204],
        [249,  60,  49],
        [ 30, 147, 255],
        [136, 216, 241],
        [255, 220,   0],
        [255, 133,  27],
        [146,  18,  49],
        [ 79, 204,  48],
        [163,  86, 208],
    ],
    dtype=np.uint8,
)

_COLOR_CHAR = "0123456789abcdef"


class PulseHud(RenderableUserDisplay):
    def __init__(self):
        self.game: Optional[Eg02] = None

    @staticmethod
    def _px(frame, x, y, col):
        if 0 <= x < 64 and 0 <= y < 64:
            frame[y, x] = col

    def _draw_tile(self, frame, px, py, value):
        g = self.game
        inner = g.tile_px - 1
        col = CELL_COLORS.get(value, WHITE)

        for dy in range(g.tile_px):
            for dx in range(g.tile_px):
                fx, fy = px + dx, py + dy
                if not (0 <= fx < 64 and 0 <= fy < 64):
                    continue
                if dx >= inner or dy >= inner:
                    frame[fy, fx] = COL_BORDER
                    continue
                frame[fy, fx] = col

    def _draw_cursor(self, frame):
        g = self.game
        px = g.offset_x + g.cursor_col * g.tile_px
        py = g.offset_y + g.cursor_row * g.tile_px
        tp = g.tile_px

        for d in range(tp):
            self._px(frame, px + d, py, COL_CURSOR)
            self._px(frame, px + d, py + tp - 1, COL_CURSOR)
            self._px(frame, px, py + d, COL_CURSOR)
            self._px(frame, px + tp - 1, py + d, COL_CURSOR)

        for cx, cy in [
            (px, py),
            (px + tp - 1, py),
            (px, py + tp - 1),
            (px + tp - 1, py + tp - 1),
        ]:
            self._px(frame, cx, cy, COL_CURSOR_DOT)

    def _draw_preview(self, frame):
        g = self.game
        cells = _affected_cells(
            g.cursor_row,
            g.cursor_col,
            g.grid_size,
            g.pulse_mode,
            g.tile_types,
        )
        half = (g.tile_px - 1) // 2
        for r, c in cells:
            if r == g.cursor_row and c == g.cursor_col:
                continue
            cx = g.offset_x + c * g.tile_px + half
            cy = g.offset_y + r * g.tile_px + half
            self._px(frame, cx, cy, COL_PREVIEW)
            if g.tile_px >= 10:
                self._px(frame, cx + 1, cy, COL_PREVIEW)
                self._px(frame, cx, cy + 1, COL_PREVIEW)
                self._px(frame, cx + 1, cy + 1, COL_PREVIEW)

    def _draw_win_border(self, frame):
        g = self.game
        gx = g.offset_x - 1
        gy = g.offset_y - 1
        gw = g.grid_px + 2
        gh = g.grid_px + 2
        for d in range(gw):
            self._px(frame, gx + d, gy, COL_WIN)
            self._px(frame, gx + d, gy + gh - 1, COL_WIN)
        for d in range(gh):
            self._px(frame, gx, gy + d, COL_WIN)
            self._px(frame, gx + gw - 1, gy + d, COL_WIN)

    def _draw_progress_bar(self, frame):
        g = self.game
        bx, by = 2, 0
        bw, bh = 44, 4
        for dx in range(bw):
            self._px(frame, bx + dx, by, GREY)
            self._px(frame, bx + dx, by + bh - 1, GREY)
        for dy in range(bh):
            self._px(frame, bx, by + dy, GREY)
            self._px(frame, bx + bw - 1, by + dy, GREY)
        ix, iy = bx + 1, by + 1
        ih = bh - 2
        for dy in range(ih):
            for dx in range(BAR_IW):
                self._px(frame, ix + dx, iy + dy, DARK_GREY)
        if g.max_moves > 0:
            fill_w = max(0, BAR_IW - round(g.move_count * BAR_IW / g.max_moves))
        else:
            fill_w = BAR_IW
        ratio = fill_w / BAR_IW if BAR_IW > 0 else 1.0
        if ratio > 0.5:
            col = GREEN
        elif ratio > 0.25:
            col = YELLOW
        else:
            col = RED
        for dy in range(ih):
            for dx in range(fill_w):
                self._px(frame, ix + dx, iy + dy, col)

    def _draw_lives(self, frame, lives):
        y = 2
        for i in range(MAX_LIVES):
            x = 61 - (i * 5)
            c = WHITE if i < lives else DARK_GREY
            for dy in range(3):
                for dx in range(3):
                    self._px(frame, x - dx, y + dy, c)

    def _draw_goal_hint(self, frame):
        hy = 61
        sq = 3
        sx = (64 - sq) // 2
        for dx in range(sq):
            for dy in range(sq):
                self._px(frame, sx + dx, hy + dy, WHITE)

    def render_interface(self, frame):
        g = self.game
        if g is None:
            return frame

        if not g.won and not g.game_over:
            bx = g.offset_x - 1
            by = g.offset_y - 1
            bw = g.grid_px + 1
            bh = g.grid_px + 1
            for d in range(bw + 1):
                self._px(frame, bx + d, by, GREY)
                self._px(frame, bx + d, by + bh, GREY)
            for d in range(bh + 1):
                self._px(frame, bx, by + d, GREY)
                self._px(frame, bx + bw, by + d, GREY)

        for r in range(g.grid_size):
            for c in range(g.grid_size):
                px = g.offset_x + c * g.tile_px
                py = g.offset_y + r * g.tile_px
                self._draw_tile(frame, px, py, g.grid[r][c])

        if not g.won and not g.game_over:
            self._draw_preview(frame)
            self._draw_cursor(frame)

        if g.won:
            self._draw_win_border(frame)

        if not g.won and not g.game_over:
            self._draw_progress_bar(frame)
            self._draw_lives(frame, g.lives)
            self._draw_goal_hint(frame)

        return frame


class Eg02(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)
        self.hud = PulseHud()
        self.hud.game = self

        self.grid_size = 5
        self.tile_px = 10
        self.grid_px = 50
        self.offset_x = 7
        self.offset_y = 7
        self.pulse_mode = MODE_ROW_COL
        self.tile_types = None
        self.mod = 2
        self.grid: List[List[int]] = [[0] * 5 for _ in range(5)]
        self.cursor_row = 0
        self.cursor_col = 0
        self.won = False
        self.game_over = False
        self.level_num = 1
        self.move_count = 0
        self.max_moves = 0
        self.lives = MAX_LIVES
        self.consecutive_resets = 0

        self._snapshots: List[Dict[str, Any]] = []
        self._game_over = False

        super().__init__(
            "eg02",
            levels,
            Camera(0, 0, 64, 64, BLACK, BLACK, [self.hud]),
            available_actions=[0, 1, 2, 3, 4, 5, 7],
        )

    def _engine_save_snapshot(self):
        self._snapshots.append({
            "grid": [row[:] for row in self.grid],
            "cursor_row": self.cursor_row,
            "cursor_col": self.cursor_col,
            "move_count": self.move_count,
            "lives": self.lives,
        })

    def _engine_restore_snapshot(self) -> bool:
        if not self._snapshots:
            return False
        snap = self._snapshots.pop()
        self.grid = [row[:] for row in snap["grid"]]
        self.cursor_row = snap["cursor_row"]
        self.cursor_col = snap["cursor_col"]
        self.lives = snap["lives"]

        return True

    def _init_level_state(self, reset_lives: bool = True):
        idx = self.current_level.get_data("idx")
        if idx is None:
            idx = 0
        cfg = LEVEL_CONFIGS[idx]

        self.level_num = idx + 1
        self.grid_size = cfg["grid_size"]
        self.pulse_mode = cfg["pulse_mode"]
        self.mod = cfg["mod"]
        self.tile_px = TILE_PX_MAP[self.grid_size]
        self.grid_px = self.tile_px * self.grid_size
        self.offset_x = (64 - self.grid_px) // 2
        self.offset_y = (64 - self.grid_px) // 2
        self.tile_types = _make_tile_types(self.pulse_mode, self.grid_size)

        self.grid = [list(row) for row in ALL_STARTS[idx]]

        self.cursor_row = self._rng.randint(0, self.grid_size - 1)
        self.cursor_col = self._rng.randint(0, self.grid_size - 1)
        self.won = False
        self.game_over = False
        self.move_count = 0
        self.max_moves = cfg.get("max_moves", 0)
        self._snapshots = []

        if reset_lives:
            self.lives = MAX_LIVES

    def _reset_current_level(self):
        self._init_level_state(reset_lives=False)

    def full_reset(self):
        self._current_level_index = 0
        self.on_set_level(self._levels[0])
        self._game_over = False
        self.lives = MAX_LIVES

    def on_set_level(self, level):
        self._init_level_state(reset_lives=True)
        self.consecutive_resets = 0
        self._game_over = False

    def handle_reset(self):
        if self._state.value == "WIN":
            self.full_reset()
            return

        if self._game_over or self._state.value == "GAME_OVER":
            self._reset_current_level()
            self.lives = MAX_LIVES
            self._game_over = False
            return

        self.consecutive_resets += 1
        if self.consecutive_resets >= 2:
            self.consecutive_resets = 0
            self.set_level(0)
        else:
            self._reset_current_level()
            if self.lives <= 0:
                self.lives = MAX_LIVES

    def _handle_death(self):
        self.lives -= 1
        self.consecutive_resets = 0
        if self.lives <= 0:
            self._game_over = True
            self.lives = MAX_LIVES
            self.lose()
            return True
        self._reset_current_level()
        return True

    def _move_cursor(self, dr, dc):
        gs = self.grid_size
        self.cursor_row = max(0, min(self.cursor_row + dr, gs - 1))
        self.cursor_col = max(0, min(self.cursor_col + dc, gs - 1))

    def step(self):
        act = self.action.id

        if act == GameAction.RESET:
            self.complete_action()
            return

        self.consecutive_resets = 0

        self.move_count += 1

        if act == GameAction.ACTION7:
            if self._snapshots:
                self._engine_restore_snapshot()
            if self.max_moves > 0 and self.move_count >= self.max_moves and not self.won:
                self._handle_death()
            self.complete_action()
            return

        self._engine_save_snapshot()

        if act == GameAction.ACTION1:
            self._move_cursor(-1, 0)
        elif act == GameAction.ACTION2:
            self._move_cursor(1, 0)
        elif act == GameAction.ACTION3:
            self._move_cursor(0, -1)
        elif act == GameAction.ACTION4:
            self._move_cursor(0, 1)
        elif act == GameAction.ACTION5:
            apply_pulse(
                self.grid,
                self.cursor_row,
                self.cursor_col,
                self.grid_size,
                self.pulse_mode,
                self.tile_types,
                self.mod,
            )

        if _check_all_zero(self.grid, self.grid_size):
            self.won = True
            self.next_level()
            self.complete_action()
            return

        if self.max_moves > 0 and self.move_count >= self.max_moves and not self.won:
            self._handle_death()
            self.complete_action()
            return

        self.complete_action()


class PuzzleEnvironment:
    _ACTION_MAP: Dict[str, GameAction] = {
        "reset": GameAction.RESET,
        "up": GameAction.ACTION1,
        "down": GameAction.ACTION2,
        "left": GameAction.ACTION3,
        "right": GameAction.ACTION4,
        "select": GameAction.ACTION5,
        "undo": GameAction.ACTION7,
    }

    _VALID_ACTIONS = list(_ACTION_MAP.keys())

    def __init__(self, seed: int = 0, **kwargs) -> None:
        self._engine = Eg02(seed=seed)
        self._total_turns = 0
        self._done = False
        self._game_won = False
        self._game_over = False
        self._last_action_was_reset = False

    def reset(self) -> GameState:
        e = self._engine

        if self._game_won or self._last_action_was_reset:
            e.perform_action(ActionInput(id=GameAction.RESET))
            e.perform_action(ActionInput(id=GameAction.RESET))
        else:
            e.perform_action(ActionInput(id=GameAction.RESET))

        self._total_turns = 0
        self._done = False
        self._last_action_was_reset = True
        self._game_won = False
        self._game_over = False

        return self._build_game_state()

    def get_actions(self) -> List[str]:
        if self._done:
            return ["reset"]
        return list(self._VALID_ACTIONS)

    def is_done(self) -> bool:
        return self._done

    def step(self, action: str) -> StepResult:
        e = self._engine

        if action == "reset":
            state = self.reset()
            return StepResult(
                state=state, reward=0.0, done=False, info={"action": "reset"}
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

        level_before = e._current_level_index

        action_input = ActionInput(id=game_action)
        frame = e.perform_action(action_input, raw=True)

        state_name = frame.state.name if frame and frame.state else ""
        game_won = state_name == "WIN"
        game_over = state_name == "GAME_OVER"

        total_levels = len(self._engine._levels)
        level_reward = 1.0 / total_levels

        if game_won:
            self._done = True
            self._game_won = True
            self._game_over = False
            info["reason"] = "game_complete"
            return StepResult(
                state=self._build_game_state(done=True),
                reward=level_reward,
                done=True,
                info=info,
            )

        if game_over or e._game_over:
            self._done = True
            self._game_won = False
            self._game_over = True
            info["reason"] = "game_over"
            return StepResult(
                state=self._build_game_state(done=True),
                reward=0.0,
                done=True,
                info=info,
            )

        reward = 0.0
        if e._current_level_index != level_before:
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
        gs = e.grid_size
        idx = e.current_level.get_data("idx") or 0
        lines = []
        lines.append(f"level:{idx + 1}/{NUM_LEVELS}")
        lines.append(f"moves:{e.move_count}/{e.max_moves}")
        lines.append(f"lives:{e.lives}/{MAX_LIVES}")
        lines.append(f"cursor:({e.cursor_row},{e.cursor_col})")
        lines.append(f"mode:{e.pulse_mode}")
        lines.append(f"mod:{e.mod}")
        lines.append("grid:")
        for r in range(gs):
            row_chars = ""
            for c in range(gs):
                row_chars += _COLOR_CHAR[e.grid[r][c] % 16]
            lines.append(row_chars)
        if e.won:
            lines.append("status:won")
        elif e._game_over:
            lines.append("status:game_over")
        else:
            lines.append("status:playing")
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
                "total_levels": len(e._levels),
                "level_index": e._current_level_index,
                "levels_completed": getattr(e, "_score", 0),
                "game_over": self._game_over,
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
        "select",
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
