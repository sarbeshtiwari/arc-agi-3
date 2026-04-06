import io
import random
import struct
import zlib
from collections import namedtuple
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from arcengine import (
    ActionInput,
    ARCBaseGame,
    Camera,
    GameAction,
    Level,
    RenderableUserDisplay,
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


WHITE = 0
OFF_WHITE = 1
LIGHT_GREY = 2
GREY = 3
DARK_GREY = 4
BLACK = 5
MAGENTA = 6
PINK = 7
RED = 8
BLUE = 9
LIGHT_BLUE = 10
YELLOW = 11
ORANGE = 12
MAROON = 13
GREEN = 14
PURPLE = 15

CIRCLE = 0
TRIANGLE = 1
DIAMOND = 2
SQUARE = 3
NUM_SHAPES = 4

SHAPE_COLORS = {
    CIRCLE: BLUE,
    TRIANGLE: RED,
    DIAMOND: YELLOW,
    SQUARE: GREEN,
}

CELL_BG = DARK_GREY
GRID_SIZE = 64
MAX_LIVES = 3

RULE_SIMPLE = 0
RULE_RIGHT = 1
RULE_CROSS = 2
RULE_CROSS_COND = 3

LayoutInfo = namedtuple(
    "LayoutInfo",
    [
        "cell",
        "gap",
        "main_ox",
        "main_oy",
        "mini_cell",
        "mini_gap",
        "mini_ox",
        "mini_oy",
    ],
)


def _add_to_cell(grid, rr, cc, amt, n):
    if 0 <= rr < n and 0 <= cc < n:
        grid[rr][cc] = (grid[rr][cc] + amt) % NUM_SHAPES


def _has_cardinal_triangle_neighbor(grid, rr, cc, n):
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = rr + dr, cc + dc
        if 0 <= nr < n and 0 <= nc < n and grid[nr][nc] == TRIANGLE:
            return True
    return False


def apply_click(grid, r, c, rule, n):
    g = [row[:] for row in grid]

    if rule == RULE_SIMPLE:
        _add_to_cell(g, r, c, 1, n)

    elif rule == RULE_RIGHT:
        _add_to_cell(g, r, c, 1, n)
        _add_to_cell(g, r, c + 1, 1, n)

    elif rule == RULE_CROSS:
        _add_to_cell(g, r, c, 1, n)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            _add_to_cell(g, r + dr, c + dc, 1, n)

    elif rule == RULE_CROSS_COND:
        _add_to_cell(
            g, r, c, 2 if _has_cardinal_triangle_neighbor(grid, r, c, n) else 1, n
        )
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            _add_to_cell(g, r + dr, c + dc, 1, n)

    return g


LEVEL_CONFIGS = [
    {
        "grid_n": 3,
        "rule": RULE_SIMPLE,
        "start": [
            [CIRCLE, TRIANGLE, CIRCLE],
            [TRIANGLE, CIRCLE, TRIANGLE],
            [CIRCLE, TRIANGLE, CIRCLE],
        ],
        "target": [
            [TRIANGLE, DIAMOND, TRIANGLE],
            [DIAMOND, TRIANGLE, DIAMOND],
            [TRIANGLE, DIAMOND, TRIANGLE],
        ],
        "min_moves": 17,
    },
    {
        "grid_n": 3,
        "rule": RULE_RIGHT,
        "start": [
            [TRIANGLE, CIRCLE, DIAMOND],
            [CIRCLE, TRIANGLE, CIRCLE],
            [DIAMOND, SQUARE, TRIANGLE],
        ],
        "target": [
            [DIAMOND, TRIANGLE, SQUARE],
            [CIRCLE, DIAMOND, TRIANGLE],
            [SQUARE, CIRCLE, DIAMOND],
        ],
        "min_moves": 10,
    },
    {
        "grid_n": 4,
        "rule": RULE_CROSS,
        "start": [
            [CIRCLE, CIRCLE, CIRCLE, CIRCLE],
            [CIRCLE, CIRCLE, CIRCLE, CIRCLE],
            [CIRCLE, CIRCLE, CIRCLE, CIRCLE],
            [CIRCLE, CIRCLE, CIRCLE, CIRCLE],
        ],
        "target": [
            [CIRCLE, TRIANGLE, TRIANGLE, TRIANGLE],
            [TRIANGLE, TRIANGLE, DIAMOND, TRIANGLE],
            [CIRCLE, DIAMOND, TRIANGLE, TRIANGLE],
            [CIRCLE, CIRCLE, TRIANGLE, CIRCLE],
        ],
        "min_moves": 9,
    },
    {
        "grid_n": 4,
        "rule": RULE_CROSS_COND,
        "start": [
            [CIRCLE, CIRCLE, CIRCLE, CIRCLE],
            [CIRCLE, CIRCLE, CIRCLE, CIRCLE],
            [CIRCLE, CIRCLE, CIRCLE, CIRCLE],
            [CIRCLE, CIRCLE, CIRCLE, CIRCLE],
        ],
        "target": [
            [CIRCLE, TRIANGLE, TRIANGLE, TRIANGLE],
            [TRIANGLE, TRIANGLE, DIAMOND, TRIANGLE],
            [TRIANGLE, DIAMOND, DIAMOND, TRIANGLE],
            [TRIANGLE, TRIANGLE, TRIANGLE, CIRCLE],
        ],
        "min_moves": 12,
    },
]


NUM_LEVELS = len(LEVEL_CONFIGS)


def _px(frame, x, y, color):
    if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
        frame[y, x] = color


def _rect(frame, x, y, w, h, color):
    for dy in range(h):
        for dx in range(w):
            _px(frame, x + dx, y + dy, color)


def _hline(frame, x, y, length, color):
    for dx in range(length):
        _px(frame, x + dx, y, color)


def _vline(frame, x, y, length, color):
    for dy in range(length):
        _px(frame, x, y + dy, color)


def _draw_circle(frame, cx, cy, size, color):
    r = size / 2.0
    for dy in range(size):
        for dx in range(size):
            ox = dx - r + 0.5
            oy = dy - r + 0.5
            if ox * ox + oy * oy <= r * r:
                _px(frame, cx + dx, cy + dy, color)


def _draw_triangle(frame, cx, cy, size, color):
    for dy in range(size):
        width = dy + 1
        start = (size - width) // 2
        for dx in range(width):
            _px(frame, cx + start + dx, cy + dy, color)


def _draw_diamond(frame, cx, cy, size, color):
    mid = (size - 1) / 2.0
    for dy in range(size):
        dist_from_mid = abs(dy - mid)
        width = max(1, size - int(2 * dist_from_mid))
        start = (size - width) // 2
        for dx in range(width):
            _px(frame, cx + start + dx, cy + dy, color)


def _draw_square(frame, cx, cy, size, color):
    m = max(1, size // 6)
    _rect(frame, cx + m, cy + m, size - 2 * m, size - 2 * m, color)


SHAPE_DRAWERS = {
    CIRCLE: _draw_circle,
    TRIANGLE: _draw_triangle,
    DIAMOND: _draw_diamond,
    SQUARE: _draw_square,
}


def _draw_shape(frame, x, y, size, shape_id):
    color = SHAPE_COLORS[shape_id]
    SHAPE_DRAWERS[shape_id](frame, x, y, size, color)


def _layout(grid_n):
    if grid_n <= 3:
        cell = 10
        gap = 2
    elif grid_n == 4:
        cell = 8
        gap = 1
    else:
        cell = 5
        gap = 1

    main_total = grid_n * cell + (grid_n - 1) * gap
    main_ox = GRID_SIZE - main_total - 2

    mini_cell = 5 if grid_n <= 3 else 4
    mini_gap = 1
    mini_total = grid_n * mini_cell + (grid_n - 1) * mini_gap
    mini_ox = 2
    mini_oy = 5

    mini_end_y = mini_oy + mini_total
    lower_top = mini_end_y + 2
    lower_bot = GRID_SIZE - 2
    main_oy = lower_top + (lower_bot - lower_top - main_total) // 2

    return LayoutInfo(
        cell, gap, main_ox, main_oy, mini_cell, mini_gap, mini_ox, mini_oy
    )


class PuzzleHud(RenderableUserDisplay):
    def __init__(self):
        super().__init__()
        self.game = None

    def _draw_board(self, frame, grid, grid_n, ox, oy, cell, gap, cursor_rc=None):
        for r in range(grid_n):
            for c in range(grid_n):
                x = ox + c * (cell + gap)
                y = oy + r * (cell + gap)
                _rect(frame, x, y, cell, cell, CELL_BG)
                if cell <= 6:
                    _draw_shape(frame, x, y, cell, grid[r][c])
                else:
                    pad = max(1, cell // 5)
                    _draw_shape(frame, x + pad, y + pad, cell - 2 * pad, grid[r][c])

                if cursor_rc is not None and (r, c) == cursor_rc:
                    _hline(frame, x - 1, y - 1, cell + 2, YELLOW)
                    _hline(frame, x - 1, y + cell, cell + 2, YELLOW)
                    _vline(frame, x - 1, y - 1, cell + 2, YELLOW)
                    _vline(frame, x + cell, y - 1, cell + 2, YELLOW)
                    for cx2, cy2 in [
                        (x - 1, y - 1),
                        (x + cell, y - 1),
                        (x - 1, y + cell),
                        (x + cell, y + cell),
                    ]:
                        _px(frame, cx2, cy2, ORANGE)

    def _draw_lives(self, frame, lives, max_lives=MAX_LIVES):
        for i in range(max_lives):
            x = 2 + i * 4
            color = RED if i < lives else DARK_GREY
            _rect(frame, x, 1, 2, 2, color)

    def _draw_move_bar(self, frame, remaining, total):
        bar_x = 18
        bar_w = 44
        bar_h = 2
        _rect(frame, bar_x, 1, bar_w, bar_h, DARK_GREY)
        if total > 0:
            filled = max(0, bar_w * remaining // total)
            if filled > 0:
                if remaining > total * 0.5:
                    fill_color = GREEN
                elif remaining > total * 0.25:
                    fill_color = YELLOW
                else:
                    fill_color = RED
                _rect(frame, bar_x, 1, filled, bar_h, fill_color)

    def _draw_border(self, frame, color):
        for d in range(GRID_SIZE):
            _px(frame, d, 0, color)
            _px(frame, d, GRID_SIZE - 1, color)
        for d in range(GRID_SIZE):
            _px(frame, 0, d, color)
            _px(frame, GRID_SIZE - 1, d, color)

    def render_interface(self, frame):
        game = self.game
        if game is None:
            return frame

        height, width = frame.shape[:2]

        for y in range(height):
            for x in range(width):
                frame[y, x] = BLACK

        cfg = LEVEL_CONFIGS[game.level_idx]
        grid_n = cfg["grid_n"]
        layout = _layout(grid_n)

        mini_total = grid_n * layout.mini_cell + (grid_n - 1) * layout.mini_gap

        _rect(
            frame,
            layout.mini_ox - 1,
            layout.mini_oy - 1,
            mini_total + 2,
            mini_total + 2,
            GREEN,
        )
        self._draw_board(
            frame,
            cfg["target"],
            grid_n,
            layout.mini_ox,
            layout.mini_oy,
            layout.mini_cell,
            layout.mini_gap,
        )

        main_total = grid_n * layout.cell + (grid_n - 1) * layout.gap
        _rect(
            frame,
            layout.main_ox - 2,
            layout.main_oy - 2,
            main_total + 4,
            main_total + 4,
            GREY,
        )
        cursor = (
            (game.cursor_r, game.cursor_c)
            if not game.won and not game.game_over
            else None
        )
        self._draw_board(
            frame,
            game.grid,
            grid_n,
            layout.main_ox,
            layout.main_oy,
            layout.cell,
            layout.gap,
            cursor_rc=cursor,
        )

        self._draw_lives(frame, game.lives)
        self._draw_move_bar(frame, game.max_moves - game.move_count, game.max_moves)

        if game.won:
            self._draw_border(frame, GREEN)

        return frame


_marker = Sprite(
    pixels=[[BLACK]],
    name="marker",
    visible=False,
    collidable=False,
    tags=["marker"],
    layer=0,
)

levels = [
    Level(
        sprites=[_marker.clone().set_position(0, 0)],
        grid_size=(64, 64),
        data={"idx": i},
        name=f"Level {i + 1}",
    )
    for i in range(NUM_LEVELS)
]


class Eg06(ARCBaseGame):
    def __init__(self, seed: int = 0):
        self._seed = seed
        self._rng = random.Random(seed)
        self.hud = PuzzleHud()
        self.hud.game = self

        self.level_idx = 0
        self.grid = []
        self.init_grid = []
        self.cursor_r = 0
        self.cursor_c = 0
        self.move_count = 0
        self.max_moves = 0
        self.won = False
        self.game_over = False
        self.lives = MAX_LIVES
        self.died_at_level = -1
        self.reset_press_count = 0
        self.play_since_reset = False
        self.turn = 0

        self._history = []
        self._game_over = False

        super().__init__(
            "eg06",
            levels,
            Camera(0, 0, 64, 64, BLACK, BLACK, [self.hud]),
            available_actions=[0, 1, 2, 3, 4, 5, 7],
        )

    def _setup_level(self, idx, reset_lives=True):
        if idx < 0 or idx >= NUM_LEVELS:
            idx = 0
        cfg = LEVEL_CONFIGS[idx]
        self.level_idx = idx
        self.init_grid = deepcopy(cfg["start"])
        self.grid = deepcopy(cfg["start"])
        self.cursor_r = self._rng.randint(0, cfg["grid_n"] - 1)
        self.cursor_c = self._rng.randint(0, cfg["grid_n"] - 1)
        self.move_count = 0
        self.max_moves = cfg["min_moves"] * 6
        self.won = False
        if reset_lives:
            self.lives = MAX_LIVES
        self.game_over = False
        self.turn = 0
        self.play_since_reset = False

    def on_set_level(self, level):
        idx = self.current_level.get_data("idx")
        if idx is None:
            idx = 0
        if self.died_at_level >= 0:
            idx = self.died_at_level
            self.died_at_level = -1
        self._setup_level(idx)

        self._history = []
        self._game_over = False

    def _activate(self, r, c):
        cfg = LEVEL_CONFIGS[self.level_idx]
        self.grid = apply_click(self.grid, r, c, cfg["rule"], cfg["grid_n"])
        self.move_count += 1

        if self.grid == cfg["target"]:
            self.won = True

    def _save_state(self):
        self._history.append(
            {
                "grid": deepcopy(self.grid),
                "init_grid": deepcopy(self.init_grid),
                "cursor_r": self.cursor_r,
                "cursor_c": self.cursor_c,
                "move_count": self.move_count,
                "won": self.won,
                "game_over": self.game_over,
                "play_since_reset": self.play_since_reset,
            }
        )

    def _undo(self):
        if not self._history:
            return
        snap = self._history.pop()
        self.grid = snap["grid"]
        self.init_grid = snap["init_grid"]
        self.cursor_r = snap["cursor_r"]
        self.cursor_c = snap["cursor_c"]
        self.won = snap["won"]
        self.game_over = snap["game_over"]
        self.play_since_reset = snap["play_since_reset"]

    def _reset_current_level(self):
        self._setup_level(self.level_idx, reset_lives=False)
        self._history = []

    def full_reset(self):
        self._game_over = False
        self.game_over = False
        self.lives = MAX_LIVES
        self.died_at_level = -1
        self.reset_press_count = 0
        self.play_since_reset = False
        self._rng = random.Random(self._seed)
        self._current_level_index = 0
        self.on_set_level(levels[0])

    def _handle_death(self):
        self.lives -= 1
        if self.lives <= 0:
            self._game_over = True
            self.game_over = True
            self.lose()
            self.lives = MAX_LIVES
            return True
        else:
            return False

    def _check_moves_exhausted(self):
        if self.move_count >= self.max_moves and not self.won:
            game_over = self._handle_death()
            if game_over:
                return
            else:
                self._reset_current_level()

    def _dispatch_action(self, act):
        cfg = LEVEL_CONFIGS[self.level_idx]
        grid_n = cfg["grid_n"]

        if act == 1:
            self.cursor_r = (self.cursor_r - 1) % grid_n
            self.move_count += 1
        elif act == 2:
            self.cursor_r = (self.cursor_r + 1) % grid_n
            self.move_count += 1
        elif act == 3:
            self.cursor_c = (self.cursor_c - 1) % grid_n
            self.move_count += 1
        elif act == 4:
            self.cursor_c = (self.cursor_c + 1) % grid_n
            self.move_count += 1
        elif act == 5:
            self._activate(self.cursor_r, self.cursor_c)

    def step(self) -> None:
        action_id = self.action.id.value

        if action_id == 0:
            if not self.play_since_reset:
                self.reset_press_count += 1
            else:
                self.reset_press_count = 1

            if self.reset_press_count >= 2:
                self._setup_level(0)
                self.reset_press_count = 0
                self.died_at_level = -1
            elif self.died_at_level >= 0:
                self._setup_level(self.died_at_level)
            else:
                self._setup_level(self.level_idx)

            self.play_since_reset = False
            self._history = []
            self.complete_action()
            return

        if action_id == 7:
            self._undo()
            self.move_count += 1
            self._check_moves_exhausted()
            self.complete_action()
            return

        if self.game_over:
            self.died_at_level = self.level_idx
            self.complete_action()
            self.lose()
            return
        if self.won:
            self.complete_action()
            self.next_level()
            return

        self.turn += 1
        self.play_since_reset = True
        self.reset_press_count = 0
        self.died_at_level = -1

        self._save_state()

        self._dispatch_action(action_id)
        self._check_moves_exhausted()

        if self.won:
            self.complete_action()
            self.next_level()
            return

        self.complete_action()


_SHAPE_CHAR = {
    CIRCLE: "O",
    TRIANGLE: "T",
    DIAMOND: "D",
    SQUARE: "S",
}

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

    def __init__(self, seed: int = 0) -> None:
        self._engine: Any = Eg06(seed=seed)
        self._done = False
        self._game_won = False
        self._last_action_was_reset = False
        self._total_turns = 0
        self._total_levels = len(levels)

    @staticmethod
    def _frame_to_png(frame: np.ndarray) -> bytes:
        h, w = frame.shape[:2]
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for idx in range(len(ARC_PALETTE)):
            mask = frame == idx
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            rgb[mask] = ARC_PALETTE[idx]

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

    def _render_frame(self) -> np.ndarray:
        e = self._engine
        return e.camera.render(e.current_level.get_sprites())

    def _build_text_observation(self) -> str:
        e = self._engine
        cfg = LEVEL_CONFIGS[e.level_idx]
        grid_n = cfg["grid_n"]

        grid_lines = []
        for r in range(grid_n):
            row_chars = []
            for c in range(grid_n):
                shape_id = e.grid[r][c]
                ch = _SHAPE_CHAR.get(shape_id, "?")
                if r == e.cursor_r and c == e.cursor_c:
                    row_chars.append("[" + ch + "]")
                else:
                    row_chars.append(" " + ch + " ")
            grid_lines.append("".join(row_chars))

        target_lines = []
        for r in range(grid_n):
            row_chars = []
            for c in range(grid_n):
                shape_id = cfg["target"][r][c]
                ch = _SHAPE_CHAR.get(shape_id, "?")
                row_chars.append(" " + ch + " ")
            target_lines.append("".join(row_chars))

        header = (
            f"Level {e.level_idx + 1}/{NUM_LEVELS}"
            f" | Moves: {e.move_count}/{e.max_moves}"
            f" | Lives: {e.lives}"
        )
        rules = "Move cursor with arrows. Select applies shape rule at cursor. Match grid to target."
        return (
            header
            + "\n"
            + rules
            + "\nGrid:\n"
            + "\n".join(grid_lines)
            + "\nTarget:\n"
            + "\n".join(target_lines)
        )

    def _build_game_state(self, done: bool = False) -> GameState:
        e = self._engine
        frame = self._render_frame()
        image_bytes = self._frame_to_png(frame)

        valid_actions = list(self._VALID_ACTIONS) if not done else None

        return GameState(
            text_observation=self._build_text_observation(),
            image_observation=image_bytes,
            valid_actions=valid_actions,
            turn=self._total_turns,
            metadata={
                "total_levels": len(e._levels),
                "level_index": e._current_level_index,
                "levels_completed": getattr(e, "_score", 0),
                "game_over": getattr(getattr(e, "_state", None), "name", "")
                == "GAME_OVER",
                "done": done,
                "info": {},
                "level": e.level_idx + 1,
                "cursor_position": [e.cursor_r, e.cursor_c],
                "grid": deepcopy(e.grid),
                "target_grid": deepcopy(LEVEL_CONFIGS[e.level_idx]["target"]),
                "move_count": e.move_count,
                "max_moves": e.max_moves,
                "lives": e.lives,
                "won": e.won,
                "grid_size": [
                    LEVEL_CONFIGS[e.level_idx]["grid_n"],
                    LEVEL_CONFIGS[e.level_idx]["grid_n"],
                ],
            },
        )

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

        return self._build_game_state()

    def is_done(self) -> bool:
        return self._done

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

        game_action = self._ACTION_MAP[action]
        info: Dict[str, Any] = {"action": action}

        level_before = e.level_index

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
        if e.level_index != level_before:
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

    env = ArcGameEnv(seed=42, render_mode="rgb_array")
    check_env(env.unwrapped, skip_render_check=True)

    obs, info = env.reset()

    mask = env.action_mask()
    valid_indices = np.where(mask == 1)[0]
    if len(valid_indices) > 0:
        obs, reward, term, trunc, info = env.step(valid_indices[0])

    env.close()
