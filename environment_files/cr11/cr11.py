import random
import struct
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from arcengine import (
    ActionInput,
    ARCBaseGame,
    Camera,
    GameAction,
    Level,
    Sprite,
)
from arcengine.interfaces import RenderableUserDisplay


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

C_BLACK = 0
C_BLUE = 1
C_BROWN = 2
C_LIME = 3
C_WHITE = 4
C_GRAY = 5
C_MAGENTA = 6
C_ORANGE = 7
C_RED = 8
C_CYAN = 9
C_LTBLUE = 10
C_YELLOW = 11
C_PINK = 12
C_OLIVE = 13
C_GREEN = 14
C_PURPLE = 15

BACKGROUND_COLOR = C_BLACK
LETTERBOX = C_GRAY
SELECTOR_BORDER = C_WHITE
FRAME_SIZE = 64


ROW_ACTIVATOR = 7
COL_ACTIVATOR = 8
TARGET = 3
LOCKED = 9
ROW_ONLY_TARGET = 12
COL_ONLY_TARGET = 13

HUD_BOTTOM_RESERVE = 1


def _compute_tile(rows: int, cols: int) -> int:
    return min(64 // cols, (64 - HUD_BOTTOM_RESERVE) // rows)


def _get_board_offset(rows: int, cols: int, tile: int) -> tuple[int, int]:
    pixel_w = cols * tile
    pixel_h = rows * tile
    usable_h = 64 - HUD_BOTTOM_RESERVE
    ox = (64 - pixel_w) // 2
    oy = (usable_h - pixel_h) // 2
    return ox, oy


ACTION_ALIASES = {
    "UP": "ACTION1",
    "DOWN": "ACTION2",
    "LEFT": "ACTION3",
    "RIGHT": "ACTION4",
    "ACTIVATE": "ACTION5",
    "SELECT": "ACTION5",
}


def _make_cell_sprite(color: int, tile: int) -> Sprite:
    pixels = [[color] * tile for _ in range(tile)]
    return Sprite(pixels=pixels, name=f"cell_{color}", visible=True, collidable=True)


LEVEL_BOARDS: list[dict] = [
    {
        "grid": [
            [0, 0, 0, 0, 0, 0, 0, 8],
            [0, 0, 0, 0, 9, 0, 0, 0],
            [0, 0, 0, 0, 3, 0, 0, 0],
            [0, 0, 0, 0, 3, 0, 0, 8],
            [0, 0, 0, 0, 3, 0, 0, 0],
            [0, 0, 7, 0, 7, 8, 0, 0],
            [7, 0, 0, 0, 0, 3, 0, 0],
            [9, 0, 0, 0, 0, 0, 0, 0],
        ],
        "locked": [(0, 7), (4, 1)],
    },
    {
        "grid": [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 7, 0, 0, 0, 8, 8, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 8, 3, 0, 0, 7, 0, 0, 0],
            [0, 0, 0, 0, 0, 9, 0, 0, 0, 3, 0, 0, 0, 0, 0, 9],
            [0, 0, 0, 0, 9, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 9, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        "locked": [(4, 6), (5, 5), (12, 10), (15, 5)],
    },
    {
        "grid": [
            [3, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0],
            [12, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0],
            [3, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0],
            [9, 0, 7, 0, 9, 9, 0, 0, 8, 0, 0, 9, 9, 0, 0, 0],
            [9, 0, 0, 0, 9, 0, 0, 9, 12, 9, 0, 9, 0, 0, 7, 0],
            [9, 7, 0, 0, 9, 9, 0, 0, 3, 0, 0, 9, 9, 0, 0, 0],
            [9, 0, 0, 0, 9, 0, 9, 0, 12, 0, 9, 0, 9, 7, 0, 0],
            [9, 0, 0, 0, 9, 9, 7, 0, 0, 3, 0, 9, 9, 0, 0, 0],
            [9, 0, 0, 0, 9, 0, 0, 9, 12, 9, 0, 9, 0, 0, 0, 7],
            [0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0],
        ],
        "locked": [
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 5),
            (0, 6),
            (0, 7),
            (0, 8),
            (0, 9),
            (1, 0),
            (1, 1),
            (1, 2),
            (1, 3),
            (4, 4),
            (4, 5),
            (4, 6),
            (4, 7),
            (4, 8),
            (4, 9),
            (5, 4),
            (5, 6),
            (5, 8),
            (6, 7),
            (7, 5),
            (7, 9),
            (8, 2),
            (8, 11),
            (9, 5),
            (9, 9),
            (10, 7),
            (11, 4),
            (11, 5),
            (11, 6),
            (11, 8),
            (11, 9),
            (12, 4),
            (12, 6),
            (12, 7),
            (12, 8),
        ],
    },
    {
        "grid": [
            [3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [12, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0],
            [13, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0],
            [9, 9, 0, 7, 0, 0, 9, 0, 0, 0, 8, 0, 0, 0, 0, 0],
            [9, 9, 0, 0, 0, 0, 9, 0, 0, 0, 12, 0, 0, 0, 7, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0],
            [9, 9, 0, 7, 0, 0, 9, 0, 0, 3, 0, 0, 0, 0, 0, 0],
            [9, 9, 0, 0, 0, 0, 9, 0, 0, 12, 0, 0, 0, 0, 7, 0],
            [9, 9, 0, 0, 0, 0, 9, 0, 0, 13, 0, 0, 0, 0, 0, 0],
            [9, 9, 0, 0, 7, 0, 9, 0, 0, 12, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0],
            [9, 9, 0, 0, 0, 0, 9, 0, 0, 3, 0, 0, 0, 0, 7, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0],
        ],
        "locked": [
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 6),
            (0, 7),
            (0, 8),
            (0, 9),
            (0, 11),
            (1, 0),
            (1, 1),
            (1, 2),
            (1, 3),
            (1, 4),
            (1, 6),
            (1, 7),
            (1, 8),
            (1, 9),
            (1, 11),
            (6, 3),
            (6, 4),
            (6, 6),
            (6, 7),
            (6, 8),
            (6, 9),
            (6, 11),
            (9, 1),
            (9, 14),
        ],
    },
]


BASELINE_ACTIONS = [10, 20, 20, 35]


class SelectorBorderOverlay(RenderableUserDisplay):
    def __init__(self, get_board_state, get_selector, border_color: int) -> None:
        self._get_board_state = get_board_state
        self._get_selector = get_selector
        self._border_color = border_color

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        tile, ox, oy = self._get_board_state()
        gx, gy = self._get_selector()
        sx = ox + gx * tile
        sy = oy + gy * tile
        ex = sx + tile - 1
        ey = sy + tile - 1
        fh, fw = frame.shape[:2]
        sx = max(0, sx)
        sy = max(0, sy)
        ex = min(fw - 1, ex)
        ey = min(fh - 1, ey)
        if sx <= ex and sy <= ey:
            frame[sy, sx : ex + 1] = self._border_color
            frame[ey, sx : ex + 1] = self._border_color
            frame[sy : ey + 1, sx] = self._border_color
            frame[sy : ey + 1, ex] = self._border_color
        return frame


class Cr11Hud(RenderableUserDisplay):

    def __init__(self, game: "Cr11") -> None:
        self.game = game

    def _get_grid_bounds(self) -> tuple[int, int, int, int] | None:
        g = self.game
        ox, oy = g._ox, g._oy
        tile = g._tile
        w, h = g._board_w, g._board_h
        grid_x_start = ox
        grid_x_end = ox + w * tile
        grid_y_start = oy
        grid_y_end = oy + h * tile
        return grid_x_start, grid_x_end, grid_y_start, grid_y_end

    @staticmethod
    def _mask_non_playable_area(
        frame: np.ndarray,
        grid_x_start: int,
        grid_x_end: int,
        grid_y_start: int,
        grid_y_end: int,
    ) -> None:
        frame[0:grid_y_start, :] = LETTERBOX
        frame[grid_y_end:FRAME_SIZE, :] = LETTERBOX
        frame[:, 0:grid_x_start] = LETTERBOX
        frame[:, grid_x_end:FRAME_SIZE] = LETTERBOX

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        bounds = self._get_grid_bounds()
        if bounds is not None:
            grid_x_start, grid_x_end, grid_y_start, grid_y_end = bounds
            self._mask_non_playable_area(
                frame, grid_x_start, grid_x_end, grid_y_start, grid_y_end
            )

        moves_left = self.game._get_max_steps() - self.game._moves
        limit = self.game._get_max_steps()
        lives = self.game._lives

        if bounds is not None:
            gxs, gxe, _, gye = bounds
            hud_y = gye
            hud_x_start = gxs
            hud_width = gxe - gxs
        else:
            hud_y = 63
            hud_x_start = 0
            hud_width = 64

        if 0 <= hud_y < 64:
            lives_width = lives * 2 - 1 if lives > 0 else 0
            bar_start = hud_x_start + lives_width + (1 if lives > 0 else 0)
            bar_width = max(0, hud_width - lives_width - (1 if lives > 0 else 0))

            for i in range(lives):
                px = hud_x_start + i * 2
                if 0 <= px < 64:
                    frame[hud_y, px] = C_RED

            filled = int(round(moves_left / max(limit, 1) * bar_width))
            filled = max(0, min(filled, bar_width))
            ratio = moves_left / max(limit, 1)
            fill_color = C_GREEN if ratio > 0.5 else (C_YELLOW if moves_left > 2 else C_RED)
            for i in range(bar_width):
                px = bar_start + i
                if 0 <= px < 64:
                    frame[hud_y, px] = fill_color if i < filled else C_BLACK

        return frame


levels = [
    Level(sprites=[], grid_size=(64, 64)),
    Level(sprites=[], grid_size=(64, 64)),
    Level(sprites=[], grid_size=(64, 64)),
    Level(sprites=[], grid_size=(64, 64)),
]


class Cr11(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._grid: list[list[int]] = []
        self._locked_positions: set[tuple[int, int]] = set()
        self._cell_sprites: dict[tuple[int, int], Sprite] = {}
        self._selector_pos: tuple[int, int] = (0, 0)
        self._tile = 7
        self._ox = 4
        self._oy = 3
        self._board_w = 8
        self._board_h = 8
        self._selector_overlay = SelectorBorderOverlay(
            get_board_state=lambda: (
                self._tile,
                self._ox,
                self._oy,
            ),
            get_selector=lambda: self._selector_pos,
            border_color=SELECTOR_BORDER,
        )

        self._lives = 3
        self._moves = 0
        self._preserve_lives = False
        self._turn_limit_reached = False
        self._turn_limit_reached_latch = False
        self._history: list[dict] = []
        self._rng = random.Random(seed)
        self._ui_overlay = Cr11Hud(self)

        camera = Camera(
            background=BACKGROUND_COLOR,
            letter_box=LETTERBOX,
            width=64,
            height=64,
            interfaces=[self._selector_overlay, self._ui_overlay],
        )

        super().__init__(
            game_id="cr11",
            levels=levels,
            camera=camera,
            seed=seed,
            available_actions=[0, 1, 2, 3, 4, 5, 7],
        )

    def _make_sprite(self, color: int, x: int, y: int) -> Sprite:
        return _make_cell_sprite(color, self._tile).set_position(x, y)

    def _rotate_row_right(self, row: int) -> set[tuple[int, int]]:
        w = len(self._grid[0])
        changed: set[tuple[int, int]] = set()

        locked_cols = {
            c
            for c in range(w)
            if (c, row) in self._locked_positions
            or self._grid[row][c] == COL_ONLY_TARGET
        }

        movable_vals: list[int] = []
        movable_cols: list[int] = []
        for c in range(w):
            if c not in locked_cols:
                movable_vals.append(self._grid[row][c])
                movable_cols.append(c)

        if len(movable_vals) < 2:
            return changed

        rotated = [movable_vals[-1]] + movable_vals[:-1]

        for idx, c in enumerate(movable_cols):
            if self._grid[row][c] != rotated[idx]:
                self._grid[row][c] = rotated[idx]
                changed.add((c, row))

        return changed

    def _rotate_col_down(self, col: int) -> set[tuple[int, int]]:
        h = len(self._grid)
        changed: set[tuple[int, int]] = set()

        locked_rows = {
            r
            for r in range(h)
            if (col, r) in self._locked_positions
            or self._grid[r][col] == ROW_ONLY_TARGET
        }

        movable_vals: list[int] = []
        movable_rows: list[int] = []
        for r in range(h):
            if r not in locked_rows:
                movable_vals.append(self._grid[r][col])
                movable_rows.append(r)

        if len(movable_vals) < 2:
            return changed

        rotated = [movable_vals[-1]] + movable_vals[:-1]

        for idx, r in enumerate(movable_rows):
            if self._grid[r][col] != rotated[idx]:
                self._grid[r][col] = rotated[idx]
                changed.add((col, r))

        return changed

    def _get_max_steps(self) -> int:
        level_idx = self.level_index
        if not (0 <= level_idx < len(BASELINE_ACTIONS)):
            return 0
        return BASELINE_ACTIONS[level_idx] * 12

    def _is_solved(self) -> bool:
        target_colors = (TARGET, ROW_ONLY_TARGET, COL_ONLY_TARGET)
        positions: list[tuple[int, int]] = []
        for r in range(len(self._grid)):
            for c in range(len(self._grid[0])):
                if (
                    self._grid[r][c] in target_colors
                    and (c, r) not in self._locked_positions
                ):
                    positions.append((c, r))

        if not positions:
            return False

        cols = set(x for x, _ in positions)
        if len(cols) != 1:
            return False

        rows = sorted(y for _, y in positions)
        for i in range(len(rows) - 1):
            if rows[i + 1] - rows[i] != 1:
                return False

        return True

    def _generate_board(self) -> None:
        board = LEVEL_BOARDS[self.level_index]
        h = len(board["grid"])
        w = len(board["grid"][0])

        self._grid = [row[:] for row in board["grid"]]
        self._locked_positions = set(board["locked"])
        self._cell_sprites.clear()

        for r in range(h):
            for c in range(w):
                color = self._grid[r][c]
                if color != C_BLACK:
                    sp = self._make_sprite(
                        color,
                        c * self._tile + self._ox,
                        r * self._tile + self._oy,
                    )
                    self.current_level.add_sprite(sp)
                    self._cell_sprites[(c, r)] = sp

    def _refresh_sprites(self, positions: set[tuple[int, int]]) -> None:
        for pos in positions:
            color = self._grid[pos[1]][pos[0]]

            old = self._cell_sprites.get(pos)
            if old is not None:
                try:
                    self.current_level.remove_sprite(old)
                except ValueError:
                    pass

            if color != C_BLACK:
                new_sp = self._make_sprite(
                    color,
                    pos[0] * self._tile + self._ox,
                    pos[1] * self._tile + self._oy,
                )
                self.current_level.add_sprite(new_sp)
                self._cell_sprites[pos] = new_sp
            else:
                self._cell_sprites.pop(pos, None)

    def _save_state(self) -> None:
        snapshot = {
            "grid": [row[:] for row in self._grid],
            "selector_pos": self._selector_pos,
        }
        self._history.append(snapshot)

    def _undo(self) -> None:
        if not self._history:
            return
        snapshot = self._history.pop()
        self._grid = snapshot["grid"]
        self._selector_pos = snapshot["selector_pos"]
        all_positions: set[tuple[int, int]] = set()
        for r in range(len(self._grid)):
            for c in range(len(self._grid[0])):
                all_positions.add((c, r))
        self._refresh_sprites(all_positions)

    def _move_selector(self, dx: int, dy: int) -> None:
        w, h = self._board_w, self._board_h
        x, y = self._selector_pos
        nx = (x + dx) % w
        ny = (y + dy) % h
        self._selector_pos = (nx, ny)

    def _try_activate_at(self, gx: int, gy: int) -> None:
        rows = len(self._grid)
        cols = len(self._grid[0]) if rows else 0
        if not (0 <= gy < rows and 0 <= gx < cols):
            return

        changed: set[tuple[int, int]] = set()
        if (gx, gy) not in self._locked_positions:
            cell = self._grid[gy][gx]
            if cell == ROW_ACTIVATOR:
                changed = self._rotate_row_right(gy)
            elif cell == COL_ACTIVATOR:
                changed = self._rotate_col_down(gx)

        if not changed:
            return

        self._refresh_sprites(changed)

        if self._is_solved():
            self.next_level()

    def _activate_selected(self) -> None:
        sx, sy = self._selector_pos
        self._try_activate_at(sx, sy)

    def on_set_level(self, level: Level) -> None:
        board = LEVEL_BOARDS[self.level_index]
        h = len(board["grid"])
        w = len(board["grid"][0])
        self._tile = _compute_tile(h, w)
        self._ox, self._oy = _get_board_offset(h, w, self._tile)
        self._board_w = w
        self._board_h = h

        self._generate_board()
        self._history = []

        row_activators: list[int] = []
        col_activators: list[int] = []
        for r in range(h):
            for c in range(w):
                cell = self._grid[r][c]
                if cell == ROW_ACTIVATOR and (c, r) not in self._locked_positions:
                    if r not in row_activators:
                        row_activators.append(r)
                elif cell == COL_ACTIVATOR and (c, r) not in self._locked_positions:
                    if c not in col_activators:
                        col_activators.append(c)

        num_shuffles = BASELINE_ACTIONS[self.level_index] if self.level_index < len(BASELINE_ACTIONS) else 4
        all_changed: set[tuple[int, int]] = set()
        for _ in range(num_shuffles):
            if self._rng.random() < 0.5 and row_activators:
                row = self._rng.choice(row_activators)
                all_changed |= self._rotate_row_right(row)
            elif col_activators:
                col = self._rng.choice(col_activators)
                all_changed |= self._rotate_col_down(col)
            elif row_activators:
                row = self._rng.choice(row_activators)
                all_changed |= self._rotate_row_right(row)

        if all_changed:
            self._refresh_sprites(all_changed)

        self._selector_pos = (w // 2, h // 2)

        if self._preserve_lives:
            self._preserve_lives = False
        else:
            self._lives = 3
        self._moves = 0
        self._turn_limit_reached = False

    def _build_text_observation(self) -> str:
        w = len(self._grid[0]) if self._grid else 0
        h = len(self._grid)
        sx, sy = self._selector_pos
        max_steps = self._get_max_steps()
        return (
            f"Level {self.level_index + 1}/{len(self._levels)} on a {w}x{h} grid. "
            f"Selector at ({sx}, {sy}). Lives: {self._lives}. "
            f"Moves: {self._moves}/{max_steps}. "
            "Goal: align all target cells into a single consecutive vertical column."
        )

    def _build_api_state(
        self, image_observation: list[list[int]] | None = None
    ) -> dict[str, object]:
        available_actions = self.get_actions()
        state_name = getattr(self._state, "name", "") if self._state else ""
        return {
            "text_observation": self._build_text_observation(),
            "image_observation": image_observation,
            "available_actions": available_actions,
            "turn_number": self._action_count,
            "metadata": {
                "game_id": self.game_id,
                "level_index": self.level_index,
                "levels_completed": self._score,
                "won": state_name == "WIN",
                "lost": state_name == "GAME_OVER",
                "turn_limit_reached": self._turn_limit_reached
                or self._turn_limit_reached_latch,
                "no_valid_actions_remaining": len(available_actions) == 0,
                "lives_remaining": self._lives,
                "moves_used": self._moves,
                "move_limit": self._get_max_steps(),
                "selector_position": self._selector_pos,
            },
        }

    def _coerce_action_input(
        self, action: ActionInput | dict[str, object] | str | int
    ) -> ActionInput:
        if isinstance(action, ActionInput):
            return action

        if isinstance(action, int):
            return ActionInput(id=GameAction.from_id(action))

        if isinstance(action, str):
            normalized = action.strip().upper()
            return ActionInput(
                id=GameAction.from_name(ACTION_ALIASES.get(normalized, normalized))
            )

        action_name = action.get(
            "action", action.get("type", action.get("name", action.get("id")))
        )
        if action_name is None:
            raise ValueError("Action must include one of: action, type, name, or id")

        if isinstance(action_name, int):
            action_id = GameAction.from_id(action_name)
        else:
            normalized = str(action_name).strip().upper()
            action_id = GameAction.from_name(
                ACTION_ALIASES.get(normalized, normalized)
            )

        return ActionInput(id=action_id)

    def reset(self) -> dict[str, object]:
        self._turn_limit_reached_latch = False
        self.full_reset()
        frame = self.camera.render(self.current_level.get_sprites()).tolist()
        return self._build_api_state(image_observation=frame)

    def get_actions(self) -> list[dict[str, object]]:
        state_name = getattr(self._state, "name", "") if self._state else ""
        if state_name in ("GAME_OVER", "WIN"):
            return [{"action": "RESET"}]

        return [
            {"action": "ACTION1"},
            {"action": "ACTION2"},
            {"action": "ACTION3"},
            {"action": "ACTION4"},
            {"action": "ACTION5"},
        ]

    def _api_step(
        self, action: ActionInput | dict[str, object] | str | int
    ) -> tuple[dict[str, object], float, bool]:
        self._turn_limit_reached = False
        self._turn_limit_reached_latch = False
        previous_score = self._score
        frame_data = self.perform_action(self._coerce_action_input(action))

        if frame_data.frame:
            image_observation = frame_data.frame[-1]
        else:
            image_observation = self.camera.render(
                self.current_level.get_sprites()
            ).tolist()

        next_state = self._build_api_state(image_observation=image_observation)

        state_name = frame_data.state.name if frame_data and frame_data.state else ""

        reward = 0.0
        if state_name == "GAME_OVER":
            reward = 0.0
        elif (
            frame_data.levels_completed > previous_score
            or state_name == "WIN"
        ):
            reward = 1.0 / len(self._levels)

        done = state_name in ("GAME_OVER", "WIN")
        return next_state, reward, done

    def step(self) -> None:
        if self.action.id == GameAction.RESET:
            self._history = []
            self.complete_action()
            return None

        self._turn_limit_reached = False

        if self.action.id == GameAction.ACTION7:
            self._undo()
            self._moves += 1
            max_steps = self._get_max_steps()
            state_name = getattr(self._state, "name", "") if self._state else ""
            if (
                not self._next_level
                and state_name != "WIN"
                and max_steps > 0
                and self._moves >= max_steps
            ):
                self._turn_limit_reached = True
                self._turn_limit_reached_latch = True
                self._lives -= 1
                if self._lives <= 0:
                    self._moves = 0
                    self.lose()
                else:
                    self._preserve_lives = True
                    self.level_reset()
            self.complete_action()
            return None

        if self.action.id == GameAction.ACTION1:
            self._move_selector(0, -1)

        elif self.action.id == GameAction.ACTION2:
            self._move_selector(0, 1)

        elif self.action.id == GameAction.ACTION3:
            self._move_selector(-1, 0)

        elif self.action.id == GameAction.ACTION4:
            self._move_selector(1, 0)

        self._moves += 1

        if self.action.id == GameAction.ACTION5:
            self._save_state()
            self._activate_selected()

        max_steps = self._get_max_steps()
        state_name = getattr(self._state, "name", "") if self._state else ""
        if (
            not self._next_level
            and state_name != "WIN"
            and max_steps > 0
            and self._moves >= max_steps
        ):
            self._turn_limit_reached = True
            self._turn_limit_reached_latch = True
            self._lives -= 1
            if self._lives <= 0:
                self._moves = 0
                self.lose()
            else:
                self._preserve_lives = True
                self.level_reset()

        self.complete_action()


class PuzzleEnvironment:
    _ACTION_MAP: Dict[str, GameAction] = {
        "up": GameAction.ACTION1,
        "down": GameAction.ACTION2,
        "left": GameAction.ACTION3,
        "right": GameAction.ACTION4,
        "select": GameAction.ACTION5,
        "undo": GameAction.ACTION7,
        "reset": GameAction.RESET,
    }

    _VALID_ACTIONS = list(_ACTION_MAP.keys())

    def __init__(self, seed: int = 0) -> None:
        self._engine = Cr11(seed=seed)
        self._total_turns = 0
        self._done = False
        self._last_action_was_reset = False
        self._game_won = False

    def _build_text_obs(self) -> str:
        return self._engine._build_text_observation()

    def _build_image_bytes(self) -> Optional[bytes]:
        e = self._engine
        index_grid = e.camera.render(e.current_level.get_sprites())
        if index_grid is None:
            return None
        arr = np.array(index_grid, dtype=np.uint8)
        h, w = arr.shape[:2]

        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for idx in range(len(ARC_PALETTE)):
            mask = index_grid == idx
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            rgb[mask] = ARC_PALETTE[idx]

        def _pack_png(data: np.ndarray) -> bytes:
            ph, pw = data.shape[:2]
            raw = b""
            for row in data:
                raw += b"\x00" + row.tobytes()
            compressed = zlib.compress(raw)

            def _chunk(tag: bytes, body: bytes) -> bytes:
                return (
                    struct.pack(">I", len(body))
                    + tag
                    + body
                    + struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF)
                )

            ihdr = struct.pack(">IIBBBBB", pw, ph, 8, 2, 0, 0, 0)
            return (
                b"\x89PNG\r\n\x1a\n"
                + _chunk(b"IHDR", ihdr)
                + _chunk(b"IDAT", compressed)
                + _chunk(b"IEND", b"")
            )

        return _pack_png(rgb)

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
                "total_levels": len(self._engine._levels),
                "level_index": e.level_index,
                "levels_completed": getattr(e, "_score", 0),
                "game_over": getattr(getattr(e, "_state", None), "name", "")
                == "GAME_OVER",
                "done": done,
                "info": info or {},
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
        info = {"action": action}

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
                state=self._build_game_state(done=True, info=info),
                reward=level_reward,
                done=True,
                info=info,
            )

        if game_over:
            self._done = True
            info["reason"] = "game_over"
            return StepResult(
                state=self._build_game_state(done=True, info=info),
                reward=0.0,
                done=True,
                info=info,
            )

        reward = 0.0
        if e.level_index != level_before:
            reward = level_reward
            info["reason"] = "level_complete"

        return StepResult(
            state=self._build_game_state(done=False, info=info),
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
        "render_fps": 1,
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

        self._seed: int = seed
        self._env: Any = None

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
