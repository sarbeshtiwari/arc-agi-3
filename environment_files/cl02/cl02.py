import io
import random
import struct
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from arcengine import ActionInput, ARCBaseGame, Camera, GameAction, Level, Sprite


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


BG = 5
WHITE = 0
OFF_BLACK = 4
LIGHT_GREY = 2
RED = 8
BLUE = 9
YELLOW = 11
GREEN = 14
PURPLE = 15
PINK = 7
ORANGE = 12
LIGHT_BLUE = 10

GRID_SIZE = 64
MAX_LIVES = 3
TILE_PAL = [RED, GREEN, BLUE, YELLOW, PURPLE, PINK, ORANGE, LIGHT_BLUE]

_PX_LOCK = [
    [1, 0, 0, 0, 1],
    [0, 1, 0, 1, 0],
    [0, 0, 1, 0, 0],
    [0, 1, 0, 1, 0],
    [1, 0, 0, 0, 1],
]

_PX_L = [
    [0, 0, 0, 1, 0],
    [0, 0, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 1, 1, 0],
    [0, 0, 0, 1, 0],
]
_PX_R = [
    [0, 1, 0, 0, 0],
    [0, 1, 1, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 0, 0],
    [0, 1, 0, 0, 0],
]
_PX_U = [
    [0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 1, 1, 1, 0],
    [1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0],
]
_PX_D = [
    [0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1],
    [0, 1, 1, 1, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0],
]
_PX_CW = [
    [1, 1, 1, 1, 1],
    [0, 0, 0, 0, 1],
    [0, 1, 0, 0, 1],
    [1, 1, 1, 1, 1],
    [0, 1, 0, 0, 0],
]
_PX_CCW = [
    [0, 1, 0, 0, 0],
    [1, 1, 1, 1, 1],
    [0, 1, 0, 0, 1],
    [0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1],
]

LEVEL_CONFIGS = [
    {
        "n": 4,
        "initial": [[3, 2, 0, 3], [0, 3, 1, 0], [3, 2, 2, 1], [0, 2, 1, 1]],
        "link_map": {0: 0, 1: 1, 2: 2, 3: 3},
        "has_rot": True,
        "locked_rows": set(),
        "reverse_links": set(),
        "link_map2": {},
        "one_way_rows": {},
        "cursor_spawns": [(0, 1), (1, 5), (5, 1), (1, 0)],
        "max_moves": 84,
    },
    {
        "n": 5,
        "initial": [
            [4, 0, 2, 0, 4],
            [0, 1, 2, 1, 0],
            [3, 2, 4, 1, 1],
            [2, 3, 0, 3, 2],
            [3, 3, 4, 1, 4],
        ],
        "link_map": {0: 0, 1: 1, 2: 2, 3: 3, 4: 4},
        "has_rot": False,
        "locked_rows": {1, 3},
        "reverse_links": set(),
        "link_map2": {},
        "one_way_rows": {},
        "cursor_spawns": [(0, 1), (1, 6), (6, 1), (1, 0)],
        "max_moves": 128,
    },
    {
        "n": 6,
        "initial": [
            [5, 0, 5, 0, 5, 0],
            [0, 1, 0, 1, 0, 1],
            [2, 1, 2, 1, 2, 1],
            [2, 3, 2, 3, 2, 3],
            [4, 4, 3, 4, 3, 3],
            [4, 5, 4, 5, 4, 5],
        ],
        "link_map": {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5},
        "has_rot": True,
        "locked_rows": set(),
        "reverse_links": {1, 4},
        "link_map2": {},
        "one_way_rows": {},
        "cursor_spawns": [(0, 1), (1, 7), (7, 1), (1, 0)],
        "max_moves": 132,
    },
    {
        "n": 8,
        "initial": [
            [1, 7, 7, 0, 0, 0, 0, 0],
            [2, 0, 0, 1, 1, 1, 1, 1],
            [3, 1, 1, 2, 2, 2, 2, 2],
            [4, 2, 2, 3, 3, 3, 3, 3],
            [3, 4, 3, 4, 4, 4, 4, 5],
            [6, 4, 4, 5, 5, 5, 5, 5],
            [6, 7, 5, 6, 6, 6, 6, 6],
            [0, 6, 5, 7, 7, 7, 7, 7],
        ],
        "link_map": {i: i % 4 for i in range(8)},
        "rev_link": {0: 4, 1: 5, 2: 6, 3: 7},
        "has_rot": True,
        "locked_rows": set(),
        "reverse_links": set(),
        "link_map2": {0: 4, 2: 6},
        "one_way_rows": {2: 1, 5: -1},
        "cursor_spawns": [(0, 1), (1, 9), (9, 1), (1, 0)],
        "max_moves": 148,
    },
]

levels = [Level(sprites=[], grid_size=(GRID_SIZE, GRID_SIZE)) for _ in LEVEL_CONFIGS]


class Cl02(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)
        self._grid: list[list[int]] = [[0] * 4 for _ in range(4)]
        self._link_map: dict[int, int] = {}
        self._rev_link: dict[int, int] = {}
        self._n: int = 4
        self._cell: int = 0
        self._cursor_gr: int = 0
        self._cursor_gc: int = 1
        self._moves_used: int = 0
        self._variation: int = 0
        self._max_moves: int = 0
        self._lives: int = MAX_LIVES
        self._has_rot: bool = False
        self._locked_rows: set[int] = set()
        self._reverse_links: set[int] = set()
        self._link_map2: dict[int, int] = {}
        self._one_way_rows: dict[int, int] = {}
        self._history: list[dict] = []
        self._game_won: bool = False
        self._game_lost: bool = False

        camera = Camera(background=BG, letter_box=BG, width=GRID_SIZE, height=GRID_SIZE)
        super().__init__(
            game_id="cl02",
            levels=levels,
            camera=camera,
            available_actions=[0, 1, 2, 3, 4, 5, 7],
            seed=seed,
        )

    def get_actions(self) -> list[int]:
        return self._available_actions

    def on_set_level(self, level: Level) -> None:
        cfg = LEVEL_CONFIGS[min(self.level_index, len(LEVEL_CONFIGS) - 1)]
        self._n = cfg["n"]
        self._cell = self._compute_cell()
        self._grid = [row[:] for row in cfg["initial"]]
        self._link_map = dict(cfg["link_map"])
        if "rev_link" in cfg:
            self._rev_link = dict(cfg["rev_link"])
        else:
            self._rev_link = {v: k for k, v in self._link_map.items()}
        self._max_moves = cfg["max_moves"]
        self._has_rot = cfg["has_rot"]
        self._locked_rows = set(cfg.get("locked_rows", set()))
        self._reverse_links = set(cfg.get("reverse_links", set()))
        self._link_map2 = dict(cfg.get("link_map2", {}))
        self._one_way_rows = dict(cfg.get("one_way_rows", {}))
        self._moves_used = 0
        self._game_won = False
        self._game_lost = False
        spawn_positions = cfg.get("cursor_spawns", [(0, 1)])
        idx = self._variation % len(spawn_positions)
        self._cursor_gr, self._cursor_gc = spawn_positions[idx]
        self._history = []
        level.remove_all_sprites()
        self._render(level)

    def handle_reset(self) -> None:
        from arcengine import GameState as EngineState

        had_progress = (
            self._state == EngineState.GAME_OVER
            or self._moves_used > 0
            or self._lives < MAX_LIVES
        )
        self._lives = MAX_LIVES
        if self._state == EngineState.WIN:
            self._variation += 1
            self.full_reset()
        elif had_progress:
            self._variation += 1
            self.level_reset()
        else:
            self.full_reset()

    def _lose_life(self) -> None:
        self._lives -= 1
        self._variation += 1
        self._game_lost = True

    def _save_state(self) -> dict:
        return {
            "grid": [row[:] for row in self._grid],
            "cursor_gr": self._cursor_gr,
            "cursor_gc": self._cursor_gc,
            "moves_used": self._moves_used,
        }

    def _restore_state(self, snap: dict) -> None:
        self._grid = [row[:] for row in snap["grid"]]
        self._cursor_gr = snap["cursor_gr"]
        self._cursor_gc = snap["cursor_gc"]
        self._moves_used = snap["moves_used"]

    def _compute_cell(self) -> int:
        t = self._n + 2
        return max(1, (GRID_SIZE - 4) // t)

    def _grid_origin(self) -> tuple[int, int]:
        t = self._n + 2
        size = t * self._cell
        x0 = (GRID_SIZE - size) // 2
        y0 = (GRID_SIZE - size - 4) // 2
        return max(0, x0), max(0, y0)

    def _cell_xy(self, gr: int, gc: int) -> tuple[int, int]:
        x0, y0 = self._grid_origin()
        return x0 + gc * self._cell, y0 + gr * self._cell

    def _cell_type(self, gr: int, gc: int) -> str:
        t = self._n + 2
        if gr == 0 and gc == 0:
            return "cw"
        if gr == 0 and gc == t - 1:
            return "ccw"
        if gr == t - 1 and gc == 0:
            return "ccw"
        if gr == t - 1 and gc == t - 1:
            return "cw"
        if gr == 0:
            return "col_up"
        if gr == t - 1:
            return "col_down"
        if gc == 0:
            return "row_left"
        if gc == t - 1:
            return "row_right"
        return "color"

    def _shift_row_left_raw(self, r: int) -> None:
        row = self._grid[r]
        self._grid[r] = row[1:] + [row[0]]

    def _shift_row_right_raw(self, r: int) -> None:
        row = self._grid[r]
        self._grid[r] = [row[-1]] + row[:-1]

    def _shift_col_up_raw(self, c: int) -> None:
        top = self._grid[0][c]
        for r in range(self._n - 1):
            self._grid[r][c] = self._grid[r + 1][c]
        self._grid[self._n - 1][c] = top

    def _shift_col_down_raw(self, c: int) -> None:
        bot = self._grid[self._n - 1][c]
        for r in range(self._n - 1, 0, -1):
            self._grid[r][c] = self._grid[r - 1][c]
        self._grid[0][c] = bot

    def _rotate_cw_raw(self) -> None:
        n = self._n
        ng = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                ng[j][n - 1 - i] = self._grid[i][j]
        self._grid = ng

    def _rotate_ccw_raw(self) -> None:
        n = self._n
        ng = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                ng[n - 1 - j][i] = self._grid[i][j]
        self._grid = ng

    def _coupled_row(self, r: int, d: int) -> bool:
        if r in self._locked_rows:
            return False
        if r in self._one_way_rows and self._one_way_rows[r] != d:
            return False
        lc = self._link_map.get(r, r % self._n)
        rev = r in self._reverse_links
        if d == 1:
            self._shift_row_right_raw(r)
            self._shift_col_up_raw(lc) if rev else self._shift_col_down_raw(lc)
        else:
            self._shift_row_left_raw(r)
            self._shift_col_down_raw(lc) if rev else self._shift_col_up_raw(lc)
        lc2 = self._link_map2.get(r)
        if lc2 is not None:
            self._shift_col_down_raw(lc2) if d == 1 else self._shift_col_up_raw(lc2)
        self._moves_used += 1
        return True

    def _coupled_col(self, c: int, d: int) -> bool:
        lr = self._rev_link.get(c, c % self._n)
        if lr in self._locked_rows:
            return False
        if d == 1:
            self._shift_col_down_raw(c)
            self._shift_row_right_raw(lr)
        else:
            self._shift_col_up_raw(c)
            self._shift_row_left_raw(lr)
        self._moves_used += 1
        return True

    def _coupled_rot_cw(self) -> bool:
        self._rotate_cw_raw()
        self._moves_used += 1
        return True

    def _coupled_rot_ccw(self) -> bool:
        self._rotate_ccw_raw()
        self._moves_used += 1
        return True

    def _execute_cursor(self) -> bool:
        gr, gc = self._cursor_gr, self._cursor_gc
        ct = self._cell_type(gr, gc)
        if ct == "row_left":
            return self._coupled_row(gr - 1, -1)
        if ct == "row_right":
            return self._coupled_row(gr - 1, 1)
        if ct == "col_up":
            return self._coupled_col(gc - 1, -1)
        if ct == "col_down":
            return self._coupled_col(gc - 1, 1)
        if ct == "cw":
            return self._coupled_rot_cw() if self._has_rot else False
        if ct == "ccw":
            return self._coupled_rot_ccw() if self._has_rot else False
        return False

    def _rows_uniform(self) -> bool:
        return all(len(set(self._grid[r])) == 1 for r in range(self._n))

    def _cols_uniform(self) -> bool:
        return all(
            len({self._grid[r][c] for r in range(self._n)}) == 1 for c in range(self._n)
        )

    def _check_win(self) -> bool:
        return self._rows_uniform() or self._cols_uniform()

    def _sprite(
        self,
        level: Level,
        x: int,
        y: int,
        w: int,
        h: int,
        color: int,
        name: str,
        layer: int = 0,
    ) -> None:
        if w <= 0 or h <= 0:
            return
        pixels = [[color] * w for _ in range(h)]
        sp = Sprite(
            pixels=pixels, name=name, visible=True, collidable=False, layer=layer
        )
        sp.set_position(x, y)
        level.add_sprite(sp)

    def _draw_icon(
        self,
        level: Level,
        x: int,
        y: int,
        cs: int,
        pattern: list,
        bg: int,
        fg: int,
        name: str,
    ) -> None:
        pixels = [[bg] * cs for _ in range(cs)]
        off = (cs - 5) // 2
        for pr in range(5):
            for pc in range(5):
                if pattern[pr][pc]:
                    ry, rx = pr + off, pc + off
                    if 0 <= ry < cs and 0 <= rx < cs:
                        pixels[ry][rx] = fg
        sp = Sprite(pixels=pixels, name=name, visible=True, collidable=False, layer=2)
        sp.set_position(x, y)
        level.add_sprite(sp)

    def _draw_color_cell(
        self, level: Level, gr: int, gc: int, x: int, y: int, cs: int
    ) -> None:
        color = TILE_PAL[self._grid[gr - 1][gc - 1]]
        self._sprite(level, x, y, cs, cs, color, f"cl{gr}{gc}", 1)
        self._sprite(level, x, y, cs, 1, LIGHT_GREY, f"bdt{gr}{gc}", 3)
        self._sprite(level, x, y + cs - 1, cs, 1, LIGHT_GREY, f"bdb{gr}{gc}", 3)
        self._sprite(level, x, y, 1, cs, LIGHT_GREY, f"bdl{gr}{gc}", 3)
        self._sprite(level, x + cs - 1, y, 1, cs, LIGHT_GREY, f"bdr{gr}{gc}", 3)

    def _draw_row_arrow(
        self, level: Level, ct: str, gr: int, x: int, y: int, cs: int
    ) -> None:
        row_idx = gr - 1
        locked = row_idx in self._locked_rows
        if locked:
            self._draw_icon(
                level, x, y, cs, _PX_LOCK, OFF_BLACK, LIGHT_GREY, f"ar{gr}{ct}"
            )
            return
        one_way_d = self._one_way_rows.get(row_idx)
        blocked = one_way_d is not None and (
            (ct == "row_left" and one_way_d == 1)
            or (ct == "row_right" and one_way_d == -1)
        )
        pat = _PX_L if ct == "row_left" else _PX_R
        row_ok = len(set(self._grid[row_idx])) == 1
        is_rev = row_idx in self._reverse_links
        icon_color = GREEN if row_ok else (RED if is_rev else YELLOW)
        bg_col = BG if blocked else OFF_BLACK
        self._draw_icon(level, x, y, cs, pat, bg_col, icon_color, f"ar{gr}{ct}")

    def _draw_col_arrow(
        self, level: Level, ct: str, gc: int, x: int, y: int, cs: int
    ) -> None:
        col_idx = gc - 1
        linked_row = self._rev_link.get(col_idx, col_idx % self._n)
        if linked_row in self._locked_rows:
            self._draw_icon(
                level, x, y, cs, _PX_LOCK, OFF_BLACK, LIGHT_GREY, f"ac{gc}{ct}"
            )
            return
        pat = _PX_U if ct == "col_up" else _PX_D
        col_ok = len({self._grid[r][col_idx] for r in range(self._n)}) == 1
        fg = GREEN if col_ok else YELLOW
        self._draw_icon(level, x, y, cs, pat, OFF_BLACK, fg, f"ac{gc}{ct}")

    def _draw_corner(
        self, level: Level, ct: str, gr: int, gc: int, x: int, y: int, cs: int
    ) -> None:
        pat = _PX_CW if ct == "cw" else _PX_CCW
        corner_fg = YELLOW if self._has_rot else OFF_BLACK
        corner_bg = OFF_BLACK if self._has_rot else BG
        self._draw_icon(level, x, y, cs, pat, corner_bg, corner_fg, f"rot{gr}{gc}")

    def _draw_cells(self, level: Level) -> None:
        t = self._n + 2
        cursor_gr, cursor_gc = self._cursor_gr, self._cursor_gc

        for gr in range(t):
            for gc in range(t):
                x, y = self._cell_xy(gr, gc)
                ct = self._cell_type(gr, gc)

                if ct == "color":
                    self._draw_color_cell(level, gr, gc, x, y, self._cell)
                elif ct in ("row_left", "row_right"):
                    self._draw_row_arrow(level, ct, gr, x, y, self._cell)
                elif ct in ("col_up", "col_down"):
                    self._draw_col_arrow(level, ct, gc, x, y, self._cell)
                elif ct in ("cw", "ccw"):
                    self._draw_corner(level, ct, gr, gc, x, y, self._cell)

                if gr == cursor_gr and gc == cursor_gc:
                    cs = self._cell
                    self._sprite(level, x, y, cs, 1, ORANGE, "cut", 5)
                    self._sprite(level, x, y + cs - 1, cs, 1, ORANGE, "cub", 5)
                    self._sprite(level, x, y, 1, cs, ORANGE, "cul", 5)
                    self._sprite(level, x + cs - 1, y, 1, cs, ORANGE, "cur", 5)

    def _draw_hud(self, level: Level) -> None:
        x0, y0 = self._grid_origin()
        t = self._n + 2
        bar_y = y0 + t * self._cell + 2
        if bar_y > GRID_SIZE - 3:
            bar_y = GRID_SIZE - 3
        bar_w = 40
        bar_x = (GRID_SIZE - bar_w - MAX_LIVES * 5) // 2
        remaining = max(0, self._max_moves - self._moves_used)
        filled = (remaining * bar_w) // self._max_moves if self._max_moves > 0 else 0
        for dx in range(bar_w):
            color = GREEN if dx < filled else OFF_BLACK
            self._sprite(level, bar_x + dx, bar_y, 1, 2, color, f"b{dx}", 4)
        for i in range(MAX_LIVES):
            color = RED if i < self._lives else OFF_BLACK
            self._sprite(
                level, bar_x + bar_w + 2 + i * 5, bar_y, 3, 2, color, f"lf{i}", 4
            )

    def _draw_win_screen(self, level: Level) -> None:
        level.remove_all_sprites()
        pixels = [[BG] * GRID_SIZE for _ in range(GRID_SIZE)]
        for x in range(GRID_SIZE):
            for t in range(3):
                pixels[t][x] = GREEN
                pixels[GRID_SIZE - 1 - t][x] = GREEN
        for y in range(GRID_SIZE):
            for t in range(3):
                pixels[y][t] = GREEN
                pixels[y][GRID_SIZE - 1 - t] = GREEN
        cx, cy = GRID_SIZE // 2, GRID_SIZE // 2
        for i in range(12):
            for j in range(12):
                pixels[cy - 6 + i][cx - 6 + j] = GREEN
        for i in range(6):
            for j in range(6):
                pixels[cy - 3 + i][cx - 3 + j] = YELLOW
        sp = Sprite(pixels=pixels, name="win", visible=True, collidable=False, layer=10)
        sp.set_position(0, 0)
        level.add_sprite(sp)

    def _draw_lose_screen(self, level: Level) -> None:
        level.remove_all_sprites()
        pixels = [[BG] * GRID_SIZE for _ in range(GRID_SIZE)]
        for x in range(GRID_SIZE):
            for t in range(3):
                pixels[t][x] = RED
                pixels[GRID_SIZE - 1 - t][x] = RED
        for y in range(GRID_SIZE):
            for t in range(3):
                pixels[y][t] = RED
                pixels[y][GRID_SIZE - 1 - t] = RED
        cx, cy = GRID_SIZE // 2, GRID_SIZE // 2
        for i in range(10):
            for j in range(10):
                pixels[cy - 5 + i][cx - 5 + j] = RED
        for i in range(4):
            for j in range(4):
                pixels[cy - 2 + i][cx - 2 + j] = OFF_BLACK
        sp = Sprite(
            pixels=pixels, name="lose", visible=True, collidable=False, layer=10
        )
        sp.set_position(0, 0)
        level.add_sprite(sp)

    def _render(self, level: Level) -> None:
        level.remove_all_sprites()
        self._sprite(level, 0, 0, GRID_SIZE, GRID_SIZE, BG, "bg", 0)
        self._draw_cells(level)
        self._draw_hud(level)

    def _process_outcome(self) -> None:
        level = self.current_level
        if self._game_won:
            self._lives = MAX_LIVES
            if self.is_last_level():
                self.win()
            else:
                self._variation += 1
                self.next_level()
        elif self._game_lost:
            self._render(level)
            if self._lives <= 0:
                self.lose()
            else:
                self.level_reset()
        self.complete_action()

    def _handle_action(self) -> None:
        action = self.action.id
        level = self.current_level
        t = self._n + 2

        if action == GameAction.ACTION7:
            if self._history and self._moves_used < self._max_moves:
                charged_moves = self._moves_used + 1
                self._restore_state(self._history.pop())
                self._moves_used = charged_moves
            elif self._moves_used < self._max_moves:
                self._moves_used += 1
            if self._moves_used >= self._max_moves and not self._game_won:
                self._lose_life()
            self._render(level)
            return

        snap = self._save_state()
        moved = False
        cursor_moved = False

        if action == GameAction.ACTION1:
            prev = self._cursor_gr
            self._cursor_gr = max(0, self._cursor_gr - 1)
            if self._cursor_gr != prev:
                self._moves_used += 1
                cursor_moved = True

        elif action == GameAction.ACTION2:
            prev = self._cursor_gr
            self._cursor_gr = min(t - 1, self._cursor_gr + 1)
            if self._cursor_gr != prev:
                self._moves_used += 1
                cursor_moved = True

        elif action == GameAction.ACTION3:
            prev = self._cursor_gc
            self._cursor_gc = max(0, self._cursor_gc - 1)
            if self._cursor_gc != prev:
                self._moves_used += 1
                cursor_moved = True

        elif action == GameAction.ACTION4:
            prev = self._cursor_gc
            self._cursor_gc = min(t - 1, self._cursor_gc + 1)
            if self._cursor_gc != prev:
                self._moves_used += 1
                cursor_moved = True

        elif action == GameAction.ACTION5:
            moved = self._execute_cursor()

        if moved or cursor_moved:
            self._history.append(snap)
            if len(self._history) > 50:
                self._history.pop(0)

        self._render(level)

        if moved and self._check_win():
            self._game_won = True
            return

        if (moved or cursor_moved) and self._moves_used >= self._max_moves:
            self._lose_life()

    def step(self) -> None:
        self._handle_action()

        if self._game_won or self._game_lost:
            self._process_outcome()
            return

        self.complete_action()


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
        self._engine = Cl02(seed=seed)
        self._total_turns = 0
        self._done = False
        self._game_won = False
        self._game_over = False
        self._last_action_was_reset = False

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
        total = len(e._levels)
        remaining = max(0, e._max_moves - e._moves_used)

        header = (
            f"Level {e.level_index + 1}/{total}"
            f" | Moves: {e._moves_used}/{e._max_moves}"
            f" | Lives: {e._lives}/{MAX_LIVES}"
        )

        n = e._n
        grid_lines = []
        for r in range(n):
            row_vals = " ".join(str(e._grid[r][c]) for c in range(n))
            grid_lines.append(row_vals)
        grid_str = "\n".join(grid_lines)

        ct = e._cell_type(e._cursor_gr, e._cursor_gc)
        cursor_info = f"Cursor: ({e._cursor_gr},{e._cursor_gc}) [{ct}]"

        rules = "Shift rows/cols to make all rows or all columns uniform."
        if e._has_rot:
            rules += " Rotation available (CW corners, undo=CCW)."
        if e._locked_rows:
            rules += f" Locked rows: {sorted(e._locked_rows)}."
        if e._reverse_links:
            rules += f" Reverse-linked rows: {sorted(e._reverse_links)}."

        rows_done = sum(1 for r in range(n) if len(set(e._grid[r])) == 1)
        cols_done = sum(
            1 for c in range(n) if len({e._grid[r][c] for r in range(n)}) == 1
        )

        return (
            header
            + f"\nGrid ({n}x{n}):\n{grid_str}"
            + f"\n{cursor_info}"
            + f"\nRows uniform: {rows_done}/{n} | Cols uniform: {cols_done}/{n}"
            + f"\nMoves remaining: {remaining}"
            + f"\n{rules}"
        )

    def _build_game_state(self, done: bool = False) -> GameState:
        e = self._engine
        frame = self._render_frame()
        image_bytes = self._frame_to_png(frame)

        valid_actions = self.get_actions() if not done else None

        return GameState(
            text_observation=self._build_text_observation(),
            image_observation=image_bytes,
            valid_actions=valid_actions,
            turn=self._total_turns,
            metadata={
                "total_levels": len(e._levels),
                "level_index": e.level_index,
                "levels_completed": e._score,
                "game_over": self._game_over,
                "done": done,
                "info": {},
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
        self._game_over = False

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
                f"Invalid action '{action}'. "
                f"Must be one of {list(self._ACTION_MAP.keys())}"
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
            self._game_over = False
            info["reason"] = "game_complete"
            return StepResult(
                state=self._build_game_state(done=True),
                reward=level_reward,
                done=True,
                info=info,
            )

        if game_over:
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
