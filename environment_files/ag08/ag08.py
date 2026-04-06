import io
import random
import struct
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import gymnasium as gym
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

PADDING_COLOR = 5

BG = 5
WALL = 3
REAGENT_RED = 8
VAT_COLOR = 2
REAGENT_YELLOW = 11
SHELF_COLOR = 4
REAGENT_BLUE = 9
PIVOT_COLOR = 1
CURSOR_COLOR = 14
SELECTED_COLOR = 6
RESULT_PURPLE = 15
RESULT_TEAL = 10
RESULT_ORANGE = 12
RESULT_BROWN = 13
EMPTY_SLOT = 0

GRID_W = 11
GRID_H = 11
VAT_L_COL = 2
VAT_M_COL = 5
VAT_R_COL = 8
VAT_ROW = 8
SHELF_ROW = 7
PIVOT_COL = 5
SHELF_COLS = list(range(VAT_L_COL, VAT_R_COL + 1))

PALETTE_ROW = 0
PALETTE_COLORS = [REAGENT_RED, REAGENT_YELLOW, REAGENT_BLUE]
PALETTE_COLS = [4, 5, 6]

ORIGINAL_REAGENTS = [REAGENT_RED, REAGENT_YELLOW, REAGENT_BLUE]
ORIGINAL_RESULTS = [RESULT_PURPLE, RESULT_TEAL, RESULT_ORANGE, RESULT_BROWN]
ALL_GAME_COLORS = ORIGINAL_REAGENTS + ORIGINAL_RESULTS
UI_COLORS = {
    BG,
    WALL,
    VAT_COLOR,
    SHELF_COLOR,
    PIVOT_COLOR,
    CURSOR_COLOR,
    SELECTED_COLOR,
    EMPTY_SLOT,
    PADDING_COLOR,
}
AVAILABLE_COLORS = [c for c in range(16) if c not in UI_COLORS]


def _make_levels():
    data = []

    data.append(
        {
            "palette": {REAGENT_RED: 2, REAGENT_YELLOW: 2, REAGENT_BLUE: 2},
            "vats": [
                {
                    "col": VAT_L_COL,
                    "recipe": [REAGENT_RED, REAGENT_YELLOW],
                    "result": RESULT_ORANGE,
                    "hint": [REAGENT_RED, REAGENT_YELLOW],
                },
                {
                    "col": VAT_M_COL,
                    "recipe": [REAGENT_RED, REAGENT_BLUE],
                    "result": RESULT_PURPLE,
                    "hint": [REAGENT_RED, REAGENT_BLUE],
                },
                {
                    "col": VAT_R_COL,
                    "recipe": [REAGENT_YELLOW, REAGENT_BLUE],
                    "result": RESULT_TEAL,
                    "hint": [REAGENT_YELLOW, REAGENT_BLUE],
                },
            ],
            "move_limit": 320,
            "two_shelves": False,
        }
    )

    data.append(
        {
            "palette": {REAGENT_RED: 3, REAGENT_YELLOW: 2, REAGENT_BLUE: 2},
            "vats": [
                {
                    "col": VAT_L_COL,
                    "recipe": [REAGENT_RED, REAGENT_YELLOW, REAGENT_BLUE],
                    "result": RESULT_BROWN,
                    "hint": [REAGENT_RED, REAGENT_YELLOW, REAGENT_BLUE],
                },
                {
                    "col": VAT_M_COL,
                    "recipe": [REAGENT_RED, REAGENT_YELLOW],
                    "result": RESULT_ORANGE,
                    "hint": [REAGENT_RED, REAGENT_YELLOW],
                },
                {
                    "col": VAT_R_COL,
                    "recipe": [REAGENT_RED, REAGENT_BLUE],
                    "result": RESULT_PURPLE,
                    "hint": [REAGENT_RED, REAGENT_BLUE],
                },
            ],
            "move_limit": 480,
            "two_shelves": False,
        }
    )

    data.append(
        {
            "palette": {REAGENT_RED: 3, REAGENT_YELLOW: 3, REAGENT_BLUE: 3},
            "vats": [
                {
                    "col": VAT_L_COL,
                    "recipe": [REAGENT_RED, REAGENT_BLUE],
                    "result": RESULT_PURPLE,
                    "hint": [REAGENT_RED, REAGENT_BLUE],
                },
                {
                    "col": VAT_R_COL,
                    "recipe": [REAGENT_YELLOW, REAGENT_BLUE],
                    "result": RESULT_TEAL,
                    "hint": [REAGENT_YELLOW, REAGENT_BLUE],
                },
            ],
            "move_limit": 624,
            "two_shelves": True,
            "shelf2_beam_row": 4,
            "shelf2_vat_row": 5,
            "shelf2_vats": [
                {
                    "col": VAT_L_COL,
                    "recipe": [REAGENT_RED, REAGENT_YELLOW],
                    "result": RESULT_ORANGE,
                    "hint": [REAGENT_RED, REAGENT_YELLOW],
                },
                {
                    "col": VAT_R_COL,
                    "recipe": [REAGENT_RED, REAGENT_YELLOW, REAGENT_BLUE],
                    "result": RESULT_BROWN,
                    "hint": [REAGENT_RED, REAGENT_YELLOW, REAGENT_BLUE],
                },
            ],
        }
    )

    data.append(
        {
            "palette": {REAGENT_RED: 4, REAGENT_YELLOW: 3, REAGENT_BLUE: 4},
            "vats": [
                {
                    "col": VAT_L_COL,
                    "recipe": [REAGENT_RED, REAGENT_YELLOW, REAGENT_BLUE],
                    "result": RESULT_BROWN,
                    "hint": [REAGENT_RED, REAGENT_YELLOW, REAGENT_BLUE],
                },
                {
                    "col": VAT_M_COL,
                    "recipe": [REAGENT_RED, REAGENT_BLUE],
                    "result": RESULT_PURPLE,
                    "hint": [REAGENT_RED, REAGENT_BLUE],
                },
                {
                    "col": VAT_R_COL,
                    "recipe": [REAGENT_RED, REAGENT_YELLOW],
                    "result": RESULT_ORANGE,
                    "hint": [REAGENT_RED, REAGENT_YELLOW],
                },
            ],
            "move_limit": 680,
            "two_shelves": True,
            "shelf2_beam_row": 4,
            "shelf2_vat_row": 5,
            "shelf2_vats": [
                {
                    "col": VAT_L_COL,
                    "recipe": [REAGENT_YELLOW, REAGENT_BLUE],
                    "result": RESULT_TEAL,
                    "hint": [REAGENT_YELLOW, REAGENT_BLUE],
                },
                {
                    "col": VAT_R_COL,
                    "recipe": [REAGENT_RED, REAGENT_BLUE],
                    "result": RESULT_PURPLE,
                    "hint": [REAGENT_RED, REAGENT_BLUE],
                },
            ],
        }
    )

    return data


LEVEL_DATA = _make_levels()

levels = [Level(sprites=[], grid_size=(GRID_W, GRID_H), data=ld) for ld in LEVEL_DATA]


class HudDisplay(RenderableUserDisplay):
    def __init__(self, game):
        self.game = game

    def render_interface(self, frame):
        g = self.game
        fh, fw = frame.shape[:2]

        scale = min(fw // GRID_W, fh // GRID_H)
        if scale == 0:
            return frame
        x_off = (fw - GRID_W * scale) // 2

        grid_left = x_off
        grid_right = x_off + GRID_W * scale
        grid_top = 2
        grid_bot = grid_top + GRID_H * scale

        border_t = grid_top - 1
        border_b = grid_bot
        border_l = grid_left - 1
        border_r = grid_right

        if 0 <= border_t < fh:
            for x in range(max(0, border_l), min(fw, border_r + 1)):
                frame[border_t, x] = WALL
        if 0 <= border_b < fh:
            for x in range(max(0, border_l), min(fw, border_r + 1)):
                frame[border_b, x] = WALL
        for y in range(max(0, border_t), min(fh, border_b + 1)):
            if 0 <= border_l < fw:
                frame[y, border_l] = WALL
            if 0 <= border_r < fw:
                frame[y, border_r] = WALL

        life_sz = 2
        life_gap = 1
        life_total = 3 * life_sz + 2 * life_gap
        lx = grid_left
        for i in range(3):
            cx = lx + i * (life_sz + life_gap)
            if g._lives > i:
                clr = g._remap(REAGENT_RED)
            else:
                clr = SHELF_COLOR
            for dx in range(life_sz):
                if 0 <= cx + dx < fw:
                    frame[0, cx + dx] = clr

        bar_x = grid_left + life_total + 2
        bar_w = grid_right - bar_x
        if g.max_moves > 0 and bar_w > 0:
            remaining = max(0, g.max_moves - g.moves_used)
            ratio = remaining / g.max_moves
            filled_px = int(ratio * bar_w)
            bar_clr = CURSOR_COLOR if ratio > 0.25 else g._remap(REAGENT_RED)
            if filled_px > 0:
                frame[0, bar_x : bar_x + filled_px] = bar_clr
            if filled_px < bar_w:
                frame[0, bar_x + filled_px : grid_right] = SHELF_COLOR

        all_vats = g._all_vats()
        num_t = len(all_vats)
        if num_t > 0:
            ty = border_b - 2
            tx = grid_right - 2
            for i in range(num_t - 1, -1, -1):
                for dy in range(2):
                    for dx in range(2):
                        py = ty + dy
                        px = tx + dx
                        if 0 <= py < fh and 0 <= px < fw:
                            frame[py, px] = all_vats[i]["result"]
                tx -= 3

        bot_py = border_b + 1

        cell_w = scale
        cell_h = scale
        for v in all_vats:
            contents = g._vat_contents.get(v["key"], [])
            if not contents:
                continue
            vx = grid_left + v["col"] * cell_w
            vy = grid_top + v["row"] * cell_h
            for ci, clr in enumerate(contents):
                ly = vy + cell_h - 2 - ci
                if ly < vy + 1:
                    break
                for dx in range(1, cell_w - 1):
                    _x = vx + dx
                    if 0 <= ly < fh and 0 <= _x < fw:
                        frame[ly, _x] = clr

        ind_x = grid_left + (GRID_W - 1) * scale
        ind_y = grid_top
        for dy in range(2):
            for dx in range(2):
                py = ind_y + dy
                px = ind_x + dx
                if 0 <= py < fh and 0 <= px < fw:
                    frame[py, px] = SELECTED_COLOR if g.holding else CURSOR_COLOR
        if g.holding and g.held_color != BG:
            for dy in range(2):
                for dx in range(2):
                    py = ind_y + 2 + dy
                    px = ind_x + dx
                    if 0 <= py < fh and 0 <= px < fw:
                        frame[py, px] = g.held_color

        def _px(x, y, c):
            if 0 <= x < fw and 0 <= y < fh:
                frame[y, x] = c

        def _rect(x, y, w, h, c):
            for dy in range(h):
                for dx in range(w):
                    _px(x + dx, y + dy, c)

        def _arrow_right(x, y, s):
            mid = y + s // 2
            _px(x, mid, WALL)
            _px(x + 1, mid, WALL)
            if s > 1:
                _px(x + 1, mid - 1, WALL)
            if s > 1:
                _px(x + 1, mid + 1, WALL)

        sq = 2
        recipe_w = sq + sq + 2 + 1 + sq
        num_recipes = 4
        total_gap = (fw - num_recipes * recipe_w) // (num_recipes + 1)
        start_x = total_gap

        recipes_fwd = [
            (g._remap(REAGENT_RED), g._remap(REAGENT_BLUE), g._remap(RESULT_PURPLE)),
            (g._remap(REAGENT_BLUE), g._remap(REAGENT_YELLOW), g._remap(RESULT_TEAL)),
            (g._remap(REAGENT_RED), g._remap(REAGENT_YELLOW), g._remap(RESULT_ORANGE)),
            (g._remap(REAGENT_BLUE), g._remap(RESULT_ORANGE), g._remap(RESULT_BROWN)),
        ]
        recipes_rev = [
            (g._remap(RESULT_PURPLE), g._remap(REAGENT_RED), g._remap(REAGENT_BLUE)),
            (g._remap(RESULT_TEAL), g._remap(REAGENT_BLUE), g._remap(REAGENT_YELLOW)),
            (g._remap(RESULT_ORANGE), g._remap(REAGENT_RED), g._remap(REAGENT_YELLOW)),
            (g._remap(RESULT_BROWN), g._remap(REAGENT_BLUE), g._remap(RESULT_ORANGE)),
        ]

        hints_y = bot_py
        hints_y2 = hints_y + sq + 1

        for ri, (c1, c2, cr) in enumerate(recipes_fwd):
            rx = start_x + ri * (recipe_w + total_gap)
            _rect(rx, hints_y, sq, sq, c1)
            _rect(rx + sq, hints_y, sq, sq, c2)
            ax = rx + sq * 2 + 1
            _arrow_right(ax, hints_y, sq)
            _rect(ax + 3, hints_y, sq, sq, cr)

        for ri, (src, d1, d2) in enumerate(recipes_rev):
            sx = start_x + ri * (recipe_w + total_gap)
            _rect(sx, hints_y2, sq, sq, src)
            ax = sx + sq + 1
            _arrow_right(ax, hints_y2, sq)
            _rect(ax + 3, hints_y2, sq, sq, d1)
            _rect(ax + 3 + sq, hints_y2, sq, sq, d2)

        return frame


class Ag08(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)
        self._lives = 3
        self.moves_used = 0
        self.max_moves = 0
        self.holding = False
        self.held_color = BG

        self._cursor_row = 0
        self._cursor_col = 5
        self._two_shelves = False
        self._vats = []
        self._s2_vats = []
        self._s2_beam_row = 4
        self._s2_vat_row = 5
        self._palette = {}
        self._vat_contents = {}

        self._color_map = {}
        self._palette_colors = list(PALETTE_COLORS)
        self._hud = HudDisplay(self)
        self._has_acted = False

        self._game_over = False
        self._level_done = False
        self._undo_stack = []

        super().__init__(
            "ag08",
            levels,
            Camera(0, 0, 64, 64, BG, PADDING_COLOR, [self._hud]),
            available_actions=[0, 1, 2, 3, 4, 5, 7],
        )

    def _shuffle_colors(self):
        shuffled = list(AVAILABLE_COLORS)
        self._rng.shuffle(shuffled)
        chosen = shuffled[: len(ALL_GAME_COLORS)]
        self._color_map = {orig: new for orig, new in zip(ALL_GAME_COLORS, chosen)}
        self._palette_colors = [self._color_map[c] for c in ORIGINAL_REAGENTS]

    def _remap(self, color):
        return self._color_map.get(color, color)

    def on_set_level(self, level):
        self._lives = 3
        self.moves_used = 0
        self.holding = False
        self.held_color = BG
        self._has_acted = False
        self._level_done = False
        self._game_over = False
        self._cursor_row = 0
        self._cursor_col = 1
        self._undo_stack = []

        self._shuffle_colors()

        ld = self.current_level.get_data

        self.max_moves = ld("move_limit")
        self._two_shelves = ld("two_shelves")

        raw_palette = ld("palette")
        self._palette = {self._remap(k): v for k, v in raw_palette.items()}

        raw_vats = ld("vats")
        self._vats = []
        self._vat_contents = {}
        for v in raw_vats:
            key = "m" + str(v["col"])
            self._vats.append(
                {
                    "col": v["col"],
                    "row": VAT_ROW,
                    "recipe": [self._remap(c) for c in v["recipe"]],
                    "result": self._remap(v["result"]),
                    "hint": [self._remap(c) for c in v["hint"]],
                    "key": key,
                }
            )
            self._vat_contents[key] = []

        self._s2_vats = []
        if self._two_shelves:
            self._s2_beam_row = ld("shelf2_beam_row")
            self._s2_vat_row = ld("shelf2_vat_row")
            raw_s2 = ld("shelf2_vats")
            for v in raw_s2:
                key = "s" + str(v["col"])
                self._s2_vats.append(
                    {
                        "col": v["col"],
                        "row": self._s2_vat_row,
                        "recipe": [self._remap(c) for c in v["recipe"]],
                        "result": self._remap(v["result"]),
                        "hint": [self._remap(c) for c in v["hint"]],
                        "key": key,
                    }
                )
                self._vat_contents[key] = []

        self._rebuild_board()

    def _all_vats(self):
        return self._vats + self._s2_vats

    def _recipe_match(self, vat_def):
        contents = sorted(self._vat_contents.get(vat_def["key"], []))
        recipe = sorted(vat_def["recipe"])
        return contents == recipe

    def _check_win(self):
        for v in self._all_vats():
            if not self._recipe_match(v):
                return False
        return True

    def _vat_at(self, row, col):
        for v in self._all_vats():
            if v["row"] == row and v["col"] == col:
                return v
        return None

    def _palette_at(self, row, col):
        if row != PALETTE_ROW:
            return None
        for i, pc in enumerate(self._palette_colors):
            if col == PALETTE_COLS[i]:
                return pc
        return None

    def _handle_select(self):
        cr = self._cursor_row
        cc = self._cursor_col

        if self.holding:
            v = self._vat_at(cr, cc)
            if v is not None:
                self._vat_contents[v["key"]].append(self.held_color)
                self.holding = False
                self.held_color = BG
                return
            pc = self._palette_at(cr, cc)
            if pc is not None and pc == self.held_color:
                self._palette[pc] = self._palette.get(pc, 0) + 1
                self.holding = False
                self.held_color = BG
                return
            return

        pc = self._palette_at(cr, cc)
        if pc is not None and self._palette.get(pc, 0) > 0:
            self._palette[pc] -= 1
            self.held_color = pc
            self.holding = True
            return

        v = self._vat_at(cr, cc)
        if v is not None:
            contents = self._vat_contents.get(v["key"], [])
            if contents:
                self.held_color = contents.pop()
                self.holding = True
            return

    def _draw_hints(self, grid, vat_row, vat_col, hint_colors):
        for i, hc in enumerate(hint_colors):
            hr = vat_row - 1 - i
            if 0 <= hr < GRID_H:
                grid[hr][vat_col] = hc

    def _build_frame(self):
        grid = [[BG] * GRID_W for _ in range(GRID_H)]

        for i, pc in enumerate(self._palette_colors):
            col = PALETTE_COLS[i]
            if self._palette.get(pc, 0) > 0:
                grid[PALETTE_ROW][col] = pc
            else:
                grid[PALETTE_ROW][col] = EMPTY_SLOT

        for c in SHELF_COLS:
            grid[SHELF_ROW][c] = SHELF_COLOR
        grid[SHELF_ROW][PIVOT_COL] = PIVOT_COLOR

        for v in self._vats:
            vc = v["col"]
            if self._recipe_match(v):
                grid[VAT_ROW][vc] = v["result"]
            else:
                grid[VAT_ROW][vc] = EMPTY_SLOT
                self._draw_hints(grid, VAT_ROW, vc, v["hint"])

        if self._two_shelves:
            br = self._s2_beam_row
            for c in SHELF_COLS:
                grid[br][c] = SHELF_COLOR
            grid[br][PIVOT_COL] = PIVOT_COLOR

            for v in self._s2_vats:
                vc = v["col"]
                vr = v["row"]
                if self._recipe_match(v):
                    grid[vr][vc] = v["result"]
                else:
                    grid[vr][vc] = EMPTY_SLOT
                    self._draw_hints(grid, vr, vc, v["hint"])

        cr = self._cursor_row
        cc = self._cursor_col
        if 0 <= cr < GRID_H and 0 <= cc < GRID_W:
            if self.holding:
                grid[cr][cc] = SELECTED_COLOR
            else:
                grid[cr][cc] = CURSOR_COLOR

        return grid

    def _rebuild_board(self):
        frame = self._build_frame()
        self.current_level.remove_all_sprites()
        spr = Sprite(
            pixels=frame,
            name="board",
            visible=True,
            collidable=False,
            tags=["board"],
            layer=0,
        )
        spr.set_position(0, 0)
        self.current_level.add_sprite(spr)

    def _consume_move(self):
        self.moves_used += 1
        if self.max_moves > 0 and self.moves_used >= self.max_moves:
            self._lose_life()
            return True
        return False

    def _lose_life(self):
        self._lives -= 1
        if self._lives <= 0:
            self._game_over = True
            self.lose()
            return
        self.moves_used = 0
        self._rebuild_board()

    def _save_undo_snapshot(self):
        self._undo_stack.append(
            {
                "cursor_row": self._cursor_row,
                "cursor_col": self._cursor_col,
                "holding": self.holding,
                "held_color": self.held_color,
                "palette": dict(self._palette),
                "vat_contents": {k: list(v) for k, v in self._vat_contents.items()},
                "lives": self._lives,
                "has_acted": self._has_acted,
            }
        )

    def _restore_undo_snapshot(self):
        if not self._undo_stack:
            return
        state = self._undo_stack.pop()
        self._cursor_row = state["cursor_row"]
        self._cursor_col = state["cursor_col"]
        self.holding = state["holding"]
        self.held_color = state["held_color"]
        self._palette = state["palette"]
        self._vat_contents = state["vat_contents"]
        self._lives = state["lives"]
        self._has_acted = state["has_acted"]
        self._rebuild_board()

    def step(self):
        act = self.action.id.value

        if self.action.id == GameAction.ACTION7:
            self._restore_undo_snapshot()
            if self._consume_move():
                self.complete_action()
                return
            self.complete_action()
            return

        self._save_undo_snapshot()

        self._has_acted = True

        if act == 5:
            self._handle_select()
            self._rebuild_board()
            if self._check_win():
                self._undo_stack = []
                self._level_done = True
                self.next_level()
                self.complete_action()
                return
            if self._consume_move():
                self.complete_action()
                return
            self.complete_action()
            return

        dx = dy = 0
        if act == 1:
            dy = -1
        elif act == 2:
            dy = 1
        elif act == 3:
            dx = -1
        elif act == 4:
            dx = 1

        new_row = self._cursor_row + dy
        new_col = self._cursor_col + dx

        if 0 <= new_row <= 10 and 0 <= new_col <= 10:
            if new_row == 10 or new_col == 0 or new_col == 10:
                pass
            else:
                self._cursor_row = new_row
                self._cursor_col = new_col

        self._rebuild_board()

        if self._consume_move():
            self.complete_action()
            return

        self.complete_action()


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


def _encode_png(frame):
    h, w = frame.shape[:2]
    if frame.ndim == 2:
        raw = np.zeros((h, w, 3), dtype=np.uint8)
        raw[:] = frame[:, :, None]
    else:
        raw = frame.astype(np.uint8)
    rows = [b"\x00" + raw[y].tobytes() for y in range(h)]
    raw_data = b"".join(rows)
    compressed = zlib.compress(raw_data)

    def _chunk(ctype, data):
        c = ctype + data
        return (
            struct.pack(">I", len(data))
            + c
            + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        )

    buf = io.BytesIO()
    buf.write(b"\x89PNG\r\n\x1a\n")
    buf.write(_chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)))
    buf.write(_chunk(b"IDAT", compressed))
    buf.write(_chunk(b"IEND", b""))
    return buf.getvalue()


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
        self._seed = seed
        self._engine = Ag08(seed=seed)
        self._total_turns = 0
        self._done = False
        self._game_won = False
        self._last_action_was_reset = False

    def reset(self) -> GameState:
        e = self._engine
        if self._game_won or self._last_action_was_reset:
            e.perform_action(ActionInput(id=GameAction.RESET, data={}))
            e.perform_action(ActionInput(id=GameAction.RESET, data={}))
        else:
            e.perform_action(ActionInput(id=GameAction.RESET, data={}))
        self._total_turns = 0
        self._done = False
        self._last_action_was_reset = True
        self._game_won = False
        return self._build_game_state()

    def step(self, action: str) -> StepResult:
        if action == "reset":
            state = self.reset()
            return StepResult(
                state=state, reward=0.0, done=False, info={"action": "reset"}
            )

        self._last_action_was_reset = False
        self._total_turns += 1

        e = self._engine
        game_action = self._ACTION_MAP.get(action, GameAction.RESET)
        info: Dict = {"action": action}
        level_before = e.level_index

        action_input = ActionInput(id=game_action, data={})
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
                state=self._build_game_state(done=False),
                reward=0.0,
                done=False,
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

    def _build_text_observation(self) -> str:
        STATIC_CHARS = {
            0: ".",
            5: " ",
            3: "#",
            2: "V",
            4: "=",
            1: "+",
            14: "@",
            6: "*",
        }
        GAME_COLOR_CHARS = {
            REAGENT_RED: "R",
            REAGENT_YELLOW: "Y",
            REAGENT_BLUE: "B",
            RESULT_PURPLE: "P",
            RESULT_TEAL: "T",
            RESULT_ORANGE: "O",
            RESULT_BROWN: "W",
        }
        cm = self._engine._color_map
        COLOR_CHARS = dict(STATIC_CHARS)
        for orig, char in GAME_COLOR_CHARS.items():
            COLOR_CHARS[cm.get(orig, orig)] = char
        grid = self._engine._build_frame()
        lines = []
        for row in grid:
            lines.append("".join(COLOR_CHARS.get(c, "?") for c in row))
        e = self._engine
        lines.append(f"lives:{e._lives} moves:{e.moves_used}/{e.max_moves}")
        lines.append(f"level:{e.level_index + 1}/{len(e._levels)}")
        if e.holding:
            lines.append(f"holding:{COLOR_CHARS.get(e.held_color, '?')}")
        return "\n".join(lines)

    def _build_game_state(self, done: bool = False) -> GameState:
        valid_actions = self.get_actions() if not done else None
        rgb = self.render(mode="rgb_array")
        img_bytes = _encode_png(rgb)
        return GameState(
            text_observation=self._build_text_observation(),
            image_observation=img_bytes,
            valid_actions=valid_actions,
            turn=self._total_turns,
            metadata={
                "total_levels": len(self._engine._levels),
                "level_index": self._engine.level_index,
                "levels_completed": getattr(self._engine, "_score", 0),
                "game_over": getattr(getattr(self._engine, "_state", None), "name", "")
                == "GAME_OVER",
                "done": done,
                "info": {},
            },
        )

    def get_actions(self) -> List[str]:
        if self._done:
            return ["reset"]
        return list(self._VALID_ACTIONS)

    def is_done(self) -> bool:
        return self._done

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError(f"Unsupported render mode: {mode}")
        e = self._engine
        idx_frame = e.camera.render(e.current_level.get_sprites())
        h, w = idx_frame.shape[:2]
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for idx in range(len(ARC_PALETTE)):
            mask = idx_frame == idx
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            rgb[mask] = ARC_PALETTE[idx]
        return rgb

    def close(self) -> None:
        self._engine = None


class ArcGameEnv(gym.Env):
    metadata: Dict[str, Any] = {"render_modes": ["rgb_array"], "render_fps": 5}
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

    def __init__(self, seed: int = 0, render_mode: Optional[str] = None) -> None:
        super().__init__()
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"Unsupported render_mode '{render_mode}'. "
                f"Supported: {self.metadata['render_modes']}"
            )
        self.render_mode = render_mode
        self._action_to_string = {i: a for i, a in enumerate(self.ACTION_LIST)}
        self._string_to_action = {a: i for i, a in enumerate(self.ACTION_LIST)}
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.OBS_HEIGHT, self.OBS_WIDTH, 3),
            dtype=np.uint8,
        )
        self.action_space = spaces.Discrete(len(self.ACTION_LIST))
        self._seed = seed
        self._env = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed
        self._env = PuzzleEnvironment(seed=self._seed)
        state = self._env.reset()
        return self._get_obs(), self._build_info(state)

    def step(self, action: int):
        action_str = self._action_to_string[int(action)]
        result = self._env.step(action_str)
        obs = self._get_obs()
        reward = result.reward
        terminated = result.done
        truncated = False
        info = self._build_info(result.state, result.info)
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._get_obs()
        return None

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None

    def action_mask(self):
        mask = np.zeros(len(self.ACTION_LIST), dtype=np.int8)
        if self._env is not None:
            for a in self._env.get_actions():
                idx = self._string_to_action.get(a)
                if idx is not None:
                    mask[idx] = 1
        return mask

    def _get_obs(self):
        frame = self._env.render(mode="rgb_array")
        if frame.shape[0] != self.OBS_HEIGHT or frame.shape[1] != self.OBS_WIDTH:
            frame = self._resize_nearest(frame, self.OBS_HEIGHT, self.OBS_WIDTH)
        return frame

    @staticmethod
    def _resize_nearest(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        h, w = img.shape[:2]
        row_idx = (np.arange(target_h) * h // target_h).astype(int)
        col_idx = (np.arange(target_w) * w // target_w).astype(int)
        return img[np.ix_(row_idx, col_idx)].astype(np.uint8)

    def _build_info(self, state, step_info=None):
        info = {
            "text_observation": state.text_observation,
            "valid_actions": state.valid_actions,
            "turn": state.turn,
            "game_metadata": state.metadata,
        }
        if step_info:
            info["step_info"] = step_info
        return info
