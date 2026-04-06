from __future__ import annotations

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
    Sprite,
)
from arcengine import GameState as EngineGameState
from arcengine.enums import BlockingMode
from arcengine.interfaces import RenderableUserDisplay

BG       = 5
BLUE     = 9
GREEN    = 14
MAGENTA  = 6
ORANGE   = 12
RED      = 8
YELLOW   = 11
WALL     = 0

CURSOR_C  = 3
CURSOR_ON = 10
TARGET_OK = 15
LIFE_ON   = 7
LIFE_OFF  = 4
LBOX      = 5
FLASH     = 13
GRAV_IND  = 4

WEIGHTS = {BLUE: 1, GREEN: 2, MAGENTA: 3, ORANGE: 4, RED: 5, YELLOW: 6}

GRAV_DOWN  = 0
GRAV_RIGHT = 1
GRAV_UP    = 2
GRAV_LEFT  = 3
GRAV_NAMES = ["DOWN", "RIGHT", "UP", "LEFT"]

CANVAS = 64
CELL_SIZE = 2

LEVELS = [
    {
        "grid_w": 18, "grid_h": 18,
        "blocks": [
            (5, 4, RED), (5, 5, YELLOW), (5, 6, BLUE),
            (5, 7, YELLOW), (5, 8, RED),
            (5, 10, BLUE), (5, 11, RED), (5, 12, YELLOW),
            (5, 13, BLUE), (5, 14, RED),
            (9, 4, YELLOW), (9, 5, RED), (9, 6, BLUE),
            (9, 7, RED), (9, 8, YELLOW),
            (9, 10, RED), (9, 11, BLUE), (9, 12, YELLOW),
            (9, 13, RED), (9, 14, BLUE),
            (13, 4, RED), (13, 5, BLUE), (13, 6, YELLOW),
            (13, 7, BLUE), (13, 8, RED),
            (13, 10, YELLOW), (13, 11, RED), (13, 12, BLUE),
            (13, 13, RED), (13, 14, YELLOW),
        ],
        "walls": [(x, 9) for x in range(3, 15)],
        "directions": [GRAV_DOWN],
        "cursor_start": (9, 8),
        "base": 40,
    },
    {
        "grid_w": 18, "grid_h": 18,
        "blocks": [
            (5, 3, MAGENTA), (5, 4, BLUE), (5, 5, RED),
            (5, 6, BLUE), (5, 7, MAGENTA),
            (12, 3, MAGENTA), (12, 4, BLUE), (12, 5, RED),
            (12, 6, BLUE), (12, 7, MAGENTA),
            (3, 11, GREEN),  (4, 11, RED),     (5, 11, BLUE),
            (6, 11, MAGENTA),(7, 11, GREEN),   (8, 11, RED),
            (9, 11, BLUE),   (10, 11, MAGENTA),(11, 11, RED),
            (12, 11, GREEN), (13, 11, BLUE),   (14, 11, MAGENTA),
        ],
        "walls": [],
        "directions": [GRAV_DOWN, GRAV_RIGHT],
        "cursor_start": (8, 6),
        "base": 50,
    },
    {
        "grid_w": 20, "grid_h": 18,
        "blocks": [
            (4, 3, RED), (4, 4, BLUE), (4, 5, YELLOW),
            (4, 6, GREEN), (4, 7, MAGENTA),
            (8, 3, YELLOW), (8, 4, MAGENTA), (8, 5, BLUE),
            (8, 6, RED), (8, 7, GREEN),
            (12, 3, GREEN), (12, 4, RED), (12, 5, MAGENTA),
            (12, 6, YELLOW), (12, 7, BLUE),
            (16, 3, MAGENTA), (16, 4, YELLOW), (16, 5, GREEN),
            (16, 6, BLUE), (16, 7, RED),
            (4, 12, GREEN),   (5, 12, RED),     (6, 12, BLUE),
            (7, 12, YELLOW),  (8, 12, MAGENTA), (9, 12, RED),
            (10, 12, BLUE),   (11, 12, MAGENTA),(12, 12, GREEN),
            (13, 12, YELLOW), (14, 12, RED),    (15, 12, BLUE),
            (4, 16, GREEN),   (5, 16, RED),     (6, 16, BLUE),
            (7, 16, YELLOW),  (8, 16, MAGENTA), (9, 16, RED),
            (10, 16, BLUE),   (11, 16, MAGENTA),(12, 16, GREEN),
            (13, 16, YELLOW), (14, 16, RED),    (15, 16, BLUE),
        ],
        "walls": [],
        "directions": [GRAV_DOWN, GRAV_RIGHT],
        "cursor_start": (10, 5),
        "base": 45,
    },
    {
        "grid_w": 19, "grid_h": 19,
        "blocks": [
            (4, 2, ORANGE), (4, 3, BLUE), (4, 4, YELLOW),
            (4, 5, GREEN), (4, 6, RED), (4, 7, MAGENTA),
            (10, 2, RED), (10, 3, MAGENTA), (10, 4, GREEN),
            (10, 5, ORANGE), (10, 6, BLUE), (10, 7, YELLOW),
            (16, 2, YELLOW), (16, 3, RED), (16, 4, MAGENTA),
            (16, 5, BLUE), (16, 6, ORANGE), (16, 7, GREEN),
            (3, 12, RED),    (4, 12, BLUE),    (5, 12, ORANGE),
            (6, 12, GREEN),  (7, 12, MAGENTA), (8, 12, YELLOW),
            (9, 12, ORANGE), (10, 12, MAGENTA),(11, 12, BLUE),
            (12, 12, RED),   (13, 12, GREEN),  (14, 12, YELLOW),
            (3, 16, MAGENTA),(4, 16, YELLOW),  (5, 16, BLUE),
            (6, 16, RED),    (7, 16, GREEN),   (8, 16, ORANGE),
            (9, 16, YELLOW), (10, 16, GREEN),  (11, 16, RED),
            (12, 16, ORANGE),(13, 16, BLUE),   (14, 16, MAGENTA),
        ],
        "walls": [],
        "directions": [GRAV_DOWN, GRAV_RIGHT],
        "cursor_start": (10, 5),
        "base": 75,
    },
]

CURSOR_START_POSITIONS = [
    [(9, 8), (5, 6), (13, 6), (9, 12)],
    [(8, 6), (5, 5), (12, 5), (8, 11)],
    [(10, 5), (4, 5), (12, 5), (10, 12)],
    [(10, 5), (4, 4), (16, 4), (10, 12)],
]

BASE_MOVES = [lv["base"] for lv in LEVELS]
MAX_MOVES  = [om * 6 for om in BASE_MOVES]


def _offset(grid_w, grid_h):
    return (CANVAS - grid_w * CELL_SIZE) // 2, (CANVAS - grid_h * CELL_SIZE) // 2



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


@dataclass
class GameState:
    text_observation: str
    image_observation: Optional[bytes]
    valid_actions: Optional[List[str]]
    turn: int
    metadata: Dict = field(default_factory=dict)


@dataclass
class StepResult:
    state: GameState
    reward: float
    done: bool
    info: Dict = field(default_factory=dict)


def _png_chunk(chunk_type, data):
    chunk = chunk_type + data
    return (
        struct.pack(">I", len(data))
        + chunk
        + struct.pack(">I", zlib.crc32(chunk) & 0xFFFFFFFF)
    )


def _encode_png(rgb):
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


class GameHUD(RenderableUserDisplay):
    def __init__(self):
        self.lives        = 3
        self.max_lives    = 3
        self.flash_frames = 0
        self.moves_left   = 0
        self.max_moves    = 1
        self.grav_dir     = GRAV_DOWN
        self.blocks_ok    = 0
        self.blocks_total = 1
        self.level_num    = 1
        self.ref_grid     = None
        self.ref_w        = 0
        self.ref_h        = 0
        self.ref_actual_gw = 0
        self.ref_actual_gh = 0

    @staticmethod
    def _px(frame, x, y, color):
        if 0 <= x < 64 and 0 <= y < 64:
            frame[y, x] = color

    def render_interface(self, frame):
        self._draw_lives(frame)
        self._draw_progress_bar(frame)
        self._draw_reference(frame)
        if self.flash_frames > 0:
            self._draw_flash(frame)
            self.flash_frames -= 1
        return frame

    def _draw_lives(self, frame):
        for i in range(self.max_lives):
            color = LIFE_ON if i < self.lives else LIFE_OFF
            x = 54 + i * 3
            self._px(frame, x, 0, color)
            self._px(frame, x + 1, 0, color)
            self._px(frame, x, 1, color)
            self._px(frame, x + 1, 1, color)

    def _draw_progress_bar(self, frame):
        if self.max_moves <= 0:
            return
        bar_w  = 60
        filled = int(bar_w * self.moves_left / self.max_moves)
        filled = max(0, min(bar_w, filled))
        for x in range(bar_w):
            px = 2 + x
            self._px(frame, px, 63, RED if x < filled else WALL)

    def _draw_reference(self, frame):
        if self.ref_grid is None:
            return
        gw, gh = self.ref_w, self.ref_h

        min_x, min_y = gw, gh
        max_x, max_y = 0, 0
        for y in range(1, gh - 1):
            for x in range(1, gw - 1):
                if self.ref_grid[y][x] in WEIGHTS:
                    min_x = min(min_x, x)
                    max_x = max(max_x, x)
                    min_y = min(min_y, y)
                    max_y = max(max_y, y)
        if max_x < min_x:
            return

        draw_w = max_x - min_x + 1
        draw_h = max_y - min_y + 1

        start_x = 1
        start_y = 2

        grid_ox = (CANVAS - self.ref_actual_gw * CELL_SIZE) // 2
        max_bx2 = grid_ox + CELL_SIZE

        max_content_w = max_bx2 - start_x
        if draw_w > max_content_w:
            draw_w = max(1, max_content_w)

        max_by2 = 62
        max_content_h = max_by2 - start_y
        if draw_h > max_content_h:
            draw_h = max(1, max_content_h)

        bx1, by1 = max(0, start_x - 1), start_y - 1
        bx2, by2 = start_x + draw_w, start_y + draw_h

        for y in range(by1, by2 + 1):
            for x in range(bx1, bx2 + 1):
                self._px(frame, x, y, BG)

        for x in range(bx1, bx2 + 1):
            self._px(frame, x, by1, WALL)
            self._px(frame, x, by2, WALL)
        for y in range(by1 + 1, by2):
            self._px(frame, bx1, y, WALL)
            self._px(frame, bx2, y, WALL)

        for dy in range(draw_h):
            for dx in range(draw_w):
                gy = min_y + dy
                gx = min_x + dx
                if gy <= max_y and gx <= max_x:
                    c = self.ref_grid[gy][gx]
                    if c in WEIGHTS:
                        self._px(frame, start_x + dx, start_y + dy, c)

    def _draw_flash(self, frame):
        for i in range(64):
            self._px(frame, i, 0, FLASH)
            self._px(frame, i, 63, FLASH)
            self._px(frame, 0, i, FLASH)
            self._px(frame, 63, i, FLASH)


class Nm03(ARCBaseGame):
    GW, GH = CANVAS, CANVAS

    def __init__(self, seed: int = 0):
        self._rng = random.Random(seed)
        self.hud = GameHUD()
        self.lives      = 3
        self.cursor_x   = 32
        self.cursor_y   = 32
        self.grid_w     = 10
        self.grid_h     = 10
        self.ox         = 0
        self.oy         = 0
        self.grav_dir   = GRAV_DOWN
        self.avail_dirs = [GRAV_DOWN]
        self.cells      = []
        self.target     = []
        self.wall_set   = set()
        self.undo_stack = []
        self.moves_left = 0
        self._consecutive_resets = 0
        self.board_sprite = None

        levels = [
            Level(sprites=[], grid_size=(self.GW, self.GH),
                  name=f"Level {i + 1}")
            for i in range(len(LEVELS))
        ]
        camera = Camera(
            x=0,
            y=0,
            width=self.GW,
            height=self.GH,
            background=5,
            letter_box=LBOX,
            interfaces=[self.hud],
        )
        super().__init__(
            "nm03", levels, camera, available_actions=[0, 1, 2, 3, 4, 5, 7]
        )

    def on_set_level(self, level):
        cfg = LEVELS[self.level_index]
        self.grid_w = cfg["grid_w"]
        self.grid_h = cfg["grid_h"]
        self.ox, self.oy = _offset(self.grid_w, self.grid_h)
        if self.level_index == 3:
            self.oy += 2

        cx, cy = self._rng.choice(CURSOR_START_POSITIONS[self.level_index])
        self.cursor_x = cx * CELL_SIZE + self.ox
        self.cursor_y = cy * CELL_SIZE + self.oy

        self.grav_dir   = cfg["directions"][0]
        self.avail_dirs = cfg["directions"]
        self.lives      = 3
        self.moves_left = MAX_MOVES[self.level_index]
        self.undo_stack = []

        self.cells  = [[BG] * self.grid_w for _ in range(self.grid_h)]
        self.target = [[BG] * self.grid_w for _ in range(self.grid_h)]

        for x in range(self.grid_w):
            self.cells[0][x] = WALL
            self.cells[self.grid_h - 1][x] = WALL
            self.target[0][x] = WALL
            self.target[self.grid_h - 1][x] = WALL
        for y in range(self.grid_h):
            self.cells[y][0] = WALL
            self.cells[y][self.grid_w - 1] = WALL
            self.target[y][0] = WALL
            self.target[y][self.grid_w - 1] = WALL

        self.wall_set = set()
        for wx, wy in cfg["walls"]:
            self.cells[wy][wx] = WALL
            self.target[wy][wx] = WALL
            self.wall_set.add((wx, wy))

        for bx, by, bc in cfg["blocks"]:
            self.cells[by][bx] = bc
        self._compute_targets(cfg["blocks"])

        self.board_sprite = self._make_board_sprite()
        level.add_sprite(self.board_sprite)
        self._sync_hud()

    def _compute_targets(self, blocks):
        block_map = {(bx, by): bc for bx, by, bc in blocks}
        self.block_segment_dir = {}

        by_col = {}
        by_row = {}
        for bx, by, _bc in blocks:
            by_col.setdefault(bx, []).append(by)
            by_row.setdefault(by, []).append(bx)

        col_groups = {x for x, ys in by_col.items() if len(ys) >= 2}
        row_groups = {y for y, xs in by_row.items() if len(xs) >= 2}
        assigned = set()

        for x in sorted(col_groups):
            segments, seg = [], []
            for y in range(1, self.grid_h - 1):
                if self.cells[y][x] == WALL:
                    if seg:
                        segments.append(seg)
                        seg = []
                elif (x, y) in block_map:
                    seg.append(y)
                else:
                    if seg:
                        segments.append(seg)
                        seg = []
            if seg:
                segments.append(seg)

            for seg in segments:
                if len(seg) < 2:
                    continue
                colors = sorted(
                    [block_map[(x, y)] for y in seg],
                    key=lambda c: WEIGHTS[c],
                )
                for i, y in enumerate(seg):
                    self.target[y][x] = colors[i]
                    assigned.add((x, y))
                    self.block_segment_dir[(x, y)] = GRAV_DOWN

        for y in sorted(row_groups):
            segments, seg = [], []
            for x in range(1, self.grid_w - 1):
                if self.cells[y][x] == WALL:
                    if seg:
                        segments.append(seg)
                        seg = []
                elif (x, y) in block_map and (x, y) not in assigned:
                    seg.append(x)
                else:
                    if seg:
                        segments.append(seg)
                        seg = []
            if seg:
                segments.append(seg)

            for seg in segments:
                if len(seg) < 2:
                    continue
                colors = sorted(
                    [block_map[(x, y)] for x in seg],
                    key=lambda c: WEIGHTS[c],
                )
                for i, x in enumerate(seg):
                    self.target[y][x] = colors[i]
                    assigned.add((x, y))
                    self.block_segment_dir[(x, y)] = GRAV_RIGHT

        for bx, by, bc in blocks:
            if (bx, by) not in assigned:
                self.target[by][bx] = bc

    def _save_state(self):
        return {
            "cells": [row[:] for row in self.cells],
            "grav_dir": self.grav_dir,
            "cursor_x": self.cursor_x,
            "cursor_y": self.cursor_y,
        }

    def _restore_state(self, state):
        self.cells = [row[:] for row in state["cells"]]
        self.grav_dir = state["grav_dir"]
        self.cursor_x = state["cursor_x"]
        self.cursor_y = state["cursor_y"]

    def handle_reset(self):
        if self._consecutive_resets > 0:
            self._full_game_reset()
        else:
            self._consecutive_resets += 1
            super().handle_reset()

    def _level_reset(self):
        saved_lives = self.lives
        self.set_level(self.level_index)
        self.lives = saved_lives
        self.hud.lives = saved_lives

    def _full_game_reset(self):
        self._consecutive_resets = 0
        self.lives = 3
        self.set_level(0)
        self._state = EngineGameState.NOT_FINISHED

    def step(self):
        self._consecutive_resets = 0
        aid = self.action.id
        level_before = self.level_index

        x_min = self.ox + 1 * CELL_SIZE
        x_max = self.ox + (self.grid_w - 2) * CELL_SIZE
        y_min = self.oy + 1 * CELL_SIZE
        y_max = self.oy + (self.grid_h - 2) * CELL_SIZE

        if aid == GameAction.ACTION1:
            self.cursor_y = max(y_min, self.cursor_y - CELL_SIZE)
        elif aid == GameAction.ACTION2:
            self.cursor_y = min(y_max, self.cursor_y + CELL_SIZE)
        elif aid == GameAction.ACTION3:
            self.cursor_x = max(x_min, self.cursor_x - CELL_SIZE)
        elif aid == GameAction.ACTION4:
            self.cursor_x = min(x_max, self.cursor_x + CELL_SIZE)
        elif aid == GameAction.ACTION5:
            self._do_swap()
        elif aid == GameAction.ACTION7:
            self._cycle_or_undo()

        if self.level_index == level_before and self.lives > 0:
            self.moves_left -= 1
            if self.moves_left <= 0:
                self._moves_exhausted()

        if self.level_index == level_before:
            self._rebuild_board()
            self._sync_hud()

        self.complete_action()

    def _do_swap(self):
        local_cx = (self.cursor_x - self.ox) // CELL_SIZE
        local_cy = (self.cursor_y - self.oy) // CELL_SIZE

        if not (0 < local_cx < self.grid_w - 1 and
                0 < local_cy < self.grid_h - 1):
            return
        cell = self.cells[local_cy][local_cx]
        if cell not in WEIGHTS:
            return

        swap_dir = self.block_segment_dir.get(
            (local_cx, local_cy), self.grav_dir
        )

        dx, dy = 0, 0
        if   swap_dir == GRAV_DOWN:  dy =  1
        elif swap_dir == GRAV_UP:    dy = -1
        elif swap_dir == GRAV_RIGHT: dx =  1
        elif swap_dir == GRAV_LEFT:  dx = -1

        nx, ny = local_cx + dx, local_cy + dy

        if not (0 < nx < self.grid_w - 1 and 0 < ny < self.grid_h - 1):
            return
        neighbour = self.cells[ny][nx]
        if neighbour not in WEIGHTS:
            return
        if cell == neighbour:
            return

        snapshot  = self._save_state()

        self.cells[local_cy][local_cx] = neighbour
        self.cells[ny][nx]             = cell

        self._apply_level_rule(local_cx, local_cy, nx, ny)

        self.cursor_x += dx * CELL_SIZE
        self.cursor_y += dy * CELL_SIZE

        self.undo_stack.append(snapshot)

        if self._check_win():
            self.next_level()

    def _apply_level_rule(self, cx, cy, nx, ny):
        li = self.level_index
        if li == 1:
            self._rule_mirror_column(cx, cy, nx, ny)
        elif li == 2:
            self._rule_mirror_rows(cx, cy, nx, ny)
        elif li == 3:
            self._rule_ripple(cx, cy, nx, ny)

    def _rule_mirror_column(self, cx, cy, nx, ny):
        if cx != nx:
            return
        mirror_x = self.grid_w - 1 - cx
        if mirror_x == cx:
            return
        if not (0 < mirror_x < self.grid_w - 1):
            return
        a = self.cells[cy][mirror_x]
        b = self.cells[ny][mirror_x]
        if a in WEIGHTS and b in WEIGHTS and a != b:
            self.cells[cy][mirror_x] = b
            self.cells[ny][mirror_x] = a

    def _rule_mirror_rows(self, cx, cy, nx, ny):
        if cy != ny:
            return
        row_a = 12
        row_b = 16
        if cy == row_a:
            mirror_y = row_b
        elif cy == row_b:
            mirror_y = row_a
        else:
            return
        if not (0 < mirror_y < self.grid_h - 1):
            return
        a = self.cells[mirror_y][cx]
        b = self.cells[mirror_y][nx]
        if a in WEIGHTS and b in WEIGHTS and a != b:
            self.cells[mirror_y][cx] = b
            self.cells[mirror_y][nx] = a

    def _rule_ripple(self, cx, cy, nx, ny):
        dx = nx - cx
        dy = ny - cy
        rx1 = nx + dx
        ry1 = ny + dy
        rx2 = rx1 + dx
        ry2 = ry1 + dy
        if not (0 < rx1 < self.grid_w - 1 and 0 < ry1 < self.grid_h - 1):
            return
        if not (0 < rx2 < self.grid_w - 1 and 0 < ry2 < self.grid_h - 1):
            return
        a = self.cells[ry1][rx1]
        b = self.cells[ry2][rx2]
        if a in WEIGHTS and b in WEIGHTS and a != b:
            self.cells[ry1][rx1] = b
            self.cells[ry2][rx2] = a

    def _cycle_or_undo(self):
        self._undo()

    def _undo(self):
        if not self.undo_stack:
            return
        self._restore_state(self.undo_stack.pop())

    def _check_win(self):
        for y in range(1, self.grid_h - 1):
            for x in range(1, self.grid_w - 1):
                t = self.target[y][x]
                if t != BG and t != WALL:
                    if self.cells[y][x] != t:
                        return False
        return True

    def _moves_exhausted(self):
        self.hud.flash_frames = 3
        self.lives -= 1
        if self.lives <= 0:
            self.lose()
            return
        self._level_reset()

    def _make_board_sprite(self):
        pixels = self._render_grid()
        return Sprite(
            pixels=pixels, name="board", x=0, y=0, layer=0,
            blocking=BlockingMode.NOT_BLOCKED, collidable=False,
        )

    def _render_grid(self):
        pixels = []
        for py in range(self.GH):
            row = []
            for px in range(self.GW):
                row.append(self._pixel_at(px, py))
            pixels.append(row)
        return pixels

    def _pixel_at(self, px, py):
        rx = px - self.ox
        ry = py - self.oy
        scaled_w = self.grid_w * CELL_SIZE
        scaled_h = self.grid_h * CELL_SIZE
        in_grid = (0 <= rx < scaled_w and 0 <= ry < scaled_h)

        if (px == 1 or px == 62) and 3 <= py <= 58:
            return BG
        if (py == 3 or py == 58) and 1 <= px <= 62:
            return BG

        if in_grid:
            lx = rx // CELL_SIZE
            ly = ry // CELL_SIZE
        else:
            lx = ly = -1

        cur_lx = (self.cursor_x - self.ox) // CELL_SIZE
        cur_ly = (self.cursor_y - self.oy) // CELL_SIZE

        if in_grid and lx == cur_lx and ly == cur_ly:
            c = self.cells[ly][lx]
            t = self.target[ly][lx]
            if c != BG and c != WALL:
                return TARGET_OK if c == t else CURSOR_ON
            return CURSOR_C

        if not in_grid:
            return 5

        cell = self.cells[ly][lx]
        tgt  = self.target[ly][lx]

        if cell == WALL:
            on_border = (lx == 0 or lx == self.grid_w - 1 or
                         ly == 0 or ly == self.grid_h - 1)
            return WALL if on_border else CURSOR_C

        if cell != BG and cell in WEIGHTS:
            return cell

        if tgt != BG and tgt != WALL:
            return tgt

        return BG

    def _rebuild_board(self):
        if self.board_sprite is None:
            return
        for py in range(self.GH):
            for px in range(self.GW):
                self.board_sprite.pixels[py, px] = self._pixel_at(px, py)

    def _sync_hud(self):
        self.hud.lives      = self.lives
        self.hud.max_lives   = 3
        self.hud.level_num   = self.level_index + 1
        self.hud.moves_left  = self.moves_left
        self.hud.max_moves   = MAX_MOVES[self.level_index]
        self.hud.grav_dir    = self.grav_dir

        ref_grid, ref_w, ref_h = self._shrink_ref_for_hud()
        self.hud.ref_grid = ref_grid
        self.hud.ref_w    = ref_w
        self.hud.ref_h    = ref_h
        self.hud.ref_actual_gw = self.grid_w
        self.hud.ref_actual_gh = self.grid_h

        ok = 0
        total = 0
        for y in range(1, self.grid_h - 1):
            for x in range(1, self.grid_w - 1):
                t = self.target[y][x]
                if t != BG and t != WALL:
                    total += 1
                    if self.cells[y][x] == t:
                        ok += 1
        self.hud.blocks_ok    = ok
        self.hud.blocks_total = max(total, 1)

    def _compact_ref_for_hud(self):
        block_positions = []
        for y in range(1, self.grid_h - 1):
            for x in range(1, self.grid_w - 1):
                if self.target[y][x] in WEIGHTS:
                    block_positions.append((x, y))
        xs = sorted(set(p[0] for p in block_positions))
        ys = sorted(set(p[1] for p in block_positions))

        row_ys = set()
        for y_val in ys:
            count = sum(1 for bx, by in block_positions if by == y_val)
            if count >= 6:
                row_ys.add(y_val)

        sorted_row_ys = sorted(row_ys)
        y_map = {}
        idx = 1
        for y_val in ys:
            y_map[y_val] = idx
            idx += 1
            if len(sorted_row_ys) == 2 and y_val == sorted_row_ys[0]:
                idx += 1

        x_map = {v: i + 1 for i, v in enumerate(xs)}
        cw = len(xs) + 2
        ch = idx + 1
        grid = [[BG] * cw for _ in range(ch)]
        for x in range(cw):
            grid[0][x] = WALL
            grid[ch - 1][x] = WALL
        for y in range(ch):
            grid[y][0] = WALL
            grid[y][cw - 1] = WALL
        for bx, by in block_positions:
            grid[y_map[by]][x_map[bx]] = self.target[by][bx]
        return grid, cw, ch

    def _shrink_ref_for_hud(self):
        block_rows = set()
        block_cols = set()
        for y in range(1, self.grid_h - 1):
            for x in range(1, self.grid_w - 1):
                if self.target[y][x] in WEIGHTS:
                    block_rows.add(y)
                    block_cols.add(x)
        if not block_rows:
            return self.target, self.grid_w, self.grid_h

        sorted_rows = sorted(block_rows)
        row_map = {}
        gap_rows = []
        ny = 1
        for i, y in enumerate(sorted_rows):
            if i > 0 and y > sorted_rows[i - 1] + 1:
                gap_rows.append(ny)
                ny += 1
            row_map[y] = ny
            ny += 1

        sorted_cols = sorted(block_cols)
        col_map = {}
        nx = 1
        for i, x in enumerate(sorted_cols):
            if i > 0 and x > sorted_cols[i - 1] + 2:
                nx += 1
            col_map[x] = nx
            nx += 1

        cw = nx + 1
        ch = ny + 1

        grid = [[BG] * cw for _ in range(ch)]
        for x in range(cw):
            grid[0][x] = WALL
            grid[ch - 1][x] = WALL
        for y in range(ch):
            grid[y][0] = WALL
            grid[y][cw - 1] = WALL
        for y in sorted_rows:
            for x in sorted_cols:
                c = self.target[y][x]
                if c in WEIGHTS:
                    mapped_y = row_map[y]
                    mapped_x = col_map[x]
                    if mapped_y < ch - 1 and mapped_x < cw - 1:
                        grid[mapped_y][mapped_x] = c
        return grid, cw, ch


class PuzzleEnvironment:
    _ACTION_MAP: Dict[str, GameAction] = {
        "up": GameAction.ACTION1,
        "down": GameAction.ACTION2,
        "left": GameAction.ACTION3,
        "right": GameAction.ACTION4,
        "select": GameAction.ACTION5,
        "undo": GameAction.ACTION7,
    }

    _VALID_ACTIONS: List[str] = ["reset", "up", "down", "left", "right", "select", "undo"]

    def __init__(self, seed: int = 0) -> None:
        self._engine = Nm03(seed=seed)
        self._seed = seed
        self._total_turns = 0
        self._done = False
        self._last_action_was_reset = True
        self._game_over = False

    def reset(self) -> GameState:
        if self._is_won() or self._last_action_was_reset:
            self._engine._full_game_reset()
        else:
            self._engine.perform_action(ActionInput(id=GameAction.RESET))
        self._last_action_was_reset = True
        self._done = False
        self._total_turns = 0
        self._game_over = False
        return self._build_state()

    def get_actions(self) -> List[str]:
        if self._done or self._game_over or self._is_game_over():
            return ["reset"]
        return list(self._VALID_ACTIONS)

    def step(self, action: str) -> StepResult:
        if action == "reset":
            state = self.reset()
            return StepResult(state=state, reward=0.0, done=False)

        self._last_action_was_reset = False

        if action not in self._ACTION_MAP:
            return StepResult(
                state=self._build_state(),
                reward=0.0,
                done=self._done,
                info={"error": f"Invalid action: {action}"},
            )

        lives_before = self._engine.lives
        level_before = self._engine.level_index

        frame = self._engine.perform_action(ActionInput(id=self._ACTION_MAP[action]), raw=True)
        self._total_turns += 1

        state_name = frame.state.name if frame and frame.state else ""
        game_won = state_name == "WIN"
        game_over = state_name == "GAME_OVER"

        reward = 0.0
        done = False
        info: dict = {
            "lives": self._engine.lives,
            "level": self._engine.level_index + 1,
            "moves_left": self._engine.moves_left,
        }

        total_levels = len(self._engine._levels)
        level_reward_step = 1.0 / total_levels

        if game_over:
            reward = 0.0
            info["event"] = "game_over"
            done = True
            self._game_over = True
        elif self._engine.lives < lives_before:
            reward = 0.0
            info["event"] = "life_lost"
        elif game_won:
            reward = level_reward_step
            info["event"] = "game_complete"
            done = True
        elif self._engine.level_index != level_before:
            reward = level_reward_step
            info["event"] = "level_complete"

        self._done = done

        return StepResult(
            state=self._build_state(),
            reward=reward,
            done=done,
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
        return rgb

    def close(self) -> None:
        self._engine = None

    def is_done(self) -> bool:
        return self._is_won() or self._game_over or self._is_game_over()

    def _is_game_over(self) -> bool:
        try:
            return self._engine._state == EngineGameState.GAME_OVER
        except Exception:
            return False

    def _is_won(self) -> bool:
        try:
            return self._engine._state == EngineGameState.WIN
        except Exception:
            return False

    def _build_text_observation(self) -> str:
        lines = []
        lines.append(
            f"Level {self._engine.level_index + 1}/{len(self._engine._levels)}"
            f" | Lives: {self._engine.lives}"
            f" | Moves: {self._engine.moves_left}/{MAX_MOVES[self._engine.level_index]}"
            f" | Turn: {self._total_turns}"
        )
        lines.append("")

        g = self._engine
        cursor_gx = (g.cursor_x - g.ox) // CELL_SIZE
        cursor_gy = (g.cursor_y - g.oy) // CELL_SIZE

        lines.append(
            f"Blocks correct: {g.hud.blocks_ok}/{g.hud.blocks_total}"
        )
        lines.append(
            f"Gravity: {GRAV_NAMES[g.grav_dir]}"
            f" | Dirs: {', '.join(GRAV_NAMES[d] for d in g.avail_dirs)}"
        )
        lines.append("")

        lines.append("Grid (. = empty, W = wall, * = target hint):")
        for y in range(1, g.grid_h - 1):
            row_str = ""
            for x in range(1, g.grid_w - 1):
                c = g.cells[y][x]
                t = g.target[y][x]
                if x == cursor_gx and y == cursor_gy:
                    row_str += "@"
                elif c == WALL:
                    row_str += "W"
                elif c in WEIGHTS:
                    row_str += str(WEIGHTS[c])
                elif t in WEIGHTS:
                    row_str += "*"
                else:
                    row_str += "."
            lines.append(f"  {row_str}")

        lines.append("")
        lines.append(f"Cursor: ({cursor_gx}, {cursor_gy})")
        lines.append("Weight key: Blue=1  Green=2  Magenta=3  Orange=4  Red=5  Yellow=6")
        lines.append(f"Actions: {', '.join(self._VALID_ACTIONS)}")

        return "\n".join(lines)

    def _build_state(self) -> GameState:
        valid = self.get_actions()

        image_bytes = None
        try:
            rgb = self.render()
            image_bytes = _encode_png(rgb)
        except Exception:
            pass

        return GameState(
            text_observation=self._build_text_observation(),
            image_observation=image_bytes,
            valid_actions=valid,
            turn=self._total_turns,
            metadata={
                "total_levels": len(self._engine._levels),
                "level_index": self._engine.level_index,
                "levels_completed": self._engine.level_index,
                "game_over": self._game_over or self._is_game_over(),
                "done": self._done,
                "info": {
                    "lives": self._engine.lives,
                    "blocks_ok": self._engine.hud.blocks_ok,
                    "blocks_total": self._engine.hud.blocks_total,
                    "gravity": GRAV_NAMES[self._engine.grav_dir],
                    "moves_left": self._engine.moves_left,
                },
            },
        )


class ArcGameEnv(gym.Env):
    metadata: Dict[str, Any] = {
        "render_modes": ["rgb_array"],
        "render_fps": 5,
    }

    ACTION_LIST: List[str] = ["reset", "up", "down", "left", "right", "select", "undo"]

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