from __future__ import annotations

import io
import random
import struct
import zlib
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

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


@dataclass
class _SimpleAction:
    id: int
    data: dict


def _encode_png(frame: np.ndarray) -> bytes:
    h, w = frame.shape[:2]
    if frame.ndim == 2:
        raw = np.zeros((h, w, 3), dtype=np.uint8)
        raw[:] = frame[:, :, None]
    else:
        raw = frame.astype(np.uint8)
    rows = []
    for y in range(h):
        rows.append(b"\x00" + raw[y].tobytes())
    raw_data = b"".join(rows)
    compressed = zlib.compress(raw_data)

    def _chunk(ctype: bytes, data: bytes) -> bytes:
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


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
ORANGE = 7
BROWN = 11

BG_COLOR = BLACK
GAP_COLOR = 5
GAP_ACCENT = 9
SELECTOR_COLOR = 11

TILE_SIZE = 2
CAM_SIZE = 32
MAX_LIVES = 3

PALETTE_2 = [BLUE, RED]
PALETTE_3 = [BLUE, RED, GREEN]
PALETTE_4 = [BLUE, RED, GREEN, ORANGE]


def _make_solid_tile(color: int, name: str, tag: str) -> Sprite:
    return Sprite(
        pixels=[
            [color, color],
            [color, color],
        ],
        name=name,
        visible=True,
        collidable=False,
        layer=0,
        tags=[tag],
    )


def _make_gap_tile() -> Sprite:
    return Sprite(
        pixels=[
            [GAP_COLOR, GAP_ACCENT],
            [GAP_ACCENT, GAP_COLOR],
        ],
        name="gap",
        visible=True,
        collidable=False,
        layer=0,
        tags=["gap"],
    )


def _make_selector() -> Sprite:
    return Sprite(
        pixels=[
            [SELECTOR_COLOR, SELECTOR_COLOR],
            [SELECTOR_COLOR, SELECTOR_COLOR],
        ],
        name="selector",
        visible=False,
        collidable=False,
        layer=3,
        tags=["selector"],
    )


def _make_palette_indicator(color: int) -> Sprite:
    return Sprite(
        pixels=[
            [color, color],
            [color, color],
        ],
        name="pal_ind",
        visible=True,
        collidable=False,
        layer=1,
        tags=["palette_ind"],
    )


def _make_current_color_indicator() -> Sprite:
    return Sprite(
        pixels=[
            [BROWN, BROWN, BROWN, BROWN],
            [BROWN, BLUE, BLUE, BROWN],
            [BROWN, BLUE, BLUE, BROWN],
            [BROWN, BROWN, BROWN, BROWN],
        ],
        name="current_color",
        visible=True,
        collidable=False,
        layer=2,
        tags=["current_color"],
    )


def _make_life_indicator() -> Sprite:
    return Sprite(
        pixels=[
            [RED, RED],
            [RED, RED],
        ],
        name="life",
        visible=True,
        collidable=False,
        layer=2,
        tags=["life"],
    )


def _grid_offset(grid_w: int, grid_h: int) -> tuple[int, int]:
    ox = (CAM_SIZE - grid_w * TILE_SIZE) // 2
    oy = (CAM_SIZE - grid_h * TILE_SIZE) // 2
    return (ox, oy)


def _tile_pos(gx: int, gy: int, ox: int, oy: int) -> tuple[int, int]:
    return (ox + gx * TILE_SIZE, oy + gy * TILE_SIZE)


L1_W, L1_H = 6, 6
L1_PALETTE = PALETTE_2
L1_PATTERN = [
    [0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1],
]
L1_GAPS = {(2, 1), (4, 4)}

L2_W, L2_H = 6, 6
L2_PALETTE = PALETTE_3
L2_PATTERN = [
    [0, 1, 2, 0, 1, 2],
    [0, 1, 2, 0, 1, 2],
    [0, 1, 2, 0, 1, 2],
    [0, 1, 2, 0, 1, 2],
    [0, 1, 2, 0, 1, 2],
    [0, 1, 2, 0, 1, 2],
]
L2_GAPS = {(1, 0), (2, 2), (0, 4), (1, 5)}

L3_W, L3_H = 8, 8
L3_PALETTE = PALETTE_3
L3_PATTERN = [[(x + y) % 3 for x in range(8)] for y in range(8)]
L3_GAPS = {(1, 1), (3, 2), (5, 3), (2, 5), (6, 4), (7, 7)}


def _diagonal_pattern(w: int, h: int) -> list[list[int]]:
    grid = []
    for y in range(h):
        row = []
        for x in range(w):
            row_shift = y // 2
            row.append((x + y + row_shift) % 4)
        grid.append(row)
    return grid


L4_W, L4_H = 8, 8
L4_PALETTE = PALETTE_4
L4_PATTERN = _diagonal_pattern(8, 8)
L4_GAPS = {(0, 3), (2, 2), (5, 1), (7, 4), (3, 5), (6, 6), (1, 7), (4, 0)}


def _mirror_pattern(w: int, h: int) -> list[list[int]]:
    grid = []
    mid_x = w // 2
    mid_y = h // 2
    for y in range(h):
        row = []
        for x in range(w):
            if x < mid_x and y < mid_y:
                base = y % 4
            elif x >= mid_x and y < mid_y:
                base = x % 4
            elif x < mid_x and y >= mid_y:
                base = (w - 1 - x) % 4
            else:
                base = (h - 1 - y) % 4

            on_border = x == mid_x - 1 or x == mid_x or y == mid_y - 1 or y == mid_y
            if on_border:
                base = (base + 1) % 4

            row.append(base)
        grid.append(row)
    return grid


L5_W, L5_H = 10, 10
L5_PALETTE = PALETTE_4
L5_PATTERN = _mirror_pattern(10, 10)
L5_GAPS = {
    (1, 1),
    (3, 3),
    (0, 6),
    (4, 8),
    (2, 4),
    (6, 0),
    (8, 2),
    (5, 5),
    (7, 7),
    (9, 9),
}


def _build_level_sprites(
    grid_w: int,
    grid_h: int,
    pattern: list[list[int]],
    palette: list[int],
    gaps: set[tuple[int, int]],
) -> list[Sprite]:
    ox, oy = _grid_offset(grid_w, grid_h)
    level_sprites: list[Sprite] = []

    for gy in range(grid_h):
        for gx in range(grid_w):
            px, py = _tile_pos(gx, gy, ox, oy)
            if (gx, gy) in gaps:
                s = _make_gap_tile().clone().set_position(px, py)
                level_sprites.append(s)
            else:
                cidx = pattern[gy][gx]
                color = palette[cidx]
                s = _make_solid_tile(color, "tile", "tile").clone().set_position(px, py)
                level_sprites.append(s)

    sel = _make_selector().clone().set_position(0, 0)
    level_sprites.append(sel)

    for i, c in enumerate(palette):
        ind = _make_palette_indicator(c).clone().set_position(i * 3, 0)
        level_sprites.append(ind)

    cci = _make_current_color_indicator().clone().set_position(0, CAM_SIZE - 4)
    level_sprites.append(cci)

    for i in range(MAX_LIVES):
        life_x = CAM_SIZE - (MAX_LIVES - i) * 3
        life_y = 0
        life_s = _make_life_indicator().clone().set_position(life_x, life_y)
        level_sprites.append(life_s)

    return level_sprites


LEVELS_DATA = [
    Level(
        sprites=_build_level_sprites(L1_W, L1_H, L1_PATTERN, L1_PALETTE, L1_GAPS),
        grid_size=(CAM_SIZE, CAM_SIZE),
        data={
            "grid_w": L1_W,
            "grid_h": L1_H,
            "palette": L1_PALETTE,
            "pattern": L1_PATTERN,
            "gaps": [[gx, gy] for gx, gy in L1_GAPS],
            "num_colors": len(L1_PALETTE),
            "max_moves": int(len(L1_GAPS) * len(L1_PALETTE) * 3 * 1.1),
        },
        name="Level 1",
    ),
    Level(
        sprites=_build_level_sprites(L2_W, L2_H, L2_PATTERN, L2_PALETTE, L2_GAPS),
        grid_size=(CAM_SIZE, CAM_SIZE),
        data={
            "grid_w": L2_W,
            "grid_h": L2_H,
            "palette": L2_PALETTE,
            "pattern": L2_PATTERN,
            "gaps": [[gx, gy] for gx, gy in L2_GAPS],
            "num_colors": len(L2_PALETTE),
            "max_moves": int(len(L2_GAPS) * len(L2_PALETTE) * 3 * 1.1),
        },
        name="Level 2",
    ),
    Level(
        sprites=_build_level_sprites(L3_W, L3_H, L3_PATTERN, L3_PALETTE, L3_GAPS),
        grid_size=(CAM_SIZE, CAM_SIZE),
        data={
            "grid_w": L3_W,
            "grid_h": L3_H,
            "palette": L3_PALETTE,
            "pattern": L3_PATTERN,
            "gaps": [[gx, gy] for gx, gy in L3_GAPS],
            "num_colors": len(L3_PALETTE),
            "max_moves": int(len(L3_GAPS) * len(L3_PALETTE) * 3 * 1.1),
        },
        name="Level 3",
    ),
    Level(
        sprites=_build_level_sprites(L4_W, L4_H, L4_PATTERN, L4_PALETTE, L4_GAPS),
        grid_size=(CAM_SIZE, CAM_SIZE),
        data={
            "grid_w": L4_W,
            "grid_h": L4_H,
            "palette": L4_PALETTE,
            "pattern": L4_PATTERN,
            "gaps": [[gx, gy] for gx, gy in L4_GAPS],
            "num_colors": len(L4_PALETTE),
            "max_moves": int(len(L4_GAPS) * len(L4_PALETTE) * 3 * 1.1),
        },
        name="Level 4",
    ),
    Level(
        sprites=_build_level_sprites(L5_W, L5_H, L5_PATTERN, L5_PALETTE, L5_GAPS),
        grid_size=(CAM_SIZE, CAM_SIZE),
        data={
            "grid_w": L5_W,
            "grid_h": L5_H,
            "palette": L5_PALETTE,
            "pattern": L5_PATTERN,
            "gaps": [[gx, gy] for gx, gy in L5_GAPS],
            "num_colors": len(L5_PALETTE),
            "max_moves": int(len(L5_GAPS) * len(L5_PALETTE) * 3 * 1.1),
        },
        name="Level 5",
    ),
]


BAR_CORRECT_BG = 0
BAR_CORRECT = 3
BAR_WIN = 14
BAR_MOVES_FULL = 8
BAR_MOVES_LOW = 2
BAR_MOVES_EMPTY = 0


class CorrectFillBar(RenderableUserDisplay):
    def __init__(self, game: "Cp01") -> None:
        self._game = game

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        game = self._game
        total = len(game._gap_set) if hasattr(game, "_gap_set") and game._gap_set else 1

        correct = 0
        if hasattr(game, "_gap_colors") and hasattr(game, "_pattern"):
            for (gx, gy), cidx in game._gap_colors.items():
                if cidx != -1 and cidx == game._pattern[gy][gx]:
                    correct += 1

        fill_color = BAR_WIN if correct == total else BAR_CORRECT
        fill_px = int(round(correct / total * 64))

        frame[62, :] = BAR_CORRECT_BG
        frame[63, :] = BAR_CORRECT_BG
        if fill_px > 0:
            frame[62, :fill_px] = fill_color
            frame[63, :fill_px] = fill_color

        return frame


class MoveBudgetBar(RenderableUserDisplay):
    def __init__(self, game: "Cp01") -> None:
        self._game = game

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        game = self._game
        max_moves = getattr(game, "_max_moves", 1)
        moves_left = getattr(game, "_moves_left", 0)
        remaining = max(0, moves_left)

        fill_px = int(round(remaining / max_moves * 64)) if max_moves > 0 else 0

        bar_color = BAR_MOVES_LOW if remaining <= max_moves * 0.25 else BAR_MOVES_FULL

        frame[0, :] = BAR_MOVES_EMPTY
        frame[1, :] = BAR_MOVES_EMPTY
        if fill_px > 0:
            frame[0, :fill_px] = bar_color
            frame[1, :fill_px] = bar_color

        return frame


class Cp01(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)
        self._progress_bar = CorrectFillBar(self)
        self._move_bar = MoveBudgetBar(self)

        super().__init__(
            "cp01",
            LEVELS_DATA,
            Camera(
                x=0,
                y=0,
                width=CAM_SIZE,
                height=CAM_SIZE,
                background=BG_COLOR,
                letter_box=BG_COLOR,
                interfaces=[self._move_bar, self._progress_bar],
            ),
            available_actions=[0, 1, 2, 3, 4, 5, 6, 7],
        )

    def get_actions(self):
        return [1, 2, 3, 4, 5, 6, 7]

    def on_set_level(self, level: Level) -> None:
        self._grid_w: int = level.get_data("grid_w")
        self._grid_h: int = level.get_data("grid_h")
        self._palette: list[int] = level.get_data("palette")
        self._num_colors: int = level.get_data("num_colors")
        self._pattern: list[list[int]] = level.get_data("pattern")
        self._ox, self._oy = _grid_offset(self._grid_w, self._grid_h)

        self._moves_left: int = level.get_data("max_moves")
        self._max_moves: int = self._moves_left
        self._level_done: bool = False

        original_gap_data = level.get_data("gaps")
        num_gaps = len(original_gap_data)

        all_cells = [(x, y) for x in range(self._grid_w) for y in range(self._grid_h)]
        new_gaps = self._rng.sample(all_cells, num_gaps)

        self._gap_set: set[tuple[int, int]] = set(new_gaps)

        old_gap_sprites = level.get_sprites_by_tag("gap")
        old_tile_sprites = level.get_sprites_by_tag("tile")
        for s in old_gap_sprites:
            level.remove_sprite(s)
        for s in old_tile_sprites:
            level.remove_sprite(s)

        for gy in range(self._grid_h):
            for gx in range(self._grid_w):
                px, py = _tile_pos(gx, gy, self._ox, self._oy)
                if (gx, gy) in self._gap_set:
                    s = _make_gap_tile().clone().set_position(px, py)
                    level.add_sprite(s)
                else:
                    cidx = self._pattern[gy][gx]
                    color = self._palette[cidx]
                    s = (
                        _make_solid_tile(color, "tile", "tile")
                        .clone()
                        .set_position(px, py)
                    )
                    level.add_sprite(s)

        self._gap_colors: dict[tuple[int, int], int] = {}
        for gx, gy in self._gap_set:
            self._gap_colors[(gx, gy)] = -1

        self._selected_gap: tuple[int, int] | None = None

        self._gap_sprites: dict[tuple[int, int], Sprite] = {}
        gap_sprite_list = level.get_sprites_by_tag("gap")

        for s in gap_sprite_list:
            gx = (s.x - self._ox) // TILE_SIZE
            gy = (s.y - self._oy) // TILE_SIZE
            if (gx, gy) in self._gap_set:
                self._gap_sprites[(gx, gy)] = s

        sel_list = level.get_sprites_by_tag("selector")
        self._selector_sprite = sel_list[0] if sel_list else None

        cci_list = level.get_sprites_by_tag("current_color")
        self._current_color_ind = cci_list[0] if cci_list else None

        self._pal_indicators = level.get_sprites_by_tag("palette_ind")

        self._lives: int = MAX_LIVES
        self._game_over: bool = False
        self._life_sprites: list[Sprite] = level.get_sprites_by_tag("life")
        self._can_undo: bool = False
        self._undo_stack: list[
            tuple[dict[tuple[int, int], int], tuple[int, int] | None, int]
        ] = []
        self._update_life_display()
        self._update_color_indicator(-1)

    def _update_gap_visual(self, gx: int, gy: int) -> None:
        s = self._gap_sprites.get((gx, gy))
        if s is None:
            return

        cidx = self._gap_colors.get((gx, gy), -1)
        if cidx == -1:
            px, py = _tile_pos(gx, gy, self._ox, self._oy)
            self.current_level.remove_sprite(s)
            new_s = _make_gap_tile().clone().set_position(px, py)
            self.current_level.add_sprite(new_s)
            self._gap_sprites[(gx, gy)] = new_s
        else:
            color = self._palette[cidx]
            s.color_remap(None, color)

    def _update_selector(self, gx: int, gy: int) -> None:
        if self._selector_sprite is None:
            return
        px, py = _tile_pos(gx, gy, self._ox, self._oy)
        self._selector_sprite.set_position(px, py)
        self._selector_sprite.set_visible(True)

    def _hide_selector(self) -> None:
        if self._selector_sprite:
            self._selector_sprite.set_visible(False)

    def _display_to_grid(self, x: int, y: int) -> tuple[int, int] | None:
        px = x * CAM_SIZE // 64
        py = y * CAM_SIZE // 64
        gx = (px - self._ox) // TILE_SIZE
        gy = (py - self._oy) // TILE_SIZE
        if 0 <= gx < self._grid_w and 0 <= gy < self._grid_h:
            return (gx, gy)
        return None

    def _update_color_indicator(self, cidx: int) -> None:
        if self._current_color_ind is None:
            return
        if cidx == -1:
            self._current_color_ind.color_remap(None, GAP_COLOR)
        else:
            color = self._palette[cidx % self._num_colors]
            self._current_color_ind.color_remap(None, color)

    def _reset_all_gaps(self) -> None:
        for gx, gy in self._gap_set:
            self._gap_colors[(gx, gy)] = -1
            self._update_gap_visual(gx, gy)
        self._selected_gap = None
        self._hide_selector()
        self._update_color_indicator(-1)

    def _update_life_display(self) -> None:
        for i, life_sprite in enumerate(self._life_sprites):
            if i < self._lives:
                life_sprite.set_visible(True)
            else:
                life_sprite.set_visible(False)

    def _lose_life(self) -> None:
        self._lives -= 1
        self._update_life_display()
        self._can_undo = False
        self._undo_stack.clear()

        if self._lives <= 0:
            self._game_over = True
            self._reset_all_gaps()
            self.lose()
        else:
            self._reset_all_gaps()
            self._moves_left = self._max_moves

    def _check_complete(self) -> bool:
        for gx, gy in self._gap_set:
            cidx = self._gap_colors.get((gx, gy), -1)
            if cidx == -1:
                return False
            if cidx != self._pattern[gy][gx]:
                return False
        return True

    def _select_gap(self, gx: int, gy: int) -> None:
        self._selected_gap = (gx, gy)
        self._update_selector(gx, gy)
        self._update_color_indicator(self._gap_colors.get((gx, gy), -1))

    def _move_selection(self, dx: int, dy: int) -> None:
        if not self._gap_set:
            return

        ordered_gaps = sorted(self._gap_set, key=lambda p: (p[1], p[0]))

        if self._selected_gap is None:
            gx, gy = ordered_gaps[0]
            self._select_gap(gx, gy)
            return

        try:
            idx = ordered_gaps.index(self._selected_gap)
        except ValueError:
            idx = 0

        step = 0
        if dx > 0 or dy > 0:
            step = 1
        elif dx < 0 or dy < 0:
            step = -1

        if step == 0:
            return

        next_idx = (idx + step) % len(ordered_gaps)
        gx, gy = ordered_gaps[next_idx]
        self._select_gap(gx, gy)

    def _cycle_selected_gap(self, forward: bool = True) -> None:
        if self._selected_gap is None or self._selected_gap not in self._gap_set:
            return

        gx, gy = self._selected_gap
        old_cidx = self._gap_colors.get((gx, gy), -1)
        new_cidx = old_cidx + 1 if forward else old_cidx - 1

        if new_cidx >= self._num_colors:
            new_cidx = -1
        elif new_cidx < -1:
            new_cidx = self._num_colors - 1

        self._gap_colors[(gx, gy)] = new_cidx
        self._update_gap_visual(gx, gy)
        self._update_color_indicator(new_cidx)

    def _save_undo_snapshot(self) -> None:
        self._undo_stack.append(
            (
                dict(self._gap_colors),
                self._selected_gap,
                self._lives,
            )
        )
        self._can_undo = True

    def _restore_undo_snapshot(self) -> None:
        if not self._undo_stack:
            return
        gap_colors, selected_gap, lives = self._undo_stack.pop()
        self._gap_colors = dict(gap_colors)
        self._selected_gap = selected_gap
        self._lives = lives
        self._update_life_display()
        for gx, gy in self._gap_set:
            self._update_gap_visual(gx, gy)
        if self._selected_gap is not None:
            self._update_selector(self._selected_gap[0], self._selected_gap[1])
            self._update_color_indicator(self._gap_colors.get(self._selected_gap, -1))
        else:
            self._hide_selector()
            self._update_color_indicator(-1)
        self._can_undo = len(self._undo_stack) > 0

    def step(self) -> None:
        if self._game_over:
            self.complete_action()
            return

        if self.action.id == GameAction.RESET:
            self.complete_action()
            return

        if self.action.id == GameAction.ACTION7:
            if self._can_undo:
                self._restore_undo_snapshot()
            self._moves_left -= 1
            if self._moves_left <= 0:
                self._lose_life()
            self.complete_action()
            return

        if self.action.id == GameAction.ACTION6:
            self._save_undo_snapshot()
            has_coords = (
                isinstance(self.action.data, dict)
                and "x" in self.action.data
                and "y" in self.action.data
            )
            if has_coords:
                x = self.action.data.get("x", 0)
                y = self.action.data.get("y", 0)
                cell = self._display_to_grid(x, y)
                if cell is not None and cell in self._gap_set:
                    self._select_gap(cell[0], cell[1])
                    self._cycle_selected_gap(forward=True)
            else:
                self._cycle_selected_gap(forward=True)

            if self._check_complete():
                self._can_undo = False
                self._undo_stack.clear()
                self._level_done = True
                self.next_level()
                self.complete_action()
                return

            self._moves_left -= 1
            if self._moves_left <= 0:
                self._can_undo = False
                self._undo_stack.clear()
                self._lose_life()
            self.complete_action()
            return

        if self.action.id == GameAction.ACTION1:
            self._save_undo_snapshot()
            self._cycle_selected_gap(forward=False)
            if self._check_complete():
                self._can_undo = False
                self._undo_stack.clear()
                self._level_done = True
                self.next_level()
                self.complete_action()
                return

        elif self.action.id == GameAction.ACTION2:
            self._save_undo_snapshot()
            self._cycle_selected_gap(forward=True)
            if self._check_complete():
                self._can_undo = False
                self._undo_stack.clear()
                self._level_done = True
                self.next_level()
                self.complete_action()
                return

        elif self.action.id == GameAction.ACTION3:
            self._save_undo_snapshot()
            self._move_selection(-1, 0)

        elif self.action.id == GameAction.ACTION4:
            self._save_undo_snapshot()
            self._move_selection(1, 0)

        elif self.action.id == GameAction.ACTION5:
            self._save_undo_snapshot()
            self._reset_all_gaps()

        else:
            self.complete_action()
            return

        self._moves_left -= 1
        if self._moves_left <= 0:
            self._can_undo = False
            self._undo_stack.clear()
            self._lose_life()

        self.complete_action()


class PuzzleEnvironment:
    _ARC_PALETTE = [
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

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self._engine = Cp01(seed=seed)
        self._total_turns = 0
        self._done = False
        self._last_action_was_reset = False

    def reset(self) -> GameState:
        self._engine.perform_action(ActionInput(id=GameAction.RESET))
        self._total_turns = 0
        self._done = False
        self._last_action_was_reset = False
        return self._get_state()

    def get_actions(self) -> list[str]:
        if self._done:
            return ["reset"]
        actions = ["up", "down", "left", "right", "select"]
        engine = self._engine
        gap_set = getattr(engine, "_gap_set", set())
        ox = getattr(engine, "_ox", 0)
        oy = getattr(engine, "_oy", 0)
        actions.append("click")
        for gx, gy in sorted(gap_set, key=lambda p: (p[1], p[0])):
            dx = (ox + gx * TILE_SIZE) * 64 // CAM_SIZE
            dy = (oy + gy * TILE_SIZE) * 64 // CAM_SIZE
            actions.append(f"click_{dx}_{dy}")
        actions.append("undo")
        actions.append("reset")
        return actions

    def step(self, action: str) -> StepResult:
        if self._done and action != "reset":
            return StepResult(
                state=self._get_state(),
                reward=0.0,
                done=True,
                info={"action": action},
            )

        action_map = {
            "up": GameAction.ACTION1,
            "down": GameAction.ACTION2,
            "left": GameAction.ACTION3,
            "right": GameAction.ACTION4,
            "select": GameAction.ACTION5,
            "click": GameAction.ACTION6,
            "undo": GameAction.ACTION7,
            "reset": GameAction.RESET,
        }

        if action == "reset":
            if self._last_action_was_reset:
                self._engine = Cp01(seed=self._seed)
                self._engine.perform_action(ActionInput(id=GameAction.RESET))
                self._total_turns = 0
                self._done = False
                self._last_action_was_reset = False
            else:
                self._engine.perform_action(ActionInput(id=GameAction.RESET))
                self._total_turns = 0
                self._done = False
                self._last_action_was_reset = True
            return StepResult(
                state=self._get_state(),
                reward=0.0,
                done=False,
                info={"reset": True},
            )

        self._last_action_was_reset = False

        if action.startswith("click_"):
            parts = action.split("_")
            if len(parts) == 3:
                x, y = int(parts[1]), int(parts[2])
                self._total_turns += 1
                level_before = self._engine.level_index
                self._engine.perform_action(ActionInput(
                    id=GameAction.ACTION6, data={"x": x, "y": y}
                ))
                level_done = getattr(self._engine, "_level_done", False)
                game_over = getattr(self._engine, "_game_over", False)
                if level_done:
                    self._engine._level_done = False
                total_levels = len(self._engine._levels)
                last_level_idx = total_levels - 1
                done = game_over or (level_done and level_before == last_level_idx)
                self._done = done
                reward = (1.0 / total_levels) if level_done else 0.0
                return StepResult(
                    state=self._get_state(),
                    reward=reward,
                    done=self._done,
                    info={"action": action},
                )

        engine_action = action_map.get(action)
        if engine_action is None:
            return StepResult(
                state=self._get_state(),
                reward=0.0,
                done=self._done,
                info={"action": action},
            )

        self._total_turns += 1
        level_before = self._engine.level_index
        self._engine.perform_action(ActionInput(id=engine_action, data={}))

        level_done = getattr(self._engine, "_level_done", False)
        game_over = getattr(self._engine, "_game_over", False)
        if level_done:
            self._engine._level_done = False
        total_levels = len(self._engine._levels)
        last_level_idx = total_levels - 1
        done = game_over or (level_done and level_before == last_level_idx)
        self._done = done
        reward = (1.0 / total_levels) if level_done else 0.0
        return StepResult(
            state=self._get_state(),
            reward=reward,
            done=self._done,
            info={"action": action},
        )

    def is_done(self) -> bool:
        return self._done

    def _get_state(self) -> GameState:
        engine = self._engine
        grid_w = getattr(engine, "_grid_w", 0)
        grid_h = getattr(engine, "_grid_h", 0)
        pattern = getattr(engine, "_pattern", [])
        gap_set = getattr(engine, "_gap_set", set())
        gap_colors = getattr(engine, "_gap_colors", {})
        palette = getattr(engine, "_palette", [])
        moves_left = getattr(engine, "_moves_left", 0)
        lives = getattr(engine, "_lives", 0)
        num_colors = getattr(engine, "_num_colors", 0)

        lines = []
        for gy in range(grid_h):
            row_chars = []
            for gx in range(grid_w):
                if (gx, gy) in gap_set:
                    cidx = gap_colors.get((gx, gy), -1)
                    if cidx == -1:
                        row_chars.append("?")
                    else:
                        row_chars.append(str(palette[cidx]))
                else:
                    cidx = pattern[gy][gx]
                    row_chars.append(str(palette[cidx]))
            lines.append(" ".join(row_chars))

        lines.append(f"lives:{lives} moves:{moves_left}")
        lines.append(f"palette:{','.join(str(c) for c in palette)}")
        lines.append(f"colors:{num_colors}")

        gap_info = []
        for gx, gy in sorted(gap_set, key=lambda p: (p[1], p[0])):
            cidx = gap_colors.get((gx, gy), -1)
            gap_info.append(f"({gx},{gy})={'?' if cidx == -1 else palette[cidx]}")
        lines.append(f"gaps:{' '.join(gap_info)}")

        text = "\n".join(lines)

        image = None
        rgb_frame = self.render()
        if rgb_frame is not None:
            image = _encode_png(rgb_frame)

        selected_gap = getattr(engine, "_selected_gap", None)

        return GameState(
            text_observation=text,
            image_observation=image,
            valid_actions=self.get_actions() if not self._done else None,
            turn=self._total_turns,
            metadata={
                "total_levels": len(self._engine._levels),
                "lives": lives,
                "moves_left": moves_left,
                "selected_gap": list(selected_gap) if selected_gap else None,
            },
        )

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError(f"Unsupported render mode: {mode}")
        index_grid = self._engine.camera.render(
            self._engine.current_level.get_sprites()
        )
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        if index_grid is not None:
            for idx, color in enumerate(self._ARC_PALETTE):
                mask = index_grid == idx
                rgb[mask] = color
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
        "click",
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
                base_action = a.split("_")[0] if a.startswith("click_") else a
                idx = self._string_to_action.get(base_action)
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
    check_env(env, skip_render_check=False)
    obs, info = env.reset()
    obs, reward, term, trunc, info = env.step(0)
    env.render()
    env.close()
