import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import gymnasium as gym
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
from gymnasium import spaces


C_WHITE = 0
C_OFFWHITE = 1
C_LGREY = 2
C_GREY = 3
C_DGREY = 4
C_BLACK = 5
C_MAGENTA = 6
C_PINK = 7
C_RED = 8
C_BLUE = 9
C_LBLUE = 10
C_YELLOW = 11
C_ORANGE = 12
C_MAROON = 13
C_GREEN = 14
C_PURPLE = 15

_T = -1
PX = 4
MAX_LIVES = 3
NUM_LEVELS = 4

FOLD_H = "horizontal"
FOLD_V = "vertical"

_MIX: Dict[Tuple[int, int], int] = {
    (C_RED, C_BLUE): C_PURPLE,
    (C_BLUE, C_YELLOW): C_GREEN,
    (C_RED, C_YELLOW): C_ORANGE,
    (C_RED, C_PINK): C_MAGENTA,
    (C_BLUE, C_PINK): C_PURPLE,
    (C_GREEN, C_BLUE): C_LBLUE,
    (C_ORANGE, C_BLUE): C_MAROON,
    (C_GREEN, C_RED): C_MAROON,
    (C_PURPLE, C_YELLOW): C_LGREY,
    (C_PURPLE, C_RED): C_MAGENTA,
    (C_PURPLE, C_BLUE): C_BLUE,
    (C_MAGENTA, C_BLUE): C_PURPLE,
    (C_MAGENTA, C_YELLOW): C_PINK,
    (C_MAGENTA, C_RED): C_RED,
    (C_LBLUE, C_RED): C_PURPLE,
    (C_LBLUE, C_YELLOW): C_GREEN,
    (C_MAROON, C_BLUE): C_PURPLE,
    (C_MAROON, C_YELLOW): C_ORANGE,
    (C_MAROON, C_RED): C_RED,
    (C_ORANGE, C_RED): C_RED,
    (C_ORANGE, C_YELLOW): C_YELLOW,
    (C_GREEN, C_YELLOW): C_GREEN,
    (C_OFFWHITE, C_RED): C_PINK,
    (C_OFFWHITE, C_BLUE): C_LBLUE,
    (C_OFFWHITE, C_YELLOW): C_OFFWHITE,
    (C_OFFWHITE, C_GREEN): C_LBLUE,
    (C_OFFWHITE, C_PURPLE): C_PINK,
    (C_OFFWHITE, C_ORANGE): C_YELLOW,
    (C_GREY, C_RED): C_MAROON,
    (C_GREY, C_BLUE): C_DGREY,
    (C_GREY, C_YELLOW): C_LGREY,
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


def mix_colors(a: int, b: int) -> int:
    if a == C_BLACK:
        return b
    if b == C_BLACK:
        return a
    if a == b:
        return a
    key = (a, b)
    if key in _MIX:
        return _MIX[key]
    key2 = (b, a)
    if key2 in _MIX:
        return _MIX[key2]
    return max(a, b)


def fold_grid(grid: List[List[int]], fold_type: str, fold_pos: int) -> List[List[int]]:
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0

    if fold_type == FOLD_V:
        left_part = [row[:fold_pos] for row in grid]
        right_part = [row[fold_pos:] for row in grid]
        left_w = fold_pos
        right_w = cols - fold_pos

        if left_w <= right_w:
            new_grid = [row[:] for row in right_part]
            for r in range(rows):
                for c in range(left_w):
                    mirror_c = left_w - 1 - c
                    old_color = left_part[r][c]
                    new_grid[r][mirror_c] = mix_colors(old_color, new_grid[r][mirror_c])
        else:
            new_grid = [row[:] for row in left_part]
            for r in range(rows):
                for c in range(right_w):
                    mirror_c = left_w - 1 - c
                    if not (0 <= mirror_c < left_w):
                        continue
                    old_color = right_part[r][c]
                    new_grid[r][mirror_c] = mix_colors(old_color, new_grid[r][mirror_c])
        return new_grid

    elif fold_type == FOLD_H:
        top_part = grid[:fold_pos]
        bottom_part = grid[fold_pos:]
        top_h = fold_pos
        bottom_h = rows - fold_pos

        if top_h <= bottom_h:
            new_grid = [row[:] for row in bottom_part]
            for r in range(top_h):
                mirror_r = top_h - 1 - r
                for c in range(cols):
                    old_color = top_part[r][c]
                    new_grid[mirror_r][c] = mix_colors(old_color, new_grid[mirror_r][c])
        else:
            new_grid = [row[:] for row in top_part]
            for r in range(bottom_h):
                mirror_r = top_h - 1 - r
                if not (0 <= mirror_r < top_h):
                    continue
                for c in range(cols):
                    old_color = bottom_part[r][c]
                    new_grid[mirror_r][c] = mix_colors(old_color, new_grid[mirror_r][c])
        return new_grid

    return grid


def _level_1() -> Dict[str, Any]:
    target = [[C_GREEN] * 2 for _ in range(2)]
    g = [
        [C_YELLOW, C_BLACK, C_BLUE, C_BLACK],
        [C_YELLOW, C_BLUE, C_BLACK, C_BLACK],
        [C_BLACK, C_BLACK, C_YELLOW, C_BLUE],
        [C_BLACK, C_YELLOW, C_BLACK, C_BLUE],
    ]
    return dict(grid=g, target_grid=target, max_moves=15)


def _level_2() -> Dict[str, Any]:
    target = [
        [C_BLACK, C_ORANGE, C_BLACK],
        [C_ORANGE, C_ORANGE, C_ORANGE],
        [C_BLACK, C_ORANGE, C_BLACK],
    ]
    g = [[C_BLACK] * 6 for _ in range(6)]
    for r, c in [(0, 2), (1, 2), (2, 2)]:
        g[r][c] = C_YELLOW
    for r, c in [(3, 2), (4, 1), (4, 2), (4, 3), (5, 2)]:
        g[r][c] = C_RED
    for r, c in [(4, 0), (4, 4)]:
        g[r][c] = C_YELLOW
    return dict(grid=g, target_grid=target, max_moves=50)


def _level_3() -> Dict[str, Any]:
    target = [[C_MAGENTA] * 4 for _ in range(4)]
    target[1][1] = C_BLACK
    target[1][2] = C_BLACK
    target[2][1] = C_BLACK
    target[2][2] = C_BLACK
    g = [[C_BLACK] * 8 for _ in range(8)]
    for r, c, col in [
        (0, 3, C_PINK),
        (0, 7, C_PINK),
        (3, 4, C_RED),
        (3, 5, C_RED),
        (3, 6, C_RED),
        (3, 7, C_RED),
        (4, 4, C_PINK),
        (4, 5, C_PINK),
        (4, 6, C_PINK),
        (4, 7, C_PINK),
        (5, 0, C_RED),
        (5, 4, C_RED),
        (6, 0, C_RED),
        (6, 3, C_PINK),
        (6, 4, C_RED),
        (6, 7, C_PINK),
        (7, 0, C_PINK),
        (7, 1, C_PINK),
        (7, 2, C_PINK),
        (7, 3, C_PINK),
        (7, 4, C_RED),
        (7, 5, C_RED),
        (7, 6, C_RED),
        (7, 7, C_RED),
    ]:
        g[r][c] = col
    return dict(grid=g, target_grid=target, max_moves=70)


def _level_4() -> Dict[str, Any]:
    target = [[C_BLACK] * 3 for _ in range(3)]
    target[0][0] = C_LBLUE
    target[1][0] = C_LBLUE
    target[2][0] = C_LBLUE
    target[2][1] = C_LBLUE
    target[2][2] = C_LBLUE
    g = [[C_BLACK] * 6 for _ in range(8)]
    for r, c, col in [
        (0, 1, C_GREEN),
        (1, 1, C_BLUE),
        (4, 0, C_BLUE),
        (4, 1, C_GREEN),
        (4, 2, C_GREEN),
        (4, 3, C_GREEN),
        (5, 2, C_BLUE),
        (5, 3, C_BLUE),
        (6, 1, C_BLUE),
        (7, 1, C_GREEN),
    ]:
        g[r][c] = col
    return dict(grid=g, target_grid=target, max_moves=90)


LEVEL_BUILDERS = [
    _level_1,
    _level_2,
    _level_3,
    _level_4,
]

LEVEL_NAMES = [
    "L1: Color Mix",
    "L2: Cross Align",
    "L3: The Ring",
    "L4: The L-Shift",
]


def _build_levels() -> List[Level]:
    result: List[Level] = []
    for idx, builder in enumerate(LEVEL_BUILDERS):
        data = builder()
        level = Level(sprites=[], grid_size=(64, 64), data=data, name=LEVEL_NAMES[idx])
        result.append(level)
    return result


levels = _build_levels()


class _HUD(RenderableUserDisplay):
    def __init__(self) -> None:
        self.lives: int = MAX_LIVES
        self.max_lives: int = MAX_LIVES
        self.current_level: int = 0
        self.total_levels: int = NUM_LEVELS
        self.moves_used: int = 0
        self.max_moves: int = 20
        self.target_grid: List[List[int]] = [[C_BLACK]]

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        for r in range(6):
            for c in range(64):
                frame[r, c] = C_BLACK

        for r in range(43, 64):
            for c in range(64):
                frame[r, c] = C_BLACK

        for i in range(self.total_levels):
            dc = (
                C_PURPLE
                if i < self.current_level
                else (C_LBLUE if i == self.current_level else C_DGREY)
            )
            x = 2 + i * 4
            frame[2, x] = dc
            frame[2, x + 1] = dc

        for i in range(self.max_lives):
            lc = C_GREEN if i < self.lives else C_DGREY
            bx = 52 + i * 4
            frame[2, bx] = lc
            frame[2, bx + 1] = lc
            frame[3, bx] = lc
            frame[3, bx + 1] = lc

        bar_w = 38
        moves_left = max(0, self.max_moves - self.moves_used)
        filled = (
            int(bar_w * moves_left / self.max_moves) if self.max_moves > 0 else bar_w
        )

        for r in range(58, 62):
            for c in range(23, 63):
                frame[r, c] = C_DGREY

        for c in range(bar_w):
            col_c = 24 + c
            if c < filled:
                ratio = moves_left / self.max_moves
                bar_col = (
                    C_GREEN if ratio > 0.5 else (C_ORANGE if ratio > 0.25 else C_RED)
                )
            else:
                bar_col = C_BLACK
            frame[59, col_c] = bar_col
            frame[60, col_c] = bar_col

        box_r0, box_r1 = 43, 62
        box_c0, box_c1 = 1, 20

        for r in range(box_r0, box_r1 + 1):
            frame[r, box_c0] = C_WHITE
            frame[r, box_c1] = C_WHITE
        for c in range(box_c0, box_c1 + 1):
            frame[box_r0, c] = C_WHITE
            frame[box_r1, c] = C_WHITE

        tr = len(self.target_grid)
        tc = len(self.target_grid[0]) if tr > 0 else 0

        if tr > 0 and tc > 0:
            cpx = 16 // max(tr, tc)
            if cpx > 7:
                cpx = 7

            tw = tc * cpx + (tc - 1)
            th = tr * cpx + (tr - 1)

            sr = box_r0 + 1 + (18 - th) // 2
            sc = box_c0 + 1 + (18 - tw) // 2

            for r in range(th):
                for c in range(tw):
                    if 0 <= sr + r < 64 and 0 <= sc + c < 64:
                        frame[sr + r, sc + c] = C_DGREY

            for r in range(tr):
                for c in range(tc):
                    color = self.target_grid[r][c]
                    for pr in range(cpx):
                        for pc in range(cpx):
                            px_r = sr + r * (cpx + 1) + pr
                            px_c = sc + c * (cpx + 1) + pc
                            if 0 <= px_r < 64 and 0 <= px_c < 64:
                                frame[px_r, px_c] = color

        return frame


def _cell_sprite(color: int, px: int) -> Sprite:
    pixels = [[color] * px for _ in range(px)]
    return Sprite(
        pixels=pixels, name=f"cell_{color}", visible=True, collidable=False, layer=0
    )


def _fold_line_h_sprite(width: int) -> Sprite:
    pixels = [[C_GREY if c % 2 == 0 else C_DGREY for c in range(width)]]
    return Sprite(pixels=pixels, name="foldh", visible=True, collidable=False, layer=2)


def _fold_line_v_sprite(height: int) -> Sprite:
    pixels = [[C_GREY if r % 2 == 0 else C_DGREY] for r in range(height)]
    return Sprite(pixels=pixels, name="foldv", visible=True, collidable=False, layer=2)


def _fold_line_h_sel_sprite(width: int) -> Sprite:
    pixels = [[C_LBLUE] * width]
    return Sprite(
        pixels=pixels, name="foldh_sel", visible=True, collidable=False, layer=3
    )


def _fold_line_v_sel_sprite(height: int) -> Sprite:
    pixels = [[C_LBLUE] for _ in range(height)]
    return Sprite(
        pixels=pixels, name="foldv_sel", visible=True, collidable=False, layer=3
    )


class Fc01(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self._rng = random.Random(seed)
        self._lives: int = MAX_LIVES
        self._hud = _HUD()

        self._grid: List[List[int]] = []
        self._initial_grid: List[List[int]] = []
        self._grid_rows: int = 0
        self._grid_cols: int = 0
        self._target_grid: List[List[int]] = [[C_BLACK]]
        self._max_moves: int = 20
        self._moves_used: int = 0

        self._selected_fold_type: Optional[str] = None
        self._selected_fold_pos: int = 0
        self._available_folds: List[Tuple[str, int]] = []
        self._fold_index: int = 0
        self._level_just_loaded: bool = False

        self._render_ox: int = 0
        self._render_oy: int = 0
        self._cell_px: int = PX
        self._last_was_reset: bool = False
        self._history: List[Dict[str, Any]] = []

        cam = Camera(
            x=0,
            y=0,
            width=64,
            height=64,
            background=C_BLACK,
            letter_box=C_BLACK,
            interfaces=[self._hud],
        )
        super().__init__(
            game_id="fc01",
            levels=levels,
            camera=cam,
            available_actions=[0, 1, 2, 3, 4, 5, 7],
            win_score=NUM_LEVELS,
        )
        self._level_just_loaded = False

    def on_set_level(self, level: Level) -> None:
        data = self.current_level.get_data("grid")
        self._initial_grid = [row[:] for row in data]
        self._grid = [row[:] for row in data]
        self._grid_rows = len(self._grid)
        self._grid_cols = len(self._grid[0])
        self._target_grid = self.current_level.get_data("target_grid")
        self._hud.target_grid = self._target_grid
        self._max_moves = self.current_level.get_data("max_moves")
        self._moves_used = 0

        self._lives = MAX_LIVES
        self._level_just_loaded = True
        self._history = []

        self._compute_available_folds()
        self._fold_index = self._seed_starting_fold()
        if self._available_folds:
            ft, fp = self._available_folds[self._fold_index]
            self._selected_fold_type = ft
            self._selected_fold_pos = fp
        else:
            self._selected_fold_type = None
            self._selected_fold_pos = 0

        self._rebuild()
        self._sync_hud()

    def _sync_hud(self) -> None:
        self._hud.lives = self._lives
        self._hud.current_level = self.level_index
        self._hud.moves_used = self._moves_used
        self._hud.max_moves = self._max_moves

    def _compute_available_folds(self) -> None:
        rows, cols = self._grid_rows, self._grid_cols
        folds: List[Tuple[str, int]] = []
        for r in range(1, rows):
            folds.append((FOLD_H, r))
        for c in range(1, cols):
            folds.append((FOLD_V, c))
        self._available_folds = folds

    def _seed_starting_fold(self) -> int:
        if not self._available_folds:
            return 0
        fold_rng = random.Random(self._seed + self.level_index)
        return fold_rng.randint(0, len(self._available_folds) - 1)

    def _compute_cell_px(self) -> int:
        avail_w, avail_h = 60, 35
        rows, cols = self._grid_rows, self._grid_cols
        if rows == 0 or cols == 0:
            return PX
        px_w = (avail_w + 1) // cols - 1
        px_h = (avail_h + 1) // rows - 1
        return min(max(min(px_w, px_h), 1), 10)

    def _rebuild(self) -> None:
        self.current_level.remove_all_sprites()
        rows, cols = self._grid_rows, self._grid_cols
        self._cell_px = cpx = self._compute_cell_px()

        total_w = cols * cpx + max(0, cols - 1)
        total_h = rows * cpx + max(0, rows - 1)

        area_x0, area_y0 = 2, 7
        area_w, area_h = 60, 35
        self._render_ox = area_x0 + max(0, (area_w - total_w) // 2)
        self._render_oy = area_y0 + max(0, (area_h - total_h) // 2)

        for r in range(rows):
            for c in range(cols):
                color = self._grid[r][c]
                px_x = self._render_ox + c * (cpx + 1)
                px_y = self._render_oy + r * (cpx + 1)
                spr = _cell_sprite(color, cpx)
                spr.set_position(px_x, px_y)
                self.current_level.add_sprite(spr)

        for ft, fp in self._available_folds:
            selected = ft == self._selected_fold_type and fp == self._selected_fold_pos
            if ft == FOLD_H:
                line_y = self._render_oy + fp * (cpx + 1) - 1
                spr = (
                    _fold_line_h_sel_sprite(total_w)
                    if selected
                    else _fold_line_h_sprite(total_w)
                )
                spr.set_position(self._render_ox, line_y)
                self.current_level.add_sprite(spr)
            elif ft == FOLD_V:
                line_x = self._render_ox + fp * (cpx + 1) - 1
                spr = (
                    _fold_line_v_sel_sprite(total_h)
                    if selected
                    else _fold_line_v_sprite(total_h)
                )
                spr.set_position(line_x, self._render_oy)
                self.current_level.add_sprite(spr)

    def _execute_fold(self, fold_type: str, fold_pos: int) -> None:
        new_grid = fold_grid(self._grid, fold_type, fold_pos)
        self._grid = new_grid
        self._grid_rows = len(new_grid)
        self._grid_cols = len(new_grid[0]) if self._grid_rows > 0 else 0

        self._compute_available_folds()
        if self._available_folds:
            self._selected_fold_type, self._selected_fold_pos = self._available_folds[0]
            self._fold_index = 0
        else:
            self._selected_fold_type, self._selected_fold_pos = None, 0

        self._rebuild()

    def _cycle_h_folds(self, direction: int) -> None:
        h_folds = [
            (i, ft, fp)
            for i, (ft, fp) in enumerate(self._available_folds)
            if ft == FOLD_H
        ]
        if not h_folds:
            return
        if self._selected_fold_type == FOLD_H:
            cur = [
                j
                for j, (i, ft, fp) in enumerate(h_folds)
                if fp == self._selected_fold_pos
            ]
            pos = (cur[0] + direction) % len(h_folds) if cur else 0
        else:
            pos = 0 if direction > 0 else len(h_folds) - 1
        idx, ft, fp = h_folds[pos]
        self._fold_index = idx
        self._selected_fold_type, self._selected_fold_pos = ft, fp
        self._rebuild()

    def _cycle_v_folds(self, direction: int) -> None:
        v_folds = [
            (i, ft, fp)
            for i, (ft, fp) in enumerate(self._available_folds)
            if ft == FOLD_V
        ]
        if not v_folds:
            return
        if self._selected_fold_type == FOLD_V:
            cur = [
                j
                for j, (i, ft, fp) in enumerate(v_folds)
                if fp == self._selected_fold_pos
            ]
            pos = (cur[0] + direction) % len(v_folds) if cur else 0
        else:
            pos = 0 if direction > 0 else len(v_folds) - 1
        idx, ft, fp = v_folds[pos]
        self._fold_index = idx
        self._selected_fold_type, self._selected_fold_pos = ft, fp
        self._rebuild()

    def _save_state(self) -> None:
        self._history.append(
            {
                "grid": [row[:] for row in self._grid],
                "grid_rows": self._grid_rows,
                "grid_cols": self._grid_cols,
                "available_folds": list(self._available_folds),
                "fold_index": self._fold_index,
                "selected_fold_type": self._selected_fold_type,
                "selected_fold_pos": self._selected_fold_pos,
            }
        )

    def _restore_from_undo(self) -> None:
        if not self._history:
            return
        state = self._history.pop()
        self._grid = state["grid"]
        self._grid_rows = state["grid_rows"]
        self._grid_cols = state["grid_cols"]
        self._available_folds = state["available_folds"]
        self._fold_index = state["fold_index"]
        self._selected_fold_type = state["selected_fold_type"]
        self._selected_fold_pos = state["selected_fold_pos"]
        self._rebuild()

    def _check_win_loss(self) -> None:
        if self._grid_rows == len(self._target_grid) and self._grid_cols == len(
            self._target_grid[0]
        ):
            match = True
            for r in range(self._grid_rows):
                for c in range(self._grid_cols):
                    if self._grid[r][c] != self._target_grid[r][c]:
                        match = False
                        break
            if match:
                self._sync_hud()
                self.next_level()
                return

        if self._moves_used >= self._max_moves or len(self._available_folds) == 0:
            self._lives -= 1
            if self._lives <= 0:
                self._sync_hud()
                self._rebuild()
                self.lose()
            else:
                self._grid = [row[:] for row in self._initial_grid]
                self._grid_rows = len(self._grid)
                self._grid_cols = len(self._grid[0])
                self._moves_used = 0
                self._history = []
                self._compute_available_folds()
                if self._available_folds:
                    self._fold_index = self._seed_starting_fold()
                    self._selected_fold_type, self._selected_fold_pos = (
                        self._available_folds[self._fold_index]
                    )
                else:
                    self._selected_fold_type, self._selected_fold_pos = None, 0
                self._sync_hud()
                self._rebuild()

    def step(self) -> None:
        if self._level_just_loaded:
            self._level_just_loaded = False
            self.complete_action()
            return

        aid = self.action.id
        action_taken = False

        if aid != GameAction.RESET:
            self._last_was_reset = False

        if aid == GameAction.ACTION1:
            self._save_state()
            self._moves_used += 1
            self._cycle_h_folds(-1)
            action_taken = True
        elif aid == GameAction.ACTION2:
            self._save_state()
            self._moves_used += 1
            self._cycle_h_folds(+1)
            action_taken = True
        elif aid == GameAction.ACTION3:
            self._save_state()
            self._moves_used += 1
            self._cycle_v_folds(-1)
            action_taken = True
        elif aid == GameAction.ACTION4:
            self._save_state()
            self._moves_used += 1
            self._cycle_v_folds(+1)
            action_taken = True
        elif aid == GameAction.ACTION5:
            if self._selected_fold_type is not None:
                self._save_state()
                self._moves_used += 1
                self._execute_fold(self._selected_fold_type, self._selected_fold_pos)
                action_taken = True
        elif aid == GameAction.ACTION7:
            self._restore_from_undo()
            self._moves_used += 1
            action_taken = True

        if action_taken:
            self._sync_hud()
            self._check_win_loss()

        self.complete_action()

    def handle_reset(self) -> None:
        if self._last_was_reset:
            self._last_was_reset = False
            self._lives = MAX_LIVES
            self.set_level(0)
            self.complete_action()
            return
        self._last_was_reset = True
        self._lives = MAX_LIVES
        self.level_reset()
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

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self._engine = Fc01(seed=seed)
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
        self._total_turns = 0
        self._done = False
        self._last_action_was_reset = True
        self._game_won = False
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
            return StepResult(
                state=self._build_game_state(),
                reward=0.0,
                done=self._done,
                info={"action": action, "reason": "invalid_action"},
            )

        self._last_action_was_reset = False
        self._total_turns += 1

        game_action = self._ACTION_MAP[action]
        info: Dict = {"action": action}
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

    def _build_text_observation(self) -> str:
        e = self._engine
        rows, cols = len(e._grid), len(e._grid[0]) if e._grid else 0
        lines = [
            f"Level {e.level_index + 1}/{NUM_LEVELS} | Lives {e._lives}/{MAX_LIVES} | Moves {e._moves_used}/{e._max_moves}"
        ]
        lines.append(
            f"Grid {rows}x{cols} | Target {len(e._target_grid)}x{len(e._target_grid[0]) if e._target_grid else 0}"
        )
        if e._selected_fold_type:
            lines.append(
                f"Selected fold: {e._selected_fold_type} at {e._selected_fold_pos}"
            )
        for r in range(rows):
            lines.append(" ".join(str(e._grid[r][c]).rjust(2) for c in range(cols)))
        return "\n".join(lines)

    def _build_game_state(self, done: bool = False) -> GameState:
        valid_actions = self.get_actions() if not done else None
        return GameState(
            text_observation=self._build_text_observation(),
            image_observation=None,
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


class ArcGameEnv(gym.Env):
    metadata: Dict[str, Any] = {
        "render_modes": ["rgb_array"],
        "render_fps": 10,
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
