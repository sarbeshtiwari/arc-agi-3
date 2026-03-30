import random
import struct
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from arcengine import ActionInput, ARCBaseGame, GameAction
from arcengine import GameState as EngineGameState
from arcengine.camera import Camera, RenderableUserDisplay
from arcengine.level import Level
from arcengine.sprites import Sprite
from gymnasium import spaces

BACKGROUND = 0

GRID_LINE_COLOR = 1
AXIS_LINE_COLOR = 9

FOREGROUND_COLORS = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

LEVEL_CONFIGS: List[Tuple[int, int, str, int, int, int]] = [
    (10, 10, "vertical", 3, 6, 4),
    (12, 12, "horizontal", 4, 10, 4),
    (16, 16, "dual", 4, 14, 4),
    (20, 20, "diagonal_main", 5, 18, 4),
    (
        24,
        24,
        "diagonal_anti",
        6,
        12,
        3,
    ),
]

LEVEL_MAX_MOVES = [270, 450, 480, 660, 390]

DISPLAY_SIZE = 64
HUD_HEIGHT = 3
HUD_TOP = DISPLAY_SIZE - HUD_HEIGHT
TOTAL_LIVES = 3


def mirror_vertical(x: int, y: int, w: int, h: int) -> List[Tuple[int, int]]:
    return [(w - 1 - x, y)]


def mirror_horizontal(x: int, y: int, w: int, h: int) -> List[Tuple[int, int]]:
    return [(x, h - 1 - y)]


def mirror_dual(x: int, y: int, w: int, h: int) -> List[Tuple[int, int]]:
    return [
        (w - 1 - x, y),
        (x, h - 1 - y),
        (w - 1 - x, h - 1 - y),
    ]


def mirror_diagonal_main(x: int, y: int, w: int, h: int) -> List[Tuple[int, int]]:
    return [(y, x)]


def mirror_diagonal_anti(x: int, y: int, w: int, h: int) -> List[Tuple[int, int]]:
    n = w
    return [(n - 1 - y, n - 1 - x)]


MIRROR_FUNCTIONS = {
    "vertical": mirror_vertical,
    "horizontal": mirror_horizontal,
    "dual": mirror_dual,
    "diagonal_main": mirror_diagonal_main,
    "diagonal_anti": mirror_diagonal_anti,
}


def _place_shape_cluster(
    grid: np.ndarray,
    candidates: List[Tuple[int, int]],
    color: int,
    rng: random.Random,
    min_size: int = 2,
    max_size: int = 5,
) -> List[Tuple[int, int]]:
    if not candidates:
        return []

    h, w = grid.shape
    start = rng.choice(candidates)
    cluster = [start]
    frontier = [start]
    target_size = rng.randint(min_size, max_size)

    while len(cluster) < target_size and frontier:
        cell = rng.choice(frontier)
        neighbors = []
        cx, cy = cell
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = cx + dx, cy + dy
            if (nx, ny) in candidates and (nx, ny) not in cluster:
                neighbors.append((nx, ny))
        if neighbors:
            pick = rng.choice(neighbors)
            cluster.append(pick)
            frontier.append(pick)
        else:
            frontier.remove(cell)

    for cx, cy in cluster:
        grid[cy, cx] = color

    return cluster


def _get_source_region_cells(w: int, h: int, axis_type: str) -> List[Tuple[int, int]]:
    cells = []
    if axis_type == "vertical":
        for y in range(h):
            for x in range(w // 2):
                cells.append((x, y))
    elif axis_type == "horizontal":
        for y in range(h // 2):
            for x in range(w):
                cells.append((x, y))
    elif axis_type == "dual":
        for y in range(h // 2):
            for x in range(w // 2):
                cells.append((x, y))
    elif axis_type == "diagonal_main":
        for y in range(h):
            for x in range(y + 1, w):
                cells.append((x, y))
    elif axis_type == "diagonal_anti":
        n = w
        for y in range(h):
            for x in range(w):
                if x + y < n - 1:
                    cells.append((x, y))
    return cells


def generate_level_grid(
    w: int,
    h: int,
    axis_type: str,
    num_colors: int,
    num_shapes: int,
    rng: random.Random,
    max_cluster: int = 4,
) -> np.ndarray:
    grid = np.full((h, w), BACKGROUND, dtype=np.int8)
    mirror_fn = MIRROR_FUNCTIONS[axis_type]

    colors = rng.sample(FOREGROUND_COLORS, num_colors)

    source_cells = _get_source_region_cells(w, h, axis_type)
    if not source_cells:
        return grid

    available = list(source_cells)
    shapes_placed = 0
    attempts = 0
    max_attempts = num_shapes * 3

    while shapes_placed < num_shapes and available and attempts < max_attempts:
        attempts += 1
        color = rng.choice(colors)
        cluster = _place_shape_cluster(
            grid, available, color, rng, min_size=1, max_size=max_cluster
        )
        if cluster:
            shapes_placed += 1
            for c in cluster:
                if c in available:
                    available.remove(c)

    if axis_type == "diagonal_anti":
        pre_mirror_low, pre_mirror_high = 0.55, 0.70
    else:
        pre_mirror_low, pre_mirror_high = 0.25, 0.45

    source_filled = [(x, y) for (x, y) in source_cells if grid[y, x] != BACKGROUND]
    if source_filled:
        pre_mirror_count = max(
            1, int(len(source_filled) * rng.uniform(pre_mirror_low, pre_mirror_high))
        )
        pre_mirror_cells = rng.sample(
            source_filled, min(pre_mirror_count, len(source_filled))
        )

        for sx, sy in pre_mirror_cells:
            color = grid[sy, sx]
            mirrors = mirror_fn(sx, sy, w, h)
            for mx, my in mirrors:
                if 0 <= mx < w and 0 <= my < h:
                    grid[my, mx] = color

    if axis_type == "diagonal_anti":
        _place_decoys_for_anti_diagonal(grid, w, h, colors, rng)
    elif axis_type == "diagonal_main":
        _place_decoys_for_main_diagonal(grid, w, h, colors, rng)

    return grid


def _place_decoys_for_anti_diagonal(
    grid: np.ndarray, w: int, h: int, colors: List[int], rng: random.Random
) -> None:
    n = w
    decoy_count = rng.randint(2, 4)
    for _ in range(decoy_count):
        y = rng.randint(1, n - 2)
        x = rng.randint(max(0, y - 2), min(n - 1, y + 2))
        if x + y >= n - 1 and grid[y, x] == BACKGROUND:
            grid[y, x] = rng.choice(colors)


def _place_decoys_for_main_diagonal(
    grid: np.ndarray, w: int, h: int, colors: List[int], rng: random.Random
) -> None:
    n = w
    decoy_count = rng.randint(1, 3)
    for _ in range(decoy_count):
        y = rng.randint(1, n - 2)
        x = n - 1 - y + rng.choice([-1, 0, 1])
        x = max(0, min(n - 1, x))
        if y >= x and grid[y, x] == BACKGROUND:
            grid[y, x] = rng.choice(colors)


def check_symmetry(grid: np.ndarray, axis_type: str) -> bool:
    h, w = grid.shape
    mirror_fn = MIRROR_FUNCTIONS[axis_type]

    for y in range(h):
        for x in range(w):
            color = grid[y, x]
            if color == BACKGROUND:
                continue
            mirrors = mirror_fn(x, y, w, h)
            for mx, my in mirrors:
                if 0 <= mx < w and 0 <= my < h:
                    if grid[my, mx] != color:
                        return False
    return True


def has_any_asymmetry(grid: np.ndarray, axis_type: str) -> bool:
    return not check_symmetry(grid, axis_type)


def grid_to_display(gx: int, gy: int, gw: int, gh: int) -> Tuple[int, int]:
    scale = min(64 // gw, 64 // gh)
    x_off = (64 - gw * scale) // 2
    y_off = (64 - gh * scale) // 2
    return x_off + gx * scale + scale // 2, y_off + gy * scale + scale // 2


class Ms01(ARCBaseGame):
    _grids: List[np.ndarray]
    _axis_types: List[str]
    _grid_sizes: List[Tuple[int, int]]
    _rng: random.Random
    _moves_used: List[int]
    _lives: int
    _cursor_x: int
    _cursor_y: int
    _consecutive_resets: int
    _undo_stack: List[Tuple]

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self._rng = random.Random(seed)

        self._grids = []
        self._axis_types = []
        self._grid_sizes = []
        self._moves_used = [0] * len(LEVEL_CONFIGS)
        self._lives = TOTAL_LIVES
        self._cursor_x = 0
        self._cursor_y = 0
        self._consecutive_resets = 0
        self._undo_stack: List[Tuple] = []
        self._game_over = False

        levels: List[Level] = []
        for i, (gw, gh, axis, ncol, nshape, maxcl) in enumerate(LEVEL_CONFIGS):
            grid = generate_level_grid(gw, gh, axis, ncol, nshape, self._rng, maxcl)

            safety = 0
            while not has_any_asymmetry(grid, axis) and safety < 50:
                grid = generate_level_grid(gw, gh, axis, ncol, nshape, self._rng, maxcl)
                safety += 1

            self._grids.append(grid)
            self._axis_types.append(axis)
            self._grid_sizes.append((gw, gh))

            sprite = Sprite(
                pixels=grid.tolist(),
                name=f"grid_{i}",
                x=0,
                y=0,
                layer=0,
            )

            level = Level(
                sprites=[sprite],
                grid_size=(gw, gh),
                name=f"Level {i + 1}",
                data={"axis_type": axis, "grid_index": i},
            )
            levels.append(level)

        camera = Camera(
            x=0,
            y=0,
            width=64,
            height=64,
            background=BACKGROUND,
            letter_box=BACKGROUND,
        )

        super().__init__(
            game_id="ms01",
            levels=levels,
            camera=camera,
            available_actions=[0, 1, 2, 3, 4, 5, 6, 7],
            seed=seed,
        )

        overlay = _GridAxisOverlay(self)
        self.camera.replace_interface([overlay])

    @property
    def level_index(self) -> int:
        return self._current_level_index

    def on_set_level(self, level: Level) -> None:
        idx = level.get_data("grid_index")
        if idx is not None and idx < len(self._grid_sizes):
            gw, gh = self._grid_sizes[idx]
            self.camera.x = 0
            self.camera.y = 0
            self._cursor_x = self._rng.randint(0, gw - 1)
            self._cursor_y = self._rng.randint(0, gh - 1)
            self._moves_used[idx] = 0
            self._undo_stack = []

    def full_reset(self) -> None:
        self._game_over = False
        self._moves_used = [0] * len(LEVEL_CONFIGS)
        self._lives = TOTAL_LIVES
        self._consecutive_resets = 0
        self._undo_stack = []

        super().full_reset()

        for i in range(len(self._grids)):
            clean_sprites = self._clean_levels[i].get_sprites_by_name(f"grid_{i}")
            if clean_sprites:
                self._grids[i] = np.array(clean_sprites[0].pixels, dtype=np.int8)

        self._sync_current_level_sprite()

    def level_reset(self) -> None:
        self._game_over = False
        idx = self._current_level_index
        self._undo_stack = []
        super().level_reset()

        clean_sprites = self._clean_levels[idx].get_sprites_by_name(f"grid_{idx}")
        if clean_sprites:
            self._grids[idx] = np.array(clean_sprites[0].pixels, dtype=np.int8)

        self._sync_current_level_sprite()
        self._lives = TOTAL_LIVES

    def _sync_current_level_sprite(self) -> None:
        idx = self.current_level.get_data("grid_index")
        if idx is not None and idx < len(self._grids):
            sprites = self.current_level.get_sprites_by_name(f"grid_{idx}")
            if sprites:
                sprites[0].pixels = np.array(self._grids[idx], dtype=np.int8)

    def _get_current_grid(self) -> np.ndarray:
        idx = self.current_level.get_data("grid_index")
        if idx is not None:
            return self._grids[idx]
        raise ValueError("No grid index found for current level")

    def _get_current_axis(self) -> str:
        idx = self.current_level.get_data("grid_index")
        if idx is not None:
            return self._axis_types[idx]
        raise ValueError("No axis type found for current level")

    def step(self) -> None:
        action = self.action

        if action.id == GameAction.RESET:
            self.complete_action()
            return

        if self._state == EngineGameState.GAME_OVER:
            self.complete_action()
            return

        self._consecutive_resets = 0

        if action.id == GameAction.ACTION7:
            level_idx = self._current_level_index
            self._moves_used[level_idx] += 1
            if self._moves_used[level_idx] > LEVEL_MAX_MOVES[level_idx]:
                self._lives -= 1
                if self._lives <= 0:
                    self._game_over = True
                    self.lose()
                else:
                    self.level_reset()
                self.complete_action()
                return
            self._undo()
            self.complete_action()
            return

        level_idx = self._current_level_index
        self._save_undo_state(level_idx)
        self._moves_used[level_idx] += 1

        if self._moves_used[level_idx] > LEVEL_MAX_MOVES[level_idx]:
            self._lives -= 1
            if self._lives <= 0:
                self._game_over = True
                self.lose()
            else:
                self.level_reset()
            self.complete_action()
            return

        if action.id == GameAction.ACTION1:
            self._move_cursor(0, -1)
        elif action.id == GameAction.ACTION2:
            self._move_cursor(0, 1)
        elif action.id == GameAction.ACTION3:
            self._move_cursor(-1, 0)
        elif action.id == GameAction.ACTION4:
            self._move_cursor(1, 0)
        elif action.id == GameAction.ACTION5:
            self._handle_click(
                ActionInput(id=GameAction.ACTION6, data={"x": -1, "y": -1})
            )
        elif action.id == GameAction.ACTION6:
            self._handle_click(action)

        self.complete_action()

    def _move_cursor(self, dx: int, dy: int) -> None:
        idx = self._current_level_index
        gw, gh = self._grid_sizes[idx]
        self._cursor_x = max(0, min(gw - 1, self._cursor_x + dx))
        self._cursor_y = max(0, min(gh - 1, self._cursor_y + dy))

    def _save_undo_state(self, level_idx: int) -> None:
        self._undo_stack = [
            (
                np.copy(self._grids[level_idx]),
                self._cursor_x,
                self._cursor_y,
            )
        ]

    def _undo(self) -> None:
        if not self._undo_stack:
            return
        grid_copy, cx, cy = self._undo_stack.pop()
        level_idx = self._current_level_index
        self._grids[level_idx] = grid_copy
        self._cursor_x = cx
        self._cursor_y = cy
        self._undo_stack = []
        self._sync_current_level_sprite()

    def _handle_click(self, action: ActionInput) -> None:
        data = action.data
        display_x = data.get("x", -1)
        display_y = data.get("y", -1)

        if display_x == -1 and display_y == -1:
            grid_x, grid_y = self._cursor_x, self._cursor_y
        else:
            result = self.camera.display_to_grid(int(display_x), int(display_y))
            if result is None:
                return
            grid_x, grid_y = result
            self._cursor_x, self._cursor_y = grid_x, grid_y

        grid = self._get_current_grid()
        h, w = grid.shape

        if grid_x < 0 or grid_x >= w or grid_y < 0 or grid_y >= h:
            return

        clicked_color = int(grid[grid_y, grid_x])

        if clicked_color == BACKGROUND:
            return

        axis_type = self._get_current_axis()
        mirror_fn = MIRROR_FUNCTIONS[axis_type]

        mirrors = mirror_fn(grid_x, grid_y, w, h)
        state_changed = False
        for mx, my in mirrors:
            if 0 <= mx < w and 0 <= my < h:
                if grid[my, mx] != clicked_color:
                    grid[my, mx] = clicked_color
                    state_changed = True

        if state_changed:
            self._update_sprite_from_grid()

        if check_symmetry(grid, axis_type):
            self.next_level()

    def _update_sprite_from_grid(self) -> None:
        idx = self.current_level.get_data("grid_index")
        if idx is not None:
            grid = self._grids[idx]
            sprites = self.current_level.get_sprites_by_name(f"grid_{idx}")
            if sprites:
                sprites[0].pixels = np.array(grid, dtype=np.int8)


C_HUD_LIFE = 4
C_HUD_DARK = 3
C_HUD_BAR = 14


def _draw_hud_background(frame: np.ndarray) -> None:
    for y in range(HUD_TOP, DISPLAY_SIZE):
        for x in range(DISPLAY_SIZE):
            frame[y, x] = BACKGROUND


def _draw_lives(frame: np.ndarray, lives: int) -> None:
    dot_size = 1
    gap = 1
    total_dots = TOTAL_LIVES
    bar_width = total_dots * dot_size + (total_dots - 1) * gap
    bar_x = DISPLAY_SIZE - bar_width - 1
    bar_y = HUD_TOP

    for i in range(total_dots):
        px = bar_x + i * (dot_size + gap)
        if 0 <= px < DISPLAY_SIZE and 0 <= bar_y < DISPLAY_SIZE:
            frame[bar_y, px] = C_HUD_LIFE if i < lives else C_HUD_DARK


def _draw_progress_bar(frame: np.ndarray, moves_used: int, max_moves: int) -> None:
    bar_y = HUD_TOP + 1
    bar_x_start = 1
    bar_x_end = DISPLAY_SIZE - bar_x_start - 6
    bar_width = bar_x_end - bar_x_start

    moves_remaining = max(0, max_moves - moves_used)
    fill_length = int((moves_remaining / max_moves) * bar_width) if max_moves > 0 else 0

    if 0 <= bar_y < DISPLAY_SIZE:
        for x in range(bar_x_start, bar_x_start + fill_length):
            if 0 <= x < DISPLAY_SIZE:
                frame[bar_y, x] = C_HUD_BAR
        for x in range(bar_x_start + fill_length, bar_x_end):
            if 0 <= x < DISPLAY_SIZE:
                frame[bar_y, x] = C_HUD_DARK


_FONT_3x5: Dict[str, List[int]] = {
    "G": [0b111, 0b100, 0b101, 0b101, 0b111],
    "A": [0b010, 0b101, 0b111, 0b101, 0b101],
    "M": [0b101, 0b111, 0b111, 0b101, 0b101],
    "E": [0b111, 0b100, 0b110, 0b100, 0b111],
    "O": [0b111, 0b101, 0b101, 0b101, 0b111],
    "V": [0b101, 0b101, 0b101, 0b101, 0b010],
    "R": [0b110, 0b101, 0b110, 0b101, 0b101],
    " ": [0b000, 0b000, 0b000, 0b000, 0b000],
}

_GAME_OVER_LINE1 = "GAME"
_GAME_OVER_LINE2 = "OVER"

C_GAMEOVER_TEXT = 4
C_GAMEOVER_BG = BACKGROUND


def _draw_game_over(frame: np.ndarray) -> None:
    glyph_w, glyph_h = 3, 5
    gap = 1

    def _line_width(text: str) -> int:
        return len(text) * (glyph_w + gap) - gap

    line1_w = _line_width(_GAME_OVER_LINE1)
    line2_w = _line_width(_GAME_OVER_LINE2)
    panel_w = max(line1_w, line2_w) + 6
    line_gap = 3
    panel_h = glyph_h * 2 + line_gap + 6

    panel_x = (DISPLAY_SIZE - panel_w) // 2
    panel_y = (DISPLAY_SIZE - panel_h) // 2

    for py in range(panel_y, panel_y + panel_h):
        for px in range(panel_x, panel_x + panel_w):
            if 0 <= px < DISPLAY_SIZE and 0 <= py < DISPLAY_SIZE:
                frame[py, px] = C_GAMEOVER_BG

    for px in range(panel_x, panel_x + panel_w):
        if 0 <= px < DISPLAY_SIZE:
            if 0 <= panel_y < DISPLAY_SIZE:
                frame[panel_y, px] = C_HUD_DARK
            if 0 <= panel_y + panel_h - 1 < DISPLAY_SIZE:
                frame[panel_y + panel_h - 1, px] = C_HUD_DARK
    for py in range(panel_y, panel_y + panel_h):
        if 0 <= py < DISPLAY_SIZE:
            if 0 <= panel_x < DISPLAY_SIZE:
                frame[py, panel_x] = C_HUD_DARK
            if 0 <= panel_x + panel_w - 1 < DISPLAY_SIZE:
                frame[py, panel_x + panel_w - 1] = C_HUD_DARK

    def _draw_line(text: str, text_y: int) -> None:
        line_w = _line_width(text)
        text_x = (DISPLAY_SIZE - line_w) // 2
        for ci, ch in enumerate(text):
            glyph = _FONT_3x5.get(ch, _FONT_3x5[" "])
            gx0 = text_x + ci * (glyph_w + gap)
            for row_i, row_bits in enumerate(glyph):
                for col_i in range(glyph_w):
                    bit = (row_bits >> (glyph_w - 1 - col_i)) & 1
                    if bit:
                        px = gx0 + col_i
                        py = text_y + row_i
                        if 0 <= px < DISPLAY_SIZE and 0 <= py < DISPLAY_SIZE:
                            frame[py, px] = C_GAMEOVER_TEXT

    text_y1 = panel_y + 3
    text_y2 = text_y1 + glyph_h + line_gap
    _draw_line(_GAME_OVER_LINE1, text_y1)
    _draw_line(_GAME_OVER_LINE2, text_y2)


C_CURSOR = 5


def _draw_cursor_box(
    frame: np.ndarray, cursor_x: int, cursor_y: int, gw: int, gh: int
) -> None:
    scale = min(64 // gw, 64 // gh)
    x_off = (64 - gw * scale) // 2
    y_off = (64 - gh * scale) // 2

    px_left = x_off + cursor_x * scale
    px_right = px_left + scale - 1
    px_top = y_off + cursor_y * scale
    px_bottom = px_top + scale - 1

    for px in range(px_left, px_right + 1):
        if 0 <= px < 64:
            if 0 <= px_top < 64:
                frame[px_top, px] = C_CURSOR
            if 0 <= px_bottom < 64:
                frame[px_bottom, px] = C_CURSOR

    for py in range(px_top, px_bottom + 1):
        if 0 <= py < 64:
            if 0 <= px_left < 64:
                frame[py, px_left] = C_CURSOR
            if 0 <= px_right < 64:
                frame[py, px_right] = C_CURSOR


class _GridAxisOverlay(RenderableUserDisplay):
    def __init__(self, game: Ms01) -> None:
        self._game = game

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        game = self._game
        idx = game.level_index
        if idx >= len(game._grid_sizes):
            return frame

        gw, gh = game._grid_sizes[idx]
        axis = game._axis_types[idx]

        scale = min(64 // gw, 64 // gh)
        x_off = (64 - gw * scale) // 2
        y_off = (64 - gh * scale) // 2

        _draw_hud_background(frame)

        moves_used = game._moves_used[idx]
        max_moves = LEVEL_MAX_MOVES[idx]
        _draw_progress_bar(frame, moves_used, max_moves)

        _draw_lives(frame, game._lives)

        _draw_cursor_box(frame, game._cursor_x, game._cursor_y, gw, gh)

        grid_y_end = min(HUD_TOP, y_off + gh * scale)
        for i in range(gw + 1):
            px = x_off + i * scale
            if 0 <= px < 64:
                for py in range(y_off, grid_y_end):
                    frame[py, px] = GRID_LINE_COLOR

        for i in range(gh + 1):
            py = y_off + i * scale
            if 0 <= py < HUD_TOP:
                for px in range(x_off, min(64, x_off + gw * scale)):
                    frame[py, px] = GRID_LINE_COLOR

        self._draw_axis(frame, axis, gw, gh, scale, x_off, y_off)

        if game._state == EngineGameState.GAME_OVER and game._lives <= 0:
            _draw_game_over(frame)

        return frame

    @staticmethod
    def _draw_axis(
        frame: np.ndarray,
        axis: str,
        gw: int,
        gh: int,
        scale: int,
        x_off: int,
        y_off: int,
    ) -> None:
        x0 = x_off
        y0 = y_off
        w = gw * scale
        h = gh * scale

        if axis == "vertical":
            mx = x0 + w // 2
            for py in range(y0, y0 + h):
                if (py // 3) % 2 == 0:
                    if 0 <= mx < 64 and 0 <= py < 64:
                        frame[py, mx] = AXIS_LINE_COLOR

        elif axis == "horizontal":
            my = y0 + h // 2
            for px in range(x0, x0 + w):
                if (px // 3) % 2 == 0:
                    if 0 <= px < 64 and 0 <= my < 64:
                        frame[my, px] = AXIS_LINE_COLOR

        elif axis == "dual":
            mx = x0 + w // 2
            for py in range(y0, y0 + h):
                if (py // 3) % 2 == 0:
                    if 0 <= mx < 64 and 0 <= py < 64:
                        frame[py, mx] = AXIS_LINE_COLOR
            my = y0 + h // 2
            for px in range(x0, x0 + w):
                if (px // 3) % 2 == 0:
                    if 0 <= px < 64 and 0 <= my < 64:
                        frame[my, px] = AXIS_LINE_COLOR

        elif axis == "diagonal_main":
            n = min(w, h)
            for i in range(n):
                px = x0 + i
                py = y0 + i
                if (i // 3) % 2 == 0:
                    if 0 <= px < 64 and 0 <= py < 64:
                        frame[py, px] = AXIS_LINE_COLOR

        elif axis == "diagonal_anti":
            n = min(w, h)
            for i in range(n):
                px = x0 + w - 1 - i
                py = y0 + i
                if (i // 3) % 2 == 0:
                    if 0 <= px < 64 and 0 <= py < 64:
                        frame[py, px] = AXIS_LINE_COLOR


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


def _frame_to_png(frame: Any) -> bytes:
    arr = np.array(frame[0]) if isinstance(frame, list) else np.array(frame)
    h, w = arr.shape
    raw = bytearray()
    for y in range(h):
        raw.append(0)
        for x in range(w):
            ci = max(0, min(int(arr[y, x]), len(_ARC_PALETTE) - 1))
            r, g, b = _ARC_PALETTE[ci]
            raw.extend([r, g, b])
    compressed = zlib.compress(bytes(raw))

    def _chunk(ct: bytes, d: bytes) -> bytes:
        c = ct + d
        return (
            struct.pack(">I", len(d))
            + c
            + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", compressed)
        + _chunk(b"IEND", b"")
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


class PuzzleEnvironment:
    ACTION_MAP: Dict[str, int] = {
        "up": 1,
        "down": 2,
        "left": 3,
        "right": 4,
        "select": 5,
        "undo": 7,
        "reset": 0,
    }

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

    def __init__(self, seed: int = 0) -> None:
        self._engine = Ms01(seed=seed)
        self._total_turns = 0
        self._last_action_was_reset = False

    def reset(self) -> GameState:
        e = self._engine
        e._game_over = False
        self._total_turns = 0
        game_won = hasattr(e, "_state") and getattr(e._state, "name", "") == "WIN"
        if game_won or self._last_action_was_reset:
            e._consecutive_resets = 0
            e.full_reset()
        else:
            e._consecutive_resets = 0
            e.level_reset()
        self._last_action_was_reset = True
        return self._build_game_state()

    def get_actions(self) -> List[str]:
        if self._engine._game_over:
            return ["reset"]
        return ["reset", "up", "down", "left", "right", "select", "undo"]

    def get_state(self) -> GameState:
        return self._build_game_state()

    def is_done(self) -> bool:
        e = self._engine
        if e._game_over:
            return True
        if getattr(e, "_state", None) == EngineGameState.WIN:
            return True
        if getattr(e, "_terminated", False):
            return True
        return False

    def step(self, action: str) -> StepResult:
        e = self._engine

        if action == "reset":
            state = self.reset()
            return StepResult(
                state=state, reward=0.0, done=self.is_done(), info={"action": "reset"}
            )

        if action.startswith("click"):
            parts = action.split()
            if len(parts) == 3:
                gx, gy = int(parts[1]), int(parts[2])
                idx = e._current_level_index
                gw, gh = e._grid_sizes[idx]
                scale = min(64 // gw, 64 // gh)
                x_off = (64 - gw * scale) // 2
                y_off = (64 - gh * scale) // 2
                dx = x_off + gx * scale + scale // 2
                dy = y_off + gy * scale + scale // 2
                action_input = ActionInput(
                    id=GameAction.ACTION6, data={"x": dx, "y": dy}
                )
            else:
                return StepResult(
                    state=self._build_game_state(),
                    reward=0.0,
                    done=False,
                    info={"action": action},
                )
        elif action not in self.ACTION_MAP:
            raise ValueError(
                f"Invalid action '{action}'. "
                f"Must be one of {list(self.ACTION_MAP.keys())}"
            )
        else:
            game_action_id = self.ACTION_MAP[action]
            action_map = {
                1: GameAction.ACTION1,
                2: GameAction.ACTION2,
                3: GameAction.ACTION3,
                4: GameAction.ACTION4,
                5: GameAction.ACTION5,
                7: GameAction.ACTION7,
            }
            game_action = action_map[game_action_id]
            action_input = ActionInput(id=game_action)

        self._last_action_was_reset = False
        self._total_turns += 1
        info: Dict = {"action": action}

        level_before = e._current_level_index
        frame = e.perform_action(action_input, raw=True)
        level_after = e._current_level_index

        game_won = frame and frame.state and frame.state.name == "WIN"
        level_advanced = level_after > level_before
        per_level_reward = 1.0 / len(e._levels)

        if game_won:
            info["reason"] = "game_complete"
            return StepResult(
                state=self._build_game_state(done=True, info=info),
                reward=per_level_reward,
                done=True,
                info=info,
            )

        if level_advanced:
            info["reason"] = "level_complete"
            return StepResult(
                state=self._build_game_state(done=False, info=info),
                reward=per_level_reward,
                done=False,
                info=info,
            )

        if e._game_over:
            info["reason"] = "death"
            return StepResult(
                state=self._build_game_state(done=True, info=info),
                reward=0.0,
                done=True,
                info=info,
            )

        return StepResult(
            state=self._build_game_state(done=False, info=info),
            reward=0.0,
            done=False,
            info=info,
        )

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError(f"Unsupported render mode: {mode}")
        e = self._engine
        index_grid = e.camera.render(e.current_level.get_sprites())
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        for idx, color in enumerate(self.ARC_PALETTE):
            mask = index_grid == idx
            rgb[mask] = color
        return rgb

    def close(self) -> None:
        self._engine = None

    def _build_text_obs(self) -> str:
        e = self._engine
        idx = e._current_level_index
        if idx >= len(e._grid_sizes):
            idx = len(e._grid_sizes) - 1
        gw, gh = e._grid_sizes[idx]
        grid = e._grids[idx]
        axis = e._axis_types[idx]
        moves_used = e._moves_used[idx]
        max_moves = LEVEL_MAX_MOVES[idx]

        remaining = max(0, max_moves - moves_used)
        header = (
            f"Level:{idx + 1}/{len(LEVEL_CONFIGS)} "
            f"Lives:{e._lives} "
            f"Moves:{remaining}/{max_moves} "
            f"Axis:{axis} "
            f"Cursor:({e._cursor_x},{e._cursor_y})"
        )

        lines = []
        for y in range(gh):
            row = " ".join(f"{int(grid[y, x]):2d}" for x in range(gw))
            lines.append(row)

        grid_text = "\n".join(lines)

        if e._game_over:
            header += " GAME_OVER"

        return header.strip() + "\n" + grid_text

    @staticmethod
    def _encode_png(rgb: np.ndarray) -> bytes:
        h, w, _ = rgb.shape
        raw_rows = []
        for y in range(h):
            raw_rows.append(b"\x00" + rgb[y].tobytes())
        raw_data = b"".join(raw_rows)
        compressed = zlib.compress(raw_data)

        def _chunk(ctype: bytes, data: bytes) -> bytes:
            c = ctype + data
            return (
                struct.pack(">I", len(data))
                + c
                + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
            )

        ihdr_data = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
        png = b"\x89PNG\r\n\x1a\n"
        png += _chunk(b"IHDR", ihdr_data)
        png += _chunk(b"IDAT", compressed)
        png += _chunk(b"IEND", b"")
        return png

    def _build_image_bytes(self) -> Optional[bytes]:
        e = self._engine
        rendered = e.camera.render(e.current_level.get_sprites())
        if hasattr(rendered, "tolist"):
            grid = rendered.tolist()
        else:
            grid = rendered if rendered else []
        if not grid:
            return None
        arr = np.array(grid, dtype=np.uint8)
        h, w = arr.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for ci, color in enumerate(self.ARC_PALETTE):
            mask = arr == ci
            rgb[mask] = color
        return self._encode_png(rgb)

    def _build_game_state(
        self, done: bool = False, info: Optional[Dict] = None
    ) -> GameState:
        e = self._engine
        valid_actions = self.get_actions() if not done else None
        idx = e._current_level_index
        if idx >= len(e._grid_sizes):
            idx = len(e._grid_sizes) - 1

        return GameState(
            text_observation=self._build_text_obs(),
            image_observation=self._build_image_bytes(),
            valid_actions=valid_actions,
            turn=self._total_turns,
            metadata={
                "total_levels": len(self._engine._levels),
                "level_index": idx,
                "lives": e._lives,
                "max_lives": TOTAL_LIVES,
                "moves_used": e._moves_used[idx],
                "move_limit": LEVEL_MAX_MOVES[idx],
                "axis_type": e._axis_types[idx],
                "grid_width": e._grid_sizes[idx][0],
                "grid_height": e._grid_sizes[idx][1],
                "cursor_x": e._cursor_x,
                "cursor_y": e._cursor_y,
                "game_over": e._game_over,
                "done": done,
                "info": info or {},
                "levels_completed": getattr(self._engine, "_score", 0),
            },
        )


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
        if self._env is None:
            raise RuntimeError("Call reset() before _get_obs()")
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
