import random
import struct
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from arcengine import ActionInput, ARCBaseGame, GameAction, GameState as EngineGameState
from arcengine.camera import Camera, RenderableUserDisplay
from arcengine.level import Level
from arcengine.sprites import Sprite

C_MIRROR_SLASH = 0
C_MIRROR_BACK = 1
C_BEAM_BLUE = 2
C_WALL = 3
C_GRID_LINE = 4
C_BG = 5
C_FILTER = 6
C_CURSOR = 7
C_DIAMOND = 7
C_SRC_RED = 8
C_SRC_BLUE = 9
C_TARGET_LIT = 10
C_BEAM_RED = 11
C_BLACKHOLE = 12
C_TARGET_RED = 13
C_TARGET_ANY = 14
C_SPLITTER = 15

SOURCES = {C_SRC_BLUE, C_SRC_RED}
TARGETS = {C_TARGET_ANY, C_TARGET_RED, C_TARGET_LIT}
TARGET_UNLIT = {C_TARGET_ANY, C_TARGET_RED}
BEAM_COLORS = {C_BEAM_BLUE, C_BEAM_RED}
CLICKABLE = {C_BG, C_MIRROR_SLASH, C_MIRROR_BACK}
BLOCKERS = {C_WALL, C_SRC_BLUE, C_SRC_RED}
STATIC = {
    C_WALL,
    C_SRC_BLUE,
    C_SRC_RED,
    C_TARGET_ANY,
    C_TARGET_RED,
    C_TARGET_LIT,
    C_MIRROR_SLASH,
    C_MIRROR_BACK,
    C_FILTER,
    C_SPLITTER,
    C_BLACKHOLE,
}

DISPLAY_SIZE = 64
HUD_HEIGHT = 3
HUD_TOP = DISPLAY_SIZE - HUD_HEIGHT


def _grid_layout(gw: int, gh: int) -> Tuple[int, int, int]:
    scale = min(DISPLAY_SIZE // gw, DISPLAY_SIZE // gh)
    x_off = (DISPLAY_SIZE - gw * scale) // 2
    y_off = (DISPLAY_SIZE - gh * scale) // 2
    return scale, x_off, y_off


DIR_RIGHT = (1, 0)
DIR_LEFT = (-1, 0)
DIR_DOWN = (0, 1)
DIR_UP = (0, -1)

SOURCE_DIRECTIONS: Dict[int, Tuple[int, int]] = {
    C_SRC_BLUE: DIR_RIGHT,
    C_SRC_RED: DIR_DOWN,
}

SLASH_REFLECT: Dict[Tuple[int, int], Tuple[int, int]] = {
    DIR_RIGHT: DIR_UP,
    DIR_DOWN: DIR_LEFT,
    DIR_LEFT: DIR_DOWN,
    DIR_UP: DIR_RIGHT,
}
BACK_REFLECT: Dict[Tuple[int, int], Tuple[int, int]] = {
    DIR_RIGHT: DIR_DOWN,
    DIR_DOWN: DIR_RIGHT,
    DIR_LEFT: DIR_UP,
    DIR_UP: DIR_LEFT,
}


def propagate_light(grid: np.ndarray, base_grid: np.ndarray) -> bool:
    h, w = base_grid.shape
    np.copyto(grid, base_grid)

    hit_blackhole = False

    sources: List[Tuple[int, int, Tuple[int, int], str]] = []
    for y in range(h):
        for x in range(w):
            cell = int(base_grid[y, x])
            if cell in SOURCE_DIRECTIONS:
                dx, dy = SOURCE_DIRECTIONS[cell]
                color = "blue" if cell == C_SRC_BLUE else "red"
                sources.append((x, y, (dx, dy), color))

    beam_queue: List[Tuple[int, int, Tuple[int, int], str]] = []
    for sx, sy, direction, color in sources:
        nx, ny = sx + direction[0], sy + direction[1]
        beam_queue.append((nx, ny, direction, color))

    visited: Set[Tuple[int, int, int, int]] = set()

    while beam_queue:
        bx, by, (dx, dy), beam_color = beam_queue.pop(0)

        steps = 0
        max_steps = w * h * 2

        while steps < max_steps:
            steps += 1

            if bx < 0 or bx >= w or by < 0 or by >= h:
                break

            state_key = (bx, by, dx, dy)
            if state_key in visited:
                break
            visited.add(state_key)

            cell = int(base_grid[by, bx])

            if cell == C_BLACKHOLE:
                hit_blackhole = True
                break

            if cell in BLOCKERS:
                break

            if cell in TARGET_UNLIT:
                if cell == C_TARGET_ANY:
                    grid[by, bx] = C_TARGET_LIT
                elif cell == C_TARGET_RED:
                    if beam_color == "red":
                        grid[by, bx] = C_TARGET_LIT
                break

            if cell == C_TARGET_LIT:
                break

            if cell == C_MIRROR_SLASH:
                new_dir = SLASH_REFLECT[(dx, dy)]
                dx, dy = new_dir
                bx += dx
                by += dy
                continue

            if cell == C_MIRROR_BACK:
                new_dir = BACK_REFLECT[(dx, dy)]
                dx, dy = new_dir
                bx += dx
                by += dy
                continue

            if cell == C_DIAMOND:
                if beam_color == "blue":
                    grid[by, bx] = C_BEAM_BLUE
                elif beam_color == "red":
                    grid[by, bx] = C_BEAM_RED
                bx += dx
                by += dy
                continue

            if cell == C_FILTER:
                if beam_color == "blue":
                    beam_color = "red"
                elif beam_color == "red":
                    beam_color = "blue"
                bx += dx
                by += dy
                continue

            if cell == C_SPLITTER:
                if dx != 0:
                    perp1 = (0, -1)
                    perp2 = (0, 1)
                else:
                    perp1 = (-1, 0)
                    perp2 = (1, 0)

                nx1, ny1 = bx + perp1[0], by + perp1[1]
                nx2, ny2 = bx + perp2[0], by + perp2[1]
                beam_queue.append((nx1, ny1, perp1, beam_color))
                beam_queue.append((nx2, ny2, perp2, beam_color))

                bx += dx
                by += dy
                continue

            if cell == C_BG or cell in BEAM_COLORS:
                if beam_color == "blue":
                    grid[by, bx] = C_BEAM_BLUE
                elif beam_color == "red":
                    grid[by, bx] = C_BEAM_RED

            bx += dx
            by += dy

    return hit_blackhole


def check_all_targets_lit(grid: np.ndarray, base_grid: np.ndarray) -> bool:
    h, w = base_grid.shape
    for y in range(h):
        for x in range(w):
            cell = int(base_grid[y, x])
            if cell in TARGET_UNLIT:
                if int(grid[y, x]) != C_TARGET_LIT:
                    return False
            elif cell == C_DIAMOND:
                display_val = int(grid[y, x])
                if display_val not in BEAM_COLORS:
                    return False
    return True


def find_clickable_cells(base_grid: np.ndarray) -> List[Tuple[int, int]]:
    h, w = base_grid.shape
    result = []
    for y in range(h):
        for x in range(w):
            if int(base_grid[y, x]) in CLICKABLE:
                result.append((x, y))
    return result


def generate_level_1(rng: random.Random) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    w, h = 10, 10
    grid = np.full((h, w), C_BG, dtype=np.int8)

    grid[3, 1] = C_SRC_BLUE

    grid[6, 7] = C_TARGET_ANY

    for row in range(1, 6):
        grid[row, 4] = C_WALL

    for col in range(2, 7):
        grid[8, col] = C_WALL

    _seq = [
        (3, 3),
        (3, 3),
        (3, 6),
        (3, 6),
    ]

    return grid, _seq


def generate_level_2(rng: random.Random) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    w, h = 12, 12
    grid = np.full((h, w), C_BG, dtype=np.int8)

    grid[3, 1] = C_SRC_BLUE

    grid[7, 9] = C_TARGET_ANY

    for row in range(1, 6):
        grid[row, 5] = C_WALL

    grid[1, 9] = C_SRC_RED

    grid[10, 3] = C_TARGET_RED

    for col in range(6, 11):
        grid[6, col] = C_WALL

    grid[:, 5] = C_BG
    for row in range(1, 5):
        grid[row, 5] = C_WALL

    _seq = [
        (4, 3),
        (4, 3),
        (4, 7),
        (4, 7),
        (9, 5),
        (3, 5),
    ]

    return grid, _seq


def generate_level_3(rng: random.Random) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    w, h = 16, 16
    grid = np.full((h, w), C_BG, dtype=np.int8)

    grid[3, 1] = C_SRC_BLUE

    grid[1, 14] = C_SRC_RED

    grid[10, 8] = C_TARGET_ANY

    grid[13, 6] = C_TARGET_RED

    grid[6, 6] = C_DIAMOND

    for row in range(1, 6):
        grid[row, 5] = C_WALL

    for col in range(2, 8):
        grid[7, col] = C_WALL

    for col in range(10, 15):
        grid[9, col] = C_WALL

    _seq = [
        (4, 3),
        (4, 3),
        (4, 6),
        (4, 6),
        (8, 6),
        (8, 6),
        (14, 8),
        (6, 8),
    ]

    return grid, _seq


def generate_level_4(rng: random.Random) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    w, h = 16, 15
    grid = np.full((h, w), C_BG, dtype=np.int8)

    grid[3, 1] = C_SRC_BLUE

    grid[13, 8] = C_TARGET_ANY

    grid[13, 12] = C_TARGET_RED

    for col in range(3, 13):
        grid[6, col] = C_WALL

    grid[9, 8] = C_SPLITTER

    grid[9, 12] = C_FILTER

    grid[12, 2] = C_BLACKHOLE

    grid[9, 5] = C_DIAMOND

    grid[12, 7] = C_DIAMOND

    _seq = [
        (2, 9),
        (2, 9),
        (8, 11),
        (7, 11),
        (7, 13),
        (7, 13),
        (14, 9),
        (14, 9),
        (14, 13),
        (2, 3),
        (2, 3),
    ]

    return grid, _seq


def generate_level_5(rng: random.Random) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    w, h = 20, 19
    grid = np.full((h, w), C_BG, dtype=np.int8)

    grid[2, 1] = C_SRC_BLUE

    grid[17, 10] = C_TARGET_ANY
    grid[18, 3] = C_TARGET_RED
    grid[17, 15] = C_TARGET_ANY

    for col in range(4, 17):
        grid[6, col] = C_WALL

    grid[12, 10] = C_SPLITTER

    grid[15, 3] = C_FILTER

    grid[16, 2] = C_BLACKHOLE

    grid[12, 18] = C_BLACKHOLE

    grid[10, 6] = C_DIAMOND

    grid[15, 10] = C_DIAMOND

    _seq = [
        (2, 10),
        (2, 10),
        (10, 10),
        (10, 10),
        (3, 12),
        (15, 12),
        (15, 12),
        (2, 2),
        (2, 2),
    ]

    return grid, _seq


LEVEL_GENERATORS = [
    generate_level_1,
    generate_level_2,
    generate_level_3,
    generate_level_4,
    generate_level_5,
]

LEVEL_GRID_SIZES = [
    (10, 10),
    (12, 12),
    (16, 16),
    (16, 15),
    (20, 19),
]

LEVEL_MAX_MOVES = [600, 900, 1200, 1650, 1350]

TOTAL_LIVES = 3


def grid_to_display(gx: int, gy: int, gw: int, gh: int) -> Tuple[int, int]:
    scale, x_off, y_off = _grid_layout(gw, gh)
    return x_off + gx * scale + scale // 2, y_off + gy * scale + scale // 2


class Ms03(ARCBaseGame):
    _grids: List[np.ndarray]
    _base_grids: List[np.ndarray]
    _grid_sizes: List[Tuple[int, int]]
    _rng: random.Random
    _moves_used: List[int]
    _lives: int
    _cursor_x: int
    _cursor_y: int

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self._rng = random.Random(seed)

        self._grids = []
        self._base_grids = []
        self._grid_sizes = []
        self._moves_used = [0] * len(LEVEL_GRID_SIZES)
        self._lives = TOTAL_LIVES
        self._cursor_x = 0
        self._cursor_y = 0
        self._undo_base = None
        self._undo_cursor = None

        levels: List[Level] = []
        for i, gen_fn in enumerate(LEVEL_GENERATORS):
            gw, gh = LEVEL_GRID_SIZES[i]
            grid, _ = gen_fn(self._rng)

            base_grid = np.copy(grid)
            display_grid = np.copy(grid)

            propagate_light(display_grid, base_grid)

            self._grids.append(display_grid)
            self._base_grids.append(base_grid)
            self._grid_sizes.append((gw, gh))

            sprite = Sprite(
                pixels=display_grid.tolist(),
                name=f"grid_{i}",
                x=0,
                y=0,
                layer=0,
            )

            level = Level(
                sprites=[sprite],
                grid_size=(gw, gh),
                name=f"Level {i + 1}",
                data={"grid_index": i},
            )
            levels.append(level)

        camera = Camera(
            x=0,
            y=0,
            width=levels[0].grid_size[0],
            height=levels[0].grid_size[1],
            background=C_BG,
            letter_box=C_BG,
        )

        super().__init__(
            game_id="ms03",
            levels=levels,
            camera=camera,
            debug=False,
            win_score=len(levels),
            available_actions=[0, 1, 2, 3, 4, 5, 6, 7],
            seed=seed,
        )

        overlay = _LightBridgeOverlay(self)
        self.camera.replace_interface([overlay])

    @property
    def level_index(self) -> int:
        return self._current_level_index

    def on_set_level(self, level: Level) -> None:
        idx = level.get_data("grid_index")
        if idx is not None and idx < len(self._grid_sizes):
            gw, gh = self._grid_sizes[idx]
            self.camera.resize(gw, gh)
            self.camera.x = 0
            self.camera.y = 0
            self._cursor_x = self._rng.randint(0, gw - 1)
            self._cursor_y = self._rng.randint(0, gh - 1)
            self._moves_used[idx] = 0
            self._undo_base = None
            self._undo_cursor = None

    def full_reset(self) -> None:
        self._undo_base = None
        self._undo_cursor = None
        super().full_reset()
        self._moves_used = [0] * len(LEVEL_GRID_SIZES)
        self._lives = TOTAL_LIVES
        for i in range(len(self._grids)):
            rng = random.Random(self._seed)
            gen_fn = LEVEL_GENERATORS[i]
            base, _ = gen_fn(rng)
            self._base_grids[i] = base
            self._grids[i] = np.copy(base)
            propagate_light(self._grids[i], self._base_grids[i])
        self._sync_current_level_sprite()

    def level_reset(self, preserve_moves: bool = False) -> None:
        idx = self._current_level_index
        saved_moves = self._moves_used[idx]
        self._undo_base = None
        self._undo_cursor = None
        super().level_reset()
        if preserve_moves:
            self._moves_used[idx] = saved_moves
        rng = random.Random(self._seed)
        gen_fn = LEVEL_GENERATORS[idx]
        base, _ = gen_fn(rng)
        self._base_grids[idx] = base
        self._grids[idx] = np.copy(base)
        propagate_light(self._grids[idx], self._base_grids[idx])
        self._sync_current_level_sprite()

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
        raise ValueError("No grid index found")

    def _get_current_base_grid(self) -> np.ndarray:
        idx = self.current_level.get_data("grid_index")
        if idx is not None:
            return self._base_grids[idx]
        raise ValueError("No grid index found")

    def _consume_move(self) -> bool:
        level_idx = self._current_level_index
        self._moves_used[level_idx] += 1
        if self._moves_used[level_idx] > LEVEL_MAX_MOVES[level_idx]:
            self._lives -= 1
            if self._lives <= 0:
                self.lose()
            else:
                self.level_reset(preserve_moves=False)
            return True
        return False

    def _save_undo(self) -> None:
        idx = self._current_level_index
        self._undo_base = np.copy(self._base_grids[idx])
        self._undo_cursor = (self._cursor_x, self._cursor_y)

    def _restore_undo(self) -> None:
        if self._undo_base is None:
            return
        idx = self._current_level_index
        self._base_grids[idx] = self._undo_base
        self._grids[idx] = np.copy(self._undo_base)
        propagate_light(self._grids[idx], self._base_grids[idx])
        self._update_sprite_from_grid()
        if self._undo_cursor is not None:
            self._cursor_x, self._cursor_y = self._undo_cursor
        self._undo_base = None
        self._undo_cursor = None

    def _handle_undo(self) -> None:
        if self._undo_base is not None:
            self._restore_undo()
        self._consume_move()

    def step(self) -> None:
        action = self.action

        if action.id == GameAction.RESET:
            if self._state in (EngineGameState.GAME_OVER, EngineGameState.WIN):
                self.full_reset()
            else:
                self.level_reset()
            self.complete_action()
            return

        if self._state in (EngineGameState.GAME_OVER, EngineGameState.WIN):
            self.complete_action()
            return

        if action.id == GameAction.ACTION7:
            self._handle_undo()
        elif action.id == GameAction.ACTION1:
            self._save_undo()
            self._move_cursor(0, -1)
            self._consume_move()
        elif action.id == GameAction.ACTION2:
            self._save_undo()
            self._move_cursor(0, 1)
            self._consume_move()
        elif action.id == GameAction.ACTION3:
            self._save_undo()
            self._move_cursor(-1, 0)
            self._consume_move()
        elif action.id == GameAction.ACTION4:
            self._save_undo()
            self._move_cursor(1, 0)
            self._consume_move()
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

    def _handle_click(self, action: ActionInput) -> None:
        data = action.data
        display_x = data.get("x", -1)
        display_y = data.get("y", -1)

        if display_x == -1 and display_y == -1:
            grid_x, grid_y = self._cursor_x, self._cursor_y
        else:
            idx = self._current_level_index
            gw, gh = self._grid_sizes[idx]
            scale, x_off, y_off = _grid_layout(gw, gh)
            dx_px = int(display_x) - x_off
            dy_px = int(display_y) - y_off
            if scale <= 0:
                return
            gx = dx_px // scale
            gy = dy_px // scale
            if gx < 0 or gx >= gw or gy < 0 or gy >= gh:
                return
            grid_x, grid_y = gx, gy
            self._cursor_x, self._cursor_y = grid_x, grid_y

        base_grid = self._get_current_base_grid()
        h, w = base_grid.shape
        level_idx = self._current_level_index

        if grid_x < 0 or grid_x >= w or grid_y < 0 or grid_y >= h:
            return

        cell = int(base_grid[grid_y, grid_x])
        if cell not in CLICKABLE:
            return

        self._save_undo()

        if self._consume_move():
            return

        if cell == C_BG:
            base_grid[grid_y, grid_x] = C_MIRROR_SLASH
        elif cell == C_MIRROR_SLASH:
            base_grid[grid_y, grid_x] = C_MIRROR_BACK
        elif cell == C_MIRROR_BACK:
            base_grid[grid_y, grid_x] = C_BG

        grid = self._get_current_grid()
        hit_blackhole = propagate_light(grid, base_grid)

        self._update_sprite_from_grid()

        if hit_blackhole:
            self._lives -= 1
            if self._lives <= 0:
                self.lose()
            else:
                self.level_reset(preserve_moves=False)
            return

        if check_all_targets_lit(grid, base_grid):
            self.next_level()

    def _update_sprite_from_grid(self) -> None:
        idx = self.current_level.get_data("grid_index")
        if idx is not None:
            grid = self._grids[idx]
            sprites = self.current_level.get_sprites_by_name(f"grid_{idx}")
            if sprites:
                sprites[0].pixels = np.array(grid, dtype=np.int8)


def _draw_cursor_box(
    frame: np.ndarray, cursor_x: int, cursor_y: int, gw: int, gh: int
) -> None:
    scale, x_off, y_off = _grid_layout(gw, gh)

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


def _draw_hud_background(frame: np.ndarray) -> None:
    for y in range(HUD_TOP, DISPLAY_SIZE):
        for x in range(DISPLAY_SIZE):
            frame[y, x] = C_BG


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
            if i < lives:
                frame[bar_y, px] = C_SRC_RED
            else:
                frame[bar_y, px] = C_WALL


def _draw_progress_bar(frame: np.ndarray, moves_used: int, max_moves: int) -> None:
    bar_y = HUD_TOP + 1
    bar_x_start = 1
    bar_x_end = DISPLAY_SIZE - bar_x_start - 6
    bar_width = bar_x_end - bar_x_start

    moves_remaining = max(0, max_moves - moves_used)
    fill_length = int((moves_remaining / max_moves) * bar_width)

    if 0 <= bar_y < DISPLAY_SIZE:
        for x in range(bar_x_start, bar_x_start + fill_length):
            if 0 <= x < DISPLAY_SIZE:
                frame[bar_y, x] = C_TARGET_ANY

        for x in range(bar_x_start + fill_length, bar_x_end):
            if 0 <= x < DISPLAY_SIZE:
                frame[bar_y, x] = C_WALL


def _draw_diamond_shape(
    frame: np.ndarray, px_left: int, px_top: int, scale: int, collected: bool
) -> None:
    if scale <= 1:
        return

    fill_color = C_TARGET_LIT if collected else C_MIRROR_SLASH

    for dy in range(scale):
        for dx in range(scale):
            px, py = px_left + dx, px_top + dy
            if 0 <= px < 64 and 0 <= py < 64:
                frame[py, px] = C_BG

    cx = px_left + scale // 2
    cy = px_top + scale // 2
    if 0 <= cx < 64 and 0 <= cy < 64:
        frame[cy, cx] = fill_color


def _draw_round_blackhole(
    frame: np.ndarray, px_left: int, px_top: int, scale: int
) -> None:
    if scale <= 2:
        return

    for dy in range(scale):
        for dx in range(scale):
            px, py = px_left + dx, px_top + dy
            if 0 <= px < 64 and 0 <= py < 64:
                if dy == 0 or dx == 0:
                    frame[py, px] = C_GRID_LINE
                else:
                    frame[py, px] = C_BLACKHOLE


class _LightBridgeOverlay(RenderableUserDisplay):
    def __init__(self, game: Ms03) -> None:
        self._game = game

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        game = self._game
        idx = game.level_index
        if idx >= len(game._grid_sizes):
            return frame

        gw, gh = game._grid_sizes[idx]
        scale, x_off, y_off = _grid_layout(gw, gh)

        _draw_hud_background(frame)

        grid_px_right = min(DISPLAY_SIZE, x_off + gw * scale)
        grid_px_bottom = min(HUD_TOP, y_off + gh * scale)
        for py in range(y_off, grid_px_bottom):
            for px in range(x_off, grid_px_right):
                val = int(frame[py, px])
                if val == C_TARGET_ANY:
                    frame[py, px] = C_SRC_BLUE
                elif val == C_SRC_RED:
                    frame[py, px] = C_TARGET_RED

        FILLED_CELLS = {
            C_WALL,
            C_SRC_BLUE,
            C_SRC_RED,
            C_BEAM_BLUE,
            C_BEAM_RED,
            C_TARGET_ANY,
            C_TARGET_RED,
            C_TARGET_LIT,
            C_FILTER,
            C_SPLITTER,
        }

        if idx < len(game._grids) and scale >= 3:
            base = game._base_grids[idx]
            disp = game._grids[idx]
            bh, bw = base.shape
            for gy in range(bh):
                for gx in range(bw):
                    cell_val = int(disp[gy, gx])
                    if cell_val not in FILLED_CELLS:
                        continue
                    px_left = x_off + gx * scale
                    px_top = y_off + gy * scale
                    above_val = int(disp[gy - 1, gx]) if gy > 0 else -1
                    above_same = above_val == cell_val
                    if not above_same:
                        for dx in range(scale):
                            px, py = px_left + dx, px_top
                            if 0 <= px < 64 and 0 <= py < 64:
                                frame[py, px] = C_GRID_LINE
                    left_val = int(disp[gy, gx - 1]) if gx > 0 else -1
                    left_same = left_val == cell_val
                    if not left_same:
                        for dy in range(scale):
                            px, py = px_left, px_top + dy
                            if 0 <= px < 64 and 0 <= py < 64:
                                frame[py, px] = C_GRID_LINE

        if idx < len(game._base_grids):
            base = game._base_grids[idx]
            disp = game._grids[idx] if idx < len(game._grids) else None
            bh, bw = base.shape
            for gy in range(bh):
                for gx in range(bw):
                    cell_val = int(base[gy, gx])
                    px_left = x_off + gx * scale
                    px_top = y_off + gy * scale
                    if cell_val == C_BLACKHOLE:
                        _draw_round_blackhole(frame, px_left, px_top, scale)
                    elif cell_val == C_DIAMOND:
                        collected = False
                        if disp is not None:
                            collected = int(disp[gy, gx]) in BEAM_COLORS
                        _draw_diamond_shape(frame, px_left, px_top, scale, collected)

        moves_used = game._moves_used[idx]
        max_moves = LEVEL_MAX_MOVES[idx]
        _draw_progress_bar(frame, moves_used, max_moves)

        _draw_lives(frame, game._lives)

        _draw_cursor_box(frame, game._cursor_x, game._cursor_y, gw, gh)

        grid_y_end = min(HUD_TOP, y_off + gh * scale)
        grid_x_end = min(DISPLAY_SIZE, x_off + gw * scale)
        if scale >= 3:
            for i in range(gw + 1):
                px = x_off + i * scale
                if 0 <= px < DISPLAY_SIZE:
                    for py in range(y_off, grid_y_end):
                        if int(frame[py, px]) == C_BG:
                            frame[py, px] = C_GRID_LINE

            for i in range(gh + 1):
                py = y_off + i * scale
                if 0 <= py < HUD_TOP:
                    for px in range(x_off, grid_x_end):
                        if int(frame[py, px]) == C_BG:
                            frame[py, px] = C_GRID_LINE
        else:
            py = y_off
            if 0 <= py < HUD_TOP:
                for px in range(x_off, grid_x_end):
                    if int(frame[py, px]) == C_BG:
                        frame[py, px] = C_GRID_LINE
            py = y_off + gh * scale
            if 0 <= py < HUD_TOP:
                for px in range(x_off, grid_x_end):
                    if int(frame[py, px]) == C_BG:
                        frame[py, px] = C_GRID_LINE
            px = x_off
            if 0 <= px < DISPLAY_SIZE:
                for py_i in range(y_off, grid_y_end):
                    if int(frame[py_i, px]) == C_BG:
                        frame[py_i, px] = C_GRID_LINE
            px = x_off + gw * scale
            if 0 <= px < DISPLAY_SIZE:
                for py_i in range(y_off, grid_y_end):
                    if int(frame[py_i, px]) == C_BG:
                        frame[py_i, px] = C_GRID_LINE

        return frame


_CELL_CHARS: Dict[int, str] = {
    C_BG: ".",
    C_MIRROR_SLASH: "/",
    C_MIRROR_BACK: "\\",
    C_BEAM_BLUE: "~",
    C_WALL: "#",
    C_GRID_LINE: ".",
    C_FILTER: "F",
    C_DIAMOND: "D",
    C_SRC_RED: "R",
    C_SRC_BLUE: "B",
    C_TARGET_LIT: "*",
    C_BEAM_RED: "r",
    C_BLACKHOLE: "H",
    C_TARGET_RED: "T",
    C_TARGET_ANY: "O",
    C_SPLITTER: "S",
}

_ACTION_MAP: Dict[str, GameAction] = {
    "reset": GameAction.RESET,
    "up": GameAction.ACTION1,
    "down": GameAction.ACTION2,
    "left": GameAction.ACTION3,
    "right": GameAction.ACTION4,
    "select": GameAction.ACTION5,
    "undo": GameAction.ACTION7,
}

_VALID_ACTIONS: List[str] = ["up", "down", "left", "right", "select", "click", "undo"]

_ARC_PALETTE: List[Tuple[int, int, int]] = [
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
    _TOTAL_LEVELS: int = len(LEVEL_GENERATORS)

    def __init__(self, seed: int = 0) -> None:
        self._engine = Ms03(seed=seed)
        self._turn: int = 0
        self._last_action_was_reset: bool = False

    @staticmethod
    def _encode_png(data: bytes, width: int, height: int) -> bytes:
        def _chunk(chunk_type: bytes, chunk_data: bytes) -> bytes:
            c = chunk_type + chunk_data
            crc = struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
            return struct.pack(">I", len(chunk_data)) + c + crc

        header = b"\x89PNG\r\n\x1a\n"
        ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
        raw_rows = bytearray()
        stride = width * 3
        for y in range(height):
            raw_rows.append(0)
            raw_rows.extend(data[y * stride : y * stride + stride])
        compressed = zlib.compress(bytes(raw_rows))
        return (
            header
            + _chunk(b"IHDR", ihdr_data)
            + _chunk(b"IDAT", compressed)
            + _chunk(b"IEND", b"")
        )

    def _build_image_bytes(self) -> Optional[bytes]:
        try:
            index_grid = self._engine.camera.render(
                self._engine.current_level.get_sprites()
            )
            h, w = index_grid.shape[:2]
            rgb = np.zeros((h, w, 3), dtype=np.uint8)
            for idx, color in enumerate(_ARC_PALETTE):
                mask = index_grid == idx
                rgb[mask] = color
            return self._encode_png(rgb.tobytes(), w, h)
        except Exception:
            return None

    def _build_text_observation(self) -> str:
        g = self._engine
        idx = g._current_level_index
        if idx >= len(g._grid_sizes):
            idx = len(g._grid_sizes) - 1
        gw, gh = g._grid_sizes[idx]
        grid = g._grids[idx]
        moves_used = g._moves_used[idx]
        max_moves = LEVEL_MAX_MOVES[idx]
        lives = g._lives

        lines: List[str] = [
            f"Level: {idx + 1}/{len(LEVEL_GENERATORS)}",
            f"Moves: {moves_used}/{max_moves}",
            f"Lives: {lives}/{TOTAL_LIVES}",
            f"Cursor: ({g._cursor_x}, {g._cursor_y})",
        ]

        if g._state == EngineGameState.GAME_OVER:
            lines.append("Status: GAME OVER")
        else:
            remaining = max(0, max_moves - moves_used)
            lines.append(f"Moves remaining: {remaining}")

        lines.append(f"Grid ({gw}x{gh}):")
        for y in range(gh):
            row_chars: List[str] = []
            for x in range(gw):
                cell = int(grid[y, x])
                row_chars.append(_CELL_CHARS.get(cell, "?"))
            lines.append(" ".join(row_chars))

        lines.append(
            "Legend: .=empty #=wall B=blue_src R=red_src "
            "/=mirror \\=mirror O=target *=lit_target "
            "T=red_target D=diamond F=filter S=splitter "
            "H=blackhole ~=blue_beam r=red_beam"
        )

        return "\n".join(lines)

    def _build_metadata(self) -> dict:
        g = self._engine
        idx = g._current_level_index
        if idx >= len(g._grid_sizes):
            idx = len(g._grid_sizes) - 1
        max_moves = LEVEL_MAX_MOVES[idx]
        return {
            "total_levels": self._TOTAL_LEVELS,
            "level": idx + 1,
            "moves_used": g._moves_used[idx],
            "move_limit": max_moves,
            "lives": g._lives,
            "max_lives": TOTAL_LIVES,
            "grid_width": g._grid_sizes[idx][0],
            "grid_height": g._grid_sizes[idx][1],
            "cursor_x": g._cursor_x,
            "cursor_y": g._cursor_y,
        }

    def _make_game_state(self, done: bool = False) -> "GameState":
        image_bytes = self._build_image_bytes()

        game_over = self._engine._state in (
            EngineGameState.GAME_OVER,
            EngineGameState.WIN,
        )
        valid = None if game_over else list(_VALID_ACTIONS)

        return GameState(
            text_observation=self._build_text_observation(),
            image_observation=image_bytes,
            valid_actions=valid,
            turn=self._turn,
            metadata=self._build_metadata(),
        )

    def reset(self) -> "GameState":
        self._turn = 0
        is_won = self._engine._state == EngineGameState.WIN
        if is_won or self._last_action_was_reset:
            self._engine.full_reset()
        self._engine.perform_action(ActionInput(id=GameAction.RESET))
        self._last_action_was_reset = True
        return self._make_game_state()

    def get_actions(self) -> List[str]:
        if self._engine._state in (EngineGameState.GAME_OVER, EngineGameState.WIN):
            return ["reset"]
        return ["reset"] + list(_VALID_ACTIONS)

    def step(self, action: str) -> StepResult:
        action = action.strip().lower()
        self._last_action_was_reset = False

        prev_level = self._engine._current_level_index
        prev_lives = self._engine._lives

        if action.startswith("click"):
            parts = action.split()
            x = int(parts[1]) if len(parts) > 1 else 0
            y = int(parts[2]) if len(parts) > 2 else 0
            self._engine.perform_action(
                ActionInput(id=GameAction.ACTION6, data={"x": x, "y": y})
            )
        else:
            game_action = _ACTION_MAP.get(action)
            if game_action is None:
                raise ValueError(
                    f"Unknown action '{action}'. "
                    f"Valid: {list(_ACTION_MAP.keys()) + ['click X Y']}"
                )
            self._engine.perform_action(ActionInput(id=game_action))

        if action != "reset":
            self._turn += 1

        engine_state = self._engine._state
        done = engine_state in (
            EngineGameState.WIN,
            EngineGameState.GAME_OVER,
        )

        cur_level = self._engine._current_level_index
        cur_lives = self._engine._lives

        total = self._TOTAL_LEVELS
        reward = 0.0
        if engine_state == EngineGameState.WIN:
            reward = 1.0 / total
        elif engine_state == EngineGameState.GAME_OVER:
            reward = 0.0
        elif cur_level > prev_level:
            reward = 1.0 / total

        info: dict = {
            "action": action,
            "engine_state": str(engine_state),
            "level_changed": cur_level != prev_level,
            "life_lost": cur_level == prev_level and cur_lives < prev_lives,
        }

        return StepResult(
            state=self._make_game_state(done=done),
            reward=reward,
            done=done,
            info=info,
        )

    def get_state(self) -> "GameState":
        return self._make_game_state()

    def is_done(self) -> bool:
        return self._engine._state in (
            EngineGameState.WIN,
            EngineGameState.GAME_OVER,
        )

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError(f"Unsupported render mode: {mode}")
        index_grid = self._engine.camera.render(
            self._engine.current_level.get_sprites()
        )
        h, w = index_grid.shape[:2]
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for idx, color in enumerate(_ARC_PALETTE):
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
        self, state: "GameState", step_info: Optional[Dict] = None
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
