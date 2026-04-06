from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import random
import struct
import zlib

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from arcengine import (
    ActionInput,
    ARCBaseGame,
    Camera,
    GameAction,
    GameState as EngineGameState,
    Level,
    Sprite,
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


BG = 0
BORDER_COL = 5
DARK = 4
OBSTACLE_COL = 5
LIFE_ON = 8
LIFE_OFF = 4

GAME_COLORS = [9, 8, 14, 11, 12]

BAR_COL = 10

CANVAS_W = 64
CANVAS_H = 64
CELL_SIZE = 4

LIVES_PER_LEVEL = [3, 3, 3, 3]

HARDCODED_GRIDS = [
    [
        [3, 2, 3, 2, 3, 0, 3, 3, 1, 4],
        [1, 2, 2, 4, 2, 2, 4, 0, 4, 2],
        [4, 1, 2, 0, 4, 3, 0, 3, 1, 4],
        [1, 0, 2, 0, 1, 1, 3, 2, 2, 3],
        [2, 3, 0, 1, 4, 2, 2, 3, 4, 1],
        [0, 4, 3, 0, 0, 2, 0, 1, 2, 0],
        [0, 1, 1, 1, 3, 1, 3, 4, 1, 4],
        [2, 3, 0, 4, 3, 2, 2, 3, 0, 3],
        [4, 4, 3, 4, 0, 4, 2, 3, 3, 0],
        [0, 3, 0, 3, 1, 0, 2, 1, 0, 4],
    ],
    [
        [0, 1, 3, 2, 1, 1, 3, 2, 0, 3, 4, 2],
        [3, 4, 3, 1, 3, 4, 4, 2, 0, 4, 0, 1],
        [3, 2, 1, 2, 1, 3, 3, 2, 2, 0, 3, 2],
        [0, 2, 4, 0, 1, 0, 4, 2, 4, 4, 4, 4],
        [2, 0, 1, 4, 0, 3, 4, 0, 1, 2, 1, 1],
        [4, 3, 1, 4, 3, 1, 3, 2, 2, 4, 3, 1],
        [3, 4, 4, 3, 4, 0, 1, 3, 0, 0, 1, 0],
        [2, 1, 4, 0, 4, 2, 3, 3, 1, 1, 0, 1],
        [4, 3, 1, 3, 0, 0, 3, 4, 2, 4, 3, 4],
        [2, 0, 0, 2, 0, 3, 2, 2, 2, 3, 0, 3],
        [2, 0, 3, 0, 4, 1, 3, 4, 3, 0, 4, 1],
        [0, 1, 1, 2, 2, 0, 3, 1, 4, 3, 2, 2],
    ],
    [
        [3, 2, 3, 2, 3, 0, 3, 3, 1, 4, 1, 2],
        [2, 4, 2, 2, 4, 0, 4, 2, 4, 1, 2, 0],
        [4, 3, 0, 3, 1, 4, 1, 0, 2, 0, 1, 1],
        [3, 2, 2, 3, 2, 3, 0, 1, 4, 2, 2, 3],
        [4, 1, 0, 4, 3, 0, 0, 2, 0, 1, 2, 0],
        [0, 1, 1, 1, 3, 1, 3, 4, 1, 4, 2, 3],
        [0, 4, 3, 2, 2, 3, 0, 3, 4, 4, 3, 4],
        [0, 4, 2, 3, 3, 0, 0, 3, 0, 3, 1, 0],
        [2, 1, 0, 4, 4, 0, 0, 1, 0, 2, 2, 2],
        [4, 0, 1, 2, 1, 3, 0, 1, 1, 3, 1, 3],
        [2, 3, 1, 2, 0, 2, 2, 3, 2, 1, 3, 4],
        [4, 4, 3, 0, 0, 0, 3, 2, 1, 2, 4, 2],
    ],
    [
        [3, 0, 0, 1, 2, 1, 4, 1, 0, 1, 0, 2],
        [4, 4, 4, 4, 0, 1, 1, 3, 2, 4, 2, 4],
        [4, 1, 1, 3, 3, 0, 4, 3, 2, 0, 4, 2],
        [0, 1, 1, 3, 1, 2, 0, 0, 4, 2, 0, 4],
        [3, 2, 4, 4, 3, 0, 1, 0, 0, 1, 2, 1],
        [2, 3, 4, 4, 1, 3, 3, 3, 0, 1, 3, 2],
        [4, 3, 2, 1, 0, 1, 3, 2, 0, 0, 0, 3],
        [1, 2, 0, 0, 1, 4, 1, 1, 1, 2, 1, 3],
        [3, 3, 1, 4, 0, 2, 4, 4, 2, 2, 4, 0],
        [4, 2, 0, 3, 1, 0, 1, 4, 2, 4, 1, 4],
        [1, 2, 2, 1, 1, 2, 4, 1, 2, 0, 0, 3],
        [0, 3, 1, 4, 3, 3, 4, 0, 2, 4, 2, 0],
    ],
]

LEVEL_DEFS = [
    {
        "grid_size": 10,
        "num_colors": 5,
        "obstacles": [(5, c) for c in range(2, 8)],
    },
    {
        "grid_size": 12,
        "num_colors": 5,
        "obstacles": (
            [(r, 6) for r in range(2, 10)]
            + [(6, c) for c in range(3, 10)]
        ),
    },
    {
        "grid_size": 12,
        "num_colors": 5,
        "obstacles": (
            [(4, c) for c in range(0, 9)]
            + [(8, c) for c in range(3, 12)]
        ),
    },
    {
        "grid_size": 12,
        "num_colors": 5,
        "obstacles": (
            [(3, c) for c in range(0, 9)]
            + [(6, c) for c in range(3, 12)]
            + [(9, c) for c in range(0, 9)]
        ),
    },
]

N_LEVELS = len(LEVEL_DEFS)

BUDGET_SLACK = [2, 3, 4, 5]

ACTION_LIST : List[str] = ["reset", "up", "down", "left", "right", "select", "undo"]


def _find_region(
    grid: List[List[int]],
    grid_size: int,
    obstacles: Optional[Set[Tuple[int, int]]] = None,
) -> Set[Tuple[int, int]]:
    if obstacles is None:
        obstacles = set()
    color = grid[0][0]
    visited: Set[Tuple[int, int]] = set()
    queue = deque([(0, 0)])
    visited.add((0, 0))
    while queue:
        row, col = queue.popleft()
        for delta_r, delta_c in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor_r, neighbor_c = row + delta_r, col + delta_c
            if (
                0 <= neighbor_r < grid_size
                and 0 <= neighbor_c < grid_size
                and (neighbor_r, neighbor_c) not in visited
                and (neighbor_r, neighbor_c) not in obstacles
            ):
                if grid[neighbor_r][neighbor_c] == color:
                    visited.add((neighbor_r, neighbor_c))
                    queue.append((neighbor_r, neighbor_c))
    return visited


def _apply_flood(
    grid: List[List[int]],
    grid_size: int,
    new_color: int,
    obstacles: Optional[Set[Tuple[int, int]]] = None,
) -> Set[Tuple[int, int]]:
    if obstacles is None:
        obstacles = set()
    old_color = grid[0][0]
    if new_color == old_color:
        return _find_region(grid, grid_size, obstacles)
    region = _find_region(grid, grid_size, obstacles)
    for row, col in region:
        grid[row][col] = new_color
    return _find_region(grid, grid_size, obstacles)


def _greedy_solve(
    grid: List[List[int]],
    grid_size: int,
    num_colors: int,
    obstacles: Optional[Set[Tuple[int, int]]] = None,
) -> int:
    if obstacles is None:
        obstacles = set()
    working_grid = [row[:] for row in grid]
    total_cells = grid_size * grid_size - len(obstacles)
    moves = 0
    region = _find_region(working_grid, grid_size, obstacles)
    while len(region) < total_cells:
        current_color = working_grid[0][0]
        best_color = -1
        best_gain = -1
        for color_idx in range(num_colors):
            if color_idx == current_color:
                continue
            gain = 0
            for row, col in region:
                for delta_r, delta_c in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    neighbor_r = row + delta_r
                    neighbor_c = col + delta_c
                    if (
                        0 <= neighbor_r < grid_size
                        and 0 <= neighbor_c < grid_size
                        and (neighbor_r, neighbor_c) not in region
                        and (neighbor_r, neighbor_c) not in obstacles
                        and working_grid[neighbor_r][neighbor_c] == color_idx
                    ):
                        gain += 1
            if gain > best_gain:
                best_gain = gain
                best_color = color_idx
        if best_color < 0 or best_gain <= 0:
            for row, col in region:
                for delta_r, delta_c in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    neighbor_r = row + delta_r
                    neighbor_c = col + delta_c
                    if (
                        0 <= neighbor_r < grid_size
                        and 0 <= neighbor_c < grid_size
                        and (neighbor_r, neighbor_c) not in region
                        and (neighbor_r, neighbor_c) not in obstacles
                    ):
                        best_color = working_grid[neighbor_r][neighbor_c]
                        break
                if best_color >= 0:
                    break
            if best_color < 0:
                break
        region = _apply_flood(working_grid, grid_size, best_color, obstacles)
        moves += 1
    return moves


_PRECOMPUTED: List[dict] = []

for _i, _defn in enumerate(LEVEL_DEFS):
    _grid = [row[:] for row in HARDCODED_GRIDS[_i]]
    _obs = set(map(tuple, _defn.get("obstacles", [])))
    _greedy_count = _greedy_solve(
        [row[:] for row in _grid],
        _defn["grid_size"],
        _defn["num_colors"],
        _obs,
    )
    _slack = BUDGET_SLACK[_i] if _i < len(BUDGET_SLACK) else 2
    _budget = (_greedy_count * 2 + _slack) * 2
    _PRECOMPUTED.append({
        "grid": _grid,
        "greedy_count": _greedy_count,
        "budget": _budget,
        "obstacles": _obs,
    })

levels = [Level(sprites=[], grid_size=(CANVAS_W, CANVAS_H)) for _ in range(N_LEVELS)]


def _grid_origin(grid_size: int) -> Tuple[int, int]:
    grid_px = grid_size * CELL_SIZE
    pixel_x = (CANVAS_W - grid_px) // 2
    pixel_y = 14
    return pixel_x, pixel_y


def _draw_palette_strip(
    frame: np.ndarray,
    num_colors: int,
    cursor_color: int,
) -> None:
    swatch_w = 8
    swatch_h = 4
    total_palette_w = num_colors * swatch_w + (num_colors - 1) * 2
    palette_x = (CANVAS_W - total_palette_w) // 2
    palette_y = 7
    for color_index in range(num_colors):
        swatch_x = palette_x + color_index * (swatch_w + 2)
        palette_color = GAME_COLORS[color_index]
        if color_index == cursor_color:
            for border_x in range(swatch_x - 1, swatch_x + swatch_w + 1):
                if 0 <= border_x < CANVAS_W:
                    if palette_y - 1 >= 0:
                        frame[palette_y - 1, border_x] = BORDER_COL
                    if palette_y + swatch_h < CANVAS_H:
                        frame[palette_y + swatch_h, border_x] = BORDER_COL
            for border_y in range(palette_y - 1, palette_y + swatch_h + 1):
                if 0 <= border_y < CANVAS_H:
                    if swatch_x - 1 >= 0:
                        frame[border_y, swatch_x - 1] = BORDER_COL
                    if swatch_x + swatch_w < CANVAS_W:
                        frame[border_y, swatch_x + swatch_w] = BORDER_COL
        for delta_y in range(swatch_h):
            for delta_x in range(swatch_w):
                if 0 <= swatch_x + delta_x < CANVAS_W and 0 <= palette_y + delta_y < CANVAS_H:
                    frame[palette_y + delta_y, swatch_x + delta_x] = palette_color


def _draw_grid_cells(
    frame: np.ndarray,
    grid: List[List[int]],
    grid_size: int,
    obstacles: Set[Tuple[int, int]],
    region: Set[Tuple[int, int]],
) -> None:
    origin_x, origin_y = _grid_origin(grid_size)
    for row in range(grid_size):
        for col in range(grid_size):
            cell_color_idx = grid[row][col]
            palette_color = GAME_COLORS[cell_color_idx]
            pixel_x = origin_x + col * CELL_SIZE
            pixel_y = origin_y + row * CELL_SIZE
            for delta_y in range(3):
                for delta_x in range(3):
                    canvas_x = pixel_x + delta_x
                    canvas_y = pixel_y + delta_y
                    if 0 <= canvas_x < CANVAS_W and 0 <= canvas_y < CANVAS_H:
                        frame[canvas_y, canvas_x] = palette_color
    for obs_row, obs_col in obstacles:
        pixel_x = origin_x + obs_col * CELL_SIZE
        pixel_y = origin_y + obs_row * CELL_SIZE
        for delta_y in range(3):
            for delta_x in range(3):
                canvas_x = pixel_x + delta_x
                canvas_y = pixel_y + delta_y
                if 0 <= canvas_x < CANVAS_W and 0 <= canvas_y < CANVAS_H:
                    frame[canvas_y, canvas_x] = OBSTACLE_COL
    for reg_row, reg_col in region:
        center_x = origin_x + reg_col * CELL_SIZE + 1
        center_y = origin_y + reg_row * CELL_SIZE + 1
        if 0 <= center_x < CANVAS_W and 0 <= center_y < CANVAS_H:
            frame[center_y, center_x] = BG


def _draw_budget_bar(
    frame: np.ndarray,
    budget: int,
    budget_max: int,
) -> None:
    bar_y = 0
    bar_h = 2
    if budget_max <= 0:
        return
    filled = round((budget / budget_max) * CANVAS_W)
    for col in range(CANVAS_W):
        bar_color = BAR_COL if col < filled else DARK
        for delta_y in range(bar_h):
            frame[bar_y + delta_y, col] = bar_color


def _draw_lives_dots(
    frame: np.ndarray,
    lives: int,
    lives_max: int,
) -> None:
    lives_y = 3
    dot_size = 2
    gap = 1
    for index in range(lives_max):
        dot_x = CANVAS_W - (lives_max - index) * (dot_size + gap)
        lost_count = lives_max - lives
        dot_color = LIFE_OFF if index < lost_count else LIFE_ON
        for delta_y in range(dot_size):
            for delta_x in range(dot_size):
                if 0 <= dot_x + delta_x < CANVAS_W and 0 <= lives_y + delta_y < CANVAS_H:
                    frame[lives_y + delta_y, dot_x + delta_x] = dot_color


def _build_frame(
    grid: List[List[int]],
    grid_size: int,
    num_colors: int,
    region: Set[Tuple[int, int]],
    current_color: int,
    cursor_color: int,
    budget: int,
    budget_max: int,
    lives: int,
    lives_max: int,
    obstacles: Optional[Set[Tuple[int, int]]] = None,
) -> np.ndarray:
    if obstacles is None:
        obstacles = set()
    frame = np.full((CANVAS_H, CANVAS_W), BG, dtype=np.uint8)
    _draw_palette_strip(frame, num_colors, cursor_color)
    _draw_grid_cells(frame, grid, grid_size, obstacles, region)
    _draw_budget_bar(frame, budget, budget_max)
    _draw_lives_dots(frame, lives, lives_max)
    return frame


ACTION_TO_COLOR = {
    GameAction.ACTION1: 0,
    GameAction.ACTION2: 1,
    GameAction.ACTION3: 2,
    GameAction.ACTION4: 3,
    GameAction.ACTION5: 4,
}

COLOR_NAMES = ["Blue", "Red", "Green", "Yellow", "Orange"]

_COLOR_CHAR = {0: "B", 1: "R", 2: "G", 3: "Y", 4: "O"}
_COLOR_NAME = {0: "Blue", 1: "Red", 2: "Green", 3: "Yellow", 4: "Orange"}


class Ap04(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)
        camera = Camera(
            background=BG,
            letter_box=BORDER_COL,
            width=CANVAS_W,
            height=CANVAS_H,
        )
        super().__init__(
            game_id="ap04",
            levels=levels,
            camera=camera,
            available_actions=[0, 1, 2, 3, 4, 5, 7],
        )
        self._grid: List[List[int]] = []
        self._grid_size: int = 0
        self._num_colors: int = 0
        self._obstacles: Set[Tuple[int, int]] = set()
        self._region: Set[Tuple[int, int]] = set()
        self._cursor_color: int = -1
        self._budget: int = 0
        self._budget_max: int = 0
        self._lives: int = 5
        self._lives_max: int = 5
        self._ready: bool = False
        self._undo_stack: List[dict] = []
        self._ready = True
        self.on_set_level(self.current_level)

    def on_set_level(self, level: Level) -> None:
        if not getattr(self, "_ready", False):
            return
        self.current_level.remove_all_sprites()
        idx = self._current_level_index
        defn = LEVEL_DEFS[idx]
        precomp = _PRECOMPUTED[idx]
        self._grid_size = defn["grid_size"]
        self._num_colors = defn["num_colors"]
        self._obstacles = precomp["obstacles"]
        self._grid = [row[:] for row in precomp["grid"]]
        self._budget_max = precomp["budget"]
        self._budget = precomp["budget"]
        self._region = _find_region(self._grid, self._grid_size, self._obstacles)
        self._cursor_color = -1
        self._lives_max = LIVES_PER_LEVEL[idx] if idx < len(LIVES_PER_LEVEL) else 3
        self._lives = self._lives_max
        self._undo_stack = []
        self._ready = True
        self._render()

    def _save_undo_state(self) -> None:
        self._undo_stack.append({
            "grid": [row[:] for row in self._grid],
            "cursor_color": self._cursor_color,
            "budget": self._budget,
            "region": set(self._region),
        })

    def _perform_undo(self) -> bool:
        if not self._undo_stack:
            return False
        snapshot = self._undo_stack.pop()
        self._grid = snapshot["grid"]
        self._cursor_color = snapshot["cursor_color"]
        self._region = snapshot["region"]
        return True

    def step(self) -> None:
        if not self._ready:
            self.complete_action()
            return
        action = self.action
        if action.id == GameAction.RESET:
            self.complete_action()
            return
        if action.id == GameAction.ACTION7:
            self._perform_undo()
            self._budget = max(0, self._budget - 1)
            self._render()
            if self._budget <= 0:
                self._handle_life_lost()
            self.complete_action()
            return
        if action.id in (GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4, GameAction.ACTION5):
            chosen_color = ACTION_TO_COLOR[action.id]
            self._save_undo_state()
            self._budget = max(0, self._budget - 1)
            if chosen_color < self._num_colors:
                current_color = self._grid[0][0]
                if chosen_color == self._cursor_color:
                    if chosen_color != current_color:
                        self._region = _apply_flood(
                            self._grid, self._grid_size, chosen_color, self._obstacles
                        )
                elif chosen_color == current_color and self._cursor_color == -1:
                    pass
                elif chosen_color == current_color and self._cursor_color != -1:
                    self._cursor_color = -1
                else:
                    self._cursor_color = chosen_color
            self._render()
            total_cells = self._grid_size * self._grid_size - len(self._obstacles)
            if len(self._region) >= total_cells:
                self.next_level()
                self.complete_action()
                return
            if self._budget <= 0:
                self._handle_life_lost()
                self.complete_action()
                return
        else:
            self._render()
        self.complete_action()

    def _handle_life_lost(self) -> None:
        self._lives -= 1
        if self._lives <= 0:
            self._render()
            self.lose()
            return
        idx = self._current_level_index
        precomp = _PRECOMPUTED[idx]
        self._grid = [row[:] for row in precomp["grid"]]
        self._budget = self._budget_max
        self._region = _find_region(self._grid, self._grid_size, self._obstacles)
        self._cursor_color = -1
        self._undo_stack = []
        self._render()

    def _render(self) -> None:
        self.current_level.remove_all_sprites()
        current_color = self._grid[0][0]
        cursor = self._cursor_color if self._cursor_color >= 0 else current_color
        frame = _build_frame(
            self._grid,
            self._grid_size,
            self._num_colors,
            self._region,
            current_color,
            cursor,
            self._budget,
            self._budget_max,
            self._lives,
            self._lives_max,
            self._obstacles,
        )
        bg_sprite = Sprite(
            pixels=frame.tolist(),
            name="background",
            visible=True,
            collidable=False,
            layer=0,
        )
        bg_sprite.set_position(0, 0)
        self.current_level.add_sprite(bg_sprite)

    @property
    def extra_state(self) -> dict:
        idx = self._current_level_index
        total_cells = self._grid_size * self._grid_size - len(self._obstacles)
        region_size = len(self._region)
        current_color = self._grid[0][0] if self._grid else -1
        cursor = self._cursor_color if self._cursor_color >= 0 else current_color
        cursor_name = COLOR_NAMES[cursor] if 0 <= cursor < len(COLOR_NAMES) else "?"
        flood_percent = round(100 * region_size / total_cells, 1) if total_cells else 0
        return {
            "snake_length": region_size,
            "target_length": total_cells,
            "speed_cells_s": 0,
            "speed_interval_ms": 0,
            "remaining_cells": total_cells - region_size,
            "circuit_title": "Color Flood - Level %d/%d" % (idx + 1, N_LEVELS),
            "level_title": "Level %d" % (idx + 1),
            "grid_size": self._grid_size,
            "num_colors": self._num_colors,
            "num_obstacles": len(self._obstacles),
            "region_size": region_size,
            "total_cells": total_cells,
            "flood_percent": flood_percent,
            "current_color": current_color,
            "current_color_name": (
                COLOR_NAMES[current_color]
                if 0 <= current_color < len(COLOR_NAMES)
                else "?"
            ),
            "cursor_color": cursor,
            "cursor_color_name": cursor_name,
            "budget_remaining": self._budget,
            "budget_max": self._budget_max,
            "lives": self._lives,
            "lives_max": self._lives_max,
            "lives_lost": self._lives_max - self._lives,
            "level_features": [
                "Level %d/%d" % (idx + 1, N_LEVELS),
                "Grid: %dx%d" % (self._grid_size, self._grid_size),
                "Colors: %d" % self._num_colors,
                "Obstacles: %d" % len(self._obstacles),
                "Flooded: %d/%d (%.0f%%)" % (
                    region_size,
                    total_cells,
                    100 * region_size / total_cells if total_cells else 0,
                ),
                "Current: %s" % (
                    COLOR_NAMES[current_color]
                    if 0 <= current_color < len(COLOR_NAMES)
                    else "?"
                ),
                "Cursor: %s" % cursor_name,
                "Budget: %d/%d" % (self._budget, self._budget_max),
                "Lives: %d/%d" % (self._lives, self._lives_max),
            ],
        }


class PuzzleEnvironment:
    ACTION_MAP: Dict[str, int] = {
        "reset": 0,
        "up": 1,
        "down": 2,
        "left": 3,
        "right": 4,
        "select": 5,
        "undo": 7,
    }

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

    _TAG_TO_CHAR: Dict[str, str] = {}

    def __init__(self, seed: int = 0) -> None:
        self._engine = Ap04(seed=seed)
        self._total_turns = 0
        self._last_action_was_reset = False
        self._game_won = False
        self._levels_completed = 0

    def _build_text_observation(self) -> str:
        e = self._engine
        idx = e._current_level_index
        grid = e._grid
        grid_size = e._grid_size
        obstacles = e._obstacles
        region = e._region
        total_cells = grid_size * grid_size - len(obstacles)
        region_size = len(region)
        current_color = grid[0][0] if grid else -1
        cursor = e._cursor_color if e._cursor_color >= 0 else current_color
        lines: List[str] = []
        lines.append(
            "Level %d/%d | %dx%d | %d colours | Obstacles: %d"
            % (
                idx + 1,
                N_LEVELS,
                grid_size,
                grid_size,
                e._num_colors,
                len(obstacles),
            )
        )
        lines.append(
            "Budget: %d/%d | Lives: %d/%d | Flooded: %d/%d (%.0f%%)"
            % (
                e._budget,
                e._budget_max,
                e._lives,
                e._lives_max,
                region_size,
                total_cells,
                100 * region_size / total_cells if total_cells else 0,
            )
        )
        lines.append(
            "Current colour: %s | Cursor: %s"
            % (
                _COLOR_NAME.get(current_color, "?"),
                _COLOR_NAME.get(cursor, "?"),
            )
        )
        lines.append("")
        for row in range(grid_size):
            row_chars: List[str] = []
            for col in range(grid_size):
                if (row, col) in obstacles:
                    row_chars.append("#")
                else:
                    character = _COLOR_CHAR.get(grid[row][col], "?")
                    if (row, col) in region:
                        character = character.lower()
                    row_chars.append(character)
            lines.append(" ".join(row_chars))
        return "\n".join(lines)

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
        for idx, color in enumerate(self.ARC_PALETTE):
            mask = arr == idx
            rgb[mask] = color
        return self._encode_png(rgb)

    def _build_game_state(
        self, done: bool = False, info: Optional[Dict] = None
    ) -> GameState:
        e = self._engine
        engine_done = e._state in (EngineGameState.WIN, EngineGameState.GAME_OVER)
        valid_actions = ["reset"] if engine_done else self.get_actions()
        idx = e._current_level_index
        game_over = e._state == EngineGameState.GAME_OVER
        return GameState(
            text_observation=self._build_text_observation(),
            image_observation=self._build_image_bytes(),
            valid_actions=valid_actions,
            turn=self._total_turns,
            metadata={
                "total_levels": len(self._engine._levels),
                "level_index": idx,
                "levels_completed": self._levels_completed,
                "lives": e._lives,
                "game_over": game_over,
                "done": engine_done,
                "info": {},
            },
        )

    def reset(self) -> GameState:
        e = self._engine
        reset_input = ActionInput(id=GameAction.RESET)
        if self._game_won or self._last_action_was_reset:
            e.perform_action(reset_input)
            e.perform_action(reset_input)
        else:
            e.perform_action(reset_input)
        self._total_turns = 0
        self._last_action_was_reset = True
        self._game_won = False
        self._levels_completed = 0
        return self._build_game_state()

    def get_actions(self) -> List[str]:
        e = self._engine
        if e._state in (EngineGameState.WIN, EngineGameState.GAME_OVER):
            return ["reset"]
        return ACTION_LIST

    def is_done(self) -> bool:
        return self._engine._state in (EngineGameState.WIN, EngineGameState.GAME_OVER)

    def step(self, action: str) -> StepResult:
        e = self._engine

        if action == "reset":
            state = self.reset()
            return StepResult(
                state=state, reward=0.0, done=False, info={"action": "reset"}
            )

        if action not in self.ACTION_MAP:
            return StepResult(
                state=self._build_game_state(done=False),
                reward=0.0,
                done=False,
                info={"action": action, "error": "invalid_action"},
            )

        self._last_action_was_reset = False
        self._total_turns += 1

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
        info: Dict = {"action": action}

        total_levels = len(e._levels)
        level_before = e.level_index

        action_input = ActionInput(id=game_action)
        frame = e.perform_action(action_input, raw=True)

        game_won = e._state == EngineGameState.WIN
        game_over = e._state == EngineGameState.GAME_OVER
        level_advanced = e.level_index > level_before
        done = game_won or game_over

        reward = 0.0
        if game_won or level_advanced:
            self._levels_completed += 1
            reward = 1.0 / total_levels

        if game_won:
            self._game_won = True
            info["reason"] = "game_complete"
            return StepResult(
                state=self._build_game_state(done=True, info=info),
                reward=reward,
                done=True,
                info=info,
            )

        if game_over:
            info["reason"] = "death"
            return StepResult(
                state=self._build_game_state(done=True, info=info),
                reward=0.0,
                done=True,
                info=info,
            )

        return StepResult(
            state=self._build_game_state(done=False, info=info),
            reward=reward,
            done=False,
            info=info,
        )

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError("Unsupported render mode: %s" % mode)
        e = self._engine
        index_grid = e.camera.render(e.current_level.get_sprites())
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        for idx, color in enumerate(self.ARC_PALETTE):
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
                "Unsupported render_mode '%s'. Supported: %s"
                % (render_mode, self.metadata["render_modes"])
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
        return img[row_idx[:, None], col_idx[None, :]]

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
