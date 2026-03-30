import random
import struct
import zlib
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from arcengine import ActionInput, ARCBaseGame, GameAction, GameState as EngineGameState
from arcengine.camera import Camera, RenderableUserDisplay
from arcengine.level import Level
from arcengine.sprites import Sprite


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


C_COIN = 0
C_COIN_COLLECTED = 12
C_DEBRIS = 1
C_TRAVELER = 2
C_WALL = 3
C_GRID_LINE = 4
C_LAND = 5
C_CRACKED = 6
C_CURSOR = 7
C_MINE = 8
C_RED = 8
C_WATER = 9
C_MUDWATER = 10
C_BRIDGE = 11
C_TRAIL = 12
C_LAVA = 13
C_GOAL = 14
C_SIGNPOST = 15

WALKABLE = {
    C_LAND,
    C_BRIDGE,
    C_GOAL,
    C_TRAIL,
    C_CRACKED,
    C_LAVA,
    C_MINE,
    C_COIN,
    C_COIN_COLLECTED,
}

DISPLAY_SIZE = 64
HUD_HEIGHT = 3
HUD_TOP = DISPLAY_SIZE - HUD_HEIGHT

TOTAL_LIVES = 3


def _grid_layout(gw: int, gh: int) -> Tuple[int, int, int]:
    scale = min(DISPLAY_SIZE // gw, DISPLAY_SIZE // gh)
    x_off = (DISPLAY_SIZE - gw * scale) // 2
    y_off = (DISPLAY_SIZE - gh * scale) // 2
    return scale, x_off, y_off


def bfs_path(
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
) -> Optional[List[Tuple[int, int]]]:
    h, w = grid.shape
    sx, sy = start
    gx, gy = goal

    if sx == gx and sy == gy:
        return [start]

    visited = set()
    visited.add((sx, sy))
    parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {(sx, sy): None}
    queue = deque([(sx, sy)])

    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

    while queue:
        cx, cy = queue.popleft()

        for dx, dy in directions:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in visited:
                cell = int(grid[ny, nx])
                if cell in WALKABLE:
                    visited.add((nx, ny))
                    parent[(nx, ny)] = (cx, cy)
                    if nx == gx and ny == gy:
                        path = []
                        node: Optional[Tuple[int, int]] = (gx, gy)
                        while node is not None:
                            path.append(node)
                            node = parent[node]
                        path.reverse()
                        return path
                    queue.append((nx, ny))

    return None


def generate_level_1(
    rng: random.Random,
) -> Tuple[np.ndarray, List[Tuple[int, int]], Tuple[int, int], Tuple[int, int]]:
    gw, gh = 12, 12
    grid = np.full((gh, gw), C_LAND, dtype=np.int8)

    for x in range(gw):
        grid[2, x] = C_WATER

    for x in range(gw):
        if x not in (3, 8):
            grid[4, x] = C_WALL

    for y in [5, 6]:
        grid[y, 2] = C_WALL
        grid[y, 3] = C_LAVA
        grid[y, 4] = C_WALL
        grid[y, 7] = C_WALL
        grid[y, 9] = C_WALL
        for x in [0, 1, 5, 6, 10, 11]:
            grid[y, x] = C_WALL

    for x in range(gw):
        if x not in (3, 8):
            grid[7, x] = C_WALL

    for x in range(gw):
        grid[9, x] = C_WATER

    grid[3, 6] = C_MINE

    traveler = (1, 1)
    goal = (10, 10)
    grid[goal[1], goal[0]] = C_GOAL

    click_seq = [(8, 2), (8, 9)]

    return grid, click_seq, traveler, goal


def generate_level_2(
    rng: random.Random,
) -> Tuple[np.ndarray, List[Tuple[int, int]], Tuple[int, int], Tuple[int, int]]:
    gw, gh = 14, 14
    grid = np.full((gh, gw), C_LAND, dtype=np.int8)

    for x in range(gw):
        grid[2, x] = C_MUDWATER
    grid[2, 1] = C_WATER
    grid[2, 7] = C_WATER

    for x in range(gw):
        grid[3, x] = C_WALL
    grid[3, 1] = C_LAND
    grid[3, 7] = C_LAND

    grid[3, 7] = C_MINE

    grid[4, 0] = C_WALL
    grid[4, 6] = C_WATER
    for x in range(8, 13):
        grid[4, x] = C_WALL
    grid[4, 13] = C_WALL

    for x in range(gw):
        grid[5, x] = C_WALL
    grid[5, 6] = C_LAND
    grid[5, 7] = C_LAND
    grid[5, 12] = C_LAND

    for x in range(gw):
        grid[6, x] = C_WALL
    grid[6, 6] = C_LAND
    grid[6, 7] = C_LAND
    grid[6, 12] = C_LAND

    for x in range(gw):
        grid[7, x] = C_WALL
    for x in range(5, 13):
        grid[7, x] = C_LAND

    for x in range(gw):
        grid[8, x] = C_MUDWATER
    grid[8, 5] = C_WATER
    grid[8, 12] = C_WATER

    for x in range(gw):
        grid[9, x] = C_WALL
    grid[9, 5] = C_LAVA
    grid[9, 12] = C_LAND

    grid[10, 0] = C_WALL
    grid[10, 13] = C_WALL

    for x in range(gw):
        grid[11, x] = C_WATER

    for x in range(gw):
        grid[12, x] = C_WALL
    grid[12, 1] = C_LAND

    grid[13, 0] = C_WALL
    grid[13, 13] = C_WALL

    traveler = (1, 0)
    goal = (12, 13)
    grid[goal[1], goal[0]] = C_GOAL

    click_seq = [(1, 2), (6, 4), (12, 8), (1, 11)]

    return grid, click_seq, traveler, goal


def generate_level_3(
    rng: random.Random,
) -> Tuple[np.ndarray, List[Tuple[int, int]], Tuple[int, int], Tuple[int, int]]:
    gw, gh = 14, 14
    grid = np.full((gh, gw), C_LAND, dtype=np.int8)

    grid[1, 4] = C_MINE
    grid[1, 9] = C_MINE

    for x in range(gw):
        grid[2, x] = C_MUDWATER
    grid[2, 1] = C_WATER
    grid[2, 5] = C_WATER
    grid[2, 10] = C_WATER

    for x in range(gw):
        grid[3, x] = C_WALL
    grid[3, 1] = C_LAND
    grid[3, 5] = C_WALL
    grid[3, 10] = C_MINE

    grid[4, 0] = C_WALL
    grid[4, 13] = C_WALL

    for x in range(gw):
        grid[5, x] = C_WALL
    grid[5, 4] = C_SIGNPOST
    grid[5, 5] = C_LAND
    grid[5, 6] = C_MINE

    for x in range(gw):
        grid[6, x] = C_WALL
    grid[6, 4] = C_LAND
    grid[6, 5] = C_WATER
    grid[6, 6] = C_LAND

    for x in range(gw):
        grid[7, x] = C_WALL
    for x in range(4, 13):
        grid[7, x] = C_LAND

    for x in range(gw):
        grid[8, x] = C_MUDWATER
    grid[8, 5] = C_WATER
    grid[8, 12] = C_WATER

    for x in range(gw):
        grid[9, x] = C_WALL
    grid[9, 5] = C_LAVA
    grid[9, 12] = C_LAND

    grid[10, 0] = C_WALL
    grid[10, 13] = C_WALL
    grid[10, 3] = C_COIN

    for x in range(gw):
        grid[11, x] = C_WALL
    grid[11, 1] = C_SIGNPOST

    for x in range(gw):
        grid[12, x] = C_WATER

    grid[13, 0] = C_WALL
    grid[13, 13] = C_LAND

    traveler = (1, 1)
    goal = (12, 13)
    grid[goal[1], goal[0]] = C_GOAL

    click_seq = [
        (3, 10),
        (1, 2),
        (4, 5),
        (4, 5),
        (5, 6),
        (12, 8),
        (1, 11),
        (1, 11),
        (1, 12),
    ]

    return grid, click_seq, traveler, goal


def generate_level_4(
    rng: random.Random,
) -> Tuple[np.ndarray, List[Tuple[int, int]], Tuple[int, int], Tuple[int, int]]:
    gw, gh = 16, 17
    grid = np.full((gh, gw), C_LAND, dtype=np.int8)

    grid[0, 6] = C_COIN

    for x in range(gw):
        grid[2, x] = C_WALL
    grid[2, 13] = C_LAND

    for x in range(gw):
        grid[3, x] = C_MUDWATER
    grid[3, 1] = C_WATER
    grid[3, 7] = C_WATER
    grid[3, 13] = C_WATER

    for x in range(gw):
        grid[4, x] = C_WALL
    grid[4, 1] = C_LAND
    grid[4, 7] = C_MINE
    grid[4, 13] = C_LAND

    grid[5, 0] = C_WALL
    grid[5, 1] = C_MINE
    for x in range(7, 15):
        grid[5, x] = C_WALL
    grid[5, 13] = C_LAND
    grid[5, 14] = C_LAND
    grid[5, 15] = C_WALL

    for x in range(gw):
        grid[6, x] = C_WALL
    grid[6, 5] = C_LAND
    grid[6, 6] = C_SIGNPOST
    grid[6, 14] = C_LAND

    for x in range(gw):
        grid[7, x] = C_WALL
    grid[7, 5] = C_LAND
    grid[7, 6] = C_LAND
    grid[7, 7] = C_WATER
    grid[7, 14] = C_LAND

    for x in range(gw):
        grid[8, x] = C_WALL
    grid[8, 5] = C_LAND
    for x in range(7, 15):
        grid[8, x] = C_LAND

    for x in range(gw):
        grid[9, x] = C_MUDWATER
    grid[9, 5] = C_WATER
    grid[9, 14] = C_WATER

    for x in range(gw):
        grid[10, x] = C_WALL
    grid[10, 5] = C_LAVA
    grid[10, 14] = C_LAND

    grid[11, 0] = C_WALL
    grid[11, 15] = C_WALL
    grid[11, 8] = C_WATER
    grid[11, 10] = C_COIN

    for x in range(gw):
        grid[12, x] = C_WATER

    for x in range(gw):
        grid[13, x] = C_WALL
    grid[13, 1] = C_LAND

    grid[14, 0] = C_WALL
    grid[14, 15] = C_WALL
    grid[14, 2] = C_WATER

    for x in range(gw):
        grid[15, x] = C_WALL
    grid[15, 5] = C_SIGNPOST

    for x in range(gw):
        grid[16, x] = C_WALL
    grid[16, 5] = C_LAND
    grid[16, 6] = C_LAND
    grid[16, 7] = C_LAND
    grid[16, 8] = C_LAND
    grid[16, 9] = C_LAND
    for x in range(10, 15):
        grid[16, x] = C_LAND
    grid[16, 14] = C_GOAL

    traveler = (0, 0)
    goal = (14, 16)

    click_seq = [
        (6, 0),
        (10, 11),
        (13, 3),
        (14, 9),
        (8, 11),
        (1, 12),
        (2, 14),
        (5, 15),
        (5, 15),
    ]

    return grid, click_seq, traveler, goal


def generate_level_5(
    rng: random.Random,
) -> Tuple[np.ndarray, List[Tuple[int, int]], Tuple[int, int], Tuple[int, int]]:
    gw, gh = 17, 17
    grid = np.full((gh, gw), C_LAND, dtype=np.int8)

    for x in range(gw):
        grid[2, x] = C_MUDWATER
    grid[2, 1] = C_WATER
    grid[2, 5] = C_WATER
    grid[2, 10] = C_WATER
    grid[2, 15] = C_WATER

    grid[3, 4] = C_MINE
    grid[3, 9] = C_MINE
    grid[3, 14] = C_MINE

    grid[4, 1] = C_COIN
    for x in range(2, gw):
        grid[4, x] = C_WALL

    grid[5, 6] = C_WATER
    grid[5, 11] = C_SIGNPOST

    for x in range(gw):
        grid[6, x] = C_WALL
    grid[6, 15] = C_SIGNPOST

    grid[7, 15] = C_COIN

    for x in range(gw):
        grid[8, x] = C_MUDWATER
    grid[8, 15] = C_WATER

    for x in range(gw):
        grid[9, x] = C_WALL
    grid[9, 6] = C_LAVA
    grid[9, 11] = C_LAVA
    grid[9, 15] = C_LAND

    for x in range(2, gw):
        grid[11, x] = C_WALL

    grid[12, 1] = C_SIGNPOST
    grid[12, 8] = C_MINE

    for x in range(gw):
        grid[13, x] = C_MUDWATER
    grid[13, 1] = C_WATER
    grid[13, 8] = C_WATER

    grid[14, 1] = C_SIGNPOST
    grid[14, 6] = C_MINE

    grid[16, 8] = C_COIN
    grid[16, 15] = C_GOAL

    traveler = (0, 1)
    goal = (15, 16)

    click_seq = [
        (1, 4),
        (15, 7),
        (8, 16),
        (1, 2),
        (6, 5),
        (11, 5),
        (11, 5),
        (15, 6),
        (15, 6),
        (15, 8),
        (1, 12),
        (1, 12),
        (1, 13),
        (1, 14),
        (1, 14),
    ]

    return grid, click_seq, traveler, goal


LEVEL_GENERATORS = [
    generate_level_1,
    generate_level_2,
    generate_level_3,
    generate_level_4,
    generate_level_5,
]

LEVEL_GRID_SIZES = [
    (12, 12),
    (14, 14),
    (14, 14),
    (16, 17),
    (17, 17),
]

LEVEL_MAX_MOVES = [54, 210, 306, 390, 654]

LEVEL_COINS = [
    [],
    [],
    [(3, 10)],
    [(6, 0), (10, 11)],
    [(1, 4), (15, 7), (8, 16)],
]


class Ms04(ARCBaseGame):
    _grids: List[np.ndarray]
    _base_grids: List[np.ndarray]
    _grid_sizes: List[Tuple[int, int]]
    _click_seqs: List[List[Tuple[int, int]]]
    _traveler_positions: List[Tuple[int, int]]
    _goal_positions: List[Tuple[int, int]]
    _trap_bridges: set
    _weakened_walls: set
    _coins_collected: List[set]
    _rng: random.Random
    _moves_used: List[int]
    _lives: int
    _last_level_index: int
    _cursor_x: int
    _cursor_y: int

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self._rng = random.Random(seed)

        self._grids = []
        self._base_grids = []
        self._grid_sizes = []
        self._click_seqs = []
        self._traveler_positions = []
        self._goal_positions = []
        self._trap_bridges: set = set()
        self._weakened_walls: set = set()
        self._coins_collected = [set() for _ in range(len(LEVEL_GRID_SIZES))]
        self._moves_used = [0] * len(LEVEL_GRID_SIZES)
        self._lives = TOTAL_LIVES
        self._last_level_index = 0
        self._cursor_x = 0
        self._cursor_y = 0
        self._game_over = False
        self._undo_snapshot: Optional[Dict[str, Any]] = None

        levels: List[Level] = []
        for i, gen_fn in enumerate(LEVEL_GENERATORS):
            gw, gh = LEVEL_GRID_SIZES[i]
            grid, click_seq, traveler, goal = gen_fn(self._rng)

            base_grid = np.copy(grid)
            display_grid = np.copy(grid)

            display_grid[traveler[1], traveler[0]] = C_TRAVELER

            self._grids.append(display_grid)
            self._base_grids.append(base_grid)
            self._grid_sizes.append((gw, gh))
            self._click_seqs.append(click_seq)
            self._traveler_positions.append(traveler)
            self._goal_positions.append(goal)

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
            background=C_LAND,
            letter_box=C_LAND,
        )

        super().__init__(
            game_id="ms04",
            levels=levels,
            camera=camera,
            available_actions=[0, 1, 2, 3, 4, 5, 6, 7],
            seed=seed,
        )

        overlay = _BridgeBuilderOverlay(self)
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
            self._trap_bridges.clear()
            self._weakened_walls.clear()
            self._game_over = False
            self._undo_snapshot = None
            if idx != self._last_level_index:
                self._lives = TOTAL_LIVES
                self._last_level_index = idx

    def handle_reset(self) -> None:
        self._lives = TOTAL_LIVES
        self._game_over = False
        super().handle_reset()

    def full_reset(self) -> None:
        self._game_over = False
        self._undo_snapshot = None
        super().full_reset()
        self._moves_used = [0] * len(LEVEL_GRID_SIZES)
        self._lives = TOTAL_LIVES
        self._last_level_index = 0
        self._trap_bridges.clear()
        self._weakened_walls.clear()
        self._coins_collected = [set() for _ in range(len(LEVEL_GRID_SIZES))]
        rng = random.Random(self._seed)
        for i in range(len(self._grids)):
            gen_fn = LEVEL_GENERATORS[i]
            base, _, traveler, goal = gen_fn(rng)
            self._base_grids[i] = base
            self._grids[i] = np.copy(base)
            self._traveler_positions[i] = traveler
            self._goal_positions[i] = goal
            self._grids[i][traveler[1], traveler[0]] = C_TRAVELER
        self._sync_current_level_sprite()

    def level_reset(self, preserve_moves: bool = False) -> None:
        self._game_over = False
        self._undo_snapshot = None
        idx = self._current_level_index
        if self._lives <= 0:
            self._lives = TOTAL_LIVES
        super().level_reset()
        if not preserve_moves:
            self._moves_used[idx] = 0
        rng = random.Random(self._seed)
        gen_fn = LEVEL_GENERATORS[idx]
        base, _, traveler, goal = gen_fn(rng)
        self._base_grids[idx] = base
        self._grids[idx] = np.copy(base)
        self._traveler_positions[idx] = traveler
        self._goal_positions[idx] = goal
        self._trap_bridges.clear()
        self._weakened_walls.clear()
        self._coins_collected[idx] = set()
        self._grids[idx][traveler[1], traveler[0]] = C_TRAVELER
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
        idx = self._current_level_index
        self._moves_used[idx] += 1
        if self._moves_used[idx] > LEVEL_MAX_MOVES[idx]:
            self._lives -= 1
            if self._lives <= 0:
                self._game_over = True
                self.lose()
            else:
                self.level_reset(preserve_moves=False)
            return True
        return False

    def _save_snapshot(self) -> None:
        idx = self._current_level_index
        self._undo_snapshot = {
            "base_grid": np.copy(self._base_grids[idx]),
            "display_grid": np.copy(self._grids[idx]),
            "traveler": self._traveler_positions[idx],
            "trap_bridges": set(self._trap_bridges),
            "weakened_walls": set(self._weakened_walls),
            "coins": set(self._coins_collected[idx]),
            "cursor_x": self._cursor_x,
            "cursor_y": self._cursor_y,
        }

    def _restore_snapshot(self) -> None:
        if self._undo_snapshot is None:
            return
        idx = self._current_level_index
        snap = self._undo_snapshot
        self._base_grids[idx] = snap["base_grid"]
        self._grids[idx] = snap["display_grid"]
        self._traveler_positions[idx] = snap["traveler"]
        self._trap_bridges = snap["trap_bridges"]
        self._weakened_walls = snap["weakened_walls"]
        self._coins_collected[idx] = snap["coins"]
        self._cursor_x = snap["cursor_x"]
        self._cursor_y = snap["cursor_y"]
        self._undo_snapshot = None
        self._sync_current_level_sprite()

    def step(self) -> None:
        action = self.action

        if self._game_over:
            self.complete_action()
            return

        if action.id == GameAction.ACTION7:
            if not self._consume_move():
                self._restore_snapshot()
            self.complete_action()
            return

        if action.id in (
            GameAction.ACTION1,
            GameAction.ACTION2,
            GameAction.ACTION3,
            GameAction.ACTION4,
            GameAction.ACTION5,
            GameAction.ACTION6,
        ):
            self._save_snapshot()
            if self._consume_move():
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

    def _handle_click(self, action: ActionInput) -> None:
        data = action.data
        display_x = data.get("x", -1)
        display_y = data.get("y", -1)

        idx = self._current_level_index
        gw, gh = self._grid_sizes[idx]

        if display_x == -1 and display_y == -1:
            grid_x, grid_y = self._cursor_x, self._cursor_y
        else:
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
        display_grid = self._get_current_grid()
        h, w = base_grid.shape

        if grid_x < 0 or grid_x >= w or grid_y < 0 or grid_y >= h:
            return

        cell = int(base_grid[grid_y, grid_x])
        if cell == C_COIN:
            base_grid[grid_y, grid_x] = C_COIN_COLLECTED
            self._coins_collected[idx].add((grid_x, grid_y))

            traveler = self._traveler_positions[idx]
            goal = self._goal_positions[idx]
            if traveler[0] == goal[0] and traveler[1] == goal[1]:
                required_coins = set()
                if idx < len(LEVEL_COINS):
                    required_coins = set(LEVEL_COINS[idx])
                if not required_coins or required_coins.issubset(
                    self._coins_collected[idx]
                ):
                    np.copyto(display_grid, base_grid)
                    display_grid[goal[1], goal[0]] = C_TRAVELER
                    self._update_sprite_from_grid()
                    self.next_level()
                    return
        elif cell == C_WATER:
            base_grid[grid_y, grid_x] = C_BRIDGE
        elif cell == C_MUDWATER:
            base_grid[grid_y, grid_x] = C_BRIDGE
            self._trap_bridges.add((grid_x, grid_y))
        elif cell == C_BRIDGE:
            base_grid[grid_y, grid_x] = C_WATER
            self._trap_bridges.discard((grid_x, grid_y))
        elif cell == C_SIGNPOST:
            if (grid_x, grid_y) in self._weakened_walls:
                base_grid[grid_y, grid_x] = C_LAND
                self._weakened_walls.discard((grid_x, grid_y))
            else:
                self._weakened_walls.add((grid_x, grid_y))

        np.copyto(display_grid, base_grid)

        for wx, wy in self._weakened_walls:
            if 0 <= wy < h and 0 <= wx < w:
                display_grid[wy, wx] = C_CRACKED

        traveler = self._traveler_positions[idx]
        goal = self._goal_positions[idx]

        path = bfs_path(base_grid, traveler, goal)

        if path is not None and len(path) > 1:
            hit_lava = False
            hit_mine = False
            hit_trap_bridge = False
            death_cell = None
            h, w = base_grid.shape
            for px, py in path[1:]:
                if int(base_grid[py, px]) == C_LAVA:
                    hit_lava = True
                    death_cell = (px, py)
                    break
                if int(base_grid[py, px]) == C_MINE:
                    hit_mine = True
                    death_cell = (px, py)
                    break
                for adx, ady in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                    nx, ny = px + adx, py + ady
                    if 0 <= nx < w and 0 <= ny < h:
                        if int(base_grid[ny, nx]) == C_MINE:
                            hit_mine = True
                            death_cell = (px, py)
                            break
                if hit_mine:
                    break
                if (px, py) in self._trap_bridges:
                    hit_trap_bridge = True
                    death_cell = (px, py)
                    break

            if hit_lava or hit_mine or hit_trap_bridge:
                for step_idx in range(1, len(path)):
                    sx, sy = path[step_idx]
                    prev_sx, prev_sy = path[step_idx - 1]
                    cell_leaving = int(base_grid[prev_sy, prev_sx])
                    if cell_leaving != C_GOAL:
                        display_grid[prev_sy, prev_sx] = C_TRAIL
                    is_lava = int(base_grid[sy, sx]) == C_LAVA
                    is_mine = int(base_grid[sy, sx]) == C_MINE
                    is_mine_adjacent = False
                    if not is_mine:
                        for adx, ady in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                            nx, ny = sx + adx, sy + ady
                            if 0 <= nx < w and 0 <= ny < h:
                                if int(base_grid[ny, nx]) == C_MINE:
                                    is_mine_adjacent = True
                                    break
                    is_trap = (sx, sy) in self._trap_bridges
                    if is_lava or is_mine or is_mine_adjacent or is_trap:
                        if is_trap:
                            base_grid[sy, sx] = C_DEBRIS
                            display_grid[sy, sx] = C_DEBRIS
                        display_grid[sy, sx] = C_TRAVELER
                        self._traveler_positions[idx] = (sx, sy)
                        break
                self._update_sprite_from_grid()
                self._lives -= 1
                if self._lives <= 0:
                    self._game_over = True
                    self.lose()
                else:
                    self.level_reset(preserve_moves=False)
                return

            for step_idx in range(1, len(path)):
                prev_x, prev_y = path[step_idx - 1]
                curr_x, curr_y = path[step_idx]

                cell_leaving = int(base_grid[prev_y, prev_x])
                if cell_leaving != C_GOAL:
                    display_grid[prev_y, prev_x] = C_TRAIL

                if cell_leaving == C_CRACKED:
                    base_grid[prev_y, prev_x] = C_DEBRIS
                    display_grid[prev_y, prev_x] = C_DEBRIS

            final_x, final_y = path[-1]
            self._traveler_positions[idx] = (final_x, final_y)

            display_grid[final_y, final_x] = C_TRAVELER

            if final_x == goal[0] and final_y == goal[1]:
                required_coins = set()
                if idx < len(LEVEL_COINS):
                    required_coins = set(LEVEL_COINS[idx])
                if required_coins and not required_coins.issubset(
                    self._coins_collected[idx]
                ):
                    pass
                else:
                    display_grid[final_y, final_x] = C_TRAVELER
                    self._update_sprite_from_grid()
                    self.next_level()
                    return
        else:
            display_grid[traveler[1], traveler[0]] = C_TRAVELER

        self._update_sprite_from_grid()

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
            frame[y, x] = C_LAND


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
                frame[bar_y, px] = C_RED
            else:
                frame[bar_y, px] = C_WALL


def _draw_coins(frame: np.ndarray, collected: int, total: int) -> None:
    if total == 0:
        return
    dot_size = 1
    gap = 1
    bar_y = HUD_TOP + 2

    for i in range(total):
        px = 1 + i * (dot_size + gap)
        if 0 <= px < DISPLAY_SIZE and 0 <= bar_y < DISPLAY_SIZE:
            if i < collected:
                frame[bar_y, px] = C_COIN_COLLECTED
            else:
                frame[bar_y, px] = C_WALL


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
                frame[bar_y, x] = C_GOAL
        for x in range(bar_x_start + fill_length, bar_x_end):
            if 0 <= x < DISPLAY_SIZE:
                frame[bar_y, x] = C_WALL


class _BridgeBuilderOverlay(RenderableUserDisplay):
    def __init__(self, game: Ms04) -> None:
        self._game = game

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        game = self._game
        idx = game.level_index
        if idx >= len(game._grid_sizes):
            return frame

        gw, gh = game._grid_sizes[idx]
        scale, x_off, y_off = _grid_layout(gw, gh)

        _draw_hud_background(frame)

        FILLED_CELLS = {
            C_WALL,
            C_WATER,
            C_MUDWATER,
            C_BRIDGE,
            C_TRAVELER,
            C_GOAL,
            C_TRAIL,
            C_CRACKED,
            C_DEBRIS,
            C_LAVA,
            C_MINE,
            C_SIGNPOST,
            C_COIN_COLLECTED,
        }

        if idx < len(game._grids) and scale >= 3:
            disp = game._grids[idx]
            dh, dw = disp.shape
            for gy in range(dh):
                for gx in range(dw):
                    cell_val = int(disp[gy, gx])
                    if cell_val not in FILLED_CELLS:
                        continue
                    px_left = x_off + gx * scale
                    px_top = y_off + gy * scale
                    above_val = int(disp[gy - 1, gx]) if gy > 0 else -1
                    if above_val != cell_val:
                        for dx in range(scale):
                            px, py = px_left + dx, px_top
                            if 0 <= px < 64 and 0 <= py < HUD_TOP:
                                frame[py, px] = C_GRID_LINE
                    left_val = int(disp[gy, gx - 1]) if gx > 0 else -1
                    if left_val != cell_val:
                        for dy in range(scale):
                            px, py = px_left, px_top + dy
                            if 0 <= px < 64 and 0 <= py < HUD_TOP:
                                frame[py, px] = C_GRID_LINE

        grid_y_end = min(HUD_TOP, y_off + gh * scale)
        grid_x_end = min(DISPLAY_SIZE, x_off + gw * scale)

        if scale >= 3:
            for i in range(gw + 1):
                px = x_off + i * scale
                if 0 <= px < DISPLAY_SIZE:
                    for py in range(y_off, grid_y_end):
                        if int(frame[py, px]) == C_LAND:
                            frame[py, px] = C_GRID_LINE
            for i in range(gh + 1):
                py = y_off + i * scale
                if 0 <= py < HUD_TOP:
                    for px in range(x_off, grid_x_end):
                        if int(frame[py, px]) == C_LAND:
                            frame[py, px] = C_GRID_LINE
        else:
            py = y_off
            if 0 <= py < HUD_TOP:
                for px in range(x_off, grid_x_end):
                    if int(frame[py, px]) == C_LAND:
                        frame[py, px] = C_GRID_LINE
            py = y_off + gh * scale
            if 0 <= py < HUD_TOP:
                for px in range(x_off, grid_x_end):
                    if int(frame[py, px]) == C_LAND:
                        frame[py, px] = C_GRID_LINE
            px = x_off
            if 0 <= px < DISPLAY_SIZE:
                for py_i in range(y_off, grid_y_end):
                    if int(frame[py_i, px]) == C_LAND:
                        frame[py_i, px] = C_GRID_LINE
            px = x_off + gw * scale
            if 0 <= px < DISPLAY_SIZE:
                for py_i in range(y_off, grid_y_end):
                    if int(frame[py_i, px]) == C_LAND:
                        frame[py_i, px] = C_GRID_LINE

        if idx < len(game._grids) and scale >= 3:
            disp = game._grids[idx]
            dh, dw = disp.shape
            for gy in range(dh):
                for gx in range(dw):
                    cell_v = int(disp[gy, gx])
                    if cell_v != C_COIN and cell_v != C_COIN_COLLECTED:
                        continue
                    dot_color = (
                        C_COIN_COLLECTED if cell_v == C_COIN_COLLECTED else C_COIN
                    )
                    px_left = x_off + gx * scale
                    px_top = y_off + gy * scale
                    for dy in range(scale):
                        for dx in range(scale):
                            px, py = px_left + dx, px_top + dy
                            if 0 <= px < 64 and 0 <= py < HUD_TOP:
                                frame[py, px] = C_LAND
                    for dx in range(scale):
                        px, py = px_left + dx, px_top
                        if 0 <= px < 64 and 0 <= py < HUD_TOP:
                            frame[py, px] = C_GRID_LINE
                    for dy in range(scale):
                        px, py = px_left, px_top + dy
                        if 0 <= px < 64 and 0 <= py < HUD_TOP:
                            frame[py, px] = C_GRID_LINE
                    if scale <= 4:
                        dot_x = px_left + scale // 2
                        dot_y = px_top + scale // 2
                        if 0 <= dot_x < 64 and 0 <= dot_y < HUD_TOP:
                            frame[dot_y, dot_x] = dot_color
                    else:
                        dot_offset = max(1, (scale - 2) // 2)
                        for dy in range(2):
                            for dx in range(2):
                                px = px_left + dot_offset + dx
                                py = px_top + dot_offset + dy
                                if 0 <= px < 64 and 0 <= py < HUD_TOP:
                                    frame[py, px] = dot_color

        moves_used = game._moves_used[idx]
        max_moves = LEVEL_MAX_MOVES[idx]
        _draw_progress_bar(frame, moves_used, max_moves)

        _draw_lives(frame, game._lives)

        if idx < len(LEVEL_COINS) and LEVEL_COINS[idx]:
            total_coins = len(LEVEL_COINS[idx])
            collected_coins = len(game._coins_collected[idx])
            _draw_coins(frame, collected_coins, total_coins)

        _draw_cursor_box(frame, game._cursor_x, game._cursor_y, gw, gh)

        return frame


_CELL_CHARS: Dict[int, str] = {
    C_LAND: ".",
    C_WATER: "~",
    C_MUDWATER: "%",
    C_WALL: "#",
    C_BRIDGE: "=",
    C_TRAVELER: "@",
    C_GOAL: "G",
    C_LAVA: "!",
    C_MINE: "*",
    C_SIGNPOST: "S",
    C_COIN: "o",
    C_COIN_COLLECTED: "x",
    C_CRACKED: "c",
    C_DEBRIS: "d",
    C_CURSOR: ".",
    C_GRID_LINE: ".",
    C_TRAIL: "x",
}

_ACTION_MAP: Dict[str, GameAction] = {
    "up": GameAction.ACTION1,
    "down": GameAction.ACTION2,
    "left": GameAction.ACTION3,
    "right": GameAction.ACTION4,
    "select": GameAction.ACTION5,
    "click": GameAction.ACTION6,
    "undo": GameAction.ACTION7,
    "reset": GameAction.RESET,
}

_VALID_ACTIONS: List[str] = [
    "up",
    "down",
    "left",
    "right",
    "select",
    "click",
    "undo",
    "reset",
]


class PuzzleEnvironment:
    ARC_PALETTE: List[Tuple[int, int, int]] = [
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
        self._engine = Ms04(seed=seed)
        self._turn: int = 0
        self._last_action_was_reset: bool = False

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

    def _build_text_observation(self) -> str:
        g = self._engine
        idx = g._current_level_index
        gw, gh = g._grid_sizes[idx]
        grid = g._grids[idx]

        lines: List[str] = []
        lines.append(f"Level {idx + 1}/{len(LEVEL_GENERATORS)}")
        lines.append(f"Moves: {g._moves_used[idx]}/{LEVEL_MAX_MOVES[idx]}")
        lines.append(f"Lives: {g._lives}/{TOTAL_LIVES}")

        if idx < len(LEVEL_COINS) and LEVEL_COINS[idx]:
            total_coins = len(LEVEL_COINS[idx])
            collected = len(g._coins_collected[idx])
            lines.append(f"Coins: {collected}/{total_coins}")

        if g._game_over:
            lines.append("Status: GAME OVER")

        lines.append(f"Cursor: ({g._cursor_x},{g._cursor_y})")
        lines.append("")

        for gy in range(gh):
            row = ""
            for gx in range(gw):
                cell = int(grid[gy, gx])
                row += _CELL_CHARS.get(cell, "?")
            lines.append(row)

        return "\n".join(lines)

    def _build_metadata(self) -> Dict[str, Any]:
        g = self._engine
        idx = g._current_level_index
        return {
            "total_levels": len(LEVEL_GENERATORS),
            "level": idx + 1,
            "moves_used": g._moves_used[idx],
            "move_limit": LEVEL_MAX_MOVES[idx],
            "lives": g._lives,
            "max_lives": TOTAL_LIVES,
            "game_over": g._game_over,
            "levels_completed": getattr(g, "_score", 0),
            "level_index": idx,
        }

    def _make_game_state(self) -> GameState:
        valid: Optional[List[str]] = (
            ["reset"] if self._engine._game_over else list(_VALID_ACTIONS)
        )
        return GameState(
            text_observation=self._build_text_observation(),
            image_observation=self._build_image_bytes(),
            valid_actions=valid,
            turn=self._turn,
            metadata=self._build_metadata(),
        )

    def _is_won(self) -> bool:
        return self._engine._state == EngineGameState.WIN

    def reset(self) -> GameState:
        self._turn = 0
        if self._is_won() or self._last_action_was_reset:
            self._engine.full_reset()
            self._engine.perform_action(ActionInput(id=GameAction.RESET))
        else:
            self._engine.perform_action(ActionInput(id=GameAction.RESET))
        self._last_action_was_reset = True
        return self._make_game_state()

    def get_actions(self) -> List[str]:
        if self._engine._game_over:
            return ["reset"]
        return list(_VALID_ACTIONS)

    def step(self, action: str) -> StepResult:
        self._last_action_was_reset = False
        action = action.strip().lower()

        if action == "reset":
            state = self.reset()
            return StepResult(
                state=state, reward=0.0, done=self.is_done(), info={"action": "reset"}
            )

        game_action = _ACTION_MAP.get(action)
        if game_action is None:
            raise ValueError(f"Unknown action '{action}'. Valid: {_VALID_ACTIONS}")

        prev_level = self._engine._current_level_index

        if game_action == GameAction.ACTION6:
            parts = action.replace("click", "").strip().split()
            x_val = int(parts[0]) if len(parts) > 0 and parts[0].isdigit() else 0
            y_val = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
            self._engine.perform_action(
                ActionInput(
                    id=GameAction.ACTION6,
                    data={"x": x_val, "y": y_val},
                )
            )
        else:
            self._engine.perform_action(ActionInput(id=game_action))

        if game_action != GameAction.RESET:
            self._turn += 1

        engine_state = self._engine._state
        done = engine_state in (
            EngineGameState.WIN,
            EngineGameState.GAME_OVER,
        )

        reward = 0.0
        if engine_state == EngineGameState.WIN:
            reward = 1.0 / len(LEVEL_GENERATORS)
        elif self._engine._current_level_index > prev_level:
            reward = 1.0 / len(LEVEL_GENERATORS)

        info: Dict[str, Any] = {
            "action": action,
            "engine_state": engine_state,
            "level_changed": self._engine._current_level_index != prev_level,
        }

        return StepResult(
            state=self._make_game_state(),
            reward=reward,
            done=done,
            info=info,
        )

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
        for idx, color in enumerate(self.ARC_PALETTE):
            mask = index_grid == idx
            rgb[mask] = color
        return rgb

    def get_state(self) -> GameState:
        return self._make_game_state()

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
