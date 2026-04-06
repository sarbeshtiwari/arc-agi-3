import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import gymnasium as gym
import numpy as np
from arcengine import ActionInput, ARCBaseGame, Camera, GameAction, Level, Sprite
from arcengine.interfaces import RenderableUserDisplay
from gymnasium import spaces


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

CELL_EMPTY = 0
CELL_WALL = 1
CELL_PLAYER = 2
CELL_AI = 3
CELL_SPEED = 5
CELL_TRAP = 6
CELL_BARRIER = 7
CELL_BONUS = 8
CELL_LOCKED = 9

CELL_COLORS = {
    CELL_EMPTY: C_DGREY,
    CELL_WALL: C_GREY,
    CELL_PLAYER: C_BLUE,
    CELL_AI: C_ORANGE,
    CELL_SPEED: C_GREEN,
    CELL_TRAP: C_RED,
    CELL_BARRIER: C_MAROON,
    CELL_BONUS: C_YELLOW,
    CELL_LOCKED: C_LBLUE,
}

MAX_LIVES = 3
NUM_LEVELS = 4

AI_RANDOM = 0
AI_GREEDY = 1
AI_SPEED_SEEKER = 2
AI_TRAP_HERDER = 3
AI_CUTTER = 4

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

SPEED_BOOST_TURNS = 2
TRAP_FREEZE_TURNS = 1
AI_SPEED_BOOST_TURNS = 2
SPEED_STEAL_CELLS = 2


def _validate_placements(
    rows: int,
    cols: int,
    player_start: Tuple[int, int],
    ai_starts: List[Tuple[int, int]],
    walls: List[Tuple[int, int]],
    speeds: List[Tuple[int, int]],
    traps: List[Tuple[int, int]],
    barriers: List[Tuple[int, int]],
    bonus_zone: List[Tuple[int, int]],
) -> None:
    occupied: Set[Tuple[int, int]] = set()
    occupied.add(player_start)
    for pos in ai_starts:
        occupied.add(pos)
    for pos in walls:
        if pos in occupied:
            raise ValueError(f"Wall overlap at {pos}")
        occupied.add(pos)
    for pos in barriers:
        if pos in occupied:
            raise ValueError(f"Barrier overlap at {pos}")
        occupied.add(pos)
    for pos in bonus_zone:
        if pos in occupied:
            raise ValueError(f"Bonus overlap at {pos}")
        occupied.add(pos)
    for pos in speeds:
        if pos in occupied:
            raise ValueError(f"Speed overlap at {pos}")
        occupied.add(pos)
    for pos in traps:
        if pos in occupied:
            raise ValueError(f"Trap overlap at {pos}")
        occupied.add(pos)


def _build_level_2() -> Dict:
    rows, cols, px = 12, 12, 2
    grid = [[CELL_EMPTY] * cols for _ in range(rows)]
    walls = [(2, 2), (9, 9)]
    barriers = []
    speeds = [(5, 5), (6, 6)]
    traps = [(4, 5), (5, 4), (5, 6), (6, 5), (6, 7), (7, 6)]
    bonus_zone = [(0, 11), (11, 0)]
    player_start = (0, 0)
    ai_starts = [(11, 11)]
    _validate_placements(
        rows, cols, player_start, ai_starts, walls, speeds, traps, barriers, bonus_zone
    )
    for r, c in walls:
        grid[r][c] = CELL_WALL
    return {
        "rows": rows,
        "cols": cols,
        "px": px,
        "grid": grid,
        "player_start": player_start,
        "ai_starts": ai_starts,
        "ai_types": [AI_SPEED_SEEKER],
        "speeds": speeds,
        "traps": traps,
        "barriers": barriers,
        "bonus_zone": bonus_zone,
        "max_moves": 200,
        "ai_move_freq": 1,
        "win_pct": 35,
    }


def _build_level_3() -> Dict:
    rows, cols, px = 14, 14, 2
    grid = [[CELL_EMPTY] * cols for _ in range(rows)]
    walls = [(5, 5), (5, 8), (8, 5), (8, 8)]
    barriers = []
    speeds = [(0, 6), (13, 7)]
    traps = [(6, 6), (6, 7), (7, 6), (7, 7)]
    bonus_zone = [(0, 13), (1, 13), (2, 13), (13, 0), (12, 0), (11, 0)]
    player_start = (0, 0)
    ai_starts = [(13, 13)]
    _validate_placements(
        rows, cols, player_start, ai_starts, walls, speeds, traps, barriers, bonus_zone
    )
    for r, c in walls:
        grid[r][c] = CELL_WALL
    return {
        "rows": rows,
        "cols": cols,
        "px": px,
        "grid": grid,
        "player_start": player_start,
        "ai_starts": ai_starts,
        "ai_types": [AI_TRAP_HERDER],
        "speeds": speeds,
        "traps": traps,
        "barriers": barriers,
        "bonus_zone": bonus_zone,
        "max_moves": 300,
        "ai_move_freq": 1,
        "win_pct": 38,
    }


def _build_level_4() -> Dict:
    rows, cols, px = 16, 16, 2
    grid = [[CELL_EMPTY] * cols for _ in range(rows)]
    walls = [
        (5, 5),
        (5, 6),
        (5, 8),
        (5, 9),
        (9, 5),
        (9, 6),
        (9, 8),
        (9, 9),
        (6, 5),
        (8, 5),
        (6, 9),
        (8, 9),
    ]
    bonus_zone = [(5, 7), (9, 7), (7, 5), (7, 9)]
    traps = [(6, 6), (8, 8)]
    barriers = []
    speeds = [(2, 7), (13, 7), (7, 2), (7, 13)]
    player_start = (0, 0)
    ai_starts = [(7, 7)]
    _validate_placements(
        rows, cols, player_start, ai_starts, walls, speeds, traps, barriers, bonus_zone
    )
    for r, c in walls:
        grid[r][c] = CELL_WALL
    return {
        "rows": rows,
        "cols": cols,
        "px": px,
        "grid": grid,
        "player_start": player_start,
        "ai_starts": ai_starts,
        "ai_types": [AI_CUTTER],
        "speeds": speeds,
        "traps": traps,
        "barriers": barriers,
        "bonus_zone": bonus_zone,
        "max_moves": 500,
        "ai_move_freq": 3,
        "win_pct": 35,
    }


def _build_level_5() -> Dict:
    rows, cols, px = 18, 18, 2
    grid = [[CELL_EMPTY] * cols for _ in range(rows)]
    walls = [(3, 3), (14, 14)]
    barriers = [
        (7, 8),
        (7, 9),
        (7, 10),
        (10, 7),
        (10, 8),
        (10, 9),
        (8, 7),
        (9, 7),
        (8, 10),
        (9, 10),
    ]
    speeds = [(1, 8), (1, 9), (16, 8), (8, 1), (5, 13), (12, 4)]
    traps = [(6, 6), (11, 11)]
    bonus_zone = [(8, 8), (8, 9), (9, 8), (9, 9)]
    player_start = (0, 0)
    ai_starts = [(17, 17), (17, 0)]
    _validate_placements(
        rows, cols, player_start, ai_starts, walls, speeds, traps, barriers, bonus_zone
    )
    for r, c in walls:
        grid[r][c] = CELL_WALL
    return {
        "rows": rows,
        "cols": cols,
        "px": px,
        "grid": grid,
        "player_start": player_start,
        "ai_starts": ai_starts,
        "ai_types": [AI_SPEED_SEEKER, AI_GREEDY],
        "speeds": speeds,
        "traps": traps,
        "barriers": barriers,
        "bonus_zone": bonus_zone,
        "max_moves": 600,
        "ai_move_freq": 2,
        "win_pct": 42,
    }


LEVEL_BUILDERS = [
    _build_level_2,
    _build_level_3,
    _build_level_4,
    _build_level_5,
]

PLAYER_START_POSITIONS = [
    [(0, 0), (0, 6), (6, 0), (11, 5)],
    [(0, 0), (0, 7), (7, 0), (13, 6)],
    [(0, 0), (0, 15), (15, 0), (15, 15)],
    [(0, 0), (0, 17), (9, 0), (0, 9)],
]


def _build_levels() -> List[Level]:
    result: List[Level] = []
    for i, builder in enumerate(LEVEL_BUILDERS):
        data = builder()
        level = Level(sprites=[], grid_size=(64, 64), data=data, name=f"Level {i + 1}")
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
        self.max_moves: int = 1
        self.player_frozen: bool = False
        self.player_speed: bool = False
        self.win_pct: int = 35
        self.player_pct: int = 0
        self.ai_pct: int = 0

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        for r in range(6):
            for c in range(64):
                frame[r, c] = C_BLACK

        bar_w = 60
        if self.max_moves > 0:
            moves_left = max(0, self.max_moves - self.moves_used)
            ratio = moves_left / self.max_moves
            filled = int(bar_w * ratio)
        else:
            ratio = 1.0
            filled = bar_w

        bar_col = C_GREEN if ratio > 0.5 else (C_ORANGE if ratio > 0.25 else C_RED)
        for c in range(bar_w):
            col_c = 2 + c
            if col_c < 64:
                frame[0, col_c] = bar_col if c < filled else C_DGREY

        race_w = 60
        blue_fill = min(race_w, int(race_w * self.player_pct / 100))
        orange_fill = min(race_w, int(race_w * self.ai_pct / 100))
        target_idx = min(race_w - 1, int(race_w * self.win_pct / 100))

        for c in range(race_w):
            cx = 2 + c
            if cx < 64:
                frame[2, cx] = C_BLUE if c < blue_fill else C_DGREY
                frame[3, cx] = C_ORANGE if c < orange_fill else C_DGREY
                if c == target_idx:
                    frame[1, cx] = C_WHITE
                    frame[2, cx] = C_WHITE
                    frame[3, cx] = C_WHITE
                    frame[4, cx] = C_WHITE

        for i in range(self.total_levels):
            if i < self.current_level:
                dc = C_PURPLE
            elif i == self.current_level:
                dc = C_LBLUE
            else:
                dc = C_DGREY
            x = 2 + i * 4
            if x < 64:
                frame[5, x] = dc
            if x + 1 < 64:
                frame[5, x + 1] = dc

        for i in range(self.max_lives):
            lc = C_GREEN if i < self.lives else C_DGREY
            bx = 52 + i * 4
            if bx < 64:
                frame[5, bx] = lc
            if bx + 1 < 64:
                frame[5, bx + 1] = lc

        if self.player_frozen:
            frame[5, 28] = C_RED
            frame[5, 29] = C_RED
        if self.player_speed:
            frame[5, 32] = C_GREEN
            frame[5, 33] = C_GREEN

        return frame


def _flood_reachable(
    grid: List[List[int]],
    start_r: int,
    start_c: int,
    rows: int,
    cols: int,
    own_cell: int,
) -> int:
    visited: Set[Tuple[int, int]] = set()
    stack = [(start_r, start_c)]
    count = 0
    while stack:
        r, c = stack.pop()
        if (r, c) in visited:
            continue
        visited.add((r, c))
        cell = grid[r][c]
        if cell in (CELL_WALL, CELL_BARRIER, CELL_LOCKED, own_cell):
            continue
        count += 1
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                stack.append((nr, nc))
    return count


def _ai_move_random(
    grid: List[List[int]],
    ar: int,
    ac: int,
    rows: int,
    cols: int,
    own_cell: int,
    seed_val: int,
) -> Tuple[int, int]:
    good_options = []
    fallback = []
    for dr, dc in DIRS:
        nr, nc = ar + dr, ac + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            cell = grid[nr][nc]
            if cell in (CELL_WALL, CELL_BARRIER, CELL_LOCKED):
                continue
            if cell != own_cell:
                good_options.append((nr, nc))
            else:
                fallback.append((nr, nc))
    options = good_options if good_options else fallback
    if not options:
        return ar, ac
    return options[seed_val % len(options)]


def _ai_move_greedy(
    grid: List[List[int]],
    ar: int,
    ac: int,
    rows: int,
    cols: int,
    own_cell: int,
    seed_val: int,
) -> Tuple[int, int]:
    options = []
    for dr, dc in DIRS:
        nr, nc = ar + dr, ac + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            cell = grid[nr][nc]
            if cell in (CELL_WALL, CELL_BARRIER, CELL_LOCKED):
                continue
            if cell != own_cell:
                reach = _flood_reachable(grid, nr, nc, rows, cols, own_cell)
                options.append((nr, nc, reach))
    if not options:
        for dr, dc in DIRS:
            nr, nc = ar + dr, ac + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                cell = grid[nr][nc]
                if cell not in (CELL_WALL, CELL_BARRIER, CELL_LOCKED):
                    options.append((nr, nc, 0))
    if not options:
        return ar, ac
    options.sort(key=lambda x: -x[2])
    best_reach = options[0][2]
    ties = [o for o in options if o[2] == best_reach]
    pick = ties[seed_val % len(ties)]
    return pick[0], pick[1]


def _ai_move_speed_seeker(
    grid: List[List[int]],
    ar: int,
    ac: int,
    rows: int,
    cols: int,
    own_cell: int,
    speed_cells: List[Tuple[int, int]],
    seed_val: int,
) -> Tuple[int, int]:
    for sr, sc in speed_cells:
        if grid[sr][sc] == CELL_SPEED:
            best_dir = None
            best_dist = abs(ar - sr) + abs(ac - sc)
            for dr, dc in DIRS:
                nr, nc = ar + dr, ac + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    cell = grid[nr][nc]
                    if cell in (CELL_WALL, CELL_BARRIER, CELL_LOCKED):
                        continue
                    if cell != own_cell:
                        d = abs(nr - sr) + abs(nc - sc)
                        if d < best_dist:
                            best_dist = d
                            best_dir = (nr, nc)
            if best_dir:
                return best_dir
    return _ai_move_greedy(grid, ar, ac, rows, cols, own_cell, seed_val)


def _ai_move_trap_herder(
    grid: List[List[int]],
    ar: int,
    ac: int,
    pr: int,
    pc: int,
    rows: int,
    cols: int,
    own_cell: int,
    trap_cells: List[Tuple[int, int]],
    seed_val: int,
) -> Tuple[int, int]:
    nearest_trap = None
    nearest_dist = 9999
    for tr, tc in trap_cells:
        if grid[tr][tc] == CELL_TRAP:
            d = abs(pr - tr) + abs(pc - tc)
            if d < nearest_dist:
                nearest_dist = d
                nearest_trap = (tr, tc)

    if nearest_trap and nearest_dist < 6:
        target_r = pr
        target_c = pc
        tr, tc = nearest_trap
        if pr < tr:
            target_r = pr - 1
        elif pr > tr:
            target_r = pr + 1
        if pc < tc:
            target_c = pc - 1
        elif pc > tc:
            target_c = pc + 1

        best_dir = None
        best_dist = 9999
        for dr, dc in DIRS:
            nr, nc = ar + dr, ac + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                cell = grid[nr][nc]
                if cell not in (CELL_WALL, CELL_BARRIER, CELL_LOCKED, own_cell):
                    d = abs(nr - target_r) + abs(nc - target_c)
                    if d < best_dist:
                        best_dist = d
                        best_dir = (nr, nc)
        if best_dir:
            return best_dir

    return _ai_move_greedy(grid, ar, ac, rows, cols, own_cell, seed_val)


def _ai_move_cutter(
    grid: List[List[int]],
    ar: int,
    ac: int,
    pr: int,
    pc: int,
    rows: int,
    cols: int,
    own_cell: int,
    seed_val: int,
) -> Tuple[int, int]:
    best_pos = None
    best_score = -9999
    for dr, dc in DIRS:
        nr, nc = ar + dr, ac + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            cell = grid[nr][nc]
            if cell in (CELL_WALL, CELL_BARRIER, CELL_LOCKED):
                continue
            if cell != own_cell:
                temp_grid = [row[:] for row in grid]
                temp_grid[nr][nc] = own_cell
                player_reach = _flood_reachable(
                    temp_grid, pr, pc, rows, cols, CELL_PLAYER
                )
                ai_reach = _flood_reachable(temp_grid, nr, nc, rows, cols, own_cell)
                closeness = -(abs(nr - pr) + abs(nc - pc))
                score = ai_reach - player_reach * 2 + closeness
                if score > best_score:
                    best_score = score
                    best_pos = (nr, nc)
    if best_pos:
        return best_pos
    return _ai_move_random(grid, ar, ac, rows, cols, own_cell, seed_val)


class Nt01(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self._rng = random.Random(seed)
        self._hud = _HUD()
        self._rows: int = 0
        self._cols: int = 0
        self._px: int = 2
        self._max_moves: int = 1
        self._moves_used: int = 0
        self._lives: int = MAX_LIVES
        self._level_just_loaded: bool = False
        self._grid: List[List[int]] = []
        self._initial_grid: List[List[int]] = []
        self._player_r: int = 0
        self._player_c: int = 0
        self._player_start: Tuple[int, int] = (0, 0)
        self._ai_positions: List[Tuple[int, int]] = []
        self._ai_starts: List[Tuple[int, int]] = []
        self._ai_types: List[int] = []
        self._speed_cells: List[Tuple[int, int]] = []
        self._trap_cells: List[Tuple[int, int]] = []
        self._barrier_cells: List[Tuple[int, int]] = []
        self._player_frozen: int = 0
        self._player_speed: int = 0
        self._ai_frozen: List[int] = []
        self._ai_speed: List[int] = []
        self._step_counter: int = 0
        self._undo_stack: List[Dict] = []
        self._render_ox: int = 0
        self._render_oy: int = 0
        self._ai_move_freq: int = 1
        self._win_pct: int = 35

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
            "nt01",
            levels,
            cam,
            available_actions=[0, 1, 2, 3, 4, 7],
            win_score=NUM_LEVELS,
            seed=seed,
        )

    def _save_state(self) -> Dict:
        return {
            "grid": [row[:] for row in self._grid],
            "player_r": self._player_r,
            "player_c": self._player_c,
            "ai_positions": list(self._ai_positions),
            "moves_used": self._moves_used,
            "step_counter": self._step_counter,
            "player_frozen": self._player_frozen,
            "player_speed": self._player_speed,
            "ai_frozen": list(self._ai_frozen),
            "ai_speed": list(self._ai_speed),
            "lives": self._lives,
        }

    def _restore_state(self, state) -> None:
        self._grid = [row[:] for row in state["grid"]]
        self._player_r = state["player_r"]
        self._player_c = state["player_c"]
        self._ai_positions = list(state["ai_positions"])
        self._moves_used = state["moves_used"]
        self._step_counter = state["step_counter"]
        self._player_frozen = state["player_frozen"]
        self._player_speed = state["player_speed"]
        self._ai_frozen = list(state["ai_frozen"])
        self._ai_speed = list(state["ai_speed"])
        self._lives = state["lives"]
        self._sync_hud()
        self._rebuild()

    def handle_reset(self) -> None:
        self._lives = MAX_LIVES
        self._undo_stack = []
        super().handle_reset()

    def _count_paintable_from_initial(self) -> int:
        total = self._rows * self._cols
        for r in range(self._rows):
            for c in range(self._cols):
                if self._initial_grid[r][c] == CELL_WALL:
                    total -= 1
        for _ in self._barrier_cells:
            total -= 1
        return max(1, total)

    def _compute_player_pct(self) -> int:
        p_count = 0
        for r in range(self._rows):
            for c in range(self._cols):
                if self._grid[r][c] in (CELL_PLAYER, CELL_LOCKED):
                    p_count += 1
        paintable = self._count_paintable_from_initial()
        return int(100 * p_count / paintable) if paintable > 0 else 0

    def _compute_ai_pct(self) -> int:
        ai_count = 0
        for r in range(self._rows):
            for c in range(self._cols):
                if self._grid[r][c] == CELL_AI:
                    ai_count += 1
        paintable = self._count_paintable_from_initial()
        return int(100 * ai_count / paintable) if paintable > 0 else 0

    def on_set_level(self, level: Level) -> None:
        self._rows = level.get_data("rows")
        self._cols = level.get_data("cols")
        self._px = level.get_data("px")
        self._max_moves = level.get_data("max_moves")
        self._moves_used = 0
        self._step_counter = 0
        self._lives = MAX_LIVES
        self._ai_move_freq = level.get_data("ai_move_freq")
        self._win_pct = level.get_data("win_pct")
        self._level_just_loaded = True

        base_grid = level.get_data("grid")
        self._initial_grid = [row[:] for row in base_grid]
        self._grid = [row[:] for row in base_grid]

        self._player_start = self._rng.choice(PLAYER_START_POSITIONS[self.level_index])
        self._player_r, self._player_c = self._player_start
        self._grid[self._player_r][self._player_c] = CELL_PLAYER

        self._ai_starts = list(level.get_data("ai_starts"))
        self._ai_types = list(level.get_data("ai_types"))
        self._ai_positions = list(self._ai_starts)
        for ar, ac in self._ai_positions:
            self._grid[ar][ac] = CELL_AI

        self._speed_cells = list(level.get_data("speeds"))
        for sr, sc in self._speed_cells:
            if self._grid[sr][sc] == CELL_EMPTY:
                self._grid[sr][sc] = CELL_SPEED

        self._trap_cells = list(level.get_data("traps"))
        for tr, tc in self._trap_cells:
            if self._grid[tr][tc] == CELL_EMPTY:
                self._grid[tr][tc] = CELL_TRAP

        self._barrier_cells = list(level.get_data("barriers"))
        for br, bc in self._barrier_cells:
            if self._grid[br][bc] == CELL_EMPTY:
                self._grid[br][bc] = CELL_BARRIER

        bonus_zone = list(level.get_data("bonus_zone"))
        for br, bc in bonus_zone:
            if self._grid[br][bc] == CELL_EMPTY:
                self._grid[br][bc] = CELL_BONUS

        self._player_frozen = 0
        self._player_speed = 0
        self._ai_frozen = [0] * len(self._ai_positions)
        self._ai_speed = [0] * len(self._ai_positions)
        self._undo_stack = []

        self._rebuild()
        self._sync_hud()

    def _reset_current_level(self) -> None:
        self._moves_used = 0
        self._step_counter = 0

        self._grid = [row[:] for row in self._initial_grid]
        self._player_start = self._rng.choice(PLAYER_START_POSITIONS[self.level_index])
        self._player_r, self._player_c = self._player_start
        self._grid[self._player_r][self._player_c] = CELL_PLAYER

        self._ai_positions = list(self._ai_starts)
        for ar, ac in self._ai_positions:
            self._grid[ar][ac] = CELL_AI

        for sr, sc in self._speed_cells:
            if self._grid[sr][sc] == CELL_EMPTY:
                self._grid[sr][sc] = CELL_SPEED
        for tr, tc in self._trap_cells:
            if self._grid[tr][tc] == CELL_EMPTY:
                self._grid[tr][tc] = CELL_TRAP
        for br, bc in self._barrier_cells:
            if self._grid[br][bc] == CELL_EMPTY:
                self._grid[br][bc] = CELL_BARRIER

        bonus_zone = list(self.current_level.get_data("bonus_zone"))
        for br, bc in bonus_zone:
            if self._grid[br][bc] == CELL_EMPTY:
                self._grid[br][bc] = CELL_BONUS

        self._player_frozen = 0
        self._player_speed = 0
        self._ai_frozen = [0] * len(self._ai_positions)
        self._ai_speed = [0] * len(self._ai_positions)
        self._undo_stack = []

        self._rebuild()
        self._sync_hud()

    def _sync_hud(self) -> None:
        self._hud.lives = self._lives
        self._hud.current_level = self.level_index
        self._hud.moves_used = self._moves_used
        self._hud.max_moves = self._max_moves
        self._hud.player_frozen = self._player_frozen > 0
        self._hud.player_speed = self._player_speed > 0
        self._hud.win_pct = self._win_pct
        self._hud.player_pct = self._compute_player_pct()
        self._hud.ai_pct = self._compute_ai_pct()

    def _rebuild(self) -> None:
        self.current_level.remove_all_sprites()
        px = self._px
        gap = 1 if px <= 2 else 0
        total_w = self._cols * px + max(0, self._cols - 1) * gap
        total_h = self._rows * px + max(0, self._rows - 1) * gap

        area_y0 = 6
        area_x0 = 0
        avail_w = 64
        avail_h = 64 - area_y0

        self._render_ox = area_x0 + max(0, (avail_w - total_w) // 2)
        self._render_oy = area_y0 + max(0, (avail_h - total_h) // 2)

        for r in range(self._rows):
            for c in range(self._cols):
                cell = self._grid[r][c]
                color = CELL_COLORS.get(cell, C_DGREY)
                pixels = [[color] * px for _ in range(px)]
                if cell == CELL_LOCKED:
                    if px >= 3:
                        pixels[px // 2][px // 2] = C_WHITE
                    elif px == 2:
                        pixels[0][0] = C_WHITE
                self.current_level.add_sprite(
                    Sprite(
                        pixels=pixels,
                        name=f"cell_{r}_{c}",
                        x=self._render_ox + c * (px + gap),
                        y=self._render_oy + r * (px + gap),
                        layer=0,
                        tags=["cell"],
                    )
                )

        pr, pc = self._player_r, self._player_c
        p_pixels = [[C_LBLUE] * px for _ in range(px)]
        if px >= 3:
            p_pixels[px // 2][px // 2] = C_WHITE
        self.current_level.add_sprite(
            Sprite(
                pixels=p_pixels,
                name="player",
                x=self._render_ox + pc * (px + gap),
                y=self._render_oy + pr * (px + gap),
                layer=2,
                tags=["player"],
            )
        )

        for i, (ar, ac) in enumerate(self._ai_positions):
            a_pixels = [[C_ORANGE] * px for _ in range(px)]
            if px >= 3:
                a_pixels[px // 2][px // 2] = C_YELLOW
            self.current_level.add_sprite(
                Sprite(
                    pixels=a_pixels,
                    name=f"ai_{i}",
                    x=self._render_ox + ac * (px + gap),
                    y=self._render_oy + ar * (px + gap),
                    layer=2,
                    tags=["ai"],
                )
            )

    def _can_move_to(self, r: int, c: int) -> bool:
        if r < 0 or r >= self._rows or c < 0 or c >= self._cols:
            return False
        if self._grid[r][c] == CELL_WALL:
            return False
        return True

    def _steal_cells_near(self, r: int, c: int, count: int) -> None:
        stolen = 0
        for dist in range(1, 4):
            if stolen >= count:
                break
            for dr in range(-dist, dist + 1):
                for dc_val in range(-dist, dist + 1):
                    if abs(dr) + abs(dc_val) != dist:
                        continue
                    sr, sc = r + dr, c + dc_val
                    if 0 <= sr < self._rows and 0 <= sc < self._cols:
                        if self._grid[sr][sc] == CELL_AI:
                            is_ai_pos = any(
                                apr == sr and apc == sc
                                for apr, apc in self._ai_positions
                            )
                            if not is_ai_pos:
                                self._grid[sr][sc] = CELL_PLAYER
                                stolen += 1
                                if stolen >= count:
                                    return

    def _trigger_loss(self) -> None:
        self._lives -= 1
        if self._lives <= 0:
            self._sync_hud()
            self._rebuild()
            self.lose()
        else:
            self._reset_current_level()

    def _move_player(self, dr: int, dc: int) -> bool:
        if self._player_frozen > 0:
            self._player_frozen -= 1
            return True

        moves = 2 if self._player_speed > 0 else 1
        if self._player_speed > 0:
            self._player_speed -= 1

        for _ in range(moves):
            nr = self._player_r + dr
            nc = self._player_c + dc
            if not self._can_move_to(nr, nc):
                break
            old_cell = self._grid[nr][nc]
            if old_cell == CELL_BARRIER:
                self._trigger_loss()
                return False
            self._player_r, self._player_c = nr, nc
            if old_cell == CELL_BONUS:
                self._grid[nr][nc] = CELL_LOCKED
            elif old_cell != CELL_LOCKED:
                self._grid[nr][nc] = CELL_PLAYER
            if old_cell == CELL_SPEED:
                self._player_speed += SPEED_BOOST_TURNS
                self._steal_cells_near(nr, nc, SPEED_STEAL_CELLS)
            elif old_cell == CELL_TRAP:
                self._player_frozen += TRAP_FREEZE_TURNS

        return True

    def _move_ai(self) -> None:
        for i in range(len(self._ai_positions)):
            if self._ai_frozen[i] > 0:
                self._ai_frozen[i] -= 1
                continue

            ar, ac = self._ai_positions[i]
            own_cell = CELL_AI
            ai_type = self._ai_types[i] if i < len(self._ai_types) else AI_RANDOM
            seed_val = (
                self._seed * 31 + self._step_counter * 7 + i * 13 + ar * 3 + ac * 5
            )

            moves = 2 if self._ai_speed[i] > 0 else 1
            if self._ai_speed[i] > 0:
                self._ai_speed[i] -= 1

            for _ in range(moves):
                ar, ac = self._ai_positions[i]

                if ai_type == AI_RANDOM:
                    nr, nc = _ai_move_random(
                        self._grid, ar, ac, self._rows, self._cols, own_cell, seed_val
                    )
                elif ai_type == AI_GREEDY:
                    nr, nc = _ai_move_greedy(
                        self._grid, ar, ac, self._rows, self._cols, own_cell, seed_val
                    )
                elif ai_type == AI_SPEED_SEEKER:
                    nr, nc = _ai_move_speed_seeker(
                        self._grid,
                        ar,
                        ac,
                        self._rows,
                        self._cols,
                        own_cell,
                        self._speed_cells,
                        seed_val,
                    )
                elif ai_type == AI_TRAP_HERDER:
                    nr, nc = _ai_move_trap_herder(
                        self._grid,
                        ar,
                        ac,
                        self._player_r,
                        self._player_c,
                        self._rows,
                        self._cols,
                        own_cell,
                        self._trap_cells,
                        seed_val,
                    )
                elif ai_type == AI_CUTTER:
                    nr, nc = _ai_move_cutter(
                        self._grid,
                        ar,
                        ac,
                        self._player_r,
                        self._player_c,
                        self._rows,
                        self._cols,
                        own_cell,
                        seed_val,
                    )
                else:
                    nr, nc = _ai_move_random(
                        self._grid, ar, ac, self._rows, self._cols, own_cell, seed_val
                    )

                if nr == ar and nc == ac:
                    break

                old_cell = self._grid[nr][nc]
                if old_cell in (CELL_BARRIER, CELL_LOCKED):
                    break

                if old_cell != own_cell:
                    self._grid[nr][nc] = own_cell
                self._ai_positions[i] = (nr, nc)

                if old_cell == CELL_SPEED:
                    self._ai_speed[i] += AI_SPEED_BOOST_TURNS
                elif old_cell == CELL_TRAP:
                    self._ai_frozen[i] += 1
                    break

                seed_val += 17

    def _is_grid_full(self) -> bool:
        for r in range(self._rows):
            for c in range(self._cols):
                if self._grid[r][c] in (CELL_EMPTY, CELL_SPEED, CELL_TRAP, CELL_BONUS):
                    return False
        return True

    def _check_win_loss(self) -> bool:
        pct = self._compute_player_pct()
        ai_pct = self._compute_ai_pct()

        if pct >= self._win_pct:
            self._sync_hud()
            self.next_level()
            return True

        if ai_pct >= self._win_pct:
            self._trigger_loss()
            return True

        if self._is_grid_full() or self._moves_used >= self._max_moves:
            if pct > ai_pct and pct >= self._win_pct:
                self._sync_hud()
                self.next_level()
            else:
                self._trigger_loss()
            return True

        return False

    def _do_move(self, dr: int, dc: int) -> None:
        self._moves_used += 1
        self._step_counter += 1
        if self._move_player(dr, dc) is False:
            self.complete_action()
            return

        if self._ai_move_freq <= 1 or self._step_counter % self._ai_move_freq == 0:
            self._move_ai()

        self._sync_hud()
        self._rebuild()
        if self._check_win_loss():
            self.complete_action()
            return

        self.complete_action()

    def step(self) -> None:
        if self._level_just_loaded:
            self._level_just_loaded = False
            self.complete_action()
            return

        aid = self.action.id
        if aid == GameAction.ACTION7:
            had_undo = bool(self._undo_stack)
            current_moves = self._moves_used
            current_steps = self._step_counter
            if had_undo:
                prev = self._undo_stack.pop()
                self._restore_state(prev)
            self._moves_used = current_moves + 1
            self._step_counter = current_steps + 1
            if had_undo:
                if (
                    self._ai_move_freq <= 1
                    or self._step_counter % self._ai_move_freq == 0
                ):
                    self._move_ai()
            self._sync_hud()
            self._rebuild()
            self._check_win_loss()
            self.complete_action()
            return
        dr, dc = 0, 0
        if aid == GameAction.ACTION1:
            dr, dc = -1, 0
        elif aid == GameAction.ACTION2:
            dr, dc = 1, 0
        elif aid == GameAction.ACTION3:
            dr, dc = 0, -1
        elif aid == GameAction.ACTION4:
            dr, dc = 0, 1
        self._undo_stack.append(self._save_state())
        self._do_move(dr, dc)


CELL_CHARS = {
    CELL_EMPTY: ".",
    CELL_WALL: "#",
    CELL_PLAYER: "p",
    CELL_AI: "a",
    CELL_SPEED: "S",
    CELL_TRAP: "T",
    CELL_BARRIER: "B",
    CELL_BONUS: "+",
    CELL_LOCKED: "L",
}


class PuzzleEnvironment:
    _ACTION_MAP: Dict[str, GameAction] = {
        "reset": GameAction.RESET,
        "up": GameAction.ACTION1,
        "down": GameAction.ACTION2,
        "left": GameAction.ACTION3,
        "right": GameAction.ACTION4,
        "undo": GameAction.ACTION7,
    }

    _VALID_ACTIONS = list(_ACTION_MAP.keys())

    def __init__(self, seed: int = 0) -> None:
        self._engine = Nt01(seed=seed)
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
            raise ValueError(
                f"Invalid action '{action}'. Must be one of {list(self._ACTION_MAP.keys())}"
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

        total_levels = len(e._levels)
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
        lines = []
        lines.append(
            f"Level {e.level_index + 1}/{NUM_LEVELS} | "
            f"Lives: {e._lives}/{MAX_LIVES} | "
            f"Moves: {e._moves_used}/{e._max_moves}"
        )
        p_pct = e._compute_player_pct()
        a_pct = e._compute_ai_pct()
        lines.append(f"Player: {p_pct}% | AI: {a_pct}% | Target: {e._win_pct}%")
        status_parts = []
        if e._player_speed > 0:
            status_parts.append(f"Speed({e._player_speed})")
        if e._player_frozen > 0:
            status_parts.append(f"Frozen({e._player_frozen})")
        if status_parts:
            lines.append("Status: " + " ".join(status_parts))
        ai_set = set(e._ai_positions)
        for r in range(e._rows):
            row_chars = []
            for c in range(e._cols):
                if r == e._player_r and c == e._player_c:
                    row_chars.append("@")
                elif (r, c) in ai_set:
                    row_chars.append("X")
                else:
                    row_chars.append(CELL_CHARS.get(e._grid[r][c], "?"))
            lines.append("".join(row_chars))
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
        "render_fps": 5,
    }

    ACTION_LIST: List[str] = [
        "reset",
        "up",
        "down",
        "left",
        "right",
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

    env = ArcGameEnv(seed=0, render_mode="rgb_array")
    check_env(env.unwrapped, skip_render_check=True)

    obs, info = env.reset()
    mask = env.action_mask()
    valid_indices = np.where(mask == 1)[0]
    if len(valid_indices) > 0:
        obs, reward, term, trunc, info = env.step(valid_indices[0])

    env.close()
