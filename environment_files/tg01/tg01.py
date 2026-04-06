from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

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


BG = 0
GHOST = 1
BORDER = 5
YELLOW = 11
DARK = 4
GREEN = 14
RED = 8

PINK = 6
TEAL = 3
GRAY = 10
MAROON = 9

CELL_COLORS: List[int] = [0, 9, 12, 13]
WALL = 4

GRID_W = 24
GRID_H = 24
GRID_ROWS = 20
GRID_COLS = 20

LEFT_R0 = 2
LEFT_C0 = 2

BUDGET_DOT_START = 0
LIVES_DOT_START = 23

_MAX_LIVES = 3
_LIVES_AREA_START = LIVES_DOT_START - (_MAX_LIVES - 1) * 2
_BUDGET_BAR_W = _LIVES_AREA_START - BUDGET_DOT_START - 1

FLASH_TICKS = 3

BUDGET_MULTIPLIER = 5

_STATIC_COLOR_CYCLE: Dict[int, int] = {PINK: TEAL, TEAL: GRAY, GRAY: PINK}

_VALID_ACTIONS: List[str] = ["up", "down", "left", "right", "select", "undo"]

ZONE_CROSS = 0
ZONE_DIAG = 1
ZONE_HLINE = 2
ZONE_VLINE = 3
ZONE_KING = 4

ZONE_NAMES = {
    ZONE_CROSS: "CROSS (+)",
    ZONE_DIAG: "DIAG (x)",
    ZONE_HLINE: "HLINE (-)",
    ZONE_VLINE: "VLINE (|)",
    ZONE_KING: "KING (*)",
}


def _zone_neighbors(r: int, c: int, zone: int) -> List[Tuple[int, int]]:
    if zone == ZONE_CROSS:
        return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    elif zone == ZONE_DIAG:
        return [(r - 1, c - 1), (r - 1, c + 1), (r + 1, c - 1), (r + 1, c + 1)]
    elif zone == ZONE_HLINE:
        return [(r, c - 1), (r, c + 1)]
    elif zone == ZONE_VLINE:
        return [(r - 1, c), (r + 1, c)]
    elif zone == ZONE_KING:
        return [
            (r - 1, c - 1),
            (r - 1, c),
            (r - 1, c + 1),
            (r, c - 1),
            (r, c + 1),
            (r + 1, c - 1),
            (r + 1, c),
            (r + 1, c + 1),
        ]
    return []


def _pulse_cells(
    r: int,
    c: int,
    zone_map: List[List[int]],
    walls: Set[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    zone = zone_map[r][c]
    return [
        (nr, nc)
        for (nr, nc) in [(r, c)] + _zone_neighbors(r, c, zone)
        if 0 <= nr < GRID_ROWS and 0 <= nc < GRID_COLS and (nr, nc) not in walls
    ]


def _pulse_cells_cross(
    r: int, c: int, walls: Set[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    return [
        (nr, nc)
        for (nr, nc) in [
            (r, c),
            (r - 1, c),
            (r + 1, c),
            (r, c - 1),
            (r, c + 1),
        ]
        if 0 <= nr < GRID_ROWS and 0 <= nc < GRID_COLS and (nr, nc) not in walls
    ]


def _apply_pulse(
    grid: List[List[int]],
    r: int,
    c: int,
    zone_map: List[List[int]],
    walls: Set[Tuple[int, int]],
) -> None:
    for nr, nc in _pulse_cells(r, c, zone_map, walls):
        grid[nr][nc] = (grid[nr][nc] + 1) % 4


def _make_initial(
    seq: List[Tuple[int, int]],
    zone_map: List[List[int]],
    walls: Set[Tuple[int, int]],
) -> List[List[int]]:
    grid = [[0] * GRID_COLS for _ in range(GRID_ROWS)]
    for r, c in seq:
        for _ in range(3):
            _apply_pulse(grid, r, c, zone_map, walls)
    return grid


def _build_static_overlay(
    static_figures: List[Dict],
    walls: Set[Tuple[int, int]],
    zone_map: Optional[List[List[int]]] = None,
) -> List[List[int]]:
    overlay = [[0] * GRID_COLS for _ in range(GRID_ROWS)]
    for fig in static_figures:
        fr, fc = fig["center"]
        color = fig["color"]
        if zone_map is not None:
            cells = _pulse_cells(fr, fc, zone_map, walls)
        else:
            cells = _pulse_cells_cross(fr, fc, walls)
        for cell_r, cell_c in cells:
            overlay[cell_r][cell_c] = color
    return overlay


def _zone_all(zone_type: int) -> List[List[int]]:
    return [[zone_type] * GRID_COLS for _ in range(GRID_ROWS)]


def _zone_vsplit(col: int, left: int, right: int) -> List[List[int]]:
    return [
        [left if c < col else right for c in range(GRID_COLS)] for _ in range(GRID_ROWS)
    ]


def _zone_three(
    row_split: int, col_split: int, tl: int, tr: int, bottom: int
) -> List[List[int]]:
    return [
        [
            tl if r < row_split and c < col_split else tr if r < row_split else bottom
            for c in range(GRID_COLS)
        ]
        for r in range(GRID_ROWS)
    ]


def _zone_quad(
    row_split: int, col_split: int, tl: int, tr: int, bl: int, br: int
) -> List[List[int]]:
    return [
        [
            tl
            if r < row_split and c < col_split
            else tr
            if r < row_split
            else bl
            if c < col_split
            else br
            for c in range(GRID_COLS)
        ]
        for r in range(GRID_ROWS)
    ]


def _zone_king_center(
    king_r0: int,
    king_r1: int,
    king_c0: int,
    king_c1: int,
    tl: int,
    tr: int,
    bl: int,
    br: int,
) -> List[List[int]]:
    result: List[List[int]] = []
    for r in range(GRID_ROWS):
        row: List[int] = []
        for c in range(GRID_COLS):
            if king_r0 <= r <= king_r1 and king_c0 <= c <= king_c1:
                row.append(ZONE_KING)
            elif r <= (king_r0 + king_r1) // 2 and c <= (king_c0 + king_c1) // 2:
                row.append(tl)
            elif r <= (king_r0 + king_r1) // 2:
                row.append(tr)
            elif c <= (king_c0 + king_c1) // 2:
                row.append(bl)
            else:
                row.append(br)
        result.append(row)
    return result


def _bfs_dist(
    sr: int,
    sc: int,
    er: int,
    ec: int,
    walls: Set[Tuple[int, int]],
) -> int:
    if (sr, sc) == (er, ec):
        return 0
    visited = {(sr, sc)}
    q: deque = deque([(sr, sc, 0)])
    while q:
        r, c, d = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < GRID_ROWS
                and 0 <= nc < GRID_COLS
                and (nr, nc) not in walls
                and (nr, nc) not in visited
            ):
                if (nr, nc) == (er, ec):
                    return d + 1
                visited.add((nr, nc))
                q.append((nr, nc, d + 1))
    return 10**9


def _compute_min_moves(
    start: Tuple[int, int],
    waypoints: List[Tuple[int, int]],
    pulse_counts: List[int],
    walls: Set[Tuple[int, int]],
) -> int:
    n = len(waypoints)
    if n == 0:
        return 0
    all_pts = [start] + list(waypoints)
    m = len(all_pts)
    dist = [[0] * m for _ in range(m)]
    for i in range(m):
        for j in range(i + 1, m):
            d = _bfs_dist(
                all_pts[i][0],
                all_pts[i][1],
                all_pts[j][0],
                all_pts[j][1],
                walls,
            )
            dist[i][j] = d
            dist[j][i] = d
    INF = 10**9
    full = (1 << n) - 1
    dp = [[INF] * n for _ in range(1 << n)]
    for i in range(n):
        dp[1 << i][i] = dist[0][i + 1]
    for mask in range(1, 1 << n):
        for last in range(n):
            if not (mask & (1 << last)):
                continue
            if dp[mask][last] >= INF:
                continue
            for nxt in range(n):
                if mask & (1 << nxt):
                    continue
                new_mask = mask | (1 << nxt)
                nd = dp[mask][last] + dist[last + 1][nxt + 1]
                if nd < dp[new_mask][nxt]:
                    dp[new_mask][nxt] = nd
    min_path = min(dp[full][i] for i in range(n))
    return min_path + sum(pulse_counts)


def _level_waypoints(spec: dict) -> Tuple[List[Tuple[int, int]], List[int]]:
    waypoints: List[Tuple[int, int]] = []
    pulse_counts: List[int] = []
    seen: Set[Tuple[int, int]] = set()
    for r, c in spec["target_seq"]:
        if (r, c) not in seen:
            waypoints.append((r, c))
            pulse_counts.append(1)
            seen.add((r, c))
    for fig in spec.get("static_figures", []):
        r, c = fig["center"]
        if (r, c) not in seen:
            waypoints.append((r, c))
            pulse_counts.append(fig.get("hits_needed", 1))
            seen.add((r, c))
    for lock in spec.get("locks", []):
        kr, kc = lock["key"]
        if (kr, kc) not in seen:
            waypoints.append((kr, kc))
            pulse_counts.append(0)
            seen.add((kr, kc))
    return waypoints, pulse_counts


_NO_WALLS: Set[Tuple[int, int]] = set()

_VERT_BAR: Set[Tuple[int, int]] = {(r, 10) for r in range(0, 6)} | {
    (r, 10) for r in range(14, 20)
}

_L_PARTITION: Set[Tuple[int, int]] = (
    {(10, c) for c in range(0, 4)}
    | {(10, c) for c in range(6, 14)}
    | {(r, 10) for r in range(0, 3)}
    | {(r, 10) for r in range(5, 10)}
)

_QUAD_VAULT: Set[Tuple[int, int]] = (
    {(10, c) for c in range(0, 3)}
    | {(10, c) for c in range(7, 10)}
    | {(10, c) for c in range(17, 20)}
    | {(r, 10) for r in range(0, 3)}
    | {(r, 10) for r in range(7, 10)}
    | {(r, 10) for r in range(17, 20)}
    | {(0, 15), (0, 16), (0, 17), (1, 15), (1, 17), (2, 15), (2, 17)}
)


_L1_ZONE = _zone_all(ZONE_CROSS)
_L2_ZONE = _zone_vsplit(10, ZONE_CROSS, ZONE_DIAG)
_L3_ZONE = _zone_three(10, 10, ZONE_CROSS, ZONE_DIAG, ZONE_HLINE)
_L4_ZONE = _zone_quad(10, 10, ZONE_CROSS, ZONE_DIAG, ZONE_HLINE, ZONE_VLINE)


_L1_SEQ: List[Tuple[int, int]] = [(5, 5), (5, 7), (14, 14)]
_L2_SEQ: List[Tuple[int, int]] = [(3, 4), (4, 5), (8, 9), (12, 14), (13, 15)]
_L3_SEQ: List[Tuple[int, int]] = [(2, 2), (3, 3), (4, 14), (5, 15), (14, 4), (14, 12)]
_L4_SEQ: List[Tuple[int, int]] = [
    (3, 3),
    (4, 4),
    (3, 14),
    (14, 3),
    (14, 5),
    (14, 14),
    (16, 14),
]


LEVEL_SPECS: List[dict] = [
    {
        "title": "Compass Cross",
        "target_seq": _L1_SEQ,
        "walls": _NO_WALLS,
        "zone_map": _L1_ZONE,
        "ghost": True,
        "lives": 3,
        "static_figures": [],
        "locks": [],
        "ghost_budget_threshold": 0.0,
        "zone_clear": False,
        "budget_multiplier": 5,
    },
    {
        "title": "Diagonal Shift",
        "target_seq": _L2_SEQ,
        "walls": _VERT_BAR,
        "zone_map": _L2_ZONE,
        "ghost": True,
        "lives": 3,
        "static_figures": [
            {"center": (16, 3), "color": PINK, "hits_needed": 2},
        ],
        "locks": [],
        "ghost_budget_threshold": 0.0,
        "zone_clear": False,
        "budget_multiplier": 5,
    },
    {
        "title": "Triple Domain",
        "target_seq": _L3_SEQ,
        "walls": _L_PARTITION,
        "zone_map": _L3_ZONE,
        "ghost": True,
        "lives": 3,
        "static_figures": [
            {"center": (7, 7), "color": PINK, "hits_needed": 2},
            {"center": (17, 16), "color": GRAY, "hits_needed": 2},
        ],
        "locks": [],
        "ghost_budget_threshold": 0.0,
        "zone_clear": True,
        "budget_multiplier": 5,
    },
    {
        "title": "Warp Vault",
        "target_seq": _L4_SEQ,
        "walls": _QUAD_VAULT,
        "zone_map": _L4_ZONE,
        "ghost": True,
        "lives": 3,
        "static_figures": [
            {"center": (7, 4), "color": TEAL, "hits_needed": 1},
            {"center": (1, 16), "color": PINK, "hits_needed": 3},
        ],
        "locks": [
            {"key": (4, 3), "door": (2, 16)},
        ],
        "ghost_budget_threshold": 0.25,
        "zone_clear": True,
        "budget_multiplier": 5,
    },
]

for _spec in LEVEL_SPECS:
    _wps, _pcs = _level_waypoints(_spec)
    _min = _compute_min_moves((0, 0), _wps, _pcs, _spec["walls"])
    _bm = _spec.get("budget_multiplier", BUDGET_MULTIPLIER)
    _spec["budget"] = _bm * _min
    _spec["min_moves"] = _min

_VERIFY_ERRORS: List[str] = []

for _spec in LEVEL_SPECS:
    _sol = _spec["target_seq"]
    _walls = _spec["walls"]
    _bud = _spec["budget"]
    _zmap = _spec["zone_map"]
    _statics = _spec.get("static_figures", [])
    _locks = _spec.get("locks", [])
    _tag = _spec["title"]
    _bm = _spec.get("budget_multiplier", BUDGET_MULTIPLIER)

    _wps, _pcs = _level_waypoints(_spec)
    _min = _compute_min_moves((0, 0), _wps, _pcs, _walls)
    _expected = _bm * _min
    if _bud != _expected:
        _VERIFY_ERRORS.append(
            "%s: budget=%d != %dx%d=%d" % (_tag, _bud, _bm, _min, _expected)
        )

    for _r, _c in _sol:
        if not (0 <= _r < GRID_ROWS and 0 <= _c < GRID_COLS):
            _VERIFY_ERRORS.append("%s: (%d,%d) out of bounds" % (_tag, _r, _c))
        elif (_r, _c) in _walls:
            _VERIFY_ERRORS.append("%s: seq (%d,%d) on wall" % (_tag, _r, _c))

    for _fig in _statics:
        _r, _c = _fig["center"]
        if not (0 <= _r < GRID_ROWS and 0 <= _c < GRID_COLS):
            _VERIFY_ERRORS.append("%s: static (%d,%d) out of bounds" % (_tag, _r, _c))
        elif (_r, _c) in _walls:
            _VERIFY_ERRORS.append("%s: static (%d,%d) on wall" % (_tag, _r, _c))

    for _lock in _locks:
        _kr, _kc = _lock["key"]
        _dr, _dc = _lock["door"]
        if not (0 <= _kr < GRID_ROWS and 0 <= _kc < GRID_COLS):
            _VERIFY_ERRORS.append("%s: key (%d,%d) out of bounds" % (_tag, _kr, _kc))
        elif (_kr, _kc) in _walls:
            _VERIFY_ERRORS.append("%s: key (%d,%d) on wall" % (_tag, _kr, _kc))
        if not (0 <= _dr < GRID_ROWS and 0 <= _dc < GRID_COLS):
            _VERIFY_ERRORS.append("%s: door (%d,%d) out of bounds" % (_tag, _dr, _dc))
        elif (_dr, _dc) in _walls:
            _VERIFY_ERRORS.append("%s: door (%d,%d) on wall" % (_tag, _dr, _dc))

    _door_cells = {lock["door"] for lock in _locks}
    _eff_walls = _walls | _door_cells
    _init = _make_initial(_sol, _zmap, _eff_walls)
    _test = [row[:] for row in _init]
    for _r, _c in _sol:
        _apply_pulse(_test, _r, _c, _zmap, _eff_walls)
    if not all(_test[r][c] == 0 for r in range(GRID_ROWS) for c in range(GRID_COLS)):
        _VERIFY_ERRORS.append(
            "%s: target_seq does NOT reduce initial grid to all-zeros!" % _tag
        )

    for _wp_r, _wp_c in _wps:
        d = _bfs_dist(0, 0, _wp_r, _wp_c, _walls)
        if d >= 10**9:
            _VERIFY_ERRORS.append(
                "%s: waypoint (%d,%d) unreachable from (0,0)" % (_tag, _wp_r, _wp_c)
            )

    _spec["initial"] = _init

if _VERIFY_ERRORS:
    raise AssertionError("Level verification failed: " + "; ".join(_VERIFY_ERRORS))


N_LEVELS = len(LEVEL_SPECS)
levels = [Level(sprites=[], grid_size=(GRID_W, GRID_H)) for _ in range(N_LEVELS)]


class Tg01(ARCBaseGame):
    _BLINK_PERIOD = 6

    def __init__(self, seed: int = 0) -> None:
        camera = Camera(
            x=0,
            y=0,
            background=BG,
            letter_box=BORDER,
            width=GRID_W,
            height=GRID_H,
        )
        super().__init__(
            game_id="tg01",
            levels=levels,
            camera=camera,
            available_actions=[0, 1, 2, 3, 4, 5, 7],
        )
        self._spec: dict = {}
        self._grid: List[List[int]] = [[0] * GRID_COLS for _ in range(GRID_ROWS)]
        self._budget: int = 0
        self._budget_max: int = 0
        self._zone_map: List[List[int]] = [
            [ZONE_CROSS] * GRID_COLS for _ in range(GRID_ROWS)
        ]
        self._static_overlay: List[List[int]] = [
            [0] * GRID_COLS for _ in range(GRID_ROWS)
        ]
        self._static_figures: List[Optional[Dict]] = []
        self._static_hits: List[int] = []
        self._locks: List[Dict] = []
        self._keys_collected: List[bool] = []
        self._doors_open: List[bool] = []
        self._lives: int = 0
        self._lives_max: int = 0
        self._flash_win: bool = False
        self._flash_hit: int = 0
        self._flash_dead: int = 0
        self._cur_r: int = 0
        self._cur_c: int = 0
        self._blink: int = 0
        self._cur_vis: bool = True
        self._just_transitioned: bool = False
        self._ready: bool = False
        self._life_lost_this_step: bool = False
        self._game_over: bool = False
        self._pending_retry: bool = False
        self._history: List[Dict] = []
        self._rng: random.Random = random.Random(seed)

    def _clamp(self, r: int, c: int) -> Tuple[int, int]:
        return (
            max(0, min(GRID_ROWS - 1, r)),
            max(0, min(GRID_COLS - 1, c)),
        )

    def _random_start(self) -> Tuple[int, int]:
        walls = self._effective_walls()
        free = [
            (r, c)
            for r in range(GRID_ROWS)
            for c in range(GRID_COLS)
            if (r, c) not in walls
        ]
        return free[self._rng.randint(0, len(free) - 1)]

    def _effective_walls(self) -> Set[Tuple[int, int]]:
        walls = set(self._spec.get("walls", set()))
        for i, lock in enumerate(self._locks):
            if not self._doors_open[i]:
                walls.add(lock["door"])
        return walls

    def _is_wall(self, r: int, c: int) -> bool:
        if (r, c) in self._spec.get("walls", set()):
            return True
        for i, lock in enumerate(self._locks):
            if not self._doors_open[i] and (r, c) == lock["door"]:
                return True
        return False

    def _is_solved(self) -> bool:
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                if self._grid[r][c] != 0:
                    return False
                if self._static_overlay[r][c] != 0:
                    return False
        return True

    def _restore_grid(self) -> None:
        self._grid = [row[:] for row in self._spec["initial"]]

    def _reset_lock_state(self) -> None:
        self._locks = list(self._spec.get("locks", []))
        self._keys_collected = [False] * len(self._locks)
        self._doors_open = [False] * len(self._locks)

    def _rebuild_static(self) -> None:
        raw = self._spec.get("static_figures", [])
        self._static_figures = [dict(f) for f in raw]
        self._static_hits = [0] * len(self._static_figures)
        active = [f for f in self._static_figures if f is not None]
        zone_map = self._zone_map if self._spec.get("zone_clear", False) else None
        self._static_overlay = _build_static_overlay(
            active, self._effective_walls(), zone_map
        )

    def _reset_level_state(self) -> None:
        self._restore_grid()
        self._reset_lock_state()
        self._rebuild_static()
        self._budget = self._budget_max
        self._lives = self._lives_max
        self._flash_win = False
        self._flash_hit = 0
        self._flash_dead = 0
        self._cur_r, self._cur_c = self._random_start()
        self._blink = 0
        self._cur_vis = True
        self._history = []

    def _soft_reset(self) -> None:
        self._restore_grid()
        self._reset_lock_state()
        self._rebuild_static()
        self._budget = self._budget_max
        self._cur_r, self._cur_c = self._random_start()
        self._blink = 0
        self._cur_vis = True
        self._history = []

    def _lose_life(self, reason: str) -> bool:
        self._lives -= 1
        self._life_lost_this_step = True
        if self._lives <= 0:
            self._flash_dead = FLASH_TICKS
            self._render()
            self._game_over = True
            self.lose()
            self.complete_action()
            return True
        self._flash_hit = FLASH_TICKS
        return False

    def on_set_level(self, level: Level) -> None:
        if not hasattr(self, "_ready"):
            return
        self.current_level.remove_all_sprites()
        idx = self._current_level_index
        spec = LEVEL_SPECS[idx]
        self._spec = spec
        self._budget_max = spec["budget"]
        self._lives_max = spec["lives"]
        self._zone_map = spec["zone_map"]
        self._reset_level_state()
        self._just_transitioned = True
        self._render()
        self._ready = True

    def _try_win(self) -> bool:
        if not self._is_solved():
            return False
        self._flash_win = True
        self._render()
        if self._current_level_index < len(self._levels) - 1:
            self.next_level()
            self.complete_action()
        else:
            self.win()
            self.complete_action()
        return True

    def _save_state(self) -> None:
        self._history.append(
            {
                "grid": [row[:] for row in self._grid],
                "static_overlay": [row[:] for row in self._static_overlay],
                "static_figures": [
                    dict(f) if f is not None else None for f in self._static_figures
                ],
                "static_hits": list(self._static_hits),
                "keys_collected": list(self._keys_collected),
                "doors_open": list(self._doors_open),
                "budget": self._budget,
                "cur_r": self._cur_r,
                "cur_c": self._cur_c,
            }
        )

    def _restore_from_undo(self) -> None:
        if not self._history:
            return
        state = self._history.pop()
        self._grid = state["grid"]
        self._static_overlay = state["static_overlay"]
        self._static_figures = state["static_figures"]
        self._static_hits = state["static_hits"]
        self._keys_collected = state["keys_collected"]
        self._doors_open = state["doors_open"]
        self._cur_r = state["cur_r"]
        self._cur_c = state["cur_c"]

    def step(self) -> None:
        if not self._ready:
            self.complete_action()
            return

        if self._just_transitioned:
            self._just_transitioned = False
            self._render()
            self.complete_action()
            return

        action = self.action

        if action.id == GameAction.ACTION7:
            if self._budget <= 0:
                self._render()
                self.complete_action()
                return
            self._budget -= 1
            self._restore_from_undo()
            if self._budget == 0:
                if self._lose_life("budget_exhausted"):
                    return
                self._soft_reset()
            self._render()
            self.complete_action()
            return

        self._blink += 1
        self._cur_vis = (self._blink % self._BLINK_PERIOD) < (self._BLINK_PERIOD // 2)

        if self._flash_hit > 0:
            self._flash_hit -= 1
        if self._flash_dead > 0:
            self._flash_dead -= 1

        delta = {
            GameAction.ACTION1: (-1, 0),
            GameAction.ACTION2: (1, 0),
            GameAction.ACTION3: (0, -1),
            GameAction.ACTION4: (0, 1),
        }.get(action.id)

        is_action = delta is not None or action.id == GameAction.ACTION5
        self._life_lost_this_step = False

        if is_action:
            if self._budget <= 0:
                self._render()
                self.complete_action()
                return
            self._save_state()
            self._budget -= 1
            self._pending_retry = False

        if delta is not None:
            dr, dc = delta
            nr, nc = self._clamp(self._cur_r + dr, self._cur_c + dc)

            if self._is_wall(nr, nc):
                if self._lose_life("wall_hit"):
                    return
                self._soft_reset()
                self._render()
                self.complete_action()
                return
            else:
                self._cur_r, self._cur_c = nr, nc

                for i, lock in enumerate(self._locks):
                    if self._keys_collected[i]:
                        continue
                    kr, kc = lock["key"]
                    if (self._cur_r, self._cur_c) != (kr, kc):
                        continue
                    if self._grid[kr][kc] == 0 and self._static_overlay[kr][kc] == 0:
                        self._keys_collected[i] = True
                        self._doors_open[i] = True

        elif action.id == GameAction.ACTION5:
            r, c = self._cur_r, self._cur_c

            if not self._is_wall(r, c):
                eff_walls = self._effective_walls()
                _apply_pulse(self._grid, r, c, self._zone_map, eff_walls)

                use_zone = self._spec.get("zone_clear", False)

                for fig_idx in range(len(self._static_figures)):
                    fig = self._static_figures[fig_idx]
                    if fig is None:
                        continue
                    if (r, c) != fig["center"]:
                        continue

                    self._static_hits[fig_idx] += 1
                    needed = fig.get("hits_needed", 1)

                    if use_zone:
                        clear_cells = _pulse_cells(r, c, self._zone_map, eff_walls)
                    else:
                        clear_cells = _pulse_cells_cross(r, c, eff_walls)

                    if self._static_hits[fig_idx] >= needed:
                        color = fig["color"]
                        for cell_r, cell_c in clear_cells:
                            if self._static_overlay[cell_r][cell_c] == color:
                                self._static_overlay[cell_r][cell_c] = 0
                        self._static_figures[fig_idx] = None
                    else:
                        old_color = fig["color"]
                        new_color = _STATIC_COLOR_CYCLE.get(old_color, old_color)
                        fig["color"] = new_color
                        for cell_r, cell_c in clear_cells:
                            if self._static_overlay[cell_r][cell_c] == old_color:
                                self._static_overlay[cell_r][cell_c] = new_color

                if self._is_solved():
                    if self._try_win():
                        return

        if is_action and self._budget == 0:
            if not self._life_lost_this_step:
                if self._lose_life("budget_exhausted"):
                    return
            self._soft_reset()
            self._render()
            self.complete_action()
            return

        self._render()
        self.complete_action()

    def _render(self) -> None:
        self.current_level.remove_all_sprites()

        frame = np.full((GRID_H, GRID_W), BG, dtype=np.uint8)

        if self._flash_win:
            border_col = GREEN
            self._flash_win = False
        else:
            border_col = BORDER

        frame[0, :] = border_col
        frame[1, :] = border_col
        frame[22, :] = border_col
        frame[23, :] = border_col

        for row in range(LEFT_R0, LEFT_R0 + GRID_ROWS):
            frame[row, 0] = BORDER
            frame[row, 1] = BORDER
            frame[row, 22] = BORDER
            frame[row, 23] = BORDER

        if self._budget_max > 0:
            filled_px = round(self._budget * _BUDGET_BAR_W / self._budget_max)
        else:
            filled_px = 0
        for i in range(_BUDGET_BAR_W):
            frame[0, BUDGET_DOT_START + i] = YELLOW if i < filled_px else DARK

        spent_lives = self._lives_max - self._lives
        for i in range(self._lives_max):
            dot_col = LIVES_DOT_START - i * 2
            if 0 <= dot_col < GRID_W:
                frame[0, dot_col] = DARK if i < spent_lives else RED
            gap_col = dot_col - 1
            if i < self._lives_max - 1 and 0 <= gap_col < GRID_W:
                frame[0, gap_col] = BG

        for wr, wc in self._spec.get("walls", set()):
            frame[LEFT_R0 + wr, LEFT_C0 + wc] = WALL

        for i, lock in enumerate(self._locks):
            if not self._doors_open[i]:
                dr, dc = lock["door"]
                frame[LEFT_R0 + dr, LEFT_C0 + dc] = RED

        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                val = self._grid[r][c]
                if val != 0:
                    frame[LEFT_R0 + r, LEFT_C0 + c] = CELL_COLORS[val]

        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                if self._static_overlay[r][c] != 0:
                    frame[LEFT_R0 + r, LEFT_C0 + c] = self._static_overlay[r][c]

        for i, lock in enumerate(self._locks):
            if self._keys_collected[i]:
                continue
            kr, kc = lock["key"]
            if self._grid[kr][kc] == 0 and self._static_overlay[kr][kc] == 0:
                frame[LEFT_R0 + kr, LEFT_C0 + kc] = RED

        ghost_threshold = self._spec.get("ghost_budget_threshold", 0.0)
        budget_frac = self._budget / max(1, self._budget_max)
        ghost_ok = budget_frac > ghost_threshold

        if self._cur_vis and self._budget > 0 and ghost_ok:
            cr, cc = self._cur_r, self._cur_c
            if not self._is_wall(cr, cc):
                eff_walls = self._effective_walls()
                for nr, nc in _pulse_cells(cr, cc, self._zone_map, eff_walls):
                    if (nr, nc) != (cr, cc):
                        if (
                            self._grid[nr][nc] == 0
                            and self._static_overlay[nr][nc] == 0
                        ):
                            frame[LEFT_R0 + nr, LEFT_C0 + nc] = GHOST

        bg_sp = Sprite(
            pixels=frame.tolist(),
            name="background",
            visible=True,
            collidable=False,
            layer=0,
        )
        bg_sp.set_position(0, 0)
        self.current_level.add_sprite(bg_sp)

        if self._cur_vis:
            cur_sp = Sprite(
                pixels=[[GREEN]],
                name="cursor",
                visible=True,
                collidable=False,
                layer=3,
            )
            cur_sp.set_position(LEFT_C0 + self._cur_c, LEFT_R0 + self._cur_r)
            self.current_level.add_sprite(cur_sp)

    def _build_text_observation(self) -> str:
        spec = self._spec
        walls = spec.get("walls", set())
        bm = spec.get("budget_multiplier", BUDGET_MULTIPLIER)
        ghost_threshold = spec.get("ghost_budget_threshold", 0.0)
        budget_frac = self._budget / max(1, self._budget_max)
        ghost_active = budget_frac > ghost_threshold
        keys_held = sum(1 for k in self._keys_collected if k)
        doors_open_count = sum(1 for d in self._doors_open if d)
        cr, cc = self._cur_r, self._cur_c
        cur_zone_name = ZONE_NAMES.get(self._zone_map[cr][cc], "?")
        revealed: Set[Tuple[int, int]] = set()
        for i, lock in enumerate(self._locks):
            if not self._keys_collected[i]:
                kr, kc = lock["key"]
                if self._grid[kr][kc] == 0 and self._static_overlay[kr][kc] == 0:
                    revealed.add((kr, kc))
        locked: Set[Tuple[int, int]] = set()
        for i, lock in enumerate(self._locks):
            if not self._doors_open[i]:
                locked.add(lock["door"])
        header = (
            "L%d/%d %s | Bud:%d/%d(%dx) Lv:%d/%d | Cur:(%d,%d) Zone:%s | Ghost:%s(%d%%) | Keys:%d/%d Doors:%d/%d"
            % (
                self._current_level_index + 1,
                N_LEVELS,
                spec.get("title", ""),
                self._budget,
                self._budget_max,
                bm,
                self._lives,
                self._lives_max,
                cr,
                cc,
                cur_zone_name,
                "ON" if ghost_active else "OFF",
                int(ghost_threshold * 100),
                keys_held,
                len(self._locks),
                doors_open_count,
                len(self._locks),
            )
        )
        legend = ".=clear 1=triple 2=double 3=single #=wall D=door K=key @=cursor p=pink t=teal g=gray"
        col_hdr = "   " + "".join(str(c % 10) for c in range(GRID_COLS))
        rows = [header, legend, col_hdr]
        for r in range(GRID_ROWS):
            chars = []
            for c in range(GRID_COLS):
                if (r, c) == (cr, cc):
                    ch = "@"
                elif (r, c) in walls:
                    ch = "#"
                elif (r, c) in locked:
                    ch = "D"
                elif self._static_overlay[r][c] == PINK:
                    ch = "p"
                elif self._static_overlay[r][c] == TEAL:
                    ch = "t"
                elif self._static_overlay[r][c] == GRAY:
                    ch = "g"
                elif (r, c) in revealed:
                    ch = "K"
                elif self._grid[r][c] != 0:
                    ch = str(self._grid[r][c])
                else:
                    ch = "."
                chars.append(ch)
            rows.append("%2d " % r + "".join(chars))
        return "\n".join(rows)

    @property
    def extra_state(self) -> dict:
        idx = self._current_level_index
        spec = self._spec
        walls = spec.get("walls", set())

        non_zero_grid = sum(
            1
            for r in range(GRID_ROWS)
            for c in range(GRID_COLS)
            if self._grid[r][c] != 0
        )
        non_zero_static = sum(
            1
            for r in range(GRID_ROWS)
            for c in range(GRID_COLS)
            if self._static_overlay[r][c] != 0
        )
        non_zero = non_zero_grid + non_zero_static
        solved = self._is_solved()
        total_cells = GRID_ROWS * GRID_COLS - len(walls)
        pct = 100 - int(100 * non_zero / max(1, total_cells))

        keys_held = sum(1 for k in self._keys_collected if k)
        doors_open = sum(1 for d in self._doors_open if d)

        cr, cc = self._cur_r, self._cur_c
        cur_zone = self._zone_map[cr][cc]
        cur_zone_name = ZONE_NAMES.get(cur_zone, "UNKNOWN")

        zone_counts: Dict[int, int] = {}
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                z = self._zone_map[r][c]
                zone_counts[z] = zone_counts.get(z, 0) + 1
        zone_desc = ", ".join(
            f"{ZONE_NAMES.get(z, '?')}={n}" for z, n in sorted(zone_counts.items())
        )

        ghost_threshold = spec.get("ghost_budget_threshold", 0.0)
        budget_frac = self._budget / max(1, self._budget_max)
        ghost_active = budget_frac > ghost_threshold

        rules: List[str] = ["R0: budget-exhaust=life-lost"]
        if any(f.get("hits_needed", 1) > 1 for f in spec.get("static_figures", [])):
            rules.append("R1: multi-hit-statics+colour-cycle")
        if spec.get("zone_clear", False):
            rules.append("R2: zone-specific-static-clear")
        if ghost_threshold >= 0.25:
            rules.append("R3-ghost-off-below-%dpct" % int(ghost_threshold * 100))

        static_progress = []
        for fig_idx, fig in enumerate(self._static_figures):
            if fig is None:
                static_progress.append("CLEARED")
            else:
                needed = fig.get("hits_needed", 1)
                current = (
                    self._static_hits[fig_idx]
                    if fig_idx < len(self._static_hits)
                    else 0
                )
                static_progress.append(
                    "(%d,%d) color=%d hits=%d/%d"
                    % (
                        fig["center"][0],
                        fig["center"][1],
                        fig["color"],
                        current,
                        needed,
                    )
                )

        return {
            "snake_length": self._budget_max - self._budget,
            "target_length": self._budget_max,
            "speed_cells_s": 0,
            "speed_interval_ms": 0,
            "remaining_cells": self._budget,
            "cursor_row": self._cur_r,
            "cursor_col": self._cur_c,
            "cursor_zone": cur_zone_name,
            "ghost_visible": spec.get("ghost", False),
            "ghost_active": ghost_active,
            "ghost_budget_threshold_pct": int(ghost_threshold * 100),
            "lives": self._lives,
            "lives_max": self._lives_max,
            "lives_spent": self._lives_max - self._lives,
            "budget_remaining": self._budget,
            "budget_max": self._budget_max,
            "budget_used": self._budget_max - self._budget,
            "budget_multiplier": spec.get("budget_multiplier", BUDGET_MULTIPLIER),
            "non_zero_cells": non_zero,
            "non_zero_grid": non_zero_grid,
            "non_zero_static": non_zero_static,
            "clear_pct": pct,
            "is_solved": solved,
            "keys_collected": keys_held,
            "keys_total": len(self._locks),
            "doors_open": doors_open,
            "zone_distribution": zone_desc,
            "zone_clear_active": spec.get("zone_clear", False),
            "active_rules": rules,
            "static_progress": static_progress,
            "circuit_title": "Transform Grid -- Level %d/%d" % (idx + 1, N_LEVELS),
            "level_title": spec.get("title", ""),
            "text_observation": self._build_text_observation(),
            "valid_actions": _VALID_ACTIONS,
            "turn": self._action_count,
            "level_features": [
                "Level %d/%d: %s" % (idx + 1, N_LEVELS, spec.get("title", "")),
                "Budget=%d/%d(%dx)  Lives=%d/%d  Walls=%d  Statics=%d  Locks=%d"
                % (
                    self._budget,
                    self._budget_max,
                    spec.get("budget_multiplier", BUDGET_MULTIPLIER),
                    self._lives,
                    self._lives_max,
                    len(walls),
                    len(spec.get("static_figures", [])),
                    len(spec.get("locks", [])),
                ),
                "Keys: %d/%d  Doors: %d/%d"
                % (keys_held, len(self._locks), doors_open, len(self._locks)),
                "Cursor (%d, %d) -- Zone: %s"
                % (self._cur_r, self._cur_c, cur_zone_name),
                "Ghost: %s (threshold=%d%%)"
                % ("ON" if ghost_active else "OFF", int(ghost_threshold * 100)),
                "Zones: %s" % zone_desc,
                "Active rules: %s" % " | ".join(rules),
                "Remaining coloured cells: %d  (%d%% clear)" % (non_zero, pct),
                "\u2713 SOLVED" if solved else "\u2026",
            ],
        }

    def get_actions(self) -> List[str]:
        return list(_VALID_ACTIONS)

    def reset(self) -> dict:
        if self._pending_retry:
            self._pending_retry = False
            target_idx = 0
        else:
            self._pending_retry = True
            target_idx = self._current_level_index
        self._game_over = False
        self._current_level_index = target_idx
        spec = LEVEL_SPECS[target_idx]
        self._spec = spec
        self._budget_max = spec["budget"]
        self._lives_max = spec["lives"]
        self._zone_map = spec["zone_map"]
        self._reset_level_state()
        self._ready = True
        return self.extra_state


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
        self._engine = Tg01(seed=seed)
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
                info={"action": action, "error": "invalid_action"},
            )
        self._last_action_was_reset = False
        self._total_turns += 1
        game_action = self._ACTION_MAP[action]
        info: Dict[str, Any] = {"action": action}
        level_before = e.level_index
        frame = e.perform_action(ActionInput(id=game_action), raw=True)
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
        return rgb

    def close(self) -> None:
        self._engine = None

    def _build_text_observation(self) -> str:
        e = self._engine
        extra = e.extra_state
        return extra.get("text_observation", "")

    def _build_game_state(self, done: bool = False) -> GameState:
        valid_actions = self.get_actions() if not done else None
        e = self._engine
        return GameState(
            text_observation=self._build_text_observation(),
            image_observation=None,
            valid_actions=valid_actions,
            turn=self._total_turns,
            metadata={
                "total_levels": len(e._levels),
                "level_index": e.level_index,
                "levels_completed": getattr(e, "_score", 0),
                "game_over": getattr(getattr(e, "_state", None), "name", "")
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
