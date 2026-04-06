from __future__ import annotations

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


PAL_BG = 0
PAL_BORDER = 3
PAL_FLOOR = 1
PAL_GRID_LINE = 1
PAL_WALL = 5
PAL_PLAYER = 10
PAL_TOWER_OFF = 6
PAL_TOWER_ON = 14
PAL_TOWER_CHAIN = 9
PAL_DECOY = 4
PAL_EXIT_LOCKED = 6
PAL_EXIT_OPEN = 13
PAL_GATE_NS = 11
PAL_GATE_EW = 7
PAL_GATE_FLIPPED = 3
PAL_JAMMER = 2
PAL_MOVING_JAMMER = 12
PAL_GUARD = 8
PAL_PLATE = 5
PAL_KEY = 13
PAL_DOOR_CLOSED = 5
PAL_DOOR_OPEN = 1
PAL_BUDGET_LEFT = 10
PAL_BUDGET_SPENT = 5
PAL_LIVES_LEFT = 14
PAL_LIVES_SPENT = 4
PAL_WIN = 10
PAL_HIT = 4

GRID_ROWS = 25
GRID_COLS = 25
CELL_SIZE = 2
BORDER_OFFSET = 1
FLASH_TICKS = 3

DISP_W = GRID_COLS * CELL_SIZE + 2 * BORDER_OFFSET
DISP_H = GRID_ROWS * CELL_SIZE + 2 * BORDER_OFFSET

NORTH = (-1, 0)
SOUTH = (1, 0)
WEST = (0, -1)
EAST = (0, 1)

GATE_FROM_NORTH = "GN"
GATE_FROM_SOUTH = "GS"
GATE_FROM_WEST = "GW"
GATE_FROM_EAST = "GE"

_GATE_ALLOWED: Dict[str, Tuple[int, int]] = {
    GATE_FROM_NORTH: SOUTH,
    GATE_FROM_SOUTH: NORTH,
    GATE_FROM_WEST: EAST,
    GATE_FROM_EAST: WEST,
}


@dataclass
class TowerSpec:
    cell: Tuple[int, int]
    order: int
    is_chain: bool


@dataclass
class GuardSpec:
    path: List[Tuple[int, int]]
    start_index: int = 0


@dataclass
class LevelSpec:
    name: str
    subtitle: str
    rows: int
    cols: int
    budget: int
    lives: int
    walls: Set[Tuple[int, int]]
    player_start: Tuple[int, int]
    exit_cell: Tuple[int, int]
    towers: List[TowerSpec]
    par_moves: int
    hint: str
    decoys: Set[Tuple[int, int]] = field(default_factory=set)
    gates: Dict[Tuple[int, int], str] = field(default_factory=dict)
    jammers: Set[Tuple[int, int]] = field(default_factory=set)
    guard: Optional[GuardSpec] = None
    moving_jammers: List[GuardSpec] = field(default_factory=list)
    plates: Dict[Tuple[int, int], Tuple[int, int]] = field(default_factory=dict)
    keys: Set[Tuple[int, int]] = field(default_factory=set)
    doors: Set[Tuple[int, int]] = field(default_factory=set)
    plate_groups: List[Tuple[frozenset, frozenset]] = field(default_factory=list)
    has_decoys: bool = False
    has_chain: bool = False
    has_gates: bool = False
    has_jammers: bool = False
    has_guard: bool = False
    has_plates: bool = False
    has_keys: bool = False
    gate_flips: Dict[Tuple[int, int], Tuple[Tuple[int, int], str]] = field(
        default_factory=dict
    )
    gate_flip_reverse: Dict[Tuple[int, int], str] = field(default_factory=dict)


@dataclass
class RuntimeState:
    player: Tuple[int, int]
    next_tower: int
    activated: List[bool]
    guard_index: int
    guard_move_counter: int
    doors_open: Set[Tuple[int, int]] = field(default_factory=set)
    keys_collected: Set[Tuple[int, int]] = field(default_factory=set)
    plates_active: Set[Tuple[int, int]] = field(default_factory=set)
    mj_indices: List[int] = field(default_factory=list)
    mj_counters: List[int] = field(default_factory=list)
    flipped_gates: Set[Tuple[int, int]] = field(default_factory=set)


def _walls_border(rows: int, cols: int) -> Set[Tuple[int, int]]:
    w: Set[Tuple[int, int]] = set()
    for c in range(cols):
        w.add((0, c))
        w.add((rows - 1, c))
    for r in range(rows):
        w.add((r, 0))
        w.add((r, cols - 1))
    return w


def _make_runtime(spec: LevelSpec) -> RuntimeState:
    gi = spec.guard.start_index if spec.guard else 0
    return RuntimeState(
        player=spec.player_start,
        next_tower=0,
        activated=[False] * len(spec.towers),
        guard_index=gi,
        guard_move_counter=0,
        doors_open=set(),
        keys_collected=set(),
        plates_active=set(),
        mj_indices=[mj.start_index for mj in spec.moving_jammers],
        mj_counters=[0] * len(spec.moving_jammers),
        flipped_gates=set(),
    )


def _all_activated(rt: RuntimeState) -> bool:
    return all(rt.activated)


def _tower_at(spec: LevelSpec, cell: Tuple[int, int]) -> Optional[TowerSpec]:
    for t in spec.towers:
        if t.cell == cell:
            return t
    return None


def _guard_pos(spec: LevelSpec, rt: RuntimeState) -> Optional[Tuple[int, int]]:
    if spec.guard is None:
        return None
    return spec.guard.path[rt.guard_index]


def _advance_guard(spec: LevelSpec, rt: RuntimeState) -> RuntimeState:
    if spec.guard is None:
        return rt
    rt.guard_move_counter += 1
    if rt.guard_move_counter >= 2:
        rt.guard_move_counter = 0
        rt.guard_index = (rt.guard_index + 1) % len(spec.guard.path)
    return rt


def _moving_jammer_positions(
    spec: LevelSpec, rt: RuntimeState
) -> List[Tuple[int, int]]:
    return [mj.path[rt.mj_indices[i]] for i, mj in enumerate(spec.moving_jammers)]


def _moving_jammer_danger_cells(
    spec: LevelSpec, rt: RuntimeState
) -> Set[Tuple[int, int]]:
    danger: Set[Tuple[int, int]] = set()
    for pos in _moving_jammer_positions(spec, rt):
        r, c = pos
        danger.add((r, c))
        danger.add((r - 1, c))
        danger.add((r + 1, c))
        danger.add((r, c - 1))
        danger.add((r, c + 1))
    return danger


def _advance_moving_jammers(spec: LevelSpec, rt: RuntimeState) -> RuntimeState:
    for i, mj in enumerate(spec.moving_jammers):
        rt.mj_counters[i] += 1
        if rt.mj_counters[i] >= 2:
            rt.mj_counters[i] = 0
            rt.mj_indices[i] = (rt.mj_indices[i] + 1) % len(mj.path)
    return rt


def _effective_gate_type(
    spec: LevelSpec,
    cell: Tuple[int, int],
    flipped_gates: Optional[Set[Tuple[int, int]]] = None,
) -> Optional[str]:
    gtype = spec.gates.get(cell)
    if gtype is None:
        return None
    if flipped_gates and cell in flipped_gates:
        flipped = spec.gate_flip_reverse.get(cell)
        if flipped is not None:
            return flipped
    return gtype


def _gate_passable(
    spec: LevelSpec,
    cell: Tuple[int, int],
    delta: Tuple[int, int],
    flipped_gates: Optional[Set[Tuple[int, int]]] = None,
) -> bool:
    gtype = _effective_gate_type(spec, cell, flipped_gates)
    if gtype is None:
        return True
    return delta == _GATE_ALLOWED[gtype]


def _with_flip_reverse(spec: LevelSpec) -> LevelSpec:
    spec.gate_flip_reverse = {g: ft for _p, (g, ft) in spec.gate_flips.items()}
    return spec


def _build_level_1() -> LevelSpec:
    rows, cols = GRID_ROWS, GRID_COLS
    walls = _walls_border(rows, cols)
    for c in range(1, 11):
        walls.add((8, c))
    for c in range(13, 24):
        walls.add((16, c))
    for r in range(9, 16):
        walls.add((r, 12))
    walls.discard((12, 12))
    towers = [
        TowerSpec(cell=(4, 4), order=1, is_chain=False),
        TowerSpec(cell=(5, 17), order=2, is_chain=False),
        TowerSpec(cell=(20, 3), order=3, is_chain=False),
    ]
    decoys = {(4, 17), (5, 16), (6, 17), (20, 17)}
    exit_cell = (22, 22)
    return LevelSpec(
        name="Decoy Signal",
        subtitle="Decoy towers (D) reset your sequence. Identify the real numbered towers.",
        rows=rows,
        cols=cols,
        budget=10 * 35,
        lives=3,
        walls=walls,
        player_start=(2, 2),
        exit_cell=exit_cell,
        towers=towers,
        par_moves=35,
        hint="",
        decoys=decoys,
        has_decoys=True,
    )


def _build_level_2() -> LevelSpec:
    rows, cols = GRID_ROWS, GRID_COLS
    walls = _walls_border(rows, cols)
    for c in range(1, 18):
        walls.add((8, c))
    for c in range(13, 24):
        walls.add((16, c))
    for r in range(9, 16):
        walls.add((r, 12))
    walls.discard((12, 12))
    for r in range(1, 8):
        walls.add((r, 6))
    walls.discard((4, 6))
    towers = [
        TowerSpec(cell=(3, 3), order=1, is_chain=False),
        TowerSpec(cell=(3, 18), order=2, is_chain=False),
        TowerSpec(cell=(12, 20), order=3, is_chain=True),
        TowerSpec(cell=(20, 5), order=4, is_chain=False),
    ]
    gates: Dict[Tuple[int, int], str] = {
        (12, 12): GATE_FROM_WEST,
    }
    plates: Dict[Tuple[int, int], Tuple[int, int]] = {
        (2, 18): (0, 0),
    }
    gate_flips: Dict[Tuple[int, int], Tuple[Tuple[int, int], str]] = {
        (2, 18): ((12, 12), GATE_FROM_EAST),
    }
    decoys = {(20, 20), (12, 3), (3, 17), (4, 18)}
    exit_cell = (22, 22)
    return _with_flip_reverse(
        LevelSpec(
            name="Chain Reaction",
            subtitle="Chain towers (C) deactivate the previous tower. Step plate A to flip the gate.",
            rows=rows,
            cols=cols,
            budget=10 * 60,
            lives=3,
            walls=walls,
            player_start=(2, 2),
            exit_cell=exit_cell,
            towers=towers,
            par_moves=60,
            hint="",
            decoys=decoys,
            gates=gates,
            plates=plates,
            gate_flips=gate_flips,
            has_decoys=True,
            has_chain=True,
            has_gates=True,
            has_plates=True,
        )
    )


def _build_level_3() -> LevelSpec:
    rows, cols = GRID_ROWS, GRID_COLS
    walls = _walls_border(rows, cols)
    for c in range(1, 24):
        walls.add((12, c))
    for c in range(10, 13):
        walls.discard((12, c))
    for c in range(16, 19):
        walls.discard((12, c))
    for r in range(1, 12):
        walls.add((r, 7))
    walls.discard((6, 7))
    for r in range(1, 5):
        walls.add((r, 18))
    for r in range(13, 23):
        walls.add((r, 15))
    walls.discard((17, 15))
    for r in range(18, 23):
        walls.add((r, 9))
    walls.discard((20, 9))
    towers = [
        TowerSpec(cell=(5, 3), order=1, is_chain=False),
        TowerSpec(cell=(4, 20), order=2, is_chain=False),
        TowerSpec(cell=(15, 5), order=3, is_chain=True),
        TowerSpec(cell=(21, 8), order=4, is_chain=False),
    ]
    gates: Dict[Tuple[int, int], str] = {
        (12, 10): GATE_FROM_NORTH,
        (12, 11): GATE_FROM_NORTH,
        (12, 12): GATE_FROM_NORTH,
        (12, 16): GATE_FROM_SOUTH,
        (12, 17): GATE_FROM_SOUTH,
        (12, 18): GATE_FROM_SOUTH,
    }
    plates = {
        (8, 3): (19, 20),
        (2, 13): (0, 0),
        (2, 15): (0, 0),
    }
    plate_groups = [
        (
            frozenset({(2, 13), (2, 15)}),
            frozenset({(12, 10), (12, 11), (12, 12)}),
        ),
    ]
    keys: Set[Tuple[int, int]] = {(20, 20)}
    doors = {
        (12, 10),
        (12, 11),
        (12, 12),
        (12, 16),
        (12, 17),
        (12, 18),
        (19, 20),
    }
    decoys = {(4, 9), (14, 3), (20, 19), (20, 21), (21, 20), (2, 14)}
    exit_cell = (22, 22)
    return LevelSpec(
        name="One-Way Gates",
        subtitle="Gates are CLOSED. Use plates to open the south gate, collect the key to open the north gate.",
        rows=rows,
        cols=cols,
        budget=10 * 80,
        lives=3,
        walls=walls,
        player_start=(2, 2),
        exit_cell=exit_cell,
        towers=towers,
        par_moves=80,
        hint="",
        plates=plates,
        keys=keys,
        doors=doors,
        plate_groups=plate_groups,
        decoys=decoys,
        gates=gates,
        has_decoys=True,
        has_chain=True,
        has_gates=True,
        has_plates=True,
        has_keys=True,
    )


def _build_level_4() -> LevelSpec:
    rows, cols = GRID_ROWS, GRID_COLS
    walls = _walls_border(rows, cols)
    for c in range(1, 7):
        walls.add((8, c))
    for c in range(13, 24):
        walls.add((8, c))
    for c in range(1, 11):
        walls.add((16, c))
    for c in range(14, 24):
        walls.add((16, c))
    for r in range(1, 8):
        walls.add((r, 6))
    walls.discard((4, 6))
    for r in range(1, 8):
        walls.add((r, 18))
    walls.discard((4, 18))
    for r in range(17, 24):
        walls.add((r, 6))
    walls.discard((20, 6))
    for r in range(17, 24):
        walls.add((r, 18))
    walls.discard((20, 18))
    for r in range(22, 24):
        walls.add((r, 20))
    towers = [
        TowerSpec(cell=(3, 3), order=1, is_chain=False),
        TowerSpec(cell=(3, 21), order=2, is_chain=False),
        TowerSpec(cell=(12, 11), order=3, is_chain=True),
        TowerSpec(cell=(20, 3), order=4, is_chain=False),
        TowerSpec(cell=(20, 21), order=5, is_chain=False),
    ]
    jammers: Set[Tuple[int, int]] = set()
    for c in range(7, 18):
        if c != 12:
            jammers.add((10, c))
    for c in range(7, 18):
        if c != 12:
            jammers.add((14, c))
    for r in range(11, 14):
        jammers.add((r, 7))
        jammers.add((r, 17))
    gates: Dict[Tuple[int, int], str] = {
        (10, 12): GATE_FROM_SOUTH,
        (14, 12): GATE_FROM_SOUTH,
    }
    plates: Dict[Tuple[int, int], Tuple[int, int]] = {
        (11, 3): (0, 0),
    }
    plate_groups = [
        (
            frozenset({(11, 3)}),
            frozenset({(16, 11), (16, 12), (16, 13)}),
        ),
    ]
    doors: Set[Tuple[int, int]] = {(16, 11), (16, 12), (16, 13)}
    mj1_path = [
        (9, 1),
        (10, 1),
        (11, 1),
        (12, 1),
        (13, 1),
        (14, 1),
        (15, 1),
        (15, 2),
        (15, 3),
        (15, 4),
        (15, 5),
        (15, 6),
        (14, 6),
        (13, 6),
        (12, 6),
        (11, 6),
        (10, 6),
        (9, 6),
        (9, 5),
        (9, 4),
        (9, 3),
        (9, 2),
    ]
    mj2_path = [
        (9, 23),
        (10, 23),
        (11, 23),
        (12, 23),
        (13, 23),
        (14, 23),
        (15, 23),
        (15, 22),
        (15, 21),
        (15, 20),
        (15, 19),
        (15, 18),
        (14, 18),
        (13, 18),
        (12, 18),
        (11, 18),
        (10, 18),
        (9, 18),
        (9, 19),
        (9, 20),
        (9, 21),
        (9, 22),
    ]
    moving_jammers = [
        GuardSpec(path=mj1_path, start_index=0),
        GuardSpec(path=mj1_path, start_index=7),
        GuardSpec(path=mj1_path, start_index=14),
        GuardSpec(path=mj2_path, start_index=0),
        GuardSpec(path=mj2_path, start_index=7),
        GuardSpec(path=mj2_path, start_index=14),
    ]
    decoys = {(2, 11), (10, 3), (10, 20), (14, 20), (20, 12)}
    exit_cell = (21, 22)
    return LevelSpec(
        name="Jammer Field",
        subtitle="J-box core is mandatory. Time moving jammers, step the plate, avoid decoys.",
        rows=rows,
        cols=cols,
        budget=10 * 100,
        lives=3,
        walls=walls,
        player_start=(2, 2),
        exit_cell=exit_cell,
        towers=towers,
        par_moves=100,
        hint="",
        plates=plates,
        plate_groups=plate_groups,
        doors=doors,
        decoys=decoys,
        gates=gates,
        jammers=jammers,
        moving_jammers=moving_jammers,
        has_decoys=True,
        has_chain=True,
        has_gates=True,
        has_plates=True,
        has_jammers=True,
    )


LEVEL_BUILDERS = [
    _build_level_1,
    _build_level_2,
    _build_level_3,
    _build_level_4,
]
N_LEVELS = len(LEVEL_BUILDERS)

levels = [Level(sprites=[], grid_size=(DISP_W, DISP_H)) for _ in range(N_LEVELS)]


class St02(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)
        self._lives = 3
        self._lives_max = 3
        self._level_scores: List[float] = []
        self._history: List[Dict] = []
        self._consecutive_reset_count = 0

        self._spec: Optional[LevelSpec] = None
        self._runtime: Optional[RuntimeState] = None
        self._budget: int = 0
        self._budget_max: int = 0
        self._flash_win: bool = False
        self._flash_hit: int = 0

        camera = Camera(
            background=PAL_BG,
            letter_box=PAL_BORDER,
            width=DISP_W,
            height=DISP_H,
        )
        super().__init__(
            game_id="st02",
            levels=levels,
            camera=camera,
            available_actions=[0, 1, 2, 3, 4, 7],
        )

    def on_set_level(self, level: Level) -> None:
        self.current_level.remove_all_sprites()
        idx = self._current_level_index
        self._spec = LEVEL_BUILDERS[idx]()
        self._runtime = _make_runtime(self._spec)
        self._budget = self._spec.budget
        self._budget_max = self._spec.budget
        self._lives = self._spec.lives
        self._lives_max = self._spec.lives
        self._flash_win = False
        self._flash_hit = 0
        self._history = []
        self._render()

    def handle_reset(self) -> None:
        self._consecutive_reset_count += 1
        if self._consecutive_reset_count >= 2:
            self._consecutive_reset_count = 0
            self._lives = self._lives_max
            super().full_reset()
        else:
            self._lives = self._lives_max
            super().level_reset()

    def _is_blocked(self, cell: Tuple[int, int], delta: Tuple[int, int]) -> bool:
        assert self._spec is not None
        assert self._runtime is not None
        r, c = cell
        if not (0 <= r < self._spec.rows and 0 <= c < self._spec.cols):
            return True
        if cell in self._spec.walls:
            return True
        if self._spec.has_jammers and cell in self._spec.jammers:
            return True
        if cell in self._spec.doors and cell not in self._runtime.doors_open:
            return True
        if self._spec.has_gates and cell in self._spec.gates:
            return not _gate_passable(
                self._spec, cell, delta, self._runtime.flipped_gates
            )
        return False

    def _reload_level(self) -> None:
        assert self._spec is not None
        self._runtime = _make_runtime(self._spec)
        self._budget = self._budget_max
        self._history = []

    def _save_state(self) -> None:
        assert self._runtime is not None
        self._history.append(
            {
                "player": self._runtime.player,
                "next_tower": self._runtime.next_tower,
                "activated": list(self._runtime.activated),
                "guard_index": self._runtime.guard_index,
                "guard_move_counter": self._runtime.guard_move_counter,
                "doors_open": set(self._runtime.doors_open),
                "keys_collected": set(self._runtime.keys_collected),
                "plates_active": set(self._runtime.plates_active),
                "mj_indices": list(self._runtime.mj_indices),
                "mj_counters": list(self._runtime.mj_counters),
                "flipped_gates": set(self._runtime.flipped_gates),
                "lives": self._lives,
            }
        )

    def _restore_from_undo(self) -> None:
        if not self._history:
            return
        state = self._history.pop()
        assert self._runtime is not None
        self._runtime.player = state["player"]
        self._runtime.next_tower = state["next_tower"]
        self._runtime.activated = state["activated"]
        self._runtime.guard_index = state["guard_index"]
        self._runtime.guard_move_counter = state["guard_move_counter"]
        self._runtime.doors_open = state["doors_open"]
        self._runtime.keys_collected = state["keys_collected"]
        self._runtime.plates_active = state["plates_active"]
        self._runtime.mj_indices = state["mj_indices"]
        self._runtime.mj_counters = state["mj_counters"]
        self._runtime.flipped_gates = state["flipped_gates"]
        self._lives = state["lives"]

    def _try_win(self) -> None:
        self._flash_win = True
        self._render()
        assert self._spec is not None
        sc = min(1.0, self._spec.par_moves / max(self._budget_max - self._budget, 1))
        self._level_scores.append(sc)
        if self._current_level_index < len(self._levels) - 1:
            self.next_level()
        else:
            self.win()
            self.complete_action()

    def _lose_life(self, reason: str) -> None:
        self._lives -= 1
        self._flash_hit = FLASH_TICKS
        if self._lives <= 0:
            self.lose()
            self.complete_action()
        else:
            self._reload_level()
            self._render()
            self.complete_action()

    def step(self) -> None:
        assert self._spec is not None
        assert self._runtime is not None

        if self._flash_hit > 0:
            self._flash_hit -= 1

        if self.action.id == GameAction.RESET:
            self.complete_action()
            return

        if self.action.id == GameAction.ACTION7:
            if self._budget <= 0:
                self._lose_life("budget_exhausted")
                return
            self._restore_from_undo()
            self._budget -= 1
            self._render()
            self.complete_action()
            return

        action = self.action
        delta: Optional[Tuple[int, int]] = {
            GameAction.ACTION1: NORTH,
            GameAction.ACTION2: SOUTH,
            GameAction.ACTION3: WEST,
            GameAction.ACTION4: EAST,
        }.get(action.id)

        if delta is not None:
            self._consecutive_reset_count = 0
            if self._budget <= 0:
                self._lose_life("budget_exhausted")
                return

            self._save_state()

            dr, dc = delta
            pr, pc = self._runtime.player
            nr, nc = pr + dr, pc + dc

            if self._is_blocked((nr, nc), delta):
                self._budget -= 1
                self._render()
                self.complete_action()
                return

            self._runtime.player = (nr, nc)
            self._budget -= 1
            cell = (nr, nc)

            if self._spec.has_plates and cell in self._spec.plates:
                if cell not in self._runtime.plates_active:
                    self._runtime.plates_active.add(cell)
                    door = self._spec.plates[cell]
                    if door != (0, 0):
                        self._runtime.doors_open.add(door)
                    for grp_plates, grp_doors in self._spec.plate_groups:
                        if grp_plates <= self._runtime.plates_active:
                            self._runtime.doors_open.update(grp_doors)
                    if cell in self._spec.gate_flips:
                        gate_cell, _flipped_type = self._spec.gate_flips[cell]
                        self._runtime.flipped_gates.add(gate_cell)

            if self._spec.has_keys and cell in self._spec.keys:
                if cell not in self._runtime.keys_collected:
                    self._runtime.keys_collected.add(cell)
                    plate_doors = set(self._spec.plates.values()) - {(0, 0)}
                    for grp_plates, grp_doors in self._spec.plate_groups:
                        plate_doors.update(grp_doors)
                    for d in self._spec.doors:
                        if d not in plate_doors:
                            self._runtime.doors_open.add(d)

            if self._spec.has_guard:
                gpos = _guard_pos(self._spec, self._runtime)
                if gpos == cell:
                    self._lose_life("guard_collision")
                    return

            if self._spec.moving_jammers:
                _mj_check = (
                    _moving_jammer_danger_cells(self._spec, self._runtime)
                    if self._spec.name == "Jammer Field"
                    else set(_moving_jammer_positions(self._spec, self._runtime))
                )
                if cell in _mj_check:
                    self._lose_life("moving_jammer_collision")
                    return

            if self._spec.has_decoys and cell in self._spec.decoys:
                self._runtime.next_tower = 0
                self._runtime.activated = [False] * len(self._spec.towers)
            else:
                tower = _tower_at(self._spec, cell)
                if tower is not None:
                    tidx = tower.order - 1
                    if tidx == self._runtime.next_tower:
                        self._runtime.activated[tidx] = True
                        self._runtime.next_tower += 1
                        if tower.is_chain and tidx > 0:
                            prev = tidx - 1
                            self._runtime.activated[prev] = False
                            self._runtime.next_tower = prev
                        else:
                            nt = self._runtime.next_tower
                            towers_list = self._spec.towers
                            while (
                                nt < len(towers_list)
                                and self._runtime.activated[nt]
                                and towers_list[nt].is_chain
                            ):
                                nt += 1
                            self._runtime.next_tower = nt
                    elif not self._runtime.activated[tidx]:
                        self._lose_life("wrong_tower_order")
                        return

            if cell == self._spec.exit_cell:
                if _all_activated(self._runtime):
                    self._try_win()
                    return
                else:
                    self._lose_life("exit_before_all_towers")
                    return

            if self._spec.has_guard:
                self._runtime = _advance_guard(self._spec, self._runtime)
                gpos = _guard_pos(self._spec, self._runtime)
                if gpos == self._runtime.player:
                    self._lose_life("guard_collision")
                    return

            if self._spec.moving_jammers:
                self._runtime = _advance_moving_jammers(self._spec, self._runtime)
                _mj_check = (
                    _moving_jammer_danger_cells(self._spec, self._runtime)
                    if self._spec.name == "Jammer Field"
                    else set(_moving_jammer_positions(self._spec, self._runtime))
                )
                if self._runtime.player in _mj_check:
                    self._lose_life("moving_jammer_collision")
                    return

        self._render()
        self.complete_action()

    def _render(self) -> None:
        assert self._spec is not None
        assert self._runtime is not None
        self.current_level.remove_all_sprites()

        CS = CELL_SIZE
        frame = np.full((DISP_H, DISP_W), PAL_BG, dtype=np.uint8)

        bcol = (
            PAL_WIN
            if self._flash_win
            else (PAL_HIT if self._flash_hit > 0 else PAL_BORDER)
        )
        self._flash_win = False
        frame[0, :] = bcol
        frame[-1, :] = bcol
        frame[:, 0] = bcol
        frame[:, -1] = bcol

        pip_w = 3
        pip_gap = 1
        bar_start = BORDER_OFFSET
        bar_end = DISP_W - BORDER_OFFSET - 1
        inner_w = bar_end - bar_start + 1
        lives_px = self._lives_max * pip_w + (self._lives_max - 1) * pip_gap
        bbar_w = inner_w - lives_px - 1
        filled_px = round(self._budget * bbar_w / max(self._budget_max, 1))
        for px in range(bbar_w):
            frame[0, bar_start + px] = (
                PAL_BUDGET_LEFT if px < filled_px else PAL_BUDGET_SPENT
            )
        frame[0, bar_start + bbar_w] = PAL_BUDGET_SPENT
        ls = bar_start + bbar_w + 1
        for i in range(self._lives_max):
            pc2 = PAL_LIVES_LEFT if i < max(self._lives, 0) else PAL_LIVES_SPENT
            for px in range(pip_w):
                col2 = ls + i * (pip_w + pip_gap) + px
                if col2 <= bar_end:
                    frame[0, col2] = pc2

        spec = self._spec
        rt = self._runtime
        pr, pc = rt.player
        gpos = _guard_pos(spec, rt)
        tower_cells = {t.cell: t for t in spec.towers}

        def _fill(r: int, c: int, color: int) -> None:
            py = BORDER_OFFSET + r * CS
            px = BORDER_OFFSET + c * CS
            frame[py : py + CS, px : px + CS] = color

        def _dot(r: int, c: int) -> None:
            py = BORDER_OFFSET + r * CS
            px = BORDER_OFFSET + c * CS
            if c < spec.cols - 1:
                frame[py : py + CS, px + CS - 1] = PAL_GRID_LINE
            if r < spec.rows - 1:
                frame[py + CS - 1, px : px + CS] = PAL_GRID_LINE

        for r in range(spec.rows):
            for c in range(spec.cols):
                cell = (r, c)
                if cell in spec.walls:
                    _fill(r, c, PAL_WALL)
                    continue
                _fill(r, c, PAL_FLOOR)
                _dot(r, c)
                if spec.doors and cell in spec.doors:
                    if cell not in rt.doors_open:
                        _fill(r, c, PAL_DOOR_CLOSED)
                if spec.has_plates and cell in spec.plates:
                    _fill(r, c, PAL_PLATE)
                if spec.has_keys and cell in spec.keys:
                    if cell not in rt.keys_collected:
                        _fill(r, c, PAL_KEY)
                if spec.has_gates and cell in spec.gates:
                    if cell in rt.flipped_gates:
                        _fill(r, c, PAL_GATE_FLIPPED)
                    else:
                        gtype = spec.gates[cell]
                        gcol = (
                            PAL_GATE_NS
                            if gtype in (GATE_FROM_NORTH, GATE_FROM_SOUTH)
                            else PAL_GATE_EW
                        )
                        _fill(r, c, gcol)
                if spec.has_jammers and cell in spec.jammers:
                    _fill(r, c, PAL_JAMMER)
                if spec.has_decoys and cell in spec.decoys:
                    _fill(r, c, PAL_DECOY)
                if cell in tower_cells:
                    t = tower_cells[cell]
                    tidx = t.order - 1
                    if rt.activated[tidx]:
                        _fill(r, c, PAL_TOWER_ON)
                    elif t.is_chain:
                        _fill(r, c, PAL_TOWER_CHAIN)
                    else:
                        _fill(r, c, PAL_TOWER_OFF)
                if cell == spec.exit_cell:
                    ecol = PAL_EXIT_OPEN if _all_activated(rt) else PAL_EXIT_LOCKED
                    _fill(r, c, ecol)
                if spec.has_guard and gpos == cell:
                    _fill(r, c, PAL_GUARD)
                if spec.moving_jammers and cell in _moving_jammer_positions(spec, rt):
                    _fill(r, c, PAL_MOVING_JAMMER)
                if cell == (pr, pc):
                    _fill(r, c, PAL_PLAYER)

        bg_sp = Sprite(
            pixels=frame.tolist(),
            name="background",
            visible=True,
            collidable=False,
            layer=0,
        )
        bg_sp.set_position(0, 0)
        self.current_level.add_sprite(bg_sp)

    @property
    def extra_state(self) -> dict:
        idx = self._current_level_index
        spec = self._spec
        rt = self._runtime
        active_rules = ["sequencing"]
        if spec:
            if spec.has_decoys:
                active_rules.append("decoys")
            if spec.has_chain:
                active_rules.append("chain_reaction")
            if spec.has_gates:
                active_rules.append("one_way_gates")
            if spec.has_jammers:
                active_rules.append("jammers")
            if spec.has_guard:
                active_rules.append("guard_patrol")
        return {
            "budget_remaining": self._budget,
            "budget_used": self._budget_max - self._budget,
            "lives": self._lives,
            "player": list(rt.player) if rt else [],
            "next_tower": rt.next_tower + 1 if rt else 1,
            "activated": list(rt.activated) if rt else [],
            "exit": list(spec.exit_cell) if spec else [],
            "active_rules": active_rules,
            "circuit_title": "SIGNAL TOWER -- Level %d/%d" % (idx + 1, N_LEVELS),
            "level_title": spec.name if spec else "",
            "level_hint": spec.hint if spec else "",
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
        self._engine = St02(seed=seed)
        self._total_turns = 0
        self._done = False
        self._game_won = False
        self._game_over = False
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
        self._game_over = False
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

        frame = e.perform_action(ActionInput(id=game_action), raw=True)

        state_name = frame.state.name if frame and frame.state else ""
        game_won = state_name == "WIN"
        game_over = state_name == "GAME_OVER"

        total_levels = len(self._engine._levels)
        level_reward = 1.0 / total_levels

        if game_won:
            self._done = True
            self._game_won = True
            self._game_over = False
            info["reason"] = "game_complete"
            return StepResult(
                state=self._build_game_state(done=True),
                reward=level_reward,
                done=True,
                info=info,
            )

        if game_over:
            self._done = True
            self._game_won = False
            self._game_over = True
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
        spec = e._spec
        rt = e._runtime
        if spec is None or rt is None:
            return ""

        if e._lives <= 0:
            return (
                "=== GAME OVER ===\nAll lives lost on Level %d: %s.\n"
                "Type 'reset' to retry this level. Type 'reset' again to restart from Level 1."
                % (e._current_level_index + 1, spec.name)
            )

        bar_w = 40
        filled = round(e._budget * bar_w / max(e._budget_max, 1))
        budget_bar = "[" + "#" * filled + "-" * (bar_w - filled) + "]"
        lives_bar = ("*" * max(e._lives, 0)) + ("." * (e._lives_max - max(e._lives, 0)))

        active_rules = ["sequencing"]
        if spec.has_decoys:
            active_rules.append("decoys")
        if spec.has_chain:
            active_rules.append("chain")
        if spec.has_gates:
            active_rules.append("gates")
        if spec.has_jammers:
            active_rules.append("jammers")
        if spec.has_guard:
            active_rules.append("guard")

        act_str = " ".join(
            ("t%d" if rt.activated[i] else "T%d") % (i + 1)
            for i in range(len(spec.towers))
        )

        lines: List[str] = [
            "=== SIGNAL TOWER L%d/%d: %s ==="
            % (e._current_level_index + 1, N_LEVELS, spec.name),
            "Budget %s %d/%d  Lives [%s]"
            % (budget_bar, e._budget, e._budget_max, lives_bar),
            "Rules: %s" % " | ".join(active_rules),
            "Towers: %s  Next: T%d  Player=%s"
            % (act_str, rt.next_tower + 1, rt.player),
            "",
        ]

        pr, pc = rt.player
        tower_cells = {t.cell: t for t in spec.towers}
        gpos = _guard_pos(spec, rt)

        for r in range(spec.rows):
            row: List[str] = []
            for c in range(spec.cols):
                cell = (r, c)
                if cell == (pr, pc):
                    row.append("P")
                elif cell == spec.exit_cell:
                    row.append("E" if _all_activated(rt) else "e")
                elif cell in spec.walls:
                    row.append("#")
                elif spec.doors and cell in spec.doors and cell not in rt.doors_open:
                    row.append("X")
                elif spec.has_guard and gpos == cell:
                    row.append("!")
                elif spec.moving_jammers and cell in _moving_jammer_positions(spec, rt):
                    row.append("M")
                elif spec.has_jammers and cell in spec.jammers:
                    row.append("J")
                elif spec.has_decoys and cell in spec.decoys:
                    row.append("D")
                elif cell in tower_cells:
                    t = tower_cells[cell]
                    tidx = t.order - 1
                    row.append(
                        str(t.order).lower() if rt.activated[tidx] else str(t.order)
                    )
                elif spec.has_plates and cell in spec.plates:
                    row.append("A")
                elif (
                    spec.has_keys
                    and cell in spec.keys
                    and cell not in rt.keys_collected
                ):
                    row.append("K")
                elif spec.has_gates and cell in spec.gates:
                    if cell in rt.flipped_gates:
                        eff = (
                            _effective_gate_type(spec, cell, rt.flipped_gates)
                            or spec.gates[cell]
                        )
                        row.append({"GN": "v", "GS": "^", "GW": ">", "GE": "<"}[eff])
                    else:
                        gtype = spec.gates[cell]
                        row.append({"GN": "v", "GS": "^", "GW": ">", "GE": "<"}[gtype])
                else:
                    row.append(".")
            lines.append(" ".join(row))

        return "\n".join(lines)

    def _build_game_state(self, done: bool = False) -> GameState:
        e = self._engine
        valid_actions = self.get_actions() if not done else None
        return GameState(
            text_observation=self._build_text_observation(),
            image_observation=None,
            valid_actions=valid_actions,
            turn=self._total_turns,
            metadata={
                "total_levels": len(e._levels),
                "level_index": e.level_index,
                "levels_completed": getattr(e, "_score", 0),
                "game_over": self._game_over,
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
