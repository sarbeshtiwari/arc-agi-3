import struct
import zlib
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from arcengine import (
    ARCBaseGame,
    ActionInput,
    Camera,
    GameAction,
    Level,
    Sprite,
)


@dataclass
class GameState:
    text_observation: str
    image_observation: Optional[bytes]
    valid_actions: Optional[List[str]]
    turn: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StepResult:
    state: GameState
    reward: float
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)


class Fluid(IntEnum):
    EMPTY = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    WALL = 5
    GARBAGE_CYAN = 6
    GARBAGE_MAROON = 7
    GARBAGE_GRAY = 8
    PINK = 9
    ORANGE = 10


FLUID_PALETTE: Dict[int, int] = {
    Fluid.EMPTY: 0,
    Fluid.BLUE: 9,
    Fluid.RED: 8,
    Fluid.GREEN: 14,
    Fluid.YELLOW: 11,
    Fluid.WALL: 4,
    Fluid.GARBAGE_CYAN: 10,
    Fluid.GARBAGE_MAROON: 13,
    Fluid.GARBAGE_GRAY: 2,
    Fluid.PINK: 6,
    Fluid.ORANGE: 12,
}

GARBAGE_TYPES: List[int] = [
    int(Fluid.GARBAGE_CYAN),
    int(Fluid.GARBAGE_MAROON),
    int(Fluid.GARBAGE_GRAY),
]

PAL_BG = 0
PAL_BORDER = 5
PAL_WALL = 4
PAL_CURSOR = 11
PAL_SELECTED = 6
PAL_WIN = 14
PAL_TARGET = 7
PAL_BUDGET_FULL = 14
PAL_BUDGET_MID = 11
PAL_BUDGET_LOW = 8
PAL_BUDGET_EMPTY = 4
PAL_LIVES_LEFT = 8
PAL_LIVES_SPENT = 4
PAL_BEAKER_INTERIOR = 5
PAL_TRASH = 13
PAL_TRASH_INNER = 6

TRASH_CURSOR_IDX = -1

BUDGET_MULTIPLIER = 6

GRID_W = 64
GRID_H = 64
UNIT_H = 4
BEAKER_W = 8
BEAKER_INNER = BEAKER_W - 2
HUD_H = 3
HUD_Y = 1

GRID_ROWS = 30
GRID_COLS = 30

FLUID_CHARS: Dict[int, str] = {
    Fluid.EMPTY: ".",
    Fluid.BLUE: "B",
    Fluid.RED: "R",
    Fluid.GREEN: "G",
    Fluid.YELLOW: "Y",
    Fluid.WALL: "#",
    Fluid.GARBAGE_CYAN: "c",
    Fluid.GARBAGE_MAROON: "m",
    Fluid.GARBAGE_GRAY: "g",
    Fluid.PINK: "P",
    Fluid.ORANGE: "O",
}

_ARC_PALETTE = [
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


@dataclass
class ReactionRule:
    color_a: int
    color_b: int


@dataclass
class FluidLevelSpec:
    name: str
    subtitle: str
    rows: int
    cols: int
    num_beakers: int
    beaker_capacity: int
    initial_layers: List[List[int]]
    target_level: int
    par_pours: int
    max_actions: int
    lives: int
    reactions: List[ReactionRule] = field(default_factory=list)
    base_pours: List[Tuple[int, int]] = field(default_factory=list)
    par: int = 0
    trash_capacity: int = 0
    target_levels: List[int] = field(default_factory=list)


def _jar_layout(
    num_beakers: int, max_cap: int
) -> Tuple[List[Tuple[int, int]], int, int]:
    gap = 4 if num_beakers <= 4 else 2
    total_w = num_beakers * BEAKER_W + (num_beakers - 1) * gap
    start_x = (GRID_W - total_w) // 2
    jar_h = max_cap * UNIT_H + 2
    area_top = HUD_Y + HUD_H + 1
    area_bot = GRID_H - 6
    avail = area_bot - area_top
    jar_top = area_top + max(0, (avail - jar_h) // 2)
    if jar_top < area_top:
        jar_top = area_top
    positions = []
    for i in range(num_beakers):
        x = start_x + i * (BEAKER_W + gap)
        positions.append((x, jar_top))
    return positions, jar_h, gap


def _top_contiguous_block(layers: List[int]) -> Tuple[int, int]:
    if not layers:
        return 0, 0
    top_idx = len(layers) - 1
    color = layers[top_idx]
    count = 0
    for i in range(top_idx, -1, -1):
        if layers[i] == color:
            count += 1
        else:
            break
    return color, count


def pour_fluid(
    layers: List[List[int]],
    source_idx: int,
    target_idx: int,
    beaker_capacity: int,
    reactions: List[ReactionRule],
) -> Tuple[bool, int]:
    source = layers[source_idx]
    target = layers[target_idx]
    if not source:
        return False, 0
    color, block_size = _top_contiguous_block(source)
    if block_size == 0:
        return False, 0
    space = beaker_capacity - len(target)
    if space <= 0:
        return False, 0
    actual = min(block_size, space)
    total_destroyed = 0
    for _ in range(actual):
        unit = source.pop()
        target.append(unit)
        destroyed = _apply_reactions(target, reactions)
        total_destroyed += destroyed
    return True, total_destroyed


def _apply_reactions(
    layers: List[int],
    reactions: List[ReactionRule],
) -> int:
    total_destroyed = 0
    while len(layers) >= 2:
        top_idx = len(layers) - 1
        below_idx = top_idx - 1
        top_color = layers[top_idx]
        below_color = layers[below_idx]
        if (top_color == Fluid.YELLOW and _is_garbage(below_color)) or (
            _is_garbage(top_color) and below_color == Fluid.YELLOW
        ):
            layers.pop()
            layers.pop()
            layers.append(int(Fluid.BLUE))
            total_destroyed += 2
            continue
        if (top_color == Fluid.PINK and below_color == Fluid.ORANGE) or (
            top_color == Fluid.ORANGE and below_color == Fluid.PINK
        ):
            layers.pop()
            layers.pop()
            remaining = len(layers)
            if remaining % 3 == 1:
                layers.append(int(Fluid.GREEN))
            elif remaining % 3 == 2:
                layers.append(int(Fluid.RED))
            total_destroyed += 2
            continue
        reacted = False
        for r in reactions:
            if (top_color == r.color_a and below_color == r.color_b) or (
                top_color == r.color_b and below_color == r.color_a
            ):
                layers.pop()
                layers.pop()
                total_destroyed += 2
                reacted = True
                break
        if not reacted:
            break
    return total_destroyed


def check_balanced(
    layers: List[List[int]],
    target_level: int,
    target_levels: Optional[List[int]] = None,
) -> bool:
    for b in layers:
        for cell in b:
            if _is_garbage(cell):
                return False
    if target_levels:
        return all(len(layers[i]) == target_levels[i] for i in range(len(layers)))
    return all(len(b) == target_level for b in layers)


def trash_pour(
    layers: List[List[int]],
    source_idx: int,
    trash_used: int = 0,
    trash_capacity: int = 0,
) -> bool:
    if trash_capacity > 0 and trash_used >= trash_capacity:
        return False
    source = layers[source_idx]
    if not source:
        return False
    color, block_size = _top_contiguous_block(source)
    if block_size == 0:
        return False
    if not _is_garbage(color):
        return False
    for _ in range(block_size):
        source.pop()
    return True


def _is_garbage(value: int) -> bool:
    return value in GARBAGE_TYPES


def _compute_min_actions(
    base_pours: List[Tuple[int, int]],
) -> int:
    if not base_pours:
        return 0
    cursor = 0
    total = 0
    for src, tgt in base_pours:
        total += abs(cursor - src) + 1 + abs(src - tgt) + 1
        cursor = tgt
    return total


def _verify_level_setup(idx: int, spec: FluidLevelSpec) -> List[str]:
    errors: List[str] = []
    layers = [list(l) for l in spec.initial_layers]
    pour_count = 0
    trash_used = 0
    for step_idx, (source, target) in enumerate(spec.base_pours):
        if source < 0 or source >= spec.num_beakers:
            errors.append(
                "L%d step %d: source %d out of range" % (idx + 1, step_idx, source)
            )
            continue
        if target == TRASH_CURSOR_IDX:
            ok = trash_pour(layers, source, trash_used, spec.trash_capacity)
            if not ok:
                errors.append(
                    "L%d step %d: trash pour from %d failed"
                    % (idx + 1, step_idx, source)
                )
            else:
                trash_used += 1
            continue
        if target < 0 or target >= spec.num_beakers:
            errors.append(
                "L%d step %d: target %d out of range" % (idx + 1, step_idx, target)
            )
            continue
        if source == target:
            errors.append(
                "L%d step %d: source == target (%d)" % (idx + 1, step_idx, source)
            )
            continue
        success, destroyed = pour_fluid(
            layers,
            source,
            target,
            spec.beaker_capacity,
            spec.reactions,
        )
        if not success:
            errors.append(
                "L%d step %d: pour (%d,%d) failed, layers=%s"
                % (idx + 1, step_idx, source, target, [len(l) for l in layers])
            )
            continue
        pour_count += 1
    tgt_list = spec.target_levels if spec.target_levels else None
    if not check_balanced(layers, spec.target_level, tgt_list):
        errors.append(
            "L%d '%s': solution doesn't balance! levels=%s target=%s"
            % (
                idx + 1,
                spec.name,
                [len(l) for l in layers],
                tgt_list if tgt_list else spec.target_level,
            )
        )
    if spec.max_actions <= 0:
        errors.append(
            "L%d: computed budget is %d (must be positive)"
            % (idx + 1, spec.max_actions)
        )
    return errors


def _tight_capacity(initial_layers: List[List[int]], target_level: int) -> int:
    max_fill = max((len(l) for l in initial_layers), default=0)
    return max(target_level + 2, max_fill + 2)


def _build_level_3() -> FluidLevelSpec:
    C = int(Fluid.GARBAGE_CYAN)
    M = int(Fluid.GARBAGE_MAROON)
    G_G = int(Fluid.GARBAGE_GRAY)
    init: List[List[int]] = [
        [
            Fluid.YELLOW,
            Fluid.PINK,
            Fluid.RED,
            Fluid.GREEN,
            Fluid.ORANGE,
            Fluid.BLUE,
            Fluid.RED,
            C,
        ],
        [Fluid.YELLOW, Fluid.GREEN, Fluid.ORANGE, Fluid.PINK, Fluid.BLUE, Fluid.RED, M],
        [
            Fluid.YELLOW,
            Fluid.RED,
            Fluid.BLUE,
            Fluid.ORANGE,
            Fluid.GREEN,
            Fluid.PINK,
            G_G,
        ],
        [
            Fluid.YELLOW,
            Fluid.ORANGE,
            Fluid.GREEN,
            Fluid.RED,
            Fluid.PINK,
            Fluid.BLUE,
            Fluid.GREEN,
        ],
        [Fluid.YELLOW, Fluid.BLUE, Fluid.ORANGE, Fluid.RED],
    ]
    base_sequence: List[Tuple[int, int]] = [
        (0, -1),
        (1, -1),
        (2, -1),
        (0, 4),
        (3, 4),
    ]
    cap = _tight_capacity(init, 6)
    return FluidLevelSpec(
        name="Toxic Brew",
        subtitle="",
        rows=GRID_ROWS,
        cols=GRID_COLS,
        num_beakers=5,
        beaker_capacity=cap,
        initial_layers=init,
        target_level=6,
        par_pours=5,
        max_actions=0,
        lives=3,
        trash_capacity=3,
        base_pours=base_sequence,
    )


def _build_level_4() -> FluidLevelSpec:
    C = int(Fluid.GARBAGE_CYAN)
    M = int(Fluid.GARBAGE_MAROON)
    G_G = int(Fluid.GARBAGE_GRAY)
    init: List[List[int]] = [
        [
            Fluid.YELLOW,
            Fluid.GREEN,
            Fluid.BLUE,
            Fluid.PINK,
            Fluid.RED,
            Fluid.ORANGE,
            Fluid.BLUE,
            Fluid.RED,
            M,
        ],
        [
            Fluid.YELLOW,
            Fluid.ORANGE,
            Fluid.RED,
            Fluid.BLUE,
            Fluid.PINK,
            Fluid.GREEN,
            Fluid.BLUE,
            Fluid.RED,
        ],
        [Fluid.YELLOW, Fluid.PINK, Fluid.RED, Fluid.ORANGE, Fluid.GREEN, Fluid.BLUE, C],
        [
            Fluid.YELLOW,
            Fluid.RED,
            Fluid.BLUE,
            Fluid.GREEN,
            Fluid.ORANGE,
            Fluid.BLUE,
            Fluid.PINK,
        ],
        [
            Fluid.YELLOW,
            Fluid.ORANGE,
            Fluid.BLUE,
            Fluid.GREEN,
            Fluid.PINK,
            Fluid.BLUE,
            Fluid.RED,
            G_G,
        ],
    ]
    base_sequence: List[Tuple[int, int]] = [
        (0, -1),
        (2, -1),
        (4, -1),
        (0, 2),
        (0, 1),
        (3, 2),
        (4, 1),
    ]
    cap = _tight_capacity(init, 6)
    return FluidLevelSpec(
        name="Frozen Depths",
        subtitle="",
        rows=GRID_ROWS,
        cols=GRID_COLS,
        num_beakers=5,
        beaker_capacity=cap,
        initial_layers=init,
        target_level=6,
        par_pours=7,
        max_actions=0,
        lives=3,
        reactions=[ReactionRule(Fluid.RED, Fluid.BLUE)],
        trash_capacity=3,
        base_pours=base_sequence,
    )


def _build_level_5() -> FluidLevelSpec:
    M = int(Fluid.GARBAGE_MAROON)
    G_G = int(Fluid.GARBAGE_GRAY)
    init: List[List[int]] = [
        [
            Fluid.YELLOW,
            Fluid.GREEN,
            Fluid.ORANGE,
            Fluid.RED,
            Fluid.PINK,
            Fluid.BLUE,
            Fluid.RED,
            M,
        ],
        [
            Fluid.YELLOW,
            Fluid.ORANGE,
            Fluid.PINK,
            Fluid.BLUE,
            Fluid.GREEN,
            Fluid.RED,
            Fluid.PINK,
            Fluid.ORANGE,
        ],
        [
            Fluid.YELLOW,
            Fluid.RED,
            Fluid.GREEN,
            Fluid.PINK,
            Fluid.ORANGE,
            Fluid.BLUE,
            Fluid.RED,
            Fluid.BLUE,
        ],
        [Fluid.BLUE, Fluid.ORANGE, Fluid.GREEN, Fluid.PINK, G_G],
        [
            Fluid.YELLOW,
            Fluid.PINK,
            Fluid.RED,
            Fluid.ORANGE,
            Fluid.GREEN,
            Fluid.BLUE,
            Fluid.RED,
            Fluid.ORANGE,
        ],
    ]
    base_sequence: List[Tuple[int, int]] = [
        (0, -1),
        (3, -1),
        (2, 0),
        (1, 2),
        (2, 3),
        (2, 0),
    ]
    cap = _tight_capacity(init, 8)
    return FluidLevelSpec(
        name="Siphon Cascade",
        subtitle="",
        rows=GRID_ROWS,
        cols=GRID_COLS,
        num_beakers=5,
        beaker_capacity=cap,
        initial_layers=init,
        target_level=0,
        par_pours=6,
        max_actions=0,
        lives=3,
        reactions=[ReactionRule(Fluid.RED, Fluid.BLUE)],
        trash_capacity=2,
        base_pours=base_sequence,
        target_levels=[5, 7, 6, 3, 8],
    )


def _build_level_6() -> FluidLevelSpec:
    M = int(Fluid.GARBAGE_MAROON)
    G_G = int(Fluid.GARBAGE_GRAY)
    init: List[List[int]] = [
        [Fluid.YELLOW, Fluid.GREEN, Fluid.ORANGE, Fluid.PINK, Fluid.BLUE, Fluid.RED, M],
        [Fluid.YELLOW, Fluid.GREEN, Fluid.PINK, Fluid.ORANGE, Fluid.RED, Fluid.GREEN],
        [Fluid.ORANGE, Fluid.GREEN, Fluid.BLUE, Fluid.PINK, Fluid.ORANGE, Fluid.BLUE],
        [
            Fluid.YELLOW,
            Fluid.PINK,
            Fluid.RED,
            Fluid.GREEN,
            Fluid.ORANGE,
            Fluid.BLUE,
            G_G,
        ],
        [
            Fluid.YELLOW,
            Fluid.GREEN,
            Fluid.ORANGE,
            Fluid.PINK,
            Fluid.RED,
            Fluid.BLUE,
            Fluid.GREEN,
        ],
        [
            Fluid.YELLOW,
            Fluid.RED,
            Fluid.GREEN,
            Fluid.PINK,
            Fluid.ORANGE,
            Fluid.BLUE,
            Fluid.GREEN,
            Fluid.ORANGE,
        ],
    ]
    base_sequence: List[Tuple[int, int]] = [
        (0, -1),
        (3, -1),
        (0, 2),
        (5, 1),
        (5, 2),
        (0, 1),
    ]
    cap = _tight_capacity(init, 8)
    return FluidLevelSpec(
        name="Deep Synthesis",
        subtitle="",
        rows=GRID_ROWS,
        cols=GRID_COLS,
        num_beakers=6,
        beaker_capacity=cap,
        initial_layers=init,
        target_level=0,
        par_pours=6,
        max_actions=0,
        lives=3,
        reactions=[ReactionRule(Fluid.RED, Fluid.BLUE)],
        trash_capacity=2,
        base_pours=base_sequence,
        target_levels=[4, 8, 4, 6, 7, 6],
    )


LEVEL_BUILDERS = [
    _build_level_3,
    _build_level_4,
    _build_level_5,
    _build_level_6,
]
N_LEVELS = len(LEVEL_BUILDERS)

_SPECS: List[FluidLevelSpec] = [b() for b in LEVEL_BUILDERS]

for _spec in _SPECS:
    _min = _compute_min_actions(_spec.base_pours)
    _spec.par = _min
    _spec.max_actions = BUDGET_MULTIPLIER * _min


def _display_size_for_level(spec: FluidLevelSpec) -> Tuple[int, int]:
    return GRID_W, GRID_H


_LEVEL_DISPLAY_SIZES: List[Tuple[int, int]] = [
    _display_size_for_level(s) for s in _SPECS
]
_MAX_DISPLAY_W = GRID_W
_MAX_DISPLAY_H = GRID_H

levels = [Level(sprites=[], grid_size=(GRID_W, GRID_H)) for _ in _SPECS]

_VERIFY_ERRORS: List[str] = []
for _idx, _spec in enumerate(_SPECS):
    _VERIFY_ERRORS.extend(_verify_level_setup(_idx, _spec))

if _VERIFY_ERRORS:
    raise AssertionError("Level verification failed: " + "; ".join(_VERIFY_ERRORS))


class Fb12(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._history: List[Dict[str, Any]] = []
        self._spec: Optional[FluidLevelSpec] = None
        self._layers: List[List[int]] = []
        self._budget: int = 0
        self._budget_max: int = 0
        self._lives: int = 3
        self._lives_max: int = 3
        self._cursor: int = 0
        self._selected_source: int = -1
        self._pour_count: int = 0
        self._total_pours: int = 0
        self._level_scores: List[float] = []
        self._flash_win: bool = False
        self._just_transitioned: bool = False
        self._game_over: bool = False
        self._won: bool = False
        self._pending_retry: bool = False
        self._trash_used: int = 0
        camera = Camera(
            background=PAL_BG,
            letter_box=PAL_BORDER,
            width=GRID_W,
            height=GRID_H,
        )
        super().__init__(
            game_id="fb12",
            levels=levels,
            camera=camera,
            available_actions=[0, 3, 4, 5, 7],
            seed=seed,
        )

    def on_set_level(self, level: object) -> None:
        self.current_level.remove_all_sprites()
        idx = self._current_level_index
        spec = LEVEL_BUILDERS[idx]()
        self._spec = spec
        self._layers = [list(l) for l in spec.initial_layers]
        self._budget = _SPECS[idx].max_actions
        self._budget_max = _SPECS[idx].max_actions
        self._lives = spec.lives
        self._lives_max = spec.lives
        self._cursor = 0
        self._selected_source = -1
        self._pour_count = 0
        self._total_pours = 0
        self._trash_used = 0
        self._flash_win = False
        self._just_transitioned = True
        self._game_over = False
        self._won = False
        self._pending_retry = False
        self._history = []
        self._render()

    def _reload_level(self) -> None:
        assert self._spec is not None
        self._layers = [list(l) for l in self._spec.initial_layers]
        self._budget = self._budget_max
        self._cursor = 0
        self._selected_source = -1
        self._pour_count = 0
        self._total_pours = 0
        self._trash_used = 0
        self._history = []

    def full_reset(self) -> None:
        self._current_level_index = 0
        self._level_scores = []
        self._game_over = False
        self._won = False
        self._pending_retry = False
        self._history = []

    def _save_state(self) -> None:
        self._history.append(
            {
                "layers": [list(l) for l in self._layers],
                "budget": self._budget,
                "cursor": self._cursor,
                "selected_source": self._selected_source,
                "pour_count": self._pour_count,
                "total_pours": self._total_pours,
                "trash_used": self._trash_used,
            }
        )

    def _handle_undo(self) -> None:
        if not self._history:
            return
        if self._budget <= 0:
            return
        state = self._history.pop()
        self._layers = state["layers"]
        self._cursor = state["cursor"]
        self._selected_source = state["selected_source"]
        self._pour_count = state["pour_count"]
        self._total_pours = state["total_pours"]
        self._trash_used = state["trash_used"]
        self._budget -= 1

    def _try_win(self) -> bool:
        self._flash_win = True
        self._render()
        assert self._spec is not None
        sc = min(1.0, self._spec.par_pours / max(self._total_pours, 1))
        self._level_scores.append(sc)
        if self._current_level_index < len(self._levels) - 1:
            self.next_level()
            self.complete_action()
        else:
            self._won = True
            self.win()
            self.complete_action()
        return True

    def _lose_life(self, reason: str) -> bool:
        self._lives -= 1
        if self._lives <= 0:
            self._game_over = True
            self._render()
            self.lose()
            self.complete_action()
            return True
        self._render()
        self._reload_level()
        return False

    def _do_pour_logic(self) -> None:
        assert self._spec is not None
        if self._cursor == TRASH_CURSOR_IDX:
            if self._selected_source >= 0:
                ok = trash_pour(
                    self._layers,
                    self._selected_source,
                    self._trash_used,
                    self._spec.trash_capacity,
                )
                if ok:
                    self._total_pours += 1
                    self._trash_used += 1
                self._selected_source = -1
            return
        if self._selected_source < 0:
            if self._cursor >= 0:
                self._selected_source = self._cursor
        else:
            src = self._selected_source
            tgt = self._cursor
            self._selected_source = -1
            if src == tgt:
                return
            all_rxns = self._spec.reactions
            success, _ = pour_fluid(
                self._layers,
                src,
                tgt,
                self._spec.beaker_capacity,
                all_rxns,
            )
            if success:
                self._total_pours += 1
                self._pour_count += 1

    def step(self) -> None:
        if self._just_transitioned:
            self._just_transitioned = False
            self._render()
            self.complete_action()
            return

        assert self._spec is not None

        if self._game_over:
            if self.action.id == GameAction.RESET:
                self._game_over = False
                self._pending_retry = True
                self._reload_level()
                self._render()
                self.complete_action()
                return
            self._render()
            self.complete_action()
            return

        action = self.action

        if action.id == GameAction.ACTION7:
            self._handle_undo()
            if self._budget == 0:
                if self._lose_life("budget_exhausted"):
                    return
            self._render()
            self.complete_action()
            return

        is_game_action = action.id in (
            GameAction.ACTION3,
            GameAction.ACTION4,
            GameAction.ACTION5,
        )

        if is_game_action:
            if self._budget <= 0:
                self._render()
                self.complete_action()
                return
            self._save_state()
            self._budget -= 1
            self._pending_retry = False

        if action.id == GameAction.ACTION3:
            if self._cursor > TRASH_CURSOR_IDX:
                self._cursor -= 1

        elif action.id == GameAction.ACTION4:
            if self._cursor < self._spec.num_beakers - 1:
                self._cursor += 1

        elif action.id == GameAction.ACTION5:
            self._do_pour_logic()
            if check_balanced(
                self._layers, self._spec.target_level, self._spec.target_levels or None
            ):
                self._try_win()
                return

        elif action.id == GameAction.RESET:
            if self._pending_retry:
                self.full_reset()
                self.on_set_level(None)
                self._render()
                self.complete_action()
                return
            self._reload_level()
            self._render()
            self.complete_action()
            return

        if is_game_action and self._budget == 0:
            if self._lose_life("budget_exhausted"):
                return
            self._render()
            self.complete_action()
            return

        self._render()
        self.complete_action()

    def _make_frame(self) -> list:
        if not self._spec:
            return [[0] * GRID_W for _ in range(GRID_H)]
        nb = self._spec.num_beakers
        cap = self._spec.beaker_capacity
        frame = np.full((GRID_H, GRID_W), PAL_BG, dtype=np.uint8)

        def _px(x: int, y: int, c: int) -> None:
            if 0 <= x < GRID_W and 0 <= y < GRID_H:
                frame[y, x] = c

        def _hline(x: int, y: int, length: int, c: int) -> None:
            for dx in range(length):
                _px(x + dx, y, c)

        def _vline(x: int, y: int, length: int, c: int) -> None:
            for dy in range(length):
                _px(x, y + dy, c)

        def _rect(x: int, y: int, w: int, h: int, c: int) -> None:
            for dy in range(h):
                for dx in range(w):
                    _px(x + dx, y + dy, c)

        self._render_border(frame, _hline, _vline)
        self._render_hud(frame, _rect)
        positions, jar_h = self._render_trash(frame, nb, cap, _px, _hline)
        self._render_beakers(
            frame, nb, cap, positions, jar_h, _px, _hline, _vline, _rect
        )
        return frame.tolist()

    def _render(self) -> None:
        assert self._spec is not None
        self.current_level.remove_all_sprites()
        frame_list = self._make_frame()
        bg_sp = Sprite(
            pixels=frame_list,
            name="background",
            visible=True,
            collidable=False,
            layer=0,
        )
        bg_sp.set_position(0, 0)
        self.current_level.add_sprite(bg_sp)

    def _render_border(self, frame: Any, _hline: Any, _vline: Any) -> None:
        bcol = PAL_WIN if self._flash_win else PAL_BORDER
        if self._flash_win:
            self._flash_win = False
        _hline(0, 0, GRID_W, bcol)
        _hline(0, GRID_H - 1, GRID_W, bcol)
        _vline(0, 0, GRID_H, bcol)
        _vline(GRID_W - 1, 0, GRID_H, bcol)

    def _render_hud(self, frame: Any, _rect: Any) -> None:
        for i in range(self._lives_max):
            lx = GRID_W - 4 - i * 4
            col = PAL_LIVES_LEFT if i < self._lives else PAL_LIVES_SPENT
            _rect(lx, HUD_Y, 2, 2, col)

        bar_x = 2
        bar_w = GRID_W - 4 - self._lives_max * 4 - 2
        bar_h = 2
        _rect(bar_x, HUD_Y, bar_w, bar_h, PAL_BUDGET_EMPTY)
        if self._budget_max > 0:
            filled = max(0, bar_w * self._budget // self._budget_max)
            if filled > 0:
                ratio = self._budget / self._budget_max
                if ratio > 0.5:
                    fill_c = PAL_BUDGET_FULL
                elif ratio > 0.25:
                    fill_c = PAL_BUDGET_MID
                else:
                    fill_c = PAL_BUDGET_LOW
                _rect(bar_x, HUD_Y, filled, bar_h, fill_c)

    def _render_trash(
        self, frame: Any, nb: int, cap: int, _px: Any, _hline: Any
    ) -> Tuple[List[Tuple[int, int]], int]:
        positions, jar_h, gap = _jar_layout(nb, cap)

        trash_cx = GRID_W // 2
        trash_cy = HUD_Y + HUD_H
        _px(trash_cx - 3, trash_cy, PAL_TRASH)
        _px(trash_cx + 3, trash_cy, PAL_TRASH)
        _hline(trash_cx - 2, trash_cy, 5, PAL_TRASH)
        _hline(trash_cx - 2, trash_cy + 1, 5, PAL_TRASH_INNER)
        _hline(trash_cx - 1, trash_cy + 2, 3, PAL_TRASH_INNER)
        _px(trash_cx, trash_cy + 3, PAL_TRASH_INNER)

        if self._cursor == TRASH_CURSOR_IDX and not self._game_over:
            _hline(trash_cx - 4, trash_cy - 1, 9, PAL_CURSOR)
            _px(trash_cx - 4, trash_cy, PAL_CURSOR)
            _px(trash_cx + 4, trash_cy, PAL_CURSOR)
            _px(trash_cx - 3, trash_cy + 3, PAL_CURSOR)
            _px(trash_cx + 3, trash_cy + 3, PAL_CURSOR)
            _hline(trash_cx - 2, trash_cy + 4, 5, PAL_CURSOR)

        return positions, jar_h

    def _render_beakers(
        self,
        frame: Any,
        nb: int,
        cap: int,
        positions: List[Tuple[int, int]],
        jar_h: int,
        _px: Any,
        _hline: Any,
        _vline: Any,
        _rect: Any,
    ) -> None:
        assert self._spec is not None
        for bi in range(nb):
            jx, jy = positions[bi]
            blayers = self._layers[bi]

            _vline(jx, jy, jar_h, PAL_WALL)
            _vline(jx + BEAKER_W - 1, jy, jar_h, PAL_WALL)
            _hline(jx, jy + jar_h - 1, BEAKER_W, PAL_WALL)

            inner_h = jar_h - 2
            inner_top = jy + 1
            cap_px = cap * UNIT_H
            empty_offset = inner_h - cap_px

            for row in range(inner_h):
                ry = inner_top + row
                unit_from_top = row - empty_offset
                unit_idx = cap - 1 - (unit_from_top // UNIT_H)

                if unit_from_top < 0:
                    _hline(jx + 1, ry, BEAKER_INNER, PAL_BG)
                elif unit_idx < len(blayers) and blayers[unit_idx] is not None:
                    color = blayers[unit_idx]
                    pal = FLUID_PALETTE.get(color, 0)
                    _hline(jx + 1, ry, BEAKER_INNER, pal)
                else:
                    _hline(jx + 1, ry, BEAKER_INNER, PAL_BEAKER_INTERIOR)

            tgt_for_bi = self._spec.target_level
            if self._spec.target_levels and bi < len(self._spec.target_levels):
                tgt_for_bi = self._spec.target_levels[bi]
            if 0 < tgt_for_bi < cap:
                marker_unit = tgt_for_bi
                marker_y = inner_top + inner_h - marker_unit * UNIT_H - 1
                if inner_top <= marker_y < inner_top + inner_h:
                    for dx in range(BEAKER_INNER):
                        if dx % 2 == 0:
                            _px(jx + 1 + dx, marker_y, PAL_TARGET)

            if (bi == self._cursor) and not self._game_over:
                ax = jx + BEAKER_W // 2
                ay = jy + jar_h + 2
                _px(ax - 2, ay, PAL_CURSOR)
                _px(ax - 1, ay, PAL_CURSOR)
                _px(ax, ay, PAL_CURSOR)
                _px(ax + 1, ay, PAL_CURSOR)
                _px(ax + 2, ay, PAL_CURSOR)
                _px(ax - 1, ay + 1, PAL_CURSOR)
                _px(ax, ay + 1, PAL_CURSOR)
                _px(ax + 1, ay + 1, PAL_CURSOR)
                _px(ax, ay + 2, PAL_CURSOR)

    @property
    def extra_state(self) -> dict:
        idx = self._current_level_index
        spec = self._spec
        nb = spec.num_beakers if spec else 0
        beaker_levels = [len(l) for l in self._layers]
        balanced = (
            check_balanced(self._layers, spec.target_level, spec.target_levels or None)
            if spec
            else False
        )

        features = [
            "Level %d/%d: %s" % (idx + 1, N_LEVELS, spec.name if spec else ""),
            "Budget=%d/%d  Pours=%d  Target=%d"
            % (
                self._budget,
                self._budget_max,
                self._total_pours,
                spec.target_level if spec else 0,
            ),
            "Lives: %d / %d" % (self._lives, self._lives_max),
            "Cursor: Beaker %d" % self._cursor,
            "Beaker levels: %s" % beaker_levels,
        ]

        return {
            "total_pours": self._total_pours,
            "par_pours": spec.par_pours if spec else 0,
            "cursor": self._cursor,
            "selected_source": self._selected_source,
            "lives": self._lives,
            "lives_max": self._lives_max,
            "budget_remaining": self._budget,
            "budget_max": self._budget_max,
            "budget_used": self._budget_max - self._budget,
            "beaker_levels": [len(l) for l in self._layers],
            "target_level": spec.target_level if spec else 0,
            "all_balanced": balanced,
            "circuit_title": "FLUIDBALANCE -- Level %d/%d" % (idx + 1, N_LEVELS),
            "level_title": spec.name if spec else "",
            "has_reactions": bool(spec.reactions) if spec else False,
            "game_over": self._game_over,
            "level_features": features,
        }


def _render_png(frame: list) -> Optional[bytes]:
    try:
        arr = np.array(frame, dtype=np.uint8)
        rows, cols = arr.shape
        rgb = np.zeros((rows, cols, 3), dtype=np.uint8)
        for idx, color in enumerate(_ARC_PALETTE):
            mask = arr == idx
            rgb[mask] = color
        height, width = rgb.shape[:2]
        raw_rows = []
        for y in range(height):
            raw_rows.append(b"\x00" + rgb[y].tobytes())
        raw_data = b"".join(raw_rows)
        compressed = zlib.compress(raw_data)

        def _chunk(ctype, data):
            c = ctype + data
            return (
                struct.pack(">I", len(data))
                + c
                + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
            )

        sig = b"\x89PNG\r\n\x1a\n"
        ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
        return (
            sig
            + _chunk(b"IHDR", ihdr)
            + _chunk(b"IDAT", compressed)
            + _chunk(b"IEND", b"")
        )
    except Exception:
        return None


class PuzzleEnvironment:
    _ACTION_MAP = {
        "left": GameAction.ACTION3,
        "right": GameAction.ACTION4,
        "select": GameAction.ACTION5,
        "undo": GameAction.ACTION7,
        "reset": GameAction.RESET,
    }
    _VALID_ACTIONS = ["left", "right", "select", "undo", "reset"]

    def __init__(self, seed: int = 0):
        self._engine = Fb12(seed=seed)
        self._turn = 0
        self._last_action_was_reset = False
        self._last_frame = None

    def _build_text_observation(self) -> str:
        g = self._engine
        if g._game_over:
            return "GAME OVER"
        if not g._spec:
            return ""
        spec = g._spec
        nb = spec.num_beakers
        cap = spec.beaker_capacity
        idx = g._current_level_index
        total = N_LEVELS
        lines = [
            f"=== Fluid Balance  Level {min(idx + 1, total)}/{total} — {spec.name} ===",
            f"Budget        : {g._budget}/{g._budget_max}  ({g._budget} remaining)",
            f"Lives         : {g._lives}/{g._lives_max}",
            f"Cursor        : {'Trash' if g._cursor == TRASH_CURSOR_IDX else f'Beaker {g._cursor}'}",
            f"Selected src  : {g._selected_source if g._selected_source >= 0 else 'None'}",
            f"Total pours   : {g._total_pours}",
            "",
        ]
        header = " ".join(f"B{i:<2d}" for i in range(nb))
        lines.append(header)
        for row_idx in range(cap - 1, -1, -1):
            row_chars = []
            for bi in range(nb):
                if row_idx < len(g._layers[bi]):
                    color = g._layers[bi][row_idx]
                    ch = " " + FLUID_CHARS.get(color, "?")
                else:
                    ch = " ."
                row_chars.append(ch)
            tgt = spec.target_level
            if spec.target_levels and len(spec.target_levels) > 0:
                tgt = 0
            tgt_marker = "<" if row_idx == tgt else " "
            lines.append(" ".join(row_chars) + " " + tgt_marker)
        cursor_line = " ".join(" ^" if i == g._cursor else "  " for i in range(nb))
        lines.append(cursor_line)
        if g._selected_source >= 0:
            sel_line = " ".join(
                " S" if i == g._selected_source else "  " for i in range(nb)
            )
            lines.append(sel_line)
        if g._flash_win:
            lines.append("\n*** BALANCED — LEVEL COMPLETE! ***")
        return "\n".join(lines)

    def _build_meta(self) -> dict:
        g = self._engine
        idx = g._current_level_index
        total = N_LEVELS
        spec = g._spec
        return {
            "level": min(idx + 1, total),
            "total_levels": total,
            "level_index": min(idx, total - 1),
            "levels_completed": idx,
            "done": g._game_over or g._won or (idx >= total and not g._game_over),
            "info": "",
            "game_over": g._game_over,
            "won": g._won,
            "budget_remaining": g._budget,
            "budget_max": g._budget_max,
            "lives": g._lives,
            "lives_max": g._lives_max,
            "cursor": g._cursor,
            "selected_source": g._selected_source,
            "total_pours": g._total_pours,
            "beaker_levels": [len(l) for l in g._layers] if g._layers else [],
            "target_level": spec.target_level if spec else 0,
            "has_reactions": bool(spec.reactions) if spec else False,
        }

    def _make_state(self, fd) -> GameState:
        img = None
        if fd and not fd.is_empty():
            f = fd.frame[-1]
            self._last_frame = f
            img = _render_png(f)
        valid = (
            None
            if (self._engine._game_over or self._engine._won)
            else list(self._VALID_ACTIONS)
        )
        return GameState(
            text_observation=self._build_text_observation(),
            image_observation=img,
            valid_actions=valid,
            turn=self._turn,
            metadata=self._build_meta(),
        )

    def reset(self) -> GameState:
        self._turn = 0
        self._last_action_was_reset = False
        fd = self._engine.perform_action(ActionInput(id=GameAction.RESET))
        return self._make_state(fd)

    def get_actions(self) -> List[str]:
        if self._engine._game_over or self._engine._won:
            return ["reset"]
        return list(self._VALID_ACTIONS)

    def is_done(self) -> bool:
        return self._engine._game_over or self._engine._won

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        frame = self._engine._make_frame()
        arr = np.array(frame, dtype=np.uint8)
        rgb = np.zeros((*arr.shape, 3), dtype=np.uint8)
        for idx, color in enumerate(_ARC_PALETTE):
            mask = arr == idx
            rgb[mask] = color
        return rgb

    def close(self) -> None:
        self._engine = None

    def step(self, action: str) -> StepResult:
        action = action.strip().lower()
        ga = self._ACTION_MAP.get(action)
        if ga is None:
            return StepResult(
                state=self._make_state(None),
                reward=0.0,
                done=self._engine._game_over or self._engine._won,
                info={"action": action, "invalid": True},
            )

        if action == "reset":
            if self._last_action_was_reset:
                self._engine.full_reset()
                self._last_action_was_reset = False
                fd = self._engine.perform_action(ActionInput(id=GameAction.RESET))
                return StepResult(
                    state=self._make_state(fd),
                    reward=0.0,
                    done=False,
                    info={
                        "action": "reset",
                        "level_changed": True,
                        "life_lost": False,
                        "level_completed": False,
                    },
                )
            self._last_action_was_reset = True
        else:
            self._last_action_was_reset = False

        prev_level = self._engine._current_level_index
        prev_lives = self._engine._lives
        prev_scores = len(self._engine._level_scores)

        fd = self._engine.perform_action(ActionInput(id=ga))
        if ga != GameAction.RESET:
            self._turn += 1

        done = self._engine._game_over or self._engine._won
        reward = 0.0
        total_levels = N_LEVELS

        level_advanced = self._engine._current_level_index > prev_level
        life_lost = self._engine._lives < prev_lives
        level_completed = len(self._engine._level_scores) > prev_scores

        if self._engine._won:
            reward = 1.0
        elif self._engine._game_over:
            reward = 0.0
        elif level_advanced:
            reward = 1.0 / total_levels
        elif life_lost:
            reward = 0.0

        return StepResult(
            state=self._make_state(fd),
            reward=reward,
            done=done,
            info={
                "action": action,
                "level_changed": level_advanced,
                "life_lost": life_lost,
                "level_completed": level_completed,
            },
        )


class ArcGameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    ACTION_LIST = ["reset", "left", "right", "select", "undo"]
    OBS_WIDTH = 64
    OBS_HEIGHT = 64

    def __init__(self, render_mode: Optional[str] = None, seed: int = 0):
        super().__init__()
        self.render_mode = render_mode
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode: {render_mode}")
        self._env = None
        self._seed = seed
        self._action_to_string = {i: a for i, a in enumerate(self.ACTION_LIST)}
        self._string_to_action = {a: i for i, a in enumerate(self.ACTION_LIST)}
        self.action_space = spaces.Discrete(len(self.ACTION_LIST))
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.OBS_HEIGHT, self.OBS_WIDTH, 3),
            dtype=np.uint8,
        )

    @staticmethod
    def _resize_nearest(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        h, w = img.shape[:2]
        row_idx = (np.arange(target_h) * h) // target_h
        col_idx = (np.arange(target_w) * w) // target_w
        return img[row_idx[:, None], col_idx[None, :]]

    def _get_obs(self) -> np.ndarray:
        frame = self._env._engine._make_frame()
        arr = np.array(frame, dtype=np.uint8)
        rgb = np.zeros((*arr.shape, 3), dtype=np.uint8)
        for idx, color in enumerate(_ARC_PALETTE):
            mask = arr == idx
            rgb[mask] = color
        return self._resize_nearest(rgb, self.OBS_HEIGHT, self.OBS_WIDTH)

    def _build_info(self) -> Dict[str, Any]:
        meta = self._env._build_meta()
        return {
            "text_observation": self._env._build_text_observation(),
            "valid_actions": self._env.get_actions(),
            "turn": self._env._turn,
            "game_metadata": meta,
            "step_info": {},
        }

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ):
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed
        self._env = PuzzleEnvironment(seed=self._seed)
        self._env.reset()
        return self._get_obs(), self._build_info()

    def step(self, action: int):
        action_str = self._action_to_string[action]
        result = self._env.step(action_str)
        obs = self._get_obs()
        terminated = result.done
        truncated = False
        return obs, result.reward, terminated, truncated, self._build_info()

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode == "rgb_array":
            return self._get_obs()
        return None

    def action_mask(self) -> np.ndarray:
        valid = self._env.get_actions()
        mask = np.zeros(len(self.ACTION_LIST), dtype=np.int8)
        for i, a in enumerate(self.ACTION_LIST):
            if a in valid:
                mask[i] = 1
        return mask

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None