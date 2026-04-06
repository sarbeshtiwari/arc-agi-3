from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
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
    Level,
    RenderableUserDisplay,
    Sprite,
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


C_EMPTY = 0
C_PLAYER = 1
C_CRATE = 2
C_TARGET = 3
C_CRATE_ON_TGT = 4
C_PIT = 5
C_FRAGILE = 6
C_FRAGILE_CRACK = 7
C_WALL = 8
C_POISON_TRIGGER = 9
C_POISON_SPREAD = 10
C_CRATE_IN_PIT = 11
C_ICE = 12
C_EXIT_LOCKED = 13
C_EXIT_OPEN = 14
C_MOB_POISON = 15
C_TELEPORT_DECOY = 16

_SOLID = {C_WALL, C_CRATE, C_CRATE_ON_TGT, C_CRATE_IN_PIT, C_EXIT_LOCKED}

_GUARD_BLOCKED = {
    C_WALL,
    C_CRATE,
    C_CRATE_ON_TGT,
    C_CRATE_IN_PIT,
    C_EXIT_LOCKED,
    C_EXIT_OPEN,
    C_FRAGILE,
    C_FRAGILE_CRACK,
    C_ICE,
    C_PIT,
    C_MOB_POISON,
}

BACKGROUND_COLOR = 0
PADDING_COLOR = 8
GRID_MAX = 16
MAX_LIVES = 3
CLR_LIFE = 14
CLR_LIFE_LOST = 5
CLR_BAR_FULL = 10
CLR_BAR_LOW = 12

CELL = 4

_CELL_CHAR = {
    C_EMPTY: ".",
    C_PLAYER: "@",
    C_CRATE: "B",
    C_TARGET: "T",
    C_CRATE_ON_TGT: "*",
    C_PIT: "V",
    C_FRAGILE: "~",
    C_FRAGILE_CRACK: "!",
    C_WALL: "#",
    C_POISON_TRIGGER: "P",
    C_POISON_SPREAD: "X",
    C_CRATE_IN_PIT: "=",
    C_ICE: "I",
    C_EXIT_LOCKED: "L",
    C_EXIT_OPEN: "E",
    C_MOB_POISON: "M",
    C_TELEPORT_DECOY: "D",
}

_L1_PATROLS = [
    {"patrol": [(3, 7), (4, 7), (5, 7), (5, 8), (4, 8), (3, 8)], "start": 0},
]

_L2_PATROLS = [
    {"patrol": [(1, 9), (2, 9), (3, 9), (3, 10), (2, 10), (1, 10)], "start": 0},
    {"patrol": [(5, 9), (4, 9), (3, 9), (3, 10), (4, 10), (5, 10)], "start": 1},
]

_L3_PATROLS = [
    {"patrol": [(1, 11), (2, 11), (3, 11), (2, 11)], "start": 0},
    {"patrol": [(1, 12), (2, 12), (3, 12), (2, 12)], "start": 3},
    {"patrol": [(3, 11), (4, 11), (4, 12), (3, 12)], "start": 0},
]

_L4_PATROLS = [
    {"patrol": [(2, 11), (3, 11), (4, 11), (4, 12), (3, 12), (2, 12)], "start": 0},
    {"patrol": [(6, 11), (7, 11), (8, 11), (8, 12), (7, 12), (6, 12)], "start": 1},
    {"patrol": [(2, 12), (2, 13), (3, 13), (3, 12)], "start": 0},
    {"patrol": [(6, 12), (6, 13), (7, 13), (7, 12)], "start": 2},
]

_L5_PATROLS = [
    {"patrol": [(2, 13), (3, 13), (3, 12), (2, 12)], "start": 0},
    {"patrol": [(5, 13), (6, 13), (7, 13), (7, 12), (6, 12), (5, 12)], "start": 1},
    {"patrol": [(3, 15), (4, 15), (5, 15), (4, 15)], "start": 0},
    {"patrol": [(7, 15), (8, 15), (9, 15), (8, 15)], "start": 2},
    {"patrol": [(9, 13), (10, 13), (11, 13), (11, 12), (10, 12), (9, 12)], "start": 3},
]

LEVEL1_GRID = [
    [8, 8, 8, 8, 8, 8, 8, 8, 8],
    [8, 1, 0, 0, 0, 0, 16, 0, 8],
    [8, 0, 0, 2, 0, 2, 0, 0, 8],
    [8, 0, 0, 0, 0, 0, 0, 0, 8],
    [8, 0, 0, 8, 0, 8, 0, 0, 8],
    [8, 0, 6, 0, 0, 0, 6, 0, 8],
    [8, 0, 0, 0, 9, 0, 0, 0, 8],
    [8, 0, 0, 0, 0, 0, 0, 0, 8],
    [8, 0, 3, 0, 0, 0, 3, 0, 8],
    [8, 8, 8, 8, 13, 8, 8, 8, 8],
]
_L1_TELEPORT = {(6, 1): (7, 7)}

LEVEL2_GRID = [
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [8, 1, 0, 0, 0, 0, 0, 0, 16, 0, 8],
    [8, 0, 0, 2, 0, 2, 0, 2, 0, 0, 8],
    [8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8],
    [8, 0, 0, 8, 0, 8, 0, 8, 0, 0, 8],
    [8, 0, 6, 0, 6, 6, 0, 0, 6, 0, 8],
    [8, 12, 12, 12, 12, 12, 12, 12, 12, 12, 8],
    [8, 0, 0, 9, 0, 9, 0, 9, 0, 0, 8],
    [8, 0, 0, 0, 6, 0, 6, 0, 0, 0, 8],
    [8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8],
    [8, 0, 3, 0, 0, 0, 3, 0, 3, 0, 8],
    [8, 8, 8, 13, 8, 8, 8, 8, 8, 8, 8],
]
_L2_TELEPORT = {(8, 1): (9, 9)}

LEVEL3_GRID = [
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [8, 1, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 8],
    [8, 0, 0, 2, 0, 2, 0, 2, 0, 2, 0, 0, 8],
    [8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8],
    [8, 0, 0, 8, 0, 8, 0, 0, 0, 8, 0, 0, 8],
    [8, 0, 6, 0, 0, 0, 0, 6, 0, 0, 6, 0, 8],
    [8, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 8],
    [8, 0, 0, 9, 0, 0, 0, 0, 9, 0, 9, 0, 8],
    [8, 0, 0, 0, 0, 6, 0, 6, 0, 6, 0, 6, 8],
    [8, 0, 0, 0, 0, 0, 9, 0, 9, 0, 9, 0, 8],
    [8, 0, 6, 0, 0, 6, 0, 0, 6, 0, 0, 6, 8],
    [8, 0, 0, 0, 0, 6, 9, 6, 0, 9, 0, 0, 8],
    [8, 0, 3, 0, 0, 0, 3, 3, 0, 0, 3, 0, 8],
    [8, 8, 8, 13, 8, 8, 8, 8, 8, 8, 8, 8, 8],
]
_L3_TELEPORT = {(9, 1): (11, 11)}

LEVEL4_GRID = [
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [8, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 8],
    [8, 0, 0, 2, 0, 2, 0, 2, 0, 2, 0, 0, 0, 0, 8],
    [8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8],
    [8, 0, 0, 8, 0, 8, 0, 8, 0, 8, 0, 0, 0, 0, 8],
    [8, 0, 6, 0, 0, 0, 6, 0, 0, 0, 6, 0, 6, 0, 8],
    [8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8],
    [8, 0, 0, 0, 9, 0, 9, 0, 9, 0, 0, 0, 0, 0, 8],
    [8, 0, 6, 0, 0, 0, 6, 0, 0, 0, 6, 0, 6, 0, 8],
    [8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8],
    [8, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 8],
    [8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8],
    [8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8],
    [8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8],
    [8, 0, 3, 0, 3, 0, 3, 0, 3, 0, 0, 0, 0, 0, 8],
    [8, 13, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
]
_L4_TELEPORT = {(11, 1): (13, 11)}

LEVEL5_GRID = [
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [8, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 8],
    [8, 0, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 0, 8],
    [8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8],
    [8, 0, 0, 8, 0, 8, 0, 8, 0, 8, 0, 8, 0, 0, 8],
    [8, 0, 6, 0, 0, 0, 6, 0, 0, 0, 6, 0, 6, 0, 8],
    [8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8],
    [8, 0, 0, 0, 9, 0, 9, 0, 9, 0, 9, 0, 0, 0, 8],
    [8, 0, 6, 0, 0, 0, 6, 0, 0, 0, 6, 0, 6, 0, 8],
    [8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8],
    [8, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 8],
    [8, 0, 6, 0, 6, 0, 6, 0, 6, 0, 6, 0, 6, 0, 8],
    [8, 0, 0, 0, 9, 0, 0, 0, 9, 0, 0, 0, 9, 0, 8],
    [8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8],
    [8, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 8],
    [8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8],
    [8, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 0, 0, 8],
    [8, 13, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
]
_L5_TELEPORT = {(13, 1): (13, 15)}

LEVELS_CONFIG = {
    1: {
        "grid": LEVEL1_GRID,
        "move_limit": 21,
        "par_moves": 19,
        "gw": 9,
        "gh": 10,
        "patrols": _L1_PATROLS,
        "teleports": _L1_TELEPORT,
    },
    2: {
        "grid": LEVEL2_GRID,
        "move_limit": 27,
        "par_moves": 25,
        "gw": 11,
        "gh": 12,
        "patrols": _L2_PATROLS,
        "teleports": _L2_TELEPORT,
    },
    3: {
        "grid": LEVEL4_GRID,
        "move_limit": 35,
        "par_moves": 31,
        "gw": 15,
        "gh": 16,
        "cell": 4,
        "patrols": _L4_PATROLS,
        "teleports": _L4_TELEPORT,
    },
    4: {
        "grid": LEVEL5_GRID,
        "move_limit": 40,
        "par_moves": 36,
        "gw": 15,
        "gh": 18,
        "cell": 3,
        "patrols": _L5_PATROLS,
        "teleports": _L5_TELEPORT,
    },
}


class HUD(RenderableUserDisplay):
    def __init__(self, game: "Dr01") -> None:
        self.game = game

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        g = self.game
        fh, fw = frame.shape
        scale = min(fw // GRID_MAX, fh // GRID_MAX)
        if scale < 1:
            return frame

        x_off = (fw - GRID_MAX * scale) // 2
        y_off = (fh - GRID_MAX * scale) // 2
        sq = max(2, scale - 2)
        gap = max(1, scale // 4)

        for i in range(MAX_LIVES):
            x0 = x_off + gap + i * (sq + gap)
            y0 = y_off + gap
            x1 = min(fw, x0 + sq)
            y1 = min(fh, y0 + sq)
            if x0 < x1 and y0 < y1:
                frame[y0:y1, x0:x1] = CLR_LIFE if i < g.lives else CLR_LIFE_LOST

        bar_total = g.move_limit
        if bar_total > 0:
            ratio = max(0.0, (bar_total - g.move_count) / bar_total)
            bar_h = max(2, scale // 3)
            bar_y0 = max(0, fh - y_off - bar_h)
            bar_y1 = max(0, fh - y_off)
            bar_w = fw - 2 * x_off
            filled = int(ratio * bar_w)
            colour = CLR_BAR_FULL if ratio > 0.25 else CLR_BAR_LOW
            if bar_y0 < bar_y1 and bar_w > 0:
                frame[bar_y0:bar_y1, x_off : x_off + filled] = colour
                frame[bar_y0:bar_y1, x_off + filled : x_off + bar_w] = BACKGROUND_COLOR

        return frame


def _solid(color: int, size: int) -> list:
    return [[color] * size for _ in range(size)]


def _make_levels() -> list:
    return [
        Level(
            sprites=[],
            grid_size=(
                LEVELS_CONFIG[lv]["gw"] * LEVELS_CONFIG[lv].get("cell", CELL),
                LEVELS_CONFIG[lv]["gh"] * LEVELS_CONFIG[lv].get("cell", CELL),
            ),
            name="Level {}".format(lv),
        )
        for lv in range(1, 5)
    ]


_levels = _make_levels()


class Dr01(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)
        self.grid = []
        self.gw = 0
        self.gh = 0
        self.cell = CELL
        self.player = (0, 0)
        self.move_count = 0
        self.move_limit = 21
        self.bar_limit = 76
        self.lives = MAX_LIVES
        self._fragile_under = {}
        self._entered_exit = False
        self._hud = HUD(self)

        self._guards: list = []
        self._teleports: dict = {}
        self._teleported: bool = False

        self._undo_stack: list = []
        self._last_was_reset = False
        self._game_over = False
        self._game_won = False

        super().__init__(
            "dr01",
            _levels,
            Camera(
                0,
                0,
                GRID_MAX,
                GRID_MAX,
                BACKGROUND_COLOR,
                PADDING_COLOR,
                [self._hud],
            ),
            available_actions=[0, 1, 2, 3, 4, 7],
        )

    def on_set_level(self, level) -> None:
        self._fragile_under = {}
        self._entered_exit = False
        self._game_over = False
        self._game_won = False
        self._last_was_reset = False
        self._undo_stack = []
        lv_num = self.level_index + 1
        cfg = LEVELS_CONFIG.get(lv_num, LEVELS_CONFIG[1])
        self.gw = cfg["gw"]
        self.gh = cfg["gh"]
        self.cell = cfg.get("cell", CELL)
        self.move_limit = cfg["move_limit"]
        self.bar_limit = cfg["par_moves"] * 4
        self.lives = MAX_LIVES
        self._load_grid(cfg)
        self._init_guards(cfg)
        self._teleports = dict(cfg.get("teleports", {}))
        self.rebuild(level)

    def _load_grid(self, cfg: dict) -> None:
        self.grid = [row[:] for row in cfg["grid"]]
        self.move_count = 0
        self.player = (0, 0)
        for r in range(self.gh):
            for c in range(self.gw):
                if self.grid[r][c] == C_PLAYER:
                    self.player = (c, r)

    def _init_guards(self, cfg: dict) -> None:
        self._guards = []
        for pd in cfg.get("patrols", []):
            patrol = pd["patrol"]
            start_idx = pd["start"]
            if not patrol:
                continue
            cx, cy = patrol[start_idx]
            underneath = self.grid[cy][cx]
            guard = {"patrol": patrol, "idx": start_idx, "underneath": underneath}
            self._guards.append(guard)
            if underneath not in _GUARD_BLOCKED:
                self.grid[cy][cx] = C_MOB_POISON

    def rebuild(self, level) -> None:
        level.remove_all_sprites()
        cell = getattr(self, "cell", CELL)
        idx = 0
        for r in range(self.gh):
            for c in range(self.gw):
                level.add_sprite(
                    Sprite(
                        pixels=_solid(self.grid[r][c], cell),
                        name="t{}".format(idx),
                        x=c * cell,
                        y=r * cell,
                        layer=1,
                    )
                )
                idx += 1

    def _step_guards(self) -> None:
        px, py = self.player

        for g in self._guards:
            patrol = g["patrol"]
            old_idx = g["idx"]
            cx, cy = patrol[old_idx]

            self.grid[cy][cx] = g["underneath"]

            new_idx = (old_idx + 1) % len(patrol)
            nx, ny = patrol[new_idx]

            dest = self.grid[ny][nx]
            if dest in _GUARD_BLOCKED:
                g["idx"] = old_idx
                g["underneath"] = self.grid[cy][cx]
                self.grid[cy][cx] = C_MOB_POISON
            else:
                g["idx"] = new_idx
                g["underneath"] = dest
                self.grid[ny][nx] = C_MOB_POISON

        for g in self._guards:
            gx, gy = g["patrol"][g["idx"]]
            if (gx, gy) == (px, py):
                self._lose_life()
                return

    def _apply_gravity(self) -> None:
        moved = True
        while moved:
            moved = False
            for r in range(self.gh - 2, 0, -1):
                for c in range(1, self.gw - 1):
                    tile = self.grid[r][c]
                    if tile not in (C_CRATE, C_CRATE_ON_TGT):
                        continue
                    below = self.grid[r + 1][c]
                    if below in _SOLID:
                        continue

                    src_restore = C_TARGET if tile == C_CRATE_ON_TGT else C_EMPTY

                    if below == C_PIT:
                        self._vacate(c, r, src_restore)
                        self.grid[r + 1][c] = C_CRATE_IN_PIT
                        moved = True
                    elif below == C_FRAGILE:
                        self._vacate(c, r, src_restore)
                        self.grid[r + 1][c] = C_CRATE
                        self._fragile_under[(c, r + 1)] = C_FRAGILE_CRACK
                        moved = True
                    elif below == C_FRAGILE_CRACK:
                        self._vacate(c, r, src_restore)
                        self.grid[r + 1][c] = C_CRATE_IN_PIT
                        self._fragile_under.pop((c, r + 1), None)
                        moved = True
                    elif below == C_TARGET:
                        self._vacate(c, r, src_restore)
                        self.grid[r + 1][c] = C_CRATE_ON_TGT
                        moved = True
                    elif below not in _SOLID:
                        self._vacate(c, r, src_restore)
                        self.grid[r + 1][c] = (
                            C_CRATE_ON_TGT if below == C_TARGET else C_CRATE
                        )
                        moved = True

    def _vacate(self, c: int, r: int, default: int) -> None:
        restore = self._fragile_under.pop((c, r), None)
        self.grid[r][c] = restore if restore is not None else default

    def _unlock_exit(self) -> None:
        for r in range(self.gh):
            for c in range(self.gw):
                if self.grid[r][c] == C_EXIT_LOCKED:
                    self.grid[r][c] = C_EXIT_OPEN

    def _all_placed(self) -> bool:
        for r in range(self.gh):
            for c in range(self.gw):
                if self.grid[r][c] == C_CRATE:
                    return False
        return True

    def _lose_life(self) -> None:
        self.lives -= 1
        self._undo_stack = []
        if self.lives > 0:
            self._reset_level_state()
        else:
            self._game_over = True
            self.lose()

    def _go_to_level_1(self) -> None:
        self._game_over = False
        self._game_won = False
        self._last_was_reset = False
        self._undo_stack = []
        self._level_idx = 0
        self.current_level = self._levels[0]
        self.on_set_level(self.current_level)

    def on_reset(self) -> None:
        is_consecutive = self._last_was_reset
        self._last_was_reset = True

        if self._game_won:
            self._go_to_level_1()
            return

        if self._game_over:
            if is_consecutive:
                self._go_to_level_1()
            else:
                self._game_over = False
                self._undo_stack = []
                self.current_level = self._levels[self._level_idx]
                self.on_set_level(self.current_level)
            return

        if self.move_count == 0:
            self._go_to_level_1()
        elif is_consecutive:
            self._go_to_level_1()
        else:
            self._lose_life()

    def _reset_level_state(self) -> None:
        lv_num = self.level_index + 1
        cfg = LEVELS_CONFIG.get(lv_num, LEVELS_CONFIG[1])
        self.gw = cfg["gw"]
        self.gh = cfg["gh"]
        self.cell = cfg.get("cell", CELL)
        self.move_limit = cfg["move_limit"]
        self.bar_limit = cfg["par_moves"] * 4
        self._fragile_under = {}
        self._entered_exit = False
        self._undo_stack = []
        self._load_grid(cfg)
        self._init_guards(cfg)
        self._teleports = dict(cfg.get("teleports", {}))
        self.rebuild(self.current_level)

    def _save_state(self) -> None:
        self._undo_stack.append(
            {
                "grid": [row[:] for row in self.grid],
                "player": self.player,
                "fragile_under": dict(self._fragile_under),
                "guards": [
                    {
                        "patrol": g["patrol"],
                        "idx": g["idx"],
                        "underneath": g["underneath"],
                    }
                    for g in self._guards
                ],
                "teleports": dict(self._teleports),
            }
        )

    def _do_undo(self) -> None:
        if self._undo_stack:
            state = self._undo_stack.pop()
            self.grid = state["grid"]
            self.player = state["player"]
            self._fragile_under = state["fragile_under"]
            self._guards = state["guards"]
            self._teleports = state["teleports"]
            self._entered_exit = False
        self.move_count += 1
        if self.move_count >= self.move_limit:
            self._lose_life()
        else:
            self.rebuild(self.current_level)

    def _ice_slide(self, sx: int, sy: int, dx: int, dy: int) -> tuple:
        cx, cy = sx, sy
        while True:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < self.gw and 0 <= ny < self.gh):
                break
            t = self.grid[ny][nx]
            if t in _SOLID:
                break
            cx, cy = nx, ny
            if t != C_ICE:
                break
        return (cx, cy)

    def _apply_teleport(self, tx: int, ty: int) -> None:
        dest = self._teleports.get((tx, ty))
        if dest is None:
            return
        dx, dy = dest
        px, py = self.player

        if self.grid[py][px] == C_PLAYER:
            restore = self._fragile_under.pop((px, py), None)
            self.grid[py][px] = restore if restore is not None else C_EMPTY
        self.grid[ty][tx] = C_TELEPORT_DECOY

        dest_tile = self.grid[dy][dx]
        self.grid[dy][dx] = C_PLAYER
        self.player = (dx, dy)

        if dest_tile in (C_PIT, C_POISON_SPREAD, C_MOB_POISON):
            self._lose_life()
        elif dest_tile == C_FRAGILE:
            self._fragile_under[(dx, dy)] = C_FRAGILE_CRACK
        elif dest_tile == C_FRAGILE_CRACK:
            self._fragile_under[(dx, dy)] = C_PIT

    def _try_move(self, dx: int, dy: int) -> bool:
        self._entered_exit = False
        self._teleported = False
        px, py = self.player
        tx, ty = px + dx, py + dy

        if not (0 <= tx < self.gw and 0 <= ty < self.gh):
            return False

        dest = self.grid[ty][tx]

        if dest == C_WALL or dest == C_EXIT_LOCKED:
            return False

        if dest == C_EXIT_OPEN:
            self._entered_exit = True
            self._move_player(px, py, tx, ty)
            return True

        if dest == C_TELEPORT_DECOY:
            self._move_player(px, py, tx, ty)
            self._apply_teleport(tx, ty)
            self._teleported = True
            return True

        if dest == C_PIT:
            self._move_player(px, py, tx, ty)
            self._lose_life()
            return True

        if dest in (C_POISON_SPREAD, C_MOB_POISON):
            self._move_player(px, py, tx, ty)
            self._lose_life()
            return True

        if dest == C_FRAGILE:
            self._move_player(px, py, tx, ty)
            self._fragile_under[(tx, ty)] = C_FRAGILE_CRACK
            return True

        if dest == C_FRAGILE_CRACK:
            self._move_player(px, py, tx, ty)
            self._fragile_under[(tx, ty)] = C_PIT
            return True

        if dest == C_POISON_TRIGGER:
            self._move_player(px, py, tx, ty)
            self._activate_poison(tx, ty)
            return True

        if dest == C_ICE:
            fx, fy = self._ice_slide(px, py, dx, dy)
            if (fx, fy) == (px, py):
                return False
            ft = self.grid[fy][fx]
            self._move_player(px, py, fx, fy)
            if ft == C_EXIT_OPEN:
                self._entered_exit = True
            elif ft in (C_PIT, C_POISON_SPREAD, C_MOB_POISON):
                self._lose_life()
            elif ft == C_POISON_TRIGGER:
                self._activate_poison(fx, fy)
            elif ft == C_FRAGILE:
                self._fragile_under[(fx, fy)] = C_FRAGILE_CRACK
            elif ft == C_FRAGILE_CRACK:
                self._fragile_under[(fx, fy)] = C_PIT
            elif ft == C_TELEPORT_DECOY:
                self._apply_teleport(fx, fy)
                self._teleported = True
            return True

        if dest in (C_CRATE, C_CRATE_ON_TGT):
            cx2, cy2 = tx + dx, ty + dy
            if not (0 <= cx2 < self.gw and 0 <= cy2 < self.gh):
                return False
            cd = self.grid[cy2][cx2]
            if cd in _SOLID:
                return False

            restore = self._fragile_under.pop((tx, ty), None)
            vacated = (
                restore
                if restore is not None
                else (C_TARGET if dest == C_CRATE_ON_TGT else C_EMPTY)
            )

            if cd == C_PIT:
                self.grid[cy2][cx2] = C_CRATE_IN_PIT
            elif cd == C_FRAGILE:
                self.grid[cy2][cx2] = C_CRATE
                self._fragile_under[(cx2, cy2)] = C_FRAGILE_CRACK
            elif cd == C_FRAGILE_CRACK:
                self.grid[cy2][cx2] = C_CRATE_IN_PIT
                self._fragile_under.pop((cx2, cy2), None)
            elif cd == C_TARGET:
                self.grid[cy2][cx2] = C_CRATE_ON_TGT
            else:
                self.grid[cy2][cx2] = C_CRATE

            self.grid[ty][tx] = vacated
            self._move_player(px, py, tx, ty)
            return True

        self._move_player(px, py, tx, ty)
        return True

    def _move_player(self, px: int, py: int, tx: int, ty: int) -> None:
        if self.grid[py][px] == C_PLAYER:
            restore = self._fragile_under.pop((px, py), None)
            self.grid[py][px] = restore if restore is not None else C_EMPTY
        self.grid[ty][tx] = C_PLAYER
        self.player = (tx, ty)

    def _activate_poison(self, tx: int, ty: int) -> None:
        for dc, dr in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nc, nr = tx + dc, ty + dr
            if 0 <= nc < self.gw and 0 <= nr < self.gh and self.grid[nr][nc] == C_EMPTY:
                self.grid[nr][nc] = C_POISON_SPREAD

    def step(self) -> None:
        self._last_was_reset = False

        if self._game_over or self._game_won:
            self.complete_action()
            return

        if self.action.id == GameAction.RESET:
            self.complete_action()
            return

        if self.action.id == GameAction.ACTION7:
            self._do_undo()
            self.complete_action()
            return

        dx, dy = 0, 0
        if self.action.id == GameAction.ACTION1:
            dy = -1
        elif self.action.id == GameAction.ACTION2:
            dy = 1
        elif self.action.id == GameAction.ACTION3:
            dx = -1
        elif self.action.id == GameAction.ACTION4:
            dx = 1
        else:
            self.complete_action()
            return

        self._save_state()
        self._teleported = False
        moved = self._try_move(dx, dy)

        if moved:
            self.move_count += 1
            self._apply_gravity()

            if self._all_placed():
                self._unlock_exit()

            self._step_guards()

            self.rebuild(self.current_level)

            if self._entered_exit:
                is_last = self.level_index >= len(self._levels) - 1
                self.next_level()
                if is_last:
                    self._game_won = True
                self.complete_action()
                return

            if self.move_count >= self.move_limit:
                self._lose_life()
        else:
            self._undo_stack.pop()

        self.complete_action()


Game = Dr01
DryRun = Dr01


class PuzzleEnvironment:
    _ACTION_MAP: dict = {
        "up": GameAction.ACTION1,
        "down": GameAction.ACTION2,
        "left": GameAction.ACTION3,
        "right": GameAction.ACTION4,
        "undo": GameAction.ACTION7,
        "reset": GameAction.RESET,
    }

    _VALID_ACTIONS = ["reset", "up", "down", "left", "right", "undo"]

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
        (163, 86, 214),
    ]

    def __init__(self, seed: int = 0) -> None:
        self._engine = Dr01(seed=seed)
        self._total_turns = 0
        self._last_action_was_reset = False
        self._game_won = False

    def _build_text_obs(self) -> str:
        e = self._engine
        lines: List[str] = []
        for r in range(e.gh):
            row_chars: List[str] = []
            for c in range(e.gw):
                row_chars.append(_CELL_CHAR.get(e.grid[r][c], "?"))
            lines.append("".join(row_chars))
        grid_text = "\n".join(lines)
        remaining = max(0, e.move_limit - e.move_count)
        header = "Level:{}/{} Lives:{} Moves:{}/{}".format(
            e.level_index + 1,
            len(e._levels),
            e.lives,
            remaining,
            e.move_limit,
        )
        return header + "\n" + grid_text

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
        valid_actions = self.get_actions() if not done else None
        return GameState(
            text_observation=self._build_text_obs(),
            image_observation=self._build_image_bytes(),
            valid_actions=valid_actions,
            turn=self._total_turns,
            metadata={
                "total_levels": len(e._levels),
                "level_index": e.level_index,
                "levels_completed": e.level_index,
                "lives": e.lives,
                "game_over": e._game_over,
                "done": done,
                "info": info or {},
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
        return self._build_game_state()

    def get_actions(self) -> List[str]:
        if self._engine._game_over:
            return ["reset"]
        return list(self._VALID_ACTIONS)

    def is_done(self) -> bool:
        return self._engine._game_over or self._game_won

    def step(self, action: str) -> StepResult:
        e = self._engine

        if action not in self._ACTION_MAP:
            raise ValueError(
                "Invalid action '{}'. Must be one of {}".format(
                    action, list(self._ACTION_MAP.keys())
                )
            )

        game_action = self._ACTION_MAP[action]

        if game_action == GameAction.RESET:
            state = self.reset()
            return StepResult(
                state=state, reward=0.0, done=False, info={"action": "reset"}
            )

        self._last_action_was_reset = False
        self._total_turns += 1

        info: Dict = {"action": action}
        total_levels = len(e._levels)
        level_before = e.level_index

        action_input = ActionInput(id=game_action)
        frame = e.perform_action(action_input, raw=True)

        game_won = frame and frame.state and frame.state.name == "WIN"
        level_advanced = e.level_index > level_before
        done = e._game_over or game_won

        reward = 0.0
        if game_won or level_advanced:
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
            reward=reward,
            done=False,
            info=info,
        )

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError("Unsupported render mode: {}".format(mode))
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
        "render_fps": 1,
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
                "Unsupported render_mode '{}'. Supported: {}".format(
                    render_mode, self.metadata["render_modes"]
                )
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


if __name__ == "__main__":
    from gymnasium.utils.env_checker import check_env

    env = ArcGameEnv(seed=0, render_mode="rgb_array")
    check_env(env, skip_render_check=False)
    obs, info = env.reset()
    obs, reward, term, trunc, info = env.step(0)
    env.render()
    env.close()
