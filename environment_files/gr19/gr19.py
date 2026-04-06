from __future__ import annotations

import copy
import io
import random
import struct
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import gymnasium as gym
from gymnasium import spaces

from arcengine import (
    ActionInput,
    ARCBaseGame,
    Camera,
    GameAction,
    Level,
    Sprite,
)


CELL = 5
GRID_W = 12
GRID_H = 11
CAM_W = GRID_W * CELL
CAM_H = GRID_H * CELL + 9

WHITE = 0
OFFWHITE = 1
LIGHT = 2
GRAY = 3
BLACK = 5
MAGENTA = 6
RED = 8
BLUE = 9
YELLOW = 11
ORANGE = 12
GREEN = 14
CYAN = 15

EMPTY = 0
WIRE_H = 1
WIRE_V = 2
WIRE_CROSS = 3
SOURCE = 4
TARGET = 5
BLOCKER = 6
PINK = 7

LEVEL_SPECS = [
    {
        "name": "Level 1",
        "max_moves": 200,
        "sources": [
            (1, 1, 1),
            (10, 1, 1),
            (1, 9, 1),
            (10, 9, 1),
        ],
        "targets": [
            (5, 3, 1),
            (3, 6, 1),
            (8, 6, 1),
            (6, 8, 1),
        ],
        "blockers": [
            (5, 5),
            (6, 5),
        ],
        "mechanic": "basic",
        "description": "Connect all sources to targets - find the path!",
    },
    {
        "name": "Level 2",
        "max_moves": 200,
        "mechanic": "multi_stage",
        "description": "H->Yellow, V->Cyan, H->Purple->Pink=WIN! Complex maze!",
        "blue_targets": [
            (3, 2),
            (8, 2),
            (2, 5),
            (5, 5),
            (9, 5),
            (3, 8),
            (6, 8),
            (9, 8),
        ],
        "green_sources": [
            (1, 2),
            (3, 0),
            (8, 0),
            (0, 5),
            (11, 5),
            (0, 8),
            (11, 8),
        ],
        "interconnected_targets": {
            (3, 2): [(8, 2)],
            (2, 5): [(5, 5)],
            (3, 8): [(6, 8)],
        },
        "blockers": [
            (2, 1),
            (4, 1),
            (5, 1),
            (6, 1),
            (7, 1),
            (9, 1),
            (1, 3),
            (2, 3),
            (4, 3),
            (5, 3),
            (6, 3),
            (7, 3),
            (9, 3),
            (10, 3),
            (2, 4),
            (3, 4),
            (4, 4),
            (5, 4),
            (6, 4),
            (7, 4),
            (8, 4),
            (9, 4),
            (1, 6),
            (2, 6),
            (4, 6),
            (5, 6),
            (6, 6),
            (7, 6),
            (9, 6),
            (10, 6),
            (2, 7),
            (4, 7),
            (5, 7),
            (6, 7),
            (7, 7),
            (9, 7),
            (1, 9),
            (2, 9),
            (4, 9),
            (5, 9),
            (7, 9),
            (8, 9),
            (10, 9),
        ],
    },
    {
        "name": "Level 3",
        "max_moves": 200,
        "mechanic": "chain_reaction",
        "description": "H->Yellow, V->Orange, H->Cyan(+Chain), V->Pink=WIN!",
        "blue_targets": [
            (3, 3),
            (8, 3),
            (3, 7),
            (8, 7),
        ],
        "green_sources": [
            (2, 3),
            (3, 1),
            (8, 1),
            (11, 3),
            (3, 9),
            (8, 9),
        ],
        "interconnected_targets": {
            (3, 3): [(8, 3)],
        },
        "chain_targets": {
            (8, 3): [(3, 3)],
        },
        "blockers": [],
    },
    {
        "name": "Level 4",
        "max_moves": 250,
        "mechanic": "connector_chain",
        "description": "Source->Connector=Yellow, Connector->Connector=Cyan, All Cyan=WIN!",
        "green_sources": [
            (1, 1),
            (10, 1),
            (1, 9),
            (10, 9),
        ],
        "connectors": [
            (3, 3),
            (8, 3),
            (3, 7),
            (8, 7),
            (5, 5),
            (6, 5),
        ],
        "blockers": [
            (4, 2),
            (7, 2),
            (2, 5),
            (9, 5),
            (4, 8),
            (7, 8),
        ],
    },
]

LEVELS = [
    Level(sprites=[], grid_size=(CAM_W, CAM_H), name=s["name"], data=s)
    for s in LEVEL_SPECS
]


def _px(h, w, color):
    return np.full((h, w), color, dtype=np.int32)


def _cell_empty():
    p = np.full((CELL, CELL), WHITE, dtype=np.int32)
    p[0, :] = LIGHT
    p[-1, :] = LIGHT
    p[:, 0] = LIGHT
    p[:, -1] = LIGHT
    return p


def _cell_wire_h():
    p = _cell_empty()
    p[2, 1:-1] = GRAY
    p[3, 1:-1] = GRAY
    return p


def _cell_wire_v():
    p = _cell_empty()
    p[1:-1, 2] = GRAY
    p[1:-1, 3] = GRAY
    return p


def _cell_wire_cross():
    p = _cell_empty()
    p[2, 1:-1] = GRAY
    p[3, 1:-1] = GRAY
    p[1:-1, 2] = GRAY
    p[1:-1, 3] = GRAY
    p[2, 2] = BLACK
    p[2, 3] = BLACK
    p[3, 2] = BLACK
    p[3, 3] = BLACK
    return p


def _cell_source(polarity):
    color = GREEN if polarity > 0 else RED
    p = np.full((CELL, CELL), color, dtype=np.int32)
    p[0, :] = BLACK
    p[-1, :] = BLACK
    p[:, 0] = BLACK
    p[:, -1] = BLACK
    p[2, 2] = WHITE
    p[2, 3] = WHITE
    p[3, 2] = WHITE
    p[3, 3] = WHITE
    return p


def _cell_target(polarity, active=False):
    if active:
        color = YELLOW
    else:
        color = BLUE if polarity > 0 else MAGENTA
    p = np.full((CELL, CELL), WHITE, dtype=np.int32)
    p[0, :] = color
    p[-1, :] = color
    p[:, 0] = color
    p[:, -1] = color
    p[1, 1] = color
    p[1, -2] = color
    p[-2, 1] = color
    p[-2, -2] = color
    return p


def _cell_blocker():
    p = np.full((CELL, CELL), BLACK, dtype=np.int32)
    p[1:-1, 1:-1] = GRAY
    p[2, 2] = BLACK
    p[2, 3] = BLACK
    p[3, 2] = BLACK
    p[3, 3] = BLACK
    return p


def _cell_multiplier():
    p = _cell_empty()
    p[1, 1] = ORANGE
    p[1, -2] = ORANGE
    p[-2, 1] = ORANGE
    p[-2, -2] = ORANGE
    p[2, 2] = ORANGE
    p[2, 3] = ORANGE
    p[3, 2] = ORANGE
    p[3, 3] = ORANGE
    return p


def _cell_splitter():
    p = _cell_empty()
    p[1, 2] = CYAN
    p[1, 3] = CYAN
    p[2, 1] = CYAN
    p[2, 4] = CYAN
    p[3, 1] = CYAN
    p[3, 4] = CYAN
    p[4, 2] = CYAN
    p[4, 3] = CYAN
    return p


def _cell_relay():
    p = _cell_empty()
    p[1, 1] = YELLOW
    p[1, 2] = YELLOW
    p[1, 3] = YELLOW
    p[2, 1] = YELLOW
    p[2, 3] = YELLOW
    p[3, 1] = YELLOW
    p[3, 2] = YELLOW
    p[3, 3] = YELLOW
    p[2, 2] = WHITE
    return p


def _cell_pink():
    p = np.full((CELL, CELL), WHITE, dtype=np.int32)
    p[0, :] = MAGENTA
    p[-1, :] = MAGENTA
    p[:, 0] = MAGENTA
    p[:, -1] = MAGENTA
    p[1, 1] = MAGENTA
    p[1, -2] = MAGENTA
    p[-2, 1] = MAGENTA
    p[-2, -2] = MAGENTA
    return p


def _cell_purple():
    p = np.full((CELL, CELL), WHITE, dtype=np.int32)
    p[0, :] = MAGENTA
    p[-1, :] = MAGENTA
    p[:, 0] = MAGENTA
    p[:, -1] = MAGENTA
    p[1, 1] = BLUE
    p[1, -2] = BLUE
    p[-2, 1] = BLUE
    p[-2, -2] = BLUE
    p[2, 2] = MAGENTA
    p[2, 3] = MAGENTA
    p[3, 2] = MAGENTA
    p[3, 3] = MAGENTA
    return p


def _cell_yellow_complete():
    p = np.full((CELL, CELL), WHITE, dtype=np.int32)
    p[0, :] = YELLOW
    p[-1, :] = YELLOW
    p[:, 0] = YELLOW
    p[:, -1] = YELLOW
    p[1, 1] = YELLOW
    p[1, -2] = YELLOW
    p[-2, 1] = YELLOW
    p[-2, -2] = YELLOW
    return p


def _cell_orange_complete():
    p = np.full((CELL, CELL), WHITE, dtype=np.int32)
    p[0, :] = ORANGE
    p[-1, :] = ORANGE
    p[:, 0] = ORANGE
    p[:, -1] = ORANGE
    p[1, 1] = ORANGE
    p[1, -2] = ORANGE
    p[-2, 1] = ORANGE
    p[-2, -2] = ORANGE
    return p


def _cell_green_complete():
    p = np.full((CELL, CELL), GREEN, dtype=np.int32)
    p[0, :] = BLACK
    p[-1, :] = BLACK
    p[:, 0] = BLACK
    p[:, -1] = BLACK
    p[2, 2] = WHITE
    p[2, 3] = WHITE
    p[3, 2] = WHITE
    p[3, 3] = WHITE
    return p


def _cell_cyan_complete():
    p = np.full((CELL, CELL), CYAN, dtype=np.int32)
    p[0, :] = BLACK
    p[-1, :] = BLACK
    p[:, 0] = BLACK
    p[:, -1] = BLACK
    p[2, 2] = WHITE
    p[2, 3] = WHITE
    p[3, 2] = WHITE
    p[3, 3] = WHITE
    return p


def _cursor():
    p = np.full((CELL, CELL), -1, dtype=np.int32)
    p[0, :] = ORANGE
    p[-1, :] = ORANGE
    p[:, 0] = ORANGE
    p[:, -1] = ORANGE
    p[1, 1] = ORANGE
    p[1, -2] = ORANGE
    p[-2, 1] = ORANGE
    p[-2, -2] = ORANGE
    return p


class Gr19(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._grid = [[EMPTY for _ in range(GRID_W)] for _ in range(GRID_H)]
        self._sources = {}
        self._targets = {}
        self._blockers = set()
        self._multipliers = set()
        self._splitters = set()
        self._relays = {}
        self._decay_rate = 0.15
        self._relay_boost = 0
        self._max_distance = 999
        self._max_energy = 999
        self._cursor_x = 0
        self._cursor_y = 0
        self._moves = 0
        self._max_moves = 35
        self._lifelines = 3
        self._game_over = False
        self._won = False
        self._level_idx = 0
        self._level_name = ""
        self._mechanic = "basic"
        self._description = ""
        self._consecutive_resets = 0
        self._last_reset_level = None
        self._current_phase = 1
        self._phase1_sources = []
        self._phase1_targets = []
        self._phase2_sources = []
        self._phase2_targets = []
        self._phase3_sources = []
        self._phase3_targets = []
        self._completed_phase1 = set()
        self._completed_phase2 = set()
        self._completed_phase3 = set()
        self._pink_cells = set()
        self._purple_cells = set()
        self._yellow_cells = set()
        self._cyan_cells = set()
        self._orange_cells = set()
        self._green_completed_cells = set()
        self._all_blue_targets = set()
        self._all_green_sources = set()
        self._interconnected_targets = {}
        self._chain_targets = {}
        self._target_required_path = {}
        self._used_sources = set()
        self._wire_limit = 999
        self._wire_count = 0
        self._no_cross_zone = set()
        self._connectors = set()
        self._yellow_connectors = set()
        self._cyan_connectors = set()
        self._history = []
        self._rng = random.Random(seed)

        camera = Camera(0, 0, CAM_W, CAM_H, WHITE, WHITE, [])
        super().__init__(
            "gr19",
            LEVELS,
            camera,
            available_actions=[0, 1, 2, 3, 4, 5, 7],
            seed=seed,
        )

    def on_set_level(self, level):
        self._grid = [[EMPTY for _ in range(GRID_W)] for _ in range(GRID_H)]
        self._sources = {}
        self._targets = {}
        self._blockers = set()
        self._multipliers = set()
        self._splitters = set()
        self._relays = {}
        self._current_phase = 1
        self._phase1_sources = []
        self._phase1_targets = []
        self._phase2_sources = []
        self._phase2_targets = []
        self._phase3_sources = []
        self._phase3_targets = []
        self._completed_phase1 = set()
        self._completed_phase2 = set()
        self._completed_phase3 = set()
        self._pink_cells = set()
        self._purple_cells = set()
        self._yellow_cells = set()
        self._cyan_cells = set()
        self._orange_cells = set()
        self._green_completed_cells = set()
        self._all_blue_targets = set()
        self._all_green_sources = set()
        self._interconnected_targets = {}
        self._chain_targets = {}
        self._target_required_path = {}
        self._used_sources = set()
        self._connectors = set()
        self._yellow_connectors = set()
        self._cyan_connectors = set()
        self._history = []

        self._mechanic = level.get_data("mechanic") or "basic"

        if self._mechanic == "connector_chain":
            green_sources = level.get_data("green_sources") or []
            connectors = level.get_data("connectors") or []

            for x, y in green_sources:
                self._sources[(x, y)] = 1
                self._grid[y][x] = SOURCE

            for x, y in connectors:
                self._targets[(x, y)] = 1
                self._grid[y][x] = TARGET

            self._connectors = set(connectors)
            self._all_green_sources = set(green_sources)
            self._yellow_connectors = set()
            self._cyan_connectors = set()

        elif self._mechanic == "multi_stage" or self._mechanic == "chain_reaction":
            green_sources = level.get_data("green_sources") or []
            blue_targets = level.get_data("blue_targets") or []

            for x, y in green_sources:
                self._sources[(x, y)] = 1
                self._grid[y][x] = SOURCE

            for x, y in blue_targets:
                self._targets[(x, y)] = 1
                self._grid[y][x] = TARGET

            self._all_blue_targets = set(blue_targets)
            self._all_green_sources = set(green_sources)

            interconnected_config = level.get_data("interconnected_targets") or {}
            for main_target, dependent_list in interconnected_config.items():
                main_key = (
                    tuple(main_target) if isinstance(main_target, list) else main_target
                )
                self._interconnected_targets[main_key] = [
                    tuple(dep) if isinstance(dep, list) else dep
                    for dep in dependent_list
                ]

            chain_config = level.get_data("chain_targets") or {}
            for main_target, dependent_list in chain_config.items():
                main_key = (
                    tuple(main_target) if isinstance(main_target, list) else main_target
                )
                self._chain_targets[main_key] = [
                    tuple(dep) if isinstance(dep, list) else dep
                    for dep in dependent_list
                ]

            for x, y in blue_targets:
                self._target_required_path[(x, y)] = "H"
        else:
            sources = level.get_data("sources") or []
            for x, y, polarity in sources:
                self._sources[(x, y)] = polarity
                self._grid[y][x] = SOURCE

            targets = level.get_data("targets") or []
            for x, y, energy in targets:
                self._targets[(x, y)] = energy
                self._grid[y][x] = TARGET

        blockers = level.get_data("blockers") or []
        for x, y in blockers:
            self._blockers.add((x, y))
            self._grid[y][x] = BLOCKER

        multipliers = level.get_data("multipliers") or []
        for x, y in multipliers:
            self._multipliers.add((x, y))

        splitters = level.get_data("splitters") or []
        for x, y in splitters:
            self._splitters.add((x, y))

        relays = level.get_data("relays") or []
        for item in relays:
            if len(item) == 3:
                x, y, boost = item
                self._relays[(x, y)] = boost

        self._level_name = level.name
        self._max_distance = level.get_data("max_distance") or 999
        self._max_energy = level.get_data("max_energy") or 999
        self._decay_rate = level.get_data("decay_rate") or 0.15
        self._relay_boost = level.get_data("relay_boost") or 0
        self._description = level.get_data("description") or ""

        self._wire_limit = level.get_data("wire_limit") or 999
        self._wire_count = 0
        no_cross = level.get_data("no_cross_zone") or []
        self._no_cross_zone = set()
        for x, y in no_cross:
            self._no_cross_zone.add((x, y))

        for i, lv in enumerate(LEVELS):
            if lv.name == level.name:
                self._level_idx = i
                break

        candidates = [
            (x, y)
            for y in range(2)
            for x in range(GRID_W)
            if self._grid[y][x] == EMPTY
        ]
        if candidates:
            self._cursor_x, self._cursor_y = self._rng.choice(candidates)
        else:
            self._cursor_x = 0
            self._cursor_y = 0
        self._moves = 0
        self._max_moves = level.get_data("max_moves") or 35
        self._lifelines = 3
        self._game_over = False
        self._won = False
        self._consecutive_resets = 0

        self._draw()

    def _can_place_wire(self, x, y):
        if x < 0 or x >= GRID_W or y < 0 or y >= GRID_H:
            return False
        cell = self._grid[y][x]
        return cell == EMPTY or cell in (WIRE_H, WIRE_V)

    def _place_wire(self, x, y, wire_type):
        if not self._can_place_wire(x, y):
            return False

        current = self._grid[y][x]
        if current == EMPTY:
            self._grid[y][x] = wire_type
            return True
        elif current == WIRE_H:
            if wire_type == WIRE_H:
                self._grid[y][x] = WIRE_V
                return True
            else:
                self._grid[y][x] = WIRE_CROSS
                return True
        elif current == WIRE_V:
            if wire_type == WIRE_V:
                self._grid[y][x] = WIRE_H
                return True
            else:
                self._grid[y][x] = WIRE_CROSS
                return True
        return False

    def _remove_wire(self, x, y):
        if x < 0 or x >= GRID_W or y < 0 or y >= GRID_H:
            return False
        if self._grid[y][x] in (WIRE_H, WIRE_V, WIRE_CROSS):
            self._grid[y][x] = EMPTY
            return True
        return False

    def _check_win(self):
        if self._mechanic == "multi_stage":
            return self._check_multi_stage_win()
        if self._mechanic == "chain_reaction":
            return self._check_chain_reaction_win()
        if self._mechanic == "connector_chain":
            return self._check_connector_chain_win()

        energy_map = self._calculate_energy_flow()

        for pos, required_energy in self._targets.items():
            if pos not in energy_map:
                return False

            actual_energy = energy_map[pos]

            if self._mechanic == "polarity":
                source_pos = None
                for spos in self._sources:
                    if spos == pos:
                        source_pos = spos
                        break
                if source_pos is not None:
                    continue
                if abs(actual_energy - required_energy) > 0.01:
                    return False
            elif self._mechanic == "overload":
                if actual_energy < required_energy:
                    return False
                if actual_energy > self._max_energy:
                    return False
            else:
                if actual_energy < required_energy:
                    return False

        return True

    def _check_multi_stage_win(self):
        if self._current_phase == 1:
            if self._check_phase1_connections():
                self._draw()
                return False
            return False
        elif self._current_phase == 2:
            if self._check_phase2_connections():
                self._draw()

            if len(self._yellow_cells) == 0 and len(self._cyan_cells) > 0:
                for pos in list(self._all_blue_targets):
                    if (
                        pos not in self._cyan_cells
                        and pos not in self._pink_cells
                        and pos not in self._yellow_cells
                    ):
                        self._purple_cells.add(pos)

                if len(self._purple_cells) > 0:
                    self._current_phase = 3
                    self._draw()
                else:
                    return True
            return False
        elif self._current_phase == 3:
            if self._check_phase3_connections():
                self._draw()

            if len(self._purple_cells) == 0 and len(self._pink_cells) > 0:
                return True
            return False
        return False

    def _check_chain_reaction_win(self):
        if self._current_phase == 1:
            if self._check_cr_phase1():
                self._draw()
                return False
            return False
        elif self._current_phase == 2:
            if self._check_cr_phase2():
                self._draw()

            if len(self._yellow_cells) == 0 and len(self._orange_cells) > 0:
                self._current_phase = 3
                self._draw()
            return False
        elif self._current_phase == 3:
            if self._check_cr_phase3():
                self._draw()

            if len(self._orange_cells) == 0 and len(self._cyan_cells) > 0:
                for pos in list(self._all_blue_targets):
                    if pos not in self._cyan_cells and pos not in self._pink_cells:
                        self._purple_cells.add(pos)

                if len(self._purple_cells) > 0:
                    self._current_phase = 4
                    self._draw()
                else:
                    return True
            return False
        elif self._current_phase == 4:
            if self._check_cr_phase4():
                self._draw()

            if len(self._purple_cells) == 0 and len(self._pink_cells) > 0:
                return True
            return False
        return False

    def _check_connector_chain_win(self):
        self._yellow_connectors = set()
        self._cyan_connectors = set()

        self._check_source_to_connector()

        self._check_connector_to_connector()

        self._draw()

        if (
            len(self._cyan_connectors) == len(self._connectors)
            and len(self._connectors) > 0
        ):
            return True
        return False

    def _check_source_to_connector(self):
        for sx, sy in self._all_green_sources:
            connected = self._trace_any_path_to_connector(sx, sy)
            if connected and connected in self._connectors:
                if (
                    connected not in self._yellow_connectors
                    and connected not in self._cyan_connectors
                ):
                    self._yellow_connectors.add(connected)

    def _check_connector_to_connector(self):
        changed = True
        while changed:
            changed = False
            for cx, cy in list(self._yellow_connectors):
                connected = self._trace_connector_to_connector(cx, cy)
                if connected and connected in self._connectors:
                    if connected in self._yellow_connectors or connected in self._cyan_connectors:
                        if (cx, cy) in self._yellow_connectors:
                            self._yellow_connectors.discard((cx, cy))
                            self._cyan_connectors.add((cx, cy))
                            changed = True
                        if connected in self._yellow_connectors:
                            self._yellow_connectors.discard(connected)
                            self._cyan_connectors.add(connected)
                            changed = True
                    elif (
                        connected not in self._yellow_connectors
                        and connected not in self._cyan_connectors
                    ):
                        self._yellow_connectors.add(connected)
                        changed = True

    def _trace_any_path_to_connector(self, start_x, start_y):
        visited = set()
        queue = [(start_x, start_y)]
        visited.add((start_x, start_y))
        max_iterations = GRID_W * GRID_H
        iterations = 0

        while queue and iterations < max_iterations:
            iterations += 1
            x, y = queue.pop(0)

            if (x, y) in self._connectors and (x, y) != (start_x, start_y):
                return (x, y)

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < GRID_W and 0 <= ny < GRID_H and (nx, ny) not in visited:
                    cell = self._grid[ny][nx]
                    if cell in (WIRE_H, WIRE_V, WIRE_CROSS, SOURCE, TARGET):
                        if cell == WIRE_H and dy != 0:
                            continue
                        if cell == WIRE_V and dx != 0:
                            continue
                        visited.add((nx, ny))
                        queue.append((nx, ny))
                    elif (nx, ny) in self._connectors:
                        visited.add((nx, ny))
                        queue.append((nx, ny))
        return None

    def _trace_connector_to_connector(self, start_x, start_y):
        visited = set()
        queue = [(start_x, start_y)]
        visited.add((start_x, start_y))
        max_iterations = GRID_W * GRID_H
        iterations = 0

        while queue and iterations < max_iterations:
            iterations += 1
            x, y = queue.pop(0)

            if (x, y) in self._connectors and (x, y) != (start_x, start_y):
                if (x, y) not in self._cyan_connectors:
                    return (x, y)

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < GRID_W and 0 <= ny < GRID_H and (nx, ny) not in visited:
                    cell = self._grid[ny][nx]
                    if cell in (WIRE_H, WIRE_V, WIRE_CROSS, TARGET):
                        if cell == WIRE_H and dy != 0:
                            continue
                        if cell == WIRE_V and dx != 0:
                            continue
                        visited.add((nx, ny))
                        queue.append((nx, ny))
                    elif (nx, ny) in self._connectors:
                        visited.add((nx, ny))
                        queue.append((nx, ny))
        return None

    def _check_cr_phase1(self):
        for sx, sy in self._all_green_sources:
            if (sx, sy) in self._used_sources:
                continue

            connected_target = self._trace_horizontal_to_blue(sx, sy)
            if connected_target and connected_target in self._all_blue_targets:
                if (
                    connected_target not in self._yellow_cells
                    and connected_target not in self._cyan_cells
                ):
                    nearest = self._get_nearest_unused_source(connected_target)
                    if nearest != (sx, sy):
                        continue

                    self._used_sources.add((sx, sy))
                    self._yellow_cells.add(connected_target)

                    if connected_target in self._interconnected_targets:
                        for dep_target in self._interconnected_targets[
                            connected_target
                        ]:
                            if (
                                dep_target in self._all_blue_targets
                                and dep_target not in self._yellow_cells
                            ):
                                self._yellow_cells.add(dep_target)

                    self._current_phase = 2
                    return True
        return False

    def _check_cr_phase2(self):
        made_connection = False

        for sx, sy in self._all_green_sources:
            if (sx, sy) in self._used_sources:
                continue

            connected_target = self._trace_vertical_to_yellow(sx, sy)

            if connected_target and connected_target in self._yellow_cells:
                nearest = self._get_nearest_unused_source(connected_target)
                if nearest != (sx, sy):
                    continue

                self._used_sources.add((sx, sy))
                self._yellow_cells.discard(connected_target)
                self._orange_cells.add(connected_target)
                made_connection = True

        return made_connection

    def _check_cr_phase3(self):
        made_connection = False

        for sx, sy in self._all_green_sources:
            if (sx, sy) in self._used_sources:
                continue

            connected_target = self._trace_horizontal_to_orange(sx, sy)

            if connected_target and connected_target in self._orange_cells:
                nearest = self._get_nearest_unused_source(connected_target)
                if nearest != (sx, sy):
                    continue

                self._used_sources.add((sx, sy))
                self._orange_cells.discard(connected_target)
                self._cyan_cells.add(connected_target)

                if connected_target in self._chain_targets:
                    for chain_target in self._chain_targets[connected_target]:
                        if chain_target in self._all_blue_targets:
                            self._orange_cells.discard(chain_target)
                            self._yellow_cells.discard(chain_target)
                            self._cyan_cells.add(chain_target)

                made_connection = True

        return made_connection

    def _check_cr_phase4(self):
        made_connection = False

        for sx, sy in self._all_green_sources:
            if (sx, sy) in self._used_sources:
                continue

            connected_target = self._trace_vertical_to_purple(sx, sy)

            if connected_target and connected_target in self._purple_cells:
                nearest = self._get_nearest_unused_source(connected_target)
                if nearest != (sx, sy):
                    continue

                self._used_sources.add((sx, sy))
                self._purple_cells.discard(connected_target)
                self._pink_cells.add(connected_target)
                made_connection = True

        return made_connection

    def _trace_horizontal_to_orange(self, start_x, start_y):
        visited = set()
        queue = [(start_x, start_y)]
        visited.add((start_x, start_y))
        max_iterations = GRID_W
        iterations = 0

        while queue and iterations < max_iterations:
            iterations += 1
            x, y = queue.pop(0)

            if (x, y) in self._orange_cells:
                if (x, y) != (start_x, start_y):
                    return (x, y)

            for dx in [-1, 1]:
                nx, ny = x + dx, y
                if 0 <= nx < GRID_W and (nx, ny) not in visited:
                    cell = self._grid[ny][nx]
                    if (
                        cell == WIRE_H
                        or cell == TARGET
                        or cell == SOURCE
                        or (nx, ny) in self._orange_cells
                    ):
                        visited.add((nx, ny))
                        queue.append((nx, ny))
        return None

    def _trace_vertical_to_purple(self, start_x, start_y):
        visited = set()
        queue = [(start_x, start_y)]
        visited.add((start_x, start_y))
        max_iterations = GRID_H
        iterations = 0

        while queue and iterations < max_iterations:
            iterations += 1
            x, y = queue.pop(0)

            if (x, y) in self._purple_cells:
                if (x, y) != (start_x, start_y):
                    return (x, y)

            for dy in [-1, 1]:
                nx, ny = x, y + dy
                if 0 <= ny < GRID_H and (nx, ny) not in visited:
                    cell = self._grid[ny][nx]
                    if (
                        cell == WIRE_V
                        or cell == TARGET
                        or cell == SOURCE
                        or (nx, ny) in self._purple_cells
                    ):
                        visited.add((nx, ny))
                        queue.append((nx, ny))
        return None

    def _get_nearest_unused_source(self, target_pos):
        tx, ty = target_pos
        nearest_source = None
        min_dist = float("inf")
        for sx, sy in self._all_green_sources:
            if (sx, sy) not in self._used_sources:
                dist = abs(sx - tx) + abs(sy - ty)
                if dist < min_dist:
                    min_dist = dist
                    nearest_source = (sx, sy)
        return nearest_source

    def _get_nearest_blue_targets(self, target_pos, count):
        tx, ty = target_pos
        distances = []
        for pos in self._all_blue_targets:
            if (
                pos != target_pos
                and pos not in self._yellow_cells
                and pos not in self._purple_cells
                and pos not in self._pink_cells
            ):
                bx, by = pos
                dist = abs(bx - tx) + abs(by - ty)
                distances.append((dist, pos))
        distances.sort(key=lambda x: x[0])
        return [pos for _, pos in distances[:count]]

    def _get_nearest_yellow_cells(self, target_pos, count):
        tx, ty = target_pos
        distances = []
        for pos in self._yellow_cells:
            if pos != target_pos:
                bx, by = pos
                dist = abs(bx - tx) + abs(by - ty)
                distances.append((dist, pos))
        distances.sort(key=lambda x: x[0])
        return [pos for _, pos in distances[:count]]

    def _get_nearest_purple_cells(self, target_pos, count):
        tx, ty = target_pos
        distances = []
        for pos in self._purple_cells:
            if pos != target_pos:
                bx, by = pos
                dist = abs(bx - tx) + abs(by - ty)
                distances.append((dist, pos))
        distances.sort(key=lambda x: x[0])
        return [pos for _, pos in distances[:count]]

    def _get_nearest_source_to_target(self, target_pos):
        tx, ty = target_pos
        nearest_source = None
        min_dist = float("inf")
        for sx, sy in self._all_green_sources:
            dist = abs(sx - tx) + abs(sy - ty)
            if dist < min_dist:
                min_dist = dist
                nearest_source = (sx, sy)
        return nearest_source

    def _check_phase1_connections(self):
        for sx, sy in self._all_green_sources:
            if (sx, sy) in self._used_sources:
                continue

            connected_target = self._trace_horizontal_to_blue(sx, sy)
            if connected_target and connected_target in self._all_blue_targets:
                if (
                    connected_target not in self._yellow_cells
                    and connected_target not in self._cyan_cells
                ):
                    nearest = self._get_nearest_unused_source(connected_target)
                    if nearest != (sx, sy):
                        continue

                    self._used_sources.add((sx, sy))

                    self._yellow_cells.add(connected_target)

                    if connected_target in self._interconnected_targets:
                        dependent_targets = self._interconnected_targets[
                            connected_target
                        ]
                        for dep_target in dependent_targets:
                            if (
                                dep_target in self._all_blue_targets
                                and dep_target not in self._yellow_cells
                            ):
                                self._yellow_cells.add(dep_target)

                    self._current_phase = 2
                    return True
        return False

    def _check_phase2_connections(self):
        made_connection = False

        for sx, sy in self._all_green_sources:
            if (sx, sy) in self._used_sources:
                continue

            connected_target = self._trace_vertical_to_yellow(sx, sy)

            if connected_target and connected_target in self._yellow_cells:
                nearest = self._get_nearest_unused_source(connected_target)
                if nearest != (sx, sy):
                    continue

                self._used_sources.add((sx, sy))

                self._yellow_cells.discard(connected_target)
                self._cyan_cells.add(connected_target)
                made_connection = True

        return made_connection

    def _check_phase3_connections(self):
        made_connection = False

        for sx, sy in self._all_green_sources:
            if (sx, sy) in self._used_sources:
                continue

            connected_target = self._trace_horizontal_to_purple(sx, sy)

            if connected_target and connected_target in self._purple_cells:
                nearest = self._get_nearest_unused_source(connected_target)
                if nearest != (sx, sy):
                    continue

                self._used_sources.add((sx, sy))

                self._purple_cells.discard(connected_target)
                self._pink_cells.add(connected_target)

                if connected_target in self._interconnected_targets:
                    for dep_target in self._interconnected_targets[connected_target]:
                        if dep_target in self._purple_cells:
                            self._purple_cells.discard(dep_target)
                            self._pink_cells.add(dep_target)

                made_connection = True

        return made_connection

    def _trace_horizontal_to_blue(self, start_x, start_y):
        visited = set()
        queue = [(start_x, start_y)]
        visited.add((start_x, start_y))
        max_iterations = GRID_W
        iterations = 0

        while queue and iterations < max_iterations:
            iterations += 1
            x, y = queue.pop(0)

            if (x, y) in self._all_blue_targets:
                if (x, y) not in self._yellow_cells and (x, y) not in self._cyan_cells:
                    if (x, y) != (start_x, start_y):
                        return (x, y)

            for dx in [-1, 1]:
                nx, ny = x + dx, y
                if 0 <= nx < GRID_W and (nx, ny) not in visited:
                    cell = self._grid[ny][nx]
                    if cell == WIRE_H or cell == TARGET or cell == SOURCE:
                        visited.add((nx, ny))
                        queue.append((nx, ny))
        return None

    def _trace_vertical_to_yellow(self, start_x, start_y):
        visited = set()
        queue = [(start_x, start_y)]
        visited.add((start_x, start_y))
        max_iterations = GRID_H
        iterations = 0

        while queue and iterations < max_iterations:
            iterations += 1
            x, y = queue.pop(0)

            if (x, y) in self._yellow_cells:
                if (x, y) != (start_x, start_y):
                    return (x, y)

            for dy in [-1, 1]:
                nx, ny = x, y + dy
                if 0 <= ny < GRID_H and (nx, ny) not in visited:
                    cell = self._grid[ny][nx]
                    if (
                        cell == WIRE_V
                        or cell == TARGET
                        or cell == SOURCE
                        or (nx, ny) in self._yellow_cells
                    ):
                        visited.add((nx, ny))
                        queue.append((nx, ny))
        return None

    def _trace_horizontal_to_yellow(self, start_x, start_y):
        visited = set()
        queue = [(start_x, start_y)]
        visited.add((start_x, start_y))

        while queue:
            x, y = queue.pop(0)

            if (x, y) in self._yellow_cells:
                if (x, y) != (start_x, start_y):
                    return (x, y)

            for dx in [-1, 1]:
                nx, ny = x + dx, y
                if 0 <= nx < GRID_W and (nx, ny) not in visited:
                    cell = self._grid[ny][nx]
                    if (
                        cell == WIRE_H
                        or cell == TARGET
                        or cell == SOURCE
                        or (nx, ny) in self._yellow_cells
                    ):
                        visited.add((nx, ny))
                        queue.append((nx, ny))
        return None

    def _trace_horizontal_to_cyan(self, start_x, start_y):
        visited = set()
        queue = [(start_x, start_y)]
        visited.add((start_x, start_y))

        while queue:
            x, y = queue.pop(0)

            if (x, y) in self._cyan_cells:
                if (x, y) != (start_x, start_y):
                    return (x, y)

            for dx in [-1, 1]:
                nx, ny = x + dx, y
                if 0 <= nx < GRID_W and (nx, ny) not in visited:
                    cell = self._grid[ny][nx]
                    if (
                        cell == WIRE_H
                        or cell == TARGET
                        or cell == SOURCE
                        or (nx, ny) in self._cyan_cells
                    ):
                        visited.add((nx, ny))
                        queue.append((nx, ny))
        return None

    def _trace_horizontal_to_purple(self, start_x, start_y):
        visited = set()
        queue = [(start_x, start_y)]
        visited.add((start_x, start_y))

        while queue:
            x, y = queue.pop(0)

            if (x, y) in self._purple_cells:
                if (x, y) != (start_x, start_y):
                    return (x, y)

            for dx in [-1, 1]:
                nx, ny = x + dx, y
                if 0 <= nx < GRID_W and (nx, ny) not in visited:
                    cell = self._grid[ny][nx]
                    if (
                        cell in (WIRE_H, WIRE_CROSS, TARGET, SOURCE)
                        or (nx, ny) in self._purple_cells
                    ):
                        visited.add((nx, ny))
                        queue.append((nx, ny))
        return None

    def _calculate_energy_flow(self):
        energy_map = {}

        for pos, polarity in self._sources.items():
            visited = set()
            self._trace_energy(
                pos[0], pos[1], polarity, visited, energy_map, 0, polarity, 1
            )

        return energy_map

    def _trace_energy(
        self,
        x,
        y,
        energy,
        visited,
        energy_map,
        distance,
        source_polarity,
        flow_fraction,
    ):
        if distance > GRID_W * GRID_H:
            return

        key = (x, y, source_polarity)
        if key in visited:
            return
        visited.add(key)

        if self._mechanic == "decay" and distance > 0:
            decay_per_step = 0.15
            energy = energy - (distance * decay_per_step)
            if energy <= 0.1:
                return

        if self._mechanic == "decay_relay" and distance > 0:
            decay_per_step = self._decay_rate
            energy = energy - (distance * decay_per_step)
            if energy <= 0.1:
                return

        is_source_cell = (x, y) in self._sources
        if (
            (x, y) in self._relays
            and self._mechanic == "decay_relay"
            and not is_source_cell
        ):
            energy = min(energy + self._relay_boost, 999)

        if (x, y) in self._targets:
            if self._mechanic == "polarity":
                target_polarity = self._targets[(x, y)]
                if (source_polarity > 0 and target_polarity > 0) or (
                    source_polarity < 0 and target_polarity < 0
                ):
                    energy_map[(x, y)] = energy_map.get((x, y), 0) + (
                        energy * flow_fraction
                    )
            else:
                energy_map[(x, y)] = energy_map.get((x, y), 0) + (
                    abs(energy) * flow_fraction
                )
            return

        if (
            (x, y) in self._multipliers
            and self._mechanic == "multiplier"
            and not is_source_cell
        ):
            energy = min(energy * 2, 999)

        if (
            (x, y) in self._splitters
            and self._mechanic == "splitter"
            and not is_source_cell
        ):
            flow_fraction = flow_fraction * 0.5

        cell = self._grid[y][x] if 0 <= y < GRID_H and 0 <= x < GRID_W else EMPTY

        neighbors = []

        if cell == WIRE_H:
            for dx, dy in [(-1, 0), (1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < GRID_W and 0 <= ny < GRID_H:
                    next_cell = self._grid[ny][nx]
                    if next_cell in (WIRE_H, WIRE_CROSS, TARGET):
                        neighbors.append((nx, ny))
        elif cell == WIRE_V:
            for dx, dy in [(0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < GRID_W and 0 <= ny < GRID_H:
                    next_cell = self._grid[ny][nx]
                    if next_cell in (WIRE_V, WIRE_CROSS, TARGET):
                        neighbors.append((nx, ny))
        elif cell == WIRE_CROSS:
            for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < GRID_W and 0 <= ny < GRID_H:
                    next_cell = self._grid[ny][nx]
                    if next_cell in (WIRE_H, WIRE_V, WIRE_CROSS, TARGET):
                        neighbors.append((nx, ny))
        elif cell == SOURCE:
            for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < GRID_W and 0 <= ny < GRID_H:
                    next_cell = self._grid[ny][nx]
                    if next_cell in (WIRE_H, WIRE_V, WIRE_CROSS):
                        neighbors.append((nx, ny))

        for nx, ny in neighbors:
            self._trace_energy(
                nx,
                ny,
                energy,
                visited,
                energy_map,
                distance + 1,
                source_polarity,
                flow_fraction,
            )

    def _move_cursor(self, dx, dy):
        self._save_state()
        new_x = self._cursor_x + dx
        new_y = self._cursor_y + dy
        if 0 <= new_x < GRID_W and 0 <= new_y < GRID_H:
            self._cursor_x = new_x
            self._cursor_y = new_y
        self._moves += 1
        self._check_moves()
        if not self._game_over:
            self._draw()

    def _place_action(self, wire_type):
        self._moves += 1
        placed = self._place_wire(self._cursor_x, self._cursor_y, wire_type)
        self._check_moves()
        if not self._game_over:
            if placed and self._check_win():
                self._won = True
                self._game_over = True
            self._draw()

    def _remove_action(self):
        self._moves += 1
        self._remove_wire(self._cursor_x, self._cursor_y)
        self._check_moves()
        if not self._game_over:
            self._draw()

    def _check_moves(self):
        if self._moves >= self._max_moves:
            if self._lifelines > 0:
                self._lifelines -= 1
                if self._lifelines > 0:
                    self._moves = 0
                    self._history = []
                    self._grid = [[EMPTY for _ in range(GRID_W)] for _ in range(GRID_H)]
                    for pos, polarity in self._sources.items():
                        x, y = pos
                        self._grid[y][x] = SOURCE
                    for pos in self._targets.keys():
                        x, y = pos
                        self._grid[y][x] = TARGET
                    for pos in self._blockers:
                        x, y = pos
                        self._grid[y][x] = BLOCKER
                    self._current_phase = 1
                    self._completed_phase1 = set()
                    self._completed_phase2 = set()
                    self._completed_phase3 = set()
                    self._pink_cells = set()
                    self._purple_cells = set()
                    self._yellow_cells = set()
                    self._cyan_cells = set()
                    self._orange_cells = set()
                    self._green_completed_cells = set()
                    self._used_sources = set()
                    self._yellow_connectors = set()
                    self._cyan_connectors = set()
                    self._wire_count = 0
                    self._cursor_x = 0
                    self._cursor_y = 0
                    self._draw()
                else:
                    self._game_over = True
                    self.lose()
                    self._draw()
            else:
                self._game_over = True
                self.lose()
                self._draw()

    def _draw(self):
        level = self.current_level
        level.remove_all_sprites()

        for y in range(GRID_H):
            for x in range(GRID_W):
                cell = self._grid[y][x]
                px = None

                if self._mechanic == "connector_chain":
                    if (x, y) in self._cyan_connectors:
                        px = _cell_cyan_complete()
                    elif (x, y) in self._yellow_connectors:
                        px = _cell_yellow_complete()
                    elif (x, y) in self._connectors:
                        px = _cell_target(1, False)
                    elif (x, y) in self._multipliers:
                        px = _cell_multiplier()
                    elif (x, y) in self._splitters:
                        px = _cell_splitter()
                    elif (x, y) in self._relays:
                        px = _cell_relay()
                    elif cell == EMPTY:
                        px = _cell_empty()
                    elif cell == WIRE_H:
                        px = _cell_wire_h()
                    elif cell == WIRE_V:
                        px = _cell_wire_v()
                    elif cell == WIRE_CROSS:
                        px = _cell_wire_cross()
                    elif cell == SOURCE:
                        polarity = self._sources.get((x, y), 1)
                        px = _cell_source(polarity)
                    elif cell == TARGET:
                        if (x, y) not in self._connectors:
                            px = _cell_target(1, False)
                    elif cell == BLOCKER:
                        px = _cell_blocker()
                elif (
                    self._mechanic == "multi_stage"
                    or self._mechanic == "chain_reaction"
                ):
                    if (x, y) in self._green_completed_cells:
                        px = _cell_green_complete()
                    elif (x, y) in self._cyan_cells:
                        px = _cell_cyan_complete()
                    elif (x, y) in self._orange_cells:
                        px = _cell_orange_complete()
                    elif (x, y) in self._yellow_cells:
                        px = _cell_yellow_complete()
                    elif (x, y) in self._purple_cells:
                        px = _cell_purple()
                    elif (x, y) in self._pink_cells:
                        px = _cell_pink()
                    elif (x, y) in self._multipliers:
                        px = _cell_multiplier()
                    elif (x, y) in self._splitters:
                        px = _cell_splitter()
                    elif (x, y) in self._relays:
                        px = _cell_relay()
                    elif cell == EMPTY:
                        px = _cell_empty()
                    elif cell == WIRE_H:
                        px = _cell_wire_h()
                    elif cell == WIRE_V:
                        px = _cell_wire_v()
                    elif cell == WIRE_CROSS:
                        px = _cell_wire_cross()
                    elif cell == SOURCE:
                        polarity = self._sources.get((x, y), 1)
                        px = _cell_source(polarity)
                    elif cell == TARGET:
                        energy = self._targets.get((x, y), 1)
                        px = _cell_target(energy, False)
                    elif cell == BLOCKER:
                        px = _cell_blocker()
                else:
                    if (x, y) in self._multipliers:
                        px = _cell_multiplier()
                    elif (x, y) in self._splitters:
                        px = _cell_splitter()
                    elif (x, y) in self._relays:
                        px = _cell_relay()
                    elif cell == EMPTY:
                        px = _cell_empty()
                    elif cell == WIRE_H:
                        px = _cell_wire_h()
                    elif cell == WIRE_V:
                        px = _cell_wire_v()
                    elif cell == WIRE_CROSS:
                        px = _cell_wire_cross()
                    elif cell == SOURCE:
                        polarity = self._sources.get((x, y), 1)
                        px = _cell_source(polarity)
                    elif cell == TARGET:
                        energy = self._targets.get((x, y), 1)
                        energy_map = self._calculate_energy_flow()
                        actual = energy_map.get((x, y), 0)
                        active = actual >= energy
                        if self._mechanic == "overload":
                            active = active and actual <= self._max_energy
                        px = _cell_target(energy, active)
                    elif cell == BLOCKER:
                        px = _cell_blocker()

                if px is not None:
                    level.add_sprite(
                        Sprite(
                            pixels=px,
                            name=f"cell_{x}_{y}",
                            x=x * CELL,
                            y=y * CELL,
                            layer=1,
                            visible=True,
                            collidable=False,
                            tags=[],
                        )
                    )

        level.add_sprite(
            Sprite(
                pixels=_cursor(),
                name="cursor",
                x=self._cursor_x * CELL,
                y=self._cursor_y * CELL,
                layer=2,
                visible=True,
                collidable=False,
                tags=[],
            )
        )

        hud_y = GRID_H * CELL
        hud_bg = _px(9, CAM_W, BLACK)
        level.add_sprite(
            Sprite(
                pixels=hud_bg,
                name="hud",
                x=0,
                y=hud_y,
                layer=3,
                visible=True,
                collidable=False,
                tags=[],
            )
        )

        pb_width = 42
        pb_x = 2
        pb_y = hud_y + 3
        remaining = max(0, self._max_moves - self._moves)
        fill_pct = remaining / self._max_moves if self._max_moves > 0 else 0
        filled = int(pb_width * fill_pct)

        if filled > 0:
            color = GREEN if fill_pct > 0.5 else YELLOW if fill_pct > 0.2 else RED
            pb_fill = _px(2, filled, color)
            level.add_sprite(
                Sprite(
                    pixels=pb_fill,
                    name="pb_fill",
                    x=pb_x,
                    y=pb_y,
                    layer=4,
                    visible=True,
                    collidable=False,
                    tags=[],
                )
            )

        lf_x = pb_x + pb_width + 3
        for i in range(3):
            lf_color = RED if i < self._lifelines else GRAY
            lf_px = _px(2, 2, lf_color)
            level.add_sprite(
                Sprite(
                    pixels=lf_px,
                    name=f"lf_{i}",
                    x=lf_x + i * 3,
                    y=pb_y,
                    layer=4,
                    visible=True,
                    collidable=False,
                    tags=[],
                )
            )

    def step(self):
        if self._game_over:
            self.complete_action()
            return

        aid = self.action.id
        if aid == GameAction.RESET:
            self.complete_action()
            return
        elif aid == GameAction.ACTION1:
            self._move_cursor(0, -1)
        elif aid == GameAction.ACTION2:
            self._move_cursor(0, 1)
        elif aid == GameAction.ACTION3:
            self._move_cursor(-1, 0)
        elif aid == GameAction.ACTION4:
            self._move_cursor(1, 0)
        elif aid == GameAction.ACTION5:
            self._handle_space_press()
        elif aid == GameAction.ACTION7:
            self._handle_undo()

        if self._won:
            self.next_level()

        self.complete_action()

    def _save_state(self):
        self._history.append({
            "grid": copy.deepcopy(self._grid),
            "wire_count": self._wire_count,
            "current_phase": self._current_phase,
            "yellow_cells": set(self._yellow_cells),
            "cyan_cells": set(self._cyan_cells),
            "orange_cells": set(self._orange_cells),
            "purple_cells": set(self._purple_cells),
            "pink_cells": set(self._pink_cells),
            "green_completed_cells": set(self._green_completed_cells),
            "used_sources": set(self._used_sources),
            "yellow_connectors": set(self._yellow_connectors),
            "cyan_connectors": set(self._cyan_connectors),
            "cursor_x": self._cursor_x,
            "cursor_y": self._cursor_y,
        })

    def _restore_state(self):
        if not self._history:
            return False
        snap = self._history.pop()
        self._grid = snap["grid"]
        self._wire_count = snap["wire_count"]
        self._current_phase = snap["current_phase"]
        self._yellow_cells = snap["yellow_cells"]
        self._cyan_cells = snap["cyan_cells"]
        self._orange_cells = snap["orange_cells"]
        self._purple_cells = snap["purple_cells"]
        self._pink_cells = snap["pink_cells"]
        self._green_completed_cells = snap["green_completed_cells"]
        self._used_sources = snap["used_sources"]
        self._yellow_connectors = snap["yellow_connectors"]
        self._cyan_connectors = snap["cyan_connectors"]
        self._cursor_x = snap["cursor_x"]
        self._cursor_y = snap["cursor_y"]
        return True

    def _handle_undo(self):
        self._moves += 1
        self._restore_state()
        self._check_moves()
        if not self._game_over:
            self._draw()

    def _handle_space_press(self):
        x, y = self._cursor_x, self._cursor_y
        current = self._grid[y][x]

        if current == EMPTY:
            if self._wire_count >= self._wire_limit:
                return
            self._save_state()
            self._grid[y][x] = WIRE_H
            self._wire_count += 1
            self._moves += 1
            self._check_moves()
            if not self._game_over:
                if self._check_win():
                    self._won = True
                    self._game_over = True
                self._draw()
        elif current == WIRE_H:
            self._save_state()
            self._grid[y][x] = WIRE_V
            self._moves += 1
            self._check_moves()
            if not self._game_over:
                if self._check_win():
                    self._won = True
                    self._game_over = True
                self._draw()
        elif current == WIRE_V:
            if (x, y) in self._no_cross_zone:
                self._save_state()
                self._grid[y][x] = EMPTY
                self._wire_count -= 1
                self._moves += 1
                self._check_moves()
                if not self._game_over:
                    self._draw()
            else:
                self._save_state()
                self._grid[y][x] = WIRE_CROSS
                self._moves += 1
                self._check_moves()
                if not self._game_over:
                    if self._check_win():
                        self._won = True
                        self._game_over = True
                    self._draw()
        elif current == WIRE_CROSS:
            self._save_state()
            self._grid[y][x] = EMPTY
            self._wire_count -= 1
            self._moves += 1
            self._check_moves()
            if not self._game_over:
                self._draw()


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


def _encode_png(frame):
    h, w = frame.shape[:2]
    if frame.ndim == 2:
        raw = np.zeros((h, w, 3), dtype=np.uint8)
        raw[:] = frame[:, :, None]
    else:
        raw = frame.astype(np.uint8)
    rows = [b"\x00" + raw[y].tobytes() for y in range(h)]
    raw_data = b"".join(rows)
    compressed = zlib.compress(raw_data)

    def _chunk(ctype, data):
        c = ctype + data
        return (
            struct.pack(">I", len(data))
            + c
            + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        )

    buf = io.BytesIO()
    buf.write(b"\x89PNG\r\n\x1a\n")
    buf.write(_chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)))
    buf.write(_chunk(b"IDAT", compressed))
    buf.write(_chunk(b"IEND", b""))
    return buf.getvalue()


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
        self._engine = Gr19(seed=seed)
        self._total_turns = 0
        self._done = False
        self._game_won = False
        self._last_action_was_reset = False

    def reset(self) -> GameState:
        e = self._engine
        if self._game_won or self._last_action_was_reset:
            e.perform_action(ActionInput(id=GameAction.RESET, data={}))
            e.perform_action(ActionInput(id=GameAction.RESET, data={}))
        else:
            e.perform_action(ActionInput(id=GameAction.RESET, data={}))
        self._total_turns = 0
        self._done = False
        self._last_action_was_reset = True
        self._game_won = False
        return self._build_game_state()

    def step(self, action: str) -> StepResult:
        if action == "reset":
            state = self.reset()
            return StepResult(
                state=state, reward=0.0, done=False, info={"action": "reset"}
            )

        self._last_action_was_reset = False
        self._total_turns += 1

        e = self._engine
        game_action = self._ACTION_MAP.get(action, GameAction.RESET)
        info = {"action": action}
        level_before = e.level_index

        action_input = ActionInput(id=game_action, data={})
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
            state=self._build_game_state(),
            reward=reward,
            done=False,
            info=info,
        )

    def _build_text_observation(self) -> str:
        e = self._engine
        lines = [f"=== Grid Reactor — {e._level_name} ==="]
        lines.append(f"{e._description}")
        lines.append(f"Moves: {e._moves}/{e._max_moves}  Lifelines: {e._lifelines}")
        lines.append(f"Cursor: ({e._cursor_x}, {e._cursor_y})")

        if e._mechanic == "basic":
            energy_map = e._calculate_energy_flow()
            lines.append("Targets:")
            for pos, required in e._targets.items():
                actual = energy_map.get(pos, 0)
                status = "OK" if actual >= required else "NEED MORE"
                lines.append(f"  {pos}: {actual:.1f}/{required} [{status}]")

        return "\n".join(lines)

    def _build_game_state(self, done: bool = False) -> GameState:
        return GameState(
            text_observation=self._build_text_observation(),
            image_observation=None,
            valid_actions=self.get_actions() if not done else None,
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

    def get_actions(self) -> List[str]:
        if self._done:
            return ["reset"]
        return list(self._VALID_ACTIONS)

    def is_done(self) -> bool:
        return self._done

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError(f"Unsupported render mode: {mode}")
        e = self._engine
        idx_frame = e.camera.render(e.current_level.get_sprites())
        h, w = idx_frame.shape[:2]
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for idx in range(len(ARC_PALETTE)):
            mask = idx_frame == idx
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            rgb[mask] = ARC_PALETTE[idx]
        return rgb

    def close(self) -> None:
        self._engine = None


class ArcGameEnv(gym.Env):
    metadata: Dict[str, Any] = {"render_modes": ["rgb_array"], "render_fps": 5}
    ACTION_LIST: List[str] = [
        "reset",
        "up",
        "down",
        "left",
        "right",
        "select",
        "undo",
    ]
    OBS_HEIGHT = 64
    OBS_WIDTH = 64

    def __init__(self, seed: int = 0, render_mode: Optional[str] = None) -> None:
        super().__init__()
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"Unsupported render_mode '{render_mode}'. "
                f"Supported: {self.metadata['render_modes']}"
            )
        self.render_mode = render_mode
        self._action_to_string = {i: a for i, a in enumerate(self.ACTION_LIST)}
        self._string_to_action = {a: i for i, a in enumerate(self.ACTION_LIST)}
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.OBS_HEIGHT, self.OBS_WIDTH, 3),
            dtype=np.uint8,
        )
        self.action_space = spaces.Discrete(len(self.ACTION_LIST))
        self._seed = seed
        self._env = None

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed
        self._env = PuzzleEnvironment(seed=self._seed)
        state = self._env.reset()
        return self._get_obs(), self._build_info(state)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        action_str = self._action_to_string[int(action)]
        result = self._env.step(action_str)
        obs = self._get_obs()
        reward = result.reward
        terminated = result.done
        truncated = False
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
        h, w = img.shape[:2]
        row_idx = (np.arange(target_h) * h // target_h).astype(int)
        col_idx = (np.arange(target_w) * w // target_w).astype(int)
        return img[np.ix_(row_idx, col_idx)].astype(np.uint8)

    def _build_info(self, state: GameState, step_info: Optional[Dict] = None) -> Dict:
        info = {
            "text_observation": state.text_observation,
            "valid_actions": state.valid_actions,
            "turn": state.turn,
            "game_metadata": state.metadata,
        }
        if step_info:
            info["step_info"] = step_info
        return info
