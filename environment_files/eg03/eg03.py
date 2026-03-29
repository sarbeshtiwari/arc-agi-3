import io
import random
import struct
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
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
    metadata: Dict = field(default_factory=dict)


@dataclass
class StepResult:
    state: GameState
    reward: float
    done: bool
    info: Dict = field(default_factory=dict)


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
LIGHT_BLUE = 8
DARK_RED = 9
DARK_BLUE = 10
BROWN = 11
PINK = 12
DARK_GREY = 13
LIME = 14
PURPLE = 15

GRID_W = 64
GRID_H = 64
BOARD_SIZE = 8

TILE_PX = 7
BOARD_PX = TILE_PX * BOARD_SIZE
OFFSET_X = (GRID_W - BOARD_PX) // 2
OFFSET_Y = (GRID_H - BOARD_PX) // 2 + 3

MTYPE_PRODUCER = 0
MTYPE_CONVERTER = 1
MTYPE_DISTRIBUTOR = 2
MTYPE_LOCK = 3
MTYPE_ASSEMBLER = 4

MTYPE_COLORS = {
    MTYPE_PRODUCER: DARK_RED,
    MTYPE_CONVERTER: PINK,
    MTYPE_DISTRIBUTOR: PURPLE,
    MTYPE_LOCK: DARK_GREY,
    MTYPE_ASSEMBLER: MAGENTA,
}

STATE_LOCKED = 0
STATE_READY = 1
STATE_ACTIVE = 2
STATE_BLOCKED = 3

RES_METAL = 0
RES_POWER = 1
RES_FUEL = 2

NUM_RES_TYPES = 3

COL_CURSOR = BROWN
COL_CURSOR_DOT = PINK
COL_DEP_LINE = RED
LIFE_ON = GREEN
LIFE_OFF = DARK_GREY

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


def _index_to_rgb(index_grid):
    arr = np.array(index_grid, dtype=np.uint8)
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    for idx, color in enumerate(ARC_PALETTE):
        mask = arr == idx
        rgb[mask] = color
    return rgb


LEVEL_DEFS = [
    {
        "machines": [
            {
                "id": 0,
                "row": 0,
                "col": 1,
                "mtype": MTYPE_PRODUCER,
                "deps": [],
                "res_req": {},
                "res_prod": {RES_METAL: 1},
                "effects": [],
            },
            {
                "id": 1,
                "row": 0,
                "col": 5,
                "mtype": MTYPE_PRODUCER,
                "deps": [],
                "res_req": {},
                "res_prod": {RES_POWER: 1},
                "effects": [],
            },
            {
                "id": 2,
                "row": 2,
                "col": 1,
                "mtype": MTYPE_LOCK,
                "deps": [0],
                "res_req": {},
                "res_prod": {},
                "effects": [("block", 4, None)],
            },
            {
                "id": 3,
                "row": 2,
                "col": 5,
                "mtype": MTYPE_CONVERTER,
                "deps": [1],
                "res_req": {RES_POWER: 1},
                "res_prod": {RES_FUEL: 1},
                "effects": [],
            },
            {
                "id": 4,
                "row": 4,
                "col": 3,
                "mtype": MTYPE_DISTRIBUTOR,
                "deps": [0],
                "res_req": {RES_METAL: 1},
                "res_prod": {},
                "effects": [("unblock", 5, None)],
            },
            {
                "id": 5,
                "row": 6,
                "col": 3,
                "mtype": MTYPE_ASSEMBLER,
                "deps": [2, 3, 4],
                "res_req": {RES_FUEL: 1},
                "res_prod": {},
                "effects": [],
            },
            {
                "id": 6,
                "row": 4,
                "col": 0,
                "mtype": MTYPE_LOCK,
                "deps": [0],
                "res_req": {},
                "res_prod": {},
                "effects": [("block", 5, None), ("add_dep", 5, 6)],
            },
            {
                "id": 7,
                "row": 4,
                "col": 6,
                "mtype": MTYPE_LOCK,
                "deps": [0],
                "res_req": {RES_METAL: 1},
                "res_prod": {},
                "effects": [("consume", RES_POWER, 1)],
            },
        ],
        "initial_blocked": [5],
        "initial_resources": {RES_METAL: 0, RES_POWER: 0, RES_FUEL: 0},
        "move_limit": 18,
        "goal_machines": [5],
    },
    {
        "machines": [
            {
                "id": 0,
                "row": 0,
                "col": 3,
                "mtype": MTYPE_PRODUCER,
                "deps": [],
                "res_req": {},
                "res_prod": {RES_METAL: 2},
                "effects": [],
            },
            {
                "id": 1,
                "row": 2,
                "col": 1,
                "mtype": MTYPE_LOCK,
                "deps": [0],
                "res_req": {},
                "res_prod": {},
                "effects": [("add_dep", 9, 7), ("block", 4, None)],
            },
            {
                "id": 2,
                "row": 4,
                "col": 1,
                "mtype": MTYPE_CONVERTER,
                "deps": [1],
                "res_req": {RES_METAL: 1},
                "res_prod": {RES_FUEL: 1},
                "effects": [],
            },
            {
                "id": 3,
                "row": 2,
                "col": 5,
                "mtype": MTYPE_PRODUCER,
                "deps": [0],
                "res_req": {},
                "res_prod": {RES_POWER: 2},
                "effects": [],
            },
            {
                "id": 4,
                "row": 4,
                "col": 3,
                "mtype": MTYPE_CONVERTER,
                "deps": [0],
                "res_req": {RES_METAL: 1},
                "res_prod": {RES_FUEL: 1},
                "effects": [("add_dep", 9, 4)],
            },
            {
                "id": 5,
                "row": 4,
                "col": 5,
                "mtype": MTYPE_DISTRIBUTOR,
                "deps": [3],
                "res_req": {RES_POWER: 1},
                "res_prod": {},
                "effects": [
                    ("remove_dep", 9, 1),
                    ("remove_dep", 9, 2),
                    ("unblock", 9, None),
                ],
            },
            {
                "id": 6,
                "row": 2,
                "col": 3,
                "mtype": MTYPE_LOCK,
                "deps": [0],
                "res_req": {},
                "res_prod": {},
                "effects": [("consume", RES_POWER, 2), ("block", 3, None)],
            },
            {
                "id": 7,
                "row": 6,
                "col": 1,
                "mtype": MTYPE_CONVERTER,
                "deps": [2],
                "res_req": {RES_FUEL: 1},
                "res_prod": {RES_POWER: 1},
                "effects": [],
            },
            {
                "id": 8,
                "row": 6,
                "col": 5,
                "mtype": MTYPE_LOCK,
                "deps": [3],
                "res_req": {},
                "res_prod": {},
                "effects": [("block", 7, None), ("add_dep", 9, 8)],
            },
            {
                "id": 9,
                "row": 7,
                "col": 3,
                "mtype": MTYPE_ASSEMBLER,
                "deps": [1, 2, 5],
                "res_req": {RES_METAL: 1, RES_POWER: 1},
                "res_prod": {},
                "effects": [],
            },
        ],
        "initial_blocked": [9],
        "initial_resources": {RES_METAL: 0, RES_POWER: 0, RES_FUEL: 0},
        "move_limit": 15,
        "goal_machines": [9],
    },
    {
        "machines": [
            {
                "id": 0,
                "row": 0,
                "col": 1,
                "mtype": MTYPE_PRODUCER,
                "deps": [],
                "res_req": {},
                "res_prod": {RES_METAL: 2},
                "effects": [],
            },
            {
                "id": 1,
                "row": 0,
                "col": 5,
                "mtype": MTYPE_PRODUCER,
                "deps": [],
                "res_req": {},
                "res_prod": {RES_POWER: 2},
                "effects": [],
            },
            {
                "id": 2,
                "row": 2,
                "col": 3,
                "mtype": MTYPE_CONVERTER,
                "deps": [0],
                "res_req": {RES_METAL: 1},
                "res_prod": {RES_FUEL: 2},
                "effects": [],
            },
            {
                "id": 3,
                "row": 2,
                "col": 1,
                "mtype": MTYPE_LOCK,
                "deps": [0],
                "res_req": {},
                "res_prod": {},
                "effects": [("block", 4, None), ("block", 6, None)],
            },
            {
                "id": 4,
                "row": 2,
                "col": 5,
                "mtype": MTYPE_CONVERTER,
                "deps": [1],
                "res_req": {RES_POWER: 1},
                "res_prod": {RES_FUEL: 1},
                "effects": [],
            },
            {
                "id": 5,
                "row": 6,
                "col": 5,
                "mtype": MTYPE_CONVERTER,
                "deps": [10],
                "res_req": {RES_FUEL: 1, RES_POWER: 1},
                "res_prod": {RES_METAL: 1},
                "effects": [],
            },
            {
                "id": 6,
                "row": 4,
                "col": 1,
                "mtype": MTYPE_CONVERTER,
                "deps": [2],
                "res_req": {RES_FUEL: 1},
                "res_prod": {RES_POWER: 1},
                "effects": [],
            },
            {
                "id": 7,
                "row": 0,
                "col": 3,
                "mtype": MTYPE_LOCK,
                "deps": [],
                "res_req": {},
                "res_prod": {},
                "effects": [("block", 2, None), ("consume", RES_METAL, 1)],
            },
            {
                "id": 8,
                "row": 4,
                "col": 5,
                "mtype": MTYPE_LOCK,
                "deps": [4],
                "res_req": {RES_FUEL: 1},
                "res_prod": {},
                "effects": [("block", 10, None), ("consume", RES_FUEL, 1)],
            },
            {
                "id": 9,
                "row": 4,
                "col": 3,
                "mtype": MTYPE_CONVERTER,
                "deps": [2],
                "res_req": {RES_FUEL: 1},
                "res_prod": {RES_METAL: 1},
                "effects": [("consume", RES_POWER, 1)],
            },
            {
                "id": 10,
                "row": 6,
                "col": 1,
                "mtype": MTYPE_DISTRIBUTOR,
                "deps": [6],
                "res_req": {},
                "res_prod": {},
                "effects": [
                    ("remove_dep", 11, 8),
                    ("remove_dep", 11, 9),
                    ("unblock", 11, None),
                    ("produce", RES_POWER, 1),
                ],
            },
            {
                "id": 11,
                "row": 7,
                "col": 3,
                "mtype": MTYPE_ASSEMBLER,
                "deps": [3, 5, 8, 9],
                "res_req": {RES_METAL: 1, RES_FUEL: 1},
                "res_prod": {},
                "effects": [],
            },
        ],
        "initial_blocked": [11],
        "initial_resources": {RES_METAL: 0, RES_POWER: 0, RES_FUEL: 0},
        "move_limit": 27,
        "goal_machines": [11],
    },
    {
        "machines": [
            {
                "id": 0,
                "row": 0,
                "col": 0,
                "mtype": MTYPE_PRODUCER,
                "deps": [],
                "res_req": {},
                "res_prod": {RES_METAL: 2},
                "effects": [],
            },
            {
                "id": 1,
                "row": 2,
                "col": 0,
                "mtype": MTYPE_LOCK,
                "deps": [0],
                "res_req": {},
                "res_prod": {},
                "effects": [("block", 5, None), ("block", 7, None)],
            },
            {
                "id": 2,
                "row": 0,
                "col": 6,
                "mtype": MTYPE_PRODUCER,
                "deps": [],
                "res_req": {},
                "res_prod": {RES_POWER: 2},
                "effects": [],
            },
            {
                "id": 3,
                "row": 2,
                "col": 6,
                "mtype": MTYPE_LOCK,
                "deps": [2],
                "res_req": {},
                "res_prod": {},
                "effects": [("block", 8, None), ("add_dep", 12, 11)],
            },
            {
                "id": 4,
                "row": 4,
                "col": 0,
                "mtype": MTYPE_CONVERTER,
                "deps": [1],
                "res_req": {RES_METAL: 1},
                "res_prod": {RES_FUEL: 2},
                "effects": [],
            },
            {
                "id": 5,
                "row": 2,
                "col": 3,
                "mtype": MTYPE_CONVERTER,
                "deps": [0],
                "res_req": {RES_METAL: 1},
                "res_prod": {RES_FUEL: 1},
                "effects": [],
            },
            {
                "id": 6,
                "row": 6,
                "col": 3,
                "mtype": MTYPE_CONVERTER,
                "deps": [10],
                "res_req": {RES_FUEL: 1, RES_POWER: 1},
                "res_prod": {RES_METAL: 1},
                "effects": [("remove_dep", 12, 3)],
            },
            {
                "id": 7,
                "row": 2,
                "col": 5,
                "mtype": MTYPE_CONVERTER,
                "deps": [2],
                "res_req": {RES_POWER: 1},
                "res_prod": {RES_FUEL: 1},
                "effects": [],
            },
            {
                "id": 8,
                "row": 4,
                "col": 6,
                "mtype": MTYPE_DISTRIBUTOR,
                "deps": [4],
                "res_req": {RES_FUEL: 1},
                "res_prod": {},
                "effects": [("unblock", 12, None), ("produce", RES_POWER, 1)],
            },
            {
                "id": 9,
                "row": 0,
                "col": 3,
                "mtype": MTYPE_LOCK,
                "deps": [],
                "res_req": {},
                "res_prod": {},
                "effects": [("block", 6, None), ("consume", RES_METAL, 2)],
            },
            {
                "id": 10,
                "row": 4,
                "col": 3,
                "mtype": MTYPE_DISTRIBUTOR,
                "deps": [3, 8],
                "res_req": {},
                "res_prod": {},
                "effects": [("remove_dep", 12, 9), ("produce", RES_FUEL, 1)],
            },
            {
                "id": 11,
                "row": 6,
                "col": 6,
                "mtype": MTYPE_LOCK,
                "deps": [7],
                "res_req": {RES_FUEL: 1},
                "res_prod": {},
                "effects": [("block", 10, None), ("consume", RES_FUEL, 1)],
            },
            {
                "id": 12,
                "row": 7,
                "col": 3,
                "mtype": MTYPE_ASSEMBLER,
                "deps": [1, 3, 6, 9],
                "res_req": {RES_METAL: 1, RES_FUEL: 1},
                "res_prod": {},
                "effects": [],
            },
        ],
        "initial_blocked": [12],
        "initial_resources": {RES_METAL: 0, RES_POWER: 0, RES_FUEL: 0},
        "move_limit": 36,
        "goal_machines": [12],
    },
    {
        "machines": [
            {
                "id": 0,
                "row": 0,
                "col": 0,
                "mtype": MTYPE_PRODUCER,
                "deps": [],
                "res_req": {},
                "res_prod": {RES_METAL: 3},
                "effects": [],
            },
            {
                "id": 1,
                "row": 0,
                "col": 7,
                "mtype": MTYPE_PRODUCER,
                "deps": [],
                "res_req": {},
                "res_prod": {RES_POWER: 3},
                "effects": [],
            },
            {
                "id": 2,
                "row": 2,
                "col": 0,
                "mtype": MTYPE_LOCK,
                "deps": [0],
                "res_req": {},
                "res_prod": {},
                "effects": [("block", 6, None), ("block", 7, None), ("block", 8, None)],
            },
            {
                "id": 3,
                "row": 2,
                "col": 3,
                "mtype": MTYPE_DISTRIBUTOR,
                "deps": [0, 1],
                "res_req": {RES_METAL: 1},
                "res_prod": {},
                "effects": [
                    ("remove_dep", 13, 2),
                    ("unblock", 6, None),
                    ("unblock", 7, None),
                    ("unblock", 8, None),
                    ("produce", RES_FUEL, 1),
                ],
            },
            {
                "id": 4,
                "row": 2,
                "col": 7,
                "mtype": MTYPE_LOCK,
                "deps": [1],
                "res_req": {},
                "res_prod": {},
                "effects": [("add_dep", 13, 4), ("consume", RES_FUEL, 1)],
            },
            {
                "id": 5,
                "row": 6,
                "col": 0,
                "mtype": MTYPE_CONVERTER,
                "deps": [8],
                "res_req": {RES_FUEL: 1},
                "res_prod": {RES_POWER: 2},
                "effects": [("remove_dep", 13, 11)],
            },
            {
                "id": 6,
                "row": 4,
                "col": 0,
                "mtype": MTYPE_CONVERTER,
                "deps": [3],
                "res_req": {RES_METAL: 1},
                "res_prod": {RES_FUEL: 2},
                "effects": [("unblock", 10, None)],
            },
            {
                "id": 7,
                "row": 7,
                "col": 1,
                "mtype": MTYPE_CONVERTER,
                "deps": [5, 12],
                "res_req": {RES_POWER: 1},
                "res_prod": {RES_METAL: 1},
                "effects": [("remove_dep", 13, 4)],
            },
            {
                "id": 8,
                "row": 4,
                "col": 7,
                "mtype": MTYPE_CONVERTER,
                "deps": [3],
                "res_req": {RES_POWER: 1},
                "res_prod": {RES_FUEL: 1},
                "effects": [],
            },
            {
                "id": 9,
                "row": 0,
                "col": 3,
                "mtype": MTYPE_PRODUCER,
                "deps": [],
                "res_req": {},
                "res_prod": {RES_FUEL: 1},
                "effects": [("add_dep", 13, 9), ("consume", RES_METAL, 1)],
            },
            {
                "id": 10,
                "row": 4,
                "col": 3,
                "mtype": MTYPE_CONVERTER,
                "deps": [6],
                "res_req": {RES_FUEL: 1},
                "res_prod": {RES_POWER: 1},
                "effects": [],
            },
            {
                "id": 11,
                "row": 6,
                "col": 7,
                "mtype": MTYPE_LOCK,
                "deps": [8],
                "res_req": {RES_FUEL: 1},
                "res_prod": {},
                "effects": [("block", 12, None), ("consume", RES_FUEL, 1)],
            },
            {
                "id": 12,
                "row": 6,
                "col": 3,
                "mtype": MTYPE_DISTRIBUTOR,
                "deps": [10],
                "res_req": {},
                "res_prod": {},
                "effects": [
                    ("unblock", 13, None),
                    ("remove_dep", 13, 11),
                    ("produce", RES_METAL, 1),
                ],
            },
            {
                "id": 13,
                "row": 7,
                "col": 5,
                "mtype": MTYPE_ASSEMBLER,
                "deps": [2, 4, 7, 11],
                "res_req": {RES_METAL: 1, RES_FUEL: 1, RES_POWER: 1},
                "res_prod": {},
                "effects": [],
            },
        ],
        "initial_blocked": [6, 7, 8, 10, 13],
        "initial_resources": {RES_METAL: 0, RES_POWER: 0, RES_FUEL: 0},
        "move_limit": 36,
        "goal_machines": [13],
    },
]

NUM_LEVELS = len(LEVEL_DEFS)

LIVES_PER_LEVEL = [5, 5, 4, 3, 3]

LIFE_ON = LIGHT_BLUE
LIFE_OFF = YELLOW


def _deep_copy_dict(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = _deep_copy_dict(v)
        elif isinstance(v, list):
            out[k] = list(v)
        else:
            out[k] = v
    return out


def _deep_copy_list_of_lists(lst):
    return [list(row) for row in lst]


def _build_machine_grid(level_def):
    grid = [[-1] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    for m in level_def["machines"]:
        r, c = m["row"], m["col"]
        if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
            grid[r][c] = m["id"]
    return grid


def _build_machine_types(level_def):
    types = {}
    for m in level_def["machines"]:
        types[m["id"]] = m["mtype"]
    return types


def _build_initial_deps(level_def):
    deps = {}
    for m in level_def["machines"]:
        deps[m["id"]] = list(m["deps"])
    return deps


def _build_initial_states(level_def, deps):
    states = {}
    blocked = set(level_def.get("initial_blocked", []))
    for m in level_def["machines"]:
        mid = m["id"]
        if mid in blocked:
            states[mid] = STATE_BLOCKED
        elif len(deps.get(mid, [])) == 0:
            states[mid] = STATE_READY
        else:
            states[mid] = STATE_LOCKED
    return states


def _build_res_reqs(level_def):
    reqs = {}
    for m in level_def["machines"]:
        reqs[m["id"]] = dict(m.get("res_req", {}))
    return reqs


def _build_res_prods(level_def):
    prods = {}
    for m in level_def["machines"]:
        prods[m["id"]] = dict(m.get("res_prod", {}))
    return prods


def _build_effects(level_def):
    effs = {}
    for m in level_def["machines"]:
        effs[m["id"]] = list(m.get("effects", []))
    return effs


def _build_initial_resources(level_def):
    res = {}
    init = level_def.get("initial_resources", {})
    for i in range(NUM_RES_TYPES):
        res[i] = init.get(i, 0)
    return res


def _update_readiness(states, deps, blocked_set):
    changed = True
    while changed:
        changed = False
        for mid in list(states.keys()):
            if states[mid] == STATE_LOCKED:
                dep_list = deps.get(mid, [])
                all_met = True
                for d in dep_list:
                    if states.get(d, STATE_LOCKED) != STATE_ACTIVE:
                        all_met = False
                        break
                if all_met and mid not in blocked_set:
                    states[mid] = STATE_READY
                    changed = True


def _check_resources(resources, req):
    for rtype, amount in req.items():
        if resources.get(rtype, 0) < amount:
            return False
    return True


def _consume_resources(resources, req):
    for rtype, amount in req.items():
        resources[rtype] = resources.get(rtype, 0) - amount


def _produce_resources(resources, prod):
    for rtype, amount in prod.items():
        resources[rtype] = resources.get(rtype, 0) + amount


def _all_goals_active(states, goal_machines):
    for mid in goal_machines:
        if states.get(mid, STATE_LOCKED) != STATE_ACTIVE:
            return False
    return True



def _machine_center_px(row, col):
    px = OFFSET_X + col * TILE_PX + TILE_PX // 2
    py = OFFSET_Y + row * TILE_PX + TILE_PX // 2
    return px, py


def _draw_line_segment(frame, start, end, step, axis, fixed, col, set_pixel_fn):
    if (step > 0 and start > end) or (step < 0 and start < end):
        return
    pos = start
    while True:
        if axis == "v":
            set_pixel_fn(frame, fixed, pos, col)
        else:
            set_pixel_fn(frame, pos, fixed, col)
        if pos == end:
            break
        pos += step


def _dep_line_color(dep_state):
    if dep_state == STATE_ACTIVE:
        return LIME
    if dep_state == STATE_BLOCKED:
        return LIGHT_BLUE
    return COL_DEP_LINE


def _draw_dep_lines(frame, machines_by_id, deps, states, set_pixel_fn):
    margin = 4
    for mid, dep_list in deps.items():
        if mid not in machines_by_id:
            continue
        tr, tc = machines_by_id[mid]
        tx, ty = _machine_center_px(tr, tc)

        for dep_id in dep_list:
            if dep_id not in machines_by_id:
                continue
            sr, sc = machines_by_id[dep_id]
            sx, sy = _machine_center_px(sr, sc)

            col = _dep_line_color(states.get(dep_id, STATE_LOCKED))
            same_col = sx == tx
            same_row = sy == ty

            if not same_row:
                y_step = 1 if ty > sy else -1
                y_start = sy + margin * y_step
                y_end = (ty - margin * y_step) if same_col else ty
                _draw_line_segment(
                    frame, y_start, y_end, y_step, "v", sx, col, set_pixel_fn
                )

            if not same_col:
                x_step = 1 if tx > sx else -1
                x_start = (sx + margin * x_step) if same_row else sx
                x_end = tx - margin * x_step
                _draw_line_segment(
                    frame, x_start, x_end, x_step, "h", ty, col, set_pixel_fn
                )


class GameHUD(RenderableUserDisplay):
    def __init__(self, game: "Eg03"):
        self._game = game

    @staticmethod
    def _set_pixel(frame, x, y, color):
        if 0 <= x < GRID_W and 0 <= y < GRID_H:
            frame[y, x] = color

    @staticmethod
    def _fill_rect(frame, x, y, w, h, color):
        for dy in range(h):
            for dx in range(w):
                fx, fy = x + dx, y + dy
                if 0 <= fx < GRID_W and 0 <= fy < GRID_H:
                    frame[fy, fx] = color

    def _draw_machine(self, frame, px, py, state, mtype, mid):
        inner = TILE_PX - 1

        if state == STATE_ACTIVE:
            fill_col = LIME
        elif state == STATE_BLOCKED:
            fill_col = YELLOW
        elif state == STATE_READY:
            fill_col = MTYPE_COLORS.get(mtype, DARK_RED)
        else:
            fill_col = YELLOW

        border_col = MTYPE_COLORS.get(mtype, GREEN)
        if state == STATE_BLOCKED:
            border_col = LIGHT_BLUE

        for dy in range(TILE_PX):
            for dx in range(TILE_PX):
                fx, fy = px + dx, py + dy
                if 0 <= fx < GRID_W and 0 <= fy < GRID_H:
                    if dx == 0 or dy == 0 or dx == TILE_PX - 1 or dy == TILE_PX - 1:
                        frame[fy, fx] = border_col
                    else:
                        frame[fy, fx] = fill_col

    def _draw_cursor(self, frame, crow, ccol):
        px = OFFSET_X + ccol * TILE_PX
        py = OFFSET_Y + crow * TILE_PX

        for dx in range(TILE_PX):
            self._set_pixel(frame, px + dx, py, COL_CURSOR)
            self._set_pixel(frame, px + dx, py + TILE_PX - 1, COL_CURSOR)
        for dy in range(TILE_PX):
            self._set_pixel(frame, px, py + dy, COL_CURSOR)
            self._set_pixel(frame, px + TILE_PX - 1, py + dy, COL_CURSOR)

        for cx, cy in [
            (px, py),
            (px + TILE_PX - 1, py),
            (px, py + TILE_PX - 1),
            (px + TILE_PX - 1, py + TILE_PX - 1),
        ]:
            self._set_pixel(frame, cx, cy, COL_CURSOR_DOT)

    def _draw_level_indicator(self, frame, current_level, total_levels):
        dot_y = 1
        start_x = 2
        for i in range(total_levels):
            x = start_x + i * 3
            col = LIME if i < current_level else YELLOW
            self._set_pixel(frame, x, dot_y, col)
            if x + 1 < GRID_W:
                self._set_pixel(frame, x + 1, dot_y, col)

    def _draw_lives(self, frame, lives, lives_max):
        dot_y = 3
        start_x = 2
        lost = lives_max - lives
        for i in range(lives_max):
            col = LIFE_OFF if i < lost else LIFE_ON
            x = start_x + i * 3
            self._set_pixel(frame, x, dot_y, col)
            if x + 1 < GRID_W:
                self._set_pixel(frame, x + 1, dot_y, col)

    def _draw_move_counter(self, frame, moves_used, move_limit):
        remaining = move_limit - moves_used
        fraction = remaining / move_limit if move_limit > 0 else 0

        bar_max_w = 20
        bar_h = 3
        bar_x = GRID_W - bar_max_w - 1
        bar_y = 0

        self._fill_rect(frame, bar_x, bar_y, bar_max_w, bar_h, YELLOW)

        filled_w = max(0, min(bar_max_w, int(fraction * bar_max_w + 0.5)))

        if fraction > 0.5:
            bar_col = LIME
        elif fraction > 0.25:
            bar_col = BROWN
        elif fraction > 0.1:
            bar_col = PINK
        else:
            bar_col = LIGHT_BLUE

        self._fill_rect(frame, bar_x, bar_y, filled_w, bar_h, bar_col)

    def render_interface(self, frame):
        g = self._game
        if g is None:
            return frame

        level_def = LEVEL_DEFS[g._level_index]

        machines_by_id = {}
        for m in level_def["machines"]:
            machines_by_id[m["id"]] = (m["row"], m["col"])

        _draw_dep_lines(frame, machines_by_id, g._deps, g._states, self._set_pixel)

        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                px = OFFSET_X + c * TILE_PX
                py = OFFSET_Y + r * TILE_PX
                mid = g._machine_grid[r][c]
                if mid >= 0:
                    state = g._states.get(mid, STATE_LOCKED)
                    mtype = g._machine_types.get(mid, MTYPE_PRODUCER)
                    self._draw_machine(frame, px, py, state, mtype, mid)

        if not g._won:
            self._draw_cursor(frame, g._cursor_row, g._cursor_col)

        self._draw_level_indicator(frame, g._level_index + 1, NUM_LEVELS)
        self._draw_lives(frame, g._lives, g._lives_max)

        if not g._won:
            self._draw_move_counter(frame, g._moves_used, level_def["move_limit"])

        return frame


sprites = {
    "marker": Sprite(
        pixels=[[GREY]],
        name="marker",
        visible=False,
        collidable=False,
        tags=["marker"],
        layer=0,
    ),
}


def _build_levels():
    out = []
    for i in range(NUM_LEVELS):
        out.append(
            Level(
                sprites=[sprites["marker"].clone().set_position(0, 0)],
                grid_size=(64, 64),
                data={"idx": i},
                name="Level " + str(i + 1),
            )
        )
    return out


class Eg03(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)
        self._hud = GameHUD(self)

        self._level_index = 0
        self._machine_grid = [[-1] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        self._machine_types = {}
        self._states = {}
        self._deps = {}
        self._res_reqs = {}
        self._res_prods = {}
        self._effects = {}
        self._resources = {}
        self._blocked_set = set()
        self._cursor_row = 0
        self._cursor_col = 0
        self._won = False
        self._moves_used = 0
        self._history = []
        self._lives = 5
        self._lives_max = 5
        self._consecutive_resets = 0
        self._game_complete = False
        super().__init__(
            "eg03",
            _build_levels(),
            Camera(
                0,
                0,
                width=64,
                height=64,
                background=GREY,
                letter_box=GREY,
                interfaces=[self._hud],
            ),
            available_actions=[0, 1, 2, 3, 4, 5, 6, 7],
        )

    def on_set_level(self, level):
        idx = self.current_level.get_data("idx")
        if idx is None:
            idx = 0

        self._level_index = idx
        self._init_level_state(idx)

        self._lives_max = LIVES_PER_LEVEL[idx] if idx < len(LIVES_PER_LEVEL) else 3
        self._lives = self._lives_max

    def full_reset(self):
        self._consecutive_resets = 0
        self._game_complete = False
        super().full_reset()

    def handle_reset(self):
        if self._game_complete:
            self.full_reset()
            return
        self._consecutive_resets += 1
        if self._consecutive_resets >= 2:
            self._consecutive_resets = 0
            self.full_reset()
        else:
            self.level_reset()

    def level_reset(self):
        saved_lives = self._lives
        saved_lives_max = self._lives_max
        super().level_reset()
        if saved_lives > 0:
            self._lives = saved_lives
            self._lives_max = saved_lives_max

    def _init_level_state(self, idx):
        level_def = LEVEL_DEFS[idx]
        self._machine_grid = _build_machine_grid(level_def)
        self._machine_types = _build_machine_types(level_def)
        self._deps = _build_initial_deps(level_def)
        self._res_reqs = _build_res_reqs(level_def)
        self._res_prods = _build_res_prods(level_def)
        self._effects = _build_effects(level_def)
        self._resources = _build_initial_resources(level_def)
        self._blocked_set = set(level_def.get("initial_blocked", []))
        self._states = _build_initial_states(level_def, self._deps)

        if len(level_def["machines"]) > 0:
            m = self._rng.choice(level_def["machines"])
            self._cursor_row = m["row"]
            self._cursor_col = m["col"]
        else:
            self._cursor_row = 0
            self._cursor_col = 0

        self._won = False
        self._moves_used = 0
        self._history = []

    def _save_snapshot(self):
        snapshot = {
            "machine_grid": _deep_copy_list_of_lists(self._machine_grid),
            "machine_types": dict(self._machine_types),
            "states": dict(self._states),
            "deps": _deep_copy_dict(self._deps),
            "res_reqs": _deep_copy_dict(self._res_reqs),
            "res_prods": _deep_copy_dict(self._res_prods),
            "effects": _deep_copy_dict(self._effects),
            "resources": dict(self._resources),
            "blocked_set": set(self._blocked_set),
            "cursor_row": self._cursor_row,
            "cursor_col": self._cursor_col,
            "moves_used": self._moves_used,
        }
        self._history.append(snapshot)

    def _restore_snapshot(self):
        if not self._history:
            return False
        snap = self._history.pop()
        self._machine_grid = snap["machine_grid"]
        self._machine_types = snap["machine_types"]
        self._states = snap["states"]
        self._deps = snap["deps"]
        self._res_reqs = snap["res_reqs"]
        self._res_prods = snap["res_prods"]
        self._effects = snap["effects"]
        self._resources = snap["resources"]
        self._blocked_set = snap["blocked_set"]
        self._cursor_row = snap["cursor_row"]
        self._cursor_col = snap["cursor_col"]
        self._moves_used = snap["moves_used"]
        return True

    def _reset_current_level(self):
        self._init_level_state(self._level_index)

    def _lose_life(self):
        self._lives -= 1
        if self._lives <= 0:
            self.lose()
            return True
        self._reset_current_level()
        return False

    def _apply_effects(self, mid):
        effs = self._effects.get(mid, [])
        for eff in effs:
            etype = eff[0]
            target = eff[1]
            param = eff[2] if len(eff) > 2 else None

            if etype == "add_dep":
                dep_list = self._deps.get(target, [])
                if param not in dep_list:
                    dep_list.append(param)
                    self._deps[target] = dep_list
                if self._states.get(target, STATE_LOCKED) == STATE_READY:
                    if self._states.get(param, STATE_LOCKED) != STATE_ACTIVE:
                        self._states[target] = STATE_LOCKED

            elif etype == "remove_dep":
                dep_list = self._deps.get(target, [])
                if param in dep_list:
                    dep_list.remove(param)
                    self._deps[target] = dep_list

            elif etype == "block":
                self._blocked_set.add(target)
                if self._states.get(target, STATE_LOCKED) != STATE_ACTIVE:
                    self._states[target] = STATE_BLOCKED

            elif etype == "unblock":
                if target in self._blocked_set:
                    self._blocked_set.discard(target)
                if self._states.get(target, STATE_LOCKED) == STATE_BLOCKED:
                    self._states[target] = STATE_LOCKED

            elif etype == "produce":
                self._resources[target] = self._resources.get(target, 0) + param

            elif etype == "consume":
                self._resources[target] = max(0, self._resources.get(target, 0) - param)

    def _try_activate(self, row, col):
        mid = self._machine_grid[row][col]
        if mid < 0:
            return "nothing"

        state = self._states.get(mid, STATE_LOCKED)

        if state == STATE_ACTIVE:
            return "nothing"

        if state == STATE_BLOCKED:
            return "error"

        if state == STATE_LOCKED:
            return "error"

        if state == STATE_READY:
            req = self._res_reqs.get(mid, {})
            if not _check_resources(self._resources, req):
                return "error"

            _consume_resources(self._resources, req)

            prod = self._res_prods.get(mid, {})
            _produce_resources(self._resources, prod)

            self._states[mid] = STATE_ACTIVE

            self._apply_effects(mid)

            _update_readiness(self._states, self._deps, self._blocked_set)

            return "success"

        return "nothing"

    def _process_undo(self, move_limit):
        self._consecutive_resets = 0
        if self._moves_used < move_limit:
            current_moves = self._moves_used + 1
            self._restore_snapshot()
            self._moves_used = current_moves
            if self._moves_used >= move_limit:
                self._lose_life()

    def _process_action(self, action, move_limit):
        clicked = False
        if action == GameAction.ACTION1:
            self._cursor_row = max(0, min(self._cursor_row - 1, BOARD_SIZE - 1))
        elif action == GameAction.ACTION2:
            self._cursor_row = max(0, min(self._cursor_row + 1, BOARD_SIZE - 1))
        elif action == GameAction.ACTION3:
            self._cursor_col = max(0, min(self._cursor_col - 1, BOARD_SIZE - 1))
        elif action == GameAction.ACTION4:
            self._cursor_col = max(0, min(self._cursor_col + 1, BOARD_SIZE - 1))
        elif action == GameAction.ACTION5:
            if self._moves_used < move_limit:
                result = self._try_activate(self._cursor_row, self._cursor_col)
                if result in ("success", "error"):
                    self._moves_used += 1
                    clicked = True
        elif action == GameAction.ACTION6:
            data = self.action.data if self.action and hasattr(self.action, "data") and self.action.data else {}
            px = data.get("x", 0)
            py = data.get("y", 0)
            col = (px - OFFSET_X) // TILE_PX
            row = (py - OFFSET_Y) // TILE_PX
            if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
                row = self._cursor_row
                col = self._cursor_col
            self._cursor_row = row
            self._cursor_col = col
            if self._moves_used < move_limit:
                result = self._try_activate(row, col)
                if result in ("success", "error"):
                    self._moves_used += 1
                    clicked = True
        return clicked

    def _check_level_completion(self, clicked, move_limit):
        goal_machines = LEVEL_DEFS[self._level_index].get("goal_machines", [])
        if len(goal_machines) > 0:
            if _all_goals_active(self._states, goal_machines):
                self._won = True
                if self._level_index >= NUM_LEVELS - 1:
                    self._game_complete = True
                self.next_level()
        else:
            all_active = True
            for mid, st in self._states.items():
                if st != STATE_ACTIVE:
                    all_active = False
                    break
            if all_active:
                self._won = True
                if self._level_index >= NUM_LEVELS - 1:
                    self._game_complete = True
                self.next_level()
        if not self._won and clicked and self._moves_used >= move_limit:
            self._lose_life()

    def step(self):
        if self._won:
            self.complete_action()
            return

        action = self.action.id

        if action == GameAction.RESET:
            self.complete_action()
            return

        level_def = LEVEL_DEFS[self._level_index]
        move_limit = level_def["move_limit"]

        if action == GameAction.ACTION7:
            self._process_undo(move_limit)
            self.complete_action()
            return

        self._consecutive_resets = 0
        self._save_snapshot()
        clicked = self._process_action(action, move_limit)
        self._check_level_completion(clicked, move_limit)
        self.complete_action()


class PuzzleEnvironment:
    _ACTION_MAP: Dict[str, GameAction] = {
        "reset": GameAction.RESET,
        "up": GameAction.ACTION1,
        "down": GameAction.ACTION2,
        "left": GameAction.ACTION3,
        "right": GameAction.ACTION4,
        "select": GameAction.ACTION5,
        "click": GameAction.ACTION6,
        "undo": GameAction.ACTION7,
    }

    _VALID_ACTIONS: List[str] = [
        "reset",
        "up",
        "down",
        "left",
        "right",
        "select",
        "click",
        "undo",
    ]

    def __init__(self, seed: int = 0) -> None:
        self._engine = Eg03(seed=seed)
        self._total_turns = 0
        self._last_action_was_reset = False
        self._game_won = False

    def _format_board_rows(self, e):
        state_chars = {
            STATE_LOCKED: "L",
            STATE_READY: "R",
            STATE_ACTIVE: "A",
            STATE_BLOCKED: "X",
        }
        mtype_chars = {
            MTYPE_PRODUCER: "P",
            MTYPE_CONVERTER: "C",
            MTYPE_DISTRIBUTOR: "D",
            MTYPE_LOCK: "K",
            MTYPE_ASSEMBLER: "G",
        }
        rows = []
        for r in range(BOARD_SIZE):
            row_parts = []
            for c in range(BOARD_SIZE):
                mid = e._machine_grid[r][c]
                if mid < 0:
                    cell = " .. "
                else:
                    mt = mtype_chars.get(e._machine_types.get(mid, 0), "?")
                    st = state_chars.get(e._states.get(mid, STATE_LOCKED), "?")
                    cell = mt + str(mid)
                    if len(cell) < 3:
                        cell = cell + " "
                    cell = cell + st
                cursor = "*" if r == e._cursor_row and c == e._cursor_col else " "
                row_parts.append(cursor + cell)
            rows.append("|".join(row_parts))
        return rows

    def _format_dep_parts(self, e, level_def):
        dep_parts = []
        for m in level_def["machines"]:
            mid = m["id"]
            dep_list = e._deps.get(mid, [])
            if dep_list:
                dep_parts.append(
                    str(mid) + "<-[" + ",".join(str(d) for d in dep_list) + "]"
                )
        return dep_parts

    def _build_text_observation(self) -> str:
        e = self._engine
        level_def = LEVEL_DEFS[e._level_index]
        move_limit = level_def["move_limit"]
        lines = []
        lines.append(
            "Level:"
            + str(e._level_index + 1)
            + "/"
            + str(NUM_LEVELS)
            + " Lives:"
            + str(e._lives)
            + "/"
            + str(e._lives_max)
            + " Moves:"
            + str(e._moves_used)
            + "/"
            + str(move_limit)
        )
        res_names = {RES_METAL: "Metal", RES_POWER: "Power", RES_FUEL: "Fuel"}
        res_parts = []
        for rtype in range(NUM_RES_TYPES):
            name = res_names.get(rtype, str(rtype))
            count = e._resources.get(rtype, 0)
            res_parts.append(name + ":" + str(count))
        lines.append("Resources: " + " ".join(res_parts))
        lines.append("Board (8x8):")
        lines.extend(self._format_board_rows(e))
        lines.append("Cursor: row=" + str(e._cursor_row) + " col=" + str(e._cursor_col))
        dep_parts = self._format_dep_parts(e, level_def)
        if dep_parts:
            lines.append("Deps: " + " ".join(dep_parts))
        goal_machines = level_def.get("goal_machines", [])
        if goal_machines:
            lines.append("Goals: " + ",".join(str(g) for g in goal_machines))
        if e._won:
            lines.append("STATUS: WON")
        return "\n".join(lines)

    def _build_image_observation(self) -> Optional[bytes]:
        e = self._engine
        rendered = e.camera.render(e.current_level.get_sprites())
        if hasattr(rendered, "tolist"):
            grid = rendered.tolist()
        else:
            grid = rendered if rendered else []
        if not grid:
            return None
        rgb = _index_to_rgb(grid)
        height, width = 64, 64
        raw_data = b""
        for y in range(height):
            raw_data += b"\x00"
            for x in range(width):
                raw_data += bytes(rgb[y, x])

        def _png_chunk(chunk_type, data):
            c = chunk_type + data
            crc = struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
            return struct.pack(">I", len(data)) + c + crc

        buf = io.BytesIO()
        buf.write(b"\x89PNG\r\n\x1a\n")
        ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
        buf.write(_png_chunk(b"IHDR", ihdr_data))
        compressed = zlib.compress(raw_data)
        buf.write(_png_chunk(b"IDAT", compressed))
        buf.write(_png_chunk(b"IEND", b""))
        return buf.getvalue()

    def _build_game_state(
        self, done: bool = False, info: Optional[Dict] = None
    ) -> GameState:
        e = self._engine
        valid_actions = self._VALID_ACTIONS[:] if not done else None
        return GameState(
            text_observation=self._build_text_observation(),
            image_observation=self._build_image_observation(),
            valid_actions=valid_actions,
            turn=self._total_turns,
            metadata={
                "total_levels": len(self._engine._levels),
                "level_index": e._level_index,
                "lives": e._lives,
                "lives_max": e._lives_max,
                "moves_used": e._moves_used,
                "move_limit": LEVEL_DEFS[e._level_index]["move_limit"],
                "resources": dict(e._resources),
                "won": e._won,
                "done": done,
                "info": info or {},
            },
        )

    def _is_won(self) -> bool:
        e = self._engine
        if e is None:
            return False
        return e._won

    def _is_game_over(self) -> bool:
        e = self._engine
        if e is None:
            return True
        if e._lives <= 0:
            return True
        if hasattr(e, "_state") and hasattr(e._state, "name"):
            return e._state.name == "GAME_OVER"
        return False

    def is_done(self) -> bool:
        return self._is_won() or self._is_game_over()

    def reset(self) -> GameState:
        if self._engine is None:
            raise RuntimeError("Environment is closed")
        self._total_turns = 0
        self._last_action_was_reset = False
        self._game_won = False
        self._engine.perform_action(ActionInput(id=GameAction.RESET))
        return self._build_game_state()

    def get_actions(self) -> List[str]:
        if self._engine is None:
            return []
        if self.is_done():
            return ["reset"]
        return self._VALID_ACTIONS[:]

    def _execute_and_build_result(self, act_key, action):
        e = self._engine
        info: Dict = {"action": action}

        total_levels = len(LEVEL_DEFS)
        level_before = e._level_index

        game_action = self._ACTION_MAP[act_key]
        action_input = ActionInput(id=game_action, data={"x": 0, "y": 0})
        frame = e.perform_action(action_input, raw=True)

        game_won = frame and frame.state and frame.state.name == "WIN"
        level_advanced = e._level_index > level_before
        done = self._is_game_over() or game_won

        reward = 0.0
        if game_won or level_advanced:
            reward = 1.0 / total_levels

        if game_won:
            self._game_won = True
            info["reason"] = "game_complete"
            info["level_index"] = e._level_index
            info["total_levels"] = total_levels
            return StepResult(
                state=self._build_game_state(done=True, info=info),
                reward=reward,
                done=True,
                info=info,
            )

        if self._is_game_over():
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

    def step(self, action: str) -> StepResult:
        if self._engine is None:
            raise RuntimeError("Environment is closed")

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
        return self._execute_and_build_result(action, action)

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError(f"Unsupported render mode: {mode}")
        if self._engine is None:
            return np.zeros((64, 64, 3), dtype=np.uint8)
        e = self._engine
        index_grid = e.camera.render(e.current_level.get_sprites())
        if not hasattr(index_grid, "__len__"):
            return np.zeros((64, 64, 3), dtype=np.uint8)
        return _index_to_rgb(index_grid)

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

    try:
        check_env(env, skip_render_check=False)
        print("[PASS] check_env passed — environment is Gymnasium-compliant.")
    except Exception as e:
        print(f"[FAIL] check_env failed: {e}")

    obs, info = env.reset()
    print(f"  obs shape: {obs.shape}, dtype: {obs.dtype}")
    print(f"  info keys: {list(info.keys())}")

    obs, reward, term, trunc, info = env.step(0)
    print(f"  step → reward={reward}, terminated={term}, truncated={trunc}")

    frame = env.render()
    print(f"  render → shape={frame.shape if frame is not None else None}")

    env.close()
    print("  close() OK")
