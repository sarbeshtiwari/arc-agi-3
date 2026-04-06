import io
import random
import struct
import zlib

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from arcengine import (
    ActionInput,
    ARCBaseGame,
    Camera,
    GameAction,
    Level,
    RenderableUserDisplay,
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

_CELL = 6
_GAP = 1
_GRID_PX = 64
_Y_PAD = 4

BACKGROUND_COLOR = 5
PADDING_COLOR = 4

T_EMPTY = 0
T_PIPE = 1
T_MACHINE = 2
T_GOAL = 3
T_SOURCE = 4
T_WALL = 5
T_SPLITTER = 6
T_SELECTOR = 7
T_LAUNCH = 8

D_RIGHT = 0
D_DOWN = 1
D_LEFT = 2
D_UP = 3
D_BEND_UR = 10

DIR_DR = {
    D_RIGHT: (0, 1),
    D_DOWN: (1, 0),
    D_LEFT: (0, -1),
    D_UP: (-1, 0),
    D_BEND_UR: (0, 1),
}

FLOW_BLOCKERS = (T_WALL, T_EMPTY, T_LAUNCH)

C_RED = 1
C_BLUE = 2
C_GREEN = 3
C_ORANGE = 4

RULE_IDENTITY = 0
RULE_R2B = 1
RULE_B2G = 2
RULE_R2G = 3
RULE_G2O = 4
RULE_B2R = 5

TRANSFORM_TABLE = {
    RULE_IDENTITY: {},
    RULE_R2B: {C_RED: C_BLUE},
    RULE_B2G: {C_BLUE: C_GREEN},
    RULE_R2G: {C_RED: C_GREEN},
    RULE_G2O: {C_GREEN: C_ORANGE},
    RULE_B2R: {C_BLUE: C_RED},
}

RULE_VISUAL = {
    RULE_IDENTITY: (2, 2),
    RULE_R2B: (8, 9),
    RULE_B2G: (9, 14),
    RULE_R2G: (8, 14),
    RULE_G2O: (14, 12),
    RULE_B2R: (9, 8),
}


def _apply_rule(rule: int, color: int) -> int:
    return TRANSFORM_TABLE.get(rule, {}).get(color, color)


sprites = {
    "empty": Sprite(
        pixels=[
            [5, 5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5, 5],
            [5, 5, 4, 4, 5, 5],
            [5, 5, 4, 4, 5, 5],
            [5, 5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5, 5],
        ],
        name="empty",
        visible=True,
        collidable=True,
        tags=["tile", "floor"],
        layer=0,
    ),
    "wall": Sprite(
        pixels=[
            [3, 3, 3, 3, 3, 3],
            [3, 2, 3, 3, 2, 3],
            [3, 3, 4, 4, 3, 3],
            [3, 3, 4, 4, 3, 3],
            [3, 2, 3, 3, 2, 3],
            [3, 3, 3, 3, 3, 3],
        ],
        name="wall",
        visible=True,
        collidable=True,
        tags=["tile", "wall"],
        layer=0,
    ),
    "pipe_r": Sprite(
        pixels=[
            [5, 5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5, 5],
            [4, 4, 4, 4, 0, 4],
            [4, 4, 4, 4, 0, 4],
            [5, 5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5, 5],
        ],
        name="pipe_r",
        visible=True,
        collidable=True,
        tags=["tile", "pipe"],
        layer=0,
    ),
    "pipe_br": Sprite(
        pixels=[
            [5, 5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5, 5],
            [5, 5, 4, 4, 0, 4],
            [5, 5, 4, 4, 0, 4],
            [5, 5, 4, 4, 5, 5],
            [5, 5, 4, 4, 5, 5],
        ],
        name="pipe_br",
        visible=True,
        collidable=True,
        tags=["tile", "pipe"],
        layer=0,
    ),
    "pipe_d": Sprite(
        pixels=[
            [5, 5, 5, 5, 5, 5],
            [5, 5, 4, 4, 5, 5],
            [5, 5, 4, 4, 5, 5],
            [5, 5, 4, 4, 5, 5],
            [5, 5, 0, 0, 5, 5],
            [5, 5, 5, 5, 5, 5],
        ],
        name="pipe_d",
        visible=True,
        collidable=True,
        tags=["tile", "pipe"],
        layer=0,
    ),
    "pipe_l": Sprite(
        pixels=[
            [5, 5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5, 5],
            [4, 0, 4, 4, 4, 4],
            [4, 0, 4, 4, 4, 4],
            [5, 5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5, 5],
        ],
        name="pipe_l",
        visible=True,
        collidable=True,
        tags=["tile", "pipe"],
        layer=0,
    ),
    "pipe_u": Sprite(
        pixels=[
            [5, 5, 5, 5, 5, 5],
            [5, 5, 0, 0, 5, 5],
            [5, 5, 4, 4, 5, 5],
            [5, 5, 4, 4, 5, 5],
            [5, 5, 4, 4, 5, 5],
            [5, 5, 5, 5, 5, 5],
        ],
        name="pipe_u",
        visible=True,
        collidable=True,
        tags=["tile", "pipe"],
        layer=0,
    ),
    "mach": Sprite(
        pixels=[
            [12, 12, 12, 12, 12, 12],
            [12, 5, 5, 5, 5, 12],
            [12, 5, 13, 13, 5, 12],
            [12, 5, 13, 13, 5, 12],
            [12, 5, 5, 5, 5, 12],
            [12, 12, 12, 12, 12, 12],
        ],
        name="mach",
        visible=True,
        collidable=True,
        tags=["tile", "machine"],
        layer=0,
    ),
    "goal": Sprite(
        pixels=[
            [7, 7, 7, 7, 7, 7],
            [7, 5, 5, 5, 5, 7],
            [7, 5, 0, 0, 5, 7],
            [7, 5, 0, 0, 5, 7],
            [7, 5, 5, 5, 5, 7],
            [7, 7, 7, 7, 7, 7],
        ],
        name="goal",
        visible=True,
        collidable=True,
        tags=["tile", "goal"],
        layer=0,
    ),
    "source": Sprite(
        pixels=[
            [8, 8, 8, 8, 8, 8],
            [8, 5, 5, 5, 5, 8],
            [8, 5, 10, 10, 5, 8],
            [8, 5, 10, 10, 5, 8],
            [8, 5, 5, 5, 5, 8],
            [8, 8, 8, 8, 8, 8],
        ],
        name="source",
        visible=True,
        collidable=True,
        tags=["tile", "source"],
        layer=0,
    ),
    "source_blue": Sprite(
        pixels=[
            [9, 9, 9, 9, 9, 9],
            [9, 5, 5, 5, 5, 9],
            [9, 5, 10, 10, 5, 9],
            [9, 5, 10, 10, 5, 9],
            [9, 5, 5, 5, 5, 9],
            [9, 9, 9, 9, 9, 9],
        ],
        name="source_blue",
        visible=True,
        collidable=True,
        tags=["tile", "source"],
        layer=0,
    ),
    "source_green": Sprite(
        pixels=[
            [14, 14, 14, 14, 14, 14],
            [14, 5, 5, 5, 5, 14],
            [14, 5, 10, 10, 5, 14],
            [14, 5, 10, 10, 5, 14],
            [14, 5, 5, 5, 5, 14],
            [14, 14, 14, 14, 14, 14],
        ],
        name="source_green",
        visible=True,
        collidable=True,
        tags=["tile", "source"],
        layer=0,
    ),
    "source_orange": Sprite(
        pixels=[
            [12, 12, 12, 12, 12, 12],
            [12, 5, 5, 5, 5, 12],
            [12, 5, 10, 10, 5, 12],
            [12, 5, 10, 10, 5, 12],
            [12, 5, 5, 5, 5, 12],
            [12, 12, 12, 12, 12, 12],
        ],
        name="source_orange",
        visible=True,
        collidable=True,
        tags=["tile", "source"],
        layer=0,
    ),
    "split": Sprite(
        pixels=[
            [6, 6, 6, 6, 6, 6],
            [6, 5, 5, 5, 5, 6],
            [6, 5, 1, 1, 5, 6],
            [6, 5, 1, 1, 5, 6],
            [6, 5, 5, 5, 5, 6],
            [6, 6, 6, 6, 6, 6],
        ],
        name="split",
        visible=True,
        collidable=True,
        tags=["tile", "splitter"],
        layer=0,
    ),
    "selector": Sprite(
        pixels=[
            [13, 13, 13, 13, 13, 13],
            [13, 5, 5, 5, 5, 13],
            [13, 5, 12, 12, 5, 13],
            [13, 5, 12, 12, 5, 13],
            [13, 5, 5, 5, 5, 13],
            [13, 13, 13, 13, 13, 13],
        ],
        name="selector",
        visible=True,
        collidable=True,
        tags=["tile", "selector"],
        layer=0,
    ),
    "launch": Sprite(
        pixels=[
            [14, 14, 14, 14, 14, 14],
            [14, 5, 5, 5, 5, 14],
            [14, 5, 11, 11, 5, 14],
            [14, 5, 11, 11, 5, 14],
            [14, 5, 5, 5, 5, 14],
            [14, 14, 14, 14, 14, 14],
        ],
        name="launch",
        visible=True,
        collidable=True,
        tags=["tile", "launch"],
        layer=0,
    ),
    "obj_red": Sprite(
        pixels=[
            [-1, -1, 8, 8, -1, -1],
            [-1, 8, 0, 0, 8, -1],
            [8, 0, 0, 0, 0, 8],
            [8, 0, 0, 0, 0, 8],
            [-1, 8, 0, 0, 8, -1],
            [-1, -1, 8, 8, -1, -1],
        ],
        name="obj_red",
        visible=True,
        collidable=False,
        tags=["object"],
        layer=2,
    ),
    "obj_blue": Sprite(
        pixels=[
            [-1, -1, 9, 9, -1, -1],
            [-1, 9, 0, 0, 9, -1],
            [9, 0, 0, 0, 0, 9],
            [9, 0, 0, 0, 0, 9],
            [-1, 9, 0, 0, 9, -1],
            [-1, -1, 9, 9, -1, -1],
        ],
        name="obj_blue",
        visible=True,
        collidable=False,
        tags=["object"],
        layer=2,
    ),
    "obj_green": Sprite(
        pixels=[
            [-1, -1, 14, 14, -1, -1],
            [-1, 14, 0, 0, 14, -1],
            [14, 0, 0, 0, 0, 14],
            [14, 0, 0, 0, 0, 14],
            [-1, 14, 0, 0, 14, -1],
            [-1, -1, 14, 14, -1, -1],
        ],
        name="obj_green",
        visible=True,
        collidable=False,
        tags=["object"],
        layer=2,
    ),
    "obj_orange": Sprite(
        pixels=[
            [-1, -1, 12, 12, -1, -1],
            [-1, 12, 0, 0, 12, -1],
            [12, 0, 0, 0, 0, 12],
            [12, 0, 0, 0, 0, 12],
            [-1, 12, 0, 0, 12, -1],
            [-1, -1, 12, 12, -1, -1],
        ],
        name="obj_orange",
        visible=True,
        collidable=False,
        tags=["object"],
        layer=2,
    ),
    "tgt_red": Sprite(
        pixels=[
            [8, 8, -1, -1, 8, 8],
            [8, -1, -1, -1, -1, 8],
            [-1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1],
            [8, -1, -1, -1, -1, 8],
            [8, 8, -1, -1, 8, 8],
        ],
        name="tgt_red",
        visible=True,
        collidable=False,
        tags=["marker"],
        layer=1,
    ),
    "tgt_blue": Sprite(
        pixels=[
            [9, 9, -1, -1, 9, 9],
            [9, -1, -1, -1, -1, 9],
            [-1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1],
            [9, -1, -1, -1, -1, 9],
            [9, 9, -1, -1, 9, 9],
        ],
        name="tgt_blue",
        visible=True,
        collidable=False,
        tags=["marker"],
        layer=1,
    ),
    "tgt_green": Sprite(
        pixels=[
            [14, 14, -1, -1, 14, 14],
            [14, -1, -1, -1, -1, 14],
            [-1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1],
            [14, -1, -1, -1, -1, 14],
            [14, 14, -1, -1, 14, 14],
        ],
        name="tgt_green",
        visible=True,
        collidable=False,
        tags=["marker"],
        layer=1,
    ),
    "tgt_orange": Sprite(
        pixels=[
            [12, 12, -1, -1, 12, 12],
            [12, -1, -1, -1, -1, 12],
            [-1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1],
            [12, -1, -1, -1, -1, 12],
            [12, 12, -1, -1, 12, 12],
        ],
        name="tgt_orange",
        visible=True,
        collidable=False,
        tags=["marker"],
        layer=1,
    ),
    "cur": Sprite(
        pixels=[
            [11, 11, 11, 11, 11, 11],
            [11, -1, -1, -1, -1, 11],
            [11, -1, -1, -1, -1, 11],
            [11, -1, -1, -1, -1, 11],
            [11, -1, -1, -1, -1, 11],
            [11, 11, 11, 11, 11, 11],
        ],
        name="cur",
        visible=True,
        collidable=False,
        tags=["cursor"],
        layer=4,
    ),
}

OBJ_SPRITE_KEY = {
    C_RED: "obj_red",
    C_BLUE: "obj_blue",
    C_GREEN: "obj_green",
    C_ORANGE: "obj_orange",
}
TARGET_MARKER_KEY = {
    C_RED: "tgt_red",
    C_BLUE: "tgt_blue",
    C_GREEN: "tgt_green",
    C_ORANGE: "tgt_orange",
}
SOURCE_SPRITE_KEY = {
    C_RED: "source",
    C_BLUE: "source_blue",
    C_GREEN: "source_green",
    C_ORANGE: "source_orange",
}
PIPE_SPRITE_KEY = {
    D_RIGHT: "pipe_r",
    D_DOWN: "pipe_d",
    D_LEFT: "pipe_l",
    D_UP: "pipe_u",
    D_BEND_UR: "pipe_br",
}


def _grid_positions(dim: int) -> List[Tuple[int, int]]:
    total = _CELL * dim + _GAP * (dim - 1)
    ox = (_GRID_PX - total) // 2
    oy = (_GRID_PX - total) // 2 + _Y_PAD
    return [
        (ox + c * (_CELL + _GAP), oy + r * (_CELL + _GAP))
        for r in range(dim)
        for c in range(dim)
    ]


_L1_GRID = [
    [T_EMPTY, T_EMPTY, T_EMPTY, T_EMPTY, T_EMPTY],
    [T_EMPTY, T_SOURCE, T_PIPE, T_MACHINE, T_EMPTY],
    [T_EMPTY, T_EMPTY, T_EMPTY, T_PIPE, T_EMPTY],
    [T_EMPTY, T_LAUNCH, T_EMPTY, T_GOAL, T_EMPTY],
    [T_EMPTY, T_EMPTY, T_EMPTY, T_EMPTY, T_EMPTY],
]
_L1_PIPES = {(1, 2): D_RIGHT, (2, 3): D_DOWN}
_L1_MACHINES = {(1, 3): [RULE_IDENTITY, RULE_R2B, RULE_R2G]}
_L1_SOURCES = [(1, 1, C_RED)]
_L1_GOALS = {(3, 3): C_BLUE}
_L1_SPLITTERS: Dict[Tuple[int, int], Tuple[int, int]] = {}
_L1_SELECTORS: Dict[Tuple[int, int], Dict[int, int]] = {}

_L2_GRID = [
    [T_EMPTY, T_EMPTY, T_EMPTY, T_EMPTY, T_EMPTY, T_EMPTY],
    [T_EMPTY, T_SOURCE, T_PIPE, T_MACHINE, T_PIPE, T_MACHINE],
    [T_EMPTY, T_EMPTY, T_EMPTY, T_EMPTY, T_EMPTY, T_PIPE],
    [T_EMPTY, T_LAUNCH, T_EMPTY, T_EMPTY, T_EMPTY, T_PIPE],
    [T_EMPTY, T_EMPTY, T_EMPTY, T_EMPTY, T_EMPTY, T_GOAL],
    [T_EMPTY, T_EMPTY, T_EMPTY, T_EMPTY, T_EMPTY, T_EMPTY],
]
_L2_PIPES = {(1, 2): D_RIGHT, (1, 4): D_RIGHT, (2, 5): D_DOWN, (3, 5): D_DOWN}
_L2_MACHINES = {
    (1, 3): [RULE_IDENTITY, RULE_R2B, RULE_R2G],
    (1, 5): [RULE_IDENTITY, RULE_B2G, RULE_G2O],
}
_L2_SOURCES = [(1, 1, C_RED)]
_L2_GOALS = {(4, 5): C_ORANGE}
_L2_SPLITTERS: Dict[Tuple[int, int], Tuple[int, int]] = {}
_L2_SELECTORS: Dict[Tuple[int, int], Dict[int, int]] = {}

_L3_GRID = [
    [T_EMPTY, T_EMPTY, T_EMPTY, T_EMPTY, T_EMPTY, T_EMPTY],
    [T_EMPTY, T_EMPTY, T_EMPTY, T_EMPTY, T_PIPE, T_GOAL],
    [T_EMPTY, T_SOURCE, T_PIPE, T_MACHINE, T_SPLITTER, T_EMPTY],
    [T_EMPTY, T_LAUNCH, T_EMPTY, T_EMPTY, T_PIPE, T_EMPTY],
    [T_EMPTY, T_EMPTY, T_EMPTY, T_EMPTY, T_MACHINE, T_PIPE],
    [T_EMPTY, T_EMPTY, T_EMPTY, T_EMPTY, T_EMPTY, T_GOAL],
]
_L3_PIPES = {
    (2, 2): D_RIGHT,
    (1, 4): D_BEND_UR,
    (3, 4): D_DOWN,
    (4, 5): D_DOWN,
}
_L3_MACHINES = {
    (2, 3): [RULE_IDENTITY, RULE_R2B, RULE_R2G],
    (4, 4): [RULE_IDENTITY, RULE_B2G, RULE_G2O],
}
_L3_SOURCES = [(2, 1, C_RED)]
_L3_GOALS = {(1, 5): C_BLUE, (5, 5): C_GREEN}
_L3_SPLITTERS = {(2, 4): (D_UP, D_DOWN)}
_L3_SELECTORS: Dict[Tuple[int, int], Dict[int, int]] = {}

_L4_GRID = [
    [T_EMPTY, T_EMPTY, T_EMPTY, T_EMPTY, T_EMPTY, T_EMPTY, T_EMPTY],
    [T_EMPTY, T_SOURCE, T_PIPE, T_MACHINE, T_PIPE, T_SELECTOR, T_EMPTY],
    [T_EMPTY, T_EMPTY, T_EMPTY, T_EMPTY, T_EMPTY, T_PIPE, T_EMPTY],
    [T_EMPTY, T_LAUNCH, T_EMPTY, T_EMPTY, T_EMPTY, T_GOAL, T_EMPTY],
    [T_EMPTY, T_EMPTY, T_EMPTY, T_EMPTY, T_EMPTY, T_EMPTY, T_EMPTY],
    [T_EMPTY, T_SOURCE, T_PIPE, T_PIPE, T_MACHINE, T_PIPE, T_GOAL],
    [T_EMPTY, T_EMPTY, T_EMPTY, T_EMPTY, T_EMPTY, T_EMPTY, T_EMPTY],
]
_L4_PIPES = {
    (1, 2): D_RIGHT,
    (1, 4): D_RIGHT,
    (2, 5): D_DOWN,
    (5, 2): D_RIGHT,
    (5, 3): D_RIGHT,
    (5, 5): D_RIGHT,
}
_L4_MACHINES = {
    (1, 3): [RULE_IDENTITY, RULE_R2B, RULE_R2G],
    (5, 4): [RULE_IDENTITY, RULE_B2R, RULE_B2G],
}
_L4_SOURCES = [(1, 1, C_RED), (5, 1, C_BLUE)]
_L4_GOALS = {(3, 5): C_GREEN, (5, 6): C_RED}
_L4_SPLITTERS: Dict[Tuple[int, int], Tuple[int, int]] = {}
_L4_SELECTORS = {(1, 5): {C_GREEN: D_DOWN}}

_ALL_GRIDS = [_L1_GRID, _L2_GRID, _L3_GRID, _L4_GRID]
_ALL_PIPES = [_L1_PIPES, _L2_PIPES, _L3_PIPES, _L4_PIPES]
_ALL_MACHINES = [_L1_MACHINES, _L2_MACHINES, _L3_MACHINES, _L4_MACHINES]
_ALL_SOURCES = [_L1_SOURCES, _L2_SOURCES, _L3_SOURCES, _L4_SOURCES]
_ALL_GOALS = [_L1_GOALS, _L2_GOALS, _L3_GOALS, _L4_GOALS]
_ALL_SPLITTERS = [_L1_SPLITTERS, _L2_SPLITTERS, _L3_SPLITTERS, _L4_SPLITTERS]
_ALL_SELECTORS = [_L1_SELECTORS, _L2_SELECTORS, _L3_SELECTORS, _L4_SELECTORS]

_MOVE_LIMITS = [42, 70, 91, 126]
_LIVES = [3, 3, 3, 3]
_SIM_STEPS = [10, 16, 20, 24]

TILE_SPRITE_MAP = {
    T_EMPTY: "empty",
    T_WALL: "wall",
    T_MACHINE: "mach",
    T_GOAL: "goal",
    T_SOURCE: "source",
    T_SPLITTER: "split",
    T_SELECTOR: "selector",
    T_LAUNCH: "launch",
}


def _build_level_sprites(
    grid: List[List[int]],
    dim: int,
    pipes: Dict[Tuple[int, int], int],
    goals: Dict[Tuple[int, int], int],
    sources: List[Tuple[int, int, int]],
) -> List[Sprite]:
    positions = _grid_positions(dim)
    source_colors: Dict[Tuple[int, int], int] = {}
    for sr, sc, col in sources:
        source_colors[(sr, sc)] = col
    result: List[Sprite] = []
    for r in range(dim):
        for c in range(dim):
            idx = r * dim + c
            px, py = positions[idx]
            tile = grid[r][c]
            if tile == T_PIPE:
                key = PIPE_SPRITE_KEY[pipes.get((r, c), D_RIGHT)]
            elif tile == T_SOURCE:
                key = SOURCE_SPRITE_KEY.get(source_colors.get((r, c), C_RED), "source")
            else:
                key = TILE_SPRITE_MAP.get(tile, "empty")
            result.append(sprites[key].clone().set_position(px, py))
    for (gr, gc), req_color in goals.items():
        idx = gr * dim + gc
        px, py = positions[idx]
        result.append(
            sprites[TARGET_MARKER_KEY.get(req_color, "tgt_red")]
            .clone()
            .set_position(px, py)
        )
    result.append(sprites["cur"].clone().set_position(positions[0][0], positions[0][1]))
    return result


def _build_levels() -> List[Level]:
    result: List[Level] = []
    for li in range(4):
        grid = _ALL_GRIDS[li]
        dim = len(grid)
        spr_list = _build_level_sprites(
            grid, dim, _ALL_PIPES[li], _ALL_GOALS[li], _ALL_SOURCES[li]
        )
        result.append(
            Level(
                sprites=spr_list,
                grid_size=(64, 64),
                data={
                    "dim": dim,
                    "grid": grid,
                    "mvl": _MOVE_LIMITS[li],
                    "lvs": _LIVES[li],
                    "sim": _SIM_STEPS[li],
                    "pip": [[r, c, d] for (r, c), d in _ALL_PIPES[li].items()],
                    "mch": [
                        [r, c, rules] for (r, c), rules in _ALL_MACHINES[li].items()
                    ],
                    "src": [list(s) for s in _ALL_SOURCES[li]],
                    "gol": [[r, c, col] for (r, c), col in _ALL_GOALS[li].items()],
                    "spl": [
                        [r, c, d1, d2]
                        for (r, c), (d1, d2) in _ALL_SPLITTERS[li].items()
                    ],
                    "sel": [
                        [r, c, mapping]
                        for (r, c), mapping in _ALL_SELECTORS[li].items()
                    ],
                    "lvn": li + 1,
                },
                name="ft02_l{}".format(li + 1),
            )
        )
    return result


levels: List[Level] = _build_levels()


class _MoveBar(RenderableUserDisplay):
    def __init__(self) -> None:
        self.move_limit = 0
        self.moves_remaining = 0

    def reset(self, limit: int) -> None:
        self.move_limit = limit
        self.moves_remaining = limit

    def use_move(self) -> bool:
        if self.moves_remaining <= 0:
            return False
        self.moves_remaining -= 1
        return True

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        if self.move_limit == 0:
            return frame
        ratio = self.moves_remaining / self.move_limit
        filled = round(64 * ratio)
        h = frame.shape[0]
        for x in range(64):
            frame[h - 2, x] = 10 if x < filled else 13
            frame[h - 1, x] = 10 if x < filled else 13
        return frame


class _LivesDisplay(RenderableUserDisplay):
    def __init__(self) -> None:
        self.lives = 3
        self.max_lives = 3

    def reset(self, lives: int) -> None:
        self.max_lives = lives
        self.lives = lives

    def lose_life(self) -> int:
        if self.lives > 0:
            self.lives -= 1
        return self.lives

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        for i in range(self.max_lives):
            x = 62 - i * 4
            if x < 0:
                break
            color = 7 if i < self.lives else 4
            frame[2, x] = color
            frame[2, x + 1] = color
            frame[3, x] = color
            frame[3, x + 1] = color
        return frame


class Ft02(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self._move_bar = _MoveBar()
        self._lives_display = _LivesDisplay()
        self._game_over = False
        self._game_won = False
        self._consecutive_resets = 0
        self._current_lvl = 1
        camera = Camera(
            0,
            0,
            _GRID_PX,
            _GRID_PX,
            BACKGROUND_COLOR,
            PADDING_COLOR,
            [self._move_bar, self._lives_display],
        )
        self._engine_snapshots: List[dict] = []
        self._rng = random.Random(seed)
        super().__init__("ft02", levels, camera, False, 1, [0, 1, 2, 3, 4, 5, 7])

    def _init_pipes(self) -> None:
        pip_data = self.current_level.get_data("pip")
        self._pipes: Dict[Tuple[int, int], int] = {}
        for item in pip_data:
            self._pipes[(item[0], item[1])] = item[2]

    def _init_machines(self) -> None:
        mch_data = self.current_level.get_data("mch")
        self._machine_rules: Dict[Tuple[int, int], List[int]] = {}
        for item in mch_data:
            self._machine_rules[(item[0], item[1])] = item[2]
        self._machine_current: Dict[Tuple[int, int], int] = {}
        for pos in self._machine_rules:
            self._machine_current[pos] = 0

    def _init_sources_and_goals(self) -> None:
        src_data = self.current_level.get_data("src")
        self._sources: List[Tuple[int, int, int]] = []
        for item in src_data:
            self._sources.append((item[0], item[1], item[2]))
        gol_data = self.current_level.get_data("gol")
        self._goals: Dict[Tuple[int, int], int] = {}
        for item in gol_data:
            self._goals[(item[0], item[1])] = item[2]

    def _init_splitters_and_selectors(self) -> None:
        spl_data = self.current_level.get_data("spl")
        self._splitters: Dict[Tuple[int, int], Tuple[int, int]] = {}
        for item in spl_data:
            self._splitters[(item[0], item[1])] = (item[2], item[3])
        sel_data = self.current_level.get_data("sel")
        self._selectors: Dict[Tuple[int, int], Dict[int, int]] = {}
        for item in sel_data:
            cmap: Dict[int, int] = {}
            for k, v in item[2].items():
                cmap[int(k)] = int(v)
            self._selectors[(item[0], item[1])] = cmap

    def _init_launch_pos(self) -> None:
        self._launch_pos: Optional[Tuple[int, int]] = None
        for r in range(self._dim):
            for c in range(self._dim):
                if self._grid[r][c] == T_LAUNCH:
                    self._launch_pos = (r, c)
                    return

    def _init_tile_map(self) -> None:
        all_tiles: List[Sprite] = []
        for tag in [
            "floor",
            "wall",
            "pipe",
            "machine",
            "goal",
            "source",
            "splitter",
            "selector",
            "launch",
        ]:
            all_tiles.extend(self.current_level.get_sprites_by_tag(tag))
        self._tile_map: Dict[Tuple[int, int], Sprite] = {}
        for spr in all_tiles:
            col = round((spr.x - self._offset_x) / (_CELL + _GAP))
            row = round((spr.y - self._offset_y) / (_CELL + _GAP))
            if 0 <= row < self._dim and 0 <= col < self._dim:
                self._tile_map[(row, col)] = spr

    def on_set_level(self, level: Level) -> None:
        self._dim = self.current_level.get_data("dim")
        self._grid = self.current_level.get_data("grid")
        self._current_lvl = self.current_level.get_data("lvn")
        self._sim_steps = self.current_level.get_data("sim")

        self._move_bar.reset(self.current_level.get_data("mvl"))
        self._lives_display.reset(self.current_level.get_data("lvs"))

        self._positions = _grid_positions(self._dim)
        total = _CELL * self._dim + _GAP * (self._dim - 1)
        self._offset_x = (_GRID_PX - total) // 2
        self._offset_y = (_GRID_PX - total) // 2 + _Y_PAD

        self._init_pipes()
        self._init_machines()
        self._init_sources_and_goals()
        self._init_splitters_and_selectors()
        self._init_launch_pos()
        self._init_tile_map()

        cur_list = self.current_level.get_sprites_by_tag("cursor")
        self._cursor: Optional[Sprite] = cur_list[0] if cur_list else None
        self._rng = random.Random(self._seed + self._current_lvl)
        empty = [(r, c) for r in range(self._dim) for c in range(self._dim) if self._grid[r][c] == T_EMPTY]
        sr, sc = self._rng.choice(empty) if empty else (0, 0)
        self._start_cursor = (sr, sc)
        self._cur_row = sr
        self._cur_col = sc
        self._move_cursor_to(sr, sc)

        self._rule_indicators: Dict[Tuple[int, int], Sprite] = {}
        for pos in self._machine_rules:
            self._update_machine_visual(pos[0], pos[1])

        self._objects: Dict[Tuple[int, int], int] = {}
        self._object_sprites: Dict[Tuple[int, int], Sprite] = {}
        self._simulating = False
        self._sim_count = 0
        self._budget_exhausted_handled = False
        self._game_over = False
        self._engine_snapshots = []

    def _engine_save_snapshot(self) -> None:
        self._engine_snapshots.append({
            "machine_current": dict(self._machine_current),
            "objects": dict(self._objects),
            "cursor": (self._cur_row, self._cur_col),
            "simulating": self._simulating,
            "sim_count": self._sim_count,
            "budget_exhausted_handled": self._budget_exhausted_handled,
        })

    def _engine_restore_snapshot(self) -> None:
        if not self._engine_snapshots:
            return
        snap = self._engine_snapshots.pop()
        self._machine_current = snap["machine_current"]
        for pos in self._machine_current:
            self._update_machine_visual(pos[0], pos[1])
        self._clear_all_objects()
        self._objects = snap["objects"]
        for (r, c), color in self._objects.items():
            self._place_object_sprite(r, c, color)
        self._move_cursor_to(snap["cursor"][0], snap["cursor"][1])
        self._simulating = snap["simulating"]
        self._sim_count = snap["sim_count"]
        self._budget_exhausted_handled = snap["budget_exhausted_handled"]

    def full_reset(self) -> None:
        self._current_level_index = 0
        self._game_over = False
        self._game_won = False
        self._consecutive_resets = 0
        self.on_set_level(levels[0])

    def level_reset(self) -> None:
        self._reset_factory()
        self._move_bar.reset(self.current_level.get_data("mvl"))
        self._lives_display.reset(self.current_level.get_data("lvs"))
        self._engine_snapshots.clear()

    def _update_machine_visual(self, row: int, col: int) -> None:
        pos = (row, col)
        if pos in self._rule_indicators:
            self.current_level.remove_sprite(self._rule_indicators[pos])
            del self._rule_indicators[pos]
        rules = self._machine_rules.get(pos)
        if not rules:
            return
        rule = rules[self._machine_current.get(pos, 0)]
        in_c, out_c = RULE_VISUAL.get(rule, (2, 2))
        pix = [
            [-1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1],
            [-1, -1, in_c, out_c, -1, -1],
            [-1, -1, in_c, out_c, -1, -1],
            [-1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1],
        ]
        idx = row * self._dim + col
        px, py = self._positions[idx]
        ind = Sprite(
            pixels=pix,
            name="ind_{}_{}".format(row, col),
            visible=True,
            collidable=False,
            tags=["indicator"],
            layer=3,
        )
        ind = ind.set_position(px, py)
        self.current_level.add_sprite(ind)
        self._rule_indicators[pos] = ind

    def _move_cursor_to(self, row: int, col: int) -> None:
        self._cur_row = max(0, min(row, self._dim - 1))
        self._cur_col = max(0, min(col, self._dim - 1))
        if self._cursor:
            idx = self._cur_row * self._dim + self._cur_col
            px, py = self._positions[idx]
            self._cursor.set_position(px, py)

    def _cycle_machine_rule(self, row: int, col: int) -> bool:
        pos = (row, col)
        if pos not in self._machine_rules:
            return False
        rules = self._machine_rules[pos]
        self._machine_current[pos] = (self._machine_current.get(pos, 0) + 1) % len(
            rules
        )
        self._update_machine_visual(row, col)
        return True

    def _place_object_sprite(self, row: int, col: int, color: int) -> None:
        if (row, col) in self._object_sprites:
            self.current_level.remove_sprite(self._object_sprites[(row, col)])
            del self._object_sprites[(row, col)]
        ekey = OBJ_SPRITE_KEY.get(color)
        if ekey:
            idx = row * self._dim + col
            px, py = self._positions[idx]
            ov = sprites[ekey].clone().set_position(px, py)
            self.current_level.add_sprite(ov)
            self._object_sprites[(row, col)] = ov

    def _remove_object_sprite(self, row: int, col: int) -> None:
        if (row, col) in self._object_sprites:
            self.current_level.remove_sprite(self._object_sprites[(row, col)])
            del self._object_sprites[(row, col)]

    def _clear_all_objects(self) -> None:
        for pos in list(self._object_sprites.keys()):
            self.current_level.remove_sprite(self._object_sprites[pos])
        self._object_sprites.clear()
        self._objects.clear()

    def _spawn_objects(self) -> None:
        for sr, sc, color in self._sources:
            self._objects[(sr, sc)] = color
            self._place_object_sprite(sr, sc, color)

    def _resolve_pipe_target(self, or_: int, oc: int) -> List[Tuple[int, int]]:
        direction = self._pipes.get((or_, oc), D_RIGHT)
        dr, dc = DIR_DR[direction]
        nr, nc = or_ + dr, oc + dc
        if 0 <= nr < self._dim and 0 <= nc < self._dim:
            if self._grid[nr][nc] not in FLOW_BLOCKERS:
                return [(nr, nc)]
        return []

    def _resolve_source_target(self, or_: int, oc: int) -> List[Tuple[int, int]]:
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = or_ + dr, oc + dc
            if 0 <= nr < self._dim and 0 <= nc < self._dim:
                if self._grid[nr][nc] not in FLOW_BLOCKERS:
                    return [(nr, nc)]
        return []

    def _resolve_machine_target(self, or_: int, oc: int) -> List[Tuple[int, int]]:
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = or_ + dr, oc + dc
            if 0 <= nr < self._dim and 0 <= nc < self._dim:
                neighbor_tile = self._grid[nr][nc]
                if neighbor_tile == T_PIPE:
                    pd = self._pipes.get((nr, nc), D_RIGHT)
                    pdr, pdc = DIR_DR[pd]
                    if nr + pdr == or_ and nc + pdc == oc:
                        continue
                    return [(nr, nc)]
                elif neighbor_tile in (T_GOAL, T_SELECTOR, T_SPLITTER):
                    return [(nr, nc)]
        return []

    def _resolve_splitter_targets(self, or_: int, oc: int) -> List[Tuple[int, int]]:
        dirs = self._splitters.get((or_, oc), (D_UP, D_DOWN))
        targets: List[Tuple[int, int]] = []
        for direction in dirs:
            dr, dc = DIR_DR[direction]
            nr, nc = or_ + dr, oc + dc
            if 0 <= nr < self._dim and 0 <= nc < self._dim:
                if self._grid[nr][nc] not in FLOW_BLOCKERS:
                    targets.append((nr, nc))
        return targets

    def _resolve_selector_target(
        self, or_: int, oc: int, obj_color: int
    ) -> List[Tuple[int, int]]:
        cmap = self._selectors.get((or_, oc), {})
        if obj_color in cmap:
            dr, dc = DIR_DR[cmap[obj_color]]
            nr, nc = or_ + dr, oc + dc
            if 0 <= nr < self._dim and 0 <= nc < self._dim:
                if self._grid[nr][nc] not in FLOW_BLOCKERS:
                    return [(nr, nc)]
        return []

    def _simulate_step(self) -> bool:
        moved = False
        new_objects: Dict[Tuple[int, int], int] = {}
        pending: List[Tuple[Tuple[int, int], Tuple[int, int], int]] = []

        for (or_, oc), obj_color in self._objects.items():
            tile = self._grid[or_][oc]
            targets: List[Tuple[int, int]] = []
            out_color = obj_color

            if tile == T_PIPE:
                targets = self._resolve_pipe_target(or_, oc)
            elif tile == T_SOURCE:
                targets = self._resolve_source_target(or_, oc)
            elif tile == T_MACHINE:
                rules = self._machine_rules.get((or_, oc), [RULE_IDENTITY])
                out_color = _apply_rule(
                    rules[self._machine_current.get((or_, oc), 0)], obj_color
                )
                targets = self._resolve_machine_target(or_, oc)
            elif tile == T_SPLITTER:
                targets = self._resolve_splitter_targets(or_, oc)
            elif tile == T_SELECTOR:
                targets = self._resolve_selector_target(or_, oc, obj_color)
            elif tile == T_GOAL:
                new_objects[(or_, oc)] = obj_color
                continue

            if not targets:
                continue
            for tgt in targets:
                pending.append(((or_, oc), tgt, out_color))
                moved = True

        occupied: Dict[Tuple[int, int], int] = {}
        collided: set = set()
        for src, dst, color in pending:
            if dst in occupied:
                collided.add(dst)
            occupied[dst] = color

        for src, dst, color in pending:
            self._remove_object_sprite(src[0], src[1])
            if dst in collided:
                continue
            new_objects[dst] = color
            self._place_object_sprite(dst[0], dst[1], color)

        for pos in list(self._object_sprites.keys()):
            if pos not in new_objects:
                self._remove_object_sprite(pos[0], pos[1])

        self._objects = new_objects
        self._sim_count += 1
        return moved

    def _check_all_goals_met(self) -> bool:
        for gpos, req in self._goals.items():
            if gpos not in self._objects or self._objects[gpos] != req:
                return False
        return True

    def _try_launch_simulation(self) -> None:
        self._clear_all_objects()
        self._spawn_objects()
        self._simulating = True
        self._sim_count = 0

    def _handle_simulation_end(self) -> None:
        if self._check_all_goals_met():
            if self._current_lvl >= 4:
                self._game_won = True
            self.next_level()
            return
        remaining = self._lives_display.lose_life()
        if remaining <= 0:
            self._game_over = True
            self.lose()
        else:
            self._reset_factory()
            self._move_bar.reset(self.current_level.get_data("mvl"))

    def _reset_factory(self) -> None:
        self._clear_all_objects()
        self._simulating = False
        self._sim_count = 0
        self._budget_exhausted_handled = False
        for pos in self._machine_current:
            self._machine_current[pos] = 0
            self._update_machine_visual(pos[0], pos[1])
        self._move_cursor_to(self._start_cursor[0], self._start_cursor[1])

    def _handle_cursor_move(self, act: int) -> None:
        if self._move_bar.moves_remaining <= 0:
            return
        self._move_bar.use_move()
        if act == 1:
            self._move_cursor_to(self._cur_row - 1, self._cur_col)
        elif act == 2:
            self._move_cursor_to(self._cur_row + 1, self._cur_col)
        elif act == 3:
            self._move_cursor_to(self._cur_row, self._cur_col - 1)
        elif act == 4:
            self._move_cursor_to(self._cur_row, self._cur_col + 1)

    def _handle_action5(self) -> None:
        if self._move_bar.moves_remaining <= 0:
            return
        self._move_bar.use_move()
        tile = self._grid[self._cur_row][self._cur_col]
        if tile == T_MACHINE:
            self._cycle_machine_rule(self._cur_row, self._cur_col)
        elif tile == T_LAUNCH:
            self._try_launch_simulation()

    def _handle_budget_exhaustion(self) -> None:
        if self._budget_exhausted_handled:
            return
        self._budget_exhausted_handled = True
        remaining = self._lives_display.lose_life()
        if remaining <= 0:
            self._game_over = True
            self.lose()
        else:
            self._reset_factory()
            self._move_bar.reset(self.current_level.get_data("mvl"))

    def step(self) -> None:
        act = self.action.id.value

        if self._game_over:
            self.complete_action()
            return

        if act == 7:
            if self._move_bar.moves_remaining > 0:
                self._move_bar.use_move()
            if self._engine_snapshots:
                self._engine_restore_snapshot()
            if self._move_bar.moves_remaining <= 0:
                self._handle_budget_exhaustion()
            self.complete_action()
            return

        if self._simulating:
            if self._move_bar.moves_remaining <= 0:
                self._simulating = False
                self._handle_simulation_end()
                self.complete_action()
                return
            self._move_bar.use_move()
            on_launch = (self._cur_row, self._cur_col) == self._launch_pos
            if act == 5 and on_launch:
                any_moved = self._simulate_step()
                if not any_moved or self._sim_count >= self._sim_steps:
                    self._simulating = False
                    self._handle_simulation_end()
            self.complete_action()
            return

        self._engine_save_snapshot()

        if act in (1, 2, 3, 4):
            self._handle_cursor_move(act)
        elif act == 5:
            self._handle_action5()

        if act in (1, 2, 3, 4, 5):
            self._consecutive_resets = 0

        if (
            not self._simulating
            and not self._game_over
            and self._move_bar.moves_remaining <= 0
        ):
            self._handle_budget_exhaustion()

        self.complete_action()

    def handle_reset(self) -> None:
        if self._game_won:
            self._game_won = False
            self._game_over = False
            self._consecutive_resets = 0
            self.full_reset()
            return

        self._consecutive_resets += 1
        self._game_over = False

        if self._consecutive_resets >= 2:
            self._consecutive_resets = 0
            self.full_reset()
        else:
            self.level_reset()


ARC_PALETTE = np.array([
    [255, 255, 255],
    [204, 204, 204],
    [153, 153, 153],
    [102, 102, 102],
    [51,  51,  51],
    [0,   0,   0],
    [229, 58,  163],
    [255, 123, 204],
    [249, 60,  49],
    [30,  147, 255],
    [136, 216, 241],
    [255, 220, 0],
    [255, 133, 27],
    [146, 18,  49],
    [79,  204, 48],
    [163, 86,  208],
], dtype=np.uint8)


_TILE_CHAR = {
    T_EMPTY: ".",
    T_PIPE: "=",
    T_MACHINE: "M",
    T_GOAL: "G",
    T_SOURCE: "S",
    T_WALL: "#",
    T_SPLITTER: "Y",
    T_SELECTOR: "?",
    T_LAUNCH: "L",
}

_OBJ_CHAR = {
    C_RED: "r",
    C_BLUE: "b",
    C_GREEN: "g",
    C_ORANGE: "o",
}


class PuzzleEnvironment:

    _ACTION_MAP: Dict[str, GameAction] = {
        "reset":  GameAction.RESET,
        "up":     GameAction.ACTION1,
        "down":   GameAction.ACTION2,
        "left":   GameAction.ACTION3,
        "right":  GameAction.ACTION4,
        "select": GameAction.ACTION5,
        "undo":   GameAction.ACTION7,
    }

    _VALID_ACTIONS = list(_ACTION_MAP.keys())

    def __init__(self, seed: int = 0) -> None:
        self._engine = Ft02(seed)
        self._total_turns = 0
        self._done = False
        self._game_won = False
        self._game_over = False
        self._last_action_was_reset = False

    @staticmethod
    def _encode_png(frame: np.ndarray) -> bytes:
        h, w = frame.shape[:2]
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for idx in range(len(ARC_PALETTE)):
            mask = frame == idx
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            rgb[mask] = ARC_PALETTE[idx]

        buf = io.BytesIO()
        buf.write(b"\x89PNG\r\n\x1a\n")

        def _chunk(ctype: bytes, data: bytes) -> None:
            buf.write(struct.pack(">I", len(data)))
            buf.write(ctype)
            buf.write(data)
            buf.write(struct.pack(">I", zlib.crc32(ctype + data) & 0xFFFFFFFF))

        _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        raw = b""
        for y in range(h):
            raw += b"\x00" + rgb[y].tobytes()
        _chunk(b"IDAT", zlib.compress(raw))
        _chunk(b"IEND", b"")
        return buf.getvalue()

    def _render_frame(self) -> np.ndarray:
        e = self._engine
        return e.camera.render(e.current_level.get_sprites())

    def _build_text_observation(self) -> str:
        e = self._engine
        dim = e._dim
        grid = e._grid

        grid_lines = []
        for r in range(dim):
            row_chars = []
            for c in range(dim):
                tile = grid[r][c]
                ch = _TILE_CHAR.get(tile, ".")
                obj_color = e._objects.get((r, c))
                if obj_color is not None:
                    ch = _OBJ_CHAR.get(obj_color, ch)
                if (r, c) == (e._cur_row, e._cur_col):
                    row_chars.append("[" + ch + "]")
                else:
                    row_chars.append(" " + ch + " ")
            grid_lines.append("".join(row_chars))

        header = (
            f"Level {e._current_lvl}/{len(levels)}"
            f" | Lives: {e._lives_display.lives}"
        )
        if e._simulating:
            header += " | SIMULATING"
        rules = (
            "Move cursor with arrows. Select cycles machines or launches simulation. "
            "Route colored objects to matching goals. Undo reverses last action."
        )
        progress = (
            f"Moves: {e._move_bar.moves_remaining}/{e._move_bar.move_limit}"
        )
        return header + "\n" + rules + "\nBoard:\n" + "\n".join(grid_lines) + "\n" + progress

    def _build_game_state(self, done: bool = False) -> GameState:
        e = self._engine
        frame = self._render_frame()
        image_bytes = self._encode_png(frame)

        valid_actions = self.get_actions() if not done else None

        return GameState(
            text_observation=self._build_text_observation(),
            image_observation=image_bytes,
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

    def is_done(self) -> bool:
        return self._done

    def get_actions(self) -> List[str]:
        if self._done:
            return ["reset"]
        return list(self._VALID_ACTIONS)

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
        info: Dict[str, Any] = {"action": action}

        level_before = e._current_level_index

        action_input = ActionInput(id=game_action)
        frame = e.perform_action(action_input, raw=True)

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

        if game_over or e._game_over:
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
        if e._current_level_index != level_before:
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
        index_grid = self._render_frame()
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

    env = ArcGameEnv(seed=42, render_mode="rgb_array")
    check_env(env.unwrapped, skip_render_check=True)

    obs, info = env.reset()

    mask = env.action_mask()
    valid_indices = np.where(mask == 1)[0]
    if len(valid_indices) > 0:
        obs, reward, term, trunc, info = env.step(valid_indices[0])

    env.close()
