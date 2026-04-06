import random
import struct
import zlib
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from arcengine import (
    ARCBaseGame,
    ActionInput,
    Camera,
    GameAction,
    GameState as EngineGameState,
    Level,
    RenderableUserDisplay,
    Sprite,
)

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

ACTION_FROM_NAME: Dict[str, GameAction] = {
    "up": GameAction.ACTION1,
    "down": GameAction.ACTION2,
    "left": GameAction.ACTION3,
    "right": GameAction.ACTION4,
    "undo": GameAction.ACTION7,
    "reset": GameAction.RESET,
}

GRID_MAX       = 16
MAX_LIVES      = 3
CLR_LIFE       = 6
CLR_LIFE_LOST  = 5
CLR_BAR_FULL   = 10
CLR_BAR_LOW    = 12

C_BLACK         = 0
C_WALL          = 9
C_ALARM         = 2
C_DONE          = 3
C_KEYCARD_A     = 4
C_FLOOR         = 5
C_KEYCARD_B     = 6
C_KEYCARD_C     = 7
C_TERMINAL      = 8
C_PORT_INACTIVE = 9
C_ICE           = 10
C_PORT_ACTIVE   = 11
C_ZONE_A        = 12
C_DRONE         = 13
C_PANEL         = 14
C_PLAYER        = 15

BACKGROUND_COLOR = C_BLACK
PADDING_COLOR    = C_WALL

HUD_BOTTOM_RESERVE = 2

def _compute_tile(rows, cols):
    return min(64 // cols, (64 - HUD_BOTTOM_RESERVE) // rows)

def _make_border(T, color):
    if T == 1:
        return [[color]]
    return [[color] * T for _ in range(T)]

def _make_floor(T):
    if T == 1:
        return [[C_FLOOR]]
    if T == 2:
        return [[C_FLOOR, C_FLOOR], [C_FLOOR, C_FLOOR]]
    rows = []
    for r in range(T):
        row = [C_BLACK] * T
        if r == 0 or r == T - 1:
            row[0] = C_FLOOR
            row[T - 1] = C_FLOOR
        rows.append(row)
    return rows

def _make_cross(T, color, bg=-1):
    if T == 1:
        return [[color]]
    if T == 2:
        return [[color, color], [color, color]]
    arm = max(1, T // 3)
    mid = T // 2
    lo = mid - arm // 2
    hi = lo + arm
    rows = []
    for r in range(T):
        row = [bg] * T
        if lo <= r < hi:
            row = [color] * T
        else:
            for c in range(lo, hi):
                row[c] = color
        rows.append(row)
    return rows

def _make_drone_cross(T):
    if T == 1:
        return [[C_ALARM]]
    if T == 2:
        return [[C_ALARM, C_DRONE], [C_DRONE, C_DRONE]]
    mid = T // 2
    rows = []
    for r in range(T):
        row = [-1] * T
        if r == mid:
            row = [C_DRONE] * T
        else:
            row[mid] = C_DRONE
        rows.append(row)
    rows[mid][mid] = C_ALARM
    return rows

def _make_x_pattern(T, color):
    if T == 1:
        return [[color]]
    rows = []
    for r in range(T):
        row = [C_BLACK] * T
        row[r] = color
        row[T - 1 - r] = color
        rows.append(row)
    return rows

def _make_arrow_up(T):
    if T == 1:
        return [[C_KEYCARD_C]]
    if T == 2:
        return [[C_FLOOR, C_KEYCARD_C], [C_FLOOR, C_FLOOR]]
    if T == 3:
        return [
            [C_FLOOR, C_KEYCARD_C, C_FLOOR],
            [C_KEYCARD_C, C_FLOOR, C_KEYCARD_C],
            [C_FLOOR, C_FLOOR, C_FLOOR],
        ]
    mid = T // 2
    rows = []
    for r in range(T):
        row = [C_FLOOR] * T
        if r == 0:
            row[mid] = C_KEYCARD_C
        elif r < T - 1:
            spread = min(r, mid)
            left = max(0, mid - spread)
            right = min(T - 1, mid + spread)
            row[left] = C_KEYCARD_C
            row[right] = C_KEYCARD_C
        rows.append(row)
    return rows

def _make_arrow_down(T):
    if T == 1:
        return [[C_KEYCARD_C]]
    if T == 2:
        return [[C_FLOOR, C_FLOOR], [C_KEYCARD_C, C_FLOOR]]
    if T == 3:
        return [
            [C_FLOOR, C_FLOOR, C_FLOOR],
            [C_KEYCARD_C, C_FLOOR, C_KEYCARD_C],
            [C_FLOOR, C_KEYCARD_C, C_FLOOR],
        ]
    mid = T // 2
    rows = []
    for r in range(T):
        row = [C_FLOOR] * T
        if r == T - 1:
            row[mid] = C_KEYCARD_C
        elif r > 0:
            spread = min(T - 1 - r, mid)
            left = max(0, mid - spread)
            right = min(T - 1, mid + spread)
            row[left] = C_KEYCARD_C
            row[right] = C_KEYCARD_C
        rows.append(row)
    return rows

def _make_arrow_left(T):
    if T == 1:
        return [[C_KEYCARD_C]]
    if T == 2:
        return [[C_FLOOR, C_KEYCARD_C], [C_KEYCARD_C, C_FLOOR]]
    if T == 3:
        return [
            [C_FLOOR, C_KEYCARD_C, C_FLOOR],
            [C_KEYCARD_C, C_FLOOR, C_FLOOR],
            [C_FLOOR, C_KEYCARD_C, C_FLOOR],
        ]
    mid = T // 2
    rows = []
    for r in range(T):
        row = [C_FLOOR] * T
        if r == mid:
            row[0] = C_KEYCARD_C
        else:
            spread = mid - abs(r - mid)
            if spread > 0:
                row[min(spread, T - 1)] = C_KEYCARD_C
        rows.append(row)
    return rows

def _make_arrow_right(T):
    if T == 1:
        return [[C_KEYCARD_C]]
    if T == 2:
        return [[C_KEYCARD_C, C_FLOOR], [C_FLOOR, C_KEYCARD_C]]
    if T == 3:
        return [
            [C_FLOOR, C_KEYCARD_C, C_FLOOR],
            [C_FLOOR, C_FLOOR, C_KEYCARD_C],
            [C_FLOOR, C_KEYCARD_C, C_FLOOR],
        ]
    mid = T // 2
    rows = []
    for r in range(T):
        row = [C_FLOOR] * T
        if r == mid:
            row[T - 1] = C_KEYCARD_C
        else:
            spread = mid - abs(r - mid)
            if spread > 0:
                row[max(0, T - 1 - spread)] = C_KEYCARD_C
        rows.append(row)
    return rows

def _make_sprites(tile):
    T = tile
    return {
        "wall": Sprite(
            pixels=_make_border(T, C_WALL),
            name="wall", visible=True, collidable=True, layer=0, tags=["wall"],
        ),
        "floor": Sprite(
            pixels=_make_floor(T),
            name="floor", visible=True, collidable=False, layer=-1, tags=["floor"],
        ),
        "ice": Sprite(
            pixels=_make_border(T, C_ICE),
            name="ice", visible=True, collidable=False, layer=-1, tags=["ice"],
        ),
        "player": Sprite(
            pixels=_make_cross(T, C_PLAYER, -1),
            name="player", visible=True, collidable=False, layer=5, tags=["player"],
        ),
        "keycard_a": Sprite(
            pixels=_make_border(T, C_KEYCARD_A),
            name="keycard_a", visible=True, collidable=False, layer=2, tags=["keycard", "keycard_a"],
        ),
        "keycard_b": Sprite(
            pixels=_make_border(T, C_KEYCARD_B),
            name="keycard_b", visible=True, collidable=False, layer=2, tags=["keycard", "keycard_b"],
        ),
        "keycard_c": Sprite(
            pixels=_make_border(T, C_KEYCARD_C),
            name="keycard_c", visible=True, collidable=False, layer=2, tags=["keycard", "keycard_c"],
        ),
        "zone_a": Sprite(
            pixels=[[C_ZONE_A]], name="zone_a", visible=True, collidable=False, layer=-2,
            tags=["zone", "zone_a"],
        ),
        "zone_b": Sprite(
            pixels=[[C_KEYCARD_B]], name="zone_b", visible=True, collidable=False, layer=-2,
            tags=["zone", "zone_b"],
        ),
        "zone_c": Sprite(
            pixels=[[C_KEYCARD_C]], name="zone_c", visible=True, collidable=False, layer=-2,
            tags=["zone", "zone_c"],
        ),
        "terminal": Sprite(
            pixels=_make_border(T, C_TERMINAL),
            name="terminal", visible=True, collidable=True, layer=3, tags=["terminal"],
        ),
        "port_inactive": Sprite(
            pixels=_make_border(T, C_PORT_INACTIVE),
            name="port_inactive", visible=True, collidable=False, layer=1, tags=["port"],
        ),
        "port_active": Sprite(
            pixels=_make_border(T, C_PORT_ACTIVE),
            name="port_active", visible=True, collidable=False, layer=1, tags=["port_active"],
        ),
        "panel": Sprite(
            pixels=_make_border(T, C_PANEL),
            name="panel", visible=True, collidable=False, layer=2, tags=["panel"],
        ),
        "panel_done": Sprite(
            pixels=_make_border(T, C_DONE),
            name="panel_done", visible=True, collidable=False, layer=2, tags=["panel_done"],
        ),
        "extraction_closed": Sprite(
            pixels=_make_x_pattern(T, C_ALARM),
            name="extraction_closed", visible=True, collidable=False, layer=2,
            tags=["extraction_closed"],
        ),
        "extraction_open": Sprite(
            pixels=_make_x_pattern(T, C_DONE),
            name="extraction_open", visible=True, collidable=False, layer=2,
            tags=["extraction_open"],
        ),
        "drone": Sprite(
            pixels=_make_drone_cross(T),
            name="drone", visible=True, collidable=True, layer=4, tags=["drone"],
        ),
        "decoy": Sprite(
            pixels=_make_border(T, C_ALARM),
            name="decoy", visible=True, collidable=False, layer=2, tags=["decoy"],
        ),
        "one_way_up": Sprite(
            pixels=_make_arrow_up(T),
            name="one_way_up", visible=True, collidable=False, layer=1, tags=["one_way_up"],
        ),
        "one_way_down": Sprite(
            pixels=_make_arrow_down(T),
            name="one_way_down", visible=True, collidable=False, layer=1, tags=["one_way_down"],
        ),
        "one_way_left": Sprite(
            pixels=_make_arrow_left(T),
            name="one_way_left", visible=True, collidable=False, layer=1, tags=["one_way_left"],
        ),
        "one_way_right": Sprite(
            pixels=_make_arrow_right(T),
            name="one_way_right", visible=True, collidable=False, layer=1, tags=["one_way_right"],
        ),
    }

ZONE_KEYCARD_MAP = {
    "zone_a": "keycard_a",
    "zone_b": "keycard_b",
    "zone_c": "keycard_c",
}

ONE_WAY_ALLOWED_DELTA = {
    "one_way_up":    (0, -1),
    "one_way_down":  (0,  1),
    "one_way_left":  (-1, 0),
    "one_way_right": ( 1, 0),
}

LEVEL_NAMES = [
    "Level 1: Decoy Maze",
    "Level 2: Ice Corridor",
    "Level 3: Double Lock",
    "Level 4: Triple Lockdown",
]

LEVEL_MOVE_LIMITS   = [80, 100, 120, 140]
LEVEL_EST_MOVES     = [18,  38,  38,  43]

L1_MAP = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1],
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
    [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]
L1_PLAYER      = (1, 14)
L1_PLAYER_SPAWNS = [(1, 14), (1, 1), (4, 13), (9, 14)]
L1_KEYCARDS    = [{"tag": "keycard_a", "x": 7, "y": 13}]
L1_ZONES       = [{"tag": "zone_a", "cells": [
    (11, 1), (12, 1), (13, 1), (14, 1),
    (11, 2), (12, 2), (13, 2), (14, 2),
    (11, 3), (12, 3), (13, 3), (14, 3),
    (11, 4), (12, 4), (13, 4), (14, 4),
    (11, 5), (12, 5), (13, 5), (14, 5),
    (11, 6), (12, 6), (13, 6), (14, 6),
]}]
L1_TERMINALS   = []
L1_PORTS       = []
L1_PORT_ORDER  = []
L1_PANELS      = []
L1_EXTRACTION  = (14, 14)
L1_DRONES      = []
L1_ICE         = []
L1_DECOYS      = [{"x": 6, "y": 13}, {"x": 8, "y": 13}]
L1_ONE_WAY     = []

L2_MAP = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]
L2_PLAYER      = (1, 14)
L2_PLAYER_SPAWNS = [(1, 14), (1, 12), (4, 14), (7, 14)]
L2_KEYCARDS    = [{"tag": "keycard_a", "x": 1, "y": 10}]
L2_ZONES       = [{"tag": "zone_a", "cells": [
    (12, 1), (13, 1), (14, 1),
    (12, 2), (13, 2), (14, 2),
]}]
L2_TERMINALS   = [{"x": 2, "y": 4}]
L2_PORTS       = [{"x": 14, "y": 4, "color_idx": 0}]
L2_PORT_ORDER  = [0]
L2_PANELS      = [{"x": 3, "y": 2}]
L2_EXTRACTION  = (14, 14)
L2_DRONES      = []
L2_ICE         = [
    (3, 4),  (4, 4),  (5, 4),  (6, 4),  (7, 4),  (8, 4),  (9, 4),  (10, 4), (11, 4), (12, 4), (13, 4), (14, 4),
]
L2_DECOYS      = [{"x": 2, "y": 8}, {"x": 2, "y": 12}]
L2_ONE_WAY     = []

L3_MAP = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]
L3_PLAYER      = (1, 14)
L3_PLAYER_SPAWNS = [(1, 14), (1, 12), (9, 14), (11, 14)]
L3_KEYCARDS    = [
    {"tag": "keycard_a", "x": 1, "y": 10},
    {"tag": "keycard_b", "x": 5, "y": 14},
]
L3_ZONES       = [
    {"tag": "zone_a", "cells": [
        (12, 1), (13, 1), (14, 1),
        (12, 2), (13, 2), (14, 2),
    ]},
    {"tag": "zone_b", "cells": [
        (12, 6), (13, 6), (14, 6),
    ]},
]
L3_TERMINALS   = [{"x": 2, "y": 4}, {"x": 2, "y": 8}]
L3_PORTS       = [
    {"x": 14, "y": 4, "color_idx": 0},
    {"x": 14, "y": 8, "color_idx": 1},
]
L3_PORT_ORDER  = [0, 1]
L3_PANELS      = [{"x": 3, "y": 2}, {"x": 8, "y": 14}]
L3_EXTRACTION  = (14, 14)
L3_DRONES      = []
L3_ICE         = [
    (3, 4),  (4, 4),  (5, 4),  (6, 4),  (7, 4),  (8, 4),  (9, 4),  (10, 4), (11, 4), (12, 4), (13, 4), (14, 4),
    (3, 8),  (4, 8),  (5, 8),  (6, 8),  (7, 8),  (8, 8),  (9, 8),  (10, 8), (11, 8), (12, 8), (13, 8), (14, 8),
]
L3_DECOYS      = [{"x": 4, "y": 14}, {"x": 6, "y": 14}]
L3_ONE_WAY     = []

L4_MAP = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]
L4_PLAYER      = (1, 14)
L4_PLAYER_SPAWNS = [(1, 14), (7, 14), (9, 14), (11, 14)]
L4_KEYCARDS    = [
    {"tag": "keycard_a", "x": 1, "y": 10},
    {"tag": "keycard_b", "x": 5, "y": 14},
    {"tag": "keycard_c", "x": 1, "y": 4},
]
L4_ZONES       = [
    {"tag": "zone_a", "cells": [
        (12, 1), (13, 1), (14, 1),
        (12, 2), (13, 2), (14, 2),
    ]},
    {"tag": "zone_b", "cells": [
        (12, 6), (13, 6), (14, 6),
    ]},
    {"tag": "zone_c", "cells": [
        (12, 10), (13, 10), (14, 10),
    ]},
]
L4_TERMINALS   = [{"x": 2, "y": 4}, {"x": 2, "y": 8}, {"x": 2, "y": 12}]
L4_PORTS       = [
    {"x": 14, "y": 4,  "color_idx": 0},
    {"x": 14, "y": 8,  "color_idx": 1},
    {"x": 14, "y": 12, "color_idx": 2},
]
L4_PORT_ORDER  = [0, 1, 2]
L4_PANELS      = [{"x": 3, "y": 2}, {"x": 8, "y": 14}, {"x": 4, "y": 6}]
L4_EXTRACTION  = (14, 14)
L4_DRONES      = []
L4_ICE         = [
    (3, 4),  (4, 4),  (5, 4),  (6, 4),  (7, 4),  (8, 4),  (9, 4),  (10, 4), (11, 4), (12, 4), (13, 4), (14, 4),
    (3, 8),  (4, 8),  (5, 8),  (6, 8),  (7, 8),  (8, 8),  (9, 8),  (10, 8), (11, 8), (12, 8), (13, 8), (14, 8),
    (3, 12), (4, 12), (5, 12), (6, 12), (7, 12), (8, 12), (9, 12), (10, 12),(11, 12),(12, 12),(13, 12),(14, 12),
]
L4_DECOYS      = [
    {"x": 2, "y": 10}, {"x": 6, "y": 6},
]
L4_ONE_WAY     = []

ALL_LEVEL_DATA = [
    {
        "map": L1_MAP, "player": L1_PLAYER, "player_spawns": L1_PLAYER_SPAWNS, "keycards": L1_KEYCARDS,
        "zones": L1_ZONES, "terminals": L1_TERMINALS, "ports": L1_PORTS,
        "port_order": L1_PORT_ORDER, "panels": L1_PANELS,
        "extraction": L1_EXTRACTION, "drones": L1_DRONES, "ice": L1_ICE,
        "decoys": L1_DECOYS, "one_way": L1_ONE_WAY,
        "move_limit": LEVEL_MOVE_LIMITS[0],
        "est_moves": LEVEL_EST_MOVES[0],
    },
    {
        "map": L2_MAP, "player": L2_PLAYER, "player_spawns": L2_PLAYER_SPAWNS, "keycards": L2_KEYCARDS,
        "zones": L2_ZONES, "terminals": L2_TERMINALS, "ports": L2_PORTS,
        "port_order": L2_PORT_ORDER, "panels": L2_PANELS,
        "extraction": L2_EXTRACTION, "drones": L2_DRONES, "ice": L2_ICE,
        "decoys": L2_DECOYS, "one_way": L2_ONE_WAY,
        "move_limit": LEVEL_MOVE_LIMITS[1],
        "est_moves": LEVEL_EST_MOVES[1],
    },
    {
        "map": L3_MAP, "player": L3_PLAYER, "player_spawns": L3_PLAYER_SPAWNS, "keycards": L3_KEYCARDS,
        "zones": L3_ZONES, "terminals": L3_TERMINALS, "ports": L3_PORTS,
        "port_order": L3_PORT_ORDER, "panels": L3_PANELS,
        "extraction": L3_EXTRACTION, "drones": L3_DRONES, "ice": L3_ICE,
        "decoys": L3_DECOYS, "one_way": L3_ONE_WAY,
        "move_limit": LEVEL_MOVE_LIMITS[2],
        "est_moves": LEVEL_EST_MOVES[2],
    },
    {
        "map": L4_MAP, "player": L4_PLAYER, "player_spawns": L4_PLAYER_SPAWNS, "keycards": L4_KEYCARDS,
        "zones": L4_ZONES, "terminals": L4_TERMINALS, "ports": L4_PORTS,
        "port_order": L4_PORT_ORDER, "panels": L4_PANELS,
        "extraction": L4_EXTRACTION, "drones": L4_DRONES, "ice": L4_ICE,
        "decoys": L4_DECOYS, "one_way": L4_ONE_WAY,
        "move_limit": LEVEL_MOVE_LIMITS[3],
        "est_moves": LEVEL_EST_MOVES[3],
    },
]

def _get_map_dims(game_map):
    rows = len(game_map)
    cols = len(game_map[0]) if rows > 0 else 0
    return rows, cols

def _get_canvas_offset(rows, cols, tile):
    pixel_w = cols * tile
    pixel_h = rows * tile
    usable_h = 64 - HUD_BOTTOM_RESERVE
    ox = max(0, (64 - pixel_w) // 2)
    oy = max(0, (usable_h - pixel_h) // 2)
    return ox, oy

def _tp(mx, my, ox, oy, tile):
    return ox + mx * tile, oy + my * tile

def _build_level_sprites(data):
    game_map = data["map"]
    rows, cols = _get_map_dims(game_map)
    tile = _compute_tile(rows, cols)
    ox, oy = _get_canvas_offset(rows, cols, tile)
    spr = _make_sprites(tile)
    result = []

    for r in range(rows):
        for c in range(cols):
            cell = game_map[r][c]
            gx, gy = _tp(c, r, ox, oy, tile)
            if cell == 1:
                result.append(spr["wall"].clone().set_position(gx, gy))
            else:
                result.append(spr["floor"].clone().set_position(gx, gy))

    for zone_def in data["zones"]:
        for cx, cy in zone_def["cells"]:
            gx, gy = _tp(cx, cy, ox, oy, tile)
            result.append(spr[zone_def["tag"]].clone().set_position(gx, gy))

    for ice_pos in data["ice"]:
        gx, gy = _tp(ice_pos[0], ice_pos[1], ox, oy, tile)
        result.append(spr["ice"].clone().set_position(gx, gy))

    for ow_def in data["one_way"]:
        gx, gy = _tp(ow_def["x"], ow_def["y"], ox, oy, tile)
        result.append(spr[ow_def["kind"]].clone().set_position(gx, gy))

    for port_def in data["ports"]:
        gx, gy = _tp(port_def["x"], port_def["y"], ox, oy, tile)
        result.append(spr["port_inactive"].clone().set_position(gx, gy))

    for panel_def in data["panels"]:
        gx, gy = _tp(panel_def["x"], panel_def["y"], ox, oy, tile)
        result.append(spr["panel"].clone().set_position(gx, gy))

    for decoy_def in data["decoys"]:
        gx, gy = _tp(decoy_def["x"], decoy_def["y"], ox, oy, tile)
        result.append(spr["decoy"].clone().set_position(gx, gy))

    for kc_def in data["keycards"]:
        gx, gy = _tp(kc_def["x"], kc_def["y"], ox, oy, tile)
        result.append(spr[kc_def["tag"]].clone().set_position(gx, gy))

    for term_def in data["terminals"]:
        gx, gy = _tp(term_def["x"], term_def["y"], ox, oy, tile)
        result.append(spr["terminal"].clone().set_position(gx, gy))

    for drone_def in data["drones"]:
        gx, gy = _tp(drone_def["x"], drone_def["y"], ox, oy, tile)
        result.append(spr["drone"].clone().set_position(gx, gy))

    ex, ey = data["extraction"]
    gx, gy = _tp(ex, ey, ox, oy, tile)
    result.append(spr["extraction_closed"].clone().set_position(gx, gy))

    px, py = data["player"]
    gx, gy = _tp(px, py, ox, oy, tile)
    result.append(spr["player"].clone().set_position(gx, gy))

    return result, ox, oy, tile

def _build_all_levels():
    all_levels = []
    for idx, data in enumerate(ALL_LEVEL_DATA):
        sprite_list, ox, oy, tile = _build_level_sprites(data)
        rows, cols = _get_map_dims(data["map"])
        level = Level(
            sprites=sprite_list,
            grid_size=(64, 64),
            data={
                "ox": ox,
                "oy": oy,
                "rows": rows,
                "cols": cols,
                "tile": tile,
                "move_limit": data["move_limit"],
                "bar_limit": data["est_moves"] * 4,
                "port_order": list(data["port_order"]),
                "panel_count": len(data["panels"]),
            },
            name=LEVEL_NAMES[idx],
        )
        all_levels.append(level)
    return all_levels

_LEVELS = _build_all_levels()

class VaultHud(RenderableUserDisplay):

    def __init__(self, game: "Vx01") -> None:
        self._game = game

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        g      = self._game
        fh, fw = frame.shape

        tile   = max(1, g._tile)
        oy     = g._oy
        ox     = g._ox
        rows   = g._rows
        cols   = g._cols

        map_x0 = ox
        map_y0 = oy
        map_x1 = ox + cols * tile
        map_y1 = oy + rows * tile

        sq  = max(1, tile // 2)
        gap = max(1, tile // 4)

        for i in range(MAX_LIVES):
            x0 = map_x0 + gap + i * (sq + gap)
            y0 = map_y0 + 1
            x1 = min(map_x1, x0 + sq)
            y1 = min(map_y1, y0 + sq)
            if x0 < x1 and y0 < y1:
                frame[y0:y1, x0:x1] = (
                    CLR_LIFE if i < g._lives else CLR_LIFE_LOST
                )

        bar_total = g._move_limit
        if bar_total > 0:
            ratio  = max(0.0, (bar_total - g._move_count) / bar_total)
            bar_h  = max(1, HUD_BOTTOM_RESERVE)
            bar_y0 = map_y1
            bar_y1 = min(fh, bar_y0 + bar_h)
            bar_x0 = map_x0
            bar_w  = map_x1 - map_x0
            filled = int(ratio * bar_w)
            colour = CLR_BAR_FULL if ratio > 0.25 else CLR_BAR_LOW
            if bar_y0 < bar_y1 and bar_w > 0:
                frame[bar_y0:bar_y1, bar_x0:bar_x0 + filled]         = colour
                frame[bar_y0:bar_y1, bar_x0 + filled:bar_x0 + bar_w] = BACKGROUND_COLOR

        return frame

class Vx01(ARCBaseGame):
    MAX_LIVES = MAX_LIVES

    def __init__(self, seed: int = 0) -> None:
        self._hud = VaultHud(self)

        self._lives            = MAX_LIVES
        self._move_count       = 0
        self._move_limit       = 0
        self._bar_limit        = 0

        self._moves_remaining  = 0
        self._held_keycard     = None
        self._ports_filled     = []
        self._port_fill_index  = 0
        self._panels_done      = 0
        self._panel_count      = 0
        self._extraction_open  = False

        self._wall_cache       = set()
        self._ice_cache        = set()
        self._zone_cache       = {}
        self._one_way_cache    = {}
        self._terminal_positions = {}
        self._drone_patrollers = []
        self._ox               = 0
        self._oy               = 0
        self._rows             = 0
        self._cols             = 0
        self._tile             = 4
        self._sprites          = _make_sprites(4)
        self._rng              = random.Random(seed)
        self._chosen_spawn     = None

        camera = Camera(0, 0, 64, 64, BACKGROUND_COLOR, PADDING_COLOR, [self._hud])
        super().__init__(
            game_id="vx01",
            levels=_LEVELS,
            camera=camera,
            available_actions=[0, 1, 2, 3, 4, 7],
            seed=seed,
        )

        self.seed = seed
        self.game_seed = seed

        self.ready: bool = False
        self.has_played: bool = False
        self.preserve_lives: bool = False
        self.just_reset: bool = False
        self.undo_stack: deque = deque(maxlen=50)

        self.ready = True

    def on_set_level(self, level: Level) -> None:
        if not getattr(self, "ready", False):
            return

        self.undo_stack.clear()

        idx  = self.level_index
        if idx == 0:
            self._rng = random.Random(self.game_seed)
        data = ALL_LEVEL_DATA[idx]

        self._ox          = self.current_level.get_data("ox")
        self._oy          = self.current_level.get_data("oy")
        self._rows        = self.current_level.get_data("rows")
        self._cols        = self.current_level.get_data("cols")
        self._tile        = self.current_level.get_data("tile")
        self._sprites     = _make_sprites(self._tile)
        self._move_limit  = self.current_level.get_data("move_limit")
        self._bar_limit   = self.current_level.get_data("bar_limit")
        self._move_count  = 0
        self._moves_remaining = self._move_limit

        if self.preserve_lives:
            self.preserve_lives = False
        else:
            self._lives = MAX_LIVES

        self._held_keycard    = None
        self._ports_filled    = [False] * len(data["ports"])
        self._port_fill_index = 0
        self._panel_count     = self.current_level.get_data("panel_count")
        self._panels_done     = 0
        self._extraction_open = False

        self._build_wall_cache(data)
        self._build_ice_cache(data)
        self._build_zone_cache(data)
        self._build_one_way_cache(data)
        self._build_terminal_cache(data)
        self._setup_drone_patrol(data)

        self._spawn_positions = {}
        for spr in (
            self.current_level.get_sprites_by_tag("player")
            + self.current_level.get_sprites_by_tag("terminal")
            + self.current_level.get_sprites_by_tag("drone")
        ):
            self._spawn_positions[id(spr)] = (spr.x, spr.y)

        spawns = data.get("player_spawns")
        if spawns and len(spawns) > 1:
            self._chosen_spawn = self._rng.choice(spawns)
        else:
            self._chosen_spawn = data["player"]
        player_spr = self._get_player_sprite()
        if player_spr:
            sx, sy = _tp(self._chosen_spawn[0], self._chosen_spawn[1], self._ox, self._oy, self._tile)
            player_spr.set_position(sx, sy)
            self._spawn_positions[id(player_spr)] = (sx, sy)

        self._keycard_init_state = {}
        for tag in ("keycard_a", "keycard_b", "keycard_c", "decoy"):
            for spr in self.current_level.get_sprites_by_tag(tag):
                self._keycard_init_state[id(spr)] = {
                    "x": spr.x, "y": spr.y, "visible": spr.is_visible
                }

    def save_undo(self) -> None:
        self.undo_stack.append({
            "move_count": self._move_count,
            "moves_remaining": self._moves_remaining,
            "lives": self._lives,
            "held_keycard": self._held_keycard,
            "ports_filled": list(self._ports_filled),
            "port_fill_index": self._port_fill_index,
            "panels_done": self._panels_done,
            "extraction_open": self._extraction_open,
            "terminal_positions": dict(self._terminal_positions),
            "drone_patrollers": deepcopy([
                {"goal_idx": p["goal_idx"],
                 "pos": (p["sprite"].x, p["sprite"].y)}
                for p in self._drone_patrollers
            ]),
            "spawn_positions": dict(self._spawn_positions),
            "keycard_init_state": deepcopy(self._keycard_init_state),
            "player_pos": self._get_player_pos_tuple(),
            "terminal_sprite_positions": self._get_terminal_sprite_positions(),
            "keycard_visibility": self._get_keycard_visibility(),
            "port_visibility": self._get_port_visibility(),
            "panel_visibility": self._get_panel_visibility(),
        })

    def _get_player_pos_tuple(self) -> Tuple[int, int]:
        p = self._get_player_sprite()
        if p:
            return (p.x, p.y)
        return (0, 0)

    def _get_terminal_sprite_positions(self) -> List[Tuple[int, int]]:
        return [(s.x, s.y) for s in self.current_level.get_sprites_by_tag("terminal")]

    def _get_keycard_visibility(self) -> Dict[str, List[Tuple[int, int, bool]]]:
        result = {}
        for tag in ("keycard_a", "keycard_b", "keycard_c", "decoy"):
            result[tag] = [(s.x, s.y, s.is_visible) for s in self.current_level.get_sprites_by_tag(tag)]
        return result

    def _get_port_visibility(self) -> List[Tuple[int, int, bool]]:
        return [(s.x, s.y, s.is_visible) for s in self.current_level.get_sprites_by_tag("port")]

    def _get_panel_visibility(self) -> List[Tuple[int, int, bool]]:
        return [(s.x, s.y, s.is_visible) for s in self.current_level.get_sprites_by_tag("panel")]

    def restore_undo(self) -> bool:
        if not self.undo_stack:
            return False
        snap = self.undo_stack.pop()

        self._move_count       = snap["move_count"]
        self._moves_remaining  = snap["moves_remaining"]
        self._lives            = snap["lives"]
        self._held_keycard     = snap["held_keycard"]
        self._ports_filled     = snap["ports_filled"]
        self._port_fill_index  = snap["port_fill_index"]
        self._panels_done      = snap["panels_done"]
        self._extraction_open  = snap["extraction_open"]
        self._terminal_positions = snap["terminal_positions"]

        p = self._get_player_sprite()
        if p:
            p.set_position(snap["player_pos"][0], snap["player_pos"][1])

        terminals = self.current_level.get_sprites_by_tag("terminal")
        for i, (tx, ty) in enumerate(snap["terminal_sprite_positions"]):
            if i < len(terminals):
                terminals[i].set_position(tx, ty)

        for tag, states in snap["keycard_visibility"].items():
            tag_sprites = self.current_level.get_sprites_by_tag(tag)
            for i, (sx, sy, vis) in enumerate(states):
                if i < len(tag_sprites):
                    tag_sprites[i].set_position(sx, sy)
                    tag_sprites[i].set_visible(vis)

        port_sprites = self.current_level.get_sprites_by_tag("port")
        for i, (sx, sy, vis) in enumerate(snap["port_visibility"]):
            if i < len(port_sprites):
                port_sprites[i].set_position(sx, sy)
                port_sprites[i].set_visible(vis)

        panel_sprites = self.current_level.get_sprites_by_tag("panel")
        for i, (sx, sy, vis) in enumerate(snap["panel_visibility"]):
            if i < len(panel_sprites):
                panel_sprites[i].set_position(sx, sy)
                panel_sprites[i].set_visible(vis)

        for i, dp in enumerate(snap["drone_patrollers"]):
            if i < len(self._drone_patrollers):
                self._drone_patrollers[i]["goal_idx"] = dp["goal_idx"]
                self._drone_patrollers[i]["sprite"].set_position(dp["pos"][0], dp["pos"][1])

        return True

    def prepare_for_reset(self) -> None:
        self.preserve_lives = False
        self.has_played = False

    def _is_level_pristine(self) -> bool:
        if self._move_count != 0:
            return False
        if self._lives != MAX_LIVES:
            return False
        player = self._get_player_sprite()
        if player and self._chosen_spawn:
            spawn_px, spawn_py = _tp(
                self._chosen_spawn[0], self._chosen_spawn[1],
                self._ox, self._oy, self._tile,
            )
            if player.x != spawn_px or player.y != spawn_py:
                return False
        return True

    def handle_reset(self) -> None:
        if self._state == EngineGameState.WIN:
            self.just_reset = False
            self.full_reset()
        elif self._current_level_index == 0:
            self.just_reset = False
            self.full_reset()
        elif self.just_reset:
            self.just_reset = False
            self.full_reset()
        elif self._is_level_pristine():
            self.just_reset = False
            self.full_reset()
        else:
            self.just_reset = True
            self.preserve_lives = False
            self._restore_level()
            self._lives = MAX_LIVES

    def _build_wall_cache(self, data) -> None:
        game_map = data["map"]
        self._wall_cache = set()
        for r, row in enumerate(game_map):
            for c, tile in enumerate(row):
                if tile == 1:
                    self._wall_cache.add((c, r))

    def _build_ice_cache(self, data) -> None:
        self._ice_cache = {(pos[0], pos[1]) for pos in data["ice"]}

    def _build_zone_cache(self, data) -> None:
        self._zone_cache = {}
        for zone_def in data["zones"]:
            for cx, cy in zone_def["cells"]:
                self._zone_cache[(cx, cy)] = zone_def["tag"]

    def _build_one_way_cache(self, data) -> None:
        self._one_way_cache = {}
        for ow_def in data["one_way"]:
            self._one_way_cache[(ow_def["x"], ow_def["y"])] = ow_def["kind"]

    def _build_terminal_cache(self, data) -> None:
        self._terminal_positions = {}
        for term_def in data["terminals"]:
            self._terminal_positions[(term_def["x"], term_def["y"])] = True

    def _setup_drone_patrol(self, data) -> None:
        self._drone_patrollers = []
        drone_sprites = self.current_level.get_sprites_by_tag("drone")
        for i, drone_def in enumerate(data["drones"]):
            if i < len(drone_sprites):
                self._drone_patrollers.append({
                    "sprite":    drone_sprites[i],
                    "patrol":    list(drone_def["patrol"]),
                    "goal_idx":  0,
                })

    def _restore_level(self) -> None:
        idx  = self.level_index
        data = ALL_LEVEL_DATA[idx]

        self.undo_stack.clear()

        for spr in (
            self.current_level.get_sprites_by_tag("player")
            + self.current_level.get_sprites_by_tag("terminal")
            + self.current_level.get_sprites_by_tag("drone")
        ):
            pos = self._spawn_positions.get(id(spr))
            if pos:
                spr.set_position(pos[0], pos[1])

        for tag in ("keycard_a", "keycard_b", "keycard_c", "decoy"):
            for spr in self.current_level.get_sprites_by_tag(tag):
                state = self._keycard_init_state.get(id(spr))
                if state:
                    spr.set_position(state["x"], state["y"])
                    spr.set_visible(state["visible"])

        for spr in self.current_level.get_sprites_by_tag("port"):
            spr.set_visible(True)
        for spr in self.current_level.get_sprites_by_tag("port_active"):
            self.current_level.remove_sprite(spr)

        for spr in self.current_level.get_sprites_by_tag("panel_done"):
            self.current_level.remove_sprite(spr)
        for spr in self.current_level.get_sprites_by_tag("panel"):
            spr.set_visible(True)

        for spr in self.current_level.get_sprites_by_tag("extraction_open"):
            self.current_level.remove_sprite(spr)
        for spr in self.current_level.get_sprites_by_tag("extraction_closed"):
            spr.set_visible(True)

        self._held_keycard    = None
        self._ports_filled    = [False] * len(data["ports"])
        self._port_fill_index = 0
        self._panels_done     = 0
        self._extraction_open = False

        self._move_count      = 0
        self._moves_remaining = self._move_limit

        self._build_terminal_cache(data)
        self._setup_drone_patrol(data)

    def lose_life(self) -> bool:
        self._lives -= 1
        if self._lives <= 0:
            self.lose()
            return True
        self._restore_level()
        return False

    def _is_wall(self, gx, gy) -> bool:
        return (gx, gy) in self._wall_cache

    def _is_terminal_at(self, gx, gy) -> bool:
        return (gx, gy) in self._terminal_positions

    def _get_terminal_sprite_at(self, gx, gy):
        tx, ty = _tp(gx, gy, self._ox, self._oy, self._tile)
        for spr in self.current_level.get_sprites_by_tag("terminal"):
            if spr.x == tx and spr.y == ty:
                return spr
        return None

    def _get_player_sprite(self):
        sprs = self.current_level.get_sprites_by_tag("player")
        return sprs[0] if sprs else None

    def _grid_pos_of(self, spr):
        gx = (spr.x - self._ox) // self._tile
        gy = (spr.y - self._oy) // self._tile
        return gx, gy

    def _can_enter_cell(self, gx, gy, dx, dy) -> bool:
        if gx < 0 or gy < 0 or gx >= self._cols or gy >= self._rows:
            return False
        if self._is_wall(gx, gy):
            return False
        if (gx, gy) in self._one_way_cache:
            kind = self._one_way_cache[(gx, gy)]
            allowed_delta = ONE_WAY_ALLOWED_DELTA[kind]
            if (dx, dy) != allowed_delta:
                return False
        zone_tag = self._zone_cache.get((gx, gy))
        if zone_tag is not None:
            required_keycard = ZONE_KEYCARD_MAP.get(zone_tag)
            if self._held_keycard != required_keycard:
                return False
        return True

    def _try_push_terminal(self, term_gx, term_gy, dx, dy) -> bool:
        dest_gx = term_gx + dx
        dest_gy = term_gy + dy

        if dest_gx < 0 or dest_gy < 0 or dest_gx >= self._cols or dest_gy >= self._rows:
            return False
        if self._is_wall(dest_gx, dest_gy):
            return False
        if self._is_terminal_at(dest_gx, dest_gy):
            return False

        spr = self._get_terminal_sprite_at(term_gx, term_gy)
        if spr is None:
            return False

        if (dest_gx, dest_gy) in self._ice_cache:
            slide_gx, slide_gy = dest_gx, dest_gy
            while True:
                next_gx = slide_gx + dx
                next_gy = slide_gy + dy
                if next_gx < 0 or next_gy < 0 or next_gx >= self._cols or next_gy >= self._rows:
                    break
                if self._is_wall(next_gx, next_gy):
                    break
                if self._is_terminal_at(next_gx, next_gy):
                    break
                slide_gx = next_gx
                slide_gy = next_gy
                if (slide_gx, slide_gy) not in self._ice_cache:
                    break
            dest_gx, dest_gy = slide_gx, slide_gy

        del self._terminal_positions[(term_gx, term_gy)]
        self._terminal_positions[(dest_gx, dest_gy)] = True

        new_tx, new_ty = _tp(dest_gx, dest_gy, self._ox, self._oy, self._tile)
        spr.set_position(new_tx, new_ty)

        self._check_port_delivery(dest_gx, dest_gy)
        return True

    def _check_port_delivery(self, term_gx, term_gy) -> None:
        idx  = self.level_index
        data = ALL_LEVEL_DATA[idx]
        port_order = data["port_order"]

        if self._port_fill_index >= len(port_order):
            return

        expected_port_idx = port_order[self._port_fill_index]
        if expected_port_idx >= len(data["ports"]):
            return
        port_def = data["ports"][expected_port_idx]

        if term_gx == port_def["x"] and term_gy == port_def["y"]:
            self._ports_filled[expected_port_idx] = True
            self._port_fill_index += 1
            self._activate_port_visual(port_def["x"], port_def["y"])

    def _activate_port_visual(self, gx, gy) -> None:
        tx, ty = _tp(gx, gy, self._ox, self._oy, self._tile)
        for spr in self.current_level.get_sprites_by_tag("port"):
            if spr.x == tx and spr.y == ty:
                self.current_level.remove_sprite(spr)
                break
        self.current_level.add_sprite(
            self._sprites["port_active"].clone().set_position(tx, ty)
        )

    def _try_collect_keycard(self, gx, gy) -> bool:
        tx, ty = _tp(gx, gy, self._ox, self._oy, self._tile)
        for tag in ("keycard_a", "keycard_b", "keycard_c"):
            for spr in self.current_level.get_sprites_by_tag(tag):
                if spr.x == tx and spr.y == ty and spr.is_visible:
                    if self._held_keycard is None:
                        self._held_keycard = tag
                        spr.set_visible(False)
                        return False

        for spr in self.current_level.get_sprites_by_tag("decoy"):
            if spr.x == tx and spr.y == ty and spr.is_visible:
                if self.lose_life():
                    return True
                return False
        return False

    def _try_activate_panel(self, gx, gy) -> None:
        tx, ty = _tp(gx, gy, self._ox, self._oy, self._tile)
        for spr in list(self.current_level.get_sprites_by_tag("panel")):
            if spr.x == tx and spr.y == ty and spr.is_visible:
                spr.set_visible(False)
                self.current_level.add_sprite(
                    self._sprites["panel_done"].clone().set_position(tx, ty)
                )
                self._panels_done += 1
                return

    def _check_all_objectives_complete(self) -> bool:
        idx  = self.level_index
        data = ALL_LEVEL_DATA[idx]
        ports_needed = len(data["port_order"])
        if self._port_fill_index < ports_needed:
            return False
        if self._panels_done < self._panel_count:
            return False
        return True

    def _open_extraction(self) -> None:
        if self._extraction_open:
            return
        self._extraction_open = True
        for spr in list(self.current_level.get_sprites_by_tag("extraction_closed")):
            self.current_level.remove_sprite(spr)
        idx  = self.level_index
        data = ALL_LEVEL_DATA[idx]
        ex, ey = data["extraction"]
        tx, ty = _tp(ex, ey, self._ox, self._oy, self._tile)
        self.current_level.add_sprite(
            self._sprites["extraction_open"].clone().set_position(tx, ty)
        )

    def _is_at_extraction(self, gx, gy) -> bool:
        if not self._extraction_open:
            return False
        idx  = self.level_index
        data = ALL_LEVEL_DATA[idx]
        ex, ey = data["extraction"]
        return gx == ex and gy == ey

    def _advance_drone(self, patroller) -> None:
        drone_spr = patroller["sprite"]
        patrol    = patroller["patrol"]
        drone_gx, drone_gy = self._grid_pos_of(drone_spr)

        goal_idx   = patroller["goal_idx"]
        num_goals  = len(patrol)
        current_goal = patrol[goal_idx]

        if drone_gx == current_goal[0] and drone_gy == current_goal[1]:
            patroller["goal_idx"] = (goal_idx + 1) % num_goals
            goal_idx              = patroller["goal_idx"]
            current_goal          = patrol[goal_idx]

        tx, ty    = current_goal
        step_dx   = 0
        step_dy   = 0
        if   drone_gx < tx: step_dx =  1
        elif drone_gx > tx: step_dx = -1
        elif drone_gy < ty: step_dy =  1
        elif drone_gy > ty: step_dy = -1

        new_gx = drone_gx + step_dx
        new_gy = drone_gy + step_dy

        if self._is_wall(new_gx, new_gy):
            return
        if self._is_terminal_at(new_gx, new_gy):
            return

        new_tx, new_ty = _tp(new_gx, new_gy, self._ox, self._oy, self._tile)
        drone_spr.set_position(new_tx, new_ty)

    def _check_drone_collision(self, player_gx, player_gy) -> bool:
        for patroller in self._drone_patrollers:
            drone_spr = patroller["sprite"]
            drone_gx, drone_gy = self._grid_pos_of(drone_spr)
            if drone_gx == player_gx and drone_gy == player_gy:
                return True
        return False

    def step(self) -> None:
        if not self.ready:
            self.complete_action()
            return

        if self.action and self.action.id == GameAction.RESET:
            self.complete_action()
            return

        self.has_played = True
        self.just_reset = False

        action = self.action.id

        if action == GameAction.ACTION7:
            cur_mc = self._move_count
            cur_mr = self._moves_remaining
            self.restore_undo()
            self._move_count = cur_mc + 1
            self._moves_remaining = max(0, self._move_limit - self._move_count)
            if self._move_count >= self._move_limit:
                if self.lose_life():
                    self.complete_action()
                    return
            self.complete_action()
            return

        dx, dy = 0, 0
        if   action == GameAction.ACTION1: dy = -1
        elif action == GameAction.ACTION2: dy =  1
        elif action == GameAction.ACTION3: dx = -1
        elif action == GameAction.ACTION4: dx =  1
        else:
            self.complete_action()
            return

        self.save_undo()

        self._move_count      += 1
        self._moves_remaining  = max(0, self._move_limit - self._move_count)

        player = self._get_player_sprite()
        if player is None:
            self.complete_action()
            return

        player_gx, player_gy = self._grid_pos_of(player)
        new_gx = player_gx + dx
        new_gy = player_gy + dy

        if self._is_terminal_at(new_gx, new_gy):
            pushed = self._try_push_terminal(new_gx, new_gy, dx, dy)
            if not pushed:
                if self._move_count >= self._move_limit:
                    if self.lose_life():
                        self.complete_action()
                        return
                self.complete_action()
                return

        elif not self._can_enter_cell(new_gx, new_gy, dx, dy):
            if self._move_count >= self._move_limit:
                if self.lose_life():
                    self.complete_action()
                    return
            self.complete_action()
            return

        if not self._is_terminal_at(new_gx, new_gy):
            new_tx, new_ty = _tp(new_gx, new_gy, self._ox, self._oy, self._tile)
            player.set_position(new_tx, new_ty)
            player_gx, player_gy = new_gx, new_gy

        if self._try_collect_keycard(player_gx, player_gy):
            self.complete_action()
            return
        self._try_activate_panel(player_gx, player_gy)

        if self._check_all_objectives_complete():
            self._open_extraction()

        if self._is_at_extraction(player_gx, player_gy):
            self.next_level()
            self.complete_action()
            return

        for patroller in self._drone_patrollers:
            self._advance_drone(patroller)

        if self._check_drone_collision(player_gx, player_gy):
            if self.lose_life():
                self.complete_action()
                return
            self.complete_action()
            return

        if self._move_count >= self._move_limit:
            if self.lose_life():
                self.complete_action()
                return
            self.complete_action()
            return

        self.complete_action()

    @property
    def extra_state(self) -> dict:
        idx = self._current_level_index
        data = ALL_LEVEL_DATA[idx]
        level_name = LEVEL_NAMES[idx] if idx < len(LEVEL_NAMES) else f"Level {idx + 1}"
        return {
            "level_title": level_name,
            "circuit_title": f"Vault Heist - Level {idx + 1}/{len(self._levels)}: {level_name}",
            "lives": self._lives,
            "lives_max": MAX_LIVES,
            "lives_lost": MAX_LIVES - self._lives,
            "moves_used": self._move_count,
            "moves_left": self._moves_remaining,
            "max_moves": self._move_limit,
            "bar_limit": self._bar_limit,
            "held_keycard": self._held_keycard,
            "ports_filled": sum(self._ports_filled),
            "ports_total": len(self._ports_filled),
            "panels_done": self._panels_done,
            "panels_total": self._panel_count,
            "extraction_open": self._extraction_open,
            "level_features": [
                f"Level {idx + 1}/{len(self._levels)}: {level_name}",
                f"Lives: {self._lives}/{MAX_LIVES}",
                f"Moves: {self._move_count}/{self._move_limit} ({self._moves_remaining} left)",
                f"Ports: {sum(self._ports_filled)}/{len(self._ports_filled)}",
                f"Panels: {self._panels_done}/{self._panel_count}",
                f"Extraction: {'OPEN' if self._extraction_open else 'CLOSED'}",
            ],
        }

@dataclass
class GameState:
    text_observation: str = ""
    image_observation: Optional[bytes] = None
    valid_actions: Optional[List[str]] = None
    turn: int = 0
    metadata: Dict = field(default_factory=dict)

@dataclass
class StepResult:
    state: GameState = field(default_factory=GameState)
    reward: float = 0.0
    done: bool = False
    info: Dict = field(default_factory=dict)

ACTION_LIST: list[str] = ["reset", "up", "down", "left", "right", "undo"]

class PuzzleEnvironment:

    ACTION_MAP: Dict[str, GameAction] = ACTION_FROM_NAME

    def __init__(self, seed: int = 0) -> None:
        self._engine = Vx01(seed=seed)
        self.turn: int = 0
        self.last_was_reset: bool = False
        self._total_levels: int = len(self._engine._levels)

    def frame_to_png(self, frame) -> bytes | None:
        try:
            arr = np.array(frame, dtype=np.uint8)
            h, w = arr.shape[:2]
            rgb = np.zeros((h, w, 3), dtype=np.uint8)
            for idx, color in enumerate(ARC_PALETTE):
                mask = arr == idx
                if mask.ndim == 3:
                    mask = mask.all(axis=2)
                rgb[mask] = color

            def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
                chunk = chunk_type + data
                return struct.pack(">I", len(data)) + chunk + struct.pack(">I", zlib.crc32(chunk) & 0xFFFFFFFF)

            raw_rows = b"".join(b"\x00" + rgb[r].tobytes() for r in range(h))
            return (
                b"\x89PNG\r\n\x1a\n"
                + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                + _png_chunk(b"IDAT", zlib.compress(raw_rows))
                + _png_chunk(b"IEND", b"")
            )
        except Exception:
            return None

    def _build_text_observation(self) -> str:
        g = self._engine
        idx = g._current_level_index
        data = ALL_LEVEL_DATA[idx]
        game_map = data["map"]
        rows, cols = _get_map_dims(game_map)

        grid = [["." for _ in range(cols)] for _ in range(rows)]

        for r in range(rows):
            for c in range(cols):
                if game_map[r][c] == 1:
                    grid[r][c] = "#"

        for ice_pos in data["ice"]:
            ix, iy = ice_pos[0], ice_pos[1]
            if 0 <= iy < rows and 0 <= ix < cols and grid[iy][ix] == ".":
                grid[iy][ix] = "~"

        for zone_def in data["zones"]:
            tag = zone_def["tag"]
            ch = {"zone_a": "a", "zone_b": "b", "zone_c": "c"}.get(tag, "z")
            for cx, cy in zone_def["cells"]:
                if 0 <= cy < rows and 0 <= cx < cols and grid[cy][cx] == ".":
                    grid[cy][cx] = ch

        for ow_def in data["one_way"]:
            ox, oy = ow_def["x"], ow_def["y"]
            ch = {"one_way_up": "^", "one_way_down": "v", "one_way_left": "<", "one_way_right": ">"}.get(ow_def["kind"], "?")
            if 0 <= oy < rows and 0 <= ox < cols:
                grid[oy][ox] = ch

        for i, port_def in enumerate(data["ports"]):
            px, py = port_def["x"], port_def["y"]
            if 0 <= py < rows and 0 <= px < cols:
                grid[py][px] = "O" if not g._ports_filled[i] else "o"

        for spr in g.current_level.get_sprites_by_tag("panel"):
            if spr.is_visible:
                gx, gy = g._grid_pos_of(spr)
                if 0 <= gy < rows and 0 <= gx < cols:
                    grid[gy][gx] = "N"
        for spr in g.current_level.get_sprites_by_tag("panel_done"):
            gx, gy = g._grid_pos_of(spr)
            if 0 <= gy < rows and 0 <= gx < cols:
                grid[gy][gx] = "n"

        for tag_name in ("keycard_a", "keycard_b", "keycard_c"):
            ch = {"keycard_a": "A", "keycard_b": "B", "keycard_c": "C"}[tag_name]
            for spr in g.current_level.get_sprites_by_tag(tag_name):
                if spr.is_visible:
                    gx, gy = g._grid_pos_of(spr)
                    if 0 <= gy < rows and 0 <= gx < cols:
                        grid[gy][gx] = ch

        for spr in g.current_level.get_sprites_by_tag("decoy"):
            if spr.is_visible:
                gx, gy = g._grid_pos_of(spr)
                if 0 <= gy < rows and 0 <= gx < cols:
                    grid[gy][gx] = "!"

        for gx, gy in g._terminal_positions:
            if 0 <= gy < rows and 0 <= gx < cols:
                grid[gy][gx] = "T"

        for patroller in g._drone_patrollers:
            gx, gy = g._grid_pos_of(patroller["sprite"])
            if 0 <= gy < rows and 0 <= gx < cols:
                grid[gy][gx] = "D"

        ex, ey = data["extraction"]
        if 0 <= ey < rows and 0 <= ex < cols:
            grid[ey][ex] = "X" if g._extraction_open else "x"

        player = g._get_player_sprite()
        if player:
            px, py = g._grid_pos_of(player)
            if 0 <= py < rows and 0 <= px < cols:
                grid[py][px] = "P"

        return "\n".join("".join(row) for row in grid)

    def build_text_observation(self) -> str:
        g = self._engine
        idx = g._current_level_index
        level_name = LEVEL_NAMES[idx] if idx < len(LEVEL_NAMES) else f"Level {idx + 1}"
        lines = [
            f"=== Vault Heist (Level {idx + 1}/{len(g._levels)}: {level_name}) ===",
            f"Lives: {g._lives}/{MAX_LIVES}",
            f"Moves: {g._move_count}/{g._move_limit} ({g._moves_remaining} left)",
            f"Keycard: {g._held_keycard or 'none'}",
            f"Ports: {sum(g._ports_filled)}/{len(g._ports_filled)}",
            f"Panels: {g._panels_done}/{g._panel_count}",
            f"Extraction: {'OPEN' if g._extraction_open else 'CLOSED'}",
        ]
        if g._state == EngineGameState.WIN:
            lines.append("State: won")
        elif g._state == EngineGameState.GAME_OVER:
            lines.append("State: game-over")
        else:
            lines.append("State: playing")
        lines.append("Legend: #=wall .=floor ~=ice P=player T=terminal O=port(empty) o=port(filled)")
        lines.append("  A/B/C=keycard !=decoy N=panel n=panel(done) x=exit(closed) X=exit(open)")
        lines.append("  a/b/c=zone D=drone ^v<>=one-way")
        lines.append(self._build_text_observation())
        return "\n".join(lines)

    def _build_game_state(self, frame_data) -> GameState:
        image_bytes: bytes | None = None
        if frame_data and not frame_data.is_empty():
            image_bytes = self.frame_to_png(frame_data.frame)

        g = self._engine
        done = g._state in (EngineGameState.WIN, EngineGameState.GAME_OVER)
        valid = self.get_actions()

        return GameState(
            text_observation=self.build_text_observation(),
            image_observation=image_bytes,
            valid_actions=valid,
            turn=self.turn,
            metadata={
                "total_levels": len(g._levels),
                "level_index": g.level_index,
                "levels_completed": getattr(g, "_score", 0),
                "game_over": g._state == EngineGameState.GAME_OVER,
                "done": done,
                "info": {},
            },
        )

    def reset(self) -> GameState:
        self.turn = 0
        self.last_was_reset = False
        self._engine.prepare_for_reset()
        frame_data = self._engine.perform_action(
            ActionInput(id=GameAction.RESET)
        )
        return self._build_game_state(frame_data)

    def get_actions(self) -> list[str]:
        done = self._engine._state in (EngineGameState.WIN, EngineGameState.GAME_OVER)
        if done:
            return ["reset"]
        return ACTION_LIST

    def is_done(self) -> bool:
        return self._engine._state in (EngineGameState.WIN, EngineGameState.GAME_OVER)

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError(
                f"Unsupported render mode '{mode}'. Only 'rgb_array' is supported."
            )
        frame: np.ndarray = self._engine.camera.render(self._engine.current_level.get_sprites())
        h, w = frame.shape[:2]
        target = 64
        if h < target or w < target:
            sy = target // h
            sx = target // w
            frame = np.repeat(np.repeat(frame, sy, axis=0), sx, axis=1)
            h, w = frame.shape[:2]
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for idx_val, color in enumerate(ARC_PALETTE):
            mask = frame == idx_val
            if mask.ndim == 3:
                mask = mask.all(axis=2)
            rgb[mask] = color
        return rgb

    def close(self) -> None:
        self._engine = None

    def step(self, action: str) -> StepResult:
        action = action.strip().lower()
        game_action = ACTION_FROM_NAME.get(action)
        if game_action is None:
            raise ValueError(
                f"Unknown action '{action}'. Valid: {ACTION_LIST}"
            )

        if game_action == GameAction.RESET and self.last_was_reset:
            self.last_was_reset = False
            self._engine.prepare_for_reset()
            frame_data = self._engine.perform_action(ActionInput(id=GameAction.RESET))
            return StepResult(
                state=self._build_game_state(frame_data),
                reward=0.0,
                done=False,
                info={"action": action, "engine_state": frame_data.state, "level_changed": True, "life_lost": False, "full_reset": True},
            )

        prev_level = self._engine.level_index
        prev_lives = self._engine._lives

        action_input = ActionInput(id=game_action)
        frame_data = self._engine.perform_action(action_input)

        if game_action == GameAction.RESET:
            self.last_was_reset = True
        else:
            self.last_was_reset = False
            self.turn += 1

        engine_state = frame_data.state
        done = engine_state in (EngineGameState.WIN, EngineGameState.GAME_OVER)

        reward = 0.0
        if engine_state == EngineGameState.WIN:
            reward = 1.0 / self._total_levels
        elif self._engine.level_index > prev_level:
            reward = 1.0 / self._total_levels

        info = {
            "action": action,
            "engine_state": engine_state,
            "level_changed": self._engine.level_index != prev_level,
            "life_lost": self._engine._lives < prev_lives,
            "full_reset": frame_data.full_reset,
        }

        return StepResult(
            state=self._build_game_state(frame_data),
            reward=reward,
            done=done,
            info=info,
        )

class ArcGameEnv(gym.Env):

    metadata: Dict[str, Any] = {
        "render_modes": ["rgb_array"],
        "render_fps": 5,
    }

    ACTION_LIST: List[str] = ["reset", "up", "down", "left", "right", "undo"]

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

