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
    metadata: Dict[str, Any]


@dataclass
class StepResult:
    state: GameState
    reward: float
    done: bool
    info: Dict[str, Any]


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

FRAME_SIZE = 64
UI_ROWS = 2
PLAY_ROWS = FRAME_SIZE - UI_ROWS

C_EMPTY = 0
C_WALL = 1
C_TRACK_H = 2
C_TRACK_V = 3
C_CURVE_TR = 4
C_CURVE_TL = 5
C_CURVE_BR = 6
C_CURVE_BL = 7
C_SWITCH_A = 8
C_SWITCH_B = 9
C_SIGNAL = 10
C_DEPOT_R = 11
C_DEPOT_G = 12
C_DEPOT_B = 13
C_DEPOT_Y = 14
C_SPAWN_R = 15
C_SPAWN_G = 16
C_SPAWN_B = 17
C_SPAWN_Y = 18
C_CROSS = 19
C_HAZARD = 20
C_ONEWAY_R = 21
C_ONEWAY_D = 22
C_ONEWAY_L = 23
C_ONEWAY_U = 24
C_TUNNEL_A = 25
C_TUNNEL_B = 26
C_BOMB = 27
C_DEFUSE = 28

DIR_R = (1, 0)
DIR_L = (-1, 0)
DIR_D = (0, 1)
DIR_U = (0, -1)

CURVE_CONNECTIONS = {
    C_CURVE_TR: {DIR_L: DIR_U, DIR_D: DIR_R},
    C_CURVE_TL: {DIR_R: DIR_U, DIR_D: DIR_L},
    C_CURVE_BR: {DIR_L: DIR_D, DIR_U: DIR_R},
    C_CURVE_BL: {DIR_R: DIR_D, DIR_U: DIR_L},
}

SWITCH_MODES = {
    C_SWITCH_A: {
        False: {DIR_L: DIR_L, DIR_R: DIR_R, DIR_U: DIR_U, DIR_D: DIR_D},
        True: {DIR_L: DIR_U, DIR_D: DIR_R, DIR_R: DIR_D, DIR_U: DIR_L},
    },
    C_SWITCH_B: {
        False: {DIR_L: DIR_L, DIR_R: DIR_R, DIR_U: DIR_U, DIR_D: DIR_D},
        True: {DIR_R: DIR_U, DIR_D: DIR_L, DIR_L: DIR_D, DIR_U: DIR_R},
    },
}

ONEWAY_ALLOWED = {
    C_ONEWAY_R: DIR_R,
    C_ONEWAY_D: DIR_D,
    C_ONEWAY_L: DIR_L,
    C_ONEWAY_U: DIR_U,
}

TRAIN_RED = 0
TRAIN_GREEN = 1
TRAIN_BLUE = 2
TRAIN_YELLOW = 3

DEPOT_FOR_CELL = {
    C_DEPOT_R: TRAIN_RED,
    C_DEPOT_G: TRAIN_GREEN,
    C_DEPOT_B: TRAIN_BLUE,
    C_DEPOT_Y: TRAIN_YELLOW,
}

SPAWN_TO_COLOR = {
    C_SPAWN_R: TRAIN_RED,
    C_SPAWN_G: TRAIN_GREEN,
    C_SPAWN_B: TRAIN_BLUE,
    C_SPAWN_Y: TRAIN_YELLOW,
}

COL_BG = 0
COL_FLOOR = 1
COL_WALL = 5
COL_TRACK = 5
COL_CURSOR = 12
COL_TRAIN_R = 8
COL_TRAIN_G = 2
COL_TRAIN_B = 9
COL_TRAIN_Y = 14
COL_SWITCH_OFF = 4
COL_SWITCH_ON = 11
COL_SIGNAL_STOP = 6
COL_SIGNAL_GO = 10
COL_CROSS = 13
COL_HAZARD = 6
COL_HAZARD_BOOM = 8
COL_ONEWAY = 15
COL_TUNNEL = 3
COL_BOMB = 6
COL_BOMB_EXPLODE = 8
COL_DEFUSE_ARMED = 14
COL_DEFUSE_SAFE = 10
COL_BAR_FILL = 11
COL_BAR_EMPTY = 5
COL_LIFE = 8
COL_LIFE_EMPTY = 3

TRAIN_COLORS = {
    TRAIN_RED: COL_TRAIN_R,
    TRAIN_GREEN: COL_TRAIN_G,
    TRAIN_BLUE: COL_TRAIN_B,
    TRAIN_YELLOW: COL_TRAIN_Y,
}

GRID_SIZES = [11, 13, 15, 17]
TILE_SIZES = [PLAY_ROWS // g for g in GRID_SIZES]

LEVEL_DATA = [
    {
        "grid": [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 15, 2, 2, 8, 2, 2, 2, 2, 20, 1],
            [1, 0, 0, 0, 3, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 4, 2, 2, 9, 2, 11, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 16, 2, 2, 8, 2, 2, 2, 2, 20, 1],
            [1, 0, 0, 0, 3, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 4, 2, 2, 2, 2, 12, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
        "switches": {(4, 1): False, (4, 6): False, (7, 3): False},
        "signals": {},
        "trains": [
            {"color": TRAIN_RED, "x": 1, "y": 1, "dx": 1, "dy": 0},
            {"color": TRAIN_GREEN, "x": 1, "y": 6, "dx": 1, "dy": 0},
        ],
        "tunnels": {},
        "cursor": [4, 2],
        "switch_links": {(4, 1): (4, 6), (4, 6): (4, 1)},
        "signal_decay": {},
        "inversion_interval": 0,
        "phantom_tracks": {},
        "bombs": [],
        "defuse_cells": {},
        "defuse_links": {},
        "depot_swap_tick": 0,
        "depot_swap_pairs": [],
    },
    {
        "grid": [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 15, 2, 2, 8, 2, 2, 7, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 3, 0, 0, 3, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 4, 2, 7, 4, 2, 2, 27, 0, 1],
            [1, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 1],
            [1, 16, 2, 2, 7, 0, 4, 10, 2, 7, 0, 0, 1],
            [1, 0, 0, 0, 3, 0, 0, 0, 0, 3, 0, 0, 1],
            [1, 0, 0, 0, 4, 2, 2, 2, 8, 4, 2, 11, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 1],
            [1, 17, 2, 2, 8, 2, 2, 20, 4, 2, 2, 12, 1],
            [1, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 4, 2, 2, 10, 2, 2, 2, 13, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
        "switches": {(4, 1): False, (8, 7): False, (4, 9): False},
        "signals": {(7, 5): False, (7, 11): False},
        "trains": [
            {"color": TRAIN_RED, "x": 1, "y": 1, "dx": 1, "dy": 0},
            {"color": TRAIN_GREEN, "x": 1, "y": 5, "dx": 1, "dy": 0},
            {"color": TRAIN_BLUE, "x": 1, "y": 9, "dx": 1, "dy": 0},
        ],
        "tunnels": {},
        "cursor": [4, 2],
        "switch_links": {(4, 1): (8, 7), (8, 7): (4, 1)},
        "signal_decay": {(7, 5): 4, (7, 11): 4},
        "inversion_interval": 0,
        "phantom_tracks": {},
        "bombs": [(10, 3)],
        "defuse_cells": {},
        "defuse_links": {},
        "depot_swap_tick": 0,
        "depot_swap_pairs": [],
    },
    {
        "grid": [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 15, 2, 2, 8, 2, 2, 27, 0, 0, 0, 3, 0, 0, 1],
            [1, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 6, 2, 12, 1],
            [1, 0, 0, 0, 4, 2, 10, 2, 2, 7, 0, 3, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 3, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 4, 2, 19, 2, 11, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 1],
            [1, 16, 2, 2, 8, 2, 2, 20, 0, 0, 0, 3, 0, 0, 1],
            [1, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 3, 0, 0, 1],
            [1, 0, 0, 0, 4, 2, 10, 2, 2, 2, 2, 5, 0, 0, 1],
            [1, 18, 2, 2, 2, 25, 0, 0, 26, 2, 2, 2, 2, 14, 1],
            [1, 17, 2, 2, 8, 2, 2, 27, 0, 0, 0, 0, 0, 0, 1],
            [1, 28, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 4, 2, 10, 2, 2, 2, 27, 2, 2, 13, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
        "switches": {(4, 1): False, (4, 7): False, (4, 11): False},
        "signals": {(6, 3): False, (6, 9): False, (6, 13): False},
        "trains": [
            {"color": TRAIN_RED, "x": 1, "y": 1, "dx": 1, "dy": 0},
            {"color": TRAIN_GREEN, "x": 1, "y": 7, "dx": 1, "dy": 0},
            {"color": TRAIN_BLUE, "x": 1, "y": 11, "dx": 1, "dy": 0},
            {"color": TRAIN_YELLOW, "x": 1, "y": 10, "dx": 1, "dy": 0},
        ],
        "tunnels": {(5, 10): (8, 10)},
        "cursor": [1, 12],
        "switch_links": {(4, 1): (4, 11), (4, 11): (4, 1)},
        "signal_decay": {(6, 3): 3, (6, 9): 3, (6, 13): 3},
        "inversion_interval": 4,
        "phantom_tracks": {},
        "bombs": [(7, 1), (7, 11), (10, 13)],
        "defuse_cells": {(1, 12): 6},
        "defuse_links": {(1, 12): (10, 13)},
        "depot_swap_tick": 0,
        "depot_swap_pairs": [],
    },
    {
        "grid": [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 15, 2, 8, 2, 27, 0, 27, 0, 0, 6, 2, 2, 2, 2, 12, 1],
            [1, 0, 0, 3, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 4, 2, 10, 2, 7, 0, 0, 3, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 3, 0, 0, 3, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 4, 2, 2, 19, 2, 7, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 3, 0, 0, 0, 1],
            [1, 28, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 4, 2, 2, 11, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 1],
            [1, 16, 2, 2, 2, 2, 10, 2, 2, 2, 5, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 17, 2, 2, 2, 2, 8, 2, 27, 0, 3, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 3, 0, 0, 0, 6, 2, 2, 2, 2, 14, 1],
            [1, 0, 0, 0, 0, 0, 4, 2, 10, 2, 19, 2, 27, 2, 2, 13, 1],
            [1, 0, 0, 0, 0, 6, 2, 2, 7, 0, 3, 0, 0, 0, 0, 0, 1],
            [1, 18, 2, 2, 2, 5, 0, 0, 4, 2, 5, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
        "switches": {(3, 1): False, (6, 11): False},
        "signals": {(5, 3): False, (6, 9): False, (8, 13): False},
        "trains": [
            {"color": TRAIN_RED, "x": 1, "y": 1, "dx": 1, "dy": 0},
            {"color": TRAIN_GREEN, "x": 1, "y": 9, "dx": 1, "dy": 0},
            {"color": TRAIN_BLUE, "x": 1, "y": 11, "dx": 1, "dy": 0},
            {"color": TRAIN_YELLOW, "x": 1, "y": 15, "dx": 1, "dy": 0},
        ],
        "tunnels": {},
        "cursor": [1, 7],
        "switch_links": {(3, 1): (6, 11), (6, 11): (3, 1)},
        "signal_decay": {(5, 3): 3, (6, 9): 3, (8, 13): 3},
        "inversion_interval": 4,
        "phantom_tracks": {},
        "bombs": [(5, 1), (7, 1), (8, 11), (12, 13)],
        "defuse_cells": {(1, 7): 7},
        "defuse_links": {(1, 7): (12, 13)},
        "depot_swap_tick": 0,
        "depot_swap_pairs": [],
    },
]

MOVE_LIMITS = [936, 2340, 3420, 4860]

_sprite_cache = {}


class MoveDisplay(RenderableUserDisplay):
    MAX_LIVES = 3

    def __init__(self, game):
        self._game = game
        self.max_moves = 0
        self.remaining = 0

    def set_limit(self, max_moves):
        self.max_moves = max_moves
        self.remaining = max_moves

    def decrement(self):
        self.remaining = max(0, self.remaining - 6)

    def reset(self):
        self.remaining = self.max_moves

    def render_interface(self, frame):
        if self.max_moves == 0:
            return frame
        fw = frame.shape[1]
        fh = frame.shape[0]
        rs = fh - 2
        re = fh
        frame[rs:re, :] = COL_BAR_EMPTY
        bw = int(fw * 0.7)
        filled = int(bw * self.remaining / self.max_moves)
        for x in range(filled):
            frame[rs:re, x] = COL_BAR_FILL
        lives = getattr(self._game, "_lives", self.MAX_LIVES)
        ls = bw
        lw = fw - bw
        lbw = 2
        lgap = 2
        tlw = (lbw * self.MAX_LIVES) + (lgap * (self.MAX_LIVES - 1))
        off = (lw - tlw) // 2
        for i in range(self.MAX_LIVES):
            xs = ls + off + i * (lbw + lgap)
            xe = xs + lbw
            c = COL_LIFE if i < lives else COL_LIFE_EMPTY
            if xe <= fw:
                frame[rs:re, xs:xe] = c
        return frame


def _make_tile(color, size):
    return [[color] * size for _ in range(size)]


def _tile_border(inner, border, size):
    if size <= 2:
        return _make_tile(inner, size)
    t = []
    for r in range(size):
        row = []
        for c in range(size):
            if r == 0 or r == size - 1 or c == 0 or c == size - 1:
                row.append(border)
            else:
                row.append(inner)
        t.append(row)
    return t


def _tile_arrow(bg, arrow_c, size, dx, dy):
    t = _make_tile(bg, size)
    mid = size // 2
    if size >= 3:
        if dx == 1:
            for i in range(size):
                t[mid][i] = arrow_c
            if mid - 1 >= 0:
                t[mid - 1][size - 2] = arrow_c
            if mid + 1 < size:
                t[mid + 1][size - 2] = arrow_c
        elif dx == -1:
            for i in range(size):
                t[mid][i] = arrow_c
            if mid - 1 >= 0:
                t[mid - 1][1] = arrow_c
            if mid + 1 < size:
                t[mid + 1][1] = arrow_c
        elif dy == 1:
            for i in range(size):
                t[i][mid] = arrow_c
            if mid - 1 >= 0:
                t[size - 2][mid - 1] = arrow_c
            if mid + 1 < size:
                t[size - 2][mid + 1] = arrow_c
        elif dy == -1:
            for i in range(size):
                t[i][mid] = arrow_c
            if mid - 1 >= 0:
                t[1][mid - 1] = arrow_c
            if mid + 1 < size:
                t[1][mid + 1] = arrow_c
    return t


def _tile_cross_track(bg, track_c, size):
    t = _make_tile(bg, size)
    mid = size // 2
    if size >= 3:
        for i in range(size):
            t[mid][i] = track_c
            t[i][mid] = track_c
    return t


def _tile_h_track(bg, track_c, size):
    t = _make_tile(bg, size)
    mid = size // 2
    for i in range(size):
        t[mid][i] = track_c
    return t


def _tile_v_track(bg, track_c, size):
    t = _make_tile(bg, size)
    mid = size // 2
    for i in range(size):
        t[i][mid] = track_c
    return t


def _tile_curve(bg, track_c, size, ctype):
    t = _make_tile(bg, size)
    mid = size // 2
    if size >= 3:
        if ctype == C_CURVE_TR:
            for i in range(mid, size):
                t[mid][i] = track_c
            for i in range(mid + 1):
                t[i][mid] = track_c
        elif ctype == C_CURVE_TL:
            for i in range(mid + 1):
                t[mid][i] = track_c
            for i in range(mid + 1):
                t[i][mid] = track_c
        elif ctype == C_CURVE_BR:
            for i in range(mid, size):
                t[mid][i] = track_c
            for i in range(mid, size):
                t[i][mid] = track_c
        elif ctype == C_CURVE_BL:
            for i in range(mid + 1):
                t[mid][i] = track_c
            for i in range(mid, size):
                t[i][mid] = track_c
    return t


def _tile_train(bg, train_c, size):
    t = _make_tile(bg, size)
    if size >= 4:
        for r in range(1, size - 1):
            for c in range(1, size - 1):
                t[r][c] = train_c
    elif size >= 2:
        for r in range(size):
            for c in range(size):
                t[r][c] = train_c
    return t


def _get_sprites(ts):
    if ts in _sprite_cache:
        return _sprite_cache[ts]
    s = {}
    s["empty"] = Sprite(
        pixels=_make_tile(COL_FLOOR, ts), name="empty", visible=True, collidable=False
    )
    s["wall"] = Sprite(
        pixels=_make_tile(COL_WALL, ts), name="wall", visible=True, collidable=False
    )
    s["track_h"] = Sprite(
        pixels=_tile_h_track(COL_FLOOR, COL_TRACK, ts),
        name="th",
        visible=True,
        collidable=False,
    )
    s["track_v"] = Sprite(
        pixels=_tile_v_track(COL_FLOOR, COL_TRACK, ts),
        name="tv",
        visible=True,
        collidable=False,
    )
    s["curve_tr"] = Sprite(
        pixels=_tile_curve(COL_FLOOR, COL_TRACK, ts, C_CURVE_TR),
        name="ctr",
        visible=True,
        collidable=False,
    )
    s["curve_tl"] = Sprite(
        pixels=_tile_curve(COL_FLOOR, COL_TRACK, ts, C_CURVE_TL),
        name="ctl",
        visible=True,
        collidable=False,
    )
    s["curve_br"] = Sprite(
        pixels=_tile_curve(COL_FLOOR, COL_TRACK, ts, C_CURVE_BR),
        name="cbr",
        visible=True,
        collidable=False,
    )
    s["curve_bl"] = Sprite(
        pixels=_tile_curve(COL_FLOOR, COL_TRACK, ts, C_CURVE_BL),
        name="cbl",
        visible=True,
        collidable=False,
    )
    s["switch_off"] = Sprite(
        pixels=_tile_border(COL_SWITCH_OFF, COL_TRACK, ts),
        name="swoff",
        visible=True,
        collidable=False,
    )
    s["switch_on"] = Sprite(
        pixels=_tile_border(COL_SWITCH_ON, COL_TRACK, ts),
        name="swon",
        visible=True,
        collidable=False,
    )
    s["signal_stop"] = Sprite(
        pixels=_tile_border(COL_SIGNAL_STOP, COL_FLOOR, ts),
        name="sigst",
        visible=True,
        collidable=False,
    )
    s["signal_go"] = Sprite(
        pixels=_tile_border(COL_SIGNAL_GO, COL_FLOOR, ts),
        name="siggo",
        visible=True,
        collidable=False,
    )
    s["cross"] = Sprite(
        pixels=_tile_cross_track(COL_FLOOR, COL_CROSS, ts),
        name="cross",
        visible=True,
        collidable=False,
    )
    s["hazard"] = Sprite(
        pixels=_tile_border(COL_HAZARD, COL_FLOOR, ts),
        name="haz",
        visible=True,
        collidable=False,
    )
    s["hazard_boom"] = Sprite(
        pixels=_make_tile(COL_HAZARD_BOOM, ts),
        name="boom",
        visible=True,
        collidable=False,
    )
    s["cursor"] = Sprite(
        pixels=_tile_border(COL_BG, COL_CURSOR, ts),
        name="cursor",
        visible=True,
        collidable=False,
    )
    s["oneway_r"] = Sprite(
        pixels=_tile_arrow(COL_FLOOR, COL_ONEWAY, ts, 1, 0),
        name="owr",
        visible=True,
        collidable=False,
    )
    s["oneway_d"] = Sprite(
        pixels=_tile_arrow(COL_FLOOR, COL_ONEWAY, ts, 0, 1),
        name="owd",
        visible=True,
        collidable=False,
    )
    s["oneway_l"] = Sprite(
        pixels=_tile_arrow(COL_FLOOR, COL_ONEWAY, ts, -1, 0),
        name="owl",
        visible=True,
        collidable=False,
    )
    s["oneway_u"] = Sprite(
        pixels=_tile_arrow(COL_FLOOR, COL_ONEWAY, ts, 0, -1),
        name="owu",
        visible=True,
        collidable=False,
    )
    s["tunnel"] = Sprite(
        pixels=_tile_border(COL_TUNNEL, COL_WALL, ts),
        name="tun",
        visible=True,
        collidable=False,
    )
    s["bomb"] = Sprite(
        pixels=_tile_border(COL_BOMB, COL_WALL, ts),
        name="bomb",
        visible=True,
        collidable=False,
    )
    s["bomb_explode"] = Sprite(
        pixels=_make_tile(COL_BOMB_EXPLODE, ts),
        name="bombx",
        visible=True,
        collidable=False,
    )
    s["defuse_armed"] = Sprite(
        pixels=_tile_border(COL_DEFUSE_ARMED, COL_FLOOR, ts),
        name="dfarm",
        visible=True,
        collidable=False,
    )
    s["defuse_safe"] = Sprite(
        pixels=_tile_border(COL_DEFUSE_SAFE, COL_FLOOR, ts),
        name="dfsaf",
        visible=True,
        collidable=False,
    )
    for tc, col in TRAIN_COLORS.items():
        s[f"train_{tc}"] = Sprite(
            pixels=_tile_train(COL_FLOOR, col, ts),
            name=f"tr{tc}",
            visible=True,
            collidable=False,
        )
        s[f"depot_{tc}"] = Sprite(
            pixels=_tile_border(col, COL_WALL, ts),
            name=f"dp{tc}",
            visible=True,
            collidable=False,
        )
        s[f"depot_{tc}_done"] = Sprite(
            pixels=_tile_border(col, COL_SIGNAL_GO, ts),
            name=f"dpd{tc}",
            visible=True,
            collidable=False,
        )
        s[f"spawn_{tc}"] = Sprite(
            pixels=_tile_border(col, COL_TRACK, ts),
            name=f"sp{tc}",
            visible=True,
            collidable=False,
        )
    _sprite_cache[ts] = s
    return s


CELL_SPRITE_MAP = {
    C_WALL: "wall",
    C_TRACK_H: "track_h",
    C_TRACK_V: "track_v",
    C_CURVE_TR: "curve_tr",
    C_CURVE_TL: "curve_tl",
    C_CURVE_BR: "curve_br",
    C_CURVE_BL: "curve_bl",
    C_CROSS: "cross",
    C_ONEWAY_R: "oneway_r",
    C_ONEWAY_D: "oneway_d",
    C_ONEWAY_L: "oneway_l",
    C_ONEWAY_U: "oneway_u",
    C_TUNNEL_A: "tunnel",
    C_TUNNEL_B: "tunnel",
}

TRACK_CELLS = {
    C_TRACK_H,
    C_TRACK_V,
    C_CURVE_TR,
    C_CURVE_TL,
    C_CURVE_BR,
    C_CURVE_BL,
    C_SWITCH_A,
    C_SWITCH_B,
    C_SIGNAL,
    C_CROSS,
    C_HAZARD,
    C_BOMB,
    C_ONEWAY_R,
    C_ONEWAY_D,
    C_ONEWAY_L,
    C_ONEWAY_U,
    C_TUNNEL_A,
    C_TUNNEL_B,
    C_DEPOT_R,
    C_DEPOT_G,
    C_DEPOT_B,
    C_DEPOT_Y,
    C_SPAWN_R,
    C_SPAWN_G,
    C_SPAWN_B,
    C_SPAWN_Y,
}

levels = [
    Level(sprites=[], grid_size=(FRAME_SIZE, FRAME_SIZE)),
    Level(sprites=[], grid_size=(FRAME_SIZE, FRAME_SIZE)),
    Level(sprites=[], grid_size=(FRAME_SIZE, FRAME_SIZE)),
    Level(sprites=[], grid_size=(FRAME_SIZE, FRAME_SIZE)),
]

BACKGROUND_COLOR = 0
PADDING_COLOR = 0


def _next_dir(cell_type, incoming_dir, switch_state=False):
    if cell_type == C_TRACK_H:
        if incoming_dir in (DIR_L, DIR_R):
            return incoming_dir
        return None
    if cell_type == C_TRACK_V:
        if incoming_dir in (DIR_U, DIR_D):
            return incoming_dir
        return None
    if cell_type in CURVE_CONNECTIONS:
        return CURVE_CONNECTIONS[cell_type].get(incoming_dir)
    if cell_type in (C_SWITCH_A, C_SWITCH_B):
        modes = SWITCH_MODES[cell_type]
        return modes[switch_state].get(incoming_dir)
    if cell_type == C_CROSS:
        return incoming_dir
    if cell_type in ONEWAY_ALLOWED:
        allowed = ONEWAY_ALLOWED[cell_type]
        if incoming_dir == allowed:
            return incoming_dir
        return None
    if cell_type in DEPOT_FOR_CELL:
        return incoming_dir
    if cell_type in SPAWN_TO_COLOR:
        return incoming_dir
    if cell_type == C_HAZARD:
        return incoming_dir
    if cell_type in (C_TUNNEL_A, C_TUNNEL_B):
        return incoming_dir
    if cell_type == C_SIGNAL:
        return incoming_dir
    if cell_type == C_BOMB:
        return incoming_dir
    return None


class Rd01(ARCBaseGame):
    def __init__(self, seed=0):
        self._seed = seed
        self._grid = []
        self._grid_w = 0
        self._grid_h = 0
        self._cursor = [1, 1]
        self._switches = {}
        self._signals = {}
        self._trains = []
        self._trains_delivered = set()
        self._tunnels = {}
        self._depots_done = set()
        self._hazards_blown = set()
        self._move_count = 0
        self._lives = 3
        self._max_moves = 0
        self._moves_remaining = 0
        self._game_over = False
        self._reset_count = 0
        self._tick_rate = 3
        self._init_state = {}
        self._switch_links = {}
        self._signal_decay = {}
        self._signal_open_tick = {}
        self._inversion_interval = 0
        self._tick_count = 0
        self._phantom_tracks = {}
        self._bombs = set()
        self._bombs_exploded = set()
        self._defuse_cells = {}
        self._defused = set()
        self._defuse_deadline = 0
        self._defuse_links = {}
        self._depot_swap_tick = 0
        self._depot_swap_pairs = []
        self._depot_swapped = False
        self._history = []
        self._move_display = MoveDisplay(self)
        camera = Camera(
            0, 0, 64, 64, BACKGROUND_COLOR, PADDING_COLOR, [self._move_display]
        )
        available_actions = [0, 1, 2, 3, 4, 5, 7]
        super().__init__(
            game_id="rd01",
            levels=levels,
            camera=camera,
            available_actions=available_actions,
        )

    def on_set_level(self, level):
        self._game_over = False
        self._lives = 3
        self._move_count = 0
        self._history = []
        self._load_level()
        self._save_init_state()
        self._render_level(level)
        self._move_display.set_limit(self._max_moves)

    def _load_level(self):
        idx = self.level_index
        data = LEVEL_DATA[idx]
        self._grid = [row[:] for row in data["grid"]]
        self._grid_h = len(self._grid)
        self._grid_w = len(self._grid[0])
        self._cursor = list(data["cursor"])
        self._switches = {(k[0], k[1]): v for k, v in data["switches"].items()}
        self._signals = {(k[0], k[1]): v for k, v in data["signals"].items()}
        self._trains = []
        for td in data["trains"]:
            self._trains.append(
                {
                    "color": td["color"],
                    "x": td["x"],
                    "y": td["y"],
                    "dx": td["dx"],
                    "dy": td["dy"],
                    "alive": True,
                    "delivered": False,
                }
            )
        self._trains_delivered = set()
        self._tunnels = {}
        for k, v in data["tunnels"].items():
            self._tunnels[(k[0], k[1])] = (v[0], v[1])
            self._tunnels[(v[0], v[1])] = (k[0], k[1])
        self._depots_done = set()
        self._hazards_blown = set()
        self._max_moves = MOVE_LIMITS[idx]
        self._moves_remaining = self._max_moves
        self._switch_links = {}
        for k, v in data.get("switch_links", {}).items():
            self._switch_links[(k[0], k[1])] = (v[0], v[1])
        self._signal_decay = {}
        for k, v in data.get("signal_decay", {}).items():
            self._signal_decay[(k[0], k[1])] = v
        self._signal_open_tick = {}
        self._inversion_interval = data.get("inversion_interval", 0)
        self._tick_count = 0
        self._phantom_tracks = {}
        for k, v in data.get("phantom_tracks", {}).items():
            self._phantom_tracks[(k[0], k[1])] = v
        self._bombs = set()
        for pos in data.get("bombs", []):
            self._bombs.add((pos[0], pos[1]))
        self._bombs_exploded = set()
        self._defuse_cells = {}
        for k, v in data.get("defuse_cells", {}).items():
            self._defuse_cells[(k[0], k[1])] = v
        self._defused = set()
        self._defuse_deadline = data.get("defuse_deadline", 0)
        self._defuse_links = {}
        for k, v in data.get("defuse_links", {}).items():
            self._defuse_links[(k[0], k[1])] = (v[0], v[1])
        self._depot_swap_tick = data.get("depot_swap_tick", 0)
        self._depot_swap_pairs = []
        for pair in data.get("depot_swap_pairs", []):
            self._depot_swap_pairs.append(
                ((pair[0][0], pair[0][1]), (pair[1][0], pair[1][1]))
            )
        self._depot_swapped = False

    def _save_init_state(self):
        self._init_state = {
            "grid": [row[:] for row in self._grid],
            "cursor": self._cursor[:],
            "switches": dict(self._switches),
            "signals": dict(self._signals),
            "trains": [dict(t) for t in self._trains],
            "max_moves": self._max_moves,
            "switch_links": dict(self._switch_links),
            "signal_decay": dict(self._signal_decay),
            "signal_open_tick": dict(self._signal_open_tick),
            "inversion_interval": self._inversion_interval,
            "tick_count": self._tick_count,
            "phantom_tracks": dict(self._phantom_tracks),
            "bombs": set(self._bombs),
            "bombs_exploded": set(self._bombs_exploded),
            "defuse_cells": dict(self._defuse_cells),
            "defused": set(self._defused),
            "defuse_deadline": self._defuse_deadline,
            "defuse_links": dict(self._defuse_links),
            "depot_swap_tick": self._depot_swap_tick,
            "depot_swap_pairs": list(self._depot_swap_pairs),
            "depot_swapped": self._depot_swapped,
        }

    def _restore_init_state(self):
        saved = self._init_state
        self._grid = [row[:] for row in saved["grid"]]
        self._cursor = saved["cursor"][:]
        self._switches = dict(saved["switches"])
        self._signals = dict(saved["signals"])
        self._trains = [dict(t) for t in saved["trains"]]
        self._trains_delivered = set()
        self._depots_done = set()
        self._hazards_blown = set()
        self._max_moves = saved["max_moves"]
        self._moves_remaining = self._max_moves
        self._move_count = 0
        self._game_over = False
        self._history = []
        self._move_display.set_limit(self._max_moves)
        self._switch_links = dict(saved["switch_links"])
        self._signal_decay = dict(saved["signal_decay"])
        self._signal_open_tick = dict(saved["signal_open_tick"])
        self._inversion_interval = saved["inversion_interval"]
        self._tick_count = saved["tick_count"]
        self._phantom_tracks = dict(saved["phantom_tracks"])
        self._bombs = set(saved["bombs"])
        self._bombs_exploded = set(saved["bombs_exploded"])
        self._defuse_cells = dict(saved["defuse_cells"])
        self._defused = set(saved["defused"])
        self._defuse_deadline = saved["defuse_deadline"]
        self._defuse_links = dict(saved["defuse_links"])
        self._depot_swap_tick = saved["depot_swap_tick"]
        self._depot_swap_pairs = list(saved["depot_swap_pairs"])
        self._depot_swapped = saved["depot_swapped"]

    def _get_cell_type(self, x, y):
        pos = (x, y)
        if pos in self._phantom_tracks:
            return self._phantom_tracks[pos]
        return self._grid[y][x]

    def _advance_trains(self):
        self._tick_count += 1
        if (
            self._depot_swap_tick > 0
            and self._tick_count == self._depot_swap_tick
            and not self._depot_swapped
        ):
            self._depot_swapped = True
            for (x1, y1), (x2, y2) in self._depot_swap_pairs:
                self._grid[y1][x1], self._grid[y2][x2] = (
                    self._grid[y2][x2],
                    self._grid[y1][x1],
                )
        for i, t in enumerate(self._trains):
            if not t["alive"] or t["delivered"]:
                continue
            cx, cy = t["x"], t["y"]
            cdx, cdy = t["dx"], t["dy"]
            cell = self._get_cell_type(cx, cy)
            sw_state = self._switches.get((cx, cy), False)
            out_dir = _next_dir(cell, (cdx, cdy), sw_state)
            if out_dir is None:
                t["alive"] = False
                self._game_over = True
                continue
            nx, ny = cx + out_dir[0], cy + out_dir[1]
            ndx, ndy = out_dir
            if nx < 0 or ny < 0 or nx >= self._grid_w or ny >= self._grid_h:
                t["alive"] = False
                self._game_over = True
                continue
            ncell = self._get_cell_type(nx, ny)
            if ncell == C_WALL or ncell == C_EMPTY or ncell == C_DEFUSE:
                t["alive"] = False
                self._game_over = True
                continue
            if ncell == C_SIGNAL and not self._signals.get((nx, ny), False):
                continue
            if ncell in ONEWAY_ALLOWED:
                allowed = ONEWAY_ALLOWED[ncell]
                if (ndx, ndy) != allowed:
                    t["alive"] = False
                    self._game_over = True
                    continue
            if ncell in (C_TUNNEL_A, C_TUNNEL_B):
                dest = self._tunnels.get((nx, ny))
                if dest:
                    nx, ny = dest
                    ncell = self._get_cell_type(nx, ny)
            if ncell == C_HAZARD:
                self._hazards_blown.add((nx, ny))
                t["alive"] = False
                self._game_over = True
                continue
            if ncell == C_BOMB:
                if (nx, ny) in self._bombs and (nx, ny) not in self._defused:
                    self._bombs_exploded.add((nx, ny))
                    t["alive"] = False
                    self._game_over = True
                    continue
            if ncell in DEPOT_FOR_CELL:
                depot_color = DEPOT_FOR_CELL[ncell]
                if depot_color == t["color"]:
                    t["delivered"] = True
                    t["x"] = nx
                    t["y"] = ny
                    self._depots_done.add((nx, ny))
                    self._trains_delivered.add(i)
                else:
                    t["alive"] = False
                    self._game_over = True
                continue
            t["x"] = nx
            t["y"] = ny
            t["dx"] = ndx
            t["dy"] = ndy
        positions = {}
        for i, t in enumerate(self._trains):
            if not t["alive"] or t["delivered"]:
                continue
            pos = (t["x"], t["y"])
            if pos in positions:
                self._game_over = True
                t["alive"] = False
                self._trains[positions[pos]]["alive"] = False
            else:
                positions[pos] = i
        for sig_pos, decay_ticks in self._signal_decay.items():
            if self._signals.get(sig_pos, False):
                open_tick = self._signal_open_tick.get(sig_pos, 0)
                if self._tick_count - open_tick >= decay_ticks:
                    self._signals[sig_pos] = False
        if (
            self._inversion_interval > 0
            and self._tick_count % self._inversion_interval == 0
        ):
            for sw_pos in self._switches:
                self._switches[sw_pos] = not self._switches[sw_pos]
        for df_pos, deadline in self._defuse_cells.items():
            if df_pos not in self._defused and self._tick_count >= deadline:
                self._game_over = True

    def _check_win(self):
        for t in self._trains:
            if not t["delivered"]:
                return False
        return True

    def _resolve_sprite_key(self, x, y):
        ct = self._grid[y][x]
        pos = (x, y)
        if pos in self._phantom_tracks:
            return "empty"
        if ct == C_EMPTY:
            return "empty"
        if ct in CELL_SPRITE_MAP:
            return CELL_SPRITE_MAP[ct]
        if ct in (C_SWITCH_A, C_SWITCH_B):
            return "switch_on" if self._switches.get(pos, False) else "switch_off"
        if ct == C_SIGNAL:
            return "signal_go" if self._signals.get(pos, False) else "signal_stop"
        if ct == C_HAZARD:
            if pos in self._hazards_blown:
                return "hazard_boom"
            return "hazard"
        if ct == C_BOMB:
            if pos in self._bombs_exploded:
                return "bomb_explode"
            return "bomb"
        if ct == C_DEFUSE:
            if pos in self._defused:
                return "defuse_safe"
            return "defuse_armed"
        if ct in DEPOT_FOR_CELL:
            tc = DEPOT_FOR_CELL[ct]
            if pos in self._depots_done:
                return f"depot_{tc}_done"
            return f"depot_{tc}"
        if ct in SPAWN_TO_COLOR:
            tc = SPAWN_TO_COLOR[ct]
            return f"spawn_{tc}"
        return "empty"

    def _render_level(self, level):
        for sp in list(level._sprites):
            level.remove_sprite(sp)
        gs = max(self._grid_w, self._grid_h)
        ts = PLAY_ROWS // gs
        if ts < 2:
            ts = 2
        offset_x = (FRAME_SIZE - ts * self._grid_w) // 2
        offset_y = (PLAY_ROWS - ts * self._grid_h) // 2
        spr = _get_sprites(ts)
        for y in range(self._grid_h):
            for x in range(self._grid_w):
                px = offset_x + x * ts
                py = offset_y + y * ts
                key = self._resolve_sprite_key(x, y)
                level.add_sprite(spr[key].clone().set_position(px, py))
        for t in self._trains:
            if not t["alive"] or t["delivered"]:
                continue
            tx = offset_x + t["x"] * ts
            ty = offset_y + t["y"] * ts
            level.add_sprite(spr[f"train_{t['color']}"].clone().set_position(tx, ty))
        cx, cy = self._cursor
        cpx = offset_x + cx * ts
        cpy = offset_y + cy * ts
        level.add_sprite(spr["cursor"].clone().set_position(cpx, cpy))

    def _restart_level(self):
        self._lives -= 1
        if self._lives <= 0:
            self._game_over = True
            self.lose()
            return
        self._restore_init_state()
        self._render_level(self.current_level)

    def _save_state(self):
        self._history.append(
            {
                "grid": [row[:] for row in self._grid],
                "cursor": self._cursor[:],
                "switches": dict(self._switches),
                "signals": dict(self._signals),
                "trains": [dict(t) for t in self._trains],
                "trains_delivered": set(self._trains_delivered),
                "depots_done": set(self._depots_done),
                "hazards_blown": set(self._hazards_blown),
                "tick_count": self._tick_count,
                "signal_open_tick": dict(self._signal_open_tick),
                "phantom_tracks": dict(self._phantom_tracks),
                "bombs": set(self._bombs),
                "bombs_exploded": set(self._bombs_exploded),
                "defused": set(self._defused),
                "depot_swapped": self._depot_swapped,
            }
        )

    def _do_undo(self):
        if not self._history:
            return
        snap = self._history.pop()
        self._grid = snap["grid"]
        self._cursor = snap["cursor"]
        self._switches = snap["switches"]
        self._signals = snap["signals"]
        self._trains = snap["trains"]
        self._trains_delivered = snap["trains_delivered"]
        self._depots_done = snap["depots_done"]
        self._hazards_blown = snap["hazards_blown"]
        self._tick_count = snap["tick_count"]
        self._signal_open_tick = snap["signal_open_tick"]
        self._phantom_tracks = snap["phantom_tracks"]
        self._bombs = snap["bombs"]
        self._bombs_exploded = snap["bombs_exploded"]
        self._defused = snap["defused"]
        self._depot_swapped = snap["depot_swapped"]
        self._game_over = False

    def step(self):
        if not self.action:
            self.complete_action()
            return
        if self.action.id == GameAction.RESET:
            self.complete_action()
            return
        self._reset_count = 0
        if self._game_over:
            self.complete_action()
            return
        action_id = self.action.id
        if action_id == GameAction.ACTION7:
            if self._history:
                self._do_undo()
            self._moves_remaining = max(0, self._moves_remaining - 6)
            self._move_display.decrement()
            self._move_count += 1
            if self._moves_remaining <= 0 and not self._check_win():
                self._restart_level()
            else:
                self._render_level(self.current_level)
            self.complete_action()
            return
        self._save_state()
        dx, dy = 0, 0
        do_action5 = False
        if action_id == GameAction.ACTION1:
            dy = -1
        elif action_id == GameAction.ACTION2:
            dy = 1
        elif action_id == GameAction.ACTION3:
            dx = -1
        elif action_id == GameAction.ACTION4:
            dx = 1
        elif action_id == GameAction.ACTION5:
            do_action5 = True
        if dx == 0 and dy == 0 and not do_action5:
            if self._history:
                self._history.pop()
            self.complete_action()
            return
        self._moves_remaining = max(0, self._moves_remaining - 6)
        self._move_display.decrement()
        self._move_count += 1
        if do_action5:
            cx, cy = self._cursor
            pos = (cx, cy)
            ct = self._grid[cy][cx]
            if ct in (C_SWITCH_A, C_SWITCH_B):
                self._switches[pos] = not self._switches.get(pos, False)
                linked = self._switch_links.get(pos)
                if linked and linked in self._switches:
                    self._switches[linked] = not self._switches[linked]
            elif ct == C_SIGNAL:
                new_state = not self._signals.get(pos, False)
                self._signals[pos] = new_state
                if new_state:
                    self._signal_open_tick[pos] = self._tick_count
            elif ct == C_DEFUSE:
                if pos not in self._defused:
                    self._defused.add(pos)
                    linked_bomb = self._defuse_links.get(pos)
                    if linked_bomb:
                        self._defused.add(linked_bomb)
            else:
                self._advance_trains()
        else:
            nx, ny = self._cursor[0] + dx, self._cursor[1] + dy
            if 0 <= nx < self._grid_w and 0 <= ny < self._grid_h:
                if self._grid[ny][nx] != C_WALL:
                    self._cursor = [nx, ny]
        if self._game_over:
            self._restart_level()
            self.complete_action()
            return
        if self._moves_remaining <= 0 and not self._check_win():
            self._restart_level()
            self.complete_action()
            return
        self._render_level(self.current_level)
        if self._check_win():
            self.next_level()
        self.complete_action()

    def reset(self):
        self._reset_count += 1
        if self._reset_count >= 2:
            self._reset_count = 0
            self.set_level(0)
            return
        if self._game_over:
            self._lives = 3
        self._restore_init_state()
        self._render_level(self.current_level)

    def get_actions(self):
        return [
            GameAction.ACTION1,
            GameAction.ACTION2,
            GameAction.ACTION3,
            GameAction.ACTION4,
            GameAction.ACTION5,
        ]


class PuzzleEnvironment:
    _VALID_ACTIONS = ["reset", "up", "down", "left", "right", "select", "undo"]
    _ACTION_MAP = {
        "up": GameAction.ACTION1,
        "down": GameAction.ACTION2,
        "left": GameAction.ACTION3,
        "right": GameAction.ACTION4,
        "select": GameAction.ACTION5,
        "undo": GameAction.ACTION7,
    }

    def __init__(self, seed: int = 0):
        self._engine = Rd01(seed=seed)
        self._total_levels = len(self._engine._levels)
        self._total_turns = 0
        self._last_action_was_reset = False
        self._game_won = False
        self._palette = np.array(ARC_PALETTE, dtype=np.uint8)

    def reset(self):
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

    def step(self, action):
        e = self._engine

        parts = action.split()
        action_key = parts[0] if parts else action

        if action_key == "reset":
            state = self.reset()
            return StepResult(
                state=state, reward=0.0, done=False, info={"action": "reset"}
            )

        if action_key not in self._ACTION_MAP:
            raise ValueError(
                "Invalid action '{}'. Must be one of {}".format(
                    action_key, list(self._ACTION_MAP.keys())
                )
            )

        if e._game_over or self._game_won:
            return StepResult(
                state=self._build_game_state(done=self._game_won),
                reward=0.0,
                done=self._game_won,
                info={"action": action},
            )

        self._last_action_was_reset = False
        self._total_turns += 1

        game_action = self._ACTION_MAP[action_key]
        info: Dict = {"action": action}

        total_levels = self._total_levels
        level_before = e.level_index

        action_input = ActionInput(id=game_action)
        frame = e.perform_action(action_input, raw=True)

        game_won = frame and frame.state and frame.state.name == "WIN"
        level_advanced = e.level_index > level_before

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
                state=self._build_game_state(done=False, info=info),
                reward=0.0,
                done=False,
                info=info,
            )

        return StepResult(
            state=self._build_game_state(done=False, info=info),
            reward=reward,
            done=False,
            info=info,
        )

    def get_actions(self):
        if self._engine._game_over or self._game_won:
            return ["reset"]
        return list(self._VALID_ACTIONS)

    def is_done(self):
        return self._game_won

    def render(self, mode="rgb_array"):
        if self._engine is None:
            return np.zeros((FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8)
        sprites = self._engine.current_level.get_sprites()
        frame = self._engine.camera.render(sprites)
        rgb = self._palette[frame]
        return rgb.astype(np.uint8)

    def close(self):
        self._engine = None

    def _build_text_observation(self):
        e = self._engine
        if e is None:
            return "Game closed."
        level_num = e.level_index + 1
        total = self._total_levels
        moves = e._moves_remaining
        lives = e._lives
        gw = e._grid_w
        gh = e._grid_h
        cx, cy = e._cursor
        cnames = {0: "R", 1: "G", 2: "B", 3: "Y"}
        dnames = {(1, 0): "R", (-1, 0): "L", (0, 1): "D", (0, -1): "U"}
        trains_parts = []
        for t in e._trains:
            cn = cnames.get(t["color"], "?")
            dn = dnames.get((t["dx"], t["dy"]), "?")
            st = ""
            if t.get("delivered"):
                st = "*"
            elif not t.get("alive"):
                st = "X"
            trains_parts.append(f"{cn}({t['x']},{t['y']}){dn}{st}")
        sw_parts = []
        for pos, val in e._switches.items():
            sw_parts.append(f"({pos[0]},{pos[1]})={'ON' if val else 'OFF'}")
        sig_parts = []
        for pos, val in e._signals.items():
            sig_parts.append(f"({pos[0]},{pos[1]})={'OPEN' if val else 'SHUT'}")
        if self._game_won:
            status = "DONE"
        elif e._game_over:
            status = "GAME_OVER"
        else:
            status = "IN_PROGRESS"
        levels_completed = self._total_levels if self._game_won else e.level_index
        score = levels_completed / self._total_levels if self._total_levels > 0 else 0.0
        lines = [
            f"LEVEL {level_num}/{total} MOVES {moves} LIVES {lives}",
            f"GRID {gw}x{gh} CURSOR ({cx},{cy})",
            f"TRAINS: {' '.join(trains_parts) if trains_parts else 'none'}",
            f"SWITCHES: {' '.join(sw_parts) if sw_parts else 'none'}",
            f"SIGNALS: {' '.join(sig_parts) if sig_parts else 'none'}",
            f"STATUS: {status} SCORE: {score:.4f}",
        ]
        return "\n".join(lines)

    def _build_game_state(self, done=False, info=None):
        e = self._engine
        text_obs = self._build_text_observation()
        try:
            rgb = self.render()
            img_bytes = self._frame_to_png(rgb)
        except Exception:
            img_bytes = None
        valid = self.get_actions() if not done else None
        go = e._game_over if e is not None else False
        levels_completed = (
            self._total_levels
            if self._game_won
            else (e.level_index if e is not None else 0)
        )
        score = levels_completed / self._total_levels if self._total_levels > 0 else 0.0
        meta = {
            "total_levels": self._total_levels,
            "level_index": (e.level_index if e is not None else 0),
            "levels_completed": levels_completed,
            "game_over": go,
            "done": done,
            "info": info
            or {
                "score": score,
                "lives": (e._lives if e is not None else 0),
                "moves_remaining": (e._moves_remaining if e is not None else 0),
            },
        }
        return GameState(
            text_observation=text_obs,
            image_observation=img_bytes,
            valid_actions=valid,
            turn=self._total_turns,
            metadata=meta,
        )

    @staticmethod
    def _frame_to_png(rgb_frame):
        h, w = rgb_frame.shape[:2]
        raw = bytearray()
        for y in range(h):
            raw.append(0)
            for x in range(w):
                raw.extend(rgb_frame[y, x].tobytes())
        compressed = zlib.compress(bytes(raw))

        def _chunk(ctype, data):
            c = ctype + data
            crc = struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
            return struct.pack(">I", len(data)) + c + crc

        sig = b"\x89PNG\r\n\x1a\n"
        ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
        return (
            sig
            + _chunk(b"IHDR", ihdr)
            + _chunk(b"IDAT", compressed)
            + _chunk(b"IEND", b"")
        )


class ArcGameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    ACTION_LIST = ["reset", "up", "down", "left", "right", "select", "undo"]

    def __init__(self, seed=0, render_mode=None):
        super().__init__()
        self._seed = seed
        self._env: Optional[PuzzleEnvironment] = None
        self.render_mode = render_mode
        self.observation_space = spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8)
        self.action_space = spaces.Discrete(len(self.ACTION_LIST))
        self._action_to_string = {i: a for i, a in enumerate(self.ACTION_LIST)}
        self._string_to_action = {a: i for i, a in enumerate(self.ACTION_LIST)}

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ):
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed
        self._env = PuzzleEnvironment(seed=self._seed)
        state = self._env.reset()
        return self._get_obs(), self._build_info(state)

    def step(self, action):
        if self._env is None:
            obs = np.zeros((64, 64, 3), dtype=np.uint8)
            truncated = False
            return obs, 0.0, True, truncated, {}
        action_str = self._action_to_string.get(action, "reset")
        result = self._env.step(action_str)
        obs = self._get_obs()
        info = self._build_info(result.state, result.info)
        truncated = False
        return obs, result.reward, result.done, truncated, info

    def render(self):
        if self._env is None:
            return np.zeros((64, 64, 3), dtype=np.uint8)
        frame = self._env.render()
        return self._resize_nearest(frame, 64, 64)

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None

    def _get_obs(self):
        if self._env is None:
            return np.zeros((64, 64, 3), dtype=np.uint8)
        frame = self._env.render()
        return self._resize_nearest(frame, 64, 64)

    def _build_info(self, state: GameState = None, step_info=None):
        if state is None:
            return {}
        info: Dict[str, Any] = {
            "text_observation": state.text_observation,
            "valid_actions": state.valid_actions,
            "turn": state.turn,
            "game_metadata": state.metadata,
        }
        if step_info:
            info["step_info"] = step_info
        return info

    @staticmethod
    def _resize_nearest(frame, target_h, target_w):
        h, w = frame.shape[:2]
        if h == target_h and w == target_w:
            return frame.astype(np.uint8)
        row_idx = (np.arange(target_h) * h // target_h).astype(int)
        col_idx = (np.arange(target_w) * w // target_w).astype(int)
        return frame[row_idx][:, col_idx].astype(np.uint8)
