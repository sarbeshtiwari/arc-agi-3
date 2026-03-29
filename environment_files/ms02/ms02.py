import random
import struct
import zlib
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from arcengine import ActionInput, ARCBaseGame, GameAction, GameState as EngineGameState
from arcengine.camera import Camera, RenderableUserDisplay
from arcengine.level import Level
from arcengine.sprites import Sprite

C_BG = 5
C_BLUE = 9
C_RED = 8
C_GREEN = 14
C_WIRE = 11
C_WALL = 3
C_MAGENTA = 6
C_GATE_CLOSED = 12
C_GATE_OPEN = 10
C_HAZARD = 13

SIGNAL_COLORS = {C_BLUE, C_RED, C_MAGENTA}

CONDUCTIVE = {C_WIRE, C_BLUE, C_RED, C_MAGENTA, C_GATE_OPEN, C_GREEN}

SOURCES = {C_BLUE, C_RED}

GRID_LINE_COLOR = 4

SHAPE_STRAIGHT_H = [
    [0, 0, 0],
    [1, 1, 1],
    [0, 0, 0],
]

SHAPE_STRAIGHT_V = [
    [0, 1, 0],
    [0, 1, 0],
    [0, 1, 0],
]

SHAPE_CORNER_RB = [
    [0, 0, 0],
    [0, 1, 1],
    [0, 1, 0],
]

SHAPE_CORNER_LB = [
    [0, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
]

SHAPE_CORNER_LT = [
    [0, 1, 0],
    [1, 1, 0],
    [0, 0, 0],
]

SHAPE_CORNER_RT = [
    [0, 1, 0],
    [0, 1, 1],
    [0, 0, 0],
]

SHAPE_T_DOWN = [
    [0, 0, 0],
    [1, 1, 1],
    [0, 1, 0],
]

SHAPE_T_UP = [
    [0, 1, 0],
    [1, 1, 1],
    [0, 0, 0],
]

SHAPE_T_RIGHT = [
    [0, 1, 0],
    [0, 1, 1],
    [0, 1, 0],
]

SHAPE_T_LEFT = [
    [0, 1, 0],
    [1, 1, 0],
    [0, 1, 0],
]

SHAPE_CROSS = [
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0],
]


def rotate_shape_cw(shape: List[List[int]]) -> List[List[int]]:
    return [[shape[2 - c][r] for c in range(3)] for r in range(3)]


def rotate_block_cw(grid: np.ndarray, cx: int, cy: int) -> None:
    block = np.copy(grid[cy - 1 : cy + 2, cx - 1 : cx + 2])
    rotated = np.zeros_like(block)

    for r in range(3):
        for c in range(3):
            src_val = int(block[2 - c, r])
            dst_val = int(block[r, c])
            if r == 1 and c == 1:
                rotated[r, c] = block[r, c]
            elif src_val == C_WIRE:
                rotated[r, c] = C_WIRE
            elif src_val == C_BG and dst_val == C_WIRE:
                rotated[r, c] = C_BG
            else:
                rotated[r, c] = block[r, c]

    wire_mask = block == C_WIRE
    bg_mask = block == C_BG
    rotatable = wire_mask | bg_mask

    new_block = np.copy(block)
    for r in range(3):
        for c in range(3):
            if rotatable[r, c]:
                sr, sc = 2 - c, r
                if rotatable[sr, sc]:
                    new_block[r, c] = block[sr, sc]

    grid[cy - 1 : cy + 2, cx - 1 : cx + 2] = new_block


def propagate_signals(grid: np.ndarray, base_grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape

    for y in range(h):
        for x in range(w):
            base_val = int(base_grid[y, x])
            cur_val = int(grid[y, x])
            if cur_val == C_GATE_OPEN:
                base_grid[y, x] = C_GATE_OPEN
                continue
            if base_val == C_WIRE or base_val == C_GREEN:
                grid[y, x] = base_val
            elif base_val == C_GATE_OPEN:
                grid[y, x] = C_GATE_OPEN

    blue_reached = set()
    red_reached = set()

    def bfs_from_sources(source_color: int) -> Set[Tuple[int, int]]:
        visited = set()
        queue = deque()

        for sy in range(h):
            for sx in range(w):
                if (
                    int(grid[sy, sx]) == source_color
                    or int(base_grid[sy, sx]) == source_color
                ):
                    visited.add((sx, sy))
                    queue.append((sx, sy))

        while queue:
            px, py = queue.popleft()
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = px + dx, py + dy
                if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in visited:
                    cell = int(grid[ny, nx])
                    base_cell = int(base_grid[ny, nx])
                    if cell in {C_WIRE, C_GREEN, C_GATE_OPEN} or base_cell in {
                        C_WIRE,
                        C_GREEN,
                        C_GATE_OPEN,
                    }:
                        visited.add((nx, ny))
                        queue.append((nx, ny))
        return visited

    blue_reached = bfs_from_sources(C_BLUE)
    red_reached = bfs_from_sources(C_RED)

    for px, py in blue_reached:
        cell = int(base_grid[py, px])
        if cell == C_WIRE or cell == C_GATE_OPEN:
            grid[py, px] = C_BLUE
        elif cell == C_GREEN:
            grid[py, px] = C_GREEN

    for px, py in red_reached:
        cell = int(base_grid[py, px])
        if cell == C_WIRE or cell == C_GATE_OPEN:
            if int(grid[py, px]) == C_BLUE:
                grid[py, px] = C_MAGENTA
            else:
                grid[py, px] = C_RED
        elif cell == C_GREEN:
            grid[py, px] = C_GREEN

    gates_opened = True
    while gates_opened:
        gates_opened = False
        for y in range(h):
            for x in range(w):
                if (
                    int(grid[y, x]) == C_GATE_CLOSED
                    or int(base_grid[y, x]) == C_GATE_CLOSED
                ):
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < w and 0 <= ny < h:
                            if int(grid[ny, nx]) == C_MAGENTA:
                                grid[y, x] = C_GATE_OPEN
                                base_grid[y, x] = C_GATE_OPEN
                                gates_opened = True
                                break

        if gates_opened:
            for y in range(h):
                for x in range(w):
                    base_val = int(base_grid[y, x])
                    if base_val == C_WIRE or base_val == C_GREEN:
                        grid[y, x] = base_val

            blue_reached = bfs_from_sources(C_BLUE)
            red_reached = bfs_from_sources(C_RED)

            for px, py in blue_reached:
                cell = int(base_grid[py, px])
                if cell == C_WIRE or cell == C_GATE_OPEN:
                    grid[py, px] = C_BLUE
                elif cell == C_GREEN:
                    grid[py, px] = C_GREEN

            for px, py in red_reached:
                cell = int(base_grid[py, px])
                if cell == C_WIRE or cell == C_GATE_OPEN:
                    if int(grid[py, px]) == C_BLUE:
                        grid[py, px] = C_MAGENTA
                    else:
                        grid[py, px] = C_RED
                elif cell == C_GREEN:
                    grid[py, px] = C_GREEN

    return grid


def check_all_targets_powered(grid: np.ndarray, base_grid: np.ndarray) -> bool:
    h, w = grid.shape

    blue_reached = set()
    red_reached = set()

    def bfs_check(source_color: int) -> Set[Tuple[int, int]]:
        visited = set()
        queue = deque()
        for sy in range(h):
            for sx in range(w):
                if (
                    int(grid[sy, sx]) == source_color
                    or int(base_grid[sy, sx]) == source_color
                ):
                    visited.add((sx, sy))
                    queue.append((sx, sy))
        while queue:
            px, py = queue.popleft()
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = px + dx, py + dy
                if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in visited:
                    cell = int(grid[ny, nx])
                    base_cell = int(base_grid[ny, nx])
                    if cell in {
                        C_WIRE,
                        C_GREEN,
                        C_GATE_OPEN,
                        C_BLUE,
                        C_RED,
                        C_MAGENTA,
                    } or base_cell in {C_WIRE, C_GREEN, C_GATE_OPEN}:
                        visited.add((nx, ny))
                        queue.append((nx, ny))
        return visited

    blue_reached = bfs_check(C_BLUE)
    red_reached = bfs_check(C_RED)
    all_reached = blue_reached | red_reached

    for y in range(h):
        for x in range(w):
            if int(base_grid[y, x]) == C_GREEN:
                if (x, y) not in all_reached:
                    return False
    return True


def check_hazard_contact(grid: np.ndarray) -> bool:
    h, w = grid.shape
    for y in range(h):
        for x in range(w):
            if int(grid[y, x]) == C_HAZARD:
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        if int(grid[ny, nx]) in SIGNAL_COLORS:
                            return True
    return False


def _place_wire_path(
    grid: np.ndarray,
    path: List[Tuple[int, int]],
    color: int = C_WIRE,
) -> None:
    h, w = grid.shape
    for x, y in path:
        if 0 <= x < w and 0 <= y < h:
            if int(grid[y, x]) == C_BG:
                grid[y, x] = color


def _place_3x3_segment(
    grid: np.ndarray,
    cx: int,
    cy: int,
    shape: List[List[int]],
    color: int = C_WIRE,
) -> None:
    h, w = grid.shape
    for dr in range(3):
        for dc in range(3):
            if shape[dr][dc] == 1:
                px, py = cx - 1 + dc, cy - 1 + dr
                if 0 <= px < w and 0 <= py < h:
                    if int(grid[py, px]) == C_BG:
                        grid[py, px] = color


def _find_rotatable_centers(grid: np.ndarray) -> List[Tuple[int, int]]:
    h, w = grid.shape
    centers = []
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if int(grid[y, x]) != C_WIRE:
                continue
            valid = True
            has_bg = False
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    val = int(grid[y + dr, x + dc])
                    if val not in {C_WIRE, C_BG}:
                        valid = False
                        break
                    if val == C_BG:
                        has_bg = True
                if not valid:
                    break
            if valid and has_bg:
                centers.append((x, y))
    return centers


def _is_rotatable_center(grid: np.ndarray, cx: int, cy: int) -> bool:
    h, w = grid.shape
    if cx < 1 or cx >= w - 1 or cy < 1 or cy >= h - 1:
        return False
    if int(grid[cy, cx]) != C_WIRE:
        val = int(grid[cy, cx])
        if val not in {C_WIRE, C_BLUE, C_RED, C_MAGENTA}:
            return False

    has_bg = False
    has_wire = False
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            val = int(grid[cy + dr, cx + dc])
            if dr == 0 and dc == 0:
                continue
            if val not in {C_WIRE, C_BG, C_BLUE, C_RED, C_MAGENTA}:
                return False
            if val == C_BG:
                has_bg = True
            if val in {C_WIRE, C_BLUE, C_RED, C_MAGENTA}:
                has_wire = True
    return has_bg and has_wire


def generate_level_1(rng: random.Random) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    w, h = 14, 14
    grid = np.full((h, w), C_BG, dtype=np.int8)

    grid[4, 1] = C_BLUE

    for x in range(2, 4):
        grid[4, x] = C_WIRE

    grid[3, 5] = C_WIRE
    grid[4, 5] = C_WIRE
    grid[5, 5] = C_WIRE

    grid[4, 7] = C_WIRE

    grid[3, 9] = C_WIRE
    grid[4, 9] = C_WIRE
    grid[4, 10] = C_WIRE

    grid[6, 9] = C_WIRE
    grid[7, 9] = C_WIRE

    grid[9, 8] = C_WIRE
    grid[9, 9] = C_WIRE
    grid[9, 10] = C_WIRE

    grid[11, 9] = C_WIRE

    grid[12, 9] = C_GREEN

    _rotations = [(5, 4), (9, 4), (9, 4), (9, 9)]

    return grid, _rotations


def generate_level_2(rng: random.Random) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    w, h = 18, 18
    grid = np.full((h, w), C_BG, dtype=np.int8)

    grid[3, 1] = C_BLUE
    for x in range(2, 4):
        grid[3, x] = C_WIRE

    grid[2, 5] = C_WIRE
    grid[3, 5] = C_WIRE
    grid[4, 5] = C_WIRE

    grid[3, 7] = C_WIRE

    grid[2, 9] = C_WIRE
    grid[3, 9] = C_WIRE
    grid[3, 10] = C_WIRE

    grid[5, 9] = C_WIRE
    grid[6, 9] = C_WIRE

    grid[8, 8] = C_WIRE
    grid[8, 9] = C_WIRE
    grid[8, 10] = C_WIRE

    grid[10, 9] = C_WIRE
    grid[11, 9] = C_WIRE

    grid[12, 9] = C_WIRE
    grid[12, 8] = C_WIRE
    grid[12, 10] = C_WIRE

    grid[12, 7] = C_WIRE

    grid[11, 5] = C_WIRE
    grid[12, 5] = C_WIRE
    grid[13, 5] = C_WIRE

    grid[12, 3] = C_WIRE
    grid[12, 2] = C_GREEN

    grid[12, 11] = C_WIRE

    grid[11, 13] = C_WIRE
    grid[12, 13] = C_WIRE
    grid[13, 13] = C_WIRE

    grid[12, 15] = C_WIRE
    grid[12, 16] = C_GREEN

    _rotations = [
        (5, 3),
        (9, 3),
        (9, 3),
        (9, 8),
        (5, 12),
        (13, 12),
    ]

    return grid, _rotations


def generate_level_3(rng: random.Random) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    w, h = 22, 22
    grid = np.full((h, w), C_BG, dtype=np.int8)

    grid[5, 1] = C_BLUE
    for x in range(2, 4):
        grid[5, x] = C_WIRE

    grid[4, 5] = C_WIRE
    grid[5, 5] = C_WIRE
    grid[6, 5] = C_WIRE

    grid[5, 7] = C_WIRE

    grid[4, 9] = C_WIRE
    grid[5, 9] = C_WIRE
    grid[5, 10] = C_WIRE

    for y in range(7, 14):
        grid[y, 9] = C_WIRE

    grid[5, 20] = C_RED
    for x in range(18, 20):
        grid[5, x] = C_WIRE

    grid[4, 16] = C_WIRE
    grid[5, 16] = C_WIRE
    grid[6, 16] = C_WIRE

    grid[5, 14] = C_WIRE

    grid[4, 12] = C_WIRE
    grid[5, 11] = C_WIRE
    grid[5, 12] = C_WIRE

    for y in range(7, 14):
        grid[y, 12] = C_WIRE

    grid[14, 9] = C_WIRE
    for x in range(10, 12):
        grid[14, x] = C_WIRE
    grid[14, 12] = C_WIRE

    grid[14, 11] = C_WIRE
    grid[15, 11] = C_WIRE

    grid[16, 11] = C_GATE_CLOSED

    grid[17, 11] = C_WIRE

    grid[19, 10] = C_WIRE
    grid[19, 11] = C_WIRE
    grid[19, 12] = C_WIRE

    grid[21, 11] = C_GREEN

    _rotations = [
        (5, 5),
        (9, 5),
        (9, 5),
        (16, 5),
        (12, 5),
        (12, 5),
        (11, 19),
    ]

    return grid, _rotations


def generate_level_4(rng: random.Random) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    w, h = 32, 32
    grid = np.full((h, w), C_BG, dtype=np.int8)

    grid[4, 1] = C_BLUE
    for x in range(2, 4):
        grid[4, x] = C_WIRE

    grid[3, 5] = C_WIRE
    grid[4, 5] = C_WIRE
    grid[5, 5] = C_WIRE

    grid[4, 7] = C_WIRE

    grid[3, 9] = C_WIRE
    grid[4, 9] = C_WIRE
    grid[4, 10] = C_WIRE

    grid[6, 9] = C_WIRE
    grid[7, 9] = C_WIRE

    grid[8, 9] = C_WIRE
    grid[9, 9] = C_WIRE
    grid[10, 9] = C_WIRE

    grid[11, 9] = C_WIRE
    grid[12, 9] = C_HAZARD

    grid[7, 10] = C_WIRE
    grid[7, 11] = C_WIRE
    grid[8, 11] = C_WIRE
    grid[9, 11] = C_WIRE
    grid[10, 11] = C_WIRE
    grid[11, 11] = C_WIRE
    grid[12, 11] = C_WIRE
    grid[13, 11] = C_WIRE

    grid[9, 7] = C_WIRE
    grid[10, 7] = C_WIRE
    grid[11, 7] = C_WIRE
    grid[12, 7] = C_WIRE
    grid[13, 7] = C_WIRE

    grid[13, 8] = C_WIRE

    grid[4, 30] = C_RED
    for x in range(28, 30):
        grid[4, x] = C_WIRE

    grid[3, 26] = C_WIRE
    grid[4, 26] = C_WIRE
    grid[5, 26] = C_WIRE

    grid[4, 23] = C_WIRE
    grid[4, 24] = C_WIRE

    grid[3, 21] = C_WIRE
    grid[4, 20] = C_WIRE
    grid[4, 21] = C_WIRE

    for y in range(6, 13):
        grid[y, 21] = C_WIRE

    grid[13, 7] = C_WIRE
    grid[13, 8] = C_WIRE
    for x in range(11, 22):
        grid[13, x] = C_WIRE

    grid[14, 14] = C_WIRE
    grid[15, 14] = C_WIRE

    grid[16, 14] = C_GATE_CLOSED

    grid[17, 14] = C_WIRE

    grid[19, 13] = C_WIRE
    grid[19, 14] = C_WIRE
    grid[19, 15] = C_WIRE

    grid[21, 14] = C_WIRE
    grid[22, 14] = C_WIRE

    grid[23, 14] = C_GATE_CLOSED

    grid[24, 14] = C_WIRE
    grid[25, 14] = C_WIRE

    grid[26, 14] = C_WIRE
    grid[26, 13] = C_WIRE
    grid[26, 15] = C_WIRE

    grid[26, 12] = C_WIRE

    grid[25, 10] = C_WIRE
    grid[26, 10] = C_WIRE
    grid[27, 10] = C_WIRE

    grid[26, 8] = C_WIRE
    grid[26, 7] = C_GREEN

    grid[26, 16] = C_WIRE

    grid[25, 18] = C_WIRE
    grid[26, 18] = C_WIRE
    grid[27, 18] = C_WIRE

    grid[26, 20] = C_WIRE
    grid[26, 21] = C_GREEN

    _rotations = [
        (9, 9),
        (5, 4),
        (9, 4),
        (9, 4),
        (26, 4),
        (21, 4),
        (21, 4),
        (14, 19),
        (10, 26),
        (18, 26),
    ]

    return grid, _rotations


def generate_level_5(rng: random.Random) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    w, h = 48, 48
    grid = np.full((h, w), C_BG, dtype=np.int8)

    grid[6, 2] = C_BLUE
    for x in range(3, 6):
        grid[6, x] = C_WIRE

    grid[5, 7] = C_WIRE
    grid[6, 7] = C_WIRE
    grid[7, 7] = C_WIRE

    for x in range(9, 12):
        grid[6, x] = C_WIRE

    grid[5, 13] = C_WIRE
    grid[6, 13] = C_WIRE
    grid[6, 14] = C_WIRE

    for y in range(8, 13):
        grid[y, 13] = C_WIRE

    grid[14, 12] = C_WIRE
    grid[14, 13] = C_WIRE
    grid[14, 14] = C_WIRE

    for y in range(16, 19):
        grid[y, 13] = C_WIRE

    grid[6, 15] = C_HAZARD
    grid[6, 16] = C_HAZARD

    grid[19, 13] = C_WIRE
    grid[20, 13] = C_WIRE
    grid[21, 13] = C_WIRE

    grid[22, 13] = C_WIRE
    grid[23, 13] = C_HAZARD

    grid[18, 14] = C_WIRE
    grid[18, 15] = C_WIRE
    grid[19, 15] = C_WIRE
    grid[20, 15] = C_WIRE
    grid[21, 15] = C_WIRE
    grid[22, 15] = C_WIRE
    grid[23, 15] = C_WIRE

    grid[6, 45] = C_RED
    for x in range(42, 45):
        grid[6, x] = C_WIRE

    grid[5, 40] = C_WIRE
    grid[6, 40] = C_WIRE
    grid[7, 40] = C_WIRE

    for x in range(36, 39):
        grid[6, x] = C_WIRE

    grid[5, 34] = C_WIRE
    grid[6, 33] = C_WIRE
    grid[6, 34] = C_WIRE

    for y in range(8, 13):
        grid[y, 34] = C_WIRE

    grid[14, 33] = C_WIRE
    grid[14, 34] = C_WIRE
    grid[14, 35] = C_WIRE

    for y in range(16, 19):
        grid[y, 34] = C_WIRE

    grid[6, 32] = C_HAZARD
    grid[6, 31] = C_HAZARD

    grid[19, 34] = C_WIRE
    grid[20, 34] = C_WIRE
    grid[21, 34] = C_WIRE

    grid[22, 34] = C_WIRE
    grid[23, 34] = C_HAZARD

    grid[18, 33] = C_WIRE
    grid[18, 32] = C_WIRE
    grid[19, 32] = C_WIRE
    grid[20, 32] = C_WIRE
    grid[21, 32] = C_WIRE
    grid[22, 32] = C_WIRE
    grid[23, 32] = C_WIRE

    grid[23, 15] = C_WIRE
    for x in range(16, 32):
        grid[23, x] = C_WIRE
    grid[23, 32] = C_WIRE

    for y in range(24, 27):
        grid[y, 24] = C_WIRE

    grid[27, 24] = C_GATE_CLOSED

    for y in range(28, 31):
        grid[y, 24] = C_WIRE

    grid[31, 24] = C_GATE_CLOSED

    for y in range(32, 37):
        grid[y, 24] = C_WIRE

    grid[37, 24] = C_WIRE
    grid[37, 23] = C_WIRE
    grid[37, 25] = C_WIRE

    for x in range(18, 23):
        grid[37, x] = C_WIRE

    grid[36, 16] = C_WIRE
    grid[37, 16] = C_WIRE
    grid[38, 16] = C_WIRE

    for x in range(10, 15):
        grid[37, x] = C_WIRE
    grid[37, 9] = C_GREEN

    for x in range(26, 31):
        grid[37, x] = C_WIRE

    grid[36, 32] = C_WIRE
    grid[37, 32] = C_WIRE
    grid[38, 32] = C_WIRE

    for x in range(34, 39):
        grid[37, x] = C_WIRE
    grid[37, 39] = C_GREEN

    for y in range(38, 43):
        grid[y, 24] = C_WIRE
    grid[43, 24] = C_GREEN

    grid[40, 20] = C_HAZARD
    grid[40, 21] = C_HAZARD
    grid[40, 22] = C_HAZARD

    grid[21, 16] = C_WALL
    grid[21, 31] = C_WALL

    _rotations = [
        (13, 20),
        (34, 20),
        (7, 6),
        (13, 6),
        (13, 6),
        (13, 14),
        (40, 6),
        (34, 6),
        (34, 6),
        (34, 14),
        (16, 37),
        (32, 37),
    ]

    return grid, _rotations


LEVEL_GENERATORS = [
    generate_level_1,
    generate_level_2,
    generate_level_3,
    generate_level_4,
    generate_level_5,
]

LEVEL_GRID_SIZES = [
    (14, 14),
    (18, 18),
    (22, 22),
    (32, 32),
    (48, 48),
]


def grid_to_display(gx: int, gy: int, gw: int, gh: int) -> Tuple[int, int]:
    scale = min(64 // gw, 64 // gh)
    x_off = (64 - gw * scale) // 2
    y_off = (64 - gh * scale) // 2
    return x_off + gx * scale + scale // 2, y_off + gy * scale + scale // 2


LEVEL_MAX_MOVES = [120, 180, 210, 300, 360]

TOTAL_LIVES = 3


class Ms02(ARCBaseGame):
    _grids: List[np.ndarray]
    _base_grids: List[np.ndarray]
    _grid_sizes: List[Tuple[int, int]]
    _rng: random.Random
    _moves_used: List[int]
    _lives: int
    _last_level_index: int
    _cursor_x: int
    _cursor_y: int
    _undo_stack: List[Tuple]
    _consecutive_resets: int

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self._rng = random.Random(seed)

        self._grids = []
        self._base_grids = []
        self._grid_sizes = []
        self._moves_used = [0] * len(LEVEL_GRID_SIZES)
        self._lives = TOTAL_LIVES
        self._last_level_index = 0
        self._cursor_x = 0
        self._cursor_y = 0
        self._undo_stack: List[Tuple] = []
        self._consecutive_resets = 0

        levels = []
        for i, gen_fn in enumerate(LEVEL_GENERATORS):
            gw, gh = LEVEL_GRID_SIZES[i]
            grid, _ = gen_fn(self._rng)

            base_grid = np.copy(grid)
            display_grid = np.copy(grid)

            propagate_signals(display_grid, base_grid)

            self._grids.append(display_grid)
            self._base_grids.append(base_grid)
            self._grid_sizes.append((gw, gh))

            sprite = Sprite(
                pixels=base_grid.tolist(),
                name=f"grid_{i}",
                x=0,
                y=0,
                layer=0,
            )

            level = Level(
                sprites=[sprite],
                grid_size=(gw, gh),
                name=f"Level {i + 1}",
                data={"grid_index": i},
            )
            levels.append(level)

        camera = Camera(
            x=0,
            y=0,
            width=levels[0].grid_size[0],
            height=levels[0].grid_size[1],
            background=C_BG,
            letter_box=C_BG,
        )

        super().__init__(
            "ms02",
            levels,
            camera,
            available_actions=[0, 1, 2, 3, 4, 5, 6, 7],
        )

        overlay = _CircuitOverlay(self)
        self.camera.replace_interface([overlay])

    @property
    def level_index(self) -> int:
        return self._current_level_index

    def on_set_level(self, level: Level) -> None:
        idx = level.get_data("grid_index")
        if idx is not None and idx < len(self._grid_sizes):
            gw, gh = self._grid_sizes[idx]
            self.camera.resize(gw, gh)
            self.camera.x = 0
            self.camera.y = 0
            self._cursor_x = self._rng.randint(0, gw - 1)
            self._cursor_y = self._rng.randint(0, gh - 1)
            if idx != self._last_level_index:
                self._lives = TOTAL_LIVES
                self._last_level_index = idx
            self._undo_stack = []

    def full_reset(self) -> None:
        super().full_reset()
        self._moves_used = [0] * len(LEVEL_GRID_SIZES)
        self._lives = TOTAL_LIVES
        self._last_level_index = 0
        self._consecutive_resets = 0
        self._undo_stack = []
        for i in range(len(self._grids)):
            clean_sprites = self._clean_levels[i].get_sprites_by_name(f"grid_{i}")
            if clean_sprites:
                self._base_grids[i] = np.array(clean_sprites[0].pixels, dtype=np.int8)
                self._grids[i] = np.copy(self._base_grids[i])
                propagate_signals(self._grids[i], self._base_grids[i])
        self._sync_current_level_sprite()

    def level_reset(self, preserve_moves: bool = False) -> None:
        idx = self._current_level_index
        super().level_reset()
        if not preserve_moves:
            self._moves_used[idx] = 0
        clean_sprites = self._clean_levels[idx].get_sprites_by_name(f"grid_{idx}")
        if clean_sprites:
            self._base_grids[idx] = np.array(clean_sprites[0].pixels, dtype=np.int8)
            self._grids[idx] = np.copy(self._base_grids[idx])
            propagate_signals(self._grids[idx], self._base_grids[idx])
        self._sync_current_level_sprite()

    def _sync_current_level_sprite(self) -> None:
        idx = self.current_level.get_data("grid_index")
        if idx is not None and idx < len(self._grids):
            sprites = self.current_level.get_sprites_by_name(f"grid_{idx}")
            if sprites:
                sprites[0].pixels = np.array(self._grids[idx], dtype=np.int8)

    def _get_current_grid(self) -> np.ndarray:
        idx = self.current_level.get_data("grid_index")
        if idx is not None:
            return self._grids[idx]
        raise ValueError("No grid index found")

    def _get_current_base_grid(self) -> np.ndarray:
        idx = self.current_level.get_data("grid_index")
        if idx is not None:
            return self._base_grids[idx]
        raise ValueError("No grid index found")

    def step(self) -> None:
        action = self.action

        if action.id == GameAction.RESET:
            if self._state == EngineGameState.GAME_OVER:
                self.full_reset()
            else:
                self._consecutive_resets += 1
                if self._consecutive_resets >= 2:
                    self.full_reset()
                else:
                    self.level_reset(preserve_moves=False)
            self.complete_action()
            return

        if self._state == EngineGameState.GAME_OVER:
            self.complete_action()
            return

        self._consecutive_resets = 0

        if action.id == GameAction.ACTION7:
            level_idx = self._current_level_index
            self._moves_used[level_idx] += 1
            if self._moves_used[level_idx] >= LEVEL_MAX_MOVES[level_idx]:
                self._lives -= 1
                if self._lives <= 0:
                    self.lose()
                else:
                    self.level_reset(preserve_moves=False)
                self.complete_action()
                return
            self._undo()
            self.complete_action()
            return

        level_idx = self._current_level_index
        self._save_undo_state(level_idx)
        self._moves_used[level_idx] += 1

        if self._moves_used[level_idx] >= LEVEL_MAX_MOVES[level_idx]:
            self._lives -= 1
            if self._lives <= 0:
                self.lose()
            else:
                self.level_reset(preserve_moves=False)
            self.complete_action()
            return

        if action.id == GameAction.ACTION1:
            self._move_cursor(0, -1)
        elif action.id == GameAction.ACTION2:
            self._move_cursor(0, 1)
        elif action.id == GameAction.ACTION3:
            self._move_cursor(-1, 0)
        elif action.id == GameAction.ACTION4:
            self._move_cursor(1, 0)
        elif action.id == GameAction.ACTION5:
            self._handle_click(
                ActionInput(id=GameAction.ACTION6, data={"x": -1, "y": -1})
            )
        elif action.id == GameAction.ACTION6:
            self._handle_click(action)

        self.complete_action()

    def _move_cursor(self, dx: int, dy: int) -> None:
        idx = self._current_level_index
        gw, gh = self._grid_sizes[idx]
        self._cursor_x = max(0, min(gw - 1, self._cursor_x + dx))
        self._cursor_y = max(0, min(gh - 1, self._cursor_y + dy))

    def _save_undo_state(self, level_idx: int) -> None:
        self._undo_stack = [
            (
                np.copy(self._grids[level_idx]),
                np.copy(self._base_grids[level_idx]),
                self._cursor_x,
                self._cursor_y,
            )
        ]

    def _undo(self) -> None:
        if not self._undo_stack:
            return
        grid_copy, base_copy, cx, cy = self._undo_stack.pop()
        level_idx = self._current_level_index
        self._grids[level_idx] = grid_copy
        self._base_grids[level_idx] = base_copy
        self._cursor_x = cx
        self._cursor_y = cy
        self._sync_current_level_sprite()
        self._undo_stack = []

    def _handle_click(self, action: ActionInput) -> None:
        data = action.data
        display_x = data.get("x", -1)
        display_y = data.get("y", -1)

        if display_x == -1 and display_y == -1:
            grid_x, grid_y = self._cursor_x, self._cursor_y
        else:
            result = self.camera.display_to_grid(int(display_x), int(display_y))
            if result is None:
                return
            grid_x, grid_y = result
            self._cursor_x, self._cursor_y = grid_x, grid_y
        grid = self._get_current_grid()
        base_grid = self._get_current_base_grid()
        h, w = grid.shape
        level_idx = self._current_level_index

        if grid_x < 0 or grid_x >= w or grid_y < 0 or grid_y >= h:
            return

        if not _is_rotatable_center(base_grid, grid_x, grid_y):
            return

        rotate_block_cw(base_grid, grid_x, grid_y)

        np.copyto(grid, base_grid)

        propagate_signals(grid, base_grid)

        self._update_sprite_from_grid()

        if check_hazard_contact(grid):
            rotate_block_cw(base_grid, grid_x, grid_y)
            rotate_block_cw(base_grid, grid_x, grid_y)
            rotate_block_cw(base_grid, grid_x, grid_y)
            np.copyto(grid, base_grid)
            propagate_signals(grid, base_grid)
            self._update_sprite_from_grid()
            return

        if check_all_targets_powered(grid, base_grid):
            self.next_level()

    def _update_sprite_from_grid(self) -> None:
        idx = self.current_level.get_data("grid_index")
        if idx is not None:
            grid = self._grids[idx]
            sprites = self.current_level.get_sprites_by_name(f"grid_{idx}")
            if sprites:
                sprites[0].pixels = np.array(grid, dtype=np.int8)


C_CURSOR = 1


def _draw_cursor_box(
    frame: np.ndarray, cursor_x: int, cursor_y: int, gw: int, gh: int
) -> None:
    scale = min(64 // gw, 64 // gh)
    x_off = (64 - gw * scale) // 2
    y_off = (64 - gh * scale) // 2

    px_left = x_off + cursor_x * scale
    px_right = px_left + scale - 1
    px_top = y_off + cursor_y * scale
    px_bottom = px_top + scale - 1

    for px in range(px_left, px_right + 1):
        if 0 <= px < 64:
            if 0 <= px_top < 64:
                frame[px_top, px] = C_CURSOR
            if 0 <= px_bottom < 64:
                frame[px_bottom, px] = C_CURSOR

    for py in range(px_top, px_bottom + 1):
        if 0 <= py < 64:
            if 0 <= px_left < 64:
                frame[py, px_left] = C_CURSOR
            if 0 <= px_right < 64:
                frame[py, px_right] = C_CURSOR


def _draw_lives(frame: np.ndarray, lives: int) -> None:
    block_size = 3
    gap = 1
    padding = 1
    total_blocks = TOTAL_LIVES
    bar_width = padding + total_blocks * block_size + (total_blocks - 1) * gap + padding
    bar_height = block_size + 2 * padding

    bar_x = 64 - bar_width - 1
    bar_y = 60

    for y in range(bar_y, bar_y + bar_height):
        for x in range(bar_x, bar_x + bar_width):
            if 0 <= x < 64 and 0 <= y < 64:
                frame[y, x] = C_GATE_OPEN

    for i in range(lives):
        bx = bar_x + padding + i * (block_size + gap)
        by = bar_y + padding
        for dy in range(block_size):
            for dx in range(block_size):
                px, py = bx + dx, by + dy
                if 0 <= px < 64 and 0 <= py < 64:
                    frame[py, px] = C_RED


def _draw_progress_bar(frame: np.ndarray, moves_used: int, max_moves: int) -> None:
    bar_y = 62
    bar_x_start = 2
    bar_x_end = 49
    bar_width = bar_x_end - bar_x_start

    moves_remaining = max_moves - moves_used
    if moves_remaining < 0:
        moves_remaining = 0

    fill_length = int((moves_remaining / max_moves) * bar_width)

    for y in range(bar_y, bar_y + 2):
        for x in range(bar_x_start, bar_x_start + fill_length):
            if 0 <= x < 64 and 0 <= y < 64:
                frame[y, x] = C_GREEN

    for y in range(bar_y, bar_y + 2):
        for x in range(bar_x_start + fill_length, bar_x_end):
            if 0 <= x < 64 and 0 <= y < 64:
                frame[y, x] = 3


class _CircuitOverlay(RenderableUserDisplay):
    def __init__(self, game: Ms02) -> None:
        self._game = game

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        game = self._game
        idx = game.level_index
        if idx >= len(game._grid_sizes):
            return frame

        gw, gh = game._grid_sizes[idx]
        scale = min(64 // gw, 64 // gh)
        x_off = (64 - gw * scale) // 2
        y_off = (64 - gh * scale) // 2

        moves_used = game._moves_used[idx]
        max_moves = LEVEL_MAX_MOVES[idx]
        _draw_progress_bar(frame, moves_used, max_moves)

        _draw_lives(frame, game._lives)

        _draw_cursor_box(frame, game._cursor_x, game._cursor_y, gw, gh)

        if scale >= 4:
            for i in range(gw + 1):
                px = x_off + i * scale
                if 0 <= px < 64:
                    for py in range(y_off, min(64, y_off + gh * scale)):
                        if int(frame[py, px]) == C_BG:
                            frame[py, px] = GRID_LINE_COLOR

            for i in range(gh + 1):
                py = y_off + i * scale
                if 0 <= py < 64:
                    for px in range(x_off, min(64, x_off + gw * scale)):
                        if int(frame[py, px]) == C_BG:
                            frame[py, px] = GRID_LINE_COLOR
        else:
            py = y_off
            if 0 <= py < 64:
                for px in range(x_off, min(64, x_off + gw * scale)):
                    if int(frame[py, px]) == C_BG:
                        frame[py, px] = GRID_LINE_COLOR
            py = y_off + gh * scale
            if 0 <= py < 64:
                for px in range(x_off, min(64, x_off + gw * scale)):
                    if int(frame[py, px]) == C_BG:
                        frame[py, px] = GRID_LINE_COLOR
            px = x_off
            if 0 <= px < 64:
                for py_i in range(y_off, min(64, y_off + gh * scale)):
                    if int(frame[py_i, px]) == C_BG:
                        frame[py_i, px] = GRID_LINE_COLOR
            px = x_off + gw * scale
            if 0 <= px < 64:
                for py_i in range(y_off, min(64, y_off + gh * scale)):
                    if int(frame[py_i, px]) == C_BG:
                        frame[py_i, px] = GRID_LINE_COLOR

        return frame


_ARC_PALETTE = [
    (0, 0, 0),
    (0, 116, 217),
    (255, 65, 54),
    (46, 204, 64),
    (255, 220, 0),
    (127, 127, 127),
    (240, 18, 190),
    (255, 133, 27),
    (0, 191, 255),
    (135, 12, 37),
    (128, 0, 128),
    (128, 128, 0),
    (220, 50, 50),
    (0, 128, 0),
    (200, 200, 50),
    (200, 200, 200),
]


def _frame_to_png(frame: Any) -> bytes:
    arr = np.array(frame[0]) if isinstance(frame, list) else np.array(frame)
    h, w = arr.shape
    raw = bytearray()
    for y in range(h):
        raw.append(0)
        for x in range(w):
            ci = max(0, min(int(arr[y, x]), len(_ARC_PALETTE) - 1))
            r, g, b = _ARC_PALETTE[ci]
            raw.extend([r, g, b])
    compressed = zlib.compress(bytes(raw))

    def _chunk(ct: bytes, d: bytes) -> bytes:
        c = ct + d
        return (
            struct.pack(">I", len(d))
            + c
            + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", compressed)
        + _chunk(b"IEND", b"")
    )


_CURSOR_ACTION_MAP: Dict[str, GameAction] = {
    "up": GameAction.ACTION1,
    "down": GameAction.ACTION2,
    "left": GameAction.ACTION3,
    "right": GameAction.ACTION4,
    "select": GameAction.ACTION5,
    "undo": GameAction.ACTION7,
    "reset": GameAction.RESET,
}


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
    def __init__(self, seed: int = 0) -> None:
        self._engine = Ms02(seed=seed)
        self._total_turns: int = 0
        self._last_action_was_reset: bool = False

    def reset(self) -> GameState:
        self._total_turns = 0
        self._engine.perform_action(ActionInput(id=GameAction.RESET, data={}))
        self._last_action_was_reset = True
        return self.get_state()

    def get_actions(self) -> List[str]:
        if self.is_done():
            return ["reset"]
        return ["reset", "up", "down", "left", "right", "select", "click", "undo"]

    def step(self, action: str) -> StepResult:
        action_str = action.strip().lower()

        if action_str == "reset":
            state = self.reset()
            return StepResult(state=state, reward=0.0, done=False, info={})

        prev_level = self._engine._current_level_index

        action_input = self._parse_action(action_str)
        self._engine.perform_action(action_input)

        self._total_turns += 1
        self._last_action_was_reset = False

        done = self.is_done()

        cur_level = self._engine._current_level_index

        reward = 0.0
        if cur_level > prev_level or self._engine._state == EngineGameState.WIN:
            reward = 1.0 / len(self._engine._grids)

        return StepResult(
            state=self.get_state(),
            reward=reward,
            done=done,
            info={},
        )

    def is_done(self) -> bool:
        return (
            self._engine._state == EngineGameState.GAME_OVER
            or self._engine._state == EngineGameState.WIN
        )

    def _parse_action(self, action: str) -> ActionInput:
        parts = action.split()

        if parts[0] == "click" and len(parts) == 3:
            gx, gy = int(parts[1]), int(parts[2])
            idx = self._engine._current_level_index
            gw, gh = self._engine._grid_sizes[idx]
            dx, dy = grid_to_display(gx, gy, gw, gh)
            return ActionInput(id=GameAction.ACTION6, data={"x": dx, "y": dy})

        game_action = _CURSOR_ACTION_MAP.get(action)
        if game_action is not None:
            return ActionInput(id=game_action)

        if action == "click":
            return ActionInput(
                id=GameAction.ACTION6,
                data={"x": 0, "y": 0},
            )

        return ActionInput(id=GameAction.RESET, data={})

    def get_state(self) -> GameState:
        return GameState(
            text_observation=self._build_text_observation(),
            image_observation=self._render_image(),
            valid_actions=self.get_actions(),
            turn=self._total_turns,
            metadata=self._build_metadata(),
        )

    def _build_text_observation(self) -> str:
        g = self._engine
        idx = g._current_level_index
        if idx >= len(g._grid_sizes):
            idx = len(g._grid_sizes) - 1
        gw, gh = g._grid_sizes[idx]
        grid = g._grids[idx]
        moves_used = g._moves_used[idx]
        max_moves = LEVEL_MAX_MOVES[idx]

        lines = [
            f"Level: {idx + 1}/{len(LEVEL_GRID_SIZES)}",
            f"Moves: {moves_used}/{max_moves}",
            f"Lives: {g._lives}/{TOTAL_LIVES}",
            f"Cursor: ({g._cursor_x}, {g._cursor_y})",
        ]

        if g._state == EngineGameState.GAME_OVER:
            lines.append("Status: GAME OVER")
        elif g._state == EngineGameState.WIN:
            lines.append("Status: WIN")
        else:
            remaining = max(0, max_moves - moves_used)
            lines.append(f"Moves remaining: {remaining}")

        lines.append(f"Grid ({gw}x{gh}):")
        for y in range(gh):
            row = " ".join(f"{int(grid[y, x]):2d}" for x in range(gw))
            lines.append(row)

        return "\n".join(lines)

    def _build_metadata(self) -> dict:
        g = self._engine
        idx = g._current_level_index
        if idx >= len(g._grid_sizes):
            idx = len(g._grid_sizes) - 1
        return {
            "total_levels": len(g._grid_sizes),
            "level": idx + 1,
            "moves_used": g._moves_used[idx],
            "move_limit": LEVEL_MAX_MOVES[idx],
            "lives": g._lives,
            "max_lives": TOTAL_LIVES,
            "grid_width": g._grid_sizes[idx][0],
            "grid_height": g._grid_sizes[idx][1],
            "cursor_x": g._cursor_x,
            "cursor_y": g._cursor_y,
        }

    def _render_image(self) -> bytes:
        game = self._engine
        idx = game._current_level_index
        if idx >= len(game._grid_sizes):
            idx = len(game._grid_sizes) - 1
        grid = game._grids[idx]
        gw, gh = game._grid_sizes[idx]

        frame = np.full((64, 64), C_BG, dtype=np.int8)
        scale = min(64 // gw, 64 // gh)
        x_off = (64 - gw * scale) // 2
        y_off = (64 - gh * scale) // 2

        for gy in range(gh):
            for gx in range(gw):
                color = int(grid[gy, gx])
                for dy in range(scale):
                    for dx in range(scale):
                        py = y_off + gy * scale + dy
                        px = x_off + gx * scale + dx
                        if 0 <= px < 64 and 0 <= py < 64:
                            frame[py, px] = color

        overlay = _CircuitOverlay(game)
        frame = overlay.render_interface(frame)

        return _frame_to_png(frame)

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError(f"Unsupported render mode: {mode}")
        game = self._engine
        idx = game._current_level_index
        if idx >= len(game._grid_sizes):
            idx = len(game._grid_sizes) - 1
        grid = game._grids[idx]
        gw, gh = game._grid_sizes[idx]

        index_frame = np.full((64, 64), C_BG, dtype=np.int8)
        scale = min(64 // gw, 64 // gh)
        x_off = (64 - gw * scale) // 2
        y_off = (64 - gh * scale) // 2

        for gy in range(gh):
            for gx in range(gw):
                color = int(grid[gy, gx])
                for dy in range(scale):
                    for dx in range(scale):
                        py = y_off + gy * scale + dy
                        px = x_off + gx * scale + dx
                        if 0 <= px < 64 and 0 <= py < 64:
                            index_frame[py, px] = color

        overlay = _CircuitOverlay(game)
        index_frame = overlay.render_interface(index_frame)

        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        for ci, color in enumerate(_ARC_PALETTE):
            mask = index_frame == ci
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
    check_env(env, skip_render_check=False)

    obs, info = env.reset()
    assert obs.shape == (64, 64, 3)
    assert obs.dtype == np.uint8
    assert "text_observation" in info

    obs, reward, term, trunc, info = env.step(0)
    assert not trunc

    frame = env.render()
    assert frame is not None and frame.shape == (64, 64, 3)

    env.close()
