import struct
import zlib
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
    GameState as EngineState,
    Level,
    RenderableUserDisplay,
    Sprite,
)
import random as _random_mod


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


GAME_ID = "nv48"
total_levels = 4

BACKGROUND_COLOR = 5
PADDING_COLOR = 4

N = 0
E = 1
S = 2
W = 3

FLOW_RED = 0
FLOW_PURPLE = 1

PIPE_STRAIGHT = 0
PIPE_BEND = 1
PIPE_TEE = 2
PIPE_CROSS = 3

PIPE_CONNECTIONS = {
    PIPE_STRAIGHT: {0: (N, S), 1: (E, W), 2: (N, S), 3: (E, W)},
    PIPE_BEND: {0: (N, E), 1: (E, S), 2: (S, W), 3: (W, N)},
    PIPE_TEE: {0: (N, E, S), 1: (E, S, W), 2: (S, W, N), 3: (W, N, E)},
    PIPE_CROSS: {0: (N, E, S, W), 1: (N, E, S, W), 2: (N, E, S, W), 3: (N, E, S, W)},
}

DIR_DELTA = {N: (0, -1), E: (1, 0), S: (0, 1), W: (-1, 0)}
OPPOSITE = {N: S, S: N, E: W, W: E}

COL_BG = 5
COL_PIPE = 3
COL_PIPE_LIT_RED = 8
COL_PIPE_LIT_PURPLE = 15
COL_PIPE_CENTER = 2
COL_PIPE_CENTER_LIT_RED = 12
COL_PIPE_CENTER_LIT_PURPLE = 13
COL_BLUE_PIPE_BG = 9
COL_SOURCE = 8
COL_SOURCE_GLOW = 12
COL_DRAIN_RED = 8
COL_DRAIN_PURPLE = 15
COL_DRAIN_OK = 14
COL_DRAIN_INNER_RED = 12
COL_DRAIN_INNER_PURPLE = 13
COL_LOCKED_BORDER = 6
COL_CURSOR = 11
COL_EMPTY_DOT = 1
COL_GRID_LINE = 4
COL_BAR_FULL = 14
COL_BAR_EMPTY = 4
COL_LIFE_ON = 8
COL_LIFE_OFF = 4
COL_RUST = 13
COL_LEAK = 8
COL_RUST_WARN = 12
COL_FLOW_PULSE = 10


def _make_pipe(pipe_type, rotation, locked=False):
    return {"type": pipe_type, "rot": rotation % 4, "locked": locked}


def _get_connections(pipe):
    return PIPE_CONNECTIONS[pipe["type"]][pipe["rot"]]


def _has_dir(pipe, d):
    return d in _get_connections(pipe)


def _source_output_dirs(source_pos, cols, rows):
    sx, sy = source_pos
    dirs = []
    if sx == 0:
        dirs.append(E)
    if sx == cols - 1:
        dirs.append(W)
    if sy == 0:
        dirs.append(S)
    if sy == rows - 1:
        dirs.append(N)
    if not dirs:
        dirs = [N, E, S, W]
    return dirs


def _compute_flow(
    source_pos, drains, pipes, cols, rows, blue_pipes=None, drain_colors=None
):
    filled = {}
    drain_reached = {}
    drain_positions = set(drains)
    flow_order = []

    blue_set = set()
    if blue_pipes:
        for bp in blue_pipes:
            blue_set.add((bp[0], bp[1]))

    start_dirs = _source_output_dirs(source_pos, cols, rows)
    queue = []
    for d in start_dirs:
        dx, dy = DIR_DELTA[d]
        nx, ny = source_pos[0] + dx, source_pos[1] + dy
        if (nx, ny) in drain_positions:
            drain_reached[(nx, ny)] = FLOW_RED
            continue
        if (nx, ny) in pipes:
            pipe = pipes[(nx, ny)]
            incoming = OPPOSITE[d]
            if _has_dir(pipe, incoming):
                if (nx, ny) not in filled:
                    color = FLOW_PURPLE if (nx, ny) in blue_set else FLOW_RED
                    filled[(nx, ny)] = color
                    flow_order.append((nx, ny))
                    queue.append((nx, ny))

    while queue:
        cx, cy = queue.pop(0)
        pipe = pipes[(cx, cy)]
        my_color = filled[(cx, cy)]
        conns = _get_connections(pipe)

        for d in conns:
            dx, dy = DIR_DELTA[d]
            nx, ny = cx + dx, cy + dy
            if (nx, ny) == source_pos:
                continue
            if (nx, ny) in drain_positions:
                if (nx, ny) not in drain_reached:
                    drain_reached[(nx, ny)] = my_color
                continue
            if (nx, ny) in pipes and (nx, ny) not in filled:
                neighbor = pipes[(nx, ny)]
                incoming = OPPOSITE[d]
                if _has_dir(neighbor, incoming):
                    color = (1 - my_color) if (nx, ny) in blue_set else my_color
                    filled[(nx, ny)] = color
                    flow_order.append((nx, ny))
                    queue.append((nx, ny))

    return filled, drain_reached, flow_order


def _compute_leaks(flow_set, pipes, source_pos, drains, cols, rows):
    leak_set = set()
    drain_positions = set(drains)
    for pos in flow_set:
        if pos not in pipes:
            continue
        pipe = pipes[pos]
        conns = _get_connections(pipe)
        for d in conns:
            dx, dy = DIR_DELTA[d]
            nx, ny = pos[0] + dx, pos[1] + dy
            if (nx, ny) == source_pos:
                continue
            if (nx, ny) in drain_positions:
                continue
            if (nx, ny) in pipes:
                neighbor = pipes[(nx, ny)]
                incoming = OPPOSITE[d]
                if _has_dir(neighbor, incoming):
                    continue
            leak_set.add(pos)
    return leak_set


def _check_color_win(drain_list, drain_reached, drain_colors):
    if not drain_list:
        return False
    for dpos in drain_list:
        if dpos not in drain_reached:
            return False
        expected = drain_colors.get(dpos, FLOW_RED)
        if drain_reached[dpos] != expected:
            return False
    return True


LEVEL_DATA = [
    {
        "cols": 6,
        "rows": 6,
        "max_moves": 152,
        "source": (0, 0),
        "drains": [(5, 5), (5, 2)],
        "drain_colors": {(5, 5): FLOW_RED, (5, 2): FLOW_PURPLE},
        "rust_interval": 7,
        "leak_cost": 1,
        "blue_pipes": [(2, 2), (2, 1)],
        "pipes": {
            (1, 0): _make_pipe(PIPE_BEND, 2),
            (1, 1): _make_pipe(PIPE_STRAIGHT, 0),
            (1, 2): _make_pipe(PIPE_TEE, 0),
            (2, 2): _make_pipe(PIPE_STRAIGHT, 1),
            (3, 2): _make_pipe(PIPE_STRAIGHT, 1),
            (4, 2): _make_pipe(PIPE_STRAIGHT, 1, locked=True),
            (1, 3): _make_pipe(PIPE_BEND, 0),
            (2, 3): _make_pipe(PIPE_BEND, 2),
            (2, 4): _make_pipe(PIPE_BEND, 0),
            (3, 4): _make_pipe(PIPE_BEND, 2),
            (3, 5): _make_pipe(PIPE_BEND, 0, locked=True),
            (4, 5): _make_pipe(PIPE_STRAIGHT, 1),
            (2, 0): _make_pipe(PIPE_STRAIGHT, 1),
            (3, 0): _make_pipe(PIPE_BEND, 0),
            (4, 0): _make_pipe(PIPE_STRAIGHT, 0),
            (4, 1): _make_pipe(PIPE_BEND, 1),
            (3, 1): _make_pipe(PIPE_STRAIGHT, 0),
            (2, 1): _make_pipe(PIPE_BEND, 3),
            (1, 4): _make_pipe(PIPE_STRAIGHT, 0),
            (1, 5): _make_pipe(PIPE_BEND, 1),
            (2, 5): _make_pipe(PIPE_STRAIGHT, 1),
            (3, 3): _make_pipe(PIPE_BEND, 3),
            (4, 3): _make_pipe(PIPE_STRAIGHT, 0),
            (4, 4): _make_pipe(PIPE_BEND, 0),
        },
        "target_rots": {
            (1, 0): 2,
            (1, 1): 0,
            (1, 2): 0,
            (2, 2): 1,
            (3, 2): 1,
            (4, 2): 1,
            (1, 3): 0,
            (2, 3): 2,
            (2, 4): 0,
            (3, 4): 2,
            (3, 5): 0,
            (4, 5): 1,
            (2, 0): 1,
            (3, 0): 0,
            (4, 0): 0,
            (4, 1): 1,
            (3, 1): 0,
            (2, 1): 3,
            (1, 4): 0,
            (1, 5): 1,
            (2, 5): 1,
            (3, 3): 3,
            (4, 3): 0,
            (4, 4): 0,
        },
    },
    {
        "cols": 6,
        "rows": 6,
        "max_moves": 164,
        "source": (0, 5),
        "drains": [(5, 0), (5, 3)],
        "drain_colors": {(5, 0): FLOW_PURPLE, (5, 3): FLOW_RED},
        "rust_interval": 8,
        "leak_cost": 1,
        "blue_pipes": [(2, 2)],
        "pipes": {
            (1, 5): _make_pipe(PIPE_BEND, 3),
            (1, 4): _make_pipe(PIPE_STRAIGHT, 0),
            (1, 3): _make_pipe(PIPE_TEE, 0),
            (1, 2): _make_pipe(PIPE_BEND, 1),
            (2, 2): _make_pipe(PIPE_STRAIGHT, 1),
            (3, 2): _make_pipe(PIPE_BEND, 3),
            (3, 1): _make_pipe(PIPE_BEND, 1),
            (4, 1): _make_pipe(PIPE_BEND, 3),
            (4, 0): _make_pipe(PIPE_BEND, 1),
            (2, 3): _make_pipe(PIPE_STRAIGHT, 1),
            (3, 3): _make_pipe(PIPE_STRAIGHT, 1),
            (4, 3): _make_pipe(PIPE_STRAIGHT, 1),
            (2, 0): _make_pipe(PIPE_BEND, 2),
            (2, 1): _make_pipe(PIPE_STRAIGHT, 0),
            (4, 2): _make_pipe(PIPE_STRAIGHT, 0),
            (2, 4): _make_pipe(PIPE_BEND, 0),
            (3, 4): _make_pipe(PIPE_STRAIGHT, 1),
            (4, 4): _make_pipe(PIPE_BEND, 3),
            (2, 5): _make_pipe(PIPE_BEND, 1),
            (3, 5): _make_pipe(PIPE_STRAIGHT, 0),
            (4, 5): _make_pipe(PIPE_BEND, 2),
            (1, 1): _make_pipe(PIPE_STRAIGHT, 0),
            (1, 0): _make_pipe(PIPE_BEND, 0),
        },
        "target_rots": {
            (1, 5): 3,
            (1, 4): 0,
            (1, 3): 0,
            (1, 2): 1,
            (2, 2): 1,
            (3, 2): 3,
            (3, 1): 1,
            (4, 1): 3,
            (4, 0): 1,
            (2, 3): 1,
            (3, 3): 1,
            (4, 3): 1,
            (2, 0): 2,
            (2, 1): 0,
            (4, 2): 0,
            (2, 4): 0,
            (3, 4): 1,
            (4, 4): 3,
            (2, 5): 1,
            (3, 5): 0,
            (4, 5): 2,
            (1, 1): 0,
            (1, 0): 0,
        },
    },
    {
        "cols": 6,
        "rows": 6,
        "max_moves": 160,
        "source": (0, 3),
        "drains": [(5, 0), (5, 5)],
        "drain_colors": {(5, 0): FLOW_RED, (5, 5): FLOW_PURPLE},
        "rust_interval": 6,
        "leak_cost": 2,
        "blue_pipes": [(1, 4)],
        "pipes": {
            (1, 3): _make_pipe(PIPE_TEE, 2),
            (1, 2): _make_pipe(PIPE_STRAIGHT, 0),
            (1, 1): _make_pipe(PIPE_BEND, 1),
            (2, 1): _make_pipe(PIPE_STRAIGHT, 1),
            (3, 1): _make_pipe(PIPE_BEND, 3),
            (3, 0): _make_pipe(PIPE_BEND, 1, locked=True),
            (4, 0): _make_pipe(PIPE_STRAIGHT, 1),
            (1, 4): _make_pipe(PIPE_BEND, 0),
            (2, 4): _make_pipe(PIPE_STRAIGHT, 1),
            (3, 4): _make_pipe(PIPE_BEND, 2),
            (3, 5): _make_pipe(PIPE_BEND, 0, locked=True),
            (4, 5): _make_pipe(PIPE_STRAIGHT, 1),
            (2, 0): _make_pipe(PIPE_BEND, 0),
            (2, 2): _make_pipe(PIPE_STRAIGHT, 1),
            (4, 1): _make_pipe(PIPE_BEND, 1),
            (4, 2): _make_pipe(PIPE_STRAIGHT, 0),
            (2, 3): _make_pipe(PIPE_BEND, 3),
            (4, 3): _make_pipe(PIPE_STRAIGHT, 0),
            (3, 2): _make_pipe(PIPE_BEND, 2),
            (3, 3): _make_pipe(PIPE_STRAIGHT, 0),
            (1, 5): _make_pipe(PIPE_BEND, 1),
            (2, 5): _make_pipe(PIPE_STRAIGHT, 1),
            (4, 4): _make_pipe(PIPE_BEND, 0),
        },
        "target_rots": {
            (1, 3): 2,
            (1, 2): 0,
            (1, 1): 1,
            (2, 1): 1,
            (3, 1): 3,
            (3, 0): 1,
            (4, 0): 1,
            (1, 4): 0,
            (2, 4): 1,
            (3, 4): 2,
            (3, 5): 0,
            (4, 5): 1,
            (2, 0): 0,
            (2, 2): 1,
            (4, 1): 1,
            (4, 2): 0,
            (2, 3): 3,
            (4, 3): 0,
            (3, 2): 2,
            (3, 3): 0,
            (1, 5): 1,
            (2, 5): 1,
            (4, 4): 0,
        },
    },
    {
        "cols": 6,
        "rows": 6,
        "max_moves": 200,
        "source": (0, 3),
        "drains": [(5, 0), (5, 3), (5, 5)],
        "drain_colors": {(5, 0): FLOW_RED, (5, 3): FLOW_PURPLE, (5, 5): FLOW_RED},
        "rust_interval": 5,
        "leak_cost": 2,
        "blue_pipes": [(2, 3)],
        "pipes": {
            (1, 3): _make_pipe(PIPE_CROSS, 0),
            (1, 2): _make_pipe(PIPE_STRAIGHT, 0),
            (1, 1): _make_pipe(PIPE_BEND, 1),
            (2, 1): _make_pipe(PIPE_STRAIGHT, 1),
            (3, 1): _make_pipe(PIPE_BEND, 3, locked=True),
            (3, 0): _make_pipe(PIPE_BEND, 1),
            (4, 0): _make_pipe(PIPE_STRAIGHT, 1),
            (2, 3): _make_pipe(PIPE_STRAIGHT, 1),
            (3, 3): _make_pipe(PIPE_STRAIGHT, 1, locked=True),
            (4, 3): _make_pipe(PIPE_STRAIGHT, 1),
            (1, 4): _make_pipe(PIPE_STRAIGHT, 0),
            (1, 5): _make_pipe(PIPE_BEND, 0),
            (2, 5): _make_pipe(PIPE_STRAIGHT, 1),
            (3, 5): _make_pipe(PIPE_BEND, 3),
            (3, 4): _make_pipe(PIPE_BEND, 1),
            (4, 4): _make_pipe(PIPE_BEND, 2),
            (4, 5): _make_pipe(PIPE_BEND, 0),
            (2, 0): _make_pipe(PIPE_BEND, 0),
            (2, 2): _make_pipe(PIPE_BEND, 2),
            (4, 1): _make_pipe(PIPE_STRAIGHT, 0),
            (4, 2): _make_pipe(PIPE_BEND, 3),
            (2, 4): _make_pipe(PIPE_STRAIGHT, 1),
            (1, 0): _make_pipe(PIPE_STRAIGHT, 1),
            (3, 2): _make_pipe(PIPE_BEND, 1),
        },
        "target_rots": {
            (1, 3): 0,
            (1, 2): 0,
            (1, 1): 1,
            (2, 1): 1,
            (3, 1): 3,
            (3, 0): 1,
            (4, 0): 1,
            (2, 3): 1,
            (3, 3): 1,
            (4, 3): 1,
            (1, 4): 0,
            (1, 5): 0,
            (2, 5): 1,
            (3, 5): 3,
            (3, 4): 1,
            (4, 4): 2,
            (4, 5): 0,
            (2, 0): 0,
            (2, 2): 2,
            (4, 1): 0,
            (4, 2): 3,
            (2, 4): 1,
            (1, 0): 1,
            (3, 2): 1,
        },
    },
]


def _scramble_pipes(pipes, target_rots, seed):
    rng = _random_mod.Random(seed)
    result = {}
    for pos, pipe in pipes.items():
        sol_rot = target_rots[pos]
        if pipe["locked"]:
            result[pos] = {"type": pipe["type"], "rot": sol_rot, "locked": True}
        elif pipe["type"] == PIPE_CROSS:
            result[pos] = {"type": pipe["type"], "rot": sol_rot, "locked": False}
        elif pipe["type"] == PIPE_STRAIGHT:
            new_rot = (sol_rot + rng.choice([1, 3])) % 4
            result[pos] = {"type": pipe["type"], "rot": new_rot, "locked": False}
        else:
            chosen = rng.choice([1, 2, 3])
            new_rot = (sol_rot + chosen) % 4
            result[pos] = {"type": pipe["type"], "rot": new_rot, "locked": False}
    return result


sprites = {
    "dot": Sprite(
        pixels=[[0]],
        name="dot",
        visible=True,
        collidable=False,
        tags=["dot"],
        layer=0,
    ),
}


def _build_level(cfg, idx):
    return Level(
        sprites=[sprites["dot"].clone().set_position(0, 0)],
        grid_size=(64, 64),
        data=cfg,
        name=f"lv{idx + 1}",
    )


_LEVELS = [
    _build_level(LEVEL_DATA[0], 0),
    _build_level(LEVEL_DATA[1], 1),
    _build_level(LEVEL_DATA[2], 2),
    _build_level(LEVEL_DATA[3], 3),
]


class PipeHud(RenderableUserDisplay):
    def __init__(self, game: "Nv48"):
        self.game = game

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        g = self.game
        if g.grid_cols == 0 or g.grid_rows == 0:
            return frame

        cs = self._cell_size(g)
        tw = g.grid_cols * cs
        th = g.grid_rows * cs
        ox = (64 - tw) // 2
        oy = max(1, (52 - th) // 2)

        self._draw_board_bg(frame, ox, oy, tw, th)

        pulse_vis = set()
        if g.pulse_timer > 0 and g.flow_order:
            depth = len(g.flow_order)
            vis_count = min(depth, max(1, int(depth * g.pulse_timer / g.pulse_max)))
            pulse_vis = set(g.flow_order[:vis_count])

        for gy in range(g.grid_rows):
            for gx in range(g.grid_cols):
                px = ox + gx * cs
                py = oy + gy * cs
                pos = (gx, gy)
                if pos == g.source_pos:
                    self._draw_source(frame, px, py, cs)
                elif pos in g.drain_set:
                    dc = g.drain_colors.get(pos, FLOW_RED)
                    reached_ok = pos in g.drain_reached and g.drain_reached[pos] == dc
                    self._draw_drain(frame, px, py, cs, dc, reached_ok)
                elif pos in g.pipes:
                    pipe = g.pipes[pos]
                    flow_color = g.flow_colors.get(pos, -1)
                    leaking = pos in g.leak_set
                    rusted = pos in g.rusted_pipes
                    pulsing = pos in pulse_vis and flow_color >= 0
                    is_blue = pos in g.blue_set
                    self._draw_pipe(
                        frame,
                        px,
                        py,
                        cs,
                        pipe,
                        flow_color,
                        leaking,
                        rusted,
                        pulsing,
                        is_blue,
                    )
                else:
                    self._draw_empty_cell(frame, px, py, cs)

        if g.cursor_x is not None and g.cursor_y is not None:
            cpx = ox + g.cursor_x * cs
            cpy = oy + g.cursor_y * cs
            self._draw_cursor(frame, cpx, cpy, cs)

        self._draw_hud(frame, g)
        return frame

    def _cell_size(self, g):
        cs = min(10, min(52 // g.grid_cols, 48 // g.grid_rows))
        return max(7, cs)

    def _draw_board_bg(self, frame, ox, oy, tw, th):
        for dy in range(-1, th + 1):
            for dx in range(-1, tw + 1):
                fx, fy = ox + dx, oy + dy
                if 0 <= fx < 64 and 0 <= fy < 64:
                    if dx == -1 or dx == tw or dy == -1 or dy == th:
                        frame[fy, fx] = COL_GRID_LINE
                    else:
                        frame[fy, fx] = COL_BG

    def _draw_empty_cell(self, frame, px, py, cs):
        for dy in range(cs):
            for dx in range(cs):
                fx, fy = px + dx, py + dy
                if 0 <= fx < 64 and 0 <= fy < 64:
                    if dx == 0 or dy == 0:
                        frame[fy, fx] = COL_GRID_LINE
                    else:
                        frame[fy, fx] = COL_BG
        cx = px + cs // 2
        cy = py + cs // 2
        if 0 <= cx < 64 and 0 <= cy < 64:
            frame[cy, cx] = COL_EMPTY_DOT

    def _draw_source(self, frame, px, py, cs):
        for dy in range(cs):
            for dx in range(cs):
                fx, fy = px + dx, py + dy
                if 0 <= fx < 64 and 0 <= fy < 64:
                    if dx == 0 or dy == 0:
                        frame[fy, fx] = COL_GRID_LINE
                    else:
                        frame[fy, fx] = COL_BG
        m = cs // 2
        for dy in range(2, cs - 1):
            for dx in range(2, cs - 1):
                fx, fy = px + dx, py + dy
                if 0 <= fx < 64 and 0 <= fy < 64:
                    dist = abs(dx - m) + abs(dy - m)
                    if dist <= m - 2:
                        frame[fy, fx] = COL_SOURCE_GLOW
                    elif dist <= m - 1:
                        frame[fy, fx] = COL_SOURCE

    def _draw_drain(self, frame, px, py, cs, drain_color, reached_ok):
        for dy in range(cs):
            for dx in range(cs):
                fx, fy = px + dx, py + dy
                if 0 <= fx < 64 and 0 <= fy < 64:
                    if dx == 0 or dy == 0:
                        frame[fy, fx] = COL_GRID_LINE
                    else:
                        frame[fy, fx] = COL_BG
        if reached_ok:
            c_outer = COL_DRAIN_OK
            c_inner = COL_DRAIN_OK
        elif drain_color == FLOW_PURPLE:
            c_outer = COL_DRAIN_PURPLE
            c_inner = COL_DRAIN_INNER_PURPLE
        else:
            c_outer = COL_DRAIN_RED
            c_inner = COL_DRAIN_INNER_RED
        m = cs // 2
        for dy in range(2, cs - 1):
            for dx in range(2, cs - 1):
                fx, fy = px + dx, py + dy
                if 0 <= fx < 64 and 0 <= fy < 64:
                    dist = max(abs(dx - m), abs(dy - m))
                    if dist <= 1:
                        frame[fy, fx] = c_inner
                    elif dist <= m - 1:
                        frame[fy, fx] = c_outer

    def _draw_pipe(
        self, frame, px, py, cs, pipe, flow_color, leaking, rusted, pulsing, is_blue
    ):
        bg = COL_BLUE_PIPE_BG if is_blue else COL_BG
        for dy in range(cs):
            for dx in range(cs):
                fx, fy = px + dx, py + dy
                if 0 <= fx < 64 and 0 <= fy < 64:
                    if dx == 0 or dy == 0:
                        frame[fy, fx] = COL_GRID_LINE
                    else:
                        frame[fy, fx] = bg

        conns = _get_connections(pipe)
        locked = pipe["locked"]

        if leaking:
            pc = COL_LEAK
            cc = COL_LEAK
        elif pulsing:
            pc = COL_FLOW_PULSE
            cc = COL_FLOW_PULSE
        elif flow_color == FLOW_PURPLE:
            pc = COL_PIPE_LIT_PURPLE
            cc = COL_PIPE_CENTER_LIT_PURPLE
        elif flow_color == FLOW_RED:
            pc = COL_PIPE_LIT_RED
            cc = COL_PIPE_CENTER_LIT_RED
        else:
            pc = COL_PIPE
            cc = COL_PIPE

        m = cs // 2
        pw = max(1, (cs - 2) // 3)
        half_pw = pw // 2
        cx_px = px + m
        cy_px = py + m

        if N in conns:
            for dy in range(1, m):
                for dw in range(-half_pw, half_pw + 1):
                    fx = cx_px + dw
                    fy = py + dy
                    if 0 <= fx < 64 and 0 <= fy < 64:
                        frame[fy, fx] = pc
        if S in conns:
            for dy in range(m + 1, cs):
                for dw in range(-half_pw, half_pw + 1):
                    fx = cx_px + dw
                    fy = py + dy
                    if 0 <= fx < 64 and 0 <= fy < 64:
                        frame[fy, fx] = pc
        if W in conns:
            for dx in range(1, m):
                for dw in range(-half_pw, half_pw + 1):
                    fx = px + dx
                    fy = cy_px + dw
                    if 0 <= fx < 64 and 0 <= fy < 64:
                        frame[fy, fx] = pc
        if E in conns:
            for dx in range(m + 1, cs):
                for dw in range(-half_pw, half_pw + 1):
                    fx = px + dx
                    fy = cy_px + dw
                    if 0 <= fx < 64 and 0 <= fy < 64:
                        frame[fy, fx] = pc

        for dw_y in range(-half_pw, half_pw + 1):
            for dw_x in range(-half_pw, half_pw + 1):
                fx = cx_px + dw_x
                fy = cy_px + dw_y
                if 0 <= fx < 64 and 0 <= fy < 64:
                    frame[fy, fx] = cc

        if locked:
            for dx in range(1, cs):
                for fy_pos in [py + 1, py + cs - 1]:
                    fx = px + dx
                    if 0 <= fx < 64 and 0 <= fy_pos < 64:
                        if frame[fy_pos, fx] == bg:
                            frame[fy_pos, fx] = COL_LOCKED_BORDER
            for dy in range(1, cs):
                for fx_pos in [px + 1, px + cs - 1]:
                    fy = py + dy
                    if 0 <= fx_pos < 64 and 0 <= fy < 64:
                        if frame[fy, fx_pos] == bg:
                            frame[fy, fx_pos] = COL_LOCKED_BORDER

        if rusted and not locked:
            corners = [
                (px + 1, py + 1),
                (px + cs - 2, py + 1),
                (px + 1, py + cs - 2),
                (px + cs - 2, py + cs - 2),
            ]
            for fx, fy in corners:
                if 0 <= fx < 64 and 0 <= fy < 64:
                    frame[fy, fx] = COL_RUST

    def _draw_cursor(self, frame, px, py, cs):
        for i in range(cs):
            pts = [
                (px + i, py),
                (px + i, py + cs - 1),
                (px, py + i),
                (px + cs - 1, py + i),
            ]
            for fx, fy in pts:
                if 0 <= fx < 64 and 0 <= fy < 64:
                    frame[fy, fx] = COL_CURSOR

    def _draw_hud(self, frame, g):
        bar_y = 54
        bar_xs = 1
        bar_xe = 50
        bar_w = bar_xe - bar_xs + 1
        filled = 0
        if g.max_moves > 0:
            filled = max(0, min(bar_w, int(bar_w * g.moves_remaining / g.max_moves)))
        for dy in range(2):
            for x in range(bar_xs, bar_xe + 1):
                fy = bar_y + dy
                if 0 <= fy < 64:
                    frame[fy, x] = (
                        COL_BAR_FULL if (x - bar_xs) < filled else COL_BAR_EMPTY
                    )

        for i in range(3):
            lx = 54 + i * 4
            c = COL_LIFE_ON if i < g.lives else COL_LIFE_OFF
            for dy in range(2):
                for dx in range(2):
                    fy = bar_y + dy
                    fx = lx + dx
                    if 0 <= fy < 64 and 0 <= fx < 64:
                        frame[fy, fx] = c


class Nv48(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self._rng = _random_mod.Random(seed)
        self.hud = PipeHud(self)

        self.lives: int = 3
        self.moves_remaining: int = 0
        self.max_moves: int = 0

        self.grid_cols: int = 0
        self.grid_rows: int = 0

        self.source_pos: Tuple[int, int] = (0, 0)
        self.drain_list: List[Tuple[int, int]] = []
        self.drain_set: Set[Tuple[int, int]] = set()
        self.drain_colors: Dict[Tuple[int, int], int] = {}
        self.pipes: Dict[Tuple[int, int], dict] = {}
        self.init_pipes: Dict[Tuple[int, int], dict] = {}

        self.flow_colors: Dict[Tuple[int, int], int] = {}
        self.drain_reached: Dict[Tuple[int, int], int] = {}
        self.flow_order: List[Tuple[int, int]] = []
        self.leak_set: Set[Tuple[int, int]] = set()
        self.rusted_pipes: Set[Tuple[int, int]] = set()

        self.blue_set: Set[Tuple[int, int]] = set()

        self.cursor_x: Optional[int] = None
        self.cursor_y: Optional[int] = None

        self.rust_interval: int = 0
        self.leak_cost: int = 0
        self.rot_since_rust: int = 0
        self.total_rotations: int = 0
        self.total_actions: int = 0

        self.pulse_timer: int = 0
        self.pulse_max: int = 8

        self._undo_snapshot: Optional[dict] = None
        self._can_undo: bool = False

        game_levels = list(_LEVELS)
        super().__init__(
            GAME_ID,
            game_levels,
            Camera(0, 0, 64, 64, BACKGROUND_COLOR, PADDING_COLOR, [self.hud]),
            False,
            1,
            [0, 1, 2, 3, 4, 5, 6, 7],
        )

    def on_set_level(self, level: Level) -> None:
        cfg_cols = self.current_level.get_data("cols")
        cfg_rows = self.current_level.get_data("rows")
        cfg_max = self.current_level.get_data("max_moves")
        cfg_source = self.current_level.get_data("source")
        cfg_drains = self.current_level.get_data("drains")
        cfg_pipes = self.current_level.get_data("pipes")
        cfg_sol = self.current_level.get_data("target_rots")

        self.grid_cols = cfg_cols if cfg_cols else 6
        self.grid_rows = cfg_rows if cfg_rows else 6
        self.max_moves = cfg_max if cfg_max else 84
        self.moves_remaining = self.max_moves
        self.lives = 3
        self.total_actions = 0

        ri = self.current_level.get_data("rust_interval")
        self.rust_interval = ri if ri else 0
        lc = self.current_level.get_data("leak_cost")
        self.leak_cost = lc if lc else 0
        self.rot_since_rust = 0
        self.total_rotations = 0
        self.rusted_pipes = set()
        self.pulse_timer = 0

        raw_dc = self.current_level.get_data("drain_colors")
        self.drain_colors = {}
        if raw_dc:
            for k, v in raw_dc.items():
                self.drain_colors[(k[0], k[1])] = v

        raw_bp = self.current_level.get_data("blue_pipes")
        self.blue_set = set()
        if raw_bp:
            for bp in raw_bp:
                self.blue_set.add((bp[0], bp[1]))

        if cfg_source:
            self.source_pos = (cfg_source[0], cfg_source[1])
        else:
            self.source_pos = (0, 0)

        self.drain_list = []
        self.drain_set = set()
        if cfg_drains:
            for d in cfg_drains:
                pos = (d[0], d[1])
                self.drain_list.append(pos)
                self.drain_set.add(pos)

        if cfg_pipes and cfg_sol:
            self.pipes = _scramble_pipes(cfg_pipes, cfg_sol, self._seed)
        elif cfg_pipes:
            self.pipes = {}
            for pos, pipe in cfg_pipes.items():
                self.pipes[pos] = {
                    "type": pipe["type"],
                    "rot": pipe["rot"],
                    "locked": pipe["locked"],
                }
        else:
            self.pipes = {}

        self.init_pipes = {}
        for pos, pipe in self.pipes.items():
            self.init_pipes[pos] = {
                "type": pipe["type"],
                "rot": pipe["rot"],
                "locked": pipe["locked"],
            }

        self._undo_snapshot = None
        self._can_undo = False

        self._init_cursor()
        self._update_flow()

    def _init_cursor(self) -> None:
        self.cursor_x = None
        self.cursor_y = None
        unlocked = [
            (gx, gy)
            for gy in range(self.grid_rows)
            for gx in range(self.grid_cols)
            if (gx, gy) in self.pipes and not self.pipes[(gx, gy)]["locked"]
        ]
        if unlocked:
            pos = self._rng.choice(unlocked)
            self.cursor_x, self.cursor_y = pos
            return
        if self.pipes:
            p0 = list(self.pipes.keys())[0]
            self.cursor_x, self.cursor_y = p0

    def _reset_level_state(self) -> None:
        self.pipes = {}
        for pos, pipe in self.init_pipes.items():
            self.pipes[pos] = {
                "type": pipe["type"],
                "rot": pipe["rot"],
                "locked": pipe["locked"],
            }
        self.moves_remaining = self.max_moves
        self.rot_since_rust = 0
        self.total_rotations = 0
        self.total_actions = 0
        self.rusted_pipes = set()
        self.pulse_timer = 0
        self._init_cursor()
        self._update_flow()

    def _update_flow(self) -> None:
        raw_blue = list(self.blue_set)
        self.flow_colors, self.drain_reached, self.flow_order = _compute_flow(
            self.source_pos,
            self.drain_list,
            self.pipes,
            self.grid_cols,
            self.grid_rows,
            raw_blue,
            self.drain_colors,
        )
        if self.leak_cost > 0:
            self.leak_set = _compute_leaks(
                set(self.flow_colors.keys()),
                self.pipes,
                self.source_pos,
                self.drain_list,
                self.grid_cols,
                self.grid_rows,
            )
        else:
            self.leak_set = set()
        self.pulse_timer = self.pulse_max

    def _check_win(self) -> bool:
        return _check_color_win(self.drain_list, self.drain_reached, self.drain_colors)

    def _apply_rust(self) -> None:
        if self.rust_interval <= 0:
            return
        self.rot_since_rust += 1
        if self.rot_since_rust >= self.rust_interval:
            self.rot_since_rust = 0
            unlocked = [
                pos
                for pos, pipe in self.pipes.items()
                if not pipe["locked"] and pipe["type"] != PIPE_CROSS
            ]
            if unlocked:
                target = self._rng.choice(unlocked)
                pipe = self.pipes[target]
                old_rot = pipe["rot"]
                pipe["rot"] = (old_rot + self._rng.choice([1, 2, 3])) % 4
                self.rusted_pipes.add(target)

    def _apply_leak_penalty(self) -> int:
        if self.leak_cost <= 0:
            return 0
        return len(self.leak_set) * self.leak_cost

    def _rotate_at_cursor(self) -> bool:
        if self.cursor_x is None or self.cursor_y is None:
            return False
        pos = (self.cursor_x, self.cursor_y)
        if pos not in self.pipes:
            return False
        pipe = self.pipes[pos]
        if pipe["locked"]:
            return False
        pipe["rot"] = (pipe["rot"] + 1) % 4
        if pos in self.rusted_pipes:
            self.rusted_pipes.discard(pos)
        self.total_rotations += 1
        self._apply_rust()
        self._update_flow()
        base_cost = 1
        leak_penalty = self._apply_leak_penalty()
        total_cost = base_cost + leak_penalty
        self.moves_remaining = max(0, self.moves_remaining - total_cost)
        return True

    def _move_cursor(self, dx: int, dy: int) -> None:
        if self.cursor_x is None or self.cursor_y is None:
            return
        cx, cy = self.cursor_x + dx, self.cursor_y + dy
        while 0 <= cx < self.grid_cols and 0 <= cy < self.grid_rows:
            if (cx, cy) in self.pipes:
                self.cursor_x = cx
                self.cursor_y = cy
                return
            cx += dx
            cy += dy

    def _save_snapshot(self) -> None:
        pipes_copy = {}
        for pos, pipe in self.pipes.items():
            pipes_copy[pos] = {
                "type": pipe["type"],
                "rot": pipe["rot"],
                "locked": pipe["locked"],
            }
        self._undo_snapshot = {
            "pipes": pipes_copy,
            "cursor_x": self.cursor_x,
            "cursor_y": self.cursor_y,
            "total_rotations": self.total_rotations,
            "total_actions": self.total_actions,
            "rot_since_rust": self.rot_since_rust,
            "rusted_pipes": set(self.rusted_pipes),
            "pulse_timer": self.pulse_timer,
        }

    def _restore_snapshot(self) -> None:
        snap = self._undo_snapshot
        if snap is None:
            return
        self.pipes = snap["pipes"]
        self.cursor_x = snap["cursor_x"]
        self.cursor_y = snap["cursor_y"]
        self.total_rotations = snap["total_rotations"]
        self.total_actions = snap["total_actions"]
        self.rot_since_rust = snap["rot_since_rust"]
        self.rusted_pipes = snap["rusted_pipes"]
        self.pulse_timer = snap["pulse_timer"]
        self._update_flow()

    def step(self) -> None:
        aid = self.action.id

        if self.pulse_timer > 0:
            self.pulse_timer -= 1

        if aid == GameAction.ACTION7:
            if self._can_undo and self._undo_snapshot is not None:
                self._restore_snapshot()
                self.moves_remaining = max(0, self.moves_remaining - 1)
                self._can_undo = False
                self._undo_snapshot = None
                if self.moves_remaining <= 0:
                    self._lose_life()
            self.complete_action()
            return

        if aid == GameAction.ACTION1:
            self._save_snapshot()
            self._move_cursor(0, -1)
            self.total_actions += 1
            self.moves_remaining = max(0, self.moves_remaining - 1)
            self._can_undo = True
            if self.moves_remaining <= 0:
                self._lose_life()
            self.complete_action()
            return

        if aid == GameAction.ACTION2:
            self._save_snapshot()
            self._move_cursor(0, 1)
            self.total_actions += 1
            self.moves_remaining = max(0, self.moves_remaining - 1)
            self._can_undo = True
            if self.moves_remaining <= 0:
                self._lose_life()
            self.complete_action()
            return

        if aid == GameAction.ACTION3:
            self._save_snapshot()
            self._move_cursor(-1, 0)
            self.total_actions += 1
            self.moves_remaining = max(0, self.moves_remaining - 1)
            self._can_undo = True
            if self.moves_remaining <= 0:
                self._lose_life()
            self.complete_action()
            return

        if aid == GameAction.ACTION4:
            self._save_snapshot()
            self._move_cursor(1, 0)
            self.total_actions += 1
            self.moves_remaining = max(0, self.moves_remaining - 1)
            self._can_undo = True
            if self.moves_remaining <= 0:
                self._lose_life()
            self.complete_action()
            return

        if aid == GameAction.ACTION5:
            self._save_snapshot()
            rotated = self._rotate_at_cursor()
            if rotated:
                self.total_actions += 1
                self._can_undo = True
                if self._check_win():
                    self._can_undo = False
                    self._undo_snapshot = None
                    self.next_level()
                    self.complete_action()
                    return
                if self.moves_remaining <= 0:
                    self._lose_life()
                    self.complete_action()
                    return
            else:
                self._can_undo = True
            self.complete_action()
            return

        if aid == GameAction.ACTION6:
            self._save_snapshot()
            raw_x = self.action.data.get("x", 0)
            raw_y = self.action.data.get("y", 0)
            coords = self.camera.display_to_grid(raw_x, raw_y)
            if coords:
                gx, gy = coords
                cs = self.hud._cell_size(self)
                total_w = self.grid_cols * cs
                total_h = self.grid_rows * cs
                grid_ox = (64 - total_w) // 2
                grid_oy = max(1, (52 - total_h) // 2)
                rel_x = gx - grid_ox
                rel_y = gy - grid_oy
                if rel_x >= 0 and rel_y >= 0:
                    tile_x = rel_x // cs
                    tile_y = rel_y // cs
                    if 0 <= tile_x < self.grid_cols and 0 <= tile_y < self.grid_rows:
                        if (tile_x, tile_y) in self.pipes:
                            self.cursor_x = tile_x
                            self.cursor_y = tile_y
                            rotated = self._rotate_at_cursor()
                            if rotated:
                                self.total_actions += 1
                                self._can_undo = True
                                if self._check_win():
                                    self._can_undo = False
                                    self._undo_snapshot = None
                                    self.next_level()
                                    self.complete_action()
                                    return
                                if self.moves_remaining <= 0:
                                    self._lose_life()
                                    self.complete_action()
                                    return
            self._can_undo = True
            self.complete_action()
            return

        self.complete_action()

    def _lose_life(self) -> None:
        self.lives -= 1
        if self.lives <= 0:
            self.lose()
            return
        self._reset_level_state()


DIR_NAME = {N: "N", E: "E", S: "S", W: "W"}
PIPE_CHAR = {PIPE_STRAIGHT: "I", PIPE_BEND: "L", PIPE_TEE: "T", PIPE_CROSS: "+"}


class PuzzleEnvironment:
    ACTION_MAP: Dict[str, GameAction] = {
        "reset": GameAction.RESET,
        "up": GameAction.ACTION1,
        "down": GameAction.ACTION2,
        "left": GameAction.ACTION3,
        "right": GameAction.ACTION4,
        "select": GameAction.ACTION5,
        "click": GameAction.ACTION6,
        "undo": GameAction.ACTION7,
    }

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

    def __init__(self, seed: int = 0) -> None:
        self._engine = Nv48(seed=seed)
        self._total_turns: int = 0
        self._prev_score: int = 0
        self._last_action_was_reset: bool = True

    def _build_text_obs(self) -> str:
        e = self._engine
        grid = [["." for _ in range(e.grid_cols)] for _ in range(e.grid_rows)]

        sx, sy = e.source_pos
        if 0 <= sx < e.grid_cols and 0 <= sy < e.grid_rows:
            grid[sy][sx] = "S"

        for dpos in e.drain_list:
            dx, dy = dpos
            if 0 <= dx < e.grid_cols and 0 <= dy < e.grid_rows:
                dc = e.drain_colors.get(dpos, FLOW_RED)
                reached_ok = dpos in e.drain_reached and e.drain_reached[dpos] == dc
                if reached_ok:
                    grid[dy][dx] = "X"
                elif dc == FLOW_PURPLE:
                    grid[dy][dx] = "P"
                else:
                    grid[dy][dx] = "D"

        for pos, pipe in e.pipes.items():
            px, py = pos
            if 0 <= px < e.grid_cols and 0 <= py < e.grid_rows:
                conns = _get_connections(pipe)
                dirs = "".join(DIR_NAME[d] for d in sorted(conns))
                flow = ""
                if pos in e.flow_colors:
                    flow = "r" if e.flow_colors[pos] == FLOW_RED else "p"
                lock = "*" if pipe["locked"] else ""
                grid[py][px] = dirs + flow + lock

        if e.cursor_x is not None and e.cursor_y is not None:
            cx, cy = e.cursor_x, e.cursor_y
            if 0 <= cx < e.grid_cols and 0 <= cy < e.grid_rows:
                cell = grid[cy][cx]
                grid[cy][cx] = "[" + cell + "]"

        drains_ok = sum(
            1
            for d in e.drain_list
            if d in e.drain_reached
            and e.drain_reached[d] == e.drain_colors.get(d, FLOW_RED)
        )
        header = (
            f"Level:{e.level_index + 1}/{len(e._levels)} "
            f"Lives:{e.lives} "
            f"Moves:{e.moves_remaining}/{e.max_moves} "
            f"Drains:{drains_ok}/{len(e.drain_list)}"
        )
        if e.leak_set:
            header += f" Leaks:{len(e.leak_set)}"

        grid_text = "\n".join(" ".join(str(c).ljust(5) for c in row) for row in grid)
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
        index_grid = e.camera.render(e.current_level.get_sprites())
        if index_grid is None or index_grid.size == 0:
            return None
        h, w = index_grid.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for idx, color in enumerate(self.ARC_PALETTE):
            mask = index_grid == idx
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
                "levels_completed": getattr(e, "_score", 0),
                "game_over": getattr(e._state, "name", "") == "GAME_OVER",
                "lives": e.lives,
                "moves_remaining": e.moves_remaining,
                "moves_max": e.max_moves,
                "done": done,
                "info": info or {},
            },
        )

    def reset(self) -> GameState:
        e = self._engine
        if self.is_done() and e._state == EngineState.WIN:
            e.full_reset()
        elif self._last_action_was_reset:
            e.full_reset()
        else:
            e.perform_action(ActionInput(id=GameAction.RESET))
        self._total_turns = 0
        self._prev_score = getattr(e, "_score", 0)
        self._last_action_was_reset = True
        return self._build_game_state()

    def get_actions(self) -> List[str]:
        if self.is_done():
            return ["reset"]
        return ["reset", "up", "down", "left", "right", "select", "click", "undo"]

    def is_done(self) -> bool:
        e = self._engine
        state_name = getattr(e._state, "name", "")
        return state_name in ("WIN", "GAME_OVER")

    def step(self, action: str) -> StepResult:
        e = self._engine

        if action == "reset":
            state = self.reset()
            return StepResult(
                state=state, reward=0.0, done=False, info={"action": "reset"}
            )

        parts = action.split() if isinstance(action, str) else []
        base_action = parts[0] if parts else ""
        if base_action not in self.ACTION_MAP:
            return StepResult(
                state=self._build_game_state(),
                reward=0.0,
                done=self.is_done(),
                info={"action": action, "error": "invalid_action"},
            )

        self._last_action_was_reset = False
        self._total_turns += 1

        game_action = self.ACTION_MAP[base_action]

        x_val = None
        y_val = None
        if base_action == "click" and len(parts) >= 3:
            x_val = int(parts[1])
            y_val = int(parts[2])

        action_input = ActionInput(id=game_action, x=x_val, y=y_val)

        prev_score = self._prev_score
        frame = e.perform_action(action_input, raw=True)
        levels_advanced = frame.levels_completed - prev_score
        self._prev_score = frame.levels_completed

        reward = levels_advanced * (1.0 / len(e._levels))
        done = frame.state.name in ("WIN", "GAME_OVER")

        info: Dict = {"action": action}
        if done:
            info["reason"] = (
                "game_complete" if frame.state.name == "WIN" else "game_over"
            )

        return StepResult(
            state=self._build_game_state(done=done, info=info),
            reward=reward,
            done=done,
            info=info,
        )

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError(f"Unsupported render mode: {mode}")
        e = self._engine
        index_grid = e.camera.render(e.current_level.get_sprites())
        h, w = index_grid.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for idx, color in enumerate(self.ARC_PALETTE):
            mask = index_grid == idx
            rgb[mask] = color
        if h != 64 or w != 64:
            row_idx = (np.arange(64) * h // 64).astype(int)
            col_idx = (np.arange(64) * w // 64).astype(int)
            rgb = rgb[np.ix_(row_idx, col_idx)]
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
        "click",
        "undo",
    ]

    OBS_HEIGHT: int = 64
    OBS_WIDTH: int = 64

    def __init__(self, seed: int = 0, render_mode: Optional[str] = None) -> None:
        super().__init__()

        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode '{render_mode}'.")
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
        self._env: Optional[PuzzleEnvironment] = None

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
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
        terminated: bool = result.done
        truncated: bool = False
        return (
            obs,
            result.reward,
            terminated,
            truncated,
            self._build_info(result.state, result.info),
        )

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
