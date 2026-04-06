from collections import deque
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

FRAME_SIZE = 64
UI_ROWS = 2
PLAY_ROWS = FRAME_SIZE - UI_ROWS

CELL_EMPTY = 0
CELL_WALL = 1
CELL_SOURCE_A = 2
CELL_SOURCE_B = 3
CELL_RECV_A = 4
CELL_RECV_B = 5
CELL_INSULATOR = 7
CELL_SWITCH = 9
CELL_CONDUIT_R = 10
CELL_CONDUIT_D = 11
CELL_CONDUIT_L = 12
CELL_CONDUIT_U = 13
CELL_GATE = 14
CELL_CAPACITOR = 15
CELL_DRAIN = 16

COLOR_BG = 0
COLOR_FLOOR = 1
COLOR_WALL = 5
COLOR_SOURCE_A = 8
COLOR_SOURCE_B = 9
COLOR_RECV_A_OFF = 3
COLOR_RECV_A_ON = 2
COLOR_RECV_B_OFF = 4
COLOR_RECV_B_ON = 10
COLOR_WIRE_OFF = 7
COLOR_WIRE_ON_A = 2
COLOR_WIRE_ON_B = 10
COLOR_INSULATOR = 13
COLOR_TRAP = 6
COLOR_TRAP_BOOM = 8
COLOR_CURSOR = 12
COLOR_SWITCH_OFF = 4
COLOR_SWITCH_ON = 11
COLOR_CONDUIT = 15
COLOR_GATE = 14
COLOR_CAP_EMPTY = 3
COLOR_CAP_FULL = 11
COLOR_DRAIN = 13
COLOR_BAR_FILL = 11
COLOR_BAR_EMPTY = 5
COLOR_LIFE = 8
COLOR_LIFE_EMPTY = 3

GRID_SIZES = [11, 12, 13, 14]
TILE_SIZES = [PLAY_ROWS // g for g in GRID_SIZES]

DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]

CONDUIT_DIR = {
    CELL_CONDUIT_R: (1, 0),
    CELL_CONDUIT_D: (0, 1),
    CELL_CONDUIT_L: (-1, 0),
    CELL_CONDUIT_U: (0, -1),
}

LEVEL_DATA = [
    {
        "grid": [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 4, 0, 0, 0, 0, 7, 0, 0, 4, 1],
            [1, 0, 7, 0, 0, 7, 7, 0, 0, 1, 1],
            [1, 0, 0, 0, 0, 0, 9, 0, 0, 13, 1],
            [1, 7, 11, 7, 0, 3, 7, 0, 0, 0, 1],
            [1, 0, 0, 7, 0, 0, 0, 7, 1, 0, 1],
            [1, 7, 12, 7, 0, 0, 9, 0, 0, 0, 1],
            [1, 4, 0, 7, 0, 0, 0, 0, 0, 0, 1],
            [1, 5, 0, 0, 0, 0, 0, 0, 2, 0, 1],
            [1, 7, 5, 0, 0, 0, 0, 0, 7, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
        "cursor": [1, 1],
        "sources_a": [(8, 8)],
        "sources_b": [(5, 4)],
        "receivers_a": [(1, 1), (1, 7), (9, 1)],
        "receivers_b": [(1, 8), (2, 9)],
        "traps": [(1, 3), (3, 2), (7, 1)],
        "switches": {(6, 6): False, (6, 3): False},
        "switch_walls": {(6, 6): (9, 2), (6, 3): (8, 5)},
        "conduits": {(9, 3): 13, (2, 4): 11, (2, 6): 12},
        "gates": [],
        "drains": [],
        "capacitors": [],
    },
    {
        "grid": [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 7, 7, 0, 0, 0, 0, 5, 7, 4, 7, 1],
            [1, 0, 0, 0, 0, 0, 0, 5, 4, 0, 4, 1],
            [1, 12, 0, 16, 0, 14, 0, 0, 0, 0, 5, 1],
            [1, 0, 0, 0, 0, 0, 7, 7, 11, 0, 0, 1],
            [1, 16, 0, 0, 0, 0, 0, 0, 0, 7, 0, 1],
            [1, 7, 0, 0, 0, 7, 0, 12, 0, 0, 0, 1],
            [1, 0, 0, 1, 3, 0, 0, 0, 0, 9, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 9, 0, 7, 1],
            [1, 2, 14, 0, 0, 0, 7, 7, 0, 0, 7, 1],
            [1, 0, 0, 0, 7, 0, 0, 0, 13, 2, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
        "cursor": [3, 1],
        "sources_a": [(1, 9), (9, 10)],
        "sources_b": [(4, 7)],
        "receivers_a": [(8, 2), (9, 1), (10, 2)],
        "receivers_b": [(7, 1), (7, 2), (10, 3)],
        "traps": [(4, 1), (6, 7), (8, 5), (10, 7)],
        "switches": {(9, 7): False, (8, 8): False},
        "switch_walls": {(9, 7): (3, 7), (8, 8): (10, 10)},
        "conduits": {(7, 6): 12, (1, 3): 12, (8, 10): 13, (8, 4): 11},
        "gates": [(5, 3), (2, 9)],
        "drains": [(1, 5), (3, 3)],
        "capacitors": [],
    },
    {
        "grid": [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 7, 5, 0, 4, 4, 0, 0, 0, 14, 0, 4, 1],
            [1, 7, 0, 16, 0, 5, 7, 0, 0, 0, 0, 5, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 7, 1],
            [1, 9, 0, 15, 0, 10, 16, 0, 7, 0, 15, 0, 1],
            [1, 0, 0, 0, 0, 7, 0, 0, 0, 11, 0, 0, 1],
            [1, 14, 0, 7, 0, 0, 9, 0, 2, 11, 0, 0, 1],
            [1, 0, 0, 3, 0, 7, 7, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 7, 1, 0, 0, 7, 1],
            [1, 1, 0, 16, 0, 0, 0, 7, 0, 0, 7, 7, 1],
            [1, 7, 0, 0, 3, 0, 7, 9, 13, 0, 1, 2, 1],
            [1, 0, 0, 7, 0, 0, 0, 14, 0, 0, 7, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
        "cursor": [2, 1],
        "sources_a": [(8, 6), (11, 10)],
        "sources_b": [(3, 7), (4, 10)],
        "receivers_a": [(4, 1), (5, 1), (11, 1)],
        "receivers_b": [(2, 1), (5, 2), (11, 2)],
        "traps": [(5, 6), (5, 9), (9, 7), (10, 8)],
        "switches": {(6, 6): False, (7, 10): False, (1, 4): False},
        "switch_walls": {(6, 6): (10, 10), (7, 10): (8, 8), (1, 4): (1, 9)},
        "conduits": {(10, 3): 12, (8, 10): 13, (9, 6): 11, (9, 5): 11, (5, 4): 10},
        "gates": [(9, 1), (1, 6), (7, 11)],
        "drains": [(3, 2), (3, 9), (6, 4)],
        "capacitors": [(3, 4), (10, 4)],
    },
    {
        "grid": [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 4, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 1],
            [1, 0, 7, 16, 0, 0, 0, 7, 3, 0, 0, 12, 0, 1],
            [1, 4, 0, 14, 7, 7, 12, 14, 7, 0, 0, 0, 7, 1],
            [1, 4, 0, 0, 0, 0, 7, 7, 9, 3, 16, 0, 7, 1],
            [1, 5, 0, 0, 7, 0, 0, 0, 0, 15, 0, 0, 0, 1],
            [1, 0, 1, 15, 7, 0, 0, 0, 0, 0, 0, 14, 0, 1],
            [1, 0, 10, 0, 0, 0, 14, 2, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 7, 11, 1],
            [1, 5, 7, 0, 13, 1, 0, 0, 7, 0, 7, 16, 0, 1],
            [1, 4, 7, 0, 15, 9, 0, 0, 0, 0, 0, 16, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 13, 2, 0, 0, 0, 7, 1],
            [1, 5, 0, 7, 7, 0, 0, 0, 1, 0, 7, 7, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
        "cursor": [1, 1],
        "sources_a": [(7, 7), (8, 11)],
        "sources_b": [(8, 2), (9, 4)],
        "receivers_a": [(1, 1), (1, 3), (1, 4), (1, 10)],
        "receivers_b": [(1, 5), (1, 9), (1, 12)],
        "traps": [(6, 11), (7, 6), (7, 9), (8, 6), (9, 9)],
        "switches": {(5, 10): False, (8, 4): False, (10, 1): False},
        "switch_walls": {(5, 10): (2, 6), (8, 4): (8, 12), (10, 1): (5, 9)},
        "conduits": {
            (6, 3): 12,
            (12, 8): 11,
            (11, 2): 12,
            (7, 11): 13,
            (2, 7): 10,
            (4, 9): 13,
        },
        "gates": [(3, 3), (7, 3), (11, 6), (6, 7)],
        "drains": [(3, 2), (10, 4), (11, 9), (11, 10)],
        "capacitors": [(3, 6), (4, 10), (9, 5)],
    },
]

MOVE_LIMITS = [250, 260, 342, 486]

BACKGROUND_COLOR = 0
PADDING_COLOR = 0

CELL_SPRITE_MAP = {
    CELL_WALL: "wall",
    CELL_INSULATOR: "insulator",
    CELL_SOURCE_A: "source_a",
    CELL_SOURCE_B: "source_b",
    CELL_CONDUIT_R: "conduit_r",
    CELL_CONDUIT_D: "conduit_d",
    CELL_CONDUIT_L: "conduit_l",
    CELL_CONDUIT_U: "conduit_u",
    CELL_DRAIN: "drain",
}

_sprite_cache = {}


class MoveDisplay(RenderableUserDisplay):
    MAX_LIVES = 3

    def __init__(self, game: "Cw03") -> None:
        self._game = game
        self.max_moves: int = 0
        self.remaining: int = 0

    def set_limit(self, max_moves: int) -> None:
        self.max_moves = max_moves
        self.remaining = max_moves

    def decrement(self) -> None:
        if self.remaining > 0:
            self.remaining -= 1

    def reset(self) -> None:
        self.remaining = self.max_moves

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        if self.max_moves == 0:
            return frame
        fw = frame.shape[1]
        fh = frame.shape[0]
        rs = fh - 2
        re = fh
        frame[rs:re, :] = COLOR_BAR_EMPTY
        bw = int(fw * 0.7)
        filled = int(bw * self.remaining / self.max_moves)
        for x in range(filled):
            frame[rs:re, x] = COLOR_BAR_FILL
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
            c = COLOR_LIFE if i < lives else COLOR_LIFE_EMPTY
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


def _get_sprites(ts):
    if ts in _sprite_cache:
        return _sprite_cache[ts]
    s = {}
    s["empty"] = Sprite(
        pixels=_make_tile(COLOR_FLOOR, ts), name="empty", visible=True, collidable=False
    )
    s["wall"] = Sprite(
        pixels=_make_tile(COLOR_WALL, ts), name="wall", visible=True, collidable=False
    )
    s["source_a"] = Sprite(
        pixels=_tile_border(COLOR_SOURCE_A, COLOR_WALL, ts),
        name="source_a",
        visible=True,
        collidable=False,
    )
    s["source_b"] = Sprite(
        pixels=_tile_border(COLOR_SOURCE_B, COLOR_WALL, ts),
        name="source_b",
        visible=True,
        collidable=False,
    )
    s["recv_a_off"] = Sprite(
        pixels=_tile_border(COLOR_RECV_A_OFF, COLOR_SOURCE_A, ts),
        name="ra_off",
        visible=True,
        collidable=False,
    )
    s["recv_a_on"] = Sprite(
        pixels=_tile_border(COLOR_RECV_A_ON, COLOR_SOURCE_A, ts),
        name="ra_on",
        visible=True,
        collidable=False,
    )
    s["recv_b_off"] = Sprite(
        pixels=_tile_border(COLOR_RECV_B_OFF, COLOR_SOURCE_B, ts),
        name="rb_off",
        visible=True,
        collidable=False,
    )
    s["recv_b_on"] = Sprite(
        pixels=_tile_border(COLOR_RECV_B_ON, COLOR_SOURCE_B, ts),
        name="rb_on",
        visible=True,
        collidable=False,
    )
    s["wire_off"] = Sprite(
        pixels=_make_tile(COLOR_WIRE_OFF, ts),
        name="wire_off",
        visible=True,
        collidable=False,
    )
    s["wire_on_a"] = Sprite(
        pixels=_make_tile(COLOR_WIRE_ON_A, ts),
        name="wire_a",
        visible=True,
        collidable=False,
    )
    s["wire_on_b"] = Sprite(
        pixels=_make_tile(COLOR_WIRE_ON_B, ts),
        name="wire_b",
        visible=True,
        collidable=False,
    )
    s["insulator"] = Sprite(
        pixels=_make_tile(COLOR_INSULATOR, ts),
        name="ins",
        visible=True,
        collidable=False,
    )
    s["trap"] = Sprite(
        pixels=_tile_border(COLOR_TRAP, COLOR_FLOOR, ts),
        name="trap",
        visible=True,
        collidable=False,
    )
    s["trap_boom"] = Sprite(
        pixels=_make_tile(COLOR_TRAP_BOOM, ts),
        name="boom",
        visible=True,
        collidable=False,
    )
    s["cursor"] = Sprite(
        pixels=_tile_border(COLOR_BG, COLOR_CURSOR, ts),
        name="cursor",
        visible=True,
        collidable=False,
    )
    s["switch_off"] = Sprite(
        pixels=_tile_border(COLOR_SWITCH_OFF, COLOR_FLOOR, ts),
        name="sw_off",
        visible=True,
        collidable=False,
    )
    s["switch_on"] = Sprite(
        pixels=_tile_border(COLOR_SWITCH_ON, COLOR_FLOOR, ts),
        name="sw_on",
        visible=True,
        collidable=False,
    )
    s["conduit_r"] = Sprite(
        pixels=_tile_arrow(COLOR_FLOOR, COLOR_CONDUIT, ts, 1, 0),
        name="cd_r",
        visible=True,
        collidable=False,
    )
    s["conduit_d"] = Sprite(
        pixels=_tile_arrow(COLOR_FLOOR, COLOR_CONDUIT, ts, 0, 1),
        name="cd_d",
        visible=True,
        collidable=False,
    )
    s["conduit_l"] = Sprite(
        pixels=_tile_arrow(COLOR_FLOOR, COLOR_CONDUIT, ts, -1, 0),
        name="cd_l",
        visible=True,
        collidable=False,
    )
    s["conduit_u"] = Sprite(
        pixels=_tile_arrow(COLOR_FLOOR, COLOR_CONDUIT, ts, 0, -1),
        name="cd_u",
        visible=True,
        collidable=False,
    )
    s["gate"] = Sprite(
        pixels=_tile_border(COLOR_GATE, COLOR_WALL, ts),
        name="gate",
        visible=True,
        collidable=False,
    )
    s["cap_empty"] = Sprite(
        pixels=_tile_border(COLOR_CAP_EMPTY, COLOR_FLOOR, ts),
        name="cap_e",
        visible=True,
        collidable=False,
    )
    s["cap_full"] = Sprite(
        pixels=_tile_border(COLOR_CAP_FULL, COLOR_FLOOR, ts),
        name="cap_f",
        visible=True,
        collidable=False,
    )
    s["drain"] = Sprite(
        pixels=_tile_border(COLOR_DRAIN, COLOR_FLOOR, ts),
        name="drain",
        visible=True,
        collidable=False,
    )
    _sprite_cache[ts] = s
    return s


levels = [
    Level(sprites=[], grid_size=(FRAME_SIZE, FRAME_SIZE)),
    Level(sprites=[], grid_size=(FRAME_SIZE, FRAME_SIZE)),
    Level(sprites=[], grid_size=(FRAME_SIZE, FRAME_SIZE)),
    Level(sprites=[], grid_size=(FRAME_SIZE, FRAME_SIZE)),
]


def _passable_normal(grid, gw, gh, pos, wires):
    x, y = pos
    if x < 0 or y < 0 or x >= gw or y >= gh:
        return False
    ct = grid[y][x]
    if ct in (CELL_WALL, CELL_INSULATOR, CELL_CAPACITOR):
        return False
    if ct == CELL_EMPTY:
        return pos in wires
    return True


def _passable_charged(grid, gw, gh, pos, wires, charged):
    x, y = pos
    if x < 0 or y < 0 or x >= gw or y >= gh:
        return False
    ct = grid[y][x]
    if ct in (CELL_WALL, CELL_INSULATOR):
        return False
    if ct == CELL_CAPACITOR:
        return pos in charged
    if ct == CELL_EMPTY:
        return pos in wires
    return True


def _power_bfs(
    grid, grid_w, grid_h, sources, wires, gate_open, charged=None, exclude=None
):
    passable_fn = _passable_normal
    if charged is not None:
        passable_fn = lambda g, gw, gh, pos, w: _passable_charged(
            g, gw, gh, pos, w, charged
        )

    powered = set()
    queue = deque()
    for source in sources:
        powered.add(source)
        queue.append(source)

    while queue:
        cur = queue.popleft()
        ct_cur = grid[cur[1]][cur[0]]

        if ct_cur in CONDUIT_DIR:
            direction = CONDUIT_DIR[ct_cur]
            nx, ny = cur[0] + direction[0], cur[1] + direction[1]
            neighbor = (nx, ny)
            if neighbor not in powered and passable_fn(
                grid, grid_w, grid_h, neighbor, wires
            ):
                if exclude is None or neighbor not in exclude:
                    powered.add(neighbor)
                    queue.append(neighbor)
            continue

        if ct_cur in (CELL_DRAIN, CELL_RECV_A, CELL_RECV_B):
            continue

        for ddx, ddy in DIRS:
            nx, ny = cur[0] + ddx, cur[1] + ddy
            neighbor = (nx, ny)
            if neighbor in powered:
                continue
            if exclude is not None and neighbor in exclude:
                continue
            if not passable_fn(grid, grid_w, grid_h, neighbor, wires):
                continue
            nct = grid[ny][nx]
            if nct in CONDUIT_DIR:
                cd = CONDUIT_DIR[nct]
                if (ddx, ddy) != (-cd[0], -cd[1]):
                    continue
            if nct == CELL_GATE and not gate_open.get(neighbor, True):
                continue
            powered.add(neighbor)
            queue.append(neighbor)

    return powered


class Cw03(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._seed = seed

        self._grid = []
        self._grid_w = 0
        self._grid_h = 0
        self._cursor = [1, 1]
        self._wires = set()
        self._powered_a = set()
        self._powered_b = set()
        self._sources_a = set()
        self._sources_b = set()
        self._receivers_a = set()
        self._receivers_b = set()
        self._traps = set()
        self._traps_blown = set()
        self._switches = {}
        self._switch_walls = {}
        self._gates = set()
        self._gate_open = {}
        self._capacitors = set()
        self._cap_charged = set()
        self._drains = set()
        self._move_count = 0

        self._lives = 3
        self._max_moves = 0
        self._moves_remaining = 0
        self._game_over = False
        self._action_count = 0

        self._init_state = {}
        self._undo_stack: List[dict] = []

        self._move_display = MoveDisplay(self)
        camera = Camera(
            0, 0, 64, 64, BACKGROUND_COLOR, PADDING_COLOR, [self._move_display]
        )
        super().__init__(
            "cw03",
            levels,
            camera,
            available_actions=[0, 1, 2, 3, 4, 5, 7],
        )

    def on_set_level(self, level: Level) -> None:
        self._game_over = False
        self._lives = 3
        self._move_count = 0
        self._action_count = 0
        self._undo_stack = []
        self._load_level()
        self._save_init_state()
        self._propagate_power()
        self._render_level(level)
        self._move_display.set_limit(self._max_moves)

    def _load_level(self):
        idx = self.level_index
        data = LEVEL_DATA[idx]
        gs = GRID_SIZES[idx]
        self._grid_w = gs
        self._grid_h = gs
        self._grid = [row[:] for row in data["grid"]]
        self._cursor = list(data["cursor"])
        self._wires = set()
        self._powered_a = set()
        self._powered_b = set()
        self._sources_a = set(data["sources_a"])
        self._sources_b = set(data["sources_b"])
        self._receivers_a = set(data["receivers_a"])
        self._receivers_b = set(data["receivers_b"])
        self._traps = set(data["traps"])
        self._traps_blown = set()
        self._switches = dict(data["switches"])
        self._switch_walls = dict(data["switch_walls"])
        self._gates = set(data["gates"])
        self._gate_open = {g: True for g in data["gates"]}
        self._drains = set(data["drains"])
        self._capacitors = set(data["capacitors"])
        self._cap_charged = set()
        self._max_moves = MOVE_LIMITS[idx]
        self._moves_remaining = self._max_moves

    def _save_init_state(self):
        self._init_state = {
            "grid": [row[:] for row in self._grid],
            "cursor": self._cursor[:],
            "wires": set(self._wires),
            "traps": set(self._traps),
            "switches": dict(self._switches),
            "switch_walls": dict(self._switch_walls),
            "gates": set(self._gates),
            "gate_open": dict(self._gate_open),
            "capacitors": set(self._capacitors),
            "drains": set(self._drains),
            "max_moves": self._max_moves,
        }

    def _restore_init_state(self):
        saved = self._init_state
        self._grid = [row[:] for row in saved["grid"]]
        self._cursor = saved["cursor"][:]
        self._wires = set(saved["wires"])
        self._traps = set(saved["traps"])
        self._traps_blown = set()
        self._switches = dict(saved["switches"])
        self._switch_walls = dict(saved["switch_walls"])
        self._gates = set(saved["gates"])
        self._gate_open = dict(saved["gate_open"])
        self._capacitors = set(saved["capacitors"])
        self._cap_charged = set()
        self._drains = set(saved["drains"])
        self._powered_a = set()
        self._powered_b = set()
        self._max_moves = saved["max_moves"]
        self._moves_remaining = self._max_moves
        self._move_count = 0
        self._move_display.set_limit(self._max_moves)
        self._game_over = False
        self._undo_stack = []

    def _snapshot_state(self):
        return {
            "grid": [row[:] for row in self._grid],
            "cursor": self._cursor[:],
            "wires": set(self._wires),
            "traps": set(self._traps),
            "traps_blown": set(self._traps_blown),
            "switches": dict(self._switches),
            "switch_walls": dict(self._switch_walls),
            "gates": set(self._gates),
            "gate_open": dict(self._gate_open),
            "cap_charged": set(self._cap_charged),
            "moves_remaining": self._moves_remaining,
            "move_count": self._move_count,
            "game_over": self._game_over,
        }

    def _restore_snapshot(self, snap):
        self._grid = [row[:] for row in snap["grid"]]
        self._cursor = snap["cursor"][:]
        self._wires = set(snap["wires"])
        self._traps = set(snap["traps"])
        self._traps_blown = set(snap["traps_blown"])
        self._switches = dict(snap["switches"])
        self._switch_walls = dict(snap["switch_walls"])
        self._gates = set(snap["gates"])
        self._gate_open = dict(snap["gate_open"])
        self._cap_charged = set(snap["cap_charged"])
        self._moves_remaining = snap["moves_remaining"]
        self._move_count = snap["move_count"]
        self._game_over = snap["game_over"]
        self._move_display.remaining = self._moves_remaining

    def _undo(self):
        if not self._undo_stack:
            self._moves_remaining -= 1
            self._move_display.decrement()
            self._move_count += 1
            if self._moves_remaining <= 0 and not self._check_win():
                self._restart_level()
                return
            self._render_level(self.current_level)
            return
        cur_remaining = self._moves_remaining
        cur_move_count = self._move_count
        snap = self._undo_stack.pop()
        self._restore_snapshot(snap)
        self._moves_remaining = cur_remaining - 1
        self._move_count = cur_move_count + 1
        self._move_display.remaining = self._moves_remaining
        if self._moves_remaining <= 0 and not self._check_win():
            self._restart_level()
            return
        self._propagate_power()
        self._render_level(self.current_level)

    def _propagate_power(self):
        self._powered_a = set()
        self._powered_b = set()
        self._cap_charged = set()
        gate_state = self._gate_open

        self._powered_a = _power_bfs(
            self._grid,
            self._grid_w,
            self._grid_h,
            self._sources_a,
            self._wires,
            gate_state,
        )
        self._powered_b = _power_bfs(
            self._grid,
            self._grid_w,
            self._grid_h,
            self._sources_b,
            self._wires,
            gate_state,
        )

        if self._capacitors:
            for cap in self._capacitors:
                for ddx, ddy in DIRS:
                    nx, ny = cap[0] + ddx, cap[1] + ddy
                    if (nx, ny) in self._powered_a or (nx, ny) in self._powered_b:
                        self._cap_charged.add(cap)
                        break

            if self._cap_charged:
                for cap in self._cap_charged:
                    for ddx, ddy in DIRS:
                        if (cap[0] + ddx, cap[1] + ddy) in self._powered_a:
                            self._powered_a |= _power_bfs(
                                self._grid,
                                self._grid_w,
                                self._grid_h,
                                {cap},
                                self._wires,
                                gate_state,
                                self._cap_charged,
                                exclude=self._powered_b,
                            )
                            break
                for cap in self._cap_charged:
                    for ddx, ddy in DIRS:
                        if (cap[0] + ddx, cap[1] + ddy) in self._powered_b:
                            self._powered_b |= _power_bfs(
                                self._grid,
                                self._grid_w,
                                self._grid_h,
                                {cap},
                                self._wires,
                                gate_state,
                                self._cap_charged,
                                exclude=self._powered_a,
                            )
                            break

    def _check_win(self):
        for receiver in self._receivers_a:
            if receiver not in self._powered_a:
                return False
        for receiver in self._receivers_b:
            if receiver not in self._powered_b:
                return False
        return True

    def _toggle_gates(self):
        for gate_pos in self._gates:
            self._gate_open[gate_pos] = not self._gate_open[gate_pos]

    def _resolve_sprite_key(self, cell_type, pos):
        if cell_type in CELL_SPRITE_MAP:
            return CELL_SPRITE_MAP[cell_type]
        if cell_type == CELL_RECV_A:
            return "recv_a_on" if pos in self._powered_a else "recv_a_off"
        if cell_type == CELL_RECV_B:
            return "recv_b_on" if pos in self._powered_b else "recv_b_off"
        if cell_type == CELL_EMPTY and pos in self._traps_blown:
            return "trap_boom"
        if cell_type == CELL_EMPTY and pos in self._traps:
            return "trap"
        if cell_type == CELL_SWITCH:
            return "switch_on" if self._switches.get(pos, False) else "switch_off"
        if cell_type == CELL_GATE:
            return "gate" if self._gate_open.get(pos, True) else "wall"
        if cell_type == CELL_CAPACITOR:
            return "cap_full" if pos in self._cap_charged else "cap_empty"
        if cell_type == CELL_EMPTY:
            if pos in self._wires:
                if pos in self._powered_a:
                    return "wire_on_a"
                if pos in self._powered_b:
                    return "wire_on_b"
                return "wire_off"
            return "empty"
        return "empty"

    def _render_level(self, level):
        for sp in list(level._sprites):
            level.remove_sprite(sp)

        idx = self.level_index
        gs = GRID_SIZES[idx]
        ts = TILE_SIZES[idx]
        offset_x = (FRAME_SIZE - ts * gs) // 2
        offset_y = (PLAY_ROWS - ts * gs) // 2
        spr = _get_sprites(ts)

        for y in range(gs):
            for x in range(gs):
                cell_type = self._grid[y][x]
                pos = (x, y)
                px = offset_x + x * ts
                py = offset_y + y * ts
                key = self._resolve_sprite_key(cell_type, pos)
                level.add_sprite(spr[key].clone().set_position(px, py))

        cx, cy = self._cursor
        cpx = offset_x + cx * ts
        cpy = offset_y + cy * ts
        level.add_sprite(spr["cursor"].clone().set_position(cpx, cpy))

    def _restart_level(self):
        self._lives -= 1
        if self._lives <= 0:
            self.lose()
            return
        self._restore_init_state()
        self._propagate_power()
        self._render_level(self.current_level)

    def handle_reset(self) -> None:
        if self._action_count == 0:
            self.full_reset()
        else:
            self.level_reset()
            self._action_count = 0
            self._restore_init_state()
            self._propagate_power()
            self._render_level(self.current_level)

    def step(self) -> None:
        if not self.action:
            self.complete_action()
            return

        if self.action.id == GameAction.RESET:
            self.complete_action()
            return

        self._action_count += 1

        if self._game_over:
            self._restart_level()
            self.complete_action()
            return

        action_id = self.action.id

        if action_id == GameAction.ACTION7:
            self._undo()
            self.complete_action()
            return

        dx, dy = 0, 0
        place_wire = False

        if action_id == GameAction.ACTION1:
            dy = -1
        elif action_id == GameAction.ACTION2:
            dy = 1
        elif action_id == GameAction.ACTION3:
            dx = -1
        elif action_id == GameAction.ACTION4:
            dx = 1
        elif action_id == GameAction.ACTION5:
            place_wire = True

        if dx == 0 and dy == 0 and not place_wire:
            self.complete_action()
            return

        self._undo_stack.append(self._snapshot_state())

        self._moves_remaining -= 1
        self._move_display.decrement()
        self._move_count += 1

        if self._gates and self._move_count % 3 == 0:
            self._toggle_gates()

        if place_wire:
            cx, cy = self._cursor
            pos = (cx, cy)
            ct = self._grid[cy][cx]

            if ct == CELL_EMPTY:
                if pos in self._wires:
                    self._wires.discard(pos)
                else:
                    self._wires.add(pos)
                    if pos in self._traps:
                        self._traps_blown.add(pos)
                        self._propagate_power()
                        self._render_level(self.current_level)
                        self._restart_level()
                        self.complete_action()
                        return

            elif ct == CELL_SWITCH:
                cur_st = self._switches.get(pos, False)
                self._switches[pos] = not cur_st
                wp = self._switch_walls.get(pos)
                if wp:
                    if self._switches[pos]:
                        self._grid[wp[1]][wp[0]] = CELL_EMPTY
                    else:
                        self._grid[wp[1]][wp[0]] = CELL_WALL

            self._propagate_power()

            if self._moves_remaining <= 0 and not self._check_win():
                self._restart_level()
                self.complete_action()
                return

            self._render_level(self.current_level)
            if self._check_win():
                self.next_level()
            self.complete_action()
            return

        nx, ny = self._cursor[0] + dx, self._cursor[1] + dy
        if 0 <= nx < self._grid_w and 0 <= ny < self._grid_h:
            ct = self._grid[ny][nx]
            if ct != CELL_WALL and ct != CELL_INSULATOR:
                if ct == CELL_GATE and not self._gate_open.get((nx, ny), True):
                    pass
                else:
                    self._cursor = [nx, ny]

        self._propagate_power()

        if self._moves_remaining <= 0 and not self._check_win():
            self._restart_level()
            self.complete_action()
            return
        self._render_level(self.current_level)
        if self._check_win():
            self.next_level()
        self.complete_action()


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
        self._engine = Cw03(seed=seed)
        self._total_turns = 0
        self._done = False
        self._game_won = False
        self._last_action_was_reset = False
        self._levels_completed = 0

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
        self._levels_completed = 0

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
            self._levels_completed += 1
            info["reason"] = "game_complete"
            return StepResult(
                state=self._build_game_state(done=True),
                reward=level_reward,
                done=True,
                info=info,
            )

        if game_over:
            info["reason"] = "game_over"
            return StepResult(
                state=self._build_game_state(done=False),
                reward=0.0,
                done=False,
                info=info,
            )

        reward = 0.0
        if e.level_index != level_before:
            reward = level_reward
            self._levels_completed += 1
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
        gs = GRID_SIZES[e.level_index]
        lines = []
        lines.append(
            f"Level {e.level_index + 1}/{len(e._levels)} | "
            f"Lives {e._lives} | "
            f"Moves {e._moves_remaining}/{e._max_moves}"
        )
        char_map = {
            CELL_WALL: "#",
            CELL_SOURCE_A: "A",
            CELL_SOURCE_B: "B",
            CELL_RECV_A: "a",
            CELL_RECV_B: "b",
            CELL_INSULATOR: "I",
            CELL_SWITCH: "S",
            CELL_CONDUIT_R: ">",
            CELL_CONDUIT_D: "v",
            CELL_CONDUIT_L: "<",
            CELL_CONDUIT_U: "^",
            CELL_GATE: "G",
            CELL_CAPACITOR: "C",
            CELL_DRAIN: "D",
        }
        for y in range(gs):
            row_str = ""
            for x in range(gs):
                pos = (x, y)
                if [x, y] == e._cursor:
                    row_str += "@"
                elif pos in e._wires:
                    if pos in e._powered_a:
                        row_str += "+"
                    elif pos in e._powered_b:
                        row_str += "="
                    else:
                        row_str += "~"
                elif pos in e._traps_blown:
                    row_str += "X"
                elif pos in e._traps:
                    row_str += "T"
                else:
                    ct = e._grid[y][x]
                    row_str += char_map.get(ct, ".")
            lines.append(row_str)
        return "\n".join(lines)

    def _build_game_state(self, done: bool = False) -> GameState:
        valid_actions = self.get_actions()
        return GameState(
            text_observation=self._build_text_observation(),
            image_observation=None,
            valid_actions=valid_actions,
            turn=self._total_turns,
            metadata={
                "total_levels": len(self._engine._levels),
                "level_index": self._engine.level_index,
                "levels_completed": self._levels_completed,
                "game_over": getattr(self._engine, "_game_over", False),
                "done": done,
                "info": {},
            },
        )


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

    env = ArcGameEnv(seed=0, render_mode="rgb_array")
    check_env(env.unwrapped, skip_render_check=True)

    obs, info = env.reset()
    mask = env.action_mask()
    valid_indices = np.where(mask == 1)[0]
    if len(valid_indices) > 0:
        obs, reward, term, trunc, info = env.step(valid_indices[0])

    env.close()
