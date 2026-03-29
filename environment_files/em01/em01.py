from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import struct
import zlib
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from arcengine import (
    ARCBaseGame,
    ActionInput,
    Camera,
    Level,
    RenderableUserDisplay,
    Sprite,
)
from arcengine.enums import GameState as EngineGameState, GameAction

C_EMPTY = 0
C_NODE = 1
C_BARRIER = 2
C_BLOCK = 3
C_PLAYER = 4
C_WALL = 5
C_EXIT = 6
C_CONV = 7
C_ICE = 8
C_GATE = 9
C_ENERGY = 10
C_OUTER = 11
C_TELEPORT = 12
C_TRAP = 13
C_PLACED = 14
C_SWITCH = 15

BACKGROUND_COLOR = 0
PADDING_COLOR = 0

SOLID_CELLS = {C_WALL, C_OUTER, C_BARRIER, C_BLOCK, C_GATE}

COLOR_NAMES = {
    C_EMPTY: ".",
    C_NODE: "N",
    C_BARRIER: "B",
    C_BLOCK: "K",
    C_PLAYER: "P",
    C_WALL: "W",
    C_EXIT: "X",
    C_CONV: "C",
    C_ICE: "I",
    C_GATE: "G",
    C_ENERGY: "E",
    C_OUTER: "O",
    C_TELEPORT: "T",
    C_TRAP: "V",
    C_PLACED: "S",
    C_SWITCH: "U",
}

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


def _zeros(h: int, w: int) -> List[List[int]]:
    return [[C_EMPTY] * w for _ in range(h)]


def count_placed_blocks(grid: List[List[int]], h: int, w: int) -> int:
    return sum(row.count(C_PLACED) for row in grid)


def all_barriers_cleared(grid: List[List[int]], h: int, w: int) -> bool:
    return sum(row.count(C_BARRIER) for row in grid) == 0


def _make_level_1() -> dict:
    S = 10
    g = _zeros(S, S)
    for i in range(S):
        g[0][i] = C_OUTER
        g[S - 1][i] = C_OUTER
        g[i][0] = C_OUTER
        g[i][S - 1] = C_OUTER

    for r in range(2, 8):
        g[r][2] = C_TRAP
        g[r][7] = C_TRAP

    g[2][4] = C_BARRIER
    g[2][5] = C_BARRIER
    g[5][4] = C_BLOCK
    g[5][5] = C_BLOCK
    g[3][3] = C_NODE
    g[3][6] = C_NODE
    g[1][4] = C_EXIT
    g[1][5] = C_EXIT

    return {
        "grid": g,
        "size": S,
        "player_r": 8,
        "player_c": 4,
        "step_limit": 50,
        "conveyor_dirs": {},
        "switch_order": [],
        "total_blocks": 2,
        "wraparound": False,
        "barrier_map": {"3,3": (2, 4), "3,6": (2, 5)},
        "teleport_links": {},
        "energy_cells": [],
        "fixed_spawn": False,
    }


def _make_level_2() -> dict:
    S = 12
    g = _zeros(S, S)
    for i in range(S):
        g[0][i] = C_OUTER
        g[S - 1][i] = C_OUTER
        g[i][0] = C_OUTER
        g[i][S - 1] = C_OUTER

    for r in range(4, 9):
        for c in range(2, 10):
            g[r][c] = C_ICE

    g[6][2] = C_TRAP
    g[5][9] = C_TRAP
    g[7][9] = C_TRAP
    g[5][5] = C_NODE
    g[5][6] = C_NODE
    g[1][5] = C_BARRIER
    g[1][6] = C_BARRIER
    g[0][5] = C_EXIT
    g[0][6] = C_EXIT
    g[9][3] = C_BLOCK
    g[9][8] = C_BLOCK

    return {
        "grid": g,
        "size": S,
        "player_r": 10,
        "player_c": 5,
        "step_limit": 75,
        "conveyor_dirs": {},
        "switch_order": [],
        "total_blocks": 2,
        "wraparound": False,
        "barrier_map": {"5,5": (1, 5), "5,6": (1, 6)},
        "teleport_links": {},
        "energy_cells": [],
        "fixed_spawn": False,
    }


def _make_level_3() -> dict:
    S = 16
    g = _zeros(S, S)
    for i in range(S):
        g[0][i] = C_OUTER
        g[S - 1][i] = C_OUTER
        g[i][0] = C_OUTER
        g[i][S - 1] = C_OUTER

    conveyor_dirs = {}

    for c in range(5, 11):
        g[5][c] = C_CONV
        conveyor_dirs[(5, c)] = "l"
    for c in range(5, 11):
        g[10][c] = C_CONV
        conveyor_dirs[(10, c)] = "r"
    for r in range(5, 11):
        g[r][5] = C_CONV
        conveyor_dirs[(r, 5)] = "d"
    for r in range(5, 11):
        g[r][10] = C_CONV
        conveyor_dirs[(r, 10)] = "u"

    for c in range(2, 14):
        g[2][c] = C_CONV
        conveyor_dirs[(2, c)] = "r"
    for c in range(2, 14):
        g[13][c] = C_CONV
        conveyor_dirs[(13, c)] = "l"
    for r in range(2, 14):
        g[r][2] = C_CONV
        conveyor_dirs[(r, 2)] = "u"
    for r in range(2, 14):
        g[r][13] = C_CONV
        conveyor_dirs[(r, 13)] = "d"

    g[7][7] = C_BLOCK
    g[7][8] = C_BLOCK
    g[8][7] = C_BLOCK
    g[8][8] = C_BLOCK

    g[1][1] = C_NODE
    g[1][14] = C_NODE
    g[14][1] = C_NODE
    g[14][14] = C_NODE

    g[0][8] = C_EXIT
    g[1][8] = C_BARRIER
    g[2][8] = C_BARRIER

    return {
        "grid": g,
        "size": S,
        "player_r": 8,
        "player_c": 8,
        "step_limit": 150,
        "conveyor_dirs": conveyor_dirs,
        "switch_order": [],
        "total_blocks": 4,
        "wraparound": False,
        "barrier_map": {"1,1": (1, 8), "1,14": (1, 8), "14,1": (2, 8), "14,14": (2, 8)},
        "teleport_links": {},
        "energy_cells": [],
        "fixed_spawn": True,
    }


def _make_level_4() -> dict:
    S = 16
    g = _zeros(S, S)

    for r in range(1, S - 1):
        for c in range(1, S - 1):
            g[r][c] = C_ICE

    for i in range(S):
        g[0][i] = C_WALL
        g[S - 1][i] = C_WALL
        g[i][0] = C_WALL
        g[i][S - 1] = C_WALL

    for r in range(6, 11):
        for c in range(6, 11):
            g[r][c] = C_EMPTY

    g[7][7] = C_BLOCK
    g[7][8] = C_BLOCK
    g[7][9] = C_BLOCK
    g[8][8] = C_BLOCK
    g[9][8] = C_BLOCK

    g[8][1] = C_ICE
    g[8][0] = C_WALL
    g[8][2] = C_NODE
    g[6][2] = C_BLOCK

    g[8][14] = C_ICE
    g[8][15] = C_WALL
    g[8][13] = C_NODE
    g[10][13] = C_BLOCK

    g[1][8] = C_ICE
    g[0][8] = C_WALL
    g[2][8] = C_NODE
    g[2][6] = C_BLOCK

    g[12][8] = C_EXIT
    g[11][7] = C_BARRIER
    g[11][8] = C_BARRIER
    g[11][9] = C_BARRIER

    return {
        "grid": g,
        "size": S,
        "player_r": 8,
        "player_c": 8,
        "step_limit": 200,
        "conveyor_dirs": {},
        "switch_order": [],
        "total_blocks": 3 + 5,
        "wraparound": False,
        "barrier_map": {"8,2": (11, 7), "8,13": (11, 9), "2,8": (11, 8)},
        "teleport_links": {},
        "energy_cells": [],
        "fixed_spawn": False,
    }


def _make_level_5() -> dict:
    S = 16
    g = _zeros(S, S)

    for r in range(S):
        for c in range(S):
            g[r][c] = C_TRAP

    for r in range(2, 6):
        for c in range(2, 6):
            g[r][c] = C_EMPTY
    for r in range(2, 6):
        for c in range(10, 14):
            g[r][c] = C_EMPTY
    for r in range(10, 14):
        for c in range(2, 6):
            g[r][c] = C_EMPTY
    for r in range(10, 14):
        for c in range(10, 14):
            g[r][c] = C_EMPTY

    for i in range(2, 6):
        g[0][i] = C_EMPTY
        g[1][i] = C_EMPTY
        g[i][0] = C_EMPTY
        g[i][1] = C_EMPTY

        g[0][i + 8] = C_EMPTY
        g[1][i + 8] = C_EMPTY
        g[i][15] = C_EMPTY
        g[i][14] = C_EMPTY

        g[15][i] = C_EMPTY
        g[14][i] = C_EMPTY
        g[i + 8][0] = C_EMPTY
        g[i + 8][1] = C_EMPTY

        g[15][i + 8] = C_EMPTY
        g[14][i + 8] = C_EMPTY
        g[i + 8][15] = C_EMPTY
        g[i + 8][14] = C_EMPTY

    g[3][3] = C_BLOCK
    g[11][11] = C_NODE
    g[3][11] = C_BLOCK
    g[11][3] = C_NODE

    g[12][12] = C_EXIT
    g[11][12] = C_BARRIER
    g[12][11] = C_BARRIER

    return {
        "grid": g,
        "size": S,
        "player_r": 4,
        "player_c": 4,
        "step_limit": 100,
        "conveyor_dirs": {},
        "switch_order": [],
        "total_blocks": 2,
        "wraparound": True,
        "barrier_map": {"11,11": (11, 12), "11,3": (12, 11)},
        "teleport_links": {},
        "energy_cells": [],
        "fixed_spawn": False,
    }


LEVEL_CONFIGS = [
    {"name": "The Filter", "size": 10, "make": _make_level_1},
    {"name": "Frictionless Death", "size": 12, "make": _make_level_2},
    {"name": "The Assembly Line", "size": 16, "make": _make_level_3},
    {"name": "Zero-G Warehouse", "size": 16, "make": _make_level_4},
    {"name": "The Tesseract", "size": 16, "make": _make_level_5},
]
_PIXEL_SIZES = [4, 4, 3, 3, 4]

_sprite_cache: dict = {}


def _get_sprite(color: int, px: int) -> Sprite:
    key = (color, px)
    if key not in _sprite_cache:
        is_solid = color in SOLID_CELLS
        tag = "solid" if is_solid else "cell"
        _sprite_cache[key] = Sprite(
            pixels=[[color] * px for _ in range(px)],
            name=f"c{color}_{px}",
            visible=True,
            collidable=is_solid,
            layer=1 if is_solid else (3 if color == C_PLAYER else 2),
            tags=[tag],
        )
    return _sprite_cache[key]


def _grid_to_sprites(grid: List[List[int]], px: int, pr: int, pc: int) -> List[Sprite]:
    result = []
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            if val != C_EMPTY:
                sp = _get_sprite(val, px).clone().set_position(c * px, r * px)
                result.append(sp)
    sp = _get_sprite(C_PLAYER, px).clone().set_position(pc * px, pr * px)
    result.append(sp)
    return result


def _build_levels() -> List[Level]:
    result = []
    for idx, cfg in enumerate(LEVEL_CONFIGS):
        data = cfg["make"]()
        grid = data["grid"]
        px = _PIXEL_SIZES[idx]
        cam_size = min(cfg["size"] * px, 64)
        level = Level(
            sprites=_grid_to_sprites(grid, px, data["player_r"], data["player_c"]),
            grid_size=(cam_size, cam_size),
            data={
                "grid": [row[:] for row in grid],
                "size": cfg["size"],
                "player_r": data["player_r"],
                "player_c": data["player_c"],
                "step_limit": data["step_limit"],
                "px": px,
                "cam_size": cam_size,
                "conveyor_dirs": {
                    f"{k[0]},{k[1]}": v for k, v in data["conveyor_dirs"].items()
                },
                "switch_order": [f"{s[0]},{s[1]}" for s in data["switch_order"]],
                "total_blocks": data["total_blocks"],
                "wraparound": data["wraparound"],
                "barrier_map": data["barrier_map"],
                "teleport_links": {
                    k: f"{v[0]},{v[1]}" for k, v in data["teleport_links"].items()
                },
                "energy_cells": [f"{e[0]},{e[1]}" for e in data["energy_cells"]],
                "fixed_spawn": data.get("fixed_spawn", False),
            },
            name=cfg["name"],
        )
        result.append(level)
    return result


levels = _build_levels()


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


class MatrixHUD(RenderableUserDisplay):
    def __init__(self, total_blocks: int, step_limit: int):
        self.total_blocks = total_blocks
        self.placed_blocks = 0
        self.step_limit = step_limit
        self.steps_used = 0
        self.lives = 3

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        for i in range(self.lives):
            r_off = 1
            c_off = w - 5 - (i * 4)
            if 0 <= c_off < w - 2:
                frame[r_off : r_off + 3, c_off : c_off + 3] = 2
        if self.total_blocks > 0:
            for i in range(self.total_blocks):
                col = 2 + (i * 4)
                color = 14 if i < self.placed_blocks else 1
                if col < w - 2:
                    frame[h - 2, col : col + 2] = color
        remaining = max(0, self.step_limit - self.steps_used)
        bar_max = min(w - 4, 20)
        if self.step_limit > 0:
            fill = max(0, int(bar_max * remaining / self.step_limit))
        else:
            fill = 0
        bar_row = h - 4
        if 0 <= bar_row < h:
            for bc in range(2, 2 + bar_max):
                if bc < w:
                    frame[bar_row, bc] = 14 if (bc - 2) < fill else 2
        return frame


class Em01(ARCBaseGame):
    MAX_LIVES = 3

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self._rng = np.random.RandomState(seed)
        self._grid: List[List[int]] = []
        self._player_r: int = 0
        self._player_c: int = 0
        self._px: int = 3
        self._grid_size: int = 8
        self._conveyor_dirs: Dict[str, str] = {}
        self._switch_order: List[str] = []
        self._switches_activated: int = 0
        self._total_blocks: int = 0
        self._wraparound: bool = False
        self._barrier_map: Dict[str, str] = {}
        self._teleport_links: Dict[str, str] = {}
        self._energy_cells: List[str] = []

        self._step_limit: int = 50
        self._steps_used: int = 0
        self._lives: int = self.MAX_LIVES
        self._died_this_frame: bool = False
        self._undo_stack: List[dict] = []

        first_data = LEVEL_CONFIGS[0]["make"]()
        self._hud = MatrixHUD(first_data["total_blocks"], first_data["step_limit"])
        self._hud.lives = self._lives

        first_cam_size = min(LEVEL_CONFIGS[0]["size"] * _PIXEL_SIZES[0], 64)
        cam = Camera(
            x=0,
            y=0,
            width=first_cam_size,
            height=first_cam_size,
            background=BACKGROUND_COLOR,
            letter_box=PADDING_COLOR,
            interfaces=[self._hud],
        )
        super().__init__(
            "em01",
            levels,
            cam,
            available_actions=[0, 1, 2, 3, 4, 5, 7],
        )

    def on_set_level(self, level: Level) -> None:
        self._load_level_data()
        self._lives = self.MAX_LIVES
        self._hud.lives = self._lives
        self._rebuild_sprites()

    def _load_level_data(self, level=None):
        lv = level if level is not None else self.current_level
        grid_data = lv.get_data("grid")
        self._grid = [row[:] for row in grid_data]
        self._px = lv.get_data("px")
        self._grid_size = lv.get_data("size")
        self._total_blocks = lv.get_data("total_blocks")
        self._wraparound = lv.get_data("wraparound")

        default_r = lv.get_data("player_r")
        default_c = lv.get_data("player_c")
        fixed_spawn = lv.get_data("fixed_spawn")

        if fixed_spawn:
            self._player_r = default_r
            self._player_c = default_c
        else:
            empty_cells = []
            for r in range(1, self._grid_size - 1):
                for c in range(1, self._grid_size - 1):
                    if self._grid[r][c] == C_EMPTY:
                        empty_cells.append((r, c))
            if empty_cells:
                idx = self._rng.randint(0, len(empty_cells))
                spawn = empty_cells[idx]
                self._player_r, self._player_c = spawn[0], spawn[1]
            else:
                self._player_r = default_r
                self._player_c = default_c

        self._step_limit = lv.get_data("step_limit")
        self._steps_used = 0
        self._switches_activated = 0
        self._died_this_frame = False
        self._undo_stack = []

        self._conveyor_dirs = lv.get_data("conveyor_dirs")
        self._switch_order = lv.get_data("switch_order")

        raw_bmap = lv.get_data("barrier_map")
        self._barrier_map = {}
        for nk, bp in raw_bmap.items():
            self._barrier_map[nk] = (
                f"{bp[0]},{bp[1]}" if isinstance(bp, tuple) else str(bp)
            )

        self._teleport_links = lv.get_data("teleport_links")
        self._energy_cells = list(lv.get_data("energy_cells"))

        self._hud.total_blocks = self._total_blocks
        self._hud.placed_blocks = 0
        self._hud.step_limit = self._step_limit
        self._hud.steps_used = 0
        self._hud.lives = self._lives

        cam_size = lv.get_data("cam_size")
        self.camera.width = cam_size
        self.camera.height = cam_size

    def _handle_death(self):
        self._lives -= 1
        self._died_this_frame = True

        if self._lives <= 0:
            self._hud.lives = 0
            self.lose()
        else:
            self._hud.lives = self._lives
            self._load_level_data()
            self._rebuild_sprites()

    def _rebuild_sprites(self) -> None:
        self.current_level.remove_all_sprites()
        px = self._px
        for r, row in enumerate(self._grid):
            for c, val in enumerate(row):
                if val != C_EMPTY:
                    sp = _get_sprite(val, px).clone().set_position(c * px, r * px)
                    self.current_level.add_sprite(sp)
        sp = (
            _get_sprite(C_PLAYER, px)
            .clone()
            .set_position(self._player_c * px, self._player_r * px)
        )
        self.current_level.add_sprite(sp)

    def _remove_linked_barrier(self, r, c):
        nk = f"{r},{c}"
        if nk in self._barrier_map:
            bk = self._barrier_map[nk]
            br, bc = map(int, bk.split(","))
            if self._grid[br][bc] == C_BARRIER:
                self._grid[br][bc] = C_EMPTY

    def _try_push_block(self, br, bc, dr, dc):
        h, w = self._grid_size, self._grid_size
        nr, nc = br + dr, bc + dc

        if self._wraparound:
            nr, nc = nr % h, nc % w
        elif not (0 <= nr < h and 0 <= nc < w):
            return False

        tgt = self._grid[nr][nc]
        if tgt == C_TRAP:
            self._grid[br][bc] = C_EMPTY
            return True

        if tgt in (C_EMPTY, C_ICE, C_CONV, C_ENERGY, C_TELEPORT):
            self._grid[br][bc], self._grid[nr][nc] = C_EMPTY, C_BLOCK
            return True
        elif tgt == C_NODE:
            self._grid[br][bc], self._grid[nr][nc] = C_EMPTY, C_PLACED
            self._remove_linked_barrier(nr, nc)
            self._hud.placed_blocks = count_placed_blocks(self._grid, h, w)
            return True
        return False

    def _move_player(self, dr: int, dc: int) -> None:
        h, w = self._grid_size, self._grid_size
        nr, nc = self._player_r + dr, self._player_c + dc

        if self._wraparound:
            nr, nc = nr % h, nc % w
        elif not (0 <= nr < h and 0 <= nc < w):
            return

        target = self._grid[nr][nc]

        if target == C_TRAP:
            self._handle_death()
            return

        if target == C_BLOCK:
            if not self._try_push_block(nr, nc, dr, dc):
                return
            if self._grid[nr][nc] == C_BLOCK:
                return

        if self._grid[nr][nc] in SOLID_CELLS:
            return

        self._player_r, self._player_c = nr, nc

        pk = f"{nr},{nc}"
        if pk in self._energy_cells:
            self._energy_cells.remove(pk)
            self._step_limit += 10
            if self._grid[nr][nc] == C_ENERGY:
                self._grid[nr][nc] = C_EMPTY

        if self._grid[nr][nc] == C_ICE:
            self._ice_slide(dr, dc)

        self._apply_conveyor(0)

    def _ice_slide(self, dr, dc):
        h, w = self._grid_size, self._grid_size
        for _ in range(h * w):
            nr, nc = self._player_r + dr, self._player_c + dc
            if self._wraparound:
                nr, nc = nr % h, nc % w
            elif not (0 <= nr < h and 0 <= nc < w):
                break

            tgt = self._grid[nr][nc]
            if tgt == C_TRAP:
                self._handle_death()
                return
            if tgt in SOLID_CELLS:
                break

            self._player_r, self._player_c = nr, nc
            if tgt != C_ICE:
                break

    def _apply_conveyor(self, depth):
        if depth > 10:
            return
        key = f"{self._player_r},{self._player_c}"
        if key not in self._conveyor_dirs:
            return

        d = self._conveyor_dirs[key]
        dr, dc = {"u": (-1, 0), "d": (1, 0), "l": (0, -1), "r": (0, 1)}.get(
            d[0], (0, 0)
        )
        self._move_player(dr, dc)

    def _try_activate_switch(self):
        pr, pc = self._player_r, self._player_c
        if self._grid[pr][pc] != C_SWITCH or not self._switch_order:
            return

        pk = f"{pr},{pc}"
        if self._switches_activated < len(self._switch_order):
            if pk == self._switch_order[self._switches_activated]:
                self._switches_activated += 1
                self._grid[pr][pc] = C_PLACED

                if self._switches_activated == len(self._switch_order):
                    for r in range(self._grid_size):
                        for c in range(self._grid_size):
                            if self._grid[r][c] == C_GATE:
                                self._grid[r][c] = C_EMPTY
            else:
                self._switches_activated = 0
                grid_data = self.current_level.get_data("grid")
                for r in range(self._grid_size):
                    for c in range(self._grid_size):
                        if self._grid[r][c] == C_PLACED and grid_data[r][c] == C_SWITCH:
                            self._grid[r][c] = C_SWITCH
                        if grid_data[r][c] == C_GATE and self._grid[r][c] == C_EMPTY:
                            self._grid[r][c] = C_GATE

    def _try_teleport(self):
        pk = f"{self._player_r},{self._player_c}"
        if pk in self._teleport_links:
            dest = self._teleport_links[pk]
            dr, dc = map(int, dest.split(","))
            self._player_r, self._player_c = dr, dc
            return True
        return False

    def _save_undo_state(self) -> None:
        snapshot = {
            "grid": [row[:] for row in self._grid],
            "player_r": self._player_r,
            "player_c": self._player_c,
            "switches_activated": self._switches_activated,
            "energy_cells": list(self._energy_cells),
            "step_limit": self._step_limit,
        }
        self._undo_stack.append(snapshot)
        if len(self._undo_stack) > 50:
            self._undo_stack.pop(0)

    def _apply_undo(self) -> bool:
        if not self._undo_stack:
            return False
        snapshot = self._undo_stack.pop()
        self._grid = snapshot["grid"]
        self._player_r = snapshot["player_r"]
        self._player_c = snapshot["player_c"]
        self._switches_activated = snapshot["switches_activated"]
        self._energy_cells = snapshot["energy_cells"]
        self._step_limit = snapshot["step_limit"]
        self._hud.placed_blocks = count_placed_blocks(
            self._grid, self._grid_size, self._grid_size
        )
        self._hud.step_limit = self._step_limit
        self._hud.lives = self._lives
        self._rebuild_sprites()
        return True

    def step(self) -> None:
        self._died_this_frame = False
        moved = False

        try:
            aid = self.action.id
            if hasattr(aid, "value"):
                aid = aid.value
            aid = int(aid)
        except (ValueError, TypeError, AttributeError):
            aid = 0

        if aid in (1, 2, 3, 4, 5):
            self._save_undo_state()

        if aid == 1:
            self._move_player(-1, 0)
            moved = True
        elif aid == 2:
            self._move_player(1, 0)
            moved = True
        elif aid == 3:
            self._move_player(0, -1)
            moved = True
        elif aid == 4:
            self._move_player(0, 1)
            moved = True
        elif aid == 5:
            c = self._grid[self._player_r][self._player_c]
            if c == C_SWITCH:
                self._try_activate_switch()
            elif c == C_TELEPORT:
                self._try_teleport()
            moved = True

        elif aid == 7:
            if self._apply_undo():
                self._steps_used += 1
                self._hud.steps_used = self._steps_used
                if self._steps_used >= self._step_limit:
                    self._handle_death()

        if moved and not self._died_this_frame:
            self._steps_used += 1
            self._hud.steps_used = self._steps_used
            if self._steps_used >= self._step_limit:
                self._handle_death()
            self._rebuild_sprites()

        if not self._died_this_frame:
            if self._grid[self._player_r][
                self._player_c
            ] == C_EXIT and all_barriers_cleared(
                self._grid, self._grid_size, self._grid_size
            ):
                self.next_level()

        self.complete_action()

    def render_text(self) -> str:
        rows = []
        for r in range(self._grid_size):
            row_chars = []
            for c in range(self._grid_size):
                if r == self._player_r and c == self._player_c:
                    row_chars.append(COLOR_NAMES[C_PLAYER])
                else:
                    row_chars.append(COLOR_NAMES.get(self._grid[r][c], "?"))
            rows.append(" ".join(row_chars))
        lv_num = self.level_index + 1
        lv_name = self.current_level.name or ""
        placed = count_placed_blocks(self._grid, self._grid_size, self._grid_size)
        header = f"Level:{lv_num} {lv_name} Lives:{self._lives} Moves:{self._steps_used}/{self._step_limit} Blocks:{placed}/{self._total_blocks}"
        extra = []
        if self._wraparound:
            extra.append("Wraparound: ON (edges connect)")
        if self._energy_cells:
            extra.append(f"EnergyCells: {' '.join(f'({e})' for e in self._energy_cells)}")
        if self._switch_order:
            remaining = self._switch_order[self._switches_activated:]
            extra.append(f"Switches: {self._switches_activated}/{len(self._switch_order)} activated, next={','.join(remaining) if remaining else 'done'}")
        if self._conveyor_dirs:
            dirs = {v for v in self._conveyor_dirs.values()}
            conv_summary = ", ".join(
                f"({k}) dir={v}" for k, v in sorted(self._conveyor_dirs.items())
            )
            extra.append(f"Conveyors[{','.join(sorted(dirs))}]: {conv_summary}")
        if self._barrier_map:
            links = ", ".join(
                f"node({k})->barrier({v})" for k, v in sorted(self._barrier_map.items())
            )
            extra.append(f"NodeBarrierLinks: {links}")
        if self._teleport_links:
            pairs = ", ".join(
                f"({k})->({v})" for k, v in sorted(self._teleport_links.items())
            )
            extra.append(f"Teleports: {pairs}")
        ice_cells = []
        for r in range(self._grid_size):
            for c in range(self._grid_size):
                if self._grid[r][c] == C_ICE:
                    ice_cells.append(f"({r},{c})")
        if ice_cells:
            extra.append(f"Ice: {' '.join(ice_cells)} (sliding)")
        legend = "\n".join(extra)
        result = header + "\n" + "\n".join(rows)
        if legend:
            result += "\n" + legend
        return result


_ACTION_MAP: Dict[str, GameAction] = {
    "up": GameAction.ACTION1,
    "down": GameAction.ACTION2,
    "left": GameAction.ACTION3,
    "right": GameAction.ACTION4,
    "select": GameAction.ACTION5,
    "undo": GameAction.ACTION7,
    "reset": GameAction.RESET,
}

_VALID_ACTIONS = ["reset", "up", "down", "left", "right", "select", "undo"]


class PuzzleEnvironment:
    def __init__(self, seed: int = 0) -> None:
        self._engine = Em01(seed=seed)
        self._total_turns: int = 0
        self._done: bool = False
        self._last_action_was_reset: bool = False

    def _is_won(self) -> bool:
        return self._engine._state == EngineGameState.WIN

    def _is_game_over(self) -> bool:
        return self._engine._state == EngineGameState.GAME_OVER

    def _build_text(self) -> str:
        return self._engine.render_text()

    @staticmethod
    def _encode_png(arr: np.ndarray) -> bytes:
        if arr.ndim == 2:
            h, w = arr.shape
            channels = 1
            color_type = 0
        else:
            h, w, channels = arr.shape
            color_type = 2
        raw = bytearray()
        for row in arr:
            raw.append(0)
            raw.extend(row.tobytes())
        compressed = zlib.compress(bytes(raw))
        def _chunk(ctype: bytes, data: bytes) -> bytes:
            c = ctype + data
            return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        ihdr = struct.pack(">IIBBBBB", w, h, 8, color_type, 0, 0, 0)
        out = b"\x89PNG\r\n\x1a\n"
        out += _chunk(b"IHDR", ihdr)
        out += _chunk(b"IDAT", compressed)
        out += _chunk(b"IEND", b"")
        return out

    def _build_image(self) -> Optional[bytes]:
        try:
            frame = self._engine.camera.render(self._engine.current_level.get_sprites())
            return self._encode_png(frame.astype(np.uint8))
        except Exception:
            return None

    def _build_state(self) -> GameState:
        text = self._build_text()
        img = self._build_image()
        valid = None if self._done else self.get_actions()
        return GameState(
            text_observation=text,
            image_observation=img,
            valid_actions=valid,
            turn=self._total_turns,
            metadata={"total_levels": len(self._engine._levels)},
        )

    def reset(self) -> GameState:
        self._total_turns = 0
        self._done = False
        if self._is_won() or self._last_action_was_reset:
            self._engine.full_reset()
        else:
            self._engine.perform_action(ActionInput(id=GameAction.RESET), raw=True)
        self._last_action_was_reset = True
        return self._build_state()

    def get_actions(self) -> List[str]:
        if self._done:
            return ["reset"]
        return list(_VALID_ACTIONS)

    def step(self, action: str) -> StepResult:
        if action == "reset":
            state = self.reset()
            return StepResult(
                state=state,
                reward=0.0,
                done=False,
            )

        self._last_action_was_reset = False

        ga = _ACTION_MAP.get(action)
        if ga is None:
            return StepResult(
                state=self._build_state(),
                reward=0.0,
                done=self._done,
                info={"error": "invalid_action"},
            )

        level_before = self._engine.level_index
        self._engine.perform_action(ActionInput(id=ga), raw=True)
        self._total_turns += 1

        total_levels = len(self._engine._levels)
        reward_per_level = 1.0 / total_levels

        game_won = self._is_won()

        if game_won:
            self._done = True
            return StepResult(
                state=self._build_state(),
                reward=reward_per_level,
                done=True,
                info={"reason": "game_complete"},
            )

        if self._is_game_over():
            self._done = True
            return StepResult(
                state=self._build_state(),
                reward=0.0,
                done=True,
                info={"reason": "death"},
            )

        level_after = self._engine.level_index
        if level_after != level_before:
            return StepResult(
                state=self._build_state(),
                reward=reward_per_level,
                done=False,
            )

        return StepResult(
            state=self._build_state(),
            reward=0.0,
            done=False,
        )

    def is_done(self) -> bool:
        return self._done

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError(f"Unsupported render mode: {mode}")
        index_grid = self._engine.camera.render(
            self._engine.current_level.get_sprites()
        )
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        for idx, color in enumerate(ARC_PALETTE):
            mask = index_grid == idx
            rgb[mask] = color
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
