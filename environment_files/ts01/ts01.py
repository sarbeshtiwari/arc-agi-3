from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import struct
import zlib
from random import Random

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
    metadata: Dict = field(default_factory=dict)


@dataclass
class StepResult:
    state: GameState
    reward: float
    done: bool
    info: Dict = field(default_factory=dict)


BG = 0

WALL = 10

ZONE = 11

GATE_OFF = 3

GATE_ON = 9

PLAYER = 14

FLASH = 8

LIFE_ON = 14

LIFE_OFF = 4

LBOX = 5

_C = -1


def _shape_offsets(shape):
    offsets = []
    for r, row in enumerate(shape):
        for c, v in enumerate(row):
            if v >= 0:
                offsets.append((c, r))
    return offsets


def _zone_cells(zones):
    cells = set()
    for z in zones:
        for dy in range(z["h"]):
            for dx in range(z["w"]):
                cells.add((z["x"] + dx, z["y"] + dy))
    return cells


LEVELS = [
    {
        "player_start": (15, 15),
        "gate_pos": (28, 2),
        "gate_size": (2, 3),
        "zones": [{"x": 1, "y": 1, "w": 5, "h": 6}],
        "pieces": [
            {"shape": [[9, 9]], "sx": 10, "sy": 8, "color": 9},
            {"shape": [[12, 12]], "sx": 24, "sy": 20, "color": 12},
            {
                "shape": [[15, 15, 15, 15], [15, _C, _C, _C]],
                "sx": 18,
                "sy": 6,
                "color": 15,
            },
            {
                "shape": [[6, 6, 6], [6, 6, 6], [6, 6, 6]],
                "sx": 22,
                "sy": 22,
                "color": 6,
            },
            {"shape": [[7, 7], [7, 7], [7, 7], [7, 7]], "sx": 6, "sy": 22, "color": 7},
            {"shape": [[_C, 8], [8, 8], [_C, 8]], "sx": 26, "sy": 10, "color": 8},
        ],
    },
    {
        "player_start": (15, 15),
        "gate_pos": (28, 14),
        "gate_size": (2, 3),
        "zones": [{"x": 1, "y": 1, "w": 5, "h": 8}],
        "pieces": [
            {
                "shape": [[9, _C, _C], [9, 9, _C], [9, _C, _C], [9, 9, _C]],
                "sx": 20,
                "sy": 3,
                "color": 9,
            },
            {
                "shape": [
                    [_C, _C, 12],
                    [_C, _C, 12],
                    [_C, 12, 12],
                    [_C, _C, 12],
                    [12, 12, 12],
                ],
                "sx": 9,
                "sy": 3,
                "color": 12,
            },
            {"shape": [[15, 15], [15, 15]], "sx": 24, "sy": 8, "color": 15},
            {"shape": [[6, 6, 6], [_C, 6, _C]], "sx": 18, "sy": 13, "color": 6},
            {"shape": [[7, 7, 7, 7, 7]], "sx": 8, "sy": 20, "color": 7},
            {
                "shape": [[8, 8, 8, 8, 8], [8, 8, _C, _C, _C]],
                "sx": 9,
                "sy": 24,
                "color": 8,
            },
            {"shape": [[9, 9], [9, 9], [9, 9]], "sx": 24, "sy": 20, "color": 9},
            {"shape": [[13, 13], [13, 13]], "sx": 5, "sy": 9, "color": 13},
            {"shape": [[1, 1]], "sx": 18, "sy": 18, "color": 1},
            {"shape": [[2, _C], [2, 2]], "sx": 22, "sy": 25, "color": 2},
            {
                "shape": [[13, _C, _C], [13, 13, _C], [13, _C, _C], [13, 13, _C]],
                "sx": 14,
                "sy": 3,
                "color": 13,
            },
            {"shape": [[1, 1], [1, 1]], "sx": 26, "sy": 3, "color": 1},
            {"shape": [[2, 2, 2], [_C, 2, _C]], "sx": 13, "sy": 10, "color": 2},
            {
                "shape": [[13, 13, 13, 13, 13], [13, 13, _C, _C, _C]],
                "sx": 14,
                "sy": 26,
                "color": 13,
            },
            {"shape": [[1, 1], [1, 1], [1, 1]], "sx": 15, "sy": 21, "color": 1},
            {"shape": [[2, 2, 2, 2, 2]], "sx": 3, "sy": 16, "color": 2},
            {
                "shape": [[_C, _C, 13], [_C, 13, 13], [_C, _C, 13], [13, 13, 13]],
                "sx": 26,
                "sy": 20,
                "color": 13,
            },
            {"shape": [[1], [1]], "sx": 7, "sy": 12, "color": 1},
        ],
    },
    {
        "player_start": (15, 15),
        "gate_pos": (14, 29),
        "gate_size": (2, 2),
        "fake_gates": [
            {"x": 5, "y": 29, "w": 2, "h": 2},
            {"x": 26, "y": 29, "w": 2, "h": 2},
        ],
        "zones": [{"x": 23, "y": 1, "w": 8, "h": 8}],
        "pieces": [
            {"shape": [[_C, 9, _C], [9, 9, 9]], "sx": 2, "sy": 4, "color": 9},
            {"shape": [[12, 12, 12, 12, 12, 12, 12]], "sx": 3, "sy": 20, "color": 12},
            {"shape": [[15, 15, 15], [15, _C, _C]], "sx": 4, "sy": 13, "color": 15},
            {"shape": [[6, 6, 6], [6, _C, _C]], "sx": 17, "sy": 4, "color": 6},
            {"shape": [[7, 7, 7, 7]], "sx": 14, "sy": 14, "color": 7},
            {
                "shape": [
                    [8, _C, _C, _C],
                    [8, _C, _C, _C],
                    [8, _C, _C, _C],
                    [8, _C, _C, _C],
                    [8, 8, 8, 8],
                ],
                "sx": 10,
                "sy": 22,
                "color": 8,
            },
            {
                "shape": [
                    [9, 9, 9, 9],
                    [9, 9, 9, 9],
                    [9, 9, 9, 9],
                    [9, 9, 9, 9],
                    [9, 9, 9, 9],
                ],
                "sx": 19,
                "sy": 19,
                "color": 9,
            },
            {"shape": [[12, 12, 12], [12, 12, 12]], "sx": 3, "sy": 27, "color": 12},
            {"shape": [[15, 15, 15], [15, 15, 15]], "sx": 18, "sy": 12, "color": 15},
            {"shape": [[15, 15, 15], [_C, 15, 15]], "sx": 14, "sy": 8, "color": 15},
            {
                "shape": [[_C, 13, _C], [13, 13, 13], [_C, 13, _C]],
                "sx": 8,
                "sy": 9,
                "color": 13,
            },
            {"shape": [[1, 1, 1], [_C, 1, _C]], "sx": 19, "sy": 9, "color": 1},
            {"shape": [[2, _C], [2, _C], [2, 2]], "sx": 9, "sy": 17, "color": 2},
            {"shape": [[13, 13], [13, 13]], "sx": 14, "sy": 25, "color": 13},
            {"shape": [[_C, 1, 1], [1, 1, _C]], "sx": 24, "sy": 14, "color": 1},
            {"shape": [[2, 2, 2]], "sx": 7, "sy": 25, "color": 2},
            {
                "shape": [[13, _C], [13, 13], [13, _C], [13, 13]],
                "sx": 16,
                "sy": 18,
                "color": 13,
            },
        ],
    },
    {
        "player_start": (15, 15),
        "gate_pos": (1, 14),
        "gate_size": (2, 3),
        "fake_gates": [
            {"x": 5, "y": 29, "w": 2, "h": 2},
        ],
        "zones": [
            {"x": 22, "y": 19, "w": 5, "h": 5},
            {"x": 18, "y": 24, "w": 9, "h": 5},
        ],
        "pieces": [
            {
                "shape": [
                    [_C, _C, 9, _C, _C],
                    [_C, _C, 9, _C, _C],
                    [9, 9, 9, 9, 9],
                    [_C, _C, 9, _C, _C],
                    [_C, _C, 9, _C, _C],
                ],
                "sx": 3,
                "sy": 2,
                "color": 9,
            },
            {
                "shape": [
                    [12, 12, _C, _C, _C],
                    [_C, 12, 12, _C, _C],
                    [_C, _C, 12, 12, _C],
                    [_C, _C, _C, 12, 12],
                ],
                "sx": 9,
                "sy": 2,
                "color": 12,
            },
            {"shape": [[15, 15], [_C, 15]], "sx": 23, "sy": 2, "color": 15},
            {"shape": [[6]], "sx": 20, "sy": 8, "color": 6},
            {
                "shape": [[7, _C, _C, _C], [7, 7, _C, _C], [7, 7, 7, _C], [7, 7, 7, 7]],
                "sx": 25,
                "sy": 5,
                "color": 7,
            },
            {"shape": [[_C, 8], [8, 8]], "sx": 8, "sy": 10, "color": 8},
            {"shape": [[9, 9, 9, 9], [9, 9, 9, 9]], "sx": 3, "sy": 11, "color": 9},
            {
                "shape": [[12, 12, _C, _C], [12, 12, 12, 12]],
                "sx": 3,
                "sy": 17,
                "color": 12,
            },
            {"shape": [[15, 15], [15, 15]], "sx": 10, "sy": 14, "color": 15},
            {"shape": [[6, 6, 6, 6], [6, 6, 6, 6]], "sx": 24, "sy": 14, "color": 6},
            {
                "shape": [[7, _C, _C], [7, 7, _C], [7, 7, 7]],
                "sx": 9,
                "sy": 17,
                "color": 7,
            },
            {"shape": [[8, 8], [8, 8]], "sx": 27, "sy": 17, "color": 8},
            {"shape": [[9, 9]], "sx": 21, "sy": 11, "color": 9},
            {"shape": [[12, 12], [_C, 12]], "sx": 24, "sy": 10, "color": 12},
            {"shape": [[13, 13, _C], [_C, 13, 13]], "sx": 8, "sy": 6, "color": 13},
            {"shape": [[_C, 1], [1, 1], [_C, 1]], "sx": 27, "sy": 10, "color": 1},
            {"shape": [[_C, 2, 2], [2, 2, _C]], "sx": 20, "sy": 17, "color": 2},
            {
                "shape": [[_C, 13, _C], [13, 13, 13], [_C, 13, _C]],
                "sx": 6,
                "sy": 14,
                "color": 13,
            },
            {
                "shape": [[1, 1, _C], [1, _C, _C], [1, 1, _C]],
                "sx": 20,
                "sy": 5,
                "color": 1,
            },
            {"shape": [[_C, 2], [_C, 2], [2, 2]], "sx": 27, "sy": 3, "color": 2},
            {"shape": [[13], [13], [13]], "sx": 13, "sy": 9, "color": 13},
            {"shape": [[1, 1]], "sx": 13, "sy": 14, "color": 1},
        ],
    },
    {
        "player_start": (15, 15),
        "gate_pos": (1, 15),
        "gate_size": (4, 1),
        "fake_gates": [
            {"x": 10, "y": 15, "w": 4, "h": 1},
            {"x": 20, "y": 15, "w": 4, "h": 1},
        ],
        "zone_order": [[0], [1, 2]],
        "zones": [
            {"x": 1, "y": 8, "w": 6, "h": 7},
            {"x": 1, "y": 16, "w": 4, "h": 3},
            {"x": 1, "y": 19, "w": 2, "h": 3},
        ],
        "pieces": [
            {
                "shape": [
                    [9, 9, 9],
                    [9, _C, _C],
                    [9, 9, _C],
                    [9, _C, _C],
                    [9, 9, 9],
                    [9, _C, _C],
                ],
                "sx": 20,
                "sy": 2,
                "color": 9,
            },
            {"shape": [[12, 12], [12, _C], [12, 12]], "sx": 8, "sy": 2, "color": 12},
            {"shape": [[15, 15], [15, 15], [15, 15]], "sx": 25, "sy": 8, "color": 15},
            {"shape": [[6, 6], [_C, 6], [6, 6]], "sx": 14, "sy": 2, "color": 6},
            {"shape": [[7]], "sx": 11, "sy": 9, "color": 7},
            {"shape": [[8, 8, 8, 8]], "sx": 17, "sy": 12, "color": 8},
            {"shape": [[9], [9], [9]], "sx": 28, "sy": 4, "color": 9},
            {
                "shape": [
                    [12, 12, 12, 12],
                    [12, _C, _C, _C],
                    [12, _C, _C, _C],
                    [12, _C, _C, _C],
                ],
                "sx": 22,
                "sy": 16,
                "color": 12,
            },
            {"shape": [[_C, 15], [15, 15]], "sx": 9, "sy": 13, "color": 15},
            {"shape": [[_C, 6], [6, 6], [6, _C]], "sx": 12, "sy": 19, "color": 6},
            {
                "shape": [[_C, _C, _C, 7], [_C, 7, 7, _C], [7, _C, _C, _C]],
                "sx": 17,
                "sy": 22,
                "color": 7,
            },
            {"shape": [[8]], "sx": 25, "sy": 14, "color": 8},
            {"shape": [[9], [9]], "sx": 8, "sy": 22, "color": 9},
            {"shape": [[12]], "sx": 20, "sy": 26, "color": 12},
            {"shape": [[_C, 15], [15, 15]], "sx": 26, "sy": 22, "color": 15},
            {
                "shape": [[_C, 13, _C], [13, 13, 13], [_C, 13, _C]],
                "sx": 10,
                "sy": 5,
                "color": 13,
            },
            {"shape": [[_C, 1, 1], [1, 1, _C]], "sx": 17, "sy": 7, "color": 1},
            {"shape": [[2, 2, _C], [_C, 2, 2]], "sx": 24, "sy": 12, "color": 2},
            {"shape": [[13, _C], [13, 13], [13, _C]], "sx": 7, "sy": 17, "color": 13},
            {
                "shape": [[1, 1], [1, _C], [1, 1], [1, _C]],
                "sx": 13,
                "sy": 14,
                "color": 1,
            },
            {
                "shape": [[2, _C, _C], [2, 2, _C], [_C, 2, 2]],
                "sx": 19,
                "sy": 17,
                "color": 2,
            },
            {"shape": [[13], [13], [13], [13]], "sx": 28, "sy": 20, "color": 13},
            {"shape": [[1, 1, 1]], "sx": 10, "sy": 25, "color": 1},
            {"shape": [[_C, 2, 2], [2, 2, _C]], "sx": 14, "sy": 26, "color": 2},
            {"shape": [[13, _C], [13, _C], [13, 13]], "sx": 20, "sy": 22, "color": 13},
            {"shape": [[1, 1], [1, 1]], "sx": 26, "sy": 26, "color": 1},
            {"shape": [[2, 2, 2], [_C, 2, _C]], "sx": 8, "sy": 28, "color": 2},
        ],
    },
]


class GameHUD(RenderableUserDisplay):
    def __init__(self):
        self.lives = 3
        self.max_lives = 3
        self.flash_frames = 0

    @staticmethod
    def _px(frame, x, y, color):
        if 0 <= x < 64 and 0 <= y < 64:
            frame[y, x] = color

    def render_interface(self, frame):
        self._draw_lives(frame)
        if self.flash_frames > 0:
            self._draw_flash(frame)
            self.flash_frames -= 1
        return frame

    def _draw_lives(self, frame):
        for i in range(self.max_lives):
            color = LIFE_ON if i < self.lives else LIFE_OFF
            x = 54 + i * 3
            self._px(frame, x, 60, color)
            self._px(frame, x + 1, 60, color)
            self._px(frame, x, 61, color)
            self._px(frame, x + 1, 61, color)

    def _draw_flash(self, frame):
        for i in range(64):
            self._px(frame, i, 0, FLASH)
            self._px(frame, i, 63, FLASH)
            self._px(frame, 0, i, FLASH)
            self._px(frame, 63, i, FLASH)


class Ts01(ARCBaseGame):
    GW, GH = 32, 32

    def __init__(self, seed: int = 0) -> None:
        self._rng = Random(seed)
        self._seed = seed
        self.hud = GameHUD()

        self.lives = 3
        self.gate_active = False
        self._game_over = False
        self._full_reset_requested = False
        self._undo_stack = []
        self._won = False
        self._consecutive_resets = 0

        self.player_x = 15
        self.player_y = 15
        self.initial_player = (15, 15)

        self.pieces = {}
        self.initial_positions = {}
        self.wall_set = set()
        self.zone_set = set()
        self.zone_boundary = set()
        self.zone_interior = set()
        self.gate_cells = set()
        self.fake_gate_cells = set()
        self.all_gate_cells = set()
        self.safe_sides = set()
        self.board_sprite = None

        levels = [
            Level(sprites=[], grid_size=(self.GW, self.GH), name=f"Level {i + 1}")
            for i in range(len(LEVELS))
        ]

        camera = Camera(
            background=BG,
            letter_box=LBOX,
            width=self.GW,
            height=self.GH,
            interfaces=[self.hud],
        )

        super().__init__(
            "ts01", levels, camera, available_actions=[0, 1, 2, 3, 4, 7], seed=seed
        )

    def _piece_cells(self, pid):
        p = self.pieces[pid]
        px, py = p["pos"]
        return [(px + dx, py + dy) for dx, dy in p["offsets"]]

    def _piece_cell_set(self, pid):
        p = self.pieces[pid]
        px, py = p["pos"]
        return {(px + dx, py + dy) for dx, dy in p["offsets"]}

    def _find_piece_at(self, x, y):
        for pid, p in self.pieces.items():
            px, py = p["pos"]
            if (x - px, y - py) in p["offset_set"]:
                return pid
        return None

    def handle_reset(self):
        if self._game_over or self._full_reset_requested or self._won:
            self._full_reset_requested = False
            self.lives = 3
            self._game_over = False
            self._won = False
            self.full_reset()
        else:
            self._full_reset_requested = True
            self._level_reset()

    def _level_reset(self):
        for pid in self.pieces:
            self.pieces[pid]["pos"] = self.initial_positions[pid]
        self.gate_active = False
        self._undo_stack = []
        self.player_x, self.player_y = self.initial_player
        self._rebuild_board()
        self._sync_hud()

    def on_set_level(self, level):
        cfg = LEVELS[self.level_index]

        self.player_x, self.player_y = cfg["player_start"]
        self.initial_player = cfg["player_start"]
        gx, gy = cfg["gate_pos"]
        gw, gh = cfg.get("gate_size", (2, 3))
        self.gate_cells = {(gx + dx, gy + dy) for dx in range(gw) for dy in range(gh)}

        self.fake_gate_cells = set()
        for fg in cfg.get("fake_gates", []):
            fgw, fgh = fg.get("w", 2), fg.get("h", 2)
            for fdx in range(fgw):
                for fdy in range(fgh):
                    self.fake_gate_cells.add((fg["x"] + fdx, fg["y"] + fdy))
        self.all_gate_cells = self.gate_cells | self.fake_gate_cells

        self.wall_set = set()
        for x in range(self.GW):
            self.wall_set.add((x, 0))
            self.wall_set.add((x, self.GH - 1))
        for y in range(self.GH):
            self.wall_set.add((0, y))
            self.wall_set.add((self.GW - 1, y))

        self.zone_set = _zone_cells(cfg["zones"])

        self.zone_boundary = set()
        self.zone_interior = set()
        for zx, zy in self.zone_set:
            is_boundary = False
            for nx, ny in [(zx - 1, zy), (zx + 1, zy), (zx, zy - 1), (zx, zy + 1)]:
                if (nx, ny) not in self.zone_set:
                    is_boundary = True
                    break
            if is_boundary:
                self.zone_boundary.add((zx, zy))
            else:
                self.zone_interior.add((zx, zy))

        self.safe_sides = set()
        for zx, zy in self.zone_set:
            if zx == 1:
                self.safe_sides.add("left")
            if zx == self.GW - 2:
                self.safe_sides.add("right")
            if zy == 1:
                self.safe_sides.add("top")
            if zy == self.GH - 2:
                self.safe_sides.add("bottom")

        self.pieces = {}
        self.initial_positions = {}
        for i, pd in enumerate(cfg["pieces"]):
            offs = _shape_offsets(pd["shape"])
            self.pieces[i] = {
                "offsets": offs,
                "offset_set": set(offs),
                "pos": (pd["sx"], pd["sy"]),
                "color": pd["color"],
            }
            self.initial_positions[i] = (pd["sx"], pd["sy"])

        self.gate_active = False
        self._game_over = False
        self._undo_stack = []

        occupy = set(self.wall_set) | self.zone_set | self.all_gate_cells
        for pdata in self.pieces.values():
            px, py = pdata["pos"]
            for dx, dy in pdata["offsets"]:
                occupy.add((px + dx, py + dy))
        for _ in range(50):
            rx = self._rng.randint(1, self.GW - 2)
            ry = self._rng.randint(1, self.GH - 2)
            if (rx, ry) not in occupy:
                self.player_x = rx
                self.player_y = ry
                self.initial_player = (rx, ry)
                break

        self.board_sprite = self._make_board_sprite()
        level.add_sprite(self.board_sprite)
        self._sync_hud()

    def _save_undo(self):
        positions = {pid: p["pos"] for pid, p in self.pieces.items()}
        self._undo_stack.append(
            (self.player_x, self.player_y, positions, self.gate_active)
        )

    def _restore_undo(self):
        if not self._undo_stack:
            return
        s = self._undo_stack.pop()
        self.player_x, self.player_y = s[0], s[1]
        for pid, pos in s[2].items():
            self.pieces[pid]["pos"] = pos
        self.gate_active = s[3]

    def step(self):
        aid = self.action.id

        if aid == GameAction.RESET:
            self.complete_action()
            return

        if aid == GameAction.ACTION7:
            self._restore_undo()
            self._rebuild_board()
            self._sync_hud()
            self.complete_action()
            return

        self._full_reset_requested = False
        level_before = self.level_index

        if aid in (
            GameAction.ACTION1,
            GameAction.ACTION2,
            GameAction.ACTION3,
            GameAction.ACTION4,
        ):
            self._save_undo()
            dx, dy = {
                GameAction.ACTION1: (0, -1),
                GameAction.ACTION2: (0, 1),
                GameAction.ACTION3: (-1, 0),
                GameAction.ACTION4: (1, 0),
            }[aid]
            self._move_player(dx, dy)

        if self.level_index == level_before:
            if not self.gate_active and self._check_assembly():
                self.gate_active = True

        self._rebuild_board()
        self._sync_hud()
        self.complete_action()

    def _move_player(self, dx, dy):
        nx, ny = self.player_x + dx, self.player_y + dy

        if (nx, ny) in self.wall_set:
            return

        hit_pid = self._find_piece_at(nx, ny)
        if hit_pid is not None:
            if self._would_hit_gate(hit_pid, dx, dy):
                self._penalty()
                return

            if self._try_push(hit_pid, dx, dy):
                self.player_x, self.player_y = nx, ny

                if self._piece_touches_border(hit_pid):
                    self._penalty()
                    return

                if self._check_overfit():
                    self._penalty()
                    return

                if self._check_gap():
                    self._penalty()
                    return

                if self._check_checkmate():
                    self._penalty()
                    return
            return

        self.player_x, self.player_y = nx, ny

        if self.gate_active and (nx, ny) in self.gate_cells:
            self.next_level()
        elif self.gate_active and (nx, ny) in self.fake_gate_cells:
            self._penalty()

    def _can_push(self, pid, dx, dy):
        p = self.pieces[pid]
        px, py = p["pos"]
        for ox, oy in p["offsets"]:
            ncx, ncy = px + dx + ox, py + dy + oy
            if (ncx, ncy) in self.wall_set or (ncx, ncy) in self.all_gate_cells:
                return False
            for other_pid, other_p in self.pieces.items():
                if other_pid != pid:
                    opx, opy = other_p["pos"]
                    if (ncx - opx, ncy - opy) in other_p["offset_set"]:
                        return False
        return True

    def _try_push(self, pid, dx, dy):
        if not self._can_push(pid, dx, dy):
            return False
        p = self.pieces[pid]
        px, py = p["pos"]
        p["pos"] = (px + dx, py + dy)
        return True

    def _would_hit_gate(self, pid, dx, dy):
        p = self.pieces[pid]
        px, py = p["pos"]
        for ox, oy in p["offsets"]:
            if (px + dx + ox, py + dy + oy) in self.all_gate_cells:
                return True
        return False

    def _piece_touches_border(self, pid):
        for cx, cy in self._piece_cells(pid):
            if cx == 1 and "left" not in self.safe_sides:
                return True
            if cx == self.GW - 2 and "right" not in self.safe_sides:
                return True
            if cy == 1 and "top" not in self.safe_sides:
                return True
            if cy == self.GH - 2 and "bottom" not in self.safe_sides:
                return True
        return False

    def _all_covered(self):
        covered = set()
        for pdata in self.pieces.values():
            px, py = pdata["pos"]
            for dx, dy in pdata["offsets"]:
                covered.add((px + dx, py + dy))
        return covered

    def _check_assembly(self):
        covered = self._all_covered()
        if not self.zone_set.issubset(covered):
            return False
        for pdata in self.pieces.values():
            px, py = pdata["pos"]
            in_zone = out_zone = False
            for dx, dy in pdata["offsets"]:
                if (px + dx, py + dy) in self.zone_set:
                    in_zone = True
                else:
                    out_zone = True
            if in_zone and out_zone:
                return False
        return True

    def _check_overfit(self):
        covered = self._all_covered()
        if not self.zone_boundary.issubset(covered):
            return False
        for pdata in self.pieces.values():
            px, py = pdata["pos"]
            in_zone = out_zone = False
            for dx, dy in pdata["offsets"]:
                if (px + dx, py + dy) in self.zone_set:
                    in_zone = True
                else:
                    out_zone = True
            if in_zone and out_zone:
                return True
        return False

    def _check_gap(self):
        if not self.zone_interior:
            return False
        covered = self._all_covered()
        boundary_full = self.zone_boundary.issubset(covered)
        interior_gaps = not self.zone_interior.issubset(covered)
        return boundary_full and interior_gaps

    def _check_checkmate(self):
        if self._check_assembly():
            return False
        for pid in self.pieces:
            for ddx, ddy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                if self._can_push(pid, ddx, ddy):
                    return False
        return True

    def _penalty(self):
        self.hud.flash_frames = 2
        self.lives -= 1
        if self.lives <= 0:
            self._game_over = True
            self.lose()
            return
        self._level_reset()

    def _make_board_sprite(self):
        pixels = self._render_grid()
        return Sprite(
            pixels=pixels,
            name="board",
            x=0,
            y=0,
            layer=0,
            collidable=False,
        )

    def _render_grid(self):
        piece_map = {}
        for pdata in self.pieces.values():
            px, py = pdata["pos"]
            for dx, dy in pdata["offsets"]:
                piece_map[(px + dx, py + dy)] = pdata["color"]

        pixels = []
        for y in range(self.GH):
            row = []
            for x in range(self.GW):
                if (x, y) in self.wall_set:
                    row.append(WALL)
                elif (x, y) == (self.player_x, self.player_y):
                    row.append(PLAYER)
                elif (x, y) in piece_map:
                    row.append(piece_map[(x, y)])
                elif (x, y) in self.all_gate_cells:
                    row.append(GATE_ON if self.gate_active else GATE_OFF)
                elif (x, y) in self.zone_set:
                    row.append(ZONE)
                else:
                    row.append(BG)
            pixels.append(row)
        return pixels

    def _rebuild_board(self):
        if self.board_sprite is None:
            return
        pixels = self._render_grid()
        for y in range(self.GH):
            for x in range(self.GW):
                self.board_sprite.pixels[y, x] = pixels[y][x]

    def _sync_hud(self):
        self.hud.lives = self.lives


class PuzzleEnvironment:
    _ACTION_MAP: Dict[str, GameAction] = {
        "reset": GameAction.RESET,
        "up": GameAction.ACTION1,
        "down": GameAction.ACTION2,
        "left": GameAction.ACTION3,
        "right": GameAction.ACTION4,
        "undo": GameAction.ACTION7,
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

    def __init__(self, seed: int = 0) -> None:
        self._engine = Ts01(seed=seed)
        self._total_turns = 0
        self._last_action_was_reset = False
        self._game_won = False

    def _build_text_obs(self) -> str:
        g = self._engine
        parts = []
        parts.append(f"LEVEL: {g.level_index + 1}/{len(LEVELS)}")
        parts.append(f"LIVES: {g.lives}")
        parts.append(f"PLAYER: {g.player_x} {g.player_y}")
        parts.append(f"GATE_ACTIVE: {g.gate_active}")
        gate_cells = " ".join(f"({gx},{gy})" for gx, gy in sorted(g.gate_cells))
        parts.append(f"GATE: {gate_cells}")
        if g.fake_gate_cells:
            fakes = " ".join(f"({fx},{fy})" for fx, fy in sorted(g.fake_gate_cells))
            parts.append(f"FAKE_GATES: {fakes}")
        zone_str = " ".join(f"({zx},{zy})" for zx, zy in sorted(g.zone_set))
        parts.append(f"ZONE: {zone_str}")
        for pid, pdata in g.pieces.items():
            px, py = pdata["pos"]
            cells = " ".join(f"({px + dx},{py + dy})" for dx, dy in pdata["offsets"])
            parts.append(
                f"PIECE_{pid}: ({px},{py}) color={pdata['color']} cells=[{cells}]"
            )
        return "\n".join(parts)

    @staticmethod
    def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
        chunk = chunk_type + data
        crc = struct.pack(">I", zlib.crc32(chunk) & 0xFFFFFFFF)
        return struct.pack(">I", len(data)) + chunk + crc

    def _build_image_bytes(self) -> Optional[bytes]:
        eng = self._engine
        frame = eng.camera.render(eng.current_level.get_sprites())
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        for idx, color in enumerate(self.ARC_PALETTE):
            mask = frame == idx
            rgb[mask] = color
        raw = b""
        for row in range(64):
            raw += b"\x00" + rgb[row].tobytes()
        ihdr = struct.pack(">IIBBBBB", 64, 64, 8, 2, 0, 0, 0)
        png = b"\x89PNG\r\n\x1a\n"
        png += self._png_chunk(b"IHDR", ihdr)
        png += self._png_chunk(b"IDAT", zlib.compress(raw))
        png += self._png_chunk(b"IEND", b"")
        return png

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
                "total_levels": len(self._engine._levels),
                "level_index": e.level_index,
                "lives": e.lives,
                "player": [e.player_x, e.player_y],
                "gate_active": e.gate_active,
                "game_over": e._game_over,
                "done": done,
                "info": info or {},
                "levels_completed": getattr(self._engine, "_score", 0),
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

    def is_done(self) -> bool:
        return self._engine._game_over or self._game_won

    def get_actions(self) -> List[str]:
        if self._engine._game_over or self._game_won:
            return ["reset"]
        return ["up", "down", "left", "right", "undo", "reset"]

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
        info: Dict = {"action": action}

        prev_level = e.level_index
        action_input = ActionInput(id=game_action)
        frame = e.perform_action(action_input, raw=True)

        game_won = frame and frame.state and frame.state.name == "WIN"
        total_levels = len(LEVELS)
        level_completed = game_won or (e.level_index > prev_level)
        reward = (1.0 / total_levels) if level_completed else 0.0

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
        e = self._engine
        index_grid = e.camera.render(e.current_level.get_sprites())
        arr = (
            np.array(index_grid, dtype=np.uint8)
            if not isinstance(index_grid, np.ndarray)
            else index_grid.astype(np.uint8)
        )
        h, w = arr.shape[:2]
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for idx, color in enumerate(self.ARC_PALETTE):
            mask = arr == idx
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

    @staticmethod
    def _resize_nearest(frame: np.ndarray, h: int, w: int) -> np.ndarray:
        src_h, src_w = frame.shape[0], frame.shape[1]
        row_idx = (np.arange(h) * src_h // h).astype(int)
        col_idx = (np.arange(w) * src_w // w).astype(int)
        return frame[np.ix_(row_idx, col_idx)]

    def _get_obs(self) -> np.ndarray:
        frame = self._env.render(mode="rgb_array")
        if frame.shape[0] != self.OBS_HEIGHT or frame.shape[1] != self.OBS_WIDTH:
            frame = self._resize_nearest(frame, self.OBS_HEIGHT, self.OBS_WIDTH)
        return frame

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
