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
    GameState as EngineState,
    Level,
    RenderableUserDisplay,
    Sprite,
)


@dataclass
class GameState:
    text_observation: str
    image_observation: Optional[bytes]
    valid_actions: Optional[list]
    turn: int
    metadata: dict = field(default_factory=dict)


@dataclass
class StepResult:
    state: GameState
    reward: float
    done: bool
    info: dict = field(default_factory=dict)


CELL = 3

C_BLACK = 0
C_BLUE = 1
C_RED = 2
C_GREEN = 3
C_YELLOW = 4
C_GREY = 5
C_MAGENTA = 6
C_ORANGE = 7
C_AZURE = 8

BG_COLOR = C_BLACK
PAD_COLOR = C_BLACK
MAX_LIVES = 3

_BLOCK_COLORS = {
    "red": C_RED,
    "blu": C_BLUE,
    "grn": C_GREEN,
    "mag": C_MAGENTA,
}

sprites = {}


for _cname, _cidx in _BLOCK_COLORS.items():
    _key = f"block_{_cname}"
    sprites[_key] = Sprite(
        pixels=np.full((CELL, CELL), _cidx, dtype=np.int32),
        name=_key,
        visible=True,
        collidable=True,
        tags=["block", f"color_{_cname}"],
        layer=2,
    )


for _cname, _cidx in _BLOCK_COLORS.items():
    _key = f"target_{_cname}"
    _px = np.full((CELL, CELL), -1, dtype=np.int32)
    _px[0, :] = _cidx
    _px[-1, :] = _cidx
    _px[:, 0] = _cidx
    _px[:, -1] = _cidx
    sprites[_key] = Sprite(
        pixels=_px,
        name=_key,
        visible=True,
        collidable=False,
        tags=["target", f"color_{_cname}"],
        layer=0,
    )


sprites["wall"] = Sprite(
    pixels=np.full((CELL, CELL), C_GREY, dtype=np.int32),
    name="wall",
    visible=True,
    collidable=True,
    tags=["wall"],
    layer=1,
)

sprites["arrow_up"] = Sprite(
    pixels=np.array(
        [
            [-1, C_ORANGE, -1],
            [C_ORANGE, C_ORANGE, C_ORANGE],
            [-1, -1, -1],
        ],
        dtype=np.int32,
    ),
    name="arrow_up",
    visible=True,
    collidable=False,
    tags=["arrow", "dir_up"],
    layer=5,
)

sprites["arrow_down"] = Sprite(
    pixels=np.array(
        [
            [-1, -1, -1],
            [C_ORANGE, C_ORANGE, C_ORANGE],
            [-1, C_ORANGE, -1],
        ],
        dtype=np.int32,
    ),
    name="arrow_down",
    visible=True,
    collidable=False,
    tags=["arrow", "dir_down"],
    layer=5,
)

sprites["arrow_left"] = Sprite(
    pixels=np.array(
        [
            [-1, C_ORANGE, -1],
            [C_ORANGE, C_ORANGE, -1],
            [-1, C_ORANGE, -1],
        ],
        dtype=np.int32,
    ),
    name="arrow_left",
    visible=True,
    collidable=False,
    tags=["arrow", "dir_left"],
    layer=5,
)

sprites["arrow_right"] = Sprite(
    pixels=np.array(
        [
            [-1, C_ORANGE, -1],
            [-1, C_ORANGE, C_ORANGE],
            [-1, C_ORANGE, -1],
        ],
        dtype=np.int32,
    ),
    name="arrow_right",
    visible=True,
    collidable=False,
    tags=["arrow", "dir_right"],
    layer=5,
)


_roamer_px = np.full((CELL, CELL), C_ORANGE, dtype=np.int32)
if CELL >= 3:
    _roamer_px[1, 1] = C_GREY
sprites["roamer"] = Sprite(
    pixels=_roamer_px,
    name="roamer",
    visible=True,
    collidable=True,
    tags=["roamer", "wall"],
    layer=3,
)

sprites["chaos_wall"] = Sprite(
    pixels=np.full((CELL, CELL), C_GREY, dtype=np.int32),
    name="chaos_wall",
    visible=True,
    collidable=True,
    tags=["wall", "chaos_wall"],
    layer=1,
)


def _build_level(spec):

    lev_sprites = []
    W, H = spec["grid"]
    gw, gh = W * CELL, H * CELL

    ox = (64 - gw) // 2
    oy = (64 - gh) // 2

    bw = gw + 2
    bh = gh + 2
    border_px = np.full((bh, bw), -1, dtype=np.int32)
    border_px[0, :] = C_AZURE
    border_px[-1, :] = C_AZURE
    border_px[:, 0] = C_AZURE
    border_px[:, -1] = C_AZURE
    border_sprite = Sprite(
        pixels=border_px,
        name="border",
        visible=True,
        collidable=False,
        tags=["border"],
        layer=-1,
    )
    border_sprite.set_position(ox - 1, oy - 1)
    lev_sprites.append(border_sprite)

    for cx, cy, cname in spec["targets"]:
        s = (
            sprites[f"target_{cname}"]
            .clone()
            .set_position(ox + cx * CELL, oy + cy * CELL)
        )
        lev_sprites.append(s)

    for cx, cy in spec["walls"]:
        s = sprites["wall"].clone().set_position(ox + cx * CELL, oy + cy * CELL)
        lev_sprites.append(s)

    for cx, cy in spec.get("chaos_walls", []):
        s = sprites["chaos_wall"].clone().set_position(ox + cx * CELL, oy + cy * CELL)
        lev_sprites.append(s)

    for roamer_cycle in spec.get("roamers", []):
        if roamer_cycle:
            cx, cy = roamer_cycle[0]
            s = sprites["roamer"].clone().set_position(ox + cx * CELL, oy + cy * CELL)
            lev_sprites.append(s)

    for cx, cy, cname in spec["blocks"]:
        s = (
            sprites[f"block_{cname}"]
            .clone()
            .set_position(ox + cx * CELL, oy + cy * CELL)
        )
        lev_sprites.append(s)

    return Level(
        sprites=lev_sprites,
        grid_size=(64, 64),
        data={
            "cell_w": W,
            "cell_h": H,
            "ox": ox,
            "oy": oy,
            "blocks": spec["blocks"],
            "targets": spec["targets"],
            "walls": spec["walls"],
            "chaos_walls": spec.get("chaos_walls", []),
            "roamers": spec.get("roamers", []),
            "max_moves": spec.get("max_moves", 30),
            "chaos_reserved": spec.get("chaos_reserved", set()),
        },
    )


_spec1 = {
    "grid": (9, 9),
    "blocks": [(1, 6, "red")],
    "targets": [(5, 2, "red")],
    "walls": [
        (6, 6),
        (5, 7),
        (3, 2),
        (5, 1),
        (7, 4),
        (2, 4),
        (8, 2),
    ],
    "max_moves": 12,
}

_spec1b = {
    "grid": (9, 9),
    "blocks": [(2, 1, "red")],
    "targets": [(6, 5, "red")],
    "walls": [
        (2, 6),
        (1, 5),
        (6, 3),
        (7, 5),
        (4, 7),
        (4, 2),
        (6, 8),
    ],
    "max_moves": 12,
}

_spec1c = {
    "grid": (9, 9),
    "blocks": [(7, 2, "red")],
    "targets": [(3, 6, "red")],
    "walls": [
        (2, 2),
        (3, 1),
        (5, 6),
        (3, 7),
        (1, 4),
        (6, 4),
        (0, 6),
    ],
    "max_moves": 12,
}

_spec1d = {
    "grid": (9, 9),
    "blocks": [(6, 7, "red")],
    "targets": [(2, 3, "red")],
    "walls": [
        (6, 2),
        (7, 3),
        (2, 5),
        (1, 3),
        (4, 1),
        (4, 6),
        (2, 0),
    ],
    "max_moves": 12,
}


_spec2 = {
    "grid": (10, 10),
    "blocks": [(1, 3, "red"), (7, 1, "blu")],
    "targets": [(5, 8, "red"), (3, 6, "blu")],
    "walls": [
        (6, 3),
        (5, 9),
        (2, 1),
        (3, 7),
        (9, 2),
        (4, 8),
        (8, 5),
        (1, 7),
        (6, 6),
        (0, 5),
    ],
    "max_moves": 24,
}

_spec2b = {
    "grid": (10, 10),
    "blocks": [(6, 1, "red"), (8, 7, "blu")],
    "targets": [(1, 5, "red"), (3, 3, "blu")],
    "walls": [
        (6, 6),
        (0, 5),
        (8, 2),
        (2, 3),
        (7, 9),
        (1, 4),
        (4, 8),
        (2, 1),
        (3, 6),
        (4, 0),
    ],
    "max_moves": 24,
}

_spec2c = {
    "grid": (10, 10),
    "blocks": [(8, 6, "red"), (2, 8, "blu")],
    "targets": [(4, 1, "red"), (6, 3, "blu")],
    "walls": [
        (3, 6),
        (4, 0),
        (7, 8),
        (6, 2),
        (0, 7),
        (5, 1),
        (1, 4),
        (8, 2),
        (3, 3),
        (9, 4),
    ],
    "max_moves": 24,
}

_spec2d = {
    "grid": (10, 10),
    "blocks": [(3, 8, "red"), (1, 2, "blu")],
    "targets": [(8, 4, "red"), (6, 6, "blu")],
    "walls": [
        (3, 3),
        (9, 4),
        (1, 7),
        (7, 6),
        (2, 0),
        (8, 5),
        (5, 1),
        (7, 8),
        (6, 3),
        (5, 9),
    ],
    "max_moves": 24,
}


_spec3 = {
    "grid": (11, 11),
    "blocks": [(2, 1, "red"), (8, 2, "blu")],
    "targets": [(9, 7, "red"), (2, 8, "blu")],
    "walls": [
        (9, 2),
        (8, 3),
        (1, 2),
        (2, 9),
        (10, 4),
        (9, 8),
        (5, 5),
        (4, 9),
        (0, 6),
        (6, 0),
        (4, 10),
        (10, 9),
    ],
    "max_moves": 24,
}

_spec3b = {
    "grid": (11, 11),
    "blocks": [(9, 2, "red"), (8, 8, "blu")],
    "targets": [(3, 9, "red"), (2, 2, "blu")],
    "walls": [
        (8, 9),
        (7, 8),
        (8, 1),
        (1, 2),
        (6, 10),
        (2, 9),
        (5, 5),
        (1, 4),
        (4, 0),
        (10, 6),
        (0, 4),
        (1, 10),
    ],
    "max_moves": 24,
}

_spec3c = {
    "grid": (11, 11),
    "blocks": [(8, 9, "red"), (2, 8, "blu")],
    "targets": [(1, 3, "red"), (8, 2, "blu")],
    "walls": [
        (1, 8),
        (2, 7),
        (9, 8),
        (8, 1),
        (0, 6),
        (1, 2),
        (5, 5),
        (6, 1),
        (10, 4),
        (4, 10),
        (6, 0),
        (0, 1),
    ],
    "max_moves": 24,
}

_spec3d = {
    "grid": (11, 11),
    "blocks": [(1, 8, "red"), (2, 2, "blu")],
    "targets": [(7, 1, "red"), (8, 8, "blu")],
    "walls": [
        (2, 1),
        (3, 2),
        (2, 9),
        (9, 8),
        (4, 0),
        (8, 1),
        (5, 5),
        (9, 6),
        (6, 10),
        (0, 4),
        (10, 6),
        (9, 0),
    ],
    "max_moves": 24,
}


_spec4 = {
    "grid": (16, 16),
    "blocks": [(1, 1, "red"), (1, 13, "grn"), (1, 3, "blu")],
    "targets": [(13, 10, "red"), (3, 12, "grn"), (10, 8, "blu")],
    "walls": [
        (15, 1),
        (14, 7),
        (11, 6),
        (14, 0),
        (13, 11),
        (2, 13),
        (4, 0),
        (3, 13),
        (11, 3),
        (10, 9),
        (5, 8),
        (11, 0),
        (5, 2),
        (8, 2),
        (15, 3),
        (4, 5),
        (7, 5),
        (8, 5),
        (15, 5),
        (8, 6),
        (15, 7),
        (1, 9),
        (8, 9),
        (15, 9),
        (5, 10),
        (2, 11),
        (8, 11),
        (15, 11),
        (5, 12),
        (8, 14),
        (5, 14),
        (15, 13),
    ],
    "roamers": [],
    "max_moves": 36,
}

_spec4b = {
    "grid": (16, 16),
    "blocks": [(14, 1, "red"), (2, 1, "grn"), (12, 1, "blu")],
    "targets": [(5, 13, "red"), (3, 3, "grn"), (7, 10, "blu")],
    "walls": [
        (14, 15),
        (8, 14),
        (9, 11),
        (15, 14),
        (4, 13),
        (2, 2),
        (15, 4),
        (2, 3),
        (12, 11),
        (6, 10),
        (7, 5),
        (15, 11),
        (13, 5),
        (13, 8),
        (12, 15),
        (10, 4),
        (10, 7),
        (10, 8),
        (10, 15),
        (9, 8),
        (8, 15),
        (6, 1),
        (6, 8),
        (6, 15),
        (5, 5),
        (4, 2),
        (4, 8),
        (4, 15),
        (3, 5),
        (1, 8),
        (1, 5),
        (2, 15),
    ],
    "roamers": [],
    "max_moves": 36,
}

_spec4c = {
    "grid": (16, 16),
    "blocks": [(14, 14, "red"), (14, 2, "grn"), (14, 12, "blu")],
    "targets": [(2, 5, "red"), (12, 3, "grn"), (5, 7, "blu")],
    "walls": [
        (0, 14),
        (1, 8),
        (4, 9),
        (1, 15),
        (2, 4),
        (13, 2),
        (11, 15),
        (12, 2),
        (4, 12),
        (5, 6),
        (10, 7),
        (4, 15),
        (10, 13),
        (7, 13),
        (0, 12),
        (11, 10),
        (8, 10),
        (7, 10),
        (0, 10),
        (7, 9),
        (0, 8),
        (14, 6),
        (7, 6),
        (0, 6),
        (10, 5),
        (13, 4),
        (7, 4),
        (0, 4),
        (10, 3),
        (7, 1),
        (10, 1),
        (0, 2),
    ],
    "roamers": [],
    "max_moves": 36,
}

_spec4d = {
    "grid": (16, 16),
    "blocks": [(1, 14, "red"), (13, 14, "grn"), (3, 14, "blu")],
    "targets": [(10, 2, "red"), (12, 12, "grn"), (8, 5, "blu")],
    "walls": [
        (1, 0),
        (7, 1),
        (6, 4),
        (0, 1),
        (11, 2),
        (13, 13),
        (0, 11),
        (13, 12),
        (3, 4),
        (9, 5),
        (8, 10),
        (0, 4),
        (2, 10),
        (2, 7),
        (3, 0),
        (5, 11),
        (5, 8),
        (5, 7),
        (5, 0),
        (6, 7),
        (7, 0),
        (9, 14),
        (9, 7),
        (9, 0),
        (10, 10),
        (11, 13),
        (11, 7),
        (11, 0),
        (12, 10),
        (14, 7),
        (14, 10),
        (13, 0),
    ],
    "roamers": [],
    "max_moves": 36,
}


_spec5 = {
    "grid": (16, 16),
    "blocks": [(1, 9, "red"), (9, 3, "grn"), (12, 9, "blu"), (9, 11, "mag")],
    "targets": [(3, 12, "red"), (9, 15, "blu"), (10, 9, "mag"), (13, 11, "grn")],
    "walls": [
        (15, 2),
        (14, 8),
        (12, 7),
        (14, 0),
        (13, 12),
        (3, 13),
        (4, 0),
        (12, 4),
        (11, 10),
        (6, 9),
        (11, 0),
        (10, 10),
        (4, 7),
        (6, 7),
        (5, 6),
        (5, 8),
        (8, 3),
        (15, 3),
        (8, 5),
        (15, 5),
        (2, 6),
        (15, 6),
        (2, 8),
        (15, 8),
        (2, 11),
        (8, 10),
        (15, 10),
        (1, 11),
        (8, 11),
        (15, 11),
        (2, 12),
        (15, 12),
        (2, 14),
        (8, 14),
        (8, 15),
        (15, 15),
    ],
    "roamers": [],
    "max_moves": 36,
}

_spec5b = {
    "grid": (16, 16),
    "blocks": [(6, 1, "red"), (12, 9, "grn"), (6, 12, "blu"), (4, 9, "mag")],
    "targets": [(3, 3, "red"), (0, 9, "blu"), (6, 10, "mag"), (4, 13, "grn")],
    "walls": [
        (13, 15),
        (7, 14),
        (8, 12),
        (15, 14),
        (3, 13),
        (2, 3),
        (15, 4),
        (11, 12),
        (5, 11),
        (6, 6),
        (15, 11),
        (5, 10),
        (8, 4),
        (8, 6),
        (9, 5),
        (7, 5),
        (12, 8),
        (12, 15),
        (10, 8),
        (10, 15),
        (9, 2),
        (9, 15),
        (7, 2),
        (7, 15),
        (4, 2),
        (5, 8),
        (5, 15),
        (4, 1),
        (4, 8),
        (4, 15),
        (3, 2),
        (3, 15),
        (1, 2),
        (1, 8),
        (0, 8),
        (0, 15),
    ],
    "roamers": [],
    "max_moves": 36,
}

_spec5c = {
    "grid": (16, 16),
    "blocks": [(14, 6, "red"), (6, 12, "grn"), (3, 6, "blu"), (6, 4, "mag")],
    "targets": [(12, 3, "red"), (6, 0, "blu"), (5, 6, "mag"), (2, 4, "grn")],
    "walls": [
        (0, 13),
        (1, 7),
        (3, 8),
        (1, 15),
        (2, 3),
        (12, 2),
        (11, 15),
        (3, 11),
        (4, 5),
        (9, 6),
        (4, 15),
        (5, 5),
        (11, 8),
        (9, 8),
        (10, 9),
        (10, 7),
        (7, 12),
        (0, 12),
        (7, 10),
        (0, 10),
        (13, 9),
        (0, 9),
        (13, 7),
        (0, 7),
        (13, 4),
        (7, 5),
        (0, 5),
        (14, 4),
        (7, 4),
        (0, 4),
        (13, 3),
        (0, 3),
        (13, 1),
        (7, 1),
        (7, 0),
        (0, 0),
    ],
    "roamers": [],
    "max_moves": 36,
}

_spec5d = {
    "grid": (16, 16),
    "blocks": [(9, 14, "red"), (3, 6, "grn"), (9, 3, "blu"), (11, 6, "mag")],
    "targets": [(12, 12, "red"), (15, 6, "blu"), (9, 5, "mag"), (11, 2, "grn")],
    "walls": [
        (2, 0),
        (8, 1),
        (7, 3),
        (0, 1),
        (12, 2),
        (13, 12),
        (0, 11),
        (4, 3),
        (10, 4),
        (9, 9),
        (0, 4),
        (10, 5),
        (7, 11),
        (7, 9),
        (6, 10),
        (8, 10),
        (3, 7),
        (3, 0),
        (5, 7),
        (5, 0),
        (6, 13),
        (6, 0),
        (8, 13),
        (8, 0),
        (11, 13),
        (10, 7),
        (10, 0),
        (11, 14),
        (11, 7),
        (11, 0),
        (12, 13),
        (12, 0),
        (14, 13),
        (14, 7),
        (15, 7),
        (15, 0),
    ],
    "roamers": [],
    "max_moves": 36,
}

_spec1_variants = [_spec1, _spec1b, _spec1c, _spec1d]
_spec2_variants = [_spec2, _spec2b, _spec2c, _spec2d]
_spec3_variants = [_spec3, _spec3b, _spec3c, _spec3d]
_spec4_variants = [_spec4, _spec4b, _spec4c, _spec4d]
_spec5_variants = [_spec5, _spec5b, _spec5c, _spec5d]
LEVEL_VARIANTS = [
    _spec1_variants,
    _spec2_variants,
    _spec3_variants,
    _spec4_variants,
    _spec5_variants,
]

levels = [_build_level(LEVEL_VARIANTS[i][0]) for i in range(5)]


class MoveCounter(RenderableUserDisplay):
    def __init__(self, max_moves: int):
        self.moves = 0
        self.max_moves = max_moves
        self.solved = False

    def reset(self, max_moves: int):
        self.moves = 0
        self.max_moves = max_moves
        self.solved = False

    def increment(self):
        self.moves += 1

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if self.max_moves == 0:
            return frame

        bar_w = min(w - 4, 36)
        used = min(bar_w, int(bar_w * self.moves / max(self.max_moves, 1)))
        bar_y = h - 2
        bar_x = (w - bar_w) // 2

        for i in range(bar_w):
            col = C_RED if i < used else C_GREY
            if 0 <= bar_y < h and 0 <= bar_x + i < w:
                frame[bar_y, bar_x + i] = col

        if self.solved:
            for i in range(w):
                frame[0, i] = C_GREEN

        return frame


class LivesDisplay(RenderableUserDisplay):
    def __init__(self):
        self.lives = MAX_LIVES

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        for i in range(MAX_LIVES):
            color = C_GREEN if i < self.lives else C_GREY
            x = 2 + i * 3
            frame[2, x] = color
            frame[2, x + 1] = color
            frame[3, x] = color
            frame[3, x + 1] = color
        return frame


class Gb49(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._mc = MoveCounter(max_moves=30)
        self._ld = LivesDisplay()
        self._lives = MAX_LIVES
        self._mid_level_reset = False
        self._rng = random.Random(seed)
        self._engine_snapshot = None
        self._engine_can_undo = False
        super().__init__(
            "gb49",
            levels,
            Camera(0, 0, 64, 64, BG_COLOR, PAD_COLOR, [self._mc, self._ld]),
            available_actions=[0, 1, 2, 3, 4, 6, 7],
        )

    def set_level(self, index: int) -> None:
        variant = self._rng.choice(LEVEL_VARIANTS[index])
        new_lv = _build_level(variant)
        self._levels[index] = new_lv.clone()
        self._clean_levels[index] = new_lv.clone()
        super().set_level(index)

    def on_set_level(self, level: Level) -> None:
        self._cw = self.current_level.get_data("cell_w")
        self._ch = self.current_level.get_data("cell_h")
        self._ox = self.current_level.get_data("ox")
        self._oy = self.current_level.get_data("oy")
        self._max = self.current_level.get_data("max_moves")

        self._blocks = self.current_level.get_sprites_by_tag("block")

        self._walls = self.current_level.get_sprites_by_tag("wall")
        self._arrows = self.current_level.get_sprites_by_tag("arrow")

        roamer_specs = self.current_level.get_data("roamers")
        self._roamer_sprites = self.current_level.get_sprites_by_tag("roamer")
        self._roamer_cycles = roamer_specs
        self._roamer_step = 0

        self._chaos_sprites = self.current_level.get_sprites_by_tag("chaos_wall")
        self._chaos_reserved: set = self.current_level.get_data("chaos_reserved")

        self._mc.reset(self._max)

        self._engine_can_undo = False
        self._engine_snapshot = None

        if not self._mid_level_reset:
            self._lives = MAX_LIVES
        self._mid_level_reset = False
        self._ld.lives = self._lives

    def handle_reset(self) -> None:
        state = getattr(self, "_state", None)
        if state in (EngineState.GAME_OVER, EngineState.WIN):
            self._lives = MAX_LIVES
            self._ld.lives = MAX_LIVES
            super().handle_reset()
        else:
            self._mc.reset(self._max)
            super().handle_reset()

    def _cell(self, sp):
        return ((sp.x - self._ox) // CELL, (sp.y - self._oy) // CELL)

    def _wall_at(self, c, r):
        px = self._ox + c * CELL
        py = self._oy + r * CELL
        for w in self._walls:
            if w.x == px and w.y == py:
                return True
        return False

    def _block_at(self, c, r, skip=None):
        px = self._ox + c * CELL
        py = self._oy + r * CELL
        for b in self._blocks:
            if b is skip:
                continue
            if b.x == px and b.y == py:
                return True
        return False

    def _advance_roamers(self):

        if not self._roamer_cycles:
            return
        next_step = (self._roamer_step + 1) % 3
        for i, sp in enumerate(self._roamer_sprites):
            if i < len(self._roamer_cycles):
                cycle = self._roamer_cycles[i]
                if cycle:
                    cx, cy = cycle[next_step % len(cycle)]
                    sp.set_position(self._ox + cx * CELL, self._oy + cy * CELL)
        self._roamer_step = next_step

    def _shuffle_chaos_walls(self):

        if not self._chaos_sprites:
            return

        occupied: set = set(self._chaos_reserved)

        for sp in self._roamer_sprites:
            c = (sp.x - self._ox) // CELL
            r = (sp.y - self._oy) // CELL
            occupied.add((c, r))

        for sp in self._blocks:
            c = (sp.x - self._ox) // CELL
            r = (sp.y - self._oy) // CELL
            occupied.add((c, r))

        pool = [
            (c, r)
            for c in range(1, self._cw - 1)
            for r in range(1, self._ch - 1)
            if (c, r) not in occupied
        ]
        self._rng.shuffle(pool)

        chosen: set = set()
        for sp in self._chaos_sprites:
            for pos in pool:
                if pos not in chosen:
                    chosen.add(pos)
                    cx, cy = pos
                    sp.set_position(self._ox + cx * CELL, self._oy + cy * CELL)
                    break

    def _tilt(self, dx, dy):

        order = list(self._blocks)

        if dx > 0:
            order.sort(key=lambda s: -s.x)
        elif dx < 0:
            order.sort(key=lambda s: s.x)
        elif dy > 0:
            order.sort(key=lambda s: -s.y)
        elif dy < 0:
            order.sort(key=lambda s: s.y)

        any_moved = False
        for blk in order:
            c, r = self._cell(blk)
            nc, nr = c, r

            while True:
                tc, tr = nc + dx, nr + dy

                if tc < 0 or tc >= self._cw or tr < 0 or tr >= self._ch:
                    break

                if self._wall_at(tc, tr):
                    break

                if self._block_at(tc, tr, skip=blk):
                    break
                nc, nr = tc, tr

            if nc != c or nr != r:
                blk.set_position(self._ox + nc * CELL, self._oy + nr * CELL)
                any_moved = True

        if any_moved:
            self._advance_roamers()
            self._shuffle_chaos_walls()

        return any_moved

    def _check_win(self):

        targets = self.current_level.get_data("targets")

        for tcx, tcy, tcname in targets:
            tpx = self._ox + tcx * CELL
            tpy = self._oy + tcy * CELL
            hit = False
            for blk in self._blocks:
                blk_color = ""
                if blk.tags:
                    for tg in blk.tags:
                        if tg.startswith("color_"):
                            blk_color = tg[6:]
                            break
                if blk.x == tpx and blk.y == tpy and blk_color == tcname:
                    hit = True
                    break
            if not hit:
                return False
        return True

    def _engine_save_snapshot(self) -> None:
        self._engine_snapshot = {
            "block_positions": [(b.x, b.y) for b in self._blocks],
        }

    def _engine_restore_snapshot(self) -> None:
        snap = self._engine_snapshot
        for i, (bx, by) in enumerate(snap["block_positions"]):
            self._blocks[i].set_position(bx, by)

    def step(self) -> None:
        aid = self.action.id

        if aid == GameAction.ACTION7:
            if self._engine_can_undo and self._engine_snapshot is not None:
                self._engine_restore_snapshot()
                self._engine_can_undo = False
                self._engine_snapshot = None
            self._mc.increment()
            if self._mc.moves >= self._mc.max_moves:
                self._lives -= 1
                self._ld.lives = self._lives
                if self._lives <= 0:
                    self.lose()
                else:
                    self._mid_level_reset = True
                    self.level_reset()
            self.complete_action()
            return

        self._engine_save_snapshot()

        dx, dy = 0, 0
        aid = self.action.id

        if aid == GameAction.ACTION1:
            dx, dy = 0, -1
        elif aid == GameAction.ACTION2:
            dx, dy = 0, 1
        elif aid == GameAction.ACTION3:
            dx, dy = -1, 0
        elif aid == GameAction.ACTION4:
            dx, dy = 1, 0

        elif aid == GameAction.ACTION6:
            raw_x = self.action.data.get("x", 0)
            raw_y = self.action.data.get("y", 0)
            coords = self.camera.display_to_grid(raw_x, raw_y)
            if coords:
                gx, gy = coords
                gw = self._cw * CELL
                gh = self._ch * CELL
                grid_left = self._ox
                grid_right = self._ox + gw
                grid_top = self._oy
                grid_bottom = self._oy + gh

                if gy < grid_top:
                    dx, dy = 0, -1
                elif gy >= grid_bottom:
                    dx, dy = 0, 1
                elif gx < grid_left:
                    dx, dy = -1, 0
                elif gx >= grid_right:
                    dx, dy = 1, 0
                else:
                    gcx = grid_left + gw // 2
                    gcy = grid_top + gh // 2
                    rel_x = gx - gcx
                    rel_y = gy - gcy
                    if abs(rel_x) >= abs(rel_y):
                        dx = 1 if rel_x > 0 else -1
                    else:
                        dy = 1 if rel_y > 0 else -1

        if dx != 0 or dy != 0:
            moved = self._tilt(dx, dy)
            if moved:
                self._mc.increment()

        if self._check_win():
            self._mc.solved = True
            self._engine_can_undo = False
            self._engine_snapshot = None
            self.next_level()
            self.complete_action()
            return

        if self._mc.moves >= self._mc.max_moves:
            self._lives -= 1
            self._ld.lives = self._lives
            self._engine_can_undo = False
            self._engine_snapshot = None
            if self._lives <= 0:
                self.lose()
            else:
                self._mid_level_reset = True
                self.level_reset()
            self.complete_action()
            return

        self._engine_can_undo = True
        self.complete_action()

    def _grid_to_text(self) -> str:
        _color_char = {"red": "R", "blu": "B", "grn": "G", "mag": "M"}
        _target_char = {"red": "r", "blu": "b", "grn": "g", "mag": "m"}

        target_map: Dict[tuple, str] = {}
        for tcx, tcy, tcname in self.current_level.get_data("targets"):
            target_map[(tcx, tcy)] = tcname

        rows = []
        for r in range(self._ch):
            row_chars = []
            for c in range(self._cw):
                px = self._ox + c * CELL
                py = self._oy + r * CELL
                char = "."
                for w in self._walls:
                    if w.x == px and w.y == py:
                        char = "W"
                        break
                if char == ".":
                    for b in self._blocks:
                        if b.x == px and b.y == py:
                            blk_color = ""
                            if b.tags:
                                for tg in b.tags:
                                    if tg.startswith("color_"):
                                        blk_color = tg[6:]
                                        break
                            char = _color_char.get(blk_color, "B")
                            break
                if char == ".":
                    for a in self._arrows:
                        if a.x == px and a.y == py:
                            char = "A"
                            break
                if char == "." and (c, r) in target_map:
                    char = _target_char.get(target_map[(c, r)], "t")
                row_chars.append(char)
            rows.append(" ".join(row_chars))
        return (
            "\n".join(rows)
            + f"\nMoves: {self._mc.moves}/{self._mc.max_moves} | Lives: {self._lives}"
        )


class PuzzleEnvironment:
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

    _ACTION_MAP: Dict[str, GameAction] = {
        "reset": GameAction.RESET,
        "up": GameAction.ACTION1,
        "down": GameAction.ACTION2,
        "left": GameAction.ACTION3,
        "right": GameAction.ACTION4,
        "click": GameAction.ACTION6,
        "undo": GameAction.ACTION7,
    }

    _VALID_ACTIONS: List[str] = [
        "reset",
        "up",
        "down",
        "left",
        "right",
        "click",
        "undo",
    ]

    def __init__(self, seed: int = 0) -> None:
        self._engine = Gb49(seed=seed)
        self._done = False
        self._last_action_was_reset = False
        self._total_turns = 0

    def _render_frame(self) -> np.ndarray:
        e = self._engine
        return e.camera.render(e.current_level.get_sprites())

    @staticmethod
    def _frame_to_png(frame: np.ndarray) -> bytes:
        h, w = frame.shape[:2]
        buf = io.BytesIO()
        buf.write(b"\x89PNG\r\n\x1a\n")

        def _chunk(ctype: bytes, data: bytes) -> None:
            buf.write(struct.pack(">I", len(data)))
            buf.write(ctype)
            buf.write(data)
            buf.write(struct.pack(">I", zlib.crc32(ctype + data) & 0xFFFFFFFF))

        _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
        raw = b""
        for y in range(h):
            raw += b"\x00" + frame[y].astype(np.uint8).tobytes()
        _chunk(b"IDAT", zlib.compress(raw))
        _chunk(b"IEND", b"")
        return buf.getvalue()

    def _build_state(self) -> GameState:
        e = self._engine
        frame = self._render_frame()
        png_bytes = self._frame_to_png(frame)

        return GameState(
            text_observation=e._grid_to_text(),
            image_observation=png_bytes,
            valid_actions=self.get_actions() if not self._done else None,
            turn=self._total_turns,
            metadata={
                "total_levels": len(self._engine._levels),
                "level": e._current_level_index,
                "lives": e._lives,
                "moves": e._mc.moves,
                "max_moves": e._mc.max_moves,
                "grid_w": e._cw,
                "grid_h": e._ch,
                "levels_completed": getattr(self._engine, "_score", 0),
                "level_index": e._current_level_index,
                "game_over": getattr(getattr(self._engine, "_state", None), "name", "") == "GAME_OVER",
            },
        )

    def reset(self) -> GameState:
        e = self._engine
        if self._done and e._state == EngineState.WIN:
            e.full_reset()
        elif self._last_action_was_reset:
            e.full_reset()
        else:
            e.perform_action(ActionInput(id=GameAction.RESET))
        self._done = False
        self._last_action_was_reset = True
        self._total_turns = 0
        return self._build_state()

    def get_actions(self) -> List[str]:
        if self._done:
            return ["reset"]
        return list(self._VALID_ACTIONS)

    def step(self, action: str) -> StepResult:
        parts = action.split()
        action_key = parts[0] if parts else action

        if action_key not in self._ACTION_MAP:
            raise ValueError(
                f"Invalid action '{action_key}'. Must be one of {list(self._ACTION_MAP.keys())}"
            )

        e = self._engine

        if action_key == "reset":
            state = self.reset()
            return StepResult(
                state=state, reward=0.0, done=False, info={"action": "reset"}
            )

        self._last_action_was_reset = False

        game_action = self._ACTION_MAP[action_key]

        if game_action == GameAction.ACTION6:
            if len(parts) < 3:
                parts = ["click", "32", "32"]
            cx = int(parts[1])
            cy = int(parts[2])
            action_input = ActionInput(id=game_action, data={"x": cx, "y": cy})
        else:
            action_input = ActionInput(id=game_action)

        level_before = e._current_level_index
        result = e.perform_action(action_input)

        self._total_turns += 1

        total_levels = len(e._levels)
        info: Dict[str, Any] = {"action": action}

        if result.state == EngineState.WIN:
            self._done = True
            info["reason"] = "game_complete"
            return StepResult(
                state=self._build_state(),
                reward=1.0 / total_levels,
                done=True,
                info=info,
            )

        if result.state == EngineState.GAME_OVER:
            self._done = True
            info["reason"] = "game_over"
            return StepResult(
                state=self._build_state(),
                reward=0.0,
                done=True,
                info=info,
            )

        level_after = e._current_level_index
        if level_after > level_before:
            reward = 1.0 / total_levels
            info["reason"] = "level_complete"
        else:
            reward = 0.0

        return StepResult(
            state=self._build_state(),
            reward=reward,
            done=False,
            info=info,
        )

    def is_done(self) -> bool:
        return self._done

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        frame = self._render_frame()
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        for idx, color in enumerate(self.ARC_PALETTE):
            mask = frame == idx
            rgb[mask] = color
        return rgb

    def close(self) -> None:
        self._engine = None


class ArcGameEnv(gym.Env):
    metadata: Dict[str, Any] = {
        "render_modes": ["rgb_array"],
        "render_fps": 10,
    }

    ACTION_LIST: List[str] = [
        "reset",
        "up",
        "down",
        "left",
        "right",
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
        self._env: Optional[PuzzleEnvironment] = None

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

    def step(
        self,
        action: int,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action_str: str = self._action_to_string[int(action)]

        if action_str == "click":
            action_str = "click 32 32"

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
        frame = self._env.render()
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
    except Exception:
        pass

    obs, info = env.reset()

    obs, reward, term, trunc, info = env.step(1)

    obs, reward, term, trunc, info = env.step(5)

    frame = env.render()

    env.close()
