import random
import struct
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import gymnasium as gym
from gymnasium import spaces

from arcengine import (
    ARCBaseGame,
    ActionInput,
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

C_EMPTY = 0
C_WALL = 1
C_POISON = 2
C_PLAYER = 3
C_FOOD1 = 4
C_FOOD2 = 5
C_FOOD3 = 6
C_FOOD4 = 7
C_KEY = 8
C_GATE = 9
C_GATE_OPEN = 11
C_FOOD5 = 12
C_FOOD6 = 13
C_DECOY = 14

CLR_LIFE = 14
CLR_LIFE_LOST = 4
CLR_BAR_HI = 10
CLR_BAR_LOW = 12
CLR_BAR_EMPTY = 5

MAX_LIVES = 3

W = C_WALL
E = C_EMPTY
P = C_POISON
Pl = C_PLAYER
D = C_DECOY

CELL_CHARS = {
    C_EMPTY: ".",
    C_WALL: "#",
    C_POISON: "X",
    C_PLAYER: "@",
    C_FOOD1: "1",
    C_FOOD2: "2",
    C_FOOD3: "3",
    C_FOOD4: "4",
    C_KEY: "K",
    C_GATE: "]",
    C_GATE_OPEN: "[",
    C_FOOD5: "5",
    C_FOOD6: "6",
    C_DECOY: "~",
}

LEVEL1_GRID = [
    [W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W],
    [W, Pl, E, E, 4, E, E, E, E, E, E, E, D, E, E, E, E, E, E, E, E, E, W, W],
    [W, E, W, W, E, W, W, W, W, W, W, W, E, W, W, W, W, W, W, E, W, W, W, W],
    [W, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, W, W],
    [W, E, W, W, E, W, W, W, E, W, W, W, E, W, W, W, W, W, W, E, W, W, W, W],
    [W, E, E, E, E, E, P, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, W, W],
    [W, E, W, W, E, W, W, W, E, W, W, W, E, W, W, W, W, W, W, E, W, W, W, W],
    [W, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, W, W],
    [W, E, W, W, E, W, W, W, E, W, W, W, E, W, W, W, P, W, W, E, W, W, W, W],
    [W, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, W, W],
    [W, E, W, W, E, W, W, W, E, W, W, W, E, W, W, W, W, W, W, E, W, W, W, W],
    [W, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, W, W],
    [W, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, W, W],
    [W, E, W, W, E, W, W, W, E, W, W, W, E, W, W, W, W, W, W, E, W, W, W, W],
    [W, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, W, W],
    [W, E, W, W, E, W, W, W, E, W, W, W, E, W, W, W, W, W, W, E, W, W, W, W],
    [W, E, E, E, E, E, E, E, E, P, E, E, E, E, E, E, E, E, E, E, E, E, W, W],
    [W, E, W, W, E, W, W, W, E, W, W, W, E, W, W, W, W, W, W, E, W, W, W, W],
    [W, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, W, W],
    [W, E, W, W, E, W, W, W, E, W, W, W, E, W, W, W, W, W, W, E, W, W, W, W],
    [W, E, E, E, D, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, W, W],
    [W, E, W, W, E, W, W, W, E, W, W, W, E, W, W, W, W, D, D, D, D, W, W, W],
    [W, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, 9, W],
    [W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W],
]
LEVEL1_GRID[3][8] = C_FOOD2
LEVEL1_GRID[3][19] = C_FOOD4
LEVEL1_GRID[12][4] = C_FOOD3
LEVEL1_GRID[12][19] = C_FOOD5
LEVEL1_GRID[20][8] = C_FOOD6

LEVEL2_GRID = [
    [W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W],
    [W, Pl, E, 4, E, E, E, D, E, E, E, E, E, E, E, E, E, E, W, W],
    [W, W, W, E, W, W, W, W, W, W, W, W, W, W, E, W, W, W, W, W],
    [W, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, W, W],
    [W, W, W, E, W, W, W, W, W, W, W, W, W, E, W, W, W, W, W, W],
    [W, E, E, E, E, E, E, P, E, E, E, E, E, E, E, E, E, E, W, W],
    [W, W, E, W, W, W, W, W, W, W, W, W, W, W, E, W, W, W, W, W],
    [W, E, E, E, E, E, E, E, E, E, E, E, D, E, E, E, E, E, W, W],
    [W, W, E, W, W, W, W, W, W, W, W, W, W, W, E, W, W, W, W, W],
    [W, E, E, E, E, E, E, E, E, E, P, E, E, E, E, E, E, E, W, W],
    [W, W, W, E, W, W, W, W, W, W, W, W, W, W, E, W, W, W, W, W],
    [W, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, W, W],
    [W, W, W, E, W, W, W, W, W, W, W, W, W, W, E, W, W, W, W, W],
    [W, E, E, E, E, E, P, E, E, E, E, E, E, E, E, E, E, E, W, W],
    [W, W, W, E, W, W, W, W, W, W, W, E, W, W, W, W, W, W, W, W],
    [W, E, E, E, E, E, E, E, E, E, E, E, E, P, E, E, E, E, W, W],
    [W, W, W, W, W, W, W, W, W, W, W, E, W, W, D, D, W, W, W, W],
    [W, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, W, W],
    [W, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, 9, W],
    [W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W],
]
LEVEL2_GRID[3][14] = C_FOOD2
LEVEL2_GRID[7][3] = C_FOOD3
LEVEL2_GRID[11][14] = C_FOOD4
LEVEL2_GRID[15][3] = C_FOOD5
LEVEL2_GRID[17][11] = C_FOOD6

LEVEL3_GRID = [
    [W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W],
    [W, Pl, E, 4, E, E, E, E, E, W, E, E, E, E, E, E, E, E, W, W],
    [W, E, W, W, E, W, W, W, E, W, E, W, W, W, E, W, W, W, W, W],
    [W, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, W, W],
    [W, E, W, W, E, W, W, W, E, W, E, W, W, W, E, W, W, W, W, W],
    [W, E, E, E, P, E, E, E, E, W, E, E, E, P, E, E, E, E, W, W],
    [W, E, W, W, E, W, W, W, E, W, E, W, W, W, E, W, W, W, W, W],
    [W, E, E, E, E, E, E, E, E, W, E, E, E, E, E, E, E, E, W, W],
    [W, E, W, W, E, W, W, W, E, W, E, W, W, W, E, W, W, W, W, W],
    [W, E, E, E, E, P, E, E, D, W, D, E, E, P, E, E, E, E, W, W],
    [W, E, W, W, E, W, W, W, E, W, E, W, W, W, E, W, W, W, W, W],
    [W, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, W, W],
    [W, E, W, W, E, W, W, W, E, W, E, W, W, W, E, W, W, W, W, W],
    [W, E, E, E, E, E, E, E, E, W, E, E, E, E, E, E, E, E, W, W],
    [W, E, W, W, E, W, W, W, E, W, E, W, W, W, E, W, W, W, W, W],
    [W, E, E, E, E, E, E, E, E, W, E, E, E, E, E, E, E, E, W, W],
    [W, E, W, W, E, W, W, W, E, W, E, W, W, D, D, D, W, W, W, W],
    [W, E, E, E, E, E, E, E, E, W, E, E, E, E, E, E, E, E, W, W],
    [W, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, 9, W],
    [W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W],
]
LEVEL3_GRID[3][14] = C_FOOD2
LEVEL3_GRID[7][3] = C_FOOD3
LEVEL3_GRID[11][14] = C_FOOD4
LEVEL3_GRID[13][3] = C_FOOD5
LEVEL3_GRID[15][14] = C_FOOD6

LEVEL4_GRID = [
    [W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W],
    [W, Pl, E, E, 4, E, E, E, E, E, E, E, E, W, E, E, E, E, E, E, E, E, E, E, W, W],
    [W, E, W, W, E, W, W, W, E, W, W, W, E, W, E, W, W, W, E, W, W, W, E, W, W, W],
    [W, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, W, W],
    [W, E, W, W, E, W, W, W, E, W, W, W, E, W, E, W, W, W, E, W, W, W, E, W, W, W],
    [W, E, E, E, P, E, E, E, E, E, E, E, E, W, E, E, E, P, E, E, E, E, E, E, W, W],
    [W, E, W, W, E, W, W, W, E, W, W, W, E, W, E, W, W, W, E, W, W, W, E, W, W, W],
    [W, E, E, E, E, E, E, E, E, E, E, E, E, W, E, E, E, E, E, E, E, E, E, E, W, W],
    [W, E, W, W, E, W, W, W, E, W, W, W, E, W, E, W, W, W, E, W, W, W, E, W, W, W],
    [W, E, E, E, E, E, E, E, E, E, E, E, E, W, E, E, E, E, E, E, E, E, E, E, W, W],
    [W, E, W, W, E, W, W, W, E, W, W, W, E, W, E, W, W, W, E, W, W, W, E, W, W, W],
    [W, E, E, E, E, E, P, E, D, E, E, E, E, W, E, E, E, E, D, E, P, E, E, E, W, W],
    [W, E, W, W, E, W, W, W, E, W, W, W, E, W, E, W, W, W, E, W, W, W, E, W, W, W],
    [W, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, W, W],
    [W, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, W, W],
    [W, E, W, W, E, W, W, W, E, W, W, W, E, W, E, W, W, W, E, W, W, W, E, W, W, W],
    [W, E, E, E, E, E, E, E, E, E, E, E, E, W, E, E, E, E, E, E, E, E, E, E, W, W],
    [W, E, W, W, E, W, W, W, E, W, W, W, E, W, E, W, W, W, E, W, W, W, E, W, W, W],
    [W, E, E, E, E, E, E, E, E, E, E, E, E, W, E, E, E, E, E, E, E, E, E, E, W, W],
    [W, E, W, W, E, W, W, W, E, W, W, W, E, W, E, W, W, W, E, W, W, W, E, W, W, W],
    [W, E, E, E, E, E, E, E, E, P, E, E, E, W, E, E, E, E, E, E, E, E, E, E, W, W],
    [W, E, W, W, E, W, W, W, E, W, W, W, E, W, E, W, W, W, E, W, W, W, E, W, W, W],
    [W, E, E, E, E, E, E, E, E, E, E, E, E, W, E, E, E, E, E, E, E, E, E, E, W, W],
    [W, E, W, W, E, W, W, W, E, W, W, W, E, W, E, W, W, W, E, D, D, D, D, W, W, W],
    [W, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, 9, W],
    [W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W],
]
LEVEL4_GRID[3][8] = C_FOOD2
LEVEL4_GRID[7][4] = C_FOOD3
LEVEL4_GRID[3][18] = C_FOOD4
LEVEL4_GRID[14][4] = C_FOOD5
LEVEL4_GRID[14][18] = C_FOOD6

LEVEL_CONFIGS = [
    {
        "grid": LEVEL1_GRID,
        "gw": 24,
        "gh": 24,
        "cell": 2,
        "food_seq": [C_FOOD1, C_FOOD2, C_FOOD3, C_FOOD4, C_FOOD5, C_FOOD6],
        "move_limit": 360,
        "decoy_walls": {
            (12, 1): [
                (12, 2),
                (12, 3),
                (12, 4),
                (12, 5),
                (12, 6),
                (12, 7),
                (12, 8),
                (12, 9),
                (12, 10),
                (12, 11),
            ],
            (4, 20): [(5, 20), (6, 20), (7, 20), (8, 20)],
            (17, 21): [(18, 21)],
            (18, 21): [(19, 21)],
            (19, 21): [(20, 21)],
            (20, 21): [(17, 21)],
        },
    },
    {
        "grid": LEVEL2_GRID,
        "gw": 20,
        "gh": 20,
        "cell": 3,
        "food_seq": [C_FOOD1, C_FOOD2, C_FOOD3, C_FOOD4, C_FOOD5, C_FOOD6],
        "move_limit": 320,
        "decoy_walls": {
            (7, 1): [(7, 2), (7, 3)],
            (12, 7): [(12, 8), (12, 9)],
            (14, 16): [(15, 16)],
            (15, 16): [(13, 16)],
        },
    },
    {
        "grid": LEVEL3_GRID,
        "gw": 20,
        "gh": 20,
        "cell": 3,
        "food_seq": [C_FOOD1, C_FOOD2, C_FOOD3, C_FOOD4, C_FOOD5, C_FOOD6],
        "move_limit": 328,
        "decoy_walls": {
            (8, 9): [(9, 9)],
            (10, 9): [(9, 9)],
            (13, 16): [(14, 16)],
            (14, 16): [(15, 16)],
            (15, 16): [(16, 16)],
        },
    },
    {
        "grid": LEVEL4_GRID,
        "gw": 26,
        "gh": 26,
        "cell": 2,
        "food_seq": [C_FOOD1, C_FOOD2, C_FOOD3, C_FOOD4, C_FOOD5, C_FOOD6],
        "move_limit": 440,
        "decoy_walls": {
            (8, 11): [(9, 11), (10, 11)],
            (18, 11): [(17, 11), (16, 11)],
            (19, 23): [(20, 23)],
            (20, 23): [(21, 23)],
            (21, 23): [(22, 23)],
            (22, 23): [(19, 23)],
        },
    },
]

PLAYER_START_POSITIONS = [
    [(1, 1), (20, 1)],
    [(1, 1), (14, 1)],
    [(1, 1), (14, 1)],
    [(1, 1), (22, 1)],
]

_levels = [
    Level(
        sprites=[],
        grid_size=(cfg["gw"] * cfg["cell"], cfg["gh"] * cfg["cell"]),
        name=f"Level {i + 1}",
        data=cfg,
    )
    for i, cfg in enumerate(LEVEL_CONFIGS)
]


def solid(c, size):
    return [[c] * size for _ in range(size)]


class ProgressDisplay(RenderableUserDisplay):
    def __init__(self, game: "Fe01") -> None:
        self.game = game

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        g = self.game
        fh, fw = frame.shape[:2] if frame.ndim >= 2 else (frame.shape[0], 1)

        cam_w = g.gw
        cam_h = g.gh
        if cam_w == 0 or cam_h == 0:
            return frame
        scale = min(fw // cam_w, fh // cam_h)
        if scale == 0:
            return frame
        half = max(1, scale // 3)

        x_off = (fw - cam_w * scale) // 2
        y_off = (fh - cam_h * scale) // 2

        def col_slice(c):
            return slice(x_off + c * scale, x_off + (c + 1) * scale)

        if g.move_limit > 0:
            remaining = max(0, g.move_limit - g.move_count)
            ratio = remaining / g.move_limit
            bar_pixel_width = cam_w * scale
            filled_px = max(0, int(ratio * bar_pixel_width))

            bar_y = slice(
                y_off + (cam_h - 1) * scale + (scale - half),
                y_off + cam_h * scale,
            )
            bar_color = CLR_BAR_HI if ratio > 0.25 else CLR_BAR_LOW

            for px in range(bar_pixel_width):
                x = x_off + px
                if 0 <= x < fw:
                    frame[bar_y, x] = bar_color if px < filled_px else CLR_BAR_EMPTY

        sq = max(2, scale)
        gap = 1
        for i in range(MAX_LIVES):
            right_edge = x_off + cam_w * scale
            sq_x1 = right_edge - (i + 1) * sq - i * gap
            sq_x2 = sq_x1 + sq
            sq_y1 = y_off
            sq_y2 = sq_y1 + sq
            color = CLR_LIFE if (MAX_LIVES - 1 - i) < g._lives else CLR_LIFE_LOST
            if sq_x1 >= 0 and sq_x2 <= fw and sq_y2 <= fh:
                frame[sq_y1:sq_y2, sq_x1:sq_x2] = color

        return frame


class Fe01(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self._rng = random.Random(seed)
        cfg = LEVEL_CONFIGS[0]

        self.grid = []
        self.gw = cfg["gw"]
        self.gh = cfg["gh"]
        self.cell = cfg["cell"]
        self.food_seq = list(cfg["food_seq"])
        self.decoy_walls = cfg["decoy_walls"]
        self.move_limit = cfg["move_limit"]
        self.player_x = 0
        self.player_y = 0
        self.next_food_idx = 0
        self.key_active = False
        self.level_won = False
        self.move_count = 0
        self._lives = MAX_LIVES
        self._undo_stack: List[Dict] = []

        self._hud = ProgressDisplay(self)

        camera = Camera(
            x=0,
            y=0,
            background=C_EMPTY,
            letter_box=0,
            width=cfg["gw"] * cfg["cell"],
            height=cfg["gh"] * cfg["cell"],
            interfaces=[self._hud],
        )

        super().__init__(
            game_id="fe01",
            levels=_levels,
            camera=camera,
            available_actions=[0, 1, 2, 3, 4, 7],
        )

    def on_set_level(self, level: Level) -> None:
        cfg = LEVEL_CONFIGS[self.level_index]

        self.gw = cfg["gw"]
        self.gh = cfg["gh"]
        self.cell = cfg["cell"]
        self.camera.width = self.gw * self.cell
        self.camera.height = self.gh * self.cell
        self.food_seq = list(cfg["food_seq"])
        self.decoy_walls = cfg["decoy_walls"]
        self.move_limit = cfg["move_limit"]
        self.grid = [row[:] for row in cfg["grid"]]

        self.next_food_idx = 0
        self.key_active = False
        self.level_won = False
        self.move_count = 0
        self._lives = MAX_LIVES
        self._undo_stack = []

        for r in range(self.gh):
            for c in range(self.gw):
                if self.grid[r][c] == C_PLAYER:
                    self.grid[r][c] = C_EMPTY

        sx, sy = self._rng.choice(PLAYER_START_POSITIONS[self.level_index])
        self.player_x = sx
        self.player_y = sy
        self.grid[sy][sx] = C_PLAYER

        self.rebuild(level)

    def handle_reset(self):
        self._lives = MAX_LIVES
        self._undo_stack = []
        super().handle_reset()

    def _lose_life(self) -> None:
        self._lives -= 1
        if self._lives > 0:
            self._reset_level_state()
        else:
            self.lose()

    def _reset_level_state(self) -> None:
        cfg = LEVEL_CONFIGS[self.level_index]
        self.gw = cfg["gw"]
        self.gh = cfg["gh"]
        self.cell = cfg["cell"]
        self.food_seq = list(cfg["food_seq"])
        self.decoy_walls = cfg["decoy_walls"]
        self.move_limit = cfg["move_limit"]
        self.grid = [row[:] for row in cfg["grid"]]

        self.next_food_idx = 0
        self.key_active = False
        self.level_won = False
        self.move_count = 0
        self._undo_stack = []

        for r in range(self.gh):
            for c in range(self.gw):
                if self.grid[r][c] == C_PLAYER:
                    self.grid[r][c] = C_EMPTY

        sx, sy = self._rng.choice(PLAYER_START_POSITIONS[self.level_index])
        self.player_x = sx
        self.player_y = sy
        self.grid[sy][sx] = C_PLAYER

        self.rebuild(self.current_level)

    def _save_undo(self) -> None:
        self._undo_stack.append(
            {
                "player_x": self.player_x,
                "player_y": self.player_y,
                "grid": [row[:] for row in self.grid],
                "next_food_idx": self.next_food_idx,
                "key_active": self.key_active,
            }
        )

    def _restore_undo(self) -> None:
        if not self._undo_stack:
            return
        state = self._undo_stack.pop()
        self.player_x = state["player_x"]
        self.player_y = state["player_y"]
        self.grid = state["grid"]
        self.next_food_idx = state["next_food_idx"]
        self.key_active = state["key_active"]

    def rebuild(self, level: Level) -> None:
        level.remove_all_sprites()
        idx = 0
        for r in range(self.gh):
            for c in range(self.gw):
                level.add_sprite(
                    Sprite(
                        pixels=solid(self.grid[r][c], self.cell),
                        name=f"c{idx}",
                        x=c * self.cell,
                        y=r * self.cell,
                        layer=1,
                    )
                )
                idx += 1

    def _try_move(self, dx: int, dy: int) -> str:
        nx = self.player_x + dx
        ny = self.player_y + dy

        if nx < 0 or ny < 0 or nx >= self.gw or ny >= self.gh:
            return "ok"

        tile = self.grid[ny][nx]

        if tile == C_WALL:
            return "ok"

        if tile == C_POISON:
            return "poison"

        if tile == C_GATE:
            return "ok"

        if tile == C_GATE_OPEN:
            self.grid[self.player_y][self.player_x] = C_EMPTY
            self.player_x = nx
            self.player_y = ny
            self.grid[ny][nx] = C_PLAYER
            self.level_won = True
            return "win"

        if tile == C_DECOY:
            self.grid[ny][nx] = C_EMPTY
            for wc, wr in self.decoy_walls.get((nx, ny), []):
                self.grid[wr][wc] = C_WALL

        if tile in self.food_seq:
            expected = self.food_seq[self.next_food_idx]
            if tile == expected:
                self.grid[ny][nx] = C_EMPTY
                self.next_food_idx += 1
                if self.next_food_idx == len(self.food_seq):
                    for r in range(self.gh):
                        for c in range(self.gw):
                            if self.grid[r][c] == C_GATE:
                                self.grid[r][c - 1] = C_KEY
                    self.key_active = True
            else:
                return "ok"

        if tile == C_KEY:
            self.grid[ny][nx] = C_EMPTY
            self.key_active = False
            for r in range(self.gh):
                for c in range(self.gw):
                    if self.grid[r][c] == C_GATE:
                        self.grid[r][c] = C_GATE_OPEN

        self.grid[self.player_y][self.player_x] = C_EMPTY
        self.player_x = nx
        self.player_y = ny
        self.grid[ny][nx] = C_PLAYER

        return "ok"

    def step(self) -> None:
        action = self.action

        if action.id == GameAction.RESET:
            self.complete_action()
            return

        if action.id == GameAction.ACTION7:
            self._restore_undo()
            self.move_count += 1
            self.rebuild(self.current_level)
            if self.move_count >= self.move_limit:
                self._lose_life()
                self.complete_action()
                return
            self.complete_action()
            return

        dx, dy = 0, 0
        if action.id == GameAction.ACTION1:
            dx, dy = 0, -1
        elif action.id == GameAction.ACTION2:
            dx, dy = 0, 1
        elif action.id == GameAction.ACTION3:
            dx, dy = -1, 0
        elif action.id == GameAction.ACTION4:
            dx, dy = 1, 0

        if dx != 0 or dy != 0:
            self._save_undo()
            result = self._try_move(dx, dy)

            if result == "poison":
                self.grid[self.player_y][self.player_x] = C_EMPTY
                self.player_x += dx
                self.player_y += dy
                self.rebuild(self.current_level)
                self._lose_life()
                self.complete_action()
                return

            if result == "win":
                self.move_count += 1
                self.rebuild(self.current_level)
                self.complete_action()
                self.next_level()
                return

            self.move_count += 1
            self.rebuild(self.current_level)

            if self.move_count >= self.move_limit:
                self._lose_life()
                self.complete_action()
                return
        else:
            self.rebuild(self.current_level)

        self.complete_action()


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    chunk = chunk_type + data
    return (
        struct.pack(">I", len(data))
        + chunk
        + struct.pack(">I", zlib.crc32(chunk) & 0xFFFFFFFF)
    )


def _encode_png(rgb: np.ndarray) -> bytes:
    h, w = rgb.shape[0], rgb.shape[1]
    raw = b""
    for y in range(h):
        raw += b"\x00" + rgb[y].tobytes()
    compressed = zlib.compress(raw)
    out = b"\x89PNG\r\n\x1a\n"
    out += _png_chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    out += _png_chunk(b"IDAT", compressed)
    out += _png_chunk(b"IEND", b"")
    return out


class PuzzleEnvironment:
    _ACTION_MAP: Dict[str, GameAction] = {
        "reset": GameAction.RESET,
        "up": GameAction.ACTION1,
        "down": GameAction.ACTION2,
        "left": GameAction.ACTION3,
        "right": GameAction.ACTION4,
        "undo": GameAction.ACTION7,
    }

    _VALID_ACTIONS = list(_ACTION_MAP.keys())

    def __init__(self, seed: int = 0) -> None:
        self._engine = Fe01(seed=seed)
        self._total_turns = 0
        self._done = False
        self._game_won = False
        self._game_over = False
        self._last_action_was_reset = False

    def reset(self) -> GameState:
        reset_input = ActionInput(id=GameAction.RESET)
        if self._game_won or self._last_action_was_reset:
            self._engine.perform_action(reset_input)
            self._engine.perform_action(reset_input)
        else:
            self._engine.perform_action(reset_input)
        self._last_action_was_reset = True
        self._done = False
        self._game_won = False
        self._game_over = False
        self._total_turns = 0
        return self._build_game_state()

    def get_actions(self) -> List[str]:
        if self._done or self._game_over:
            return ["reset"]
        return list(self._VALID_ACTIONS)

    def step(self, action: str) -> StepResult:
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
                info={"action": action, "error": "invalid_action"},
            )

        if self._done or self._game_over:
            return StepResult(
                state=self._build_game_state(done=self._done),
                reward=0.0,
                done=self._done,
                info={"action": action},
            )

        self._last_action_was_reset = False
        self._total_turns += 1

        level_before = self._engine.level_index

        frame = self._engine.perform_action(
            ActionInput(id=self._ACTION_MAP[action]), raw=True
        )
        state_name = frame.state.name if frame and frame.state else ""
        game_won = state_name == "WIN"
        game_over = state_name == "GAME_OVER"

        total_levels = len(self._engine._levels)
        level_reward = 1.0 / total_levels
        info: dict = {
            "action": action,
            "lives": self._engine._lives,
            "level": self._engine.level_index + 1,
            "moves": self._engine.move_count,
            "move_limit": self._engine.move_limit,
        }

        if game_won:
            self._done = True
            self._game_won = True
            info["reason"] = "game_complete"
            return StepResult(
                state=self._build_game_state(done=True),
                reward=level_reward,
                done=True,
                info=info,
            )

        if game_over:
            self._game_over = True
            info["reason"] = "game_over"
            return StepResult(
                state=self._build_game_state(),
                reward=0.0,
                done=False,
                info=info,
            )

        reward = 0.0
        if self._engine.level_index != level_before:
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
        index_grid = self._engine.camera.render(
            self._engine.current_level.get_sprites()
        )
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

    def is_done(self) -> bool:
        return self._done

    def _build_text_observation(self) -> str:
        engine = self._engine
        level_idx = engine.level_index
        total_levels = len(engine._levels)

        lines = []
        lines.append(
            f"Level {level_idx + 1}/{total_levels}"
            f" | Lives: {engine._lives}/{MAX_LIVES}"
            f" | Moves: {engine.move_count}/{engine.move_limit}"
            f" | Turn: {self._total_turns}"
        )
        lines.append("")

        for row in engine.grid:
            row_str = ""
            for cell in row:
                row_str += CELL_CHARS.get(cell, "?")
            lines.append(row_str)

        lines.append("")
        food_total = len(engine.food_seq)
        food_done = engine.next_food_idx
        lines.append(f"Food: {food_done}/{food_total}")
        if engine.key_active:
            lines.append("Key spawned - collect it to open the gate!")
        elif food_done >= food_total:
            lines.append("All food collected - find the key!")
        else:
            lines.append(f"Next food: #{food_done + 1}")
        lines.append("")
        lines.append(f"Actions: {', '.join(self._VALID_ACTIONS)}")
        return "\n".join(lines)

    def _build_game_state(
        self, done: bool = False
    ) -> GameState:
        engine = self._engine
        level_idx = engine.level_index
        total_levels = len(engine._levels)

        valid = self.get_actions() if not done else None

        image_bytes = None
        try:
            rgb = self.render()
            image_bytes = _encode_png(rgb)
        except Exception:
            pass

        return GameState(
            text_observation=self._build_text_observation(),
            image_observation=image_bytes,
            valid_actions=valid,
            turn=self._total_turns,
            metadata={
                "total_levels": total_levels,
                "level_index": level_idx,
                "levels_completed": getattr(engine, "_score", 0),
                "level": level_idx + 1,
                "lives": engine._lives,
                "moves": engine.move_count,
                "move_limit": engine.move_limit,
                "food_collected": engine.next_food_idx,
                "food_total": len(engine.food_seq),
                "key_active": engine.key_active,
                "game_over": getattr(getattr(engine, "_state", None), "name", "")
                == "GAME_OVER",
                "done": done,
                "info": {},
            },
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
