import random
import struct
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

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


sprites = {
    "tBg": Sprite(
        pixels=[
            [4, 4, 4, 4],
            [4, 3, 3, 4],
            [4, 3, 3, 4],
            [4, 4, 4, 4],
        ],
        name="tBg",
        visible=True,
        collidable=False,
        tags=["cel"],
        layer=-5,
    ),
    "tWl": Sprite(
        pixels=[
            [5, 5, 5, 5],
            [5, 5, 5, 5],
            [5, 5, 5, 5],
            [5, 5, 5, 5],
        ],
        name="tWl",
        visible=True,
        collidable=True,
        tags=["wal"],
        layer=-4,
    ),
    "eRd": Sprite(
        pixels=[
            [8, 8, 8, 8],
            [8, 0, 0, 8],
            [8, 0, 0, 8],
            [8, 8, 8, 8],
        ],
        name="eRd",
        visible=True,
        collidable=True,
        tags=["emt", "cR"],
        layer=0,
    ),
    "eBl": Sprite(
        pixels=[
            [9, 9, 9, 9],
            [9, 0, 0, 9],
            [9, 0, 0, 9],
            [9, 9, 9, 9],
        ],
        name="eBl",
        visible=True,
        collidable=True,
        tags=["emt", "cB"],
        layer=0,
    ),
    "eGn": Sprite(
        pixels=[
            [14, 14, 14, 14],
            [14, 0, 0, 14],
            [14, 0, 0, 14],
            [14, 14, 14, 14],
        ],
        name="eGn",
        visible=True,
        collidable=True,
        tags=["emt", "cG"],
        layer=0,
    ),
    "rRd": Sprite(
        pixels=[
            [4, 8, 8, 4],
            [8, 8, 8, 8],
            [8, 8, 8, 8],
            [4, 8, 8, 4],
        ],
        name="rRd",
        visible=True,
        collidable=True,
        tags=["rcv", "cR"],
        layer=0,
    ),
    "rBl": Sprite(
        pixels=[
            [4, 9, 9, 4],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [4, 9, 9, 4],
        ],
        name="rBl",
        visible=True,
        collidable=True,
        tags=["rcv", "cB"],
        layer=0,
    ),
    "rGn": Sprite(
        pixels=[
            [4, 14, 14, 4],
            [14, 14, 14, 14],
            [14, 14, 14, 14],
            [4, 14, 14, 4],
        ],
        name="rGn",
        visible=True,
        collidable=True,
        tags=["rcv", "cG"],
        layer=0,
    ),
    "rYl": Sprite(
        pixels=[
            [4, 11, 11, 4],
            [11, 11, 11, 11],
            [11, 11, 11, 11],
            [4, 11, 11, 4],
        ],
        name="rYl",
        visible=True,
        collidable=True,
        tags=["rcv", "cY"],
        layer=0,
    ),
    "bHz": Sprite(
        pixels=[
            [4, 4, 4, 4],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [4, 4, 4, 4],
        ],
        name="bHz",
        visible=True,
        collidable=True,
        tags=["brg", "pip"],
        layer=1,
    ),
    "bVt": Sprite(
        pixels=[
            [4, 2, 2, 4],
            [4, 2, 2, 4],
            [4, 2, 2, 4],
            [4, 2, 2, 4],
        ],
        name="bVt",
        visible=True,
        collidable=True,
        tags=["brg", "pip"],
        layer=1,
    ),
    "bTR": Sprite(
        pixels=[
            [4, 2, 2, 4],
            [4, 2, 2, 2],
            [4, 2, 2, 2],
            [4, 4, 4, 4],
        ],
        name="bTR",
        visible=True,
        collidable=True,
        tags=["brg", "pip"],
        layer=1,
    ),
    "bTL": Sprite(
        pixels=[
            [4, 2, 2, 4],
            [2, 2, 2, 4],
            [2, 2, 2, 4],
            [4, 4, 4, 4],
        ],
        name="bTL",
        visible=True,
        collidable=True,
        tags=["brg", "pip"],
        layer=1,
    ),
    "bBR": Sprite(
        pixels=[
            [4, 4, 4, 4],
            [4, 2, 2, 2],
            [4, 2, 2, 2],
            [4, 2, 2, 4],
        ],
        name="bBR",
        visible=True,
        collidable=True,
        tags=["brg", "pip"],
        layer=1,
    ),
    "bBL": Sprite(
        pixels=[
            [4, 4, 4, 4],
            [2, 2, 2, 4],
            [2, 2, 2, 4],
            [4, 2, 2, 4],
        ],
        name="bBL",
        visible=True,
        collidable=True,
        tags=["brg", "pip"],
        layer=1,
    ),
    "bCr": Sprite(
        pixels=[
            [4, 2, 2, 4],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [4, 2, 2, 4],
        ],
        name="bCr",
        visible=True,
        collidable=True,
        tags=["brg", "pip"],
        layer=1,
    ),
    "bTe": Sprite(
        pixels=[
            [4, 4, 4, 4],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [4, 2, 2, 4],
        ],
        name="bTe",
        visible=True,
        collidable=True,
        tags=["brg", "pip"],
        layer=1,
    ),
    "hLt": Sprite(
        pixels=[
            [12, 12, 12, 12],
            [12, -1, -1, 12],
            [12, -1, -1, 12],
            [12, 12, 12, 12],
        ],
        name="hLt",
        visible=False,
        collidable=False,
        tags=["hlt"],
        layer=5,
    ),
    "kCr": Sprite(
        pixels=[
            [11, 11, 11, 11],
            [11, -1, -1, 11],
            [11, -1, -1, 11],
            [11, 11, 11, 11],
        ],
        name="kCr",
        visible=False,
        collidable=False,
        tags=["kcr"],
        layer=6,
    ),
    "sOk": Sprite(
        pixels=[
            [14, 14, 14, 14],
            [14, -1, -1, 14],
            [14, -1, -1, 14],
            [14, 14, 14, 14],
        ],
        name="sOk",
        visible=False,
        collidable=False,
        tags=["sok"],
        layer=4,
    ),
}


BASE_CONNECTIONS = {
    "bHz": {1, 3},
    "bVt": {0, 2},
    "bTR": {0, 1},
    "bTL": {0, 3},
    "bBR": {2, 1},
    "bBL": {2, 3},
    "bCr": {0, 1, 2, 3},
    "bTe": {1, 2, 3},
}

SIDE_OFFSETS = {
    0: (-1, 0),
    1: (0, 1),
    2: (1, 0),
    3: (0, -1),
}

OPPOSITE_SIDE = {0: 2, 1: 3, 2: 0, 3: 1}

COLOR_MAP = {"cR": 8, "cB": 9, "cG": 14, "cY": 11}

MAX_LIVES = 3

CURSOR_START_INDICES = {
    0: [0],
    1: [0, 1],
    2: [0, 1, 2, 3],
    3: [0, 2, 4, 6],
}


def rotated_connections(base_name: str, rotation: int) -> set:
    base = BASE_CONNECTIONS.get(base_name, set())
    steps = (rotation // 90) % 4
    return {(s + steps) % 4 for s in base}


BACKGROUND_COLOR = 5
PADDING_COLOR = 4


class xqm(RenderableUserDisplay):
    def __init__(self, lpw: int):
        self.lpw = lpw
        self.olv = lpw
        self.lives = MAX_LIVES
        self.total_moves = 0
        self.show_info = False

    def lnu(self, edj: int) -> None:
        self.olv = max(0, min(edj, self.lpw))

    def czh(self) -> bool:
        self.total_moves += 1
        if self.olv > 0:
            self.olv -= 1
        return self.olv > 0

    def ivs(self) -> None:
        self.olv = self.lpw

    def reset_for_level(self, new_max: int) -> None:
        self.lpw = new_max
        self.olv = new_max
        self.lives = MAX_LIVES
        self.total_moves = 0
        self.show_info = False

    def lose_life(self) -> bool:
        self.lives -= 1
        if self.lives > 0:
            self.olv = self.lpw
            return True
        return False

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        if self.lpw == 0:
            return frame

        ratio = self.olv / self.lpw if self.lpw > 0 else 0
        filled = round(64 * ratio)
        for col in range(64):
            frame[0, col] = 14 if col < filled else 8

        for i in range(MAX_LIVES):
            col_start = 55 + i * 3
            color = 8 if i < self.lives else 4
            for r in range(2):
                for c in range(2):
                    if col_start + c < 64:
                        frame[62 + r, col_start + c] = color

        digits = str(self.total_moves)
        col = 1
        for ch in digits:
            d = int(ch)
            self._draw_tiny_digit(frame, 62, col, d)
            col += 4
            if col > 20:
                break

        if self.show_info:
            frame = self._draw_info_overlay(frame)

        return frame

    @staticmethod
    def _draw_tiny_digit(frame: np.ndarray, row: int, col: int, digit: int) -> None:
        FONT = {
            0: [0b111, 0b101, 0b111],
            1: [0b010, 0b110, 0b010],
            2: [0b111, 0b011, 0b111],
            3: [0b111, 0b011, 0b111],
            4: [0b101, 0b111, 0b001],
            5: [0b111, 0b110, 0b111],
            6: [0b111, 0b110, 0b111],
            7: [0b111, 0b001, 0b001],
            8: [0b111, 0b111, 0b111],
            9: [0b111, 0b111, 0b001],
        }
        bmp = FONT.get(digit, FONT[0])
        for dr, bits in enumerate(bmp):
            for dc in range(3):
                if bits & (1 << (2 - dc)):
                    r, c = row + dr, col + dc
                    if 0 <= r < 64 and 0 <= c < 64:
                        frame[r, c] = 1

    @staticmethod
    def _draw_info_overlay(frame: np.ndarray) -> np.ndarray:
        for r in range(64):
            for c in range(64):
                frame[r, c] = 4

        for c in range(2, 62):
            frame[2, c] = 7
            frame[61, c] = 7
        for r in range(2, 62):
            frame[r, 2] = 7
            frame[r, 61] = 7

        for r in range(5, 12):
            frame[r, 10] = 11
        frame[5, 13] = 11
        frame[6, 12] = 11
        frame[7, 11] = 11
        frame[8, 11] = 11
        frame[9, 12] = 11
        frame[10, 13] = 11
        for r in range(5, 12):
            frame[r, 16] = 11
            frame[r, 22] = 11
        frame[6, 17] = 11
        frame[7, 18] = 11
        frame[6, 21] = 11
        frame[7, 20] = 11
        frame[8, 19] = 11
        for r in range(5, 9):
            frame[r, 26] = 11
        for c in range(26, 31):
            frame[8, c] = 11
        for r in range(5, 12):
            frame[r, 29] = 11
        for c in range(33, 38):
            frame[5, c] = 11
        frame[6, 37] = 11
        frame[7, 36] = 11
        frame[8, 35] = 11
        frame[9, 35] = 11
        frame[10, 35] = 11
        frame[11, 35] = 11

        for c in range(8, 56):
            frame[16, c] = 2

        labels_row = 20
        for r in range(labels_row, labels_row + 3):
            for c in range(8, 11):
                frame[r, c] = 8
        for r in range(labels_row, labels_row + 3):
            for c in range(16, 19):
                frame[r, c] = 9
        for r in range(labels_row, labels_row + 3):
            for c in range(24, 27):
                frame[r, c] = 14
        for r in range(labels_row, labels_row + 3):
            for c in range(32, 35):
                frame[r, c] = 11

        frame[28, 12] = 0
        frame[29, 11] = 0
        frame[29, 12] = 0
        frame[29, 13] = 0
        frame[28, 21] = 0
        frame[28, 22] = 0
        frame[28, 23] = 0
        frame[29, 22] = 0
        frame[33, 11] = 0
        frame[34, 10] = 0
        frame[34, 11] = 0
        frame[34, 12] = 0
        frame[33, 22] = 0
        frame[34, 21] = 0
        frame[34, 22] = 0
        frame[34, 23] = 0

        for i in range(3):
            cx = 10 + i * 8
            for r in range(38, 40):
                for c in range(cx, cx + 2):
                    frame[r, c] = 8

        for c in range(8, 40):
            frame[44, c] = 14 if c < 30 else 8

        for c in range(10, 54):
            if c % 2 == 0:
                frame[50, c] = 7

        return frame


class Km47(ARCBaseGame):
    vrr: xqm

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self._rng = random.Random(seed)
        self.vrr = xqm(0)

        self._cursor_idx: int = 0
        self._cursor_active: bool = False
        self._show_intro: bool = True
        self._life_flash: int = 0
        self._history: List[Dict] = []

        game_levels = self._create_levels()

        super().__init__(
            game_id="km47",
            levels=game_levels,
            camera=Camera(0, 0, 64, 64, BACKGROUND_COLOR, PADDING_COLOR, [self.vrr]),
            debug=False,
            win_score=1,
            available_actions=[0, 1, 2, 3, 4, 5, 6, 7],
        )

    @staticmethod
    def _create_levels() -> List[Level]:
        return [
            Level(
                sprites=[
                    sprites["tBg"].clone().set_position(4, 8),
                    sprites["tBg"].clone().set_position(8, 8),
                    sprites["tBg"].clone().set_position(12, 8),
                    sprites["eRd"].clone().set_position(4, 8),
                    sprites["rRd"].clone().set_position(12, 8),
                    sprites["bHz"].clone().set_position(8, 8),
                    sprites["hLt"].clone().set_position(8, 8),
                    sprites["kCr"].clone().set_position(8, 8),
                    sprites["sOk"].clone().set_position(12, 8),
                ],
                grid_size=(20, 20),
                data={
                    "mvx": 6,
                    "grd": 3,
                    "off": (4, 8),
                    "csz": 4,
                    "pip_cells": [(1, 0)],
                },
                name="lvl1",
            ),
            Level(
                sprites=[
                    sprites["tBg"].clone().set_position(8, 4),
                    sprites["tBg"].clone().set_position(12, 4),
                    sprites["tBg"].clone().set_position(16, 4),
                    sprites["tBg"].clone().set_position(8, 8),
                    sprites["tBg"].clone().set_position(12, 8),
                    sprites["tBg"].clone().set_position(16, 8),
                    sprites["tBg"].clone().set_position(8, 12),
                    sprites["tBg"].clone().set_position(12, 12),
                    sprites["tBg"].clone().set_position(16, 12),
                    sprites["eRd"].clone().set_position(8, 4),
                    sprites["rRd"].clone().set_position(12, 12),
                    sprites["bBR"].clone().set_position(12, 4),
                    sprites["bVt"].clone().set_position(12, 8),
                    sprites["hLt"].clone().set_position(12, 4),
                    sprites["hLt"].clone().set_position(12, 8),
                    sprites["kCr"].clone().set_position(12, 4),
                    sprites["sOk"].clone().set_position(12, 12),
                ],
                grid_size=(24, 20),
                data={
                    "mvx": 10,
                    "grd": 3,
                    "off": (8, 4),
                    "csz": 4,
                    "pip_cells": [(1, 0), (1, 1)],
                },
                name="lvl2",
            ),
            Level(
                sprites=[
                    sprites["tBg"].clone().set_position(4, 4),
                    sprites["tBg"].clone().set_position(8, 4),
                    sprites["tBg"].clone().set_position(12, 4),
                    sprites["tBg"].clone().set_position(16, 4),
                    sprites["tBg"].clone().set_position(4, 8),
                    sprites["tBg"].clone().set_position(8, 8),
                    sprites["tBg"].clone().set_position(12, 8),
                    sprites["tBg"].clone().set_position(16, 8),
                    sprites["tBg"].clone().set_position(4, 12),
                    sprites["tBg"].clone().set_position(8, 12),
                    sprites["tBg"].clone().set_position(12, 12),
                    sprites["tBg"].clone().set_position(16, 12),
                    sprites["tBg"].clone().set_position(4, 16),
                    sprites["tBg"].clone().set_position(8, 16),
                    sprites["tBg"].clone().set_position(12, 16),
                    sprites["tBg"].clone().set_position(16, 16),
                    sprites["eRd"].clone().set_position(4, 4),
                    sprites["eBl"].clone().set_position(4, 12),
                    sprites["rRd"].clone().set_position(16, 4),
                    sprites["rBl"].clone().set_position(16, 12),
                    sprites["bHz"].clone().set_position(8, 4),
                    sprites["bHz"].clone().set_position(12, 4),
                    sprites["bVt"].clone().set_position(8, 12),
                    sprites["bVt"].clone().set_position(12, 12),
                    sprites["hLt"].clone().set_position(8, 4),
                    sprites["hLt"].clone().set_position(12, 4),
                    sprites["hLt"].clone().set_position(8, 12),
                    sprites["hLt"].clone().set_position(12, 12),
                    sprites["kCr"].clone().set_position(8, 4),
                    sprites["sOk"].clone().set_position(16, 4),
                    sprites["sOk"].clone().set_position(16, 12),
                ],
                grid_size=(24, 24),
                data={
                    "mvx": 20,
                    "grd": 4,
                    "off": (4, 4),
                    "csz": 4,
                    "pip_cells": [(1, 0), (2, 0), (1, 2), (2, 2)],
                },
                name="lvl3",
            ),
            Level(
                sprites=[
                    sprites["tBg"].clone().set_position(2, 2),
                    sprites["tBg"].clone().set_position(6, 2),
                    sprites["tBg"].clone().set_position(10, 2),
                    sprites["tBg"].clone().set_position(14, 2),
                    sprites["tBg"].clone().set_position(18, 2),
                    sprites["tBg"].clone().set_position(2, 6),
                    sprites["tBg"].clone().set_position(6, 6),
                    sprites["tBg"].clone().set_position(10, 6),
                    sprites["tBg"].clone().set_position(14, 6),
                    sprites["tBg"].clone().set_position(18, 6),
                    sprites["tBg"].clone().set_position(2, 10),
                    sprites["tBg"].clone().set_position(6, 10),
                    sprites["tBg"].clone().set_position(10, 10),
                    sprites["tBg"].clone().set_position(14, 10),
                    sprites["tBg"].clone().set_position(18, 10),
                    sprites["tBg"].clone().set_position(2, 14),
                    sprites["tBg"].clone().set_position(6, 14),
                    sprites["tBg"].clone().set_position(10, 14),
                    sprites["tBg"].clone().set_position(14, 14),
                    sprites["tBg"].clone().set_position(18, 14),
                    sprites["tBg"].clone().set_position(2, 18),
                    sprites["tBg"].clone().set_position(6, 18),
                    sprites["tBg"].clone().set_position(10, 18),
                    sprites["tBg"].clone().set_position(14, 18),
                    sprites["tBg"].clone().set_position(18, 18),
                    sprites["eRd"].clone().set_position(2, 10),
                    sprites["eBl"].clone().set_position(10, 2),
                    sprites["rRd"].clone().set_position(18, 10),
                    sprites["rBl"].clone().set_position(10, 18),
                    sprites["bBR"].clone().set_position(6, 6),
                    sprites["bHz"].clone().set_position(10, 6),
                    sprites["bTL"].clone().set_position(14, 6),
                    sprites["bVt"].clone().set_position(6, 10),
                    sprites["bCr"].clone().set_position(10, 10),
                    sprites["bVt"].clone().set_position(14, 10),
                    sprites["bTR"].clone().set_position(6, 14),
                    sprites["bHz"].clone().set_position(10, 14),
                    sprites["bBL"].clone().set_position(14, 14),
                    sprites["hLt"].clone().set_position(6, 6),
                    sprites["hLt"].clone().set_position(10, 6),
                    sprites["hLt"].clone().set_position(14, 6),
                    sprites["hLt"].clone().set_position(6, 10),
                    sprites["hLt"].clone().set_position(10, 10),
                    sprites["hLt"].clone().set_position(14, 10),
                    sprites["hLt"].clone().set_position(6, 14),
                    sprites["hLt"].clone().set_position(10, 14),
                    sprites["hLt"].clone().set_position(14, 14),
                    sprites["kCr"].clone().set_position(6, 6),
                    sprites["sOk"].clone().set_position(18, 10),
                    sprites["sOk"].clone().set_position(10, 18),
                ],
                grid_size=(24, 24),
                data={
                    "mvx": 30,
                    "grd": 5,
                    "off": (2, 2),
                    "csz": 4,
                    "pip_cells": [
                        (1, 1),
                        (2, 1),
                        (3, 1),
                        (1, 2),
                        (2, 2),
                        (3, 2),
                        (1, 3),
                        (2, 3),
                        (3, 3),
                    ],
                },
                name="lvl4",
            ),
        ]

    def handle_reset(self):
        self._rng = random.Random(self._seed)
        self.vrr.lives = MAX_LIVES
        self._history = []
        super().handle_reset()

    def kbn(self) -> None:
        mvx = self.current_level.get_data("mvx")
        self.vrr.reset_for_level(mvx)

    def on_set_level(self, level: Level) -> None:
        self.kbn()
        self._cursor_idx = 0
        self._cursor_active = False
        self._show_intro = False
        self._life_flash = 0
        self._history = []

        for h in self.current_level.get_sprites_by_tag("hlt"):
            h.set_visible(False)
        for s in self.current_level.get_sprites_by_tag("sok"):
            s.set_visible(False)
        for k in self.current_level.get_sprites_by_tag("kcr"):
            k.set_visible(False)

        self._update_success_overlays()

        cells = self._get_pip_cells()
        if cells:
            choices = CURSOR_START_INDICES.get(self.level_index, [0])
            valid = [i for i in choices if i < len(cells)]
            if valid:
                self._cursor_idx = self._rng.choice(valid)

    def _cell_to_px(self, col: int, row: int) -> Tuple[int, int]:
        off_x, off_y = self.current_level.get_data("off")
        csz = self.current_level.get_data("csz")
        return (off_x + col * csz, off_y + row * csz)

    def _get_pip_cells(self) -> List[Tuple[int, int]]:
        return self.current_level.get_data("pip_cells") or []

    def _find_pipe_at_cell(self, col: int, row: int) -> Optional[Sprite]:
        px, py = self._cell_to_px(col, row)
        for spr in self.current_level.get_sprites_by_tag("pip"):
            if spr.x == px and spr.y == py:
                return spr
        return None

    def _update_cursor_sprite(self) -> None:
        cells = self._get_pip_cells()
        if not cells or self._cursor_idx >= len(cells):
            return
        col, row = cells[self._cursor_idx]
        px, py = self._cell_to_px(col, row)

        for k in self.current_level.get_sprites_by_tag("kcr"):
            k.set_position(px, py)
            k.set_visible(self._cursor_active)

    def _move_cursor(self, dc: int, dr: int) -> None:
        cells = self._get_pip_cells()
        if not cells:
            return

        self._cursor_active = True
        cur_col, cur_row = cells[self._cursor_idx]
        target_col = cur_col + dc
        target_row = cur_row + dr

        best_idx = self._cursor_idx
        best_dist = 9999
        for i, (cc, cr) in enumerate(cells):
            if dc != 0 and dr == 0:
                if (dc > 0 and cc > cur_col) or (dc < 0 and cc < cur_col):
                    dist = abs(cc - target_col) + abs(cr - cur_row) * 10
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = i
            elif dr != 0 and dc == 0:
                if (dr > 0 and cr > cur_row) or (dr < 0 and cr < cur_row):
                    dist = abs(cr - target_row) + abs(cc - cur_col) * 10
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = i

        self._cursor_idx = best_idx
        self._update_cursor_sprite()

    def _build_grid(self) -> dict:
        off_x, off_y = self.current_level.get_data("off")
        csz = self.current_level.get_data("csz")
        grid = {}

        for spr in self.current_level.get_sprites_by_tag("pip"):
            col = (spr.x - off_x) // csz
            row = (spr.y - off_y) // csz
            conns = rotated_connections(spr.name, spr.rotation)
            grid[(col, row)] = {"sprite": spr, "conns": conns, "tags": spr.tags or []}

        for spr in self.current_level.get_sprites_by_tag("emt"):
            col = (spr.x - off_x) // csz
            row = (spr.y - off_y) // csz
            grid[(col, row)] = {
                "sprite": spr,
                "conns": {0, 1, 2, 3},
                "tags": spr.tags or [],
            }

        for spr in self.current_level.get_sprites_by_tag("rcv"):
            col = (spr.x - off_x) // csz
            row = (spr.y - off_y) // csz
            grid[(col, row)] = {
                "sprite": spr,
                "conns": {0, 1, 2, 3},
                "tags": spr.tags or [],
            }

        return grid

    def _flood_from_emitter(self, grid: dict, start_col: int, start_row: int) -> set:
        visited = set()
        queue = [(start_col, start_row)]
        visited.add((start_col, start_row))

        while queue:
            cx, cy = queue.pop(0)
            cell = grid.get((cx, cy))
            if cell is None:
                continue
            for side in cell["conns"]:
                dr, dc = SIDE_OFFSETS[side]
                nx, ny = cx + dc, cy + dr
                if (nx, ny) in visited:
                    continue
                nbr = grid.get((nx, ny))
                if nbr is None:
                    continue
                opp = OPPOSITE_SIDE[side]
                if opp in nbr["conns"]:
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        return visited

    def _check_all_connected(self) -> bool:
        grid = self._build_grid()
        emitters = self.current_level.get_sprites_by_tag("emt")
        receivers = self.current_level.get_sprites_by_tag("rcv")
        off_x, off_y = self.current_level.get_data("off")
        csz = self.current_level.get_data("csz")

        for emt in emitters:
            ecol = (emt.x - off_x) // csz
            erow = (emt.y - off_y) // csz
            ctag = None
            for t in emt.tags or []:
                if t.startswith("c"):
                    ctag = t
                    break
            if ctag is None:
                continue
            reached = self._flood_from_emitter(grid, ecol, erow)
            found = False
            for rcv in receivers:
                if ctag in (rcv.tags or []):
                    rcol = (rcv.x - off_x) // csz
                    rrow = (rcv.y - off_y) // csz
                    if (rcol, rrow) in reached:
                        found = True
                        break
            if not found:
                return False
        return True

    def _update_success_overlays(self) -> None:
        grid = self._build_grid()
        off_x, off_y = self.current_level.get_data("off")
        csz = self.current_level.get_data("csz")
        emitters = self.current_level.get_sprites_by_tag("emt")
        receivers = self.current_level.get_sprites_by_tag("rcv")
        overlays = self.current_level.get_sprites_by_tag("sok")

        connected_rcv_positions = set()
        for emt in emitters:
            ecol = (emt.x - off_x) // csz
            erow = (emt.y - off_y) // csz
            ctag = None
            for t in emt.tags or []:
                if t.startswith("c"):
                    ctag = t
                    break
            if ctag is None:
                continue
            reached = self._flood_from_emitter(grid, ecol, erow)
            for rcv in receivers:
                if ctag in (rcv.tags or []):
                    rcol = (rcv.x - off_x) // csz
                    rrow = (rcv.y - off_y) // csz
                    if (rcol, rrow) in reached:
                        connected_rcv_positions.add((rcv.x, rcv.y))

        for ov in overlays:
            ov.set_visible((ov.x, ov.y) in connected_rcv_positions)

    def _save_state(self, pipe: Sprite) -> None:
        self._history.append(
            {
                "pipe_x": pipe.x,
                "pipe_y": pipe.y,
                "pipe_rotation": pipe.rotation,
                "cursor_idx": self._cursor_idx,
                "cursor_active": self._cursor_active,
                "moves_olv": self.vrr.olv,
                "moves_total": self.vrr.total_moves,
                "lives": self.vrr.lives,
            }
        )

    def _undo(self) -> None:
        if not self._history:
            return
        snap = self._history.pop()
        px, py = snap["pipe_x"], snap["pipe_y"]
        for spr in self.current_level.get_sprites_by_tag("pip"):
            if spr.x == px and spr.y == py:
                spr.set_rotation(snap["pipe_rotation"])
                break
        self._cursor_idx = snap["cursor_idx"]
        self._cursor_active = snap["cursor_active"]
        self.vrr.olv = snap["moves_olv"]
        self.vrr.total_moves = snap["moves_total"]
        self.vrr.lives = snap["lives"]
        self._update_cursor_sprite()
        self._update_success_overlays()
        for h in self.current_level.get_sprites_by_tag("hlt"):
            h.set_visible(False)

    def _rotate_pipe(self, pipe: Sprite) -> None:
        self._save_state(pipe)
        new_rot = (pipe.rotation + 90) % 360
        pipe.set_rotation(new_rot)

        for h in self.current_level.get_sprites_by_tag("hlt"):
            h.set_visible(h.x == pipe.x and h.y == pipe.y)

        has_moves = self.vrr.czh()

        self._update_success_overlays()

        if self._check_all_connected():
            self.next_level()
            self.complete_action()
            return

        if not has_moves:
            still_alive = self.vrr.lose_life()
            if not still_alive:
                self.lose()
                self.complete_action()
                return
            else:
                self._life_flash = 3

        self.complete_action()

    def _consume_move(self) -> None:
        has_moves = self.vrr.czh()
        if not has_moves:
            still_alive = self.vrr.lose_life()
            if not still_alive:
                self.lose()
                self.complete_action()
                return
            else:
                self._life_flash = 3
        self.complete_action()

    def step(self) -> None:
        action_id = self.action.id.value

        if self._show_intro:
            self.vrr.show_info = True
            self._show_intro = False
            self.complete_action()
            return

        if self.vrr.show_info:
            self.vrr.show_info = False
            self.complete_action()
            return

        if self._life_flash > 0:
            self._life_flash -= 1
            self.complete_action()
            return

        if action_id == 7:
            self._undo()
            self.complete_action()
            return

        if action_id == 1:
            self._move_cursor(0, -1)
            self._consume_move()
            return

        if action_id == 2:
            self._move_cursor(0, 1)
            self._consume_move()
            return

        if action_id == 3:
            self._move_cursor(-1, 0)
            self._consume_move()
            return

        if action_id == 4:
            self._move_cursor(1, 0)
            self._consume_move()
            return

        if action_id == 5:
            if self._cursor_active:
                cells = self._get_pip_cells()
                if cells and self._cursor_idx < len(cells):
                    col, row = cells[self._cursor_idx]
                    pipe = self._find_pipe_at_cell(col, row)
                    if pipe is not None:
                        self._rotate_pipe(pipe)
                        return
            self.complete_action()
            return

        if action_id == 6:
            owP = self.action.data.get("x", 0)
            sSK = self.action.data.get("y", 0)

            gcm = self.camera.display_to_grid(owP, sSK)
            if gcm:
                biu, ryv = gcm

                clicked_pipe = None
                for spr in self.current_level.get_sprites_by_tag("pip"):
                    if (
                        spr.x <= biu < spr.x + spr.width
                        and spr.y <= ryv < spr.y + spr.height
                    ):
                        clicked_pipe = spr
                        break

                if clicked_pipe is not None:
                    off_x, off_y = self.current_level.get_data("off")
                    csz = self.current_level.get_data("csz")
                    click_col = (clicked_pipe.x - off_x) // csz
                    click_row = (clicked_pipe.y - off_y) // csz
                    cells = self._get_pip_cells()
                    for i, (cc, cr) in enumerate(cells):
                        if cc == click_col and cr == click_row:
                            self._cursor_idx = i
                            self._cursor_active = True
                            self._update_cursor_sprite()
                            break

                    self._rotate_pipe(clicked_pipe)
                    return
                else:
                    for h in self.current_level.get_sprites_by_tag("hlt"):
                        h.set_visible(False)

        self.complete_action()


class PuzzleEnvironment:
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

    ACTION_MAP = {
        "up": 1,
        "down": 2,
        "left": 3,
        "right": 4,
        "select": 5,
        "click": 6,
        "reset": 0,
        "undo": 7,
    }

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self._engine = Km47(seed=seed)
        self._total_turns = 0
        self._last_action_was_reset = False
        self._prev_score = 0

    def _build_text_obs(self) -> str:
        e = self._engine
        off_x, off_y = e.current_level.get_data("off")
        csz = e.current_level.get_data("csz")
        grd = e.current_level.get_data("grd")

        grid = [["." for _ in range(grd)] for _ in range(grd)]

        for spr in e.current_level.get_sprites_by_tag("emt"):
            col = (spr.x - off_x) // csz
            row = (spr.y - off_y) // csz
            if 0 <= col < grd and 0 <= row < grd:
                for t in spr.tags or []:
                    if t.startswith("c"):
                        grid[row][col] = t[1].upper()

        for spr in e.current_level.get_sprites_by_tag("rcv"):
            col = (spr.x - off_x) // csz
            row = (spr.y - off_y) // csz
            if 0 <= col < grd and 0 <= row < grd:
                for t in spr.tags or []:
                    if t.startswith("c"):
                        grid[row][col] = t[1].lower()

        for spr in e.current_level.get_sprites_by_tag("pip"):
            col = (spr.x - off_x) // csz
            row = (spr.y - off_y) // csz
            if 0 <= col < grd and 0 <= row < grd:
                conns = rotated_connections(spr.name, spr.rotation)
                dirs = ""
                if 0 in conns:
                    dirs += "U"
                if 1 in conns:
                    dirs += "R"
                if 2 in conns:
                    dirs += "D"
                if 3 in conns:
                    dirs += "L"
                grid[row][col] = dirs

        cells = e._get_pip_cells()
        if e._cursor_active and cells and e._cursor_idx < len(cells):
            cc, cr = cells[e._cursor_idx]
            if 0 <= cc < grd and 0 <= cr < grd:
                grid[cr][cc] = "[" + grid[cr][cc] + "]"

        header = (
            f"Level:{e.level_index + 1}/{len(e._levels)} "
            f"Lives:{e.vrr.lives} "
            f"Moves:{e.vrr.olv}/{e.vrr.lpw}"
        )

        grid_text = "\n".join(" ".join(str(c).ljust(4) for c in row) for row in grid)
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
                "lives": e.vrr.lives,
                "moves_remaining": e.vrr.olv,
                "moves_max": e.vrr.lpw,
                "done": done,
                "info": info or {},
            },
        )

    def reset(self) -> GameState:
        e = self._engine
        reset_input = ActionInput(id=GameAction.RESET)
        e.perform_action(reset_input)
        self._total_turns = 0
        self._prev_score = getattr(e, "_score", 0)
        self._last_action_was_reset = True
        return self._build_game_state()

    def get_actions(self) -> List[str]:
        e = self._engine
        state_name = getattr(e._state, "name", "")
        if state_name in ("WIN", "GAME_OVER"):
            return ["reset"]
        return ["reset", "up", "down", "left", "right", "select", "click", "undo"]

    def is_done(self) -> bool:
        e = self._engine
        state_name = getattr(e._state, "name", "")
        return state_name == "WIN"

    def step(self, action: str) -> StepResult:
        e = self._engine

        if action == "reset":
            if self._last_action_was_reset:
                self._engine = Km47(seed=self._seed)
            state = self.reset()
            return StepResult(
                state=state, reward=0.0, done=False, info={"action": "reset"}
            )

        parts = action.split()
        base_action = parts[0] if parts else action
        if base_action not in self.ACTION_MAP:
            return StepResult(
                state=self._build_game_state(),
                reward=0.0,
                done=False,
                info={"action": action, "error": "invalid_action"},
            )

        self._last_action_was_reset = False
        self._total_turns += 1

        game_action_id = self.ACTION_MAP[base_action]
        action_map = {
            1: GameAction.ACTION1,
            2: GameAction.ACTION2,
            3: GameAction.ACTION3,
            4: GameAction.ACTION4,
            5: GameAction.ACTION5,
            6: GameAction.ACTION6,
            7: GameAction.ACTION7,
        }
        game_action = action_map[game_action_id]

        x_val = None
        y_val = None
        if base_action == "click" and len(parts) >= 3:
            try:
                x_val = int(float(parts[1]))
                y_val = int(float(parts[2]))
            except (ValueError, OverflowError):
                x_val = 0
                y_val = 0

        action_input = ActionInput(id=game_action, x=x_val, y=y_val)

        prev_score = self._prev_score
        frame = e.perform_action(action_input, raw=True)
        new_score = frame.levels_completed
        self._prev_score = new_score

        total_levels = len(e._levels)
        if new_score > prev_score and total_levels > 0:
            reward = new_score / total_levels
        else:
            reward = 0.0
        done = frame.state.name == "WIN"

        info: Dict = {"action": action}
        if frame.state.name == "WIN":
            info["reason"] = "game_complete"
        elif frame.state.name == "GAME_OVER":
            info["reason"] = "game_over"

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
        "render_fps": 10,
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
            raise ValueError(
                f"Unsupported render_mode '{render_mode}'. Supported: {self.metadata['render_modes']}"
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
