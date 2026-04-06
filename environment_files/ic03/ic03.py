from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import random
import struct
import zlib

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from arcengine import (
    ActionInput,
    ARCBaseGame,
    Camera,
    GameAction,
    Level,
    Sprite,
)
from arcengine.enums import BlockingMode, GameState as EngineGameState


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


ARC_PALETTE = [
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
]
ARC_PALETTE = np.array(ARC_PALETTE, dtype=np.uint8)

TILE: int = 4
BOARD_W: int = 15
BOARD_H: int = 15
HUD_ROWS: int = 1
INPUT_COL: int = 1
OUTPUT_COL: int = 11
BUTTON_ROW: int = 11
SUBMIT_COL: int = 6
UNDO_COL: int = 7
MAX_LIVES: int = 3
HISTORY_LIMIT: int = 48
HUD_BAR_WIDTH: int = BOARD_W - (MAX_LIVES + 1)

_CIRCUIT_W: int = 12
_CIRCUIT_H: int = 12
COL_OFFSET: int = (BOARD_W - _CIRCUIT_W) // 2
ROW_OFFSET: int = (BOARD_H - _CIRCUIT_H) // 2

TURN_LIMITS: dict[str, int] = {"easy": 220, "standard": 180, "hard": 150}
PROBE_SPANS: dict[str, int] = {"easy": 3, "standard": 2, "hard": 1}
PROBE_COOLDOWNS: dict[str, int] = {"easy": 1, "standard": 2, "hard": 3}
LEVEL_MOVE_LIMITS: dict[int, int] = {0: 30, 1: 45, 2: 60, 3: 80}
SUBMIT_PENALTIES: dict[int, int] = {0: 1, 1: 2, 2: 3, 3: 4}


@dataclass(frozen=True)
class NodeSpec:
    sources: tuple[str, str]
    gate: str


@dataclass(frozen=True)
class LevelConfig:
    index: int
    input_rows: tuple[int, ...]
    output_rows: tuple[int, ...]
    anchor_positions: tuple[tuple[int, int], ...]
    always_visible: tuple[int, ...]
    node_specs: tuple[NodeSpec, ...]
    output_specs: tuple[NodeSpec, ...]
    initial_inputs: tuple[int, ...]
    target_outputs: tuple[int, ...]


@dataclass(frozen=True)
class GateCell:
    row: int
    col: int
    gate: str
    layer: int


@dataclass(frozen=True)
class WirePath:
    waypoints: tuple[tuple[int, int], ...]
    is_memory: bool = False


@dataclass(frozen=True)
class CircuitLayout:
    gates: tuple[GateCell, ...]
    paths: tuple[WirePath, ...]


HARDCODED_LEVELS: tuple[LevelConfig, ...] = (
    LevelConfig(
        index=0,
        input_rows=(2, 5, 8),
        output_rows=(2, 5, 8),
        anchor_positions=((2, 3), (5, 3), (8, 4)),
        always_visible=(0, 1, 2),
        node_specs=(),
        output_specs=(
            NodeSpec(sources=("i0", "i1"), gate="AND"),
            NodeSpec(sources=("i1", "i2"), gate="XOR"),
            NodeSpec(sources=("i0", "i2"), gate="OR"),
        ),
        initial_inputs=(0, 0, 0),
        target_outputs=(1, 1, 1),
    ),
    LevelConfig(
        index=1,
        input_rows=(1, 3, 7, 9),
        output_rows=(2, 5, 8),
        anchor_positions=((2, 6), (8, 6)),
        always_visible=(0, 2),
        node_specs=(
            NodeSpec(sources=("i0", "i1"), gate="AND"),
            NodeSpec(sources=("i2", "i3"), gate="NAND"),
        ),
        output_specs=(
            NodeSpec(sources=("n0", "i2"), gate="OR"),
            NodeSpec(sources=("n0", "n1"), gate="XOR"),
            NodeSpec(sources=("n1", "i3"), gate="AND"),
        ),
        initial_inputs=(0, 0, 0, 0),
        target_outputs=(1, 0, 1),
    ),
    LevelConfig(
        index=2,
        input_rows=(1, 4, 6, 9),
        output_rows=(2, 5, 8),
        anchor_positions=((2, 6), (8, 6)),
        always_visible=(1,),
        node_specs=(
            NodeSpec(sources=("i0", "i3"), gate="XOR"),
            NodeSpec(sources=("i1", "i2"), gate="OR"),
        ),
        output_specs=(
            NodeSpec(sources=("n0", "n1"), gate="AND"),
            NodeSpec(sources=("i3", "i0"), gate="XOR"),
            NodeSpec(sources=("n0", "i2"), gate="NOR"),
        ),
        initial_inputs=(0, 0, 0, 0),
        target_outputs=(1, 1, 0),
    ),
    LevelConfig(
        index=3,
        input_rows=(1, 3, 5, 7, 9),
        output_rows=(1, 4, 7, 10),
        anchor_positions=((2, 6), (6, 6), (8, 6)),
        always_visible=(0, 3),
        node_specs=(
            NodeSpec(sources=("i0", "i1"), gate="NOR"),
            NodeSpec(sources=("i2", "i4"), gate="XOR"),
            NodeSpec(sources=("i3", "i4"), gate="AND"),
        ),
        output_specs=(
            NodeSpec(sources=("n0", "i3"), gate="OR"),
            NodeSpec(sources=("n1", "n2"), gate="XOR"),
            NodeSpec(sources=("n0", "n1"), gate="NOR"),
            NodeSpec(sources=("n2", "i4"), gate="AND"),
        ),
        initial_inputs=(0, 0, 0, 0, 0),
        target_outputs=(1, 1, 0, 1),
    ),
)

CIRCUIT_LAYOUTS: tuple[CircuitLayout, ...] = (
    CircuitLayout(
        gates=(
            GateCell(row=2, col=6, gate="AND", layer=2),
            GateCell(row=5, col=6, gate="XOR", layer=2),
            GateCell(row=8, col=6, gate="OR", layer=2),
        ),
        paths=(
            WirePath(waypoints=((2, 2), (2, 3), (2, 4), (2, 5))),
            WirePath(
                waypoints=((5, 2), (5, 3), (4, 3), (3, 3), (2, 3), (2, 4), (2, 5))
            ),
            WirePath(waypoints=((5, 2), (5, 3), (5, 4), (5, 5))),
            WirePath(
                waypoints=((8, 2), (8, 3), (8, 4), (7, 4), (6, 4), (5, 4), (5, 5))
            ),
            WirePath(
                waypoints=(
                    (2, 2),
                    (2, 3),
                    (3, 3),
                    (4, 3),
                    (5, 3),
                    (6, 3),
                    (7, 3),
                    (8, 3),
                    (8, 4),
                    (8, 5),
                )
            ),
            WirePath(waypoints=((8, 2), (8, 3), (8, 4), (8, 5))),
            WirePath(waypoints=((2, 7), (2, 8), (2, 9), (2, 10))),
            WirePath(waypoints=((5, 7), (5, 8), (5, 9), (5, 10))),
            WirePath(waypoints=((8, 7), (8, 8), (8, 9), (8, 10))),
        ),
    ),
    CircuitLayout(
        gates=(
            GateCell(row=2, col=4, gate="AND", layer=1),
            GateCell(row=8, col=4, gate="NAND", layer=1),
            GateCell(row=2, col=8, gate="OR", layer=2),
            GateCell(row=5, col=8, gate="XOR", layer=2),
            GateCell(row=8, col=8, gate="AND", layer=2),
        ),
        paths=(
            WirePath(waypoints=((1, 2), (1, 3), (2, 3))),
            WirePath(waypoints=((3, 2), (3, 3), (2, 3))),
            WirePath(waypoints=((7, 2), (7, 3), (8, 3))),
            WirePath(waypoints=((9, 2), (9, 3), (8, 3))),
            WirePath(waypoints=((2, 5), (2, 6), (2, 7))),
            WirePath(waypoints=((2, 5), (2, 6), (3, 6), (4, 6), (5, 6), (5, 7))),
            WirePath(waypoints=((8, 5), (8, 6), (7, 6), (6, 6), (5, 6), (5, 7))),
            WirePath(waypoints=((8, 5), (8, 6), (8, 7))),
            WirePath(
                waypoints=(
                    (7, 2),
                    (7, 3),
                    (7, 4),
                    (7, 5),
                    (7, 6),
                    (6, 6),
                    (5, 6),
                    (4, 6),
                    (3, 6),
                    (2, 6),
                    (2, 7),
                )
            ),
            WirePath(
                waypoints=(
                    (9, 2),
                    (9, 3),
                    (9, 4),
                    (9, 5),
                    (9, 6),
                    (9, 7),
                    (8, 7),
                )
            ),
            WirePath(waypoints=((2, 9), (2, 10))),
            WirePath(waypoints=((5, 9), (5, 10))),
            WirePath(waypoints=((8, 9), (8, 10))),
        ),
    ),
    CircuitLayout(
        gates=(
            GateCell(row=2, col=4, gate="XOR", layer=1),
            GateCell(row=5, col=4, gate="OR", layer=1),
            GateCell(row=2, col=8, gate="AND", layer=2),
            GateCell(row=5, col=8, gate="XOR", layer=2),
            GateCell(row=8, col=8, gate="NOR", layer=2),
        ),
        paths=(
            WirePath(waypoints=((1, 2), (1, 3), (2, 3))),
            WirePath(
                waypoints=(
                    (9, 2),
                    (9, 1),
                    (9, 0),
                    (8, 0),
                    (7, 0),
                    (6, 0),
                    (5, 0),
                    (4, 0),
                    (3, 0),
                    (2, 0),
                    (2, 1),
                    (2, 2),
                    (2, 3),
                )
            ),
            WirePath(waypoints=((4, 2), (4, 3), (5, 3))),
            WirePath(waypoints=((6, 2), (6, 3), (5, 3))),
            WirePath(waypoints=((2, 5), (2, 6), (2, 7))),
            WirePath(waypoints=((5, 5), (5, 6), (4, 6), (3, 6), (2, 6), (2, 7))),
            WirePath(
                waypoints=(
                    (9, 2),
                    (9, 1),
                    (9, 0),
                    (8, 0),
                    (7, 0),
                    (6, 0),
                    (5, 0),
                    (4, 0),
                    (3, 0),
                    (3, 1),
                    (3, 2),
                    (3, 3),
                    (3, 4),
                    (3, 5),
                    (3, 6),
                    (3, 7),
                    (4, 7),
                    (5, 7),
                )
            ),
            WirePath(
                waypoints=(
                    (1, 2),
                    (1, 3),
                    (1, 4),
                    (1, 5),
                    (1, 6),
                    (1, 7),
                    (2, 7),
                    (3, 7),
                    (4, 7),
                    (5, 7),
                )
            ),
            WirePath(
                waypoints=(
                    (2, 5),
                    (2, 6),
                    (3, 6),
                    (4, 6),
                    (5, 6),
                    (6, 6),
                    (7, 6),
                    (8, 6),
                    (8, 7),
                )
            ),
            WirePath(
                waypoints=(
                    (6, 2),
                    (6, 3),
                    (6, 4),
                    (6, 5),
                    (6, 6),
                    (7, 6),
                    (8, 6),
                    (8, 7),
                )
            ),
            WirePath(waypoints=((2, 9), (2, 10))),
            WirePath(waypoints=((5, 9), (5, 10))),
            WirePath(waypoints=((8, 9), (8, 10))),
        ),
    ),
    CircuitLayout(
        gates=(
            GateCell(row=2, col=4, gate="NOR", layer=1),
            GateCell(row=6, col=4, gate="XOR", layer=1),
            GateCell(row=8, col=4, gate="AND", layer=1),
            GateCell(row=1, col=8, gate="OR", layer=2),
            GateCell(row=4, col=8, gate="XOR", layer=2),
            GateCell(row=7, col=8, gate="NOR", layer=2),
            GateCell(row=10, col=8, gate="AND", layer=2),
        ),
        paths=(
            WirePath(waypoints=((1, 2), (1, 3), (2, 3))),
            WirePath(waypoints=((3, 2), (3, 3), (2, 3))),
            WirePath(waypoints=((5, 2), (5, 3), (6, 3))),
            WirePath(
                waypoints=(
                    (9, 2),
                    (9, 1),
                    (9, 0),
                    (8, 0),
                    (7, 0),
                    (6, 0),
                    (6, 1),
                    (6, 2),
                    (6, 3),
                )
            ),
            WirePath(waypoints=((7, 2), (7, 3), (8, 3))),
            WirePath(waypoints=((9, 2), (9, 3), (8, 3))),
            WirePath(waypoints=((2, 5), (2, 6), (1, 6), (1, 7))),
            WirePath(
                waypoints=(
                    (2, 5),
                    (2, 6),
                    (3, 6),
                    (4, 6),
                    (5, 6),
                    (6, 6),
                    (7, 6),
                    (7, 7),
                )
            ),
            WirePath(waypoints=((6, 5), (6, 6), (5, 6), (4, 6), (4, 7))),
            WirePath(waypoints=((6, 5), (6, 6), (7, 6), (7, 7))),
            WirePath(
                waypoints=(
                    (8, 5),
                    (8, 6),
                    (7, 6),
                    (6, 6),
                    (5, 6),
                    (4, 6),
                    (4, 7),
                )
            ),
            WirePath(waypoints=((8, 5), (8, 6), (9, 6), (10, 6), (10, 7))),
            WirePath(
                waypoints=(
                    (7, 2),
                    (7, 3),
                    (7, 4),
                    (7, 5),
                    (6, 5),
                    (5, 5),
                    (4, 5),
                    (3, 5),
                    (2, 5),
                    (1, 5),
                    (1, 6),
                    (1, 7),
                )
            ),
            WirePath(
                waypoints=(
                    (9, 2),
                    (9, 3),
                    (9, 4),
                    (9, 5),
                    (9, 6),
                    (10, 6),
                    (10, 7),
                )
            ),
            WirePath(waypoints=((1, 9), (1, 10))),
            WirePath(waypoints=((4, 9), (4, 10))),
            WirePath(waypoints=((7, 9), (7, 10))),
            WirePath(waypoints=((10, 9), (10, 10))),
        ),
    ),
)

_BG_PX: list[list[int]] = [
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
]

PIXELS: dict[str, list[list[int]]] = {
    "bg": [row[:] for row in _BG_PX],
    "input_off": [
        [0, 1, 1, 0],
        [1, 0, 0, 1],
        [1, 0, 0, 1],
        [0, 1, 1, 0],
    ],
    "input_on": [
        [0, 3, 3, 0],
        [3, 3, 3, 3],
        [3, 3, 3, 3],
        [0, 3, 3, 0],
    ],
    "output_off": [
        [0, 2, 2, 0],
        [2, 0, 0, 2],
        [2, 0, 0, 2],
        [0, 2, 2, 0],
    ],
    "output_on": [
        [0, 4, 4, 0],
        [4, 7, 7, 4],
        [4, 7, 7, 4],
        [0, 4, 4, 0],
    ],
    "output_hidden": [
        [0, 5, 5, 0],
        [5, 5, 5, 5],
        [5, 5, 5, 5],
        [0, 5, 5, 0],
    ],
    "target_off": [
        [0, 2, 2, 0],
        [2, 0, 0, 2],
        [2, 0, 0, 2],
        [0, 2, 2, 0],
    ],
    "target_on": [
        [0, 4, 4, 0],
        [4, 7, 7, 4],
        [4, 7, 7, 4],
        [0, 4, 4, 0],
    ],
    "submit_locked": [
        [5, 5, 5, 5],
        [5, 9, 9, 5],
        [5, 9, 9, 5],
        [5, 5, 5, 5],
    ],
    "submit_ready": [
        [3, 3, 3, 3],
        [3, 7, 7, 3],
        [3, 7, 7, 3],
        [3, 3, 3, 3],
    ],
    "undo_btn": [
        [0, 1, 1, 0],
        [1, 8, 8, 1],
        [1, 8, 8, 1],
        [0, 1, 1, 0],
    ],
    "cursor": [
        [4, 4, 4, 4],
        [4, -1, -1, 4],
        [4, -1, -1, 4],
        [4, 4, 4, 4],
    ],
    "anchor": [
        [0, 8, 8, 0],
        [8, 5, 5, 8],
        [8, 5, 5, 8],
        [0, 8, 8, 0],
    ],
    "gate_AND": [
        [5, 5, 5, 5],
        [5, 3, 3, 5],
        [5, 3, 3, 5],
        [5, 5, 5, 5],
    ],
    "gate_OR": [
        [5, 5, 5, 5],
        [5, 1, 1, 5],
        [5, 1, 1, 5],
        [5, 5, 5, 5],
    ],
    "gate_XOR": [
        [5, 5, 5, 5],
        [5, 4, 4, 5],
        [5, 4, 4, 5],
        [5, 5, 5, 5],
    ],
    "gate_NAND": [
        [5, 5, 5, 5],
        [5, 3, 3, 5],
        [5, 3, 3, 4],
        [5, 5, 5, 5],
    ],
    "gate_NOR": [
        [5, 5, 5, 5],
        [5, 1, 1, 5],
        [5, 1, 1, 4],
        [5, 5, 5, 5],
    ],
    "wire_EW": [
        [0, 0, 0, 0],
        [3, 3, 3, 3],
        [3, 3, 3, 3],
        [0, 0, 0, 0],
    ],
    "wire_NS": [
        [0, 3, 3, 0],
        [0, 3, 3, 0],
        [0, 3, 3, 0],
        [0, 3, 3, 0],
    ],
    "wire_NE": [
        [0, 3, 3, 0],
        [0, 3, 3, 3],
        [0, 3, 3, 3],
        [0, 0, 0, 0],
    ],
    "wire_NW": [
        [0, 3, 3, 0],
        [3, 3, 3, 0],
        [3, 3, 3, 0],
        [0, 0, 0, 0],
    ],
    "wire_SE": [
        [0, 0, 0, 0],
        [0, 3, 3, 3],
        [0, 3, 3, 3],
        [0, 3, 3, 0],
    ],
    "wire_SW": [
        [0, 0, 0, 0],
        [3, 3, 3, 0],
        [3, 3, 3, 0],
        [0, 3, 3, 0],
    ],
    "wire_NSE": [
        [0, 3, 3, 0],
        [0, 3, 3, 3],
        [0, 3, 3, 3],
        [0, 3, 3, 0],
    ],
    "wire_NSW": [
        [0, 3, 3, 0],
        [3, 3, 3, 0],
        [3, 3, 3, 0],
        [0, 3, 3, 0],
    ],
    "wire_NEW": [
        [0, 3, 3, 0],
        [3, 3, 3, 3],
        [3, 3, 3, 3],
        [0, 0, 0, 0],
    ],
    "wire_SEW": [
        [0, 0, 0, 0],
        [3, 3, 3, 3],
        [3, 3, 3, 3],
        [0, 3, 3, 0],
    ],
    "wire_NSEW": [
        [0, 3, 3, 0],
        [3, 3, 3, 3],
        [3, 3, 3, 3],
        [0, 3, 3, 0],
    ],
    "mwire_EW": [
        [0, 0, 0, 0],
        [6, 6, 6, 6],
        [6, 6, 6, 6],
        [0, 0, 0, 0],
    ],
    "mwire_NS": [
        [0, 6, 6, 0],
        [0, 6, 6, 0],
        [0, 6, 6, 0],
        [0, 6, 6, 0],
    ],
    "mwire_NE": [
        [0, 6, 6, 0],
        [0, 6, 6, 6],
        [0, 6, 6, 6],
        [0, 0, 0, 0],
    ],
    "mwire_NW": [
        [0, 6, 6, 0],
        [6, 6, 6, 0],
        [6, 6, 6, 0],
        [0, 0, 0, 0],
    ],
    "mwire_SE": [
        [0, 0, 0, 0],
        [0, 6, 6, 6],
        [0, 6, 6, 6],
        [0, 6, 6, 0],
    ],
    "mwire_SW": [
        [0, 0, 0, 0],
        [6, 6, 6, 0],
        [6, 6, 6, 0],
        [0, 6, 6, 0],
    ],
    "mwire_NSE": [
        [0, 6, 6, 0],
        [0, 6, 6, 6],
        [0, 6, 6, 6],
        [0, 6, 6, 0],
    ],
    "mwire_NSW": [
        [0, 6, 6, 0],
        [6, 6, 6, 0],
        [6, 6, 6, 0],
        [0, 6, 6, 0],
    ],
    "mwire_NEW": [
        [0, 6, 6, 0],
        [6, 6, 6, 6],
        [6, 6, 6, 6],
        [0, 0, 0, 0],
    ],
    "mwire_SEW": [
        [0, 0, 0, 0],
        [6, 6, 6, 6],
        [6, 6, 6, 6],
        [0, 6, 6, 0],
    ],
    "mwire_NSEW": [
        [0, 6, 6, 0],
        [6, 6, 6, 6],
        [6, 6, 6, 6],
        [0, 6, 6, 0],
    ],
    "hud_bg": [row[:] for row in _BG_PX],
    "hud_spacer": [
        [0, 0, 8, 0],
        [0, 0, 8, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ],
    "hud_life_full": [
        [0, 0, 0, 0],
        [0, 6, 6, 0],
        [0, 6, 6, 0],
        [0, 0, 0, 0],
    ],
    "hud_life_empty": [
        [0, 0, 0, 0],
        [0, 5, 5, 0],
        [0, 5, 5, 0],
        [0, 0, 0, 0],
    ],
    "hud_bar_filled": [
        [3, 3, 3, 3],
        [3, 3, 3, 3],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ],
    "hud_bar_remain": [
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ],
}

_ACTION_MAP: dict[int, GameAction] = {
    0: GameAction.RESET,
    1: GameAction.ACTION1,
    2: GameAction.ACTION2,
    3: GameAction.ACTION3,
    4: GameAction.ACTION4,
    5: GameAction.ACTION5,
}


def _int_to_action_input(action: int) -> ActionInput:
    ga = _ACTION_MAP.get(action)
    if ga is None:
        raise ValueError(f"Unknown action id: {action}")
    return ActionInput(id=ga)


def _gate_value(gate: str, left: int, right: int) -> int:
    if gate == "AND":
        return int(left == 1 and right == 1)
    if gate == "OR":
        return int(left == 1 or right == 1)
    if gate == "XOR":
        return int(left != right)
    if gate == "NAND":
        return int(not (left == 1 and right == 1))
    if gate == "NOR":
        return int(not (left == 1 or right == 1))
    return 0


def _make_sprite(
    kind: str,
    row: int,
    col: int,
    *,
    layer: int = 0,
    name: str = "",
) -> Sprite:
    px = PIXELS.get(kind, PIXELS["bg"])
    return Sprite(
        pixels=px,
        name=name or f"{kind}_{row}_{col}",
        x=(col + COL_OFFSET) * TILE,
        y=(row + ROW_OFFSET) * TILE,
        layer=layer,
        blocking=BlockingMode.NOT_BLOCKED,
        visible=True,
        collidable=False,
    )


def _compute_wire_cells(
    layout: CircuitLayout,
) -> dict[tuple[int, int], tuple[str, bool]]:
    directions: dict[tuple[int, int], set[str]] = {}
    memory_cells: set[tuple[int, int]] = set()

    for path in layout.paths:
        wps = path.waypoints
        for i, (r, c) in enumerate(wps):
            if (r, c) not in directions:
                directions[(r, c)] = set()
            if path.is_memory:
                memory_cells.add((r, c))

            if i > 0:
                pr, pc = wps[i - 1]
                if pr < r:
                    directions[(r, c)].add("N")
                elif pr > r:
                    directions[(r, c)].add("S")
                elif pc < c:
                    directions[(r, c)].add("W")
                elif pc > c:
                    directions[(r, c)].add("E")

            if i < len(wps) - 1:
                nr, nc = wps[i + 1]
                if nr < r:
                    directions[(r, c)].add("N")
                elif nr > r:
                    directions[(r, c)].add("S")
                elif nc < c:
                    directions[(r, c)].add("W")
                elif nc > c:
                    directions[(r, c)].add("E")

    result: dict[tuple[int, int], tuple[str, bool]] = {}
    for pos, dirs in directions.items():
        if not dirs:
            key = "EW"
        else:
            order = {"N": 0, "S": 1, "E": 2, "W": 3}
            key = "".join(sorted(dirs, key=lambda d: order[d]))
        is_mem = pos in memory_cells
        result[pos] = (key, is_mem)
    return result


def _build_board_sprites(
    index: int,
    config: LevelConfig,
    layout: CircuitLayout,
) -> list[Sprite]:
    sprites: list[Sprite] = []

    for r in range(BOARD_H):
        for c in range(BOARD_W):
            px = PIXELS.get("bg", PIXELS["bg"])
            sprites.append(
                Sprite(
                    pixels=px,
                    name=f"bg_{r}_{c}",
                    x=c * TILE,
                    y=r * TILE,
                    layer=0,
                    blocking=BlockingMode.NOT_BLOCKED,
                    visible=True,
                    collidable=False,
                )
            )

    wire_cells = _compute_wire_cells(layout)
    input_set = {(row, INPUT_COL) for row in config.input_rows}
    output_set = {(row, OUTPUT_COL) for row in config.output_rows}
    gate_set = {(g.row, g.col) for g in layout.gates}
    button_set = {(BUTTON_ROW, SUBMIT_COL)}

    for (r, c), (dir_key, is_mem) in wire_cells.items():
        if (r, c) in input_set or (r, c) in output_set or (r, c) in button_set:
            continue
        if (r, c) in gate_set:
            continue
        prefix = "mwire_" if is_mem else "wire_"
        kind = prefix + dir_key
        if kind not in PIXELS:
            kind = prefix + "EW"
        sprites.append(_make_sprite(kind, r, c, layer=1, name=f"wire_{r}_{c}"))

    for g in layout.gates:
        kind = f"gate_{g.gate}"
        sprites.append(
            _make_sprite(kind, g.row, g.col, layer=2, name=f"gate_{g.row}_{g.col}")
        )

    target_count = len(config.target_outputs)
    t_start = BOARD_W - (target_count * 2)
    for i in range(target_count):
        tc = t_start + i * 2
        kind = "target_on" if config.target_outputs[i] == 1 else "target_off"
        px = PIXELS.get(kind, PIXELS["bg"])
        sprites.append(
            Sprite(
                pixels=px,
                name=f"target_{i}",
                x=tc * TILE,
                y=0,
                layer=3,
                blocking=BlockingMode.NOT_BLOCKED,
                visible=True,
                collidable=False,
            )
        )

    for i, row in enumerate(config.input_rows):
        kind = "input_on" if config.initial_inputs[i] == 1 else "input_off"
        sprites.append(_make_sprite(kind, row, INPUT_COL, layer=3, name=f"input_{i}"))

    for i, row in enumerate(config.output_rows):
        sprites.append(
            _make_sprite("output_hidden", row, OUTPUT_COL, layer=3, name=f"output_{i}")
        )

    for r, c in config.anchor_positions:
        sprites.append(_make_sprite("anchor", r, c, layer=2, name=f"anchor_{r}_{c}"))

    sprites.append(
        _make_sprite("submit_locked", BUTTON_ROW, SUBMIT_COL, layer=3, name="submit")
    )

    sprites.append(_make_sprite("cursor", 0, 0, layer=5, name="cursor"))

    return sprites


def _build_hud_sprites(hud_row: int) -> list[Sprite]:
    sprites: list[Sprite] = []

    def _hud_sprite(kind: str, col: int, *, layer: int = 0, name: str = "") -> Sprite:
        px = PIXELS.get(kind, PIXELS["bg"])
        return Sprite(
            pixels=px,
            name=name or f"{kind}_{hud_row}_{col}",
            x=col * TILE,
            y=hud_row * TILE,
            layer=layer,
            blocking=BlockingMode.NOT_BLOCKED,
            visible=True,
            collidable=False,
        )

    for c in range(BOARD_W):
        sprites.append(_hud_sprite("hud_bg", c, layer=0))

    for i in range(MAX_LIVES):
        sprites.append(_hud_sprite("hud_life_full", i, layer=1, name=f"hud_life_{i}"))

    sprites.append(_hud_sprite("hud_spacer", MAX_LIVES, layer=1))

    bar_start = BOARD_W - HUD_BAR_WIDTH
    for c in range(bar_start, BOARD_W):
        sprites.append(
            _hud_sprite("hud_bar_filled", c, layer=1, name=f"hud_bar_{c - bar_start}")
        )

    return sprites


def _build_level(index: int) -> Level:
    config = HARDCODED_LEVELS[index]
    layout = CIRCUIT_LAYOUTS[index]
    sprites = _build_board_sprites(index, config, layout)
    sprites.extend(_build_hud_sprites(BOARD_H))
    return Level(
        sprites=sprites,
        grid_size=(BOARD_W * TILE, (BOARD_H + HUD_ROWS) * TILE),
        data={"index": index},
        name=f"level_{index + 1}",
    )


def _build_all_levels() -> list[Level]:
    return [_build_level(i) for i in range(len(HARDCODED_LEVELS))]


class Ic03(ARCBaseGame):
    def __init__(self, seed: int = 0, difficulty: str = "standard") -> None:
        if difficulty not in TURN_LIMITS:
            raise ValueError(f"Unsupported difficulty: {difficulty}")

        self._ic_difficulty: str = difficulty
        self._ic_seed: int = seed
        self._rng = random.Random(seed)
        self._ic_turn_limit: int = TURN_LIMITS[difficulty]
        self._ic_probe_span: int = PROBE_SPANS[difficulty]
        self._ic_level_configs: list[LevelConfig] = list(HARDCODED_LEVELS)
        self._ic_level: LevelConfig | None = None
        self._ic_inputs: list[int] = []
        self._ic_current_outputs: list[int] = []
        self._ic_charge: int = 0
        self._ic_turn: int = 0
        self._ic_done: bool = False
        self._ic_cursor: tuple[int, int] = (0, 0)
        self._ic_faults: int = 0
        self._ic_lives: int = MAX_LIVES
        self._ic_is_new_level: bool = True
        self._ic_transition_pending: str | None = None
        self._ic_consecutive_resets: int = 0
        self._ic_reveal_timers: list[int] = []
        self._ic_probe_cooldown: int = 0
        self._ic_level_turns: int = 0
        self._ic_level_turn_limit: int | None = None
        self._ic_history: list[
            tuple[
                tuple[int, ...],
                tuple[int, ...],
                tuple[int, int],
                int,
                int,
                int,
            ]
        ] = []
        self._ic_interactive: dict[tuple[int, int], tuple[str, int]] = {}
        self._ic_nav_nodes: list[tuple[int, int]] = []
        self._ic_pending_reward: float = 0.0
        self._ic_cursor_positions: list[tuple[int, int]] = []
        self._ic_cursor_pos_idx: int = 0
        self._ic_last_feedback: str = "ready"
        self._ic_last_changes: str = "none"
        self._ic_last_submit_feedback: str = "untested"

        camera = Camera(
            x=0,
            y=0,
            width=BOARD_W * TILE,
            height=(BOARD_H + HUD_ROWS) * TILE,
            background=0,
            letter_box=0,
        )
        super().__init__(
            game_id="ic03",
            levels=_build_all_levels(),
            camera=camera,
            available_actions=[0, 1, 2, 3, 4, 5, 7],
        )

    def reset(self) -> None:
        self._ic_lives = MAX_LIVES
        self._ic_consecutive_resets = 0
        self._ic_transition_pending = None
        self._ic_is_new_level = True
        self._ic_done = False
        self._ic_turn = 0
        self._ic_faults = 0
        self._ic_pending_reward = 0.0
        self.full_reset()

    def step(self) -> None:
        if self._ic_process_transition():
            return
        self._ic_process_action()

    def _ic_process_transition(self) -> bool:
        pending = self._ic_transition_pending
        if pending is None:
            return False
        self._ic_transition_pending = None
        self._ic_consecutive_resets = 0

        if pending in ("fail", "reset"):
            self._ic_lose_life(f"life lost: {pending}")
            return True

        if pending == "win":
            self._ic_is_new_level = True
            self.next_level()
            self._ic_finish_step()
            return True

        return False

    def _ic_process_action(self) -> None:
        action_id: int = self._action.id.value

        if self._action.id == GameAction.RESET:
            self._ic_finish_step()
            return

        self._ic_consecutive_resets = 0

        if self._ic_done:
            self._ic_finish_step()
            return

        before_outputs = tuple(self._ic_current_outputs)
        before_charge = self._ic_charge
        self._ic_last_changes = "none"
        self._ic_last_submit_feedback = "untested"
        r, c = self._ic_cursor
        self._ic_last_feedback = f"cursor {r},{c} > {self._ic_focus_label()}"
        probe_fired = False
        turn_consumed = False

        if action_id in (1, 2, 3, 4):
            prev_cursor = self._ic_cursor
            moved = self._ic_jump_cursor(action_id)
            self._ic_push_snapshot_with(prev_cursor)
            self._ic_turn += 1
            self._ic_level_turns += 1
            turn_consumed = True
            if moved:
                r, c = self._ic_cursor
                self._ic_last_feedback = f"cursor {r},{c} > {self._ic_focus_label()}"
                self._ic_move_cursor_sprite()
            else:
                self._ic_last_feedback = f"cursor {r},{c} blocked"

        elif action_id == 7:
            if self._ic_history:
                self._ic_restore_snapshot()
                self._ic_last_feedback = "undo"
            else:
                self._ic_last_feedback = "nothing to undo"
            self._ic_turn += 1
            self._ic_level_turns += 1
            turn_consumed = True

        elif action_id == 5:
            focus = self._ic_interactive.get(self._ic_cursor)
            if focus is not None:
                self._ic_push_snapshot()
                self._ic_turn += 1
                self._ic_level_turns += 1
                turn_consumed = True
                probe_fired = self._ic_perform_focus_action()
            else:
                self._ic_last_feedback = (
                    f"cursor {self._ic_cursor[0]},{self._ic_cursor[1]} no action"
                )

        if turn_consumed:
            self._ic_decay_reveals()
            self._ic_tick_probe_cooldown(probe_fired)

        self._ic_record_action_feedback(before_outputs, before_charge)

        if (
            self._ic_level_turn_limit is not None
            and self._ic_level_turns >= self._ic_level_turn_limit
            and not self._ic_done
        ):
            self._ic_last_submit_feedback = "budget exceeded"
            self._ic_transition_pending = "fail"
            self._ic_last_feedback = "move limit reached"

        if not self._ic_done and self._ic_turn >= self._ic_turn_limit:
            self._ic_transition_pending = "fail"
            self._ic_last_feedback = "time up"

        self._ic_sync_all_sprites()
        self._ic_finish_step()

    def get_actions(self) -> list[int]:
        if self._state in (EngineGameState.GAME_OVER, EngineGameState.WIN):
            return []
        pending_win = (
            self._ic_transition_pending == "win"
            and self._current_level_index >= len(self._ic_level_configs) - 1
        )
        pending_loss = (
            self._ic_transition_pending in ("fail", "reset") and self._ic_lives <= 1
        )
        if pending_win or pending_loss:
            return []
        if self._ic_done:
            return []
        return [1, 2, 3, 4, 5, 7]

    def on_set_level(self, level: Level) -> None:
        if self._ic_is_new_level:
            self._ic_lives = MAX_LIVES
        self._ic_transition_pending = None

        cfg = self._ic_level_configs[self._current_level_index]
        self._ic_level = cfg
        self._ic_done = False
        self._ic_inputs = list(cfg.initial_inputs)
        self._ic_current_outputs = []
        self._ic_charge = 0
        self._ic_faults = 0
        self._ic_reveal_timers = [0] * len(cfg.output_rows)
        self._ic_probe_cooldown = 0
        self._ic_level_turns = 0
        self._ic_level_turn_limit = LEVEL_MOVE_LIMITS.get(self._current_level_index)
        self._ic_history = []
        self._ic_interactive = self._ic_build_interactive_map(cfg)
        self._ic_nav_nodes = sorted(self._ic_interactive.keys())
        if self._ic_is_new_level:
            self._ic_generate_cursor_positions()
        self._ic_is_new_level = False
        self._ic_cursor = self._ic_next_cursor_position()
        self._ic_pending_reward = 0.0
        self._ic_last_feedback = f"level {self._current_level_index + 1} start"
        self._ic_last_changes = "none"
        self._ic_last_submit_feedback = "untested"

        self._ic_refresh_outputs()
        self._ic_sync_all_sprites()
        self._ic_update_hud()

    def _ic_generate_cursor_positions(self) -> None:
        nodes = list(self._ic_nav_nodes)
        if not nodes:
            self._ic_cursor_positions = [(0, 0)] * 4
        elif len(nodes) == 1:
            self._ic_cursor_positions = [nodes[0]] * 4
        elif len(nodes) <= 4:
            shuffled = list(nodes)
            self._rng.shuffle(shuffled)
            self._ic_cursor_positions = (shuffled * 2)[:4]
        else:
            sampled = self._rng.sample(nodes, 4)
            self._ic_cursor_positions = sampled
        self._ic_cursor_pos_idx = 0

    def _ic_next_cursor_position(self) -> tuple[int, int]:
        if not self._ic_cursor_positions:
            return (0, 0)
        pos = self._ic_cursor_positions[self._ic_cursor_pos_idx]
        self._ic_cursor_pos_idx = (self._ic_cursor_pos_idx + 1) % len(
            self._ic_cursor_positions
        )
        return pos

    def handle_reset(self) -> None:
        self._ic_consecutive_resets += 1

        if self._ic_consecutive_resets >= 2:
            self._ic_consecutive_resets = 0
            self._ic_is_new_level = True
            self._ic_lives = MAX_LIVES
            self._ic_turn = 0
            self._ic_done = False
            self._ic_transition_pending = None
            self._ic_faults = 0
            self._ic_pending_reward = 0.0
            self._ic_last_feedback = "full reset"
            self.full_reset()
            self._ic_update_hud()
        else:
            self._ic_is_new_level = True
            self._ic_lives = MAX_LIVES
            self._ic_turn = 0
            self._ic_done = False
            self._ic_transition_pending = None
            self._ic_faults = 0
            self._ic_pending_reward = 0.0
            self._ic_last_feedback = "level reset"
            self.level_reset()
            self._ic_update_hud()

    def _ic_build_state(self) -> dict[str, Any]:
        pending_win = (
            self._ic_transition_pending == "win"
            and self._current_level_index >= len(self._ic_level_configs) - 1
        )
        pending_loss = (
            self._ic_transition_pending in ("fail", "reset") and self._ic_lives <= 1
        )
        done = (
            self._state in (EngineGameState.GAME_OVER, EngineGameState.WIN)
            or pending_win
            or pending_loss
            or self._ic_done
        )
        won = self._state == EngineGameState.WIN or pending_win
        lost = self._state == EngineGameState.GAME_OVER or pending_loss

        visible_outputs = "".join(
            self._ic_visible_output(i) for i in range(len(self._ic_current_outputs))
        )
        target_outputs = "".join(
            str(v) for v in (self._ic_level.target_outputs if self._ic_level else ())
        )
        moves_left = (
            max(0, self._ic_level_turn_limit - self._ic_level_turns)
            if self._ic_level_turn_limit is not None
            else "?"
        )

        bar = self._ic_steps_bar()
        text_observation = (
            f"Level {self._current_level_index + 1}/{len(self._ic_level_configs)} | "
            f"Lives: {self._ic_lives}/{MAX_LIVES} | "
            f"Steps: [{bar}] {self._ic_level_turns}/{self._ic_level_turn_limit or '?'}\n"
            f"target {target_outputs} seen {visible_outputs}\n"
            f"action {self._ic_last_feedback}\n"
            f"change {self._ic_last_changes}"
        )

        if won:
            text_observation += " | YOU WIN"
        elif lost:
            text_observation += " | GAME OVER"

        return {
            "text_observation": text_observation,
            "image_observation": None,
            "available_actions": self.get_actions(),
            "current_turn": self._ic_turn,
            "done": done,
            "won": won,
            "lost": lost,
            "lives": self._ic_lives,
            "level": self._current_level_index + 1,
            "moves_remaining": moves_left,
            "metadata": {
                "game_id": "ic03",
                "level": self._current_level_index + 1,
                "turn_limit": self._ic_turn_limit,
                "turns_remaining": max(0, self._ic_turn_limit - self._ic_turn),
                "charge": self._ic_charge,
                "faults": self._ic_faults,
                "lives": self._ic_lives,
                "seed": self._ic_seed,
                "visible_outputs": [
                    self._ic_visible_output(i)
                    for i in range(len(self._ic_current_outputs))
                ],
                "target_outputs": list(self._ic_level.target_outputs)
                if self._ic_level is not None
                else [],
                "feedback": self._ic_last_feedback,
                "changes": self._ic_last_changes,
                "submit_feedback": self._ic_last_submit_feedback,
                "probe_cooldown": self._ic_probe_cooldown,
                "level_turns": self._ic_level_turns,
                "level_turn_limit": self._ic_level_turn_limit,
                "done": done,
            },
        }

    def _ic_steps_bar(self) -> str:
        limit = self._ic_level_turn_limit or 1
        remaining = max(limit - self._ic_level_turns, 0)
        filled = min(
            (remaining * HUD_BAR_WIDTH + limit - 1) // limit,
            HUD_BAR_WIDTH,
        )
        return "#" * filled + "." * (HUD_BAR_WIDTH - filled)

    def _ic_perform_focus_action(self) -> bool:
        focus = self._ic_interactive.get(self._ic_cursor)
        if focus is None:
            self._ic_last_feedback = "no action"
            return False

        if focus[0] == "input":
            self._ic_last_feedback = f"toggled input-{focus[1] + 1}"
            self._ic_toggle_input(focus[1])
            return False

        if focus[0] == "output":
            return self._ic_probe_focus()

        if focus[0] == "submit":
            self._ic_submit_state()
            return False

        return False

    def _ic_build_interactive_map(
        self, level: LevelConfig
    ) -> dict[tuple[int, int], tuple[str, int]]:
        mapping: dict[tuple[int, int], tuple[str, int]] = {}
        for i, row in enumerate(level.input_rows):
            mapping[(row, INPUT_COL)] = ("input", i)
        for i, row in enumerate(level.output_rows):
            mapping[(row, OUTPUT_COL)] = ("output", i)
        mapping[(BUTTON_ROW, SUBMIT_COL)] = ("submit", 0)

        return mapping

    def _ic_jump_cursor(self, action_id: int) -> bool:
        row, col = self._ic_cursor
        best: tuple[int, int] | None = None
        best_score: tuple[bool, int, int, int, int] | None = None

        for pos in self._ic_nav_nodes:
            if pos == self._ic_cursor:
                continue
            pr, pc = pos

            if action_id == 1:
                if pr >= row:
                    continue
                score = (pc != col, row - pr, abs(pc - col), pr, pc)
            elif action_id == 2:
                if pr <= row:
                    continue
                score = (pc != col, pr - row, abs(pc - col), pr, pc)
            elif action_id == 3:
                if pc >= col:
                    continue
                score = (pr != row, col - pc, abs(pr - row), pr, pc)
            elif action_id == 4:
                if pc <= col:
                    continue
                score = (pr != row, pc - col, abs(pr - row), pr, pc)
            else:
                continue

            if best_score is None or score < best_score:
                best = pos
                best_score = score

        if best is None:
            self._ic_last_feedback = "edge blocked"
            return False

        self._ic_cursor = best
        return True

    def _ic_focus_label(self) -> str:
        focus = self._ic_interactive.get(self._ic_cursor)
        if focus is None:
            return f"grid {self._ic_cursor[0]},{self._ic_cursor[1]}"
        if focus[0] == "input":
            return f"input-{focus[1] + 1}"
        if focus[0] == "output":
            return f"output-{focus[1] + 1}"
        if focus[0] == "submit":
            return "submit"
        return "undo"

    def _ic_push_snapshot(self) -> None:
        snapshot = (
            tuple(self._ic_inputs),
            tuple(self._ic_reveal_timers),
            self._ic_cursor,
            self._ic_faults,
            self._ic_turn,
            self._ic_level_turns,
        )
        self._ic_history.append(snapshot)
        if len(self._ic_history) > HISTORY_LIMIT:
            self._ic_history.pop(0)

    def _ic_push_snapshot_with(self, cursor_override: tuple[int, int]) -> None:
        snapshot = (
            tuple(self._ic_inputs),
            tuple(self._ic_reveal_timers),
            cursor_override,
            self._ic_faults,
            self._ic_turn,
            self._ic_level_turns,
        )
        self._ic_history.append(snapshot)
        if len(self._ic_history) > HISTORY_LIMIT:
            self._ic_history.pop(0)

    def _ic_restore_snapshot(self) -> None:
        if not self._ic_history:
            return
        inputs, reveal_timers, cursor, faults, turn, level_turns = (
            self._ic_history.pop()
        )
        self._ic_inputs = list(inputs)
        self._ic_reveal_timers = list(reveal_timers)
        self._ic_cursor = cursor
        self._ic_faults = faults
        self._ic_refresh_outputs()
        self._ic_move_cursor_sprite()

    def _ic_probe_focus(self) -> bool:
        focus = self._ic_interactive.get(self._ic_cursor)
        if focus is None or focus[0] != "output":
            self._ic_last_feedback = "probe drift"
            return False
        if self._ic_probe_cooldown > 0:
            self._ic_last_feedback = f"probe cooling {self._ic_probe_cooldown}"
            return False
        self._ic_reveal_timers[focus[1]] = self._ic_probe_span + 1
        self._ic_probe_cooldown = PROBE_COOLDOWNS[self._ic_difficulty]
        self._ic_last_feedback = f"probe output-{focus[1] + 1}"
        return True

    def _ic_toggle_input(self, input_index: int) -> None:
        if self._ic_level is None:
            return
        next_inputs, outputs, charge = self._ic_toggle_state(
            self._ic_level,
            tuple(self._ic_inputs),
            input_index,
        )
        self._ic_inputs = list(next_inputs)
        self._ic_current_outputs = list(outputs)
        self._ic_charge = charge

    def _ic_submit_state(self) -> None:
        if self._ic_level is None:
            return
        matches = sum(
            1
            for cur, tgt in zip(self._ic_current_outputs, self._ic_level.target_outputs)
            if cur == tgt
        )
        total = len(self._ic_level.target_outputs)

        if self._ic_seal_open():
            for i in range(len(self._ic_reveal_timers)):
                self._ic_reveal_timers[i] = max(
                    self._ic_reveal_timers[i], self._ic_probe_span + 1
                )
            if self._current_level_index >= len(self._ic_level_configs) - 1:
                self._ic_pending_reward = 1.0
                self._ic_last_feedback = "game complete"
                self._ic_last_submit_feedback = "final seal accepted"
                self._ic_sync_all_sprites()
                self._ic_is_new_level = True
                self.next_level()
            else:
                self._ic_last_feedback = "level complete"
                self._ic_last_submit_feedback = "level complete"
                self._ic_sync_all_sprites()
                self._ic_is_new_level = True
                self.next_level()
            return

        self._ic_faults += SUBMIT_PENALTIES[self._current_level_index]
        self._ic_last_feedback = f"rejected {matches}/{total} aligned"
        self._ic_last_submit_feedback = f"{matches}/{total} outputs aligned"

        for i in range(len(self._ic_reveal_timers)):
            self._ic_reveal_timers[i] = max(
                self._ic_reveal_timers[i], self._ic_probe_span + 1
            )

        if self._ic_faults >= MAX_LIVES:
            self._ic_transition_pending = "fail"
            self._ic_last_feedback = "too many faults"

    def _ic_seal_open(self) -> bool:
        if self._ic_level is None:
            return False
        return tuple(self._ic_current_outputs) == self._ic_level.target_outputs

    def _ic_lose_life(self, reason: str) -> None:
        self._ic_lives -= 1
        self._ic_last_feedback = reason

        if self._ic_lives <= 0:
            self.lose()
            self.complete_action()
        else:
            self._ic_faults = 0
            self._ic_turn = 0
            self._ic_reveal_timers = [0] * len(self._ic_reveal_timers)
            self._ic_probe_cooldown = 0
            self._ic_history = []
            self._ic_is_new_level = False
            self.level_reset()
            self._ic_finish_step()

    def _ic_refresh_outputs(self) -> None:
        if self._ic_level is None:
            self._ic_current_outputs = []
            self._ic_charge = 0
            return
        _, outputs, charge = self._ic_evaluate(self._ic_level, tuple(self._ic_inputs))
        self._ic_current_outputs = list(outputs)
        self._ic_charge = charge

    def _ic_toggle_state(
        self,
        level: LevelConfig,
        inputs: tuple[int, ...],
        input_index: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...], int]:
        next_inputs = list(inputs)
        next_inputs[input_index] = 1 - next_inputs[input_index]

        _, outputs, charge = self._ic_evaluate(level, tuple(next_inputs))
        return tuple(next_inputs), outputs, charge

    def _ic_evaluate(
        self,
        level: LevelConfig,
        inputs: tuple[int, ...],
    ) -> tuple[tuple[int, ...], tuple[int, ...], int]:
        node_values: list[int] = []
        for spec in level.node_specs:
            left = self._ic_resolve_source(spec.sources[0], inputs, node_values)
            right = self._ic_resolve_source(spec.sources[1], inputs, node_values)
            node_values.append(_gate_value(spec.gate, left, right))

        outputs: list[int] = []
        for spec in level.output_specs:
            left = self._ic_resolve_source(spec.sources[0], inputs, node_values)
            right = self._ic_resolve_source(spec.sources[1], inputs, node_values)
            outputs.append(_gate_value(spec.gate, left, right))

        charge = self._ic_compute_charge(level.index, inputs, tuple(outputs))
        return tuple(node_values), tuple(outputs), charge

    def _ic_resolve_source(
        self,
        token: str,
        inputs: tuple[int, ...],
        node_values: list[int],
    ) -> int:
        bucket = token[0]
        index = int(token[1:])
        if bucket == "i":
            return inputs[index]
        return node_values[index]

    def _ic_compute_charge(
        self,
        level_index: int,
        inputs: tuple[int, ...],
        outputs: tuple[int, ...],
    ) -> int:
        if level_index == 0:
            return (sum(inputs) + sum(outputs)) % 2
        if level_index == 1:
            return (sum(inputs) + 2 * sum(outputs)) % 3
        if level_index == 2:
            return (sum(inputs) + sum(outputs)) % 4
        return (2 * sum(inputs) + outputs[0] + outputs[-1]) % 4

    def _ic_decay_reveals(self) -> None:
        for i in range(len(self._ic_reveal_timers)):
            if self._ic_reveal_timers[i] > 0:
                self._ic_reveal_timers[i] -= 1

    def _ic_tick_probe_cooldown(self, probe_fired: bool) -> None:
        if probe_fired:
            return
        if self._ic_probe_cooldown > 0:
            self._ic_probe_cooldown -= 1

    def _ic_visible_output(self, index: int) -> str:
        if self._ic_level is None:
            return "?"
        if self._ic_seal_open():
            return str(self._ic_current_outputs[index])
        visible = set(self._ic_level.always_visible)
        if self._ic_difficulty == "easy":
            for candidate in range(len(self._ic_current_outputs)):
                if candidate not in visible:
                    visible.add(candidate)
                    break
        if index in visible or self._ic_reveal_timers[index] > 0:
            return str(self._ic_current_outputs[index])
        return "?"

    def _ic_record_action_feedback(
        self,
        before_outputs: tuple[int, ...],
        before_charge: int,
    ) -> None:
        output_changes: list[str] = []
        for i, (before, after) in enumerate(
            zip(before_outputs, self._ic_current_outputs)
        ):
            if before != after:
                output_changes.append(f"o{i + 1}:{before}->{after}")

        if output_changes:
            self._ic_last_changes = ", ".join(output_changes)
        elif before_charge != self._ic_charge:
            self._ic_last_changes = f"charge:{before_charge}->{self._ic_charge}"
        else:
            self._ic_last_changes = "none"

        if output_changes and self._current_level_index == 1:
            self._ic_last_feedback += " | hidden dependency"

    def _ic_swap_sprite(self, sprite_name: str, new_kind: str) -> None:
        sprites = self.current_level.get_sprites_by_name(sprite_name)
        if sprites:
            sprites[0].pixels = np.array(PIXELS[new_kind], dtype=np.int8)

    def _ic_sprite_pos(self, name: str, row: int, col: int) -> None:
        sprites = self.current_level.get_sprites_by_name(name)
        if sprites:
            sprites[0].set_position(
                (col + COL_OFFSET) * TILE, (row + ROW_OFFSET) * TILE
            )

    def _ic_move_cursor_sprite(self) -> None:
        r, c = self._ic_cursor
        self._ic_sprite_pos("cursor", r, c)

    def _ic_sync_all_sprites(self) -> None:
        if self._ic_level is None:
            return

        for i, row in enumerate(self._ic_level.input_rows):
            kind = "input_on" if self._ic_inputs[i] == 1 else "input_off"
            self._ic_swap_sprite(f"input_{i}", kind)

        for i, row in enumerate(self._ic_level.output_rows):
            vis = self._ic_visible_output(i)
            if vis == "?":
                kind = "output_hidden"
            elif vis == "1":
                kind = "output_on"
            else:
                kind = "output_off"
            self._ic_swap_sprite(f"output_{i}", kind)

        kind = "submit_ready" if self._ic_seal_open() else "submit_locked"
        self._ic_swap_sprite("submit", kind)

        self._ic_move_cursor_sprite()

    def _ic_update_hud(self) -> None:
        for i in range(MAX_LIVES):
            kind = "hud_life_full" if i < self._ic_lives else "hud_life_empty"
            sprites = self.current_level.get_sprites_by_name(f"hud_life_{i}")
            if sprites:
                sprites[0].pixels = np.array(PIXELS[kind], dtype=np.int8)

        bar_width = HUD_BAR_WIDTH
        limit = max(self._ic_level_turn_limit or 1, 1)
        remaining = max(limit - self._ic_level_turns, 0)
        filled = min((remaining * bar_width + limit - 1) // limit, bar_width)

        for c in range(bar_width):
            kind = "hud_bar_filled" if c < filled else "hud_bar_remain"
            sprites = self.current_level.get_sprites_by_name(f"hud_bar_{c}")
            if sprites:
                sprites[0].pixels = np.array(PIXELS[kind], dtype=np.int8)

    def _ic_finish_step(self) -> None:
        self._ic_update_hud()
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

    def __init__(self, seed: int = 0, difficulty: str = "standard") -> None:
        self._engine = Ic03(seed=seed, difficulty=difficulty)
        self._difficulty = difficulty
        self._total_turns = 0
        self._done = False
        self._game_won = False
        self._game_over = False
        self._last_action_was_reset = False

    def _build_text_obs(self) -> str:
        return self._engine._ic_build_state()["text_observation"]

    def _build_image_bytes(self) -> Optional[bytes]:
        e = self._engine
        rendered = e.camera.render(e.current_level.get_sprites())
        if hasattr(rendered, "tolist"):
            grid = rendered.tolist()
        else:
            grid = rendered if rendered else []
        if not grid:
            return None
        arr = np.array(grid, dtype=np.uint8)
        h, w = arr.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for idx, color in enumerate(ARC_PALETTE):
            mask = arr == idx
            rgb[mask] = color

        def _pack_png(data: np.ndarray) -> bytes:
            raw = b""
            for row in data:
                raw += b"\x00" + row.tobytes()
            compressed = zlib.compress(raw)

            def _chunk(tag: bytes, body: bytes) -> bytes:
                return (
                    struct.pack(">I", len(body))
                    + tag
                    + body
                    + struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF)
                )

            ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
            return (
                b"\x89PNG\r\n\x1a\n"
                + _chunk(b"IHDR", ihdr)
                + _chunk(b"IDAT", compressed)
                + _chunk(b"IEND", b"")
            )

        return _pack_png(rgb)

    def _build_game_state(
        self, done: bool = False, info: Optional[Dict] = None
    ) -> GameState:
        e = self._engine
        valid_actions = self.get_actions() if not done else None
        engine_state = e._ic_build_state()
        return GameState(
            text_observation=engine_state["text_observation"],
            image_observation=self._build_image_bytes(),
            valid_actions=valid_actions,
            turn=self._total_turns,
            metadata={
                "level_index": e._current_level_index,
                "total_levels": len(e._ic_level_configs),
                "levels_completed": e._current_level_index,
                "game_over": engine_state.get("lost", False),
                "lives": e._ic_lives,
                "done": done,
                "info": info or {},
                **engine_state.get("metadata", {}),
            },
        )

    def reset(self) -> GameState:
        e = self._engine

        if self._game_won or self._game_over or self._last_action_was_reset:
            e.perform_action(ActionInput(id=GameAction.RESET))
            e.perform_action(ActionInput(id=GameAction.RESET))
        else:
            e.perform_action(ActionInput(id=GameAction.RESET))

        self._total_turns = 0
        self._done = False
        self._game_over = False
        self._last_action_was_reset = True
        self._game_won = False

        return self._build_game_state()

    def is_done(self) -> bool:
        return self._done

    def get_actions(self) -> List[str]:
        if self._done or self._game_over:
            return ["reset"]
        if self._engine is not None:
            if self._engine._state in (
                EngineGameState.GAME_OVER,
                EngineGameState.WIN,
            ):
                return ["reset"]
            if not self._engine.get_actions():
                return ["reset"]
        return list(self._VALID_ACTIONS)

    def step(self, action: str) -> StepResult:
        e = self._engine

        if action == "reset":
            state = self.reset()
            return StepResult(
                state=state, reward=0.0, done=False, info={"action": "reset"}
            )

        if action not in self._ACTION_MAP:
            return StepResult(
                state=self._build_game_state(done=self._done),
                reward=0.0,
                done=self._done,
                info={"action": action, "reason": "invalid_action"},
            )

        if self._game_over:
            return StepResult(
                state=self._build_game_state(done=False),
                reward=0.0,
                done=False,
                info={"action": action, "reason": "game_over_only_reset"},
            )

        self._last_action_was_reset = False
        self._total_turns += 1

        game_action = self._ACTION_MAP[action]
        info: Dict = {"action": action}

        level_before = e._current_level_index
        action_input = ActionInput(id=game_action)
        frame = e.perform_action(action_input, raw=True)
        level_after = e._current_level_index

        state_name = frame.state.name if frame and frame.state else ""
        game_won = state_name == "WIN"
        game_over = state_name == "GAME_OVER"

        total_levels = len(e._ic_level_configs)
        level_reward = 1.0 / total_levels

        if game_won:
            self._done = True
            self._game_won = True
            info["reason"] = "game_complete"
            return StepResult(
                state=self._build_game_state(done=True, info=info),
                reward=level_reward,
                done=True,
                info=info,
            )

        if game_over:
            self._game_over = True
            info["reason"] = "game_over"
            return StepResult(
                state=self._build_game_state(done=False, info=info),
                reward=0.0,
                done=False,
                info=info,
            )

        reward = 0.0
        if level_after != level_before:
            reward = level_reward
            info["reason"] = "level_complete"

        return StepResult(
            state=self._build_game_state(done=False, info=info),
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
        difficulty: str = "standard",
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()

        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"Unsupported render_mode '{render_mode}'. "
                f"Supported: {self.metadata['render_modes']}"
            )
        self.render_mode: Optional[str] = render_mode
        self._difficulty: str = difficulty

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

        self._env = PuzzleEnvironment(seed=self._seed, difficulty=self._difficulty)
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

    env = ArcGameEnv(render_mode="rgb_array")
    try:
        check_env(env.unwrapped, skip_render_check=True)
        print("[PASS] check_env passed — environment is Gymnasium-compliant.")
    except Exception as e:
        print(f"[FAIL] check_env failed: {e}")

    obs, info = env.reset()
    print(f"Obs shape: {obs.shape}, dtype: {obs.dtype}")
    print(f"Valid actions: {info.get('valid_actions')}")

    mask = env.action_mask()
    print(f"Action mask: {mask}")

    valid_indices = np.where(mask == 1)[0]
    if len(valid_indices) > 0:
        obs, reward, term, trunc, info = env.step(valid_indices[0])
        print(f"Step → reward={reward}, terminated={term}, truncated={trunc}")

    env.close()
    print("Smoke test passed!")
