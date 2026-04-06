from __future__ import annotations

import io
import random
import struct
import zlib
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import gymnasium as gym
import numpy as np
from arcengine import (
    ARCBaseGame,
    ActionInput,
    Camera,
    GameAction,
)
from arcengine.enums import BlockingMode
from arcengine.level import Level
from arcengine.sprites import Sprite
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

T_BL = "\u2514"
T_BR = "\u2518"
T_TL = "\u250c"
T_TR = "\u2510"

T_EMPTY = "."
T_BLOCKED = "X"
T_SOURCE = "S"
T_CITY = "C"
T_HORIZ = "-"
T_VERT = "|"
T_TRAP = "T"
T_ENEMY = "E"

ROTATABLE: set[str] = {T_HORIZ, T_VERT, T_BL, T_BR, T_TL, T_TR}

DIRS: dict[str, Tuple[int, int]] = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}
OPPOSITE: dict[str, str] = {"U": "D", "D": "U", "L": "R", "R": "L"}

BLACK = 5
LIGHT_GREY = 2
DARK_GREY = 4
GREEN = 14
BLUE = 9
YELLOW = 11
RED = 8
MAROON = 13

TILE: int = 4
GRID_W: int = 16
GRID_H: int = 16
PIP_COUNT: int = 14

MAX_LIVES: int = 3
LIVES_ROW: int = 0
STEPS_ROW: int = 15

LEVEL_RULES: List[Dict[str, Any]] = [
    {"type": "basic"},
    {"type": "directional"},
    {"type": "decay"},
    {"type": "hidden"},
]

BASELINE_STEPS: List[int] = [24, 30, 24, 33]

LEVEL_GRIDS: List[List[str]] = [
    [
        "..........",
        "." + T_TR + T_BR + T_BR + "C.....",
        "." + T_BL + "........",
        ".S" + T_BR + T_BR + "C.....",
        "." + T_BL + "........",
        ".-" + T_BR + T_BR + "C.....",
        "..........",
        ".....|....",
        "..........",
        "........-.",
    ],
    [
        "C|.........",
        ".-|........",
        "..S" + T_BR + T_BR + "|.....",
        ".." + T_BL + "..-" + T_BR + "C...",
        ".." + T_BL + "........",
        "..-" + T_BR + "C......",
        "...........",
        "...........",
        ".........-.",
        "...........",
        "....|......",
    ],
    [
        "............",
        "." + T_TR + T_BR + T_BR + "C.......",
        "." + T_BL + "..........",
        ".S" + T_BR + T_BR + "C.......",
        "." + T_BL + "..........",
        ".-" + T_BR + T_BR + "C.......",
        "............",
        "X...........",
        "T...........",
        "............",
        "............",
        "...........|",
    ],
    [
        "X.........X.",
        ".....C......",
        "..-.." + T_BR + "......",
        "..|.E" + T_BR + "......",
        "..-.." + T_BR + "......",
        ".C" + T_BL + T_BL + T_BL + "S" + T_BL + T_BL + T_BL + "C..",
        "..-.." + T_BR + "......",
        "..|.." + T_BR + "..E...",
        "..T.." + T_BR + "......",
        ".....C......",
        ".T..........",
        "........T...",
    ],
]

_PIPE_CONNECTION_MAP: dict[str, set[str]] = {
    T_HORIZ: {"L", "R"},
    T_VERT: {"U", "D"},
    T_BL: {"U", "R"},
    T_BR: {"U", "L"},
    T_TL: {"D", "R"},
    T_TR: {"D", "L"},
    T_SOURCE: {"U", "D", "L", "R"},
    T_CITY: {"U", "D", "L", "R"},
    T_BLOCKED: set(),
    T_EMPTY: set(),
    T_TRAP: {"U", "D", "L", "R"},
    T_ENEMY: set(),
}

_DIRECTIONAL_OUTGOING_MAP: dict[str, set[str]] = {
    T_HORIZ: {"R"},
    T_VERT: {"D"},
    T_TL: {"D", "R"},
    T_TR: {"D", "L"},
    T_BL: {"U", "R"},
    T_BR: {"U", "L"},
    T_SOURCE: {"U", "D", "L", "R"},
    T_CITY: {"U", "D", "L", "R"},
    T_BLOCKED: set(),
    T_EMPTY: set(),
    T_TRAP: {"U", "D", "L", "R"},
    T_ENEMY: set(),
}

_DIRECTIONAL_INCOMING_MAP: dict[str, set[str]] = {
    T_HORIZ: {"L"},
    T_VERT: {"U"},
    T_TL: {"D", "R"},
    T_TR: {"D", "L"},
    T_BL: {"U", "R"},
    T_BR: {"U", "L"},
    T_SOURCE: {"U", "D", "L", "R"},
    T_CITY: {"U", "D", "L", "R"},
    T_BLOCKED: set(),
    T_EMPTY: set(),
    T_TRAP: {"U", "D", "L", "R"},
    T_ENEMY: set(),
}

_HIDDEN_CONNECTION_MAP: dict[str, set[str]] = {
    T_HORIZ: {"U", "D"},
    T_VERT: {"L", "R"},
    T_BL: {"D", "L"},
    T_BR: {"D", "R"},
    T_TL: {"U", "R"},
    T_TR: {"U", "L"},
    T_SOURCE: {"U", "D", "L", "R"},
    T_CITY: {"U", "D", "L", "R"},
    T_BLOCKED: set(),
    T_EMPTY: set(),
    T_TRAP: {"U", "D", "L", "R"},
    T_ENEMY: set(),
}

_ROTATION_CYCLE: dict[str, str] = {
    T_HORIZ: T_BL,
    T_BL: T_VERT,
    T_VERT: T_TR,
    T_TR: T_TL,
    T_TL: T_BR,
    T_BR: T_HORIZ,
}

_EMPTY_CONNECTION: set[str] = set()


def _rotate_tile(tile: str) -> str:
    return _ROTATION_CYCLE.get(tile, tile)


def _get_connections(tile: str, rule_type: str) -> set[str]:
    if rule_type == "hidden":
        return _HIDDEN_CONNECTION_MAP.get(tile, _EMPTY_CONNECTION)
    return _PIPE_CONNECTION_MAP.get(tile, _EMPTY_CONNECTION)


def _get_directional_outgoing(tile: str) -> set[str]:
    return _DIRECTIONAL_OUTGOING_MAP.get(tile, _EMPTY_CONNECTION)


def _get_directional_incoming(tile: str) -> set[str]:
    return _DIRECTIONAL_INCOMING_MAP.get(tile, _EMPTY_CONNECTION)


def _tile_pixels(tile: str) -> List[List[int]]:
    if tile == T_BLOCKED:
        return _get_blocked_tile_pixels()
    if tile == T_SOURCE:
        return _get_source_tile_pixels()
    if tile == T_CITY:
        return _get_city_tile_pixels()
    if tile == T_TRAP:
        return _get_trap_tile_pixels()
    if tile == T_ENEMY:
        return _get_enemy_tile_pixels()
    if tile == T_HORIZ:
        return _get_horizontal_pipe_pixels()
    if tile == T_VERT:
        return _get_vertical_pipe_pixels()
    if tile == T_BL:
        return _get_bottom_left_corner_pixels()
    if tile == T_BR:
        return _get_bottom_right_corner_pixels()
    if tile == T_TL:
        return _get_top_left_corner_pixels()
    if tile == T_TR:
        return _get_top_right_corner_pixels()
    return _get_empty_tile_pixels()


def _get_empty_tile_pixels() -> List[List[int]]:
    return [
        [BLACK, BLACK, BLACK, BLACK],
        [BLACK, DARK_GREY, DARK_GREY, BLACK],
        [BLACK, DARK_GREY, DARK_GREY, BLACK],
        [BLACK, BLACK, BLACK, BLACK],
    ]


def _get_blocked_tile_pixels() -> List[List[int]]:
    return [
        [MAROON, DARK_GREY, DARK_GREY, MAROON],
        [DARK_GREY, MAROON, MAROON, DARK_GREY],
        [DARK_GREY, MAROON, MAROON, DARK_GREY],
        [MAROON, DARK_GREY, DARK_GREY, MAROON],
    ]


def _get_source_tile_pixels() -> List[List[int]]:
    return [
        [DARK_GREY, GREEN, GREEN, DARK_GREY],
        [GREEN, GREEN, GREEN, GREEN],
        [GREEN, GREEN, GREEN, GREEN],
        [DARK_GREY, GREEN, GREEN, DARK_GREY],
    ]


def _get_city_tile_pixels() -> List[List[int]]:
    return [
        [DARK_GREY, BLUE, BLUE, DARK_GREY],
        [BLUE, BLUE, BLUE, BLUE],
        [BLUE, BLUE, BLUE, BLUE],
        [DARK_GREY, BLUE, BLUE, DARK_GREY],
    ]


def _get_city_powered_pixels() -> List[List[int]]:
    return [
        [DARK_GREY, GREEN, GREEN, DARK_GREY],
        [GREEN, GREEN, GREEN, GREEN],
        [GREEN, GREEN, GREEN, GREEN],
        [DARK_GREY, GREEN, GREEN, DARK_GREY],
    ]


def _get_trap_tile_pixels() -> List[List[int]]:
    return [
        [DARK_GREY, RED, RED, DARK_GREY],
        [RED, RED, RED, RED],
        [RED, RED, RED, RED],
        [DARK_GREY, RED, RED, DARK_GREY],
    ]


def _get_enemy_tile_pixels() -> List[List[int]]:
    return [
        [RED, RED, RED, RED],
        [RED, MAROON, MAROON, RED],
        [RED, MAROON, MAROON, RED],
        [RED, RED, RED, RED],
    ]


def _get_horizontal_pipe_pixels() -> List[List[int]]:
    return [
        [DARK_GREY, DARK_GREY, DARK_GREY, DARK_GREY],
        [LIGHT_GREY, LIGHT_GREY, LIGHT_GREY, LIGHT_GREY],
        [LIGHT_GREY, LIGHT_GREY, LIGHT_GREY, LIGHT_GREY],
        [DARK_GREY, DARK_GREY, DARK_GREY, DARK_GREY],
    ]


def _get_vertical_pipe_pixels() -> List[List[int]]:
    return [
        [DARK_GREY, LIGHT_GREY, LIGHT_GREY, DARK_GREY],
        [DARK_GREY, LIGHT_GREY, LIGHT_GREY, DARK_GREY],
        [DARK_GREY, LIGHT_GREY, LIGHT_GREY, DARK_GREY],
        [DARK_GREY, LIGHT_GREY, LIGHT_GREY, DARK_GREY],
    ]


def _get_horizontal_pipe_directional_pixels() -> List[List[int]]:
    return [
        [DARK_GREY, DARK_GREY, DARK_GREY, DARK_GREY],
        [DARK_GREY, LIGHT_GREY, LIGHT_GREY, YELLOW],
        [DARK_GREY, LIGHT_GREY, LIGHT_GREY, YELLOW],
        [DARK_GREY, DARK_GREY, DARK_GREY, DARK_GREY],
    ]


def _get_vertical_pipe_directional_pixels() -> List[List[int]]:
    return [
        [DARK_GREY, DARK_GREY, DARK_GREY, DARK_GREY],
        [DARK_GREY, LIGHT_GREY, LIGHT_GREY, DARK_GREY],
        [DARK_GREY, LIGHT_GREY, LIGHT_GREY, DARK_GREY],
        [DARK_GREY, YELLOW, YELLOW, DARK_GREY],
    ]


def _get_bottom_left_corner_pixels() -> List[List[int]]:
    return [
        [DARK_GREY, LIGHT_GREY, LIGHT_GREY, DARK_GREY],
        [DARK_GREY, LIGHT_GREY, LIGHT_GREY, LIGHT_GREY],
        [DARK_GREY, LIGHT_GREY, LIGHT_GREY, LIGHT_GREY],
        [DARK_GREY, DARK_GREY, DARK_GREY, DARK_GREY],
    ]


def _get_bottom_right_corner_pixels() -> List[List[int]]:
    return [
        [DARK_GREY, LIGHT_GREY, LIGHT_GREY, DARK_GREY],
        [LIGHT_GREY, LIGHT_GREY, LIGHT_GREY, DARK_GREY],
        [LIGHT_GREY, LIGHT_GREY, LIGHT_GREY, DARK_GREY],
        [DARK_GREY, DARK_GREY, DARK_GREY, DARK_GREY],
    ]


def _get_top_left_corner_pixels() -> List[List[int]]:
    return [
        [DARK_GREY, DARK_GREY, DARK_GREY, DARK_GREY],
        [DARK_GREY, LIGHT_GREY, LIGHT_GREY, LIGHT_GREY],
        [DARK_GREY, LIGHT_GREY, LIGHT_GREY, LIGHT_GREY],
        [DARK_GREY, LIGHT_GREY, LIGHT_GREY, DARK_GREY],
    ]


def _get_top_right_corner_pixels() -> List[List[int]]:
    return [
        [DARK_GREY, DARK_GREY, DARK_GREY, DARK_GREY],
        [LIGHT_GREY, LIGHT_GREY, LIGHT_GREY, DARK_GREY],
        [LIGHT_GREY, LIGHT_GREY, LIGHT_GREY, DARK_GREY],
        [DARK_GREY, LIGHT_GREY, LIGHT_GREY, DARK_GREY],
    ]


def _pipe_pixels_colored(tile: str, color: int, center_color: int) -> List[List[int]]:
    base = _tile_pixels(tile)
    result: List[List[int]] = []
    for r, row in enumerate(base):
        new_row: List[int] = []
        for c, val in enumerate(row):
            if val == LIGHT_GREY or (r in (1, 2) and c in (1, 2) and val != DARK_GREY):
                new_row.append(color)
            else:
                new_row.append(val)
        result.append(new_row)
    return result


def _cursor_pixels() -> List[List[int]]:
    Y = YELLOW
    return [
        [Y, Y, Y, Y],
        [Y, -1, -1, Y],
        [Y, -1, -1, Y],
        [Y, Y, Y, Y],
    ]


def _cursor_pixels_colored(color: int) -> List[List[int]]:
    return [
        [color, color, color, color],
        [color, -1, -1, color],
        [color, -1, -1, color],
        [color, color, color, color],
    ]


def _life_on_pixels() -> List[List[int]]:
    R = RED
    DG = DARK_GREY
    return [
        [DG, R, R, DG],
        [R, R, R, R],
        [R, R, R, R],
        [DG, R, R, DG],
    ]


def _life_off_pixels() -> List[List[int]]:
    DG = DARK_GREY
    B = BLACK
    return [
        [B, DG, DG, B],
        [DG, B, B, DG],
        [DG, B, B, DG],
        [B, DG, DG, B],
    ]


def _pip_on_pixels() -> List[List[int]]:
    G = GREEN
    DG = DARK_GREY
    return [
        [DG, DG, DG, DG],
        [G, G, G, G],
        [G, G, G, G],
        [DG, DG, DG, DG],
    ]


def _pip_used_pixels() -> List[List[int]]:
    DG = DARK_GREY
    B = BLACK
    return [
        [DG, DG, DG, DG],
        [DG, DG, DG, DG],
        [DG, DG, DG, DG],
        [DG, DG, DG, DG],
    ]


def _find_source_position(grid_strs: List[str]) -> Tuple[int, int]:
    for row, row_str in enumerate(grid_strs):
        for col, char in enumerate(row_str):
            if char == T_SOURCE:
                return row, col
    return 0, 0


def _make_sprite(
    name: str,
    row: int,
    col: int,
    pixels: List[List[int]],
    layer: int = 0,
) -> Sprite:
    return Sprite(
        pixels=pixels,
        name=name,
        x=col * TILE,
        y=row * TILE,
        layer=layer,
        blocking=BlockingMode.NOT_BLOCKED,
        visible=True,
        collidable=False,
    )


def _grid_offset(grid_rows: int, grid_cols: int) -> Tuple[int, int]:
    return max(0, (GRID_H - grid_rows) // 2), max(0, (GRID_W - grid_cols) // 2)


def _build_level(grid_strs: List[str], level_name: str) -> Level:
    grid_rows = len(grid_strs)
    grid_cols = len(grid_strs[0]) if grid_rows > 0 else 0
    row_offset, col_offset = _grid_offset(grid_rows, grid_cols)

    sprites: List[Sprite] = []
    sprites.extend(_create_background_sprites())
    sprites.extend(_create_slot_sprites(grid_rows, grid_cols, row_offset, col_offset))
    sprites.extend(_create_tile_sprites(grid_strs, row_offset, col_offset))
    sprites.extend(_create_cursor_sprite(grid_strs, row_offset, col_offset))
    sprites.extend(_create_life_sprites())
    sprites.extend(_create_step_pip_sprites())

    return Level(
        sprites=sprites,
        grid_size=(GRID_W * TILE, GRID_H * TILE),
        name=level_name,
    )


def _create_background_sprites() -> List[Sprite]:
    background_pixels = [
        [BLACK, BLACK, BLACK, BLACK],
        [BLACK, BLACK, BLACK, BLACK],
        [BLACK, BLACK, BLACK, BLACK],
        [BLACK, BLACK, BLACK, BLACK],
    ]
    sprites = []
    for row in range(GRID_H):
        for col in range(GRID_W):
            sprites.append(
                _make_sprite(f"bg_{row}_{col}", row, col, background_pixels, layer=0)
            )
    return sprites


def _create_slot_sprites(
    grid_rows: int, grid_cols: int, row_offset: int, col_offset: int
) -> List[Sprite]:
    dot_pixels = [
        [BLACK, BLACK, BLACK, BLACK],
        [BLACK, DARK_GREY, DARK_GREY, BLACK],
        [BLACK, DARK_GREY, DARK_GREY, BLACK],
        [BLACK, BLACK, BLACK, BLACK],
    ]
    sprites = []
    for row in range(grid_rows):
        for col in range(grid_cols):
            sprite_row, sprite_col = row_offset + row, col_offset + col
            sprites.append(
                _make_sprite(
                    f"slot_{row}_{col}", sprite_row, sprite_col, dot_pixels, layer=1
                )
            )
    return sprites


def _create_tile_sprites(
    grid_strs: List[str], row_offset: int, col_offset: int
) -> List[Sprite]:
    sprites = []
    for row, row_str in enumerate(grid_strs):
        for col, char in enumerate(row_str):
            sprite_row, sprite_col = row_offset + row, col_offset + col
            sprites.append(
                _make_sprite(
                    f"tile_{row}_{col}",
                    sprite_row,
                    sprite_col,
                    _tile_pixels(char),
                    layer=2,
                )
            )
    return sprites


def _create_cursor_sprite(
    grid_strs: List[str], row_offset: int, col_offset: int
) -> List[Sprite]:
    source_row, source_col = _find_source_position(grid_strs)
    cursor_sprite = _make_sprite(
        "cursor",
        row_offset + source_row,
        col_offset + source_col,
        _cursor_pixels(),
        layer=3,
    )
    return [cursor_sprite]


def _create_life_sprites() -> List[Sprite]:
    sprites = []
    for life_index in range(MAX_LIVES):
        sprite_col = 1 + life_index * 2
        sprites.append(
            _make_sprite(
                f"life_{life_index}", LIVES_ROW, sprite_col, _life_on_pixels(), layer=1
            )
        )
    return sprites


def _create_step_pip_sprites() -> List[Sprite]:
    sprites = []
    pip_offset = max(1, (GRID_W - PIP_COUNT) // 2)
    for step_index in range(PIP_COUNT):
        sprite_col = pip_offset + step_index
        if sprite_col < GRID_W:
            sprites.append(
                _make_sprite(
                    f"pip_{step_index}",
                    STEPS_ROW,
                    sprite_col,
                    _pip_on_pixels(),
                    layer=1,
                )
            )
    return sprites


ARC_PALETTE = [
    [255, 255, 255],
    [204, 204, 204],
    [153, 153, 153],
    [102, 102, 102],
    [51,  51,  51],
    [0,   0,   0],
    [229, 58,  163],
    [255, 123, 204],
    [249, 60,  49],
    [30,  147, 255],
    [136, 216, 241],
    [255, 220, 0],
    [255, 133, 27],
    [146, 18,  49],
    [79,  204, 48],
    [163, 86,  208],
]
ARC_PALETTE = np.array(ARC_PALETTE, dtype=np.uint8)

_COLOR_CHAR: Dict[int, str] = {
    BLACK: "K",
    BLUE: "B",
    GREEN: "G",
    YELLOW: "Y",
    RED: "R",
    MAROON: "M",
    LIGHT_GREY: "L",
    DARK_GREY: "D",
}

BACKGROUND_COLOR = BLACK
PADDING_COLOR = BLACK


class Pg05(ARCBaseGame):

    def __init__(self, seed: int = 0) -> None:
        self._consecutive_resets: int = 0
        self._game_over_state: bool = False
        self._game_won: bool = False
        self._grid: List[List[str]] = []
        self._grid_rows: int = 0
        self._grid_cols: int = 0
        self._row_off: int = 0
        self._col_off: int = 0
        self._cursor_r: int = 0
        self._cursor_c: int = 0
        self._steps_remaining: int = 0
        self._max_steps: int = 0
        self._lives: int = MAX_LIVES
        self._transition_pending: Optional[str] = None
        self._cached_propagation: Optional[Dict[str, Any]] = None
        self._engine_snapshots: List[Dict[str, Any]] = []

        self._seed = seed
        self._reset_counter: int = 0
        self._rng = random.Random(seed)

        levels = self._build_all_levels()

        camera = Camera(
            x=0, y=0,
            width=64, height=64,
            background=BACKGROUND_COLOR,
            letter_box=PADDING_COLOR,
        )

        super().__init__(
            game_id="pg05",
            levels=levels,
            camera=camera,
            available_actions=[0, 1, 2, 3, 4, 5, 7],
        )

    def on_set_level(self, level: Level) -> None:
        self._reset_counter += 1
        self._rng = random.Random(self._seed + self._current_level_index * 1000 + self._reset_counter)
        self._parse_level()
        self._sync_all_tiles()
        self._sync_cursor()
        self._sync_lives()
        self._sync_step_pips()
        result = self._propagate_signal()
        self._cached_propagation = result
        self._sync_pipe_colors(
            result["flow_nodes"],
            result.get("blocked", set()),
            result.get("decay_strengths"),
        )
        self._transition_pending = None
        self._engine_snapshots = []

    def _engine_save_snapshot(self) -> None:
        self._engine_snapshots.append({
            "grid": deepcopy(self._grid),
            "cursor_r": self._cursor_r,
            "cursor_c": self._cursor_c,
            "steps_remaining": self._steps_remaining,
            "transition_pending": self._transition_pending,
        })

    def _engine_restore_snapshot(self) -> None:
        if not self._engine_snapshots:
            return
        snap = self._engine_snapshots.pop()
        self._grid = snap["grid"]
        self._cursor_r = snap["cursor_r"]
        self._cursor_c = snap["cursor_c"]
        self._transition_pending = snap["transition_pending"]
        self._sync_all_tiles()
        self._sync_cursor()
        result = self._propagate_signal()
        self._cached_propagation = result
        self._sync_pipe_colors(
            result["flow_nodes"],
            result.get("blocked", set()),
            result.get("decay_strengths"),
        )

    def step(self):
        if self._game_over_state or self._game_won:
            self.complete_action()
            return

        if self._action.id == GameAction.RESET:
            self.complete_action()
            return

        self._consecutive_resets = 0
        self._process_action(self._action.id.value)
        self.complete_action()

    def _process_action(self, action_id: int) -> None:
        if self._transition_pending is not None:
            self._consume_transition()
            return

        if action_id == 7:
            self._engine_restore_snapshot()
            self._steps_remaining -= 1
            self._sync_step_pips()
            if self._steps_remaining <= 0 and not self._game_won:
                self._lives -= 1
                self._sync_lives()
                if self._lives <= 0:
                    self._transition_pending = "lose"
                else:
                    self._transition_pending = "fail"
            return

        self._engine_save_snapshot()

        performed_rotation = self._execute_action(action_id)

        self._steps_remaining -= 1
        self._sync_step_pips()

        if performed_rotation:
            result = self._propagate_signal()
            self._cached_propagation = result
            self._sync_pipe_colors(
                result["flow_nodes"],
                result.get("blocked", set()),
                result.get("decay_strengths"),
            )

            if result["total"] > 0 and result["powered"] == result["total"]:
                self._transition_pending = "win"
                self._consume_transition()
                return

        if self._steps_remaining <= 0:
            self._lives -= 1
            self._sync_lives()
            if self._lives <= 0:
                self._transition_pending = "lose"
            else:
                self._transition_pending = "fail"

    def _execute_action(self, action_id: int) -> bool:
        if action_id == 1:
            self._do_move(-1, 0)
            return False
        if action_id == 2:
            self._do_move(1, 0)
            return False
        if action_id == 3:
            self._do_move(0, -1)
            return False
        if action_id == 4:
            self._do_move(0, 1)
            return False
        if action_id == 5:
            return self._do_rotate()
        return False

    def handle_reset(self) -> None:
        if self._game_won:
            self.full_reset()
            return

        if self._game_over_state:
            self._lives = MAX_LIVES
            self._reset_current_level()
            self._game_over_state = False
            return

        self._consecutive_resets += 1
        if self._consecutive_resets >= 2:
            self._consecutive_resets = 0
            self.set_level(0)
        else:
            self._lives = MAX_LIVES
            self._reset_current_level()

    def _reset_current_level(self) -> None:
        self._game_over_state = False
        self._game_won = False
        self._engine_snapshots.clear()
        self.level_reset()

    def full_reset(self) -> None:
        self._current_level_index = 0
        self._consecutive_resets = 0
        self._game_over_state = False
        self._game_won = False
        self._lives = MAX_LIVES
        self._engine_snapshots.clear()
        self.level_reset()

    def _restart_full_game(self) -> None:
        self.full_reset()

    def reset(self) -> dict[str, Any]:
        self._restart_full_game()
        return self.get_state()

    def get_actions(self) -> List[int]:
        if self._game_over_state or self._game_won or self._lives <= 0:
            return []
        return [1, 2, 3, 4, 5, 7]

    def get_state(self) -> dict[str, Any]:
        if self._cached_propagation is not None:
            result = self._cached_propagation
        else:
            result = self._propagate_signal()
            self._cached_propagation = result

        text_obs = (
            f"Power Grid Level {self._current_level_index + 1}: "
            f"Connect {result['total']} cities. Lives: {self._lives}/{MAX_LIVES}, "
            f"Steps: {self._steps_remaining}/{self._max_steps}, "
            f"Cursor: ({self._cursor_r},{self._cursor_c})"
        )

        if result["total"] > 0:
            progress = result["powered"] / result["total"]
            text_obs += f" Progress: {int(progress * 100)}%"

        if self._transition_pending == "win":
            text_obs += " LEVEL COMPLETE!"
        elif self._transition_pending == "lose":
            text_obs += " GAME OVER"
        elif result["powered"] > 0:
            text_obs += f" ({result['powered']}/{result['total']} powered)"

        done = (
            self._transition_pending in ("lose", "win")
            or self._game_over_state
            or self._game_won
        )

        return {
            "text_observation": text_obs,
            "image_observation": None,
            "available_actions": self.get_actions(),
            "turn": self._max_steps - self._steps_remaining,
            "metadata": {
                "total_levels": len(self._levels),
                "level_index": self._current_level_index,
                "levels_completed": self._current_level_index,
                "game_over": self._game_over_state,
                "done": done,
                "info": {
                    "win": self._transition_pending == "win" or self._game_won,
                    "loss": self._transition_pending == "lose" or self._game_over_state,
                    "turn_limit_reached": self._steps_remaining <= 0,
                    "no_valid_actions": len(self.get_actions()) == 0,
                    "level": self._current_level_index + 1,
                    "lives": self._lives,
                    "steps_remaining": self._steps_remaining,
                    "cities_connected": result["powered"],
                    "cities_total": result["total"],
                    "propagation": result,
                },
            },
        }

    def _propagate_signal(self) -> Dict[str, Any]:
        source = self._find_source()
        cities = set(self._find_cities())
        cities_total = len(cities)

        if source is None or cities_total == 0:
            return {
                "powered": 0,
                "total": cities_total,
                "flow_nodes": set(),
                "blocked": set(),
                "decay_strengths": None,
            }

        rule = LEVEL_RULES[self._current_level_index]
        rule_type = rule["type"]
        is_decay = rule_type == "decay"
        is_directional = rule_type == "directional"
        is_hidden = rule_type == "hidden"

        visited: Set[Tuple[int, int]] = set()
        blocked: Set[Tuple[int, int]] = set()
        decay_strengths: Optional[Dict[Tuple[int, int], int]] = None

        if is_decay:
            max_strength = max(8, cities_total * 2)
            decay_strengths = {}
            queue: deque = deque([(source, max_strength)])
            decay_strengths[source] = max_strength
            visited.add(source)
        else:
            queue = deque([source])
            visited.add(source)

        while queue:
            strength = 0
            if is_decay:
                pos, strength = queue.popleft()
                if strength <= 0:
                    continue
                row, col = pos
            else:
                pos = queue.popleft()
                row, col = pos

            tile = self._grid[row][col]

            if is_directional:
                directions = _get_directional_outgoing(tile)
            elif is_hidden:
                directions = _HIDDEN_CONNECTION_MAP.get(tile, _EMPTY_CONNECTION)
            else:
                directions = _PIPE_CONNECTION_MAP.get(tile, _EMPTY_CONNECTION)

            for direction in directions:
                dr, dc = DIRS[direction]
                nr, nc = row + dr, col + dc
                if not self._in_bounds(nr, nc):
                    continue

                neighbor = self._grid[nr][nc]
                npos = (nr, nc)

                if neighbor == T_ENEMY:
                    blocked.add(npos)
                    continue

                if is_directional:
                    incoming = _get_directional_incoming(neighbor)
                    if OPPOSITE[direction] not in incoming:
                        blocked.add(npos)
                        continue
                elif is_hidden:
                    neighbor_conns = _HIDDEN_CONNECTION_MAP.get(neighbor, _EMPTY_CONNECTION)
                    if OPPOSITE[direction] not in neighbor_conns:
                        blocked.add(npos)
                        continue
                else:
                    neighbor_conns = _PIPE_CONNECTION_MAP.get(neighbor, _EMPTY_CONNECTION)
                    if OPPOSITE[direction] not in neighbor_conns:
                        blocked.add(npos)
                        continue

                if is_decay:
                    assert decay_strengths is not None
                    new_str = strength - (3 if neighbor == T_TRAP else 1)
                    if new_str <= 0:
                        continue
                    prev = decay_strengths.get(npos, -1)
                    if new_str > prev:
                        decay_strengths[npos] = new_str
                        visited.add(npos)
                        queue.append((npos, new_str))
                else:
                    if npos not in visited:
                        visited.add(npos)
                        queue.append(npos)

        if is_decay:
            assert decay_strengths is not None
            powered = sum(
                1 for c in cities
                if c in visited and decay_strengths.get(c, 0) > 0
            )
        else:
            powered = sum(1 for c in cities if c in visited)

        return {
            "powered": powered,
            "total": cities_total,
            "flow_nodes": visited,
            "blocked": blocked,
            "decay_strengths": decay_strengths,
        }

    def _estimate_baseline_length(self) -> int:
        return BASELINE_STEPS[self._current_level_index]

    def _build_all_levels(self) -> List[Level]:
        levels: List[Level] = []
        for pos, grid_strs in enumerate(LEVEL_GRIDS):
            lvl = _build_level(grid_strs, level_name=f"level_{pos + 1}")
            levels.append(lvl)
        return levels

    def _parse_level(self) -> None:
        level_grid: List[str] = LEVEL_GRIDS[self._current_level_index]
        self._grid = [list(row) for row in level_grid]

        if not self._grid or not self._grid[0]:
            raise ValueError("Level grid is empty or malformed")

        self._grid_rows = len(self._grid)
        self._grid_cols = len(self._grid[0])

        for row_index, row in enumerate(self._grid):
            if len(row) != self._grid_cols:
                raise ValueError(
                    f"Row {row_index} length {len(row)} != expected {self._grid_cols}"
                )

        self._row_off, self._col_off = _grid_offset(self._grid_rows, self._grid_cols)
        base_steps = self._estimate_baseline_length()
        self._max_steps = base_steps * 4
        self._steps_remaining = self._max_steps

        self._cursor_r = self._rng.randint(0, self._grid_rows - 1)
        self._cursor_c = self._rng.randint(0, self._grid_cols - 1)

    def _do_move(self, delta_row: int, delta_col: int) -> None:
        new_row = max(0, min(self._grid_rows - 1, self._cursor_r + delta_row))
        new_col = max(0, min(self._grid_cols - 1, self._cursor_c + delta_col))
        self._cursor_r = new_row
        self._cursor_c = new_col
        self._sync_cursor()

    def _do_rotate(self) -> bool:
        cursor_row, cursor_col = self._cursor_r, self._cursor_c
        current_tile = self._grid[cursor_row][cursor_col]

        if current_tile not in ROTATABLE:
            return False

        self._grid[cursor_row][cursor_col] = _rotate_tile(current_tile)
        self._sync_tile(cursor_row, cursor_col)
        self._sync_cursor()
        return True

    def _consume_transition(self) -> None:
        pending = self._transition_pending
        self._transition_pending = None

        if pending == "win":
            self._lives = MAX_LIVES
            self._sync_lives()
            if self.is_last_level():
                self._game_won = True
            self.next_level()
        elif pending == "fail":
            self.level_reset()
        elif pending == "lose":
            self.lose()

    def _sync_cursor(self) -> None:
        sr = self._row_off + self._cursor_r
        sc = self._col_off + self._cursor_c
        sprites = self.current_level.get_sprites_by_name("cursor")
        if sprites:
            sprites[0].set_position(sc * TILE, sr * TILE)
            tile = self._grid[self._cursor_r][self._cursor_c]
            can_rotate = tile in ROTATABLE
            color = GREEN if can_rotate else RED
            sprites[0].pixels = np.array(
                _cursor_pixels_colored(color), dtype=np.int8
            )

    def _sync_tile(self, row: int, col: int) -> None:
        current_tile = self._grid[row][col]
        tile_pixels = np.array(_tile_pixels(current_tile), dtype=np.int8)
        tile_sprites = self.current_level.get_sprites_by_name(f"tile_{row}_{col}")
        if tile_sprites:
            tile_sprites[0].pixels = tile_pixels

    def _sync_all_tiles(self) -> None:
        for row_index in range(self._grid_rows):
            for col_index in range(self._grid_cols):
                self._sync_tile(row_index, col_index)

    def _sync_pipe_colors(
        self,
        flow_nodes: Set[Tuple[int, int]],
        blocked: Optional[Set[Tuple[int, int]]] = None,
        decay_strengths: Optional[Dict[Tuple[int, int], int]] = None,
    ) -> None:
        for row_index in range(self._grid_rows):
            for col_index in range(self._grid_cols):
                tile = self._grid[row_index][col_index]
                tile_sprites = self.current_level.get_sprites_by_name(
                    f"tile_{row_index}_{col_index}"
                )
                if not tile_sprites:
                    continue
                pos = (row_index, col_index)
                pixels = self._resolve_tile_pixels(
                    tile, pos, flow_nodes, blocked, decay_strengths
                )
                if pixels is not None:
                    tile_sprites[0].pixels = np.array(pixels, dtype=np.int8)

    def _resolve_tile_pixels(
        self,
        tile: str,
        pos: Tuple[int, int],
        flow_nodes: Set[Tuple[int, int]],
        blocked: Optional[Set[Tuple[int, int]]],
        decay_strengths: Optional[Dict[Tuple[int, int], int]],
    ) -> Optional[List[List[int]]]:
        if tile == T_CITY:
            if pos in flow_nodes:
                return _get_city_powered_pixels()
            return _get_city_tile_pixels()
        if tile == T_TRAP:
            if pos in flow_nodes:
                return [
                    [DARK_GREY, RED, RED, DARK_GREY],
                    [RED, RED, RED, RED],
                    [RED, RED, RED, RED],
                    [DARK_GREY, RED, RED, DARK_GREY],
                ]
            return _get_trap_tile_pixels()
        if tile == T_ENEMY:
            return _get_enemy_tile_pixels()
        if tile == T_BLOCKED:
            return _get_blocked_tile_pixels()
        if tile not in ROTATABLE:
            return None
        if pos in flow_nodes:
            return self._flow_pipe_pixels(tile, pos, decay_strengths)
        if blocked is not None and self._current_level_index > 0 and pos in blocked:
            return _pipe_pixels_colored(tile, RED, RED)
        return self._default_pipe_pixels(tile)

    def _flow_pipe_pixels(
        self,
        tile: str,
        pos: Tuple[int, int],
        decay_strengths: Optional[Dict[Tuple[int, int], int]],
    ) -> List[List[int]]:
        if self._current_level_index == 2 and decay_strengths is not None:
            strength = decay_strengths.get(pos, 0)
            max_strength = max(8, len(self._find_cities()) * 2)
            threshold_high = max_strength * 2 // 3
            threshold_low = max_strength // 3
            if strength >= threshold_high:
                return _pipe_pixels_colored(tile, GREEN, GREEN)
            if strength >= threshold_low:
                return _pipe_pixels_colored(tile, YELLOW, YELLOW)
            return _pipe_pixels_colored(tile, RED, RED)
        return _pipe_pixels_colored(tile, GREEN, GREEN)

    def _default_pipe_pixels(self, tile: str) -> List[List[int]]:
        if self._current_level_index == 1:
            if tile == T_HORIZ:
                return _get_horizontal_pipe_directional_pixels()
            if tile == T_VERT:
                return _get_vertical_pipe_directional_pixels()
        if self._current_level_index == 3:
            if tile in ROTATABLE:
                base = _tile_pixels(tile)
                base[1][1] = MAROON
                base[1][2] = MAROON
                base[2][1] = MAROON
                base[2][2] = MAROON
                return base
        return _tile_pixels(tile)

    def _sync_lives(self) -> None:
        for life_index in range(MAX_LIVES):
            life_sprites = self.current_level.get_sprites_by_name(f"life_{life_index}")
            if life_sprites:
                life_pixels = (
                    _life_on_pixels()
                    if life_index < self._lives
                    else _life_off_pixels()
                )
                life_sprites[0].pixels = np.array(life_pixels, dtype=np.int8)

    def _sync_step_pips(self) -> None:
        steps_used = self._max_steps - self._steps_remaining
        pips_used = min(PIP_COUNT, (steps_used * PIP_COUNT) // max(1, self._max_steps))

        for step_index in range(PIP_COUNT):
            pip_sprites = self.current_level.get_sprites_by_name(f"pip_{step_index}")
            if pip_sprites:
                pip_pixels = (
                    _pip_used_pixels() if step_index < pips_used else _pip_on_pixels()
                )
                pip_sprites[0].pixels = np.array(pip_pixels, dtype=np.int8)

    def _in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self._grid_rows and 0 <= col < self._grid_cols

    def _find_source(self) -> Optional[Tuple[int, int]]:
        for row_index, row in enumerate(self._grid):
            for col_index, tile_char in enumerate(row):
                if tile_char == T_SOURCE:
                    return (row_index, col_index)
        return None

    def _find_cities(self) -> List[Tuple[int, int]]:
        return [
            (row_index, col_index)
            for row_index, row in enumerate(self._grid)
            for col_index, tile_char in enumerate(row)
            if tile_char == T_CITY
        ]


class PuzzleEnvironment:

    _ACTION_MAP: Dict[str, GameAction] = {
        "reset":  GameAction.RESET,
        "up":     GameAction.ACTION1,
        "down":   GameAction.ACTION2,
        "left":   GameAction.ACTION3,
        "right":  GameAction.ACTION4,
        "select": GameAction.ACTION5,
        "undo":   GameAction.ACTION7,
    }

    _VALID_ACTIONS = list(_ACTION_MAP.keys())

    def __init__(self, seed: int = 0) -> None:
        self._engine: Any = Pg05(seed=seed)
        self._total_turns = 0
        self._done = False
        self._game_won = False
        self._game_over = False
        self._last_action_was_reset = False

    @staticmethod
    def _encode_png(frame: np.ndarray) -> bytes:
        h, w = frame.shape[:2]
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for idx in range(len(ARC_PALETTE)):
            mask = frame == idx
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            rgb[mask] = ARC_PALETTE[idx]

        buf = io.BytesIO()
        buf.write(b"\x89PNG\r\n\x1a\n")

        def _chunk(ctype: bytes, data: bytes) -> None:
            buf.write(struct.pack(">I", len(data)))
            buf.write(ctype)
            buf.write(data)
            buf.write(struct.pack(">I", zlib.crc32(ctype + data) & 0xFFFFFFFF))

        _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        raw = b""
        for y in range(h):
            raw += b"\x00" + rgb[y].tobytes()
        _chunk(b"IDAT", zlib.compress(raw))
        _chunk(b"IEND", b"")
        return buf.getvalue()

    def _render_frame(self) -> np.ndarray:
        e = self._engine
        return e.camera.render(e.current_level.get_sprites())

    def _build_text_observation(self) -> str:
        e = self._engine
        grid_rows = e._grid_rows
        grid_cols = e._grid_cols

        grid_lines: List[str] = []
        for r in range(grid_rows):
            row_chars: List[str] = []
            for c in range(grid_cols):
                tile = e._grid[r][c]
                if (r, c) == (e._cursor_r, e._cursor_c):
                    row_chars.append("[" + tile + "]")
                else:
                    row_chars.append(" " + tile + " ")
            grid_lines.append("".join(row_chars))

        prop = e._cached_propagation
        powered = prop["powered"] if prop else 0
        total = prop["total"] if prop else 0

        header = (
            f"Level {e._current_level_index + 1}/{len(e._levels)}"
            f" | Steps: {e._steps_remaining}/{e._max_steps}"
            f" | Lives: {e._lives}"
            f" | Cities: {powered}/{total}"
        )
        rules = (
            "Move cursor with arrows. Select pipe (action5) to rotate. "
            "Connect all cities to the source to win. Undo with action7."
        )
        return header + "\n" + rules + "\nBoard:\n" + "\n".join(grid_lines)

    def _build_game_state(self, done: bool = False) -> GameState:
        e = self._engine
        frame = self._render_frame()
        image_bytes = self._encode_png(frame)

        valid_actions = self.get_actions() if not done else None

        return GameState(
            text_observation=self._build_text_observation(),
            image_observation=image_bytes,
            valid_actions=valid_actions,
            turn=self._total_turns,
            metadata={
                "total_levels": len(e._levels),
                "level_index": e._current_level_index,
                "levels_completed": e._current_level_index,
                "game_over": self._game_over,
                "done": done,
                "info": {},
            },
        )

    def reset(self) -> GameState:
        e = self._engine

        e.perform_action(ActionInput(id=GameAction.RESET))

        self._total_turns = 0
        self._done = False
        self._last_action_was_reset = True
        self._game_won = False
        self._game_over = False

        return self._build_game_state()

    def is_done(self) -> bool:
        return self._done

    def get_actions(self) -> List[str]:
        if self._done:
            return ["reset"]
        return list(self._VALID_ACTIONS)

    @staticmethod
    def _parse_action(action: str) -> Tuple[str, Optional[int], Optional[int]]:
        parts = action.strip().split()
        base = parts[0].lower() if parts else action.lower()
        row: Optional[int] = None
        col: Optional[int] = None
        if len(parts) >= 3:
            try:
                row = int(float(parts[1]))
                col = int(float(parts[2]))
            except (ValueError, OverflowError):
                row, col = None, None
        return base, row, col

    def step(self, action: str) -> StepResult:
        e = self._engine

        base_action, coord_row, coord_col = self._parse_action(action)

        if base_action == "reset":
            state = self.reset()
            return StepResult(
                state=state, reward=0.0, done=False, info={"action": "reset"}
            )

        if base_action not in self._ACTION_MAP:
            return StepResult(
                state=self._build_game_state(done=False),
                reward=0.0,
                done=False,
                info={"action": action, "reason": "invalid_action"},
            )

        self._last_action_was_reset = False
        self._total_turns += 1

        game_action = self._ACTION_MAP[base_action]
        info: Dict[str, Any] = {"action": action}

        if coord_row is not None and coord_col is not None:
            clamped_row = max(0, min(e._grid_rows - 1, coord_row))
            clamped_col = max(0, min(e._grid_cols - 1, coord_col))
            e._cursor_r = clamped_row
            e._cursor_c = clamped_col
            e._sync_cursor()

        level_before = e._current_level_index

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
            self._game_over = False
            info["reason"] = "game_complete"
            return StepResult(
                state=self._build_game_state(done=True),
                reward=level_reward,
                done=True,
                info=info,
            )

        if game_over or e._game_over_state:
            self._done = True
            self._game_won = False
            self._game_over = True
            info["reason"] = "game_over"
            return StepResult(
                state=self._build_game_state(done=True),
                reward=0.0,
                done=True,
                info=info,
            )

        reward = 0.0
        if e._current_level_index != level_before:
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
        index_grid = self._render_frame()
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

    env = ArcGameEnv(seed=42, render_mode="rgb_array")
    check_env(env.unwrapped, skip_render_check=True)

    obs, info = env.reset()

    mask = env.action_mask()
    valid_indices = np.where(mask == 1)[0]
    if len(valid_indices) > 0:
        obs, reward, term, trunc, info = env.step(valid_indices[0])

    env.close()
