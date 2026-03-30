import io
import random
import struct
import zlib
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

import gymnasium as gym
import numpy as np
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
    metadata: Dict = field(default_factory=dict)


@dataclass
class StepResult:
    state: GameState
    reward: float
    done: bool
    info: Dict = field(default_factory=dict)


BACKGROUND_COLOR = 0
PADDING_COLOR = 4
MAX_LIVES = 3
TILE = 5


sprites = {
    "player": Sprite(
        pixels=[
            [-1, 8, 8, 8, -1],
            [8, 8, 7, 8, 8],
            [8, 7, 8, 7, 8],
            [8, 8, 7, 8, 8],
            [-1, 8, 8, 8, -1],
        ],
        name="player",
        visible=True,
        collidable=True,
        tags=["player"],
        layer=3,
    ),
    "wall": Sprite(
        pixels=[
            [5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5],
        ],
        name="wall",
        visible=True,
        collidable=True,
        tags=["wall", "solid"],
        layer=0,
    ),
    "floor_off": Sprite(
        pixels=[
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        name="floor_off",
        visible=True,
        collidable=False,
        tags=["floor", "floor_off"],
        layer=-1,
    ),
    "floor_on": Sprite(
        pixels=[
            [4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4],
        ],
        name="floor_on",
        visible=True,
        collidable=False,
        tags=["floor", "floor_on"],
        layer=-1,
    ),
    "gate_locked": Sprite(
        pixels=[
            [2, 2, 2, 2, 2],
            [2, 0, 2, 0, 2],
            [2, 2, 0, 2, 2],
            [2, 0, 2, 0, 2],
            [2, 2, 2, 2, 2],
        ],
        name="gate_locked",
        visible=True,
        collidable=True,
        tags=["gate", "solid"],
        layer=1,
    ),
    "gate_open": Sprite(
        pixels=[
            [3, 3, 3, 3, 3],
            [3, 7, 3, 7, 3],
            [3, 3, 7, 3, 3],
            [3, 7, 3, 7, 3],
            [3, 3, 3, 3, 3],
        ],
        name="gate_open",
        visible=True,
        collidable=False,
        tags=["gate_open", "exit"],
        layer=1,
    ),
    "clue_wall": Sprite(
        pixels=[
            [9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9],
        ],
        name="clue_wall",
        visible=True,
        collidable=True,
        tags=["clue_wall", "solid"],
        layer=0,
    ),
    "clue_on": Sprite(
        pixels=[
            [4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4],
        ],
        name="clue_on",
        visible=True,
        collidable=True,
        tags=["clue_on", "solid"],
        layer=0,
    ),
    "clue_off": Sprite(
        pixels=[
            [9, 9, 9, 9, 9],
            [9, 0, 0, 0, 9],
            [9, 0, 0, 0, 9],
            [9, 0, 0, 0, 9],
            [9, 9, 9, 9, 9],
        ],
        name="clue_off",
        visible=True,
        collidable=True,
        tags=["clue_off", "solid"],
        layer=0,
    ),
    "sym_triangle": Sprite(
        pixels=[
            [-1, -1, 2, -1, -1],
            [-1, 2, 2, 2, -1],
            [2, 2, 2, 2, 2],
            [-1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1],
        ],
        name="sym_triangle",
        visible=True,
        collidable=True,
        tags=["symbol", "solid"],
        layer=1,
    ),
    "sym_square": Sprite(
        pixels=[
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [-1, -1, -1, -1, -1],
        ],
        name="sym_square",
        visible=True,
        collidable=True,
        tags=["symbol", "solid"],
        layer=1,
    ),
    "sym_circle": Sprite(
        pixels=[
            [-1, 3, 3, 3, -1],
            [3, 3, 3, 3, 3],
            [3, 3, 3, 3, 3],
            [-1, 3, 3, 3, -1],
            [-1, -1, -1, -1, -1],
        ],
        name="sym_circle",
        visible=True,
        collidable=True,
        tags=["symbol", "solid"],
        layer=1,
    ),
    "arrow_right": Sprite(
        pixels=[
            [-1, -1, 7, -1, -1],
            [-1, -1, -1, 7, -1],
            [7, 7, 7, 7, 7],
            [-1, -1, -1, 7, -1],
            [-1, -1, 7, -1, -1],
        ],
        name="arrow_right",
        visible=True,
        collidable=True,
        tags=["arrow", "solid"],
        layer=1,
    ),
    "reset_pad": Sprite(
        pixels=[
            [6, -1, 6, -1, 6],
            [-1, 6, -1, 6, -1],
            [6, -1, 6, -1, 6],
            [-1, 6, -1, 6, -1],
            [6, -1, 6, -1, 6],
        ],
        name="reset_pad",
        visible=True,
        collidable=False,
        tags=["reset_pad"],
        layer=1,
    ),
    "flash": Sprite(
        pixels=[[2]],
        name="flash",
        visible=False,
        collidable=False,
        tags=["flash"],
        layer=10,
    ),
}


def _border(w_tiles: int, h_tiles: int) -> List[Sprite]:
    walls = []
    for c in range(w_tiles):
        walls.append(sprites["wall"].clone().set_position(c * TILE, 0))
        walls.append(
            sprites["wall"].clone().set_position(c * TILE, (h_tiles - 1) * TILE)
        )
    for r in range(1, h_tiles - 1):
        walls.append(sprites["wall"].clone().set_position(0, r * TILE))
        walls.append(
            sprites["wall"].clone().set_position((w_tiles - 1) * TILE, r * TILE)
        )
    return walls


def _floor_grid(
    col_start: int,
    row_start: int,
    cols: int,
    rows: int,
    on_coords: Optional[set] = None,
) -> List[Sprite]:
    if on_coords is None:
        on_coords = set()
    tiles = []
    for r in range(rows):
        for c in range(cols):
            px = (col_start + c) * TILE
            py = (row_start + r) * TILE
            if (c, r) in on_coords:
                tiles.append(sprites["floor_on"].clone().set_position(px, py))
            else:
                tiles.append(sprites["floor_off"].clone().set_position(px, py))
    return tiles


def _clue_pattern(
    col_start: int, row_start: int, pattern: List[List[int]]
) -> List[Sprite]:
    clue_sprites = []
    for r, row in enumerate(pattern):
        for c, val in enumerate(row):
            px = (col_start + c) * TILE
            py = (row_start + r) * TILE
            if val == 1:
                clue_sprites.append(sprites["clue_on"].clone().set_position(px, py))
            else:
                clue_sprites.append(sprites["clue_off"].clone().set_position(px, py))
    return clue_sprites


def _floor_border(col_start: int, row_start: int, cols: int, rows: int) -> List[Sprite]:
    border_sprites = []
    left_px = col_start * TILE
    top_px = row_start * TILE
    right_px = left_px + cols * TILE
    bottom_px = top_px + rows * TILE
    total_w = cols * TILE
    total_h = rows * TILE
    border_color = 14

    top_line = Sprite(
        pixels=[[border_color] * total_w],
        name="floor_border_top",
        visible=True,
        collidable=False,
        tags=["floor_border"],
        layer=2,
    )
    top_line.set_position(left_px, top_px - 1)
    border_sprites.append(top_line)

    bottom_line = Sprite(
        pixels=[[border_color] * total_w],
        name="floor_border_bottom",
        visible=True,
        collidable=False,
        tags=["floor_border"],
        layer=2,
    )
    bottom_line.set_position(left_px, bottom_px)
    border_sprites.append(bottom_line)

    left_line = Sprite(
        pixels=[[border_color] for _ in range(total_h)],
        name="floor_border_left",
        visible=True,
        collidable=False,
        tags=["floor_border"],
        layer=2,
    )
    left_line.set_position(left_px - 1, top_px)
    border_sprites.append(left_line)

    right_line = Sprite(
        pixels=[[border_color] for _ in range(total_h)],
        name="floor_border_right",
        visible=True,
        collidable=False,
        tags=["floor_border"],
        layer=2,
    )
    right_line.set_position(right_px, top_px)
    border_sprites.append(right_line)

    return border_sprites


_l1_target = [
    [1, 0, 1],
    [1, 1, 0],
]

_l1_sprites = (
    _border(11, 9)
    + _clue_pattern(1, 1, _l1_target)
    + _floor_grid(5, 2, 5, 4)
    + _floor_border(5, 2, 5, 4)
    + [
        sprites["player"].clone().set_position(1 * TILE, 7 * TILE),
        sprites["wall"].clone().set_position(4 * TILE, 1 * TILE),
        sprites["wall"].clone().set_position(4 * TILE, 2 * TILE),
        sprites["wall"].clone().set_position(4 * TILE, 3 * TILE),
        sprites["wall"].clone().set_position(4 * TILE, 4 * TILE),
        sprites["wall"].clone().set_position(4 * TILE, 5 * TILE),
        sprites["wall"].clone().set_position(4 * TILE, 6 * TILE),
        sprites["gate_locked"].clone().set_position(8 * TILE, 7 * TILE),
        sprites["reset_pad"].clone().set_position(3 * TILE, 7 * TILE),
    ]
)

level1 = Level(
    sprites=_l1_sprites,
    grid_size=(11 * TILE, 9 * TILE),
    data={
        "type": "pattern",
        "target_pattern": _l1_target,
        "floor_origin": [5, 2],
        "floor_size": [5, 4],
        "pattern_offset": [1, 1],
        "gate_pos": [8, 7],
        "max_moves": 50,
        "spawn_positions": [[1, 7], [2, 7], [6, 7], [9, 7]],
    },
)


_l2_target = [
    [1, 0, 1],
    [0, 1, 0],
    [1, 0, 1],
]

_l2_sprites = (
    _border(12, 10)
    + _clue_pattern(2, 1, _l2_target)
    + _floor_grid(6, 1, 5, 5)
    + _floor_border(6, 1, 5, 5)
    + [
        sprites["player"].clone().set_position(2 * TILE, 7 * TILE),
        sprites["gate_locked"].clone().set_position(8 * TILE, 7 * TILE),
        sprites["reset_pad"].clone().set_position(5 * TILE, 8 * TILE),
    ]
)

level2 = Level(
    sprites=_l2_sprites,
    grid_size=(12 * TILE, 10 * TILE),
    data={
        "type": "pattern",
        "target_pattern": _l2_target,
        "floor_origin": [6, 1],
        "floor_size": [5, 5],
        "pattern_offset": [1, 1],
        "gate_pos": [8, 7],
        "max_moves": 100,
        "spawn_positions": [[2, 7], [4, 7], [7, 7], [10, 7]],
    },
)


_l3_clue = [
    [1, 0, 1, 0, 1],
    [0, 1, 0, 1, 0],
    [1, 1, 0, 1, 1],
]

_l3_sprites = (
    _border(12, 11)
    + _clue_pattern(1, 1, _l3_clue)
    + _floor_grid(2, 4, 7, 5)
    + _floor_border(2, 4, 7, 5)
    + [
        sprites["player"].clone().set_position(2 * TILE, 9 * TILE),
        sprites["gate_locked"].clone().set_position(9 * TILE, 9 * TILE),
        sprites["reset_pad"].clone().set_position(4 * TILE, 9 * TILE),
        sprites["wall"].clone().set_position(9 * TILE, 8 * TILE),
        sprites["wall"].clone().set_position(10 * TILE, 8 * TILE),
    ]
)

level3 = Level(
    sprites=_l3_sprites,
    grid_size=(12 * TILE, 11 * TILE),
    data={
        "type": "pattern",
        "target_pattern": _l3_clue,
        "floor_origin": [2, 4],
        "floor_size": [7, 5],
        "pattern_offset": [1, 1],
        "gate_pos": [9, 9],
        "max_moves": 120,
        "spawn_positions": [[1, 9], [2, 9], [5, 9], [8, 9]],
    },
)


_l4_target = [
    [0, 1, 1, 0],
    [1, 0, 0, 1],
    [1, 0, 0, 1],
    [0, 1, 1, 0],
]

_l4_sprites = (
    _border(12, 12)
    + _clue_pattern(1, 1, _l4_target)
    + _floor_grid(5, 4, 6, 6)
    + _floor_border(5, 4, 6, 6)
    + [
        sprites["player"].clone().set_position(2 * TILE, 10 * TILE),
        sprites["gate_locked"].clone().set_position(9 * TILE, 10 * TILE),
        sprites["reset_pad"].clone().set_position(4 * TILE, 10 * TILE),
    ]
)

level4 = Level(
    sprites=_l4_sprites,
    grid_size=(12 * TILE, 12 * TILE),
    data={
        "type": "toggle_neighbors",
        "target_pattern": _l4_target,
        "floor_origin": [5, 4],
        "floor_size": [6, 6],
        "pattern_offset": [1, 1],
        "gate_pos": [9, 10],
        "max_moves": 150,
        "spawn_positions": [[1, 10], [2, 10], [6, 10], [10, 10]],
    },
)


_l5_target = [
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [1, 1, 0, 1],
    [0, 1, 0, 0],
]

_l5_sprites = (
    _border(12, 12)
    + _clue_pattern(1, 1, _l5_target)
    + _floor_grid(5, 4, 6, 6)
    + _floor_border(5, 4, 6, 6)
    + [
        sprites["player"].clone().set_position(2 * TILE, 10 * TILE),
        sprites["gate_locked"].clone().set_position(10 * TILE, 10 * TILE),
        sprites["reset_pad"].clone().set_position(4 * TILE, 10 * TILE),
        sprites["wall"].clone().set_position(7 * TILE, 6 * TILE),
        sprites["wall"].clone().set_position(8 * TILE, 8 * TILE),
    ]
)

level5 = Level(
    sprites=_l5_sprites,
    grid_size=(12 * TILE, 12 * TILE),
    data={
        "type": "pattern",
        "target_pattern": _l5_target,
        "floor_origin": [5, 4],
        "floor_size": [6, 6],
        "pattern_offset": [1, 1],
        "gate_pos": [10, 10],
        "max_moves": 130,
        "spawn_positions": [[1, 10], [2, 10], [6, 10], [9, 10]],
    },
)

levels = [level1, level2, level3, level4, level5]


class PatternHUD(RenderableUserDisplay):
    def __init__(self, game: "Pp77") -> None:
        self.game = game
        self.moves_remaining = 0
        self.max_moves = 0
        self.lives = MAX_LIVES
        self.gate_open = False
        self.message = ""

    def update(
        self,
        moves_remaining: int = 0,
        max_moves: int = 0,
        lives: int = MAX_LIVES,
        gate_open: bool = False,
    ) -> None:
        self.moves_remaining = moves_remaining
        self.max_moves = max_moves
        self.lives = lives
        self.gate_open = gate_open

    def render_interface(self, frame: Any) -> Any:
        height, width = frame.shape[:2]

        for i in range(MAX_LIVES):
            px = width - 2 - i * 4
            color = 2 if i < self.lives else 5
            if 0 <= px < width:
                frame[0, px] = color
            if 0 <= px - 1:
                frame[0, px - 1] = color
            if 0 <= px - 2:
                frame[0, px - 2] = color

        gate_color = 3 if self.gate_open else 2
        for i in range(3):
            if i < width:
                frame[0, i] = gate_color

        if self.max_moves > 0:
            bar_row = height - 1
            ratio = self.moves_remaining / self.max_moves
            filled = int(width * ratio)
            for i in range(width):
                if i < filled:
                    if ratio > 0.5:
                        frame[bar_row, i] = 3
                    elif ratio > 0.25:
                        frame[bar_row, i] = 4
                    else:
                        frame[bar_row, i] = 2
                else:
                    frame[bar_row, i] = 5

        return frame


class Pp77(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)
        self.hud = PatternHUD(self)
        self.player = None
        self.lives = MAX_LIVES
        self.moves_remaining = 0
        self.max_moves = 0

        self.floor_states = {}
        self.floor_sprites = {}

        self.gate_sprite = None
        self.gate_open = False
        self.gate_pos = (0, 0)

        self._start_x = 0
        self._start_y = 0
        self._removed_sprites = []

        self.flash_sprite = None
        self.flash_active = False

        self._history: List[Dict[str, Any]] = []
        self._consecutive_resets = 0
        self._game_over = False

        cam_w = min(max(lev.grid_size[0] for lev in levels), 64)
        cam_h = min(max(lev.grid_size[1] for lev in levels), 64)

        camera = Camera(0, 0, cam_w, cam_h, BACKGROUND_COLOR, PADDING_COLOR, [self.hud])
        super().__init__("pp77", levels, camera, available_actions=[0, 1, 2, 3, 4, 7])

    def on_set_level(self, level: Level) -> None:
        self._level_idx = 0
        for i, lv in enumerate(levels):
            if lv is self.current_level:
                self._level_idx = i
                break

        self.lives = MAX_LIVES

        self._history = []

        self.player = self.current_level.get_sprites_by_tag("player")[0]

        self._spawn_positions = self.current_level.get_data("spawn_positions") or []
        self._randomize_spawn()

        self.max_moves = self.current_level.get_data("max_moves") or 100
        self.moves_remaining = self.max_moves

        self.gate_open = False
        self._removed_sprites = []

        gates = self.current_level.get_sprites_by_tag("gate")
        if gates:
            self.gate_sprite = gates[0]
        else:
            self.gate_sprite = None

        gate_pos_data = self.current_level.get_data("gate_pos")
        if gate_pos_data:
            self.gate_pos = tuple(gate_pos_data)

        self.floor_states = {}
        self.floor_sprites = {}
        floor_tiles = self.current_level.get_sprites_by_tag("floor")
        for ft in floor_tiles:
            tc = ft.x // TILE
            tr = ft.y // TILE
            is_on = ft.tags is not None and "floor_on" in ft.tags
            self.floor_states[(tc, tr)] = is_on
            self.floor_sprites[(tc, tr)] = ft

        flash_list = self.current_level.get_sprites_by_tag("flash")
        if flash_list:
            self.flash_sprite = flash_list[0]
        else:
            self.flash_sprite = sprites["flash"].clone()
            self.current_level.add_sprite(self.flash_sprite)
        self.flash_sprite.set_visible(False)
        self.flash_active = False

        self.hud.update(
            self.moves_remaining, self.max_moves, self.lives, self.gate_open
        )

    def check_collision(self, x: int, y: int) -> bool:
        for s in self.current_level.get_sprites_by_tag("solid"):
            if s.x <= x < s.x + s.width and s.y <= y < s.y + s.height:
                return True
        return False

    def get_sprite_at(self, x: int, y: int, tag: str) -> Optional[Sprite]:
        for s in self.current_level.get_sprites_by_tag(tag):
            if s.x <= x < s.x + s.width and s.y <= y < s.y + s.height:
                return s
        return None

    def _tile_coord(self, px_x: int, px_y: int) -> Tuple[int, int]:
        return (px_x // TILE, px_y // TILE)

    def _toggle_floor(self, tc: int, tr: int) -> None:
        if (tc, tr) not in self.floor_sprites:
            return
        current = self.floor_states.get((tc, tr), False)
        new_state = not current
        self.floor_states[(tc, tr)] = new_state
        sprite = self.floor_sprites[(tc, tr)]
        if new_state:
            sprite.color_remap(None, 4)
        else:
            sprite.color_remap(None, 0)

    def _set_floor(self, tc: int, tr: int, state: bool) -> None:
        if (tc, tr) not in self.floor_sprites:
            return
        self.floor_states[(tc, tr)] = state
        sprite = self.floor_sprites[(tc, tr)]
        if state:
            sprite.color_remap(None, 4)
        else:
            sprite.color_remap(None, 0)

    def _reset_all_floors(self) -> None:
        for key in self.floor_states:
            self._set_floor(key[0], key[1], False)

    def _get_floor_pattern(self) -> Optional[List[List[int]]]:
        origin = self.current_level.get_data("floor_origin")
        size = self.current_level.get_data("floor_size")
        if not origin or not size:
            return None
        pattern = []
        for r in range(size[1]):
            row = []
            for c in range(size[0]):
                tc = origin[0] + c
                tr = origin[1] + r
                row.append(1 if self.floor_states.get((tc, tr), False) else 0)
            pattern.append(row)
        return pattern

    def _check_pattern_match(self) -> bool:
        target = self.current_level.get_data("target_pattern")
        if not target:
            return False
        current = self._get_floor_pattern()
        if not current:
            return False
        target_rows = len(target)
        target_cols = len(target[0]) if target_rows > 0 else 0
        floor_rows = len(current)
        floor_cols = len(current[0]) if floor_rows > 0 else 0

        if target_rows > floor_rows or target_cols > floor_cols:
            return False

        for off_r in range(floor_rows - target_rows + 1):
            for off_c in range(floor_cols - target_cols + 1):
                match = True
                for r in range(target_rows):
                    for c in range(target_cols):
                        if current[off_r + r][off_c + c] != target[r][c]:
                            match = False
                            break
                    if not match:
                        break

                if match:
                    return True

        return False

    def _check_level_complete(self) -> bool:
        level_type = self.current_level.get_data("type")
        if level_type in ("pattern", "toggle_neighbors"):
            return self._check_pattern_match()
        return False

    def _open_gate(self) -> None:
        if self.gate_open:
            return
        self.gate_open = True
        if self.gate_sprite:
            gx = self.gate_sprite.x
            gy = self.gate_sprite.y
            self.current_level.remove_sprite(self.gate_sprite)
            self._removed_sprites.append(self.gate_sprite)
            new_gate = sprites["gate_open"].clone().set_position(gx, gy)
            self.current_level.add_sprite(new_gate)
            self.gate_sprite = new_gate
        self.hud.update(self.moves_remaining, self.max_moves, self.lives, True)

    def _close_gate(self) -> None:
        if not self.gate_open:
            return
        self.gate_open = False
        if self.gate_sprite:
            gx = self.gate_sprite.x
            gy = self.gate_sprite.y
            self.current_level.remove_sprite(self.gate_sprite)
            new_gate = sprites["gate_locked"].clone().set_position(gx, gy)
            self.current_level.add_sprite(new_gate)
            self.gate_sprite = new_gate
        self.hud.update(self.moves_remaining, self.max_moves, self.lives, False)

    def _handle_floor_step(self, tile_x: int, tile_y: int) -> None:
        level_type = self.current_level.get_data("type")

        if level_type == "toggle_neighbors":
            self._toggle_floor(tile_x, tile_y)
            self._toggle_floor(tile_x - 1, tile_y)
            self._toggle_floor(tile_x + 1, tile_y)
            self._toggle_floor(tile_x, tile_y - 1)
            self._toggle_floor(tile_x, tile_y + 1)
        elif level_type in ("pattern", "symmetry"):
            self._toggle_floor(tile_x, tile_y)

    def _randomize_spawn(self) -> None:
        if self._spawn_positions:
            pos = self._rng.choice(self._spawn_positions)
            self._start_x = pos[0] * TILE
            self._start_y = pos[1] * TILE
        else:
            self._start_x = self.player.x
            self._start_y = self.player.y
        self.player.set_position(self._start_x, self._start_y)

    def _reset_current_level(self) -> None:
        self._reset_all_floors()

        self._close_gate()

        self._randomize_spawn()

        self.moves_remaining = self.max_moves
        self._history = []
        self.hud.update(self.moves_remaining, self.max_moves, self.lives, False)

    def handle_reset(self) -> None:
        self._consecutive_resets += 1
        if self._consecutive_resets >= 2:
            self._consecutive_resets = 0
            self.full_reset()
        else:
            self.level_reset()

    def _save_state(self) -> None:
        self._history.append(
            {
                "player_x": self.player.x,
                "player_y": self.player.y,
                "floor_states": dict(self.floor_states),
                "gate_open": self.gate_open,
            }
        )

    def _undo(self) -> None:
        if not self._history:
            return
        snap = self._history.pop()
        self.player.set_position(snap["player_x"], snap["player_y"])

        for key, state in snap["floor_states"].items():
            self._set_floor(key[0], key[1], state)

        was_open = snap["gate_open"]
        if was_open and not self.gate_open:
            self._open_gate()
        elif not was_open and self.gate_open:
            self._close_gate()

    def _process_move(self, dx: int, dy: int) -> None:
        self._save_state()

        self.moves_remaining -= 1
        self.hud.update(
            self.moves_remaining, self.max_moves, self.lives, self.gate_open
        )

        if self.moves_remaining <= 0:
            self.lives -= 1
            if self.lives <= 0:
                self._game_over = True
                self.lose()
            else:
                self._reset_current_level()
            return

        new_x = self.player.x + dx
        new_y = self.player.y + dy

        if self.check_collision(new_x, new_y):
            return

        self.player.set_position(new_x, new_y)

        tile_x, tile_y = self._tile_coord(new_x, new_y)

        if (tile_x, tile_y) in self.floor_sprites:
            self._handle_floor_step(tile_x, tile_y)

        reset_spr = self.get_sprite_at(new_x, new_y, "reset_pad")
        if reset_spr:
            self._reset_all_floors()
            self._close_gate()

        if self._check_level_complete():
            self._open_gate()
        else:
            if self.gate_open:
                self._close_gate()

        exit_sprite = self.get_sprite_at(new_x, new_y, "exit")
        if exit_sprite and self.gate_open:
            self.next_level()

    def step(self) -> None:
        if self._game_over:
            self.complete_action()
            return

        if self.flash_active:
            if self.flash_sprite:
                self.flash_sprite.set_visible(False)
            self.flash_active = False
            self.complete_action()
            return

        if self.action.id == GameAction.ACTION7:
            self._consecutive_resets = 0
            self._undo()
            self.moves_remaining -= 1
            self.hud.update(
                self.moves_remaining, self.max_moves, self.lives, self.gate_open
            )
            if self.moves_remaining <= 0:
                self.lives -= 1
                if self.lives <= 0:
                    self._game_over = True
                    self.lose()
                else:
                    self._reset_current_level()
            self.complete_action()
            return

        dx, dy = 0, 0
        if self.action.id == GameAction.ACTION1:
            dy = -TILE
        elif self.action.id == GameAction.ACTION2:
            dy = TILE
        elif self.action.id == GameAction.ACTION3:
            dx = -TILE
        elif self.action.id == GameAction.ACTION4:
            dx = TILE

        if dx == 0 and dy == 0:
            self.complete_action()
            return

        self._consecutive_resets = 0
        prev_lives = self.lives
        self._process_move(dx, dy)
        if self.lives < prev_lives and self.lives > 0:
            if self.flash_sprite:
                self.flash_sprite.set_visible(True)
                self.flash_sprite.set_scale(
                    max(
                        self.current_level.grid_size[0],
                        self.current_level.grid_size[1],
                    )
                )
                self.flash_sprite.set_position(0, 0)
            self.flash_active = True

        self.complete_action()


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


class PuzzleEnvironment:
    _ACTION_MAP: Dict[str, GameAction] = {
        "up": GameAction.ACTION1,
        "down": GameAction.ACTION2,
        "left": GameAction.ACTION3,
        "right": GameAction.ACTION4,
        "undo": GameAction.ACTION7,
        "reset": GameAction.RESET,
    }

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self._engine = Pp77(seed=seed)
        self._total_turns = 0
        self._last_action_was_reset = False
        self._game_won = False
        self._total_levels = len(levels)

    def _build_text_obs(self) -> str:
        e = self._engine
        gs = e.current_level.grid_size
        cell = TILE
        cols = gs[0] // cell
        rows = gs[1] // cell
        grid = [["." for _ in range(cols)] for _ in range(rows)]

        for s in e.current_level.get_sprites_by_tag("solid"):
            cx, cy = s.x // cell, s.y // cell
            if 0 <= cx < cols and 0 <= cy < rows:
                grid[cy][cx] = "#"
        for s in e.current_level.get_sprites_by_tag("floor"):
            cx, cy = s.x // cell, s.y // cell
            if 0 <= cx < cols and 0 <= cy < rows:
                grid[cy][cx] = "F" if e.floor_states.get((cx, cy), False) else "o"
        for s in e.current_level.get_sprites_by_tag("gate"):
            cx, cy = s.x // cell, s.y // cell
            if 0 <= cx < cols and 0 <= cy < rows:
                grid[cy][cx] = "G" if e.gate_open else "g"
        for s in e.current_level.get_sprites_by_tag("reset_pad"):
            cx, cy = s.x // cell, s.y // cell
            if 0 <= cx < cols and 0 <= cy < rows:
                grid[cy][cx] = "R"
        for s in e.current_level.get_sprites_by_tag("exit"):
            cx, cy = s.x // cell, s.y // cell
            if 0 <= cx < cols and 0 <= cy < rows:
                grid[cy][cx] = "E"
        for s in e.current_level.get_sprites_by_tag("player"):
            cx, cy = s.x // cell, s.y // cell
            if 0 <= cx < cols and 0 <= cy < rows:
                grid[cy][cx] = "P"

        target = e.current_level.get_data("target_pattern") or []
        target_str = ""
        if target:
            for row in target:
                target_str += "".join("F" if c else "." for c in row) + "\n"

        remaining = e.moves_remaining
        header = (
            f"Level {e._level_idx + 1}/{len(levels)} | "
            f"Moves: {remaining}/{e.max_moves} | "
            f"Lives: {e.lives} | "
            f"Gate: {'open' if e.gate_open else 'closed'}"
        )
        ascii_rows = ["".join(row) for row in grid]
        text_observation = header + "\n" + "\n".join(ascii_rows)
        if target_str:
            text_observation += "\nTarget:\n" + target_str.rstrip()
        return text_observation

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
        raw_data = b""
        for row_idx in range(h):
            raw_data += b"\x00" + rgb[row_idx].tobytes()

        png_buf = io.BytesIO()
        png_buf.write(b"\x89PNG\r\n\x1a\n")

        def _chunk(ctype: bytes, data: bytes) -> None:
            png_buf.write(struct.pack(">I", len(data)))
            png_buf.write(ctype)
            png_buf.write(data)
            png_buf.write(struct.pack(">I", zlib.crc32(ctype + data) & 0xFFFFFFFF))

        _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        _chunk(b"IDAT", zlib.compress(raw_data))
        _chunk(b"IEND", b"")
        return png_buf.getvalue()

    def _build_game_state(
        self, done: bool = False, info: Optional[Dict[str, Any]] = None
    ) -> GameState:
        e = self._engine

        valid_actions = self.get_actions() if not done else None

        return GameState(
            text_observation=self._build_text_obs(),
            image_observation=self._build_image_bytes(),
            valid_actions=valid_actions,
            turn=self._total_turns,
            metadata={
                "moves_remaining": e.moves_remaining,
                "max_moves": e.max_moves,
                "lives": e.lives,
                "gate_open": e.gate_open,
                "level_index": e._level_idx,
                "total_levels": len(levels),
                "game_over": e._game_over,
                "done": done,
                "info": info or {},
                "levels_completed": getattr(self._engine, "_score", 0),
            },
        )

    def _internal_reset(self) -> GameState:
        e = self._engine
        e._game_over = False
        if self._game_won or self._last_action_was_reset:
            e._consecutive_resets = 0
            e.full_reset()
            self._game_won = False
        else:
            e._consecutive_resets = 0
            e.level_reset()
        self._last_action_was_reset = True
        return self._build_game_state()

    def reset(self) -> GameState:
        self._total_turns = 0
        return self._internal_reset()

    def get_actions(self) -> List[str]:
        e = self._engine
        if e._game_over or self._game_won:
            return ["reset"]
        return ["up", "down", "left", "right", "undo", "reset"]

    def is_done(self) -> bool:
        e = self._engine
        return e._game_over or self._game_won or getattr(e, "_terminated", False)

    def step(self, action: str) -> StepResult:
        e = self._engine

        if action == "reset":
            state = self._internal_reset()
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
        info: Dict[str, Any] = {"action": action}

        level_before = e._level_idx
        action_input = ActionInput(id=game_action)
        frame = e.perform_action(action_input, raw=True)
        level_after = e._level_idx

        game_won = frame and frame.state and frame.state.name == "WIN"
        done = e._game_over or game_won

        if game_won:
            self._game_won = True
            info["reason"] = "game_complete"
            reward = 1.0 / self._total_levels
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

        if level_after > level_before:
            info["reason"] = "level_complete"
            reward = 1.0 / self._total_levels
            return StepResult(
                state=self._build_game_state(done=False, info=info),
                reward=reward,
                done=False,
                info=info,
            )

        return StepResult(
            state=self._build_game_state(done=False, info=info),
            reward=0.0,
            done=False,
            info=info,
        )

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError(f"Unsupported render mode: {mode}")
        e = self._engine
        index_grid = e.camera.render(e.current_level.get_sprites())
        h, w = index_grid.shape[0], index_grid.shape[1]
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
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

        self._seed: int = seed
        self._env: Optional[PuzzleEnvironment] = None

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
