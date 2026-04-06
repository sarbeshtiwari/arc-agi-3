from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import io
import random
import struct
import zlib

import gymnasium as gym
import numpy as np
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
    metadata: dict = field(default_factory=dict)


@dataclass
class StepResult:
    state: GameState
    reward: float
    done: bool
    info: dict = field(default_factory=dict)


BACKGROUND_COLOR = 0
PADDING_COLOR = 4
MAX_LIVES = 3
TILE = 5
TILE_SM = 3

RED = "red"
BLUE = "blue"
GREEN = "green"

COLOR_MAP = {RED: 8, BLUE: 9, GREEN: 14}


sprites = {
    "player": Sprite(
        pixels=[
            [3, 3, 3, 3, 3],
            [3, 11, 11, 11, 3],
            [3, 11, 0, 11, 3],
            [3, 11, 11, 11, 3],
            [3, 3, 3, 3, 3],
        ],
        name="player",
        visible=True,
        collidable=True,
        tags=["player"],
        layer=5,
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
    "cell_red": Sprite(
        pixels=[
            [4, 4, 4, 4, 4],
            [4, 8, 8, 8, 4],
            [4, 8, 0, 8, 4],
            [4, 8, 8, 8, 4],
            [4, 4, 4, 4, 4],
        ],
        name="cell_red",
        visible=True,
        collidable=False,
        tags=["cell", "cell_red"],
        layer=1,
    ),
    "cell_blue": Sprite(
        pixels=[
            [4, 4, 4, 4, 4],
            [4, 9, 9, 9, 4],
            [4, 9, 0, 9, 4],
            [4, 9, 9, 9, 4],
            [4, 4, 4, 4, 4],
        ],
        name="cell_blue",
        visible=True,
        collidable=False,
        tags=["cell", "cell_blue"],
        layer=1,
    ),
    "cell_green": Sprite(
        pixels=[
            [4, 4, 4, 4, 4],
            [4, 14, 14, 14, 4],
            [4, 14, 0, 14, 4],
            [4, 14, 14, 14, 4],
            [4, 4, 4, 4, 4],
        ],
        name="cell_green",
        visible=True,
        collidable=False,
        tags=["cell", "cell_green"],
        layer=1,
    ),
    "poison": Sprite(
        pixels=[
            [13, 5, 5, 5, 13],
            [5, 13, 5, 13, 5],
            [5, 5, 13, 5, 5],
            [5, 13, 5, 13, 5],
            [13, 5, 5, 5, 13],
        ],
        name="poison",
        visible=True,
        collidable=False,
        tags=["poison"],
        layer=1,
    ),
    "shield": Sprite(
        pixels=[
            [0, 0, 12, 0, 0],
            [0, 12, 12, 12, 0],
            [12, 12, 0, 12, 12],
            [0, 12, 12, 12, 0],
            [0, 0, 12, 0, 0],
        ],
        name="shield",
        visible=True,
        collidable=False,
        tags=["shield"],
        layer=2,
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
            [3, 14, 3, 14, 3],
            [3, 3, 14, 3, 3],
            [3, 14, 3, 14, 3],
            [3, 3, 3, 3, 3],
        ],
        name="gate_open",
        visible=True,
        collidable=False,
        tags=["gate_open", "exit"],
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


sprites_sm = {
    "player": Sprite(
        pixels=[
            [3, 11, 3],
            [11, 0, 11],
            [3, 11, 3],
        ],
        name="player",
        visible=True,
        collidable=True,
        tags=["player"],
        layer=5,
    ),
    "wall": Sprite(
        pixels=[
            [5, 5, 5],
            [5, 5, 5],
            [5, 5, 5],
        ],
        name="wall",
        visible=True,
        collidable=True,
        tags=["wall", "solid"],
        layer=0,
    ),
    "cell_red": Sprite(
        pixels=[
            [4, 8, 4],
            [8, 0, 8],
            [4, 8, 4],
        ],
        name="cell_red",
        visible=True,
        collidable=False,
        tags=["cell", "cell_red"],
        layer=1,
    ),
    "cell_blue": Sprite(
        pixels=[
            [4, 9, 4],
            [9, 0, 9],
            [4, 9, 4],
        ],
        name="cell_blue",
        visible=True,
        collidable=False,
        tags=["cell", "cell_blue"],
        layer=1,
    ),
    "cell_green": Sprite(
        pixels=[
            [4, 14, 4],
            [14, 0, 14],
            [4, 14, 4],
        ],
        name="cell_green",
        visible=True,
        collidable=False,
        tags=["cell", "cell_green"],
        layer=1,
    ),
    "poison": Sprite(
        pixels=[
            [13, 5, 13],
            [5, 13, 5],
            [13, 5, 13],
        ],
        name="poison",
        visible=True,
        collidable=False,
        tags=["poison"],
        layer=1,
    ),
    "shield": Sprite(
        pixels=[
            [0, 12, 0],
            [12, 0, 12],
            [0, 12, 0],
        ],
        name="shield",
        visible=True,
        collidable=False,
        tags=["shield"],
        layer=2,
    ),
    "gate_locked": Sprite(
        pixels=[
            [2, 0, 2],
            [0, 2, 0],
            [2, 0, 2],
        ],
        name="gate_locked",
        visible=True,
        collidable=True,
        tags=["gate", "solid"],
        layer=1,
    ),
    "gate_open": Sprite(
        pixels=[
            [3, 14, 3],
            [14, 3, 14],
            [3, 14, 3],
        ],
        name="gate_open",
        visible=True,
        collidable=False,
        tags=["gate_open", "exit"],
        layer=1,
    ),
    "reset_pad": Sprite(
        pixels=[
            [6, -1, 6],
            [-1, 6, -1],
            [6, -1, 6],
        ],
        name="reset_pad",
        visible=True,
        collidable=False,
        tags=["reset_pad"],
        layer=1,
    ),
}


def _border(w_tiles, h_tiles):
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


def _border_sm(w_tiles, h_tiles):
    T = TILE_SM
    walls = []
    for c in range(w_tiles):
        walls.append(sprites_sm["wall"].clone().set_position(c * T, 0))
        walls.append(sprites_sm["wall"].clone().set_position(c * T, (h_tiles - 1) * T))
    for r in range(1, h_tiles - 1):
        walls.append(sprites_sm["wall"].clone().set_position(0, r * T))
        walls.append(sprites_sm["wall"].clone().set_position((w_tiles - 1) * T, r * T))
    return walls


_l1_sprites = _border(9, 7) + [
    sprites["player"].clone().set_position(1 * TILE, 5 * TILE),
    sprites["gate_locked"].clone().set_position(1 * TILE, 1 * TILE),
    sprites["reset_pad"].clone().set_position(4 * TILE, 5 * TILE),
]

level1 = Level(
    sprites=_l1_sprites,
    grid_size=(9 * TILE, 7 * TILE),
    data={
        "cell_configs": [
            [3, 1, RED],
            [5, 1, RED],
            [7, 1, RED],
        ],
        "extra_walls": [
            [4, 3],
        ],
        "poisons": [
            [3, 3],
            [5, 3],
        ],
        "shields": [],
        "gate_pos": [1, 1],
        "max_moves": 48,
        "poison_grow_rate": 8,
        "three_color": False,
        "tile_size": TILE,
    },
)

_l2_sprites = _border(11, 11) + [
    sprites["player"].clone().set_position(1 * TILE, 9 * TILE),
    sprites["gate_locked"].clone().set_position(9 * TILE, 1 * TILE),
    sprites["reset_pad"].clone().set_position(5 * TILE, 9 * TILE),
]

level2 = Level(
    sprites=_l2_sprites,
    grid_size=(11 * TILE, 11 * TILE),
    data={
        "cell_configs": [
            [2, 1, RED],
            [5, 1, BLUE],
            [8, 1, RED],
            [2, 3, BLUE],
            [8, 3, BLUE],
            [5, 5, RED],
            [2, 7, RED],
            [5, 7, BLUE],
            [8, 7, RED],
        ],
        "extra_walls": [],
        "poisons": [
            [1, 3],
            [9, 3],
            [1, 8],
            [9, 8],
        ],
        "shields": [
            [5, 3],
            [2, 5],
            [8, 5],
        ],
        "gate_pos": [9, 1],
        "max_moves": 70,
        "poison_grow_rate": 5,
        "three_color": False,
        "tile_size": TILE,
    },
)

_T4 = TILE_SM

_l3_sprites = _border_sm(15, 13) + [
    sprites_sm["player"].clone().set_position(1 * _T4, 1 * _T4),
    sprites_sm["gate_locked"].clone().set_position(8 * _T4, 1 * _T4),
    sprites_sm["reset_pad"].clone().set_position(7 * _T4, 6 * _T4),
]

level3 = Level(
    sprites=_l3_sprites,
    grid_size=(15 * _T4, 13 * _T4),
    data={
        "cell_configs": [
            [3, 2, RED],
            [7, 2, RED],
            [11, 2, GREEN],
            [3, 5, GREEN],
            [7, 5, RED],
            [11, 5, BLUE],
            [3, 8, BLUE],
            [7, 8, GREEN],
            [11, 8, RED],
            [3, 11, RED],
            [7, 11, BLUE],
            [11, 11, GREEN],
        ],
        "extra_walls": [],
        "poisons": [
            [1, 3],
            [1, 4],
            [13, 3],
            [13, 4],
            [1, 9],
            [1, 10],
            [13, 9],
            [13, 10],
        ],
        "shields": [
            [7, 4],
            [7, 7],
            [11, 7],
            [7, 10],
        ],
        "gate_pos": [8, 1],
        "max_moves": 70,
        "poison_grow_rate": 3,
        "three_color": True,
        "tile_size": TILE_SM,
    },
)

_l4_sprites = _border_sm(21, 17) + [
    sprites_sm["player"].clone().set_position(1 * _T4, 1 * _T4),
    sprites_sm["gate_locked"].clone().set_position(1 * _T4, 13 * _T4),
    sprites_sm["reset_pad"].clone().set_position(10 * _T4, 8 * _T4),
]

level4 = Level(
    sprites=_l4_sprites,
    grid_size=(21 * _T4, 17 * _T4),
    data={
        "cell_configs": [
            [2, 2, RED],
            [5, 2, BLUE],
            [8, 2, GREEN],
            [11, 2, RED],
            [14, 2, BLUE],
            [2, 6, GREEN],
            [5, 6, RED],
            [8, 6, BLUE],
            [11, 6, GREEN],
            [14, 6, RED],
            [2, 10, BLUE],
            [5, 10, GREEN],
            [8, 10, RED],
            [11, 10, BLUE],
            [14, 10, GREEN],
            [2, 14, RED],
            [5, 14, BLUE],
            [8, 14, GREEN],
            [11, 14, RED],
            [14, 14, BLUE],
        ],
        "extra_walls": [
            [4, 4],
            [7, 4],
            [10, 4],
            [13, 4],
            [4, 12],
            [7, 12],
            [10, 12],
            [13, 12],
        ],
        "poisons": [
            [1, 2],
            [1, 3],
            [19, 2],
            [19, 3],
            [1, 6],
            [1, 7],
            [19, 6],
            [19, 7],
            [1, 10],
            [1, 11],
            [19, 10],
            [19, 11],
            [1, 14],
            [1, 15],
            [9, 1],
            [11, 1],
            [3, 13],
        ],
        "shields": [
            [5, 4],
            [8, 4],
            [11, 4],
            [2, 8],
            [14, 8],
            [5, 12],
            [8, 12],
            [11, 12],
        ],
        "gate_pos": [1, 13],
        "max_moves": 140,
        "poison_grow_rate": 3,
        "three_color": True,
        "tile_size": TILE_SM,
    },
)


levels = [level1, level2, level3, level4]

_START_CANDIDATES: list[list[tuple[int, int]]] = [
    [(1, 5), (7, 5), (2, 3), (7, 3)],
    [(1, 9), (9, 9), (1, 1), (3, 9)],
    [(1, 1), (13, 1), (1, 11), (13, 11)],
    [(1, 1), (19, 15), (19, 1), (10, 1)],
]


class CascadeHUD(RenderableUserDisplay):
    def __init__(self, game):
        self.game = game
        self.moves_remaining = 0
        self.max_moves = 0
        self.lives = MAX_LIVES
        self.gate_open = False
        self.cells_left = 0

    def update(
        self,
        moves_remaining=0,
        max_moves=0,
        lives=MAX_LIVES,
        gate_open=False,
        cells_left=0,
    ):
        self.moves_remaining = moves_remaining
        self.max_moves = max_moves
        self.lives = lives
        self.gate_open = gate_open
        self.cells_left = cells_left

    def render_interface(self, frame):
        height, width = frame.shape[:2]

        for i in range(MAX_LIVES):
            px = width - 2 - i * 4
            color = 8 if i < self.lives else 5
            if 0 <= px < width:
                frame[0, px] = color
            if 0 <= px - 1 < width:
                frame[0, px - 1] = color
            if 0 <= px - 2 < width:
                frame[0, px - 2] = color

        if self.max_moves > 0:
            bar_row = height - 1
            ratio = self.moves_remaining / self.max_moves
            filled = int(width * ratio)
            for i in range(width):
                if i < filled:
                    if ratio > 0.4:
                        frame[bar_row, i] = 3
                    elif ratio > 0.05:
                        frame[bar_row, i] = 4
                    else:
                        frame[bar_row, i] = 2
                else:
                    frame[bar_row, i] = 5

        return frame


class Qx47(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self.hud = CascadeHUD(self)
        self._rng = random.Random(seed)
        self.player = None
        self.lives = MAX_LIVES
        self.moves_remaining = 0
        self.max_moves = 0
        self.tile_size = TILE

        self.cell_data = {}
        self.cell_sprites = {}
        self.initial_cells = {}

        self.shield_data = {}
        self.shield_sprites = {}
        self.initial_shields = []

        self.poison_positions = set()
        self.poison_sprites = {}
        self.initial_poisons = set()
        self.poison_grow_rate = 0
        self.action_count = 0

        self.gate_sprite = None
        self.gate_open = False

        self._start_x = 0
        self._start_y = 0
        self._removed_sprites = []

        self.flash_sprite = None
        self.flash_active = False
        self._game_over = False

        self._three_color = False

        self._sprites = sprites

        self._engine_snapshot = None
        self._engine_can_undo = False

        cam_w = min(max(lev.grid_size[0] for lev in levels), 64)
        cam_h = min(max(lev.grid_size[1] for lev in levels), 64)

        camera = Camera(0, 0, cam_w, cam_h, BACKGROUND_COLOR, PADDING_COLOR, [self.hud])
        super().__init__(
            "qx47",
            levels,
            camera=camera,
            available_actions=[0, 1, 2, 3, 4, 5, 7],
        )

    def _get_sprites(self):
        return sprites_sm if self.tile_size == TILE_SM else sprites

    def on_set_level(self, level: Level) -> None:
        self.tile_size = self.current_level.get_data("tile_size") or TILE
        self._sprites = self._get_sprites()

        self.player = self.current_level.get_sprites_by_tag("player")[0]

        T = self.tile_size
        candidates = _START_CANDIDATES[self.level_index]
        chosen = self._rng.choice(candidates)
        start_x = chosen[0] * T
        start_y = chosen[1] * T
        self.player.set_position(start_x, start_y)

        self._start_x = start_x
        self._start_y = start_y

        self.max_moves = self.current_level.get_data("max_moves") or 50
        self.moves_remaining = self.max_moves
        self.lives = MAX_LIVES
        self.gate_open = False
        self._removed_sprites = []
        self._engine_can_undo = False
        self._engine_snapshot = None
        self._three_color = self.current_level.get_data("three_color") or False

        gates = self.current_level.get_sprites_by_tag("gate")
        if gates:
            self.gate_sprite = gates[0]
        else:
            open_gates = self.current_level.get_sprites_by_tag("gate_open")
            if open_gates:
                gx, gy = open_gates[0].x, open_gates[0].y
                self.current_level.remove_sprite(open_gates[0])
                new_gate = self._sprites["gate_locked"].clone().set_position(gx, gy)
                self.current_level.add_sprite(new_gate)
                self.gate_sprite = new_gate
            else:
                self.gate_sprite = None

        for s in list(self.current_level.get_sprites_by_tag("cell")):
            self.current_level.remove_sprite(s)
        for s in list(self.current_level.get_sprites_by_tag("shield")):
            self.current_level.remove_sprite(s)
        for s in list(self.current_level.get_sprites_by_tag("poison")):
            self.current_level.remove_sprite(s)

        self.cell_data = {}
        self.cell_sprites = {}
        self.initial_cells = {}
        self.shield_data = {}
        self.shield_sprites = {}
        self.initial_shields = []
        self.poison_positions = set()
        self.poison_sprites = {}
        self.initial_poisons = set()
        self.action_count = 0

        T = self.tile_size

        cell_configs = self.current_level.get_data("cell_configs") or []
        for cfg in cell_configs:
            tc, tr, color = cfg
            self._place_cell(tc, tr, color)
            self.initial_cells[(tc, tr)] = color

        shield_configs = self.current_level.get_data("shields") or []
        for sc in shield_configs:
            tc, tr = sc
            self._place_shield(tc, tr)
            self.initial_shields.append((tc, tr))

        poison_configs = self.current_level.get_data("poisons") or []
        for pc in poison_configs:
            tc, tr = pc
            self._place_poison(tc, tr)
        self.initial_poisons = set(self.poison_positions)
        self.poison_grow_rate = self.current_level.get_data("poison_grow_rate") or 0

        extra_walls = self.current_level.get_data("extra_walls") or []
        for w in extra_walls:
            wc, wr = w
            wall_sprite = self._sprites["wall"].clone().set_position(wc * T, wr * T)
            self.current_level.add_sprite(wall_sprite)

        flash_list = self.current_level.get_sprites_by_tag("flash")
        if flash_list:
            self.flash_sprite = flash_list[0]
        else:
            self.flash_sprite = sprites["flash"].clone()
            self.current_level.add_sprite(self.flash_sprite)
        self.flash_sprite.set_visible(False)
        self.flash_active = True

        self._update_hud()

    def _place_cell(self, tc, tr, color):
        T = self.tile_size
        sp = self._sprites
        if color == RED:
            spr = sp["cell_red"].clone()
        elif color == BLUE:
            spr = sp["cell_blue"].clone()
        else:
            spr = sp["cell_green"].clone()
        spr.set_position(tc * T, tr * T)
        self.current_level.add_sprite(spr)
        self.cell_sprites[(tc, tr)] = spr
        self.cell_data[(tc, tr)] = {"color": color}

    def _remove_cell(self, tc, tr):
        if (tc, tr) in self.cell_sprites:
            self.current_level.remove_sprite(self.cell_sprites[(tc, tr)])
            del self.cell_sprites[(tc, tr)]
        if (tc, tr) in self.cell_data:
            del self.cell_data[(tc, tr)]

    def _update_cell_sprite(self, tc, tr):
        if (tc, tr) not in self.cell_data:
            return
        T = self.tile_size
        sp = self._sprites
        color = self.cell_data[(tc, tr)]["color"]
        old_spr = self.cell_sprites[(tc, tr)]
        px, py = old_spr.x, old_spr.y
        self.current_level.remove_sprite(old_spr)
        if color == RED:
            new_spr = sp["cell_red"].clone()
        elif color == BLUE:
            new_spr = sp["cell_blue"].clone()
        else:
            new_spr = sp["cell_green"].clone()
        new_spr.set_position(px, py)
        self.current_level.add_sprite(new_spr)
        self.cell_sprites[(tc, tr)] = new_spr

    def _place_shield(self, tc, tr):
        T = self.tile_size
        spr = self._sprites["shield"].clone().set_position(tc * T, tr * T)
        self.current_level.add_sprite(spr)
        self.shield_sprites[(tc, tr)] = spr
        self.shield_data[(tc, tr)] = True

    def _flip_color(self, current_color, trigger_color):
        if self._three_color:
            cycle = [RED, BLUE, GREEN]
            idx = cycle.index(current_color)
            return cycle[(idx + 1) % 3]
        else:
            return BLUE if current_color == RED else RED

    def _trigger_cell(self, tc, tr):
        if (tc, tr) not in self.cell_data:
            return

        T = self.tile_size
        trigger_color = self.cell_data[(tc, tr)]["color"]
        self._remove_cell(tc, tr)

        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        chain_targets = []

        for dx, dy in directions:
            cx, cy = tc + dx, tr + dy
            while True:
                grid_w = self.current_level.grid_size[0] // T
                grid_h = self.current_level.grid_size[1] // T
                if cx < 0 or cx >= grid_w or cy < 0 or cy >= grid_h:
                    break
                if self._is_wall_at(cx, cy):
                    break
                if (cx, cy) in self.shield_data:
                    break
                if (cx, cy) in self.cell_data:
                    hit_color = self.cell_data[(cx, cy)]["color"]
                    if hit_color == trigger_color:
                        chain_targets.append((cx, cy))
                    else:
                        self.cell_data[(cx, cy)]["color"] = self._flip_color(
                            hit_color, trigger_color
                        )
                        self._update_cell_sprite(cx, cy)
                    break
                cx += dx
                cy += dy

        for ctc, ctr in chain_targets:
            if (ctc, ctr) in self.cell_data:
                self._trigger_cell(ctc, ctr)

    def _update_hud(self):
        self.hud.update(
            self.moves_remaining,
            self.max_moves,
            self.lives,
            self.gate_open,
            len(self.cell_data),
        )

    def check_collision(self, x, y):
        for s in self.current_level._sprites:
            if s.x <= x < s.x + s.width and s.y <= y < s.y + s.height:
                if s.tags and "solid" in s.tags:
                    return True
        return False

    def get_sprite_at(self, x, y, tag=None):
        for s in self.current_level._sprites:
            if s.x <= x < s.x + s.width and s.y <= y < s.y + s.height:
                if tag is None or (s.tags and tag in s.tags):
                    return s
        return None

    def _tile_coord(self, px_x, px_y):
        T = self.tile_size
        return (px_x // T, px_y // T)

    def _is_wall_at(self, tc, tr):
        T = self.tile_size
        px, py = tc * T, tr * T
        return self.check_collision(px, py)

    def _is_poison_at(self, tc, tr):
        return (tc, tr) in self.poison_positions

    def _place_poison(self, tc, tr):
        if (tc, tr) in self.poison_positions:
            return
        T = self.tile_size
        spr = self._sprites["poison"].clone().set_position(tc * T, tr * T)
        self.current_level.add_sprite(spr)
        self.poison_positions.add((tc, tr))
        self.poison_sprites[(tc, tr)] = spr

    def _remove_all_poisons(self):
        for (tc, tr), spr in list(self.poison_sprites.items()):
            self.current_level.remove_sprite(spr)
        self.poison_sprites.clear()
        self.poison_positions.clear()

    def _reset_all_poisons(self):
        self._remove_all_poisons()
        for tc, tr in self.initial_poisons:
            self._place_poison(tc, tr)

    def _grow_poison(self):
        T = self.tile_size
        grid_w = self.current_level.grid_size[0] // T
        grid_h = self.current_level.grid_size[1] // T
        candidates = []
        for tc, tr in list(self.poison_positions):
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nc, nr = tc + dx, tr + dy
                if nc <= 0 or nc >= grid_w - 1 or nr <= 0 or nr >= grid_h - 1:
                    continue
                if (nc, nr) in self.poison_positions:
                    continue
                if self._is_wall_at(nc, nr):
                    continue
                if (nc, nr) in self.cell_data:
                    continue
                if (nc, nr) in self.shield_data:
                    continue
                gate_tc = self.gate_sprite.x // T if self.gate_sprite else -1
                gate_tr = self.gate_sprite.y // T if self.gate_sprite else -1
                if (nc, nr) == (gate_tc, gate_tr):
                    continue
                start_tc = self._start_x // T
                start_tr = self._start_y // T
                if (nc, nr) == (start_tc, start_tr):
                    continue
                reset_pads = self.current_level.get_sprites_by_tag("reset_pad")
                is_reset = False
                for rp in reset_pads:
                    if nc == rp.x // T and nr == rp.y // T:
                        is_reset = True
                        break
                if is_reset:
                    continue
                candidates.append((nc, nr))
        if candidates:
            mid_c = (grid_w - 1) / 2.0
            mid_r = (grid_h - 1) / 2.0
            candidates.sort(
                key=lambda p: (-(abs(p[0] - mid_c) + abs(p[1] - mid_r)), p[1], p[0])
            )
            self._place_poison(candidates[0][0], candidates[0][1])

    def _open_gate(self):
        if self.gate_open:
            return
        self.gate_open = True
        if self.gate_sprite:
            gx, gy = self.gate_sprite.x, self.gate_sprite.y
            self.current_level.remove_sprite(self.gate_sprite)
            self._removed_sprites.append(self.gate_sprite)
            new_gate = self._sprites["gate_open"].clone().set_position(gx, gy)
            self.current_level.add_sprite(new_gate)
            self.gate_sprite = new_gate
        self._update_hud()

    def _close_gate(self):
        if not self.gate_open:
            return
        self.gate_open = False
        if self.gate_sprite:
            gx, gy = self.gate_sprite.x, self.gate_sprite.y
            self.current_level.remove_sprite(self.gate_sprite)
            self._removed_sprites.append(self.gate_sprite)
            new_gate = self._sprites["gate_locked"].clone().set_position(gx, gy)
            self.current_level.add_sprite(new_gate)
            self.gate_sprite = new_gate
        self._update_hud()

    def _reset_all_cells(self):
        for (tc, tr), spr in list(self.cell_sprites.items()):
            self.current_level.remove_sprite(spr)
        self.cell_data.clear()
        self.cell_sprites.clear()
        for (tc, tr), color in self.initial_cells.items():
            self._place_cell(tc, tr, color)

    def _reset_all_shields(self):
        for (tc, tr), spr in list(self.shield_sprites.items()):
            self.current_level.remove_sprite(spr)
        self.shield_data.clear()
        self.shield_sprites.clear()
        for tc, tr in self.initial_shields:
            self._place_shield(tc, tr)

    def _reset_current_level(self):
        self._reset_all_cells()
        self._reset_all_shields()
        self._reset_all_poisons()
        self._close_gate()
        self.player.set_position(self._start_x, self._start_y)
        self.moves_remaining = self.max_moves
        self.action_count = 0
        self._update_hud()

    def _handle_death(self):
        self.lives -= 1
        self._engine_can_undo = False
        self._engine_snapshot = None
        if self.lives <= 0:
            self._game_over = True
            self.lose()
        else:
            self.flash_active = True
            self._reset_current_level()

    def _engine_save_snapshot(self) -> None:
        self._engine_snapshot = {
            "player_x": self.player.x,
            "player_y": self.player.y,
            "lives": self.lives,
            "moves_remaining": self.moves_remaining,
            "action_count": self.action_count,
            "gate_open": self.gate_open,
            "cell_data": {k: dict(v) for k, v in self.cell_data.items()},
            "shield_data": dict(self.shield_data),
            "poison_positions": set(self.poison_positions),
        }

    def _rebuild_sprites_from_data(self) -> None:
        for spr in self.cell_sprites.values():
            self.current_level.remove_sprite(spr)
        self.cell_sprites.clear()
        for (tc, tr), info in self.cell_data.items():
            self._place_cell(tc, tr, info["color"])

        for spr in self.shield_sprites.values():
            self.current_level.remove_sprite(spr)
        self.shield_sprites.clear()
        for tc, tr in list(self.shield_data.keys()):
            T = self.tile_size
            spr = self._sprites["shield"].clone().set_position(tc * T, tr * T)
            self.current_level.add_sprite(spr)
            self.shield_sprites[(tc, tr)] = spr

        for spr in self.poison_sprites.values():
            self.current_level.remove_sprite(spr)
        self.poison_sprites.clear()
        for tc, tr in self.poison_positions:
            T = self.tile_size
            spr = self._sprites["poison"].clone().set_position(tc * T, tr * T)
            self.current_level.add_sprite(spr)
            self.poison_sprites[(tc, tr)] = spr

        if self.gate_sprite:
            gx, gy = self.gate_sprite.x, self.gate_sprite.y
            self.current_level.remove_sprite(self.gate_sprite)
            if self.gate_open:
                new_gate = self._sprites["gate_open"].clone().set_position(gx, gy)
            else:
                new_gate = self._sprites["gate_locked"].clone().set_position(gx, gy)
            self.current_level.add_sprite(new_gate)
            self.gate_sprite = new_gate

    def _engine_restore_snapshot(self) -> None:
        snap = self._engine_snapshot
        self.player.set_position(snap["player_x"], snap["player_y"])
        self.lives = snap["lives"]
        self.moves_remaining = snap["moves_remaining"]
        self.action_count = snap["action_count"]
        self.gate_open = snap["gate_open"]
        self.cell_data = {k: dict(v) for k, v in snap["cell_data"].items()}
        self.shield_data = dict(snap["shield_data"])
        self.poison_positions = set(snap["poison_positions"])
        self._rebuild_sprites_from_data()
        self._update_hud()

    def step(self) -> None:
        if self.flash_active:
            if self.flash_sprite:
                self.flash_sprite.set_visible(False)
            self.flash_active = False
            self.complete_action()
            return

        aid = self.action.id

        if aid == GameAction.ACTION7:
            if not self._engine_can_undo or self._engine_snapshot is None:
                self.complete_action()
                return
            new_moves = self.moves_remaining - 1
            self._engine_restore_snapshot()
            self.moves_remaining = new_moves
            self._engine_can_undo = False
            self._engine_snapshot = None
            self._update_hud()
            if self.moves_remaining <= 0:
                self._handle_death()
                if self.flash_active:
                    self.complete_action()
                    return
                self.complete_action()
                return
            if self.poison_grow_rate > 0:
                self.action_count += 1
                if self.action_count >= self.poison_grow_rate:
                    self.action_count = 0
                    self._grow_poison()
                    ptx, pty = self._tile_coord(self.player.x, self.player.y)
                    if self._is_poison_at(ptx, pty):
                        self._handle_death()
                        if self.flash_active:
                            self.complete_action()
                            return
            self.complete_action()
            return

        T = self.tile_size
        action = aid.value

        self._engine_save_snapshot()

        if action in (1, 2, 3, 4):
            dx, dy = 0, 0
            if action == 1:
                dy = -T
            elif action == 2:
                dy = T
            elif action == 3:
                dx = -T
            elif action == 4:
                dx = T

            self.moves_remaining -= 1
            self._update_hud()

            if self.moves_remaining <= 0:
                self._handle_death()
                if self.flash_active:
                    self.complete_action()
                    return
                self.complete_action()
                return

            new_x = self.player.x + dx
            new_y = self.player.y + dy

            if not self.check_collision(new_x, new_y):
                self.player.set_position(new_x, new_y)

                tile_x, tile_y = self._tile_coord(new_x, new_y)
                if self._is_poison_at(tile_x, tile_y):
                    self._handle_death()
                    if self.flash_active:
                        self.complete_action()
                        return
                    self.complete_action()
                    return

                reset = self.get_sprite_at(new_x, new_y, "reset_pad")
                if reset:
                    self._reset_all_cells()
                    self._reset_all_shields()
                    self._reset_all_poisons()
                    self._close_gate()
                    self.action_count = 0

                exit_sprite = self.get_sprite_at(new_x, new_y, "exit")
                if exit_sprite and self.gate_open:
                    self._engine_can_undo = False
                    self._engine_snapshot = None
                    self.next_level()
                    self.complete_action()
                    return

        elif action == 5:
            tile_x, tile_y = self._tile_coord(self.player.x, self.player.y)

            if (tile_x, tile_y) in self.cell_data:
                self.moves_remaining -= 1
                self._update_hud()

                if self.moves_remaining <= 0:
                    self._handle_death()
                    if self.flash_active:
                        self.complete_action()
                        return
                    self.complete_action()
                    return

                self._trigger_cell(tile_x, tile_y)

                if len(self.cell_data) == 0:
                    self._open_gate()
                else:
                    if self.gate_open:
                        self._close_gate()

                self._update_hud()
            else:
                self.moves_remaining -= 1
                self._update_hud()

                if self.moves_remaining <= 0:
                    self._handle_death()
                    if self.flash_active:
                        self.complete_action()
                        return

                self.complete_action()
                return

        if self.poison_grow_rate > 0:
            self.action_count += 1
            if self.action_count >= self.poison_grow_rate:
                self.action_count = 0
                self._grow_poison()
                ptx, pty = self._tile_coord(self.player.x, self.player.y)
                if self._is_poison_at(ptx, pty):
                    self._handle_death()
                    if self.flash_active:
                        self.complete_action()
                        return

        self._engine_can_undo = True
        self.complete_action()


class PuzzleEnvironment:
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

    _ACTION_MAP: Dict[str, GameAction] = {
        "reset": GameAction.RESET,
        "up": GameAction.ACTION1,
        "down": GameAction.ACTION2,
        "left": GameAction.ACTION3,
        "right": GameAction.ACTION4,
        "select": GameAction.ACTION5,
        "undo": GameAction.ACTION7,
    }

    _VALID_ACTIONS: List[str] = list(_ACTION_MAP.keys())

    def __init__(self, seed: int = 0) -> None:
        self._engine = Qx47(seed=seed)
        self._turn = 0
        self._done = False
        self._last_action_was_reset = False

    def _build_text_observation(self) -> str:
        g = self._engine
        T = g.tile_size
        grid_w = g.current_level.grid_size[0] // T
        grid_h = g.current_level.grid_size[1] // T
        lines = [
            f"Level {self._engine.level_index + 1}  "
            f"Moves: {g.moves_remaining}/{g.max_moves}  "
            f"Lives: {g.lives}/{MAX_LIVES}  "
            f"Gate: {'OPEN' if g.gate_open else 'LOCKED'}"
        ]

        grid = [["." for _ in range(grid_w)] for _ in range(grid_h)]

        for s in g.current_level._sprites:
            if s.tags and "wall" in s.tags:
                tc, tr = s.x // T, s.y // T
                if 0 <= tc < grid_w and 0 <= tr < grid_h:
                    grid[tr][tc] = "W"

        for (tc, tr), data in g.cell_data.items():
            if 0 <= tc < grid_w and 0 <= tr < grid_h and grid[tr][tc] == ".":
                c = data["color"]
                if c == RED:
                    grid[tr][tc] = "R"
                elif c == BLUE:
                    grid[tr][tc] = "B"
                else:
                    grid[tr][tc] = "G"

        for tc, tr in g.shield_data:
            if 0 <= tc < grid_w and 0 <= tr < grid_h and grid[tr][tc] == ".":
                grid[tr][tc] = "S"

        for tc, tr in g.poison_positions:
            if 0 <= tc < grid_w and 0 <= tr < grid_h and grid[tr][tc] == ".":
                grid[tr][tc] = "X"

        reset_pads = g.current_level.get_sprites_by_tag("reset_pad")
        for rp in reset_pads:
            rtc, rtr = rp.x // T, rp.y // T
            if 0 <= rtc < grid_w and 0 <= rtr < grid_h and grid[rtr][rtc] == ".":
                grid[rtr][rtc] = "~"

        if g.gate_sprite:
            gtc, gtr = g.gate_sprite.x // T, g.gate_sprite.y // T
            if 0 <= gtc < grid_w and 0 <= gtr < grid_h:
                grid[gtr][gtc] = "O" if g.gate_open else "D"

        if g.player:
            ptc, ptr = g.player.x // T, g.player.y // T
            if 0 <= ptc < grid_w and 0 <= ptr < grid_h:
                grid[ptr][ptc] = "P"

        for row in grid:
            lines.append(" ".join(row))

        return "\n".join(lines)

    def _render_frame(self) -> np.ndarray:
        e = self._engine
        return e.camera.render(e.current_level.get_sprites())

    def _build_image_observation(self) -> Optional[bytes]:
        try:
            frame = self._render_frame()
            if frame is not None:
                return self._frame_to_png(frame)
        except (ValueError, TypeError, OSError):
            pass
        return None

    def _frame_to_png(self, frame: np.ndarray) -> bytes:
        h, w = frame.shape[:2]
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for idx, color in enumerate(self.ARC_PALETTE):
            mask = frame == idx
            rgb[mask] = color
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

    def _build_state(self, done: bool = False) -> GameState:
        e = self._engine
        return GameState(
            text_observation=self._build_text_observation(),
            image_observation=self._build_image_observation(),
            valid_actions=self.get_actions(),
            turn=self._turn,
            metadata={
                "total_levels": len(e._levels),
                "level_index": e.level_index,
                "levels_completed": getattr(e, "_score", 0),
                "game_over": getattr(getattr(e, "_state", None), "name", "")
                == "GAME_OVER",
                "done": done,
                "info": {},
                "level": e.level_index + 1,
                "moves_remaining": e.moves_remaining,
                "max_moves": e.max_moves,
                "lives": e.lives,
                "gate_open": e.gate_open,
                "cells_left": len(e.cell_data),
            },
        )

    def reset(self) -> GameState:
        e = self._engine
        _reset = ActionInput(id=GameAction.RESET)

        if e._state.name == "WIN" or self._last_action_was_reset:
            e._action_count = 0
            e.perform_action(_reset)
        else:
            e.level_reset()

        self._done = False
        self._last_action_was_reset = True
        self._turn = 0
        return self._build_state()

    def get_actions(self) -> List[str]:
        if self._done:
            return ["reset"]
        return list(self._VALID_ACTIONS)

    def step(self, action: str) -> StepResult:
        action_lower = action.strip().lower()

        if action_lower == "reset":
            state = self.reset()
            return StepResult(
                state=state, reward=0.0, done=False, info={"action": "reset"}
            )

        if self._done:
            return StepResult(
                state=self._build_state(done=True),
                reward=0.0,
                done=True,
                info={"error": "Episode already finished. Call reset()."},
            )

        if action_lower not in self._ACTION_MAP:
            return StepResult(
                state=self._build_state(done=self._done),
                reward=0.0,
                done=self._done,
                info={"error": f"Invalid action '{action}'."},
            )

        self._last_action_was_reset = False

        e = self._engine
        game_action = self._ACTION_MAP[action_lower]
        action_input = ActionInput(id=game_action)

        level_before = e.level_index
        total_levels = len(e._levels)

        result = e.perform_action(action_input, raw=True)

        self._turn += 1

        reward = 0.0
        done = False
        info: Dict[str, Any] = {"action": action}

        if result.state.name == "WIN":
            done = True
            self._done = True
            reward = 1.0 / total_levels
            info["reason"] = "game_complete"
        elif result.state.name == "GAME_OVER":
            done = True
            self._done = True
            info["reason"] = "game_over"
        elif e.level_index > level_before:
            reward = 1.0 / total_levels
            info["reason"] = "level_complete"

        return StepResult(
            state=self._build_state(done=done),
            reward=reward,
            done=done,
            info=info,
        )

    def is_done(self) -> bool:
        return self._done

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError(f"Unsupported render mode: {mode}")
        frame = self._render_frame()
        h, w = frame.shape[:2]
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for idx, color in enumerate(self.ARC_PALETTE):
            mask = frame == idx
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
