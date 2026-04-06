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


BG = 5
WALL_C = 3
FLOOR_C = 4
PLAYER_C = 11
WHITE = 0
OFFWHT = 1
LGREY = 2
RED = 8
BLUE = 9
GREEN = 14
ORANGE = 12
PURPLE = 15
MAGENTA = 6
PINK = 7
LBLUE = 10
MAROON = 13
YELLOW = 11

MAX_LIVES = 3

GAME_DESCRIPTION = (
    "Sweet Venom is a turn-based candy-crush puzzle with poison, conveyor, "
    "and warp mechanics. Push colored candy gems into rows or columns of 3+ "
    "matching colors to cascade-clear them. Conveyors automatically redirect "
    "gems that land on them. Warp portals teleport gems and players between "
    "linked tiles. Poison tiles spread — collect shields and antidotes to "
    "survive. Clear EVERY gem to unlock the exit gate and escape."
)

GAME_RULES = (
    "1. Move the yellow player with up/down/left/right.\n"
    "2. Walk into a candy gem to PUSH it in your movement direction. "
    "The gem slides until it hits a wall, another gem, or an obstacle.\n"
    "3. Three candy colors: Red, Blue, Green.\n"
    "4. When 3+ gems of the same color align in a row or column after a push, "
    "they CASCADE-CLEAR (are removed from the board).\n"
    "5. CONVEYORS (pink, shown as > < ^ v): When a gem stops on a conveyor, "
    "it is immediately pushed one cell in the arrow direction. Conveyors "
    "chain — if the next cell is also a conveyor, the gem keeps moving. "
    "The player walks over conveyors without being affected.\n"
    "6. WARP PORTALS (off-white, shown as 1/2 or 3/4): Paired tiles. When "
    "a sliding gem reaches a warp, it teleports to the partner warp and "
    "continues sliding in the same direction. The player can also walk "
    "through warps to teleport.\n"
    "7. The exit gate opens ONLY when ALL gems have been cleared.\n"
    "8. Poison tiles (maroon) spread to an adjacent empty cell every N turns.\n"
    "9. Stepping on poison WITHOUT a shield kills you (lose 1 life, reset).\n"
    "10. Shields (orange) — carry up to 2. Consumed instead of dying on poison.\n"
    "11. Antidotes (light blue) — cleanse all poison in a 3x3 area on pickup.\n"
    "12. Sticky tiles (purple) lock gems permanently — cannot push stuck gems.\n"
    "13. Poison spawners (magenta) generate poison periodically. "
    "Clear gems adjacent to a spawner to destroy it.\n"
    "14. You have 3 lives per level. Running out of moves costs a life.\n"
    "15. HUD: shields (top-left), gate status (top-center), lives (top-right), "
    "move budget bar (bottom row)."
)


def _px(color, name, tags=None, layer=0, collidable=False, visible=True):
    return Sprite(
        pixels=[[color]],
        name=name,
        visible=visible,
        collidable=collidable,
        tags=tags or [],
        layer=layer,
    )


sprites = {
    "player": _px(PLAYER_C, "player", ["player"], layer=8, collidable=True),
    "wall": _px(WALL_C, "wall", ["wall", "solid"], layer=0, collidable=True),
    "floor": _px(FLOOR_C, "floor", ["floor"], layer=-1),
    "gem_red": _px(RED, "gem_red", ["gem", "gem_red"], layer=2),
    "gem_blue": _px(BLUE, "gem_blue", ["gem", "gem_blue"], layer=2),
    "gem_green": _px(GREEN, "gem_green", ["gem", "gem_green"], layer=2),
    "poison": _px(MAROON, "poison", ["poison"], layer=-1),
    "spawner": _px(MAGENTA, "spawner", ["spawner"], layer=1),
    "shield": _px(ORANGE, "shield", ["shield_pickup"], layer=2),
    "antidote": _px(LBLUE, "antidote", ["antidote"], layer=2),
    "sticky": _px(PURPLE, "sticky", ["sticky"], layer=-1),
    "conveyor": _px(PINK, "conveyor", ["conveyor"], layer=-1),
    "warp": _px(OFFWHT, "warp", ["warp"], layer=-1),
    "gate_locked": _px(
        LGREY, "gate_locked", ["gate", "solid"], layer=1, collidable=True
    ),
    "gate_open": _px(GREEN, "gate_open", ["gate_open", "exit"], layer=1),
    "flash": _px(LGREY, "flash", ["flash"], layer=10, visible=False),
}

DIR_MAP = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}


def _build_border(w, h):
    walls = []
    for x in range(w):
        walls.append(sprites["wall"].clone().set_position(x, 0))
        walls.append(sprites["wall"].clone().set_position(x, h - 1))
    for y in range(1, h - 1):
        walls.append(sprites["wall"].clone().set_position(0, y))
        walls.append(sprites["wall"].clone().set_position(w - 1, y))
    return walls


def _walls_from_list(positions):
    return [sprites["wall"].clone().set_position(x, y) for x, y in positions]


_L1_W, _L1_H = 12, 12
_l1_walls = [
    (6, 1),
    (6, 2),
    (6, 4),
    (6, 5),
    (6, 6),
    (6, 7),
    (6, 9),
    (6, 10),
    (3, 5),
    (9, 5),
]
_l1 = (
    _build_border(_L1_W, _L1_H)
    + _walls_from_list(_l1_walls)
    + [
        sprites["player"].clone().set_position(1, 10),
        sprites["gate_locked"].clone().set_position(10, 1),
    ]
)
level1 = Level(
    sprites=_l1,
    grid_size=(_L1_W, _L1_H),
    data={
        "gems": [
            [1, 2, "red"],
            [2, 2, "red"],
            [4, 2, "red"],
            [1, 4, "blue"],
            [1, 7, "blue"],
            [1, 8, "blue"],
            [2, 9, "green"],
            [4, 9, "green"],
            [5, 9, "green"],
            [8, 1, "red"],
            [8, 2, "red"],
            [8, 4, "red"],
            [7, 5, "green"],
            [7, 7, "green"],
            [7, 8, "green"],
            [10, 4, "blue"],
            [10, 7, "blue"],
            [10, 8, "blue"],
        ],
        "poisons": [[1, 1], [10, 10]],
        "shields": [[1, 3], [10, 9]],
        "antidotes": [[5, 10]],
        "sticky_tiles": [],
        "spawners": [],
        "conveyors": [[1, 7, "up"], [7, 4, "right"]],
        "warps": [[[5, 1], [7, 10]]],
        "gate_pos": [10, 1],
        "start_pos": [1, 10],
        "max_moves": 180,
        "poison_grow_rate": 10,
    },
)

_L2_W, _L2_H = 14, 14
_l2_walls = [
    (5, 1),
    (5, 2),
    (5, 3),
    (5, 5),
    (5, 6),
    (5, 7),
    (5, 8),
    (5, 9),
    (5, 11),
    (5, 12),
    (9, 1),
    (9, 2),
    (9, 4),
    (9, 5),
    (9, 6),
    (9, 7),
    (9, 8),
    (9, 10),
    (9, 11),
    (9, 12),
    (3, 7),
    (7, 6),
    (11, 7),
]
_l2 = (
    _build_border(_L2_W, _L2_H)
    + _walls_from_list(_l2_walls)
    + [
        sprites["player"].clone().set_position(1, 12),
        sprites["gate_locked"].clone().set_position(12, 1),
    ]
)
level2 = Level(
    sprites=_l2,
    grid_size=(_L2_W, _L2_H),
    data={
        "gems": [
            [2, 1, "red"],
            [2, 2, "red"],
            [2, 5, "red"],
            [4, 6, "blue"],
            [4, 8, "blue"],
            [4, 9, "blue"],
            [6, 2, "green"],
            [6, 3, "green"],
            [6, 5, "green"],
            [6, 8, "red"],
            [6, 11, "red"],
            [6, 12, "red"],
            [11, 1, "blue"],
            [11, 2, "blue"],
            [11, 5, "blue"],
            [10, 8, "green"],
            [10, 10, "green"],
            [10, 11, "green"],
        ],
        "poisons": [[4, 6], [8, 5], [12, 6]],
        "shields": [[1, 4], [8, 12], [12, 4]],
        "antidotes": [[7, 8]],
        "sticky_tiles": [[3, 4], [7, 4], [11, 4]],
        "spawners": [],
        "conveyors": [
            [4, 5, "down"],
            [6, 11, "right"],
            [7, 11, "right"],
            [12, 9, "up"],
        ],
        "warps": [[[1, 11], [12, 3]]],
        "gate_pos": [12, 1],
        "start_pos": [1, 12],
        "max_moves": 192,
        "poison_grow_rate": 7,
    },
)

_L3_W, _L3_H = 16, 16
_l3_walls = [
    (5, 1),
    (5, 2),
    (5, 3),
    (5, 5),
    (5, 6),
    (5, 7),
    (5, 8),
    (5, 10),
    (5, 11),
    (5, 12),
    (5, 14),
    (10, 1),
    (10, 2),
    (10, 4),
    (10, 5),
    (10, 6),
    (10, 7),
    (10, 9),
    (10, 10),
    (10, 11),
    (10, 13),
    (10, 14),
    (3, 7),
    (8, 5),
    (13, 7),
    (3, 12),
    (8, 11),
    (13, 12),
]
_l3 = (
    _build_border(_L3_W, _L3_H)
    + _walls_from_list(_l3_walls)
    + [
        sprites["player"].clone().set_position(1, 14),
        sprites["gate_locked"].clone().set_position(14, 1),
    ]
)
level3 = Level(
    sprites=_l3,
    grid_size=(_L3_W, _L3_H),
    data={
        "gems": [
            [2, 1, "red"],
            [2, 2, "red"],
            [2, 4, "red"],
            [2, 5, "blue"],
            [2, 6, "blue"],
            [2, 9, "blue"],
            [2, 10, "green"],
            [2, 11, "green"],
            [2, 13, "green"],
            [7, 2, "green"],
            [7, 3, "green"],
            [7, 6, "green"],
            [9, 6, "red"],
            [9, 7, "red"],
            [9, 9, "red"],
            [7, 10, "blue"],
            [7, 13, "blue"],
            [7, 14, "blue"],
            [12, 1, "blue"],
            [12, 2, "blue"],
            [12, 4, "blue"],
            [14, 5, "green"],
            [14, 6, "green"],
            [14, 9, "green"],
            [14, 10, "red"],
            [14, 11, "red"],
            [14, 13, "red"],
        ],
        "poisons": [[4, 4], [9, 4], [14, 4]],
        "shields": [[1, 5], [6, 14], [11, 5]],
        "antidotes": [[4, 14], [14, 14]],
        "sticky_tiles": [[3, 5], [8, 9], [13, 5]],
        "spawners": [[4, 8], [11, 3]],
        "conveyors": [
            [4, 5, "down"],
            [6, 6, "right"],
            [7, 6, "right"],
            [9, 12, "left"],
            [14, 9, "up"],
            [14, 8, "up"],
        ],
        "warps": [[[1, 1], [14, 13]], [[4, 13], [11, 1]]],
        "gate_pos": [14, 1],
        "start_pos": [1, 14],
        "max_moves": 288,
        "poison_grow_rate": 6,
    },
)

_L4_W, _L4_H = 18, 18
_l4_walls = [
    (6, 1),
    (6, 2),
    (6, 3),
    (6, 5),
    (6, 6),
    (6, 7),
    (6, 8),
    (6, 10),
    (6, 11),
    (6, 12),
    (6, 13),
    (6, 15),
    (6, 16),
    (12, 1),
    (12, 2),
    (12, 4),
    (12, 5),
    (12, 6),
    (12, 7),
    (12, 9),
    (12, 10),
    (12, 11),
    (12, 12),
    (12, 14),
    (12, 15),
    (12, 16),
    (2, 9),
    (3, 9),
    (4, 9),
    (8, 9),
    (9, 9),
    (10, 9),
    (14, 9),
    (15, 9),
    (3, 5),
    (9, 4),
    (15, 5),
    (3, 13),
    (9, 13),
    (15, 13),
]
_l4 = (
    _build_border(_L4_W, _L4_H)
    + _walls_from_list(_l4_walls)
    + [
        sprites["player"].clone().set_position(1, 16),
        sprites["gate_locked"].clone().set_position(16, 1),
    ]
)
level4 = Level(
    sprites=_l4,
    grid_size=(_L4_W, _L4_H),
    data={
        "gems": [
            [2, 1, "red"],
            [2, 2, "red"],
            [2, 5, "red"],
            [4, 5, "blue"],
            [4, 7, "blue"],
            [4, 8, "blue"],
            [1, 11, "green"],
            [2, 11, "green"],
            [4, 11, "green"],
            [1, 12, "red"],
            [1, 14, "red"],
            [1, 15, "red"],
            [8, 1, "blue"],
            [8, 2, "blue"],
            [8, 5, "blue"],
            [8, 6, "green"],
            [10, 6, "green"],
            [11, 6, "green"],
            [8, 10, "red"],
            [8, 11, "red"],
            [8, 14, "red"],
            [10, 11, "blue"],
            [10, 14, "blue"],
            [10, 15, "blue"],
            [14, 1, "green"],
            [14, 2, "green"],
            [14, 5, "green"],
            [13, 4, "red"],
            [13, 6, "red"],
            [13, 7, "red"],
            [14, 10, "blue"],
            [14, 11, "blue"],
            [14, 14, "blue"],
            [16, 12, "green"],
            [16, 14, "green"],
            [16, 15, "green"],
        ],
        "poisons": [[5, 3], [11, 5], [5, 14], [11, 11], [16, 8]],
        "shields": [[1, 4], [7, 16], [13, 3], [15, 16]],
        "antidotes": [[5, 8], [11, 8], [15, 14]],
        "sticky_tiles": [[3, 6], [9, 7], [15, 6], [3, 14], [9, 12]],
        "spawners": [[5, 10], [11, 3], [16, 10]],
        "conveyors": [
            [1, 6, "down"],
            [1, 7, "down"],
            [7, 5, "right"],
            [8, 5, "right"],
            [11, 14, "left"],
            [10, 14, "left"],
            [16, 7, "up"],
            [16, 6, "up"],
            [5, 4, "right"],
            [5, 15, "left"],
        ],
        "warps": [[[1, 1], [16, 16]], [[5, 16], [13, 1]], [[1, 10], [16, 3]]],
        "gate_pos": [16, 1],
        "start_pos": [1, 16],
        "max_moves": 360,
        "poison_grow_rate": 5,
    },
)

levels = [level1, level2, level3, level4]


class SweetVenomHUD(RenderableUserDisplay):
    def __init__(self):
        self.shields_held = 0
        self.is_gate_open = False
        self.lives = MAX_LIVES
        self.moves_remaining = 0
        self.max_moves = 0
        self.goals_met = False

    def update(
        self,
        shields_held,
        is_gate_open,
        lives,
        moves_remaining=0,
        max_moves=0,
        goals_met=False,
    ):
        self.shields_held = shields_held
        self.is_gate_open = is_gate_open
        self.lives = lives
        self.moves_remaining = moves_remaining
        self.max_moves = max_moves
        self.goals_met = goals_met

    def render_interface(self, frame):
        h, w = frame.shape[:2]
        for i in range(2):
            c = ORANGE if i < self.shields_held else BG
            if 0 <= i < w:
                frame[0, i] = c
        for i in range(MAX_LIVES):
            px = w - 1 - i * 2
            c = RED if i < self.lives else BG
            if 0 <= px < w:
                frame[0, px] = c
        if self.max_moves > 0 and h > 1:
            bar_row = h - 1
            ratio = self.moves_remaining / self.max_moves
            filled = int(w * ratio)
            for i in range(w):
                if i < filled:
                    if ratio > 0.5:
                        frame[bar_row, i] = GREEN
                    elif ratio > 0.25:
                        frame[bar_row, i] = ORANGE
                    else:
                        frame[bar_row, i] = RED
                else:
                    frame[bar_row, i] = BG
        return frame


class Fw91(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self.hud = SweetVenomHUD()
        self.player = None
        self.lives = MAX_LIVES
        self.moves_remaining = 0
        self.max_moves = 0
        self.shields_held = 0
        self.gem_data = {}
        self.gem_sprites = {}
        self.initial_gems = {}
        self.poison_set = set()
        self.poison_sprites = {}
        self.initial_poisons = set()
        self.poison_grow_rate = 0
        self.action_count = 0
        self.shield_positions = set()
        self.shield_sprites_map = {}
        self.initial_shields = []
        self.antidote_positions = set()
        self.antidote_sprites_map = {}
        self.initial_antidotes = []
        self.sticky_set = set()
        self.sticky_sprites_map = {}
        self.spawner_positions = set()
        self.spawner_sprites = {}
        self.initial_spawners = []
        self.conveyor_map = {}
        self.conveyor_sprites_map = {}
        self.warp_map = {}
        self.warp_sprites_map = {}
        self.gate_sprite = None
        self.gate_is_open = False
        self._start_x = 0
        self._start_y = 0
        self._removed_sprites = []
        self.flash_sprite = None
        self.flash_active = False
        self._state_history: List[dict] = []
        cam_w = min(max(lev.grid_size[0] for lev in levels), 64)
        cam_h = min(max(lev.grid_size[1] for lev in levels), 64)
        camera = Camera(x=0, y=0, width=cam_w, height=cam_h, background=BG, letter_box=FLOOR_C, interfaces=[self.hud])
        super().__init__("fw91", levels, camera, available_actions=[0, 1, 2, 3, 4, 7])

    def on_set_level(self, level: Level) -> None:
        self.player = self.current_level.get_sprites_by_tag("player")[0]
        sp = self.current_level.get_data("start_pos")
        if sp:
            self._start_x, self._start_y = sp
        else:
            self._start_x = self.player.x
            self._start_y = self.player.y
        self.max_moves = self.current_level.get_data("max_moves") or 50
        self.moves_remaining = self.max_moves
        self.lives = MAX_LIVES
        self.shields_held = 0
        self.gate_is_open = False
        self._removed_sprites = []
        self.action_count = 0
        self._state_history.clear()
        for tag in (
            "gem",
            "poison",
            "shield_pickup",
            "antidote",
            "sticky",
            "spawner",
            "conveyor",
            "warp",
            "flash",
        ):
            for s in list(self.current_level.get_sprites_by_tag(tag)):
                self.current_level.remove_sprite(s)
        self.gem_data.clear()
        self.gem_sprites.clear()
        self.initial_gems.clear()
        self.poison_set.clear()
        self.poison_sprites.clear()
        self.initial_poisons.clear()
        self.shield_positions.clear()
        self.shield_sprites_map.clear()
        self.initial_shields.clear()
        self.antidote_positions.clear()
        self.antidote_sprites_map.clear()
        self.initial_antidotes.clear()
        self.sticky_set.clear()
        self.sticky_sprites_map.clear()
        self.spawner_positions.clear()
        self.spawner_sprites.clear()
        self.initial_spawners.clear()
        self.conveyor_map.clear()
        self.conveyor_sprites_map.clear()
        self.warp_map.clear()
        self.warp_sprites_map.clear()
        gates = self.current_level.get_sprites_by_tag("gate")
        if gates:
            self.gate_sprite = gates[0]
        else:
            open_gates = self.current_level.get_sprites_by_tag("gate_open")
            if open_gates:
                gx, gy = open_gates[0].x, open_gates[0].y
                self.current_level.remove_sprite(open_gates[0])
                new_gate = sprites["gate_locked"].clone().set_position(gx, gy)
                self.current_level.add_sprite(new_gate)
                self.gate_sprite = new_gate
            else:
                self.gate_sprite = None
        for cfg in self.current_level.get_data("gems") or []:
            x, y, color = cfg
            self._place_gem(x, y, color)
            self.initial_gems[(x, y)] = color
        for cfg in self.current_level.get_data("poisons") or []:
            x, y = cfg
            self._place_poison(x, y)
        self.initial_poisons = set(self.poison_set)
        for cfg in self.current_level.get_data("shields") or []:
            x, y = cfg
            self._place_shield_pickup(x, y)
            self.initial_shields.append((x, y))
        for cfg in self.current_level.get_data("antidotes") or []:
            x, y = cfg
            self._place_antidote(x, y)
            self.initial_antidotes.append((x, y))
        for cfg in self.current_level.get_data("sticky_tiles") or []:
            x, y = cfg
            self._place_sticky(x, y)
        for cfg in self.current_level.get_data("spawners") or []:
            x, y = cfg
            self._place_spawner(x, y)
            self.initial_spawners.append((x, y))
        for cfg in self.current_level.get_data("conveyors") or []:
            x, y, direction = cfg
            self._place_conveyor(x, y, direction)
        for pair in self.current_level.get_data("warps") or []:
            a, b = pair
            self._place_warp_pair(a[0], a[1], b[0], b[1])
        self.poison_grow_rate = self.current_level.get_data("poison_grow_rate") or 0
        flash_list = self.current_level.get_sprites_by_tag("flash")
        if flash_list:
            self.flash_sprite = flash_list[0]
        else:
            self.flash_sprite = sprites["flash"].clone()
            self.current_level.add_sprite(self.flash_sprite)
        self.flash_sprite.set_visible(False)
        self.flash_active = False
        self._update_hud()

    def _place_gem(self, x, y, color, stuck=False):
        tag = "gem_" + color
        spr = sprites[tag].clone().set_position(x, y)
        self.current_level.add_sprite(spr)
        self.gem_sprites[(x, y)] = spr
        self.gem_data[(x, y)] = {"color": color, "stuck": stuck}

    def _place_poison(self, x, y):
        if (x, y) in self.poison_set:
            return
        spr = sprites["poison"].clone().set_position(x, y)
        self.current_level.add_sprite(spr)
        self.poison_set.add((x, y))
        self.poison_sprites[(x, y)] = spr

    def _place_shield_pickup(self, x, y):
        spr = sprites["shield"].clone().set_position(x, y)
        self.current_level.add_sprite(spr)
        self.shield_positions.add((x, y))
        self.shield_sprites_map[(x, y)] = spr

    def _place_antidote(self, x, y):
        spr = sprites["antidote"].clone().set_position(x, y)
        self.current_level.add_sprite(spr)
        self.antidote_positions.add((x, y))
        self.antidote_sprites_map[(x, y)] = spr

    def _place_sticky(self, x, y):
        spr = sprites["sticky"].clone().set_position(x, y)
        self.current_level.add_sprite(spr)
        self.sticky_set.add((x, y))
        self.sticky_sprites_map[(x, y)] = spr

    def _place_spawner(self, x, y):
        spr = sprites["spawner"].clone().set_position(x, y)
        self.current_level.add_sprite(spr)
        self.spawner_positions.add((x, y))
        self.spawner_sprites[(x, y)] = spr

    def _place_conveyor(self, x, y, direction):
        dx, dy = DIR_MAP.get(direction, (0, 0))
        spr = sprites["conveyor"].clone().set_position(x, y)
        self.current_level.add_sprite(spr)
        self.conveyor_map[(x, y)] = (dx, dy)
        self.conveyor_sprites_map[(x, y)] = spr

    def _place_warp_pair(self, x1, y1, x2, y2):
        spr1 = sprites["warp"].clone().set_position(x1, y1)
        spr2 = sprites["warp"].clone().set_position(x2, y2)
        self.current_level.add_sprite(spr1)
        self.current_level.add_sprite(spr2)
        self.warp_map[(x1, y1)] = (x2, y2)
        self.warp_map[(x2, y2)] = (x1, y1)
        self.warp_sprites_map[(x1, y1)] = spr1
        self.warp_sprites_map[(x2, y2)] = spr2

    def _is_blocked(self, x, y):
        for s in self.current_level._sprites:
            if s.x == x and s.y == y and s.tags:
                if "wall" in s.tags:
                    return True
                if "gate" in s.tags and "solid" in s.tags:
                    return True
        if (x, y) in self.gem_data:
            return True
        if (x, y) in self.spawner_positions:
            return True
        return False

    def _is_wall_or_border(self, x, y):
        for s in self.current_level._sprites:
            if s.x == x and s.y == y and s.tags:
                if "wall" in s.tags:
                    return True
                if "gate" in s.tags and "solid" in s.tags:
                    return True
        return False

    def _is_gem_blocked(self, x, y):
        if self._is_wall_or_border(x, y):
            return True
        if (x, y) in self.gem_data:
            return True
        if (x, y) in self.spawner_positions:
            return True
        return False

    def _push_gem(self, gx, gy, dx, dy):
        if (gx, gy) not in self.gem_data:
            return False
        gem_info = self.gem_data[(gx, gy)]
        if gem_info["stuck"]:
            return False
        final_x, final_y = self._slide_gem(gx, gy, dx, dy)
        if final_x == gx and final_y == gy:
            return False
        color = gem_info["color"]
        self.current_level.remove_sprite(self.gem_sprites[(gx, gy)])
        del self.gem_sprites[(gx, gy)]
        del self.gem_data[(gx, gy)]
        new_stuck = (final_x, final_y) in self.sticky_set
        self._place_gem(final_x, final_y, color, stuck=new_stuck)
        self._apply_conveyors(final_x, final_y, max_steps=20)
        return True

    def _slide_gem(self, gx, gy, dx, dy):
        cx, cy = gx, gy
        warps_used = set()
        for _ in range(200):
            nx, ny = cx + dx, cy + dy
            if self._is_gem_blocked(nx, ny):
                break
            cx, cy = nx, ny
            if (cx, cy) in self.warp_map and (cx, cy) not in warps_used:
                partner = self.warp_map[(cx, cy)]
                warps_used.add((cx, cy))
                warps_used.add(partner)
                px, py = partner
                exit_x, exit_y = px + dx, py + dy
                if not self._is_gem_blocked(exit_x, exit_y):
                    cx, cy = exit_x, exit_y
                else:
                    cx, cy = px, py
                    if self._is_gem_blocked(cx, cy):
                        for wk in warps_used:
                            if self.warp_map.get(wk) == (px, py):
                                cx, cy = wk
                                break
                    break
        return cx, cy

    def _apply_conveyors(self, gx, gy, max_steps=20):
        for _ in range(max_steps):
            if (gx, gy) not in self.conveyor_map:
                break
            if (gx, gy) not in self.gem_data:
                break
            gem_info = self.gem_data[(gx, gy)]
            if gem_info["stuck"]:
                break
            cdx, cdy = self.conveyor_map[(gx, gy)]
            nx, ny = gx + cdx, gy + cdy
            if self._is_gem_blocked(nx, ny):
                break
            color = gem_info["color"]
            stuck = gem_info["stuck"]
            self.current_level.remove_sprite(self.gem_sprites[(gx, gy)])
            del self.gem_sprites[(gx, gy)]
            del self.gem_data[(gx, gy)]
            new_stuck = stuck or (nx, ny) in self.sticky_set
            self._place_gem(nx, ny, color, stuck=new_stuck)
            gx, gy = nx, ny

    def _check_matches(self):
        to_clear = set()
        grid_w, grid_h = self.current_level.grid_size
        for y in range(grid_h):
            run_start = 0
            run_color = None
            run_len = 0
            for x in range(grid_w):
                if (x, y) in self.gem_data:
                    c = self.gem_data[(x, y)]["color"]
                    if c == run_color:
                        run_len += 1
                    else:
                        if run_len >= 3 and run_color is not None:
                            for rx in range(run_start, run_start + run_len):
                                to_clear.add((rx, y))
                        run_start = x
                        run_color = c
                        run_len = 1
                else:
                    if run_len >= 3 and run_color is not None:
                        for rx in range(run_start, run_start + run_len):
                            to_clear.add((rx, y))
                    run_start = 0
                    run_color = None
                    run_len = 0
            if run_len >= 3 and run_color is not None:
                for rx in range(run_start, run_start + run_len):
                    to_clear.add((rx, y))
        for x in range(grid_w):
            run_start = 0
            run_color = None
            run_len = 0
            for y in range(grid_h):
                if (x, y) in self.gem_data:
                    c = self.gem_data[(x, y)]["color"]
                    if c == run_color:
                        run_len += 1
                    else:
                        if run_len >= 3 and run_color is not None:
                            for ry in range(run_start, run_start + run_len):
                                to_clear.add((x, ry))
                        run_start = y
                        run_color = c
                        run_len = 1
                else:
                    if run_len >= 3 and run_color is not None:
                        for ry in range(run_start, run_start + run_len):
                            to_clear.add((x, ry))
                    run_start = 0
                    run_color = None
                    run_len = 0
            if run_len >= 3 and run_color is not None:
                for ry in range(run_start, run_start + run_len):
                    to_clear.add((x, ry))
        return to_clear

    def _clear_gems(self, positions):
        for x, y in positions:
            if (x, y) in self.gem_data:
                self.current_level.remove_sprite(self.gem_sprites[(x, y)])
                del self.gem_sprites[(x, y)]
                del self.gem_data[(x, y)]
        self._check_spawner_destruction(positions)

    def _check_spawner_destruction(self, cleared_positions):
        to_destroy = set()
        for cx, cy in cleared_positions:
            for ddx, ddy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                sx, sy = cx + ddx, cy + ddy
                if (sx, sy) in self.spawner_positions:
                    to_destroy.add((sx, sy))
        for sx, sy in to_destroy:
            self.current_level.remove_sprite(self.spawner_sprites[(sx, sy)])
            del self.spawner_sprites[(sx, sy)]
            self.spawner_positions.discard((sx, sy))

    def _do_cascade(self):
        for _ in range(50):
            matches = self._check_matches()
            if not matches:
                break
            self._clear_gems(matches)

    def _goals_met(self):
        return len(self.gem_data) == 0

    def _open_gate(self):
        if self.gate_is_open:
            return
        self.gate_is_open = True
        if self.gate_sprite:
            gx, gy = self.gate_sprite.x, self.gate_sprite.y
            self.current_level.remove_sprite(self.gate_sprite)
            self._removed_sprites.append(self.gate_sprite)
            new_gate = sprites["gate_open"].clone().set_position(gx, gy)
            self.current_level.add_sprite(new_gate)
            self.gate_sprite = new_gate
        self._update_hud()

    def _close_gate(self):
        if not self.gate_is_open:
            return
        self.gate_is_open = False
        if self.gate_sprite:
            gx, gy = self.gate_sprite.x, self.gate_sprite.y
            self.current_level.remove_sprite(self.gate_sprite)
            self._removed_sprites.append(self.gate_sprite)
            new_gate = sprites["gate_locked"].clone().set_position(gx, gy)
            self.current_level.add_sprite(new_gate)
            self.gate_sprite = new_gate
        self._update_hud()

    def _check_gate(self):
        if self._goals_met():
            if not self.gate_is_open:
                self._open_gate()
        else:
            if self.gate_is_open:
                self._close_gate()

    def _grow_poison(self):
        grid_w, grid_h = self.current_level.grid_size
        candidates = []
        for px, py in list(self.poison_set):
            for ddx, ddy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nx, ny = px + ddx, py + ddy
                if nx <= 0 or nx >= grid_w - 1 or ny <= 0 or ny >= grid_h - 1:
                    continue
                if (nx, ny) in self.poison_set:
                    continue
                if self._is_wall_or_border(nx, ny):
                    continue
                if (nx, ny) in self.gem_data:
                    continue
                if (nx, ny) in self.spawner_positions:
                    continue
                if (nx, ny) in self.shield_positions:
                    continue
                if (nx, ny) in self.sticky_set:
                    continue
                if (
                    self.gate_sprite
                    and nx == self.gate_sprite.x
                    and ny == self.gate_sprite.y
                ):
                    continue
                if (nx, ny) == (self._start_x, self._start_y):
                    continue
                candidates.append((nx, ny))
        if candidates:
            mid_x = (grid_w - 1) / 2.0
            mid_y = (grid_h - 1) / 2.0
            candidates.sort(
                key=lambda p: (-(abs(p[0] - mid_x) + abs(p[1] - mid_y)), p[1], p[0])
            )
            self._place_poison(candidates[0][0], candidates[0][1])

    def _spawner_tick(self):
        grid_w, grid_h = self.current_level.grid_size
        for sx, sy in list(self.spawner_positions):
            for ddx, ddy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nx, ny = sx + ddx, sy + ddy
                if nx <= 0 or nx >= grid_w - 1 or ny <= 0 or ny >= grid_h - 1:
                    continue
                if (nx, ny) in self.poison_set:
                    continue
                if self._is_wall_or_border(nx, ny):
                    continue
                if (nx, ny) in self.gem_data:
                    continue
                if (nx, ny) in self.shield_positions:
                    continue
                if (nx, ny) == (self._start_x, self._start_y):
                    continue
                if (
                    self.gate_sprite
                    and nx == self.gate_sprite.x
                    and ny == self.gate_sprite.y
                ):
                    continue
                self._place_poison(nx, ny)
                break

    def _use_antidote(self, px, py):
        for ddx in range(-1, 2):
            for ddy in range(-1, 2):
                cx, cy = px + ddx, py + ddy
                if (cx, cy) in self.poison_set:
                    self.current_level.remove_sprite(self.poison_sprites[(cx, cy)])
                    del self.poison_sprites[(cx, cy)]
                    self.poison_set.discard((cx, cy))

    def _handle_death(self):
        self.lives -= 1
        if self.lives <= 0:
            self.lives = MAX_LIVES
            self.lose()
        else:
            if self.flash_sprite:
                self.flash_sprite.set_visible(True)
                self.flash_sprite.set_scale(
                    max(
                        self.current_level.grid_size[0], self.current_level.grid_size[1]
                    )
                )
                self.flash_sprite.set_position(0, 0)
            self.flash_active = True
            self._reset_level_state()

    def _save_state_snapshot(self) -> dict:
        return {
            "player_x": self.player.x,
            "player_y": self.player.y,
            "gem_data": {pos: dict(info) for pos, info in self.gem_data.items()},
            "poison_set": set(self.poison_set),
            "shields_held": self.shields_held,
            "shield_positions": set(self.shield_positions),
            "antidote_positions": set(self.antidote_positions),
            "spawner_positions": set(self.spawner_positions),
            "gate_is_open": self.gate_is_open,
            "moves_remaining": self.moves_remaining,
            "action_count": self.action_count,
            "lives": self.lives,
        }

    def _restore_state_snapshot(self, snap: dict, restore_moves: bool = True) -> None:
        for tag in ("gem", "poison", "shield_pickup", "antidote", "spawner"):
            for s in list(self.current_level.get_sprites_by_tag(tag)):
                self.current_level.remove_sprite(s)
        self.gem_data.clear()
        self.gem_sprites.clear()
        self.poison_set.clear()
        self.poison_sprites.clear()
        self.shield_positions.clear()
        self.shield_sprites_map.clear()
        self.antidote_positions.clear()
        self.antidote_sprites_map.clear()
        self.spawner_positions.clear()
        self.spawner_sprites.clear()
        for pos, info in snap["gem_data"].items():
            self._place_gem(pos[0], pos[1], info["color"], stuck=info.get("stuck", False))
        for x, y in snap["poison_set"]:
            self._place_poison(x, y)
        for x, y in snap["shield_positions"]:
            self._place_shield_pickup(x, y)
        for x, y in snap["antidote_positions"]:
            self._place_antidote(x, y)
        for x, y in snap["spawner_positions"]:
            self._place_spawner(x, y)
        self.shields_held = snap["shields_held"]
        if restore_moves:
            self.moves_remaining = snap["moves_remaining"]
        self.action_count = snap["action_count"]
        self.lives = snap["lives"]
        self.player.set_position(snap["player_x"], snap["player_y"])
        if snap["gate_is_open"] and not self.gate_is_open:
            self._open_gate()
        elif not snap["gate_is_open"] and self.gate_is_open:
            self._close_gate()
        self._update_hud()

    def _handle_undo(self) -> None:
        if not self._state_history:
            self.moves_remaining -= 1
            self._update_hud()
            if self.moves_remaining <= 0:
                self._handle_death()
                if not self.flash_active:
                    self.complete_action()
                return
            self.complete_action()
            return
        snap = self._state_history.pop()
        self._restore_state_snapshot(snap, restore_moves=False)
        self.moves_remaining -= 1
        self._update_hud()
        if self.moves_remaining <= 0:
            self._handle_death()
            if not self.flash_active:
                self.complete_action()
            return
        self.complete_action()

    def _reset_level_state(self):
        self._state_history.clear()
        for tag in ("gem", "poison", "shield_pickup", "antidote", "spawner"):
            for s in list(self.current_level.get_sprites_by_tag(tag)):
                self.current_level.remove_sprite(s)
        self.gem_data.clear()
        self.gem_sprites.clear()
        self.poison_set.clear()
        self.poison_sprites.clear()
        self.shield_positions.clear()
        self.shield_sprites_map.clear()
        self.antidote_positions.clear()
        self.antidote_sprites_map.clear()
        self.spawner_positions.clear()
        self.spawner_sprites.clear()
        for (x, y), color in self.initial_gems.items():
            self._place_gem(x, y, color)
        for cfg in self.initial_poisons:
            self._place_poison(cfg[0], cfg[1])
        for x, y in self.initial_shields:
            self._place_shield_pickup(x, y)
        for x, y in self.initial_antidotes:
            self._place_antidote(x, y)
        for x, y in self.initial_spawners:
            self._place_spawner(x, y)
        self.shields_held = 0
        self._close_gate()
        self.player.set_position(self._start_x, self._start_y)
        self.moves_remaining = self.max_moves
        self.action_count = 0
        self._update_hud()

    def _update_hud(self):
        self.hud.update(
            self.shields_held,
            self.gate_is_open,
            self.lives,
            self.moves_remaining,
            self.max_moves,
            self._goals_met(),
        )

    def _handle_tile(self, x, y):
        pos = (x, y)
        if pos in self.poison_set:
            if self.shields_held > 0:
                self.shields_held -= 1
                self.current_level.remove_sprite(self.poison_sprites[pos])
                del self.poison_sprites[pos]
                self.poison_set.discard(pos)
            else:
                self._handle_death()
                return False
        if pos in self.shield_positions:
            if self.shields_held < 2:
                self.shields_held += 1
            self.current_level.remove_sprite(self.shield_sprites_map[pos])
            del self.shield_sprites_map[pos]
            self.shield_positions.discard(pos)
        if pos in self.antidote_positions:
            self.current_level.remove_sprite(self.antidote_sprites_map[pos])
            del self.antidote_sprites_map[pos]
            self.antidote_positions.discard(pos)
            self._use_antidote(x, y)
        if pos in self.warp_map:
            partner = self.warp_map[pos]
            self.player.set_position(partner[0], partner[1])
            dest_pos = partner
            if dest_pos in self.poison_set:
                if self.shields_held > 0:
                    self.shields_held -= 1
                    self.current_level.remove_sprite(self.poison_sprites[dest_pos])
                    del self.poison_sprites[dest_pos]
                    self.poison_set.discard(dest_pos)
                else:
                    self._handle_death()
                    return False
            if dest_pos in self.shield_positions:
                if self.shields_held < 2:
                    self.shields_held += 1
                self.current_level.remove_sprite(self.shield_sprites_map[dest_pos])
                del self.shield_sprites_map[dest_pos]
                self.shield_positions.discard(dest_pos)
            if dest_pos in self.antidote_positions:
                self.current_level.remove_sprite(self.antidote_sprites_map[dest_pos])
                del self.antidote_sprites_map[dest_pos]
                self.antidote_positions.discard(dest_pos)
                self._use_antidote(dest_pos[0], dest_pos[1])
        exit_sprite = None
        px, py = self.player.x, self.player.y
        for s in self.current_level._sprites:
            if s.x == px and s.y == py and s.tags and "exit" in s.tags:
                exit_sprite = s
                break
        if exit_sprite and self.gate_is_open:
            self.next_level()
            return False
        return True

    def step(self) -> None:
        if self.flash_active:
            if self.flash_sprite:
                self.flash_sprite.set_visible(False)
            self.flash_active = False
            self.complete_action()
            return
        action = self.action.id.value
        if action == 7:
            self._handle_undo()
            return
        if action not in (1, 2, 3, 4):
            self.complete_action()
            return
        self._state_history.append(self._save_state_snapshot())
        dx, dy = 0, 0
        if action == 1:
            dy = -1
        elif action == 2:
            dy = 1
        elif action == 3:
            dx = -1
        elif action == 4:
            dx = 1
        self.moves_remaining -= 1
        self._update_hud()
        if self.moves_remaining <= 0:
            self._handle_death()
            if not self.flash_active:
                self.complete_action()
            return
        nx = self.player.x + dx
        ny = self.player.y + dy
        if (nx, ny) in self.gem_data:
            pushed = self._push_gem(nx, ny, dx, dy)
            if pushed:
                self._do_cascade()
                self._check_gate()
                if not self._is_blocked(nx, ny):
                    self.player.set_position(nx, ny)
                    ok = self._handle_tile(nx, ny)
                    if not ok:
                        if not self.flash_active:
                            self.complete_action()
                        return
        elif self._is_blocked(nx, ny):
            pass
        else:
            self.player.set_position(nx, ny)
            ok = self._handle_tile(nx, ny)
            if not ok:
                if not self.flash_active:
                    self.complete_action()
                return
        if self.poison_grow_rate > 0:
            self.action_count += 1
            if self.action_count >= self.poison_grow_rate:
                self.action_count = 0
                self._grow_poison()
                self._spawner_tick()
                px, py = self.player.x, self.player.y
                if (px, py) in self.poison_set:
                    if self.shields_held > 0:
                        self.shields_held -= 1
                        self.current_level.remove_sprite(self.poison_sprites[(px, py)])
                        del self.poison_sprites[(px, py)]
                        self.poison_set.discard((px, py))
                    else:
                        self._handle_death()
        if not self.flash_active:
            self._update_hud()
            self.complete_action()


ACTION_MAP = {"up": 1, "down": 2, "left": 3, "right": 4}
VALID_ACTIONS = list(ACTION_MAP.keys())
_CONV_CHARS = {(1, 0): ">", (-1, 0): "<", (0, -1): "^", (0, 1): "v"}

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


class PuzzleEnvironment:
    _ACTION_MAP = {
        "reset": GameAction.RESET,
        "up": GameAction.ACTION1,
        "down": GameAction.ACTION2,
        "left": GameAction.ACTION3,
        "right": GameAction.ACTION4,
        "undo": GameAction.ACTION7,
    }

    _VALID_ACTIONS = list(_ACTION_MAP.keys())

    def __init__(self, seed: int = 0) -> None:
        self._engine = Fw91(seed=seed)
        self._total_turns = 0
        self._done = False
        self._game_won = False
        self._last_action_was_reset = False

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
        return self._build_game_state()

    def get_actions(self) -> List[str]:
        if self._done:
            return ["reset"]
        return list(self._VALID_ACTIONS)

    def step(self, action: str) -> StepResult:
        e = self._engine
        if action == "reset":
            state = self.reset()
            return StepResult(state=state, reward=0.0, done=False, info={"action": "reset"})

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
        info = {"action": action}
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
            info["reason"] = "game_complete"
            return StepResult(
                state=self._build_game_state(done=True),
                reward=level_reward,
                done=True,
                info=info,
            )

        if game_over:
            self._done = True
            info["reason"] = "game_over"
            return StepResult(
                state=self._build_game_state(done=True),
                reward=0.0,
                done=True,
                info=info,
            )

        reward = 0.0
        if e.level_index != level_before:
            reward = level_reward
            info["reason"] = "level_complete"

        return StepResult(
            state=self._build_game_state(done=False),
            reward=reward,
            done=False,
            info=info,
        )

    def _build_game_state(self, done: bool = False) -> GameState:
        valid_actions = self.get_actions() if not done else None
        return GameState(
            text_observation=self._build_text_observation(),
            image_observation=None,
            valid_actions=valid_actions,
            turn=self._total_turns,
            metadata={
                "total_levels": len(self._engine._levels),
                "level_index": self._engine.level_index,
                "levels_completed": self._engine.level_index,
                "game_over": getattr(getattr(self._engine, "_state", None), "name", "") == "GAME_OVER",
                "done": done,
                "info": {},
            },
        )

    def _build_text_observation(self) -> str:
        e = self._engine
        grid_w, grid_h = e.current_level.grid_size
        gems_left = len(e.gem_data)
        lines = [
            f"Level {e.level_index + 1}  "
            f"Moves: {e.moves_remaining}/{e.max_moves}  "
            f"Lives: {e.lives}/{MAX_LIVES}  "
            f"Shields: {e.shields_held}/2  "
            f"Gate: {'OPEN' if e.gate_is_open else 'LOCKED'}  "
            f"Gems remaining: {gems_left}"
        ]
        grid = [["." for _ in range(grid_w)] for _ in range(grid_h)]
        for s in e.current_level._sprites:
            if s.tags and "wall" in s.tags:
                if 0 <= s.x < grid_w and 0 <= s.y < grid_h:
                    grid[s.y][s.x] = "#"
        for (x, y), (cdx, cdy) in e.conveyor_map.items():
            if 0 <= x < grid_w and 0 <= y < grid_h and grid[y][x] == ".":
                grid[y][x] = _CONV_CHARS.get((cdx, cdy), "?")
        warp_labels = {}
        label_num = 1
        for (x, y), (px, py) in e.warp_map.items():
            if (x, y) not in warp_labels:
                warp_labels[(x, y)] = str(label_num)
                warp_labels[(px, py)] = str(label_num)
                label_num += 1
        for (x, y), lbl in warp_labels.items():
            if 0 <= x < grid_w and 0 <= y < grid_h and grid[y][x] == ".":
                grid[y][x] = lbl
        for x, y in e.sticky_set:
            if 0 <= x < grid_w and 0 <= y < grid_h and grid[y][x] == ".":
                grid[y][x] = "~"
        for x, y in e.poison_set:
            if (
                0 <= x < grid_w
                and 0 <= y < grid_h
                and grid[y][x] in (".", "~", ">", "<", "^", "v")
            ):
                grid[y][x] = "X"
        for (x, y), gdata in e.gem_data.items():
            if 0 <= x < grid_w and 0 <= y < grid_h:
                c = gdata["color"]
                ch = {"red": "R", "blue": "B", "green": "G"}.get(c, "?")
                if gdata["stuck"]:
                    ch = ch.lower()
                grid[y][x] = ch
        for x, y in e.spawner_positions:
            if 0 <= x < grid_w and 0 <= y < grid_h and grid[y][x] == ".":
                grid[y][x] = "M"
        for x, y in e.shield_positions:
            if 0 <= x < grid_w and 0 <= y < grid_h and grid[y][x] == ".":
                grid[y][x] = "S"
        for x, y in e.antidote_positions:
            if 0 <= x < grid_w and 0 <= y < grid_h and grid[y][x] == ".":
                grid[y][x] = "A"
        if e.gate_sprite:
            gx, gy = e.gate_sprite.x, e.gate_sprite.y
            if 0 <= gx < grid_w and 0 <= gy < grid_h:
                grid[gy][gx] = "O" if e.gate_is_open else "D"
        if e.player:
            ppx, ppy = e.player.x, e.player.y
            if 0 <= ppx < grid_w and 0 <= ppy < grid_h:
                grid[ppy][ppx] = "P"
        for row in grid:
            lines.append(" ".join(row))
        return "\n".join(lines)

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

    def is_done(self) -> bool:
        return self._done

    def close(self) -> None:
        self._engine = None


class ArcGameEnv(gym.Env):
    metadata: Dict[str, Any] = {"render_modes": ["rgb_array"], "render_fps": 5}
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

    def __init__(self, seed: int = 0, render_mode: Optional[str] = None) -> None:
        super().__init__()
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode: {render_mode}")
        self.render_mode = render_mode
        self._seed = seed
        self._env = None
        self._action_to_string = {i: a for i, a in enumerate(self.ACTION_LIST)}
        self._string_to_action = {a: i for i, a in enumerate(self.ACTION_LIST)}
        self.observation_space = spaces.Box(0, 255, (self.OBS_HEIGHT, self.OBS_WIDTH, 3), dtype=np.uint8)
        self.action_space = spaces.Discrete(len(self.ACTION_LIST))
        self._last_state = None

    def reset(self, *, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed
        self._env = PuzzleEnvironment(seed=self._seed)
        state = self._env.reset()
        self._last_state = state
        obs = self._get_obs()
        info = self._build_info(state)
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action_str = self._action_to_string.get(action, "reset")
        result = self._env.step(action_str)
        self._last_state = result.state
        obs = self._get_obs()
        info = self._build_info(result.state, step_info=result.info)
        truncated = False
        return obs, result.reward, result.done, truncated, info

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode == "rgb_array":
            return self._get_obs()
        return None

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
        self._env = None

    def action_mask(self) -> np.ndarray:
        mask = np.zeros(len(self.ACTION_LIST), dtype=bool)
        if self._env is None:
            return mask
        valid = self._env.get_actions()
        for a in valid:
            if a in self._string_to_action:
                mask[self._string_to_action[a]] = True
        return mask

    def _get_obs(self) -> np.ndarray:
        if self._env is None:
            return np.zeros((self.OBS_HEIGHT, self.OBS_WIDTH, 3), dtype=np.uint8)
        img = self._env.render(mode="rgb_array")
        if img.shape[0] != self.OBS_HEIGHT or img.shape[1] != self.OBS_WIDTH:
            img = self._resize_nearest(img, self.OBS_HEIGHT, self.OBS_WIDTH)
        return img

    @staticmethod
    def _resize_nearest(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        h, w = img.shape[:2]
        ys = (np.arange(target_h) * h / target_h).astype(int)
        xs = (np.arange(target_w) * w / target_w).astype(int)
        ys = np.clip(ys, 0, h - 1)
        xs = np.clip(xs, 0, w - 1)
        return img[np.ix_(ys, xs)]

    def _build_info(self, state, step_info=None) -> Dict[str, Any]:
        info = {
            "text_observation": state.text_observation if state else "",
            "valid_actions": state.valid_actions if state else [],
            "turn": state.turn if state else 0,
            "game_metadata": state.metadata if state else {},
        }
        if step_info:
            info["step_info"] = step_info
        return info
