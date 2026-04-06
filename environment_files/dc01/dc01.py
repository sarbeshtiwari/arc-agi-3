import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from arcengine import (
    ActionInput,
    ARCBaseGame,
    Camera,
    GameAction,
    Level,
    RenderableUserDisplay,
    Sprite,
)
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

BACKGROUND_COLOR = 0
PADDING_COLOR = 4
MAX_LIVES = 3
TILE = 5

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
ORIENT_NAMES = ["up", "right", "down", "left"]

DIR_DELTA = {
    UP: (0, -1),
    RIGHT: (1, 0),
    DOWN: (0, 1),
    LEFT: (-1, 0),
}

FORK_SIDES = {
    UP: (LEFT, RIGHT),
    RIGHT: (UP, DOWN),
    DOWN: (RIGHT, LEFT),
    LEFT: (DOWN, UP),
}

NORMAL = "normal"
FORK = "fork"


sprites = {
    "player": Sprite(
        pixels=[
            [3, 3, -1, 3, 3],
            [3, -1, -1, -1, 3],
            [-1, -1, -1, -1, -1],
            [3, -1, -1, -1, 3],
            [3, 3, -1, 3, 3],
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
    "ndom_up": Sprite(
        pixels=[
            [0, 7, 7, 7, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ],
        name="ndom_up",
        visible=True,
        collidable=False,
        tags=["domino", "normal_domino"],
        layer=1,
    ),
    "ndom_right": Sprite(
        pixels=[
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 7],
            [1, 1, 1, 1, 7],
            [1, 1, 1, 1, 7],
            [0, 0, 0, 0, 0],
        ],
        name="ndom_right",
        visible=True,
        collidable=False,
        tags=["domino", "normal_domino"],
        layer=1,
    ),
    "ndom_down": Sprite(
        pixels=[
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 7, 7, 7, 0],
        ],
        name="ndom_down",
        visible=True,
        collidable=False,
        tags=["domino", "normal_domino"],
        layer=1,
    ),
    "ndom_left": Sprite(
        pixels=[
            [0, 0, 0, 0, 0],
            [7, 1, 1, 1, 1],
            [7, 1, 1, 1, 1],
            [7, 1, 1, 1, 1],
            [0, 0, 0, 0, 0],
        ],
        name="ndom_left",
        visible=True,
        collidable=False,
        tags=["domino", "normal_domino"],
        layer=1,
    ),
    "sdom_up": Sprite(
        pixels=[
            [0, 4, 4, 4, 0],
            [0, 7, 7, 7, 0],
            [0, 7, 7, 7, 0],
            [0, 7, 7, 7, 0],
            [0, 0, 0, 0, 0],
        ],
        name="sdom_up",
        visible=True,
        collidable=False,
        tags=["domino", "starter_domino"],
        layer=1,
    ),
    "sdom_right": Sprite(
        pixels=[
            [0, 0, 0, 0, 0],
            [7, 7, 7, 7, 4],
            [7, 7, 7, 7, 4],
            [7, 7, 7, 7, 4],
            [0, 0, 0, 0, 0],
        ],
        name="sdom_right",
        visible=True,
        collidable=False,
        tags=["domino", "starter_domino"],
        layer=1,
    ),
    "sdom_down": Sprite(
        pixels=[
            [0, 0, 0, 0, 0],
            [0, 7, 7, 7, 0],
            [0, 7, 7, 7, 0],
            [0, 7, 7, 7, 0],
            [0, 4, 4, 4, 0],
        ],
        name="sdom_down",
        visible=True,
        collidable=False,
        tags=["domino", "starter_domino"],
        layer=1,
    ),
    "sdom_left": Sprite(
        pixels=[
            [0, 0, 0, 0, 0],
            [4, 7, 7, 7, 7],
            [4, 7, 7, 7, 7],
            [4, 7, 7, 7, 7],
            [0, 0, 0, 0, 0],
        ],
        name="sdom_left",
        visible=True,
        collidable=False,
        tags=["domino", "starter_domino"],
        layer=1,
    ),
    "fdom_up": Sprite(
        pixels=[
            [7, 7, 7, 7, 7],
            [7, 8, 8, 8, 7],
            [7, 8, 8, 8, 7],
            [7, 8, 8, 8, 7],
            [0, 0, 0, 0, 0],
        ],
        name="fdom_up",
        visible=True,
        collidable=False,
        tags=["domino", "fork_domino"],
        layer=1,
    ),
    "fdom_right": Sprite(
        pixels=[
            [0, 7, 7, 7, 7],
            [0, 8, 8, 8, 7],
            [0, 8, 8, 8, 7],
            [0, 8, 8, 8, 7],
            [0, 7, 7, 7, 7],
        ],
        name="fdom_right",
        visible=True,
        collidable=False,
        tags=["domino", "fork_domino"],
        layer=1,
    ),
    "fdom_down": Sprite(
        pixels=[
            [0, 0, 0, 0, 0],
            [7, 8, 8, 8, 7],
            [7, 8, 8, 8, 7],
            [7, 8, 8, 8, 7],
            [7, 7, 7, 7, 7],
        ],
        name="fdom_down",
        visible=True,
        collidable=False,
        tags=["domino", "fork_domino"],
        layer=1,
    ),
    "fdom_left": Sprite(
        pixels=[
            [7, 7, 7, 7, 0],
            [7, 8, 8, 8, 0],
            [7, 8, 8, 8, 0],
            [7, 8, 8, 8, 0],
            [7, 7, 7, 7, 0],
        ],
        name="fdom_left",
        visible=True,
        collidable=False,
        tags=["domino", "fork_domino"],
        layer=1,
    ),
    "rail_up": Sprite(
        pixels=[
            [-1, 4, 4, 4, -1],
            [-1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1],
        ],
        name="rail_up",
        visible=True,
        collidable=False,
        tags=["rail"],
        layer=2,
    ),
    "rail_right": Sprite(
        pixels=[
            [-1, -1, -1, -1, -1],
            [-1, -1, -1, -1, 4],
            [-1, -1, -1, -1, 4],
            [-1, -1, -1, -1, 4],
            [-1, -1, -1, -1, -1],
        ],
        name="rail_right",
        visible=True,
        collidable=False,
        tags=["rail"],
        layer=2,
    ),
    "rail_down": Sprite(
        pixels=[
            [-1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1],
            [-1, 4, 4, 4, -1],
        ],
        name="rail_down",
        visible=True,
        collidable=False,
        tags=["rail"],
        layer=2,
    ),
    "rail_left": Sprite(
        pixels=[
            [-1, -1, -1, -1, -1],
            [4, -1, -1, -1, -1],
            [4, -1, -1, -1, -1],
            [4, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1],
        ],
        name="rail_left",
        visible=True,
        collidable=False,
        tags=["rail"],
        layer=2,
    ),
    "fallen": Sprite(
        pixels=[
            [0, 0, 0, 0, 0],
            [0, 5, 5, 5, 0],
            [0, 5, 9, 5, 0],
            [0, 5, 5, 5, 0],
            [0, 0, 0, 0, 0],
        ],
        name="fallen",
        visible=True,
        collidable=False,
        tags=["fallen_domino"],
        layer=1,
    ),
    "launch": Sprite(
        pixels=[
            [4, 4, 4, 4, 4],
            [4, 7, 7, 7, 4],
            [4, 7, 4, 7, 4],
            [4, 7, 7, 7, 4],
            [4, 4, 4, 4, 4],
        ],
        name="launch",
        visible=True,
        collidable=False,
        tags=["launch_switch"],
        layer=1,
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
    "flash": Sprite(
        pixels=[[2]],
        name="flash",
        visible=False,
        collidable=False,
        tags=["flash"],
        layer=10,
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


def _domino_sprite_key(dtype, orientation, is_starter=False):
    if is_starter:
        return "sdom_" + ORIENT_NAMES[orientation]
    elif dtype == FORK:
        return "fdom_" + ORIENT_NAMES[orientation]
    else:
        return "ndom_" + ORIENT_NAMES[orientation]


def _get_hit_directions(orientation, dtype):
    directions = [orientation]
    if dtype == FORK:
        left_dir, right_dir = FORK_SIDES[orientation]
        directions.append(left_dir)
        directions.append(right_dir)
    return directions


_l1_sprites = _border(9, 8) + [
    sprites["launch"].clone().set_position(3 * TILE, 2 * TILE),
    sprites["player"].clone().set_position(1 * TILE, 6 * TILE),
    sprites["gate_locked"].clone().set_position(7 * TILE, 6 * TILE),
]

level1 = Level(
    sprites=_l1_sprites,
    grid_size=(9 * TILE, 8 * TILE),
    data={
        "domino_configs": [
            [3, 3, RIGHT, NORMAL, True, RIGHT],
            [4, 3, UP, NORMAL, False, RIGHT],
            [5, 3, RIGHT, NORMAL, False, RIGHT],
        ],
        "launch_pos": [3, 2],
        "starter_pos": [3, 3],
        "gate_pos": [7, 6],
        "max_moves": 108,
    },
)

_l2_sprites = _border(10, 9) + [
    sprites["launch"].clone().set_position(2 * TILE, 1 * TILE),
    sprites["player"].clone().set_position(1 * TILE, 7 * TILE),
    sprites["gate_locked"].clone().set_position(8 * TILE, 7 * TILE),
]

level2 = Level(
    sprites=_l2_sprites,
    grid_size=(10 * TILE, 9 * TILE),
    data={
        "domino_configs": [
            [2, 2, RIGHT, NORMAL, True, RIGHT],
            [3, 2, DOWN, NORMAL, False, DOWN],
            [3, 3, UP, NORMAL, False, RIGHT],
            [4, 3, RIGHT, NORMAL, False, DOWN],
            [4, 4, RIGHT, NORMAL, False, RIGHT],
            [5, 4, UP, NORMAL, False, RIGHT],
        ],
        "launch_pos": [2, 1],
        "starter_pos": [2, 2],
        "gate_pos": [8, 7],
        "max_moves": 174,
    },
)

_l3_sprites = _border(11, 10) + [
    sprites["launch"].clone().set_position(4 * TILE, 2 * TILE),
    sprites["player"].clone().set_position(1 * TILE, 8 * TILE),
    sprites["gate_locked"].clone().set_position(9 * TILE, 8 * TILE),
]

level3 = Level(
    sprites=_l3_sprites,
    grid_size=(11 * TILE, 10 * TILE),
    data={
        "domino_configs": [
            [4, 3, DOWN, NORMAL, True, DOWN],
            [4, 4, RIGHT, FORK, False, DOWN],
            [4, 5, RIGHT, NORMAL, False, DOWN],
            [5, 4, RIGHT, NORMAL, False, RIGHT],
        ],
        "launch_pos": [4, 2],
        "starter_pos": [4, 3],
        "gate_pos": [9, 8],
        "max_moves": 138,
    },
)

_l4_sprites = _border(12, 12) + [
    sprites["launch"].clone().set_position(2 * TILE, 2 * TILE),
    sprites["player"].clone().set_position(1 * TILE, 10 * TILE),
    sprites["gate_locked"].clone().set_position(10 * TILE, 10 * TILE),
]

level4 = Level(
    sprites=_l4_sprites,
    grid_size=(12 * TILE, 12 * TILE),
    data={
        "domino_configs": [
            [2, 3, DOWN, NORMAL, True, DOWN],
            [2, 4, DOWN, NORMAL, False, DOWN],
            [2, 5, UP, FORK, False, RIGHT],
            [2, 6, DOWN, NORMAL, False, DOWN],
            [3, 5, UP, NORMAL, False, RIGHT],
            [4, 5, LEFT, NORMAL, False, UP],
            [4, 4, LEFT, FORK, False, UP],
            [4, 3, UP, NORMAL, False, UP],
            [5, 4, UP, NORMAL, False, RIGHT],
            [6, 4, RIGHT, NORMAL, False, RIGHT],
        ],
        "launch_pos": [2, 2],
        "starter_pos": [2, 3],
        "gate_pos": [10, 10],
        "max_moves": 222,
    },
)

GAME_LEVELS = [level1, level2, level3, level4]


class DominoHUD(RenderableUserDisplay):
    def __init__(self, game):
        self._game = game
        self._moves_remaining = 0
        self._max_moves = 0
        self._lives = MAX_LIVES
        self._gate_open = False

    def update(self, moves_remaining=0, max_moves=0, lives=MAX_LIVES, gate_open=False):
        self._moves_remaining = moves_remaining
        self._max_moves = max_moves
        self._lives = lives
        self._gate_open = gate_open

    def render_interface(self, frame):
        height, width = frame.shape[:2]

        gate_color = 3 if self._gate_open else 2
        for i in range(min(3, width)):
            frame[0, i] = gate_color

        for i in range(MAX_LIVES):
            px = width - 2 - i * 4
            color = 2 if i < self._lives else 5
            for offset in range(3):
                col = px - offset
                if 0 <= col < width:
                    frame[0, col] = color

        if self._max_moves > 0:
            bar_row = height - 1
            ratio = self._moves_remaining / self._max_moves
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


class Dc01(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)
        self._hud = DominoHUD(self)
        self._player = None
        self._lives = MAX_LIVES
        self._moves_remaining = 0
        self._max_moves = 0
        self._history: List[Dict] = []
        self._game_over = False
        self._consecutive_resets = 0

        self._dominoes: Dict = {}
        self._domino_sprites: Dict = {}
        self._initial_dominoes: Dict = {}

        self._launch_pos = None
        self._starter_pos = None
        self._cascade_done = False

        self._gate_sprite = None
        self._gate_open = False

        self._start_x = 0
        self._start_y = 0
        self._level_start_positions: Dict = {}

        self._flash_sprite = None
        self._flash_active = False

        cam_w = min(max(lev.grid_size[0] for lev in GAME_LEVELS), 64)
        cam_h = min(max(lev.grid_size[1] for lev in GAME_LEVELS), 64)
        camera = Camera(
            0, 0, cam_w, cam_h, BACKGROUND_COLOR, PADDING_COLOR, [self._hud]
        )

        super().__init__(
            "dc01",
            GAME_LEVELS,
            camera,
            available_actions=[0, 1, 2, 3, 4, 5, 7],
        )

    def handle_reset(self) -> None:
        self._lives = MAX_LIVES
        self._game_over = False
        self._history = []
        self._consecutive_resets += 1
        if self._consecutive_resets >= 2:
            self._consecutive_resets = 0
            self.set_level(0)
        else:
            self.set_level(self.level_index)

    def on_set_level(self, level: Level) -> None:
        self._lives = MAX_LIVES
        self._history = []
        self._game_over = False

        self._player = self.current_level.get_sprites_by_tag("player")[0]
        lvl_idx = self.level_index
        if lvl_idx not in self._level_start_positions:
            self._level_start_positions[lvl_idx] = (self._player.x, self._player.y)
        sx, sy = self._level_start_positions[lvl_idx]
        self._player.set_position(sx, sy)
        self._start_x = sx
        self._start_y = sy

        self._max_moves = self.current_level.get_data("max_moves") or 50
        self._moves_remaining = self._max_moves
        self._gate_open = False
        self._cascade_done = False

        self._init_gate()
        self._init_dominoes()
        self._init_rails()
        self._init_flash()

        self._hud.update(
            self._moves_remaining, self._max_moves, self._lives, self._gate_open
        )

    def _init_gate(self):
        gates = self.current_level.get_sprites_by_tag("gate")
        if gates:
            self._gate_sprite = gates[0]
        else:
            open_gates = self.current_level.get_sprites_by_tag("gate_open")
            if open_gates:
                gx, gy = open_gates[0].x, open_gates[0].y
                self.current_level.remove_sprite(open_gates[0])
                new_gate = sprites["gate_locked"].clone().set_position(gx, gy)
                self.current_level.add_sprite(new_gate)
                self._gate_sprite = new_gate
            else:
                self._gate_sprite = None

    def _init_dominoes(self):
        for tag in (
            "domino",
            "normal_domino",
            "starter_domino",
            "fork_domino",
            "fallen_domino",
            "rail",
        ):
            for s in list(self.current_level.get_sprites_by_tag(tag)):
                self.current_level.remove_sprite(s)

        self._dominoes = {}
        self._domino_sprites = {}
        self._initial_dominoes = {}

        self._launch_pos = tuple(self.current_level.get_data("launch_pos"))
        self._starter_pos = tuple(self.current_level.get_data("starter_pos"))

        domino_configs = self.current_level.get_data("domino_configs") or []
        for cfg in domino_configs:
            tc, tr, orient, dtype, is_starter, rail_dir = cfg
            px, py = tc * TILE, tr * TILE

            sprite_key = _domino_sprite_key(dtype, orient, is_starter)
            sprite = sprites[sprite_key].clone().set_position(px, py)
            self.current_level.add_sprite(sprite)

            self._domino_sprites[(tc, tr)] = sprite
            self._dominoes[(tc, tr)] = {
                "orientation": orient,
                "dtype": dtype,
                "fallen": False,
                "is_starter": is_starter,
                "rail_dir": rail_dir,
            }
            self._initial_dominoes[(tc, tr)] = {
                "orientation": orient,
                "dtype": dtype,
                "fallen": False,
                "is_starter": is_starter,
                "rail_dir": rail_dir,
            }

    def _init_rails(self):
        all_positions = set(self._dominoes.keys())
        rail_key_map = {
            UP: "rail_up",
            RIGHT: "rail_right",
            DOWN: "rail_down",
            LEFT: "rail_left",
        }
        for pos, dom in self._dominoes.items():
            if dom["is_starter"]:
                continue
            hit_dirs = _get_hit_directions(dom["rail_dir"], dom["dtype"])
            is_leaf = True
            for d in hit_dirs:
                dc, dr = DIR_DELTA[d]
                neighbor = (pos[0] + dc, pos[1] + dr)
                if neighbor in all_positions:
                    is_leaf = False
                    break
            if is_leaf:
                rk = rail_key_map[dom["rail_dir"]]
                dc, dr = DIR_DELTA[dom["rail_dir"]]
                nx, ny = (pos[0] + dc) * TILE, (pos[1] + dr) * TILE
                rail_sprite = sprites[rk].clone().set_position(nx, ny)
                self.current_level.add_sprite(rail_sprite)

    def _init_flash(self):
        flash_list = self.current_level.get_sprites_by_tag("flash")
        if flash_list:
            self._flash_sprite = flash_list[0]
        else:
            self._flash_sprite = sprites["flash"].clone()
            self.current_level.add_sprite(self._flash_sprite)
        self._flash_sprite.set_visible(False)
        self._flash_active = False

    def step(self) -> None:
        if self._flash_active:
            if self._flash_sprite:
                self._flash_sprite.set_visible(False)
            self._flash_active = False
            self.complete_action()
            return

        if self.action.id == GameAction.RESET:
            self.complete_action()
            return

        self._consecutive_resets = 0

        if self.action.id == GameAction.ACTION7:
            if self._history:
                self._restore_from_undo()
                self._moves_remaining -= 1
                self._hud.update(
                    self._moves_remaining,
                    self._max_moves,
                    self._lives,
                    self._gate_open,
                )
                if self._moves_remaining <= 0:
                    self._handle_death()
            self.complete_action()
            return

        self._save_state()

        if self.action.id in (
            GameAction.ACTION1,
            GameAction.ACTION2,
            GameAction.ACTION3,
            GameAction.ACTION4,
        ):
            dx, dy = 0, 0
            if self.action.id == GameAction.ACTION1:
                dy = -TILE
            elif self.action.id == GameAction.ACTION2:
                dy = TILE
            elif self.action.id == GameAction.ACTION3:
                dx = -TILE
            elif self.action.id == GameAction.ACTION4:
                dx = TILE

            self._moves_remaining -= 1
            self._hud.update(
                self._moves_remaining,
                self._max_moves,
                self._lives,
                self._gate_open,
            )

            if self._moves_remaining <= 0:
                self._handle_death()
                self.complete_action()
                return

            new_x = self._player.x + dx
            new_y = self._player.y + dy

            if not self._check_collision(new_x, new_y):
                self._player.set_position(new_x, new_y)

                exit_sprite = self._get_sprite_at(new_x, new_y, "exit")
                if exit_sprite and self._gate_open:
                    self.next_level()
                    self.complete_action()
                    return

        elif self.action.id == GameAction.ACTION5:
            self._moves_remaining -= 1
            self._hud.update(
                self._moves_remaining,
                self._max_moves,
                self._lives,
                self._gate_open,
            )

            if self._moves_remaining <= 0:
                self._handle_death()
                self.complete_action()
                return

            tile_x, tile_y = self._tile_coord(self._player.x, self._player.y)

            if (tile_x, tile_y) == self._launch_pos and not self._cascade_done:
                all_fell = self._run_cascade()
                self._cascade_done = True

                if all_fell:
                    self._open_gate()
                else:
                    self._auto_reset_after_failed_launch()

            elif (
                (tile_x, tile_y) in self._dominoes
                and not self._dominoes[(tile_x, tile_y)]["fallen"]
                and not self._cascade_done
            ):
                self._rotate_domino(tile_x, tile_y)

        self.complete_action()

    def _check_collision(self, x, y):
        for s in self.current_level.get_sprites():
            if s.x <= x < s.x + s.width and s.y <= y < s.y + s.height:
                if s.tags and "solid" in s.tags:
                    return True
        return False

    def _get_sprite_at(self, x, y, tag=None):
        for s in self.current_level.get_sprites():
            if s.x <= x < s.x + s.width and s.y <= y < s.y + s.height:
                if tag is None or (s.tags and tag in s.tags):
                    return s
        return None

    def _tile_coord(self, px_x, px_y):
        return (px_x // TILE, px_y // TILE)

    def _replace_domino_sprite(self, tc, tr):
        if (tc, tr) not in self._dominoes:
            return
        dom = self._dominoes[(tc, tr)]
        old_sprite = self._domino_sprites.get((tc, tr))
        px, py = tc * TILE, tr * TILE

        if old_sprite:
            self.current_level.remove_sprite(old_sprite)

        if dom["fallen"]:
            new_sprite = sprites["fallen"].clone().set_position(px, py)
        else:
            sprite_key = _domino_sprite_key(
                dom["dtype"], dom["orientation"], dom["is_starter"]
            )
            new_sprite = sprites[sprite_key].clone().set_position(px, py)

        self.current_level.add_sprite(new_sprite)
        self._domino_sprites[(tc, tr)] = new_sprite

    def _rotate_domino(self, tc, tr):
        if (tc, tr) not in self._dominoes:
            return False
        dom = self._dominoes[(tc, tr)]
        if dom["fallen"]:
            return False
        dom["orientation"] = (dom["orientation"] + 1) % 4
        self._replace_domino_sprite(tc, tr)
        return True

    def _run_cascade(self):
        starter = self._starter_pos
        if starter not in self._dominoes:
            return False

        queue = [starter]
        queued = {starter}

        while queue:
            pos = queue.pop(0)
            tc, tr = pos
            dom = self._dominoes.get(pos)
            if dom is None or dom["fallen"]:
                continue

            dom["fallen"] = True
            self._replace_domino_sprite(tc, tr)

            hit_dirs = _get_hit_directions(dom["orientation"], dom["dtype"])
            for direction in hit_dirs:
                dc, dr = DIR_DELTA[direction]
                neighbor = (tc + dc, tr + dr)
                if neighbor in self._dominoes:
                    ndom = self._dominoes[neighbor]
                    if not ndom["fallen"] and neighbor not in queued:
                        queue.append(neighbor)
                        queued.add(neighbor)

        if not all(d["fallen"] for d in self._dominoes.values()):
            return False

        all_positions = set(self._dominoes.keys())
        for pos, dom in self._dominoes.items():
            if dom["is_starter"]:
                continue
            hit_dirs = _get_hit_directions(dom["rail_dir"], dom["dtype"])
            is_leaf = True
            for d in hit_dirs:
                dc, dr = DIR_DELTA[d]
                neighbor = (pos[0] + dc, pos[1] + dr)
                if neighbor in all_positions:
                    is_leaf = False
                    break
            if is_leaf and dom["orientation"] != dom["rail_dir"]:
                return False

        return True

    def _open_gate(self):
        if self._gate_open:
            return
        self._gate_open = True
        if self._gate_sprite:
            gx, gy = self._gate_sprite.x, self._gate_sprite.y
            self.current_level.remove_sprite(self._gate_sprite)
            new_gate = sprites["gate_open"].clone().set_position(gx, gy)
            self.current_level.add_sprite(new_gate)
            self._gate_sprite = new_gate
        self._hud.update(
            self._moves_remaining, self._max_moves, self._lives, True
        )

    def _close_gate(self):
        if not self._gate_open:
            return
        self._gate_open = False
        if self._gate_sprite:
            gx, gy = self._gate_sprite.x, self._gate_sprite.y
            self.current_level.remove_sprite(self._gate_sprite)
            new_gate = sprites["gate_locked"].clone().set_position(gx, gy)
            self.current_level.add_sprite(new_gate)
            self._gate_sprite = new_gate
        self._hud.update(
            self._moves_remaining, self._max_moves, self._lives, False
        )


    def _handle_death(self) -> None:
        self._lives -= 1
        if self._lives <= 0:
            self.lose()
        else:
            remaining = self._lives
            self.set_level(self.level_index)
            self._lives = remaining
            self._hud.update(
                self._moves_remaining, self._max_moves, self._lives, self._gate_open
            )

    def _auto_reset_after_failed_launch(self):
        for pos, dom in self._dominoes.items():
            if dom["fallen"]:
                dom["fallen"] = False
                self._replace_domino_sprite(*pos)
        self._cascade_done = False

    def _save_state(self) -> None:
        domino_snap = {}
        for pos, dom in self._dominoes.items():
            domino_snap[pos] = {
                "orientation": dom["orientation"],
                "fallen": dom["fallen"],
            }
        self._history.append(
            {
                "player_x": self._player.x,
                "player_y": self._player.y,
                "moves_remaining": self._moves_remaining,
                "domino_states": domino_snap,
                "cascade_done": self._cascade_done,
                "gate_open": self._gate_open,
            }
        )

    def _restore_from_undo(self) -> None:
        if not self._history:
            return
        state = self._history.pop()
        self._player.set_position(state["player_x"], state["player_y"])
        self._moves_remaining = state["moves_remaining"]
        self._cascade_done = state["cascade_done"]

        for pos, snap in state["domino_states"].items():
            if pos in self._dominoes:
                self._dominoes[pos]["orientation"] = snap["orientation"]
                self._dominoes[pos]["fallen"] = snap["fallen"]
                self._replace_domino_sprite(*pos)

        if state["gate_open"] != self._gate_open:
            if state["gate_open"]:
                self._open_gate()
            else:
                self._close_gate()


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

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self._engine = Dc01(seed=seed)
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

    def is_done(self) -> bool:
        return self._done

    def step(self, action: str) -> StepResult:
        e = self._engine

        if action == "reset":
            state = self.reset()
            return StepResult(
                state=state, reward=0.0, done=False, info={"action": "reset"}
            )

        if self._done:
            return StepResult(
                state=self._build_game_state(done=True),
                reward=0.0,
                done=True,
                info={"action": action},
            )

        if action not in self._ACTION_MAP:
            raise ValueError(
                f"Invalid action '{action}'. "
                f"Must be one of {list(self._ACTION_MAP.keys())}"
            )

        self._last_action_was_reset = False
        self._total_turns += 1

        game_action = self._ACTION_MAP[action]
        info: Dict = {"action": action}

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

    def _build_text_observation(self) -> str:
        e = self._engine
        grid_w = e.current_level.grid_size[0] // TILE
        grid_h = e.current_level.grid_size[1] // TILE
        lines = [
            f"Level {e.level_index + 1}  "
            f"Moves: {e._moves_remaining}/{e._max_moves}  "
            f"Lives: {e._lives}/{MAX_LIVES}  "
            f"Gate: {'OPEN' if e._gate_open else 'LOCKED'}"
        ]

        grid = [[" ." for _ in range(grid_w)] for _ in range(grid_h)]

        for s in e.current_level.get_sprites():
            if s.tags and "wall" in s.tags:
                tc, tr = s.x // TILE, s.y // TILE
                if 0 <= tc < grid_w and 0 <= tr < grid_h:
                    grid[tr][tc] = " W"

        if e._launch_pos:
            lc, lr = e._launch_pos
            if 0 <= lc < grid_w and 0 <= lr < grid_h:
                grid[lr][lc] = " L"

        orient_chars = ["^", ">", "v", "<"]

        all_dom_positions = set(e._dominoes.keys())
        for (tc, tr), dom in e._dominoes.items():
            if not dom["is_starter"]:
                hit_dirs = _get_hit_directions(dom["rail_dir"], dom["dtype"])
                is_leaf = True
                for d in hit_dirs:
                    dc, dr = DIR_DELTA[d]
                    neighbor = (tc + dc, tr + dr)
                    if neighbor in all_dom_positions:
                        is_leaf = False
                        break
                if is_leaf:
                    dc, dr = DIR_DELTA[dom["rail_dir"]]
                    rc, rr = tc + dc, tr + dr
                    if 0 <= rc < grid_w and 0 <= rr < grid_h:
                        grid[rr][rc] = "r" + orient_chars[dom["rail_dir"]]

        for (tc, tr), dom in e._dominoes.items():
            if 0 <= tc < grid_w and 0 <= tr < grid_h:
                if dom["fallen"]:
                    grid[tr][tc] = " x"
                elif dom["is_starter"]:
                    grid[tr][tc] = "S" + orient_chars[dom["orientation"]]
                elif dom["dtype"] == FORK:
                    grid[tr][tc] = "F" + orient_chars[dom["orientation"]]
                else:
                    grid[tr][tc] = " " + orient_chars[dom["orientation"]]

        if e._gate_sprite:
            gtc, gtr = e._gate_sprite.x // TILE, e._gate_sprite.y // TILE
            if 0 <= gtc < grid_w and 0 <= gtr < grid_h:
                grid[gtr][gtc] = " O" if e._gate_open else " X"

        if e._player:
            ptc, ptr = e._player.x // TILE, e._player.y // TILE
            if 0 <= ptc < grid_w and 0 <= ptr < grid_h:
                grid[ptr][ptc] = " P"

        for row in grid:
            lines.append(" ".join(row))
        return "\n".join(lines)

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
                "levels_completed": getattr(self._engine, "_score", 0),
                "game_over": getattr(
                    getattr(self._engine, "_state", None), "name", ""
                )
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

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
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
