from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import struct
import zlib
import random

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


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
FUCHSIA = 6
ORANGE = 7
LIGHT_BLUE = 8
MAROON = 9

BACKGROUND_COLOR = BLACK
PADDING_COLOR = BLACK

MAX_LIVES = 3

_SPRITES = {
    "player": Sprite(
        pixels=[
            [-1, RED, RED, RED, -1],
            [RED, RED, RED, RED, RED],
            [RED, ORANGE, RED, ORANGE, RED],
            [RED, RED, RED, RED, RED],
            [-1, RED, -1, RED, -1],
        ],
        name="player",
        visible=True,
        collidable=True,
        tags=["player"],
        layer=2,
    ),
    "wall": Sprite(
        pixels=[
            [GREY, GREY, GREY, GREY, GREY],
            [GREY, GREY, GREY, GREY, GREY],
            [GREY, GREY, GREY, GREY, GREY],
            [GREY, GREY, GREY, GREY, GREY],
            [GREY, GREY, GREY, GREY, GREY],
        ],
        name="wall",
        visible=True,
        collidable=True,
        tags=["wall", "solid"],
        layer=0,
    ),
    "gem": Sprite(
        pixels=[
            [-1, -1, YELLOW, -1, -1],
            [-1, YELLOW, YELLOW, YELLOW, -1],
            [YELLOW, YELLOW, ORANGE, YELLOW, YELLOW],
            [-1, YELLOW, YELLOW, YELLOW, -1],
            [-1, -1, YELLOW, -1, -1],
        ],
        name="gem",
        visible=True,
        collidable=False,
        tags=["collectible", "gem"],
        layer=1,
    ),
    "key": Sprite(
        pixels=[
            [-1, -1, GREEN, GREEN, -1],
            [-1, GREEN, -1, -1, GREEN],
            [GREEN, GREEN, GREEN, GREEN, GREEN],
            [-1, -1, GREEN, -1, -1],
            [-1, -1, GREEN, -1, -1],
        ],
        name="key",
        visible=True,
        collidable=False,
        tags=["collectible", "key"],
        layer=1,
    ),
    "door": Sprite(
        pixels=[
            [FUCHSIA, FUCHSIA, FUCHSIA, FUCHSIA, FUCHSIA],
            [FUCHSIA, GREEN, FUCHSIA, GREEN, FUCHSIA],
            [FUCHSIA, FUCHSIA, FUCHSIA, FUCHSIA, FUCHSIA],
            [FUCHSIA, GREEN, FUCHSIA, GREEN, FUCHSIA],
            [FUCHSIA, FUCHSIA, FUCHSIA, FUCHSIA, FUCHSIA],
        ],
        name="door",
        visible=True,
        collidable=True,
        tags=["door", "solid"],
        layer=0,
    ),
    "exit": Sprite(
        pixels=[
            [BLUE, BLUE, BLUE, BLUE, BLUE],
            [BLUE, ORANGE, ORANGE, ORANGE, BLUE],
            [BLUE, ORANGE, BLUE, ORANGE, BLUE],
            [BLUE, ORANGE, ORANGE, ORANGE, BLUE],
            [BLUE, BLUE, BLUE, BLUE, BLUE],
        ],
        name="exit",
        visible=True,
        collidable=False,
        tags=["exit"],
        layer=1,
    ),
    "floor": Sprite(
        pixels=[
            [BLACK, BLACK, BLACK, BLACK, BLACK],
            [BLACK, BLACK, BLACK, BLACK, BLACK],
            [BLACK, BLACK, BLACK, BLACK, BLACK],
            [BLACK, BLACK, BLACK, BLACK, BLACK],
            [BLACK, BLACK, BLACK, BLACK, BLACK],
        ],
        name="floor",
        visible=True,
        collidable=False,
        tags=[],
        layer=-1,
    ),
    "spike": Sprite(
        pixels=[
            [BLACK, -1, BLACK, -1, BLACK],
            [-1, RED, RED, RED, -1],
            [BLACK, RED, RED, RED, BLACK],
            [-1, RED, RED, RED, -1],
            [BLACK, BLACK, BLACK, BLACK, BLACK],
        ],
        name="spike",
        visible=True,
        collidable=True,
        tags=["spike", "solid", "hazard"],
        layer=0,
    ),
    "teleporter": Sprite(
        pixels=[
            [BLUE, BLUE, BLUE, BLUE, BLUE],
            [BLUE, FUCHSIA, FUCHSIA, FUCHSIA, BLUE],
            [BLUE, FUCHSIA, ORANGE, FUCHSIA, BLUE],
            [BLUE, FUCHSIA, FUCHSIA, FUCHSIA, BLUE],
            [BLUE, BLUE, BLUE, BLUE, BLUE],
        ],
        name="teleporter",
        visible=True,
        collidable=False,
        tags=["teleporter"],
        layer=1,
    ),
    "mine": Sprite(
        pixels=[
            [RED, -1, -1, -1, RED],
            [-1, RED, -1, RED, -1],
            [-1, -1, RED, -1, -1],
            [-1, RED, -1, RED, -1],
            [RED, -1, -1, -1, RED],
        ],
        name="mine",
        visible=True,
        collidable=False,
        tags=["mine"],
        layer=1,
    ),
    "pink_block": Sprite(
        pixels=[
            [FUCHSIA, FUCHSIA, FUCHSIA, FUCHSIA, FUCHSIA],
            [FUCHSIA, FUCHSIA, FUCHSIA, FUCHSIA, FUCHSIA],
            [FUCHSIA, FUCHSIA, FUCHSIA, FUCHSIA, FUCHSIA],
            [FUCHSIA, FUCHSIA, FUCHSIA, FUCHSIA, FUCHSIA],
            [FUCHSIA, FUCHSIA, FUCHSIA, FUCHSIA, FUCHSIA],
        ],
        name="pink_block",
        visible=True,
        collidable=True,
        tags=["pink_block", "solid"],
        layer=0,
    ),
}

_LEVEL_DATA = [
    {
        "total_gems": 6,
        "required_gems": 6,
        "max_moves": 250,
        "walls": [
            (0, 0),
            (5, 0),
            (10, 0),
            (15, 0),
            (20, 0),
            (25, 0),
            (30, 0),
            (35, 0),
            (40, 0),
            (45, 0),
            (50, 0),
            (55, 0),
            (0, 5),
            (0, 10),
            (0, 15),
            (0, 20),
            (0, 25),
            (0, 30),
            (0, 35),
            (0, 40),
            (0, 45),
            (0, 50),
            (0, 55),
            (55, 5),
            (55, 10),
            (55, 15),
            (55, 20),
            (55, 25),
            (55, 30),
            (55, 35),
            (55, 40),
            (55, 45),
            (55, 50),
            (55, 55),
            (5, 55),
            (10, 55),
            (15, 55),
            (20, 55),
            (25, 55),
            (30, 55),
            (35, 55),
            (40, 55),
            (45, 55),
            (50, 55),
            (15, 5),
            (15, 10),
            (15, 15),
            (30, 10),
            (30, 15),
            (30, 20),
            (45, 15),
            (45, 20),
            (45, 25),
            (5, 25),
            (10, 25),
            (15, 25),
            (25, 35),
            (30, 35),
            (35, 35),
            (35, 45),
            (40, 45),
            (45, 45),
            (10, 40),
            (15, 40),
            (40, 30),
        ],
        "spikes": [(20, 10), (35, 20), (10, 35), (50, 40)],
        "mines": [(10, 15), (35, 15), (45, 30)],
        "gems": [(10, 10), (25, 15), (40, 10), (50, 30), (20, 45), (50, 50)],
        "keys": [],
        "doors": [],
        "pink_blocks": [],
        "teleporters": [],
        "player": (5, 5),
        "exit": (50, 5),
        "spawn_positions": [(5, 5), (25, 5), (50, 20), (5, 30)],
    },
    {
        "total_gems": 8,
        "required_gems": 8,
        "max_moves": 300,
        "walls": [
            (0, 0),
            (5, 0),
            (10, 0),
            (15, 0),
            (20, 0),
            (25, 0),
            (30, 0),
            (35, 0),
            (40, 0),
            (45, 0),
            (50, 0),
            (55, 0),
            (0, 5),
            (0, 10),
            (0, 15),
            (0, 20),
            (0, 25),
            (0, 30),
            (0, 35),
            (0, 40),
            (0, 45),
            (0, 50),
            (0, 55),
            (55, 5),
            (55, 10),
            (55, 15),
            (55, 20),
            (55, 25),
            (55, 30),
            (55, 35),
            (55, 40),
            (55, 45),
            (55, 50),
            (55, 55),
            (5, 55),
            (10, 55),
            (15, 55),
            (20, 55),
            (25, 55),
            (30, 55),
            (35, 55),
            (40, 55),
            (45, 55),
            (50, 55),
            (25, 5),
            (25, 10),
            (25, 15),
            (25, 20),
            (25, 25),
            (10, 20),
            (15, 20),
            (40, 20),
            (45, 20),
            (15, 40),
            (20, 40),
            (35, 40),
            (40, 40),
        ],
        "spikes": [(10, 10), (30, 10), (45, 10), (10, 45), (45, 45)],
        "mines": [(15, 30), (35, 5), (50, 25), (30, 45)],
        "gems": [
            (10, 5),
            (15, 10),
            (5, 25),
            (15, 50),
            (35, 10),
            (45, 15),
            (50, 30),
            (40, 50),
        ],
        "keys": [(5, 50)],
        "doors": [(25, 30)],
        "pink_blocks": [(25, 35), (50, 45)],
        "teleporters": [],
        "player": (5, 5),
        "exit": (50, 50),
        "spawn_positions": [(5, 5), (20, 5), (50, 15), (5, 35)],
    },
    {
        "total_gems": 8,
        "required_gems": 8,
        "max_moves": 350,
        "walls": [
            (0, 0),
            (5, 0),
            (10, 0),
            (15, 0),
            (20, 0),
            (25, 0),
            (30, 0),
            (35, 0),
            (40, 0),
            (45, 0),
            (50, 0),
            (55, 0),
            (0, 5),
            (0, 10),
            (0, 15),
            (0, 20),
            (0, 25),
            (0, 30),
            (0, 35),
            (0, 40),
            (0, 45),
            (0, 50),
            (0, 55),
            (55, 5),
            (55, 10),
            (55, 15),
            (55, 20),
            (55, 25),
            (55, 30),
            (55, 35),
            (55, 40),
            (55, 45),
            (55, 50),
            (55, 55),
            (5, 55),
            (10, 55),
            (15, 55),
            (20, 55),
            (25, 55),
            (30, 55),
            (35, 55),
            (40, 55),
            (45, 55),
            (50, 55),
            (15, 10),
            (20, 10),
            (25, 10),
            (30, 10),
            (35, 10),
            (15, 20),
            (20, 20),
            (25, 20),
            (30, 20),
            (40, 20),
            (45, 20),
            (15, 30),
            (25, 30),
            (30, 30),
            (35, 30),
            (45, 30),
            (10, 40),
            (15, 40),
            (25, 40),
            (30, 40),
            (40, 40),
        ],
        "spikes": [(30, 5), (45, 15), (10, 25), (40, 35)],
        "mines": [
            (35, 5),
            (25, 25),
            (35, 25),
            (15, 35),
            (35, 35),
            (30, 45),
        ],
        "gems": [
            (25, 5),
            (40, 5),
            (50, 15),
            (5, 15),
            (5, 30),
            (25, 45),
            (50, 45),
            (5, 50),
        ],
        "keys": [],
        "doors": [],
        "pink_blocks": [],
        "teleporters": [],
        "player": (5, 5),
        "exit": (50, 50),
        "spawn_positions": [(5, 5), (50, 5), (5, 25), (50, 25)],
    },
    {
        "total_gems": 10,
        "required_gems": 10,
        "max_moves": 425,
        "walls": [
            (0, 0),
            (5, 0),
            (10, 0),
            (15, 0),
            (20, 0),
            (25, 0),
            (30, 0),
            (35, 0),
            (40, 0),
            (45, 0),
            (50, 0),
            (55, 0),
            (0, 5),
            (0, 10),
            (0, 15),
            (0, 20),
            (0, 25),
            (0, 30),
            (0, 35),
            (0, 40),
            (0, 45),
            (0, 50),
            (0, 55),
            (55, 5),
            (55, 10),
            (55, 15),
            (55, 20),
            (55, 25),
            (55, 30),
            (55, 35),
            (55, 40),
            (55, 45),
            (55, 50),
            (55, 55),
            (5, 55),
            (10, 55),
            (15, 55),
            (20, 55),
            (25, 55),
            (30, 55),
            (35, 55),
            (40, 55),
            (45, 55),
            (50, 55),
            (25, 5),
            (25, 10),
            (25, 15),
            (25, 20),
            (15, 10),
            (15, 15),
            (45, 10),
            (40, 20),
            (40, 25),
            (10, 25),
            (10, 30),
            (20, 35),
            (35, 35),
            (15, 40),
            (35, 40),
        ],
        "spikes": [(20, 10), (45, 15), (30, 30), (10, 45)],
        "mines": [
            (10, 15),
            (15, 5),
            (30, 5),
            (50, 20),
            (30, 35),
            (40, 45),
            (20, 50),
            (40, 50),
        ],
        "gems": [
            (10, 5),
            (40, 5),
            (35, 15),
            (5, 20),
            (50, 30),
            (50, 40),
            (5, 45),
            (45, 45),
            (25, 50),
            (30, 50),
        ],
        "keys": [(5, 35)],
        "doors": [(25, 25)],
        "pink_blocks": [],
        "teleporters": [],
        "player": (5, 5),
        "exit": (50, 50),
        "spawn_positions": [(5, 5), (50, 5), (5, 25), (50, 25)],
    },
    {
        "total_gems": 12,
        "required_gems": 12,
        "max_moves": 500,
        "walls": [
            (0, 0),
            (5, 0),
            (10, 0),
            (15, 0),
            (20, 0),
            (25, 0),
            (30, 0),
            (35, 0),
            (40, 0),
            (45, 0),
            (50, 0),
            (55, 0),
            (0, 5),
            (0, 10),
            (0, 15),
            (0, 20),
            (0, 25),
            (0, 30),
            (0, 35),
            (0, 40),
            (0, 45),
            (0, 50),
            (0, 55),
            (55, 5),
            (55, 10),
            (55, 15),
            (55, 20),
            (55, 25),
            (55, 30),
            (55, 35),
            (55, 40),
            (55, 45),
            (55, 50),
            (55, 55),
            (5, 55),
            (10, 55),
            (15, 55),
            (20, 55),
            (25, 55),
            (30, 55),
            (35, 55),
            (40, 55),
            (45, 55),
            (50, 55),
            (10, 10),
            (20, 10),
            (30, 10),
            (40, 10),
            (15, 20),
            (35, 20),
            (15, 30),
            (35, 30),
            (10, 40),
            (40, 40),
        ],
        "spikes": [(25, 5), (45, 15), (30, 25), (10, 35), (50, 35), (15, 50)],
        "mines": [
            (20, 5),
            (30, 5),
            (50, 15),
            (25, 15),
            (10, 25),
            (40, 25),
            (30, 35),
            (45, 35),
            (15, 45),
            (35, 45),
            (10, 50),
            (45, 50),
        ],
        "gems": [
            (15, 5),
            (35, 5),
            (50, 10),
            (5, 15),
            (5, 35),
            (45, 25),
            (25, 30),
            (25, 45),
            (50, 45),
            (5, 50),
            (40, 50),
            (30, 50),
        ],
        "keys": [(5, 25), (45, 40)],
        "doors": [(25, 20), (25, 40)],
        "pink_blocks": [],
        "teleporters": [],
        "player": (5, 5),
        "exit": (50, 50),
        "spawn_positions": [(5, 5), (50, 5), (15, 25), (50, 25)],
    },
]


def _build_level(data: Dict) -> Level:
    sprite_list = []
    for pos in data.get("walls", []):
        sprite_list.append(_SPRITES["wall"].clone().set_position(pos[0], pos[1]))
    for pos in data.get("spikes", []):
        sprite_list.append(_SPRITES["spike"].clone().set_position(pos[0], pos[1]))
    for pos in data.get("mines", []):
        sprite_list.append(_SPRITES["mine"].clone().set_position(pos[0], pos[1]))
    for pos in data.get("gems", []):
        sprite_list.append(_SPRITES["gem"].clone().set_position(pos[0], pos[1]))
    for pos in data.get("keys", []):
        sprite_list.append(_SPRITES["key"].clone().set_position(pos[0], pos[1]))
    for pos in data.get("doors", []):
        sprite_list.append(_SPRITES["door"].clone().set_position(pos[0], pos[1]))
    for pos in data.get("pink_blocks", []):
        sprite_list.append(_SPRITES["pink_block"].clone().set_position(pos[0], pos[1]))
    for pos in data.get("teleporters", []):
        sprite_list.append(_SPRITES["teleporter"].clone().set_position(pos[0], pos[1]))
    px, py = data["player"]
    sprite_list.append(_SPRITES["player"].clone().set_position(px, py))
    ex, ey = data["exit"]
    sprite_list.append(_SPRITES["exit"].clone().set_position(ex, ey))
    return Level(
        sprites=sprite_list,
        grid_size=(60, 60),
        data={
            "total_gems": data["total_gems"],
            "required_gems": data["required_gems"],
            "max_moves": data["max_moves"],
            "player_x": px,
            "player_y": py,
            "spawn_positions": data.get("spawn_positions", [(px, py)]),
        },
    )


_TAG_TO_CHAR: Dict[str, str] = {
    "player": "P",
    "wall": "#",
    "gem": "G",
    "key": "K",
    "door": "D",
    "exit": "E",
    "spike": "X",
    "teleporter": "T",
    "mine": "M",
    "pink_block": "B",
}

_TAG_PRIORITY: Dict[str, int] = {
    "wall": 0,
    "solid": -1,
    "spike": 1,
    "door": 1,
    "pink_block": 1,
    "mine": 2,
    "gem": 2,
    "key": 2,
    "collectible": -1,
    "teleporter": 2,
    "exit": 3,
    "hazard": -1,
    "player": 5,
}


class GameHUD(RenderableUserDisplay):
    def __init__(self, game: "Mz47"):
        self._game = game

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        game = self._game

        frame_h, frame_w = frame.shape
        cam_w, cam_h = 60, 60
        scale = min(frame_w // cam_w, frame_h // cam_h)
        if scale < 1:
            scale = 1
        x_off = (frame_w - cam_w * scale) // 2
        y_off = (frame_h - cam_h * scale) // 2

        def grid_y(row):
            return slice(y_off + row * scale, y_off + (row + 1) * scale)

        def grid_x(col):
            return slice(x_off + col * scale, x_off + (col + 1) * scale)

        for i in range(MAX_LIVES):
            frame[grid_y(0), grid_x(1 + i)] = RED if game._lives > i else BLACK

        gem_count = min(game._gems_collected, 15)
        for i in range(gem_count):
            col = 5 + i * 2
            if col < cam_w:
                frame[grid_y(0), grid_x(col)] = YELLOW

        if game._keys_collected > 0:
            for i in range(min(game._keys_collected, 5)):
                col = 40 + i * 2
                if col < cam_w:
                    frame[grid_y(0), grid_x(col)] = GREEN

        if game._max_moves > 0:
            remaining = max(0, game._max_moves - game._moves_used)
            ratio = remaining / game._max_moves
            cells_filled = int(ratio * cam_w)
            bar_row = grid_y(cam_h - 1)
            for col in range(cam_w):
                if ratio > 0.5:
                    color = GREEN
                elif ratio > 0.25:
                    color = YELLOW
                else:
                    color = RED
                frame[bar_row, grid_x(col)] = color if col < cells_filled else GREY

        return frame


class Mz47(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

        self._gems_collected = 0
        self._keys_collected = 0
        self._player = None
        self._moves_used = 0
        self._max_moves = 0
        self._lives = MAX_LIVES

        self._removed_gems: List = []
        self._removed_keys: List = []
        self._removed_doors: List = []
        self._removed_pink_blocks: List = []
        self._start_x = 0
        self._start_y = 0

        self._consecutive_resets = 0
        self._game_over = False
        self._undo_snapshot: Optional[Dict] = None
        self._can_undo = False

        self._hud = GameHUD(self)

        game_levels = [_build_level(d) for d in _LEVEL_DATA]

        super().__init__(
            "mz47",
            game_levels,
            Camera(
                x=0,
                y=0,
                width=60,
                height=60,
                background=BACKGROUND_COLOR,
                letter_box=PADDING_COLOR,
                interfaces=[self._hud],
            ),
            available_actions=[0, 1, 2, 3, 4, 7],
        )

    def on_set_level(self, level: Level) -> None:
        self._player = self.current_level.get_sprites_by_tag("player")[0]
        self._gems_collected = 0
        self._keys_collected = 0
        self._moves_used = 0

        self._removed_gems = []
        self._removed_keys = []
        self._removed_doors = []
        self._removed_pink_blocks = []

        self._spawn_positions = self.current_level.get_data("spawn_positions")
        self._start_x = self.current_level.get_data("player_x")
        self._start_y = self.current_level.get_data("player_y")
        self._randomize_spawn()

        max_moves_raw = self.current_level.get_data("max_moves")
        self._max_moves = int(max_moves_raw) if max_moves_raw is not None else 250
        self._lives = MAX_LIVES
        self._game_over = False
        self._undo_snapshot = None
        self._can_undo = False

    def handle_reset(self) -> None:
        self._consecutive_resets += 1
        if self._consecutive_resets >= 2:
            self._consecutive_resets = 0
            self.full_reset()
        else:
            self.level_reset()

    def _randomize_spawn(self) -> None:
        positions = self._spawn_positions
        if positions:
            sx, sy = self._rng.choice(positions)
            self._start_x = sx
            self._start_y = sy
        self._player.set_position(self._start_x, self._start_y)

    def _reset_current_level(self) -> None:
        for gem_sprite in self._removed_gems:
            self.current_level.add_sprite(gem_sprite)
        self._removed_gems = []

        for key_sprite in self._removed_keys:
            self.current_level.add_sprite(key_sprite)
        self._removed_keys = []

        for door_sprite in self._removed_doors:
            self.current_level.add_sprite(door_sprite)
        self._removed_doors = []

        for pink_sprite in self._removed_pink_blocks:
            self.current_level.add_sprite(pink_sprite)
        self._removed_pink_blocks = []

        self._randomize_spawn()

        self._gems_collected = 0
        self._keys_collected = 0
        self._moves_used = 0
        self._undo_snapshot = None
        self._can_undo = False

    def _check_collision(self, x: int, y: int) -> bool:
        for sprite in self.current_level.get_sprites():
            if sprite.tags and "solid" in sprite.tags:
                if (
                    sprite.x <= x < sprite.x + sprite.width
                    and sprite.y <= y < sprite.y + sprite.height
                ):
                    return True
        return False

    def _get_sprite_at(self, x: int, y: int, tag: Optional[str] = None):
        for sprite in self.current_level.get_sprites():
            if (
                sprite.x <= x < sprite.x + sprite.width
                and sprite.y <= y < sprite.y + sprite.height
            ):
                if tag is None:
                    return sprite
                if sprite.tags and tag in sprite.tags:
                    return sprite
        return None

    def _collect_item(self, x: int, y: int) -> None:
        collectibles = self.current_level.get_sprites_by_tag("collectible")
        for item in collectibles:
            if item.x <= x < item.x + item.width and item.y <= y < item.y + item.height:
                if item.tags and "gem" in item.tags:
                    self._gems_collected += 1
                    self.current_level.remove_sprite(item)
                    self._removed_gems.append(item)
                elif item.tags and "key" in item.tags:
                    self._keys_collected += 1
                    self.current_level.remove_sprite(item)
                    self._removed_keys.append(item)
                    doors = self.current_level.get_sprites_by_tag("door")
                    if doors:
                        door = doors[0]
                        self.current_level.remove_sprite(door)
                        self._removed_doors.append(door)
                    pink_blocks = self.current_level.get_sprites_by_tag("pink_block")
                    if pink_blocks:
                        pink_block = pink_blocks[0]
                        self.current_level.remove_sprite(pink_block)
                        self._removed_pink_blocks.append(pink_block)

    def _check_mine(self, x: int, y: int) -> bool:
        mine = self._get_sprite_at(x, y, "mine")
        return mine is not None

    def _handle_mine_hit(self) -> None:
        self._lives -= 1
        if self._lives <= 0:
            self._game_over = True
            self.lose()
            return
        self._reset_current_level()

    def _check_exit(self, x: int, y: int) -> bool:
        required_gems = self.current_level.get_data("required_gems")
        if self._gems_collected < required_gems:
            return False
        exits = self.current_level.get_sprites_by_tag("exit")
        for exit_sprite in exits:
            if (
                exit_sprite.x <= x < exit_sprite.x + exit_sprite.width
                and exit_sprite.y <= y < exit_sprite.y + exit_sprite.height
            ):
                return True
        return False

    def _save_state(self) -> None:
        self._undo_snapshot = {
            "px": self._player.x,
            "py": self._player.y,
            "gems_collected": self._gems_collected,
            "keys_collected": self._keys_collected,
            "removed_gems": list(self._removed_gems),
            "removed_keys": list(self._removed_keys),
            "removed_doors": list(self._removed_doors),
            "removed_pink_blocks": list(self._removed_pink_blocks),
        }
        self._can_undo = True

    def _undo(self) -> None:
        if not self._undo_snapshot:
            return
        snap = self._undo_snapshot

        self._player.set_position(snap["px"], snap["py"])

        snap_removed_gems = set(id(s) for s in snap["removed_gems"])
        for gem in self._removed_gems:
            if id(gem) not in snap_removed_gems:
                self.current_level.add_sprite(gem)
        self._removed_gems = snap["removed_gems"]

        snap_removed_keys = set(id(s) for s in snap["removed_keys"])
        for key in self._removed_keys:
            if id(key) not in snap_removed_keys:
                self.current_level.add_sprite(key)
        self._removed_keys = snap["removed_keys"]

        snap_removed_doors = set(id(s) for s in snap["removed_doors"])
        for door in self._removed_doors:
            if id(door) not in snap_removed_doors:
                self.current_level.add_sprite(door)
        self._removed_doors = snap["removed_doors"]

        snap_removed_pinks = set(id(s) for s in snap["removed_pink_blocks"])
        for pink in self._removed_pink_blocks:
            if id(pink) not in snap_removed_pinks:
                self.current_level.add_sprite(pink)
        self._removed_pink_blocks = snap["removed_pink_blocks"]

        self._gems_collected = snap["gems_collected"]
        self._keys_collected = snap["keys_collected"]

        self._undo_snapshot = None
        self._can_undo = False

    def _parse_direction(self) -> Tuple[int, int]:
        dx = dy = 0
        if self.action.id == GameAction.ACTION1:
            dy = -5
        elif self.action.id == GameAction.ACTION2:
            dy = 5
        elif self.action.id == GameAction.ACTION3:
            dx = -5
        elif self.action.id == GameAction.ACTION4:
            dx = 5
        return dx, dy

    def step(self) -> None:
        if self._game_over:
            self.complete_action()
            return

        if self.action.id == GameAction.ACTION7:
            self._consecutive_resets = 0
            if self._can_undo:
                self._undo()
            self._moves_used += 1
            if self._max_moves > 0 and self._moves_used >= self._max_moves:
                self._game_over = True
                self.lose()
            self.complete_action()
            return

        dx, dy = self._parse_direction()

        if dx == 0 and dy == 0:
            self.complete_action()
            return

        self._consecutive_resets = 0
        self._save_state()

        self._moves_used += 1

        if self._max_moves > 0 and self._moves_used >= self._max_moves:
            self._game_over = True
            self.lose()
            self.complete_action()
            return

        new_x = self._player.x + dx
        new_y = self._player.y + dy

        if not self._check_collision(new_x, new_y):
            self._player.set_position(new_x, new_y)

            if self._check_mine(new_x, new_y):
                self._handle_mine_hit()
                self.complete_action()
                return

            self._collect_item(new_x, new_y)

            if self._check_exit(new_x, new_y):
                self.next_level()

        self.complete_action()


class PuzzleEnvironment:
    _ACTION_MAP: Dict[str, GameAction] = {
        "up": GameAction.ACTION1,
        "down": GameAction.ACTION2,
        "left": GameAction.ACTION3,
        "right": GameAction.ACTION4,
        "reset": GameAction.RESET,
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
        self._engine = Mz47(seed=seed)
        self._total_turns = 0
        self._last_action_was_reset = False
        self._game_won = False
        self._total_levels = len(self._engine._levels)

    def _build_text_obs(self) -> str:
        e = self._engine
        gs = e.current_level.grid_size
        pw, ph = (gs[0] if gs else 60), (gs[1] if gs else 60)
        tile_size = 5
        tw, th = pw // tile_size, ph // tile_size
        text_grid: List[List[str]] = [["." for _ in range(tw)] for _ in range(th)]
        prio_grid: List[List[int]] = [[-2 for _ in range(tw)] for _ in range(th)]
        tag_char = _TAG_TO_CHAR
        tag_prio = _TAG_PRIORITY
        for sprite in e.current_level.get_sprites():
            if not sprite.is_visible:
                continue
            tx = sprite.x // tile_size
            ty = sprite.y // tile_size
            if 0 <= tx < tw and 0 <= ty < th:
                for tag in sprite.tags or []:
                    ch = tag_char.get(tag, "")
                    pr = tag_prio.get(tag, -1)
                    if ch and pr > prio_grid[ty][tx]:
                        text_grid[ty][tx] = ch
                        prio_grid[ty][tx] = pr
        grid_text = "\n".join("".join(row) for row in text_grid)
        remaining = max(0, e._max_moves - e._moves_used)
        header = (
            f"Level:{e.level_index + 1} Lives:{e._lives} "
            f"Gems:{e._gems_collected}/{e.current_level.get_data('required_gems')} "
            f"Keys:{e._keys_collected} Moves:{remaining}/{e._max_moves}"
        )
        return header.strip() + "\n" + grid_text

    @staticmethod
    def _make_png_chunk(chunk_type: bytes, data: bytes) -> bytes:
        chunk = chunk_type + data
        crc = struct.pack(">I", zlib.crc32(chunk) & 0xFFFFFFFF)
        return struct.pack(">I", len(data)) + chunk + crc

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
        for idx, color in enumerate(self.ARC_PALETTE):
            mask = arr == idx
            rgb[mask] = color
        raw_rows = b""
        for row in range(h):
            raw_rows += b"\x00" + rgb[row].tobytes()
        sig = b"\x89PNG\r\n\x1a\n"
        ihdr_data = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
        ihdr = self._make_png_chunk(b"IHDR", ihdr_data)
        idat = self._make_png_chunk(b"IDAT", zlib.compress(raw_rows))
        iend = self._make_png_chunk(b"IEND", b"")
        return sig + ihdr + idat + iend

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
                "level_index": e.level_index,
                "total_levels": self._total_levels,
                "lives": e._lives,
                "max_moves": e._max_moves,
                "moves_used": e._moves_used,
                "gems_collected": e._gems_collected,
                "keys_collected": e._keys_collected,
                "game_over": e._game_over,
                "done": done,
                "info": info or {},
                "levels_completed": getattr(self._engine, "_score", 0),
            },
        )

    def reset(self) -> GameState:
        e = self._engine
        is_full = self._game_won or self._last_action_was_reset
        if is_full:
            e._consecutive_resets = 1
        self._game_won = False
        e.perform_action(ActionInput(id=GameAction.RESET))
        if is_full:
            self._total_turns = 0
        self._last_action_was_reset = True
        return self._build_game_state()

    def get_actions(self) -> List[str]:
        if self._engine._game_over or self._game_won:
            return ["reset"]
        return ["up", "down", "left", "right", "undo", "reset"]

    def is_done(self) -> bool:
        return self._engine._game_over or self._game_won

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

        level_before = e.level_index
        action_input = ActionInput(id=game_action)
        frame = e.perform_action(action_input, raw=True)
        level_after = e.level_index

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
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        for idx, color in enumerate(self.ARC_PALETTE):
            mask = index_grid == idx
            rgb[mask] = color
        return rgb

    def close(self) -> None:
        self._engine = None


class ArcGameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 5}

    ACTION_LIST = ["reset", "up", "down", "left", "right", "undo"]

    OBS_HEIGHT = 64
    OBS_WIDTH = 64

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
        self.render_mode = render_mode

        self._seed = seed
        self._env = None

        self._action_to_string = {i: a for i, a in enumerate(self.ACTION_LIST)}
        self._string_to_action = {a: i for i, a in enumerate(self.ACTION_LIST)}

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.OBS_HEIGHT, self.OBS_WIDTH, 3),
            dtype=np.uint8,
        )
        self.action_space = spaces.Discrete(len(self.ACTION_LIST))

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        if seed is not None:
            self._seed = seed

        self._env = PuzzleEnvironment(seed=self._seed)
        state = self._env.reset()

        return self._get_obs(), self._build_info(state)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        action_str = self._action_to_string[int(action)]

        result = self._env.step(action_str)

        obs = self._get_obs()
        reward = result.reward
        terminated = result.done
        truncated = False
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

    def _build_info(self, state: GameState, step_info: Optional[Dict] = None) -> Dict:
        info = {
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

    try:
        check_env(env.unwrapped, skip_render_check=True)
        print("[PASS] check_env passed — environment is Gymnasium-compliant.")
    except Exception as e:
        print(f"[FAIL] check_env failed: {e}")

    obs, info = env.reset()
    print(f"Obs shape: {obs.shape}, dtype: {obs.dtype}")
    print(f"Valid actions: {info.get('valid_actions')}")

    mask = env.action_mask()
    valid_indices = np.where(mask == 1)[0]
    if len(valid_indices) > 0:
        obs, reward, term, trunc, info = env.step(valid_indices[0])
        print(f"Step → reward={reward}, terminated={term}, truncated={trunc}")

    env.close()
    print("Smoke test passed!")
