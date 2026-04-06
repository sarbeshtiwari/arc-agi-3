import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import struct
import zlib

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


WHITE = 0
GREY = 3
DARK_GREY = 4
BLACK = 5
MAGENTA = 6
RED = 8
LIGHT_BLUE = 10
YELLOW = 11
ORANGE = 12
GREEN = 14
PURPLE = 15

TILE_EMPTY = 0
TILE_CACTUS = 1
TILE_BIRD = 2
TILE_GRAVITY = 3
TILE_FINISH = 4

FRAME_SIZE = 64
VISIBLE_TILES = 8
TILE_PX = FRAME_SIZE // VISIBLE_TILES
GROUND_TOP = 40
GROUND_H = 8
UNDER_TOP = GROUND_TOP + GROUND_H
CACTUS_TOP = 28
CACTUS_H = GROUND_TOP - CACTUS_TOP
BIRD_TOP = 14
LIVES_MAX = 3

ACT_JUMP = GameAction.ACTION1
ACT_DUCK = GameAction.ACTION2
ACT_MOVE = GameAction.ACTION4
ACT_UNDO = GameAction.ACTION7

GROUND_COLORS = {
    TILE_EMPTY: GREY,
    TILE_CACTUS: GREY,
    TILE_BIRD: GREY,
    TILE_FINISH: GREEN,
}

LEVEL_DEFS = [
    {
        "path": [
            TILE_EMPTY,
            TILE_EMPTY,
            TILE_CACTUS,
            TILE_EMPTY,
            TILE_CACTUS,
            TILE_EMPTY,
            TILE_CACTUS,
            TILE_EMPTY,
            TILE_CACTUS,
            TILE_EMPTY,
            TILE_EMPTY,
            TILE_FINISH,
        ],
        "move_cap": 30,
    },
    {
        "path": [
            TILE_EMPTY,
            TILE_EMPTY,
            TILE_CACTUS,
            TILE_EMPTY,
            TILE_GRAVITY,
            TILE_EMPTY,
            TILE_CACTUS,
            TILE_EMPTY,
            TILE_BIRD,
            TILE_EMPTY,
            TILE_FINISH,
        ],
        "move_cap": 30,
    },
    {
        "path": [
            TILE_EMPTY,
            TILE_EMPTY,
            TILE_CACTUS,
            TILE_BIRD,
            TILE_EMPTY,
            TILE_GRAVITY,
            TILE_EMPTY,
            TILE_BIRD,
            TILE_CACTUS,
            TILE_EMPTY,
            TILE_GRAVITY,
            TILE_EMPTY,
            TILE_CACTUS,
            TILE_BIRD,
            TILE_EMPTY,
            TILE_FINISH,
        ],
        "move_cap": 40,
    },
    {
        "path": [
            TILE_EMPTY,
            TILE_EMPTY,
            TILE_CACTUS,
            TILE_BIRD,
            TILE_EMPTY,
            TILE_GRAVITY,
            TILE_EMPTY,
            TILE_BIRD,
            TILE_CACTUS,
            TILE_EMPTY,
            TILE_CACTUS,
            TILE_BIRD,
            TILE_EMPTY,
            TILE_GRAVITY,
            TILE_EMPTY,
            TILE_CACTUS,
            TILE_BIRD,
            TILE_EMPTY,
            TILE_GRAVITY,
            TILE_EMPTY,
            TILE_BIRD,
            TILE_CACTUS,
            TILE_EMPTY,
            TILE_FINISH,
        ],
        "move_cap": 60,
    },
]

TOTAL_LEVELS = len(LEVEL_DEFS)


def _set_pixel(frame, x, y, color):
    if 0 <= x < FRAME_SIZE and 0 <= y < FRAME_SIZE:
        frame[y, x] = color


def _fill_rect(frame, x, y, w, h, color):
    for row_off in range(h):
        for col_off in range(w):
            _set_pixel(frame, x + col_off, y + row_off, color)


def _draw_cactus(frame, tile_x):
    stem_x = tile_x + 3
    _fill_rect(frame, stem_x, CACTUS_TOP, 2, CACTUS_H, GREEN)
    _fill_rect(frame, stem_x - 2, CACTUS_TOP + 2, 2, 5, GREEN)
    _fill_rect(frame, stem_x + 2, CACTUS_TOP + 4, 2, 4, GREEN)
    _set_pixel(frame, stem_x - 2, CACTUS_TOP + 1, GREEN)
    _set_pixel(frame, stem_x + 3, CACTUS_TOP + 3, GREEN)


def _draw_bird(frame, tile_x):
    tx = tile_x
    _fill_rect(frame, tx + 2, BIRD_TOP + 3, 4, 2, RED)
    _fill_rect(frame, tx + 5, BIRD_TOP + 2, 2, 2, RED)
    _set_pixel(frame, tx + 6, BIRD_TOP + 2, WHITE)
    _set_pixel(frame, tx + 7, BIRD_TOP + 3, ORANGE)
    _fill_rect(frame, tx + 1, BIRD_TOP, 2, 3, RED)
    _fill_rect(frame, tx + 4, BIRD_TOP, 2, 2, RED)
    _set_pixel(frame, tx + 1, BIRD_TOP + 4, RED)
    _set_pixel(frame, tx + 0, BIRD_TOP + 5, RED)


def _draw_dino_normal(frame, tile_x):
    tx = tile_x
    _fill_rect(frame, tx + 3, 26, 4, 3, YELLOW)
    _set_pixel(frame, tx + 6, 27, BLACK)
    _set_pixel(frame, tx + 7, 28, ORANGE)
    _fill_rect(frame, tx + 3, 29, 2, 1, YELLOW)
    _fill_rect(frame, tx + 1, 30, 5, 4, YELLOW)
    _fill_rect(frame, tx + 0, 31, 1, 2, YELLOW)
    _set_pixel(frame, tx + 6, 31, ORANGE)
    _set_pixel(frame, tx + 6, 32, ORANGE)
    _fill_rect(frame, tx + 2, 34, 2, 6, YELLOW)
    _fill_rect(frame, tx + 4, 34, 2, 6, YELLOW)


def _draw_dino_jump(frame, tile_x):
    tx = tile_x
    _fill_rect(frame, tx + 3, 12, 4, 3, YELLOW)
    _set_pixel(frame, tx + 6, 13, BLACK)
    _set_pixel(frame, tx + 7, 14, ORANGE)
    _fill_rect(frame, tx + 3, 15, 2, 1, YELLOW)
    _fill_rect(frame, tx + 1, 16, 5, 4, YELLOW)
    _fill_rect(frame, tx + 0, 17, 1, 2, YELLOW)
    _set_pixel(frame, tx + 6, 17, ORANGE)
    _fill_rect(frame, tx + 2, 20, 4, 6, YELLOW)


def _draw_dino_duck(frame, tile_x):
    tx = tile_x
    _fill_rect(frame, tx + 5, 32, 2, 2, YELLOW)
    _set_pixel(frame, tx + 7, 32, BLACK)
    _set_pixel(frame, tx + 7, 33, ORANGE)
    _fill_rect(frame, tx + 1, 33, 5, 4, YELLOW)
    _fill_rect(frame, tx + 0, 34, 1, 2, YELLOW)
    _fill_rect(frame, tx + 2, 37, 2, 3, YELLOW)
    _fill_rect(frame, tx + 4, 37, 2, 3, YELLOW)


def _draw_dino_normal_flipped(frame, tile_x):
    tx = tile_x
    _fill_rect(frame, tx + 2, 4, 2, 6, PURPLE)
    _fill_rect(frame, tx + 4, 4, 2, 6, PURPLE)
    _fill_rect(frame, tx + 1, 10, 5, 4, PURPLE)
    _fill_rect(frame, tx + 0, 11, 1, 2, PURPLE)
    _set_pixel(frame, tx + 6, 11, ORANGE)
    _set_pixel(frame, tx + 6, 12, ORANGE)
    _fill_rect(frame, tx + 3, 14, 2, 1, PURPLE)
    _fill_rect(frame, tx + 3, 15, 4, 3, PURPLE)
    _set_pixel(frame, tx + 6, 16, BLACK)
    _set_pixel(frame, tx + 7, 15, ORANGE)


def _draw_dino_jump_flipped(frame, tile_x):
    tx = tile_x
    _fill_rect(frame, tx + 2, 22, 4, 6, PURPLE)
    _fill_rect(frame, tx + 1, 28, 5, 4, PURPLE)
    _fill_rect(frame, tx + 0, 29, 1, 2, PURPLE)
    _set_pixel(frame, tx + 6, 30, ORANGE)
    _fill_rect(frame, tx + 3, 32, 2, 1, PURPLE)
    _fill_rect(frame, tx + 3, 33, 4, 3, PURPLE)
    _set_pixel(frame, tx + 6, 34, BLACK)
    _set_pixel(frame, tx + 7, 33, ORANGE)


def _draw_dino_duck_flipped(frame, tile_x):
    tx = tile_x
    _fill_rect(frame, tx + 2, 4, 2, 3, PURPLE)
    _fill_rect(frame, tx + 4, 4, 2, 3, PURPLE)
    _fill_rect(frame, tx + 1, 7, 5, 4, PURPLE)
    _fill_rect(frame, tx + 0, 8, 1, 2, PURPLE)
    _fill_rect(frame, tx + 5, 10, 2, 2, PURPLE)
    _set_pixel(frame, tx + 7, 11, BLACK)
    _set_pixel(frame, tx + 7, 10, ORANGE)


def _draw_gravity_marker(frame, tile_x):
    mx = tile_x + 1
    my = GROUND_TOP - 8
    _set_pixel(frame, mx + 3, my, MAGENTA)
    _fill_rect(frame, mx + 2, my + 1, 3, 1, MAGENTA)
    _fill_rect(frame, mx + 1, my + 2, 5, 1, MAGENTA)
    _fill_rect(frame, mx + 1, my + 3, 5, 2, PURPLE)
    _fill_rect(frame, mx + 1, my + 5, 5, 1, MAGENTA)
    _fill_rect(frame, mx + 2, my + 6, 3, 1, MAGENTA)
    _set_pixel(frame, mx + 3, my + 7, MAGENTA)


def _draw_finish_flag(frame, tile_x):
    pole_x = tile_x + 3
    _fill_rect(frame, pole_x, 24, 1, GROUND_TOP - 24, WHITE)
    _fill_rect(frame, pole_x + 1, 24, 3, 4, GREEN)
    _fill_rect(frame, pole_x + 1, 28, 3, 1, WHITE)


def _draw_solid_border(frame, color):
    for pixel in range(FRAME_SIZE):
        _set_pixel(frame, pixel, 0, color)
        _set_pixel(frame, pixel, FRAME_SIZE - 1, color)
        _set_pixel(frame, 0, pixel, color)
        _set_pixel(frame, FRAME_SIZE - 1, pixel, color)


def _draw_inner_border(frame, color):
    for pixel in range(1, FRAME_SIZE - 1):
        _set_pixel(frame, pixel, 1, color)
        _set_pixel(frame, pixel, FRAME_SIZE - 2, color)
        _set_pixel(frame, 1, pixel, color)
        _set_pixel(frame, FRAME_SIZE - 2, pixel, color)


def _draw_dashed_border(frame, color):
    for pixel in range(0, FRAME_SIZE, 2):
        _set_pixel(frame, pixel, 0, color)
        _set_pixel(frame, pixel, FRAME_SIZE - 1, color)
        _set_pixel(frame, 0, pixel, color)
        _set_pixel(frame, FRAME_SIZE - 1, pixel, color)


def _render_background(frame, env):
    _fill_rect(frame, 0, 0, FRAME_SIZE, FRAME_SIZE, BLACK)
    _fill_rect(frame, 0, 4, FRAME_SIZE, GROUND_TOP - 4, LIGHT_BLUE)
    _fill_rect(frame, 0, UNDER_TOP, FRAME_SIZE, FRAME_SIZE - UNDER_TOP, DARK_GREY)
    if env.gravity_flipped:
        _fill_rect(frame, 0, 3, FRAME_SIZE, 2, PURPLE)


def _render_tiles(frame, env):
    for i in range(VISIBLE_TILES):
        path_idx = env.pos + i
        tile_x = i * TILE_PX
        if path_idx < len(env.path):
            tile = env.path[path_idx]
            ground_color = (
                PURPLE if tile == TILE_GRAVITY else GROUND_COLORS.get(tile, GREY)
            )
            _fill_rect(frame, tile_x, GROUND_TOP, TILE_PX, GROUND_H, ground_color)
            if tile == TILE_CACTUS:
                _draw_cactus(frame, tile_x)
            elif tile == TILE_BIRD:
                _draw_bird(frame, tile_x)
            elif tile == TILE_GRAVITY:
                _draw_gravity_marker(frame, tile_x)
            elif tile == TILE_FINISH:
                _draw_finish_flag(frame, tile_x)
        else:
            _fill_rect(frame, tile_x, GROUND_TOP, TILE_PX, GROUND_H, LIGHT_BLUE)


def _render_dino(frame, env):
    if env.gravity_flipped:
        if env.last_action == ACT_JUMP:
            _draw_dino_jump_flipped(frame, 0)
        elif env.last_action == ACT_DUCK:
            _draw_dino_duck_flipped(frame, 0)
        else:
            _draw_dino_normal_flipped(frame, 0)
    else:
        if env.last_action == ACT_JUMP:
            _draw_dino_jump(frame, 0)
        elif env.last_action == ACT_DUCK:
            _draw_dino_duck(frame, 0)
        else:
            _draw_dino_normal(frame, 0)


def _render_hud(frame, env):
    for i in range(LIVES_MAX):
        life_color = RED if i < env.lives else DARK_GREY
        _fill_rect(frame, FRAME_SIZE - 4 * (LIVES_MAX - i), 1, 2, 2, life_color)
    if env.move_cap > 0 and not env.won:
        remaining = max(0, env.move_cap - env.total_moves)
        ratio = remaining / env.move_cap
        bar_width = FRAME_SIZE - 4
        bar_filled = int(ratio * bar_width)
        _fill_rect(frame, 2, FRAME_SIZE - 3, bar_width, 1, DARK_GREY)
        if bar_filled > 0:
            _fill_rect(frame, 2, FRAME_SIZE - 3, bar_filled, 1, GREEN)


def _render_borders(frame, env):
    if env.won:
        _draw_solid_border(frame, GREEN)
    elif env.game_over:
        _draw_solid_border(frame, RED)
    elif env.just_flipped:
        _draw_solid_border(frame, PURPLE)
        _draw_inner_border(frame, MAGENTA)
    elif env.just_collided:
        _draw_dashed_border(frame, RED)


class RunnerHud(RenderableUserDisplay):
    def __init__(self):
        super().__init__()
        self.env = None

    def render_interface(self, frame):
        env = self.env
        if env is None:
            return frame
        _render_background(frame, env)
        _render_tiles(frame, env)
        _render_dino(frame, env)
        _render_hud(frame, env)
        _render_borders(frame, env)
        return frame


_anchor = Sprite(
    pixels=[[BLACK]],
    name="anchor",
    visible=False,
    collidable=False,
    tags=["anchor"],
    layer=0,
)

levels = [
    Level(
        sprites=[_anchor.clone().set_position(0, 0)],
        grid_size=(64, 64),
        data={"idx": i},
        name=f"Level {i + 1}",
    )
    for i in range(TOTAL_LEVELS)
]


class Eg07(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)
        self.hud = RunnerHud()
        self.hud.env = self
        self.level_idx = 0
        self.path = []
        self.pos = 0
        self.gravity_flipped = False
        self.total_moves = 0
        self.move_cap = 0
        self.lives = LIVES_MAX
        self.won = False
        self.game_over = False
        self.last_action = None
        self.just_flipped = False
        self.just_collided = False
        self.played_since_level_set = False
        self.level_desync = False
        self._first_load = True
        self.consecutive_resets = 0
        self._history = []

        super().__init__(
            "eg07",
            levels,
            Camera(
                x=0,
                y=0,
                width=64,
                height=64,
                background=BLACK,
                letter_box=BLACK,
                interfaces=[self.hud],
            ),
            available_actions=[0, 1, 2, 4, 7],
        )

    def _init_level(self, idx):
        cfg = LEVEL_DEFS[idx]
        self.level_idx = idx
        base_path = cfg["path"]
        first_obstacle = 0
        for i, t in enumerate(base_path):
            if t != TILE_EMPTY:
                first_obstacle = i
                break
        lead = self._rng.randint(1, 3)
        self.path = [TILE_EMPTY] * lead + base_path[first_obstacle:]
        self.pos = 0
        self.gravity_flipped = False
        self.total_moves = 0
        self.move_cap = cfg["move_cap"] + (lead - first_obstacle)
        self.lives = LIVES_MAX
        self.won = False
        self.game_over = False
        self.last_action = None
        self.just_flipped = False
        self.just_collided = False
        self.played_since_level_set = False
        self._history = []

    def on_set_level(self, level):
        idx = self.current_level.get_data("idx")
        if idx is None:
            idx = 0

        if self.level_desync:
            idx = self.level_idx

        if self._first_load:
            self._first_load = False
            self.consecutive_resets = 0
        elif idx == self.level_idx and not self.played_since_level_set:
            self.consecutive_resets += 1
        elif idx == self.level_idx and self.played_since_level_set:
            self.consecutive_resets = 1
        else:
            self.consecutive_resets = 0

        if self.consecutive_resets >= 2:
            self.consecutive_resets = 0
            idx = 0
            self.level_desync = True

        self._init_level(idx)

    def _get_effective_tile(self, path_idx):
        if path_idx >= len(self.path):
            return TILE_EMPTY
        return self.path[path_idx]

    def _required_action(self, tile):
        if tile == TILE_CACTUS:
            return ACT_DUCK if self.gravity_flipped else ACT_JUMP
        if tile == TILE_BIRD:
            return ACT_JUMP if self.gravity_flipped else ACT_DUCK
        return ACT_MOVE

    def _reset_run(self):
        self.pos = 0
        self.gravity_flipped = False
        self.last_action = None
        self.total_moves = 0
        self._history = []

    def _lose_life(self):
        self.lives -= 1
        self.just_collided = True
        if self.lives <= 0:
            self.game_over = True
            self.lose()
        else:
            self._reset_run()

    def _handle_tile_effect(self):
        if self.path[self.pos] == TILE_FINISH:
            self.won = True
            if self.level_idx >= TOTAL_LEVELS - 1:
                self.win()
            elif self.level_desync:
                self.consecutive_resets = 0
                self._init_level(self.level_idx + 1)
            else:
                self.next_level()
            return

        if self.path[self.pos] == TILE_GRAVITY:
            self.gravity_flipped = not self.gravity_flipped
            self.just_flipped = True

    def _save_state(self):
        self._history.append(
            {
                "pos": self.pos,
                "gravity_flipped": self.gravity_flipped,
                "lives": self.lives,
                "last_action": self.last_action,
                "won": self.won,
                "game_over": self.game_over,
                "just_flipped": self.just_flipped,
                "just_collided": self.just_collided,
                "played_since_level_set": self.played_since_level_set,
            }
        )

    def _undo(self):
        if not self._history:
            return
        snap = self._history.pop()
        self.pos = snap["pos"]
        self.gravity_flipped = snap["gravity_flipped"]
        self.lives = snap["lives"]
        self.last_action = snap["last_action"]
        self.won = snap["won"]
        self.game_over = snap["game_over"]
        self.just_flipped = snap["just_flipped"]
        self.just_collided = snap["just_collided"]
        self.played_since_level_set = snap["played_since_level_set"]

    def _process_action(self, action_id):
        if self.won or self.game_over:
            return

        self.played_since_level_set = True

        next_pos = self.pos + 1
        if next_pos >= len(self.path):
            return

        self.total_moves += 1

        if self.total_moves >= self.move_cap:
            self._lose_life()
            return

        effective_tile = self._get_effective_tile(next_pos)
        required = self._required_action(effective_tile)

        if action_id != required:
            self._lose_life()
            return

        self.last_action = action_id
        self.pos = next_pos
        self._handle_tile_effect()

    def step(self):
        if self.won or self.game_over:
            self.complete_action()
            return

        self.just_flipped = False
        self.just_collided = False
        action_id = self.action.id.value

        if action_id == 0:
            self.complete_action()
            return

        if action_id == 7:
            self._undo()
            self.total_moves += 1
            if self.move_cap > 0 and self.total_moves >= self.move_cap:
                self._lose_life()
            self.complete_action()
            return

        if action_id == 1:
            self._save_state()
            self._process_action(ACT_JUMP)
            self.complete_action()
            return

        if action_id == 2:
            self._save_state()
            self._process_action(ACT_DUCK)
            self.complete_action()
            return

        if action_id == 4:
            self._save_state()
            self._process_action(ACT_MOVE)
            self.complete_action()
            return

        self.complete_action()


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
        [163, 86, 214],
    ],
    dtype=np.uint8,
)


class PuzzleEnvironment:
    _ACTION_MAP: Dict[str, GameAction] = {
        "reset": GameAction.RESET,
        "up": GameAction.ACTION1,
        "down": GameAction.ACTION2,
        "right": GameAction.ACTION4,
        "undo": GameAction.ACTION7,
    }

    _VALID_ACTIONS = list(_ACTION_MAP.keys())

    def __init__(self, seed: int = 0) -> None:
        self._engine: Eg07 = Eg07(seed=seed)
        self._total_turns = 0
        self._done = False
        self._last_action_was_reset = False
        self._game_won = False

    def _build_text_obs(self) -> str:
        e = self._engine
        remaining = max(0, e.move_cap - e.total_moves)
        header = (
            f"Level:{e.level_idx + 1} Lives:{e.lives} "
            f"Moves:{remaining}/{e.move_cap} Pos:{e.pos}/{len(e.path) - 1}"
        )
        tiles = e.path
        tile_chars = {
            TILE_EMPTY: ".",
            TILE_CACTUS: "C",
            TILE_BIRD: "B",
            TILE_GRAVITY: "G",
            TILE_FINISH: "F",
        }
        row = "".join(tile_chars.get(t, "?") for t in tiles)
        pointer = " " * e.pos + "^"
        gravity = "FLIPPED" if e.gravity_flipped else "NORMAL"
        return f"{header} Gravity:{gravity}\n{row}\n{pointer}"

    @staticmethod
    def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
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
        for idx in range(len(ARC_PALETTE)):
            mask = arr == idx
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            rgb[mask] = ARC_PALETTE[idx]
        raw = b""
        for row in range(h):
            raw += b"\x00" + rgb[row].tobytes()
        ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
        png = b"\x89PNG\r\n\x1a\n"
        png += self._png_chunk(b"IHDR", ihdr)
        png += self._png_chunk(b"IDAT", zlib.compress(raw))
        png += self._png_chunk(b"IEND", b"")
        return png

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
                "total_levels": len(LEVEL_DEFS),
                "level_index": e.level_idx,
                "levels_completed": getattr(e, "_score", 0),
                "lives": e.lives,
                "total_moves": e.total_moves,
                "move_cap": e.move_cap,
                "pos": e.pos,
                "gravity_flipped": e.gravity_flipped,
                "game_over": e.game_over,
                "done": done,
                "info": info or {},
            },
        )

    def reset(self) -> GameState:
        e = self._engine
        reset_input = ActionInput(id=GameAction.RESET)
        if self._game_won or self._last_action_was_reset:
            e.perform_action(reset_input)
            e.perform_action(reset_input)
        else:
            e.perform_action(reset_input)
        self._total_turns = 0
        self._done = False
        self._last_action_was_reset = True
        self._game_won = False
        return self._build_game_state()

    def is_done(self) -> bool:
        return self._done

    def get_actions(self) -> List[str]:
        if self._done:
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
            raise ValueError(
                f"Invalid action '{action}'. Must be one of {list(self._ACTION_MAP.keys())}"
            )

        self._last_action_was_reset = False
        self._total_turns += 1

        game_action = self._ACTION_MAP[action]
        info: Dict = {"action": action}

        prev_level = e.level_idx
        action_input = ActionInput(id=game_action)
        frame = e.perform_action(action_input, raw=True)

        game_won = frame and frame.state and frame.state.name == "WIN"
        game_over = frame and frame.state and frame.state.name == "GAME_OVER"
        total_levels = len(LEVEL_DEFS)
        level_completed = game_won or (e.level_idx > prev_level)
        reward = (1.0 / total_levels) if level_completed else 0.0

        if game_won:
            self._done = True
            self._game_won = True
            info["reason"] = "game_complete"
            return StepResult(
                state=self._build_game_state(done=True, info=info),
                reward=reward,
                done=True,
                info=info,
            )

        if game_over:
            self._done = True
            info["reason"] = "death"
            return StepResult(
                state=self._build_game_state(done=True, info=info),
                reward=0.0,
                done=True,
                info=info,
            )

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
        arr = (
            np.array(index_grid, dtype=np.uint8)
            if not isinstance(index_grid, np.ndarray)
            else index_grid.astype(np.uint8)
        )
        h, w = arr.shape[:2]
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for idx in range(len(ARC_PALETTE)):
            mask = arr == idx
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            rgb[mask] = ARC_PALETTE[idx]
        return rgb

    def close(self) -> None:
        self._engine = None


class ArcGameEnv(gym.Env):
    metadata: Dict[str, Any] = {
        "render_modes": ["rgb_array"],
        "render_fps": 5,
    }

    ACTION_LIST: List[str] = ["up", "down", "right", "undo", "reset"]
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

    @staticmethod
    def _resize_nearest(frame: np.ndarray, h: int, w: int) -> np.ndarray:
        src_h, src_w = frame.shape[:2]
        row_idx = (np.arange(h) * src_h // h).astype(int)
        col_idx = (np.arange(w) * src_w // w).astype(int)
        return frame[np.ix_(row_idx, col_idx)].astype(np.uint8)

    def _get_obs(self) -> np.ndarray:
        frame = self._env.render(mode="rgb_array")
        if frame.shape[0] != self.OBS_HEIGHT or frame.shape[1] != self.OBS_WIDTH:
            frame = self._resize_nearest(frame, self.OBS_HEIGHT, self.OBS_WIDTH)
        return frame

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
