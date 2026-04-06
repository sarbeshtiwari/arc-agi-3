import io
import random
import struct
import zlib
from copy import deepcopy
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


WHITE = 0
GREY = 3
DARK_GREY = 4
BLACK = 5
MAGENTA = 6
PINK = 7
RED = 8
LIGHT_BLUE = 10
YELLOW = 11
ORANGE = 12
GREEN = 14
PURPLE = 15

TILE_GROUND = 0
TILE_COIN = 1
TILE_BLOCK = 2
TILE_GAP = 3
TILE_BOMB_LOW = 4
TILE_BOMB_HIGH = 5
TILE_REVERSE = 6
TILE_WALL = 7
TILE_FINISH = 8

FRAME_SIZE = 64
VISIBLE_TILES = 8
TILE_PX = FRAME_SIZE // VISIBLE_TILES
GROUND_TOP = 40
GROUND_H = 8
UNDER_TOP = GROUND_TOP + GROUND_H
WALL_TOP = 22
MAX_COINS_DISPLAY = 5
MAX_BLOCKS_DISPLAY = 5
MAX_LIVES = 3

ACT_JUMP = GameAction.ACTION1
ACT_DUCK = GameAction.ACTION2
ACT_MOVE = GameAction.ACTION4

LEVEL_DEFS = [
    {
        "path": [
            TILE_GROUND, TILE_COIN, TILE_GROUND, TILE_BLOCK, TILE_BOMB_LOW,
            TILE_BLOCK, TILE_GAP, TILE_GROUND, TILE_WALL, TILE_COIN,
            TILE_BLOCK, TILE_GAP, TILE_BOMB_HIGH, TILE_GROUND, TILE_FINISH,
        ],
        "move_cap": 35,
    },
    {
        "path": [
            TILE_GROUND, TILE_COIN, TILE_GROUND, TILE_BLOCK, TILE_BLOCK,
            TILE_GAP, TILE_BOMB_LOW, TILE_GROUND, TILE_COIN, TILE_WALL,
            TILE_BLOCK, TILE_GAP, TILE_BOMB_HIGH, TILE_GROUND, TILE_COIN,
            TILE_REVERSE, TILE_BOMB_LOW, TILE_GROUND, TILE_FINISH,
        ],
        "move_cap": 42,
    },
    {
        "path": [
            TILE_GROUND, TILE_COIN, TILE_GROUND, TILE_BLOCK, TILE_BLOCK,
            TILE_WALL, TILE_GROUND, TILE_COIN, TILE_GAP, TILE_BOMB_LOW,
            TILE_BLOCK, TILE_BLOCK, TILE_REVERSE, TILE_BOMB_HIGH,
            TILE_GROUND, TILE_COIN, TILE_WALL, TILE_GAP, TILE_REVERSE,
            TILE_BOMB_LOW, TILE_BOMB_HIGH, TILE_BLOCK, TILE_GAP,
            TILE_GROUND, TILE_FINISH,
        ],
        "move_cap": 58,
    },
    {
        "path": [
            TILE_GROUND, TILE_COIN, TILE_GROUND, TILE_BLOCK, TILE_GAP,
            TILE_WALL, TILE_GROUND, TILE_COIN, TILE_BLOCK, TILE_REVERSE,
            TILE_BOMB_HIGH, TILE_WALL, TILE_GROUND, TILE_COIN, TILE_BLOCK,
            TILE_GAP, TILE_REVERSE, TILE_BOMB_LOW, TILE_WALL, TILE_BOMB_HIGH,
            TILE_BLOCK, TILE_REVERSE, TILE_BOMB_HIGH, TILE_GROUND, TILE_COIN,
            TILE_GAP, TILE_BOMB_LOW, TILE_GROUND, TILE_FINISH,
        ],
        "move_cap": 65,
    },
]

TOTAL_LEVELS = len(LEVEL_DEFS)

ARC_PALETTE = np.array([
    [255, 255, 255],
    [204, 204, 204],
    [153, 153, 153],
    [102, 102, 102],
    [ 51,  51,  51],
    [  0,   0,   0],
    [229,  58, 163],
    [255, 123, 204],
    [249,  60,  49],
    [ 30, 147, 255],
    [136, 216, 241],
    [255, 220,   0],
    [255, 133,  27],
    [146,  18,  49],
    [ 79, 204,  48],
    [163,  86, 208],
], dtype=np.uint8)

_TILE_CHAR = {
    TILE_GROUND: ".",
    TILE_COIN: "C",
    TILE_BLOCK: "B",
    TILE_GAP: "_",
    TILE_BOMB_LOW: "b",
    TILE_BOMB_HIGH: "^",
    TILE_REVERSE: "R",
    TILE_WALL: "W",
    TILE_FINISH: "F",
}



def _set_pixel(frame, x, y, color):
    if 0 <= x < FRAME_SIZE and 0 <= y < FRAME_SIZE:
        frame[y, x] = color


def _fill_rect(frame, x, y, w, h, color):
    for row_off in range(h):
        for col_off in range(w):
            _set_pixel(frame, x + col_off, y + row_off, color)


def _draw_coin(frame, tile_x):
    cx = tile_x + 3
    cy = GROUND_TOP - 5
    _set_pixel(frame, cx, cy - 2, YELLOW)
    _fill_rect(frame, cx - 1, cy - 1, 3, 1, YELLOW)
    _fill_rect(frame, cx - 2, cy, 5, 1, YELLOW)
    _fill_rect(frame, cx - 1, cy + 1, 3, 1, YELLOW)
    _set_pixel(frame, cx, cy + 2, YELLOW)
    _set_pixel(frame, cx, cy - 1, ORANGE)
    _set_pixel(frame, cx - 1, cy, ORANGE)
    _set_pixel(frame, cx, cy, ORANGE)


def _draw_block_brick(frame, tile_x):
    bx = tile_x + 1
    by = GROUND_TOP - 7
    _fill_rect(frame, bx, by, 6, 3, ORANGE)
    _fill_rect(frame, bx + 1, by + 1, 4, 1, YELLOW)
    _fill_rect(frame, bx, by + 3, 3, 3, ORANGE)
    _fill_rect(frame, bx + 3, by + 3, 3, 3, ORANGE)
    _set_pixel(frame, bx + 1, by + 4, YELLOW)
    _set_pixel(frame, bx + 4, by + 4, YELLOW)


def _draw_filled_gap(frame, tile_x):
    _fill_rect(frame, tile_x, GROUND_TOP, TILE_PX, GROUND_H, ORANGE)
    _fill_rect(frame, tile_x + 1, GROUND_TOP + 1, TILE_PX - 2, 2, YELLOW)


def _draw_bomb_low(frame, tile_x):
    bx = tile_x + 2
    by = GROUND_TOP - 6
    _set_pixel(frame, bx + 2, by - 2, YELLOW)
    _set_pixel(frame, bx + 2, by - 1, ORANGE)
    _fill_rect(frame, bx + 1, by, 2, 1, RED)
    _fill_rect(frame, bx, by + 1, 4, 3, RED)
    _fill_rect(frame, bx + 1, by + 4, 2, 1, RED)


def _draw_bomb_high(frame, tile_x):
    bx = tile_x + 2
    by = GROUND_TOP - 20
    _set_pixel(frame, bx + 2, by - 2, YELLOW)
    _set_pixel(frame, bx + 2, by - 1, ORANGE)
    _fill_rect(frame, bx + 1, by, 2, 1, RED)
    _fill_rect(frame, bx, by + 1, 4, 3, RED)
    _fill_rect(frame, bx + 1, by + 4, 2, 1, RED)


def _draw_reverse_crystal(frame, tile_x):
    mx = tile_x + 1
    my = GROUND_TOP - 8
    _set_pixel(frame, mx + 3, my, MAGENTA)
    _fill_rect(frame, mx + 2, my + 1, 3, 1, MAGENTA)
    _fill_rect(frame, mx + 1, my + 2, 5, 1, MAGENTA)
    _fill_rect(frame, mx + 1, my + 3, 5, 2, PURPLE)
    _fill_rect(frame, mx + 1, my + 5, 5, 1, MAGENTA)
    _fill_rect(frame, mx + 2, my + 6, 3, 1, MAGENTA)
    _set_pixel(frame, mx + 3, my + 7, MAGENTA)


def _draw_wall(frame, tile_x):
    _fill_rect(frame, tile_x, WALL_TOP, TILE_PX, GROUND_TOP - WALL_TOP, DARK_GREY)
    _fill_rect(frame, tile_x + 1, WALL_TOP + 1, TILE_PX - 2, 2, GREY)
    _fill_rect(frame, tile_x + 1, WALL_TOP + 5, TILE_PX - 2, 2, GREY)
    _fill_rect(frame, tile_x + 1, WALL_TOP + 9, TILE_PX - 2, 2, GREY)


def _draw_finish_flag(frame, tile_x):
    pole_x = tile_x + 3
    _fill_rect(frame, pole_x, 26, 1, GROUND_TOP - 26, WHITE)
    _fill_rect(frame, pole_x + 1, 26, 3, 3, GREEN)
    _fill_rect(frame, pole_x + 1, 29, 3, 1, WHITE)


def _draw_runner_walk(frame, tile_x):
    tx = tile_x + 2
    _fill_rect(frame, tx + 1, 35, 2, 2, ORANGE)
    _set_pixel(frame, tx + 2, 35, BLACK)
    _fill_rect(frame, tx, 37, 4, 1, ORANGE)
    _set_pixel(frame, tx, 38, ORANGE)
    _set_pixel(frame, tx + 3, 38, ORANGE)
    _set_pixel(frame, tx, 39, ORANGE)
    _set_pixel(frame, tx + 3, 39, ORANGE)


def _draw_runner_jump(frame, tile_x):
    tx = tile_x + 2
    _fill_rect(frame, tx + 1, 21, 2, 2, ORANGE)
    _set_pixel(frame, tx + 2, 21, BLACK)
    _fill_rect(frame, tx, 23, 4, 1, ORANGE)
    _fill_rect(frame, tx, 24, 4, 1, ORANGE)


def _draw_runner_duck(frame, tile_x):
    tx = tile_x + 2
    _fill_rect(frame, tx + 1, 37, 2, 1, ORANGE)
    _set_pixel(frame, tx + 2, 37, BLACK)
    _fill_rect(frame, tx, 38, 4, 1, ORANGE)
    _fill_rect(frame, tx, 39, 4, 1, ORANGE)


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
    _fill_rect(frame, 0, 8, FRAME_SIZE, GROUND_TOP - 8, LIGHT_BLUE)
    _fill_rect(frame, 0, UNDER_TOP, FRAME_SIZE, FRAME_SIZE - UNDER_TOP, DARK_GREY)
    if env.controls_reversed:
        _fill_rect(frame, 0, 0, FRAME_SIZE, 2, PURPLE)
        _fill_rect(frame, 0, FRAME_SIZE - 2, FRAME_SIZE, 2, PURPLE)


def _render_tiles(frame, env):
    for i in range(VISIBLE_TILES):
        path_idx = env.pos + i
        tile_x = i * TILE_PX
        if path_idx < len(env.path):
            tile = env.path[path_idx]
            if tile == TILE_GAP:
                if path_idx in env.filled_gaps:
                    _draw_filled_gap(frame, tile_x)
                else:
                    _fill_rect(frame, tile_x, GROUND_TOP, TILE_PX, GROUND_H, BLACK)
            else:
                ground_color = GREY
                if tile == TILE_REVERSE:
                    ground_color = PURPLE
                elif tile == TILE_FINISH:
                    ground_color = GREEN
                _fill_rect(frame, tile_x, GROUND_TOP, TILE_PX, GROUND_H, ground_color)
            if tile == TILE_COIN and path_idx not in env.collected_coins:
                _draw_coin(frame, tile_x)
            elif tile == TILE_BLOCK and path_idx not in env.collected_blocks:
                _draw_block_brick(frame, tile_x)
            elif tile == TILE_BOMB_LOW:
                _draw_bomb_low(frame, tile_x)
            elif tile == TILE_BOMB_HIGH:
                _draw_bomb_high(frame, tile_x)
            elif tile == TILE_REVERSE:
                _draw_reverse_crystal(frame, tile_x)
            elif tile == TILE_WALL:
                _draw_wall(frame, tile_x)
            elif tile == TILE_FINISH:
                _draw_finish_flag(frame, tile_x)
        else:
            _fill_rect(frame, tile_x, GROUND_TOP, TILE_PX, GROUND_H, LIGHT_BLUE)


def _render_player(frame, env):
    if env.last_action == ACT_JUMP:
        _draw_runner_jump(frame, 0)
    elif env.last_action == ACT_DUCK:
        _draw_runner_duck(frame, 0)
    else:
        _draw_runner_walk(frame, 0)


def _render_hud(frame, env):
    total_coins = env.total_coins
    cx = 2
    for i in range(total_coins):
        if i < env.coins_collected:
            _set_pixel(frame, cx + 1, 1, YELLOW)
            _fill_rect(frame, cx, 2, 3, 1, YELLOW)
            _set_pixel(frame, cx + 1, 3, YELLOW)
            _set_pixel(frame, cx + 1, 2, ORANGE)
        else:
            _set_pixel(frame, cx + 1, 1, DARK_GREY)
            _fill_rect(frame, cx, 2, 3, 1, DARK_GREY)
            _set_pixel(frame, cx + 1, 3, DARK_GREY)
            _set_pixel(frame, cx + 1, 2, BLACK)
        cx += 5

    bx = FRAME_SIZE - 4
    for i in range(min(env.blocks, MAX_BLOCKS_DISPLAY)):
        _fill_rect(frame, bx - 3, 1, 4, 2, ORANGE)
        _set_pixel(frame, bx - 2, 2, YELLOW)
        _fill_rect(frame, bx - 3, 3, 4, 2, ORANGE)
        _set_pixel(frame, bx - 2, 4, YELLOW)
        bx -= 6

    max_moves = env.max_moves
    rem_moves = max(0, env.rem_moves)
    if max_moves > 0:
        bar_x = 2
        bar_w = FRAME_SIZE - 4
        bar_h = 1
        bar_y = 6
        _fill_rect(frame, bar_x, bar_y, bar_w, bar_h, DARK_GREY)
        filled = bar_w * rem_moves // max_moves
        if filled > 0:
            if rem_moves > max_moves * 0.5:
                fill_color = GREEN
            elif rem_moves > max_moves * 0.25:
                fill_color = YELLOW
            else:
                fill_color = RED
            _fill_rect(frame, bar_x, bar_y, filled, bar_h, fill_color)

    lives_y = FRAME_SIZE - 6
    for i in range(env._max_lives):
        lx = 2 + i * 5
        color = PINK if i < env._lives else DARK_GREY
        _fill_rect(frame, lx, lives_y, 3, 3, color)


def _render_borders(frame, env):
    if env.won:
        _draw_solid_border(frame, GREEN)
    elif env.game_over:
        _draw_solid_border(frame, RED)
    elif env.just_reversed:
        _draw_solid_border(frame, PURPLE)
        _draw_inner_border(frame, MAGENTA)
    elif env.just_hit:
        _draw_dashed_border(frame, ORANGE)
    elif env.just_collided:
        _draw_dashed_border(frame, RED)


class Eg08Hud(RenderableUserDisplay):
    def __init__(self):
        super().__init__()
        self.env = None

    def render_interface(self, frame):
        env = self.env
        if env is None:
            return frame
        _render_background(frame, env)
        _render_tiles(frame, env)
        _render_player(frame, env)
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



class Eg08(ARCBaseGame):
    def __init__(self, seed=0):
        self._rng = random.Random(seed)
        self.hud = Eg08Hud()
        self.hud.env = self
        self.level_idx = 0
        self.path = []
        self.pos = 0
        self.coins_collected = 0
        self.collected_coins = set()
        self.total_coins = 0
        self.blocks = 0
        self.collected_blocks = set()
        self.filled_gaps = set()
        self.controls_reversed = False
        self.total_moves = 0
        self.move_cap = 0
        self.max_moves = 0
        self.rem_moves = 0
        self.won = False
        self.game_over = False
        self.last_action = None
        self.just_collected = False
        self.just_built = False
        self.just_reversed = False
        self.just_hit = False
        self.just_collided = False
        self.played_since_level_set = False
        self.level_desync = False
        self._first_load = True
        self.consecutive_resets = 0

        self._max_lives = MAX_LIVES
        self._lives = MAX_LIVES
        self._engine_snapshots: List[dict] = []
        self._game_over = False

        super().__init__(
            "eg08",
            levels,
            Camera(0, 0, 64, 64, BLACK, BLACK, [self.hud]),
            available_actions=[0, 1, 2, 4, 5, 7],
        )

    def _init_level(self, idx):
        cfg = LEVEL_DEFS[idx]
        self.level_idx = idx
        self.path = list(cfg["path"])

        interesting = {TILE_COIN, TILE_BLOCK, TILE_GAP, TILE_BOMB_LOW,
                       TILE_BOMB_HIGH, TILE_WALL, TILE_REVERSE}
        max_spawn = len(self.path) // 2
        candidates = [0]
        for i in range(1, min(max_spawn + 1, len(self.path) - 1)):
            if self.path[i] != TILE_GROUND:
                continue
            prev_tile = self.path[i - 1]
            next_tile = self.path[i + 1] if i + 1 < len(self.path) else None
            if next_tile in interesting or prev_tile in interesting:
                candidates.append(i)
        candidates = sorted(set(candidates))
        self.pos = self._rng.choice(candidates)

        self.coins_collected = 0
        self.collected_coins = set()
        self.total_coins = sum(1 for t in self.path if t == TILE_COIN)
        self.blocks = 0
        self.collected_blocks = set()
        self.filled_gaps = set()
        for i in range(self.pos):
            tile = self.path[i]
            if tile == TILE_COIN:
                self.collected_coins.add(i)
                self.coins_collected += 1
            elif tile == TILE_BLOCK:
                self.collected_blocks.add(i)
                self.blocks += 1
            elif tile == TILE_GAP:
                if self.blocks > 0:
                    self.blocks -= 1
                    self.filled_gaps.add(i)
        self.controls_reversed = False
        self.total_moves = 0
        self.move_cap = cfg["move_cap"]
        self.max_moves = cfg["move_cap"]
        self.rem_moves = cfg["move_cap"]
        self.won = False
        self.game_over = False
        self.last_action = None
        self.just_collected = False
        self.just_built = False
        self.just_reversed = False
        self.just_hit = False
        self.just_collided = False
        self.played_since_level_set = False

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
        self._engine_snapshots.clear()
        self._game_over = False
        self._lives = self._max_lives

    def _engine_save_snapshot(self):
        self._engine_snapshots.append({
            "pos": self.pos,
            "coins_collected": self.coins_collected,
            "collected_coins": set(self.collected_coins),
            "blocks": self.blocks,
            "collected_blocks": set(self.collected_blocks),
            "filled_gaps": set(self.filled_gaps),
            "controls_reversed": self.controls_reversed,
            "total_moves": self.total_moves,
            "last_action": self.last_action,
            "won": self.won,
            "game_over": self.game_over,
            "rem_moves": self.rem_moves,
        })

    def _engine_restore_snapshot(self):
        if not self._engine_snapshots:
            return
        snap = self._engine_snapshots.pop()
        self.pos = snap["pos"]
        self.coins_collected = snap["coins_collected"]
        self.collected_coins = snap["collected_coins"]
        self.blocks = snap["blocks"]
        self.collected_blocks = snap["collected_blocks"]
        self.filled_gaps = snap["filled_gaps"]
        self.controls_reversed = snap["controls_reversed"]
        self.total_moves = snap["total_moves"]
        self.last_action = snap["last_action"]
        self.won = snap["won"]
        self.game_over = snap["game_over"]

    def _reset_current_level(self):
        self._init_level(self.level_idx)
        self._engine_snapshots.clear()

    def full_reset(self):
        self._current_level_index = 0
        self.on_set_level(levels[0])
        self._game_over = False
        self._lives = self._max_lives

    def _handle_death(self):
        self._lives -= 1
        self.consecutive_resets = 0
        if self._lives <= 0:
            self._game_over = True
            self.game_over = True
            self.lose()
            return True
        self._reset_current_level()
        return True

    def handle_reset(self):
        if self._state.value == "WIN":
            self.full_reset()
            return

        if self._game_over or self._state.value == "GAME_OVER":
            self._reset_current_level()
            self._lives = self._max_lives
            self._game_over = False
            return

        self.consecutive_resets += 1
        if self.consecutive_resets >= 2:
            self.consecutive_resets = 0
            self.set_level(0)
        else:
            self._reset_current_level()
            if self._lives <= 0:
                self._lives = self._max_lives

    def _get_effective_tile(self, path_idx):
        if path_idx >= len(self.path):
            return TILE_GROUND
        return self.path[path_idx]

    def _required_action(self, tile):
        if tile in (TILE_BOMB_LOW, TILE_WALL):
            return ACT_JUMP
        if tile == TILE_BOMB_HIGH:
            return ACT_DUCK
        return ACT_MOVE

    def _handle_tile_effect(self):
        tile = self.path[self.pos]

        if tile == TILE_COIN:
            self.coins_collected += 1
            self.collected_coins.add(self.pos)
            self.just_collected = True

        elif tile == TILE_BLOCK:
            self.blocks += 1
            self.collected_blocks.add(self.pos)
            self.just_collected = True

        elif tile == TILE_GAP:
            if self.blocks <= 0:
                self.just_collided = True
                self._handle_death()
                return
            self.blocks -= 1
            self.filled_gaps.add(self.pos)
            self.just_built = True

        elif tile == TILE_REVERSE:
            self.controls_reversed = not self.controls_reversed
            self.just_reversed = True

        elif tile == TILE_FINISH:
            self.won = True
            if self.level_idx >= TOTAL_LEVELS - 1:
                self.win()
            elif self.level_desync:
                self.consecutive_resets = 0
                self._init_level(self.level_idx + 1)
                self._lives = self._max_lives
                self._engine_snapshots.clear()
            else:
                self.next_level()

    def _process_action(self, action_id):
        if self.won or self.game_over:
            return

        self.played_since_level_set = True

        if self.controls_reversed:
            if action_id == ACT_JUMP:
                action_id = ACT_DUCK
            elif action_id == ACT_DUCK:
                action_id = ACT_JUMP

        next_pos = self.pos + 1
        if next_pos >= len(self.path):
            return

        self.total_moves += 1

        effective_tile = self._get_effective_tile(next_pos)
        required = self._required_action(effective_tile)

        if action_id != required:
            self.just_collided = True
            self._handle_death()
            return

        self.last_action = action_id
        self.pos = next_pos
        self._handle_tile_effect()

    def step(self):
        if self.won or self.game_over:
            self.complete_action()
            return

        self.just_collected = False
        self.just_built = False
        self.just_reversed = False
        self.just_hit = False
        self.just_collided = False
        action_id = self.action.id

        if action_id == GameAction.RESET:
            self.complete_action()
            return

        self.consecutive_resets = 0

        if action_id == GameAction.ACTION7:
            self.rem_moves -= 1
            if self._engine_snapshots:
                self._engine_restore_snapshot()
            if self.rem_moves <= 0 and not self.won:
                self._handle_death()
            self.complete_action()
            return

        if action_id == GameAction.ACTION5:
            self.rem_moves -= 1
            if self.rem_moves <= 0 and not self.won and not self.game_over:
                self.just_collided = True
                self._handle_death()
            self.complete_action()
            return

        self._engine_save_snapshot()
        self.rem_moves -= 1

        if action_id == GameAction.ACTION1:
            self._process_action(action_id)
        elif action_id == GameAction.ACTION2:
            self._process_action(action_id)
        elif action_id == GameAction.ACTION4:
            self._process_action(action_id)

        if self.rem_moves <= 0 and not self.won and not self.game_over:
            self.just_collided = True
            self._handle_death()

        self.complete_action()



class PuzzleEnvironment:

    _ACTION_MAP: Dict[str, GameAction] = {
        "reset":  GameAction.RESET,
        "up":     GameAction.ACTION1,
        "down":   GameAction.ACTION2,
        "right":  GameAction.ACTION4,
        "select": GameAction.ACTION5,
        "undo":   GameAction.ACTION7,
    }

    _VALID_ACTIONS = list(_ACTION_MAP.keys())

    def __init__(self, seed: int = 0) -> None:
        self._engine = Eg08(seed=seed)
        self._total_turns = 0
        self._done = False
        self._game_won = False
        self._game_over = False
        self._last_action_was_reset = False
        self._rng = random.Random(seed)

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

        path_chars = []
        for i, tile in enumerate(e.path):
            ch = _TILE_CHAR.get(tile, "?")
            if tile == TILE_GAP and i in e.filled_gaps:
                ch = "="
            if tile == TILE_COIN and i in e.collected_coins:
                ch = "."
            if tile == TILE_BLOCK and i in e.collected_blocks:
                ch = "."
            if i == e.pos:
                path_chars.append("[" + ch + "]")
            else:
                path_chars.append(" " + ch + " ")

        header = (
            f"Level {e.level_idx + 1}/{TOTAL_LEVELS}"
            f" | Moves: {e.rem_moves}/{e.max_moves}"
            f" | Lives: {e._lives}"
            f" | Coins: {e.coins_collected}"
            f" | Blocks: {e.blocks}"
        )
        controls = "reversed" if e.controls_reversed else "normal"
        rules = (
            f"Controls: {controls}. "
            "Up=jump (over bombs/walls), Down=duck (high bombs), Right=move forward. "
            "Undo restores last state but costs a move."
        )
        return header + "\n" + rules + "\nPath:\n" + "".join(path_chars)

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
                "level_index": e.level_index,
                "levels_completed": getattr(e, "_score", 0),
                "game_over": self._game_over,
                "done": done,
                "info": {},
            },
        )

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
        self._game_over = False

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
        info: Dict[str, Any] = {"action": action}

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

        if game_over or e._game_over:
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
