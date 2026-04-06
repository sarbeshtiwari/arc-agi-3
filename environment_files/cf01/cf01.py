from __future__ import annotations

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
    Level,
    Sprite,
)
from arcengine import GameState as EngineState


@dataclass
class GameState:
    text_observation: str
    image_observation: bytes | None
    valid_actions: list[str] | None
    turn: int
    metadata: dict = field(default_factory=dict)


@dataclass
class StepResult:
    state: GameState
    reward: float
    done: bool
    info: dict = field(default_factory=dict)


WHITE = 0
OFF_WHITE = 1
NEUTRAL_LIGHT = 2
NEUTRAL = 3
OFF_BLACK = 4
BLACK = 5
MAGENTA = 6
MAGENTA_LIGHT = 7
RED = 8
BLUE = 9
BLUE_LIGHT = 10
YELLOW = 11
ORANGE = 12
MAROON = 13
GREEN = 14
PURPLE = 15

BACKGROUND_COLOR = BLACK
BORDER_COLOR = NEUTRAL
HIDDEN_COLOR = OFF_BLACK
CURSOR_COLOR = WHITE
MOVES_BAR_COLOR = GREEN
MOVES_BAR_EMPTY = NEUTRAL

GRID_SIZE = 64
BORDER_WIDTH = 2
MOVES_BAR_Y = 60
MOVES_BAR_HEIGHT = 2

MISMATCH_DELAY_TURNS = 2

MAX_LIVES = 3
LIFE_COLOR = RED
LIFE_EMPTY_COLOR = NEUTRAL

BOMB_SENTINEL = -1
BOMB_COLOR = MAROON
BOMB_MOVE_PENALTY = 5

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


def _make_png(index_grid: np.ndarray) -> bytes:
    h, w = index_grid.shape
    raw_rows = bytearray()
    for y in range(h):
        raw_rows.append(0)
        for x in range(w):
            idx = int(index_grid[y, x])
            if idx < len(ARC_PALETTE):
                r, g, b = (
                    int(ARC_PALETTE[idx][0]),
                    int(ARC_PALETTE[idx][1]),
                    int(ARC_PALETTE[idx][2]),
                )
            else:
                r, g, b = 0, 0, 0
            raw_rows.append(r)
            raw_rows.append(g)
            raw_rows.append(b)

    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        c = chunk_type + data
        return (
            struct.pack(">I", len(data))
            + c
            + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        )

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return (
        sig
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(bytes(raw_rows)))
        + _chunk(b"IEND", b"")
    )


TILE_COLORS = [
    RED,
    BLUE,
    YELLOW,
    GREEN,
    PURPLE,
    ORANGE,
    MAGENTA,
    BLUE_LIGHT,
    MAROON,
    MAGENTA_LIGHT,
    WHITE,
    NEUTRAL_LIGHT,
]

LEVEL_1_LAYOUT = [
    [3, 4, 2, 3],
    [7, 5, 6, 4],
    [0, 1, 6, 7],
    [2, 5, 0, 1],
]

LEVEL_2_LAYOUT = [
    [4, 6, 2, 1, 4, 9],
    [3, 0, 11, 1, 3, 5],
    [5, -1, 4, 10, 8, 5],
    [3, 0, 0, 5, 6, 9],
    [1, 4, 2, 10, 1, 11],
    [2, 2, -1, 8, 0, 3],
]

LEVEL_3_LAYOUT = [
    [3, 11, 1, 5, 8, 10, 1],
    [3, 5, 9, 4, 4, 8, 10],
    [2, 3, 1, 10, 11, 2, 2],
    [5, 9, 11, 7, 0, 0, 0],
    [0, 9, 3, 6, 1, 6, 7],
    [4, 0, 6, 10, 2, 1, 1],
    [6, 2, 5, 9, 3, 2, 4],
    [7, 7, 8, 11, 0, 3, 8],
]

LEVEL_4_LAYOUT = [
    [11, 0, 3, 11, 10, 4, 9, 5],
    [11, 4, 8, 9, 9, 3, 8, 10],
    [6, 8, 2, 10, 5, 1, -1, 0],
    [1, 3, 0, 3, 2, 4, 3, 0],
    [5, 0, 6, 2, 10, 1, 2, 6],
    [5, 2, -1, 4, 5, 6, 1, -1],
    [1, 1, 6, 2, 5, 4, 9, 3],
    [6, 4, 7, 7, 8, 11, 0, -1],
]

LEVEL_CONFIGS = [
    {
        "tile_grid": (4, 4),
        "tile_size": 13,
        "move_mult": 20.0,
        "layout": LEVEL_1_LAYOUT,
        "num_pairs": 8,
        "rotate_after": 0,
        "cursor_spawns": [(0, 0), (3, 0), (0, 3), (3, 3)],
    },
    {
        "tile_grid": (6, 6),
        "tile_size": 9,
        "move_mult": 20.0,
        "layout": LEVEL_2_LAYOUT,
        "num_pairs": 17,
        "rotate_after": 0,
        "cursor_spawns": [(0, 0), (5, 0), (0, 5), (5, 5)],
    },
    {
        "tile_grid": (7, 8),
        "tile_size": 6,
        "move_mult": 20.0,
        "layout": LEVEL_3_LAYOUT,
        "num_pairs": 28,
        "rotate_after": 4,
        "cursor_spawns": [(0, 0), (6, 0), (0, 7), (6, 7)],
    },
    {
        "tile_grid": (8, 8),
        "tile_size": 6,
        "move_mult": 20.0,
        "layout": LEVEL_4_LAYOUT,
        "num_pairs": 30,
        "rotate_after": 3,
        "cursor_spawns": [(0, 0), (7, 0), (0, 7), (7, 7)],
    },
]

levels = [Level(sprites=[], grid_size=(GRID_SIZE, GRID_SIZE)) for _ in LEVEL_CONFIGS]


class Cf01(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._tile_grid: tuple[int, int] = (6, 6)
        self._tile_size: int = 8

        self._tile_colors: list[list[int]] = []
        self._tile_revealed: list[list[bool]] = []
        self._tile_matched: list[list[bool]] = []
        self._tile_is_bomb: list[list[bool]] = []

        self._revealed_tiles: list[tuple[int, int]] = []

        self._matched_pairs: int = 0
        self._total_pairs: int = 0
        self._moves_remaining: int = 0
        self._max_moves: int = 0

        self._cursor: tuple[int, int] = (0, 0)
        self._cursor_active: bool = False

        self._mismatch_pending: bool = False
        self._mismatch_timer: int = 0
        self._mismatch_count: int = 0
        self._rotate_after: int = 0

        self._game_won: bool = False
        self._game_lost: bool = False
        self._lives: int = MAX_LIVES
        self._undo_stack: List[dict] = []
        self._variation: int = 0

        camera = Camera(
            x=0,
            y=0,
            background=BACKGROUND_COLOR,
            letter_box=BORDER_COLOR,
            width=GRID_SIZE,
            height=GRID_SIZE,
            interfaces=[],
        )

        super().__init__(
            "cf01",
            levels,
            camera,
            available_actions=[0, 1, 2, 3, 4, 5, 7],
        )

    def get_actions(self) -> list[int]:
        return self._available_actions

    def on_set_level(self, level: Level) -> None:
        config = LEVEL_CONFIGS[min(self.level_index, len(LEVEL_CONFIGS) - 1)]

        self._tile_grid = config["tile_grid"]
        self._tile_size = config["tile_size"]
        self._total_pairs = config["num_pairs"]
        self._rotate_after = config.get("rotate_after", 0)

        cols, rows = self._tile_grid

        move_mult = config.get("move_mult", 2.0)
        self._max_moves = int(self._total_pairs * move_mult)
        self._moves_remaining = self._max_moves

        self._matched_pairs = 0
        self._revealed_tiles = []
        self._cursor_active = True
        self._mismatch_pending = False
        self._mismatch_timer = 0
        self._mismatch_count = 0
        self._game_won = False
        self._game_lost = False
        self._undo_stack = []

        self._tile_colors = [[0 for _ in range(rows)] for _ in range(cols)]
        self._tile_revealed = [[False for _ in range(rows)] for _ in range(cols)]
        self._tile_matched = [[False for _ in range(rows)] for _ in range(cols)]
        self._tile_is_bomb = [[False for _ in range(rows)] for _ in range(cols)]

        self._load_static_layout(config)

        spawn_positions = config.get("cursor_spawns", [(0, 0)])
        idx = self._variation % len(spawn_positions)
        self._cursor = spawn_positions[idx]
        self._variation += 1

        self._render()

    def _load_static_layout(self, config: dict) -> None:
        layout = config["layout"]
        cols, rows = self._tile_grid

        for y in range(rows):
            for x in range(cols):
                if y < len(layout) and x < len(layout[y]):
                    color_id = layout[y][x]
                    if color_id == BOMB_SENTINEL:
                        self._tile_is_bomb[x][y] = True
                        self._tile_colors[x][y] = BOMB_COLOR
                    else:
                        color = TILE_COLORS[color_id % len(TILE_COLORS)]
                        self._tile_colors[x][y] = color

    def _get_tile_area(self) -> tuple[int, int, int, int]:
        cols, rows = self._tile_grid
        total_width = cols * self._tile_size
        total_height = rows * self._tile_size

        min_pad = 3
        usable_h = MOVES_BAR_Y - BORDER_WIDTH - min_pad
        start_x = (GRID_SIZE - total_width) // 2
        start_y = (usable_h - total_height) // 2 + BORDER_WIDTH

        return (start_x, start_y, total_width, total_height)

    def _tile_to_pixel(self, tx: int, ty: int) -> tuple[int, int]:
        start_x, start_y, _, _ = self._get_tile_area()
        return (start_x + tx * self._tile_size, start_y + ty * self._tile_size)

    def _clear_mismatch(self) -> None:
        if self._mismatch_pending:
            for tx, ty in self._revealed_tiles:
                self._tile_revealed[tx][ty] = False
            self._revealed_tiles = []
            self._mismatch_pending = False
            self._mismatch_timer = 0

    def _handle_tile_click(self, tx: int, ty: int) -> None:
        if self._game_won or self._game_lost:
            return

        if self._mismatch_pending:
            return

        if self._tile_matched[tx][ty] or self._tile_revealed[tx][ty]:
            return

        if self._tile_is_bomb[tx][ty]:
            self._tile_revealed[tx][ty] = True
            self._revealed_tiles.append((tx, ty))
            self._mismatch_pending = True
            self._mismatch_timer = 0
            self._moves_remaining = max(0, self._moves_remaining - BOMB_MOVE_PENALTY)
            if self._moves_remaining <= 0 and not self._game_won:
                self._lose_life()
            return

        if self._moves_remaining > 0:
            self._moves_remaining -= 1

        self._tile_revealed[tx][ty] = True
        self._revealed_tiles.append((tx, ty))

        if len(self._revealed_tiles) == 2:
            self._check_pair_match()

        if self._moves_remaining <= 0 and not self._game_won:
            self._lose_life()

    def _lose_life(self) -> None:
        self._lives -= 1
        self._game_lost = True

    def _check_pair_match(self) -> None:
        if len(self._revealed_tiles) != 2:
            return

        t1 = self._revealed_tiles[0]
        t2 = self._revealed_tiles[1]

        color1 = self._tile_colors[t1[0]][t1[1]]
        color2 = self._tile_colors[t2[0]][t2[1]]

        if color1 == color2:
            for tile_x, tile_y in self._revealed_tiles:
                self._tile_matched[tile_x][tile_y] = True
            self._matched_pairs += 1
            self._revealed_tiles = []

            if self._matched_pairs >= self._total_pairs:
                self._game_won = True
        else:
            self._mismatch_pending = True
            self._mismatch_timer = 0
            self._mismatch_count += 1

            if (
                self._rotate_after > 0
                and self._mismatch_count % self._rotate_after == 0
            ):
                self._rotate_unmatched()

    def _rotate_unmatched(self) -> None:
        cols, rows = self._tile_grid
        positions: list[tuple[int, int]] = []
        colors: list[int] = []
        for x in range(cols):
            for y in range(rows):
                if (
                    not self._tile_matched[x][y]
                    and not self._tile_is_bomb[x][y]
                    and (x, y) not in self._revealed_tiles
                ):
                    positions.append((x, y))
                    colors.append(self._tile_colors[x][y])
        if len(colors) > 1:
            shift = len(colors) // 2
            colors = colors[shift:] + colors[:shift]
        for i, (x, y) in enumerate(positions):
            self._tile_colors[x][y] = colors[i]

    def _render(self) -> None:
        self.current_level.remove_all_sprites()

        grid = [[BACKGROUND_COLOR] * GRID_SIZE for _ in range(GRID_SIZE)]

        for x in range(GRID_SIZE):
            grid[0][x] = BORDER_COLOR
            grid[GRID_SIZE - 1][x] = BORDER_COLOR
        for y in range(GRID_SIZE):
            grid[y][0] = BORDER_COLOR
            grid[y][GRID_SIZE - 1] = BORDER_COLOR

        cols, rows = self._tile_grid
        for ty in range(rows):
            for tx in range(cols):
                px, py = self._tile_to_pixel(tx, ty)

                if self._tile_matched[tx][ty] or self._tile_revealed[tx][ty]:
                    color = self._tile_colors[tx][ty]
                else:
                    color = HIDDEN_COLOR

                for dy in range(self._tile_size - 1):
                    for dx in range(self._tile_size - 1):
                        if 0 <= py + dy < GRID_SIZE and 0 <= px + dx < GRID_SIZE:
                            grid[py + dy][px + dx] = color

                if self._tile_is_bomb[tx][ty] and (
                    self._tile_matched[tx][ty] or self._tile_revealed[tx][ty]
                ):
                    tw = self._tile_size - 1
                    th = self._tile_size - 1
                    for i in range(min(tw, th)):
                        x1 = (
                            px + (i * (tw - 1)) // (min(tw, th) - 1)
                            if min(tw, th) > 1
                            else px
                        )
                        y1 = (
                            py + (i * (th - 1)) // (min(tw, th) - 1)
                            if min(tw, th) > 1
                            else py
                        )
                        if 0 <= y1 < GRID_SIZE and 0 <= x1 < GRID_SIZE:
                            grid[y1][x1] = BLACK
                        x2 = (
                            px + tw - 1 - (i * (tw - 1)) // (min(tw, th) - 1)
                            if min(tw, th) > 1
                            else px
                        )
                        if 0 <= y1 < GRID_SIZE and 0 <= x2 < GRID_SIZE:
                            grid[y1][x2] = BLACK

        if self._cursor_active:
            self._draw_cursor_brackets(grid)

        self._draw_moves_bar(grid, BORDER_COLOR)

        sprite = Sprite(pixels=grid, name="display", visible=True, collidable=False)
        sprite.set_position(0, 0)
        self.current_level.add_sprite(sprite)

    def _draw_cursor_brackets(self, grid: list[list[int]]) -> None:
        cx, cy = self._cursor
        px, py = self._tile_to_pixel(cx, cy)

        tile_w = self._tile_size - 1
        tile_h = self._tile_size - 1
        bracket_len = max(2, tile_w // 2)

        for i in range(bracket_len):
            if 0 <= py < GRID_SIZE and 0 <= px + i < GRID_SIZE:
                grid[py][px + i] = CURSOR_COLOR
            if 0 <= py + i < GRID_SIZE and 0 <= px < GRID_SIZE:
                grid[py + i][px] = CURSOR_COLOR

        for i in range(bracket_len):
            right_x = px + tile_w - 1 - i
            if 0 <= py < GRID_SIZE and 0 <= right_x < GRID_SIZE:
                grid[py][right_x] = CURSOR_COLOR
            if 0 <= py + i < GRID_SIZE and 0 <= px + tile_w - 1 < GRID_SIZE:
                grid[py + i][px + tile_w - 1] = CURSOR_COLOR

        for i in range(bracket_len):
            if 0 <= py + tile_h - 1 < GRID_SIZE and 0 <= px + i < GRID_SIZE:
                grid[py + tile_h - 1][px + i] = CURSOR_COLOR
            bottom_y = py + tile_h - 1 - i
            if 0 <= bottom_y < GRID_SIZE and 0 <= px < GRID_SIZE:
                grid[bottom_y][px] = CURSOR_COLOR

        for i in range(bracket_len):
            right_x = px + tile_w - 1 - i
            if 0 <= py + tile_h - 1 < GRID_SIZE and 0 <= right_x < GRID_SIZE:
                grid[py + tile_h - 1][right_x] = CURSOR_COLOR
            bottom_y = py + tile_h - 1 - i
            if 0 <= bottom_y < GRID_SIZE and 0 <= px + tile_w - 1 < GRID_SIZE:
                grid[bottom_y][px + tile_w - 1] = CURSOR_COLOR

    def _draw_moves_bar(self, grid: list[list[int]], border_color: int) -> None:
        bar_x = 4
        pip_area = MAX_LIVES * 5
        bar_width = GRID_SIZE - 8 - pip_area
        bar_y = MOVES_BAR_Y

        if self._max_moves > 0:
            fill_pct = self._moves_remaining / self._max_moves
        else:
            fill_pct = 0
        fill_width = int(bar_width * fill_pct)

        for dy in range(MOVES_BAR_HEIGHT):
            for dx in range(bar_width):
                draw_px = bar_x + dx
                draw_py = bar_y + dy
                if 0 <= draw_py < GRID_SIZE and 0 <= draw_px < GRID_SIZE:
                    if dx < fill_width:
                        grid[draw_py][draw_px] = MOVES_BAR_COLOR
                    else:
                        grid[draw_py][draw_px] = MOVES_BAR_EMPTY

        pip_x0 = GRID_SIZE - 2 - MAX_LIVES * 5
        for i in range(MAX_LIVES):
            cx = pip_x0 + i * 5
            c = LIFE_COLOR if i < self._lives else LIFE_EMPTY_COLOR
            for dy in range(MOVES_BAR_HEIGHT):
                for dx in range(3):
                    px = cx + dx
                    py = bar_y + dy
                    if 0 <= px < GRID_SIZE and 0 <= py < GRID_SIZE:
                        grid[py][px] = c

    def _move_cursor(self, dx: int, dy: int) -> None:
        cols, rows = self._tile_grid
        new_x = max(0, min(cols - 1, self._cursor[0] + dx))
        new_y = max(0, min(rows - 1, self._cursor[1] + dy))
        if (new_x, new_y) != self._cursor:
            if self._moves_remaining > 0:
                self._moves_remaining -= 1
            self._cursor = (new_x, new_y)
            if self._moves_remaining <= 0 and not self._game_won:
                self._lose_life()
                return
        self._cursor_active = True

    def _select_at_cursor(self) -> None:
        tx, ty = self._cursor
        self._handle_tile_click(tx, ty)

    def _save_snapshot(self) -> None:
        self._undo_stack.append(
            {
                "tile_revealed": [
                    [
                        self._tile_revealed[x][y]
                        for y in range(len(self._tile_revealed[x]))
                    ]
                    for x in range(len(self._tile_revealed))
                ],
                "tile_matched": [
                    [
                        self._tile_matched[x][y]
                        for y in range(len(self._tile_matched[x]))
                    ]
                    for x in range(len(self._tile_matched))
                ],
                "tile_colors": [
                    [self._tile_colors[x][y] for y in range(len(self._tile_colors[x]))]
                    for x in range(len(self._tile_colors))
                ],
                "tile_is_bomb": [
                    [
                        self._tile_is_bomb[x][y]
                        for y in range(len(self._tile_is_bomb[x]))
                    ]
                    for x in range(len(self._tile_is_bomb))
                ],
                "revealed_tiles": list(self._revealed_tiles),
                "matched_pairs": self._matched_pairs,
                "moves_remaining": self._moves_remaining,
                "cursor": self._cursor,
                "cursor_active": self._cursor_active,
                "mismatch_pending": self._mismatch_pending,
                "mismatch_timer": self._mismatch_timer,
                "mismatch_count": self._mismatch_count,
                "game_won": self._game_won,
                "game_lost": self._game_lost,
                "lives": self._lives,
            }
        )
        if len(self._undo_stack) > 50:
            self._undo_stack.pop(0)

    def _apply_undo(self) -> None:
        if not self._undo_stack:
            return
        snap = self._undo_stack.pop()
        self._tile_revealed = snap["tile_revealed"]
        self._tile_matched = snap["tile_matched"]
        self._tile_colors = snap["tile_colors"]
        self._tile_is_bomb = snap["tile_is_bomb"]
        self._revealed_tiles = snap["revealed_tiles"]
        self._matched_pairs = snap["matched_pairs"]
        self._moves_remaining = snap["moves_remaining"]
        self._cursor = snap["cursor"]
        self._cursor_active = snap["cursor_active"]
        self._mismatch_pending = snap["mismatch_pending"]
        self._mismatch_timer = snap["mismatch_timer"]
        self._mismatch_count = snap["mismatch_count"]
        self._game_won = snap["game_won"]
        self._game_lost = snap["game_lost"]
        self._lives = snap["lives"]

    def handle_reset(self) -> None:
        had_progress = (
            self._state == EngineState.GAME_OVER
            or self._moves_remaining < self._max_moves
            or self._lives < MAX_LIVES
            or self._matched_pairs > 0
        )
        self._lives = MAX_LIVES
        if self._state == EngineState.WIN:
            self.full_reset()
        elif had_progress:
            self.level_reset()
        else:
            self.full_reset()

    def _process_outcome(self) -> None:
        if self._game_won:
            self._lives = MAX_LIVES
            self.next_level()
        elif self._game_lost:
            if self._lives <= 0:
                self.lose()
            else:
                self.level_reset()
        self.complete_action()

    def _handle_action(self) -> None:
        action_id = self.action.id
        if action_id == GameAction.ACTION7:
            if self._undo_stack and self._moves_remaining > 0:
                charged_moves = self._moves_remaining - 1
                self._apply_undo()
                self._moves_remaining = charged_moves
            elif self._moves_remaining > 0:
                self._moves_remaining -= 1
            if self._moves_remaining <= 0 and not self._game_won:
                self._lose_life()
            self._render()
            return
        self._save_snapshot()
        if action_id == GameAction.ACTION1:
            self._move_cursor(0, -1)
        elif action_id == GameAction.ACTION2:
            self._move_cursor(0, 1)
        elif action_id == GameAction.ACTION3:
            self._move_cursor(-1, 0)
        elif action_id == GameAction.ACTION4:
            self._move_cursor(1, 0)
        elif action_id == GameAction.ACTION5:
            self._select_at_cursor()
        self._render()

    def step(self) -> None:
        if self._mismatch_pending:
            self._mismatch_timer += 1
            if self._mismatch_timer >= MISMATCH_DELAY_TURNS:
                self._clear_mismatch()
                self._render()

        self._handle_action()

        if self._game_won or self._game_lost:
            self._process_outcome()
            return

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

    def __init__(self, seed: int = 0) -> None:
        self._engine = Cf01(seed=seed)
        self._total_turns = 0
        self._done = False
        self._game_won = False
        self._game_over = False
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
        self._game_over = False

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

        frame = e.perform_action(ActionInput(id=game_action), raw=True)

        state_name = frame.state.name if frame and frame.state else ""
        game_won = state_name == "WIN"
        game_over = state_name == "GAME_OVER"

        total_levels = len(e._levels)
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

        if game_over:
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

    _COLOR_CHAR: Dict[int, str] = {
        RED: "R",
        BLUE: "B",
        YELLOW: "Y",
        GREEN: "G",
        PURPLE: "P",
        ORANGE: "O",
        MAGENTA: "M",
        BLUE_LIGHT: "b",
        MAROON: "m",
        MAGENTA_LIGHT: "k",
        WHITE: "W",
        NEUTRAL_LIGHT: "n",
    }

    def _build_text_observation(self) -> str:
        e = self._engine
        cols, rows = e._tile_grid
        header_parts = [
            f"level:{e.level_index + 1}/{len(e._levels)}",
            f"moves:{e._moves_remaining}/{e._max_moves}",
            f"lives:{e._lives}/{MAX_LIVES}",
            f"pairs:{e._matched_pairs}/{e._total_pairs}",
        ]
        if e._rotate_after > 0:
            until_rotate = e._rotate_after - (e._mismatch_count % e._rotate_after)
            header_parts.append(f"rotate_in:{until_rotate}")
        header = " ".join(header_parts)

        grid_lines: list[str] = []
        for y in range(rows):
            row_cells: list[str] = []
            for x in range(cols):
                is_cursor = (x, y) == e._cursor
                if e._tile_is_bomb[x][y] and (
                    e._tile_matched[x][y] or e._tile_revealed[x][y]
                ):
                    cell = "X"
                elif e._tile_matched[x][y]:
                    cell = "="
                elif e._tile_revealed[x][y]:
                    cell = self._COLOR_CHAR.get(e._tile_colors[x][y], "?")
                else:
                    cell = "."
                if is_cursor:
                    cell = f"[{cell}]"
                else:
                    cell = f" {cell} "
                row_cells.append(cell)
            grid_lines.append("".join(row_cells))

        if e._mismatch_pending:
            grid_lines.append("mismatch_pending")

        return header + "\n" + "\n".join(grid_lines)

    def _build_game_state(self, done: bool = False) -> GameState:
        e = self._engine
        valid_actions = self.get_actions() if not done else None
        index_grid = e.camera.render(e.current_level.get_sprites())
        return GameState(
            text_observation=self._build_text_observation(),
            image_observation=_make_png(index_grid),
            valid_actions=valid_actions,
            turn=self._total_turns,
            metadata={
                "total_levels": len(e._levels),
                "level_index": e.level_index,
                "levels_completed": getattr(e, "_score", 0),
                "game_over": getattr(getattr(e, "_state", None), "name", "")
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

    env = ArcGameEnv(seed=0, render_mode="rgb_array")

    try:
        check_env(env.unwrapped, skip_render_check=True)
        print("[PASS] check_env passed — environment is Gymnasium-compliant.")
    except Exception as e:
        print(f"[FAIL] check_env failed: {e}")

    obs, info = env.reset()
    print(f"  obs shape: {obs.shape}, dtype: {obs.dtype}")
    print(f"  info keys: {list(info.keys())}")

    obs, reward, term, trunc, info = env.step(0)
    print(f"  step → reward={reward}, terminated={term}, truncated={trunc}")

    frame = env.render()
    print(f"  render → shape={frame.shape if frame is not None else None}")

    env.close()
    print("  close() OK")
