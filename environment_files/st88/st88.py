from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
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
PADDING_COLOR = 5
MAX_LIVES = 3
TILE = 5

NUM_COLORS = 7
PIECE_COLORS = [14, 9, 8, 11, 6, 12, 15]
COLOR_NAMES = ["Green", "Blue", "Red", "Yellow", "Magenta", "Orange", "Purple"]

CONTAINER_W = 6
CONTAINER_H = 10

GRID_W_TILES = 12
GRID_H_TILES = 12
GRID_W_PX = GRID_W_TILES * TILE
GRID_H_PX = GRID_H_TILES * TILE

CONT_LEFT = 1
CONT_RIGHT = 6
CONT_TOP = 1
CONT_BOTTOM = 10

STAGE_LEFT = 8
STAGE_RIGHT = 10
STAGE_TOP = 1
STAGE_BOTTOM = 10

MAX_VISIBLE_SLOTS = 3
SLOT_POSITIONS = [
    (8, 1),
    (8, 4),
    (8, 7),
]

SIDEBAR_SEPARATOR_ROWS = [3 * TILE + TILE // 2, 6 * TILE + TILE // 2]

SIDEBAR_SEP_X_START = STAGE_LEFT * TILE
SIDEBAR_SEP_X_END = (STAGE_RIGHT + 1) * TILE

OFFSCREEN_X = -20 * TILE
OFFSCREEN_Y = -20 * TILE

SHAPES = {
    "single": [(0, 0)],
    "pair_h": [(0, 0), (1, 0)],
    "pair_v": [(0, 0), (0, 1)],
    "L_right": [(0, 0), (0, 1), (1, 1)],
    "L_left": [(1, 0), (0, 1), (1, 1)],
    "L_up": [(0, 0), (1, 0), (0, 1)],
    "L_down": [(0, 0), (1, 0), (1, 1)],
    "line3_h": [(0, 0), (1, 0), (2, 0)],
    "line3_v": [(0, 0), (0, 1), (0, 2)],
    "T_up": [(0, 0), (1, 0), (2, 0)],
    "S_shape": [(1, 0), (0, 1), (1, 1)],
    "Z_shape": [(0, 0), (0, 1), (1, 1)],
    "square": [(0, 0), (1, 0), (0, 1), (1, 1)],
    "T_right": [(0, 0), (0, 1), (1, 1), (0, 2)],
    "T_down": [(0, 1), (1, 0), (1, 1), (2, 1)],
    "T_left": [(1, 0), (0, 1), (1, 1), (1, 2)],
    "S4": [(1, 0), (0, 1), (1, 1), (0, 2)],
    "Z4": [(0, 0), (0, 1), (1, 1), (1, 2)],
    "L4_a": [(0, 0), (0, 1), (0, 2), (1, 2)],
    "L4_b": [(1, 0), (1, 1), (0, 2), (1, 2)],
    "cross": [(1, 0), (0, 1), (1, 1), (2, 1), (1, 2)],
}

sprites = {
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
    "container_floor": Sprite(
        pixels=[
            [9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9],
        ],
        name="container_floor",
        visible=True,
        collidable=True,
        tags=["wall", "solid", "floor_marker"],
        layer=0,
    ),
    "container_ceiling": Sprite(
        pixels=[
            [9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9],
        ],
        name="container_ceiling",
        visible=True,
        collidable=True,
        tags=["wall", "solid", "ceiling_marker"],
        layer=0,
    ),
    "cell_empty": Sprite(
        pixels=[
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        name="cell_empty",
        visible=True,
        collidable=False,
        tags=["cell"],
        layer=-1,
    ),
    "block": Sprite(
        pixels=[
            [8, 8, 8, 8, 8],
            [8, 8, 8, 8, 8],
            [8, 8, 8, 8, 8],
            [8, 8, 8, 8, 8],
            [8, 8, 8, 8, 8],
        ],
        name="block",
        visible=True,
        collidable=False,
        tags=["block"],
        layer=2,
    ),
    "cursor": Sprite(
        pixels=[
            [7, 7, 7, 7, 7],
            [7, -1, -1, -1, 7],
            [7, -1, -1, -1, 7],
            [7, -1, -1, -1, 7],
            [7, 7, 7, 7, 7],
        ],
        name="cursor",
        visible=True,
        collidable=False,
        tags=["cursor"],
        layer=5,
    ),
    "ghost": Sprite(
        pixels=[
            [5, 5, 5, 5, 5],
            [5, 0, 0, 0, 5],
            [5, 0, 0, 0, 5],
            [5, 0, 0, 0, 5],
            [5, 5, 5, 5, 5],
        ],
        name="ghost",
        visible=False,
        collidable=False,
        tags=["ghost"],
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
    "stage_bg": Sprite(
        pixels=[
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        name="stage_bg",
        visible=True,
        collidable=False,
        tags=["stage_bg"],
        layer=-2,
    ),
    "separator": Sprite(
        pixels=[
            [5, 5, 5, 5, 5],
            [5, 0, 0, 0, 5],
            [5, 0, 0, 0, 5],
            [5, 0, 0, 0, 5],
            [5, 5, 5, 5, 5],
        ],
        name="separator",
        visible=True,
        collidable=True,
        tags=["wall", "solid"],
        layer=0,
    ),
}


def _shape_size(key: str) -> int:
    return len(SHAPES[key])


def _shape_bounds(key: str) -> Tuple[int, int]:
    cells = SHAPES[key]
    return (max(c for c, r in cells) + 1, max(r for c, r in cells) + 1)


def _make_block_pixels(color: int) -> list:
    return [[color] * 5 for _ in range(5)]


def _make_ghost_pixels() -> list:
    return [
        [5, 5, 5, 5, 5],
        [5, 0, 0, 0, 5],
        [5, 0, 0, 0, 5],
        [5, 0, 0, 0, 5],
        [5, 5, 5, 5, 5],
    ]


def _rotate_cells_90(cells: list) -> list:
    max_r = max(r for c, r in cells)
    rotated = [(max_r - r, c) for c, r in cells]
    min_c = min(c for c, r in rotated)
    min_r = min(r for c, r in rotated)
    return [(c - min_c, r - min_r) for c, r in rotated]


_l1_pieces = [
    {"shape": "single", "color_id": 0},
    {"shape": "pair_h", "color_id": 1},
    {"shape": "pair_v", "color_id": 2},
    {"shape": "L_up", "color_id": 3},
    {"shape": "L_right", "color_id": 4},
    {"shape": "square", "color_id": 5},
    {"shape": "line3_h", "color_id": 6},
    {"shape": "T_down", "color_id": 0},
    {"shape": "L_down", "color_id": 1},
    {"shape": "pair_h", "color_id": 2},
    {"shape": "S_shape", "color_id": 3},
    {"shape": "single", "color_id": 4},
]

_l2_pieces = [
    {"shape": "single", "color_id": 0},
    {"shape": "pair_v", "color_id": 1},
    {"shape": "pair_h", "color_id": 2},
    {"shape": "L_right", "color_id": 3},
    {"shape": "T_up", "color_id": 4},
    {"shape": "square", "color_id": 5},
    {"shape": "L_left", "color_id": 0},
    {"shape": "Z_shape", "color_id": 6},
    {"shape": "T_down", "color_id": 1},
    {"shape": "S_shape", "color_id": 2},
    {"shape": "L_up", "color_id": 3},
    {"shape": "pair_h", "color_id": 4},
    {"shape": "single", "color_id": 5},
]

_l3_pieces = [
    {"shape": "T_up", "color_id": 0},
    {"shape": "L_down", "color_id": 1},
    {"shape": "square", "color_id": 2},
    {"shape": "line3_h", "color_id": 0},
    {"shape": "Z_shape", "color_id": 3},
    {"shape": "T_down", "color_id": 4},
    {"shape": "S_shape", "color_id": 1},
    {"shape": "pair_h", "color_id": 5},
    {"shape": "L_right", "color_id": 6},
    {"shape": "pair_v", "color_id": 2},
]

_l4_pieces = [
    {"shape": "T_up", "color_id": 0},
    {"shape": "square", "color_id": 1},
    {"shape": "L_left", "color_id": 2},
    {"shape": "T_down", "color_id": 0},
    {"shape": "S_shape", "color_id": 3},
    {"shape": "pair_h", "color_id": 4},
    {"shape": "Z_shape", "color_id": 0},
    {"shape": "square", "color_id": 2},
    {"shape": "line3_h", "color_id": 5},
    {"shape": "pair_v", "color_id": 3},
    {"shape": "L_right", "color_id": 6},
]

_l5_pieces = [
    {"shape": "L_up", "color_id": 0},
    {"shape": "pair_h", "color_id": 1},
    {"shape": "L_right", "color_id": 2},
    {"shape": "square", "color_id": 3},
    {"shape": "L_down", "color_id": 0},
    {"shape": "pair_v", "color_id": 4},
    {"shape": "T_up", "color_id": 5},
    {"shape": "L_left", "color_id": 1},
    {"shape": "S_shape", "color_id": 6},
    {"shape": "T_down", "color_id": 2},
    {"shape": "Z_shape", "color_id": 5},
    {"shape": "line3_h", "color_id": 3},
]


def _build_container():
    out = []

    for c in range(GRID_W_TILES):
        if CONT_LEFT <= c <= CONT_RIGHT:
            out.append(sprites["container_ceiling"].clone().set_position(c * TILE, 0))
        else:
            out.append(sprites["wall"].clone().set_position(c * TILE, 0))

    for c in range(GRID_W_TILES):
        if CONT_LEFT <= c <= CONT_RIGHT:
            out.append(
                sprites["container_floor"].clone().set_position(c * TILE, 11 * TILE)
            )
        else:
            out.append(sprites["wall"].clone().set_position(c * TILE, 11 * TILE))

    for r in range(1, 11):
        out.append(sprites["wall"].clone().set_position(0, r * TILE))

    for r in range(1, 11):
        out.append(sprites["separator"].clone().set_position(7 * TILE, r * TILE))

    for r in range(1, 11):
        out.append(sprites["wall"].clone().set_position(11 * TILE, r * TILE))

    return out


def _build_staging_bg():
    out = []
    for r in range(STAGE_TOP, STAGE_BOTTOM + 1):
        for c in range(STAGE_LEFT, STAGE_RIGHT + 1):
            out.append(sprites["stage_bg"].clone().set_position(c * TILE, r * TILE))
    return out


def _build_container_bg():
    out = []
    for r in range(CONT_TOP, CONT_BOTTOM + 1):
        for c in range(CONT_LEFT, CONT_RIGHT + 1):
            out.append(sprites["cell_empty"].clone().set_position(c * TILE, r * TILE))
    return out


def _build_level(pieces_data, max_moves):
    all_sprites = _build_container() + _build_staging_bg() + _build_container_bg()

    for i, pdata in enumerate(pieces_data):
        shape_cells = SHAPES[pdata["shape"]]
        color = PIECE_COLORS[pdata["color_id"]]

        if i < len(SLOT_POSITIONS):
            sc, sr = SLOT_POSITIONS[i]
        else:
            sc = OFFSCREEN_X // TILE
            sr = OFFSCREEN_Y // TILE

        for dc, dr in shape_cells:
            block = Sprite(
                pixels=_make_block_pixels(color),
                name=f"block_p{i}",
                visible=(i < len(SLOT_POSITIONS)),
                collidable=False,
                tags=[
                    "block",
                    f"piece_{i}",
                    f"color_{pdata['color_id']}",
                    f"size_{len(shape_cells)}",
                ],
                layer=2,
            )
            block.set_position((sc + dc) * TILE, (sr + dr) * TILE)
            all_sprites.append(block)

    first = SLOT_POSITIONS[0]
    cursor = sprites["cursor"].clone().set_position(first[0] * TILE, first[1] * TILE)
    all_sprites.append(cursor)

    flash = sprites["flash"].clone()
    all_sprites.append(flash)

    return Level(
        sprites=all_sprites,
        grid_size=(GRID_W_PX, GRID_H_PX),
        data={
            "pieces": pieces_data,
            "max_moves": max_moves,
            "num_pieces": len(pieces_data),
        },
    )


levels = [
    _build_level(_l1_pieces, 120),
    _build_level(_l2_pieces, 140),
    _build_level(_l3_pieces, 110),
    _build_level(_l4_pieces, 130),
    _build_level(_l5_pieces, 150),
]


class StackHUD(RenderableUserDisplay):
    def __init__(self, game):
        self.game = game
        self.moves_remaining = 0
        self.max_moves = 0
        self.lives = MAX_LIVES
        self.pieces_placed = 0
        self.total_pieces = 0

    def update(
        self,
        moves_remaining=0,
        max_moves=0,
        lives=MAX_LIVES,
        pieces_placed=0,
        total_pieces=0,
    ):
        self.moves_remaining = moves_remaining
        self.max_moves = max_moves
        self.lives = lives
        self.pieces_placed = pieces_placed
        self.total_pieces = total_pieces

    def render_interface(self, frame):
        height, width = frame.shape[:2]

        x_off = (width - GRID_W_PX) // 2
        y_off = (height - GRID_H_PX) // 2

        sep_x_start = x_off + SIDEBAR_SEP_X_START
        sep_x_end = x_off + SIDEBAR_SEP_X_END

        for game_y in SIDEBAR_SEPARATOR_ROWS:
            for dy in (-1, 0, 1):
                fy = y_off + game_y + dy
                if 0 <= fy < height:
                    for fx in range(sep_x_start, sep_x_end):
                        if 0 <= fx < width:
                            frame[fy, fx] = 5

        for i in range(MAX_LIVES):
            px = width - 2 - i * 4
            color = 2 if i < self.lives else 5
            for dx in range(3):
                if 0 <= px - dx < width:
                    frame[0, px - dx] = color

        for i in range(min(self.total_pieces, 15)):
            px = 1 + i * 3
            if px < width:
                frame[1, px] = 4 if i < self.pieces_placed else 5

        if self.max_moves > 0:
            bar_row = height - 1
            ratio = max(0.0, self.moves_remaining / self.max_moves)
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


STATE_SELECT_PIECE = 0
STATE_PLACE_COLUMN = 1
STATE_CHECK_RULES = 3
STATE_ANIMATE_DISAPPEAR = 4


class St88(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self.hud = StackHUD(self)
        self.lives = MAX_LIVES
        self.moves_remaining = 0
        self.max_moves = 0

        self.state = STATE_SELECT_PIECE
        self.cursor_sprite = None

        self.container = []

        self.pieces_data = []
        self.piece_sprites = {}
        self.placed_sprites = []
        self.pieces_placed_set = set()
        self.selected_piece = -1
        self.place_column = 0

        self.ghost_sprites = []

        self.disappear_timer = 0

        self.flash_sprite = None
        self.flash_active = False

        self._start_piece_positions = {}
        self.staging_scroll_offset = 0

        self._game_over = False
        self._consecutive_resets = 0
        self._history: List[Dict] = []

        super().__init__(
            "st88",
            levels,
            Camera(
                width=GRID_W_PX,
                height=GRID_H_PX,
                background=BACKGROUND_COLOR,
                letter_box=PADDING_COLOR,
                interfaces=[self.hud],
            ),
            available_actions=[0, 1, 2, 3, 4, 5, 7],
        )

    def on_set_level(self, level: Level) -> None:
        self.state = STATE_SELECT_PIECE
        self.pieces_data = self.current_level.get_data("pieces") or []
        self.max_moves = self.current_level.get_data("max_moves") or 100
        self.moves_remaining = self.max_moves
        num_pieces = self.current_level.get_data("num_pieces") or len(self.pieces_data)
        self.staging_scroll_offset = 0

        self.container = [[None] * CONTAINER_W for _ in range(CONTAINER_H)]

        cl = self.current_level.get_sprites_by_tag("cursor")
        self.cursor_sprite = cl[0] if cl else sprites["cursor"].clone()
        if not cl:
            self.current_level.add_sprite(self.cursor_sprite)

        fl = self.current_level.get_sprites_by_tag("flash")
        self.flash_sprite = fl[0] if fl else sprites["flash"].clone()
        if not fl:
            self.current_level.add_sprite(self.flash_sprite)
        self.flash_sprite.set_visible(False)
        self.flash_active = False

        self.piece_sprites = {}
        self.placed_sprites = []
        self.pieces_placed_set = set()
        self.selected_piece = 0
        self.place_column = 0
        self.ghost_sprites = []
        self.disappear_timer = 0

        for i in range(num_pieces):
            blocks = self.current_level.get_sprites_by_tag(f"piece_{i}")
            self.piece_sprites[i] = blocks
            self._start_piece_positions[i] = [(b.x, b.y) for b in blocks]

        self._update_cursor_to_piece()
        self.lives = MAX_LIVES
        self._hud_sync()
        self._game_over = False
        self._history = []

    def _hud_sync(self):
        num = self.current_level.get_data("num_pieces") or len(self.pieces_data)
        self.hud.update(
            moves_remaining=self.moves_remaining,
            max_moves=self.max_moves,
            lives=self.lives,
            pieces_placed=len(self.pieces_placed_set),
            total_pieces=num,
        )

    def _get_available_pieces(self) -> List[int]:
        return [
            i for i in range(len(self.pieces_data)) if i not in self.pieces_placed_set
        ]

    def _refresh_staging_view(self):
        available = self._get_available_pieces()
        for pid in range(len(self.pieces_data)):
            blocks = self.piece_sprites.get(pid, [])
            if pid in self.pieces_placed_set:
                for b in blocks:
                    b.set_visible(False)
                continue
            if pid not in available:
                for b in blocks:
                    b.set_visible(False)
                continue
            aidx = available.index(pid)
            vslot = aidx - self.staging_scroll_offset
            if 0 <= vslot < MAX_VISIBLE_SLOTS and vslot < len(SLOT_POSITIONS):
                sc, sr = SLOT_POSITIONS[vslot]
                cells = SHAPES[self.pieces_data[pid]["shape"]]
                for j, (dc, dr) in enumerate(cells):
                    if j < len(blocks):
                        blocks[j].set_position((sc + dc) * TILE, (sr + dr) * TILE)
                        blocks[j].set_visible(True)
            else:
                for b in blocks:
                    b.set_visible(False)

    def _update_cursor_to_piece(self):
        available = self._get_available_pieces()
        if not available:
            if self.cursor_sprite:
                self.cursor_sprite.set_visible(False)
            return
        if self.selected_piece not in available:
            self.selected_piece = available[0]
        aidx = available.index(self.selected_piece)
        if aidx < self.staging_scroll_offset:
            self.staging_scroll_offset = aidx
        elif aidx >= self.staging_scroll_offset + MAX_VISIBLE_SLOTS:
            self.staging_scroll_offset = aidx - MAX_VISIBLE_SLOTS + 1
        mx = max(0, len(available) - MAX_VISIBLE_SLOTS)
        self.staging_scroll_offset = max(0, min(self.staging_scroll_offset, mx))
        self._refresh_staging_view()
        vslot = aidx - self.staging_scroll_offset
        if 0 <= vslot < len(SLOT_POSITIONS) and self.cursor_sprite:
            sc, sr = SLOT_POSITIONS[vslot]
            self.cursor_sprite.set_visible(True)
            self.cursor_sprite.set_position(sc * TILE, sr * TILE)

    def _update_cursor_to_column(self):
        if self.cursor_sprite:
            self.cursor_sprite.set_visible(True)
            self.cursor_sprite.set_position((CONT_LEFT + self.place_column) * TILE, 0)

    def _clear_ghosts(self):
        for g in self.ghost_sprites:
            self.current_level.remove_sprite(g)
        self.ghost_sprites = []

    def _show_drop_preview(self):
        self._clear_ghosts()
        if self.selected_piece < 0 or self.selected_piece >= len(self.pieces_data):
            return
        pdata = self.pieces_data[self.selected_piece]
        cells = SHAPES[pdata["shape"]]
        sw, sh = _shape_bounds(pdata["shape"])
        if self.place_column + sw > CONTAINER_W:
            return
        lr = self._find_landing_row(cells, self.place_column)
        if lr is None:
            return
        for dc, dr in cells:
            g = Sprite(
                pixels=_make_ghost_pixels(),
                name="ghost_preview",
                visible=True,
                collidable=False,
                tags=["ghost"],
                layer=1,
            )
            g.set_position(
                (CONT_LEFT + self.place_column + dc) * TILE, (CONT_TOP + lr + dr) * TILE
            )
            self.current_level.add_sprite(g)
            self.ghost_sprites.append(g)

    def _find_landing_row(self, shape_cells, col_offset) -> Optional[int]:
        sw, sh = 0, 0
        for dc, dr in shape_cells:
            sw = max(sw, dc + 1)
            sh = max(sh, dr + 1)
        for start_row in range(CONTAINER_H - sh, -1, -1):
            ok = True
            for dc, dr in shape_cells:
                c, r = col_offset + dc, start_row + dr
                if c < 0 or c >= CONTAINER_W or r < 0 or r >= CONTAINER_H:
                    ok = False
                    break
                if self.container[r][c] is not None:
                    ok = False
                    break
            if not ok:
                continue
            ppos = {(col_offset + dc, start_row + dr) for dc, dr in shape_cells}
            supported = False
            for dc, dr in shape_cells:
                c, r = col_offset + dc, start_row + dr
                if r == CONTAINER_H - 1:
                    supported = True
                    break
                if (
                    (c, r + 1) not in ppos
                    and r + 1 < CONTAINER_H
                    and self.container[r + 1][c] is not None
                ):
                    supported = True
                    break
            if supported:
                return start_row
        return None

    def _place_piece(self, pid, col, row, cells=None):
        pdata = self.pieces_data[pid]
        if cells is None:
            cells = SHAPES[pdata["shape"]]
        color = PIECE_COLORS[pdata["color_id"]]

        piece_size = len(cells)
        for dc, dr in cells:
            c, r = col + dc, row + dr
            self.container[r][c] = {
                "piece_id": pid,
                "color_id": pdata["color_id"],
                "size": piece_size,
            }
            blk = Sprite(
                pixels=_make_block_pixels(color),
                name=f"placed_p{pid}",
                visible=True,
                collidable=False,
                tags=[
                    "placed_block",
                    f"placed_piece_{pid}",
                    f"color_{pdata['color_id']}",
                ],
                layer=2,
            )
            blk.set_position((CONT_LEFT + c) * TILE, (CONT_TOP + r) * TILE)
            self.current_level.add_sprite(blk)
            self.placed_sprites.append(blk)

        for s in self.piece_sprites.get(pid, []):
            s.set_visible(False)
        self.pieces_placed_set.add(pid)

    def _swap_shape_to_next(self, pid: int, steps: int = 1) -> None:
        available = self._get_available_pieces()
        if pid not in available:
            return
        aidx = available.index(pid)
        if aidx + steps >= len(available):
            return

        next_pid = available[aidx + steps]
        next_shape = self.pieces_data[next_pid]["shape"]

        if self.pieces_data[pid]["shape"] == next_shape:
            return

        self.pieces_data[pid]["shape"] = next_shape

        old_blocks = self.piece_sprites.get(pid, [])
        for b in old_blocks:
            self.current_level.remove_sprite(b)

        color = PIECE_COLORS[self.pieces_data[pid]["color_id"]]
        cells = SHAPES[next_shape]
        new_blocks: List = []
        for dc, dr in cells:
            blk = Sprite(
                pixels=_make_block_pixels(color),
                name=f"block_p{pid}",
                visible=False,
                collidable=False,
                tags=[
                    "block",
                    f"piece_{pid}",
                    f"color_{self.pieces_data[pid]['color_id']}",
                    f"size_{len(cells)}",
                ],
                layer=2,
            )
            self.current_level.add_sprite(blk)
            new_blocks.append(blk)

        self.piece_sprites[pid] = new_blocks
        self._start_piece_positions[pid] = [(b.x, b.y) for b in new_blocks]

        self._refresh_staging_view()

    def _get_neighbors(self, r, c):
        out = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < CONTAINER_H and 0 <= nc < CONTAINER_W:
                out.append((nr, nc))
        return out

    def _check_disappearance(self) -> Set[Tuple[int, int]]:
        to_remove: Set[Tuple[int, int]] = set()
        for r in range(CONTAINER_H):
            for c in range(CONTAINER_W):
                cell = self.container[r][c]
                if cell is None:
                    continue
                for nr, nc in self._get_neighbors(r, c):
                    nb = self.container[nr][nc]
                    if nb is None:
                        continue
                    if cell["piece_id"] == nb["piece_id"]:
                        continue
                    color_match = cell["color_id"] == nb["color_id"]
                    size_match = cell.get("size", 0) == nb.get("size", 0)
                    if color_match or size_match:
                        pa, pb = cell["piece_id"], nb["piece_id"]
                        for rr in range(CONTAINER_H):
                            for cc in range(CONTAINER_W):
                                if self.container[rr][cc] is not None:
                                    if self.container[rr][cc]["piece_id"] in (pa, pb):
                                        to_remove.add((rr, cc))
        return to_remove

    def _remove_cells(self, cells):
        removed_pids = set()
        for r, c in cells:
            if self.container[r][c] is not None:
                removed_pids.add(self.container[r][c]["piece_id"])
                self.container[r][c] = None
        rm = []
        for s in self.placed_sprites:
            if s.tags:
                for pid in removed_pids:
                    if f"placed_piece_{pid}" in s.tags:
                        rm.append(s)
                        break
        for s in rm:
            self.current_level.remove_sprite(s)
            self.placed_sprites.remove(s)

    def _apply_gravity(self):
        changed = True
        while changed:
            changed = False
            for r in range(CONTAINER_H - 2, -1, -1):
                for c in range(CONTAINER_W):
                    if (
                        self.container[r][c] is not None
                        and self.container[r + 1][c] is None
                    ):
                        self.container[r + 1][c] = self.container[r][c]
                        self.container[r][c] = None
                        changed = True
        self._sync_placed_sprites()

    def _sync_placed_sprites(self):
        for s in self.placed_sprites:
            self.current_level.remove_sprite(s)
        self.placed_sprites = []
        for r in range(CONTAINER_H):
            for c in range(CONTAINER_W):
                cell = self.container[r][c]
                if cell is not None:
                    color = PIECE_COLORS[cell["color_id"]]
                    blk = Sprite(
                        pixels=_make_block_pixels(color),
                        name=f"placed_p{cell['piece_id']}",
                        visible=True,
                        collidable=False,
                        tags=[
                            "placed_block",
                            f"placed_piece_{cell['piece_id']}",
                            f"color_{cell['color_id']}",
                            f"size_{cell.get('size', 0)}",
                        ],
                        layer=2,
                    )
                    blk.set_position((CONT_LEFT + c) * TILE, (CONT_TOP + r) * TILE)
                    self.current_level.add_sprite(blk)
                    self.placed_sprites.append(blk)

    def _check_win(self) -> bool:
        starts = [
            (CONTAINER_H - 1, c)
            for c in range(CONTAINER_W)
            if self.container[CONTAINER_H - 1][c] is not None
        ]
        if not starts:
            return False
        visited = set(starts)
        queue = list(starts)
        while queue:
            r, c = queue.pop(0)
            if r == 0:
                return True
            for nr, nc in self._get_neighbors(r, c):
                if (nr, nc) not in visited and self.container[nr][nc] is not None:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        return False

    def _check_level_complete(self) -> bool:
        if not any(self.container[0][c] is not None for c in range(CONTAINER_W)):
            return False
        if not any(
            self.container[CONTAINER_H - 1][c] is not None for c in range(CONTAINER_W)
        ):
            return False
        return self._check_win()

    def _reset_current_level(self):
        self.container = [[None] * CONTAINER_W for _ in range(CONTAINER_H)]
        for s in self.placed_sprites:
            self.current_level.remove_sprite(s)
        self.placed_sprites = []
        self._clear_ghosts()
        self.pieces_placed_set = set()
        self.selected_piece = 0
        self.place_column = 0
        self.staging_scroll_offset = 0
        self.state = STATE_SELECT_PIECE
        self.moves_remaining = self.max_moves
        for pid, positions in self._start_piece_positions.items():
            blocks = self.piece_sprites.get(pid, [])
            for j, b in enumerate(blocks):
                if j < len(positions):
                    b.set_position(positions[j][0], positions[j][1])
                b.set_visible(True)
        self.disappear_timer = 0
        self._update_cursor_to_piece()
        self._hud_sync()

    def _handle_death(self):
        self.lives -= 1
        if self.lives <= 0:
            self._game_over = True
            self.lose()
            return
        if self.flash_sprite:
            self.flash_sprite.set_visible(True)
            self.flash_sprite.set_scale(max(GRID_W_PX, GRID_H_PX))
            self.flash_sprite.set_position(0, 0)
        self.flash_active = True
        self._reset_current_level()

    def _save_state(self) -> None:
        container_snap = [
            [(dict(cell) if cell is not None else None) for cell in row]
            for row in self.container
        ]
        self._history.append(
            {
                "container": container_snap,
                "pieces_placed_set": set(self.pieces_placed_set),
                "selected_piece": self.selected_piece,
                "place_column": self.place_column,
                "staging_scroll_offset": self.staging_scroll_offset,
                "state": self.state,
            }
        )

    def _undo(self) -> None:
        if not self._history:
            return
        snap = self._history.pop()
        self.container = snap["container"]
        self.pieces_placed_set = snap["pieces_placed_set"]
        self.selected_piece = snap["selected_piece"]
        self.place_column = snap["place_column"]
        self.staging_scroll_offset = snap["staging_scroll_offset"]
        self.state = snap["state"]
        self._sync_placed_sprites()
        self._clear_ghosts()
        if self.state == STATE_PLACE_COLUMN:
            self._update_cursor_to_column()
        else:
            self._update_cursor_to_piece()
        self._hud_sync()

    def step(self) -> None:
        if self.action.id == GameAction.RESET:
            self.complete_action()
            return

        num_pieces = self.current_level.get_data("num_pieces") or len(self.pieces_data)

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
            if self.moves_remaining <= 0:
                self._handle_death()
            self._hud_sync()
            self.complete_action()
            return

        self._consecutive_resets = 0

        if self.state == STATE_ANIMATE_DISAPPEAR:
            self.disappear_timer -= 1
            if self.disappear_timer <= 0:
                to_rm = self._check_disappearance()
                if to_rm:
                    self._remove_cells(to_rm)
                    self._apply_gravity()
                    more = self._check_disappearance()
                    if more:
                        self.disappear_timer = 2
                        self.state = STATE_ANIMATE_DISAPPEAR
                        self.complete_action()
                        return
                    else:
                        self.state = STATE_SELECT_PIECE
                        self._update_cursor_to_piece()
                else:
                    self.state = STATE_SELECT_PIECE
                    self._update_cursor_to_piece()

                if self._check_level_complete():
                    self.next_level()
                    self.complete_action()
                    return

            self.complete_action()
            return

        if self.state == STATE_CHECK_RULES:
            to_rm = self._check_disappearance()
            if to_rm:
                for r, c in to_rm:
                    cell = self.container[r][c]
                    if cell is not None:
                        px = (CONT_LEFT + c) * TILE
                        py = (CONT_TOP + r) * TILE
                        for s in self.placed_sprites:
                            if s.x == px and s.y == py:
                                s.color_remap(None, 7)
                                break
                self.disappear_timer = 2
                self.state = STATE_ANIMATE_DISAPPEAR
                self.complete_action()
                return

            if self._check_level_complete():
                self.next_level()
                self.complete_action()
                return

            if len(self.pieces_placed_set) >= num_pieces:
                avail = self._get_available_pieces()
                if not avail:
                    self._handle_death()
                    if self.flash_active:
                        self.complete_action()
                        return
                    self.complete_action()
                    return

            self.state = STATE_SELECT_PIECE
            self._update_cursor_to_piece()
            self.complete_action()
            return

        if self.state == STATE_SELECT_PIECE:
            available = self._get_available_pieces()
            if not available:
                self.complete_action()
                return
            action_id = self.action.id.value

            if action_id == 1:
                self._save_state()
                self.moves_remaining -= 1
                idx = (
                    available.index(self.selected_piece)
                    if self.selected_piece in available
                    else 0
                )
                idx = (idx - 1) % len(available)
                self.selected_piece = available[idx]
                self._update_cursor_to_piece()

            elif action_id == 2:
                self._save_state()
                self.moves_remaining -= 1
                idx = (
                    available.index(self.selected_piece)
                    if self.selected_piece in available
                    else 0
                )
                idx = (idx + 1) % len(available)
                self.selected_piece = available[idx]
                self._update_cursor_to_piece()

            elif action_id == 5:
                self._save_state()
                self.moves_remaining -= 1
                self.state = STATE_PLACE_COLUMN
                self.place_column = 0
                self._update_cursor_to_column()
                self._show_drop_preview()

            if self.moves_remaining <= 0:
                self._handle_death()
                if self.flash_active:
                    self.complete_action()
                    return
                self.complete_action()
                return

            self._hud_sync()
            self.complete_action()
            return

        if self.state == STATE_PLACE_COLUMN:
            action_id = self.action.id.value

            if action_id == 3:
                self._save_state()
                self.moves_remaining -= 1
                self.place_column = max(0, self.place_column - 1)
                self._update_cursor_to_column()
                self._show_drop_preview()

            elif action_id == 4:
                self._save_state()
                self.moves_remaining -= 1
                self.place_column = min(CONTAINER_W - 1, self.place_column + 1)
                self._update_cursor_to_column()
                self._show_drop_preview()

            elif action_id == 1:
                self._save_state()
                self._clear_ghosts()
                self.state = STATE_SELECT_PIECE
                self._update_cursor_to_piece()
                self.complete_action()
                return

            elif action_id == 5:
                self._save_state()
                self.moves_remaining -= 1
                self._clear_ghosts()
                if self.level_index == 3:
                    self._swap_shape_to_next(self.selected_piece, steps=1)

                pdata = self.pieces_data[self.selected_piece]
                cells = SHAPES[pdata["shape"]]
                if self.level_index == 4:
                    cells = _rotate_cells_90(cells)

                lr = self._find_landing_row(cells, self.place_column)
                if lr is not None:
                    self._place_piece(
                        self.selected_piece, self.place_column, lr, cells=cells
                    )
                    self._apply_gravity()
                    self.state = STATE_CHECK_RULES
                    self._hud_sync()
                    self.complete_action()
                    return

            if self.moves_remaining <= 0:
                self._clear_ghosts()
                self._handle_death()
                if self.flash_active:
                    self.complete_action()
                    return
                self.complete_action()
                return

            self._hud_sync()
            self.complete_action()
            return

        self.complete_action()


class PuzzleEnvironment:
    _ACTION_MAP: Dict[str, GameAction] = {
        "up": GameAction.ACTION1,
        "down": GameAction.ACTION2,
        "left": GameAction.ACTION3,
        "right": GameAction.ACTION4,
        "confirm": GameAction.ACTION5,
        "undo": GameAction.ACTION7,
        "reset": GameAction.RESET,
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

    _TAG_TO_CHAR: Dict[str, str] = {
        "wall": "#",
        "cell": ".",
        "block": "B",
        "cursor": "^",
        "ghost": "G",
        "placed_block": "B",
        "flash": "!",
        "stage_bg": " ",
        "floor_marker": "=",
        "ceiling_marker": "=",
        "color_0": "g",
        "color_1": "b",
        "color_2": "r",
        "color_3": "y",
        "color_4": "m",
        "color_5": "o",
        "color_6": "p",
    }

    _TAG_PRIORITY: Dict[str, int] = {
        "wall": 0,
        "floor_marker": 0,
        "ceiling_marker": 0,
        "cell": 1,
        "stage_bg": -1,
        "ghost": 2,
        "block": 3,
        "placed_block": 3,
        "color_0": 4,
        "color_1": 4,
        "color_2": 4,
        "color_3": 4,
        "color_4": 4,
        "color_5": 4,
        "color_6": 4,
        "cursor": 5,
        "flash": 10,
    }

    def __init__(self, seed: int = 0) -> None:
        self._engine = St88(seed=seed)
        self._total_turns = 0
        self._last_action_was_reset = False
        self._game_won = False
        self._total_levels = len(self._engine._levels)

    def _build_text_obs(self) -> str:
        e = self._engine
        w, h = GRID_W_TILES, GRID_H_TILES
        text_grid: List[List[str]] = [["." for _ in range(w)] for _ in range(h)]
        prio_grid: List[List[int]] = [[-2 for _ in range(w)] for _ in range(h)]
        tag_char = self._TAG_TO_CHAR
        tag_prio = self._TAG_PRIORITY
        for sprite in e.current_level.get_sprites():
            if not sprite.is_visible:
                continue
            sx = sprite.x // TILE
            sy = sprite.y // TILE
            if 0 <= sx < w and 0 <= sy < h:
                for tag in sprite.tags or []:
                    ch = tag_char.get(tag, "")
                    pr = tag_prio.get(tag, -1)
                    if ch and pr > prio_grid[sy][sx]:
                        text_grid[sy][sx] = ch
                        prio_grid[sy][sx] = pr
        grid_text = "\n".join("".join(row) for row in text_grid)
        num = e.current_level.get_data("num_pieces") or len(e.pieces_data)
        header = (
            f"Level:{e.level_index + 1} Lives:{e.lives} "
            f"Moves:{e.moves_remaining}/{e.max_moves} "
            f"Pieces:{len(e.pieces_placed_set)}/{num}"
        )
        return header.strip() + "\n" + grid_text

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
        num = e.current_level.get_data("num_pieces") or len(e.pieces_data)
        return GameState(
            text_observation=self._build_text_obs(),
            image_observation=self._build_image_bytes(),
            valid_actions=valid_actions,
            turn=self._total_turns,
            metadata={
                "level_index": e.level_index,
                "total_levels": len(levels),
                "lives": e.lives,
                "moves_remaining": e.moves_remaining,
                "max_moves": e.max_moves,
                "pieces_placed": len(e.pieces_placed_set),
                "total_pieces": num,
                "game_over": e._game_over,
                "done": done,
                "info": info or {},
            },
        )

    def reset(self) -> GameState:
        e = self._engine
        e._game_over = False
        self._total_turns = 0
        game_won = hasattr(e, "_state") and getattr(e._state, "name", "") == "WIN"
        if game_won or self._last_action_was_reset:
            e._consecutive_resets = 1
        else:
            e._consecutive_resets = 0
        e.perform_action(ActionInput(id=GameAction.RESET))
        self._last_action_was_reset = True
        self._game_won = False
        return self._build_game_state()

    def is_done(self) -> bool:
        return self._engine._game_over or self._game_won

    def get_actions(self) -> List[str]:
        if self._engine._game_over or self._game_won:
            return ["reset"]
        return ["up", "down", "left", "right", "confirm", "undo", "reset"]

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

    _GYM_TO_INTERNAL: Dict[str, str] = {"select": "confirm"}
    _INTERNAL_TO_GYM: Dict[str, str] = {"confirm": "select"}

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
        self._env: Any = None

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

        internal_action: str = self._GYM_TO_INTERNAL.get(action_str, action_str)

        result: StepResult = self._env.step(internal_action)

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
                std_name: str = self._INTERNAL_TO_GYM.get(a, a)
                idx = self._string_to_action.get(std_name)
                if idx is not None:
                    mask[idx] = 1
        return mask

    def _get_obs(self) -> np.ndarray:
        frame = self._env.render()
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
