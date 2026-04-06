import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from arcengine import (
    ARCBaseGame,
    ActionInput,
    Camera,
    GameAction,
    Level,
    Sprite,
)

C_BG = 5
C_O_BLOCK = 11
C_T_BLOCK = 15
C_L_BLOCK = 12
C_I_BLOCK = 10
C_Z_BLOCK = 8
C_ALLOY = 9
C_PRIZE = 14
C_OBSTACLE = 3
C_GROUND = 13
C_LIFE = 8
C_LIFE_LOST = 4
C_SEPARATOR = 4
C_INDICATOR = 0
C_BAR_FILL = 12
C_BAR_EMPTY = 4
C_CURSOR = 0
C_GHOST_OK = 2
C_GHOST_BAD = 6
C_BORDER = 1

BACKGROUND_COLOR = 5
PADDING_COLOR = 5

CELL_SIZE = 3
MAX_GRID_W = 12
MAX_GRID_H = 16
HEADER_PX = 8
CAM_W = MAX_GRID_W * CELL_SIZE
CAM_H = HEADER_PX + MAX_GRID_H * CELL_SIZE + 5

MAX_LIVES = 3
EMPTY = 0
OBSTACLE = -1

BLOCK_ORDER = ["O", "T", "L", "I", "Z"]

MAX_MOVES_PER_LEVEL = [225, 300, 325, 350]

BLOCKS = {
    "O": {
        "shape": [(0, 0), (0, 1), (1, 0), (1, 1)],
        "color": C_O_BLOCK,
        "name": "O-Block",
    },
    "T": {
        "shape": [(0, 1), (1, 0), (1, 1), (1, 2)],
        "color": C_T_BLOCK,
        "name": "T-Block",
    },
    "L": {
        "shape": [(0, 0), (1, 0), (2, 0), (2, 1)],
        "color": C_L_BLOCK,
        "name": "L-Block",
    },
    "I": {
        "shape": [(0, 0), (1, 0), (2, 0), (3, 0)],
        "color": C_I_BLOCK,
        "name": "I-Block",
    },
    "Z": {
        "shape": [(0, 0), (0, 1), (1, 1), (1, 2)],
        "color": C_Z_BLOCK,
        "name": "Z-Block",
    },
}

LEVEL_CONFIGS = [
    {
        "name": "Foundation",
        "grid_w": 8,
        "grid_h": 10,
        "prize": [(r, c) for r in range(0, 2) for c in range(3, 5)],
        "inventory": {"O": 5, "I": 3, "T": 2},
        "obstacles": [
            (8, 1),
            (8, 6),
            (6, 2),
            (6, 5),
            (4, 1),
            (4, 6),
        ],
        "ground": (0, 7),
    },
    {
        "name": "The Wall",
        "grid_w": 10,
        "grid_h": 12,
        "prize": [(r, c) for r in range(0, 2) for c in range(4, 6)],
        "inventory": {"O": 5, "L": 4, "I": 5},
        "obstacles": [
            (9, 2),
            (9, 7),
            (7, 3),
            (7, 4),
            (7, 5),
            (7, 6),
            (5, 2),
            (5, 7),
            (3, 3),
            (3, 6),
        ],
        "ground": (0, 9),
    },
    {
        "name": "The Forge",
        "grid_w": 10,
        "grid_h": 12,
        "prize": [(r, c) for r in range(0, 2) for c in range(4, 6)],
        "inventory": {"O": 4, "I": 8, "Z": 3},
        "obstacles": [
            (9, 2),
            (9, 7),
            (8, 3),
            (8, 4),
            (6, 6),
            (6, 7),
            (5, 2),
            (5, 3),
            (3, 3),
            (3, 6),
        ],
        "ground": (2, 7),
    },
    {
        "name": "Master Builder",
        "grid_w": 12,
        "grid_h": 16,
        "prize": [(r, c) for r in range(0, 2) for c in range(10, 12)],
        "inventory": {"O": 6, "I": 10, "L": 4, "Z": 3, "T": 3},
        "obstacles": [
            (13, 8),
            (13, 9),
            (13, 10),
            (13, 11),
            (13, 2),
            (13, 3),
            (12, 1),
            (11, 8),
            (11, 9),
            (10, 3),
            (10, 4),
            (10, 5),
            (10, 8),
            (10, 9),
            (10, 10),
            (10, 11),
            (8, 6),
            (8, 7),
            (8, 2),
            (8, 3),
            (6, 10),
            (6, 11),
            (5, 4),
            (5, 5),
            (4, 8),
            (4, 9),
            (3, 7),
            (3, 8),
            (2, 9),
        ],
        "ground": (3, 7),
    },
]

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

ARC_PALETTE = np.array(ARC_PALETTE, dtype=np.uint8)

_BLOCK_CHAR = {
    C_O_BLOCK: "O",
    C_T_BLOCK: "T",
    C_L_BLOCK: "L",
    C_I_BLOCK: "I",
    C_Z_BLOCK: "Z",
    C_ALLOY: "A",
}
_TILE_CHAR = {
    C_BG: ".",
    C_O_BLOCK: "O",
    C_T_BLOCK: "T",
    C_L_BLOCK: "L",
    C_I_BLOCK: "I",
    C_Z_BLOCK: "Z",
    C_ALLOY: "A",
    C_PRIZE: "*",
    C_OBSTACLE: "#",
}


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


_cell_cache = {}


def _cell_sprite(color):
    if color not in _cell_cache:
        _cell_cache[color] = Sprite(
            pixels=[[color] * CELL_SIZE for _ in range(CELL_SIZE)],
            name=f"cell_{color}",
            visible=True,
            collidable=False,
            layer=1,
            tags=["cell"],
        )
    return _cell_cache[color]


def _prize_sprite():
    if "_prize" not in _cell_cache:
        _cell_cache["_prize"] = Sprite(
            pixels=[
                [C_PRIZE, C_BG, C_PRIZE],
                [C_BG, C_PRIZE, C_BG],
                [C_PRIZE, C_BG, C_PRIZE],
            ],
            name="prize",
            visible=True,
            collidable=False,
            layer=1,
            tags=["cell"],
        )
    return _cell_cache["_prize"]


def _ghost_cell_sprite(color, valid):
    gc = color if valid else C_GHOST_BAD
    return Sprite(
        pixels=[
            [gc, C_BG, gc],
            [C_BG, gc, C_BG],
            [gc, C_BG, gc],
        ],
        name="ghost",
        visible=True,
        collidable=False,
        layer=2,
        tags=["ghost"],
    )


def _grid_x_offset(grid_w):
    px_w = grid_w * CELL_SIZE
    return (CAM_W - px_w) // 2


def _grid_y_offset(grid_h):
    return HEADER_PX + (MAX_GRID_H - grid_h) * CELL_SIZE + 1


def _build_hud_sprites(lives, placed_count, total_blocks, inv_keys, selected_idx):
    result = []

    blocks_left = total_blocks - placed_count
    if total_blocks > 0:
        filled = CAM_W * blocks_left // total_blocks
        filled = max(1, min(filled, CAM_W)) if blocks_left > 0 else 0
    else:
        filled = 0
    bar_row = [C_BAR_FILL] * filled + [C_BAR_EMPTY] * (CAM_W - filled)
    sp = Sprite(
        pixels=[bar_row],
        name="blocks_bar",
        visible=True,
        collidable=False,
        layer=2,
        tags=["hud"],
    )
    sp.set_position(0, 0)
    result.append(sp)

    for i in range(MAX_LIVES):
        color = C_LIFE if i < lives else C_LIFE_LOST
        sp = Sprite(
            pixels=[[color] * CELL_SIZE for _ in range(CELL_SIZE)],
            name=f"life_{i}",
            visible=True,
            collidable=False,
            layer=2,
            tags=["hud"],
        )
        sp.set_position(i * (CELL_SIZE + 1), 2)
        result.append(sp)

    n_palette = len(inv_keys)
    if n_palette > 0:
        palette_x = CAM_W - n_palette * CELL_SIZE

        for i, bt in enumerate(inv_keys):
            bc = BLOCKS[bt]["color"]
            sp = Sprite(
                pixels=[[bc] * CELL_SIZE for _ in range(CELL_SIZE)],
                name=f"palette_{bt}",
                visible=True,
                collidable=False,
                layer=2,
                tags=["hud"],
            )
            sp.set_position(palette_x + i * CELL_SIZE, 2)
            result.append(sp)

        if 0 <= selected_idx < n_palette:
            ind_x = palette_x + selected_idx * CELL_SIZE
            sp = Sprite(
                pixels=[[C_INDICATOR] * CELL_SIZE],
                name="sel_indicator",
                visible=True,
                collidable=False,
                layer=3,
                tags=["hud"],
            )
            sp.set_position(ind_x, 5)
            result.append(sp)

    sp = Sprite(
        pixels=[[C_SEPARATOR] * CAM_W],
        name="separator",
        visible=True,
        collidable=False,
        layer=2,
        tags=["hud"],
    )
    sp.set_position(0, 6)
    result.append(sp)

    return result


def _build_grid_sprites(
    state_grid,
    color_grid,
    grid_w,
    grid_h,
    prize_cells,
    ghost_cells=None,
    ghost_color=0,
    ghost_valid=True,
):
    result = []
    gy_off = _grid_y_offset(grid_h)
    gx_off = _grid_x_offset(grid_w)

    for r in range(grid_h):
        for c in range(grid_w):
            px_x = gx_off + c * CELL_SIZE
            px_y = gy_off + r * CELL_SIZE

            if (r, c) in prize_cells and state_grid[r][c] <= 0:
                sp = _prize_sprite().clone()
                sp.set_position(px_x, px_y)
                result.append(sp)
            elif state_grid[r][c] == OBSTACLE:
                sp = _cell_sprite(C_OBSTACLE).clone()
                sp.set_position(px_x, px_y)
                result.append(sp)
            elif state_grid[r][c] > 0 and color_grid[r][c] != 0:
                sp = _cell_sprite(color_grid[r][c]).clone()
                sp.set_position(px_x, px_y)
                result.append(sp)

    if ghost_cells:
        for r, c in ghost_cells:
            if (
                0 <= r < grid_h
                and 0 <= c < grid_w
                and state_grid[r][c] <= 0
                and state_grid[r][c] != OBSTACLE
            ):
                px_x = gx_off + c * CELL_SIZE
                px_y = gy_off + r * CELL_SIZE
                sp = _ghost_cell_sprite(ghost_color, ghost_valid)
                sp.set_position(px_x, px_y)
                result.append(sp)

    return result


def _build_ground_sprites(grid_w, grid_h, ground_lo, ground_hi):
    result = []
    gy_off = _grid_y_offset(grid_h)
    gx_off = _grid_x_offset(grid_w)
    ground_y = gy_off + grid_h * CELL_SIZE
    px_w = grid_w * CELL_SIZE

    sp = Sprite(
        pixels=[[C_GROUND] * px_w],
        name="ground_full",
        visible=True,
        collidable=False,
        layer=2,
        tags=["ground"],
    )
    sp.set_position(gx_off, ground_y)
    result.append(sp)

    return result


def _build_border_sprites(grid_w, grid_h):
    result = []
    gy_off = _grid_y_offset(grid_h)
    gx_off = _grid_x_offset(grid_w)
    px_w = grid_w * CELL_SIZE
    px_h = grid_h * CELL_SIZE

    border_top_y = gy_off - 1
    border_bot_y = gy_off + px_h + 1
    border_total_w = px_w + 2
    border_total_h = px_h + 3

    top_sp = Sprite(
        pixels=[[C_BORDER] * border_total_w],
        name="border_top",
        visible=True,
        collidable=False,
        layer=3,
        tags=["border"],
    )
    top_sp.set_position(gx_off - 1, border_top_y)
    result.append(top_sp)

    bot_sp = Sprite(
        pixels=[[C_BORDER] * border_total_w],
        name="border_bot",
        visible=True,
        collidable=False,
        layer=3,
        tags=["border"],
    )
    bot_sp.set_position(gx_off - 1, border_bot_y)
    result.append(bot_sp)

    left_sp = Sprite(
        pixels=[[C_BORDER]] * border_total_h,
        name="border_left",
        visible=True,
        collidable=False,
        layer=3,
        tags=["border"],
    )
    left_sp.set_position(gx_off - 1, border_top_y)
    result.append(left_sp)

    right_sp = Sprite(
        pixels=[[C_BORDER]] * border_total_h,
        name="border_right",
        visible=True,
        collidable=False,
        layer=3,
        tags=["border"],
    )
    right_sp.set_position(gx_off + px_w, border_top_y)
    result.append(right_sp)

    return result


def _build_all_sprites(
    state_grid,
    color_grid,
    grid_w,
    grid_h,
    prize_cells,
    ground_lo,
    ground_hi,
    inv_keys,
    selected_idx,
    lives,
    placed_count,
    total_blocks,
    ghost_cells=None,
    ghost_color=0,
    ghost_valid=True,
):
    result = []
    result.extend(
        _build_hud_sprites(
            lives,
            placed_count,
            total_blocks,
            inv_keys,
            selected_idx,
        )
    )
    result.extend(
        _build_grid_sprites(
            state_grid,
            color_grid,
            grid_w,
            grid_h,
            prize_cells,
            ghost_cells,
            ghost_color,
            ghost_valid,
        )
    )
    result.extend(_build_ground_sprites(grid_w, grid_h, ground_lo, ground_hi))
    result.extend(_build_border_sprites(grid_w, grid_h))
    return result


def _build_levels():
    result = []
    for i, cfg in enumerate(LEVEL_CONFIGS):
        grid_w = cfg["grid_w"]
        grid_h = cfg["grid_h"]
        inv = {k: v for k, v in cfg["inventory"].items() if v > 0}
        inv_keys = [b for b in BLOCK_ORDER if b in inv]

        state_grid = [[EMPTY] * grid_w for _ in range(grid_h)]
        color_grid = [[0] * grid_w for _ in range(grid_h)]
        for item in cfg.get("obstacles", []):
            if isinstance(item, (list, tuple)) and len(item) == 2:
                r, c = int(item[0]), int(item[1])
            else:
                continue
            if 0 <= r < grid_h and 0 <= c < grid_w:
                state_grid[r][c] = OBSTACLE
                color_grid[r][c] = C_OBSTACLE

        prize = set()
        for item in cfg.get("prize", []):
            if isinstance(item, (list, tuple)) and len(item) == 2:
                r, c = int(item[0]), int(item[1])
            else:
                continue
            if 0 <= r < grid_h and 0 <= c < grid_w:
                prize.add((r, c))

        ground = cfg.get("ground", (0, grid_w - 1))

        sprites_list = _build_all_sprites(
            state_grid=state_grid,
            color_grid=color_grid,
            grid_w=grid_w,
            grid_h=grid_h,
            prize_cells=prize,
            ground_lo=ground[0],
            ground_hi=ground[1],
            inv_keys=inv_keys,
            selected_idx=0,
            lives=MAX_LIVES,
            placed_count=0,
            total_blocks=MAX_MOVES_PER_LEVEL[i]
            if i < len(MAX_MOVES_PER_LEVEL)
            else 500,
        )

        level = Level(
            sprites=sprites_list,
            grid_size=(CAM_W, CAM_H),
            data={
                "grid_w": grid_w,
                "grid_h": grid_h,
                "prize": list(prize),
                "inventory": {k: v for k, v in inv.items()},
                "obstacles": list(cfg.get("obstacles", [])),
                "ground": ground,
                "max_moves": MAX_MOVES_PER_LEVEL[i]
                if i < len(MAX_MOVES_PER_LEVEL)
                else 500,
            },
            name=cfg.get("name", ""),
        )
        result.append(level)
    return result


_levels_store = {}


def _get_levels():
    if "levels" not in _levels_store:
        _levels_store["levels"] = _build_levels()
    return _levels_store["levels"]


class Ha01(ARCBaseGame):
    _CURSOR_DELTAS = {
        GameAction.ACTION3: (0, -1),
        GameAction.ACTION4: (0, 1),
    }

    def __init__(self, seed: int = 0) -> None:
        self._grid_w = 0
        self._grid_h = 0
        self._state_grid = []
        self._color_grid = []
        self._prize_cells = set()
        self._ground_lo = 0
        self._ground_hi = 0
        self._inventory = {}
        self._inv_keys = []
        self._selected_idx = 0
        self._lives = MAX_LIVES
        self._placed_count = 0
        self._total_blocks = 0
        self._cursor_r = 0
        self._cursor_c = 0
        self._moves_used = 0
        self._max_moves = 0
        self._history = []
        self._rng = random.Random(seed)

        cam = Camera(
            x=0,
            y=0,
            width=CAM_W,
            height=CAM_H,
            background=BACKGROUND_COLOR,
            letter_box=PADDING_COLOR,
            interfaces=[],
        )
        super().__init__(
            "ha01",
            _get_levels(),
            cam,
            available_actions=[0, 1, 2, 3, 4, 5, 6, 7],
        )

    @property
    def lives(self):
        return self._lives

    @property
    def placed_count(self):
        return self._placed_count

    @property
    def total_blocks(self):
        return self._total_blocks

    @property
    def moves_used(self):
        return self._moves_used

    @property
    def max_moves(self):
        return self._max_moves

    @property
    def inventory(self):
        return dict(self._inventory)

    @property
    def inv_keys(self):
        return list(self._inv_keys)

    @property
    def grid_h(self):
        return self._grid_h

    @property
    def grid_w(self):
        return self._grid_w

    @property
    def selected_block(self):
        return self._selected_block()

    @property
    def cursor_pos(self):
        return (self._cursor_r, self._cursor_c)

    def on_set_level(self, level: Level) -> None:
        self._grid_w = level.get_data("grid_w")
        self._grid_h = level.get_data("grid_h")
        self._lives = MAX_LIVES
        self._placed_count = 0
        self._max_moves = level.get_data("max_moves")
        self._moves_used = 0

        self._state_grid = [[EMPTY] * self._grid_w for _ in range(self._grid_h)]
        self._color_grid = [[0] * self._grid_w for _ in range(self._grid_h)]

        for item in level.get_data("obstacles"):
            if isinstance(item, (list, tuple)) and len(item) == 2:
                r, c = int(item[0]), int(item[1])
            else:
                continue
            if 0 <= r < self._grid_h and 0 <= c < self._grid_w:
                self._state_grid[r][c] = OBSTACLE
                self._color_grid[r][c] = C_OBSTACLE

        self._prize_cells = set()
        for item in level.get_data("prize"):
            if isinstance(item, (list, tuple)) and len(item) == 2:
                r, c = int(item[0]), int(item[1])
                if 0 <= r < self._grid_h and 0 <= c < self._grid_w:
                    self._prize_cells.add((r, c))

        ground = level.get_data("ground")
        ground_width = ground[1] - ground[0] + 1
        max_lo = self._grid_w - ground_width
        self._ground_lo = self._rng.randint(0, max_lo)
        self._ground_hi = self._ground_lo + ground_width - 1

        inv_data = level.get_data("inventory")
        self._inventory = {k: v for k, v in inv_data.items() if v > 0}
        self._total_blocks = sum(self._inventory.values())
        self._rebuild_inv_keys()
        self._selected_idx = 0

        self._cursor_c = self._rng.randint(self._ground_lo, self._ground_hi)
        self._cursor_r = self._initial_cursor_row()
        self._apply_gravity()

        self._history = []
        self._rebuild_sprites()

    def _initial_cursor_row(self):
        btype = self._selected_block()
        if btype is None:
            return self._grid_h - 1
        max_dr = max(r for r, _ in BLOCKS[btype]["shape"])
        return self._grid_h - 1 - max_dr

    def _rebuild_inv_keys(self):
        self._inv_keys = [b for b in BLOCK_ORDER if b in self._inventory]

    def _selected_block(self):
        if self._inv_keys and 0 <= self._selected_idx < len(self._inv_keys):
            return self._inv_keys[self._selected_idx]
        return None

    def _block_height(self, btype):
        rows = [r for r, c in BLOCKS[btype]["shape"]]
        return max(rows) - min(rows) + 1

    def _find_top_of_placed_blocks(self):
        for r in range(self._grid_h):
            for c in range(self._grid_w):
                if self._state_grid[r][c] > 0:
                    return r
        return self._grid_h

    def _is_valid_cursor_position(self, btype, anchor_r, col):
        max_dr = max(r for r, _ in BLOCKS[btype]["shape"])
        if anchor_r + max_dr >= self._grid_h:
            return False
        cells = self._cells_for(btype, anchor_r, col)
        for r, c in cells:
            if not self._in_bounds(r, c):
                return False
            if self._state_grid[r][c] == OBSTACLE or self._state_grid[r][c] > 0:
                return False
            if (r, c) in self._prize_cells:
                return False
        ok, _ = self._validate(btype, anchor_r, col)
        return ok

    def _find_safe_cursor_row(self, col, hint_row=None):
        btype = self._selected_block()
        if btype is None:
            return 0
        max_dr = max(r for r, _ in BLOCKS[btype]["shape"])

        top_occupied = self._grid_h
        for r in range(self._grid_h):
            if self._state_grid[r][col] == OBSTACLE or self._state_grid[r][col] > 0:
                top_occupied = r
                break

        search_start = min(top_occupied - 1, self._grid_h - 1 - max_dr)
        if search_start < 0:
            search_start = 0

        for anchor_r in range(search_start, -1, -1):
            if self._is_valid_cursor_position(btype, anchor_r, col):
                return anchor_r

        if hint_row is not None:
            for offset in range(self._grid_h):
                for try_r in [hint_row - offset, hint_row + offset]:
                    if try_r < 0 or try_r + max_dr >= self._grid_h:
                        continue
                    if self._is_valid_cursor_position(btype, try_r, col):
                        return try_r
        return 0

    def _in_bounds(self, r, c):
        return 0 <= r < self._grid_h and 0 <= c < self._grid_w

    def _has_ground(self, col):
        return self._ground_lo <= col <= self._ground_hi

    def _is_supported(self, row, col):
        below = row + 1
        if below >= self._grid_h:
            return self._has_ground(col)
        cell_value = self._state_grid[below][col]
        if cell_value == OBSTACLE:
            return True
        if cell_value > 0:
            return True
        return False

    def _is_ground_supported(self, row, col):
        below = row + 1
        return below >= self._grid_h and self._has_ground(col)

    @staticmethod
    def _cells_for(btype, ar, ac):
        return [(ar + dr, ac + dc) for dr, dc in BLOCKS[btype]["shape"]]

    def _validate(self, btype, ar, ac):
        cells = self._cells_for(btype, ar, ac)

        for r, c in cells:
            if not self._in_bounds(r, c):
                return False, "Out of bounds"
            cell_value = self._state_grid[r][c]
            if cell_value == OBSTACLE:
                return False, "Blocked by obstacle"
            if cell_value > 0:
                return False, "Overlaps a block"

        col_bot = {}
        for r, c in cells:
            if c not in col_bot or r > col_bot[c]:
                col_bot[c] = r

        supported = 0
        unsupported = 0
        for c, br in col_bot.items():
            if self._is_supported(br, c):
                supported += 1
            else:
                unsupported += 1

        if supported == 0:
            return False, "No support below"

        if supported < unsupported:
            return False, "Too much overhang -- block tips over"

        if self._placed_count == 0:
            ground_supported = sum(
                1 for c, br in col_bot.items() if self._is_ground_supported(br, c)
            )
            if ground_supported == 0:
                return False, "First block must be placed on the ground"
        else:
            touching = False
            for r, c in cells:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if self._in_bounds(nr, nc) and self._state_grid[nr][nc] > 0:
                        touching = True
                        break
                if touching:
                    break
            if not touching:
                return False, "Must build adjacent to existing blocks"

        return True, "OK"

    def _reset_level_progress(self):
        self._state_grid = [[EMPTY] * self._grid_w for _ in range(self._grid_h)]
        self._color_grid = [[0] * self._grid_w for _ in range(self._grid_h)]

        for item in self.current_level.get_data("obstacles"):
            if isinstance(item, (list, tuple)) and len(item) == 2:
                r, c = int(item[0]), int(item[1])
            else:
                continue
            if 0 <= r < self._grid_h and 0 <= c < self._grid_w:
                self._state_grid[r][c] = OBSTACLE
                self._color_grid[r][c] = C_OBSTACLE

        inv_data = self.current_level.get_data("inventory")
        self._inventory = {k: v for k, v in inv_data.items() if v > 0}
        self._total_blocks = sum(self._inventory.values())
        self._placed_count = 0
        self._moves_used = 0
        self._rebuild_inv_keys()
        self._selected_idx = 0

        self._cursor_c = self._rng.randint(self._ground_lo, self._ground_hi)
        self._cursor_r = self._initial_cursor_row()
        self._apply_gravity()
        self._history = []

    def _try_place(self, btype, ar, ac):
        if btype not in self._inventory or self._inventory[btype] <= 0:
            return

        ok, reason = self._validate(btype, ar, ac)

        if ok:
            bdef = BLOCKS[btype]
            cells = self._cells_for(btype, ar, ac)

            for r, c in cells:
                self._state_grid[r][c] = 1
                self._color_grid[r][c] = bdef["color"]

            self._inventory[btype] -= 1
            if self._inventory[btype] <= 0:
                del self._inventory[btype]
            self._placed_count += 1

            self._rebuild_inv_keys()
            if self._selected_idx >= len(self._inv_keys):
                self._selected_idx = max(0, len(self._inv_keys) - 1)

            for r in range(self._grid_h):
                for c in range(self._grid_w):
                    if (r, c) in self._prize_cells and self._state_grid[r][c] > 0:
                        self._rebuild_sprites()
                        self.next_level()
                        return

            self._rebuild_sprites()
        else:
            self._lives -= 1
            if self._lives <= 0:
                self._rebuild_sprites()
                self.lose()
            else:
                self._reset_level_progress()
                self._rebuild_sprites()

    def _compute_ghost(self):
        btype = self._selected_block()
        if btype is None:
            return None, 0, False
        cells = self._cells_for(btype, self._cursor_r, self._cursor_c)
        visible = [(r, c) for r, c in cells if self._in_bounds(r, c)]
        if not visible:
            return None, 0, False
        valid, _ = self._validate(btype, self._cursor_r, self._cursor_c)
        return visible, BLOCKS[btype]["color"], valid

    def _rebuild_sprites(self):
        ghost_cells, ghost_color, ghost_valid = self._compute_ghost()
        self.current_level.remove_all_sprites()
        for sp in _build_all_sprites(
            state_grid=self._state_grid,
            color_grid=self._color_grid,
            grid_w=self._grid_w,
            grid_h=self._grid_h,
            prize_cells=self._prize_cells,
            ground_lo=self._ground_lo,
            ground_hi=self._ground_hi,
            inv_keys=self._inv_keys,
            selected_idx=self._selected_idx,
            lives=self._lives,
            placed_count=self._moves_used,
            total_blocks=self._max_moves,
            ghost_cells=ghost_cells,
            ghost_color=ghost_color,
            ghost_valid=ghost_valid,
        ):
            self.current_level.add_sprite(sp)

    def _palette_hit(self, px_x, px_y):
        n = len(self._inv_keys)
        if n == 0:
            return None
        palette_x = CAM_W - n * CELL_SIZE
        palette_y_lo = 2
        palette_y_hi = 2 + CELL_SIZE
        if not (palette_y_lo <= px_y < palette_y_hi):
            return None
        if px_x < palette_x or px_x >= CAM_W:
            return None
        idx = (px_x - palette_x) // CELL_SIZE
        if 0 <= idx < n:
            return idx
        return None

    def step(self) -> None:
        if self.action.id == GameAction.RESET:
            self.complete_action()
            return
        self._internal_step()

    def _handle_move_exceeded(self):
        self._lives -= 1
        if self._lives <= 0:
            self._rebuild_sprites()
            self.lose()
        else:
            self._reset_level_progress()
            self._rebuild_sprites()

    def _handle_place_at_cursor(self):
        btype = self._selected_block()
        if btype is not None:
            self._try_place(btype, self._cursor_r, self._cursor_c)

    def _handle_click_place(self):
        display_x = self.action.data.get("x", 0)
        display_y = self.action.data.get("y", 0)
        coords = self.camera.display_to_grid(display_x, display_y)

        if coords:
            gx, gy = coords

            palette_idx = self._palette_hit(gx, gy)
            if palette_idx is not None:
                self._selected_idx = palette_idx
                self._rebuild_sprites()
                return

            gy_off = _grid_y_offset(self._grid_h)
            gx_off = _grid_x_offset(self._grid_w)
            btype = self._selected_block()
            if btype is not None:
                cell_col = (gx - gx_off) // CELL_SIZE
                pixel_row = gy - gy_off
                cell_row = pixel_row // CELL_SIZE

                if 0 <= cell_row < self._grid_h and 0 <= cell_col < self._grid_w:
                    self._try_place(btype, cell_row, cell_col)

    def _handle_cycle_block(self, forward=True):
        if not self._inv_keys:
            return
        if forward:
            self._selected_idx = (self._selected_idx + 1) % len(self._inv_keys)
        else:
            self._selected_idx = (self._selected_idx - 1) % len(self._inv_keys)

    def _save_undo_snapshot(self):
        snapshot = {
            "state_grid": [row[:] for row in self._state_grid],
            "color_grid": [row[:] for row in self._color_grid],
            "inventory": dict(self._inventory),
            "inv_keys": list(self._inv_keys),
            "selected_idx": self._selected_idx,
            "placed_count": self._placed_count,
            "cursor_r": self._cursor_r,
            "cursor_c": self._cursor_c,
            "lives": self._lives,
        }
        self._history.append(snapshot)

    def _restore_undo_snapshot(self):
        if not self._history:
            return
        snapshot = self._history.pop()
        self._state_grid = snapshot["state_grid"]
        self._color_grid = snapshot["color_grid"]
        self._inventory = snapshot["inventory"]
        self._inv_keys = snapshot["inv_keys"]
        self._selected_idx = snapshot["selected_idx"]
        self._placed_count = snapshot["placed_count"]
        self._cursor_r = snapshot["cursor_r"]
        self._cursor_c = snapshot["cursor_c"]
        self._lives = snapshot["lives"]
        self._rebuild_sprites()

    def _auto_move_cursor_up(self):
        btype = self._selected_block()
        if btype is None:
            return
        block_h = max(r for r, _ in BLOCKS[btype]["shape"]) + 1
        self._cursor_r = max(0, self._cursor_r - block_h)
        self._apply_gravity()

    def _has_collision(self, btype, row, col):
        for r, c in self._cells_for(btype, row, col):
            if not self._in_bounds(r, c):
                continue
            val = self._state_grid[r][c]
            if val == OBSTACLE or val > 0:
                return True
        return False

    def _push_up_until_clear(self, btype):
        while self._cursor_r > 0 and self._has_collision(
            btype, self._cursor_r, self._cursor_c
        ):
            self._cursor_r -= 1

    def _apply_gravity(self):
        btype = self._selected_block()
        if btype is None:
            return
        max_dr = max(r for r, _ in BLOCKS[btype]["shape"])
        max_row = self._grid_h - 1 - max_dr
        self._cursor_r = min(self._cursor_r, max_row)

        if self._has_collision(btype, self._cursor_r, self._cursor_c):
            self._push_up_until_clear(btype)
            return

        while self._cursor_r < max_row:
            next_r = self._cursor_r + 1
            if self._has_collision(btype, next_r, self._cursor_c):
                break
            cells = self._cells_for(btype, next_r, self._cursor_c)
            col_bot = {}
            for r, c in cells:
                if not self._in_bounds(r, c):
                    continue
                if c not in col_bot or r > col_bot[c]:
                    col_bot[c] = r
            any_supported = False
            for c, br in col_bot.items():
                if self._is_supported(br, c):
                    any_supported = True
                    break
            if any_supported:
                self._cursor_r = next_r
                break
            self._cursor_r = next_r

    def _internal_step(self):
        action_id = self.action.id

        if action_id == GameAction.ACTION7:
            if self._history:
                self._moves_used += 1
                self._restore_undo_snapshot()
            self.complete_action()
            return

        self._moves_used += 1

        if self._moves_used > self._max_moves:
            self._handle_move_exceeded()
            self.complete_action()
            return

        if action_id == GameAction.ACTION1:
            self._handle_cycle_block(forward=True)
            self._apply_gravity()
            self._rebuild_sprites()
            self.complete_action()
            return

        if action_id == GameAction.ACTION2:
            btype = self._selected_block()
            if btype is not None:
                max_dr = max(r for r, _ in BLOCKS[btype]["shape"])
                max_row = self._grid_h - 1 - max_dr
                self._cursor_r = min(max_row, self._cursor_r + 1)
            else:
                self._cursor_r = min(self._grid_h - 1, self._cursor_r + 1)
            self._rebuild_sprites()
            self.complete_action()
            return

        if action_id in self._CURSOR_DELTAS:
            dr, dc = self._CURSOR_DELTAS[action_id]
            new_c = max(0, min(self._grid_w - 1, self._cursor_c + dc))
            if new_c == self._cursor_c:
                self._rebuild_sprites()
                self.complete_action()
                return
            btype = self._selected_block()
            if btype is not None and self._has_collision(btype, self._cursor_r, new_c):
                saved_c = self._cursor_c
                saved_r = self._cursor_r
                self._cursor_c = new_c
                self._push_up_until_clear(btype)
                if self._has_collision(btype, self._cursor_r, self._cursor_c):
                    self._cursor_c = saved_c
                    self._cursor_r = saved_r
                else:
                    self._apply_gravity()
            else:
                self._cursor_c = new_c
                self._apply_gravity()
            self._rebuild_sprites()
            self.complete_action()
            return

        if action_id == GameAction.ACTION5:
            self._save_undo_snapshot()
            self._handle_place_at_cursor()
            self._auto_move_cursor_up()
            self._rebuild_sprites()
            self.complete_action()
            return

        if action_id == GameAction.ACTION6:
            self._save_undo_snapshot()
            self._handle_click_place()
            self.complete_action()
            return

        self.complete_action()


class PuzzleEnvironment:
    _ACTION_MAP = {
        "reset": GameAction.RESET,
        "up": GameAction.ACTION1,
        "down": GameAction.ACTION2,
        "left": GameAction.ACTION3,
        "right": GameAction.ACTION4,
        "select": GameAction.ACTION5,
        "click": GameAction.ACTION6,
        "undo": GameAction.ACTION7,
    }

    _VALID_ACTIONS = list(_ACTION_MAP.keys())

    def __init__(self, seed: int = 0) -> None:
        self._engine = Ha01(seed=seed)
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
            return StepResult(
                state=state, reward=0.0, done=False, info={"action": "reset"}
            )

        click_data = {}
        base_action = action

        if action.startswith("click:"):
            try:
                parts = action.split(":")
                coords = parts[1].split(",")
                click_data = {"x": int(coords[0]), "y": int(coords[1])}
            except (IndexError, ValueError):
                click_data = {"x": 0, "y": 0}
            base_action = "click"
        elif action == "click":
            click_data = {"x": 0, "y": 0}
            base_action = "click"
        elif action.startswith("click "):
            try:
                parts = action.split()
                click_data = {"x": int(parts[1]), "y": int(parts[2])}
            except (IndexError, ValueError):
                click_data = {"x": 0, "y": 0}
            base_action = "click"

        if base_action not in self._ACTION_MAP:
            return StepResult(
                state=self._build_game_state(done=self._done),
                reward=0.0,
                done=self._done,
                info={"action": action, "reason": "invalid_action"},
            )

        self._last_action_was_reset = False
        self._total_turns += 1

        game_action = self._ACTION_MAP[base_action]
        info = {"action": action}
        level_before = e.level_index

        action_input = ActionInput(id=game_action)
        if click_data:
            action_input = ActionInput(id=game_action, data=click_data)
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
                "game_over": getattr(getattr(self._engine, "_state", None), "name", "")
                == "GAME_OVER",
                "done": done,
                "info": {},
            },
        )

    def _build_text_observation(self) -> str:
        e = self._engine
        level_idx = e.level_index
        cfg = LEVEL_CONFIGS[level_idx]
        lines = []

        lines.append(f'Level {level_idx + 1}/{len(LEVEL_CONFIGS)} "{cfg["name"]}"')
        lines.append(
            f"Lives: {e._lives}/{MAX_LIVES} | "
            f"Blocks placed: {e._placed_count}/{e._total_blocks} | "
            f"Moves: {e._moves_used}/{e._max_moves}"
        )

        inv_parts = [f"{bt}:{cnt}" for bt, cnt in e._inventory.items()]
        sel = e._selected_block() or "none"
        lines.append(f"Inventory: {' '.join(inv_parts)} | Selected: {sel}")
        lines.append(f"Cursor: row={e._cursor_r} col={e._cursor_c}")
        lines.append("")

        lines.append("   " + " ".join(str(c % 10) for c in range(e._grid_w)))
        for r in range(e._grid_h):
            cells_str = []
            for c in range(e._grid_w):
                is_cursor = r == e._cursor_r and c == e._cursor_c
                if (r, c) in e._prize_cells and e._state_grid[r][c] <= 0:
                    cells_str.append("@" if is_cursor else "*")
                else:
                    ch = (
                        _TILE_CHAR.get(e._color_grid[r][c], ".")
                        if e._state_grid[r][c] != OBSTACLE
                        else "#"
                    )
                    if is_cursor and ch == ".":
                        ch = "+"
                    cells_str.append(ch)
            lines.append(f"{r:2d} " + " ".join(cells_str))

        ground_str = "".join(
            "=" if e._ground_lo <= c <= e._ground_hi else " " for c in range(e._grid_w)
        )
        lines.append(f"   {ground_str}")

        return "\n".join(lines)

    def is_done(self) -> bool:
        return self._done

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


class ArcGameEnv(gym.Env):
    metadata: Dict[str, Any] = {"render_modes": ["rgb_array"], "render_fps": 5}
    ACTION_LIST: List[str] = [
        "reset",
        "up",
        "down",
        "left",
        "right",
        "select",
        "click",
        "undo",
    ]
    OBS_HEIGHT: int = 64
    OBS_WIDTH: int = 64

    def __init__(self, seed: int = 0, render_mode: Optional[str] = None) -> None:
        super().__init__()
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode: {render_mode}")
        self.render_mode: Optional[str] = render_mode
        self._seed = seed
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
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
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
        if self._env is None:
            return np.zeros((self.OBS_HEIGHT, self.OBS_WIDTH, 3), dtype=np.uint8)
        rgb = self._env.render(mode="rgb_array")
        h, w = rgb.shape[:2]
        if h != self.OBS_HEIGHT or w != self.OBS_WIDTH:
            rgb = self._resize_nearest(rgb, self.OBS_HEIGHT, self.OBS_WIDTH)
        return rgb

    @staticmethod
    def _resize_nearest(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        src_h, src_w = img.shape[0], img.shape[1]
        row_idx = (np.arange(target_h) * src_h // target_h).astype(int)
        col_idx = (np.arange(target_w) * src_w // target_w).astype(int)
        return img[np.ix_(row_idx, col_idx)].astype(np.uint8)

    def _build_info(
        self, state: GameState, step_info: Optional[Dict] = None
    ) -> Dict[str, Any]:
        info = {
            "text_observation": state.text_observation,
            "valid_actions": state.valid_actions,
            "turn": state.turn,
            "game_metadata": state.metadata,
        }
        if step_info:
            info["step_info"] = step_info
        return info
