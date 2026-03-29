import struct
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

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

PALETTE_2 = [1, 2]
PALETTE_3 = [1, 2, 3]

PLAYER_COLOR = 4
FROZEN_ACCENT = 5
DANGER_COLOR = 6
LIFE_COLOR = 2
BG_COLOR = 0
BAR_FULL = 3
BAR_EMPTY = 2

_SHIFT_SEQ = "31415926535897932384626433832795028841971693993751"


def _seq(n: int) -> int:
    idx = (n - 1) % len(_SHIFT_SEQ)
    return int(_SHIFT_SEQ[idx])


sprites = {
    "tile": Sprite(
        pixels=[
            [1, 1],
            [1, 1],
        ],
        name="tile",
        collidable=False,
        layer=0,
        tags=["tile"],
    ),
    "frozen": Sprite(
        pixels=[
            [1, FROZEN_ACCENT],
            [FROZEN_ACCENT, 1],
        ],
        name="frozen",
        collidable=False,
        layer=0,
        tags=["frozen"],
    ),
    "danger": Sprite(
        pixels=[
            [DANGER_COLOR, BG_COLOR],
            [BG_COLOR, DANGER_COLOR],
        ],
        name="danger",
        collidable=False,
        layer=0,
        tags=["danger"],
    ),
    "player": Sprite(
        pixels=[
            [PLAYER_COLOR, PLAYER_COLOR],
            [PLAYER_COLOR, PLAYER_COLOR],
        ],
        name="player",
        collidable=False,
        layer=2,
        tags=["player"],
    ),
    "bar_pip": Sprite(
        pixels=[[BAR_FULL]],
        name="bar_pip",
        collidable=False,
        layer=3,
        tags=["bar"],
    ),
    "life_pip": Sprite(
        pixels=[[LIFE_COLOR, LIFE_COLOR]],
        name="life_pip",
        collidable=False,
        layer=3,
        tags=["life"],
    ),
}

TILE_SIZE = 2
CAM_SIZE = 32
BAR_Y = 31
BAR_WIDTH = 32


def _tile_pos(gx: int, gy: int, ox: int, oy: int) -> Tuple[int, int]:
    return (ox + gx * TILE_SIZE, oy + gy * TILE_SIZE)


def _grid_offset(grid_w: int, grid_h: int) -> Tuple[int, int]:
    ox = (CAM_SIZE - grid_w * TILE_SIZE) // 2
    oy = (CAM_SIZE - grid_h * TILE_SIZE) // 2
    return (ox, oy)


def _build_level_sprites(
    grid_w: int,
    grid_h: int,
    initial_colors: List[List[int]],
    palette: List[int],
    player_gx: int,
    player_gy: int,
    frozen_set: Set[Tuple[int, int]],
    danger_set: Set[Tuple[int, int]],
    move_limit: int,
    lives: int,
) -> List[Sprite]:
    ox, oy = _grid_offset(grid_w, grid_h)
    level_sprites: List[Sprite] = []

    for gy in range(grid_h):
        for gx in range(grid_w):
            px, py = _tile_pos(gx, gy, ox, oy)
            if (gx, gy) in danger_set:
                s = sprites["danger"].clone().set_position(px, py)
            elif (gx, gy) in frozen_set:
                cidx = initial_colors[gy][gx]
                color = palette[cidx]
                s = sprites["frozen"].clone().set_position(px, py)
                s.color_remap(1, color)
            else:
                cidx = initial_colors[gy][gx]
                color = palette[cidx]
                s = sprites["tile"].clone().set_position(px, py)
                s.color_remap(1, color)
            level_sprites.append(s)

    ppx, ppy = _tile_pos(player_gx, player_gy, ox, oy)
    level_sprites.append(sprites["player"].clone().set_position(ppx, ppy))

    for i in range(BAR_WIDTH):
        pip = sprites["bar_pip"].clone().set_position(i, BAR_Y)
        pip.color_remap(None, BAR_FULL)
        level_sprites.append(pip)

    for i in range(lives):
        lp = sprites["life_pip"].clone().set_position(i * 3, 0)
        level_sprites.append(lp)

    return level_sprites


L1_W, L1_H = 8, 8
L1_LIMIT = 50
L1_GRID = [
    [0, 1, 1, 1, 0, 0, 0, 0],
    [1, 0, 0, 0, 1, 1, 1, 1],
    [1, 0, 0, 0, 1, 1, 1, 1],
    [1, 0, 0, 0, 1, 1, 1, 1],
    [0, 1, 1, 1, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 0, 0, 0],
]

L2_W, L2_H = 8, 8
L2_LIMIT = 45
L2_GRID = [
    [0, 2, 1, 1, 1, 1, 0, 0],
    [1, 2, 1, 2, 2, 2, 1, 1],
    [0, 2, 0, 2, 1, 1, 0, 0],
    [1, 0, 2, 2, 0, 2, 1, 1],
    [1, 0, 2, 2, 2, 0, 1, 1],
    [0, 2, 1, 1, 1, 1, 0, 0],
    [0, 2, 1, 1, 1, 1, 0, 0],
    [0, 2, 1, 1, 1, 1, 0, 0],
]

L3_W, L3_H = 10, 10
L3_LIMIT = 40
L3_GRID = [
    [1, 1, 2, 2, 2, 2, 1, 1, 1, 1],
    [2, 2, 1, 0, 0, 0, 2, 2, 2, 2],
    [1, 1, 0, 1, 2, 2, 1, 1, 1, 1],
    [1, 1, 0, 0, 0, 2, 1, 1, 1, 1],
    [1, 1, 0, 0, 2, 0, 1, 1, 1, 1],
    [0, 0, 2, 2, 1, 1, 0, 0, 0, 0],
    [0, 0, 2, 2, 1, 1, 0, 0, 0, 0],
    [0, 0, 2, 2, 1, 1, 0, 0, 0, 0],
    [0, 0, 2, 2, 1, 1, 0, 0, 0, 0],
    [0, 0, 2, 2, 1, 1, 0, 0, 0, 0],
]

L4_W, L4_H = 10, 10
L4_LIMIT = 35
L4_LIVES = 3
L4_FROZEN = {(0, 9), (9, 0), (9, 9)}
L4_DANGER = {(1, 1), (3, 1), (5, 2), (7, 3)}
L4_GRID = [
    [2, 2, 1, 0, 0, 0, 2, 2, 2, 0],
    [2, 0, 1, 0, 0, 0, 2, 2, 2, 2],
    [0, 0, 2, 2, 1, 0, 0, 0, 0, 0],
    [1, 1, 1, 2, 0, 2, 1, 0, 1, 1],
    [1, 1, 1, 2, 2, 0, 1, 1, 1, 1],
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
]

L5_W, L5_H = 12, 12
L5_LIMIT = 30
L5_LIVES = 3
L5_FROZEN = {(0, 11), (11, 0), (11, 11), (5, 5), (6, 6)}
L5_DANGER = {(1, 1), (2, 2), (4, 1), (5, 2), (8, 4)}
L5_GRID = [
    [1, 1, 2, 2, 2, 1, 2, 0, 1, 1, 1, 0],
    [2, 0, 1, 0, 0, 2, 0, 1, 2, 2, 2, 2],
    [1, 1, 0, 1, 2, 0, 2, 0, 1, 1, 1, 1],
    [2, 2, 1, 1, 1, 2, 1, 1, 2, 2, 2, 2],
    [1, 1, 0, 0, 2, 1, 2, 1, 0, 1, 1, 1],
    [1, 1, 0, 0, 2, 0, 2, 1, 1, 1, 1, 1],
    [0, 0, 2, 2, 1, 0, 0, 2, 0, 0, 0, 0],
    [0, 0, 2, 2, 1, 0, 1, 2, 0, 0, 0, 0],
    [0, 0, 2, 2, 1, 0, 1, 2, 0, 0, 0, 0],
    [0, 0, 2, 2, 1, 0, 1, 2, 0, 0, 0, 0],
    [0, 0, 2, 2, 1, 0, 1, 2, 0, 0, 0, 0],
    [0, 0, 2, 2, 1, 0, 1, 2, 0, 0, 0, 0],
]


def _make_levels() -> List[Level]:
    return [
        Level(
            sprites=_build_level_sprites(
                L1_W, L1_H, L1_GRID, PALETTE_2, 0, 0, set(), set(), L1_LIMIT, 3
            ),
            grid_size=(CAM_SIZE, CAM_SIZE),
            data={
                "grid_w": L1_W,
                "grid_h": L1_H,
                "num_colors": 2,
                "palette": PALETTE_2,
                "initial_colors": L1_GRID,
                "player_gx": 0,
                "player_gy": 0,
                "frozen": [],
                "danger": [],
                "move_limit": L1_LIMIT,
                "lives": 3,
            },
            name="Level 1",
        ),
        Level(
            sprites=_build_level_sprites(
                L2_W, L2_H, L2_GRID, PALETTE_3, 0, 0, set(), set(), L2_LIMIT, 3
            ),
            grid_size=(CAM_SIZE, CAM_SIZE),
            data={
                "grid_w": L2_W,
                "grid_h": L2_H,
                "num_colors": 3,
                "palette": PALETTE_3,
                "initial_colors": L2_GRID,
                "player_gx": 0,
                "player_gy": 0,
                "frozen": [],
                "danger": [],
                "move_limit": L2_LIMIT,
                "lives": 3,
            },
            name="Level 2",
        ),
        Level(
            sprites=_build_level_sprites(
                L3_W, L3_H, L3_GRID, PALETTE_3, 0, 0, set(), set(), L3_LIMIT, 3
            ),
            grid_size=(CAM_SIZE, CAM_SIZE),
            data={
                "grid_w": L3_W,
                "grid_h": L3_H,
                "num_colors": 3,
                "palette": PALETTE_3,
                "initial_colors": L3_GRID,
                "player_gx": 0,
                "player_gy": 0,
                "frozen": [],
                "danger": [],
                "move_limit": L3_LIMIT,
                "lives": 3,
            },
            name="Level 3",
        ),
        Level(
            sprites=_build_level_sprites(
                L4_W,
                L4_H,
                L4_GRID,
                PALETTE_3,
                0,
                0,
                L4_FROZEN,
                L4_DANGER,
                L4_LIMIT,
                L4_LIVES,
            ),
            grid_size=(CAM_SIZE, CAM_SIZE),
            data={
                "grid_w": L4_W,
                "grid_h": L4_H,
                "num_colors": 3,
                "palette": PALETTE_3,
                "initial_colors": L4_GRID,
                "player_gx": 0,
                "player_gy": 0,
                "frozen": [[0, 9], [9, 0], [9, 9]],
                "danger": [[1, 1], [3, 1], [5, 2], [7, 3]],
                "move_limit": L4_LIMIT,
                "lives": L4_LIVES,
            },
            name="Level 4",
        ),
        Level(
            sprites=_build_level_sprites(
                L5_W,
                L5_H,
                L5_GRID,
                PALETTE_3,
                0,
                0,
                L5_FROZEN,
                L5_DANGER,
                L5_LIMIT,
                L5_LIVES,
            ),
            grid_size=(CAM_SIZE, CAM_SIZE),
            data={
                "grid_w": L5_W,
                "grid_h": L5_H,
                "num_colors": 3,
                "palette": PALETTE_3,
                "initial_colors": L5_GRID,
                "player_gx": 0,
                "player_gy": 0,
                "frozen": [[0, 11], [11, 0], [11, 11], [5, 5], [6, 6]],
                "danger": [[1, 1], [2, 2], [4, 1], [5, 2], [8, 4]],
                "move_limit": L5_LIMIT,
                "lives": L5_LIVES,
            },
            name="Level 5",
        ),
    ]


class Pi01(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._lost = False
        self._won = False
        self._level_num = 1
        self._level_just_won = False
        self._level_won_num = 0
        super().__init__(
            game_id="pi01",
            levels=_make_levels(),
            camera=Camera(
                x=0,
                y=0,
                width=CAM_SIZE,
                height=CAM_SIZE,
                background=BG_COLOR,
                letter_box=BG_COLOR,
            ),
            available_actions=[0, 1, 2, 3, 4, 7],
        )

    def on_set_level(self, level: Level) -> None:
        self._lost = False
        self._history: List[Dict] = []

        self.grid_w: int = self.current_level.get_data("grid_w")
        self.grid_h: int = self.current_level.get_data("grid_h")
        self.num_colors: int = self.current_level.get_data("num_colors")
        self.palette: List[int] = self.current_level.get_data("palette")
        self.ox, self.oy = _grid_offset(self.grid_w, self.grid_h)
        self.move_count: int = 0
        self.move_limit: int = self.current_level.get_data("move_limit")
        self.lives: int = self.current_level.get_data("lives")

        frozen_data = self.current_level.get_data("frozen")
        self.frozen_set: Set[Tuple[int, int]] = set()
        for pair in frozen_data:
            self.frozen_set.add((pair[0], pair[1]))

        danger_data = self.current_level.get_data("danger")
        self.danger_set: Set[Tuple[int, int]] = set()
        for pair in danger_data:
            self.danger_set.add((pair[0], pair[1]))

        self.no_shift_set: Set[Tuple[int, int]] = self.frozen_set | self.danger_set

        initial = self.current_level.get_data("initial_colors")
        self.color_grid: List[List[int]] = []
        for gy in range(self.grid_h):
            row: List[int] = []
            for gx in range(self.grid_w):
                row.append(initial[gy][gx])
            self.color_grid.append(row)

        self.player_sprite = self.current_level.get_sprites_by_tag("player")[0]
        self.player_gx: int = self.current_level.get_data("player_gx")
        self.player_gy: int = self.current_level.get_data("player_gy")
        ppx, ppy = _tile_pos(self.player_gx, self.player_gy, self.ox, self.oy)
        self.player_sprite.set_position(ppx, ppy)

        self.tile_sprites: Dict[Tuple[int, int], Sprite] = {}
        tile_list = self.current_level.get_sprites_by_tag("tile")
        frozen_list = self.current_level.get_sprites_by_tag("frozen")
        for s in tile_list + frozen_list:
            gx = (s.x - self.ox) // TILE_SIZE
            gy = (s.y - self.oy) // TILE_SIZE
            self.tile_sprites[(gx, gy)] = s

        self.bar_sprites: List[Sprite] = self.current_level.get_sprites_by_tag("bar")
        self.bar_sprites.sort(key=lambda s: s.x)
        self._update_bar()

        self.life_sprites: List[Sprite] = self.current_level.get_sprites_by_tag("life")
        self.life_sprites.sort(key=lambda s: s.x)
        self._update_lives_display()

    def _restart_board(self) -> None:
        initial = self.current_level.get_data("initial_colors")
        self.color_grid = []
        for gy in range(self.grid_h):
            row: List[int] = []
            for gx in range(self.grid_w):
                row.append(initial[gy][gx])
            self.color_grid.append(row)

        self.player_gx = self.current_level.get_data("player_gx")
        self.player_gy = self.current_level.get_data("player_gy")
        ppx, ppy = _tile_pos(self.player_gx, self.player_gy, self.ox, self.oy)
        self.player_sprite.set_position(ppx, ppy)

        self.move_count = 0
        self._history = []
        self._update_bar()

        for gy in range(self.grid_h):
            for gx in range(self.grid_w):
                self._update_tile_visual(gx, gy)

        self._update_lives_display()

    def _in_bounds(self, gx: int, gy: int) -> bool:
        return 0 <= gx < self.grid_w and 0 <= gy < self.grid_h

    def _get_cross(self, gx: int, gy: int) -> List[Tuple[int, int]]:
        result: List[Tuple[int, int]] = []
        for x in range(self.grid_w):
            if x != gx:
                result.append((x, gy))
        for y in range(self.grid_h):
            if y != gy:
                result.append((gx, y))
        return result

    def _update_tile_visual(self, gx: int, gy: int) -> None:
        s = self.tile_sprites.get((gx, gy))
        if s is None:
            return
        cidx = self.color_grid[gy][gx]
        new_color = self.palette[cidx]
        s.color_remap(None, new_color)

    def _update_bar(self) -> None:
        if self.move_limit <= 0:
            return
        remaining = max(0, self.move_limit - self.move_count)
        filled = int(round(remaining / self.move_limit * BAR_WIDTH))
        for i, pip in enumerate(self.bar_sprites):
            if i < filled:
                pip.color_remap(None, BAR_FULL)
            else:
                pip.color_remap(None, BAR_EMPTY)

    def _update_lives_display(self) -> None:
        for i, pip in enumerate(self.life_sprites):
            if i < self.lives:
                pip.set_visible(True)
            else:
                pip.set_visible(False)

    def _check_win(self) -> bool:
        target: Optional[int] = None
        for gy in range(self.grid_h):
            for gx in range(self.grid_w):
                if (gx, gy) in self.danger_set:
                    continue
                c = self.color_grid[gy][gx]
                if target is None:
                    target = c
                elif c != target:
                    return False
        return True

    def _save_state(self) -> Dict:
        return {
            "color_grid": [row[:] for row in self.color_grid],
            "player_gx": self.player_gx,
            "player_gy": self.player_gy,
            "lives": self.lives,
        }

    def _restore_state(self, snap: Dict) -> None:
        self.color_grid = snap["color_grid"]
        self.player_gx = snap["player_gx"]
        self.player_gy = snap["player_gy"]
        self.lives = snap["lives"]

        ppx, ppy = _tile_pos(self.player_gx, self.player_gy, self.ox, self.oy)
        self.player_sprite.set_position(ppx, ppy)

        for gy in range(self.grid_h):
            for gx in range(self.grid_w):
                self._update_tile_visual(gx, gy)

        self._update_lives_display()

    def step(self) -> None:
        if self.action.id == GameAction.ACTION7:
            self.move_count += 1
            if self._history:
                self._restore_state(self._history.pop())
            self._update_bar()
            if self.move_count >= self.move_limit:
                self.lives -= 1
                self._update_lives_display()
                if self.lives <= 0:
                    self._lost = True
                    self.lose()
                else:
                    self._restart_board()
            self.complete_action()
            return

        dx, dy = 0, 0
        if self.action.id == GameAction.ACTION1:
            dy = -1
        elif self.action.id == GameAction.ACTION2:
            dy = 1
        elif self.action.id == GameAction.ACTION3:
            dx = -1
        elif self.action.id == GameAction.ACTION4:
            dx = 1
        else:
            self.complete_action()
            return

        new_gx = self.player_gx + dx
        new_gy = self.player_gy + dy

        if not self._in_bounds(new_gx, new_gy):
            self.complete_action()
            return

        if (new_gx, new_gy) in self.frozen_set:
            self.complete_action()
            return

        self._history.append(self._save_state())

        self.player_gx = new_gx
        self.player_gy = new_gy
        ppx, ppy = _tile_pos(new_gx, new_gy, self.ox, self.oy)
        self.player_sprite.set_position(ppx, ppy)

        self.move_count += 1
        shift = _seq(self.move_count) % self.num_colors

        if shift > 0:
            cross = self._get_cross(new_gx, new_gy)
            for cx, cy in cross:
                if (cx, cy) not in self.no_shift_set:
                    old_idx = self.color_grid[cy][cx]
                    new_idx = (old_idx + shift) % self.num_colors
                    self.color_grid[cy][cx] = new_idx
                    self._update_tile_visual(cx, cy)

        self._update_bar()

        if (new_gx, new_gy) in self.danger_set and self.lives > 0:
            self.lives -= 1
            self._update_lives_display()
            if self.lives <= 0:
                self._lost = True
                self.lose()
                self.complete_action()
                return
            else:
                self._restart_board()
                self.complete_action()
                return

        if self._check_win():
            self._level_just_won = True
            self._level_won_num = self._level_num
            if self._level_num >= len(self._levels):
                self._won = True
            self._level_num += 1
            self.next_level()
        elif self.move_count >= self.move_limit:
            self.lives -= 1
            self._update_lives_display()
            if self.lives <= 0:
                self._lost = True
                self.lose()
            else:
                self._restart_board()

        self.complete_action()


_ARC_PALETTE = [
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


def _to_png(rgb: np.ndarray) -> bytes:
    h, w, _ = rgb.shape
    raw = b""
    for y in range(h):
        raw += b"\x00" + rgb[y].tobytes()
    compressed = zlib.compress(raw)

    def _chunk(tag: bytes, data: bytes) -> bytes:
        c = tag + data
        crc = struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        return struct.pack(">I", len(data)) + c + crc

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", compressed)
        + _chunk(b"IEND", b"")
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


class PuzzleEnvironment:
    _ACTION_MAP: Dict[str, GameAction] = {
        "up": GameAction.ACTION1,
        "down": GameAction.ACTION2,
        "left": GameAction.ACTION3,
        "right": GameAction.ACTION4,
        "undo": GameAction.ACTION7,
        "reset": GameAction.RESET,
    }

    _VALID_ACTIONS: List[str] = ["up", "down", "left", "right", "undo"]

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self._engine: Optional[Pi01] = Pi01(seed=seed)
        self._total_turns = 0
        self._consecutive_resets = 0

    def _require_engine(self) -> "Pi01":
        if self._engine is None:
            raise RuntimeError("Environment is closed")
        return self._engine

    def reset(self) -> GameState:
        engine = self._require_engine()
        if engine._won or self._consecutive_resets >= 1:
            self._engine = Pi01(seed=self._seed)
            self._consecutive_resets = 0
        else:
            self._consecutive_resets += 1
        self._require_engine().perform_action(ActionInput(id=GameAction.RESET))
        self._total_turns = 0
        return self._build_state()

    def get_actions(self) -> List[str]:
        engine = self._require_engine()
        if engine._won or engine._lost:
            return ["reset"]
        return list(self._VALID_ACTIONS)

    def step(self, action: str) -> StepResult:
        if action not in self._ACTION_MAP:
            raise ValueError(f"Invalid action: {action}")

        game_action = self._ACTION_MAP[action]

        if game_action == GameAction.RESET:
            state = self.reset()
            return StepResult(state=state, reward=0.0, done=False)

        self._consecutive_resets = 0
        engine = self._require_engine()
        engine.perform_action(ActionInput(id=game_action))
        self._total_turns += 1

        won = engine._won
        lost = engine._lost
        done = won or lost

        if engine._level_just_won:
            reward = 1.0 / len(engine._levels)
            engine._level_just_won = False
        else:
            reward = 0.0

        state = self._build_state()
        return StepResult(state=state, reward=reward, done=done)

    def is_done(self) -> bool:
        engine = self._require_engine()
        return engine._won or engine._lost

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError(f"Unsupported render mode: {mode}")
        engine = self._require_engine()
        index_grid = engine.camera.render(engine.current_level.get_sprites())
        h, w = index_grid.shape[:2]
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for idx, color in enumerate(_ARC_PALETTE):
            mask = index_grid == idx
            rgb[mask] = color
        return rgb

    def close(self) -> None:
        self._engine = None

    def _build_state(self) -> GameState:
        e = self._engine
        if e is None:
            return GameState(
                text_observation="",
                image_observation=None,
                valid_actions=None,
                turn=self._total_turns,
            )

        text = self._build_text(e)
        rgb = self.render()
        image_bytes = _to_png(rgb)

        valid: Optional[List[str]] = None
        if not (e._won or e._lost):
            valid = list(self._VALID_ACTIONS)

        return GameState(
            text_observation=text,
            image_observation=image_bytes,
            valid_actions=valid,
            turn=self._total_turns,
            metadata={
                "total_levels": len(self._engine._levels),
                "level": min(e._level_num, len(e._levels)),
                "moves_used": e.move_count,
                "move_limit": e.move_limit,
                "lives": e.lives,
            },
        )

    def _build_text(self, e: "Pi01") -> str:
        parts: List[str] = []

        if e._won:
            parts.append("WIN")
        elif e._lost:
            parts.append("GAME OVER")

        level_display = min(e._level_num, len(e._levels))
        parts.append(
            f"Level:{level_display} Moves:{e.move_count}/{e.move_limit} Lives:{e.lives}"
        )

        for gy in range(e.grid_h):
            row_parts: List[str] = []
            for gx in range(e.grid_w):
                row_parts.append(str(e.color_grid[gy][gx]))
            parts.append(" ".join(row_parts))

        footer = f"Player:({e.player_gx},{e.player_gy})"
        if e.frozen_set:
            frozen_str = ",".join(f"({x},{y})" for x, y in sorted(e.frozen_set))
            footer += f" Frozen:{frozen_str}"
        if e.danger_set:
            danger_str = ",".join(f"({x},{y})" for x, y in sorted(e.danger_set))
            footer += f" Danger:{danger_str}"
        parts.append(footer)

        return "\n".join(parts)


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
        terminated: bool = result.done
        truncated: bool = False
        return (
            obs,
            result.reward,
            terminated,
            truncated,
            self._build_info(result.state, result.info),
        )

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
