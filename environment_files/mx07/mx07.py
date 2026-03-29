import io
import random
import struct
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from arcengine import (
    ActionInput,
    ARCBaseGame,
    Camera,
    GameAction,
    GameState as EngineState,
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
    info: dict = field(default_factory=dict)


DIR_RIGHT = (1, 0)
DIR_LEFT = (-1, 0)
DIR_DOWN = (0, 1)
DIR_UP = (0, -1)

MR_SLASH = 0
MR_BACKSLASH = 1
MR_HORIZ = 2
MR_VERT = 3


def _emitter_pixels(direction, color):

    c = color
    b = 0
    base = np.array(
        [
            [c, c, c],
            [c, c, c],
            [c, c, c],
        ],
        dtype=np.int32,
    )

    if direction == DIR_RIGHT:
        base[1, 2] = b
    elif direction == DIR_LEFT:
        base[1, 0] = b
    elif direction == DIR_DOWN:
        base[2, 1] = b
    elif direction == DIR_UP:
        base[0, 1] = b
    return base


def _receiver_pixels(req_color, active=False):
    inner = 11 if active else req_color
    return np.array(
        [
            [5, 5, 5],
            [5, inner, 5],
            [5, 5, 5],
        ],
        dtype=np.int32,
    )


_MIRROR_PIXELS = {
    MR_SLASH: np.array(
        [
            [-1, -1, 5],
            [-1, 5, -1],
            [5, -1, -1],
        ],
        dtype=np.int32,
    ),
    MR_BACKSLASH: np.array(
        [
            [5, -1, -1],
            [-1, 5, -1],
            [-1, -1, 5],
        ],
        dtype=np.int32,
    ),
    MR_HORIZ: np.array(
        [
            [-1, -1, -1],
            [5, 5, 5],
            [-1, -1, -1],
        ],
        dtype=np.int32,
    ),
    MR_VERT: np.array(
        [
            [-1, 5, -1],
            [-1, 5, -1],
            [-1, 5, -1],
        ],
        dtype=np.int32,
    ),
}


def _beam_pixel_horiz(color):
    return np.array(
        [
            [-1, -1, -1],
            [color, color, color],
            [-1, -1, -1],
        ],
        dtype=np.int32,
    )


def _beam_pixel_vert(color):
    return np.array(
        [
            [-1, color, -1],
            [-1, color, -1],
            [-1, color, -1],
        ],
        dtype=np.int32,
    )


_BG_TILE = np.array([[0]], dtype=np.int32)

sprites = {}


_EMIT_COLORS = [2, 3, 1, 6]

_BEAM_TRAIL_COLORS = {
    "red": 2,
    "grn": 10,
    "blu": 9,
    "mag": 6,
}
_EMIT_DIRS = [DIR_RIGHT, DIR_LEFT, DIR_DOWN, DIR_UP]
_DIR_NAMES = ["r", "l", "d", "u"]
_COLOR_NAMES = ["red", "grn", "blu", "mag"]

for _ci, _col in enumerate(_EMIT_COLORS):
    for _di, _dir in enumerate(_EMIT_DIRS):
        _key = f"emit_{_COLOR_NAMES[_ci]}_{_DIR_NAMES[_di]}"
        sprites[_key] = Sprite(
            pixels=_emitter_pixels(_dir, 14),
            name=_key,
            visible=True,
            collidable=True,
            tags=["emitter", f"color_{_COLOR_NAMES[_ci]}", f"dir_{_DIR_NAMES[_di]}"],
            layer=1,
        )


for _ci, _col in enumerate(_EMIT_COLORS):
    _key_off = f"recv_{_COLOR_NAMES[_ci]}_off"
    _key_on = f"recv_{_COLOR_NAMES[_ci]}_on"
    sprites[_key_off] = Sprite(
        pixels=_receiver_pixels(_col, active=False),
        name=_key_off,
        visible=True,
        collidable=True,
        tags=["receiver", f"color_{_COLOR_NAMES[_ci]}", "inactive"],
        layer=1,
    )
    sprites[_key_on] = Sprite(
        pixels=_receiver_pixels(_col, active=True),
        name=_key_on,
        visible=True,
        collidable=False,
        tags=["receiver", f"color_{_COLOR_NAMES[_ci]}", "active"],
        layer=1,
    )


sprites["recv_wrong"] = Sprite(
    pixels=np.array(
        [
            [5, 5, 5],
            [5, 12, 5],
            [5, 5, 5],
        ],
        dtype=np.int32,
    ),
    name="recv_wrong",
    visible=True,
    collidable=False,
    tags=["receiver", "wrong"],
    layer=1,
)


for _ms in range(4):
    _key = f"mirror_{_ms}"
    sprites[_key] = Sprite(
        pixels=_MIRROR_PIXELS[_ms],
        name=_key,
        visible=True,
        collidable=True,
        tags=["mirror"],
        layer=2,
    )

for _ci, _cname in enumerate(_COLOR_NAMES):
    _trail_col = _BEAM_TRAIL_COLORS[_cname]
    _key_h = f"beam_{_cname}_h"
    sprites[_key_h] = Sprite(
        pixels=_beam_pixel_horiz(_trail_col),
        name=_key_h,
        visible=True,
        collidable=False,
        tags=["beam", f"color_{_cname}", "horiz"],
        layer=0,
    )
    _key_v = f"beam_{_cname}_v"
    sprites[_key_v] = Sprite(
        pixels=_beam_pixel_vert(_trail_col),
        name=_key_v,
        visible=True,
        collidable=False,
        tags=["beam", f"color_{_cname}", "vert"],
        layer=0,
    )


sprites["cursor"] = Sprite(
    pixels=np.array(
        [
            [4, 4, 4],
            [4, -1, 4],
            [4, 4, 4],
        ],
        dtype=np.int32,
    ),
    name="cursor",
    visible=True,
    collidable=False,
    tags=["cursor"],
    layer=10,
)


sprites["solved_dot"] = Sprite(
    pixels=np.array([[3]], dtype=np.int32),
    name="solved_dot",
    visible=False,
    collidable=False,
    tags=["ui"],
    layer=5,
)


sprites["bg"] = Sprite(
    pixels=np.array([[0]], dtype=np.int32),
    name="bg",
    visible=True,
    collidable=False,
    tags=["bg"],
    layer=-10,
)


CELL = 3


LIVES_PER_LEVEL = 3
C_LIFE = 8
C_LIFE_LOST = 5


def _build_level(spec):
    level_sprites = []
    W, H = spec["grid"]

    for cx, cy, cname, dname in spec["emitters"]:
        key = f"emit_{cname}_{dname}"
        s = sprites[key].clone().set_position(cx * CELL, cy * CELL)
        level_sprites.append(s)

    for cx, cy, cname in spec["receivers"]:
        key = f"recv_{cname}_off"
        s = sprites[key].clone().set_position(cx * CELL, cy * CELL)
        level_sprites.append(s)

    for cx, cy, init_state in spec["mirrors"]:
        key = f"mirror_{init_state}"
        s = sprites[key].clone().set_position(cx * CELL, cy * CELL)
        level_sprites.append(s)

    cursor = sprites["cursor"].clone().set_position(0, 0)
    level_sprites.append(cursor)

    return Level(
        sprites=level_sprites,
        grid_size=(W * CELL, H * CELL),
        data={
            "cell_w": W,
            "cell_h": H,
            "emitters": spec["emitters"],
            "receivers": spec["receivers"],
            "mirrors": [(cx, cy, st) for (cx, cy, st) in spec["mirrors"]],
            "wrap": spec.get("wrap", False),
        },
    )


_DIR_ROT_CW = {"r": "d", "d": "l", "l": "u", "u": "r"}

_MR_ROT_CW = {
    MR_SLASH: MR_BACKSLASH,
    MR_BACKSLASH: MR_SLASH,
    MR_HORIZ: MR_VERT,
    MR_VERT: MR_HORIZ,
}


def _rotate_spec_90(spec):
    W, H = spec["grid"]
    new_grid = (H, W)
    new_emitters = [
        (H - 1 - cy, cx, color, _DIR_ROT_CW[d])
        for (cx, cy, color, d) in spec["emitters"]
    ]
    new_receivers = [(H - 1 - cy, cx, color) for (cx, cy, color) in spec["receivers"]]
    new_mirrors = [(H - 1 - cy, cx, _MR_ROT_CW[st]) for (cx, cy, st) in spec["mirrors"]]
    result = {
        "grid": new_grid,
        "emitters": new_emitters,
        "receivers": new_receivers,
        "mirrors": new_mirrors,
    }
    if "wrap" in spec:
        result["wrap"] = spec["wrap"]
    return result


def _make_variants(spec):
    r1 = _rotate_spec_90(spec)
    r2 = _rotate_spec_90(r1)
    r3 = _rotate_spec_90(r2)
    return [spec, r1, r2, r3]


_spec1 = {
    "grid": (10, 10),
    "emitters": [(1, 5, "red", "r")],
    "receivers": [(5, 2, "red")],
    "mirrors": [(5, 5, MR_BACKSLASH)],
}


_spec2 = {
    "grid": (21, 21),
    "emitters": [
        (1, 3, "blu", "r"),
        (9, 1, "blu", "d"),
    ],
    "receivers": [
        (5, 9, "blu"),
        (2, 5, "blu"),
    ],
    "mirrors": [
        (5, 3, MR_SLASH),
        (9, 5, MR_BACKSLASH),
    ],
}


_spec3 = {
    "grid": (14, 14),
    "emitters": [
        (1, 7, "red", "r"),
        (8, 1, "grn", "d"),
    ],
    "receivers": [
        (4, 11, "red"),
        (2, 6, "grn"),
    ],
    "mirrors": [
        (4, 7, MR_SLASH),
        (8, 6, MR_BACKSLASH),
        (11, 3, MR_HORIZ),
    ],
}

_spec4 = {
    "grid": (16, 16),
    "wrap": False,
    "emitters": [
        (1, 4, "red", "r"),
        (14, 1, "blu", "d"),
        (1, 13, "grn", "r"),
    ],
    "receivers": [
        (2, 5, "red"),
        (13, 2, "blu"),
        (3, 10, "grn"),
    ],
    "mirrors": [
        (5, 4, MR_SLASH),
        (5, 9, MR_BACKSLASH),
        (2, 9, MR_SLASH),
        (14, 6, MR_BACKSLASH),
        (8, 6, MR_SLASH),
        (8, 2, MR_BACKSLASH),
        (7, 13, MR_SLASH),
        (7, 15, MR_BACKSLASH),
        (3, 15, MR_SLASH),
    ],
}

_spec5 = {
    "grid": (18, 18),
    "wrap": True,
    "emitters": [
        (1, 3, "red", "r"),
        (14, 1, "grn", "d"),
        (1, 14, "blu", "r"),
        (9, 1, "mag", "d"),
    ],
    "receivers": [
        (2, 4, "red"),
        (13, 3, "grn"),
        (12, 11, "blu"),
        (5, 2, "mag"),
    ],
    "mirrors": [
        (6, 3, MR_SLASH),
        (6, 8, MR_BACKSLASH),
        (2, 8, MR_SLASH),
        (14, 7, MR_BACKSLASH),
        (8, 7, MR_SLASH),
        (8, 3, MR_BACKSLASH),
        (0, 14, MR_SLASH),
        (0, 16, MR_BACKSLASH),
        (12, 16, MR_SLASH),
        (9, 10, MR_SLASH),
        (5, 10, MR_BACKSLASH),
        (11, 5, MR_HORIZ),
        (3, 12, MR_BACKSLASH),
    ],
}

_spec6 = {
    "grid": (20, 20),
    "wrap": True,
    "emitters": [
        (1, 2, "red", "r"),
        (17, 1, "grn", "d"),
        (1, 15, "blu", "r"),
        (12, 1, "mag", "d"),
        (18, 17, "red", "u"),
    ],
    "receivers": [
        (3, 3, "red"),
        (18, 3, "grn"),
        (13, 11, "blu"),
        (2, 6, "mag"),
        (11, 4, "red"),
    ],
    "mirrors": [
        (7, 2, MR_HORIZ),
        (7, 9, MR_BACKSLASH),
        (3, 9, MR_HORIZ),
        (17, 8, MR_BACKSLASH),
        (10, 8, MR_HORIZ),
        (10, 3, MR_BACKSLASH),
        (0, 15, MR_HORIZ),
        (0, 18, MR_BACKSLASH),
        (13, 18, MR_HORIZ),
        (12, 13, MR_HORIZ),
        (5, 13, MR_BACKSLASH),
        (5, 6, MR_HORIZ),
        (18, 12, MR_HORIZ),
        (11, 12, MR_BACKSLASH),
        (14, 6, MR_SLASH),
        (8, 16, MR_BACKSLASH),
    ],
}

LEVEL_VARIANTS = [
    _make_variants(_spec1),
    _make_variants(_spec2),
    _make_variants(_spec3),
    _make_variants(_spec4),
    _make_variants(_spec5),
    _make_variants(_spec6),
]

levels = [_build_level(LEVEL_VARIANTS[i][0]) for i in range(6)]


class MoveCounter(RenderableUserDisplay):
    def __init__(self, max_moves: int):
        self.moves = 0
        self.max_moves = max_moves
        self.solved = False
        self.lives = LIVES_PER_LEVEL

    def reset(self, max_moves: int):
        self.moves = 0
        self.max_moves = max_moves
        self.solved = False

    def reset_lives(self):
        self.lives = LIVES_PER_LEVEL

    def increment(self):
        self.moves += 1

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if self.max_moves == 0:
            return frame
        bar_w = min(w - 2, 40)
        used = min(bar_w, int(bar_w * self.moves / max(self.max_moves, 1)))
        bar_y = h - 2
        bar_x = 1
        for i in range(bar_w):
            col = 2 if i < used else 5
            if bar_y < h and bar_x + i < w:
                frame[bar_y, bar_x + i] = col

        for i in range(LIVES_PER_LEVEL):
            lx = 53 + i * 4
            ly = h - 2
            color = C_LIFE if self.lives > i else C_LIFE_LOST
            for y in range(ly, ly + 2):
                for x in range(lx, lx + 2):
                    if 0 <= x < w and 0 <= y < h:
                        frame[y, x] = color

        if self.solved:
            for i in range(w):
                frame[0, i] = 3

        return frame


_CNAME_TO_IDX = {"red": 2, "grn": 3, "blu": 1, "mag": 6}

_DEFLECT = {
    MR_SLASH: {
        DIR_RIGHT: DIR_UP,
        DIR_LEFT: DIR_DOWN,
        DIR_UP: DIR_RIGHT,
        DIR_DOWN: DIR_LEFT,
    },
    MR_BACKSLASH: {
        DIR_RIGHT: DIR_DOWN,
        DIR_LEFT: DIR_UP,
        DIR_UP: DIR_LEFT,
        DIR_DOWN: DIR_RIGHT,
    },
    MR_HORIZ: {
        DIR_RIGHT: DIR_RIGHT,
        DIR_LEFT: DIR_LEFT,
        DIR_UP: None,
        DIR_DOWN: None,
    },
    MR_VERT: {
        DIR_RIGHT: None,
        DIR_LEFT: None,
        DIR_UP: DIR_UP,
        DIR_DOWN: DIR_DOWN,
    },
}

_DIR_TO_NAME = {DIR_RIGHT: "r", DIR_LEFT: "l", DIR_DOWN: "d", DIR_UP: "u"}
_DNAME_TO_DIR = {"r": DIR_RIGHT, "l": DIR_LEFT, "d": DIR_DOWN, "u": DIR_UP}


class Mx07(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._move_counter = MoveCounter(max_moves=100)
        self._rng = random.Random(seed)
        self._engine_snapshot = None
        self._engine_can_undo = False
        super().__init__(
            "mx07",
            levels,
            Camera(0, 0, 64, 64, 0, 4, [self._move_counter]),
            available_actions=[0, 1, 2, 3, 4, 5, 6, 7],
        )

    def set_level(self, index: int) -> None:
        variant = self._rng.choice(LEVEL_VARIANTS[index])
        new_lv = _build_level(variant)
        self._levels[index] = new_lv.clone()
        self._clean_levels[index] = new_lv.clone()
        super().set_level(index)

    def on_set_level(self, level: Level) -> None:
        cw = self.current_level.get_data("cell_w")
        ch = self.current_level.get_data("cell_h")
        self._cell_w = cw
        self._cell_h = ch

        self._emitter_sprites = self.current_level.get_sprites_by_tag("emitter")
        self._receiver_sprites = self.current_level.get_sprites_by_tag("receiver")
        self._mirror_sprites = self.current_level.get_sprites_by_tag("mirror")

        cursor_list = self.current_level.get_sprites_by_tag("cursor")
        self._cursor_sprite = cursor_list[0] if cursor_list else None
        self._cursor_col = 0
        self._cursor_row = 0
        if self._cursor_sprite is not None:
            self._cursor_sprite.set_position(0, 0)

        self._mirror_state: dict = {}
        mirror_data = self.current_level.get_data("mirrors")
        for cx, cy, init_st in mirror_data:
            px, py = cx * CELL, cy * CELL
            for sp in self._mirror_sprites:
                if sp.x == px and sp.y == py:
                    self._mirror_state[id(sp)] = init_st
                    break

        n_mirrors = len(self._mirror_sprites)
        self._move_counter.reset(max_moves=n_mirrors * 8)

        self._move_counter.lives = LIVES_PER_LEVEL

        self._engine_can_undo = False
        self._engine_snapshot = None

        self._beam_trails: list = []

        self._trace_all_beams()

    def handle_reset(self) -> None:
        self._move_counter.reset_lives()
        super().handle_reset()

    def _cell_of(self, sprite) -> tuple:
        return (sprite.x // CELL, sprite.y // CELL)

    def _mirror_at_cell(self, col: int, row: int):
        px, py = col * CELL, row * CELL
        for sp in self._mirror_sprites:
            if sp.x == px and sp.y == py:
                return (sp, self._mirror_state[id(sp)])
        return None

    def _emitter_at_cell(self, col: int, row: int):
        px, py = col * CELL, row * CELL
        for sp in self._emitter_sprites:
            if sp.x == px and sp.y == py:
                return sp
        return None

    def _receiver_at_cell(self, col: int, row: int):
        px, py = col * CELL, row * CELL
        for sp in self._receiver_sprites:
            if sp.x == px and sp.y == py:
                return sp
        return None

    def _clear_beam_trails(self):
        for sp in self._beam_trails:
            self.current_level.remove_sprite(sp)
        self._beam_trails.clear()

    def _reset_receivers(self):
        recv_data = self.current_level.get_data("receivers")
        for sp in list(self._receiver_sprites):
            px, py = sp.x, sp.y
            col_name = None
            for cx, cy, cname in recv_data:
                if cx * CELL == px and cy * CELL == py:
                    col_name = cname
                    break
            if col_name is None:
                continue
            sp.pixels = sprites[f"recv_{col_name}_off"].pixels.copy()

    def _tag_color_name(self, sp) -> str:

        if sp.tags:
            for t in sp.tags:
                if t.startswith("color_"):
                    return t[6:]
        return ""

    def _add_beam_trail(self, col: int, row: int, dx: int, color_name: str):
        _axis = "h" if dx != 0 else "v"
        trail_key = f"beam_{color_name}_{_axis}"
        trail = sprites[trail_key].clone().set_position(col * CELL, row * CELL)
        self.current_level.add_sprite(trail)
        self._beam_trails.append(trail)

    def _handle_receiver_hit(self, recv_sp, color_name: str):
        recv_color = self._tag_color_name(recv_sp)
        if recv_color == color_name:
            recv_sp.pixels = sprites[f"recv_{color_name}_on"].pixels.copy()
        else:
            recv_sp.pixels = sprites["recv_wrong"].pixels.copy()

    def _handle_mirror_deflect(
        self, mirror_info, col, row, dx, dy, color_name, visited, depth
    ):
        sp, state = mirror_info
        new_dir = _DEFLECT[state].get((dx, dy))
        if new_dir is None:
            return "stop", dx, dy
        if new_dir == (dx, dy):
            return "pass", dx, dy
        self._add_beam_trail(col, row, dx, color_name)
        new_dx, new_dy = new_dir
        self._trace_beam(col, row, new_dx, new_dy, color_name, visited, depth + 1)
        return "recurse", dx, dy

    def _trace_beam(self, start_col, start_row, dx, dy, color_name, visited, depth=0):
        if depth > 200:
            return

        col, row = start_col + dx, start_row + dy
        cw, ch = self._cell_w, self._cell_h
        wrap = self.current_level.get_data("wrap") or False

        while True:
            if not (0 <= col < cw and 0 <= row < ch):
                if wrap:
                    col, row = col % cw, row % ch
                else:
                    break

            cell_key = (col, row, dx, dy)
            if cell_key in visited:
                break
            visited.add(cell_key)

            mirror_info = self._mirror_at_cell(col, row)
            if mirror_info is not None:
                action, dx, dy = self._handle_mirror_deflect(
                    mirror_info, col, row, dx, dy, color_name, visited, depth
                )
                if action == "pass":
                    col, row = col + dx, row + dy
                    continue
                break

            recv_sp = self._receiver_at_cell(col, row)
            if recv_sp is not None:
                self._handle_receiver_hit(recv_sp, color_name)
                break

            if self._emitter_at_cell(col, row) is not None:
                break

            self._add_beam_trail(col, row, dx, color_name)
            col, row = col + dx, row + dy

    def _trace_all_beams(self):

        self._clear_beam_trails()
        self._reset_receivers()

        visited: set = set()
        emitter_data = self.current_level.get_data("emitters")

        for cx, cy, cname, dname in emitter_data:
            dx, dy = _DNAME_TO_DIR[dname]
            self._trace_beam(cx, cy, dx, dy, cname, visited)

    def _all_receivers_active(self) -> bool:
        recv_data = self.current_level.get_data("receivers")
        for cx, cy, cname in recv_data:
            px, py = cx * CELL, cy * CELL
            recv_sp = None
            for sp in self._receiver_sprites:
                if sp.x == px and sp.y == py:
                    recv_sp = sp
                    break
            if recv_sp is None:
                return False

            try:
                if recv_sp.pixels[1, 1] != 11:
                    return False
            except Exception:
                return False
        return True

    def _reset_mirrors_to_initial(self) -> None:
        mirror_data = self.current_level.get_data("mirrors")
        for cx, cy, init_st in mirror_data:
            px, py = cx * CELL, cy * CELL
            for sp in self._mirror_sprites:
                if sp.x == px and sp.y == py:
                    self._mirror_state[id(sp)] = init_st
                    sp.pixels = _MIRROR_PIXELS[init_st].copy()
                    break
        self._trace_all_beams()

    def _trigger_life_loss(self) -> None:
        self._move_counter.lives -= 1
        if self._move_counter.lives <= 0:
            self.lose()
            return

        n_mirrors = len(self._mirror_sprites)
        self._move_counter.reset(max_moves=n_mirrors * 8)
        self._reset_mirrors_to_initial()

    def _rotate_mirror_at(self, col: int, row: int) -> None:

        mirror_info = self._mirror_at_cell(col, row)
        if mirror_info is not None:
            sp, state = mirror_info
            new_state = (state + 1) % 4
            self._mirror_state[id(sp)] = new_state
            sp.pixels = _MIRROR_PIXELS[new_state].copy()
            self._move_counter.increment()
            self._trace_all_beams()

    def _move_cursor(self, dcol: int, drow: int) -> None:

        cw = self._cell_w
        ch = self._cell_h
        wrap = self.current_level.get_data("wrap") or False
        if wrap:
            new_col = (self._cursor_col + dcol) % cw
            new_row = (self._cursor_row + drow) % ch
        else:
            new_col = max(0, min(cw - 1, self._cursor_col + dcol))
            new_row = max(0, min(ch - 1, self._cursor_row + drow))
        self._cursor_col = new_col
        self._cursor_row = new_row
        if self._cursor_sprite is not None:
            self._cursor_sprite.set_position(new_col * CELL, new_row * CELL)

    def _engine_save_snapshot(self) -> None:
        self._engine_snapshot = {
            "mirror_state": {sid: st for sid, st in self._mirror_state.items()},
            "mirror_pixels": {id(sp): sp.pixels.copy() for sp in self._mirror_sprites},
            "cursor_col": self._cursor_col,
            "cursor_row": self._cursor_row,
        }

    def _engine_restore_snapshot(self) -> None:
        snap = self._engine_snapshot
        for sid, st in snap["mirror_state"].items():
            self._mirror_state[sid] = st
        for sp in self._mirror_sprites:
            if id(sp) in snap["mirror_pixels"]:
                sp.pixels = snap["mirror_pixels"][id(sp)].copy()
        self._cursor_col = snap["cursor_col"]
        self._cursor_row = snap["cursor_row"]
        if self._cursor_sprite is not None:
            self._cursor_sprite.set_position(
                self._cursor_col * CELL, self._cursor_row * CELL
            )
        self._trace_all_beams()

    def step(self) -> None:
        aid = self.action.id

        if aid == GameAction.ACTION7:
            if self._engine_can_undo and self._engine_snapshot is not None:
                self._engine_restore_snapshot()
                self._move_counter.increment()
                self._engine_can_undo = False
                self._engine_snapshot = None
            self.complete_action()
            return

        self._engine_save_snapshot()

        if aid == GameAction.ACTION1:
            self._move_cursor(0, -1)
        elif aid == GameAction.ACTION2:
            self._move_cursor(0, 1)
        elif aid == GameAction.ACTION3:
            self._move_cursor(-1, 0)
        elif aid == GameAction.ACTION4:
            self._move_cursor(1, 0)

        elif aid == GameAction.ACTION5:
            self._rotate_mirror_at(self._cursor_col, self._cursor_row)

        elif aid == GameAction.ACTION6:
            raw_x = self.action.data.get("x", 0)
            raw_y = self.action.data.get("y", 0)

            coords = self.camera.display_to_grid(raw_x, raw_y)
            if coords:
                gx, gy = coords
                col = gx // CELL
                row = gy // CELL
                self._cursor_col = col
                self._cursor_row = row
                if self._cursor_sprite is not None:
                    self._cursor_sprite.set_position(col * CELL, row * CELL)
                self._rotate_mirror_at(col, row)

        if self._all_receivers_active():
            self._move_counter.solved = True
            self._engine_can_undo = False
            self._engine_snapshot = None
            self.next_level()
            self.complete_action()
            return

        if self._move_counter.moves >= self._move_counter.max_moves:
            self._engine_can_undo = False
            self._engine_snapshot = None
            self._trigger_life_loss()
            self.complete_action()
            return

        self._engine_can_undo = True
        self.complete_action()

    def _place_emitters_on_grid(self, grid, cw, ch):
        _dir_char = {"r": ">", "l": "<", "d": "v", "u": "^"}
        emitter_data = self.current_level.get_data("emitters")
        legend = []
        for cx, cy, cname, dname in emitter_data:
            if 0 <= cx < cw and 0 <= cy < ch:
                grid[cy][cx] = _dir_char.get(dname, "E")
                legend.append(f"{_dir_char.get(dname, '?')}={cname}")
        return legend

    def _place_receivers_on_grid(self, grid, cw, ch):
        _color_upper = {"red": "R", "grn": "G", "blu": "B", "mag": "M"}
        _color_lower = {"red": "r", "grn": "g", "blu": "b", "mag": "m"}
        recv_data = self.current_level.get_data("receivers")
        for sp in self._receiver_sprites:
            c, r = sp.x // CELL, sp.y // CELL
            if 0 <= c < cw and 0 <= r < ch:
                recv_color = ""
                for rcx, rcy, rcname in recv_data:
                    if rcx == c and rcy == r:
                        recv_color = rcname
                        break
                try:
                    active = sp.pixels[1, 1] == 11
                except Exception:
                    active = False
                if active:
                    grid[r][c] = _color_upper.get(recv_color, "R")
                else:
                    grid[r][c] = _color_lower.get(recv_color, "r")

    def _grid_to_text(self) -> str:
        cw = self._cell_w
        ch = self._cell_h
        grid = [["." for _ in range(cw)] for _ in range(ch)]

        emitter_legend = self._place_emitters_on_grid(grid, cw, ch)
        self._place_receivers_on_grid(grid, cw, ch)

        _state_char = {0: "/", 1: "\\", 2: "-", 3: "|"}
        for sp in self._mirror_sprites:
            c, r = sp.x // CELL, sp.y // CELL
            if 0 <= c < cw and 0 <= r < ch:
                st = self._mirror_state.get(id(sp), 0)
                grid[r][c] = _state_char.get(st, "M")

        if 0 <= self._cursor_col < cw and 0 <= self._cursor_row < ch:
            if grid[self._cursor_row][self._cursor_col] == ".":
                grid[self._cursor_row][self._cursor_col] = "*"

        rows = [" ".join(grid[r]) for r in range(ch)]
        text = "\n".join(rows)
        if emitter_legend:
            text += "\nemitters: " + ", ".join(emitter_legend)
        return text


class PuzzleEnvironment:
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

    _ACTION_MAP: Dict[str, GameAction] = {
        "reset": GameAction.RESET,
        "up": GameAction.ACTION1,
        "down": GameAction.ACTION2,
        "left": GameAction.ACTION3,
        "right": GameAction.ACTION4,
        "select": GameAction.ACTION5,
        "click": GameAction.ACTION6,
        "undo": GameAction.ACTION7,
    }

    _VALID_ACTIONS: List[str] = [
        "reset",
        "up",
        "down",
        "left",
        "right",
        "select",
        "click",
        "undo",
    ]

    def __init__(self, seed: int = 0) -> None:
        self._engine = Mx07(seed=seed)
        self._done = False
        self._last_action_was_reset = False
        self._game_won = False
        self._total_turns = 0

    def _render_frame(self) -> np.ndarray:
        e = self._engine
        return e.camera.render(e.current_level.get_sprites())

    @staticmethod
    def _frame_to_png(frame: np.ndarray) -> bytes:
        h, w = frame.shape[:2]
        buf = io.BytesIO()
        buf.write(b"\x89PNG\r\n\x1a\n")

        def _chunk(ctype: bytes, data: bytes) -> None:
            buf.write(struct.pack(">I", len(data)))
            buf.write(ctype)
            buf.write(data)
            buf.write(struct.pack(">I", zlib.crc32(ctype + data) & 0xFFFFFFFF))

        _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
        raw = b""
        for y in range(h):
            raw += b"\x00" + frame[y].astype(np.uint8).tobytes()
        _chunk(b"IDAT", zlib.compress(raw))
        _chunk(b"IEND", b"")
        return buf.getvalue()

    def _build_state(self) -> GameState:
        e = self._engine
        frame = self._render_frame()
        png_bytes = self._frame_to_png(frame)

        return GameState(
            text_observation=e._grid_to_text(),
            image_observation=png_bytes,
            valid_actions=self.get_actions() if not self._done else None,
            turn=self._total_turns,
            metadata={
                "total_levels": len(self._engine._levels),
                "level": e._current_level_index,
                "lives": e._move_counter.lives,
                "moves": e._move_counter.moves,
                "max_moves": e._move_counter.max_moves,
                "grid_w": e._cell_w,
                "grid_h": e._cell_h,
                "cursor": [e._cursor_col, e._cursor_row],
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
        self._done = False
        self._total_turns = 0
        self._last_action_was_reset = True
        self._game_won = False
        return self._build_state()

    def get_actions(self) -> List[str]:
        if self._done:
            return ["reset"]
        return list(self._VALID_ACTIONS)

    def _build_action_input(self, game_action, parts) -> "ActionInput":
        e = self._engine
        if game_action == GameAction.ACTION6:
            if len(parts) >= 3:
                cx, cy = int(parts[1]), int(parts[2])
                return ActionInput(id=game_action, data={"x": cx, "y": cy})
            gx = e._cursor_col * CELL
            gy = e._cursor_row * CELL
            return ActionInput(id=game_action, data={"x": gx, "y": gy})
        return ActionInput(id=game_action)

    def _compute_step_result(self, result, action, total_levels, level_before):
        e = self._engine
        reward = 0.0
        done = False
        info: Dict[str, Any] = {"action": action}

        if result.state == EngineState.WIN:
            done = True
            self._done = True
            self._game_won = True
            reward = 1.0 / total_levels
            info["reason"] = "game_complete"
        elif result.state == EngineState.GAME_OVER:
            done = True
            self._done = True
            info["reason"] = "game_over"
        elif e._current_level_index > level_before:
            reward = 1.0 / total_levels

        return StepResult(
            state=self._build_state(), reward=reward, done=done, info=info
        )

    def step(self, action: str) -> StepResult:
        parts = action.split()
        action_key = parts[0] if parts else action

        if action_key not in self._ACTION_MAP:
            raise ValueError(
                f"Invalid action '{action_key}'. Must be one of {list(self._ACTION_MAP.keys())}"
            )

        if action_key == "reset":
            state = self.reset()
            return StepResult(
                state=state, reward=0.0, done=False, info={"action": "reset"}
            )

        self._last_action_was_reset = False
        e = self._engine
        game_action = self._ACTION_MAP[action_key]
        action_input = self._build_action_input(game_action, parts)

        total_levels = len(e._levels)
        level_before = e._current_level_index
        result = e.perform_action(action_input)
        self._total_turns += 1

        return self._compute_step_result(result, action, total_levels, level_before)

    def is_done(self) -> bool:
        return self._done

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError(f"Unsupported render mode: {mode}")
        frame = self._render_frame()
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
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
        "click",
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
