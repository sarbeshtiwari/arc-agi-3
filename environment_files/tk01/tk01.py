from __future__ import annotations

import base64
import io
import struct
import zlib
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

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


def _encode_png(frame: np.ndarray) -> bytes:
    h, w = frame.shape[:2]
    if frame.ndim == 2:
        raw = np.zeros((h, w, 3), dtype=np.uint8)
        raw[:] = frame[:, :, None]
    else:
        raw = frame.astype(np.uint8)
    rows = []
    for y in range(h):
        rows.append(b"\x00" + raw[y].tobytes())
    raw_data = b"".join(rows)
    compressed = zlib.compress(raw_data)

    def _chunk(ctype: bytes, data: bytes) -> bytes:
        c = ctype + data
        return (
            struct.pack(">I", len(data))
            + c
            + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        )

    buf = io.BytesIO()
    buf.write(b"\x89PNG\r\n\x1a\n")
    buf.write(_chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)))
    buf.write(_chunk(b"IDAT", compressed))
    buf.write(_chunk(b"IEND", b""))
    return buf.getvalue()


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


COL_BG = 5
COL_GRID = 4
COL_EMPTY = 2
COL_A = 9
COL_B = 12
COL_FIXED = 0
COL_XMARK = 8
COL_ERR = 8
COL_HEART = 8
COL_HEART_LOST = 4
COL_SELECT = 14

MAX_LIVES = 3

E = -1
A = 0
B = 1


def _decode_target(encoded: str) -> list[list[int]]:
    data = base64.b64decode(encoded)
    n = data[0]
    grid: list[list[int]] = []
    idx = 1
    for _ in range(n):
        if n <= 8:
            val = data[idx]
            idx += 1
        else:
            val = data[idx] | (data[idx + 1] << 8)
            idx += 2
        grid.append([B if (val >> c) & 1 else A for c in range(n)])
    return grid


PUZZLES = [
    {
        "grid_n": 4,
        "cell_size": 10,
        "gap": 2,
        "rules": ["no_three"],
        "target": _decode_target("BAUKAww="),
        "initial": [
            [B, E, E, A],
            [A, E, A, E],
            [E, E, A, A],
            [A, E, E, B],
        ],
        "x_constraints": [],
    },
    {
        "grid_n": 4,
        "cell_size": 10,
        "gap": 2,
        "rules": ["no_three", "balanced_rows"],
        "target": _decode_target("BAYJAww="),
        "initial": [
            [A, E, E, A],
            [E, E, A, E],
            [B, B, E, E],
            [E, E, E, B],
        ],
        "x_constraints": [],
    },
    {
        "grid_n": 6,
        "cell_size": 7,
        "gap": 2,
        "rules": ["no_three", "balanced_rows", "balanced_cols"],
        "target": _decode_target("BiUaDTIVKg=="),
        "initial": [
            [E, A, E, E, A, E],
            [E, E, E, B, E, E],
            [B, E, E, E, E, A],
            [E, E, A, E, E, E],
            [E, A, E, E, B, E],
            [A, E, E, B, E, E],
        ],
        "x_constraints": [],
    },
    {
        "grid_n": 8,
        "cell_size": 6,
        "gap": 1,
        "rules": ["no_three", "balanced_rows", "balanced_cols", "x_constraints"],
        "target": _decode_target("CGqVplmsU5pl"),
        "initial": [
            [E, B, E, E, E, B, E, E],
            [E, E, B, E, E, E, A, E],
            [A, E, E, E, A, E, E, B],
            [E, E, E, B, E, E, B, E],
            [E, A, E, E, E, B, E, E],
            [B, E, E, E, B, E, E, A],
            [E, E, A, E, E, E, E, E],
            [E, A, E, E, A, E, E, E],
        ],
        "x_constraints": [
            ((0, 0), (0, 1)),
            ((3, 2), (3, 3)),
            ((5, 0), (6, 0)),
        ],
    },
    {
        "grid_n": 10,
        "cell_size": 5,
        "gap": 1,
        "rules": ["no_three", "balanced_rows", "balanced_cols", "x_constraints"],
        "target": _decode_target("CpoCZQEsA9MApgFZArICTQFqApUB"),
        "initial": [
            [E, B, E, E, B, E, E, B, E, E],
            [E, E, B, E, E, E, B, E, E, A],
            [A, E, E, B, E, E, E, E, B, E],
            [E, E, E, E, B, E, E, B, E, E],
            [E, B, E, E, E, B, E, E, B, E],
            [B, E, E, B, E, E, B, E, E, E],
            [E, E, A, E, E, B, E, E, E, B],
            [E, A, E, E, E, E, E, A, E, E],
            [A, E, E, B, E, E, B, E, E, E],
            [E, E, B, E, E, A, E, E, B, E],
        ],
        "x_constraints": [
            ((0, 0), (0, 1)),
            ((4, 0), (5, 0)),
            ((7, 1), (7, 2)),
            ((8, 0), (8, 1)),
        ],
    },
]


def _solid(size, color):
    return [[color] * size for _ in range(size)]


def _frame(size, color, border=1):
    t = -1
    pixels = []
    for r in range(size):
        row = []
        for c in range(size):
            if r < border or r >= size - border or c < border or c >= size - border:
                row.append(color)
            else:
                row.append(t)
        pixels.append(row)
    return pixels


sprites = {
    "cell10": Sprite(
        pixels=_solid(10, COL_EMPTY),
        name="cell10",
        visible=True,
        collidable=False,
        layer=1,
        tags=["cell"],
    ),
    "cell7": Sprite(
        pixels=_solid(7, COL_EMPTY),
        name="cell7",
        visible=True,
        collidable=False,
        layer=1,
        tags=["cell"],
    ),
    "cell6": Sprite(
        pixels=_solid(6, COL_EMPTY),
        name="cell6",
        visible=True,
        collidable=False,
        layer=1,
        tags=["cell"],
    ),
    "cell5": Sprite(
        pixels=_solid(5, COL_EMPTY),
        name="cell5",
        visible=True,
        collidable=False,
        layer=1,
        tags=["cell"],
    ),
    "dot2": Sprite(
        pixels=_solid(2, COL_FIXED),
        name="dot2",
        visible=True,
        collidable=False,
        layer=3,
        tags=["fixed_dot"],
    ),
    "xmark": Sprite(
        pixels=_solid(2, COL_XMARK),
        name="xmark",
        visible=True,
        collidable=False,
        layer=2,
        tags=["xmark"],
    ),
    "err": Sprite(
        pixels=_solid(2, COL_ERR),
        name="err",
        visible=False,
        collidable=False,
        layer=4,
        tags=["err"],
    ),
    "sel10": Sprite(
        pixels=_frame(10, COL_SELECT, border=1),
        name="sel10",
        visible=False,
        collidable=False,
        layer=5,
        tags=["sel"],
    ),
    "sel7": Sprite(
        pixels=_frame(7, COL_SELECT, border=1),
        name="sel7",
        visible=False,
        collidable=False,
        layer=5,
        tags=["sel"],
    ),
    "sel6": Sprite(
        pixels=_frame(6, COL_SELECT, border=1),
        name="sel6",
        visible=False,
        collidable=False,
        layer=5,
        tags=["sel"],
    ),
    "sel5": Sprite(
        pixels=_frame(5, COL_SELECT, border=1),
        name="sel5",
        visible=False,
        collidable=False,
        layer=5,
        tags=["sel"],
    ),
}

_SEL_KEY = {10: "sel10", 7: "sel7", 6: "sel6", 5: "sel5"}

_CELL_KEY = {10: "cell10", 7: "cell7", 6: "cell6", 5: "cell5"}


def _color_for(val):
    if val == A:
        return COL_A
    if val == B:
        return COL_B
    return COL_EMPTY


def _build_level(puz, idx):
    n = puz["grid_n"]
    cs = puz["cell_size"]
    gap = puz["gap"]
    grid_total = n * cs + (n - 1) * gap
    ox = (64 - grid_total) // 2
    oy = ox + 2

    cell_key = _CELL_KEY[cs]
    spr_list = []

    bg_w = grid_total + 2
    bg_h = bg_w
    bg_sprite = Sprite(
        pixels=[[COL_GRID] * bg_w for _ in range(bg_h)],
        name="bg",
        visible=True,
        collidable=False,
        layer=0,
    )
    spr_list.append(bg_sprite.clone().set_position(ox - 1, oy - 1))

    for r in range(n):
        for c in range(n):
            x = ox + c * (cs + gap)
            y = oy + r * (cs + gap)
            val = puz["initial"][r][c]
            col = _color_for(val)
            spr_list.append(
                sprites[cell_key].clone().set_position(x, y).color_remap(None, col)
            )
            if val != E:
                dx = cs // 2 - 1
                dy = cs // 2 - 1
                spr_list.append(sprites["dot2"].clone().set_position(x + dx, y + dy))
            spr_list.append(sprites["err"].clone().set_position(x + cs - 2, y + cs - 2))

    sel_key = _SEL_KEY[cs]
    spr_list.append(sprites[sel_key].clone().set_position(0, 0))

    for (r1, c1), (r2, c2) in puz["x_constraints"]:
        if r1 == r2:
            x = ox + c1 * (cs + gap) + cs
            y = oy + r1 * (cs + gap) + cs // 2 - 1
        else:
            x = ox + c1 * (cs + gap) + cs // 2 - 1
            y = oy + r1 * (cs + gap) + cs
        spr_list.append(sprites["xmark"].clone().set_position(x, y))

    data = dict(puz)
    data["ox"] = ox
    data["oy"] = oy
    data["level_idx"] = idx

    return Level(
        sprites=spr_list,
        grid_size=(64, 64),
        data=data,
        name=f"Level {idx + 1}",
    )


class LivesHud(RenderableUserDisplay):
    def __init__(self, game: "Tk01") -> None:
        self._game = game

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        lives = self._game._lives
        for i in range(MAX_LIVES):
            x = 2 + i * 4
            y = 1
            color = COL_HEART if i < lives else COL_HEART_LOST
            frame[y : y + 2, x : x + 2] = color
        return frame


class Tk01(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._lives = MAX_LIVES
        self._game_over = False
        self._undo_stack: list = []
        self._hud = LivesHud(self)
        game_levels = [_build_level(p, i) for i, p in enumerate(PUZZLES)]
        camera = Camera(
            x=0,
            y=0,
            width=64,
            height=64,
            background=COL_BG,
            letter_box=COL_BG,
            interfaces=[self._hud],
        )

        super().__init__(
            "tk01",
            game_levels,
            camera,
            available_actions=[0, 1, 2, 3, 4, 6, 7],
            win_score=1,
        )

    def reset(self) -> None:
        super().reset()

    def get_actions(self):
        return [1, 2, 3, 4, 6, 7]

    def _save_undo_snapshot(self) -> None:
        self._undo_stack.append(([row[:] for row in self._grid], self._selected_rc))

    def _restore_undo_snapshot(self) -> None:
        grid, selected_rc = self._undo_stack.pop()
        self._grid = [row[:] for row in grid]
        self._selected_rc = selected_rc
        for r in range(self._n):
            for c in range(self._n):
                col = _color_for(self._grid[r][c])
                spr = self._cell_spr[r][c]
                if spr:
                    spr.color_remap(None, col)
        if self._selected_rc:
            sr, sc = self._selected_rc
            self._select_cell(sr, sc)
        self._update_errors()

    def step(self) -> None:
        if self.action.id == GameAction.ACTION7:
            if self._undo_stack:
                self._restore_undo_snapshot()
            self.complete_action()
            return

        if self.action.id == GameAction.ACTION6:
            self._save_undo_snapshot()
            x = self.action.data.get("x", 0)
            y = self.action.data.get("y", 0)
            coords = self.camera.display_to_grid(x, y)
            if coords:
                gx, gy = coords
                rc = self._click_to_rc(gx, gy)
                if rc:
                    r, c = rc
                    if self._selected_rc and (r, c) != self._selected_rc:
                        if self._check_on_leave():
                            return
                    self._select_cell(r, c)
                    if not self._fixed[r][c]:
                        self._toggle(r, c)
                        self._update_errors()
                        if self._is_grid_full() and self._check_complete():
                            self.next_level()
                else:
                    if self._selected_rc:
                        if self._check_on_leave():
                            return
                        self._deselect_cell()
            else:
                if self._selected_rc:
                    if self._check_on_leave():
                        return
                    self._deselect_cell()

        elif self.action.id == GameAction.ACTION1:
            if self._selected_rc:
                r, c = self._selected_rc
                if not self._fixed[r][c]:
                    self._save_undo_snapshot()
                    self._set_cell(r, c, A)
                    self._update_errors()
                    if self._is_grid_full() and self._check_complete():
                        self.next_level()

        elif self.action.id == GameAction.ACTION2:
            if self._selected_rc:
                r, c = self._selected_rc
                if not self._fixed[r][c]:
                    self._save_undo_snapshot()
                    self._set_cell(r, c, B)
                    self._update_errors()
                    if self._is_grid_full() and self._check_complete():
                        self.next_level()

        elif self.action.id == GameAction.ACTION3:
            if self._selected_rc:
                if self._check_on_leave():
                    return
                r, c = self._selected_rc
                c -= 1
                if c < 0:
                    c = self._n - 1
                    r -= 1
                    if r < 0:
                        r = self._n - 1
                self._select_cell(r, c)
            else:
                self._select_cell(0, 0)

        elif self.action.id == GameAction.ACTION4:
            if self._selected_rc:
                if self._check_on_leave():
                    return
                r, c = self._selected_rc
                c += 1
                if c >= self._n:
                    c = 0
                    r += 1
                    if r >= self._n:
                        r = 0
                self._select_cell(r, c)
            else:
                self._select_cell(0, 0)

        self.complete_action()

    def on_set_level(self, level: Level) -> None:
        self._lives = MAX_LIVES
        self._game_over = False
        self._selected_rc = None
        self._undo_stack = []

        n = self.current_level.get_data("grid_n")
        cs = self.current_level.get_data("cell_size")
        gap = self.current_level.get_data("gap")
        ox = self.current_level.get_data("ox")
        oy = self.current_level.get_data("oy")
        sol = self.current_level.get_data("target")
        init = self.current_level.get_data("initial")
        rules = self.current_level.get_data("rules")
        xcon = self.current_level.get_data("x_constraints")

        self._n = n
        self._cs = cs
        self._gap = gap
        self._ox = ox
        self._oy = oy
        self._tgt = sol
        self._rules = set(rules)
        self._xcon = xcon

        self._grid = [[init[r][c] for c in range(n)] for r in range(n)]
        self._fixed = [[init[r][c] != E for c in range(n)] for r in range(n)]

        all_cells = self.current_level.get_sprites_by_tag("cell")
        all_errs = self.current_level.get_sprites_by_tag("err")

        self._cell_spr = [[None] * n for _ in range(n)]
        self._err_spr = [[None] * n for _ in range(n)]

        for spr in all_cells:
            rc = self._pos_to_rc(spr.x, spr.y)
            if rc:
                r, c = rc
                self._cell_spr[r][c] = spr

        for spr in all_errs:
            rc = self._pos_to_rc(spr.x - (cs - 2), spr.y - (cs - 2))
            if rc:
                r, c = rc
                if self._err_spr[r][c] is None:
                    self._err_spr[r][c] = spr

        sel_sprites = self.current_level.get_sprites_by_tag("sel")
        self._sel_spr = sel_sprites[0] if sel_sprites else None

    def _pos_to_rc(self, px, py):
        step = self._cs + self._gap
        c = round((px - self._ox) / step)
        r = round((py - self._oy) / step)
        if 0 <= r < self._n and 0 <= c < self._n:
            return (r, c)
        return None

    def _click_to_rc(self, gx, gy):
        step = self._cs + self._gap
        for r in range(self._n):
            for c in range(self._n):
                cx = self._ox + c * step
                cy = self._oy + r * step
                if cx <= gx < cx + self._cs and cy <= gy < cy + self._cs:
                    return (r, c)
        return None

    def _select_cell(self, r, c):
        self._selected_rc = (r, c)
        if self._sel_spr:
            step = self._cs + self._gap
            x = self._ox + c * step
            y = self._oy + r * step
            self._sel_spr.set_position(x, y)
            self._sel_spr.set_visible(True)

    def _deselect_cell(self):
        self._selected_rc = None
        if self._sel_spr:
            self._sel_spr.set_visible(False)

    def _check_on_leave(self):
        if self._is_grid_full() and not self._check_complete():
            self._lives -= 1
            if self._lives <= 0:
                self._game_over = True
                self.lose()
                self.complete_action()
                return True
            self._reset_grid()
            self._deselect_cell()
            self.complete_action()
            return True
        return False

    def _set_cell(self, r, c, value):
        self._grid[r][c] = value
        col = _color_for(value)
        spr = self._cell_spr[r][c]
        if spr:
            spr.color_remap(None, col)

    def _toggle(self, r, c):
        val = self._grid[r][c]
        if val == E:
            nv = A
        elif val == A:
            nv = B
        else:
            if self._is_grid_full():
                nv = A
            else:
                nv = E
        self._grid[r][c] = nv
        col = _color_for(nv)
        spr = self._cell_spr[r][c]
        if spr:
            spr.color_remap(None, col)

    def _is_grid_full(self):
        for r in range(self._n):
            for c in range(self._n):
                if self._grid[r][c] == E:
                    return False
        return True

    def _reset_grid(self):
        init = self.current_level.get_data("initial")
        for r in range(self._n):
            for c in range(self._n):
                init_val = init[r][c]
                self._grid[r][c] = init_val
                self._fixed[r][c] = init_val != E
                col = _color_for(init_val)
                spr = self._cell_spr[r][c]
                if spr:
                    spr.color_remap(None, col)
        for dot in self.current_level.get_sprites_by_tag("fixed_dot"):
            rc = self._pos_to_rc(
                dot.x - (self._cs // 2 - 1),
                dot.y - (self._cs // 2 - 1),
            )
            if rc:
                r, c = rc
                if init[r][c] == E:
                    dot.set_visible(False)
        self._deselect_cell()
        self._update_errors()

    def _check_complete(self):
        for r in range(self._n):
            for c in range(self._n):
                if self._grid[r][c] != self._tgt[r][c]:
                    return False
        return True

    def _update_errors(self):
        errs = set()

        if "no_three" in self._rules:
            errs |= self._find_no_three()
        if "balanced_rows" in self._rules:
            errs |= self._find_balance_rows()
        if "balanced_cols" in self._rules:
            errs |= self._find_balance_cols()
        if "x_constraints" in self._rules:
            errs |= self._find_x_errors()

        for r in range(self._n):
            for c in range(self._n):
                spr = self._err_spr[r][c]
                if spr:
                    spr.set_visible((r, c) in errs)

    def _find_no_three(self):
        errs = set()
        n = self._n
        g = self._grid
        for r in range(n):
            for c in range(n - 2):
                if g[r][c] >= 0 and g[r][c] == g[r][c + 1] == g[r][c + 2]:
                    errs |= {(r, c), (r, c + 1), (r, c + 2)}
        for c in range(n):
            for r in range(n - 2):
                if g[r][c] >= 0 and g[r][c] == g[r + 1][c] == g[r + 2][c]:
                    errs |= {(r, c), (r + 1, c), (r + 2, c)}
        return errs

    def _find_balance_rows(self):
        errs = set()
        n = self._n
        half = n // 2
        for r in range(n):
            ca = sum(1 for c in range(n) if self._grid[r][c] == A)
            cb = sum(1 for c in range(n) if self._grid[r][c] == B)
            if ca > half or cb > half:
                for c in range(n):
                    if self._grid[r][c] >= 0:
                        errs.add((r, c))
        return errs

    def _find_balance_cols(self):
        errs = set()
        n = self._n
        half = n // 2
        for c in range(n):
            ca = sum(1 for r in range(n) if self._grid[r][c] == A)
            cb = sum(1 for r in range(n) if self._grid[r][c] == B)
            if ca > half or cb > half:
                for r in range(n):
                    if self._grid[r][c] >= 0:
                        errs.add((r, c))
        return errs

    def _find_x_errors(self):
        errs = set()
        for (r1, c1), (r2, c2) in self._xcon:
            v1 = self._grid[r1][c1]
            v2 = self._grid[r2][c2]
            if v1 >= 0 and v2 >= 0 and v1 == v2:
                errs |= {(r1, c1), (r2, c2)}
        return errs


class PuzzleEnvironment:
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

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self._engine = Tk01(seed=seed)
        self._turn = 0
        self._done = False
        self._last_action_was_reset = False

    def reset(self) -> GameState:
        if self._last_action_was_reset:
            self._engine.perform_action(ActionInput(id=GameAction.RESET, data={}))
            self._engine.perform_action(ActionInput(id=GameAction.RESET, data={}))
            self._last_action_was_reset = False
        else:
            self._engine.perform_action(ActionInput(id=GameAction.RESET, data={}))
            self._last_action_was_reset = True
        self._turn = 0
        self._done = False
        return self._get_state()

    def get_actions(self) -> list[str]:
        if self._done:
            return ["reset"]
        actions = ["reset", "up", "down", "left", "right", "click", "undo"]
        return actions

    def step(self, action: str) -> StepResult:
        if action == "reset":
            state = self.reset()
            return StepResult(
                state=state,
                reward=0.0,
                done=False,
                info={"reset": True},
            )

        if action == "click" or action.startswith("click_"):
            self._last_action_was_reset = False
            x, y = 0, 0
            if action.startswith("click_"):
                parts = action.split("_")
                if len(parts) == 3:
                    x, y = int(parts[1]), int(parts[2])
            lives_before = self._engine._lives
            level_before = self._engine.level_index
            self._turn += 1
            self._engine.perform_action(
                ActionInput(id=GameAction.ACTION6, data={"x": x, "y": y})
            )
            return self._build_step_result(lives_before, level_before, action)

        action_map = {
            "reset": GameAction.RESET,
            "up": GameAction.ACTION1,
            "down": GameAction.ACTION2,
            "left": GameAction.ACTION3,
            "right": GameAction.ACTION4,
            "undo": GameAction.ACTION7
        }

        engine_action = action_map.get(action)
        if engine_action is None:
            return StepResult(
                state=self._get_state(),
                reward=0.0,
                done=self._done,
                info={"error": f"Invalid action: {action}"},
            )

        self._last_action_was_reset = False
        lives_before = self._engine._lives
        level_before = self._engine.level_index
        self._turn += 1
        self._engine.perform_action(ActionInput(id=engine_action, data={}))
        return self._build_step_result(lives_before, level_before, action)

    def _build_step_result(
        self, lives_before: int, level_before: int, action: str
    ) -> StepResult:
        game_over = self._engine._game_over
        lives_now = self._engine._lives
        level_now = self._engine.level_index

        reward = 0.0
        done = False
        info: dict = {
            "lives": lives_now,
            "level": level_now + 1,
            "action": action,
        }

        total_levels = len(self._engine._levels)
        level_reward_step = 1.0 / total_levels

        terminated = getattr(self._engine, "_terminated", False)

        if terminated or (level_now != level_before):
            reward = level_reward_step
            last_level_idx = total_levels - 1
            if level_before == last_level_idx:
                info["event"] = "game_complete"
                done = True
            else:
                info["event"] = "level_complete"
        elif game_over or lives_now < lives_before:
            if game_over:
                reward = 0.0
                info["event"] = "game_over"
                done = True
            else:
                reward = 0.0
                info["event"] = "life_lost"

        self._done = done
        return StepResult(
            state=self._get_state(),
            reward=reward,
            done=done,
            info=info,
        )

    def is_done(self) -> bool:
        return self._done

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError(f"Unsupported render mode: {mode}")
        index_grid = self._engine.camera.render(
            self._engine.current_level.get_sprites()
        )
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        if index_grid is not None:
            for idx, color in enumerate(self._ARC_PALETTE):
                mask = index_grid == idx
                rgb[mask] = color
        return rgb

    def close(self) -> None:
        self._engine = None

    def _get_state(self) -> GameState:
        n = self._engine._n if hasattr(self._engine, "_n") else 0
        grid = self._engine._grid if hasattr(self._engine, "_grid") else []
        fixed = self._engine._fixed if hasattr(self._engine, "_fixed") else []
        lives = getattr(self._engine, "_lives", 0)
        lines = []
        val_map = {E: ".", A: "A", B: "B"}
        for r in range(n):
            row_chars = []
            for c in range(n):
                ch = val_map.get(grid[r][c], "?")
                if fixed[r][c] and grid[r][c] != E:
                    ch = ch.lower()
                row_chars.append(ch)
            lines.append(" ".join(row_chars))
        xcon = self._engine._xcon if hasattr(self._engine, "_xcon") else []
        if xcon:
            pairs = []
            for (r1, c1), (r2, c2) in xcon:
                pairs.append(f"({r1},{c1})!=({r2},{c2})")
            lines.append("x:" + " ".join(pairs))
        rules = self._engine._rules if hasattr(self._engine, "_rules") else set()
        if rules:
            lines.append("rules:" + ",".join(sorted(rules)))
        sel = getattr(self._engine, "_selected_rc", None)
        if sel is not None:
            lines.append(f"sel:{sel[0]},{sel[1]}")
        lines.append(f"lives:{lives}")
        text = "\n".join(lines)
        image = None
        try:
            rgb = self.render()
            image = _encode_png(rgb)
        except Exception:
            pass
        return GameState(
            text_observation=text,
            image_observation=image,
            valid_actions=self.get_actions(),
            turn=self._turn,
            metadata={
                "total_levels": len(self._engine._levels),"lives": lives},
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
                base = a.split("_")[0] if a.startswith("click_") else a
                idx = self._string_to_action.get(base)
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
