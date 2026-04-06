from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import random
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

C_EMPTY      = 5
C_WALL       = 3
C_INF_A      = 8
C_INF_B      = 12
C_CORE       = 11
C_BARRIER    = 2
C_INF_C      = 6
C_SEL_BAR    = 10
C_CURSOR     = 9
C_DEAD_CORE  = 13
C_WIN        = 14
C_HUD_BAR    = 0
C_HUD_SEL    = 15
C_HUD_LIFE   = 8
C_HUD_TURN   = 12
C_HUD_SPENT  = 4

DISPLAY_SIZE = 64

TOGGLE_X_MIN = DISPLAY_SIZE - 3
TOGGLE_Y_MIN = DISPLAY_SIZE - 3
TOGGLE_X_MAX = DISPLAY_SIZE - 1
TOGGLE_Y_MAX = DISPLAY_SIZE - 1

INF_TYPES = {C_INF_A, C_INF_B, C_INF_C}

ORTHO_DIRS = [(0, -1), (0, 1), (-1, 0), (1, 0)]

DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def _grid_layout(gw: int, gh: int):
    scale = min(DISPLAY_SIZE // gw, DISPLAY_SIZE // gh)
    x_off = (DISPLAY_SIZE - gw * scale) // 2
    y_off = (DISPLAY_SIZE - gh * scale) // 2
    return scale, x_off, y_off


_L1_GRID = [
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [3, 5, 5, 5, 3, 5, 5, 5, 5, 3],
    [3, 8, 5, 5, 3, 5, 5,11, 5, 3],
    [3, 5, 5, 5, 3, 5, 5, 5, 5, 3],
    [3, 5, 5, 5, 5, 5, 5, 5, 5, 3],
    [3, 5, 5, 5, 5, 5, 5, 5, 5, 3],
    [3, 5, 5, 5, 5, 5, 5, 5, 5, 3],
    [3, 5, 5, 5, 3, 5, 5,11, 5, 3],
    [3, 5, 5, 5, 3, 5, 5, 5, 5, 3],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
]


_L2_GRID = [
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [3, 5, 5, 3, 5, 3, 5, 5, 5, 5, 3, 5, 5, 3],
    [3, 5, 5, 5, 5, 3, 5, 5, 5, 5, 3, 5, 5, 3],
    [3, 5, 5, 3, 5, 3, 5, 5, 5, 5, 3, 5, 5, 3],
    [3, 5, 5, 3, 3, 3, 5, 5, 5, 5, 3, 5, 5, 3],
    [3, 8, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 3],
    [3, 5, 5, 5, 5,11, 5, 5,11, 5, 3, 5, 5, 3],
    [3, 5, 5, 3, 5, 5,11, 5, 5, 5, 3, 5, 5, 3],
    [3, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 8, 3],
    [3, 5, 5, 3, 5, 5, 5, 5, 5, 5, 3, 5, 5, 3],
    [3, 5, 5, 3, 5, 5, 5, 5, 5, 5, 3, 5, 5, 3],
    [3, 5, 5, 3, 5, 5, 5, 5, 5, 5, 3, 5, 5, 3],
    [3, 5, 5, 3, 5, 5, 5, 5, 5, 5, 3, 5, 5, 3],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
]


_L3_GRID = [
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [3, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 3],
    [3, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 3],
    [3, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 3],
    [3, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 3],
    [3, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 3],
    [3, 5, 5, 8, 5, 5, 5, 5, 5, 5, 5, 5,12, 5, 5, 3],
    [3, 3, 3, 3, 5, 3, 5, 3, 5, 3, 3, 3, 5, 3, 3, 3],
    [3, 5, 5, 5, 5, 5, 3, 3,11, 5, 5, 5, 5, 5, 5, 3],
    [3, 5, 5, 5, 5,11, 5, 5, 5, 5,11, 5, 5, 5, 5, 3],
    [3, 5, 5, 5, 5, 5, 5,11, 5,11, 5, 5, 5, 5, 5, 3],
    [3, 3, 3, 3, 5, 3, 3, 3, 5, 3, 3, 3, 5, 3, 3, 3],
    [3, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 3],
    [3, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 3],
    [3, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 3],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
]


_L4_GRID = [
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [3, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 3],
    [3, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 3],
    [3, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 3],
    [3, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 3],
    [3, 5, 5, 5, 8, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 3],
    [3, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 3],
    [3, 3, 3, 5, 3, 3, 3, 5, 3, 3, 3, 3, 5, 3, 3, 3, 5, 3, 5, 3],
    [3, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 3, 5, 3],
    [3, 5, 5, 5, 5,11, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 3, 5, 3],
    [3, 5, 5, 5, 5, 5, 5, 5, 5,11, 5, 5, 5, 5,11, 5, 5, 5, 3, 3],
    [3, 5, 5,11, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 3],
    [3, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,11, 5, 5, 5, 5, 5, 5, 5, 3],
    [3, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 3],
    [3, 3, 3, 5, 3, 5, 3, 3, 3, 5, 3, 3, 3, 5, 3, 3, 3, 5, 3, 3],
    [3, 5, 5, 5, 5, 5, 5, 5, 5, 5, 3, 5, 5, 5, 5, 5, 5, 5, 5, 3],
    [3, 5, 5, 5, 5, 5, 5, 5, 5, 5, 3, 5, 5, 5, 5, 5, 5, 5, 5, 3],
    [3, 5, 6, 5, 5, 5, 5, 5, 5, 5, 3, 5, 5, 5, 5, 5,12, 5, 5, 3],
    [3, 5, 5, 5, 5, 5, 5, 5, 5, 5, 3, 5, 5, 5, 5, 5, 5, 5, 5, 3],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
]





LEVEL_CONFIGS = [
    {
        "grid": _L1_GRID, "barriers": 3, "sel_barriers": 0,
        "max_moves": 48, "name": "Basic Block",
    },
    {
        "grid": _L2_GRID, "barriers": 5, "sel_barriers": 0,
        "max_moves": 80, "name": "Two Fronts",
    },
    {
        "grid": _L3_GRID, "barriers": 8, "sel_barriers": 0,
        "max_moves": 96, "name": "Speed Differential",
    },
    {
        "grid": _L4_GRID, "barriers": 7, "sel_barriers": 2,
        "max_moves": 225, "name": "Diagonal Threat",
    },
]


def _build_levels():
    levels = []
    for i, cfg in enumerate(LEVEL_CONFIGS):
        raw = cfg["grid"]
        gh = len(raw)
        gw = len(raw[0])
        grid_arr = np.array(raw, dtype=np.int8)

        sprite = Sprite(
            pixels=grid_arr.tolist(),
            name=f"grid_{i}",
            x=0, y=0,
            layer=0,
        )
        level = Level(
            sprites=[sprite],
            grid_size=(gw, gh),
            data={
                "grid_index": i,
                "gw": gw,
                "gh": gh,
                "barriers": cfg["barriers"],
                "sel_barriers": cfg["sel_barriers"],
            },
            name=cfg["name"],
        )
        levels.append(level)
    return levels


_LEVELS = _build_levels()


def _spread_infections(grid, turn, walls_set, barriers_set, sel_barriers_set):
    gh, gw = grid.shape
    new_infections = set()

    inf_a_cells = set()
    inf_b_cells = set()
    inf_c_cells = set()

    for y in range(gh):
        for x in range(gw):
            val = grid[y, x]
            if val == C_INF_A:
                inf_a_cells.add((x, y))
            elif val == C_INF_B:
                inf_b_cells.add((x, y))
            elif val == C_INF_C:
                inf_c_cells.add((x, y))

    def _can_infect(nx, ny, inf_type):
        if nx < 0 or nx >= gw or ny < 0 or ny >= gh:
            return False
        pos = (nx, ny)
        if pos in walls_set:
            return False
        if pos in barriers_set:
            return False
        if pos in sel_barriers_set:
            if inf_type == C_INF_A:
                return False
            return True
        cell = grid[ny, nx]
        if cell in (C_EMPTY, C_CORE):
            return True
        return False

    for (x, y) in inf_a_cells:
        for dx, dy in ORTHO_DIRS:
            nx, ny = x + dx, y + dy
            if _can_infect(nx, ny, C_INF_A):
                new_infections.add((nx, ny, C_INF_A))

    if turn % 2 == 0:
        for (x, y) in inf_b_cells:
            for dx, dy in ORTHO_DIRS:
                nx, ny = x + dx, y + dy
                if _can_infect(nx, ny, C_INF_B):
                    new_infections.add((nx, ny, C_INF_B))

    for (x, y) in inf_c_cells:
        for dx, dy in DIAG_DIRS:
            nx, ny = x + dx, y + dy
            if _can_infect(nx, ny, C_INF_C):
                new_infections.add((nx, ny, C_INF_C))

    return new_infections


def _check_contained(grid, walls_set, barriers_set, sel_barriers_set):
    gh, gw = grid.shape

    def _can_infect_any(nx, ny, inf_type):
        if nx < 0 or nx >= gw or ny < 0 or ny >= gh:
            return False
        pos = (nx, ny)
        if pos in walls_set:
            return False
        if pos in barriers_set:
            return False
        if pos in sel_barriers_set:
            if inf_type == C_INF_A:
                return False
            return True
        cell = grid[ny, nx]
        return cell in (C_EMPTY, C_CORE)

    for y in range(gh):
        for x in range(gw):
            val = grid[y, x]
            if val == C_INF_A:
                for dx, dy in ORTHO_DIRS:
                    if _can_infect_any(x + dx, y + dy, C_INF_A):
                        return False
            elif val == C_INF_B:
                for dx, dy in ORTHO_DIRS:
                    if _can_infect_any(x + dx, y + dy, C_INF_B):
                        return False
            elif val == C_INF_C:
                for dx, dy in DIAG_DIRS:
                    if _can_infect_any(x + dx, y + dy, C_INF_C):
                        return False
    return True


def _infections_met(grid):
    gh, gw = grid.shape
    for y in range(gh):
        for x in range(gw):
            val = grid[y, x]
            if val not in INF_TYPES:
                continue
            for dx, dy in ORTHO_DIRS:
                nx, ny = x + dx, y + dy
                if 0 <= nx < gw and 0 <= ny < gh:
                    nval = grid[ny, nx]
                    if nval in INF_TYPES and nval != val:
                        return True
    return False


class IfHUD(RenderableUserDisplay):
    def __init__(self, game: "If04") -> None:
        self._game = game

    def _render_barrier_indicators(self, frame, fw, idx):
        max_bar = LEVEL_CONFIGS[idx]["barriers"]
        rem_bar = self._game._barriers_remaining
        for i in range(max_bar):
            if i < fw:
                frame[0, i] = C_HUD_BAR if i < rem_bar else C_HUD_SPENT

        max_sel = LEVEL_CONFIGS[idx]["sel_barriers"]
        rem_sel = self._game._sel_remaining
        sel_start = max_bar + 1
        for i in range(max_sel):
            px = sel_start + i
            if px < fw:
                frame[0, px] = C_HUD_SEL if i < rem_sel else C_HUD_SPENT

    def _render_lives_and_levels(self, frame, fw, idx):
        for i in range(3):
            px = fw - 1 - i
            if px >= 0:
                frame[0, px] = C_HUD_LIFE if i < self._game._lives else C_HUD_SPENT

        for i in range(6):
            px = fw - 1 - 3 - 1 - i
            level_i = 5 - i
            if px >= 0:
                if level_i < idx:
                    frame[0, px] = C_WIN
                elif level_i == idx:
                    frame[0, px] = C_CORE
                else:
                    frame[0, px] = C_HUD_SPENT

    def _render_move_bar(self, frame, fh, fw, idx):
        remaining_moves = max(0, self._game._max_moves - self._game._moves_used)
        max_sel = LEVEL_CONFIGS[idx]["sel_barriers"]
        toggle_width = 3 if max_sel > 0 else 0
        bar_width = fw - 2 - toggle_width

        if self._game._max_moves > 0:
            fill = int((remaining_moves / self._game._max_moves) * bar_width)
        else:
            fill = bar_width

        for i in range(bar_width):
            px = 1 + i
            if px < fw:
                frame[fh - 1, px] = C_HUD_TURN if i < fill else C_HUD_SPENT

    def _render_toggle_button(self, frame, fh, fw, idx):
        max_sel = LEVEL_CONFIGS[idx]["sel_barriers"]
        if max_sel <= 0:
            return

        toggle_color = C_SEL_BAR if self._game._placing_selective else C_BARRIER
        for ty in range(fh - 3, fh):
            for tx in range(fw - 3, fw):
                if 0 <= ty < fh and 0 <= tx < fw:
                    frame[ty, tx] = toggle_color

        if self._game._cursor_on_toggle:
            cursor_color = C_SEL_BAR if self._game._placing_selective else C_CURSOR
            for tx in range(fw - 4, fw):
                if 0 <= tx < fw and fh - 4 >= 0:
                    frame[fh - 4, tx] = cursor_color
            for tx in range(fw - 4, fw):
                if 0 <= tx < fw:
                    frame[fh - 1, tx] = cursor_color
            for ty in range(fh - 4, fh):
                if 0 <= ty < fh and fw - 4 >= 0:
                    frame[ty, fw - 4] = cursor_color
            for ty in range(fh - 4, fh):
                if 0 <= ty < fh:
                    frame[ty, fw - 1] = cursor_color

    def _render_grid_cursor(self, frame, fh, fw, scale, x_off, y_off):
        if self._game._cursor_on_toggle:
            return

        cx, cy = self._game._cursor_x, self._game._cursor_y
        px_left = x_off + cx * scale
        px_right = px_left + scale - 1
        py_top = y_off + cy * scale
        py_bot = py_top + scale - 1

        cursor_color = C_SEL_BAR if self._game._placing_selective else C_CURSOR

        for px in range(px_left, px_right + 1):
            if 0 <= px < fw and 0 <= py_top < fh:
                frame[py_top, px] = cursor_color
        for px in range(px_left, px_right + 1):
            if 0 <= px < fw and 0 <= py_bot < fh:
                frame[py_bot, px] = cursor_color
        for py in range(py_top, py_bot + 1):
            if 0 <= px_left < fw and 0 <= py < fh:
                frame[py, px_left] = cursor_color
        for py in range(py_top, py_bot + 1):
            if 0 <= px_right < fw and 0 <= py < fh:
                frame[py, px_right] = cursor_color

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        idx = self._game._current_level_index
        if idx < 0 or idx >= len(LEVEL_CONFIGS):
            return frame

        fh, fw = frame.shape
        gw, gh = self._game._gw, self._game._gh
        scale, x_off, y_off = _grid_layout(gw, gh)

        self._render_barrier_indicators(frame, fw, idx)
        self._render_lives_and_levels(frame, fw, idx)
        self._render_move_bar(frame, fh, fw, idx)
        self._render_toggle_button(frame, fh, fw, idx)
        self._render_grid_cursor(frame, fh, fw, scale, x_off, y_off)

        return frame


class If04(ARCBaseGame):

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self._rng = random.Random(seed)
        self._grids: list = []
        self._gw: int = 0
        self._gh: int = 0
        self._cursor_x: int = 0
        self._cursor_y: int = 0
        self._turn: int = 0
        self._barriers_remaining: int = 0
        self._sel_remaining: int = 0
        self._placing_selective: bool = False
        self._cursor_on_toggle: bool = False
        self._lives: int = 3
        self._moves_used: int = 0
        self._max_moves: int = 0
        self._walls_set: set = set()
        self._barriers_set: set = set()
        self._sel_barriers_set: set = set()
        self._core_cells: set = set()
        self._history: list = []

        for cfg in LEVEL_CONFIGS:
            raw = cfg["grid"]
            self._grids.append(np.array(raw, dtype=np.int8))

        hud = IfHUD(self)

        gw0 = len(LEVEL_CONFIGS[0]["grid"][0])
        gh0 = len(LEVEL_CONFIGS[0]["grid"])

        super().__init__(
            game_id="if04",
            levels=_LEVELS,
            camera=Camera(
                0, 0, gw0, gh0,
                C_EMPTY,
                C_EMPTY,
                [hud],
            ),
            available_actions=[0, 1, 2, 3, 4, 5, 7],
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        idx = level.get_data("grid_index")
        cfg = LEVEL_CONFIGS[idx]
        self._gw = level.get_data("gw")
        self._gh = level.get_data("gh")

        self.camera.width = self._gw
        self.camera.height = self._gh

        self._grids[idx] = np.array(cfg["grid"], dtype=np.int8)

        self._lives = 3
        self._turn = 0
        self._barriers_remaining = cfg["barriers"]
        self._sel_remaining = cfg["sel_barriers"]
        self._placing_selective = False
        self._cursor_on_toggle = False
        self._moves_used = 0
        self._max_moves = cfg["max_moves"]

        self._cursor_x, self._cursor_y = self._random_empty_position(idx)

        self._rebuild_sets(idx)
        self._sync_sprite()
        self._history = []

    def _rebuild_sets(self, idx: int) -> None:
        grid = self._grids[idx]
        gh, gw = grid.shape

        self._walls_set = set()
        self._barriers_set = set()
        self._sel_barriers_set = set()
        self._core_cells = set()

        for y in range(gh):
            for x in range(gw):
                val = grid[y, x]
                if val == C_WALL:
                    self._walls_set.add((x, y))
                elif val == C_BARRIER:
                    self._barriers_set.add((x, y))
                elif val == C_SEL_BAR:
                    self._sel_barriers_set.add((x, y))
                elif val == C_CORE:
                    self._core_cells.add((x, y))

    def _sync_sprite(self) -> None:
        idx = self._current_level_index
        sprites = self.current_level.get_sprites_by_name(f"grid_{idx}")
        if sprites:
            sprites[0].pixels = np.array(self._grids[idx], dtype=np.int8)

    def _random_empty_position(self, idx: int) -> Tuple[int, int]:
        grid = self._grids[idx]
        gh, gw = grid.shape
        empty_cells = []
        for y in range(gh):
            for x in range(gw):
                if grid[y, x] == C_EMPTY:
                    empty_cells.append((x, y))
        if empty_cells:
            return self._rng.choice(empty_cells)
        return (gw // 2, gh // 2)

    def _save_state(self) -> None:
        idx = self._current_level_index
        self._history.append({
            "grid": self._grids[idx].copy(),
            "turn": self._turn,
            "barriers_remaining": self._barriers_remaining,
            "sel_remaining": self._sel_remaining,
            "placing_selective": self._placing_selective,
            "cursor_x": self._cursor_x,
            "cursor_y": self._cursor_y,
            "cursor_on_toggle": self._cursor_on_toggle,
            "moves_used": self._moves_used,
            "walls_set": set(self._walls_set),
            "barriers_set": set(self._barriers_set),
            "sel_barriers_set": set(self._sel_barriers_set),
            "core_cells": set(self._core_cells),
        })

    def _undo(self) -> None:
        if not self._history:
            return
        snap = self._history.pop()
        idx = self._current_level_index
        self._grids[idx] = snap["grid"]
        self._turn = snap["turn"]
        self._barriers_remaining = snap["barriers_remaining"]
        self._sel_remaining = snap["sel_remaining"]
        self._placing_selective = snap["placing_selective"]
        self._cursor_x = snap["cursor_x"]
        self._cursor_y = snap["cursor_y"]
        self._cursor_on_toggle = snap["cursor_on_toggle"]
        self._moves_used += 1
        self._walls_set = snap["walls_set"]
        self._barriers_set = snap["barriers_set"]
        self._sel_barriers_set = snap["sel_barriers_set"]
        self._core_cells = snap["core_cells"]
        self._sync_sprite()

    def step(self) -> None:
        action = self.action

        if action.id == GameAction.RESET:
            self._history = []
            self.complete_action()
            return

        action_id = action.id.value

        if action_id == 7:
            self._undo()
            if self._moves_used >= self._max_moves:
                if self._barriers_remaining > 0 or self._sel_remaining > 0:
                    self._die()
            self.complete_action()
            return

        self._moves_used += 1
        self._save_state()

        if action_id == 1:
            self._move_cursor(0, -1)
        elif action_id == 2:
            self._move_cursor(0, 1)
        elif action_id == 3:
            self._move_cursor(-1, 0)
        elif action_id == 4:
            self._move_cursor(1, 0)
        elif action_id == 5:
            if self._cursor_on_toggle:
                idx = self._current_level_index
                if LEVEL_CONFIGS[idx]["sel_barriers"] > 0:
                    self._placing_selective = not self._placing_selective
            else:
                self._place_barrier_at_cursor()

        if self._moves_used >= self._max_moves:
            if self._barriers_remaining > 0 or self._sel_remaining > 0:
                self._die()

        self.complete_action()

    def _move_cursor(self, dx: int, dy: int) -> None:
        idx = self._current_level_index
        has_sel = LEVEL_CONFIGS[idx]["sel_barriers"] > 0

        if self._cursor_on_toggle:
            self._cursor_on_toggle = False
            self._cursor_x = self._gw - 1
            self._cursor_y = self._gh - 1
            return

        new_x = self._cursor_x + dx
        new_y = self._cursor_y + dy

        if has_sel and (new_x >= self._gw or new_y >= self._gh):
            if self._cursor_x >= self._gw - 1 and self._cursor_y >= self._gh - 1:
                self._cursor_on_toggle = True
                return

        self._cursor_x = max(0, min(self._gw - 1, new_x))
        self._cursor_y = max(0, min(self._gh - 1, new_y))

    def _place_barrier_at_cursor(self) -> None:
        gx, gy = self._cursor_x, self._cursor_y

        idx = self._current_level_index
        grid = self._grids[idx]

        if not self._place_barrier(grid, gx, gy):
            return

        new_inf = _spread_infections(
            grid, self._turn,
            self._walls_set, self._barriers_set, self._sel_barriers_set
        )

        core_hit = self._apply_infections(grid, new_inf)
        self._turn += 1
        self._sync_sprite()

        if core_hit:
            self._die()
            return

        if _infections_met(grid):
            self._die()
            return

        if _check_contained(grid, self._walls_set, self._barriers_set,
                            self._sel_barriers_set):
            if len(self._core_cells) > 0:
                self.next_level()
                return

        if self._barriers_remaining <= 0 and self._sel_remaining <= 0:
            self._simulate_remaining_infections(grid)

    def _place_barrier(self, grid, gx, gy):
        if grid[gy, gx] != C_EMPTY:
            return False

        if self._placing_selective:
            if self._sel_remaining <= 0:
                return False
            grid[gy, gx] = C_SEL_BAR
            self._sel_barriers_set.add((gx, gy))
            self._sel_remaining -= 1
        else:
            if self._barriers_remaining <= 0:
                if self._sel_remaining > 0:
                    self._placing_selective = True
                    grid[gy, gx] = C_SEL_BAR
                    self._sel_barriers_set.add((gx, gy))
                    self._sel_remaining -= 1
                else:
                    return False
            else:
                grid[gy, gx] = C_BARRIER
                self._barriers_set.add((gx, gy))
                self._barriers_remaining -= 1

        return True

    def _apply_infections(self, grid, new_inf):
        core_hit = False
        for (nx, ny, color) in new_inf:
            if grid[ny, nx] == C_CORE:
                grid[ny, nx] = C_DEAD_CORE
                self._core_cells.discard((nx, ny))
                core_hit = True
            elif grid[ny, nx] == C_SEL_BAR:
                grid[ny, nx] = color
                self._sel_barriers_set.discard((nx, ny))
            else:
                grid[ny, nx] = color
        return core_hit

    def _simulate_remaining_infections(self, grid):
        consecutive_empty = 0
        for _ in range(500):
            new_inf = _spread_infections(
                grid, self._turn,
                self._walls_set, self._barriers_set, self._sel_barriers_set
            )
            if not new_inf:
                consecutive_empty += 1
            if consecutive_empty >= 4:
                if _infections_met(grid):
                    self._sync_sprite()
                    self._die()
                    return
                if _check_contained(grid, self._walls_set,
                                    self._barriers_set,
                                    self._sel_barriers_set):
                    if len(self._core_cells) > 0:
                        self._sync_sprite()
                        self.next_level()
                        return
                self._sync_sprite()
                self._die()
                return

            consecutive_empty = 0

            core_hit = self._apply_infections(grid, new_inf)
            if core_hit:
                self._sync_sprite()
                self._die()
                return

            if _infections_met(grid):
                self._sync_sprite()
                self._die()
                return

            self._turn += 1

            if _check_contained(grid, self._walls_set,
                                self._barriers_set,
                                self._sel_barriers_set):
                if len(self._core_cells) > 0:
                    self._sync_sprite()
                    self.next_level()
                    return

        self._sync_sprite()
        self._die()

    def handle_reset(self) -> None:
        if getattr(self._state, "name", None) == "GAME_OVER":
            if self._action_count == 0:
                self.full_reset()
            else:
                self._lives = 3
                self.level_reset()
        else:
            super().handle_reset()

    def level_reset(self) -> None:
        idx = self._current_level_index
        self._grids[idx] = np.array(LEVEL_CONFIGS[idx]["grid"], dtype=np.int8)
        self._history = []
        super().level_reset()

    def full_reset(self) -> None:
        self._lives = 3
        self._cursor_on_toggle = False
        self._history = []
        for i, cfg in enumerate(LEVEL_CONFIGS):
            self._grids[i] = np.array(cfg["grid"], dtype=np.int8)
        super().full_reset()

    def _die(self) -> None:
        self._lives -= 1
        if self._lives <= 0:
            self.lose()
            return
        idx = self._current_level_index
        cfg = LEVEL_CONFIGS[idx]
        self._grids[idx] = np.array(cfg["grid"], dtype=np.int8)
        self._turn = 0
        self._barriers_remaining = cfg["barriers"]
        self._sel_remaining = cfg["sel_barriers"]
        self._placing_selective = False
        self._cursor_on_toggle = False
        self._moves_used = 0
        self._max_moves = cfg["max_moves"]
        self._cursor_x, self._cursor_y = self._random_empty_position(idx)
        self._rebuild_sets(idx)
        self._sync_sprite()
        self._history = []


ARC_PALETTE = np.array([
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
], dtype=np.uint8)

_CELL_CHARS = {
    C_EMPTY: ".",
    C_WALL: "#",
    C_INF_A: "A",
    C_INF_B: "B",
    C_INF_C: "D",
    C_CORE: "O",
    C_BARRIER: "=",
    C_SEL_BAR: "~",
    C_DEAD_CORE: "X",
    C_WIN: "W",
}


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
        self._engine: Any = If04(seed=seed)
        self._total_turns = 0
        self._last_action_was_reset = False
        self._game_won = False
        self._game_over = False
        self._done = False

    def _build_text_obs(self) -> str:
        e = self._engine
        idx = e._current_level_index
        grid = e._grids[idx]
        gh, gw = grid.shape
        rows = []
        for y in range(gh):
            chars = []
            for x in range(gw):
                if not e._cursor_on_toggle and x == e._cursor_x and y == e._cursor_y:
                    chars.append("@")
                else:
                    chars.append(_CELL_CHARS.get(int(grid[y, x]), "?"))
            rows.append("".join(chars))
        remaining = max(0, e._max_moves - e._moves_used)
        mode = "selective" if e._placing_selective else "normal"
        header = (
            f"Level:{idx + 1}/{len(e._levels)} "
            f"Lives:{e._lives} "
            f"Barriers:{e._barriers_remaining} "
            f"SelBarriers:{e._sel_remaining} "
            f"Moves:{remaining}/{e._max_moves} "
            f"Mode:{mode}"
        )
        return header + "\n" + "\n".join(rows)

    @staticmethod
    def _encode_png(rgb: np.ndarray) -> bytes:
        h, w, _ = rgb.shape
        raw_rows = []
        for y in range(h):
            raw_rows.append(b"\x00" + rgb[y].tobytes())
        raw_data = b"".join(raw_rows)
        compressed = zlib.compress(raw_data)

        def _chunk(ctype: bytes, data: bytes) -> bytes:
            c = ctype + data
            return (
                struct.pack(">I", len(data))
                + c
                + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
            )

        ihdr_data = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
        png = b"\x89PNG\r\n\x1a\n"
        png += _chunk(b"IHDR", ihdr_data)
        png += _chunk(b"IDAT", compressed)
        png += _chunk(b"IEND", b"")
        return png

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
        for idx, color in enumerate(ARC_PALETTE):
            mask = arr == idx
            rgb[mask] = color
        return self._encode_png(rgb)

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
                "total_levels": len(e._levels),
                "level_index": e.level_index,
                "levels_completed": e._current_level_index,
                "lives": e._lives,
                "game_over": self._game_over,
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
        self._last_action_was_reset = True
        self._game_won = False
        self._game_over = False
        self._done = False
        return self._build_game_state()

    def get_actions(self) -> List[str]:
        if self._done or self._game_over:
            return ["reset"]
        return list(self._VALID_ACTIONS)

    def is_done(self) -> bool:
        return self._game_won

    def step(self, action: str) -> StepResult:
        e = self._engine

        action = str(action).strip().lower()

        if action == "reset":
            state = self.reset()
            return StepResult(
                state=state, reward=0.0, done=False, info={"action": "reset"}
            )

        if action not in self._ACTION_MAP:
            return StepResult(
                state=self._build_game_state(),
                reward=0.0,
                done=False,
                info={"action": action, "error": "invalid_action"},
            )

        self._last_action_was_reset = False
        self._total_turns += 1

        game_action = self._ACTION_MAP[action]
        info: Dict = {"action": action}

        total_levels = len(e._levels)
        level_before = e.level_index

        action_input = ActionInput(id=game_action)
        frame = e.perform_action(action_input, raw=True)

        state_name = frame.state.name if frame and frame.state else ""
        game_won = state_name == "WIN"
        game_over = state_name == "GAME_OVER"
        level_advanced = e.level_index > level_before
        done = game_over or game_won

        reward = 0.0
        level_reward = 1.0 / total_levels

        if game_won or level_advanced:
            reward = level_reward

        if game_won:
            self._game_won = True
            self._done = True
            info["reason"] = "game_complete"
            return StepResult(
                state=self._build_game_state(done=True, info=info),
                reward=reward,
                done=True,
                info=info,
            )

        if game_over:
            self._game_over = True
            info["reason"] = "death"
            return StepResult(
                state=self._build_game_state(done=False, info=info),
                reward=0.0,
                done=False,
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
        arr = np.array(index_grid, dtype=np.uint8)
        h, w = arr.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for idx, color in enumerate(ARC_PALETTE):
            mask = arr == idx
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
        return img[row_idx[:, None], col_idx[None, :]]

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
    obs, reward, term, trunc, info = env.step(0)
    env.render()
    env.close()
