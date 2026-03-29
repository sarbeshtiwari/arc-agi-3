from __future__ import annotations

import random
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
from arcengine import GameState as EngineGameState

C_BLACK = 0
C_BLUE = 1
C_RED = 2
C_GREEN = 3
C_YELLOW = 4
C_GRAY = 5
C_MAGENTA = 6
C_ORANGE = 7
C_CYAN = 8
C_BROWN = 9
C_LTBLUE = 10
C_WHITE = 11
C_PINK = 12
C_OLIVE = 13
C_LIME = 14
C_PURPLE = 15

BACKGROUND_COLOR = C_BLACK
GRID_COLOR = C_GRAY
SEPARATOR_COLOR = C_WHITE

CELL_PX = 4

HUD_TOP_HEIGHT = 1
HUD_BOTTOM_HEIGHT = 1


def make_block_sprite(color, name):
    return Sprite(
        pixels=[
            [color, color, color, color],
            [color, color, color, color],
            [color, color, color, color],
            [color, color, color, color],
        ],
        name=name,
        visible=True,
        collidable=False,
        tags=["block", name],
        layer=2,
    )


def make_selection_sprite():
    return Sprite(
        pixels=[
            [C_WHITE, C_WHITE, C_WHITE, C_WHITE],
            [C_WHITE, C_BLACK, C_BLACK, C_WHITE],
            [C_WHITE, C_BLACK, C_BLACK, C_WHITE],
            [C_WHITE, C_WHITE, C_WHITE, C_WHITE],
        ],
        name="selection",
        visible=True,
        collidable=False,
        tags=["selection"],
        layer=3,
    )


def make_grid_cell():
    return Sprite(
        pixels=[
            [C_GRAY, C_GRAY, C_GRAY, C_GRAY],
            [C_GRAY, C_BLACK, C_BLACK, C_GRAY],
            [C_GRAY, C_BLACK, C_BLACK, C_GRAY],
            [C_GRAY, C_GRAY, C_GRAY, C_GRAY],
        ],
        name="grid_cell",
        visible=True,
        collidable=False,
        tags=["grid"],
        layer=0,
    )


def make_separator_sprite(height):
    pixels = []
    for _ in range(height):
        pixels.append([C_WHITE, C_GRAY])
    return Sprite(
        pixels=pixels,
        name="separator",
        visible=True,
        collidable=False,
        tags=["separator"],
        layer=1,
    )


LEVEL_DEFS = [
    {
        "grid_w": 2,
        "grid_h": 2,
        "move_limit": 36,
        "start": [
            [C_RED, C_BLUE],
            [C_RED, C_RED],
        ],
        "target": [
            [C_RED, C_RED],
            [C_BLUE, C_RED],
        ],
    },
    {
        "grid_w": 3,
        "grid_h": 3,
        "move_limit": 44,
        "start": [
            [C_BLUE, C_RED, C_BLUE],
            [C_BLUE, C_RED, C_BLUE],
            [C_BLUE, C_RED, C_BLUE],
        ],
        "target": [
            [C_BLUE, C_BLUE, C_BLUE],
            [C_RED, C_RED, C_RED],
            [C_BLUE, C_BLUE, C_BLUE],
        ],
    },
    {
        "grid_w": 4,
        "grid_h": 4,
        "move_limit": 52,
        "start": [
            [C_RED, C_BLUE, C_RED, C_BLUE],
            [C_BLUE, C_RED, C_BLUE, C_RED],
            [C_BLUE, C_RED, C_BLUE, C_RED],
            [C_RED, C_BLUE, C_RED, C_BLUE],
        ],
        "target": [
            [C_RED, C_RED, C_RED, C_RED],
            [C_BLUE, C_BLUE, C_BLUE, C_BLUE],
            [C_RED, C_RED, C_RED, C_RED],
            [C_BLUE, C_BLUE, C_BLUE, C_BLUE],
        ],
    },
    {
        "grid_w": 5,
        "grid_h": 5,
        "move_limit": 60,
        "start": [
            [C_RED, C_RED, C_RED, C_RED, C_RED],
            [C_RED, C_RED, C_RED, C_RED, C_RED],
            [C_BLUE, C_BLUE, C_BLUE, C_BLUE, C_BLUE],
            [C_RED, C_BLUE, C_RED, C_BLUE, C_RED],
            [C_BLUE, C_RED, C_BLUE, C_RED, C_BLUE],
        ],
        "target": [
            [C_RED, C_RED, C_RED, C_RED, C_RED],
            [C_RED, C_RED, C_RED, C_RED, C_RED],
            [C_RED, C_RED, C_RED, C_RED, C_RED],
            [C_BLUE, C_BLUE, C_BLUE, C_BLUE, C_BLUE],
            [C_BLUE, C_BLUE, C_BLUE, C_BLUE, C_BLUE],
        ],
    },
    {
        "grid_w": 6,
        "grid_h": 6,
        "move_limit": 68,
        "start": [
            [C_RED, C_RED, C_PURPLE, C_RED, C_PURPLE, C_PURPLE],
            [C_RED, C_RED, C_PURPLE, C_RED, C_PURPLE, C_PURPLE],
            [C_PINK, C_PINK, C_PURPLE, C_RED, C_BLUE, C_BLUE],
            [C_RED, C_RED, C_BLUE, C_PINK, C_PURPLE, C_PURPLE],
            [C_PINK, C_PINK, C_BLUE, C_PINK, C_BLUE, C_BLUE],
            [C_PINK, C_PINK, C_BLUE, C_PINK, C_BLUE, C_BLUE],
        ],
        "target": [
            [C_RED, C_RED, C_RED, C_PURPLE, C_PURPLE, C_PURPLE],
            [C_RED, C_RED, C_RED, C_PURPLE, C_PURPLE, C_PURPLE],
            [C_RED, C_RED, C_RED, C_PURPLE, C_PURPLE, C_PURPLE],
            [C_PINK, C_PINK, C_PINK, C_BLUE, C_BLUE, C_BLUE],
            [C_PINK, C_PINK, C_PINK, C_BLUE, C_BLUE, C_BLUE],
            [C_PINK, C_PINK, C_PINK, C_BLUE, C_BLUE, C_BLUE],
        ],
    },
]


def _build_level(ldef, level_idx):
    gw = ldef["grid_w"]
    gh = ldef["grid_h"]
    start_grid = ldef["start"]
    target_grid = ldef["target"]

    y_offset = HUD_TOP_HEIGHT

    lvl_sprites = []

    for gy in range(gh):
        for gx in range(gw):
            px, py = gx * CELL_PX, y_offset + gy * CELL_PX
            cell = make_grid_cell()
            lvl_sprites.append(cell.clone().set_position(px, py))

    separator_x = gw * CELL_PX

    sep_spr = make_separator_sprite(gh * CELL_PX)
    lvl_sprites.append(sep_spr.clone().set_position(separator_x, y_offset))

    target_offset_x = gw * CELL_PX + 2
    for gy in range(gh):
        for gx in range(gw):
            px, py = target_offset_x + gx * CELL_PX, y_offset + gy * CELL_PX
            cell = make_grid_cell()
            lvl_sprites.append(cell.clone().set_position(px, py))

    block_data = []
    for gy in range(gh):
        row_data = []
        for gx in range(gw):
            color = start_grid[gy][gx]
            px, py = gx * CELL_PX, y_offset + gy * CELL_PX
            block_spr = make_block_sprite(color, f"block_{gx}_{gy}")
            lvl_sprites.append(block_spr.clone().set_position(px, py))
            row_data.append(color)
        block_data.append(row_data)

    for gy in range(gh):
        for gx in range(gw):
            color = target_grid[gy][gx]
            px, py = target_offset_x + gx * CELL_PX, y_offset + gy * CELL_PX
            target_spr = Sprite(
                pixels=[
                    [color, color, color, color],
                    [color, color, color, color],
                    [color, color, color, color],
                    [color, color, color, color],
                ],
                name=f"target_{gx}_{gy}",
                visible=True,
                collidable=False,
                tags=["target"],
                layer=2,
            )
            lvl_sprites.append(target_spr.clone().set_position(px, py))

    sel_spr = make_selection_sprite()
    lvl_sprites.append(sel_spr)

    cam_w = (gw * 2) * CELL_PX + 2
    cam_h = gh * CELL_PX + HUD_TOP_HEIGHT + HUD_BOTTOM_HEIGHT

    data = {
        "grid_w": gw,
        "grid_h": gh,
        "blocks": block_data,
        "target": target_grid,
        "start": [row[:] for row in start_grid],
        "level_idx": level_idx,
        "name": f"Level {level_idx + 1}",
    }

    return Level(
        sprites=lvl_sprites,
        grid_size=(cam_w, cam_h),
        data=data,
        name=data["name"],
    )


def _build_all_levels():
    lvls = []
    for i, ldef in enumerate(LEVEL_DEFS):
        lvls.append(_build_level(ldef, i))
    return lvls


def _compute_camera_sizes():
    sizes = []
    for ldef in LEVEL_DEFS:
        gw = ldef["grid_w"]
        gh = ldef["grid_h"]
        cam_w = (gw * 2) * CELL_PX + 2
        cam_h = gh * CELL_PX + HUD_TOP_HEIGHT + HUD_BOTTOM_HEIGHT
        sizes.append((cam_w, cam_h))
    return sizes


_CAMERA_SIZES = _compute_camera_sizes()

CURSOR_START_POSITIONS = [
    [(0, 0), (1, 0), (0, 1), (1, 1)],
    [(0, 0), (1, 1), (2, 0), (0, 2)],
    [(0, 0), (2, 1), (1, 3), (3, 0)],
    [(0, 0), (2, 2), (4, 1), (1, 4)],
    [(0, 0), (3, 2), (5, 1), (2, 5)],
]

MOVE_LIMITS = {
    0: 36,
    1: 44,
    2: 52,
    3: 60,
    4: 68,
}


class SwapHUD(RenderableUserDisplay):
    def __init__(self, game):
        self._game = game
        self._swaps = 0
        self._moves_remaining = 0
        self._game_over = False

    def reset(self):
        self._swaps = 0
        self._game_over = False
        level_idx = self._game.level_index
        if level_idx in MOVE_LIMITS:
            self._moves_remaining = MOVE_LIMITS[level_idx]
        else:
            self._moves_remaining = max(MOVE_LIMITS.values())

    def set_move_limit(self, limit):
        self._moves_remaining = limit
        self._game_over = False

    def add_swap(self):
        self._swaps += 1
        if self._game.level_index >= 0:
            self._moves_remaining -= 1
            if self._moves_remaining <= 0:
                self._game_over = True
                self._game.lose()

    def get_moves_remaining(self):
        return self._moves_remaining

    def is_game_over(self):
        return self._game_over

    def has_move_limit(self):
        return self._game.level_index >= 0

    def render_interface(self, frame):
        level_idx = self._game.level_index
        frame_width = frame.shape[1]
        frame_height = frame.shape[0]
        bottom_row = frame_height - 1

        if level_idx >= 0:
            remaining = max(0, self._moves_remaining)
            max_moves = MOVE_LIMITS.get(level_idx, 36)

            filled_pixels = int((remaining / max_moves) * frame_width)
            filled_pixels = min(filled_pixels, frame_width)

            for i in range(frame_width):
                if i < filled_pixels:
                    frame[0, i] = C_GREEN
                else:
                    frame[0, i] = C_GRAY

            if self._game_over or remaining <= 0:
                for i in range(frame_width):
                    frame[0, i] = C_RED

        level_num = self._game.level_index + 1
        for i in range(level_num):
            px = frame_width - 1 - i
            if px >= 0:
                frame[0, px] = C_YELLOW

        if self._game._all_matched:
            for i in range(frame_width):
                frame[bottom_row, i] = C_GREEN

        return frame


class Gf42(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)
        self._hud = SwapHUD(self)
        self._grid = []
        self._target = []
        self._gw = 0
        self._gh = 0
        self._sel_x = 0
        self._sel_y = 0
        self._all_matched = False
        self._block_sprites = []
        self._selection_sprite = None
        self._locked = False
        self._undo_stack = []
        self._game_over = False

        game_levels = _build_all_levels()

        super().__init__(
            "gf42",
            game_levels,
            Camera(
                x=0,
                y=0,
                width=_CAMERA_SIZES[0][0],
                height=_CAMERA_SIZES[0][1],
                background=BACKGROUND_COLOR,
                letter_box=C_GRAY,
                interfaces=[self._hud],
            ),
            available_actions=[0, 1, 2, 3, 4, 5, 7],
        )

    def on_set_level(self, level):
        idx = min(self.level_index, len(_CAMERA_SIZES) - 1)
        cw, ch = _CAMERA_SIZES[idx]
        self.camera.width = cw
        self.camera.height = ch

        self._gw = self.current_level.get_data("grid_w")
        self._gh = self.current_level.get_data("grid_h")

        start_data = self.current_level.get_data("start")
        self._grid = [row[:] for row in start_data]

        self._target = self.current_level.get_data("target")

        self._block_sprites = []
        for gy in range(self._gh):
            row_sprites = []
            for gx in range(self._gw):
                tag = f"block_{gx}_{gy}"
                sprites = self.current_level.get_sprites_by_tag(tag)
                if sprites:
                    row_sprites.append(sprites[0])
                else:
                    row_sprites.append(None)
            self._block_sprites.append(row_sprites)

        sel_sprites = self.current_level.get_sprites_by_tag("selection")
        if sel_sprites:
            self._selection_sprite = sel_sprites[0]

        self._sel_x, self._sel_y = self._rng.choice(
            CURSOR_START_POSITIONS[self.current_level.get_data("level_idx")]
        )
        self._all_matched = False
        self._locked = False
        self._undo_stack = []
        self._game_over = False
        self._hud.reset()

        self._update_display()
        self._update_selection()
        self._check_match()

    def _save_state(self):
        return {
            "grid": [row[:] for row in self._grid],
            "sel_x": self._sel_x,
            "sel_y": self._sel_y,
            "locked": self._locked,
        }

    def _restore_state(self, state):
        self._grid = [row[:] for row in state["grid"]]
        self._sel_x = state["sel_x"]
        self._sel_y = state["sel_y"]
        self._locked = state["locked"]
        self._all_matched = False
        self._update_display()
        self._update_selection()

    def _update_display(self):
        for gy in range(self._gh):
            for gx in range(self._gw):
                color = self._grid[gy][gx]
                sprite = self._block_sprites[gy][gx]
                if sprite:
                    new_pixels = np.array(
                        [
                            [color, color, color, color],
                            [color, color, color, color],
                            [color, color, color, color],
                            [color, color, color, color],
                        ],
                        dtype=np.uint8,
                    )
                    sprite.pixels = new_pixels

    def _update_selection(self):
        if self._selection_sprite:
            px = self._sel_x * CELL_PX
            py = HUD_TOP_HEIGHT + self._sel_y * CELL_PX
            self._selection_sprite.set_position(px, py)
            self._selection_sprite.set_visible(True)
            border_color = C_YELLOW if self._locked else C_WHITE
            new_pixels = np.array(
                [
                    [border_color, border_color, border_color, border_color],
                    [border_color, C_BLACK, C_BLACK, border_color],
                    [border_color, C_BLACK, C_BLACK, border_color],
                    [border_color, border_color, border_color, border_color],
                ],
                dtype=np.uint8,
            )
            self._selection_sprite.pixels = new_pixels

    def _can_swap(self, dx, dy):
        nx = self._sel_x + dx
        ny = self._sel_y + dy
        return 0 <= nx < self._gw and 0 <= ny < self._gh

    def _swap_blocks(self, dx, dy):
        if not self._can_swap(dx, dy):
            return False

        nx = self._sel_x + dx
        ny = self._sel_y + dy

        temp = self._grid[self._sel_y][self._sel_x]
        self._grid[self._sel_y][self._sel_x] = self._grid[ny][nx]
        self._grid[ny][nx] = temp

        self._sel_x = nx
        self._sel_y = ny

        self._update_display()
        self._update_selection()
        self._hud.add_swap()

        self._check_match()

        return True

    def _move_selection(self, dx, dy):
        nx = self._sel_x + dx
        ny = self._sel_y + dy
        if 0 <= nx < self._gw and 0 <= ny < self._gh:
            self._sel_x = nx
            self._sel_y = ny
            self._update_selection()

    def _check_match(self):
        self._all_matched = True
        for gy in range(self._gh):
            for gx in range(self._gw):
                if self._grid[gy][gx] != self._target[gy][gx]:
                    self._all_matched = False
                    return

        if self._all_matched:
            self.next_level()

    def _reset_level(self):
        start_data = self.current_level.get_data("start")
        self._grid = [row[:] for row in start_data]

        level_idx = self.current_level.get_data("level_idx")
        self._sel_x, self._sel_y = self._rng.choice(CURSOR_START_POSITIONS[level_idx])
        self._all_matched = False
        self._locked = False
        self._undo_stack = []
        self._game_over = False
        self._hud.reset()

        self._update_display()
        self._update_selection()

    def step(self):
        action = self.action.id

        if action == GameAction.RESET:
            self._reset_level()
            self.complete_action()
            return

        if action == GameAction.ACTION1:
            self._undo_stack.append(self._save_state())
            if self._locked:
                self._swap_blocks(0, -1)
            else:
                self._move_selection(0, -1)
            self.complete_action()
            return

        if action == GameAction.ACTION2:
            self._undo_stack.append(self._save_state())
            if self._locked:
                self._swap_blocks(0, 1)
            else:
                self._move_selection(0, 1)
            self.complete_action()
            return

        if action == GameAction.ACTION3:
            self._undo_stack.append(self._save_state())
            if self._locked:
                self._swap_blocks(-1, 0)
            else:
                self._move_selection(-1, 0)
            self.complete_action()
            return

        if action == GameAction.ACTION4:
            self._undo_stack.append(self._save_state())
            if self._locked:
                self._swap_blocks(1, 0)
            else:
                self._move_selection(1, 0)
            self.complete_action()
            return

        if action == GameAction.ACTION5:
            self._undo_stack.append(self._save_state())
            self._locked = not self._locked
            self._update_selection()
            self.complete_action()
            return

        if action == GameAction.ACTION7:
            if self._undo_stack:
                prev = self._undo_stack.pop()
                self._restore_state(prev)
            self._hud.add_swap()
            self.complete_action()
            return

        self.complete_action()


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

COLOR_NAMES = {
    0: "K",
    1: "B",
    2: "R",
    3: "G",
    4: "Y",
    5: "A",
    6: "M",
    7: "O",
    8: "C",
    9: "W",
    10: "L",
    11: "N",
    12: "P",
    13: "D",
    14: "E",
    15: "U",
}


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


def _png_chunk(chunk_type, data):
    chunk = chunk_type + data
    return (
        struct.pack(">I", len(data))
        + chunk
        + struct.pack(">I", zlib.crc32(chunk) & 0xFFFFFFFF)
    )


def _encode_png(rgb):
    h, w = rgb.shape[0], rgb.shape[1]
    raw = b""
    for y in range(h):
        raw += b"\x00" + rgb[y].tobytes()
    compressed = zlib.compress(raw)
    out = b"\x89PNG\r\n\x1a\n"
    out += _png_chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    out += _png_chunk(b"IDAT", compressed)
    out += _png_chunk(b"IEND", b"")
    return out


class PuzzleEnvironment:
    _ACTION_MAP: dict = {
        "reset": GameAction.RESET,
        "up": GameAction.ACTION1,
        "down": GameAction.ACTION2,
        "left": GameAction.ACTION3,
        "right": GameAction.ACTION4,
        "select": GameAction.ACTION5,
        "undo": GameAction.ACTION7,
    }

    _VALID_ACTIONS: list = [
        "reset",
        "up",
        "down",
        "left",
        "right",
        "select",
        "undo",
    ]

    def __init__(self, seed: int = 0) -> None:
        self._engine = Gf42(seed=seed)
        self._total_turns = 0
        self._last_action_was_reset = False
        self._game_over = False

    def reset(self) -> GameState:
        if self._is_won() or self._last_action_was_reset:
            self._engine.full_reset()
        else:
            self._engine.perform_action(ActionInput(id=GameAction.RESET))

        self._last_action_was_reset = True
        self._total_turns = 0
        self._game_over = False
        return self._build_state()

    def step(self, action: str) -> StepResult:
        if action == "reset":
            state = self.reset()
            return StepResult(state=state, reward=0.0, done=False, info={})

        self._last_action_was_reset = False

        game_action = self._ACTION_MAP.get(action)
        if game_action is None:
            state = self._build_state()
            return StepResult(
                state=state,
                reward=0.0,
                done=False,
                info={"error": f"Invalid action: {action}"},
            )

        level_before = self._engine.level_index

        self._engine.perform_action(ActionInput(id=game_action))
        self._total_turns += 1

        reward = 0.0
        done = False
        info: dict = {
            "level": self._engine.level_index + 1,
            "moves_remaining": self._engine._hud.get_moves_remaining(),
        }

        total_levels = len(self._engine._levels)
        level_reward_step = 1.0 / total_levels

        if self._is_game_over():
            reward = 0.0
            info["event"] = "game_over"
            done = True
            self._game_over = True
        elif self._is_won():
            reward = level_reward_step
            info["event"] = "game_complete"
            done = True
        elif self._engine.level_index != level_before:
            reward = level_reward_step
            info["event"] = "level_complete"

        self._game_over = self._game_over or done

        state = self._build_state()
        return StepResult(state=state, reward=reward, done=done, info=info)

    def get_actions(self) -> list[str]:
        if self._game_over or self._is_won() or self._is_game_over():
            return ["reset"]
        return list(self._VALID_ACTIONS)

    def is_done(self) -> bool:
        return self._is_won() or self._game_over or self._is_game_over()

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError(f"Unsupported render mode: {mode}")
        index_grid = self._engine.camera.render(
            self._engine.current_level.get_sprites()
        )
        h, w = index_grid.shape[:2]
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for idx, color in enumerate(ARC_PALETTE):
            mask = index_grid == idx
            rgb[mask] = color
        return rgb

    def close(self) -> None:
        self._engine = None

    def _is_won(self) -> bool:
        if self._engine is None:
            return False
        try:
            return self._engine._state == EngineGameState.WIN
        except Exception:
            return False

    def _is_game_over(self) -> bool:
        try:
            return self._engine._state == EngineGameState.GAME_OVER
        except Exception:
            return False

    def _build_state(self) -> GameState:
        grid = self._engine._grid
        target = self._engine._target
        gw = self._engine._gw
        gh = self._engine._gh
        sel_x = self._engine._sel_x
        sel_y = self._engine._sel_y
        locked = self._engine._locked
        moves_remaining = self._engine._hud.get_moves_remaining()
        level_idx = self._engine.level_index + 1

        lines = []
        lines.append(
            f"Level:{level_idx} Moves:{moves_remaining} Locked:{'Y' if locked else 'N'} Cursor:({sel_x},{sel_y})"
        )
        lines.append("")
        lines.append("Current:")
        for gy in range(gh):
            row_str = ""
            for gx in range(gw):
                c = grid[gy][gx]
                marker = COLOR_NAMES.get(c, "?")
                if gx == sel_x and gy == sel_y:
                    marker = f"[{marker}]"
                else:
                    marker = f" {marker} "
                row_str += marker
            lines.append(row_str)

        lines.append("")
        lines.append("Target:")
        for gy in range(gh):
            row_str = ""
            for gx in range(gw):
                c = target[gy][gx]
                marker = COLOR_NAMES.get(c, "?")
                row_str += f" {marker} "
            lines.append(row_str)

        matched = 0
        total = gw * gh
        for gy in range(gh):
            for gx in range(gw):
                if grid[gy][gx] == target[gy][gx]:
                    matched += 1
        lines.append("")
        lines.append(f"Matched:{matched}/{total}")

        text = "\n".join(lines)

        try:
            index_grid = self._engine.camera.render(
                self._engine.current_level.get_sprites()
            )
            h, w = index_grid.shape[:2]
            rgb = np.zeros((h, w, 3), dtype=np.uint8)
            for idx_c, color in enumerate(ARC_PALETTE):
                mask = index_grid == idx_c
                rgb[mask] = color
            image_bytes = _encode_png(rgb)
        except Exception:
            image_bytes = None

        valid = self.get_actions()

        return GameState(
            text_observation=text,
            image_observation=image_bytes,
            valid_actions=valid,
            turn=self._total_turns,
            metadata={
                "total_levels": len(self._engine._levels),
                "level": level_idx,
                "moves_remaining": moves_remaining,
                "locked": locked,
                "cursor": (sel_x, sel_y),
                "matched": matched,
                "total": total,
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
