from __future__ import annotations

import random
import struct
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import gymnasium as gym
from gymnasium import spaces

from arcengine import (
    ARCBaseGame,
    ActionInput,
    Camera,
    GameAction,
    GameState as EngineGameState,
    Level,
    RenderableUserDisplay,
)


BIT_OFF      = 3
BIT_ON       = 14
TARGET_OFF   = 8
TARGET_ON    = 9
MATCH_OFF    = TARGET_OFF
MATCH_ON     = TARGET_ON

SW_FLIP      = 11
SW_RANGE     = 6
SW_XOR       = 2
SW_ROT_L     = 5
SW_ROT_R     = 13
SW_COND      = 10
SW_DECOY     = 1
SW_PRESSED   = 7

BORDER_CLR   = 15
LIFE_ON      = 14
LIFE_OFF     = 3
MOVE_ON      = 12
MOVE_OFF     = 3
BG_CLR       = 0
LBOX_CLR     = 0
WIN_CLR      = 12
LOSE_CLR     = 8
FLASH_CLR    = 15

CURSOR_CLR   = 4

PROGRESS_MULTIPLIER = 6

SWITCH_FLIP_SINGLE  = "FLIP"
SWITCH_FLIP_RANGE   = "RANGE"
SWITCH_XOR          = "XOR"
SWITCH_ROTATE_LEFT  = "ROT_L"
SWITCH_ROTATE_RIGHT = "ROT_R"
SWITCH_CONDITIONAL  = "COND"
SWITCH_DECOY_TYPE   = "DECOY"


@dataclass
class SwitchDef:
    switch_id      : str
    switch_type    : str
    label          : str
    param_a        : int = 0
    param_b        : int = 0
    param_mask     : int = 0
    condition_bit  : int = -1
    condition_value: int = 1


@dataclass
class LevelDef:
    level_number   : int
    bit_width      : int
    initial_state  : Tuple[int, ...]
    target_state   : Tuple[int, ...]
    switches       : List[SwitchDef]
    move_limit     : int
    par_moves      : int


LEVELS: List[LevelDef] = [

    LevelDef(
        level_number=1,
        bit_width=5,
        initial_state=(1, 0, 1, 0, 0),
        target_state=(0, 1, 0, 0, 1),
        switches=[
            SwitchDef("S1", SWITCH_XOR,         "S1", param_mask=0b11001),
            SwitchDef("S2", SWITCH_FLIP_SINGLE, "S2", param_a=2),
            SwitchDef("S3", SWITCH_FLIP_RANGE,  "S3", param_a=0, param_b=1),
        ],
        move_limit=8,
        par_moves=2,
    ),

    LevelDef(
        level_number=2,
        bit_width=6,
        initial_state=(0, 0, 1, 0, 1, 1),
        target_state=(1, 1, 0, 1, 0, 0),
        switches=[
            SwitchDef("S1", SWITCH_ROTATE_LEFT,  "S1"),
            SwitchDef("S2", SWITCH_ROTATE_RIGHT, "S2"),
            SwitchDef("S3", SWITCH_XOR,          "S3", param_mask=0b101010),
            SwitchDef("S4", SWITCH_FLIP_SINGLE,  "S4", param_a=0),
        ],
        move_limit=12,
        par_moves=4,
    ),

    LevelDef(
        level_number=3,
        bit_width=6,
        initial_state=(1, 0, 0, 1, 0, 0),
        target_state=(0, 1, 0, 0, 1, 1),
        switches=[
            SwitchDef("S1", SWITCH_CONDITIONAL, "S1", param_a=3,
                      condition_bit=0, condition_value=1),
            SwitchDef("S2", SWITCH_XOR,         "S2", param_mask=0b010011),
            SwitchDef("S3", SWITCH_FLIP_SINGLE, "S3", param_a=0),
            SwitchDef("S4", SWITCH_FLIP_SINGLE, "S4", param_a=5),
        ],
        move_limit=12,
        par_moves=3,
    ),

    LevelDef(
        level_number=4,
        bit_width=8,
        initial_state=(0, 0, 0, 0, 1, 1, 1, 1),
        target_state=(1, 1, 1, 0, 0, 0, 1, 0),
        switches=[
            SwitchDef("S1", SWITCH_XOR,          "S1", param_mask=0b11110000),
            SwitchDef("S2", SWITCH_ROTATE_LEFT,  "S2"),
            SwitchDef("S3", SWITCH_FLIP_RANGE,   "S3", param_a=4, param_b=7),
            SwitchDef("S4", SWITCH_CONDITIONAL,  "S4", param_a=3,
                      condition_bit=7, condition_value=1),
            SwitchDef("S5", SWITCH_FLIP_SINGLE,  "S5", param_a=0),
        ],
        move_limit=15,
        par_moves=5,
    ),
]

CURSOR_START_POSITIONS = [
    [0, 1, 2, 0],
    [0, 1, 2, 3],
    [0, 1, 2, 3],
    [0, 2, 3, 4],
]


def apply_switch(state: Tuple[int, ...], switch: SwitchDef) -> Tuple[int, ...]:
    bits = list(state)
    n = len(bits)

    if switch.switch_type == SWITCH_FLIP_SINGLE:
        idx = switch.param_a
        if 0 <= idx < n:
            bits[idx] = 1 - bits[idx]

    elif switch.switch_type == SWITCH_FLIP_RANGE:
        for i in range(switch.param_a, min(switch.param_b + 1, n)):
            bits[i] = 1 - bits[i]

    elif switch.switch_type == SWITCH_XOR:
        mask = switch.param_mask
        for i in range(n):
            bits[i] ^= (mask >> (n - 1 - i)) & 1

    elif switch.switch_type == SWITCH_ROTATE_LEFT:
        bits = bits[1:] + [bits[0]]

    elif switch.switch_type == SWITCH_ROTATE_RIGHT:
        bits = [bits[-1]] + bits[:-1]

    elif switch.switch_type == SWITCH_CONDITIONAL:
        ci, cv = switch.condition_bit, switch.condition_value
        if 0 <= ci < n and bits[ci] == cv:
            fi = switch.param_a
            if 0 <= fi < n:
                bits[fi] = 1 - bits[fi]

    elif switch.switch_type == SWITCH_DECOY_TYPE:
        pass

    return tuple(bits)


CANVAS_W      = 64
CANVAS_H      = 64
BIT_CELL_H    = 7
BIT_ROW_Y     = 5
MATCH_ROW_Y   = 13
TARGET_ROW_Y  = 15
SW_AREA_Y     = 26
SW_CELL_H     = 13
LIFE_ROW_Y    = 46
MOVE_ROW_Y    = 52


class BinaryLockHUD(RenderableUserDisplay):

    def __init__(self):
        self.lives          = 3
        self.max_lives      = 3
        self.progress       = 1
        self.progress_max   = 1
        self.current_state  : Tuple[int, ...] = ()
        self.target_state   : Tuple[int, ...] = ()
        self.switches       : List[SwitchDef] = []
        self.cursor_index   = 0
        self.last_pressed   = -1
        self.flash_frames   = 0
        self.game_over      = False
        self.game_won       = False

    @staticmethod
    def _px(frame, x: int, y: int, color: int) -> None:
        if 0 <= x < CANVAS_W and 0 <= y < CANVAS_H:
            frame[y, x] = color

    def _rect(self, frame, x: int, y: int, w: int, h: int, color: int) -> None:
        for dy in range(h):
            for dx in range(w):
                self._px(frame, x + dx, y + dy, color)

    def _bordered_rect(self, frame, x: int, y: int, w: int, h: int,
                       fill: int, border: int) -> None:
        for dx in range(w):
            self._px(frame, x + dx, y, border)
            self._px(frame, x + dx, y + h - 1, border)
        for dy in range(h):
            self._px(frame, x, y + dy, border)
            self._px(frame, x + w - 1, y + dy, border)
        for dy in range(1, h - 1):
            for dx in range(1, w - 1):
                self._px(frame, x + dx, y + dy, fill)

    def _diamond(self, frame, cx: int, cy: int, r: int, color: int) -> None:
        for dy in range(-r, r + 1):
            w = r - abs(dy)
            for dx in range(-w, w + 1):
                self._px(frame, cx + dx, cy + dy, color)

    def _bit_cell_w(self, n: int) -> int:
        usable = CANVAS_W - 6
        gap = 2
        w = (usable - (n - 1) * gap) // n
        return max(3, w)

    def _bit_x(self, n: int, i: int) -> int:
        cw = self._bit_cell_w(n)
        gap = 2
        total = n * cw + (n - 1) * gap
        return (CANVAS_W - total) // 2 + i * (cw + gap)

    def render_interface(self, frame):
        self._draw_border(frame)
        self._draw_bit_row(frame)
        self._draw_target_row(frame)
        self._draw_switches(frame)
        self._draw_lives(frame)
        self._draw_moves(frame)
        if self.flash_frames > 0:
            self._draw_flash(frame)
            self.flash_frames -= 1
        if self.game_over:
            self._draw_overlay(frame, LOSE_CLR)
        if self.game_won:
            self._draw_overlay(frame, WIN_CLR)
        return frame

    def _draw_border(self, frame) -> None:
        for x in range(CANVAS_W):
            self._px(frame, x, 0, BORDER_CLR)
            self._px(frame, x, CANVAS_H - 1, BORDER_CLR)
        for y in range(CANVAS_H):
            self._px(frame, 0, y, BORDER_CLR)
            self._px(frame, CANVAS_W - 1, y, BORDER_CLR)

    def _draw_header(self, frame) -> None:
        bw = CANVAS_W - 8
        sx = 4
        fill = int(bw * self.progress / self.progress_max) if self.progress_max else 0
        fill = max(0, min(bw, fill))
        for dx in range(bw):
            color = 14 if dx < fill else 13
            self._px(frame, sx + dx, 2, color)

    def _draw_section_labels(self, frame) -> None:
        cy_cur = BIT_ROW_Y + BIT_CELL_H // 2
        self._px(frame, 2, cy_cur, CURSOR_CLR)
        self._px(frame, 3, cy_cur, CURSOR_CLR)
        self._px(frame, 2, cy_cur - 1, CURSOR_CLR)
        self._px(frame, 2, cy_cur + 1, CURSOR_CLR)
        cy_tgt = TARGET_ROW_Y + BIT_CELL_H // 2
        self._px(frame, 2, cy_tgt, 8)
        self._px(frame, 3, cy_tgt, 8)
        self._px(frame, 2, cy_tgt - 1, 8)
        self._px(frame, 2, cy_tgt + 1, 8)

    def _draw_bit_row(self, frame) -> None:
        n = len(self.current_state)
        cw = self._bit_cell_w(n)
        for i, bit in enumerate(self.current_state):
            matched = len(self.target_state) > i and bit == self.target_state[i]
            if matched:
                fill = MATCH_ON if bit else MATCH_OFF
            else:
                fill = BIT_ON if bit else BIT_OFF
            bx = self._bit_x(n, i)
            self._bordered_rect(frame, bx, BIT_ROW_Y, cw, BIT_CELL_H, fill, 13)

    def _draw_match_indicators(self, frame) -> None:
        n = min(len(self.current_state), len(self.target_state))
        if n == 0:
            return
        cw = self._bit_cell_w(n)
        for i in range(n):
            cx = self._bit_x(n, i) + cw // 2
            if self.current_state[i] == self.target_state[i]:
                self._px(frame, cx - 1, MATCH_ROW_Y, 3)
                self._px(frame, cx, MATCH_ROW_Y, 3)
                self._px(frame, cx + 1, MATCH_ROW_Y, 3)
            else:
                self._px(frame, cx, MATCH_ROW_Y, 2)

    def _draw_target_row(self, frame) -> None:
        n = len(self.target_state)
        cw = self._bit_cell_w(n)
        for i, bit in enumerate(self.target_state):
            fill = TARGET_ON if bit else TARGET_OFF
            bx = self._bit_x(n, i)
            self._bordered_rect(frame, bx, TARGET_ROW_Y, cw, BIT_CELL_H, fill, 10)

    def _draw_separator(self, frame, y: int) -> None:
        for dx in range(0, CANVAS_W - 4, 3):
            self._px(frame, 2 + dx, y, 13)

    def _sw_cell_w(self, n: int) -> int:
        usable = CANVAS_W - 6
        gap = 2
        w = (usable - (n - 1) * gap) // n
        return max(3, w)

    def _draw_switches(self, frame) -> None:
        n = len(self.switches)
        if not n:
            return
        cw = self._sw_cell_w(n)
        gap = 2
        total = n * cw + (n - 1) * gap
        sx = (CANVAS_W - total) // 2
        for i, sw in enumerate(self.switches):
            bx = sx + i * (cw + gap)
            main_color = SW_PRESSED if i == self.last_pressed else self._sw_color(sw)
            self._bordered_rect(frame, bx, SW_AREA_Y, cw, SW_CELL_H, main_color, 13)
            for dx in range(1, cw - 1):
                self._px(frame, bx + dx, SW_AREA_Y + 1, 5)
            if i == self.cursor_index:
                arrow_cx = bx + cw // 2
                self._px(frame, arrow_cx, SW_AREA_Y - 3, CURSOR_CLR)
                self._px(frame, arrow_cx - 1, SW_AREA_Y - 2, CURSOR_CLR)
                self._px(frame, arrow_cx, SW_AREA_Y - 2, CURSOR_CLR)
                self._px(frame, arrow_cx + 1, SW_AREA_Y - 2, CURSOR_CLR)
                for dx in range(cw + 2):
                    self._px(frame, bx - 1 + dx, SW_AREA_Y - 1, CURSOR_CLR)
                    self._px(frame, bx - 1 + dx, SW_AREA_Y + SW_CELL_H, CURSOR_CLR)
                for dy in range(SW_CELL_H + 2):
                    self._px(frame, bx - 1, SW_AREA_Y - 1 + dy, CURSOR_CLR)
                    self._px(frame, bx + cw, SW_AREA_Y - 1 + dy, CURSOR_CLR)
        self.last_pressed = -1

    def _sw_color(self, sw: SwitchDef) -> int:
        return {
            SWITCH_DECOY_TYPE  : SW_DECOY,
            SWITCH_CONDITIONAL : SW_COND,
            SWITCH_XOR         : SW_XOR,
            SWITCH_ROTATE_LEFT : SW_ROT_L,
            SWITCH_ROTATE_RIGHT: SW_ROT_R,
            SWITCH_FLIP_RANGE  : SW_RANGE,
        }.get(sw.switch_type, SW_FLIP)

    def _draw_lives(self, frame) -> None:
        r = 2
        dia_w = 2 * r + 1
        gap = 3
        total = self.max_lives * dia_w + (self.max_lives - 1) * gap
        sx = (CANVAS_W - total) // 2 + r
        for i in range(self.max_lives):
            cx = sx + i * (dia_w + gap)
            color = LIFE_ON if i < self.lives else LIFE_OFF
            self._diamond(frame, cx, LIFE_ROW_Y, r, color)

    def _draw_moves(self, frame) -> None:
        bw = CANVAS_W - 8
        sx = 4
        fill = int(bw * self.progress / self.progress_max) if self.progress_max else 0
        fill = max(0, min(bw, fill))
        ratio = self.progress / self.progress_max if self.progress_max else 0
        if ratio > 0.66:
            fill_color = MOVE_ON
        elif ratio > 0.33:
            fill_color = 4
        else:
            fill_color = 7
        if self.progress <= 0:
            fill_color = 2
        for dx in range(bw + 2):
            self._px(frame, sx - 1 + dx, MOVE_ROW_Y, 13)
            self._px(frame, sx - 1 + dx, MOVE_ROW_Y + 3, 13)
        self._px(frame, sx - 1, MOVE_ROW_Y + 1, 13)
        self._px(frame, sx - 1, MOVE_ROW_Y + 2, 13)
        self._px(frame, sx + bw, MOVE_ROW_Y + 1, 13)
        self._px(frame, sx + bw, MOVE_ROW_Y + 2, 13)
        for dx in range(bw):
            color = fill_color if dx < fill else MOVE_OFF
            self._px(frame, sx + dx, MOVE_ROW_Y + 1, color)
            self._px(frame, sx + dx, MOVE_ROW_Y + 2, color)

    def _draw_flash(self, frame) -> None:
        for x in range(CANVAS_W):
            self._px(frame, x, 1, FLASH_CLR)
            self._px(frame, x, CANVAS_H - 2, FLASH_CLR)
        for y in range(CANVAS_H):
            self._px(frame, 1, y, FLASH_CLR)
            self._px(frame, CANVAS_W - 2, y, FLASH_CLR)

    def _draw_overlay(self, frame, color: int) -> None:
        cx = CANVAS_W // 2
        cy = CANVAS_H // 2
        ow, oh = 28, 14
        ox = cx - ow // 2
        oy = cy - oh // 2
        self._bordered_rect(frame, ox, oy, ow, oh, BG_CLR, color)
        self._bordered_rect(frame, ox + 3, oy + 3, ow - 6, oh - 6, color, BG_CLR)


_ACTION_MAP: dict[str, GameAction] = {
    "up": GameAction.ACTION1,
    "down": GameAction.ACTION2,
    "left": GameAction.ACTION3,
    "right": GameAction.ACTION4,
    "select": GameAction.ACTION5,
    "undo": GameAction.ACTION7,
    "reset": GameAction.RESET,
}

_VALID_ACTIONS = ["reset", "up", "down", "left", "right", "select", "undo"]

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
        [163, 86, 208],
    ],
    dtype=np.uint8,
)


class Bp62(ARCBaseGame):

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)
        self.hud = BinaryLockHUD()

        self.current_bits: Tuple[int, ...] = ()
        self.target_bits: Tuple[int, ...] = ()
        self.lives = 3
        self.progress = 1
        self.progress_max = 1
        self.game_won = False
        self._undo_stack: list[dict] = []

        lvls = [
            Level(sprites=[], grid_size=(CANVAS_W, CANVAS_H), name=f"Level {i + 1}")
            for i in range(len(LEVELS))
        ]
        cam = Camera(
            x=0, y=0,
            background=BG_CLR, letter_box=LBOX_CLR,
            width=CANVAS_W, height=CANVAS_H,
            interfaces=[self.hud],
        )
        super().__init__(
            "bp62", lvls, cam, available_actions=[0, 1, 2, 3, 4, 5, 7]
        )

    def _save_state(self) -> dict:
        return {
            "current_bits": self.current_bits,
            "cursor_index": self.hud.cursor_index,
        }

    def _restore_state(self, state: dict) -> None:
        self.current_bits = state["current_bits"]
        self.hud.cursor_index = state["cursor_index"]
        self._sync_hud()

    def on_set_level(self, level: Level) -> None:
        ld = LEVELS[self.level_index]
        self.current_bits = ld.initial_state
        self.target_bits = ld.target_state
        self.progress_max = ld.par_moves * PROGRESS_MULTIPLIER
        self.progress = self.progress_max
        self.lives = 3
        self._undo_stack = []
        self.game_won = False
        self.hud.cursor_index = self._rng.choice(CURSOR_START_POSITIONS[self.level_index])
        self.hud.last_pressed = -1
        self.hud.game_over = False
        self.hud.game_won = False
        self._sync_hud()

    def _reset_level(self) -> None:
        ld = LEVELS[self.level_index]
        self.current_bits = ld.initial_state
        self.target_bits = ld.target_state
        self.progress_max = ld.par_moves * PROGRESS_MULTIPLIER
        self.progress = self.progress_max
        self.lives = 3
        self._undo_stack = []
        self.game_won = False
        self.hud.cursor_index = self._rng.choice(CURSOR_START_POSITIONS[self.level_index])
        self.hud.last_pressed = -1
        self.hud.game_over = False
        self.hud.game_won = False
        self._sync_hud()

    def step(self) -> None:
        action_id = self.action.id
        ld = LEVELS[self.level_index]
        n_switches = len(ld.switches)

        if action_id == GameAction.RESET:
            self._reset_level()
            self.complete_action()
            return

        if action_id == GameAction.ACTION1 or action_id == GameAction.ACTION2:
            self._undo_stack.append(self._save_state())
            self.complete_action()
            return

        if action_id == GameAction.ACTION3:
            self._undo_stack.append(self._save_state())
            self.hud.cursor_index = max(0, self.hud.cursor_index - 1)
            self._sync_hud()
            self.complete_action()
            return

        if action_id == GameAction.ACTION4:
            self._undo_stack.append(self._save_state())
            self.hud.cursor_index = min(n_switches - 1, self.hud.cursor_index + 1)
            self._sync_hud()
            self.complete_action()
            return

        if action_id == GameAction.ACTION5:
            self._undo_stack.append(self._save_state())

            if self.progress <= 0:
                self._handle_out_of_moves()
                self.complete_action()
                return

            switch_index = self.hud.cursor_index
            chosen = ld.switches[switch_index]
            self.current_bits = apply_switch(self.current_bits, chosen)
            self.progress -= 1
            self.hud.last_pressed = switch_index

            if self.current_bits == self.target_bits:
                self._handle_level_victory()
                return

            if self.progress <= 0:
                self._handle_out_of_moves()
                self.complete_action()
                return

            self._sync_hud()
            self.complete_action()
            return

        if action_id == GameAction.ACTION7:
            if self._undo_stack:
                prev = self._undo_stack.pop()
                self._restore_state(prev)
            self.progress -= 1
            if self.progress <= 0:
                self._handle_out_of_moves()
            else:
                self._sync_hud()
            self.complete_action()
            return

        self.complete_action()

    def _handle_level_victory(self) -> None:
        self.next_level()
        self.complete_action()

    def _handle_out_of_moves(self) -> None:
        self.hud.flash_frames = 3
        self.lives -= 1
        self._sync_hud()
        if self.lives <= 0:
            self.hud.game_over = True
            self._sync_hud()
            self.lose()
        else:
            saved = self.lives
            self.set_level(self.level_index)
            self.lives = saved
            self._sync_hud()

    def _sync_hud(self) -> None:
        ld = LEVELS[self.level_index]
        self.hud.lives = self.lives
        self.hud.progress = self.progress
        self.hud.progress_max = self.progress_max
        self.hud.current_state = self.current_bits
        self.hud.target_state = self.target_bits
        self.hud.switches = ld.switches


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


class PuzzleEnvironment:

    def __init__(self, seed: int = 0) -> None:
        self._engine = Bp62(seed=seed)
        self._seed = seed
        self._total_turns = 0
        self._done = False
        self._last_action_was_reset = False

    def reset(self) -> GameState:
        if self._is_won() or self._last_action_was_reset:
            self._engine.full_reset()
        else:
            self._engine.perform_action(ActionInput(id=GameAction.RESET))
        self._last_action_was_reset = True
        self._done = False
        self._total_turns = 0
        return self._build_state()

    def get_actions(self) -> List[str]:
        if self._done or self._is_game_over():
            return ["reset"]
        return list(_VALID_ACTIONS)

    def step(self, action: str) -> StepResult:
        if action == "reset":
            state = self.reset()
            return StepResult(
                state=state,
                reward=0.0,
                done=False,
            )

        self._last_action_was_reset = False

        if action not in _ACTION_MAP:
            return StepResult(
                state=self._build_state(),
                reward=0.0,
                done=self._done,
                info={"error": f"Invalid action: {action}"},
            )

        lives_before = self._engine.lives
        level_before = self._engine.level_index

        self._engine.perform_action(ActionInput(id=_ACTION_MAP[action]))
        self._total_turns += 1

        reward = 0.0
        done = False
        info: dict = {
            "lives": self._engine.lives,
            "level": self._engine.level_index + 1,
            "moves_left": self._engine.progress,
            "moves_max": self._engine.progress_max,
        }

        total_levels = len(self._engine._levels)
        level_reward_step = 1.0 / total_levels

        if self._is_game_over():
            reward = 0.0
            info["event"] = "game_over"
            done = True
        elif self._engine.lives < lives_before:
            reward = 0.0
            info["event"] = "life_lost"
        elif self._is_won():
            reward = level_reward_step
            info["event"] = "game_complete"
            done = True
        elif self._engine.level_index != level_before:
            reward = level_reward_step
            info["event"] = "level_complete"

        self._done = done

        return StepResult(
            state=self._build_state(),
            reward=reward,
            done=done,
            info=info,
        )

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

    def is_done(self) -> bool:
        return self._done

    def _encode_png(self, rgb: np.ndarray) -> bytes:
        h, w = rgb.shape[:2]
        raw = b""
        for y in range(h):
            raw += b"\x00" + rgb[y].tobytes()
        compressed = zlib.compress(raw)

        def _chunk(ctype: bytes, data: bytes) -> bytes:
            c = ctype + data
            return (
                struct.pack(">I", len(data))
                + c
                + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
            )

        ihdr_data = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
        out = b"\x89PNG\r\n\x1a\n"
        out += _chunk(b"IHDR", ihdr_data)
        out += _chunk(b"IDAT", compressed)
        out += _chunk(b"IEND", b"")
        return out

    def _is_game_over(self) -> bool:
        try:
            return self._engine._state == EngineGameState.GAME_OVER
        except Exception:
            return False

    def _is_won(self) -> bool:
        try:
            return self._engine._state == EngineGameState.WIN
        except Exception:
            return False

    def _build_state(self) -> GameState:
        ld = LEVELS[self._engine.level_index]
        lines = []

        lines.append(
            f"Level {self._engine.level_index + 1}/{len(self._engine._levels)}"
            f" | Lives: {self._engine.lives}"
            f" | Moves: {self._engine.progress}/{self._engine.progress_max}"
            f" | Turn: {self._total_turns}"
        )
        lines.append("")

        state_str = " ".join(str(b) for b in self._engine.current_bits)
        target_str = " ".join(str(b) for b in self._engine.target_bits)
        lines.append(f"Current : [ {state_str} ]")
        lines.append(f"Target  : [ {target_str} ]")
        lines.append("")

        lines.append("Switches:")
        for i, sw in enumerate(ld.switches):
            cursor_marker = " <--" if i == self._engine.hud.cursor_index else ""
            lines.append(f"  S{i + 1} [{sw.switch_type}]{cursor_marker}")
        lines.append("")

        match_flags = [
            "OK" if self._engine.current_bits[i] == self._engine.target_bits[i] else "--"
            for i in range(ld.bit_width)
        ]
        lines.append(f"Bit match: {' '.join(match_flags)}")
        lines.append("")
        lines.append(f"Actions: {', '.join(_VALID_ACTIONS)}")

        if self._engine.game_won:
            lines.append("*** PUZZLE COMPLETE ***")
        elif self._engine.hud.game_over:
            lines.append("*** GAME OVER ***")

        valid = self.get_actions()

        image_bytes = None
        rgb = self.render()
        image_bytes = self._encode_png(rgb)

        return GameState(
            text_observation="\n".join(lines),
            image_observation=image_bytes,
            valid_actions=valid,
            turn=self._total_turns,
            metadata={
                "total_levels": len(self._engine._levels),
                "level": self._engine.level_index + 1,
                "level_index": self._engine.level_index,
                "levels_completed": getattr(self._engine, "_score", 0),
                "lives": self._engine.lives,
                "moves_left": self._engine.progress,
                "moves_max": self._engine.progress_max,
                "bit_width": ld.bit_width,
                "cursor": self._engine.hud.cursor_index,
                "current_bits": list(self._engine.current_bits),
                "target_bits": list(self._engine.target_bits),
                "game_over": getattr(getattr(self._engine, "_state", None), "name", "") == "GAME_OVER",
                "game_won": self._engine.game_won,
                "done": self._done,
                "info": {},
            },
        )


class ArcGameEnv(gym.Env):
    metadata: Dict[str, Any] = {
        "render_modes": ["rgb_array"],
        "render_fps": 5,
    }

    ACTION_LIST: List[str] = ["reset", "up", "down", "left", "right", "select", "undo"]

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
