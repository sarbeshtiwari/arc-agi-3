import io
import random
import struct
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import gymnasium as gym
import numpy as np
from arcengine import (
    ARCBaseGame,
    Camera,
    GameAction,
    Level,
    Sprite,
)
from arcengine import (
    GameState as EngineGameState,
)
from arcengine.enums import ActionInput
from gymnasium import spaces

GRID_SIZE = (64, 64)
CAM_W, CAM_H = 64, 64

BG_COLOR = 5
PAD_COLOR = 5

BLOCK_COLORS = [9, 14, 8, 11]

CURSOR_COLOR = 0
BUFFER_BORDER_COLOR = 3
EMPTY_COLOR = 4
HAND_BORDER_COLOR = 12
BAR_FULL_COLOR = 6
BAR_EMPTY_COLOR = 4

BS = 4
BG = 1

DEQUE_Y = 6
HAND_Y = 18
BUF_Y = 32
BAR_Y = 47
BAR_H = 3
BAR_W = 56

GOAL_Y = 57
GOAL_BORDER_COLOR = 3

MAX_LIVES = 3
LIFE_ACTIVE_COLOR = 8
LIFE_LOST_COLOR = 4
LIVES_Y = 52

POS_DQ_LEFT = 0
POS_DQ_RIGHT = 1
POS_BUF_1 = 2
POS_BUF_2 = 3


class LevelDef(TypedDict):
    num_blocks: int
    num_colors: int
    counts: List[int]
    num_buffers: int
    max_moves: int


LEVEL_DEFS: List[LevelDef] = [
    {
        "num_blocks": 5,
        "num_colors": 2,
        "counts": [3, 2],
        "num_buffers": 1,
        "max_moves": 40,
    },
    {
        "num_blocks": 6,
        "num_colors": 3,
        "counts": [2, 2, 2],
        "num_buffers": 1,
        "max_moves": 50,
    },
    {
        "num_blocks": 7,
        "num_colors": 3,
        "counts": [3, 2, 2],
        "num_buffers": 1,
        "max_moves": 60,
    },
    {
        "num_blocks": 7,
        "num_colors": 4,
        "counts": [2, 2, 2, 1],
        "num_buffers": 2,
        "max_moves": 70,
    },
    {
        "num_blocks": 8,
        "num_colors": 4,
        "counts": [2, 2, 2, 2],
        "num_buffers": 2,
        "max_moves": 80,
    },
]


def _make_seq(ldef: LevelDef) -> List[int]:
    dq = []
    for i in range(ldef["num_colors"]):
        for _ in range(ldef["counts"][i]):
            dq.append(BLOCK_COLORS[i])
    return dq


def _eval_state(dq: List[int]) -> bool:
    if len(dq) == 0:
        return False
    for i in range(len(dq) - 1):
        idx_a = BLOCK_COLORS.index(dq[i])
        idx_b = BLOCK_COLORS.index(dq[i + 1])
        if idx_a > idx_b:
            return False
    return True


def _generate_puzzle(rng: random.Random, ldef: LevelDef) -> List[int]:
    base = _make_seq(ldef)
    for _ in range(500):
        scrambled = base[:]
        rng.shuffle(scrambled)
        if not _eval_state(scrambled):
            return scrambled
    result = base[:]
    result[0], result[1] = result[1], result[0]
    return result


_ARC_PALETTE: List[Tuple[int, int, int]] = [
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

_COLOR_NAMES: Dict[int, str] = {
    9: "Blue",
    14: "Green",
    8: "Red",
    11: "Yellow",
}

_CURSOR_NAMES: Dict[int, str] = {
    POS_DQ_LEFT: "deque-left",
    POS_DQ_RIGHT: "deque-right",
    POS_BUF_1: "buffer-1",
    POS_BUF_2: "buffer-2",
}


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


class Dq42(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

        self._deque: List[int] = []
        self._initial_deque: List[int] = []
        self._hand: Optional[int] = None
        self._buffers: List[Optional[int]] = [None]
        self._cursor_pos = POS_DQ_LEFT
        self._moves_used = 0
        self._max_moves = 50
        self._ldef: LevelDef = LEVEL_DEFS[0]
        self._history: List[Dict[str, Any]] = []
        self._lives = MAX_LIVES

        camera = Camera(0, 0, CAM_W, CAM_H, BG_COLOR, PAD_COLOR, [])

        levels = [
            Level(sprites=[], grid_size=GRID_SIZE, name=f"Level {i + 1}")
            for i in range(len(LEVEL_DEFS))
        ]

        super().__init__(
            game_id="dq42",
            levels=levels,
            camera=camera,
            available_actions=[0, 1, 2, 3, 4, 5, 6, 7],
            seed=seed,
        )

    def _init_level_state(self, level: Level, regenerate: bool = True) -> None:
        ldef = self._ldef

        self._max_moves = ldef["max_moves"]
        self._moves_used = 0
        self._hand = None
        self._buffers = [None] * ldef["num_buffers"]
        self._cursor_pos = POS_DQ_LEFT
        self._history = []

        if regenerate:
            self._initial_deque = _generate_puzzle(self._rng, ldef)
        self._deque = self._initial_deque[:]

        self._rebuild_sprites(level)

    def on_set_level(self, level: Level) -> None:
        self._ldef = LEVEL_DEFS[self.level_index]
        self._lives = MAX_LIVES
        self._init_level_state(level)

    def _retry_level_inline(self) -> None:
        self._init_level_state(self.current_level, regenerate=False)

    def _rebuild_sprites(self, level: Level) -> None:
        level.remove_all_sprites()
        self._draw_deque_blocks(level)
        self._draw_hand(level)
        self._draw_buffers(level)
        self._draw_move_bar(level)
        self._draw_lives(level)
        self._draw_goal(level)
        self._draw_cursor(level)

    def _deque_origin_x(self):
        n = len(self._deque)
        total_w = n * (BS + BG) - BG if n > 0 else 0
        return (CAM_W - total_w) // 2

    def _draw_deque_blocks(self, level: Level) -> None:
        ox = self._deque_origin_x()
        for i, color in enumerate(self._deque):
            bx = ox + i * (BS + BG)
            pixels = [[color] * BS for _ in range(BS)]
            level.add_sprite(
                Sprite(
                    pixels=pixels,
                    name=f"dq_{i}",
                    x=bx,
                    y=DEQUE_Y,
                    layer=1,
                    tags=["deque"],
                )
            )

    def _draw_hand(self, level: Level) -> None:
        border_size = BS + 2
        hx = (CAM_W - border_size) // 2
        hy = HAND_Y

        border_pixels = [[HAND_BORDER_COLOR] * border_size for _ in range(border_size)]
        level.add_sprite(
            Sprite(
                pixels=border_pixels,
                name="hand_border",
                x=hx,
                y=hy,
                layer=1,
                tags=["hand"],
            )
        )

        inner_color = self._hand if self._hand is not None else EMPTY_COLOR
        inner_pixels = [[inner_color] * BS for _ in range(BS)]
        level.add_sprite(
            Sprite(
                pixels=inner_pixels,
                name="hand_content",
                x=hx + 1,
                y=hy + 1,
                layer=2,
                tags=["hand"],
            )
        )

    def _draw_buffers(self, level: Level) -> None:
        num_buffers = self._ldef["num_buffers"]
        border_size = BS + 2
        slot_gap = 2
        total_w = num_buffers * border_size + (num_buffers - 1) * slot_gap
        start_x = (CAM_W - total_w) // 2

        for i in range(num_buffers):
            bx = start_x + i * (border_size + slot_gap)

            border_pixels = [
                [BUFFER_BORDER_COLOR] * border_size for _ in range(border_size)
            ]
            level.add_sprite(
                Sprite(
                    pixels=border_pixels,
                    name=f"buf_border_{i}",
                    x=bx,
                    y=BUF_Y,
                    layer=1,
                    tags=["buffer"],
                )
            )

            buf_val = self._buffers[i]
            inner_color: int = buf_val if buf_val is not None else EMPTY_COLOR
            inner_pixels = [[inner_color] * BS for _ in range(BS)]
            level.add_sprite(
                Sprite(
                    pixels=inner_pixels,
                    name=f"buf_content_{i}",
                    x=bx + 1,
                    y=BUF_Y + 1,
                    layer=2,
                    tags=["buffer"],
                )
            )

    def _draw_move_bar(self, level: Level) -> None:
        bar_x = (CAM_W - BAR_W) // 2

        remaining = max(0, self._max_moves - self._moves_used)
        filled_w = max(0, (remaining * BAR_W) // self._max_moves)
        empty_w = BAR_W - filled_w

        if filled_w > 0:
            filled_pixels = [[BAR_FULL_COLOR] * filled_w for _ in range(BAR_H)]
            level.add_sprite(
                Sprite(
                    pixels=filled_pixels,
                    name="bar_filled",
                    x=bar_x,
                    y=BAR_Y,
                    layer=1,
                    tags=["bar"],
                )
            )

        if empty_w > 0:
            empty_pixels = [[BAR_EMPTY_COLOR] * empty_w for _ in range(BAR_H)]
            level.add_sprite(
                Sprite(
                    pixels=empty_pixels,
                    name="bar_empty",
                    x=bar_x + filled_w,
                    y=BAR_Y,
                    layer=1,
                    tags=["bar"],
                )
            )

    def _draw_lives(self, level: Level) -> None:
        total_w = MAX_LIVES * 3 - 1
        start_x = (CAM_W - total_w) // 2
        for i in range(MAX_LIVES):
            lx = start_x + i * 3
            color = LIFE_ACTIVE_COLOR if i < self._lives else LIFE_LOST_COLOR
            pixels = [[color] * 2 for _ in range(2)]
            level.add_sprite(
                Sprite(
                    pixels=pixels,
                    name=f"life_{i}",
                    x=lx,
                    y=LIVES_Y,
                    layer=2,
                    tags=["lives"],
                )
            )

    def _draw_goal(self, level: Level) -> None:
        ldef = self._ldef
        goal_colors = []
        for i in range(ldef["num_colors"]):
            goal_colors.append(BLOCK_COLORS[i])

        num_goal = len(goal_colors)
        inner_gap = 1
        inner_w = num_goal * BS + (num_goal - 1) * inner_gap
        inner_h = BS

        box_w = inner_w + 2
        box_h = inner_h + 2
        box_x = (CAM_W - box_w) // 2
        box_y = GOAL_Y

        box_pixels = []
        for row in range(box_h):
            line = []
            for col in range(box_w):
                if row == 0 or row == box_h - 1 or col == 0 or col == box_w - 1:
                    line.append(GOAL_BORDER_COLOR)
                else:
                    line.append(EMPTY_COLOR)
            box_pixels.append(line)

        level.add_sprite(
            Sprite(
                pixels=box_pixels,
                name="goal_box",
                x=box_x,
                y=box_y,
                layer=1,
                tags=["goal"],
            )
        )

        blocks_start_x = box_x + 1
        blocks_start_y = box_y + 1
        for i, color in enumerate(goal_colors):
            bx = blocks_start_x + i * (BS + inner_gap)
            pixels = [[color] * BS for _ in range(BS)]
            level.add_sprite(
                Sprite(
                    pixels=pixels,
                    name=f"goal_{i}",
                    x=bx,
                    y=blocks_start_y,
                    layer=2,
                    tags=["goal"],
                )
            )

    def _draw_cursor(self, level: Level) -> None:
        for s in level.get_sprites_by_tag("cursor"):
            level.remove_sprite(s)

        cx, cy, cw, ch = self._get_cursor_rect(self._cursor_pos)

        if cw <= 0 or ch <= 0:
            return

        if cx < 0:
            cw += cx
            cx = 0
        if cy < 0:
            ch += cy
            cy = 0
        if cx + cw > CAM_W:
            cw = CAM_W - cx
        if cy + ch > CAM_H:
            ch = CAM_H - cy

        if cw <= 0 or ch <= 0:
            return

        cursor_pixels = []
        for row in range(ch):
            line = []
            for col in range(cw):
                if row == 0 or row == ch - 1 or col == 0 or col == cw - 1:
                    line.append(CURSOR_COLOR)
                else:
                    line.append(-1)
            cursor_pixels.append(line)

        level.add_sprite(
            Sprite(
                pixels=cursor_pixels,
                name="cursor",
                x=cx,
                y=cy,
                layer=5,
                tags=["cursor"],
            )
        )

    def _get_cursor_rect(self, pos: int) -> Tuple[int, int, int, int]:
        n = len(self._deque)
        ox = self._deque_origin_x()

        if pos == POS_DQ_LEFT:
            if n > 0:
                return (ox - 1, DEQUE_Y - 1, BS + 2, BS + 2)
            else:
                mid = CAM_W // 2 - 1
                return (mid - 1, DEQUE_Y - 1, BS + 2, BS + 2)

        elif pos == POS_DQ_RIGHT:
            if n > 0:
                right_x = ox + (n - 1) * (BS + BG)
                return (right_x - 1, DEQUE_Y - 1, BS + 2, BS + 2)
            else:
                mid = CAM_W // 2 - 1
                return (mid - 1, DEQUE_Y - 1, BS + 2, BS + 2)

        elif pos == POS_BUF_1:
            num_buffers = self._ldef["num_buffers"]
            border_size = BS + 2
            slot_gap = 2
            total_w = num_buffers * border_size + (num_buffers - 1) * slot_gap
            start_x = (CAM_W - total_w) // 2
            return (start_x - 1, BUF_Y - 1, border_size + 2, border_size + 2)

        elif pos == POS_BUF_2:
            num_buffers = self._ldef["num_buffers"]
            border_size = BS + 2
            slot_gap = 2
            total_w = num_buffers * border_size + (num_buffers - 1) * slot_gap
            start_x = (CAM_W - total_w) // 2
            bx = start_x + border_size + slot_gap
            return (bx - 1, BUF_Y - 1, border_size + 2, border_size + 2)

        return (0, 0, 0, 0)

    def _save_state(self) -> None:
        self._history.append(
            {
                "deque": self._deque[:],
                "hand": self._hand,
                "buffers": self._buffers[:],
            }
        )

    def _restore_state(self) -> None:
        if self._history:
            state = self._history.pop()
            self._deque = state["deque"]
            self._hand = state["hand"]
            self._buffers = state["buffers"]

    def _activate(self) -> bool:
        pos = self._cursor_pos
        num_buffers = self._ldef["num_buffers"]

        if self._hand is None:
            if pos == POS_DQ_LEFT and len(self._deque) > 0:
                self._save_state()
                self._hand = self._deque.pop(0)
                self._moves_used += 1
                return True

            elif pos == POS_DQ_RIGHT and len(self._deque) > 0:
                self._save_state()
                self._hand = self._deque.pop(-1)
                self._moves_used += 1
                return True

            elif pos == POS_BUF_1 and num_buffers > 0 and self._buffers[0] is not None:
                self._save_state()
                self._hand = self._buffers[0]
                self._buffers[0] = None
                self._moves_used += 1
                return True

            elif pos == POS_BUF_2 and num_buffers > 1 and self._buffers[1] is not None:
                self._save_state()
                self._hand = self._buffers[1]
                self._buffers[1] = None
                self._moves_used += 1
                return True

        else:
            if pos == POS_DQ_LEFT:
                self._save_state()
                self._deque.insert(0, self._hand)
                self._hand = None
                self._moves_used += 1
                return True

            elif pos == POS_DQ_RIGHT:
                self._save_state()
                self._deque.append(self._hand)
                self._hand = None
                self._moves_used += 1
                return True

            elif pos == POS_BUF_1 and num_buffers > 0 and self._buffers[0] is None:
                self._save_state()
                self._buffers[0] = self._hand
                self._hand = None
                self._moves_used += 1
                return True

            elif pos == POS_BUF_2 and num_buffers > 1 and self._buffers[1] is None:
                self._save_state()
                self._buffers[1] = self._hand
                self._hand = None
                self._moves_used += 1
                return True

        return False

    def _assess(self) -> bool:
        if self._hand is not None:
            return False
        if any(b is not None for b in self._buffers):
            return False
        if len(self._deque) != self._ldef["num_blocks"]:
            return False
        return _eval_state(self._deque)

    def _draw_win_screen(self, level: Level) -> None:
        level.remove_all_sprites()

        w, h = CAM_W, CAM_H
        pixels = [[BG_COLOR] * w for _ in range(h)]

        for x in range(w):
            pixels[0][x] = pixels[1][x] = 11
            pixels[h - 1][x] = pixels[h - 2][x] = 11
        for y in range(h):
            pixels[y][0] = pixels[y][1] = 11
            pixels[y][w - 1] = pixels[y][w - 2] = 11

        for i in range(15):
            cx, cy = 14 + i, 34 + i
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        pixels[ny][nx] = 14
        for i in range(28):
            cx, cy = 28 + i, 48 - i
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        pixels[ny][nx] = 14

        for sx, sy in [(8, 8), (55, 8), (8, 55), (55, 55)]:
            pixels[sy][sx] = 11
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = sx + dx, sy + dy
                if 0 <= nx < w and 0 <= ny < h:
                    pixels[ny][nx] = 11

        level.add_sprite(
            Sprite(
                pixels=pixels,
                name="win_screen",
                x=0,
                y=0,
                layer=10,
                tags=["message"],
            )
        )

    def handle_reset(self) -> None:
        if self._action_count == 0 or self._state == EngineGameState.WIN:
            self.full_reset()
        else:
            self.level_reset()

    def _grid_to_position(self, gx: int, gy: int) -> Optional[int]:
        num_buffers = self._ldef["num_buffers"]

        if DEQUE_Y - 1 <= gy <= DEQUE_Y + BS:
            if gx < CAM_W // 2:
                return POS_DQ_LEFT
            else:
                return POS_DQ_RIGHT

        if BUF_Y - 1 <= gy <= BUF_Y + BS + 2:
            if num_buffers == 1:
                return POS_BUF_1
            else:
                if gx < CAM_W // 2:
                    return POS_BUF_1
                else:
                    return POS_BUF_2

        return None

    def step(self) -> None:
        action = self.action
        level = self.current_level
        ldef = self._ldef
        num_buffers = ldef["num_buffers"]

        acted = False

        cursor_moved = False
        undid = False

        if action.id == GameAction.ACTION1:
            if self._cursor_pos >= POS_BUF_1:
                if self._cursor_pos == POS_BUF_1:
                    self._cursor_pos = POS_DQ_LEFT
                else:
                    self._cursor_pos = POS_DQ_RIGHT
                cursor_moved = True
            self._moves_used += 1

        elif action.id == GameAction.ACTION2:
            if self._cursor_pos <= POS_DQ_RIGHT:
                if self._cursor_pos == POS_DQ_LEFT:
                    self._cursor_pos = POS_BUF_1
                else:
                    if num_buffers > 1:
                        self._cursor_pos = POS_BUF_2
                    else:
                        self._cursor_pos = POS_BUF_1
                cursor_moved = True
            self._moves_used += 1

        elif action.id == GameAction.ACTION3:
            old_pos = self._cursor_pos
            if self._cursor_pos == POS_DQ_RIGHT:
                self._cursor_pos = POS_DQ_LEFT
            elif self._cursor_pos == POS_BUF_2:
                self._cursor_pos = POS_BUF_1
            if self._cursor_pos != old_pos:
                cursor_moved = True
            self._moves_used += 1

        elif action.id == GameAction.ACTION4:
            old_pos = self._cursor_pos
            if self._cursor_pos == POS_DQ_LEFT:
                self._cursor_pos = POS_DQ_RIGHT
            elif self._cursor_pos == POS_BUF_1 and num_buffers > 1:
                self._cursor_pos = POS_BUF_2
            if self._cursor_pos != old_pos:
                cursor_moved = True
            self._moves_used += 1

        elif action.id == GameAction.ACTION5:
            acted = self._activate()
            if not acted:
                self._moves_used += 1

        elif action.id == GameAction.ACTION6:
            display_x = action.data.get("x", None)
            display_y = action.data.get("y", None)
            if display_x is not None and display_y is not None:
                grid_coords = self.camera.display_to_grid(display_x, display_y)
                if grid_coords is not None:
                    pos = self._grid_to_position(grid_coords[0], grid_coords[1])
                    if pos is not None:
                        old_pos = self._cursor_pos
                        self._cursor_pos = pos
                        acted = self._activate()
                        if not acted:
                            if self._cursor_pos != old_pos:
                                cursor_moved = True
                            self._moves_used += 1
                            self._draw_cursor(level)
                    else:
                        self._moves_used += 1
                else:
                    self._moves_used += 1
            else:
                self._moves_used += 1

        elif action.id == GameAction.ACTION7:
            if self._history:
                self._restore_state()
            self._moves_used += 1
            undid = True
            self._rebuild_sprites(level)

        gameplay_action = action.id in (
            GameAction.ACTION1,
            GameAction.ACTION2,
            GameAction.ACTION3,
            GameAction.ACTION4,
            GameAction.ACTION5,
            GameAction.ACTION6,
            GameAction.ACTION7,
        )

        if cursor_moved or (gameplay_action and not acted):
            self._rebuild_sprites(level)

        if acted:
            self._rebuild_sprites(level)

            if self._assess():
                if self.is_last_level():
                    self._draw_win_screen(level)
                self.next_level()
                self.complete_action()
                return

        if self._moves_used >= self._max_moves:
            self._lives -= 1
            if self._lives > 0:
                self._retry_level_inline()
            else:
                self._rebuild_sprites(level)
                self.lose()
            self.complete_action()
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
        "click": GameAction.ACTION6,
        "undo": GameAction.ACTION7,
    }

    ARC_PALETTE = _ARC_PALETTE

    def __init__(self, seed: int = 0) -> None:
        self._engine: Optional[Dq42] = Dq42(seed=seed)
        self._total_turns = 0
        self._game_won = False
        self._consecutive_resets = 0

    @property
    def _eng(self) -> "Dq42":
        if self._engine is None:
            raise RuntimeError("Engine is not initialized or has been closed")
        return self._engine

    def _build_text_obs(self) -> str:
        e = self._eng
        ldef = e._ldef
        deque_str = ", ".join(_COLOR_NAMES.get(c, str(c)) for c in e._deque)
        hand_str = (
            _COLOR_NAMES.get(e._hand, str(e._hand))
            if e._hand is not None
            else "(empty)"
        )
        buf_lines: List[str] = []
        for i, b in enumerate(e._buffers):
            val = _COLOR_NAMES.get(b, str(b)) if b is not None else "(empty)"
            buf_lines.append(f"  Buffer {i + 1}: {val}")

        remaining = max(0, e._max_moves - e._moves_used)
        goal_seq = _make_seq(ldef)
        goal_str = ", ".join(_COLOR_NAMES.get(c, str(c)) for c in goal_seq)
        cursor_names = {
            POS_DQ_LEFT: "deque-left",
            POS_DQ_RIGHT: "deque-right",
            POS_BUF_1: "buffer-1",
            POS_BUF_2: "buffer-2",
        }

        lines = [
            f"Level:{e.level_index + 1}/{len(LEVEL_DEFS)} Lives:{e._lives}",
            f"Moves:{remaining}/{e._max_moves}",
            f"Goal: [{goal_str}]",
            f"Deque: [{deque_str}]",
            f"Hand: {hand_str}",
            *buf_lines,
            f"Cursor: {cursor_names.get(e._cursor_pos, 'unknown')}",
        ]
        return "\n".join(lines)

    def _build_image_bytes(self) -> Optional[bytes]:
        e = self._eng
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

    def _build_game_state(
        self, done: bool = False, info: Optional[Dict] = None
    ) -> GameState:
        e = self._eng
        valid_actions = self.get_actions() if not done else None
        return GameState(
            text_observation=self._build_text_obs(),
            image_observation=self._build_image_bytes(),
            valid_actions=valid_actions,
            turn=self._total_turns,
            metadata={
                "total_levels": len(self._engine._levels),
                "level_index": e.level_index,
                "lives": e._lives,
                "max_moves": e._ldef["max_moves"],
                "moves_used": e._moves_used,
                "num_blocks": e._ldef["num_blocks"],
                "num_buffers": e._ldef["num_buffers"],
                "hand": e._hand,
                "done": done,
                "info": info or {},
            },
        )

    def reset(self) -> GameState:
        e = self._eng
        self._total_turns = 0
        self._consecutive_resets += 1
        if self._consecutive_resets >= 2:
            e.perform_action(ActionInput(id=GameAction.RESET))
            e.perform_action(ActionInput(id=GameAction.RESET))
            self._consecutive_resets = 0
        else:
            e.perform_action(ActionInput(id=GameAction.RESET))
        self._game_won = False
        return self._build_game_state()

    def get_actions(self) -> List[str]:
        e = self._eng
        game_over = e._state == EngineGameState.GAME_OVER
        if self._game_won or game_over:
            return ["reset"]
        return ["up", "down", "left", "right", "select", "click", "undo", "reset"]

    def is_done(self) -> bool:
        e = self._eng
        return (
            self._game_won
            or e._state == EngineGameState.GAME_OVER
            or e._state == EngineGameState.WIN
        )

    def step(self, action: str) -> StepResult:
        e = self._eng

        if action == "reset":
            state = self.reset()
            return StepResult(
                state=state, reward=0.0, done=False, info={"action": "reset"}
            )

        click_data: Dict[str, Any] = {}
        base_action = action
        if action.startswith("click:"):
            parts = action.split(":")
            if len(parts) == 2:
                try:
                    coords = parts[1].split(",")
                    click_data = {"x": int(coords[0]), "y": int(coords[1])}
                except (ValueError, IndexError):
                    raise ValueError(
                        f"Invalid click format '{action}'. Use 'click:x,y'."
                    )
            base_action = "click"

        if base_action not in self._ACTION_MAP:
            raise ValueError(
                f"Invalid action '{action}'. Must be one of {list(self._ACTION_MAP.keys())}"
            )

        self._consecutive_resets = 0
        self._total_turns += 1

        game_action = self._ACTION_MAP[base_action]
        info: Dict = {"action": action}

        level_before = e.level_index

        action_input = ActionInput(id=game_action, data=click_data)
        frame = e.perform_action(action_input, raw=True)

        game_won = frame and frame.state and frame.state.name == "WIN"
        game_over = e._state == EngineGameState.GAME_OVER
        done = game_over or game_won

        total_levels = len(self._eng._levels)
        reward_per_level = 1.0 / total_levels

        if game_won:
            self._game_won = True
            info["reason"] = "game_complete"
            return StepResult(
                state=self._build_game_state(done=True, info=info),
                reward=reward_per_level,
                done=True,
                info=info,
            )

        if game_over:
            info["reason"] = "death"
            return StepResult(
                state=self._build_game_state(done=True, info=info),
                reward=0.0,
                done=True,
                info=info,
            )

        reward = 0.0
        if e.level_index != level_before:
            reward = reward_per_level
            info["reason"] = "level_complete"

        return StepResult(
            state=self._build_game_state(done=False, info=info),
            reward=reward,
            done=False,
            info=info,
        )

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError(f"Unsupported render mode: {mode}")
        e = self._eng
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
    check_env(env, skip_render_check=False)
    env.close()
