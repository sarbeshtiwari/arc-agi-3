import io
import random
import struct
import zlib
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

import gymnasium as gym
import numpy as np
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
    metadata: dict = field(default_factory=dict)


@dataclass
class StepResult:
    state: GameState
    reward: float
    done: bool
    info: dict = field(default_factory=dict)


WHITE = 0
OFF_WHITE = 1
LIGHT_GREY = 2
GREY = 3
DARK_GREY = 4
BLACK = 5
MAGENTA = 6
PINK = 7
RED = 8
BLUE = 9
LIGHT_BLUE = 10
YELLOW = 11
ORANGE = 12
MAROON = 13
GREEN = 14
PURPLE = 15

GRID_W = 64
GRID_H = 64

TOKEN_COLORS = {
    "R": RED,
    "B": BLUE,
    "Y": YELLOW,
    "P": PURPLE,
    "G": GREEN,
    "O": ORANGE,
}

PRIMARY_COLORS = {"R", "B", "Y"}
SECONDARY_COLORS = {"P", "G", "O"}

MIX_TABLE = {
    ("R", "B"): "P",
    ("B", "R"): "P",
    ("B", "Y"): "G",
    ("Y", "B"): "G",
    ("R", "Y"): "O",
    ("Y", "R"): "O",
}

SPLIT_TABLE = {
    "P": ("R", "B"),
    "G": ("B", "Y"),
    "O": ("R", "Y"),
}

CONVERT_TABLE = {
    "R": "B",
    "B": "Y",
    "Y": "R",
}

REVERSE_CONVERT_TABLE = {
    "R": "Y",
    "B": "R",
    "Y": "B",
}

MACHINE_MIX = 0
MACHINE_DUPLICATE = 1
MACHINE_DELETE = 2
MACHINE_SPLIT = 3
MACHINE_CONVERT = 4
MACHINE_BLUFF = 5

MACHINE_NAMES = {
    MACHINE_MIX: "MIX",
    MACHINE_DUPLICATE: "DUP",
    MACHINE_DELETE: "DEL",
    MACHINE_SPLIT: "SPL",
    MACHINE_CONVERT: "CNV",
    MACHINE_BLUFF: "BLF",
}

MACHINE_COLORS = {
    MACHINE_MIX: MAGENTA,
    MACHINE_DUPLICATE: LIGHT_BLUE,
    MACHINE_DELETE: MAROON,
    MACHINE_SPLIT: PINK,
    MACHINE_CONVERT: LIGHT_GREY,
    MACHINE_BLUFF: ORANGE,
}

T = -1
K = 5

ICON_MIX = [
    [K, T, T, T, K],
    [T, K, T, K, T],
    [T, T, K, T, T],
    [T, T, K, T, T],
    [T, T, K, T, T],
]

ICON_DUPLICATE = [
    [T, T, K, T, T],
    [T, T, K, T, T],
    [T, T, K, T, T],
    [T, K, T, K, T],
    [K, T, T, T, K],
]

ICON_DELETE = [
    [K, T, T, T, K],
    [T, K, T, K, T],
    [T, T, K, T, T],
    [T, K, T, K, T],
    [K, T, T, T, K],
]

ICON_SPLIT = [
    [T, T, K, T, T],
    [T, K, K, K, T],
    [K, K, T, K, K],
    [T, K, K, K, T],
    [T, T, K, T, T],
]

ICON_CONVERT = [
    [T, K, K, K, T],
    [K, T, T, T, K],
    [T, T, T, T, K],
    [K, T, T, T, K],
    [T, K, K, K, T],
]

ICON_BLUFF = [
    [T, T, K, K, T],
    [T, K, K, T, T],
    [K, K, K, K, K],
    [T, T, K, K, T],
    [T, K, K, T, T],
]

MACHINE_ICONS = {
    MACHINE_MIX: ICON_MIX,
    MACHINE_DUPLICATE: ICON_DUPLICATE,
    MACHINE_DELETE: ICON_DELETE,
    MACHINE_SPLIT: ICON_SPLIT,
    MACHINE_CONVERT: ICON_CONVERT,
    MACHINE_BLUFF: ICON_BLUFF,
}

LEVEL_DEFS = [
    {
        "start": ["R", "B", "Y"],
        "goal": ["G", "R"],
        "machines": [MACHINE_MIX],
        "max_moves": 21,
    },
    {
        "start": ["G", "R"],
        "goal": ["O", "B"],
        "machines": [MACHINE_MIX, MACHINE_SPLIT],
        "max_moves": 42,
    },
    {
        "start": ["G"],
        "goal": ["B", "Y", "B"],
        "machines": [MACHINE_MIX, MACHINE_SPLIT, MACHINE_DUPLICATE],
        "max_moves": 63,
    },
    {
        "start": ["R", "Y"],
        "goal": ["G", "G"],
        "machines": [MACHINE_MIX, MACHINE_SPLIT, MACHINE_DUPLICATE, MACHINE_CONVERT],
        "max_moves": 84,
    },
]

NUM_LEVELS = len(LEVEL_DEFS)

TOKEN_SIZE = 8
TOKEN_GAP = 2

MACHINE_W = 9
MACHINE_H = 9
MACHINE_GAP = 2

LABEL_TOKENS_Y = 3
TOKENS_Y = 5

ARROW_Y = 14

LABEL_GOAL_Y = 17
GOAL_Y = 19

LABEL_MACHINES_Y = 30
MACHINES_Y = 33

STATUS_Y = 46

HINTS_Y = 53
HINTS_Y2 = 58

CURSOR_ROW_TOKENS = 0
CURSOR_ROW_MACHINES = 1
NUM_CURSOR_ROWS = 2


def tokens_match(current, goal):
    return sorted(current) == sorted(goal)


def apply_mix(tokens, idx1, idx2):
    if idx1 == idx2:
        return None
    if idx1 < 0 or idx1 >= len(tokens) or idx2 < 0 or idx2 >= len(tokens):
        return None
    t1 = tokens[idx1]
    t2 = tokens[idx2]
    key = (t1, t2)
    if key not in MIX_TABLE:
        return None
    result = MIX_TABLE[key]
    new_tokens = []
    lo = min(idx1, idx2)
    hi = max(idx1, idx2)
    for i, t in enumerate(tokens):
        if i == lo:
            new_tokens.append(result)
        elif i == hi:
            continue
        else:
            new_tokens.append(t)
    return new_tokens


def apply_duplicate(tokens, idx):
    if idx < 0 or idx >= len(tokens):
        return None
    if len(tokens) >= 8:
        return None
    new_tokens = list(tokens)
    new_tokens.insert(idx + 1, tokens[idx])
    return new_tokens


def apply_delete(tokens, idx):
    if idx < 0 or idx >= len(tokens):
        return None
    if len(tokens) <= 1:
        return None
    new_tokens = list(tokens)
    new_tokens.pop(idx)
    return new_tokens


def apply_split(tokens, idx):
    if idx < 0 or idx >= len(tokens):
        return None
    t = tokens[idx]
    if t not in SPLIT_TABLE:
        return None
    if len(tokens) >= 8:
        return None
    a, b = SPLIT_TABLE[t]
    new_tokens = list(tokens)
    new_tokens.pop(idx)
    new_tokens.insert(idx, b)
    new_tokens.insert(idx, a)
    return new_tokens


def apply_convert(tokens, idx):
    if idx < 0 or idx >= len(tokens):
        return None
    t = tokens[idx]
    if t not in CONVERT_TABLE:
        return None
    new_tokens = list(tokens)
    new_tokens[idx] = CONVERT_TABLE[t]
    return new_tokens


def apply_bluff(tokens, idx):
    if idx < 0 or idx >= len(tokens):
        return None
    new_tokens = list(tokens)
    new_tokens.reverse()
    new_idx = len(new_tokens) - 1 - idx
    t = new_tokens[new_idx]
    if t in REVERSE_CONVERT_TABLE:
        new_tokens[new_idx] = REVERSE_CONVERT_TABLE[t]
    elif t in SPLIT_TABLE:
        bluff_secondary = {"P": "O", "G": "P", "O": "G"}
        new_tokens[new_idx] = bluff_secondary[t]
    return new_tokens


SINGLE_TOKEN_OPS = {
    MACHINE_DUPLICATE: apply_duplicate,
    MACHINE_DELETE: apply_delete,
    MACHINE_SPLIT: apply_split,
    MACHINE_CONVERT: apply_convert,
    MACHINE_BLUFF: apply_bluff,
}


class ColorFlowHud(RenderableUserDisplay):
    def __init__(self):
        super().__init__()
        self.game: Any = None

    def _px(self, frame, x, y, color):
        if 0 <= y < frame.shape[0] and 0 <= x < frame.shape[1]:
            frame[y, x] = color

    def _rect(self, frame, x, y, w, h, color):
        for dy in range(h):
            for dx in range(w):
                self._px(frame, x + dx, y + dy, color)

    def _hline(self, frame, x, y, length, color):
        for dx in range(length):
            self._px(frame, x + dx, y, color)

    def _vline(self, frame, x, y, length, color):
        for dy in range(length):
            self._px(frame, x, y + dy, color)

    def _border(self, frame, x, y, w, h, color):
        self._hline(frame, x, y, w, color)
        self._hline(frame, x, y + h - 1, w, color)
        self._vline(frame, x, y, h, color)
        self._vline(frame, x + w - 1, y, h, color)

    def _draw_token(
        self, frame, x, y, token, selected=False, highlight=False, cursor=False
    ):
        color = TOKEN_COLORS.get(token, WHITE)
        self._rect(frame, x + 1, y + 1, TOKEN_SIZE - 2, TOKEN_SIZE - 2, color)
        if cursor:
            self._border(frame, x, y, TOKEN_SIZE, TOKEN_SIZE, WHITE)
            self._border(frame, x - 1, y - 1, TOKEN_SIZE + 2, TOKEN_SIZE + 2, YELLOW)
        elif selected:
            self._border(frame, x, y, TOKEN_SIZE, TOKEN_SIZE, WHITE)
        elif highlight:
            self._border(frame, x, y, TOKEN_SIZE, TOKEN_SIZE, YELLOW)
        else:
            self._border(frame, x, y, TOKEN_SIZE, TOKEN_SIZE, DARK_GREY)

    def _draw_token_row(
        self, frame, tokens, y, selected_idx=-1, highlight_idx=-1, cursor_idx=-1
    ):
        n = len(tokens)
        if n == 0:
            return
        total_w = n * TOKEN_SIZE + (n - 1) * TOKEN_GAP
        start_x = (GRID_W - total_w) // 2
        for i, t in enumerate(tokens):
            tx = start_x + i * (TOKEN_SIZE + TOKEN_GAP)
            sel = i == selected_idx
            hl = i == highlight_idx
            cur = i == cursor_idx
            self._draw_token(frame, tx, y, t, selected=sel, highlight=hl, cursor=cur)

    def _draw_machine_button(
        self, frame, x, y, machine_id, selected=False, cursor=False
    ):
        bg_color = MACHINE_COLORS.get(machine_id, GREY)
        icon = MACHINE_ICONS.get(machine_id, None)
        self._rect(frame, x + 1, y + 1, MACHINE_W - 2, MACHINE_H - 2, bg_color)
        if icon is not None:
            ix = x + 2
            iy = y + 2
            for row in range(5):
                for col in range(5):
                    if icon[row][col] == K:
                        self._px(frame, ix + col, iy + row, BLACK)
        if cursor:
            self._border(frame, x, y, MACHINE_W, MACHINE_H, WHITE)
            self._border(frame, x - 1, y - 1, MACHINE_W + 2, MACHINE_H + 2, YELLOW)
        elif selected:
            self._border(frame, x, y, MACHINE_W, MACHINE_H, WHITE)
            self._border(frame, x - 1, y - 1, MACHINE_W + 2, MACHINE_H + 2, GREEN)
        else:
            self._border(frame, x, y, MACHINE_W, MACHINE_H, DARK_GREY)

    def _draw_machines(
        self, frame, machines, y, selected_machine=-1, cursor_machine_idx=-1
    ):
        n = len(machines)
        if n == 0:
            return
        total_w = n * MACHINE_W + (n - 1) * MACHINE_GAP
        start_x = (GRID_W - total_w) // 2
        for i, m in enumerate(machines):
            mx = start_x + i * (MACHINE_W + MACHINE_GAP)
            sel = m == selected_machine
            cur = i == cursor_machine_idx
            self._draw_machine_button(frame, mx, y, m, selected=sel, cursor=cur)

    def _draw_label_bar(self, frame, y, color):
        self._hline(frame, 2, y, GRID_W - 4, color)

    def _draw_arrow_down(self, frame, y):
        cx = GRID_W // 2
        self._vline(frame, cx, y, 2, LIGHT_GREY)
        self._px(frame, cx - 1, y + 1, LIGHT_GREY)
        self._px(frame, cx + 1, y + 1, LIGHT_GREY)

    def _draw_match_indicator(self, frame, y, match):
        cx = GRID_W // 2
        if match:
            self._rect(frame, cx - 2, y, 5, 3, GREEN)
            self._px(frame, cx, y + 1, BLACK)
        else:
            self._rect(frame, cx - 1, y, 3, 3, ORANGE)

    def _draw_mix_prompt(self, frame):
        g = self.game
        if g.selected_machine == MACHINE_MIX and g.mix_first_idx >= 0:
            n = len(g.tokens)
            total_w = n * TOKEN_SIZE + (n - 1) * TOKEN_GAP
            start_x = (GRID_W - total_w) // 2
            idx = g.mix_first_idx
            if 0 <= idx < n:
                tx = start_x + idx * (TOKEN_SIZE + TOKEN_GAP)
                ty = TOKENS_Y
                self._border(
                    frame, tx - 1, ty - 1, TOKEN_SIZE + 2, TOKEN_SIZE + 2, YELLOW
                )

    def _draw_win(self, frame):
        g = self.game
        self._draw_token_row(frame, g.goal, TOKENS_Y)

    def _draw_lives(self, frame, lives):
        y = 2
        for i in range(3):
            x = 61 - (i * 5)
            c = WHITE if i < lives else DARK_GREY
            self._rect(frame, x - 2, y, 3, 3, c)

    def render_interface(self, frame):
        if self.game is None:
            return frame

        g = self.game
        height, width = frame.shape[:2]

        for y in range(height):
            for x in range(width):
                frame[y, x] = BLACK

        hl_idx = -1
        sel_idx = -1
        cur_token_idx = -1
        if g.selected_machine == MACHINE_MIX and g.mix_first_idx >= 0:
            sel_idx = g.mix_first_idx
        if g.cursor_row == CURSOR_ROW_TOKENS:
            cur_token_idx = g.cursor_col

        self._draw_token_row(
            frame,
            g.tokens,
            TOKENS_Y,
            selected_idx=sel_idx,
            highlight_idx=hl_idx,
            cursor_idx=cur_token_idx,
        )

        if g.flash_timer > 0:
            flash_color = GREEN if g.flash_success else RED
            self._hline(frame, 2, TOKENS_Y - 1, GRID_W - 4, flash_color)
            self._hline(frame, 2, TOKENS_Y + TOKEN_SIZE, GRID_W - 4, flash_color)

        self._draw_arrow_down(frame, ARROW_Y)

        self._draw_label_bar(frame, LABEL_GOAL_Y, GREEN)
        self._draw_token_row(frame, g.goal, GOAL_Y)

        match = tokens_match(g.tokens, g.goal)
        self._draw_match_indicator(frame, 28, match)

        self._draw_label_bar(frame, LABEL_MACHINES_Y, MAGENTA)

        ldef = LEVEL_DEFS[g.lidx]
        cur_machine_idx = -1
        if g.cursor_row == CURSOR_ROW_MACHINES:
            cur_machine_idx = g.cursor_col

        self._draw_machines(
            frame,
            ldef["machines"],
            MACHINES_Y,
            g.selected_machine,
            cursor_machine_idx=cur_machine_idx,
        )

        if g.selected_machine >= 0:
            self._draw_machine_hint(frame, g.selected_machine)

        self._draw_mix_prompt(frame)

        self._draw_move_dots(frame)

        self._draw_mix_recipes(frame)

        if g.won:
            self._draw_win(frame)

        if not g.won:
            self._draw_lives(frame, g.lives)

        return frame

    def _draw_move_dots(self, frame):
        g = self.game
        total = g.max_moves
        rem = max(0, g.rem_moves)

        bar_x = 4
        bar_y = 1
        bar_w = GRID_W - 8
        bar_h = 2

        self._rect(frame, bar_x, bar_y, bar_w, bar_h, DARK_GREY)

        fill_w = int((rem / total) * bar_w) if total > 0 else 0

        if fill_w > 0:
            pct = rem / total if total > 0 else 0
            if pct > 0.5:
                c = GREEN
            elif pct > 0.25:
                c = YELLOW
            else:
                c = RED
            self._rect(frame, bar_x, bar_y, fill_w, bar_h, c)

    def _draw_mix_recipes(self, frame):
        y = HINTS_Y
        sq = 3

        recipes = [
            (RED, BLUE, PURPLE),
            (BLUE, YELLOW, GREEN),
            (RED, YELLOW, ORANGE),
        ]

        recipe_w = 13
        total_w = len(recipes) * recipe_w + (len(recipes) - 1) * 6
        start_x = (GRID_W - total_w) // 2

        for ri, (c1, c2, cr) in enumerate(recipes):
            rx = start_x + ri * (recipe_w + 6)
            self._rect(frame, rx, y, sq, sq, c1)
            self._rect(frame, rx + sq, y, sq, sq, c2)
            ax = rx + sq * 2 + 1
            self._px(frame, ax, y + 1, LIGHT_GREY)
            self._px(frame, ax + 1, y + 1, LIGHT_GREY)
            self._px(frame, ax + 1, y, LIGHT_GREY)
            self._px(frame, ax + 1, y + 2, LIGHT_GREY)
            self._rect(frame, ax + 3, y, sq, sq, cr)

        self._hline(frame, 2, HINTS_Y - 2, GRID_W - 4, DARK_GREY)

        y2 = HINTS_Y2

        split_recipes = [
            (PURPLE, RED, BLUE),
            (GREEN, BLUE, YELLOW),
            (ORANGE, RED, YELLOW),
        ]

        split_w = 13
        total_sw = len(split_recipes) * split_w + (len(split_recipes) - 1) * 6
        start_sx = (GRID_W - total_sw) // 2

        for ri, (src, d1, d2) in enumerate(split_recipes):
            sx = start_sx + ri * (split_w + 6)
            self._rect(frame, sx, y2, sq, sq, src)
            ax = sx + sq + 1
            self._px(frame, ax, y2 + 1, LIGHT_GREY)
            self._px(frame, ax + 1, y2 + 1, LIGHT_GREY)
            self._px(frame, ax + 1, y2, LIGHT_GREY)
            self._px(frame, ax + 1, y2 + 2, LIGHT_GREY)
            self._rect(frame, ax + 3, y2, sq, sq, d1)
            self._rect(frame, ax + 3 + sq, y2, sq, sq, d2)

    def _draw_machine_hint(self, frame, machine_id):
        y = STATUS_Y
        cx = GRID_W // 2

        if machine_id == MACHINE_MIX:
            self._rect(frame, cx - 12, y, 4, 4, RED)
            self._rect(frame, cx - 6, y, 4, 4, BLUE)
            self._px(frame, cx + 0, y + 1, LIGHT_GREY)
            self._px(frame, cx + 1, y + 1, LIGHT_GREY)
            self._px(frame, cx + 1, y, LIGHT_GREY)
            self._px(frame, cx + 1, y + 2, LIGHT_GREY)
            self._rect(frame, cx + 4, y, 4, 4, PURPLE)

        elif machine_id == MACHINE_DUPLICATE:
            self._rect(frame, cx - 8, y, 4, 4, LIGHT_BLUE)
            self._px(frame, cx - 2, y + 1, LIGHT_GREY)
            self._px(frame, cx - 1, y + 1, LIGHT_GREY)
            self._rect(frame, cx + 2, y, 4, 4, LIGHT_BLUE)
            self._rect(frame, cx + 7, y, 4, 4, LIGHT_BLUE)

        elif machine_id == MACHINE_DELETE:
            self._rect(frame, cx - 6, y, 4, 4, MAROON)
            self._px(frame, cx, y + 1, LIGHT_GREY)
            self._px(frame, cx + 1, y + 1, LIGHT_GREY)
            self._px(frame, cx + 4, y, RED)
            self._px(frame, cx + 6, y, RED)
            self._px(frame, cx + 5, y + 1, RED)
            self._px(frame, cx + 4, y + 2, RED)
            self._px(frame, cx + 6, y + 2, RED)

        elif machine_id == MACHINE_SPLIT:
            self._rect(frame, cx - 10, y, 4, 4, PURPLE)
            self._px(frame, cx - 4, y + 1, LIGHT_GREY)
            self._px(frame, cx - 3, y + 1, LIGHT_GREY)
            self._rect(frame, cx, y, 4, 4, RED)
            self._rect(frame, cx + 5, y, 4, 4, BLUE)

        elif machine_id == MACHINE_CONVERT:
            self._rect(frame, cx - 10, y, 4, 4, RED)
            self._px(frame, cx - 4, y + 1, LIGHT_GREY)
            self._px(frame, cx - 3, y + 1, LIGHT_GREY)
            self._rect(frame, cx, y, 4, 4, BLUE)
            self._px(frame, cx + 6, y + 1, LIGHT_GREY)
            self._px(frame, cx + 7, y + 1, LIGHT_GREY)
            self._rect(frame, cx + 10, y, 4, 4, YELLOW)

        elif machine_id == MACHINE_BLUFF:
            self._rect(frame, cx - 10, y, 4, 4, BLUE)
            self._rect(frame, cx - 5, y, 4, 4, RED)
            self._px(frame, cx + 1, y, LIGHT_GREY)
            self._px(frame, cx + 2, y + 1, LIGHT_GREY)
            self._px(frame, cx + 1, y + 2, LIGHT_GREY)
            self._rect(frame, cx + 5, y, 4, 4, YELLOW)
            self._rect(frame, cx + 10, y, 4, 4, BLUE)


def _hit_test_tokens(tokens, click_x, click_y, base_y):
    n = len(tokens)
    if n == 0:
        return -1
    total_w = n * TOKEN_SIZE + (n - 1) * TOKEN_GAP
    start_x = (GRID_W - total_w) // 2

    for i in range(n):
        tx = start_x + i * (TOKEN_SIZE + TOKEN_GAP)
        ty = base_y
        if tx <= click_x < tx + TOKEN_SIZE and ty <= click_y < ty + TOKEN_SIZE:
            return i
    return -1


def _hit_test_machines(machines, click_x, click_y, base_y):
    n = len(machines)
    if n == 0:
        return -1
    total_w = n * MACHINE_W + (n - 1) * MACHINE_GAP
    start_x = (GRID_W - total_w) // 2

    for i, m in enumerate(machines):
        mx = start_x + i * (MACHINE_W + MACHINE_GAP)
        my = base_y
        if mx <= click_x < mx + MACHINE_W and my <= click_y < my + MACHINE_H:
            return m
    return -1


sprites = {
    "marker": Sprite(
        pixels=[[-1]],
        name="marker",
        visible=False,
        collidable=False,
        layer=0,
        tags=["marker"],
    )
}


def _build_levels():
    out = []
    names = ["Level 1", "Level 2", "Level 3", "Level 4"]
    for i in range(NUM_LEVELS):
        out.append(
            Level(
                sprites=[sprites["marker"].clone().set_position(0, 0)],
                grid_size=(64, 64),
                data={"idx": i},
                name=names[i],
            )
        )
    return out


levels = _build_levels()


class Eg04(ARCBaseGame):
    def __init__(self, seed: int = 0):
        self.hud = ColorFlowHud()
        self.hud.game = self

        self._rng = random.Random(seed)
        self._seed = seed

        self.lidx = 0
        self.tokens = []
        self.init_tokens = []
        self.goal = []
        self.max_moves = 10
        self.rem_moves = 10
        self.selected_machine = -1
        self.mix_first_idx = -1
        self.flash_timer = 0
        self.flash_success = False
        self.won = False
        self.lives = 3
        self.game_over = False
        self.game_over_delay = 0
        self.died_at_level = -1
        self.reset_press_count = 0
        self.play_since_reset = False
        self.turn = 0

        self.cursor_row = CURSOR_ROW_MACHINES
        self.cursor_col = 0

        self._history = []
        self._game_over = False

        super().__init__(
            "eg04",
            levels,
            Camera(0, 0, 64, 64, BLACK, BLACK, [self.hud]),
            available_actions=[0, 1, 2, 3, 4, 5, 6, 7],
        )

    def on_set_level(self, level):
        idx = self.current_level.get_data("idx")
        if idx is None:
            idx = 0
        if self.died_at_level >= 0:
            idx = self.died_at_level
            self.died_at_level = -1
        self._setup_level(idx)

        self._history = []
        self._game_over = False

    def _setup_level(self, idx, reset_lives=True):
        if idx < 0 or idx >= NUM_LEVELS:
            idx = 0
        self.lidx = idx
        ldef = LEVEL_DEFS[idx]
        shuffled = list(ldef["start"])
        self._rng.shuffle(shuffled)
        self.init_tokens = list(shuffled)
        self.tokens = list(shuffled)
        self.goal = list(ldef["goal"])
        self.max_moves = ldef["max_moves"]
        self.rem_moves = ldef["max_moves"]
        self.selected_machine = -1
        self.mix_first_idx = -1
        self.flash_timer = 0
        self.flash_success = False
        self.won = False
        if reset_lives:
            self.lives = 3
        self.game_over = False
        self.game_over_delay = 0
        self.cursor_row = CURSOR_ROW_MACHINES
        self.cursor_col = 0
        self.turn = 0

    def _flash(self, success):
        self.flash_timer = 4
        self.flash_success = success

    def _consume_move(self):
        self.rem_moves = max(0, self.rem_moves - 1)

    def _check_win(self):
        return tokens_match(self.tokens, self.goal)

    def get_actions(self):
        if self.game_over or self.won:
            return []
        return [0, 1, 2, 3, 4, 5, 6, 7]

    def _get_state(self):
        return {
            "text": f"Tokens: {self.tokens} Goal: {self.goal}",
            "image": None,
            "actions": self.get_actions(),
            "turn": self.turn,
            "metadata": {
                "level": self.lidx + 1,
                "moves_remaining": self.rem_moves,
                "lives": self.lives,
                "won": self.won,
                "game_over": self.game_over,
            },
        }

    def _save_state(self):
        self._history.append(
            {
                "tokens": list(self.tokens),
                "init_tokens": list(self.init_tokens),
                "goal": list(self.goal),
                "selected_machine": self.selected_machine,
                "mix_first_idx": self.mix_first_idx,
                "cursor_row": self.cursor_row,
                "cursor_col": self.cursor_col,
                "flash_timer": self.flash_timer,
                "flash_success": self.flash_success,
                "won": self.won,
                "game_over": self.game_over,
                "play_since_reset": self.play_since_reset,
            }
        )

    def _undo(self):
        if not self._history:
            return
        snap = self._history.pop()
        self.tokens = list(snap["tokens"])
        self.init_tokens = list(snap["init_tokens"])
        self.goal = list(snap["goal"])
        self.selected_machine = snap["selected_machine"]
        self.mix_first_idx = snap["mix_first_idx"]
        self.cursor_row = snap["cursor_row"]
        self.cursor_col = snap["cursor_col"]
        self.flash_timer = snap["flash_timer"]
        self.flash_success = snap["flash_success"]
        self.won = snap["won"]
        self.game_over = snap["game_over"]
        self.play_since_reset = snap["play_since_reset"]

    def _reset_current_level(self):
        self._setup_level(self.lidx, reset_lives=False)
        self._history = []

    def full_reset(self):
        self._game_over = False
        self.game_over = False
        self.lives = 3
        self.died_at_level = -1
        self.reset_press_count = 0
        self.play_since_reset = False
        self._rng = random.Random(self._seed)
        self._current_level_index = 0
        self.on_set_level(levels[0])

    def _handle_death(self):
        self.lives -= 1
        if self.lives <= 0:
            self._game_over = True
            self.game_over = True
            self.lose()
            self.lives = 3
            return True
        else:
            return False

    def _handle_machine_click(self, machine_id):
        if machine_id == self.selected_machine:
            self.selected_machine = -1
            self.mix_first_idx = -1
        else:
            self.selected_machine = machine_id
            self.mix_first_idx = -1

    def _handle_token_click(self, token_idx):
        if self.selected_machine < 0:
            return
        if self.rem_moves <= 0:
            return
        if self.won:
            return

        machine = self.selected_machine

        if machine == MACHINE_MIX:
            self._handle_mix_click(token_idx)
        elif machine in SINGLE_TOKEN_OPS:
            apply_fn = SINGLE_TOKEN_OPS[machine]
            result = apply_fn(self.tokens, token_idx)
            if result is not None:
                self.tokens = result
                self._consume_move()
                self._flash(True)
                self.selected_machine = -1
            else:
                self._flash(False)

    def _handle_mix_click(self, token_idx):
        if self.mix_first_idx < 0:
            self.mix_first_idx = token_idx
        else:
            if token_idx == self.mix_first_idx:
                self.mix_first_idx = -1
                return

            result = apply_mix(self.tokens, self.mix_first_idx, token_idx)
            if result is not None:
                self.tokens = result
                self._consume_move()
                self._flash(True)
                self.selected_machine = -1
                self.mix_first_idx = -1
            else:
                self._flash(False)
                self.mix_first_idx = -1

    def _get_row_length(self, row):
        if row == CURSOR_ROW_TOKENS:
            return len(self.tokens)
        elif row == CURSOR_ROW_MACHINES:
            ldef = LEVEL_DEFS[self.lidx]
            return len(ldef["machines"])
        return 0

    def _clamp_cursor_col(self):
        max_col = self._get_row_length(self.cursor_row) - 1
        if max_col < 0:
            max_col = 0
        if self.cursor_col > max_col:
            self.cursor_col = max_col
        if self.cursor_col < 0:
            self.cursor_col = 0

    def _handle_cursor_up(self):
        if self.cursor_row > 0:
            self.cursor_row -= 1
            self._clamp_cursor_col()
            if self.rem_moves > 0:
                self._consume_move()

    def _handle_cursor_down(self):
        if self.cursor_row < NUM_CURSOR_ROWS - 1:
            self.cursor_row += 1
            self._clamp_cursor_col()
            if self.rem_moves > 0:
                self._consume_move()

    def _handle_cursor_left(self):
        if self.cursor_col > 0:
            self.cursor_col -= 1
            if self.rem_moves > 0:
                self._consume_move()

    def _handle_cursor_right(self):
        max_col = self._get_row_length(self.cursor_row) - 1
        if self.cursor_col < max_col:
            self.cursor_col += 1
            if self.rem_moves > 0:
                self._consume_move()

    def _handle_space(self):
        if self.won:
            return
        if self.cursor_row == CURSOR_ROW_MACHINES:
            ldef = LEVEL_DEFS[self.lidx]
            machines = ldef["machines"]
            if 0 <= self.cursor_col < len(machines):
                machine_id = machines[self.cursor_col]
                self._handle_machine_click(machine_id)
        elif self.cursor_row == CURSOR_ROW_TOKENS:
            if 0 <= self.cursor_col < len(self.tokens):
                self._handle_token_click(self.cursor_col)
                if self._check_win():
                    self.won = True
                    self._flash(True)

    def _handle_click_action(self):
        cx = self.action.data.get("x", 0)
        cy = self.action.data.get("y", 0)
        coords = self.camera.display_to_grid(cx, cy)
        if coords:
            grid_x, grid_y = coords

            ldef = LEVEL_DEFS[self.lidx]

            machine_hit = _hit_test_machines(
                ldef["machines"], grid_x, grid_y, MACHINES_Y
            )
            token_hit = _hit_test_tokens(self.tokens, grid_x, grid_y, TOKENS_Y)

            if machine_hit >= 0:
                self.cursor_row = CURSOR_ROW_MACHINES
                self.cursor_col = ldef["machines"].index(machine_hit)
                self._handle_machine_click(machine_hit)
            elif token_hit >= 0:
                self.cursor_row = CURSOR_ROW_TOKENS
                self.cursor_col = token_hit
                self._handle_token_click(token_hit)

                if self._check_win():
                    self.won = True
                    self._flash(True)
            else:
                self.selected_machine = -1
                self.mix_first_idx = -1

    def _dispatch_action(self, act):
        if act == 1:
            self._handle_cursor_up()
        elif act == 2:
            self._handle_cursor_down()
        elif act == 3:
            self._handle_cursor_left()
        elif act == 4:
            self._handle_cursor_right()
        elif act == 5:
            self._handle_space()
        elif act == 6:
            self._handle_click_action()

    def _check_moves_exhausted(self):
        if self.rem_moves <= 0 and not self.won:
            game_over = self._handle_death()
            if game_over:
                return
            else:
                self._reset_current_level()

    def step(self) -> None:
        action_id = self.action.id.value

        if action_id == 0:
            if not self.play_since_reset:
                self.reset_press_count += 1
            else:
                self.reset_press_count = 1

            if self.reset_press_count >= 2:
                self._setup_level(0)
                self.reset_press_count = 0
                self.died_at_level = -1
            elif self.died_at_level >= 0:
                self._setup_level(self.died_at_level)
            else:
                self._setup_level(self.lidx)

            self.play_since_reset = False
            self._clamp_cursor_col()
            self._history = []
            self.complete_action()
            return

        if action_id == 7:
            self._undo()
            self.rem_moves -= 1
            self._check_moves_exhausted()
            self.complete_action()
            return

        if self.game_over:
            self.died_at_level = self.lidx
            self.complete_action()
            self.lose()
            return
        if self.won:
            self.complete_action()
            self.next_level()
            return

        if self.flash_timer > 0:
            self.flash_timer -= 1

        self.turn += 1
        self.play_since_reset = True
        self.reset_press_count = 0
        self.died_at_level = -1

        self._save_state()

        self._dispatch_action(action_id)
        self._check_moves_exhausted()
        self._clamp_cursor_col()

        if self.won:
            self.complete_action()
            self.next_level()
            return

        self.complete_action()


_COLOR_CHAR = {
    RED: "R",
    BLUE: "B",
    YELLOW: "Y",
    PURPLE: "P",
    GREEN: "G",
    ORANGE: "O",
    WHITE: "W",
    MAGENTA: "M",
    MAROON: "N",
    PINK: "K",
}

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
        [163, 86, 214],
    ],
    dtype=np.uint8,
)


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

    _VALID_ACTIONS = list(_ACTION_MAP.keys())

    def __init__(self, seed: int = 0) -> None:
        self._engine: Any = Eg04(seed=seed)
        self._done = False
        self._game_won = False
        self._last_action_was_reset = False
        self._total_turns = 0
        self._total_levels = len(levels)

    @staticmethod
    def _frame_to_png(frame: np.ndarray) -> bytes:
        h, w = frame.shape[:2]
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for idx in range(len(ARC_PALETTE)):
            mask = frame == idx
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            rgb[mask] = ARC_PALETTE[idx]

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

    def _render_frame(self) -> np.ndarray:
        e = self._engine
        return e.camera.render(e.current_level.get_sprites())

    def _build_text_observation(self) -> str:
        e = self._engine
        ldef = LEVEL_DEFS[e.lidx]

        token_strs = [
            f"{t}({_COLOR_CHAR.get(TOKEN_COLORS.get(t, 0), '?')})" if False else t
            for t in e.tokens
        ]
        goal_strs = [t for t in e.goal]

        machine_names = [MACHINE_NAMES[m] for m in ldef["machines"]]
        selected_name = MACHINE_NAMES.get(e.selected_machine, "none")

        header = (
            f"Level {e.lidx + 1}/{NUM_LEVELS}"
            f" | Moves: {e.rem_moves}/{e.max_moves}"
            f" | Lives: {e.lives}"
            f" | Won: {e.won}"
        )
        tokens_line = f"Tokens: [{', '.join(token_strs)}]"
        goal_line = f"Goal: [{', '.join(goal_strs)}]"
        machine_line = (
            f"Selected: {selected_name} | Available: [{', '.join(machine_names)}]"
        )

        cursor_row_name = "tokens" if e.cursor_row == CURSOR_ROW_TOKENS else "machines"
        cursor_line = f"Cursor: row={cursor_row_name} col={e.cursor_col}"

        if e.selected_machine == MACHINE_MIX and e.mix_first_idx >= 0:
            mix_line = f"Mix first token: index {e.mix_first_idx}"
        else:
            mix_line = ""

        rules = "Select a machine, then apply it to a token. Match tokens to goal."

        parts = [header, rules, tokens_line, goal_line, machine_line, cursor_line]
        if mix_line:
            parts.append(mix_line)
        return "\n".join(parts)

    def _build_game_state(self, done: bool = False) -> GameState:
        e = self._engine
        frame = self._render_frame()
        image_bytes = self._frame_to_png(frame)

        valid_actions = list(self._VALID_ACTIONS) if not done else None

        return GameState(
            text_observation=self._build_text_observation(),
            image_observation=image_bytes,
            valid_actions=valid_actions,
            turn=self._total_turns,
            metadata={
                "total_levels": len(e._levels),
                "level_index": e._current_level_index,
                "levels_completed": getattr(e, "_score", 0),
                "game_over": getattr(getattr(e, "_state", None), "name", "")
                == "GAME_OVER",
                "done": done,
                "info": {},
                "level": e.lidx + 1,
                "moves_remaining": e.rem_moves,
                "max_moves": e.max_moves,
                "lives": e.lives,
                "won": e.won,
                "tokens": list(e.tokens),
                "goal": list(e.goal),
                "selected_machine": e.selected_machine,
                "cursor_row": e.cursor_row,
                "cursor_col": e.cursor_col,
            },
        )

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

    def is_done(self) -> bool:
        return self._done

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

        if action not in self._ACTION_MAP:
            base_action = action
            click_data: Dict[str, Any] = {}
            if action.startswith("click:"):
                parts = action.split(":")
                coords = parts[1].split(",")
                click_data = {"x": int(coords[0]), "y": int(coords[1])}
                base_action = "click"
            elif action.startswith("click "):
                remainder = action[6:].strip()
                if "," in remainder:
                    coords = remainder.split(",")
                else:
                    coords = remainder.split()
                click_data = {"x": int(coords[0]), "y": int(coords[1])}
                base_action = "click"
            else:
                raise ValueError(
                    f"Invalid action '{action}'. Must be one of {list(self._ACTION_MAP.keys())}"
                )
            action = base_action
        else:
            click_data = {}

        self._last_action_was_reset = False
        self._total_turns += 1

        game_action = self._ACTION_MAP[action]
        info: Dict[str, Any] = {"action": action}

        level_before = e.level_index

        action_input = (
            ActionInput(id=game_action, data=click_data)
            if click_data
            else ActionInput(id=game_action)
        )
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

    env = ArcGameEnv(seed=42, render_mode="rgb_array")
    check_env(env.unwrapped, skip_render_check=True)

    obs, info = env.reset()

    mask = env.action_mask()
    valid_indices = np.where(mask == 1)[0]
    if len(valid_indices) > 0:
        obs, reward, term, trunc, info = env.step(valid_indices[0])

    env.close()
