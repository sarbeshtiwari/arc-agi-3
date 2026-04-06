import io
import random
import struct
import zlib
from copy import deepcopy
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

JAR_NORMAL = 0
JAR_MIXER = 1
JAR_FILTER = 2

MIX_TABLE = {
    (BLUE, YELLOW): GREEN,
    (YELLOW, BLUE): GREEN,
    (RED, BLUE): PURPLE,
    (BLUE, RED): PURPLE,
    (RED, YELLOW): ORANGE,
    (YELLOW, RED): ORANGE,
}

LEVEL_MIXING_HINTS = {
    2: [(BLUE, YELLOW, GREEN)],
    3: [(RED, BLUE, PURPLE), (RED, YELLOW, ORANGE)],
}

GRID_W = 64
GRID_H = 64
JAR_W = 8
JAR_INNER = 6
UNIT_H = 4
HUD_Y = 1


def _make_jar(cap, amount=0, color=None, jar_type=JAR_NORMAL, filter_color=None):
    return {
        "cap": cap,
        "amount": amount,
        "color": color,
        "type": jar_type,
        "filter_color": filter_color,
    }


LEVEL_CONFIGS = [
    {
        "jars": [
            _make_jar(4, 4, BLUE),
            _make_jar(3, 3, BLUE),
            _make_jar(5, 0, None),
        ],
        "targets": [(2, 2, BLUE)],
        "min_moves": 3,
        "max_moves": 72,
    },
    {
        "jars": [
            _make_jar(5, 5, BLUE),
            _make_jar(3, 0, None),
            _make_jar(2, 0, None),
            _make_jar(6, 0, None),
        ],
        "targets": [(3, 4, BLUE)],
        "min_moves": 4,
        "max_moves": 96,
    },
    {
        "jars": [
            _make_jar(4, 4, BLUE),
            _make_jar(3, 3, YELLOW),
            _make_jar(3, 0, None, JAR_MIXER),
            _make_jar(3, 0, None),
        ],
        "targets": [(3, 3, GREEN)],
        "min_moves": 5,
        "max_moves": 120,
    },
    {
        "jars": [
            _make_jar(5, 5, RED),
            _make_jar(5, 5, BLUE),
            _make_jar(3, 3, YELLOW),
            _make_jar(4, 0, None, JAR_MIXER),
            _make_jar(3, 0, None, JAR_FILTER, PURPLE),
            _make_jar(3, 0, None, JAR_FILTER, ORANGE),
        ],
        "targets": [(4, 3, PURPLE), (5, 3, ORANGE)],
        "min_moves": 9,
        "max_moves": 216,
    },
]

NUM_LEVELS = len(LEVEL_CONFIGS)


def try_pour(jars, src, dst):
    s = jars[src]
    d = jars[dst]

    if s["amount"] == 0:
        return None

    room = d["cap"] - d["amount"]
    if room == 0:
        return None

    transfer = min(s["amount"], room)
    src_color = s["color"]
    dst_color = d["color"]
    dst_type = d.get("type", JAR_NORMAL)

    if d["amount"] == 0:
        if dst_type == JAR_FILTER and src_color != d.get("filter_color"):
            return None
        new_color = src_color

    elif dst_color == src_color:
        new_color = dst_color

    else:
        if dst_type == JAR_NORMAL:
            return None
        elif dst_type == JAR_MIXER:
            mixed = MIX_TABLE.get((dst_color, src_color))
            if mixed is None:
                return None
            new_color = mixed
        elif dst_type == JAR_FILTER:
            if src_color != d.get("filter_color"):
                return None
            new_color = dst_color
        else:
            return None

    new_jars = []
    for i, j in enumerate(jars):
        new_jar = dict(j)
        if i == src:
            new_jar["amount"] = s["amount"] - transfer
            if new_jar["amount"] == 0:
                new_jar["color"] = None
        elif i == dst:
            new_jar["amount"] = d["amount"] + transfer
            new_jar["color"] = new_color
        new_jars.append(new_jar)
    return new_jars


def check_targets(jars, targets):
    for idx, amt, col in targets:
        j = jars[idx]
        if j["amount"] != amt or j["color"] != col:
            return False
    return True


def _jar_layout(num_jars, max_cap):
    gap = 4 if num_jars <= 4 else 2
    total_w = num_jars * JAR_W + (num_jars - 1) * gap
    start_x = (GRID_W - total_w) // 2
    jar_h = max_cap * UNIT_H + 2
    content_h = jar_h + 5
    top_margin = 6
    bottom_margin = 6
    avail = GRID_H - top_margin - bottom_margin
    jar_top = top_margin + (avail - content_h) // 2
    if jar_top < 6:
        jar_top = 6
    positions = []
    for i in range(num_jars):
        x = start_x + i * (JAR_W + gap)
        positions.append((x, jar_top))
    return positions, jar_h, gap


class LiquidHud(RenderableUserDisplay):
    def __init__(self):
        super().__init__()
        self.game: Any = None

    def _set_pixel(self, frame, x, y, col):
        if 0 <= x < GRID_W and 0 <= y < GRID_H:
            frame[y, x] = col

    def _hline(self, frame, x, y, length, col):
        for dx in range(length):
            self._set_pixel(frame, x + dx, y, col)

    def _vline(self, frame, x, y, length, col):
        for dy in range(length):
            self._set_pixel(frame, x, y + dy, col)

    def _rect(self, frame, x, y, w, h, col):
        for dy in range(h):
            for dx in range(w):
                self._set_pixel(frame, x + dx, y + dy, col)

    def _draw_jar_body(self, frame, jx, jy, jar, jar_h):
        jar_type = jar.get("type", JAR_NORMAL)
        filter_color = jar.get("filter_color")

        if jar_type == JAR_MIXER:
            border_color = MAGENTA
        elif jar_type == JAR_FILTER and filter_color is not None:
            border_color = filter_color
        else:
            border_color = GREY

        self._vline(frame, jx, jy, jar_h, border_color)
        self._vline(frame, jx + JAR_W - 1, jy, jar_h, border_color)
        self._hline(frame, jx, jy + jar_h - 1, JAR_W, border_color)

        if jar_type == JAR_FILTER and filter_color is not None:
            for dx in range(JAR_INNER):
                self._set_pixel(frame, jx + 1 + dx, jy, filter_color)

    def _draw_jar_fill(self, frame, jx, jy, jar, max_cap, jar_h):
        cap = jar["cap"]
        amt = jar["amount"]
        color = jar["color"]

        inner_top = jy + 1
        inner_h = jar_h - 2
        cap_pixels = cap * UNIT_H
        empty_offset = inner_h - cap_pixels

        for row in range(inner_h):
            ry = inner_top + row
            unit_from_top = row - empty_offset
            unit_idx = cap - 1 - (unit_from_top // UNIT_H)

            if unit_from_top < 0:
                for dx in range(JAR_INNER):
                    self._set_pixel(frame, jx + 1 + dx, ry, BLACK)
            elif unit_idx < amt and color is not None:
                for dx in range(JAR_INNER):
                    self._set_pixel(frame, jx + 1 + dx, ry, color)
            else:
                for dx in range(JAR_INNER):
                    self._set_pixel(frame, jx + 1 + dx, ry, DARK_GREY)

    def _draw_target_marker(self, frame, jx, jy, jar_h, target_amount, cap):
        if 0 < target_amount < cap:
            inner_top = jy + 1
            inner_h = jar_h - 2
            marker_y = inner_top + inner_h - target_amount * UNIT_H - 1
            if inner_top <= marker_y < inner_top + inner_h:
                for dx in range(JAR_INNER):
                    if dx % 2 == 0:
                        self._set_pixel(frame, jx + 1 + dx, marker_y, LIGHT_BLUE)

    def _draw_cursor_highlight(self, frame, jx, jy, jar_h):
        self._hline(frame, jx - 1, jy - 1, JAR_W + 2, YELLOW)
        self._hline(frame, jx - 1, jy + jar_h, JAR_W + 2, YELLOW)
        self._vline(frame, jx - 1, jy - 1, jar_h + 2, YELLOW)
        self._vline(frame, jx + JAR_W, jy - 1, jar_h + 2, YELLOW)
        for cx, cy in [
            (jx - 1, jy - 1),
            (jx + JAR_W, jy - 1),
            (jx - 1, jy + jar_h),
            (jx + JAR_W, jy + jar_h),
        ]:
            self._set_pixel(frame, cx, cy, ORANGE)

    def _draw_selection_highlight(self, frame, jx, jy, jar_h):
        self._hline(frame, jx, jy - 1, JAR_W, GREEN)
        self._hline(frame, jx, jy + jar_h, JAR_W, GREEN)
        self._vline(frame, jx, jy, jar_h, GREEN)
        self._vline(frame, jx + JAR_W - 1, jy, jar_h, GREEN)

    def _draw_jar(
        self, frame, jx, jy, jar, max_cap, is_cursor, is_selected, target_amount
    ):
        jar_h = max_cap * UNIT_H + 2

        self._draw_jar_body(frame, jx, jy, jar, jar_h)
        self._draw_jar_fill(frame, jx, jy, jar, max_cap, jar_h)
        self._draw_target_marker(frame, jx, jy, jar_h, target_amount, jar["cap"])

        if is_cursor:
            self._draw_cursor_highlight(frame, jx, jy, jar_h)

        if is_selected:
            self._draw_selection_highlight(frame, jx, jy, jar_h)

    def _draw_level_dots(self, frame, current, total):
        for i in range(total):
            x = 1 + i * 4
            color = GREEN if i < current else GREY
            self._rect(frame, x, HUD_Y, 2, 2, color)

    def _draw_lives(self, frame, lives):
        for i in range(3):
            x = GRID_W - 4 - i * 4
            color = RED if i < lives else DARK_GREY
            self._rect(frame, x, HUD_Y, 2, 2, color)

    def _draw_move_bar(self, frame, remaining, total):
        bar_x = 24
        bar_w = 16
        bar_h = 2
        self._rect(frame, bar_x, HUD_Y, bar_w, bar_h, DARK_GREY)
        if total > 0:
            filled = max(0, bar_w * remaining // total)
            if filled > 0:
                if remaining > total * 0.5:
                    fill_color = GREEN
                elif remaining > total * 0.25:
                    fill_color = YELLOW
                else:
                    fill_color = RED
                self._rect(frame, bar_x, HUD_Y, filled, bar_h, fill_color)

    def _draw_mixing_hints(self, frame, level_idx):
        hints = LEVEL_MIXING_HINTS.get(level_idx)
        if not hints:
            return

        sq = 3
        hint_w = 15
        row_h = 6

        num = len(hints)
        start_y = 54 - ((num - 1) * row_h) // 2

        for i, (c1, c2, result) in enumerate(hints):
            y = start_y + i * row_h
            x = (GRID_W - hint_w) // 2

            self._rect(frame, x, y, sq, sq, c1)
            self._rect(frame, x + 4, y, sq, sq, c2)
            self._vline(frame, x + 9, y, sq, WHITE)
            self._rect(frame, x + 12, y, sq, sq, result)

    def _draw_border(self, frame, col):
        for d in range(GRID_W):
            self._set_pixel(frame, d, 0, col)
            self._set_pixel(frame, d, GRID_H - 1, col)
        for d in range(GRID_H):
            self._set_pixel(frame, 0, d, col)
            self._set_pixel(frame, GRID_W - 1, d, col)

    def render_interface(self, frame):
        g = self.game
        if g is None or not g.jars:
            return frame

        jars = g.jars
        num = len(jars)
        max_cap = max(j["cap"] for j in jars)
        positions, jar_h, gap = _jar_layout(num, max_cap)

        tgt_lookup = {}
        for idx, amt, col in g._shuffled_targets:
            tgt_lookup[idx] = (amt, col)

        for i, jar in enumerate(jars):
            jx, jy = positions[i]
            is_cur = (i == g.cursor) and not g.won and not g.game_over
            is_sel = (i == g.selected) and not g.won and not g.game_over
            tgt_amt = tgt_lookup[i][0] if i in tgt_lookup else 0
            self._draw_jar(frame, jx, jy, jar, max_cap, is_cur, is_sel, tgt_amt)

        for idx, (amt, col) in tgt_lookup.items():
            jx, jy = positions[idx]
            hint_x = jx + (JAR_W - 4) // 2
            hint_y = jy + jar_h + 2
            self._rect(frame, hint_x, hint_y, 4, 3, col)

        self._draw_mixing_hints(frame, g.level_idx)

        self._draw_level_dots(frame, g.level_num, NUM_LEVELS)
        self._draw_lives(frame, g.lives)
        self._draw_move_bar(frame, g.max_moves - g.move_count, g.max_moves)

        if g.won:
            self._draw_border(frame, GREEN)

        return frame


_marker = Sprite(
    pixels=[[BLACK]],
    name="marker",
    visible=False,
    collidable=False,
    tags=["marker"],
    layer=0,
)

levels = []
for _i in range(NUM_LEVELS):
    levels.append(
        Level(
            sprites=[_marker.clone().set_position(0, 0)],
            grid_size=(64, 64),
            data={"idx": _i},
            name="Level " + str(_i + 1),
        )
    )


def _hit_test_jar(jars, click_x, click_y):
    num = len(jars)
    if num == 0:
        return -1
    max_cap = max(j["cap"] for j in jars)
    positions, jar_h, gap = _jar_layout(num, max_cap)
    for i in range(num):
        jx, jy = positions[i]
        if jx <= click_x < jx + JAR_W and jy <= click_y < jy + jar_h:
            return i
    return -1


class Eg05(ARCBaseGame):
    def __init__(self, seed: int = 0):
        self.hud = LiquidHud()
        self.hud.game = self

        self._rng = random.Random(seed)
        self._seed = seed

        self.level_idx = 0
        self.level_num = 1
        self.jars = []
        self.init_jars = []
        self.cursor = 0
        self.selected = -1
        self.move_count = 0
        self.max_moves = 0
        self.lives = 3
        self.won = False
        self.game_over = False
        self.reset_press_count = 0
        self.play_since_reset = False
        self.died_at_level = -1
        self.turn = 0
        self._shuffled_targets = []
        self._history = []
        self._game_over = False

        super().__init__(
            "eg05",
            levels,
            Camera(0, 0, 64, 64, BLACK, BLACK, [self.hud]),
            available_actions=[0, 3, 4, 5, 6, 7],
        )

    def on_set_level(self, level):
        idx = self.current_level.get_data("idx")
        if idx is None:
            idx = 0
        if self.died_at_level >= 0:
            idx = self.died_at_level
            self.died_at_level = -1
        self._init_level(idx)

        self._history = []
        self._game_over = False

    def _init_level(self, idx, reset_lives=True):
        cfg = LEVEL_CONFIGS[idx]
        self.level_idx = idx
        self.level_num = idx + 1

        num_jars = len(cfg["jars"])
        target_indices = {ti for ti, _, _ in cfg["targets"]}
        non_target = [i for i in range(num_jars) if i not in target_indices]
        shuffled_non_target = list(non_target)
        self._rng.shuffle(shuffled_non_target)

        order = list(range(num_jars))
        for orig, shuf in zip(non_target, shuffled_non_target):
            order[orig] = shuf

        src_jars = deepcopy(cfg["jars"])
        arranged = [src_jars[order[i]] for i in range(num_jars)]

        old_to_new = {}
        for new_pos in range(num_jars):
            old_to_new[order[new_pos]] = new_pos
        remapped_targets = [(old_to_new[ti], ta, tc) for ti, ta, tc in cfg["targets"]]

        self._shuffled_targets = remapped_targets
        self.init_jars = deepcopy(arranged)
        self.jars = deepcopy(arranged)
        self.cursor = 0
        self.selected = -1
        self.move_count = 0
        self.max_moves = cfg["max_moves"]
        self.won = False
        self.game_over = False
        if reset_lives:
            self.lives = 3
        self._history = []
        self.turn = 0

    def _do_select_or_pour(self, jar_idx):
        if self.won or self.game_over:
            return
        if jar_idx < 0 or jar_idx >= len(self.jars):
            return

        if self.selected < 0:
            if self.jars[jar_idx]["amount"] > 0:
                self.selected = jar_idx
        elif self.selected == jar_idx:
            self.selected = -1
        else:
            result = try_pour(self.jars, self.selected, jar_idx)
            if result is not None:
                self.jars = result
                self.selected = -1

                if check_targets(self.jars, self._shuffled_targets):
                    self.won = True
            else:
                self.selected = -1

    def _save_state(self):
        self._history.append(
            {
                "jars": deepcopy(self.jars),
                "init_jars": deepcopy(self.init_jars),
                "cursor": self.cursor,
                "selected": self.selected,
                "won": self.won,
                "game_over": self.game_over,
                "play_since_reset": self.play_since_reset,
            }
        )

    def _undo(self):
        if not self._history:
            return
        snap = self._history.pop()
        self.jars = snap["jars"]
        self.init_jars = snap["init_jars"]
        self.cursor = snap["cursor"]
        self.selected = snap["selected"]
        self.won = snap["won"]
        self.game_over = snap["game_over"]
        self.play_since_reset = snap["play_since_reset"]

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

    def _reset_current_level(self):
        self._init_level(self.level_idx, reset_lives=False)
        self._history = []

    def _check_moves_exhausted(self):
        if self.move_count >= self.max_moves and not self.won:
            game_over = self._handle_death()
            if game_over:
                return
            else:
                self._reset_current_level()

    def get_actions(self):
        if self.game_over or self.won:
            return []
        return [0, 3, 4, 5, 6, 7]

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

    def _handle_click_action(self):
        cx = self.action.data.get("x", 0)
        cy = self.action.data.get("y", 0)
        coords = self.camera.display_to_grid(cx, cy)
        if coords:
            jar_idx = _hit_test_jar(self.jars, coords[0], coords[1])
            if jar_idx >= 0:
                self.cursor = jar_idx
                self._do_select_or_pour(jar_idx)

    def _dispatch_action(self, act):
        num = len(self.jars)
        if act == 3:
            self.cursor = (self.cursor - 1) % num
            self.move_count += 1
        elif act == 4:
            self.cursor = (self.cursor + 1) % num
            self.move_count += 1
        elif act == 5:
            self._do_select_or_pour(self.cursor)
            self.move_count += 1
        elif act == 6:
            self._handle_click_action()
            self.move_count += 1

    def step(self) -> None:
        action_id = self.action.id.value

        if action_id == 0:
            if not self.play_since_reset:
                self.reset_press_count += 1
            else:
                self.reset_press_count = 1

            if self.reset_press_count >= 2:
                self._init_level(0)
                self.reset_press_count = 0
                self.died_at_level = -1
            elif self.died_at_level >= 0:
                self._init_level(self.died_at_level)
            else:
                self._init_level(self.level_idx)

            self.play_since_reset = False
            self._history = []
            self.complete_action()
            return

        if action_id == 7:
            self._undo()
            self.move_count += 1
            self._check_moves_exhausted()
            self.complete_action()
            return

        if self.game_over:
            self.died_at_level = self.level_idx
            self.complete_action()
            self.lose()
            return

        if self.won:
            self.complete_action()
            self.next_level()
            return

        self.turn += 1
        self.play_since_reset = True
        self.reset_press_count = 0
        self.died_at_level = -1

        self._save_state()

        self._dispatch_action(action_id)
        self._check_moves_exhausted()

        if self.won:
            self.complete_action()
            self.next_level()
            return

        self.complete_action()


_COLOR_CHAR = {
    RED: "R",
    BLUE: "B",
    YELLOW: "Y",
    WHITE: "W",
    GREEN: "G",
    MAGENTA: "M",
    MAROON: "N",
    PURPLE: "P",
    ORANGE: "O",
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
        "left": GameAction.ACTION3,
        "right": GameAction.ACTION4,
        "select": GameAction.ACTION5,
        "click": GameAction.ACTION6,
        "undo": GameAction.ACTION7,
    }

    _VALID_ACTIONS = list(_ACTION_MAP.keys())

    def __init__(self, seed: int = 0) -> None:
        self._engine: Any = Eg05(seed=seed)
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

        header = (
            f"Level {e.level_num}/{NUM_LEVELS}"
            f" | Moves: {e.move_count}/{e.max_moves}"
            f" | Lives: {e.lives}"
            f" | Won: {e.won}"
        )
        rules = (
            "Move cursor with left/right. Select to pick up or pour. Match jar targets."
        )

        jar_parts = []
        for i, jar in enumerate(e.jars):
            jar_type = jar.get("type", JAR_NORMAL)
            if jar_type == JAR_MIXER:
                type_ind = "M"
            elif jar_type == JAR_FILTER:
                type_ind = "F"
            else:
                type_ind = " "

            color = jar["color"]
            if color is not None:
                ch = _COLOR_CHAR.get(color, "?")
            else:
                ch = "_"

            content = f"{type_ind}{ch}:{jar['amount']}/{jar['cap']}"

            if i == e.cursor and i == e.selected:
                content = "{[" + content + "]}"
            elif i == e.cursor:
                content = "[" + content + "]"
            elif i == e.selected:
                content = "{" + content + "}"

            jar_parts.append(content)

        jar_line = " ".join(jar_parts)

        tgt_parts = []
        for idx, amt, col in e._shuffled_targets:
            ch = _COLOR_CHAR.get(col, "?")
            tgt_parts.append(f"jar{idx}={amt}x{ch}")
        target_line = ", ".join(tgt_parts)

        cursor_line = f"Cursor: {e.cursor}  Selected: {e.selected}"

        parts = [
            header,
            rules,
            "Jars:",
            jar_line,
            cursor_line,
            f"Targets: {target_line}",
        ]
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
                "level": e.level_idx + 1,
                "moves_remaining": e.max_moves - e.move_count,
                "max_moves": e.max_moves,
                "lives": e.lives,
                "won": e.won,
                "jars": deepcopy(e.jars),
                "cursor": e.cursor,
                "selected": e.selected,
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
