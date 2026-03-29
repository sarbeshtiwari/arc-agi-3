import io
import random
import struct
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import gymnasium as gym
import numpy as np
from arcengine import (
    ActionInput,
    ARCBaseGame,
    Camera,
    GameAction,
    Level,
    Sprite,
)
from arcengine import (
    GameState as EngineGameState,
)
from gymnasium import spaces


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


GRID_SIZE = (64, 64)
CAM_W, CAM_H = 64, 64

BG_COLOR = 0
PAD_COLOR = 0
LINE_COLOR = 3

BLOCK_COLORS = [
    9,
    14,
    8,
    11,
    15,
    13,
    10,
    6,
]

CURSOR_COLOR = 5
EMPTY_COLOR = 2
HAND_BORDER_COLOR = 12

BS = 7
BG = 1

QUEUE_X = 4
STACK_X = 53
COL_START_Y = 4

HAND_X = 27
HAND_Y = 25

BAR_Y = 48
BAR_EMPTY_COLOR = 2
LIVES_Y = 52
TARGET_Y = 56
TARGET_BS = 4

POS_Q_FRONT = 0
POS_Q_BACK = 1
POS_STACK = 2

LEVEL_DEFS = [
    {
        "num_blocks": 4,
        "num_colors": 4,
        "counts": [1, 1, 1, 1],
        "max_moves": 48,
        "difficulty": 0.3,
    },
    {
        "num_blocks": 5,
        "num_colors": 5,
        "counts": [1, 1, 1, 1, 1],
        "max_moves": 64,
        "difficulty": 0.4,
    },
    {
        "num_blocks": 6,
        "num_colors": 6,
        "counts": [1, 1, 1, 1, 1, 1],
        "max_moves": 80,
        "difficulty": 0.5,
    },
    {
        "num_blocks": 7,
        "num_colors": 7,
        "counts": [1, 1, 1, 1, 1, 1, 1],
        "max_moves": 92,
        "difficulty": 0.6,
    },
    {
        "num_blocks": 8,
        "num_colors": 8,
        "counts": [1, 1, 1, 1, 1, 1, 1, 1],
        "max_moves": 104,
        "difficulty": 0.75,
    },
]


def _square_pixels(color, size=BS):
    return [[color] * size for _ in range(size)]


def _build_sorted(ldef):
    q = []
    for i in range(ldef["num_colors"]):
        for _ in range(ldef["counts"][i]):
            q.append(BLOCK_COLORS[i])
    return q


def _is_sorted(q):
    if len(q) == 0:
        return False
    for i in range(len(q) - 1):
        idx_a = BLOCK_COLORS.index(q[i])
        idx_b = BLOCK_COLORS.index(q[i + 1])
        if idx_a > idx_b:
            return False
    return True


def _generate_puzzle(rng, ldef):
    sorted_q = _build_sorted(ldef)
    num_blocks = ldef["num_blocks"]
    max_m = ldef["max_moves"]
    difficulty = ldef["difficulty"]

    min_scramble = int(max_m * difficulty)
    max_scramble = int(max_m * min(difficulty + 0.15, 0.95))

    queue = sorted_q[:]
    stack = []
    hand = None

    last_clean = None

    for step in range(max_scramble):
        moves = []
        if hand is None:
            if queue:
                moves.append("take_back")
            if stack:
                moves.append("pop_stack")
        else:
            moves.append("prepend_front")
            moves.append("push_stack")

        if not moves:
            break

        move = rng.choice(moves)

        if move == "take_back":
            hand = queue.pop()
        elif move == "pop_stack":
            hand = stack.pop()
        elif move == "prepend_front":
            queue.insert(0, hand)
            hand = None
        elif move == "push_stack":
            stack.append(hand)
            hand = None

        if (
            step + 1 >= min_scramble
            and hand is None
            and not stack
            and len(queue) == num_blocks
            and not _is_sorted(queue)
        ):
            last_clean = queue[:]

    if last_clean is not None:
        return last_clean

    result = sorted_q[:]
    result.reverse()
    if _is_sorted(result):
        result[0], result[1] = result[1], result[0]
    return result


class Ps42(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)
        self._seed = seed

        self._queue = []
        self._stack = []
        self._hand = None
        self._cursor_pos = POS_Q_FRONT
        self._moves_used = 0
        self._max_moves = 18
        self._ldef = None
        self._history = []
        self._bs = BS
        self._bg = BG
        self._hand_y = HAND_Y
        self._lives = 3

        camera = Camera(0, 0, CAM_W, CAM_H, BG_COLOR, PAD_COLOR, [])

        levels = [
            Level(sprites=[], grid_size=GRID_SIZE, name=f"Level {i + 1}")
            for i in range(len(LEVEL_DEFS))
        ]

        super().__init__(
            game_id="ps42",
            levels=levels,
            camera=camera,
            available_actions=[0, 1, 2, 3, 4, 5, 6, 7],
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        idx = self.level_index
        ldef = LEVEL_DEFS[idx]
        self._ldef = ldef

        self._max_moves = ldef["max_moves"]
        self._moves_used = 0
        self._hand = None
        self._stack = []
        self._cursor_pos = POS_Q_FRONT
        self._history = []
        self._lives = 3

        max_col_height = BAR_Y - COL_START_Y - 1
        num_blocks = ldef["num_blocks"]
        chosen_bs = BS
        chosen_bg = BG
        for try_bs in range(BS, 2, -1):
            needed = num_blocks * try_bs + (num_blocks - 1) * chosen_bg
            if needed <= max_col_height:
                chosen_bs = try_bs
                break
        else:
            chosen_bs = 3
            chosen_bg = 0

        self._bs = chosen_bs
        self._bg = chosen_bg

        col_height = num_blocks * self._bs + (num_blocks - 1) * self._bg
        self._hand_y = COL_START_Y + (col_height - (self._bs + 2)) // 2

        self._queue = _generate_puzzle(self._rng, ldef)

        self._rebuild_sprites(level)

    def _rebuild_sprites(self, level: Level) -> None:
        level.remove_all_sprites()
        self._draw_stack_slots(level)
        self._draw_queue_blocks(level)
        self._draw_stack_blocks(level)
        self._draw_hand(level)
        self._draw_move_bar(level)
        self._draw_lives(level)
        self._draw_target_sequence(level)
        self._draw_cursor(level)

    def _draw_stack_slots(self, level: Level) -> None:
        num_slots = self._ldef["num_blocks"]
        bs, bg = self._bs, self._bg
        for i in range(num_slots):
            by = COL_START_Y + i * (bs + bg)
            pixels = _square_pixels(LINE_COLOR, bs)
            level.add_sprite(
                Sprite(
                    pixels=pixels,
                    name=f"stk_slot_{i}",
                    x=STACK_X,
                    y=by,
                    layer=0,
                    tags=["stack_slot"],
                )
            )

    def _draw_queue_blocks(self, level: Level) -> None:
        bs, bg = self._bs, self._bg
        for i, color in enumerate(self._queue):
            by = COL_START_Y + i * (bs + bg)
            pixels = _square_pixels(color, bs)
            level.add_sprite(
                Sprite(
                    pixels=pixels,
                    name=f"q_{i}",
                    x=QUEUE_X,
                    y=by,
                    layer=1,
                    tags=["queue"],
                )
            )

    def _draw_stack_blocks(self, level: Level) -> None:
        bs, bg = self._bs, self._bg
        rendered = list(reversed(self._stack))
        for i, color in enumerate(rendered):
            by = COL_START_Y + i * (bs + bg)
            pixels = _square_pixels(color, bs)
            level.add_sprite(
                Sprite(
                    pixels=pixels,
                    name=f"stk_{i}",
                    x=STACK_X,
                    y=by,
                    layer=1,
                    tags=["stack"],
                )
            )

    def _draw_hand(self, level: Level) -> None:
        bs = self._bs
        border_size = bs + 2
        hx = HAND_X
        hy = self._hand_y

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
        inner_pixels = _square_pixels(inner_color, bs)
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

    def _draw_move_bar(self, level: Level) -> None:
        remaining = max(0, self._max_moves - self._moves_used)

        max_dots = 20
        dot_size = 2
        dot_gap = 0
        total_dots = max_dots
        filled_dots = max(
            0, (remaining * max_dots + self._max_moves - 1) // self._max_moves
        )

        total_w = total_dots * (dot_size + dot_gap) - dot_gap
        start_x = (CAM_W - total_w) // 2

        pct = filled_dots / max_dots if max_dots > 0 else 0
        if pct > 0.6:
            fill_color = 14
        elif pct > 0.3:
            fill_color = 11
        else:
            fill_color = 8

        for i in range(total_dots):
            dx = start_x + i * (dot_size + dot_gap)
            color = fill_color if i < filled_dots else BAR_EMPTY_COLOR
            dot_pixels = [[color] * dot_size for _ in range(dot_size)]
            level.add_sprite(
                Sprite(
                    pixels=dot_pixels,
                    name=f"dot_{i}",
                    x=dx,
                    y=BAR_Y,
                    layer=1,
                    tags=["bar"],
                )
            )

    def _draw_lives(self, level: Level) -> None:
        life_size = 2
        life_gap = 2
        total_lives = 3
        total_w = total_lives * life_size + (total_lives - 1) * life_gap
        start_x = (CAM_W - total_w) // 2
        ly = LIVES_Y

        for i in range(total_lives):
            lx = start_x + i * (life_size + life_gap)
            if i < self._lives:
                color = 8
            else:
                color = 2
            pixels = [[color] * life_size for _ in range(life_size)]
            level.add_sprite(
                Sprite(
                    pixels=pixels,
                    name=f"life_{i}",
                    x=lx,
                    y=ly,
                    layer=1,
                    tags=["lives"],
                )
            )

    def _draw_target_sequence(self, level: Level) -> None:
        sorted_q = _build_sorted(self._ldef)
        num = len(sorted_q)

        border = 1
        inner_w = num * TARGET_BS
        inner_h = TARGET_BS
        box_w = inner_w + border * 2
        box_h = inner_h + border * 2
        box_x = (CAM_W - box_w) // 2
        box_y = TARGET_Y

        box_pixels = []
        for row in range(box_h):
            line = []
            for col in range(box_w):
                if (
                    row < border
                    or row >= box_h - border
                    or col < border
                    or col >= box_w - border
                ):
                    line.append(LINE_COLOR)
                else:
                    inner_col = col - border
                    block_idx = inner_col // TARGET_BS
                    if block_idx < num:
                        line.append(sorted_q[block_idx])
                    else:
                        line.append(BG_COLOR)
            box_pixels.append(line)

        level.add_sprite(
            Sprite(
                pixels=box_pixels,
                name="target_box",
                x=box_x,
                y=box_y,
                layer=1,
                tags=["target"],
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

    def _get_cursor_rect(self, pos):
        n_q = len(self._queue)
        bs, bg = self._bs, self._bg

        if pos == POS_Q_FRONT:
            return (QUEUE_X - 1, COL_START_Y - 1, bs + 2, bs + 2)

        elif pos == POS_Q_BACK:
            if n_q > 0:
                last_y = COL_START_Y + (n_q - 1) * (bs + bg)
                return (QUEUE_X - 1, last_y - 1, bs + 2, bs + 2)
            else:
                return (QUEUE_X - 1, COL_START_Y - 1, bs + 2, bs + 2)

        elif pos == POS_STACK:
            return (STACK_X - 1, COL_START_Y - 1, bs + 2, bs + 2)

        return (0, 0, 0, 0)

    def _save_state(self):
        self._history.append(
            {
                "queue": self._queue[:],
                "stack": self._stack[:],
                "hand": self._hand,
                "moves": self._moves_used,
                "cursor": self._cursor_pos,
            }
        )

    def _restore_state(self):
        if self._history:
            state = self._history.pop()
            self._queue = state["queue"]
            self._stack = state["stack"]
            self._hand = state["hand"]
            self._moves_used = state["moves"]
            self._cursor_pos = state["cursor"]

    def _activate(self) -> bool:
        pos = self._cursor_pos

        if len(self._queue) <= 1 and pos in (POS_Q_FRONT, POS_Q_BACK):
            if self._hand is None:
                pos = POS_Q_FRONT
            else:
                pos = POS_Q_BACK

        if self._hand is None:
            if pos == POS_Q_FRONT and len(self._queue) > 0:
                self._hand = self._queue.pop(0)
                self._moves_used += 1
                return True

            elif pos == POS_Q_BACK:
                return False

            elif pos == POS_STACK and len(self._stack) > 0:
                self._hand = self._stack.pop()
                self._moves_used += 1
                return True

        else:
            if pos == POS_Q_FRONT:
                return False

            elif pos == POS_Q_BACK:
                self._queue.append(self._hand)
                self._hand = None
                self._moves_used += 1
                return True

            elif pos == POS_STACK:
                self._stack.append(self._hand)
                self._hand = None
                self._moves_used += 1
                return True

        return False

    def _check_win(self) -> bool:
        if self._hand is not None:
            return False
        if len(self._stack) > 0:
            return False
        if len(self._queue) != self._ldef["num_blocks"]:
            return False
        return _is_sorted(self._queue)

    def _draw_win_screen(self, level):
        level.remove_all_sprites()

        w, h = CAM_W, CAM_H
        pixels = [[BG_COLOR] * w for _ in range(h)]

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
        super().handle_reset()

    def _reset_puzzle(self) -> None:
        level = self.current_level
        ldef = self._ldef

        self._moves_used = 0
        self._hand = None
        self._stack = []
        self._cursor_pos = POS_Q_FRONT
        self._history = []

        self._queue = _generate_puzzle(self._rng, ldef)

        self._rebuild_sprites(level)

    def step(self) -> None:
        action = self.action
        level = self.current_level

        acted = False

        if action.id != GameAction.ACTION7:
            self._save_state()

        if action.id == GameAction.ACTION1:
            if self._cursor_pos == POS_Q_BACK:
                self._cursor_pos = POS_Q_FRONT
            self._moves_used += 1
            self._rebuild_sprites(level)

        elif action.id == GameAction.ACTION2:
            if self._cursor_pos == POS_Q_FRONT:
                self._cursor_pos = POS_Q_BACK
            self._moves_used += 1
            self._rebuild_sprites(level)

        elif action.id == GameAction.ACTION3:
            if self._cursor_pos == POS_STACK:
                self._cursor_pos = POS_Q_FRONT
            self._moves_used += 1
            self._rebuild_sprites(level)

        elif action.id == GameAction.ACTION4:
            if self._cursor_pos == POS_Q_FRONT or self._cursor_pos == POS_Q_BACK:
                self._cursor_pos = POS_STACK
            self._moves_used += 1
            self._rebuild_sprites(level)

        elif action.id == GameAction.ACTION5:
            acted = self._activate()
            if not acted:
                self._moves_used += 1
                self._rebuild_sprites(level)

        elif action.id == GameAction.ACTION7:
            if self._history:
                current_moves = self._moves_used
                self._restore_state()
                self._moves_used = current_moves + 1
                self._rebuild_sprites(level)
            else:
                self._moves_used += 1
                self._rebuild_sprites(level)

        elif action.id == GameAction.ACTION6:
            cx = action.data.get("x", 0)
            cy = action.data.get("y", 0)
            midpoint_x = (QUEUE_X + STACK_X) // 2
            if cx >= midpoint_x:
                new_pos = POS_STACK
            else:
                n_q = len(self._queue)
                bs, bg = self._bs, self._bg
                if n_q > 1:
                    mid_y = COL_START_Y + (n_q - 1) * (bs + bg) // 2
                    if cy <= mid_y:
                        new_pos = POS_Q_FRONT
                    else:
                        new_pos = POS_Q_BACK
                else:
                    new_pos = POS_Q_FRONT
            self._cursor_pos = new_pos
            acted = self._activate()
            if not acted:
                self._moves_used += 1
            self._rebuild_sprites(level)

        if acted:
            self._rebuild_sprites(level)

            if self._check_win():
                if self.is_last_level():
                    self._draw_win_screen(level)
                self.next_level()
                self.complete_action()
                return

        if self._moves_used >= self._max_moves:
            self._lives -= 1
            if self._lives <= 0:
                self.lose()
            else:
                self._reset_puzzle()

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

    _COLOR_NAMES: Dict[int, str] = {
        9: "Blue",
        14: "Green",
        8: "Red",
        11: "Yellow",
        15: "Purple",
        13: "Maroon",
        10: "Blue Light",
        6: "Magenta",
    }

    def __init__(self, seed: int = 0) -> None:
        self._engine: Optional[Ps42] = Ps42(seed=seed)
        self._total_turns = 0
        self._last_action_was_reset = False
        self._game_won = False

    @property
    def _eng(self) -> "Ps42":
        assert self._engine is not None
        return self._engine

    def _build_text_obs(self) -> str:
        e = self._eng
        ldef = e._ldef
        queue_str = ", ".join(self._COLOR_NAMES.get(c, str(c)) for c in e._queue)
        stack_display = list(reversed(e._stack))
        stack_str = ", ".join(self._COLOR_NAMES.get(c, str(c)) for c in stack_display)
        hand_str = (
            self._COLOR_NAMES.get(e._hand, str(e._hand))
            if e._hand is not None
            else "(empty)"
        )

        remaining = max(0, e._max_moves - e._moves_used)
        cursor_names = {
            POS_Q_FRONT: "queue-front",
            POS_Q_BACK: "queue-back",
            POS_STACK: "stack-top",
        }

        target_q = _build_sorted(e._ldef)
        target_str = ", ".join(self._COLOR_NAMES.get(c, str(c)) for c in target_q)

        lines = [
            f"Level:{e.level_index + 1}/{len(LEVEL_DEFS)} Lives:{e._lives}",
            f"Moves:{remaining}/{e._max_moves}",
            f"Queue: [{queue_str}]",
            f"Stack: [{stack_str}]",
            f"Hand: {hand_str}",
            f"Cursor: {cursor_names.get(e._cursor_pos, 'unknown')}",
            f"Target: [{target_str}]",
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
                "max_moves": e._max_moves,
                "moves_used": e._moves_used,
                "queue_length": len(e._queue),
                "stack_length": len(e._stack),
                "hand": e._hand,
                "done": done,
                "info": info or {},
            },
        )

    def reset(self) -> GameState:
        e = self._eng
        reset_input = ActionInput(id=GameAction.RESET)
        if self._game_won or self._last_action_was_reset:
            e.perform_action(reset_input)
            e.perform_action(reset_input)
            self._total_turns = 0
        else:
            e.perform_action(reset_input)
            self._total_turns = 0
        self._game_won = False
        self._last_action_was_reset = True
        return self._build_game_state()

    def get_actions(self) -> List[str]:
        e = self._eng
        game_over = (
            getattr(e, "_game_over", False) or e._state == EngineGameState.GAME_OVER
        )
        if self._game_won or game_over:
            return ["reset"]
        return ["up", "down", "left", "right", "select", "click", "undo", "reset"]

    def is_done(self) -> bool:
        e = self._eng
        game_over = (
            getattr(e, "_game_over", False) or e._state == EngineGameState.GAME_OVER
        )
        return self._game_won or game_over or getattr(e, "_terminated", False)

    def step(self, action: str) -> StepResult:
        e = self._eng

        if action == "reset":
            state = self.reset()
            return StepResult(
                state=state, reward=0.0, done=False, info={"action": "reset"}
            )

        if action not in self._ACTION_MAP:
            raise ValueError(
                f"Invalid action '{action}'. Must be one of {list(self._ACTION_MAP.keys())}"
            )

        self._last_action_was_reset = False
        self._total_turns += 1

        game_action = self._ACTION_MAP[action]
        info: Dict = {"action": action}

        prev_level = e.level_index
        action_input = ActionInput(id=game_action, data={})
        frame = e.perform_action(action_input, raw=True)

        total_levels = len(e._levels)
        reward_per_level = 1.0 / total_levels

        game_won = frame and frame.state and frame.state.name == "WIN"
        level_advanced = (not game_won) and (e.level_index != prev_level)
        game_over = e._state == EngineGameState.GAME_OVER
        done = game_over or game_won

        if game_won:
            self._game_won = True
            info["reason"] = "game_complete"
            return StepResult(
                state=self._build_game_state(done=True, info=info),
                reward=reward_per_level,
                done=True,
                info=info,
            )

        if level_advanced:
            info["reason"] = "level_complete"
            return StepResult(
                state=self._build_game_state(done=False, info=info),
                reward=reward_per_level,
                done=False,
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

        return StepResult(
            state=self._build_game_state(done=False, info=info),
            reward=0.0,
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
    ) -> tuple:
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed

        self._env = PuzzleEnvironment(seed=self._seed)
        state = self._env.reset()
        return self._get_obs(), self._build_info(state)

    def step(self, action: int) -> tuple:
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
