import io
import random
import struct
import zlib
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from arcengine import (
    ActionInput,
    ARCBaseGame,
    Camera,
    GameAction,
    Level,
    RenderableUserDisplay,
    Sprite,
)
from gymnasium import spaces


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
OFFWHITE = 1
LTGREY = 2
GREY = 3
DKGREY = 4
BLACK = 5
MAGENTA = 6
PINK = 7
RED = 8
BLUE = 9
LTBLUE = 10
YELLOW = 11
ORANGE = 12
MAROON = 13
GREEN = 14
PURPLE = 15

BACKGROUND_COLOR = BLACK
PADDING_COLOR = BLACK

GRID_W = 64
GRID_H = 64
DEFAULT_BOARD_SIZE = 8
TILE_PX = 6

BOARD_ORIGIN_X = 4
BOARD_ORIGIN_Y = 10

HUD_Y = 1
STACK_BAR_Y = 59

MAX_LIVES = 3

TILE_COLOR = {
    0: BLACK,
    1: BLUE,
    2: ORANGE,
    3: RED,
}

CURSOR_COLOR = YELLOW
CURSOR_ACTIVE = WHITE

LEVEL_DEFS = [
    {"label": 0, "stack_cap": 0, "min_val": 1, "board_size": 4},
    {"label": 1, "stack_cap": 0, "min_val": 1, "board_size": 8},
    {"label": 2, "stack_cap": 4, "min_val": 1, "board_size": 8},
    {"label": 3, "stack_cap": 0, "min_val": 1, "board_size": 8},
]

NUM_LEVELS = len(LEVEL_DEFS)

_SETUP_CLICKS = {
    0: [(1, 2, 3), (0, 3, 1), (2, 0, 2), (3, 1, 3)],
    1: [(0, 0, 2), (1, 1, 3), (2, 2, 1), (3, 3, 2), (4, 4, 3), (5, 5, 1)],
    2: [(1, 1, 2), (3, 5, 3), (0, 7, 1), (6, 2, 2)],
    3: [(0, 0, 3), (1, 2, 2), (2, 1, 1), (3, 5, 3), (4, 7, 2)],
}


def _xor_row(board, row, val, board_size):
    for col in range(board_size):
        board[row][col] ^= val


def _xor_col(board, col, val, board_size):
    for row in range(board_size):
        board[row][col] ^= val


def _xor_main_diag(board, row, col, val, board_size):
    diff = row - col
    for r in range(board_size):
        c = r - diff
        if 0 <= c < board_size:
            board[r][c] ^= val


def _xor_block3(board, row, col, val, board_size):
    for delta_row in range(-1, 2):
        for delta_col in range(-1, 2):
            r, c = row + delta_row, col + delta_col
            if 0 <= r < board_size and 0 <= c < board_size:
                board[r][c] ^= val


def _propagate_cross(board, row, col, val, board_size):
    _xor_row(board, row, val, board_size)
    _xor_col(board, col, val, board_size)


def _mask_board(board, board_size):
    for r in range(board_size):
        for c in range(board_size):
            board[r][c] &= 3


def _setup_propagation(board, row, col, val, level_index, board_size):
    if level_index in (0, 2):
        _propagate_cross(board, row, col, val, board_size)
        board[row][col] ^= val
    elif level_index == 3:
        _propagate_cross(board, row, col, val, board_size)
        _xor_main_diag(board, row, col, val, board_size)
    elif level_index == 1:
        if (row + col) % 2 == 0:
            _propagate_cross(board, row, col, val, board_size)
            board[row][col] ^= val
        else:
            _xor_block3(board, row, col, val, board_size)

    _mask_board(board, board_size)


def _apply_propagation(board, row, col, val, level_index, board_size):
    if level_index in (0, 2):
        _propagate_cross(board, row, col, val, board_size)
    elif level_index == 3:
        _propagate_cross(board, row, col, val, board_size)
        _xor_main_diag(board, row, col, val, board_size)
    elif level_index == 1:
        if (row + col) % 2 == 0:
            _propagate_cross(board, row, col, val, board_size)
        else:
            _xor_block3(board, row, col, val, board_size)

    board[row][col] = 0
    _mask_board(board, board_size)


def _build_initial_board(level_index, board_size):
    board = [[0] * board_size for _ in range(board_size)]
    for row, col, val in _SETUP_CLICKS[level_index]:
        _setup_propagation(board, row, col, val, level_index, board_size)
    return board


def _board_is_zero(board):
    for row in board:
        for val in row:
            if val != 0:
                return False
    return True


def _has_clickable_tile(board, min_val, board_size):
    for row in range(board_size):
        for col in range(board_size):
            if board[row][col] >= min_val:
                return True
    return False


def _board_origin(board_size):
    offset_x = (DEFAULT_BOARD_SIZE - board_size) * TILE_PX // 2
    offset_y = (DEFAULT_BOARD_SIZE - board_size) * TILE_PX // 2
    return BOARD_ORIGIN_X + offset_x, BOARD_ORIGIN_Y + offset_y


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
    names = [f"Level {i + 1}" for i in range(NUM_LEVELS)]
    return [
        Level(
            sprites=[sprites["marker"].clone().set_position(0, 0)],
            grid_size=(64, 64),
            data={"idx": i},
            name=names[i],
        )
        for i in range(NUM_LEVELS)
    ]


levels = _build_levels()


class StackReversalHud(RenderableUserDisplay):
    def __init__(self):
        super().__init__()
        self.game = None

    def _px(self, buf, x, y, color):
        if 0 <= x < GRID_W and 0 <= y < GRID_H:
            buf[y][x] = color

    def _rect(self, buf, x, y, w, h, color):
        for dy in range(h):
            for dx in range(w):
                self._px(buf, x + dx, y + dy, color)

    def _hline(self, buf, x, y, length, color):
        for dx in range(length):
            self._px(buf, x + dx, y, color)

    def _vline(self, buf, x, y, length, color):
        for dy in range(length):
            self._px(buf, x, y + dy, color)

    def _draw_tiles(self, buf):
        game = self.game
        level_def = LEVEL_DEFS[game.lidx]
        board_size = level_def["board_size"]
        origin_x, origin_y = _board_origin(board_size)

        for row in range(board_size):
            for col in range(board_size):
                tx = origin_x + col * TILE_PX
                ty = origin_y + row * TILE_PX
                val = game.board[row][col]

                fill = TILE_COLOR[val]

                self._rect(buf, tx + 1, ty + 1, TILE_PX - 2, TILE_PX - 2, fill)

                border = DKGREY if val == 0 else GREY
                self._hline(buf, tx, ty, TILE_PX, border)
                self._hline(buf, tx, ty + TILE_PX - 1, TILE_PX, border)
                self._vline(buf, tx, ty, TILE_PX, border)
                self._vline(buf, tx + TILE_PX - 1, ty, TILE_PX, border)

    def _draw_cursor(self, buf):
        game = self.game
        board_size = LEVEL_DEFS[game.lidx]["board_size"]
        origin_x, origin_y = _board_origin(board_size)
        cursor_row, cursor_col = game.cursor
        tx = origin_x + cursor_col * TILE_PX
        ty = origin_y + cursor_row * TILE_PX

        if game.flash > 0:
            cur_col = CURSOR_ACTIVE if game.flash % 2 == 0 else CURSOR_COLOR
        else:
            cur_col = CURSOR_COLOR

        self._hline(buf, tx, ty, TILE_PX, cur_col)
        self._hline(buf, tx, ty + TILE_PX - 1, TILE_PX, cur_col)
        self._vline(buf, tx, ty, TILE_PX, cur_col)
        self._vline(buf, tx + TILE_PX - 1, ty, TILE_PX, cur_col)

        for dx, dy in [(-1, -1), (TILE_PX, -1), (-1, TILE_PX), (TILE_PX, TILE_PX)]:
            self._px(buf, tx + dx, ty + dy, cur_col)

    def _draw_flash_border(self, buf):
        game = self.game
        if game.flash <= 0:
            return

        board_size = LEVEL_DEFS[game.lidx]["board_size"]
        origin_x, origin_y = _board_origin(board_size)
        bx = origin_x - 1
        by = origin_y - 1
        bw = board_size * TILE_PX + 2
        bh = board_size * TILE_PX + 2
        flash_color = WHITE if game.flash % 2 == 0 else YELLOW
        self._hline(buf, bx, by, bw, flash_color)
        self._hline(buf, bx, by + bh - 1, bw, flash_color)
        self._vline(buf, bx, by, bh, flash_color)
        self._vline(buf, bx + bw - 1, by, bh, flash_color)

    def _draw_hud(self, buf):
        game = self.game
        total = game.max_moves
        remaining = max(0, game.rem_moves)
        bar_x = 24
        bar_w = 32
        bar_h = 3
        self._rect(buf, bar_x, HUD_Y, bar_w, bar_h, DKGREY)
        filled = (bar_w * remaining // total) if total > 0 else 0
        if filled > 0:
            if remaining > total * 0.5:
                fill_color = GREEN
            elif remaining > total * 0.25:
                fill_color = YELLOW
            else:
                fill_color = RED
            self._rect(buf, bar_x, HUD_Y, filled, bar_h, fill_color)

    def _draw_lives(self, buf):
        game = self.game
        lives_y = HUD_Y + 5
        for i in range(game._max_lives):
            lx = 1 + i * 5
            color = PINK if i < game._lives else DKGREY
            self._rect(buf, lx, lives_y, 3, 3, color)

    def render_interface(self, buf):
        if self.game is None:
            return buf

        for y in range(GRID_H):
            for x in range(GRID_W):
                buf[y][x] = BLACK

        self._draw_hud(buf)
        self._draw_lives(buf)
        self._draw_tiles(buf)
        self._draw_cursor(buf)
        self._draw_flash_border(buf)

        return buf


class Eg01(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self.hud = StackReversalHud()
        self.hud.game = self

        self.lidx = 0
        self.board = []
        self.init_board = []
        self.stack = []
        self.max_moves = 40
        self.rem_moves = 40
        self.cursor = (0, 0)
        self.flash = 0
        self._won = False
        self._max_lives = MAX_LIVES
        self._lives = MAX_LIVES
        self.consecutive_resets = 0

        self._rng = random.Random(seed)
        self._engine_snapshots: List[dict] = []
        self._game_over = False

        camera = Camera(
            x=0, y=0,
            width=64, height=64,
            background=BACKGROUND_COLOR,
            letter_box=PADDING_COLOR,
            interfaces=[self.hud],
        )
        super().__init__(
            "eg01",
            levels,
            camera,
            available_actions=[0, 1, 2, 3, 4, 5, 7],
        )

    def on_set_level(self, _level):
        idx = self.current_level.get_data("idx")
        if idx is None:
            idx = 0
        self.lidx = idx
        board_size = LEVEL_DEFS[idx]["board_size"]
        self.init_board = _build_initial_board(idx, board_size)
        self.board = deepcopy(self.init_board)
        self.stack = []

        self.max_moves = len(_SETUP_CLICKS[idx]) * (board_size * 2 + 4)
        self.rem_moves = self.max_moves
        self.cursor = (
            self._rng.randint(0, board_size - 1),
            self._rng.randint(0, board_size - 1),
        )
        self.flash = 0
        self._won = False
        self.consecutive_resets = 0

        self._engine_snapshots = []
        self._game_over = False

    def _engine_save_snapshot(self):
        self._engine_snapshots.append({
            "board": deepcopy(self.board),
            "stack": deepcopy(self.stack),
            "cursor": self.cursor,
            "rem_moves": self.rem_moves,
            "flash": self.flash,
        })

    def _engine_restore_snapshot(self):
        snap = self._engine_snapshots.pop()
        self.board = snap["board"]
        self.stack = snap["stack"]
        self.cursor = snap["cursor"]
        self.flash = snap["flash"]

    def _reset_current_level(self):
        board_size = LEVEL_DEFS[self.lidx]["board_size"]
        self.board = deepcopy(self.init_board)
        self.stack = []
        self.rem_moves = self.max_moves
        self.cursor = (
            self._rng.randint(0, board_size - 1),
            self._rng.randint(0, board_size - 1),
        )
        self.flash = 0
        self._won = False

        self._engine_snapshots.clear()

    def full_reset(self):
        self._current_level_index = 0
        self.on_set_level(levels[0])
        self._game_over = False
        self._lives = self._max_lives

    def _move_cursor(self, delta_row, delta_col):
        board_size = LEVEL_DEFS[self.lidx]["board_size"]
        row, col = self.cursor
        row = (row + delta_row) % board_size
        col = (col + delta_col) % board_size
        self.cursor = (row, col)

    def _handle_action(self):
        if self.rem_moves <= 0:
            return

        row, col = self.cursor
        val = self.board[row][col]
        level_def = LEVEL_DEFS[self.lidx]

        if val == 0:
            return

        if level_def["min_val"] >= 2 and val < level_def["min_val"]:
            return

        self.stack.append((row, col, val))

        cap = level_def["stack_cap"]
        if cap > 0 and len(self.stack) > cap:
            dropped_row, dropped_col, dropped_val = self.stack.pop(0)
            _setup_propagation(
                self.board, dropped_row, dropped_col, dropped_val,
                self.lidx, level_def["board_size"],
            )

        _apply_propagation(
            self.board, row, col, val,
            self.lidx, level_def["board_size"],
        )

        self.flash = 6

    def _handle_death(self):
        self._lives -= 1
        self.consecutive_resets = 0
        if self._lives <= 0:
            self._game_over = True
            self._lives = self._max_lives
            self.lose()
            return True
        self._reset_current_level()
        return True

    def handle_reset(self):
        if self._state.value == "WIN":
            self.full_reset()
            return

        if self._game_over or self._state.value == "GAME_OVER":
            self._reset_current_level()
            self._lives = self._max_lives
            self._game_over = False
            return

        self.consecutive_resets += 1
        if self.consecutive_resets >= 2:
            self.consecutive_resets = 0
            self.set_level(0)
        else:
            self._reset_current_level()
            if self._lives <= 0:
                self._lives = self._max_lives

    def step(self):
        if self.flash > 0:
            self.flash -= 1

        act = self.action.id

        if act == GameAction.RESET:
            self.complete_action()
            return

        self.consecutive_resets = 0

        self.rem_moves -= 1

        if act == GameAction.ACTION7:
            if self._engine_snapshots:
                self._engine_restore_snapshot()
            if self.rem_moves <= 0 and not self._won:
                self._handle_death()
            self.complete_action()
            return

        self._engine_save_snapshot()

        if act == GameAction.ACTION1:
            self._move_cursor(-1, 0)
        elif act == GameAction.ACTION2:
            self._move_cursor(1, 0)
        elif act == GameAction.ACTION3:
            self._move_cursor(0, -1)
        elif act == GameAction.ACTION4:
            self._move_cursor(0, 1)
        elif act == GameAction.ACTION5:
            self._handle_action()

        if _board_is_zero(self.board):
            self._won = True
            self.next_level()
            self.complete_action()
            return

        if self.rem_moves <= 0 and not self._won:
            self._handle_death()
            self.complete_action()
            return

        level_def = LEVEL_DEFS[self.lidx]
        if level_def["min_val"] > 1 and not self._won and not _board_is_zero(self.board):
            if not _has_clickable_tile(self.board, level_def["min_val"], level_def["board_size"]):
                self._handle_death()
                self.complete_action()
                return

        self.complete_action()


ARC_PALETTE = np.array([
    [255, 255, 255],
    [204, 204, 204],
    [153, 153, 153],
    [102, 102, 102],
    [51,  51,  51],
    [0,   0,   0],
    [229, 58,  163],
    [255, 123, 204],
    [249, 60,  49],
    [30,  147, 255],
    [136, 216, 241],
    [255, 220, 0],
    [255, 133, 27],
    [146, 18,  49],
    [79,  204, 48],
    [163, 86,  208],
], dtype=np.uint8)

_COLOR_CHAR = {
    BLACK: "K",
    BLUE: "B",
    ORANGE: "O",
    RED: "R",
    WHITE: "W",
    GREEN: "G",
    MAGENTA: "M",
    YELLOW: "Y",
    PINK: "P",
    DKGREY: "D",
    GREY: "g",
}


class PuzzleEnvironment:

    _ACTION_MAP: Dict[str, GameAction] = {
        "reset":  GameAction.RESET,
        "up":     GameAction.ACTION1,
        "down":   GameAction.ACTION2,
        "left":   GameAction.ACTION3,
        "right":  GameAction.ACTION4,
        "select": GameAction.ACTION5,
        "undo":   GameAction.ACTION7,
    }

    _VALID_ACTIONS = list(_ACTION_MAP.keys())

    def __init__(self, seed: int = 0) -> None:
        self._engine = Eg01(seed=seed)
        self._total_turns = 0
        self._done = False
        self._game_won = False
        self._game_over = False
        self._last_action_was_reset = False

    @staticmethod
    def _encode_png(frame: np.ndarray) -> bytes:
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
        level_def = LEVEL_DEFS[e.lidx]
        board_size = level_def["board_size"]

        grid_lines = []
        for r in range(board_size):
            row_chars = []
            for c in range(board_size):
                val = e.board[r][c]
                color = TILE_COLOR[val]
                ch = _COLOR_CHAR.get(color, "?")
                if (r, c) == e.cursor:
                    row_chars.append("[" + ch + "]")
                else:
                    row_chars.append(" " + ch + " ")
            grid_lines.append("".join(row_chars))

        header = (
            f"Level {e.level_index + 1}/{NUM_LEVELS}"
            f" | Moves: {e.rem_moves}/{e.max_moves}"
            f" | Lives: {e._lives}"
        )
        rules = "Move cursor with arrows. Select clears tile and propagates XOR. Clear all tiles to win."
        return (
            header
            + "\n"
            + rules
            + "\nBoard:\n"
            + "\n".join(grid_lines)
        )

    def _build_game_state(self, done: bool = False) -> GameState:
        e = self._engine
        frame = self._render_frame()
        image_bytes = self._encode_png(frame)

        valid_actions = self.get_actions() if not done else None

        return GameState(
            text_observation=self._build_text_observation(),
            image_observation=image_bytes,
            valid_actions=valid_actions,
            turn=self._total_turns,
            metadata={
                "total_levels": len(e._levels),
                "level_index": e.level_index,
                "levels_completed": getattr(e, "_score", 0),
                "game_over": self._game_over,
                "done": done,
                "info": {},
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
        self._game_over = False

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
            raise ValueError(
                f"Invalid action '{action}'. Must be one of {list(self._ACTION_MAP.keys())}"
            )

        self._last_action_was_reset = False
        self._total_turns += 1

        game_action = self._ACTION_MAP[action]
        info: Dict[str, Any] = {"action": action}

        level_before = e._current_level_index

        action_input = ActionInput(id=game_action)
        frame = e.perform_action(action_input, raw=True)

        state_name = frame.state.name if frame and frame.state else ""
        game_won = state_name == "WIN"
        game_over = state_name == "GAME_OVER"

        total_levels = len(self._engine._levels)
        level_reward = 1.0 / total_levels

        if game_won:
            self._done = True
            self._game_won = True
            self._game_over = False
            info["reason"] = "game_complete"
            return StepResult(
                state=self._build_game_state(done=True),
                reward=level_reward,
                done=True,
                info=info,
            )

        if game_over or e._game_over:
            self._done = True
            self._game_won = False
            self._game_over = True
            info["reason"] = "game_over"
            return StepResult(
                state=self._build_game_state(done=True),
                reward=0.0,
                done=True,
                info=info,
            )

        reward = 0.0
        if e._current_level_index != level_before:
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
        index_grid = self._render_frame()
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
