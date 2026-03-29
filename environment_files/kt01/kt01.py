import struct
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from arcengine import ARCBaseGame, ActionInput, Camera, GameAction, Level, Sprite
from gymnasium import spaces

ARC_PALETTE = np.array(
    [
        [0, 0, 0],
        [0, 116, 217],
        [255, 65, 54],
        [46, 204, 64],
        [255, 220, 0],
        [170, 170, 170],
        [240, 18, 190],
        [255, 133, 27],
        [127, 219, 255],
        [135, 12, 37],
        [0, 48, 73],
        [106, 76, 48],
        [255, 182, 193],
        [80, 80, 80],
        [50, 205, 50],
        [128, 0, 128],
    ],
    dtype=np.uint8,
)

BACKGROUND_COLOR = 5
PADDING_COLOR = 3
COLOR_LIGHT_SQUARE = 0
COLOR_DARK_SQUARE = 3
COLOR_KNIGHT = 2
COLOR_VISITED = 8
COLOR_CURSOR = 11
CAM_SIZE = 16

KNIGHT_OFFSETS = [
    (-2, -1), (-2, 1), (-1, -2), (-1, 2),
    (1, -2), (1, 2), (2, -1), (2, 1),
]

_LEVELS = [
    {"n": 5, "bombs": []},
    {"n": 6, "bombs": []},
    {"n": 7, "bombs": [(3, 3)]},
    {"n": 8, "bombs": [(3, 5), (5, 2)]},
    {"n": 9, "bombs": [(2, 2), (5, 2), (7, 3)]},
]

MAX_LIVES = 3

ACTION_UP = "up"
ACTION_DOWN = "down"
ACTION_LEFT = "left"
ACTION_RIGHT = "right"
ACTION_SELECT = "select"
ACTION_UNDO = "undo"
ACTION_RESET_STR = "reset"

def _action_slot(action: object) -> int:
    if action is None:
        return -1
    v = getattr(action, "value", action)
    try:
        return int(v)
    except (TypeError, ValueError):
        return -1


S_UP = _action_slot(GameAction.ACTION1)
S_DOWN = _action_slot(GameAction.ACTION2)
S_LEFT = _action_slot(GameAction.ACTION3)
S_RIGHT = _action_slot(GameAction.ACTION4)
S_SELECT = _action_slot(GameAction.ACTION5)
S_UNDO = _action_slot(GameAction.ACTION7)
S_RESET = _action_slot(GameAction.RESET)

_INT_TO_STR = {
    S_UP: ACTION_UP,
    S_DOWN: ACTION_DOWN,
    S_LEFT: ACTION_LEFT,
    S_RIGHT: ACTION_RIGHT,
    S_SELECT: ACTION_SELECT,
    S_UNDO: ACTION_UNDO,
}

_PLAY_ACTION_SLOTS = frozenset(
    {S_UP, S_DOWN, S_LEFT, S_RIGHT, S_SELECT, S_UNDO}
)

ACTION_MAP = {
    ACTION_RESET_STR: getattr(GameAction, "RESET"),
    ACTION_UP: GameAction.ACTION1,
    ACTION_DOWN: GameAction.ACTION2,
    ACTION_LEFT: GameAction.ACTION3,
    ACTION_RIGHT: GameAction.ACTION4,
    ACTION_SELECT: GameAction.ACTION5,
    ACTION_UNDO: GameAction.ACTION7,
}

NUM_LEVELS = len(_LEVELS)
REWARD_PER_LEVEL = 1.0 / float(NUM_LEVELS)

_RESET_ACTIONS = tuple(
    x for x in (getattr(GameAction, "RESET", None),) if x is not None
)


def _is_reset_action(action: object) -> bool:
    if action in _RESET_ACTIONS:
        return True
    vid = getattr(action, "value", action)
    try:
        return int(vid) == S_RESET
    except (TypeError, ValueError):
        return False


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


def _create_pixel(color: int, layer: int = 0) -> Sprite:
    return Sprite(
        pixels=np.array([[color]], dtype=np.int32),
        name="px",
        visible=True,
        collidable=False,
        tags=[],
        layer=layer,
    )


def _rgb_frame_from_game(g: ARCBaseGame) -> np.ndarray:
    cam = g.camera
    lvl = g.current_level
    sprites_m = getattr(lvl, "get_sprites", None)
    sprites = sprites_m() if callable(sprites_m) else []
    render_m = getattr(cam, "render", None)
    if not callable(render_m):
        return np.zeros((64, 64, 3), dtype=np.uint8)
    index_grid = np.asarray(render_m(sprites), dtype=np.intp)
    clipped = np.clip(index_grid, 0, 15)
    return ARC_PALETTE[clipped]


def _encode_png_rgb(rgb: np.ndarray) -> bytes:
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("expected uint8 (H,W,3) RGB")
    h, w = int(rgb.shape[0]), int(rgb.shape[1])

    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    rows = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))
    compressed = zlib.compress(rows, 9)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", compressed)
        + chunk(b"IEND", b"")
    )


class _KnightsTourEngine:
    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self._current_level_index = 0
        self._n = 5
        self._bombs: List[Tuple[int, int]] = []
        self._lives = MAX_LIVES
        self._board = np.zeros((self._n, self._n), dtype=np.int32)
        self._visit_order: List[Tuple[int, int]] = []
        self._cursor_x = 0
        self._cursor_y = 0
        self._knight_x = 0
        self._knight_y = 0
        self._knight_placed = False
        self._turn = 0
        self._episode_done = False
        self._history: List[dict] = []
        self.load_level(0)

    def load_level(self, idx: int) -> None:
        self._current_level_index = min(max(0, idx), len(_LEVELS) - 1)
        data = _LEVELS[self._current_level_index]
        self._n = data["n"]
        self._bombs = list(data["bombs"])
        self._lives = MAX_LIVES
        self._board = np.zeros((self._n, self._n), dtype=np.int32)
        self._visit_order = []
        for bomb_row, bomb_col in self._bombs:
            self._board[bomb_row, bomb_col] = 2
        start_col = self._seed % self._n
        start_row = (self._seed // self._n) % self._n
        if self._board[start_row, start_col] == 2:
            start_row, start_col = 0, 0
        self._cursor_x = start_col
        self._cursor_y = start_row
        self._knight_x = 0
        self._knight_y = 0
        self._knight_placed = False
        self._turn = 0
        self._episode_done = False
        self._history.clear()

    def _push_snapshot(self) -> None:
        self._history.append(
            {
                "board": self._board.copy(),
                "visit_order": list(self._visit_order),
                "cursor_x": self._cursor_x,
                "cursor_y": self._cursor_y,
                "knight_x": self._knight_x,
                "knight_y": self._knight_y,
                "knight_placed": self._knight_placed,
                "lives": self._lives,
            }
        )

    def _pop_undo(self) -> bool:
        if not self._history:
            return False
        snap = self._history.pop()
        self._board = snap["board"].copy()
        self._visit_order = list(snap["visit_order"])
        self._cursor_x = snap["cursor_x"]
        self._cursor_y = snap["cursor_y"]
        self._knight_x = snap["knight_x"]
        self._knight_y = snap["knight_y"]
        self._knight_placed = snap["knight_placed"]
        self._lives = snap["lives"]
        self._episode_done = False
        return True

    @property
    def current_level_index(self) -> int:
        return self._current_level_index

    @property
    def n(self) -> int:
        return self._n

    @property
    def offset_x(self) -> int:
        return (CAM_SIZE - self._n) // 2

    @property
    def offset_y(self) -> int:
        return (CAM_SIZE - self._n) // 2

    @property
    def cursor_x(self) -> int:
        return self._cursor_x

    @property
    def cursor_y(self) -> int:
        return self._cursor_y

    @property
    def knight_x(self) -> int:
        return self._knight_x

    @property
    def knight_y(self) -> int:
        return self._knight_y

    @property
    def knight_placed(self) -> bool:
        return self._knight_placed

    @property
    def lives(self) -> int:
        return self._lives

    @property
    def board(self) -> np.ndarray:
        return self._board

    @property
    def visit_order(self) -> List[Tuple[int, int]]:
        return self._visit_order

    @property
    def episode_done(self) -> bool:
        return self._episode_done

    @property
    def moves_used(self) -> int:
        return self._turn

    def _text_observation(self) -> str:
        lines = [
            f"level={self._current_level_index + 1}/{len(_LEVELS)} "
            f"grid={self._n}x{self._n}",
            "legend: .=empty V=visited B=bomb",
        ]
        for row in range(self._n):
            chars = []
            for col in range(self._n):
                v = int(self._board[row, col])
                if v == 2:
                    chars.append("B")
                elif v == 1:
                    chars.append("V")
                else:
                    chars.append(".")
            lines.append("".join(chars))
        kx, ky = self._knight_x, self._knight_y
        cx, cy = self._cursor_x, self._cursor_y
        lines.append(
            f"lives={self._lives} moves_used={self._turn} "
            f"knight_placed={self._knight_placed} cursor=({cx},{cy}) "
            f"knight_cell=({kx},{ky})"
        )
        return "\n".join(lines)

    def _valid_action_strings(self) -> list[str]:
        out: list[str] = []
        if self._cursor_y > 0:
            out.append(ACTION_UP)
        if self._cursor_y < self._n - 1:
            out.append(ACTION_DOWN)
        if self._cursor_x > 0:
            out.append(ACTION_LEFT)
        if self._cursor_x < self._n - 1:
            out.append(ACTION_RIGHT)
        out.append(ACTION_SELECT)
        out.append(ACTION_UNDO)
        return out

    def _is_valid_knight_move(
        self, from_row: int, from_col: int, to_row: int, to_col: int
    ) -> bool:
        row_diff = abs(to_row - from_row)
        col_diff = abs(to_col - from_col)
        return (row_diff == 2 and col_diff == 1) or (row_diff == 1 and col_diff == 2)

    def _mark_visited(self, row: int, col: int) -> None:
        if self._board[row, col] == 2:
            return
        self._board[row, col] = 1
        self._visit_order.append((row, col))

    def _tour_finished(self) -> bool:
        required = self._n * self._n - len(self._bombs)
        return len(self._visit_order) == required

    def _has_any_valid_move(self) -> bool:
        if not self._knight_placed:
            return True
        for delta_row, delta_col in KNIGHT_OFFSETS:
            row = self._knight_y + delta_row
            col = self._knight_x + delta_col
            if 0 <= row < self._n and 0 <= col < self._n and self._board[row, col] == 0:
                return True
        return False

    def _reset_board(self) -> None:
        self._board = np.zeros((self._n, self._n), dtype=np.int32)
        for bomb_row, bomb_col in self._bombs:
            self._board[bomb_row, bomb_col] = 2
        self._visit_order = []
        self._cursor_x = 0
        self._cursor_y = 0
        self._knight_placed = False

    def _trigger_life_loss(self) -> bool:
        self._lives -= 1
        if self._lives <= 0:
            self._episode_done = True
            return True
        self._reset_board()
        return False

    def _handle_cursor_move_str(self, action: str) -> None:
        if action == ACTION_UP and self._cursor_y > 0:
            self._cursor_y -= 1
        elif action == ACTION_DOWN and self._cursor_y < self._n - 1:
            self._cursor_y += 1
        elif action == ACTION_LEFT and self._cursor_x > 0:
            self._cursor_x -= 1
        elif action == ACTION_RIGHT and self._cursor_x < self._n - 1:
            self._cursor_x += 1

    def _handle_cursor_move_slot(self, aid: int) -> None:
        if aid == S_UP and self._cursor_y > 0:
            self._cursor_y -= 1
        elif aid == S_DOWN and self._cursor_y < self._n - 1:
            self._cursor_y += 1
        elif aid == S_LEFT and self._cursor_x > 0:
            self._cursor_x -= 1
        elif aid == S_RIGHT and self._cursor_x < self._n - 1:
            self._cursor_x += 1

    def _place_knight(self) -> str | None:
        if self._board[self._cursor_y, self._cursor_x] == 2:
            return None
        self._knight_x = self._cursor_x
        self._knight_y = self._cursor_y
        self._knight_placed = True
        self._mark_visited(self._knight_y, self._knight_x)
        if self._tour_finished():
            return "win_level"
        if not self._has_any_valid_move():
            if self._trigger_life_loss():
                return "lose"
            return None
        return None

    def _move_knight(self) -> str | None:
        is_valid = self._is_valid_knight_move(
            self._knight_y,
            self._knight_x,
            self._cursor_y,
            self._cursor_x,
        )
        if not is_valid:
            return None
        target = self._board[self._cursor_y, self._cursor_x]
        if target == 2:
            if self._trigger_life_loss():
                return "lose"
            return None
        if target != 0:
            return None
        self._knight_x = self._cursor_x
        self._knight_y = self._cursor_y
        self._mark_visited(self._knight_y, self._knight_x)
        if self._tour_finished():
            return "win_level"
        if not self._has_any_valid_move():
            if self._trigger_life_loss():
                return "lose"
            return None
        return None

    def step_int(self, action: object) -> tuple[float, bool, dict]:
        reward = 0.0
        done = False
        info: dict = {}
        aid = _action_slot(action)
        if aid == S_UNDO:
            self._turn += 1
            self._pop_undo()
            return reward, done, info
        if self._episode_done:
            return 0.0, True, {}
        if aid in (S_UP, S_DOWN, S_LEFT, S_RIGHT):
            self._push_snapshot()
            self._turn += 1
            self._handle_cursor_move_slot(aid)
        elif aid == S_SELECT:
            self._push_snapshot()
            self._turn += 1
            if not self._knight_placed:
                outcome = self._place_knight()
            else:
                outcome = self._move_knight()
            if outcome == "win_level":
                info = {"outcome": "win_level"}
            elif outcome == "lose":
                reward = 0.0
                done = True
                self._episode_done = True
                info = {"outcome": "lose"}
        return reward, done, info


class Kt01(ARCBaseGame):
    MAX_LIVES = MAX_LIVES

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self._acted_since_reset = False
        self._reset_press_count = 0
        self._kt01_slot_level_idx: int | None = None
        self._engine = _KnightsTourEngine(seed=seed)
        levels = [
            Level(sprites=[], grid_size=(CAM_SIZE, CAM_SIZE), data=d)
            for d in _LEVELS
        ]
        vw, vh = levels[0].grid_size
        camera = Camera(
            x=0,
            y=0,
            width=vw,
            height=vh,
            background=BACKGROUND_COLOR,
            letter_box=PADDING_COLOR,
            interfaces=[],
        )
        self._visited_sprites: Dict[Tuple[int, int], Sprite] = {}
        self._cursor_sprite: Sprite | None = None
        self._knight_sprite: Sprite | None = None
        self._life_sprites: List[Sprite] = []
        super().__init__(
            "kt01",
            levels,
            camera,
            available_actions=[0, 1, 2, 3, 4, 5, 7],
        )

    def _get_valid_actions(self) -> list[ActionInput]:
        valid_actions: list[ActionInput] = []
        for aid in self._available_actions:
            if aid == S_UNDO:
                valid_actions.append(ActionInput(id=GameAction.ACTION7))
            elif aid in (S_UP, S_DOWN, S_LEFT, S_RIGHT, S_SELECT):
                valid_actions.append(ActionInput(id=GameAction.from_id(aid)))
        return valid_actions

    @property
    def lives(self) -> int:
        return self._engine.lives

    @property
    def level_index(self) -> int:
        return self._current_level_index

    def on_set_level(self, level: Level) -> None:
        self._acted_since_reset = False
        cur = self._current_level_index
        prev = self._kt01_slot_level_idx
        if prev is not None and cur != prev:
            self._reset_press_count = 0
        self._kt01_slot_level_idx = cur
        self._engine.load_level(self._current_level_index)
        self._visited_sprites.clear()
        self._cursor_sprite = None
        self._knight_sprite = None
        self._life_sprites.clear()
        self._build_board()
        self._cursor_sprite = _create_pixel(COLOR_CURSOR, layer=10)
        self._update_cursor_position()
        self.current_level.add_sprite(self._cursor_sprite)
        self._knight_sprite = _create_pixel(COLOR_KNIGHT, layer=15)
        self._knight_sprite.set_visible(False)
        self.current_level.add_sprite(self._knight_sprite)
        for i in range(self.MAX_LIVES):
            life_sprite = _create_pixel(COLOR_VISITED, layer=25)
            life_sprite.set_position(CAM_SIZE - 3 + i, 0)
            life_sprite.set_visible(True)
            self.current_level.add_sprite(life_sprite)
            self._life_sprites.append(life_sprite)

    def _build_board(self) -> None:
        n = self._engine.n
        ox = self._engine.offset_x
        oy = self._engine.offset_y
        for row in range(n):
            for col in range(n):
                color = COLOR_LIGHT_SQUARE if (row + col) % 2 == 0 else COLOR_DARK_SQUARE
                square = _create_pixel(color, layer=0)
                square.set_position(ox + col, oy + row)
                self.current_level.add_sprite(square)
                visited_marker = _create_pixel(COLOR_VISITED, layer=5)
                visited_marker.set_position(ox + col, oy + row)
                visited_marker.set_visible(False)
                self.current_level.add_sprite(visited_marker)
                self._visited_sprites[(row, col)] = visited_marker

    def _update_cursor_position(self) -> None:
        if self._cursor_sprite is None:
            return
        ox = self._engine.offset_x
        oy = self._engine.offset_y
        self._cursor_sprite.set_position(
            ox + self._engine.cursor_x,
            oy + self._engine.cursor_y,
        )

    def _update_knight_position(self) -> None:
        if self._knight_sprite is None:
            return
        ox = self._engine.offset_x
        oy = self._engine.offset_y
        self._knight_sprite.set_position(
            ox + self._engine.knight_x,
            oy + self._engine.knight_y,
        )

    def _sync_visited_sprites(self) -> None:
        for (row, col), sprite in self._visited_sprites.items():
            sprite.set_visible(self._engine.board[row, col] == 1)

    def _sync_life_sprites(self) -> None:
        for i, heart in enumerate(self._life_sprites):
            heart.set_visible(i < self._engine.lives)

    def handle_reset(self) -> None:
        self.level_reset()

    def step(self) -> None:
        action = self.action.id
        if _is_reset_action(action):
            term = bool(getattr(self, "_terminated", False))
            win_restart = term and self._engine.lives > 0
            if win_restart:
                self._current_level_index = 0
                self._reset_press_count = 0
            else:
                if self._acted_since_reset:
                    self._reset_press_count = 0
                self._reset_press_count += 1
                if self._reset_press_count >= 2:
                    self._current_level_index = 0
                    self._reset_press_count = 0
                self._acted_since_reset = False
            self.handle_reset()
            self.complete_action()
            return
        self._acted_since_reset = True
        aid = _action_slot(action)
        if aid in _PLAY_ACTION_SLOTS:
            _, done_flag, info = self._engine.step_int(action)
            if info.get("outcome") == "win_level":
                self.next_level()
            elif done_flag and info.get("outcome") == "lose":
                self.lose()
            else:
                if self._knight_sprite:
                    self._knight_sprite.set_visible(self._engine.knight_placed)
                self._update_knight_position()
                self._update_cursor_position()
                self._sync_visited_sprites()
                self._sync_life_sprites()
            self.complete_action()
        else:
            self.complete_action()


class PuzzleEnvironment:
    def __init__(self, seed: int = 0) -> None:
        self._engine = Kt01(seed)
        self.seed = seed
        self._total_turns = 0
        self._done = False
        self._last_action_was_reset = False

    @property
    def game(self) -> Kt01 | None:
        return self._engine

    def _episode_terminal(self) -> bool:
        if self._engine is None:
            return True
        g = self._engine
        return g.lives <= 0 or bool(getattr(g, "_terminated", False))

    def reset(self) -> GameState:
        if self._engine is None:
            self._engine = Kt01(self.seed)
        self._engine.full_reset()
        self._total_turns = 0
        self._done = False
        self._last_action_was_reset = True
        return self._create_game_state()

    def get_actions(self) -> list[str]:
        if self._engine is None:
            return [ACTION_RESET_STR]
        if self._done:
            return [ACTION_RESET_STR]
        if not hasattr(self._engine, "lives") or self._engine.lives <= 0:
            return [ACTION_RESET_STR]
        return [*self._engine._engine._valid_action_strings(), ACTION_RESET_STR]

    def _outcome_after_step(
        self, lives_before: int, level_before: int
    ) -> Tuple[float, bool, dict]:
        g = self._engine
        info: dict = {
            "lives": g.lives,
            "level": g.level_index + 1,
            "moves_used": g._engine.moves_used,
        }
        reward = 0.0
        done = False
        last_level_idx = NUM_LEVELS - 1
        if g.lives < lives_before:
            info["event"] = "life_lost"
            if g.lives <= 0:
                info["event"] = "game_over"
                done = True
        elif bool(getattr(g, "_terminated", False)) and level_before == last_level_idx:
            reward = REWARD_PER_LEVEL
            info["event"] = "game_complete"
            done = True
        elif g.level_index != level_before:
            reward = REWARD_PER_LEVEL
            if level_before == last_level_idx:
                info["event"] = "game_complete"
                done = True
            else:
                info["event"] = "level_complete"
        elif bool(getattr(g, "_terminated", False)):
            reward = REWARD_PER_LEVEL
            done = True
            info["event"] = "game_complete"
        return reward, done, info

    def step(self, action: str) -> StepResult:
        if self._engine is None:
            return StepResult(
                state=GameState(
                    text_observation="",
                    image_observation=None,
                    valid_actions=None,
                    turn=0,
                ),
                reward=0.0,
                done=True,
                info={"error": "environment closed"},
            )
        if action == ACTION_RESET_STR:
            ra = getattr(GameAction, "RESET", None)
            if ra is None:
                return StepResult(
                    state=self._create_game_state(),
                    reward=0.0,
                    done=self.is_done(),
                    info={"error": "reset unavailable"},
                )
            self._engine._set_action(ActionInput(id=ra))
            self._engine.step()
            self._last_action_was_reset = True
            self._done = False
            return StepResult(
                state=self._create_game_state(),
                reward=0.0,
                done=False,
                info={
                    "event": "reset",
                    "lives": self._engine.lives,
                    "level": self._engine.level_index + 1,
                    "moves_used": self._engine._engine.moves_used,
                },
            )
        if action not in ACTION_MAP:
            return StepResult(
                state=self._create_game_state(),
                reward=0.0,
                done=self.is_done(),
                info={"error": f"Invalid action: {action}"},
            )

        self._last_action_was_reset = False
        lives_before = self._engine.lives
        level_before = self._engine.level_index

        self._engine._set_action(ActionInput(id=ACTION_MAP[action]))
        self._engine.step()
        self._total_turns += 1

        reward, done, info = self._outcome_after_step(lives_before, level_before)
        self._done = done or self._episode_terminal()

        return StepResult(
            state=self._create_game_state(),
            reward=reward,
            done=self._done,
            info=info,
        )

    def is_done(self) -> bool:
        if self._engine is None:
            return True
        return self._done or self._episode_terminal()

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if self._engine is None:
            raise RuntimeError("environment closed")
        if mode != "rgb_array":
            raise ValueError(f"Unsupported render mode: {mode}")
        return _rgb_frame_from_game(self._engine)

    def close(self) -> None:
        self._engine = None

    def _create_game_state(self) -> GameState:
        if self._engine is None:
            return GameState(
                text_observation="",
                image_observation=None,
                valid_actions=None,
                turn=self._total_turns,
            )
        g = self._engine
        lines = [
            f"Level {g.level_index + 1}/{NUM_LEVELS}"
            f" | Lives: {g.lives}"
            f" | Moves used: {g._engine.moves_used}"
            f" | Turn: {self._total_turns}",
            "",
        ]
        lines.append(g._engine._text_observation())
        lines.append("")
        lines.append(
            f"Actions: {ACTION_UP}, {ACTION_DOWN}, {ACTION_LEFT}, "
            f"{ACTION_RIGHT}, {ACTION_SELECT}, {ACTION_UNDO}, {ACTION_RESET_STR}"
        )
        terminal = self._done or g.lives <= 0
        va = (
            None
            if terminal
            else [*g._engine._valid_action_strings(), ACTION_RESET_STR]
        )
        rgb = _rgb_frame_from_game(g)
        img = _encode_png_rgb(rgb)
        return GameState(
            text_observation="\n".join(lines),
            image_observation=img,
            valid_actions=va,
            turn=self._total_turns,
            metadata={
                "total_levels": len(self._engine._levels),
                "level": g.level_index + 1,
                "lives": g.lives,
                "moves_used": g._engine.moves_used,
                "n": g._engine.n,
                "knight_placed": g._engine.knight_placed,
                "cursor_x": g._engine.cursor_x,
                "cursor_y": g._engine.cursor_y,
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

    OBS_HEIGHT: int = 16
    OBS_WIDTH: int = 16

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
