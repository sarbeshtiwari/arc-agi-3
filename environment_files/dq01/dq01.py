import random
import struct
import zlib
from collections import deque
from dataclasses import dataclass, field
from itertools import islice
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from arcengine import (
    ARCBaseGame,
    ActionInput,
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


FRAME_SIZE = 64
UI_ROWS = 2

BACKGROUND_COLOR = 0
PADDING_COLOR = 0

LETTER_TO_COLOR = {
    "A": 2,
    "B": 3,
    "C": 6,
    "D": 8,
}
LETTERS = tuple(LETTER_TO_COLOR.keys())

SEQUENCE_BY_LEVEL = {
    1: ("A", "C", "B", "D", "B", "A", "D", "C"),
    2: ("A", "C", "A", "B", "D", "B", "C", "D", "A", "C", "B", "D"),
    3: ("A", "C", "B", "D", "B", "A", "D", "C", "A", "B", "C", "D", "C", "A", "D", "B"),
    4: (
        "A",
        "B",
        "A",
        "A",
        "B",
        "C",
        "A",
        "D",
        "A",
        "B",
        "B",
        "C",
        "B",
        "D",
        "C",
        "C",
        "D",
        "C",
        "D",
        "D",
    ),
}

PREVIEW_BY_STAGE = {
    1: 7,
    2: 5,
    3: 3,
    4: 1,
}

MOVES_BY_LEVEL = {
    1: 220,
    2: 400,
    3: 1840,
    4: 2640,
}

C_CURSOR = 9
C_BAR_FILL = 8
C_BAR_EMPTY = 5
C_LIFE = 7
C_LIFE_EMPTY = 3

CURSOR_START_POSITIONS = {
    1: [(6, 10), (3, 10), (9, 10), (6, 8)],
    2: [(8, 14), (4, 14), (12, 14), (8, 12)],
    3: [(12, 22), (6, 22), (18, 22), (12, 20)],
    4: [(16, 30), (8, 30), (24, 30), (16, 28)],
}


class HUDDisplay(RenderableUserDisplay):
    MAX_LIVES = 3

    def __init__(self, game: "Dq01") -> None:
        self._game = game

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        max_moves = self._game._max_moves
        if max_moves == 0:
            return frame

        fw = frame.shape[1]
        fh = frame.shape[0]
        row0 = fh - UI_ROWS

        frame[row0:fh, :] = C_BAR_EMPTY
        bar_w = int(fw * 0.7)
        filled = int(bar_w * self._game._moves_left / max_moves)
        for x in range(filled):
            frame[row0:fh, x] = C_BAR_FILL

        lives = self._game._lives
        lx_start = bar_w
        lx_width = fw - bar_w
        blk_w = 2
        gap = 2
        total_lw = blk_w * self.MAX_LIVES + gap * (self.MAX_LIVES - 1)
        offset = (lx_width - total_lw) // 2
        for i in range(self.MAX_LIVES):
            x0 = lx_start + offset + i * (blk_w + gap)
            x1 = x0 + blk_w
            color = C_LIFE if i < lives else C_LIFE_EMPTY
            if x1 <= fw:
                frame[row0:fh, x0:x1] = color

        gw, gh = self._game.current_level.grid_size
        cx = self._game._cursor_x
        cy = self._game._cursor_y
        scale_x = fw // gw
        scale_y = fh // gh
        ox = (fw - scale_x * gw) // 2
        oy = (fh - scale_y * gh) // 2
        px = ox + cx * scale_x
        py = max(0, oy + cy * scale_y)
        px_end = min(px + scale_x, fw)
        py_end = min(py + scale_y, row0)

        if py_end > py and px_end > px:
            frame[py, px:px_end] = C_CURSOR
            frame[py_end - 1, px:px_end] = C_CURSOR
            frame[py:py_end, px] = C_CURSOR
            frame[py:py_end, px_end - 1] = C_CURSOR

        return frame


sprites = {
    "token-A": Sprite(
        pixels=[[LETTER_TO_COLOR["A"]]], name="token-A", visible=True, collidable=True
    ),
    "token-B": Sprite(
        pixels=[[LETTER_TO_COLOR["B"]]], name="token-B", visible=True, collidable=True
    ),
    "token-C": Sprite(
        pixels=[[LETTER_TO_COLOR["C"]]], name="token-C", visible=True, collidable=True
    ),
    "token-D": Sprite(
        pixels=[[LETTER_TO_COLOR["D"]]], name="token-D", visible=True, collidable=True
    ),
    "bin-A": Sprite(
        pixels=[[LETTER_TO_COLOR["A"]]], name="bin-A", visible=True, collidable=True
    ),
    "bin-B": Sprite(
        pixels=[[LETTER_TO_COLOR["B"]]], name="bin-B", visible=True, collidable=True
    ),
    "bin-C": Sprite(
        pixels=[[LETTER_TO_COLOR["C"]]], name="bin-C", visible=True, collidable=True
    ),
    "bin-D": Sprite(
        pixels=[[LETTER_TO_COLOR["D"]]], name="bin-D", visible=True, collidable=True
    ),
    "stack-zone": Sprite(
        pixels=[[14]], name="stack-zone", visible=True, collidable=True
    ),
}

levels = [
    Level(sprites=[], grid_size=(12, 12)),
    Level(sprites=[], grid_size=(16, 16)),
    Level(sprites=[], grid_size=(24, 24)),
    Level(sprites=[], grid_size=(32, 32)),
]


class Dq01(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self._rng = random.Random(seed)

        self._queue: deque = deque()
        self._stack: list = []
        self._remaining = {letter: 0 for letter in LETTERS}
        self._target_idx = 0
        self._stage_num = 1

        self._moves_left = 0
        self._max_moves = 0
        self._lives = 3

        self._cursor_x = 0
        self._cursor_y = 0

        self._history: List[Dict] = []

        self._hud = HUDDisplay(self)
        game_levels = list(levels)
        camera = Camera(
            0, 0, FRAME_SIZE, FRAME_SIZE, BACKGROUND_COLOR, PADDING_COLOR, [self._hud]
        )
        super().__init__(
            game_id="dq01",
            levels=game_levels,
            camera=camera,
            available_actions=[0, 1, 2, 3, 4, 5, 7],
        )

    def on_set_level(self, level: Level) -> None:
        self._lives = 3
        self._history = []

        stage = None
        for idx, known_level in enumerate(self._levels):
            if known_level is level:
                stage = idx + 1
                break
        if stage is None:
            stage = 1
        self._stage_num = max(1, min(4, stage))

        self._max_moves = MOVES_BY_LEVEL[self._stage_num]
        self._moves_left = self._max_moves

        start_x, start_y = self._rng.choice(CURSOR_START_POSITIONS[self._stage_num])
        self._cursor_x = start_x
        self._cursor_y = start_y

        self._load_sequence()
        self._render_state()

    def _load_sequence(self) -> None:
        self._queue.clear()
        self._stack.clear()
        self._remaining = {letter: 0 for letter in LETTERS}
        self._target_idx = 0
        self._history = []

        for letter in SEQUENCE_BY_LEVEL[self._stage_num]:
            self._queue.append(letter)
            self._remaining[letter] += 1

        self._advance_block_target()

    def _advance_block_target(self) -> None:
        while self._target_idx < len(LETTERS):
            if self._remaining[LETTERS[self._target_idx]] > 0:
                return
            self._target_idx += 1

    def _get_current_target(self):
        stage = self._stage_num

        if stage == 3:
            if sum(self._remaining.values()) == 0:
                return None
            if self._remaining[LETTERS[self._target_idx]] == 0:
                for shift in range(1, len(LETTERS) + 1):
                    nxt = (self._target_idx + shift) % len(LETTERS)
                    if self._remaining[LETTERS[nxt]] > 0:
                        return LETTERS[nxt]
            return LETTERS[self._target_idx]

        if stage >= 4:
            if self._target_idx >= len(LETTERS):
                return None
            return LETTERS[self._target_idx]

        return None

    def _advance_target_after_consume(self, token: str) -> None:
        stage = self._stage_num
        self._remaining[token] = max(0, self._remaining[token] - 1)

        if stage == 3:
            for shift in range(1, len(LETTERS) + 1):
                nxt = (self._target_idx + shift) % len(LETTERS)
                if self._remaining[LETTERS[nxt]] > 0:
                    self._target_idx = nxt
                    return
        elif stage >= 4:
            self._advance_block_target()

    def _restore_stack(self) -> None:
        if self._stage_num <= 3:
            if self._stack:
                self._queue.appendleft(self._stack.pop())
            return

        while self._stack:
            self._queue.appendleft(self._stack.pop())

    def _render_state(self) -> None:
        level = self.current_level
        for sp in list(level._sprites):
            level.remove_sprite(sp)

        self._build_static_board()
        self._place_target_sprite()
        self._place_queue_preview()
        self._place_stack_display()

    def _build_static_board(self) -> None:
        grid_w, grid_h = self.current_level.grid_size
        y_bin = grid_h - 2
        x_positions = [
            1,
            max(2, grid_w // 4),
            max(3, grid_w // 2),
            max(4, (3 * grid_w) // 4),
        ]
        stack_x = grid_w - 1
        bin_xs = [min(xp, stack_x - 1) for xp in x_positions]

        for letter, bx in zip(LETTERS, bin_xs):
            self.current_level.add_sprite(
                sprites[f"bin-{letter}"].clone().set_position(bx, y_bin)
            )
        self.current_level.add_sprite(
            sprites["stack-zone"].clone().set_position(stack_x, y_bin)
        )

    def _place_target_sprite(self) -> None:
        _, grid_h = self.current_level.grid_size
        target = self._get_current_target()
        if target is not None:
            self.current_level.add_sprite(
                sprites[f"token-{target}"].clone().set_position(1, max(1, grid_h // 3))
            )

    def _place_queue_preview(self) -> None:
        grid_w, grid_h = self.current_level.grid_size

        if self._queue:
            front = self._queue[0]
            self.current_level.add_sprite(
                sprites[f"token-{front}"]
                .clone()
                .set_position(grid_w // 2, max(1, grid_h // 3))
            )

        preview_count = min(PREVIEW_BY_STAGE[self._stage_num], max(1, grid_w - 2))
        for idx, letter in enumerate(islice(self._queue, preview_count)):
            self.current_level.add_sprite(
                sprites[f"token-{letter}"].clone().set_position(1 + idx, 1)
            )

    def _place_stack_display(self) -> None:
        grid_w, grid_h = self.current_level.grid_size
        if self._stack:
            top = self._stack[-1]
            self.current_level.add_sprite(
                sprites[f"token-{top}"]
                .clone()
                .set_position(grid_w - 2, max(1, grid_h - 4))
            )

    def _sprite_name_at(self, gx: int, gy: int) -> str:
        for sp in self.current_level._sprites:
            sx, sy = sp.x, sp.y
            sw = len(sp.pixels[0]) if sp.pixels else 1
            sh = len(sp.pixels) if sp.pixels else 1
            if sx <= gx < sx + sw and sy <= gy < sy + sh:
                return sp.name
        return ""

    def _clamp_cursor(self) -> None:
        grid_w, grid_h = self.current_level.grid_size
        self._cursor_x = max(0, min(grid_w - 1, self._cursor_x))
        self._cursor_y = max(0, min(grid_h - 1, self._cursor_y))

    def _resolve_stage1(self, chosen_letter: str, token: str) -> None:
        if chosen_letter == token:
            self._advance_target_after_consume(token)
        else:
            self._stack.append(token)

    def _resolve_stage2(self, chosen_letter: str, token: str) -> None:
        if chosen_letter == token:
            self._advance_target_after_consume(token)
        else:
            self._stack.append(token)
            if self._queue:
                self._queue.append(self._queue.popleft())

    def _resolve_targeted_stage(self, chosen_letter: str, token: str) -> None:
        target = self._get_current_target()
        if chosen_letter == token and token == target:
            self._advance_target_after_consume(token)
        else:
            self._stack.append(token)

    def _resolve_click(self, sprite_name: str) -> None:
        if sprite_name == "stack-zone":
            self._restore_stack()
            return

        if not self._queue:
            return

        if not sprite_name.startswith("bin-"):
            return

        chosen_letter = sprite_name.split("-")[-1]
        token = self._queue.popleft()

        if self._stage_num == 1:
            self._resolve_stage1(chosen_letter, token)
        elif self._stage_num == 2:
            self._resolve_stage2(chosen_letter, token)
        else:
            self._resolve_targeted_stage(chosen_letter, token)

    def _check_win(self) -> bool:
        return len(self._queue) == 0 and len(self._stack) == 0

    def _restart_level(self):
        self._lives -= 1
        if self._lives <= 0:
            self.lose()
            return
        self._load_sequence()
        self._moves_left = self._max_moves
        start_x, start_y = self._rng.choice(CURSOR_START_POSITIONS[self._stage_num])
        self._cursor_x = start_x
        self._cursor_y = start_y
        self._render_state()

    def _save_state(self) -> None:
        self._history.append(
            {
                "queue": list(self._queue),
                "stack": list(self._stack),
                "remaining": dict(self._remaining),
                "target_idx": self._target_idx,
                "moves_left": self._moves_left,
                "cursor_x": self._cursor_x,
                "cursor_y": self._cursor_y,
            }
        )

    def _undo(self) -> None:
        if not self._history:
            return
        snap = self._history.pop()
        self._queue = deque(snap["queue"])
        self._stack = snap["stack"]
        self._remaining = snap["remaining"]
        self._target_idx = snap["target_idx"]
        self._moves_left = snap["moves_left"]
        self._cursor_x = snap["cursor_x"]
        self._cursor_y = snap["cursor_y"]
        self._render_state()

    def step(self) -> None:
        if not self.action:
            self.complete_action()
            return

        action_id = self.action.id

        if action_id == GameAction.RESET:
            self.complete_action()
            return

        if action_id == GameAction.ACTION7:
            self._undo()
            self.complete_action()
            return

        self._save_state()
        self._moves_left -= 1

        if action_id == GameAction.ACTION1:
            self._cursor_y -= 1
            self._clamp_cursor()
        elif action_id == GameAction.ACTION2:
            self._cursor_y += 1
            self._clamp_cursor()
        elif action_id == GameAction.ACTION3:
            self._cursor_x -= 1
            self._clamp_cursor()
        elif action_id == GameAction.ACTION4:
            self._cursor_x += 1
            self._clamp_cursor()
        elif action_id == GameAction.ACTION5:
            self._do_click(self._cursor_x, self._cursor_y)

        if self._moves_left <= 0 and not self._check_win():
            self._restart_level()
            self.complete_action()
            return

        self._render_state()

        if self._check_win():
            self.next_level()

        self.complete_action()

    def _do_click(self, gx: int, gy: int) -> None:
        hit_name = self._sprite_name_at(gx, gy)
        if not hit_name:
            return
        if hit_name.startswith("bin-") and not self._queue:
            return
        if hit_name == "stack-zone" and not self._stack:
            return
        self._resolve_click(hit_name)

    def handle_reset(self):
        self._rng = random.Random(self._seed)
        self._lives = 3
        self._history = []
        super().handle_reset()

    def get_actions(self) -> list:
        return [
            GameAction.ACTION1,
            GameAction.ACTION2,
            GameAction.ACTION3,
            GameAction.ACTION4,
            GameAction.ACTION5,
        ]


class PuzzleEnvironment:
    ACTION_MAP = {
        "reset": 0,
        "up": 1,
        "down": 2,
        "left": 3,
        "right": 4,
        "select": 5,
        "undo": 7,
    }

    ARC_PALETTE = [
        (255, 255, 255),
        (204, 204, 204),
        (153, 153, 153),
        (102, 102, 102),
        (51, 51, 51),
        (0, 0, 0),
        (229, 58, 163),
        (255, 123, 204),
        (249, 60, 49),
        (30, 147, 255),
        (136, 216, 241),
        (255, 220, 0),
        (255, 133, 27),
        (146, 18, 49),
        (79, 204, 48),
        (163, 86, 208),
    ]

    def __init__(self, seed: int = 0) -> None:
        self._engine = Dq01(seed=seed)
        self._total_turns: int = 0
        self._prev_score: int = 0
        self._last_action_was_reset: bool = True

    def _build_text_obs(self) -> str:
        e = self._engine
        preview_count = PREVIEW_BY_STAGE.get(e._stage_num, 3)
        queue_items = list(islice(e._queue, preview_count))
        queue_str = " ".join(queue_items) if queue_items else "(empty)"
        front = e._queue[0] if e._queue else "-"

        stack_str = " ".join(e._stack) if e._stack else "(empty)"
        top = e._stack[-1] if e._stack else "-"

        target = e._get_current_target()
        target_str = target if target else "none"

        remaining_str = " ".join(f"{k}:{v}" for k, v in e._remaining.items() if v > 0)
        if not remaining_str:
            remaining_str = "all done"

        header = (
            f"Level:{e.level_index + 1}/{len(e._levels)} "
            f"Moves:{e._moves_left}/{e._max_moves} "
            f"Lives:{e._lives}"
        )
        lines = [
            header,
            f"Target:{target_str} Front:{front}",
            f"Queue[{len(e._queue)}]:{queue_str}",
            f"Stack[{len(e._stack)}]:{stack_str} Top:{top}",
            f"Remaining:{remaining_str}",
            f"Bins: A B C D  Stack-zone",
            f"Cursor:({e._cursor_x},{e._cursor_y})",
        ]
        return "\n".join(lines)

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
        index_grid = e.camera.render(e.current_level.get_sprites())
        if index_grid is None or index_grid.size == 0:
            return None
        h, w = index_grid.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for idx, color in enumerate(self.ARC_PALETTE):
            mask = index_grid == idx
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
                "levels_completed": getattr(e, "_score", 0),
                "game_over": getattr(e._state, "name", "") == "GAME_OVER",
                "lives": e._lives,
                "moves_remaining": e._moves_left,
                "moves_max": e._max_moves,
                "done": done,
                "info": info or {},
            },
        )

    def reset(self) -> GameState:
        e = self._engine
        reset_input = ActionInput(id=GameAction.RESET)
        e.perform_action(reset_input)
        self._total_turns = 0
        self._prev_score = getattr(e, "_score", 0)
        self._last_action_was_reset = True
        return self._build_game_state()

    def get_actions(self) -> List[str]:
        e = self._engine
        state_name = getattr(e._state, "name", "")
        if state_name in ("WIN", "GAME_OVER"):
            return ["reset"]
        return ["reset", "up", "down", "left", "right", "select", "undo"]

    def is_done(self) -> bool:
        e = self._engine
        state_name = getattr(e._state, "name", "")
        return state_name == "WIN"

    def step(self, action: str) -> StepResult:
        e = self._engine

        if action == "reset":
            state = self.reset()
            return StepResult(
                state=state, reward=0.0, done=False, info={"action": "reset"}
            )

        if action not in self.ACTION_MAP:
            done = self.is_done()
            return StepResult(
                state=self._build_game_state(done=done),
                reward=0.0,
                done=done,
                info={"action": action, "error": "invalid_action"},
            )

        cur_state = getattr(e._state, "name", "")
        if cur_state in ("WIN", "GAME_OVER"):
            done = cur_state == "WIN"
            return StepResult(
                state=self._build_game_state(done=done),
                reward=0.0,
                done=done,
                info={"action": action, "ignored": True},
            )

        self._last_action_was_reset = False
        self._total_turns += 1

        game_action_id = self.ACTION_MAP[action]
        action_map = {
            1: GameAction.ACTION1,
            2: GameAction.ACTION2,
            3: GameAction.ACTION3,
            4: GameAction.ACTION4,
            5: GameAction.ACTION5,
            7: GameAction.ACTION7,
        }
        game_action = action_map[game_action_id]
        action_input = ActionInput(id=game_action)

        prev_score = self._prev_score
        frame = e.perform_action(action_input, raw=True)
        levels_advanced = frame.levels_completed - prev_score
        self._prev_score = frame.levels_completed

        reward = levels_advanced * (1.0 / len(e._levels))
        done = frame.state.name == "WIN"

        info: Dict = {"action": action}
        if done:
            info["reason"] = "game_complete"
        elif frame.state.name == "GAME_OVER":
            info["reason"] = "game_over"

        return StepResult(
            state=self._build_game_state(done=done, info=info),
            reward=reward,
            done=done,
            info=info,
        )

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError(f"Unsupported render mode: {mode}")
        e = self._engine
        index_grid = e.camera.render(e.current_level.get_sprites())
        h, w = index_grid.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for idx, color in enumerate(self.ARC_PALETTE):
            mask = index_grid == idx
            rgb[mask] = color
        if h != 64 or w != 64:
            row_idx = (np.arange(64) * h // 64).astype(int)
            col_idx = (np.arange(64) * w // 64).astype(int)
            rgb = rgb[np.ix_(row_idx, col_idx)]
        return rgb

    def close(self) -> None:
        self._engine = None


class ArcGameEnv(gym.Env):
    metadata: Dict[str, Any] = {
        "render_modes": ["rgb_array"],
        "render_fps": 10,
    }

    ACTION_LIST: List[str] = ["reset", "up", "down", "left", "right", "select", "undo"]

    OBS_HEIGHT: int = 64
    OBS_WIDTH: int = 64

    def __init__(self, seed: int = 0, render_mode: Optional[str] = None) -> None:
        super().__init__()

        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode '{render_mode}'.")
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
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
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
        terminated: bool = result.done
        truncated: bool = False
        return (
            obs,
            result.reward,
            terminated,
            truncated,
            self._build_info(result.state, result.info),
        )

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
