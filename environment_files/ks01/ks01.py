from __future__ import annotations

import binascii
from dataclasses import dataclass, field
import random
import struct
from typing import Any, Dict, List, Optional, Tuple
import zlib

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from arcengine import (
    ARCBaseGame,
    ActionInput,
    Camera,
    GameAction,
    GameState as ArcPlayState,
    Level,
    RenderableUserDisplay,
    Sprite,
)

ARC_PALETTE: List[Tuple[int, int, int]] = [
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

C_EMPTY = 3
C_CAPACITY = 12
C_USED = 12
C_TARGET = 14
C_VALUE = 14
C_CURSOR = 11
C_SELECTED = 0
C_UNSELECTED = 4
C_LIFE = 8
C_MOVE = 9
C_MOVE_EMPTY = 3

BACKGROUND_COLOR = 5
PADDING_COLOR = 3
CAM_SIZE = 16
RENDER_OUT = 64

_LEVELS = [
    {
        "items": [(1, 1), (2, 3), (3, 4), (2, 2)],
        "capacity": 5,
        "target": 7,
        "max_moves": 32,
    },
    {
        "items": [(1, 1), (2, 2), (3, 5), (4, 5), (2, 4)],
        "capacity": 6,
        "target": 10,
        "max_moves": 48,
    },
    {
        "items": [(2, 1), (3, 4), (4, 5), (5, 6), (1, 1)],
        "capacity": 7,
        "target": 9,
        "max_moves": 32,
    },
    {
        "items": [(1, 1), (2, 2), (3, 5), (4, 7), (5, 6), (2, 3)],
        "capacity": 8,
        "target": 13,
        "max_moves": 48,
    },
    {
        "items": [(2, 2), (3, 4), (4, 5), (5, 8), (6, 9), (2, 3)],
        "capacity": 10,
        "target": 15,
        "max_moves": 48,
    },
]

NUM_LEVELS = len(_LEVELS)

ACTION_MAP = {
    "reset": GameAction.RESET,
    "up": GameAction.ACTION1,
    "down": GameAction.ACTION2,
    "select": GameAction.ACTION5,
    "undo": GameAction.ACTION7,
}

_PLAY_ACTIONS = frozenset(
    {
        GameAction.ACTION1,
        GameAction.ACTION2,
        GameAction.ACTION5,
    }
)


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


def _px(color: int, layer: int = 0, name: str = "px") -> Sprite:
    return Sprite(
        pixels=np.array([[color]], dtype=np.int32),
        name=name,
        visible=True,
        collidable=False,
        tags=[],
        layer=layer,
    )


def _grayscale_frame_to_png(frame: np.ndarray) -> bytes:
    arr = np.asarray(frame, dtype=np.uint8)
    h, w = int(arr.shape[0]), int(arr.shape[1])
    raw = b"".join(b"\0" + arr[y, :].tobytes() for y in range(h))
    compressed = zlib.compress(raw, 9)

    def _chunk(tag: bytes, data: bytes) -> bytes:
        piece = tag + data
        return (
            struct.pack(">I", len(data))
            + piece
            + struct.pack(">I", binascii.crc32(piece) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", compressed)
        + _chunk(b"IEND", b"")
    )


def _camera_index_grid(game: "Ks01") -> np.ndarray:
    lvl = game.current_level
    sprites = lvl.get_sprites() if hasattr(lvl, "get_sprites") else []
    return np.asarray(game.camera.render(sprites), dtype=np.int32)


def _upscale_index_to_render_out(idx: np.ndarray) -> np.ndarray:
    arr = np.asarray(idx, dtype=np.intp)
    h, w = int(arr.shape[0]), int(arr.shape[1])
    if h == RENDER_OUT and w == RENDER_OUT:
        return arr
    if RENDER_OUT % h != 0 or RENDER_OUT % w != 0:
        raise ValueError("camera frame size must divide RENDER_OUT")
    sy, sx = RENDER_OUT // h, RENDER_OUT // w
    return np.repeat(np.repeat(arr, sy, axis=0), sx, axis=1)


def _render_game_png(game: "Ks01") -> bytes:
    try:
        idx = _camera_index_grid(game)
        idx = _upscale_index_to_render_out(idx)
        return _grayscale_frame_to_png(np.clip(idx, 0, 255).astype(np.uint8))
    except (AttributeError, TypeError, ValueError, RuntimeError, MemoryError):
        return _grayscale_frame_to_png(np.zeros((1, 1), dtype=np.uint8))


class KnapsackHUD(RenderableUserDisplay):
    MOVE_W = 12

    def __init__(self, game: Ks01) -> None:
        self._g = game

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape
        cap = getattr(self._g, "_capacity", 0)
        target = getattr(self._g, "_target", 0)
        cur_w = self._g._current_weight() if hasattr(self._g, "_selected") else 0
        cur_v = self._g._current_value() if hasattr(self._g, "_selected") else 0
        moves = getattr(self._g, "_move_count", 0)
        max_moves = getattr(self._g, "_max_moves", 0)
        lives = getattr(self._g, "_lives", 0)

        for x in range(min(cap, w)):
            frame[0, x] = C_CAPACITY

        for i in range(3):
            x = w - 3 + i
            if 0 <= x < w:
                frame[0, x] = C_LIFE if i < lives else C_EMPTY

        if max_moves > 0:
            remain = max(0, max_moves - moves)
            filled = int(round(self.MOVE_W * (remain / max_moves)))
            filled = max(0, min(self.MOVE_W, filled))
            move_x = max(0, w - self.MOVE_W - 6)
            for i in range(self.MOVE_W):
                x = move_x + i
                if 0 <= x < w:
                    frame[0, x] = C_MOVE if i < filled else C_MOVE_EMPTY

        for x in range(min(cap, w)):
            frame[1, x] = C_EMPTY
        for x in range(min(cur_w, cap, w)):
            frame[1, x] = C_USED

        for x in range(min(target, w)):
            frame[2, x] = C_TARGET

        for x in range(min(target, w)):
            frame[3, x] = C_EMPTY
        for x in range(min(cur_v, target, w)):
            frame[3, x] = C_VALUE

        return frame


class Ks01(ARCBaseGame):
    MAX_LIVES = 3

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self._rng = random.Random(seed)
        self._undo_stack: List[Tuple[int, List[bool]]] = []
        levels = [
            Level(sprites=[], grid_size=(CAM_SIZE, CAM_SIZE), data=d) for d in _LEVELS
        ]
        self._hud = KnapsackHUD(self)
        self._lives = self.MAX_LIVES
        camera = Camera(
            x=0,
            y=0,
            width=CAM_SIZE,
            height=CAM_SIZE,
            background=BACKGROUND_COLOR,
            letter_box=PADDING_COLOR,
            interfaces=[self._hud],
        )
        super().__init__(
            "ks01",
            levels,
            camera,
            available_actions=[0, 1, 2, 5, 7],
        )

    def reseed(self) -> None:
        self._rng = random.Random(self._seed)

    def full_reset(self) -> None:
        self._undo_stack.clear()
        super().full_reset()

    def _push_undo_snapshot(self) -> None:
        self._undo_stack.append((self._cursor_idx, list(self._selected)))

    def _apply_undo_snapshot(self, snap: Tuple[int, List[bool]]) -> None:
        cur, sel = snap
        self._cursor_idx = cur
        self._selected = list(sel)
        self._refresh_selection()
        self._update_cursor()

    def _random_cursor_on_item_bar(self) -> None:
        n = len(self._items)
        if n <= 0:
            self._cursor_idx = 0
        else:
            self._cursor_idx = self._rng.randrange(0, n)

    def level_reset(self) -> None:
        self._undo_stack.clear()
        self._selected = [False] * len(self._items)
        self._move_count = 0
        self._action_count = 0
        self._random_cursor_on_item_bar()
        self._refresh_selection()
        self._update_cursor()

    def next_level(self) -> None:
        self._lives = self.MAX_LIVES
        super().next_level()

    def on_set_level(self, level: Level) -> None:
        self._undo_stack.clear()
        idx = min(self._current_level_index, len(_LEVELS) - 1)
        data = _LEVELS[idx]
        self._items: List[Tuple[int, int]] = list(data["items"])
        self._capacity = data["capacity"]
        self._target = data["target"]
        self._max_moves = data["max_moves"]
        self._move_count = 0
        self._lives = self.MAX_LIVES

        self._selected = [False] * len(self._items)
        self._random_cursor_on_item_bar()

        self._create_item_sprites()
        self._create_cursor()

    def _create_item_sprites(self) -> None:
        self._item_markers = []
        self._item_weight_sprites = []
        self._item_value_sprites = []

        for idx, (weight, value) in enumerate(self._items):
            y0 = 4 + idx * 2

            left_top = _px(C_UNSELECTED, layer=4, name="mark")
            left_bot = _px(C_UNSELECTED, layer=4, name="mark")
            left_top.set_position(1, y0)
            left_bot.set_position(1, y0 + 1)
            self.current_level.add_sprite(left_top)
            self.current_level.add_sprite(left_bot)
            self._item_markers.append((left_top, left_bot))

            weight_row = []
            for x in range(4, 4 + weight):
                sp = _px(C_CAPACITY, layer=2, name="weight")
                sp.set_position(x, y0)
                self.current_level.add_sprite(sp)
                weight_row.append(sp)
            self._item_weight_sprites.append(weight_row)

            value_row = []
            for x in range(4, 4 + value):
                sp = _px(C_VALUE, layer=2, name="value")
                sp.set_position(x, y0 + 1)
                self.current_level.add_sprite(sp)
                value_row.append(sp)
            self._item_value_sprites.append(value_row)

        self._refresh_selection()

    def _create_cursor(self) -> None:
        self._cursor_top = _px(C_CURSOR, layer=10, name="cursor")
        self._cursor_bot = _px(C_CURSOR, layer=10, name="cursor")
        self.current_level.add_sprite(self._cursor_top)
        self.current_level.add_sprite(self._cursor_bot)
        self._update_cursor()

    def _update_cursor(self) -> None:
        y0 = 4 + self._cursor_idx * 2
        self._cursor_top.set_position(1, y0)
        self._cursor_bot.set_position(1, y0 + 1)

    def _current_weight(self) -> int:
        return sum(self._items[i][0] for i, sel in enumerate(self._selected) if sel)

    def _current_value(self) -> int:
        return sum(self._items[i][1] for i, sel in enumerate(self._selected) if sel)

    def _refresh_selection(self) -> None:
        for idx, selected in enumerate(self._selected):
            color = C_SELECTED if selected else C_UNSELECTED
            left_top, left_bot = self._item_markers[idx]
            left_top.pixels = np.array([[color]], dtype=np.int32)
            left_bot.pixels = np.array([[color]], dtype=np.int32)

    def _goal_reached(self) -> bool:
        return (
            self._current_weight() <= self._capacity
            and self._current_value() == self._target
        )

    def _trigger_life_loss(self) -> bool:
        self._lives -= 1
        if self._lives <= 0:
            self.lose()
            self.complete_action()
            return True
        self.level_reset()
        return False

    def _enforce_move_limit(self) -> bool:
        if self._move_count < self._max_moves:
            return False
        if self._trigger_life_loss():
            return True
        self.complete_action()
        return True

    def _toggle_current_item(self, count_move: bool = True) -> bool:
        idx = self._cursor_idx

        if self._selected[idx]:
            self._selected[idx] = False
            if count_move:
                self._move_count += 1
            self._refresh_selection()
            return True

        self._selected[idx] = True
        if count_move:
            self._move_count += 1
        self._refresh_selection()
        return True

    def _handle_toggle_result(self) -> bool:
        if (
            self._current_weight() > self._capacity
            or self._current_value() > self._target
        ):
            if self._trigger_life_loss():
                return True
            self.complete_action()
            return True

        if self._goal_reached():
            self.next_level()
            self.complete_action()
            return True

        return self._enforce_move_limit()

    def _text_observation(self) -> str:
        moves_left = max(0, self._max_moves - self._move_count)
        n = len(self._items)
        lines = [
            f"Level {self._current_level_index + 1}/{NUM_LEVELS}.",
            f"Capacity {self._capacity}. Target total value {self._target}.",
            f"Gameplay actions used (turn) {self._move_count}. Moves budget remaining {moves_left}. Lives {self._lives}.",
            f"Cursor on item index {self._cursor_idx} (0..{max(0, n - 1)}).",
            f"Current selection weight {self._current_weight()} value {self._current_value()}.",
            "Items (index, weight, value, included y/n):",
        ]
        for i in range(n):
            w, v = self._items[i]
            on = "y" if self._selected[i] else "n"
            mark = " *" if i == self._cursor_idx else ""
            lines.append(f"  {i}: w={w} v={v} included={on}{mark}")
        lines.append(
            "Actions: up/down move cursor; toggle/action toggles inclusion; "
            "undo reverts last move and consumes one move; reset: first press stays on this level, "
            "clears selection, and moves the cursor to a random item row (seeded RNG); losing a life "
            "does the same; second consecutive reset (R twice with no other actions) restarts from "
            "level 1."
        )
        return "\n".join(lines)

    def step(self) -> None:
        act = self.action.id

        if act == GameAction.RESET:
            self.level_reset()
            self.complete_action()
            return

        if act == GameAction.ACTION7:
            if self._undo_stack:
                snap = self._undo_stack.pop()
                self._apply_undo_snapshot(snap)
            self._move_count += 1
            if self._enforce_move_limit():
                return
            self.complete_action()
            return

        if act not in _PLAY_ACTIONS:
            self.complete_action()
            return

        if act == GameAction.ACTION1:
            self._push_undo_snapshot()
            if self._cursor_idx > 0:
                self._cursor_idx -= 1
                self._update_cursor()
            self._move_count += 1
            if self._enforce_move_limit():
                return
        elif act == GameAction.ACTION2:
            self._push_undo_snapshot()
            if self._cursor_idx < len(self._items) - 1:
                self._cursor_idx += 1
                self._update_cursor()
            self._move_count += 1
            if self._enforce_move_limit():
                return
        elif act == GameAction.ACTION5:
            self._push_undo_snapshot()
            changed = self._toggle_current_item()
            if changed and self._handle_toggle_result():
                return
        else:
            pass

        self.complete_action()


class PuzzleEnvironment:
    def __init__(self, seed: int = 0) -> None:
        self._engine = Ks01(seed)
        self.seed = seed
        self._total_turns = 0
        self._done = False
        self._last_action_was_reset = False

    @property
    def game(self) -> Ks01 | None:
        return self._engine

    def _episode_terminal(self) -> bool:
        g = self._engine
        return g._current_level_index >= NUM_LEVELS or g._lives <= 0

    def reset(self) -> GameState:
        e = self._engine
        self._total_turns = 0
        self._done = False
        game_won = e._current_level_index >= NUM_LEVELS
        if game_won or e._move_count == 0 or self._last_action_was_reset:
            self._engine = Ks01(self.seed)
            e = self._engine
        else:
            e.perform_action(ActionInput(id=GameAction.RESET))
        self._last_action_was_reset = True
        return self._create_game_state()

    def get_actions(self) -> list[str]:
        if self._episode_terminal():
            return ["reset"]
        return ["reset", "up", "down", "select", "undo"]

    def _outcome_after_step(
        self, lives_before: int, level_before: int
    ) -> tuple[float, bool, dict]:
        g = self._engine
        info: dict = {
            "lives": g._lives,
            "level": g._current_level_index + 1,
            "moves_used": g._move_count,
            "move_limit": g._max_moves,
        }
        reward = 0.0
        done = False
        if g._lives < lives_before:
            info["event"] = "life_lost"
            if g._lives <= 0:
                info["event"] = "game_over"
                done = True
        elif g._current_level_index != level_before:
            reward = 1.0 / NUM_LEVELS
            last_level_idx = NUM_LEVELS - 1
            if level_before == last_level_idx:
                info["event"] = "game_complete"
                done = True
            else:
                info["event"] = "level_complete"
        return reward, done, info

    @staticmethod
    def _resolve_action_key(action: str) -> str | None:
        key = action.strip().lower()
        return key if key in ACTION_MAP else None

    def step(self, action: str) -> StepResult:
        ak = self._resolve_action_key(action)
        if ak is None:
            return StepResult(
                state=self._create_game_state(),
                reward=0.0,
                done=self.is_done(),
                info={"error": f"Invalid action: {action}"},
            )

        if ak == "reset":
            e = self._engine
            game_won = e._current_level_index >= NUM_LEVELS
            full_restart = game_won or e._move_count == 0 or self._last_action_was_reset
            state = self.reset()
            return StepResult(
                state=state,
                reward=0.0,
                done=False,
                info={"action": "reset", "full_restart": full_restart},
            )

        self._last_action_was_reset = False
        self._total_turns += 1

        lives_before = self._engine._lives
        level_before = self._engine._current_level_index

        ga = ACTION_MAP[ak]
        self._engine.perform_action(ActionInput(id=ga))

        reward, done, info = self._outcome_after_step(lives_before, level_before)
        self._done = done or self._episode_terminal()

        return StepResult(
            state=self._create_game_state(),
            reward=reward,
            done=self._done,
            info=info,
        )

    def is_done(self) -> bool:
        return self._done or self._episode_terminal()

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if self._engine is None:
            raise RuntimeError("Environment closed")
        if mode != "rgb_array":
            raise ValueError(f"Unsupported render mode: {mode!r}; use 'rgb_array'")
        idx = _upscale_index_to_render_out(_camera_index_grid(self._engine))
        pal = np.array(ARC_PALETTE, dtype=np.uint8)
        clipped = np.clip(idx.astype(np.intp, copy=False), 0, len(pal) - 1)
        rgb = pal[clipped]
        return np.ascontiguousarray(rgb, dtype=np.uint8)

    def close(self) -> None:
        self._engine = None

    def _create_game_state(self) -> GameState:
        g = self._engine
        if g is None:
            return GameState(
                text_observation=f"Agent turn: {self._total_turns}\n\n(closed)",
                image_observation=None,
                valid_actions=None,
                turn=self._total_turns,
                metadata={
                    "total_levels": len(self._engine._levels),"terminal": True},
            )
        if hasattr(g, "_items") and g._items:
            body = g._text_observation()
            image_observation = _render_game_png(g)
        else:
            body = "(no level loaded)"
            image_observation = None
        text_observation = f"Agent turn: {self._total_turns}\n\n{body}"
        terminal = self._episode_terminal()
        va = ["reset"] if terminal else self.get_actions()
        return GameState(
            text_observation=text_observation,
            image_observation=image_observation,
            valid_actions=va,
            turn=self._total_turns,
            metadata={
                "total_levels": len(self._engine._levels),
                "level": g._current_level_index + 1,
                "lives": g._lives,
                "moves_used": g._move_count,
                "max_moves": g._max_moves,
                "goal_reached": g._goal_reached()
                if hasattr(g, "_goal_reached")
                else False,
                "over_capacity": g._current_weight() > g._capacity
                if hasattr(g, "_capacity")
                else False,
                "over_target_value": g._current_value() > g._target
                if hasattr(g, "_target")
                else False,
                "terminal": terminal,
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

    env = ArcGameEnv(seed=0, render_mode="rgb_array")

    check_env(env, skip_render_check=False)

    obs, info = env.reset()
    obs, reward, term, trunc, info = env.step(0)
    frame = env.render()
    env.close()
