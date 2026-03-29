from __future__ import annotations

import random
import struct
import zlib
from dataclasses import dataclass, field
from typing import Any, List, Optional, Dict, Tuple

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
    Sprite,
)


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


BAR_COLORS = [9, 12, 11, 14, 8, 15, 6, 10, 7, 13, 9, 12, 11, 14, 8]

BACKGROUND = 5
SLIDER_COLOR = 0

GRID_SIZE = 32
INNER_ORIGIN = 1
INNER_SIZE = GRID_SIZE - 2

PERIM_LENGTH = 4 * (GRID_SIZE - 1)
PERIM_OFF_COLOR = BACKGROUND

LIFE_BORDER_COLORS = {
    3: 14,
    2: 11,
    1: 8,
}

MOVE_LIMITS = [60, 80, 100, 220, 300]

MAX_LIVES = 3
LIFE_COLOR = 8
LIFE_EMPTY_COLOR = 3
LIFE_BAR_HEIGHT = 5

BAR_WIDTH = 1

LEVEL_ARRAYS = [
    [3, 1, 2, 4],
    [3, 6, 1, 5, 4, 2],
    [5, 4, 3, 7, 8, 6, 2, 1],
    [12, 7, 9, 2, 4, 5, 3, 8, 1, 11, 6, 10],
    [1, 6, 12, 3, 13, 4, 14, 7, 11, 8, 10, 2, 15, 9, 5],
]

SLIDER_START_POSITIONS = [
    [0, 1, 2, 0],
    [0, 2, 4, 1],
    [0, 3, 6, 2],
    [0, 4, 8, 6],
    [0, 5, 10, 8],
]


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


def make_bar_sprite(height: int, width: int, color: int, name: str) -> Sprite:
    pixels = [[color] * width for _ in range(height)]
    return Sprite(
        pixels=pixels,
        name=name,
        visible=True,
        collidable=False,
        layer=2,
        tags=["bar"],
    )


def make_slider_sprite(width: int, height: int, color: int, name: str) -> Sprite:
    pixels = [[color] * width for _ in range(height)]
    return Sprite(
        pixels=pixels,
        name=name,
        visible=True,
        collidable=False,
        layer=3,
        tags=["slider"],
    )


def make_border_dot(color: int, name: str) -> Sprite:
    return Sprite(
        pixels=[[color]],
        name=name,
        visible=True,
        collidable=False,
        layer=5,
        tags=["border"],
    )


def make_life_bar(height: int, color: int, name: str) -> Sprite:
    pixels = [[color] for _ in range(height)]
    return Sprite(
        pixels=pixels,
        name=name,
        visible=True,
        collidable=False,
        layer=6,
        tags=["life"],
    )


def _perimeter_coords(size: int) -> list[tuple[int, int]]:
    coords = []
    last = size - 1
    for c in range(last):
        coords.append((c, 0))
    for r in range(last):
        coords.append((last, r))
    for c in range(last, 0, -1):
        coords.append((c, last))
    for r in range(last, 0, -1):
        coords.append((0, r))
    return coords


_placeholder = Sprite(
    pixels=[[5]],
    name="ph",
    visible=False,
    collidable=False,
    layer=-10,
)


class Sq04(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)
        self.bar_sprites: list[Sprite] = []
        self.slider_sprite: Sprite | None = None
        self.border_sprites: list[Sprite] = []
        self.life_sprites: list[Sprite] = []
        self.lives = MAX_LIVES
        self._undo_stack: list[dict] = []
        self._game_over = False

        levels = []
        for i, arr in enumerate(LEVEL_ARRAYS):
            level_sprites = [
                _placeholder.clone().set_position(0, 0),
            ]
            levels.append(
                Level(
                    sprites=level_sprites,
                    grid_size=(GRID_SIZE, GRID_SIZE),
                    data={
                        "array": arr,
                        "level_idx": i,
                        "move_limit": MOVE_LIMITS[i] if i < len(MOVE_LIMITS) else 999,
                    },
                    name=f"Level {i + 1}",
                )
            )

        camera = Camera(
            x=0,
            y=0,
            width=32,
            height=32,
            background=BACKGROUND,
            letter_box=BACKGROUND,
        )

        super().__init__(
            "sq04", levels, camera, available_actions=[0, 1, 2, 3, 4, 5, 7]
        )

    def _save_state(self) -> dict:
        return {
            "arr": list(self.arr),
            "slider_pos": self.slider_pos,
        }

    def _restore_state(self, state: dict) -> None:
        self.arr = list(state["arr"])
        self.sorted_arr = sorted(self.arr)
        self.slider_pos = state["slider_pos"]
        self._rebuild_bars()
        self._create_slider()
        self._update_slider()

    def on_set_level(self, level: Level) -> None:
        self.arr = list(self.current_level.get_data("array"))
        self.n = len(self.arr)
        self.sorted_arr = sorted(self.arr)
        self.bar_w = BAR_WIDTH

        level_idx = self.current_level.get_data("level_idx")
        self.slider_pos = self._rng.choice(SLIDER_START_POSITIONS[level_idx])

        self.move_limit = self.current_level.get_data("move_limit")
        self.moves_used = 0

        self.lives = MAX_LIVES
        self._undo_stack = []
        self._game_over = False

        self.bar_sprites = []
        self.slider_sprite = None
        self.border_sprites = []
        self.life_sprites = []

        self._rebuild_bars()
        self._create_slider()
        self._create_border()
        self._create_life_display()
        self._update_slider()

    def _bar_x(self, idx: int) -> int:
        total_width = self.n * self.bar_w
        start_x = INNER_ORIGIN + max(0, (INNER_SIZE - total_width) // 2)
        return start_x + idx * self.bar_w

    def _bar_baseline(self) -> int:
        if self.n <= 12:
            return INNER_ORIGIN + INNER_SIZE - 3
        return INNER_ORIGIN + INNER_SIZE - 2

    def _bar_h(self, value: int) -> int:
        if self.n <= 8:
            return (value + 1) * 2
        elif self.n <= 12:
            return value * 2
        else:
            max_val = max(self.arr)
            avail = self._bar_baseline() - INNER_ORIGIN
            return max(1, value * avail // max_val)

    def _bar_y(self, value: int) -> int:
        return self._bar_baseline() - self._bar_h(value)

    def _slider_y(self) -> int:
        return self._bar_baseline()

    def _slider_x(self) -> int:
        return self._bar_x(self.slider_pos)

    def _slider_w(self) -> int:
        return self.bar_w * 2

    def _slider_h(self) -> int:
        return 2 if self.n <= 12 else 1

    def _rebuild_bars(self) -> None:
        for s in self.bar_sprites:
            try:
                self.current_level.remove_sprite(s)
            except Exception:
                pass
        self.bar_sprites = []

        for i, val in enumerate(self.arr):
            h = self._bar_h(val)
            color = BAR_COLORS[(val - 1) % len(BAR_COLORS)]
            bar = make_bar_sprite(h, self.bar_w, color, f"bar_{i}")
            bar.set_position(self._bar_x(i), self._bar_y(val))
            self.current_level.add_sprite(bar)
            self.bar_sprites.append(bar)

    def _create_slider(self) -> None:
        if self.slider_sprite:
            try:
                self.current_level.remove_sprite(self.slider_sprite)
            except Exception:
                pass
        self.slider_sprite = make_slider_sprite(
            self._slider_w(), self._slider_h(), SLIDER_COLOR, "slider"
        )
        self.current_level.add_sprite(self.slider_sprite)

    def _update_slider(self) -> None:
        if self.slider_sprite:
            self.slider_sprite.set_position(self._slider_x(), self._slider_y())
            self.slider_sprite.color_remap(None, SLIDER_COLOR)

    def _border_color(self) -> int:
        return LIFE_BORDER_COLORS.get(self.lives, 8)

    def _create_border(self) -> None:
        self._remove_border()
        self.border_sprites = []

        color = self._border_color()
        coords = _perimeter_coords(GRID_SIZE)

        for i, (cx, cy) in enumerate(coords):
            dot = make_border_dot(color, f"bdr_{i}")
            dot.set_position(cx, cy)
            self.current_level.add_sprite(dot)
            self.border_sprites.append(dot)

        total_border_pixels = PERIM_LENGTH
        off_color = PERIM_OFF_COLOR
        events = []
        for pix_idx in range(total_border_pixels - 1, -1, -1):
            events.append((pix_idx, off_color))
        total_events = len(events)

        move_budget = self.move_limit
        self._border_schedule = [[] for _ in range(move_budget + 1)]
        if move_budget > 0:
            for e_idx, (pix_idx, clr) in enumerate(events):
                trigger_move = -(-((e_idx + 1) * move_budget) // total_events)
                trigger_move = min(trigger_move, move_budget)
                self._border_schedule[trigger_move].append((pix_idx, clr))

    def _remove_border(self) -> None:
        for s in self.border_sprites:
            try:
                self.current_level.remove_sprite(s)
            except Exception:
                pass
        self.border_sprites = []

    def _update_border(self) -> None:
        if not self.border_sprites or not hasattr(self, "_border_schedule"):
            return
        current_move = self.moves_used
        if current_move < len(self._border_schedule):
            for pix_idx, clr in self._border_schedule[current_move]:
                self.border_sprites[pix_idx].color_remap(None, clr)

    def _create_life_display(self) -> None:
        for s in self.life_sprites:
            try:
                self.current_level.remove_sprite(s)
            except Exception:
                pass
        self.life_sprites = []

        for i in range(MAX_LIVES):
            color = LIFE_COLOR if i < self.lives else LIFE_EMPTY_COLOR
            bar = make_life_bar(LIFE_BAR_HEIGHT, color, f"life_{i}")
            bar.set_position(INNER_ORIGIN + i * 2, INNER_ORIGIN)
            self.current_level.add_sprite(bar)
            self.life_sprites.append(bar)

    def _update_life_display(self) -> None:
        for i, dot in enumerate(self.life_sprites):
            if i < self.lives:
                dot.color_remap(None, LIFE_COLOR)
            else:
                dot.color_remap(None, LIFE_EMPTY_COLOR)

    def _use_move(self) -> bool:
        self.moves_used += 1
        self._update_border()
        if self.moves_used >= self.move_limit:
            return True
        return False

    def _lose_life(self) -> None:
        self.lives -= 1
        self._update_life_display()

        if self.lives <= 0:
            self._remove_border()
            self._game_over = True
            self.lose()
            return

        self.arr = list(self.current_level.get_data("array"))
        self.sorted_arr = sorted(self.arr)
        level_idx = self.current_level.get_data("level_idx")
        self.slider_pos = self._rng.choice(SLIDER_START_POSITIONS[level_idx])
        self.moves_used = 0
        self._undo_stack = []

        self._rebuild_bars()
        self._create_slider()
        self._update_slider()
        self._create_border()

    def _swap_bar_sprites(self, pos: int) -> None:
        self.arr[pos], self.arr[pos + 1] = self.arr[pos + 1], self.arr[pos]

        self.bar_sprites[pos], self.bar_sprites[pos + 1] = (
            self.bar_sprites[pos + 1],
            self.bar_sprites[pos],
        )

        for idx in (pos, pos + 1):
            val = self.arr[idx]
            self.bar_sprites[idx].set_position(self._bar_x(idx), self._bar_y(val))

    def _is_complete(self) -> bool:
        return self.arr == self.sorted_arr

    def step(self) -> None:
        action = self.action.id

        if action == GameAction.ACTION1 or action == GameAction.ACTION2:
            self._undo_stack.append(self._save_state())
            if self._use_move():
                self._lose_life()
            self.complete_action()
            return

        if action == GameAction.ACTION3:
            self._undo_stack.append(self._save_state())
            if self.slider_pos > 0:
                self.slider_pos -= 1
                self._update_slider()
            if self._use_move():
                self._lose_life()
            self.complete_action()
            return

        if action == GameAction.ACTION4:
            self._undo_stack.append(self._save_state())
            if self.slider_pos < self.n - 2:
                self.slider_pos += 1
                self._update_slider()
            if self._use_move():
                self._lose_life()
            self.complete_action()
            return

        if action == GameAction.ACTION5:
            self._undo_stack.append(self._save_state())
            self._swap_bar_sprites(self.slider_pos)
            out_of_moves = self._use_move()

            if self._is_complete():
                self.next_level()
            elif out_of_moves:
                self._lose_life()

            self.complete_action()
            return

        if action == GameAction.ACTION7:
            if self._undo_stack:
                prev = self._undo_stack.pop()
                self._restore_state(prev)
            if self._use_move():
                self._lose_life()
            self.complete_action()
            return

        self.complete_action()


class PuzzleEnvironment:
    def __init__(self, seed: int = 0) -> None:
        self._engine = Sq04(seed=seed)
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
            "moves_used": self._engine.moves_used,
            "move_limit": self._engine.move_limit,
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
        lines = []
        lines.append(
            f"Level {self._engine.level_index + 1}/{len(self._engine._levels)}"
            f" | Lives: {self._engine.lives}"
            f" | Moves: {self._engine.moves_used}/{self._engine.move_limit}"
            f" | Turn: {self._total_turns}"
        )
        lines.append("")

        if hasattr(self._engine, "arr") and hasattr(self._engine, "n"):
            max_val = max(self._engine.arr)
            max_h = max_val
            if self._engine.n <= 8:
                max_h = (max_val + 1) * 2
            elif self._engine.n <= 12:
                max_h = max_val * 2

            for row in range(max_h, 0, -1):
                row_str = ""
                for i, val in enumerate(self._engine.arr):
                    h = self._engine._bar_h(val)
                    if row <= h:
                        row_str += str(val).rjust(2) + " "
                    else:
                        row_str += "   "
                lines.append(row_str.rstrip())

            lines.append("-" * (self._engine.n * 3))

            slider_line = ""
            for i in range(self._engine.n):
                if i == self._engine.slider_pos or i == self._engine.slider_pos + 1:
                    slider_line += "^^ "
                else:
                    slider_line += "   "
            lines.append(slider_line.rstrip())

            lines.append("")
            lines.append(
                f"Cursor: columns"
                f" {self._engine.slider_pos + 1}"
                f" and {self._engine.slider_pos + 2}"
            )
        else:
            lines.append("(no level loaded)")

        lines.append("")
        lines.append(f"Actions: {', '.join(_VALID_ACTIONS)}")

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
                "lives": self._engine.lives,
                "moves_used": self._engine.moves_used
                if hasattr(self._engine, "moves_used")
                else 0,
                "move_limit": self._engine.move_limit
                if hasattr(self._engine, "move_limit")
                else 0,
                "array": list(self._engine.arr) if hasattr(self._engine, "arr") else [],
                "slider_pos": self._engine.slider_pos
                if hasattr(self._engine, "slider_pos")
                else 0,
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
