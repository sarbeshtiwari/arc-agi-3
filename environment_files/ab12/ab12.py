from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import struct
import zlib

import numpy as np
import gymnasium as gym
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
from arcengine.enums import GameState as EngineGameState


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


FRAME_SIZE = 64
UI_ROWS = 2

BACKGROUND_COLOR = 0
PADDING_COLOR = 0
CURSOR_COLOR = 15
SELECTED_COLOR = 14

COLOR_MIX_RULES = {
    (8, 8): 8,
    (8, 7): 8,
    (6, 9): 8,
    (11, 8): 8,
    (10, 10): 10,
    (10, 11): 10,
    (7, 9): 10,
    (8, 6): 10,
    (11, 11): 11,
    (8, 9): 11,
    (6, 7): 11,
    (10, 9): 11,
}

COLOR_BAR_FILL = 10
COLOR_BAR_EMPTY = 1
COLOR_LIFE = 8
COLOR_LIFE_EMPTY = 3

LEVEL_DATA = [
    {
        "mode": "COLLECT",
        "source_colors": [11, 8, 6, 9],
        "target_pattern": [6, 8, 9, 11],
    },
    {
        "mode": "MIX",
        "source_colors": [6, 9, 7, 9, 6, 7],
        "target_pattern": [8, 10, 11],
    },
    {
        "mode": "MIX",
        "source_colors": [6, 9, 10, 11, 8, 9, 8, 7],
        "target_pattern": [8, 10, 11, 8],
    },
    {
        "mode": "MIX",
        "source_colors": [11, 8, 7, 9, 8, 9, 10, 11, 8, 8],
        "target_pattern": [8, 10, 11, 10, 8],
    },
]

MOVE_LIMITS = [32, 48, 80, 100]

GUIDE_GROUPS = [
    (8, [(8, 8), (8, 7), (6, 9), (11, 8)]),
    (10, [(10, 10), (10, 11), (7, 9), (8, 6)]),
    (11, [(11, 11), (8, 9), (6, 7), (10, 9)]),
]

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


class MoveDisplay(RenderableUserDisplay):
    MAX_LIVES = 3

    def __init__(self, game: "Ab12") -> None:
        self._game = game
        self.max_moves: int = 0
        self.remaining: int = 0

    def set_limit(self, max_moves: int) -> None:
        self.max_moves = max_moves
        self.remaining = max_moves

    def decrement(self) -> None:
        if self.remaining > 0:
            self.remaining -= 1

    def reset(self) -> None:
        self.remaining = self.max_moves

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        if self.max_moves == 0:
            return frame

        fw = frame.shape[1]
        fh = frame.shape[0]
        rs = fh - UI_ROWS
        re = fh

        frame[rs:re, :] = COLOR_BAR_EMPTY
        bw = int(fw * 0.7)
        filled = int(bw * self.remaining / self.max_moves)
        for x in range(filled):
            frame[rs:re, x] = COLOR_BAR_FILL

        lives = getattr(self._game, "_lives", self.MAX_LIVES)
        ls = bw
        lw = fw - bw
        lbw = 2
        lgap = 2
        tlw = (lbw * self.MAX_LIVES) + (lgap * (self.MAX_LIVES - 1))
        off = (lw - tlw) // 2
        for i in range(self.MAX_LIVES):
            xs = ls + off + i * (lbw + lgap)
            xe = xs + lbw
            c = COLOR_LIFE if i < lives else COLOR_LIFE_EMPTY
            if xe <= fw:
                frame[rs:re, xs:xe] = c
        return frame


sprites = {
    "block": Sprite(
        pixels=[[6]],
        name="block",
        visible=True,
        collidable=True,
    ),
}

levels = [
    Level(sprites=[], grid_size=(FRAME_SIZE, FRAME_SIZE)),
    Level(sprites=[], grid_size=(FRAME_SIZE, FRAME_SIZE)),
    Level(sprites=[], grid_size=(FRAME_SIZE, FRAME_SIZE)),
    Level(sprites=[], grid_size=(FRAME_SIZE, FRAME_SIZE)),
]


class Ab12(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._cursor_pos = 0
        self._selected_index = None
        self._inventory: List[int] = []
        self._collect_positions: List[int] = []
        self._mix_history: List[List[int]] = []
        self._cursor_history: List[Tuple[int, Optional[int]]] = []
        self._target_pattern: List[int] = []
        self._source_colors: List[int] = []
        self._sources_remaining: List[int] = []
        self._level_mode = "COLLECT"
        self._max_moves = 0
        self._moves_remaining = 0
        self._lives = 3
        self._game_over = False
        self._seed = seed
        self._rng = np.random.RandomState(seed)

        self._init_state: Dict[str, Any] = {}

        self._move_display = MoveDisplay(self)
        camera = Camera(0, 0, FRAME_SIZE, FRAME_SIZE, BACKGROUND_COLOR, PADDING_COLOR, [self._move_display])
        super().__init__(game_id="ab12", levels=levels, camera=camera, available_actions=[0, 1, 2, 4, 5, 7], seed=seed)

    def on_set_level(self, level: Level) -> None:
        self._game_over = False
        self._lives = 3
        self._load_level()
        self._save_init_state()
        self._render_level(level)
        self._move_display.set_limit(self._max_moves)

    def _load_level(self) -> None:
        idx = self.level_index
        self._rng = np.random.RandomState(self._seed + idx)
        data = LEVEL_DATA[idx]
        self._level_mode = data["mode"]
        self._source_colors = list(data["source_colors"])
        self._target_pattern = list(data["target_pattern"])
        self._sources_remaining = list(data["source_colors"])
        self._inventory = []
        self._collect_positions = []
        self._mix_history = []
        self._cursor_history = []
        self._cursor_pos = self._rng.randint(0, len(self._sources_remaining))
        self._selected_index = None
        self._max_moves = MOVE_LIMITS[idx]
        self._moves_remaining = self._max_moves

    def _save_init_state(self) -> None:
        self._init_state = {
            "source_colors": list(self._source_colors),
            "target_pattern": list(self._target_pattern),
            "sources_remaining": list(self._sources_remaining),
            "level_mode": self._level_mode,
            "max_moves": self._max_moves,
        }

    def _restore_init_state(self) -> None:
        s = self._init_state
        self._source_colors = list(s["source_colors"])
        self._target_pattern = list(s["target_pattern"])
        self._sources_remaining = list(s["sources_remaining"])
        self._level_mode = s["level_mode"]
        self._inventory = []
        self._collect_positions = []
        self._mix_history = []
        self._cursor_history = []
        self._cursor_pos = self._rng.randint(0, len(self._sources_remaining))
        self._selected_index = None
        self._max_moves = s["max_moves"]
        self._moves_remaining = self._max_moves
        self._move_display.set_limit(self._max_moves)
        self._game_over = False

    def _get_layout(self) -> Tuple[int, int]:
        total_sources = len(self._source_colors)
        if total_sources <= 6:
            return 5, 4
        if total_sources <= 9:
            return 4, 3
        return 3, 2

    def _render_level(self, level: Level) -> None:
        for sp in list(level._sprites):
            level.remove_sprite(sp)

        item_spacing, block_scale = self._get_layout()
        self._render_sources(level, item_spacing, block_scale)
        self._render_cursor(level, item_spacing)
        self._render_selection(level, item_spacing)
        self._render_inventory(level, item_spacing, block_scale)
        self._render_target(level, item_spacing, block_scale)
        self._render_mode_indicator(level)
        self._render_guide(level)

    def _render_sources(self, level: Level, item_spacing: int, block_scale: int) -> None:
        for idx, color in enumerate(self._sources_remaining):
            py = 2 + (idx * item_spacing)
            level.add_sprite(
                sprites["block"].clone()
                .color_remap(None, color)
                .set_scale(block_scale)
                .set_position(5, py)
            )

    def _render_cursor(self, level: Level, item_spacing: int) -> None:
        cursor_y = (
            2 + (self._cursor_pos * item_spacing)
            if self._cursor_pos < len(self._sources_remaining)
            else 2
        )
        level.add_sprite(
            sprites["block"].clone()
            .color_remap(None, CURSOR_COLOR)
            .set_scale(2)
            .set_position(2, cursor_y)
        )

    def _render_selection(self, level: Level, item_spacing: int) -> None:
        if self._selected_index is None:
            return
        sel_y = (
            2 + (self._selected_index * item_spacing)
            if self._selected_index < len(self._sources_remaining)
            else 2
        )
        level.add_sprite(
            sprites["block"].clone()
            .color_remap(None, SELECTED_COLOR)
            .set_scale(2)
            .set_position(10, sel_y)
        )

    def _render_inventory(self, level: Level, item_spacing: int, block_scale: int) -> None:
        for idx, color in enumerate(self._inventory):
            py = 2 + (idx * item_spacing)
            level.add_sprite(
                sprites["block"].clone()
                .color_remap(None, color)
                .set_scale(block_scale)
                .set_position(28, py)
            )

    def _render_target(self, level: Level, item_spacing: int, block_scale: int) -> None:
        for idx, color in enumerate(self._target_pattern):
            py = 2 + (idx * item_spacing)
            level.add_sprite(
                sprites["block"].clone()
                .color_remap(None, color)
                .set_scale(block_scale)
                .set_position(50, py)
            )

    def _render_mode_indicator(self, level: Level) -> None:
        mode_color = 2 if self._level_mode == "COLLECT" else 3
        for i in range(5):
            level.add_sprite(
                sprites["block"].clone()
                .color_remap(None, mode_color)
                .set_position(38 + i, 0)
            )

    def _render_guide(self, level: Level) -> None:
        col_x = [1, 22, 43]
        guide_y_top = 44
        row_h = 3

        for col_i, (result_c, combos) in enumerate(GUIDE_GROUPS):
            cx = col_x[col_i]
            level.add_sprite(
                sprites["block"].clone()
                .color_remap(None, result_c)
                .set_scale(2)
                .set_position(cx, guide_y_top)
            )
            for row_i, (c1, c2) in enumerate(combos):
                ry = guide_y_top + 3 + row_i * row_h
                for px_off, color in [(0, c1), (2, c2), (4, 15), (6, result_c)]:
                    level.add_sprite(
                        sprites["block"].clone()
                        .color_remap(None, color)
                        .set_scale(1)
                        .set_position(cx + px_off, ry)
                    )

    def _mix_colors(self, color1: int, color2: int) -> int:
        if (color1, color2) in COLOR_MIX_RULES:
            return COLOR_MIX_RULES[(color1, color2)]
        if (color2, color1) in COLOR_MIX_RULES:
            return COLOR_MIX_RULES[(color2, color1)]
        return color1

    def _restart_level(self) -> None:
        self._lives -= 1
        if self._lives <= 0:
            self.lose()
            return
        self._restore_init_state()
        self._render_level(self.current_level)

    def _handle_cursor_up(self) -> None:
        self._cursor_pos -= 1
        if self._cursor_pos < 0:
            self._cursor_pos = 0

    def _handle_cursor_down(self) -> None:
        self._cursor_pos += 1
        if self._cursor_pos >= len(self._sources_remaining):
            self._cursor_pos = (
                len(self._sources_remaining) - 1
                if len(self._sources_remaining) > 0
                else 0
            )

    def _handle_undo(self) -> None:
        if not self._cursor_history:
            return
        prev_cursor, prev_selected = self._cursor_history.pop()
        if len(self._inventory) > 0:
            removed = self._inventory.pop()
            if self._level_mode == "COLLECT":
                original_pos = self._collect_positions.pop() if self._collect_positions else len(self._sources_remaining)
                self._sources_remaining.insert(original_pos, removed)
            elif self._level_mode == "MIX" and self._mix_history:
                original_colors = self._mix_history.pop()
                for color in original_colors:
                    self._sources_remaining.append(color)
        self._cursor_pos = prev_cursor
        self._selected_index = prev_selected

    def _handle_action(self) -> None:
        if len(self._sources_remaining) == 0:
            return
        if self._cursor_pos >= len(self._sources_remaining):
            return

        if self._level_mode == "COLLECT":
            color = self._sources_remaining.pop(self._cursor_pos)
            self._inventory.append(color)
            self._collect_positions.append(self._cursor_pos)
            if self._cursor_pos >= len(self._sources_remaining):
                self._cursor_pos = (
                    len(self._sources_remaining) - 1
                    if len(self._sources_remaining) > 0
                    else 0
                )

        elif self._level_mode == "MIX":
            if self._selected_index is None:
                self._selected_index = self._cursor_pos
            else:
                if self._selected_index < len(self._sources_remaining) and self._cursor_pos < len(self._sources_remaining):
                    color1 = self._sources_remaining[self._selected_index]
                    color2 = self._sources_remaining[self._cursor_pos]
                    result = self._mix_colors(color1, color2)

                    if self._selected_index == self._cursor_pos:
                        self._sources_remaining.pop(self._cursor_pos)
                        self._mix_history.append([color1])
                    else:
                        indices = sorted([self._selected_index, self._cursor_pos], reverse=True)
                        for idx in indices:
                            if idx < len(self._sources_remaining):
                                self._sources_remaining.pop(idx)
                        self._mix_history.append([color1, color2])

                    self._inventory.append(result)
                    self._selected_index = None
                    if self._cursor_pos >= len(self._sources_remaining):
                        self._cursor_pos = (
                            len(self._sources_remaining) - 1
                            if len(self._sources_remaining) > 0
                            else 0
                        )

    def _handle_deselect(self) -> None:
        self._selected_index = None

    def step(self) -> None:
        if not self.action:
            self.complete_action()
            return

        if self._game_over:
            self._restart_level()
            self.complete_action()
            return

        action_id = self.action.id

        if action_id == GameAction.RESET:
            self.reset()
            self.complete_action()
            return

        self._moves_remaining -= 1
        self._move_display.decrement()

        if action_id != GameAction.ACTION7:
            self._cursor_history.append((self._cursor_pos, self._selected_index))

        if action_id == GameAction.ACTION1:
            self._handle_cursor_up()
        elif action_id == GameAction.ACTION2:
            self._handle_cursor_down()
        elif action_id == GameAction.ACTION3:
            pass
        elif action_id == GameAction.ACTION4:
            self._handle_deselect()
        elif action_id == GameAction.ACTION5:
            self._handle_action()
        elif action_id == GameAction.ACTION7:
            self._handle_undo()

        if self._moves_remaining <= 0 and self._inventory != self._target_pattern:
            self._game_over = True
            self._render_level(self.current_level)
            self.complete_action()
            return

        self._render_level(self.current_level)

        if self._inventory == self._target_pattern:
            self.next_level()

        self.complete_action()

    def reset(self) -> None:
        self._restore_init_state()
        self._render_level(self.current_level)

    def get_actions(self) -> list:
        return [
            GameAction.ACTION1,
            GameAction.ACTION2,
            GameAction.ACTION4,
            GameAction.ACTION5,
            GameAction.ACTION7,
        ]


class PuzzleEnvironment:
    _ACTION_MAP: Dict[str, GameAction] = {
        "reset": GameAction.RESET,
        "up": GameAction.ACTION1,
        "down": GameAction.ACTION2,
        "right": GameAction.ACTION4,
        "select": GameAction.ACTION5,
        "undo": GameAction.ACTION7,
    }

    def __init__(self, seed: int = 0) -> None:
        self._engine: Any = Ab12(seed=seed)
        self._total_turns = 0
        self._last_action_was_reset = False
        self._prev_level_index = 0
        self._consecutive_resets = 0
        self._done = False
        self._game_won = False
        self._game_over = False

    def _build_text_obs(self) -> str:
        e = self._engine
        parts = []
        parts.append(f"Mode: {e._level_mode} | Lives: {e._lives} | Moves: {e._moves_remaining}/{e._max_moves}")
        parts.append(f"Sources: {e._sources_remaining}")
        parts.append(f"Cursor: {e._cursor_pos}")
        parts.append(f"Selected: {e._selected_index}")
        parts.append(f"Inventory: {e._inventory}")
        parts.append(f"Target: {e._target_pattern}")
        parts.append(f"Level: {e.level_index + 1}/{len(e._levels)}")
        if e._level_mode == "MIX":
            parts.append("Mix Guide: " + ", ".join(
                f"{c1}+{c2}={r}" for r, combos in GUIDE_GROUPS for c1, c2 in combos
            ))
        return "\n".join(parts)

    @staticmethod
    def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
        chunk = chunk_type + data
        crc = struct.pack(">I", zlib.crc32(chunk) & 0xFFFFFFFF)
        return struct.pack(">I", len(data)) + chunk + crc

    def _build_image_bytes(self) -> Optional[bytes]:
        eng = self._engine
        frame = eng.camera.render(eng.current_level.get_sprites())
        rgb = np.zeros((FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8)
        for idx, color in enumerate(ARC_PALETTE):
            mask = frame == idx
            rgb[mask] = color
        raw = b""
        for row in range(FRAME_SIZE):
            raw += b"\x00" + rgb[row].tobytes()
        ihdr = struct.pack(">IIBBBBB", FRAME_SIZE, FRAME_SIZE, 8, 2, 0, 0, 0)
        png = b"\x89PNG\r\n\x1a\n"
        png += self._png_chunk(b"IHDR", ihdr)
        png += self._png_chunk(b"IDAT", zlib.compress(raw))
        png += self._png_chunk(b"IEND", b"")
        return png

    def _build_game_state(self, done: bool = False) -> GameState:
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
                "game_over": self._game_over,
                "done": done,
                "info": {},
            },
        )

    def reset(self) -> GameState:
        e = self._engine
        self._consecutive_resets += 1
        if self._consecutive_resets >= 2:
            self._consecutive_resets = 0
            e.set_level(0)
        else:
            reset_input = ActionInput(id=GameAction.RESET)
            e.perform_action(reset_input)
        e._lives = 3
        self._total_turns = 0
        self._done = False
        self._game_won = False
        self._game_over = False
        self._prev_level_index = e.level_index
        return self._build_game_state()

    def is_done(self) -> bool:
        return self._engine._state == EngineGameState.WIN or self._engine._state == EngineGameState.GAME_OVER

    def get_actions(self) -> List[str]:
        if self._done:
            return ["reset"]
        return ["reset", "up", "down", "right", "select", "undo"]

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
        self._consecutive_resets = 0
        self._total_turns += 1

        game_action = self._ACTION_MAP[action]
        info: Dict[str, Any] = {"action": action}

        prev_level = e.level_index
        action_input = ActionInput(id=game_action)
        e.perform_action(action_input, raw=True)

        game_won = e._state == EngineGameState.WIN
        total_levels = len(e._levels)
        level_completed = game_won or (e.level_index > prev_level)
        reward = (1.0 / total_levels) if level_completed else 0.0

        if game_won:
            self._done = True
            self._game_won = True
            self._game_over = False
            info["reason"] = "game_complete"
            return StepResult(
                state=self._build_game_state(done=True),
                reward=reward,
                done=True,
                info=info,
            )

        if e._state == EngineGameState.GAME_OVER:
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

        self._prev_level_index = e.level_index

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
        arr = (
            np.array(index_grid, dtype=np.uint8)
            if not isinstance(index_grid, np.ndarray)
            else index_grid.astype(np.uint8)
        )
        h, w = arr.shape[:2]
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for idx, color in enumerate(ARC_PALETTE):
            mask = arr == idx
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

    @staticmethod
    def _resize_nearest(frame: np.ndarray, h: int, w: int) -> np.ndarray:
        src_h, src_w = frame.shape[0], frame.shape[1]
        row_idx = (np.arange(h) * src_h // h).astype(int)
        col_idx = (np.arange(w) * src_w // w).astype(int)
        return frame[np.ix_(row_idx, col_idx)]

    def _get_obs(self) -> np.ndarray:
        frame = self._env.render(mode="rgb_array")
        if frame.shape[0] != self.OBS_HEIGHT or frame.shape[1] != self.OBS_WIDTH:
            frame = self._resize_nearest(frame, self.OBS_HEIGHT, self.OBS_WIDTH)
        return frame

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
