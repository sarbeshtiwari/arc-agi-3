import random
import struct
from copy import deepcopy
from dataclasses import dataclass, field
from math import gcd
from typing import Any, Dict, List, Optional, Tuple
import gymnasium as gym
import numpy as np
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
import zlib

ARC_PALETTE: Dict[int, Tuple[int, int, int]] = {
    0: (0, 0, 0),
    1: (0, 116, 217),
    2: (255, 65, 54),
    3: (46, 204, 64),
    4: (255, 220, 0),
    5: (170, 170, 170),
    6: (240, 18, 190),
    7: (255, 133, 27),
    8: (127, 219, 255),
    9: (135, 12, 37),
    10: (0, 48, 73),
    11: (106, 76, 48),
    12: (255, 182, 193),
    13: (80, 80, 80),
    14: (50, 205, 50),
    15: (128, 0, 128),
}

ACTION_FROM_NAME: Dict[str, GameAction] = {
    "up": GameAction.ACTION1,
    "down": GameAction.ACTION2,
    "left": GameAction.ACTION3,
    "right": GameAction.ACTION4,
    "select": GameAction.ACTION5,
    "undo": GameAction.ACTION7,
    "reset": GameAction.RESET,
}

LEVEL_SEEDS = [42, 137, 256, 999, 1337, 2024, 7777, 31415]

BG = 0
BORDER = 5
LIFE_COL = 8
LIFE_LOST_COL = 4
BAR_OUTLINE = 3

MOVE_BAR_COL = 2
CURSOR_COL = 13
CURSOR_ROW = 11
MATCHED = 14

LIVES_PER_LEVEL = [5, 5, 5, 4, 3, 3]

BUDGET_PER_LEVEL = [48, 64, 80, 96, 124, 140]

BAR_COLORS = [9, 6, 7, 8, 10, 15, 12]

GRID_W = 32
GRID_H = 32
BORDER_COL_L = 0
BORDER_COL_R = 31
BORDER_ROW_T = 0
BORDER_ROW_B = 31
PTR_X = 1
BAR_X = 2
BAR_MAX_LEN = 28
INNER_TOP = 1
INNER_BOTTOM = 30

def bar_len(num: int, den: int) -> int:
    return max(1, round(num / den * BAR_MAX_LEN))


def build_pool() -> List[Tuple[int, int, int]]:
    seen_lengths: Dict[int, Tuple[int, int]] = {}
    for den in range(2, 16):
        for num in range(1, den):
            g = gcd(num, den)
            rn, rd = num // g, den // g
            bl = bar_len(rn, rd)
            if bl not in seen_lengths:
                seen_lengths[bl] = (rn, rd)
    return sorted((bl, n, d) for bl, (n, d) in seen_lengths.items())


POOL = build_pool()

LEVEL_PAIR_INDICES = [
    [1, 3, 6],
    [1, 3, 5, 7, 9],
    [1, 3, 5, 7, 9, 11],
    [0, 2, 4, 6, 8, 10, 11],
    [0, 1, 3, 5, 7, 9, 10, 11],
    list(range(12)),
]

LEVEL_N_PAIRS = [2, 3, 4, 5, 6, 7]


def level_fractions(level_idx: int) -> List[Tuple[int, int, int]]:
    n = LEVEL_N_PAIRS[level_idx]
    candidates = [POOL[i % len(POOL)] for i in LEVEL_PAIR_INDICES[level_idx]]
    seen: Dict[int, Tuple[int, int]] = {}
    for bl, num, den in candidates:
        if bl not in seen:
            seen[bl] = (num, den)
    unique = [(bl, n2, d2) for bl, (n2, d2) in seen.items()]
    unique.sort(key=lambda t: t[0])
    return unique[:n]


@dataclass
class GameState:
    text_observation: str = ""
    image_observation: Optional[bytes] = None
    valid_actions: Optional[List[str]] = None
    turn: int = 0
    metadata: Dict = field(default_factory=dict)


@dataclass
class StepResult:
    state: GameState = field(default_factory=GameState)
    reward: float = 0.0
    done: bool = False
    info: Dict = field(default_factory=dict)


class Fm01(ARCBaseGame):

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self.game_seed = random.Random(seed).choice(LEVEL_SEEDS)
        self._rng = random.Random(self.game_seed)
        self.bars: List[dict] = []
        self.cursor_row: int = 0
        self.selected_row: Optional[int] = None
        self.matches: int = 0
        self.n_pairs: int = 0
        self.ready: bool = False
        self.active_pair_id: int = 0
        self._lives: int = 5
        self._lives_max: int = 5
        self.budget: int = 0
        self.budget_max: int = 0
        self.has_played: bool = False
        self.preserve_lives: bool = False
        self.just_reset: bool = False
        self.undo_stack: List[dict] = []
        self._initialized: bool = False

        camera = Camera(
            background=BG,
            letter_box=BORDER,
            width=GRID_W,
            height=GRID_H,
        )
        levels = [Level(sprites=[], grid_size=(GRID_W, GRID_H)) for _ in range(6)]
        super().__init__(
            game_id="fm01",
            levels=levels,
            camera=camera,
            available_actions = [0, 1, 2, 3, 4, 5, 7]
        )
        self._initialized = True

    def on_set_level(self, level: Level) -> None:
        if not self._initialized:
            return

        self.current_level.remove_all_sprites()
        self.undo_stack.clear()

        idx = self.level_index

        if self.preserve_lives:
            self.preserve_lives = False
        else:
            self._lives_max = LIVES_PER_LEVEL[idx] if idx < len(LIVES_PER_LEVEL) else 3
            self._lives = self._lives_max

        self.budget_max = BUDGET_PER_LEVEL[idx] if idx < len(BUDGET_PER_LEVEL) else 64
        self.budget = self.budget_max
        fractions = level_fractions(idx)
        n_pairs = len(fractions)
        self.n_pairs = n_pairs
        self.matches = 0
        self.selected_row = None

        self._rng = random.Random(LEVEL_SEEDS[idx % len(LEVEL_SEEDS)] + self.game_seed)

        bars: List[dict] = []
        for pair_id, (bl, num, den) in enumerate(fractions):
            colour = BAR_COLORS[pair_id % len(BAR_COLORS)]
            for _ in range(2):
                bars.append(
                    {
                        "bar_len": bl,
                        "num": num,
                        "den": den,
                        "colour": colour,
                        "pair_id": pair_id,
                        "matched": False,
                    }
                )

        self._rng.shuffle(bars)

        for row_idx, bar in enumerate(bars):
            bar["row"] = row_idx

        self.bars = bars
        self.cursor_row = 0
        self.active_pair_id = 0
        self.ready = True
        self.render_frame()

    def save_undo(self) -> None:
        self.undo_stack.append({
            "bars": deepcopy(self.bars),
            "cursor_row": self.cursor_row,
            "selected_row": self.selected_row,
            "matches": self.matches,
            "active_pair_id": self.active_pair_id,
            "lives": self._lives,
        })
        if len(self.undo_stack) > 50:
            self.undo_stack.pop(0)

    def restore_undo(self) -> bool:
        if not self.undo_stack:
            return False
        snap = self.undo_stack.pop()
        self.bars = snap["bars"]
        self.cursor_row = snap["cursor_row"]
        self.selected_row = snap["selected_row"]
        self.matches = snap["matches"]
        self.active_pair_id = snap["active_pair_id"]
        self._lives = snap["lives"]
        return True

    def _step_undo(self) -> None:
        self.restore_undo()
        self.budget = max(0, self.budget - 1)
        self.render_frame()
        if self.budget <= 0:
            self.lose_life()

    def _step_move_cursor(self, action_id: "GameAction", n_rows: int) -> None:
        if action_id == GameAction.ACTION1:
            self.cursor_row = max(0, self.cursor_row - 1)
        elif action_id == GameAction.ACTION2:
            self.cursor_row = min(n_rows - 1, self.cursor_row + 1)
        elif action_id == GameAction.ACTION3:
            for r in range(self.cursor_row - 1, -1, -1):
                if not self.bars[r]["matched"]:
                    self.cursor_row = r
                    break
        elif action_id == GameAction.ACTION4:
            for r in range(self.cursor_row + 1, n_rows):
                if not self.bars[r]["matched"]:
                    self.cursor_row = r
                    break

    def _step_select(self) -> bool:
        bar_here = self.bars[self.cursor_row]
        is_locked = (
            not bar_here["matched"] and bar_here["pair_id"] != self.active_pair_id
        )

        if bar_here["matched"]:
            return False

        if is_locked:
            self.selected_row = None
            self._lives -= 1
            return True

        if self.selected_row is None:
            self.selected_row = self.cursor_row
            return False

        if self.selected_row == self.cursor_row:
            self.selected_row = None
            return False

        bar_sel = self.bars[self.selected_row]
        if bar_here["pair_id"] == bar_sel["pair_id"]:
            bar_here["matched"] = True
            bar_sel["matched"] = True
            self.matches += 1
            self.selected_row = None
            self.active_pair_id = self.next_active_pair_id()
            return False

        self.selected_row = None
        self._lives -= 1
        return True

    def _check_win_or_budget(self, life_lost_this_step: bool) -> None:
        if self.matches >= self.n_pairs:
            if self.level_index < len(self._levels) - 1:
                self.next_level()
            else:
                self.win()
            return

        if self.budget <= 0 and not life_lost_this_step:
            self.lose_life()

    def step(self) -> None:
        if not self.ready:
            self.complete_action()
            return

        if self.action and self.action.id == GameAction.RESET:
            self.complete_action()
            return

        self.has_played = True
        self.just_reset = False

        action = self.action
        n_rows = len(self.bars)
        life_lost_this_step = False

        if action.id == GameAction.ACTION7:
            self._step_undo()
            self.complete_action()
            return

        self.budget = max(0, self.budget - 1)
        self.save_undo()

        if action.id in (GameAction.ACTION1, GameAction.ACTION2,
                         GameAction.ACTION3, GameAction.ACTION4):
            self._step_move_cursor(action.id, n_rows)
        elif action.id == GameAction.ACTION5:
            life_lost_this_step = self._step_select()
            if life_lost_this_step and self._lives <= 0:
                self.render_frame()
                self.lose()
                self.complete_action()
                return

        self.render_frame()
        self._check_win_or_budget(life_lost_this_step)
        self.complete_action()

    def handle_reset(self) -> None:
        if self._state == EngineGameState.GAME_OVER:
            self.just_reset = False
            self.preserve_lives = False
            self.level_reset()
        elif self._state == EngineGameState.WIN or not self.has_played:
            self.just_reset = False
            self.preserve_lives = False
            self.full_reset()
        elif self.level_index == 0:
            self.just_reset = False
            self.preserve_lives = False
            self.full_reset()
        elif self.just_reset:
            self.just_reset = False
            self.preserve_lives = False
            self.full_reset()
        else:
            self.just_reset = True
            self.preserve_lives = False
            self.level_reset()

    def next_active_pair_id(self) -> int:
        matched_ids = {bar["pair_id"] for bar in self.bars if bar["matched"]}
        for pid in range(self.n_pairs):
            if pid not in matched_ids:
                return pid
        return self.n_pairs

    def lose_life(self) -> bool:
        self._lives -= 1
        if self._lives <= 0:
            self.render_frame()
            self.lose()
            return True
        self.reset_level()
        return False

    def reset_level(self) -> None:
        self.preserve_lives = True
        self.level_reset()

    def render_frame(self) -> None:
        self.current_level.remove_all_sprites()

        n_rows = len(self.bars)

        usable_rows = INNER_BOTTOM - INNER_TOP + 1
        spacing = max(1, usable_rows // max(n_rows, 1))

        for bar in self.bars:
            row_idx = bar["row"]
            gy = INNER_TOP + min(row_idx * spacing, usable_rows - 1)
            bar["_gy"] = gy

            bl = bar["bar_len"]
            is_matched = bar["matched"]
            is_locked = not is_matched and bar["pair_id"] != self.active_pair_id
            is_selected = (
                self.selected_row is not None and self.selected_row == row_idx
            )
            is_cursor = self.cursor_row == row_idx

            if is_matched:
                fill = MATCHED
            elif is_selected:
                fill = CURSOR_ROW
            else:
                fill = bar["colour"]

            if is_cursor and is_selected:
                self.px(PTR_X, gy, CURSOR_ROW, f"ptr_{row_idx}", layer=4)
            elif is_cursor:
                self.px(PTR_X, gy, CURSOR_COL, f"ptr_{row_idx}", layer=4)
            elif is_selected:
                self.px(PTR_X, gy, CURSOR_ROW, f"ptr_{row_idx}", layer=3)

            for x in range(BAR_X, BAR_X + BAR_MAX_LEN):
                self.px(x, gy, BAR_OUTLINE, f"track_{row_idx}_{x}", layer=0)

            for x in range(BAR_X, BAR_X + bl):
                self.px(x, gy, fill, f"bar_{row_idx}_{x}", layer=1)

        self.draw_border()

    def draw_border(self) -> None:
        for x in range(GRID_W):
            self.px(x, BORDER_ROW_T, BORDER, f"brt_{x}", layer=5)
        for x in range(GRID_W):
            self.px(x, BORDER_ROW_B, BORDER, f"brb_{x}", layer=5)
        for y in range(1, GRID_H - 1):
            self.px(BORDER_COL_L, y, BORDER, f"brl_{y}", layer=5)
            self.px(BORDER_COL_R, y, BORDER, f"brr_{y}", layer=5)

        lives_max = self._lives_max
        dot_spacing = 2
        total_dot_width = lives_max + (lives_max - 1) * (dot_spacing - 1)
        inner_w = GRID_W - 2
        start_x = 1 + (inner_w - total_dot_width) // 2
        lost = lives_max - self._lives
        for i in range(lives_max):
            col = LIFE_LOST_COL if i < lost else LIFE_COL
            self.px(start_x + i * dot_spacing, BORDER_ROW_T, col, f"life_{i}", layer=6)

        bar_slots = GRID_W - 2
        if self.budget_max > 0:
            filled = round((self.budget / self.budget_max) * bar_slots)
        else:
            filled = 0
        for slot in range(bar_slots):
            col = MOVE_BAR_COL if slot < filled else LIFE_LOST_COL
            self.px(1 + slot, BORDER_ROW_B, col, f"bar_{slot}", layer=6)

    def px(self, x: int, y: int, color: int, name: str, layer: int = 0) -> None:
        if not (0 <= x < GRID_W and 0 <= y < GRID_H):
            return
        sp = Sprite(
            pixels=[[color]],
            name=name,
            visible=True,
            collidable=False,
            layer=layer,
        )
        sp.set_position(x, y)
        self.current_level.add_sprite(sp)

    @property
    def extra_state(self) -> dict:
        n_pairs = self.n_pairs
        matches = self.matches
        remaining = n_pairs - matches

        bar_info = []
        for bar in sorted(self.bars, key=lambda b: b["row"]):
            bar_info.append(
                {
                    "row": bar["row"],
                    "bar_len": bar["bar_len"],
                    "matched": bar["matched"],
                    "selected": self.selected_row == bar["row"],
                    "cursor": self.cursor_row == bar["row"],
                }
            )

        return {
            "remaining_cells": remaining,
            "circuit_title": f"Fraction Matching – Level {self.level_index + 1}",
            "pairs_matched": matches,
            "pairs_total": n_pairs,
            "pairs_remaining": remaining,
            "cursor_row": self.cursor_row,
            "selected_row": self.selected_row,
            "bars": bar_info,
            "lives": self._lives,
            "lives_max": self._lives_max,
            "budget_remaining": self.budget,
            "budget_max": self.budget_max,
            "level_features": [
                f"Lives: {self._lives}/{self._lives_max}",
                f"Budget: {self.budget}/{self.budget_max}",
                f"Pairs: {matches}/{n_pairs} matched",
                f"Remaining: {remaining}",
            ],
        }


VALID_ACTIONS: list[str] = ["reset", "up", "down", "left", "right", "select", "undo"]


class PuzzleEnvironment:

    def __init__(self, seed: int = 0) -> None:
        self._engine: Fm01 = Fm01(seed=seed)
        self._total_levels: int = len(self._engine._levels)
        self._total_turns: int = 0
        self._last_action_was_reset: bool = False

    def frame_to_png(self, frame) -> bytes | None:
        try:
            arr = np.array(frame, dtype=np.uint8)
            h, w = arr.shape[:2]
            rgb = np.zeros((h, w, 3), dtype=np.uint8)
            for idx, color in ARC_PALETTE.items():
                mask = arr == idx
                if mask.ndim == 3:
                    mask = mask.all(axis=2)
                rgb[mask] = color

            def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
                chunk = chunk_type + data
                return struct.pack(">I", len(data)) + chunk + struct.pack(">I", zlib.crc32(chunk) & 0xFFFFFFFF)

            raw_rows = b"".join(b"\x00" + rgb[r].tobytes() for r in range(h))
            return (
                b"\x89PNG\r\n\x1a\n"
                + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                + _png_chunk(b"IDAT", zlib.compress(raw_rows))
                + _png_chunk(b"IEND", b"")
            )
        except Exception:
            return None

    def build_text_observation(self) -> str:
        g = self._engine
        idx = g.level_index
        n_pairs = g.n_pairs
        matches = g.matches
        bars = sorted(g.bars, key=lambda b: b["row"])
        lines = [
            f"=== Fraction Matching (Level {idx + 1}/{self._total_levels}) ===",
            f"Pairs matched: {matches}/{n_pairs}",
            f"Budget: {g.budget}/{g.budget_max}",
            f"Lives: {g._lives}/{g._lives_max}",
            f"Cursor row: {g.cursor_row}",
            f"Selected: {g.selected_row}",
            "--- Bars ---",
        ]
        for bar in bars:
            if bar["matched"]:
                status = "MATCHED"
            elif bar["pair_id"] != g.active_pair_id:
                status = "LOCKED"
            else:
                status = "UNLOCKED"
            cursor = " <CURSOR>" if g.cursor_row == bar["row"] else ""
            sel = " <SELECTED>" if g.selected_row == bar["row"] else ""
            lines.append(
                f"  Row {bar['row']}: len={bar['bar_len']} {status}{cursor}{sel}"
            )
        if g._state == EngineGameState.WIN:
            lines.append("State: won")
        elif g._state == EngineGameState.GAME_OVER:
            lines.append("State: game-over")
        else:
            lines.append("State: playing")
        return "\n".join(lines)

    def build_metadata(self) -> dict:
        g = self._engine
        return {
            "game_id": g.game_id,
            "level_index": g.level_index,
            "total_levels": self._total_levels,
            "pairs_matched": g.matches,
            "pairs_total": g.n_pairs,
            "budget_remaining": g.budget,
            "budget_max": g.budget_max,
            "lives": g._lives,
            "lives_max": g._lives_max,
            "cursor_row": g.cursor_row,
        }

    def make_game_state(self, frame_data) -> GameState:
        image_bytes: bytes | None = None
        if frame_data and not frame_data.is_empty():
            image_bytes = self.frame_to_png(frame_data.frame)

        done = self._engine._state in (EngineGameState.WIN, EngineGameState.GAME_OVER)
        valid = None if done else VALID_ACTIONS

        return GameState(
            text_observation=self.build_text_observation(),
            image_observation=image_bytes,
            valid_actions=valid,
            turn=self._total_turns,
            metadata=self.build_metadata(),
        )

    def reset(self) -> GameState:
        self._total_turns = 0
        self._last_action_was_reset = False
        frame_data = self._engine.perform_action(ActionInput(id=GameAction.RESET))
        return self.make_game_state(frame_data)

    def get_actions(self) -> list[str]:
        done = self._engine._state in (EngineGameState.WIN, EngineGameState.GAME_OVER)
        if done:
            return ["reset"]
        return VALID_ACTIONS

    def is_done(self) -> bool:
        return self._engine._state in (EngineGameState.WIN, EngineGameState.GAME_OVER)

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError(f"Unsupported render mode '{mode}'. Only 'rgb_array' is supported.")
        frame: np.ndarray = self._engine.camera.render(self._engine.current_level.get_sprites())
        h, w = frame.shape[:2]
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for idx_val, color in ARC_PALETTE.items():
            mask = frame == idx_val
            if mask.ndim == 3:
                mask = mask.all(axis=2)
            rgb[mask] = color
        if h < 64 or w < 64:
            scale_h = 64 // h
            scale_w = 64 // w
            rgb = np.kron(rgb, np.ones((scale_h, scale_w, 1))).astype(np.uint8)
        return rgb

    def close(self) -> None:
        self._engine = None

    def step(self, action: str) -> StepResult:
        action = action.strip().lower()
        game_action = ACTION_FROM_NAME.get(action)
        if game_action is None:
            raise ValueError(
                f"Unknown action '{action}'. Valid: {VALID_ACTIONS}"
            )

        if game_action == GameAction.RESET:
            state = self.reset()
            return StepResult(
                state=state,
                reward=0.0,
                done=False,
                info={"action": action},
            )

        self._last_action_was_reset = False
        self._total_turns += 1

        prev_level = self._engine.level_index
        prev_lives = self._engine._lives

        action_input = ActionInput(id=game_action)
        frame_data = self._engine.perform_action(action_input)

        engine_state = frame_data.state
        done = engine_state in (EngineGameState.WIN, EngineGameState.GAME_OVER)

        reward = 0.0
        if engine_state == EngineGameState.WIN:
            reward = 1.0 / self._total_levels
        elif self._engine.level_index > prev_level:
            reward = 1.0 / self._total_levels

        info = {
            "action": action,
            "engine_state": engine_state,
            "level_changed": self._engine.level_index != prev_level,
            "life_lost": self._engine._lives < prev_lives,
            "full_reset": frame_data.full_reset,
        }

        return StepResult(
            state=self.make_game_state(frame_data),
            reward=reward,
            done=done,
            info=info,
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
