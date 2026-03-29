import random
import struct
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import gymnasium as gym
from gymnasium import spaces
import numpy as np
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

_ARC_PALETTE: Dict[int, Tuple[int, int, int]] = {
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

_ACTION_NAMES: Dict[int, str] = {
    1: "up",
    2: "down",
    3: "left",
    4: "right",
    5: "select",
    7: "undo",
}

_ACTION_FROM_NAME: Dict[str, GameAction] = {
    "reset": GameAction.RESET,
    "up": GameAction.ACTION1,
    "down": GameAction.ACTION2,
    "left": GameAction.ACTION3,
    "right": GameAction.ACTION4,
    "select": GameAction.ACTION5,
    "undo": GameAction.ACTION7,
}

_LEVEL_SEEDS = [42, 137, 256, 999]

BG = 0
DIM = 1
LT_GREY = 2
GREY = 3
DARK = 4
BORDER = 5
MAGENTA = 6
PINK = 7
RED = 8
BLUE = 9
YELLOW = 11
ORANGE = 12
GREEN = 14
PURPLE = 15

PRESSURE_COLOR: Dict[int, int] = {
    0: BG,
    1: DIM,
    2: LT_GREY,
    3: GREY,
    4: DARK,
    5: ORANGE,
    6: PINK,
    7: PURPLE,
    8: BLUE,
    9: MAGENTA,
}

GRID_W = 16
GRID_H = 16
GRID_ROWS = 16
GRID_COLS = 6

LEFT_R0 = 0
LEFT_C0 = 0
RIGHT_R0 = 0
RIGHT_C0 = 8
BUDGET_C = 6
DIVIDER_C = 7
GAP_R_C = 14
LIVES_C = 15

LIVES_PER_LEVEL = [5, 5, 4, 3, 3]


def _compute_pressure(
    placements: Set[Tuple[int, int]],
    weight_mag: int,
    spread: int,
    rows: int = GRID_ROWS,
    cols: int = GRID_COLS,
) -> List[List[int]]:
    pressure = [[0] * cols for _ in range(rows)]
    for wr, wc in placements:
        for r in range(rows):
            for c in range(cols):
                dist = abs(r - wr) + abs(c - wc)
                if dist == 0:
                    pressure[r][c] += weight_mag
                elif dist == 1 and spread >= 1:
                    pressure[r][c] += weight_mag // 2
                elif dist == 2 and spread >= 2:
                    pressure[r][c] += weight_mag // 4
    return pressure


def _make_target(coords, mag, spread):
    return _compute_pressure(set(coords), mag, spread)

def _decode_ref(encoded):
    _k = 0x35
    return [(r ^ _k, c ^ _k) for r, c in encoded]

LEVEL_SPECS: List[dict] = [
    {
        "title": "First Weights",
        "weight_mag": 3,
        "spread": 0,
        "budget": 5,
        "_ref_enc": [(55, 52), (61, 49), (59, 55)],
    },
    {
        "title": "Four Points",
        "weight_mag": 4,
        "spread": 0,
        "budget": 6,
        "_ref_enc": [(52, 52), (52, 49), (56, 52), (56, 49)],
    },
    {
        "title": "Ripple Clusters",
        "weight_mag": 4,
        "spread": 1,
        "budget": 7,
        "_ref_enc": [(54, 55), (54, 49), (61, 54), (57, 52), (57, 49)],
    },
    {
        "title": "Shockwave",
        "weight_mag": 5,
        "spread": 2,
        "budget": 6,
        "_ref_enc": [(52, 52), (48, 49), (63, 55), (56, 49)],
    },
    {
        "title": "Full Pressure",
        "weight_mag": 4,
        "spread": 2,
        "budget": 7,
        "_ref_enc": [(52, 55), (49, 53), (50, 49), (63, 52), (56, 54)],
    },
]

N_LEVELS = len(LEVEL_SPECS)


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


class Pm01(ARCBaseGame):

    _BLINK_PERIOD = 6

    def __init__(self, seed: int = 0) -> None:
        camera = Camera(
            background=BG,
            letter_box=BORDER,
            width=GRID_W,
            height=GRID_H,
        )
        levels = [Level(sprites=[], grid_size=(GRID_W, GRID_H)) for _ in range(N_LEVELS)]
        super().__init__(
            game_id="pm01",
            levels=levels,
            camera=camera,
            available_actions=[0,1,2,3,4,5,7]
        )
        self._seed = seed
        self._game_seed = random.Random(seed).choice(_LEVEL_SEEDS)
        self._rng = random.Random(self._game_seed)
        self._spec: dict = {}
        self._placements: Set[Tuple[int, int]] = set()
        self._pressure: List[List[int]] = []
        self._target: List[List[int]] = []
        self._budget: int = 0
        self._budget_max: int = 0
        self._overflow: bool = False
        self._flash_win: bool = False
        self._flash_over: bool = False
        self._lives: int = 5
        self._lives_max: int = 5
        self._cur_r: int = 0
        self._cur_c: int = 0
        self._blink: int = 0
        self._cur_vis: bool = True
        self._action_limit: int = 0
        self._actions_used: int = 0
        self._ready: bool = False
        self._has_played: bool = False
        self._preserve_lives: bool = False
        self._just_reset: bool = False
        self._level_just_won: bool = False
        self._undo_stack: List[dict] = []

    def _clamp(self, r: int, c: int) -> Tuple[int, int]:
        return (
            max(0, min(GRID_ROWS - 1, r)),
            max(0, min(GRID_COLS - 1, c)),
        )

    def _recompute(self) -> None:
        spec = self._spec
        self._pressure = _compute_pressure(
            self._placements, spec["weight_mag"], spec["spread"]
        )
        self._overflow = any(
            self._pressure[r][c] > 9 for r in range(GRID_ROWS) for c in range(GRID_COLS)
        )

    def _is_solved(self) -> bool:
        if self._overflow:
            return False
        return all(
            self._pressure[r][c] == self._target[r][c]
            for r in range(GRID_ROWS)
            for c in range(GRID_COLS)
        )

    def on_set_level(self, level: Level) -> None:
        if not hasattr(self, "_ready"):
            return
        self.current_level.remove_all_sprites()
        self._undo_stack.clear()
        idx = self._current_level_index
        spec = LEVEL_SPECS[idx]
        self._spec = spec
        ref_coords = _decode_ref(spec["_ref_enc"])
        computed_target = _make_target(ref_coords, spec["weight_mag"], spec["spread"])
        self._target = [row[:] for row in computed_target]
        self._placements = set()
        self._budget = spec["budget"]
        self._budget_max = spec["budget"]
        self._overflow = False
        self._flash_win = False
        self._flash_over = False

        if self._preserve_lives:
            self._preserve_lives = False
        else:
            self._lives_max = LIVES_PER_LEVEL[idx] if idx < len(LIVES_PER_LEVEL) else 3
            self._lives = self._lives_max

        self._rng = random.Random(_LEVEL_SEEDS[idx % len(_LEVEL_SEEDS)] + self._game_seed)
        self._cur_r = self._rng.randint(0, GRID_ROWS - 1)
        self._cur_c = self._rng.randint(0, GRID_COLS - 1)
        self._blink = 0
        self._cur_vis = True
        self._action_limit = spec["budget"] * 15
        self._actions_used = 0
        self._recompute()
        self._render()
        self._ready = True

    def _lose_life(self) -> bool:
        self._lives -= 1
        if self._lives <= 0:
            self._render()
            self.lose()
            return True
        self._placements = set()
        self._budget = self._budget_max
        self._overflow = False
        self._flash_over = False
        self._recompute()
        return False

    def _try_win(self) -> bool:
        if not self._is_solved():
            return False
        self._flash_win = True
        self._render()
        if self._current_level_index < len(self._levels) - 1:
            self._level_just_won = True
            self.next_level()
            self.complete_action()
        else:
            self.win()
            self.complete_action()
        return True

    def _save_undo(self) -> None:
        self._undo_stack.append({
            "placements": set(self._placements),
            "pressure": [row[:] for row in self._pressure],
            "budget": self._budget,
            "lives": self._lives,
            "cur_r": self._cur_r,
            "cur_c": self._cur_c,
            "overflow": self._overflow,
            "actions_used": self._actions_used,
        })
        if len(self._undo_stack) > 50:
            self._undo_stack.pop(0)

    def _restore_undo(self) -> bool:
        if not self._undo_stack:
            return False
        current_budget = self._budget
        current_actions = self._actions_used
        snap = self._undo_stack.pop()
        self._placements = snap["placements"]
        self._pressure = snap["pressure"]
        self._budget = current_budget
        self._cur_r = snap["cur_r"]
        self._cur_c = snap["cur_c"]
        self._overflow = snap["overflow"]
        self._actions_used = current_actions
        return True

    def _step_undo(self) -> None:
        self._restore_undo()
        self._budget = max(0, self._budget - 1)
        self._actions_used += 1
        self._render()
        if self._budget <= 0:
            if self._lose_life():
                self.complete_action()
                return
        self.complete_action()

    def _step_move(self, action_id: GameAction) -> None:
        if action_id == GameAction.ACTION1:
            self._cur_r, self._cur_c = self._clamp(self._cur_r - 1, self._cur_c)
        elif action_id == GameAction.ACTION2:
            self._cur_r, self._cur_c = self._clamp(self._cur_r + 1, self._cur_c)
        elif action_id == GameAction.ACTION3:
            self._cur_r, self._cur_c = self._clamp(self._cur_r, self._cur_c - 1)
        elif action_id == GameAction.ACTION4:
            self._cur_r, self._cur_c = self._clamp(self._cur_r, self._cur_c + 1)

    def _step_place(self) -> bool:
        cell = (self._cur_r, self._cur_c)
        if cell in self._placements:
            self._placements.discard(cell)
            self._budget = min(self._budget_max, self._budget + 1)
        elif self._budget > 0:
            self._placements.add(cell)
            self._budget -= 1

        self._recompute()

        if self._overflow:
            self._flash_over = True
            self._render()
            self._flash_over = False
            if self._lose_life():
                self.complete_action()
                return True
            self._render()
            self.complete_action()
            return True

        if self._try_win():
            return True

        if self._budget == 0:
            self._flash_over = True
            self._render()
            self._flash_over = False
            if self._lose_life():
                self.complete_action()
                return True
            self._render()
            self.complete_action()
            return True

        return False

    def step(self) -> None:
        if not self._ready:
            self.complete_action()
            return

        if self.action and self.action.id == GameAction.RESET:
            self.complete_action()
            return

        self._has_played = True
        self._just_reset = False
        self._level_just_won = False

        action = self.action

        _KNOWN = (
            GameAction.ACTION1,
            GameAction.ACTION2,
            GameAction.ACTION3,
            GameAction.ACTION4,
            GameAction.ACTION5,
            GameAction.ACTION7,
        )
        if action.id not in _KNOWN:
            self._render()
            self.complete_action()
            return

        if action.id == GameAction.ACTION7:
            self._step_undo()
            return

        self._save_undo()
        self._actions_used += 1

        self._blink += 1
        self._cur_vis = (self._blink % self._BLINK_PERIOD) < (self._BLINK_PERIOD // 2)

        if action.id in (GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4):
            self._step_move(action.id)
        elif action.id == GameAction.ACTION5:
            if self._step_place():
                return

        if self._actions_used >= self._action_limit:
            self._render()
            if self._lose_life():
                self.complete_action()
                return

        self._render()
        self.complete_action()

    def prepare_for_reset(self) -> None:
        self._has_played = False
        self._preserve_lives = False

    def handle_reset(self) -> None:
        if self._state == EngineGameState.GAME_OVER:
            self._just_reset = False
            self._level_just_won = False
            self.level_reset()
        elif self._state == EngineGameState.WIN or not self._has_played or self._level_just_won:
            self._just_reset = False
            self._level_just_won = False
            self.full_reset()
        elif self._current_level_index == 0:
            self._just_reset = False
            self.full_reset()
        elif self._just_reset:
            self._just_reset = False
            self.full_reset()
        else:
            self._just_reset = True
            self._preserve_lives = False
            self.level_reset()

    def _render_frame(self) -> np.ndarray:
        frame = np.full((GRID_H, GRID_W), BG, dtype=np.uint8)
        frame[:, DIVIDER_C] = BORDER

        used = self._budget_max - self._budget
        for i in range(self._budget_max):
            frame[i, BUDGET_C] = DARK if i < used else YELLOW

        if self._flash_win:
            frame[:, DIVIDER_C] = GREEN
            frame[:, BUDGET_C] = GREEN
        elif self._flash_over:
            frame[:, DIVIDER_C] = RED
            frame[:, BUDGET_C] = RED

        lost = self._lives_max - self._lives
        for i in range(self._lives_max):
            frame[i, LIVES_C] = DARK if i < lost else RED

        for gr in range(GRID_ROWS):
            for gc in range(GRID_COLS):
                val = self._pressure[gr][gc]
                frame[LEFT_R0 + gr, LEFT_C0 + gc] = (
                    RED if val > 9 else PRESSURE_COLOR.get(val, BG)
                )

        for gr in range(GRID_ROWS):
            for gc in range(GRID_COLS):
                val = self._target[gr][gc]
                frame[RIGHT_R0 + gr, RIGHT_C0 + gc] = PRESSURE_COLOR.get(val, BG)

        self._flash_win = False
        self._flash_over = False
        return frame

    def _render_sprites(self) -> None:
        for wr, wc in self._placements:
            sp = Sprite(
                pixels=[[YELLOW]],
                name=f"w_{wr}_{wc}",
                visible=True,
                collidable=False,
                layer=2,
            )
            sp.set_position(LEFT_C0 + wc, LEFT_R0 + wr)
            self.current_level.add_sprite(sp)

        if not self._cur_vis:
            return

        dr = LEFT_R0 + self._cur_r
        dc = LEFT_C0 + self._cur_c

        cur_sp = Sprite(
            pixels=[[GREEN]],
            name="cursor",
            visible=True,
            collidable=False,
            layer=3,
        )
        cur_sp.set_position(dc, dr)
        self.current_level.add_sprite(cur_sp)

        for (ar, ac), arm_name in [
            ((-1, 0), "cu"),
            ((1, 0), "cd"),
            ((0, -1), "cl"),
            ((0, 1), "cr"),
        ]:
            nr, nc = dr + ar, dc + ac
            in_panel = (
                LEFT_R0 <= nr <= LEFT_R0 + GRID_ROWS - 1
                and LEFT_C0 <= nc <= LEFT_C0 + GRID_COLS - 1
            )
            if in_panel:
                arm_sp = Sprite(
                    pixels=[[DARK]],
                    name=arm_name,
                    visible=True,
                    collidable=False,
                    layer=3,
                )
                arm_sp.set_position(nc, nr)
                self.current_level.add_sprite(arm_sp)

    def _render(self) -> None:
        self.current_level.remove_all_sprites()
        frame = self._render_frame()

        bg_sp = Sprite(
            pixels=frame.tolist(),
            name="background",
            visible=True,
            collidable=False,
            layer=0,
        )
        bg_sp.set_position(0, 0)
        self.current_level.add_sprite(bg_sp)

        self._render_sprites()


_VALID_ACTIONS: list[str] = ["reset", "up", "down", "left", "right","select", "undo"]


class PuzzleEnvironment:

    def __init__(self, seed: int = 0) -> None:
        self._engine = Pm01(seed=seed)
        self._turn: int = 0
        self._last_was_reset: bool = False

    def _frame_to_png(self, frame) -> bytes | None:
        try:
            arr = np.array(frame, dtype=np.uint8)
            h, w = arr.shape[:2]
            rgb = np.zeros((h, w, 3), dtype=np.uint8)
            for idx, color in _ARC_PALETTE.items():
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

    def _build_text_observation(self) -> str:
        g = self._engine
        spec = g._spec
        idx = g._current_level_index
        total_cells = GRID_ROWS * GRID_COLS
        matched_cells = sum(
            1
            for r in range(GRID_ROWS)
            for c in range(GRID_COLS)
            if g._pressure[r][c] == g._target[r][c]
        )
        lines = [
            f"=== Pressure Map (Level {idx + 1}/{N_LEVELS}: {spec.get('title', '')}) ===",
            f"Placed: {len(g._placements)}/{g._budget_max}",
            f"Budget: {g._budget}/{g._budget_max}",
            f"Lives: {g._lives}/{g._lives_max}",
            f"Cursor: ({g._cur_r}, {g._cur_c})",
            f"Matched cells: {matched_cells}/{total_cells} ({int(100 * matched_cells / total_cells)}%)",
            "--- Pressure Grid (current) ---",
        ]
        for r in range(GRID_ROWS):
            row_str = " ".join(str(g._pressure[r][c]) for c in range(GRID_COLS))
            lines.append(f"  {row_str}")
        lines.append("--- Target Grid ---")
        for r in range(GRID_ROWS):
            row_str = " ".join(str(g._target[r][c]) for c in range(GRID_COLS))
            lines.append(f"  {row_str}")
        if g._state == EngineGameState.WIN:
            lines.append("State: won")
        elif g._state == EngineGameState.GAME_OVER:
            lines.append("State: game-over")
        elif g._overflow:
            lines.append("State: overflow")
        else:
            lines.append("State: playing")
        return "\n".join(lines)

    def _build_metadata(self) -> dict:
        g = self._engine
        total_cells = GRID_ROWS * GRID_COLS
        matched_cells = sum(
            1
            for r in range(GRID_ROWS)
            for c in range(GRID_COLS)
            if g._pressure[r][c] == g._target[r][c]
        )
        return {
            "game_id": g._game_id,
            "level_index": g.level_index,
            "total_levels": N_LEVELS,
            "n_placed": len(g._placements),
            "budget_remaining": g._budget,
            "budget_max": g._budget_max,
            "lives": g._lives,
            "lives_max": g._lives_max,
            "matched_cells": matched_cells,
            "total_cells": total_cells,
            "cursor": (g._cur_r, g._cur_c),
        }

    def _make_game_state(self, frame_data) -> GameState:
        image_bytes: bytes | None = None
        if frame_data and not frame_data.is_empty():
            image_bytes = self._frame_to_png(frame_data.frame)

        done = self._engine._state in (EngineGameState.WIN, EngineGameState.GAME_OVER)
        valid = ["reset"] if done else _VALID_ACTIONS

        return GameState(
            text_observation=self._build_text_observation(),
            image_observation=image_bytes,
            valid_actions=valid,
            turn=self._turn,
            metadata=self._build_metadata(),
        )

    def _is_won(self) -> bool:
        return self._engine._state == EngineGameState.WIN

    def reset(self) -> GameState:
        self._turn = 0
        if self._is_won() or self._last_was_reset:
            self._engine._has_played = False
            self._engine._preserve_lives = False
            frame_data = self._engine.perform_action(
                ActionInput(id=GameAction.RESET)
            )
        else:
            frame_data = self._engine.perform_action(
                ActionInput(id=GameAction.RESET)
            )
        self._last_was_reset = True
        return self._make_game_state(frame_data)

    def get_actions(self) -> list[str]:
        done = self._engine._state in (EngineGameState.WIN, EngineGameState.GAME_OVER)
        if done:
            return ["reset"]
        return _VALID_ACTIONS

    def is_done(self) -> bool:
        return self._engine._state in (EngineGameState.WIN, EngineGameState.GAME_OVER)

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        frame: np.ndarray = self._engine.camera.render(self._engine.current_level.get_sprites())
        h, w = frame.shape[:2]
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        for idx_val, color in _ARC_PALETTE.items():
            mask = frame == idx_val
            if mask.ndim == 3:
                mask = mask.all(axis=2)
            rgb[:h, :w][mask] = color
        return rgb

    def close(self) -> None:
        self._engine = None

    def step(self, action: str) -> StepResult:
        action = action.strip().lower()
        game_action = _ACTION_FROM_NAME.get(action)
        if game_action is None:
            raise ValueError(
                f"Unknown action '{action}'. Valid: {_VALID_ACTIONS}"
            )

        if game_action == GameAction.RESET and self._last_was_reset:
            self._last_was_reset = False
            self._engine.prepare_for_reset()
            frame_data = self._engine.perform_action(ActionInput(id=GameAction.RESET))
            return StepResult(
                state=self._make_game_state(frame_data),
                reward=0.0,
                done=False,
                info={"action": action, "engine_state": frame_data.state, "level_changed": True, "life_lost": False, "full_reset": True},
            )

        prev_level = self._engine.level_index
        prev_lives = self._engine._lives

        action_input = ActionInput(id=game_action)
        frame_data = self._engine.perform_action(action_input)

        if game_action == GameAction.RESET:
            self._last_was_reset = True
        else:
            self._last_was_reset = False
            self._turn += 1

        engine_state = frame_data.state
        done = engine_state in (EngineGameState.WIN, EngineGameState.GAME_OVER)

        reward = 0.0
        if engine_state == EngineGameState.WIN:
            reward = 1.0 / N_LEVELS
        elif engine_state == EngineGameState.GAME_OVER:
            reward = 0.0
        elif self._engine.level_index > prev_level:
            reward = 1.0 / N_LEVELS

        info = {
            "action": action,
            "engine_state": engine_state,
            "level_changed": self._engine.level_index != prev_level,
            "life_lost": self._engine._lives < prev_lives,
            "full_reset": frame_data.full_reset,
        }

        return StepResult(
            state=self._make_game_state(frame_data),
            reward=reward,
            done=done,
            info=info,
        )


class ArcGameEnv(gym.Env):

    ACTION_LIST: list[str] = ["reset", "up", "down", "left", "right", "select", "undo"]
    OBS_HEIGHT: int = 64
    OBS_WIDTH: int = 64

    metadata: dict = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, seed: int = 0, render_mode: Optional[str] = None) -> None:
        super().__init__()

        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"Unsupported render_mode '{render_mode}'. "
                f"Supported: {self.metadata['render_modes']}"
            )
        self.render_mode: Optional[str] = render_mode

        self._seed = seed
        self._env: Optional[PuzzleEnvironment] = None
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.OBS_HEIGHT, self.OBS_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = spaces.Discrete(len(self.ACTION_LIST))

        self._action_to_string: dict[int, str] = {i: a for i, a in enumerate(self.ACTION_LIST)}
        self._string_to_action: dict[str, int] = {a: i for i, a in enumerate(self.ACTION_LIST)}

    @staticmethod
    def _resize_nearest(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        h, w = img.shape[:2]
        if h == target_h and w == target_w:
            return img
        row_idx = (np.arange(target_h) * h // target_h).astype(int)
        col_idx = (np.arange(target_w) * w // target_w).astype(int)
        return img[np.ix_(row_idx, col_idx)]

    def _get_obs(self) -> np.ndarray:
        if self._env is None:
            return np.zeros((self.OBS_HEIGHT, self.OBS_WIDTH, 3), dtype=np.uint8)
        raw = self._env.render(mode="rgb_array")
        return self._resize_nearest(raw, self.OBS_HEIGHT, self.OBS_WIDTH)

    def _build_info(self, state: GameState, step_info: Optional[dict] = None) -> dict:
        info = {
            "text_observation": state.text_observation,
            "valid_actions": state.valid_actions,
            "turn": state.turn,
            "game_metadata": state.metadata,
        }
        if step_info is not None:
            info["step_info"] = step_info
        return info

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed
        self._env = PuzzleEnvironment(seed=self._seed)
        state = self._env.reset()
        return self._get_obs(), self._build_info(state)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if action < 0 or action >= len(self.ACTION_LIST):
            raise ValueError(f"Invalid action {action}. Must be in [0, {len(self.ACTION_LIST) - 1}].")
        action_name = self.ACTION_LIST[action]
        result = self._env.step(action_name)
        obs = self._get_obs()
        terminated = result.done
        truncated = False
        info = self._build_info(result.state, result.info)
        return obs, result.reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode == "rgb_array":
            return self._get_obs()
        return None

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None

    def action_mask(self) -> np.ndarray:
        if self._env is None:
            return np.zeros(len(self.ACTION_LIST), dtype=np.int8)
        valid = self._env.get_actions()
        return np.array([a in valid for a in self.ACTION_LIST], dtype=np.int8)
