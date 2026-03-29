import random
import struct
import zlib
from dataclasses import dataclass, field
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

_ACTION_FROM_NAME: Dict[str, GameAction] = {
    "up": GameAction.ACTION1,
    "down": GameAction.ACTION2,
    "left": GameAction.ACTION3,
    "right": GameAction.ACTION4,
    "select": GameAction.ACTION5,
    "undo": GameAction.ACTION7,
    "reset": GameAction.RESET,
}

BG = 0
BORDER = 5
DARK = 4
AGENT_COL = 9
LOCK_ON = 14
LOCK_OFF = 8
HINT_BG = 2
BUDGET_ON = 11
BLOCKER = 4
LIFE_ON = 8
LIFE_OFF = 4

LIVES_PER_LEVEL = [5, 5, 4, 3, 3]

DIGIT_COLORS = [
    10,
    11,
    12,
    13,
    14,
    15,
    6,
    7,
    8,
    9,
]

GRID_W = 16
GRID_H = 16

AGENT_START = (1, 8)

MOVE_X0, MOVE_X1 = 1, 14
MOVE_Y0, MOVE_Y1 = 1, 14


def _decode_targets(encoded: tuple) -> tuple:
    _k = 0x25
    return tuple((v ^ _k) % 10 for v in encoded)


LEVEL_CONFIG = [
    {
        "_targets_enc": (35, 45, 32),
        "dial_pos": [(3, 4), (8, 8), (13, 4)],
        "budget": 120,
        "blockers": [],
    },
    {
        "_targets_enc": (44, 33, 34),
        "dial_pos": [(3, 4), (8, 8), (13, 4)],
        "budget": 108,
        "blockers": [],
    },
    {
        "_targets_enc": (33, 37, 35),
        "dial_pos": [(3, 4), (8, 8), (13, 4)],
        "budget": 84,
        "blockers": [(5, 6), (5, 7)],
    },
    {
        "_targets_enc": (36, 32, 38, 45),
        "dial_pos": [(3, 4), (8, 8), (13, 4), (11, 12)],
        "budget": 144,
        "blockers": [(5, 6), (5, 7), (10, 6), (10, 7)],
    },
    {
        "_targets_enc": (34, 44, 39, 35, 33),
        "dial_pos": [(3, 4), (8, 8), (13, 4), (11, 12), (3, 12)],
        "budget": 188,
        "blockers": [(5, 5), (5, 6), (5, 7), (10, 5), (10, 6), (10, 7)],
    },
]

N_LEVELS = len(LEVEL_CONFIG)


def _dial_to_rotate(
    px: int,
    py: int,
    dx: int,
    dy: int,
    dial_pos: List[Tuple[int, int]],
) -> int:
    nx, ny = px + dx, py + dy
    for i, (cx, cy) in enumerate(dial_pos):
        if nx == cx and ny == cy:
            return i
    return -1


def _is_blocked_cell(
    x: int,
    y: int,
    dial_pos: List[Tuple[int, int]],
    blocker_set: set,
) -> bool:
    for cx, cy in dial_pos:
        if x == cx and y == cy:
            return True
    return (x, y) in blocker_set


def _build_background(
    dials: List[int],
    targets: tuple,
    dial_pos: List[Tuple[int, int]],
    blockers: List[Tuple[int, int]],
    budget: int,
    budget_max: int,
    lives: int,
    lives_max: int,
) -> np.ndarray:
    frame = np.full((GRID_H, GRID_W), BG, dtype=np.uint8)

    frame[0, :] = BORDER
    frame[15, :] = BORDER
    frame[:, 0] = BORDER
    frame[:, 15] = BORDER

    for bx, by in blockers:
        if 0 < bx < GRID_W - 1 and 0 < by < GRID_H - 1:
            frame[by, bx] = BLOCKER

    frame[1, 1:15] = HINT_BG
    available_cols = list(range(1, 15))
    hint_cols = []
    for dx, _ in dial_pos:
        if available_cols:
            col = min(available_cols, key=lambda c: (abs(c - dx), c))
            available_cols.remove(col)
        else:
            col = max(1, min(14, dx))
        hint_cols.append(col)
    for i, col in enumerate(hint_cols):
        frame[1, col] = DIGIT_COLORS[targets[i]]

    for i, (dx, dy) in enumerate(dial_pos):
        frame[dy, dx] = DIGIT_COLORS[dials[i]]

    frame[14, 1:15] = DARK
    for i, col in enumerate(hint_cols):
        matched = dials[i] == targets[i]
        frame[14, col] = LOCK_ON if matched else LOCK_OFF

    if budget_max > 0:
        total_slots = 14
        filled = round((budget / budget_max) * total_slots)
        for slot in range(total_slots):
            frame[0, 1 + slot] = BUDGET_ON if slot < filled else DARK

    lost = lives_max - lives
    for i in range(lives_max):
        frame[15, 1 + i] = LIFE_OFF if i < lost else LIFE_ON

    return frame


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


class Pd01(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._dials: List[int] = []
        self._targets: tuple = ()
        self._dial_pos: List[Tuple[int, int]] = []
        self._blockers: List[Tuple[int, int]] = []
        self._blocker_set: set = set()
        self._budget: int = 0
        self._budget_max: int = 0
        self._agent_pos: Tuple[int, int] = AGENT_START
        self._unlocked: bool = False
        self._lives: int = 5
        self._lives_max: int = 5
        self._undo_stack: List[dict] = []

        self._rng = random.Random(seed)
        self._start_positions: List[List[Tuple[int, int]]] = []
        for cfg in LEVEL_CONFIG:
            occupied = set(map(tuple, cfg["dial_pos"])) | set(
                map(tuple, cfg["blockers"])
            )
            valid = [
                (x, y)
                for x in range(MOVE_X0, MOVE_X1 + 1)
                for y in range(MOVE_Y0, MOVE_Y1 + 1)
                if (x, y) not in occupied
            ]
            self._start_positions.append(self._rng.sample(valid, min(4, len(valid))))
        self._visit_count: List[int] = [0] * N_LEVELS

        camera = Camera(
            x=0,
            y=0,
            width=GRID_W,
            height=GRID_H,
            background=BG,
            letter_box=BORDER,
        )
        levels = [
            Level(sprites=[], grid_size=(GRID_W, GRID_H)) for _ in range(N_LEVELS)
        ]
        super().__init__(
            game_id="pd01",
            levels=levels,
            camera=camera,
            available_actions=[0, 1, 2, 3, 4, 5, 7],
        )

    def on_set_level(self, level: Level) -> None:
        self.current_level.remove_all_sprites()
        self._undo_stack.clear()

        idx = self._current_level_index
        cfg = LEVEL_CONFIG[idx]

        self._targets = _decode_targets(cfg["_targets_enc"])
        self._dial_pos = cfg["dial_pos"]
        self._blockers = cfg["blockers"]
        self._blocker_set = set(map(tuple, cfg["blockers"]))
        self._budget_max = cfg["budget"]
        self._budget = cfg["budget"]
        self._dials = [0] * len(self._targets)
        self._unlocked = False

        positions = self._start_positions[idx]
        visit = self._visit_count[idx]
        self._agent_pos = positions[visit % len(positions)]
        self._visit_count[idx] = visit + 1

        self._lives_max = LIVES_PER_LEVEL[idx] if idx < len(LIVES_PER_LEVEL) else 3
        self._lives = self._lives_max

        self._render()

    def _save_undo(self) -> None:
        self._undo_stack.append(
            {
                "dials": list(self._dials),
                "agent_pos": self._agent_pos,
                "budget": self._budget,
                "unlocked": self._unlocked,
                "lives": self._lives,
            }
        )
        if len(self._undo_stack) > 50:
            self._undo_stack.pop(0)

    _MOVE_MAP = {
        GameAction.ACTION1: ((0, -1), +1),
        GameAction.ACTION2: ((0, +1), -1),
        GameAction.ACTION3: ((-1, 0), +3),
        GameAction.ACTION4: ((+1, 0), -3),
    }

    def step(self) -> None:
        action = self.action

        if action.id == GameAction.RESET:
            pass
        elif action.id == GameAction.ACTION7:
            self._handle_undo()
        elif action.id in self._MOVE_MAP:
            self._handle_move(action.id)
        elif action.id == GameAction.ACTION5:
            self._handle_submit()

        self.complete_action()

    def _handle_undo(self) -> None:
        if self._undo_stack:
            current_budget = self._budget
            snap = self._undo_stack.pop()
            self._dials = snap["dials"]
            self._agent_pos = snap["agent_pos"]
            self._budget = current_budget
            self._unlocked = snap["unlocked"]
            self._lives = snap["lives"]
        self._spend_move()
        self._render()
        if self._budget <= 0:
            self._lose_life()

    def _handle_move(self, action_id: GameAction) -> None:
        self._save_undo()
        (dx, dy), rot_delta = self._MOVE_MAP[action_id]
        px, py = self._agent_pos

        dial_idx = _dial_to_rotate(px, py, dx, dy, self._dial_pos)

        if dial_idx >= 0:
            self._dials[dial_idx] = (self._dials[dial_idx] + rot_delta) % 10
        else:
            nx = max(MOVE_X0, min(MOVE_X1, px + dx))
            ny = max(MOVE_Y0, min(MOVE_Y1, py + dy))
            if not _is_blocked_cell(nx, ny, self._dial_pos, self._blocker_set):
                self._agent_pos = (nx, ny)

        self._spend_move()
        self._render()
        if self._budget <= 0:
            self._lose_life()

    def _handle_submit(self) -> None:
        self._save_undo()
        self._spend_move()

        self._unlocked = all(
            self._dials[i] == self._targets[i] for i in range(len(self._targets))
        )

        self._render()

        if self._unlocked:
            self.next_level()
            return

        if self._budget <= 0:
            self._lose_life()

    def _spend_move(self) -> None:
        self._budget = max(0, self._budget - 1)

    def _lose_life(self) -> bool:
        self._lives -= 1
        if self._lives <= 0:
            self._render()
            self.lose()
            return True
        self._dials = [0] * len(self._targets)
        idx = self._current_level_index
        positions = self._start_positions[idx]
        visit = self._visit_count[idx] - 1
        self._agent_pos = positions[visit % len(positions)]
        self._unlocked = False
        self._budget = self._budget_max
        self._render()
        return False

    def _render(self) -> None:
        self.current_level.remove_all_sprites()

        bg_pixels = _build_background(
            self._dials,
            self._targets,
            self._dial_pos,
            self._blockers,
            self._budget,
            self._budget_max,
            self._lives,
            self._lives_max,
        )

        bg_sprite = Sprite(
            pixels=bg_pixels.tolist(),
            name="background",
            visible=True,
            collidable=False,
            layer=0,
        )
        bg_sprite.set_position(0, 0)
        self.current_level.add_sprite(bg_sprite)

        ax, ay = self._agent_pos
        near = any(
            _dial_to_rotate(ax, ay, ddx, ddy, self._dial_pos) >= 0
            for ddx, ddy in [(0, -1), (0, 1), (-1, 0), (1, 0)]
        )
        agent_col = 11 if near else AGENT_COL

        agent_sp = Sprite(
            pixels=[[agent_col]],
            name="agent",
            visible=True,
            collidable=False,
            layer=2,
        )
        agent_sp.set_position(ax, ay)
        self.current_level.add_sprite(agent_sp)

    @property
    def extra_state(self) -> dict:
        idx = self._current_level_index
        n = len(self._targets)
        near = any(
            _dial_to_rotate(
                self._agent_pos[0], self._agent_pos[1], ddx, ddy, self._dial_pos
            )
            >= 0
            for ddx, ddy in [(0, -1), (0, 1), (-1, 0), (1, 0)]
        )
        matches = sum(1 for i in range(n) if self._dials[i] == self._targets[i])
        budget_str = str(self._budget)
        dials_str = "-".join(str(d) for d in self._dials)

        return {
            "remaining_cells": n - matches,
            "circuit_title": "Password Dial - Level %d/%d" % (idx + 1, N_LEVELS),
            "level_title": "Level %d" % (idx + 1),
            "dials": list(self._dials),
            "dial_count": n,
            "unlocked": self._unlocked,
            "agent_pos": list(self._agent_pos),
            "near_dial": near,
            "budget_remaining": self._budget,
            "budget_max": self._budget_max,
            "lives": self._lives,
            "lives_max": self._lives_max,
            "lives_lost": self._lives_max - self._lives,
            "num_blockers": len(self._blockers),
            "level_features": [
                "Level %d/%d" % (idx + 1, N_LEVELS),
                "Dials: %d" % n,
                "Current: %s" % dials_str,
                "Matched: %d/%d" % (matches, n),
                "Agent:   %s" % str(self._agent_pos),
                "Near dial: %s" % ("yes" if near else "no"),
                "Budget:  %s" % budget_str,
                "Lives:   %d/%d" % (self._lives, self._lives_max),
                "Blockers: %d" % len(self._blockers),
                "Status:  %s" % ("UNLOCKED" if self._unlocked else "LOCKED"),
            ],
        }


ACTION_LIST: list[str] = ["reset", "up", "down", "left", "right", "select", "undo"]


class PuzzleEnvironment:
    def __init__(self, seed: int = 0) -> None:
        self._engine: Pd01 = Pd01(seed=seed)
        self._turn: int = 0
        self._last_was_reset: bool = False

    def _frame_to_png(self, frame) -> bytes | None:
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
            return (
                struct.pack(">I", len(data))
                + chunk
                + struct.pack(">I", zlib.crc32(chunk) & 0xFFFFFFFF)
            )

        raw_rows = b"".join(b"\x00" + rgb[r].tobytes() for r in range(h))
        return (
            b"\x89PNG\r\n\x1a\n"
            + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + _png_chunk(b"IDAT", zlib.compress(raw_rows))
            + _png_chunk(b"IEND", b"")
        )

    def _build_text_observation(self) -> str:
        g = self._engine
        idx = g._current_level_index
        n = len(g._targets)
        matches = sum(1 for i in range(n) if g._dials[i] == g._targets[i])
        dials_str = "-".join(str(d) for d in g._dials)
        near = any(
            _dial_to_rotate(g._agent_pos[0], g._agent_pos[1], ddx, ddy, g._dial_pos)
            >= 0
            for ddx, ddy in [(0, -1), (0, 1), (-1, 0), (1, 0)]
        )
        targets_str = "-".join(str(t) for t in g._targets)
        blocker_str = ", ".join(str(b) for b in g._blockers) if g._blockers else "none"
        lines = [
            f"=== Password Dial (Level {idx + 1}/{N_LEVELS}) ===",
            f"Dials: {dials_str}",
            f"Targets: {targets_str}",
            f"Matched: {matches}/{n}",
            f"Agent: {g._agent_pos}",
            f"Near dial: {'yes' if near else 'no'}",
            f"Budget: {g._budget}/{g._budget_max}",
            f"Lives: {g._lives}/{g._lives_max}",
            f"Blockers: {blocker_str}",
            f"Dial positions: {g._dial_pos}",
            f"Rotation: up +1, down -1, left +3, right -3 (when adjacent to dial)",
        ]
        if g._state == EngineGameState.WIN:
            lines.append("State: won")
        elif g._state == EngineGameState.GAME_OVER:
            lines.append("State: game-over")
        else:
            lines.append("State: playing")
        return "\n".join(lines)

    def _build_metadata(self) -> dict:
        g = self._engine
        n = len(g._targets)
        return {
            "game_id": g._game_id,
            "level_index": g.level_index,
            "total_levels": N_LEVELS,
            "dials": list(g._dials),
            "dial_count": n,
            "budget_remaining": g._budget,
            "budget_max": g._budget_max,
            "lives": g._lives,
            "lives_max": g._lives_max,
            "agent_pos": list(g._agent_pos),
        }

    def _make_game_state(self, frame_data) -> GameState:
        image_bytes: bytes | None = None
        if frame_data and not frame_data.is_empty():
            image_bytes = self._frame_to_png(frame_data.frame)

        done = self._engine._state in (EngineGameState.WIN, EngineGameState.GAME_OVER)
        valid = ["reset"] if done else ACTION_LIST

        return GameState(
            text_observation=self._build_text_observation(),
            image_observation=image_bytes,
            valid_actions=valid,
            turn=self._turn,
            metadata=self._build_metadata(),
        )

    def _make_game_state_from_render(self) -> GameState:
        g = self._engine
        frame = g.camera.render(g.current_level.get_sprites())
        image_bytes: bytes | None = self._frame_to_png(frame)

        done = g._state in (EngineGameState.WIN, EngineGameState.GAME_OVER)
        valid = ["reset"] if done else ACTION_LIST

        return GameState(
            text_observation=self._build_text_observation(),
            image_observation=image_bytes,
            valid_actions=valid,
            turn=self._turn,
            metadata=self._build_metadata(),
        )

    def reset(self) -> GameState:
        g = self._engine
        if g._state in (EngineGameState.WIN, EngineGameState.GAME_OVER):
            self._turn = 0
            self._last_was_reset = False
            frame_data = g.perform_action(ActionInput(id=GameAction.RESET))
            return self._make_game_state(frame_data)

        if self._last_was_reset:
            self._turn = 0
            self._last_was_reset = False
            frame_data = g.perform_action(ActionInput(id=GameAction.RESET))
            return self._make_game_state(frame_data)

        self._last_was_reset = True
        g.on_set_level(g.current_level)
        return self._make_game_state_from_render()

    def get_actions(self) -> list[str]:
        done = self._engine._state in (EngineGameState.WIN, EngineGameState.GAME_OVER)
        if done:
            return ["reset"]
        return ACTION_LIST

    def is_done(self) -> bool:
        return self._engine._state in (EngineGameState.WIN, EngineGameState.GAME_OVER)

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError(
                f"Unsupported render mode '{mode}'. Supported: 'rgb_array'"
            )
        frame: np.ndarray = self._engine.camera.render(
            self._engine.current_level.get_sprites()
        )
        h, w = frame.shape[:2]
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for idx_val, color in _ARC_PALETTE.items():
            mask = frame == idx_val
            if mask.ndim == 3:
                mask = mask.all(axis=2)
            rgb[mask] = color
        if h != 64 or w != 64:
            row_idx = (np.arange(64) * h // 64).astype(int)
            col_idx = (np.arange(64) * w // 64).astype(int)
            rgb = rgb[np.ix_(row_idx, col_idx)]
        return rgb

    def close(self) -> None:
        self._engine = None

    def step(self, action: str) -> StepResult:
        action = action.strip().lower()
        game_action = _ACTION_FROM_NAME.get(action)
        if game_action is None:
            raise ValueError(f"Unknown action '{action}'. Valid: {ACTION_LIST}")

        if game_action == GameAction.RESET and self._last_was_reset:
            self._turn = 0
            self._last_was_reset = False
            frame_data = self._engine.perform_action(ActionInput(id=GameAction.RESET))
            return StepResult(
                state=self._make_game_state(frame_data),
                reward=0.0,
                done=False,
                info={
                    "action": action,
                    "engine_state": frame_data.state,
                    "level_changed": True,
                    "life_lost": False,
                    "full_reset": True,
                },
            )

        if game_action == GameAction.RESET:
            self._last_was_reset = True
            self._engine.on_set_level(self._engine.current_level)
            return StepResult(
                state=self._make_game_state_from_render(),
                reward=0.0,
                done=False,
                info={
                    "action": action,
                    "engine_state": self._engine._state,
                    "level_changed": False,
                    "life_lost": False,
                    "full_reset": False,
                },
            )

        prev_level = self._engine.level_index
        prev_lives = self._engine._lives

        action_input = ActionInput(id=game_action)
        frame_data = self._engine.perform_action(action_input)

        self._last_was_reset = False
        self._turn += 1

        engine_state = frame_data.state
        done = engine_state in (EngineGameState.WIN, EngineGameState.GAME_OVER)

        total_levels = len(self._engine._levels)
        level_reward_step = 1.0 / total_levels

        reward = 0.0
        if engine_state == EngineGameState.WIN:
            reward = level_reward_step
        elif self._engine.level_index > prev_level:
            reward = level_reward_step

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
        self._last_level: int = 0

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
        self._last_level = 0

        return self._get_obs(), self._build_info(state)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action_str: str = self._action_to_string[int(action)]
        result: StepResult = self._env.step(action_str)

        obs = self._get_obs()
        terminated: bool = result.done
        truncated: bool = False
        info = self._build_info(result.state, result.info)

        reward: float = result.reward
        current_level = result.state.metadata.get("level_index", self._last_level)
        self._last_level = current_level

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
