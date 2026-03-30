from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from arcengine import (
    ActionInput,
    ARCBaseGame,
    BlockingMode,
    Camera,
    GameAction,
    GameState as EngineGameState,
    Level,
    Sprite,
)

FLOOR = 0
WALL = 1
BLOCK = 2
ACTIVE = 3
TARGET = 4
LOCKED = 13
BORDER = 6
CRACKED = 7
HOLE = 5

BAR_FILLED = 9
BAR_EMPTY = 10
BAR_BG = 11

STEP_BUDGET = {0: 10, 1: 9, 2: 9, 3: 14, 4: 16}
STEP_MULTIPLIER = 1.6
BAR_HEIGHT = 3
NUM_LEVELS = 5
MAX_LIVES = 3

LIFE_FULL = 8
LIFE_EMPTY = 10


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


def _parse_grid(lines):
    char_map = {
        ".": FLOOR,
        "W": WALL,
        "B": BLOCK,
        "T": TARGET,
        "#": BORDER,
        "C": CRACKED,
        "H": HOLE,
    }
    grid = []
    blocks = []
    targets = []
    for y, line in enumerate(lines):
        row = []
        for x, ch in enumerate(line):
            if ch == "B":
                row.append(BLOCK)
                blocks.append((x, y))
            elif ch == "T":
                row.append(TARGET)
                targets.append((x, y))
            else:
                row.append(char_map.get(ch, FLOOR))
        grid.append(row)
    return grid, blocks, targets


_L1 = [
    "##############",
    "#............#",
    "#B...W..W...B#",
    "#.WW......WW.#",
    "#............#",
    "#............#",
    "#.T..WB.W....#",
    "#...W....W..T#",
    "#...W...W....#",
    "#....WW.W....#",
    "#.......W....#",
    "#......W.....#",
    "#T...........#",
    "##############",
]
_L2 = [
    "##############",
    "#W...........#",
    "#....W..WB...#",
    "#.WW.........#",
    "#B.......WWW.#",
    "#............#",
    "#.T..WB.W....#",
    "#...W....W..T#",
    "#...W...W....#",
    "#....WW.W....#",
    "#.......W....#",
    "#......W.....#",
    "#T...........#",
    "##############",
]
_L3 = [
    "##############",
    "#............#",
    "#B...W..W....#",
    "#....W..W.WW.#",
    "#.......W..B.#",
    "#.......W....#",
    "#.......W..T.#",
    "#...W....B...#",
    "#.....W......#",
    "#.....W.W....#",
    "#......TW....#",
    "#......W.....#",
    "#.....T......#",
    "##############",
]


_L4 = [
    "######################",
    "#B.....W...........C.#",
    "#...........W..B.....#",
    "#..WWW......W.WW.....#",
    "#......W....W........#",
    "#......W....W.....H..#",
    "#.T....W....W..T.....#",
    "#.....W......W.......#",
    "#....................#",
    "#....H...............#",
    "#....................#",
    "#..B....WW....B......#",
    "#..WW...WW...WW...C..#",
    "#......T.............#",
    "#.....W....W.........#",
    "#..........C.........#",
    "#....................#",
    "#............C.......#",
    "#.................H..#",
    "#....C...............#",
    "#........T...........#",
    "######################",
]
_L5 = [
    "######################",
    "#....................#",
    "#..B....W......WB....#",
    "#..WWWW.W......W.....#",
    "#.......W......W.....#",
    "#.......W......W...T.#",
    "#..T....W.......W....#",
    "#......W.............#",
    "#....................#",
    "#....................#",
    "#.B..........W....B..#",
    "#..WW........W....WW.#",
    "#............WT......#",
    "#.......W............#",
    "#.W..................#",
    "#....................#",
    "#....................#",
    "#..B.....W....H......#",
    "#..WW....W.......B...#",
    "#.......TW...........#",
    "#....W.......W.......#",
    "#..T.........W.......#",
    "######################",
]


def _build_level(level_index):
    templates = [_L1, _L2, _L3, _L4, _L5]
    grid, blocks, targets = _parse_grid(templates[level_index])
    puzzle_h, w = len(grid), len(grid[0])
    budget = STEP_BUDGET.get(level_index, 1)
    bar_len = min(int(budget * STEP_MULTIPLIER), w - 2)
    grid.append([BAR_BG] * w)
    pad_left = (w - bar_len) // 2
    pad_right = w - bar_len - pad_left
    grid.append([BAR_BG] * pad_left + [BAR_EMPTY] * bar_len + [BAR_BG] * pad_right)
    grid.append([BAR_BG] * w)
    return {
        "grid": grid,
        "w": w,
        "h": puzzle_h + BAR_HEIGHT,
        "puzzle_h": puzzle_h,
        "blocks": blocks,
        "targets": targets,
        "max_steps": bar_len,
        "bar_row_y": puzzle_h + 1,
        "bar_start_x": pad_left,
        "lives_row_y": puzzle_h + 2,
    }


class Rs01(ARCBaseGame):
    def __init__(self, seed: int = 0):
        self._grid, self._blocks, self._locked = None, [], set()
        self._steps_taken, self._active_block_pos = 0, None
        self._lives = MAX_LIVES
        self._lives_row_y = 0
        self._mid_level_reset = False
        self._level_idx = 0
        self._history = []
        self._retry_level_on_reset = False
        levels = [
            Level(
                name=f"Level {i + 1}",
                grid_size=(_build_level(i)["w"], _build_level(i)["h"]),
                data={"idx": i},
            )
            for i in range(NUM_LEVELS)
        ]
        super().__init__(
            game_id="rs01",
            levels=levels,
            camera=Camera(width=64, height=64, background=5),
            win_score=NUM_LEVELS,
            available_actions=[0, 1, 2, 3, 4, 5, 7],
            seed=seed,
        )

    def handle_reset(self) -> None:
        self._lives = MAX_LIVES
        if self._retry_level_on_reset:
            self._retry_level_on_reset = False
            self.on_set_level(self.current_level)
        else:
            super().handle_reset()

    def on_set_level(self, level: Level) -> None:
        self._history = []
        self._level_idx = level.get_data("idx")
        data = _build_level(self._level_idx)
        self._grid = [row[:] for row in data["grid"]]
        self._gw, self._gh, self._puzzle_h = data["w"], data["h"], data["puzzle_h"]
        self._blocks, self._targets, self._locked = (
            list(data["blocks"]),
            list(data["targets"]),
            set(),
        )
        self._steps_taken, self._max_steps = 0, data["max_steps"]
        self._bar_row_y, self._bar_start_x, self._bar_len = (
            data["bar_row_y"],
            data["bar_start_x"],
            data["max_steps"],
        )
        self._lives_row_y = data["lives_row_y"]
        if not self._mid_level_reset:
            self._lives = MAX_LIVES
        self._mid_level_reset = False
        self._active_block_pos = self._blocks[0] if self._blocks else None
        self._refresh_board()

    def _save_state(self):
        self._history.append({
            "grid": [row[:] for row in self._grid],
            "blocks": list(self._blocks),
            "locked": set(self._locked),
            "steps_taken": self._steps_taken,
            "active_block_pos": self._active_block_pos,
        })

    def _restore_state(self):
        if not self._history:
            return
        state = self._history.pop()
        self._grid = state["grid"]
        self._blocks = state["blocks"]
        self._locked = state["locked"]
        self._steps_taken = state["steps_taken"]
        self._active_block_pos = state["active_block_pos"]
        self._refresh_board()

    def step(self) -> None:
        act = self.action
        dirs = {
            GameAction.ACTION1: (0, -1),
            GameAction.ACTION2: (0, 1),
            GameAction.ACTION3: (-1, 0),
            GameAction.ACTION4: (1, 0),
        }
        if act.id == GameAction.ACTION7:
            if self._history:
                self._restore_state()
                self._steps_taken += 1
        elif act.id in dirs:
            self._handle_move(*dirs[act.id])
        elif act.id == GameAction.ACTION5:
            self._cycle_block()
        elif act.id == GameAction.ACTION6:
            self._handle_click(act.data.get("x", 0), act.data.get("y", 0))

        self._refresh_board()
        if all(t in self._locked for t in self._targets):
            self.next_level()
        elif self._steps_taken >= self._max_steps:
            self._lives -= 1
            if self._lives <= 0:
                self.lose()
            else:
                self._mid_level_reset = True
                self.on_set_level(self.current_level)
        self.complete_action()

    def _handle_move(self, dx, dy):
        if not self._active_block_pos:
            return
        bx, by = self._active_block_pos
        nx, ny = bx + dx, by + dy
        if (
            0 <= nx < self._gw
            and 0 <= ny < self._puzzle_h
            and self._grid[ny][nx] in (FLOOR, TARGET, CRACKED)
        ):
            self._save_state()
            self._slide(bx, by, dx, dy)
            self._steps_taken += 1

    def _handle_click(self, cx, cy):
        gx, gy = self._canvas_to_grid(cx, cy)
        if (
            0 <= gx < self._gw
            and 0 <= gy < self._puzzle_h
            and self._grid[gy][gx] in (FLOOR, TARGET, CRACKED)
        ):
            for bx, by, dx, dy in [
                (gx + 1, gy, 1, 0),
                (gx - 1, gy, -1, 0),
                (gx, gy + 1, 0, 1),
                (gx, gy - 1, 0, -1),
            ]:
                if (
                    0 <= bx < self._gw
                    and 0 <= by < self._gh
                    and self._grid[by][bx] in (BLOCK, ACTIVE)
                ):
                    self._save_state()
                    self._slide(bx, by, dx, dy)
                    self._steps_taken += 1
                    break

    def _slide(self, bx, by, dx, dy):
        orig_active = self._active_block_pos == (bx, by)
        self._grid[by][bx] = TARGET if (bx, by) in self._targets else FLOOR
        self._blocks.remove((bx, by))
        cx, cy = bx, by
        while True:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < self._gw and 0 <= ny < self._puzzle_h):
                break
            cell = self._grid[ny][nx]
            if cell in (WALL, BORDER, BLOCK, LOCKED):
                break

            if cell == CRACKED:
                self._grid[ny][nx] = HOLE
            elif cell == HOLE:
                self._grid[ny][nx] = FLOOR
                if orig_active:
                    self._cycle_block()
                return

            cx, cy = nx, ny
            if (cx, cy) in self._targets and (cx, cy) not in self._locked:
                break

        self._blocks.append((cx, cy))
        if (cx, cy) in self._targets:
            self._grid[cy][cx] = LOCKED
            self._locked.add((cx, cy))
            if orig_active:
                self._cycle_block()
        else:
            self._grid[cy][cx] = BLOCK
            if orig_active:
                self._active_block_pos = (cx, cy)

    def _cycle_block(self):
        m = [b for b in self._blocks if b not in self._locked]
        if not m:
            self._active_block_pos = None
        else:
            idx = m.index(self._active_block_pos) if self._active_block_pos in m else -1
            self._active_block_pos = m[(idx + 1) % len(m)]

    def _refresh_board(self):
        for bx, by in self._blocks:
            if (bx, by) not in self._locked:
                self._grid[by][bx] = (
                    ACTIVE if (bx, by) == self._active_block_pos else BLOCK
                )
        y, fill = self._bar_row_y, min(self._steps_taken, self._bar_len)
        for i in range(self._bar_len):
            self._grid[y][self._bar_start_x + i] = BAR_FILLED if i < fill else BAR_EMPTY

        ly = self._lives_row_y
        pip_start = (self._gw - MAX_LIVES * 2 + 1) // 2
        for x in range(self._gw):
            self._grid[ly][x] = BAR_BG
        for i in range(MAX_LIVES):
            px = pip_start + i * 2
            if 0 <= px < self._gw:
                self._grid[ly][px] = LIFE_FULL if i < self._lives else LIFE_EMPTY

        self.current_level.remove_all_sprites()
        self.current_level.add_sprite(
            Sprite(pixels=[r[:] for r in self._grid], name="b", x=0, y=0, layer=0)
        )

    def _canvas_to_grid(self, cx, cy):
        s = min(64 // self._gw, 64 // self._gh) or 1
        return (cx - (64 - self._gw * s) // 2) // s, (
            cy - (64 - self._gh * s) // 2
        ) // s


_ACTION_MAP = {
    "up": GameAction.ACTION1,
    "down": GameAction.ACTION2,
    "left": GameAction.ACTION3,
    "right": GameAction.ACTION4,
    "select": GameAction.ACTION5,
    "undo": GameAction.ACTION7,
}
_VALID_ACTIONS = ["reset", "up", "down", "left", "right", "select", "undo"]


class PuzzleEnvironment:
    def __init__(self, seed: int = 0):
        self._engine = Rs01(seed=seed)
        self._total_turns = 0
        self._consecutive_resets = 0
        self._last_frame_data = None

    def reset(self) -> GameState:
        self._total_turns = 0
        self._consecutive_resets = 0
        self._last_frame_data = self._engine.perform_action(ActionInput(id=GameAction.RESET))
        return GameState(
            text_observation=self._build_text_observation(),
            image_observation=None,
            valid_actions=self.get_actions(),
            turn=self._total_turns,
            metadata=self._build_metadata(),
        )

    def step(self, action: str) -> StepResult:
        if action == "reset":
            self._consecutive_resets += 1
            if self._consecutive_resets >= 2:
                self._consecutive_resets = 0
                self._engine._retry_level_on_reset = False
            else:
                self._engine._retry_level_on_reset = True
            self._last_frame_data = self._engine.perform_action(ActionInput(id=GameAction.RESET))
            self._total_turns = 0
            state = GameState(
                text_observation=self._build_text_observation(),
                image_observation=None,
                valid_actions=self.get_actions(),
                turn=self._total_turns,
                metadata=self._build_metadata(),
            )
            return StepResult(state=state, reward=0.0, done=False)
        self._consecutive_resets = 0
        ga = _ACTION_MAP.get(action)
        if ga is None:
            state = GameState(
                text_observation=self._build_text_observation(),
                image_observation=None,
                valid_actions=self.get_actions(),
                turn=self._total_turns,
                metadata=self._build_metadata(),
            )
            return StepResult(state=state, reward=0.0, done=False)
        prev_level = self._engine._level_idx
        self._last_frame_data = self._engine.perform_action(ActionInput(id=ga))
        self._total_turns += 1
        reward = 0.0
        eng_state = self._last_frame_data.state
        if eng_state == EngineGameState.WIN:
            reward = 1.0 / len(self._engine._levels)
        elif self._engine._level_idx != prev_level and eng_state != EngineGameState.GAME_OVER:
            reward = 1.0 / len(self._engine._levels)
        done = eng_state in (EngineGameState.WIN, EngineGameState.GAME_OVER)
        state = GameState(
            text_observation=self._build_text_observation(),
            image_observation=None,
            valid_actions=self.get_actions(),
            turn=self._total_turns,
            metadata=self._build_metadata(),
        )
        return StepResult(state=state, reward=reward, done=done)

    def get_actions(self) -> List[str]:
        if self.is_done():
            return ["reset"]
        return list(_VALID_ACTIONS)

    def is_done(self) -> bool:
        if self._last_frame_data is None:
            return False
        return self._last_frame_data.state in (EngineGameState.WIN, EngineGameState.GAME_OVER)

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError(f"Unsupported render mode: {mode}")
        if self._last_frame_data is None:
            return np.zeros((64, 64, 3), dtype=np.uint8)
        return np.array(self._last_frame_data.frame, dtype=np.uint8)

    def close(self) -> None:
        self._engine = None

    def _build_text_observation(self):
        g = self._engine
        color_map = {0: '.', 1: 'W', 2: 'B', 3: '*', 4: 'T', 13: 'L', 6: '#', 7: 'C', 5: 'H'}
        lines = []
        lines.append(f"Level:{g._level_idx+1}/5 Steps:{g._steps_taken}/{g._max_steps} Lives:{g._lives}/{MAX_LIVES}")
        for y in range(g._puzzle_h):
            row = ""
            for x in range(g._gw):
                row += color_map.get(g._grid[y][x], '?')
            lines.append(row)
        return "\n".join(lines)

    def _build_metadata(self):
        g = self._engine
        return {
            "total_levels": len(self._engine._levels),
            "level": g._level_idx + 1,
            "steps_used": g._steps_taken,
            "step_limit": g._max_steps,
            "lives": g._lives,
            "max_lives": MAX_LIVES,
            "blocks_total": len(g._targets),
            "blocks_locked": len(g._locked),
            "levels_completed": getattr(g, "_score", 0),
            "level_index": g._level_idx,
            "game_over": getattr(getattr(g, "_state", None), "name", "") == "GAME_OVER",
        }


class ArcGameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 5}
    ACTION_LIST = ["reset", "up", "down", "left", "right", "select", "undo"]
    OBS_HEIGHT = 64
    OBS_WIDTH = 64

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
