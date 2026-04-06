import random
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


ARC_PALETTE = [
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
]

C_BG = 5
C_EMPTY_A = 1
C_EMPTY_B = 2
C_GRID = 3
C_HUB = 10
C_BLOCK = 4
C_SMALL = 14
C_MEDIUM = 12
C_LARGE = 8
C_CURSOR = 0
C_SELECTED = 11
C_LIFE = 7
C_STEP = 9
C_STEP_WARN = 12
C_STEP_CRIT = 8

BACKGROUND_COLOR = 5
PADDING_COLOR = 3
CAM_SIZE = 16

EMPTY = 0
SMALL = 1
MEDIUM = 2
LARGE = 3
BLOCK = -1
HUB = 9

_LEVELS = [
    {
        "n": 5,
        "max_steps": 90,
        "grid": [
            ".....",
            ".M.L.",
            "..H..",
            ".S.M.",
            ".....",
        ],
    },
    {
        "n": 5,
        "max_steps": 120,
        "grid": [
            ".S.M.",
            "M.L.L",
            "..H..",
            "M.L.S",
            ".S.M.",
        ],
    },
    {
        "n": 5,
        "max_steps": 130,
        "grid": [
            "S.#M.",
            "M.#..",
            ".LHL.",
            "..#.M",
            ".M#.S",
        ],
    },
    {
        "n": 7,
        "max_steps": 170,
        "grid": [
            "S....M.",
            ".M.....",
            "..LL..M",
            "..LHL..",
            "M..L...",
            ".....M.",
            "......S",
        ],
    },
]


def _px(color: int, layer: int = 0, name: str = "px") -> Sprite:
    return Sprite(
        pixels=np.array([[color]], dtype=np.int32),
        name=name,
        visible=True,
        collidable=False,
        tags=[],
        layer=layer,
    )


class Db07(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)
        self._history: List[Dict] = []
        game_levels = [Level(sprites=[], grid_size=(CAM_SIZE, CAM_SIZE), data=d) for d in _LEVELS]
        self._lives = 3
        camera = Camera(0, 0, CAM_SIZE, CAM_SIZE, BACKGROUND_COLOR, PADDING_COLOR, [])
        super().__init__("db07", game_levels, camera, available_actions=[0, 1, 2, 3, 4, 5, 7])

    def _parse_grid(self, rows: List[str]) -> np.ndarray:
        n = len(rows)
        arr = np.zeros((n, n), dtype=np.int32)
        for y, row in enumerate(rows):
            for x, ch in enumerate(row):
                if ch == ".":
                    arr[y, x] = EMPTY
                elif ch == "S":
                    arr[y, x] = SMALL
                elif ch == "M":
                    arr[y, x] = MEDIUM
                elif ch == "L":
                    arr[y, x] = LARGE
                elif ch == "#":
                    arr[y, x] = BLOCK
                elif ch == "H":
                    arr[y, x] = HUB
                else:
                    arr[y, x] = EMPTY
        return arr

    def on_set_level(self, level: Level) -> None:
        self._history = []
        idx = min(self._current_level_index, len(_LEVELS) - 1)
        data = _LEVELS[idx]
        self._n = data["n"]
        self._max_steps = data["max_steps"]
        self._steps_left = self._max_steps
        self._offset_x = (CAM_SIZE - self._n) // 2
        self._offset_y = (CAM_SIZE - self._n) // 2
        self._selected: Optional[Tuple[int, int]] = None
        self._lives = 3

        self._initial_board = self._parse_grid(data["grid"])
        self._board = self._initial_board.copy()
        self._center = (self._n // 2, self._n // 2)

        self._build_board_sprites()
        self._build_hud_sprites()

        empty_cells = []
        for y in range(self._n):
            for x in range(self._n):
                if self._board[y, x] == EMPTY:
                    empty_cells.append((x, y))
        if empty_cells:
            self._cursor_x, self._cursor_y = self._rng.choice(empty_cells)
        else:
            self._cursor_x, self._cursor_y = self._center
        self._cursor = _px(C_CURSOR, layer=20, name="cursor")
        self._place_cursor()
        self.current_level.add_sprite(self._cursor)

        self._draw_hud()
        self._render_board()

    def _build_board_sprites(self) -> None:
        self._cell_sprites: Dict[Tuple[int, int], Sprite] = {}
        for y in range(self._n):
            for x in range(self._n):
                base = _px(C_GRID, layer=0, name="cell")
                base.set_position(self._offset_x + x, self._offset_y + y)
                self.current_level.add_sprite(base)
                self._cell_sprites[(x, y)] = base

        for x in range(-1, self._n + 1):
            top = _px(C_GRID, layer=1, name="outline")
            bot = _px(C_GRID, layer=1, name="outline")
            top.set_position(self._offset_x + x, self._offset_y - 1)
            bot.set_position(self._offset_x + x, self._offset_y + self._n)
            self.current_level.add_sprite(top)
            self.current_level.add_sprite(bot)
        for y in range(self._n):
            left = _px(C_GRID, layer=1, name="outline")
            right = _px(C_GRID, layer=1, name="outline")
            left.set_position(self._offset_x - 1, self._offset_y + y)
            right.set_position(self._offset_x + self._n, self._offset_y + y)
            self.current_level.add_sprite(left)
            self.current_level.add_sprite(right)

    def _build_hud_sprites(self) -> None:
        self._step_sprites: List[Sprite] = []
        self._life_sprites: List[Sprite] = []
        for i in range(10):
            sp = _px(C_GRID, layer=6, name="step")
            sp.set_position(i, 0)
            self.current_level.add_sprite(sp)
            self._step_sprites.append(sp)
        for i in range(3):
            sp = _px(C_GRID, layer=6, name="life")
            sp.set_position(12 + i, 0)
            self.current_level.add_sprite(sp)
            self._life_sprites.append(sp)

    def _cell_color(self, value: int, selected: bool, x: int, y: int) -> int:
        if selected:
            return C_SELECTED
        if value == EMPTY:
            return C_EMPTY_A if (x + y) % 2 == 0 else C_EMPTY_B
        if value == BLOCK:
            return C_BLOCK
        if value == HUB:
            return C_HUB
        if value == SMALL:
            return C_SMALL
        if value == MEDIUM:
            return C_MEDIUM
        if value == LARGE:
            return C_LARGE
        return C_EMPTY_A

    def _render_board(self) -> None:
        for y in range(self._n):
            for x in range(self._n):
                selected = self._selected == (x, y)
                color = self._cell_color(self._board[y, x], selected, x, y)
                self._cell_sprites[(x, y)].pixels = np.array([[color]], dtype=np.int32)
        self._draw_hud()

    def _place_cursor(self) -> None:
        self._cursor.set_position(self._offset_x + self._cursor_x, self._offset_y + self._cursor_y)

    def _draw_hud(self) -> None:
        frac = self._steps_left / self._max_steps if self._max_steps > 0 else 0
        step_color = C_STEP if frac >= 0.5 else (C_STEP_WARN if frac >= 0.25 else C_STEP_CRIT)
        filled = int(10 * frac)
        for i in range(10):
            color = step_color if i < filled else C_GRID
            self._step_sprites[i].pixels = np.array([[color]], dtype=np.int32)

        for i in range(3):
            color = C_LIFE if i < self._lives else C_GRID
            self._life_sprites[i].pixels = np.array([[color]], dtype=np.int32)

    def _distance(self, x: int, y: int) -> int:
        cx, cy = self._center
        return abs(x - cx) + abs(y - cy)

    def _ship_distances(self) -> Dict[int, List[int]]:
        out: Dict[int, List[int]] = {SMALL: [], MEDIUM: [], LARGE: []}
        for y in range(self._n):
            for x in range(self._n):
                v = self._board[y, x]
                if v in out:
                    out[v].append(self._distance(x, y))
        return out

    def _is_priority_satisfied(self) -> bool:
        dists = self._ship_distances()
        ordered = [SMALL, MEDIUM, LARGE]
        for i in range(len(ordered) - 1):
            a = ordered[i]
            b = ordered[i + 1]
            if not dists[a] or not dists[b]:
                continue
            if max(dists[a]) >= min(dists[b]):
                return False
        return True

    def _reset_level_state(self) -> None:
        self._board = self._initial_board.copy()
        self._steps_left = self._max_steps
        self._selected = None
        self._history.clear()
        empty_cells = []
        for y in range(self._n):
            for x in range(self._n):
                if self._board[y, x] == EMPTY:
                    empty_cells.append((x, y))
        if empty_cells:
            self._cursor_x, self._cursor_y = self._rng.choice(empty_cells)
        else:
            self._cursor_x, self._cursor_y = self._center
        self._place_cursor()
        self._render_board()

    def _trigger_life_loss(self) -> bool:
        self._lives -= 1
        if self._lives <= 0:
            self.lose()
            self.complete_action()
            return True
        self._reset_level_state()
        return False

    def _valid_target(self, x: int, y: int) -> bool:
        v = self._board[y, x]
        return v != BLOCK and v != HUB

    def _save_undo(self) -> None:
        self._history.append({
            "board": self._board.copy(),
            "cursor_x": self._cursor_x,
            "cursor_y": self._cursor_y,
            "selected": self._selected,
        })

    def _do_undo(self) -> None:
        if not self._history:
            return
        snap = self._history.pop()
        self._board = snap["board"]
        self._cursor_x = snap["cursor_x"]
        self._cursor_y = snap["cursor_y"]
        self._selected = snap["selected"]
        self._place_cursor()
        self._render_board()

    def _spend_step(self) -> bool:
        self._steps_left -= 1
        self._draw_hud()
        if self._steps_left <= 0:
            return self._trigger_life_loss()
        return False

    def _interact(self) -> None:
        x, y = self._cursor_x, self._cursor_y
        cell = self._board[y, x]

        if self._selected is None:
            if cell in (SMALL, MEDIUM, LARGE):
                self._selected = (x, y)
                self._render_board()
            return

        sx, sy = self._selected
        if not self._valid_target(x, y):
            self._selected = None
            self._render_board()
            return

        if (sx, sy) != (x, y):
            self._board[sy, sx], self._board[y, x] = self._board[y, x], self._board[sy, sx]

        self._selected = None
        self._render_board()

        if self._is_priority_satisfied():
            self.next_level()

    def step(self) -> None:
        act = self.action.id

        if act == GameAction.RESET:
            self.complete_action()
            return

        if act == GameAction.ACTION7:
            if self._spend_step():
                self.complete_action()
                return
            self._do_undo()
            self.complete_action()
            return

        if act == GameAction.ACTION5:
            self._save_undo()
            self._interact()
            if self._spend_step():
                self.complete_action()
                return
            self.complete_action()
            return

        if act == GameAction.ACTION1:
            if self._cursor_y > 0:
                self._save_undo()
                self._cursor_y -= 1
                self._place_cursor()
                if self._spend_step():
                    self.complete_action()
                    return
            self.complete_action()
            return

        if act == GameAction.ACTION2:
            if self._cursor_y < self._n - 1:
                self._save_undo()
                self._cursor_y += 1
                self._place_cursor()
                if self._spend_step():
                    self.complete_action()
                    return
            self.complete_action()
            return

        if act == GameAction.ACTION3:
            if self._cursor_x > 0:
                self._save_undo()
                self._cursor_x -= 1
                self._place_cursor()
                if self._spend_step():
                    self.complete_action()
                    return
            self.complete_action()
            return

        if act == GameAction.ACTION4:
            if self._cursor_x < self._n - 1:
                self._save_undo()
                self._cursor_x += 1
                self._place_cursor()
                if self._spend_step():
                    self.complete_action()
                    return
            self.complete_action()
            return

        self.complete_action()


class PuzzleEnvironment:
    _ACTION_MAP: Dict[str, GameAction] = {
        "reset": GameAction.RESET,
        "up": GameAction.ACTION1,
        "down": GameAction.ACTION2,
        "left": GameAction.ACTION3,
        "right": GameAction.ACTION4,
        "select": GameAction.ACTION5,
        "undo": GameAction.ACTION7,
    }
    _VALID_ACTIONS = list(_ACTION_MAP.keys())

    def __init__(self, seed: int = 0) -> None:
        self._engine = Db07(seed=seed)
        self._total_turns = 0
        self._done = False
        self._game_won = False
        self._last_action_was_reset = False

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
        return self._build_game_state()

    def get_actions(self) -> List[str]:
        if self._done:
            return ["reset"]
        return list(self._VALID_ACTIONS)

    def is_done(self) -> bool:
        return self._done

    def step(self, action: str) -> StepResult:
        e = self._engine
        if action == "reset":
            state = self.reset()
            return StepResult(state=state, reward=0.0, done=False, info={"action": "reset"})
        if action not in self._ACTION_MAP:
            raise ValueError(f"Invalid action '{action}'. Must be one of {list(self._ACTION_MAP.keys())}")
        self._last_action_was_reset = False
        self._total_turns += 1
        game_action = self._ACTION_MAP[action]
        info: Dict[str, Any] = {"action": action}
        level_before = e.level_index
        frame = e.perform_action(ActionInput(id=game_action), raw=True)
        state_name = frame.state.name if frame and frame.state else ""
        game_won = state_name == "WIN"
        game_over = state_name == "GAME_OVER"
        total_levels = len(self._engine._levels)
        level_reward = 1.0 / total_levels
        if game_won:
            self._done = True
            self._game_won = True
            info["reason"] = "game_complete"
            return StepResult(state=self._build_game_state(done=True), reward=level_reward, done=True, info=info)
        if game_over:
            self._done = True
            info["reason"] = "game_over"
            return StepResult(state=self._build_game_state(done=True), reward=0.0, done=True, info=info)
        reward = 0.0
        if e.level_index != level_before:
            reward = level_reward
            info["reason"] = "level_complete"
        return StepResult(state=self._build_game_state(done=False), reward=reward, done=False, info=info)

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError(f"Unsupported render mode: {mode}")
        e = self._engine
        index_grid = e.camera.render(e.current_level.get_sprites())
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

    def _build_text_observation(self) -> str:
        e = self._engine
        n = e._n
        board = e._board
        lines: List[str] = []
        lines.append(f"Level {e._current_level_index + 1} | Steps {e._steps_left}/{e._max_steps} | Lives {e._lives}")
        char_map = {0: ".", 1: "S", 2: "M", 3: "L", -1: "#", 9: "H"}
        for y in range(n):
            row = ""
            for x in range(n):
                val = int(board[y, x])
                ch = char_map.get(val, "?")
                if e._selected == (x, y):
                    ch = "*"
                if (x, y) == (e._cursor_x, e._cursor_y):
                    ch = f"[{ch}]"
                else:
                    ch = f" {ch} "
                row += ch
            lines.append(row)
        return "\n".join(lines)

    def _build_game_state(self, done: bool = False) -> GameState:
        valid_actions = self.get_actions() if not done else None
        return GameState(
            text_observation=self._build_text_observation(),
            image_observation=None,
            valid_actions=valid_actions,
            turn=self._total_turns,
            metadata={
                "total_levels": len(self._engine._levels),
                "level_index": self._engine.level_index,
                "levels_completed": getattr(self._engine, "_score", 0),
                "game_over": getattr(getattr(self._engine, "_state", None), "name", "") == "GAME_OVER",
                "done": done,
                "info": {},
            },
        )


class ArcGameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"]}

    ACTION_LIST = ["reset", "up", "down", "left", "right", "select", "undo"]
    OBS_HEIGHT = 64
    OBS_WIDTH = 64

    def __init__(self, seed: int = 0, render_mode: Optional[str] = None) -> None:
        super().__init__()
        self._seed = seed
        self.render_mode = render_mode
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.OBS_HEIGHT, self.OBS_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = spaces.Discrete(len(self.ACTION_LIST))
        self._action_to_string: Dict[int, str] = {i: a for i, a in enumerate(self.ACTION_LIST)}
        self._string_to_action: Dict[str, int] = {a: i for i, a in enumerate(self.ACTION_LIST)}
        self._env: Optional[PuzzleEnvironment] = None
        self._last_obs: Optional[np.ndarray] = None

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
        obs = self._get_obs()
        self._last_obs = obs
        return obs, self._build_info(state)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self._env is None:
            raise RuntimeError("Call reset() before step()")
        action_str = self._action_to_string[int(action)]
        result = self._env.step(action_str)
        obs = self._get_obs()
        self._last_obs = obs
        reward = result.reward
        terminated = result.done
        truncated: bool = False
        info = self._build_info(result.state, result.info)
        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        if self._env is None:
            return None
        if self.render_mode == "rgb_array":
            return self._get_obs()
        return None

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
        self._env = None

    def _get_obs(self) -> np.ndarray:
        if self._env is None:
            return np.zeros((self.OBS_HEIGHT, self.OBS_WIDTH, 3), dtype=np.uint8)
        rgb = self._env.render(mode="rgb_array")
        if rgb.shape[0] != self.OBS_HEIGHT or rgb.shape[1] != self.OBS_WIDTH:
            rgb = self._resize_nearest(rgb, self.OBS_HEIGHT, self.OBS_WIDTH)
        return rgb

    @staticmethod
    def _resize_nearest(img: np.ndarray, h: int, w: int) -> np.ndarray:
        src_h, src_w = img.shape[:2]
        ys = (np.arange(h) * src_h / h).astype(int)
        xs = (np.arange(w) * src_w / w).astype(int)
        ys = np.clip(ys, 0, src_h - 1)
        xs = np.clip(xs, 0, src_w - 1)
        return img[np.ix_(ys, xs)]

    @staticmethod
    def _build_info(state: GameState, extra: Optional[Dict] = None) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "text_observation": state.text_observation,
            "valid_actions": state.valid_actions,
            "turn": state.turn,
        }
        if state.metadata:
            info["metadata"] = state.metadata
        if extra:
            info.update(extra)
        return info

    def action_mask(self) -> np.ndarray:
        mask = np.zeros(len(self.ACTION_LIST), dtype=np.int8)
        if self._env is None:
            mask[0] = 1
            return mask
        valid = self._env.get_actions()
        for a in valid:
            idx = self._string_to_action.get(a)
            if idx is not None:
                mask[idx] = 1
        return mask


if __name__ == "__main__":
    from gymnasium.utils.env_checker import check_env
    env = ArcGameEnv(seed=0, render_mode="rgb_array")
    check_env(env.unwrapped, skip_render_check=True)
    obs, info = env.reset()
    mask = env.action_mask()
    valid_indices = np.where(mask == 1)[0]
    if len(valid_indices) > 0:
        obs, reward, term, trunc, info = env.step(valid_indices[0])
    env.close()
