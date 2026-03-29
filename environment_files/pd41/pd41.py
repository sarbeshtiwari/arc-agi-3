import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from arcengine import (
    ARCBaseGame,
    GameState as EngineGameState,
    GameAction,
    ActionInput,
    Level,
    Sprite,
    Camera,
    BlockingMode,
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


FLOOR = 0
WALL = 5
BLOCK = 8
ACTIVE = 10
TARGET = 9
LOCKED = 3
BORDER = 11

BAR_FILLED = 6
BAR_EMPTY = 4
BAR_BG = 5

MAX_LIVES = 3
LIFE_FULL = 2
LIFE_LOST = 8
GO_BG = 5
GO_FG = 6

STEP_BUDGET = {0: 2, 1: 6, 2: 9, 3: 10}

STEP_MULTIPLIER = 3

BAR_HEIGHT = 2

NUM_LEVELS = 5

_GLYPHS: dict[str, list[list[int]]] = {
    "G": [[1, 1, 1], [1, 0, 0], [1, 0, 1], [1, 0, 1], [1, 1, 1]],
    "A": [[0, 1, 0], [1, 0, 1], [1, 1, 1], [1, 0, 1], [1, 0, 1]],
    "M": [[1, 0, 1], [1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1]],
    "E": [[1, 1, 1], [1, 0, 0], [1, 1, 0], [1, 0, 0], [1, 1, 1]],
    "O": [[0, 1, 0], [1, 0, 1], [1, 0, 1], [1, 0, 1], [0, 1, 0]],
    "V": [[1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [0, 1, 0]],
    "R": [[1, 1, 0], [1, 0, 1], [1, 1, 0], [1, 0, 1], [1, 0, 1]],
}


def _parse_grid(lines):
    char_map = {
        ".": FLOOR,
        "W": WALL,
        "B": BLOCK,
        "T": TARGET,
        "#": BORDER,
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
    "##########",
    "#........#",
    "#.WWWWW..#",
    "#.W...W..#",
    "#.W.B.W..#",
    "#.W...W..#",
    "#.W..TW..#",
    "#.WWWWW..#",
    "#........#",
    "##########",
]

_L2 = [
    "############",
    "#..........#",
    "#.B..W.....#",
    "#.WW.W.....#",
    "#....W.....#",
    "#.T..W.....#",
    "#...W.W..B.#",
    "#.....W.WW.#",
    "#.....W....#",
    "#.....W..T.#",
    "#......W...#",
    "############",
]

_L3 = [
    "##############",
    "#............#",
    "#.B..W..W..B.#",
    "#.WW.W..W.WW.#",
    "#....W..W....#",
    "#....W..W....#",
    "#.T..W..W..T.#",
    "#...W....W...#",
    "#....B..W....#",
    "#....WW.W....#",
    "#....T..W....#",
    "#......W.....#",
    "#............#",
    "##############",
]

_L4 = [
    "##################",
    "#................#",
    "#..B...W....W..B.#",
    "#..WWW.W....W.WW.#",
    "#......W....W....#",
    "#......W....W....#",
    "#.WT...W....W..T.#",
    "#.....W......W...#",
    "#................#",
    "#................#",
    "#................#",
    "#................#",
    "#..B....WW....B..#",
    "#..WW...WW...WW..#",
    "#......T..T......#",
    "#.....W....W.....#",
    "#................#",
    "##################",
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
    "#.B...W......W....B..#",
    "#.WWW.W......W....WW.#",
    "#.....W......WT......#",
    "#.......TW...........#",
    "#....W...............#",
    "#....................#",
    "#....................#",
    "#....................#",
    "#..B.....W...........#",
    "#..WW....W...........#",
    "#.......TW...........#",
    "######################",
]


def _build_level(level_index):
    templates = [_L1, _L2, _L3, _L4, _L5]
    lines = templates[level_index]
    grid, blocks, targets = _parse_grid(lines)
    puzzle_h = len(grid)
    w = len(grid[0]) if puzzle_h > 0 else 0

    budget = STEP_BUDGET.get(level_index, 1)
    bar_len = min(budget * STEP_MULTIPLIER, w - 2)
    max_steps = bar_len

    grid.append([BAR_BG] * w)

    pad_left = (w - bar_len) // 2
    pad_right = w - bar_len - pad_left
    bar_row = [BAR_BG] * pad_left + [BAR_EMPTY] * bar_len + [BAR_BG] * pad_right
    grid.append(bar_row)

    h = puzzle_h + BAR_HEIGHT

    return {
        "grid": grid,
        "w": w,
        "h": h,
        "puzzle_h": puzzle_h,
        "blocks": blocks,
        "targets": targets,
        "max_steps": max_steps,
        "budget": budget,
        "bar_len": bar_len,
        "bar_row_y": puzzle_h + 1,
        "bar_start_x": pad_left,
    }


def _grid_to_sprite(grid):
    return Sprite(
        pixels=grid,
        name="board",
        x=0,
        y=0,
        layer=0,
        blocking=BlockingMode.PIXEL_PERFECT,
        tags=["sys_static"],
    )


class Pd41(ARCBaseGame):
    GAME_ID = "pd41"

    def __init__(self, seed: int = 0):
        self._seed = seed
        self._grid = None
        self._gw = 0
        self._gh = 0
        self._puzzle_h = 0
        self._blocks = []
        self._targets = []
        self._locked = set()
        self._level_index = 0
        self._active_block_pos = None
        self._steps_taken = 0
        self._max_steps = 0
        self._budget = 0
        self._bar_len = 0
        self._bar_row_y = 0
        self._bar_start_x = 0
        self._lives = MAX_LIVES
        self._game_over = False
        self._history = []

        levels = []
        for i in range(NUM_LEVELS):
            data = _build_level(i)
            gw, gh = data["w"], data["h"]
            lvl = Level(
                name=f"Level {i + 1}",
                grid_size=(gw, gh),
                sprites=[],
                data={"level_index": i},
            )
            levels.append(lvl)

        super().__init__(
            game_id="pd41",
            levels=levels,
            camera=Camera(width=64, height=64, background=5, letter_box=5),
            win_score=NUM_LEVELS,
            available_actions=[0, 1, 2, 3, 4, 5, 7],
            seed=seed,
        )

    def full_reset(self) -> None:
        self._lives = MAX_LIVES
        self._game_over = False
        self._history = []
        super().full_reset()

    def level_reset(self) -> None:
        self._lives = MAX_LIVES
        self._game_over = False
        self._history = []
        super().level_reset()

    def handle_reset(self) -> None:
        self._lives = MAX_LIVES
        self._game_over = False
        self._history = []
        self._init_puzzle(self._level_index)

    def on_set_level(self, level: Level) -> None:
        li = level.get_data("level_index")
        if li is None:
            li = 0
        self._level_index = li
        self._lives = MAX_LIVES
        self._game_over = False
        self._history = []
        self._init_puzzle(li)

    _ARROW_DIRS = {
        GameAction.ACTION1: (0, -1),
        GameAction.ACTION2: (0, +1),
        GameAction.ACTION3: (-1, 0),
        GameAction.ACTION4: (+1, 0),
    }

    def step(self) -> None:
        if self._game_over:
            self.complete_action()
            return

        act = self.action

        if act.id in self._ARROW_DIRS:
            dx, dy = self._ARROW_DIRS[act.id]
            self._handle_arrow(dx, dy)
        elif act.id == GameAction.ACTION5:
            self._cycle_active_block()
        elif act.id == GameAction.ACTION7:
            self._handle_undo()

        self._refresh_board()

        if self._check_win():
            self.next_level()
        elif self._steps_taken >= self._max_steps:
            self._lives -= 1
            if self._lives <= 0:
                self._game_over = True
                self._refresh_board()
                self._show_game_over_screen()
                self.lose()
            else:
                self._init_puzzle(self._level_index)

        self.complete_action()

    def _save_state(self):
        self._history.append(
            {
                "blocks": list(self._blocks),
                "locked": set(self._locked),
                "active_block_pos": self._active_block_pos,
                "steps_taken": self._steps_taken,
                "grid_puzzle": [row[:] for row in self._grid[: self._puzzle_h]],
            }
        )

    def _handle_undo(self):
        if not self._history:
            return
        state = self._history.pop()
        self._blocks = list(state["blocks"])
        self._locked = set(state["locked"])
        self._active_block_pos = state["active_block_pos"]
        self._steps_taken = state["steps_taken"]
        for y in range(self._puzzle_h):
            self._grid[y] = state["grid_puzzle"][y][:]
        self._steps_taken += 1

    def _init_puzzle(self, level_index):
        data = _build_level(level_index)
        self._grid = [row[:] for row in data["grid"]]
        self._gw = data["w"]
        self._gh = data["h"]
        self._puzzle_h = data["puzzle_h"]
        self._blocks = list(data["blocks"])
        self._targets = list(data["targets"])
        self._locked = set()
        self._steps_taken = 0
        self._max_steps = data["max_steps"]
        self._budget = data["budget"]
        self._bar_len = data["bar_len"]
        self._bar_row_y = data["bar_row_y"]
        self._bar_start_x = data["bar_start_x"]
        self._history = []
        if self._blocks:
            self._active_block_pos = self._blocks[0]
        else:
            self._active_block_pos = None
        self._refresh_board()

    def _refresh_board(self):
        self._update_block_colors()
        self._update_progress_bar()
        self._render_lives()
        level = self.current_level
        level.remove_all_sprites()
        board_sprite = _grid_to_sprite(self._grid)
        level.add_sprite(board_sprite)

    def _update_block_colors(self):
        active_pos = self._get_active_block()
        for bx, by in self._blocks:
            if (bx, by) in self._locked:
                continue
            if active_pos and (bx, by) == active_pos:
                self._grid[by][bx] = ACTIVE
            else:
                self._grid[by][bx] = BLOCK

    def _get_active_block(self):
        movable = [(bx, by) for bx, by in self._blocks if (bx, by) not in self._locked]
        if not movable:
            self._active_block_pos = None
            return None
        if self._active_block_pos in movable:
            return self._active_block_pos
        self._active_block_pos = movable[0]
        return self._active_block_pos

    def _cycle_active_block(self):
        movable = [(bx, by) for bx, by in self._blocks if (bx, by) not in self._locked]
        if len(movable) <= 1:
            return
        try:
            idx = movable.index(self._active_block_pos)
        except ValueError:
            idx = -1
        self._active_block_pos = movable[(idx + 1) % len(movable)]

    def _advance_active_after_lock(self):
        movable = [(bx, by) for bx, by in self._blocks if (bx, by) not in self._locked]
        if not movable:
            self._active_block_pos = None
            return
        if self._active_block_pos in movable:
            return
        self._active_block_pos = movable[0]

    def _update_progress_bar(self):
        y = self._bar_row_y
        if y <= 0 or y >= self._gh:
            return
        filled = min(self._steps_taken, self._bar_len)
        for i in range(self._bar_len):
            x = self._bar_start_x + i
            if x < self._gw:
                self._grid[y][x] = BAR_FILLED if i < filled else BAR_EMPTY

    def _render_lives(self):
        sep_y = self._puzzle_h
        if sep_y < 0 or sep_y >= self._gh:
            return
        w = self._gw
        box_xs = [w // 2 - 3, w // 2, w // 2 + 3]
        lives_lost = MAX_LIVES - self._lives
        for i, bx in enumerate(box_xs):
            if 0 <= bx < w:
                self._grid[sep_y][bx] = LIFE_LOST if i < lives_lost else LIFE_FULL

    def _show_game_over_screen(self):
        ph = self._puzzle_h
        w = self._gw
        for gy in range(1, ph - 1):
            for gx in range(1, w - 1):
                self._grid[gy][gx] = GO_BG

        def _draw_word(word: str, row_start: int) -> None:
            col_start = max(1, (w - len(word) * 3) // 2)
            for li, ch in enumerate(word):
                glyph = _GLYPHS.get(ch, _GLYPHS["E"])
                lx = col_start + li * 3
                for gy_off, grow in enumerate(glyph):
                    for gx_off, pixel in enumerate(grow):
                        if pixel:
                            px = lx + gx_off
                            py = row_start + gy_off
                            if 1 <= px < w - 1 and 1 <= py < ph - 1:
                                self._grid[py][px] = GO_FG

        top_row = max(1, ph // 4 - 2)
        bottom_row = max(1, ph * 2 // 4)
        _draw_word("GAME", top_row)
        _draw_word("OVER", bottom_row)

        self._render_lives()

        level = self.current_level
        level.remove_all_sprites()
        board_sprite = _grid_to_sprite(self._grid)
        level.add_sprite(board_sprite)

    def _handle_arrow(self, dx, dy):
        active_pos = self._get_active_block()
        if active_pos is None:
            return
        bx, by = active_pos
        nx, ny = bx + dx, by + dy
        if nx < 0 or nx >= self._gw or ny < 0 or ny >= self._puzzle_h:
            return
        next_cell = self._grid[ny][nx]
        if next_cell in (WALL, BORDER, BLOCK, ACTIVE, LOCKED):
            return
        self._save_state()
        self._grid[by][bx] = BLOCK
        self._slide_block(bx, by, dx, dy)
        self._steps_taken += 1
        new_pos = self._blocks[-1]
        if new_pos in self._locked:
            self._advance_active_after_lock()
        else:
            self._active_block_pos = new_pos

    def _slide_block(self, bx, by, dx, dy):
        if (bx, by) in [(tx, ty) for tx, ty in self._targets]:
            self._grid[by][bx] = TARGET
        else:
            self._grid[by][bx] = FLOOR
        self._blocks.remove((bx, by))
        cx, cy = bx, by
        while True:
            nx, ny = cx + dx, cy + dy
            if nx < 0 or nx >= self._gw or ny < 0 or ny >= self._puzzle_h:
                break
            next_cell = self._grid[ny][nx]
            if next_cell in (WALL, BORDER, BLOCK, ACTIVE, LOCKED):
                break
            cx, cy = nx, ny
            if (cx, cy) in self._targets and (cx, cy) not in self._locked:
                break
        final_x, final_y = cx, cy
        if (final_x, final_y) in self._targets and (
            final_x,
            final_y,
        ) not in self._locked:
            self._grid[final_y][final_x] = LOCKED
            self._locked.add((final_x, final_y))
            self._blocks.append((final_x, final_y))
        else:
            self._grid[final_y][final_x] = BLOCK
            self._blocks.append((final_x, final_y))

    def _check_win(self):
        for tx, ty in self._targets:
            if (tx, ty) not in self._locked:
                return False
        return True


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
        self._engine = Pd41(seed=seed)
        self._total_turns = 0
        self._last_action_was_reset = False
        self._last_frame_data = None

    def reset(self) -> GameState:
        self._total_turns = 0
        self._last_action_was_reset = False
        self._last_frame_data = self._engine.perform_action(
            ActionInput(id=GameAction.RESET)
        )
        return self._make_game_state()

    def step(self, action: str) -> StepResult:
        action = action.strip().lower()

        if action == "reset":
            if self._last_action_was_reset:
                self._last_action_was_reset = False
                self._engine.full_reset()
                self._last_frame_data = self._engine.perform_action(
                    ActionInput(id=GameAction.RESET)
                )
                state = self._make_game_state()
                return StepResult(state=state, reward=0.0, done=False)
            else:
                self._last_action_was_reset = True
                self._last_frame_data = self._engine.perform_action(
                    ActionInput(id=GameAction.RESET)
                )
                state = self._make_game_state()
                return StepResult(state=state, reward=0.0, done=False)

        self._last_action_was_reset = False

        game_action = _ACTION_MAP.get(action)
        if game_action is None:
            raise ValueError(f"Unknown action '{action}'. Valid: {_VALID_ACTIONS}")

        prev_level = self._engine._level_index
        prev_lives = self._engine._lives

        self._last_frame_data = self._engine.perform_action(ActionInput(id=game_action))
        self._total_turns += 1

        engine_state = self._last_frame_data.state
        done = engine_state in (EngineGameState.WIN, EngineGameState.GAME_OVER)

        reward = 0.0
        info: Dict[str, Any] = {"action": action, "engine_state": engine_state}

        if engine_state == EngineGameState.WIN:
            reward = 1.0 / len(self._engine._levels)
            info["outcome"] = "game_complete"
        elif engine_state == EngineGameState.GAME_OVER:
            reward = 0.0
            info["outcome"] = "game_over"
        elif self._engine._level_index > prev_level:
            reward = 1.0 / len(self._engine._levels)
            info["outcome"] = "level_complete"
        elif self._engine._lives < prev_lives:
            reward = 0.0
            info["outcome"] = "death"

        state = self._make_game_state()
        return StepResult(state=state, reward=reward, done=done, info=info)

    def get_actions(self) -> List[str]:
        engine_state = self._last_frame_data.state if self._last_frame_data else None
        if self._engine._game_over or engine_state in (
            EngineGameState.WIN,
            EngineGameState.GAME_OVER,
        ):
            return ["reset"]
        return _VALID_ACTIONS

    def is_done(self) -> bool:
        if self._last_frame_data is None:
            return False
        engine_state = self._last_frame_data.state
        return engine_state in (EngineGameState.WIN, EngineGameState.GAME_OVER)

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError(
                f"Unsupported render mode '{mode}'. Only 'rgb_array' is supported."
            )
        if self._last_frame_data is None or self._last_frame_data.is_empty():
            return np.zeros((64, 64, 3), dtype=np.uint8)
        return np.array(self._last_frame_data.frame, dtype=np.uint8)

    def close(self) -> None:
        self._engine = None

    def _build_text_observation(self):
        g = self._engine
        color_map = {0: ".", 5: "W", 8: "B", 10: "*", 9: "T", 3: "L", 11: "#"}
        lines = []
        lines.append(
            f"Level:{g._level_index + 1}/{NUM_LEVELS} Steps:{g._steps_taken}/{g._max_steps} Lives:{g._lives}/{MAX_LIVES}"
        )
        if g._game_over:
            lines.append("GAME_OVER")
        for y in range(g._puzzle_h):
            row = ""
            for x in range(g._gw):
                row += color_map.get(g._grid[y][x], "?")
            lines.append(row)
        return "\n".join(lines)

    def _make_game_state(self) -> GameState:
        image_bytes = None
        if self._last_frame_data and not self._last_frame_data.is_empty():
            try:
                frame = np.array(self._last_frame_data.frame, dtype=np.uint8)
                image_bytes = frame.tobytes()
            except Exception:
                pass

        valid = self.get_actions()

        return GameState(
            text_observation=self._build_text_observation(),
            image_observation=image_bytes,
            valid_actions=valid,
            turn=self._total_turns,
            metadata={
                "total_levels": len(self._engine._levels),
                "level": self._engine._level_index + 1,
                "steps_used": self._engine._steps_taken,
                "step_limit": self._engine._max_steps,
                "lives": self._engine._lives,
                "max_lives": MAX_LIVES,
                "game_over": self._engine._game_over,
            },
        )


class ArcGameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 5}

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
