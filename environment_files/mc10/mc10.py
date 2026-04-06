import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

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

C_BG = 1
C_BOUNDARY = 3
C_WALL = 5
C_PATH = 0
C_REVEAL = 6
C_OBSTACLE = 13
C_ENTRANCE = 10
C_EXIT = 14
C_CURSOR = 11
C_ISLAND = 8
C_CLUE_PENDING = 9
C_CLUE_OK = 14
C_CLUE_OVER = 8
C_HEART = 8
C_BAR = 9
C_BAR_EMPTY = 10

BACKGROUND_COLOR = C_BG
PADDING_COLOR = C_BOUNDARY
CAM_SIZE = 26
MAX_LIVES = 3
BAR_WIDTH = CAM_SIZE - 2

_LEVELS = [
    {
        "n": 6,
        "max_moves": 48,
        "seed": 1001,
        "target_cells": 7,
        "extra_reveals": 1,
    },
    {
        "n": 6,
        "max_moves": 52,
        "seed": 1002,
        "target_cells": 8,
        "extra_reveals": 1,
    },
    {
        "n": 6,
        "max_moves": 56,
        "seed": 1003,
        "target_cells": 9,
        "extra_reveals": 2,
        "obstacles": [(1, 1), (2, 2)],
    },
    {
        "n": 7,
        "max_moves": 64,
        "seed": 1004,
        "target_cells": 12,
        "extra_reveals": 2,
        "obstacles": [(1, 1), (2, 3), (5, 1)],
    },
]

GridPos = Tuple[int, int]

_DIRECTIONS: Dict[Any, Tuple[int, int]] = {
    GameAction.ACTION1: (-1, 0),
    GameAction.ACTION2: (1, 0),
    GameAction.ACTION3: (0, -1),
    GameAction.ACTION4: (0, 1),
}


def _px(color: int, layer: int = 0, name: str = "px") -> Sprite:
    return Sprite(
        pixels=np.array([[color]], dtype=np.int32),
        name=name,
        visible=True,
        collidable=False,
        tags=[],
        layer=layer,
    )


def _color_array(color: int) -> np.ndarray:
    return np.array([[color]], dtype=np.int32)


def _neighbors(n: int, row: int, col: int) -> List[GridPos]:
    out: List[GridPos] = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = row + dr, col + dc
        if 0 <= rr < n and 0 <= cc < n:
            out.append((rr, cc))
    return out


def _build_dfs_tree(
    n: int, seed: int, start: GridPos, goal: GridPos
) -> Dict[GridPos, Optional[GridPos]]:
    rng = np.random.default_rng(seed)
    parent: Dict[GridPos, Optional[GridPos]] = {start: None}
    visited: Set[GridPos] = {start}
    stack = [start]

    while stack:
        row, col = stack[-1]
        nxt = [pos for pos in _neighbors(n, row, col) if pos not in visited]
        if not nxt:
            stack.pop()
            continue

        nxt.sort(
            key=lambda pos: (
                abs(pos[0] - goal[0]) + abs(pos[1] - goal[1]),
                rng.random(),
            )
        )
        chosen = nxt[0]
        visited.add(chosen)
        parent[chosen] = (row, col)
        stack.append(chosen)

    return parent


def _path_to_root(
    parent: Dict[GridPos, Optional[GridPos]], node: GridPos
) -> List[GridPos]:
    out: List[GridPos] = [node]
    cur: GridPos = node
    nxt = parent[cur]
    while nxt is not None:
        cur = nxt
        out.append(cur)
        nxt = parent[cur]
    out.reverse()
    return out


def _tree_children(
    parent: Dict[GridPos, Optional[GridPos]],
) -> Dict[GridPos, List[GridPos]]:
    children: Dict[GridPos, List[GridPos]] = {pos: [] for pos in parent}
    for node, par in parent.items():
        if par is not None:
            children[par].append(node)
    return children


def _generate_layout(
    level_data: dict,
) -> Tuple[Set[GridPos], Set[GridPos], GridPos, GridPos]:
    n = level_data["n"]
    start = (n // 2, 0)
    goal = (n // 2, n - 1)
    parent = _build_dfs_tree(n, level_data["seed"], start, goal)
    children = _tree_children(parent)
    rng = np.random.default_rng(level_data["seed"] + 101)

    path = _path_to_root(parent, goal)
    target_cells: Set[GridPos] = set(path)
    frontier: List[GridPos] = list(path)

    while len(target_cells) < level_data["target_cells"]:
        expandable = [
            pos for pos in frontier if any(ch not in target_cells for ch in children[pos])
        ]
        if not expandable:
            break
        rng.shuffle(expandable)
        root = expandable[0]
        options = [ch for ch in children[root] if ch not in target_cells]
        options.sort(key=lambda pos: len(children[pos]), reverse=True)
        choice = options[0]
        target_cells.add(choice)
        frontier.append(choice)

    fixed_open: Set[GridPos] = {start, goal}
    degrees: Dict[GridPos, int] = {}
    for row, col in target_cells:
        deg = 0
        for rr, cc in _neighbors(n, row, col):
            if (rr, cc) in target_cells:
                deg += 1
        degrees[(row, col)] = deg

    endpoints = [
        pos for pos in target_cells if pos not in (start, goal) and degrees[pos] == 1
    ]
    endpoints.sort(
        key=lambda pos: abs(pos[0] - start[0]) + abs(pos[1] - start[1]),
        reverse=True,
    )
    for pos in endpoints[: level_data["extra_reveals"]]:
        fixed_open.add(pos)

    if len(fixed_open) < level_data["extra_reveals"] + 2:
        remaining = [pos for pos in path if pos not in fixed_open]
        remaining.reverse()
        for pos in remaining:
            fixed_open.add(pos)
            if len(fixed_open) >= level_data["extra_reveals"] + 2:
                break

    return target_cells, fixed_open, start, goal


class Mc10(ARCBaseGame):
    _DIRECTIONS: Dict[Any, Tuple[int, int]] = {
        GameAction.ACTION1: (-1, 0),
        GameAction.ACTION2: (1, 0),
        GameAction.ACTION3: (0, -1),
        GameAction.ACTION4: (0, 1),
    }

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)
        self._history: List[Dict] = []
        levels = [
            Level(sprites=[], grid_size=(CAM_SIZE, CAM_SIZE), data=d) for d in _LEVELS
        ]
        self._lives = MAX_LIVES
        self._mid_level_reset = False
        self._lost_game = False
        cam = Camera(0, 0, CAM_SIZE, CAM_SIZE, BACKGROUND_COLOR, PADDING_COLOR, [])
        super().__init__("mc10", levels, cam, available_actions=[0, 1, 2, 3, 4, 5, 7])

    def next_level(self) -> None:
        self._lives = MAX_LIVES
        super().next_level()

    def handle_reset(self) -> None:
        if self._lost_game:
            self._lost_game = False
            self._lives = MAX_LIVES
            self._mid_level_reset = True
            self.level_reset()
            return
        self._lives = MAX_LIVES
        self._lost_game = False
        super().handle_reset()

    def on_set_level(self, level: Level) -> None:
        self._lives = MAX_LIVES
        self._history = []
        data = _LEVELS[min(self._current_level_index, len(_LEVELS) - 1)]
        self._n = data["n"]
        self._max_moves = data["max_moves"]
        self._move_count = 0

        if not self._mid_level_reset:
            self._lives = MAX_LIVES
        self._mid_level_reset = False

        assert self._max_moves > 0

        self._pixel_cache: Dict[int, np.ndarray] = {}

        self._target_cells, self._fixed_open, self._start, self._goal = _generate_layout(
            data
        )
        self._obstacles = {tuple(pos) for pos in data.get("obstacles", [])}
        self._row_targets = [
            sum((row, col) in self._target_cells for col in range(self._n))
            for row in range(self._n)
        ]
        self._col_targets = [
            sum((row, col) in self._target_cells for row in range(self._n))
            for col in range(self._n)
        ]

        self._offset_x = (CAM_SIZE - self._n) // 2
        self._offset_y = (CAM_SIZE - self._n) // 2
        self._layout_x = self._offset_x - self._n - 1
        self._layout_y = self._offset_y - self._n - 1
        self._board = np.zeros((self._n, self._n), dtype=np.int32)
        for row, col in self._fixed_open:
            self._board[row, col] = 1

        self._cell_sprites: List[List[Sprite]] = []
        for row in range(self._n):
            sprite_row: List[Sprite] = []
            for col in range(self._n):
                sp = _px(C_WALL, layer=1, name="cell")
                sp.set_position(self._offset_x + col, self._offset_y + row)
                self.current_level.add_sprite(sp)
                sprite_row.append(sp)
            self._cell_sprites.append(sprite_row)

        self._row_clues: List[List[Sprite]] = []
        for row in range(self._n):
            clue_row: List[Sprite] = []
            for x in range(self._n):
                sp = _px(C_BG, layer=0, name="row_clue")
                sp.set_position(self._layout_x + x, self._offset_y + row)
                self.current_level.add_sprite(sp)
                clue_row.append(sp)
            self._row_clues.append(clue_row)

        self._col_clues: List[List[Sprite]] = []
        for col in range(self._n):
            clue_col: List[Sprite] = []
            for y in range(self._n):
                sp = _px(C_BG, layer=0, name="col_clue")
                sp.set_position(self._offset_x + col, self._layout_y + y)
                self.current_level.add_sprite(sp)
                clue_col.append(sp)
            self._col_clues.append(clue_col)

        self._life_sprites: List[Sprite] = []
        for idx in range(MAX_LIVES):
            sp = _px(C_HEART, layer=5, name="life")
            sp.set_position(CAM_SIZE - 4 + idx, 0)
            self.current_level.add_sprite(sp)
            self._life_sprites.append(sp)

        self._move_bar_sprites: List[Sprite] = []
        self._move_bar_x = 1
        self._move_bar_y = CAM_SIZE - 1
        for col in range(BAR_WIDTH):
            sp = _px(C_BAR_EMPTY, layer=5, name="move_bar")
            sp.set_position(self._move_bar_x + col, self._move_bar_y)
            self.current_level.add_sprite(sp)
            self._move_bar_sprites.append(sp)

        open_cells = list(self._fixed_open)
        if open_cells:
            self._cursor_row, self._cursor_col = self._rng.choice(open_cells)
        else:
            self._cursor_row, self._cursor_col = self._start
        self._cursor_sprite = _px(C_CURSOR, layer=10, name="cursor")
        self._update_cursor()
        self.current_level.add_sprite(self._cursor_sprite)

        self._refresh()

    def _cached_pixel(self, color: int) -> np.ndarray:
        arr = self._pixel_cache.get(color)
        if arr is None:
            arr = _color_array(color)
            self._pixel_cache[color] = arr
        return arr

    def _update_cursor(self) -> None:
        self._cursor_sprite.set_position(
            self._offset_x + self._cursor_col, self._offset_y + self._cursor_row
        )

    def _current_open_cells(self) -> Set[GridPos]:
        carved: Set[GridPos] = set()
        for row in range(self._n):
            for col in range(self._n):
                if self._board[row, col] == 1:
                    carved.add((row, col))
        return carved

    def _reachable_from_start(self, carved: Set[GridPos]) -> Set[GridPos]:
        if self._start not in carved:
            return set()
        seen: Set[GridPos] = {self._start}
        queue: deque[GridPos] = deque([self._start])
        while queue:
            row, col = queue.popleft()
            for nxt in _neighbors(self._n, row, col):
                if nxt in carved and nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        return seen

    def _edge_count(self, carved: Set[GridPos]) -> int:
        edges = 0
        for row, col in carved:
            if (row + 1, col) in carved:
                edges += 1
            if (row, col + 1) in carved:
                edges += 1
        return edges

    def _analyze_board(self) -> dict:
        carved = self._current_open_cells()
        reachable = self._reachable_from_start(carved)
        isolated = carved - reachable
        row_counts = [
            sum(self._board[row, col] == 1 for col in range(self._n))
            for row in range(self._n)
        ]
        col_counts = [
            sum(self._board[row, col] == 1 for row in range(self._n))
            for col in range(self._n)
        ]
        counts_match = (
            row_counts == self._row_targets and col_counts == self._col_targets
        )
        connected = len(isolated) == 0 and self._start in carved
        has_exit_path = self._goal in reachable
        edges = self._edge_count(carved)
        is_tree = connected and edges == len(carved) - 1
        solved = counts_match and has_exit_path and is_tree
        return {
            "carved": carved,
            "reachable": reachable,
            "isolated": isolated,
            "row_counts": row_counts,
            "col_counts": col_counts,
            "solved": solved,
        }

    def _refresh_clues(self, row_counts: List[int], col_counts: List[int]) -> None:
        for row in range(self._n):
            target = self._row_targets[row]
            current = row_counts[row]
            for x in range(self._n):
                from_right = self._n - 1 - x
                color = C_BG
                if from_right < min(current, target):
                    color = C_CLUE_OK
                elif current <= target and from_right < target:
                    color = C_CLUE_PENDING
                elif current > target:
                    overflow = current - target
                    if from_right < target:
                        color = C_CLUE_OVER
                    elif from_right < target + overflow:
                        color = C_CLUE_OVER
                self._row_clues[row][x].pixels = self._cached_pixel(color)

        for col in range(self._n):
            target = self._col_targets[col]
            current = col_counts[col]
            for y in range(self._n):
                from_bottom = self._n - 1 - y
                color = C_BG
                if from_bottom < min(current, target):
                    color = C_CLUE_OK
                elif current <= target and from_bottom < target:
                    color = C_CLUE_PENDING
                elif current > target:
                    overflow = current - target
                    if from_bottom < target:
                        color = C_CLUE_OVER
                    elif from_bottom < target + overflow:
                        color = C_CLUE_OVER
                self._col_clues[col][y].pixels = self._cached_pixel(color)

    def _refresh_move_bar(self) -> None:
        filled = int(
            BAR_WIDTH * min(self._move_count, self._max_moves) / self._max_moves
        )
        for col in range(BAR_WIDTH):
            color = C_BAR if col < filled else C_BAR_EMPTY
            self._move_bar_sprites[col].pixels = self._cached_pixel(color)

    def _refresh(self) -> dict:
        info = self._analyze_board()
        isolated = info["isolated"]
        for row in range(self._n):
            for col in range(self._n):
                pos = (row, col)
                if pos == self._start:
                    color = C_ENTRANCE
                elif pos == self._goal:
                    color = C_EXIT
                elif pos in self._obstacles:
                    color = C_OBSTACLE
                elif pos in self._fixed_open:
                    color = C_REVEAL
                elif self._board[row, col] == 1:
                    color = C_ISLAND if pos in isolated else C_PATH
                else:
                    color = C_WALL
                self._cell_sprites[row][col].pixels = self._cached_pixel(color)

        self._refresh_clues(info["row_counts"], info["col_counts"])
        self._refresh_move_bar()
        for idx, sp in enumerate(self._life_sprites):
            sp.set_visible(idx < self._lives)
        return info

    def _refresh_after_cursor_move(self) -> None:
        self._refresh_move_bar()

    def _reset_level_state(self) -> None:
        self._board[:, :] = 0
        for row, col in self._fixed_open:
            self._board[row, col] = 1
        self._move_count = 0
        self._history.clear()
        open_cells = list(self._fixed_open)
        if open_cells:
            self._cursor_row, self._cursor_col = self._rng.choice(open_cells)
        else:
            self._cursor_row, self._cursor_col = self._start
        self._update_cursor()
        self._refresh()

    def _trigger_life_loss(self) -> bool:
        self._lives -= 1
        if self._lives <= 0:
            self._lost_game = True
            self.lose()
            self.complete_action()
            return True
        self._reset_level_state()
        return False

    def _save_undo(self) -> None:
        self._history.append({
            "board": self._board.copy(),
            "cursor_row": self._cursor_row,
            "cursor_col": self._cursor_col,
        })

    def _do_undo(self) -> None:
        if not self._history:
            return
        snap = self._history.pop()
        self._board[:, :] = snap["board"]
        self._cursor_row = snap["cursor_row"]
        self._cursor_col = snap["cursor_col"]
        self._update_cursor()
        self._refresh()

    def _spend_step(self) -> bool:
        self._move_count += 1
        self._refresh_move_bar()
        if self._move_count >= self._max_moves:
            return self._trigger_life_loss()
        return False

    def _move_cursor(self, delta_row: int, delta_col: int) -> bool:
        new_row = self._cursor_row + delta_row
        new_col = self._cursor_col + delta_col
        if not (0 <= new_row < self._n and 0 <= new_col < self._n):
            return False
        self._cursor_row = new_row
        self._cursor_col = new_col
        self._update_cursor()
        self._refresh_after_cursor_move()
        return True

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

        direction = self._DIRECTIONS.get(act)
        if direction is not None:
            self._save_undo()
            if self._move_cursor(direction[0], direction[1]):
                if self._spend_step():
                    self.complete_action()
                    return
            self.complete_action()
            return

        if act == GameAction.ACTION5:
            pos = (self._cursor_row, self._cursor_col)
            if pos not in self._fixed_open and pos not in self._obstacles:
                self._save_undo()
                self._board[self._cursor_row, self._cursor_col] ^= 1
                info = self._refresh()
                if info["solved"]:
                    self.next_level()
                    self.complete_action()
                    return
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
        self._engine = Mc10(seed=seed)
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
        lines.append(f"Level {e._current_level_index + 1} | Moves {e._move_count}/{e._max_moves} | Lives {e._lives}")
        for row in range(n):
            row_str = ""
            for col in range(n):
                pos = (row, col)
                if pos == e._start:
                    ch = "E"
                elif pos == e._goal:
                    ch = "X"
                elif pos in e._obstacles:
                    ch = "O"
                elif board[row, col] == 1 and pos in e._fixed_open:
                    ch = "R"
                elif board[row, col] == 1:
                    ch = "P"
                else:
                    ch = "W"
                if (row, col) == (e._cursor_row, e._cursor_col):
                    ch = f"[{ch}]"
                else:
                    ch = f" {ch} "
                row_str += ch
            lines.append(row_str)
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
