from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import struct
import zlib

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from arcengine import (
    ActionInput,
    ARCBaseGame,
    Camera,
    GameAction,
    GameState as _ArcGameState,
    Level,
    RenderableUserDisplay,
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


C_BG = 5
C_CELL_BORDER = 3
C_EMPTY = 2
C_CURSOR = 4
C_EDGE = 1
C_RUNNER = 8
C_TRAIL = 6
C_DEATH = 7
C_WALL = 9
C_PREWALL = 10
C_HUD_CLICKS = 0
C_WIN = 11
C_HUD_LEVEL = 12
C_HUD_MOVES = 13
C_EDGE_VIZ = 14
C_RESERVED = 15

DISPLAY_SIZE = 64
HUD_HEIGHT = 3
HUD_TOP = DISPLAY_SIZE - HUD_HEIGHT

MAX_LIVES = 3


def _square_cell_size(cols: int, rows: int) -> int:
    max_w = DISPLAY_SIZE // cols
    max_h = HUD_TOP // rows
    return max(4, min(max_w, max_h))


def cardinal_neighbors(c: int, r: int) -> List[Tuple[int, int]]:
    return [
        (c, r - 1),
        (c + 1, r),
        (c, r + 1),
        (c - 1, r),
    ]


def bfs_to_edge_cardinal(
    c: int,
    r: int,
    cols: int,
    rows: int,
    walls: Set[Tuple[int, int]],
    edge_order: List[Tuple[int, int]],
) -> Optional[List[Tuple[int, int]]]:
    if is_edge_cell(c, r, cols, rows) and (c, r) not in walls:
        return [(c, r)]

    visited: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {(c, r): None}
    queue: deque = deque([(c, r, 0)])
    edge_hits: Dict[Tuple[int, int], int] = {}
    best_dist = float("inf")

    while queue:
        cx, cy, dist = queue.popleft()
        if dist > best_dist:
            break

        for nc, nr in cardinal_neighbors(cx, cy):
            if nc < 0 or nc >= cols or nr < 0 or nr >= rows:
                continue
            if (nc, nr) in visited:
                continue
            if (nc, nr) in walls:
                continue

            visited[(nc, nr)] = (cx, cy)

            if is_edge_cell(nc, nr, cols, rows):
                new_dist = dist + 1
                if new_dist <= best_dist:
                    best_dist = new_dist
                    edge_hits[(nc, nr)] = new_dist
                    queue.append((nc, nr, new_dist))
                continue

            queue.append((nc, nr, dist + 1))

    if not edge_hits:
        return None

    best_edge = None
    best_order = float("inf")
    edge_order_map = {pos: i for i, pos in enumerate(edge_order)}
    for edge_pos, d in edge_hits.items():
        if d == best_dist:
            order = edge_order_map.get(edge_pos, float("inf"))
            if order < best_order:
                best_order = order
                best_edge = edge_pos

    if best_edge is None:
        return None

    path = []
    node: Optional[Tuple[int, int]] = best_edge
    while node is not None:
        path.append(node)
        node = visited.get(node)
    path.reverse()
    return path


def cardinal_neighbors_wrap(
    c: int,
    r: int,
    cols: int,
    rows: int,
) -> List[Tuple[int, int]]:
    return [
        (c, (r - 1) % rows),
        ((c + 1) % cols, r),
        (c, (r + 1) % rows),
        ((c - 1) % cols, r),
    ]


def bfs_to_edge_cardinal_wrap(
    c: int,
    r: int,
    cols: int,
    rows: int,
    walls: Set[Tuple[int, int]],
    edge_order: List[Tuple[int, int]],
) -> Optional[List[Tuple[int, int]]]:
    if is_edge_cell(c, r, cols, rows) and (c, r) not in walls:
        return [(c, r)]

    visited: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {(c, r): None}
    queue: deque = deque([(c, r, 0)])
    edge_hits: Dict[Tuple[int, int], int] = {}
    best_dist = float("inf")

    while queue:
        cx, cy, dist = queue.popleft()
        if dist > best_dist:
            break

        for nc, nr in cardinal_neighbors_wrap(cx, cy, cols, rows):
            if (nc, nr) in visited:
                continue
            if (nc, nr) in walls:
                continue

            visited[(nc, nr)] = (cx, cy)

            if is_edge_cell(nc, nr, cols, rows):
                new_dist = dist + 1
                if new_dist <= best_dist:
                    best_dist = new_dist
                    edge_hits[(nc, nr)] = new_dist
                    queue.append((nc, nr, new_dist))
                continue

            queue.append((nc, nr, dist + 1))

    if not edge_hits:
        return None

    best_edge = None
    best_order = float("inf")
    edge_order_map = {pos: i for i, pos in enumerate(edge_order)}
    for edge_pos, d in edge_hits.items():
        if d == best_dist:
            order = edge_order_map.get(edge_pos, float("inf"))
            if order < best_order:
                best_order = order
                best_edge = edge_pos

    if best_edge is None:
        return None

    path = []
    node: Optional[Tuple[int, int]] = best_edge
    while node is not None:
        path.append(node)
        node = visited.get(node)
    path.reverse()
    return path


def compute_square_grid_offset(
    cols: int,
    rows: int,
) -> Tuple[int, int]:
    cs = _square_cell_size(cols, rows)
    total_w = cols * cs
    total_h = rows * cs
    x_off = (DISPLAY_SIZE - total_w) // 2
    y_off = (HUD_TOP - total_h) // 2
    return max(0, x_off), max(0, y_off)


def render_square_grid(
    frame: np.ndarray,
    cols: int,
    rows: int,
    x_off: int,
    y_off: int,
    cell_colors: Dict[Tuple[int, int], int],
    edge_cells: Set[Tuple[int, int]],
    show_borders: bool = True,
) -> None:
    cs = _square_cell_size(cols, rows)
    for r in range(rows):
        for c in range(cols):
            color = cell_colors.get((c, r), C_EMPTY)
            base_x = x_off + c * cs
            base_y = y_off + r * cs

            for dy in range(cs):
                for dx in range(cs):
                    px_x = base_x + dx
                    px_y = base_y + dy
                    if 0 <= px_x < DISPLAY_SIZE and 0 <= px_y < HUD_TOP:
                        is_border = dx == 0 or dx == cs - 1 or dy == 0 or dy == cs - 1
                        if show_borders and is_border:
                            frame[px_y, px_x] = C_CELL_BORDER
                        else:
                            frame[px_y, px_x] = color


def is_edge_cell(c: int, r: int, cols: int, rows: int) -> bool:
    return r == 0 or r == rows - 1 or c == 0 or c == cols - 1


def get_edge_cells(cols: int, rows: int) -> List[Tuple[int, int]]:
    edges = []
    seen = set()

    for c in range(cols):
        if (c, 0) not in seen:
            edges.append((c, 0))
            seen.add((c, 0))

    for r in range(rows):
        if (cols - 1, r) not in seen:
            edges.append((cols - 1, r))
            seen.add((cols - 1, r))

    for c in range(cols - 1, -1, -1):
        if (c, rows - 1) not in seen:
            edges.append((c, rows - 1))
            seen.add((c, rows - 1))

    for r in range(rows - 1, -1, -1):
        if (0, r) not in seen:
            edges.append((0, r))
            seen.add((0, r))

    return edges


def generate_level_1(_seed: int) -> Dict[str, Any]:
    cols, rows = 6, 6
    runner_start = (3, 1)

    prewalls: List[Tuple[int, int]] = []
    for c in range(cols):
        prewalls.append((c, 0))
    for r in range(rows):
        if (0, r) not in [(c, 0) for c in range(cols)]:
            prewalls.append((0, r))
        if (cols - 1, r) not in [(c, 0) for c in range(cols)]:
            prewalls.append((cols - 1, r))

    prewalls = sorted(set(prewalls))

    return {
        "cols": cols,
        "rows": rows,
        "runner_start": runner_start,
        "prewalls": prewalls,
        "click_budget": 6,
        "max_moves": 30,
        "move_type": "cardinal",
        "grid_type": "square",
        "num_runners": 1,
        "wrap": False,
        "panic_radius": 0,
    }


def generate_level_2(_seed: int) -> Dict[str, Any]:
    cols, rows = 8, 8
    runner_start = (4, 4)

    prewalls: List[Tuple[int, int]] = []
    for r in range(rows):
        prewalls.append((0, r))
        prewalls.append((cols - 1, r))

    prewalls = sorted(set(prewalls))

    return {
        "cols": cols,
        "rows": rows,
        "runner_start": runner_start,
        "prewalls": prewalls,
        "click_budget": 8,
        "max_moves": 50,
        "move_type": "cardinal",
        "grid_type": "square",
        "num_runners": 1,
        "wrap": False,
        "panic_radius": 0,
    }


def generate_level_3(_seed: int) -> Dict[str, Any]:
    cols, rows = 12, 12
    runner_start = (6, 6)

    prewalls = [
        (8, 4),
        (8, 5),
        (8, 6),
        (8, 7),
        (8, 8),
    ]

    return {
        "cols": cols,
        "rows": rows,
        "runner_start": runner_start,
        "prewalls": prewalls,
        "click_budget": 14,
        "max_moves": 90,
        "grid_type": "square",
        "move_type": "cardinal",
        "num_runners": 1,
        "wrap": True,
        "panic_radius": 0,
    }


def generate_level_4(_seed: int) -> Dict[str, Any]:
    cols, rows = 8, 8
    runner_start = (4, 4)

    prewalls: List[Tuple[int, int]] = []

    return {
        "cols": cols,
        "rows": rows,
        "runner_start": runner_start,
        "prewalls": prewalls,
        "click_budget": 10,
        "max_moves": 60,
        "move_type": "cardinal",
        "grid_type": "square",
        "num_runners": 1,
        "wrap": False,
        "panic_radius": 0,
    }


LEVEL_GENERATORS = [
    generate_level_1,
    generate_level_2,
    generate_level_3,
    generate_level_4,
]


class _LevelState:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.cols: int = config["cols"]
        self.rows: int = config["rows"]
        self.wrap: bool = config["wrap"]
        self.num_runners: int = config["num_runners"]
        self.click_budget: int = config["click_budget"]

        if self.num_runners > 1 and "runner_starts" in config:
            self.runner_positions: List[Tuple[int, int]] = list(config["runner_starts"])
            self.runner_start_positions: List[Tuple[int, int]] = list(
                config["runner_starts"]
            )
        else:
            self.runner_positions = [config["runner_start"]]
            self.runner_start_positions = [config["runner_start"]]

        self.prewalls: Set[Tuple[int, int]] = set(config["prewalls"])
        self.player_walls: Set[Tuple[int, int]] = set()
        self.all_walls: Set[Tuple[int, int]] = set(self.prewalls)
        self.edge_order: List[Tuple[int, int]] = get_edge_cells(self.cols, self.rows)
        self.edge_set: Set[Tuple[int, int]] = set(self.edge_order)
        self.trails: Set[Tuple[int, int]] = set()
        self.clicks_used: int = 0
        self.moves_used: int = 0
        self.max_moves: int = config["max_moves"]
        self.runner_trapped: bool = False
        self.runner_escaped: bool = False

    def reset(self, config: Dict[str, Any]) -> None:
        if self.num_runners > 1 and "runner_starts" in config:
            self.runner_positions = list(config["runner_starts"])
        else:
            self.runner_positions = [config["runner_start"]]
        self.player_walls = set()
        self.all_walls = set(self.prewalls)
        self.trails = set()
        self.clicks_used = 0
        self.moves_used = 0
        self.runner_trapped = False
        self.runner_escaped = False

    def _find_path(self, pos: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        if self.wrap:
            return bfs_to_edge_cardinal_wrap(
                pos[0],
                pos[1],
                self.cols,
                self.rows,
                self.all_walls,
                self.edge_order,
            )
        return bfs_to_edge_cardinal(
            pos[0],
            pos[1],
            self.cols,
            self.rows,
            self.all_walls,
            self.edge_order,
        )

    def all_runners_trapped(self) -> bool:
        for pos in self.runner_positions:
            path = self._find_path(pos)
            if path is not None:
                return False
        return True

    def move_runners(self, last_wall_pos: Optional[Tuple[int, int]] = None) -> bool:
        escaped = False
        for i, pos in enumerate(self.runner_positions):
            path = self._find_path(pos)

            if path is None:
                continue

            if len(path) <= 1:
                escaped = True
                continue

            self.trails.add(pos)
            new_pos = path[1]
            self.runner_positions[i] = new_pos
            pos = new_pos

            if is_edge_cell(pos[0], pos[1], self.cols, self.rows) and not self.wrap:
                escaped = True

        self.runner_escaped = escaped
        return escaped


class _NeuralBlockadeHUD(RenderableUserDisplay):
    def __init__(self, game: "Hx01") -> None:
        self._game = game

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        g = self._game
        idx = g._current_level_index
        if idx >= len(g._level_states):
            return frame

        state = g._level_states[idx]
        config = g._level_configs[idx]

        for y in range(HUD_TOP, DISPLAY_SIZE):
            for x in range(DISPLAY_SIZE):
                frame[y, x] = C_BG

        hud_y1 = HUD_TOP

        remaining = state.click_budget - state.clicks_used
        total = state.click_budget
        for i in range(min(total, 20)):
            px = 1 + i * 2
            if px < DISPLAY_SIZE:
                if i < remaining:
                    frame[hud_y1, px] = C_HUD_CLICKS
                else:
                    frame[hud_y1, px] = C_CURSOR

        lives = g._lives
        center_x = DISPLAY_SIZE // 2
        lives_start = center_x - (MAX_LIVES - 1)
        for i in range(MAX_LIVES):
            px = lives_start + i * 2
            if 0 <= px < DISPLAY_SIZE:
                if i < lives:
                    frame[hud_y1, px] = C_RUNNER
                else:
                    frame[hud_y1, px] = C_CURSOR

        num_levels = len(g._level_states)
        for i in range(num_levels):
            px = DISPLAY_SIZE - 2 - i * 2
            if px >= 0:
                if i < idx:
                    frame[hud_y1, px] = C_WIN
                elif i == idx:
                    frame[hud_y1, px] = C_HUD_LEVEL
                else:
                    frame[hud_y1, px] = C_CURSOR

        hud_y2 = HUD_TOP + 1
        remaining_moves = max(0, state.max_moves - state.moves_used)
        bar_width = DISPLAY_SIZE - 2
        if state.max_moves > 0:
            fill = int((remaining_moves / state.max_moves) * bar_width)
        else:
            fill = bar_width

        for x in range(bar_width):
            px = 1 + x
            if x < fill:
                frame[hud_y2, px] = C_HUD_MOVES
            else:
                frame[hud_y2, px] = C_CURSOR

        hud_y3 = HUD_TOP + 2
        if state.runner_trapped:
            for x in range(DISPLAY_SIZE):
                frame[hud_y3, x] = C_WIN
        elif state.runner_escaped:
            for x in range(DISPLAY_SIZE):
                frame[hud_y3, x] = C_DEATH

        cc = g._cursor_col
        cr = g._cursor_row
        cs = _square_cell_size(state.cols, state.rows)
        x_off, y_off = compute_square_grid_offset(state.cols, state.rows)
        if 0 <= cc < state.cols and 0 <= cr < state.rows:
            px_left = x_off + cc * cs
            px_right = px_left + cs - 1
            px_top = y_off + cr * cs
            px_bottom = px_top + cs - 1

            for px in range(px_left, px_right + 1):
                if 0 <= px < DISPLAY_SIZE:
                    if 0 <= px_top < HUD_TOP:
                        frame[px_top, px] = C_HUD_CLICKS
                    if 0 <= px_bottom < HUD_TOP:
                        frame[px_bottom, px] = C_HUD_CLICKS
            for py in range(px_top, px_bottom + 1):
                if 0 <= py < HUD_TOP:
                    if 0 <= px_left < DISPLAY_SIZE:
                        frame[py, px_left] = C_HUD_CLICKS
                    if 0 <= px_right < DISPLAY_SIZE:
                        frame[py, px_right] = C_HUD_CLICKS

        return frame


class Hx01(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self._history: List[Dict] = []

        self._level_configs: List[Dict[str, Any]] = []
        self._level_states: List[_LevelState] = []

        levels: List[Level] = []
        for i, gen_fn in enumerate(LEVEL_GENERATORS):
            config = gen_fn(seed)
            self._level_configs.append(config)
            self._level_states.append(_LevelState(config))

            frame_pixels = np.full((DISPLAY_SIZE, DISPLAY_SIZE), C_BG, dtype=np.int8)
            sprite = Sprite(
                pixels=frame_pixels.tolist(),
                name=f"grid_{i}",
                x=0,
                y=0,
                layer=0,
            )

            level = Level(
                sprites=[sprite],
                grid_size=(DISPLAY_SIZE, DISPLAY_SIZE),
                name=f"Level {i + 1}",
                data={"grid_index": i},
            )
            levels.append(level)

        self._lives = MAX_LIVES
        self._consecutive_reset_count: int = 0

        self._cursor_col = 0
        self._cursor_row = 0

        camera = Camera(
            x=0,
            y=0,
            width=DISPLAY_SIZE,
            height=DISPLAY_SIZE,
            background=C_BG,
            letter_box=C_BG,
        )

        super().__init__(
            game_id="hx01",
            levels=levels,
            camera=camera,
            available_actions=[0, 1, 2, 3, 4, 5, 7],
        )

        overlay = _NeuralBlockadeHUD(self)
        self.camera.replace_interface([overlay])

    def on_set_level(self, level: Level) -> None:
        idx = level.get_data("grid_index")
        if idx is not None and idx < len(self._level_states):
            self._history = []
            state = self._level_states[idx]
            self._cursor_col = state.cols // 2
            self._cursor_row = state.rows // 2
            self._render_level(idx)

    def handle_reset(self) -> None:
        self._consecutive_reset_count += 1

        if self._consecutive_reset_count >= 2:
            self._consecutive_reset_count = 0
            self._lives = MAX_LIVES
            self.full_reset()
        else:
            self._lives = MAX_LIVES
            self.level_reset()

    def next_level(self) -> None:
        self._lives = MAX_LIVES
        super().next_level()

    def full_reset(self) -> None:
        super().full_reset()
        self._lives = MAX_LIVES
        self._history = []
        for i, config in enumerate(self._level_configs):
            self._level_states[i].reset(config)
        idx = self._current_level_index
        if idx < len(self._level_states):
            state = self._level_states[idx]
            self._cursor_col = state.cols // 2
            self._cursor_row = state.rows // 2
            self._render_level(idx)

    def level_reset(self) -> None:
        idx = self._current_level_index
        super().level_reset()
        self._history = []
        config = self._level_configs[idx]
        self._level_states[idx].reset(config)
        state = self._level_states[idx]
        self._cursor_col = state.cols // 2
        self._cursor_row = state.rows // 2
        self._render_level(idx)

    def _save_state(self) -> None:
        idx = self._current_level_index
        if idx >= len(self._level_states):
            return
        state = self._level_states[idx]
        snap: Dict = {
            "cursor_col": self._cursor_col,
            "cursor_row": self._cursor_row,
            "runner_positions": [pos for pos in state.runner_positions],
            "player_walls": set(state.player_walls),
            "all_walls": set(state.all_walls),
            "trails": set(state.trails),
            "clicks_used": state.clicks_used,
            "moves_used": state.moves_used,
            "runner_trapped": state.runner_trapped,
            "runner_escaped": state.runner_escaped,
        }
        self._history.append(snap)

    def _undo(self) -> None:
        if not self._history:
            return
        idx = self._current_level_index
        if idx >= len(self._level_states):
            return
        snap = self._history.pop()
        state = self._level_states[idx]
        self._cursor_col = snap["cursor_col"]
        self._cursor_row = snap["cursor_row"]
        state.runner_positions = snap["runner_positions"]
        state.player_walls = snap["player_walls"]
        state.all_walls = snap["all_walls"]
        state.trails = snap["trails"]
        state.clicks_used = snap["clicks_used"]
        state.runner_trapped = snap["runner_trapped"]
        state.runner_escaped = snap["runner_escaped"]
        self._render_level(idx)

    def _render_level(self, idx: int) -> None:
        if idx >= len(self._level_states):
            return

        state = self._level_states[idx]
        sprites = self.current_level.get_sprites_by_name(f"grid_{idx}")
        if not sprites:
            return

        frame = np.full((DISPLAY_SIZE, DISPLAY_SIZE), C_BG, dtype=np.int8)

        cell_colors: Dict[Tuple[int, int], int] = {}
        edge_cells = state.edge_set

        for r in range(state.rows):
            for c in range(state.cols):
                pos = (c, r)
                if pos in state.prewalls:
                    cell_colors[pos] = C_PREWALL
                elif pos in state.player_walls:
                    cell_colors[pos] = C_WALL
                elif pos in state.trails:
                    cell_colors[pos] = C_TRAIL
                elif pos in edge_cells and not state.wrap:
                    cell_colors[pos] = C_EDGE
                else:
                    cell_colors[pos] = C_EMPTY

        for rpos in state.runner_positions:
            if state.runner_escaped:
                cell_colors[rpos] = C_DEATH
            else:
                cell_colors[rpos] = C_RUNNER

        x_off, y_off = compute_square_grid_offset(state.cols, state.rows)
        render_square_grid(
            frame,
            state.cols,
            state.rows,
            x_off,
            y_off,
            cell_colors,
            edge_cells,
        )

        sprites[0].pixels = frame

    def step(self) -> None:
        action = self.action

        if action.id == GameAction.RESET:
            self.complete_action()
            return

        self._consecutive_reset_count = 0
        idx = self._current_level_index
        if idx < len(self._level_states):
            state = self._level_states[idx]

            state.moves_used += 1

            if action.id == GameAction.ACTION7:
                self._undo()
                if state.moves_used >= state.max_moves:
                    if not state.runner_trapped and not state.runner_escaped:
                        self._on_moves_exhausted()
                self.complete_action()
                return

            if action.id == GameAction.ACTION1:
                self._cursor_row = max(0, self._cursor_row - 1)
                self._render_level(idx)
            elif action.id == GameAction.ACTION2:
                self._cursor_row = min(state.rows - 1, self._cursor_row + 1)
                self._render_level(idx)
            elif action.id == GameAction.ACTION3:
                self._cursor_col = max(0, self._cursor_col - 1)
                self._render_level(idx)
            elif action.id == GameAction.ACTION4:
                self._cursor_col = min(state.cols - 1, self._cursor_col + 1)
                self._render_level(idx)
            elif action.id == GameAction.ACTION5:
                self._save_state()
                self._place_at_grid(idx, self._cursor_col, self._cursor_row)

            if state.moves_used >= state.max_moves:
                if not state.runner_trapped and not state.runner_escaped:
                    self._on_moves_exhausted()

        self.complete_action()

    def _place_at_grid(self, idx: int, c: int, r: int) -> None:
        state = self._level_states[idx]

        if state.runner_trapped or state.runner_escaped:
            return

        if (c, r) in state.all_walls:
            state.clicks_used += 1
            if state.clicks_used >= state.click_budget:
                self._on_budget_exhausted()
            self._render_level(idx)
            return

        if (c, r) in [(p[0], p[1]) for p in state.runner_positions]:
            state.clicks_used += 1
            if state.clicks_used >= state.click_budget:
                self._on_budget_exhausted()
            self._render_level(idx)
            return

        state.player_walls.add((c, r))
        state.all_walls.add((c, r))
        state.clicks_used += 1

        if state.all_runners_trapped():
            state.runner_trapped = True
            self._render_level(idx)
            self.next_level()
            return

        escaped = state.move_runners(last_wall_pos=(c, r))

        if escaped:
            state.runner_escaped = True
            self._render_level(idx)
            self._lives -= 1
            if self._lives <= 0:
                self.lose()
            else:
                self.level_reset()
            return

        if state.all_runners_trapped():
            state.runner_trapped = True
            self._render_level(idx)
            self.next_level()
            return

        if state.clicks_used >= state.click_budget:
            self._on_budget_exhausted()
            return

        self._render_level(idx)

    def _on_budget_exhausted(self) -> None:
        self._lives -= 1
        if self._lives <= 0:
            self.lose()
        else:
            self.level_reset()

    def _on_moves_exhausted(self) -> None:
        self._lives -= 1
        if self._lives <= 0:
            self.lose()
        else:
            self.level_reset()

    def _build_state(self) -> None:
        idx = self._current_level_index
        if idx >= len(self._level_states):
            self.text_observation = ""
            self.image_observation = None
            return
        state = self._level_states[idx]
        grid = [["." for _ in range(state.cols)] for _ in range(state.rows)]
        for r in range(state.rows):
            for c in range(state.cols):
                pos = (c, r)
                if pos in state.prewalls:
                    grid[r][c] = "#"
                elif pos in state.player_walls:
                    grid[r][c] = "W"
                elif pos in state.trails:
                    grid[r][c] = "~"
                elif is_edge_cell(c, r, state.cols, state.rows) and not state.wrap:
                    grid[r][c] = "E"
        for rpos in state.runner_positions:
            rx, ry = rpos
            if 0 <= ry < state.rows and 0 <= rx < state.cols:
                grid[ry][rx] = "R"
        cc, cr = self._cursor_col, self._cursor_row
        if 0 <= cr < state.rows and 0 <= cc < state.cols:
            if grid[cr][cc] == ".":
                grid[cr][cc] = "@"
        self.image_observation = "\n".join(" ".join(row) for row in grid)
        self.text_observation = (
            "NEURAL BLOCKADE | "
            "Actions: 1=Up 2=Down 3=Left 4=Right 5=PlaceWall 7=Undo | "
            "Level: {lvl}/{tot} | Grid: {cols}x{rows} | "
            "Clicks: {cu}/{cb} | Moves: {mu}/{mm} | "
            "Lives: {lives} | Cursor: ({cc},{cr}) | "
            "Trapped: {trap} | Escaped: {esc}"
        ).format(
            lvl=idx + 1,
            tot=len(self._level_states),
            cols=state.cols,
            rows=state.rows,
            cu=state.clicks_used,
            cb=state.click_budget,
            mu=state.moves_used,
            mm=state.max_moves,
            lives=self._lives,
            cc=self._cursor_col,
            cr=self._cursor_row,
            trap=state.runner_trapped,
            esc=state.runner_escaped,
        )


class PuzzleEnvironment:
    _ACTION_MAP: Dict[str, GameAction] = {
        "up": GameAction.ACTION1,
        "down": GameAction.ACTION2,
        "left": GameAction.ACTION3,
        "right": GameAction.ACTION4,
        "select": GameAction.ACTION5,
        "undo": GameAction.ACTION7,
        "reset": GameAction.RESET,
    }

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

    def __init__(self, seed: int = 0) -> None:
        self._engine = Hx01(seed=seed)
        self._total_turns = 0
        self._last_action_was_reset = False
        self._game_won = False
        self._total_levels = len(self._engine._levels)

    def _build_text_obs(self) -> str:
        e = self._engine
        if not hasattr(e, "text_observation"):
            e._build_state()
        return e.text_observation

    def _build_image_bytes(self) -> Optional[bytes]:
        e = self._engine
        rendered = e.camera.render(e.current_level.get_sprites())
        if hasattr(rendered, "tolist"):
            grid = rendered.tolist()
        else:
            grid = rendered if rendered else []
        if not grid:
            return None
        arr = np.array(grid, dtype=np.uint8)
        h, w = arr.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for idx, color in enumerate(self.ARC_PALETTE):
            mask = arr == idx
            rgb[mask] = color

        def _pack_png(data: np.ndarray) -> bytes:
            raw_bytes = b""
            for row in data:
                raw_bytes += b"\x00" + row.tobytes()
            compressed = zlib.compress(raw_bytes)

            def _chunk(tag: bytes, body: bytes) -> bytes:
                return (
                    struct.pack(">I", len(body))
                    + tag
                    + body
                    + struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF)
                )

            ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
            return (
                b"\x89PNG\r\n\x1a\n"
                + _chunk(b"IHDR", ihdr)
                + _chunk(b"IDAT", compressed)
                + _chunk(b"IEND", b"")
            )

        return _pack_png(rgb)

    def _build_game_state(
        self, done: bool = False, info: Optional[Dict] = None
    ) -> GameState:
        e = self._engine
        valid_actions = self.get_actions() if not done else None
        idx = e.level_index
        state = e._level_states[idx] if idx < len(e._level_states) else None
        return GameState(
            text_observation=self._build_text_obs(),
            image_observation=self._build_image_bytes(),
            valid_actions=valid_actions,
            turn=self._total_turns,
            metadata={
                "level_index": e.level_index,
                "total_levels": self._total_levels,
                "levels_completed": getattr(e, "_score", 0),
                "game_over": getattr(getattr(e, "_state", None), "name", "")
                == "GAME_OVER",
                "lives": e._lives,
                "clicks_used": state.clicks_used if state else 0,
                "click_budget": state.click_budget if state else 0,
                "moves_used": state.moves_used if state else 0,
                "max_moves": state.max_moves if state else 0,
                "done": done,
                "info": info or {},
            },
        )

    def reset(self) -> GameState:
        e = self._engine
        self._total_turns = 0
        game_won = hasattr(e, "_state") and getattr(e._state, "name", "") == "WIN"
        if game_won or self._last_action_was_reset:
            e._consecutive_reset_count = getattr(e, "_consecutive_reset_count", 0) + 1
        else:
            e._consecutive_reset_count = 0
        e.perform_action(ActionInput(id=GameAction.RESET))
        e._build_state()
        self._last_action_was_reset = True
        self._game_won = False
        return self._build_game_state()

    def is_done(self) -> bool:
        return self._game_won

    def get_actions(self) -> List[str]:
        if self._game_won:
            return ["reset"]
        e = self._engine
        game_over = getattr(getattr(e, "_state", None), "name", "") == "GAME_OVER"
        if game_over:
            return ["reset"]
        return ["up", "down", "left", "right", "select", "undo", "reset"]

    def step(self, action: str) -> StepResult:
        e = self._engine

        if action == "reset":
            state = self.reset()
            return StepResult(
                state=state, reward=0.0, done=False, info={"action": "reset"}
            )

        if action not in self._ACTION_MAP:
            self._total_turns += 1
            return StepResult(
                state=self._build_game_state(done=False),
                reward=0.0,
                done=False,
                info={"action": action, "reason": "invalid_action"},
            )

        self._last_action_was_reset = False
        self._total_turns += 1

        game_action = self._ACTION_MAP[action]
        info: Dict = {"action": action}

        level_before = e.level_index
        action_input = ActionInput(id=game_action)
        frame = e.perform_action(action_input, raw=True)
        e._build_state()
        level_after = e.level_index

        game_won = frame and frame.state and frame.state.name == "WIN"
        game_over = frame and frame.state and frame.state.name == "GAME_OVER"
        done = game_over or game_won

        if game_won:
            self._game_won = True
            info["reason"] = "game_complete"
            reward = 1.0 / self._total_levels
            return StepResult(
                state=self._build_game_state(done=True, info=info),
                reward=reward,
                done=True,
                info=info,
            )

        if game_over:
            info["reason"] = "death"
            return StepResult(
                state=self._build_game_state(done=False, info=info),
                reward=0.0,
                done=False,
                info=info,
            )

        if level_after > level_before:
            info["reason"] = "level_complete"
            reward = 1.0 / self._total_levels
            return StepResult(
                state=self._build_game_state(done=False, info=info),
                reward=reward,
                done=False,
                info=info,
            )

        return StepResult(
            state=self._build_game_state(done=False, info=info),
            reward=0.0,
            done=False,
            info=info,
        )

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        e = self._engine
        index_grid = e.camera.render(e.current_level.get_sprites())
        h, w = index_grid.shape[0], index_grid.shape[1]
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for idx, color in enumerate(self.ARC_PALETTE):
            mask = index_grid == idx
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
        self._seed: int = seed

        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"Unsupported render_mode '{render_mode}'. "
                f"Supported: {self.metadata['render_modes']}"
            )
        self.render_mode: Optional[str] = render_mode

        self._env: Optional[PuzzleEnvironment] = None

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
        frame = self._env.render()
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

    env = ArcGameEnv(render_mode="rgb_array")
    try:
        check_env(env.unwrapped, skip_render_check=True)
        print("[PASS] check_env passed — environment is Gymnasium-compliant.")
    except Exception as e:
        print(f"[FAIL] check_env failed: {e}")

    obs, info = env.reset()
    print(f"Obs shape: {obs.shape}, dtype: {obs.dtype}")
    print(f"Valid actions: {info.get('valid_actions')}")

    mask = env.action_mask()
    print(f"Action mask: {mask}")

    valid_indices = np.where(mask == 1)[0]
    if len(valid_indices) > 0:
        obs, reward, term, trunc, info = env.step(valid_indices[0])
        print(f"Step → reward={reward}, terminated={term}, truncated={trunc}")

    env.close()
    print("Smoke test passed!")
