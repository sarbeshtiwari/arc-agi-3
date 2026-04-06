import random
import struct
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from arcengine import (
    ActionInput,
    ARCBaseGame,
    Camera,
    GameAction,
    Level,
    Sprite,
)
import gymnasium as gym
from gymnasium import spaces

C_BG = 1
C_GRID = 2
C_BOUNDARY = 3
C_CONTINENT = 14
C_ROUTE = 9
C_BRIDGE = 0
C_BAD = 8
C_ANCHOR = 11
C_FALSE = 12
C_OBSTACLE = 5
C_HEART = 8
C_BAR = 9
C_BAR_EMPTY = 10
C_BAR_MID = 2
C_BAR_HIGH = 14

CAM_SIZE = 32
GRAPH_SCALE = 2
BACKGROUND_COLOR = C_BG
PADDING_COLOR = C_BOUNDARY

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


def _index_grid_to_rgb_64(index_grid: np.ndarray) -> np.ndarray:
    arr = np.asarray(index_grid, dtype=np.uint8)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    h, w = arr.shape[:2]
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for idx in range(len(ARC_PALETTE)):
        mask = arr == idx
        rgb[mask] = ARC_PALETTE[idx]
    out_size = 64
    if h == out_size and w == out_size:
        return rgb
    scale_y = out_size / h
    scale_x = out_size / w
    ys = (np.arange(out_size) / scale_y).astype(int)
    xs = (np.arange(out_size) / scale_x).astype(int)
    ys = np.clip(ys, 0, h - 1)
    xs = np.clip(xs, 0, w - 1)
    return rgb[np.ix_(ys, xs)]


GridPos = Tuple[int, int]

_LEVELS = [
    {
        "continents": [(1, 1), (1, 9), (7, 1), (7, 9)],
        "edges": [
            {"nodes": (0, 1), "waypoints": [(1, 1), (1, 9)]},
            {"nodes": (0, 2), "waypoints": [(1, 1), (7, 1)]},
            {"nodes": (1, 3), "waypoints": [(1, 9), (7, 9)]},
            {"nodes": (2, 3), "waypoints": [(7, 1), (7, 9)]},
            {"nodes": (0, 3), "waypoints": [(1, 1), (4, 1), (4, 7), (7, 7), (7, 9)]},
        ],
        "max_moves": 108,
    },
    {
        "continents": [(0, 6), (6, 0), (6, 6), (6, 12), (12, 6)],
        "edges": [
            {"nodes": (0, 2), "waypoints": [(0, 6), (6, 6)]},
            {"nodes": (1, 2), "waypoints": [(6, 0), (6, 6)]},
            {"nodes": (2, 3), "waypoints": [(6, 6), (6, 12)]},
            {"nodes": (2, 4), "waypoints": [(6, 6), (12, 6)]},
            {"nodes": (0, 1), "waypoints": [(0, 6), (0, 0), (6, 0)]},
            {"nodes": (0, 3), "waypoints": [(0, 6), (0, 12), (6, 12)]},
            {"nodes": (1, 4), "waypoints": [(6, 0), (12, 0), (12, 6)]},
            {"nodes": (3, 4), "waypoints": [(6, 12), (12, 12), (12, 6)]},
        ],
        "max_moves": 132,
    },
    {
        "continents": [(0, 6), (6, 0), (6, 6), (6, 12), (12, 6), (12, 12)],
        "edges": [
            {"nodes": (0, 2), "waypoints": [(0, 6), (2, 6), (2, 8), (6, 8), (6, 6)]},
            {"nodes": (1, 2), "waypoints": [(6, 0), (4, 0), (4, 4), (6, 4), (6, 6)]},
            {"nodes": (2, 3), "waypoints": [(6, 6), (8, 6), (8, 10), (6, 10), (6, 12)]},
            {"nodes": (2, 4), "waypoints": [(6, 6), (6, 4), (10, 4), (10, 6), (12, 6)]},
            {"nodes": (4, 5), "waypoints": [(12, 6), (10, 6), (10, 10), (12, 10), (12, 12)]},
            {"nodes": (3, 5), "waypoints": [(6, 12), (8, 12), (8, 10), (12, 10), (12, 12)]},
            {"nodes": (0, 1), "waypoints": [(0, 6), (0, 0), (6, 0)]},
            {"nodes": (1, 4), "waypoints": [(6, 0), (12, 0), (12, 6)]},
        ],
        "max_moves": 204,
        "false_activators": [(0, 3)],
        "obstacles": [(9, 0), (12, 3)],
    },
    {
        "continents": [(0, 0), (0, 6), (0, 12), (8, 0), (8, 6), (8, 12)],
        "edges": [
            {"nodes": (0, 1), "waypoints": [(0, 0), (2, 0), (2, 4), (0, 4), (0, 6)]},
            {"nodes": (1, 2), "waypoints": [(0, 6), (0, 8), (2, 8), (2, 12), (0, 12)]},
            {"nodes": (1, 4), "waypoints": [(0, 6), (4, 6), (4, 4), (8, 4), (8, 6)]},
            {"nodes": (3, 4), "waypoints": [(8, 0), (6, 0), (6, 4), (8, 4), (8, 6)]},
            {"nodes": (4, 5), "waypoints": [(8, 6), (8, 8), (6, 8), (6, 12), (8, 12)]},
            {"nodes": (0, 3), "waypoints": [(0, 0), (8, 0)]},
            {"nodes": (2, 5), "waypoints": [(0, 12), (8, 12)]},
            {"nodes": (0, 4), "waypoints": [(0, 0), (4, 0), (4, 6), (8, 6)]},
            {"nodes": (2, 4), "waypoints": [(0, 12), (4, 12), (4, 6), (8, 6)]},
        ],
        "max_moves": 168,
        "false_activators": [(4, 2), (4, 10)],
        "obstacles": [(2, 0), (2, 12), (6, 0), (6, 12)],
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


def _line_cells(start: GridPos, end: GridPos) -> List[GridPos]:
    row1, col1 = start
    row2, col2 = end
    cells: List[GridPos] = []
    if row1 == row2:
        step = 1 if col2 > col1 else -1
        for col in range(col1 + step, col2, step):
            cells.append((row1, col))
    elif col1 == col2:
        step = 1 if row2 > row1 else -1
        for row in range(row1 + step, row2, step):
            cells.append((row, col1))
    return cells


def _polyline_cells(points: List[GridPos]) -> List[GridPos]:
    cells: List[GridPos] = []
    for idx in range(len(points) - 1):
        segment = _line_cells(points[idx], points[idx + 1])
        cells.extend(segment)
        if idx < len(points) - 2:
            cells.append(points[idx + 1])
    return cells


def _scale_point(pos: GridPos) -> GridPos:
    return (pos[0] * GRAPH_SCALE, pos[1] * GRAPH_SCALE)


def _make_grid_background() -> np.ndarray:
    grid = np.full((CAM_SIZE, CAM_SIZE), C_BG, dtype=np.int32)
    for row in range(CAM_SIZE):
        for col in range(CAM_SIZE):
            if row % GRAPH_SCALE == 0 or col % GRAPH_SCALE == 0:
                grid[row, col] = C_GRID
    return grid


class _DSU:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False
        self.parent[rb] = ra
        return True


class Cm10(ARCBaseGame):
    def __init__(self, seed: int = 0, cursor_nonce: int = 0) -> None:
        self._rng_seed = seed
        self._cursor_spawn_salt = cursor_nonce
        self._rng = random.Random(seed)
        levels = [
            Level(sprites=[], grid_size=(CAM_SIZE, CAM_SIZE), data=dict(d))
            for d in _LEVELS
        ]
        self.total_levels = len(levels)
        self._game_won = False
        super().__init__(
            "cm10",
            levels,
            Camera(
                width=CAM_SIZE,
                height=CAM_SIZE,
                background=BACKGROUND_COLOR,
                letter_box=PADDING_COLOR,
                interfaces=[],
            ),
            available_actions=[0, 1, 2, 3, 4, 5, 7],
        )

    def next_level(self) -> None:
        self._lives = 3
        super().next_level()

    def on_set_level(self, level: Level) -> None:
        level_idx = self._current_level_index % len(_LEVELS)
        data = _LEVELS[level_idx]
        self._max_moves = data["max_moves"]
        self._move_count = 0
        self._lives = 3
        self._history: List[Dict] = []

        raw_continents = [_scale_point(tuple(pos)) for pos in data["continents"]]
        raw_obstacles = [_scale_point(tuple(pos)) for pos in data.get("obstacles", [])]
        raw_false_activators = [
            _scale_point(tuple(pos)) for pos in data.get("false_activators", [])
        ]
        raw_edges = []
        all_points: List[GridPos] = list(raw_continents) + raw_obstacles + raw_false_activators
        for edge in data["edges"]:
            waypoints = [_scale_point(tuple(pos)) for pos in edge["waypoints"]]
            path = _polyline_cells(waypoints)
            anchor = path[len(path) // 2]
            raw_edges.append({"nodes": tuple(edge["nodes"]), "path": path, "anchor": anchor})
            all_points.extend(path)
            all_points.append(anchor)

        min_row = min(row for row, _ in all_points)
        max_row = max(row for row, _ in all_points)
        min_col = min(col for _, col in all_points)
        max_col = max(col for _, col in all_points)
        graph_h = max_row - min_row + 1
        graph_w = max_col - min_col + 1
        shift_row = (CAM_SIZE - graph_h) // 2 - min_row
        shift_col = (CAM_SIZE - graph_w) // 2 - min_col

        self._continents = [(row + shift_row, col + shift_col) for row, col in raw_continents]
        self._obstacles = {(row + shift_row, col + shift_col) for row, col in raw_obstacles}
        self._false_anchors = {
            (row + shift_row, col + shift_col) for row, col in raw_false_activators
        }
        self._edges = []
        for edge in raw_edges:
            self._edges.append(
                {
                    "nodes": edge["nodes"],
                    "path": [
                        (row + shift_row, col + shift_col) for row, col in edge["path"]
                    ],
                    "anchor": (
                        edge["anchor"][0] + shift_row,
                        edge["anchor"][1] + shift_col,
                    ),
                }
            )

        self._active_edges: Set[int] = set()
        self._active_false: Set[GridPos] = set()

        self._grid_background = Sprite(
            pixels=_make_grid_background(),
            name="grid_background",
            visible=True,
            collidable=False,
            tags=[],
            layer=0,
        )
        self._grid_background.set_position(0, 0)
        self.current_level.add_sprite(self._grid_background)

        self._continent_sprites = []
        for row, col in self._continents:
            sp = _px(C_CONTINENT, layer=4, name="continent")
            sp.set_position(col, row)
            self.current_level.add_sprite(sp)
            self._continent_sprites.append(sp)

        self._path_sprites: Dict[GridPos, Sprite] = {}
        self._path_to_edges: Dict[GridPos, List[int]] = {}
        self._anchor_sprites: List[Sprite] = []
        self._anchor_to_edge: Dict[GridPos, int] = {}
        for idx, edge in enumerate(self._edges):
            for row, col in edge["path"]:
                pos = (row, col)
                if pos not in self._path_sprites:
                    sp = _px(C_ROUTE, layer=1, name="bridge_path")
                    sp.set_position(col, row)
                    self.current_level.add_sprite(sp)
                    self._path_sprites[pos] = sp
                    self._path_to_edges[pos] = []
                self._path_to_edges[pos].append(idx)

            anchor = _px(C_ANCHOR, layer=5, name="bridge_anchor")
            anchor.set_position(edge["anchor"][1], edge["anchor"][0])
            self.current_level.add_sprite(anchor)
            self._anchor_sprites.append(anchor)
            self._anchor_to_edge[edge["anchor"]] = idx

        self._obstacle_sprites = []
        for row, col in sorted(self._obstacles):
            sp = _px(C_OBSTACLE, layer=4, name="obstacle")
            sp.set_position(col, row)
            self.current_level.add_sprite(sp)
            self._obstacle_sprites.append(sp)

        self._false_anchor_sprites: Dict[GridPos, Sprite] = {}
        for row, col in sorted(self._false_anchors):
            sp = _px(C_FALSE, layer=5, name="false_anchor")
            sp.set_position(col, row)
            self.current_level.add_sprite(sp)
            self._false_anchor_sprites[(row, col)] = sp

        self._life_sprites = []
        for idx in range(3):
            sp = _px(C_HEART, layer=8, name="life")
            sp.set_position(CAM_SIZE - 4 + idx, 0)
            self.current_level.add_sprite(sp)
            self._life_sprites.append(sp)

        self._bar_width = CAM_SIZE - 4
        self._bar_height = 2
        self._bar_capacity = self._bar_width * self._bar_height
        self._move_bar_sprites: List[List[Sprite]] = []
        self._move_bar_x = 2
        self._move_bar_y = CAM_SIZE - self._bar_height
        for row in range(self._bar_height):
            sprite_row: List[Sprite] = []
            for col in range(self._bar_width):
                sp = _px(C_BAR_EMPTY, layer=8, name="move_bar")
                sp.set_position(self._move_bar_x + col, self._move_bar_y + row)
                self.current_level.add_sprite(sp)
                sprite_row.append(sp)
            self._move_bar_sprites.append(sprite_row)

        self._bump_cursor_rng()
        cr, cc = self._random_interactable_cursor()
        self._cursor_row, self._cursor_col = cr, cc
        self._cursor_sprite = _px(C_ANCHOR, layer=10, name="cursor")
        self._update_cursor()
        self.current_level.add_sprite(self._cursor_sprite)

        self._refresh()

    def _update_cursor(self) -> None:
        self._cursor_sprite.set_position(self._cursor_col, self._cursor_row)

    def _bump_cursor_rng(self) -> None:
        self._cursor_spawn_salt += 1
        li = self._current_level_index % len(_LEVELS)
        mix = (
            self._rng_seed
            + (li + 1) * 0x9E3779B9
            + self._cursor_spawn_salt * 0x517CC1B7
        ) & 0x7FFFFFFF
        self._rng = random.Random(mix if mix != 0 else 1)

    def _random_interactable_cursor(self) -> GridPos:
        anchors: List[GridPos] = [e["anchor"] for e in self._edges]
        pool: List[GridPos] = list(anchors) + sorted(self._false_anchors)
        if not pool:
            return (0, 0)
        return self._rng.choice(pool)

    def _analyze_graph(self) -> dict:
        dsu = _DSU(len(self._continents))
        bad_edges: Set[int] = set()
        for idx in sorted(self._active_edges):
            a, b = self._edges[idx]["nodes"]
            if not dsu.union(a, b):
                bad_edges.add(idx)

        roots = {dsu.find(idx) for idx in range(len(self._continents))}
        connected = len(roots) == 1
        objective_met = (
            connected
            and len(self._active_edges) == len(self._continents) - 1
            and not bad_edges
            and not self._active_false
        )
        return {
            "bad_edges": bad_edges,
            "connected": connected,
            "objective_met": objective_met,
            "bad_false": set(self._active_false),
        }

    def _refresh_move_bar(self) -> None:
        used = min(self._move_count, self._max_moves)
        m = self._max_moves
        cap = self._bar_capacity
        if m <= 0:
            filled = 0
        else:
            filled = min(cap, (used * cap + m - 1) // m)
        stress = used / m if m > 0 else 0.0
        t = 0
        for row in range(self._bar_height):
            for col in range(self._bar_width):
                if t >= filled:
                    color = C_BAR_EMPTY
                elif stress >= 0.85:
                    color = C_BAR_HIGH
                elif stress >= 0.55:
                    color = C_BAR
                else:
                    color = C_BAR_MID
                t += 1
                self._move_bar_sprites[row][col].pixels = np.array([[color]], dtype=np.int32)

    def _refresh(self) -> dict:
        info = self._analyze_graph()
        bad_edges = info["bad_edges"]
        for pos, sp in self._path_sprites.items():
            edge_ids = self._path_to_edges[pos]
            has_bad = any(idx in bad_edges for idx in edge_ids)
            has_active = any(idx in self._active_edges for idx in edge_ids)
            color = C_BAD if has_bad else C_BRIDGE if has_active else C_ROUTE
            sp.pixels = np.array([[color]], dtype=np.int32)

        for idx, _edge in enumerate(self._edges):
            anchor_color = C_BAD if idx in bad_edges else C_ANCHOR
            self._anchor_sprites[idx].pixels = np.array([[anchor_color]], dtype=np.int32)

        for pos, sp in self._false_anchor_sprites.items():
            color = C_BAD if pos in info["bad_false"] else C_FALSE
            sp.pixels = np.array([[color]], dtype=np.int32)

        self._refresh_move_bar()
        for idx, sp in enumerate(self._life_sprites):
            sp.set_visible(idx < self._lives)
        return info

    def _reset_level(self) -> None:
        self._active_edges.clear()
        self._active_false.clear()
        self._move_count = 0
        self._history.clear()
        self._bump_cursor_rng()
        cr, cc = self._random_interactable_cursor()
        self._cursor_row, self._cursor_col = cr, cc
        self._update_cursor()
        self._refresh()

    def _trigger_life_loss(self) -> bool:
        self._lives -= 1
        if self._lives <= 0:
            self.lose()
            return True
        self._reset_level()
        return False

    def _spend_move(self) -> bool:
        self._move_count += 1
        self._refresh_move_bar()
        if self._move_count >= self._max_moves:
            return self._trigger_life_loss()
        return False

    def _save_state(self) -> None:
        self._history.append(
            {
                "active_edges": set(self._active_edges),
                "active_false": set(self._active_false),
                "cursor_row": self._cursor_row,
                "cursor_col": self._cursor_col,
            }
        )

    def _undo(self) -> None:
        if not self._history:
            return
        snap = self._history.pop()
        self._active_edges = set(snap["active_edges"])
        self._active_false = set(snap["active_false"])
        self._cursor_row = snap["cursor_row"]
        self._cursor_col = snap["cursor_col"]
        self._update_cursor()
        self._refresh()

    def _toggle_at_cursor(self) -> bool:
        pos = (self._cursor_row, self._cursor_col)
        if pos in self._anchor_to_edge:
            idx = self._anchor_to_edge[pos]
            if idx in self._active_edges:
                self._active_edges.remove(idx)
            else:
                self._active_edges.add(idx)
        elif pos in self._false_anchors:
            if pos in self._active_false:
                self._active_false.remove(pos)
            else:
                self._active_false.add(pos)
        else:
            return False

        self._move_count += 1
        info = self._refresh()
        if info["objective_met"]:
            if self._current_level_index >= self.total_levels - 1:
                self._game_won = True
            self.next_level()
            return True
        if self._move_count >= self._max_moves:
            self._trigger_life_loss()
            return True
        return False

    def step(self) -> None:
        if self.action.id == GameAction.RESET:
            self.complete_action()
            return

        if self.action.id == GameAction.ACTION7:
            if self._spend_move():
                self.complete_action()
                return
            self._undo()
            self.complete_action()
            return

        if self.action.id == GameAction.ACTION5:
            pos = (self._cursor_row, self._cursor_col)
            if pos not in self._anchor_to_edge and pos not in self._false_anchors:
                self.complete_action()
                return
            self._save_state()
            done_level = self._toggle_at_cursor()
            if done_level:
                self.complete_action()
                return
            self.complete_action()
            return

        if self.action.id == GameAction.ACTION1:
            if self._cursor_row > 0:
                self._save_state()
                self._cursor_row -= 1
                self._update_cursor()
                if self._spend_move():
                    self.complete_action()
                    return
            self.complete_action()
            return

        if self.action.id == GameAction.ACTION2:
            if self._cursor_row < CAM_SIZE - 1:
                self._save_state()
                self._cursor_row += 1
                self._update_cursor()
                if self._spend_move():
                    self.complete_action()
                    return
            self.complete_action()
            return

        if self.action.id == GameAction.ACTION3:
            if self._cursor_col > 0:
                self._save_state()
                self._cursor_col -= 1
                self._update_cursor()
                if self._spend_move():
                    self.complete_action()
                    return
            self.complete_action()
            return

        if self.action.id == GameAction.ACTION4:
            if self._cursor_col < CAM_SIZE - 1:
                self._save_state()
                self._cursor_col += 1
                self._update_cursor()
                if self._spend_move():
                    self.complete_action()
                    return
            self.complete_action()
            return

        self.complete_action()


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

    _VALID_ACTIONS: List[str] = list(_ACTION_MAP.keys())

    def __init__(self, seed: int = 0) -> None:
        self._engine = Cm10(seed, cursor_nonce=0)
        self.seed = seed
        self._total_turns = 0
        self._done = False
        self._game_over = False
        self._game_won = False
        self._last_action_was_reset = False
        self._pending_reset_metadata_info: Optional[Dict] = None
        self._reset_seq = 0

    def _episode_terminal(self) -> bool:
        g = self._engine
        return (
            self._game_won
            or g._game_won
            or g._current_level_index >= len(g._levels)
            or g._lives <= 0
        )

    def reset(self) -> GameState:
        e = self._engine
        self._reset_seq += 1
        self._total_turns = 0
        self._done = False
        self._game_over = False
        self._game_won = False
        game_won = e._game_won or e._current_level_index >= e.total_levels
        if game_won or e._move_count == 0 or self._last_action_was_reset:
            self._engine = Cm10(self.seed, cursor_nonce=self._reset_seq)
            e = self._engine
        else:
            e.perform_action(ActionInput(id=GameAction.RESET))
        self._last_action_was_reset = True
        pinfo = self._pending_reset_metadata_info
        self._pending_reset_metadata_info = None
        return self._build_game_state(step_info=pinfo)

    def get_actions(self) -> List[str]:
        if self._episode_terminal():
            return ["reset"]
        return list(self._VALID_ACTIONS)

    def step(self, action: str) -> StepResult:
        e = self._engine

        if action == "reset":
            game_won = e._game_won or e._current_level_index >= e.total_levels
            full_restart = game_won or e._move_count == 0 or self._last_action_was_reset
            si = {"action": "reset", "full_restart": full_restart}
            self._pending_reset_metadata_info = si
            state = self.reset()
            return StepResult(
                state=state,
                reward=0.0,
                done=False,
                info=si,
            )

        if action not in self._ACTION_MAP:
            raise ValueError(
                f"Invalid action '{action}'. Must be one of {sorted(self._ACTION_MAP.keys())}"
            )

        self._last_action_was_reset = False
        self._total_turns += 1

        lives_before = e._lives
        level_before = e.level_index

        ga = self._ACTION_MAP[action]
        action_input = ActionInput(id=ga)
        frame = e.perform_action(action_input, raw=True)
        st = getattr(frame, "state", None)
        state_name = st.name if st is not None else ""

        game_won = state_name == "WIN"
        game_over = state_name == "GAME_OVER"
        total_lv = len(e._levels)
        level_reward = 1.0 / total_lv

        info: Dict = {
            "lives": e._lives,
            "level": e.level_index + 1,
            "moves_used": e._move_count,
            "move_limit": e._max_moves,
        }

        if game_won:
            self._game_won = True
            self._done = True
            info["event"] = "game_complete"
            return StepResult(
                state=self._build_game_state(step_info=info),
                reward=level_reward,
                done=True,
                info=info,
            )

        if game_over:
            self._game_over = True
            self._done = True
            info["event"] = "game_over"
            return StepResult(
                state=self._build_game_state(step_info=info),
                reward=0.0,
                done=True,
                info=info,
            )

        reward = 0.0
        if e.level_index != level_before:
            reward = level_reward
            info["event"] = "level_complete"
        if e._lives < lives_before:
            info["event"] = "life_lost"

        self._done = self._episode_terminal()

        return StepResult(
            state=self._build_game_state(step_info=info),
            reward=reward,
            done=self._done,
            info=info,
        )

    def is_done(self) -> bool:
        return self._done or self._episode_terminal()

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
        rgb = _index_grid_to_rgb_64(arr)
        h, w, _ = rgb.shape
        raw = bytearray()
        for y in range(h):
            raw.append(0)
            raw.extend(rgb[y].tobytes())
        compressed = zlib.compress(bytes(raw))

        def _chunk(ctype: bytes, data: bytes) -> bytes:
            c = ctype + data
            return (
                struct.pack(">I", len(data))
                + c
                + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
            )

        sig = b"\x89PNG\r\n\x1a\n"
        ihdr = _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        idat = _chunk(b"IDAT", compressed)
        iend = _chunk(b"IEND", b"")
        return sig + ihdr + idat + iend

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError(f"Unsupported render mode: {mode}")
        e = self._engine
        index_grid = np.asarray(
            e.camera.render(e.current_level.get_sprites()), dtype=np.uint8
        )
        return _index_grid_to_rgb_64(index_grid)

    def close(self) -> None:
        self._engine = None

    def _build_game_state(self, step_info: Optional[Dict] = None) -> GameState:
        e = self._engine
        body = self._build_text_obs()
        text_observation = f"Agent turn: {self._total_turns}\n\n{body}"
        done = self._done or self._episode_terminal()
        valid_actions = self.get_actions() if not done else None
        return GameState(
            text_observation=text_observation,
            image_observation=self._build_image_bytes(),
            valid_actions=valid_actions,
            turn=self._total_turns,
            metadata={
                "total_levels": len(e._levels),
                "level_index": e.level_index,
                "levels_completed": getattr(e, "_score", 0),
                "game_over": getattr(getattr(e, "_state", None), "name", "")
                == "GAME_OVER",
                "done": done,
                "info": dict(step_info) if step_info else {},
            },
        )

    def _build_text_obs(self) -> str:
        g = self._engine
        info = g._analyze_graph()
        lines = []
        lines.append(
            f"LEVEL {g._current_level_index + 1}/{g.total_levels} "
            f"LIVES {g._lives}/3 "
            f"MOVES {g._move_count}/{g._max_moves}"
        )
        lines.append(
            f"BRIDGES {len(g._active_edges)}/{len(g._continents) - 1} "
            f"CONNECTED:{info['connected']} OBJECTIVE_MET:{info['objective_met']}"
        )
        lines.append(f"CURSOR ({g._cursor_col},{g._cursor_row})")
        lines.append(f"BAD_EDGES {sorted(info['bad_edges'])} FALSE_ON {sorted(info['bad_false'])}")
        return "\n".join(lines)


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
                base = a.split()[0] if " " in a else a
                idx = self._string_to_action.get(base)
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
