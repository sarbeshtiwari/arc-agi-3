from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple, Union
from collections import deque

import random
import numpy as np
import zlib
import struct
import gymnasium as gym
from gymnasium import spaces
from arcengine import (
    ActionInput,
    ARCBaseGame,
    Camera,
    Level,
    RenderableUserDisplay,
    Sprite,
)
from arcengine.enums import GameState as EngineGameState, GameAction


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


C_EMPTY = 5
C_WALL = 4

MOVABLE_COLORS = {6, 7, 8, 9, 10, 11, 12, 14, 15}

BACKGROUND_COLOR = 5
PADDING_COLOR = 5

COLOR_NAMES = {
    C_EMPTY: ".",
    C_WALL: "W",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
    10: "A",
    11: "B",
    12: "C",
    14: "E",
    15: "F",
}

ARC_PALETTE = [
    (0, 0, 0),
    (0, 116, 217),
    (255, 65, 54),
    (46, 204, 64),
    (255, 220, 0),
    (170, 170, 170),
    (240, 18, 190),
    (255, 133, 27),
    (127, 219, 255),
    (135, 12, 37),
    (0, 48, 73),
    (106, 76, 48),
    (255, 182, 193),
    (80, 80, 80),
    (50, 205, 50),
    (128, 0, 128),
]

_CAMERA_SIZES = [8, 16, 32, 48, 64]


def _is_movable(v: int) -> bool:
    return v in MOVABLE_COLORS


def apply_gravity(grid: List[List[int]], direction: str) -> List[List[int]]:
    h = len(grid)
    w = len(grid[0]) if h > 0 else 0
    g = [row[:] for row in grid]

    if direction == 'down':
        for x in range(w):
            for y in range(h - 2, -1, -1):
                if _is_movable(g[y][x]):
                    ny = y
                    while ny + 1 < h and g[ny + 1][x] == C_EMPTY:
                        ny += 1
                    if ny != y:
                        g[ny][x] = g[y][x]
                        g[y][x] = C_EMPTY

    elif direction == 'up':
        for x in range(w):
            for y in range(1, h):
                if _is_movable(g[y][x]):
                    ny = y
                    while ny - 1 >= 0 and g[ny - 1][x] == C_EMPTY:
                        ny -= 1
                    if ny != y:
                        g[ny][x] = g[y][x]
                        g[y][x] = C_EMPTY

    elif direction == 'right':
        for y in range(h):
            for x in range(w - 2, -1, -1):
                if _is_movable(g[y][x]):
                    nx = x
                    while nx + 1 < w and g[y][nx + 1] == C_EMPTY:
                        nx += 1
                    if nx != x:
                        g[y][nx] = g[y][x]
                        g[y][x] = C_EMPTY

    elif direction == 'left':
        for y in range(h):
            for x in range(1, w):
                if _is_movable(g[y][x]):
                    nx = x
                    while nx - 1 >= 0 and g[y][nx - 1] == C_EMPTY:
                        nx -= 1
                    if nx != x:
                        g[y][nx] = g[y][x]
                        g[y][x] = C_EMPTY

    return g


def apply_merge(grid: List[List[int]]) -> List[List[int]]:
    h = len(grid)
    w = len(grid[0]) if h > 0 else 0
    g = [row[:] for row in grid]
    nbrs = [(0, -1), (0, 1), (-1, 0), (1, 0)]

    changed = True
    while changed:
        changed = False
        to_remove: set = set()
        visited = [[False] * w for _ in range(h)]

        for y in range(h):
            for x in range(w):
                if visited[y][x] or not _is_movable(g[y][x]):
                    continue
                color = g[y][x]
                component: List[Tuple[int, int]] = []
                queue: deque = deque()
                queue.append((x, y))
                visited[y][x] = True
                while queue:
                    cx, cy = queue.popleft()
                    component.append((cx, cy))
                    for dx, dy in nbrs:
                        nx2, ny2 = cx + dx, cy + dy
                        if 0 <= nx2 < w and 0 <= ny2 < h and not visited[ny2][nx2]:
                            if g[ny2][nx2] == color:
                                visited[ny2][nx2] = True
                                queue.append((nx2, ny2))
                if len(component) >= 2:
                    for cx, cy in component:
                        to_remove.add((cx, cy))

        if to_remove:
            changed = True
            for cx, cy in to_remove:
                g[cy][cx] = C_EMPTY

    return g


def _grid_cleared(grid: List[List[int]]) -> bool:
    for row in grid:
        for v in row:
            if _is_movable(v):
                return False
    return True


def count_blocks(grid: List[List[int]]) -> int:
    return sum(1 for row in grid for v in row if _is_movable(v))


def _make_level_1() -> List[List[int]]:
    W = C_WALL
    E = C_EMPTY
    return [
        [W, W, W, W],
        [W, 8, E, W],
        [W, W, E, W],
        [W, 8, E, W],
    ]


def _make_level_2() -> List[List[int]]:
    W = C_WALL
    E = C_EMPTY
    return [
        [W, W, W, W, W, W, W, W],
        [W, 8, E, E, W, E, E, W],
        [W, E, E, E, W, E, 9, W],
        [W, E, W, W, W, E, E, W],
        [W, E, E, E, E, E, E, W],
        [W, E, E, 8, E, W, W, W],
        [W, 9, E, E, E, W, E, W],
        [W, W, W, W, W, W, W, W],
    ]


def _make_level_3() -> List[List[int]]:
    W = C_WALL
    g = [[C_EMPTY] * 16 for _ in range(16)]
    for i in range(16):
        g[0][i] = W
        g[15][i] = W
        g[i][0] = W
        g[i][15] = W
    for x in range(6, 9):
        g[5][x] = W
        g[10][x] = W
    for y in range(6, 10):
        g[y][5] = W
        g[y][10] = W
    g[2][3] = 8
    g[2][12] = 8
    g[13][3] = 9
    g[13][12] = 9
    g[3][2] = 14
    g[12][2] = 14
    g[3][13] = 11
    g[12][13] = 11
    g[7][3] = 12
    g[7][12] = 12
    g[3][7] = 15
    g[12][7] = 15
    g[8][2] = 6
    g[8][13] = 6
    return g


def _make_level_4() -> List[List[int]]:
    W = C_WALL
    g = [[C_EMPTY] * 24 for _ in range(24)]
    for i in range(24):
        g[0][i] = W
        g[23][i] = W
        g[i][0] = W
        g[i][23] = W
    g[5][7] = W
    g[5][8] = W
    g[6][7] = W
    g[6][8] = W
    g[5][15] = W
    g[5][16] = W
    g[6][15] = W
    g[6][16] = W
    g[11][11] = W
    g[11][12] = W
    g[12][11] = W
    g[12][12] = W
    g[17][7] = W
    g[17][8] = W
    g[18][7] = W
    g[18][8] = W
    g[17][15] = W
    g[17][16] = W
    g[18][15] = W
    g[18][16] = W
    g[5][5] = 8
    g[5][10] = 8
    g[5][13] = 9
    g[5][18] = 9
    g[9][11] = 14
    g[14][11] = 14
    g[11][9] = 11
    g[11][14] = 11
    g[17][5] = 12
    g[17][10] = 12
    g[17][13] = 6
    g[17][18] = 6
    g[3][4] = 15
    g[20][4] = 15
    g[3][19] = 7
    g[20][19] = 7
    return g


def _make_level_5() -> List[List[int]]:
    W = C_WALL
    g = [[C_EMPTY] * 32 for _ in range(32)]
    for i in range(32):
        g[0][i] = W
        g[31][i] = W
        g[i][0] = W
        g[i][31] = W
    for x in range(10, 15):
        g[10][x] = W
        g[21][x] = W
    for x in range(17, 22):
        g[10][x] = W
        g[21][x] = W
    for y in range(10, 15):
        g[y][10] = W
        g[y][21] = W
    for y in range(17, 22):
        g[y][10] = W
        g[y][21] = W
    for y in range(11, 14):
        g[y][15] = W
    for y in range(17, 21):
        g[y][15] = W
    g[4][4] = W
    g[4][5] = W
    g[5][4] = W
    g[5][5] = W
    g[4][26] = W
    g[4][27] = W
    g[5][26] = W
    g[5][27] = W
    g[26][4] = W
    g[26][5] = W
    g[27][4] = W
    g[27][5] = W
    g[26][26] = W
    g[26][27] = W
    g[27][26] = W
    g[27][27] = W
    g[13][12] = 8
    g[18][12] = 8
    g[13][19] = 9
    g[18][19] = 9
    g[4][2] = 14
    g[4][8] = 14
    g[4][19] = 11
    g[4][25] = 11
    g[27][2] = 12
    g[27][8] = 12
    g[27][19] = 6
    g[27][25] = 6
    g[7][15] = 15
    g[12][15] = 15
    g[19][16] = 7
    g[24][16] = 7
    return g


LEVEL_CONFIGS = [
    {"name": "Routing Puzzle", "grid": _make_level_1, "size": 4, "max_moves": 48},
    {"name": "Wall Navigation", "grid": _make_level_2, "size": 8, "max_moves": 72},
    {"name": "Stepping Stones", "grid": _make_level_3, "size": 16, "max_moves": 108},
    {"name": "Scattered Isles", "grid": _make_level_4, "size": 24, "max_moves": 144},
    {"name": "The Fortress", "grid": _make_level_5, "size": 32, "max_moves": 180},
]

_sprite_cache: dict = {}


def _get_sprite_template(color: int) -> Sprite:
    if color not in _sprite_cache:
        tag = "wall" if color == C_WALL else "block"
        layer = 1 if color == C_WALL else 2
        _sprite_cache[color] = Sprite(
            pixels=[[color, color], [color, color]],
            name=f"cell_{color}",
            visible=True,
            collidable=(color == C_WALL),
            layer=layer,
            tags=[tag],
        )
    return _sprite_cache[color]


def _grid_to_sprites(grid: List[List[int]]) -> List[Sprite]:
    result = []
    for y, row in enumerate(grid):
        for x, val in enumerate(row):
            if val != C_EMPTY:
                sp = _get_sprite_template(val).clone().set_position(x * 2, y * 2)
                result.append(sp)
    return result


def _build_levels() -> List[Level]:
    result = []
    for idx, cfg in enumerate(LEVEL_CONFIGS):
        grid_data = cfg["grid"]()
        cam_size = _CAMERA_SIZES[idx]
        level = Level(
            sprites=_grid_to_sprites(grid_data),
            grid_size=(cam_size, cam_size),
            data={
                "grid": [row[:] for row in grid_data],
                "size": cfg["size"],
                "max_moves": cfg["max_moves"],
                "cam_size": cam_size,
            },
            name=cfg["name"],
        )
        result.append(level)
    return result


levels = _build_levels()


class GravityHUD(RenderableUserDisplay):
    MAX_LIVES = 3

    def __init__(self, max_moves: int):
        self.max_moves = max_moves
        self.moves_used = 0
        self.lives = 3

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if h < 4 or w < 12:
            return frame

        remaining = max(0, self.max_moves - self.moves_used)
        bar_max = min(w - 10, 50)
        if self.max_moves > 0:
            fill = max(0, int(bar_max * remaining / self.max_moves))
        else:
            fill = 0
        for bc in range(1, 1 + bar_max):
            if bc < w:
                frame[0, bc] = 3 if (bc - 1) < fill else 2

        for i in range(self.lives):
            c_off = w - 2 - (i * 2)
            if 0 <= c_off < w:
                frame[0, c_off] = 3

        return frame


class Gm01(ARCBaseGame):
    MAX_LIVES = 3

    def __init__(self, seed: int = 0) -> None:
        _sprite_cache.clear()
        self._seed = seed
        self._rng = random.Random(seed)
        self._grid: List[List[int]] = []
        self._grid_size: int = 4
        self._moves_used: int = 0
        self._max_moves: int = 0
        self._lives: int = self.MAX_LIVES
        self._game_over_pending: bool = False

        self._undo_stack: Deque[Dict] = deque(maxlen=50)
        self._last_level_won: bool = False
        self._consecutive_resets: int = 0
        self._actions_since_reset: int = 0

        first_data = LEVEL_CONFIGS[0]
        self._hud = GravityHUD(first_data["max_moves"])
        self._hud.lives = self._lives

        first_cam_size = _CAMERA_SIZES[0]
        cam = Camera(
            x=0,
            y=0,
            width=first_cam_size,
            height=first_cam_size,
            background=BACKGROUND_COLOR,
            letter_box=PADDING_COLOR,
            interfaces=[self._hud],
        )
        super().__init__(
            "gm01",
            levels,
            cam,
            available_actions=[0, 1, 2, 3, 4, 5, 7],
        )

    def on_set_level(self, level: Level) -> None:
        self._undo_stack = deque(maxlen=50)
        self._last_level_won = False
        self._load_level_data()

    def _load_level_data(self, level=None, restore_lives: bool = True):
        lv = level if level is not None else self.current_level
        grid_data = lv.get_data("grid")
        self._grid = [row[:] for row in grid_data]
        self._grid_size = lv.get_data("size")
        self._max_moves = lv.get_data("max_moves")
        self._moves_used = 0

        if restore_lives:
            self._lives = self.MAX_LIVES

        self._randomize_positions()

        self._hud.max_moves = self._max_moves
        self._hud.moves_used = 0
        self._hud.lives = self._lives
        self._undo_stack = deque(maxlen=50)

        cam_size = lv.get_data("cam_size")
        self.camera.width = cam_size
        self.camera.height = cam_size

        self._rebuild_sprites()

    def _randomize_positions(self) -> None:
        color_counts: Dict[int, int] = {}
        for r in range(len(self._grid)):
            for c in range(len(self._grid[r])):
                v = self._grid[r][c]
                if _is_movable(v):
                    color_counts[v] = color_counts.get(v, 0) + 1
                    self._grid[r][c] = C_EMPTY
        if not color_counts:
            return

        h = len(self._grid)
        w = len(self._grid[0]) if h else 0
        visited = [[False] * w for _ in range(h)]
        regions: List[List[Tuple[int, int]]] = []
        for r in range(h):
            for c in range(w):
                if not visited[r][c] and self._grid[r][c] == C_EMPTY:
                    region: List[Tuple[int, int]] = []
                    stack = [(r, c)]
                    visited[r][c] = True
                    while stack:
                        cr, cc = stack.pop()
                        region.append((cr, cc))
                        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc]:
                                if self._grid[nr][nc] == C_EMPTY:
                                    visited[nr][nc] = True
                                    stack.append((nr, nc))
                    if len(region) >= 2:
                        regions.append(region)

        if not regions:
            return

        colors_sorted = sorted(color_counts.keys())
        for color in colors_sorted:
            count = color_counts[color]
            valid_regions = [reg for reg in regions if len(reg) >= count]
            if not valid_regions:
                biggest = max(regions, key=len)
                count = min(count, len(biggest))
                chosen_region = biggest
            else:
                idx = self._rng.randint(0, len(valid_regions) - 1)
                chosen_region = valid_regions[idx]
            indices = self._rng.sample(range(len(chosen_region)), count)
            for i in range(count):
                r, c = chosen_region[indices[i]]
                self._grid[r][c] = color
            for i in sorted(indices, reverse=True):
                chosen_region.pop(i)
            regions = [reg for reg in regions if len(reg) >= 2]

    def _rebuild_sprites(self) -> None:
        self.current_level.remove_all_sprites()
        for y, row in enumerate(self._grid):
            for x, val in enumerate(row):
                if val != C_EMPTY:
                    sp = _get_sprite_template(val).clone().set_position(x * 2, y * 2)
                    self.current_level.add_sprite(sp)

    def _check_cleared(self) -> bool:
        return _grid_cleared(self._grid)

    def _handle_death(self) -> None:
        self._lives -= 1
        self._hud.lives = self._lives

        if self._lives <= 0:
            self._hud.lives = self._lives
            self._game_over_pending = True
            self._rebuild_sprites()
            return

        self._load_level_data(restore_lives=False)

    def handle_reset(self) -> None:
        self._game_over_pending = False
        if self._last_level_won or (self._consecutive_resets >= 1 and self._actions_since_reset == 0):
            self._consecutive_resets = 0
            self._actions_since_reset = 0
            self._last_level_won = False
            self._lives = self.MAX_LIVES
            self._hud.lives = self._lives
            self.full_reset()
        else:
            self._consecutive_resets += 1
            self._actions_since_reset = 0
            self._lives = self.MAX_LIVES
            self._hud.lives = self._lives
            self._load_level_data()

    def step(self) -> None:
        if self.action.id == GameAction.RESET:
            self.complete_action()
            return

        if self._game_over_pending:
            self.lose()
            self.complete_action()
            return

        try:
            aid = self.action.id
            if hasattr(aid, "value"):
                aid = aid.value
            aid = int(aid)
        except (ValueError, TypeError, AttributeError):
            aid = 0

        if aid == 0:
            self.complete_action()
            return

        moved = False
        new_grid = self._grid

        if aid == 7:
            if self._undo_stack:
                snapshot = self._undo_stack.pop()
                self._grid = snapshot["grid"]
                self._rebuild_sprites()
            self._moves_used += 1
            self._hud.moves_used = self._moves_used
            self._actions_since_reset += 1
            if self._moves_used >= self._max_moves:
                self._handle_death()
            if self._game_over_pending:
                self.lose()
            self.complete_action()
            return

        if aid in (1, 2, 3, 4, 5):
            snapshot = {
                "grid": [row[:] for row in self._grid],
                "moves_used": self._moves_used,
            }
            self._undo_stack.append(snapshot)

        if aid == 1:
            new_grid = apply_gravity(self._grid, 'up')
        elif aid == 2:
            new_grid = apply_gravity(self._grid, 'down')
        elif aid == 3:
            new_grid = apply_gravity(self._grid, 'left')
        elif aid == 4:
            new_grid = apply_gravity(self._grid, 'right')
        elif aid == 5:
            new_grid = apply_merge(self._grid)

        if new_grid != self._grid:
            self._grid = new_grid
            moved = True

        if aid in (1, 2, 3, 4, 5):
            self._moves_used += 1
            self._hud.moves_used = self._moves_used
            self._actions_since_reset += 1

        if moved:
            self._rebuild_sprites()

            if self._check_cleared():
                if self._current_level_index >= len(self._levels) - 1:
                    self._last_level_won = True
                self.next_level()
                self.complete_action()
                return

        if self._moves_used >= self._max_moves:
            self._handle_death()

        if self._game_over_pending:
            self.lose()

        self.complete_action()

    def render_text(self) -> str:
        rows = []
        for r in range(self._grid_size):
            row_chars = []
            for c in range(self._grid_size):
                row_chars.append(COLOR_NAMES.get(self._grid[r][c], "?"))
            rows.append(" ".join(row_chars))
        header = f"Level:{self._current_level_index + 1}/{len(self._levels)} Lives:{self._lives} Moves:{self._moves_used}/{self._max_moves}"
        controls = "Arrow Keys:Move Space:Action T:Undo R:Reset"
        return header + "\n" + "\n".join(rows) + "\n" + controls


_ACTION_MAP: Dict[str, GameAction] = {
    "reset": GameAction.RESET,
    "up": GameAction.ACTION1,
    "down": GameAction.ACTION2,
    "left": GameAction.ACTION3,
    "right": GameAction.ACTION4,
    "select": GameAction.ACTION5,
    "undo": GameAction.ACTION7,
}

_VALID_ACTIONS = ["reset","up", "down", "left", "right", "select", "undo"]


class PuzzleEnvironment:
    def __init__(self, seed: int = 0) -> None:
        self._engine = Gm01(seed=seed)
        self._total_turns: int = 0
        self._done: bool = False
        self._last_action_was_reset: bool = False

    def _is_won(self) -> bool:
        return self._engine._state == EngineGameState.WIN

    def _is_game_over(self) -> bool:
        return self._engine._state == EngineGameState.GAME_OVER

    def _build_text(self) -> str:
        return self._engine.render_text()

    def _build_image(self) -> Optional[bytes]:
        try:
            frame = self._engine.camera.render(self._engine.current_level.get_sprites())
            arr = frame.astype(np.uint8)
            if arr.ndim == 2:
                h, w = arr.shape
                color_type = 0
                raw_rows = b"".join(
                    b"\x00" + arr[y].tobytes() for y in range(h)
                )
            else:
                h, w = arr.shape[:2]
                color_type = 2
                raw_rows = b"".join(
                    b"\x00" + arr[y].tobytes() for y in range(h)
                )
            def _png_chunk(tag: bytes, data: bytes) -> bytes:
                chunk = tag + data
                return struct.pack(">I", len(data)) + chunk + struct.pack(">I", zlib.crc32(chunk) & 0xFFFFFFFF)
            ihdr_data = struct.pack(">IIBBBBB", w, h, 8, color_type, 0, 0, 0)
            png = b"\x89PNG\r\n\x1a\n"
            png += _png_chunk(b"IHDR", ihdr_data)
            png += _png_chunk(b"IDAT", zlib.compress(raw_rows))
            png += _png_chunk(b"IEND", b"")
            return png
        except Exception:
            return None

    def _build_state(self) -> GameState:
        text = self._build_text()
        img = self._build_image()
        if self._done:
            valid = None
        else:
            valid = self.get_actions()
        return GameState(
            text_observation=text,
            image_observation=img,
            valid_actions=valid,
            turn=self._total_turns,
            metadata={
                "total_levels": len(self._engine._levels),
                "current_level": self._engine._current_level_index + 1,
                "lives": self._engine._lives,
                "moves_used": self._engine._moves_used,
                "max_moves": self._engine._max_moves,
            },
        )

    def reset(self) -> GameState:
        if self._is_won() or self._last_action_was_reset:
            self._engine.full_reset()
        else:
            self._engine.perform_action(ActionInput(id=GameAction.RESET))
        self._last_action_was_reset = True
        self._done = False
        self._total_turns = 0
        return self._build_state()

    def get_actions(self) -> List[str]:
        if self._done:
            return ["reset"]
        return list(_VALID_ACTIONS)

    def step(self, action: str) -> StepResult:
        self._last_action_was_reset = False
        e = self._engine
        total_levels = len(self._engine._levels)

        if action == "reset":
            state = self.reset()
            return StepResult(
                state=state,
                reward=0.0,
                done=False,
                info={"event": "reset"},
            )

        ga = _ACTION_MAP.get(action)
        if ga is None:
            return StepResult(
                state=self._build_state(),
                reward=0.0,
                done=self._done,
                info={"error": "invalid_action"},
            )

        self._total_turns += 1
        info: Dict = {"action": action}

        level_before = e._current_level_index

        e.perform_action(ActionInput(id=ga))

        level_advanced = e._current_level_index != level_before or self._is_won()
        reward = (1.0 / total_levels) if level_advanced else 0.0

        if self._is_won():
            self._done = True
            info["reason"] = "game_complete"
            info["level_index"] = e._current_level_index
            info["total_levels"] = total_levels
            return StepResult(
                state=self._build_state(),
                reward=reward,
                done=True,
                info=info,
            )

        if self._is_game_over():
            self._done = True
            info["reason"] = "death"
            return StepResult(
                state=self._build_state(),
                reward=0.0,
                done=True,
                info=info,
            )

        return StepResult(
            state=self._build_state(),
            reward=reward,
            done=False,
            info=info,
        )

    def is_done(self) -> bool:
        return self._done

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError(f"Unsupported render mode: {mode}")
        index_grid = self._engine.camera.render(
            self._engine.current_level.get_sprites()
        )
        h, w = index_grid.shape[:2]
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for idx, color in enumerate(ARC_PALETTE):
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
