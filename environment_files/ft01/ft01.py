import copy
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
    RenderableUserDisplay,
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

C_WHITE    = 0
C_LTGREY   = 2
C_GREY     = 3
C_DKGREY   = 4
C_BLACK    = 5
C_MAGENTA  = 6
C_RED      = 8
C_BLUE     = 9
C_LTBLUE   = 10
C_YELLOW   = 11
C_ORANGE   = 12
C_GREEN    = 14

BG_COLOR  = C_BLACK
PAD_COLOR = C_BLACK

SIGNAL_COLORS = {
    0: C_BLACK,
    1: C_LTBLUE,
    2: C_BLUE,
    3: C_ORANGE,
    4: C_RED,
}

TILE_EMPTY     = 0
TILE_EMITTER   = 1
TILE_WALL      = 2

MAX_FREQ   = 4
MOVE_LIMIT = 200
MAX_LIVES  = 3
NUM_LEVELS = 4
HUD_ROWS   = 5


def _tile_scale(grid_w: int, grid_h: int) -> int:
    max_sc_x = 64 // grid_w
    max_sc_y = (64 - HUD_ROWS) // grid_h
    return max(1, min(max_sc_x, max_sc_y))


def _level_1() -> Dict:
    return {
        "grid_w": 10, "grid_h": 10,
        "emitters": [
            (2, 2, 0, False),
            (7, 2, 0, False),
            (4, 7, 0, False),
        ],
        "walls": [
            (4, 4), (5, 4), (4, 5), (5, 5),
        ],
        "links": {
            0: [(1, 1)],
            1: [(2, 1)],
            2: [],
        },
        "target_freqs": [3, 2, 4],
    }



def _level_2() -> Dict:
    return {
        "grid_w": 12, "grid_h": 12,
        "emitters": [
            (2, 2, 0, False),
            (9, 2, 0, False),
            (10, 7, 0, False),
            (6, 10, 0, False),
            (1, 7, 0, False),
        ],
        "walls": [
            (5, 4), (6, 4),
            (5, 7), (6, 7),
            (4, 5), (4, 6),
            (7, 5), (7, 6),
        ],
        "links": {
            0: [(1, 1)],
            1: [(2, 1)],
            2: [(3, 1)],
            3: [(4, 1)],
            4: [(0, 1)],
        },
        "target_freqs": [3, 2, 4, 1, 3],
    }


def _level_3() -> Dict:
    return {
        "grid_w": 12, "grid_h": 12,
        "emitters": [
            (1, 1, 0, False),
            (10, 1, 0, False),
            (5, 5, 0, False),
            (1, 10, 0, False),
            (10, 10, 2, True),
            (5, 10, 4, True),
        ],
        "walls": [
            (3, 3), (4, 3), (7, 3), (8, 3),
            (3, 7), (4, 7), (7, 7), (8, 7),
            (5, 3), (5, 7),
            (6, 3), (6, 7),
        ],
        "links": {
            0: [(2, 1)],
            1: [(2, -1)],
            2: [(0, 1), (1, 1)],
            3: [(2, -1)],
            4: [],
            5: [],
        },
        "target_freqs": [3, 4, 2, 3, 2, 4],
    }



def _level_4() -> Dict:
    return {
        "grid_w": 14, "grid_h": 13,
        "emitters": [
            (1, 1, 0, False),
            (7, 0, 0, False),
            (12, 1, 0, False),
            (1, 6, 4, True),
            (7, 6, 0, False),
            (12, 6, 0, False),
            (3, 11, 2, True),
            (10, 11, 3, True),
        ],
        "walls": [
            (4, 2), (5, 2), (9, 2), (10, 2),
            (4, 4), (5, 4), (9, 4), (10, 4),
            (4, 5), (4, 6), (4, 7), (4, 8),
            (9, 5), (9, 6), (9, 7), (9, 8),
            (5, 9), (6, 9), (8, 9), (9, 9),
            (6, 10), (8, 10),
        ],
        "links": {
            0: [(1, 1), (4, -1)],
            1: [(2, 1), (0, -1)],
            2: [(5, 1)],
            3: [],
            4: [(1, -1), (5, 1)],
            5: [(4, -1), (2, -1)],
            6: [],
            7: [],
        },
        "target_freqs": [3, 4, 2, 4, 1, 3, 2, 3],
    }


_LEVEL_BUILDERS = [_level_1, _level_2, _level_3, _level_4]

_anchor = Sprite(
    pixels=[[-1]], name="anchor",
    visible=False, collidable=False, tags=["anchor"], layer=0,
)


def _compute_signal_grid(grid_w: int, grid_h: int,
                         emitters: List[Dict],
                         walls: Set[Tuple[int, int]]) -> List[List[int]]:
    signal = [[0] * grid_w for _ in range(grid_h)]

    for em in emitters:
        freq = em["freq"]
        if freq <= 0:
            continue
        ex, ey = em["x"], em["y"]

        visited: Dict[Tuple[int, int], int] = {}
        visited[(ex, ey)] = freq
        queue = [(ex, ey, freq)]

        while queue:
            next_queue: List[Tuple[int, int, int]] = []
            for cx, cy, val in queue:
                if val <= 0:
                    continue
                for ddx, ddy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                    nx, ny = cx + ddx, cy + ddy
                    if not (0 <= nx < grid_w and 0 <= ny < grid_h):
                        continue
                    if (nx, ny) in walls:
                        continue
                    new_val = val - 1
                    if new_val <= 0:
                        continue
                    if (nx, ny) in visited and new_val <= visited[(nx, ny)]:
                        continue
                    visited[(nx, ny)] = new_val
                    next_queue.append((nx, ny, new_val))
            queue = next_queue

        for (sx, sy), val in visited.items():
            signal[sy][sx] += val

    for y in range(grid_h):
        for x in range(grid_w):
            signal[y][x] = max(0, min(MAX_FREQ, signal[y][x]))

    return signal


def _compute_target(level_data: Dict) -> List[List[int]]:
    gw, gh = level_data["grid_w"], level_data["grid_h"]
    walls_set = {(wx, wy) for wx, wy in level_data["walls"]}

    emitters = []
    for i, (ex, ey, _init, locked) in enumerate(level_data["emitters"]):
        emitters.append({
            "x": ex, "y": ey,
            "freq": level_data["target_freqs"][i],
            "locked": locked,
        })

    return _compute_signal_grid(gw, gh, emitters, walls_set)


class _Hud(RenderableUserDisplay):

    def __init__(self, game: "Ft01") -> None:
        self._g = game

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        g = self._g
        if g._tiles is None:
            return frame
        gw, gh = g._grid_w, g._grid_h
        sc = _tile_scale(gw, gh)

        frame[:, :] = BG_COLOR

        vw, vh = gw * sc, gh * sc
        ox = (64 - vw) // 2
        oy = max(0, (64 - HUD_ROWS - vh) // 2)

        for ty in range(gh):
            for tx in range(gw):
                px, py = ox + tx * sc, oy + ty * sc
                self._draw_tile(frame, tx, ty, px, py, sc, g)

        self._draw_links(frame, ox, oy, sc, g)

        cx, cy = g._cursor_x, g._cursor_y
        self._draw_cursor(frame, ox + cx * sc, oy + cy * sc, sc)

        self._draw_hud(frame, g, oy + vh)

        return frame

    def _draw_tile(self, f, tx, ty, px, py, sc, g):
        tile_type = g._tiles[ty][tx]
        current_sig = g._current_signal[ty][tx]
        target_sig = g._target[ty][tx]

        if tile_type == TILE_WALL:
            self._rect(f, px, py, sc, sc, C_DKGREY)
            return

        if current_sig == target_sig:
            bg = C_GREEN if target_sig > 0 else C_BLACK
        elif current_sig > target_sig:
            bg = C_RED
        else:
            bg = SIGNAL_COLORS.get(target_sig, C_DKGREY)
        self._rect(f, px, py, sc, sc, bg)

        if tile_type == TILE_EMITTER:
            self._draw_emitter(f, tx, ty, px, py, sc, g)
            return

        if current_sig != target_sig and target_sig > 0:
            self._draw_ghost(f, px, py, sc, target_sig)

    def _draw_emitter(self, f, tx, ty, px, py, sc, g):
        em = g._get_emitter_at(tx, ty)
        if em is None:
            return
        freq = em["freq"]
        locked = em["locked"]

        body_c = C_DKGREY if locked else C_YELLOW
        self._rect(f, px, py, sc, sc, body_c)

        if freq > 0:
            inner = sc - 2
            fill_h = max(1, (freq * inner + MAX_FREQ - 1) // MAX_FREQ)
            fill_y = py + sc - 1 - fill_h
            fill_c = C_WHITE if not locked else C_LTGREY
            self._rect(f, px + 1, fill_y, inner, fill_h, fill_c)

        if locked:
            for i in range(sc):
                self._px(f, px + i, py, C_RED)
                self._px(f, px + i, py + sc - 1, C_RED)
                self._px(f, px, py + i, C_RED)
                self._px(f, px + sc - 1, py + i, C_RED)

    def _draw_ghost(self, f, px, py, sc, target_sig):
        tc = SIGNAL_COLORS.get(target_sig, C_DKGREY)
        inner = sc - 2
        if inner >= 1:
            self._rect(f, px + 1, py + 1, inner, inner, tc)
        else:
            self._px(f, px, py, tc)

    def _draw_links(self, f, ox, oy, sc, g):
        half = sc // 2
        for src_idx, targets in g._links.items():
            if src_idx >= len(g._emitters):
                continue
            src = g._emitters[src_idx]
            sx = ox + src["x"] * sc + half
            sy = oy + src["y"] * sc + half
            for tgt_idx, effect in targets:
                if tgt_idx >= len(g._emitters):
                    continue
                tgt = g._emitters[tgt_idx]
                tx = ox + tgt["x"] * sc + half
                ty = oy + tgt["y"] * sc + half
                lc = C_GREEN if effect > 0 else C_MAGENTA
                self._draw_dotted_line(f, sx, sy, tx, ty, lc)

    def _draw_dotted_line(self, f, x0, y0, x1, y1, color):
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        steps = max(dx, dy, 1)
        for i in range(0, steps + 1, 2):
            t = i / steps if steps > 0 else 0
            px = int(x0 + t * (x1 - x0))
            py = int(y0 + t * (y1 - y0))
            self._px(f, px, py, color)

    def _draw_cursor(self, f, px, py, sc):
        for i in range(sc):
            self._px(f, px + i, py, C_WHITE)
            self._px(f, px + i, py + sc - 1, C_WHITE)
            self._px(f, px, py + i, C_WHITE)
            self._px(f, px + sc - 1, py + i, C_WHITE)

    def _draw_hud(self, f, g, hud_y):
        hy = min(hud_y + 1, 64 - HUD_ROWS)
        if hy >= 63:
            return

        f[hy, :] = C_GREY

        for i in range(MAX_LIVES):
            c = C_RED if i < g._lives else C_DKGREY
            self._rect(f, 1 + i * 4, hy + 2, 2, 2, c)

        mv_x = 16
        mv_w = 30
        mv_ratio = max(0, g._moves_remaining) / MOVE_LIMIT
        mv_fw = int(mv_ratio * mv_w)
        self._rect(f, mv_x, hy + 2, mv_w, 2, C_DKGREY)
        if mv_fw > 0:
            mc = C_GREEN if mv_ratio > 0.5 else (C_YELLOW if mv_ratio > 0.25 else C_RED)
            self._rect(f, mv_x, hy + 2, mv_fw, 2, mc)

    @staticmethod
    def _rect(f, x, y, w, h, c):
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(64, x + w), min(64, y + h)
        if x0 < x1 and y0 < y1:
            f[y0:y1, x0:x1] = c

    @staticmethod
    def _px(f, x, y, c):
        if 0 <= x < 64 and 0 <= y < 64:
            f[y, x] = c


class Ft01(ARCBaseGame):

    def __init__(self, seed: int = 0) -> None:
        self._hud = _Hud(self)
        self._lives: int = MAX_LIVES
        self._tiles: Optional[List[List[int]]] = None
        self._grid_w: int = 0
        self._grid_h: int = 0
        self._cursor_x: int = 0
        self._cursor_y: int = 0
        self._moves_remaining: int = MOVE_LIMIT
        self._emitters: List[Dict] = []
        self._walls: Set[Tuple[int, int]] = set()
        self._links: Dict[int, List[Tuple[int, int]]] = {}
        self._target: List[List[int]] = []
        self._current_signal: List[List[int]] = []
        self._current_level_idx: int = 0
        self._level_data: List[Dict] = [b() for b in _LEVEL_BUILDERS]
        self._saved_states: List[Optional[List[Dict]]] = [None] * NUM_LEVELS
        self._history: List[Dict[str, Any]] = []

        levels = [
            Level(
                sprites=[_anchor.clone().set_position(0, 0)],
                grid_size=(64, 64),
                data={"level_idx": i},
                name=f"Level {i + 1}",
            )
            for i in range(NUM_LEVELS)
        ]
        camera = Camera(0, 0, 64, 64, BG_COLOR, PAD_COLOR, [self._hud])
        super().__init__(
            "ft01", levels, camera,
            available_actions=[0, 1, 2, 3, 4, 5, 7],
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        idx = self.current_level.get_data("level_idx")
        self._current_level_idx = idx
        ld = self._level_data[idx]
        self._grid_w = ld["grid_w"]
        self._grid_h = ld["grid_h"]
        self._lives = MAX_LIVES
        self._moves_remaining = MOVE_LIMIT
        self._links = ld["links"]

        gw, gh = self._grid_w, self._grid_h
        self._tiles = [[TILE_EMPTY] * gw for _ in range(gh)]

        self._walls = set()
        for wx, wy in ld["walls"]:
            self._tiles[wy][wx] = TILE_WALL
            self._walls.add((wx, wy))

        self._emitters = []
        self._emitter_pos: Dict[Tuple[int, int], int] = {}
        for i, (ex, ey, init_freq, locked) in enumerate(ld["emitters"]):
            self._tiles[ey][ex] = TILE_EMITTER
            self._emitters.append({
                "x": ex, "y": ey,
                "freq": init_freq,
                "locked": locked,
                "init_freq": init_freq,
            })
            self._emitter_pos[(ex, ey)] = i

        self._target = _compute_target(ld)
        self._update_signal()
        self._saved_states[idx] = copy.deepcopy(self._emitters)
        self._history = []

        for em in self._emitters:
            if not em["locked"]:
                self._cursor_x = em["x"]
                self._cursor_y = em["y"]
                break
        else:
            self._cursor_x = 0
            self._cursor_y = 0

    def _update_signal(self) -> None:
        if self._tiles is None:
            return
        self._current_signal = _compute_signal_grid(
            self._grid_w, self._grid_h,
            self._emitters, self._walls,
        )

    def _get_emitter_at(self, x: int, y: int) -> Optional[Dict]:
        idx = self._emitter_pos.get((x, y), -1)
        return self._emitters[idx] if idx >= 0 else None

    def _get_emitter_index(self, x: int, y: int) -> int:
        return self._emitter_pos.get((x, y), -1)

    def _check_win(self) -> bool:
        if self._tiles is None:
            return False
        for y in range(self._grid_h):
            for x in range(self._grid_w):
                if self._tiles[y][x] == TILE_WALL:
                    continue
                if self._current_signal[y][x] != self._target[y][x]:
                    return False
        return True

    def _move_cursor(self, dx: int, dy: int) -> None:
        if self._tiles is None:
            return
        gw, gh = self._grid_w, self._grid_h
        orig_x, orig_y = self._cursor_x, self._cursor_y
        nx = (self._cursor_x + dx) % gw
        ny = (self._cursor_y + dy) % gh
        for _ in range(max(gw, gh)):
            if (nx, ny) not in self._walls:
                break
            nx = (nx + dx) % gw
            ny = (ny + dy) % gh
        else:
            nx, ny = orig_x, orig_y
        self._cursor_x = nx
        self._cursor_y = ny

    def _on_space(self) -> None:
        if self._tiles is None:
            return
        cx, cy = self._cursor_x, self._cursor_y
        em_idx = self._get_emitter_index(cx, cy)
        if em_idx < 0:
            return
        em = self._emitters[em_idx]
        if em["locked"]:
            return

        em["freq"] = (em["freq"] + 1) % (MAX_FREQ + 1)

        for tgt_idx, effect in self._links.get(em_idx, []):
            if tgt_idx < 0 or tgt_idx >= len(self._emitters):
                continue
            tgt = self._emitters[tgt_idx]
            if tgt["locked"]:
                continue
            tgt["freq"] = (tgt["freq"] + effect) % (MAX_FREQ + 1)

        self._update_signal()

        if self._check_win():
            self.next_level()

    def _fail(self) -> None:
        self._lives -= 1
        if self._lives <= 0:
            self.lose()
            return
        idx = self._current_level_idx
        saved = self._saved_states[idx]
        if saved is not None:
            self._emitters = copy.deepcopy(saved)
        self._moves_remaining = MOVE_LIMIT
        self._update_signal()
        self._history = []
        for em in self._emitters:
            if not em["locked"]:
                self._cursor_x = em["x"]
                self._cursor_y = em["y"]
                break

    def _reset_level_state(self) -> None:
        self._lives = MAX_LIVES
        idx = self._current_level_idx
        saved = self._saved_states[idx]
        if saved is not None:
            self._emitters = copy.deepcopy(saved)
        self._moves_remaining = MOVE_LIMIT
        self._update_signal()
        self._history = []
        for em in self._emitters:
            if not em["locked"]:
                self._cursor_x = em["x"]
                self._cursor_y = em["y"]
                break

    def _save_state(self) -> Dict[str, Any]:
        return {
            "emitters": copy.deepcopy(self._emitters),
            "cursor_x": self._cursor_x,
            "cursor_y": self._cursor_y,
            "lives": self._lives,
        }

    def _restore_state(self, snapshot: Dict[str, Any]) -> None:
        self._emitters = copy.deepcopy(snapshot["emitters"])
        self._cursor_x = snapshot["cursor_x"]
        self._cursor_y = snapshot["cursor_y"]
        self._lives = snapshot["lives"]
        self._update_signal()

    def step(self) -> None:
        aid = self.action.id

        if aid == GameAction.ACTION7:
            if self._history:
                self._restore_state(self._history.pop())
            self._moves_remaining -= 1
            if self._moves_remaining <= 0:
                self._fail()
            self.complete_action()
            return

        self._history.append(self._save_state())

        if aid == GameAction.ACTION1:
            self._move_cursor(0, -1)
        elif aid == GameAction.ACTION2:
            self._move_cursor(0, 1)
        elif aid == GameAction.ACTION3:
            self._move_cursor(-1, 0)
        elif aid == GameAction.ACTION4:
            self._move_cursor(1, 0)
        elif aid == GameAction.ACTION5:
            self._on_space()
        self._moves_remaining -= 1
        if self._moves_remaining <= 0:
            self._fail()
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
        self._engine = Ft01(seed=seed)
        self._total_turns = 0
        self._done = False
        self._game_won = False
        self._game_over = False
        self._last_action_was_reset = False
        self._levels_completed = 0

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
        self._game_over = False
        self._levels_completed = 0
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
            return StepResult(
                state=self._build_game_state(), reward=0.0, done=self._done,
                info={"action": action, "reason": "invalid_action"},
            )

        self._last_action_was_reset = False
        self._total_turns += 1

        game_action = self._ACTION_MAP[action]
        info: Dict = {"action": action}

        level_before = e.level_index

        action_input = ActionInput(id=game_action)
        frame = e.perform_action(action_input, raw=True)

        state_name = frame.state.name if frame and frame.state else ""
        game_won = state_name == "WIN"
        game_over = state_name == "GAME_OVER"

        total_levels = len(e._levels)
        level_reward = 1.0 / total_levels

        if game_won:
            self._done = True
            self._game_won = True
            self._game_over = False
            self._levels_completed += 1
            info["reason"] = "game_complete"
            return StepResult(
                state=self._build_game_state(done=True),
                reward=level_reward, done=True, info=info,
            )

        if game_over:
            self._done = True
            self._game_over = True
            info["reason"] = "game_over"
            return StepResult(
                state=self._build_game_state(done=True),
                reward=0.0, done=True, info=info,
            )

        reward = 0.0
        if e.level_index != level_before:
            reward = level_reward
            self._levels_completed += 1
            info["reason"] = "level_complete"

        return StepResult(
            state=self._build_game_state(done=False),
            reward=reward, done=False, info=info,
        )

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

    def _build_text_obs(self) -> str:
        e = self._engine
        lines = []
        lines.append(f"Level {e._current_level_idx + 1}/{NUM_LEVELS} | Lives {e._lives}/{MAX_LIVES} | Moves {e._moves_remaining}/{MOVE_LIMIT}")
        lines.append(f"Cursor ({e._cursor_x},{e._cursor_y})")
        gw, gh = e._grid_w, e._grid_h
        if e._tiles is not None:
            for y in range(gh):
                row = []
                for x in range(gw):
                    if (x, y) in e._walls:
                        row.append("#")
                    elif e._get_emitter_index(x, y) >= 0:
                        em = e._get_emitter_at(x, y)
                        if em["locked"]:
                            row.append(f"L{em['freq']}")
                        else:
                            row.append(f"E{em['freq']}")
                    else:
                        cs = e._current_signal[y][x]
                        ts = e._target[y][x]
                        if cs == ts:
                            row.append(".")
                        else:
                            row.append(f"{cs}/{ts}")
                lines.append(" ".join(row))
        return "\n".join(lines)

    def _build_game_state(self, done: bool = False) -> GameState:
        e = self._engine
        valid_actions = self.get_actions() if not done else None
        return GameState(
            text_observation=self._build_text_obs(),
            image_observation=None,
            valid_actions=valid_actions,
            turn=self._total_turns,
            metadata={
                "total_levels": len(e._levels),
                "level_index": e.level_index,
                "levels_completed": self._levels_completed,
                "game_over": self._game_over,
                "done": done,
                "info": {},
            },
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
            low=0, high=255,
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
        self, state: GameState, step_info: Optional[Dict] = None,
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

    env = ArcGameEnv(seed=0, render_mode="rgb_array")
    check_env(env.unwrapped, skip_render_check=True)

    obs, info = env.reset()
    mask = env.action_mask()
    valid_indices = np.where(mask == 1)[0]
    if len(valid_indices) > 0:
        obs, reward, term, trunc, info = env.step(valid_indices[0])

    env.close()
