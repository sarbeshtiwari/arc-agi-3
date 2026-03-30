import random
import struct
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from arcengine import (
    ActionInput,
    ARCBaseGame,
    BlockingMode,
    Camera,
    GameAction,
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


EMPTY = 0
BRIDGE_BG = 4
CURSOR_CLR = 3
BLACK = 5

PATH_PALETTE = [
    8,
    9,
    11,
    14,
    12,
    15,
    6,
    10,
    13,
    7,
    1,
    2,
]

DIRS = [(0, -1), (0, 1), (-1, 0), (1, 0)]

LEVEL_CONFIGS = [
    {"grid": (8, 8), "colors": 5, "bridges": 0, "max_steps": 200},
    {"grid": (16, 16), "colors": 6, "bridges": 0, "max_steps": 600},
    {"grid": (10, 9), "colors": 7, "bridges": 1, "max_steps": 400},
    {"grid": (16, 16), "colors": 9, "bridges": 1, "max_steps": 1000},
    {"grid": (32, 32), "colors": 10, "bridges": 2, "max_steps": 2000},
]

PUZZLES = {
    0: {
        "endpoints": {
            1: ((6, 6), (0, 4)),
            2: ((0, 3), (2, 0)),
            3: ((3, 0), (5, 3)),
            4: ((5, 4), (3, 4)),
            5: ((3, 5), (2, 1)),
        },
        "bridges": set(),
    },
    1: {
        "endpoints": {
            1: ((2, 13), (15, 14)),
            2: ((14, 14), (14, 12)),
            3: ((14, 11), (13, 11)),
            4: ((13, 12), (8, 12)),
            5: ((9, 12), (10, 3)),
            6: ((9, 3), (4, 8)),
        },
        "bridges": set(),
    },
    2: {
        "endpoints": {
            4: ((0, 0), (9, 8)),
            7: ((9, 0), (0, 8)),
            1: ((7, 1), (9, 2)),
            2: ((2, 1), (4, 2)),
            3: ((4, 4), (0, 7)),
            5: ((8, 4), (7, 7)),
        },
        "bridges": {(5, 3)},
    },
    3: {
        "endpoints": {
            1: ((2, 0), (0, 3)),
            2: ((5, 1), (14, 4)),
            3: ((5, 2), (7, 13)),
            4: ((0, 14), (15, 8)),
            5: ((0, 15), (13, 4)),
            6: ((8, 5), (14, 7)),
            7: ((9, 9), (15, 11)),
            8: ((7, 8), (11, 14)),
            9: ((12, 12), (15, 14)),
        },
        "bridges": {(6, 7)},
    },
    4: {
        "endpoints": {
            1: ((0, 0), (31, 31)),
            2: ((0, 30), (31, 3)),
            3: ((9, 24), (0, 29)),
            4: ((11, 9), (23, 10)),
            5: ((29, 14), (30, 24)),
            6: ((5, 26), (13, 20)),
            7: ((2, 15), (7, 8)),
            8: ((9, 14), (28, 1)),
            9: ((16, 26), (2, 30)),
            10: ((19, 15), (15, 15)),
        },
        "bridges": {(10, 17), (17, 17)},
    },
}


class GameOverlay(RenderableUserDisplay):
    def __init__(self):
        self.cursor_x = 0
        self.cursor_y = 0
        self.grid_w = 8
        self.grid_h = 8
        self.endpoints = {}
        self.selected_color = 0
        self.bridges = set()
        self.drawn_paths = {}
        self.step_count = 0
        self.max_steps = 300

    def _scale_info(self):
        s = min(64 // self.grid_w, 64 // self.grid_h)
        ox = (64 - self.grid_w * s) // 2
        oy = (64 - self.grid_h * s) // 2
        return s, ox, oy

    def _set_px(self, frame, x, y, color):
        if 0 <= x < 64 and 0 <= y < 64:
            frame[y, x] = color

    def render_interface(self, frame):
        s, ox, oy = self._scale_info()
        self._draw_endpoint_dots(frame, s, ox, oy)
        self._draw_bridge_marks(frame, s, ox, oy)
        self._draw_cursor(frame, s, ox, oy)
        self._draw_progress_bar(frame)
        return frame

    def _draw_progress_bar(self, frame):
        if self.max_steps <= 0:
            return
        fraction = max(0.0, 1.0 - self.step_count / self.max_steps)
        filled = int(round(fraction * 64))
        for col in range(64):
            self._set_px(frame, col, 63, 14 if col < filled else 4)

    def _draw_endpoint_dots(self, frame, s, ox, oy):
        for color_id, (ep1, ep2) in self.endpoints.items():
            for ex, ey in (ep1, ep2):
                sx = ox + ex * s
                sy = oy + ey * s
                self._set_px(frame, sx + s // 2, sy + s // 2, BLACK)

    def _draw_bridge_marks(self, frame, s, ox, oy):
        if s < 4:
            for bx, by in self.bridges:
                sx = ox + bx * s
                sy = oy + by * s
                self._set_px(frame, sx + s // 2, sy + s // 2, BRIDGE_BG)
            return
        for bx, by in self.bridges:
            sx = ox + bx * s
            sy = oy + by * s
            mark = 2 if s < 8 else 3
            for i in range(mark):
                self._set_px(frame, sx + 1 + i, sy + 1, BRIDGE_BG)
                self._set_px(frame, sx + 1, sy + 1 + i, BRIDGE_BG)
                self._set_px(frame, sx + s - 2 - i, sy + 1, BRIDGE_BG)
                self._set_px(frame, sx + s - 2, sy + 1 + i, BRIDGE_BG)
                self._set_px(frame, sx + 1 + i, sy + s - 2, BRIDGE_BG)
                self._set_px(frame, sx + 1, sy + s - 2 - i, BRIDGE_BG)
                self._set_px(frame, sx + s - 2 - i, sy + s - 2, BRIDGE_BG)
                self._set_px(frame, sx + s - 2, sy + s - 2 - i, BRIDGE_BG)

    def _draw_cursor(self, frame, s, ox, oy):
        sx = ox + self.cursor_x * s
        sy = oy + self.cursor_y * s
        clr = (
            PATH_PALETTE[self.selected_color - 1]
            if self.selected_color > 0
            else CURSOR_CLR
        )
        for i in range(s):
            self._set_px(frame, sx + i, sy, clr)
            self._set_px(frame, sx + i, sy + s - 1, clr)
            self._set_px(frame, sx, sy + i, clr)
            self._set_px(frame, sx + s - 1, sy + i, clr)


class Cd01(ARCBaseGame):
    def __init__(self, seed: int = 0):
        self._rng = random.Random(seed)
        self.overlay = GameOverlay()

        self.grid_w = 8
        self.grid_h = 8
        self.grid = []
        self.endpoints = {}
        self.bridges = set()
        self.cursor_x = 0
        self.cursor_y = 0
        self.selected_color = 0
        self.drawn_paths = {}
        self.board_sprite = None
        self.step_count = 0
        self.max_steps = 150
        self._game_over = False

        levels = [
            Level(sprites=[], grid_size=cfg["grid"], name=f"Level {i + 1}")
            for i, cfg in enumerate(LEVEL_CONFIGS)
        ]

        camera = Camera(
            background=EMPTY,
            letter_box=BLACK,
            width=LEVEL_CONFIGS[0]["grid"][0],
            height=LEVEL_CONFIGS[0]["grid"][1],
            interfaces=[self.overlay],
        )

        super().__init__(
            game_id="cd01",
            levels=levels,
            camera=camera,
            available_actions=[0, 1, 2, 3, 4, 5, 7],
            seed=seed,
        )

    def on_set_level(self, level):
        cfg = LEVEL_CONFIGS[self.level_index]
        gw, gh = cfg["grid"]
        self.grid_w, self.grid_h = gw, gh

        pdata = PUZZLES[self.level_index]
        self.endpoints = pdata["endpoints"]
        self.bridges = pdata["bridges"]

        self.grid = [[0] * gw for _ in range(gh)]

        for cid, (ep1, ep2) in self.endpoints.items():
            self.grid[ep1[1]][ep1[0]] = cid
            self.grid[ep2[1]][ep2[0]] = cid

        self.selected_color = 0
        self.drawn_paths = {cid: [] for cid in self.endpoints}

        first_ep = self.endpoints[1][0]
        self.cursor_x, self.cursor_y = first_ep

        self.max_steps = cfg.get("max_steps", 300)
        self.step_count = 0
        self._game_over = False

        self.board_sprite = self._make_board_sprite()
        level.add_sprite(self.board_sprite)
        self._sync_overlay()

    def step(self):
        self.step_count += 1
        aid = self.action.id

        if aid in (
            GameAction.ACTION1,
            GameAction.ACTION2,
            GameAction.ACTION3,
            GameAction.ACTION4,
        ):
            dx, dy = {
                GameAction.ACTION1: (0, -1),
                GameAction.ACTION2: (0, 1),
                GameAction.ACTION3: (-1, 0),
                GameAction.ACTION4: (1, 0),
            }[aid]
            if self.selected_color > 0:
                self._extend_path(dx, dy)
            else:
                self._move_cursor(dx, dy)
        elif aid == GameAction.ACTION5:
            self._handle_select()
        elif aid == GameAction.ACTION7:
            self._handle_undo()

        self._refresh_board()
        self._sync_overlay()

        if self._check_win():
            self.next_level()
        elif self.step_count >= self.max_steps:
            self._game_over = True
            self.lose()

        self.complete_action()

    def _move_cursor(self, dx, dy):
        nx = self.cursor_x + dx
        ny = self.cursor_y + dy
        if 0 <= nx < self.grid_w and 0 <= ny < self.grid_h:
            self.cursor_x, self.cursor_y = nx, ny

    def _handle_select(self):
        pos = (self.cursor_x, self.cursor_y)
        if self.selected_color > 0:
            self.selected_color = 0
            return
        for cid, (ep1, ep2) in self.endpoints.items():
            if pos in (ep1, ep2):
                self._clear_drawn_path(cid)
                self.selected_color = cid
                self.drawn_paths[cid] = [pos]
                return

    def _handle_undo(self):
        if self.selected_color <= 0:
            return
        path = self.drawn_paths.get(self.selected_color, [])
        if len(path) <= 1:
            return
        self._undo_last_cell(self.selected_color)
        path = self.drawn_paths.get(self.selected_color, [])
        if path:
            self.cursor_x, self.cursor_y = path[-1]

    def _extend_path(self, dx, dy):
        nx = self.cursor_x + dx
        ny = self.cursor_y + dy
        if not (0 <= nx < self.grid_w and 0 <= ny < self.grid_h):
            return

        color = self.selected_color
        path = self.drawn_paths.get(color)
        if not path:
            return

        target = (nx, ny)

        if len(path) >= 2 and target == path[-2]:
            self._undo_last_cell(color)
            self.cursor_x, self.cursor_y = nx, ny
            return

        if target in self._path_set(color):
            return

        ep1, ep2 = self.endpoints[color]
        start_ep = path[0]
        other_ep = ep2 if start_ep == ep1 else ep1

        if target == other_ep:
            path.append(target)
            self.cursor_x, self.cursor_y = nx, ny
            self.selected_color = 0
            return

        if self.grid[ny][nx] == 0:
            path.append(target)
            self.grid[ny][nx] = color
            self.cursor_x, self.cursor_y = nx, ny
            return

        if target in self.bridges and self.grid[ny][nx] != color:
            path.append(target)
            self.cursor_x, self.cursor_y = nx, ny
            return

    def _undo_last_cell(self, color):
        path = self.drawn_paths[color]
        if not path:
            return
        removed = path.pop()
        rx, ry = removed

        ep1, ep2 = self.endpoints[color]
        if removed in (ep1, ep2):
            return

        if self.grid[ry][rx] == color:
            other = self._other_occupant(removed, color)
            self.grid[ry][rx] = other

    def _clear_drawn_path(self, color):
        ep1, ep2 = self.endpoints[color]
        old_path = self.drawn_paths.get(color, [])
        for pos in old_path:
            if pos in (ep1, ep2):
                continue
            px, py = pos
            if self.grid[py][px] == color:
                other = self._other_occupant(pos, color)
                self.grid[py][px] = other
        self.drawn_paths[color] = []

    def _path_set(self, color):
        return set(self.drawn_paths.get(color, []))

    def _other_occupant(self, pos, exclude_color):
        for cid, p in self.drawn_paths.items():
            if cid != exclude_color and pos in p:
                return cid
        return 0

    def _check_win(self):
        covered = set()
        for cid, (ep1, ep2) in self.endpoints.items():
            covered.add(ep1)
            covered.add(ep2)
        for path in self.drawn_paths.values():
            covered.update(path)

        if len(covered) < self.grid_w * self.grid_h:
            return False

        for cid, (ep1, ep2) in self.endpoints.items():
            if not self._path_connects(cid, ep1, ep2):
                return False

        return True

    def _path_connects(self, color, start, end):
        cells = self._path_set(color)
        cells.add(start)
        cells.add(end)

        visited = {start}
        queue = [start]
        while queue:
            pos = queue.pop(0)
            if pos == end:
                return True
            x, y = pos
            for dx, dy in DIRS:
                nb = (x + dx, y + dy)
                if nb not in visited and nb in cells:
                    visited.add(nb)
                    queue.append(nb)
        return False

    def _make_board_sprite(self):
        pixels = []
        for y in range(self.grid_h):
            row = []
            for x in range(self.grid_w):
                cid = self.grid[y][x]
                if cid > 0:
                    row.append(PATH_PALETTE[cid - 1])
                elif (x, y) in self.bridges:
                    row.append(BRIDGE_BG)
                else:
                    row.append(EMPTY)
            pixels.append(row)
        return Sprite(
            pixels=pixels,
            name="board",
            x=0,
            y=0,
            layer=0,
            blocking=BlockingMode.NOT_BLOCKED,
            collidable=False,
        )

    def _refresh_board(self):
        if self.board_sprite is None:
            return
        for y in range(self.grid_h):
            for x in range(self.grid_w):
                cid = self.grid[y][x]
                if cid > 0:
                    self.board_sprite.pixels[y, x] = PATH_PALETTE[cid - 1]
                elif (x, y) in self.bridges:
                    self.board_sprite.pixels[y, x] = BRIDGE_BG
                else:
                    self.board_sprite.pixels[y, x] = EMPTY

    def _sync_overlay(self):
        self.overlay.cursor_x = self.cursor_x
        self.overlay.cursor_y = self.cursor_y
        self.overlay.grid_w = self.grid_w
        self.overlay.grid_h = self.grid_h
        self.overlay.endpoints = self.endpoints
        self.overlay.selected_color = self.selected_color
        self.overlay.bridges = self.bridges
        self.overlay.drawn_paths = self.drawn_paths
        self.overlay.step_count = self.step_count
        self.overlay.max_steps = self.max_steps


class PuzzleEnvironment:
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

    ACTION_MAP = {
        "up": 1,
        "down": 2,
        "left": 3,
        "right": 4,
        "select": 5,
        "undo": 7,
        "reset": 0,
    }

    _GAME_ACTION_MAP = {
        1: GameAction.ACTION1,
        2: GameAction.ACTION2,
        3: GameAction.ACTION3,
        4: GameAction.ACTION4,
        5: GameAction.ACTION5,
        7: GameAction.ACTION7,
    }

    _COLOR_CHARS = "0123456789abc"

    def __init__(self, seed: int = 0) -> None:
        self._engine = Cd01(seed=seed)
        self._total_turns = 0
        self._last_action_was_reset = False
        self._prev_score = 0

    def _build_text_obs(self) -> str:
        e = self._engine
        chars = self._COLOR_CHARS

        rows = []
        for y in range(e.grid_h):
            row = []
            for x in range(e.grid_w):
                cid = e.grid[y][x]
                if (x, y) == (e.cursor_x, e.cursor_y):
                    row.append("*")
                elif (x, y) in e.bridges:
                    row.append("+")
                elif cid > 0:
                    row.append(chars[cid])
                else:
                    row.append(".")
            rows.append("".join(row))

        selected_str = "none" if e.selected_color == 0 else chars[e.selected_color]
        header = (
            f"Level:{e.level_index + 1} Steps:{e.step_count}/{e.max_steps} "
            f"Cursor:({e.cursor_x},{e.cursor_y}) Selected:{selected_str}"
        )

        ep_lines = []
        for cid, (ep1, ep2) in sorted(e.endpoints.items()):
            ch = chars[cid]
            ep_lines.append(f"  {ch}: ({ep1[0]},{ep1[1]})->({ep2[0]},{ep2[1]})")

        return header + "\n" + "\n".join(rows) + "\nEndpoints:\n" + "\n".join(ep_lines)

    @staticmethod
    def _encode_png(rgb: np.ndarray) -> bytes:
        h, w, _ = rgb.shape
        raw_rows = []
        for y in range(h):
            raw_rows.append(b"\x00" + rgb[y].tobytes())
        raw_data = b"".join(raw_rows)
        compressed = zlib.compress(raw_data)

        def _chunk(ctype: bytes, data: bytes) -> bytes:
            c = ctype + data
            return (
                struct.pack(">I", len(data))
                + c
                + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
            )

        ihdr_data = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
        png = b"\x89PNG\r\n\x1a\n"
        png += _chunk(b"IHDR", ihdr_data)
        png += _chunk(b"IDAT", compressed)
        png += _chunk(b"IEND", b"")
        return png

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
        return self._encode_png(rgb)

    def _build_game_state(
        self, done: bool = False, info: Optional[Dict] = None
    ) -> GameState:
        e = self._engine
        valid_actions = self.get_actions() if not done else None
        return GameState(
            text_observation=self._build_text_obs(),
            image_observation=self._build_image_bytes(),
            valid_actions=valid_actions,
            turn=self._total_turns,
            metadata={
                "total_levels": len(self._engine._levels),
                "level_index": e.level_index,
                "step_count": e.step_count,
                "max_steps": e.max_steps,
                "cursor_x": e.cursor_x,
                "cursor_y": e.cursor_y,
                "selected_color": e.selected_color,
                "game_over": e._game_over,
                "done": done,
                "info": info or {},
                "levels_completed": getattr(self._engine, "_score", 0),
            },
        )

    def reset(self) -> GameState:
        e = self._engine
        self._total_turns = 0
        game_won = hasattr(e, "_state") and getattr(e._state, "name", "") == "WIN"
        if game_won or self._last_action_was_reset:
            e.full_reset()
        else:
            e.level_reset()
        self._last_action_was_reset = True
        self._prev_score = e._score
        return self._build_game_state()

    def get_actions(self) -> List[str]:
        if self.is_done():
            return ["reset"]
        return ["reset", "up", "down", "left", "right", "select", "undo"]

    def is_done(self) -> bool:
        e = self._engine
        state_name = getattr(e._state, "name", "")
        return state_name in ("WIN", "GAME_OVER") or e._game_over

    def step(self, action: str) -> StepResult:
        e = self._engine

        if action == "reset":
            state = self.reset()
            return StepResult(
                state=state, reward=0.0, done=False, info={"action": "reset"}
            )

        if action not in self.ACTION_MAP:
            raise ValueError(
                f"Invalid action '{action}'. Must be one of {list(self.ACTION_MAP.keys())}"
            )

        self._last_action_was_reset = False
        self._total_turns += 1

        game_action_id = self.ACTION_MAP[action]
        game_action = self._GAME_ACTION_MAP[game_action_id]
        info: Dict = {"action": action}

        prev_score = e._score
        action_input = ActionInput(id=game_action)
        frame = e.perform_action(action_input, raw=True)

        game_won = frame and frame.state and frame.state.name == "WIN"
        level_completed = e._score > prev_score
        done = e._game_over or game_won

        total_levels = len(e._levels)
        reward = 1.0 / total_levels if level_completed else 0.0

        if done:
            if game_won:
                info["reason"] = "game_complete"
            else:
                info["reason"] = "game_over"
            return StepResult(
                state=self._build_game_state(done=True, info=info),
                reward=reward,
                done=True,
                info=info,
            )

        return StepResult(
            state=self._build_game_state(done=False, info=info),
            reward=reward,
            done=False,
            info=info,
        )

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError(f"Unsupported render mode: {mode}")
        e = self._engine
        index_grid = e.camera.render(e.current_level.get_sprites())
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
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

    ACTION_LIST: List[str] = ["reset", "up", "down", "left", "right", "select", "undo"]

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
        self._env: Optional[PuzzleEnvironment] = None

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
