import random
import struct
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from arcengine import (
    ActionInput,
    ARCBaseGame,
    Camera,
    GameAction,
    Level,
    RenderableUserDisplay,
    Sprite,
)
from arcengine.enums import BlockingMode


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

EMPTY = 4
WALL = 2
PLACED = 9
PRE_PLACED = 15
FLASH = 8
GLOW = 14
CURSOR_CLR = 11
GRID_CLR = 3
BLACK = 5
BAR_EMPTY = 4
LIFE_ACTIVE = 8
LIFE_LOST = 4
LIVES_PER_LEVEL = 3

LEVEL_CONFIGS = [
    {
        "grid": (5, 5),
        "required": 5,
        "walls": [(0, 4), (2, 2), (4, 0)],
        "pre_placed": [],
        "max_moves": 30,
    },
    {
        "grid": (6, 6),
        "required": 6,
        "walls": [(0, 5), (1, 1), (2, 3), (3, 2), (4, 4), (5, 0)],
        "pre_placed": [],
        "max_moves": 40,
    },
    {
        "grid": (7, 7),
        "required": 7,
        "walls": [(0, 0), (1, 5), (2, 2), (3, 0), (3, 3), (4, 4), (5, 1), (6, 6)],
        "pre_placed": [],
        "max_moves": 50,
    },
    {
        "grid": (8, 8),
        "required": 8,
        "walls": [
            (0, 0),
            (1, 1),
            (2, 5),
            (3, 3),
            (3, 6),
            (4, 4),
            (5, 2),
            (6, 3),
            (6, 6),
            (7, 7),
        ],
        "pre_placed": [(0, 3)],
        "max_moves": 60,
    },
    {
        "grid": (9, 9),
        "required": 9,
        "walls": [
            (0, 0),
            (0, 4),
            (1, 1),
            (2, 6),
            (3, 3),
            (4, 0),
            (4, 4),
            (5, 5),
            (6, 2),
            (7, 7),
            (8, 4),
            (8, 8),
        ],
        "pre_placed": [(1, 3), (5, 1)],
        "max_moves": 70,
    },
    {
        "grid": (10, 10),
        "required": 10,
        "walls": [
            (0, 0),
            (0, 9),
            (1, 1),
            (2, 7),
            (3, 6),
            (4, 4),
            (4, 8),
            (5, 0),
            (5, 5),
            (6, 3),
            (7, 2),
            (8, 4),
            (8, 8),
            (9, 0),
            (9, 9),
        ],
        "pre_placed": [(0, 3), (6, 0)],
        "max_moves": 90,
    },
]

NUM_LEVELS = len(LEVEL_CONFIGS)

_levels = [
    Level(
        sprites=[],
        grid_size=cfg["grid"],
        name=f"Level {i + 1}",
    )
    for i, cfg in enumerate(LEVEL_CONFIGS)
]


class NqOverlay(RenderableUserDisplay):

    def __init__(self):
        self.cursor_x = 0
        self.cursor_y = 0
        self.grid_w = 5
        self.grid_h = 5
        self.flash_timer = 0
        self.win_glow = False
        self.game_over = False
        self.placed_set = set()
        self.move_count = 0
        self.max_moves = 30
        self.lives = LIVES_PER_LEVEL

    def _scale_info(self):
        s = min(64 // self.grid_w, 64 // self.grid_h)
        ox = (64 - self.grid_w * s) // 2
        oy = (64 - self.grid_h * s) // 2
        return s, ox, oy

    def _set_px(self, frame, x, y, color):
        if 0 <= x < 64 and 0 <= y < 64:
            frame[y, x] = color

    def _draw_cell_border(self, frame, cx, cy, s, ox, oy, color):
        sx = ox + cx * s
        sy = oy + cy * s
        for i in range(s):
            self._set_px(frame, sx + i, sy, color)
            self._set_px(frame, sx + i, sy + s - 1, color)
            self._set_px(frame, sx, sy + i, color)
            self._set_px(frame, sx + s - 1, sy + i, color)

    def _draw_grid_lines(self, frame, s, ox, oy):
        gw = self.grid_w
        gh = self.grid_h
        total_w = gw * s
        total_h = gh * s

        for i in range(total_w):
            self._set_px(frame, ox + i, oy, GRID_CLR)
            self._set_px(frame, ox + i, oy + total_h - 1, GRID_CLR)
        for i in range(total_h):
            self._set_px(frame, ox, oy + i, GRID_CLR)
            self._set_px(frame, ox + total_w - 1, oy + i, GRID_CLR)

        for col in range(1, gw):
            x = ox + col * s
            for i in range(total_h):
                self._set_px(frame, x, oy + i, GRID_CLR)

        for row in range(1, gh):
            y = oy + row * s
            for i in range(total_w):
                self._set_px(frame, ox + i, y, GRID_CLR)

    def _draw_progress_bar(self, frame):
        if self.max_moves <= 0:
            return

        if self.game_over:
            for col in range(64):
                self._set_px(frame, col, 63, FLASH)
            return

        if self.win_glow:
            for col in range(64):
                self._set_px(frame, col, 63, GLOW)
            return

        fraction = max(0.0, 1.0 - self.move_count / self.max_moves)
        filled = int(round(fraction * 64))
        for col in range(64):
            if col < filled:
                self._set_px(frame, col, 63, GLOW)
            else:
                self._set_px(frame, col, 63, BAR_EMPTY)

    def _draw_lives(self, frame):
        for i in range(LIVES_PER_LEVEL):
            base_x = 56 + i * 3
            color = LIFE_ACTIVE if i < self.lives else LIFE_LOST
            for dy in range(2):
                for dx in range(2):
                    self._set_px(frame, base_x + dx, 61 + dy, color)

    def render_interface(self, frame):
        s, ox, oy = self._scale_info()

        self._draw_grid_lines(frame, s, ox, oy)

        if self.win_glow:
            for pcx, pcy in self.placed_set:
                self._draw_cell_border(frame, pcx, pcy, s, ox, oy, GLOW)
            self._draw_progress_bar(frame)
            self._draw_lives(frame)
            return frame

        if self.game_over:
            for pcx, pcy in self.placed_set:
                self._draw_cell_border(frame, pcx, pcy, s, ox, oy, FLASH)
            self._draw_progress_bar(frame)
            self._draw_lives(frame)
            return frame

        if self.flash_timer > 0:
            self._draw_cell_border(
                frame, self.cursor_x, self.cursor_y, s, ox, oy, FLASH
            )
        else:
            self._draw_cell_border(
                frame, self.cursor_x, self.cursor_y, s, ox, oy, CURSOR_CLR
            )

        self._draw_progress_bar(frame)
        self._draw_lives(frame)

        return frame


class Nq01(ARCBaseGame):

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)
        self.overlay = NqOverlay()

        self.grid_w: int = 5
        self.grid_h: int = 5
        self.cursor_x: int = 0
        self.cursor_y: int = 0
        self.placed: set = set()
        self.walls: set = set()
        self.pre_placed: set = set()
        self.required: int = 0
        self.flash_timer: int = 0
        self.board_sprite: Sprite | None = None
        self.move_count: int = 0
        self.max_moves: int = 30
        self._game_over: bool = False
        self.lives: int = LIVES_PER_LEVEL
        self._preserve_lives: bool = False
        self._history: list = []
        self._game_complete: bool = False
        self._last_action_was_reset: bool = False

        first_grid = LEVEL_CONFIGS[0]["grid"]
        camera = Camera(
            background=EMPTY,
            letter_box=BLACK,
            width=first_grid[0],
            height=first_grid[1],
            interfaces=[self.overlay],
        )

        local_levels = [
            Level(sprites=[], grid_size=lv.grid_size, name=lv.name)
            for lv in _levels
        ]

        super().__init__(
            game_id="nq01",
            levels=local_levels,
            camera=camera,
            available_actions=[0,1, 2, 3, 4, 5, 6, 7],
            seed=seed,
        )

    def full_reset(self) -> None:
        self._game_complete = False
        super().full_reset()

    def handle_reset(self) -> None:
        if self._game_complete:
            self.full_reset()
            return
        if self._game_over:
            self._game_over = False
            self.full_reset()
            return
        if self._last_action_was_reset:
            self.full_reset()
        else:
            self._preserve_lives = True
            self.level_reset()

    def on_set_level(self, level: Level) -> None:
        cfg = LEVEL_CONFIGS[self.level_index]
        gw, gh = cfg["grid"]
        self.grid_w = gw
        self.grid_h = gh
        self.camera.resize(gw, gh)

        self.walls = set(cfg["walls"])
        self.pre_placed = set(cfg["pre_placed"])
        self.placed = set(cfg["pre_placed"])
        self.required = cfg["required"]

        valid_starts = [
            (x, y)
            for x in range(gw)
            for y in range(gh)
            if (x, y) not in self.walls and (x, y) not in self.pre_placed
        ]
        if valid_starts:
            sx, sy = self._rng.choice(valid_starts)
            self.cursor_x = sx
            self.cursor_y = sy
        else:
            self.cursor_x = 0
            self.cursor_y = 0
        self.flash_timer = 0
        self.move_count = 0
        self.max_moves = cfg.get("max_moves", 100)
        self._game_over = False

        self._history = []

        if self._preserve_lives:
            self._preserve_lives = False
        else:
            self.lives = LIVES_PER_LEVEL

        self.board_sprite = self._make_board_sprite()
        level.add_sprite(self.board_sprite)
        self._sync_overlay()

    def step(self) -> None:
        aid = self.action.id

        if aid == GameAction.RESET:
            self.handle_reset()
            self._last_action_was_reset = True
            self.complete_action()
            return

        self._last_action_was_reset = False

        if self._game_over:
            self.complete_action()
            return

        if self.flash_timer > 0:
            self.flash_timer -= 1

        if aid == GameAction.ACTION1:
            self._move_cursor(0, -1)
        elif aid == GameAction.ACTION2:
            self._move_cursor(0, 1)
        elif aid == GameAction.ACTION3:
            self._move_cursor(-1, 0)
        elif aid == GameAction.ACTION4:
            self._move_cursor(1, 0)
        elif aid == GameAction.ACTION5:
            self._history.append(self._save_state())
            self._toggle_at(self.cursor_x, self.cursor_y)
        elif aid == GameAction.ACTION6:
            self._handle_click()
        elif aid == GameAction.ACTION7:
            self._undo()

        if aid in (GameAction.ACTION5, GameAction.ACTION6, GameAction.ACTION7):
            self.move_count += 1

        self._refresh_board()
        self._sync_overlay()

        if self._check_complete():
            self.overlay.win_glow = True
            self.overlay.placed_set = set(self.placed)
            if self.level_index >= NUM_LEVELS - 1:
                self._game_complete = True
            self.next_level()
            self.complete_action()
            return

        if self.move_count >= self.max_moves:
            self.lives -= 1
            if self.lives <= 0:
                self._game_over = True
                self.overlay.game_over = True
                self.overlay.lives = 0
                self.lose()
            else:
                self._preserve_lives = True
                self.level_reset()
                self._sync_overlay()
            self.complete_action()
            return

        self.complete_action()

    def _move_cursor(self, dx: int, dy: int) -> None:
        nx = self.cursor_x + dx
        ny = self.cursor_y + dy
        if 0 <= nx < self.grid_w and 0 <= ny < self.grid_h:
            self.cursor_x = nx
            self.cursor_y = ny

    def _handle_click(self) -> None:
        data = self.action.data if self.action and hasattr(self.action, "data") and self.action.data else {}
        px = data.get("x", 0)
        py = data.get("y", 0)
        s = min(64 // self.grid_w, 64 // self.grid_h)
        ox = (64 - self.grid_w * s) // 2
        oy = (64 - self.grid_h * s) // 2
        col = (px - ox) // s
        row = (py - oy) // s
        if not (0 <= col < self.grid_w and 0 <= row < self.grid_h):
            col = self.cursor_x
            row = self.cursor_y
        self.cursor_x = col
        self.cursor_y = row
        self._history.append(self._save_state())
        self._toggle_at(col, row)

    def _toggle_at(self, x: int, y: int) -> None:
        pos = (x, y)
        if pos in self.placed and pos not in self.pre_placed:
            self._remove(x, y)
        else:
            self._try_place(x, y)

    def _conflicts(self, x: int, y: int) -> bool:
        for bx, by in self.placed:
            if bx == x:
                return True
            if by == y:
                return True
            if abs(bx - x) == abs(by - y):
                return True
        return False

    def _try_place(self, x: int, y: int) -> None:
        pos = (x, y)
        if pos in self.walls:
            return
        if pos in self.placed:
            return
        if self._conflicts(x, y):
            self.flash_timer = 6
            return
        self.placed.add(pos)

    def _remove(self, x: int, y: int) -> None:
        pos = (x, y)
        if pos in self.placed and pos not in self.pre_placed:
            self.placed.discard(pos)

    def _save_state(self) -> dict:
        return {
            "placed": set(self.placed),
            "cursor_x": self.cursor_x,
            "cursor_y": self.cursor_y,
            "flash_timer": self.flash_timer,
        }

    def _undo(self) -> None:
        if not self._history:
            return
        snapshot = self._history.pop()
        self.placed = snapshot["placed"]
        self.cursor_x = snapshot["cursor_x"]
        self.cursor_y = snapshot["cursor_y"]
        self.flash_timer = snapshot["flash_timer"]

    def _check_complete(self) -> bool:
        return len(self.placed) >= self.required

    def _make_board_sprite(self) -> Sprite:
        pixels = []
        for y in range(self.grid_h):
            row = []
            for x in range(self.grid_w):
                row.append(self._cell_colour(x, y))
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

    def _cell_colour(self, x: int, y: int) -> int:
        pos = (x, y)
        if pos in self.placed:
            return PRE_PLACED if pos in self.pre_placed else PLACED
        if pos in self.walls:
            return WALL
        return EMPTY

    def _refresh_board(self) -> None:
        if self.board_sprite is None:
            return
        for y in range(self.grid_h):
            for x in range(self.grid_w):
                self.board_sprite.pixels[y, x] = self._cell_colour(x, y)

    def _sync_overlay(self) -> None:
        self.overlay.cursor_x = self.cursor_x
        self.overlay.cursor_y = self.cursor_y
        self.overlay.grid_w = self.grid_w
        self.overlay.grid_h = self.grid_h
        self.overlay.flash_timer = self.flash_timer
        self.overlay.win_glow = False
        self.overlay.game_over = self._game_over
        self.overlay.placed_set = set(self.placed)
        self.overlay.move_count = self.move_count
        self.overlay.max_moves = self.max_moves
        self.overlay.lives = self.lives



class PuzzleEnvironment:

    ACTION_MAP = {
        "reset": GameAction.RESET,
        "up": GameAction.ACTION1,
        "down": GameAction.ACTION2,
        "left": GameAction.ACTION3,
        "right": GameAction.ACTION4,
        "select": GameAction.ACTION5,
        "click": GameAction.ACTION6,
        "undo": GameAction.ACTION7,

    }

    ARC_PALETTE = [
        (255, 255, 255),
        (204, 204, 204),
        (153, 153, 153),
        (102, 102, 102),
        (51, 51, 51),
        (0, 0, 0),
        (229, 58, 163),
        (255, 123, 204),
        (249, 60, 49),
        (30, 147, 255),
        (136, 216, 241),
        (255, 220, 0),
        (255, 133, 27),
        (146, 18, 49),
        (79, 204, 48),
        (163, 86, 214),
    ]


    def __init__(self, seed: int = 0) -> None:
        self._engine = Nq01(seed=seed)
        self._total_turns = 0

    def _build_text_obs(self) -> str:
        e = self._engine
        gw, gh = e.grid_w, e.grid_h

        text_grid: List[List[str]] = [["." for _ in range(gw)] for _ in range(gh)]

        for wx, wy in e.walls:
            if 0 <= wx < gw and 0 <= wy < gh:
                text_grid[wy][wx] = "#"

        for px, py in e.placed:
            if 0 <= px < gw and 0 <= py < gh:
                if (px, py) in e.pre_placed:
                    text_grid[py][px] = "P"
                else:
                    text_grid[py][px] = "Q"

        cx, cy = e.cursor_x, e.cursor_y
        if 0 <= cx < gw and 0 <= cy < gh:
            if (cx, cy) in e.pre_placed:
                text_grid[cy][cx] = "%"
            elif (cx, cy) in e.placed:
                text_grid[cy][cx] = "*"
            else:
                text_grid[cy][cx] = "@"

        grid_text = "\n".join("".join(row) for row in text_grid)

        remaining = max(0, e.max_moves - e.move_count)
        header = (
            f"Level:{e.level_index + 1} Lives:{e.lives} "
            f"Moves:{remaining}/{e.max_moves} "
            f"Queens:{len(e.placed)}/{e.required}"
        )

        return header.strip() + "\n" + grid_text

    @staticmethod
    def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
        c = chunk_type + data
        crc = struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        return struct.pack(">I", len(data)) + c + crc

    def _build_image_bytes(self) -> Optional[bytes]:
        e = self._engine
        index_grid = np.asarray(
            e.camera.render(e.current_level.get_sprites()), dtype=np.uint8
        )
        if index_grid.size == 0:
            return None
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        for idx, color in enumerate(self.ARC_PALETTE):
            mask = index_grid == idx
            rgb[mask] = color
        raw = bytearray()
        for y in range(64):
            raw.append(0)
            raw.extend(rgb[y].tobytes())
        compressed = zlib.compress(bytes(raw))
        sig = b"\x89PNG\r\n\x1a\n"
        ihdr = struct.pack(">IIBBBBB", 64, 64, 8, 2, 0, 0, 0)
        return sig + self._png_chunk(b"IHDR", ihdr) + self._png_chunk(b"IDAT", compressed) + self._png_chunk(b"IEND", b"")

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
                "lives": e.lives,
                "max_moves": e.max_moves,
                "moves_used": e.move_count,
                "queens_placed": len(e.placed),
                "queens_required": e.required,
                "cursor_x": e.cursor_x,
                "cursor_y": e.cursor_y,
                "grid_w": e.grid_w,
                "grid_h": e.grid_h,
                "flash_timer": e.flash_timer,
                "game_over": e._game_over,
                "done": done,
                "info": info or {},
                "levels_completed": getattr(self._engine, "_score", 0),
            },
        )

    def reset(self) -> GameState:
        self._total_turns = 0
        e = self._engine
        e.perform_action(ActionInput(id=GameAction.RESET), raw=True)
        return self._build_game_state()

    def get_actions(self) -> List[str]:
        if self.is_done():
            return ["reset"]
        return ["reset","up", "down", "left", "right", "select", "click", "undo"]

    def is_done(self) -> bool:
        e = self._engine
        return getattr(e, "_game_complete", False) or getattr(e, "_game_over", False)

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

        self._total_turns += 1

        level_before = e.level_index

        game_action = self.ACTION_MAP[action]
        info: Dict = {"action": action}

        if action == "click":
            action_input = ActionInput(id=game_action, data={"x": 0, "y": 0})
        else:
            action_input = ActionInput(id=game_action)
        frame = e.perform_action(action_input, raw=True)

        game_won = frame and frame.state and frame.state.name == "WIN"

        per_level_reward = 1.0 / NUM_LEVELS

        if game_won:
            info["reason"] = "game_complete"
            return StepResult(
                state=self._build_game_state(done=True, info=info),
                reward=per_level_reward,
                done=True,
                info=info,
            )

        if e._game_over:
            info["reason"] = "death"
            return StepResult(
                state=self._build_game_state(done=True, info=info),
                reward=0.0,
                done=True,
                info=info,
            )

        if e.level_index != level_before:
            info["reason"] = "level_complete"
            return StepResult(
                state=self._build_game_state(done=False, info=info),
                reward=per_level_reward,
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

    ACTION_LIST: List[str] = [
        "reset",
        "up",
        "down",
        "left",
        "right",
        "select",
        "click",
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
        result = self._env.step(action_str)

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


