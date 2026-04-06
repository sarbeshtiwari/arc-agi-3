from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import gymnasium as gym
import numpy as np
from arcengine import (
    ActionInput,
    ARCBaseGame,
    Camera,
    FrameData,
    FrameDataRaw,
    GameAction,
    Level,
    RenderableUserDisplay,
    Sprite,
)
from arcengine.enums import BlockingMode, GameState as EngineState
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

C_BG = 5
C_PAD = 4
C_WALL = 3
C_FLOOR = 0
C_PLAYER = 11
C_GHOST_A = 15
C_GHOST_B = 7
C_GHOST_C = 10
C_GHOST_D = 6
C_PLATE_RED = 8
C_PLATE_BLUE = 9
C_GATE_RED_CLOSED = 8
C_GATE_BLUE_CLOSED = 9
C_GOAL = 14
C_TOKEN = 12
C_SLOT_HINT = 12
C_REC_INDICATOR = 8
C_LIFE_ON = 8
C_LIFE_OFF = 4
C_BAR_FULL = 14
C_BAR_EMPTY = 3
C_SEP = 3
C_GHOST_AVAIL = 15
C_GHOST_USED = 4
C_TOGGLE_RED = 8
C_TOGGLE_BLUE = 9
C_INVERSE = 1

MAX_LIVES = 3
HUD_ROWS = 3
PROG_ROWS = 2
CANVAS = 64
MOVE_MULTIPLIER = 4
INVERSE_DURATION = 4

GHOST_COLORS = [C_GHOST_A, C_GHOST_B, C_GHOST_C, C_GHOST_D]

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

TILE_EMPTY = 0
TILE_WALL = 1
TILE_PLATE_RED = 2
TILE_PLATE_BLUE = 3
TILE_GATE_RED = 4
TILE_GATE_BLUE = 5
TILE_GOAL = 6
TILE_TOKEN_SRC = 7
TILE_GOAL_SLOT = 8
TILE_TOGGLE_RED = 13
TILE_TOGGLE_BLUE = 14
TILE_INVERSE = 15

LEVEL_CONFIGS: List[Dict] = [
    {
        "grid_w": 7,
        "grid_h": 7,
        "max_ghosts": 2,
        "budget_base": 24,
        "player_start": (1, 4),
        "has_toggle": False,
        "has_ghost_player_col": False,
        "has_ghost_ghost_col": False,
        "has_inverse": False,
        "grid": [
            [1, 1, 1, 1, 1, 1, 1],
            [1, 0, 7, 1, 0, 0, 1],
            [1, 0, 0, 4, 0, 0, 1],
            [1, 2, 0, 1, 0, 0, 1],
            [1, 0, 0, 1, 0, 0, 1],
            [1, 0, 8, 1, 0, 6, 1],
            [1, 1, 1, 1, 1, 1, 1],
        ],
    },
    {
        "grid_w": 9,
        "grid_h": 8,
        "max_ghosts": 2,
        "budget_base": 28,
        "player_start": (1, 3),
        "has_toggle": False,
        "has_ghost_player_col": True,
        "has_ghost_ghost_col": False,
        "has_inverse": False,
        "grid": [
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 2, 0, 1, 0, 0, 1, 0, 1],
            [1, 0, 0, 4, 0, 0, 5, 0, 1],
            [1, 0, 0, 1, 0, 0, 1, 0, 1],
            [1, 0, 0, 1, 0, 0, 1, 6, 1],
            [1, 3, 0, 1, 0, 0, 1, 0, 1],
            [1, 0, 0, 1, 0, 0, 1, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
    },
    {
        "grid_w": 11,
        "grid_h": 8,
        "max_ghosts": 3,
        "budget_base": 34,
        "player_start": (1, 3),
        "has_toggle": True,
        "has_ghost_player_col": True,
        "has_ghost_ghost_col": True,
        "has_inverse": False,
        "grid": [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 1, 0, 0, 0, 1, 0, 6, 1],
            [1, 0, 0, 4, 0, 0, 0, 1, 0, 0, 1],
            [1, 0, 2, 1, 13, 0, 0, 1, 0, 0, 1],
            [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
            [1, 0, 0, 1, 0, 14, 3, 5, 0, 0, 1],
            [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
    },
    {
        "grid_w": 11,
        "grid_h": 9,
        "max_ghosts": 3,
        "budget_base": 40,
        "player_start": (1, 3),
        "has_toggle": True,
        "has_ghost_player_col": True,
        "has_ghost_ghost_col": True,
        "has_inverse": True,
        "grid": [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 1, 0, 0, 0, 1, 0, 6, 1],
            [1, 0, 0, 4, 0, 0, 15, 1, 0, 0, 1],
            [1, 0, 2, 1, 13, 0, 0, 1, 0, 0, 1],
            [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
            [1, 0, 0, 1, 0, 14, 3, 5, 15, 0, 1],
            [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
            [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
    },
]


def _cell_size(grid_w: int, grid_h: int) -> int:
    usable_h = CANVAS - HUD_ROWS - PROG_ROWS
    size_by_w = CANVAS // grid_w
    size_by_h = usable_h // grid_h
    return max(2, min(size_by_w, size_by_h))


def _grid_origin(grid_w: int, grid_h: int) -> Tuple[int, int]:
    cs = _cell_size(grid_w, grid_h)
    total_w = grid_w * cs
    total_h = grid_h * cs
    usable_h = CANVAS - HUD_ROWS - PROG_ROWS
    ox = (CANVAS - total_w) // 2
    oy = HUD_ROWS + (usable_h - total_h) // 2
    return ox, oy


class EchoWeaveHUD(RenderableUserDisplay):
    def __init__(self) -> None:
        self.lives: int = MAX_LIVES
        self.recording: bool = False
        self.ghosts_used: int = 0
        self.ghosts_max: int = 1
        self.moves_used: int = 0
        self.move_budget: int = 1
        self.sim_running: bool = False
        self.inverted_left: int = 0

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        self._draw_move_bar(frame)
        self._draw_lives(frame)
        frame[2, :] = C_SEP
        self._draw_bottom_bar(frame)
        return frame

    def _draw_move_bar(self, frame: np.ndarray) -> None:
        if self.move_budget <= 0:
            return
        bar_w = 52
        ratio = min(self.moves_used / self.move_budget, 1.0)
        filled = int(bar_w * (1.0 - ratio))
        bar_color = C_INVERSE if self.inverted_left > 0 else C_BAR_FULL
        for col in range(bar_w):
            color = bar_color if col < filled else C_BAR_EMPTY
            frame[0, col] = color
            frame[1, col] = color

    def _draw_lives(self, frame: np.ndarray) -> None:
        for i in range(MAX_LIVES):
            lx = 56 + i * 3
            color = C_LIFE_ON if i < self.lives else C_LIFE_OFF
            if lx + 1 < CANVAS:
                frame[0, lx:lx + 2] = color
                frame[1, lx:lx + 2] = color

    def _draw_bottom_bar(self, frame: np.ndarray) -> None:
        bot_y = CANVAS - PROG_ROWS
        frame[bot_y, :] = C_SEP
        for i in range(self.ghosts_max):
            bx = 2 + i * 5
            color = C_GHOST_USED if i < self.ghosts_used else C_GHOST_AVAIL
            if bx + 2 < CANVAS and bot_y + 1 < CANVAS:
                frame[bot_y + 1, bx:bx + 3] = color


class Ghost:
    def __init__(
        self,
        spawn: Tuple[int, int],
        moves: List[Tuple[int, int]],
        color: int,
    ) -> None:
        self.spawn = spawn
        self.moves = moves
        self.color = color
        self.pos: Tuple[int, int] = spawn
        self.tick: int = 0
        self.carrying: Optional[int] = None
        self.active: bool = True

    def advance(
        self,
        grid: List[List[int]],
        grid_w: int,
        grid_h: int,
        gate_open: Dict[int, bool],
    ) -> None:
        if not self.active or not self.moves:
            return
        idx = self.tick % len(self.moves)
        dx, dy = self.moves[idx]
        nx, ny = self.pos[0] + dx, self.pos[1] + dy
        if _can_enter_directed(nx, ny, dx, dy, grid, grid_w, grid_h, gate_open):
            self.pos = (nx, ny)
        else:
            self.active = False
        self.tick += 1

    def reset_loop(self) -> None:
        self.pos = self.spawn
        self.tick = 0
        self.carrying = None
        self.active = True


def _can_enter_directed(
    cx: int,
    cy: int,
    dx: int,
    dy: int,
    grid: List[List[int]],
    grid_w: int,
    grid_h: int,
    gate_open: Dict[int, bool],
) -> bool:
    if cx < 0 or cx >= grid_w or cy < 0 or cy >= grid_h:
        return False
    tile = grid[cy][cx]
    if tile == TILE_WALL:
        return False
    if tile == TILE_GATE_RED and not gate_open.get(TILE_GATE_RED, False):
        return False
    if tile == TILE_GATE_BLUE and not gate_open.get(TILE_GATE_BLUE, False):
        return False
    return True


def _compute_gate_state(
    ghosts: List[Ghost],
    grid: List[List[int]],
    toggle_state: Dict[int, bool],
) -> Dict[int, bool]:
    red_held = False
    blue_held = False
    for gh in ghosts:
        if not gh.active:
            continue
        entity_x, entity_y = gh.pos
        if entity_y < 0 or entity_y >= len(grid):
            continue
        if entity_x < 0 or entity_x >= len(grid[entity_y]):
            continue
        tile = grid[entity_y][entity_x]
        if tile == TILE_PLATE_RED:
            red_held = True
        if tile == TILE_PLATE_BLUE:
            blue_held = True
    red_open = red_held or toggle_state.get(TILE_GATE_RED, False)
    blue_open = blue_held or toggle_state.get(TILE_GATE_BLUE, False)
    return {TILE_GATE_RED: red_open, TILE_GATE_BLUE: blue_open}


def _tile_color(tile: int) -> int:
    mapping = {
        TILE_WALL: C_WALL,
        TILE_PLATE_RED: C_PLATE_RED,
        TILE_PLATE_BLUE: C_PLATE_BLUE,
        TILE_GATE_RED: C_GATE_RED_CLOSED,
        TILE_GATE_BLUE: C_GATE_BLUE_CLOSED,
        TILE_GOAL: C_GOAL,
        TILE_TOKEN_SRC: C_TOKEN,
        TILE_GOAL_SLOT: C_SLOT_HINT,
        TILE_TOGGLE_RED: C_TOGGLE_RED,
        TILE_TOGGLE_BLUE: C_TOGGLE_BLUE,
        TILE_INVERSE: C_INVERSE,
    }
    return mapping.get(tile, C_FLOOR)


class Ew53(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._undo_stack: List[Dict] = []
        self._hud = EchoWeaveHUD()
        self._init_game_state()

        all_levels = self._build_levels()

        camera = Camera(
            background=C_BG,
            letter_box=C_PAD,
            width=CANVAS,
            height=CANVAS,
            interfaces=[self._hud],
        )

        super().__init__(
            game_id="ew53",
            levels=all_levels,
            camera=camera,
            available_actions=[0, 1, 2, 3, 4, 5, 7],
            seed=seed,
        )

    def _init_game_state(self) -> None:
        self._lives: int = MAX_LIVES
        self._grid: List[List[int]] = []
        self._grid_w: int = 0
        self._grid_h: int = 0
        self._player_pos: Tuple[int, int] = (0, 0)
        self._player_start: Tuple[int, int] = (0, 0)
        self._ghosts: List[Ghost] = []
        self._max_ghosts: int = 1
        self._recording: bool = False
        self._rec_moves: List[Tuple[int, int]] = []
        self._rec_start: Tuple[int, int] = (0, 0)
        self._sim_running: bool = False
        self._sim_tick: int = 0
        self._moves_used: int = 0
        self._move_budget: int = 1
        self._gate_open: Dict[int, bool] = {}
        self._toggle_state: Dict[int, bool] = {}
        self._toggle_prev_positions: Set[Tuple[int, int]] = set()
        self._sprite: Optional[Sprite] = None
        self._goal_slots: Dict[Tuple[int, int], int] = {}
        self._goal_filled: Set[Tuple[int, int]] = set()
        self._level_goal_type: str = "reach"
        self._consecutive_resets: int = 0
        self._has_toggle: bool = False
        self._has_ghost_player_col: bool = False
        self._has_ghost_ghost_col: bool = False
        self._has_inverse: bool = False
        self._inverted_moves_left: int = 0
        self._available_actions: List[int] = [0, 1, 2, 3, 4, 5, 7]

    def _build_levels(self) -> List[Level]:
        return [
            Level(
                sprites=[],
                grid_size=(CANVAS, CANVAS),
                data={"level_index": i},
                name=f"Level {i + 1}",
            )
            for i in range(len(LEVEL_CONFIGS))
        ]

    def reset(self) -> Union[FrameData, FrameDataRaw]:
        self._consecutive_resets = 2
        return self.perform_action(ActionInput(id=GameAction.RESET))

    def get_actions(self) -> List[int]:
        return list(self._available_actions)

    def env_step(self, action: int) -> Union[FrameData, FrameDataRaw]:
        return self.perform_action(ActionInput(id=GameAction.from_id(action)))

    def on_set_level(self, level: Level) -> None:
        cfg = LEVEL_CONFIGS[self.level_index]
        self._grid_w = cfg["grid_w"]
        self._grid_h = cfg["grid_h"]
        self._grid = [row[:] for row in cfg["grid"]]
        self._max_ghosts = cfg["max_ghosts"]
        self._move_budget = cfg["budget_base"] * MOVE_MULTIPLIER
        self._moves_used = 0
        self._lives = MAX_LIVES
        self._player_start = cfg["player_start"]
        self._player_pos = cfg["player_start"]
        self._ghosts = []
        self._recording = False
        self._rec_moves = []
        self._sim_running = False
        self._sim_tick = 0
        self._gate_open = {TILE_GATE_RED: False, TILE_GATE_BLUE: False}
        self._toggle_state = {TILE_GATE_RED: False, TILE_GATE_BLUE: False}
        self._toggle_prev_positions = set()
        self._goal_slots = {}
        self._goal_filled = set()
        self._level_goal_type = "reach"
        self._has_toggle = cfg.get("has_toggle", False)
        self._has_ghost_player_col = cfg.get("has_ghost_player_col", False)
        self._has_ghost_ghost_col = cfg.get("has_ghost_ghost_col", False)
        self._has_inverse = cfg.get("has_inverse", False)
        self._inverted_moves_left = 0
        self._available_actions = [0, 1, 2, 3, 4, 5, 7]
        self._undo_stack = []
        self._scan_special_tiles()
        self._sync_hud()
        level.remove_all_sprites()
        self._create_board_sprite(level)

    def _scan_special_tiles(self) -> None:
        for gy in range(self._grid_h):
            for gx in range(self._grid_w):
                if self._grid[gy][gx] == TILE_GOAL_SLOT:
                    self._goal_slots[(gx, gy)] = C_TOKEN
        if self._goal_slots:
            self._level_goal_type = "token"

    def _sync_hud(self) -> None:
        self._hud.lives = self._lives
        self._hud.ghosts_max = self._max_ghosts
        self._hud.ghosts_used = len(self._ghosts)
        self._hud.recording = self._recording
        self._hud.sim_running = self._sim_running
        self._hud.moves_used = self._moves_used
        self._hud.move_budget = self._move_budget
        self._hud.inverted_left = self._inverted_moves_left

    def _create_board_sprite(self, level: Level) -> None:
        pixels = self._render_frame()
        sp = Sprite(
            pixels=pixels,
            name="board",
            x=0,
            y=0,
            layer=0,
            blocking=BlockingMode.NOT_BLOCKED,
            collidable=False,
        )
        level.add_sprite(sp)
        self._sprite = sp

    def level_reset(self) -> None:
        lives = self._lives
        super().level_reset()
        self._lives = lives
        self._hud.lives = lives

    def full_reset(self) -> None:
        self._lives = MAX_LIVES
        self._hud.lives = MAX_LIVES
        self._consecutive_resets = 0
        super().full_reset()

    def _lose_life(self) -> None:
        self._lives -= 1
        self._hud.lives = self._lives
        if self._lives > 0:
            self.level_reset()
        else:
            self._available_actions = [GameAction.RESET.value]
            self.lose()

    def _consume_move(self) -> bool:
        self._moves_used += 1
        self._hud.moves_used = self._moves_used
        if self._moves_used >= self._move_budget:
            self._lose_life()
            return True
        return False

    def _push_frame(self) -> None:
        if self._sprite is None:
            return
        pixels = self._render_frame()
        np.copyto(self._sprite.pixels, pixels)

    def _render_frame(self) -> np.ndarray:
        frame = np.full((CANVAS, CANVAS), C_BG, dtype=np.int32)
        cs = _cell_size(self._grid_w, self._grid_h)
        ox, oy = _grid_origin(self._grid_w, self._grid_h)
        render_gate_open = _compute_gate_state(
            self._ghosts, self._grid, self._toggle_state
        )
        self._render_tiles(frame, cs, ox, oy, render_gate_open)
        self._render_ghosts(frame, cs, ox, oy)
        self._render_player(frame, cs, ox, oy)
        return frame

    def _render_tiles(
        self, frame: np.ndarray, cs: int, ox: int, oy: int,
        gate_open_for_render: Dict[int, bool],
    ) -> None:
        for gy in range(self._grid_h):
            for gx in range(self._grid_w):
                tile = self._grid[gy][gx]
                px_x = ox + gx * cs
                px_y = oy + gy * cs
                if px_x + cs > CANVAS or px_y + cs > CANVAS:
                    continue
                self._render_single_tile(
                    frame, tile, gx, gy, px_x, px_y, cs, gate_open_for_render
                )

    def _render_single_tile(
        self, frame: np.ndarray, tile: int,
        gx: int, gy: int, px_x: int, px_y: int, cs: int,
        gate_open_for_render: Dict[int, bool],
    ) -> None:
        color = _tile_color(tile)
        if tile == TILE_GATE_RED:
            color = C_FLOOR if gate_open_for_render.get(TILE_GATE_RED, False) else C_GATE_RED_CLOSED
        elif tile == TILE_GATE_BLUE:
            color = C_FLOOR if gate_open_for_render.get(TILE_GATE_BLUE, False) else C_GATE_BLUE_CLOSED
        elif tile == TILE_GOAL_SLOT:
            self._render_goal_slot(frame, gx, gy, px_x, px_y, cs)
            return
        frame[px_y:px_y + cs, px_x:px_x + cs] = color
        if tile in (TILE_PLATE_RED, TILE_PLATE_BLUE):
            inner = max(1, cs // 4)
            frame[px_y + inner:px_y + cs - inner, px_x + inner:px_x + cs - inner] = C_FLOOR
        elif tile == TILE_GOAL:
            inner = max(1, cs // 4)
            frame[px_y + inner:px_y + cs - inner, px_x + inner:px_x + cs - inner] = C_BG
        elif tile in (TILE_TOGGLE_RED, TILE_TOGGLE_BLUE):
            inner = max(1, cs // 4)
            frame[px_y + inner:px_y + cs - inner, px_x + inner:px_x + cs - inner] = C_BG
        elif tile == TILE_INVERSE:
            inner = max(1, cs // 4)
            frame[px_y + inner:px_y + cs - inner, px_x + inner:px_x + cs - inner] = C_BG

    def _render_goal_slot(
        self, frame: np.ndarray,
        gx: int, gy: int, px_x: int, px_y: int, cs: int,
    ) -> None:
        if (gx, gy) in self._goal_filled:
            frame[px_y:px_y + cs, px_x:px_x + cs] = C_TOKEN
        else:
            frame[px_y:px_y + cs, px_x:px_x + cs] = C_WALL
            inner = max(1, cs // 3)
            inner_x = px_x + inner
            inner_y = px_y + inner
            inner_w = max(1, cs - 2 * inner)
            inner_h = max(1, cs - 2 * inner)
            frame[inner_y:inner_y + inner_h, inner_x:inner_x + inner_w] = C_SLOT_HINT

    def _render_ghosts(
        self, frame: np.ndarray, cs: int, ox: int, oy: int
    ) -> None:
        for ghost in self._ghosts:
            if not ghost.active:
                continue
            gx, gy = ghost.pos
            px_x = ox + gx * cs
            px_y = oy + gy * cs
            if px_x + cs > CANVAS or px_y + cs > CANVAS:
                continue
            frame[px_y, px_x:px_x + cs] = ghost.color
            frame[px_y + cs - 1, px_x:px_x + cs] = ghost.color
            frame[px_y:px_y + cs, px_x] = ghost.color
            frame[px_y:px_y + cs, px_x + cs - 1] = ghost.color
            if ghost.carrying is not None:
                mid = cs // 2
                if 0 < mid < cs and px_y + mid < CANVAS and px_x + mid < CANVAS:
                    frame[px_y + mid, px_x + mid] = ghost.carrying

    def _render_player(
        self, frame: np.ndarray, cs: int, ox: int, oy: int
    ) -> None:
        gx, gy = self._player_pos
        px_x = ox + gx * cs
        px_y = oy + gy * cs
        if px_x + cs > CANVAS or px_y + cs > CANVAS:
            return
        frame[px_y:px_y + cs, px_x:px_x + cs] = C_PLAYER
        if self._recording:
            frame[px_y, px_x:px_x + cs] = C_REC_INDICATOR
            frame[px_y + cs - 1, px_x:px_x + cs] = C_REC_INDICATOR
        elif self._inverted_moves_left > 0:
            frame[px_y, px_x:px_x + cs] = C_INVERSE
            frame[px_y + cs - 1, px_x:px_x + cs] = C_INVERSE

    def _try_player_move(self, dx: int, dy: int) -> bool:
        nx = self._player_pos[0] + dx
        ny = self._player_pos[1] + dy
        if not _can_enter_directed(
            nx, ny, dx, dy, self._grid, self._grid_w, self._grid_h, self._gate_open
        ):
            return False
        self._player_pos = (nx, ny)
        self._check_inverse_tile()
        return True

    def _check_inverse_tile(self) -> None:
        if not self._has_inverse:
            return
        px, py = self._player_pos
        if self._grid[py][px] == TILE_INVERSE:
            self._inverted_moves_left = INVERSE_DURATION
            self._hud.inverted_left = self._inverted_moves_left

    def _apply_inversion(self, dx: int, dy: int) -> Tuple[int, int]:
        if self._inverted_moves_left > 0:
            return -dx, -dy
        return dx, dy

    def _tick_inversion(self) -> None:
        if self._inverted_moves_left > 0:
            self._inverted_moves_left -= 1
            self._hud.inverted_left = self._inverted_moves_left

    def _start_recording(self) -> None:
        if len(self._ghosts) >= self._max_ghosts:
            return
        self._recording = True
        self._rec_moves = []
        self._rec_start = self._player_pos
        self._hud.recording = True

    def _stop_recording(self) -> None:
        if not self._recording:
            return
        self._recording = False
        self._hud.recording = False
        if not self._rec_moves:
            return
        ghost_idx = len(self._ghosts)
        color = GHOST_COLORS[ghost_idx % len(GHOST_COLORS)]
        new_ghost = Ghost(
            spawn=self._rec_start,
            moves=list(self._rec_moves),
            color=color,
        )
        self._ghosts.append(new_ghost)
        self._hud.ghosts_used = len(self._ghosts)
        self._player_pos = self._player_start
        self._inverted_moves_left = 0
        self._hud.inverted_left = 0
        self._start_simulation()

    def _start_simulation(self) -> None:
        self._sim_running = True
        self._sim_tick = 0
        self._hud.sim_running = True
        self._goal_filled = set()
        self._toggle_state = {TILE_GATE_RED: False, TILE_GATE_BLUE: False}
        self._toggle_prev_positions = set()
        for gh in self._ghosts:
            gh.reset_loop()

    def _advance_simulation(self) -> None:
        if not self._sim_running:
            return
        self._gate_open = _compute_gate_state(
            self._ghosts, self._grid, self._toggle_state
        )
        for gh in self._ghosts:
            gh.advance(self._grid, self._grid_w, self._grid_h, self._gate_open)
        self._gate_open = _compute_gate_state(
            self._ghosts, self._grid, self._toggle_state
        )
        self._process_ghost_interactions()
        if self._has_toggle:
            self._process_toggle_plates()
        if self._has_ghost_ghost_col:
            self._process_ghost_ghost_collision()
        self._sim_tick += 1

    def _process_ghost_interactions(self) -> None:
        for gh in self._ghosts:
            if not gh.active:
                continue
            gx, gy = gh.pos
            tile = self._grid[gy][gx]
            if tile == TILE_TOKEN_SRC and gh.carrying is None:
                gh.carrying = C_TOKEN
            if (gx, gy) in self._goal_slots and gh.carrying is not None:
                expected = self._goal_slots[(gx, gy)]
                if gh.carrying == expected:
                    self._goal_filled.add((gx, gy))
                    gh.carrying = None

    def _process_toggle_plates(self) -> None:
        current_on_toggle: Set[Tuple[int, int]] = set()
        for gh in self._ghosts:
            if not gh.active:
                continue
            entity_x, entity_y = gh.pos
            if entity_y < 0 or entity_y >= self._grid_h:
                continue
            if entity_x < 0 or entity_x >= self._grid_w:
                continue
            tile = self._grid[entity_y][entity_x]
            if tile in (TILE_TOGGLE_RED, TILE_TOGGLE_BLUE):
                current_on_toggle.add((entity_x, entity_y))
        newly_stepped = current_on_toggle - self._toggle_prev_positions
        for pos in newly_stepped:
            tile = self._grid[pos[1]][pos[0]]
            if tile == TILE_TOGGLE_RED:
                self._toggle_state[TILE_GATE_RED] = not self._toggle_state[TILE_GATE_RED]
            elif tile == TILE_TOGGLE_BLUE:
                self._toggle_state[TILE_GATE_BLUE] = not self._toggle_state[TILE_GATE_BLUE]
        self._toggle_prev_positions = current_on_toggle

    def _process_ghost_ghost_collision(self) -> None:
        active_ghosts = [gh for gh in self._ghosts if gh.active]
        if len(active_ghosts) < 2:
            return
        positions: Dict[Tuple[int, int], List[Ghost]] = {}
        for gh in active_ghosts:
            if gh.pos not in positions:
                positions[gh.pos] = []
            positions[gh.pos].append(gh)
        collision_count = 0
        for pos_ghosts in positions.values():
            if len(pos_ghosts) >= 2:
                collision_count += 1
        if collision_count > 0:
            half_remaining = max(1, (self._move_budget - self._moves_used) // 2)
            penalty = half_remaining * collision_count
            self._moves_used += penalty
            self._hud.moves_used = self._moves_used

    def _check_ghost_player_collision(self) -> bool:
        if not self._has_ghost_player_col:
            return False
        for gh in self._ghosts:
            if gh.active and gh.pos == self._player_pos:
                self._lose_life()
                return True
        return False

    def _check_win(self) -> bool:
        player_x, player_y = self._player_pos
        player_tile = self._grid[player_y][player_x]
        if self._level_goal_type == "reach":
            return player_tile == TILE_GOAL
        if self._level_goal_type == "token":
            all_filled = (
                len(self._goal_filled) == len(self._goal_slots)
                and len(self._goal_slots) > 0
            )
            return all_filled and player_tile == TILE_GOAL
        return False

    def _check_all_ghosts_dead(self) -> bool:
        if not self._ghosts:
            return False
        if self._recording:
            return False
        if len(self._ghosts) < self._max_ghosts:
            return False
        for gh in self._ghosts:
            if gh.active:
                return False
        self._lose_life()
        return True

    def handle_reset(self) -> None:
        self._consecutive_resets += 1
        if self._consecutive_resets >= 2:
            self.full_reset()
        else:
            self._lives = MAX_LIVES
            self._hud.lives = MAX_LIVES
            self.level_reset()

    def _pixel_to_grid_cell(self, raw_x: int, raw_y: int) -> Optional[Tuple[int, int]]:
        cs = _cell_size(self._grid_w, self._grid_h)
        ox, oy = _grid_origin(self._grid_w, self._grid_h)
        gx = (raw_x - ox) // cs
        gy = (raw_y - oy) // cs
        if gx < 0 or gx >= self._grid_w or gy < 0 or gy >= self._grid_h:
            return None
        return (gx, gy)

    def _handle_mouse_click(self) -> None:
        raw_x = int(self.action.data.get("x", 0))
        raw_y = int(self.action.data.get("y", 0))
        cell = self._pixel_to_grid_cell(raw_x, raw_y)
        if cell is None:
            if self._consume_move():
                self.complete_action()
                return
            self._push_frame()
            self.complete_action()
            return
        target_x, target_y = cell
        player_x, player_y = self._player_pos
        if target_x == player_x and target_y == player_y:
            lost = self._handle_action5()
            if lost:
                self.complete_action()
                return
            if self._consume_move():
                self.complete_action()
                return
            self.complete_action()
            return
        diff_x = target_x - player_x
        diff_y = target_y - player_y
        dx, dy = 0, 0
        if abs(diff_x) >= abs(diff_y):
            dx = 1 if diff_x > 0 else -1
        else:
            dy = 1 if diff_y > 0 else -1
        self._step_direction(dx, dy)

    def _handle_action5(self) -> bool:
        if self._recording:
            self._stop_recording()
            if self._check_ghost_player_collision():
                return True
            self._push_frame()
            return False
        if len(self._ghosts) < self._max_ghosts:
            self._start_recording()
            if self._check_ghost_player_collision():
                return True
            self._push_frame()
            return False
        self._push_frame()
        return False

    def _handle_recording_move(self, dx: int, dy: int) -> bool:
        moved = self._try_player_move(dx, dy)
        if moved:
            self._rec_moves.append((dx, dy))
        self._tick_inversion()
        if self._sim_running:
            self._advance_simulation()
        if self._check_ghost_player_collision():
            return True
        self._push_frame()
        return False

    def _handle_navigation_move(self, dx: int, dy: int) -> bool:
        moved = self._try_player_move(dx, dy)
        self._tick_inversion()
        if self._sim_running:
            self._advance_simulation()
            if self._check_all_ghosts_dead():
                return True
        self._gate_open = _compute_gate_state(
            self._ghosts, self._grid, self._toggle_state
        )
        if self._check_ghost_player_collision():
            return True
        self._push_frame()
        if moved and self._check_win():
            self._lives = MAX_LIVES
            self._consecutive_resets = 0
            self._undo_stack = []
            self.next_level()
            return True
        return False

    def _handle_direction(self, dx: int, dy: int) -> bool:
        actual_dx, actual_dy = self._apply_inversion(dx, dy)
        if self._recording:
            return self._handle_recording_move(actual_dx, actual_dy)
        return self._handle_navigation_move(actual_dx, actual_dy)

    def _step_action5(self) -> None:
        lost = self._handle_action5()
        if lost:
            self.complete_action()
            return
        if self._consume_move():
            self.complete_action()
            return
        self.complete_action()

    def _step_direction(self, dx: int, dy: int) -> None:
        lost = self._handle_direction(dx, dy)
        if lost:
            self.complete_action()
            return
        if self._consume_move():
            self.complete_action()
            return
        self.complete_action()

    def _step_fallthrough(self) -> None:
        if self._consume_move():
            self.complete_action()
            return
        self._push_frame()
        self.complete_action()

    def _save_undo_snapshot(self) -> None:
        ghost_snapshots = []
        for gh in self._ghosts:
            ghost_snapshots.append({
                "spawn": gh.spawn,
                "moves": list(gh.moves),
                "color": gh.color,
                "pos": gh.pos,
                "tick": gh.tick,
                "carrying": gh.carrying,
                "active": gh.active,
            })
        snapshot = {
            "player_pos": self._player_pos,
            "player_start": self._player_start,
            "grid": [row[:] for row in self._grid],
            "ghosts": ghost_snapshots,
            "recording": self._recording,
            "rec_moves": list(self._rec_moves),
            "rec_start": self._rec_start,
            "sim_running": self._sim_running,
            "sim_tick": self._sim_tick,
            "moves_used": self._moves_used,
            "gate_open": dict(self._gate_open),
            "toggle_state": dict(self._toggle_state),
            "toggle_prev_positions": set(self._toggle_prev_positions),
            "goal_filled": set(self._goal_filled),
            "inverted_moves_left": self._inverted_moves_left,
            "lives": self._lives,
        }
        self._undo_stack.append(snapshot)

    def _restore_undo_snapshot(self) -> None:
        if not self._undo_stack:
            return
        snapshot = self._undo_stack.pop()
        current_moves = self._moves_used
        self._player_pos = snapshot["player_pos"]
        self._player_start = snapshot["player_start"]
        self._grid = snapshot["grid"]
        self._recording = snapshot["recording"]
        self._rec_moves = snapshot["rec_moves"]
        self._rec_start = snapshot["rec_start"]
        self._sim_running = snapshot["sim_running"]
        self._sim_tick = snapshot["sim_tick"]
        self._moves_used = current_moves
        self._gate_open = snapshot["gate_open"]
        self._toggle_state = snapshot["toggle_state"]
        self._toggle_prev_positions = snapshot["toggle_prev_positions"]
        self._goal_filled = snapshot["goal_filled"]
        self._inverted_moves_left = snapshot["inverted_moves_left"]
        self._lives = snapshot["lives"]
        self._ghosts = []
        for gs in snapshot["ghosts"]:
            gh = Ghost(spawn=gs["spawn"], moves=gs["moves"], color=gs["color"])
            gh.pos = gs["pos"]
            gh.tick = gs["tick"]
            gh.carrying = gs["carrying"]
            gh.active = gs["active"]
            self._ghosts.append(gh)
        self._sync_hud()
        self._push_frame()

    def step(self) -> None:
        aid = self.action.id

        if aid == GameAction.RESET:
            self.complete_action()
            return

        self._consecutive_resets = 0

        if aid == GameAction.ACTION7:
            self._restore_undo_snapshot()
            if self._consume_move():
                self.complete_action()
                return
            self.complete_action()
            return

        if aid == GameAction.ACTION6:
            self._handle_mouse_click()
            return

        if aid in (GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3,
                   GameAction.ACTION4, GameAction.ACTION5):
            self._save_undo_snapshot()

        if aid == GameAction.ACTION5:
            self._step_action5()
            return

        dx, dy = 0, 0
        if aid == GameAction.ACTION1:
            dy = -1
        elif aid == GameAction.ACTION2:
            dy = 1
        elif aid == GameAction.ACTION3:
            dx = -1
        elif aid == GameAction.ACTION4:
            dx = 1

        if dx != 0 or dy != 0:
            self._step_direction(dx, dy)
            return

        self._step_fallthrough()


TILE_NAMES: Dict[int, str] = {
    TILE_EMPTY: ".",
    TILE_WALL: "#",
    TILE_PLATE_RED: "r",
    TILE_PLATE_BLUE: "b",
    TILE_GATE_RED: "R",
    TILE_GATE_BLUE: "B",
    TILE_GOAL: "G",
    TILE_TOKEN_SRC: "T",
    TILE_GOAL_SLOT: "S",
    TILE_TOGGLE_RED: "x",
    TILE_TOGGLE_BLUE: "y",
    TILE_INVERSE: "I",
}


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
        self._engine: Optional[Ew53] = Ew53(seed=seed)
        self._total_turns = 0
        self._done = False
        self._game_won = False
        self._last_action_was_reset = False

    def reset(self) -> GameState:
        assert self._engine is not None
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

    def step(self, action: str) -> StepResult:
        assert self._engine is not None
        e = self._engine

        if action == "reset":
            state = self.reset()
            return StepResult(
                state=state, reward=0.0, done=False, info={"action": "reset"}
            )

        if action not in self._ACTION_MAP:
            raise ValueError(
                f"Invalid action '{action}'. Must be one of {list(self._ACTION_MAP.keys())}"
            )

        self._last_action_was_reset = False
        self._total_turns += 1

        game_action = self._ACTION_MAP[action]
        info: Dict[str, Any] = {"action": action}

        level_before = e.level_index

        action_input = ActionInput(id=game_action)
        frame = e.perform_action(action_input, raw=True)

        state_name = frame.state.name if frame and frame.state else ""
        game_won = state_name == "WIN"
        game_over = state_name == "GAME_OVER"

        total_levels = len(self._engine._levels)
        level_reward = 1.0 / total_levels

        if game_won:
            self._done = True
            self._game_won = True
            info["reason"] = "game_complete"
            return StepResult(
                state=self._build_game_state(done=True),
                reward=level_reward,
                done=True,
                info=info,
            )

        if game_over:
            self._done = True
            info["reason"] = "game_over"
            return StepResult(
                state=self._build_game_state(done=True),
                reward=0.0,
                done=True,
                info=info,
            )

        reward = 0.0
        if e.level_index != level_before:
            reward = level_reward
            info["reason"] = "level_complete"

        return StepResult(
            state=self._build_game_state(done=False),
            reward=reward,
            done=False,
            info=info,
        )

    def get_actions(self) -> List[str]:
        if self._done:
            return ["reset"]
        return list(self._VALID_ACTIONS)

    def is_done(self) -> bool:
        return self._done

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError(f"Unsupported render mode: {mode}")
        assert self._engine is not None
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

    def _build_game_state(self, done: bool = False) -> GameState:
        assert self._engine is not None
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

    def _build_text_observation(self) -> str:
        assert self._engine is not None
        g = self._engine
        lines: List[str] = []

        lines.append(
            f"Level {g.level_index + 1}/4 | "
            f"Lives {g._lives}/{MAX_LIVES} | "
            f"Moves {g._moves_used}/{g._move_budget}"
        )

        if g._recording:
            lines.append(
                f"Mode: RECORDING ghost {len(g._ghosts) + 1}/{g._max_ghosts}"
            )
        elif g._sim_running:
            lines.append(
                f"Mode: NAVIGATING | Ghosts active | "
                f"Sim tick {g._sim_tick}"
            )
        else:
            lines.append(
                f"Mode: SETUP | Ghosts placed {len(g._ghosts)}/{g._max_ghosts}"
            )

        if g._inverted_moves_left > 0:
            lines.append(f"INVERTED controls ({g._inverted_moves_left} moves left)")

        if g._level_goal_type == "token":
            filled = len(g._goal_filled)
            total = len(g._goal_slots)
            lines.append(f"Goal: deliver tokens to slots ({filled}/{total} filled)")
        else:
            lines.append("Goal: reach the goal tile (G)")

        if g._gate_open.get(TILE_GATE_RED, False):
            lines.append("Red gate: OPEN")
        else:
            lines.append("Red gate: CLOSED")
        if g._gate_open.get(TILE_GATE_BLUE, False):
            lines.append("Blue gate: OPEN")
        else:
            lines.append("Blue gate: CLOSED")

        lines.append("")
        px, py = g._player_pos
        for gy in range(g._grid_h):
            row_chars: List[str] = []
            for gx in range(g._grid_w):
                if gx == px and gy == py:
                    if g._recording:
                        row_chars.append("@")
                    else:
                        row_chars.append("P")
                else:
                    ghost_here = False
                    for gi, gh in enumerate(g._ghosts):
                        if gh.active and gh.pos == (gx, gy):
                            row_chars.append(str(gi + 1))
                            ghost_here = True
                            break
                    if not ghost_here:
                        tile = g._grid[gy][gx]
                        if tile == TILE_GATE_RED:
                            row_chars.append(
                                "." if g._gate_open.get(TILE_GATE_RED, False)
                                else "R"
                            )
                        elif tile == TILE_GATE_BLUE:
                            row_chars.append(
                                "." if g._gate_open.get(TILE_GATE_BLUE, False)
                                else "B"
                            )
                        elif tile == TILE_GOAL_SLOT:
                            if (gx, gy) in g._goal_filled:
                                row_chars.append("*")
                            else:
                                row_chars.append("S")
                        else:
                            row_chars.append(
                                TILE_NAMES.get(tile, "?")
                            )
            lines.append(" ".join(row_chars))

        lines.append("")
        for gi, gh in enumerate(g._ghosts):
            status = "active" if gh.active else "inactive"
            carrying = "carrying token" if gh.carrying is not None else ""
            lines.append(
                f"Ghost {gi + 1}: {gh.pos} {status}"
                + (f" {carrying}" if carrying else "")
            )

        return "\n".join(lines)


class ArcGameEnv(gym.Env):
    metadata: Dict[str, Any] = {"render_modes": ["rgb_array"], "render_fps": 5}

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
