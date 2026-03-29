from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from arcengine import (
    ARCBaseGame,
    ActionInput,
    Camera,
    GameAction,
    Level,
    Sprite,
)
from arcengine import GameState as EngineState


@dataclass
class GameState:
    text_observation: str
    image_observation: bytes | None
    valid_actions: list[str] | None
    turn: int
    metadata: dict = field(default_factory=dict)


@dataclass
class StepResult:
    state: GameState
    reward: float
    done: bool
    info: dict = field(default_factory=dict)


WHITE = 0
OFF_WHITE = 1
NEUTRAL = 3
BLACK = 5
MAGENTA = 6
MAGENTA_LIGHT = 7
RED = 8
BLUE = 9
BLUE_LIGHT = 10
YELLOW = 11
MAROON = 13
GREEN = 14
PURPLE = 15

COLOR_BACKGROUND = OFF_WHITE
COLOR_WALL = BLACK
COLOR_PLAYER = MAGENTA
COLOR_SHADOW = MAGENTA_LIGHT
COLOR_GOAL = GREEN
COLOR_SHADOW_GOAL = GREEN
COLOR_MOVABLE = OFF_WHITE
COLOR_KEY = YELLOW
COLOR_GATE = MAROON
COLOR_SWITCH = YELLOW
COLOR_SWITCH_ACTIVE = WHITE
COLOR_TOGGLE_GATE = MAROON
COLOR_TOGGLE_GATE_OPEN = WHITE
COLOR_TRAP_FLOOR = PURPLE
COLOR_HAZARD = RED
COLOR_TELEPORTER = BLUE
COLOR_BOUNDARY = NEUTRAL

MAX_LIVES = 3

DIR_UP = (0, -1)
DIR_DOWN = (0, 1)
DIR_LEFT = (-1, 0)
DIR_RIGHT = (1, 0)

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

_ACTION_MAP: dict[str, GameAction] = {
    "reset": GameAction.RESET,
    "up": GameAction.ACTION1,
    "down": GameAction.ACTION2,
    "left": GameAction.ACTION3,
    "right": GameAction.ACTION4,
    "undo": GameAction.ACTION7,
}

_VALID_ACTIONS = ["reset","up", "down", "left", "right", "undo"]

LEVEL_DATA: list[dict[str, Any]] = [
    {
        "max_moves": 84,
        "grid": [
            list("################"),
            list("################"),
            list("##S#   #    g###"),
            list("## # # ##### ###"),
            list("##   # #   # ###"),
            list("###### # # # ###"),
            list("##   # # #   ###"),
            list("#### # # #######"),
            list("##   # #     ###"),
            list("## # # # ### ###"),
            list("## # # # #   ###"),
            list("## ### ### # ###"),
            list("##s        #G###"),
            list("################"),
            list("################"),
            list("################"),
        ],
    },
    {
        "max_moves": 156,
        "grid": [
            list("####################"),
            list("####################"),
            list("##S#       #    g###"),
            list("## # ##### ### # ###"),
            list("##   #   # #   # ###"),
            list("###### # # # #######"),
            list("## #   #   #     ###"),
            list("## # ####### ### ###"),
            list("##   #   #     # ###"),
            list("## ### #K####### ###"),
            list("## #   # #     # ###"),
            list("## # ### # ### # ###"),
            list("## # # # # #     ###"),
            list("## # # # # ##### ###"),
            list("##   #   #   #   ###"),
            list("###### ##### # #L###"),
            list("##s          #  G###"),
            list("####################"),
            list("####################"),
            list("####################"),
        ],
    },
    {
        "max_moves": 176,
        "grid": [
            list("################################"),
            list("################################"),
            list("################################"),
            list("################################"),
            list("################################"),
            list("################################"),
            list("######S    #     #   #   g######"),
            list("########## # ### # # # # #######"),
            list("###### #   #   #   #   # #######"),
            list("###### # ##### ######### #######"),
            list("######   #       #   #   #######"),
            list("###### ########### # # #########"),
            list("###### #           # #   #######"),
            list("###### # ##### ##### ### #######"),
            list("###### #   #  M#   #     #######"),
            list("###### # # ##### # #############"),
            list("###### # #      P#       #######"),
            list("###### # ############### #######"),
            list("###### #     #     # #   #######"),
            list("###### ##### ### # # # # #######"),
            list("###### #         #   # # #######"),
            list("###### ########### ### # #######"),
            list("######   #   #     #   # #######"),
            list("######## # # ####### ### #######"),
            list("######     #         #   Q######"),
            list("######s##################G######"),
            list("################################"),
            list("################################"),
            list("################################"),
            list("################################"),
            list("################################"),
            list("################################"),
        ],
    },
    {
        "max_moves": 124,
        "grid": [
            list("################################"),
            list("################################"),
            list("################################"),
            list("###S# #     #    ! !#      g####"),
            list("### # # # # ##### ### ### # ####"),
            list("###       #   # #   #       ####"),
            list("######### # # # ### ##### # ####"),
            list("###             # # #       ####"),
            list("### # # # ####### # # # ### ####"),
            list("###     #   #         #     ####"),
            list("### ####### ### ####### #!# ####"),
            list("### # #    !      #   # # # ####"),
            list("### ##### ##### ##### # ########"),
            list("### #     # # # #       #   ####"),
            list("### # # # ###!### ##### ### ####"),
            list("###!# #       # # #       # ####"),
            list("### #!# ########### # # # ######"),
            list("### # # # #       # # #!#   ####"),
            list("### ####### # ### ### # # ######"),
            list("###   ! !     #   #     #   ####"),
            list("####### # ### # # # # # # # ####"),
            list("### #   # #           #     ####"),
            list("### # # ### ##### # ### ### ####"),
            list("###       #       # #     # ####"),
            list("### # ### # # # # ##### # # ####"),
            list("### # # #     # #   # # #!! ####"),
            list("### # # ######### ### # ########"),
            list("###s  #           #        G####"),
            list("################################"),
            list("################################"),
            list("################################"),
            list("################################"),
        ],
    },
    {
        "max_moves": 152,
        "grid": [
            list("################################"),
            list("#S#   #   #           #      g##"),
            list("# # # # # # ######### ##### #P##"),
            list("#   # # #   # #     # #     # ##"),
            list("##### # ##### # # ### # ########"),
            list("#    T#   # #   #     #       ##"),
            list("# ####### # # ######### ##### ##"),
            list("#   #     #   #           #   ##"),
            list("### # ##### ### ########### # ##"),
            list("# #   #     #   #           # ##"),
            list("# ##### ##K## ### # ######### ##"),
            list("#       # #   # # # #   #     ##"),
            list("# ####### # ### # ### # # ######"),
            list("# #     # #   # #     # # #   ##"),
            list("# ### # # ### # ####### # # ####"),
            list("# #   # #     #M      # # #   ##"),
            list("# # ### # ##### # ##### # ### ##"),
            list("# # #   # #     # #     # #   ##"),
            list("# # # ### ####### # ##### # # ##"),
            list("#   # # # #     # #     #   # ##"),
            list("##### # # # ### # ##### ##### ##"),
            list("#   # #     #         # #     ##"),
            list("# ### # ############# # # ######"),
            list("#   # #   #   #       # # #   ##"),
            list("# # # ### # # ######### # # # ##"),
            list("# #   #   # # #   #     #T  # ##"),
            list("# ######### # # # # ######### ##"),
            list("# #         #   #   #   #  !  ##"),
            list("#Q# ################# # # ###L##"),
            list("#s  #                 #      G##"),
            list("################################"),
            list("################################"),
        ],
    },
]


@dataclass(frozen=True)
class UndoState:
    player_pos: tuple[int, int]
    shadow_pos: tuple[int, int]
    movable_blocks: frozenset[tuple[int, int]]
    keys: frozenset[tuple[int, int]]
    keys_count: int
    gates_opened: frozenset[tuple[int, int]]
    switches_active: bool
    trap_floors_triggered: frozenset[tuple[int, int]]


def _make_png(index_grid: np.ndarray) -> bytes:
    h, w = index_grid.shape
    raw_rows = bytearray()
    for y in range(h):
        raw_rows.append(0)
        for x in range(w):
            idx = int(index_grid[y, x])
            r, g, b = ARC_PALETTE[idx] if idx < len(ARC_PALETTE) else (0, 0, 0)
            raw_rows.append(r)
            raw_rows.append(g)
            raw_rows.append(b)

    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        c = chunk_type + data
        return (
            struct.pack(">I", len(data))
            + c
            + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        )

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return (
        sig
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(bytes(raw_rows)))
        + _chunk(b"IEND", b"")
    )


class Gs01(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._current_level_index = 0
        self._lives = MAX_LIVES
        self._game_over = False
        self._level_cleared = False

        self.player_pos: tuple[int, int] = (0, 0)
        self.spawn_pos: tuple[int, int] = (0, 0)
        self.goal_pos: tuple[int, int] = (0, 0)

        self.shadow_pos: tuple[int, int] = (0, 0)
        self.shadow_spawn_pos: tuple[int, int] = (0, 0)
        self.shadow_goal_pos: tuple[int, int] = (0, 0)

        self.walls: set[tuple[int, int]] = set()
        self.movable_blocks: set[tuple[int, int]] = set()
        self.keys: set[tuple[int, int]] = set()
        self.gates: set[tuple[int, int]] = set()
        self.gates_opened: set[tuple[int, int]] = set()
        self.keys_count: int = 0

        self.switches: set[tuple[int, int]] = set()
        self.toggle_gates: set[tuple[int, int]] = set()
        self.switches_active: bool = False

        self.teleporters: dict[tuple[int, int], tuple[int, int]] = {}
        self.trap_floors: set[tuple[int, int]] = set()
        self.trap_floors_triggered: set[tuple[int, int]] = set()
        self.hazards: set[tuple[int, int]] = set()

        self.initial_movable_blocks: set[tuple[int, int]] = set()
        self.initial_keys: set[tuple[int, int]] = set()

        self.move_count: int = 0
        self.max_moves: int = 100
        self.undo_stack: list[UndoState] = []

        self._grid_width: int = 0
        self._grid_height: int = 0

        levels = []
        for ld in LEVEL_DATA:
            g = ld["grid"]
            gw = len(g[0]) if g else 0
            gh = len(g)
            levels.append(Level(sprites=[], grid_size=(gw, gh)))
        first_grid = LEVEL_DATA[0]["grid"]
        first_w = len(first_grid[0])
        first_h = len(first_grid)
        camera = Camera(
            background=COLOR_BACKGROUND,
            letter_box=COLOR_BOUNDARY,
            width=first_w,
            height=first_h,
            x=0,
            y=0,
            interfaces=[],
        )
        super().__init__(
            game_id="gs01",
            levels=levels,
            camera=camera,
            available_actions=[0, 1, 2, 3, 4, 7],
        )

    def on_set_level(self, level: Level) -> None:
        self._current_level_index = self.level_index
        self._game_over = False
        self._level_cleared = False
        self._load_level()
        self.camera.width = self._grid_width
        self.camera.height = self._grid_height
        self._render()

    def handle_reset(self) -> None:
        self._game_over = False
        self._level_cleared = False
        if self._state == EngineState.WIN:
            self._lives = MAX_LIVES
            self.full_reset()
        elif self._state == EngineState.GAME_OVER:
            self._lives = MAX_LIVES
            self.level_reset()
        elif self.move_count == 0:
            self._lives = MAX_LIVES
            self.full_reset()
        else:
            self._lives = MAX_LIVES
            self.level_reset()

    def step(self) -> None:
        action = self.action.id

        if action == GameAction.RESET:
            self._render()
            self.complete_action()
            return

        if action == GameAction.ACTION1:
            self._do_move(DIR_UP)
        elif action == GameAction.ACTION2:
            self._do_move(DIR_DOWN)
        elif action == GameAction.ACTION3:
            self._do_move(DIR_LEFT)
        elif action == GameAction.ACTION4:
            self._do_move(DIR_RIGHT)
        elif action == GameAction.ACTION7:
            self._pop_state()
            self.move_count += 1
            if self.move_count >= self.max_moves:
                self._handle_death()
            self._render()
        else:
            self._render()

        self.complete_action()

    def _do_move(self, direction: tuple[int, int]) -> None:
        dx, dy = direction
        self._push_state()

        p_bumped = self._handle_bump_gates(
            self.player_pos[0], self.player_pos[1], dx, dy
        )
        s_bumped = self._handle_bump_gates(
            self.shadow_pos[0], self.shadow_pos[1], -dx, dy
        )

        self.move_count += 1

        moved = self._apply_movement((dx, dy), p_bumped, s_bumped)

        if moved is None:
            self._render()
            return

        if self.move_count >= self.max_moves:
            self._handle_death()
            self._render()
            return

        self._render()

    def build_canvas(self) -> list[list[int]]:
        gw, gh = self._grid_width, self._grid_height
        canvas: list[list[int]] = [
            [COLOR_BACKGROUND for _ in range(gw)] for _ in range(gh)
        ]
        self._draw_entities(canvas, gw, gh)
        self._draw_hud(canvas, gw, gh)
        return canvas

    def _draw_entities(self, canvas: list[list[int]], gw: int, gh: int) -> None:
        for wx, wy in self.walls:
            if 0 <= wx < gw and 0 <= wy < gh:
                is_edge = wx == 0 or wx == gw - 1 or wy == 0 or wy == gh - 1
                canvas[wy][wx] = COLOR_BOUNDARY if is_edge else COLOR_WALL

        for hx, hy in self.hazards:
            if 0 <= hx < gw and 0 <= hy < gh:
                canvas[hy][hx] = COLOR_HAZARD

        for mx, my in self.movable_blocks:
            if 0 <= mx < gw and 0 <= my < gh:
                canvas[my][mx] = COLOR_MOVABLE

        for kx, ky in self.keys:
            if 0 <= kx < gw and 0 <= ky < gh:
                canvas[ky][kx] = COLOR_KEY

        for gx, gy in self.gates:
            if (gx, gy) not in self.gates_opened:
                if 0 <= gx < gw and 0 <= gy < gh:
                    canvas[gy][gx] = COLOR_GATE

        for sx, sy in self.switches:
            if 0 <= sx < gw and 0 <= sy < gh:
                canvas[sy][sx] = (
                    COLOR_SWITCH_ACTIVE if self.switches_active else COLOR_SWITCH
                )

        for tx, ty in self.toggle_gates:
            if 0 <= tx < gw and 0 <= ty < gh:
                canvas[ty][tx] = (
                    COLOR_TOGGLE_GATE_OPEN
                    if self.switches_active
                    else COLOR_TOGGLE_GATE
                )

        for tx, ty in self.trap_floors:
            if (tx, ty) not in self.trap_floors_triggered:
                if 0 <= tx < gw and 0 <= ty < gh:
                    canvas[ty][tx] = COLOR_TRAP_FLOOR

        for tx, ty in self.teleporters:
            if 0 <= tx < gw and 0 <= ty < gh:
                canvas[ty][tx] = COLOR_TELEPORTER

        sgx, sgy = self.shadow_goal_pos
        if 0 <= sgx < gw and 0 <= sgy < gh:
            canvas[sgy][sgx] = COLOR_SHADOW_GOAL

        gx2, gy2 = self.goal_pos
        if 0 <= gx2 < gw and 0 <= gy2 < gh:
            canvas[gy2][gx2] = COLOR_GOAL

        sx2, sy2 = self.shadow_pos
        if 0 <= sx2 < gw and 0 <= sy2 < gh:
            canvas[sy2][sx2] = COLOR_SHADOW

        px, py = self.player_pos
        if 0 <= px < gw and 0 <= py < gh:
            canvas[py][px] = COLOR_PLAYER

    def _draw_hud(self, canvas: list[list[int]], gw: int, gh: int) -> None:
        for i in range(MAX_LIVES):
            lx = gw - 2 - i
            if 0 <= lx < gw:
                canvas[0][lx] = RED if i < self._lives else COLOR_BOUNDARY

        bar_w = min(10, gw - 4)
        if bar_w > 0 and gh > 0:
            bar_x = (gw - bar_w) // 2
            fill = (
                int((self.move_count / self.max_moves) * bar_w)
                if self.max_moves > 0
                else 0
            )
            for i in range(bar_w):
                if 0 <= bar_x + i < gw:
                    canvas[gh - 1][bar_x + i] = BLUE if i < fill else BLUE_LIGHT

    def _render(self) -> None:
        canvas = self.build_canvas()
        self.current_level.remove_all_sprites()
        self.current_level.add_sprite(
            Sprite(pixels=canvas, name="display", visible=True, collidable=False)
        )

    def _load_level(self) -> None:
        if self._current_level_index < 0 or self._current_level_index >= len(
            LEVEL_DATA
        ):
            self._current_level_index = 0
        level_data = LEVEL_DATA[self._current_level_index]
        grid = self._prepare_grid(level_data["grid"])

        self._grid_width = len(grid[0]) if grid else 0
        self._grid_height = len(grid)

        self._parse_grid(grid)

        self.player_pos = self.spawn_pos
        self.shadow_pos = self.shadow_spawn_pos
        self.keys_count = 0
        self.gates_opened = set()
        self.switches_active = False
        self.trap_floors_triggered = set()
        self.move_count = 0
        self.max_moves = level_data.get("max_moves", 100)
        self.undo_stack = []

    def _prepare_grid(self, source: list[list[str]]) -> list[list[str]]:
        return [row[:] for row in source]

    def _parse_grid(self, grid: list[list[str]]) -> None:
        self.walls = set()
        self.movable_blocks = set()
        self.keys = set()
        self.gates = set()
        self.switches = set()
        self.toggle_gates = set()
        self.trap_floors = set()
        self.hazards = set()
        self.teleporters = {}

        t_pos: list[tuple[int, int]] = []
        for y, row in enumerate(grid):
            for x, char in enumerate(row):
                if char == "#":
                    self.walls.add((x, y))
                elif char == "S":
                    self.spawn_pos = (x, y)
                elif char == "s":
                    self.shadow_spawn_pos = (x, y)
                elif char == "G":
                    self.goal_pos = (x, y)
                elif char == "g":
                    self.shadow_goal_pos = (x, y)
                elif char == "M":
                    self.movable_blocks.add((x, y))
                elif char == "K":
                    self.keys.add((x, y))
                elif char == "L":
                    self.gates.add((x, y))
                elif char == "P":
                    self.switches.add((x, y))
                elif char == "Q":
                    self.toggle_gates.add((x, y))
                elif char == "!":
                    self.trap_floors.add((x, y))
                elif char == "X":
                    self.hazards.add((x, y))
                elif char == "T":
                    t_pos.append((x, y))

        for i in range(0, len(t_pos) - 1, 2):
            self.teleporters[t_pos[i]] = t_pos[i + 1]
            self.teleporters[t_pos[i + 1]] = t_pos[i]

        self.initial_movable_blocks = self.movable_blocks.copy()
        self.initial_keys = self.keys.copy()

    def _is_solid(self, x: int, y: int) -> bool:
        if not (0 <= x < self._grid_width and 0 <= y < self._grid_height):
            return True
        if (x, y) in self.walls:
            return True
        if (x, y) in self.gates and (x, y) not in self.gates_opened:
            return True
        if (x, y) in self.toggle_gates and not self.switches_active:
            return True
        if (x, y) in self.trap_floors_triggered:
            return True
        return False

    def _slide_entity(
        self,
        x: int,
        y: int,
        dx: int,
        dy: int,
        goal: tuple[int, int],
        other: tuple[int, int],
    ) -> tuple[int, int, bool, bool]:
        nx, ny = x + dx, y + dy
        can_move = True
        if self._is_solid(nx, ny):
            can_move = False
        elif (nx, ny) in self.movable_blocks:
            nbx, nby = nx + dx, ny + dy
            if (
                self._is_solid(nbx, nby)
                or (nbx, nby) in self.movable_blocks
                or (nbx, nby) in self.hazards
                or (nbx, nby) == other
            ):
                can_move = False
            else:
                self.movable_blocks.remove((nx, ny))
                self.movable_blocks.add((nbx, nby))

        if not can_move:
            return x, y, True, False

        x, y = nx, ny
        if (x, y) in self.keys:
            self.keys.remove((x, y))
            self.keys_count += 1

        if (x, y) in self.trap_floors and (x, y) not in self.trap_floors_triggered:
            self.trap_floors_triggered.add((x, y))

        if (x, y) in self.hazards:
            self._handle_death()
            return x, y, True, True

        if (x, y) in self.teleporters:
            x, y = self.teleporters[(x, y)]

        stopped = (x, y) == goal
        return x, y, stopped, False

    def _apply_movement(
        self, direction: tuple[int, int], p_bumped: bool = False, s_bumped: bool = False
    ) -> bool | None:
        dx, dy = direction
        sdx, sdy = -dx, dy

        p_x, p_y = self.player_pos
        s_x, s_y = self.shadow_pos

        p_start = (p_x, p_y)
        s_start = (s_x, s_y)

        p_stopped = p_bumped
        s_stopped = s_bumped
        max_slide = self._grid_width * self._grid_height * 4
        slide_iter = 0

        while not (p_stopped and s_stopped):
            slide_iter += 1
            if slide_iter > max_slide:
                break
            if not p_stopped:
                p_x, p_y, p_stopped, died = self._slide_entity(
                    p_x, p_y, dx, dy, self.goal_pos, (s_x, s_y)
                )
                if died:
                    return None
                self.player_pos = (p_x, p_y)
                self._update_switch_state()

            if not s_stopped:
                s_x, s_y, s_stopped, died = self._slide_entity(
                    s_x, s_y, sdx, sdy, self.shadow_goal_pos, (p_x, p_y)
                )
                if died:
                    return None
                self.shadow_pos = (s_x, s_y)
                self._update_switch_state()

        self.player_pos = (p_x, p_y)
        self.shadow_pos = (s_x, s_y)

        if self.player_pos == self.goal_pos and self.shadow_pos == self.shadow_goal_pos:
            self._lives = MAX_LIVES
            self._level_cleared = True
            self.next_level()
            return None

        self._update_switch_state()
        return self.player_pos != p_start or self.shadow_pos != s_start

    def _handle_bump_gates(self, x: int, y: int, dx: int, dy: int) -> bool:
        nx, ny = x + dx, y + dy
        if (nx, ny) in self.gates and (nx, ny) not in self.gates_opened:
            if self.keys_count > 0:
                self.keys_count -= 1
                self.gates_opened.add((nx, ny))
                return True
        return False

    def _update_switch_state(self) -> None:
        on_switch = False
        for s_pos in self.switches:
            if (
                self.player_pos == s_pos
                or self.shadow_pos == s_pos
                or s_pos in self.movable_blocks
            ):
                on_switch = True
                break
        self.switches_active = on_switch

    def _handle_death(self) -> None:
        self._lives -= 1
        if self._lives == 0:
            self._game_over = True
            self.lose()
        else:
            self._load_level()

    def _push_state(self) -> None:
        state = UndoState(
            player_pos=self.player_pos,
            shadow_pos=self.shadow_pos,
            movable_blocks=frozenset(self.movable_blocks),
            keys=frozenset(self.keys),
            keys_count=self.keys_count,
            gates_opened=frozenset(self.gates_opened),
            switches_active=self.switches_active,
            trap_floors_triggered=frozenset(self.trap_floors_triggered),
        )
        self.undo_stack.append(state)
        if len(self.undo_stack) > self.max_moves:
            self.undo_stack.pop(0)

    def _pop_state(self) -> bool:
        if not self.undo_stack:
            return False
        s = self.undo_stack.pop()
        self.player_pos = s.player_pos
        self.shadow_pos = s.shadow_pos
        self.movable_blocks = set(s.movable_blocks)
        self.keys = set(s.keys)
        self.keys_count = s.keys_count
        self.gates_opened = set(s.gates_opened)
        self.switches_active = s.switches_active
        self.trap_floors_triggered = set(s.trap_floors_triggered)
        return True


class PuzzleEnvironment:
    def __init__(self, seed: int = 0) -> None:
        self._engine: Gs01 | None = Gs01(seed=seed)
        self._total_turns = 0
        self._last_action_was_reset = False

    @property
    def _eng(self) -> Gs01:
        assert self._engine is not None
        return self._engine

    def reset(self) -> GameState:
        eng = self._eng
        eng.perform_action(ActionInput(id=GameAction.RESET))
        self._last_action_was_reset = True
        self._total_turns = 0
        return self._build_state()

    def get_actions(self) -> list[str]:
        if self._is_done():
            return ["reset"]
        return list(_VALID_ACTIONS)

    def step(self, action: str) -> StepResult:
        if action == "reset":
            return StepResult(
                state=self.reset(),
                reward=0.0,
                done=self._is_done(),
                info=self._build_info(),
            )

        self._last_action_was_reset = False

        if self._is_done():
            return StepResult(
                state=self._build_state(),
                reward=0.0,
                done=True,
                info=self._build_info(),
            )

        ga = _ACTION_MAP.get(action)
        if ga is None:
            return StepResult(
                state=self._build_state(),
                reward=0.0,
                done=self._is_done(),
                info=self._build_info(),
            )

        level_before = self._eng._current_level_index
        self._eng.perform_action(ActionInput(id=ga))
        self._total_turns += 1
        level_after = self._eng._current_level_index
        total_levels = len(self._eng._levels)

        if self._eng._state == EngineState.WIN or level_after != level_before:
            reward = 1.0 / total_levels
        else:
            reward = 0.0

        return StepResult(
            state=self._build_state(),
            reward=reward,
            done=self._is_done(),
            info=self._build_info(),
        )

    def is_done(self) -> bool:
        return self._is_done()

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError(f"Unsupported render mode: {mode}")
        index_grid = self._eng.camera.render(self._eng.current_level.get_sprites())
        h, w = index_grid.shape[:2]
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for idx, color in enumerate(ARC_PALETTE):
            mask = index_grid == idx
            rgb[mask] = color
        return rgb

    def close(self) -> None:
        self._engine = None

    def _is_done(self) -> bool:
        return self._eng._state in (EngineState.GAME_OVER, EngineState.WIN)

    def _is_level_cleared(self) -> bool:
        return self._eng._state == EngineState.WIN

    def _build_state(self) -> GameState:
        done = self._is_done()
        canvas = self._eng.build_canvas()
        index_grid = self._eng.camera.render(self._eng.current_level.get_sprites())
        return GameState(
            text_observation=self._build_text(canvas),
            image_observation=_make_png(index_grid),
            valid_actions=None if done else list(_VALID_ACTIONS),
            turn=self._total_turns,
            metadata=self._build_info(),
        )

    def _build_text(self, canvas: list[list[int]]) -> str:
        e = self._eng
        header_parts = [
            f"level:{e.level_index + 1}/{len(e._levels)}",
            f"moves:{e.max_moves - e.move_count}/{e.max_moves}",
            f"lives:{e._lives}/{MAX_LIVES}",
        ]
        header = " ".join(header_parts)
        color_names = {
            COLOR_SWITCH_ACTIVE: "switch_on",
            COLOR_BACKGROUND: "empty",
            COLOR_BOUNDARY: "border",
            COLOR_WALL: "wall",
            COLOR_PLAYER: "player",
            COLOR_SHADOW: "shadow",
            COLOR_HAZARD: "hazard",
            COLOR_TELEPORTER: "teleporter",
            COLOR_KEY: "key",
            COLOR_SWITCH: "switch",
            COLOR_GATE: "gate",
            COLOR_TOGGLE_GATE: "toggle_gate",
            COLOR_TOGGLE_GATE_OPEN: "toggle_open",
            COLOR_GOAL: "goal",
            COLOR_SHADOW_GOAL: "shadow_goal",
            COLOR_TRAP_FLOOR: "trap",
            COLOR_MOVABLE: "block",
            RED: "life_on",
            BLUE: "move_bar",
            BLUE_LIGHT: "move_bar_empty",
        }
        grid_lines = []
        for row in canvas:
            grid_lines.append(" ".join(color_names.get(c, str(c)) for c in row))
        return header + "\n" + "\n".join(grid_lines)

    def _build_info(self) -> dict[str, Any]:
        e = self._eng
        return {
            "total_levels": len(e._levels),
            "level": e.level_index,
            "lives": e._lives,
            "moves_remaining": e.max_moves - e.move_count,
            "max_moves": e.max_moves,
        }


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
        if self._env is None:
            return np.zeros((self.OBS_HEIGHT, self.OBS_WIDTH, 3), dtype=np.uint8)
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
