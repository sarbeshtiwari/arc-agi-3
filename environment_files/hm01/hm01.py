from __future__ import annotations

import heapq
import random
import struct
import zlib
from collections import deque
from collections.abc import Generator
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from arcengine import (
    ARCBaseGame,
    ActionInput,
    Camera,
    GameAction,
    Level,
    Sprite,
)
from arcengine.enums import BlockingMode
from gymnasium import spaces

Position = tuple[int, int]


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

TILE: int = 3
GRID_W: int = 20
GRID_H: int = 20
HUD_ROWS: int = 1
MAX_LIVES: int = 3
ENEMY_TICK: int = 2
PATHFIND_INF: float = float("inf")
AMBUSH_RANGE: int = 3
HUD_BAR_WIDTH: int = GRID_W - (MAX_LIVES + 1)

MOVE_DELTAS: dict[int, tuple[int, int]] = {
    1: (-1, 0),
    2: (1, 0),
    3: (0, -1),
    4: (0, 1),
}

PATROL_CYCLE: list[tuple[int, int]] = [
    (0, 1),
    (1, 0),
    (0, -1),
    (-1, 0),
]

_HIDDEN_FLOOR_PX: list[list[int]] = [
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 0],
]

_GOAL_PX: list[list[int]] = [
    [-1, 4, -1],
    [4, 7, 4],
    [-1, 4, -1],
]

PIXELS: dict[str, list[list[int]]] = {
    "floor": [row[:] for row in _HIDDEN_FLOOR_PX],
    "wall": [
        [5, 5, 5],
        [5, 6, 5],
        [5, 5, 5],
    ],
    "player": [
        [-1, 2, -1],
        [2, 8, 2],
        [-1, 2, -1],
    ],
    "enemy": [
        [-1, 3, -1],
        [3, 9, 3],
        [-1, 3, -1],
    ],
    "goal": [row[:] for row in _GOAL_PX],
    "door_closed": [
        [10, 10, 10],
        [10, 6, 10],
        [10, 10, 10],
    ],
    "door_open": [
        [0, 0, 0],
        [0, 10, 0],
        [0, 0, 0],
    ],
    "switch_off": [
        [-1, 8, -1],
        [8, 0, 8],
        [-1, 8, -1],
    ],
    "switch_on": [
        [-1, 7, -1],
        [7, 4, 7],
        [-1, 7, -1],
    ],
    "fake_goal": [row[:] for row in _GOAL_PX],
    "fake_revealed": [
        [-1, 9, -1],
        [9, 0, 9],
        [-1, 9, -1],
    ],
    "ambush_hidden": [row[:] for row in _HIDDEN_FLOOR_PX],
    "ambush_active": [
        [-1, 3, -1],
        [3, 2, 3],
        [-1, 3, -1],
    ],
    "trap_hidden": [row[:] for row in _HIDDEN_FLOOR_PX],
    "trap_active": [
        [9, 9, 9],
        [9, 5, 9],
        [9, 9, 9],
    ],
    "key_uncollected": [
        [-1, 6, -1],
        [6, 4, 6],
        [-1, 6, -1],
    ],
    "key_collected": [row[:] for row in _HIDDEN_FLOOR_PX],
    "hud_bg": [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ],
    "hud_spacer": [
        [0, 8, 0],
        [0, 8, 0],
        [0, 0, 0],
    ],
    "hud_life_full": [
        [6, 6, 6],
        [6, 6, 6],
        [0, 0, 0],
    ],
    "hud_life_empty": [
        [5, 5, 5],
        [5, 5, 5],
        [0, 0, 0],
    ],
    "hud_bar_filled": [
        [3, 3, 3],
        [3, 3, 3],
        [0, 0, 0],
    ],
    "hud_bar_remain": [
        [1, 1, 1],
        [1, 1, 1],
        [0, 0, 0],
    ],
}

_L1 = [
    "WWWWWWWWWWWWWWWWWWWW",
    "WP...W.............W",
    "W.WW.W.WWWWWWWWWWW.W",
    "W.WW.W.............W",
    "W....WWWWWWWWWWWWW.W",
    "W.WW...............W",
    "W.WWWWWWWWWWWW.WWWWW",
    "W..............E...W",
    "WWWWWWWW.WWWWWWWWW.W",
    "W..................W",
    "W.WWWWWW.WWWWWWWWWWW",
    "W........E.........W",
    "WWWWWWWW.WWWWWWWWW.W",
    "W..................W",
    "W.WWWWWWWWWWWWWWWW.W",
    "W..................W",
    "WWWWWWWWWWWWWWWWWW.W",
    "W..................W",
    "W................G.W",
    "WWWWWWWWWWWWWWWWWWWW",
]

_L2 = [
    "WWWWWWWWWWWWWWWWWWWW",
    "WP.......W.........W",
    "W.WWWWWW.W.WWWWWWW.W",
    "W.W......W.......W.W",
    "W.W.WWWWWWWWWWWW.W.W",
    "W.W..............W.W",
    "W.WWWWWWWWWWWWWWWW.W",
    "W.........E........W",
    "WWWWWWWW.WWWWWWWWW.W",
    "W........W.........W",
    "W.WWWWWW.W.WWWWWWWWW",
    "W......W.W.........W",
    "WWWWWW.W.WWWWWWWWW.W",
    "W......W...........W",
    "W.WWWWWWWW.WWWWWWW.W",
    "W......S...........W",
    "W.WWWWWWWWWWWWWWWWWW",
    "W........D.........W",
    "W........W...E...G.W",
    "WWWWWWWWWWWWWWWWWWWW",
]

_L3 = [
    "WWWWWWWWWWWWWWWWWWWW",
    "W.P....W...........W",
    "W.WWWW.W.WWWWWWWWW.W",
    "W.W....W.W.........W",
    "W.W.WWWW.W.WWWWWWWWW",
    "W.W......W.........W",
    "W.WWWWWWWWWWWWWWWW.W",
    "W..................W",
    "WWWWWWWW.WWWWWWWWW.W",
    "W........A.........W",
    "W.WWWWWWWWWWWWWWWW.W",
    "W........A.........W",
    "W.WWWWWW.WWWWWWWWWWW",
    "W........W....A....W",
    "WWWWWWWW.WWWWWWWWW.W",
    "W.........A........W",
    "W.WWWWWWWWWWWWWWWW.W",
    "W......F...F...F...W",
    "W.................GW",
    "WWWWWWWWWWWWWWWWWWWW",
]

_L4 = [
    "WWWWWWWWWWWWWWWWWWWW",
    "WP.......W.........W",
    "W.WWWWWW.W.WWWWWWW.W",
    "W.W......W.......W.W",
    "W.W.WWWW.WWWWWWW.W.W",
    "W.W....W....K....W.W",
    "W.WWWW.WWWWWWWWWWW.W",
    "W......W.....S.....W",
    "W.WWWWWW.WWWWWWWWWWW",
    "W....A.............W",
    "WWWWWWWWWDWWWWWWWWWW",
    "W....E.............W",
    "W.WWWWWWW.WWWWWWWW.W",
    "W.......W........W.W",
    "W.WWWWW.WSWWWWWW.W.W",
    "W.......W........W.W",
    "WWWWWWWWWWWDWWWWWWWW",
    "W......F..F..F.....W",
    "W..........E.....G.W",
    "WWWWWWWWWWWWWWWWWWWW",
]

LEVEL_LAYOUTS: list[list[str]] = [_L1, _L2, _L3, _L4]

PLAYER_START_POSITIONS: list[list[Position]] = [
    [(1, 1), (1, 3), (4, 1), (5, 4)],
    [(1, 1), (1, 3), (1, 5), (5, 3)],
    [(1, 2), (1, 1), (1, 4), (5, 3)],
    [(1, 1), (1, 3), (1, 5), (3, 3)],
]

LEVEL_DATA: list[dict[str, Any]] = [
    {
        "enemy_type": "patrol",
        "enemy_tick": ENEMY_TICK,
        "max_moves": 320,
    },
    {
        "enemy_type": "bfs",
        "enemy_tick": ENEMY_TICK,
        "max_moves": 400,
    },
    {
        "enemy_type": "ambush",
        "enemy_tick": ENEMY_TICK,
        "max_moves": 520,
    },
    {
        "enemy_type": "astar",
        "enemy_tick": ENEMY_TICK,
        "max_moves": 640,
    },
]


def _make_sprite(
    kind: str,
    row: int,
    col: int,
    *,
    layer: int = 0,
    name: str = "",
) -> Sprite:
    px = PIXELS.get(kind, PIXELS["floor"])
    return Sprite(
        pixels=px,
        name=name or f"{kind}_{row}_{col}",
        x=col * TILE,
        y=row * TILE,
        layer=layer,
        blocking=BlockingMode.NOT_BLOCKED,
        visible=True,
        collidable=False,
    )


def _build_maze_sprites(layout: list[str]) -> tuple[list[Sprite], int]:
    sprites: list[Sprite] = []
    enemy_count = 0

    for r, row_str in enumerate(layout):
        for c, ch in enumerate(row_str):
            if ch != "W":
                sprites.append(_make_sprite("floor", r, c, layer=0))

            if ch == "W":
                sprites.append(
                    _make_sprite("wall", r, c, layer=1, name=f"wall_{r}_{c}")
                )
            elif ch == "G":
                sprites.append(_make_sprite("goal", r, c, layer=1, name="goal"))
            elif ch == "P":
                sprites.append(_make_sprite("player", r, c, layer=3, name="player"))
            elif ch == "E":
                sprites.append(
                    _make_sprite("enemy", r, c, layer=3, name=f"enemy_{enemy_count}")
                )
                enemy_count += 1
            elif ch == "D":
                sprites.append(
                    _make_sprite("door_closed", r, c, layer=2, name=f"door_{r}_{c}")
                )
            elif ch == "S":
                sprites.append(
                    _make_sprite("switch_off", r, c, layer=2, name=f"switch_{r}_{c}")
                )
            elif ch == "F":
                sprites.append(
                    _make_sprite("fake_goal", r, c, layer=1, name=f"fake_{r}_{c}")
                )
            elif ch == "A":
                sprites.append(
                    _make_sprite("ambush_hidden", r, c, layer=1, name=f"ambush_{r}_{c}")
                )
            elif ch == "T":
                sprites.append(
                    _make_sprite("trap_hidden", r, c, layer=1, name=f"trap_{r}_{c}")
                )
            elif ch == "K":
                sprites.append(
                    _make_sprite("key_uncollected", r, c, layer=1, name=f"key_{r}_{c}")
                )

    return sprites, enemy_count


def _build_hud_sprites(hud_row: int) -> list[Sprite]:
    sprites: list[Sprite] = []

    for c in range(GRID_W):
        sprites.append(_make_sprite("hud_bg", hud_row, c, layer=0))

    for i in range(MAX_LIVES):
        sprites.append(
            _make_sprite("hud_life_full", hud_row, i, layer=1, name=f"hud_life_{i}")
        )

    sprites.append(_make_sprite("hud_spacer", hud_row, MAX_LIVES, layer=1))

    bar_start = GRID_W - HUD_BAR_WIDTH
    for c in range(bar_start, GRID_W):
        sprites.append(
            _make_sprite(
                "hud_bar_filled", hud_row, c, layer=1, name=f"hud_bar_{c - bar_start}"
            )
        )

    return sprites


def _build_level(index: int) -> Level:
    layout = LEVEL_LAYOUTS[index]
    data = dict(LEVEL_DATA[index])
    sprites, _enemy_count = _build_maze_sprites(layout)
    sprites.extend(_build_hud_sprites(GRID_H))
    return Level(
        sprites=sprites,
        grid_size=(GRID_W * TILE, (GRID_H + HUD_ROWS) * TILE),
        data=data,
        name=f"level_{index + 1}",
    )


def _build_all_levels() -> list[Level]:
    return [_build_level(i) for i in range(len(LEVEL_LAYOUTS))]


def _manhattan(a: Position, b: Position) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _reconstruct(
    parent: dict[Position, Position | None],
    target: Position,
) -> list[Position]:
    path: list[Position] = []
    cur: Position | None = target
    while cur is not None:
        path.append(cur)
        cur = parent.get(cur)
    path.reverse()
    return path


class Hm01(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self._rng = random.Random(seed)
        self._undo_stack: list[dict[str, Any]] = []
        self._hm_lives: int = MAX_LIVES
        self._hm_is_new_level: bool = True
        self._hm_transition_pending: str | None = None
        self._hm_move_limit: int = 0
        self._hm_moves_used: int = 0
        self._hm_turn: int = 0
        self._hm_rows: int = 0
        self._hm_cols: int = 0
        self._hm_walls: set[Position] = set()
        self._hm_player: Position = (0, 0)
        self._hm_goal: Position = (0, 0)
        self._hm_enemies: list[Position] = []
        self._hm_enemy_type: str = "none"
        self._hm_enemy_tick: int = 1
        self._hm_doors: dict[Position, bool] = {}
        self._hm_switches: dict[Position, bool] = {}
        self._hm_fakes: dict[Position, bool] = {}
        self._hm_patrol_idx: list[int] = []
        self._hm_patrol_steps: list[int] = []
        self._hm_ambush_positions: list[Position] = []
        self._hm_ambush_active: list[bool] = []
        self._hm_ambush_sprite_map: dict[int, str] = {}
        self._hm_traps: dict[Position, bool] = {}
        self._hm_keys: dict[Position, bool] = {}
        self._hm_switch_order: list[Position] = []
        self._hm_door_order: list[Position] = []
        self._hm_switches_activated: int = 0
        levels = _build_all_levels()
        cam_w = GRID_W * TILE
        cam_h = (GRID_H + HUD_ROWS) * TILE
        camera = Camera(
            x=0,
            y=0,
            width=cam_w,
            height=cam_h,
            background=0,
            letter_box=0,
            interfaces=[],
        )
        super().__init__(
            game_id="hm01",
            levels=levels,
            camera=camera,
            available_actions=[0, 1, 2, 3, 4, 5, 7],
        )

    def on_set_level(self, level: Level) -> None:
        if self._hm_is_new_level:
            self._hm_lives = MAX_LIVES
        self._hm_is_new_level = False
        self._hm_transition_pending = None
        self._undo_stack = []
        self._hm_parse_level(level)
        self._hm_update_hud()

    def step(self) -> None:
        if self._hm_process_transition():
            return
        self._hm_process_action()

    def handle_reset(self) -> None:
        self._hm_lives = MAX_LIVES
        self._hm_is_new_level = True
        self._undo_stack = []
        super().handle_reset()

    def _hm_save_state(self) -> dict[str, Any]:
        return {
            "player": self._hm_player,
            "enemies": list(self._hm_enemies),
            "patrol_idx": list(self._hm_patrol_idx),
            "patrol_steps": list(self._hm_patrol_steps),
            "doors": dict(self._hm_doors),
            "switches": dict(self._hm_switches),
            "switches_activated": self._hm_switches_activated,
            "fakes": dict(self._hm_fakes),
            "ambush_positions": list(self._hm_ambush_positions),
            "ambush_active": list(self._hm_ambush_active),
            "ambush_sprite_map": dict(self._hm_ambush_sprite_map),
            "traps": dict(self._hm_traps),
            "keys": dict(self._hm_keys),
            "walls": set(self._hm_walls),
            "moves_used": self._hm_moves_used,
            "turn": self._hm_turn,
            "transition_pending": self._hm_transition_pending,
            "move_limit": self._hm_move_limit,
        }

    def _hm_restore_state(self, state: dict[str, Any]) -> None:
        self._hm_player = state["player"]
        self._hm_enemies = list(state["enemies"])
        self._hm_patrol_idx = list(state["patrol_idx"])
        self._hm_patrol_steps = list(state["patrol_steps"])
        self._hm_doors = dict(state["doors"])
        self._hm_switches = dict(state["switches"])
        self._hm_switches_activated = state["switches_activated"]
        self._hm_fakes = dict(state["fakes"])
        self._hm_ambush_positions = list(state["ambush_positions"])
        self._hm_ambush_active = list(state["ambush_active"])
        self._hm_ambush_sprite_map = dict(state["ambush_sprite_map"])
        self._hm_traps = dict(state["traps"])
        self._hm_keys = dict(state["keys"])
        self._hm_walls = set(state["walls"])
        self._hm_moves_used = state["moves_used"]
        self._hm_turn = state["turn"]
        self._hm_transition_pending = state["transition_pending"]
        self._hm_move_limit = state["move_limit"]
        self._hm_restore_sprites()

    def _hm_restore_sprites(self) -> None:
        for pos, activated in self._hm_traps.items():
            kind = "trap_active" if activated else "trap_hidden"
            self._hm_swap_sprite(f"trap_{pos[0]}_{pos[1]}", kind)
        for pos, collected in self._hm_keys.items():
            kind = "key_collected" if collected else "key_uncollected"
            self._hm_swap_sprite(f"key_{pos[0]}_{pos[1]}", kind)
        for pos, opened in self._hm_doors.items():
            kind = "door_open" if opened else "door_closed"
            self._hm_swap_sprite(f"door_{pos[0]}_{pos[1]}", kind)
        for pos, activated in self._hm_switches.items():
            kind = "switch_on" if activated else "switch_off"
            self._hm_swap_sprite(f"switch_{pos[0]}_{pos[1]}", kind)
        for pos, revealed in self._hm_fakes.items():
            kind = "fake_revealed" if revealed else "fake_goal"
            self._hm_swap_sprite(f"fake_{pos[0]}_{pos[1]}", kind)
        for i, pos in enumerate(self._hm_ambush_positions):
            kind = "ambush_active" if self._hm_ambush_active[i] else "ambush_hidden"
            self._hm_swap_sprite(f"ambush_{pos[0]}_{pos[1]}", kind)
        pr, pc = self._hm_player
        self._hm_sprite_pos("player", pr, pc)
        self._hm_sync_enemies()

    def _hm_process_transition(self) -> bool:
        pending = self._hm_transition_pending
        if pending is None:
            return False
        self._hm_transition_pending = None
        if pending in ("fail", "reset"):
            self._hm_lives -= 1
            if self._hm_lives <= 0:
                self.lose()
                self.complete_action()
            else:
                self._hm_is_new_level = False
                self.level_reset()
                self._hm_finish_step()
            return True
        if pending == "win":
            self._hm_is_new_level = True
            self.next_level()
            self._hm_finish_step()
            return True
        return False

    def _hm_process_action(self) -> None:
        aid = self.action.id
        if aid == GameAction.RESET:
            self._hm_finish_step()
            return

        if aid == GameAction.ACTION7:
            if self._undo_stack:
                cur_moves = self._hm_moves_used
                cur_turn = self._hm_turn
                self._hm_restore_state(self._undo_stack.pop())
                self._hm_moves_used = cur_moves + 1
                self._hm_turn = cur_turn + 1
                self._hm_tick_enemies()
                if not self._hm_check_capture():
                    self._hm_check_end_conditions()
            else:
                self._hm_moves_used += 1
                self._hm_turn += 1
                self._hm_check_end_conditions()
            self._hm_finish_step()
            return

        if aid == GameAction.ACTION5:
            self._undo_stack.append(self._hm_save_state())
            self._hm_do_interact()
            self._hm_advance_turn()
            self._hm_finish_step()
            return

        dr, dc = 0, 0
        if aid == GameAction.ACTION1:
            dr, dc = -1, 0
        elif aid == GameAction.ACTION2:
            dr, dc = 1, 0
        elif aid == GameAction.ACTION3:
            dr, dc = 0, -1
        elif aid == GameAction.ACTION4:
            dr, dc = 0, 1

        if dr != 0 or dc != 0:
            self._undo_stack.append(self._hm_save_state())
            self._hm_do_move(dr, dc)
            self._hm_finish_step()
            return

        self._hm_advance_turn()
        self._hm_finish_step()

    def _hm_find_activatable_switch(self) -> Position | None:
        pr, pc = self._hm_player
        for dr, dc in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]:
            pos = (pr + dr, pc + dc)
            if pos in self._hm_switches and not self._hm_switches[pos]:
                try:
                    switch_idx = self._hm_switch_order.index(pos)
                except ValueError:
                    continue
                if switch_idx <= self._hm_switches_activated:
                    return pos
        return None

    def _hm_can_interact(self) -> bool:
        return self._hm_find_activatable_switch() is not None

    def _hm_check_end_conditions(self) -> None:
        if self._hm_transition_pending is not None:
            return

        if self._hm_player == self._hm_goal:
            if all(self._hm_keys.values()):
                self._hm_transition_pending = "win"
                return

        if self._hm_moves_used >= self._hm_move_limit:
            self._hm_transition_pending = "fail"
            return

        pr, pc = self._hm_player
        if not any(
            self._hm_walkable((pr + dr, pc + dc)) for dr, dc in MOVE_DELTAS.values()
        ):
            self._hm_transition_pending = "fail"

    def _hm_do_move(self, dr: int, dc: int) -> None:
        pr, pc = self._hm_player
        nr, nc = pr + dr, pc + dc

        self._hm_moves_used += 1
        self._hm_turn += 1

        if self._hm_walkable((nr, nc)):
            self._hm_player = (nr, nc)
            self._hm_sprite_pos("player", nr, nc)
            self._hm_collect_key()

        if self._hm_check_trap():
            return
        if self._hm_check_capture():
            return
        self._hm_tick_enemies()
        if self._hm_check_capture():
            return
        self._hm_check_end_conditions()

    def _hm_advance_turn(self) -> None:
        self._hm_moves_used += 1
        self._hm_turn += 1
        self._hm_tick_enemies()
        if self._hm_check_capture():
            return
        self._hm_check_end_conditions()

    def _hm_do_interact(self) -> None:
        pos = self._hm_find_activatable_switch()
        if pos is None:
            return
        switch_idx = self._hm_switch_order.index(pos)
        self._hm_switches[pos] = True
        self._hm_switches_activated += 1
        self._hm_swap_sprite(f"switch_{pos[0]}_{pos[1]}", "switch_on")
        if switch_idx < len(self._hm_door_order):
            dpos = self._hm_door_order[switch_idx]
            if dpos in self._hm_doors and not self._hm_doors[dpos]:
                self._hm_doors[dpos] = True
                self._hm_walls.discard(dpos)
                self._hm_swap_sprite(f"door_{dpos[0]}_{dpos[1]}", "door_open")

    def _bfs_path(self, start: Position, target: Position) -> list[Position]:
        if start == target:
            return [start]
        queue: deque[Position] = deque([start])
        visited: set[Position] = {start}
        parent: dict[Position, Position | None] = {start: None}

        while queue:
            node = queue.popleft()
            if node == target:
                break
            for nbr in self._hm_neighbors(node):
                if nbr not in visited:
                    visited.add(nbr)
                    parent[nbr] = node
                    queue.append(nbr)

        if target not in parent:
            return [start]
        return _reconstruct(parent, target)

    def _astar_path(self, start: Position, target: Position) -> list[Position]:
        if start == target:
            return [start]

        insertion_order = 0
        came_from: dict[Position, Position | None] = {start: None}
        g_score: dict[Position, float] = {start: 0.0}
        f_start = _manhattan(start, target)
        heap: list[tuple[float, int, Position]] = [
            (float(f_start), insertion_order, start)
        ]
        closed: set[Position] = set()

        while heap:
            _f, _c, cur = heapq.heappop(heap)
            if cur == target:
                return _reconstruct(came_from, target)
            if cur in closed:
                continue
            closed.add(cur)
            for nbr in self._hm_neighbors(cur):
                if nbr in closed:
                    continue
                tentative_g = g_score[cur] + 1
                if tentative_g < g_score.get(nbr, PATHFIND_INF):
                    came_from[nbr] = cur
                    g_score[nbr] = tentative_g
                    f_nbr = tentative_g + _manhattan(nbr, target)
                    insertion_order += 1
                    heapq.heappush(heap, (f_nbr, insertion_order, nbr))

        return [start]

    def _hm_tick_enemies(self) -> None:
        if self._hm_enemy_type != "none" and self._hm_turn % self._hm_enemy_tick == 0:
            if self._hm_enemy_type == "patrol":
                self._hm_move_patrol()
            else:
                self._hm_move_chase()
        self._hm_check_ambush_activation()

    def _hm_check_ambush_activation(self) -> None:
        for i, apos in enumerate(self._hm_ambush_positions):
            if self._hm_ambush_active[i]:
                continue
            if _manhattan(self._hm_player, apos) <= AMBUSH_RANGE:
                self._hm_ambush_active[i] = True
                enemy_idx = len(self._hm_enemies)
                self._hm_enemies.append(apos)
                if self._hm_enemy_type == "patrol":
                    self._hm_patrol_idx.append(0)
                    self._hm_patrol_steps.append(0)
                r, c = apos
                sprite_name = f"ambush_{r}_{c}"
                self._hm_ambush_sprite_map[enemy_idx] = sprite_name
                self._hm_swap_sprite(sprite_name, "ambush_active")

    def _hm_move_patrol(self) -> None:
        moved: list[Position] = []
        for i, epos in enumerate(self._hm_enemies):
            idx = self._hm_patrol_idx[i]
            steps = self._hm_patrol_steps[i]

            dr, dc = PATROL_CYCLE[idx % 4]
            nr, nc = epos[0] + dr, epos[1] + dc

            if self._hm_walkable((nr, nc)):
                self._hm_patrol_steps[i] = steps + 1
                if self._hm_patrol_steps[i] >= 3:
                    self._hm_patrol_idx[i] = (idx + 1) % 4
                    self._hm_patrol_steps[i] = 0
                moved.append((nr, nc))
            else:
                self._hm_patrol_idx[i] = (idx + 1) % 4
                self._hm_patrol_steps[i] = 0
                moved.append(epos)

        self._hm_enemies = moved
        self._hm_sync_enemies()

    def _hm_move_chase(self) -> None:
        moved: list[Position] = []
        occupied: set[Position] = set(self._hm_enemies)

        for epos in self._hm_enemies:
            occupied.discard(epos)
            if self._hm_enemy_type == "astar":
                path = self._astar_path(epos, self._hm_player)
            else:
                path = self._bfs_path(epos, self._hm_player)

            nxt = epos
            if len(path) > 1:
                cand = path[1]
                if cand == self._hm_player or cand not in occupied:
                    nxt = cand

            moved.append(nxt)
            occupied.add(nxt)

        self._hm_enemies = moved
        self._hm_sync_enemies()

    def _hm_neighbors(
        self,
        pos: Position,
    ) -> Generator[Position, None, None]:
        r, c = pos
        for dr, dc in MOVE_DELTAS.values():
            nr, nc = r + dr, c + dc
            if self._hm_walkable((nr, nc)):
                yield (nr, nc)

    def _hm_walkable(self, pos: Position) -> bool:
        r, c = pos
        if r < 0 or c < 0 or r >= self._hm_rows or c >= self._hm_cols:
            return False
        return pos not in self._hm_walls

    def _hm_check_capture(self) -> bool:
        if self._hm_player in self._hm_enemies:
            self._hm_transition_pending = "reset"
            return True
        return False

    def _hm_check_trap(self) -> bool:
        pos = self._hm_player
        if pos in self._hm_traps and not self._hm_traps[pos]:
            self._hm_traps[pos] = True
            self._hm_walls.add(pos)
            self._hm_swap_sprite(f"trap_{pos[0]}_{pos[1]}", "trap_active")
            self._hm_transition_pending = "reset"
            return True
        return False

    def _hm_collect_key(self) -> None:
        pos = self._hm_player
        if pos in self._hm_keys and not self._hm_keys[pos]:
            self._hm_keys[pos] = True
            self._hm_swap_sprite(f"key_{pos[0]}_{pos[1]}", "key_collected")

    def _hm_sprite_pos(self, name: str, row: int, col: int) -> None:
        sprites = self.current_level.get_sprites_by_name(name)
        if sprites:
            sprites[0].set_position(col * TILE, row * TILE)

    def _hm_sync_enemies(self) -> None:
        for i, (er, ec) in enumerate(self._hm_enemies):
            name = self._hm_ambush_sprite_map.get(i, f"enemy_{i}")
            sprites = self.current_level.get_sprites_by_name(name)
            if sprites:
                sprites[0].set_position(ec * TILE, er * TILE)

    def _hm_swap_sprite(self, sprite_name: str, new_kind: str) -> None:
        sprites = self.current_level.get_sprites_by_name(sprite_name)
        if sprites:
            sprites[0].pixels = np.array(PIXELS[new_kind], dtype=np.int8)

    def _hm_update_hud(self) -> None:
        for i in range(MAX_LIVES):
            kind = "hud_life_full" if i < self._hm_lives else "hud_life_empty"
            sprites = self.current_level.get_sprites_by_name(f"hud_life_{i}")
            if sprites:
                sprites[0].pixels = np.array(PIXELS[kind], dtype=np.int8)

        bar_width = HUD_BAR_WIDTH
        limit = max(self._hm_move_limit, 1)
        remaining = max(self._hm_move_limit - self._hm_moves_used, 0)
        filled = min((remaining * bar_width + limit - 1) // limit, bar_width)

        for c in range(bar_width):
            kind = "hud_bar_filled" if c < filled else "hud_bar_remain"
            sprites = self.current_level.get_sprites_by_name(f"hud_bar_{c}")
            if sprites:
                sprites[0].pixels = np.array(PIXELS[kind], dtype=np.int8)

    def _hm_finish_step(self) -> None:
        self._hm_update_hud()
        self.complete_action()

    def _hm_parse_layout_grid(self, layout: list[str]) -> None:
        for r, row_str in enumerate(layout):
            for c, ch in enumerate(row_str):
                pos = (r, c)
                if ch == "W":
                    self._hm_walls.add(pos)
                elif ch == "P":
                    self._hm_player = pos
                elif ch == "G":
                    self._hm_goal = pos
                elif ch == "E":
                    self._hm_enemies.append(pos)
                    self._hm_patrol_idx.append(0)
                    self._hm_patrol_steps.append(0)
                elif ch == "D":
                    self._hm_doors[pos] = False
                    self._hm_walls.add(pos)
                    self._hm_door_order.append(pos)
                elif ch == "S":
                    self._hm_switches[pos] = False
                    self._hm_switch_order.append(pos)
                elif ch == "F":
                    self._hm_fakes[pos] = False
                elif ch == "A":
                    self._hm_ambush_positions.append(pos)
                    self._hm_ambush_active.append(False)
                elif ch == "T":
                    self._hm_traps[pos] = False
                elif ch == "K":
                    self._hm_keys[pos] = False

    def _hm_clear_state(self, layout: list[str]) -> None:
        self._hm_rows = len(layout)
        self._hm_cols = len(layout[0]) if layout else 0
        self._hm_walls = set()
        self._hm_enemies = []
        self._hm_player = (0, 0)
        self._hm_goal = (0, 0)
        self._hm_doors = {}
        self._hm_switches = {}
        self._hm_fakes = {}
        self._hm_patrol_idx = []
        self._hm_patrol_steps = []
        self._hm_ambush_positions = []
        self._hm_ambush_active = []
        self._hm_ambush_sprite_map = {}
        self._hm_traps = {}
        self._hm_keys = {}
        self._hm_switch_order = []
        self._hm_door_order = []
        self._hm_switches_activated = 0

    def _hm_parse_level(self, level: Level) -> None:
        layout = LEVEL_LAYOUTS[self._current_level_index]
        self._hm_clear_state(layout)
        self._hm_parse_layout_grid(layout)
        start = self._rng.choice(PLAYER_START_POSITIONS[self._current_level_index])
        self._hm_player = start
        self._hm_reset_sprites()
        self._hm_load_level_config(level)

    def _hm_reset_sprites(self) -> None:
        for pos in self._hm_traps:
            self._hm_swap_sprite(f"trap_{pos[0]}_{pos[1]}", "trap_hidden")
        for pos in self._hm_keys:
            self._hm_swap_sprite(f"key_{pos[0]}_{pos[1]}", "key_uncollected")
        for pos in self._hm_doors:
            self._hm_swap_sprite(f"door_{pos[0]}_{pos[1]}", "door_closed")
        for pos in self._hm_switches:
            self._hm_swap_sprite(f"switch_{pos[0]}_{pos[1]}", "switch_off")
        for pos in self._hm_fakes:
            self._hm_swap_sprite(f"fake_{pos[0]}_{pos[1]}", "fake_goal")
        for pos in self._hm_ambush_positions:
            self._hm_swap_sprite(f"ambush_{pos[0]}_{pos[1]}", "ambush_hidden")
        pr, pc = self._hm_player
        self._hm_sprite_pos("player", pr, pc)
        self._hm_sync_enemies()

    def _hm_load_level_config(self, level: Level) -> None:
        enemy_type = level.get_data("enemy_type")
        self._hm_enemy_type = str(enemy_type) if enemy_type is not None else "none"

        tick = level.get_data("enemy_tick")
        self._hm_enemy_tick = int(tick) if tick is not None else 1

        moves = level.get_data("max_moves")
        limit = int(moves) if moves is not None else 80
        self._hm_move_limit = limit
        self._hm_moves_used = 0
        self._hm_turn = 0


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    chunk = chunk_type + data
    return (
        struct.pack(">I", len(data))
        + chunk
        + struct.pack(">I", zlib.crc32(chunk) & 0xFFFFFFFF)
    )


def _encode_png(rgb: np.ndarray) -> bytes:
    h, w = rgb.shape[0], rgb.shape[1]
    raw = b""
    for y in range(h):
        raw += b"\x00" + rgb[y].tobytes()
    compressed = zlib.compress(raw)
    out = b"\x89PNG\r\n\x1a\n"
    out += _png_chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    out += _png_chunk(b"IDAT", compressed)
    out += _png_chunk(b"IEND", b"")
    return out


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
        self._engine = Hm01(seed=seed)
        self._total_turns = 0
        self._done = False
        self._game_won = False
        self._last_action_was_reset = False

    def reset(self) -> GameState:
        reset_input = ActionInput(id=GameAction.RESET)
        if self._game_won or self._last_action_was_reset:
            self._engine.perform_action(reset_input)
            self._engine.perform_action(reset_input)
        else:
            self._engine.perform_action(reset_input)
        self._last_action_was_reset = True
        self._done = False
        self._game_won = False
        self._total_turns = 0
        return self._build_game_state()

    def get_actions(self) -> List[str]:
        if self._done:
            return ["reset"]
        return list(self._VALID_ACTIONS)

    def is_done(self) -> bool:
        return self._done

    def step(self, action: str) -> StepResult:
        if action == "reset":
            state = self.reset()
            return StepResult(
                state=state, reward=0.0, done=False, info={"action": "reset"}
            )

        if action not in self._ACTION_MAP:
            return StepResult(
                state=self._build_game_state(),
                reward=0.0,
                done=self._done,
                info={"action": action, "reason": "invalid_action"},
            )

        self._last_action_was_reset = False
        self._total_turns += 1

        game_action = self._ACTION_MAP[action]
        info: Dict[str, Any] = {"action": action}

        level_before = self._engine.level_index

        action_input = ActionInput(id=game_action)
        frame = self._engine.perform_action(action_input, raw=True)

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
                state=self._build_game_state(done=True, info=info),
                reward=level_reward,
                done=True,
                info=info,
            )

        if game_over:
            self._done = True
            info["reason"] = "game_over"
            return StepResult(
                state=self._build_game_state(done=True, info=info),
                reward=0.0,
                done=True,
                info=info,
            )

        reward = 0.0
        if self._engine.level_index != level_before:
            reward = level_reward
            info["reason"] = "level_complete"

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
        level_idx = e.level_index
        total_levels = len(e._levels)

        bar_width = HUD_BAR_WIDTH
        limit = max(e._hm_move_limit, 1)
        remaining = max(e._hm_move_limit - e._hm_moves_used, 0)
        filled = min((remaining * bar_width + limit - 1) // limit, bar_width)
        bar = "#" * filled + "." * (bar_width - filled)

        observation = (
            f"Level {level_idx + 1}/{total_levels} | "
            f"Lives: {e._hm_lives}/{MAX_LIVES} | "
            f"Steps: [{bar}] "
            f"{e._hm_moves_used}/{e._hm_move_limit}"
        )
        if e._hm_keys:
            keys_remaining = sum(1 for v in e._hm_keys.values() if not v)
            observation += f" | Keys: {keys_remaining}"
        observation += f" | Turn: {self._total_turns}"
        return observation

    def _build_game_state(
        self, done: bool = False, info: Optional[Dict] = None
    ) -> GameState:
        engine = self._engine
        valid = self.get_actions() if not done else None

        image_bytes = None
        try:
            rgb = self.render()
            image_bytes = _encode_png(rgb)
        except Exception:
            pass

        return GameState(
            text_observation=self._build_text_observation(),
            image_observation=image_bytes,
            valid_actions=valid,
            turn=self._total_turns,
            metadata={
                "total_levels": len(engine._levels),
                "level_index": engine.level_index,
                "levels_completed": getattr(engine, "_score", 0),
                "game_over": getattr(getattr(engine, "_state", None), "name", "")
                == "GAME_OVER",
                "done": done,
                "info": info or {},
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
