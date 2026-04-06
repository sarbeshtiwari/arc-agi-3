import random
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
    RenderableUserDisplay,
    Sprite,
)


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
C_WALL = 4
C_FLOOR = 3
C_CURSOR = 11
C_PIPE = 0
C_PIPE_CONN = 10
C_SOURCE = 8
C_SINK = 14
C_SINK_OFF = 12
C_MOVE_FULL = 9
C_MOVE_EMPTY = 3
C_LIFE_ON = 8
C_LIFE_OFF = 4
C_UI_SEP = 4
C_ENEMY = 6

CELL = 8
CANVAS = 64
GRID_W = 6
GRID_H = 6
PLAY_X = 4
PLAY_Y = 4
UI_X = PLAY_X + GRID_W * CELL + 4

MAX_LIVES = 3

PIPE_OPENINGS = {
    0: (0, 2),
    1: (0, 1),
    2: (0, 1, 2),
}


def _get_openings(pipe_type, rotation):
    base = PIPE_OPENINGS[pipe_type]
    return set((d + rotation) % 4 for d in base)


def _opposite(d):
    return (d + 2) % 4


_DIR_DELTA = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}


def _solid(col, w=CELL, h=CELL):
    return [[col] * w for _ in range(h)]


def _floor_cell():
    b, c = C_WALL, C_FLOOR
    return [
        [b, b, b, b, b, b, b, b],
        [b, c, c, c, c, c, c, b],
        [b, c, c, c, c, c, c, b],
        [b, c, c, c, c, c, c, b],
        [b, c, c, c, c, c, c, b],
        [b, c, c, c, c, c, c, b],
        [b, c, c, c, c, c, c, b],
        [b, b, b, b, b, b, b, b],
    ]


def _cursor_ring():
    c, _ = C_CURSOR, -1
    return [
        [c, c, c, c, c, c, c, c],
        [c, _, _, _, _, _, _, c],
        [c, _, _, _, _, _, _, c],
        [c, _, _, _, _, _, _, c],
        [c, _, _, _, _, _, _, c],
        [c, _, _, _, _, _, _, c],
        [c, _, _, _, _, _, _, c],
        [c, c, c, c, c, c, c, c],
    ]


def _enemy_cell():
    e, _ = C_ENEMY, -1
    return [
        [e, _, _, _, _, _, _, e],
        [_, e, _, _, _, _, e, _],
        [_, _, e, _, _, e, _, _],
        [_, _, _, e, e, _, _, _],
        [_, _, _, e, e, _, _, _],
        [_, _, e, _, _, e, _, _],
        [_, e, _, _, _, _, e, _],
        [e, _, _, _, _, _, _, e],
    ]


def _source_cell():
    b, c, f = C_WALL, C_SOURCE, C_FLOOR
    return [
        [b, b, b, b, b, b, b, b],
        [b, f, f, c, c, f, f, b],
        [b, f, c, c, c, c, f, b],
        [b, c, c, c, c, c, c, b],
        [b, c, c, c, c, c, c, b],
        [b, f, c, c, c, c, f, b],
        [b, f, f, c, c, f, f, b],
        [b, b, b, b, b, b, b, b],
    ]


def _sink_off_cell():
    b, c, f = C_WALL, C_SINK_OFF, C_FLOOR
    return [
        [b, b, b, b, b, b, b, b],
        [b, c, f, f, f, f, c, b],
        [b, f, c, f, f, c, f, b],
        [b, f, f, c, c, f, f, b],
        [b, f, f, c, c, f, f, b],
        [b, f, c, f, f, c, f, b],
        [b, c, f, f, f, f, c, b],
        [b, b, b, b, b, b, b, b],
    ]


def _sink_on_cell():
    b, c, f = C_WALL, C_SINK, C_FLOOR
    return [
        [b, b, b, b, b, b, b, b],
        [b, c, c, c, c, c, c, b],
        [b, c, c, c, c, c, c, b],
        [b, c, c, c, c, c, c, b],
        [b, c, c, c, c, c, c, b],
        [b, c, c, c, c, c, c, b],
        [b, c, c, c, c, c, c, b],
        [b, b, b, b, b, b, b, b],
    ]


def _pipe_straight():
    b, c, _ = C_WALL, C_PIPE, C_FLOOR
    return [
        [b, b, _, c, c, _, b, b],
        [b, _, _, c, c, _, _, b],
        [_, _, _, c, c, _, _, _],
        [_, _, _, c, c, _, _, _],
        [_, _, _, c, c, _, _, _],
        [_, _, _, c, c, _, _, _],
        [b, _, _, c, c, _, _, b],
        [b, b, _, c, c, _, b, b],
    ]


def _pipe_elbow():
    b, c, _ = C_WALL, C_PIPE, C_FLOOR
    return [
        [b, b, _, c, c, _, b, b],
        [b, _, _, c, c, _, _, b],
        [_, _, _, c, c, c, c, _],
        [_, _, _, c, c, c, c, _],
        [_, _, _, _, _, _, _, _],
        [_, _, _, _, _, _, _, _],
        [b, _, _, _, _, _, _, b],
        [b, b, _, _, _, _, b, b],
    ]


def _pipe_tee():
    b, c, _ = C_WALL, C_PIPE, C_FLOOR
    return [
        [b, b, _, c, c, _, b, b],
        [b, _, _, c, c, _, _, b],
        [_, _, _, c, c, c, c, _],
        [_, _, _, c, c, c, c, _],
        [_, _, _, c, c, _, _, _],
        [_, _, _, c, c, _, _, _],
        [b, _, _, c, c, _, _, b],
        [b, b, _, c, c, _, b, b],
    ]


def _pipe_straight_conn():
    b, c, _ = C_WALL, C_PIPE_CONN, C_FLOOR
    return [
        [b, b, _, c, c, _, b, b],
        [b, _, _, c, c, _, _, b],
        [_, _, _, c, c, _, _, _],
        [_, _, _, c, c, _, _, _],
        [_, _, _, c, c, _, _, _],
        [_, _, _, c, c, _, _, _],
        [b, _, _, c, c, _, _, b],
        [b, b, _, c, c, _, b, b],
    ]


def _pipe_elbow_conn():
    b, c, _ = C_WALL, C_PIPE_CONN, C_FLOOR
    return [
        [b, b, _, c, c, _, b, b],
        [b, _, _, c, c, _, _, b],
        [_, _, _, c, c, c, c, _],
        [_, _, _, c, c, c, c, _],
        [_, _, _, _, _, _, _, _],
        [_, _, _, _, _, _, _, _],
        [b, _, _, _, _, _, _, b],
        [b, b, _, _, _, _, b, b],
    ]


def _pipe_tee_conn():
    b, c, _ = C_WALL, C_PIPE_CONN, C_FLOOR
    return [
        [b, b, _, c, c, _, b, b],
        [b, _, _, c, c, _, _, b],
        [_, _, _, c, c, c, c, _],
        [_, _, _, c, c, c, c, _],
        [_, _, _, c, c, _, _, _],
        [_, _, _, c, c, _, _, _],
        [b, _, _, c, c, _, _, b],
        [b, b, _, c, c, _, b, b],
    ]


sprites = {
    "bg": Sprite(
        pixels=_solid(C_BG),
        name="bg",
        visible=True,
        collidable=False,
        tags=["bg"],
        layer=-5,
    ),
    "floor": Sprite(
        pixels=_floor_cell(),
        name="floor",
        visible=True,
        collidable=False,
        tags=["floor"],
        layer=-3,
    ),
    "cursor": Sprite(
        pixels=_cursor_ring(),
        name="cursor",
        visible=True,
        collidable=False,
        tags=["cursor"],
        layer=5,
    ),
    "source": Sprite(
        pixels=_source_cell(),
        name="source",
        visible=True,
        collidable=False,
        tags=["source"],
        layer=1,
    ),
    "sink_off": Sprite(
        pixels=_sink_off_cell(),
        name="sink_off",
        visible=True,
        collidable=False,
        tags=["sink"],
        layer=1,
    ),
    "sink_on": Sprite(
        pixels=_sink_on_cell(),
        name="sink_on",
        visible=True,
        collidable=False,
        tags=["sink_on"],
        layer=1,
    ),
    "pipe_straight": Sprite(
        pixels=_pipe_straight(),
        name="pipe_straight",
        visible=True,
        collidable=False,
        tags=["pipe"],
        layer=2,
    ),
    "pipe_elbow": Sprite(
        pixels=_pipe_elbow(),
        name="pipe_elbow",
        visible=True,
        collidable=False,
        tags=["pipe"],
        layer=2,
    ),
    "pipe_tee": Sprite(
        pixels=_pipe_tee(),
        name="pipe_tee",
        visible=True,
        collidable=False,
        tags=["pipe"],
        layer=2,
    ),
    "pipe_straight_c": Sprite(
        pixels=_pipe_straight_conn(),
        name="pipe_straight_c",
        visible=True,
        collidable=False,
        tags=["pipe_c"],
        layer=2,
    ),
    "pipe_elbow_c": Sprite(
        pixels=_pipe_elbow_conn(),
        name="pipe_elbow_c",
        visible=True,
        collidable=False,
        tags=["pipe_c"],
        layer=2,
    ),
    "pipe_tee_c": Sprite(
        pixels=_pipe_tee_conn(),
        name="pipe_tee_c",
        visible=True,
        collidable=False,
        tags=["pipe_c"],
        layer=2,
    ),
    "enemy": Sprite(
        pixels=_enemy_cell(),
        name="enemy",
        visible=True,
        collidable=False,
        tags=["enemy"],
        layer=4,
    ),
}


def _px(col):
    return PLAY_X + col * CELL


def _py(row):
    return PLAY_Y + row * CELL


def _build_level(pipes, sources, sinks, moves, enemies=None):
    spr = []

    for ry in range(0, CANVAS, CELL):
        for rx in range(0, CANVAS, CELL):
            spr.append(sprites["bg"].clone().set_position(rx, ry))

    for row in range(GRID_H):
        for col in range(GRID_W):
            spr.append(sprites["floor"].clone().set_position(_px(col), _py(row)))

    for col, row in sources:
        spr.append(sprites["source"].clone().set_position(_px(col), _py(row)))

    for col, row in sinks:
        spr.append(sprites["sink_off"].clone().set_position(_px(col), _py(row)))

    pipe_names = {0: "pipe_straight", 1: "pipe_elbow", 2: "pipe_tee"}
    rot_angles = {0: 0, 1: 90, 2: 180, 3: 270}
    for col, row, ptype, rot in pipes:
        key = pipe_names[ptype]
        s = sprites[key].clone()
        s.set_position(_px(col), _py(row))
        if rot != 0:
            s.set_rotation(rot_angles[rot])
        spr.append(s)

    if enemies:
        for edata in enemies:
            path = edata["path"]
            if path:
                col, row = path[0]
                es = sprites["enemy"].clone()
                es.set_position(_px(col), _py(row))
                spr.append(es)

    return spr


_LEVELS = [
    Level(
        sprites=_build_level(
            pipes=[
                (1, 2, 0, 1),
                (2, 2, 2, 0),
                (3, 2, 0, 0),
                (2, 1, 0, 0),
                (2, 3, 0, 1),
            ],
            sources=[(0, 2)],
            sinks=[(2, 0), (4, 2)],
            moves=80,
            enemies=[
                {"path": [(4, 1), (3, 1), (3, 2), (3, 3), (4, 3), (4, 2), (4, 1)]},
            ],
        ),
        grid_size=(CANVAS, CANVAS),
        data={
            "pipes": [
                (1, 2, 0, 1),
                (2, 2, 2, 0),
                (3, 2, 0, 0),
                (2, 1, 0, 0),
                (2, 3, 0, 1),
            ],
            "sources": [(0, 2)],
            "sinks": [(2, 0), (4, 2)],
            "moves": 80,
            "enemies": [
                {"path": [(4, 1), (3, 1), (3, 2), (3, 3), (4, 3), (4, 2), (4, 1)]},
            ],
        },
        name="Level 1",
    ),
    Level(
        sprites=_build_level(
            pipes=[
                (1, 0, 2, 0),
                (2, 0, 0, 0),
                (1, 1, 0, 0),
                (1, 2, 1, 3),
                (2, 2, 0, 0),
                (3, 2, 2, 3),
                (3, 3, 0, 1),
            ],
            sources=[(0, 0)],
            sinks=[(3, 0), (4, 2), (3, 4)],
            moves=110,
            enemies=[
                {"path": [(4, 3), (3, 3), (3, 4), (4, 4), (4, 3)]},
            ],
        ),
        grid_size=(CANVAS, CANVAS),
        data={
            "pipes": [
                (1, 0, 2, 0),
                (2, 0, 0, 0),
                (1, 1, 0, 0),
                (1, 2, 1, 3),
                (2, 2, 0, 0),
                (3, 2, 2, 3),
                (3, 3, 0, 1),
            ],
            "sources": [(0, 0)],
            "sinks": [(3, 0), (4, 2), (3, 4)],
            "moves": 110,
            "enemies": [
                {"path": [(4, 3), (3, 3), (3, 4), (4, 4), (4, 3)]},
            ],
        },
        name="Level 2",
    ),
    Level(
        sprites=_build_level(
            pipes=[
                (1, 0, 2, 3),
                (2, 0, 0, 0),
                (0, 1, 1, 2),
                (1, 1, 2, 0),
                (1, 2, 0, 2),
                (1, 3, 2, 1),
                (2, 3, 0, 0),
                (3, 3, 2, 3),
                (4, 3, 0, 0),
                (3, 2, 0, 1),
                (3, 1, 1, 1),
            ],
            sources=[(0, 0), (0, 3)],
            sinks=[(3, 0), (5, 3), (3, 4)],
            moves=140,
            enemies=[
                {"path": [(2, 0), (3, 0), (4, 0), (3, 0), (2, 0)]},
                {"path": [(5, 4), (4, 3), (5, 3), (5, 4)]},
            ],
        ),
        grid_size=(CANVAS, CANVAS),
        data={
            "pipes": [
                (1, 0, 2, 3),
                (2, 0, 0, 0),
                (0, 1, 1, 2),
                (1, 1, 2, 0),
                (1, 2, 0, 2),
                (1, 3, 2, 1),
                (2, 3, 0, 0),
                (3, 3, 2, 3),
                (4, 3, 0, 0),
                (3, 2, 0, 1),
                (3, 1, 1, 1),
            ],
            "sources": [(0, 0), (0, 3)],
            "sinks": [(3, 0), (5, 3), (3, 4)],
            "moves": 140,
            "enemies": [
                {"path": [(2, 0), (3, 0), (4, 0), (3, 0), (2, 0)]},
                {"path": [(5, 4), (4, 3), (5, 3), (5, 4)]},
            ],
        },
        name="Level 3",
    ),
    Level(
        sprites=_build_level(
            pipes=[
                (1, 0, 2, 0),
                (2, 0, 0, 0),
                (3, 0, 0, 0),
                (1, 1, 0, 2),
                (1, 2, 2, 1),
                (2, 2, 0, 0),
                (3, 2, 2, 0),
                (4, 2, 0, 0),
                (3, 3, 0, 1),
                (3, 4, 1, 2),
            ],
            sources=[(0, 0), (0, 2)],
            sinks=[(4, 0), (5, 2), (4, 4)],
            moves=175,
            enemies=[
                {"path": [(3, 0), (4, 0), (4, 1), (3, 1), (3, 0)]},
                {"path": [(4, 2), (5, 2), (5, 3), (4, 3), (4, 2)]},
                {"path": [(3, 4), (4, 4), (4, 5), (3, 5), (3, 4)]},
            ],
        ),
        grid_size=(CANVAS, CANVAS),
        data={
            "pipes": [
                (1, 0, 2, 0),
                (2, 0, 0, 0),
                (3, 0, 0, 0),
                (1, 1, 0, 2),
                (1, 2, 2, 1),
                (2, 2, 0, 0),
                (3, 2, 2, 0),
                (4, 2, 0, 0),
                (3, 3, 0, 1),
                (3, 4, 1, 2),
            ],
            "sources": [(0, 0), (0, 2)],
            "sinks": [(4, 0), (5, 2), (4, 4)],
            "moves": 175,
            "enemies": [
                {"path": [(3, 0), (4, 0), (4, 1), (3, 1), (3, 0)]},
                {"path": [(4, 2), (5, 2), (5, 3), (4, 3), (4, 2)]},
                {"path": [(3, 4), (4, 4), (4, 5), (3, 5), (3, 4)]},
            ],
        },
        name="Level 4",
    ),
]


class Hud(RenderableUserDisplay):
    def __init__(self):
        self.lives = MAX_LIVES
        self.moves_left = 0
        self.moves_max = 1

    def update(self, lives, moves_left, moves_max):
        self.lives = lives
        self.moves_left = moves_left
        self.moves_max = max(1, moves_max)

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        sx = UI_X

        bar_top = 2
        bar_height = 40
        bar_filled = int(round(bar_height * self.moves_left / self.moves_max))

        for row in range(bar_height):
            colour = C_MOVE_FULL if row >= (bar_height - bar_filled) else C_MOVE_EMPTY
            fy = bar_top + row
            if 0 <= fy < CANVAS:
                for dx in range(3):
                    fx = sx + dx
                    if 0 <= fx < CANVAS:
                        frame[fy, fx] = colour

        for dy in range(2):
            for dx in range(3):
                fy = bar_top + bar_height + dy
                fx = sx + dx
                if 0 <= fy < CANVAS and 0 <= fx < CANVAS:
                    frame[fy, fx] = C_UI_SEP

        pip_y = bar_top + bar_height + 4
        for i in range(MAX_LIVES):
            col = C_LIFE_ON if i < self.lives else C_LIFE_OFF
            for dy in range(3):
                for dx in range(3):
                    fy = pip_y + i * 4 + dy
                    fx = sx + dx
                    if 0 <= fy < CANVAS and 0 <= fx < CANVAS:
                        frame[fy, fx] = col

        return frame


class Qr58(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)
        self._hud = Hud()
        game_levels = list(_LEVELS)
        cam = Camera(
            x=0,
            y=0,
            background=C_BG,
            letter_box=C_WALL,
            width=CANVAS,
            height=CANVAS,
            interfaces=[self._hud],
        )
        self._undo_stack = []
        self._can_undo = False
        super().__init__(
            "qr58",
            game_levels,
            cam,
            available_actions=[0, 1, 2, 3, 4, 5, 7],
        )

    def on_set_level(self, level: Level) -> None:
        self._undo_stack = []
        self._can_undo = False
        self._pipe_state = {}
        for col, row, ptype, rot in self.current_level.get_data("pipes"):
            self._pipe_state[(col, row)] = {"type": ptype, "rotation": rot}

        self._sources = [tuple(s) for s in self.current_level.get_data("sources")]
        self._sinks = [tuple(s) for s in self.current_level.get_data("sinks")]
        self._connected_sinks = set()

        self._lives = MAX_LIVES
        self._moves_max = self.current_level.get_data("moves")
        self._moves_left = self._moves_max

        self._cursor_col = 0
        self._cursor_row = 0

        self._cursor_spr = None
        self._pipe_sprs = {}
        self._sink_sprs = {}
        self._source_sprs = {}

        self._enemies = []
        self._enemy_sprs = []
        enemy_data = self.current_level.get_data("enemies") or []
        for edata in enemy_data:
            path = [tuple(p) for p in edata["path"]]
            if path:
                self._enemies.append({"path": path, "idx": 0})

        self._rebuild_sprite_cache()
        self._place_cursor()
        self._place_enemies()
        self._check_connections()
        self._sync_hud()

    def _rebuild_sprite_cache(self):
        self._pipe_sprs = {}
        self._sink_sprs = {}
        self._source_sprs = {}
        self._cursor_spr = None

        for spr in self.current_level.get_sprites():
            tags = spr.tags or []
            if "pipe" in tags or "pipe_c" in tags:
                col = (spr.x - PLAY_X) // CELL
                row = (spr.y - PLAY_Y) // CELL
                self._pipe_sprs[(col, row)] = spr
            elif "sink" in tags or "sink_on" in tags:
                col = (spr.x - PLAY_X) // CELL
                row = (spr.y - PLAY_Y) // CELL
                self._sink_sprs[(col, row)] = spr
            elif "source" in tags:
                col = (spr.x - PLAY_X) // CELL
                row = (spr.y - PLAY_Y) // CELL
                self._source_sprs[(col, row)] = spr
            elif "cursor" in tags:
                self._cursor_spr = spr

    def _place_cursor(self):
        if self._cursor_spr is not None:
            try:
                self.current_level.remove_sprite(self._cursor_spr)
            except Exception:
                pass
        spr = sprites["cursor"].clone()
        spr.set_position(_px(self._cursor_col), _py(self._cursor_row))
        self.current_level.add_sprite(spr)
        self._cursor_spr = spr

    def _place_enemies(self):
        for es in self._enemy_sprs:
            try:
                self.current_level.remove_sprite(es)
            except Exception:
                pass
        self._enemy_sprs = []

        for enemy in self._enemies:
            col, row = enemy["path"][enemy["idx"]]
            es = sprites["enemy"].clone()
            es.set_position(_px(col), _py(row))
            self.current_level.add_sprite(es)
            self._enemy_sprs.append(es)

    def _step_enemies(self):
        for enemy in self._enemies:
            path = enemy["path"]
            enemy["idx"] = (enemy["idx"] + 1) % len(path)
            col, row = path[enemy["idx"]]

            pos = (col, row)
            if pos in self._pipe_state:
                ps = self._pipe_state[pos]
                ps["rotation"] = (ps["rotation"] + 1) % 4

        self._place_enemies()
        self._check_connections()

    def _check_connections(self):
        connected = set()

        for src in self._sources:
            visited = set()
            queue = [src]
            visited.add(src)
            while queue:
                cur = queue.pop(0)
                cc, cr = cur
                connected.add(cur)

                if cur in self._sources:
                    cur_opens = {0, 1, 2, 3}
                elif cur in self._pipe_state:
                    ps = self._pipe_state[cur]
                    cur_opens = _get_openings(ps["type"], ps["rotation"])
                elif tuple(cur) in {tuple(s) for s in self._sinks}:
                    cur_opens = {0, 1, 2, 3}
                else:
                    continue

                for d in cur_opens:
                    dcol, drow = _DIR_DELTA[d]
                    nc, nr = cc + dcol, cr + drow
                    if not (0 <= nc < GRID_W and 0 <= nr < GRID_H):
                        continue
                    nb = (nc, nr)
                    if nb in visited:
                        continue

                    opp = _opposite(d)
                    if nb in self._sources:
                        nb_opens = {0, 1, 2, 3}
                    elif nb in self._pipe_state:
                        ps = self._pipe_state[nb]
                        nb_opens = _get_openings(ps["type"], ps["rotation"])
                    elif nb in {tuple(s) for s in self._sinks}:
                        nb_opens = {0, 1, 2, 3}
                    else:
                        continue

                    if opp in nb_opens:
                        visited.add(nb)
                        queue.append(nb)

        self._connected_sinks = set()
        for s in self._sinks:
            if s in connected:
                self._connected_sinks.add(s)

        self._update_sink_visuals()
        self._update_pipe_visuals(connected)

    def _update_sink_visuals(self):
        for col, row in self._sinks:
            pos = (col, row)
            old_spr = self._sink_sprs.get(pos)
            if old_spr is not None:
                try:
                    self.current_level.remove_sprite(old_spr)
                except Exception:
                    pass

            if pos in self._connected_sinks:
                ns = sprites["sink_on"].clone()
            else:
                ns = sprites["sink_off"].clone()
            ns.set_position(_px(col), _py(row))
            self.current_level.add_sprite(ns)
            self._sink_sprs[pos] = ns

    def _update_pipe_visuals(self, connected_cells):
        pipe_names = {0: "pipe_straight", 1: "pipe_elbow", 2: "pipe_tee"}
        pipe_names_c = {0: "pipe_straight_c", 1: "pipe_elbow_c", 2: "pipe_tee_c"}
        rot_angles = {0: 0, 1: 90, 2: 180, 3: 270}

        for pos, ps in self._pipe_state.items():
            old_spr = self._pipe_sprs.get(pos)
            if old_spr is not None:
                try:
                    self.current_level.remove_sprite(old_spr)
                except Exception:
                    pass

            if pos in connected_cells:
                key = pipe_names_c[ps["type"]]
            else:
                key = pipe_names[ps["type"]]

            ns = sprites[key].clone()
            ns.set_position(_px(pos[0]), _py(pos[1]))
            if ps["rotation"] != 0:
                ns.set_rotation(rot_angles[ps["rotation"]])
            self.current_level.add_sprite(ns)
            self._pipe_sprs[pos] = ns

    def _rotate_pipe_at_cursor(self):
        pos = (self._cursor_col, self._cursor_row)
        if pos not in self._pipe_state:
            return

        ps = self._pipe_state[pos]
        ps["rotation"] = (ps["rotation"] + 1) % 4

        self._check_connections()

    def _check_win(self):
        return len(self._connected_sinks) == len(self._sinks) and len(self._sinks) > 0

    def _sync_hud(self):
        self._hud.update(self._lives, self._moves_left, self._moves_max)

    def _spend_move(self):
        self._moves_left = max(0, self._moves_left - 1)
        self._sync_hud()
        if self._moves_left == 0:
            self._lives -= 1
            self._sync_hud()
            if self._lives <= 0:
                return "lose"
            return "reset"
        return "ok"

    def _reset_level(self):
        self._moves_left = self._moves_max

        self._pipe_state = {}
        for col, row, ptype, rot in self.current_level.get_data("pipes"):
            self._pipe_state[(col, row)] = {"type": ptype, "rotation": rot}

        for spr in list(self._pipe_sprs.values()):
            try:
                self.current_level.remove_sprite(spr)
            except Exception:
                pass
        self._pipe_sprs = {}

        for spr in list(self._sink_sprs.values()):
            try:
                self.current_level.remove_sprite(spr)
            except Exception:
                pass
        self._sink_sprs = {}

        if self._cursor_spr is not None:
            try:
                self.current_level.remove_sprite(self._cursor_spr)
            except Exception:
                pass
            self._cursor_spr = None

        for es in self._enemy_sprs:
            try:
                self.current_level.remove_sprite(es)
            except Exception:
                pass
        self._enemy_sprs = []

        for enemy in self._enemies:
            enemy["idx"] = 0

        self._cursor_col = 0
        self._cursor_row = 0
        self._connected_sinks = set()
        self._rebuild_sprite_cache()

        pipe_names = {0: "pipe_straight", 1: "pipe_elbow", 2: "pipe_tee"}
        rot_angles = {0: 0, 1: 90, 2: 180, 3: 270}
        for pos, ps in self._pipe_state.items():
            key = pipe_names[ps["type"]]
            s = sprites[key].clone()
            s.set_position(_px(pos[0]), _py(pos[1]))
            if ps["rotation"] != 0:
                s.set_rotation(rot_angles[ps["rotation"]])
            self.current_level.add_sprite(s)
            self._pipe_sprs[pos] = s

        for col, row in self._sinks:
            s = sprites["sink_off"].clone()
            s.set_position(_px(col), _py(row))
            self.current_level.add_sprite(s)
            self._sink_sprs[(col, row)] = s

        self._place_cursor()
        self._place_enemies()
        self._check_connections()
        self._sync_hud()

    def _save_undo_snapshot(self):
        snapshot = {
            "pipe_state": {k: dict(v) for k, v in self._pipe_state.items()},
            "cursor_col": self._cursor_col,
            "cursor_row": self._cursor_row,
            "moves_left": self._moves_left,
            "lives": self._lives,
            "enemies": [
                {"path": list(e["path"]), "idx": e["idx"]} for e in self._enemies
            ],
            "connected_sinks": set(self._connected_sinks),
        }
        self._undo_stack.append(snapshot)
        self._can_undo = True

    def _restore_undo_snapshot(self):
        if not self._undo_stack:
            return
        snapshot = self._undo_stack.pop()
        self._pipe_state = snapshot["pipe_state"]
        self._cursor_col = snapshot["cursor_col"]
        self._cursor_row = snapshot["cursor_row"]
        self._moves_left = snapshot["moves_left"]
        self._lives = snapshot["lives"]
        self._enemies = snapshot["enemies"]
        self._connected_sinks = snapshot["connected_sinks"]
        self._can_undo = len(self._undo_stack) > 0
        self._rebuild_sprite_cache()
        pipe_names = {0: "pipe_straight", 1: "pipe_elbow", 2: "pipe_tee"}
        rot_angles = {0: 0, 1: 90, 2: 180, 3: 270}
        for pos, ps in self._pipe_state.items():
            key = pipe_names[ps["type"]]
            s = sprites[key].clone()
            s.set_position(_px(pos[0]), _py(pos[1]))
            if ps["rotation"] != 0:
                s.set_rotation(rot_angles[ps["rotation"]])
            self.current_level.add_sprite(s)
            self._pipe_sprs[pos] = s
        for col, row in self._sinks:
            if (col, row) in self._connected_sinks:
                s = sprites["sink_on"].clone()
            else:
                s = sprites["sink_off"].clone()
            s.set_position(_px(col), _py(row))
            self.current_level.add_sprite(s)
            self._sink_sprs[(col, row)] = s
        self._place_cursor()
        self._place_enemies()
        self._sync_hud()

    def step(self) -> None:
        if self.action.id == GameAction.RESET:
            self.complete_action()
            return

        if self.action.id == GameAction.ACTION7:
            if self._can_undo:
                result = self._spend_move()
                saved_moves = self._moves_left
                self._restore_undo_snapshot()
                self._moves_left = saved_moves
                self._sync_hud()
                if result == "lose":
                    self.lose()
                    self.complete_action()
                    return
                elif result == "reset":
                    self._reset_level()
                    self.complete_action()
                    return
            self.complete_action()
            return

        self._save_undo_snapshot()

        action_val = self.action.id.value
        cursor_moved = False

        if action_val == 1:
            if self._cursor_row > 0:
                self._cursor_row -= 1
                cursor_moved = True
        elif action_val == 2:
            if self._cursor_row < GRID_H - 1:
                self._cursor_row += 1
                cursor_moved = True
        elif action_val == 3:
            if self._cursor_col > 0:
                self._cursor_col -= 1
                cursor_moved = True
        elif action_val == 4:
            if self._cursor_col < GRID_W - 1:
                self._cursor_col += 1
                cursor_moved = True
        elif action_val == 5:
            self._rotate_pipe_at_cursor()

        if cursor_moved:
            self._place_cursor()

        if self._check_win():
            self._spend_move()
            self._undo_stack = []
            self._can_undo = False
            self.next_level()
            self.complete_action()
            return

        if self._enemies:
            self._step_enemies()

        result = self._spend_move()
        if result == "lose":
            self.lose()
            self.complete_action()
            return
        elif result == "reset":
            self._reset_level()
            self.complete_action()
            return

        self.complete_action()


PIPE_TYPE_NAMES = {0: "straight", 1: "elbow", 2: "tee"}
DIR_CHARS = {0: "U", 1: "R", 2: "D", 3: "L"}

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

ARC_PALETTE = np.array(ARC_PALETTE, dtype=np.uint8)


class PuzzleEnvironment:
    _ACTION_MAP = {
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
        self._engine = Qr58(seed=seed)
        self._total_turns = 0
        self._done = False
        self._game_won = False
        self._last_action_was_reset = False
        self._levels_completed = 0

    def _build_text_observation(self) -> str:
        e = self._engine
        grid = [["." for _ in range(GRID_W)] for _ in range(GRID_H)]

        for src in e._sources:
            col, row = src
            if 0 <= col < GRID_W and 0 <= row < GRID_H:
                grid[row][col] = "S"

        for snk in e._sinks:
            col, row = snk
            if 0 <= col < GRID_W and 0 <= row < GRID_H:
                if snk in e._connected_sinks:
                    grid[row][col] = "X"
                else:
                    grid[row][col] = "O"

        for pos, ps in e._pipe_state.items():
            col, row = pos
            if 0 <= col < GRID_W and 0 <= row < GRID_H:
                openings = _get_openings(ps["type"], ps["rotation"])
                dirs = "".join(DIR_CHARS[d] for d in sorted(openings))
                grid[row][col] = dirs

        cursor_pos = (e._cursor_col, e._cursor_row)
        if 0 <= cursor_pos[0] < GRID_W and 0 <= cursor_pos[1] < GRID_H:
            cell = grid[cursor_pos[1]][cursor_pos[0]]
            grid[cursor_pos[1]][cursor_pos[0]] = "[" + cell + "]"

        header = (
            f"Level:{e.level_index + 1}/{len(e._levels)} "
            f"Lives:{e._lives} "
            f"Moves:{e._moves_left}/{e._moves_max}"
        )

        sinks_done = len(e._connected_sinks)
        sinks_total = len(e._sinks)
        header += f" Sinks:{sinks_done}/{sinks_total}"

        grid_text = "\n".join(" ".join(str(c).ljust(4) for c in row) for row in grid)
        return header + "\n" + grid_text

    def _build_game_state(self, done: bool = False) -> GameState:
        e = self._engine
        valid_actions = self.get_actions() if not done else None
        return GameState(
            text_observation=self._build_text_observation(),
            image_observation=None,
            valid_actions=valid_actions,
            turn=self._total_turns,
            metadata={
                "total_levels": len(e._levels),
                "level_index": e.level_index,
                "levels_completed": getattr(e, "_score", self._levels_completed),
                "game_over": getattr(getattr(e, "_state", None), "name", "")
                == "GAME_OVER"
                or (self._done and not self._game_won),
                "done": done,
                "info": {},
            },
        )

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
            return StepResult(
                state=state, reward=0.0, done=False, info={"action": "reset"}
            )

        if action not in self._ACTION_MAP:
            return StepResult(
                state=self._build_game_state(done=self._done),
                reward=0.0,
                done=self._done,
                info={"action": action, "reason": "invalid_action"},
            )

        self._last_action_was_reset = False
        self._total_turns += 1

        game_action = self._ACTION_MAP[action]
        info = {"action": action}
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
            self._levels_completed += 1
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
                state=self._build_game_state(done=False),
                reward=0.0,
                done=False,
                info=info,
            )

        reward = 0.0
        if e.level_index != level_before:
            reward = level_reward
            self._levels_completed += 1
            info["reason"] = "level_complete"

        return StepResult(
            state=self._build_game_state(done=False),
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


class ArcGameEnv(gym.Env):
    metadata: Dict[str, Any] = {"render_modes": ["rgb_array"], "render_fps": 5}
    ACTION_LIST: List[str] = ["reset", "up", "down", "left", "right", "select", "undo"]
    OBS_HEIGHT: int = 64
    OBS_WIDTH: int = 64

    def __init__(self, seed: int = 0, render_mode: Optional[str] = None) -> None:
        super().__init__()
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode '{render_mode}'.")
        self.render_mode = render_mode
        self._action_to_string = {i: a for i, a in enumerate(self.ACTION_LIST)}
        self._string_to_action = {a: i for i, a in enumerate(self.ACTION_LIST)}
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.OBS_HEIGHT, self.OBS_WIDTH, 3),
            dtype=np.uint8,
        )
        self.action_space = spaces.Discrete(len(self.ACTION_LIST))
        self._seed = seed
        self._env = None

    def reset(self, *, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed
        self._env = PuzzleEnvironment(seed=self._seed)
        state = self._env.reset()
        return self._get_obs(), self._build_info(state)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action_str = self._action_to_string[int(action)]
        result = self._env.step(action_str)
        obs = self._get_obs()
        terminated = result.done
        truncated = False
        return (
            obs,
            result.reward,
            terminated,
            truncated,
            self._build_info(result.state, result.info),
        )

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

    def _build_info(self, state, step_info=None) -> Dict[str, Any]:
        info = {
            "text_observation": state.text_observation,
            "valid_actions": state.valid_actions,
            "turn": state.turn,
            "game_metadata": state.metadata,
        }
        if step_info:
            info["step_info"] = step_info
        return info
