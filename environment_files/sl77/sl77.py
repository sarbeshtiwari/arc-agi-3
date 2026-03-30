import random
import zlib
import struct
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from arcengine import (
    ActionInput,
    ARCBaseGame,
    Camera,
    GameAction,
    Level,
    RenderableUserDisplay,
    Sprite,
)


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


BACKGROUND_COLOR = 0
PADDING_COLOR = 5

GRID_W = 16
GRID_H = 14

CLR_PLAYER = 15
CLR_SHADOW = 5
CLR_WALL = 1
CLR_CRYSTAL = 8
CLR_GHOST = 6
CLR_PPLATE_OFF = 4
CLR_SPLATE_OFF = 9
CLR_PLATE_ON = 3
CLR_DOOR = 11
CLR_EXIT = 7
CLR_SPIKE = 2
CLR_KEY = 1
CLR_KEY_COLLECTED = 7
CLR_FAKE_PLATE = 4


sprites = {
    "plr": Sprite(
        pixels=[[CLR_PLAYER]],
        name="plr",
        visible=True,
        collidable=False,
        tags=["plr"],
        layer=5,
    ),
    "shd": Sprite(
        pixels=[[CLR_SHADOW]],
        name="shd",
        visible=True,
        collidable=False,
        tags=["shd"],
        layer=4,
    ),
    "wal": Sprite(
        pixels=[[CLR_WALL]],
        name="wal",
        visible=True,
        collidable=True,
        tags=["wal"],
        layer=0,
    ),
    "cry": Sprite(
        pixels=[[CLR_CRYSTAL]],
        name="cry",
        visible=True,
        collidable=False,
        tags=["cry"],
        layer=0,
    ),
    "gho": Sprite(
        pixels=[[CLR_GHOST]],
        name="gho",
        visible=True,
        collidable=False,
        tags=["gho"],
        layer=0,
    ),
    "ppl": Sprite(
        pixels=[[CLR_PPLATE_OFF]],
        name="ppl",
        visible=True,
        collidable=False,
        tags=["ppl"],
        layer=-1,
    ),
    "spl": Sprite(
        pixels=[[CLR_SPLATE_OFF]],
        name="spl",
        visible=True,
        collidable=False,
        tags=["spl"],
        layer=-1,
    ),
    "dor": Sprite(
        pixels=[[CLR_DOOR]],
        name="dor",
        visible=True,
        collidable=True,
        tags=["dor"],
        layer=1,
    ),
    "mdr": Sprite(
        pixels=[[CLR_DOOR]],
        name="mdr",
        visible=True,
        collidable=False,
        tags=["mdr", "mine_gate"],
        layer=1,
    ),
    "fpl": Sprite(
        pixels=[[CLR_FAKE_PLATE]],
        name="fpl",
        visible=True,
        collidable=False,
        tags=["fpl", "mine_plate"],
        layer=-1,
    ),
    "ext": Sprite(
        pixels=[[CLR_EXIT]],
        name="ext",
        visible=True,
        collidable=False,
        tags=["ext"],
        layer=-1,
    ),
    "spk": Sprite(
        pixels=[[CLR_SPIKE]],
        name="spk",
        visible=True,
        collidable=False,
        tags=["spk"],
        layer=-1,
    ),
    "dov": Sprite(
        pixels=[[CLR_SPIKE]],
        name="dov",
        visible=False,
        collidable=False,
        tags=["dov"],
        layer=10,
    ),
    "key": Sprite(
        pixels=[[CLR_KEY]],
        name="key",
        visible=True,
        collidable=False,
        tags=["key"],
        layer=-1,
    ),
    "ex2": Sprite(
        pixels=[[CLR_EXIT]],
        name="ex2",
        visible=False,
        collidable=False,
        tags=["ex2"],
        layer=-1,
    ),
}


_MAP_CHARS = {
    "#": "wal",
    "P": "plr",
    "p": "ppl",
    "s": "spl",
    "D": "dor",
    "N": "mdr",
    "F": "fpl",
    "X": "ext",
    "H": "spk",
    "C": "cry",
    "G": "gho",
    "K": "key",
    "E": "ex2",
    ".": None,
}


def _mirror_map_horizontal(lines):
    mirrored = []
    for row in lines:
        inner = row[1:-1]
        mirrored_inner = inner[::-1]
        mirrored.append(row[0] + mirrored_inner + row[-1])
    return mirrored


_L1_NORMAL = [
    "################",
    "#..............#",
    "#.P.H..........#",
    "#..C...C.......#",
    "#..............#",
    "#...F...p......#",
    "####N###D#######",
    "#.#.G..........#",
    "#..#.......#G..#",
    "#....s....Gs.G.#",
    "#...#.......#..#",
    "#.......X..G...#",
    "#..#..........G#",
    "################",
]

_L1_MIRRORED = [
    "################",
    "#..............#",
    "#..........H.P.#",
    "#.......C...C..#",
    "#..............#",
    "#......p...F...#",
    "#######D###N####",
    "#..........G.#.#",
    "#..G#.......#..#",
    "#.G.s....Gs....#",
    "#..#.......#...#",
    "#...G..X.......#",
    "#G..........#..#",
    "################",
]

_L1_NORMAL_DATA = {
    "pairs": [{"door": [8, 6], "pplates": [[8, 5]], "splates": [[11, 9]]}],
    "max_moves": 57,
    "mine_gates": [[4, 6]],
    "mine_plates": [[4, 5]],
    "spawn_positions": [[2, 2], [3, 1], [5, 1], [8, 1]],
}

_L1_MIRRORED_DATA = {
    "pairs": [{"door": [7, 6], "pplates": [[7, 5]], "splates": [[4, 9]]}],
    "max_moves": 57,
    "mine_gates": [[11, 6]],
    "mine_plates": [[11, 5]],
    "spawn_positions": [[13, 2], [12, 1], [10, 1], [7, 1]],
}


_L2_NORMAL = [
    "################",
    "#..............#",
    "#.P.H..........#",
    "#.....CC.......#",
    "#.C............#",
    "#......p....F..#",
    "#######D####N###",
    "#.#.......G....#",
    "#.........s.#.G#",
    "#...........G..#",
    "#..#G......H.#.#",
    "#....#...s..G..#",
    "#.#...X....G..G#",
    "################",
]

_L2_MIRRORED = [
    "################",
    "#..............#",
    "#..........H.P.#",
    "#.......CC.....#",
    "#............C.#",
    "#..F....p......#",
    "###N####D#######",
    "#....G.......#.#",
    "#G.#.s.........#",
    "#..G...........#",
    "#.#.H......G#..#",
    "#..G..s..#.....#",
    "#G..G....X..#..#",
    "################",
]

_L2_NORMAL_DATA = {
    "pairs": [{"door": [7, 6], "pplates": [[7, 5]], "splates": [[10, 8]]}],
    "max_moves": 63,
    "mine_gates": [[12, 6]],
    "mine_plates": [[12, 5]],
    "spawn_positions": [[2, 2], [2, 1], [5, 1], [8, 1]],
}

_L2_MIRRORED_DATA = {
    "pairs": [{"door": [8, 6], "pplates": [[8, 5]], "splates": [[5, 8]]}],
    "max_moves": 63,
    "mine_gates": [[3, 6]],
    "mine_plates": [[3, 5]],
    "spawn_positions": [[13, 2], [13, 1], [10, 1], [7, 1]],
}


_L3_NORMAL = [
    "################",
    "#..H...........#",
    "#.P..C.........#",
    "#....C.........#",
    "#...HC.........#",
    "#.....p....F...#",
    "######D####N####",
    "#.#.......G....#",
    "#.........s.#.G#",
    "#...#....G.H...#",
    "#.........G.#.G#",
    "#..#G..........#",
    "#.G...X....G...#",
    "################",
]

_L3_MIRRORED = [
    "################",
    "#...........H..#",
    "#.........C..P.#",
    "#.........C....#",
    "#.........CH...#",
    "#...F....p.....#",
    "####N####D######",
    "#....G.......#.#",
    "#G.#.s.........#",
    "#...H.G....#...#",
    "#G.#.G.........#",
    "#..........G#..#",
    "#...G....X.G...#",
    "################",
]

_L3_NORMAL_DATA = {
    "pairs": [{"door": [6, 6], "pplates": [[6, 5]], "splates": [[10, 8]]}],
    "max_moves": 48,
    "mine_gates": [[11, 6]],
    "mine_plates": [[11, 5]],
    "spawn_positions": [[2, 2], [1, 1], [5, 1], [8, 2]],
}

_L3_MIRRORED_DATA = {
    "pairs": [{"door": [9, 6], "pplates": [[9, 5]], "splates": [[5, 8]]}],
    "max_moves": 48,
    "mine_gates": [[4, 6]],
    "mine_plates": [[4, 5]],
    "spawn_positions": [[13, 2], [14, 1], [10, 1], [7, 2]],
}


_L4_NORMAL = [
    "################",
    "#..............#",
    "#.P.H..........#",
    "#.C..H.........#",
    "#..C..H........#",
    "#...F...p...F..#",
    "####N###D###N###",
    "#....G....#....#",
    "#..#.....Gs....#",
    "#..G.......G...#",
    "#.#.s.....G#...#",
    "#.........G#G..#",
    "#...#...X.G.G..#",
    "################",
]

_L4_MIRRORED = [
    "################",
    "#..............#",
    "#..........H.P.#",
    "#.........H..C.#",
    "#........H..C..#",
    "#..F...p...F...#",
    "###N###D###N####",
    "#....#....G....#",
    "#....sG.....#..#",
    "#...G.......G..#",
    "#...#G.....s.#.#",
    "#..G#G.........#",
    "#..G.G..X..#...#",
    "################",
]

_L4_NORMAL_DATA = {
    "pairs": [{"door": [8, 6], "pplates": [[8, 5]], "splates": [[10, 8]]}],
    "max_moves": 48,
    "rotated_controls": True,
    "mine_gates": [[4, 6], [12, 6]],
    "mine_plates": [[4, 5], [12, 5]],
    "spawn_positions": [[2, 2], [2, 1], [6, 1], [8, 1]],
}

_L4_MIRRORED_DATA = {
    "pairs": [{"door": [7, 6], "pplates": [[7, 5]], "splates": [[5, 8]]}],
    "max_moves": 48,
    "rotated_controls": True,
    "mine_gates": [[3, 6], [11, 6]],
    "mine_plates": [[3, 5], [11, 5]],
    "spawn_positions": [[13, 2], [13, 1], [9, 1], [7, 2]],
}


_L5_NORMAL = [
    "################",
    "#.E............#",
    "#.P.H..........#",
    "#.C..H.........#",
    "#..C..H........#",
    "#...F...p...F..#",
    "####N###D###N###",
    "#..............#",
    "#....s....s....#",
    "#..KG....G..GK.#",
    "#...G...GK..G..#",
    "#........G..G..#",
    "#.......X...G..#",
    "################",
]

_L5_MIRRORED = [
    "################",
    "#............E.#",
    "#..........H.P.#",
    "#.........H..C.#",
    "#........H..C..#",
    "#..F...p...F...#",
    "###N###D###N####",
    "#..............#",
    "#....s....s....#",
    "#.KG..G....GK..#",
    "#..G..KG...G...#",
    "#..G..G........#",
    "#..G...X.......#",
    "################",
]

_L5_NORMAL_DATA = {
    "pairs": [{"door": [8, 6], "pplates": [[8, 5]], "splates": [[10, 8]]}],
    "max_moves": 90,
    "two_phase_exit": True,
    "key_positions": [[9, 10], [13, 9]],
    "second_exit": [2, 1],
    "mine_gates": [[4, 6], [12, 6]],
    "mine_plates": [[4, 5], [12, 5]],
    "spawn_positions": [[2, 2], [1, 1], [6, 1], [8, 1]],
}

_L5_MIRRORED_DATA = {
    "pairs": [{"door": [7, 6], "pplates": [[7, 5]], "splates": [[5, 8]]}],
    "max_moves": 90,
    "two_phase_exit": True,
    "key_positions": [[2, 9], [6, 10]],
    "second_exit": [13, 1],
    "mine_gates": [[3, 6], [11, 6]],
    "mine_plates": [[3, 5], [11, 5]],
    "spawn_positions": [[13, 2], [14, 1], [14, 2], [9, 1]],
}


def _build_level(mp, data):
    spr_list = []
    h = len(mp)
    w = len(mp[0]) if h > 0 else 0
    for y, row in enumerate(mp):
        for x, ch in enumerate(row):
            if ch in _MAP_CHARS and _MAP_CHARS[ch] is not None:
                spr_list.append(sprites[_MAP_CHARS[ch]].clone().set_position(x, y))
    spr_list.append(sprites["shd"].clone().set_position(0, 0))
    spr_list.append(sprites["dov"].clone().set_position(0, 0))
    return Level(sprites=spr_list, grid_size=(w, h), data=data)


def _pick_variant(rng, normal_map, mirrored_map, normal_data, mirrored_data):
    if rng.random() < 0.5:
        return normal_map, normal_data
    else:
        return mirrored_map, mirrored_data


def _fix_row_widths(mp, width=16):
    fixed = []
    for row in mp:
        if len(row) < width:
            inner = row[1:-1]
            inner = inner + "." * (width - 2 - len(inner))
            row = row[0] + inner + row[-1]
        elif len(row) > width:
            row = row[: width - 1] + row[-1]
        fixed.append(row)
    return fixed


def _build_levels(rng):
    _l1_map, _l1_data = _pick_variant(
        rng, _L1_NORMAL, _L1_MIRRORED, _L1_NORMAL_DATA, _L1_MIRRORED_DATA
    )
    _l2_map, _l2_data = _pick_variant(
        rng, _L2_NORMAL, _L2_MIRRORED, _L2_NORMAL_DATA, _L2_MIRRORED_DATA
    )
    _l3_map, _l3_data = _pick_variant(
        rng, _L3_NORMAL, _L3_MIRRORED, _L3_NORMAL_DATA, _L3_MIRRORED_DATA
    )
    _l4_map, _l4_data = _pick_variant(
        rng, _L4_NORMAL, _L4_MIRRORED, _L4_NORMAL_DATA, _L4_MIRRORED_DATA
    )
    _l5_map, _l5_data = _pick_variant(
        rng, _L5_NORMAL, _L5_MIRRORED, _L5_NORMAL_DATA, _L5_MIRRORED_DATA
    )
    _l1_map = _fix_row_widths(_l1_map)
    _l2_map = _fix_row_widths(_l2_map)
    _l3_map = _fix_row_widths(_l3_map)
    _l4_map = _fix_row_widths(_l4_map)
    _l5_map = _fix_row_widths(_l5_map)
    return [
        _build_level(_l1_map, _l1_data),
        _build_level(_l2_map, _l2_data),
        _build_level(_l3_map, _l3_data),
        _build_level(_l4_map, _l4_data),
        _build_level(_l5_map, _l5_data),
    ]


class HudDisplay(RenderableUserDisplay):
    def __init__(self, game: "Sl77") -> None:
        self.game = game

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        g = self.game
        fh, fw = frame.shape

        cam_w, cam_h = GRID_W, GRID_H
        scale = min(fw // cam_w, fh // cam_h)
        x_off = (fw - cam_w * scale) // 2
        y_off = (fh - cam_h * scale) // 2

        def gy(row):
            return slice(y_off + row * scale, y_off + (row + 1) * scale)

        def gx(col):
            return slice(x_off + col * scale, x_off + (col + 1) * scale)

        if g.death_flash > 0:
            for col in range(cam_w):
                frame[gy(0), gx(col)] = CLR_SPIKE
                frame[gy(cam_h - 1), gx(col)] = CLR_SPIKE
            return frame

        bar_color = PADDING_COLOR
        if g.max_moves > 0:
            remaining = max(0, g.max_moves - g.moves_used)
            ratio = remaining / g.max_moves
            if ratio > 0.25:
                bar_color = CLR_PLATE_ON
            else:
                bar_color = CLR_EXIT

        for col in range(cam_w):
            frame[gy(0), gx(col)] = bar_color
        frame[gy(0), gx(0)] = CLR_PLAYER
        frame[gy(0), gx(4)] = CLR_PLAYER
        for i in range(3):
            col = 1 + i
            frame[gy(0), gx(col)] = 5
            if g.lives > i:
                cy = y_off + 0 * scale + 1
                cx = x_off + col * scale + 1
                frame[cy : cy + 2, cx : cx + 2] = CLR_PLATE_ON

        lvl_idx = g.current_level_index if hasattr(g, "current_level_index") else 0
        for i in range(5):
            col = 10 + i
            frame[gy(0), gx(col)] = 5
            if i < lvl_idx:
                cy = y_off + 0 * scale + 1
                cx = x_off + col * scale + 1
                frame[cy : cy + 2, cx : cx + 2] = CLR_EXIT

        bar_row = gy(cam_h - 1)
        for col in range(cam_w):
            frame[bar_row, gx(col)] = PADDING_COLOR
        if g.max_moves > 0:
            remaining = max(0, g.max_moves - g.moves_used)
            ratio = remaining / g.max_moves
            cells_filled = int(ratio * cam_w)
            for col in range(cells_filled):
                frame[bar_row, gx(col)] = bar_color

        for row in range(1, cam_h - 1):
            frame[gy(row), gx(0)] = bar_color
            frame[gy(row), gx(cam_w - 1)] = bar_color

        return frame


class Sl77(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)
        self._levels = _build_levels(self._rng)

        self.lives = 3
        self.death_flash = 0
        self.moves_used = 0
        self.max_moves = 25
        self.current_level_index = 0

        self.player_x = 0
        self.player_y = 0
        self.shadow_x = 0
        self.shadow_y = 0
        self.spawn_x = 0
        self.spawn_y = 0

        self.player_spr = None
        self.shadow_spr = None
        self.dov_spr = None

        self.wall_set: set = set()
        self.crystal_set: set = set()
        self.ghost_set: set = set()
        self.spike_set: set = set()
        self.pplates_spr: dict = {}
        self.splates_spr: dict = {}
        self.doors_spr: dict = {}
        self.door_pairs: list = []

        self.rotated_controls = False

        self.two_phase_exit = False
        self.exit_phase = 1
        self.phase2_open_doors = False
        self.keys_collected = set()
        self.key_sprites: dict = {}
        self.key_positions: list = []
        self.second_exit_pos = None
        self.second_exit_spr = None

        self.mine_gate_set: set = set()
        self.mine_gate_sprites: dict = {}
        self.mine_plate_set: set = set()
        self.mine_plate_sprites: dict = {}
        self.mine_gates_revealed = False

        self._history: list[dict] = []
        self._game_over = False
        self._consecutive_resets = 0

        self._hud = HudDisplay(self)

        super().__init__(
            "sl77",
            self._levels,
            Camera(0, 0, GRID_W, GRID_H, BACKGROUND_COLOR, PADDING_COLOR, [self._hud]),
            available_actions=[0, 1, 2, 3, 4, 5, 7],
        )

    def on_set_level(self, level: Level) -> None:
        self.lives = 3
        self.death_flash = 0
        self.moves_used = 0
        self._history = []
        self._game_over = False

        for i, lv in enumerate(self._levels):
            if lv is self.current_level:
                self.current_level_index = i
                break

        self.player_spr = self.current_level.get_sprites_by_tag("plr")[0]
        self.shadow_spr = self.current_level.get_sprites_by_tag("shd")[0]
        self.shadow_spr.set_visible(True)

        self.spawn_x = self.player_spr.x
        self.spawn_y = self.player_spr.y
        self._randomize_spawn()

        self.wall_set = {
            (s.x, s.y) for s in self.current_level.get_sprites_by_tag("wal")
        }
        self.crystal_set = {
            (s.x, s.y) for s in self.current_level.get_sprites_by_tag("cry")
        }
        self.ghost_set = {
            (s.x, s.y) for s in self.current_level.get_sprites_by_tag("gho")
        }
        self.spike_set = {
            (s.x, s.y) for s in self.current_level.get_sprites_by_tag("spk")
        }

        self.pplates_spr = {
            (s.x, s.y): s for s in self.current_level.get_sprites_by_tag("ppl")
        }
        self.splates_spr = {
            (s.x, s.y): s for s in self.current_level.get_sprites_by_tag("spl")
        }
        self.doors_spr = {
            (s.x, s.y): s for s in self.current_level.get_sprites_by_tag("dor")
        }

        raw = self.current_level.get_data("pairs")
        self.door_pairs = raw if raw is not None else []

        for spr in self.doors_spr.values():
            spr.set_visible(True)
        for spr in self.pplates_spr.values():
            spr.color_remap(CLR_PLATE_ON, CLR_PPLATE_OFF)
        for spr in self.splates_spr.values():
            spr.color_remap(CLR_PLATE_ON, CLR_SPLATE_OFF)

        dov_list = self.current_level.get_sprites_by_tag("dov")
        self.dov_spr = dov_list[0] if dov_list else None
        if self.dov_spr:
            self.dov_spr.set_visible(False)

        mv = self.current_level.get_data("max_moves")
        self.max_moves = int(mv) if mv is not None else 100

        rc = self.current_level.get_data("rotated_controls")
        self.rotated_controls = bool(rc) if rc is not None else False

        tpe = self.current_level.get_data("two_phase_exit")
        self.two_phase_exit = bool(tpe) if tpe is not None else False
        self.exit_phase = 1
        self.phase2_open_doors = False
        self.keys_collected = set()
        self.key_sprites = {}
        self.key_positions = []
        self.second_exit_pos = None
        self.second_exit_spr = None

        if self.two_phase_exit:
            kp_raw = self.current_level.get_data("key_positions")
            if kp_raw:
                self.key_positions = [[k[0], k[1]] for k in kp_raw]
            for ks in self.current_level.get_sprites_by_tag("key"):
                self.key_sprites[(ks.x, ks.y)] = ks
            se_raw = self.current_level.get_data("second_exit")
            if se_raw:
                self.second_exit_pos = (se_raw[0], se_raw[1])
            ex2_list = self.current_level.get_sprites_by_tag("ex2")
            if ex2_list:
                self.second_exit_spr = ex2_list[0]
                self.second_exit_spr.set_visible(False)

        self.mine_gate_set = set()
        self.mine_gate_sprites = {}
        self.mine_plate_set = set()
        self.mine_plate_sprites = {}
        self.mine_gates_revealed = False

        mg_raw = self.current_level.get_data("mine_gates")
        if mg_raw:
            for g in mg_raw:
                self.mine_gate_set.add((g[0], g[1]))
            for ms in self.current_level.get_sprites_by_tag("mine_gate"):
                self.mine_gate_sprites[(ms.x, ms.y)] = ms

        mp_raw = self.current_level.get_data("mine_plates")
        if mp_raw:
            for g in mp_raw:
                self.mine_plate_set.add((g[0], g[1]))
            for fp in self.current_level.get_sprites_by_tag("mine_plate"):
                self.mine_plate_sprites[(fp.x, fp.y)] = fp

    def _blocks_player(self, x: int, y: int) -> bool:
        if x < 0 or x >= GRID_W or y < 0 or y >= GRID_H:
            return True
        if (x, y) in self.wall_set:
            return True
        if (x, y) in self.crystal_set:
            return True
        if not self.phase2_open_doors:
            spr = self.doors_spr.get((x, y))
            if spr is not None and spr.is_visible:
                return True
        return False

    def _blocks_shadow(self, x: int, y: int) -> bool:
        if x < 0 or x >= GRID_W or y < 0 or y >= GRID_H:
            return True
        if (x, y) in self.wall_set:
            return True
        if (x, y) in self.ghost_set:
            return True
        if not self.phase2_open_doors:
            spr = self.doors_spr.get((x, y))
            if spr is not None and spr.is_visible:
                return True
        return False

    def _evaluate_plates_doors(self) -> None:
        px, py = self.player_x, self.player_y
        sx, sy = self.shadow_x, self.shadow_y

        for pair in self.door_pairs:
            door_pos = (pair["door"][0], pair["door"][1])
            all_held = True

            for pp in pair.get("pplates", []):
                pos = (pp[0], pp[1])
                spr = self.pplates_spr.get(pos)
                if spr is not None:
                    if (px, py) == pos:
                        spr.color_remap(CLR_PPLATE_OFF, CLR_PLATE_ON)
                    else:
                        spr.color_remap(CLR_PLATE_ON, CLR_PPLATE_OFF)
                        all_held = False

            for sp in pair.get("splates", []):
                pos = (sp[0], sp[1])
                spr = self.splates_spr.get(pos)
                if spr is not None:
                    if (sx, sy) == pos:
                        spr.color_remap(CLR_SPLATE_OFF, CLR_PLATE_ON)
                    else:
                        spr.color_remap(CLR_PLATE_ON, CLR_SPLATE_OFF)
                        all_held = False

            door_spr = self.doors_spr.get(door_pos)
            if door_spr is not None:
                door_spr.set_visible(not all_held)

    def _randomize_spawn(self) -> None:
        positions = self.current_level.get_data("spawn_positions")
        if positions:
            pos = self._rng.choice(positions)
            self.spawn_x = pos[0]
            self.spawn_y = pos[1]
        self.player_x = self.spawn_x
        self.player_y = self.spawn_y
        self.player_spr.set_position(self.player_x, self.player_y)
        self.shadow_x = GRID_W - 1 - self.spawn_x
        self.shadow_y = GRID_H - 1 - self.spawn_y
        self.shadow_spr.set_position(self.shadow_x, self.shadow_y)

    def _die(self) -> None:
        self.lives -= 1

        if self.lives <= 0:
            self._game_over = True
            self.lose()
            return

        self.death_flash = 0

        self._randomize_spawn()

        self.moves_used = 0

        for spr in self.doors_spr.values():
            spr.set_visible(True)
        for spr in self.pplates_spr.values():
            spr.color_remap(CLR_PLATE_ON, CLR_PPLATE_OFF)
        for spr in self.splates_spr.values():
            spr.color_remap(CLR_PLATE_ON, CLR_SPLATE_OFF)

        if self.two_phase_exit:
            self.exit_phase = 1
            self.phase2_open_doors = False
            self.keys_collected = set()
            for pos, ks in self.key_sprites.items():
                ks.color_remap(CLR_KEY_COLLECTED, CLR_KEY)
                ks.set_visible(True)
            if self.second_exit_spr:
                self.second_exit_spr.set_visible(False)
            for ext_spr in self.current_level.get_sprites_by_tag("ext"):
                ext_spr.set_visible(True)

        self.mine_gates_revealed = False
        for pos, mspr in self.mine_gate_sprites.items():
            mspr.color_remap(CLR_SPIKE, CLR_DOOR)
            mspr.set_visible(True)
        for pos, fspr in self.mine_plate_sprites.items():
            fspr.color_remap(CLR_SPIKE, CLR_FAKE_PLATE)
            fspr.set_visible(True)

    def _save_state(self) -> None:
        self._history.append(
            {
                "player_x": self.player_x,
                "player_y": self.player_y,
                "shadow_x": self.shadow_x,
                "shadow_y": self.shadow_y,
                "moves_used": self.moves_used,
                "keys_collected": set(self.keys_collected),
                "exit_phase": self.exit_phase,
                "phase2_open_doors": self.phase2_open_doors,
                "mine_gates_revealed": self.mine_gates_revealed,
                "door_visibility": {
                    pos: spr.is_visible for pos, spr in self.doors_spr.items()
                },
                "ext_visibility": {
                    (spr.x, spr.y): spr.is_visible
                    for spr in self.current_level.get_sprites_by_tag("ext")
                },
                "second_exit_visible": self.second_exit_spr.is_visible
                if self.second_exit_spr
                else False,
                "key_visibility": {
                    pos: spr.is_visible for pos, spr in self.key_sprites.items()
                },
                "mine_gate_visibility": {
                    pos: spr.is_visible for pos, spr in self.mine_gate_sprites.items()
                },
                "mine_plate_visibility": {
                    pos: spr.is_visible for pos, spr in self.mine_plate_sprites.items()
                },
                "lives": self.lives,
            }
        )

    def _undo(self) -> None:
        if not self._history:
            return
        snap = self._history.pop()
        self.player_x = snap["player_x"]
        self.player_y = snap["player_y"]
        self.shadow_x = snap["shadow_x"]
        self.shadow_y = snap["shadow_y"]
        self.keys_collected = set(snap["keys_collected"])
        self.exit_phase = snap["exit_phase"]
        self.phase2_open_doors = snap["phase2_open_doors"]
        self.lives = snap["lives"]
        prev_revealed = self.mine_gates_revealed
        self.mine_gates_revealed = snap["mine_gates_revealed"]
        self.player_spr.set_position(self.player_x, self.player_y)
        self.shadow_spr.set_position(self.shadow_x, self.shadow_y)
        self._evaluate_plates_doors()
        for pos, spr in self.doors_spr.items():
            spr.set_visible(snap["door_visibility"].get(pos, True))
        for ext_spr in self.current_level.get_sprites_by_tag("ext"):
            key = (ext_spr.x, ext_spr.y)
            ext_spr.set_visible(snap["ext_visibility"].get(key, True))
        if self.second_exit_spr:
            self.second_exit_spr.set_visible(snap["second_exit_visible"])
        for pos, ks in self.key_sprites.items():
            if pos in self.keys_collected:
                ks.color_remap(CLR_KEY, CLR_KEY_COLLECTED)
            else:
                ks.color_remap(CLR_KEY_COLLECTED, CLR_KEY)
            ks.set_visible(snap["key_visibility"].get(pos, True))
        if prev_revealed and not self.mine_gates_revealed:
            for pos, mspr in self.mine_gate_sprites.items():
                mspr.color_remap(CLR_SPIKE, CLR_DOOR)
                mspr.set_visible(snap["mine_gate_visibility"].get(pos, True))
            for pos, fspr in self.mine_plate_sprites.items():
                fspr.color_remap(CLR_SPIKE, CLR_FAKE_PLATE)
                fspr.set_visible(snap["mine_plate_visibility"].get(pos, True))
        elif not prev_revealed and self.mine_gates_revealed:
            for pos, mspr in self.mine_gate_sprites.items():
                mspr.color_remap(CLR_DOOR, CLR_SPIKE)
                mspr.set_visible(snap["mine_gate_visibility"].get(pos, True))
            for pos, fspr in self.mine_plate_sprites.items():
                fspr.color_remap(CLR_FAKE_PLATE, CLR_SPIKE)
                fspr.set_visible(snap["mine_plate_visibility"].get(pos, True))
        else:
            for pos, mspr in self.mine_gate_sprites.items():
                mspr.set_visible(snap["mine_gate_visibility"].get(pos, True))
            for pos, fspr in self.mine_plate_sprites.items():
                fspr.set_visible(snap["mine_plate_visibility"].get(pos, True))

    def handle_reset(self) -> None:
        self._consecutive_resets += 1
        if self._consecutive_resets >= 2:
            self._consecutive_resets = 0
            self.full_reset()
        else:
            self.level_reset()

    def step(self) -> None:
        act = self.action.id.value

        if act == 7:
            self._consecutive_resets = 0
            self._undo()
            self.moves_used += 1
            self.complete_action()
            return

        if self.rotated_controls and act in (1, 2, 3, 4):
            act = {1: 4, 4: 2, 2: 3, 3: 1}[act]

        dx = dy = 0
        if act == 1:
            dy = -1
        elif act == 2:
            dy = 1
        elif act == 3:
            dx = -1
        elif act == 4:
            dx = 1
        elif act == 5:
            pass
        else:
            self.complete_action()
            return

        self._consecutive_resets = 0
        self._save_state()

        nx, ny = self.player_x + dx, self.player_y + dy
        if not self._blocks_player(nx, ny):
            self.player_x = nx
            self.player_y = ny
        self.player_spr.set_position(self.player_x, self.player_y)

        snx, sny = self.shadow_x - dx, self.shadow_y - dy
        if not self._blocks_shadow(snx, sny):
            self.shadow_x = snx
            self.shadow_y = sny
        self.shadow_spr.set_position(self.shadow_x, self.shadow_y)

        self._evaluate_plates_doors()

        self.moves_used += 1

        ppos = (self.player_x, self.player_y)
        if not self.mine_gates_revealed:
            if ppos in self.mine_gate_set:
                mspr = self.mine_gate_sprites.get(ppos)
                if mspr:
                    mspr.color_remap(CLR_DOOR, CLR_SPIKE)
                self._die()
                self.complete_action()
                return
            if ppos in self.mine_plate_set:
                fspr = self.mine_plate_sprites.get(ppos)
                if fspr:
                    fspr.color_remap(CLR_FAKE_PLATE, CLR_SPIKE)
                self._die()
                self.complete_action()
                return

        if not self.mine_gates_revealed and self.player_y > 6:
            self.mine_gates_revealed = True
            for mpos, mspr in self.mine_gate_sprites.items():
                mspr.color_remap(CLR_DOOR, CLR_SPIKE)
            for fpos, fspr in self.mine_plate_sprites.items():
                fspr.color_remap(CLR_FAKE_PLATE, CLR_SPIKE)

        if (self.player_x, self.player_y) in self.spike_set:
            self._die()
            self.complete_action()
            return

        if (self.shadow_x, self.shadow_y) in self.spike_set:
            self._die()
            self.complete_action()
            return

        if self.two_phase_exit and self.exit_phase == 2:
            ppos = (self.player_x, self.player_y)
            if ppos in self.key_sprites and ppos not in self.keys_collected:
                self.keys_collected.add(ppos)
                ks = self.key_sprites[ppos]
                ks.color_remap(CLR_KEY, CLR_KEY_COLLECTED)
                all_keys = {(k[0], k[1]) for k in self.key_positions}
                if self.keys_collected >= all_keys and self.second_exit_spr:
                    self.second_exit_spr.set_visible(True)

        if self.two_phase_exit:
            if self.exit_phase == 1:
                for ext_spr in self.current_level.get_sprites_by_tag("ext"):
                    if (
                        ext_spr.is_visible
                        and ext_spr.x == self.player_x
                        and ext_spr.y == self.player_y
                    ):
                        ext_spr.set_visible(False)
                        self.exit_phase = 2
                        self.phase2_open_doors = True
                        for dspr in self.doors_spr.values():
                            dspr.set_visible(False)
                        if self.second_exit_spr:
                            self.second_exit_spr.set_visible(True)
                        break
            elif self.exit_phase == 2:
                if (
                    self.second_exit_spr
                    and self.second_exit_spr.is_visible
                    and self.player_x == self.second_exit_spr.x
                    and self.player_y == self.second_exit_spr.y
                ):
                    self.next_level()
                    self.complete_action()
                    return
        else:
            for ext_spr in self.current_level.get_sprites_by_tag("ext"):
                if ext_spr.x == self.player_x and ext_spr.y == self.player_y:
                    self.next_level()
                    self.complete_action()
                    return

        if self.max_moves > 0 and self.moves_used >= self.max_moves:
            self._die()
            self.complete_action()
            return

        self.complete_action()


class PuzzleEnvironment:
    ACTION_MAP: dict[str, int] = {
        "up": 1,
        "down": 2,
        "left": 3,
        "right": 4,
        "wait": 5,
        "undo": 7,
        "reset": 0,
    }

    _ARC_PALETTE = [
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

    def __init__(self, seed: int = 0) -> None:
        self._engine = Sl77(seed)
        self._total_turns = 0
        self._last_action_was_reset = False
        self._total_levels = len(self._engine._levels)

    def _build_text_obs(self) -> str:
        e = self._engine
        grid = [["." for _ in range(GRID_W)] for _ in range(GRID_H)]

        for pos in e.wall_set:
            x, y = pos
            if 0 <= x < GRID_W and 0 <= y < GRID_H:
                grid[y][x] = "#"

        for pos in e.crystal_set:
            x, y = pos
            if 0 <= x < GRID_W and 0 <= y < GRID_H:
                grid[y][x] = "C"

        for pos in e.ghost_set:
            x, y = pos
            if 0 <= x < GRID_W and 0 <= y < GRID_H:
                grid[y][x] = "G"

        for pos in e.spike_set:
            x, y = pos
            if 0 <= x < GRID_W and 0 <= y < GRID_H:
                grid[y][x] = "^"

        for pos, spr in e.doors_spr.items():
            x, y = pos
            if 0 <= x < GRID_W and 0 <= y < GRID_H:
                grid[y][x] = "D" if spr.is_visible else "d"

        for pos in e.pplates_spr:
            x, y = pos
            if 0 <= x < GRID_W and 0 <= y < GRID_H:
                grid[y][x] = "p"

        for pos in e.splates_spr:
            x, y = pos
            if 0 <= x < GRID_W and 0 <= y < GRID_H:
                grid[y][x] = "s"

        for pos, spr in e.mine_gate_sprites.items():
            x, y = pos
            if 0 <= x < GRID_W and 0 <= y < GRID_H:
                if spr.is_visible:
                    if e.mine_gates_revealed:
                        grid[y][x] = "!"
                    else:
                        grid[y][x] = "N"

        for pos, spr in e.mine_plate_sprites.items():
            x, y = pos
            if 0 <= x < GRID_W and 0 <= y < GRID_H:
                if spr.is_visible:
                    if e.mine_gates_revealed:
                        grid[y][x] = "!"
                    else:
                        grid[y][x] = "F"

        for ext_spr in e.current_level.get_sprites_by_tag("ext"):
            if ext_spr.is_visible:
                ex, ey = ext_spr.x, ext_spr.y
                if 0 <= ex < GRID_W and 0 <= ey < GRID_H:
                    grid[ey][ex] = "X"

        for pos in e.key_sprites:
            x, y = pos
            if 0 <= x < GRID_W and 0 <= y < GRID_H:
                if pos not in e.keys_collected:
                    grid[y][x] = "K"

        if e.second_exit_spr and e.second_exit_spr.is_visible:
            sx, sy = e.second_exit_spr.x, e.second_exit_spr.y
            if 0 <= sx < GRID_W and 0 <= sy < GRID_H:
                grid[sy][sx] = "X"

        if 0 <= e.shadow_x < GRID_W and 0 <= e.shadow_y < GRID_H:
            grid[e.shadow_y][e.shadow_x] = "S"

        if 0 <= e.player_x < GRID_W and 0 <= e.player_y < GRID_H:
            grid[e.player_y][e.player_x] = "P"

        ascii_rows = ["".join(row) for row in grid]
        header = (
            f"Level {e.current_level_index + 1}/{len(e._levels)} | "
            f"Moves: {e.moves_used}/{e.max_moves} | "
            f"Lives: {e.lives}"
        )
        if e.rotated_controls:
            header += " | Controls: rotated"
        if e.two_phase_exit:
            header += f" | Phase: {e.exit_phase}"
            header += f" | Keys: {len(e.keys_collected)}/{len(e.key_positions)}"

        return header + "\n" + "\n".join(ascii_rows)

    def _build_image_bytes(self) -> bytes | None:
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
        for idx, color in enumerate(self._ARC_PALETTE):
            mask = arr == idx
            rgb[mask] = color

        def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
            chunk = chunk_type + data
            return (
                struct.pack(">I", len(data))
                + chunk
                + struct.pack(">I", zlib.crc32(chunk) & 0xFFFFFFFF)
            )

        raw_rows = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))
        return (
            b"\x89PNG\r\n\x1a\n"
            + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + _png_chunk(b"IDAT", zlib.compress(raw_rows))
            + _png_chunk(b"IEND", b"")
        )

    def _build_game_state(
        self, done: bool = False, info: dict[str, Any] | None = None
    ) -> GameState:
        e = self._engine
        valid_actions = self.get_actions() if not done else None
        return GameState(
            text_observation=self._build_text_obs(),
            image_observation=self._build_image_bytes(),
            valid_actions=valid_actions,
            turn=self._total_turns,
            metadata={
                "player_position": (e.player_x, e.player_y),
                "shadow_position": (e.shadow_x, e.shadow_y),
                "moves_used": e.moves_used,
                "max_moves": e.max_moves,
                "lives": e.lives,
                "level_index": e.current_level_index,
                "total_levels": len(e._levels),
                "grid_size": (GRID_W, GRID_H),
                "rotated_controls": e.rotated_controls,
                "two_phase_exit": e.two_phase_exit,
                "exit_phase": e.exit_phase,
                "keys_collected": len(e.keys_collected),
                "total_keys": len(e.key_positions),
                "done": done,
                "info": info or {},
                "levels_completed": getattr(self._engine, "_score", 0),
                "game_over": getattr(getattr(self._engine, "_state", None), "name", "") == "GAME_OVER",
            },
        )

    def reset(self) -> GameState:
        e = self._engine
        e._game_over = False
        game_won = hasattr(e, "_state") and getattr(e._state, "name", "") == "WIN"
        if game_won or self._last_action_was_reset:
            e._consecutive_resets = 0
            e.full_reset()
        else:
            e.level_reset()
        self._total_turns = 0
        self._last_action_was_reset = True
        return self._build_game_state()

    def get_actions(self) -> list[str]:
        if self._engine._game_over:
            return ["reset"]
        return ["up", "down", "left", "right", "wait", "undo", "reset"]

    def is_done(self) -> bool:
        e = self._engine
        return e._game_over or (
            hasattr(e, "_state") and getattr(e._state, "name", "") == "WIN"
        )

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
        action_map = {
            1: GameAction.ACTION1,
            2: GameAction.ACTION2,
            3: GameAction.ACTION3,
            4: GameAction.ACTION4,
            5: GameAction.ACTION5,
            7: GameAction.ACTION7,
        }
        game_action = action_map[game_action_id]
        info: dict[str, Any] = {"action": action}

        level_before = e.current_level_index
        action_input = ActionInput(id=game_action)
        frame = e.perform_action(action_input, raw=True)
        level_after = e.current_level_index

        game_won = frame and frame.state and frame.state.name == "WIN"
        done = e._game_over or game_won

        if game_won:
            info["reason"] = "game_complete"
            reward = 1.0 / self._total_levels
            return StepResult(
                state=self._build_game_state(done=True, info=info),
                reward=reward,
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
        if mode != "rgb_array":
            raise ValueError(f"Unsupported render mode: {mode}")
        index_grid = self._engine.camera.render(
            self._engine.current_level.get_sprites()
        )
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        for idx, color in enumerate(self._ARC_PALETTE):
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

    _GYM_TO_INTERNAL = {"select": "wait"}
    _INTERNAL_TO_GYM = {"wait": "select"}

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

        self._seed: int = seed
        self._env: Any = None

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

        internal_action = self._GYM_TO_INTERNAL.get(action_str, action_str)

        result: StepResult = self._env.step(internal_action)

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
            for action_name in self._env.get_actions():
                std_name = self._INTERNAL_TO_GYM.get(action_name, action_name)
                idx = self._string_to_action.get(std_name)
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
