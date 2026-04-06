import struct
import zlib
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

C_FLOOR       = 0
C_OFFWHITE    = 1
C_LTGREY      = 2
C_GREY        = 3
C_DKGREY      = 4
C_WALL        = 5
C_STRONG_EN   = 6
C_WEAPON      = 7
C_WEAK_EN     = 8
C_PLAYER      = 9
C_LTBLUE      = 10
C_TILE_OFF    = 11
C_NORMAL_EN   = 12
C_LAVA        = 13
C_TILE_ON     = 14
C_TRAP        = 15

BG_COLOR  = 0
PAD_COLOR = 5
SCALE = 3

DIR_N: Tuple[int, int] = (0, -1)
DIR_S: Tuple[int, int] = (0, 1)
DIR_E: Tuple[int, int] = (1, 0)
DIR_W: Tuple[int, int] = (-1, 0)
DIR_NONE: Tuple[int, int] = (0, 0)

DETECT_STRONG: Dict[Tuple[int, int], List[Tuple[int, int]]] = {
    DIR_N: [(0, -1), (0, -2), (-1, 0), (1, 0), (-1, -1), (1, -1)],
    DIR_S: [(0, 1), (0, 2), (-1, 0), (1, 0), (-1, 1), (1, 1)],
    DIR_E: [(1, 0), (2, 0), (0, -1), (0, 1), (1, -1), (1, 1)],
    DIR_W: [(-1, 0), (-2, 0), (0, -1), (0, 1), (-1, -1), (-1, 1)],
}

DETECT_FAST: Dict[Tuple[int, int], List[Tuple[int, int]]] = {
    DIR_N: [
        (0, -1), (0, -2), (0, -3), (0, -4),
        (-1, 0), (1, 0),
        (-1, -1), (1, -1),
        (-2, -1), (2, -1),
    ],
    DIR_S: [
        (0, 1), (0, 2), (0, 3), (0, 4),
        (-1, 0), (1, 0),
        (-1, 1), (1, 1),
        (-2, 1), (2, 1),
    ],
    DIR_E: [
        (1, 0), (2, 0), (3, 0), (4, 0),
        (0, -1), (0, 1),
        (1, -1), (1, 1),
        (1, -2), (1, 2),
    ],
    DIR_W: [
        (-1, 0), (-2, 0), (-3, 0), (-4, 0),
        (0, -1), (0, 1),
        (-1, -1), (-1, 1),
        (-1, -2), (-1, 2),
    ],
}

CHASE_TIMEOUT    = 8
SAFE_WINDOW      = 5
TILE_CARRY_STEPS = 5

MAX_LIVES  = 3
MAX_MOVES  = 400
C_LIFE_ON  = 8
C_LIFE_OFF = 4

_sp = {
    "player": Sprite(
        pixels=[[C_PLAYER] * SCALE for _ in range(SCALE)],
        name="player", visible=True, collidable=True,
        tags=["player"], layer=5,
    ),
    "wall": Sprite(
        pixels=[[C_WALL] * SCALE for _ in range(SCALE)],
        name="wall", visible=True, collidable=True,
        tags=["wall"], layer=0,
    ),
    "stone": Sprite(
        pixels=[[C_LTGREY] * SCALE for _ in range(SCALE)],
        name="stone", visible=True, collidable=True,
        tags=["stone"], layer=1,
    ),
    "tile_off": Sprite(
        pixels=[[C_TILE_OFF] * SCALE for _ in range(SCALE)],
        name="tile_off", visible=True, collidable=False,
        tags=["tile"], layer=0,
    ),
    "tile_on": Sprite(
        pixels=[[C_TILE_ON] * SCALE for _ in range(SCALE)],
        name="tile_on", visible=True, collidable=False,
        tags=["tile_on"], layer=0,
    ),
    "weak_enemy": Sprite(
        pixels=[[C_WEAK_EN] * SCALE for _ in range(SCALE)],
        name="weak_enemy", visible=True, collidable=False,
        tags=["enemy", "weak_enemy"], layer=4,
    ),
    "normal_enemy": Sprite(
        pixels=[[C_NORMAL_EN] * SCALE for _ in range(SCALE)],
        name="normal_enemy", visible=True, collidable=False,
        tags=["enemy", "normal_enemy"], layer=4,
    ),
    "strong_enemy": Sprite(
        pixels=[[C_STRONG_EN] * SCALE for _ in range(SCALE)],
        name="strong_enemy", visible=True, collidable=False,
        tags=["enemy", "strong_enemy"], layer=4,
    ),
    "weapon": Sprite(
        pixels=[
            [C_WEAPON, C_WEAPON, C_WEAPON],
            [C_WEAPON, C_FLOOR,  C_WEAPON],
            [C_WEAPON, C_WEAPON, C_WEAPON],
        ],
        name="weapon", visible=True, collidable=False,
        tags=["weapon"], layer=2,
    ),
    "lava": Sprite(
        pixels=[[C_LAVA] * SCALE for _ in range(SCALE)],
        name="lava", visible=True, collidable=False,
        tags=["lava"], layer=0,
    ),
    "hole": Sprite(
        pixels=[[C_DKGREY] * SCALE for _ in range(SCALE)],
        name="hole", visible=True, collidable=False,
        tags=["hole"], layer=0,
    ),
    "trap": Sprite(
        pixels=[[C_TRAP] * SCALE for _ in range(SCALE)],
        name="trap", visible=True, collidable=False,
        tags=["trap_visual"], layer=0,
    ),
    "stack_target": Sprite(
        pixels=[[C_GREY] * SCALE for _ in range(SCALE)],
        name="stack_target", visible=True, collidable=False,
        tags=["stack_target"], layer=0,
    ),
}


def _h_patrol(y: int, x1: int, x2: int) -> List[Tuple[int, int]]:
    p: List[Tuple[int, int]] = []
    for x in range(x1, x2 + 1):
        p.append((x, y))
    for x in range(x2 - 1, x1, -1):
        p.append((x, y))
    return p


def _v_patrol(x: int, y1: int, y2: int) -> List[Tuple[int, int]]:
    p: List[Tuple[int, int]] = []
    for y in range(y1, y2 + 1):
        p.append((x, y))
    for y in range(y2 - 1, y1, -1):
        p.append((x, y))
    return p


def _rect_patrol(
    x1: int, y1: int, x2: int, y2: int,
) -> List[Tuple[int, int]]:
    p: List[Tuple[int, int]] = []
    for x in range(x1, x2):
        p.append((x, y1))
    for y in range(y1, y2):
        p.append((x2, y))
    for x in range(x2, x1, -1):
        p.append((x, y2))
    for y in range(y2, y1, -1):
        p.append((x1, y))
    return p


def _grid(rows: List[str]) -> str:
    w = len(rows[0])
    for i, r in enumerate(rows):
        if len(r) != w:
            raise ValueError(
                f"Row {i}: len={len(r)}, expected={w}"
            )
    return "".join(rows)


W1, H1 = 20, 19
G1 = _grid([
    "####################",
    "#P.....#...........#",
    "#..O...#....O......#",
    "#......#...........#",
    "#.####.#.####.##...#",
    "#......#.......#...#",
    "#..O...........#...#",
    "#..............#.O.#",
    "###.####.####.##...#",
    "#..................#",
    "#..O.......O.......#",
    "#.####.##.####.##..#",
    "#..............#...#",
    "#..O...........#...#",
    "#......T.......#...#",
    "#......K.......#...#",
    "#..................#",
    "#..................#",
    "####################",
])

ENEMIES_L1 = [
    {
        "start": (5, 5),
        "patrol": _h_patrol(5, 1, 6),
        "etype": "weak",
        "type": "dynamic",
    },
    {
        "start": (12, 10),
        "patrol": _h_patrol(10, 5, 17),
        "etype": "weak",
        "type": "dynamic",
    },
]

L1_SPAWNS = [
    [{"etype": "weak"}],
]

L1_WAVE_WEAPONS = [[(10, 9)]]
L1_WAVE_TRAPS = [[(10, 13)]]

W2, H2 = 20, 19
G2 = _grid([
    "####################",
    "#P...#.......#.....#",
    "#....#..O....#.....#",
    "#....####.##.#.....#",
    "#..............O...#",
    "#.##.##............#",
    "#.#.W.#....LL......#",
    "#.#...#....LL......#",
    "#.##.##............#",
    "#........####.###..#",
    "#..O.....#.........#",
    "#........#....O....#",
    "#........#.........#",
    "#..####.####.......#",
    "#........O.........#",
    "#....T.T...........#",
    "#......K...........#",
    "#..........O.......#",
    "####################",
])

ENEMIES_L2 = [
    {
        "start": (8, 4),
        "patrol": _h_patrol(4, 5, 16),
        "etype": "weak",
        "type": "dynamic",
    },
    {
        "start": (15, 10),
        "patrol": _v_patrol(15, 9, 17),
        "etype": "normal",
        "type": "dynamic",
    },
    {
        "start": (4, 14),
        "patrol": _h_patrol(14, 1, 10),
        "etype": "weak",
        "type": "dynamic",
    },
]

L2_SPAWNS = [
    [{"etype": "weak"}],
    [{"etype": "normal"}, {"etype": "weak"}],
]

L2_WAVE_WEAPONS = [[(9, 14)], [(3, 10)]]
L2_WAVE_TRAPS = [[(12, 14)], [(5, 11)]]

W3, H3 = 20, 19
G3 = _grid([
    "####################",
    "#P...#.....#.......#",
    "#..O.#..O..#..O....#",
    "#....####.##.......#",
    "#.##.##....####.#..#",
    "#.#W..#........#...#",
    "#.##.##.LLLLLL.#...#",
    "#.......L....L.....#",
    "#..O....L....L..O..#",
    "#.......LLLLLL.....#",
    "#........O...####..#",
    "#.####.##..........#",
    "#..........O.......#",
    "#....X.............#",
    "#......T.T.........#",
    "#.......K..........#",
    "#..................#",
    "#..................#",
    "####################",
])

ENEMIES_L3 = [
    {
        "start": (10, 3),
        "patrol": _h_patrol(3, 5, 17),
        "etype": "weak",
        "type": "dynamic",
    },
    {
        "start": (3, 9),
        "patrol": _v_patrol(3, 5, 16),
        "etype": "normal",
        "type": "dynamic",
    },
    {
        "start": (16, 8),
        "patrol": _v_patrol(16, 5, 14),
        "etype": "strong",
        "type": "dynamic",
    },
]

L3_SPAWNS = [
    [{"etype": "weak"}],
    [{"etype": "normal"}, {"etype": "weak"}],
]

L3_WAVE_WEAPONS = [[(4, 4)], [(12, 13)]]
L3_WAVE_TRAPS = [
    [(10, 13)],
    [(3, 17)],
]

W4, H4 = 20, 19
G4 = _grid([
    "####################",
    "#P...#..#..........#",
    "#.O..#..#..O.......#",
    "#....##.####.##..O.#",
    "#..............#...#",
    "#.##.##..HH....#...#",
    "#.#W..#..HH........#",
    "#.##.##............#",
    "#......LLLL..####..#",
    "#......LLLL..#.....#",
    "#..O.........#..O..#",
    "#....####.####.....#",
    "#....#.........##..#",
    "#.X..#...O.........#",
    "#....####.####.....#",
    "#....T.T.......O...#",
    "#.....K............#",
    "#....T.............#",
    "####################",
])

ENEMIES_L4 = [
    {
        "start": (10, 4),
        "patrol": _h_patrol(4, 5, 17),
        "etype": "normal",
        "type": "dynamic",
    },
    {
        "start": (16, 10),
        "patrol": _v_patrol(16, 7, 17),
        "etype": "normal",
        "type": "dynamic",
    },
    {
        "start": (5, 8),
        "patrol": _rect_patrol(2, 5, 6, 10),
        "etype": "strong",
        "type": "dynamic",
    },
    {
        "start": (13, 13),
        "patrol": _h_patrol(13, 7, 17),
        "etype": "strong",
        "type": "dynamic",
    },
    {
        "start": (3, 14),
        "patrol": _h_patrol(14, 1, 8),
        "etype": "weak",
        "type": "dynamic",
    },
]

L4_SPAWNS = [
    [{"etype": "weak"}, {"etype": "weak"}],
    [{"etype": "normal"}],
    [{"etype": "weak"}, {"etype": "normal"}],
]

L4_WAVE_WEAPONS = [[(8, 10)], [(2, 14)], [(8, 10)]]
L4_WAVE_TRAPS = [
    [(9, 4)],
    [(11, 10)],
    [(12, 7)],
]

SPAWN_TABLES = [L1_SPAWNS, L2_SPAWNS, L3_SPAWNS, L4_SPAWNS]
WAVE_WEAPONS = [L1_WAVE_WEAPONS, L2_WAVE_WEAPONS, L3_WAVE_WEAPONS, L4_WAVE_WEAPONS]
WAVE_TRAPS = [L1_WAVE_TRAPS, L2_WAVE_TRAPS, L3_WAVE_TRAPS, L4_WAVE_TRAPS]

LEVEL_DATA = [
    {"grid": G1, "w": W1, "h": H1, "enemies": ENEMIES_L1},
    {"grid": G2, "w": W2, "h": H2, "enemies": ENEMIES_L2},
    {"grid": G3, "w": W3, "h": H3, "enemies": ENEMIES_L3},
    {"grid": G4, "w": W4, "h": H4, "enemies": ENEMIES_L4},
]


def _parse_grid(
    grid_str: str, w: int, h: int,
    level_idx: int, enemies: List[Dict],
) -> Level:
    spr_list: List[Sprite] = []
    px, py = 0, 0
    walls: List[Dict] = []
    stones: List[Dict] = []
    tiles: List[Dict] = []
    weapons: List[Dict] = []
    lavas: List[Dict] = []
    holes: List[Dict] = []
    traps: List[Dict] = []
    stack_pos: Dict[str, int] = {"x": 0, "y": 0}

    for y in range(h):
        for x in range(w):
            ch = grid_str[y * w + x]
            sx, sy = x * SCALE, y * SCALE
            if ch == "#":
                spr_list.append(_sp["wall"].clone().set_position(sx, sy))
                walls.append({"x": x, "y": y})
            elif ch == "P":
                px, py = x, y
                spr_list.append(_sp["player"].clone().set_position(sx, sy))
            elif ch == "T":
                tiles.append({"x": x, "y": y})
                spr_list.append(_sp["tile_off"].clone().set_position(sx, sy))
            elif ch == "O":
                stones.append({"x": x, "y": y})
                spr_list.append(_sp["stone"].clone().set_position(sx, sy))
            elif ch == "W":
                weapons.append({"x": x, "y": y})
                spr_list.append(_sp["weapon"].clone().set_position(sx, sy))
            elif ch == "L":
                lavas.append({"x": x, "y": y})
                spr_list.append(_sp["lava"].clone().set_position(sx, sy))
            elif ch == "H":
                holes.append({"x": x, "y": y})
                spr_list.append(_sp["hole"].clone().set_position(sx, sy))
            elif ch == "X":
                traps.append({"x": x, "y": y})
                spr_list.append(_sp["trap"].clone().set_position(sx, sy))
            elif ch == "K":
                stack_pos = {"x": x, "y": y}
                spr_list.append(
                    _sp["stack_target"].clone().set_position(sx, sy)
                )

    for ed in enemies:
        ex, ey = ed["start"]
        et = ed["etype"]
        tpl = ("weak_enemy" if et == "weak"
               else "normal_enemy" if et == "normal"
               else "strong_enemy")
        spr_list.append(_sp[tpl].clone().set_position(ex * SCALE, ey * SCALE))

    return Level(
        sprites=spr_list,
        grid_size=(w * SCALE, h * SCALE),
        data={
            "level_idx": level_idx,
            "logical_width": w, "logical_height": h,
            "player_x": px, "player_y": py,
            "walls": walls, "stones": stones,
            "tiles": tiles, "weapons": weapons,
            "lavas": lavas, "holes": holes,
            "traps": traps, "stack_pos": stack_pos,
            "enemy_data": enemies,
        },
        name=f"Level {level_idx + 1}",
    )


def _build_levels() -> List[Level]:
    return [
        _parse_grid(ld["grid"], ld["w"], ld["h"], i, ld["enemies"])
        for i, ld in enumerate(LEVEL_DATA)
    ]


class GameHud(RenderableUserDisplay):
    HUD_Y = 57

    def __init__(self, game: "Cr01") -> None:
        self._g = game

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        g = self._g
        hy = self.HUD_Y
        frame[hy:, :] = C_WALL
        frame[hy, :] = C_GREY

        lives_start = 2
        for i in range(MAX_LIVES):
            lx = lives_start + i * 3
            color = C_LIFE_ON if i < g._lives else C_LIFE_OFF
            frame[hy + 1:hy + 3, lx:lx + 2] = color

        weapon_x = 59
        frame[hy + 1:hy + 3, weapon_x:weapon_x + 3] = (
            C_WEAPON if g._has_weapon else C_DKGREY
        )

        total = len(g._tile_positions)
        if total > 0:
            tile_start = lives_start + MAX_LIVES * 3 + 2
            for i, pos in enumerate(g._tile_positions):
                tx = tile_start + i * 3
                if pos in g._collected_tiles:
                    color = C_GREY
                elif g._tile_states.get(pos, False):
                    color = C_TILE_ON
                else:
                    color = C_TILE_OFF
                frame[hy + 1:hy + 3, tx:tx + 2] = color

        bar_start = 2
        bar_max_width = 60
        if MAX_MOVES > 0:
            remaining = max(0, MAX_MOVES - g._move_count)
            pct = remaining / MAX_MOVES
            fill_width = int(pct * bar_max_width)
            if pct > 0.5:
                bar_color = 14
            elif pct > 0.25:
                bar_color = 11
            else:
                bar_color = 8
            if fill_width > 0:
                frame[hy + 3, bar_start:bar_start + fill_width] = bar_color

        return frame


class Cr01(ARCBaseGame):

    def __init__(self, seed: int = 0) -> None:
        levels = _build_levels()
        self._hud = GameHud(self)

        self._player: Optional[Sprite] = None
        self._player_x: int = 0
        self._player_y: int = 0
        self._alive: bool = True
        self._has_weapon: bool = False
        self._last_dir: Tuple[int, int] = DIR_S

        self._total_tiles: int = 0
        self._tile_positions: List[Tuple[int, int]] = []
        self._tile_map: Dict[Tuple[int, int], Sprite] = {}
        self._tile_states: Dict[Tuple[int, int], bool] = {}
        self._activation_wave: int = 0

        self._carrying_tile: bool = False
        self._carry_steps: int = 0
        self._carried_from: List[Tuple[int, int]] = []
        self._carry_count: int = 0

        self._phase: str = "kill"
        self._collected_count: int = 0
        self._collected_tiles: Set[Tuple[int, int]] = set()
        self._stack_pos: Tuple[int, int] = (0, 0)

        self._safe_timer: int = 0

        self._wall_set: Set[Tuple[int, int]] = set()
        self._stone_set: Set[Tuple[int, int]] = set()
        self._lava_set: Set[Tuple[int, int]] = set()
        self._hole_set: Set[Tuple[int, int]] = set()
        self._trap_set: Set[Tuple[int, int]] = set()
        self._log_w: int = 0
        self._log_h: int = 0

        self._enemies: List[Dict] = []
        self._enemy_sprites: List[Optional[Sprite]] = []

        self._weapon_map: Dict[Tuple[int, int], Sprite] = {}
        self._weapon_respawn: List[Tuple[int, int]] = []

        self._spawned_trap_sprites: List[Sprite] = []

        self._lives: int = MAX_LIVES
        self._move_count: int = 0
        self._init_player_x: int = 0
        self._init_player_y: int = 0
        self._history: List[Dict] = []

        cam_w = 64
        cam_h = 64
        camera = Camera(
            0, 0, cam_w, cam_h,
            BG_COLOR, PAD_COLOR,
            [self._hud],
        )

        super().__init__(
            game_id="cr01",
            levels=levels,
            camera=camera,
            available_actions=[0, 1, 2, 3, 4, 5, 7],
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        w = self.current_level.get_data("logical_width")
        h = self.current_level.get_data("logical_height")
        self.camera.width = 64
        self.camera.height = 64
        self._log_w = w
        self._log_h = h

        self._alive = True
        self._has_weapon = False
        self._last_dir = DIR_S
        self._carrying_tile = False
        self._carry_steps = 0
        self._carried_from = []
        self._carry_count = 0
        self._safe_timer = 0
        self._collected_count = 0
        self._collected_tiles = set()
        self._activation_wave = 0
        self._spawned_trap_sprites = []
        self._history = []

        self._player = self.current_level.get_sprites_by_tag("player")[0]
        self._player_x = self.current_level.get_data("player_x")
        self._player_y = self.current_level.get_data("player_y")
        self._init_player_x = self._player_x
        self._init_player_y = self._player_y

        self._lives = MAX_LIVES
        self._move_count = 0

        self._wall_set = {(p["x"], p["y"]) for p in self.current_level.get_data("walls")}
        self._stone_set = {(p["x"], p["y"]) for p in self.current_level.get_data("stones")}
        self._lava_set = {(p["x"], p["y"]) for p in self.current_level.get_data("lavas")}
        self._hole_set = {(p["x"], p["y"]) for p in self.current_level.get_data("holes")}
        self._trap_set = {(p["x"], p["y"]) for p in self.current_level.get_data("traps")}

        sp_data = self.current_level.get_data("stack_pos")
        self._stack_pos = (sp_data["x"], sp_data["y"])

        self._tile_map = {}
        self._tile_states = {}
        self._tile_positions = []
        tile_data = self.current_level.get_data("tiles")
        tile_sprites = self.current_level.get_sprites_by_tag("tile")
        self._total_tiles = len(tile_data)
        for td in tile_data:
            pos = (td["x"], td["y"])
            self._tile_states[pos] = False
            self._tile_positions.append(pos)
            for sp in tile_sprites:
                if sp.x == pos[0] * SCALE and sp.y == pos[1] * SCALE:
                    self._tile_map[pos] = sp
                    break

        self._weapon_map = {}
        self._weapon_respawn = []
        weapon_data = self.current_level.get_data("weapons")
        weapon_sprites = self.current_level.get_sprites_by_tag("weapon")
        for wd in weapon_data:
            pos = (wd["x"], wd["y"])
            self._weapon_respawn.append(pos)
            for sp in weapon_sprites:
                if sp.x == pos[0] * SCALE and sp.y == pos[1] * SCALE:
                    self._weapon_map[pos] = sp
                    break

        enemy_data = self.current_level.get_data("enemy_data")
        all_enemy_sprites = self.current_level.get_sprites_by_tag("enemy")
        self._enemies = []
        self._enemy_sprites = []
        for ed in enemy_data:
            sx, sy = ed["start"]
            patrol = ed.get("patrol", [])
            facing = DIR_S
            if patrol and len(patrol) > 1:
                ddx = patrol[1][0] - patrol[0][0]
                ddy = patrol[1][1] - patrol[0][1]
                if ddx > 0: facing = DIR_E
                elif ddx < 0: facing = DIR_W
                elif ddy > 0: facing = DIR_S
                elif ddy < 0: facing = DIR_N
            self._enemies.append({
                "x": sx, "y": sy,
                "patrol": patrol, "patrol_idx": 0,
                "etype": ed["etype"], "alive": True,
                "chasing": False, "chase_turns": 0,
                "facing": facing,
                "type": ed.get("type", "dynamic"),
            })
            matched = None
            for sp in all_enemy_sprites:
                if sp.x == sx * SCALE and sp.y == sy * SCALE:
                    matched = sp
                    break
            self._enemy_sprites.append(matched)

        initial_normals = sum(1 for e in self._enemies if e["etype"] == "normal")
        initial_weapons = len(self._weapon_map)
        extra_needed = initial_normals - initial_weapons
        if extra_needed > 0:
            wep_candidates: List[Tuple[int, int]] = []
            for x in range(1, self._log_w - 1):
                for y in range(1, self._log_h - 1):
                    wp = (x, y)
                    if wp in self._wall_set or wp in self._stone_set:
                        continue
                    if wp in self._weapon_map:
                        continue
                    if self._is_empty(x, y):
                        wep_candidates.append(wp)
            for wp in wep_candidates:
                if extra_needed <= 0:
                    break
                wsp = _sp["weapon"].clone().set_position(
                    wp[0] * SCALE, wp[1] * SCALE
                )
                self.current_level.add_sprite(wsp)
                self._weapon_map[wp] = wsp
                extra_needed -= 1

        self._update_phase()

    def _any_enemy_alive(self) -> bool:
        return any(e["alive"] for e in self._enemies)

    def _all_tiles_active(self) -> bool:
        for p in self._tile_positions:
            if p in self._collected_tiles:
                continue
            if self._carrying_tile and p in self._carried_from:
                continue
            if not self._tile_states.get(p, False):
                return False
        return True

    def _is_empty(self, x: int, y: int) -> bool:
        pos = (x, y)
        if pos in self._wall_set or pos in self._stone_set:
            return False
        if pos in self._lava_set or pos in self._hole_set:
            return False
        if pos in self._trap_set:
            return False
        if x == self._player_x and y == self._player_y:
            return False
        for e in self._enemies:
            if e["alive"] and e["x"] == x and e["y"] == y:
                return False
        return True

    def _is_walkable(self, x: int, y: int) -> bool:
        if (x, y) in self._wall_set or (x, y) in self._stone_set:
            return False
        return True

    def _update_phase(self) -> None:
        enemies_alive = self._any_enemy_alive()
        all_active = self._all_tiles_active()

        if enemies_alive:
            self._phase = "kill"
        elif all_active:
            self._phase = "collect"
        else:
            self._phase = "activate"

    def _try_move(self, dx: int, dy: int) -> bool:
        nx = self._player_x + dx
        ny = self._player_y + dy
        self._last_dir = (dx, dy)

        if nx < 0 or ny < 0 or nx >= self._log_w or ny >= self._log_h:
            return False
        if (nx, ny) in self._wall_set or (nx, ny) in self._stone_set:
            return False
        for e in self._enemies:
            if e["alive"] and e["chasing"] and e["x"] == nx and e["y"] == ny:
                return False

        self._player_x = nx
        self._player_y = ny
        if self._player is not None:
            self._player.set_position(nx * SCALE, ny * SCALE)

        self._check_weapon_pickup()
        self._check_hazards()

        if self._carrying_tile:
            self._carry_steps += 1
            if self._carry_steps >= TILE_CARRY_STEPS:
                self._drop_tile()

        return True

    def _check_weapon_pickup(self) -> None:
        pos = (self._player_x, self._player_y)
        if pos in self._weapon_map:
            self._has_weapon = True
            sp = self._weapon_map.pop(pos)
            self.current_level.remove_sprite(sp)

    def _check_hazards(self) -> None:
        pos = (self._player_x, self._player_y)
        if pos in self._lava_set or pos in self._hole_set:
            self._alive = False
        if pos in self._trap_set:
            self._alive = False

    def _interact(self) -> None:
        px, py = self._player_x, self._player_y
        pos = (px, py)

        if self._try_attack():
            return

        if self._phase == "collect" and self._carrying_tile:
            if pos == self._stack_pos:
                self._stack_tile()
                return

        if self._phase == "collect" and not self._carrying_tile:
            if pos in self._tile_states and self._tile_states[pos] and pos not in self._collected_tiles:
                self._pickup_tile(pos)
                return

        if pos in self._tile_states and pos not in self._collected_tiles:
            if self._phase == "activate":
                if not self._tile_states[pos]:
                    self._activate_tile(pos)

    def _set_tile_visual(self, pos: Tuple[int, int], active: bool) -> None:
        if pos in self._tile_map:
            old_sp = self._tile_map[pos]
            self.current_level.remove_sprite(old_sp)
        tpl = "tile_on" if active else "tile_off"
        new_sp = _sp[tpl].clone().set_position(
            pos[0] * SCALE, pos[1] * SCALE
        )
        self.current_level.add_sprite(new_sp)
        self._tile_map[pos] = new_sp

    def _activate_tile(self, pos: Tuple[int, int]) -> None:
        self._tile_states[pos] = True
        self._set_tile_visual(pos, True)

        self._safe_timer = SAFE_WINDOW

        self._spawn_wave()

        self._update_phase()

    def _spawn_hazards_near(
        self, sx: int, sy: int,
    ) -> None:
        hazard_candidates: List[Tuple[int, int]] = []
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                hx, hy = sx + dx, sy + dy
                dist = abs(dx) + abs(dy)
                if dist < 2 or dist > 3:
                    continue
                if hx < 1 or hy < 1 or hx >= self._log_w - 1 or hy >= self._log_h - 1:
                    continue
                hpos = (hx, hy)
                if hpos in self._wall_set or hpos in self._stone_set:
                    continue
                if hpos in self._lava_set or hpos in self._hole_set:
                    continue
                if hx == self._player_x and hy == self._player_y:
                    continue
                if hpos in self._tile_states or hpos == self._stack_pos:
                    continue
                hazard_candidates.append(hpos)

        if not hazard_candidates:
            return

        placed = 0
        for hpos in hazard_candidates:
            if placed >= 2:
                break
            self._lava_set.add(hpos)
            lsp = _sp["lava"].clone().set_position(
                hpos[0] * SCALE, hpos[1] * SCALE
            )
            self.current_level.add_sprite(lsp)
            placed += 1

        for hpos in hazard_candidates[placed:]:
            self._hole_set.add(hpos)
            hsp = _sp["hole"].clone().set_position(
                hpos[0] * SCALE, hpos[1] * SCALE
            )
            self.current_level.add_sprite(hsp)
            break

    def _spawn_wave(self) -> None:
        level_idx = self.current_level.get_data("level_idx")
        table = SPAWN_TABLES[level_idx]
        wave_idx = self._activation_wave % len(table)
        self._activation_wave += 1
        wave = table[wave_idx]

        mid_x = self._log_w // 2
        mid_y = self._log_h // 2

        for spawn_def in wave:
            etype = spawn_def["etype"]
            tpl = ("weak_enemy" if etype == "weak"
                   else "normal_enemy" if etype == "normal"
                   else "strong_enemy")

            alive_positions = [
                (e["x"], e["y"]) for e in self._enemies if e["alive"]
            ]

            candidates = []
            for x in range(1, self._log_w - 1):
                for y in range(1, self._log_h - 1):
                    if self._is_empty(x, y):
                        dist_player = abs(x - self._player_x) + abs(y - self._player_y)
                        if dist_player < 8:
                            continue
                        too_close = False
                        for ex, ey in alive_positions:
                            if abs(x - ex) + abs(y - ey) < 5:
                                too_close = True
                                break
                        if not too_close:
                            candidates.append((x, y))
            if not candidates:
                for x in range(1, self._log_w - 1):
                    for y in range(1, self._log_h - 1):
                        if self._is_empty(x, y):
                            dist_player = abs(x - self._player_x) + abs(y - self._player_y)
                            if dist_player < 5:
                                continue
                            too_close = False
                            for ex, ey in alive_positions:
                                if abs(x - ex) + abs(y - ey) < 4:
                                    too_close = True
                                    break
                            if not too_close:
                                candidates.append((x, y))
            if not candidates:
                continue

            candidates.sort(key=lambda c: abs(c[0] - mid_x) + abs(c[1] - mid_y))
            sx, sy = candidates[0]

            margin_left = sx - 1
            margin_right = (self._log_w - 2) - sx
            margin_top = sy - 1
            margin_bottom = (self._log_h - 2) - sy

            if margin_top <= margin_bottom and margin_top <= margin_left and margin_top <= margin_right:
                patrol = _v_patrol(sx, sy, min(self._log_h - 2, sy + 4))
                facing = DIR_S
            elif margin_bottom <= margin_top and margin_bottom <= margin_left and margin_bottom <= margin_right:
                patrol = _v_patrol(sx, max(1, sy - 4), sy)
                facing = DIR_N
            elif margin_left <= margin_right:
                patrol = _h_patrol(sy, sx, min(self._log_w - 2, sx + 4))
                facing = DIR_E
            else:
                patrol = _h_patrol(sy, max(1, sx - 4), sx)
                facing = DIR_W

            self._enemies.append({
                "x": sx, "y": sy,
                "patrol": patrol, "patrol_idx": 0,
                "etype": etype, "alive": True,
                "chasing": False, "chase_turns": 0,
                "facing": facing, "type": "dynamic",
            })
            sp = _sp[tpl].clone().set_position(sx * SCALE, sy * SCALE)
            self.current_level.add_sprite(sp)
            self._enemy_sprites.append(sp)

            if etype == "strong":
                self._spawn_hazards_near(sx, sy)

        wave_trap_idx = (self._activation_wave - 1) % len(WAVE_TRAPS[level_idx])
        trap_positions = WAVE_TRAPS[level_idx][wave_trap_idx]
        for tx, ty in trap_positions:
            tpos = (tx, ty)
            if tpos not in self._trap_set and tpos not in self._wall_set and tpos not in self._stone_set:
                if not (tx == self._player_x and ty == self._player_y):
                    if tpos not in self._tile_states and tpos != self._stack_pos:
                        self._trap_set.add(tpos)
                        trap_sp = _sp["trap"].clone().set_position(
                            tx * SCALE, ty * SCALE
                        )
                        self.current_level.add_sprite(trap_sp)
                        self._spawned_trap_sprites.append(trap_sp)
                        break

        normal_count = sum(1 for s in wave if s["etype"] == "normal")
        weapons_placed = 0
        if normal_count > 0:
            wep_candidates: List[Tuple[int, int]] = []
            for x in range(1, self._log_w - 1):
                for y in range(1, self._log_h - 1):
                    wp = (x, y)
                    if wp in self._wall_set or wp in self._stone_set:
                        continue
                    if wp in self._weapon_map:
                        continue
                    if self._is_empty(x, y) or wp in self._trap_set:
                        wep_candidates.append(wp)
            for wp in wep_candidates:
                if weapons_placed >= normal_count:
                    break
                wsp = _sp["weapon"].clone().set_position(
                    wp[0] * SCALE, wp[1] * SCALE
                )
                self.current_level.add_sprite(wsp)
                self._weapon_map[wp] = wsp
                weapons_placed += 1

        if normal_count > 0 and weapons_placed == 0 and not self._weapon_map:
            for wpos in self._weapon_respawn:
                if self._is_empty(wpos[0], wpos[1]):
                    wsp = _sp["weapon"].clone().set_position(
                        wpos[0] * SCALE, wpos[1] * SCALE
                    )
                    self.current_level.add_sprite(wsp)
                    self._weapon_map[wpos] = wsp
                    break

    def _pickup_tile(self, pos: Tuple[int, int]) -> None:
        picked: List[Tuple[int, int]] = []
        for p in list(self._tile_positions):
            if p == pos and self._tile_states.get(p, False) and p not in self._collected_tiles:
                picked.append(p)
                if p in self._tile_map:
                    sp = self._tile_map.pop(p)
                    self.current_level.remove_sprite(sp)
                self._tile_states[p] = False
        if not picked:
            return
        self._carrying_tile = True
        self._carry_steps = 0
        self._carried_from = picked
        self._carry_count = len(picked)

    def _drop_tile(self) -> None:
        if not self._carrying_tile:
            return
        self._carrying_tile = False
        self._carry_steps = 0
        px, py = self._player_x, self._player_y
        pos = (px, py)
        new_sp = _sp["tile_on"].clone().set_position(
            px * SCALE, py * SCALE
        )
        self.current_level.add_sprite(new_sp)
        self._tile_map[pos] = new_sp
        self._tile_states[pos] = True
        for orig in self._carried_from:
            if orig in self._tile_positions:
                idx = self._tile_positions.index(orig)
                self._tile_positions[idx] = pos
                if orig in self._tile_states and orig != pos:
                    del self._tile_states[orig]
        self._carried_from = []
        self._carry_count = 0

    def _stack_tile(self) -> None:
        if not self._carrying_tile:
            return
        self._carrying_tile = False
        self._carry_steps = 0
        count = self._carry_count
        for orig in self._carried_from:
            self._collected_tiles.add(orig)
        self._carried_from = []
        self._carry_count = 0
        self._collected_count += count
        if self._collected_count >= self._total_tiles:
            self.next_level()

    def _try_attack(self) -> bool:
        px, py = self._player_x, self._player_y
        check = [(px, py)]
        for ddx, ddy in [DIR_N, DIR_S, DIR_E, DIR_W]:
            check.append((px + ddx, py + ddy))

        targets: List[int] = []
        for i, e in enumerate(self._enemies):
            if e["alive"] and (e["x"], e["y"]) in check:
                targets.append(i)

        if not targets:
            return False

        idx = targets[0]
        etype = self._enemies[idx]["etype"]

        if etype == "weak":
            self._kill_enemy(idx)
        elif etype == "normal":
            if self._has_weapon:
                self._has_weapon = False
                self._kill_enemy(idx)
            else:
                self._alive = False
        elif etype == "strong":
            self._alive = False

        return True

    def _kill_enemy(self, idx: int) -> None:
        self._enemies[idx]["alive"] = False
        sp = self._enemy_sprites[idx]
        if sp is not None:
            sp.set_visible(False)
        self._update_phase()

    def _blocks_los(self, x: int, y: int) -> bool:
        return (x, y) in self._wall_set or (x, y) in self._stone_set

    def _enemy_detects_player(self, e: Dict) -> bool:
        facing = e.get("facing", DIR_NONE)
        if facing == DIR_NONE:
            return False
        etype = e.get("etype", "weak")
        if etype == "strong":
            detect_table = DETECT_STRONG
        else:
            detect_table = DETECT_FAST
        if facing not in detect_table:
            return False
        ex, ey = e["x"], e["y"]
        offsets = detect_table[facing]
        for i, (ox, oy) in enumerate(offsets):
            tx, ty = ex + ox, ey + oy
            if tx < 0 or ty < 0 or tx >= self._log_w or ty >= self._log_h:
                continue
            blocked = False
            if etype == "strong":
                if i == 1:
                    if self._blocks_los(ex + offsets[0][0], ey + offsets[0][1]):
                        blocked = True
                elif i >= 4:
                    if self._blocks_los(ex + offsets[0][0], ey + offsets[0][1]):
                        blocked = True
            else:
                if i == 1 and self._blocks_los(ex + offsets[0][0], ey + offsets[0][1]):
                    blocked = True
                elif i == 2 and (
                    self._blocks_los(ex + offsets[0][0], ey + offsets[0][1])
                    or self._blocks_los(ex + offsets[1][0], ey + offsets[1][1])
                ):
                    blocked = True
                elif i == 3 and (
                    self._blocks_los(ex + offsets[0][0], ey + offsets[0][1])
                    or self._blocks_los(ex + offsets[1][0], ey + offsets[1][1])
                    or self._blocks_los(ex + offsets[2][0], ey + offsets[2][1])
                ):
                    blocked = True
                elif i >= 6 and self._blocks_los(ex + offsets[0][0], ey + offsets[0][1]):
                    blocked = True
            if not blocked and not self._blocks_los(tx, ty):
                if tx == self._player_x and ty == self._player_y:
                    return True
        return False

    def _patrol_step(self, e: Dict) -> None:
        patrol = e["patrol"]
        if not patrol:
            self._dynamic_step(e)
            return
        ct = patrol[e["patrol_idx"]]
        dist = abs(e["x"] - ct[0]) + abs(e["y"] - ct[1])
        if dist > 1:
            best_i, best_d = 0, 9999
            for pi, (ppx, ppy) in enumerate(patrol):
                d = abs(e["x"] - ppx) + abs(e["y"] - ppy)
                if d < best_d:
                    best_d = d
                    best_i = pi
            e["patrol_idx"] = best_i
            e["x"], e["y"] = patrol[best_i]
            return
        e["patrol_idx"] = (e["patrol_idx"] + 1) % len(patrol)
        e["x"], e["y"] = patrol[e["patrol_idx"]]

    def _chase_step(self, e: Dict) -> None:
        px, py = self._player_x, self._player_y
        ex, ey = e["x"], e["y"]
        best_dir = None
        best_d = abs(px - ex) + abs(py - ey)
        for ddx, ddy in [DIR_N, DIR_S, DIR_E, DIR_W]:
            nx, ny = ex + ddx, ey + ddy
            if nx < 0 or ny < 0 or nx >= self._log_w or ny >= self._log_h:
                continue
            if (nx, ny) in self._wall_set or (nx, ny) in self._stone_set:
                continue
            d = abs(px - nx) + abs(py - ny)
            if d < best_d:
                best_d = d
                best_dir = (ddx, ddy)
        if best_dir:
            e["x"] += best_dir[0]
            e["y"] += best_dir[1]

    def _dynamic_step(self, e: Dict, idx: int = 0) -> None:
        ex, ey = e["x"], e["y"]
        dirs = [DIR_N, DIR_S, DIR_E, DIR_W]
        offset = self._move_count + idx * 3 + ex + ey
        opts = []
        for k in range(4):
            ddx, ddy = dirs[(offset + k) % 4]
            nx, ny = ex + ddx, ey + ddy
            if nx < 0 or ny < 0 or nx >= self._log_w or ny >= self._log_h:
                continue
            if (nx, ny) in self._wall_set or (nx, ny) in self._stone_set:
                continue
            opts.append((ddx, ddy))
        if opts:
            d = opts[0]
            e["x"] += d[0]
            e["y"] += d[1]

    def _move_single_enemy(self, i: int, e: Dict) -> None:
        if not e["alive"]:
            return
        if self._enemy_detects_player(e):
            e["chasing"] = True
            e["chase_turns"] = CHASE_TIMEOUT
        elif e["chasing"]:
            e["chase_turns"] -= 1
            if e["chase_turns"] <= 0:
                e["chasing"] = False

        ox, oy = e["x"], e["y"]
        if e["chasing"]:
            self._chase_step(e)
        elif e.get("type") == "dynamic":
            self._dynamic_step(e, i)
        else:
            self._patrol_step(e)

        dx, dy = e["x"] - ox, e["y"] - oy
        if dx != 0 or dy != 0:
            if dx > 0: e["facing"] = DIR_E
            elif dx < 0: e["facing"] = DIR_W
            elif dy > 0: e["facing"] = DIR_S
            elif dy < 0: e["facing"] = DIR_N

            epos = (e["x"], e["y"])
            if e["chasing"] and (epos in self._lava_set or epos in self._hole_set):
                self._kill_enemy(i)
                return

        sp = self._enemy_sprites[i]
        if sp is not None and e["alive"]:
            sp.set_position(e["x"] * SCALE, e["y"] * SCALE)

    def _move_enemies(self) -> None:
        for i, e in enumerate(self._enemies):
            if not e["alive"]:
                continue
            etype = e.get("etype", "weak")
            if etype in ("weak", "normal"):
                self._move_single_enemy(i, e)
                if e["alive"]:
                    self._move_single_enemy(i, e)
            else:
                self._move_single_enemy(i, e)

    def _check_enemy_contact(self) -> None:
        for e in self._enemies:
            if not e["alive"]:
                continue
            if e["x"] == self._player_x and e["y"] == self._player_y:
                if e["chasing"]:
                    self._alive = False
                    return

    def _check_safe_timer(self) -> None:
        if self._safe_timer > 0:
            self._safe_timer -= 1
            if self._safe_timer <= 0:
                px, py = self._player_x, self._player_y
                for e in self._enemies:
                    if e["alive"] and e["chasing"] and abs(e["x"] - px) + abs(e["y"] - py) <= 1:
                        self._alive = False
                        return

    def _respawn_player(self) -> None:
        self._alive = True
        self._player_x = self._init_player_x
        self._player_y = self._init_player_y
        if self._player is not None:
            self._player.set_position(
                self._init_player_x * SCALE,
                self._init_player_y * SCALE,
            )
        if self._carrying_tile:
            self._carrying_tile = False
            self._carry_steps = 0
            self._carried_from = []
            self._carry_count = 0
        self._safe_timer = SAFE_WINDOW
        for e in self._enemies:
            if e["alive"]:
                e["chasing"] = False
                e["chase_turns"] = 0

    def _reinit_level(self) -> None:
        lvl = self.current_level

        for sp in self._enemy_sprites:
            if sp is not None:
                lvl.remove_sprite(sp)
        self._enemies.clear()
        self._enemy_sprites.clear()

        for sp in self._tile_map.values():
            lvl.remove_sprite(sp)
        self._tile_map.clear()
        self._tile_states.clear()
        self._tile_positions.clear()
        self._collected_tiles.clear()

        for sp in self._weapon_map.values():
            lvl.remove_sprite(sp)
        self._weapon_map.clear()

        for sp in self._spawned_trap_sprites:
            lvl.remove_sprite(sp)
        self._spawned_trap_sprites.clear()

        for tag in ("lava", "hole", "trap_visual"):
            for sp in lvl.get_sprites_by_tag(tag):
                lvl.remove_sprite(sp)

        self._lava_set = {(p["x"], p["y"]) for p in lvl.get_data("lavas")}
        self._hole_set = {(p["x"], p["y"]) for p in lvl.get_data("holes")}
        self._trap_set = {(p["x"], p["y"]) for p in lvl.get_data("traps")}

        for lp in self._lava_set:
            lvl.add_sprite(_sp["lava"].clone().set_position(lp[0] * SCALE, lp[1] * SCALE))
        for hp in self._hole_set:
            lvl.add_sprite(_sp["hole"].clone().set_position(hp[0] * SCALE, hp[1] * SCALE))
        for tp in self._trap_set:
            lvl.add_sprite(_sp["trap"].clone().set_position(tp[0] * SCALE, tp[1] * SCALE))

        tile_data = lvl.get_data("tiles")
        self._total_tiles = len(tile_data)
        for td in tile_data:
            pos = (td["x"], td["y"])
            self._tile_states[pos] = False
            self._tile_positions.append(pos)
            sp = _sp["tile_off"].clone().set_position(pos[0] * SCALE, pos[1] * SCALE)
            lvl.add_sprite(sp)
            self._tile_map[pos] = sp

        weapon_data = lvl.get_data("weapons")
        for wd in weapon_data:
            pos = (wd["x"], wd["y"])
            sp = _sp["weapon"].clone().set_position(pos[0] * SCALE, pos[1] * SCALE)
            lvl.add_sprite(sp)
            self._weapon_map[pos] = sp

        enemy_data = lvl.get_data("enemy_data")
        for ed in enemy_data:
            sx, sy = ed["start"]
            patrol = ed.get("patrol", [])
            facing = DIR_S
            if patrol and len(patrol) > 1:
                ddx = patrol[1][0] - patrol[0][0]
                ddy = patrol[1][1] - patrol[0][1]
                if ddx > 0: facing = DIR_E
                elif ddx < 0: facing = DIR_W
                elif ddy > 0: facing = DIR_S
                elif ddy < 0: facing = DIR_N
            self._enemies.append({
                "x": sx, "y": sy,
                "patrol": patrol, "patrol_idx": 0,
                "etype": ed["etype"], "alive": True,
                "chasing": False, "chase_turns": 0,
                "facing": facing,
                "type": ed.get("type", "dynamic"),
            })
            tpl = ("weak_enemy" if ed["etype"] == "weak"
                   else "normal_enemy" if ed["etype"] == "normal"
                   else "strong_enemy")
            sp = _sp[tpl].clone().set_position(sx * SCALE, sy * SCALE)
            lvl.add_sprite(sp)
            self._enemy_sprites.append(sp)

        initial_normals = sum(1 for e in self._enemies if e["etype"] == "normal")
        initial_weapons = len(self._weapon_map)
        extra_needed = initial_normals - initial_weapons
        if extra_needed > 0:
            wep_candidates: List[Tuple[int, int]] = []
            for x in range(1, self._log_w - 1):
                for y in range(1, self._log_h - 1):
                    wp = (x, y)
                    if wp in self._wall_set or wp in self._stone_set:
                        continue
                    if wp in self._weapon_map:
                        continue
                    if self._is_empty(x, y):
                        wep_candidates.append(wp)
            for wp in wep_candidates:
                if extra_needed <= 0:
                    break
                wsp = _sp["weapon"].clone().set_position(wp[0] * SCALE, wp[1] * SCALE)
                lvl.add_sprite(wsp)
                self._weapon_map[wp] = wsp
                extra_needed -= 1

        self._alive = True
        self._has_weapon = False
        self._last_dir = DIR_S
        self._carrying_tile = False
        self._carry_steps = 0
        self._carried_from = []
        self._carry_count = 0
        self._safe_timer = SAFE_WINDOW
        self._collected_count = 0
        self._activation_wave = 0
        self._move_count = 0
        self._history = []

        self._player_x = self._init_player_x
        self._player_y = self._init_player_y
        if self._player is not None:
            self._player.set_position(
                self._init_player_x * SCALE,
                self._init_player_y * SCALE,
            )

        self._update_phase()

    def _save_state(self) -> Dict:
        enemy_snapshots = []
        for e in self._enemies:
            enemy_snapshots.append({
                "x": e["x"],
                "y": e["y"],
                "patrol": list(e["patrol"]),
                "patrol_idx": e["patrol_idx"],
                "etype": e["etype"],
                "alive": e["alive"],
                "chasing": e["chasing"],
                "chase_turns": e["chase_turns"],
                "facing": e["facing"],
                "type": e["type"],
            })
        return {
            "player_x": self._player_x,
            "player_y": self._player_y,
            "alive": self._alive,
            "has_weapon": self._has_weapon,
            "last_dir": self._last_dir,
            "tile_states": dict(self._tile_states),
            "activation_wave": self._activation_wave,
            "carrying_tile": self._carrying_tile,
            "carry_steps": self._carry_steps,
            "carried_from": list(self._carried_from),
            "carry_count": self._carry_count,
            "phase": self._phase,
            "collected_count": self._collected_count,
            "collected_tiles": set(self._collected_tiles),
            "stack_pos": self._stack_pos,
            "safe_timer": self._safe_timer,
            "lava_set": set(self._lava_set),
            "hole_set": set(self._hole_set),
            "trap_set": set(self._trap_set),
            "enemies": enemy_snapshots,
            "weapon_positions": list(self._weapon_map.keys()),
            "spawned_trap_positions": [(s.x // SCALE, s.y // SCALE) for s in self._spawned_trap_sprites],
        }

    def _restore_state(self, snapshot: Dict) -> None:
        self._player_x = snapshot["player_x"]
        self._player_y = snapshot["player_y"]
        self._alive = snapshot["alive"]
        self._has_weapon = snapshot["has_weapon"]
        self._last_dir = snapshot["last_dir"]
        self._tile_states = dict(snapshot["tile_states"])
        self._activation_wave = snapshot["activation_wave"]
        self._carrying_tile = snapshot["carrying_tile"]
        self._carry_steps = snapshot["carry_steps"]
        self._carried_from = list(snapshot["carried_from"])
        self._carry_count = snapshot["carry_count"]
        self._phase = snapshot["phase"]
        self._collected_count = snapshot["collected_count"]
        self._collected_tiles = set(snapshot["collected_tiles"])
        self._stack_pos = snapshot["stack_pos"]
        self._safe_timer = snapshot["safe_timer"]

        if self._player is not None:
            self._player.set_position(
                self._player_x * SCALE, self._player_y * SCALE,
            )

        saved_enemies = snapshot["enemies"]
        while len(self._enemies) > len(saved_enemies):
            sp = self._enemy_sprites.pop()
            if sp is not None:
                self.current_level.remove_sprite(sp)
            self._enemies.pop()

        for i, es in enumerate(saved_enemies):
            if i < len(self._enemies):
                self._enemies[i] = {
                    "x": es["x"], "y": es["y"],
                    "patrol": list(es["patrol"]),
                    "patrol_idx": es["patrol_idx"],
                    "etype": es["etype"], "alive": es["alive"],
                    "chasing": es["chasing"], "chase_turns": es["chase_turns"],
                    "facing": es["facing"], "type": es["type"],
                }
                spr = self._enemy_sprites[i]
                if es["alive"]:
                    if spr is None:
                        tpl = ("weak_enemy" if es["etype"] == "weak"
                               else "normal_enemy" if es["etype"] == "normal"
                               else "strong_enemy")
                        new_spr = _sp[tpl].clone().set_position(
                            es["x"] * SCALE, es["y"] * SCALE,
                        )
                        self.current_level.add_sprite(new_spr)
                        self._enemy_sprites[i] = new_spr
                    else:
                        spr.set_position(es["x"] * SCALE, es["y"] * SCALE)
                        spr.set_visible(True)
                else:
                    if spr is not None:
                        spr.set_visible(False)
            else:
                self._enemies.append({
                    "x": es["x"], "y": es["y"],
                    "patrol": list(es["patrol"]),
                    "patrol_idx": es["patrol_idx"],
                    "etype": es["etype"], "alive": es["alive"],
                    "chasing": es["chasing"], "chase_turns": es["chase_turns"],
                    "facing": es["facing"], "type": es["type"],
                })
                if es["alive"]:
                    tpl = ("weak_enemy" if es["etype"] == "weak"
                           else "normal_enemy" if es["etype"] == "normal"
                           else "strong_enemy")
                    new_spr = _sp[tpl].clone().set_position(
                        es["x"] * SCALE, es["y"] * SCALE,
                    )
                    self.current_level.add_sprite(new_spr)
                    self._enemy_sprites.append(new_spr)
                else:
                    self._enemy_sprites.append(None)

        for pos, is_active in self._tile_states.items():
            if pos in self._tile_map:
                self._set_tile_visual(pos, is_active)

        saved_weapons = set(snapshot["weapon_positions"])
        current_weapons = set(self._weapon_map.keys())
        for pos in current_weapons - saved_weapons:
            sp = self._weapon_map.pop(pos)
            self.current_level.remove_sprite(sp)
        for pos in saved_weapons - current_weapons:
            new_sp = _sp["weapon"].clone().set_position(
                pos[0] * SCALE, pos[1] * SCALE,
            )
            self._weapon_map[pos] = new_sp
            self.current_level.add_sprite(new_sp)

        for sp in self._spawned_trap_sprites:
            self.current_level.remove_sprite(sp)
        self._spawned_trap_sprites = []
        saved_trap_positions = snapshot["spawned_trap_positions"]
        for tx, ty in saved_trap_positions:
            trap_sp = _sp["trap"].clone().set_position(tx * SCALE, ty * SCALE)
            self.current_level.add_sprite(trap_sp)
            self._spawned_trap_sprites.append(trap_sp)

        saved_lava = snapshot["lava_set"]
        current_lava = set(self._lava_set)
        if saved_lava != current_lava:
            for sp in self.current_level.get_sprites_by_tag("lava"):
                self.current_level.remove_sprite(sp)
            self._lava_set = set(saved_lava)
            for lp in self._lava_set:
                lsp = _sp["lava"].clone().set_position(lp[0] * SCALE, lp[1] * SCALE)
                self.current_level.add_sprite(lsp)

        saved_holes = snapshot["hole_set"]
        current_holes = set(self._hole_set)
        if saved_holes != current_holes:
            for sp in self.current_level.get_sprites_by_tag("hole"):
                self.current_level.remove_sprite(sp)
            self._hole_set = set(saved_holes)
            for hp in self._hole_set:
                hsp = _sp["hole"].clone().set_position(hp[0] * SCALE, hp[1] * SCALE)
                self.current_level.add_sprite(hsp)

        saved_traps = snapshot["trap_set"]
        current_traps = set(self._trap_set)
        if saved_traps != current_traps:
            for sp in self.current_level.get_sprites_by_tag("trap_visual"):
                self.current_level.remove_sprite(sp)
            self._trap_set = set(saved_traps)
            for tp in self._trap_set:
                tsp = _sp["trap"].clone().set_position(tp[0] * SCALE, tp[1] * SCALE)
                self.current_level.add_sprite(tsp)

    def _lose_life(self) -> None:
        self._lives -= 1
        if self._lives <= 0:
            self.lose()
        else:
            remaining = self._lives
            self._reinit_level()
            self._lives = remaining

    def step(self) -> None:
        if not self._alive:
            self.complete_action()
            return

        if self.action.id == GameAction.ACTION7:
            if self._history:
                self._restore_state(self._history.pop())
            self._move_count += 1
            if self._move_count >= MAX_MOVES:
                self._lose_life()
            self.complete_action()
            return

        self._history.append(self._save_state())

        if self.action.id == GameAction.ACTION1:
            self._try_move(0, -1)
        elif self.action.id == GameAction.ACTION2:
            self._try_move(0, 1)
        elif self.action.id == GameAction.ACTION3:
            self._try_move(-1, 0)
        elif self.action.id == GameAction.ACTION4:
            self._try_move(1, 0)
        elif self.action.id == GameAction.ACTION5:
            self._interact()

        self._move_count += 1

        if not self._alive:
            self._lose_life()
            self.complete_action()
            return

        if self._move_count >= MAX_MOVES:
            self._lose_life()
            self.complete_action()
            return

        self._move_enemies()

        self._check_enemy_contact()
        if not self._alive:
            self._lose_life()
            self.complete_action()
            return

        self._check_safe_timer()
        if not self._alive:
            self._lose_life()
            self.complete_action()
            return

        self._update_phase()
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
        self._engine = Cr01(seed=seed)
        self._total_turns = 0
        self._done = False
        self._game_won = False
        self._game_over = False
        self._last_action_was_reset = False

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
        return self._build_game_state()

    def get_actions(self) -> List[str]:
        if self._done:
            return ["reset"]
        return list(self._VALID_ACTIONS)

    def is_done(self) -> bool:
        return self._done

    def _require_engine(self) -> Cr01:
        if self._engine is None:
            raise RuntimeError("PuzzleEnvironment is closed")
        return self._engine

    def _build_text_observation(self) -> str:
        g = self._require_engine()
        enemy_pos: Dict[Tuple[int, int], str] = {}
        for en in g._enemies:
            if not en["alive"]:
                continue
            etype = en.get("etype", "normal")
            ch = "n"
            if etype == "weak":
                ch = "w"
            elif etype == "strong":
                ch = "s"
            enemy_pos[(en["x"], en["y"])] = ch
        rows = []
        for gy in range(g._log_h):
            chars = []
            for gx in range(g._log_w):
                pos = (gx, gy)
                if gx == g._player_x and gy == g._player_y:
                    chars.append("P")
                elif pos in enemy_pos:
                    chars.append(enemy_pos[pos])
                elif pos in g._weapon_map:
                    chars.append("W")
                elif pos in g._tile_map:
                    chars.append("T" if g._tile_states.get(pos, False) else "t")
                elif pos in g._lava_set:
                    chars.append("~")
                elif pos in g._hole_set:
                    chars.append("O")
                elif pos in g._trap_set:
                    chars.append("X")
                elif pos in g._wall_set:
                    chars.append("#")
                elif pos in g._stone_set:
                    chars.append("S")
                elif g._stack_pos and pos == g._stack_pos:
                    chars.append("_")
                else:
                    chars.append(".")
            rows.append("".join(chars))
        activated = sum(1 for v in g._tile_states.values() if v)
        carrying = "yes" if g._carrying_tile else "no"
        remaining = max(0, MAX_MOVES - g._move_count)
        status = (
            f"level={g._current_level_index + 1} "
            f"phase={g._phase} "
            f"lives={g._lives} "
            f"moves={remaining} "
            f"tiles={activated}/{g._total_tiles} "
            f"carrying={carrying}"
        )
        return "\n".join(rows + [status])

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
        rgb = self.render()
        return self._encode_png(rgb)

    def _build_game_state(self, done: bool = False) -> GameState:
        e = self._engine
        valid_actions = self.get_actions() if not done else None
        return GameState(
            text_observation=self._build_text_observation(),
            image_observation=self._build_image_bytes(),
            valid_actions=valid_actions,
            turn=self._total_turns,
            metadata={
                "total_levels": len(e._levels),
                "level_index": e.level_index,
                "levels_completed": getattr(e, "_score", 0),
                "game_over": self._game_over,
                "done": done,
                "info": {},
            },
        )

    def step(self, action: str) -> StepResult:
        e = self._engine
        if action == "reset":
            state = self.reset()
            return StepResult(
                state=state, reward=0.0, done=False, info={"action": "reset"}
            )
        if action not in self._ACTION_MAP:
            raise ValueError(
                f"Invalid action '{action}'. "
                f"Must be one of {list(self._ACTION_MAP.keys())}"
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
        total_levels = len(self._engine._levels)
        level_reward = 1.0 / total_levels
        if game_won:
            self._done = True
            self._game_won = True
            self._game_over = False
            info["reason"] = "game_complete"
            return StepResult(
                state=self._build_game_state(done=True),
                reward=level_reward,
                done=True,
                info=info,
            )
        if game_over:
            self._done = True
            self._game_won = False
            self._game_over = True
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