import struct
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

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
    metadata: Dict = field(default_factory=dict)


@dataclass
class StepResult:
    state: GameState
    reward: float
    done: bool
    info: Dict = field(default_factory=dict)


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

C_FURNITURE   = 2
C_GREY        = 3
C_DKGREY      = 15
C_WALL        = 5
C_POLICE      = 6
C_ITEM_GLOW   = 7
C_ITEM        = 8
C_PLAYER      = 9
C_DETECT      = 11
C_DOOR_OPEN   = 14
C_RWALL       = 12

BG_COLOR  = 0
PAD_COLOR = 5
SCALE = 3

DIR_N: Tuple[int, int] = (0, -1)
DIR_S: Tuple[int, int] = (0, 1)
DIR_E: Tuple[int, int] = (1, 0)
DIR_W: Tuple[int, int] = (-1, 0)
DIR_NONE: Tuple[int, int] = (0, 0)

ALL_DIRS = [DIR_N, DIR_S, DIR_E, DIR_W]

MAX_LIVES  = 3
MAX_MOVES  = 300

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
    "furniture": Sprite(
        pixels=[[C_FURNITURE] * SCALE for _ in range(SCALE)],
        name="furniture", visible=True, collidable=True,
        tags=["furniture"], layer=1,
    ),
    "item": Sprite(
        pixels=[
            [C_ITEM, C_ITEM, C_ITEM],
            [C_ITEM, C_ITEM_GLOW, C_ITEM],
            [C_ITEM, C_ITEM, C_ITEM],
        ],
        name="item", visible=True, collidable=False,
        tags=["item"], layer=2,
    ),
    "door": Sprite(
        pixels=[[C_GREY] * SCALE for _ in range(SCALE)],
        name="door", visible=True, collidable=False,
        tags=["door"], layer=0,
    ),
    "police": Sprite(
        pixels=[[C_POLICE] * SCALE for _ in range(SCALE)],
        name="police", visible=True, collidable=False,
        tags=["police"], layer=4,
    ),
    "trap": Sprite(
        pixels=[[C_DKGREY] * SCALE for _ in range(SCALE)],
        name="trap", visible=True, collidable=False,
        tags=["trap"], layer=0,
    ),
    "detect_zone": Sprite(
        pixels=[[C_DETECT] * SCALE for _ in range(SCALE)],
        name="detect_zone", visible=True, collidable=False,
        tags=["detect_zone"], layer=3,
    ),
    "rwall": Sprite(
        pixels=[[C_RWALL] * SCALE for _ in range(SCALE)],
        name="rwall", visible=True, collidable=True,
        tags=["rwall"], layer=1,
    ),
}


def _h_patrol(y: int, x1: int, x2: int) -> List[Tuple[int, int]]:
    pts: List[Tuple[int, int]] = []
    for x in range(x1, x2 + 1):
        pts.append((x, y))
    for x in range(x2 - 1, x1, -1):
        pts.append((x, y))
    return pts


def _v_patrol(x: int, y1: int, y2: int) -> List[Tuple[int, int]]:
    pts: List[Tuple[int, int]] = []
    for y in range(y1, y2 + 1):
        pts.append((x, y))
    for y in range(y2 - 1, y1, -1):
        pts.append((x, y))
    return pts


def _grid(rows: List[str]) -> str:
    return "".join(rows)


W1, H1 = 20, 19
G1 = _grid([
    "####################",
    "#..................#",
    "#..FF..........FF..#",
    "#..FF..........FF..#",
    "#..................#",
    "#..................#",
    "#......FF..FF......#",
    "#......FF..FF......#",
    "#..................#",
    "#..R...........R...#",
    "#..................#",
    "#......FF..FF......#",
    "#......FF..FF......#",
    "#..................#",
    "#..................#",
    "#..FF..........FF..#",
    "#..FF....#.....FF..#",
    "#..................#",
    "#########P##########",
])

POLICE_L1 = [
    {
        "start": (10, 4),
        "patrol": _h_patrol(4, 3, 16),
    },
]

RWALLS_L1: List[List[Tuple[int, int]]] = []

TRAPS_L1: List[Tuple[int, int]] = []

W4, H4 = 20, 19
G4 = _grid([
    "####################",
    "#.R................#",
    "#..FF..........FF..#",
    "#..FF..........FF..#",
    "#..................#",
    "#......####........#",
    "#......#...........#",
    "#......#.......FF..#",
    "#..............FF..#",
    "#..................#",
    "#..FF..............#",
    "#..FF......#.......#",
    "#..........####....#",
    "#..................#",
    "#......FF..FF......#",
    "#..R...FF..FF...R..#",
    "#..................#",
    "#..................#",
    "####P###############",
])

POLICE_L4 = [
    {
        "start": (8, 1),
        "patrol": _h_patrol(1, 5, 12),
    },
    {
        "start": (15, 5),
        "patrol": _h_patrol(5, 11, 17),
    },
    {
        "start": (10, 9),
        "patrol": _h_patrol(9, 3, 16),
    },
]

RWALLS_L4: List[List[Tuple[int, int]]] = [
    [(12, 9)],
    [(3, 9)],
    [(15, 13)],
]

TRAPS_L4: List[Tuple[int, int]] = [(1, 9)]

W5, H5 = 20, 19
G5 = _grid([
    "####################",
    "#.R................#",
    "#..FF..........FF..#",
    "#..FF..........FF..#",
    "#..................#",
    "#..#.#.........#.#.#",
    "#..#...........#...#",
    "#..................#",
    "#..#..FF...FF..#...#",
    "#..#..FF...FF......#",
    "#..#...........#...#",
    "#..............#...#",
    "#..#.#.........#.#.#",
    "#....#.........#...#",
    "#.FF.#..FF.FF..#...#",
    "#....#..FF.FF..#.R.#",
    "#.R..#.........#...#",
    "#..................#",
    "######P#############",
])

POLICE_L5 = [
    {
        "start": (10, 4),
        "patrol": _h_patrol(4, 3, 16),
    },
    {
        "start": (5, 8),
        "patrol": _v_patrol(5, 7, 11),
    },
    {
        "start": (10, 11),
        "patrol": _h_patrol(11, 3, 14),
    },
    {
        "start": (17, 7),
        "patrol": _h_patrol(7, 10, 17),
    },
]

RWALLS_L5: List[List[Tuple[int, int]]] = [
    [(9, 6)],
    [(9, 11)],
    [(12, 17)],
]

TRAPS_L5: List[Tuple[int, int]] = [(10, 17), (14, 4)]

W6, H6 = 20, 19
G6 = _grid([
    "####################",
    "#..R.......#..R....#",
    "#..FF......#..FF...#",
    "#..FF..#...#.......#",
    "#......#...........#",
    "#......#..FF.......#",
    "#.#.##.#..FF..####.#",
    "#......#...........#",
    "#......#.......#.R.#",
    "#..............#...#",
    "#......####........#",
    "#.##.#.......##....#",
    "#......FF..........#",
    "#......FF..R.......#",
    "#.FF...........FF..#",
    "#.FF###........FF..#",
    "#..........R.......#",
    "#.....#............#",
    "####P###############",
])

POLICE_L6 = [
    {
        "start": (7, 1),
        "patrol": _h_patrol(1, 2, 10),
    },
    {
        "start": (15, 4),
        "patrol": _v_patrol(15, 1, 9),
    },
    {
        "start": (5, 9),
        "patrol": _v_patrol(5, 7, 10),
    },
    {
        "start": (10, 17),
        "patrol": _h_patrol(17, 7, 17),
    },
]

RWALLS_L6: List[List[Tuple[int, int]]] = [
    [(10, 4)],
    [(3, 10)],
    [(14, 10)],
    [(8, 17)],
    [(16, 12)],
]

TRAPS_L6: List[Tuple[int, int]] = [(1, 10), (18, 10), (10, 17), (5, 4)]


LEVEL_DATA = [
    {"grid": G1, "w": W1, "h": H1, "police": POLICE_L1,
     "rwalls": RWALLS_L1, "traps": TRAPS_L1},
    {"grid": G4, "w": W4, "h": H4, "police": POLICE_L4,
     "rwalls": RWALLS_L4, "traps": TRAPS_L4},
    {"grid": G5, "w": W5, "h": H5, "police": POLICE_L5,
     "rwalls": RWALLS_L5, "traps": TRAPS_L5},
    {"grid": G6, "w": W6, "h": H6, "police": POLICE_L6,
     "rwalls": RWALLS_L6, "traps": TRAPS_L6},
]


def _parse_grid(
    grid_str: str, w: int, h: int, level_idx: int,
    police_data: List[Dict], trap_positions: List[Tuple[int, int]],
) -> Level:
    spr_list: List[Sprite] = []
    px, py = 0, 0
    walls: List[Dict] = []
    furniture: List[Dict] = []
    items: List[Dict] = []
    door_pos: Dict[str, int] = {"x": 0, "y": 0}

    for y in range(h):
        for x in range(w):
            ch = grid_str[y * w + x]
            sx, sy = x * SCALE, y * SCALE
            if ch == "#":
                spr_list.append(_sp["wall"].clone().set_position(sx, sy))
                walls.append({"x": x, "y": y})
            elif ch == "P":
                px, py = x, y
                door_pos = {"x": x, "y": y}
                spr_list.append(_sp["door"].clone().set_position(sx, sy))
                spr_list.append(_sp["player"].clone().set_position(sx, sy))
            elif ch == "F":
                furniture.append({"x": x, "y": y})
                spr_list.append(
                    _sp["furniture"].clone().set_position(sx, sy)
                )
            elif ch == "R":
                items.append({"x": x, "y": y})
                spr_list.append(_sp["item"].clone().set_position(sx, sy))

    traps: List[Dict] = []
    for tx, ty in trap_positions:
        traps.append({"x": tx, "y": ty})
        spr_list.append(
            _sp["trap"].clone().set_position(tx * SCALE, ty * SCALE)
        )

    for pd in police_data:
        ex, ey = pd["start"]
        spr_list.append(
            _sp["police"].clone().set_position(ex * SCALE, ey * SCALE)
        )

    return Level(
        sprites=spr_list,
        grid_size=(w * SCALE, h * SCALE),
        data={
            "level_idx": level_idx,
            "logical_width": w, "logical_height": h,
            "player_x": px, "player_y": py,
            "walls": walls, "furniture": furniture,
            "items": items, "traps": traps,
            "door_pos": door_pos,
            "police_data": police_data,
        },
        name=f"Level {level_idx + 1}",
    )


def _build_levels() -> List[Level]:
    return [
        _parse_grid(
            ld["grid"], ld["w"], ld["h"], i,
            ld["police"], ld["traps"],
        )
        for i, ld in enumerate(LEVEL_DATA)
    ]


class GameHud(RenderableUserDisplay):
    HUD_Y = 60

    def __init__(self, game: "St01") -> None:
        self._g = game

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        g = self._g
        hy = self.HUD_Y
        frame[hy:, :] = C_WALL

        frame[hy, :] = C_GREY

        for i in range(MAX_LIVES):
            lx = 1 + i * 3
            color = C_LIFE_ON if i < g._lives else C_LIFE_OFF
            frame[hy + 1:hy + 3, lx:lx + 2] = color

        if g._can_escape:
            frame[hy + 1:hy + 3, 52:55] = C_DOOR_OPEN
        else:
            frame[hy + 1:hy + 3, 52:55] = C_GREY

        bar_start = 1
        bar_max_width = 58
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


class St01(ARCBaseGame):

    def __init__(self, seed: int = 0, **kwargs) -> None:
        levels = _build_levels()
        self._hud = GameHud(self)

        self._player: Optional[Sprite] = None
        self._player_x: int = 0
        self._player_y: int = 0
        self._alive: bool = True

        self._total_items: int = 0
        self._items_collected: int = 0
        self._item_positions: List[Tuple[int, int]] = []
        self._item_sprites: Dict[Tuple[int, int], Sprite] = {}
        self._item_init_positions: List[Tuple[int, int]] = []
        self._can_escape: bool = False

        self._door_pos: Tuple[int, int] = (0, 0)
        self._door_sprite: Optional[Sprite] = None

        self._wall_set: set = set()
        self._furniture_set: set = set()
        self._trap_set: set = set()
        self._rwall_set: set = set()
        self._log_w: int = 0
        self._log_h: int = 0

        self._police: List[Dict] = []
        self._police_sprites: List[Optional[Sprite]] = []
        self._detect_sprites: List[Sprite] = []

        self._rwall_schedule: List[List[Tuple[int, int]]] = []
        self._rwall_wave: int = 0
        self._rwall_sprites: List[Sprite] = []

        self._lives: int = MAX_LIVES
        self._move_count: int = 0
        self._init_player_x: int = 0
        self._init_player_y: int = 0

        self._history: List[dict] = []

        camera = Camera(
            0, 0, 64, 64,
            BG_COLOR, PAD_COLOR,
            [self._hud],
        )

        super().__init__(
            game_id="st01",
            levels=levels,
            camera=camera,
            available_actions=[0, 1, 2, 3, 4, 7],
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        w = self.current_level.get_data("logical_width")
        h = self.current_level.get_data("logical_height")
        self._log_w = w
        self._log_h = h

        self._alive = True
        self._items_collected = 0
        self._can_escape = False

        self._player = self.current_level.get_sprites_by_tag("player")[0]
        self._player_x = self.current_level.get_data("player_x")
        self._player_y = self.current_level.get_data("player_y")
        self._init_player_x = self._player_x
        self._init_player_y = self._player_y

        self._lives = MAX_LIVES
        self._move_count = 0
        self._history = []

        self._wall_set = {
            (p["x"], p["y"])
            for p in self.current_level.get_data("walls")
        }
        self._furniture_set = {
            (p["x"], p["y"])
            for p in self.current_level.get_data("furniture")
        }

        self._trap_set = {
            (p["x"], p["y"])
            for p in self.current_level.get_data("traps")
        }

        dp = self.current_level.get_data("door_pos")
        self._door_pos = (dp["x"], dp["y"])
        door_sprites = self.current_level.get_sprites_by_tag("door")
        self._door_sprite = door_sprites[0] if door_sprites else None

        self._item_sprites = {}
        self._item_positions = []
        self._item_init_positions = []
        item_data = self.current_level.get_data("items")
        item_sprites = self.current_level.get_sprites_by_tag("item")
        self._total_items = len(item_data)
        for idx, itd in enumerate(item_data):
            pos = (itd["x"], itd["y"])
            self._item_positions.append(pos)
            self._item_init_positions.append(pos)
            if idx < len(item_sprites):
                self._item_sprites[pos] = item_sprites[idx]

        police_data = self.current_level.get_data("police_data")
        police_sprites = self.current_level.get_sprites_by_tag("police")
        self._police = []
        self._police_sprites = []
        for idx, pd in enumerate(police_data):
            sx, sy = pd["start"]
            patrol = pd.get("patrol", [])
            start_idx = 0
            if patrol:
                for pi, wp in enumerate(patrol):
                    if wp == (sx, sy):
                        start_idx = pi
                        break
            facing = DIR_S
            if patrol and len(patrol) > 1:
                next_i = (start_idx + 1) % len(patrol)
                ddx = patrol[next_i][0] - sx
                ddy = patrol[next_i][1] - sy
                if ddx > 0:
                    facing = DIR_E
                elif ddx < 0:
                    facing = DIR_W
                elif ddy > 0:
                    facing = DIR_S
                elif ddy < 0:
                    facing = DIR_N
            self._police.append({
                "x": sx, "y": sy,
                "init_x": sx, "init_y": sy,
                "patrol": patrol, "patrol_idx": start_idx,
                "facing": facing,
                "chasing": False,
                "steps_since_look": 0,
                "look_phase": 0,
                "look_saved_facing": facing,
            })
            matched = None
            if idx < len(police_sprites):
                matched = police_sprites[idx]
            self._police_sprites.append(matched)

        level_idx = self.current_level.get_data("level_idx")
        self._rwall_schedule = LEVEL_DATA[level_idx]["rwalls"]
        self._rwall_wave = 0
        self._rwall_set = set()
        self._rwall_sprites = []

        self._detect_sprites = []

        self._update_detection_zones()

        self._build_item_protected_tiles()

    def _snapshot(self) -> dict:
        return {
            "player_x": self._player_x,
            "player_y": self._player_y,
            "alive": self._alive,
            "items_collected": self._items_collected,
            "item_positions": list(self._item_positions),
            "can_escape": self._can_escape,
            "lives": self._lives,
            "move_count": self._move_count,
            "rwall_set": set(self._rwall_set),
            "rwall_wave": self._rwall_wave,
            "police": [
                {
                    "x": p["x"], "y": p["y"],
                    "patrol_idx": p["patrol_idx"],
                    "facing": p["facing"],
                    "chasing": p["chasing"],
                    "steps_since_look": p["steps_since_look"],
                    "look_phase": p["look_phase"],
                    "look_saved_facing": p["look_saved_facing"],
                }
                for p in self._police
            ],
        }

    def _restore_snapshot(self, snapshot: dict) -> None:
        self._player_x = snapshot["player_x"]
        self._player_y = snapshot["player_y"]
        self._alive = snapshot["alive"]
        self._items_collected = snapshot["items_collected"]
        self._can_escape = snapshot["can_escape"]
        self._lives = snapshot["lives"]
        self._move_count = snapshot["move_count"]
        self._rwall_wave = snapshot["rwall_wave"]

        if self._player is not None:
            self._player.set_position(
                self._player_x * SCALE, self._player_y * SCALE
            )

        old_item_set = set(self._item_sprites.keys())
        new_item_set = set(snapshot["item_positions"])

        for pos in old_item_set - new_item_set:
            sp = self._item_sprites.pop(pos)
            self.current_level.remove_sprite(sp)

        for pos in new_item_set - old_item_set:
            sp = _sp["item"].clone().set_position(
                pos[0] * SCALE, pos[1] * SCALE
            )
            self.current_level.add_sprite(sp)
            self._item_sprites[pos] = sp

        self._item_positions = list(snapshot["item_positions"])
        self._build_item_protected_tiles()

        old_rwall = self._rwall_set
        new_rwall = snapshot["rwall_set"]
        for pos in old_rwall - new_rwall:
            for sp in list(self._rwall_sprites):
                sx = sp.x // SCALE
                sy = sp.y // SCALE
                if (sx, sy) == pos:
                    self.current_level.remove_sprite(sp)
                    self._rwall_sprites.remove(sp)
                    break
        for pos in new_rwall - old_rwall:
            sp = _sp["rwall"].clone().set_position(
                pos[0] * SCALE, pos[1] * SCALE
            )
            self.current_level.add_sprite(sp)
            self._rwall_sprites.append(sp)
        self._rwall_set = set(new_rwall)

        for i, pd in enumerate(snapshot["police"]):
            p = self._police[i]
            p["x"] = pd["x"]
            p["y"] = pd["y"]
            p["patrol_idx"] = pd["patrol_idx"]
            p["facing"] = pd["facing"]
            p["chasing"] = pd["chasing"]
            p["steps_since_look"] = pd["steps_since_look"]
            p["look_phase"] = pd["look_phase"]
            p["look_saved_facing"] = pd["look_saved_facing"]
            sp = self._police_sprites[i]
            if sp is not None:
                sp.set_position(p["x"] * SCALE, p["y"] * SCALE)

        if self._can_escape and self._door_sprite is not None:
            self.current_level.remove_sprite(self._door_sprite)
            new_door = Sprite(
                pixels=[[C_DOOR_OPEN] * SCALE for _ in range(SCALE)],
                name="door_open", visible=True, collidable=False,
                tags=["door"], layer=0,
            )
            new_door.set_position(
                self._door_pos[0] * SCALE, self._door_pos[1] * SCALE
            )
            self.current_level.add_sprite(new_door)
            self._door_sprite = new_door
        elif not self._can_escape and self._door_sprite is not None:
            self.current_level.remove_sprite(self._door_sprite)
            new_door = _sp["door"].clone().set_position(
                self._door_pos[0] * SCALE, self._door_pos[1] * SCALE
            )
            self.current_level.add_sprite(new_door)
            self._door_sprite = new_door

        self._update_detection_zones()

    def _blocks_los(self, x: int, y: int) -> bool:
        pos = (x, y)
        return (pos in self._wall_set or pos in self._furniture_set
                or pos in self._rwall_set)

    def _is_walkable(self, x: int, y: int) -> bool:
        if x < 0 or y < 0 or x >= self._log_w or y >= self._log_h:
            return False
        pos = (x, y)
        if pos in self._wall_set or pos in self._furniture_set:
            return False
        if pos in self._rwall_set:
            return False
        return True

    def _build_item_protected_tiles(self) -> None:
        self._item_protected_tiles: set = set()
        for ix, iy in self._item_positions:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    self._item_protected_tiles.add((ix + dx, iy + dy))

    def _is_police_walkable(self, x: int, y: int, chasing: bool) -> bool:
        if not self._is_walkable(x, y):
            return False
        if not chasing and (x, y) in self._item_protected_tiles:
            return False
        return True

    def _get_detection_tiles(self, p: Dict) -> List[Tuple[int, int]]:
        ex, ey = p["x"], p["y"]
        facing = p["facing"]
        tiles: List[Tuple[int, int]] = []

        if facing == DIR_NONE:
            return tiles

        fx, fy = facing

        for dist in range(1, max(self._log_w, self._log_h)):
            tx, ty = ex + fx * dist, ey + fy * dist
            if tx < 0 or ty < 0 or tx >= self._log_w or ty >= self._log_h:
                break
            if self._blocks_los(tx, ty):
                break
            tiles.append((tx, ty))

        if facing == DIR_N or facing == DIR_S:
            left_dir = (-1, 0) if facing == DIR_N else (1, 0)
            right_dir = (1, 0) if facing == DIR_N else (-1, 0)
        else:
            left_dir = (0, -1) if facing == DIR_E else (0, 1)
            right_dir = (0, 1) if facing == DIR_E else (0, -1)

        for ldx, ldy in [left_dir, right_dir]:
            for dist in range(1, 3):
                sx, sy = ex + ldx * dist, ey + ldy * dist
                if 0 <= sx < self._log_w and 0 <= sy < self._log_h:
                    if self._blocks_los(sx, sy):
                        break
                    tiles.append((sx, sy))
                else:
                    break

        if facing == DIR_N:
            diags = [(-1, -1), (1, -1)]
        elif facing == DIR_S:
            diags = [(1, 1), (-1, 1)]
        elif facing == DIR_E:
            diags = [(1, -1), (1, 1)]
        else:
            diags = [(-1, 1), (-1, -1)]

        for ddx, ddy in diags:
            dx, dy = ex + ddx, ey + ddy
            if not (0 <= dx < self._log_w and 0 <= dy < self._log_h):
                continue
            if self._blocks_los(dx, dy):
                continue
            if self._blocks_los(ex + ddx, ey):
                continue
            if self._blocks_los(ex, ey + ddy):
                continue
            tiles.append((dx, dy))

        return tiles

    def _update_detection_zones(self) -> None:
        for sp in self._detect_sprites:
            self.current_level.remove_sprite(sp)
        self._detect_sprites = []

        all_detect: set = set()
        for p in self._police:
            det = self._get_detection_tiles(p)
            all_detect.update(det)

        blocked = self._wall_set | self._furniture_set | self._rwall_set
        police_pos = {(p["x"], p["y"]) for p in self._police}

        for tx, ty in all_detect:
            pos = (tx, ty)
            if pos in blocked or pos in police_pos:
                continue
            sp = _sp["detect_zone"].clone().set_position(
                tx * SCALE, ty * SCALE
            )
            self.current_level.add_sprite(sp)
            self._detect_sprites.append(sp)

    def _try_move(self, dx: int, dy: int) -> bool:
        nx = self._player_x + dx
        ny = self._player_y + dy

        if not self._is_walkable(nx, ny):
            return False

        self._player_x = nx
        self._player_y = ny
        if self._player is not None:
            self._player.set_position(nx * SCALE, ny * SCALE)

        self._check_item_pickup()

        self._check_traps()

        self._check_escape()

        return True

    def _check_item_pickup(self) -> None:
        pos = (self._player_x, self._player_y)
        if pos in self._item_sprites:
            sp = self._item_sprites.pop(pos)
            self.current_level.remove_sprite(sp)
            self._items_collected += 1
            if pos in self._item_positions:
                self._item_positions.remove(pos)
            self._build_item_protected_tiles()
            self._can_escape = (self._items_collected >= self._total_items)
            if self._can_escape and self._door_sprite is not None:
                self.current_level.remove_sprite(self._door_sprite)
                new_door = Sprite(
                    pixels=[[C_DOOR_OPEN] * SCALE for _ in range(SCALE)],
                    name="door_open", visible=True, collidable=False,
                    tags=["door"], layer=0,
                )
                new_door.set_position(
                    self._door_pos[0] * SCALE, self._door_pos[1] * SCALE
                )
                self.current_level.add_sprite(new_door)
                self._door_sprite = new_door

    def _check_traps(self) -> None:
        pos = (self._player_x, self._player_y)
        if pos in self._trap_set:
            self._alive = False

    def _check_escape(self) -> None:
        pos = (self._player_x, self._player_y)
        if pos == self._door_pos and self._can_escape:
            self.next_level()

    def _get_left_facing(self, facing: Tuple[int, int]) -> Tuple[int, int]:
        if facing == DIR_N:
            return DIR_W
        if facing == DIR_W:
            return DIR_S
        if facing == DIR_S:
            return DIR_E
        return DIR_N

    def _get_right_facing(self, facing: Tuple[int, int]) -> Tuple[int, int]:
        if facing == DIR_N:
            return DIR_E
        if facing == DIR_E:
            return DIR_S
        if facing == DIR_S:
            return DIR_W
        return DIR_N

    def _patrol_step(self, p: Dict) -> None:
        if p["look_phase"] == 1:
            p["facing"] = self._get_right_facing(p["look_saved_facing"])
            p["look_phase"] = 2
            return

        if p["look_phase"] == 2:
            p["facing"] = p["look_saved_facing"]
            p["look_phase"] = 0
            return

        patrol = p["patrol"]
        if not patrol:
            return

        plen = len(patrol)
        next_idx = (p["patrol_idx"] + 1) % plen
        target_x, target_y = patrol[next_idx]

        if not self._is_police_walkable(target_x, target_y, False):
            found = False
            for skip in range(2, plen):
                try_idx = (p["patrol_idx"] + skip) % plen
                tx, ty = patrol[try_idx]
                if self._is_police_walkable(tx, ty, False):
                    next_idx = try_idx
                    target_x, target_y = tx, ty
                    found = True
                    break
            if not found:
                return

        step_dx = target_x - p["x"]
        step_dy = target_y - p["y"]

        p["x"] = target_x
        p["y"] = target_y
        p["patrol_idx"] = next_idx

        if step_dx > 0:
            p["facing"] = DIR_E
        elif step_dx < 0:
            p["facing"] = DIR_W
        elif step_dy > 0:
            p["facing"] = DIR_S
        elif step_dy < 0:
            p["facing"] = DIR_N

        p["steps_since_look"] += 1

        if p["steps_since_look"] >= 2:
            p["steps_since_look"] = 0
            p["look_saved_facing"] = p["facing"]
            p["facing"] = self._get_left_facing(p["facing"])
            p["look_phase"] = 1

    def _chase_step(self, p: Dict) -> None:
        px, py = self._player_x, self._player_y
        ex, ey = p["x"], p["y"]
        best_dir = None
        best_d = abs(px - ex) + abs(py - ey)
        for ddx, ddy in ALL_DIRS:
            nx, ny = ex + ddx, ey + ddy
            if not self._is_police_walkable(nx, ny, True):
                continue
            d = abs(px - nx) + abs(py - ny)
            if d < best_d:
                best_d = d
                best_dir = (ddx, ddy)
        if best_dir:
            p["x"] += best_dir[0]
            p["y"] += best_dir[1]
            if best_dir[0] > 0:
                p["facing"] = DIR_E
            elif best_dir[0] < 0:
                p["facing"] = DIR_W
            elif best_dir[1] > 0:
                p["facing"] = DIR_S
            elif best_dir[1] < 0:
                p["facing"] = DIR_N

    def _player_in_any_detection_zone(self) -> bool:
        player_pos = (self._player_x, self._player_y)
        for p in self._police:
            det_tiles = self._get_detection_tiles(p)
            if player_pos in det_tiles:
                return True
        return False

    def _alert_all_police(self) -> None:
        for p in self._police:
            p["chasing"] = True
            p["look_phase"] = 0

    def _move_single_police(self, idx: int, p: Dict) -> None:
        if p["chasing"]:
            self._chase_step(p)
        else:
            self._patrol_step(p)

        sp = self._police_sprites[idx]
        if sp is not None:
            sp.set_position(p["x"] * SCALE, p["y"] * SCALE)

    def _move_police(self) -> None:
        if self._player_in_any_detection_zone():
            self._alert_all_police()

        for i, p in enumerate(self._police):
            if p["chasing"]:
                self._move_single_police(i, p)
                self._move_single_police(i, p)
            else:
                self._move_single_police(i, p)

        if self._player_in_any_detection_zone():
            self._alert_all_police()

    def _check_police_contact(self) -> None:
        for p in self._police:
            if p["x"] == self._player_x and p["y"] == self._player_y:
                self._alive = False
                return

    def _maybe_spawn_rwall(self) -> None:
        if self._rwall_wave >= len(self._rwall_schedule):
            return

        threshold = (self._rwall_wave + 1) * 40
        if self._move_count >= threshold:
            positions = self._rwall_schedule[self._rwall_wave]
            for rx, ry in positions:
                pos = (rx, ry)
                if pos == (self._player_x, self._player_y):
                    continue
                if pos in self._wall_set or pos in self._furniture_set:
                    continue
                if pos in self._rwall_set:
                    continue
                if pos in self._item_sprites:
                    continue
                if any(pos == (p["x"], p["y"]) for p in self._police):
                    continue
                self._rwall_set.add(pos)
                sp = _sp["rwall"].clone().set_position(
                    rx * SCALE, ry * SCALE
                )
                self.current_level.add_sprite(sp)
                self._rwall_sprites.append(sp)
            self._rwall_wave += 1

    def _respawn_player(self) -> None:
        self._alive = True
        self._player_x = self._init_player_x
        self._player_y = self._init_player_y
        if self._player is not None:
            self._player.set_position(
                self._init_player_x * SCALE,
                self._init_player_y * SCALE,
            )

        for pos, sp in list(self._item_sprites.items()):
            self.current_level.remove_sprite(sp)
        self._item_sprites = {}
        self._item_positions = []
        self._items_collected = 0
        self._can_escape = False

        for pos in self._item_init_positions:
            sp = _sp["item"].clone().set_position(
                pos[0] * SCALE, pos[1] * SCALE
            )
            self.current_level.add_sprite(sp)
            self._item_sprites[pos] = sp
            self._item_positions.append(pos)

        if self._door_sprite is not None:
            self.current_level.remove_sprite(self._door_sprite)
        new_door = _sp["door"].clone().set_position(
            self._door_pos[0] * SCALE, self._door_pos[1] * SCALE
        )
        self.current_level.add_sprite(new_door)
        self._door_sprite = new_door

        for i, p in enumerate(self._police):
            p["x"] = p["init_x"]
            p["y"] = p["init_y"]
            p["chasing"] = False
            patrol = p["patrol"]
            start_idx = 0
            if patrol:
                for pi, wp in enumerate(patrol):
                    if wp == (p["init_x"], p["init_y"]):
                        start_idx = pi
                        break
            p["patrol_idx"] = start_idx
            facing = DIR_S
            if patrol and len(patrol) > 1:
                next_i = (start_idx + 1) % len(patrol)
                ddx = patrol[next_i][0] - p["init_x"]
                ddy = patrol[next_i][1] - p["init_y"]
                if ddx > 0:
                    facing = DIR_E
                elif ddx < 0:
                    facing = DIR_W
                elif ddy > 0:
                    facing = DIR_S
                elif ddy < 0:
                    facing = DIR_N
            p["facing"] = facing
            p["steps_since_look"] = 0
            p["look_phase"] = 0
            p["look_saved_facing"] = facing
            sp = self._police_sprites[i]
            if sp is not None:
                sp.set_position(p["x"] * SCALE, p["y"] * SCALE)

        self._build_item_protected_tiles()

    def _lose_life(self) -> None:
        self._lives -= 1
        if self._lives <= 0:
            self.lose()
        else:
            self._alive = True
            self._move_count = 0
            self._respawn_player()

    def step(self) -> None:
        if not self._alive:
            self.complete_action()
            return

        if self.action.id == GameAction.ACTION7:
            if self._history:
                current_move_count = self._move_count
                self._restore_snapshot(self._history.pop())
                self._move_count = current_move_count + 1
                if self._move_count >= MAX_MOVES:
                    self._lose_life()
                    self._update_detection_zones()
            self.complete_action()
            return

        self._history.append(self._snapshot())

        if self.action.id == GameAction.ACTION1:
            self._try_move(0, -1)
        elif self.action.id == GameAction.ACTION2:
            self._try_move(0, 1)
        elif self.action.id == GameAction.ACTION3:
            self._try_move(-1, 0)
        elif self.action.id == GameAction.ACTION4:
            self._try_move(1, 0)

        self._move_count += 1

        if not self._alive:
            self._lose_life()
            self._update_detection_zones()
            self.complete_action()
            return

        if self._move_count >= MAX_MOVES:
            self._lose_life()
            self._update_detection_zones()
            self.complete_action()
            return

        self._move_police()

        self._check_police_contact()
        if not self._alive:
            self._lose_life()
            self._update_detection_zones()
            self.complete_action()
            return

        self._maybe_spawn_rwall()

        self._update_detection_zones()

        self.complete_action()


class PuzzleEnvironment:
    _ACTION_MAP: Dict[str, GameAction] = {
        "reset": GameAction.RESET,
        "up": GameAction.ACTION1,
        "down": GameAction.ACTION2,
        "left": GameAction.ACTION3,
        "right": GameAction.ACTION4,
        "undo": GameAction.ACTION7,
    }

    _VALID_ACTIONS = list(_ACTION_MAP.keys())

    def __init__(self, seed: int = 0) -> None:
        self._engine = St01(seed=seed)
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

    def _require_engine(self) -> St01:
        if self._engine is None:
            raise RuntimeError("PuzzleEnvironment is closed")
        return self._engine

    def _build_text_observation(self) -> str:
        g = self._require_engine()
        police_pos: Dict[Tuple[int, int], bool] = {}
        for p in g._police:
            police_pos[(p["x"], p["y"])] = True

        detect_tiles = set()
        for p in g._police:
            detect_tiles.update(g._get_detection_tiles(p))

        rows = []
        for gy in range(g._log_h):
            chars = []
            for gx in range(g._log_w):
                pos = (gx, gy)
                if gx == g._player_x and gy == g._player_y:
                    chars.append("P")
                elif pos in police_pos:
                    chars.append("G")
                elif pos in g._item_sprites:
                    chars.append("R")
                elif pos == g._door_pos:
                    chars.append("O" if g._can_escape else "D")
                elif pos in g._trap_set:
                    chars.append("X")
                elif pos in g._rwall_set:
                    chars.append("W")
                elif pos in g._wall_set:
                    chars.append("#")
                elif pos in g._furniture_set:
                    chars.append("F")
                elif pos in detect_tiles:
                    chars.append("Y")
                else:
                    chars.append(".")
            rows.append("".join(chars))
        remaining = max(0, MAX_MOVES - g._move_count)
        status = (
            f"level={g._current_level_index + 1} "
            f"lives={g._lives} "
            f"items={g._items_collected}/{g._total_items} "
            f"moves={remaining} "
            f"escape={'yes' if g._can_escape else 'no'}"
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
