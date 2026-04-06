from __future__ import annotations

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

C_BLACK = 5
C_DARKGREY = 4
C_GREY = 3
C_WHITE = 0
C_RED = 8
C_BLUE = 9
C_LIGHTBLUE = 10
C_GREEN = 14
C_YELLOW = 11
C_ORANGE = 12
C_PINK = 7
C_PURPLE = 15
C_MAROON = 13
C_MAGENTA = 6

C_BACKGROUND = C_BLACK

CAM = 32
TILE = 2
MAX_LIVES = 3
MOVE_BAR_WIDTH = CAM - 2
N_LEVELS = 4

LEVELS = [
    {
        "rows": [
            "################",
            "#T..R....R...U.#",
            "#....W.R.R.....#",
            "#.R........R.W.#",
            "#..R......R....#",
            "#.R......W..R..#",
            "#....R.....R...#",
            "#..W.....R.....#",
            "#........R.....#",
            "#.R.....R......#",
            "#..............#",
            "#......E.......#",
            "################",
        ],
        "trucks": [
            {"px": 1, "py": 1, "cap": 2},
            {"px": 13, "py": 1, "cap": 2},
        ],
        "max_moves": 130,
    },
    {
        "rows": [
            "################",
            "#T..S..#..S..U.#",
            "#..W.R.#..W..R.#",
            "#.####.#.####..#",
            "#..R.W.#.W.R..S#",
            "#.#..S.#.S..#..#",
            "#.W..R.#..R.W..#",
            "#.#..#.#.#..#..#",
            "###..####..#####",
            "#S...........S.#",
            "#..W.#.R.#.W..S#",
            "#..S.#...#.....#",
            "#......E.......#",
            "################",
        ],
        "trucks": [
            {"px": 1, "py": 1, "cap": 4},
            {"px": 13, "py": 1, "cap": 4},
        ],
        "max_moves": 180,
        "spawn_every": 6,
    },
    {
        "rows": [
            "################",
            "#.....M..M.....#",
            "#T....#..U...V.#",
            "#.H.M.#....R...#",
            "#.....#..R..M..#",
            "#..R..##..##.R.#",
            "#.W..R..M..W...#",
            "#..M..R....W.M.#",
            "####..####..####",
            "#.R.....M......#",
            "#.H..R....H....#",
            "#....M.....W...#",
            "#...W..R...W...#",
            "#......E.......#",
            "################",
        ],
        "trucks": [
            {"px": 1, "py": 2, "cap": 4},
            {"px": 9, "py": 2, "cap": 4},
            {"px": 13, "py": 2, "cap": 4},
        ],
        "max_moves": 220,
    },
    {
        "rows": [
            "################",
            "#..R.S..S..S.R.#",
            "#T....#..U...V.#",
            "#.G.R.#.R....G.#",
            "#..R..#.R..G...#",
            "#.R...##..##.R.#",
            "#.B..R.......B.#",
            "#..R...##....R.#",
            "###..##..##..###",
            "#..R..S..S..R..#",
            "#.G..R.....R.G.#",
            "#..R..G......R.#",
            "#.R.B......B.R.#",
            "#......E.......#",
            "################",
        ],
        "trucks": [
            {"px": 1, "py": 2, "cap": 6},
            {"px": 9, "py": 2, "cap": 5},
            {"px": 13, "py": 2, "cap": 5},
        ],
        "max_moves": 280,
        "spawn_every": 12,
    },
]


def _grid_to_pixel(gx, gy, ox, oy):
    return ox + gx * TILE, oy + gy * TILE


def _to_pixel_array(rows):
    return np.array(rows, dtype=np.int8)


def _build_level(ld, num):
    rows = ld["rows"]
    gh = len(rows)
    gw = max(len(r) for r in rows)
    ox = (CAM - gw * TILE) // 2
    oy = (CAM - gh * TILE) // 2

    sprites = []
    weight_index = obstacle_index = 0

    PX_ROAD = [[C_BLACK, C_BLACK], [C_BLACK, C_BLACK]]
    PX_WALL = [[C_DARKGREY, C_GREY], [C_GREY, C_DARKGREY]]
    PX_EXIT = [[C_PINK, C_MAGENTA], [C_MAGENTA, C_PINK]]
    PX_OBS = [[C_RED, C_MAROON], [C_MAROON, C_RED]]
    PX_MOV = [[C_RED, C_ORANGE], [C_ORANGE, C_RED]]
    PX_WGT = [[C_BLUE, C_LIGHTBLUE], [C_LIGHTBLUE, C_BLUE]]
    PX_HVY = [[C_LIGHTBLUE, C_BLUE], [C_BLUE, C_LIGHTBLUE]]
    PX_MWGT = PX_WGT
    PX_MHVY = PX_HVY
    PX_TR0 = [[C_GREEN, C_WHITE], [C_WHITE, C_GREEN]]
    PX_TR1 = [[C_ORANGE, C_WHITE], [C_WHITE, C_ORANGE]]
    PX_TR2 = [[C_PURPLE, C_WHITE], [C_WHITE, C_PURPLE]]

    for gy, row in enumerate(rows):
        for gx, ch in enumerate(row):
            sx, sy = _grid_to_pixel(gx, gy, ox, oy)
            if ch != "#":
                sprites.append(
                    Sprite(
                        pixels=PX_ROAD,
                        name="rd_%d_%d" % (gx, gy),
                        x=sx,
                        y=sy,
                        visible=True,
                        collidable=False,
                        layer=0,
                        tags=["road"],
                    )
                )
            if ch == "#":
                sprites.append(
                    Sprite(
                        pixels=PX_WALL,
                        name="wl_%d_%d" % (gx, gy),
                        x=sx,
                        y=sy,
                        visible=True,
                        collidable=False,
                        layer=1,
                        tags=["wall"],
                    )
                )
            elif ch == "T":
                sprites.append(
                    Sprite(
                        pixels=PX_TR0,
                        name="truck_0",
                        x=sx,
                        y=sy,
                        visible=True,
                        collidable=False,
                        layer=5,
                        tags=["truck", "truck_0"],
                    )
                )
            elif ch == "U":
                sprites.append(
                    Sprite(
                        pixels=PX_TR1,
                        name="truck_1",
                        x=sx,
                        y=sy,
                        visible=True,
                        collidable=False,
                        layer=5,
                        tags=["truck", "truck_1"],
                    )
                )
            elif ch == "V":
                sprites.append(
                    Sprite(
                        pixels=PX_TR2,
                        name="truck_2",
                        x=sx,
                        y=sy,
                        visible=True,
                        collidable=False,
                        layer=5,
                        tags=["truck", "truck_2"],
                    )
                )
            elif ch == "W":
                sprites.append(
                    Sprite(
                        pixels=PX_WGT,
                        name="wt_%d" % weight_index,
                        x=sx,
                        y=sy,
                        visible=True,
                        collidable=False,
                        layer=3,
                        tags=["weight", "weight_light"],
                    )
                )
                weight_index += 1
            elif ch == "H":
                sprites.append(
                    Sprite(
                        pixels=PX_HVY,
                        name="wt_%d" % weight_index,
                        x=sx,
                        y=sy,
                        visible=True,
                        collidable=False,
                        layer=3,
                        tags=["weight", "weight_heavy"],
                    )
                )
                weight_index += 1
            elif ch == "B":
                sprites.append(
                    Sprite(
                        pixels=PX_MWGT,
                        name="wt_%d" % weight_index,
                        x=sx,
                        y=sy,
                        visible=True,
                        collidable=False,
                        layer=3,
                        tags=["weight", "weight_light", "movbox"],
                    )
                )
                weight_index += 1
            elif ch == "G":
                sprites.append(
                    Sprite(
                        pixels=PX_MHVY,
                        name="wt_%d" % weight_index,
                        x=sx,
                        y=sy,
                        visible=True,
                        collidable=False,
                        layer=3,
                        tags=["weight", "weight_heavy", "movbox"],
                    )
                )
                weight_index += 1
            elif ch == "R":
                sprites.append(
                    Sprite(
                        pixels=PX_OBS,
                        name="ob_%d" % obstacle_index,
                        x=sx,
                        y=sy,
                        visible=True,
                        collidable=False,
                        layer=2,
                        tags=["obstacle"],
                    )
                )
                obstacle_index += 1
            elif ch == "M":
                sprites.append(
                    Sprite(
                        pixels=PX_MOV,
                        name="mov_%d" % obstacle_index,
                        x=sx,
                        y=sy,
                        visible=True,
                        collidable=False,
                        layer=2,
                        tags=["moving"],
                    )
                )
                obstacle_index += 1
            elif ch == "S":
                sprites.append(
                    Sprite(
                        pixels=PX_OBS,
                        name="spn_%d" % obstacle_index,
                        x=sx,
                        y=sy,
                        visible=False,
                        collidable=False,
                        layer=2,
                        tags=["spawn"],
                    )
                )
                obstacle_index += 1
            elif ch == "E":
                sprites.append(
                    Sprite(
                        pixels=PX_EXIT,
                        name="exit",
                        x=sx,
                        y=sy,
                        visible=True,
                        collidable=False,
                        layer=1,
                        tags=["exit"],
                    )
                )

    for i in range(MAX_LIVES):
        sprites.append(
            Sprite(
                pixels=[[C_GREEN, C_GREEN]],
                name="life_%d" % i,
                x=1 + i * 3,
                y=0,
                visible=True,
                collidable=False,
                layer=10,
                tags=["life"],
            )
        )

    max_cap = max(t["cap"] for t in ld["trucks"]) if ld["trucks"] else 0
    for seg in range(max_cap):
        sprites.append(
            Sprite(
                pixels=[[C_DARKGREY]],
                name="cap_%d" % seg,
                x=CAM - 1 - (max_cap - seg) * 2,
                y=0,
                visible=True,
                collidable=False,
                layer=10,
                tags=["cap"],
            )
        )

    bar_pixels = [[C_GREEN] * MOVE_BAR_WIDTH, [C_GREEN] * MOVE_BAR_WIDTH]
    sprites.append(
        Sprite(
            pixels=bar_pixels,
            name="move_bar",
            x=1,
            y=CAM - 2,
            visible=True,
            collidable=False,
            layer=10,
            tags=["move_bar"],
        )
    )

    total_boxes = sum(1 for r in rows for c in r if c in ("W", "H", "B", "G"))

    data = {
        "gw": gw,
        "gh": gh,
        "grid": "\n".join(rows),
        "trucks_info": [
            {"px": t["px"], "py": t["py"], "cap": t["cap"]} for t in ld["trucks"]
        ],
        "num_trucks": len(ld["trucks"]),
        "total_boxes": total_boxes,
        "max_moves": ld["max_moves"],
        "spawn_every": ld.get("spawn_every", 0),
    }
    return Level(
        sprites=sprites, grid_size=(CAM, CAM), data=data, name="Level %d" % num
    )


levels = [_build_level(ld, i + 1) for i, ld in enumerate(LEVELS)]


class Tr02(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._consecutive_reset_count = 0
        self._undo_stack: List[Dict] = []

        super().__init__(
            game_id="tr02",
            levels=levels,
            camera=Camera(0, 0, CAM, CAM, C_BACKGROUND, C_BACKGROUND),
            available_actions=[0, 1, 2, 3, 4, 5, 7],
        )

    def handle_reset(self) -> None:
        self._consecutive_reset_count += 1
        if self._consecutive_reset_count >= 2:
            self._consecutive_reset_count = 0
            self.lives = MAX_LIVES
            super().full_reset()
        else:
            self.lives = MAX_LIVES
            super().level_reset()

    def on_set_level(self, level: Level) -> None:
        cl = self.current_level
        rows = cl.get_data("grid").split("\n")
        gw = cl.get_data("gw")
        gh = cl.get_data("gh")
        self.gw, self.gh = gw, gh
        self.ox = (CAM - gw * TILE) // 2
        self.oy = (CAM - gh * TILE) // 2
        self.grid = [list(r) for r in rows]
        self.lives = MAX_LIVES
        self.active_truck = 0
        self.moves = 0
        self.max_moves = cl.get_data("max_moves")
        self.merged = False
        self._undo_stack = []

        info_list = cl.get_data("trucks_info")
        self.num_trucks = cl.get_data("num_trucks")
        self.trucks = []
        for ti in range(self.num_trucks):
            info = info_list[ti]
            sl = cl.get_sprites_by_tag("truck_%d" % ti)
            spr = sl[0] if sl else None
            self.trucks.append(
                {
                    "spr": spr,
                    "gx": info["px"],
                    "gy": info["py"],
                    "sx": info["px"],
                    "sy": info["py"],
                    "cap": info["cap"],
                    "wt": 0,
                }
            )

        weight_sprs = cl.get_sprites_by_tag("weight")
        self.boxes = []
        si = 0
        for gy, row in enumerate(rows):
            for gx, ch in enumerate(row):
                if ch in ("W", "H", "B", "G"):
                    w = 2 if ch in ("H", "G") else 1
                    moving = ch in ("B", "G")
                    spr = weight_sprs[si] if si < len(weight_sprs) else None
                    self.boxes.append(
                        {
                            "gx": gx,
                            "gy": gy,
                            "sx": gx,
                            "sy": gy,
                            "w": w,
                            "spr": spr,
                            "picked": False,
                            "moving": moving,
                            "dx": 1,
                        }
                    )
                    si += 1

        self.static_obstacles = set()
        for gy, row in enumerate(rows):
            for gx, ch in enumerate(row):
                if ch == "R":
                    self.static_obstacles.add((gx, gy))
        self.obstacles = set(self.static_obstacles)

        mov_sprs = cl.get_sprites_by_tag("moving")
        self.movers = []
        mi = 0
        for gy, row in enumerate(rows):
            for gx, ch in enumerate(row):
                if ch == "M":
                    spr = mov_sprs[mi] if mi < len(mov_sprs) else None
                    self.movers.append(
                        {
                            "gx": gx,
                            "gy": gy,
                            "sx": gx,
                            "sy": gy,
                            "dx": 1,
                            "spr": spr,
                        }
                    )
                    mi += 1

        spn_sprs = cl.get_sprites_by_tag("spawn")
        self.spawns = []
        spawn_index = 0
        for gy, row in enumerate(rows):
            for gx, ch in enumerate(row):
                if ch == "S":
                    spr = spn_sprs[spawn_index] if spawn_index < len(spn_sprs) else None
                    self.spawns.append(
                        {"gx": gx, "gy": gy, "spr": spr, "active": False}
                    )
                    spawn_index += 1
        self.spawn_every = cl.get_data("spawn_every")
        self.next_spawn = 0

        self.exit_pos = None
        for gy, row in enumerate(rows):
            for gx, ch in enumerate(row):
                if ch == "E":
                    self.exit_pos = (gx, gy)

        self.total_boxes = cl.get_data("total_boxes")
        self.collected = 0

        self.life_sprs = list(cl.get_sprites_by_tag("life"))
        bar_list = cl.get_sprites_by_tag("move_bar")
        self.bar_spr = bar_list[0] if bar_list else None
        self.cap_sprs = list(cl.get_sprites_by_tag("cap"))

        self._draw_lives()
        self._draw_move_bar()
        self._mark_active()

    def _draw_lives(self):
        for i, s in enumerate(self.life_sprs):
            if i < self.lives:
                s.pixels = _to_pixel_array([[C_GREEN, C_GREEN]])
            else:
                s.pixels = _to_pixel_array([[C_DARKGREY, C_DARKGREY]])

    def _draw_caps(self):
        cols = [C_GREEN, C_ORANGE, C_PURPLE]
        t = self.trucks[self.active_truck]
        cap = t["cap"]
        cur = t["wt"]
        c = cols[self.active_truck % 3]
        for si, s in enumerate(self.cap_sprs):
            if si < cap:
                s.set_visible(True)
                if si < cur:
                    s.pixels = _to_pixel_array([[c]])
                else:
                    s.pixels = _to_pixel_array([[C_DARKGREY]])
            else:
                s.set_visible(False)

    def _draw_move_bar(self):
        if not self.bar_spr:
            return
        remaining = max(0, self.max_moves - self.moves)
        pct = remaining / self.max_moves if self.max_moves > 0 else 0
        fill = int(pct * MOVE_BAR_WIDTH)
        fill = max(0, min(fill, MOVE_BAR_WIDTH))
        if pct > 0.5:
            fc = C_GREEN
        elif pct > 0.25:
            fc = C_YELLOW
        elif pct > 0:
            fc = C_RED
        else:
            fc = C_DARKGREY
        row = [fc if x < fill else C_DARKGREY for x in range(MOVE_BAR_WIDTH)]
        self.bar_spr.pixels = _to_pixel_array([row, row])

    def _mark_active(self):
        cols = [C_GREEN, C_ORANGE, C_PURPLE]
        for ti, t in enumerate(self.trucks):
            if not t["spr"]:
                continue
            c = cols[ti % 3]
            if ti == self.active_truck:
                t["spr"].pixels = _to_pixel_array([[c, C_WHITE], [C_WHITE, c]])
            else:
                t["spr"].pixels = _to_pixel_array([[c, c], [c, c]])
        self._draw_caps()

    def _save_undo_state(self) -> None:
        snapshot = {
            "lives": self.lives,
            "active_truck": self.active_truck,
            "moves": self.moves,
            "merged": self.merged,
            "collected": self.collected,
            "next_spawn": self.next_spawn,
            "trucks": [
                {"gx": t["gx"], "gy": t["gy"], "wt": t["wt"]} for t in self.trucks
            ],
            "boxes": [
                {"gx": b["gx"], "gy": b["gy"], "picked": b["picked"], "dx": b["dx"]}
                for b in self.boxes
            ],
            "movers": [
                {"gx": m["gx"], "gy": m["gy"], "dx": m["dx"]} for m in self.movers
            ],
            "spawns": [{"active": s["active"]} for s in self.spawns],
            "obstacles": set(self.obstacles),
        }
        self._undo_stack.append(snapshot)
        if len(self._undo_stack) > 50:
            self._undo_stack.pop(0)

    def _apply_undo(self) -> bool:
        if not self._undo_stack:
            return False
        snapshot = self._undo_stack.pop()

        self.lives = snapshot["lives"]
        self.active_truck = snapshot["active_truck"]
        self.moves = snapshot["moves"]
        self.merged = snapshot["merged"]
        self.collected = snapshot["collected"]
        self.next_spawn = snapshot["next_spawn"]
        self.obstacles = snapshot["obstacles"]

        for i, td in enumerate(snapshot["trucks"]):
            t = self.trucks[i]
            t["gx"] = td["gx"]
            t["gy"] = td["gy"]
            t["wt"] = td["wt"]
            if t["spr"]:
                t["spr"].set_visible(True)
                t["spr"].set_position(
                    *_grid_to_pixel(t["gx"], t["gy"], self.ox, self.oy)
                )

        for i, bd in enumerate(snapshot["boxes"]):
            b = self.boxes[i]
            b["gx"] = bd["gx"]
            b["gy"] = bd["gy"]
            b["picked"] = bd["picked"]
            b["dx"] = bd["dx"]
            if b["spr"]:
                b["spr"].set_visible(not b["picked"])
                b["spr"].set_position(
                    *_grid_to_pixel(b["gx"], b["gy"], self.ox, self.oy)
                )

        for i, md in enumerate(snapshot["movers"]):
            m = self.movers[i]
            m["gx"] = md["gx"]
            m["gy"] = md["gy"]
            m["dx"] = md["dx"]
            if m["spr"]:
                m["spr"].set_position(
                    *_grid_to_pixel(m["gx"], m["gy"], self.ox, self.oy)
                )

        for i, sd in enumerate(snapshot["spawns"]):
            s = self.spawns[i]
            s["active"] = sd["active"]
            if s["spr"]:
                s["spr"].set_visible(s["active"])

        if self.merged:
            leader = self.trucks[self.active_truck]
            for ti, t in enumerate(self.trucks):
                if ti != self.active_truck and t["spr"]:
                    t["spr"].set_visible(False)
            if leader["spr"]:
                leader["spr"].pixels = _to_pixel_array(
                    [[C_WHITE, C_GREEN], [C_GREEN, C_WHITE]]
                )

        self._draw_lives()
        self._draw_move_bar()
        self._mark_active()
        return True

    def _move_patrols(self):
        for m in self.movers:
            nx = m["gx"] + m["dx"]
            if nx < 0 or nx >= self.gw or self.grid[m["gy"]][nx] == "#":
                m["dx"] = -m["dx"]
                nx = m["gx"] + m["dx"]
                if nx < 0 or nx >= self.gw or self.grid[m["gy"]][nx] == "#":
                    continue
            m["gx"] = nx
            if m["spr"]:
                m["spr"].set_position(
                    *_grid_to_pixel(m["gx"], m["gy"], self.ox, self.oy)
                )

    def _check_mover_collision(self):
        active = self.trucks[self.active_truck]
        for m in self.movers:
            if active["gx"] == m["gx"] and active["gy"] == m["gy"]:
                if self._lose_life():
                    return True
        return False

    def _is_mover_at(self, gx, gy):
        for m in self.movers:
            if m["gx"] == gx and m["gy"] == gy:
                return True
        return False

    def _move_boxes(self):
        for box in self.boxes:
            if not box["moving"] or box["picked"]:
                continue
            nx = box["gx"] + box["dx"]
            if nx < 0 or nx >= self.gw or self.grid[box["gy"]][nx] == "#":
                box["dx"] = -box["dx"]
                nx = box["gx"] + box["dx"]
                if nx < 0 or nx >= self.gw or self.grid[box["gy"]][nx] == "#":
                    continue
            if (nx, box["gy"]) in self.obstacles:
                box["dx"] = -box["dx"]
                continue
            box["gx"] = nx
            if box["spr"]:
                box["spr"].set_position(
                    *_grid_to_pixel(box["gx"], box["gy"], self.ox, self.oy)
                )

    def _tick_move(self):
        self.moves += 1
        self._draw_move_bar()
        if self.spawn_every > 0 and self.next_spawn < len(self.spawns):
            if self.moves % self.spawn_every == 0:
                s = self.spawns[self.next_spawn]
                s["active"] = True
                self.obstacles.add((s["gx"], s["gy"]))
                if s["spr"]:
                    s["spr"].set_visible(True)
                self.next_spawn += 1
        if self.moves >= self.max_moves:
            return True
        return False

    def _lose_life(self):
        self.lives -= 1
        self._draw_lives()
        if self.lives <= 0:
            return True
        self.moves = 0
        self._draw_move_bar()
        self.merged = False
        self.active_truck = 0
        self.collected = 0
        for t in self.trucks:
            t["gx"], t["gy"] = t["sx"], t["sy"]
            t["wt"] = 0
            if t["spr"]:
                t["spr"].set_visible(True)
                t["spr"].set_position(
                    *_grid_to_pixel(t["gx"], t["gy"], self.ox, self.oy)
                )
        for box in self.boxes:
            box["picked"] = False
            if box["moving"]:
                box["gx"], box["gy"] = box["sx"], box["sy"]
                box["dx"] = 1
            if box["spr"]:
                box["spr"].set_visible(True)
                if box["moving"]:
                    box["spr"].set_position(
                        *_grid_to_pixel(box["gx"], box["gy"], self.ox, self.oy)
                    )
        for m in self.movers:
            m["gx"], m["gy"] = m["sx"], m["sy"]
            m["dx"] = 1
            if m["spr"]:
                m["spr"].set_position(
                    *_grid_to_pixel(m["gx"], m["gy"], self.ox, self.oy)
                )
        self.obstacles = set(self.static_obstacles)
        for s in self.spawns:
            s["active"] = False
            if s["spr"]:
                s["spr"].set_visible(False)
        self.next_spawn = 0
        self._undo_stack = []
        self._mark_active()
        return False

    def _sink(self, ti):
        t = self.trucks[ti]
        if t["cap"] <= 0:
            return
        if t["wt"] <= t["cap"]:
            return
        ny = t["gy"] + 1
        if ny >= self.gh:
            return
        if self.grid[ny][t["gx"]] == "#":
            return
        for oti, ot in enumerate(self.trucks):
            if oti != ti and ot["gx"] == t["gx"] and ot["gy"] == ny:
                return
        if (t["gx"], ny) in self.obstacles:
            return
        if self._is_mover_at(t["gx"], ny):
            return
        t["gy"] = ny
        if t["spr"]:
            t["spr"].set_position(*_grid_to_pixel(t["gx"], t["gy"], self.ox, self.oy))

    def _pick(self, box):
        t = self.trucks[self.active_truck]
        box["picked"] = True
        t["wt"] += box["w"]
        self.collected += 1
        if box["spr"]:
            box["spr"].set_visible(False)
        self._draw_caps()
        self._sink(self.active_truck)
        if self.collected >= self.total_boxes and not self.merged:
            all_ok = True
            for truck in self.trucks:
                if truck["cap"] > 0 and truck["wt"] > truck["cap"]:
                    all_ok = False
                    break
            if all_ok:
                self._merge_trucks()

    def _try_transfer(self, other_ti):
        me = self.trucks[self.active_truck]
        other = self.trucks[other_ti]
        my_space = me["cap"] - me["wt"] if me["cap"] > 0 else 999
        other_excess = other["wt"] - other["cap"] if other["cap"] > 0 else 0
        if my_space <= 0 or other_excess <= 0:
            my_excess = me["wt"] - me["cap"] if me["cap"] > 0 else 0
            other_space = other["cap"] - other["wt"] if other["cap"] > 0 else 999
            if my_excess > 0 and other_space > 0:
                transfer = min(my_excess, other_space)
                me["wt"] -= transfer
                other["wt"] += transfer
                self._draw_caps()
            return
        transfer = min(other_excess, my_space)
        other["wt"] -= transfer
        me["wt"] += transfer
        self._draw_caps()

    def _merge_trucks(self):
        self.merged = True
        leader = self.trucks[self.active_truck]
        gx, gy = leader["gx"], leader["gy"]
        px, py = _grid_to_pixel(gx, gy, self.ox, self.oy)
        for ti, t in enumerate(self.trucks):
            t["gx"], t["gy"] = gx, gy
            if ti != self.active_truck and t["spr"]:
                t["spr"].set_visible(False)
        if leader["spr"]:
            leader["spr"].pixels = _to_pixel_array(
                [[C_WHITE, C_GREEN], [C_GREEN, C_WHITE]]
            )
            leader["spr"].set_position(px, py)

    def _check_win(self):
        if self.exit_pos is None:
            return False
        if not self.merged:
            return False
        t = self.trucks[self.active_truck]
        if (t["gx"], t["gy"]) != self.exit_pos:
            return False
        self.next_level()
        return True

    def _do_move(self, dx, dy):
        if self._tick_move():
            if self._lose_life():
                self.lose()
            self.complete_action()
            return

        if dx == 0 and dy == 0:
            self.complete_action()
            return

        t = self.trucks[self.active_truck]
        nx = t["gx"] + dx
        ny = t["gy"] + dy

        if nx < 0 or nx >= self.gw or ny < 0 or ny >= self.gh:
            self.complete_action()
            return
        if self.grid[ny][nx] == "#":
            self.complete_action()
            return

        transfer_ti = -1
        if not self.merged:
            for oti, ot in enumerate(self.trucks):
                if oti != self.active_truck:
                    if ot["gx"] == nx and ot["gy"] == ny:
                        transfer_ti = oti
                        break

        if self.merged:
            for truck in self.trucks:
                truck["gx"], truck["gy"] = nx, ny
            if t["spr"]:
                t["spr"].set_position(*_grid_to_pixel(nx, ny, self.ox, self.oy))
        else:
            t["gx"], t["gy"] = nx, ny
            if t["spr"]:
                t["spr"].set_position(*_grid_to_pixel(nx, ny, self.ox, self.oy))

        if transfer_ti >= 0:
            self._try_transfer(transfer_ti)
            if self.collected >= self.total_boxes and not self.merged:
                all_ok = True
                for truck in self.trucks:
                    if truck["cap"] > 0 and truck["wt"] > truck["cap"]:
                        all_ok = False
                        break
                if all_ok:
                    self._merge_trucks()

        if (nx, ny) in self.obstacles:
            if self._lose_life():
                self.lose()
                self.complete_action()
                return
            self._mark_active()
            self.complete_action()
            return

        if self._is_mover_at(nx, ny):
            if self._lose_life():
                self.lose()
                self.complete_action()
                return
            self._mark_active()
            self.complete_action()
            return

        if not self.merged:
            for box in self.boxes:
                if not box["picked"] and box["gx"] == nx and box["gy"] == ny:
                    self._pick(box)
                    break

        if not self.merged:
            for ti in range(self.num_trucks):
                self._sink(ti)

        self._move_patrols()
        self._move_boxes()

        if not self.merged:
            picked_any = True
            while picked_any:
                picked_any = False
                for box in self.boxes:
                    if not box["picked"] and box["moving"]:
                        for truck in self.trucks:
                            if truck["gx"] == box["gx"] and truck["gy"] == box["gy"]:
                                old_active = self.active_truck
                                for ti, t in enumerate(self.trucks):
                                    if t is truck:
                                        self.active_truck = ti
                                        break
                                self._pick(box)
                                self.active_truck = old_active
                                picked_any = True
                                break
                    if picked_any:
                        break

        if self._check_mover_collision():
            self.lose()
            self.complete_action()
            return

        if self._check_win():
            self.complete_action()
            return

        if not self.merged:
            self._mark_active()
        self.complete_action()

    def step(self) -> None:
        if self.action.id == GameAction.RESET:
            self.complete_action()
            return

        if self.action.id == GameAction.ACTION7:
            if self.moves >= self.max_moves:
                if self._lose_life():
                    self.lose()
                self.complete_action()
                return
            current_moves = self.moves
            self._apply_undo()
            self.moves = current_moves + 1
            self._draw_move_bar()
            self.complete_action()
            return

        self._consecutive_reset_count = 0

        if self.action.id == GameAction.ACTION5:
            if self.merged:
                self.complete_action()
                return
            self._save_undo_state()
            if self._tick_move():
                if self._lose_life():
                    self.lose()
                self.complete_action()
                return
            if self.num_trucks > 1:
                self.active_truck = (self.active_truck + 1) % self.num_trucks
                self._mark_active()
            self._move_patrols()
            self._move_boxes()
            if self._check_mover_collision():
                self.lose()
                self.complete_action()
                return
            self.complete_action()
            return

        dx, dy = 0, 0
        if self.action.id == GameAction.ACTION1:
            dy = -1
        elif self.action.id == GameAction.ACTION2:
            dy = 1
        elif self.action.id == GameAction.ACTION3:
            dx = -1
        elif self.action.id == GameAction.ACTION4:
            dx = 1
        else:
            self.complete_action()
            return

        self._save_undo_state()
        self._do_move(dx, dy)

    @property
    def extra_state(self) -> dict:
        idx = self.level_index
        remaining = (
            max(0, self.max_moves - self.moves) if hasattr(self, "max_moves") else 0
        )
        return {
            "moves_remaining": remaining,
            "moves_used": self.moves if hasattr(self, "moves") else 0,
            "lives": self.lives if hasattr(self, "lives") else MAX_LIVES,
            "active_truck": self.active_truck if hasattr(self, "active_truck") else 0,
            "num_trucks": self.num_trucks if hasattr(self, "num_trucks") else 0,
            "collected": self.collected if hasattr(self, "collected") else 0,
            "total_boxes": self.total_boxes if hasattr(self, "total_boxes") else 0,
            "merged": self.merged if hasattr(self, "merged") else False,
            "level_title": "TRUCK PUZZLE -- Level %d/%d" % (idx + 1, N_LEVELS),
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
        self._engine = Tr02(seed=seed)
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

    def step(self, action: str) -> StepResult:
        e = self._engine

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
        info: Dict = {"action": action}

        level_before = e.level_index

        frame = e.perform_action(ActionInput(id=game_action), raw=True)

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

    def _build_text_observation(self) -> str:
        e = self._engine

        if not hasattr(e, "grid"):
            return ""

        if e.lives <= 0:
            return (
                "=== GAME OVER ===\nAll lives lost on Level %d.\n"
                "Type 'reset' to retry this level. Type 'reset' again to restart from Level 1."
                % (e.level_index + 1)
            )

        remaining = max(0, e.max_moves - e.moves)
        bar_w = 40
        filled = round(remaining * bar_w / max(e.max_moves, 1))
        move_bar = "[" + "#" * filled + "-" * (bar_w - filled) + "]"
        lives_bar = ("*" * max(e.lives, 0)) + ("." * (MAX_LIVES - max(e.lives, 0)))

        truck_info = " ".join(
            "T%d(%d/%d)" % (i, t["wt"], t["cap"]) for i, t in enumerate(e.trucks)
        )
        active_str = "T%d" % e.active_truck
        if e.merged:
            active_str = "MERGED"

        lines: List[str] = [
            "=== TRUCK PUZZLE L%d/%d ===" % (e.level_index + 1, N_LEVELS),
            "Moves %s %d/%d  Lives [%s]"
            % (move_bar, remaining, e.max_moves, lives_bar),
            "Trucks: %s  Active: %s  Boxes: %d/%d"
            % (truck_info, active_str, e.collected, e.total_boxes),
            "",
        ]

        for gy in range(e.gh):
            row: List[str] = []
            for gx in range(e.gw):
                ch = e.grid[gy][gx]
                cell_char = "."
                if ch == "#":
                    cell_char = "#"
                elif (gx, gy) == e.exit_pos:
                    cell_char = "E" if e.merged else "e"
                elif any(t["gx"] == gx and t["gy"] == gy for t in e.trucks):
                    for ti, t in enumerate(e.trucks):
                        if t["gx"] == gx and t["gy"] == gy:
                            if e.merged:
                                cell_char = "@"
                            elif ti == e.active_truck:
                                cell_char = "@"
                            else:
                                cell_char = str(ti)
                            break
                elif any(
                    not b["picked"] and b["gx"] == gx and b["gy"] == gy for b in e.boxes
                ):
                    for b in e.boxes:
                        if not b["picked"] and b["gx"] == gx and b["gy"] == gy:
                            if b["moving"]:
                                cell_char = "b" if b["w"] == 1 else "g"
                            else:
                                cell_char = "W" if b["w"] == 1 else "H"
                            break
                elif (gx, gy) in e.obstacles:
                    cell_char = "R"
                elif any(m["gx"] == gx and m["gy"] == gy for m in e.movers):
                    cell_char = "M"
                elif any(
                    s["active"] and s["gx"] == gx and s["gy"] == gy for s in e.spawns
                ):
                    cell_char = "S"
                row.append(cell_char)
            lines.append(" ".join(row))

        return "\n".join(lines)

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
        index_grid = e.camera.render(e.current_level.get_sprites())
        if index_grid is None or index_grid.size == 0:
            return None
        h, w = index_grid.shape[:2]
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for idx in range(len(ARC_PALETTE)):
            mask = index_grid == idx
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            rgb[mask] = ARC_PALETTE[idx]
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


game = Tr02()


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
