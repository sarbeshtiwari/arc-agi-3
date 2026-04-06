import random
import struct
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import gymnasium as gym
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

BG = 0
WALL = 5
PLAYER_C = 14
PLAYER_ZONE = 3
FORTIFIED_C = 12

ENEMY_COLORS = [2, 8, 9]
SCOUT_CORE = 15

POWER_CELL = 4
TARGET_MARK = 6
TARGET_HINT = 13
BOMB_ARMED = 1
BOMB_WARN = 7
BEACON_C = 11

LIFE_ON = 3
LIFE_OFF = 5
BAR_FULL = 3
BAR_EMPTY = 9
BAR_DANGER = 2
DOMIN_SAFE = 3
DOMIN_WARN = 7
DOMIN_CRIT = 2
DEATH_FLASH = 2
GAMEOVER_X = 9
PHASE2_IND = 6

FORT_DURATION = 5
DOMINATION_THRESHOLD = 0.70

CELL_CHARS = {
    BG: ".",
    WALL: "#",
    PLAYER_C: "P",
    PLAYER_ZONE: "+",
    FORTIFIED_C: "F",
    2: "1",
    8: "2",
    9: "3",
    SCOUT_CORE: "S",
    POWER_CELL: "*",
    TARGET_MARK: "T",
    TARGET_HINT: "t",
    BOMB_ARMED: "B",
    BOMB_WARN: "!",
    BEACON_C: "R",
}


def _parse_grid(rows):
    gh = len(rows)
    gw = max(len(r) for r in rows)
    w = set()
    f = set()
    ps = None
    t = set()
    es = {}
    pc = set()
    pr = set()
    bo = []
    be = set()
    sc = []
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch == "#":
                w.add((x, y))
            else:
                f.add((x, y))
                if ch == "P":
                    ps = (x, y)
                    pr.add((x, y))
                elif ch == "T":
                    t.add((x, y))
                elif ch in "123":
                    es.setdefault(int(ch) - 1, []).append((x, y))
                elif ch == "*":
                    pc.add((x, y))
                elif ch == "B":
                    bo.append((x, y))
                elif ch == "R":
                    be.add((x, y))
                elif ch == "s":
                    sc.append({"eid": 0, "x": x, "y": y})
                elif ch == "t":
                    sc.append({"eid": 1, "x": x, "y": y})
                elif ch == "u":
                    sc.append({"eid": 2, "x": x, "y": y})
    return {
        "size": (gw, gh),
        "walls": w,
        "floor": f,
        "player_start": ps,
        "targets": t,
        "enemy_seeds": es,
        "power_cells": pc,
        "pre_claimed": pr,
        "bombs": bo,
        "beacons": be,
        "scout_spawns": sc,
    }


_L1 = [
    "###########",
    "#.........#",
    "#.1..P..2.#",
    "#...####..#",
    "#...#.....#",
    "#...#..T..#",
    "#.........#",
    "#....T....#",
    "###########",
]

_L2 = [
    "###########",
    "#.........#",
    "#....P....#",
    "#..1...2..#",
    "#.........#",
    "#R...3....#",
    "#.........#",
    "#..T...T..#",
    "###########",
]

_L3 = [
    "###########",
    "#.........#",
    "#.P.......#",
    "#..##..*..#",
    "#..##.T...#",
    "#.........#",
    "#..s1.B...#",
    "#.........#",
    "#.....T...#",
    "###########",
]

_L4 = [
    "#############",
    "#...........#",
    "#.P.........#",
    "#...##......#",
    "#...##..T...#",
    "#.....1..*..#",
    "#.....##....#",
    "#..R..##..T.#",
    "#.......2...#",
    "#..T..B..s..#",
    "#...........#",
    "#############",
]

LEVEL_DEFS = [
    {"grid": _L1, "max_turns": 65, "expand_rate": 2, "bomb_timer": 0},
    {"grid": _L2, "max_turns": 40, "expand_rate": 2, "bomb_timer": 0},
    {"grid": _L3, "max_turns": 35, "expand_rate": 2, "bomb_timer": 14},
    {"grid": _L4, "max_turns": 75, "expand_rate": 2, "bomb_timer": 18},
]
PARSED = []
for _d in LEVEL_DEFS:
    _p = _parse_grid(_d["grid"])
    _p["max_turns"] = _d["max_turns"]
    _p["expand_rate"] = _d["expand_rate"]
    _p["bomb_timer"] = _d["bomb_timer"]
    PARSED.append(_p)

PLAYER_START_POSITIONS = [
    [(5, 2), (1, 1), (9, 1), (5, 6)],
    [(5, 2), (3, 1), (7, 1), (5, 4)],
    [(2, 2), (1, 1), (9, 1), (1, 7)],
    [(2, 2), (4, 2), (2, 4), (6, 2)],
]


class ZoneHUD(RenderableUserDisplay):
    def __init__(self, game):
        self.game = game

    def render_interface(self, frame):
        g = self.game
        w = frame.shape[1]
        h = frame.shape[0]
        if g.death_flash > 0:
            for x in range(w):
                frame[0, x] = DEATH_FLASH
                frame[h - 1, x] = DEATH_FLASH
            for y in range(h):
                frame[y, 0] = DEATH_FLASH
                frame[y, w - 1] = DEATH_FLASH
            return frame
        if g._game_over:
            mx, my = w // 2, h // 2
            sz = min(mx, my) - 1
            for i in range(-sz, sz + 1):
                if 0 <= my + i < h and 0 <= mx + i < w:
                    frame[my + i, mx + i] = GAMEOVER_X
                if 0 <= my - i < h and 0 <= mx + i < w:
                    frame[my - i, mx + i] = GAMEOVER_X
            return frame
        for i in range(g.TOTAL_LIVES):
            x = 1 + i * 2
            if x < w:
                frame[1, x] = LIFE_ON if i < g._lives else LIFE_OFF
        px = w - 2
        if 0 < px < w:
            frame[1, px] = PHASE2_IND if g._in_phase2() else LIFE_OFF
        bs = g.TOTAL_LIVES * 2 + 2
        bl = w - bs - 2
        if g.max_turns > 0 and bl > 0:
            r = g.turns_left / g.max_turns
            fl = max(0, int(bl * r))
            bc = BAR_FULL if r > 0.3 else BAR_DANGER
            for i in range(bl):
                x = bs + i
                if x < w:
                    frame[0, x] = bc if i < fl else BAR_EMPTY
        tf = len(g.floor_set)
        if tf > 0:
            te = sum(len(z) for z in g.enemy_zones.values())
            er = te / tf
            dl = w - 2
            fd = int(dl * er)
            dc = DOMIN_CRIT if er > 0.5 else (DOMIN_WARN if er > 0.3 else DOMIN_SAFE)
            for i in range(dl):
                x = 1 + i
                if 2 < h and x < w:
                    frame[2, x] = dc if i < fd else BG
        return frame


class Zn16(ARCBaseGame):
    TOTAL_LIVES = 3

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self._rng = random.Random(seed)
        self.GW = 0
        self.GH = 0
        self.player_x = 1
        self.player_y = 1
        self.wall_set = set()
        self.floor_set = set()
        self.player_zone = set()
        self.enemy_zones = {}
        self.targets = set()
        self.collected_targets = set()
        self.power_cells = set()
        self.power_owners = {}
        self.fortified = {}
        self.bombs = []
        self.beacons = set()
        self.scouts = []
        self.max_turns = 0
        self.turns_left = 0
        self.turn_counter = 0
        self.expand_rate = 2
        self._lives = self.TOTAL_LIVES
        self.death_flash = 0
        self._game_over = False
        self._undo_stack = []
        self.board_sprite = None
        self.hud = ZoneHUD(self)
        levels = []
        for i, pd in enumerate(PARSED):
            gw, gh = pd["size"]
            levels.append(
                Level(
                    sprites=[],
                    grid_size=(gw, gh),
                    name="Zone Level {}".format(i + 1),
                    data=pd,
                )
            )
        gw0, gh0 = PARSED[0]["size"]
        cam = Camera(
            x=0,
            y=0,
            width=gw0,
            height=gh0,
            background=BG,
            letter_box=0,
            interfaces=[self.hud],
        )
        super().__init__(
            game_id="zn16",
            levels=levels,
            camera=cam,
            available_actions=[0, 1, 2, 3, 4, 5, 7],
        )

    def _in_phase2(self):
        return all(len(z) == 0 for z in self.enemy_zones.values())

    def _save_state(self):
        return {
            "player_x": self.player_x,
            "player_y": self.player_y,
            "player_zone": set(self.player_zone),
            "enemy_zones": {eid: set(z) for eid, z in self.enemy_zones.items()},
            "collected_targets": set(self.collected_targets),
            "power_cells": set(self.power_cells),
            "power_owners": {k: set(v) for k, v in self.power_owners.items()},
            "fortified": dict(self.fortified),
            "bombs": [dict(b) for b in self.bombs],
            "beacons": set(self.beacons),
            "scouts": [dict(s) for s in self.scouts],
            "turns_left": self.turns_left,
            "turn_counter": self.turn_counter,
            "death_flash": self.death_flash,
            "game_over": self._game_over,
        }

    def _restore_state(self, state):
        self.player_x = state["player_x"]
        self.player_y = state["player_y"]
        self.player_zone = set(state["player_zone"])
        self.enemy_zones = {eid: set(z) for eid, z in state["enemy_zones"].items()}
        self.collected_targets = set(state["collected_targets"])
        self.power_cells = set(state["power_cells"])
        self.power_owners = {k: set(v) for k, v in state["power_owners"].items()}
        self.fortified = dict(state["fortified"])
        self.bombs = [dict(b) for b in state["bombs"]]
        self.beacons = set(state["beacons"])
        self.scouts = [dict(s) for s in state["scouts"]]
        self.turns_left = state["turns_left"]
        self.turn_counter = state["turn_counter"]
        self.death_flash = state["death_flash"]
        self._game_over = state["game_over"]
        self._rebuild_board()

    def on_set_level(self, level: Level) -> None:
        self.GW, self.GH = self.current_level.get_data("size")
        self.camera.width = self.GW
        self.camera.height = self.GH
        self.wall_set = set(self.current_level.get_data("walls"))
        self.floor_set = set(self.current_level.get_data("floor"))
        start_pos = self._rng.choice(PLAYER_START_POSITIONS[self.level_index])
        self.player_x, self.player_y = start_pos
        self.player_zone = {start_pos}
        enemy_seeds = self.current_level.get_data("enemy_seeds")
        self.enemy_zones = {eid: set(seeds) for eid, seeds in enemy_seeds.items()}
        self.targets = set(self.current_level.get_data("targets"))
        self.collected_targets = set()
        self.power_cells = set(self.current_level.get_data("power_cells"))
        self.power_owners = {}
        self.fortified = {}
        self.bombs = []
        bt = self.current_level.get_data("bomb_timer")
        if bt > 0:
            for bx, by in self.current_level.get_data("bombs"):
                ne = 0
                bd = 9999
                for eid, seeds in enemy_seeds.items():
                    for sx, sy in seeds:
                        d = abs(bx - sx) + abs(by - sy)
                        if d < bd:
                            bd = d
                            ne = eid
                self.bombs.append({"x": bx, "y": by, "eid": ne, "timer": bt})
        self.beacons = set(self.current_level.get_data("beacons"))
        self.scouts = [
            {
                "eid": s["eid"],
                "x": s["x"],
                "y": s["y"],
                "dx": 1,
                "dy": 0,
                "spawn_x": s["x"],
                "spawn_y": s["y"],
            }
            for s in self.current_level.get_data("scout_spawns")
        ]
        self.max_turns = self.current_level.get_data("max_turns")
        self.turns_left = self.max_turns
        self.turn_counter = 0
        self.expand_rate = self.current_level.get_data("expand_rate")
        self.death_flash = 0
        self._game_over = False
        self._undo_stack = []
        self._build_board()

    def handle_reset(self):
        self._rng = random.Random(self._seed)
        self._lives = self.TOTAL_LIVES
        self._game_over = False
        self.death_flash = 0
        self._undo_stack = []
        super().handle_reset()

    def step(self) -> None:
        if self.death_flash > 0:
            self.death_flash -= 1
            self._rebuild_board()
            self.complete_action()
            return
        if self._game_over:
            self.complete_action()
            return
        a = self.action.id
        if a == GameAction.ACTION7:
            if self._undo_stack:
                prev = self._undo_stack.pop()
                self._restore_state(prev)
            self._tick()
            self._check_end()
            self.complete_action()
            return
        if a == GameAction.ACTION5:
            self._undo_stack.append(self._save_state())
            self._do_fortify()
            self._try_collect_target()
            self._tick()
            self._check_end()
            self.complete_action()
            return
        dx, dy = 0, 0
        if a == GameAction.ACTION1:
            dy = -1
        elif a == GameAction.ACTION2:
            dy = 1
        elif a == GameAction.ACTION3:
            dx = -1
        elif a == GameAction.ACTION4:
            dx = 1
        if dx == 0 and dy == 0:
            self.complete_action()
            return
        self._undo_stack.append(self._save_state())
        nx, ny = self.player_x + dx, self.player_y + dy
        if self._wk(nx, ny):
            self.player_x = nx
            self.player_y = ny
        self._claim(self.player_x, self.player_y)
        self._check_beacon()
        self._check_defuse()
        self._try_collect_target()
        if self._scout_hit():
            self.complete_action()
            return
        self._tick()
        self._check_end()
        self.complete_action()

    def _claim(self, x, y):
        pos = (x, y)
        if pos in self.wall_set:
            return
        if pos in self.player_zone:
            return
        split_eid = None
        for eid in list(self.enemy_zones.keys()):
            if pos in self.enemy_zones[eid]:
                self.enemy_zones[eid].discard(pos)
                split_eid = eid
                break
        self.player_zone.add(pos)
        if pos in self.power_cells:
            self.power_cells.discard(pos)
            self.power_owners.setdefault("player", set()).add(pos)
        if split_eid is not None:
            self._check_split(split_eid)

    def _check_split(self, eid):
        zone = self.enemy_zones.get(eid)
        if not zone:
            return
        rem = set(zone)
        comps = []
        while rem:
            s = next(iter(rem))
            comp = set()
            q = [s]
            while q:
                cur = q.pop()
                if cur in comp or cur not in rem:
                    continue
                comp.add(cur)
                rem.discard(cur)
                cx, cy = cur
                for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                    nb = (cx + dx, cy + dy)
                    if nb in rem:
                        q.append(nb)
            comps.append(comp)
        if len(comps) <= 1:
            return
        comps.sort(key=len, reverse=True)
        self.enemy_zones[eid] = comps[0]
        for sm in comps[1:]:
            for cell in sm:
                was_power = False
                for ok in list(self.power_owners.keys()):
                    if cell in self.power_owners[ok]:
                        self.power_owners[ok].discard(cell)
                        was_power = True
                if was_power:
                    self.power_cells.add(cell)

    def _do_fortify(self):
        pos = (self.player_x, self.player_y)
        if pos not in self.player_zone:
            self._claim(self.player_x, self.player_y)
        for dx, dy in [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]:
            fp = (pos[0] + dx, pos[1] + dy)
            if fp in self.player_zone:
                self.fortified[fp] = FORT_DURATION

    def _check_beacon(self):
        pos = (self.player_x, self.player_y)
        if pos not in self.beacons:
            return
        self.beacons.discard(pos)
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if abs(dx) + abs(dy) > 2:
                    continue
                rp = (pos[0] + dx, pos[1] + dy)
                if rp in self.floor_set and rp not in self.wall_set:
                    for eid in self.enemy_zones:
                        self.enemy_zones[eid].discard(rp)
                    self.player_zone.add(rp)

    def _check_defuse(self):
        pos = (self.player_x, self.player_y)
        self.bombs = [b for b in self.bombs if (b["x"], b["y"]) != pos]

    def _try_collect_target(self):
        if not self._in_phase2():
            return
        pos = (self.player_x, self.player_y)
        if pos in self.targets and pos not in self.collected_targets:
            self.collected_targets.add(pos)

    def _scout_hit(self):
        pos = (self.player_x, self.player_y)
        for sc in self.scouts:
            if (sc["x"], sc["y"]) == pos:
                if any(pos in z for z in self.enemy_zones.values()):
                    self._lose_life()
                    return True
                else:
                    sc["x"] = sc["spawn_x"]
                    sc["y"] = sc["spawn_y"]
        return False

    def _tick(self):
        self.turns_left -= 1
        self.turn_counter += 1
        ex = [p for p in self.fortified if self.fortified[p] <= 1]
        for p in self.fortified:
            self.fortified[p] -= 1
        for p in ex:
            del self.fortified[p]
        if self._in_phase2():
            self._rebuild_board()
            return
        if self.turn_counter % self.expand_rate == 0:
            self._expand()
        self._bomb_tick()
        self._move_scouts()
        if self._scout_hit():
            self._rebuild_board()
            return
        self._rebuild_board()

    def _expand(self):
        for eid in list(self.enemy_zones.keys()):
            z = self.enemy_zones[eid]
            if not z:
                continue
            hp = eid in self.power_owners and bool(self.power_owners[eid])
            ec = 2 if hp else 1
            fr = set()
            for cx, cy in z:
                for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                    nb = (cx + dx, cy + dy)
                    if nb in self.wall_set or nb not in self.floor_set:
                        continue
                    if nb in z or nb in self.player_zone:
                        continue
                    if any(
                        nb in oz for oeid, oz in self.enemy_zones.items() if oeid != eid
                    ):
                        continue
                    fr.add(nb)
            px, py = self.player_x, self.player_y
            fr2 = sorted(fr, key=lambda p: abs(p[0] - px) + abs(p[1] - py))
            ad = 0
            for cell in fr2:
                if ad >= ec:
                    break
                if cell in self.player_zone or any(
                    cell in oz for oz in self.enemy_zones.values()
                ):
                    continue
                z.add(cell)
                ad += 1
                if cell in self.power_cells:
                    self.power_cells.discard(cell)
                    self.power_owners.setdefault(eid, set()).add(cell)

    def _bomb_tick(self):
        exp = []
        for b in self.bombs:
            b["timer"] -= 1
            if b["timer"] <= 0:
                exp.append(b)
        for b in exp:
            self.bombs.remove(b)
            bx, by, eid = b["x"], b["y"], b["eid"]
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if abs(dx) + abs(dy) > 2:
                        continue
                    ep = (bx + dx, by + dy)
                    if ep in self.wall_set or ep not in self.floor_set:
                        continue
                    if ep == (self.player_x, self.player_y):
                        continue
                    self.player_zone.discard(ep)
                    self.fortified.pop(ep, None)
                    for oeid in self.enemy_zones:
                        if oeid != eid:
                            self.enemy_zones[oeid].discard(ep)
                    self.enemy_zones.setdefault(eid, set()).add(ep)

    def _move_scouts(self):
        for sc in self.scouts:
            if not self.enemy_zones.get(sc["eid"]):
                continue
            nx, ny = sc["x"] + sc["dx"], sc["y"] + sc["dy"]
            if not self._wk(nx, ny):
                sc["dx"] = -sc["dx"]
                sc["dy"] = -sc["dy"]
                nx, ny = sc["x"] + sc["dx"], sc["y"] + sc["dy"]
                if not self._wk(nx, ny):
                    continue
            sc["x"] = nx
            sc["y"] = ny
            pos = (nx, ny)
            eid = sc["eid"]
            if pos in self.player_zone:
                if pos not in self.fortified and pos != (self.player_x, self.player_y):
                    self.player_zone.discard(pos)
                    self.enemy_zones.setdefault(eid, set()).add(pos)

    def _check_end(self):
        if self.collected_targets == self.targets and len(self.targets) > 0:
            self._lives = self.TOTAL_LIVES
            self.next_level()
            return
        tf = len(self.floor_set)
        if tf > 0:
            te = sum(len(z) for z in self.enemy_zones.values())
            if te / tf >= DOMINATION_THRESHOLD:
                self._lose_life()
                return
        if self.turns_left <= 0:
            self._lose_life()

    def _lose_life(self):
        self._lives -= 1
        self.death_flash = 2
        if self._lives <= 0:
            self._game_over = True
            self._rebuild_board()
            self.lose()
        else:
            self.set_level(self.level_index)

    def _wk(self, x, y):
        return 0 <= x < self.GW and 0 <= y < self.GH and (x, y) not in self.wall_set

    def _build_board(self):
        for s in self.current_level.get_sprites_by_tag("board"):
            self.current_level.remove_sprite(s)
        px = np.array(self._render(), dtype=np.int8)
        b = Sprite(
            pixels=px,
            name="board",
            x=0,
            y=0,
            layer=0,
            collidable=False,
            tags=["board"],
        )
        self.current_level.add_sprite(b)
        self.board_sprite = b

    def _rebuild_board(self):
        px = self._render()
        for y in range(self.GH):
            for x in range(self.GW):
                self.board_sprite.pixels[y, x] = px[y][x]

    def _render(self):
        g = [[BG] * self.GW for _ in range(self.GH)]
        p2 = self._in_phase2()
        for wx, wy in self.wall_set:
            g[wy][wx] = WALL
        for eid, zone in self.enemy_zones.items():
            c = ENEMY_COLORS[eid % len(ENEMY_COLORS)]
            for zx, zy in zone:
                g[zy][zx] = c
        for zx, zy in self.player_zone:
            g[zy][zx] = FORTIFIED_C if (zx, zy) in self.fortified else PLAYER_ZONE
        for tx, ty in self.targets:
            if (tx, ty) in self.collected_targets:
                continue
            if p2:
                g[ty][tx] = TARGET_MARK
            else:
                on_e = any((tx, ty) in z for z in self.enemy_zones.values())
                if not on_e and (tx, ty) not in self.player_zone:
                    g[ty][tx] = TARGET_HINT
        for px, py in self.power_cells:
            g[py][px] = POWER_CELL
        for bx, by in self.beacons:
            g[by][bx] = BEACON_C
        for bomb in self.bombs:
            bx, by = bomb["x"], bomb["y"]
            g[by][bx] = BOMB_WARN if bomb["timer"] <= 3 else BOMB_ARMED
        for sc in self.scouts:
            sx, sy = sc["x"], sc["y"]
            if 0 <= sx < self.GW and 0 <= sy < self.GH:
                g[sy][sx] = SCOUT_CORE
        g[self.player_y][self.player_x] = PLAYER_C
        return g


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
        self._engine = Zn16(seed=seed)
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

    def step(self, action: str) -> StepResult:
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

        level_before = self._engine.level_index

        frame = self._engine.perform_action(
            ActionInput(id=self._ACTION_MAP[action]), raw=True
        )
        state_name = frame.state.name if frame and frame.state else ""
        game_won = state_name == "WIN"
        game_over = state_name == "GAME_OVER"

        total_levels = len(self._engine._levels)
        level_reward = 1.0 / total_levels
        info: dict = {
            "action": action,
            "lives": self._engine._lives,
            "level": self._engine.level_index + 1,
            "turns_left": self._engine.turns_left,
            "max_turns": self._engine.max_turns,
        }

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
        if self._engine.level_index > level_before:
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
        index_grid = self._engine.camera.render(
            self._engine.current_level.get_sprites()
        )
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

    def is_done(self) -> bool:
        return self._done

    def _build_text_observation(self) -> str:
        engine = self._engine
        level_idx = engine.level_index
        total_levels = len(engine._levels)

        lines = []
        lines.append(
            f"Level {level_idx + 1}/{total_levels}"
            f" | Lives: {engine._lives}/{engine.TOTAL_LIVES}"
            f" | Turns: {engine.turns_left}/{engine.max_turns}"
            f" | Turn: {self._total_turns}"
        )
        lines.append("")

        grid = engine._render()
        for row in grid:
            row_str = ""
            for cell in row:
                row_str += CELL_CHARS.get(cell, "?")
            lines.append(row_str)

        lines.append("")
        pz = len(engine.player_zone)
        ez = sum(len(z) for z in engine.enemy_zones.values())
        tf = len(engine.floor_set)
        lines.append(f"Player zone: {pz}/{tf}  Enemy zone: {ez}/{tf}")
        phase = (
            "Phase 2 (collect targets)"
            if engine._in_phase2()
            else "Phase 1 (clear enemies)"
        )
        lines.append(f"Phase: {phase}")
        collected = len(engine.collected_targets)
        total_targets = len(engine.targets)
        lines.append(f"Targets: {collected}/{total_targets}")
        lines.append("")
        lines.append(f"Actions: {', '.join(self._VALID_ACTIONS)}")
        return "\n".join(lines)

    def _build_game_state(
        self, done: bool = False, info: Optional[Dict] = None
    ) -> GameState:
        engine = self._engine
        level_idx = engine.level_index
        total_levels = len(engine._levels)

        pz = len(engine.player_zone)
        ez = sum(len(z) for z in engine.enemy_zones.values())
        tf = len(engine.floor_set)
        collected = len(engine.collected_targets)
        total_targets = len(engine.targets)

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
                "total_levels": total_levels,
                "level_index": level_idx,
                "levels_completed": getattr(engine, "_score", 0),
                "level": level_idx + 1,
                "lives": engine._lives,
                "turns_left": engine.turns_left,
                "max_turns": engine.max_turns,
                "player_zone_size": pz,
                "enemy_zone_size": ez,
                "floor_size": tf,
                "targets_collected": collected,
                "targets_total": total_targets,
                "in_phase2": engine._in_phase2(),
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
