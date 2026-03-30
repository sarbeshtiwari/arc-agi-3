from dataclasses import dataclass, field
import random
import struct
from typing import Any, Dict, List, Optional, Tuple
import zlib

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
    ToggleableUserDisplay,
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

sprites = {
    "day_bot": Sprite(
        pixels=[[3]],
        name="day_bot",
        visible=True,
        collidable=True,
        layer=2,
        tags=["player", "day_bot"],
    ),
    "night_shadow": Sprite(
        pixels=[[1]],
        name="night_shadow",
        visible=True,
        collidable=True,
        layer=2,
        tags=["player", "night_shadow"],
    ),
    "plate": Sprite(
        pixels=[[14]],
        name="plate",
        visible=True,
        collidable=False,
        layer=0,
        tags=["plate"],
    ),
    "wall": Sprite(
        pixels=[[5]],
        name="wall",
        visible=True,
        collidable=True,
        layer=1,
        tags=["wall"],
    ),
    "wall_day": Sprite(
        pixels=[[5]],
        name="wall_day",
        visible=True,
        collidable=True,
        layer=1,
        tags=["wall", "wall_day"],
    ),
    "wall_night": Sprite(
        pixels=[[5]],
        name="wall_night",
        visible=True,
        collidable=True,
        layer=1,
        tags=["wall", "wall_night"],
    ),
    "eclipse": Sprite(
        pixels=[[15]],
        name="eclipse",
        visible=True,
        collidable=False,
        layer=0,
        tags=["eclipse"],
    ),
    "pit": Sprite(
        pixels=[[6]],
        name="pit",
        visible=True,
        collidable=False,
        layer=0,
        tags=["pit"],
    ),
    "floor": Sprite(
        pixels=[[0]],
        name="floor",
        visible=True,
        collidable=False,
        layer=-1,
        tags=["floor"],
    ),
}


MAX_MOVES_PER_LEVEL = 100

_LIFE_ON_PIX = [
    [3, 3],
]
_LIFE_OFF_PIX = [
    [5, 5],
]


def _make_life_pair(slot: int):
    px = 1 + slot * 3
    py = 1
    on = Sprite(
        pixels=_LIFE_ON_PIX,
        name=f"life_on_{slot}",
        visible=True,
        collidable=False,
        layer=99,
        tags=["life_on"],
    ).set_position(px, py)
    off = Sprite(
        pixels=_LIFE_OFF_PIX,
        name=f"life_off_{slot}",
        visible=True,
        collidable=False,
        layer=99,
        tags=["life_off"],
    ).set_position(px, py)
    return on, off


class MovesBar(RenderableUserDisplay):

    FRAME_SIZE = 64
    BAR_COLOR_REMAINING = 3
    BAR_COLOR_USED = 6

    def __init__(self, max_moves: int, game_ref) -> None:
        self._max = max_moves
        self._game = game_ref

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        used = min(self._game.turn_count, self._max)
        filled = int((self._max - used) / self._max * self.FRAME_SIZE)
        frame[63, :] = self.BAR_COLOR_USED
        frame[63, :filled] = self.BAR_COLOR_REMAINING
        return frame


def build_border(gw, gh):
    out = []
    for x in range(gw):
        out.append(sprites["wall"].clone().set_position(x, 0))
        out.append(sprites["wall"].clone().set_position(x, gh - 1))
    for y in range(1, gh - 1):
        out.append(sprites["wall"].clone().set_position(0, y))
        out.append(sprites["wall"].clone().set_position(gw - 1, y))
    return out


def center_divider(gw, gh, x_pos):
    return [sprites["wall"].clone().set_position(x_pos, y) for y in range(1, gh - 1)]


def walls_at(positions, tag="wall"):
    key = "wall_day" if tag == "day" else "wall_night" if tag == "night" else "wall"
    return [sprites[key].clone().set_position(x, y) for x, y in positions]


def plates_at(positions):
    return [sprites["plate"].clone().set_position(x, y) for x, y in positions]


def eclipse_at(positions):
    return [sprites["eclipse"].clone().set_position(x, y) for x, y in positions]


def pits_at(positions):
    return [sprites["pit"].clone().set_position(x, y) for x, y in positions]


def make_level1():
    gw, gh = 9, 9
    s = build_border(gw, gh)
    s += center_divider(gw, gh, 4)
    s += walls_at([(2, 3)], "day")
    s += pits_at([(3, 2), (5, 5)])
    s += eclipse_at([(3, 5)])
    s += eclipse_at([(6, 2)])
    s += plates_at([(1, 6)])
    s += plates_at([(5, 1)])
    s.append(sprites["day_bot"].clone().set_position(2, 1))
    s.append(sprites["night_shadow"].clone().set_position(6, 6))
    return Level(
        sprites=s,
        grid_size=(gw, gh),
        name="Level 1 - First Detour",
        data={
            "gw": gw,
            "gh": gh,
            "divider_x": 4,
            "has_eclipse": True,
            "plate_order": None,
            "dynamic_divider": False,
            "fake_eclipses": [(6, 2)],
            "max_moves": 100,
        },
    )


def make_level2():
    gw, gh = 16, 16
    s = []

    s += [sprites["wall"].clone().set_position(7, y) for y in range(gh)]
    s += [sprites["wall"].clone().set_position(8, y) for y in range(gh)]

    s += walls_at([(x, 4) for x in range(0, 5)], "day")
    s += walls_at([(x, 10) for x in range(2, 7)], "day")

    s += walls_at([(x, 11) for x in range(11, 16)], "night")
    s += walls_at([(x, 5) for x in range(9, 14)], "night")

    s += pits_at(
        [
            (0, 5),
            (6, 8),
            (6, 11),
            (2, 14),
            (15, 10),
            (9, 7),
            (9, 4),
            (13, 1),
        ]
    )

    s += eclipse_at([(5, 3)])
    s += eclipse_at([(10, 12)])

    s += plates_at([(1, 15)])
    s += plates_at([(14, 0)])

    s.append(sprites["day_bot"].clone().set_position(3, 0))
    s.append(sprites["night_shadow"].clone().set_position(12, 15))

    return Level(
        sprites=s,
        grid_size=(gw, gh),
        name="Level 2 - Binary Maze",
        data={
            "gw": gw,
            "gh": gh,
            "divider_x": 7,
            "has_eclipse": True,
            "plate_order": None,
            "dynamic_divider": False,
            "fake_eclipses": [(10, 12)],
            "max_moves": 150,
        },
    )


def make_level3():
    gw, gh = 21, 21
    s = build_border(gw, gh)
    s += center_divider(gw, gh, 10)

    day_walls = []

    day_walls += [(2, y) for y in range(1, 14)]

    day_walls += [(x, 15) for x in range(1, 7)]

    day_walls += [(6, y) for y in range(6, 14)]

    day_walls += [(x, 4) for x in range(3, 8)]

    day_walls += [(4, y) for y in range(5, 10)]

    day_walls += [(x, 13) for x in range(3, 7)]

    day_walls += [(8, y) for y in range(4, 14)]

    s += walls_at(day_walls, "day")

    night_walls = []

    night_walls += [(18, y) for y in range(7, 20)]

    night_walls += [(x, 5) for x in range(14, 20)]

    night_walls += [(14, y) for y in range(7, 15)]

    night_walls += [(x, 16) for x in range(13, 18)]

    night_walls += [(16, y) for y in range(11, 16)]

    night_walls += [(x, 7) for x in range(14, 18)]

    night_walls += [(12, y) for y in range(7, 17)]

    s += walls_at(night_walls, "night")

    s += eclipse_at([(1, 8)])
    s += eclipse_at([(19, 12)])
    s += eclipse_at([(7, 10)])
    s += eclipse_at([(13, 10)])

    s += pits_at(
        [
            (3, 2),
            (8, 16),
            (5, 12),
            (17, 18),
            (12, 4),
            (15, 8),
        ]
    )

    s += plates_at([(5, 10)])
    s += plates_at([(15, 10)])

    s.append(sprites["day_bot"].clone().set_position(1, 1))
    s.append(sprites["night_shadow"].clone().set_position(19, 19))

    return Level(
        sprites=s,
        grid_size=(gw, gh),
        name="Level 3 - The Spiral of Descent",
        data={
            "gw": gw,
            "gh": gh,
            "divider_x": 10,
            "has_eclipse": True,
            "plate_order": None,
            "dynamic_divider": False,
            "fake_eclipses": [(7, 10), (13, 10)],
            "max_moves": 200,
        },
    )


def make_level4():
    gw, gh = 32, 32
    s = build_border(gw, gh)
    s += center_divider(gw, gh, 16)

    s += walls_at([(3, y) for y in range(1, 7)], "day")
    s += walls_at([(x, 8) for x in range(1, 6)], "day")
    s += walls_at([(7, y) for y in range(1, 7)], "day")
    s += walls_at([(7, y) for y in range(7, 20)], "day")
    s += walls_at([(5, y) for y in range(8, 20)], "day")

    s += walls_at([(28, y) for y in range(25, 31)], "night")
    s += walls_at([(x, 23) for x in range(26, 31)], "night")
    s += walls_at([(24, y) for y in range(25, 31)], "night")
    s += walls_at([(24, y) for y in range(11, 24)], "night")
    s += walls_at([(26, y) for y in range(12, 23)], "night")

    chasm_pits = []
    for x in range(1, 16):
        if x not in (2, 6, 12, 14):
            chasm_pits.append((x, 14))
    for x in range(17, 31):
        if x not in (17, 21, 25, 29):
            chasm_pits.append((x, 14))
    s += pits_at(chasm_pits)

    s += walls_at([(x, 21) for x in range(3, 7)], "day")
    s += walls_at([(3, y) for y in range(22, 31)], "day")

    s += walls_at([(x, 10) for x in range(25, 29)], "night")
    s += walls_at([(28, y) for y in range(2, 10)], "night")
    s += walls_at([(29, 8), (29, 9)], "night")

    s += pits_at(
        [
            (4, 3),
            (4, 5),
            (2, 10),
            (6, 22),
            (1, 25),
            (4, 28),
            (27, 28),
            (27, 26),
            (29, 21),
            (25, 10),
            (30, 5),
            (27, 3),
            (8, 10),
            (10, 16),
            (12, 20),
            (22, 11),
            (20, 15),
            (18, 19),
        ]
    )

    s += eclipse_at([(6, 13)])

    s += eclipse_at([(25, 18)])

    s += eclipse_at([(2, 4)])

    s += plates_at([(2, 30)])
    s += plates_at([(29, 1)])

    s.append(sprites["day_bot"].clone().set_position(2, 1))
    s.append(sprites["night_shadow"].clone().set_position(29, 30))

    return Level(
        sprites=s,
        grid_size=(gw, gh),
        name="Level 4 - The Eclipse Bridge",
        data={
            "gw": gw,
            "gh": gh,
            "divider_x": 16,
            "has_eclipse": True,
            "plate_order": None,
            "dynamic_divider": False,
            "fake_eclipses": [(2, 4)],
            "max_moves": 300,
        },
    )


def make_level5():
    gw, gh = 64, 64
    s = build_border(gw, gh)
    s += center_divider(gw, gh, 32)

    s += walls_at([(9, y) for y in range(1, 14)], "day")
    s += walls_at([(11, y) for y in range(1, 14)], "day")

    s += walls_at([(9, y) for y in range(14, 26)], "day")
    s += walls_at([(11, y) for y in range(14, 26)], "day")

    s += walls_at([(9, y) for y in range(26, 40)], "day")
    s += walls_at([(11, y) for y in range(26, 40)], "day")

    s += walls_at([(9, y) for y in range(40, 54)], "day")
    s += walls_at([(11, y) for y in range(40, 54)], "day")

    s += walls_at([(52, y) for y in range(50, 63)], "night")
    s += walls_at([(54, y) for y in range(50, 63)], "night")

    s += walls_at([(52, y) for y in range(38, 50)], "night")
    s += walls_at([(54, y) for y in range(38, 50)], "night")

    s += walls_at([(52, y) for y in range(24, 38)], "night")
    s += walls_at([(54, y) for y in range(24, 38)], "night")

    s += walls_at([(52, y) for y in range(10, 24)], "night")
    s += walls_at([(54, y) for y in range(10, 24)], "night")

    s += walls_at([(53, 40)], "night")

    s += walls_at([(53, 12)], "night")

    s += eclipse_at([(10, 8)])

    s += eclipse_at([(10, 32)])

    s += eclipse_at([(10, 4)])
    s += eclipse_at([(53, 59)])

    s += pits_at(
        [
            (8, 5),
            (12, 5),
            (8, 15),
            (12, 18),
            (8, 28),
            (12, 30),
            (8, 42),
            (12, 45),
            (8, 50),
            (12, 50),
            (51, 58),
            (55, 58),
            (51, 48),
            (55, 45),
            (51, 35),
            (55, 33),
            (51, 21),
            (55, 18),
            (51, 14),
            (55, 13),
            (20, 10),
            (25, 20),
            (28, 32),
            (22, 45),
            (18, 55),
            (43, 10),
            (38, 22),
            (35, 35),
            (40, 48),
            (45, 58),
        ]
    )

    s += pits_at([(x, 14) for x in range(1, 10)])
    s += pits_at([(x, 14) for x in range(11, 16)])

    s += pits_at([(x, 26) for x in range(1, 10)])
    s += pits_at([(x, 26) for x in range(11, 16)])

    s += pits_at([(x, 49) for x in range(48, 53)])
    s += pits_at([(x, 49) for x in range(54, 63)])

    s += pits_at([(x, 37) for x in range(48, 53)])
    s += pits_at([(x, 37) for x in range(54, 63)])

    plate_pairs = [
        ((10, 12), (53, 51)),
        ((10, 24), (53, 39)),
        ((10, 38), (53, 25)),
        ((10, 52), (53, 11)),
    ]
    for (dx, dy), (nx, ny) in plate_pairs:
        s += plates_at([(dx, dy)])
        s += plates_at([(nx, ny)])

    s.append(sprites["day_bot"].clone().set_position(10, 2))
    s.append(sprites["night_shadow"].clone().set_position(53, 61))

    return Level(
        sprites=s,
        grid_size=(gw, gh),
        name="Level 5 - The Paradox Relay",
        data={
            "gw": gw,
            "gh": gh,
            "divider_x": 32,
            "has_eclipse": True,
            "plate_order": plate_pairs,
            "dynamic_divider": False,
            "divider_shift": 0,
            "fake_eclipses": [(10, 4), (53, 59)],
            "max_moves": 400,
        },
    )


class Bb01(ARCBaseGame):

    MAX_LIVES = 3

    def __init__(self, seed: int = 0) -> None:
        life_pairs = [_make_life_pair(i) for i in range(self.MAX_LIVES)]
        self._lives_hud = ToggleableUserDisplay(life_pairs)
        for i in range(self.MAX_LIVES):
            self._lives_hud.enable(i)

        self._moves_bar = MovesBar(MAX_MOVES_PER_LEVEL, self)
        self._seed = seed
        self._rng = random.Random(seed)
        self._last_pos_idx = -1

        levels = [
            make_level1(),
            make_level2(),
            make_level3(),
            make_level4(),
            make_level5(),
        ]

        super().__init__(
            "bb01",
            levels,
            Camera(x=0, y=0, width=9, height=9, background=0, letter_box=0, interfaces=[self._lives_hud, self._moves_bar]),
            available_actions=[0, 1, 2, 3, 4, 7],
        )
        self.gate_state = True
        self.lives = self.MAX_LIVES
        self._consecutive_resets = 0
        self._actions_since_reset = 0
        self._undo_stack = []
        self._last_level_won = False

    @property
    def level_index(self) -> int:
        return self._current_level_index

    @property
    def eclipsed(self) -> bool:
        return self._eclipsed

    def reseed(self) -> None:
        self._rng = random.Random(self._seed)

    def on_set_level(self, level: Level) -> None:
        self.gw = self.current_level.get_data("gw")
        self.gh = self.current_level.get_data("gh")
        self.divider_x = self.current_level.get_data("divider_x")
        self.has_eclipse = self.current_level.get_data("has_eclipse") or False
        self.plate_order = self.current_level.get_data("plate_order")
        self.dynamic_divider = self.current_level.get_data("dynamic_divider") or False
        self.divider_shift = self.current_level.get_data("divider_shift") or 0
        self.fake_eclipses = self.current_level.get_data("fake_eclipses") or []
        self.max_moves = self.current_level.get_data("max_moves") or MAX_MOVES_PER_LEVEL
        self._moves_bar._max = self.max_moves

        self.day_bot = self.current_level.get_sprites_by_tag("day_bot")[0]
        self.night_shadow = self.current_level.get_sprites_by_tag("night_shadow")[0]

        self._walls = list(self.current_level.get_sprites_by_tag("wall"))
        self._eclipses = list(self.current_level.get_sprites_by_tag("eclipse"))
        self._pits = list(self.current_level.get_sprites_by_tag("pit"))

        self._wall_map = {}
        for w in self._walls:
            tags = w.tags or []
            if "wall_day" in tags:
                self._wall_map[(w.x, w.y)] = "wall_day"
            elif "wall_night" in tags:
                self._wall_map[(w.x, w.y)] = "wall_night"
            else:
                self._wall_map[(w.x, w.y)] = "wall"

        if not hasattr(self, "_level_positions"):
            self._level_positions = {}

        level_idx = self._current_level_index
        if level_idx not in self._level_positions:
            self._level_positions[level_idx] = self._compute_valid_positions()

        positions = self._level_positions[level_idx]
        available = [i for i in range(len(positions)) if i != self._last_pos_idx]
        if not available:
            available = list(range(len(positions)))
        idx = available[self._rng.randrange(0, len(available))]
        self._last_pos_idx = idx
        (dx, dy), (nx, ny) = positions[idx]
        self.day_bot.set_position(dx, dy)
        self.night_shadow.set_position(nx, ny)

        self._init_day_pos = (self.day_bot.x, self.day_bot.y)
        self._init_night_pos = (self.night_shadow.x, self.night_shadow.y)

        if self.plate_order:
            self._plate_pairs = self.plate_order
            self._plate_index = 0
        else:
            plates = self.current_level.get_sprites_by_tag("plate")
            day_plates = [p for p in plates if p.x < self.divider_x]
            night_plates = [p for p in plates if p.x >= self.divider_x]
            if day_plates and night_plates:
                self._plate_pairs = [
                    (
                        (day_plates[0].x, day_plates[0].y),
                        (night_plates[0].x, night_plates[0].y),
                    )
                ]
            else:
                self._plate_pairs = []
            self._plate_index = 0

        self._eclipsed = False
        self.turn_count = 0
        self._divider_sprites = [
            s for s in self._walls if s.x == self.divider_x and 0 < s.y < self.gh - 1
        ]

        self.gate_state = True
        self._undo_stack = []
        self._last_level_won = False

        self.lives = self.MAX_LIVES
        for i in range(self.MAX_LIVES):
            self._lives_hud.enable(i)

        self.camera.width = self.gw
        self.camera.height = self.gh
        self.camera.x = 0
        self.camera.y = 0
        self.camera.replace_interface([self._lives_hud, self._moves_bar])

    def _compute_valid_positions(self):
        occupied = set()
        for w in self._walls:
            occupied.add((w.x, w.y))
        for p in self._pits:
            occupied.add((p.x, p.y))
        for e in self._eclipses:
            occupied.add((e.x, e.y))
        for p in self.current_level.get_sprites_by_tag("plate"):
            occupied.add((p.x, p.y))

        day_cells = []
        night_cells = []
        for y in range(self.gh):
            for x in range(self.gw):
                if (x, y) in occupied:
                    continue
                if x < self.divider_x:
                    day_cells.append((x, y))
                elif x > self.divider_x:
                    night_cells.append((x, y))

        day_indices = self._rng.sample(range(len(day_cells)), min(6, len(day_cells)))
        night_indices = self._rng.sample(range(len(night_cells)), min(6, len(night_cells)))

        pairs = []
        for i in range(min(6, len(day_indices), len(night_indices))):
            pairs.append((day_cells[day_indices[i]], night_cells[night_indices[i]]))

        return pairs

    def _wall_blocks_day(self, x, y):
        wtype = self._wall_map.get((x, y))
        if wtype is None:
            return False
        if wtype == "wall_night":
            return not self.gate_state
        return True

    def _wall_blocks_night(self, x, y):
        wtype = self._wall_map.get((x, y))
        if wtype is None:
            return False
        if wtype == "wall_day":
            return False
        if wtype == "wall_night":
            return self.gate_state
        return True

    def _oob(self, x, y):
        return x < 0 or x >= self.gw or y < 0 or y >= self.gh

    def _on_pit(self, x, y):
        for p in self._pits:
            if p.x == x and p.y == y:
                return True
        return False

    def _on_eclipse(self, x, y):
        if (x, y) in self.fake_eclipses:
            return False
        for e in self._eclipses:
            if e.x == x and e.y == y:
                return True
        return False

    def _try_move_agents(self, dx, dy):
        if self._eclipsed:
            ndx, ndy = dx, dy
            sdx, sdy = dx, dy
            self._eclipsed = False
        else:
            ndx, ndy = dx, dy
            sdx, sdy = -dx, -dy

        nbx, nby = self.day_bot.x + ndx, self.day_bot.y + ndy
        if not self._oob(nbx, nby) and not self._wall_blocks_day(nbx, nby):
            self.day_bot.set_position(nbx, nby)
            if self.has_eclipse and self._on_eclipse(nbx, nby):
                self._eclipsed = True

        nsx, nsy = self.night_shadow.x + sdx, self.night_shadow.y + sdy
        if not self._oob(nsx, nsy) and not self._wall_blocks_night(nsx, nsy):
            self.night_shadow.set_position(nsx, nsy)
            if self.has_eclipse and not self._eclipsed and self._on_eclipse(nsx, nsy):
                self._eclipsed = True

    def _check_win(self):
        if self._plate_index >= len(self._plate_pairs):
            return True
        (dp_x, dp_y), (np_x, np_y) = self._plate_pairs[self._plate_index]
        if (
            self.day_bot.x == dp_x
            and self.day_bot.y == dp_y
            and self.night_shadow.x == np_x
            and self.night_shadow.y == np_y
        ):
            self._plate_index += 1

            if self.dynamic_divider and self.divider_shift and self._divider_sprites:
                self._shift_divider()

            if self._plate_index >= len(self._plate_pairs):
                return True
        return False

    def _shift_divider(self):
        for sp in self._divider_sprites:
            self.current_level.remove_sprite(sp)
        self.divider_x = max(1, self.divider_x - self.divider_shift)
        self._divider_sprites = center_divider(self.gw, self.gh, self.divider_x)
        for sp in self._divider_sprites:
            self.current_level.add_sprite(sp)
        self._walls = list(self.current_level.get_sprites_by_tag("wall"))

    def _restore_level(self) -> None:
        positions = self._level_positions[self._current_level_index]
        available = [i for i in range(len(positions)) if i != self._last_pos_idx]
        if not available:
            available = list(range(len(positions)))
        idx = available[self._rng.randrange(0, len(available))]
        self._last_pos_idx = idx
        (dx, dy), (nx, ny) = positions[idx]
        self.day_bot.set_position(dx, dy)
        self.night_shadow.set_position(nx, ny)
        self._init_day_pos = (dx, dy)
        self._init_night_pos = (nx, ny)
        self._eclipsed = False
        self.turn_count = 0
        self.gate_state = True
        self._undo_stack = []
        if self.plate_order:
            self._plate_pairs = self.plate_order
            self._plate_index = 0
        else:
            plates = self.current_level.get_sprites_by_tag("plate")
            day_plates = [p for p in plates if p.x < self.divider_x]
            night_plates = [p for p in plates if p.x >= self.divider_x]
            if day_plates and night_plates:
                self._plate_pairs = [
                    (
                        (day_plates[0].x, day_plates[0].y),
                        (night_plates[0].x, night_plates[0].y),
                    )
                ]
            else:
                self._plate_pairs = []
            self._plate_index = 0
        self._divider_sprites = [
            s for s in self._walls if s.x == self.divider_x and 0 < s.y < self.gh - 1
        ]

    def _save_state(self):
        self._undo_stack.append({
            "day_pos": (self.day_bot.x, self.day_bot.y),
            "night_pos": (self.night_shadow.x, self.night_shadow.y),
            "gate_state": self.gate_state,
            "turn_count": self.turn_count,
            "actions_since_reset": self._actions_since_reset,
            "eclipsed": self._eclipsed,
            "plate_index": self._plate_index,
        })

    def _restore_from_undo(self):
        if not self._undo_stack:
            return
        state = self._undo_stack.pop()
        self.day_bot.set_position(*state["day_pos"])
        self.night_shadow.set_position(*state["night_pos"])
        self.gate_state = state["gate_state"]
        self.turn_count = state["turn_count"]
        self._actions_since_reset = state["actions_since_reset"]
        self._eclipsed = state["eclipsed"]
        self._plate_index = state["plate_index"]

    def handle_reset(self):
        if self._last_level_won or (self._consecutive_resets >= 1 and self._actions_since_reset == 0):
            self._consecutive_resets = 0
            self._actions_since_reset = 0
            self._last_level_won = False
            self.lives = self.MAX_LIVES
            for i in range(self.MAX_LIVES):
                self._lives_hud.enable(i)
            self.full_reset()
        else:
            self._consecutive_resets += 1
            self._actions_since_reset = 0
            self.lives = self.MAX_LIVES
            for i in range(self.MAX_LIVES):
                self._lives_hud.enable(i)
            self._restore_level()

    def step(self) -> None:
        if self.action.id == GameAction.RESET:
            self.complete_action()
            return

        if self.action.id == GameAction.ACTION7:
            self._restore_from_undo()
            self.turn_count += 1
            self._actions_since_reset += 1
            if self.turn_count >= self.max_moves:
                self._lose_life()
            self.complete_action()
            return

        dx, dy = 0, 0
        if self.action.id == GameAction.ACTION1:
            dx, dy = 0, -1
        elif self.action.id == GameAction.ACTION2:
            dx, dy = 0, 1
        elif self.action.id == GameAction.ACTION3:
            dx, dy = -1, 0
        elif self.action.id == GameAction.ACTION4:
            dx, dy = 1, 0

        if dx != 0 or dy != 0:
            self._save_state()
            self._try_move_agents(dx, dy)
            self.turn_count += 1
            self._actions_since_reset += 1

            if self.turn_count % 3 == 0:
                self.gate_state = not self.gate_state

        if self._on_pit(self.day_bot.x, self.day_bot.y) or self._on_pit(
            self.night_shadow.x, self.night_shadow.y
        ):
            self._lose_life()
            self.complete_action()
            return

        if self._check_win():
            if self._current_level_index >= len(self._levels) - 1:
                self._last_level_won = True
            self.next_level()
            self.complete_action()
            return

        if self.turn_count >= self.max_moves:
            self._lose_life()
            self.complete_action()
            return

        self.complete_action()

    def _lose_life(self) -> None:
        self.lives -= 1
        if self.lives <= 0:
            self.lives = 0
            self.lose()
        else:
            self._restore_level()
            for i in range(self.lives, self.MAX_LIVES):
                self._lives_hud.disable(i)


def _rgb_to_png(rgb: np.ndarray) -> bytes:
    h, w = rgb.shape[:2]
    raw = b""
    for y in range(h):
        raw += b"\x00" + rgb[y].tobytes()
    compressed = zlib.compress(raw)
    def _chunk(tag: bytes, data: bytes) -> bytes:
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", compressed) + _chunk(b"IEND", b"")


ARC_PALETTE = np.array(
    [
        [0, 0, 0],
        [0, 116, 217],
        [255, 65, 54],
        [46, 204, 64],
        [255, 220, 0],
        [170, 170, 170],
        [240, 18, 190],
        [255, 133, 27],
        [127, 219, 255],
        [135, 12, 37],
        [0, 48, 73],
        [106, 76, 48],
        [255, 182, 193],
        [80, 80, 80],
        [50, 205, 50],
        [128, 0, 128],
    ],
    dtype=np.uint8,
)


class PuzzleEnvironment:

    _ACTION_MAP = {
        "up": GameAction.ACTION1,
        "down": GameAction.ACTION2,
        "left": GameAction.ACTION3,
        "right": GameAction.ACTION4,
        "undo": GameAction.ACTION7,
        "reset": GameAction.RESET,
    }

    _VALID_ACTIONS = ["up", "down", "left", "right", "undo", "reset"]

    def __init__(self, seed: int = 0) -> None:
        self._engine = Bb01(seed=seed)
        self.TOTAL_LEVELS: int = len(self._engine._levels)
        self._done = False
        self._total_turns = 0

    def reset(self) -> GameState:
        self._engine.perform_action(ActionInput(id=GameAction.RESET))
        self._done = False
        self._total_turns = 0
        return self._build_game_state()

    def get_actions(self) -> list[str]:
        if self._done:
            return ["reset"]
        return list(self._VALID_ACTIONS)

    def step(self, action: str) -> StepResult:

        if action == "reset":
            state = self.reset()
            return StepResult(
                state=state,
                reward=0.0,
                done=False,
                info={"event": "reset"},
            )

        if action not in self._ACTION_MAP:
            return StepResult(
                state=self._build_game_state(),
                reward=0.0,
                done=self._done,
                info={"error": f"Invalid action: {action}"},
            )

        lives_before = self._engine.lives
        level_before = self._engine.level_index

        frame = self._engine.perform_action(ActionInput(id=self._ACTION_MAP[action]), raw=True)
        self._total_turns += 1

        reward = 0.0
        done = False
        info: dict = {
            "action": action,
            "lives": self._engine.lives,
            "level": self._engine.level_index + 1,
            "turn_count": getattr(self._engine, "turn_count", 0),
            "max_moves": getattr(self._engine, "max_moves", MAX_MOVES_PER_LEVEL),
        }

        level_reward_step = 1.0 / self.TOTAL_LEVELS

        state_name = frame.state.name if frame and frame.state else ""
        game_over = state_name == "GAME_OVER"
        game_won = state_name == "WIN"

        if game_over or self._engine.lives < lives_before:
            if game_over:
                reward = 0.0
                info["event"] = "game_over"
                done = True
            else:
                reward = 0.0
                info["event"] = "life_lost"
        elif game_won:
            reward = level_reward_step
            info["event"] = "game_complete"
            done = True
        elif self._engine.level_index != level_before:
            reward = level_reward_step
            info["event"] = "level_complete"

        self._done = done

        return StepResult(
            state=self._build_game_state(),
            reward=reward,
            done=done,
            info=info,
        )

    def is_done(self) -> bool:
        return self._done

    def _is_won(self) -> bool:
        return getattr(self._engine, "_state", None) and hasattr(self._engine._state, "name") and self._engine._state.name == "WIN"

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError(f"Unsupported render mode: {mode}")
        frame = self._engine.camera.render(self._engine.current_level.get_sprites())
        h, w = frame.shape[:2]
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for idx in range(len(ARC_PALETTE)):
            mask = frame == idx
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
        g = self._engine
        gw = getattr(g, "gw", 9)
        gh = getattr(g, "gh", 9)
        grid = [["." for _ in range(gw)] for _ in range(gh)]

        for w in getattr(g, "_walls", []):
            if 0 <= w.x < gw and 0 <= w.y < gh:
                tags = w.tags or []
                if "wall_day" in tags:
                    grid[w.y][w.x] = "W"
                elif "wall_night" in tags:
                    grid[w.y][w.x] = "w"
                else:
                    grid[w.y][w.x] = "#"

        for p in getattr(g, "_pits", []):
            if 0 <= p.x < gw and 0 <= p.y < gh:
                grid[p.y][p.x] = "X"

        fake_set = set(getattr(g, "fake_eclipses", []))
        for e in getattr(g, "_eclipses", []):
            if 0 <= e.x < gw and 0 <= e.y < gh:
                if (e.x, e.y) in fake_set:
                    grid[e.y][e.x] = "e"
                else:
                    grid[e.y][e.x] = "E"

        for p in g.current_level.get_sprites_by_tag("plate"):
            if 0 <= p.x < gw and 0 <= p.y < gh:
                grid[p.y][p.x] = "P"

        plate_idx = getattr(g, "_plate_index", 0)
        plate_pairs = getattr(g, "_plate_pairs", [])
        if plate_idx < len(plate_pairs):
            (tp_dx, tp_dy), (tp_nx, tp_ny) = plate_pairs[plate_idx]
            if 0 <= tp_dx < gw and 0 <= tp_dy < gh:
                grid[tp_dy][tp_dx] = "*"
            if 0 <= tp_nx < gw and 0 <= tp_ny < gh:
                grid[tp_ny][tp_nx] = "*"

        day = getattr(g, "day_bot", None)
        if day and 0 <= day.x < gw and 0 <= day.y < gh:
            grid[day.y][day.x] = "D"

        night = getattr(g, "night_shadow", None)
        if night and 0 <= night.x < gw and 0 <= night.y < gh:
            grid[night.y][night.x] = "N"

        gate = "open" if getattr(g, "gate_state", True) else "closed"
        lives = getattr(g, "lives", 3)
        turn = getattr(g, "turn_count", 0)
        max_m = getattr(g, "max_moves", MAX_MOVES_PER_LEVEL)
        eclipsed = getattr(g, "_eclipsed", False)
        level_idx = getattr(g, "_current_level_index", 0)

        header = (
            f"Level {level_idx + 1} | Gate {gate} | Lives {lives} | "
            f"Moves {turn}/{max_m} | Eclipsed {eclipsed}"
        )
        if len(plate_pairs) > 1:
            header += f" | Target {plate_idx + 1}/{len(plate_pairs)}"

        return header + "\n" + "\n".join("".join(row) for row in grid)

    def _build_game_state(self) -> GameState:
        valid = None if self._done else self.get_actions()
        img_bytes = _rgb_to_png(self.render())
        return GameState(
            text_observation=self._build_text_observation(),
            image_observation=img_bytes,
            valid_actions=valid,
            turn=self._total_turns,
            metadata={
                "total_levels": self.TOTAL_LEVELS,
                "lives": self._engine.lives,
                "gate_state": self._engine.gate_state,
                "level": self._engine.level_index,
                "eclipsed": self._engine.eclipsed,
                "levels_completed": getattr(self._engine, "_score", 0),
                "level_index": self._engine.level_index,
                "game_over": getattr(getattr(self._engine, "_state", None), "name", "") == "GAME_OVER",
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
