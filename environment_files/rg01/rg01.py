import random
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


C_ELEV = {
    0: 0,
    1: 1,
    2: 3,
    3: 4,
    4: 7,
    5: 11,
}

C_SOURCE_RED = 2
C_SOURCE_CYAN = 8
C_SOURCE_PINK = 12
C_SOURCE_ORANGE = 7

C_WALL = 5
C_DRAIN = 3
C_CURSOR = 14
C_CONFIRM = 4
C_HAZARD = 13

C_HUD_BAR_FULL = 3
C_HUD_BAR_EMPTY = 5
C_HUD_LIFE = 2
C_HUD_LIFE_EMPTY = 5

BACKGROUND_COLOR = 0
PADDING_COLOR = 5

MAX_ELEVATION = 5
MAX_MOVES = 200

sprites: Dict[str, Sprite] = {}


def _make_sprite(name: str, color: int, tags: List[str],
                 layer: int = 0, visible: bool = True,
                 collidable: bool = False) -> Sprite:
    return Sprite(
        pixels=[[color]],
        name=name,
        visible=visible,
        collidable=collidable,
        tags=tags,
        layer=layer,
    )


sprites["terrain"] = _make_sprite("terrain", 0, ["terrain"], layer=0)
sprites["wall"] = _make_sprite("wall", C_WALL, ["wall"], layer=1)
sprites["drain"] = _make_sprite("drain", C_DRAIN, ["drain"], layer=1)
sprites["source"] = _make_sprite("source", C_SOURCE_RED, ["source"], layer=2)
sprites["reservoir"] = _make_sprite("reservoir", C_SOURCE_RED, ["reservoir"], layer=1)
sprites["cursor"] = _make_sprite("cursor", C_CURSOR, ["cursor"], layer=5)
sprites["confirm"] = _make_sprite("confirm", C_CONFIRM, ["confirm"], layer=1)
sprites["hazard"] = _make_sprite("hazard", C_HAZARD, ["hazard"], layer=1)
sprites["flash"] = _make_sprite("flash", 11, [], layer=10, visible=False)

L1_GRID = (
    "111111111X1\n"
    "110R1111111\n"
    "11101111111\n"
    "11100111111\n"
    "11100###111\n"
    "11110000111\n"
    "11110110111\n"
    "11111110111\n"
    "11111110111\n"
    "1111111a111\n"
    "11111111111"
)
L1_DATA = {
    "grid_w": 11, "grid_h": 11,
    "sources": [
        {"char": "R", "color": C_SOURCE_RED, "emission": 3},
    ],
    "reservoirs": [
        {"char": "a", "color": C_SOURCE_RED, "target": 3},
    ],
    "aqueducts": [],
    "max_moves": MAX_MOVES,
}

L2_GRID = (
    "111111111X1\n"
    "11111C11111\n"
    "11111011111\n"
    "1000H000H01\n"
    "11011011011\n"
    "11011111011\n"
    "11011111011\n"
    "11011111011\n"
    "11011111011\n"
    "11a11111b11\n"
    "11111111111"
)
L2_DATA = {
    "grid_w": 11, "grid_h": 11,
    "sources": [
        {"char": "C", "color": C_SOURCE_CYAN, "emission": 4},
    ],
    "reservoirs": [
        {"char": "a", "color": C_SOURCE_CYAN, "target": 2},
        {"char": "b", "color": C_SOURCE_CYAN, "target": 2},
    ],
    "aqueducts": [],
    "max_moves": MAX_MOVES,
}

L3_GRID = (
    "1101111111111X1\n"
    "11R1110C0111K01\n"
    "100011101111011\n"
    "0H1011101110011\n"
    "101010000011011\n"
    "101010101011011\n"
    "101010111011011\n"
    "1H1010111011011\n"
    "101010111011H11\n"
    "1a1010111011d11\n"
    "111010111011111\n"
    "111010111011111\n"
    "111H10111011111\n"
    "111D1b111c11111\n"
    "111111111111111"
)
L3_DATA = {
    "grid_w": 15, "grid_h": 15,
    "sources": [
        {"char": "R", "color": C_SOURCE_RED, "emission": 6},
        {"char": "C", "color": C_SOURCE_CYAN, "emission": 4},
        {"char": "K", "color": C_SOURCE_PINK, "emission": 4},
    ],
    "reservoirs": [
        {"char": "a", "color": C_SOURCE_RED, "target": 3},
        {"char": "b", "color": C_SOURCE_CYAN, "target": 2},
        {"char": "c", "color": C_SOURCE_CYAN, "target": 2},
        {"char": "d", "color": C_SOURCE_PINK, "target": 4},
    ],
    "aqueducts": [],
    "max_moves": MAX_MOVES,
}


L4_GRID = (
    "111111111111111X1\n"
    "10R01110C01110O01\n"
    "10H00111011110H01\n"
    "10110010H00010101\n"
    "10010110010110001\n"
    "10110110110110101\n"
    "10110110110110101\n"
    "1a11b11c11d11e1f1\n"
    "11111111011111111\n"
    "11111110K01111111\n"
    "11111111011111111\n"
    "11111111H11111111\n"
    "11111111011111111\n"
    "11111111011111111\n"
    "11111111011111111\n"
    "11111111g11111111\n"
    "11111111111111111"
)
L4_DATA = {
    "grid_w": 17, "grid_h": 17,
    "sources": [
        {"char": "R", "color": C_SOURCE_RED, "emission": 8},
        {"char": "C", "color": C_SOURCE_CYAN, "emission": 6},
        {"char": "O", "color": C_SOURCE_ORANGE, "emission": 6},
        {"char": "K", "color": C_SOURCE_PINK, "emission": 4},
    ],
    "reservoirs": [
        {"char": "a", "color": C_SOURCE_RED, "target": 4},
        {"char": "b", "color": C_SOURCE_RED, "target": 4},
        {"char": "c", "color": C_SOURCE_CYAN, "target": 3},
        {"char": "d", "color": C_SOURCE_CYAN, "target": 3},
        {"char": "e", "color": C_SOURCE_ORANGE, "target": 3},
        {"char": "f", "color": C_SOURCE_ORANGE, "target": 3},
        {"char": "g", "color": C_SOURCE_PINK, "target": 4},
    ],
    "aqueducts": [],
    "max_moves": MAX_MOVES,
}

ALL_LEVEL_GRIDS = [L1_GRID, L2_GRID, L3_GRID, L4_GRID]
ALL_LEVEL_DATA = [L1_DATA, L2_DATA, L3_DATA, L4_DATA]


def _parse_grid(grid_str: str, data: Dict) -> Level:
    rows = grid_str.split("\n")
    grid_h = len(rows)
    grid_w = max(len(row) for row in rows)

    level_sprites: List[Sprite] = []

    source_chars = {s["char"]: s for s in data.get("sources", [])}
    reservoir_chars = {r["char"]: r for r in data.get("reservoirs", [])}

    terrain_positions: List[Tuple[int, int, int]] = []
    source_positions: List[Dict] = []
    reservoir_positions: List[Dict] = []
    wall_positions: List[Tuple[int, int]] = []
    drain_positions: List[Tuple[int, int]] = []
    hazard_positions: List[Tuple[int, int]] = []
    confirm_pos: Optional[Tuple[int, int]] = None

    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch == '#':
                s = sprites["wall"].clone().set_position(x, y)
                level_sprites.append(s)
                wall_positions.append((x, y))
            elif ch == 'D':
                s = sprites["drain"].clone().set_position(x, y)
                level_sprites.append(s)
                drain_positions.append((x, y))
            elif ch == 'X':
                s = sprites["confirm"].clone().set_position(x, y)
                level_sprites.append(s)
                confirm_pos = (x, y)
            elif ch in source_chars:
                src_info = source_chars[ch]
                s = sprites["source"].clone().set_position(x, y)
                s.color_remap(None, src_info["color"])
                level_sprites.append(s)
                source_positions.append({
                    "x": x, "y": y,
                    "color": src_info["color"],
                    "emission": src_info["emission"],
                })
            elif ch in reservoir_chars:
                res_info = reservoir_chars[ch]
                s = sprites["reservoir"].clone().set_position(x, y)
                s.color_remap(None, res_info["color"])
                level_sprites.append(s)
                reservoir_positions.append({
                    "x": x, "y": y,
                    "color": res_info["color"],
                    "target": res_info["target"],
                })
            elif ch == 'H':
                s = sprites["hazard"].clone().set_position(x, y)
                level_sprites.append(s)
                hazard_positions.append((x, y))
                terrain_positions.append((x, y, 0))
            else:
                if ch.isdigit() and 0 <= int(ch) <= 5:
                    elev = int(ch)
                else:
                    elev = 0
                color = C_ELEV[elev]
                s = sprites["terrain"].clone().set_position(x, y)
                s.color_remap(None, color)
                level_sprites.append(s)
                terrain_positions.append((x, y, elev))

    cursor_sprite = sprites["cursor"].clone().set_position(0, 0)
    level_sprites.append(cursor_sprite)

    level_data = dict(data)
    level_data["terrain_positions"] = terrain_positions
    level_data["source_positions"] = source_positions
    level_data["reservoir_positions"] = reservoir_positions
    level_data["wall_positions"] = wall_positions
    level_data["drain_positions"] = drain_positions
    level_data["hazard_positions"] = hazard_positions
    level_data["confirm_pos"] = confirm_pos

    return Level(
        sprites=level_sprites,
        grid_size=(grid_w, grid_h),
        data=level_data,
    )


def _build_levels() -> List[Level]:
    return [
        _parse_grid(grid, data)
        for grid, data in zip(ALL_LEVEL_GRIDS, ALL_LEVEL_DATA)
    ]


_CAMERA_SIZES = [
    (11, 11),
    (11, 11),
    (15, 15),
    (17, 17),
]


class FlowHUD(RenderableUserDisplay):

    BAR_Y = 61
    BAR_X = 4
    BAR_WIDTH = 42

    def __init__(self, game: "Rg01") -> None:
        self._game = game

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        max_m = self._game._max_moves
        rem = self._game._moves_remaining
        filled = int(self.BAR_WIDTH * rem / max_m) if max_m > 0 else 0

        for i in range(self.BAR_WIDTH):
            color = C_HUD_BAR_FULL if i < filled else C_HUD_BAR_EMPTY
            frame[self.BAR_Y:self.BAR_Y + 2, self.BAR_X + i] = color

        for i in range(3):
            x = 52 + i * 4
            color = C_HUD_LIFE if self._game._lives > i else C_HUD_LIFE_EMPTY
            frame[self.BAR_Y:self.BAR_Y + 2, x:x + 2] = color

        return frame


def _simulate_flow(
    grid_w: int,
    grid_h: int,
    elevation: Dict[Tuple[int, int], int],
    sources: List[Dict],
    walls: Set[Tuple[int, int]],
    drains: Set[Tuple[int, int]],
    reservoir_set: Set[Tuple[int, int]],
    aqueducts: List[Dict],
) -> Tuple[Dict[Tuple[int, int], int], Dict[Tuple[int, int], Set[int]]]:
    reservoir_fill: Dict[Tuple[int, int], int] = {}
    reservoir_source_colors: Dict[Tuple[int, int], Set[int]] = {}

    aq_lookup: Dict[Tuple[int, int], Dict] = {}
    for aq in aqueducts:
        aq_lookup[(aq["input_x"], aq["input_y"])] = aq

    def _get_elev(x: int, y: int) -> int:
        if (x, y) in reservoir_set:
            return -1
        if (x, y) in drains:
            return -1
        if (x, y) in aq_lookup:
            return -1
        return elevation.get((x, y), 0)

    for src in sources:
        src_x, src_y = src["x"], src["y"]
        src_color = src["color"]
        emission = src["emission"]

        flow_queue: List[Tuple[int, int, int, int]] = []
        flow_queue.append((_get_elev(src_x, src_y), src_x, src_y, emission))
        visited: Set[Tuple[int, int]] = {(src_x, src_y)}

        while flow_queue:
            flow_queue.sort(key=lambda t: -t[0])
            _, cx, cy, volume = flow_queue.pop(0)

            if volume <= 0:
                continue

            if (cx, cy) in reservoir_set and (cx, cy) != (src_x, src_y):
                reservoir_fill[(cx, cy)] = (
                    reservoir_fill.get((cx, cy), 0) + volume
                )
                if (cx, cy) not in reservoir_source_colors:
                    reservoir_source_colors[(cx, cy)] = set()
                reservoir_source_colors[(cx, cy)].add(src_color)
                continue

            if (cx, cy) in drains and (cx, cy) != (src_x, src_y):
                continue

            if (cx, cy) in aq_lookup and (cx, cy) != (src_x, src_y):
                aq = aq_lookup[(cx, cy)]
                ox, oy = aq["output_x"], aq["output_y"]
                if (ox, oy) not in visited:
                    visited.add((ox, oy))
                    flow_queue.append((_get_elev(ox, oy), ox, oy, volume))
                continue

            current_elev = _get_elev(cx, cy)
            valid_neighbors: List[Tuple[int, int, int]] = []

            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nx, ny = cx + dx, cy + dy
                if nx < 0 or nx >= grid_w or ny < 0 or ny >= grid_h:
                    continue
                if (nx, ny) in walls:
                    continue
                if (nx, ny) in visited:
                    continue
                n_elev = _get_elev(nx, ny)
                if n_elev <= current_elev:
                    valid_neighbors.append((nx, ny, n_elev))

            if not valid_neighbors:
                continue

            min_elev = min(n[2] for n in valid_neighbors)
            lowest = [(n[0], n[1]) for n in valid_neighbors if n[2] == min_elev]

            share = volume // len(lowest)
            remainder = volume % len(lowest)

            for i, (nx, ny) in enumerate(lowest):
                v = share + (1 if i < remainder else 0)
                if v > 0:
                    visited.add((nx, ny))
                    flow_queue.append((_get_elev(nx, ny), nx, ny, v))

    return reservoir_fill, reservoir_source_colors


class Rg01(ARCBaseGame):

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)
        levels = _build_levels()
        self._hud = FlowHUD(self)
        self._lives: int = 3
        self._flash_active: bool = False
        self._max_moves: int = MAX_MOVES
        self._moves_remaining: int = MAX_MOVES
        self._reservoir_fill: Dict[Tuple[int, int], int] = {}
        self._reservoir_contaminated: Dict[Tuple[int, int], bool] = {}
        self._reservoir_data: List[Dict] = []
        self._history: List[Dict] = []
        self._action_count: int = 0

        camera = Camera(
            0, 0,
            _CAMERA_SIZES[0][0], _CAMERA_SIZES[0][1],
            BACKGROUND_COLOR, PADDING_COLOR,
            [self._hud],
        )
        super().__init__(
            game_id="rg01",
            levels=levels,
            camera=camera,
            available_actions=[0, 1, 2, 3, 4, 5, 7],
        )

    def on_set_level(self, level: Level) -> None:
        idx = min(self._current_level_index, len(_CAMERA_SIZES) - 1)
        cw, ch = _CAMERA_SIZES[idx]
        self.camera.width = cw
        self.camera.height = ch

        self._grid_w: int = self.current_level.get_data("grid_w")
        self._grid_h: int = self.current_level.get_data("grid_h")
        self._max_moves = self.current_level.get_data("max_moves") or MAX_MOVES
        self._moves_remaining = self._max_moves
        self._lives = 3

        terrain_data = self.current_level.get_data("terrain_positions") or []
        self._elevation: Dict[Tuple[int, int], int] = {}
        self._terrain_sprites: Dict[Tuple[int, int], Sprite] = {}
        self._initial_elevation: Dict[Tuple[int, int], int] = {}

        for tx, ty, elev in terrain_data:
            self._elevation[(tx, ty)] = elev
            self._initial_elevation[(tx, ty)] = elev

        terrain_list = self.current_level.get_sprites_by_tag("terrain")
        for s in terrain_list:
            self._terrain_sprites[(s.x, s.y)] = s

        self._source_data: List[Dict] = (
            self.current_level.get_data("source_positions") or []
        )

        self._reservoir_data = (
            self.current_level.get_data("reservoir_positions") or []
        )
        self._reservoir_colors: Dict[Tuple[int, int], int] = {}
        for rd in self._reservoir_data:
            self._reservoir_colors[(rd["x"], rd["y"])] = rd["color"]

        wall_data = self.current_level.get_data("wall_positions") or []
        self._walls: Set[Tuple[int, int]] = set()
        for wx, wy in wall_data:
            self._walls.add((wx, wy))

        drain_data = self.current_level.get_data("drain_positions") or []
        self._drains: Set[Tuple[int, int]] = set()
        for dx, dy in drain_data:
            self._drains.add((dx, dy))

        hazard_data = self.current_level.get_data("hazard_positions") or []
        self._hazards: Set[Tuple[int, int]] = set()
        for hx, hy in hazard_data:
            self._hazards.add((hx, hy))

        self._reservoir_set: Set[Tuple[int, int]] = set()
        for rd in self._reservoir_data:
            self._reservoir_set.add((rd["x"], rd["y"]))

        self._aqueduct_data: List[Dict] = (
            self.current_level.get_data("aqueduct_positions") or []
        )

        self._confirm_pos: Optional[Tuple[int, int]] = (
            self.current_level.get_data("confirm_pos")
        )

        self._cursor_sprite: Sprite = (
            self.current_level.get_sprites_by_tag("cursor")[0]
        )
        rx, ry = self._random_cursor_position()
        self._cursor_x: int = rx
        self._cursor_y: int = ry
        self._cursor_sprite.set_position(rx, ry)

        self._flash_sprite: Sprite = sprites["flash"].clone()
        self.current_level.add_sprite(self._flash_sprite)
        self._flash_sprite.set_visible(False)
        self._flash_active = False

        self._reservoir_sprites: Dict[Tuple[int, int], Sprite] = {}
        res_sprite_list = self.current_level.get_sprites_by_tag("reservoir")
        for s in res_sprite_list:
            self._reservoir_sprites[(s.x, s.y)] = s

        self._history = []
        self._action_count = 0

        self._run_flow_simulation()

    def _random_cursor_position(self) -> Tuple[int, int]:
        source_positions = set()
        for sd in self._source_data:
            source_positions.add((sd["x"], sd["y"]))
        valid = [
            (x, y) for (x, y) in self._elevation.keys()
            if (x, y) not in self._walls
            and (x, y) not in self._drains
            and (x, y) not in self._reservoir_set
            and (x, y) not in self._hazards
            and (x, y) not in source_positions
            and (self._confirm_pos is None or (x, y) != self._confirm_pos)
        ]
        if not valid:
            return (0, 0)
        return self._rng.choice(valid)

    def _save_state(self) -> None:
        self._history.append({
            "cursor_x": self._cursor_x,
            "cursor_y": self._cursor_y,
            "elevation": dict(self._elevation),
        })

    def _restore_from_undo(self) -> None:
        if not self._history:
            return
        state = self._history.pop()
        self._cursor_x = state["cursor_x"]
        self._cursor_y = state["cursor_y"]
        self._elevation = state["elevation"]
        for pos, elev in self._elevation.items():
            s = self._terrain_sprites.get(pos)
            if s:
                s.color_remap(None, C_ELEV[elev])
        self._run_flow_simulation()
        self._cursor_sprite.set_position(self._cursor_x, self._cursor_y)

    def handle_reset(self) -> None:
        if self._action_count == 0:
            self.full_reset()
        else:
            for pos, elev in self._initial_elevation.items():
                self._elevation[pos] = elev
                s = self._terrain_sprites.get(pos)
                if s:
                    s.color_remap(None, C_ELEV[elev])
            self._moves_remaining = self._max_moves
            self._lives = 3
            rx, ry = self._random_cursor_position()
            self._cursor_x = rx
            self._cursor_y = ry
            self._cursor_sprite.set_position(rx, ry)
            self._history = []
            self._action_count = 0
            self._run_flow_simulation()

    def _run_flow_simulation(self) -> None:
        reservoir_fill_result, reservoir_source_colors = _simulate_flow(
            self._grid_w,
            self._grid_h,
            self._elevation,
            self._source_data,
            self._walls,
            self._drains,
            self._reservoir_set,
            self._aqueduct_data,
        )

        self._reservoir_fill = reservoir_fill_result

        self._reservoir_contaminated = {}
        for pos, color_set in reservoir_source_colors.items():
            expected = self._reservoir_colors.get(pos)
            if expected is not None:
                if any(c != expected for c in color_set):
                    self._reservoir_contaminated[pos] = True
                else:
                    self._reservoir_contaminated[pos] = False

        for rd in self._reservoir_data:
            rx, ry = rd["x"], rd["y"]
            target = rd["target"]
            current = self._reservoir_fill.get((rx, ry), 0)
            contaminated = self._reservoir_contaminated.get((rx, ry), False)

            s = self._reservoir_sprites.get((rx, ry))
            if s is None:
                continue

            if contaminated or current > target:
                s.color_remap(None, 2)
            elif current == target:
                s.color_remap(None, 14)
            else:
                s.color_remap(None, rd["color"])

    def _raise_terrain(self, x: int, y: int) -> bool:
        if (x, y) not in self._elevation:
            return False
        if (x, y) in self._walls:
            return False
        if (x, y) in self._drains:
            return False
        if (x, y) in self._reservoir_set:
            return False
        for sd in self._source_data:
            if sd["x"] == x and sd["y"] == y:
                return False
        if self._confirm_pos and (x, y) == self._confirm_pos:
            return False

        old_elev = self._elevation[(x, y)]
        new_elev = (old_elev + 1) % (MAX_ELEVATION + 1)
        self._elevation[(x, y)] = new_elev

        s = self._terrain_sprites.get((x, y))
        if s:
            s.color_remap(None, C_ELEV[new_elev])

        self._run_flow_simulation()
        return True

    def _check_win(self) -> bool:
        for rd in self._reservoir_data:
            rx, ry = rd["x"], rd["y"]
            target = rd["target"]
            current = self._reservoir_fill.get((rx, ry), 0)
            contaminated = self._reservoir_contaminated.get((rx, ry), False)
            if current != target:
                return False
            if contaminated:
                return False
        return True

    def _attempt_confirm(self) -> bool:
        if self._check_win():
            return True
        has_overflow = False
        has_contamination = False
        for rd in self._reservoir_data:
            rx, ry = rd["x"], rd["y"]
            target = rd["target"]
            current = self._reservoir_fill.get((rx, ry), 0)
            contaminated = self._reservoir_contaminated.get((rx, ry), False)
            if current > target:
                has_overflow = True
            if contaminated:
                has_contamination = True

        if has_overflow or has_contamination:
            self._trigger_life_loss()
            return False
        else:
            return False

    def _reset_terrain(self) -> None:
        for pos, elev in self._initial_elevation.items():
            self._elevation[pos] = elev
            s = self._terrain_sprites.get(pos)
            if s:
                s.color_remap(None, C_ELEV[elev])
        self._run_flow_simulation()

    def _trigger_life_loss(self) -> bool:
        self._lives -= 1
        if self._lives <= 0:
            self.lose()
            self.complete_action()
            return True

        self._reset_terrain()
        self._moves_remaining = self._max_moves
        rx, ry = self._random_cursor_position()
        self._cursor_x = rx
        self._cursor_y = ry
        self._cursor_sprite.set_position(rx, ry)
        self._history = []
        return False

    def step(self) -> None:
        action = self.action.id

        if action == GameAction.RESET:
            self.complete_action()
            return

        self._action_count += 1

        if action == GameAction.ACTION7:
            self._restore_from_undo()
            self._moves_remaining -= 1
            if self._moves_remaining <= 0:
                if self._trigger_life_loss():
                    return
            self.complete_action()
            return

        if action in (GameAction.ACTION1, GameAction.ACTION2,
                      GameAction.ACTION3, GameAction.ACTION4):
            dx, dy = 0, 0
            if action == GameAction.ACTION1:
                dy = -1
            elif action == GameAction.ACTION2:
                dy = 1
            elif action == GameAction.ACTION3:
                dx = -1
            elif action == GameAction.ACTION4:
                dx = 1

            nx = self._cursor_x + dx
            ny = self._cursor_y + dy

            if 0 <= nx < self._grid_w and 0 <= ny < self._grid_h:
                self._save_state()
                self._cursor_x = nx
                self._cursor_y = ny
                self._cursor_sprite.set_position(nx, ny)
                self._moves_remaining -= 1
                if (nx, ny) in self._hazards:
                    if self._trigger_life_loss():
                        return
                elif self._moves_remaining <= 0:
                    if self._trigger_life_loss():
                        return

            self.complete_action()
            return

        if action == GameAction.ACTION5:
            cx, cy = self._cursor_x, self._cursor_y

            if self._confirm_pos and (cx, cy) == self._confirm_pos:
                self._save_state()
                self._moves_remaining -= 1
                if self._attempt_confirm():
                    self.next_level()
                    self.complete_action()
                    return
                if self._moves_remaining <= 0:
                    if self._trigger_life_loss():
                        return
                    self.complete_action()
                    return
                self.complete_action()
                return

            self._save_state()
            if self._raise_terrain(cx, cy):
                self._moves_remaining -= 1
                if self._moves_remaining <= 0:
                    if self._trigger_life_loss():
                        return
                    self.complete_action()
                    return

            self.complete_action()
            return

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
        self._engine = Rg01(seed=seed)
        self._total_turns = 0
        self._done = False
        self._game_won = False
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
            raise ValueError(
                f"Invalid action '{action}'. Must be one of {list(self._ACTION_MAP.keys())}"
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

    def _build_game_state(self, done: bool = False) -> GameState:
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
                "game_over": getattr(getattr(self._engine, "_state", None), "name", "")
                == "GAME_OVER",
                "done": done,
                "info": {},
            },
        )

    def _build_text_observation(self) -> str:
        e = self._engine
        return (
            f"Level {e.level_index + 1}/{len(e._levels)} "
            f"Moves: {e._moves_remaining}/{e._max_moves} "
            f"Lives: {e._lives}/3 "
            f"Cursor: ({e._cursor_x},{e._cursor_y})"
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
        game_seed = int(self.np_random.integers(0, 2**31))
        self._env = PuzzleEnvironment(seed=game_seed)
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
