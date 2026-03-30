
import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from arcengine import (
    ActionInput,
    ARCBaseGame,
    Camera,
    GameAction,
    GameState as EngineGameState,
    Level,
    RenderableUserDisplay,
    Sprite,
)

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

C_BG = 0
C_WALL = 5
C_PLAYER = 9
C_GATE_OPEN = 14
C_BULB_UNLIT = 2
C_BULB_LIT = 11

BACKGROUND_COLOR = 0
PADDING_COLOR = 0
SCALE = 3

SHAPE_MAP_3: Dict[int, int] = {
    1: 1, 2: 2, 3: 3, 4: 3, 5: 2, 6: 1,
}
SHAPE_MAP_6: Dict[int, int] = {
    1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6,
}

DICE_COLORS: List[int] = [9, 8, 14, 12]
DICE_COLOR_NAMES: List[str] = ["Blue", "Red", "Green", "Orange"]

DICE_PATTERNS: Dict[int, List[List[int]]] = {
    1: [
        [0, 1, 1, 0, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ],
    2: [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ],
    3: [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0],
    ],
    4: [
        [0, 0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 0, 0],
    ],
    5: [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ],
    6: [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 1, 0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ],
}

MAX_LIVES = 3
BASELINE_ACTIONS = [7, 9, 20, 28, 38, 42]
STEP_LIMITS = [int(4.0 * b) for b in BASELINE_ACTIONS]


class DiceState:
    def __init__(self) -> None:
        self.top = 1
        self.bottom = 6
        self.north = 2
        self.south = 5
        self.east = 3
        self.west = 4

    def reset(self) -> None:
        self.__init__()

    def roll_north(self) -> None:
        t, b, n, s = self.top, self.bottom, self.north, self.south
        self.top = s
        self.north = t
        self.bottom = n
        self.south = b

    def roll_south(self) -> None:
        t, b, n, s = self.top, self.bottom, self.north, self.south
        self.top = n
        self.south = t
        self.bottom = s
        self.north = b

    def roll_east(self) -> None:
        t, b, e, w = self.top, self.bottom, self.east, self.west
        self.top = w
        self.east = t
        self.bottom = e
        self.west = b

    def roll_west(self) -> None:
        t, b, e, w = self.top, self.bottom, self.east, self.west
        self.top = e
        self.west = t
        self.bottom = w
        self.east = b


sprites = {
    "player": Sprite(
        pixels=[
            [C_PLAYER, C_PLAYER, C_PLAYER],
            [C_PLAYER, C_PLAYER, C_PLAYER],
            [C_PLAYER, C_PLAYER, C_PLAYER],
        ],
        name="player",
        visible=True,
        collidable=True,
        tags=["player"],
        layer=3,
    ),
    "wall": Sprite(
        pixels=[
            [C_WALL, C_WALL, C_WALL],
            [C_WALL, C_WALL, C_WALL],
            [C_WALL, C_WALL, C_WALL],
        ],
        name="wall",
        visible=True,
        collidable=True,
        tags=["wall"],
        layer=0,
    ),
    "gate": Sprite(
        pixels=[
            [C_BG, C_BG, C_BG],
            [C_BG, C_BG, C_BG],
            [C_BG, C_BG, C_BG],
        ],
        name="gate",
        visible=True,
        collidable=False,
        tags=["gate"],
        layer=1,
    ),
    "bulb": Sprite(
        pixels=[
            [C_BULB_UNLIT, C_BULB_UNLIT, C_BULB_UNLIT],
            [C_BULB_UNLIT, C_BULB_UNLIT, C_BULB_UNLIT],
            [C_BULB_UNLIT, C_BULB_UNLIT, C_BULB_UNLIT],
        ],
        name="bulb",
        visible=True,
        collidable=False,
        tags=["bulb"],
        layer=1,
    ),
    "changer": Sprite(
        pixels=[
            [9,  14, 14],
            [9,   0,  8],
            [12, 12,  8],
        ],
        name="changer",
        visible=True,
        collidable=False,
        tags=["changer"],
        layer=1,
    ),
}


def _grid(rows: List[str]) -> str:
    w = len(rows[0])
    for i, r in enumerate(rows):
        if len(r) != w:
            raise ValueError(f"Row {i} length {len(r)} != {w}: '{r}'")
    return "".join(rows)



W1, H1 = 9, 9
G1 = _grid([
    "#########",
    "#.......#",
    "#.......#",
    "#....####",
    "#....1*##",
    "#....####",
    "#.P.....#",
    "#.......#",
    "#########",
])

W2, H2 = 9, 9
G2 = _grid([
    "#########",
    "#..###..#",
    "#..#*#..#",
    "#..#2#..#",
    "#.......#",
    "#...C...#",
    "#.......#",
    "#P......#",
    "#########",
])

W3, H3 = 11, 11
G3 = _grid([
    "###########",
    "####...####",
    "##*2.C.3*##",
    "####...####",
    "#.........#",
    "#.........#",
    "#.........#",
    "#.........#",
    "#.........#",
    "#....P....#",
    "###########",
])

W4, H4 = 13, 11
G4 = _grid([
    "#############",
    "#...........#",
    "#..###..###.#",
    "#..#*#..#*#.#",
    "#..#3#..#5#.#",
    "#.....#.....#",
    "#...........#",
    "#.....C.....#",
    "#...........#",
    "#P..........#",
    "#############",
])

W5, H5 = 15, 11
G5 = _grid([
    "###############",
    "#.............#",
    "#..###.###.####",
    "#..#*#.#*#.#*##",
    "#..#1#.#1#.#4##",
    "#.............#",
    "#.............#",
    "#......C......#",
    "#.............#",
    "#P............#",
    "###############",
])

W6, H6 = 17, 13
G6 = _grid([
    "#################",
    "#...............#",
    "#..###..###..####",
    "#..#*#..#*#..#*##",
    "#..#2#..#6#..#4##",
    "#...............#",
    "#...............#",
    "#.......C.......#",
    "#...............#",
    "#...............#",
    "#...............#",
    "#P..............#",
    "#################",
])

LEVEL_DATA = [
    {"grid": G1, "width": W1, "height": H1, "gate_colors": [9],
     "shape_mode": 3},
    {"grid": G2, "width": W2, "height": H2, "gate_colors": [8],
     "shape_mode": 3},
    {"grid": G3, "width": W3, "height": H3, "gate_colors": [9, 8],
     "shape_mode": 3},
    {"grid": G4, "width": W4, "height": H4, "gate_colors": [8, 14],
     "shape_mode": 6},
    {"grid": G5, "width": W5, "height": H5, "gate_colors": [9, 8, 14],
     "shape_mode": 6},
    {"grid": G6, "width": W6, "height": H6, "gate_colors": [9, 8, 14],
     "shape_mode": 6},
]


def _parse_grid(
    grid_str: str,
    width: int,
    height: int,
    level_idx: int,
    gate_colors: List[int],
    shape_mode: int = 6,
) -> Level:
    level_sprites: List[Sprite] = []
    player_x, player_y = 0, 0

    gate_info: List[Dict] = []
    bulb_info: List[Dict] = []
    changer_positions: List[Dict] = []
    wall_positions: List[Dict] = []
    gate_count = 0

    for y in range(height):
        for x in range(width):
            ch = grid_str[y * width + x]

            if ch == "#":
                level_sprites.append(
                    sprites["wall"].clone().set_position(
                        x * SCALE, y * SCALE)
                )
                wall_positions.append({"x": x, "y": y})
            elif ch == "P":
                player_x, player_y = x, y
                level_sprites.append(
                    sprites["player"].clone().set_position(
                        x * SCALE, y * SCALE)
                )
            elif ch.isdigit() and "1" <= ch <= "6":
                value = int(ch)
                color = (
                    gate_colors[gate_count]
                    if gate_count < len(gate_colors)
                    else 9
                )
                gate_sprite = sprites["gate"].clone().set_position(
                    x * SCALE, y * SCALE)
                gate_sprite.color_remap(None, color)
                level_sprites.append(gate_sprite)
                gate_info.append({
                    "x": x, "y": y,
                    "value": value,
                    "color": color,
                    "open": False,
                })
                gate_count += 1
            elif ch == "*":
                bulb_sprite = sprites["bulb"].clone().set_position(
                    x * SCALE, y * SCALE)
                level_sprites.append(bulb_sprite)
                bulb_info.append({
                    "x": x, "y": y,
                    "lit": False,
                })
            elif ch == "C":
                changer_sprite = (
                    sprites["changer"].clone().set_position(
                        x * SCALE, y * SCALE)
                )
                level_sprites.append(changer_sprite)
                changer_positions.append({"x": x, "y": y})

    return Level(
        sprites=level_sprites,
        grid_size=(width * SCALE, height * SCALE),
        data={
            "level_idx": level_idx,
            "logical_width": width,
            "logical_height": height,
            "player_x": player_x,
            "player_y": player_y,
            "gate_info": gate_info,
            "bulb_info": bulb_info,
            "changer_positions": changer_positions,
            "wall_positions": wall_positions,
            "shape_mode": shape_mode,
        },
        name=f"Level {level_idx + 1}",
    )


def _build_levels() -> List[Level]:
    result: List[Level] = []
    for i, ld in enumerate(LEVEL_DATA):
        result.append(
            _parse_grid(
                ld["grid"], ld["width"], ld["height"],
                i, ld["gate_colors"],
                ld.get("shape_mode", 6),
            )
        )
    return result


def _build_level_variant(level_idx: int, variant_idx: int) -> Level:
    ld = LEVEL_DATA[level_idx]
    return _parse_grid(
        ld["grid"], ld["width"], ld["height"],
        level_idx, ld["gate_colors"],
        ld.get("shape_mode", 6),
    )


_CAMERA_SIZES = [
    (W1 * SCALE, H1 * SCALE),
    (W2 * SCALE, H2 * SCALE),
    (W3 * SCALE, H3 * SCALE),
    (W4 * SCALE, H4 * SCALE),
    (W5 * SCALE, H5 * SCALE),
    (W6 * SCALE, H6 * SCALE),
]


class DiceHud(RenderableUserDisplay):
    HUD_TOP = 56
    HUD_BG = 5

    def __init__(self, game: "Dg01") -> None:
        self._g = game

    def _draw_dice_face(
        self,
        frame: np.ndarray,
        face_val: int,
        x0: int,
        y0: int,
        dot_color: int,
        bg_color: int = 4,
    ) -> None:
        y1 = min(y0 + 7, 64)
        x1 = min(x0 + 7, 64)
        frame[y0:y1, x0:x1] = bg_color

        pattern = DICE_PATTERNS.get(face_val)
        if pattern is None:
            return
        for r, row in enumerate(pattern):
            for c, val in enumerate(row):
                if val:
                    py = y0 + r
                    px = x0 + c
                    if 0 <= py < 64 and 0 <= px < 64:
                        frame[py, px] = dot_color

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        g = self._g

        frame[self.HUD_TOP:, :] = self.HUD_BG

        frame[self.HUD_TOP, :] = 3

        top_face = g._dice.top
        display_face = g._shape_map[top_face]
        dice_color = DICE_COLORS[g._dice_color_idx]
        self._draw_dice_face(
            frame, display_face, 2, self.HUD_TOP + 1, 0, dice_color
        )

        gate_info = g._get_adjacent_gate()
        if gate_info is not None:
            gate_val = gate_info["value"]
            gate_color = gate_info["color"]
            self._draw_dice_face(
                frame, gate_val, 12, self.HUD_TOP + 1, 0, gate_color
            )

        num_bulbs = len(g._bulb_info)
        if num_bulbs > 0:
            indicator_y = self.HUD_TOP + 2
            start_x = 22
            gap = 5
            for i, bi in enumerate(g._bulb_info):
                ix = start_x + i * gap
                if ix + 3 > 63:
                    break
                if bi["lit"]:
                    frame[indicator_y:indicator_y + 3, ix:ix + 3] = C_BULB_LIT
                else:
                    frame[
                        indicator_y:indicator_y + 3, ix:ix + 3
                    ] = C_BULB_UNLIT

        num_gates = len(g._gate_info)
        if num_gates > 0:
            gy = self.HUD_TOP + 6
            sx = 22
            gap = 5
            for i, gi in enumerate(g._gate_info):
                ix = sx + i * gap
                if ix + 2 > 63:
                    break
                if gi["open"]:
                    frame[gy:gy + 1, ix:ix + 3] = 14
                else:
                    frame[gy:gy + 1, ix:ix + 3] = gi["color"]

        return frame


class Dg01(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        levels = _build_levels()
        self._variant_idx: int = 0
        self._hud = DiceHud(self)

        self._dice = DiceState()
        self._dice_color_idx: int = 0

        self._player: Optional[Sprite] = None
        self._walls: List[Sprite] = []
        self._wall_set: Set[Tuple[int, int]] = set()
        self._gate_info: List[Dict] = []
        self._gate_sprites: List[Optional[Sprite]] = []
        self._gate_pos_map: Dict[Tuple[int, int], int] = {}
        self._bulb_info: List[Dict] = []
        self._bulb_sprites: List[Optional[Sprite]] = []
        self._bulb_pos_map: Dict[Tuple[int, int], int] = {}
        self._changer_set: Set[Tuple[int, int]] = set()
        self._log_width: int = 0
        self._log_height: int = 0
        self._lives: int = MAX_LIVES
        self._step_count: int = 0
        self._max_steps: int = STEP_LIMITS[0]
        self._preserve_lives: bool = False
        self._history: List[dict] = []

        camera = Camera(
            0, 0,
            _CAMERA_SIZES[0][0], _CAMERA_SIZES[0][1],
            BACKGROUND_COLOR, PADDING_COLOR,
            [self._hud],
        )

        super().__init__(
            "dg01",
            levels,
            camera,
            available_actions=[0,1, 2, 3, 4, 5, 7],
        )

    def on_set_level(self, level: Level) -> None:
        self._variant_idx = 0
        idx = self.current_level.get_data("level_idx")
        cw, ch = _CAMERA_SIZES[idx]
        self.camera.width = cw
        self.camera.height = ch

        self._dice.reset()
        self._dice_color_idx = 0

        sm = self.current_level.get_data("shape_mode")
        self._shape_map = SHAPE_MAP_3 if sm == 3 else SHAPE_MAP_6

        self._player = self.current_level.get_sprites_by_tag("player")[0]
        self._player.color_remap(None, DICE_COLORS[0])

        raw_walls = self.current_level.get_data("wall_positions")
        self._wall_set = {(w["x"], w["y"]) for w in raw_walls}

        self._log_width = self.current_level.get_data("logical_width")
        self._log_height = self.current_level.get_data("logical_height")

        raw_gate_info = self.current_level.get_data("gate_info")
        self._gate_info = [dict(g) for g in raw_gate_info]

        all_gate_sprites = sorted(
            self.current_level.get_sprites_by_tag("gate"),
            key=lambda s: (s.y, s.x),
        )
        self._gate_sprites = []
        self._gate_pos_map = {}
        for i, gi in enumerate(self._gate_info):
            pos = (gi["x"], gi["y"])
            self._gate_pos_map[pos] = i
            matched = None
            for sprite in all_gate_sprites:
                if (sprite.x == pos[0] * SCALE
                        and sprite.y == pos[1] * SCALE):
                    matched = sprite
                    break
            self._gate_sprites.append(matched)

        raw_bulb_info = self.current_level.get_data("bulb_info")
        self._bulb_info = [dict(b) for b in raw_bulb_info]

        all_bulb_sprites = sorted(
            self.current_level.get_sprites_by_tag("bulb"),
            key=lambda s: (s.y, s.x),
        )
        self._bulb_sprites = []
        self._bulb_pos_map = {}
        for i, bi in enumerate(self._bulb_info):
            pos = (bi["x"], bi["y"])
            self._bulb_pos_map[pos] = i
            matched = None
            for sprite in all_bulb_sprites:
                if (sprite.x == pos[0] * SCALE
                        and sprite.y == pos[1] * SCALE):
                    matched = sprite
                    break
            self._bulb_sprites.append(matched)

        raw_changers = self.current_level.get_data("changer_positions")
        if raw_changers:
            self._changer_set = {(c["x"], c["y"]) for c in raw_changers}
        else:
            self._changer_set = set()

        self._history = []
        if self._preserve_lives:
            self._preserve_lives = False
        else:
            self._lives = MAX_LIVES
        self._step_count = 0
        self._max_steps = STEP_LIMITS[idx]

        self._update_gate_visuals()
        self._update_bulb_visuals()

    def _update_gate_visuals(self) -> None:
        for i, gi in enumerate(self._gate_info):
            sprite = (
                self._gate_sprites[i]
                if i < len(self._gate_sprites)
                else None
            )
            if sprite is None:
                continue
            if gi["open"]:
                sprite.color_remap(None, C_GATE_OPEN)
            else:
                sprite.color_remap(None, gi["color"])

    def _update_bulb_visuals(self) -> None:
        for i, bi in enumerate(self._bulb_info):
            sprite = (
                self._bulb_sprites[i]
                if i < len(self._bulb_sprites)
                else None
            )
            if sprite is None:
                continue
            if bi["lit"]:
                sprite.color_remap(None, C_BULB_LIT)
            else:
                sprite.color_remap(None, C_BULB_UNLIT)

    def _update_player_color(self) -> None:
        if self._player is not None:
            self._player.color_remap(
                None, DICE_COLORS[self._dice_color_idx]
            )

    def _is_wall(self, x: int, y: int) -> bool:
        return (x, y) in self._wall_set

    def _gate_at(self, x: int, y: int) -> Optional[int]:
        return self._gate_pos_map.get((x, y))

    def _bulb_at(self, x: int, y: int) -> Optional[int]:
        return self._bulb_pos_map.get((x, y))

    def _player_on_bulb(self) -> Optional[int]:
        if self._player is None:
            return None
        px = self._player.x // SCALE
        py = self._player.y // SCALE
        return self._bulb_pos_map.get((px, py))

    def _get_adjacent_gate(self) -> Optional[Dict]:
        if self._player is None:
            return None
        px = self._player.x // SCALE
        py = self._player.y // SCALE
        best = None
        best_dist = 999
        for gi in self._gate_info:
            if gi["open"]:
                continue
            dx = abs(gi["x"] - px)
            dy = abs(gi["y"] - py)
            dist = dx + dy
            if dist <= 3 and dist < best_dist:
                best = gi
                best_dist = dist
        return best

    def _try_move(self, dx: int, dy: int) -> bool:
        if self._player is None:
            return False

        px = self._player.x // SCALE
        py = self._player.y // SCALE
        nx, ny = px + dx, py + dy

        if nx < 0 or ny < 0 or nx >= self._log_width or ny >= self._log_height:
            return False

        if self._is_wall(nx, ny):
            return False

        gate_idx = self._gate_at(nx, ny)
        if gate_idx is not None:
            gi = self._gate_info[gate_idx]
            if not gi["open"]:
                dice_shape = self._shape_map[self._dice.top]
                if dice_shape != gi["value"]:
                    return False
                if DICE_COLORS[self._dice_color_idx] != gi["color"]:
                    return False

                gi["open"] = True
                self._update_gate_visuals()

        self._player.set_position(nx * SCALE, ny * SCALE)

        if dx == 0 and dy == -1:
            self._dice.roll_north()
        elif dx == 0 and dy == 1:
            self._dice.roll_south()
        elif dx == 1 and dy == 0:
            self._dice.roll_east()
        elif dx == -1 and dy == 0:
            self._dice.roll_west()

        if (nx, ny) in self._changer_set:
            self._dice_color_idx = (
                (self._dice_color_idx + 1) % len(DICE_COLORS)
            )
            self._update_player_color()

        return True

    def _interact(self) -> None:
        if self._player is None:
            return

        bulb_idx = self._player_on_bulb()
        if bulb_idx is not None:
            bi = self._bulb_info[bulb_idx]
            if not bi["lit"]:
                bi["lit"] = True
                self._update_bulb_visuals()

    def _check_win(self) -> bool:
        if len(self._bulb_info) == 0:
            return False
        return all(bi["lit"] for bi in self._bulb_info)

    def _lose_life(self) -> None:
        self._lives -= 1
        if self._lives <= 0:
            self.lose()
        else:
            self._preserve_lives = True
            self.level_reset()

    def _snapshot(self) -> dict:
        return {
            "player": (self._player.x, self._player.y) if self._player else (0, 0),
            "dice": {
                "top": self._dice.top,
                "bottom": self._dice.bottom,
                "north": self._dice.north,
                "south": self._dice.south,
                "east": self._dice.east,
                "west": self._dice.west,
            },
            "dice_color_idx": self._dice_color_idx,
            "gate_info": copy.deepcopy(self._gate_info),
            "bulb_info": copy.deepcopy(self._bulb_info),
            "step_count": self._step_count,
        }

    def _restore_snapshot(self, snapshot: dict) -> None:
        if self._player is not None:
            self._player.set_position(*snapshot["player"])
        self._dice.top = snapshot["dice"]["top"]
        self._dice.bottom = snapshot["dice"]["bottom"]
        self._dice.north = snapshot["dice"]["north"]
        self._dice.south = snapshot["dice"]["south"]
        self._dice.east = snapshot["dice"]["east"]
        self._dice.west = snapshot["dice"]["west"]
        self._dice_color_idx = snapshot["dice_color_idx"]
        self._gate_info = copy.deepcopy(snapshot["gate_info"])
        self._bulb_info = copy.deepcopy(snapshot["bulb_info"])
        self._step_count = snapshot["step_count"]
        self._update_gate_visuals()
        self._update_bulb_visuals()
        self._update_player_color()

    def handle_reset(self) -> None:
        if self._state in (EngineGameState.GAME_OVER, EngineGameState.WIN):
            self._state = EngineGameState.NOT_FINISHED
        self.level_reset()

    def step(self) -> None:
        if self._player is None:
            self.complete_action()
            return

        if self.action.id == GameAction.RESET:
            self.handle_reset()
            self.complete_action()
            return

        if self.action.id == GameAction.ACTION7:
            if self._history:
                self._restore_snapshot(self._history.pop())
            self.complete_action()
            return

        self._history.append(self._snapshot())
        self._step_count += 1
        if self._step_count > self._max_steps:
            self._lose_life()
            self.complete_action()
            return

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

        if self._check_win():
            self.next_level()

        self.complete_action()


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


ACTION_MAP = {
    "reset": 0,
    "up": 1,
    "down": 2,
    "left": 3,
    "right": 4,
    "select": 5,
    "undo": 7,
}


class PuzzleEnvironment:
    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self._engine = Dg01(seed=seed)
        self._total_turns = 0
        self._done = False
        self._last_action_was_reset = False

    def reset(self) -> GameState:
        engine = self._require_engine()
        if self._last_action_was_reset:
            self._engine = Dg01(seed=self._seed)
        elif self._done or engine._state in (EngineGameState.GAME_OVER, EngineGameState.WIN):
            engine._state = EngineGameState.NOT_FINISHED
            engine.level_reset()
        else:
            engine.level_reset()
        self._total_turns = 0
        self._done = False
        self._last_action_was_reset = True
        return self._build_state()

    def get_actions(self) -> list[str]:
        if self._done:
            return ["reset"]
        return ["reset", "up", "down", "left", "right", "select", "undo"]

    def is_done(self) -> bool:
        engine = self._require_engine()
        return self._done or engine._state in (
            EngineGameState.GAME_OVER,
            EngineGameState.WIN,
        )

    def _require_engine(self) -> Dg01:
        if self._engine is None:
            raise RuntimeError("PuzzleEnvironment is closed")
        return self._engine

    def _build_text_observation(self) -> str:
        g = self._require_engine()
        wall_positions = set(g._wall_set)
        gates = {(gi["x"], gi["y"]): ("O" if gi["open"] else str(gi["value"])) for gi in g._gate_info}
        bulbs = {(bi["x"], bi["y"]): ("@" if bi["lit"] else "*") for bi in g._bulb_info}
        rows = []
        px = g._player.x // SCALE if g._player else -1
        py = g._player.y // SCALE if g._player else -1
        for y in range(g._log_height):
            chars = []
            for x in range(g._log_width):
                if (x, y) == (px, py):
                    chars.append("P")
                elif (x, y) in wall_positions:
                    chars.append("#")
                elif (x, y) in gates:
                    chars.append(gates[(x, y)])
                elif (x, y) in bulbs:
                    chars.append(bulbs[(x, y)])
                elif (x, y) in g._changer_set:
                    chars.append("C")
                else:
                    chars.append(".")
            rows.append("".join(chars))
        shape = g._shape_map[g._dice.top]
        status = f"level={g._current_level_index + 1} face={g._dice.top} shape={shape} color={DICE_COLOR_NAMES[g._dice_color_idx]} bulbs={sum(1 for b in g._bulb_info if b['lit'])}/{len(g._bulb_info)} lives={g._lives} steps={max(g._max_steps - g._step_count, 0)}"
        gate_lines = []
        for gi in g._gate_info:
            ci = DICE_COLORS.index(gi["color"]) if gi["color"] in DICE_COLORS else 0
            cn = DICE_COLOR_NAMES[ci]
            st = "open" if gi["open"] else "closed"
            gate_lines.append(f"gate({gi['x']},{gi['y']}):val={gi['value']},color={cn},{st}")
        return "\n".join(rows + [status] + gate_lines)

    def _build_image_observation(self) -> bytes | None:
        return None

    def _build_state(self) -> GameState:
        return GameState(
            text_observation=self._build_text_observation(),
            image_observation=self._build_image_observation(),
            valid_actions=None if self.is_done() else self.get_actions(),
            turn=self._total_turns,
            metadata={
                "total_levels": len(self._engine._levels),
                "level": self._require_engine()._current_level_index + 1,
                "levels_completed": getattr(self._engine, "_score", 0),
                "level_index": self._require_engine()._current_level_index,
                "game_over": getattr(getattr(self._engine, "_state", None), "name", "") == "GAME_OVER",
            },
        )

    def step(self, action: str) -> StepResult:
        action_lower = action.strip().lower()
        if action_lower not in ACTION_MAP:
            return StepResult(self._build_state(), 0.0, False, {"error": f"invalid action: {action}"})
        if action_lower == "reset":
            state = self.reset()
            return StepResult(state, 0.0, False, {"outcome": "reset"})
        if self.is_done():
            self._done = True
            return StepResult(self._build_state(), 0.0, True, {"outcome": "done"})
        self._last_action_was_reset = False
        self._total_turns += 1
        engine = self._require_engine()
        prev_level = engine._current_level_index
        prev_lives = engine._lives
        engine.perform_action(ActionInput(id=GameAction.from_id(ACTION_MAP[action_lower])))
        reward = 0.0
        info: dict = {}
        if engine._state == EngineGameState.GAME_OVER:
            self._done = True
            info["outcome"] = "game_over"
        elif engine._state == EngineGameState.WIN:
            reward = 1.0 / len(engine._levels)
            self._done = True
            info["outcome"] = "game_complete"
        elif engine._current_level_index != prev_level:
            reward = 1.0 / len(engine._levels)
            info["outcome"] = "level_complete"
        elif engine._lives < prev_lives:
            info["outcome"] = "death"
        return StepResult(self._build_state(), reward, self._done, info)

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError(f"Unsupported render mode: {mode}")
        engine = self._require_engine()
        index_grid = engine.camera.render(engine.current_level.get_sprites())
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        for idx, color in enumerate(ARC_PALETTE):
            rgb[index_grid == idx] = color
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

