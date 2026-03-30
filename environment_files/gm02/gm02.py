import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

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


C_BG = 5
C_FLOOR = 0
C_PLAYER = 9
C_CARRY = 11
C_LAVA = 8
C_HOUSE = 14
C_GOLD = 11
C_TILE = 3
C_SOIL = 12
C_ROCK = 4
C_CRACK = 2
C_KEY = 15
C_DOOR = 15
C_BOULDER = 1
C_WATER = 10
C_HUD_OFF = 3

TILE_SIZE = 2
CAM_SIZE = 64

MAX_LIVES = 3
STEP_MULTIPLIER = 10.8

STEP_LIMITS = [
    math.ceil(STEP_MULTIPLIER * 59),
    math.ceil(STEP_MULTIPLIER * 63),
    math.ceil(STEP_MULTIPLIER * 64),
    math.ceil(STEP_MULTIPLIER * 40),
    math.ceil(STEP_MULTIPLIER * 56),
]

CELL_EMPTY = 0
CELL_SOIL = 1
CELL_TILE = 2
CELL_CRACKED = 3
CELL_ROCK = 4
CELL_LAVA = 5
CELL_GOLD = 6
CELL_HOUSE = 7
CELL_KEY = 8
CELL_DOOR = 9
CELL_BOULDER = 10
CELL_WATER = 11

sprites: Dict[str, Sprite] = {
    "player": Sprite(
        pixels=[[C_PLAYER, C_PLAYER], [C_PLAYER, C_PLAYER]],
        name="player",
        visible=True,
        collidable=False,
        tags=["player"],
        layer=3,
    ),
    "soil": Sprite(
        pixels=[[C_SOIL, C_SOIL], [C_SOIL, C_SOIL]],
        name="soil",
        visible=True,
        collidable=False,
        tags=["terrain"],
        layer=1,
    ),
    "tile": Sprite(
        pixels=[[C_TILE, C_TILE], [C_TILE, C_TILE]],
        name="tile",
        visible=True,
        collidable=False,
        tags=["terrain"],
        layer=1,
    ),
    "cracked": Sprite(
        pixels=[[C_TILE, C_CRACK], [C_CRACK, C_TILE]],
        name="cracked",
        visible=True,
        collidable=False,
        tags=["terrain"],
        layer=1,
    ),
    "rock": Sprite(
        pixels=[[C_ROCK, C_ROCK], [C_ROCK, C_ROCK]],
        name="rock",
        visible=True,
        collidable=False,
        tags=["terrain"],
        layer=1,
    ),
    "lava": Sprite(
        pixels=[[C_LAVA, C_LAVA], [C_LAVA, C_LAVA]],
        name="lava",
        visible=True,
        collidable=False,
        tags=["terrain"],
        layer=1,
    ),
    "gold": Sprite(
        pixels=[[C_GOLD, C_GOLD], [C_GOLD, C_GOLD]],
        name="gold",
        visible=True,
        collidable=False,
        tags=["gold"],
        layer=2,
    ),
    "house": Sprite(
        pixels=[[C_HOUSE, C_HOUSE], [C_HOUSE, C_HOUSE]],
        name="house",
        visible=True,
        collidable=False,
        tags=["house"],
        layer=0,
    ),
    "key": Sprite(
        pixels=[[C_KEY, C_KEY], [C_KEY, C_KEY]],
        name="key",
        visible=True,
        collidable=False,
        tags=["key"],
        layer=2,
    ),
    "door": Sprite(
        pixels=[[C_DOOR, C_DOOR], [C_DOOR, C_DOOR]],
        name="door",
        visible=True,
        collidable=False,
        tags=["terrain", "door"],
        layer=1,
    ),
    "boulder": Sprite(
        pixels=[[C_BOULDER, C_BOULDER], [C_BOULDER, C_BOULDER]],
        name="boulder",
        visible=True,
        collidable=False,
        tags=["terrain", "boulder"],
        layer=1,
    ),
    "water": Sprite(
        pixels=[[C_WATER, C_WATER], [C_WATER, C_WATER]],
        name="water",
        visible=True,
        collidable=False,
        tags=["terrain"],
        layer=1,
    ),
}

DIR_UP = (0, -1)
DIR_DOWN = (0, 1)
DIR_LEFT = (-1, 0)
DIR_RIGHT = (1, 0)


LEVEL_1_GRID = (
    "################\n"
    "#H.............#\n"
    "#S##S#S.S###S.S#\n"
    "#SS#S#S#SSSSS#S#\n"
    "#S#SS#S#S#####S#\n"
    "###T############\n"
    "#TTTT#TTTTTTT#T#\n"
    "#T##T#T#####T#T#\n"
    "#TT#TTTTTTT#TTT#\n"
    "#T##L#T###L#T#T#\n"
    "############T###\n"
    "#T#TTTTSSSLSS#S#\n"
    "#S#S########S#S#\n"
    "#S#SSSSSSSS#SSS#\n"
    "#SSS######SSLGL#\n"
    "################"
)

LEVEL_2_GRID = (
    "################\n"
    "#H.............#\n"
    "#T##T#T#T#T###T#\n"
    "#TT#T#T#TSSSSSS#\n"
    "######S#########\n"
    "#SS#SSSSSSS#S#S#\n"
    "#S##T#T###T#T#T#\n"
    "#TT#T#T#TTTTSST#\n"
    "#T##T#T#T#T#T#T#\n"
    "#TTTT#TTT#T#LKT#\n"
    "#T####T###TT##T#\n"
    "###########D####\n"
    "#TTTTTTTTTTTTTT#\n"
    "#T####T##T#T##T#\n"
    "#TTTTTT#TTBTLGL#\n"
    "################"
)

LEVEL_3_GRID = (
    "################\n"
    "#H.............#\n"
    "#T##T###T####T##\n"
    "#TT#T#T#TTBTTTW#\n"
    "#TT####T#####T##\n"
    "#TT####T#####L##\n"
    "#TTTTTTTTTT#GD##\n"
    "#T############T#\n"
    "#SSSSSSSTTTTTTT#\n"
    "#S####S####S##S#\n"
    "#S#TT#T#TT#TT#B#\n"
    "#TTTT#TTTTTBTTT#\n"
    "#WWWWWWWWWWWWWW#\n"
    "#T####T####L##L#\n"
    "#TTTTTT#TTTTTKT#\n"
    "################"
)


LEVEL_4_GRID = (
    "################\n"
    "#H.............#\n"
    "#TT###T#T####T##\n"
    "#TT#K#TTTTTTTTW#\n"
    "#TT#T#T#T####L##\n"
    "#TT#T#T#B#TTTDT#\n"
    "#TT#T#T#T#T##G##\n"
    "#TT#T#T#T###T###\n"
    "#TTTSSSTTTTTTTS#\n"
    "#S####S####S#SS#\n"
    "#S#SS#S#SS#TTTT#\n"
    "#SSSS#SSSS#TTBT#\n"
    "#LLLLLLLLLLLLLL#\n"
    "#T####T####T##T#\n"
    "#TSTTTT#TTTSSGW#\n"
    "################"
)

LEVEL_5_GRID = (
    "################\n"
    "#H.............#\n"
    "#T####T#T#T###T#\n"
    "#TT#TWTTTBTBTTT#\n"
    "#T####S#########\n"
    "#TT#TSLSL#SSSKS#\n"
    "#T##T#L#T####T##\n"
    "#TTTT#L#TTTTTTT#\n"
    "########D#######\n"
    "#TTTTTTTTTTTTTG#\n"
    "#TTTTTTTTTBTT#T#\n"
    "#LLLLLLLLLLLLLL#\n"
    "#TTTTTBTTTTTT#T#\n"
    "#WWWWWWWWWWWWWW#\n"
    "#TTTTTTTTTTTTTG#\n"
    "################"
)


_CHAR_TO_CELL = {
    "#": CELL_ROCK,
    "S": CELL_SOIL,
    "T": CELL_TILE,
    ".": CELL_EMPTY,
    "H": CELL_HOUSE,
    "G": CELL_GOLD,
    "L": CELL_LAVA,
    "K": CELL_KEY,
    "D": CELL_DOOR,
    "B": CELL_BOULDER,
    "W": CELL_WATER,
}

_CELL_TO_SPRITE = {
    CELL_ROCK: "rock",
    CELL_SOIL: "soil",
    CELL_TILE: "tile",
    CELL_LAVA: "lava",
    CELL_DOOR: "door",
    CELL_BOULDER: "boulder",
    CELL_WATER: "water",
}


def _grid_offset(grid_w: int, grid_h: int) -> Tuple[int, int]:
    ox = (CAM_SIZE - grid_w * TILE_SIZE) // 2
    oy = (CAM_SIZE - grid_h * TILE_SIZE) // 2
    return ox, oy


def _cell_px(gx: int, gy: int, ox: int, oy: int) -> Tuple[int, int]:
    return ox + gx * TILE_SIZE, oy + gy * TILE_SIZE


def _parse_level(grid_str: str, name: str) -> Level:
    rows = grid_str.strip().split("\n")
    grid_h = len(rows)
    grid_w = max(len(r) for r in rows)
    ox, oy = _grid_offset(grid_w, grid_h)

    level_sprites: List[Sprite] = []
    grid: List[List[int]] = []
    player_gx, player_gy = 0, 0
    house_gx, house_gy = 0, 0
    gold_positions: List[Tuple[int, int]] = []
    key_positions: List[Tuple[int, int]] = []

    for gy, row in enumerate(rows):
        grid_row: List[int] = []
        for gx, ch in enumerate(row):
            cell = _CHAR_TO_CELL.get(ch, CELL_EMPTY)
            grid_row.append(cell)
            px, py = _cell_px(gx, gy, ox, oy)

            if ch == "H":
                player_gx, player_gy = gx, gy
                house_gx, house_gy = gx, gy
                level_sprites.append(sprites["house"].clone().set_position(px, py))
                level_sprites.append(sprites["player"].clone().set_position(px, py))
            elif ch == "G":
                gold_positions.append((gx, gy))
                level_sprites.append(sprites["gold"].clone().set_position(px, py))
            elif ch == "K":
                key_positions.append((gx, gy))
                level_sprites.append(sprites["key"].clone().set_position(px, py))
            elif cell in _CELL_TO_SPRITE:
                spr_key = _CELL_TO_SPRITE[cell]
                level_sprites.append(sprites[spr_key].clone().set_position(px, py))
        grid.append(grid_row)

    return Level(
        sprites=level_sprites,
        grid_size=(CAM_SIZE, CAM_SIZE),
        data={
            "grid_w": grid_w,
            "grid_h": grid_h,
            "ox": ox,
            "oy": oy,
            "grid": grid,
            "player_start": (player_gx, player_gy),
            "house_pos": (house_gx, house_gy),
            "gold_positions": gold_positions,
            "key_positions": key_positions,
        },
        name=name,
    )


LEVEL_GRIDS: List[Tuple[str, str]] = [
    (LEVEL_1_GRID, "Level 1 \u2014 First Mine"),
    (LEVEL_2_GRID, "Level 2 \u2014 Lock & Key"),
    (LEVEL_3_GRID, "Level 3 \u2014 Water Gate"),
    (LEVEL_4_GRID, "Level 4 \u2014 Double Gold"),
    (LEVEL_5_GRID, "Level 5 \u2014 The Gauntlet"),
]


def _build_level(level_idx: int) -> Level:
    grid, name = LEVEL_GRIDS[level_idx]
    return _parse_level(grid, name)


class MinerHUD(RenderableUserDisplay):
    def __init__(self, game: "Gm02") -> None:
        self._game = game

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        g = self._game

        frame[60:64, :] = C_BG

        lives = getattr(g, "_lives", MAX_LIVES)
        for i in range(MAX_LIVES):
            x = 2 + i * 3
            colour = C_LAVA if i < lives else C_HUD_OFF
            frame[61:63, x : x + 2] = colour

        max_s = getattr(g, "_max_steps", 100)
        used = getattr(g, "_action_count", 0)
        ratio = max(0.0, 1.0 - used / max_s) if max_s > 0 else 0.0
        bar_width = 48
        filled = max(1, round(bar_width * ratio)) if ratio > 0 else 0
        if ratio > 0.5:
            bar_col = C_HOUSE
        elif ratio > 0.25:
            bar_col = C_GOLD
        else:
            bar_col = C_LAVA
        bar_x = 14
        if filled > 0:
            frame[61:63, bar_x : bar_x + filled] = bar_col
        if filled < bar_width:
            frame[61:63, bar_x + filled : bar_x + bar_width] = C_HUD_OFF

        return frame


class Gm02(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._hud = MinerHUD(self)
        self._lives: int = MAX_LIVES
        self._max_steps: int = 200
        self._history: List[dict] = []
        super().__init__(
            "gm02",
            [_build_level(i) for i in range(len(LEVEL_GRIDS))],
            Camera(
                0,
                0,
                CAM_SIZE,
                CAM_SIZE,
                C_FLOOR,
                C_FLOOR,
                [self._hud],
            ),
            available_actions=[0, 1, 2, 3, 4, 5, 7],
        )

    def on_set_level(self, level: Level) -> None:
        self._history = []

        idx = self._current_level_index
        self._max_steps = STEP_LIMITS[idx] if idx < len(STEP_LIMITS) else 200

        self._grid_w: int = self.current_level.get_data("grid_w")
        self._grid_h: int = self.current_level.get_data("grid_h")
        self._ox: int = self.current_level.get_data("ox")
        self._oy: int = self.current_level.get_data("oy")

        src_grid = self.current_level.get_data("grid")
        self._grid: List[List[int]] = [row[:] for row in src_grid]

        self._house_gx, self._house_gy = self.current_level.get_data("house_pos")

        pgx, pgy = self.current_level.get_data("player_start")
        self._pgx: int = pgx
        self._pgy: int = pgy
        self._facing: Tuple[int, int] = DIR_RIGHT
        self._player: Sprite = self.current_level.get_sprites_by_tag("player")[0]

        gold_positions = self.current_level.get_data("gold_positions")
        self._total_gold: int = len(gold_positions)
        self._gold_collected: int = 0

        self._gold_sprites: Dict[Tuple[int, int], Sprite] = {}
        for spr in self.current_level.get_sprites_by_tag("gold"):
            gx = (spr.x - self._ox) // TILE_SIZE
            gy = (spr.y - self._oy) // TILE_SIZE
            self._gold_sprites[(gx, gy)] = spr

        self._keys: int = 0

        self._key_sprites: Dict[Tuple[int, int], Sprite] = {}
        for spr in self.current_level.get_sprites_by_tag("key"):
            gx = (spr.x - self._ox) // TILE_SIZE
            gy = (spr.y - self._oy) // TILE_SIZE
            self._key_sprites[(gx, gy)] = spr

        self._terrain_sprites: Dict[Tuple[int, int], Sprite] = {}
        for spr in self.current_level.get_sprites_by_tag("terrain"):
            gx = (spr.x - self._ox) // TILE_SIZE
            gy = (spr.y - self._oy) // TILE_SIZE
            self._terrain_sprites[(gx, gy)] = spr

    def _to_px(self, gx: int, gy: int) -> Tuple[int, int]:
        return self._ox + gx * TILE_SIZE, self._oy + gy * TILE_SIZE

    def _in_bounds(self, gx: int, gy: int) -> bool:
        return 0 <= gx < self._grid_w and 0 <= gy < self._grid_h

    def _is_solid(self, gx: int, gy: int) -> bool:
        if not self._in_bounds(gx, gy):
            return True
        cell = self._grid[gy][gx]
        return cell in (
            CELL_ROCK,
            CELL_SOIL,
            CELL_TILE,
            CELL_CRACKED,
            CELL_DOOR,
            CELL_BOULDER,
            CELL_WATER,
        )

    def _is_minable(self, gx: int, gy: int) -> bool:
        if not self._in_bounds(gx, gy):
            return False
        cell = self._grid[gy][gx]
        return cell in (CELL_SOIL, CELL_TILE, CELL_CRACKED)

    def _is_lava(self, gx: int, gy: int) -> bool:
        if not self._in_bounds(gx, gy):
            return False
        return self._grid[gy][gx] == CELL_LAVA

    def _mine(self, gx: int, gy: int) -> None:
        if not self._in_bounds(gx, gy):
            return

        cell = self._grid[gy][gx]

        if cell == CELL_SOIL:
            self._grid[gy][gx] = CELL_EMPTY
            spr = self._terrain_sprites.pop((gx, gy), None)
            if spr:
                self.current_level.remove_sprite(spr)

        elif cell == CELL_TILE:
            self._grid[gy][gx] = CELL_CRACKED
            spr = self._terrain_sprites.get((gx, gy))
            if spr:
                self.current_level.remove_sprite(spr)
                px, py = self._to_px(gx, gy)
                new_spr = sprites["cracked"].clone().set_position(px, py)
                self.current_level.add_sprite(new_spr)
                self._terrain_sprites[(gx, gy)] = new_spr

        elif cell == CELL_CRACKED:
            self._grid[gy][gx] = CELL_EMPTY
            spr = self._terrain_sprites.pop((gx, gy), None)
            if spr:
                self.current_level.remove_sprite(spr)

    def _check_auto_pickup(self) -> None:
        pos = (self._pgx, self._pgy)
        if self._grid[self._pgy][self._pgx] == CELL_KEY:
            self._keys += 1
            self._grid[self._pgy][self._pgx] = CELL_EMPTY
            spr = self._key_sprites.pop(pos, None)
            if spr:
                self.current_level.remove_sprite(spr)

    def _try_pick_gold(self) -> bool:
        pos = (self._pgx, self._pgy)
        if pos not in self._gold_sprites:
            return False

        self._gold_collected += 1
        self._grid[self._pgy][self._pgx] = CELL_EMPTY

        spr = self._gold_sprites.pop(pos)
        self.current_level.remove_sprite(spr)

        if self._gold_collected == 1:
            self._player.color_remap(C_PLAYER, C_CARRY)

        return True

    def _try_unlock_door(self) -> bool:
        if self._keys <= 0:
            return False

        dx, dy = self._facing
        dgx = self._pgx + dx
        dgy = self._pgy + dy

        if not self._in_bounds(dgx, dgy):
            return False
        if self._grid[dgy][dgx] != CELL_DOOR:
            return False

        self._keys -= 1
        self._grid[dgy][dgx] = CELL_EMPTY
        spr = self._terrain_sprites.pop((dgx, dgy), None)
        if spr:
            self.current_level.remove_sprite(spr)

        return True

    def _try_push_boulder(self) -> bool:
        dx, dy = self._facing
        bgx = self._pgx + dx
        bgy = self._pgy + dy

        if not self._in_bounds(bgx, bgy):
            return False
        if self._grid[bgy][bgx] != CELL_BOULDER:
            return False

        dgx = bgx + dx
        dgy = bgy + dy
        if not self._in_bounds(dgx, dgy):
            return False

        dest = self._grid[dgy][dgx]

        if dest == CELL_EMPTY:
            self._grid[bgy][bgx] = CELL_EMPTY
            self._grid[dgy][dgx] = CELL_BOULDER
            spr = self._terrain_sprites.pop((bgx, bgy), None)
            if spr:
                spr.set_position(*self._to_px(dgx, dgy))
                self._terrain_sprites[(dgx, dgy)] = spr
            return True

        elif dest in (CELL_LAVA, CELL_WATER):
            self._grid[bgy][bgx] = CELL_EMPTY
            self._grid[dgy][dgx] = CELL_EMPTY
            b_spr = self._terrain_sprites.pop((bgx, bgy), None)
            if b_spr:
                self.current_level.remove_sprite(b_spr)
            h_spr = self._terrain_sprites.pop((dgx, dgy), None)
            if h_spr:
                self.current_level.remove_sprite(h_spr)
            return True

        return False

    def _is_complete(self) -> bool:
        return (
            self._gold_collected == self._total_gold
            and self._pgx == self._house_gx
            and self._pgy == self._house_gy
        )

    def _snapshot(self) -> dict:
        return {
            "grid": [row[:] for row in self._grid],
            "pgx": self._pgx,
            "pgy": self._pgy,
            "facing": self._facing,
            "lives": self._lives,
            "keys": self._keys,
            "gold_collected": self._gold_collected,
            "action_count": self._action_count,
            "player_color": np.array(self._player.pixels).tolist(),
        }

    def _rebuild_sprites(self) -> None:
        for spr in self._terrain_sprites.values():
            self.current_level.remove_sprite(spr)
        self._terrain_sprites.clear()

        for spr in self._gold_sprites.values():
            self.current_level.remove_sprite(spr)
        self._gold_sprites.clear()

        for spr in self._key_sprites.values():
            self.current_level.remove_sprite(spr)
        self._key_sprites.clear()

        terrain_map = {
            CELL_ROCK: "rock",
            CELL_SOIL: "soil",
            CELL_TILE: "tile",
            CELL_CRACKED: "cracked",
            CELL_LAVA: "lava",
            CELL_DOOR: "door",
            CELL_BOULDER: "boulder",
            CELL_WATER: "water",
        }

        for gy in range(self._grid_h):
            for gx in range(self._grid_w):
                cell = self._grid[gy][gx]
                px, py = self._to_px(gx, gy)

                if cell in terrain_map:
                    new_spr = sprites[terrain_map[cell]].clone().set_position(px, py)
                    self.current_level.add_sprite(new_spr)
                    self._terrain_sprites[(gx, gy)] = new_spr
                elif cell == CELL_GOLD:
                    new_spr = sprites["gold"].clone().set_position(px, py)
                    self.current_level.add_sprite(new_spr)
                    self._gold_sprites[(gx, gy)] = new_spr
                elif cell == CELL_KEY:
                    new_spr = sprites["key"].clone().set_position(px, py)
                    self.current_level.add_sprite(new_spr)
                    self._key_sprites[(gx, gy)] = new_spr

    def _restore_snapshot(self, snapshot: dict) -> None:
        self._grid = [row[:] for row in snapshot["grid"]]
        self._pgx = snapshot["pgx"]
        self._pgy = snapshot["pgy"]
        self._facing = snapshot["facing"]
        self._lives = snapshot["lives"]
        self._keys = snapshot["keys"]
        self._gold_collected = snapshot["gold_collected"]
        self._action_count = snapshot["action_count"]
        self._player.pixels = np.array(snapshot["player_color"])
        self._player.set_position(*self._to_px(self._pgx, self._pgy))
        self._rebuild_sprites()

    def _lose_life(self) -> None:
        self._lives -= 1
        if self._lives > 0:
            self.level_reset()
        else:
            self.lose()

    def handle_reset(self) -> None:
        self._lives = MAX_LIVES
        self.level_reset()

    def step(self) -> None:
        action = self.action.id

        if action == GameAction.RESET:
            self.handle_reset()
            self.complete_action()
            return

        if self._lives <= 0:
            self.lose()
            self.complete_action()
            return

        if action == GameAction.ACTION7:
            if self._history:
                self._restore_snapshot(self._history.pop())
            self.complete_action()
            return

        if self._action_count >= self._max_steps:
            self._lose_life()
            self.complete_action()
            return

        if action in (
            GameAction.ACTION1,
            GameAction.ACTION2,
            GameAction.ACTION3,
            GameAction.ACTION4,
        ):
            snapshot = self._snapshot()
            if action == GameAction.ACTION1:
                self._facing = DIR_UP
            elif action == GameAction.ACTION2:
                self._facing = DIR_DOWN
            elif action == GameAction.ACTION3:
                self._facing = DIR_LEFT
            elif action == GameAction.ACTION4:
                self._facing = DIR_RIGHT

            dx, dy = self._facing
            ngx = self._pgx + dx
            ngy = self._pgy + dy

            if self._is_solid(ngx, ngy):
                self._facing = snapshot["facing"]
                self.complete_action()
                return

            self._history.append(snapshot)

            if self._is_lava(ngx, ngy):
                self._lose_life()
                self.complete_action()
                return

            self._pgx = ngx
            self._pgy = ngy
            self._player.set_position(*self._to_px(ngx, ngy))

            self._check_auto_pickup()

            if self._is_complete():
                self._lives = MAX_LIVES
                self.next_level()
                self.complete_action()
                return

            self.complete_action()

        elif action == GameAction.ACTION5:
            snapshot = self._snapshot()
            did_something = False

            if self._try_pick_gold():
                did_something = True
            elif self._try_unlock_door():
                did_something = True
            elif self._try_push_boulder():
                did_something = True
            else:
                dx, dy = self._facing
                tgx = self._pgx + dx
                tgy = self._pgy + dy
                if self._is_minable(tgx, tgy):
                    self._mine(tgx, tgy)
                    did_something = True

            if did_something:
                self._history.append(snapshot)

            self.complete_action()

        else:
            self.complete_action()


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


ACTION_MAP = {
    "reset": GameAction.RESET,
    "up": GameAction.ACTION1,
    "down": GameAction.ACTION2,
    "left": GameAction.ACTION3,
    "right": GameAction.ACTION4,
    "select": GameAction.ACTION5,
    "undo": GameAction.ACTION7,
}


class PuzzleEnvironment:
    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self._engine = Gm02(seed=seed)
        self._turn = 0
        self._done = False
        self._last_action_was_reset = False
        self._level_index = 0
        self._total_levels = len(LEVEL_GRIDS)

    def reset(self) -> GameState:
        self._require_engine()

        if self._last_action_was_reset:
            target_level = 0
        else:
            target_level = self._level_index

        self._engine = Gm02(seed=self._seed)
        for _ in range(target_level):
            self._engine.next_level()
        self._level_index = target_level
        self._turn = 0
        self._done = False
        self._last_action_was_reset = True
        return self._build_state()

    def get_actions(self) -> List[str]:
        if self._done:
            return ["reset"]
        return ["reset", "up", "down", "left", "right", "select", "undo"]

    def is_done(self) -> bool:
        engine = self._require_engine()
        return self._done or engine._state in (
            EngineGameState.GAME_OVER,
            EngineGameState.WIN,
        )

    def _require_engine(self) -> Gm02:
        if self._engine is None:
            raise RuntimeError("PuzzleEnvironment is closed")
        return self._engine

    def _build_text_observation(self) -> str:
        g = self._require_engine()
        rows = []
        for gy in range(g._grid_h):
            chars = []
            for gx in range(g._grid_w):
                if gx == g._pgx and gy == g._pgy:
                    chars.append("P")
                    continue
                cell = g._grid[gy][gx]
                chars.append({
                    CELL_EMPTY: ".",
                    CELL_SOIL: "S",
                    CELL_TILE: "T",
                    CELL_CRACKED: "~",
                    CELL_ROCK: "#",
                    CELL_LAVA: "L",
                    CELL_GOLD: "G",
                    CELL_HOUSE: "H",
                    CELL_KEY: "K",
                    CELL_DOOR: "D",
                    CELL_BOULDER: "B",
                    CELL_WATER: "W",
                }.get(cell, "."))
            rows.append("".join(chars))
        status = f"level={g._current_level_index + 1} gold={g._gold_collected}/{g._total_gold} keys={g._keys} lives={g._lives} steps={max(g._max_steps - g._action_count, 0)}"
        return "\n".join(rows + [status])

    def _build_image_observation(self) -> Optional[bytes]:
        return None

    def _build_state(self) -> GameState:
        return GameState(
            text_observation=self._build_text_observation(),
            image_observation=self._build_image_observation(),
            valid_actions=None if self.is_done() else self.get_actions(),
            turn=self._turn,
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
        self._turn += 1

        engine = self._require_engine()
        prev_level = engine._current_level_index
        prev_lives = engine._lives

        engine.perform_action(ActionInput(id=ACTION_MAP[action_lower]))

        reward = 0.0
        info: Dict = {}

        if engine._state == EngineGameState.GAME_OVER:
            self._done = True
            info["outcome"] = "game_over"
        elif engine._state == EngineGameState.WIN:
            reward = 1.0 / self._total_levels
            self._done = True
            info["outcome"] = "game_complete"
        elif engine._current_level_index != prev_level:
            reward = 1.0 / self._total_levels
            self._level_index = engine._current_level_index
            info["outcome"] = "level_complete"
        elif engine._lives < prev_lives:
            info["outcome"] = "death"

        return StepResult(self._build_state(), reward, self._done, info)

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError(f"Unsupported render mode: {mode}")
        engine = self._require_engine()
        index_grid = engine.camera.render(engine.current_level.get_sprites())
        h, w = index_grid.shape[:2]
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
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
