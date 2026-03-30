import struct
import zlib
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
    metadata: Dict = field(default_factory=dict)


@dataclass
class StepResult:
    state: GameState
    reward: float
    done: bool
    info: Dict = field(default_factory=dict)


MAX_LIVES = 3

_LIFE_ON_PIX = [[3, 3, 3], [3, 3, 3], [3, 3, 3]]
_LIFE_OFF_PIX = [[5, 5, 5], [5, 5, 5], [5, 5, 5]]


def _make_life_pair(slot: int):
    px = 1 + slot * 4
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


BACKGROUND_COLOR = 5
LETTER_BOX_COLOR = 5

C_AGENT = 11
C_WAVE = 9
C_GOAL = 14
C_SOURCE = 7
C_DEATH = 2
C_REVERSE = 15
C_BLINK = 12
C_WALL = 0
C_TRAIL = 13
C_SPEED = 6
C_HUD_DOT = 8
C_HUD_BAR = 10

sprites = {
    "agent": Sprite(
        pixels=[[C_AGENT]],
        name="agent",
        visible=True,
        collidable=False,
        layer=3,
        tags=["agent"],
    ),
    "wave": Sprite(
        pixels=[[C_WAVE]],
        name="wave",
        visible=True,
        collidable=False,
        layer=1,
        tags=["wave"],
    ),
    "goal": Sprite(
        pixels=[[C_GOAL]],
        name="goal",
        visible=True,
        collidable=False,
        layer=0,
        tags=["goal"],
    ),
    "source": Sprite(
        pixels=[[C_SOURCE]],
        name="source",
        visible=True,
        collidable=False,
        layer=0,
        tags=["source"],
    ),
    "reverse_tile": Sprite(
        pixels=[[C_REVERSE]],
        name="reverse_tile",
        visible=True,
        collidable=False,
        layer=0,
        tags=["reverse_tile"],
    ),
    "blink_tile": Sprite(
        pixels=[[C_BLINK]],
        name="blink_tile",
        visible=True,
        collidable=False,
        layer=1,
        tags=["blink_tile"],
    ),
    "wall": Sprite(
        pixels=[[C_WALL]],
        name="wall",
        visible=True,
        collidable=False,
        layer=1,
        tags=["wall"],
    ),
    "trail": Sprite(
        pixels=[[C_TRAIL]],
        name="trail",
        visible=True,
        collidable=False,
        layer=0,
        tags=["trail"],
    ),
    "speed_tile": Sprite(
        pixels=[[C_SPEED]],
        name="speed_tile",
        visible=True,
        collidable=False,
        layer=0,
        tags=["speed_tile"],
    ),
    "moving_wall": Sprite(
        pixels=[[C_WALL]],
        name="moving_wall",
        visible=True,
        collidable=False,
        layer=1,
        tags=["moving_wall"],
    ),
    "hud_bar": Sprite(
        pixels=[[C_HUD_BAR]],
        name="hud_bar",
        visible=True,
        collidable=False,
        layer=4,
        tags=["hud_bar"],
    ),
    "hud_dot": Sprite(
        pixels=[[C_HUD_DOT]],
        name="hud_dot",
        visible=True,
        collidable=False,
        layer=4,
        tags=["hud_dot"],
    ),
    "death_block": Sprite(
        pixels=[[C_DEATH]],
        name="death_block",
        visible=True,
        collidable=False,
        layer=1,
        tags=["death_block"],
    ),
}

LEVEL_1_GRID = (
    "WWWWWWWWWWWW\n"
    "WW..W..S..WW\n"
    "W..W.P..W..W\n"
    "WW.W..W.W.WW\n"
    "WW.WD.W.W.WW\n"
    "WWWWWRWWW.WW\n"
    "WW.WD...W.WW\n"
    "WW.W.G..W.WW\n"
    "WW.W.W..W.WW\n"
    "WW.W.WW.W.WW\n"
    "WW.W.WW.W.WW\n"
    "WWWWWWWWWWWW"
)

LEVEL_2_GRID = (
    "WWWWWWWWWWWW\n"
    "WW....S...WW\n"
    "WD.W....W..W\n"
    "WW.W....W.WW\n"
    "WW.W.P..WD.W\n"
    "WW.W....W.WW\n"
    "WW.W....W.WW\n"
    "WW.WWRWWWRWW\n"
    "WW.W....W.WW\n"
    "WW.W.G..W.WW\n"
    "WW.W.WW.W.WW\n"
    "WWWWWWWWWWWW"
)

LEVEL_3_GRID = (
    "WWWWWWWWSWWWWWWW\n"
    "W.W..........W.W\n"
    "W..W...........W\n"
    "W....W......W..W\n"
    "W.W......W....DW\n"
    "W...W..P...W...W\n"
    "W..WD.W...W.DW.W\n"
    "WWWWWWWOOWWWWWWW\n"
    "W.W..W.WR.W..W.W\n"
    "W.W..W.WW.W..W.W\n"
    "W.W.WW...GWW.W.W\n"
    "W.W..W.WWWW..W.W\n"
    "W.W..W.W..W..W.W\n"
    "W.W..W....W..W.W\n"
    "W.W..W....W..W.W\n"
    "WWWWWWWWWWWWWWWW"
)

LEVEL_4_GRID = (
    "SWWWWWWWWWWWWWWWWWWS\n"
    "....................\n"
    "..WWWW.WW..WWWWWWW..\n"
    "..W..............W..\n"
    "..W.WWWWW..WWWWW.W..\n"
    "..W.W..........W.W..\n"
    "..W.W..........W.W..\n"
    "..W.W.....M.M.MW.W..\n"
    "D.W.W......W...W.W.D\n"
    "........P..R...G....\n"
    "D..........W.V.W...D\n"
    "..W.W..........W.W..\n"
    "..W.W..........W.W..\n"
    "..W.W..........W.W..\n"
    "..W.W..........W.W..\n"
    "..W.WWWWW..WWWWW.W..\n"
    "..W..............W..\n"
    "..WWWWWWW..WWWWWWW..\n"
    "....................\n"
    "WWWWWWWWWWWWWWWWWWWW"
)

LEVEL_5_GRID = (
    "SWWWWWWWWWWWWWWWWWWWWWWW\n"
    "........................\n"
    "..WWWWWWW.W..WWWWWWWWW..\n"
    "..W..................W..\n"
    "..W.WWWWWWW..WWWWWWW.W..\n"
    "..W.WM............MW.W.S\n"
    "..W.W.WWWWW..WWWWW.W.W..\n"
    "..W.W.W..........W.W.W..\n"
    "..W.W.W.WWW..WWW.W.W.W..\n"
    "..W.W.W.W......W.W.W.W..\n"
    "..W.W.W.W...W..W.W.W.W..\n"
    "..........P.R..O........\n"
    "........................\n"
    "..W.W.W.W......WVW.W.W..\n"
    "..W.W.W.W......W.W.W.W..\n"
    "..W.W.W.WWW..WWW.W.GWW..\n"
    "..W.W.W.W......W.W.W.W..\n"
    "..W.WMWWWWW..WWWWW.W.W..\n"
    "..W.W..............W.W..\n"
    "..W.WWWWWWW..WWWWWWW.W..\n"
    "D.W..................WD.\n"
    "..WWWWWWWWW..WWWWWWWWW..\n"
    "........................\n"
    "WWWWWWWWWWWWWWWWWWWWWWWW"
)

CAM_SIZE = 32


def _parse_grid(grid_str: str) -> List[List[str]]:
    rows = grid_str.strip().split("\n")
    return [list(row) for row in rows]


def _build_level(
    grid_str: str,
    grid_size: Tuple[int, int],
    max_actions: int,
    max_clicks: int,
    level_name: str,
) -> Level:
    grid = _parse_grid(grid_str)
    grid_h = len(grid)
    grid_w = len(grid[0]) if grid_h > 0 else 0

    gs_w, gs_h = grid_size
    offset_x = (gs_w - grid_w) // 2
    offset_y = (gs_h - grid_h) // 2

    level_sprites: List[Sprite] = []
    sources: List[Tuple[int, int]] = []
    agent_pos: Optional[Tuple[int, int]] = None
    goal_pos: Optional[Tuple[int, int]] = None
    reverse_tiles: List[Tuple[int, int]] = []
    blink_tiles: List[Tuple[int, int]] = []
    speed_tiles: List[Tuple[int, int]] = []
    wall_positions: List[Tuple[int, int]] = []
    moving_wall_data: List[Dict] = []
    death_positions: List[Tuple[int, int]] = []

    for gy in range(grid_h):
        for gx in range(grid_w):
            if gx >= len(grid[gy]):
                continue
            ch = grid[gy][gx]
            wx = offset_x + gx
            wy = offset_y + gy

            if ch == "P":
                agent_pos = (wx, wy)
            elif ch == "G":
                goal_pos = (wx, wy)
                level_sprites.append(sprites["goal"].clone().set_position(wx, wy))
            elif ch == "S":
                sources.append((wx, wy))
                level_sprites.append(sprites["source"].clone().set_position(wx, wy))
            elif ch == "W":
                wall_positions.append((wx, wy))
                level_sprites.append(sprites["wall"].clone().set_position(wx, wy))
            elif ch == "R":
                reverse_tiles.append((wx, wy))
                level_sprites.append(
                    sprites["reverse_tile"].clone().set_position(wx, wy)
                )
            elif ch == "O":
                blink_tiles.append((wx, wy))
                level_sprites.append(sprites["blink_tile"].clone().set_position(wx, wy))
            elif ch == "M":
                speed_tiles.append((wx, wy))
                level_sprites.append(sprites["speed_tile"].clone().set_position(wx, wy))
            elif ch == "D":
                death_positions.append((wx, wy))
                level_sprites.append(
                    sprites["death_block"].clone().set_position(wx, wy)
                )
            elif ch == "V":
                moving_wall_data.append(
                    {
                        "x": wx,
                        "y": wy,
                        "dx": 1,
                        "dy": 0,
                    }
                )
                level_sprites.append(
                    sprites["moving_wall"].clone().set_position(wx, wy)
                )
                wall_positions.append((wx, wy))

    if agent_pos is not None:
        level_sprites.append(
            sprites["agent"].clone().set_position(agent_pos[0], agent_pos[1])
        )

    return Level(
        sprites=level_sprites,
        grid_size=grid_size,
        data={
            "grid_w": grid_w,
            "grid_h": grid_h,
            "offset_x": offset_x,
            "offset_y": offset_y,
            "agent_pos": list(agent_pos) if agent_pos else [0, 0],
            "goal_pos": list(goal_pos) if goal_pos else [0, 0],
            "sources": [list(s) for s in sources],
            "wall_positions": [list(w) for w in wall_positions],
            "reverse_tiles": [list(r) for r in reverse_tiles],
            "blink_tiles": [list(b) for b in blink_tiles],
            "speed_tiles": [list(s) for s in speed_tiles],
            "moving_wall_data": moving_wall_data,
            "death_positions": [list(d) for d in death_positions],
            "max_actions": max_actions,
            "max_clicks": max_clicks,
        },
        name=level_name,
    )


levels = [
    _build_level(LEVEL_1_GRID, (16, 16), 30, 1, "Level 1"),
    _build_level(LEVEL_2_GRID, (16, 16), 35, 2, "Level 2"),
    _build_level(LEVEL_3_GRID, (20, 20), 45, 2, "Level 3"),
    _build_level(LEVEL_4_GRID, (32, 32), 50, 3, "Level 4"),
    _build_level(LEVEL_5_GRID, (32, 32), 60, 5, "Level 5"),
]


class Bw01(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        life_pairs = [_make_life_pair(i) for i in range(MAX_LIVES)]
        self._lives_hud = ToggleableUserDisplay(life_pairs)
        for i in range(MAX_LIVES):
            self._lives_hud.enable(i)

        self.lives: int = MAX_LIVES
        self._pit_reset: bool = False
        self.agent_x: int = 0
        self.agent_y: int = 0
        self.goal_x: int = 0
        self.goal_y: int = 0
        self.grid_w: int = 8
        self.grid_h: int = 8
        self.offset_x: int = 0
        self.offset_y: int = 0
        self.wave_cells: Set[Tuple[int, int]] = set()
        self.wall_cells: Set[Tuple[int, int]] = set()
        self.reverse_tile_cells: Set[Tuple[int, int]] = set()
        self.blink_tile_positions: Set[Tuple[int, int]] = set()
        self.blink_tiles: Dict[Tuple[int, int], Sprite] = {}
        self.speed_tile_cells: Set[Tuple[int, int]] = set()
        self.speed_tile_sprites: Dict[Tuple[int, int], Sprite] = {}
        self.moving_walls: List[Dict] = []
        self.blink_solid: bool = True
        self.blink_counter: int = 0
        self.reverse_turns: int = 0
        self.speed_next_turn: bool = False
        self.game_over: bool = False
        self.total_actions: int = 0
        self.click_count: int = 0
        self.level_max_actions: int = 15
        self.level_max_clicks: int = 2
        self.wave_sprites: Dict[Tuple[int, int], Sprite] = {}
        self.trail_sprites: Dict[Tuple[int, int], Sprite] = {}
        self.hud_bar_sprites: List[Sprite] = []
        self.hud_dot_sprites: List[Sprite] = []
        self.agent_sprite: Optional[Sprite] = None
        self.sources: List[Tuple[int, int]] = []
        self.death_cells: Set[Tuple[int, int]] = set()
        self._undo_stack: List[dict] = []

        super().__init__(
            game_id="bw01",
            levels=levels,
            camera=Camera(
                x=0,
                y=0,
                width=CAM_SIZE,
                height=CAM_SIZE,
                background=BACKGROUND_COLOR,
                letter_box=LETTER_BOX_COLOR,
                interfaces=[self._lives_hud],
            ),
            available_actions=[0, 1, 2, 3, 4, 6, 7],
        )

    def on_set_level(self, level: Level) -> None:
        self.grid_w = self.current_level.get_data("grid_w")
        self.grid_h = self.current_level.get_data("grid_h")
        self.offset_x = self.current_level.get_data("offset_x")
        self.offset_y = self.current_level.get_data("offset_y")

        agent_data = self.current_level.get_data("agent_pos")
        self.agent_x = agent_data[0]
        self.agent_y = agent_data[1]

        goal_data = self.current_level.get_data("goal_pos")
        self.goal_x = goal_data[0]
        self.goal_y = goal_data[1]

        self.level_max_actions = self.current_level.get_data("max_actions")
        self.level_max_clicks = self.current_level.get_data("max_clicks")

        self.total_actions = 0
        self.click_count = 0
        self.reverse_turns = 0
        self.speed_next_turn = False
        self.blink_solid = True
        self.blink_counter = 0
        self.game_over = False
        self._undo_stack = []

        self.wave_cells = set()
        self.wave_sprites = {}
        self.trail_sprites = {}
        self.hud_bar_sprites = []
        self.hud_dot_sprites = []

        wall_data = self.current_level.get_data("wall_positions")
        self.wall_cells = set()
        for w in wall_data:
            self.wall_cells.add((w[0], w[1]))

        death_data = self.current_level.get_data("death_positions")
        self.death_cells = set()
        for d in death_data:
            self.death_cells.add((d[0], d[1]))

        rev_data = self.current_level.get_data("reverse_tiles")
        self.reverse_tile_cells = set()
        for r in rev_data:
            self.reverse_tile_cells.add((r[0], r[1]))

        blink_data = self.current_level.get_data("blink_tiles")
        self.blink_tile_positions = set()
        self.blink_tiles = {}
        blink_sprite_list = self.current_level.get_sprites_by_tag("blink_tile")
        blink_idx = 0
        for b in blink_data:
            pos = (b[0], b[1])
            self.blink_tile_positions.add(pos)
            if blink_idx < len(blink_sprite_list):
                self.blink_tiles[pos] = blink_sprite_list[blink_idx]
                blink_idx += 1

        speed_data = self.current_level.get_data("speed_tiles")
        self.speed_tile_cells = set()
        self.speed_tile_sprites = {}
        speed_sprite_list = self.current_level.get_sprites_by_tag("speed_tile")
        speed_idx = 0
        for s in speed_data:
            pos = (s[0], s[1])
            self.speed_tile_cells.add(pos)
            if speed_idx < len(speed_sprite_list):
                self.speed_tile_sprites[pos] = speed_sprite_list[speed_idx]
                speed_idx += 1

        mw_data = self.current_level.get_data("moving_wall_data")
        mw_sprite_list = self.current_level.get_sprites_by_tag("moving_wall")
        self.moving_walls = []
        for i, mw in enumerate(mw_data):
            entry = {
                "x": mw["x"],
                "y": mw["y"],
                "dx": mw["dx"],
                "dy": mw["dy"],
                "sprite": mw_sprite_list[i] if i < len(mw_sprite_list) else None,
            }
            self.moving_walls.append(entry)

        src_data = self.current_level.get_data("sources")
        self.sources = []
        for s in src_data:
            pos = (s[0], s[1])
            self.sources.append(pos)
            self.wave_cells.add(pos)
            spr = sprites["wave"].clone().set_position(pos[0], pos[1])
            self.current_level.add_sprite(spr)
            self.wave_sprites[pos] = spr

        agent_sprites = self.current_level.get_sprites_by_tag("agent")
        if agent_sprites:
            self.agent_sprite = agent_sprites[0]
        else:
            self.agent_sprite = None

        gs = self.current_level.grid_size
        if gs:
            self.camera.width = gs[0]
            self.camera.height = gs[1]

        if not self._pit_reset:
            self.lives = MAX_LIVES
            for i in range(MAX_LIVES):
                self._lives_hud.enable(i)

        self.camera.replace_interface([self._lives_hud])

        self._update_hud_bar()
        self._update_hud_dots()

    def _in_grid(self, wx: int, wy: int) -> bool:
        gx = wx - self.offset_x
        gy = wy - self.offset_y
        return 0 <= gx < self.grid_w and 0 <= gy < self.grid_h

    def _is_wall(self, wx: int, wy: int) -> bool:
        return (wx, wy) in self.wall_cells

    def _is_solid_blink(self, wx: int, wy: int) -> bool:
        return (wx, wy) in self.blink_tile_positions and self.blink_solid

    def _is_death(self, wx: int, wy: int) -> bool:
        return (wx, wy) in self.death_cells

    def _is_wave(self, wx: int, wy: int) -> bool:
        return (wx, wy) in self.wave_cells

    def _expand_wave(self, rings: int = 1) -> None:
        for _ in range(rings):
            frontier: Set[Tuple[int, int]] = set()
            for wx, wy in self.wave_cells:
                for ddx, ddy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                    nx, ny = wx + ddx, wy + ddy
                    if not self._in_grid(nx, ny):
                        continue
                    if (nx, ny) in self.wall_cells:
                        continue
                    if (nx, ny) in self.blink_tile_positions and self.blink_solid:
                        continue
                    if (nx, ny) not in self.wave_cells:
                        frontier.add((nx, ny))
            for pos in frontier:
                self.wave_cells.add(pos)
                spr = sprites["wave"].clone().set_position(pos[0], pos[1])
                self.current_level.add_sprite(spr)
                self.wave_sprites[pos] = spr
                if pos in self.death_cells:
                    self._lose()
                    return

    def _update_moving_walls(self) -> None:
        for mw in self.moving_walls:
            old_pos = (mw["x"], mw["y"])
            self.wall_cells.discard(old_pos)
            if mw["sprite"] is not None:
                self.current_level.remove_sprite(mw["sprite"])

            new_x = mw["x"] + mw["dx"]
            new_y = mw["y"] + mw["dy"]

            if not self._in_grid(new_x, new_y):
                mw["dx"] = -mw["dx"]
                mw["dy"] = -mw["dy"]
                new_x = mw["x"] + mw["dx"]
                new_y = mw["y"] + mw["dy"]
                if not self._in_grid(new_x, new_y):
                    new_x = mw["x"]
                    new_y = mw["y"]

            mw["x"] = new_x
            mw["y"] = new_y
            self.wall_cells.add((new_x, new_y))
            spr = sprites["moving_wall"].clone().set_position(new_x, new_y)
            self.current_level.add_sprite(spr)
            mw["sprite"] = spr

    def _save_undo_state(self) -> None:
        snapshot = {
            "agent_x": self.agent_x,
            "agent_y": self.agent_y,
            "wave_cells": set(self.wave_cells),
            "wall_cells": set(self.wall_cells),
            "moving_walls": [
                {"x": mw["x"], "y": mw["y"], "dx": mw["dx"], "dy": mw["dy"]}
                for mw in self.moving_walls
            ],
            "total_actions": self.total_actions,
            "click_count": self.click_count,
            "reverse_turns": self.reverse_turns,
            "speed_next_turn": self.speed_next_turn,
            "blink_solid": self.blink_solid,
            "blink_counter": self.blink_counter,
            "trail_positions": set(self.trail_sprites.keys()),
            "speed_tile_cells": set(self.speed_tile_cells),
            "death_cells": set(self.death_cells),
        }
        self._undo_stack.append(snapshot)
        if len(self._undo_stack) > 50:
            self._undo_stack.pop(0)

    def _apply_undo(self) -> bool:
        if not self._undo_stack:
            return False
        snapshot = self._undo_stack.pop()

        self.agent_x = snapshot["agent_x"]
        self.agent_y = snapshot["agent_y"]
        self.total_actions = snapshot["total_actions"]
        self.click_count = snapshot["click_count"]
        self.reverse_turns = snapshot["reverse_turns"]
        self.speed_next_turn = snapshot["speed_next_turn"]
        self.blink_solid = snapshot["blink_solid"]
        self.blink_counter = snapshot["blink_counter"]
        self.death_cells = snapshot["death_cells"]

        for pos in list(self.wave_cells):
            if pos in self.wave_sprites:
                self.current_level.remove_sprite(self.wave_sprites[pos])
                del self.wave_sprites[pos]
        self.wave_cells = snapshot["wave_cells"]
        for pos in self.wave_cells:
            if pos not in self.wave_sprites:
                spr = sprites["wave"].clone().set_position(pos[0], pos[1])
                self.current_level.add_sprite(spr)
                self.wave_sprites[pos] = spr

        for pos in list(self.trail_sprites.keys()):
            if pos not in snapshot["trail_positions"]:
                self.current_level.remove_sprite(self.trail_sprites[pos])
                del self.trail_sprites[pos]
        for pos in snapshot["trail_positions"]:
            if pos not in self.trail_sprites:
                spr = sprites["trail"].clone().set_position(pos[0], pos[1])
                self.current_level.add_sprite(spr)
                self.trail_sprites[pos] = spr

        for pos in list(self.speed_tile_cells):
            if pos not in snapshot["speed_tile_cells"] and pos in self.speed_tile_sprites:
                self.current_level.remove_sprite(self.speed_tile_sprites[pos])
                del self.speed_tile_sprites[pos]
        for pos in snapshot["speed_tile_cells"]:
            if pos not in self.speed_tile_cells and pos not in self.speed_tile_sprites:
                spr = sprites["speed_tile"].clone().set_position(pos[0], pos[1])
                self.current_level.add_sprite(spr)
                self.speed_tile_sprites[pos] = spr
        self.speed_tile_cells = snapshot["speed_tile_cells"]

        for mw in self.moving_walls:
            old_pos = (mw["x"], mw["y"])
            self.wall_cells.discard(old_pos)
            if mw["sprite"] is not None:
                self.current_level.remove_sprite(mw["sprite"])

        self.wall_cells = snapshot["wall_cells"]
        for i, mw_snap in enumerate(snapshot["moving_walls"]):
            if i < len(self.moving_walls):
                self.moving_walls[i]["x"] = mw_snap["x"]
                self.moving_walls[i]["y"] = mw_snap["y"]
                self.moving_walls[i]["dx"] = mw_snap["dx"]
                self.moving_walls[i]["dy"] = mw_snap["dy"]
                spr = sprites["moving_wall"].clone().set_position(mw_snap["x"], mw_snap["y"])
                self.current_level.add_sprite(spr)
                self.moving_walls[i]["sprite"] = spr

        for pos, spr in self.blink_tiles.items():
            spr.set_visible(self.blink_solid)

        if self.agent_sprite is not None:
            self.agent_sprite.set_position(self.agent_x, self.agent_y)

        self._update_hud_bar()
        self._update_hud_dots()
        return True

    def _lose(self) -> None:
        self.game_over = True
        self.lives -= 1
        if self.lives <= 0:
            self.lose()
        else:
            self._pit_reset = True
            self.level_reset()
            self._pit_reset = False
            for i in range(self.lives, MAX_LIVES):
                self._lives_hud.disable(i)

    def _update_hud_bar(self) -> None:
        for s in self.hud_bar_sprites:
            self.current_level.remove_sprite(s)
        self.hud_bar_sprites.clear()

        remaining = max(0, self.level_max_actions - self.total_actions)
        bar_width = self.camera.width
        if self.level_max_actions > 0:
            filled = int(bar_width * remaining / self.level_max_actions)
        else:
            filled = 0
        bar_y = self.camera.height - 1

        for i in range(filled):
            spr = sprites["hud_bar"].clone().set_position(i, bar_y)
            self.current_level.add_sprite(spr)
            self.hud_bar_sprites.append(spr)

    def _update_hud_dots(self) -> None:
        for s in self.hud_dot_sprites:
            self.current_level.remove_sprite(s)
        self.hud_dot_sprites.clear()

    def _check_tile_effects(self) -> None:
        pos = (self.agent_x, self.agent_y)

        if pos in self.reverse_tile_cells:
            self.reverse_turns = 3

        if pos in self.speed_tile_cells:
            self.speed_next_turn = True
            self.speed_tile_cells.discard(pos)
            if pos in self.speed_tile_sprites:
                self.current_level.remove_sprite(self.speed_tile_sprites[pos])
                del self.speed_tile_sprites[pos]

    def _drop_trail(self, wx: int, wy: int) -> None:
        pos = (wx, wy)
        if pos not in self.trail_sprites and pos not in self.wave_cells:
            spr = sprites["trail"].clone().set_position(wx, wy)
            self.current_level.add_sprite(spr)
            self.trail_sprites[pos] = spr

    def _shared_post_action(self) -> None:
        self._update_moving_walls()

        self.blink_counter += 1
        if self.blink_counter >= 2:
            self.blink_counter = 0
            self.blink_solid = not self.blink_solid
            for pos, spr in self.blink_tiles.items():
                spr.set_visible(self.blink_solid)

        rings = 2 if self.speed_next_turn else 1
        self.speed_next_turn = False
        self._expand_wave(rings)

        if self.game_over:
            self.complete_action()
            return

        if (self.agent_x, self.agent_y) in self.wave_cells:
            self._lose()
            self.complete_action()
            return

        if self.agent_x == self.goal_x and self.agent_y == self.goal_y:
            self.next_level()
            self.complete_action()
            return

        if (self.goal_x, self.goal_y) in self.wave_cells:
            self._lose()
            self.complete_action()
            return

        if self.reverse_turns > 0:
            self.reverse_turns -= 1

        self._update_hud_bar()

        self._update_hud_dots()

        self.complete_action()

    def step(self) -> None:
        if self.game_over:
            self.complete_action()
            return

        if self.action.id in (
            GameAction.ACTION1,
            GameAction.ACTION2,
            GameAction.ACTION3,
            GameAction.ACTION4,
        ):
            self._save_undo_state()
            dx, dy = 0, 0
            if self.reverse_turns > 0:
                if self.action.id == GameAction.ACTION1:
                    dy = 1
                elif self.action.id == GameAction.ACTION2:
                    dy = -1
                elif self.action.id == GameAction.ACTION3:
                    dx = 1
                elif self.action.id == GameAction.ACTION4:
                    dx = -1
            else:
                if self.action.id == GameAction.ACTION1:
                    dy = -1
                elif self.action.id == GameAction.ACTION2:
                    dy = 1
                elif self.action.id == GameAction.ACTION3:
                    dx = -1
                elif self.action.id == GameAction.ACTION4:
                    dx = 1

            new_x = self.agent_x + dx
            new_y = self.agent_y + dy

            if not self._in_grid(new_x, new_y):
                self.complete_action()
                return

            if self._is_wall(new_x, new_y):
                self.total_actions += 1
                if self.total_actions > self.level_max_actions:
                    self._lose()
                    self.complete_action()
                    return
                self._shared_post_action()
                return

            if self._is_solid_blink(new_x, new_y):
                self.total_actions += 1
                if self.total_actions > self.level_max_actions:
                    self._lose()
                    self.complete_action()
                    return
                self._shared_post_action()
                return

            if (new_x, new_y) in self.death_cells:
                self._lose()
                self.complete_action()
                return

            if self._is_wave(new_x, new_y):
                self._lose()
                self.complete_action()
                return

            self._drop_trail(self.agent_x, self.agent_y)

            self.agent_x = new_x
            self.agent_y = new_y
            if self.agent_sprite is not None:
                self.agent_sprite.set_position(new_x, new_y)

            self.total_actions += 1

            if self.total_actions > self.level_max_actions:
                self._lose()
                self.complete_action()
                return

            self._check_tile_effects()

            self._shared_post_action()
            return

        elif self.action.id == GameAction.ACTION6:
            self._save_undo_state()
            click_x = self.action.data.get("x", 0)
            click_y = self.action.data.get("y", 0)

            coords = self.camera.display_to_grid(click_x, click_y)
            if coords is None:
                self._shared_post_action()
                return

            tx, ty = coords

            if self._is_wall(tx, ty):
                self.complete_action()
                return

            if self._is_solid_blink(tx, ty):
                self.complete_action()
                return

            if (tx, ty) in self.death_cells:
                self._lose()
                self.complete_action()
                return

            self.click_count += 1
            self.total_actions += 1

            if self.click_count > self.level_max_clicks:
                self._lose()
                self.complete_action()
                return

            if self.total_actions > self.level_max_actions:
                self._lose()
                self.complete_action()
                return

            if self._is_wave(tx, ty):
                self._lose()
                self.complete_action()
                return

            self._drop_trail(self.agent_x, self.agent_y)

            self.agent_x = tx
            self.agent_y = ty
            if self.agent_sprite is not None:
                self.agent_sprite.set_position(tx, ty)

            self._check_tile_effects()

            self._shared_post_action()
            return

        elif self.action.id == GameAction.ACTION7:
            if self._apply_undo():
                self.total_actions += 1
                if self.total_actions > self.level_max_actions:
                    self._lose()
                self._update_hud_bar()
                self._update_hud_dots()
            self.complete_action()
            return

        else:
            self.complete_action()
            return


class PuzzleEnvironment:
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

    ACTION_MAP = {
        "up": 1,
        "down": 2,
        "left": 3,
        "right": 4,
        "click": 6,
        "undo": 7,
        "reset": 0,
    }

    def __init__(self, seed: int = 0) -> None:
        self._engine = Bw01(seed=seed)
        self._total_turns = 0
        self._last_action_was_reset = False
        self._prev_score = 0

    def _build_text_obs(self) -> str:
        e = self._engine
        gw = e.grid_w
        gh = e.grid_h
        ox = e.offset_x
        oy = e.offset_y

        grid = [[" " for _ in range(gw)] for _ in range(gh)]

        mw_positions = {(mw["x"], mw["y"]) for mw in e.moving_walls}
        static_walls = e.wall_cells - mw_positions

        for wx, wy in static_walls:
            gx, gy = wx - ox, wy - oy
            if 0 <= gx < gw and 0 <= gy < gh:
                grid[gy][gx] = "#"

        for wx, wy in mw_positions:
            gx, gy = wx - ox, wy - oy
            if 0 <= gx < gw and 0 <= gy < gh:
                grid[gy][gx] = "V"

        for wx, wy in e.death_cells:
            gx, gy = wx - ox, wy - oy
            if 0 <= gx < gw and 0 <= gy < gh:
                grid[gy][gx] = "D"

        source_set = set(e.sources)
        for wx, wy in source_set:
            gx, gy = wx - ox, wy - oy
            if 0 <= gx < gw and 0 <= gy < gh:
                grid[gy][gx] = "S"

        for wx, wy in e.wave_cells:
            if (wx, wy) not in source_set:
                gx, gy = wx - ox, wy - oy
                if 0 <= gx < gw and 0 <= gy < gh:
                    if grid[gy][gx] == " ":
                        grid[gy][gx] = "~"

        for wx, wy in e.reverse_tile_cells:
            gx, gy = wx - ox, wy - oy
            if 0 <= gx < gw and 0 <= gy < gh:
                if grid[gy][gx] == " ":
                    grid[gy][gx] = "R"

        for wx, wy in e.blink_tile_positions:
            gx, gy = wx - ox, wy - oy
            if 0 <= gx < gw and 0 <= gy < gh:
                if grid[gy][gx] == " ":
                    grid[gy][gx] = "O" if e.blink_solid else "o"

        for wx, wy in e.speed_tile_cells:
            gx, gy = wx - ox, wy - oy
            if 0 <= gx < gw and 0 <= gy < gh:
                if grid[gy][gx] == " ":
                    grid[gy][gx] = "M"

        for wx, wy in e.trail_sprites:
            gx, gy = wx - ox, wy - oy
            if 0 <= gx < gw and 0 <= gy < gh:
                if grid[gy][gx] == " ":
                    grid[gy][gx] = "."

        gx, gy = e.goal_x - ox, e.goal_y - oy
        if 0 <= gx < gw and 0 <= gy < gh:
            grid[gy][gx] = "G"

        ax, ay = e.agent_x - ox, e.agent_y - oy
        if 0 <= ax < gw and 0 <= ay < gh:
            grid[ay][ax] = "@"

        grid_text = "\n".join("".join(row) for row in grid)

        remaining = max(0, e.level_max_actions - e.total_actions)
        header = (
            f"Level:{e.level_index + 1} Lives:{e.lives} "
            f"Actions:{remaining}/{e.level_max_actions} "
            f"Clicks:{e.click_count}/{e.level_max_clicks} "
            f"Reverse:{e.reverse_turns}"
        )
        return header + "\n" + grid_text

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
        h, w = index_grid.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for idx, color in enumerate(self.ARC_PALETTE):
            mask = index_grid == idx
            rgb[mask] = color
        return self._encode_png(rgb)

    def _build_game_state(
        self, done: bool = False, info: Optional[Dict] = None
    ) -> GameState:
        e = self._engine
        valid_actions = self.get_actions() if not done else None
        return GameState(
            text_observation=self._build_text_obs(),
            image_observation=self._build_image_bytes(),
            valid_actions=valid_actions,
            turn=self._total_turns,
            metadata={
                "total_levels": len(self._engine._levels),
                "level_index": e.level_index,
                "lives": e.lives,
                "total_actions": e.total_actions,
                "level_max_actions": e.level_max_actions,
                "click_count": e.click_count,
                "level_max_clicks": e.level_max_clicks,
                "reverse_turns": e.reverse_turns,
                "blink_solid": e.blink_solid,
                "game_over": e.game_over,
                "done": done,
                "info": info or {},
                "levels_completed": getattr(self._engine, "_score", 0),
            },
        )

    def reset(self) -> GameState:
        e = self._engine
        game_won = getattr(e._state, "name", "") == "WIN"
        if self._last_action_was_reset or game_won:
            e.full_reset()
        else:
            e.level_reset()
        self._total_turns = 0
        self._prev_score = e._score
        self._last_action_was_reset = True
        return self._build_game_state()

    def get_actions(self) -> List[str]:
        if self.is_done():
            return ["reset"]
        return ["reset", "up", "down", "left", "right", "click", "undo"]

    def is_done(self) -> bool:
        e = self._engine
        state_name = getattr(e._state, "name", "")
        return state_name in ("WIN", "GAME_OVER")

    def step(self, action: str) -> StepResult:
        e = self._engine

        if action == "reset":
            state = self.reset()
            return StepResult(
                state=state, reward=0.0, done=False, info={"action": "reset"}
            )

        base_action = action.split()[0] if action else action
        if base_action not in self.ACTION_MAP:
            raise ValueError(
                f"Invalid action '{action}'. Must be one of {list(self.ACTION_MAP.keys())}"
            )

        self._last_action_was_reset = False
        self._total_turns += 1

        game_action_id = self.ACTION_MAP[base_action]
        action_map = {
            1: GameAction.ACTION1,
            2: GameAction.ACTION2,
            3: GameAction.ACTION3,
            4: GameAction.ACTION4,
            6: GameAction.ACTION6,
            7: GameAction.ACTION7,
        }
        game_action = action_map[game_action_id]
        info: Dict = {"action": action}

        if game_action == GameAction.ACTION6:
            parts = action.split()
            cx = int(parts[1]) if len(parts) > 1 else 0
            cy = int(parts[2]) if len(parts) > 2 else 0
            action_input = ActionInput(id=game_action, data={"x": cx, "y": cy})
        else:
            action_input = ActionInput(id=game_action)

        prev_score = self._prev_score
        frame = e.perform_action(action_input, raw=True)
        levels_advanced = frame.levels_completed - prev_score
        self._prev_score = frame.levels_completed

        reward = levels_advanced * (1.0 / len(e._levels))

        done = frame.state.name == "WIN" or frame.state.name == "GAME_OVER"

        if done:
            if frame.state.name == "WIN":
                info["reason"] = "game_complete"
            else:
                info["reason"] = "game_over"
            return StepResult(
                state=self._build_game_state(done=True, info=info),
                reward=reward,
                done=True,
                info=info,
            )

        return StepResult(
            state=self._build_game_state(done=False, info=info),
            reward=reward,
            done=False,
            info=info,
        )

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError(f"Unsupported render mode: {mode}")
        e = self._engine
        index_grid = e.camera.render(e.current_level.get_sprites())
        h, w = index_grid.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for idx, color in enumerate(self.ARC_PALETTE):
            mask = index_grid == idx
            rgb[mask] = color
        if h != 64 or w != 64:
            row_idx = (np.arange(64) * h // 64).astype(int)
            col_idx = (np.arange(64) * w // 64).astype(int)
            rgb = rgb[np.ix_(row_idx, col_idx)]
        return rgb

    def close(self) -> None:
        self._engine = None


class ArcGameEnv(gym.Env):
    metadata: Dict[str, Any] = {
        "render_modes": ["rgb_array"],
        "render_fps": 5,
    }

    ACTION_LIST: List[str] = ["reset", "up", "down", "left", "right", "click", "undo"]

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
