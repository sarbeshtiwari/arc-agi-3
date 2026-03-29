from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import random
import struct
import zlib

import numpy as np
import gymnasium as gym
from gymnasium import spaces
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


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
CYAN = 8
MAROON = 9
WHITE = 11
PURPLE = 15

BACKGROUND_COLOR = BLACK
PADDING_COLOR = GREY

_SPRITES = {
    "player": Sprite(
        pixels=[[YELLOW]],
        name="player",
        visible=True,
        collidable=False,
        tags=["player"],
        layer=5,
    ),
    "wall": Sprite(
        pixels=[[GREY]],
        name="wall",
        visible=True,
        collidable=True,
        tags=["wall"],
        layer=0,
    ),
    "red_phase_wall": Sprite(
        pixels=[[RED]],
        name="red_phase_wall",
        visible=True,
        collidable=True,
        tags=["phase_wall", "phase_wall_red"],
        layer=1,
    ),
    "blue_phase_wall": Sprite(
        pixels=[[BLUE]],
        name="blue_phase_wall",
        visible=True,
        collidable=True,
        tags=["phase_wall", "phase_wall_blue"],
        layer=1,
    ),
    "green_phase_wall": Sprite(
        pixels=[[GREEN]],
        name="green_phase_wall",
        visible=True,
        collidable=True,
        tags=["phase_wall", "phase_wall_green"],
        layer=1,
    ),
    "red_switch": Sprite(
        pixels=[[MAROON]],
        name="red_switch",
        visible=True,
        collidable=False,
        tags=["switch", "switch_red"],
        layer=-1,
    ),
    "blue_switch": Sprite(
        pixels=[[CYAN]],
        name="blue_switch",
        visible=True,
        collidable=False,
        tags=["switch", "switch_blue"],
        layer=-1,
    ),
    "green_switch": Sprite(
        pixels=[[WHITE]],
        name="green_switch",
        visible=True,
        collidable=False,
        tags=["switch", "switch_green"],
        layer=-1,
    ),
    "death_switch_red": Sprite(
        pixels=[[MAROON]],
        name="death_switch_red",
        visible=True,
        collidable=False,
        tags=["death_switch", "death_switch_red"],
        layer=-1,
    ),
    "death_switch_blue": Sprite(
        pixels=[[CYAN]],
        name="death_switch_blue",
        visible=True,
        collidable=False,
        tags=["death_switch", "death_switch_blue"],
        layer=-1,
    ),
    "death_switch_green": Sprite(
        pixels=[[WHITE]],
        name="death_switch_green",
        visible=True,
        collidable=False,
        tags=["death_switch", "death_switch_green"],
        layer=-1,
    ),
    "enemy": Sprite(
        pixels=[[ORANGE]],
        name="enemy",
        visible=True,
        collidable=False,
        tags=["enemy"],
        layer=4,
    ),
    "spike": Sprite(
        pixels=[[MAGENTA]],
        name="spike",
        visible=True,
        collidable=False,
        tags=["spike"],
        layer=-1,
    ),
    "exit": Sprite(
        pixels=[[YELLOW]],
        name="exit",
        visible=True,
        collidable=False,
        tags=["exit"],
        layer=-1,
    ),
    "fake_exit": Sprite(
        pixels=[[YELLOW]],
        name="fake_exit",
        visible=True,
        collidable=False,
        tags=["fake_exit"],
        layer=-1,
    ),
    "restart_exit": Sprite(
        pixels=[[GREY]],
        name="restart_exit",
        visible=True,
        collidable=False,
        tags=["restart_exit"],
        layer=-1,
    ),
    "master": Sprite(
        pixels=[[GREY]],
        name="master",
        visible=True,
        collidable=False,
        tags=["master"],
        layer=-2,
    ),
    "death_overlay": Sprite(
        pixels=[[RED]],
        name="death_overlay",
        visible=False,
        collidable=False,
        tags=["death_overlay"],
        layer=10,
    ),
}

_MAP_1 = [
    "####################",
    "#........#.........#",
    "#.P......#.........#",
    "#........#.........#",
    "#........#.........#",
    "#........R.........#",
    "#........R.........#",
    "#........#.........#",
    "#........#.........#",
    "#..r.....#.........#",
    "#........#.........#",
    "#........R.........#",
    "#........R.........#",
    "#........#.........#",
    "#........#.........#",
    "#........#.........#",
    "#........#.........#",
    "#........#......X..#",
    "#........#.........#",
    "####################",
]

_MAP_2 = [
    "####################",
    "#.P......#....#....#",
    "#........#....#....#",
    "#........#....#....#",
    "#..b.....#....#....#",
    "#........#....#....#",
    "#........#....#....#",
    "#BBBBBBBB#....#....#",
    "#........#....#....#",
    "#........R.........#",
    "#........R....#....#",
    "#........#....#....#",
    "#...r....#######.###",
    "#........#....#....#",
    "#........#....#....#",
    "#........#....#....#",
    "#........#....#....#",
    "#........#....#.X..#",
    "#........#....#....#",
    "####################",
]

_MAP_3 = [
    "####################",
    "#.P......#....#....#",
    "#........#....#....#",
    "#........#....#....#",
    "#........#....#....#",
    "#..r.....#....#....#",
    "#........#....#....#",
    "#........#....#....#",
    "#........#....#....#",
    "#........R.........#",
    "#........R....#....#",
    "#........#....#....#",
    "#........######.####",
    "#........#....#....#",
    "#........#....#E...#",
    "#........#....#....#",
    "#........#....#....#",
    "#........#....#.X..#",
    "#........#....#....#",
    "####################",
]

_MAP_4 = [
    "####################",
    "#.P......#....#....#",
    "#........#....#....#",
    "#........#....#....#",
    "#..b.....#....#....#",
    "#........#....#....#",
    "#.S......#....#....#",
    "#BBBBBBBB#....#....#",
    "#........#....#....#",
    "#........R.........#",
    "#........R....#....#",
    "#........#....#....#",
    "#...r....#..S.#....#",
    "#.S......######.####",
    "#........#....#.S..#",
    "#.....S..#..2.#....#",
    "#........#..S.#....#",
    "#........#....#.X..#",
    "#........#....#....#",
    "####################",
]

_MAP_5 = [
    "####################",
    "#.P......#....#....#",
    "#........#....#....#",
    "#........R..2.#....#",
    "#........#....#....#",
    "#........#..b.#....#",
    "#..r.....#.........#",
    "#........#....#....#",
    "#BBBBBBBB#....#....#",
    "#........#....#....#",
    "#........#....#....#",
    "#........G....#....#",
    "#........#######.###",
    "#..g.....#....#....#",
    "#........#....#....#",
    "#GGGGGGGG#....#3...#",
    "#........#....#....#",
    "#........#....#.X..#",
    "#........#....#....#",
    "####################",
]

_MAP_6 = [
    "####################",
    "#.P......#....#....#",
    "#........#....#....#",
    "#........R....#....#",
    "#..S.....#..2.#....#",
    "#........#..b.#....#",
    "#..r.....#....#....#",
    "#........#....#....#",
    "#BBBBBBBB#...E#....#",
    "#........#....#....#",
    "#........#....#....#",
    "#........G.........#",
    "#.S......#..S.#....#",
    "#..g.....#######.###",
    "#........#....#....#",
    "#GGGGGGGG#....#....#",
    "#........#....#....#",
    "#........#....#.X..#",
    "#........#....#....#",
    "####################",
]

_MAP_7 = [
    "####################",
    "#.P......#....#....#",
    "#.......M#....#....#",
    "#........R....#....#",
    "#..S.....#..S.#....#",
    "#........#..b.#.2..#",
    "#..r..1..#....#....#",
    "#........#....#....#",
    "#BBBBBBBB#...E#....#",
    "#........#....#....#",
    "#........#....#....#",
    "#........G.........#",
    "#.S......#..S.#....#",
    "#..g.....#...3#....#",
    "#........#######.###",
    "#GGGGGGGG#....#....#",
    "#........#.........#",
    "#........#..E.#.F..#",
    "#........#.Z..#....#",
    "####################",
]

_TILE_LOOKUP = {
    "#": "wall",
    "P": "player",
    "X": "exit",
    "Z": "restart_exit",
    "F": "fake_exit",
    "R": "red_phase_wall",
    "B": "blue_phase_wall",
    "G": "green_phase_wall",
    "r": "red_switch",
    "b": "blue_switch",
    "g": "green_switch",
    "1": "death_switch_red",
    "2": "death_switch_blue",
    "3": "death_switch_green",
    "E": "enemy",
    "S": "spike",
    "M": "master",
}


def _build_level(tile_map, data):
    sprite_list = []
    height = len(tile_map)
    width = len(tile_map[0]) if height > 0 else 0
    for y in range(height):
        for x in range(width):
            char = tile_map[y][x]
            if char == "P":
                data["origin_x"] = x
                data["origin_y"] = y
            if char in _TILE_LOOKUP:
                sprite_list.append(
                    _SPRITES[_TILE_LOOKUP[char]].clone().set_position(x, y)
                )
    sprite_list.append(_SPRITES["death_overlay"].clone().set_position(0, 0))
    return Level(sprites=sprite_list, grid_size=(width, height), data=data)


levels = [
    _build_level(
        _MAP_1,
        {
            "phase_red": True,
            "phase_blue": False,
            "phase_green": False,
            "enemy_paths": [],
            "reversed_controls": False,
            "static_walls": False,
            "max_moves": 114,
        },
    ),
    _build_level(
        _MAP_2,
        {
            "phase_red": True,
            "phase_blue": True,
            "phase_green": False,
            "enemy_paths": [],
            "reversed_controls": False,
            "static_walls": False,
            "max_moves": 126,
        },
    ),
    _build_level(
        _MAP_3,
        {
            "phase_red": True,
            "phase_blue": False,
            "phase_green": False,
            "enemy_paths": [{"sx": 16, "sy": 13, "ex": 16, "ey": 17}],
            "reversed_controls": False,
            "static_walls": False,
            "max_moves": 135,
        },
    ),
    _build_level(
        _MAP_4,
        {
            "phase_red": True,
            "phase_blue": True,
            "phase_green": False,
            "enemy_paths": [],
            "reversed_controls": False,
            "static_walls": False,
            "max_moves": 165,
        },
    ),
    _build_level(
        _MAP_5,
        {
            "phase_red": True,
            "phase_blue": True,
            "phase_green": True,
            "enemy_paths": [],
            "reversed_controls": False,
            "static_walls": True,
            "max_moves": 240,
        },
    ),
    _build_level(
        _MAP_6,
        {
            "phase_red": True,
            "phase_blue": True,
            "phase_green": True,
            "enemy_paths": [{"sx": 13, "sy": 2, "ex": 13, "ey": 8}],
            "reversed_controls": True,
            "static_walls": True,
            "max_moves": 300,
        },
    ),
    _build_level(
        _MAP_7,
        {
            "phase_red": True,
            "phase_blue": True,
            "phase_green": True,
            "enemy_paths": [
                {"sx": 13, "sy": 2, "ex": 13, "ey": 8},
                {"sx": 12, "sy": 15, "ex": 12, "ey": 18},
            ],
            "reversed_controls": True,
            "static_walls": True,
            "max_moves": 360,
        },
    ),
]


class GameHUD(RenderableUserDisplay):
    def __init__(self, game: "Mx47"):
        self._game = game

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        game = self._game

        frame_h, frame_w = frame.shape
        cam_w, cam_h = 20, 20
        scale = min(frame_w // cam_w, frame_h // cam_h)
        x_off = (frame_w - cam_w * scale) // 2
        y_off = (frame_h - cam_h * scale) // 2

        def grid_y(row):
            return slice(y_off + row * scale, y_off + (row + 1) * scale)

        def grid_x(col):
            return slice(x_off + col * scale, x_off + (col + 1) * scale)

        frame[grid_y(0), grid_x(14)] = (
            PURPLE if getattr(game, "_reversed_controls", False) else BLACK
        )

        frame[grid_y(0), grid_x(15)] = RED if game._red_phase_active else BLACK
        frame[grid_y(0), grid_x(16)] = BLUE if game._blue_phase_active else BLACK
        frame[grid_y(0), grid_x(17)] = GREEN if game._green_phase_active else BLACK

        if game._master_active:
            frame[grid_y(0), grid_x(18)] = PURPLE

        for i in range(3):
            frame[grid_y(0), grid_x(1 + i)] = YELLOW if game._lives > i else BLACK

        if game._max_moves > 0:
            remaining = max(0, game._max_moves - game._moves_used)
            ratio = remaining / game._max_moves
            cells_filled = int(ratio * cam_w)
            bar_row = grid_y(19)
            for col in range(cam_w):
                frame[bar_row, grid_x(col)] = GREEN if col < cells_filled else BLACK

        if game._death_flash_timer > 0:
            for col in range(cam_w):
                frame[grid_y(0), grid_x(col)] = RED
                frame[grid_y(cam_h - 1), grid_x(col)] = RED
            return frame

        return frame


class Mx47(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._red_phase_active = False
        self._blue_phase_active = False
        self._green_phase_active = False
        self._master_active = False
        self._lives = 3
        self._death_flash_timer = 0
        self._enemy_list = []

        self._reversed_controls = False
        self._static_walls = False

        self._moves_used = 0
        self._max_moves = 0

        self._hud = GameHUD(self)
        self._spawn_x = 0
        self._spawn_y = 0

        self._game_over = False
        self._rng = random.Random(seed)
        self._consecutive_resets = 0
        self._history: List[Dict] = []

        super().__init__(
            "mx47",
            levels,
            Camera(0, 0, 20, 20, BACKGROUND_COLOR, PADDING_COLOR, [self._hud]),
            available_actions=[0, 1, 2, 3, 4, 7],
        )

    def on_set_level(self, level: Level) -> None:
        self._player = self.current_level.get_sprites_by_tag("player")[0]

        origin_x = self.current_level.get_data("origin_x")
        origin_y = self.current_level.get_data("origin_y")
        self._player.set_position(origin_x, origin_y)
        self._spawn_x = origin_x
        self._spawn_y = origin_y

        self._red_phase_active = self.current_level.get_data("phase_red")
        self._blue_phase_active = self.current_level.get_data("phase_blue")
        self._green_phase_active = self.current_level.get_data("phase_green")

        self._initial_red_phase = self._red_phase_active
        self._initial_blue_phase = self._blue_phase_active
        self._initial_green_phase = self._green_phase_active

        reversed_raw = self.current_level.get_data("reversed_controls")
        static_raw = self.current_level.get_data("static_walls")
        self._reversed_controls = (
            bool(reversed_raw) if reversed_raw is not None else False
        )
        self._static_walls = bool(static_raw) if static_raw is not None else False

        max_moves_raw = self.current_level.get_data("max_moves")
        self._max_moves = int(max_moves_raw) if max_moves_raw is not None else 999
        self._moves_used = 0

        self._master_active = False
        self._death_flash_timer = 0
        self._lives = 3
        self._game_over = False
        self._history = []

        self._init_enemies()
        self._init_overlay()
        self._update_phase_walls()
        self._randomize_spawn(origin_x, origin_y)

    def _init_enemies(self) -> None:
        self._enemy_list = []
        enemies = self.current_level.get_sprites_by_tag("enemy")
        enemy_paths = self.current_level.get_data("enemy_paths")
        if enemy_paths:
            for i, enemy_sprite in enumerate(enemies):
                if i < len(enemy_paths):
                    path = enemy_paths[i]
                    enemy_sprite.set_position(path["sx"], path["sy"])
                    self._enemy_list.append(
                        {
                            "sprite": enemy_sprite,
                            "start_x": path["sx"],
                            "start_y": path["sy"],
                            "end_x": path["ex"],
                            "end_y": path["ey"],
                            "direction": 1,
                            "initial_x": path["sx"],
                            "initial_y": path["sy"],
                        }
                    )

    def _init_overlay(self) -> None:
        overlay_list = self.current_level.get_sprites_by_tag("death_overlay")
        if overlay_list:
            self._death_overlay = overlay_list[0]
            self._death_overlay.set_visible(False)
        else:
            self._death_overlay = None

    def _randomize_spawn(self, origin_x: int, origin_y: int) -> None:
        reachable = self._get_reachable_empty_tiles(origin_x, origin_y)
        if len(reachable) > 1:
            new_spawn = self._rng.choice(reachable)
            self._spawn_x = new_spawn[0]
            self._spawn_y = new_spawn[1]
            self._player.set_position(self._spawn_x, self._spawn_y)

    def _update_phase_walls(self) -> None:
        for sprite in self.current_level.get_sprites_by_tag("phase_wall_red"):
            if self._static_walls:
                sprite.set_visible(True)
            else:
                sprite.set_visible(self._red_phase_active and not self._master_active)

        for sprite in self.current_level.get_sprites_by_tag("phase_wall_blue"):
            if self._static_walls:
                sprite.set_visible(True)
            else:
                sprite.set_visible(self._blue_phase_active and not self._master_active)

        for sprite in self.current_level.get_sprites_by_tag("phase_wall_green"):
            if self._static_walls:
                sprite.set_visible(True)
            else:
                sprite.set_visible(self._green_phase_active and not self._master_active)

    def _find_sprite_at(self, x: int, y: int, tag: str):
        for sprite in self.current_level.get_sprites_by_tag(tag):
            if sprite.x == x and sprite.y == y:
                return sprite
        return None

    def _is_blocked(self, x: int, y: int) -> bool:
        if self._find_sprite_at(x, y, "wall"):
            return True

        phase_wall = self._find_sprite_at(x, y, "phase_wall")
        if phase_wall:
            tags = phase_wall.tags or []
            if "phase_wall_red" in tags:
                return self._red_phase_active and not self._master_active
            if "phase_wall_blue" in tags:
                return self._blue_phase_active and not self._master_active
            if "phase_wall_green" in tags:
                return self._green_phase_active and not self._master_active

        return False

    def _toggle_switch(self, switch) -> None:
        tags = switch.tags or []
        if "switch_red" in tags:
            self._red_phase_active = not self._red_phase_active
        elif "switch_blue" in tags:
            self._blue_phase_active = not self._blue_phase_active
        elif "switch_green" in tags:
            self._green_phase_active = not self._green_phase_active
        self._update_phase_walls()

    def _activate_master(self) -> None:
        self._master_active = True
        self._update_phase_walls()
        for master_sprite in self.current_level.get_sprites_by_tag("master"):
            master_sprite.set_visible(False)

    def _move_enemy(self, enemy: dict) -> None:
        if enemy["direction"] == 1:
            target_x, target_y = enemy["end_x"], enemy["end_y"]
        else:
            target_x, target_y = enemy["start_x"], enemy["start_y"]

        dx = dy = 0
        if target_y != enemy["sprite"].y:
            dy = 1 if target_y > enemy["sprite"].y else -1
        elif target_x != enemy["sprite"].x:
            dx = 1 if target_x > enemy["sprite"].x else -1

        enemy["sprite"].set_position(enemy["sprite"].x + dx, enemy["sprite"].y + dy)

        if enemy["sprite"].x == target_x and enemy["sprite"].y == target_y:
            enemy["direction"] *= -1

    def _handle_death(self) -> None:
        self._lives -= 1
        if self._lives <= 0:
            self._game_over = True
            self.lose()
            return

        self._death_flash_timer = 2

        self._player.set_position(self._spawn_x, self._spawn_y)

        self._red_phase_active = self._initial_red_phase
        self._blue_phase_active = self._initial_blue_phase
        self._green_phase_active = self._initial_green_phase
        self._master_active = False

        self._moves_used = 0

        for master_sprite in self.current_level.get_sprites_by_tag("master"):
            master_sprite.set_visible(True)

        for enemy in self._enemy_list:
            enemy["sprite"].set_position(enemy["initial_x"], enemy["initial_y"])
            if (
                enemy["initial_x"] == enemy["end_x"]
                and enemy["initial_y"] == enemy["end_y"]
            ):
                enemy["direction"] = -1
            else:
                enemy["direction"] = 1

        self._update_phase_walls()

    def _get_reachable_empty_tiles(
        self, start_x: int, start_y: int
    ) -> List[Tuple[int, int]]:
        blocked: Set[Tuple[int, int]] = set()
        for tag in ("wall", "phase_wall_red", "phase_wall_blue", "phase_wall_green"):
            for sprite in self.current_level.get_sprites_by_tag(tag):
                if tag == "wall" or sprite.is_visible:
                    blocked.add((sprite.x, sprite.y))

        hazard: Set[Tuple[int, int]] = set()
        for tag in ("spike", "death_switch", "fake_exit"):
            for sprite in self.current_level.get_sprites_by_tag(tag):
                hazard.add((sprite.x, sprite.y))
        for enemy in self._enemy_list:
            hazard.add((enemy["sprite"].x, enemy["sprite"].y))

        visited: Set[Tuple[int, int]] = {(start_x, start_y)}
        q: deque[Tuple[int, int]] = deque([(start_x, start_y)])
        reachable: List[Tuple[int, int]] = []

        while q:
            cx, cy = q.popleft()
            if (cx, cy) not in hazard:
                reachable.append((cx, cy))
            for ddx, ddy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                nx, ny = cx + ddx, cy + ddy
                if (nx, ny) not in visited and (nx, ny) not in blocked:
                    visited.add((nx, ny))
                    q.append((nx, ny))

        return reachable

    def _check_hazards_at(self, x: int, y: int) -> bool:
        if self._find_sprite_at(x, y, "spike"):
            self._handle_death()
            return True
        for enemy in self._enemy_list:
            if enemy["sprite"].x == x and enemy["sprite"].y == y:
                self._handle_death()
                return True
        if self._find_sprite_at(x, y, "death_switch"):
            self._handle_death()
            return True
        if self._find_sprite_at(x, y, "fake_exit"):
            self._handle_death()
            return True
        return False

    def _check_exits_at(self, x: int, y: int) -> bool:
        if self._find_sprite_at(x, y, "restart_exit"):
            self.next_level()
            return True
        if self._find_sprite_at(x, y, "exit"):
            self.next_level()
            return True
        return False

    def _save_state(self) -> None:
        enemies = [
            {"x": e["sprite"].x, "y": e["sprite"].y, "dir": e["direction"]}
            for e in self._enemy_list
        ]
        self._history.append(
            {
                "px": self._player.x,
                "py": self._player.y,
                "red": self._red_phase_active,
                "blue": self._blue_phase_active,
                "green": self._green_phase_active,
                "master": self._master_active,
                "enemies": enemies,
            }
        )

    def _undo(self) -> None:
        if not self._history:
            return
        snap = self._history.pop()
        self._player.set_position(snap["px"], snap["py"])
        self._red_phase_active = snap["red"]
        self._blue_phase_active = snap["blue"]
        self._green_phase_active = snap["green"]
        self._master_active = snap["master"]
        for i, e in enumerate(self._enemy_list):
            if i < len(snap["enemies"]):
                es = snap["enemies"][i]
                e["sprite"].set_position(es["x"], es["y"])
                e["direction"] = es["dir"]
        self._update_phase_walls()
        if self._master_active:
            for s in self.current_level.get_sprites_by_tag("master"):
                s.set_visible(False)
        else:
            for s in self.current_level.get_sprites_by_tag("master"):
                s.set_visible(True)

    def _parse_direction(self) -> Tuple[int, int]:
        dx = dy = 0
        if self.action.id == GameAction.ACTION1:
            dy = 1 if self._reversed_controls else -1
        elif self.action.id == GameAction.ACTION2:
            dy = -1 if self._reversed_controls else 1
        elif self.action.id == GameAction.ACTION3:
            dx = 1 if self._reversed_controls else -1
        elif self.action.id == GameAction.ACTION4:
            dx = -1 if self._reversed_controls else 1
        return dx, dy

    def _process_move(self, new_x: int, new_y: int) -> None:
        self._player.set_position(new_x, new_y)

        switch = self._find_sprite_at(new_x, new_y, "switch")
        if switch:
            self._toggle_switch(switch)

        master = self._find_sprite_at(new_x, new_y, "master")
        if master and master.is_visible:
            self._activate_master()

        if self._check_hazards_at(new_x, new_y):
            return

        if self._check_exits_at(new_x, new_y):
            return

        for enemy in self._enemy_list:
            self._move_enemy(enemy)

        for enemy in self._enemy_list:
            if (
                enemy["sprite"].x == self._player.x
                and enemy["sprite"].y == self._player.y
            ):
                self._handle_death()
                return

    def step(self) -> None:
        if self._game_over:
            self.complete_action()
            return

        if self._death_flash_timer > 0:
            self._death_flash_timer -= 1
            self.complete_action()
            return

        if self.action.id == GameAction.ACTION7:
            self._consecutive_resets = 0
            self._undo()
            self._moves_used += 1
            if self._max_moves > 0 and self._moves_used > self._max_moves:
                self._handle_death()
            self.complete_action()
            return

        dx, dy = self._parse_direction()

        if dx == 0 and dy == 0:
            self.complete_action()
            return

        self._consecutive_resets = 0
        self._save_state()

        self._moves_used += 1
        if self._max_moves > 0 and self._moves_used > self._max_moves:
            self._handle_death()
            self.complete_action()
            return

        new_x = self._player.x + dx
        new_y = self._player.y + dy

        if self._is_blocked(new_x, new_y):
            self.complete_action()
            return

        self._process_move(new_x, new_y)
        self.complete_action()


class PuzzleEnvironment:
    ACTION_MAP: Dict[str, int] = {
        "reset": 0,
        "up": 1,
        "down": 2,
        "left": 3,
        "right": 4,
        "undo": 7,
    }

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

    _TAG_TO_CHAR: Dict[str, str] = {
        "wall": "#",
        "player": "@",
        "exit": "X",
        "fake_exit": "F",
        "restart_exit": "Z",
        "phase_wall_red": "R",
        "phase_wall_blue": "B",
        "phase_wall_green": "G",
        "switch_red": "r",
        "switch_blue": "b",
        "switch_green": "g",
        "death_switch_red": "1",
        "death_switch_blue": "2",
        "death_switch_green": "3",
        "enemy": "E",
        "spike": "S",
        "master": "M",
        "death_overlay": "",
    }

    _TAG_PRIORITY: Dict[str, int] = {
        "death_overlay": -1,
        "wall": 0,
        "phase_wall_red": 1,
        "phase_wall_blue": 1,
        "phase_wall_green": 1,
        "switch_red": 2,
        "switch_blue": 2,
        "switch_green": 2,
        "death_switch_red": 2,
        "death_switch_blue": 2,
        "death_switch_green": 2,
        "spike": 2,
        "exit": 3,
        "fake_exit": 3,
        "restart_exit": 3,
        "master": 3,
        "enemy": 4,
        "player": 5,
    }

    def __init__(self, seed: int = 0) -> None:
        self._engine = Mx47(seed=seed)
        self._total_turns = 0
        self._last_action_was_reset = False
        self._game_won = False

    def _build_text_obs(self) -> str:
        e = self._engine
        gs = e.current_level.grid_size
        w, h = (gs[0] if gs else 20), (gs[1] if gs else 20)
        text_grid: List[List[str]] = [["." for _ in range(w)] for _ in range(h)]
        prio_grid: List[List[int]] = [[-2 for _ in range(w)] for _ in range(h)]
        tag_char = self._TAG_TO_CHAR
        tag_prio = self._TAG_PRIORITY
        for sprite in e.current_level.get_sprites():
            if not sprite.is_visible:
                continue
            sx, sy = sprite.x, sprite.y
            if 0 <= sx < w and 0 <= sy < h:
                for tag in sprite.tags or []:
                    ch = tag_char.get(tag, "")
                    pr = tag_prio.get(tag, -1)
                    if ch and pr > prio_grid[sy][sx]:
                        text_grid[sy][sx] = ch
                        prio_grid[sy][sx] = pr
        grid_text = "\n".join("".join(row) for row in text_grid)

        phase_info = []
        if e._red_phase_active:
            phase_info.append("red=ON")
        if e._blue_phase_active:
            phase_info.append("blue=ON")
        if e._green_phase_active:
            phase_info.append("green=ON")
        if e._master_active:
            phase_info.append("master=ON")
        if e._reversed_controls:
            phase_info.append("controls=REVERSED")

        remaining = max(0, e._max_moves - e._moves_used)
        header = (
            f"Level:{e.level_index + 1} Lives:{e._lives} "
            f"Moves:{remaining}/{e._max_moves} " + " ".join(phase_info)
        )
        return header.strip() + "\n" + grid_text

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
        for idx, color in enumerate(self.ARC_PALETTE):
            mask = arr == idx
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
                "lives": e._lives,
                "max_moves": e._max_moves,
                "moves_used": e._moves_used,
                "red_phase": e._red_phase_active,
                "blue_phase": e._blue_phase_active,
                "green_phase": e._green_phase_active,
                "master_active": e._master_active,
                "game_over": e._game_over,
                "done": done,
                "info": info or {},
            },
        )

    def reset(self) -> GameState:
        e = self._engine
        reset_input = ActionInput(id=GameAction.RESET)
        if self._game_won or self._last_action_was_reset:
            e.perform_action(reset_input)
            e.perform_action(reset_input)
        else:
            e.perform_action(reset_input)
        self._total_turns = 0
        self._last_action_was_reset = True
        self._game_won = False
        return self._build_game_state()

    def get_actions(self) -> List[str]:
        if self._engine._game_over:
            return ["reset"]
        return ["reset", "up", "down", "left", "right", "undo"]

    def is_done(self) -> bool:
        return self._engine._game_over or self._game_won

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
            7: GameAction.ACTION7,
        }
        game_action = action_map[game_action_id]
        info: Dict = {"action": action}

        total_levels = len(levels)
        level_before = e.level_index

        action_input = ActionInput(id=game_action)
        frame = e.perform_action(action_input, raw=True)

        game_won = frame and frame.state and frame.state.name == "WIN"
        level_advanced = e.level_index > level_before
        done = e._game_over or game_won

        reward = 0.0
        if game_won or level_advanced:
            reward = 1.0 / total_levels

        if game_won:
            self._game_won = True
            info["reason"] = "game_complete"
            info["level_index"] = e.level_index
            info["total_levels"] = total_levels
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
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        for idx, color in enumerate(self.ARC_PALETTE):
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
        "up",
        "down",
        "left",
        "right",
        "undo",
        "reset",
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
        return img[row_idx[:, None], col_idx[None, :]]

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
    check_env(env, skip_render_check=False)
    obs, info = env.reset()
    obs, reward, term, trunc, info = env.step(0)
    env.render()
    env.close()
