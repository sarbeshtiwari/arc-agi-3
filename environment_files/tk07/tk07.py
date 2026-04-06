import random
import struct
import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from arcengine import (
    ActionInput,
    ARCBaseGame,
    Camera,
    GameAction,
    Level,
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

sprites = {
    "floor": Sprite(
        pixels=[[0]],
        name="floor",
        visible=True,
        collidable=False,
        layer=0,
        tags=["floor"],
    ),
    "wall": Sprite(
        pixels=[[3]],
        name="wall",
        visible=True,
        collidable=True,
        layer=1,
        tags=["wall", "solid"],
    ),
    "player": Sprite(
        pixels=[[9]],
        name="player",
        visible=True,
        collidable=False,
        layer=3,
        tags=["player"],
    ),
    "echo": Sprite(
        pixels=[[10]],
        name="echo",
        visible=True,
        collidable=False,
        layer=2,
        tags=["echo"],
    ),
    "block": Sprite(
        pixels=[[12]],
        name="block",
        visible=True,
        collidable=False,
        layer=2,
        tags=["block", "pushable"],
    ),
    "switch_off": Sprite(
        pixels=[[4]],
        name="switch_off",
        visible=True,
        collidable=False,
        layer=1,
        tags=["switch"],
    ),
    "switch_on": Sprite(
        pixels=[[14]],
        name="switch_on",
        visible=True,
        collidable=False,
        layer=1,
        tags=["switch", "active"],
    ),
    "door_closed": Sprite(
        pixels=[[15]],
        name="door_closed",
        visible=True,
        collidable=True,
        layer=1,
        tags=["door", "solid"],
    ),
    "door_open": Sprite(
        pixels=[[1]],
        name="door_open",
        visible=True,
        collidable=False,
        layer=1,
        tags=["door_open"],
    ),
    "key": Sprite(
        pixels=[[11]],
        name="key",
        visible=True,
        collidable=False,
        layer=2,
        tags=["key"],
    ),
    "exit": Sprite(
        pixels=[[8]],
        name="exit",
        visible=True,
        collidable=False,
        layer=1,
        tags=["exit"],
    ),
    "hazard": Sprite(
        pixels=[[6]],
        name="hazard",
        visible=True,
        collidable=False,
        layer=1,
        tags=["hazard"],
    ),
    "heart": Sprite(
        pixels=[[8]],
        name="heart",
        visible=True,
        collidable=False,
        layer=4,
        tags=["heart"],
    ),
    "move_pip": Sprite(
        pixels=[[14]],
        name="move_pip",
        visible=True,
        collidable=False,
        layer=4,
        tags=["move_pip"],
    ),
}


def build_level_sprites(layout, extra_sprites=None):
    result = []
    for y, row in enumerate(layout):
        for x, cell in enumerate(row):
            result.append(sprites["floor"].clone().set_position(x, y))
            if cell == "W":
                result.append(sprites["wall"].clone().set_position(x, y))
            elif cell == "S":
                result.append(sprites["switch_off"].clone().set_position(x, y))
            elif cell == "D":
                result.append(sprites["door_closed"].clone().set_position(x, y))
            elif cell == "X":
                result.append(sprites["exit"].clone().set_position(x, y))
            elif cell == "H":
                result.append(sprites["hazard"].clone().set_position(x, y))
    if extra_sprites:
        for s in extra_sprites:
            result.append(s)
    return result


def parse_layout(ascii_str):
    return [list(line) for line in ascii_str.strip().split("\n")]


def extract_entities(layout):
    entities = []
    for y, row in enumerate(layout):
        for x, cell in enumerate(row):
            if cell == "P":
                entities.append(("player", x, y))
                layout[y][x] = "."
            elif cell == "B":
                entities.append(("block", x, y))
                layout[y][x] = "."
            elif cell == "K":
                entities.append(("key", x, y))
                layout[y][x] = "."
    return entities


def build_level(ascii_str, name, w, h):
    layout = parse_layout(ascii_str)
    entities = extract_entities(layout)
    extra = [sprites[k].clone().set_position(x, y) for k, x, y in entities]
    return Level(
        sprites=build_level_sprites(layout, extra), grid_size=(w, h), name=name
    )


level1 = build_level(
    "WWWWWWWWWWWWWWWW\n"
    "WP.............W\n"
    "W..WW.....WW...W\n"
    "W...S.....W..W.W\n"
    "W.........W....W\n"
    "W..W...S..W..W.W\n"
    "W..W.......W...W\n"
    "WWWWDWWWWWDWWWWW\n"
    "W.....W........W\n"
    "W.WW.KW...WW...W\n"
    "W.....W........W\n"
    "W..WW.W.....X..W\n"
    "W.....W........W\n"
    "W..WW.W...WW...W\n"
    "W.....W........W\n"
    "WWWWWWWWWWWWWWWW",
    name="Level 1: Two Echoes",
    w=16,
    h=16,
)

level2 = build_level(
    "WWWWWWWWWWWWWWWWWW\n"
    "WP...............W\n"
    "W.WW...WW..WW..W.W\n"
    "W.W....W...W.S.W.W\n"
    "W.W..S.W...W...W.W\n"
    "W.W....W...W.....W\n"
    "W.W....WW.H.WW..WW\n"
    "WWWDWWWWWWWWWWWWWW\n"
    "W.........WW...W.W\n"
    "W.WWH.WW..W..W.W.W\n"
    "W.W.K..W..W..W...W\n"
    "W.W....W.....WDWWW\n"
    "W.W....W..WW.W.X.W\n"
    "W.WW..WW...WWWWWWW\n"
    "W..............W.W\n"
    "W.WW..WW...WW....W\n"
    "W................W\n"
    "WWWWWWWWWWWWWWWWWW",
    name="Level 2: Hazard Maze",
    w=18,
    h=18,
)

level3 = build_level(
    "WWWWWWWWWWWWWWWWWWWW\n"
    "WP.................W\n"
    "W.WW...WW.....WW...W\n"
    "W.W....W......W..W.W\n"
    "W.W..B.W..W...WS.W.W\n"
    "W.W....WWWWK..W..W.W\n"
    "W.WW...W..H...WW...W\n"
    "W......W..WW.......W\n"
    "W.WW..WW..W..WW.W..W\n"
    "W.W....W..WH.W..W..W\n"
    "W.W....W.....W.....W\n"
    "W.W....W..WW.W.WW..W\n"
    "W.WW...WH.W..W.....W\n"
    "WWWWWWDWWWWWWWWWWWWW\n"
    "W........WW..S...W.W\n"
    "W.WWWWWW..W..WW.W..W\n"
    "W.W.X..W..W..W..K..W\n"
    "W.WWWWDW...H.W.WW..W\n"
    "W...W..........W...W\n"
    "WWWWWWWWWWWWWWWWWWWW",
    name="Level 3: Block and Switch",
    w=20,
    h=20,
)

level4 = build_level(
    "WWWWWWWWWWWWWWWWWWWWWW\n"
    "WP...................W\n"
    "W.WW...WW..H..WW...W.W\n"
    "W.W....W....K.W..W.W.W\n"
    "W.W.S..W..WW..W..W.W.W\n"
    "W.W....W......W....W.W\n"
    "W.WW...WW.S...WW..W.WW\n"
    "W......W......W......W\n"
    "W.WW..WW..WW..WW.SH.WW\n"
    "W........H.....W.....W\n"
    "WWWWDWWWWWDWWWWWDWWWWW\n"
    "W....H.W.......W.....W\n"
    "W.WW...W..WW...W.WW..W\n"
    "W.W.K..W..W.H..W..W..W\n"
    "W.W....W..W....W..W..W\n"
    "W.WW..WW..WW..WW..W..W\n"
    "W....H.W...K.H.W.....W\n"
    "W.WW...W..WW...W.WW..W\n"
    "W......W.......W..X..W\n"
    "W.WW..WW..WW..WW..W..W\n"
    "W.......H.......H....W\n"
    "WWWWWWWWWWWWWWWWWWWWWW",
    name="Level 4: Triple Echo",
    w=22,
    h=22,
)


levels = [level1, level2, level3, level4]

_LEVEL_DATA = [
    (
        "WWWWWWWWWWWWWWWW\n"
        "WP.............W\n"
        "W..WW.....WW...W\n"
        "W...S.....W..W.W\n"
        "W.........W....W\n"
        "W..W...S..W..W.W\n"
        "W..W.......W...W\n"
        "WWWWDWWWWWDWWWWW\n"
        "W.....W........W\n"
        "W.WW.KW...WW...W\n"
        "W.....W........W\n"
        "W..WW.W.....X..W\n"
        "W.....W........W\n"
        "W..WW.W...WW...W\n"
        "W.....W........W\n"
        "WWWWWWWWWWWWWWWW",
        "Level 1: Two Echoes",
        16,
        16,
    ),
    (
        "WWWWWWWWWWWWWWWWWW\n"
        "WP...............W\n"
        "W.WW...WW..WW..W.W\n"
        "W.W....W...W.S.W.W\n"
        "W.W..S.W...W...W.W\n"
        "W.W....W...W.....W\n"
        "W.W....WW.H.WW..WW\n"
        "WWWDWWWWWWWWWWWWWW\n"
        "W.........WW...W.W\n"
        "W.WWH.WW..W..W.W.W\n"
        "W.W.K..W..W..W...W\n"
        "W.W....W.....WDWWW\n"
        "W.W....W..WW.W.X.W\n"
        "W.WW..WW...WWWWWWW\n"
        "W..............W.W\n"
        "W.WW..WW...WW....W\n"
        "W................W\n"
        "WWWWWWWWWWWWWWWWWW",
        "Level 2: Hazard Maze",
        18,
        18,
    ),
    (
        "WWWWWWWWWWWWWWWWWWWW\n"
        "WP.................W\n"
        "W.WW...WW.....WW...W\n"
        "W.W....W......W..W.W\n"
        "W.W..B.W..W...WS.W.W\n"
        "W.W....WWWWK..W..W.W\n"
        "W.WW...W..H...WW...W\n"
        "W......W..WW.......W\n"
        "W.WW..WW..W..WW.W..W\n"
        "W.W....W..WH.W..W..W\n"
        "W.W....W.....W.....W\n"
        "W.W....W..WW.W.WW..W\n"
        "W.WW...WH.W..W.....W\n"
        "WWWWWWDWWWWWWWWWWWWW\n"
        "W........WW..S...W.W\n"
        "W.WWWWWW..W..WW.W..W\n"
        "W.W.X..W..W..W..K..W\n"
        "W.WWWWDW...H.W.WW..W\n"
        "W...W..........W...W\n"
        "WWWWWWWWWWWWWWWWWWWW",
        "Level 3: Block and Switch",
        20,
        20,
    ),
    (
        "WWWWWWWWWWWWWWWWWWWWWW\n"
        "WP...................W\n"
        "W.WW...WW..H..WW...W.W\n"
        "W.W....W....K.W..W.W.W\n"
        "W.W.S..W..WW..W..W.W.W\n"
        "W.W....W......W....W.W\n"
        "W.WW...WW.S...WW..W.WW\n"
        "W......W......W......W\n"
        "W.WW..WW..WW..WW.SH.WW\n"
        "W........H.....W.....W\n"
        "WWWWDWWWWWDWWWWWDWWWWW\n"
        "W....H.W.......W.....W\n"
        "W.WW...W..WW...W.WW..W\n"
        "W.W.K..W..W.H..W..W..W\n"
        "W.W....W..W....W..W..W\n"
        "W.WW..WW..WW..WW..W..W\n"
        "W....H.W...K.H.W.....W\n"
        "W.WW...W..WW...W.WW..W\n"
        "W......W.......W..X..W\n"
        "W.WW..WW..WW..WW..W..W\n"
        "W.......H.......H....W\n"
        "WWWWWWWWWWWWWWWWWWWWWW",
        "Level 4: Triple Echo",
        22,
        22,
    ),
]


def _rebuild_level(idx):
    ascii_str, name, w, h = _LEVEL_DATA[idx]
    return build_level(ascii_str, name, w, h)


camera = Camera(
    x=0,
    y=0,
    width=22,
    height=22,
    background=0,
    letter_box=0,
)


ECHO_INTERVAL = 5
_MAX_LIVES = 3

_BASELINE_MOVES = [55, 65, 80, 100]
MOVE_LIMITS = [2 * m for m in _BASELINE_MOVES]


class Tk07(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)
        self._lives = _MAX_LIVES
        self._last_action_was_reset = False
        self._init_level_state()
        super().__init__(
            game_id="tk07",
            levels=levels,
            camera=camera,
            available_actions=[1, 2, 3, 4, 0, 7],
            win_score=4,
        )

    def _init_level_state(self) -> None:
        self.move_history: list = []
        self.player_positions: list = []
        self.echoes: list = []
        self.echo_sprites: list = []
        self.has_key = False
        self.key_sprites: list = []
        self._move_count = 0
        self._move_limit = 0
        self._move_bar_sprites: list = []
        self._move_bar_total = 0
        self._grid_w = 0
        self._grid_h = 0
        self.switch_sprites: list = []
        self.door_sprites: list = []
        self.door_open_sprites: list = []
        self._hearts: list = []
        self._undo_stack: list = []

    def on_set_level(self, level: Level) -> None:
        idx = self._current_level_index
        if idx < len(_LEVEL_DATA):
            for tag in [
                "floor",
                "wall",
                "player",
                "echo",
                "block",
                "switch",
                "door",
                "door_open",
                "key",
                "exit",
                "hazard",
                "heart",
                "move_pip",
            ]:
                for s in list(self.current_level.get_sprites_by_tag(tag)):
                    try:
                        self.current_level.remove_sprite(s)
                    except Exception:
                        pass
            ascii_str, name, w, h = _LEVEL_DATA[idx]
            layout = parse_layout(ascii_str)
            entities = extract_entities(layout)
            extra = [sprites[k].clone().set_position(x, y) for k, x, y in entities]
            for s in build_level_sprites(layout, extra):
                self.current_level.add_sprite(s)

        self._init_level_state()

        self.player = self.current_level.get_sprites_by_tag("player")[0]
        self.walls = list(self.current_level.get_sprites_by_tag("wall"))
        self.blocks = list(self.current_level.get_sprites_by_tag("block"))

        self.key_sprites = list(self.current_level.get_sprites_by_tag("key"))

        self.exit_sprites = list(self.current_level.get_sprites_by_tag("exit"))
        self.hazards = list(self.current_level.get_sprites_by_tag("hazard"))
        self.switch_sprites = list(self.current_level.get_sprites_by_tag("switch"))
        self.door_sprites = list(self.current_level.get_sprites_by_tag("door"))
        self.door_open_sprites = [None] * len(self.door_sprites)

        idx = self._current_level_index
        self._move_limit = (
            MOVE_LIMITS[idx] if idx < len(MOVE_LIMITS) else MOVE_LIMITS[-1]
        )

        self._grid_w = max(w.x for w in self.walls) + 1
        self._grid_h = max(w.y for w in self.walls) + 1

        self._draw_hearts()
        self._draw_move_bar()

    def _draw_hearts(self) -> None:
        for h in list(self.current_level.get_sprites_by_tag("heart")):
            try:
                self.current_level.remove_sprite(h)
            except Exception:
                pass
        self._hearts = []

        for i in range(self._lives):
            heart = sprites["heart"].clone().set_position(1 + i * 2, 0)
            self.current_level.add_sprite(heart)
            self._hearts.append(heart)

    def _draw_move_bar(self) -> None:
        for s in list(self.current_level.get_sprites_by_tag("move_pip")):
            try:
                self.current_level.remove_sprite(s)
            except Exception:
                pass
        self._move_bar_sprites = []

        self._move_bar_total = self._grid_w - 2
        y = self._grid_h - 1
        for i in range(self._move_bar_total):
            pip = sprites["move_pip"].clone().set_position(1 + i, y)
            self.current_level.add_sprite(pip)
            self._move_bar_sprites.append(pip)

    def _update_move_bar(self) -> None:
        if self._move_limit <= 0 or self._move_bar_total <= 0:
            return
        remaining = self._move_limit - self._move_count
        pips_to_show = max(0, int(remaining * self._move_bar_total / self._move_limit))

        if len(self._move_bar_sprites) == pips_to_show:
            return

        for s in list(self.current_level.get_sprites_by_tag("move_pip")):
            try:
                self.current_level.remove_sprite(s)
            except Exception:
                pass
        self._move_bar_sprites = []

        y = self._grid_h - 1
        for i in range(pips_to_show):
            pip = sprites["move_pip"].clone().set_position(1 + i, y)
            self.current_level.add_sprite(pip)
            self._move_bar_sprites.append(pip)

    def handle_reset(self) -> None:
        if getattr(self._state, "name", "") == "GAME_OVER":
            self._lives = _MAX_LIVES
            self.full_reset()
        else:
            super().handle_reset()

    def step(self) -> None:
        if self.action.id == GameAction.RESET:
            if self._last_action_was_reset:
                self._lives = _MAX_LIVES
                self._last_action_was_reset = False
                self.full_reset()
            else:
                self._lives = _MAX_LIVES
                self._last_action_was_reset = True
                self.set_level(self._current_level_index)
            self.complete_action()
            return

        self._last_action_was_reset = False

        if self.action.id == GameAction.ACTION7:
            if self._apply_undo():
                self._move_count += 1
                self._update_move_bar()
                if self._move_count >= self._move_limit:
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

        if dx == 0 and dy == 0:
            self.complete_action()
            return

        self._save_undo_state()

        moved = self._try_move_player(dx, dy)
        if moved:
            self._move_count += 1
            self._update_move_bar()

            self.player_positions.append((self.player.x - dx, self.player.y - dy))
            self.move_history.append((dx, dy))

            self._step_echoes()

            total = len(self.move_history)
            if total > 0 and total % ECHO_INTERVAL == 0:
                self._spawn_echo()

            self._update_switches()

            self._check_key()

            if self._player_on_hazard():
                self._lose_life()
                self.complete_action()
                return

            if self._check_win():
                self.next_level()
            elif self._move_count >= self._move_limit:
                self._lose_life()
                self.complete_action()
                return

        self.complete_action()

    def _lose_life(self) -> None:
        self._lives -= 1
        if self._lives <= 0:
            self.lose()
            return
        self.set_level(self._current_level_index)

    def _spawn_echo(self) -> None:
        total = len(self.move_history)
        start_idx = total - ECHO_INTERVAL

        sx, sy = self.player_positions[start_idx]
        move_seq = list(self.move_history[start_idx:total])

        echo_sprite = sprites["echo"].clone().set_position(sx, sy)
        self.current_level.add_sprite(echo_sprite)

        echo = {
            "sprite": echo_sprite,
            "moves": move_seq,
            "step": 0,
            "active": True,
            "x": sx,
            "y": sy,
        }
        self.echoes.append(echo)
        self.echo_sprites.append(echo_sprite)

    def _step_echoes(self) -> None:
        for echo in self.echoes:
            if not echo["active"]:
                continue
            if echo["step"] >= len(echo["moves"]):
                echo["active"] = False
                continue

            dx, dy = echo["moves"][echo["step"]]
            echo["step"] += 1

            nx = echo["x"] + dx
            ny = echo["y"] + dy

            if self._is_wall(nx, ny) or self._is_closed_door(nx, ny):
                continue

            pushed = self._try_push_block(nx, ny, dx, dy)
            block_at = self._get_block_at(nx, ny)
            if block_at and not pushed:
                continue

            echo["x"] = nx
            echo["y"] = ny
            echo["sprite"].set_position(nx, ny)

            if echo["step"] >= len(echo["moves"]):
                echo["active"] = False

    def _update_switches(self) -> None:
        for idx, sw in enumerate(self.switch_sprites):
            if idx >= len(self.door_sprites):
                break

            pressed = self._entity_on(sw.x, sw.y)
            door = self.door_sprites[idx]
            currently_open = self.door_open_sprites[idx] is not None

            if pressed and not currently_open:
                self.current_level.remove_sprite(door)
                if door in self.walls:
                    self.walls.remove(door)
                open_spr = sprites["door_open"].clone().set_position(door.x, door.y)
                self.current_level.add_sprite(open_spr)
                self.door_open_sprites[idx] = open_spr
                self._set_switch_visual(idx, True)

            elif not pressed and currently_open:
                open_spr = self.door_open_sprites[idx]
                self.current_level.remove_sprite(open_spr)
                self.door_open_sprites[idx] = None
                new_closed = sprites["door_closed"].clone().set_position(door.x, door.y)
                self.current_level.add_sprite(new_closed)
                self.door_sprites[idx] = new_closed
                self.walls.append(new_closed)
                self._set_switch_visual(idx, False)

    def _set_switch_visual(self, idx, on: bool) -> None:
        old = self.switch_sprites[idx]
        new_key = "switch_on" if on else "switch_off"
        new_spr = sprites[new_key].clone().set_position(old.x, old.y)
        self.current_level.remove_sprite(old)
        self.current_level.add_sprite(new_spr)
        self.switch_sprites[idx] = new_spr

    def _entity_on(self, x: int, y: int) -> bool:
        if self.player.x == x and self.player.y == y:
            return True
        for echo in self.echoes:
            if echo["x"] == x and echo["y"] == y:
                return True
        return False

    def _try_move_player(self, dx: int, dy: int) -> bool:
        nx = self.player.x + dx
        ny = self.player.y + dy

        if self._is_wall(nx, ny) or self._is_closed_door(nx, ny):
            return False

        if not self.has_key and self._is_exit(nx, ny):
            return False

        block = self._get_block_at(nx, ny)
        if block:
            if not self._try_push_block(nx, ny, dx, dy):
                return False

        self.player.set_position(nx, ny)
        return True

    def _try_push_block(self, bx, by, dx, dy) -> bool:
        block = self._get_block_at(bx, by)
        if not block:
            return False
        tx, ty = bx + dx, by + dy
        if self._is_wall(tx, ty) or self._is_closed_door(tx, ty):
            return False
        if self._get_block_at(tx, ty):
            return False
        block.set_position(tx, ty)
        return True

    def _check_key(self) -> None:
        if self.has_key:
            return
        for ks in list(self.key_sprites):
            if ks.x == self.player.x and ks.y == self.player.y:
                self.current_level.remove_sprite(ks)
                self.key_sprites.remove(ks)
                break
        if not self.key_sprites:
            self.has_key = True

    def _check_win(self) -> bool:
        if not self.has_key:
            return False
        for ex in self.exit_sprites:
            if ex.x == self.player.x and ex.y == self.player.y:
                return True
        return False

    def _player_on_hazard(self) -> bool:
        px, py = self.player.x, self.player.y
        for hz in self.hazards:
            if hz.x == px and hz.y == py:
                for echo in self.echoes:
                    if echo["x"] == px and echo["y"] == py:
                        return False
                return True
        return False

    def _is_wall(self, x: int, y: int) -> bool:
        for w in self.walls:
            if w.x == x and w.y == y:
                return True
        return False

    def _is_closed_door(self, x: int, y: int) -> bool:
        for idx, d in enumerate(self.door_sprites):
            if d.x == x and d.y == y:
                if (
                    idx < len(self.door_open_sprites)
                    and self.door_open_sprites[idx] is not None
                ):
                    return False
                return True
        return False

    def _is_exit(self, x: int, y: int) -> bool:
        for ex in self.exit_sprites:
            if ex.x == x and ex.y == y:
                return True
        return False

    def _get_block_at(self, x: int, y: int):
        for b in self.blocks:
            if b.x == x and b.y == y:
                return b
        return None

    def _save_undo_state(self) -> None:
        snapshot = {
            "player_x": self.player.x,
            "player_y": self.player.y,
            "move_count": self._move_count,
            "has_key": self.has_key,
            "key_positions": [(ks.x, ks.y) for ks in self.key_sprites],
            "block_positions": [(b.x, b.y) for b in self.blocks],
            "move_history_len": len(self.move_history),
            "player_positions_len": len(self.player_positions),
            "echo_data": [
                {
                    "x": e["x"],
                    "y": e["y"],
                    "step": e["step"],
                    "active": e["active"],
                    "moves": list(e["moves"]),
                }
                for e in self.echoes
            ],
            "echo_count": len(self.echoes),
            "door_open": [
                self.door_open_sprites[i] is not None
                for i in range(len(self.door_sprites))
            ],
        }
        self._undo_stack.append(snapshot)
        if len(self._undo_stack) > 50:
            self._undo_stack.pop(0)

    def _apply_undo(self) -> bool:
        if not self._undo_stack:
            return False
        snapshot = self._undo_stack.pop()

        self.player.set_position(snapshot["player_x"], snapshot["player_y"])
        self.has_key = snapshot["has_key"]

        for i, (bx, by) in enumerate(snapshot["block_positions"]):
            if i < len(self.blocks):
                self.blocks[i].set_position(bx, by)

        old_key_positions = snapshot["key_positions"]
        if len(old_key_positions) > len(self.key_sprites):
            for ks in list(self.key_sprites):
                try:
                    self.current_level.remove_sprite(ks)
                except Exception:
                    pass
            self.key_sprites = []
            for kx, ky in old_key_positions:
                ks = sprites["key"].clone().set_position(kx, ky)
                self.current_level.add_sprite(ks)
                self.key_sprites.append(ks)
        elif len(old_key_positions) == len(self.key_sprites):
            for i, (kx, ky) in enumerate(old_key_positions):
                self.key_sprites[i].set_position(kx, ky)

        self.move_history = self.move_history[: snapshot["move_history_len"]]
        self.player_positions = self.player_positions[
            : snapshot["player_positions_len"]
        ]

        old_echo_data = snapshot["echo_data"]
        old_count = snapshot["echo_count"]

        while len(self.echoes) > old_count:
            removed = self.echoes.pop()
            removed_spr = self.echo_sprites.pop()
            try:
                self.current_level.remove_sprite(removed_spr)
            except Exception:
                pass

        for i, ed in enumerate(old_echo_data):
            if i < len(self.echoes):
                self.echoes[i]["x"] = ed["x"]
                self.echoes[i]["y"] = ed["y"]
                self.echoes[i]["step"] = ed["step"]
                self.echoes[i]["active"] = ed["active"]
                self.echoes[i]["moves"] = ed["moves"]
                self.echo_sprites[i].set_position(ed["x"], ed["y"])

        old_door_open = snapshot["door_open"]
        for idx in range(len(self.door_sprites)):
            if idx >= len(old_door_open):
                break
            currently_open = self.door_open_sprites[idx] is not None
            should_be_open = old_door_open[idx]

            if currently_open and not should_be_open:
                open_spr = self.door_open_sprites[idx]
                self.current_level.remove_sprite(open_spr)
                self.door_open_sprites[idx] = None
                door = self.door_sprites[idx]
                new_closed = sprites["door_closed"].clone().set_position(door.x, door.y)
                self.current_level.add_sprite(new_closed)
                self.door_sprites[idx] = new_closed
                self.walls.append(new_closed)
                self._set_switch_visual(idx, False)
            elif not currently_open and should_be_open:
                door = self.door_sprites[idx]
                self.current_level.remove_sprite(door)
                if door in self.walls:
                    self.walls.remove(door)
                open_spr = sprites["door_open"].clone().set_position(door.x, door.y)
                self.current_level.add_sprite(open_spr)
                self.door_open_sprites[idx] = open_spr
                self._set_switch_visual(idx, True)

        self._update_move_bar()
        return True


class PuzzleEnvironment:
    ACTION_MAP: Dict[str, int] = {
        "up": 1,
        "down": 2,
        "left": 3,
        "right": 4,
        "reset": 0,
        "undo": 7,
    }

    def __init__(self, seed: int = 0) -> None:
        self._engine = Tk07(seed=seed)
        self._total_turns = 0
        self._last_action_was_reset = False
        self._prev_score = 0
        self._done = False
        self._game_won = False
        self._game_over = False

    def _build_text_obs(self) -> str:
        e = self._engine
        cam = e.camera
        all_sprites = e.current_level.get_sprites()
        index_grid = cam.render(all_sprites)
        if hasattr(index_grid, "tolist"):
            grid = index_grid.tolist()
        else:
            grid = index_grid if index_grid else []
        lines = []
        for row in grid:
            lines.append("".join(str(c) for c in row))
        level_num = e.level_index + 1
        header = (
            f"Level:{level_num} Lives:{e._lives} Moves:{e._move_count}/{e._move_limit}"
        )
        return header + "\n" + "\n".join(lines)

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
        return (
            b"\x89PNG\r\n\x1a\n"
            + _chunk(b"IHDR", ihdr_data)
            + _chunk(b"IDAT", compressed)
            + _chunk(b"IEND", b"")
        )

    def _build_image_bytes(self) -> Optional[bytes]:
        e = self._engine
        index_grid = e.camera.render(e.current_level.get_sprites())
        arr = np.array(index_grid, dtype=np.uint8)
        h, w = arr.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for idx, color in enumerate(ARC_PALETTE):
            mask = arr == idx
            rgb[mask] = color
        return self._encode_png(rgb)

    def _build_game_state(self, done: bool, info: Dict) -> GameState:
        e = self._engine
        valid_actions = self.get_actions() if not done else None
        return GameState(
            text_observation=self._build_text_obs(),
            image_observation=self._build_image_bytes(),
            valid_actions=valid_actions,
            turn=self._total_turns,
            metadata={
                "total_levels": len(e._levels),
                "level_index": e.level_index,
                "levels_completed": getattr(e, "_score", 0),
                "game_over": getattr(getattr(e, "_state", None), "name", "")
                == "GAME_OVER",
                "done": done,
                "info": info,
            },
        )

    def reset(self) -> GameState:
        e = self._engine
        self._total_turns = 0
        if self._game_won or self._last_action_was_reset:
            e.full_reset()
        else:
            e.level_reset()
        self._last_action_was_reset = True
        self._prev_score = getattr(e, "_score", 0)
        self._done = False
        self._game_won = False
        self._game_over = False
        return self._build_game_state(done=False, info={})

    def get_actions(self) -> List[str]:
        if self.is_done():
            return ["reset"]
        return list(self.ACTION_MAP.keys())

    def is_done(self) -> bool:
        e = self._engine
        state_name = getattr(e._state, "name", "")
        return state_name in ("WIN", "GAME_OVER")

    def step(self, action: str) -> StepResult:
        if action not in self.ACTION_MAP:
            raise ValueError(
                f"Unknown action '{action}'. Valid: {list(self.ACTION_MAP.keys())}"
            )

        if action == "reset":
            state = self.reset()
            return StepResult(
                state=state, reward=0.0, done=False, info={"action": "reset"}
            )

        self._last_action_was_reset = False
        self._total_turns += 1
        e = self._engine
        prev_score = self._prev_score

        game_action_id = self.ACTION_MAP[action]
        _action_map = {
            1: GameAction.ACTION1,
            2: GameAction.ACTION2,
            3: GameAction.ACTION3,
            4: GameAction.ACTION4,
            7: GameAction.ACTION7,
        }
        game_action = _action_map.get(game_action_id, GameAction.RESET)
        info: Dict = {"action": action}

        action_input = ActionInput(id=game_action)
        frame = e.perform_action(action_input, raw=True)

        self._prev_score = frame.levels_completed
        levels_advanced = frame.levels_completed - prev_score

        game_won = frame.state and frame.state.name == "WIN"
        game_over = frame.state and frame.state.name == "GAME_OVER"
        done = bool(game_won or game_over)

        self._done = done
        self._game_won = bool(game_won)
        self._game_over = bool(game_over)

        reward = levels_advanced * (1.0 / len(e._levels))

        if game_won:
            info["reason"] = "game_complete"
        elif game_over:
            info["reason"] = "game_over"

        return StepResult(
            state=self._build_game_state(done=done, info=info),
            reward=reward,
            done=done,
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
    metadata: Dict = {
        "render_modes": ["rgb_array"],
        "render_fps": 5,
    }

    ACTION_LIST: List[str] = ["reset", "up", "down", "left", "right", "undo"]

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
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        if seed is not None:
            self._seed = seed

        self._env = PuzzleEnvironment(seed=self._seed)
        state = self._env.reset()

        return self._get_obs(), self._build_info(state)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
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

    def _build_info(self, state: GameState, step_info: Optional[Dict] = None) -> Dict:
        info: Dict = {
            "text_observation": state.text_observation,
            "valid_actions": state.valid_actions,
            "turn": state.turn,
            "game_metadata": state.metadata,
        }
        if step_info:
            info["step_info"] = step_info
        return info
