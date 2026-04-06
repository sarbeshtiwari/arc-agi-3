import copy
import io
import random
import struct
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

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
    metadata: dict = field(default_factory=dict)


@dataclass
class StepResult:
    state: GameState
    reward: float
    done: bool
    info: dict = field(default_factory=dict)


class Hud(RenderableUserDisplay):
    def __init__(self, ref: "Tc48"):
        self.ref = ref

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if h < 6 or w < 6:
            return frame

        mvl = self.ref.mvl
        mvc = self.ref.mvc
        if mvl > 0:
            ratio = max(0.0, mvc / mvl)
            filled = int(round(w * ratio))
            for x in range(w):
                clr = 14 if x < filled else 8
                frame[h - 1, x] = clr

        for i in range(self.ref._lives):
            pip_x = w - 3 - i * 4
            if pip_x >= 0 and pip_x + 1 < w:
                frame[1, pip_x] = 8
                frame[1, pip_x + 1] = 8
                frame[2, pip_x] = 8
                frame[2, pip_x + 1] = 8

        for i in range(self.ref._keys_needed):
            cx = 1 + i * 4
            if cx + 1 < w:
                if i < self.ref._keys_held:
                    clr = 12
                else:
                    clr = 3
                frame[1, cx] = clr
                frame[1, cx + 1] = clr
                frame[2, cx] = clr
                frame[2, cx + 1] = clr

        return frame


WALL_PIXEL = [[4]]
WALL_2X2_PIXEL = [[4, 4], [4, 4]]
THIEF_PIXEL = [[11]]
COP_PIXEL = [[8]]
DOOR_PIXEL = [[14]]
FAKE_DOOR_PIXEL = [[15]]
KEY_PIXEL = [[12]]
HIDING_SPOT_PIXEL = [[3]]
ALARM_PIXEL = [[6]]
VISION_PIXEL = [[7]]

sprites = {
    "thief": Sprite(
        pixels=THIEF_PIXEL,
        name="thief",
        visible=True,
        collidable=True,
        tags=["plr"],
        layer=5,
    ),
    "cop": Sprite(
        pixels=COP_PIXEL,
        name="cop",
        visible=True,
        collidable=True,
        tags=["cop"],
        layer=4,
    ),
    "wall": Sprite(
        pixels=WALL_PIXEL,
        name="wall",
        visible=True,
        collidable=True,
        tags=["wll"],
        layer=0,
    ),
    "wall_big": Sprite(
        pixels=WALL_2X2_PIXEL,
        name="wall_big",
        visible=True,
        collidable=True,
        tags=["wll"],
        layer=0,
    ),
    "door": Sprite(
        pixels=DOOR_PIXEL,
        name="door",
        visible=True,
        collidable=True,
        tags=["ext"],
        layer=1,
    ),
    "fake_door": Sprite(
        pixels=FAKE_DOOR_PIXEL,
        name="fake_door",
        visible=True,
        collidable=True,
        tags=["trp"],
        layer=1,
    ),
    "key": Sprite(
        pixels=KEY_PIXEL,
        name="key",
        visible=True,
        collidable=True,
        tags=["key"],
        layer=2,
    ),
    "hiding_spot": Sprite(
        pixels=HIDING_SPOT_PIXEL,
        name="hiding_spot",
        visible=True,
        collidable=False,
        tags=["hde"],
        layer=-1,
    ),
    "alarm": Sprite(
        pixels=ALARM_PIXEL,
        name="alarm",
        visible=True,
        collidable=False,
        tags=["alm"],
        layer=-1,
    ),
    "vision": Sprite(
        pixels=VISION_PIXEL,
        name="vision",
        visible=True,
        collidable=False,
        tags=["vis"],
        layer=3,
    ),
}


def _wall(x, y):
    return sprites["wall"].clone().set_position(x, y)


def _wall2(x, y):
    return sprites["wall_big"].clone().set_position(x, y)


def _cop(x, y):
    return sprites["cop"].clone().set_position(x, y)


def _key(x, y):
    return sprites["key"].clone().set_position(x, y)


def _door(x, y):
    return sprites["door"].clone().set_position(x, y)


def _fake(x, y):
    return sprites["fake_door"].clone().set_position(x, y)


def _hide(x, y):
    return sprites["hiding_spot"].clone().set_position(x, y)


def _alarm(x, y):
    return sprites["alarm"].clone().set_position(x, y)


def _thief(x, y):
    return sprites["thief"].clone().set_position(x, y)


def _border(w, h):
    walls = []
    for x in range(w):
        walls.append(_wall(x, 0))
        walls.append(_wall(x, h - 1))
    for y in range(1, h - 1):
        walls.append(_wall(0, y))
        walls.append(_wall(w - 1, y))
    return walls


def _hline(x1, x2, y):
    return [_wall(x, y) for x in range(x1, x2 + 1)]


def _vline(x, y1, y2):
    return [_wall(x, y) for y in range(y1, y2 + 1)]


_l1_sprites = []
_l1_sprites += _border(12, 12)
_l1_sprites += _hline(2, 5, 2)
_l1_sprites += _vline(5, 3, 5)
_l1_sprites += _hline(7, 9, 4)
_l1_sprites += _vline(3, 7, 9)
_l1_sprites += _hline(5, 8, 7)
_l1_sprites += _vline(8, 8, 9)
_l1_sprites += _hline(5, 6, 9)
_l1_sprites.append(_thief(1, 10))
_l1_sprites.append(_cop(6, 3))
_l1_sprites.append(_key(6, 6))
_l1_sprites.append(_hide(2, 6))
_l1_sprites.append(_hide(9, 2))
_l1_sprites.append(_door(10, 1))


_l2_sprites = []
_l2_sprites += _border(18, 18)
_l2_sprites += _hline(2, 7, 3)
_l2_sprites += _hline(10, 15, 3)
_l2_sprites += _vline(7, 4, 7)
_l2_sprites += _vline(10, 4, 8)
_l2_sprites += _hline(2, 5, 8)
_l2_sprites += _hline(12, 16, 8)
_l2_sprites += _vline(5, 9, 12)
_l2_sprites += _hline(7, 10, 11)
_l2_sprites += _vline(12, 9, 11)
_l2_sprites += _hline(2, 4, 14)
_l2_sprites += _vline(8, 13, 16)
_l2_sprites += _hline(10, 14, 14)
_l2_sprites += _vline(14, 10, 13)
_l2_sprites += _vline(3, 10, 12)
_l2_sprites.append(_thief(1, 16))
_l2_sprites.append(_cop(4, 5))
_l2_sprites.append(_cop(13, 12))
_l2_sprites.append(_key(11, 6))
_l2_sprites.append(_hide(2, 10))
_l2_sprites.append(_hide(15, 5))
_l2_sprites.append(_hide(6, 15))
_l2_sprites.append(_alarm(9, 9))
_l2_sprites.append(_door(16, 1))


_l3_sprites = []
_l3_sprites += _border(24, 24)
_l3_sprites += _hline(2, 10, 4)
_l3_sprites += _hline(13, 21, 4)
_l3_sprites += _vline(10, 5, 8)
_l3_sprites += _vline(13, 5, 8)
_l3_sprites += _hline(2, 6, 9)
_l3_sprites += _hline(15, 22, 9)
_l3_sprites += _vline(6, 10, 14)
_l3_sprites += _vline(8, 10, 12)
_l3_sprites += _hline(10, 13, 12)
_l3_sprites += _vline(16, 10, 14)
_l3_sprites += _hline(2, 5, 15)
_l3_sprites += _hline(8, 14, 15)
_l3_sprites += _hline(18, 22, 15)
_l3_sprites += _vline(5, 16, 19)
_l3_sprites += _vline(11, 16, 18)
_l3_sprites += _vline(14, 16, 19)
_l3_sprites += _vline(18, 16, 19)
_l3_sprites += _hline(2, 4, 20)
_l3_sprites += _hline(7, 10, 20)
_l3_sprites += _hline(16, 22, 20)
_l3_sprites += _vline(10, 6, 7)
_l3_sprites += _vline(20, 5, 8)
_l3_sprites += _hline(10, 12, 18)
_l3_sprites.append(_thief(1, 22))
_l3_sprites.append(_cop(5, 6))
_l3_sprites.append(_cop(18, 11))
_l3_sprites.append(_cop(12, 19))
_l3_sprites.append(_key(3, 11))
_l3_sprites.append(_key(19, 7))
_l3_sprites.append(_hide(1, 7))
_l3_sprites.append(_hide(22, 3))
_l3_sprites.append(_hide(9, 14))
_l3_sprites.append(_hide(15, 21))
_l3_sprites.append(_hide(7, 18))
_l3_sprites.append(_alarm(4, 12))
_l3_sprites.append(_alarm(21, 6))
_l3_sprites.append(_fake(22, 22))
_l3_sprites.append(_door(22, 1))


_l4_sprites = []
_l4_sprites += _border(30, 30)
_l4_sprites += _hline(2, 6, 3)
_l4_sprites += _hline(9, 13, 3)
_l4_sprites += _hline(16, 20, 3)
_l4_sprites += _hline(23, 27, 3)
_l4_sprites += _vline(6, 4, 6)
_l4_sprites += _vline(13, 4, 6)
_l4_sprites += _vline(20, 4, 6)
_l4_sprites += _vline(27, 4, 6)
_l4_sprites += _hline(2, 4, 7)
_l4_sprites += _hline(8, 11, 7)
_l4_sprites += _hline(15, 18, 7)
_l4_sprites += _hline(22, 25, 7)
_l4_sprites += _vline(4, 8, 10)
_l4_sprites += _vline(9, 8, 10)
_l4_sprites += _vline(14, 8, 10)
_l4_sprites += _vline(19, 8, 10)
_l4_sprites += _vline(25, 8, 10)
_l4_sprites += _hline(2, 3, 11)
_l4_sprites += _hline(6, 8, 11)
_l4_sprites += _hline(11, 13, 11)
_l4_sprites += _hline(16, 18, 11)
_l4_sprites += _hline(21, 24, 11)
_l4_sprites += _hline(27, 28, 11)
_l4_sprites += _vline(3, 12, 14)
_l4_sprites += _vline(8, 12, 14)
_l4_sprites += _vline(13, 12, 14)
_l4_sprites += _vline(18, 12, 14)
_l4_sprites += _vline(24, 12, 14)
_l4_sprites += _hline(2, 6, 15)
_l4_sprites += _hline(10, 12, 15)
_l4_sprites += _hline(15, 17, 15)
_l4_sprites += _hline(20, 23, 15)
_l4_sprites += _hline(26, 28, 15)
_l4_sprites += _vline(5, 16, 18)
_l4_sprites += _vline(10, 16, 18)
_l4_sprites += _vline(15, 16, 18)
_l4_sprites += _vline(22, 16, 18)
_l4_sprites += _vline(27, 16, 18)
_l4_sprites += _hline(2, 4, 19)
_l4_sprites += _hline(7, 9, 19)
_l4_sprites += _hline(12, 14, 19)
_l4_sprites += _hline(17, 21, 19)
_l4_sprites += _hline(24, 26, 19)
_l4_sprites += _vline(4, 20, 22)
_l4_sprites += _vline(9, 20, 22)
_l4_sprites += _vline(14, 20, 22)
_l4_sprites += _vline(21, 20, 22)
_l4_sprites += _vline(26, 20, 22)
_l4_sprites += _hline(2, 3, 23)
_l4_sprites += _hline(6, 8, 23)
_l4_sprites += _hline(11, 13, 23)
_l4_sprites += _hline(16, 20, 23)
_l4_sprites += _hline(23, 25, 23)
_l4_sprites += _vline(6, 24, 27)
_l4_sprites += _vline(12, 24, 27)
_l4_sprites += _vline(18, 24, 27)
_l4_sprites += _vline(24, 24, 27)

_l4_sprites.append(_thief(1, 28))
_l4_sprites.append(_cop(8, 5))
_l4_sprites.append(_cop(22, 10))
_l4_sprites.append(_cop(11, 20))
_l4_sprites.append(_cop(25, 25))
_l4_sprites.append(_key(5, 9))
_l4_sprites.append(_key(16, 5))
_l4_sprites.append(_key(26, 16))
_l4_sprites.append(_hide(2, 5))
_l4_sprites.append(_hide(14, 4))
_l4_sprites.append(_hide(26, 7))
_l4_sprites.append(_hide(7, 16))
_l4_sprites.append(_hide(20, 21))
_l4_sprites.append(_hide(16, 26))
_l4_sprites.append(_alarm(5, 10))
_l4_sprites.append(_alarm(17, 8))
_l4_sprites.append(_alarm(25, 17))
_l4_sprites.append(_fake(1, 1))
_l4_sprites.append(_fake(28, 28))
_l4_sprites.append(_door(28, 1))


levels = [
    Level(
        sprites=_l1_sprites,
        grid_size=(12, 12),
        data={
            "cop_routes": {
                0: [(6, 3), (6, 1), (9, 1), (9, 3), (6, 3)],
            },
            "cop_speed": 2,
            "keys_required": 1,
            "cop_vision": 2,
            "has_chaser": False,
            "alarm_boost": 0,
            "max_steps": 50,
        },
        name="alley",
    ),
    Level(
        sprites=_l2_sprites,
        grid_size=(18, 18),
        data={
            "cop_routes": {
                0: [(4, 5), (4, 1), (8, 1), (8, 5), (4, 5)],
                1: [(13, 12), (13, 9), (16, 9), (16, 12), (13, 12)],
            },
            "cop_speed": 1,
            "keys_required": 1,
            "cop_vision": 3,
            "has_chaser": False,
            "alarm_boost": 3,
            "max_steps": 90,
        },
        name="warehouse",
    ),
    Level(
        sprites=_l3_sprites,
        grid_size=(24, 24),
        data={
            "cop_routes": {
                0: [(5, 6), (5, 2), (9, 2), (9, 6), (5, 6)],
                1: [(18, 11), (18, 13), (21, 13), (21, 10), (18, 10), (18, 11)],
                2: [(12, 19), (12, 17), (13, 17), (13, 21), (12, 21), (12, 19)],
            },
            "cop_speed": 1,
            "keys_required": 2,
            "cop_vision": 3,
            "has_chaser": False,
            "alarm_boost": 5,
            "max_steps": 130,
        },
        name="museum",
    ),
    Level(
        sprites=_l4_sprites,
        grid_size=(30, 30),
        data={
            "cop_routes": {
                0: [(8, 5), (8, 2), (12, 2), (12, 5), (8, 5)],
                1: [(22, 10), (22, 8), (24, 8), (24, 10), (22, 10)],
                2: [(11, 20), (11, 16), (13, 16), (13, 20), (11, 20)],
                3: "chaser",
            },
            "cop_speed": 1,
            "keys_required": 3,
            "cop_vision": 4,
            "has_chaser": True,
            "alarm_boost": 6,
            "max_steps": 180,
        },
        name="vault",
    ),
]


BACKGROUND_COLOR = 5
PADDING_COLOR = 4


class Tc48(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)
        self._lives = 3
        self.mvl = 50
        self.mvc = 50
        self._keys_held = 0
        self._keys_needed = 0
        self._is_hiding = False
        self._alarm_active = False
        self._alarm_timer = 0
        self._cop_states = []
        self._cop_sprites = []
        self._vision_sprites = []
        self._step_counter = 0
        self._wall_set = set()

        self._game_over = False
        self._level_done = False
        self._undo_stack = []

        self._hud = Hud(self)

        super().__init__(
            "tc48",
            levels,
            Camera(0, 0, 32, 32, BACKGROUND_COLOR, PADDING_COLOR, [self._hud]),
            available_actions=[0, 1, 2, 3, 4, 7],
        )

    def _build_wall_set(self) -> None:
        self._wall_set = set()
        for ws in self.current_level.get_sprites_by_tag("wll"):
            pixel_w = len(ws.pixels[0]) if ws.pixels else 1
            pixel_h = len(ws.pixels) if ws.pixels else 1
            for wx in range(ws.x, ws.x + pixel_w):
                for wy in range(ws.y, ws.y + pixel_h):
                    self._wall_set.add((wx, wy))

    def _save_original_positions(self) -> None:
        self._orig_keys = []
        for ks in self.current_level.get_sprites_by_tag("key"):
            self._orig_keys.append((ks.x, ks.y))

        self._orig_alarms = []
        for als in self.current_level.get_sprites_by_tag("alm"):
            self._orig_alarms.append((als.x, als.y))

        self._orig_fakes = []
        for fks in self.current_level.get_sprites_by_tag("trp"):
            self._orig_fakes.append((fks.x, fks.y))

    def _init_cop_states(self, level: Level) -> None:
        self._cop_sprites = self.current_level.get_sprites_by_tag("cop")
        self._cop_orig_pos = [(cs.x, cs.y) for cs in self._cop_sprites]
        self._cop_states = []
        routes = level.get_data("cop_routes") or {}
        for i, cs in enumerate(self._cop_sprites):
            route = routes.get(i, routes.get(str(i), None))
            if route == "chaser":
                self._cop_states.append(
                    {
                        "type": "chaser",
                        "route_idx": 0,
                        "progress": 0.0,
                    }
                )
            elif route:
                self._cop_states.append(
                    {
                        "type": "patrol",
                        "route": route,
                        "route_idx": 0,
                        "target_idx": 1,
                        "progress": 0.0,
                    }
                )
            else:
                self._cop_states.append(
                    {
                        "type": "static",
                    }
                )

    def on_set_level(self, level: Level) -> None:
        self._lives = 3
        self._keys_held = 0
        self._keys_needed = level.get_data("keys_required") or 1
        mvl = level.get_data("max_steps")
        self.mvl = mvl if mvl is not None else 50
        self.mvc = self.mvl
        self._is_hiding = False
        self._alarm_active = False
        self._alarm_timer = 0
        self._vision_sprites = []
        self._step_counter = 0
        self._level_done = False
        self._undo_stack = []

        plr_list = self.current_level.get_sprites_by_tag("plr")
        self._player = plr_list[0] if plr_list else None

        self._start_pos = (self._player.x, self._player.y) if self._player else (1, 1)

        self._build_wall_set()
        self._save_original_positions()
        self._init_cop_states(level)

        if self._player:
            gw, gh = level.grid_size
            cam_w = self.camera.width
            cam_h = self.camera.height
            self.camera.x = max(0, min(self._player.x - cam_w // 2, gw - cam_w))
            self.camera.y = max(0, min(self._player.y - cam_h // 2, gh - cam_h))

    def _is_wall(self, x: int, y: int) -> bool:
        gw, gh = self.current_level.grid_size
        if x < 0 or y < 0 or x >= gw or y >= gh:
            return True
        return (x, y) in self._wall_set

    def _sprite_at(self, x: int, y: int, tag: str):
        for s in self.current_level.get_sprites_by_tag(tag):
            pixel_w = len(s.pixels[0]) if s.pixels else 1
            pixel_h = len(s.pixels) if s.pixels else 1
            if s.x <= x < s.x + pixel_w and s.y <= y < s.y + pixel_h:
                return s
        return None

    def _manhattan(self, x1, y1, x2, y2) -> int:
        return abs(x1 - x2) + abs(y1 - y2)

    def _move_chaser(self, cop_sprite):
        if self._is_hiding:
            return
        cx, cy = cop_sprite.x, cop_sprite.y
        px, py = self._player.x, self._player.y
        dx = 0
        dy = 0
        if px > cx:
            dx = 1
        elif px < cx:
            dx = -1
        if py > cy:
            dy = 1
        elif py < cy:
            dy = -1
        if dx != 0 and not self._is_wall(cx + dx, cy):
            cop_sprite.set_position(cx + dx, cy)
        elif dy != 0 and not self._is_wall(cx, cy + dy):
            cop_sprite.set_position(cx, cy + dy)

    def _move_patrol(self, cop_sprite, state):
        route = state["route"]
        if len(route) < 2:
            return
        patrol_target = route[state["target_idx"]]
        cx, cy = cop_sprite.x, cop_sprite.y

        is_chasing_player = self._alarm_active and not self._is_hiding
        if is_chasing_player:
            tx, ty = self._player.x, self._player.y
        else:
            tx, ty = patrol_target

        dx = 0
        dy = 0
        if tx > cx:
            dx = 1
        elif tx < cx:
            dx = -1
        if ty > cy:
            dy = 1
        elif ty < cy:
            dy = -1

        if dx != 0 and not self._is_wall(cx + dx, cy):
            cop_sprite.set_position(cx + dx, cy)
        elif dy != 0 and not self._is_wall(cx, cy + dy):
            cop_sprite.set_position(cx, cy + dy)

        if not is_chasing_player:
            if cop_sprite.x == patrol_target[0] and cop_sprite.y == patrol_target[1]:
                state["target_idx"] = (state["target_idx"] + 1) % len(route)

    def _move_cops(self):
        if not self._player:
            return
        for i, cop_sprite in enumerate(self._cop_sprites):
            state = self._cop_states[i]
            if state["type"] == "chaser":
                self._move_chaser(cop_sprite)
            elif state["type"] == "patrol":
                self._move_patrol(cop_sprite, state)

    def _check_caught(self) -> bool:
        if not self._player or self._is_hiding:
            return False
        px, py = self._player.x, self._player.y
        cop_vision = self.current_level.get_data("cop_vision") or 2
        for cs in self._cop_sprites:
            dist = self._manhattan(px, py, cs.x, cs.y)
            if dist <= 1:
                return True
            if dist <= cop_vision:
                if self._has_los(cs.x, cs.y, px, py):
                    return True
        return False

    def _has_los(self, x1, y1, x2, y2) -> bool:
        dx = x2 - x1
        dy = y2 - y1
        steps = max(abs(dx), abs(dy))
        if steps == 0:
            return True
        sx = dx / steps
        sy = dy / steps
        cx, cy = float(x1), float(y1)
        for _ in range(steps):
            cx += sx
            cy += sy
            ix, iy = int(round(cx)), int(round(cy))
            if ix == x2 and iy == y2:
                return True
            if self._is_wall(ix, iy):
                return False
        return True

    def _restore_collectibles(self):
        current_key_pos = set(
            (k.x, k.y) for k in self.current_level.get_sprites_by_tag("key")
        )
        for kx, ky in self._orig_keys:
            if (kx, ky) not in current_key_pos:
                new_key = sprites["key"].clone().set_position(kx, ky)
                self.current_level.add_sprite(new_key)

        current_alm_pos = set(
            (a.x, a.y) for a in self.current_level.get_sprites_by_tag("alm")
        )
        for ax, ay in self._orig_alarms:
            if (ax, ay) not in current_alm_pos:
                new_alm = sprites["alarm"].clone().set_position(ax, ay)
                self.current_level.add_sprite(new_alm)

        current_fake_pos = set(
            (f.x, f.y) for f in self.current_level.get_sprites_by_tag("trp")
        )
        for fx, fy in self._orig_fakes:
            if (fx, fy) not in current_fake_pos:
                new_fake = sprites["fake_door"].clone().set_position(fx, fy)
                self.current_level.add_sprite(new_fake)

    def _reset_cops_to_origin(self):
        for i, cs in enumerate(self._cop_sprites):
            ox, oy = self._cop_orig_pos[i]
            cs.set_position(ox, oy)
            state = self._cop_states[i]
            if state["type"] == "patrol":
                state["route_idx"] = 0
                state["target_idx"] = 1

    def _lose_life(self):
        self._lives -= 1
        if self._lives <= 0:
            self._game_over = True
            self.lose()
            return

        self.mvc = self.mvl
        self._step_counter = 0
        self._keys_held = 0
        self._alarm_active = False
        self._alarm_timer = 0
        self._is_hiding = False

        if self._player:
            sx, sy = self._start_pos
            self._player.set_position(sx, sy)
            self._player.color_remap(3, 11)

        self._reset_cops_to_origin()
        self._restore_collectibles()
        self._update_camera()

    def _update_camera(self):
        if not self._player:
            return
        gw, gh = self.current_level.grid_size
        cam_w = self.camera.width
        cam_h = self.camera.height
        self.camera.x = max(0, min(self._player.x - cam_w // 2, gw - cam_w))
        self.camera.y = max(0, min(self._player.y - cam_h // 2, gh - cam_h))

    def _handle_movement(self, dx, dy) -> bool:
        nx = self._player.x + dx
        ny = self._player.y + dy

        if self._is_wall(nx, ny):
            return False

        self._player.set_position(nx, ny)

        hspot = self._sprite_at(nx, ny, "hde")
        if hspot:
            self._is_hiding = True
            self._player.color_remap(11, 3)
        else:
            if self._is_hiding:
                self._player.color_remap(3, 11)
            self._is_hiding = False

        return True

    def _handle_tile_effects(self) -> str:
        nx, ny = self._player.x, self._player.y

        alm = self._sprite_at(nx, ny, "alm")
        if alm and not self._alarm_active:
            self._alarm_active = True
            boost = self.current_level.get_data("alarm_boost") or 3
            self._alarm_timer = boost
            self.current_level.remove_sprite(alm)

        key_s = self._sprite_at(nx, ny, "key")
        if key_s:
            self._keys_held += 1
            self.current_level.remove_sprite(key_s)

        fake = self._sprite_at(nx, ny, "trp")
        if fake:
            return "trap"

        door = self._sprite_at(nx, ny, "ext")
        if door and self._keys_held >= self._keys_needed:
            return "exit"

        return "continue"

    def _tick_world(self):
        self._step_counter += 1
        cop_speed = self.current_level.get_data("cop_speed") or 1
        if self._step_counter % cop_speed == 0:
            self._move_cops()

        if self._alarm_active:
            self._alarm_timer -= 1
            if self._alarm_timer <= 0:
                self._alarm_active = False

    def _save_undo_snapshot(self):
        cop_pos = [(cs.x, cs.y) for cs in self._cop_sprites]
        cop_states_copy = copy.deepcopy(self._cop_states)
        key_positions = [
            (k.x, k.y) for k in self.current_level.get_sprites_by_tag("key")
        ]
        alarm_positions = [
            (a.x, a.y) for a in self.current_level.get_sprites_by_tag("alm")
        ]
        fake_positions = [
            (f.x, f.y) for f in self.current_level.get_sprites_by_tag("trp")
        ]
        self._undo_stack.append(
            {
                "player_pos": (self._player.x, self._player.y)
                if self._player
                else None,
                "lives": self._lives,
                "keys_held": self._keys_held,
                "mvc": self.mvc,
                "step_counter": self._step_counter,
                "is_hiding": self._is_hiding,
                "alarm_active": self._alarm_active,
                "alarm_timer": self._alarm_timer,
                "cop_positions": cop_pos,
                "cop_states": cop_states_copy,
                "key_positions": key_positions,
                "alarm_positions": alarm_positions,
                "fake_positions": fake_positions,
            }
        )

    def _restore_undo_snapshot(self):
        if not self._undo_stack:
            return
        state = self._undo_stack.pop()
        if state["player_pos"] is not None and self._player:
            self._player.set_position(state["player_pos"][0], state["player_pos"][1])
        self._lives = state["lives"]
        self._keys_held = state["keys_held"]
        self.mvc = state["mvc"]
        self._step_counter = state["step_counter"]
        self._is_hiding = state["is_hiding"]
        self._alarm_active = state["alarm_active"]
        self._alarm_timer = state["alarm_timer"]
        for i, cs in enumerate(self._cop_sprites):
            if i < len(state["cop_positions"]):
                cs.set_position(
                    state["cop_positions"][i][0], state["cop_positions"][i][1]
                )
        self._cop_states = state["cop_states"]

        for ks in list(self.current_level.get_sprites_by_tag("key")):
            self.current_level.remove_sprite(ks)
        for kx, ky in state["key_positions"]:
            self.current_level.add_sprite(sprites["key"].clone().set_position(kx, ky))

        for als in list(self.current_level.get_sprites_by_tag("alm")):
            self.current_level.remove_sprite(als)
        for ax, ay in state["alarm_positions"]:
            self.current_level.add_sprite(sprites["alarm"].clone().set_position(ax, ay))

        for fks in list(self.current_level.get_sprites_by_tag("trp")):
            self.current_level.remove_sprite(fks)
        for fx, fy in state["fake_positions"]:
            self.current_level.add_sprite(
                sprites["fake_door"].clone().set_position(fx, fy)
            )

        if self._is_hiding and self._player:
            self._player.color_remap(11, 3)
        elif self._player:
            self._player.color_remap(3, 11)

        self._update_camera()

    def _consume_move(self):
        self.mvc -= 1
        if self.mvc <= 0:
            self._lose_life()
            return True
        return False

    def step(self) -> None:
        if self.action.id == GameAction.RESET:
            self.complete_action()
            return

        if not self._player:
            self.complete_action()
            return

        if self.action.id == GameAction.ACTION7:
            self._restore_undo_snapshot()
            if self._consume_move():
                self.complete_action()
                return
            self._update_camera()
            self.complete_action()
            return

        self._save_undo_snapshot()

        dx = 0
        dy = 0
        if self.action.id == GameAction.ACTION1:
            dy = -1
        elif self.action.id == GameAction.ACTION2:
            dy = 1
        elif self.action.id == GameAction.ACTION3:
            dx = -1
        elif self.action.id == GameAction.ACTION4:
            dx = 1

        moved = self._handle_movement(dx, dy)
        if moved:
            tile_result = self._handle_tile_effects()

            if tile_result == "trap":
                self._lose_life()
                self.complete_action()
                return

            if tile_result == "exit":
                self._undo_stack = []
                self._level_done = True
                self.next_level()
                self.complete_action()
                return

        self._tick_world()

        if self._check_caught():
            self._lose_life()
            self.complete_action()
            return

        if self._consume_move():
            self.complete_action()
            return

        self._update_camera()
        self.complete_action()


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


def _encode_png(frame):
    h, w = frame.shape[:2]
    if frame.ndim == 2:
        raw = np.zeros((h, w, 3), dtype=np.uint8)
        raw[:] = frame[:, :, None]
    else:
        raw = frame.astype(np.uint8)
    rows = [b"\x00" + raw[y].tobytes() for y in range(h)]
    raw_data = b"".join(rows)
    compressed = zlib.compress(raw_data)

    def _chunk(ctype, data):
        c = ctype + data
        return (
            struct.pack(">I", len(data))
            + c
            + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        )

    buf = io.BytesIO()
    buf.write(b"\x89PNG\r\n\x1a\n")
    buf.write(_chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)))
    buf.write(_chunk(b"IDAT", compressed))
    buf.write(_chunk(b"IEND", b""))
    return buf.getvalue()


class PuzzleEnvironment:
    _ACTION_MAP = {
        "up": GameAction.ACTION1,
        "down": GameAction.ACTION2,
        "left": GameAction.ACTION3,
        "right": GameAction.ACTION4,
        "undo": GameAction.ACTION7,
        "reset": GameAction.RESET,
    }

    _VALID_ACTIONS = list(_ACTION_MAP.keys())

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self._engine = Tc48(seed=seed)
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

    def step(self, action: str) -> StepResult:
        e = self._engine

        if action == "reset":
            state = self.reset()
            return StepResult(
                state=state, reward=0.0, done=False, info={"action": "reset"}
            )

        if action not in self._ACTION_MAP:
            return StepResult(
                state=self._build_game_state(done=self._done),
                reward=0.0,
                done=self._done,
                info={"action": action, "reason": "invalid_action"},
            )

        self._last_action_was_reset = False
        self._total_turns += 1

        game_action = self._ACTION_MAP[action]
        info: dict = {"action": action}

        level_before = e.level_index

        frame = e.perform_action(ActionInput(id=game_action, data={}), raw=True)

        state_name = frame.state.name if frame and frame.state else ""
        game_won = state_name == "WIN"
        game_over = state_name == "GAME_OVER"

        total_levels = len(e._levels)
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

    def _build_text_observation(self) -> str:
        COLOR_CHARS = {
            0: ".",
            1: "+",
            2: "V",
            3: "H",
            4: "#",
            5: " ",
            6: "!",
            7: "v",
            8: "C",
            9: "9",
            10: "0",
            11: "T",
            12: "K",
            13: "D",
            14: "E",
            15: "F",
        }
        sprs = self._engine.current_level.get_sprites()
        idx_frame = self._engine.camera.render(sprs)
        lines = []
        h, w = idx_frame.shape[:2]
        for y in range(h):
            lines.append(
                "".join(COLOR_CHARS.get(int(idx_frame[y, x]), "?") for x in range(w))
            )
        e = self._engine
        lines.append(f"lives:{e._lives} moves:{e.mvc}/{e.mvl}")
        lines.append(f"keys:{e._keys_held}/{e._keys_needed}")
        lines.append(f"level:{e.level_index + 1}/{len(e._levels)}")
        if e._alarm_active:
            lines.append(f"alarm:{e._alarm_timer}")
        if e._is_hiding:
            lines.append("hiding:yes")
        return "\n".join(lines)

    def _build_game_state(self, done: bool = False) -> GameState:
        e = self._engine
        rgb = self.render(mode="rgb_array")
        img_bytes = _encode_png(rgb)
        valid_actions = self.get_actions() if not done else None
        return GameState(
            text_observation=self._build_text_observation(),
            image_observation=img_bytes,
            valid_actions=valid_actions,
            turn=self._total_turns,
            metadata={
                "total_levels": len(e._levels),
                "level_index": e.level_index,
                "levels_completed": getattr(e, "_score", 0),
                "game_over": getattr(getattr(e, "_state", None), "name", "")
                == "GAME_OVER",
                "done": done,
                "info": {},
            },
        )

    def get_actions(self) -> list[str]:
        if self._done:
            return ["reset"]
        return list(self._VALID_ACTIONS)

    def is_done(self) -> bool:
        return self._done

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError(f"Unsupported render mode: {mode}")
        sprs = self._engine.current_level.get_sprites()
        idx_frame = self._engine.camera.render(sprs)
        h, w = idx_frame.shape[:2]
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for idx in range(len(ARC_PALETTE)):
            mask = idx_frame == idx
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
    metadata: Dict[str, Any] = {"render_modes": ["rgb_array"], "render_fps": 5}
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

    def __init__(self, seed: int = 0, render_mode: Optional[str] = None) -> None:
        super().__init__()
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"Unsupported render_mode '{render_mode}'. "
                f"Supported: {self.metadata['render_modes']}"
            )
        self.render_mode = render_mode
        self._action_to_string = {i: a for i, a in enumerate(self.ACTION_LIST)}
        self._string_to_action = {a: i for i, a in enumerate(self.ACTION_LIST)}
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.OBS_HEIGHT, self.OBS_WIDTH, 3),
            dtype=np.uint8,
        )
        self.action_space = spaces.Discrete(len(self.ACTION_LIST))
        self._seed = seed
        self._env = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed
        self._env = PuzzleEnvironment(seed=self._seed)
        state = self._env.reset()
        return self._get_obs(), self._build_info(state)

    def step(self, action: int):
        action_str = self._action_to_string[int(action)]
        result = self._env.step(action_str)
        obs = self._get_obs()
        reward = result.reward
        terminated = result.done
        truncated = False
        info = self._build_info(result.state, result.info)
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._get_obs()
        return None

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None

    def action_mask(self):
        mask = np.zeros(len(self.ACTION_LIST), dtype=np.int8)
        if self._env is not None:
            for a in self._env.get_actions():
                idx = self._string_to_action.get(a)
                if idx is not None:
                    mask[idx] = 1
        return mask

    def _get_obs(self):
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

    def _build_info(self, state, step_info=None):
        info = {
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
