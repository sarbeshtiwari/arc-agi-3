import random
import struct
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite
from arcengine.enums import ActionInput, BlockingMode
from arcengine.interfaces import RenderableUserDisplay

BG = 5
LBOX = 5
_T = -1

PLAYER_SHAPE = [
    [_T, 9, _T],
    [9, 9, 9],
    [_T, 9, _T],
    [9, _T, 9],
]

CANDLE_SHAPE = [
    [11],
    [_T],
    [1],
    [1],
]

FADING_CANDLE_SHAPE = [
    [11],
    [_T],
    [11],
    [11],
]

KEY_SHAPE = [
    [12, 12, _T, _T, _T],
    [12, _T, 12, 12, 12],
    [12, 12, _T, 12, _T],
]

PINK_KEY_SHAPE = [
    [7, 7, _T, _T, _T],
    [7, _T, 7, 7, 7],
    [7, 7, _T, 7, _T],
]

DOOR_LOCKED = [
    [3, 3, 3],
    [3, _T, 3],
    [3, 3, 3],
    [3, _T, 3],
    [3, 3, 3],
]

DOOR_UNLOCKED = [
    [14, 14, 14],
    [14, _T, 14],
    [14, 14, 14],
    [14, _T, 14],
    [14, 14, 14],
]

SPIKE_SHAPE = [
    [_T, 8, _T],
    [8, 8, 8],
    [8, 8, 8],
]

POISON_SHAPE = [
    [_T, 6, 6, _T],
    [_T, 6, 6, _T],
    [6, 6, 6, 6],
    [6, 6, 6, 6],
    [6, 6, 6, 6],
]

LEVER_SHAPE = [
    [13, _T],
    [13, 13],
    [13, 13],
]

MIRROR_SHAPE = [
    [2, 2],
    [2, 2],
]

BOX_SHAPE = [
    [10, 10, 10, 10, 10],
    [10, 0, 0, 0, 10],
    [10, 0, 0, 0, 10],
    [10, 0, 0, 0, 10],
    [10, 10, 10, 10, 10],
]


def _offsets(shape):
    return frozenset(
        (c, r) for r, row in enumerate(shape) for c, v in enumerate(row) if v >= 0
    )


PLAYER_OFF = _offsets(PLAYER_SHAPE)
CANDLE_OFF = _offsets(CANDLE_SHAPE)
FADING_CANDLE_OFF = _offsets(FADING_CANDLE_SHAPE)
KEY_OFF = _offsets(KEY_SHAPE)
PINK_KEY_OFF = _offsets(PINK_KEY_SHAPE)
DOOR_OFF = _offsets(DOOR_LOCKED)
SPIKE_OFF = _offsets(SPIKE_SHAPE)
POISON_OFF = _offsets(POISON_SHAPE)
BOX_OFF = _offsets(BOX_SHAPE)
LEVER_OFF = _offsets(LEVER_SHAPE)
MIRROR_OFF = _offsets(MIRROR_SHAPE)

LEVELS = [
    {
        "player": (4, 54),
        "lights": [(30, 30), (50, 16)],
        "fading_lights": [(14, 20)],
        "key": (8, 8),
        "door": (55, 6),
        "fake_doors": [],
        "dead_doors": [(6, 6)],
        "spikes": [(22, 26), (26, 26), (30, 26)],
        "poisons": [(40, 44)],
        "moving_poisons": [((20, 54), (8, 54, 36, 54))],
        "moving_spikes": [],
        "mirrors": [(36, 16)],
        "levers": [(10, 44)],
        "pink_key": None,
        "key_teleport": [],
        "boxes": [(22, 38)],
        "wall_cells": (
            [(x, 44) for x in range(14, 40)] + [(x, 34) for x in range(8, 30)]
        ),
    },
    {
        "player": (4, 54),
        "lights": [(22, 18)],
        "fading_lights": [],
        "key": (4, 10),
        "door": (55, 50),
        "fake_doors": [],
        "dead_doors": [],
        "spikes": [(38, 24), (42, 24)],
        "poisons": [(32, 40)],
        "moving_poisons": [((16, 50), (8, 50, 40, 50))],
        "moving_spikes": [],
        "mirrors": [],
        "levers": [(48, 42)],
        "pink_key": None,
        "key_teleport": [],
        "boxes": [(10, 10)],
        "wall_cells": (
            [(x, 32) for x in range(12, 50)] + [(32, y) for y in range(8, 32)]
        ),
    },
    {
        "player": (4, 54),
        "lights": [(14, 14), (48, 28)],
        "fading_lights": [],
        "key": (36, 52),
        "door": (55, 6),
        "fake_doors": [(6, 50)],
        "dead_doors": [(6, 6)],
        "spikes": [(30, 36), (34, 36)],
        "poisons": [(24, 42), (44, 16)],
        "moving_poisons": [],
        "moving_spikes": [],
        "mirrors": [(40, 40)],
        "levers": [],
        "pink_key": (4, 40),
        "key_teleport": [],
        "boxes": [],
        "wall_cells": (
            [(x, 26) for x in range(18, 46)] + [(34, y) for y in range(6, 26)]
        ),
    },
    {
        "player": (4, 6),
        "lights": [(14, 44), (50, 18), (34, 34)],
        "fading_lights": [(40, 10)],
        "key": (6, 52),
        "door": (55, 28),
        "fake_doors": [(6, 50), (55, 50)],
        "dead_doors": [],
        "spikes": [(24, 26), (28, 26)],
        "poisons": [
            (6, 38),
            (22, 18),
            (44, 40),
            (52, 28),
            (28, 8),
        ],
        "moving_poisons": [
            ((8, 36), (4, 36, 30, 36)),
            ((40, 52), (34, 52, 56, 52)),
        ],
        "moving_spikes": [],
        "mirrors": [],
        "levers": [(20, 50)],
        "pink_key": (10, 20),
        "key_teleport": [(30, 6), (6, 40)],
        "boxes": [(26, 38)],
        "wall_cells": (
            [(x, 24) for x in range(10, 52)]
            + [(26, y) for y in range(8, 24)]
            + [(x, 44) for x in range(22, 52)]
        ),
    },
    {
        "player": (4, 54),
        "lights": [(12, 12), (50, 12), (12, 42), (50, 42)],
        "fading_lights": [(30, 20), (30, 48)],
        "key": (4, 8),
        "door": (55, 32),
        "fake_doors": [(6, 6), (55, 6)],
        "dead_doors": [(6, 32)],
        "spikes": [
            (20, 28),
            (24, 28),
            (28, 28),
            (36, 28),
            (40, 28),
            (44, 28),
        ],
        "poisons": [
            (6, 18),
            (20, 12),
            (56, 18),
            (42, 12),
            (6, 36),
            (20, 46),
            (56, 36),
            (42, 46),
        ],
        "moving_poisons": [((24, 50), (8, 50, 42, 50))],
        "moving_spikes": [((36, 8), (30, 8, 50, 8))],
        "mirrors": [(32, 44)],
        "levers": [(4, 30)],
        "pink_key": (8, 54),
        "key_teleport": [(8, 8), (50, 50)],
        "boxes": [(28, 6), (28, 52)],
        "wall_cells": (
            [(x, 28) for x in range(8, 56) if x != 32]
            + [(32, y) for y in range(6, 28)]
            + [(x, 48) for x in range(18, 46)]
        ),
    },
    {
        "player": (4, 54),
        "lights": [(10, 10), (52, 10), (10, 46), (52, 46)],
        "fading_lights": [(30, 16), (30, 50)],
        "key": (10, 54),
        "door": (55, 28),
        "fake_doors": [(6, 6), (6, 28), (55, 6)],
        "dead_doors": [],
        "spikes": [
            (18, 22),
            (22, 22),
            (40, 22),
            (44, 22),
            (18, 38),
            (22, 38),
            (40, 38),
            (44, 38),
        ],
        "poisons": [
            (4, 16),
            (16, 10),
            (46, 10),
            (58, 16),
            (4, 38),
            (16, 50),
            (46, 50),
            (58, 38),
            (28, 24),
            (34, 36),
        ],
        "moving_poisons": [
            ((16, 56), (6, 56, 36, 56)),
            ((44, 4), (30, 4, 56, 4)),
        ],
        "moving_spikes": [
            ((26, 14), (20, 14, 42, 14)),
            ((26, 46), (20, 46, 42, 46)),
        ],
        "mirrors": [(38, 52), (24, 8)],
        "levers": [(4, 30), (58, 30)],
        "pink_key": (32, 4),
        "key_teleport": [(6, 40), (56, 10), (32, 32)],
        "boxes": [],
        "wall_cells": (
            [(x, 20) for x in range(8, 28)]
            + [(x, 20) for x in range(36, 56)]
            + [(x, 40) for x in range(8, 28)]
            + [(x, 40) for x in range(36, 56)]
            + [(20, y) for y in range(6, 20)]
            + [(44, y) for y in range(6, 20)]
            + [(20, y) for y in range(40, 58)]
            + [(44, y) for y in range(40, 58)]
            + [(32, y) for y in range(22, 38)]
        ),
    },
]

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


def _png_chunk(ctype, data):
    c = ctype + data
    return (
        struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
    )


def _rgb_to_png(rgb):
    h, w = rgb.shape[0], rgb.shape[1]
    raw = b""
    for y in range(h):
        raw += b"\x00" + rgb[y].tobytes()
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return (
        sig
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(raw))
        + _png_chunk(b"IEND", b"")
    )


class FogOfWarDisplay(RenderableUserDisplay):
    def __init__(self, game_ref):
        self.game = game_ref

    def render_interface(self, frame):
        g = self.game
        cx = g.player_x + 1
        cy = g.player_y + 2

        total_radius = g.light_radius + g.fading_bonus
        if total_radius <= 0:
            r = 3
        else:
            r = total_radius * 4

        r_sq = r * r
        for row in range(64):
            dy = row - cy
            dy_sq = dy * dy
            for col in range(64):
                dx = col - cx
                if dy_sq + dx * dx > r_sq:
                    frame[row, col] = BG
        return frame


class LivesDisplay(RenderableUserDisplay):
    def __init__(self, game_ref):
        self.game = game_ref

    def render_interface(self, frame):
        g = self.game
        for i in range(3):
            color = 14 if i < g.lives else 4
            x = 2 + i * 4
            frame[1, x] = color
            frame[1, x + 1] = color
            frame[2, x] = color
            frame[2, x + 1] = color
        return frame


class FlashDisplay(RenderableUserDisplay):
    def __init__(self, game_ref):
        self.game = game_ref

    def render_interface(self, frame):
        g = self.game
        if g.flash_timer > 0:
            for x in range(64):
                frame[0, x] = 14
                frame[63, x] = 14
            for y in range(64):
                frame[y, 0] = 14
                frame[y, 63] = 14
        return frame


class Dr07(ARCBaseGame):
    GW, GH = 64, 64

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)
        self._seed = seed

        self.lives = 3
        self.light_radius = 0
        self.has_key = False
        self.lights_collected = 0

        self.player_x = 0
        self.player_y = 0

        self.light_positions = []
        self.poison_positions = []
        self.key_pos = None
        self.key_visible = False
        self.door_pos = None
        self.door_unlocked = False
        self.fake_door_positions = []
        self.fake_door_cells = set()
        self.spike_cells = set()
        self.blocked_cells = set()
        self.door_cells = set()
        self.total_lights = 0

        self.fading_light_positions = []
        self.fading_timer = 0
        self.fading_bonus = 0
        self.has_pink_key = False
        self.pink_key_pos = None
        self.mirror_active = False
        self.moving_poisons = []
        self.moving_spikes = []
        self.lever_positions = []
        self.levers_activated = False
        self.dead_door_positions = []
        self.dead_door_cells = set()
        self.key_teleport_positions = []
        self.key_teleport_idx = 0
        self.flash_timer = 0
        self.mirror_positions = []
        self._consecutive_resets = 0
        self._actions_since_reset = 0
        self._last_level_won = False
        self._undo_stack = []

        self.board_sprite = None
        self.fog_display = FogOfWarDisplay(self)
        self.lives_display = LivesDisplay(self)
        self.flash_display = FlashDisplay(self)

        levels = [
            Level(
                sprites=[],
                grid_size=(self.GW, self.GH),
                name=f"Level {i + 1}",
            )
            for i in range(len(LEVELS))
        ]

        camera = Camera(
            background=BG,
            letter_box=LBOX,
            width=self.GW,
            height=self.GH,
            interfaces=[self.fog_display, self.lives_display, self.flash_display],
        )

        super().__init__(
            "dr07", levels, camera, available_actions=[0, 1, 2, 3, 4, 7], seed=seed
        )

    def _player_cells(self, px=None, py=None):
        if px is None:
            px = self.player_x
        if py is None:
            py = self.player_y
        return {(px + dx, py + dy) for dx, dy in PLAYER_OFF}

    def _entity_cells(self, pos, offsets):
        ex, ey = pos
        return {(ex + dx, ey + dy) for dx, dy in offsets}

    def _overlaps(self, pcells, pos, offsets):
        return bool(pcells & self._entity_cells(pos, offsets))

    def _find_safe_start(self, cfg):
        blocked = set()
        for bx, by in cfg.get("boxes", []):
            for ddx, ddy in BOX_OFF:
                blocked.add((bx + ddx, by + ddy))
        for wc in cfg.get("wall_cells", []):
            blocked.add(wc)
        for dx_pos, dy_pos in cfg.get("dead_doors", []):
            for ddx, ddy in DOOR_OFF:
                blocked.add((dx_pos + ddx, dy_pos + ddy))
        danger = set()
        for sx, sy in cfg.get("spikes", []):
            for ddx, ddy in SPIKE_OFF:
                danger.add((sx + ddx, sy + ddy))
        for x in range(self.GW):
            for y in (0, 1, self.GH - 2, self.GH - 1):
                danger.add((x, y))
        for y in range(self.GH):
            for x in (0, 1, self.GW - 2, self.GW - 1):
                danger.add((x, y))
        for px, py in cfg.get("poisons", []):
            for ddx, ddy in POISON_OFF:
                danger.add((px + ddx, py + ddy))
        avoid = blocked | danger
        for attempt in range(50):
            rx = self._rng.randint(4, self.GW - 8)
            ry = self._rng.randint(4, self.GH - 10)
            pcells = {(rx + dx, ry + dy) for dx, dy in PLAYER_OFF}
            if not (pcells & avoid):
                return rx, ry
        return cfg["player"]

    def on_set_level(self, level):
        cfg = LEVELS[self.level_index]

        self.light_positions = list(cfg["lights"])
        self.total_lights = len(cfg["lights"])
        self.key_pos = cfg["key"]
        self.key_visible = False
        self.door_pos = cfg["door"]
        self.door_unlocked = False

        self.light_radius = 0
        self.has_key = False
        self.lights_collected = 0

        self.fading_light_positions = list(cfg.get("fading_lights", []))
        self.fading_timer = 0
        self.fading_bonus = 0

        pk = cfg.get("pink_key")
        self.pink_key_pos = pk
        self.has_pink_key = False

        self.mirror_active = False
        self.mirror_positions = list(cfg.get("mirrors", []))

        self.moving_poisons = []
        for (mpx, mpy), (bx0, by0, bx1, by1) in cfg.get("moving_poisons", []):
            self.moving_poisons.append(
                {
                    "x": mpx,
                    "y": mpy,
                    "dx": 1,
                    "dy": 0,
                    "bounds": (bx0, by0, bx1, by1),
                }
            )

        self.moving_spikes = []
        for (sx, sy), (bx0, by0, bx1, by1) in cfg.get("moving_spikes", []):
            self.moving_spikes.append(
                {
                    "x": sx,
                    "y": sy,
                    "dx": 1,
                    "dy": 0,
                    "bounds": (bx0, by0, bx1, by1),
                }
            )

        self.lever_positions = list(cfg.get("levers", []))
        self.levers_activated = False

        self.dead_door_positions = list(cfg.get("dead_doors", []))
        self.dead_door_cells = set()
        for dx_pos, dy_pos in self.dead_door_positions:
            for ddx, ddy in DOOR_OFF:
                self.dead_door_cells.add((dx_pos + ddx, dy_pos + ddy))

        self.key_teleport_positions = list(cfg.get("key_teleport", []))
        self.key_teleport_idx = 0

        self.flash_timer = 0

        self.fake_door_positions = list(cfg.get("fake_doors", []))
        self.fake_door_cells = set()
        for fx, fy in self.fake_door_positions:
            for ddx, ddy in DOOR_OFF:
                self.fake_door_cells.add((fx + ddx, fy + ddy))

        self.blocked_cells = set()
        for bx, by in cfg.get("boxes", []):
            for ddx, ddy in BOX_OFF:
                self.blocked_cells.add((bx + ddx, by + ddy))
        for wc in cfg.get("wall_cells", []):
            self.blocked_cells.add(wc)
        self.blocked_cells |= self.dead_door_cells

        self.door_cells = self._entity_cells(self.door_pos, DOOR_OFF)

        self.spike_cells = set()
        for sx, sy in cfg.get("spikes", []):
            for ddx, ddy in SPIKE_OFF:
                self.spike_cells.add((sx + ddx, sy + ddy))
        for x in range(self.GW):
            for y in (0, 1, self.GH - 2, self.GH - 1):
                self.spike_cells.add((x, y))
        for y in range(self.GH):
            for x in (0, 1, self.GW - 2, self.GW - 1):
                self.spike_cells.add((x, y))

        self.poison_positions = list(cfg.get("poisons", []))
        self._undo_stack = []
        self._last_level_won = False

        self.lives = 3

        self.player_x, self.player_y = self._find_safe_start(cfg)

        self.board_sprite = self._make_board_sprite()
        level.add_sprite(self.board_sprite)

    def handle_reset(self):
        if self._last_level_won or (self._consecutive_resets >= 1 and self._actions_since_reset == 0):
            self._consecutive_resets = 0
            self._actions_since_reset = 0
            self._last_level_won = False
            self.lives = 3
            self.full_reset()
        else:
            self._consecutive_resets += 1
            self._actions_since_reset = 0
            self.lives = 3
            self._restore_level()

    def _restore_level(self):
        cfg = LEVELS[self.level_index]

        self.light_positions = list(cfg["lights"])
        self.total_lights = len(cfg["lights"])
        self.key_pos = cfg["key"]
        self.key_visible = False
        self.door_pos = cfg["door"]
        self.door_unlocked = False

        self.light_radius = 0
        self.has_key = False
        self.lights_collected = 0

        self.fading_light_positions = list(cfg.get("fading_lights", []))
        self.fading_timer = 0
        self.fading_bonus = 0

        pk = cfg.get("pink_key")
        self.pink_key_pos = pk
        self.has_pink_key = False

        self.mirror_active = False
        self.mirror_positions = list(cfg.get("mirrors", []))

        self.moving_poisons = []
        for (mpx, mpy), (bx0, by0, bx1, by1) in cfg.get("moving_poisons", []):
            self.moving_poisons.append(
                {
                    "x": mpx,
                    "y": mpy,
                    "dx": 1,
                    "dy": 0,
                    "bounds": (bx0, by0, bx1, by1),
                }
            )

        self.moving_spikes = []
        for (sx, sy), (bx0, by0, bx1, by1) in cfg.get("moving_spikes", []):
            self.moving_spikes.append(
                {
                    "x": sx,
                    "y": sy,
                    "dx": 1,
                    "dy": 0,
                    "bounds": (bx0, by0, bx1, by1),
                }
            )

        self.lever_positions = list(cfg.get("levers", []))
        self.levers_activated = False

        self.dead_door_positions = list(cfg.get("dead_doors", []))
        self.dead_door_cells = set()
        for dx_pos, dy_pos in self.dead_door_positions:
            for ddx, ddy in DOOR_OFF:
                self.dead_door_cells.add((dx_pos + ddx, dy_pos + ddy))

        self.key_teleport_positions = list(cfg.get("key_teleport", []))
        self.key_teleport_idx = 0

        self.flash_timer = 0

        self.fake_door_positions = list(cfg.get("fake_doors", []))
        self.fake_door_cells = set()
        for fx, fy in self.fake_door_positions:
            for ddx, ddy in DOOR_OFF:
                self.fake_door_cells.add((fx + ddx, fy + ddy))

        self.blocked_cells = set()
        for bx, by in cfg.get("boxes", []):
            for ddx, ddy in BOX_OFF:
                self.blocked_cells.add((bx + ddx, by + ddy))
        for wc in cfg.get("wall_cells", []):
            self.blocked_cells.add(wc)
        self.blocked_cells |= self.dead_door_cells

        self.door_cells = self._entity_cells(self.door_pos, DOOR_OFF)

        self.spike_cells = set()
        for sx, sy in cfg.get("spikes", []):
            for ddx, ddy in SPIKE_OFF:
                self.spike_cells.add((sx + ddx, sy + ddy))
        for x in range(self.GW):
            for y in (0, 1, self.GH - 2, self.GH - 1):
                self.spike_cells.add((x, y))
        for y in range(self.GH):
            for x in (0, 1, self.GW - 2, self.GW - 1):
                self.spike_cells.add((x, y))

        self.poison_positions = list(cfg.get("poisons", []))
        self._undo_stack = []
        self._last_level_won = False

        self.player_x, self.player_y = self._find_safe_start(cfg)

        self._rebuild_board()

    def _can_move(self, nx, ny):
        new_cells = self._player_cells(nx, ny)
        for cx, cy in new_cells:
            if cx < 0 or cx >= self.GW or cy < 0 or cy >= self.GH:
                return False
        if new_cells & self.blocked_cells:
            return False
        if not self.door_unlocked and (new_cells & self.door_cells):
            return False
        if not self.door_unlocked and (new_cells & self.fake_door_cells):
            return False
        return True

    def _move_player(self, dx, dy):
        nx1 = self.player_x + dx
        ny1 = self.player_y + dy
        if not self._can_move(nx1, ny1):
            return

        nx2 = self.player_x + dx * 2
        ny2 = self.player_y + dy * 2
        if self._can_move(nx2, ny2):
            self.player_x = nx2
            self.player_y = ny2
        else:
            self.player_x = nx1
            self.player_y = ny1

    def _update_moving_entities(self):
        if self.levers_activated:
            return
        for mp in self.moving_poisons:
            bx0, by0, bx1, by1 = mp["bounds"]
            mp["x"] += mp["dx"]
            mp["y"] += mp["dy"]
            if mp["x"] <= bx0 or mp["x"] >= bx1:
                mp["dx"] = -mp["dx"]
            if mp["y"] <= by0 or mp["y"] >= by1:
                mp["dy"] = -mp["dy"]
        for ms in self.moving_spikes:
            bx0, by0, bx1, by1 = ms["bounds"]
            ms["x"] += ms["dx"]
            ms["y"] += ms["dy"]
            if ms["x"] <= bx0 or ms["x"] >= bx1:
                ms["dx"] = -ms["dx"]
            if ms["y"] <= by0 or ms["y"] >= by1:
                ms["dy"] = -ms["dy"]

    def _check_collisions(self):
        pcells = self._player_cells()

        for lpos in list(self.light_positions):
            if self._overlaps(pcells, lpos, CANDLE_OFF):
                self.light_positions.remove(lpos)
                self.lights_collected += 1
                self.light_radius = 2 + self.lights_collected
                if len(self.light_positions) == 0:
                    if self.has_pink_key or self.pink_key_pos is None:
                        self.key_visible = True
                return

        for fpos in list(self.fading_light_positions):
            if self._overlaps(pcells, fpos, FADING_CANDLE_OFF):
                self.fading_light_positions.remove(fpos)
                self.fading_bonus = 2
                self.fading_timer = 8
                return

        if self.pink_key_pos and not self.has_pink_key:
            if self._overlaps(pcells, self.pink_key_pos, PINK_KEY_OFF):
                self.has_pink_key = True
                self.pink_key_pos = None
                if len(self.light_positions) == 0:
                    self.key_visible = True
                return

        if self.key_visible and not self.has_key and self.key_pos:
            if self._overlaps(pcells, self.key_pos, KEY_OFF):
                self.has_key = True
                self.key_pos = None
                self.door_unlocked = True
                self.flash_timer = 2
                return

        for lv_pos in list(self.lever_positions):
            if self._overlaps(pcells, lv_pos, LEVER_OFF):
                self.lever_positions.remove(lv_pos)
                self.levers_activated = True
                return

        for m_pos in list(self.mirror_positions):
            if self._overlaps(pcells, m_pos, MIRROR_OFF):
                self.mirror_positions.remove(m_pos)
                self.mirror_active = True
                return

        if self.door_unlocked and self.fake_door_cells:
            if pcells & self.fake_door_cells:
                self._lose_life()
                return

        if self.door_unlocked:
            if pcells & self.door_cells:
                if self.level_index >= len(LEVELS) - 1:
                    self._last_level_won = True
                self.next_level()
                return

        if pcells & self.spike_cells:
            self._lose_life()
            return

        for ms in self.moving_spikes:
            ms_pos = (ms["x"], ms["y"])
            if self._overlaps(pcells, ms_pos, SPIKE_OFF):
                self._lose_life()
                return

        for ppos in list(self.poison_positions):
            if self._overlaps(pcells, ppos, POISON_OFF):
                self.poison_positions.remove(ppos)
                self._handle_poison_hit()
                return

        for mp in self.moving_poisons:
            mp_pos = (mp["x"], mp["y"])
            if self._overlaps(pcells, mp_pos, POISON_OFF):
                self._handle_poison_hit()
                return

    def _lose_life(self):
        self.lives -= 1
        if self.lives <= 0:
            self.lives = 0
            self.lose()
        else:
            self._restore_level()

    def _handle_poison_hit(self):
        if self.has_key and self.key_teleport_positions:
            self.has_key = False
            self.door_unlocked = False
            if self.key_teleport_idx < len(self.key_teleport_positions):
                self.key_pos = self.key_teleport_positions[self.key_teleport_idx]
                self.key_teleport_idx += 1
            else:
                self.key_pos = None
        elif self.light_radius == 0 and self.fading_bonus == 0:
            self._lose_life()
        else:
            self.light_radius = max(0, self.light_radius - 1)

    def _paint(self, grid, shape, x, y):
        for r, row in enumerate(shape):
            for c, v in enumerate(row):
                if v >= 0:
                    gx, gy = x + c, y + r
                    if 0 <= gx < self.GW and 0 <= gy < self.GH:
                        grid[gy][gx] = v

    def _render_border(self, grid):
        for x in range(self.GW):
            grid[0][x] = 8
            grid[self.GH - 1][x] = 8
        for y in range(1, self.GH - 1):
            grid[y][0] = 8
            grid[y][self.GW - 1] = 8

        for x in range(self.GW):
            c = 8 if (x % 4) < 2 else BG
            grid[1][x] = c
            grid[self.GH - 2][x] = c
        for y in range(2, self.GH - 2):
            c = 8 if (y % 4) < 2 else BG
            grid[y][1] = c
            grid[y][self.GW - 2] = c

    def _render_grid(self):
        grid = [[BG] * self.GW for _ in range(self.GH)]

        self._render_border(grid)

        cfg = LEVELS[self.level_index]

        for wx, wy in cfg.get("wall_cells", []):
            if 2 <= wx < self.GW - 2 and 2 <= wy < self.GH - 2:
                grid[wy][wx] = 10

        for bx, by in cfg.get("boxes", []):
            self._paint(grid, BOX_SHAPE, bx, by)

        for sx, sy in cfg.get("spikes", []):
            self._paint(grid, SPIKE_SHAPE, sx, sy)

        for ppx, ppy in self.poison_positions:
            self._paint(grid, POISON_SHAPE, ppx, ppy)

        for mp in self.moving_poisons:
            self._paint(grid, POISON_SHAPE, mp["x"], mp["y"])

        for ms in self.moving_spikes:
            self._paint(grid, SPIKE_SHAPE, ms["x"], ms["y"])

        for lx, ly in self.lever_positions:
            self._paint(grid, LEVER_SHAPE, lx, ly)

        for mx, my in self.mirror_positions:
            self._paint(grid, MIRROR_SHAPE, mx, my)

        for lx, ly in self.light_positions:
            self._paint(grid, CANDLE_SHAPE, lx, ly)

        for fx, fy in self.fading_light_positions:
            self._paint(grid, FADING_CANDLE_SHAPE, fx, fy)

        if self.pink_key_pos and not self.has_pink_key:
            self._paint(
                grid, PINK_KEY_SHAPE, self.pink_key_pos[0], self.pink_key_pos[1]
            )

        if self.key_visible and self.key_pos:
            self._paint(grid, KEY_SHAPE, self.key_pos[0], self.key_pos[1])

        for ddx, ddy in self.dead_door_positions:
            if self.door_unlocked:
                self._paint(grid, DOOR_UNLOCKED, ddx, ddy)
            else:
                self._paint(grid, DOOR_LOCKED, ddx, ddy)

        if self.door_pos:
            if self.door_unlocked:
                self._paint(grid, DOOR_UNLOCKED, self.door_pos[0], self.door_pos[1])
            else:
                self._paint(grid, DOOR_LOCKED, self.door_pos[0], self.door_pos[1])

        for fx, fy in self.fake_door_positions:
            if self.door_unlocked:
                self._paint(grid, DOOR_UNLOCKED, fx, fy)
            else:
                self._paint(grid, DOOR_LOCKED, fx, fy)

        self._paint(grid, PLAYER_SHAPE, self.player_x, self.player_y)

        return grid

    def _make_board_sprite(self):
        pixels = self._render_grid()
        return Sprite(
            pixels=pixels,
            name="board",
            x=0,
            y=0,
            layer=0,
            blocking=BlockingMode.NOT_BLOCKED,
            collidable=False,
        )

    def _rebuild_board(self):
        if self.board_sprite is None:
            return
        pixels = self._render_grid()
        for y in range(self.GH):
            for x in range(self.GW):
                self.board_sprite.pixels[y, x] = pixels[y][x]

    def _save_undo(self):
        self._undo_stack.append(
            {
                "px": self.player_x,
                "py": self.player_y,
                "lr": self.light_radius,
                "lc": self.lights_collected,
                "hk": self.has_key,
                "kv": self.key_visible,
                "kp": self.key_pos,
                "du": self.door_unlocked,
                "lp": list(self.light_positions),
                "flp": list(self.fading_light_positions),
                "ft": self.fading_timer,
                "fb": self.fading_bonus,
                "hpk": self.has_pink_key,
                "pkp": self.pink_key_pos,
                "ma": self.mirror_active,
                "mps": list(self.mirror_positions),
                "pp": list(self.poison_positions),
                "mvp": [dict(m) for m in self.moving_poisons],
                "mvs": [dict(m) for m in self.moving_spikes],
                "lvp": list(self.lever_positions),
                "la": self.levers_activated,
                "flt": self.flash_timer,
                "kti": self.key_teleport_idx,
                "asr": self._actions_since_reset,
            }
        )

    def _restore_undo(self):
        if not self._undo_stack:
            return
        s = self._undo_stack.pop()
        self.player_x = s["px"]
        self.player_y = s["py"]
        self.light_radius = s["lr"]
        self.lights_collected = s["lc"]
        self.has_key = s["hk"]
        self.key_visible = s["kv"]
        self.key_pos = s["kp"]
        self.door_unlocked = s["du"]
        self.light_positions = s["lp"]
        self.fading_light_positions = s["flp"]
        self.fading_timer = s["ft"]
        self.fading_bonus = s["fb"]
        self.has_pink_key = s["hpk"]
        self.pink_key_pos = s["pkp"]
        self.mirror_active = s["ma"]
        self.mirror_positions = s["mps"]
        self.poison_positions = s["pp"]
        self.moving_poisons = s["mvp"]
        self.moving_spikes = s["mvs"]
        self.lever_positions = s["lvp"]
        self.levers_activated = s["la"]
        self.flash_timer = s["flt"]
        self.key_teleport_idx = s["kti"]
        self._actions_since_reset = s["asr"]

    def step(self):
        aid = self.action.id

        if aid == GameAction.RESET:
            self.complete_action()
            return

        if aid == GameAction.ACTION7:
            self._restore_undo()
            self._actions_since_reset += 1
            self._rebuild_board()
            self.complete_action()
            return

        self._save_undo()
        self._actions_since_reset += 1

        level_before = self.level_index

        dx, dy = {
            GameAction.ACTION1: (0, -1),
            GameAction.ACTION2: (0, 1),
            GameAction.ACTION3: (-1, 0),
            GameAction.ACTION4: (1, 0),
        }.get(aid, (0, 0))

        if self.mirror_active:
            dx = -dx
            dy = -dy

        if dx != 0 or dy != 0:
            self._move_player(dx, dy)

        if self.fading_timer > 0:
            self.fading_timer -= 1
            if self.fading_timer <= 0:
                self.fading_bonus = 0

        self._update_moving_entities()

        self._check_collisions()

        if self.level_index == level_before:
            self._rebuild_board()

        if self.flash_timer > 0:
            self.flash_timer -= 1

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
        self._seed = seed
        self._engine: Optional[Dr07] = Dr07(seed=seed)
        self._total_turns = 0
        self._done = False
        self._last_action_was_reset = False

    def _eng(self) -> "Dr07":
        if self._engine is None:
            raise RuntimeError("Environment has been closed")
        return self._engine

    def reset(self) -> GameState:
        action_input = ActionInput(id=GameAction.RESET)
        self._eng().perform_action(action_input)
        self._last_action_was_reset = True
        self._total_turns = 0
        self._done = False
        return self._build_state()

    def get_actions(self) -> List[str]:
        if self._done:
            return ["reset"]
        return list(self._VALID_ACTIONS)

    def step(self, action: str) -> StepResult:
        self._last_action_was_reset = False

        eng = self._eng()
        total_levels = len(LEVELS)

        if action == "reset":
            state = self.reset()
            return StepResult(
                state=state,
                reward=0.0,
                done=False,
                info={"action": "reset"},
            )

        if self._done:
            return StepResult(
                state=self._build_state(),
                reward=0.0,
                done=True,
            )

        ga = self._ACTION_MAP.get(action)
        if ga is None:
            return StepResult(
                state=self._build_state(),
                reward=0.0,
                done=self._done,
            )

        self._total_turns += 1

        level_before = eng.level_index

        action_input = ActionInput(id=ga)
        frame_data = eng.perform_action(action_input, raw=True)

        state_val = frame_data.state.value
        reward = 0.0
        info: Dict = {"action": action}

        if state_val == "WIN":
            reward = (eng.level_index - level_before) / total_levels
            self._done = True
            info["reason"] = "game_complete"
            info["level_index"] = eng.level_index
            info["total_levels"] = total_levels
        elif eng.level_index > level_before:
            reward = 1.0 / total_levels
        elif state_val == "GAME_OVER":
            self._done = True
            info["reason"] = "death"

        return StepResult(
            state=self._build_state(),
            reward=reward,
            done=self._done,
            info=info,
        )

    def is_done(self) -> bool:
        return self._done

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError(f"Unsupported render mode: {mode}")
        eng = self._eng()
        frame = eng.camera.render(eng.current_level.get_sprites())
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        for idx, color in enumerate(ARC_PALETTE):
            mask = frame == idx
            rgb[mask] = color
        return rgb

    def close(self) -> None:
        self._engine = None

    def _build_state(self) -> GameState:
        eng = self._eng()
        text = self._build_text()
        image = self._build_image()
        valid = None if self._done else self.get_actions()
        return GameState(
            text_observation=text,
            image_observation=image,
            valid_actions=valid,
            turn=self._total_turns,
            metadata={
                "total_levels": len(self._engine._levels),
                "level": eng.level_index + 1,
                "player": [eng.player_x, eng.player_y],
                "lives": eng.lives,
                "has_key": eng.has_key,
                "door_unlocked": eng.door_unlocked,
                "lights_collected": eng.lights_collected,
                "total_lights": eng.total_lights,
                "light_radius": eng.light_radius,
                "mirror_active": eng.mirror_active,
                "levels_completed": getattr(self._engine, "_score", 0),
                "level_index": eng.level_index,
                "game_over": getattr(getattr(self._engine, "_state", None), "name", "") == "GAME_OVER",
            },
        )

    def _build_text(self) -> str:
        g = self._eng()
        parts = []
        parts.append(f"LEVEL: {g.level_index + 1}/{len(LEVELS)}")
        parts.append(f"LIVES: {g.lives}")
        parts.append(f"PLAYER: {g.player_x} {g.player_y}")
        parts.append(f"LIGHT_RADIUS: {g.light_radius}")
        parts.append(f"LIGHTS: {g.lights_collected}/{g.total_lights}")
        if g.fading_bonus > 0:
            parts.append(f"FADING_BONUS: {g.fading_bonus} ({g.fading_timer} left)")
        parts.append(f"HAS_KEY: {g.has_key}")
        parts.append(f"KEY_VISIBLE: {g.key_visible}")
        if g.key_pos:
            parts.append(f"KEY_POS: {g.key_pos[0]} {g.key_pos[1]}")
        if g.door_pos:
            parts.append(f"DOOR: {g.door_pos[0]} {g.door_pos[1]}")
        parts.append(f"DOOR_UNLOCKED: {g.door_unlocked}")
        if g.mirror_active:
            parts.append("MIRROR: ACTIVE (controls reversed)")
        if g.pink_key_pos and not g.has_pink_key:
            parts.append(f"PINK_KEY: {g.pink_key_pos[0]} {g.pink_key_pos[1]}")
        elif g.has_pink_key:
            parts.append("PINK_KEY: collected")
        if g.light_positions:
            lp = " ".join(f"({lx},{ly})" for lx, ly in g.light_positions)
            parts.append(f"CANDLES: {lp}")
        if g.fading_light_positions:
            fp = " ".join(f"({fx},{fy})" for fx, fy in g.fading_light_positions)
            parts.append(f"FADING_CANDLES: {fp}")
        if g.lever_positions:
            lvp = " ".join(f"({lx},{ly})" for lx, ly in g.lever_positions)
            parts.append(f"LEVERS: {lvp}")
        if g.levers_activated:
            parts.append("LEVERS_ACTIVATED: True")
        sp = " ".join(
            f"({sx},{sy})" for sx, sy in LEVELS[g.level_index].get("spikes", [])
        )
        if sp:
            parts.append(f"SPIKES: {sp}")
        if g.poison_positions:
            pp = " ".join(f"({px},{py})" for px, py in g.poison_positions)
            parts.append(f"POISONS: {pp}")
        if g.moving_poisons:
            mpp = " ".join(f"({mp['x']},{mp['y']})" for mp in g.moving_poisons)
            parts.append(f"MOVING_POISONS: {mpp}")
        if g.moving_spikes:
            msp = " ".join(f"({ms['x']},{ms['y']})" for ms in g.moving_spikes)
            parts.append(f"MOVING_SPIKES: {msp}")
        if g.fake_door_positions:
            fdp = " ".join(f"({fx},{fy})" for fx, fy in g.fake_door_positions)
            parts.append(f"FAKE_DOORS: {fdp}")
        if g.dead_door_positions:
            ddp = " ".join(f"({ddx},{ddy})" for ddx, ddy in g.dead_door_positions)
            parts.append(f"DEAD_DOORS: {ddp}")
        cfg = LEVELS[g.level_index]
        wc = cfg.get("wall_cells", [])
        if wc:
            wp = " ".join(f"({wx},{wy})" for wx, wy in wc)
            parts.append(f"WALLS: {wp}")
        bxs = cfg.get("boxes", [])
        if bxs:
            bp = " ".join(f"({bx},{by})" for bx, by in bxs)
            parts.append(f"BOXES: {bp}")
        if g.key_teleport_positions:
            ktp = " ".join(
                f"({kx},{ky})" for kx, ky in g.key_teleport_positions
            )
            parts.append(f"KEY_TELEPORTS: {ktp}")
        if g.mirror_positions:
            mrp = " ".join(f"({mx},{my})" for mx, my in g.mirror_positions)
            parts.append(f"MIRRORS: {mrp}")
        return "\n".join(parts)

    def _build_image(self) -> bytes:
        eng = self._eng()
        frame = eng.camera.render(eng.current_level.get_sprites())
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        for idx, color in enumerate(ARC_PALETTE):
            mask = frame == idx
            rgb[mask] = color
        return _rgb_to_png(rgb)



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
