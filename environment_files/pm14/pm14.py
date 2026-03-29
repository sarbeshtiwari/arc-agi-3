import struct
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from arcengine import ActionInput, ARCBaseGame, Camera, GameAction, Level, Sprite
from arcengine.enums import BlockingMode
from arcengine.enums import GameState as _EngineState
from arcengine.interfaces import RenderableUserDisplay
from gymnasium import spaces

BG = 0
_T = -1

REGULAR = "regular"
SPECIAL = "special"
VERT = "v"
HORIZ = "h"

PLAYER_SHAPE = [
    [14, 14],
    [14, 14],
]

GATE_SHAPE = [
    [11, 11, 11],
    [11, _T, 11],
    [11, _T, 11],
    [11, 11, 11],
]


def _offsets(shape):
    return frozenset(
        (c, r) for r, row in enumerate(shape) for c, v in enumerate(row) if v >= 0
    )


PLAYER_OFF = _offsets(PLAYER_SHAPE)


class Portal:
    def __init__(self, sprite, portal_type, portal_id, orientation=VERT):
        self.sprite = sprite
        self.portal_type = portal_type
        self.portal_id = portal_id
        self.orientation = orientation
        self.partner: Optional["Portal"] = None
        self.destroyed = False
        self.color_at_destruction = None

    def overlaps_player(self, px, py):
        sx, sy = self.sprite.x, self.sprite.y
        if self.orientation == VERT:
            col_overlap = (sx - 1) <= px <= sx
            row_overlap = py < sy + 4 and sy < py + 2
        else:
            col_overlap = px < sx + 4 and sx < px + 2
            row_overlap = (sy - 1) <= py <= sy
        return col_overlap and row_overlap

    def destroy(self):
        self.destroyed = True
        self.color_at_destruction = int(self.sprite.pixels[0][0])
        self.sprite.pixels[:] = -1


class PressureTile:
    def __init__(self, sprite, x, y, orientation=VERT):
        self.sprite = sprite
        self.x = x
        self.y = y
        self.orientation = orientation
        self.triggered = False
        self.destroyed_portals: List[Portal] = []

    def overlaps_player(self, px, py):
        if self.orientation == VERT:
            col_overlap = (self.x - 1) <= px <= self.x
            row_overlap = py < self.y + 4 and self.y < py + 2
        else:
            col_overlap = px < self.x + 4 and self.x < px + 2
            row_overlap = (self.y - 1) <= py <= self.y
        return col_overlap and row_overlap

    def trigger(self):
        self.triggered = True
        if self.orientation == VERT:
            for r in range(4):
                self.sprite.pixels[r, 0] = 5
        else:
            for c in range(4):
                self.sprite.pixels[0, c] = 5


class ReliefTile:
    def __init__(self, sprite, x, y, orientation=VERT):
        self.sprite = sprite
        self.x = x
        self.y = y
        self.orientation = orientation
        self.used = False

    def overlaps_player(self, px, py):
        if self.orientation == VERT:
            col_overlap = (self.x - 1) <= px <= self.x
            row_overlap = py < self.y + 4 and self.y < py + 2
        else:
            col_overlap = px < self.x + 4 and self.x < px + 2
            row_overlap = (self.y - 1) <= py <= self.y
        return col_overlap and row_overlap

    def use(self):
        self.used = True
        if self.orientation == VERT:
            for r in range(4):
                self.sprite.pixels[r, 0] = 5
        else:
            for c in range(4):
                self.sprite.pixels[0, c] = 5


class MazeHUD(RenderableUserDisplay):
    def __init__(self, game_ref):
        super().__init__()
        self.game = game_ref

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        g = self.game
        ratio = g.moves_remaining / max(g.current_budget, 1)
        bar_len = int(ratio * 57)
        bar_color = 8 if ratio < 0.25 else 12
        for col in range(57):
            frame[63, col] = bar_color if col < bar_len else 4
        frame[63, 57] = 0
        for i in range(3):
            col = 58 + i * 2
            color = 14 if i < g.lives else 5
            frame[63, col] = color
            frame[63, col + 1] = color
        return frame


class Pm14(ARCBaseGame):
    GW, GH = 64, 64
    LEVEL_BUDGETS = [200, 220, 240, 260, 280, 300]

    def __init__(self, seed: int = 0) -> None:
        self.moves_remaining = 120
        self.current_budget = 120
        self.player_x = 0
        self.player_y = 0
        self.gate_x = 0
        self.gate_y = 0
        self.portals: List[Portal] = []
        self.pressure_tiles: List[PressureTile] = []
        self.relief_tiles: List[ReliefTile] = []
        self.wall_set: set = set()
        self.last_used_portal: Optional[Portal] = None
        self._portal_id_counter = 0
        self.special_last_two_orange: List[Portal] = []
        self.special_last_two_red: List[Portal] = []
        self._skip_portal: Optional[Portal] = None
        self.destroyed_portal_stack: List[PressureTile] = []
        self.last_dx = 0
        self.last_dy = 0
        self.lives = 3
        self._is_restart = False
        self._undo_stack: List[dict] = []

        self.hud = MazeHUD(self)

        levels = [
            Level(sprites=[], grid_size=(64, 64), name=f"Level {i + 1}")
            for i in range(6)
        ]

        camera = Camera(
            background=BG,
            letter_box=0,
            width=64,
            height=64,
            interfaces=[self.hud],
        )

        super().__init__(
            game_id="pm14",
            levels=levels,
            camera=camera,
            available_actions=[0, 1, 2, 3, 4, 5, 7],
            seed=seed,
        )

    def _lose_life(self):
        self.lives -= 1
        if self.lives > 0:
            self._is_restart = True
            self.level_reset()
        else:
            self.lose()

    def _next_id(self):
        self._portal_id_counter += 1
        return self._portal_id_counter

    def _reset_level_state(self):
        self.portals = []
        self.pressure_tiles = []
        self.relief_tiles = []
        self.wall_set = set()
        self.last_used_portal = None
        self._portal_id_counter = 0
        self.special_last_two_orange = []
        self.special_last_two_red = []
        self._skip_portal = None
        self.destroyed_portal_stack = []
        self.last_dx = 0
        self.last_dy = 0
        self._undo_stack = []
        self.current_budget = self.LEVEL_BUDGETS[self.level_index]
        self.moves_remaining = self.current_budget

    def _add_level_sprites(self, level):
        wall_pixels = [[-1] * self.GW for _ in range(self.GH)]
        for wx, wy in self.wall_set:
            wall_pixels[wy][wx] = 5
        level.add_sprite(
            Sprite(
                pixels=wall_pixels,
                name="wall",
                x=0,
                y=0,
                layer=0,
                blocking=BlockingMode.NOT_BLOCKED,
                tags=["sys_static"],
            )
        )
        level.add_sprite(
            Sprite(
                pixels=GATE_SHAPE,
                name="gate",
                x=self.gate_x,
                y=self.gate_y,
                layer=1,
                blocking=BlockingMode.NOT_BLOCKED,
            )
        )
        level.add_sprite(
            Sprite(
                pixels=PLAYER_SHAPE,
                name="player",
                x=self.player_x,
                y=self.player_y,
                layer=2,
            )
        )

    def on_set_level(self, level):
        if not self._is_restart:
            self.lives = 3
        self._is_restart = False
        self._reset_level_state()
        self._load_level(self.level_index, level)
        self._add_level_sprites(level)

    def _make_portal(self, level, x, y, color, portal_type, orientation=VERT):
        pid = self._next_id()
        if orientation == VERT:
            pixels = [[color], [color], [color], [color]]
        else:
            pixels = [[color, color, color, color]]
        sprite = Sprite(
            pixels=pixels,
            name=f"portal_{pid}",
            x=x,
            y=y,
            layer=1,
            blocking=BlockingMode.NOT_BLOCKED,
            tags=["portal"],
        )
        level.add_sprite(sprite)
        portal = Portal(sprite, portal_type, pid, orientation)
        self.portals.append(portal)
        return portal

    def _make_portal_pair(
        self, level, ax, ay, bx, by, color, portal_type, orient_a=VERT, orient_b=VERT
    ):
        a = self._make_portal(level, ax, ay, color, portal_type, orient_a)
        b = self._make_portal(level, bx, by, color, portal_type, orient_b)
        a.partner = b
        b.partner = a
        return a, b

    def _add_pressure(self, level, tx, ty, orientation=VERT):
        if orientation == VERT:
            pixels = [[1], [1], [1], [1]]
        else:
            pixels = [[1, 1, 1, 1]]
        pt_sprite = Sprite(
            pixels=pixels,
            name=f"pressure_{tx}_{ty}",
            x=tx,
            y=ty,
            layer=1,
            blocking=BlockingMode.NOT_BLOCKED,
        )
        level.add_sprite(pt_sprite)
        pt = PressureTile(pt_sprite, tx, ty, orientation)
        self.pressure_tiles.append(pt)

    def _add_relief(self, level, tx, ty, orientation=VERT):
        if orientation == VERT:
            pixels = [[2], [2], [2], [2]]
        else:
            pixels = [[2, 2, 2, 2]]
        rt_sprite = Sprite(
            pixels=pixels,
            name=f"relief_{tx}_{ty}",
            x=tx,
            y=ty,
            layer=1,
            blocking=BlockingMode.NOT_BLOCKED,
        )
        level.add_sprite(rt_sprite)
        rt = ReliefTile(rt_sprite, tx, ty, orientation)
        self.relief_tiles.append(rt)

    def _player_cells(self, px, py):
        return {(px + dx, py + dy) for dx, dy in PLAYER_OFF}

    def _can_move_to(self, nx, ny):
        cells = self._player_cells(nx, ny)
        for cx, cy in cells:
            if cx < 0 or cx >= self.GW or cy < 0 or cy >= self.GH:
                return False
        if cells & self.wall_set:
            return False
        return True

    def _get_pressure_at(self, px, py):
        for pt in self.pressure_tiles:
            if pt.triggered:
                continue
            if pt.overlaps_player(px, py):
                return pt
        return None

    def _get_relief_at(self, px, py):
        for rt in self.relief_tiles:
            if rt.used:
                continue
            if rt.overlaps_player(px, py):
                return rt
        return None

    def _get_portal_at(self, px, py):
        for portal in self.portals:
            if portal.destroyed:
                continue
            if portal.overlaps_player(px, py):
                return portal
        return None

    def _can_enter_portal(self, portal):
        if portal.orientation == VERT:
            return self.last_dx != 0
        else:
            return self.last_dy != 0

    def _check_gate(self, player):
        gates = self.current_level.get_sprites_by_name("gate")
        if not gates:
            return False
        g = gates[0]
        col_overlap = player.x < g.x + 3 and g.x < player.x + 2
        row_overlap = player.y < g.y + 4 and g.y < player.y + 2
        return col_overlap and row_overlap

    def _do_special_swap(self, used_portal):
        color = int(used_portal.sprite.pixels[0][0])

        if color == 12:
            tracker = self.special_last_two_orange
        elif color == 8:
            tracker = self.special_last_two_red
        else:
            return

        if used_portal not in tracker:
            tracker.append(used_portal)
        if len(tracker) > 2:
            tracker.pop(0)

        if len(tracker) == 2:
            p1 = tracker[0]
            p2 = tracker[1]
            if (
                p1 is not p2
                and not p1.destroyed
                and not p2.destroyed
                and p1.partner is not None
                and p2.partner is not None
            ):
                p1_old_partner = p1.partner
                p2_old_partner = p2.partner
                p1.partner = p2_old_partner
                p2.partner = p1_old_partner
                p1_old_partner.partner = p2
                p2_old_partner.partner = p1
                if color == 12:
                    self.special_last_two_orange = []
                else:
                    self.special_last_two_red = []

    def _snapshot(self):
        player = self.current_level.get_sprites_by_name("player")[0]
        portal_snaps = []
        for p in self.portals:
            portal_snaps.append(
                (
                    p.destroyed,
                    p.color_at_destruction,
                    np.copy(p.sprite.pixels),
                    self.portals.index(p.partner) if p.partner else None,
                )
            )
        pressure_snaps = []
        for pt in self.pressure_tiles:
            pressure_snaps.append(
                (
                    pt.triggered,
                    np.copy(pt.sprite.pixels),
                    [self.portals.index(dp) for dp in pt.destroyed_portals],
                )
            )
        relief_snaps = []
        for rt in self.relief_tiles:
            relief_snaps.append(
                (
                    rt.used,
                    np.copy(rt.sprite.pixels),
                )
            )
        return {
            "px": player.x,
            "py": player.y,
            "mr": self.moves_remaining,
            "portals": portal_snaps,
            "pressure": pressure_snaps,
            "relief": relief_snaps,
            "lup": self.portals.index(self.last_used_portal)
            if self.last_used_portal
            else None,
            "sp": self.portals.index(self._skip_portal) if self._skip_portal else None,
            "dps": [
                self.pressure_tiles.index(pt) for pt in self.destroyed_portal_stack
            ],
            "ldx": self.last_dx,
            "ldy": self.last_dy,
            "slto": [self.portals.index(p) for p in self.special_last_two_orange],
            "sltr": [self.portals.index(p) for p in self.special_last_two_red],
        }

    def _restore(self, snap):
        player = self.current_level.get_sprites_by_name("player")[0]
        player.set_position(snap["px"], snap["py"])
        for i, (destroyed, cad, pix, pidx) in enumerate(snap["portals"]):
            p = self.portals[i]
            p.destroyed = destroyed
            p.color_at_destruction = cad
            p.sprite.pixels[:] = pix
            p.partner = self.portals[pidx] if pidx is not None else None
        for i, (triggered, pix, dp_idxs) in enumerate(snap["pressure"]):
            pt = self.pressure_tiles[i]
            pt.triggered = triggered
            pt.sprite.pixels[:] = pix
            pt.destroyed_portals = [self.portals[j] for j in dp_idxs]
        for i, (used, pix) in enumerate(snap["relief"]):
            rt = self.relief_tiles[i]
            rt.used = used
            rt.sprite.pixels[:] = pix
        self.last_used_portal = (
            self.portals[snap["lup"]] if snap["lup"] is not None else None
        )
        self._skip_portal = self.portals[snap["sp"]] if snap["sp"] is not None else None
        self.destroyed_portal_stack = [self.pressure_tiles[j] for j in snap["dps"]]
        self.last_dx = snap["ldx"]
        self.last_dy = snap["ldy"]
        self.special_last_two_orange = [self.portals[j] for j in snap["slto"]]
        self.special_last_two_red = [self.portals[j] for j in snap["sltr"]]

    def _handle_undo(self):
        self.moves_remaining -= 1
        if self._undo_stack:
            saved_moves = self.moves_remaining
            self._restore(self._undo_stack.pop())
            self.moves_remaining = saved_moves
        if self.moves_remaining <= 0:
            self._lose_life()

    def _action_to_delta(self, action):
        return {
            GameAction.ACTION1: (0, -1),
            GameAction.ACTION2: (0, 1),
            GameAction.ACTION3: (-1, 0),
            GameAction.ACTION4: (1, 0),
        }.get(action)

    def _move_player(self, player, dx, dy):
        nx, ny = player.x + dx, player.y + dy
        if self._can_move_to(nx, ny):
            player.set_position(nx, ny)
            self.last_dx = dx
            self.last_dy = dy
        else:
            self.last_dx = 0
            self.last_dy = 0
        self.moves_remaining -= 1

    def _process_pressure(self, px, py):
        pt = self._get_pressure_at(px, py)
        if not pt:
            return
        pt.trigger()
        if not self.last_used_portal or self.last_used_portal.destroyed:
            return
        partner = self.last_used_portal.partner
        destroyed_pair = [self.last_used_portal]
        self.last_used_portal.destroy()
        if partner and not partner.destroyed:
            partner.destroy()
            destroyed_pair.append(partner)
        pt.destroyed_portals = destroyed_pair
        self.destroyed_portal_stack.append(pt)
        self.last_used_portal = None

    def _restore_pressure(self, last_pt):
        last_pt.triggered = False
        if last_pt.orientation == VERT:
            for r in range(4):
                last_pt.sprite.pixels[r, 0] = 1
        else:
            for c in range(4):
                last_pt.sprite.pixels[0, c] = 1
        for portal in last_pt.destroyed_portals:
            portal.destroyed = False
            color = portal.color_at_destruction
            if portal.orientation == VERT:
                for r in range(4):
                    portal.sprite.pixels[r, 0] = color
            else:
                for c in range(4):
                    portal.sprite.pixels[0, c] = color
        last_pt.destroyed_portals = []

    def _process_relief(self, px, py):
        rt = self._get_relief_at(px, py)
        if rt and self.destroyed_portal_stack:
            rt.use()
            self._restore_pressure(self.destroyed_portal_stack.pop())

    def _process_portals(self, player, px, py):
        if self._skip_portal and not self._skip_portal.overlaps_player(px, py):
            self._skip_portal = None
        portal = self._get_portal_at(px, py)
        if not (
            portal
            and portal is not self._skip_portal
            and portal.partner
            and not portal.partner.destroyed
            and self._can_enter_portal(portal)
        ):
            return
        partner = portal.partner
        player.set_position(partner.sprite.x, partner.sprite.y)
        self.last_used_portal = portal
        self._skip_portal = partner
        self.last_dx = 0
        self.last_dy = 0
        if portal.portal_type == SPECIAL:
            self._do_special_swap(portal)

    def _check_budget(self):
        if self.moves_remaining <= 0:
            self._lose_life()

    def step(self):
        if self.lives <= 0:
            self.lose()
            self.complete_action()
            return
        action = self.action.id
        if action == GameAction.RESET:
            self.level_reset()
            self.complete_action()
            return
        if action == GameAction.ACTION7:
            self._handle_undo()
            self.complete_action()
            return
        if action == GameAction.ACTION5:
            self._lose_life()
            self.complete_action()
            return
        delta = self._action_to_delta(action)
        if not delta:
            self.complete_action()
            return
        self._undo_stack.append(self._snapshot())
        player = self.current_level.get_sprites_by_name("player")[0]
        self._move_player(player, delta[0], delta[1])
        px, py = player.x, player.y
        self._process_pressure(px, py)
        self._process_relief(px, py)
        self._process_portals(player, px, py)
        if self._check_gate(player):
            self.next_level()
            self.complete_action()
            return
        self._check_budget()
        self.complete_action()

    def _load_level(self, idx, level):
        builders = [
            self._level_1,
            self._level_2,
            self._level_3,
            self._level_4,
            self._level_5,
            self._level_6,
        ]
        builders[idx](level)

    def _border(self):
        for c in range(self.GW):
            self.wall_set.add((c, 0))
        for r in range(self.GH):
            self.wall_set.add((0, r))
            self.wall_set.add((self.GW - 1, r))
        for c in range(self.GW):
            self.wall_set.add((c, 62))

    def _hwall(self, row, c1, c2):
        for c in range(c1, c2 + 1):
            self.wall_set.add((c, row))

    def _vwall(self, col, r1, r2):
        for r in range(r1, r2 + 1):
            self.wall_set.add((col, r))

    def _level_1(self, level):
        self.player_x, self.player_y = 4, 4
        self.gate_x, self.gate_y = 5, 54

        self.wall_set = set()
        self._border()

        self._vwall(32, 1, 61)
        self._vwall(20, 1, 20)
        self._hwall(20, 1, 31)
        self._hwall(20, 33, 62)
        self._hwall(32, 33, 62)
        self._hwall(44, 1, 31)
        self._hwall(44, 33, 62)
        self._vwall(48, 44, 61)

        self._hwall(8, 6, 12)
        self._vwall(12, 8, 14)
        self._hwall(14, 2, 6)

        self._hwall(8, 24, 28)
        self._vwall(26, 12, 16)

        self._hwall(8, 40, 50)
        self._vwall(50, 8, 14)
        self._hwall(14, 54, 60)
        self._vwall(38, 4, 10)

        self._hwall(26, 6, 14)
        self._vwall(14, 26, 30)
        self._vwall(24, 22, 28)

        self._hwall(38, 8, 18)
        self._vwall(18, 38, 42)
        self._vwall(26, 36, 40)

        self._hwall(52, 12, 22)
        self._vwall(22, 52, 58)
        self._vwall(8, 50, 56)

        self._hwall(28, 40, 52)
        self._vwall(44, 28, 36)
        self._hwall(36, 50, 58)
        self._vwall(54, 24, 32)
        self._hwall(38, 36, 42)

        self._hwall(52, 36, 44)
        self._vwall(40, 48, 56)

        self._hwall(52, 52, 58)
        self._vwall(56, 48, 56)

        self._make_portal_pair(level, 16, 14, 36, 22, 10, REGULAR)
        self._make_portal_pair(level, 58, 26, 22, 10, 6, REGULAR)
        self._make_portal_pair(
            level, 24, 4, 34, 54, 9, REGULAR, orient_a=HORIZ, orient_b=HORIZ
        )
        self._make_portal_pair(
            level, 42, 46, 4, 34, 8, SPECIAL, orient_a=HORIZ, orient_b=HORIZ
        )
        self._make_portal_pair(level, 28, 28, 26, 48, 15, REGULAR)

        self._add_pressure(level, 38, 24)
        self._add_pressure(level, 24, 12)
        self._add_pressure(level, 36, 56, HORIZ)
        self._add_pressure(level, 10, 28, HORIZ)
        self._add_pressure(level, 4, 52)

        self._add_relief(level, 40, 12)
        self._add_relief(level, 48, 34)

    def _level_2(self, level):
        self.player_x, self.player_y = 4, 4
        self.gate_x, self.gate_y = 58, 56

        self.wall_set = set()
        self._border()

        self._vwall(31, 0, 62)
        self._vwall(32, 0, 62)
        self._hwall(31, 0, 63)
        self._hwall(32, 0, 63)

        self._hwall(10, 4, 14)
        self._vwall(14, 10, 18)
        self._hwall(20, 1, 10)
        self._vwall(8, 20, 26)
        self._vwall(20, 6, 14)
        self._hwall(14, 20, 28)
        self._hwall(16, 4, 8)
        self._vwall(24, 8, 14)

        self._hwall(10, 40, 50)
        self._vwall(50, 10, 18)
        self._hwall(20, 54, 62)
        self._vwall(44, 16, 24)
        self._vwall(38, 6, 14)
        self._hwall(14, 34, 44)
        self._hwall(16, 54, 60)
        self._vwall(56, 4, 12)

        self._hwall(42, 4, 14)
        self._vwall(14, 42, 50)
        self._hwall(52, 1, 10)
        self._vwall(8, 52, 58)
        self._vwall(20, 38, 46)
        self._hwall(46, 20, 28)
        self._hwall(48, 4, 12)
        self._vwall(24, 46, 54)

        self._hwall(42, 40, 50)
        self._vwall(50, 42, 50)
        self._hwall(52, 54, 62)
        self._vwall(44, 48, 56)
        self._vwall(38, 38, 46)
        self._hwall(46, 34, 44)
        self._hwall(48, 52, 58)
        self._vwall(56, 38, 46)

        self._make_portal_pair(level, 16, 10, 46, 8, 9, REGULAR)
        self._make_portal_pair(level, 56, 18, 12, 38, 6, REGULAR)
        self._make_portal_pair(
            level, 22, 6, 36, 4, 12, SPECIAL, orient_a=HORIZ, orient_b=HORIZ
        )
        self._make_portal_pair(
            level, 52, 24, 4, 44, 10, REGULAR, orient_a=HORIZ, orient_b=HORIZ
        )
        self._make_portal_pair(level, 4, 54, 48, 38, 8, SPECIAL)

        self._add_pressure(level, 14, 12)
        self._add_pressure(level, 44, 10)
        self._add_pressure(level, 12, 40)
        self._add_pressure(level, 38, 8, HORIZ)
        self._add_pressure(level, 6, 46, HORIZ)

        self._add_relief(level, 10, 8)
        self._add_relief(level, 52, 48, HORIZ)

    def _level_3(self, level):
        self.player_x, self.player_y = 4, 4
        self.gate_x, self.gate_y = 58, 56

        self.wall_set = set()
        self._border()

        self._vwall(21, 0, 62)
        self._vwall(22, 0, 62)
        self._vwall(41, 0, 62)
        self._vwall(42, 0, 62)

        self._hwall(10, 4, 14)
        self._vwall(10, 10, 18)
        self._hwall(22, 1, 8)
        self._vwall(14, 18, 26)
        self._hwall(32, 4, 12)
        self._vwall(16, 28, 36)
        self._hwall(42, 1, 10)
        self._vwall(8, 42, 50)
        self._hwall(52, 6, 16)
        self._vwall(18, 48, 56)

        self._hwall(10, 28, 36)
        self._vwall(36, 10, 18)
        self._hwall(22, 24, 32)
        self._vwall(32, 18, 26)
        self._vwall(30, 26, 36)
        self._hwall(36, 26, 38)
        self._hwall(42, 28, 38)
        self._vwall(28, 42, 50)
        self._hwall(52, 32, 38)
        self._vwall(38, 48, 56)

        self._hwall(10, 48, 58)
        self._vwall(54, 10, 18)
        self._hwall(22, 44, 52)
        self._vwall(48, 22, 30)
        self._hwall(32, 52, 60)
        self._vwall(58, 28, 36)
        self._hwall(42, 44, 54)
        self._vwall(54, 42, 50)
        self._hwall(52, 48, 58)
        self._vwall(46, 50, 58)

        self._make_portal_pair(level, 8, 6, 28, 10, 9, REGULAR)
        self._make_portal_pair(level, 34, 20, 12, 34, 6, REGULAR)
        self._make_portal_pair(
            level, 4, 24, 26, 6, 10, REGULAR, orient_a=HORIZ, orient_b=HORIZ
        )
        self._make_portal_pair(
            level, 24, 44, 6, 56, 15, REGULAR, orient_a=HORIZ, orient_b=HORIZ
        )
        self._make_portal_pair(
            level, 16, 58, 46, 8, 12, SPECIAL, orient_a=HORIZ, orient_b=HORIZ
        )
        self._make_portal_pair(level, 52, 12, 32, 54, 8, SPECIAL)

        self._add_pressure(level, 26, 12)
        self._add_pressure(level, 12, 36)
        self._add_pressure(level, 28, 10, HORIZ)
        self._add_pressure(level, 8, 54, HORIZ)
        self._add_pressure(level, 48, 12, HORIZ)
        self._add_pressure(level, 30, 56, HORIZ)

        self._add_relief(level, 4, 28)
        self._add_relief(level, 36, 58, HORIZ)

    def _level_4(self, level):
        self.player_x, self.player_y = 4, 4
        self.gate_x, self.gate_y = 58, 56

        self.wall_set = set()
        self._border()

        self._vwall(21, 0, 62)
        self._vwall(22, 0, 62)

        self._vwall(41, 0, 49)
        self._vwall(42, 0, 49)
        self._vwall(41, 52, 62)
        self._vwall(42, 52, 62)

        self._vwall(41, 0, 30)
        self._vwall(42, 0, 30)

        self._hwall(31, 0, 63)
        self._hwall(32, 0, 63)

        self._hwall(10, 4, 12)
        self._vwall(12, 10, 18)
        self._vwall(16, 4, 10)
        self._hwall(22, 6, 14)
        self._hwall(16, 10, 16)
        self._vwall(8, 16, 24)

        self._hwall(10, 28, 36)
        self._vwall(36, 10, 18)
        self._hwall(22, 26, 32)
        self._vwall(28, 20, 28)
        self._vwall(32, 6, 14)
        self._hwall(14, 26, 32)

        self._hwall(10, 48, 56)
        self._vwall(56, 10, 18)
        self._vwall(48, 4, 12)
        self._hwall(22, 50, 58)
        self._hwall(16, 48, 54)
        self._vwall(52, 16, 24)

        self._hwall(42, 4, 12)
        self._vwall(12, 42, 50)
        self._hwall(52, 6, 16)
        self._vwall(8, 52, 58)
        self._vwall(16, 38, 46)
        self._hwall(46, 10, 18)

        self._hwall(42, 28, 36)
        self._vwall(36, 42, 50)
        self._hwall(52, 26, 34)
        self._vwall(28, 50, 58)
        self._hwall(46, 32, 38)
        self._vwall(32, 46, 54)

        self._hwall(42, 48, 56)
        self._vwall(56, 42, 50)
        self._hwall(52, 50, 58)
        self._vwall(48, 50, 58)
        self._vwall(52, 38, 46)
        self._hwall(46, 48, 54)

        self._hwall(36, 44, 50)
        self._vwall(50, 36, 40)
        self._hwall(40, 56, 60)
        self._vwall(60, 34, 40)

        self._vwall(46, 33, 36)
        self._hwall(34, 52, 58)

        self._make_portal_pair(level, 12, 10, 30, 6, 9, REGULAR)
        self._make_portal_pair(level, 34, 18, 6, 38, 6, REGULAR)
        self._make_portal_pair(
            level, 16, 46, 50, 8, 10, REGULAR, orient_a=HORIZ, orient_b=HORIZ
        )
        self._make_portal_pair(
            level, 46, 20, 28, 54, 15, REGULAR, orient_a=HORIZ, orient_b=HORIZ
        )
        self._make_portal_pair(
            level, 32, 38, 8, 8, 12, SPECIAL, orient_a=HORIZ, orient_b=HORIZ
        )
        self._make_portal_pair(
            level, 4, 20, 28, 10, 8, SPECIAL, orient_a=HORIZ, orient_b=HORIZ
        )
        self._make_portal_pair(level, 54, 24, 12, 52, 13, SPECIAL)
        self._make_portal_pair(level, 58, 52, 34, 56, 14, REGULAR)

        self._add_pressure(level, 28, 8)
        self._add_pressure(level, 8, 40)
        self._add_pressure(level, 48, 10, HORIZ)
        self._add_pressure(level, 30, 52, HORIZ)
        self._add_pressure(level, 10, 10, HORIZ)
        self._add_pressure(level, 30, 14, HORIZ)
        self._add_pressure(level, 12, 54)
        self._add_pressure(level, 56, 54)

        self._add_relief(level, 6, 24)
        self._add_relief(level, 36, 20, HORIZ)
        self._add_relief(level, 18, 56)

    def _level_5(self, level):
        self.player_x, self.player_y = 4, 4
        self.gate_x, self.gate_y = 58, 56

        self.wall_set = set()
        self._border()

        self._vwall(15, 0, 62)
        self._vwall(16, 0, 62)
        self._vwall(31, 0, 62)
        self._vwall(32, 0, 62)
        self._vwall(47, 0, 62)
        self._vwall(48, 0, 62)
        self._hwall(31, 0, 63)
        self._hwall(32, 0, 63)

        self._hwall(10, 4, 8)
        self._vwall(8, 10, 16)
        self._hwall(16, 2, 6)
        self._vwall(12, 20, 26)
        self._hwall(10, 20, 24)
        self._vwall(24, 10, 16)
        self._hwall(18, 18, 22)
        self._vwall(28, 20, 26)
        self._hwall(10, 36, 40)
        self._vwall(40, 10, 16)
        self._hwall(18, 34, 38)
        self._vwall(44, 20, 26)
        self._hwall(10, 52, 56)
        self._vwall(56, 10, 16)
        self._hwall(18, 50, 54)
        self._vwall(60, 20, 26)

        self._hwall(42, 4, 8)
        self._vwall(8, 42, 48)
        self._hwall(52, 2, 6)
        self._vwall(12, 52, 58)
        self._hwall(42, 20, 24)
        self._vwall(24, 42, 48)
        self._hwall(52, 18, 22)
        self._vwall(28, 52, 58)
        self._hwall(42, 36, 40)
        self._vwall(40, 42, 48)
        self._hwall(52, 34, 38)
        self._vwall(44, 52, 58)
        self._hwall(42, 52, 56)
        self._vwall(56, 42, 48)
        self._hwall(52, 50, 54)
        self._vwall(60, 52, 58)

        self._make_portal_pair(level, 4, 4, 36, 56, 9, REGULAR)
        self._make_portal_pair(level, 42, 36, 52, 4, 6, REGULAR)
        self._make_portal_pair(
            level, 54, 22, 20, 56, 10, REGULAR, orient_a=HORIZ, orient_b=HORIZ
        )
        self._make_portal_pair(
            level, 26, 36, 4, 56, 15, REGULAR, orient_a=HORIZ, orient_b=HORIZ
        )
        self._make_portal_pair(
            level, 10, 36, 52, 56, 12, SPECIAL, orient_a=HORIZ, orient_b=HORIZ
        )
        self._make_portal_pair(
            level, 2, 22, 42, 4, 8, SPECIAL, orient_a=HORIZ, orient_b=HORIZ
        )
        self._make_portal_pair(
            level, 36, 4, 20, 36, 13, SPECIAL, orient_a=HORIZ, orient_b=HORIZ
        )
        self._make_portal_pair(
            level, 26, 4, 58, 36, 7, SPECIAL, orient_a=HORIZ, orient_b=HORIZ
        )

        self._add_pressure(level, 36, 48)
        self._add_pressure(level, 54, 4)
        self._add_pressure(level, 20, 48, HORIZ)
        self._add_pressure(level, 4, 48, HORIZ)
        self._add_pressure(level, 52, 48, HORIZ)
        self._add_pressure(level, 42, 8)
        self._add_pressure(level, 20, 40, HORIZ)
        self._add_pressure(level, 58, 40, HORIZ)

        self._add_relief(level, 10, 14)
        self._add_relief(level, 42, 14)
        self._add_relief(level, 26, 46, HORIZ)

    def _level_6(self, level):
        self.player_x, self.player_y = 4, 4
        self.gate_x, self.gate_y = 58, 56

        self.wall_set = set()
        self._border()

        self._vwall(21, 0, 62)
        self._vwall(22, 0, 62)
        self._vwall(41, 0, 62)
        self._vwall(42, 0, 62)
        self._hwall(21, 0, 63)
        self._hwall(22, 0, 63)
        self._hwall(41, 0, 63)
        self._hwall(42, 0, 63)

        self._hwall(8, 6, 12)
        self._vwall(12, 8, 14)
        self._vwall(16, 4, 8)

        self._hwall(8, 28, 34)
        self._vwall(34, 8, 14)
        self._hwall(16, 26, 30)

        self._hwall(8, 50, 56)
        self._vwall(50, 8, 14)
        self._vwall(56, 4, 8)

        self._vwall(8, 28, 34)
        self._hwall(34, 4, 8)
        self._hwall(28, 12, 18)

        self._hwall(30, 28, 34)
        self._vwall(34, 30, 36)
        self._vwall(28, 34, 38)

        self._vwall(54, 28, 34)
        self._hwall(34, 54, 60)
        self._hwall(28, 46, 52)

        self._hwall(50, 4, 10)
        self._vwall(10, 50, 56)
        self._vwall(16, 44, 48)

        self._hwall(50, 28, 34)
        self._vwall(28, 50, 56)
        self._hwall(56, 32, 38)

        self._hwall(50, 50, 56)
        self._vwall(56, 50, 56)
        self._hwall(46, 46, 52)

        self._make_portal_pair(level, 4, 4, 58, 26, 9, REGULAR)
        self._make_portal_pair(level, 46, 36, 4, 46, 6, REGULAR)
        self._make_portal_pair(level, 14, 56, 46, 4, 10, REGULAR)
        self._make_portal_pair(level, 58, 14, 26, 26, 15, REGULAR)
        self._make_portal_pair(
            level, 36, 32, 26, 46, 3, REGULAR, orient_a=HORIZ, orient_b=HORIZ
        )
        self._make_portal_pair(
            level, 32, 54, 26, 4, 12, SPECIAL, orient_a=HORIZ, orient_b=HORIZ
        )
        self._make_portal_pair(
            level, 36, 14, 46, 56, 8, SPECIAL, orient_a=HORIZ, orient_b=HORIZ
        )
        self._make_portal_pair(level, 14, 24, 58, 46, 13, SPECIAL)
        self._make_portal_pair(
            level, 4, 16, 36, 46, 7, SPECIAL, orient_a=HORIZ, orient_b=HORIZ
        )
        self._make_portal_pair(
            level, 58, 4, 4, 56, 4, SPECIAL, orient_a=HORIZ, orient_b=HORIZ
        )

        self._add_pressure(level, 56, 28)
        self._add_pressure(level, 6, 44)
        self._add_pressure(level, 48, 8)
        self._add_pressure(level, 26, 28, HORIZ)
        self._add_pressure(level, 30, 46)
        self._add_pressure(level, 26, 10, HORIZ)
        self._add_pressure(level, 48, 54, HORIZ)
        self._add_pressure(level, 58, 48)
        self._add_pressure(level, 36, 48, HORIZ)
        self._add_pressure(level, 4, 54, HORIZ)

        self._add_relief(level, 6, 6, HORIZ)
        self._add_relief(level, 58, 16)
        self._add_relief(level, 14, 32, HORIZ)
        self._add_relief(level, 46, 48)


_ARC_PALETTE = [
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


def _encode_png(rgb):
    h, w = rgb.shape[0], rgb.shape[1]
    raw = b""
    for y in range(h):
        raw += b"\x00" + rgb[y].tobytes()
    compressed = zlib.compress(raw)

    def _chunk(ctype, data):
        c = ctype + data
        return (
            struct.pack(">I", len(data))
            + c
            + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", compressed)
        + _chunk(b"IEND", b"")
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


class PuzzleEnvironment:
    _ACTION_MAP: dict = {
        "up": GameAction.ACTION1,
        "down": GameAction.ACTION2,
        "left": GameAction.ACTION3,
        "right": GameAction.ACTION4,
        "select": GameAction.ACTION5,
        "undo": GameAction.ACTION7,
        "reset": GameAction.RESET,
    }

    _VALID_ACTIONS = ["reset", "up", "down", "left", "right", "select", "undo"]

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self._engine: Optional[Pm14] = Pm14(seed=seed)
        self._total_turns = 0
        self._consecutive_resets = 0

    @property
    def _eng(self) -> Pm14:
        if self._engine is None:
            raise RuntimeError("Environment is closed")
        return self._engine

    def reset(self) -> GameState:
        eng = self._eng
        if self._check_win() or self._consecutive_resets >= 1:
            self._engine = Pm14(seed=self._seed)
            self._consecutive_resets = 0
        else:
            self._consecutive_resets += 1
        self._eng.perform_action(ActionInput(id=GameAction.RESET))
        self._total_turns = 0
        return self._build_state()

    def step(self, action: str) -> StepResult:
        if self._engine is None:
            raise RuntimeError("Environment is closed")

        if action not in self._ACTION_MAP:
            raise ValueError(f"Invalid action: {action}")

        game_action = self._ACTION_MAP[action]

        if game_action == GameAction.RESET:
            state = self.reset()
            return StepResult(state=state, reward=0.0, done=False)

        self._consecutive_resets = 0

        prev_idx = self._eng.level_index
        self._eng.perform_action(ActionInput(id=game_action))
        self._total_turns += 1

        done = self._check_done()
        total = len(self._eng._levels)
        if self._check_win() or self._eng.level_index != prev_idx:
            reward = 1.0 / total
        else:
            reward = 0.0
        state = self._build_state()

        return StepResult(state=state, reward=reward, done=done)

    def get_actions(self) -> List[str]:
        if self._engine is None:
            return []
        if self._check_done():
            return ["reset"]
        return list(self._VALID_ACTIONS)

    def is_done(self) -> bool:
        if self._engine is None:
            return True
        return self._check_done()

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if self._engine is None:
            raise RuntimeError("Environment is closed")
        if mode != "rgb_array":
            raise ValueError(f"Unsupported render mode: {mode}")
        index_grid = self._eng.camera.render(self._eng.current_level.get_sprites())
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        for idx, color in enumerate(_ARC_PALETTE):
            mask = index_grid == idx
            rgb[mask] = color
        return rgb

    def close(self) -> None:
        self._engine = None

    def _check_done(self) -> bool:
        return self._eng._state in (_EngineState.GAME_OVER, _EngineState.WIN)

    def _check_win(self) -> bool:
        return self._eng._state == _EngineState.WIN

    def _build_state(self) -> GameState:
        text = self._build_text_observation()
        image = self._render_image_bytes()
        valid = None if self._check_done() else list(self._VALID_ACTIONS)
        meta = {
            "total_levels": len(self._eng._levels),
            "level": self._eng.level_index + 1,
            "moves_remaining": self._eng.moves_remaining,
            "current_budget": self._eng.current_budget,
            "lives": self._eng.lives,
        }
        return GameState(
            text_observation=text,
            image_observation=image,
            valid_actions=valid,
            turn=self._total_turns,
            metadata=meta,
        )

    def _build_text_observation(self) -> str:
        eng = self._eng
        header = (
            f"L{eng.level_index + 1}/{len(eng._levels)}"
            f" M{eng.moves_remaining}/{eng.current_budget}"
            f" HP{eng.lives}"
        )
        index_grid = eng.camera.render(eng.current_level.get_sprites())
        rows = []
        for r in range(63):
            row_chars = "".join(format(int(index_grid[r, c]), "x") for c in range(64))
            rows.append(row_chars)
        return header + "\n" + "\n".join(rows)

    def _render_image_bytes(self) -> Optional[bytes]:
        rgb = self.render()
        return _encode_png(rgb)


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
