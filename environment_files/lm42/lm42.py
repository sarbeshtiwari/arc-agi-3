import random
import struct
import zlib
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

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

BACKGROUND_COLOR = 0
PADDING_COLOR = 0


CELL = 6
SPRITE_SCALE = 2

BEAM_RED = 8
BEAM_BLUE = 9
BEAM_DEFAULT = 11


sprites = {
    "floor": Sprite(
        pixels=[
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
        name="floor",
        visible=True,
        collidable=False,
        layer=0,
        tags=["floor"],
    ),
    "wall": Sprite(
        pixels=[
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
        ],
        name="wall",
        visible=True,
        collidable=True,
        layer=1,
        tags=["wall"],
    ),
    "block": Sprite(
        pixels=[
            [4, 4, 4],
            [4, 3, 4],
            [4, 4, 4],
        ],
        name="block",
        visible=True,
        collidable=True,
        layer=1,
        tags=["block"],
    ),
    "mwall": Sprite(
        pixels=[
            [14, 14, 14],
            [14, 11, 14],
            [14, 14, 14],
        ],
        name="mwall",
        visible=True,
        collidable=True,
        layer=1,
        tags=["mwall"],
    ),
    "laser_red": Sprite(
        pixels=[
            [8, 8, -1],
            [8, 0, 11],
            [8, 8, -1],
        ],
        name="laser_red",
        visible=True,
        collidable=True,
        layer=3,
        tags=["laser"],
    ),
    "laser_blue": Sprite(
        pixels=[
            [9, 9, -1],
            [9, 0, 10],
            [9, 9, -1],
        ],
        name="laser_blue",
        visible=True,
        collidable=True,
        layer=3,
        tags=["laser"],
    ),
    "laser_default": Sprite(
        pixels=[
            [12, 11, -1],
            [11, 0, 0],
            [12, 11, -1],
        ],
        name="laser_default",
        visible=True,
        collidable=True,
        layer=3,
        tags=["laser"],
    ),
    "gem": Sprite(
        pixels=[
            [-1, 10, -1],
            [10, 0, 10],
            [-1, 10, -1],
        ],
        name="gem",
        visible=True,
        collidable=False,
        layer=3,
        tags=["gem"],
    ),
    "gem_red": Sprite(
        pixels=[
            [-1, 8, -1],
            [8, 12, 8],
            [-1, 8, -1],
        ],
        name="gem_red",
        visible=True,
        collidable=False,
        layer=3,
        tags=["gem", "gem_red"],
    ),
    "gem_blue": Sprite(
        pixels=[
            [-1, 9, -1],
            [9, 10, 9],
            [-1, 9, -1],
        ],
        name="gem_blue",
        visible=True,
        collidable=False,
        layer=3,
        tags=["gem", "gem_blue"],
    ),
    "decoy": Sprite(
        pixels=[
            [15, -1, 15],
            [-1, 6, -1],
            [15, -1, 15],
        ],
        name="decoy",
        visible=True,
        collidable=False,
        layer=3,
        tags=["decoy"],
    ),
    "bomb": Sprite(
        pixels=[
            [-1, 13, -1],
            [13, 8, 13],
            [-1, 13, -1],
        ],
        name="bomb",
        visible=True,
        collidable=False,
        layer=3,
        tags=["bomb"],
    ),
    "mirror_bs": Sprite(
        pixels=[
            [8, -1, -1],
            [-1, 8, -1],
            [-1, -1, 8],
        ],
        name="mirror_bs",
        visible=True,
        collidable=False,
        layer=4,
        tags=["mirror", "bs"],
    ),
    "mirror_fs": Sprite(
        pixels=[
            [-1, -1, 9],
            [-1, 9, -1],
            [9, -1, -1],
        ],
        name="mirror_fs",
        visible=True,
        collidable=False,
        layer=4,
        tags=["mirror", "fs"],
    ),
    "mirror_prism": Sprite(
        pixels=[
            [-1, 14, -1],
            [14, 14, 14],
            [-1, 14, -1],
        ],
        name="mirror_prism",
        visible=True,
        collidable=False,
        layer=4,
        tags=["mirror", "prism"],
    ),
    "fixed_bs": Sprite(
        pixels=[
            [13, -1, -1],
            [-1, 13, -1],
            [-1, -1, 13],
        ],
        name="fixed_bs",
        visible=True,
        collidable=False,
        layer=4,
        tags=["mirror", "bs", "fixed"],
    ),
    "fixed_fs": Sprite(
        pixels=[
            [-1, -1, 13],
            [-1, 13, -1],
            [13, -1, -1],
        ],
        name="fixed_fs",
        visible=True,
        collidable=False,
        layer=4,
        tags=["mirror", "fs", "fixed"],
    ),
    "fixed_prism": Sprite(
        pixels=[
            [-1, 3, -1],
            [3, 2, 3],
            [-1, 3, -1],
        ],
        name="fixed_prism",
        visible=True,
        collidable=False,
        layer=4,
        tags=["mirror", "prism", "fixed"],
    ),
    "beam_px": Sprite(
        pixels=[[0]],
        name="beam_px",
        visible=True,
        collidable=False,
        layer=2,
        tags=["beam"],
    ),
    "beam_glow": Sprite(
        pixels=[[3]],
        name="beam_glow",
        visible=True,
        collidable=False,
        layer=2,
        tags=["beam"],
    ),
    "sparkle": Sprite(
        pixels=[
            [10, -1, 10],
            [-1, -1, -1],
            [10, -1, 10],
        ],
        name="sparkle",
        visible=True,
        collidable=False,
        layer=5,
        tags=["sparkle"],
    ),
    "split_fx": Sprite(
        pixels=[
            [14, -1, 14],
            [-1, -1, -1],
            [14, -1, 14],
        ],
        name="split_fx",
        visible=True,
        collidable=False,
        layer=5,
        tags=["split_fx"],
    ),
    "explosion": Sprite(
        pixels=[
            [8, 11, 8],
            [11, 0, 11],
            [8, 11, 8],
        ],
        name="explosion",
        visible=True,
        collidable=False,
        layer=5,
        tags=["explosion"],
    ),
    "gem_collected": Sprite(
        pixels=[
            [-1, 14, -1],
            [14, 0, 14],
            [-1, 14, -1],
        ],
        name="gem_collected",
        visible=True,
        collidable=False,
        layer=3,
        tags=["gem_collected"],
    ),
    "portal_red_a": Sprite(
        pixels=[
            [-1, 8, -1],
            [8, -1, 8],
            [-1, 8, -1],
        ],
        name="portal_red_a",
        visible=True,
        collidable=False,
        layer=3,
        tags=["portal", "portal_red_a"],
    ),
    "portal_red_b": Sprite(
        pixels=[
            [-1, 13, -1],
            [13, 8, 13],
            [-1, 13, -1],
        ],
        name="portal_red_b",
        visible=True,
        collidable=False,
        layer=3,
        tags=["portal", "portal_red_b"],
    ),
    "portal_blue_a": Sprite(
        pixels=[
            [-1, 9, -1],
            [9, -1, 9],
            [-1, 9, -1],
        ],
        name="portal_blue_a",
        visible=True,
        collidable=False,
        layer=3,
        tags=["portal", "portal_blue_a"],
    ),
    "portal_blue_b": Sprite(
        pixels=[
            [-1, 10, -1],
            [10, 9, 10],
            [-1, 10, -1],
        ],
        name="portal_blue_b",
        visible=True,
        collidable=False,
        layer=3,
        tags=["portal", "portal_blue_b"],
    ),
    "portal_a": Sprite(
        pixels=[
            [-1, 7, -1],
            [7, -1, 7],
            [-1, 7, -1],
        ],
        name="portal_a",
        visible=True,
        collidable=False,
        layer=3,
        tags=["portal", "portal_a"],
    ),
    "portal_b": Sprite(
        pixels=[
            [-1, 6, -1],
            [6, 7, 6],
            [-1, 6, -1],
        ],
        name="portal_b",
        visible=True,
        collidable=False,
        layer=3,
        tags=["portal", "portal_b"],
    ),
    "portal_c": Sprite(
        pixels=[
            [-1, 6, -1],
            [6, -1, 6],
            [-1, 6, -1],
        ],
        name="portal_c",
        visible=True,
        collidable=False,
        layer=3,
        tags=["portal", "portal_c"],
    ),
    "portal_d": Sprite(
        pixels=[
            [-1, 15, -1],
            [15, 6, 15],
            [-1, 15, -1],
        ],
        name="portal_d",
        visible=True,
        collidable=False,
        layer=3,
        tags=["portal", "portal_d"],
    ),
    "vortex": Sprite(
        pixels=[
            [6, -1, 6],
            [-1, 0, -1],
            [6, -1, 6],
        ],
        name="vortex",
        visible=True,
        collidable=False,
        layer=5,
        tags=["vortex"],
    ),
    "shift_fx": Sprite(
        pixels=[
            [-1, 14, -1],
            [14, 0, 14],
            [-1, 14, -1],
        ],
        name="shift_fx",
        visible=True,
        collidable=False,
        layer=5,
        tags=["shift_fx"],
    ),
}


class MirrorHud(RenderableUserDisplay):
    def __init__(self, game: "Lm42"):
        self.game = game

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        g = self.game
        if not hasattr(g, "mirrors_bs_total"):
            return frame

        if hasattr(g, "cursor_col") and hasattr(g, "cursor_row"):
            cc, cr = g.cursor_col, g.cursor_row
            if 0 <= cc < g.gw and 0 <= cr < g.gh:
                cx = g.ox + cc * g.cell
                cy = g.oy + cr * g.cell
                cw = g.cell
                ch = g.cell
                cur_color = 14
                for i in range(cw):
                    if 0 <= cy < 64 and 0 <= cx + i < 64:
                        frame[cy, cx + i] = cur_color
                    if 0 <= cy + ch - 1 < 64 and 0 <= cx + i < 64:
                        frame[cy + ch - 1, cx + i] = cur_color
                for i in range(ch):
                    if 0 <= cy + i < 64 and 0 <= cx < 64:
                        frame[cy + i, cx] = cur_color
                    if 0 <= cy + i < 64 and 0 <= cx + cw - 1 < 64:
                        frame[cy + i, cx + cw - 1] = cur_color

        if hasattr(g, "gem_coords") and hasattr(g, "_last_hit_gems"):
            total_gems = len(g.gem_coords)
            hit_count = len(g._last_hit_gems)
            for i in range(total_gems):
                px = 1 + i * 4
                if px + 2 < 64:
                    c = 14 if i < hit_count else 3
                    frame[0, px + 1] = c
                    frame[1, px] = c
                    frame[1, px + 2] = c

        if hasattr(g, "total_levels") and hasattr(g, "current_level_index"):
            for i in range(g.total_levels):
                px = 53 + i * 2
                if px + 1 < 64:
                    if i < g.current_level_index:
                        c = 14
                    elif i == g.current_level_index:
                        c = 5
                    else:
                        c = 2
                    frame[0, px] = c
                    frame[0, px + 1] = c
                    frame[1, px] = c
                    frame[1, px + 1] = c

        if hasattr(g, "lives") and hasattr(g, "max_lives"):
            for i in range(g.max_lives):
                hx = 1 + i * 3
                c = 8 if i < g.lives else 2
                frame[61, hx] = c
                frame[61, hx + 1] = c
                frame[62, hx] = c
                frame[62, hx + 1] = c

        if hasattr(g, "current_mirror_type"):
            box_w = 8
            rx = 64 - box_w
            ry = 4

            types = [
                (0, g.mirrors_bs_remaining, g.mirrors_bs_total, 8),
                (1, g.mirrors_fs_remaining, g.mirrors_fs_total, 9),
                (2, g.mirrors_prism_remaining, g.mirrors_prism_total, 14),
            ]

            for idx, (mtype, remaining, total, color) in enumerate(types):
                if total == 0:
                    continue
                bx = rx
                by = ry + idx * 12
                bw = box_w
                bh = box_w
                is_sel = g.current_mirror_type == mtype
                bc = 5 if is_sel else 2

                for i in range(bw):
                    if 0 <= by < 64 and 0 <= bx + i < 64:
                        frame[by, bx + i] = bc
                    if 0 <= by + bh - 1 < 64 and 0 <= bx + i < 64:
                        frame[by + bh - 1, bx + i] = bc
                for i in range(bh):
                    if 0 <= by + i < 64 and 0 <= bx < 64:
                        frame[by + i, bx] = bc
                    if 0 <= by + i < 64 and 0 <= bx + bw - 1 < 64:
                        frame[by + i, bx + bw - 1] = bc

                for fy in range(by + 1, by + bh - 1):
                    for fx in range(bx + 1, bx + bw - 1):
                        if 0 <= fy < 64 and 0 <= fx < 64:
                            frame[fy, fx] = 0

                sc = color if is_sel else 3
                if mtype == 0:
                    for d in range(1, bh - 1):
                        py = by + d
                        px = bx + d
                        if 0 <= py < 64 and 0 <= px < 64:
                            frame[py, px] = sc
                elif mtype == 1:
                    for d in range(1, bh - 1):
                        py = by + d
                        px = bx + bw - 1 - d
                        if 0 <= py < 64 and 0 <= px < 64:
                            frame[py, px] = sc
                else:
                    mid = bh // 2
                    for i in range(1, bw - 1):
                        if 0 <= by + mid < 64 and 0 <= bx + i < 64:
                            frame[by + mid, bx + i] = sc
                    for i in range(1, bh - 1):
                        if 0 <= by + i < 64 and 0 <= bx + bw // 2 < 64:
                            frame[by + i, bx + bw // 2] = sc

                pip_y = by + bh + 1
                if pip_y < 64:
                    for si in range(total):
                        pip_x = bx + 1 + si
                        if pip_x < 64:
                            frame[pip_y, pip_x] = color if si < remaining else 2

        if hasattr(g, "max_moves") and g.max_moves > 0:
            remaining = max(0, g.max_moves - g.moves_used)
            ratio = remaining / g.max_moves
            cells_filled = int(ratio * 64)
            for col in range(64):
                frame[63, col] = 14 if col < cells_filled else 2

        return frame


DIR_MAP: Dict[str, Tuple[int, int]] = {
    "R": (1, 0),
    "L": (-1, 0),
    "D": (0, 1),
    "U": (0, -1),
}


def _reflect(
    dx: int, dy: int, mirror_type: str, beam_color: str
) -> List[Tuple[int, int, str]]:
    if mirror_type == "bs":
        return [(dy, dx, beam_color)]
    elif mirror_type == "fs":
        return [(-dy, -dx, beam_color)]
    elif mirror_type == "prism":
        if dx != 0:
            c1 = "red" if beam_color is None else beam_color
            c2 = "blue" if beam_color is None else beam_color
            return [(0, -1, c1), (0, 1, c2)]
        else:
            c1 = "red" if beam_color is None else beam_color
            c2 = "blue" if beam_color is None else beam_color
            return [(-1, 0, c1), (1, 0, c2)]
    return [(dx, dy, beam_color)]


def _beam_palette_color(beam_color: Optional[str]) -> int:
    if beam_color == "red":
        return BEAM_RED
    elif beam_color == "blue":
        return BEAM_BLUE
    return BEAM_DEFAULT


def _color_matches_portal(
    beam_color: Optional[str], portal_color: Optional[str]
) -> bool:
    if portal_color is None:
        return True
    return beam_color == portal_color


def _color_matches_gem(beam_color: Optional[str], gem_color: Optional[str]) -> bool:
    if gem_color is None:
        return True
    return beam_color == gem_color


LEVEL_DATA = [
    {
        "grid_w": 8,
        "grid_h": 8,
        "ox": 2,
        "oy": 2,
        "laser": (0, 3, "R", None),
        "gems": [(6, 1, "red"), (6, 5, "blue")],
        "decoys": [(7, 3)],
        "bombs": [],
        "walls": [(5, 3), (6, 3)],
        "fixed_mirrors": [],
        "portals": [],
        "moving_walls": [],
        "mwall_shift_every": 0,
        "mirrors_bs": 1,
        "mirrors_fs": 1,
        "mirrors_prism": 1,
        "name": "Level 1",
    },
    {
        "grid_w": 8,
        "grid_h": 8,
        "ox": 2,
        "oy": 2,
        "laser": (0, 0, "R", None),
        "gems": [(4, 0, None), (4, 7, None)],
        "decoys": [(7, 7)],
        "bombs": [
            (7, 0),
            (7, 4),
            (6, 7),
        ],
        "walls": [(5, 4), (6, 4), (5, 7)],
        "fixed_mirrors": [(5, 0, "bs")],
        "portals": [],
        "moving_walls": [],
        "mwall_shift_every": 0,
        "mirrors_bs": 1,
        "mirrors_fs": 2,
        "mirrors_prism": 0,
        "name": "Level 2",
    },
    {
        "grid_w": 8,
        "grid_h": 8,
        "ox": 2,
        "oy": 2,
        "laser": (0, 3, "R", None),
        "gems": [(7, 0, None), (7, 6, None)],
        "decoys": [(0, 7), (7, 7)],
        "bombs": [
            (3, 3),
            (4, 3),
            (5, 3),
            (6, 3),
            (7, 3),
            (2, 1),
            (4, 1),
            (6, 1),
            (2, 5),
            (4, 5),
            (6, 5),
        ],
        "walls": [],
        "fixed_mirrors": [],
        "portals": [],
        "moving_walls": [],
        "mwall_shift_every": 0,
        "mirrors_bs": 1,
        "mirrors_fs": 1,
        "mirrors_prism": 1,
        "name": "Level 3",
    },
    {
        "grid_w": 8,
        "grid_h": 8,
        "ox": 2,
        "oy": 2,
        "laser": (0, 0, "R", None),
        "gems": [(7, 5, None)],
        "decoys": [(0, 7), (7, 7)],
        "bombs": [
            (3, 0),
            (4, 0),
            (5, 0),
            (6, 0),
            (7, 0),
            (0, 1),
            (1, 1),
            (3, 1),
            (6, 1),
            (3, 2),
            (6, 2),
            (0, 3),
            (5, 3),
            (6, 3),
            (7, 3),
            (5, 4),
            (6, 4),
            (0, 5),
            (1, 5),
            (3, 5),
            (3, 6),
            (5, 6),
            (6, 6),
        ],
        "walls": [(2, 4)],
        "fixed_mirrors": [],
        "portals": [],
        "moving_walls": [],
        "mwall_shift_every": 0,
        "mirrors_bs": 4,
        "mirrors_fs": 0,
        "mirrors_prism": 0,
        "name": "Level 4",
    },
    {
        "grid_w": 8,
        "grid_h": 8,
        "ox": 2,
        "oy": 2,
        "laser": (0, 3, "R", None),
        "gems": [(7, 0, "red"), (7, 6, "blue")],
        "decoys": [(0, 7), (7, 7)],
        "bombs": [
            (0, 0),
            (3, 0),
            (4, 0),
            (5, 0),
            (3, 1),
            (4, 1),
            (6, 1),
            (7, 2),
            (3, 3),
            (4, 3),
            (6, 3),
            (7, 3),
            (7, 4),
            (3, 5),
            (4, 5),
            (6, 5),
            (0, 6),
            (3, 6),
            (4, 6),
            (5, 6),
        ],
        "walls": [(2, 1), (2, 5), (3, 2), (3, 4)],
        "fixed_mirrors": [],
        "portals": [
            {
                "a": (2, 0),
                "b": (6, 0),
                "exit_dir": "R",
                "beam_color": "red",
                "colors": ("portal_red_a", "portal_red_b"),
            },
            {
                "a": (2, 6),
                "b": (6, 6),
                "exit_dir": "R",
                "beam_color": "blue",
                "colors": ("portal_blue_a", "portal_blue_b"),
            },
        ],
        "moving_walls": [
            {"positions": [(5, 7), (5, 3)], "start": 0},
        ],
        "mwall_shift_every": 1,
        "mirrors_bs": 1,
        "mirrors_fs": 1,
        "mirrors_prism": 1,
        "name": "Level 5",
    },
]


def _build_level(ld):
    gw = ld["grid_w"]
    gh = ld["grid_h"]
    ox = ld["ox"]
    oy = ld["oy"]

    sp_list = []

    for c in range(-1, gw):
        for r in [-1, gh]:
            sp_list.append(
                sprites["wall"]
                .clone()
                .set_position(ox + c * CELL, oy + r * CELL)
                .set_scale(SPRITE_SCALE)
            )
    for r in range(gh):
        sp_list.append(
            sprites["wall"]
            .clone()
            .set_position(ox + (-1) * CELL, oy + r * CELL)
            .set_scale(SPRITE_SCALE)
        )

    for c in range(gw):
        for r in range(gh):
            sp_list.append(
                sprites["floor"]
                .clone()
                .set_position(ox + c * CELL, oy + r * CELL)
                .set_scale(SPRITE_SCALE)
            )

    for wc, wr in ld["walls"]:
        sp_list.append(
            sprites["block"]
            .clone()
            .set_position(ox + wc * CELL, oy + wr * CELL)
            .set_scale(SPRITE_SCALE)
        )

    for mw_def in ld.get("moving_walls", []):
        start_idx = mw_def.get("start", 0)
        sc, sr = mw_def["positions"][start_idx]
        sp_list.append(
            sprites["mwall"]
            .clone()
            .set_position(ox + sc * CELL, oy + sr * CELL)
            .set_scale(SPRITE_SCALE)
        )

    lc, lr, ldir, lcolor = ld["laser"]
    if lcolor == "red":
        laser_sname = "laser_red"
    elif lcolor == "blue":
        laser_sname = "laser_blue"
    else:
        laser_sname = "laser_default"
    laser_sprite = (
        sprites[laser_sname]
        .clone()
        .set_position(ox + lc * CELL, oy + lr * CELL)
        .set_scale(SPRITE_SCALE)
    )
    rot_map = {"R": 0, "D": 90, "L": 180, "U": 270}
    laser_sprite.set_rotation(rot_map.get(ldir, 0))
    sp_list.append(laser_sprite)

    for gem_entry in ld["gems"]:
        gc, gr, gcolor = gem_entry
        if gcolor == "red":
            gname = "gem_red"
        elif gcolor == "blue":
            gname = "gem_blue"
        else:
            gname = "gem"
        sp_list.append(
            sprites[gname]
            .clone()
            .set_position(ox + gc * CELL, oy + gr * CELL)
            .set_scale(SPRITE_SCALE)
        )

    for dc, dr in ld.get("decoys", []):
        sp_list.append(
            sprites["decoy"]
            .clone()
            .set_position(ox + dc * CELL, oy + dr * CELL)
            .set_scale(SPRITE_SCALE)
        )

    for bc, br in ld.get("bombs", []):
        sp_list.append(
            sprites["bomb"]
            .clone()
            .set_position(ox + bc * CELL, oy + br * CELL)
            .set_scale(SPRITE_SCALE)
        )

    for fc, fr, ft in ld.get("fixed_mirrors", []):
        if ft == "bs":
            sname = "fixed_bs"
        elif ft == "fs":
            sname = "fixed_fs"
        else:
            sname = "fixed_prism"
        sp_list.append(
            sprites[sname]
            .clone()
            .set_position(ox + fc * CELL, oy + fr * CELL)
            .set_scale(SPRITE_SCALE)
        )

    portal_data_for_level = []
    for pdef in ld.get("portals", []):
        ac, ar = pdef["a"]
        bc, br = pdef["b"]
        ca, cb = pdef["colors"]
        sp_list.append(
            sprites[ca]
            .clone()
            .set_position(ox + ac * CELL, oy + ar * CELL)
            .set_scale(SPRITE_SCALE)
        )
        sp_list.append(
            sprites[cb]
            .clone()
            .set_position(ox + bc * CELL, oy + br * CELL)
            .set_scale(SPRITE_SCALE)
        )
        portal_data_for_level.append(
            {
                "a": [ac, ar],
                "b": [bc, br],
                "exit_dir": pdef.get("exit_dir", None),
                "beam_color": pdef.get("beam_color", None),
            }
        )

    return Level(
        sprites=sp_list,
        grid_size=(64, 64),
        data={
            "grid_w": gw,
            "grid_h": gh,
            "cell": CELL,
            "ox": ox,
            "oy": oy,
            "laser_col": lc,
            "laser_row": lr,
            "laser_dir": ldir,
            "laser_color": ld["laser"][3],
            "num_mirrors_bs": ld.get("mirrors_bs", 0),
            "num_mirrors_fs": ld.get("mirrors_fs", 0),
            "num_mirrors_prism": ld.get("mirrors_prism", 0),
            "gem_data": ld["gems"],
            "decoy_coords": ld.get("decoys", []),
            "bomb_coords": ld.get("bombs", []),
            "portal_pairs": portal_data_for_level,
            "moving_walls_def": ld.get("moving_walls", []),
            "mwall_shift_every": ld.get("mwall_shift_every", 0),
        },
        name=ld["name"],
    )


class Lm42(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)
        self.hud = MirrorHud(self)
        self._levels = [_build_level(ld) for ld in LEVEL_DATA]
        self.total_levels = len(self._levels)
        self.current_level_index = 0

        self.mirrors_bs_total = 0
        self.mirrors_fs_total = 0
        self.mirrors_prism_total = 0
        self.mirrors_bs_remaining = 0
        self.mirrors_fs_remaining = 0
        self.mirrors_prism_remaining = 0
        self.current_mirror_type = 0

        self.max_lives = 3
        self.lives = 3

        self.moves_used = 0
        self.max_moves = 200

        self._game_won = False

        self.cursor_col = 0
        self.cursor_row = 0
        self._selection_mode = False
        self._prev_action_id = -1

        super().__init__(
            "lm42",
            self._levels,
            Camera(
                width=64,
                height=64,
                background=BACKGROUND_COLOR,
                letter_box=PADDING_COLOR,
                interfaces=[self.hud],
            ),
            available_actions=[0, 1, 2, 3, 4, 5, 7],
        )

    def on_set_level(self, level: Level) -> None:
        for i, lv in enumerate(self._levels):
            if lv is level:
                self.current_level_index = i
                break
        self.gw: int = self.current_level.get_data("grid_w")
        self.gh: int = self.current_level.get_data("grid_h")
        self._selection_mode = False
        self._prev_action_id = -1
        self.cell: int = self.current_level.get_data("cell")
        self.ox: int = self.current_level.get_data("ox")
        self.oy: int = self.current_level.get_data("oy")
        self.laser_col: int = self.current_level.get_data("laser_col")
        self.laser_row: int = self.current_level.get_data("laser_row")
        self.laser_dir: str = self.current_level.get_data("laser_dir")
        self.laser_color: Optional[str] = self.current_level.get_data("laser_color")
        self.mirrors_bs_total = self.current_level.get_data("num_mirrors_bs")
        self.mirrors_fs_total = self.current_level.get_data("num_mirrors_fs")
        self.mirrors_prism_total = self.current_level.get_data("num_mirrors_prism")
        self.mirrors_bs_remaining = self.mirrors_bs_total
        self.mirrors_fs_remaining = self.mirrors_fs_total
        self.mirrors_prism_remaining = self.mirrors_prism_total

        self.lives = self.max_lives

        self.moves_used = 0

        raw_gems = self.current_level.get_data("gem_data")
        self.gem_data: List[Tuple[int, int, Optional[str]]] = [
            (g[0], g[1], g[2]) for g in raw_gems
        ]
        self.gem_coords: List[Tuple[int, int]] = [(g[0], g[1]) for g in self.gem_data]
        self._gem_color_map: Dict[Tuple[int, int], Optional[str]] = {
            (g[0], g[1]): g[2] for g in self.gem_data
        }

        self.decoy_coords: list = list(self.current_level.get_data("decoy_coords"))
        self.bomb_coords: list = list(self.current_level.get_data("bomb_coords"))

        raw_portals = self.current_level.get_data("portal_pairs")
        self.portal_pairs: List[dict] = list(raw_portals) if raw_portals else []

        self._portal_map: Dict[Tuple[int, int], Tuple[int, str]] = {}
        for pi, pp in enumerate(self.portal_pairs):
            ac, ar = pp["a"][0], pp["a"][1]
            bc, br = pp["b"][0], pp["b"][1]
            self._portal_map[(ac, ar)] = (pi, "a")
            self._portal_map[(bc, br)] = (pi, "b")

        mw_defs = self.current_level.get_data("moving_walls_def")
        self._mwall_defs: list = list(mw_defs) if mw_defs else []
        self._mwall_indices: List[int] = [
            mwd.get("start", 0) for mwd in self._mwall_defs
        ]
        self.mwall_shift_every: int = self.current_level.get_data("mwall_shift_every")
        self.mwall_place_counter: int = 0

        self._mwall_set: set = set()
        self._mwall_sprites: List[Sprite] = []
        for s in self.current_level.get_sprites_by_tag("mwall"):
            gc = self._px_to_grid(s.x, s.y)
            if gc:
                self._mwall_set.add(gc)
                self._mwall_sprites.append(s)

        self._gem_set: set = set(self.gem_coords)
        self._decoy_set: set = set((dc, dr) for dc, dr in self.decoy_coords)
        self._bomb_set: set = set((bc, br) for bc, br in self.bomb_coords)

        if self.mirrors_bs_total > 0:
            self.current_mirror_type = 0
        elif self.mirrors_fs_total > 0:
            self.current_mirror_type = 1
        elif self.mirrors_prism_total > 0:
            self.current_mirror_type = 2
        else:
            self.current_mirror_type = 0

        self.placed_mirrors: List[Sprite] = []
        self.placed_positions: List[Tuple[int, int]] = []
        self.placed_types: List[str] = []

        self.beam_sprites: List[Sprite] = []
        self.effect_sprites: List[Sprite] = []

        self._last_hit_gems: List[Tuple[int, int]] = []
        self._last_hit_decoys: List[Tuple[int, int]] = []
        self._last_hit_bombs: List[Tuple[int, int]] = []

        self._history: List[Dict] = []
        self._known_hit_bombs: set = set()

        self._block_set: set = set()
        for s in self.current_level.get_sprites_by_tag("block"):
            gc = self._px_to_grid(s.x, s.y)
            if gc:
                self._block_set.add(gc)

        self._fixed_mirrors: Dict[Tuple[int, int], str] = {}
        for s in self.current_level.get_sprites_by_tag("fixed"):
            gc = self._px_to_grid(s.x, s.y)
            if gc:
                if "prism" in (s.tags or []):
                    mtype = "prism"
                elif "bs" in (s.tags or []):
                    mtype = "bs"
                else:
                    mtype = "fs"
                self._fixed_mirrors[gc] = mtype

        if self.current_level_index == 0:
            self.cursor_col = 0
            self.cursor_row = 0
        elif self.current_level_index == 1:
            self.cursor_col = self.gw - 1
            self.cursor_row = 0
        else:
            self.cursor_col, self.cursor_row = self._random_cursor_position()

        self._trace_beam()

    def _random_cursor_position(self) -> Tuple[int, int]:
        free_cells = []
        for c in range(self.gw):
            for r in range(self.gh):
                if not self._cell_occupied(c, r):
                    free_cells.append((c, r))
        if free_cells:
            return self._rng.choice(free_cells)
        return (0, 0)

    def _grid_to_px(self, col: int, row: int) -> Tuple[int, int]:
        return (self.ox + col * self.cell, self.oy + row * self.cell)

    def _px_to_grid(self, px: int, py: int) -> Optional[Tuple[int, int]]:
        if self.cell == 0:
            return None
        col = (px - self.ox) // self.cell
        row = (py - self.oy) // self.cell
        if 0 <= col < self.gw and 0 <= row < self.gh:
            return (col, row)
        return None

    def _display_to_grid(self, dx: int, dy: int) -> Optional[Tuple[int, int]]:
        coords = self.camera.display_to_grid(dx, dy)
        if coords:
            wx, wy = coords
            return self._px_to_grid(wx, wy)
        return None

    def _cell_has_wall(self, col: int, row: int) -> bool:
        return (col, row) in self._block_set or (col, row) in self._mwall_set

    def _cell_is_laser(self, col: int, row: int) -> bool:
        return col == self.laser_col and row == self.laser_row

    def _cell_has_placed_mirror(self, col: int, row: int) -> int:
        for i, (pc, pr) in enumerate(self.placed_positions):
            if pc == col and pr == row:
                return i
        return -1

    def _cell_has_fixed_mirror(self, col: int, row: int) -> bool:
        return (col, row) in self._fixed_mirrors

    def _cell_is_portal(self, col: int, row: int) -> bool:
        return (col, row) in self._portal_map

    def _cell_has_gem(self, col: int, row: int) -> bool:
        return (col, row) in self._gem_set

    def _cell_has_decoy(self, col: int, row: int) -> bool:
        return (col, row) in self._decoy_set

    def _cell_has_bomb(self, col: int, row: int) -> bool:
        return (col, row) in self._bomb_set

    def _cell_occupied(self, col: int, row: int) -> bool:
        if self._cell_is_laser(col, row):
            return True
        if self._cell_has_wall(col, row):
            return True
        if self._cell_has_placed_mirror(col, row) >= 0:
            return True
        if self._cell_has_fixed_mirror(col, row):
            return True
        if self._cell_is_portal(col, row):
            return True
        if self._cell_has_gem(col, row):
            return True
        if self._cell_has_decoy(col, row):
            return True
        if self._cell_has_bomb(col, row):
            return True
        return False

    def _get_mirror_at(self, col: int, row: int) -> Optional[str]:
        idx = self._cell_has_placed_mirror(col, row)
        if idx >= 0:
            return self.placed_types[idx]
        if (col, row) in self._fixed_mirrors:
            return self._fixed_mirrors[(col, row)]
        return None

    def _is_blocked(self, col: int, row: int) -> bool:
        if col < 0 or col >= self.gw or row < 0 or row >= self.gh:
            return True
        if (col, row) in self._block_set:
            return True
        if (col, row) in self._mwall_set:
            if self._get_mirror_at(col, row) is not None:
                return False
            return True
        return False

    def _get_portal_exit(
        self, col: int, row: int, dx: int, dy: int, beam_color: Optional[str]
    ) -> Optional[Tuple[int, int, int, int]]:
        key = (col, row)
        if key not in self._portal_map:
            return None

        pi, side = self._portal_map[key]
        pp = self.portal_pairs[pi]

        portal_beam_color = pp.get("beam_color", None)
        if portal_beam_color is not None:
            if beam_color != portal_beam_color:
                return None

        if side == "a":
            ec, er = pp["b"][0], pp["b"][1]
        else:
            ec, er = pp["a"][0], pp["a"][1]

        exit_dir_str = pp.get("exit_dir", None)
        if exit_dir_str is None:
            return (ec, er, dx, dy)
        else:
            edx, edy = DIR_MAP[exit_dir_str]
            if side == "b":
                edx, edy = -edx, -edy
            return (ec, er, edx, edy)

    def _shift_moving_walls(self) -> None:
        if not self._mwall_defs:
            return

        for s in self._mwall_sprites:
            self.current_level.remove_sprite(s)
        self._mwall_sprites.clear()
        self._mwall_set.clear()

        for i, mwd in enumerate(self._mwall_defs):
            positions = mwd["positions"]
            self._mwall_indices[i] = (self._mwall_indices[i] + 1) % len(positions)
            nc, nr = positions[self._mwall_indices[i]]
            self._mwall_set.add((nc, nr))

            px, py = self._grid_to_px(nc, nr)
            ms = sprites["mwall"].clone().set_position(px, py).set_scale(SPRITE_SCALE)
            self.current_level.add_sprite(ms)
            self._mwall_sprites.append(ms)

            sfx = (
                sprites["shift_fx"].clone().set_position(px, py).set_scale(SPRITE_SCALE)
            )
            self.current_level.add_sprite(sfx)
            self.effect_sprites.append(sfx)

    def _clear_beam(self) -> None:
        for s in self.beam_sprites:
            self.current_level.remove_sprite(s)
        self.beam_sprites.clear()

    def _clear_effects(self) -> None:
        for s in self.effect_sprites:
            self.current_level.remove_sprite(s)
        self.effect_sprites.clear()

    def _draw_beam_segment(
        self,
        col: int,
        row: int,
        dx: int,
        dy: int,
        beam_color: Optional[str],
        global_step_counter: int,
    ) -> None:
        px, py = self._grid_to_px(col, row)
        palette_color = _beam_palette_color(beam_color)

        cx_w = px + self.cell // 2
        cy_w = py + self.cell // 2

        half = self.cell // 2

        for p in range(-half, half):
            world_x = cx_w + dx * p
            world_y = cy_w + dy * p

            dot = sprites["beam_px"].clone().set_position(world_x, world_y).set_scale(1)
            dot.color_remap(0, palette_color)
            self.current_level.add_sprite(dot)
            self.beam_sprites.append(dot)

    def _trace_beam(self) -> None:
        self._clear_beam()
        self._clear_effects()

        hit_gems: List[Tuple[int, int]] = []
        hit_decoys: List[Tuple[int, int]] = []
        hit_bombs: List[Tuple[int, int]] = []

        self._beam_cells: set = set()

        visited: set = set()

        max_steps_per_branch = (self.gw + self.gh) * 4
        max_total_steps = max_steps_per_branch * 10

        queue: deque = deque()
        dx0, dy0 = DIR_MAP[self.laser_dir]
        queue.append((self.laser_col, self.laser_row, dx0, dy0, self.laser_color, 0))

        total_steps = 0

        while queue:
            start_col, start_row, dx, dy, beam_color, step_counter = queue.popleft()
            col = start_col
            row = start_row
            sc = step_counter

            for _ in range(max_steps_per_branch):
                total_steps += 1
                if total_steps > max_total_steps:
                    break

                col += dx
                row += dy

                if self._is_blocked(col, row):
                    break

                if self._cell_is_laser(col, row):
                    break

                color_key = beam_color if beam_color else "_"
                state = (col, row, dx, dy, color_key)
                if state in visited:
                    break
                visited.add(state)
                self._beam_cells.add((col, row))

                m = self._get_mirror_at(col, row)

                if not m:
                    self._draw_beam_segment(col, row, dx, dy, beam_color, sc)
                sc += 1

                if (col, row) in self._gem_set and (col, row) not in hit_gems:
                    gem_color = self._gem_color_map.get((col, row), None)
                    if _color_matches_gem(beam_color, gem_color):
                        hit_gems.append((col, row))
                        gpx, gpy = self._grid_to_px(col, row)
                        gfx = (
                            sprites["gem_collected"]
                            .clone()
                            .set_position(gpx, gpy)
                            .set_scale(SPRITE_SCALE)
                        )
                        self.current_level.add_sprite(gfx)
                        self.effect_sprites.append(gfx)

                if (col, row) in self._decoy_set and (col, row) not in hit_decoys:
                    hit_decoys.append((col, row))

                if (col, row) in self._bomb_set and (col, row) not in hit_bombs:
                    hit_bombs.append((col, row))
                    bpx, bpy = self._grid_to_px(col, row)
                    efx = (
                        sprites["explosion"]
                        .clone()
                        .set_position(bpx, bpy)
                        .set_scale(SPRITE_SCALE)
                    )
                    self.current_level.add_sprite(efx)
                    self.effect_sprites.append(efx)

                if m:
                    new_dirs = _reflect(dx, dy, m, beam_color)

                    if m == "prism":
                        if len(new_dirs) >= 1:
                            dx, dy = new_dirs[0][0], new_dirs[0][1]
                            beam_color = new_dirs[0][2]
                        for nd in new_dirs[1:]:
                            queue.append((col, row, nd[0], nd[1], nd[2], sc))
                        continue
                    else:
                        dx, dy = new_dirs[0][0], new_dirs[0][1]
                        beam_color = new_dirs[0][2]
                        continue

                portal_exit = self._get_portal_exit(col, row, dx, dy, beam_color)
                if portal_exit:
                    ppx, ppy = self._grid_to_px(col, row)
                    vfx = (
                        sprites["vortex"]
                        .clone()
                        .set_position(ppx, ppy)
                        .set_scale(SPRITE_SCALE)
                    )
                    self.current_level.add_sprite(vfx)
                    self.effect_sprites.append(vfx)

                    ec, er, ndx, ndy = portal_exit

                    epx, epy = self._grid_to_px(ec, er)
                    vfx2 = (
                        sprites["sparkle"]
                        .clone()
                        .set_position(epx, epy)
                        .set_scale(SPRITE_SCALE)
                    )
                    self.current_level.add_sprite(vfx2)
                    self.effect_sprites.append(vfx2)

                    self._draw_beam_segment(ec, er, ndx, ndy, beam_color, sc)
                    sc += 1

                    col = ec
                    row = er
                    dx = ndx
                    dy = ndy

                    if (col, row) in self._gem_set and (col, row) not in hit_gems:
                        gem_color = self._gem_color_map.get((col, row), None)
                        if _color_matches_gem(beam_color, gem_color):
                            hit_gems.append((col, row))
                    if (col, row) in self._decoy_set and (col, row) not in hit_decoys:
                        hit_decoys.append((col, row))
                    if (col, row) in self._bomb_set and (col, row) not in hit_bombs:
                        hit_bombs.append((col, row))

                    continue

        self._last_hit_gems = hit_gems
        self._last_hit_decoys = hit_decoys
        self._last_hit_bombs = hit_bombs
        self._known_hit_bombs.update(hit_bombs)

    def _check_win(self) -> bool:
        if len(self._last_hit_gems) != len(self.gem_coords):
            return False
        if len(self._last_hit_decoys) > 0:
            return False
        if len(self._last_hit_bombs) > 0:
            return False
        return True

    def _do_win_advance(self) -> None:
        self._clear_effects()
        self._clear_beam()
        if self.current_level_index >= self.total_levels - 1:
            self._game_won = True
        self.next_level()

    def _do_bomb_reset(self) -> None:
        self.lives -= 1
        if self.lives <= 0:
            self.lose()
            return
        self._do_fail_reset()

    def _do_fail_reset(self) -> None:
        self._clear_effects()
        self._clear_beam()
        self.moves_used = 0
        self._history.clear()
        for s in self.placed_mirrors:
            self.current_level.remove_sprite(s)
        self.placed_mirrors.clear()
        self.placed_positions.clear()
        self.placed_types.clear()

        if self._mwall_defs:
            for s in self._mwall_sprites:
                self.current_level.remove_sprite(s)
            self._mwall_sprites.clear()
            self._mwall_set.clear()
            for i, mwd in enumerate(self._mwall_defs):
                self._mwall_indices[i] = mwd.get("start", 0)
                msc, msr = mwd["positions"][self._mwall_indices[i]]
                self._mwall_set.add((msc, msr))
                mpx, mpy = self._grid_to_px(msc, msr)
                ms = (
                    sprites["mwall"]
                    .clone()
                    .set_position(mpx, mpy)
                    .set_scale(SPRITE_SCALE)
                )
                self.current_level.add_sprite(ms)
                self._mwall_sprites.append(ms)

        self.mirrors_bs_remaining = self.mirrors_bs_total
        self.mirrors_fs_remaining = self.mirrors_fs_total
        self.mirrors_prism_remaining = self.mirrors_prism_total
        self.mwall_place_counter = 0

        if self.mirrors_bs_total > 0:
            self.current_mirror_type = 0
        elif self.mirrors_fs_total > 0:
            self.current_mirror_type = 1
        elif self.mirrors_prism_total > 0:
            self.current_mirror_type = 2
        else:
            self.current_mirror_type = 0

        self._gem_set = set(self.gem_coords)
        self._last_hit_gems = []
        self._last_hit_decoys = []
        self._last_hit_bombs = []
        self._known_hit_bombs = set()
        self.cursor_col, self.cursor_row = self._random_cursor_position()
        self._trace_beam()

    def _auto_advance_mirror_type(self) -> None:
        cur = self.current_mirror_type
        remaining_map = {
            0: self.mirrors_bs_remaining,
            1: self.mirrors_fs_remaining,
            2: self.mirrors_prism_remaining,
        }
        if remaining_map[cur] > 0:
            return
        for offset in range(1, 4):
            nxt = (cur + offset) % 3
            if remaining_map[nxt] > 0:
                self.current_mirror_type = nxt
                return

    def _save_state(self) -> None:
        snapshot: Dict = {
            "cursor_col": self.cursor_col,
            "cursor_row": self.cursor_row,
            "current_mirror_type": self.current_mirror_type,
            "selection_mode": self._selection_mode,
            "mirrors_bs_remaining": self.mirrors_bs_remaining,
            "mirrors_fs_remaining": self.mirrors_fs_remaining,
            "mirrors_prism_remaining": self.mirrors_prism_remaining,
            "placed_positions": list(self.placed_positions),
            "placed_types": list(self.placed_types),
            "mwall_indices": list(self._mwall_indices),
            "mwall_place_counter": self.mwall_place_counter,
        }
        self._history.append(snapshot)

    def _undo(self) -> None:
        if not self._history:
            return
        snap = self._history.pop()

        self.cursor_col = snap["cursor_col"]
        self.cursor_row = snap["cursor_row"]
        self.current_mirror_type = snap["current_mirror_type"]
        self._selection_mode = snap["selection_mode"]
        self.mirrors_bs_remaining = snap["mirrors_bs_remaining"]
        self.mirrors_fs_remaining = snap["mirrors_fs_remaining"]
        self.mirrors_prism_remaining = snap["mirrors_prism_remaining"]
        self.mwall_place_counter = snap["mwall_place_counter"]

        for s in self.placed_mirrors:
            self.current_level.remove_sprite(s)
        self.placed_mirrors.clear()
        self.placed_positions.clear()
        self.placed_types.clear()

        for (col, row), mtype in zip(snap["placed_positions"], snap["placed_types"]):
            mpx, mpy = self._grid_to_px(col, row)
            if mtype == "bs":
                ms = (
                    sprites["mirror_bs"]
                    .clone()
                    .set_position(mpx, mpy)
                    .set_scale(SPRITE_SCALE)
                )
            elif mtype == "fs":
                ms = (
                    sprites["mirror_fs"]
                    .clone()
                    .set_position(mpx, mpy)
                    .set_scale(SPRITE_SCALE)
                )
            else:
                ms = (
                    sprites["mirror_prism"]
                    .clone()
                    .set_position(mpx, mpy)
                    .set_scale(SPRITE_SCALE)
                )
            self.current_level.add_sprite(ms)
            self.placed_mirrors.append(ms)
            self.placed_positions.append((col, row))
            self.placed_types.append(mtype)

        saved_indices = snap["mwall_indices"]
        if self._mwall_defs:
            for s in self._mwall_sprites:
                self.current_level.remove_sprite(s)
            self._mwall_sprites.clear()
            self._mwall_set.clear()
            for i, mwd in enumerate(self._mwall_defs):
                self._mwall_indices[i] = saved_indices[i]
                nc, nr = mwd["positions"][self._mwall_indices[i]]
                self._mwall_set.add((nc, nr))
                px, py = self._grid_to_px(nc, nr)
                ms = (
                    sprites["mwall"]
                    .clone()
                    .set_position(px, py)
                    .set_scale(SPRITE_SCALE)
                )
                self.current_level.add_sprite(ms)
                self._mwall_sprites.append(ms)

        self._trace_beam()

    def _place_or_remove_at(self, col: int, row: int) -> bool:

        removed = False
        idx = self._cell_has_placed_mirror(col, row)
        if idx >= 0:
            removed_type = self.placed_types[idx]
            self.current_level.remove_sprite(self.placed_mirrors[idx])
            self.placed_mirrors.pop(idx)
            self.placed_positions.pop(idx)
            self.placed_types.pop(idx)
            if removed_type == "bs":
                self.mirrors_bs_remaining += 1
            elif removed_type == "fs":
                self.mirrors_fs_remaining += 1
            else:
                self.mirrors_prism_remaining += 1
            removed = True

        placed_new = False
        if not removed:
            if self.current_mirror_type == 0:
                type_remaining = self.mirrors_bs_remaining
            elif self.current_mirror_type == 1:
                type_remaining = self.mirrors_fs_remaining
            else:
                type_remaining = self.mirrors_prism_remaining

            if type_remaining > 0 and not self._cell_occupied(col, row):
                mpx, mpy = self._grid_to_px(col, row)
                if self.current_mirror_type == 0:
                    ms = (
                        sprites["mirror_bs"]
                        .clone()
                        .set_position(mpx, mpy)
                        .set_scale(SPRITE_SCALE)
                    )
                    mtype = "bs"
                elif self.current_mirror_type == 1:
                    ms = (
                        sprites["mirror_fs"]
                        .clone()
                        .set_position(mpx, mpy)
                        .set_scale(SPRITE_SCALE)
                    )
                    mtype = "fs"
                else:
                    ms = (
                        sprites["mirror_prism"]
                        .clone()
                        .set_position(mpx, mpy)
                        .set_scale(SPRITE_SCALE)
                    )
                    mtype = "prism"
                self.current_level.add_sprite(ms)
                self.placed_mirrors.append(ms)
                self.placed_positions.append((col, row))
                self.placed_types.append(mtype)
                if mtype == "bs":
                    self.mirrors_bs_remaining -= 1
                elif mtype == "fs":
                    self.mirrors_fs_remaining -= 1
                else:
                    self.mirrors_prism_remaining -= 1
                placed_new = True

                self._auto_advance_mirror_type()

        if placed_new and self.mwall_shift_every > 0:
            self.mwall_place_counter += 1
            if self.mwall_place_counter >= self.mwall_shift_every:
                self.mwall_place_counter = 0
                self._shift_moving_walls()

        self._trace_beam()
        return placed_new

    def _cycle_mirror_type(self, direction: int) -> None:
        available = []
        if self.mirrors_bs_total > 0:
            available.append(0)
        if self.mirrors_fs_total > 0:
            available.append(1)
        if self.mirrors_prism_total > 0:
            available.append(2)
        if available:
            try:
                cur_idx = available.index(self.current_mirror_type)
                next_idx = (cur_idx + direction) % len(available)
            except ValueError:
                next_idx = 0
            self.current_mirror_type = available[next_idx]

    def _handle_hud_click(self, wx: int, wy: int) -> bool:
        hud_box_w = 8
        hud_rx = 64 - hud_box_w
        hud_ry = 4
        hud_types = [
            (0, self.mirrors_bs_total),
            (1, self.mirrors_fs_total),
            (2, self.mirrors_prism_total),
        ]
        for idx, (mtype, total) in enumerate(hud_types):
            if total == 0:
                continue
            bx = hud_rx
            by = hud_ry + idx * 12
            if bx <= wx < bx + hud_box_w and by <= wy < by + hud_box_w:
                self.current_mirror_type = mtype
                return True
        return False

    def _place_and_evaluate(self, col: int, row: int) -> bool:
        known_before = set(self._known_hit_bombs)
        did_place = self._place_or_remove_at(col, row)
        if did_place and set(self._last_hit_bombs) - known_before:
            self._do_bomb_reset()
            return True
        if self._check_win():
            self._do_win_advance()
            return True
        return False

    def step(self) -> None:

        if self.action.id == GameAction.RESET:
            self._do_fail_reset()
            self.complete_action()
            return

        if self.action.id == GameAction.ACTION7:
            self.moves_used += 1
            self._undo()
            self.complete_action()
            return

        self.moves_used += 1

        if self.max_moves > 0 and self.moves_used >= self.max_moves:
            self._do_bomb_reset()
            self.complete_action()
            return

        self._save_state()

        if self.action.id == GameAction.ACTION1:
            if self._selection_mode:
                self._cycle_mirror_type(-1)
            else:
                self.cursor_row = (self.cursor_row - 1) % self.gh
            self._prev_action_id = self.action.id
            self.complete_action()
            return

        if self.action.id == GameAction.ACTION2:
            if self._selection_mode:
                self._cycle_mirror_type(1)
            else:
                self.cursor_row = (self.cursor_row + 1) % self.gh
            self._prev_action_id = self.action.id
            self.complete_action()
            return

        if self.action.id == GameAction.ACTION3:
            self._selection_mode = False
            self.cursor_col = (self.cursor_col - 1) % self.gw
            self._prev_action_id = self.action.id
            self.complete_action()
            return

        if self.action.id == GameAction.ACTION4:
            self._selection_mode = False
            self.cursor_col = (self.cursor_col + 1) % self.gw
            self._prev_action_id = self.action.id
            self.complete_action()
            return

        if self.action.id == GameAction.ACTION5:
            if self._selection_mode:
                self._selection_mode = False
                self._place_and_evaluate(self.cursor_col, self.cursor_row)
                self._prev_action_id = self.action.id
                self.complete_action()
                return
            self._selection_mode = True
            self._prev_action_id = self.action.id
            self.complete_action()
            return

        if self.action.id == GameAction.ACTION6:
            self._selection_mode = False
            self._prev_action_id = self.action.id
            raw_x = self.action.data.get("x", 0)
            raw_y = self.action.data.get("y", 0)

            coords = self.camera.display_to_grid(raw_x, raw_y)
            if coords:
                wx, wy = coords
                if self._handle_hud_click(wx, wy):
                    self.complete_action()
                    return

                gc = self._px_to_grid(wx, wy)
                if gc:
                    col, row = gc
                    self.cursor_col = col
                    self.cursor_row = row
                    self._place_and_evaluate(col, row)

        self._prev_action_id = self.action.id
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

_MIRROR_NAMES = {0: "BS", 1: "FS", 2: "PR"}


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

    def __init__(self, seed: int = 0) -> None:
        self._engine = Lm42(seed)
        self.seed = seed
        self._total_turns = 0
        self._done = False
        self._last_action_was_reset = False

    def _episode_terminal(self) -> bool:
        g = self._engine
        return g._game_won or g.current_level_index >= g.total_levels or g.lives <= 0

    def reset(self) -> GameState:
        e = self._engine
        self._total_turns = 0
        self._done = False
        game_won = e._game_won or e.current_level_index >= e.total_levels
        if game_won or e.moves_used == 0 or self._last_action_was_reset:
            self._engine = Lm42(self.seed)
            e = self._engine
        else:
            e.perform_action(ActionInput(id=GameAction.RESET))
        self._last_action_was_reset = True
        return self._create_game_state()

    def get_actions(self) -> List[str]:
        if self._episode_terminal():
            return ["reset"]
        g = self._engine
        actions = ["reset","up", "down", "left", "right", "select", "undo"]
        if hasattr(g, "gw") and hasattr(g, "gh"):
            for r in range(g.gh):
                for c in range(g.gw):
                    actions.append(f"click {c} {r}")
        return actions

    def _outcome_after_step(
        self, lives_before: int, level_before: int
    ) -> Tuple[float, bool, Dict]:
        g = self._engine
        info: Dict = {
            "lives": g.lives,
            "level": g.current_level_index + 1,
            "moves_used": g.moves_used,
            "move_limit": g.max_moves,
        }
        reward = 0.0
        done = False
        if g.lives < lives_before:
            info["event"] = "life_lost"
            if g.lives <= 0:
                info["event"] = "game_over"
                done = True
        elif g._game_won and level_before == g.total_levels - 1:
            reward = 1.0 / g.total_levels
            info["event"] = "game_complete"
            done = True
        elif g.current_level_index != level_before:
            reward = 1.0 / g.total_levels
            info["event"] = "level_complete"
        return reward, done, info

    def step(self, action: str) -> StepResult:
        e = self._engine

        if action == "reset":
            game_won = e._game_won or e.current_level_index >= e.total_levels
            full_restart = game_won or e.moves_used == 0 or self._last_action_was_reset
            state = self.reset()
            return StepResult(
                state=state,
                reward=0.0,
                done=False,
                info={"action": "reset", "full_restart": full_restart},
            )

        if action.startswith("click "):
            parts = action.split()
            if len(parts) == 3:
                try:
                    col, row = int(parts[1]), int(parts[2])
                    px = e.ox + col * e.cell + e.cell // 2
                    py = e.oy + row * e.cell + e.cell // 2
                    self._last_action_was_reset = False
                    self._total_turns += 1
                    lives_before = e.lives
                    level_before = e.current_level_index
                    action_input = ActionInput(
                        id=GameAction.ACTION6, data={"x": px, "y": py}
                    )
                    e.perform_action(action_input)
                    reward, done, info = self._outcome_after_step(
                        lives_before, level_before
                    )
                    self._done = done or self._episode_terminal()
                    return StepResult(
                        state=self._create_game_state(),
                        reward=reward,
                        done=self._done,
                        info=info,
                    )
                except (ValueError, IndexError):
                    pass
            return StepResult(
                state=self._create_game_state(),
                reward=0.0,
                done=self.is_done(),
                info={"error": f"Invalid action: {action}"},
            )

        if action not in ACTION_MAP:
            return StepResult(
                state=self._create_game_state(),
                reward=0.0,
                done=self.is_done(),
                info={"error": f"Invalid action: {action}"},
            )

        self._last_action_was_reset = False
        self._total_turns += 1

        lives_before = e.lives
        level_before = e.current_level_index

        ga = ACTION_MAP[action]
        action_input = ActionInput(id=ga)
        e.perform_action(action_input)

        reward, done, info = self._outcome_after_step(lives_before, level_before)
        self._done = done or self._episode_terminal()

        return StepResult(
            state=self._create_game_state(),
            reward=reward,
            done=self._done,
            info=info,
        )

    def is_done(self) -> bool:
        return self._done or self._episode_terminal()

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
        h, w, _ = rgb.shape
        raw = bytearray()
        for y in range(h):
            raw.append(0)
            raw.extend(rgb[y].tobytes())
        compressed = zlib.compress(bytes(raw))

        def _chunk(ctype: bytes, data: bytes) -> bytes:
            c = ctype + data
            return (
                struct.pack(">I", len(data))
                + c
                + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
            )

        sig = b"\x89PNG\r\n\x1a\n"
        ihdr = _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        idat = _chunk(b"IDAT", compressed)
        iend = _chunk(b"IEND", b"")
        return sig + ihdr + idat + iend

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

    def _create_game_state(self) -> GameState:
        g = self._engine
        if hasattr(g, "laser_col"):
            body = self._build_text_obs()
        else:
            body = "(no level loaded)"
        text_observation = f"Agent turn: {self._total_turns}\n\n{body}"
        terminal = self._episode_terminal()
        va = None if terminal else self.get_actions()
        return GameState(
            text_observation=text_observation,
            image_observation=self._build_image_bytes(),
            valid_actions=va,
            turn=self._total_turns,
            metadata=self._build_metadata(),
        )

    def _build_text_obs(self) -> str:
        g = self._engine
        lines = []
        lines.append(
            f"LEVEL {g.current_level_index + 1}/{g.total_levels} "
            f"LIVES {g.lives}/{g.max_lives} "
            f"MOVES {g.moves_used}/{g.max_moves}"
        )
        lines.append(
            f"MIRRORS BS:{g.mirrors_bs_remaining} FS:{g.mirrors_fs_remaining} "
            f"PR:{g.mirrors_prism_remaining} "
            f"SELECTED:{_MIRROR_NAMES.get(g.current_mirror_type, '?')}"
        )
        sel = "ON" if g._selection_mode else "OFF"
        lines.append(f"CURSOR ({g.cursor_col},{g.cursor_row}) SELECT:{sel}")
        lines.append(
            f"LASER ({g.laser_col},{g.laser_row}) "
            f"DIR:{g.laser_dir} COLOR:{g.laser_color or 'default'}"
        )
        hit_set = set(g._last_hit_gems) if hasattr(g, "_last_hit_gems") else set()
        gem_count = len(g.gem_coords) if hasattr(g, "gem_coords") else 0
        hit_count = len(hit_set)
        gem_parts = []
        if hasattr(g, "gem_data"):
            for gc, gr, gcolor in g.gem_data:
                tag = gcolor or "any"
                marker = "*" if (gc, gr) in hit_set else ""
                gem_parts.append(f"({gc},{gr}){tag}{marker}")
        lines.append(f"GEMS {hit_count}/{gem_count}: " + " ".join(gem_parts))
        if hasattr(g, "portal_pairs") and g.portal_pairs:
            pp_parts = []
            for idx, pp in enumerate(g.portal_pairs):
                lbl = chr(65 + idx)
                ac, ar = pp["a"][0], pp["a"][1]
                bc, br = pp["b"][0], pp["b"][1]
                pcolor = pp.get("beam_color") or "any"
                pp_parts.append(f"{lbl}=({ac},{ar})-({bc},{br}) filter={pcolor}")
            lines.append("PORTALS: " + " ".join(pp_parts))
        grid: dict = {}
        if hasattr(g, "_beam_cells"):
            for pos in g._beam_cells:
                grid[pos] = "*"
        if hasattr(g, "_block_set"):
            for pos in g._block_set:
                grid[pos] = "#"
        if hasattr(g, "_mwall_set"):
            for pos in g._mwall_set:
                grid[pos] = "M"
        if hasattr(g, "_portal_map"):
            for pos in g._portal_map:
                grid[pos] = "O"
        if hasattr(g, "_gem_set"):
            color_map = g._gem_color_map if hasattr(g, "_gem_color_map") else {}
            for pos in g._gem_set:
                gc = color_map.get(pos)
                is_hit = pos in hit_set
                if gc == "red":
                    grid[pos] = "r" if is_hit else "R"
                elif gc == "blue":
                    grid[pos] = "c" if is_hit else "C"
                else:
                    grid[pos] = "g" if is_hit else "G"
        if hasattr(g, "_decoy_set"):
            for pos in g._decoy_set:
                grid[pos] = "D"
        if hasattr(g, "_bomb_set"):
            for pos in g._bomb_set:
                grid[pos] = "B"
        if hasattr(g, "_fixed_mirrors"):
            for pos, mt in g._fixed_mirrors.items():
                if mt == "prism":
                    grid[pos] = "P"
                elif mt == "bs":
                    grid[pos] = "\\"
                else:
                    grid[pos] = "/"
        if hasattr(g, "placed_positions") and hasattr(g, "placed_types"):
            for i, pos in enumerate(g.placed_positions):
                mt = g.placed_types[i]
                if mt == "bs":
                    grid[pos] = "\\"
                elif mt == "fs":
                    grid[pos] = "/"
                else:
                    grid[pos] = "P"
        grid[(g.laser_col, g.laser_row)] = "L"
        header = "  " + "".join(str(c) for c in range(g.gw))
        lines.append(header)
        for r in range(g.gh):
            row_chars = ""
            for c in range(g.gw):
                row_chars += grid.get((c, r), ".")
            lines.append(f"{r} {row_chars}")
        return "\n".join(lines)

    def _build_metadata(self) -> dict:
        g = self._engine
        return {
            "level_index": g.current_level_index,
            "total_levels": g.total_levels,
            "lives": g.lives,
            "max_lives": g.max_lives,
            "moves_used": g.moves_used,
            "max_moves": g.max_moves,
            "cursor": (g.cursor_col, g.cursor_row),
            "mirrors_remaining": {
                "bs": g.mirrors_bs_remaining,
                "fs": g.mirrors_fs_remaining,
                "prism": g.mirrors_prism_remaining,
            },
            "current_mirror_type": g.current_mirror_type,
            "gems_hit": len(g._last_hit_gems) if hasattr(g, "_last_hit_gems") else 0,
            "total_gems": len(g.gem_coords) if hasattr(g, "gem_coords") else 0,
            "terminal": self._episode_terminal(),
            "done": self._done,
        }


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
                base = a.split()[0] if " " in a else a
                idx = self._string_to_action.get(base)
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
