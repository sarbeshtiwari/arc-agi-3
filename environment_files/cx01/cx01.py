import random
import struct
import zlib
from dataclasses import dataclass, field
from collections import deque as _deque
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


BACKGROUND_COLOR = 0
PADDING_COLOR = 0

FLOOR_COLOR = 1
WALL_COLOR = 3
CURSOR_COLOR = 15

ICE_COLOR = 0
TRAIL_COLOR = 5

C_FLASH = 11

TARGET_COLORS_L1 = [9, 12, 11, 10, 7, 13, 4, 6]

TARGET_COLOR_L2 = 4

TARGET_COLORS_L3 = [12, 11, 9, 13, 7, 10, 4, 2]
PORTAL_A_COLOR = 10
PORTAL_B_COLOR = 6
PORTAL_C_COLOR = 14

TARGET_COLORS_L4 = [11, 12, 9, 10, 7, 13, 4, 2]
MIRROR_COLOR = 6

MOVE_BUDGETS = [240, 160, 320, 400]

CAMERA_W = 16
CAMERA_H = 16

levels = [
    Level(sprites=[], grid_size=(16, 16)),
    Level(sprites=[], grid_size=(16, 16)),
    Level(sprites=[], grid_size=(16, 16)),
    Level(sprites=[], grid_size=(16, 16)),
]

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


class HudDisplay(RenderableUserDisplay):
    BAR_WIDTH = 42
    BAR_X = 4
    BAR_Y = 61

    def __init__(self, game: "Cx01") -> None:
        self._game = game
        self.max_moves = 0
        self.remaining = 0

    def set_limit(self, max_moves: int) -> None:
        self.max_moves = max_moves
        self.remaining = max_moves

    def tick(self) -> bool:
        if self.remaining > 0:
            self.remaining -= 1
        return self.remaining > 0

    def reset(self) -> None:
        self.remaining = self.max_moves

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        if self.max_moves == 0 or self._game._flash_active:
            return frame

        filled = int(self.BAR_WIDTH * self.remaining / self.max_moves)
        for i in range(self.BAR_WIDTH):
            color = 11 if i < filled else 3
            frame[self.BAR_Y : self.BAR_Y + 2, self.BAR_X + i] = color

        for i in range(3):
            x = 52 + i * 4
            color = 8 if self._game._lives > i else 3
            frame[self.BAR_Y : self.BAR_Y + 2, x : x + 2] = color

        return frame


class Cx01(ARCBaseGame):
    MAX_LIVES = 3

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self._rng = random.Random(seed)

        self._cursor_x: int = 0
        self._cursor_y: int = 0

        self._lives: int = self.MAX_LIVES
        self._flash_active: bool = False

        self._targets: list[tuple[int, int]] = []
        self._trail: set[tuple[int, int]] = set()
        self._stoppers: list[tuple[int, int]] = []
        self._portals: dict[tuple, tuple] = {}
        self._mirror_x: int = 0
        self._mirror_y: int = 0
        self._walls_l4: set[tuple[int, int]] = set()

        self._saved_state: dict = {}
        self._undo_stack: list[dict] = []

        self._hud = HudDisplay(self)

        camera = Camera(
            0,
            0,
            CAMERA_W,
            CAMERA_H,
            BACKGROUND_COLOR,
            PADDING_COLOR,
            [self._hud],
        )

        super().__init__(
            game_id="cx01",
            levels=levels,
            camera=camera,
            available_actions=[0, 1, 2, 3, 4, 7],
        )

    def handle_reset(self) -> None:
        self._lives = self.MAX_LIVES
        super().handle_reset()

    def on_set_level(self, level: Level) -> None:
        self._lives = self.MAX_LIVES
        self._rng = random.Random(self._seed + self.level_index)
        idx = self.level_index
        budget = MOVE_BUDGETS[min(idx, len(MOVE_BUDGETS) - 1)]
        self._hud.set_limit(budget)
        self._flash_sprite = Sprite(
            pixels=[[C_FLASH]],
            name="flash_overlay",
            x=0,
            y=0,
            layer=10,
            visible=False,
            collidable=False,
        )
        self._flash_sprite.set_scale(CAMERA_W)
        self._flash_active = False
        self._setup_level()
        self.current_level.add_sprite(self._flash_sprite)

    def _setup_level(self) -> None:
        idx = self.level_index
        gw, gh = self.current_level.grid_size
        self._cursor_x = gw // 2
        self._cursor_y = gh // 2
        self._trail = set()
        self._targets = []
        self._stoppers = []
        self._portals = {}
        self._walls_l4 = set()
        self._undo_stack = []

        if idx == 0:
            self._setup_l1()
        elif idx == 1:
            self._setup_l2()
        elif idx == 2:
            self._setup_l3()
        elif idx == 3:
            self._setup_l4()

        self._saved_state = {
            "cursor_x": self._cursor_x,
            "cursor_y": self._cursor_y,
            "targets": list(self._targets),
            "trail": set(self._trail),
            "stoppers": list(self._stoppers),
            "portals": dict(self._portals),
            "mirror_x": self._mirror_x,
            "mirror_y": self._mirror_y,
            "walls_l4": set(self._walls_l4),
        }

        self._render()

    def _restore_snapshot(self) -> None:
        s = self._saved_state
        self._cursor_x = s["cursor_x"]
        self._cursor_y = s["cursor_y"]
        self._targets = list(s["targets"])
        self._trail = set(s["trail"])
        self._stoppers = list(s["stoppers"])
        self._portals = dict(s["portals"])
        self._mirror_x = s["mirror_x"]
        self._mirror_y = s["mirror_y"]
        self._walls_l4 = set(s["walls_l4"])
        self._undo_stack = []
        self._render()

    def _save_state(self) -> None:
        state = {
            "cursor_x": self._cursor_x,
            "cursor_y": self._cursor_y,
            "targets": list(self._targets),
            "trail": set(self._trail),
            "stoppers": list(self._stoppers),
            "portals": dict(self._portals),
            "mirror_x": self._mirror_x,
            "mirror_y": self._mirror_y,
            "walls_l4": set(self._walls_l4),
            "remaining": self._hud.remaining,
        }
        self._undo_stack.append(state)

    def _restore_from_undo(self) -> None:
        if not self._undo_stack:
            return
        state = self._undo_stack.pop()
        self._cursor_x = state["cursor_x"]
        self._cursor_y = state["cursor_y"]
        self._targets = list(state["targets"])
        self._trail = set(state["trail"])
        self._stoppers = list(state["stoppers"])
        self._portals = dict(state["portals"])
        self._mirror_x = state["mirror_x"]
        self._mirror_y = state["mirror_y"]
        self._walls_l4 = set(state["walls_l4"])
        self._hud.remaining = state["remaining"]
        self._render()

    def _setup_l1(self) -> None:
        gw, gh = self.current_level.grid_size
        occupied = {(self._cursor_x, self._cursor_y)}
        self._targets = self._place_unique(gw, gh, 8, occupied)
        self._trail = set()

    def _setup_l2(self) -> None:
        gw, gh = self.current_level.grid_size
        occupied = {(self._cursor_x, self._cursor_y)}
        corner_walls = [
            (2, 1),
            (1, 2),
            (gw - 3, 1),
            (gw - 2, 2),
            (1, gh - 3),
            (2, gh - 2),
            (gw - 2, gh - 3),
            (gw - 3, gh - 2),
        ]
        corner_walls = [(x, y) for x, y in corner_walls if (x, y) not in occupied]
        occupied.update(corner_walls)
        self._stoppers = corner_walls + self._place_unique(gw, gh, 10, occupied)
        stopper_set = set(self._stoppers)

        reachable: set = set()
        q = _deque([(self._cursor_x, self._cursor_y)])
        seen_pos: set = {(self._cursor_x, self._cursor_y)}
        while q:
            cx, cy = q.popleft()
            for dx, dy in [(1, 0), (-1, 0), (0, -1), (0, 1)]:
                px, py = cx, cy
                while True:
                    nx, ny = px + dx, py + dy
                    if nx <= 0 or nx >= gw - 1 or ny <= 0 or ny >= gh - 1:
                        break
                    if (nx, ny) in stopper_set:
                        break
                    px, py = nx, ny
                if (px, py) not in seen_pos:
                    seen_pos.add((px, py))
                    reachable.add((px, py))
                    q.append((px, py))
        reachable -= {(self._cursor_x, self._cursor_y)}
        reachable -= stopper_set
        reachable_list = sorted(reachable - occupied)
        self._rng.shuffle(reachable_list)
        self._targets = reachable_list[:5]

    def _setup_l3(self) -> None:
        gw, gh = self.current_level.grid_size
        occupied = {(self._cursor_x, self._cursor_y)}
        self._targets = self._place_unique(gw, gh, 8, occupied)
        occupied.update(self._targets)
        pa = self._place_unique(gw, gh, 2, occupied)
        occupied.update(pa)
        pb = self._place_unique(gw, gh, 2, occupied)
        occupied.update(pb)
        pc = self._place_unique(gw, gh, 2, occupied)
        self._portals = {
            tuple(pa[0]): tuple(pa[1]),
            tuple(pa[1]): tuple(pa[0]),
            tuple(pb[0]): tuple(pb[1]),
            tuple(pb[1]): tuple(pb[0]),
            tuple(pc[0]): tuple(pc[1]),
            tuple(pc[1]): tuple(pc[0]),
        }

    def _setup_l4(self) -> None:
        gw, gh = self.current_level.grid_size
        self._cursor_x, self._cursor_y = 3, 3
        self._mirror_x, self._mirror_y = gw - 4, gh - 4
        occupied = {(self._cursor_x, self._cursor_y), (self._mirror_x, self._mirror_y)}
        self._targets = self._place_unique(gw, gh, 8, occupied)
        occupied.update(self._targets)
        self._walls_l4 = set(self._place_isolated(gw, gh, 14, occupied))
        pa = self._place_unique(gw, gh, 2, occupied)
        occupied.update(pa)
        pb = self._place_unique(gw, gh, 2, occupied)
        self._portals = {
            tuple(pa[0]): tuple(pa[1]),
            tuple(pa[1]): tuple(pa[0]),
            tuple(pb[0]): tuple(pb[1]),
            tuple(pb[1]): tuple(pb[0]),
        }

    def _place_unique(self, gw, gh, count, occupied, margin=2):
        placed: list[tuple[int, int]] = []
        for i in range(count):
            found = False
            for attempt in range(200):
                x = self._rng.randint(margin, gw - 1 - margin)
                y = self._rng.randint(margin, gh - 1 - margin)
                if (x, y) not in occupied:
                    occupied.add((x, y))
                    placed.append((x, y))
                    found = True
                    break
            if not found:
                break
        return placed

    def _place_isolated(self, gw, gh, count, occupied):
        placed: list[tuple[int, int]] = []
        forbidden: set = set(occupied)
        for i in range(count):
            found = False
            for attempt in range(400):
                x = self._rng.randint(2, gw - 3)
                y = self._rng.randint(2, gh - 3)
                if (x, y) in forbidden:
                    continue
                if any(
                    (x + dx, y + dy) in set(placed)
                    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]
                ):
                    continue
                forbidden.add((x, y))
                placed.append((x, y))
                found = True
                break
            if not found:
                break
        return placed

    def _add_px(self, name, x, y, color, layer, collidable=False):
        self.current_level.add_sprite(
            Sprite(
                pixels=[[color]],
                name=name,
                x=x,
                y=y,
                layer=layer,
                collidable=collidable,
            )
        )

    def _render(self) -> None:
        for s in list(self.current_level._sprites):
            if s.name != "flash_overlay":
                self.current_level.remove_sprite(s)

        idx = self.level_index
        gw, gh = self.current_level.grid_size

        for cy in range(gh):
            for cx in range(gw):
                floor_c = ICE_COLOR if idx == 1 else FLOOR_COLOR
                self._add_px(f"f_{cx}_{cy}", cx, cy, floor_c, 1)

        for cx in range(gw):
            self._add_px(f"wt_{cx}", cx, 0, WALL_COLOR, 2)
            self._add_px(f"wb_{cx}", cx, gh - 1, WALL_COLOR, 2)
        for cy in range(1, gh - 1):
            self._add_px(f"wl_{cy}", 0, cy, WALL_COLOR, 2)
            self._add_px(f"wr_{cy}", gw - 1, cy, WALL_COLOR, 2)

        if idx == 0:
            for cell in self._trail:
                self._add_px(
                    f"tr_{cell[0]}_{cell[1]}", cell[0], cell[1], TRAIL_COLOR, 3
                )
            for i, (tx, ty) in enumerate(self._targets):
                color = TARGET_COLORS_L1[i % len(TARGET_COLORS_L1)]
                self._add_px(f"tgt_{i}", tx, ty, color, 4)

        elif idx == 1:
            for i, (sx, sy) in enumerate(self._stoppers):
                self._add_px(f"stop_{i}", sx, sy, WALL_COLOR, 2)
            for i, (tx, ty) in enumerate(self._targets):
                self._add_px(f"tgt_{i}", tx, ty, TARGET_COLOR_L2, 3)

        elif idx == 2:
            for i, (tx, ty) in enumerate(self._targets):
                color = TARGET_COLORS_L3[i % len(TARGET_COLORS_L3)]
                self._add_px(f"tgt_{i}", tx, ty, color, 3)
            portal_colors = [PORTAL_A_COLOR, PORTAL_B_COLOR, PORTAL_C_COLOR]
            seen: set = set()
            pair_idx = 0
            for src, dst in self._portals.items():
                key = frozenset([src, dst])
                if key in seen:
                    continue
                seen.add(key)
                color = portal_colors[pair_idx % len(portal_colors)]
                self._add_px(f"portal_{pair_idx}_0", src[0], src[1], color, 3)
                self._add_px(f"portal_{pair_idx}_1", dst[0], dst[1], color, 3)
                pair_idx += 1

        elif idx == 3:
            for i, (wx, wy) in enumerate(self._walls_l4):
                self._add_px(f"wall_{i}", wx, wy, WALL_COLOR, 2)
            for i, (tx, ty) in enumerate(self._targets):
                color = TARGET_COLORS_L4[i % len(TARGET_COLORS_L4)]
                self._add_px(f"tgt_{i}", tx, ty, color, 3)
            portal_colors = [PORTAL_A_COLOR, PORTAL_B_COLOR]
            seen: set = set()
            pair_idx = 0
            for src, dst in self._portals.items():
                key = frozenset([src, dst])
                if key in seen:
                    continue
                seen.add(key)
                color = portal_colors[pair_idx % len(portal_colors)]
                self._add_px(f"l4p_{pair_idx}_0", src[0], src[1], color, 3)
                self._add_px(f"l4p_{pair_idx}_1", dst[0], dst[1], color, 3)
                pair_idx += 1
            self._add_px("mirror", self._mirror_x, self._mirror_y, MIRROR_COLOR, 4)

        self._add_px("cursor", self._cursor_x, self._cursor_y, CURSOR_COLOR, 5)

    def _clamp(self, x, y):
        gw, gh = self.current_level.grid_size
        return max(1, min(gw - 2, x)), max(1, min(gh - 2, y))

    def _move_cursor(self, dx, dy):
        nx, ny = self._clamp(self._cursor_x + dx, self._cursor_y + dy)
        self._cursor_x, self._cursor_y = nx, ny

    def _slide_cursor(self, dx, dy):
        gw, gh = self.current_level.grid_size
        stopper_set = set(self._stoppers)
        while True:
            nx = self._cursor_x + dx
            ny = self._cursor_y + dy
            if nx <= 0 or nx >= gw - 1 or ny <= 0 or ny >= gh - 1:
                break
            if (nx, ny) in stopper_set:
                break
            self._cursor_x = nx
            self._cursor_y = ny
            if (self._cursor_x, self._cursor_y) in self._targets:
                break

    def _move_mirror(self, dx, dy):
        self._mirror_x, self._mirror_y = self._clamp(
            self._mirror_x + dx, self._mirror_y + dy
        )

    def _check_win(self) -> bool:
        return len(self._targets) == 0

    def _step_l1(self, dx, dy) -> bool:
        self._move_cursor(dx, dy)
        pos = (self._cursor_x, self._cursor_y)

        if pos in self._targets:
            self._targets.remove(pos)
            self._trail.discard(pos)
        elif pos in self._trail:
            self._lives -= 1
            if self._lives <= 0:
                self.lose()
                return True
            self._flash_sprite.set_visible(True)
            self._flash_active = True
            self._restore_snapshot()
            self._hud.reset()
            return True
        else:
            self._trail.add(pos)

        return False

    def _step_l2(self, dx, dy):
        self._slide_cursor(dx, dy)
        pos = (self._cursor_x, self._cursor_y)
        if pos in self._targets:
            self._targets.remove(pos)

    def _step_l3(self, dx, dy):
        self._move_cursor(dx, dy)
        pos = (self._cursor_x, self._cursor_y)
        if pos in self._portals:
            dest = self._portals[pos]
            self._cursor_x, self._cursor_y = dest
            pos = dest
        if pos in self._targets:
            self._targets.remove(pos)

    def _step_l4(self, dx, dy) -> bool:
        nx, ny = self._clamp(self._cursor_x + dx, self._cursor_y + dy)
        if (nx, ny) in self._walls_l4:
            self._lives -= 1
            if self._lives <= 0:
                self.lose()
                return True
            self._flash_sprite.set_visible(True)
            self._flash_active = True
            self._restore_snapshot()
            self._hud.reset()
            return True
        else:
            self._cursor_x, self._cursor_y = nx, ny

        pos = (self._cursor_x, self._cursor_y)
        if pos in self._portals:
            dest = self._portals[pos]
            self._cursor_x, self._cursor_y = dest

        diff_x = self._cursor_x - self._mirror_x
        diff_y = self._cursor_y - self._mirror_y
        if abs(diff_x) >= abs(diff_y):
            mdx = 1 if diff_x > 0 else -1
            mdy = 0
        else:
            mdx = 0
            mdy = 1 if diff_y > 0 else -1
        mnx, mny = self._clamp(self._mirror_x + mdx, self._mirror_y + mdy)
        if (mnx, mny) not in self._walls_l4:
            self._mirror_x, self._mirror_y = mnx, mny

        if (self._mirror_x, self._mirror_y) == (self._cursor_x, self._cursor_y):
            self._lives -= 1
            if self._lives <= 0:
                self.lose()
                return True
            self._flash_sprite.set_visible(True)
            self._flash_active = True
            self._restore_snapshot()
            self._hud.reset()
            return True

        pos_main = (self._cursor_x, self._cursor_y)
        if pos_main in self._targets:
            self._targets.remove(pos_main)

        return False

    def _after_move(self) -> bool:
        if self._check_win():
            self._lives = self.MAX_LIVES
            self._render()
            self.next_level()
            self.complete_action()
            return True

        if not self._hud.tick():
            self._lives -= 1
            if self._lives <= 0:
                self.lose()
                self.complete_action()
                return True

            self._flash_sprite.set_visible(True)
            self._flash_active = True
            self._restore_snapshot()
            self._hud.reset()
            self._render()
            self.complete_action()
            return True

        return False

    def step(self) -> None:
        if self._flash_active:
            self._flash_sprite.set_visible(False)
            self._flash_active = False

        a = self.action.id
        dx, dy = 0, 0

        if a == GameAction.ACTION7:
            current_remaining = self._hud.remaining
            self._restore_from_undo()
            self._hud.remaining = current_remaining
            if not self._hud.tick():
                self._lives -= 1
                if self._lives <= 0:
                    self.lose()
                    self.complete_action()
                    return
                self._flash_sprite.set_visible(True)
                self._flash_active = True
                self._restore_snapshot()
                self._hud.reset()
                self.complete_action()
                return
            self._render()
            self.complete_action()
            return

        if a == GameAction.ACTION1:
            dy = -1
        elif a == GameAction.ACTION2:
            dy = 1
        elif a == GameAction.ACTION3:
            dx = -1
        elif a == GameAction.ACTION4:
            dx = 1

        if dx != 0 or dy != 0:
            self._save_state()
            idx = self.level_index
            if idx == 0:
                if self._step_l1(dx, dy):
                    self._render()
                    self.complete_action()
                    return
            elif idx == 1:
                self._step_l2(dx, dy)
            elif idx == 2:
                self._step_l3(dx, dy)
            elif idx == 3:
                if self._step_l4(dx, dy):
                    self._render()
                    self.complete_action()
                    return

            if self._after_move():
                return

        self._render()
        self.complete_action()


class PuzzleEnvironment:
    _ACTION_MAP: Dict[str, GameAction] = {
        "reset": GameAction.RESET,
        "up": GameAction.ACTION1,
        "down": GameAction.ACTION2,
        "left": GameAction.ACTION3,
        "right": GameAction.ACTION4,
        "undo": GameAction.ACTION7,
    }

    _VALID_ACTIONS = list(_ACTION_MAP.keys())

    def __init__(self, seed: int = 0) -> None:
        self._engine = Cx01(seed=seed)
        self.TOTAL_LEVELS = len(self._engine._levels)
        self._done = False
        self._game_won = False
        self._game_over = False
        self._total_turns = 0
        self._last_action_was_reset = False

    @staticmethod
    def _encode_png(rgb: np.ndarray) -> bytes:
        h, w = rgb.shape[0], rgb.shape[1]
        raw = bytearray()
        for y in range(h):
            raw.append(0)
            raw.extend(rgb[y].tobytes())
        compressed = zlib.compress(bytes(raw))

        def _chunk(chunk_type: bytes, data: bytes) -> bytes:
            c = chunk_type + data
            crc = struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
            return struct.pack(">I", len(data)) + c + crc

        sig = b"\x89PNG\r\n\x1a\n"
        ihdr_data = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
        return (
            sig
            + _chunk(b"IHDR", ihdr_data)
            + _chunk(b"IDAT", compressed)
            + _chunk(b"IEND", b"")
        )

    def _build_text_obs(self) -> str:
        g = self._engine
        total_levels = self.TOTAL_LEVELS
        level_num = g.level_index + 1
        idx = g.level_index

        lines = [f"Level {level_num}/{total_levels}"]

        lines.append(f"Cursor: ({g._cursor_x}, {g._cursor_y})")
        lines.append(f"Targets remaining: {len(g._targets)}")

        if idx == 0:
            lines.append(f"Trail cells: {len(g._trail)}")
        elif idx == 1:
            lines.append(f"Stoppers: {len(g._stoppers)}")
        elif idx == 2:
            lines.append(f"Portals: {len(g._portals) // 2}")
        elif idx == 3:
            lines.append(f"Mirror: ({g._mirror_x}, {g._mirror_y})")
            lines.append(f"Walls: {len(g._walls_l4)}")
            lines.append(f"Portals: {len(g._portals) // 2}")

        if hasattr(g, "_hud"):
            lines.append(f"Moves: {g._hud.remaining}/{g._hud.max_moves}")
        lines.append(f"Lives: {g._lives}/{g.MAX_LIVES}")
        return "\n".join(lines)

    def _build_image_bytes(self) -> bytes | None:
        try:
            g = self._engine
            index_grid = g.camera.render(g.current_level.get_sprites())
            if index_grid is not None:
                rgb = np.zeros((64, 64, 3), dtype=np.uint8)
                for idx in range(len(ARC_PALETTE)):
                    mask = index_grid == idx
                    rgb[mask] = ARC_PALETTE[idx]
                return self._encode_png(rgb)
        except Exception:
            pass
        return None

    def _build_game_state(self, done: bool = False) -> GameState:
        g = self._engine
        return GameState(
            text_observation=self._build_text_obs(),
            image_observation=self._build_image_bytes(),
            valid_actions=self.get_actions() if not done else None,
            turn=self._total_turns,
            metadata={
                "total_levels": self.TOTAL_LEVELS,
                "level_index": g.level_index,
                "levels_completed": g.level_index,
                "game_over": self._game_over,
                "done": done,
                "info": {},
            },
        )

    def reset(self) -> GameState:
        if self._game_won or self._last_action_was_reset:
            self._engine.perform_action(ActionInput(id=GameAction.RESET, data={}))
            self._engine.perform_action(ActionInput(id=GameAction.RESET, data={}))
        else:
            self._engine.perform_action(ActionInput(id=GameAction.RESET, data={}))
        self._done = False
        self._game_won = False
        self._game_over = False
        self._total_turns = 0
        self._last_action_was_reset = True
        return self._build_game_state()

    def get_actions(self) -> list[str]:
        if self._done:
            return ["reset"]
        return list(self._VALID_ACTIONS)

    def is_done(self) -> bool:
        return self._done

    def step(self, action: str) -> StepResult:
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
                state=self._build_game_state(done=True),
                reward=0.0,
                done=True,
                info={},
            )

        self._last_action_was_reset = False

        if action not in self._ACTION_MAP:
            return StepResult(
                state=self._build_game_state(done=False),
                reward=0.0,
                done=False,
                info={"error": "invalid_action"},
            )

        lives_before = self._engine._lives
        level_before = self._engine.level_index

        game_action = self._ACTION_MAP[action]
        frame = self._engine.perform_action(
            ActionInput(id=game_action, data={}), raw=True
        )
        self._total_turns += 1

        info: dict = {"action": action}

        level_reward_step = 1.0 / self.TOTAL_LEVELS

        reward = 0.0
        done = False

        state_name = frame.state.name if frame and frame.state else ""
        game_won = state_name == "WIN"
        game_over = state_name == "GAME_OVER"

        if game_over or self._engine._lives < lives_before:
            if game_over:
                reward = 0.0
                info["reason"] = "death"
                done = True
                self._game_over = True
                self._game_won = False
            else:
                reward = 0.0
                info["reason"] = "life_lost"
        elif game_won:
            reward = level_reward_step
            info["reason"] = "game_complete"
            info["level_index"] = self._engine.level_index
            info["total_levels"] = self.TOTAL_LEVELS
            done = True
            self._game_won = True
            self._game_over = False
        elif self._engine.level_index != level_before:
            reward = level_reward_step
            info["reason"] = "level_complete"

        self._done = done

        return StepResult(
            state=self._build_game_state(done=done),
            reward=reward,
            done=done,
            info=info,
        )

    def render(self, mode: str = "rgb_array"):
        if mode != "rgb_array":
            return None
        g = self._engine
        index_grid = g.camera.render(g.current_level.get_sprites())
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
