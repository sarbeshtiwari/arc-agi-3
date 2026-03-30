import io
import math
import random
import struct
import zlib
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
    Sprite,
)
from arcengine import (
    GameState as EngineGameState,
)
from gymnasium import spaces


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


GRID_SIZE = (64, 64)
CAM_W, CAM_H = 64, 64

BG_COLOR = 5
PAD_COLOR = 5

BLUE = 9
RED = 8
GREEN = 14
YELLOW = 11

TOP_ORDER = [BLUE, GREEN, RED, YELLOW]

CURSOR_COLOR = 0
HAND_BORDER_COLOR = 12
EMPTY_COLOR = 4
BAR_FULL_COLOR = 6
BAR_EMPTY_COLOR = 4
CENTER_DOT_COLOR = 3
RING_GUIDE_COLOR = 4
LOCK_COLOR = 13
TARGET_BOX_COLOR = 3
LIFE_FULL_COLOR = 8
LIFE_EMPTY_COLOR = 4

MAX_LIVES = 3

BS = 4

RING_CX, RING_CY = 32, 18
HAND_Y = 39
BAR_Y = 47
BAR_H = 2
BAR_W = 52
LIVES_Y = 51
TARGET_Y = 55

RING_RADII = {4: 9, 6: 10, 8: 12, 10: 14, 12: 16}

LEVEL_DEFS = [
    {
        "colors": [BLUE, BLUE, GREEN, GREEN, YELLOW, YELLOW],
        "ring_size": 6,
        "max_moves": 40,
        "scramble": 6,
        "num_locked": 2,
    },
    {
        "colors": [BLUE, BLUE, GREEN, GREEN, RED, RED, YELLOW, YELLOW],
        "ring_size": 8,
        "max_moves": 50,
        "scramble": 10,
        "num_locked": 3,
    },
    {
        "colors": [BLUE, BLUE, RED, RED, RED, RED, YELLOW, YELLOW],
        "ring_size": 8,
        "max_moves": 45,
        "scramble": 14,
        "num_locked": 4,
    },
    {
        "colors": [BLUE, BLUE, GREEN, GREEN, GREEN, GREEN, RED, RED, YELLOW, YELLOW],
        "ring_size": 10,
        "max_moves": 60,
        "scramble": 18,
        "num_locked": 4,
    },
    {
        "colors": [BLUE, BLUE, GREEN, GREEN, GREEN, GREEN, RED, RED, YELLOW, YELLOW],
        "ring_size": 10,
        "max_moves": 55,
        "scramble": 24,
        "num_locked": 5,
    },
]


def _build_target_ring(ldef):
    colors = ldef["colors"][:]
    colors.sort(key=lambda c: TOP_ORDER.index(c))
    return colors


def _is_sorted(ring):
    if any(slot is None for slot in ring):
        return False

    n = len(ring)
    if n < 2 or n % 2 != 0:
        return False

    for i in range(n - 1):
        idx_a = TOP_ORDER.index(ring[i])
        idx_b = TOP_ORDER.index(ring[i + 1])
        if idx_a > idx_b:
            return False

    return True


def _generate_puzzle_and_locks(rng, ldef):
    target = _build_target_ring(ldef)
    n = ldef["ring_size"]
    num_locked = ldef.get("num_locked", 0)

    if num_locked <= 0 or num_locked > n - 2:
        for _ in range(500):
            scrambled = target[:]
            rng.shuffle(scrambled)
            if not _is_sorted(scrambled):
                return scrambled, set()
        result = target[:]
        for i in range(n):
            for j in range(i + 1, n):
                if result[i] != result[j]:
                    result[i], result[j] = result[j], result[i]
                    if not _is_sorted(result):
                        return result, set()
                    result[i], result[j] = result[j], result[i]
        result[0], result[1] = result[1], result[0]
        return result, set()

    for _attempt in range(200):
        positions = list(range(n))
        rng.shuffle(positions)
        locked = set(positions[:num_locked])

        valid_offsets = []

        for rot in range(n):
            r = target[:]

            if rot > 0:
                r = r[rot:] + r[:rot]

            unlocked_positions = [p for p in range(n) if p not in locked]
            unlocked_colors = [r[p] for p in unlocked_positions]

            if len(unlocked_positions) >= 2:
                valid_offsets.append((rot, r, unlocked_positions, unlocked_colors))

        if not valid_offsets:
            continue

        rot, base_ring, unlocked_pos, unlocked_cols = rng.choice(valid_offsets)

        def _force_scramble(base, upos, ucols, rng_local):
            shuffled = ucols[:]
            rng_local.shuffle(shuffled)
            candidate = base[:]
            for p, c in zip(upos, shuffled):
                candidate[p] = c
            if not _is_sorted(candidate):
                return candidate

            sorted_upos = sorted(upos)
            for i in range(len(sorted_upos)):
                for j in range(i + 1, len(sorted_upos)):
                    pi, pj = sorted_upos[i], sorted_upos[j]
                    trial = candidate[:]
                    trial[pi], trial[pj] = trial[pj], trial[pi]
                    if not _is_sorted(trial):
                        return trial

            rev = ucols[::-1]
            candidate2 = base[:]
            for p, c in zip(upos, rev):
                candidate2[p] = c
            if not _is_sorted(candidate2):
                return candidate2

            return None

        for _ in range(50):
            result = _force_scramble(base_ring, unlocked_pos, unlocked_cols, rng)
            if result is not None:
                return result, locked

    for _ in range(500):
        scrambled = target[:]
        rng.shuffle(scrambled)
        if not _is_sorted(scrambled):
            return scrambled, set()
    result = target[:]
    for i in range(n):
        for j in range(i + 1, n):
            if result[i] != result[j]:
                result[i], result[j] = result[j], result[i]
                if not _is_sorted(result):
                    return result, set()
                result[i], result[j] = result[j], result[i]
    result[0], result[-1] = result[-1], result[0]
    return result, set()


def _calc_ring_positions(ring_size):
    radius = RING_RADII.get(ring_size, 12)
    positions = []
    for i in range(ring_size):
        angle = 2.0 * math.pi * i / ring_size
        cx = RING_CX + radius * math.sin(angle)
        cy = RING_CY - radius * math.cos(angle)
        bx = int(round(cx)) - BS // 2
        by = int(round(cy)) - BS // 2
        positions.append((bx, by))
    return positions


class Rr85(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)
        self._seed = seed
        self._gen_counter = 0

        self._ring = []
        self._locked = set()
        self._hand = None
        self._cursor_pos = 0
        self._moves_used = 0
        self._max_moves = 45
        self._ldef = None
        self._history = []
        self._ring_positions = []
        self._lives = MAX_LIVES

        camera = Camera(0, 0, CAM_W, CAM_H, BG_COLOR, PAD_COLOR, [])

        levels = [
            Level(sprites=[], grid_size=GRID_SIZE, name=f"Level {i + 1}")
            for i in range(len(LEVEL_DEFS))
        ]

        super().__init__(
            game_id="rr85",
            levels=levels,
            camera=camera,
            available_actions=[0, 1, 2, 3, 4, 5, 6, 7],
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        idx = self.level_index
        ldef = LEVEL_DEFS[idx]
        self._ldef = ldef

        self._max_moves = ldef["max_moves"]
        self._moves_used = 0
        self._hand = None
        self._cursor_pos = 0
        self._history = []
        self._lives = MAX_LIVES

        self._ring_positions = _calc_ring_positions(ldef["ring_size"])

        self._gen_counter += 1
        fresh_seed = hash((self._seed, idx, self._gen_counter)) & 0xFFFFFFFF
        gen_rng = random.Random(fresh_seed)
        self._ring, self._locked = _generate_puzzle_and_locks(gen_rng, ldef)

        self._rebuild_sprites(level)

    def _restart_puzzle(self, level: Level) -> None:
        ldef = self._ldef
        self._max_moves = ldef["max_moves"]
        self._moves_used = 0
        self._hand = None
        self._cursor_pos = 0
        self._history = []

        self._ring_positions = _calc_ring_positions(ldef["ring_size"])

        self._gen_counter += 1
        fresh_seed = (
            hash((self._seed, self.level_index, self._gen_counter)) & 0xFFFFFFFF
        )
        gen_rng = random.Random(fresh_seed)
        self._ring, self._locked = _generate_puzzle_and_locks(gen_rng, ldef)

        self._rebuild_sprites(level)

    def _rebuild_sprites(self, level: Level) -> None:
        level.remove_all_sprites()
        self._draw_ring_guide(level)
        self._draw_center_dot(level)
        self._draw_ring_blocks(level)
        self._draw_hand(level)
        self._draw_lives(level)
        self._draw_move_bar(level)
        self._draw_target_sequence(level)
        self._draw_cursor(level)

    def _draw_center_dot(self, level: Level) -> None:
        pixels = [[CENTER_DOT_COLOR] * 2 for _ in range(2)]
        level.add_sprite(
            Sprite(
                pixels=pixels,
                name="center_dot",
                x=RING_CX - 1,
                y=RING_CY - 1,
                layer=0,
                tags=["decor"],
            )
        )

    def _draw_ring_guide(self, level: Level) -> None:
        if self._ldef is None:
            return
        ring_size = self._ldef["ring_size"]
        radius = RING_RADII.get(ring_size, 12)

        num_dots = ring_size * 4
        for i in range(num_dots):
            angle = 2.0 * math.pi * i / num_dots
            dx = RING_CX + radius * math.sin(angle)
            dy = RING_CY - radius * math.cos(angle)
            px = int(round(dx))
            py = int(round(dy))
            if 0 <= px < CAM_W and 0 <= py < CAM_H:
                dot_pixels = [[RING_GUIDE_COLOR]]
                level.add_sprite(
                    Sprite(
                        pixels=dot_pixels,
                        name=f"guide_{i}",
                        x=px,
                        y=py,
                        layer=0,
                        tags=["decor"],
                    )
                )

    def _draw_ring_blocks(self, level: Level) -> None:
        border_size = BS + 2

        for i, (bx, by) in enumerate(self._ring_positions):
            color = self._ring[i] if i < len(self._ring) else None
            is_locked = i in self._locked

            if is_locked and color is not None:
                lock_pixels = [[LOCK_COLOR] * border_size for _ in range(border_size)]
                level.add_sprite(
                    Sprite(
                        pixels=lock_pixels,
                        name=f"lock_{i}",
                        x=bx - 1,
                        y=by - 1,
                        layer=1,
                        tags=["ring", "lock"],
                    )
                )

            if color is not None:
                pixels = [[color] * BS for _ in range(BS)]
            else:
                pixels = [[EMPTY_COLOR] * BS for _ in range(BS)]

            level.add_sprite(
                Sprite(
                    pixels=pixels,
                    name=f"ring_{i}",
                    x=bx,
                    y=by,
                    layer=2,
                    tags=["ring"],
                )
            )

    def _draw_hand(self, level: Level) -> None:
        border_size = BS + 2
        hx = (CAM_W - border_size) // 2
        hy = HAND_Y

        border_pixels = [[HAND_BORDER_COLOR] * border_size for _ in range(border_size)]
        level.add_sprite(
            Sprite(
                pixels=border_pixels,
                name="hand_border",
                x=hx,
                y=hy,
                layer=1,
                tags=["hand"],
            )
        )

        inner_color = self._hand if self._hand is not None else EMPTY_COLOR
        inner_pixels = [[inner_color] * BS for _ in range(BS)]
        level.add_sprite(
            Sprite(
                pixels=inner_pixels,
                name="hand_content",
                x=hx + 1,
                y=hy + 1,
                layer=2,
                tags=["hand"],
            )
        )

    def _draw_lives(self, level: Level) -> None:
        dot_size = 2
        spacing = 2
        total_w = MAX_LIVES * dot_size + (MAX_LIVES - 1) * spacing
        start_x = (CAM_W - total_w) // 2
        ly = LIVES_Y

        for i in range(MAX_LIVES):
            color = LIFE_FULL_COLOR if i < self._lives else LIFE_EMPTY_COLOR
            lx = start_x + i * (dot_size + spacing)
            pixels = [[color] * dot_size for _ in range(dot_size)]
            level.add_sprite(
                Sprite(
                    pixels=pixels,
                    name=f"life_{i}",
                    x=lx,
                    y=ly,
                    layer=1,
                    tags=["lives"],
                )
            )

    def _draw_move_bar(self, level: Level) -> None:
        bar_x = (CAM_W - BAR_W) // 2

        remaining = max(0, self._max_moves - self._moves_used)
        filled_w = max(0, (remaining * BAR_W) // self._max_moves)
        empty_w = BAR_W - filled_w

        if filled_w > 0:
            filled_pixels = [[BAR_FULL_COLOR] * filled_w for _ in range(BAR_H)]
            level.add_sprite(
                Sprite(
                    pixels=filled_pixels,
                    name="bar_filled",
                    x=bar_x,
                    y=BAR_Y,
                    layer=1,
                    tags=["bar"],
                )
            )

        if empty_w > 0:
            empty_pixels = [[BAR_EMPTY_COLOR] * empty_w for _ in range(BAR_H)]
            level.add_sprite(
                Sprite(
                    pixels=empty_pixels,
                    name="bar_empty",
                    x=bar_x + filled_w,
                    y=BAR_Y,
                    layer=1,
                    tags=["bar"],
                )
            )

    def _draw_target_sequence(self, level: Level) -> None:
        if self._ldef is None:
            return

        colors_in_level = sorted(
            set(self._ldef["colors"]), key=lambda c: TOP_ORDER.index(c)
        )

        small_bs = 2
        spacing = 1
        num_colors = len(colors_in_level)

        inner_width = num_colors * small_bs + (num_colors - 1) * spacing
        box_width = inner_width + 2
        box_height = small_bs + 2

        box_x = (CAM_W - box_width) // 2
        box_y = TARGET_Y

        border_pixels = [[TARGET_BOX_COLOR] * box_width for _ in range(box_height)]
        for row in range(1, box_height - 1):
            for col in range(1, box_width - 1):
                border_pixels[row][col] = BG_COLOR

        level.add_sprite(
            Sprite(
                pixels=border_pixels,
                name="target_box",
                x=box_x,
                y=box_y,
                layer=1,
                tags=["target"],
            )
        )

        start_x = box_x + 1
        block_y = box_y + 1

        for i, color in enumerate(colors_in_level):
            bx = start_x + i * (small_bs + spacing)
            block_pixels = [[color] * small_bs for _ in range(small_bs)]
            level.add_sprite(
                Sprite(
                    pixels=block_pixels,
                    name=f"target_{i}",
                    x=bx,
                    y=block_y,
                    layer=2,
                    tags=["target"],
                )
            )

    def _draw_cursor(self, level: Level) -> None:
        for s in level.get_sprites_by_tag("cursor"):
            level.remove_sprite(s)

        if not self._ring_positions:
            return

        bx, by = self._ring_positions[self._cursor_pos]

        is_locked = self._cursor_pos in self._locked
        if is_locked:
            cx, cy = bx - 2, by - 2
            cw, ch = BS + 4, BS + 4
        else:
            cx, cy = bx - 1, by - 1
            cw, ch = BS + 2, BS + 2

        if cx < 0:
            cw += cx
            cx = 0
        if cy < 0:
            ch += cy
            cy = 0
        if cx + cw > CAM_W:
            cw = CAM_W - cx
        if cy + ch > CAM_H:
            ch = CAM_H - cy

        if cw <= 0 or ch <= 0:
            return

        cursor_pixels = []
        for row in range(ch):
            line = []
            for col in range(cw):
                if row == 0 or row == ch - 1 or col == 0 or col == cw - 1:
                    line.append(CURSOR_COLOR)
                else:
                    line.append(-1)
            cursor_pixels.append(line)

        level.add_sprite(
            Sprite(
                pixels=cursor_pixels,
                name="cursor",
                x=cx,
                y=cy,
                layer=6,
                tags=["cursor"],
            )
        )

    def _find_nearest_slot(self, x: int, y: int) -> Optional[int]:
        best_idx = None
        best_dist = float("inf")
        for i, (bx, by) in enumerate(self._ring_positions):
            cx = bx + BS // 2
            cy = by + BS // 2
            d = (x - cx) ** 2 + (y - cy) ** 2
            if d < best_dist:
                best_dist = d
                best_idx = i
        if best_idx is not None and best_dist <= (BS * 3) ** 2:
            return best_idx
        return None

    def _save_state(self):
        self._history.append(
            {
                "ring": self._ring[:],
                "locked": set(self._locked),
                "hand": self._hand,
                "moves": self._moves_used,
            }
        )

    def _restore_state(self):
        if self._history:
            state = self._history.pop()
            self._ring = state["ring"]
            self._locked = state["locked"]
            self._hand = state["hand"]
            self._moves_used += 1

    def _rotate_cw(self) -> bool:
        if len(self._ring) < 2:
            return False
        self._save_state()
        self._ring = [self._ring[-1]] + self._ring[:-1]
        n = len(self._ring)
        self._locked = {(pos + 1) % n for pos in self._locked}
        self._moves_used += 1
        return True

    def _rotate_ccw(self) -> bool:
        if len(self._ring) < 2:
            return False
        self._save_state()
        self._ring = self._ring[1:] + [self._ring[0]]
        n = len(self._ring)
        self._locked = {(pos - 1) % n for pos in self._locked}
        self._moves_used += 1
        return True

    def _activate(self) -> bool:
        pos = self._cursor_pos

        if self._hand is None:
            if self._ring[pos] is not None:
                if pos in self._locked:
                    return False
                self._save_state()
                self._hand = self._ring[pos]
                self._ring[pos] = None
                self._moves_used += 1
                return True
        else:
            if self._ring[pos] is None:
                self._save_state()
                self._ring[pos] = self._hand
                self._hand = None
                self._moves_used += 1
                return True
            else:
                if pos in self._locked:
                    return False
                self._save_state()
                old = self._ring[pos]
                self._ring[pos] = self._hand
                self._hand = old
                self._moves_used += 1
                return True

    def _check_complete(self) -> bool:
        if self._hand is not None:
            return False
        return _is_sorted(self._ring)

    def _draw_game_over_screen(self, level):
        level.remove_all_sprites()

        w, h = CAM_W, CAM_H
        pixels = [[BG_COLOR] * w for _ in range(h)]

        bar_width = 36
        bar_height = 6
        bar_x = (w - bar_width) // 2
        bar_y = (h - bar_height) // 2

        border_thickness = 2
        for y in range(bar_y - border_thickness, bar_y + bar_height + border_thickness):
            for x in range(
                bar_x - border_thickness, bar_x + bar_width + border_thickness
            ):
                if 0 <= x < w and 0 <= y < h:
                    in_bar = (bar_x <= x < bar_x + bar_width) and (
                        bar_y <= y < bar_y + bar_height
                    )
                    if not in_bar:
                        pixels[y][x] = RED

        for y in range(bar_y, bar_y + bar_height):
            for x in range(bar_x, bar_x + bar_width):
                if 0 <= x < w and 0 <= y < h:
                    pixels[y][x] = BAR_EMPTY_COLOR

        level.add_sprite(
            Sprite(
                pixels=pixels,
                name="game_over_screen",
                x=0,
                y=0,
                layer=10,
                tags=["message"],
            )
        )

    def _draw_win_screen(self, level):
        level.remove_all_sprites()

        w, h = CAM_W, CAM_H
        pixels = [[BG_COLOR] * w for _ in range(h)]

        for x in range(w):
            pixels[0][x] = pixels[1][x] = 11
            pixels[h - 1][x] = pixels[h - 2][x] = 11
        for y in range(h):
            pixels[y][0] = pixels[y][1] = 11
            pixels[y][w - 1] = pixels[y][w - 2] = 11

        for i in range(11):
            cx, cy = 10 + i, 26 + i
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        pixels[ny][nx] = 14
        for i in range(21):
            cx, cy = 20 + i, 36 - i
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        pixels[ny][nx] = 14

        for sx, sy in [(6, 6), (41, 6), (6, 41), (41, 41)]:
            pixels[sy][sx] = 11
            for ddx, ddy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = sx + ddx, sy + ddy
                if 0 <= nx < w and 0 <= ny < h:
                    pixels[ny][nx] = 11

        level.add_sprite(
            Sprite(
                pixels=pixels,
                name="win_screen",
                x=0,
                y=0,
                layer=10,
                tags=["message"],
            )
        )

    def step(self) -> None:
        action = self.action
        level = self.current_level
        if self._ldef is None:
            self.complete_action()
            return
        ring_size = self._ldef["ring_size"]

        acted = False

        if action.id == GameAction.ACTION1:
            acted = self._rotate_cw()

        elif action.id == GameAction.ACTION2:
            acted = self._rotate_ccw()

        elif action.id == GameAction.ACTION3:
            self._cursor_pos = (self._cursor_pos - 1) % ring_size
            self._moves_used += 1
            self._rebuild_sprites(level)

        elif action.id == GameAction.ACTION4:
            self._cursor_pos = (self._cursor_pos + 1) % ring_size
            self._moves_used += 1
            self._rebuild_sprites(level)

        elif action.id == GameAction.ACTION5:
            acted = self._activate()

        elif action.id == GameAction.ACTION6:
            cx = action.data.get("x", 0)
            cy = action.data.get("y", 0)
            dist_to_center = (cx - RING_CX) ** 2 + (cy - RING_CY) ** 2
            if dist_to_center <= 9:
                acted = self._rotate_cw()
            else:
                slot = self._find_nearest_slot(cx, cy)
                if slot is not None:
                    self._cursor_pos = slot
                    acted = self._activate()

        elif action.id == GameAction.ACTION7:
            if self._history:
                self._restore_state()
            else:
                self._moves_used += 1

            if self._moves_used >= self._max_moves:
                self._lives -= 1
                if self._lives <= 0:
                    self._rebuild_sprites(level)
                    self.lose()
                else:
                    self._restart_puzzle(level)
                self.complete_action()
                return
            self._rebuild_sprites(level)

        if acted:
            if self._moves_used >= self._max_moves:
                self._lives -= 1
                if self._lives <= 0:
                    self._rebuild_sprites(level)
                    self.lose()
                else:
                    self._restart_puzzle(level)
                self.complete_action()
                return
            self._rebuild_sprites(level)

        if self._check_complete():
            if self.is_last_level():
                self._draw_win_screen(level)
            self.next_level()
            self.complete_action()
            return

        if self._moves_used >= self._max_moves:
            self._lives -= 1
            if self._lives <= 0:
                self._rebuild_sprites(level)
                self.lose()
            else:
                self._restart_puzzle(level)
            self.complete_action()
            return

        self.complete_action()


_COLOR_NAMES: Dict[int, str] = {
    9: "Blue",
    14: "Green",
    8: "Red",
    11: "Yellow",
}


class PuzzleEnvironment:
    _ACTION_MAP: Dict[str, GameAction] = {
        "reset": GameAction.RESET,
        "up": GameAction.ACTION1,
        "down": GameAction.ACTION2,
        "left": GameAction.ACTION3,
        "right": GameAction.ACTION4,
        "select": GameAction.ACTION5,
        "click": GameAction.ACTION6,
        "undo": GameAction.ACTION7,
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

    def __init__(self, seed: int = 0) -> None:
        self._engine: Optional[Rr85] = Rr85(seed=seed)
        self._total_turns = 0
        self._last_action_was_reset = False
        self._game_won = False
        self._total_levels = len(LEVEL_DEFS)

    @property
    def _eng(self) -> "Rr85":
        assert self._engine is not None
        return self._engine

    def _build_text_obs(self) -> str:
        e = self._eng
        ldef = e._ldef
        ring_str = ", ".join(
            _COLOR_NAMES.get(c, "empty") if c is not None else "empty" for c in e._ring
        )
        hand_str = (
            _COLOR_NAMES.get(e._hand, str(e._hand))
            if e._hand is not None
            else "(empty)"
        )

        remaining = max(0, e._max_moves - e._moves_used)
        locked_str = (
            ", ".join(str(i) for i in sorted(e._locked)) if e._locked else "none"
        )

        lines = [
            f"Level:{e.level_index + 1}/{len(LEVEL_DEFS)} Lives:{e._lives}",
            f"Moves:{remaining}/{e._max_moves}",
            f"Ring: [{ring_str}]",
            f"Hand: {hand_str}",
            f"Cursor: {e._cursor_pos}",
            f"Locked: [{locked_str}]",
        ]
        return "\n".join(lines)

    def _build_image_bytes(self) -> Optional[bytes]:
        e = self._eng
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
        buf = io.BytesIO()
        buf.write(b"\x89PNG\r\n\x1a\n")

        def _chunk(ctype: bytes, data: bytes) -> None:
            buf.write(struct.pack(">I", len(data)))
            buf.write(ctype)
            buf.write(data)
            buf.write(struct.pack(">I", zlib.crc32(ctype + data) & 0xFFFFFFFF))

        _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        raw = b""
        for y in range(h):
            raw += b"\x00" + rgb[y].tobytes()
        _chunk(b"IDAT", zlib.compress(raw))
        _chunk(b"IEND", b"")
        return buf.getvalue()

    def _build_game_state(
        self, done: bool = False, info: Optional[Dict] = None
    ) -> GameState:
        e = self._eng
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
                "ring_size": len(e._ring),
                "hand": e._hand,
                "locked_count": len(e._locked),
                "done": done,
                "info": info or {},
                "levels_completed": getattr(self._engine, "_score", 0),
                "game_over": getattr(getattr(self._engine, "_state", None), "name", "") == "GAME_OVER",
            },
        )

    def reset(self) -> GameState:
        e = self._eng
        terminated = getattr(e, "_terminated", False)
        full_restart = self._game_won or terminated or self._last_action_was_reset
        if self._game_won or terminated:
            if hasattr(e, "_terminated"):
                e._terminated = False
        e.perform_action(ActionInput(id=GameAction.RESET))
        self._total_turns = 0
        self._game_won = False
        self._last_action_was_reset = True
        return self._build_game_state()

    def get_actions(self) -> List[str]:
        e = self._eng
        game_over = (
            getattr(e, "_game_over", False) or e._state == EngineGameState.GAME_OVER
        )
        if self._game_won or game_over:
            return ["reset"]
        return ["up", "down", "left", "right", "select", "click", "undo", "reset"]

    def is_done(self) -> bool:
        e = self._eng
        game_over = (
            getattr(e, "_game_over", False) or e._state == EngineGameState.GAME_OVER
        )
        return self._game_won or game_over or getattr(e, "_terminated", False)

    def step(self, action: str) -> StepResult:
        e = self._eng

        if action == "reset":
            state = self.reset()
            return StepResult(
                state=state, reward=0.0, done=False, info={"action": "reset"}
            )

        if action not in self._ACTION_MAP:
            raise ValueError(
                f"Invalid action '{action}'. Must be one of {list(self._ACTION_MAP.keys())}"
            )

        self._last_action_was_reset = False
        self._total_turns += 1

        game_action = self._ACTION_MAP[action]
        info: Dict = {"action": action}

        prev_level = e.level_index
        action_input = ActionInput(id=game_action)
        e.perform_action(action_input)

        game_won = e._state == EngineGameState.WIN
        game_over = e._state == EngineGameState.GAME_OVER
        level_advanced = (not game_over) and (game_won or e.level_index != prev_level)
        done = game_over or game_won

        reward = 0.0
        if level_advanced:
            reward = 1.0 / self._total_levels

        if game_won:
            self._game_won = True
            info["reason"] = "game_complete"
            return StepResult(
                state=self._build_game_state(done=True, info=info),
                reward=reward,
                done=True,
                info=info,
            )

        if game_over:
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
        e = self._eng
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
        "reset",
        "up",
        "down",
        "left",
        "right",
        "select",
        "click",
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
