import random
import struct
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from arcengine import (
    ARCBaseGame,
    ActionInput,
    Camera,
    GameAction,
    Level,
    RenderableUserDisplay,
    Sprite,
)


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


ACTION_MAP = {
    "up": GameAction.ACTION1,
    "down": GameAction.ACTION2,
    "left": GameAction.ACTION3,
    "right": GameAction.ACTION4,
    "select": GameAction.ACTION5,
    "undo": GameAction.ACTION7,
    "reset": GameAction.RESET,
}

C_POLE_LEFT = 2
C_POLE_MID = 3
C_POLE_RIGHT = 5
C_SELECTED = 7
C_CURSOR = 5
C_FLASH = 11
C_CURSOR_BORDER = 11
C_SELECTED_BORDER = 12

BACKGROUND_COLOR = 0
PADDING_COLOR = 0

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


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    c = chunk_type + data
    crc = struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
    return struct.pack(">I", len(data)) + c + crc


def _encode_rgb_png(rgb: np.ndarray) -> bytes:
    h, w = rgb.shape[0], rgb.shape[1]
    raw = bytearray()
    for y in range(h):
        raw.append(0)
        raw.extend(rgb[y].tobytes())
    compressed = zlib.compress(bytes(raw))
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return sig + _png_chunk(b"IHDR", ihdr_data) + _png_chunk(b"IDAT", compressed) + _png_chunk(b"IEND", b"")


BLOCK_TO_COLOR = {
    1: 9,
    2: 14,
    3: 8,
    4: 11,
    5: 6,
    6: 12,
    7: 10,
    8: 15,
    9: 13,
}

BLOCK_COLOR_NAMES = {
    1: "Blue",
    2: "Green",
    3: "Red",
    4: "Yellow",
    5: "Magenta",
    6: "Orange",
    7: "Blue Light",
    8: "Purple",
    9: "Maroon",
}


def _get_alpha_weights(num_blocks: int) -> dict[int, int]:
    blocks = [(val, BLOCK_COLOR_NAMES[val]) for val in range(1, num_blocks + 1)]
    blocks.sort(key=lambda x: x[1])
    return {val: weight + 1 for weight, (val, _name) in enumerate(blocks)}


sprites = {
    "pole_left": Sprite(
        pixels=[[C_POLE_LEFT]],
        name="pole_left",
        visible=True,
        collidable=False,
        tags=["pole"],
        layer=1,
    ),
    "pole_mid": Sprite(
        pixels=[[C_POLE_MID]],
        name="pole_mid",
        visible=True,
        collidable=False,
        tags=["pole"],
        layer=1,
    ),
    "pole_right": Sprite(
        pixels=[[C_POLE_RIGHT]],
        name="pole_right",
        visible=True,
        collidable=False,
        tags=["pole", "target"],
        layer=1,
    ),
    "base_left": Sprite(
        pixels=[[C_POLE_LEFT, C_POLE_LEFT, C_POLE_LEFT, C_POLE_LEFT, C_POLE_LEFT]],
        name="base_left",
        visible=True,
        collidable=False,
        tags=["base"],
        layer=0,
    ),
    "base_mid": Sprite(
        pixels=[[C_POLE_MID, C_POLE_MID, C_POLE_MID, C_POLE_MID, C_POLE_MID]],
        name="base_mid",
        visible=True,
        collidable=False,
        tags=["base"],
        layer=0,
    ),
    "base_right": Sprite(
        pixels=[[C_POLE_RIGHT, C_POLE_RIGHT, C_POLE_RIGHT, C_POLE_RIGHT, C_POLE_RIGHT]],
        name="base_right",
        visible=True,
        collidable=False,
        tags=["base"],
        layer=0,
    ),
    "select_left": Sprite(
        pixels=[
            [C_SELECTED, C_SELECTED, C_SELECTED, C_SELECTED, C_SELECTED],
            [C_SELECTED, C_SELECTED, C_SELECTED, C_SELECTED, C_SELECTED],
        ],
        name="select_left",
        visible=False,
        collidable=False,
        tags=["select"],
        layer=6,
    ),
    "select_mid": Sprite(
        pixels=[
            [C_SELECTED, C_SELECTED, C_SELECTED, C_SELECTED, C_SELECTED],
            [C_SELECTED, C_SELECTED, C_SELECTED, C_SELECTED, C_SELECTED],
        ],
        name="select_mid",
        visible=False,
        collidable=False,
        tags=["select"],
        layer=6,
    ),
    "select_right": Sprite(
        pixels=[
            [C_SELECTED, C_SELECTED, C_SELECTED, C_SELECTED, C_SELECTED],
            [C_SELECTED, C_SELECTED, C_SELECTED, C_SELECTED, C_SELECTED],
        ],
        name="select_right",
        visible=False,
        collidable=False,
        tags=["select"],
        layer=6,
    ),
    "block1": Sprite(
        pixels=[[9, 9, 9], [9, 9, 9]],
        name="block1",
        visible=True,
        collidable=False,
        tags=["block", "val:1", "color:9"],
        layer=3,
    ),
    "block2": Sprite(
        pixels=[[14, 14, 14], [14, 14, 14]],
        name="block2",
        visible=True,
        collidable=False,
        tags=["block", "val:2", "color:14"],
        layer=3,
    ),
    "block3": Sprite(
        pixels=[[8, 8, 8], [8, 8, 8]],
        name="block3",
        visible=True,
        collidable=False,
        tags=["block", "val:3", "color:8"],
        layer=3,
    ),
    "block4": Sprite(
        pixels=[[11, 11, 11], [11, 11, 11]],
        name="block4",
        visible=True,
        collidable=False,
        tags=["block", "val:4", "color:11"],
        layer=3,
    ),
    "block5": Sprite(
        pixels=[[6, 6, 6], [6, 6, 6]],
        name="block5",
        visible=True,
        collidable=False,
        tags=["block", "val:5", "color:6"],
        layer=3,
    ),
    "block6": Sprite(
        pixels=[[12, 12, 12], [12, 12, 12]],
        name="block6",
        visible=True,
        collidable=False,
        tags=["block", "val:6", "color:12"],
        layer=3,
    ),
    "block7": Sprite(
        pixels=[[10, 10, 10], [10, 10, 10]],
        name="block7",
        visible=True,
        collidable=False,
        tags=["block", "val:7", "color:10"],
        layer=3,
    ),
    "block8": Sprite(
        pixels=[[15, 15, 15], [15, 15, 15]],
        name="block8",
        visible=True,
        collidable=False,
        tags=["block", "val:8", "color:15"],
        layer=3,
    ),
    "block9": Sprite(
        pixels=[[13, 13, 13], [13, 13, 13]],
        name="block9",
        visible=True,
        collidable=False,
        tags=["block", "val:9", "color:13"],
        layer=3,
    ),
    "flash": Sprite(
        pixels=[[C_FLASH]],
        name="flash",
        visible=False,
        collidable=False,
        layer=10,
    ),
}

LEVEL_GRIDS = [
    "L M R 3",
    "L M R 4",
    "L M R 5",
    "L M R 6",
    "L M R 7",
]

_LEVEL_MAX_MOVES = [66, 138, 282, 570, 1146]


def _is_sorted_correctly(stack: list[int], num_blocks: int) -> bool:
    expected = list(range(num_blocks, 0, -1))
    return stack == expected


def _generate_scrambled_order(num_blocks: int, rng: random.Random) -> list[int]:
    sorted_order = list(range(1, num_blocks + 1))

    for _ in range(500):
        scrambled = sorted_order[:]
        rng.shuffle(scrambled)
        if not _is_sorted_correctly(scrambled, num_blocks):
            return scrambled

    result = sorted_order[:]
    if len(result) >= 2:
        result[0], result[1] = result[1], result[0]
    return result


def _parse_grid(grid_str: str, data: dict | None = None) -> Level:
    parts = grid_str.split()
    num_blocks = int(parts[3])

    LEFT_X, MID_X, RIGHT_X = 7, 15, 23
    POLE_HEIGHT = 18
    BASE_Y = 26

    level_sprites: list[Sprite] = []

    for y in range(BASE_Y - POLE_HEIGHT, BASE_Y):
        level_sprites.append(sprites["pole_left"].clone().set_position(LEFT_X, y))
        level_sprites.append(sprites["pole_mid"].clone().set_position(MID_X, y))
        level_sprites.append(sprites["pole_right"].clone().set_position(RIGHT_X, y))

    level_sprites.append(sprites["base_left"].clone().set_position(LEFT_X - 2, BASE_Y))
    level_sprites.append(sprites["base_mid"].clone().set_position(MID_X - 2, BASE_Y))
    level_sprites.append(
        sprites["base_right"].clone().set_position(RIGHT_X - 2, BASE_Y)
    )

    level_sprites.append(sprites["select_left"].clone().set_position(LEFT_X - 2, 4))
    level_sprites.append(sprites["select_mid"].clone().set_position(MID_X - 2, 4))
    level_sprites.append(sprites["select_right"].clone().set_position(RIGHT_X - 2, 4))

    for i in range(1, num_blocks + 1):
        y_pos = BASE_Y - 2 - (i - 1) * 2
        level_sprites.append(
            sprites[f"block{i}"].clone().set_position(LEFT_X - 1, y_pos)
        )

    return Level(sprites=level_sprites, grid_size=(30, 30), data=data or {})


def _build_levels() -> list[Level]:
    return [
        _parse_grid(grid, {"max_moves": mm})
        for grid, mm in zip(LEVEL_GRIDS, _LEVEL_MAX_MOVES)
    ]


_CAMERA_SIZES = [
    (30, 30),
    (30, 30),
    (30, 30),
    (30, 30),
    (30, 30),
]


class MoveDisplay(RenderableUserDisplay):
    BAR_WIDTH = 46
    BAR_X = 4
    BAR_Y = 61

    TARGET_PANEL_X = 58
    TARGET_PANEL_Y = 2
    SWATCH_H = 2
    SWATCH_W = 4

    def __init__(self, game: "Th01") -> None:
        self._game = game
        self.max_moves: int = 0
        self.remaining: int = 0

    def set_limit(self, max_moves: int) -> None:
        self.max_moves = max_moves
        self.remaining = max_moves

    def tick(self) -> bool:
        if self.remaining > 0:
            self.remaining -= 1
        return self.remaining > 0

    def reset(self) -> None:
        self.remaining = self.max_moves

    def _draw_target_pattern(self, frame):
        game = self._game
        if not hasattr(game, "_alpha_weights") or not hasattr(game, "_blocks"):
            return frame

        num_blocks = len(game._blocks)
        if num_blocks == 0:
            return frame

        weight_to_val = {w: v for v, w in game._alpha_weights.items()}
        sorted_vals = [weight_to_val[w] for w in range(1, num_blocks + 1)]

        px = self.TARGET_PANEL_X
        py = self.TARGET_PANEL_Y

        for i, val in enumerate(sorted_vals):
            arc_color = BLOCK_TO_COLOR.get(val, 3)
            y0 = py + i * self.SWATCH_H
            y1 = y0 + self.SWATCH_H
            x0 = px
            x1 = px + self.SWATCH_W
            if y1 <= frame.shape[0] and x1 <= frame.shape[1]:
                frame[y0:y1, x0:x1] = arc_color

        return frame

    def render_interface(self, frame):
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

        frame = self._draw_target_pattern(frame)

        return frame


class Th01(ARCBaseGame):
    MAX_LIVES = 3

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self._rng = random.Random(seed)
        self._gen_counter = 0
        levels = _build_levels()
        self._move_display = MoveDisplay(self)
        self._lives: int = self.MAX_LIVES
        self._flash_active: bool = False
        self._consecutive_resets: int = 0
        self._actions_since_reset: int = 0
        self._last_level_won: bool = False

        camera = Camera(
            0,
            0,
            _CAMERA_SIZES[0][0],
            _CAMERA_SIZES[0][1],
            BACKGROUND_COLOR,
            PADDING_COLOR,
            [self._move_display],
        )

        super().__init__(
            game_id="th01",
            levels=levels,
            camera=camera,
            available_actions=[0, 1, 2, 3, 4, 5, 7],
            seed=seed,
        )

    def handle_reset(self) -> None:
        if self._last_level_won or (self._consecutive_resets >= 1 and self._actions_since_reset == 0):
            self._consecutive_resets = 0
            self._actions_since_reset = 0
            self._last_level_won = False
            self._lives = self.MAX_LIVES
            self.full_reset()
        else:
            self._consecutive_resets += 1
            self._actions_since_reset = 0
            self._lives = self.MAX_LIVES
            self._restore_level()
            self._move_display.reset()

    def _load_move_limit(self) -> None:
        mm = self.current_level.get_data("max_moves")
        if mm:
            self._move_display.set_limit(mm)

    def on_set_level(self, level: Level) -> None:
        self._lives = self.MAX_LIVES
        idx = min(self._current_level_index, len(_CAMERA_SIZES) - 1)
        cw, ch = _CAMERA_SIZES[idx]
        self.camera.width = cw
        self.camera.height = ch

        self._blocks: list[Sprite] = list(
            self.current_level.get_sprites_by_tag("block")
        )
        self._selects: list[Sprite] = sorted(
            self.current_level.get_sprites_by_tag("select"),
            key=lambda s: s.x,
        )

        num_blocks = len(self._blocks)

        self._alpha_weights: dict[int, int] = _get_alpha_weights(num_blocks)

        if not hasattr(self, "_gen_counter"):
            self._gen_counter = 0
        self._gen_counter += 1
        scrambled_order = _generate_scrambled_order(num_blocks, self._rng)

        self._stacks: list[list[int]] = [scrambled_order, [], []]
        self._selected: int | None = None
        self._cursor: int = 0

        tower_x = [7, 15, 23]
        base_y = 26
        for pos, val in enumerate(scrambled_order):
            sprite = self._find_block(val)
            if sprite:
                y = base_y - 2 - pos * 2
                sprite.set_position(tower_x[0] - 1, y)

        self._init_stacks: list[list[int]] = [s[:] for s in self._stacks]
        self._init_block_positions: list[tuple[int, int]] = [
            (b.x, b.y) for b in self._blocks
        ]
        self._undo_stack: list[dict] = []

        self._load_move_limit()

        self._flash_sprite: Sprite = sprites["flash"].clone()
        self.current_level.add_sprite(self._flash_sprite)
        self._flash_sprite.set_visible(False)
        self._flash_active = False

        self._update_select()

    def _restore_level(self) -> None:
        self._stacks = [s[:] for s in self._init_stacks]
        for block, (x, y) in zip(self._blocks, self._init_block_positions):
            block.set_position(x, y)
        self._selected = None
        self._cursor = 0
        self._undo_stack = []
        self._update_select()

    def _save_state(self) -> None:
        self._undo_stack.append({
            "stacks": [s[:] for s in self._stacks],
            "selected": self._selected,
            "cursor": self._cursor,
            "remaining": self._move_display.remaining,
            "actions_since_reset": self._actions_since_reset,
        })

    def _restore_from_undo(self) -> None:
        if not self._undo_stack:
            return
        state = self._undo_stack.pop()
        self._stacks = state["stacks"]
        self._selected = state["selected"]
        self._cursor = state["cursor"]
        self._move_display.remaining = state["remaining"]
        self._actions_since_reset = state["actions_since_reset"]
        self._update_positions()
        self._update_select()

    def _draw_rod_border(
        self, tower_idx: int, color: int, tag: str, layer: int
    ) -> None:
        cx, cy, cw, ch = self._get_cursor_rect(tower_idx)
        if cw <= 0 or ch <= 0:
            return

        pixels = []
        for row in range(ch):
            line = []
            for col in range(cw):
                if row == 0 or row == ch - 1 or col == 0 or col == cw - 1:
                    line.append(color)
                else:
                    line.append(-1)
            pixels.append(line)

        self.current_level.add_sprite(
            Sprite(
                pixels=pixels,
                name=f"{tag}_{tower_idx}",
                x=cx,
                y=cy,
                layer=layer,
                tags=[tag],
                visible=True,
                collidable=False,
            )
        )

    def _draw_cursor(self) -> None:
        for s in self.current_level.get_sprites_by_tag("cursor"):
            self.current_level.remove_sprite(s)
        for s in self.current_level.get_sprites_by_tag("sel_border"):
            self.current_level.remove_sprite(s)

        if self._selected is not None:
            self._draw_rod_border(self._selected, C_SELECTED_BORDER, "sel_border", 9)

        if self._selected is None or self._cursor != self._selected:
            self._draw_rod_border(self._cursor, C_CURSOR_BORDER, "cursor", 10)

    def _get_cursor_rect(self, tower_idx: int):
        tower_x = [7, 15, 23]
        base_y = 26
        pole_height = 18

        rod_x = tower_x[tower_idx]

        box_x = rod_x - 2
        box_y = base_y - pole_height - 1
        box_width = 5
        box_height = pole_height + 3

        return (box_x, box_y, box_width, box_height)

    def _update_select(self) -> None:
        for sel in self._selects:
            sel.visible = False

        self._draw_cursor()

    def _find_block(self, val: int) -> Sprite | None:
        for block in self._blocks:
            for tag in block.tags:
                if tag == f"val:{val}":
                    return block
        return None

    def _update_positions(self) -> None:
        tower_x = [7, 15, 23]
        base_y = 26

        for tower_idx, stack in enumerate(self._stacks):
            rod_x = tower_x[tower_idx] - 1
            for pos, val in enumerate(stack):
                sprite = self._find_block(val)
                if sprite:
                    y = base_y - 2 - pos * 2
                    sprite.set_position(rod_x, y)

    def _can_move(self, from_idx: int, to_idx: int) -> bool:
        if not self._stacks[from_idx]:
            return False
        if not self._stacks[to_idx]:
            return True
        from_w = self._alpha_weights[self._stacks[from_idx][-1]]
        to_w = self._alpha_weights[self._stacks[to_idx][-1]]
        return from_w < to_w

    def _move_block(self, from_idx: int, to_idx: int) -> bool:
        if self._can_move(from_idx, to_idx):
            val = self._stacks[from_idx].pop()
            self._stacks[to_idx].append(val)
            self._update_positions()
            return True
        return False

    def _check_win(self) -> bool:
        num_blocks = len(self._blocks)

        if len(self._stacks[2]) != num_blocks:
            return False

        weights = [self._alpha_weights[v] for v in self._stacks[2]]
        expected = list(range(num_blocks, 0, -1))
        return weights == expected

    def _after_move(self) -> bool:
        if self._check_win():
            self._last_level_won = self.is_last_level()
            self.next_level()
            self.complete_action()
            return True
        return False

    def _tick_moves(self) -> bool:
        if not self._move_display.tick():
            self._lives -= 1
            if self._lives <= 0:
                self.lose()
                self.complete_action()
                return True

            self._flash_sprite.set_visible(True)
            self._flash_sprite.set_scale(64)
            self._flash_sprite.set_position(0, 0)
            self._flash_active = True
            self._restore_level()
            self._move_display.reset()
            self.complete_action()
            return True
        return False

    def step(self) -> None:
        if self.action.id == GameAction.RESET:
            self.complete_action()
            return

        if self.action.id == GameAction.ACTION7:
            self._restore_from_undo()
            self._actions_since_reset += 1
            if not self._move_display.tick():
                self._lives -= 1
                if self._lives <= 0:
                    self.lose()
                else:
                    self._flash_sprite.set_visible(True)
                    self._flash_sprite.set_scale(64)
                    self._flash_sprite.set_position(0, 0)
                    self._flash_active = True
                    self._restore_level()
                    self._move_display.reset()
            self.complete_action()
            return

        if self._flash_active:
            self._flash_sprite.set_visible(False)
            self._flash_active = False
            self._actions_since_reset += 1
            self.complete_action()
            return

        if self._check_win():
            self._last_level_won = self.is_last_level()
            self.next_level()
            self._actions_since_reset += 1
            self.complete_action()
            return

        if self._tick_moves():
            self._actions_since_reset += 1
            return

        self._actions_since_reset += 1

        if self.action.id in [
            GameAction.ACTION1,
            GameAction.ACTION2,
            GameAction.ACTION3,
            GameAction.ACTION4,
        ]:
            if self.action.id in (GameAction.ACTION3, GameAction.ACTION1):
                self._cursor = (self._cursor - 1) % 3
            elif self.action.id in (GameAction.ACTION4, GameAction.ACTION2):
                self._cursor = (self._cursor + 1) % 3

            self._update_select()

        elif self.action.id == GameAction.ACTION5:
            if self._selected is None:
                if self._stacks[self._cursor]:
                    self._selected = self._cursor
                    self._update_select()
            else:
                if self._cursor == self._selected:
                    self._selected = None
                    self._update_select()
                else:
                    self._save_state()
                    if self._move_block(self._selected, self._cursor):
                        self._selected = None
                        self._update_select()

                        if self._after_move():
                            return

        self.complete_action()


class PuzzleEnvironment:

    def __init__(self, seed: int = 0) -> None:
        self._engine = Th01(seed=seed)
        self.TOTAL_LEVELS = len(self._engine._levels)
        self._done = False
        self._total_turns = 0
        self._last_action_was_reset = False

    def _build_text_observation(self) -> str:
        g = self._engine
        tower_names = ["Left", "Middle", "Right"]
        num_blocks = len(g._blocks) if hasattr(g, "_blocks") else 0
        total_levels = len(LEVEL_GRIDS)
        level_num = g._current_level_index + 1

        if num_blocks > 0 and hasattr(g, "_alpha_weights"):
            weight_to_val = {w: v for v, w in g._alpha_weights.items()}
            target_order = [
                BLOCK_COLOR_NAMES.get(weight_to_val[w], "?")
                for w in range(1, num_blocks + 1)
            ]
            lines = [
                f"Level {level_num}/{total_levels} | {num_blocks} blocks",
                f"  Target (right tower, bottom-to-top): {target_order}",
            ]
        else:
            lines = [f"Level {level_num}/{total_levels} | {num_blocks} blocks"]

        for i, name in enumerate(tower_names):
            stack = g._stacks[i] if hasattr(g, "_stacks") else []
            color_names = [BLOCK_COLOR_NAMES.get(v, f"?{v}") for v in stack]
            marker = ""
            if hasattr(g, "_cursor") and g._cursor == i:
                marker += " [cursor]"
            if hasattr(g, "_selected") and g._selected == i:
                marker += " [selected]"
            lines.append(f"  {name}: {color_names}{marker}")

        if hasattr(g, "_move_display"):
            lines.append(
                f"Moves: {g._move_display.remaining}/{g._move_display.max_moves}"
            )
        lines.append(f"Lives: {g._lives}/{g.MAX_LIVES}")
        return "\n".join(lines)

    def _build_image_observation(self) -> bytes | None:
        try:
            g = self._engine
            index_grid = g.camera.render(g.current_level.get_sprites())
            if index_grid is not None:
                rgb = np.zeros((64, 64, 3), dtype=np.uint8)
                for idx, color in enumerate(ARC_PALETTE):
                    mask = index_grid == idx
                    rgb[mask] = color
                return _encode_rgb_png(rgb)
        except Exception:
            pass
        return None

    def _build_state(self) -> GameState:
        g = self._engine
        return GameState(
            text_observation=self._build_text_observation(),
            image_observation=self._build_image_observation(),
            valid_actions=self.get_actions() if not self._done else None,
            turn=self._total_turns,
            metadata={
                "total_levels": self.TOTAL_LEVELS,
                "level": g._current_level_index,
                "lives": g._lives,
                "moves_remaining": g._move_display.remaining,
                "moves_max": g._move_display.max_moves,
                "stacks": [s[:] for s in g._stacks],
                "selected": g._selected,
                "cursor": g._cursor,
                "levels_completed": getattr(self._engine, "_score", 0),
                "level_index": g._current_level_index,
                "game_over": getattr(getattr(self._engine, "_state", None), "name", "") == "GAME_OVER",
            },
        )

    def reset(self) -> GameState:
        self._engine.perform_action(ActionInput(id=GameAction.RESET, data={}))
        self._done = False
        self._total_turns = 0
        return self._build_state()

    def get_actions(self) -> list[str]:
        if self._done:
            return ["reset"]
        return list(ACTION_MAP.keys())

    def is_done(self) -> bool:
        return self._done

    def step(self, action: str) -> StepResult:
        if self._done:
            return StepResult(
                state=self._build_state(),
                reward=0.0,
                done=True,
                info={},
            )

        if action == "reset":
            state = self.reset()
            return StepResult(
                state=state,
                reward=0.0,
                done=False,
                info={"reason": "reset"},
            )

        action_id = ACTION_MAP.get(action)
        if action_id is None:
            return StepResult(
                state=self._build_state(),
                reward=0.0,
                done=False,
                info={"error": f"unknown action: {action}"},
            )

        lives_before = self._engine._lives
        level_before = self._engine._current_level_index

        frame = self._engine.perform_action(ActionInput(id=action_id, data={}), raw=True)
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
            else:
                reward = 0.0
                info["reason"] = "life_lost"
        elif game_won:
            reward = level_reward_step
            info["reason"] = "game_complete"
            info["level_index"] = self._engine._current_level_index
            info["total_levels"] = self.TOTAL_LEVELS
            done = True
        elif self._engine._current_level_index != level_before:
            reward = level_reward_step
            info["reason"] = "level_complete"

        self._done = done

        return StepResult(
            state=self._build_state(),
            reward=reward,
            done=done,
            info=info,
        )

    def render(self, mode: str = "rgb_array"):
        if mode != "rgb_array":
            raise ValueError(f"Unsupported render mode: {mode}")
        g = self._engine
        index_grid = g.camera.render(g.current_level.get_sprites())
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        for idx, color in enumerate(ARC_PALETTE):
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
