import random
import struct
import zlib
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from arcengine import (
    ARCBaseGame,
    ActionInput,
    Camera,
    GameAction,
    GameState as EngineGameState,
    Level,
    Sprite,
)

ARC_PALETTE: Dict[int, Tuple[int, int, int]] = {
    0: (0, 0, 0),
    1: (0, 116, 217),
    2: (255, 65, 54),
    3: (46, 204, 64),
    4: (255, 220, 0),
    5: (170, 170, 170),
    6: (240, 18, 190),
    7: (255, 133, 27),
    8: (127, 219, 255),
    9: (135, 12, 37),
    10: (0, 48, 73),
    11: (106, 76, 48),
    12: (255, 182, 193),
    13: (80, 80, 80),
    14: (50, 205, 50),
    15: (128, 0, 128),
}

ACTION_FROM_NAME: Dict[str, GameAction] = {
    "up": GameAction.ACTION1,
    "down": GameAction.ACTION2,
    "left": GameAction.ACTION3,
    "right": GameAction.ACTION4,
    "undo": GameAction.ACTION7,
    "reset": GameAction.RESET,
}

LEVEL_SEEDS = [42, 137, 256, 999, 512]


BG = 0
TARGET_CELL = 1
BORDER = 5
BLOCKER = 4
MOVE_COL = 8
MOVE_SPENT = 4
MATCHED_COL = 14

FLASH_GOOD = 14
FLASH_BAD = 8

LIFE_ON = 8
LIFE_OFF = 4

LIVES_PER_LEVEL = [3, 3, 3, 3, 3]

BLOCK_COLORS = [6, 9, 12, 7, 10, 15, 13, 8]

GRID_W = 16
GRID_H = 16

BORDER_ROW_T = 0
BORDER_ROW_B = 15
BORDER_COL_L = 0
BORDER_COL_R = 15

INNER_X0 = 1
INNER_X1 = 14
INNER_Y0 = 1
INNER_Y1 = 14

LEVEL_DEFS = [
    {
        "title": "Level 1",
        "start": [(8, 8), (10, 8), (8, 10)],
        "target": [(1, 1), (2, 1), (1, 2)],
        "max_moves": 10,
        "blockers": [],
    },
    {
        "title": "Level 2",
        "start": [(10, 3), (12, 3), (10, 6), (12, 6)],
        "target": [(1, 13), (2, 13), (1, 14), (2, 14)],
        "max_moves": 10,
        "blockers": [
            (8, 1),
            (8, 2),
            (8, 3),
            (8, 4),
            (8, 5),
            (8, 6),
            (8, 7),
            (8, 8),
            (8, 9),
            (8, 10),
            (8, 11),
        ],
    },
    {
        "title": "Level 3",
        "start": [(9, 5), (11, 5), (9, 9), (11, 9), (13, 12)],
        "target": [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2)],
        "max_moves": 20,
        "blockers": [
            (12, 2),
            (12, 3),
            (12, 4),
            (12, 5),
            (12, 6),
            (12, 7),
            (5, 7),
            (6, 7),
            (7, 7),
            (8, 7),
            (9, 7),
            (10, 7),
            (11, 7),
            (5, 8),
            (5, 9),
            (5, 10),
            (5, 11),
            (5, 12),
            (5, 13),
        ],
    },
    {
        "title": "Level 4",
        "start": [(3, 3), (5, 3), (3, 5), (5, 8), (3, 10)],
        "target": [(13, 14), (14, 11), (14, 12), (14, 13), (14, 14)],
        "max_moves": 22,
        "blockers": [
            (7, 3),
            (7, 4),
            (7, 5),
            (7, 6),
            (7, 7),
            (7, 8),
            (8, 8),
            (9, 8),
            (10, 8),
            (11, 8),
            (12, 8),
            (13, 8),
            (13, 9),
            (13, 10),
            (13, 11),
            (13, 12),
            (13, 13),
            (5, 4),
            (5, 5),
            (5, 6),
            (5, 7),
        ],
    },
    {
        "title": "Level 5",
        "start": [(3, 10), (5, 10), (3, 12), (5, 7), (3, 14), (7, 12)],
        "target": [(12, 1), (13, 1), (13, 2), (14, 1), (14, 2), (14, 3)],
        "max_moves": 22,
        "blockers": [
            (3, 5),
            (4, 5),
            (5, 5),
            (6, 5),
            (7, 5),
            (8, 5),
            (8, 6),
            (8, 7),
            (8, 8),
            (8, 9),
            (9, 9),
            (10, 9),
            (11, 9),
            (12, 9),
            (13, 9),
        ],
    },
]


def apply_gravity_to(
    positions: List[Tuple[int, int]],
    direction: int,
    blocker_set: Set[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    n = len(positions)
    result: List[Tuple[int, int]] = [(-1, -1)] * n

    occupied: Set[Tuple[int, int]] = set(blocker_set)

    def free_south(x: int, y: int) -> int:
        while y + 1 <= INNER_Y1 and (x, y + 1) not in occupied:
            y += 1
        return y

    def free_north(x: int, y: int) -> int:
        while y - 1 >= INNER_Y0 and (x, y - 1) not in occupied:
            y -= 1
        return y

    def free_east(x: int, y: int) -> int:
        while x + 1 <= INNER_X1 and (x + 1, y) not in occupied:
            x += 1
        return x

    def free_west(x: int, y: int) -> int:
        while x - 1 >= INNER_X0 and (x - 1, y) not in occupied:
            x -= 1
        return x

    if direction == 1:
        order = sorted(range(n), key=lambda i: -positions[i][1])
        for i in order:
            x, y = positions[i]
            ny = free_south(x, y)
            result[i] = (x, ny)
            occupied.add((x, ny))

    elif direction == 0:
        order = sorted(range(n), key=lambda i: positions[i][1])
        for i in order:
            x, y = positions[i]
            ny = free_north(x, y)
            result[i] = (x, ny)
            occupied.add((x, ny))

    elif direction == 2:
        order = sorted(range(n), key=lambda i: -positions[i][0])
        for i in order:
            x, y = positions[i]
            nx = free_east(x, y)
            result[i] = (nx, y)
            occupied.add((nx, y))

    elif direction == 3:
        order = sorted(range(n), key=lambda i: positions[i][0])
        for i in order:
            x, y = positions[i]
            nx = free_west(x, y)
            result[i] = (nx, y)
            occupied.add((nx, y))

    return result


def _check_state(
    positions: List[Tuple[int, int]],
    target_set: Set[Tuple[int, int]],
    blocker_set: Set[Tuple[int, int]],
    moves_left: int,
) -> bool:
    if moves_left <= 0:
        return set(map(tuple, positions)) == target_set

    start_state = frozenset(map(tuple, positions))
    if start_state == target_set:
        return True

    queue = deque()
    queue.append((start_state, 0))
    visited: Set[frozenset] = set()
    visited.add(start_state)

    while queue:
        state, depth = queue.popleft()

        for direction in range(4):
            new_pos = apply_gravity_to(
                [(x, y) for x, y in state], direction, blocker_set
            )
            new_state = frozenset(map(tuple, new_pos))

            if new_state == target_set:
                return True

            if new_state not in visited and depth + 1 < moves_left:
                visited.add(new_state)
                queue.append((new_state, depth + 1))

    return False


def _build_start(
    n_blocks: int,
    target_set: Set[Tuple[int, int]],
    blocker_set: Set[Tuple[int, int]],
    max_moves: int,
    rng: random.Random,
    max_attempts: int = 200,
) -> List[Tuple[int, int]]:
    forbidden = target_set | blocker_set
    candidates = [
        (x, y)
        for x in range(INNER_X0, INNER_X1 + 1)
        for y in range(INNER_Y0, INNER_Y1 + 1)
        if (x, y) not in forbidden
    ]
    for _ in range(max_attempts):
        positions = rng.sample(candidates, n_blocks)
        if _check_state(positions, target_set, blocker_set, max_moves):
            return positions
    return list(target_set)[:n_blocks]


def build_background(
    target_set: Set[Tuple[int, int]],
    blocker_set: Set[Tuple[int, int]],
    moves_left: int,
    max_moves: int,
    lives: int = 3,
    lives_max: int = 3,
    border_color: int = BORDER,
) -> np.ndarray:
    frame = np.full((GRID_H, GRID_W), BG, dtype=np.uint8)

    frame[BORDER_ROW_T, :] = border_color
    frame[BORDER_ROW_B, :] = border_color
    frame[1:-1, BORDER_COL_L] = border_color
    frame[1:-1, BORDER_COL_R] = border_color

    for tx, ty in target_set:
        frame[ty, tx] = TARGET_CELL

    for bx, by in blocker_set:
        frame[by, bx] = BLOCKER

    inner_w = INNER_X1 - INNER_X0 + 1
    dot_spacing = 2
    max_dots = inner_w // dot_spacing
    if max_moves <= max_dots:
        total_w = max_moves * dot_spacing
        start_x = INNER_X0 + (inner_w - total_w) // 2
        for i in range(max_moves):
            col = MOVE_COL if i < moves_left else MOVE_SPENT
            frame[BORDER_ROW_T, start_x + i * dot_spacing] = col
    else:
        filled = round(moves_left / max_moves * max_dots) if max_moves > 0 else 0
        total_w = max_dots * dot_spacing
        start_x = INNER_X0 + (inner_w - total_w) // 2
        for i in range(max_dots):
            col = MOVE_COL if i < filled else MOVE_SPENT
            frame[BORDER_ROW_T, start_x + i * dot_spacing] = col

    lost = lives_max - lives
    for i in range(lives_max):
        frame[BORDER_ROW_B, INNER_X0 + i * dot_spacing] = LIFE_OFF if i < lost else LIFE_ON

    return frame


@dataclass
class GameState:
    text_observation: str = ""
    image_observation: Optional[bytes] = None
    valid_actions: Optional[List[str]] = None
    turn: int = 0
    metadata: Dict = field(default_factory=dict)


@dataclass
class StepResult:
    state: GameState = field(default_factory=GameState)
    reward: float = 0.0
    done: bool = False
    info: Dict = field(default_factory=dict)


class Gs04(ARCBaseGame):

    FLASH_FROM_LEVEL = 2

    def __init__(self, seed: int = 0) -> None:
        camera = Camera(
            background=BG,
            letter_box=BORDER,
            width=GRID_W,
            height=GRID_H,
        )
        levels = [Level(sprites=[], grid_size=(GRID_W, GRID_H)) for _ in range(5)]
        super().__init__(
            game_id="gs04",
            levels=levels,
            camera=camera,
            available_actions = [0, 1, 2, 3, 4, 7]
        )

        self.seed = seed
        self.game_seed = seed
        self._rng = random.Random(self.game_seed)
        self.block_pos: List[Tuple[int, int]] = []
        self.target_set: Set[Tuple[int, int]] = set()
        self.blocker_set: Set[Tuple[int, int]] = set()
        self.gravity: int = 1
        self.moves_left: int = 0
        self.max_moves: int = 0
        self.n_blocks: int = 0

        self.flash_color: Optional[int] = None

        self.lives: int = 3
        self.lives_max: int = 3

        self.ready: bool = False
        self.has_played: bool = False
        self.preserve_lives: bool = False
        self.just_reset: bool = False
        self.undo_stack: deque = deque(maxlen=50)

        self.ready = True

    def on_set_level(self, level: Level) -> None:
        if not getattr(self, "ready", False):
            return

        self.current_level.remove_all_sprites()
        self.undo_stack.clear()

        idx = self._current_level_index
        defn = LEVEL_DEFS[idx]

        self.target_set = set(map(tuple, defn["target"]))
        self.blocker_set = set(map(tuple, defn.get("blockers", [])))
        self.n_blocks = len(defn["start"])
        self.max_moves = defn["max_moves"]
        self.moves_left = defn["max_moves"]
        self.gravity = 1
        self.flash_color = None

        if self.preserve_lives:
            self.preserve_lives = False
        else:
            self.lives_max = LIVES_PER_LEVEL[idx] if idx < len(LIVES_PER_LEVEL) else 3
            self.lives = self.lives_max

        self._rng = random.Random(LEVEL_SEEDS[idx % len(LEVEL_SEEDS)] + self.game_seed)
        self.block_pos = _build_start(
            self.n_blocks, self.target_set, self.blocker_set,
            self.max_moves, self._rng,
        )
        self.ready = True
        self.render()

    def save_undo(self) -> None:
        self.undo_stack.append({
            "block_pos": deepcopy(self.block_pos),
            "gravity": self.gravity,
            "moves_left": self.moves_left,
            "lives": self.lives,
            "flash_color": self.flash_color,
        })

    def restore_undo(self) -> bool:
        if not self.undo_stack:
            return False
        current_moves_left = self.moves_left
        snap = self.undo_stack.pop()
        self.block_pos = snap["block_pos"]
        self.gravity = snap["gravity"]
        self.moves_left = current_moves_left
        self.lives = snap["lives"]
        self.flash_color = snap["flash_color"]
        return True

    def step(self) -> None:
        if not self.ready:
            self.complete_action()
            return

        if self.action and self.action.id == GameAction.RESET:
            self.complete_action()
            return

        self.has_played = True
        self.just_reset = False

        action = self.action

        if action.id == GameAction.ACTION7:
            self.restore_undo()
            self.moves_left = max(0, self.moves_left - 1)
            self.render()
            if self.moves_left <= 0:
                if self.lose_life():
                    self.complete_action()
                    return
            self.complete_action()
            return

        self.save_undo()

        direction_map = {
            GameAction.ACTION1: 0,
            GameAction.ACTION2: 1,
            GameAction.ACTION3: 3,
            GameAction.ACTION4: 2,
        }

        if action.id in direction_map:
            direction = direction_map[action.id]

            on_target_before = sum(
                1 for p in self.block_pos if tuple(p) in self.target_set
            )

            self.gravity = direction
            self.block_pos = apply_gravity_to(
                self.block_pos, direction, self.blocker_set
            )
            self.moves_left -= 1

            idx = self._current_level_index
            if idx >= self.FLASH_FROM_LEVEL:
                on_target_after = sum(
                    1 for p in self.block_pos if tuple(p) in self.target_set
                )

                if on_target_after > on_target_before:
                    self.flash_color = FLASH_GOOD

                elif on_target_after < on_target_before:
                    self.flash_color = FLASH_BAD

                else:
                    self.flash_color = None

            self.render()

            if self.check_win():
                if self._current_level_index < len(self._levels) - 1:
                    self.next_level()
                else:
                    self.win()
                self.complete_action()
                return

            if self.moves_left <= 0:
                if self.lose_life():
                    self.complete_action()
                    return
                self.complete_action()
                return

        self.complete_action()

    def prepare_for_reset(self) -> None:
        self.preserve_lives = False
        self.has_played = False

    def handle_reset(self) -> None:
        if self._state == EngineGameState.WIN or not self.has_played:
            self.just_reset = False
            self.has_played = False
            self.full_reset()
        elif self._current_level_index == 0:
            self.just_reset = False
            self.has_played = False
            self.full_reset()
        elif self.just_reset:
            self.just_reset = False
            self.has_played = False
            self.full_reset()
        else:
            self.just_reset = True
            self.preserve_lives = False
            self.has_played = False
            self.reset_level(reset_lives=True)

    def check_win(self) -> bool:
        return set(map(tuple, self.block_pos)) == self.target_set

    def reset_level(self, reset_lives: bool = False) -> None:
        idx = self._current_level_index
        self._rng = random.Random(LEVEL_SEEDS[idx % len(LEVEL_SEEDS)] + self.game_seed)
        self.block_pos = _build_start(
            self.n_blocks, self.target_set, self.blocker_set,
            self.max_moves, self._rng,
        )
        self.moves_left = self.max_moves
        self.gravity = 1
        self.flash_color = None
        if reset_lives:
            self.lives_max = LIVES_PER_LEVEL[idx] if idx < len(LIVES_PER_LEVEL) else 3
            self.lives = self.lives_max
        self.render()

    def lose_life(self) -> bool:
        self.lives -= 1
        if self.lives <= 0:
            self.render()
            self.lose()
            return True
        self.reset_level()
        return False

    def render(self) -> None:
        self.current_level.remove_all_sprites()

        border_col = self.flash_color if self.flash_color is not None else BORDER
        self.flash_color = None

        bg_pixels = build_background(
            self.target_set,
            self.blocker_set,
            self.moves_left,
            self.max_moves,
            lives=self.lives,
            lives_max=self.lives_max,
            border_color=border_col,
        )
        bg_sprite = Sprite(
            pixels=bg_pixels.tolist(),
            name="background",
            visible=True,
            collidable=False,
            layer=0,
        )
        bg_sprite.set_position(0, 0)
        self.current_level.add_sprite(bg_sprite)

        target = self.target_set
        for i, (bx, by) in enumerate(self.block_pos):
            on_tgt = (bx, by) in target
            col = MATCHED_COL if on_tgt else BLOCK_COLORS[i % len(BLOCK_COLORS)]
            sp = Sprite(
                pixels=[[col]],
                name="blk_%d" % i,
                visible=True,
                collidable=False,
                layer=2,
            )
            sp.set_position(bx, by)
            self.current_level.add_sprite(sp)

    @property
    def extra_state(self) -> dict:
        idx = self._current_level_index
        defn = LEVEL_DEFS[idx]

        on_target = [p for p in self.block_pos if tuple(p) in self.target_set]
        off_target = [p for p in self.block_pos if tuple(p) not in self.target_set]

        grav_names = ["North(↑)", "South(↓)", "East(→)", "West(←)"]

        return {
            "remaining_cells": len(off_target),
            "circuit_title": "Gravity Switcher - Level %d/5: %s"
            % (idx + 1, defn["title"]),
            "level_title": defn["title"],
            "block_positions": list(self.block_pos),
            "target_positions": sorted(self.target_set),
            "blocker_positions": sorted(self.blocker_set),
            "blocks_on_target": len(on_target),
            "blocks_total": self.n_blocks,
            "blocks_off_target": len(off_target),
            "gravity_dir": self.gravity,
            "gravity_name": grav_names[self.gravity],
            "moves_left": self.moves_left,
            "max_moves": self.max_moves,
            "moves_used": self.max_moves - self.moves_left,
            "lives": self.lives,
            "lives_max": self.lives_max,
            "lives_lost": self.lives_max - self.lives,
            "level_features": [
                "Level %d/5: %s" % (idx + 1, defn["title"]),
                "Blocks: %d" % self.n_blocks,
                "On target: %d/%d" % (len(on_target), self.n_blocks),
                "Gravity: %s" % grav_names[self.gravity],
                "Moves left: %d/%d" % (self.moves_left, self.max_moves),
                "Lives:   %d/%d" % (self.lives, self.lives_max),
            ],
        }


ACTION_LIST: list[str] = [ "reset", "up", "down", "left", "right", "undo"]


class PuzzleEnvironment:

    GRAV_NAMES = ["North(↑)", "South(↓)", "East(→)", "West(←)"]

    def __init__(self, seed: int = 0) -> None:
        self._engine = Gs04(seed=seed)
        self.turn: int = 0
        self.last_was_reset: bool = False
        self._total_levels: int = len(self._engine._levels)
        self._cumulative_reward: float = 0.0

    def frame_to_png(self, frame) -> bytes | None:
        try:
            arr = np.array(frame, dtype=np.uint8)
            h, w = arr.shape[:2]
            rgb = np.zeros((h, w, 3), dtype=np.uint8)
            for idx, color in ARC_PALETTE.items():
                mask = arr == idx
                if mask.ndim == 3:
                    mask = mask.all(axis=2)
                rgb[mask] = color

            def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
                chunk = chunk_type + data
                return struct.pack(">I", len(data)) + chunk + struct.pack(">I", zlib.crc32(chunk) & 0xFFFFFFFF)

            raw_rows = b"".join(b"\x00" + rgb[r].tobytes() for r in range(h))
            return (
                b"\x89PNG\r\n\x1a\n"
                + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                + _png_chunk(b"IDAT", zlib.compress(raw_rows))
                + _png_chunk(b"IEND", b"")
            )
        except Exception:
            return None

    def build_text_observation(self) -> str:
        g = self._engine
        idx = g._current_level_index
        defn = LEVEL_DEFS[idx]
        on_target = sum(1 for p in g.block_pos if tuple(p) in g.target_set)
        lines = [
            f"=== Gravity Switcher (Level {idx + 1}/{len(g._levels)}: {defn['title']}) ===",
            f"Blocks: {g.n_blocks} ({on_target} on target)",
            f"Gravity: {self.GRAV_NAMES[g.gravity]}",
            f"Moves: {g.max_moves - g.moves_left}/{g.max_moves} ({g.moves_left} left)",
            f"Lives: {g.lives}/{g.lives_max}",
            "--- Grid (16x16) ---",
            f"Targets: {sorted(g.target_set)}",
            f"Blockers: {sorted(g.blocker_set)}",
            f"Block positions: {list(g.block_pos)}",
        ]
        if g._state == EngineGameState.WIN:
            lines.append("State: won")
        elif g._state == EngineGameState.GAME_OVER:
            lines.append("State: game-over")
        else:
            lines.append("State: playing")
        return "\n".join(lines)

    def build_metadata(self) -> dict:
        g = self._engine
        idx = g._current_level_index
        return {
            "game_id": g._game_id,
            "level_index": g.level_index,
            "level_name": LEVEL_DEFS[idx]["title"],
            "total_levels": len(g._levels),
            "blocks_total": g.n_blocks,
            "blocks_on_target": sum(1 for p in g.block_pos if tuple(p) in g.target_set),
            "moves_left": g.moves_left,
            "max_moves": g.max_moves,
            "lives": g.lives,
            "lives_max": g.lives_max,
        }

    def make_game_state(self, frame_data) -> GameState:
        image_bytes: bytes | None = None
        if frame_data and not frame_data.is_empty():
            image_bytes = self.frame_to_png(frame_data.frame)

        done = self._engine._state in (EngineGameState.WIN, EngineGameState.GAME_OVER)
        valid = None if done else ACTION_LIST

        return GameState(
            text_observation=self.build_text_observation(),
            image_observation=image_bytes,
            valid_actions=valid,
            turn=self.turn,
            metadata=self.build_metadata(),
        )

    def reset(self) -> GameState:
        self.turn = 0
        self.last_was_reset = False
        self._cumulative_reward = 0.0
        self._engine.prepare_for_reset()
        frame_data = self._engine.perform_action(
            ActionInput(id=GameAction.RESET)
        )
        return self.make_game_state(frame_data)

    def get_actions(self) -> list[str]:
        done = self._engine._state in (EngineGameState.WIN, EngineGameState.GAME_OVER)
        if done:
            return ["reset"]
        return ACTION_LIST

    def is_done(self) -> bool:
        return self._engine._state in (EngineGameState.WIN, EngineGameState.GAME_OVER)

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError(
                f"Unsupported render mode '{mode}'. Only 'rgb_array' is supported."
            )
        frame: np.ndarray = self._engine.camera.render(self._engine.current_level.get_sprites())
        h, w = frame.shape[:2]
        target = 64
        if h < target or w < target:
            sy = target // h
            sx = target // w
            frame = np.repeat(np.repeat(frame, sy, axis=0), sx, axis=1)
            h, w = frame.shape[:2]
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for idx_val, color in ARC_PALETTE.items():
            mask = frame == idx_val
            if mask.ndim == 3:
                mask = mask.all(axis=2)
            rgb[mask] = color
        return rgb

    def close(self) -> None:
        self._engine = None

    def step(self, action: str) -> StepResult:
        action = action.strip().lower()
        game_action = ACTION_FROM_NAME.get(action)
        if game_action is None:
            raise ValueError(
                f"Unknown action '{action}'. Valid: {ACTION_LIST}"
            )

        if game_action == GameAction.RESET and self.last_was_reset:
            self.last_was_reset = False
            self._cumulative_reward = 0.0
            self._engine.prepare_for_reset()
            frame_data = self._engine.perform_action(ActionInput(id=GameAction.RESET))
            return StepResult(
                state=self.make_game_state(frame_data),
                reward=0.0,
                done=False,
                info={"action": action, "engine_state": frame_data.state, "level_changed": True, "life_lost": False, "full_reset": True},
            )

        prev_level = self._engine.level_index
        prev_lives = self._engine.lives

        action_input = ActionInput(id=game_action)
        frame_data = self._engine.perform_action(action_input)

        if game_action == GameAction.RESET:
            self.last_was_reset = True
        else:
            self.last_was_reset = False
            self.turn += 1

        engine_state = frame_data.state
        done = engine_state in (EngineGameState.WIN, EngineGameState.GAME_OVER)

        reward = 0.0
        if engine_state == EngineGameState.WIN:
            reward = 1.0 / self._total_levels
        elif self._engine.level_index > prev_level:
            self._cumulative_reward = 1 / self._total_levels
            reward = self._cumulative_reward

        info = {
            "action": action,
            "engine_state": engine_state,
            "level_changed": self._engine.level_index != prev_level,
            "life_lost": self._engine.lives < prev_lives,
            "full_reset": frame_data.full_reset,
        }

        return StepResult(
            state=self.make_game_state(frame_data),
            reward=reward,
            done=done,
            info=info,
        )


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
