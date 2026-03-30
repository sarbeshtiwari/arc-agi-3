from __future__ import annotations

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
    Sprite,
)
from arcengine import GameState as EngineState


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


BACKGROUND_COLOR = 5
PLAY_AREA_COLOR = 3
PRISM_1_COLOR = 12
PRISM_2_COLOR = 6
OBSTACLE_COLOR = 4
SPLITTER_COLOR = 2
CURSOR_COLOR = 0
LIFE_ON_COLOR = 8
LIFE_OFF_COLOR = 13
LIVES_PER_LEVEL = [5, 5, 4, 3, 3]

LIGHT_COLORS_INTERNAL = [1, 2, 4]

LIGHT_COLOR_MAP = {
    0: 5,
    1: 9,
    2: 8,
    3: 15,
    4: 14,
    5: 10,
    6: 11,
    7: 0,
}


CELL_SIZE = 2
GRID_SIZE = 32
MAX_GENERATION_ATTEMPTS = 10000
BORDER = 3

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

LEVEL_VARIANT_SPAWNS: dict[int, list[tuple[int, int]]] = {
    0: [(8, 8), (24, 8), (8, 24), (24, 24)],
    1: [(10, 6), (22, 10), (6, 22), (24, 24)],
    2: [(6, 16), (26, 6), (16, 26), (26, 22)],
    3: [(8, 10), (24, 6), (10, 24), (22, 20)],
    4: [(6, 6), (26, 10), (10, 26), (22, 16)],
}

_ACTION_MAP: dict[str, GameAction] = {
    "reset": GameAction.RESET,
    "up": GameAction.ACTION1,
    "down": GameAction.ACTION2,
    "left": GameAction.ACTION3,
    "right": GameAction.ACTION4,
    "select": GameAction.ACTION5,
    "click": GameAction.ACTION6,
    "undo": GameAction.ACTION7,
}

_VALID_ACTIONS = [
    "reset",
    "up",
    "down",
    "left",
    "right",
    "select",
    "click",
    "undo",
]


def _make_png(canvas: list[list[int]], cell: int) -> bytes:
    h = len(canvas)
    w = len(canvas[0]) if h > 0 else 0
    ph = h * cell
    pw = w * cell

    raw_rows = bytearray()
    for y in range(h):
        for sy in range(cell):
            raw_rows.append(0)
            for x in range(w):
                r, g, b = (
                    ARC_PALETTE[canvas[y][x]]
                    if canvas[y][x] < len(ARC_PALETTE)
                    else (0, 0, 0)
                )
                for _ in range(cell):
                    raw_rows.append(r)
                    raw_rows.append(g)
                    raw_rows.append(b)

    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        c = chunk_type + data
        return (
            struct.pack(">I", len(data))
            + c
            + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        )

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", pw, ph, 8, 2, 0, 0, 0)
    return (
        sig
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(bytes(raw_rows)))
        + _chunk(b"IEND", b"")
    )


class Beam:
    def __init__(self, x: int, y: int, dx: int, dy: int, color: int) -> None:
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy
        self.color = color


class Rp01(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self._rng = random.Random(seed)
        self._level_sequence = list(range(5))
        self._current_level_index = 0
        self._variant_index = 0
        self._lives = LIVES_PER_LEVEL[0]

        self._emitters: list[tuple[int, int, int, int, int]] = []
        self._targets: list[tuple[int, int, int]] = []
        self._prisms: dict[tuple[int, int], bool] = {}
        self._obstacles: set[tuple[int, int]] = set()
        self._splitters: set[tuple[int, int]] = set()
        self._light_grid: list[list[int]] = []
        self._cursor: list[int] = [GRID_SIZE // 2, GRID_SIZE // 2]
        self._max_moves: int = 0
        self._moves_left: int = 0
        self._ref_prisms: dict[tuple[int, int], bool] = {}
        self._saved_levels: dict[tuple[int, int, int], dict[str, Any]] = {}
        self._undo_stack: list[dict[str, Any]] = []
        self._game_over: bool = False
        self._progress: int = 0
        self._has_acted: bool = False

        levels = [Level(sprites=[], grid_size=(GRID_SIZE, GRID_SIZE)) for _ in range(5)]
        camera = Camera(
            x=0,
            y=0,
            width=GRID_SIZE,
            height=GRID_SIZE,
            background=BACKGROUND_COLOR,
            letter_box=BACKGROUND_COLOR,
            interfaces=[],
        )
        super().__init__(
            game_id="rp01",
            levels=levels,
            camera=camera,
            available_actions=[0, 1, 2, 3, 4, 5, 6, 7],
        )

    def on_set_level(self, level: Level) -> None:
        self._current_level_index = self.level_index
        self._undo_stack = []
        self._game_over = False
        self._progress = 0
        self._has_acted = False
        self._lives = LIVES_PER_LEVEL[self._current_level_index]
        self._load_level()
        self._render()

    def handle_reset(self) -> None:
        self._game_over = False
        self._variant_index = (self._variant_index + 1) % 4
        if self._state == EngineState.WIN:
            self._lives = LIVES_PER_LEVEL[0]
            self.full_reset()
        elif self._state == EngineState.GAME_OVER:
            self._lives = LIVES_PER_LEVEL[self._current_level_index]
            self.level_reset()
        elif not self._has_acted:
            self._lives = LIVES_PER_LEVEL[0]
            self.full_reset()
        else:
            self._lives = LIVES_PER_LEVEL[self._current_level_index]
            self.level_reset()

    def step(self) -> None:
        action = self.action.id

        if action == GameAction.RESET:
            self._render()
            self.complete_action()
            return

        if action == GameAction.ACTION6:
            x = self.action.data.get("x", 0)
            y = self.action.data.get("y", 0)
            coords = self.camera.display_to_grid(x, y)
            if coords:
                gx, gy = coords
                gx = max(BORDER, min(GRID_SIZE - BORDER - 1, gx))
                gy = max(BORDER, min(GRID_SIZE - BORDER - 1, gy))
                self._cursor = [gx, gy]
                self._process_action("select")
            if self._check_level_complete():
                self._render()
                self.next_level()
                self.complete_action()
                return
            self._render()
            self.complete_action()
            return

        action_map = {
            GameAction.ACTION1: "up",
            GameAction.ACTION2: "down",
            GameAction.ACTION3: "left",
            GameAction.ACTION4: "right",
            GameAction.ACTION5: "select",
            GameAction.ACTION7: "undo",
        }

        action_name = action_map.get(action)
        if action_name is None:
            self._render()
            self.complete_action()
            return

        self._process_action(action_name)

        if self._check_level_complete():
            self._render()
            self.next_level()
            self.complete_action()
            return

        self._render()
        self.complete_action()

    def _process_action(self, action_name: str) -> None:
        gw = GRID_SIZE
        gh = GRID_SIZE

        if action_name == "undo":
            if self._undo_stack:
                snapshot = self._undo_stack.pop()
                self._cursor = snapshot["cursor"]
                self._prisms = snapshot["prisms"]
                self._progress = snapshot["progress"]
                self._simulate_light(gw, gh)
            self._moves_left -= 1
            if self._moves_left <= 0:
                self._handle_moves_exhausted()
            return

        if action_name in ("up", "down", "left", "right"):
            if action_name == "up":
                self._cursor[1] = max(BORDER, self._cursor[1] - 1)
            elif action_name == "down":
                self._cursor[1] = min(gh - BORDER - 1, self._cursor[1] + 1)
            elif action_name == "left":
                self._cursor[0] = max(BORDER, self._cursor[0] - 1)
            elif action_name == "right":
                self._cursor[0] = min(gw - BORDER - 1, self._cursor[0] + 1)
            return

        if action_name == "select":
            gx, gy = self._cursor
            if (gx, gy) not in self._prisms:
                return
            self._undo_stack.append(
                {
                    "cursor": list(self._cursor),
                    "prisms": dict(self._prisms),
                    "progress": self._progress,
                }
            )
            self._prisms[(gx, gy)] = not self._prisms[(gx, gy)]
            self._simulate_light(gw, gh)
            self._progress = self._count_matched_targets()
            self._has_acted = True
            self._moves_left -= 1
            if self._moves_left <= 0:
                self._handle_moves_exhausted()

    def _handle_moves_exhausted(self) -> None:
        self._lives -= 1
        if self._lives <= 0:
            self._game_over = True
            self.lose()
        else:
            self._variant_index = (self._variant_index + 1) % 4
            self._load_level()

    def _build_canvas(self) -> list[list[int]]:
        gw = GRID_SIZE
        gh = GRID_SIZE

        canvas: list[list[int]] = [
            [PLAY_AREA_COLOR for _ in range(gw)] for _ in range(gh)
        ]

        for y in range(gh):
            for x in range(gw):
                if x < BORDER or x >= gw - BORDER or y < BORDER or y >= gh - BORDER:
                    canvas[y][x] = BACKGROUND_COLOR

        for x in range(BORDER, gw - BORDER):
            for y in range(BORDER, gh - BORDER):
                mask = self._light_grid[x][y]
                if mask != 0:
                    canvas[y][x] = LIGHT_COLOR_MAP[mask]

        for tx, ty, mask in self._targets:
            req_color = LIGHT_COLOR_MAP.get(mask, 0)
            if 0 <= ty < gh and 0 <= tx < gw:
                canvas[ty][tx] = req_color

        for ox, oy in self._obstacles:
            canvas[oy][ox] = OBSTACLE_COLOR

        for sx, sy in self._splitters:
            canvas[sy][sx] = SPLITTER_COLOR

        for (px, py), is_forward in self._prisms.items():
            p_color = PRISM_1_COLOR if is_forward else PRISM_2_COLOR
            canvas[py][px] = p_color

        cx, cy = self._cursor
        if 0 <= cx < gw and 0 <= cy < gh:
            canvas[cy][cx] = CURSOR_COLOR

        if self._moves_left > 0 and self._max_moves > 0:
            bar_w = max(1, int((self._moves_left / self._max_moves) * (gw - 6)))
        else:
            bar_w = 0
        for bx in range(BORDER, gw - BORDER):
            c = 0 if bx - BORDER < bar_w else 3
            canvas[gh - 2][bx] = c

        for i in range(LIVES_PER_LEVEL[self._current_level_index]):
            lx = 3 + i * 2
            if 0 <= lx < gw:
                c = LIFE_ON_COLOR if i < self._lives else LIFE_OFF_COLOR
                canvas[1][lx] = c

        return canvas

    def _render(self) -> None:
        canvas = self._build_canvas()
        self.current_level.remove_all_sprites()
        self.current_level.add_sprite(
            Sprite(pixels=canvas, name="display", visible=True, collidable=False)
        )

    def _load_level(self) -> None:
        self._undo_stack = []
        if self._current_level_index < 0 or self._current_level_index >= len(
            self._level_sequence
        ):
            self._current_level_index = 0

        data_idx = self._level_sequence[self._current_level_index]
        gw = GRID_SIZE
        gh = GRID_SIZE

        level_key = (data_idx, gw, gh)

        if level_key in self._saved_levels:
            saved = self._saved_levels[level_key]
            self._emitters = [tuple(e) for e in saved["emitters"]]
            self._targets = [tuple(t) for t in saved["targets"]]
            self._prisms = dict(saved["prisms"])
            self._obstacles = set(saved["obstacles"])
            self._splitters = set(saved["splitters"])
            self._cursor = list(saved["cursor"])
            self._max_moves = saved.get("max_moves", 20)
            self._moves_left = self._max_moves
            self._ref_prisms = dict(saved["target_prisms"])
            self._apply_variant(data_idx)
            self._simulate_light(gw, gh)
            self._progress = self._count_matched_targets()
            return

        self._generate_level(data_idx, gw, gh, level_key)

    def _generate_level(
        self, data_idx: int, gw: int, gh: int, level_key: tuple[int, int, int]
    ) -> None:
        self._rng = random.Random(self._seed * 7919 + data_idx * 1000 + 3)
        level_rng = self._rng

        difficulty = data_idx
        num_emitters = [2, 2, 4, 5, 6][min(difficulty, 4)]
        num_prisms = [10, 10, 18, 24, 30][min(difficulty, 4)]
        num_obstacles = difficulty * 2
        num_splitters = max(0, difficulty - 1)
        req_targets = [1, 1, 2, 3, 4][min(difficulty, 4)]

        generation_count = 0
        generated = False
        while generation_count < MAX_GENERATION_ATTEMPTS:
            generation_count += 1
            self._reset_level_entities()

            self._place_emitters(level_rng, num_emitters, gh)
            self._place_obstacles_and_splitters(
                level_rng, num_obstacles, num_splitters, gw, gh
            )
            self._place_prisms(level_rng, num_prisms, gw, gh)

            self._simulate_light(gw, gh)
            self._collect_targets(gw, gh)

            if len(self._targets) >= req_targets:
                if self._scramble_prisms(level_rng, difficulty, gw, gh):
                    generated = True
                    break

            if generation_count % 5000 == 0:
                min_floor = [1, 1, 2, 2, 3][min(difficulty, 4)]
                req_targets = max(min_floor, req_targets - 1)

        if not generated:
            self._ref_prisms = dict(self._prisms)
            prism_positions = list(self._prisms.keys())
            if prism_positions:
                for _fallback in range(50):
                    self._prisms = dict(self._ref_prisms)
                    count = max(1, len(prism_positions) // 3)
                    to_flip = level_rng.sample(
                        prism_positions, min(count, len(prism_positions))
                    )
                    for pos in to_flip:
                        self._prisms[pos] = not self._prisms[pos]
                    self._simulate_light(gw, gh)
                    if not self._check_level_complete():
                        break

        self._finalize_level(data_idx, gw, gh, level_key)

    def _reset_level_entities(self) -> None:
        self._emitters = []
        self._targets = []
        self._prisms = {}
        self._obstacles = set()
        self._splitters = set()

    def _random_empty_pos(
        self, rng: random.Random, gw: int, gh: int
    ) -> tuple[int, int] | None:
        for _ in range(50):
            x = rng.randint(BORDER, gw - BORDER - 1)
            y = rng.randint(BORDER, gh - BORDER - 1)
            if (
                (x, y) not in self._prisms
                and (x, y) not in self._obstacles
                and (x, y) not in self._splitters
            ):
                return (x, y)
        return None

    def _place_emitters(self, rng: random.Random, num_emitters: int, gh: int) -> None:
        y_pool = list(range(BORDER, gh - BORDER))
        rng.shuffle(y_pool)
        for i in range(num_emitters):
            if i < len(y_pool):
                color = LIGHT_COLORS_INTERNAL[i % len(LIGHT_COLORS_INTERNAL)]
                self._emitters.append((2, y_pool[i], 1, 0, color))

    def _place_obstacles_and_splitters(
        self,
        rng: random.Random,
        num_obstacles: int,
        num_splitters: int,
        gw: int,
        gh: int,
    ) -> None:
        for _ in range(num_obstacles):
            pos = self._random_empty_pos(rng, gw, gh)
            if pos:
                self._obstacles.add(pos)
        for _ in range(num_splitters):
            pos = self._random_empty_pos(rng, gw, gh)
            if pos:
                self._splitters.add(pos)

    def _place_prisms(
        self, rng: random.Random, num_prisms: int, gw: int, gh: int
    ) -> None:
        num_active = int(num_prisms * 0.85)
        for _ in range(num_active):
            self._simulate_light(gw, gh)
            valid_spots = []
            for x in range(BORDER, gw - BORDER):
                for y in range(BORDER, gh - BORDER):
                    if (
                        self._light_grid[x][y] != 0
                        and (x, y) not in self._prisms
                        and (x, y) not in self._obstacles
                        and (x, y) not in self._splitters
                    ):
                        valid_spots.append((x, y))

            if valid_spots:
                px, py = rng.choice(valid_spots)
            else:
                pos = self._random_empty_pos(rng, gw, gh)
                if not pos:
                    continue
                px, py = pos

            self._prisms[(px, py)] = rng.choice([True, False])

        num_decoy = num_prisms - num_active
        for _ in range(num_decoy):
            pos = self._random_empty_pos(rng, gw, gh)
            if pos:
                self._prisms[pos] = rng.choice([True, False])

    def _collect_targets(self, gw: int, gh: int) -> None:
        self._targets = []
        for y in range(gh):
            c = self._light_grid[gw - BORDER][y]
            if c != 0:
                self._targets.append((gw - BORDER, y, c))

    def _scramble_prisms(
        self, rng: random.Random, difficulty: int, gw: int, gh: int
    ) -> bool:
        self._ref_prisms = dict(self._prisms)

        active_prisms = [
            pos for pos in self._prisms.keys() if self._light_grid[pos[0]][pos[1]] != 0
        ]
        inactive_prisms = [
            pos for pos in self._prisms.keys() if pos not in active_prisms
        ]

        active_flip_count = [2, 3, 5, 7, 9][min(difficulty, 4)]
        decoy_flip_count = [1, 2, 2, 3, 4][min(difficulty, 4)]

        for _attempt in range(50):
            self._prisms = dict(self._ref_prisms)

            to_flip = rng.sample(
                active_prisms, min(active_flip_count, len(active_prisms))
            )
            to_flip += rng.sample(
                inactive_prisms, min(decoy_flip_count, len(inactive_prisms))
            )

            for pos in to_flip:
                self._prisms[pos] = not self._prisms[pos]

            self._simulate_light(gw, gh)
            if not self._check_level_complete():
                return True

        self._prisms = dict(self._ref_prisms)
        return False

    def _finalize_level(
        self, data_idx: int, gw: int, gh: int, level_key: tuple[int, int, int]
    ) -> None:
        flipped_positions = [
            pos for pos in self._prisms if self._prisms[pos] != self._ref_prisms[pos]
        ]
        num_flips = len(flipped_positions)
        computed_min_moves = num_flips
        computed_max_moves = max(computed_min_moves * 5, 20)

        self._saved_levels[level_key] = {
            "emitters": list(self._emitters),
            "targets": list(self._targets),
            "prisms": dict(self._prisms),
            "obstacles": set(self._obstacles),
            "splitters": set(self._splitters),
            "cursor": [gw // 2, gh // 2],
            "max_moves": computed_max_moves,
            "target_prisms": dict(self._ref_prisms),
        }
        self._cursor = [gw // 2, gh // 2]
        self._max_moves = computed_max_moves
        self._moves_left = self._max_moves

        self._apply_variant(data_idx)
        self._simulate_light(gw, gh)
        self._progress = self._count_matched_targets()

    def _apply_variant(self, data_idx: int) -> None:
        variant = self._variant_index % 4
        spawns = LEVEL_VARIANT_SPAWNS.get(data_idx, LEVEL_VARIANT_SPAWNS[0])
        sx, sy = spawns[variant]
        self._cursor = [sx, sy]

    def _simulate_light(self, gw: int, gh: int) -> None:
        self._light_grid = [[0 for _ in range(gh)] for _ in range(gw)]

        beams: list[Beam] = []
        for ex, ey, edx, edy, color in self._emitters:
            if 0 <= ex < gw and 0 <= ey < gh:
                beams.append(Beam(ex, ey, edx, edy, color))
                self._light_grid[ex][ey] |= color

        max_steps = gw * gh * 2

        for _ in range(max_steps):
            if not beams:
                break

            next_beams: list[Beam] = []
            for b in beams:
                nx, ny = b.x + b.dx, b.y + b.dy

                if (
                    nx < BORDER - 1
                    or nx > gw - BORDER
                    or ny < BORDER - 1
                    or ny > gh - BORDER
                ):
                    continue

                if (nx, ny) in self._obstacles:
                    continue

                self._light_grid[nx][ny] |= b.color
                b.color = self._light_grid[nx][ny]

                if (nx, ny) in self._prisms:
                    is_forward = self._prisms[(nx, ny)]
                    if is_forward:
                        ndx, ndy = -b.dy, -b.dx
                    else:
                        ndx, ndy = b.dy, b.dx
                    next_beams.append(Beam(nx, ny, ndx, ndy, b.color))
                elif (nx, ny) in self._splitters:
                    next_beams.append(Beam(nx, ny, b.dy, b.dx, b.color))
                    next_beams.append(Beam(nx, ny, -b.dy, -b.dx, b.color))
                else:
                    next_beams.append(Beam(nx, ny, b.dx, b.dy, b.color))

            beams = next_beams

    def _count_matched_targets(self) -> int:
        matched = 0
        for tx, ty, req_color in self._targets:
            if 0 <= tx < GRID_SIZE and 0 <= ty < GRID_SIZE:
                if self._light_grid[tx][ty] == req_color:
                    matched += 1
        return matched

    def _check_level_complete(self) -> bool:
        if len(self._targets) == 0:
            return False

        for tx, ty, req_color in self._targets:
            if tx < 0 or tx >= GRID_SIZE or ty < 0 or ty >= GRID_SIZE:
                return False
            if self._light_grid[tx][ty] != req_color:
                return False

        target_set = {(tx, ty) for tx, ty, _ in self._targets}
        gw = GRID_SIZE
        gh = GRID_SIZE

        sample_tx, sample_ty, _ = self._targets[0]

        if sample_tx == gw - BORDER:
            for y in range(gh):
                if (gw - BORDER, y) not in target_set and self._light_grid[gw - BORDER][
                    y
                ] != 0:
                    return False
        elif sample_tx == BORDER - 1:
            for y in range(gh):
                if (BORDER - 1, y) not in target_set and self._light_grid[BORDER - 1][
                    y
                ] != 0:
                    return False
        elif sample_ty == gh - BORDER:
            for x in range(gw):
                if (x, gh - BORDER) not in target_set and self._light_grid[x][
                    gh - BORDER
                ] != 0:
                    return False
        elif sample_ty == BORDER - 1:
            for x in range(gw):
                if (x, BORDER - 1) not in target_set and self._light_grid[x][
                    BORDER - 1
                ] != 0:
                    return False

        return True


class PuzzleEnvironment:
    def __init__(self, seed: int = 0) -> None:
        self._engine: Rp01 | None = Rp01(seed=seed)
        self._total_turns = 0
        self._last_action_was_reset = False

    @property
    def _eng(self) -> Rp01:
        assert self._engine is not None
        return self._engine

    def reset(self) -> GameState:
        eng = self._eng
        want_full = self._is_won() or self._last_action_was_reset
        if want_full:
            eng.full_reset()
        else:
            eng.perform_action(ActionInput(id=GameAction.RESET))
        self._last_action_was_reset = True
        self._total_turns = 0
        return self._build_state()

    def get_actions(self) -> list[str]:
        if self._is_done():
            return ["reset"]
        return list(_VALID_ACTIONS)

    def step(self, action: str) -> StepResult:
        if action == "reset":
            state = self.reset()
            return StepResult(
                state=state,
                reward=0.0,
                done=False,
                info=self._build_info(),
            )

        self._last_action_was_reset = False

        if self._is_done():
            return StepResult(
                state=self._build_state(),
                reward=0.0,
                done=True,
                info=self._build_info(),
            )

        click_x = 0
        click_y = 0
        if action.startswith("click"):
            parts = action.split()
            if len(parts) == 3:
                click_x = int(parts[1])
                click_y = int(parts[2])
            action = "click"

        ga = _ACTION_MAP.get(action)
        if ga is None:
            return StepResult(
                state=self._build_state(),
                reward=0.0,
                done=self._is_done(),
                info=self._build_info(),
            )

        level_before = self._eng._current_level_index
        if ga == GameAction.ACTION6:
            self._eng.perform_action(
                ActionInput(id=ga, data={"x": click_x, "y": click_y})
            )
        else:
            self._eng.perform_action(ActionInput(id=ga))
        self._total_turns += 1
        level_after = self._eng._current_level_index
        total_levels = len(self._eng._levels)

        if self._eng._state == EngineState.WIN or level_after != level_before:
            reward = 1.0 / total_levels
        else:
            reward = 0.0

        return StepResult(
            state=self._build_state(),
            reward=reward,
            done=self._is_done(),
            info=self._build_info(),
        )

    def is_done(self) -> bool:
        return self._is_done()

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError(f"Unsupported render mode: {mode}")
        eng = self._eng
        index_grid = eng.camera.render(eng.current_level.get_sprites())
        h, w = index_grid.shape[0], index_grid.shape[1]
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for idx, color in enumerate(ARC_PALETTE):
            mask = index_grid == idx
            rgb[mask] = color
        return rgb

    def close(self) -> None:
        self._engine = None

    def _is_done(self) -> bool:
        return self._eng._state in (EngineState.GAME_OVER, EngineState.WIN)

    def _is_won(self) -> bool:
        return self._eng._state == EngineState.WIN

    def _build_state(self) -> GameState:
        done = self._is_done()
        canvas = self._eng._build_canvas()
        return GameState(
            text_observation=self._build_text(canvas),
            image_observation=_make_png(canvas, CELL_SIZE),
            valid_actions=None if done else list(_VALID_ACTIONS),
            turn=self._total_turns,
            metadata=self._build_info(),
        )

    def _build_text(self, canvas: list[list[int]]) -> str:
        e = self._eng
        header_parts = [
            f"level:{e._current_level_index + 1}/{len(e._levels)}",
            f"moves:{e._moves_left}/{e._max_moves}",
            f"lives:{e._lives}/{LIVES_PER_LEVEL[e._current_level_index]}",
            f"targets:{e._progress}/{len(e._targets)}",
            f"cursor:({e._cursor[0]},{e._cursor[1]})",
        ]
        header = " ".join(header_parts)
        color_names = {
            BACKGROUND_COLOR: "border",
            PLAY_AREA_COLOR: "empty",
            PRISM_1_COLOR: "prism_fwd",
            PRISM_2_COLOR: "prism_bwd",
            OBSTACLE_COLOR: "obstacle",
            SPLITTER_COLOR: "splitter",
            CURSOR_COLOR: "cursor",
            LIGHT_COLOR_MAP[1]: "light_blue",
            LIGHT_COLOR_MAP[2]: "light_red",
            LIGHT_COLOR_MAP[4]: "light_green",
            LIGHT_COLOR_MAP[3]: "light_purple",
            LIGHT_COLOR_MAP[5]: "light_cyan",
            LIGHT_COLOR_MAP[6]: "light_yellow",
            LIGHT_COLOR_MAP[7]: "light_white",
            LIFE_ON_COLOR: "life_on",
            LIFE_OFF_COLOR: "life_off",
        }
        grid_lines = []
        for row in canvas:
            grid_lines.append(" ".join(color_names.get(c, str(c)) for c in row))
        return header + "\n" + "\n".join(grid_lines)

    def _build_info(self) -> dict[str, Any]:
        e = self._eng
        return {
            "total_levels": len(e._levels),
            "level": e._current_level_index,
            "lives": e._lives,
            "moves_left": e._moves_left,
            "max_moves": e._max_moves,
            "progress": e._progress,
            "total_targets": len(e._targets),
            "levels_completed": getattr(e, "_score", 0),
            "level_index": e._current_level_index,
            "game_over": getattr(getattr(e, "_state", None), "name", "") == "GAME_OVER",
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
