from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set
from collections import deque
import random
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from arcengine import (
    ActionInput,
    ARCBaseGame,
    Camera,
    GameAction,
    GameState as EngineGameState,
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


C_BLACK = 0
C_BLUE = 1
C_BROWN = 2
C_LIME = 3
C_WHITE = 4
C_GRAY = 5
C_MAGENTA = 6
C_ORANGE = 7
C_RED = 8
C_CYAN = 9
C_LTBLUE = 10
C_YELLOW = 11
C_PINK = 12
C_OLIVE = 13
C_GREEN = 14
C_PURPLE = 15

BACKGROUND_COLOR = C_BLACK
PADDING_COLOR = C_BLACK

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


def _blank_pixel_canvas(grid_w: int, grid_h: int) -> np.ndarray:
    return np.full((grid_h, grid_w), -1, dtype=np.int16)


def _build_wall_sprite(cells: List[Tuple[int, int]], grid_w: int, grid_h: int,
                       color: int = C_GRAY) -> Sprite:
    pixels = _blank_pixel_canvas(grid_w, grid_h)
    for cx, cy in cells:
        if 0 <= cy < grid_h and 0 <= cx < grid_w:
            pixels[cy][cx] = color
    return Sprite(
        pixels=pixels,
        name="wall_composite",
        visible=True,
        collidable=True,
        layer=0,
        tags=["wall"],
    )


def _build_target_sprite(x: int, y: int) -> Sprite:
    return Sprite(
        pixels=np.array([[C_GREEN]], dtype=np.int16),
        name="target",
        visible=True,
        collidable=False,
        layer=-1,
        tags=["target"],
    ).clone().set_position(x, y)


def _build_decoy_sprite(x: int, y: int) -> Sprite:
    return Sprite(
        pixels=np.array([[C_PINK]], dtype=np.int16),
        name="decoy",
        visible=True,
        collidable=False,
        layer=-1,
        tags=["decoy"],
    ).clone().set_position(x, y)


def _build_player_sprite(x: int, y: int) -> Sprite:
    return Sprite(
        pixels=np.array([[C_YELLOW]], dtype=np.int16),
        name="player",
        visible=True,
        collidable=False,
        layer=2,
        tags=["player"],
    ).clone().set_position(x, y)


def _build_block_sprite(x: int, y: int) -> Sprite:
    return Sprite(
        pixels=np.array([[C_CYAN]], dtype=np.int16),
        name="block",
        visible=True,
        collidable=False,
        layer=1,
        tags=["block"],
    ).clone().set_position(x, y)


def _make_level(
    name: str,
    grid_w: int,
    grid_h: int,
    border_cells: List[Tuple[int, int]],
    ca_wall_cells: List[Tuple[int, int]],
    player_pos: Tuple[int, int],
    block_pos: Tuple[int, int],
    target_pos: Tuple[int, int],
    move_limit: int,
    locked_cells: Optional[List[Tuple[int, int]]] = None,
    decoy_targets: Optional[List[Tuple[int, int]]] = None,
) -> Level:
    locked_cells = locked_cells or []
    decoy_targets = decoy_targets or []

    all_wall_cells = list(border_cells) + list(ca_wall_cells)
    wall_sprite = _build_wall_sprite(all_wall_cells, grid_w, grid_h, C_GRAY)
    wall_sprite = wall_sprite.clone().set_position(0, 0)

    player = _build_player_sprite(*player_pos)
    block = _build_block_sprite(*block_pos)
    target = _build_target_sprite(*target_pos)

    sprites = [wall_sprite, target, player, block]
    for dx, dy in decoy_targets:
        sprites.append(_build_decoy_sprite(dx, dy))

    return Level(
        sprites=sprites,
        grid_size=(grid_w, grid_h),
        data={
            "grid_w": grid_w,
            "grid_h": grid_h,
            "player_pos": player_pos,
            "block_pos": block_pos,
            "target_pos": target_pos,
            "border_cells": border_cells,
            "ca_wall_cells": ca_wall_cells,
            "move_limit": move_limit,
            "locked_cells": locked_cells,
            "decoy_targets": decoy_targets,
        },
        name=name,
    )


def _border_walls(w: int, h: int) -> List[Tuple[int, int]]:
    cells = []
    for x in range(w):
        cells.append((x, 0))
        cells.append((x, h - 1))
    for y in range(1, h - 1):
        cells.append((0, y))
        cells.append((w - 1, y))
    return cells


def _make_bfs_state(
    player_pos: Tuple[int, int],
    block_pos: Tuple[int, int],
    ca: Set[Tuple[int, int]],
    border: Set[Tuple[int, int]],
) -> Tuple:
    return (player_pos[0], player_pos[1], block_pos[0], block_pos[1], frozenset(ca - border))


def _ca_evolve(
    walls: Set[Tuple[int, int]],
    border: Set[Tuple[int, int]],
    ca: Set[Tuple[int, int]],
    player_pos: Tuple[int, int],
    block_pos: Tuple[int, int],
    target_pos: Tuple[int, int],
    grid_w: int,
    grid_h: int,
    locked_cells: Optional[Set[Tuple[int, int]]] = None,
) -> Set[Tuple[int, int]]:
    static_non_evolving = border | (locked_cells or set())
    wall_seeds = walls | {player_pos, block_pos}
    protected = {player_pos, block_pos, target_pos}
    new_ca: Set[Tuple[int, int]] = set()

    candidates: Set[Tuple[int, int]] = set()
    for wx, wy in wall_seeds:
        for ndx in (-1, 0, 1):
            for ndy in (-1, 0, 1):
                nx, ny = wx + ndx, wy + ndy
                if 0 <= nx < grid_w and 0 <= ny < grid_h:
                    candidates.add((nx, ny))
    candidates -= static_non_evolving

    for cx, cy in candidates:
        count = 0
        for ndx in (-1, 0, 1):
            for ndy in (-1, 0, 1):
                if ndx == 0 and ndy == 0:
                    continue
                if (cx + ndx, cy + ndy) in wall_seeds:
                    count += 1

        if (cx, cy) in ca:
            if 2 <= count <= 3 and (cx, cy) not in protected:
                new_ca.add((cx, cy))
        else:
            if count == 3 and (cx, cy) not in protected:
                new_ca.add((cx, cy))

    return static_non_evolving | new_ca


def _level1_data():
    gw, gh = 10, 10
    border = _border_walls(gw, gh)
    ca_walls = [(6, 2), (6, 3), (6, 4)]
    return _make_level(
        "Level 1 - Cross-Axis Push",
        gw, gh, border, ca_walls,
        player_pos=(4, 1), block_pos=(4, 2), target_pos=(8, 4),
        move_limit=60,
    )


def _level2_data():
    gw, gh = 10, 10
    border = _border_walls(gw, gh)
    ca_walls = [(4, 3), (4, 4), (4, 5), (7, 3), (7, 4)]
    return _make_level(
        "Level 2 - Locked Gate",
        gw, gh, border, ca_walls,
        player_pos=(1, 4), block_pos=(2, 4), target_pos=(8, 4),
        move_limit=60,
        locked_cells=[(5, 6)],
    )


def _level3_data():
    gw, gh = 10, 10
    border = _border_walls(gw, gh)
    ca_walls = [
        (3, 3), (3, 4), (3, 5),
        (5, 3), (5, 4), (5, 5),
        (7, 3), (7, 4), (7, 5),
    ]
    return _make_level(
        "Level 3 - Budget Gauntlet",
        gw, gh, border, ca_walls,
        player_pos=(1, 4), block_pos=(2, 4), target_pos=(8, 4),
        move_limit=30,
    )


def _level4_data():
    gw, gh = 10, 10
    border = _border_walls(gw, gh)
    ca_walls = [
        (4, 3), (4, 4), (4, 5),
        (3, 4), (5, 4),
        (7, 3), (7, 4),
    ]
    return _make_level(
        "Level 4 - Decoy Target",
        gw, gh, border, ca_walls,
        player_pos=(1, 4), block_pos=(2, 4), target_pos=(8, 7),
        move_limit=90,
        decoy_targets=[(8, 4)],
    )


levels = [
    _level1_data(),
    _level2_data(),
    _level3_data(),
    _level4_data(),
]


def _bfs_reachable_from_initial(level: Level, grid_w: int, grid_h: int) -> bool:
    player_pos = tuple(level.get_data("player_pos"))
    block_pos = tuple(level.get_data("block_pos"))
    target_pos = tuple(level.get_data("target_pos"))
    border_cells: Set[Tuple[int, int]] = set(tuple(c) for c in level.get_data("border_cells"))
    ca_walls: Set[Tuple[int, int]] = set(tuple(c) for c in level.get_data("ca_wall_cells"))
    locked_cells: Set[Tuple[int, int]] = set(tuple(c) for c in (level.get_data("locked_cells") or []))
    decoy_targets: Set[Tuple[int, int]] = set(tuple(c) for c in (level.get_data("decoy_targets") or []))
    static_walls = border_cells | locked_cells

    MAX_STATES = 5000
    start_ca = frozenset(ca_walls)
    start = (player_pos[0], player_pos[1], block_pos[0], block_pos[1], start_ca)
    visited: Set = {start}
    queue: deque = deque([start])

    def wall_at(x: int, y: int, walls: Set[Tuple[int, int]]) -> bool:
        if x < 0 or y < 0 or x >= grid_w or y >= grid_h:
            return True
        return (x, y) in walls

    while queue and len(visited) < MAX_STATES:
        px, py, bx, by, fca = queue.popleft()
        cur_ca: Set[Tuple[int, int]] = set(fca)
        cur_walls = static_walls | cur_ca

        for move_dx, move_dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            npx, npy = px + move_dx, py + move_dy
            if wall_at(npx, npy, cur_walls):
                continue
            nbx, nby = bx, by
            if npx == bx and npy == by:
                nbx, nby = bx + move_dx, by + move_dy
                if wall_at(nbx, nby, cur_walls):
                    continue
            if (nbx, nby) in decoy_targets:
                continue
            new_walls = _ca_evolve(
                cur_walls, border_cells, cur_ca,
                (npx, npy), (nbx, nby), target_pos,
                grid_w, grid_h,
                locked_cells,
            )
            new_ca = frozenset(new_walls - static_walls)
            if wall_at(npx, npy, new_walls):
                continue
            if wall_at(nbx, nby, new_walls):
                continue
            if (nbx, nby) == target_pos:
                return True
            state = (npx, npy, nbx, nby, new_ca)
            if state not in visited:
                visited.add(state)
                queue.append(state)

    return len(visited) >= MAX_STATES


def _assert_levels_valid() -> None:
    for i, level in enumerate(levels):
        gw = level.get_data("grid_w")
        gh = level.get_data("grid_h")
        assert _bfs_reachable_from_initial(level, gw, gh), (
            f"Level {i + 1} ({level.name}) is not reachable from its initial state."
        )


if __debug__:
    _assert_levels_valid()


class Cf02Hud(RenderableUserDisplay):

    def __init__(self, game: "Cf02"):
        self.game = game

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        moves_left = self.game._move_limit - self.game._moves_used
        limit = self.game._move_limit
        bar_start = 10
        bar_length = 53 
        filled = int(round(moves_left / max(limit, 1) * bar_length))
        filled = max(0, min(filled, bar_length))
        ratio = moves_left / max(limit, 1)
        fill_color = C_GREEN if ratio > 0.5 else (C_YELLOW if moves_left > 2 else C_RED)
        for i in range(bar_length):
            px = bar_start + i
            frame[0, px] = fill_color if i < filled else C_BLACK

        lives = self.game._lives
        base_x = 1
        box_size = 2
        spacing = box_size + 1

        for i in range(lives):
            bx = base_x + i * spacing
            for dx in range(box_size):
                fx = bx + dx
                if 0 <= fx < 64:
                    frame[0, fx] = C_RED

        return frame


class Cf02(ARCBaseGame):

    def __init__(self, seed: int = 0) -> None:
        self.hud = Cf02Hud(self)

        self._lives: int = 3
        self._consecutive_resets: int = 0
        self._actions_since_reset: int = 0
        self._pending_walls: Optional[Set[Tuple[int, int]]] = None

        self._seed = seed
        self._rng = random.Random(seed)
        self._last_pos_idx = -1
        self._level_positions: Dict[int, List[Tuple[int, int]]] = {}

        first_gw = levels[0].get_data("grid_w") if levels else 8
        first_gh = levels[0].get_data("grid_h") if levels else 8

        self._undo_stack = []

        super().__init__(
            game_id="cf02",
            levels=levels,
            camera=Camera(0, 0, 64, 64, BACKGROUND_COLOR, PADDING_COLOR, [self.hud]),
            win_score=1,
            available_actions=[0, 1, 2, 3, 4, 7],
            seed=seed
        )

    def handle_reset(self) -> None:
        if self._state == EngineGameState.WIN:
            self._consecutive_resets = 0
            self._actions_since_reset = 0
            self._lives = 3
            self.full_reset()
        elif (self._consecutive_resets >= 1 and self._actions_since_reset == 0):
            self._consecutive_resets = 0
            self._actions_since_reset = 0
            self._lives = 3
            self.full_reset()
        else:
            self._consecutive_resets += 1
            self._actions_since_reset = 0
            self._lives = 3
            self._restore_level()
        self._build_state()

    def on_set_level(self, level: Level) -> None:
        gw = self.current_level.get_data("grid_w")
        gh = self.current_level.get_data("grid_h")
        self._grid_w = gw
        self._grid_h = gh
        self._lives = 3

        move_limit_raw = self.current_level.get_data("move_limit")
        self._move_limit: int = move_limit_raw if move_limit_raw is not None else 60
        self._moves_used: int = 0

        self._player_sprite = self.current_level.get_sprites_by_tag("player")[0]
        self._block_sprite = self.current_level.get_sprites_by_tag("block")[0]
        self._wall_sprite = self.current_level.get_sprites_by_tag("wall")[0]
        self._target_sprite = self.current_level.get_sprites_by_tag("target")[0]

        self._target_pos: Tuple[int, int] = tuple(self.current_level.get_data("target_pos"))

        self._border_cells: Set[Tuple[int, int]] = set(
            tuple(c) for c in self.current_level.get_data("border_cells")
        )

        self._ca_walls: Set[Tuple[int, int]] = set(
            tuple(c) for c in self.current_level.get_data("ca_wall_cells")
        )

        self._locked_cells: Set[Tuple[int, int]] = set(
            tuple(c) for c in (self.current_level.get_data("locked_cells") or [])
        )

        self._decoy_targets: Set[Tuple[int, int]] = set(
            tuple(c) for c in (self.current_level.get_data("decoy_targets") or [])
        )

        self._static_walls: Set[Tuple[int, int]] = self._border_cells | self._locked_cells
        self._all_walls: Set[Tuple[int, int]] = self._static_walls | self._ca_walls
        self._init_block_pos: Tuple[int, int] = tuple(self.current_level.get_data("block_pos"))
        self._init_ca_walls: Set[Tuple[int, int]] = set(self._ca_walls)
        self._pending_walls = None

        level_idx = self._current_level_index
        if level_idx not in self._level_positions:
            self._level_positions[level_idx] = self._compute_valid_positions()

        positions = self._level_positions[level_idx]
        available = [i for i in range(len(positions)) if i != self._last_pos_idx]
        if not available:
            available = list(range(len(positions)))
        idx = available[self._rng.randrange(0, len(available))]
        self._last_pos_idx = idx
        chosen_pos = positions[idx]

        self._player_sprite.set_position(*chosen_pos)
        self._init_player_pos = chosen_pos

        self._undo_stack = []

        self._update_wall_sprite()

    def _compute_valid_positions(self) -> List[Tuple[int, int]]:
        occupied: Set[Tuple[int, int]] = set()
        occupied |= self._border_cells
        occupied |= self._ca_walls
        occupied |= self._locked_cells
        occupied |= self._decoy_targets
        occupied.add(self._init_block_pos)
        occupied.add(self._target_pos)

        orig_player_pos = tuple(self.current_level.get_data("player_pos"))
        block_pos = self._init_block_pos
        ox, oy = orig_player_pos

        max_dist = 3
        candidates: List[Tuple[int, int]] = []
        for y in range(max(0, oy - max_dist), min(self._grid_h, oy + max_dist + 1)):
            for x in range(max(0, ox - max_dist), min(self._grid_w, ox + max_dist + 1)):
                if (x, y) in occupied:
                    continue
                if (x, y) == orig_player_pos:
                    continue
                candidates.append((x, y))

        reachable: List[Tuple[int, int]] = []
        for pos in candidates:
            if self._is_reachable_from(
                pos, block_pos,
                self._static_walls | self._ca_walls,
                self._static_walls,
                set(self._ca_walls),
                self._target_pos,
            ):
                reachable.append(pos)

        reachable.insert(0, orig_player_pos)

        if len(reachable) <= 4:
            return reachable

        sampled_indices = self._rng.sample(range(1, len(reachable)), 3)
        return [reachable[0]] + [reachable[i] for i in sorted(sampled_indices)]

    def _is_reachable_from(
        self,
        player_pos: Tuple[int, int],
        block_pos: Tuple[int, int],
        walls: Set[Tuple[int, int]],
        border: Set[Tuple[int, int]],
        ca: Set[Tuple[int, int]],
        target_pos: Tuple[int, int],
    ) -> bool:
        return self._is_reachable(player_pos, block_pos, walls, border, ca, target_pos)

    def _update_wall_sprite(self) -> None:
        gw = self._grid_w
        gh = self._grid_h
        pixels = _blank_pixel_canvas(gw, gh)

        for cx, cy in self._border_cells:
            if 0 <= cy < gh and 0 <= cx < gw:
                pixels[cy][cx] = C_GRAY

        for cx, cy in self._locked_cells:
            if 0 <= cy < gh and 0 <= cx < gw:
                pixels[cy][cx] = C_RED

        for cx, cy in self._ca_walls:
            if 0 <= cy < gh and 0 <= cx < gw:
                pixels[cy][cx] = C_ORANGE

        self._wall_sprite.pixels = pixels

    def _build_state(self) -> None:
        gw = getattr(self, "_grid_w", 8)
        gh = getattr(self, "_grid_h", 8)
        px = getattr(self._player_sprite, "x", 0)
        py = getattr(self._player_sprite, "y", 0)
        bx = getattr(self._block_sprite, "x", 0)
        by = getattr(self._block_sprite, "y", 0)
        tx, ty = getattr(self, "_target_pos", (0, 0))

        self.text_observation = (
            "CELLULAR PUSH | "
            "Controls: W=up S=down A=left D=right UNDO=undo-last-move | "
            "Colors: YELLOW=player CYAN=block GREEN=target PINK=decoy(penalty) "
            "GRAY=border RED=locked-wall ORANGE=ca-wall | "
            "CA-Rule: orange walls evolve each move using Game-of-Life rules; "
            "your position and block position count as phantom wall neighbours | "
            "Locked walls (RED) never evolve and cannot be pushed through | "
            "Decoy targets (PINK) lose a life if the block lands on them | "
            "Level {level} | Moves {used}/{limit} | Lives {lives} | "
            "Player ({px},{py}) | Block ({bx},{by}) | Target ({tx},{ty}) | "
            "Locked {locked} | Decoys {decoys}"
        ).format(
            level=getattr(self, "level_index", 0) + 1,
            used=getattr(self, "_moves_used", 0),
            limit=getattr(self, "_move_limit", 0),
            lives=getattr(self, "_lives", 3),
            px=px, py=py, bx=bx, by=by, tx=tx, ty=ty,
            locked=sorted(getattr(self, "_locked_cells", set())),
            decoys=sorted(getattr(self, "_decoy_targets", set())),
        )

        grid = [["." for _ in range(gw)] for _ in range(gh)]

        for cx, cy in getattr(self, "_border_cells", set()):
            if 0 <= cy < gh and 0 <= cx < gw:
                grid[cy][cx] = "#"
        for cx, cy in getattr(self, "_locked_cells", set()):
            if 0 <= cy < gh and 0 <= cx < gw:
                grid[cy][cx] = "L"
        for cx, cy in getattr(self, "_ca_walls", set()):
            if 0 <= cy < gh and 0 <= cx < gw:
                grid[cy][cx] = "O"
        for dx, dy in getattr(self, "_decoy_targets", set()):
            if 0 <= dy < gh and 0 <= dx < gw:
                grid[dy][dx] = "D"

        if 0 <= ty < gh and 0 <= tx < gw:
            grid[ty][tx] = "G"
        if 0 <= by < gh and 0 <= bx < gw:
            grid[by][bx] = "C"
        if 0 <= py < gh and 0 <= px < gw:
            grid[py][px] = "P"

        self.image_observation = "\n".join(" ".join(row) for row in grid)
        self.reward = 0.0
        self.done = False

    def _wall_at(self, x: int, y: int) -> bool:
        if x < 0 or y < 0 or x >= self._grid_w or y >= self._grid_h:
            return True
        return (x, y) in self._all_walls

    def _wall_at_static(
        self, x: int, y: int, walls: Set[Tuple[int, int]]
    ) -> bool:
        if x < 0 or y < 0 or x >= self._grid_w or y >= self._grid_h:
            return True
        return (x, y) in walls

    def _simulate_evolve(
        self,
        walls: Set[Tuple[int, int]],
        border: Set[Tuple[int, int]],
        ca: Set[Tuple[int, int]],
        player_pos: Tuple[int, int],
        block_pos: Tuple[int, int],
        target_pos: Tuple[int, int],
    ) -> Set[Tuple[int, int]]:
        return _ca_evolve(
            walls, border, ca,
            player_pos, block_pos, target_pos,
            self._grid_w, self._grid_h,
            self._locked_cells,
        )

    def _is_reachable(
        self,
        player_pos: Tuple[int, int],
        block_pos: Tuple[int, int],
        walls: Set[Tuple[int, int]],
        border: Set[Tuple[int, int]],
        ca: Set[Tuple[int, int]],
        target_pos: Tuple[int, int],
    ) -> bool:
        MAX_STATES = 1500

        start_ca = frozenset(ca - border)
        start = _make_bfs_state(player_pos, block_pos, ca, border)

        visited: Set = {start}
        queue: deque = deque([start])

        while queue and len(visited) < MAX_STATES:
            px, py, bx, by, fca = queue.popleft()

            cur_ca: Set[Tuple[int, int]] = set(fca)
            cur_walls = border | cur_ca

            for ddx, ddy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                npx, npy = px + ddx, py + ddy

                if self._wall_at_static(npx, npy, cur_walls):
                    continue

                nbx, nby = bx, by

                if npx == bx and npy == by:
                    nbx, nby = bx + ddx, by + ddy
                    if self._wall_at_static(nbx, nby, cur_walls):
                        continue

                if (nbx, nby) in self._decoy_targets:
                    continue

                new_walls = _ca_evolve(
                    cur_walls, border, cur_ca,
                    (npx, npy), (nbx, nby), target_pos,
                    self._grid_w, self._grid_h,
                    self._locked_cells,
                )
                new_ca = frozenset(new_walls - border)

                if self._wall_at_static(npx, npy, new_walls):
                    continue
                if self._wall_at_static(nbx, nby, new_walls):
                    continue

                if (nbx, nby) == target_pos:
                    return True

                state = _make_bfs_state((npx, npy), (nbx, nby), set(new_ca), border)
                if state not in visited:
                    visited.add(state)
                    queue.append(state)

        if len(visited) >= MAX_STATES:
            return True

        return False

    def _try_move(self, dx: int, dy: int) -> bool:
        px = self._player_sprite.x
        py = self._player_sprite.y
        bx = self._block_sprite.x
        by = self._block_sprite.y

        new_px = px + dx
        new_py = py + dy

        if self._wall_at(new_px, new_py):
            return False

        new_bx, new_by = bx, by
        if new_px == bx and new_py == by:
            new_bx = bx + dx
            new_by = by + dy
            if self._wall_at(new_bx, new_by):
                return False

        candidate_walls = _ca_evolve(
            self._all_walls,
            self._static_walls,
            self._ca_walls,
            (new_px, new_py),
            (new_bx, new_by),
            self._target_pos,
            self._grid_w,
            self._grid_h,
            self._locked_cells,
        )
        candidate_ca = candidate_walls - self._static_walls

        if (new_bx, new_by) not in self._decoy_targets and (new_bx, new_by) != self._target_pos:
            tx, ty = self._target_pos
            if abs(new_bx - tx) + abs(new_by - ty) > 1:
                if not self._is_reachable(
                    (new_px, new_py),
                    (new_bx, new_by),
                    candidate_walls,
                    self._static_walls,
                    candidate_ca,
                    self._target_pos,
                ):
                    return False

        if new_bx != bx or new_by != by:
            self._block_sprite.set_position(new_bx, new_by)
        self._player_sprite.set_position(new_px, new_py)
        self._pending_walls = candidate_walls
        return True

    def _evolve_walls(self) -> None:
        if self._pending_walls is not None:
            new_walls = self._pending_walls
            self._pending_walls = None
        else:
            new_walls = _ca_evolve(
                self._all_walls,
                self._static_walls,
                self._ca_walls,
                (self._player_sprite.x, self._player_sprite.y),
                (self._block_sprite.x, self._block_sprite.y),
                self._target_pos,
                self._grid_w,
                self._grid_h,
                self._locked_cells,
            )
        self._ca_walls = new_walls - self._static_walls
        self._all_walls = self._static_walls | self._ca_walls
        self._update_wall_sprite()

    def _check_win(self) -> bool:
        bx = self._block_sprite.x
        by = self._block_sprite.y
        return (bx, by) == self._target_pos

    def _check_decoy_hit(self) -> bool:
        bx = self._block_sprite.x
        by = self._block_sprite.y
        return (bx, by) in self._decoy_targets

    def _restore_level(self) -> None:
        self._block_sprite.set_position(*self._init_block_pos)
        self._ca_walls = set(self._init_ca_walls)
        self._all_walls = self._static_walls | self._ca_walls
        self._moves_used = 0
        self._pending_walls = None

        level_idx = self._current_level_index
        positions = self._level_positions.get(level_idx, [self._init_player_pos])
        available = [i for i in range(len(positions)) if i != self._last_pos_idx]
        if not available:
            available = list(range(len(positions)))
        idx = available[self._rng.randrange(0, len(available))]
        self._last_pos_idx = idx
        chosen_pos = positions[idx]

        self._player_sprite.set_position(*chosen_pos)
        self._init_player_pos = chosen_pos

        self._undo_stack = []

        self._update_wall_sprite()

    def _trigger_life_loss(self) -> bool:
        self._lives -= 1
        if self._lives <= 0:
            self.lose()
            self.complete_action()
            return True
        self._restore_level()
        return False

    def _save_state(self):
        self._undo_stack.append({
            "player_pos": (self._player_sprite.x, self._player_sprite.y),
            "block_pos": (self._block_sprite.x, self._block_sprite.y),
            "ca_walls": set(self._ca_walls),
            "all_walls": set(self._all_walls),
            "pending_walls": self._pending_walls,
        })

    def _restore_from_undo(self):
        if not self._undo_stack:
            return
        state = self._undo_stack.pop()
        self._player_sprite.set_position(*state["player_pos"])
        self._block_sprite.set_position(*state["block_pos"])
        self._ca_walls = state["ca_walls"]
        self._all_walls = state["all_walls"]
        self._pending_walls = state["pending_walls"]
        self._update_wall_sprite()

    def step(self) -> None:
        if self.action.id == GameAction.RESET:
            self.complete_action()
            return

        moved = False
        action_taken = False

        if self.action.id == GameAction.ACTION7:
            self._restore_from_undo()
            self._moves_used += 1
            self._actions_since_reset += 1
            if self._moves_used >= self._move_limit:
                if self._trigger_life_loss():
                    return
            self._build_state()
            self.complete_action()
            return

        if self.action.id == GameAction.ACTION1:
            action_taken = True
            self._save_state()
            moved = self._try_move(0, -1)
            if not moved:
                self._undo_stack.pop()
        elif self.action.id == GameAction.ACTION2:
            action_taken = True
            self._save_state()
            moved = self._try_move(0, 1)
            if not moved:
                self._undo_stack.pop()
        elif self.action.id == GameAction.ACTION3:
            action_taken = True
            self._save_state()
            moved = self._try_move(-1, 0)
            if not moved:
                self._undo_stack.pop()
        elif self.action.id == GameAction.ACTION4:
            action_taken = True
            self._save_state()
            moved = self._try_move(1, 0)
            if not moved:
                self._undo_stack.pop()

        if action_taken:
            self._actions_since_reset += 1
            self._consecutive_resets = 0
            self._moves_used += 1

        if moved:
            self._evolve_walls()

            if self._check_win():
                self.next_level()
                self.complete_action()
                return

            if self._check_decoy_hit():
                if self._trigger_life_loss():
                    return
                self._build_state()
                self.complete_action()
                return

        if action_taken and self._moves_used >= self._move_limit:
            if self._trigger_life_loss():
                return
            self._build_state()
            self.complete_action()
            return

        self._build_state()
        self.complete_action()


class PuzzleEnvironment:

    _ACTION_MAP: Dict[str, GameAction] = {
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
        self._engine = Cf02(seed=seed)
        self._turn = 0
        self._done = False
        self._game_over = False
        self._last_action_was_reset = False
        self._actions_since_reset = 0

    def reset(self) -> GameState:
        self._engine.perform_action(ActionInput(id=GameAction.RESET))
        self._turn = 0
        self._done = False
        self._game_over = False
        self._last_action_was_reset = True
        self._actions_since_reset = 0

        return GameState(
            text_observation=self._build_text_observation(),
            image_observation=self._build_image_observation(),
            valid_actions=self.get_actions(),
            turn=0,
            metadata={
                "total_levels": len(self._engine._levels),
                "level_index": self._engine.level_index,
                "levels_completed": getattr(self._engine, "_score", 0),
                "game_over": getattr(getattr(self._engine, "_state", None), "name", "") == "GAME_OVER",
                "done": self._done,
                "info": {},
            },
        )

    TOTAL_LEVELS: int = len(levels)

    def step(self, action: str) -> StepResult:
        if action == "reset":
            game_state = self.reset()
            return StepResult(
                state=game_state,
                reward=0.0,
                done=False,
                info={"action": "reset"},
            )

        if self._done or self._game_over:
            return StepResult(
                state=self._build_game_state(),
                reward=0.0,
                done=self._done,
                info={},
            )

        if action not in self._ACTION_MAP:
            return StepResult(
                state=self._build_game_state(),
                reward=0.0,
                done=self._done,
                info={"error": f"Invalid action '{action}'"},
            )

        self._last_action_was_reset = False
        self._turn += 1
        self._actions_since_reset += 1

        game_action = self._ACTION_MAP[action]
        level_before = self._engine._current_level_index
        info: dict = {"action": action}

        action_input = ActionInput(id=game_action)
        frame = self._engine.perform_action(action_input, raw=True)

        state_name = frame.state.name if frame and frame.state else ""
        game_won = state_name == "WIN"
        game_over = state_name == "GAME_OVER"

        level_reward_step = 1.0 / self.TOTAL_LEVELS

        if game_won:
            self._done = True
            info["event"] = "game_complete"
            return StepResult(
                state=self._build_game_state(),
                reward=level_reward_step,
                done=True,
                info=info,
            )

        if game_over:
            self._game_over = True
            info["event"] = "game_over"
            return StepResult(
                state=self._build_game_state(),
                reward=0.0,
                done=False,
                info=info,
            )

        reward = 0.0
        if self._engine._current_level_index != level_before:
            reward = level_reward_step
            info["event"] = "level_complete"

        return StepResult(
            state=self._build_game_state(),
            reward=reward,
            done=False,
            info=info,
        )

    def get_actions(self) -> List[str]:
        if self._done or self._game_over:
            return ["reset"]
        return list(self._VALID_ACTIONS)

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError(f"Unsupported render mode: {mode}")
        index_grid = self._engine.camera.render(self._engine.current_level.get_sprites())
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        for idx in range(len(ARC_PALETTE)):
            mask = index_grid == idx
            rgb[mask] = ARC_PALETTE[idx]
        return rgb

    def close(self) -> None:
        self._engine = None

    def is_done(self) -> bool:
        return self._done

    def _build_text_observation(self) -> str:
        return getattr(self._engine, "text_observation", "")

    def _build_image_observation(self) -> Optional[bytes]:
        image_obs = getattr(self._engine, "image_observation", None)
        if isinstance(image_obs, str):
            return image_obs.encode("utf-8")
        return image_obs

    def _build_game_state(self) -> GameState:
        text_obs = self._build_text_observation()
        image_obs = self._build_image_observation()

        valid = self.get_actions()

        return GameState(
            text_observation=text_obs,
            image_observation=image_obs,
            valid_actions=valid,
            turn=self._turn,
            metadata={
                "total_levels": len(self._engine._levels),
                "level_index": self._engine.level_index,
                "levels_completed": getattr(self._engine, "_score", 0),
                "game_over": self._game_over or getattr(getattr(self._engine, "_state", None), "name", "") == "GAME_OVER",
                "done": self._done,
                "info": {},
            },
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
