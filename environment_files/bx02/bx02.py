import random
import struct
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

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
from arcengine import (
    GameState as EngineGameState,
)
from gymnasium import spaces


def _encode_png(rgb: np.ndarray) -> bytes:

    height, width, _ = rgb.shape
    raw_rows = []
    for y in range(height):
        raw_rows.append(b"\x00" + rgb[y].tobytes())
    raw_data = b"".join(raw_rows)
    compressed = zlib.compress(raw_data)

    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        c = chunk_type + data
        crc = struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        return struct.pack(">I", len(data)) + c + crc

    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr_data)
        + _chunk(b"IDAT", compressed)
        + _chunk(b"IEND", b"")
    )


C_PLAYER = 14
C_WALL = 10
C_BOX = 7
C_GOAL = 4
C_COIN = 12
C_DECOY = 15
C_DANGER = 2
C_TELEPORT = 8

BACKGROUND_COLOR = 0
PADDING_COLOR = 5

_SPRITE_TEMPLATES = {
    "player": Sprite(
        pixels=[[C_PLAYER]],
        name="player",
        visible=True,
        collidable=True,
        tags=["player"],
        layer=3,
    ),
    "wall": Sprite(
        pixels=[[C_WALL]],
        name="wall",
        visible=True,
        collidable=True,
        tags=["wall"],
        layer=1,
    ),
    "box": Sprite(
        pixels=[[C_BOX]],
        name="box",
        visible=True,
        collidable=True,
        tags=["box"],
        layer=2,
    ),
    "goal": Sprite(
        pixels=[[C_GOAL]],
        name="goal",
        visible=True,
        collidable=False,
        tags=["goal"],
        layer=0,
    ),
    "coin": Sprite(
        pixels=[[C_COIN]],
        name="coin",
        visible=True,
        collidable=False,
        tags=["coin"],
        layer=0,
    ),
    "decoy": Sprite(
        pixels=[[C_DECOY]],
        name="decoy",
        visible=True,
        collidable=True,
        tags=["decoy"],
        layer=2,
    ),
    "danger": Sprite(
        pixels=[[C_DANGER]],
        name="danger",
        visible=True,
        collidable=False,
        tags=["danger"],
        layer=0,
    ),
    "teleport": Sprite(
        pixels=[[C_TELEPORT]],
        name="teleport",
        visible=True,
        collidable=False,
        tags=["teleport"],
        layer=0,
    ),
}

_DIR_DELTAS = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}
_DIR_REVERSE = {0: 1, 1: 0, 2: 3, 3: 2}

_LEVEL_GRIDS = [
    (
        "################\n"
        "#..#...........#\n"
        "#..B.......G...#\n"
        "#..............#\n"
        "#.##.......##..#\n"
        "#..........G...#\n"
        "#....#.........#\n"
        "#..B...........#\n"
        "#......#.......#\n"
        "#........R.....#\n"
        "#.T......D.....#\n"
        "#......###.....#\n"
        "#......#G#.....#\n"
        "#......###.....#\n"
        "#..P...........#\n"
        "################"
    ),
    (
        "################\n"
        "#..#...........#\n"
        "#..B.......G...#\n"
        "#..............#\n"
        "#.....##.......#\n"
        "#..B..#....G...#\n"
        "#..............#\n"
        "#........#.....#\n"
        "#..B.......G...#\n"
        "#..............#\n"
        "#..........R...#\n"
        "#.T....###.D...#\n"
        "#......#G#.....#\n"
        "#......###..R..#\n"
        "#..P...........#\n"
        "################"
    ),
    (
        "##################\n"
        "#..#.............#\n"
        "#..B..........G..#\n"
        "#................#\n"
        "#......####......#\n"
        "#..B..#.......G..#\n"
        "#.......#........#\n"
        "#...........R....#\n"
        "#......####......#\n"
        "#..B..........G..#\n"
        "#.......#........#\n"
        "#............R...#\n"
        "#.T..............#\n"
        "#..........###D..#\n"
        "#..........#G#...#\n"
        "#..........###.R.#\n"
        "#..P.............#\n"
        "##################"
    ),
    (
        "##################\n"
        "#..#.............#\n"
        "#..B.....C....G..#\n"
        "#................#\n"
        "#......####......#\n"
        "#..B.....C....G..#\n"
        "#.......#........#\n"
        "#...........R....#\n"
        "#......####......#\n"
        "#..B.....C....G..#\n"
        "#.......#........#\n"
        "#............R...#\n"
        "#.T..............#\n"
        "#..........###D..#\n"
        "#..........#G#...#\n"
        "#..........###...#\n"
        "#..P.............#\n"
        "##################"
    ),
    (
        "####################\n"
        "#..#...............#\n"
        "#..B.......C...G...#\n"
        "#..................#\n"
        "#........####......#\n"
        "#..B..#....C...G...#\n"
        "#.........#........#\n"
        "#............R.G...#\n"
        "#..................#\n"
        "#........####..D...#\n"
        "#..B.......C...G...#\n"
        "#.........#........#\n"
        "#............R.....#\n"
        "#.T................#\n"
        "#............###...#\n"
        "#............#G#...#\n"
        "#..D.........###...#\n"
        "#............R.....#\n"
        "#..P...............#\n"
        "####################"
    ),
    (
        "######################\n"
        "#..#.................#\n"
        "#..B.........C...G...#\n"
        "#....................#\n"
        "#..........####......#\n"
        "#..B..#......C...G...#\n"
        "#...........#........#\n"
        "#..............R.G...#\n"
        "#....................#\n"
        "#..........####......#\n"
        "#..B..#......C...G...#\n"
        "#...........#........#\n"
        "#..............R.D...#\n"
        "#..B.........C...G...#\n"
        "#....................#\n"
        "#..............R.....#\n"
        "#..T.................#\n"
        "#..............###...#\n"
        "#..............#G#...#\n"
        "#..D...........###...#\n"
        "#..P...........R.....#\n"
        "######################"
    ),
    (
        "########################\n"
        "#..#...................#\n"
        "#..B...........C...G...#\n"
        "#........B.............#\n"
        "#...........R..........#\n"
        "#...####.........####..#\n"
        "#..B..#........C...G...#\n"
        "#.............#........#\n"
        "#...........R..........#\n"
        "#...####.........####..#\n"
        "#..B..#........C...G...#\n"
        "#.............#........#\n"
        "#...........R..........#\n"
        "#...####.........####..#\n"
        "#..B..#........C...G...#\n"
        "#.............#........#\n"
        "#...........R......D...#\n"
        "#......................#\n"
        "#..T...................#\n"
        "#..............###.....#\n"
        "#..............#G#.....#\n"
        "#..D...........###..R..#\n"
        "#..P...................#\n"
        "########################"
    ),
]

_LEVEL_MAX_STEPS = [55, 80, 110, 130, 170, 230, 320]

_LEVEL_DECOY_AXES = [
    [(3, 0)],
    [(3, 0)],
    [(3, 0)],
    [(3, 0)],
    [(3, 0), (3, 0)],
    [(3, 0), (3, 0)],
    [(3, 0), (3, 0)],
]

_LEVEL_SPAWNS = [
    [(3, 14), (10, 14), (1, 13), (13, 14)],
    [(3, 14), (10, 14), (1, 13), (8, 14)],
    [(3, 16), (8, 16), (1, 15), (14, 16)],
    [(3, 16), (8, 16), (1, 15), (14, 16)],
    [(3, 18), (10, 18), (1, 17), (16, 18)],
    [(3, 20), (10, 20), (1, 19), (18, 20)],
    [(3, 22), (10, 22), (1, 21), (20, 22)],
]

_LEVEL_TELEPORT_DEST = [
    (8, 12),
    (8, 12),
    (12, 14),
    (12, 14),
    (14, 15),
    (16, 18),
    (16, 20),
]

_CHAR_MAP = {
    "#": "wall",
    "P": "player",
    "B": "box",
    "G": "goal",
    "C": "coin",
    "D": "decoy",
    "R": "danger",
    "T": "teleport",
}

_CAMERA_SIZES = [
    (16, 16),
    (16, 16),
    (18, 18),
    (18, 18),
    (20, 20),
    (22, 22),
    (24, 24),
]

_ACTION_MAP = {
    "reset": GameAction.RESET,
    "up": GameAction.ACTION1,
    "down": GameAction.ACTION2,
    "left": GameAction.ACTION3,
    "right": GameAction.ACTION4,
    "click": GameAction.ACTION6,
    "undo": GameAction.ACTION7,
}


def _parse_grid(grid_str: str, data: Optional[Dict] = None) -> Level:
    rows = grid_str.split("\n")
    height = len(rows)
    width = max(len(row) for row in rows)
    level_sprites: List[Sprite] = []
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            key = _CHAR_MAP.get(ch)
            if key:
                level_sprites.append(_SPRITE_TEMPLATES[key].clone().set_position(x, y))
    return Level(sprites=level_sprites, grid_size=(width, height), data=data or {})


def _build_levels() -> List[Level]:
    levels = []
    for i in range(len(_LEVEL_GRIDS)):
        grid = _LEVEL_GRIDS[i]
        data = {
            "max_steps": _LEVEL_MAX_STEPS[i],
            "decoy_axes": _LEVEL_DECOY_AXES[i],
            "teleport_dest": _LEVEL_TELEPORT_DEST[i],
            "grid_str": grid,
        }
        levels.append(_parse_grid(grid, data))
    return levels


class _StepDisplay(RenderableUserDisplay):
    BAR_WIDTH = 42
    BAR_X = 4
    BAR_Y = 61

    def __init__(self, game: "Bx02") -> None:
        self._game = game
        self.max_steps: int = 0
        self.remaining: int = 0

    def set_limit(self, max_steps: int) -> None:
        self.max_steps = max_steps
        self.remaining = max_steps

    def tick(self) -> bool:
        if self.remaining > 0:
            self.remaining -= 1
        return self.remaining > 0

    def reset(self) -> None:
        self.remaining = self.max_steps

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        if self.max_steps == 0:
            return frame
        filled = int(self.BAR_WIDTH * self.remaining / self.max_steps)
        for i in range(self.BAR_WIDTH):
            color = 11 if i < filled else 5
            frame[self.BAR_Y : self.BAR_Y + 2, self.BAR_X + i] = color
        for i in range(3):
            x = 52 + i * 4
            color = 8 if self._game._lives > i else 5
            frame[self.BAR_Y : self.BAR_Y + 2, x : x + 2] = color
        return frame


class Bx02(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)
        levels = _build_levels()
        self._step_display = _StepDisplay(self)
        self._lives: int = 3
        self._has_moved: bool = False
        self._undo_stack: List[Dict] = []

        camera = Camera(
            width=_CAMERA_SIZES[0][0],
            height=_CAMERA_SIZES[0][1],
            background=BACKGROUND_COLOR,
            letter_box=PADDING_COLOR,
            interfaces=[self._step_display],
        )
        super().__init__(
            "bx02", levels, camera, available_actions=[0, 1, 2, 3, 4, 6, 7]
        )

    def on_set_level(self, level: Level) -> None:
        idx = min(self._current_level_index, len(_CAMERA_SIZES) - 1)
        cw, ch = _CAMERA_SIZES[idx]
        self.camera.width = cw
        self.camera.height = ch

        self._player: Sprite = self.current_level.get_sprites_by_tag("player")[0]
        self._boxes: List[Sprite] = list(self.current_level.get_sprites_by_tag("box"))
        self._goals: List[Sprite] = list(self.current_level.get_sprites_by_tag("goal"))
        self._walls: List[Sprite] = list(self.current_level.get_sprites_by_tag("wall"))
        self._coins: List[Sprite] = list(self.current_level.get_sprites_by_tag("coin"))
        self._decoys: List[Sprite] = list(
            self.current_level.get_sprites_by_tag("decoy")
        )
        self._dangers: List[Sprite] = list(
            self.current_level.get_sprites_by_tag("danger")
        )
        self._teleports: List[Sprite] = list(
            self.current_level.get_sprites_by_tag("teleport")
        )

        self._select_spawn()

        self._collected_coins: Set[int] = set()
        self._removed_coins: List[Sprite] = []
        self._removed_teleports: List[Sprite] = []

        self._teleport_dest: Optional[Tuple[int, int]] = self.current_level.get_data(
            "teleport_dest"
        )

        self._init_decoy_dirs()
        self._cache_initial_positions()

        self._grid_str: str = self.current_level.get_data("grid_str") or ""
        self._has_moved = False
        self._undo_stack = []
        self._load_step_limit()

    def _init_decoy_dirs(self) -> None:
        axes_data = self.current_level.get_data("decoy_axes") or []
        self._decoy_x_dirs: List[int] = [a[0] for a in axes_data]
        self._decoy_y_dirs: List[int] = [a[1] for a in axes_data]
        self._decoy_stopped: List[bool] = [False] * len(self._decoys)

    def _select_spawn(self) -> None:
        idx = min(self._current_level_index, len(_LEVEL_SPAWNS) - 1)
        spawns = _LEVEL_SPAWNS[idx]
        choice = self._rng.randint(0, len(spawns) - 1)
        sx, sy = spawns[choice]
        self._player.set_position(sx, sy)

    def _cache_initial_positions(self) -> None:
        self._init_player_pos: Tuple[int, int] = (self._player.x, self._player.y)
        self._init_box_positions: List[Tuple[int, int]] = [
            (b.x, b.y) for b in self._boxes
        ]
        self._init_decoy_positions: List[Tuple[int, int]] = [
            (d.x, d.y) for d in self._decoys
        ]
        self._init_decoy_x_dirs: List[int] = list(self._decoy_x_dirs)
        self._init_decoy_y_dirs: List[int] = list(self._decoy_y_dirs)

    def _load_step_limit(self) -> None:
        ms = self.current_level.get_data("max_steps")
        if ms:
            self._step_display.set_limit(ms)

    def request_full_restart(self) -> None:
        self._has_moved = False

    def handle_reset(self) -> None:
        if self._state == EngineGameState.WIN or not self._has_moved:
            self._levels = _build_levels()
            self._current_level_index = 0
            self._lives = 3
            self.on_set_level(self._levels[0])
        else:
            self._restore_level()
            self._step_display.reset()
            self._lives = 3
            self._has_moved = False
            self._undo_stack = []

    def _restore_level(self) -> None:
        self._select_spawn()
        self._init_player_pos = (self._player.x, self._player.y)
        for box, (x, y) in zip(self._boxes, self._init_box_positions):
            box.set_position(x, y)
        for decoy, (x, y) in zip(self._decoys, self._init_decoy_positions):
            decoy.set_position(x, y)
        self._decoy_x_dirs = list(self._init_decoy_x_dirs)
        self._decoy_y_dirs = list(self._init_decoy_y_dirs)
        self._decoy_stopped = [False] * len(self._decoys)

        for coin in self._removed_coins:
            coin.set_visible(True)
            self.current_level.add_sprite(coin)
            self._coins.append(coin)
        self._removed_coins.clear()
        self._collected_coins.clear()

        for tp in self._removed_teleports:
            tp.set_visible(True)
            self.current_level.add_sprite(tp)
            self._teleports.append(tp)
        self._removed_teleports.clear()

    def _save_snapshot(self) -> None:
        self._undo_stack.append(
            {
                "player": (self._player.x, self._player.y),
                "boxes": [(b.x, b.y) for b in self._boxes],
                "decoys": [(d.x, d.y) for d in self._decoys],
                "decoy_x_dirs": list(self._decoy_x_dirs),
                "decoy_y_dirs": list(self._decoy_y_dirs),
                "decoy_stopped": list(self._decoy_stopped),
                "collected_coins": set(self._collected_coins),
                "removed_coin_ids": [id(c) for c in self._removed_coins],
                "removed_tp_ids": [id(t) for t in self._removed_teleports],
            }
        )

    def _apply_undo(self) -> None:
        if not self._undo_stack:
            return
        snap = self._undo_stack.pop()
        self._player.set_position(*snap["player"])
        for box, (x, y) in zip(self._boxes, snap["boxes"]):
            box.set_position(x, y)
        for decoy, (x, y) in zip(self._decoys, snap["decoys"]):
            decoy.set_position(x, y)
        self._decoy_x_dirs = snap["decoy_x_dirs"]
        self._decoy_y_dirs = snap["decoy_y_dirs"]
        self._decoy_stopped = snap["decoy_stopped"]

        coins_to_restore = []
        for c in self._removed_coins:
            if id(c) not in snap["removed_coin_ids"]:
                coins_to_restore.append(c)
        for c in coins_to_restore:
            c.set_visible(True)
            self.current_level.add_sprite(c)
            self._coins.append(c)
            self._removed_coins.remove(c)
        self._collected_coins = snap["collected_coins"]

        tps_to_restore = []
        for t in self._removed_teleports:
            if id(t) not in snap["removed_tp_ids"]:
                tps_to_restore.append(t)
        for t in tps_to_restore:
            t.set_visible(True)
            self.current_level.add_sprite(t)
            self._teleports.append(t)
            self._removed_teleports.remove(t)

    def _find_at(self, x: int, y: int, sprite_list: List[Sprite]) -> Optional[Sprite]:
        for s in sprite_list:
            if s.x == x and s.y == y:
                return s
        return None

    def _is_blocked(self, x: int, y: int) -> bool:
        return self._find_at(x, y, self._walls) is not None

    def _is_danger(self, x: int, y: int) -> bool:
        return self._find_at(x, y, self._dangers) is not None

    def _is_goal(self, x: int, y: int) -> bool:
        return self._find_at(x, y, self._goals) is not None

    def _collect_coin_at(self, x: int, y: int) -> None:
        coin = self._find_at(x, y, self._coins)
        if coin and id(coin) not in self._collected_coins:
            self._collected_coins.add(id(coin))
            coin.set_visible(False)
            self.current_level.remove_sprite(coin)
            self._coins.remove(coin)
            self._removed_coins.append(coin)

    def _try_teleport_box(self, box: Sprite, x: int, y: int) -> None:
        tp = self._find_at(x, y, self._teleports)
        if tp and self._teleport_dest:
            dest_x, dest_y = self._teleport_dest
            box.set_position(dest_x, dest_y)
            tp.set_visible(False)
            self.current_level.remove_sprite(tp)
            self._teleports.remove(tp)
            self._removed_teleports.append(tp)
            self._collect_coin_at(dest_x, dest_y)

    def _check_win(self) -> bool:
        if len(self._coins) > 0:
            return False
        for box in self._boxes:
            if not self._is_goal(box.x, box.y):
                return False
        return True

    def _is_decoy_blocked(self, x: int, y: int, others: List[Sprite]) -> bool:
        return (
            self._is_blocked(x, y)
            or self._find_at(x, y, self._boxes) is not None
            or self._find_at(x, y, others) is not None
            or (x == self._player.x and y == self._player.y)
        )

    def _move_single_decoy(self, i: int, player_axis: str) -> bool:
        decoy = self._decoys[i]
        if self._decoy_stopped[i]:
            return False

        d = self._decoy_x_dirs[i] if player_axis == "x" else self._decoy_y_dirs[i]
        dx, dy = _DIR_DELTAS[d]
        nx, ny = decoy.x + dx, decoy.y + dy
        others = [self._decoys[j] for j in range(len(self._decoys)) if j != i]

        if self._is_decoy_blocked(nx, ny, others):
            d = _DIR_REVERSE[d]
            if player_axis == "x":
                self._decoy_x_dirs[i] = d
            else:
                self._decoy_y_dirs[i] = d
            dx, dy = _DIR_DELTAS[d]
            nx, ny = decoy.x + dx, decoy.y + dy
            if self._is_decoy_blocked(nx, ny, others):
                return False

        decoy.set_position(nx, ny)
        if self._is_goal(nx, ny):
            self._decoy_stopped[i] = True
        return self._is_danger(nx, ny)

    def _move_decoys(self, player_axis: str) -> bool:
        hit_danger = False
        for i in range(len(self._decoys)):
            if self._move_single_decoy(i, player_axis):
                hit_danger = True
        return hit_danger

    def _trigger_life_loss(self) -> bool:
        self._lives -= 1
        if self._lives <= 0:
            self.lose()
            self.complete_action()
            return True

        self._restore_level()
        self._step_display.reset()
        self._undo_stack = []
        self.complete_action()
        return False

    def _complete_or_lose(self) -> None:
        if self._step_display.remaining <= 0:
            self._trigger_life_loss()
            return
        self.complete_action()

    def _handle_undo(self) -> None:
        self._has_moved = True
        self._step_display.tick()
        self._apply_undo()
        self._complete_or_lose()

    def _resolve_direction(self) -> Tuple[int, int]:
        if self.action.id == GameAction.ACTION1:
            return 0, -1
        if self.action.id == GameAction.ACTION2:
            return 0, 1
        if self.action.id == GameAction.ACTION3:
            return -1, 0
        if self.action.id == GameAction.ACTION4:
            return 1, 0
        if self.action.id == GameAction.ACTION6:
            return self._click_to_direction()
        return 0, 0

    def _try_push_decoy(self, nx: int, ny: int, dx: int, dy: int) -> bool:
        decoy = self._find_at(nx, ny, self._decoys)
        if not decoy:
            return True
        di = self._decoys.index(decoy)
        if not self._decoy_stopped[di]:
            self._complete_or_lose()
            return False
        bx, by = nx + dx, ny + dy
        if (
            self._is_blocked(bx, by)
            or self._find_at(bx, by, self._boxes) is not None
            or self._find_at(bx, by, self._decoys) is not None
        ):
            self._complete_or_lose()
            return False
        decoy.set_position(bx, by)
        self._decoy_stopped[di] = False
        return True

    def _try_push_box(self, nx: int, ny: int, dx: int, dy: int) -> bool:
        box = self._find_at(nx, ny, self._boxes)
        if not box:
            return True
        bx, by = nx + dx, ny + dy
        if self._is_blocked(bx, by):
            self._complete_or_lose()
            return False
        if self._find_at(bx, by, self._boxes):
            self._complete_or_lose()
            return False
        if self._find_at(bx, by, self._decoys):
            self._complete_or_lose()
            return False
        if self._is_danger(bx, by):
            self._trigger_life_loss()
            return False
        box.set_position(bx, by)
        self._try_teleport_box(box, bx, by)
        self._collect_coin_at(box.x, box.y)
        return True

    def _finalize_move(self, nx: int, ny: int, dx: int, dy: int) -> None:
        self._player.set_position(nx, ny)
        if self._is_danger(nx, ny):
            self._trigger_life_loss()
            return
        if self._check_win():
            self._lives = 3
            self.next_level()
            self.complete_action()
            return
        player_axis = "x" if dx != 0 else "y"
        if self._move_decoys(player_axis):
            self._trigger_life_loss()
            return
        self._complete_or_lose()

    def _build_text_observation(self) -> str:
        rows = self._grid_str.split("\n")
        height = len(rows)
        width = max(len(r) for r in rows)
        grid = [["." for _ in range(width)] for _ in range(height)]

        for s in self._walls:
            grid[s.y][s.x] = "#"
        for s in self._goals:
            grid[s.y][s.x] = "G"
        for s in self._dangers:
            grid[s.y][s.x] = "R"
        for s in self._teleports:
            grid[s.y][s.x] = "T"
        for s in self._coins:
            grid[s.y][s.x] = "C"
        for s in self._boxes:
            grid[s.y][s.x] = "B"
        for s in self._decoys:
            grid[s.y][s.x] = "D"
        grid[self._player.y][self._player.x] = "P"

        lines = ["".join(row) for row in grid]
        lines.append(
            f"Steps:{self._step_display.remaining}/{self._step_display.max_steps} Lives:{self._lives} Coins:{len(self._removed_coins)} Undo:{len(self._undo_stack)}"
        )
        lines.append(
            "Actions: [up,down,left,right]=Move [undo]=Undo(costs step) [click]=Click [reset]=Reset"
        )
        return "\n".join(lines)

    def _click_to_direction(self) -> Tuple[int, int]:
        cx = self.action.data.get("x", 0)
        cy = self.action.data.get("y", 0)
        idx = min(self._current_level_index, len(_CAMERA_SIZES) - 1)
        cw, ch = _CAMERA_SIZES[idx]
        scale_x = 64 / cw
        scale_y = 64 / ch
        gx = int(cx / scale_x)
        gy = int(cy / scale_y)
        px, py = self._player.x, self._player.y
        diff_x = gx - px
        diff_y = gy - py
        if diff_x == 0 and diff_y == 0:
            return 0, 0
        if abs(diff_x) >= abs(diff_y):
            return (1 if diff_x > 0 else -1), 0
        return 0, (1 if diff_y > 0 else -1)

    def step(self) -> None:
        if self.action.id == GameAction.RESET:
            self.complete_action()
            return

        if self.action.id == GameAction.ACTION7:
            self._handle_undo()
            return

        dx, dy = self._resolve_direction()
        if dx == 0 and dy == 0:
            self.complete_action()
            return

        self._save_snapshot()
        self._has_moved = True
        self._step_display.tick()

        nx, ny = self._player.x + dx, self._player.y + dy

        if self._is_blocked(nx, ny):
            self._complete_or_lose()
            return

        if not self._try_push_decoy(nx, ny, dx, dy):
            return

        if not self._try_push_box(nx, ny, dx, dy):
            return

        self._finalize_move(nx, ny, dx, dy)


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

_VALID_ACTIONS = ["reset", "up", "down", "left", "right", "click", "undo"]


class PuzzleEnvironment:
    TOTAL_LEVELS: int = len(_LEVEL_GRIDS)

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self._engine: Optional[Bx02] = Bx02(seed=seed)
        self._engine.perform_action(ActionInput(id=GameAction.RESET))
        self._total_turns: int = 0
        self._done: bool = False
        self._won: bool = False
        self._last_action_was_reset: bool = False
        self._total_levels: int = self.TOTAL_LEVELS

    def reset(self) -> GameState:
        if self._engine is None:
            self._engine = Bx02(seed=self._seed)
            self._engine.perform_action(ActionInput(id=GameAction.RESET))
        else:
            if self._won or self._last_action_was_reset:
                self._engine.request_full_restart()
            self._engine.perform_action(ActionInput(id=GameAction.RESET))
        self._total_turns = 0
        self._done = False
        self._won = False
        self._last_action_was_reset = True
        return self._make_state()

    def get_actions(self) -> List[str]:
        if self._done:
            return ["reset"]
        return list(_VALID_ACTIONS)

    def _handle_engine_reset(self) -> StepResult:
        if self._won or self._last_action_was_reset:
            self._engine.request_full_restart()
        self._last_action_was_reset = True
        self._engine.perform_action(ActionInput(id=GameAction.RESET))
        self._done = False
        self._won = False
        return StepResult(
            state=self._make_state(),
            reward=0.0,
            done=False,
            info={},
        )

    def _build_result(self, level_before: int, level_after: int) -> StepResult:
        reward = 0.0

        if level_after > level_before:
            reward = (
                level_after / self._total_levels - level_before / self._total_levels
            )

        if self._engine._state == EngineGameState.WIN:
            self._done = True
            self._won = True
            if reward == 0.0:
                reward = 1.0 - level_before / self._total_levels
        elif self._engine._state == EngineGameState.GAME_OVER:
            self._done = True
            self._won = False
            reward = 0.0

        return StepResult(
            state=self._make_state(),
            reward=reward,
            done=self._done,
            info={},
        )

    def step(self, action: str) -> StepResult:
        if action == "reset":
            return self._handle_engine_reset()

        if self._done:
            return StepResult(
                state=self._make_state(),
                reward=0.0,
                done=True,
                info={},
            )

        self._last_action_was_reset = False

        parts = action.split()
        action_name = parts[0]

        game_action = _ACTION_MAP.get(action_name)
        if game_action is None:
            return StepResult(
                state=self._make_state(),
                reward=0.0,
                done=self._done,
                info={},
            )

        self._total_turns += 1

        data = {}
        if action_name == "click" and len(parts) == 3:
            data = {"x": int(parts[1]), "y": int(parts[2])}

        level_before = self._engine._current_level_index
        self._engine.perform_action(ActionInput(id=game_action, data=data))
        level_after = self._engine._current_level_index
        return self._build_result(level_before, level_after)

    def is_done(self) -> bool:
        return self._done

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError(f"Unsupported render mode: {mode}")
        if self._engine is None:
            return np.zeros((64, 64, 3), dtype=np.uint8)
        index_grid = self._engine.camera.render(
            self._engine.current_level.get_sprites()
        )
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        for idx, color in enumerate(_ARC_PALETTE):
            mask = index_grid == idx
            rgb[mask] = color
        return rgb

    def close(self) -> None:
        self._engine = None

    def _make_state(self) -> GameState:
        if self._engine is None:
            return GameState(
                text_observation="",
                image_observation=None,
                valid_actions=None,
                turn=self._total_turns,
                metadata={
                    "total_levels": self._total_levels,
                    "levels_completed": getattr(self._engine, "_score", 0),
                    "level_index": getattr(self._engine, "_current_level_index", getattr(self._engine, "level_index", 0)),
                    "game_over": getattr(getattr(self._engine, "_state", None), "name", "") == "GAME_OVER",
                },
            )

        text_obs = self._engine._build_text_observation()

        image_obs: Optional[bytes] = None
        rgb = self.render()
        image_obs = _encode_png(rgb)

        valid = ["reset"] if self._done else list(_VALID_ACTIONS)

        return GameState(
            text_observation=text_obs,
            image_observation=image_obs,
            valid_actions=valid,
            turn=self._total_turns,
            metadata={
                "total_levels": self._total_levels,
                "levels_completed": getattr(self._engine, "_score", 0),
                "level_index": getattr(self._engine, "_current_level_index", getattr(self._engine, "level_index", 0)),
                "game_over": getattr(getattr(self._engine, "_state", None), "name", "") == "GAME_OVER",
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
