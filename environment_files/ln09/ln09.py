from __future__ import annotations

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
    RenderableUserDisplay,
    Sprite,
)
from arcengine import GameState as EngineState
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


def _make_png(index_grid: np.ndarray) -> bytes:
    h, w = index_grid.shape
    raw_rows = bytearray()
    for y in range(h):
        raw_rows.append(0)
        for x in range(w):
            idx = int(index_grid[y, x])
            if idx < len(ARC_PALETTE):
                r, g, b = (
                    int(ARC_PALETTE[idx][0]),
                    int(ARC_PALETTE[idx][1]),
                    int(ARC_PALETTE[idx][2]),
                )
            else:
                r, g, b = 0, 0, 0
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
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return (
        sig
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(bytes(raw_rows)))
        + _chunk(b"IEND", b"")
    )


FRAME_SIZE = 64
UI_ROWS = 2
PLAY_ROWS = FRAME_SIZE - UI_ROWS

COLOR_BG = 5
COLOR_FLOOR = 5
COLOR_WALL = 2
COLOR_PLAYER = 14
COLOR_TARGET = 12
COLOR_TARGET_NEXT = 11
COLOR_ENEMY = 8
COLOR_BAR_FILL = 9
COLOR_BAR_EMPTY = 5
COLOR_LIFE = 8
COLOR_LIFE_EMPTY = 1

GRID_SIZES = [(8, 8), (10, 10), (12, 12), (20, 20)]

LEVEL_DATA = [
    {
        "player_spawns": [[0, 0], [7, 7], [0, 7], [7, 0]],
        "walls": [
            (2, 7),
            (7, 2),
            (6, 5),
            (6, 1),
            (7, 4),
            (0, 3),
            (5, 3),
            (4, 3),
            (2, 2),
            (3, 5),
        ],
        "enemies": [],
        "targets": [(5, 6), (3, 6), (6, 2)],
    },
    {
        "player_spawns": [[0, 0], [9, 0], [0, 9], [5, 5]],
        "walls": [
            (0, 3),
            (0, 2),
            (9, 4),
            (0, 7),
            (2, 0),
            (7, 4),
            (7, 7),
            (3, 3),
            (1, 4),
            (6, 5),
            (1, 7),
            (6, 3),
            (1, 2),
            (2, 7),
            (7, 2),
            (3, 6),
            (7, 1),
        ],
        "enemies": [[9, 6]],
        "targets": [(8, 5), (8, 3), (0, 9), (5, 2), (4, 3)],
    },
    {
        "player_spawns": [[0, 0], [11, 0], [0, 11], [11, 11]],
        "walls": [
            (0, 1),
            (0, 5),
            (1, 5),
            (1, 7),
            (1, 10),
            (2, 1),
            (2, 9),
            (2, 10),
            (3, 1),
            (3, 7),
            (4, 6),
            (4, 7),
            (4, 10),
            (5, 0),
            (5, 2),
            (5, 3),
            (5, 6),
            (5, 10),
            (7, 1),
            (7, 6),
            (7, 7),
            (8, 2),
            (8, 4),
            (8, 10),
            (9, 7),
            (10, 0),
            (10, 2),
            (10, 4),
            (10, 6),
            (10, 7),
            (11, 9),
        ],
        "enemies": [[8, 8], [9, 6]],
        "targets": [
            (0, 3),
            (1, 1),
            (4, 1),
            (4, 3),
            (4, 5),
            (5, 8),
            (7, 8),
            (8, 0),
            (10, 10),
        ],
    },
    {
        "player_spawns": [[0, 0], [19, 0], [2, 19], [19, 19]],
        "walls": [
            (0, 2),
            (0, 10),
            (0, 11),
            (0, 15),
            (1, 4),
            (1, 7),
            (1, 19),
            (2, 2),
            (2, 4),
            (2, 5),
            (2, 7),
            (2, 10),
            (2, 13),
            (2, 15),
            (3, 12),
            (3, 13),
            (3, 16),
            (4, 0),
            (4, 12),
            (4, 13),
            (4, 14),
            (4, 19),
            (5, 11),
            (6, 17),
            (7, 3),
            (7, 8),
            (7, 15),
            (8, 2),
            (8, 17),
            (8, 19),
            (9, 8),
            (9, 12),
            (9, 17),
            (10, 11),
            (10, 14),
            (10, 17),
            (11, 1),
            (11, 5),
            (11, 14),
            (11, 15),
            (12, 3),
            (12, 9),
            (12, 11),
            (12, 18),
            (12, 19),
            (13, 0),
            (13, 1),
            (13, 10),
            (13, 17),
            (14, 0),
            (14, 7),
            (14, 10),
            (14, 12),
            (14, 14),
            (14, 15),
            (14, 17),
            (14, 18),
            (15, 2),
            (15, 14),
            (16, 10),
            (16, 13),
            (16, 16),
            (16, 18),
            (17, 2),
            (17, 6),
            (17, 7),
            (17, 9),
            (17, 12),
            (17, 19),
            (18, 2),
            (18, 6),
            (18, 7),
            (19, 2),
            (19, 4),
            (19, 7),
            (19, 13),
            (19, 18),
        ],
        "enemies": [[11, 10], [11, 17], [12, 14], [17, 15]],
        "targets": [
            (0, 7),
            (2, 18),
            (3, 7),
            (4, 18),
            (5, 4),
            (6, 8),
            (7, 10),
            (7, 11),
            (11, 7),
            (14, 1),
            (15, 1),
            (16, 3),
            (18, 16),
        ],
    },
]


class MoveDisplay(RenderableUserDisplay):
    MAX_LIVES = 3

    def __init__(self, game: "Ln09") -> None:
        self._game = game
        self.max_moves: int = 0
        self.remaining: int = 0

    def set_limit(self, max_moves: int) -> None:
        self.max_moves = max_moves
        self.remaining = max_moves

    def decrement(self) -> None:
        if self.remaining > 0:
            self.remaining -= 1

    def reset(self) -> None:
        self.remaining = self.max_moves

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        if self.max_moves == 0:
            return frame
        fw = frame.shape[1]
        fh = frame.shape[0]
        rs = fh - UI_ROWS
        re = fh
        frame[rs:re, :] = COLOR_BAR_EMPTY
        bw = int(fw * 0.85)
        filled = int(bw * self.remaining / self.max_moves)
        for x in range(filled):
            frame[rs:re, x] = COLOR_BAR_FILL
        lives = getattr(self._game, "_lives", self.MAX_LIVES)
        ls = bw
        lw = fw - bw
        lbw = 2
        lgap = 2
        tlw = (lbw * self.MAX_LIVES) + (lgap * (self.MAX_LIVES - 1))
        off = (lw - tlw) // 2
        for i in range(self.MAX_LIVES):
            xs = ls + off + i * (lbw + lgap)
            xe = xs + lbw
            c = COLOR_LIFE if i < lives else COLOR_LIFE_EMPTY
            if xe <= fw:
                frame[rs:re, xs:xe] = c
        return frame


def _make_tile(color, size):
    return [[color] * size for _ in range(size)]


def _make_sprites(ts):
    s = {}
    s["floor"] = Sprite(
        pixels=_make_tile(COLOR_FLOOR, ts), name="floor", visible=True, collidable=False
    )
    s["wall"] = Sprite(
        pixels=_make_tile(COLOR_WALL, ts), name="wall", visible=True, collidable=True
    )
    s["player"] = Sprite(
        pixels=_make_tile(COLOR_PLAYER, ts),
        name="player",
        visible=True,
        collidable=True,
    )
    s["target"] = Sprite(
        pixels=_make_tile(COLOR_TARGET, ts),
        name="target",
        visible=True,
        collidable=True,
    )
    s["target_next"] = Sprite(
        pixels=_make_tile(COLOR_TARGET_NEXT, ts),
        name="target_next",
        visible=True,
        collidable=True,
    )
    s["enemy"] = Sprite(
        pixels=_make_tile(COLOR_ENEMY, ts), name="enemy", visible=True, collidable=True
    )
    return s


levels = [
    Level(sprites=[], grid_size=(FRAME_SIZE, FRAME_SIZE)),
    Level(sprites=[], grid_size=(FRAME_SIZE, FRAME_SIZE)),
    Level(sprites=[], grid_size=(FRAME_SIZE, FRAME_SIZE)),
    Level(sprites=[], grid_size=(FRAME_SIZE, FRAME_SIZE)),
]

BACKGROUND_COLOR = 4
PADDING_COLOR = 4

MAX_LIVES = 3


class Ln09(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._player_pos = [0, 0]
        self._grid_w = 0
        self._grid_h = 0
        self._walls = set()
        self._enemies = []
        self._targets = []
        self._current_target_index = 0
        self._turn_count = 0
        self._move_count = 0
        self._max_moves = 0
        self._moves_remaining = 0
        self._game_over = False
        self._lives = MAX_LIVES
        self._undo_stack: List[dict] = []
        self._variation: int = 0
        self._last_reset_idle: bool = False

        self._init_state = {}

        self._move_display = MoveDisplay(self)
        camera = Camera(
            0, 0, 64, 64, BACKGROUND_COLOR, PADDING_COLOR, [self._move_display]
        )
        super().__init__(
            game_id="ln09",
            levels=levels,
            camera=camera,
            available_actions=[0, 1, 2, 3, 4, 7],
        )

    def on_set_level(self, level: Level) -> None:
        self._game_over = False
        self._move_count = 0
        self._turn_count = 0
        self._undo_stack = []
        self._load_level()
        self._save_init_state()
        self._render_level(level)
        self._move_display.set_limit(self._max_moves)

    def _load_level(self):
        idx = self.level_index
        data = LEVEL_DATA[idx]
        gw, gh = GRID_SIZES[idx]
        self._grid_w = gw
        self._grid_h = gh
        spawns = data["player_spawns"]
        self._player_pos = list(spawns[self._variation % len(spawns)])
        self._walls = set(data["walls"])
        self._enemies = [list(e) for e in data["enemies"]]
        self._targets = list(data["targets"])
        self._current_target_index = 0
        self._max_moves = [100, 225, 360, 770][idx]
        self._moves_remaining = self._max_moves
        self._turn_count = 0

    def _save_init_state(self):
        self._init_state = {
            "player_pos": self._player_pos[:],
            "walls": set(self._walls),
            "enemies": [e[:] for e in self._enemies],
            "targets": list(self._targets),
            "max_moves": self._max_moves,
        }

    def _render_level(self, level):
        for sp in list(level._sprites):
            level.remove_sprite(sp)

        idx = self.level_index
        gw, gh = GRID_SIZES[idx]
        ts = PLAY_ROWS // max(gw, gh)
        ox = (FRAME_SIZE - ts * gw) // 2
        oy = (PLAY_ROWS - ts * gh) // 2
        spr = _make_sprites(ts)

        for y in range(gh):
            for x in range(gw):
                px = ox + x * ts
                py = oy + y * ts
                if (x, y) in self._walls:
                    level.add_sprite(spr["wall"].clone().set_position(px, py))
                else:
                    level.add_sprite(spr["floor"].clone().set_position(px, py))

        for i, (tx, ty) in enumerate(self._targets):
            if i < self._current_target_index:
                continue
            px = ox + tx * ts
            py = oy + ty * ts
            if i == self._current_target_index:
                level.add_sprite(spr["target_next"].clone().set_position(px, py))
            else:
                level.add_sprite(spr["target"].clone().set_position(px, py))

        for enemy in self._enemies:
            px = ox + enemy[0] * ts
            py = oy + enemy[1] * ts
            level.add_sprite(spr["enemy"].clone().set_position(px, py))

        px = ox + self._player_pos[0] * ts
        py = oy + self._player_pos[1] * ts
        level.add_sprite(spr["player"].clone().set_position(px, py))

    def _is_checkmate(self):
        px, py = self._player_pos
        enemy_positions = [(e[0], e[1]) for e in self._enemies]
        if (px, py) in enemy_positions:
            return True
        possible_moves = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = px + dx, py + dy
            if 0 <= nx < self._grid_w and 0 <= ny < self._grid_h:
                if (nx, ny) not in self._walls:
                    possible_moves.append((nx, ny))
        if not possible_moves:
            return True
        safe_moves = 0
        for move in possible_moves:
            if move not in enemy_positions:
                safe_moves += 1
        return safe_moves == 0

    def _move_enemies(self):
        gw, gh = self._grid_w, self._grid_h
        px, py = self._player_pos

        for enemy in self._enemies:
            ex, ey = enemy[0], enemy[1]
            dx = 0
            dy = 0
            if px > ex:
                dx = 1
            elif px < ex:
                dx = -1
            if py > ey:
                dy = 1
            elif py < ey:
                dy = -1

            x_dist = abs(px - ex)
            y_dist = abs(py - ey)

            if x_dist >= y_dist:
                moves_to_try = [(dx, 0), (0, dy), (dx, dy)]
            else:
                moves_to_try = [(0, dy), (dx, 0), (dy, dx)]

            for mdx, mdy in moves_to_try:
                if mdx == 0 and mdy == 0:
                    continue
                new_ex = ex + mdx
                new_ey = ey + mdy
                if not (0 <= new_ex < gw and 0 <= new_ey < gh):
                    continue
                if (new_ex, new_ey) in self._walls:
                    continue
                other_enemy_positions = [
                    (e[0], e[1]) for e in self._enemies if e is not enemy
                ]
                if (new_ex, new_ey) in other_enemy_positions:
                    continue
                enemy[0] = new_ex
                enemy[1] = new_ey
                break

    def _check_death(self):
        px, py = self._player_pos
        for enemy in self._enemies:
            if enemy[0] == px and enemy[1] == py:
                return True
        if self._is_checkmate():
            return True
        if self._moves_remaining <= 0 and self._current_target_index < len(
            self._targets
        ):
            return True
        return False

    def _check_win(self):
        return self._current_target_index >= len(self._targets)

    def _restart_level(self):
        self._lives -= 1
        self._variation += 1
        if self._lives <= 0:
            self.lose()
            return
        self.level_reset()

    def _save_snapshot(self) -> None:
        self._undo_stack.append(
            {
                "player_pos": self._player_pos[:],
                "enemies": [e[:] for e in self._enemies],
                "current_target_index": self._current_target_index,
                "moves_remaining": self._moves_remaining,
                "move_count": self._move_count,
                "turn_count": self._turn_count,
                "game_over": self._game_over,
            }
        )
        if len(self._undo_stack) > 50:
            self._undo_stack.pop(0)

    def _apply_undo(self) -> None:
        if not self._undo_stack:
            return
        snap = self._undo_stack.pop()
        self._player_pos = snap["player_pos"]
        self._enemies = snap["enemies"]
        self._current_target_index = snap["current_target_index"]
        self._moves_remaining = snap["moves_remaining"]
        self._move_count = snap["move_count"]
        self._turn_count = snap["turn_count"]
        self._game_over = snap["game_over"]
        self._move_display.remaining = self._moves_remaining

    def handle_reset(self) -> None:
        had_progress = (
            self._state == EngineState.GAME_OVER
            or self._moves_remaining < self._max_moves
            or self._lives < MAX_LIVES
            or self._current_target_index > 0
        )
        self._lives = MAX_LIVES
        if self._state == EngineState.WIN:
            self._variation += 1
            self._last_reset_idle = False
            self.full_reset()
        elif had_progress:
            self._variation += 1
            self._last_reset_idle = False
            self.level_reset()
        else:
            if not self._last_reset_idle:
                self._variation += 1
            self._last_reset_idle = True
            self.full_reset()

    def step(self) -> None:
        if not self.action:
            self.complete_action()
            return

        if self.action.id == GameAction.RESET:
            self.complete_action()
            return

        self._last_reset_idle = False

        if self._game_over:
            self._restart_level()
            self.complete_action()
            return

        action_id = self.action.id

        if action_id == GameAction.ACTION7:
            if self._undo_stack and self._moves_remaining > 0:
                charged_moves = self._moves_remaining - 1
                self._apply_undo()
                self._moves_remaining = charged_moves
                self._move_display.remaining = charged_moves
            elif self._moves_remaining > 0:
                self._moves_remaining -= 1
                self._move_display.decrement()
            if self._moves_remaining <= 0 and not self._check_win():
                self._game_over = True
            self._render_level(self.current_level)
            if self._game_over:
                self._restart_level()
            self.complete_action()
            return

        dx, dy = 0, 0

        if action_id == GameAction.ACTION1:
            dy = -1
        elif action_id == GameAction.ACTION2:
            dy = 1
        elif action_id == GameAction.ACTION3:
            dx = -1
        elif action_id == GameAction.ACTION4:
            dx = 1

        if dx == 0 and dy == 0:
            self.complete_action()
            return

        self._save_snapshot()

        self._moves_remaining -= 1
        self._move_display.decrement()
        self._move_count += 1
        self._turn_count += 1

        nx, ny = self._player_pos[0] + dx, self._player_pos[1] + dy
        if 0 <= nx < self._grid_w and 0 <= ny < self._grid_h:
            if (nx, ny) not in self._walls:
                self._player_pos = [nx, ny]

                if self._current_target_index < len(self._targets):
                    next_target = self._targets[self._current_target_index]
                    if (nx, ny) == next_target:
                        self._current_target_index += 1

        idx = self.level_index
        if idx >= 2:
            if self._turn_count % 3 != 0:
                self._move_enemies()
        else:
            if self._turn_count % 2 == 0:
                self._move_enemies()

        if self._check_death():
            self._game_over = True
            self._render_level(self.current_level)
            self.complete_action()
            return

        self._render_level(self.current_level)

        if self._check_win():
            self._variation += 1
            self._lives = MAX_LIVES
            self.next_level()

        self.complete_action()

    def get_actions(self) -> list[int]:
        return self._available_actions


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
        self._engine: Ln09 = Ln09(seed=seed)
        self._total_turns = 0
        self._done = False
        self._game_won = False
        self._game_over = False
        self._last_action_was_reset = False

    def reset(self) -> GameState:
        e = self._engine

        if self._game_won or self._last_action_was_reset:
            e.perform_action(ActionInput(id=GameAction.RESET))
            e.perform_action(ActionInput(id=GameAction.RESET))
        else:
            e.perform_action(ActionInput(id=GameAction.RESET))

        self._total_turns = 0
        self._done = False
        self._last_action_was_reset = True
        self._game_won = False
        self._game_over = False

        return self._build_game_state()

    def get_actions(self) -> List[str]:
        if self._done:
            return ["reset"]
        return list(self._VALID_ACTIONS)

    def is_done(self) -> bool:
        return self._done

    def step(self, action: str) -> StepResult:
        e = self._engine

        if action == "reset":
            state = self.reset()
            return StepResult(
                state=state, reward=0.0, done=False, info={"action": "reset"}
            )

        if action not in self._ACTION_MAP:
            return StepResult(
                state=self._build_game_state(),
                reward=0.0,
                done=self._done,
                info={"action": action, "invalid": True},
            )

        self._last_action_was_reset = False
        self._total_turns += 1

        game_action = self._ACTION_MAP[action]
        info: Dict = {"action": action}

        level_before = e.level_index

        frame = e.perform_action(ActionInput(id=game_action), raw=True)

        state_name = frame.state.name if frame and frame.state else ""
        game_won = state_name == "WIN"
        game_over = state_name == "GAME_OVER"

        total_levels = len(e._levels)
        level_reward = 1.0 / total_levels

        if game_won:
            self._done = True
            self._game_won = True
            self._game_over = False
            info["reason"] = "game_complete"
            return StepResult(
                state=self._build_game_state(done=True),
                reward=level_reward,
                done=True,
                info=info,
            )

        if game_over:
            self._done = True
            self._game_won = False
            self._game_over = True
            info["reason"] = "game_over"
            return StepResult(
                state=self._build_game_state(done=True),
                reward=0.0,
                done=True,
                info=info,
            )

        reward = 0.0
        if e.level_index != level_before:
            reward = level_reward
            info["reason"] = "level_complete"

        return StepResult(
            state=self._build_game_state(done=False),
            reward=reward,
            done=False,
            info=info,
        )

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError(f"Unsupported render mode: {mode}")
        e = self._engine
        index_grid = e.camera.render(e.current_level.get_sprites())
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

    def _build_text_observation(self) -> str:
        e = self._engine
        idx = e.level_index
        gw, gh = GRID_SIZES[min(idx, len(GRID_SIZES) - 1)]
        parts = [
            f"level:{idx + 1}/{len(e._levels)}",
            f"moves:{e._moves_remaining}/{e._max_moves}",
            f"lives:{e._lives}/{MAX_LIVES}",
            f"targets:{e._current_target_index}/{len(e._targets)}",
        ]
        header = " ".join(parts)

        grid_lines: list[str] = []
        for y in range(gh):
            row_cells: list[str] = []
            for x in range(gw):
                if [x, y] == e._player_pos:
                    cell = "P"
                elif (x, y) in [(en[0], en[1]) for en in e._enemies]:
                    cell = "E"
                elif (x, y) in self._walls_set(e):
                    cell = "#"
                elif self._is_next_target(e, x, y):
                    cell = "N"
                elif self._is_future_target(e, x, y):
                    cell = "T"
                else:
                    cell = "."
                row_cells.append(cell)
            grid_lines.append("".join(row_cells))

        return header + "\n" + "\n".join(grid_lines)

    @staticmethod
    def _walls_set(e):
        return e._walls

    @staticmethod
    def _is_next_target(e, x, y):
        if e._current_target_index < len(e._targets):
            return (x, y) == e._targets[e._current_target_index]
        return False

    @staticmethod
    def _is_future_target(e, x, y):
        for i in range(e._current_target_index + 1, len(e._targets)):
            if (x, y) == e._targets[i]:
                return True
        return False

    def _build_game_state(self, done: bool = False) -> GameState:
        e = self._engine
        valid_actions = self.get_actions() if not done else None
        index_grid = e.camera.render(e.current_level.get_sprites())
        return GameState(
            text_observation=self._build_text_observation(),
            image_observation=_make_png(index_grid),
            valid_actions=valid_actions,
            turn=self._total_turns,
            metadata={
                "total_levels": len(e._levels),
                "level_index": e.level_index,
                "levels_completed": getattr(e, "_score", 0),
                "game_over": e._state == EngineState.GAME_OVER,
                "done": done,
                "info": {},
            },
        )


class ArcGameEnv(gym.Env):
    metadata: Dict[str, Any] = {
        "render_modes": ["rgb_array"],
        "render_fps": 10,
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

    try:
        check_env(env.unwrapped, skip_render_check=True)
        print("[PASS] check_env passed")
    except Exception as e:
        print(f"[FAIL] check_env failed: {e}")

    obs, info = env.reset()
    print(f"  obs shape: {obs.shape}, dtype: {obs.dtype}")
    print(f"  info keys: {list(info.keys())}")

    obs, reward, term, trunc, info = env.step(0)
    print(f"  step -> reward={reward}, terminated={term}, truncated={trunc}")

    frame = env.render()
    print(f"  render -> shape={frame.shape if frame is not None else None}")

    env.close()
    print("  close() OK")
