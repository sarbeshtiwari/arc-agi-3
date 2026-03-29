import random
import struct
import zlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from arcengine import (
    ActionInput,
    ARCBaseGame,
    Camera,
    GameAction,
    Level,
    Sprite,
)
from arcengine.interfaces import RenderableUserDisplay



BLACK = 0
PADDING = 5
SELECTOR_BORDER = 4

NORMAL_A = 9
NORMAL_B = 8

LOCKED_A = 11
LOCKED_B = 10

ROW_ONLY_A = 12
ROW_ONLY_B = 13

COL_ONLY_A = 14
COL_ONLY_B = 15

MAX_LIVES = 3
HUD_COLOR_LIFE = 3
HUD_COLOR_DEAD = 2
HUD_COLOR_MOVES = 3
HUD_COLOR_LOW = 7
HUD_COLOR_BORDER = 5

LEVEL_MOVES: dict[tuple[int, int], int] = {
    (5, 5): 20,
    (8, 8): 35,
    (12, 12): 55,
    (16, 16): 80,
    (20, 20): 110,
}

TOGGLE_MAP = {
    NORMAL_A: NORMAL_B,
    NORMAL_B: NORMAL_A,
    LOCKED_A: LOCKED_B,
    LOCKED_B: LOCKED_A,
    ROW_ONLY_A: ROW_ONLY_B,
    ROW_ONLY_B: ROW_ONLY_A,
    COL_ONLY_A: COL_ONLY_B,
    COL_ONLY_B: COL_ONLY_A,
}

CELL_SPRITES = {
    NORMAL_A: Sprite(
        pixels=[[NORMAL_A]],
        name="cell_red",
        visible=True,
        collidable=True,
    ),
    NORMAL_B: Sprite(
        pixels=[[NORMAL_B]],
        name="cell_blue",
        visible=True,
        collidable=True,
    ),
    LOCKED_A: Sprite(
        pixels=[[LOCKED_A]],
        name="cell_grey",
        visible=True,
        collidable=True,
    ),
    LOCKED_B: Sprite(
        pixels=[[LOCKED_B]],
        name="cell_locked_tgt",
        visible=True,
        collidable=True,
    ),
    ROW_ONLY_A: Sprite(
        pixels=[[ROW_ONLY_A]],
        name="cell_row_a",
        visible=True,
        collidable=True,
    ),
    ROW_ONLY_B: Sprite(
        pixels=[[ROW_ONLY_B]],
        name="cell_row_b",
        visible=True,
        collidable=True,
    ),
    COL_ONLY_A: Sprite(
        pixels=[[COL_ONLY_A]],
        name="cell_col_a",
        visible=True,
        collidable=True,
    ),
    COL_ONLY_B: Sprite(
        pixels=[[COL_ONLY_B]],
        name="cell_col_b",
        visible=True,
        collidable=True,
    ),
}

levels = [
    Level(sprites=[], grid_size=(5, 5)),
    Level(sprites=[], grid_size=(8, 8)),
    Level(sprites=[], grid_size=(12, 12)),
    Level(sprites=[], grid_size=(16, 16)),
    Level(sprites=[], grid_size=(20, 20)),
]

LEVEL_CONFIG: dict[tuple[int, int], dict[str, int]] = {
    (5, 5): {"normal": 6, "locked": 0, "row_only": 0, "col_only": 0, "scramble": 3},
    (8, 8): {"normal": 10, "locked": 0, "row_only": 0, "col_only": 0, "scramble": 4},
    (12, 12): {"normal": 16, "locked": 4, "row_only": 0, "col_only": 0, "scramble": 6},
    (16, 16): {"normal": 16, "locked": 4, "row_only": 4, "col_only": 0, "scramble": 8},
    (20, 20): {"normal": 16, "locked": 4, "row_only": 4, "col_only": 4, "scramble": 10},
}

_ROW_ONLY_COLORS = {ROW_ONLY_A, ROW_ONLY_B}
_COL_ONLY_COLORS = {COL_ONLY_A, COL_ONLY_B}


class HudDisplay(RenderableUserDisplay):
    def __init__(self, game: "Pa01") -> None:
        self._game = game

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        g = self._game
        fh, fw = frame.shape

        p = PADDING
        play_x0 = p
        play_y0 = p
        play_x1 = fw - p
        play_y1 = fh - p
        play_w = play_x1 - play_x0

        dot_size = max(1, p - 2)
        dot_y = (p - dot_size) // 2
        spacing = dot_size + 2
        for i in range(MAX_LIVES):
            dot_x = play_x0 + i * spacing + 1
            color = HUD_COLOR_LIFE if g.lives > i else HUD_COLOR_BORDER
            frame[dot_y : dot_y + dot_size, dot_x : dot_x + dot_size] = color

        bar_h = max(1, p - 2)
        bar_y = play_y1 + (p - bar_h) // 2

        frame[bar_y : bar_y + bar_h, play_x0:play_x1] = HUD_COLOR_BORDER

        if g.max_moves > 0:
            remaining = max(0, g.max_moves - g.moves_used)
            ratio = remaining / g.max_moves
            bar_color = HUD_COLOR_MOVES if ratio > 0.25 else HUD_COLOR_LOW
            filled_w = int(ratio * play_w)
            if filled_w > 0:
                frame[bar_y : bar_y + bar_h, play_x0 : play_x0 + filled_w] = bar_color

        return frame


class SelectorBorderOverlay(RenderableUserDisplay):
    def __init__(
        self,
        get_camera_state: Callable[[], tuple[int, int, int, int]],
        get_selector: Callable[[], tuple[int, int]],
        border_color: int,
    ) -> None:
        self._get_camera_state = get_camera_state
        self._get_selector = get_selector
        self._border_color = border_color

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        cam_x, cam_y, cam_w, cam_h = self._get_camera_state()

        scale_x = int(64 / cam_w)
        scale_y = int(64 / cam_h)
        scale = min(scale_x, scale_y)

        if scale <= 0:
            return frame

        x_padding = int((64 - (cam_w * scale)) / 2)
        y_padding = int((64 - (cam_h * scale)) / 2)

        gx, gy = self._get_selector()
        cell_x = gx - cam_x
        cell_y = gy - cam_y

        if not (0 <= cell_x < cam_w and 0 <= cell_y < cam_h):
            return frame

        sx = x_padding + cell_x * scale
        sy = y_padding + cell_y * scale
        ex = sx + scale - 1
        ey = sy + scale - 1

        frame[sy, sx : ex + 1] = self._border_color
        frame[ey, sx : ex + 1] = self._border_color
        frame[sy : ey + 1, sx] = self._border_color
        frame[sy : ey + 1, ex] = self._border_color

        return frame


class Pa01(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._seed = seed

        self._master_rng = random.Random(seed)
        self._level_seeds: dict[tuple[int, int], int] = {}
        for lvl in levels:
            self._level_seeds[lvl.grid_size] = self._master_rng.randint(0, 2**31)

        self.lives: int = MAX_LIVES
        self.moves_used: int = 0
        self.max_moves: int = 20

        self._cells: dict[tuple[int, int], int] = {}
        self._locked: set[tuple[int, int]] = set()
        self._row_only: set[tuple[int, int]] = set()
        self._col_only: set[tuple[int, int]] = set()
        self._non_clickable: set[tuple[int, int]] = set()
        self._cell_sprites: dict[tuple[int, int], Sprite] = {}
        self._selector_pos: tuple[int, int] = (0, 0)
        self._undo_stack: list[dict] = []
        self._game_won: bool = False
        self._game_complete: bool = False
        self._consecutive_resets: int = 0

        self._hud = HudDisplay(self)
        self._selector_overlay = SelectorBorderOverlay(
            get_camera_state=lambda: (
                self.camera.x,
                self.camera.y,
                self.camera.width,
                self.camera.height,
            ),
            get_selector=lambda: self._selector_pos,
            border_color=SELECTOR_BORDER,
        )

        camera = Camera(
            background=BLACK,
            letter_box=PADDING,
            width=8,
            height=8,
            interfaces=[self._hud, self._selector_overlay],
        )

        self.total_levels = len(levels)
        self.current_level_index = 0

        super().__init__(
            game_id="pa01",
            levels=levels,
            camera=camera,
            available_actions=[0,1, 2, 3, 4, 5, 7],
        )

    @property
    def game_won(self) -> bool:
        return self._game_won

    @property
    def game_complete(self) -> bool:
        return self._game_complete

    @staticmethod
    def _make_sprite(color: int, x: int, y: int) -> Sprite:
        return CELL_SPRITES[color].clone().set_position(x, y)

    def _compute_toggle_set(self, click_x: int, click_y: int) -> set[tuple[int, int]]:
        affected: set[tuple[int, int]] = set()
        for cx, cy in self._cells:
            if (cx, cy) == (click_x, click_y):
                continue

            in_row = cy == click_y
            in_col = cx == click_x

            if (cx, cy) in self._row_only:
                if in_row:
                    affected.add((cx, cy))
            elif (cx, cy) in self._col_only:
                if in_col:
                    affected.add((cx, cy))
            else:
                if in_row or in_col:
                    affected.add((cx, cy))

        affected.add((click_x, click_y))
        return affected

    def _apply_toggle(self, click_x: int, click_y: int) -> set[tuple[int, int]]:
        toggled = self._compute_toggle_set(click_x, click_y)
        for pos in toggled:
            self._cells[pos] = TOGGLE_MAP[self._cells[pos]]
        return toggled

    def _target_for(self, pos: tuple[int, int]) -> int:
        if pos in self._row_only:
            return ROW_ONLY_B
        if pos in self._col_only:
            return COL_ONLY_B
        if pos in self._locked:
            return LOCKED_B
        return NORMAL_B

    def _is_solved(self) -> bool:
        for pos, color in self._cells.items():
            if color != self._target_for(pos):
                return False
        return True

    def _push_undo(self) -> None:
        self._undo_stack.append(
            {
                "cells": dict(self._cells),
                "selector_pos": self._selector_pos,
            }
        )

    def _pop_undo(self) -> None:
        if not self._undo_stack:
            return
        snapshot = self._undo_stack.pop()
        prev_cells = self._cells
        self._cells = snapshot["cells"]
        self._selector_pos = snapshot["selector_pos"]
        changed = {
            pos
            for pos in set(prev_cells) | set(self._cells)
            if prev_cells.get(pos) != self._cells.get(pos)
        }
        if changed:
            self._refresh_sprites(changed)

    def _random_selector_position(self) -> tuple[int, int]:
        w, h = self.current_level.grid_size
        return (self._master_rng.randint(0, w - 1), self._master_rng.randint(0, h - 1))

    @staticmethod
    def _place_random(
        rng: random.Random, w: int, h: int, n: int, occupied: set[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        placed: list[tuple[int, int]] = []
        for _ in range(n):
            for _attempt in range(500):
                x = rng.randint(0, w - 1)
                y = rng.randint(0, h - 1)
                if (x, y) not in occupied:
                    break
            else:
                continue
            occupied.add((x, y))
            placed.append((x, y))
        return placed

    @staticmethod
    def _place_in_rows(
        rng: random.Random,
        w: int,
        n: int,
        valid_rows: list[int],
        occupied: set[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        placed: list[tuple[int, int]] = []
        for _ in range(n):
            for _attempt in range(500):
                y = rng.choice(valid_rows)
                x = rng.randint(0, w - 1)
                if (x, y) not in occupied:
                    break
            else:
                continue
            occupied.add((x, y))
            placed.append((x, y))
        return placed

    @staticmethod
    def _place_in_cols(
        rng: random.Random,
        h: int,
        n: int,
        valid_cols: list[int],
        occupied: set[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        placed: list[tuple[int, int]] = []
        for _ in range(n):
            for _attempt in range(500):
                x = rng.choice(valid_cols)
                y = rng.randint(0, h - 1)
                if (x, y) not in occupied:
                    break
            else:
                continue
            occupied.add((x, y))
            placed.append((x, y))
        return placed

    def _scramble_board(
        self,
        rng: random.Random,
        normal_positions: list[tuple[int, int]],
        scramble_count: int,
    ) -> None:
        clickable = list(normal_positions)
        rng.shuffle(clickable)
        sc = min(scramble_count, len(clickable))
        for cx, cy in clickable[:sc]:
            self._apply_toggle(cx, cy)
        extra_idx = sc
        while self._is_solved() and clickable:
            self._apply_toggle(*clickable[extra_idx % len(clickable)])
            extra_idx += 1
            if extra_idx - sc > len(clickable):
                break

    def _generate_board(self) -> None:
        gs = self.current_level.grid_size
        w, h = gs
        cfg = LEVEL_CONFIG[gs]
        self._rng = random.Random(self._level_seeds[gs])
        rng = self._rng

        for sp in self._cell_sprites.values():
            try:
                self.current_level.remove_sprite(sp)
            except ValueError:
                pass

        self._cells.clear()
        self._locked.clear()
        self._row_only.clear()
        self._col_only.clear()
        self._non_clickable.clear()
        self._cell_sprites.clear()

        occupied: set[tuple[int, int]] = set()

        normal_positions = self._place_random(rng, w, h, cfg["normal"], occupied)
        locked_positions = self._place_random(rng, w, h, cfg["locked"], occupied)

        normal_rows = sorted(set(y for _, y in normal_positions))
        row_only_positions = (
            self._place_in_rows(rng, w, cfg["row_only"], normal_rows, occupied)
            if normal_rows
            else []
        )

        normal_cols = sorted(set(x for x, _ in normal_positions))
        col_only_positions = (
            self._place_in_cols(rng, h, cfg["col_only"], normal_cols, occupied)
            if normal_cols
            else []
        )

        for p in normal_positions:
            self._cells[p] = NORMAL_B
        for p in locked_positions:
            self._cells[p] = LOCKED_B
            self._locked.add(p)
        for p in row_only_positions:
            self._cells[p] = ROW_ONLY_B
            self._row_only.add(p)
        for p in col_only_positions:
            self._cells[p] = COL_ONLY_B
            self._col_only.add(p)

        self._non_clickable = self._locked | self._row_only | self._col_only

        self._scramble_board(rng, normal_positions, cfg["scramble"])

        for pos, color in self._cells.items():
            sp = self._make_sprite(color, pos[0], pos[1])
            self.current_level.add_sprite(sp)
            self._cell_sprites[pos] = sp

    def _refresh_sprites(self, positions: set[tuple[int, int]]) -> None:
        for pos in positions:
            color = self._cells[pos]
            old = self._cell_sprites.get(pos)
            if old is not None:
                try:
                    self.current_level.remove_sprite(old)
                except ValueError:
                    pass
            new_sp = self._make_sprite(color, pos[0], pos[1])
            self.current_level.add_sprite(new_sp)
            self._cell_sprites[pos] = new_sp

    def _move_selector(self, dx: int, dy: int) -> None:
        w, h = self.current_level.grid_size
        x, y = self._selector_pos
        nx = min(max(x + dx, 0), w - 1)
        ny = min(max(y + dy, 0), h - 1)
        self._selector_pos = (nx, ny)

    def _try_activate_at(self, gx: int, gy: int) -> None:
        if (gx, gy) in self._cells and (gx, gy) not in self._non_clickable:
            self._push_undo()
            toggled = self._apply_toggle(gx, gy)
            self._refresh_sprites(toggled)
            self.moves_used += 1
            if self._is_solved():
                if self.current_level_index >= self.total_levels - 1:
                    self._game_won = True
                    self._game_complete = True
                self.next_level()
                return
            if self.max_moves > 0 and self.moves_used >= self.max_moves:
                self._lose_life()

    def _lose_life(self) -> None:
        self.lives -= 1
        self._undo_stack.clear()
        if self.lives <= 0:
            self.lose()
        else:
            self.moves_used = 0
            self._generate_board()
            self._selector_pos = self._random_selector_position()

    def _activate_selected(self) -> None:
        sx, sy = self._selector_pos
        self._try_activate_at(sx, sy)

    def handle_reset(self) -> None:
        if self._game_complete:
            self.full_reset()
            return
        self._consecutive_resets += 1
        if self._consecutive_resets >= 2:
            self._consecutive_resets = 0
            self.full_reset()
        else:
            self.level_reset()

    def level_reset(self) -> None:
        super().level_reset()
        self.lives = MAX_LIVES

    def full_reset(self) -> None:
        self._consecutive_resets = 0
        self._game_complete = False
        self._game_won = False
        super().full_reset()
        self.lives = MAX_LIVES

    def on_set_level(self, level: Level) -> None:
        self.current_level_index = self._current_level_index
        self._undo_stack.clear()
        self._generate_board()
        w, h = self.current_level.grid_size
        if self.current_level_index < 2:
            self._selector_pos = (w // 2, h // 2)
        else:
            self._selector_pos = self._random_selector_position()
        if self.current_level_index == 0:
            self.lives = MAX_LIVES
        self.moves_used = 0
        self.max_moves = LEVEL_MOVES.get((w, h), 30)

    def step(self) -> None:
        aid = self.action.id

        if aid == GameAction.RESET:
            self.complete_action()
            return

        self._consecutive_resets = 0

        if aid == GameAction.ACTION7:
            self.moves_used += 1
            if self.max_moves > 0 and self.moves_used >= self.max_moves:
                self._lose_life()
                self.complete_action()
                return
            self._pop_undo()
            self.complete_action()
            return

        if aid == GameAction.ACTION1:
            self._move_selector(0, -1)
            self.complete_action()
            return

        if aid == GameAction.ACTION2:
            self._move_selector(0, 1)
            self.complete_action()
            return

        if aid == GameAction.ACTION3:
            self._move_selector(-1, 0)
            self.complete_action()
            return

        if aid == GameAction.ACTION4:
            self._move_selector(1, 0)
            self.complete_action()
            return

        if aid == GameAction.ACTION5:
            self._activate_selected()

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

_CELL_CHAR = {
    NORMAL_A: "N",
    NORMAL_B: "n",
    LOCKED_A: "K",
    LOCKED_B: "k",
    ROW_ONLY_A: "H",
    ROW_ONLY_B: "h",
    COL_ONLY_A: "V",
    COL_ONLY_B: "v",
}


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
        self._engine = Pa01(seed)
        self.seed = seed
        self._total_turns = 0
        self._done = False

    def _episode_terminal(self) -> bool:
        g = self._engine
        return g.game_won or g.lives <= 0

    def reset(self) -> GameState:
        self._done = False
        self._total_turns = 0
        self._engine.perform_action(ActionInput(id=GameAction.RESET))
        return self._create_game_state()

    def get_actions(self) -> List[str]:
        if self._episode_terminal():
            return ["reset"]
        g = self._engine
        if g.lives <= 0:
            return ["reset"]
        actions = [ "reset","up", "down", "left", "right", "select", "undo"]
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
        level_reward = 1.0 / g.total_levels
        if g.lives < lives_before:
            info["event"] = "life_lost"
            if g.lives <= 0:
                info["event"] = "game_over"
                reward = 0.0
                done = True
        elif g.game_won and not done:
            info["event"] = "game_complete"
            done = True
            reward = level_reward
        elif g.current_level_index != level_before:
            info["event"] = "level_complete"
            reward = level_reward
        return reward, done, info

    def step(self, action: str) -> StepResult:
        e = self._engine

        if action == "reset":
            e.perform_action(ActionInput(id=GameAction.RESET))
            return StepResult(
                state=self._create_game_state(), reward=0.0, done=False, info={"action": "reset"}
            )

        if action not in ACTION_MAP:
            raise ValueError(
                f"Invalid action '{action}'. Must be one of {list(ACTION_MAP.keys())}"
            )

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
        rgb = self.render()
        h, w, _ = rgb.shape
        raw_rows = b""
        for y in range(h):
            raw_rows += b"\x00" + rgb[y].tobytes()
        deflated = zlib.compress(raw_rows)

        def _chunk(ctype: bytes, data: bytes) -> bytes:
            c = ctype + data
            return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

        ihdr_data = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
        png = b"\x89PNG\r\n\x1a\n"
        png += _chunk(b"IHDR", ihdr_data)
        png += _chunk(b"IDAT", deflated)
        png += _chunk(b"IEND", b"")
        return png

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
        body = self._build_text_obs()
        text_observation = f"Agent turn: {self._total_turns}\n\n{body}"
        return GameState(
            text_observation=text_observation,
            image_observation=self._build_image_bytes(),
            valid_actions=self.get_actions(),
            turn=self._total_turns,
            metadata=self._build_metadata(),
        )

    def _build_text_obs(self) -> str:
        g = self._engine
        w, h = g.current_level.grid_size
        lines = []
        lines.append(
            f"LEVEL {g.current_level_index + 1}/{g.total_levels} "
            f"LIVES {g.lives}/{MAX_LIVES} "
            f"MOVES {g.moves_used}/{g.max_moves}"
        )
        sx, sy = g._selector_pos
        lines.append(f"CURSOR ({sx},{sy})")
        total_cells = len(g._cells)
        solved_cells = sum(1 for pos, c in g._cells.items() if c == g._target_for(pos))
        lines.append(f"SOLVED {solved_cells}/{total_cells}")
        header = "  " + "".join(str(c % 10) for c in range(w))
        lines.append(header)
        for r in range(h):
            row_chars = ""
            for c in range(w):
                if (c, r) in g._cells:
                    row_chars += _CELL_CHAR.get(g._cells[(c, r)], "?")
                else:
                    row_chars += "."
            lines.append(f"{r % 10} {row_chars}")
        lines.append("")
        lines.append("LEGEND: N/n=Normal(A/B) K/k=Locked(A/B) H/h=RowOnly(A/B) V/v=ColOnly(A/B) .=empty")
        return "\n".join(lines)

    def _build_metadata(self) -> Dict:
        g = self._engine
        w, h = g.current_level.grid_size
        return {
            "level_index": g.current_level_index,
            "total_levels": g.total_levels,
            "lives": g.lives,
            "max_lives": MAX_LIVES,
            "moves_used": g.moves_used,
            "max_moves": g.max_moves,
            "selector": g._selector_pos,
            "grid_size": (w, h),
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

    check_env(env, skip_render_check=False)

    obs, info = env.reset()
    obs, reward, term, trunc, info = env.step(0)
    frame = env.render()
    env.close()
