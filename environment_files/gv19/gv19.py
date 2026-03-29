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
    "undo": GameAction.ACTION7,
}

COLOR_NAMES = {
    0: "Blue",
    1: "Red",
    2: "Green",
    3: "Yellow",
    4: "Magenta",
    5: "Orange",
}

CAM_W, CAM_H = 48, 48
GRID_SIZE = (48, 48)

BACKGROUND_COLOR = 5
PADDING_COLOR = 5

C_EMPTY = 4
C_WALL = 3
C_WALL_BORDER = 2

BALL_COLORS = [9, 8, 14, 11, 6, 12]

C_BAR_FULL = 14
C_BAR_EMPTY = 4

LEVEL_DEFS = [
    {
        "grid_w": 6,
        "grid_h": 6,
        "max_flips": 50,
        "walls": [
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 5),
            (1, 0),
            (1, 5),
            (2, 0),
            (2, 5),
            (3, 0),
            (3, 5),
            (4, 0),
            (4, 5),
            (5, 0),
            (5, 1),
            (5, 2),
            (5, 3),
            (5, 4),
            (5, 5),
            (2, 3),
            (3, 1),
            (4, 4),
        ],
        "balls": [
            (1, 1, 0),
            (1, 4, 1),
            (4, 2, 2),
        ],
        "targets": [
            (4, 1, 0),
            (3, 2, 1),
            (1, 1, 2),
        ],
    },
    {
        "grid_w": 7,
        "grid_h": 7,
        "max_flips": 50,
        "walls": [
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 5),
            (0, 6),
            (1, 0),
            (1, 6),
            (2, 0),
            (2, 6),
            (3, 0),
            (3, 6),
            (4, 0),
            (4, 6),
            (5, 0),
            (5, 6),
            (6, 0),
            (6, 1),
            (6, 2),
            (6, 3),
            (6, 4),
            (6, 5),
            (6, 6),
            (2, 3),
            (3, 1),
            (3, 5),
            (4, 3),
            (5, 2),
        ],
        "balls": [
            (1, 1, 0),
            (1, 5, 1),
            (5, 1, 2),
            (5, 5, 3),
        ],
        "targets": [
            (5, 3, 0),
            (2, 4, 1),
            (4, 1, 2),
            (5, 4, 3),
        ],
    },
    {
        "grid_w": 8,
        "grid_h": 8,
        "max_flips": 50,
        "walls": [
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 5),
            (0, 6),
            (0, 7),
            (1, 0),
            (1, 7),
            (2, 0),
            (2, 7),
            (3, 0),
            (3, 7),
            (4, 0),
            (4, 7),
            (5, 0),
            (5, 7),
            (6, 0),
            (6, 7),
            (7, 0),
            (7, 1),
            (7, 2),
            (7, 3),
            (7, 4),
            (7, 5),
            (7, 6),
            (7, 7),
            (2, 2),
            (2, 5),
            (3, 3),
            (3, 4),
            (4, 1),
            (4, 6),
            (5, 3),
            (5, 5),
            (6, 2),
        ],
        "balls": [
            (1, 1, 0),
            (1, 6, 1),
            (3, 2, 2),
            (6, 1, 3),
            (6, 6, 4),
        ],
        "targets": [
            (3, 2, 0),
            (1, 1, 1),
            (4, 2, 2),
            (5, 1, 3),
            (5, 6, 4),
        ],
    },
    {
        "grid_w": 9,
        "grid_h": 9,
        "max_flips": 50,
        "walls": [
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 5),
            (0, 6),
            (0, 7),
            (0, 8),
            (1, 0),
            (1, 8),
            (2, 0),
            (2, 8),
            (3, 0),
            (3, 8),
            (4, 0),
            (4, 8),
            (5, 0),
            (5, 8),
            (6, 0),
            (6, 8),
            (7, 0),
            (7, 8),
            (8, 0),
            (8, 1),
            (8, 2),
            (8, 3),
            (8, 4),
            (8, 5),
            (8, 6),
            (8, 7),
            (8, 8),
            (2, 2),
            (2, 6),
            (3, 4),
            (4, 2),
            (4, 6),
            (5, 4),
            (6, 2),
            (6, 6),
            (7, 4),
        ],
        "balls": [
            (1, 1, 0),
            (1, 7, 1),
            (3, 3, 2),
            (5, 5, 3),
            (7, 2, 4),
            (7, 7, 5),
        ],
        "targets": [
            (1, 3, 0),
            (1, 1, 1),
            (2, 3, 2),
            (7, 6, 3),
            (2, 1, 4),
            (1, 5, 5),
        ],
    },
    {
        "grid_w": 10,
        "grid_h": 10,
        "max_flips": 50,
        "walls": [
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 5),
            (0, 6),
            (0, 7),
            (0, 8),
            (0, 9),
            (1, 0),
            (1, 9),
            (2, 0),
            (2, 9),
            (3, 0),
            (3, 9),
            (4, 0),
            (4, 9),
            (5, 0),
            (5, 9),
            (6, 0),
            (6, 9),
            (7, 0),
            (7, 9),
            (8, 0),
            (8, 9),
            (9, 0),
            (9, 1),
            (9, 2),
            (9, 3),
            (9, 4),
            (9, 5),
            (9, 6),
            (9, 7),
            (9, 8),
            (9, 9),
            (2, 3),
            (2, 5),
            (3, 1),
            (3, 4),
            (3, 8),
            (4, 4),
            (5, 2),
            (5, 4),
            (6, 1),
            (7, 3),
            (7, 8),
            (8, 5),
            (8, 6),
        ],
        "balls": [
            (1, 7, 0),
            (6, 3, 1),
            (7, 6, 2),
            (4, 8, 3),
            (7, 2, 4),
            (1, 1, 5),
        ],
        "targets": [
            (8, 7, 0),
            (5, 1, 1),
            (2, 1, 2),
            (4, 2, 3),
            (8, 1, 4),
            (8, 2, 5),
        ],
    },
]


def _cell_size(grid_w, grid_h):
    usable_h = CAM_H - 5
    return max(2, min((CAM_W - 2) // grid_w, usable_h // grid_h))


def _make_wall_set(ldef):
    return set(ldef["walls"])


class Gv19(ARCBaseGame):
    MAX_LIVES = 3

    def __init__(self, seed: int = 0) -> None:
        self._lives: int = self.MAX_LIVES
        self._ldef = None
        self._wall_set = set()
        self._balls = []
        self._targets = []
        self._flips_used = 0
        self._max_flips = 5
        self._history = []
        self._cs = 4
        self._origin_x = 0
        self._origin_y = 0
        self._level_cleared = False

        camera = Camera(0, 0, CAM_W, CAM_H, BACKGROUND_COLOR, PADDING_COLOR, [])

        levels = [
            Level(sprites=[], grid_size=GRID_SIZE, name=f"Level {i + 1}")
            for i in range(len(LEVEL_DEFS))
        ]

        super().__init__("gv19", levels, camera, available_actions=[0, 1, 2, 3, 4, 7])

    def on_set_level(self, level: Level) -> None:
        idx = self.level_index
        ldef = LEVEL_DEFS[idx]
        self._ldef = ldef
        self._wall_set = _make_wall_set(ldef)
        self._max_flips = ldef["max_flips"]
        self._flips_used = 0
        self._history = []

        self._balls = [[r, c, ci] for (r, c, ci) in ldef["balls"]]
        self._targets = list(ldef["targets"])

        gw, gh = ldef["grid_w"], ldef["grid_h"]
        cs = _cell_size(gw, gh)
        self._cs = cs
        self._origin_x = (CAM_W - gw * cs) // 2
        self._origin_y = (CAM_H - 5 - gh * cs) // 2

        self._rebuild_sprites(level)

    def handle_reset(self) -> None:
        self._lives = self.MAX_LIVES
        super().handle_reset()

    def _rebuild_sprites(self, level: Level) -> None:
        level.remove_all_sprites()
        self._draw_grid(level)
        self._draw_targets(level)
        self._draw_balls(level)
        self._draw_flip_bar(level)

    def _draw_grid(self, level: Level) -> None:
        ldef = self._ldef
        gw, gh = ldef["grid_w"], ldef["grid_h"]
        cs = self._cs
        ox, oy = self._origin_x, self._origin_y

        for r in range(gh):
            for c in range(gw):
                is_wall = (r, c) in self._wall_set
                if is_wall:
                    pixels = []
                    for py in range(cs):
                        row = []
                        for px in range(cs):
                            if py == 0 or px == 0:
                                row.append(C_WALL_BORDER)
                            else:
                                row.append(C_WALL)
                        pixels.append(row)
                else:
                    pixels = [[C_EMPTY] * cs for _ in range(cs)]

                level.add_sprite(
                    Sprite(
                        pixels=pixels,
                        name=f"grid_{r}_{c}",
                        x=ox + c * cs,
                        y=oy + r * cs,
                        layer=0,
                        tags=["grid"],
                    )
                )

    def _draw_targets(self, level: Level) -> None:
        cs = self._cs
        ox, oy = self._origin_x, self._origin_y

        for r, c, ci in self._targets:
            tcolor = BALL_COLORS[ci]
            pixels = []
            center = (cs - 1) / 2.0
            outer_r = cs / 2.0 - 0.3
            inner_r = outer_r - 1.5
            for py in range(cs):
                row = []
                for px in range(cs):
                    dy = py - center
                    dx = px - center
                    dist_sq = dx * dx + dy * dy
                    if dist_sq <= outer_r * outer_r and dist_sq >= inner_r * inner_r:
                        row.append(tcolor)
                    else:
                        row.append(C_EMPTY)
                pixels.append(row)

            level.add_sprite(
                Sprite(
                    pixels=pixels,
                    name=f"target_{r}_{c}_{ci}",
                    x=ox + c * cs,
                    y=oy + r * cs,
                    layer=1,
                    tags=["target"],
                )
            )

    def _draw_balls(self, level: Level) -> None:
        cs = self._cs
        ox, oy = self._origin_x, self._origin_y

        for i, (r, c, ci) in enumerate(self._balls):
            bcolor = BALL_COLORS[ci]
            pixels = []
            center = (cs - 1) / 2.0
            radius = cs / 2.0 - 0.3
            for py in range(cs):
                row = []
                for px in range(cs):
                    dy = py - center
                    dx = px - center
                    if dx * dx + dy * dy <= radius * radius:
                        row.append(bcolor)
                    else:
                        row.append(-1)
                pixels.append(row)

            level.add_sprite(
                Sprite(
                    pixels=pixels,
                    name=f"ball_{i}_{ci}",
                    x=ox + c * cs,
                    y=oy + r * cs,
                    layer=2,
                    tags=["ball"],
                )
            )

    def _draw_flip_bar(self, level: Level) -> None:
        lives_area_w = self.MAX_LIVES * 4 + 1
        bar_x = 2 + lives_area_w
        bar_w = CAM_W - bar_x - 2
        bar_h = 3
        bar_y = CAM_H - 4

        remaining = max(0, self._max_flips - self._flips_used)
        filled_w = (
            max(0, (remaining * bar_w) // self._max_flips) if self._max_flips > 0 else 0
        )
        empty_w = bar_w - filled_w

        if filled_w > 0:
            filled_pixels = [[C_BAR_FULL] * filled_w for _ in range(bar_h)]
            level.add_sprite(
                Sprite(
                    pixels=filled_pixels,
                    name="bar_filled",
                    x=bar_x,
                    y=bar_y,
                    layer=3,
                    tags=["bar"],
                )
            )

        if empty_w > 0:
            empty_pixels = [[C_BAR_EMPTY] * empty_w for _ in range(bar_h)]
            level.add_sprite(
                Sprite(
                    pixels=empty_pixels,
                    name="bar_empty",
                    x=bar_x + filled_w,
                    y=bar_y,
                    layer=3,
                    tags=["bar"],
                )
            )

        for i in range(self.MAX_LIVES):
            lx = 2 + i * 4
            color = 8 if self._lives > i else 4
            level.add_sprite(
                Sprite(
                    pixels=[[color, color], [color, color]],
                    name=f"life_{i}",
                    x=lx,
                    y=bar_y,
                    layer=3,
                    tags=["lives"],
                )
            )

    def _save_state(self) -> None:
        balls_copy = [[r, c, ci] for (r, c, ci) in self._balls]
        self._history.append(balls_copy)

    def _restore_state(self) -> None:
        if self._history:
            self._balls = self._history.pop()

    def _flip_gravity(self, dr: int, dc: int) -> bool:
        ldef = self._ldef
        gw, gh = ldef["grid_w"], ldef["grid_h"]
        wall_set = self._wall_set
        n = len(self._balls)

        any_moved = False

        moved = True
        while moved:
            moved = False

            ball_positions = set()
            for b in self._balls:
                ball_positions.add((b[0], b[1]))

            indices = list(range(n))
            if dr == 1:
                indices.sort(key=lambda i: -self._balls[i][0])
            elif dr == -1:
                indices.sort(key=lambda i: self._balls[i][0])
            elif dc == 1:
                indices.sort(key=lambda i: -self._balls[i][1])
            elif dc == -1:
                indices.sort(key=lambda i: self._balls[i][1])

            for i in indices:
                br, bc = self._balls[i][0], self._balls[i][1]
                nr, nc = br + dr, bc + dc

                if nr < 0 or nr >= gh or nc < 0 or nc >= gw:
                    continue

                if (nr, nc) in wall_set:
                    continue

                ball_positions.discard((br, bc))
                if (nr, nc) in ball_positions:
                    ball_positions.add((br, bc))
                    continue

                self._balls[i][0] = nr
                self._balls[i][1] = nc
                ball_positions.add((nr, nc))
                moved = True
                any_moved = True

        return any_moved

    def _check_win(self) -> bool:
        ball_map = {}
        for r, c, ci in self._balls:
            ball_map[(r, c, ci)] = True

        for tr, tc, tci in self._targets:
            if (tr, tc, tci) not in ball_map:
                return False
        return True

    def step(self) -> None:
        action = self.action
        level = self.current_level
        self._level_cleared = False

        dr, dc = 0, 0
        is_flip = False
        is_undo = False

        if action.id == GameAction.ACTION1:
            dr, dc = -1, 0
            is_flip = True
        elif action.id == GameAction.ACTION2:
            dr, dc = 1, 0
            is_flip = True
        elif action.id == GameAction.ACTION3:
            dr, dc = 0, -1
            is_flip = True
        elif action.id == GameAction.ACTION4:
            dr, dc = 0, 1
            is_flip = True
        elif action.id == GameAction.ACTION7:
            is_undo = True
        else:
            self.complete_action()
            return

        if is_undo:
            if self._history:
                self._restore_state()
            self._flips_used += 1
            self._rebuild_sprites(level)

            if self._flips_used >= self._max_flips:
                self._lives -= 1
                self._rebuild_sprites(level)
                if self._lives <= 0:
                    self.lose()
                    self.complete_action()
                    return
                self.set_level(self.level_index)
                self.complete_action()
                return
            self.complete_action()
            return

        if is_flip:
            self._save_state()

            any_moved = self._flip_gravity(dr, dc)
            self._flips_used += 1

            if not any_moved:
                self._history.pop()
                self._rebuild_sprites(level)

                if self._flips_used >= self._max_flips:
                    self._lives -= 1
                    self._rebuild_sprites(level)
                    if self._lives <= 0:
                        self.lose()
                        self.complete_action()
                        return
                    self.set_level(self.level_index)
                    self.complete_action()
                    return

                self.complete_action()
                return

            self._rebuild_sprites(level)

            if self._check_win():
                self._level_cleared = True
                self._lives = self.MAX_LIVES
                self.next_level()
                self.complete_action()
                return

            if self._flips_used >= self._max_flips:
                self._lives -= 1
                self._rebuild_sprites(level)
                if self._lives <= 0:
                    self.lose()
                    self.complete_action()
                    return
                self.set_level(self.level_index)
                self.complete_action()
                return

        self.complete_action()


class PuzzleEnvironment:
    def __init__(self, seed: int = 0) -> None:
        self._engine = Gv19(seed=seed)
        self._done = False
        self._reward = 0.0
        self._total_turns = 0
        self._total_levels = len(self._engine._levels)
        self._reward_per_level = 1.0 / self._total_levels
        self._cumulative_reward = 0.0
        self._last_action_was_reset = False

    def _build_text_observation(self) -> str:
        g = self._engine
        ldef = g._ldef
        gw, gh = ldef["grid_w"], ldef["grid_h"]
        total_levels = len(g._levels)
        level_num = g.level_index + 1

        lines = [
            f"Level {level_num}/{total_levels} | Grid: {gw}x{gh} | "
            f"Flips: {g._flips_used}/{g._max_flips} | "
            f"Lives: {g._lives}/{g.MAX_LIVES}"
        ]

        for r, c, ci in g._balls:
            lines.append(f"  Ball {COLOR_NAMES[ci]}: row={r} col={c}")

        for r, c, ci in g._targets:
            lines.append(f"  Target {COLOR_NAMES[ci]}: row={r} col={c}")

        ball_map = {(r, c): ci for r, c, ci in g._balls}
        target_map = {(r, c): ci for r, c, ci in g._targets}
        grid_lines = []
        for r in range(gh):
            row_chars = []
            for c in range(gw):
                if (r, c) in g._wall_set:
                    row_chars.append("#")
                elif (r, c) in ball_map:
                    row_chars.append(str(ball_map[(r, c)]))
                elif (r, c) in target_map:
                    row_chars.append("*")
                else:
                    row_chars.append(".")
            grid_lines.append("".join(row_chars))
        lines.append("Grid:")
        lines.extend(grid_lines)

        lines.append(f"  Walls: {len(g._wall_set)} | Undo depth: {len(g._history)}")

        return "\n".join(lines)

    @staticmethod
    def _encode_png(rgb: np.ndarray) -> bytes:
        h, w = rgb.shape[0], rgb.shape[1]
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

        ihdr_data = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
        return (
            b"\x89PNG\r\n\x1a\n"
            + _chunk(b"IHDR", ihdr_data)
            + _chunk(b"IDAT", compressed)
            + _chunk(b"IEND", b"")
        )

    def _build_image_observation(self) -> Optional[bytes]:
        g = self._engine
        index_grid = g.camera.render(g.current_level.get_sprites())
        if index_grid is not None:
            h, w = index_grid.shape[:2]
            rgb = np.zeros((h, w, 3), dtype=np.uint8)
            for idx, color in enumerate(ARC_PALETTE):
                mask = index_grid == idx
                rgb[mask] = color
            return self._encode_png(rgb)
        return None

    def _build_state(self) -> GameState:
        g = self._engine
        return GameState(
            text_observation=self._build_text_observation(),
            image_observation=self._build_image_observation(),
            valid_actions=self.get_actions(),
            turn=self._total_turns,
            metadata={
                "total_levels": len(self._engine._levels),
                "level_index": g.level_index,
                "grid_w": g._ldef["grid_w"],
                "grid_h": g._ldef["grid_h"],
                "lives": g._lives,
                "flips_used": g._flips_used,
                "max_flips": g._max_flips,
                "balls": [[r, c, ci] for r, c, ci in g._balls],
                "targets": list(g._targets),
                "undo_depth": len(g._history),
                "available_actions": self.get_actions()
                if not self._done
                else ["reset"],
            },
        )

    def reset(self) -> GameState:
        g = self._engine
        g.perform_action(ActionInput(id=GameAction.RESET))

        self._done = False
        self._reward = 0.0
        self._cumulative_reward = 0.0
        self._total_turns = 0
        return self._build_state()

    def get_actions(self) -> List[str]:
        if self._done:
            return ["reset"]
        return list(ACTION_MAP.keys())

    def is_done(self) -> bool:
        return self._done

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError(f"Unsupported render mode: {mode}")
        index_grid = self._engine.camera.render(
            self._engine.current_level.get_sprites()
        )
        h, w = index_grid.shape[:2]
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for idx, color in enumerate(ARC_PALETTE):
            mask = index_grid == idx
            rgb[mask] = color
        return rgb

    def close(self) -> None:
        self._engine = None

    def step(self, action: str) -> StepResult:
        if self._done:
            return StepResult(
                state=self._build_state(),
                reward=0.0,
                done=True,
                info={"error": "episode already done, call reset()"},
            )

        if action == "reset":
            if self._last_action_was_reset:
                state = self.reset()
                self._last_action_was_reset = False
                return StepResult(
                    state=state,
                    reward=0.0,
                    done=False,
                    info={"reset": True, "consecutive_reset": True},
                )
            g = self._engine
            g._lives = g.MAX_LIVES
            g.set_level(g.level_index)
            self._done = False
            self._reward = 0.0
            self._total_turns = 0
            self._last_action_was_reset = True
            return StepResult(
                state=self._build_state(),
                reward=0.0,
                done=False,
                info={"reset": True},
            )

        if action not in ACTION_MAP:
            return StepResult(
                state=self._build_state(),
                reward=0.0,
                done=False,
                info={"error": f"invalid action: {action}"},
            )

        self._total_turns += 1
        self._last_action_was_reset = False

        g = self._engine
        prev_lives = g._lives
        prev_level = g.level_index

        game_action = ACTION_MAP[action]
        g.perform_action(ActionInput(id=game_action))

        won = g._level_cleared
        game_over = g._state == EngineGameState.GAME_OVER
        level_advanced = g.level_index > prev_level
        life_lost = g._lives < prev_lives

        if won:
            self._reward = self._reward_per_level
            self._cumulative_reward += self._reward
            if not level_advanced:
                self._done = True
        elif game_over:
            self._reward = 0.0
            self._done = True
        else:
            self._reward = 0.0

        return StepResult(
            state=self._build_state(),
            reward=self._reward,
            done=self._done,
            info={
                "won": won,
                "game_over": game_over,
                "level_advanced": level_advanced,
                "life_lost": life_lost,
                "cumulative_reward": self._cumulative_reward,
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
