from __future__ import annotations

import binascii
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
    RenderableUserDisplay,
    Sprite,
)
from arcengine import (
    GameState as ArcPlayState,
)
from gymnasium import spaces

C_BG = 5
C_JUG_WALL = 10
C_JUG_TOP = 14
C_JUG_BOT = 8
C_WATER = 9
C_EMPTY_BODY = 5
C_SELECTED = 11
C_TARGET_WATER = 10
C_CURSOR = 0
C_BAR_STEP = 14
C_BAR_WARN = 12
C_BAR_CRIT = 8
C_BAR_EMPTY = 3

PADDING_COLOR = 3

BASE_ROW = 12
CAM_W, CAM_H = 16, 16

_JUG2_LX = [2, 10]
_JUG3_LX = [1, 6, 11]

_LEVELS = [
    {"caps": [3, 5], "target": 4, "max_steps": 200},
    {"caps": [4, 7], "target": 5, "max_steps": 220},
    {"caps": [3, 5, 7], "target": 6, "max_steps": 220},
    {"caps": [4, 6, 9], "target": 7, "max_steps": 280},
    {"caps": [3, 5, 8], "target": 6, "max_steps": 280},
]

NUM_LEVELS = len(_LEVELS)

ACTION_MAP = {
    "reset": GameAction.RESET,
    "up": GameAction.ACTION1,
    "down": GameAction.ACTION2,
    "left": GameAction.ACTION3,
    "right": GameAction.ACTION4,
    "select": GameAction.ACTION5,
    "undo": GameAction.ACTION7,
}

_PLAY_ACTIONS = frozenset(
    {
        GameAction.ACTION1,
        GameAction.ACTION2,
        GameAction.ACTION3,
        GameAction.ACTION4,
        GameAction.ACTION5,
    }
)


def _grayscale_frame_to_png(frame: np.ndarray) -> bytes:
    arr = np.asarray(frame, dtype=np.uint8)
    h, w = int(arr.shape[0]), int(arr.shape[1])
    raw = b"".join(b"\0" + arr[y, :].tobytes() for y in range(h))
    compressed = zlib.compress(raw, 9)

    def _chunk(tag: bytes, data: bytes) -> bytes:
        piece = tag + data
        return (
            struct.pack(">I", len(data))
            + piece
            + struct.pack(">I", binascii.crc32(piece) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", compressed)
        + _chunk(b"IEND", b"")
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


def _px(
    color: int,
    layer: int = 0,
    name: str = "s",
    tags: Optional[List[str]] = None,
    collidable: bool = False,
) -> Sprite:
    return Sprite(
        pixels=np.array([[color]], dtype=np.int32),
        name=name,
        visible=True,
        collidable=collidable,
        tags=tags or [],
        layer=layer,
    )


def _lx_for(n_jugs: int) -> List[int]:
    return _JUG3_LX if n_jugs == 3 else _JUG2_LX


class JugDisplay(RenderableUserDisplay):
    STEP_X, STEP_W = 0, 52
    STEP_BAR_TOP = 59
    STEP_BAR_H = 3
    BAR_Y = 62
    TARGET_X = 1
    TARGET_Y = 0

    def __init__(self, game: "Sr01") -> None:
        self._g = game
        self.max_steps = 0
        self.remaining = 0

    def set_limit(self, ms: int) -> None:
        self.max_steps = ms
        self.remaining = ms

    def tick(self) -> bool:
        if self.remaining > 0:
            self.remaining -= 1
        return self.remaining > 0

    def reset(self) -> None:
        self.remaining = self.max_steps

    def _render_target_blocks(self, frame: np.ndarray) -> None:
        h, w = frame.shape
        target = self._g._target
        if target <= 0:
            return
        block_w = 2
        gap = 1
        block_span = block_w + gap
        max_blocks = max(0, (w - self.TARGET_X + gap) // block_span)
        for i in range(min(target, max_blocks)):
            x0 = self.TARGET_X + i * block_span
            y0 = self.TARGET_Y
            if y0 + 1 < h and x0 + 1 < w:
                frame[y0 : y0 + 2, x0 : x0 + 2] = C_WATER

    def _render_capacity_bars(self, frame: np.ndarray) -> None:
        h, w = frame.shape
        if len(self._g._jugs) == 0:
            return
        n = len(self._g._jugs)
        group_width = n * 4
        start_x = max(0, w - group_width)
        for jug_idx in range(n):
            capacity = self._g._caps[jug_idx]
            if capacity <= 0:
                continue
            cx = start_x + 2 + jug_idx * 4
            bar_height = 9
            bar_top_y = 1
            bar_bottom_y = bar_top_y + bar_height - 1
            border_left = max(0, cx - 1)
            border_right = min(w - 1, cx + 1)
            border_top = max(0, bar_top_y - 1)
            border_bottom = min(h - 1, bar_bottom_y + 1)
            self._render_capacity_border(
                frame, h, w, border_left, border_right, border_top, border_bottom
            )
            self._render_capacity_fill(
                frame,
                h,
                w,
                capacity,
                bar_height,
                bar_top_y,
                bar_bottom_y,
                border_left + 1,
                border_right - 1,
            )

    def _render_capacity_border(
        self,
        frame: np.ndarray,
        h: int,
        w: int,
        left: int,
        right: int,
        top: int,
        bottom: int,
    ) -> None:
        for y in range(top, bottom + 1):
            if not (0 <= y < h):
                continue
            for x in range(left, right + 1):
                if not (0 <= x < w):
                    continue
                is_edge = y == top or y == bottom or x == left or x == right
                if is_edge:
                    frame[y, x] = 0

    def _render_capacity_fill(
        self,
        frame: np.ndarray,
        h: int,
        w: int,
        capacity: int,
        bar_height: int,
        bar_top_y: int,
        bar_bottom_y: int,
        inner_left: int,
        inner_right: int,
    ) -> None:
        if inner_left > inner_right:
            return
        for y in range(bar_top_y, bar_bottom_y + 1):
            if 0 <= y < h:
                for x in range(inner_left, inner_right + 1):
                    if 0 <= x < w:
                        frame[y, x] = C_BG
        filled = max(0, min(bar_height, capacity))
        for i in range(filled):
            y = bar_bottom_y - i
            if 0 <= y < h:
                for x in range(inner_left, inner_right + 1):
                    if 0 <= x < w:
                        frame[y, x] = C_WATER

    def _render_step_bar(self, frame: np.ndarray) -> None:
        h, w = frame.shape
        ratio = self.remaining / self.max_steps if self.max_steps > 0 else 0.0
        if ratio >= 0.5:
            step_color = C_BAR_STEP
        elif ratio >= 0.25:
            step_color = C_BAR_WARN
        else:
            step_color = C_BAR_CRIT
        filled = int(self.STEP_W * ratio)
        for i in range(self.STEP_W):
            x = self.STEP_X + i
            if x >= w:
                break
            color = step_color if i < filled else C_BAR_EMPTY
            for dy in range(self.STEP_BAR_H):
                y = self.STEP_BAR_TOP + dy
                if 0 <= y < h:
                    frame[y, x] = color

    def _render_lives(self, frame: np.ndarray) -> None:
        h, w = frame.shape
        for i in range(self._g.MAX_LIVES):
            lx = w - 8 + i * 3
            color = 8 if self._g._lives > i else C_BAR_EMPTY
            frame[self.BAR_Y : self.BAR_Y + 2, lx : lx + 2] = color

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        if self.max_steps == 0:
            return frame
        self._render_target_blocks(frame)
        self._render_capacity_bars(frame)
        self._render_step_bar(frame)
        self._render_lives(frame)
        return frame


class Sr01(ARCBaseGame):
    MAX_LIVES = 3

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self._rng = random.Random(seed)
        self._lives: int = self.MAX_LIVES
        self._jugs: List[int] = []
        self._target: int = 0
        self._caps: List[int] = []
        self._cursor_start: Tuple[int, int] = (7, 11)
        self._cursor_pool: list[tuple[int, int]] = []
        self._undo_stack: list[dict] = []

        lvls = [Level(sprites=[], grid_size=(CAM_W, CAM_H), data=d) for d in _LEVELS]
        self._hud = JugDisplay(self)
        gw, gh = lvls[0].grid_size or (CAM_W, CAM_H)
        cam = Camera(
            x=0,
            y=0,
            width=int(gw),
            height=int(gh),
            background=C_BG,
            letter_box=PADDING_COLOR,
            interfaces=[self._hud],
        )
        super().__init__(
            "sr01",
            lvls,
            cam,
            available_actions=[0, 1, 2, 3, 4, 5, 7],
        )

    def _get_valid_actions(self) -> list[ActionInput]:
        out = super()._get_valid_actions()
        if 7 in self._available_actions:
            out.append(ActionInput(id=GameAction.ACTION7))
        return out

    def reseed(self) -> None:
        self._rng = random.Random(self._seed)

    def on_set_level(self, level: Level) -> None:
        self._prepare_level_state()
        self._mount_level_sprites()

    def _prepare_level_state(self) -> None:
        idx = min(self._current_level_index, len(_LEVELS) - 1)
        gs = self.current_level.grid_size
        if gs is not None:
            self.camera.width, self.camera.height = int(gs[0]), int(gs[1])
        else:
            self.camera.width, self.camera.height = CAM_W, CAM_H
        data = _LEVELS[idx]
        self._caps = list(data["caps"])
        self._target = data["target"]
        self._pour_src: Optional[int] = None
        self._jugs = [0] * len(self._caps)
        self._lives = self.MAX_LIVES
        self._cursor_pool = self._build_cursor_pool()
        self._pick_next_cursor_start()
        self._undo_stack.clear()
        self._hud.set_limit(data["max_steps"])

    def _mount_level_sprites(self) -> None:
        self._make_jug_sprites()
        self._rebuild_jug_sprites()
        cx0, cy0 = self._cursor_start
        self._cursor = _px(C_CURSOR, layer=10, name="cursor", tags=["cursor"])
        self._cursor.set_position(cx0, cy0)
        self.current_level.add_sprite(self._cursor)

    def level_reset(self) -> None:
        super().level_reset()
        self._pick_next_cursor_start()
        if getattr(self, "_cursor", None) is not None:
            self._cursor.set_position(*self._cursor_start)

    def handle_reset(self) -> None:
        self._lives = self.MAX_LIVES
        super().handle_reset()

    def step(self) -> None:
        raw = self.action.id
        if isinstance(raw, int):
            try:
                act = GameAction.from_id(raw)
            except ValueError:
                if not self._hud.tick():
                    if self._trigger_life_loss():
                        return
                    self.complete_action()
                    return
                self.complete_action()
                return
        else:
            act = raw

        if act == GameAction.RESET:
            self.complete_action()
            return

        if act == GameAction.ACTION7:
            if self._undo_stack:
                snap = self._undo_stack.pop()
                self._jugs = list(snap["jugs"])
                self._pour_src = snap["pour_src"]
                self._cursor.set_position(snap["cx"], snap["cy"])
                self._rebuild_jug_sprites()
            if not self._hud.tick():
                if self._trigger_life_loss():
                    return
                self.complete_action()
                return
            self.complete_action()
            return
        if act not in _PLAY_ACTIONS:
            if not self._hud.tick():
                if self._trigger_life_loss():
                    return
                self.complete_action()
                return
            self.complete_action()
            return
        self._undo_stack.append(
            {
                "jugs": list(self._jugs),
                "pour_src": self._pour_src,
                "cx": self._cursor.x,
                "cy": self._cursor.y,
            }
        )
        cx, cy = self._cursor.x, self._cursor.y

        vw, vh = self.camera.width, self.camera.height
        if act == GameAction.ACTION1:
            if cy > 0:
                self._cursor.set_position(cx, cy - 1)
        elif act == GameAction.ACTION2:
            if cy < vh - 1:
                self._cursor.set_position(cx, cy + 1)
        elif act == GameAction.ACTION3:
            if cx > 0:
                self._cursor.set_position(cx - 1, cy)
        elif act == GameAction.ACTION4:
            if cx < vw - 1:
                self._cursor.set_position(cx + 1, cy)
        elif act == GameAction.ACTION5:
            won = self._handle_interact(cx, cy)
            if won:
                self._hud.tick()
                self.next_level()
                self.complete_action()
                return

        if not self._hud.tick():
            if self._trigger_life_loss():
                return
            self.complete_action()
            return

        self.complete_action()

    def _build_cursor_pool(self) -> list[tuple[int, int]]:
        vw, vh = int(self.camera.width), int(self.camera.height)
        cells = [(x, y) for x in range(vw) for y in range(vh)]
        picks = self._rng.sample(cells, min(4, len(cells)))
        return [(int(p[0]), int(p[1])) for p in picks]

    def _jug_lx(self, j: int) -> int:
        return _lx_for(len(self._caps))[j]

    def _top_row(self, j: int) -> int:
        return BASE_ROW - self._caps[j]

    def _top_zone_row(self, j: int) -> int:
        return BASE_ROW - self._caps[j] - 1

    def _is_in_jug_cols(self, x: int, j: int) -> bool:
        lx = self._jug_lx(j)
        return lx + 1 <= x <= lx + 2

    def _is_in_jug_or_walls(self, x: int, j: int) -> bool:
        lx = self._jug_lx(j)
        return lx <= x <= lx + 3

    def _hit_zone(self, x: int, y: int) -> Tuple[Optional[int], str]:
        for j in range(len(self._caps)):
            if self._is_in_jug_cols(x, j):
                if y == self._top_zone_row(j):
                    return j, "top"
            if self._is_in_jug_or_walls(x, j):
                if y == BASE_ROW:
                    return j, "bot"
                if self._top_row(j) <= y <= BASE_ROW - 1:
                    return j, "mid"
        return None, "none"

    def _recolor(self, sp: Sprite, c: int) -> None:
        sp.pixels = np.array([[c]], dtype=np.int32)

    def _pick_next_cursor_start(self) -> None:
        if not self._cursor_pool:
            self._cursor_pool = self._build_cursor_pool()
        self._cursor_start = self._rng.choice(self._cursor_pool)

    def _rebuild_jug_sprites(self) -> None:
        for j, cap in enumerate(self._caps):
            lx = self._jug_lx(j)
            top_r = self._top_row(j)
            is_sel = j == self._pour_src
            wall_c = C_SELECTED if is_sel else C_JUG_WALL
            hit_target = self._jugs[j] == self._target

            for dx in (1, 2):
                self._recolor(self._sp_top[j][dx - 1], C_JUG_TOP)

            for y in range(top_r, BASE_ROW):
                self._recolor(self._sp_lwall[j][y - top_r], wall_c)
                self._recolor(self._sp_rwall[j][y - top_r], wall_c)

            for dx in (0, 1, 2, 3):
                self._recolor(self._sp_bot[j][dx], C_JUG_BOT)

            water = self._jugs[j]
            water_c = C_TARGET_WATER if hit_target else C_WATER
            for row_idx in range(cap):
                filled = row_idx < water
                c = water_c if filled else C_EMPTY_BODY
                self._recolor(self._sp_body[j][row_idx][0], c)
                self._recolor(self._sp_body[j][row_idx][1], c)

    def _make_jug_sprites(self) -> None:
        self._sp_top: List[List[Sprite]] = []
        self._sp_lwall: List[List[Sprite]] = []
        self._sp_rwall: List[List[Sprite]] = []
        self._sp_bot: List[List[Sprite]] = []
        self._sp_body: List[List[List[Sprite]]] = []

        for j, cap in enumerate(self._caps):
            lx = self._jug_lx(j)
            top_r = self._top_row(j)
            tzr = self._top_zone_row(j)

            tops = []
            for dx in (1, 2):
                sp = _px(C_JUG_TOP, layer=1, name="top", tags=["top"])
                sp.set_position(lx + dx, tzr)
                self.current_level.add_sprite(sp)
                tops.append(sp)
            self._sp_top.append(tops)

            lwalls, rwalls = [], []
            for y in range(top_r, BASE_ROW):
                sp_l = _px(C_JUG_WALL, layer=1, name="wall", tags=["wall"])
                sp_r = _px(C_JUG_WALL, layer=1, name="wall", tags=["wall"])
                sp_l.set_position(lx, y)
                sp_r.set_position(lx + 3, y)
                self.current_level.add_sprite(sp_l)
                self.current_level.add_sprite(sp_r)
                lwalls.append(sp_l)
                rwalls.append(sp_r)
            self._sp_lwall.append(lwalls)
            self._sp_rwall.append(rwalls)

            bots = []
            for dx in range(4):
                sp = _px(C_JUG_BOT, layer=1, name="bot", tags=["bot"])
                sp.set_position(lx + dx, BASE_ROW)
                self.current_level.add_sprite(sp)
                bots.append(sp)
            self._sp_bot.append(bots)

            body = []
            for row_idx in range(cap):
                y = BASE_ROW - 1 - row_idx
                row_sps = []
                for dx in (1, 2):
                    sp = _px(C_EMPTY_BODY, layer=2, name="body", tags=["body"])
                    sp.set_position(lx + dx, y)
                    self.current_level.add_sprite(sp)
                    row_sps.append(sp)
                body.append(row_sps)
            self._sp_body.append(body)

    def _fill(self, j: int) -> None:
        self._jugs[j] = self._caps[j]

    def _empty(self, j: int) -> None:
        self._jugs[j] = 0

    def _pour(self, src: int, dst: int) -> None:
        space = self._caps[dst] - self._jugs[dst]
        amount = min(self._jugs[src], space)
        self._jugs[src] -= amount
        self._jugs[dst] += amount

    def _jug_has_target_volume(self) -> bool:
        return self._target in self._jugs

    def _restore_level(self) -> None:
        self._jugs = [0] * len(self._caps)
        self._pour_src = None
        self._rebuild_jug_sprites()
        self._undo_stack.clear()
        vw, vh = int(self.camera.width), int(self.camera.height)
        self._cursor_start = (
            self._rng.randrange(0, vw),
            self._rng.randrange(0, vh),
        )
        self._cursor.set_position(*self._cursor_start)

    def _trigger_life_loss(self) -> bool:
        self._lives -= 1
        if self._lives <= 0:
            self.lose()
            self.complete_action()
            return True
        self._restore_level()
        self._hud.reset()
        return False

    def _handle_interact(self, cx: int, cy: int) -> bool:
        j, zone = self._hit_zone(cx, cy)

        if zone == "top" and j is not None:
            self._fill(j)
            self._pour_src = None
            self._rebuild_jug_sprites()
            if self._jug_has_target_volume():
                return True
            return False

        if zone == "bot" and j is not None:
            self._empty(j)
            self._pour_src = None
            self._rebuild_jug_sprites()
            return False

        if zone == "mid" and j is not None:
            if self._pour_src is None:
                self._pour_src = j
                self._rebuild_jug_sprites()
            elif self._pour_src == j:
                self._pour_src = None
                self._rebuild_jug_sprites()
            else:
                self._pour(self._pour_src, j)
                self._pour_src = None
                self._rebuild_jug_sprites()
                if self._jug_has_target_volume():
                    return True

        return False


def _render_game_png(game: Sr01) -> bytes:
    try:
        frame = game.camera.render(game.current_level.get_sprites())
        return _grayscale_frame_to_png(frame)
    except (AttributeError, TypeError, ValueError, RuntimeError, MemoryError):
        return _grayscale_frame_to_png(np.zeros((1, 1), dtype=np.uint8))


ARC_PALETTE = (
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
)


class PuzzleEnvironment:
    def __init__(self, seed: int = 0) -> None:
        self._engine = Sr01(seed=seed)
        self.seed = seed
        self.turn = 0
        self._done = False
        self._last_action_was_reset = False

    def reset(self) -> GameState:
        g = self._engine
        self._done = False
        game_won = g._state == ArcPlayState.WIN
        if game_won or self.turn == 0 or self._last_action_was_reset:
            self._engine = Sr01(seed=self.seed)
            g = self._engine
        else:
            g.perform_action(ActionInput(id=GameAction.RESET))
        self.turn = 0
        self._last_action_was_reset = True
        return self._create_game_state()

    def get_actions(self) -> list[str]:
        if self._done:
            return ["reset"]
        g = self._engine
        if g._state in (ArcPlayState.WIN, ArcPlayState.GAME_OVER):
            return ["reset"]
        if g._lives <= 0:
            return ["reset"]
        cur = getattr(g, "_cursor", None)
        if cur is None:
            return ["reset", "up", "down", "left", "right", "select", "undo"]
        cx, cy = cur.x, cur.y
        vw, vh = int(g.camera.width), int(g.camera.height)
        out: list[str] = ["reset"]
        if cy > 0:
            out.append("up")
        if cy < vh - 1:
            out.append("down")
        if cx > 0:
            out.append("left")
        if cx < vw - 1:
            out.append("right")
        out.append("select")
        out.append("undo")
        return out

    def is_done(self) -> bool:
        g = self._engine
        if self._done:
            return True
        if g._state in (ArcPlayState.WIN, ArcPlayState.GAME_OVER):
            return True
        if getattr(g, "_terminated", False):
            return True
        return False

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError(f"Unsupported render mode: {mode}")
        if self._engine is None:
            raise RuntimeError("Environment has been closed")
        index_grid = np.asarray(
            self._engine.camera.render(self._engine.current_level.get_sprites()),
            dtype=np.int32,
        )
        h, w = int(index_grid.shape[0]), int(index_grid.shape[1])
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for idx, color in enumerate(ARC_PALETTE):
            rgb[index_grid == idx] = color
        return rgb

    def close(self) -> None:
        self._engine = None

    def step(self, action: str) -> StepResult:
        raw = action.strip()
        key = raw.upper()
        if raw.lower() in ACTION_MAP:
            mapped = ACTION_MAP[raw.lower()]
        elif key in ACTION_MAP:
            mapped = ACTION_MAP[key]
        else:
            return StepResult(
                state=self._create_game_state(),
                reward=0.0,
                done=self._done,
                info={"error": f"Invalid action: {action}"},
            )

        g = self._engine

        if mapped == GameAction.RESET:
            game_won = g._state == ArcPlayState.WIN
            full_restart = game_won or self.turn == 0 or self._last_action_was_reset
            state = self.reset()
            return StepResult(
                state=state,
                reward=0.0,
                done=False,
                info={"action": "reset", "full_restart": full_restart},
            )

        self._last_action_was_reset = False
        lives_before = g._lives
        level_before = g.level_index

        g.perform_action(ActionInput(id=mapped))
        self.turn += 1

        reward = 0.0
        done = False
        info: dict = {
            "lives": g._lives,
            "level": g.level_index + 1,
            "steps_remaining": g._hud.remaining,
            "max_steps": g._hud.max_steps,
        }

        if g._lives < lives_before:
            if g._lives <= 0:
                info["event"] = "game_over"
                done = True
            else:
                info["event"] = "life_lost"
        elif g.level_index != level_before:
            reward = 1.0 / NUM_LEVELS
            last_level_idx = NUM_LEVELS - 1
            if level_before == last_level_idx:
                info["event"] = "game_complete"
                done = True
            else:
                info["event"] = "level_complete"
        elif g._state == ArcPlayState.WIN:
            reward = 1.0 / NUM_LEVELS
            done = True
            info["event"] = "game_complete"
        elif g._state == ArcPlayState.GAME_OVER:
            done = True
            info["event"] = "game_over"

        if getattr(g, "_terminated", False):
            done = True

        self._done = done

        return StepResult(
            state=self._create_game_state(),
            reward=reward,
            done=done,
            info=info,
        )

    def _create_game_state(self) -> GameState:
        g = self._engine
        terminal = (
            self._done
            or g._state in (ArcPlayState.WIN, ArcPlayState.GAME_OVER)
            or getattr(g, "_terminated", False)
        )
        acts = self.get_actions()
        text = _format_text_observation(g) + "\n\nValid actions: " + ", ".join(acts)
        png = _render_game_png(g)
        return GameState(
            text_observation=text,
            image_observation=png,
            valid_actions=None if terminal else acts,
            turn=self.turn,
            metadata={
                "total_levels": len(self._engine._levels),
                "level": g.level_index + 1,
                "lives": g._lives,
                "steps_remaining": g._hud.remaining if hasattr(g, "_hud") else 0,
                "max_steps": g._hud.max_steps if hasattr(g, "_hud") else 0,
            },
        )


def _format_text_observation(game: Sr01) -> str:
    cur = getattr(game, "_cursor", None)
    if cur is not None:
        cx, cy = cur.x, cur.y
    else:
        cx, cy = 0, 0
    vw, vh = int(game.camera.width), int(game.camera.height)
    n = len(game._caps)
    lxs = _lx_for(n)
    lines = [
        f"grid {vw}x{vh} cells x=0..{vw - 1} y=0..{vh - 1} y down",
        f"level_index={game._current_level_index} level_count={len(_LEVELS)} "
        f"target_volume={game._target}",
        f"lives={game._lives} "
        f"steps_remaining={game._hud.remaining} max_steps={game._hud.max_steps}",
        f"cursor x={cx} y={cy}",
        f"pour_source_jug_index={game._pour_src}",
    ]
    for j in range(n):
        cap = game._caps[j]
        lx = lxs[j]
        top_r = BASE_ROW - cap
        tzr = BASE_ROW - cap - 1
        wv = game._jugs[j]
        lines.append(
            f"jug{j}: water={wv} capacity={cap} "
            f"fill_zone cell x={lx + 1}-{lx + 2} y={tzr} "
            f"empty_zone row y={BASE_ROW} x={lx}-{lx + 3} "
            f"body_and_walls pour_select x={lx}-{lx + 3} y={top_r}-{BASE_ROW - 1}"
        )
    lines.append(
        "ACTION1=up ACTION2=down ACTION3=left ACTION4=right ACTION5=select "
        "ACTION7=undo (costs one step)"
    )
    return "\n".join(lines)


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
