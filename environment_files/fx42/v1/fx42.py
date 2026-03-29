import copy
import io
import random
import struct
import zlib
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from arcengine import (
    ActionInput,
    ARCBaseGame,
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


WHITE = 0
OFFWHITE = 1
LTGREY = 2
GREY = 3
DKGREY = 4
BLACK = 5
MAGENTA = 6
PINK = 7
RED = 8
BLUE = 9
LTBLUE = 10
YELLOW = 11
ORANGE = 12
MAROON = 13
GREEN = 14
PURPLE = 15

BACKGROUND_COLOR = WHITE
PADDING_COLOR = WHITE

MAX_LIVES = 3

LEVEL_COLORS = {
    1: [RED, BLUE],
    2: [RED, BLUE, YELLOW],
    3: [RED, BLUE],
    4: [RED, BLUE, YELLOW],
    5: [RED, BLUE, YELLOW],
}

LEVEL_GRID_SIZES = {
    1: (3, 3),
    2: (3, 3),
    3: (4, 4),
    4: (4, 4),
    5: (5, 5),
}

LEVEL_CLICK_BUDGETS = {
    1: 50,
    2: 80,
    3: 150,
    4: 250,
    5: 350,
}

LEVEL_DEPTH = {
    1: 1,
    2: 2,
    3: 2,
    4: 2,
    5: 3,
}

TILE_SIZE = 8


def _make_solid_tile(color):
    return [[color] * TILE_SIZE for _ in range(TILE_SIZE)]


sprites = {
    "base_tile": Sprite(
        pixels=_make_solid_tile(RED),
        name="base_tile",
        visible=True,
        collidable=False,
        tags=["tile"],
        layer=1,
    ),
    "flash_pixel": Sprite(
        pixels=[[WHITE]],
        name="flash_pixel",
        visible=False,
        collidable=False,
        tags=["overlay"],
        layer=10,
    ),
}

TILE_VISUAL_SIZE = TILE_SIZE

levels = [
    Level(
        sprites=[
            sprites["flash_pixel"].clone().set_position(0, 0),
        ],
        grid_size=(64, 64),
        data={
            "level_num": 1,
            "grid_rows": 3,
            "grid_cols": 3,
            "click_budget": 50,
        },
    ),
    Level(
        sprites=[
            sprites["flash_pixel"].clone().set_position(0, 0),
        ],
        grid_size=(64, 64),
        data={
            "level_num": 2,
            "grid_rows": 3,
            "grid_cols": 3,
            "click_budget": 80,
        },
    ),
    Level(
        sprites=[
            sprites["flash_pixel"].clone().set_position(0, 0),
        ],
        grid_size=(64, 64),
        data={
            "level_num": 3,
            "grid_rows": 4,
            "grid_cols": 4,
            "click_budget": 150,
        },
    ),
    Level(
        sprites=[
            sprites["flash_pixel"].clone().set_position(0, 0),
        ],
        grid_size=(64, 64),
        data={
            "level_num": 4,
            "grid_rows": 4,
            "grid_cols": 4,
            "click_budget": 250,
        },
    ),
    Level(
        sprites=[
            sprites["flash_pixel"].clone().set_position(0, 0),
        ],
        grid_size=(64, 64),
        data={
            "level_num": 5,
            "grid_rows": 5,
            "grid_cols": 5,
            "click_budget": 350,
        },
    ),
]


class FluxHUD(RenderableUserDisplay):
    def __init__(self, game):
        self.game = game

    def _draw_grid_lines(self, frame, height, width):
        g = self.game
        if g.grid_rows <= 0 or g.grid_cols <= 0:
            return
        ox, oy = g._grid_offset()
        total_w = g.grid_cols * TILE_SIZE
        total_h = g.grid_rows * TILE_SIZE
        for r in range(g.grid_rows + 1):
            py = oy + r * TILE_SIZE
            if 0 <= py < height:
                for x in range(total_w + 1):
                    px = ox + x
                    if 0 <= px < width:
                        frame[py, px] = BLACK
        for c in range(g.grid_cols + 1):
            px = ox + c * TILE_SIZE
            if 0 <= px < width:
                for y in range(total_h + 1):
                    py = oy + y
                    if 0 <= py < height:
                        frame[py, px] = BLACK

    def _draw_target_preview(self, frame, height, width):
        g = self.game
        if g.target_grid is None or len(g.target_grid) == 0:
            return
        rows = len(g.target_grid)
        cols = len(g.target_grid[0]) if rows > 0 else 0
        colors = LEVEL_COLORS.get(g.level_num, LEVEL_COLORS[1])
        ppt = 2
        bx, by = 1, 1
        for r in range(rows * ppt + 2):
            for c in range(cols * ppt + 2):
                fr = by + r
                fc = bx + c
                if 0 <= fr < height and 0 <= fc < width:
                    if (
                        r == 0
                        or r == rows * ppt + 1
                        or c == 0
                        or c == cols * ppt + 1
                    ):
                        frame[fr, fc] = GREY
                    else:
                        tr = (r - 1) // ppt
                        tc = (c - 1) // ppt
                        if 0 <= tr < rows and 0 <= tc < cols:
                            ci = g.target_grid[tr][tc]
                            frame[fr, fc] = colors[ci % len(colors)]

    def _draw_budget_bar(self, frame, height, width):
        g = self.game
        if g.max_clicks <= 0:
            return
        bar_start_x = 32
        bar_end_x = 62
        bar_width = bar_end_x - bar_start_x
        ratio = g.click_budget / g.max_clicks
        filled = int(bar_width * ratio)
        for bar_y in [1, 2]:
            for x in range(bar_width):
                if bar_start_x + x < width:
                    if x < filled:
                        frame[bar_y, bar_start_x + x] = GREEN
                    else:
                        frame[bar_y, bar_start_x + x] = GREY

    def _draw_lives(self, frame, height, width):
        g = self.game
        for i in range(MAX_LIVES):
            lx = width - 3 - i * 3
            ly = 4
            color = BLUE if i < g.lives else GREY
            for dr in range(2):
                for dc in range(2):
                    if 0 <= ly + dr < height and 0 <= lx + dc < width:
                        frame[ly + dr, lx + dc] = color

    def _draw_level_indicator(self, frame, height, width):
        g = self.game
        for lv in range(5):
            lx = 28 + lv * 3
            ly = 5
            color = WHITE if (lv + 1) <= g.level_num else GREY
            if 0 <= ly < height and 0 <= lx < width:
                frame[ly, lx] = color
                if lx + 1 < width:
                    frame[ly, lx + 1] = color

    def _draw_cursor(self, frame, height, width):
        g = self.game
        if g.grid_rows <= 0 or g.grid_cols <= 0:
            return
        ox, oy = g._grid_offset()
        cr = g.cursor_row
        cc = g.cursor_col
        px = ox + cc * TILE_SIZE
        py = oy + cr * TILE_SIZE
        cursor_color = MAGENTA
        for i in range(TILE_SIZE):
            if 0 <= py < height and 0 <= px + i < width:
                frame[py, px + i] = cursor_color
            by = py + TILE_SIZE - 1
            if 0 <= by < height and 0 <= px + i < width:
                frame[by, px + i] = cursor_color
            if 0 <= py + i < height and 0 <= px < width:
                frame[py + i, px] = cursor_color
            bx = px + TILE_SIZE - 1
            if 0 <= py + i < height and 0 <= bx < width:
                frame[py + i, bx] = cursor_color

    def render_interface(self, frame):
        height, width = frame.shape[:2]
        self._draw_grid_lines(frame, height, width)
        self._draw_target_preview(frame, height, width)
        self._draw_budget_bar(frame, height, width)
        self._draw_lives(frame, height, width)
        self._draw_level_indicator(frame, height, width)
        self._draw_cursor(frame, height, width)
        return frame


def _orthogonal_nbrs(r, c, rows, cols):
    result = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            result.append((nr, nc))
    return result


def _diagonal_nbrs(r, c, rows, cols):
    result = []
    for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            result.append((nr, nc))
    return result


def _all_8_nbrs(r, c, rows, cols):
    return _orthogonal_nbrs(r, c, rows, cols) + _diagonal_nbrs(r, c, rows, cols)


def apply_rule(level_num, grid, r, c, rows, cols):
    new = copy.deepcopy(grid)
    n = len(LEVEL_COLORS[level_num])

    if level_num == 1:
        new[r][c] = (grid[r][c] + 1) % n
        for nr, nc in _orthogonal_nbrs(r, c, rows, cols):
            new[nr][nc] = (grid[nr][nc] - 1) % n

    elif level_num == 2:
        new[r][c] = (grid[r][c] + 1) % n
        for nr, nc in _diagonal_nbrs(r, c, rows, cols):
            new[nr][nc] = (grid[nr][nc] + 1) % n

    elif level_num == 3:
        new[r][c] = (grid[r][c] + 1) % n
        for nr, nc in _orthogonal_nbrs(r, c, rows, cols):
            new[nr][nc] = (grid[nr][nc] + 1) % n

    elif level_num == 4:
        new[r][c] = (grid[r][c] + 1) % n
        for nr, nc in _orthogonal_nbrs(r, c, rows, cols):
            new[nr][nc] = (grid[nr][nc] + 1) % n
        for nr, nc in _diagonal_nbrs(r, c, rows, cols):
            new[nr][nc] = (grid[nr][nc] - 1) % n

    elif level_num == 5:
        new[r][c] = (grid[r][c] + 1) % n
        for nr, nc in _all_8_nbrs(r, c, rows, cols):
            new[nr][nc] = (grid[nr][nc] + 1) % n

    return new


def generate_random_puzzle(level_num, rng):
    rows, cols = LEVEL_GRID_SIZES[level_num]
    n = len(LEVEL_COLORS[level_num])
    depth = LEVEL_DEPTH[level_num]

    initial = [[rng.randint(0, n - 1) for _ in range(cols)] for _ in range(rows)]

    grid = copy.deepcopy(initial)
    clicks = []
    for _ in range(depth):
        cr = rng.randint(0, rows - 1)
        cc = rng.randint(0, cols - 1)
        clicks.append((cr, cc))
        grid = apply_rule(level_num, grid, cr, cc, rows, cols)

    target = grid

    if initial == target:
        cr = rng.randint(0, rows - 1)
        cc = rng.randint(0, cols - 1)
        clicks.append((cr, cc))
        target = apply_rule(level_num, target, cr, cc, rows, cols)

    return initial, target, clicks


class Fx42(ARCBaseGame):
    def __init__(self, seed: int = 0):
        self.lives = MAX_LIVES
        self.level_num = 1

        self._rng = random.Random(seed)

        self.grid = []
        self.initial_grid = []
        self.target_grid = []
        self.click_budget = 0
        self.max_clicks = 0
        self.grid_rows = 0
        self.grid_cols = 0
        self.tile_sprites = []

        self.cursor_row = 0
        self.cursor_col = 0

        self._engine_snapshot = None
        self._engine_can_undo = False
        self._game_over = False

        self.hud = FluxHUD(self)

        camera = Camera(0, 0, 64, 64, BACKGROUND_COLOR, PADDING_COLOR, [self.hud])
        super().__init__(
            "fx42",
            levels,
            camera,
            available_actions=[0, 1, 2, 3, 4, 5, 7],
        )

    def on_set_level(self, level):
        self.level_num = self.current_level.get_data("level_num")
        self.grid_rows = self.current_level.get_data("grid_rows")
        self.grid_cols = self.current_level.get_data("grid_cols")
        self.max_clicks = self.current_level.get_data("click_budget")
        self.click_budget = self.max_clicks
        self.lives = MAX_LIVES

        overlay_sprites = self.current_level.get_sprites_by_name("flash_pixel")
        if overlay_sprites:
            self.flash_overlay = overlay_sprites[0]
            self.flash_overlay.set_visible(False)

        self._generate_new_puzzle()

        self.cursor_row = 0
        self.cursor_col = 0

        self._engine_snapshot = None
        self._engine_can_undo = False
        self._game_over = False

        self._create_tile_sprites()
        self._sync_all_tiles()

    def _generate_new_puzzle(self):
        initial, target, _ = generate_random_puzzle(self.level_num, self._rng)
        self.initial_grid = initial
        self.target_grid = target
        self.grid = copy.deepcopy(self.initial_grid)

    def _grid_offset(self):
        total_w = self.grid_cols * TILE_SIZE
        total_h = self.grid_rows * TILE_SIZE
        ox = (64 - total_w) // 2
        oy = (64 - total_h) // 2 + 4
        return ox, oy

    def _create_tile_sprites(self):
        if self.tile_sprites:
            for row in self.tile_sprites:
                for ts in row:
                    if ts is not None:
                        try:
                            self.current_level.remove_sprite(ts)
                        except (ValueError, AttributeError):
                            pass
        self.tile_sprites = []

        ox, oy = self._grid_offset()
        for r in range(self.grid_rows):
            row_sprites = []
            for c in range(self.grid_cols):
                ts = sprites["base_tile"].clone()
                ts.set_position(ox + c * TILE_SIZE, oy + r * TILE_SIZE)
                self.current_level.add_sprite(ts)
                row_sprites.append(ts)
            self.tile_sprites.append(row_sprites)

    def _color_for_index(self, ci):
        colors = LEVEL_COLORS.get(self.level_num, LEVEL_COLORS[1])
        return colors[ci % len(colors)]

    def _sync_tile(self, r, c):
        if 0 <= r < self.grid_rows and 0 <= c < self.grid_cols:
            ts = self.tile_sprites[r][c]
            arc_color = self._color_for_index(self.grid[r][c])
            ts.color_remap(None, arc_color)

    def _sync_all_tiles(self):
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                self._sync_tile(r, c)

    def _flash_tile(self, r, c):
        if 0 <= r < self.grid_rows and 0 <= c < self.grid_cols:
            ts = self.tile_sprites[r][c]
            ts.color_remap(None, WHITE)

    def _apply_rule(self, r, c):
        old_grid = copy.deepcopy(self.grid)
        new_grid = apply_rule(
            self.level_num, self.grid, r, c, self.grid_rows, self.grid_cols
        )
        self.grid = new_grid

        changed = []
        for rr in range(self.grid_rows):
            for cc in range(self.grid_cols):
                if new_grid[rr][cc] != old_grid[rr][cc]:
                    changed.append((rr, cc))
        return changed

    def _check_win(self):
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                if self.grid[r][c] != self.target_grid[r][c]:
                    return False
        return True

    def _engine_save_snapshot(self):
        self._engine_snapshot = {
            "grid": copy.deepcopy(self.grid),
            "cursor_row": self.cursor_row,
            "cursor_col": self.cursor_col,
            "click_budget": self.click_budget,
        }

    def _engine_restore_snapshot(self):
        snap = self._engine_snapshot
        self.grid = snap["grid"]
        self.cursor_row = snap["cursor_row"]
        self.cursor_col = snap["cursor_col"]
        self.click_budget = snap["click_budget"]
        self._sync_all_tiles()

    def _reset_current_level(self):
        self.grid = copy.deepcopy(self.initial_grid)
        self.click_budget = self.max_clicks
        self.cursor_row = 0
        self.cursor_col = 0

        self._engine_snapshot = None
        self._engine_can_undo = False

        self._sync_all_tiles()

    def full_reset(self):
        self._game_over = False
        self.lives = MAX_LIVES
        self._current_level_index = 0
        self.on_set_level(levels[0])

    def _handle_death(self):
        self.lives -= 1
        if self.lives <= 0:
            self._game_over = True
            self.lose()
            self.lives = MAX_LIVES
            return True
        else:
            self.death_phase = 1
            return False

    def step(self) -> None:
        action_id = self.action.id.value

        if action_id == 0:
            self._reset_current_level()
            self._engine_can_undo = False
            self._engine_snapshot = None
            self.complete_action()
            return

        if action_id == 7:
            if self._engine_can_undo and self._engine_snapshot is not None:
                budget_before = self.click_budget
                self._engine_restore_snapshot()
                self._engine_can_undo = False
                self._engine_snapshot = None
                self.click_budget = budget_before - 1
            else:
                self.click_budget -= 1
            if self.click_budget <= 0:
                game_over = self._handle_death()
                if game_over:
                    self.complete_action()
                    return
                else:
                    self._reset_current_level()
                    self.complete_action()
                    return
            self.complete_action()
            return

        self._engine_save_snapshot()

        if action_id in (1, 2, 3, 4):
            if action_id == 1:
                self.cursor_row = max(0, self.cursor_row - 1)
            elif action_id == 2:
                self.cursor_row = min(self.grid_rows - 1, self.cursor_row + 1)
            elif action_id == 3:
                self.cursor_col = max(0, self.cursor_col - 1)
            elif action_id == 4:
                self.cursor_col = min(self.grid_cols - 1, self.cursor_col + 1)

            self.click_budget -= 1
            if self.click_budget <= 0:
                self._engine_can_undo = False
                self._engine_snapshot = None
                game_over = self._handle_death()
                if game_over:
                    self.complete_action()
                    return
                else:
                    self._reset_current_level()
                    self.complete_action()
                    return
            self._engine_can_undo = True
            self.complete_action()
            return

        if action_id == 5:
            r, c = self.cursor_row, self.cursor_col
            self._apply_rule(r, c)
            self._sync_all_tiles()
            self.click_budget -= 1
            self._engine_can_undo = True
            if self._check_win():
                self.next_level()
                self.complete_action()
                return
            if self.click_budget <= 0:
                self._engine_can_undo = False
                self._engine_snapshot = None
                game_over = self._handle_death()
                if game_over:
                    self.complete_action()
                    return
                else:
                    self._reset_current_level()
                    self.complete_action()
                    return
            self.complete_action()
            return

        self.complete_action()


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

_COLOR_CHAR = {
    RED: "R",
    BLUE: "B",
    YELLOW: "Y",
    WHITE: "W",
    GREEN: "G",
    MAGENTA: "M",
    MAROON: "N",
    PURPLE: "P",
    ORANGE: "O",
}


class PuzzleEnvironment:
    ACTION_MAP: Dict[str, int] = {
        "reset": 0,
        "up": 1,
        "down": 2,
        "left": 3,
        "right": 4,
        "select": 5,
        "undo": 7,

    }

    _INT_TO_ACTION = {
        1: GameAction.ACTION1,
        2: GameAction.ACTION2,
        3: GameAction.ACTION3,
        4: GameAction.ACTION4,
        5: GameAction.ACTION5,
        7: GameAction.ACTION7,
    }

    def __init__(self, seed: int = 0) -> None:
        self._engine = Fx42(seed=seed)
        self._done = False
        self._last_action_was_reset = False
        self._total_turns = 0
        self._total_levels = len(levels)

    @staticmethod
    def _frame_to_png(frame: np.ndarray) -> bytes:
        h, w = frame.shape[:2]
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for idx, color in enumerate(ARC_PALETTE):
            mask = frame == idx
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

    def _render_frame(self) -> np.ndarray:
        e = self._engine
        return e.camera.render(e.current_level.get_sprites())

    def _build_text_observation(self) -> str:
        e = self._engine
        colors = LEVEL_COLORS.get(e.level_num, LEVEL_COLORS[1])

        grid_lines = []
        for r in range(e.grid_rows):
            row_chars = []
            for c in range(e.grid_cols):
                ci = e.grid[r][c]
                actual_color = colors[ci % len(colors)]
                ch = _COLOR_CHAR.get(actual_color, "?")
                if r == e.cursor_row and c == e.cursor_col:
                    row_chars.append("[" + ch + "]")
                else:
                    row_chars.append(" " + ch + " ")
            grid_lines.append("".join(row_chars))

        target_lines = []
        for r in range(e.grid_rows):
            row_chars = []
            for c in range(e.grid_cols):
                ci = e.target_grid[r][c]
                actual_color = colors[ci % len(colors)]
                ch = _COLOR_CHAR.get(actual_color, "?")
                row_chars.append(" " + ch + " ")
            target_lines.append("".join(row_chars))

        header = (
            f"Level {e._current_level_index + 1}/{len(levels)}"
            f" | Clicks: {e.click_budget}/{e.max_clicks}"
            f" | Lives: {e.lives}"
        )
        rules = "Move cursor with arrows. Select applies a color rule at cursor. Match grid to target."
        return (
            header
            + "\n"
            + rules
            + "\nGrid:\n"
            + "\n".join(grid_lines)
            + "\nTarget:\n"
            + "\n".join(target_lines)
        )

    def _build_state(self, done: bool = False) -> GameState:
        e = self._engine
        frame = self._render_frame()
        image_bytes = self._frame_to_png(frame)

        valid_actions = list(self.ACTION_MAP.keys()) if not done else None

        return GameState(
            text_observation=self._build_text_observation(),
            image_observation=image_bytes,
            valid_actions=valid_actions,
            turn=self._total_turns,
            metadata={
                "cursor_position": [e.cursor_row, e.cursor_col],
                "grid": copy.deepcopy(e.grid),
                "target_grid": copy.deepcopy(e.target_grid),
                "click_budget": e.click_budget,
                "max_clicks": e.max_clicks,
                "lives": e.lives,
                "level_index": e._current_level_index,
                "total_levels": len(levels),
                "grid_size": [e.grid_rows, e.grid_cols],
            },
        )

    def reset(self) -> GameState:
        e = self._engine

        if self._done or self._last_action_was_reset:
            e.full_reset()
        else:
            e._game_over = False

        reset_input = ActionInput(id=GameAction.RESET)
        e.perform_action(reset_input)

        self._done = False
        self._total_turns = 0
        self._last_action_was_reset = True
        return self._build_state()

    def is_done(self) -> bool:
        return self._done

    def get_actions(self) -> List[str]:
        if self._done:
            return ["reset"]
        return list(self.ACTION_MAP.keys())

    def step(self, action: str) -> StepResult:
        e = self._engine

        if action == "reset":
            state = self.reset()
            return StepResult(
                state=state, reward=0.0, done=False, info={"action": "reset"}
            )

        if action not in self.ACTION_MAP:
            return StepResult(
                state=self._build_state(done=self._done),
                reward=0.0,
                done=self._done,
                info={"action": action, "reason": "unknown_action"},
            )

        self._last_action_was_reset = False
        self._total_turns += 1

        game_action_id = self.ACTION_MAP[action]
        game_action = self._INT_TO_ACTION[game_action_id]
        info: Dict[str, Any] = {"action": action}

        action_input = ActionInput(id=game_action)
        level_before = e._current_level_index
        result = e.perform_action(action_input)
        level_after = e._current_level_index

        game_won = result.state.value == "WIN"
        game_over = result.state.value == "GAME_OVER"

        if game_won:
            self._done = True
            info["reason"] = "game_complete"
            reward = 1.0 / self._total_levels
            return StepResult(
                state=self._build_state(done=True),
                reward=reward,
                done=True,
                info=info,
            )

        if game_over or e._game_over:
            self._done = True
            info["reason"] = "death"
            return StepResult(
                state=self._build_state(done=True),
                reward=0.0,
                done=True,
                info=info,
            )

        if level_after > level_before:
            info["reason"] = "level_complete"
            reward = 1.0 / self._total_levels
            return StepResult(
                state=self._build_state(done=False),
                reward=reward,
                done=False,
                info=info,
            )

        return StepResult(
            state=self._build_state(done=False),
            reward=0.0,
            done=False,
            info=info,
        )

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError(f"Unsupported render mode: {mode}")
        index_grid = self._render_frame()
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


if __name__ == "__main__":
    from gymnasium.utils.env_checker import check_env

    env = ArcGameEnv(seed=42, render_mode="rgb_array")
    check_env(env.unwrapped, skip_render_check=True)

    obs, info = env.reset()

    mask = env.action_mask()
    valid_indices = np.where(mask == 1)[0]
    if len(valid_indices) > 0:
        obs, reward, term, trunc, info = env.step(valid_indices[0])

    env.close()
