from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import struct
import zlib

import numpy as np
import gymnasium as gym
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


FRAME_SIZE = 64
UI_ROWS = 2
PLAY_ROWS = FRAME_SIZE - UI_ROWS

GRID_SIZES = [8, 8, 10, 12]
TILE_SIZES = [PLAY_ROWS // g for g in GRID_SIZES]
OFFSETS = [
    ((FRAME_SIZE - t * g) // 2, (PLAY_ROWS - t * g) // 2)
    for t, g in zip(TILE_SIZES, GRID_SIZES)
]

GRAVITY_DIRS = {
    "down": (0, 1),
    "up": (0, -1),
    "left": (-1, 0),
    "right": (1, 0),
}

WALK_DIRS = {
    "down": ((-1, 0), (1, 0)),
    "up": ((-1, 0), (1, 0)),
    "left": ((0, -1), (0, 1)),
    "right": ((0, -1), (0, 1)),
}

BACKGROUND_COLOR = 0
PADDING_COLOR = 4


class MoveDisplay(RenderableUserDisplay):
    MAX_LIVES = 3
    C_BAR_FILL = 11
    C_BAR_EMPTY = 5
    C_LIFE = 8
    C_LIFE_EMPTY = 3

    def __init__(self, game: "Gv01") -> None:
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

        frame_w = frame.shape[1]
        frame_h = frame.shape[0]
        ui_start = frame_h - 2

        frame[ui_start:frame_h, :] = self.C_BAR_EMPTY

        bar_width = int(frame_w * 0.7)
        filled = int(bar_width * self.remaining / self.max_moves)
        for x in range(filled):
            frame[ui_start:frame_h, x] = self.C_BAR_FILL

        lives = getattr(self._game, "_lives", self.MAX_LIVES)
        block_w = 2
        gap = 2
        total_w = (block_w * self.MAX_LIVES) + (gap * (self.MAX_LIVES - 1))
        offset = ((frame_w - bar_width) - total_w) // 2

        for i in range(self.MAX_LIVES):
            x0 = bar_width + offset + i * (block_w + gap)
            x1 = x0 + block_w
            color = self.C_LIFE if i < lives else self.C_LIFE_EMPTY
            if x1 <= frame_w:
                frame[ui_start:frame_h, x0:x1] = color

        return frame


def _make_tile(color: int, size: int) -> list:
    row = [color] * size
    return [row[:] for _ in range(size)]


def _make_sprites(tile_size: int) -> dict:
    return {
        "player": Sprite(
            pixels=_make_tile(12, tile_size),
            name="player",
            visible=True,
            collidable=False,
        ),
        "stone": Sprite(
            pixels=_make_tile(9, tile_size),
            name="stone",
            visible=True,
            collidable=False,
        ),
        "target": Sprite(
            pixels=_make_tile(11, tile_size),
            name="target",
            visible=True,
            collidable=False,
        ),
        "landed": Sprite(
            pixels=_make_tile(14, tile_size),
            name="landed",
            visible=True,
            collidable=False,
        ),
        "wall": Sprite(
            pixels=_make_tile(5, tile_size), name="wall", visible=True, collidable=False
        ),
        "trap": Sprite(
            pixels=_make_tile(8, tile_size), name="trap", visible=True, collidable=False
        ),
    }


LEVEL_DEFS = [
    {
        "grid": [
            "########",
            "#......#",
            "##.....#",
            "#......#",
            "#.....##",
            "##.....#",
            "#......#",
            "########",
        ],
        "player_start": [4, 1],
        "stones": [[1, 1]],
        "targets": [[6, 6]],
        "gravity": "down",
        "moves": 50,
        "rule": "no_double_same",
    },
    {
        "grid": [
            "########",
            "#......#",
            "#......#",
            "#.#.#..#",
            "#..#...#",
            "#......#",
            "#......#",
            "########",
        ],
        "player_start": [1, 1],
        "stones": [[2, 1], [5, 1]],
        "targets": [[2, 6], [6, 6]],
        "gravity": "down",
        "moves": 40,
        "rule": "trap_cell",
        "trap": (5, 3),
    },
    {
        "grid": [
            "##########",
            "#........#",
            "#........#",
            "#.#....#.#",
            "#........#",
            "#........#",
            "#.#....#.#",
            "#........#",
            "#........#",
            "##########",
        ],
        "player_start": [5, 1],
        "stones": [[2, 1], [5, 1], [8, 1]],
        "targets": [[8, 8], [8, 7], [1, 6]],
        "gravity": "down",
        "moves": 70,
        "rule": "max_up",
        "max_up": 3,
    },
    {
        "grid": [
            "############",
            "#..........#",
            "#..........#",
            "#.#......#.#",
            "#..........#",
            "#..........#",
            "#.#......#.#",
            "#..........#",
            "#..........#",
            "#..........#",
            "#..........#",
            "############",
        ],
        "player_start": [6, 1],
        "stones": [[2, 1], [4, 1], [6, 1], [8, 1], [10, 1]],
        "targets": [[1, 6], [1, 8], [1, 9], [1, 7], [1, 10]],
        "gravity": "down",
        "moves": 125,
        "rule": "no_repeat_direction",
    },
]

levels = [
    Level(sprites=[], grid_size=(FRAME_SIZE, FRAME_SIZE)),
    Level(sprites=[], grid_size=(FRAME_SIZE, FRAME_SIZE)),
    Level(sprites=[], grid_size=(FRAME_SIZE, FRAME_SIZE)),
    Level(sprites=[], grid_size=(FRAME_SIZE, FRAME_SIZE)),
]


class Gv01(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._seed = seed

        self._gravity: str = "down"
        self._player: list = [0, 0]
        self._stones: list = []
        self._targets: list = []
        self._landed: set = set()
        self._walls: set = set()
        self._trap: tuple = ()
        self._player_before_gravity: list = []

        self._moves_remaining: int = 0
        self._max_moves: int = 0
        self._game_over: bool = False
        self._lives: int = 3

        self._last_gravity: str = ""
        self._consec_count: int = 0
        self._up_uses: int = 0
        self._prev_gravity: str = ""

        self._init_gravity: str = "down"
        self._init_player: list = [0, 0]
        self._init_stones: list = []
        self._init_targets: list = []
        self._init_walls: set = set()
        self._init_max_moves: int = 0

        self._history: list = []

        self._move_display = MoveDisplay(self)
        camera = Camera(
            0, 0, 64, 64, BACKGROUND_COLOR, PADDING_COLOR, [self._move_display]
        )
        super().__init__(
            game_id="gv01",
            levels=levels,
            camera=camera,
            available_actions=[0, 1, 2, 3, 4, 5, 7],
        )

    def on_set_level(self, level: Level) -> None:
        self._game_over = False
        self._lives = 3
        self._history = []
        self._load_level_def(self.level_index)
        self._save_initial_state()
        self._render_level(level)
        self._move_display.set_limit(self._max_moves)

    def _load_level_def(self, idx: int) -> None:
        defn = LEVEL_DEFS[idx]
        self._gravity = defn["gravity"]
        self._walls = set()
        self._targets = [list(t) for t in defn["targets"]]
        self._stones = [list(s) for s in defn["stones"]]
        self._landed = set()
        self._max_moves = defn["moves"]
        self._moves_remaining = self._max_moves
        self._player = list(defn["player_start"])
        self._trap = defn.get("trap", ())

        for row_y, row_str in enumerate(defn["grid"]):
            for col_x, ch in enumerate(row_str):
                if ch == "#":
                    self._walls.add((col_x, row_y))

        self._landed = self._calc_landed()

        self._last_gravity = ""
        self._consec_count = 0
        self._up_uses = 0
        self._prev_gravity = ""

    def _save_initial_state(self) -> None:
        self._init_gravity = self._gravity
        self._init_player = self._player.copy()
        self._init_stones = [s.copy() for s in self._stones]
        self._init_targets = [t.copy() for t in self._targets]
        self._init_walls = set(self._walls)
        self._init_max_moves = self._max_moves

    def _grid_size(self) -> int:
        return GRID_SIZES[self.level_index]

    def _in_bounds(self, x: int, y: int) -> bool:
        g = self._grid_size()
        return 0 <= x < g and 0 <= y < g

    def _stone_at(self, x: int, y: int, exclude: int = -1) -> bool:
        return any(
            i != exclude and s[0] == x and s[1] == y for i, s in enumerate(self._stones)
        )

    def _is_solid(self, x: int, y: int, exclude_stone: int = -1) -> bool:
        if (x, y) in self._walls:
            return True
        return self._stone_at(x, y, exclude=exclude_stone)

    def _fall_to_rest(
        self, x: int, y: int, dx: int, dy: int, exclude_stone: int = -1, snap_at=None
    ) -> tuple:
        cx, cy = x, y
        while True:
            nx, ny = cx + dx, cy + dy
            if not self._in_bounds(nx, ny):
                break
            if self._is_solid(nx, ny, exclude_stone=exclude_stone):
                break
            cx, cy = nx, ny
            if snap_at is not None and (cx, cy) == snap_at:
                break
        return cx, cy

    def _calc_landed(self) -> set:
        return {
            i
            for i, stone in enumerate(self._stones)
            if i < len(self._targets) and stone == self._targets[i]
        }

    def _apply_gravity_all(self) -> None:
        dx, dy = GRAVITY_DIRS[self._gravity]

        def fall_priority(i: int) -> int:
            sx, sy = self._stones[i]
            return -(sx * dx + sy * dy)

        for i in sorted(range(len(self._stones)), key=fall_priority):
            sx, sy = self._stones[i]
            snap = tuple(self._targets[i]) if i < len(self._targets) else None
            fx, fy = self._fall_to_rest(sx, sy, dx, dy, exclude_stone=i, snap_at=snap)
            self._stones[i] = [fx, fy]

        self._landed = self._calc_landed()

        px, py = self._player
        fpx, fpy = self._fall_to_rest(px, py, dx, dy, exclude_stone=-1)
        self._player = [fpx, fpy]

    def _walk_player(self, wx: int, wy: int) -> bool:
        px, py = self._player
        nx, ny = px + wx, py + wy
        if not self._in_bounds(nx, ny) or self._is_solid(nx, ny):
            return False
        self._player = [nx, ny]
        dx, dy = GRAVITY_DIRS[self._gravity]
        fpx, fpy = self._fall_to_rest(nx, ny, dx, dy, exclude_stone=-1)
        self._player = [fpx, fpy]
        return True

    def _all_landed(self) -> bool:
        return len(self._landed) >= len(self._stones)

    def _save_state(self) -> None:
        self._history.append(
            {
                "gravity": self._gravity,
                "player": self._player.copy(),
                "stones": [s.copy() for s in self._stones],
                "landed": set(self._landed),
                "last_gravity": self._last_gravity,
                "consec_count": self._consec_count,
                "up_uses": self._up_uses,
                "prev_gravity": self._prev_gravity,
            }
        )

    def _do_undo(self) -> None:
        if not self._history:
            return
        snap = self._history.pop()
        self._gravity = snap["gravity"]
        self._player = snap["player"]
        self._stones = snap["stones"]
        self._landed = snap["landed"]
        self._last_gravity = snap["last_gravity"]
        self._consec_count = snap["consec_count"]
        self._up_uses = snap["up_uses"]
        self._prev_gravity = snap["prev_gravity"]

    def _restart_level(self) -> None:
        self._lives -= 1
        if self._lives <= 0:
            self._game_over = True
            self.lose()
            return
        self._game_over = False
        self._history = []
        self._gravity = self._init_gravity
        self._player = self._init_player.copy()
        self._stones = [s.copy() for s in self._init_stones]
        self._targets = [t.copy() for t in self._init_targets]
        self._walls = set(self._init_walls)
        self._landed = set()
        self._max_moves = self._init_max_moves
        self._moves_remaining = self._max_moves
        self._move_display.set_limit(self._max_moves)
        self._last_gravity = ""
        self._consec_count = 0
        self._up_uses = 0
        self._prev_gravity = ""
        self._render_level(self.current_level)

    def _render_level(self, level: Level) -> None:
        for sp in list(level._sprites):
            level.remove_sprite(sp)

        idx = self.level_index
        tile = TILE_SIZES[idx]
        ox, oy = OFFSETS[idx]
        g = self._grid_size()
        spr = _make_sprites(tile)

        stone_pos = {tuple(s) for s in self._stones}
        target_pos = {tuple(t) for t in self._targets}
        landed_pos = {tuple(self._stones[i]) for i in self._landed}

        for y in range(g):
            for x in range(g):
                cell = (x, y)
                px_x = ox + x * tile
                px_y = oy + y * tile
                if cell in self._walls:
                    level.add_sprite(spr["wall"].clone().set_position(px_x, px_y))
                elif cell == self._trap:
                    level.add_sprite(spr["trap"].clone().set_position(px_x, px_y))
                elif cell in landed_pos:
                    level.add_sprite(spr["landed"].clone().set_position(px_x, px_y))
                elif cell in target_pos:
                    level.add_sprite(spr["target"].clone().set_position(px_x, px_y))
                elif cell in stone_pos:
                    level.add_sprite(spr["stone"].clone().set_position(px_x, px_y))

        px_x = ox + self._player[0] * tile
        px_y = oy + self._player[1] * tile
        level.add_sprite(spr["player"].clone().set_position(px_x, px_y))

    def _player_crossed_trap(self, direction: str) -> bool:
        if not self._trap or not self._player_before_gravity:
            return False
        dx, dy = GRAVITY_DIRS[direction]
        cx, cy = self._player_before_gravity
        while True:
            cx, cy = cx + dx, cy + dy
            if not self._in_bounds(cx, cy):
                break
            if (cx, cy) == self._trap:
                return True
            if [cx, cy] == self._player:
                break
        return False

    def _check_rule_after_gravity(self, direction: str) -> bool:
        defn = LEVEL_DEFS[self.level_index]
        rule = defn.get("rule", "")

        if rule == "no_double_same":
            if direction == self._last_gravity:
                self._consec_count += 1
            else:
                self._consec_count = 1
            self._last_gravity = direction
            if self._consec_count >= 2:
                return True

        elif rule == "trap_cell":
            if self._trap and self._player_crossed_trap(direction):
                return True

        elif rule == "max_up":
            if direction == "up":
                self._up_uses += 1
            if self._up_uses >= defn.get("max_up", 3):
                return True

        elif rule == "no_repeat_direction":
            if self._prev_gravity and direction == self._prev_gravity:
                return True
            self._prev_gravity = direction

        return False

    def step(self) -> None:
        if not self.action:
            self.complete_action()
            return

        if self.action.id == GameAction.RESET:
            self.complete_action()
            return

        if self._game_over:
            self.complete_action()
            return

        action_id = self.action.id

        if action_id == GameAction.ACTION7:
            if self._history:
                self._do_undo()
            self._moves_remaining -= 1
            self._move_display.decrement()
            if self._moves_remaining <= 0 and not self._all_landed():
                self._restart_level()
            else:
                self._render_level(self.current_level)
            self.complete_action()
            return

        self._save_state()

        acted = False
        gravity_used = ""

        if action_id == GameAction.ACTION1:
            self._gravity = "up"
            self._player_before_gravity = self._player.copy()
            self._apply_gravity_all()
            acted = True
            gravity_used = "up"
        elif action_id == GameAction.ACTION2:
            self._gravity = "down"
            self._player_before_gravity = self._player.copy()
            self._apply_gravity_all()
            acted = True
            gravity_used = "down"
        elif action_id == GameAction.ACTION3:
            self._gravity = "left"
            self._player_before_gravity = self._player.copy()
            self._apply_gravity_all()
            acted = True
            gravity_used = "left"
        elif action_id == GameAction.ACTION4:
            self._gravity = "right"
            self._player_before_gravity = self._player.copy()
            self._apply_gravity_all()
            acted = True
            gravity_used = "right"
        elif action_id == GameAction.ACTION5:
            walk_neg, _ = WALK_DIRS[self._gravity]
            acted = self._walk_player(walk_neg[0], walk_neg[1])

        if not acted:
            if self._history:
                self._history.pop()

        if acted:
            self._moves_remaining -= 1
            self._move_display.decrement()

        if gravity_used and self._check_rule_after_gravity(gravity_used):
            self._render_level(self.current_level)
            self._restart_level()
            self.complete_action()
            return

        if self._moves_remaining <= 0 and not self._all_landed():
            self._render_level(self.current_level)
            self._restart_level()
            self.complete_action()
            return

        self._render_level(self.current_level)

        if self._all_landed():
            self.next_level()

        self.complete_action()


class PuzzleEnvironment:
    _ACTION_MAP: Dict[str, int] = {
        "reset": 0,
        "up": 1,
        "down": 2,
        "left": 3,
        "right": 4,
        "select": 5,
        "undo": 7,
    }

    _VALID_ACTIONS: List[str] = [
        "reset",
        "up",
        "down",
        "left",
        "right",
        "select",
        "undo",
    ]

    ARC_PALETTE = [
        (255, 255, 255),
        (204, 204, 204),
        (153, 153, 153),
        (102, 102, 102),
        (51, 51, 51),
        (0, 0, 0),
        (229, 58, 163),
        (255, 123, 204),
        (249, 60, 49),
        (30, 147, 255),
        (136, 216, 241),
        (255, 220, 0),
        (255, 133, 27),
        (146, 18, 49),
        (79, 204, 48),
        (163, 86, 208),
    ]

    _CELL_CHAR: Dict[str, str] = {
        "wall": "#",
        "player": "P",
        "stone": "S",
        "target": "T",
        "landed": "L",
        "trap": "X",
    }

    def __init__(self, seed: int = 0) -> None:
        self._engine = Gv01(seed=seed)
        self._total_turns = 0
        self._last_action_was_reset = False
        self._game_won = False

    def _build_text_obs(self) -> str:
        e = self._engine
        g = e._grid_size()
        grid: List[List[str]] = [["." for _ in range(g)] for _ in range(g)]

        for wx, wy in e._walls:
            if 0 <= wx < g and 0 <= wy < g:
                grid[wy][wx] = "#"

        if e._trap:
            tx, ty = e._trap
            if 0 <= tx < g and 0 <= ty < g:
                grid[ty][tx] = "X"

        for i, t in enumerate(e._targets):
            if i not in e._landed:
                if 0 <= t[0] < g and 0 <= t[1] < g:
                    grid[t[1]][t[0]] = "T"

        for i, s in enumerate(e._stones):
            if i in e._landed:
                grid[s[1]][s[0]] = "L"
            else:
                grid[s[1]][s[0]] = "S"

        grid[e._player[1]][e._player[0]] = "P"

        grid_text = "\n".join("".join(row) for row in grid)
        header = "Level:{}/{} Lives:{} Moves:{}/{}".format(
            e.level_index + 1,
            len(e._levels),
            e._lives,
            e._moves_remaining,
            e._max_moves,
        )
        return header + "\n" + grid_text

    @staticmethod
    def _encode_png(rgb: np.ndarray) -> bytes:
        h, w, _ = rgb.shape
        raw_rows = []
        for y in range(h):
            raw_rows.append(b"\x00" + rgb[y].tobytes())
        raw_data = b"".join(raw_rows)
        compressed = zlib.compress(raw_data)

        def _chunk(ctype: bytes, data: bytes) -> bytes:
            c = ctype + data
            return (
                struct.pack(">I", len(data))
                + c
                + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
            )

        ihdr_data = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
        png = b"\x89PNG\r\n\x1a\n"
        png += _chunk(b"IHDR", ihdr_data)
        png += _chunk(b"IDAT", compressed)
        png += _chunk(b"IEND", b"")
        return png

    def _build_image_bytes(self) -> Optional[bytes]:
        e = self._engine
        rendered = e.camera.render(e.current_level.get_sprites())
        if hasattr(rendered, "tolist"):
            grid = rendered.tolist()
        else:
            grid = rendered if rendered else []
        if not grid:
            return None
        arr = np.array(grid, dtype=np.uint8)
        h, w = arr.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for idx, color in enumerate(self.ARC_PALETTE):
            mask = arr == idx
            rgb[mask] = color
        return self._encode_png(rgb)

    def _build_game_state(
        self, done: bool = False, info: Optional[Dict] = None
    ) -> GameState:
        e = self._engine
        valid_actions = self.get_actions() if not done else None
        levels_completed = len(e._levels) if self._game_won else e.level_index
        return GameState(
            text_observation=self._build_text_obs(),
            image_observation=self._build_image_bytes(),
            valid_actions=valid_actions,
            turn=self._total_turns,
            metadata={
                "total_levels": len(e._levels),
                "level_index": e.level_index,
                "levels_completed": levels_completed,
                "game_over": e._game_over,
                "done": done,
                "info": info or {},
            },
        )

    def reset(self) -> GameState:
        e = self._engine
        reset_input = ActionInput(id=GameAction.RESET)
        if self._game_won or self._last_action_was_reset:
            e.perform_action(reset_input)
            e.perform_action(reset_input)
        else:
            e.perform_action(reset_input)
        self._total_turns = 0
        self._last_action_was_reset = True
        self._game_won = False
        return self._build_game_state()

    def get_actions(self) -> List[str]:
        if self._engine._game_over or self._game_won:
            return ["reset"]
        return list(self._VALID_ACTIONS)

    def is_done(self) -> bool:
        return self._game_won

    def step(self, action: str) -> StepResult:
        e = self._engine

        parts = action.split()
        action_key = parts[0] if parts else action

        if action_key == "reset":
            state = self.reset()
            return StepResult(
                state=state, reward=0.0, done=False, info={"action": "reset"}
            )

        if action_key not in self._ACTION_MAP:
            raise ValueError(
                "Invalid action '{}'. Must be one of {}".format(
                    action_key, list(self._ACTION_MAP.keys())
                )
            )

        if e._game_over or self._game_won:
            return StepResult(
                state=self._build_game_state(done=self._game_won),
                reward=0.0,
                done=self._game_won,
                info={"action": action},
            )

        self._last_action_was_reset = False
        self._total_turns += 1

        game_action_id = self._ACTION_MAP[action_key]
        action_map = {
            1: GameAction.ACTION1,
            2: GameAction.ACTION2,
            3: GameAction.ACTION3,
            4: GameAction.ACTION4,
            5: GameAction.ACTION5,
            7: GameAction.ACTION7,
        }
        game_action = action_map[game_action_id]
        info: Dict = {"action": action}

        total_levels = len(e._levels)
        level_before = e.level_index

        action_input = ActionInput(id=game_action)
        frame = e.perform_action(action_input, raw=True)

        game_won = frame and frame.state and frame.state.name == "WIN"
        level_advanced = e.level_index > level_before
        done = game_won

        reward = 0.0
        if game_won or level_advanced:
            reward = 1.0 / total_levels

        if game_won:
            self._game_won = True
            info["reason"] = "game_complete"
            return StepResult(
                state=self._build_game_state(done=True, info=info),
                reward=reward,
                done=True,
                info=info,
            )

        if e._game_over:
            info["reason"] = "death"
            return StepResult(
                state=self._build_game_state(done=False, info=info),
                reward=0.0,
                done=False,
                info=info,
            )

        return StepResult(
            state=self._build_game_state(done=False, info=info),
            reward=reward,
            done=False,
            info=info,
        )

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError("Unsupported render mode: {}".format(mode))
        e = self._engine
        index_grid = e.camera.render(e.current_level.get_sprites())
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        for idx, color in enumerate(self.ARC_PALETTE):
            mask = index_grid == idx
            rgb[mask] = color
        return rgb

    def close(self) -> None:
        self._engine = None


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
                "Unsupported render_mode '{}'. Supported: {}".format(
                    render_mode, self.metadata["render_modes"]
                )
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
        return img[row_idx[:, None], col_idx[None, :]]

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
    env.render()
    env.close()
