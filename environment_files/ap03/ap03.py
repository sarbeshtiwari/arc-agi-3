import random
import struct
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from arcengine import (
    ActionInput,
    ARCBaseGame,
    Camera,
    GameAction,
    GameState as EngineGameState,
    Level,
    Sprite,
)


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


ARC_PALETTE: List[Tuple[int, int, int]] = [
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

FRAME_SIZE = 64
N_LEVELS = 4

LIVES_PER_LEVEL = [3, 3, 3, 3]

LEVEL_CONFIGS = [
    (10, 6),
    (14, 4),
    (14, 4),
    (14, 4),
]

BASE_GRID = 14

BASE_CONFIGS = [
    {
        "max_moves": 144,
        "blocks": [(1, 1, 8), (1, 12, 8)],
        "targets": [(10, 4, 8), (12, 9, 8)],
        "blockers": [
            (3, 3),
            (3, 4),
            (3, 9),
            (3, 10),
            (7, 2),
            (7, 6),
            (7, 7),
            (7, 11),
        ],
        "colors": [8],
        "reference_moves": 18,
    },
    {
        "max_moves": 192,
        "blocks": [(1, 0, 8), (1, 13, 8)],
        "targets": [(11, 6, 8), (13, 10, 8)],
        "blockers": [
            (3, 2),
            (3, 3),
            (3, 10),
            (3, 11),
            (6, 4),
            (6, 5),
            (6, 8),
            (6, 9),
            (9, 1),
            (9, 7),
            (9, 11),
            (9, 12),
        ],
        "colors": [8],
        "reference_moves": 24,
    },
    {
        "max_moves": 304,
        "blocks": [(1, 1, 8), (1, 11, 8), (12, 6, 8)],
        "targets": [(11, 8, 8), (8, 2, 8), (3, 12, 8)],
        "blockers": [
            (5, 5),
            (5, 6),
            (5, 7),
            (5, 8),
            (6, 5),
            (6, 8),
            (7, 5),
            (7, 8),
            (8, 5),
            (8, 6),
            (8, 7),
            (8, 8),
            (3, 3),
            (3, 9),
            (10, 4),
            (10, 10),
        ],
        "colors": [8],
        "reference_moves": 38,
    },
    {
        "max_moves": 416,
        "blocks": [(0, 0, 8), (0, 13, 8), (13, 6, 8)],
        "targets": [(12, 9, 8), (9, 3, 8), (4, 13, 8)],
        "blockers": [
            (2, 2),
            (2, 4),
            (2, 9),
            (2, 11),
            (5, 4),
            (5, 5),
            (5, 6),
            (5, 8),
            (5, 9),
            (6, 4),
            (6, 6),
            (6, 9),
            (7, 4),
            (7, 6),
            (7, 7),
            (7, 9),
            (8, 4),
            (8, 6),
            (8, 9),
            (9, 4),
            (9, 5),
            (9, 8),
            (9, 9),
        ],
        "colors": [8],
        "reference_moves": 52,
    },
]

COLOR_NAMES = {8: "Red", 6: "Pink", 9: "Blue", 7: "Light Pink"}

_levels = [
    Level(sprites=[], grid_size=(FRAME_SIZE, FRAME_SIZE)) for _ in range(N_LEVELS)
]

ACTION_LIST: List[str] = ["reset", "up", "down", "left", "right", "select", "undo"]

ACTION_FROM_NAME: Dict[str, GameAction] = {
    "up": GameAction.ACTION1,
    "down": GameAction.ACTION2,
    "left": GameAction.ACTION3,
    "right": GameAction.ACTION4,
    "select": GameAction.ACTION5,
    "undo": GameAction.ACTION7,
    "reset": GameAction.RESET,
}


def _scale_coord(val: int, logical_size: int) -> int:
    scale = logical_size / BASE_GRID
    scaled = int(round(val * scale))
    return min(max(scaled, 0), logical_size - 1)


class Ap03(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)
        camera = Camera(
            background=5,
            letter_box=5,
            width=FRAME_SIZE,
            height=FRAME_SIZE,
        )
        super().__init__(
            game_id="ap03",
            levels=_levels,
            camera=camera,
            available_actions=[0, 1, 2, 3, 4, 5, 7],
        )
        self.seed = seed
        self.logical_size: int = LEVEL_CONFIGS[0][0]
        self.cell_size: int = LEVEL_CONFIGS[0][1]
        self.current_color_idx: int = 0
        self.available_colors: List[int] = []
        self.blocks: List[List[int]] = []
        self.targets: List[Tuple[int, int, int]] = []
        self.blockers: List[Tuple[int, int]] = []
        self.max_moves: int = 0
        self.moves_used: int = 0
        self._lives: int = 0
        self._lives_max: int = 0
        self.last_on_target: int = 0
        self.flash_state: Optional[str] = None
        self.flash_timer: int = 0
        self._awaiting_confirm: bool = False
        self._ready: bool = False
        self.undo_stack: List[dict] = []
        self._ready = True

    def on_set_level(self, level: Level) -> None:
        if not getattr(self, "_ready", False):
            return
        self.undo_stack = []
        config = LEVEL_CONFIGS[self._current_level_index]
        self.logical_size = config[0]
        self.cell_size = config[1]
        self.camera.width = FRAME_SIZE
        self.camera.height = FRAME_SIZE
        self.current_level.remove_all_sprites()
        self._init_level(self._current_level_index)
        self._render()

    def _init_level(self, level_idx: int) -> None:
        self.current_color_idx = 0
        self.moves_used = 0
        self.last_on_target = 0
        self.flash_state = None
        self.flash_timer = 0
        self._awaiting_confirm = False
        self._lives_max = (
            LIVES_PER_LEVEL[level_idx] if level_idx < len(LIVES_PER_LEVEL) else 3
        )
        self._lives = self._lives_max
        self.undo_stack = []

        idx = min(level_idx, len(BASE_CONFIGS) - 1)
        base_config = BASE_CONFIGS[idx]

        self.targets = [
            (
                _scale_coord(r, self.logical_size),
                _scale_coord(c, self.logical_size),
                color,
            )
            for r, c, color in base_config["targets"]
        ]
        self.blockers = [
            (_scale_coord(r, self.logical_size), _scale_coord(c, self.logical_size))
            for r, c in base_config.get("blockers", [])
        ]

        occupied: Set[Tuple[int, int]] = set()
        for tr, tc, _ in self.targets:
            occupied.add((tr, tc))
        for br, bc in self.blockers:
            occupied.add((br, bc))

        self.blocks = []
        for _, _, color in base_config["blocks"]:
            for _ in range(200):
                rr = self._rng.randint(0, self.logical_size - 1)
                rc = self._rng.randint(0, self.logical_size - 1)
                if (rr, rc) not in occupied:
                    self.blocks.append([rr, rc, color])
                    occupied.add((rr, rc))
                    break

        self.max_moves = base_config["max_moves"]
        self.available_colors = base_config["colors"]
        self.current_color_idx = 0

    def _save_undo(self) -> None:
        self.undo_stack.append(
            {
                "blocks": [[r, c, col] for r, c, col in self.blocks],
                "current_color_idx": self.current_color_idx,
                "last_on_target": self.last_on_target,
            }
        )

    def _perform_undo(self) -> bool:
        if not self.undo_stack:
            return False
        snap = self.undo_stack.pop()
        self.blocks = snap["blocks"]
        self.current_color_idx = snap["current_color_idx"]
        self.last_on_target = snap["last_on_target"]
        return True

    def _get_current_color(self) -> int:
        if not self.available_colors:
            return 8
        return self.available_colors[self.current_color_idx]

    def _count_on_target(self) -> int:
        count = 0
        for br, bc, bcolor in self.blocks:
            for tr, tc, tcolor in self.targets:
                if br == tr and bc == tc and bcolor == tcolor:
                    count += 1
        return count

    def _check_win(self) -> bool:
        return self._count_on_target() == len(self.blocks)

    def _get_block_at(self, row: int, col: int) -> Optional[int]:
        for idx, (br, bc, _) in enumerate(self.blocks):
            if br == row and bc == col:
                return idx
        return None

    def _is_blocker_at(self, row: int, col: int) -> bool:
        return (row, col) in self.blockers

    def _is_occupied(self, row: int, col: int, exclude_idx: int = -1) -> bool:
        for idx, (br, bc, _) in enumerate(self.blocks):
            if idx == exclude_idx:
                continue
            if br == row and bc == col:
                return True
        return False

    def _move_blocks(self, dr: int, dc: int) -> None:
        current_color = self._get_current_color()
        color_indices = [
            i for i, (_, _, c) in enumerate(self.blocks) if c == current_color
        ]
        if not color_indices:
            return

        if dr < 0:
            color_indices.sort(key=lambda i: self.blocks[i][0])
        elif dr > 0:
            color_indices.sort(key=lambda i: -self.blocks[i][0])
        elif dc < 0:
            color_indices.sort(key=lambda i: self.blocks[i][1])
        elif dc > 0:
            color_indices.sort(key=lambda i: -self.blocks[i][1])

        moved = False
        for idx in color_indices:
            row, col, color = self.blocks[idx]
            new_row = (row + dr) % self.logical_size
            new_col = (col + dc) % self.logical_size

            if self._is_blocker_at(new_row, new_col):
                continue

            blocking_idx = self._get_block_at(new_row, new_col)
            if blocking_idx is not None:
                if blocking_idx in color_indices:
                    continue
                push_row = (new_row + dr) % self.logical_size
                push_col = (new_col + dc) % self.logical_size
                if not self._is_blocker_at(
                    push_row, push_col
                ) and not self._is_occupied(
                    push_row, push_col, exclude_idx=blocking_idx
                ):
                    self.blocks[blocking_idx][0] = push_row
                    self.blocks[blocking_idx][1] = push_col
                    self.blocks[idx][0] = new_row
                    self.blocks[idx][1] = new_col
                    moved = True
            else:
                self.blocks[idx][0] = new_row
                self.blocks[idx][1] = new_col
                moved = True

        if moved:
            current_on_target = self._count_on_target()
            if current_on_target > self.last_on_target:
                self.flash_state = "progress"
            elif current_on_target < self.last_on_target:
                self.flash_state = "regress"
            else:
                self.flash_state = "neutral"
            self.flash_timer = 2
            self.last_on_target = current_on_target

    def _lose_life(self) -> bool:
        self._lives -= 1
        if self._lives <= 0:
            self.flash_state = "regress"
            self.flash_timer = 3
            self._render()
            self.lose()
            return True
        self.flash_state = "regress"
        self.flash_timer = 3
        self._render()
        self._reset_board()
        return False

    def _reset_board(self) -> None:
        saved_lives = self._lives
        saved_lives_max = self._lives_max
        self._init_level(self._current_level_index)
        self._lives = saved_lives
        self._lives_max = saved_lives_max

    def step(self) -> None:
        if not self._ready:
            self.complete_action()
            return

        if self.action and self.action.id == GameAction.RESET:
            self.complete_action()
            return

        if self.flash_timer > 0:
            self.flash_timer -= 1
            if self.flash_timer == 0:
                self.flash_state = None

        action = self.action

        if self._awaiting_confirm:
            if action.id == GameAction.ACTION5:
                self._awaiting_confirm = False
                self.next_level()
                self._render()
                self.complete_action()
                return
            if action.id == GameAction.ACTION7:
                self._perform_undo()
                self._awaiting_confirm = self._check_win()
                self._render()
                self.complete_action()
                return
            self._render()
            self.complete_action()
            return

        if action.id == GameAction.ACTION7:
            self._perform_undo()
            self.moves_used += 1
            if self.moves_used >= self.max_moves:
                if self._lose_life():
                    self.complete_action()
                    return
            self._render()
            self.complete_action()
            return

        if action.id == GameAction.ACTION5:
            if self.moves_used >= self.max_moves:
                if self._lose_life():
                    self.complete_action()
                    return
                self._render()
                self.complete_action()
                return
            self._save_undo()
            self.current_color_idx = (self.current_color_idx + 1) % len(
                self.available_colors
            )
            self.moves_used += 1
            if self.moves_used >= self.max_moves:
                if self._lose_life():
                    self.complete_action()
                    return
            self._render()
            self.complete_action()
            return

        if action.id in (
            GameAction.ACTION1,
            GameAction.ACTION2,
            GameAction.ACTION3,
            GameAction.ACTION4,
        ):
            if self.moves_used >= self.max_moves:
                if self._lose_life():
                    self.complete_action()
                    return
                self._render()
                self.complete_action()
                return

            self._save_undo()

            dr, dc = 0, 0
            if action.id == GameAction.ACTION1:
                dr, dc = -1, 0
            elif action.id == GameAction.ACTION2:
                dr, dc = 1, 0
            elif action.id == GameAction.ACTION3:
                dr, dc = 0, -1
            elif action.id == GameAction.ACTION4:
                dr, dc = 0, 1

            self._move_blocks(dr, dc)
            self.moves_used += 1

            if self._check_win():
                self._awaiting_confirm = True
                self.flash_state = "progress"
                self.flash_timer = 0
                self._render()
                self.complete_action()
                return

            if self.moves_used >= self.max_moves:
                if self._lose_life():
                    self.complete_action()
                    return
                self._render()
                self.complete_action()
                return

        self._render()
        self.complete_action()

    def _render(self) -> None:
        self.current_level.remove_all_sprites()
        frame = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=np.uint8)

        border_color = 5
        if self.flash_state == "progress":
            border_color = 14
        elif self.flash_state == "regress":
            border_color = 8
        elif self.flash_state == "neutral":
            border_color = 9

        frame[0, :] = border_color
        frame[FRAME_SIZE - 1, :] = border_color
        frame[:, 0] = border_color
        frame[:, FRAME_SIZE - 1] = border_color

        moves_left = self.max_moves - self.moves_used
        max_dots_fit = FRAME_SIZE - 2
        dots_to_show = min(self.max_moves, max_dots_fit)
        start_col = 1 + (max_dots_fit - dots_to_show) // 2

        current_color = self._get_current_color()
        for i in range(dots_to_show):
            move_index = int(i * self.max_moves / dots_to_show)
            if move_index < moves_left:
                frame[0, start_col + i] = current_color
            else:
                frame[0, start_col + i] = 4

        spent_lives = self._lives_max - self._lives
        dot_w = 3
        gap = 1
        for i in range(self._lives_max):
            dot_x = 1 + i * (dot_w + gap)
            dot_color = 4 if i < spent_lives else 8
            for dx in range(dot_w):
                px = dot_x + dx
                if 0 < px < FRAME_SIZE - 1:
                    frame[FRAME_SIZE - 1, px] = dot_color

        playfield_px = self.logical_size * self.cell_size
        offset_y = 1 + (FRAME_SIZE - 2 - playfield_px) // 2
        offset_x = 1 + (FRAME_SIZE - 2 - playfield_px) // 2

        frame[
            offset_y : offset_y + playfield_px, offset_x : offset_x + playfield_px
        ] = 0

        def draw_cell(row, col, color):
            py = offset_y + row * self.cell_size
            px_start = offset_x + col * self.cell_size
            frame[py : py + self.cell_size, px_start : px_start + self.cell_size] = (
                color
            )

        for tr, tc, tcolor in self.targets:
            draw_cell(tr, tc, 1)

        for br, bc in self.blockers:
            draw_cell(br, bc, 4)

        for br, bc, bcolor in self.blocks:
            on_target = False
            for tr, tc, tcolor in self.targets:
                if br == tr and bc == tc and bcolor == tcolor:
                    on_target = True
                    break
            display_color = 12 if on_target else bcolor
            draw_cell(br, bc, display_color)

        confirm_color = 11 if self._awaiting_confirm else 14
        for dy in range(3):
            for dx in range(3):
                frame[dy, FRAME_SIZE - 3 + dx] = confirm_color

        sprite = Sprite(frame, x=0, y=0)
        self.current_level.add_sprite(sprite)

    @property
    def extra_state(self) -> dict:
        idx = self._current_level_index
        on_target = self._count_on_target()
        total_blocks = len(self.blocks)
        current_color = self._get_current_color()
        color_name = COLOR_NAMES.get(current_color, "Unknown")
        moves_left = self.max_moves - self.moves_used
        return {
            "circuit_title": "Quantum Shift - Level %d/%d" % (idx + 1, N_LEVELS),
            "level_title": "Level %d" % (idx + 1),
            "moves_used": self.moves_used,
            "max_moves": self.max_moves,
            "moves_left": moves_left,
            "on_target": on_target,
            "total_blocks": total_blocks,
            "current_color": current_color,
            "current_color_name": color_name,
            "lives": self._lives,
            "lives_max": self._lives_max,
            "lives_lost": self._lives_max - self._lives,
            "awaiting_confirm": self._awaiting_confirm,
            "level_features": [
                "Level %d/%d" % (idx + 1, N_LEVELS),
                "Moves: %d/%d (%d left)"
                % (self.moves_used, self.max_moves, moves_left),
                "On target: %d/%d" % (on_target, total_blocks),
                "Color: %s" % color_name,
                "Lives: %d/%d" % (self._lives, self._lives_max),
            ],
        }


class PuzzleEnvironment:
    ACTION_MAP: Dict[str, GameAction] = ACTION_FROM_NAME

    def __init__(self, seed: int = 0) -> None:
        self._engine = Ap03(seed=seed)
        self._total_turns: int = 0
        self._last_action_was_reset: bool = False
        self._game_won: bool = False
        self._levels_completed: int = 0
        self._total_levels: int = len(self._engine._levels)

    def _build_text_observation(self) -> str:
        e = self._engine
        idx = e._current_level_index
        on_target = e._count_on_target()
        total_blocks = len(e.blocks)
        current_color = e._get_current_color()
        color_name = COLOR_NAMES.get(current_color, "Unknown")
        moves_left = e.max_moves - e.moves_used

        lines: List[str] = []
        lines.append(
            "Level %d/%d | Grid: %dx%d | Color: %s"
            % (idx + 1, N_LEVELS, e.logical_size, e.logical_size, color_name)
        )
        lines.append(
            "Moves: %d/%d (%d left) | Lives: %d/%d | On target: %d/%d"
            % (
                e.moves_used,
                e.max_moves,
                moves_left,
                e._lives,
                e._lives_max,
                on_target,
                total_blocks,
            )
        )

        if e._state == EngineGameState.WIN:
            lines.append("State: won")
        elif e._state == EngineGameState.GAME_OVER:
            lines.append("State: game-over")
        elif e._awaiting_confirm:
            lines.append("State: confirm-ready (press select to advance)")
        else:
            lines.append("State: playing")

        lines.append("")

        target_set = {(tr, tc): tcolor for tr, tc, tcolor in e.targets}
        blocker_set = set(e.blockers)
        block_map = {}
        for br, bc, bcolor in e.blocks:
            block_map[(br, bc)] = bcolor

        for row in range(e.logical_size):
            row_chars: List[str] = []
            for col in range(e.logical_size):
                if (row, col) in block_map:
                    bcolor = block_map[(row, col)]
                    if (row, col) in target_set and target_set[(row, col)] == bcolor:
                        row_chars.append("*")
                    else:
                        row_chars.append("B")
                elif (row, col) in blocker_set:
                    row_chars.append("#")
                elif (row, col) in target_set:
                    row_chars.append("T")
                else:
                    row_chars.append(".")
            lines.append(" ".join(row_chars))

        lines.append("")
        lines.append("Legend: B=block T=target *=block-on-target #=blocker .=empty")
        return "\n".join(lines)

    @staticmethod
    def _encode_png(rgb: np.ndarray) -> bytes:
        h, w, _ = rgb.shape
        raw_rows = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))
        compressed = zlib.compress(raw_rows)

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
        for idx, color in enumerate(ARC_PALETTE):
            mask = arr == idx
            rgb[mask] = color
        return self._encode_png(rgb)

    def _build_game_state(self) -> GameState:
        e = self._engine
        engine_done = e._state in (EngineGameState.WIN, EngineGameState.GAME_OVER)
        valid_actions = ["reset"] if engine_done else self.get_actions()
        idx = e._current_level_index
        game_over = e._state == EngineGameState.GAME_OVER

        return GameState(
            text_observation=self._build_text_observation(),
            image_observation=self._build_image_bytes(),
            valid_actions=valid_actions,
            turn=self._total_turns,
            metadata={
                "total_levels": self._total_levels,
                "level_index": idx,
                "levels_completed": self._levels_completed,
                "game_over": game_over,
                "done": engine_done,
                "info": {},
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
        self._levels_completed = 0
        return self._build_game_state()

    def get_actions(self) -> List[str]:
        e = self._engine
        if e._state in (EngineGameState.WIN, EngineGameState.GAME_OVER):
            return ["reset"]
        if e._awaiting_confirm:
            return ["reset", "select", "undo"]
        return ACTION_LIST

    def is_done(self) -> bool:
        return self._engine._state == EngineGameState.WIN

    def step(self, action: str) -> StepResult:
        action = action.strip().lower()
        game_action = ACTION_FROM_NAME.get(action)
        if game_action is None:
            return StepResult(
                state=self._build_game_state(),
                reward=0.0,
                done=False,
                info={"action": action, "error": "invalid_action"},
            )

        if game_action == GameAction.RESET:
            state = self.reset()
            return StepResult(
                state=state, reward=0.0, done=False, info={"action": "reset"}
            )

        self._last_action_was_reset = False
        self._total_turns += 1

        e = self._engine
        level_before = e.level_index

        action_input = ActionInput(id=game_action)
        frame = e.perform_action(action_input)

        game_won = e._state == EngineGameState.WIN
        game_over = e._state == EngineGameState.GAME_OVER
        level_advanced = e.level_index > level_before
        done = game_won or game_over

        reward = 0.0
        if game_won or level_advanced:
            self._levels_completed += 1
            reward = 1.0 / self._total_levels

        if game_won:
            self._game_won = True

        return StepResult(
            state=self._build_game_state(),
            reward=reward,
            done=done,
            info={
                "action": action,
                "engine_state": e._state,
                "level_changed": level_advanced,
                "life_lost": False,
            },
        )

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError("Unsupported render mode: %s" % mode)
        e = self._engine
        index_grid = e.camera.render(e.current_level.get_sprites())
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
        "render_fps": 10,
    }

    ACTION_LIST: List[str] = ["reset", "up", "down", "left", "right", "select", "undo"]

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
                "Unsupported render_mode '%s'. Supported: %s"
                % (render_mode, self.metadata["render_modes"])
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
        return img[row_idx[:, None], col_idx[None, :]].astype(np.uint8)

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
