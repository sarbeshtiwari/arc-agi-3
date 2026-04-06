import numpy as np
from typing import Any, Dict, List, Optional, Sequence, Tuple
from dataclasses import dataclass, field
import gymnasium as gym
from gymnasium import spaces
from arcengine import (
    ARCBaseGame,
    ActionInput,
    Camera,
    GameAction,
    GameState as ArcGameState,
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


_ARC_PALETTE = [
    (255, 255, 255), (204, 204, 204), (153, 153, 153), (102, 102, 102),
    (51, 51, 51), (0, 0, 0), (229, 58, 163), (255, 123, 204),
    (249, 60, 49), (30, 147, 255), (136, 216, 241), (255, 220, 0),
    (255, 133, 27), (146, 18, 49), (79, 204, 48), (163, 86, 208),
]


C_BG = 0
C_WALL = 10
C_PLAYER = 14
C_GOAL = 7
C_KNIGHT = 12
C_KING = 6
C_QUEEN = 15
C_PENALTY_A = 9
C_PENALTY_B = 11
C_MINE = 2
C_INACTIVE = 13
C_FLOOR_DARK = 5
C_CURSOR = 3
C_LIFE = 8
C_LIFE_EMPTY = 0
C_MOVES_BAR_FILLED = 11
C_MOVES_BAR_EMPTY = C_BG
C_PIECE_COLLECTED = 12
C_PIECE_NOT_COLLECTED = C_CURSOR

MAX_LIVES = 3

MOVES_BAR_WIDTH = 46

LIVES_PIP_W = 2
LIVES_GAP = 1
LIVES_RIGHT_MARGIN = 2


class MoveHUD(RenderableUserDisplay):
    def __init__(self, game: "Ch91") -> None:
        self._game = game

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        if self._game.move_limit <= 0:
            return frame

        frame_height = frame.shape[0]
        frame_width = frame.shape[1]


        ratio = max(0.0, min(1.0, self._game.moves_left / self._game.move_limit))
        filled = int(MOVES_BAR_WIDTH * ratio)
        for col in range(MOVES_BAR_WIDTH):
            frame[0:2, col] = C_MOVES_BAR_FILLED if col < filled else C_MOVES_BAR_EMPTY

        lives_total_width = MAX_LIVES * LIVES_PIP_W + (MAX_LIVES - 1) * LIVES_GAP
        lives_start_x = frame_width - LIVES_RIGHT_MARGIN - lives_total_width
        for i in range(MAX_LIVES):
            lx = lives_start_x + i * (LIVES_PIP_W + LIVES_GAP)
            color = C_LIFE if i < self._game._lives else C_LIFE_EMPTY
            frame[0:2, lx : lx + LIVES_PIP_W] = color


        mode_color = {
            "pawn": C_PLAYER,
            "knight": C_KNIGHT,
            "king": C_KING,
            "queen": C_QUEEN,
        }.get(self._game.current_mode, C_PLAYER)
        frame[frame_height - 2 : frame_height, 1:3] = mode_color

        order = ["knight", "king", "queen"]
        for i, key in enumerate(order):
            indicator_color = C_PIECE_COLLECTED if key in self._game.activated else C_PIECE_NOT_COLLECTED
            indicator_col_start = 6 + i * 4
            frame[frame_height - 2 : frame_height, indicator_col_start : indicator_col_start + 2] = indicator_color

        return frame


sprites = {
    "floor_dark": Sprite(
        pixels=[[C_FLOOR_DARK]],
        name="floor_dark",
        visible=True,
        collidable=False,
        tags=["floor"],
        layer=0,
    ),
    "floor_light": Sprite(
        pixels=[[C_BG]],
        name="floor_light",
        visible=True,
        collidable=False,
        tags=["floor"],
        layer=0,
    ),
    "wall": Sprite(
        pixels=[[C_WALL]],
        name="wall",
        visible=True,
        collidable=True,
        tags=["wall"],
        layer=2,
    ),
    "extra_wall": Sprite(
        pixels=[[C_WALL]],
        name="extra_wall",
        visible=True,
        collidable=True,
        tags=["wall", "extra_wall"],
        layer=2,
    ),
    "player": Sprite(
        pixels=[[C_PLAYER]],
        name="player",
        visible=True,
        collidable=True,
        tags=["player"],
        layer=6,
    ),
    "goal": Sprite(
        pixels=[[C_GOAL]],
        name="goal",
        visible=True,
        collidable=False,
        tags=["goal"],
        layer=1,
    ),
    "knight": Sprite(
        pixels=[[C_KNIGHT]],
        name="knight",
        visible=True,
        collidable=False,
        tags=["piece", "knight"],
        layer=1,
    ),
    "king": Sprite(
        pixels=[[C_KING]],
        name="king",
        visible=True,
        collidable=False,
        tags=["piece", "king"],
        layer=1,
    ),
    "queen": Sprite(
        pixels=[[C_QUEEN]],
        name="queen",
        visible=True,
        collidable=False,
        tags=["piece", "queen"],
        layer=1,
    ),
    "penalty_a": Sprite(
        pixels=[[C_PENALTY_A]],
        name="penalty_a",
        visible=True,
        collidable=False,
        tags=["penalty", "penalty_a"],
        layer=1,
    ),
    "penalty_b": Sprite(
        pixels=[[C_PENALTY_B]],
        name="penalty_b",
        visible=True,
        collidable=False,
        tags=["penalty", "penalty_b"],
        layer=1,
    ),
    "mine": Sprite(
        pixels=[[C_MINE]],
        name="mine",
        visible=True,
        collidable=False,
        tags=["mine"],
        layer=1,
    ),
    "cursor": Sprite(
        pixels=[[C_CURSOR]],
        name="cursor",
        visible=False,
        collidable=False,
        tags=["cursor"],
        layer=5,
    ),
}


LEVELS = [
    {
        "name": "Level 1",
        "move_limit": 136,
        "required_piece_count": 1,
        "extra_walls": set(),
        "grid": (
            "################\n"
            "#P.............#\n"
            "#......N.......#\n"
            "#..............#\n"
            "#..............#\n"
            "#....K.........#\n"
            "#..............#\n"
            "#..............#\n"
            "#........Q.....#\n"
            "#..............#\n"
            "#..............#\n"
            "#..............#\n"
            "#..............#\n"
            "#............G.#\n"
            "#..............#\n"
            "################"
        ),
    },
    {
        "name": "Level 2",
        "move_limit": 120,
        "required_piece_count": 1,
        "extra_walls": set(),
        "grid": (
            "################\n"
            "#P....#........#\n"
            "#.....#.N......#\n"
            "#.....#........#\n"
            "#.....#####....#\n"
            "#..............#\n"
            "#..K...........#\n"
            "#..............#\n"
            "#......#####...#\n"
            "#...........Q..#\n"
            "#..............#\n"
            "#..#####.......#\n"
            "#..............#\n"
            "#............G.#\n"
            "#..............#\n"
            "################"
        ),
    },
    {
        "name": "Level 3",
        "move_limit": 104,
        "required_piece_count": 1,
        "extra_walls": {(10, 2), (4, 10), (8, 6)},
        "grid": (
            "################\n"
            "#P....#........#\n"
            "#..A..#.N......#\n"
            "#.....#........#\n"
            "#.....#####..B.#\n"
            "#..............#\n"
            "#..K...........#\n"
            "#..............#\n"
            "#......#####...#\n"
            "#...........Q..#\n"
            "#..............#\n"
            "#..#####.......#\n"
            "#..B...........#\n"
            "#............G.#\n"
            "#..............#\n"
            "################"
        ),
    },
    {
        "name": "Level 4",
        "move_limit": 88,
        "required_piece_count": 1,
        "extra_walls": {
            (2, 3),
            (3, 10),
            (4, 8),
            (7, 6),
            (7, 9),
            (7, 10),
            (10, 5),
            (11, 7),
            (12, 5),
        },
        "grid": (
            "################\n"
            "#P....#........#\n"
            "#..A..#.NM.....#\n"
            "#.....#........#\n"
            "#.....#####..B.#\n"
            "#..............#\n"
            "#..K.M.........#\n"
            "#..............#\n"
            "#......#####...#\n"
            "#.........MQ...#\n"
            "#..............#\n"
            "#..#####.......#\n"
            "#..B...........#\n"
            "#............G.#\n"
            "#..............#\n"
            "################"
        ),
    },
]


def _parse_grid_chars(rows):
    level_sprites = []
    spawn = (1, 1)

    for row_y, row in enumerate(rows):
        for col_x, ch in enumerate(row):
            if ch != "#":
                floor_sprite = (
                    sprites["floor_dark"]
                    if (col_x + row_y) % 2 == 0
                    else sprites["floor_light"]
                )
                level_sprites.append(floor_sprite.clone().set_position(col_x, row_y))

            if ch == "#":
                level_sprites.append(sprites["wall"].clone().set_position(col_x, row_y))
            elif ch == "P":
                spawn = (col_x, row_y)
                level_sprites.append(sprites["player"].clone().set_position(col_x, row_y))
            elif ch == "G":
                level_sprites.append(sprites["goal"].clone().set_position(col_x, row_y))
            elif ch == "N":
                level_sprites.append(sprites["knight"].clone().set_position(col_x, row_y))
            elif ch == "K":
                level_sprites.append(sprites["king"].clone().set_position(col_x, row_y))
            elif ch == "Q":
                level_sprites.append(sprites["queen"].clone().set_position(col_x, row_y))
            elif ch == "A":
                level_sprites.append(sprites["penalty_a"].clone().set_position(col_x, row_y))
            elif ch == "B":
                level_sprites.append(sprites["penalty_b"].clone().set_position(col_x, row_y))
            elif ch == "M":
                level_sprites.append(sprites["mine"].clone().set_position(col_x, row_y))

    return level_sprites, spawn


def _parse_level(entry):
    rows = str(entry["grid"]).split("\n")
    row_lengths = {len(r) for r in rows}
    assert len(row_lengths) == 1, (
        f"Level '{entry['name']}' has inconsistent row widths: {row_lengths}"
    )
    grid_height = len(rows)
    grid_width = row_lengths.pop()

    level_sprites, spawn = _parse_grid_chars(rows)

    return Level(
        sprites=level_sprites,
        grid_size=(grid_width, grid_height),
        data={
            "move_limit": int(entry["move_limit"]),
            "required_piece_count": int(entry.get("required_piece_count", 1)),
            "spawn": [spawn[0], spawn[1]],
            "extra_walls": entry.get("extra_walls", set()),
        },
        name=str(entry["name"]),
    )


levels = [_parse_level(level) for level in LEVELS]


class Ch91(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self.hud = MoveHUD(self)
        self.move_limit = 0
        self.moves_left = 0
        self.activated = set()
        self.current_mode = "pawn"
        self._cursor_x = 1
        self._cursor_y = 1
        self._cursor_active = False
        self._wall_set = set()
        self._goal_set = set()
        self._penalty_set = set()
        self._mine_set = set()
        self._lives = MAX_LIVES
        self._was_game_over = False
        self._consecutive_reset_count = 0
        self._game_over = False
        self._undo_stack = []
        self._level_done = False

        super().__init__(
            "ch91",
            levels,
            Camera(0, 0, 64, 64, C_BG, C_BG, [self.hud]),
            available_actions=[0, 1, 2, 3, 4, 5, 6, 7],
            win_score=4,
        )



    def level_reset(self) -> None:
        lives = self._lives
        self._available_actions = [0, 1, 2, 3, 4, 5, 6, 7]
        self._undo_stack = []
        self._level_done = False
        super().level_reset()
        self._lives = lives

    def full_reset(self) -> None:
        self._lives = MAX_LIVES
        self._was_game_over = False
        self._consecutive_reset_count = 0
        self._available_actions = [0, 1, 2, 3, 4, 5, 6, 7]
        self._undo_stack = []
        self._level_done = False
        super().full_reset()

    def handle_reset(self) -> None:
        if self._state == ArcGameState.WIN:
            self.full_reset()
            return

        self._apply_reset_tier(
            retry_fn=self._retry_level,
            full_fn=self.full_reset,
        )

    def _apply_reset_tier(self, retry_fn: Any, full_fn: Any) -> None:

        self._consecutive_reset_count += 1
        if self._consecutive_reset_count >= 2:
            full_fn()
        else:
            retry_fn()

    def _retry_level(self) -> None:

        self._lives = MAX_LIVES
        self._was_game_over = False
        self.level_reset()



    def on_set_level(self, level: Level) -> None:
        self._level_done = False
        self._undo_stack = []
        self.player = self.current_level.get_sprites_by_tag("player")[0]
        self.goals = list(self.current_level.get_sprites_by_tag("goal"))
        self.knights = list(self.current_level.get_sprites_by_tag("knight"))
        self.kings = list(self.current_level.get_sprites_by_tag("king"))
        self.queens = list(self.current_level.get_sprites_by_tag("queen"))
        self.penalties = list(self.current_level.get_sprites_by_tag("penalty"))
        self.mines = list(self.current_level.get_sprites_by_tag("mine"))
        self._clear_extra_walls()
        self.walls = list(self.current_level.get_sprites_by_tag("wall"))

        self._load_level_limits()
        self._restore_player_spawn()

        self._initialise_lookup_sets()
        self._initialise_cursor(
            self.current_level.get_data("spawn") or [self.player.x, self.player.y]
        )

        self._refresh_piece_visuals()
        self._refresh_player_visual()

    def _load_level_limits(self) -> None:

        move_limit_data = self.current_level.get_data("move_limit")
        self.move_limit = int(move_limit_data if move_limit_data is not None else 96)
        req_data = self.current_level.get_data("required_piece_count")
        self.required_piece_count = int(req_data if req_data is not None else 1)
        self.moves_left = self.move_limit
        self.activated = set()
        self.current_mode = "pawn"
        self.stun_turns = {}

    def _restore_player_spawn(self) -> None:

        spawn = self.current_level.get_data("spawn") or [self.player.x, self.player.y]
        self.player.set_position(spawn[0], spawn[1])
        self._place_extra_walls()
        self.walls = list(self.current_level.get_sprites_by_tag("wall"))

    def _initialise_lookup_sets(self) -> None:

        self._wall_set = {(w.x, w.y) for w in self.walls}
        self._goal_set = {(g.x, g.y) for g in self.goals}
        self._penalty_set = {(p.x, p.y) for p in self.penalties}
        self._mine_set = {(m.x, m.y) for m in self.mines}
        overlap = self._penalty_set & self._mine_set
        assert not overlap, (
            f"Level '{self.current_level.name}' has cells that are both penalty and mine: {overlap}"
        )

    def _initialise_cursor(self, spawn: Sequence[int]) -> None:

        self._cursor_x = int(spawn[0])
        self._cursor_y = int(spawn[1])
        self._cursor_active = False
        self._cursor_sprite = sprites["cursor"].clone().set_position(
            self._cursor_x, self._cursor_y
        )
        self.current_level.add_sprite(self._cursor_sprite)

    def _clear_extra_walls(self) -> None:

        for wall_sprite in list(self.current_level.get_sprites_by_tag("extra_wall")):
            self.current_level.remove_sprite(wall_sprite)

    def _place_extra_walls(self) -> None:

        extra_walls = self.current_level.get_data("extra_walls") or set()
        base_wall_positions = {(w.x, w.y) for w in self.walls}
        occupied = {
            (self.player.x, self.player.y),
            *[(g.x, g.y) for g in self.goals],
            *[(s.x, s.y) for s in self.knights],
            *[(s.x, s.y) for s in self.kings],
            *[(s.x, s.y) for s in self.queens],
            *[(p.x, p.y) for p in self.penalties],
            *[(m.x, m.y) for m in self.mines],
        }
        for wall_x, wall_y in extra_walls:
            if (wall_x, wall_y) not in base_wall_positions and (wall_x, wall_y) not in occupied:
                self.current_level.add_sprite(
                    sprites["extra_wall"].clone().set_position(wall_x, wall_y)
                )



    def _refresh_player_visual(self) -> None:

        mode_color = {
            "pawn": C_PLAYER,
            "knight": C_KNIGHT,
            "king": C_KING,
            "queen": C_QUEEN,
        }.get(self.current_mode, C_PLAYER)
        self.player.color_remap(None, mode_color)

    def _piece_at(self, pos_x, pos_y):
        for knight_sprite in self.knights:
            if knight_sprite.x == pos_x and knight_sprite.y == pos_y:
                return "knight"
        for king_sprite in self.kings:
            if king_sprite.x == pos_x and king_sprite.y == pos_y:
                return "king"
        for queen_sprite in self.queens:
            if queen_sprite.x == pos_x and queen_sprite.y == pos_y:
                return "queen"
        return None

    def _piece_active(self, pos_x, pos_y, piece_type):
        return self.stun_turns.get((pos_x, pos_y, piece_type), 0) == 0

    def _tick_stun(self):
        self.stun_turns = {
            key: turns - 1
            for key, turns in self.stun_turns.items()
            if turns > 1
        }

    def _stun_nearby_pieces(self, blast_x, blast_y):

        for piece_list, piece_type in [
            (self.knights, "knight"),
            (self.kings, "king"),
            (self.queens, "queen"),
        ]:
            for piece_sprite in piece_list:
                if max(abs(piece_sprite.x - blast_x), abs(piece_sprite.y - blast_y)) <= 1:
                    self.stun_turns[(piece_sprite.x, piece_sprite.y, piece_type)] = 3

    def _refresh_piece_visuals(self) -> None:

        for knight_sprite in self.knights:
            active_color = (
                C_KNIGHT if self._piece_active(knight_sprite.x, knight_sprite.y, "knight")
                else C_INACTIVE
            )
            knight_sprite.color_remap(None, active_color)
        for king_sprite in self.kings:
            active_color = (
                C_KING if self._piece_active(king_sprite.x, king_sprite.y, "king")
                else C_INACTIVE
            )
            king_sprite.color_remap(None, active_color)
        for queen_sprite in self.queens:
            active_color = (
                C_QUEEN if self._piece_active(queen_sprite.x, queen_sprite.y, "queen")
                else C_INACTIVE
            )
            queen_sprite.color_remap(None, active_color)

    def _line_clear(self, src_x, src_y, dst_x, dst_y):
        step_x = 0 if dst_x == src_x else (1 if dst_x > src_x else -1)
        step_y = 0 if dst_y == src_y else (1 if dst_y > src_y else -1)
        cx, cy = src_x + step_x, src_y + step_y
        while (cx, cy) != (dst_x, dst_y):
            if (cx, cy) in self._wall_set:
                return False
            cx += step_x
            cy += step_y
        return True

    def _legal_move(self, src_x, src_y, dst_x, dst_y):

        if (dst_x, dst_y) in self._wall_set:
            return False

        delta_x = dst_x - src_x
        delta_y = dst_y - src_y
        abs_dx = abs(delta_x)
        abs_dy = abs(delta_y)

        if abs_dx == 0 and abs_dy == 0:
            return False

        if abs_dx + abs_dy == 1:
            return True

        mode = self.current_mode

        if mode == "knight":
            return (abs_dx, abs_dy) in {(1, 2), (2, 1)}

        if mode == "king":
            return max(abs_dx, abs_dy) == 1 and (abs_dx + abs_dy) > 0

        if mode == "queen" and (delta_x == 0 or delta_y == 0 or abs_dx == abs_dy):
            return self._line_clear(src_x, src_y, dst_x, dst_y)

        return False

    def _goal_requirement_met(self) -> bool:

        if self.required_piece_count <= 0:
            return True
        return len(self.activated) >= self.required_piece_count

    def _show_cursor(self) -> None:

        self._cursor_active = True
        self._cursor_sprite.set_visible(True)

    def _hide_cursor(self) -> None:

        self._cursor_active = False
        self._cursor_sprite.set_visible(False)

    def _lose_life(self) -> None:

        self._lives -= 1
        if self._lives > 0:
            self.level_reset()
        else:
            self._was_game_over = True
            self._available_actions = [GameAction.RESET.value]
            self.lose()



    def _move_cursor(self, delta_x, delta_y):
        size = self.current_level.grid_size
        assert size is not None, "grid_size must be set before moving cursor"
        grid_width, grid_height = size
        nx = max(1, min(self._cursor_x + delta_x, grid_width - 2))
        ny = max(1, min(self._cursor_y + delta_y, grid_height - 2))
        self._cursor_x = nx
        self._cursor_y = ny
        self._cursor_sprite.set_position(nx, ny)
        self._show_cursor()

        self._tick_stun()
        self.moves_left -= 1
        if self.moves_left <= 0:
            self._lose_life()

    def _execute_move(self, src_x, src_y, dst_x, dst_y):

        self._tick_stun()
        self.moves_left -= 1

        if not self._legal_move(src_x, src_y, dst_x, dst_y):
            if self.moves_left <= 0:
                self._lose_life()
            return

        self.player.set_position(dst_x, dst_y)

        mode_before_landing = self.current_mode
        landed_piece = self._piece_at(dst_x, dst_y)
        if landed_piece and landed_piece not in self.activated:
            self.activated.add(landed_piece)
            self.current_mode = landed_piece

        if (dst_x, dst_y) in self._penalty_set:
            self.moves_left -= 3
            if self.moves_left <= 0:
                self._refresh_piece_visuals()
                self._refresh_player_visual()
                self._lose_life()
                return

        if (dst_x, dst_y) in self._mine_set:
            self.moves_left -= 4
            self._stun_nearby_pieces(dst_x, dst_y)
            if not self._piece_active(dst_x, dst_y, mode_before_landing):
                self.current_mode = "pawn"
            if self.moves_left <= 0:
                self._refresh_piece_visuals()
                self._refresh_player_visual()
                self._lose_life()
                return

        self._refresh_piece_visuals()
        self._refresh_player_visual()

        if (dst_x, dst_y) in self._goal_set and self._goal_requirement_met():
            self._lives = MAX_LIVES
            self._consecutive_reset_count = 0
            self._level_done = True
            self._undo_stack = []
            if self.is_last_level():
                self._available_actions = [GameAction.RESET.value]
            self.next_level()
            return

        if self.moves_left <= 0:
            self._lose_life()



    def _handle_mouse_click(self) -> None:

        src_x, src_y = self.player.x, self.player.y
        raw_x = int(self.action.data.get("x", 0))
        raw_y = int(self.action.data.get("y", 0))
        coords = self.camera.display_to_grid(raw_x, raw_y)
        if not coords:
            self._tick_stun()
            self.moves_left -= 1
            if self.moves_left <= 0:
                self._lose_life()
            self.complete_action()
            return
        dst_x, dst_y = coords
        self._hide_cursor()
        self._execute_move(src_x, src_y, dst_x, dst_y)
        self.complete_action()

    def _handle_cursor_move(self, delta_x: int, delta_y: int) -> None:

        self._move_cursor(delta_x, delta_y)
        self.complete_action()

    def _handle_confirm(self) -> None:

        src_x, src_y = self.player.x, self.player.y
        if self._cursor_active:
            dst_x, dst_y = self._cursor_x, self._cursor_y
            self._execute_move(src_x, src_y, dst_x, dst_y)
            self._cursor_x = self.player.x
            self._cursor_y = self.player.y
            self._cursor_sprite.set_position(self._cursor_x, self._cursor_y)
        else:
            self._tick_stun()
            self.moves_left -= 1
            if self.moves_left <= 0:
                self._lose_life()
        self.complete_action()



    def step(self) -> None:
        if self.action.id == GameAction.RESET:
            self.complete_action()
            return

        if self.action.id == GameAction.ACTION7:
            self._restore_undo_snapshot()
            self._consume_move()
            self.complete_action()
            return

        self._save_undo_snapshot()

        if self.action.id == GameAction.ACTION6:
            self._handle_mouse_click()
            return

        if self.action.id == GameAction.ACTION1:
            self._handle_cursor_move(0, -1)
            return

        if self.action.id == GameAction.ACTION2:
            self._handle_cursor_move(0, 1)
            return

        if self.action.id == GameAction.ACTION3:
            self._handle_cursor_move(-1, 0)
            return

        if self.action.id == GameAction.ACTION4:
            self._handle_cursor_move(1, 0)
            return

        if self.action.id == GameAction.ACTION5:
            self._handle_confirm()
            return

        self.complete_action()

    def _save_undo_snapshot(self) -> None:
        self._undo_stack.append({
            "player_x": self.player.x,
            "player_y": self.player.y,
            "moves_left": self.moves_left,
            "activated": set(self.activated),
            "current_mode": self.current_mode,
            "stun_turns": dict(self.stun_turns),
        })

    def _restore_undo_snapshot(self) -> None:
        if not self._undo_stack:
            return
        state = self._undo_stack.pop()
        self.player.set_position(state["player_x"], state["player_y"])
        self.activated = set(state["activated"])
        self.current_mode = state["current_mode"]
        self.stun_turns = dict(state["stun_turns"])
        self._refresh_piece_visuals()
        self._refresh_player_visual()

    def _consume_move(self) -> None:
        self.moves_left -= 1
        if self.moves_left <= 0:
            self._lose_life()


class PuzzleEnvironment:
    _ACTION_MAP = {
        "up": GameAction.ACTION1,
        "down": GameAction.ACTION2,
        "left": GameAction.ACTION3,
        "right": GameAction.ACTION4,
        "select": GameAction.ACTION5,
        "click": GameAction.ACTION6,
        "undo": GameAction.ACTION7,
    }

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self._engine = Ch91(seed=seed)
        self._turn = 0
        self._done = False
        self._game_over_flag = False
        self._game_won = False
        self._last_action_was_reset = False
        self._levels_completed = 0
        self._total_levels = len(self._engine._levels)

    def reset(self) -> GameState:
        e = self._engine
        if self._game_won or self._last_action_was_reset:
            e.perform_action(ActionInput(id=GameAction.RESET, data={}))
            e.perform_action(ActionInput(id=GameAction.RESET, data={}))
        else:
            e.perform_action(ActionInput(id=GameAction.RESET, data={}))
        self._turn = 0
        self._done = False
        self._game_over_flag = False
        self._last_action_was_reset = True
        self._game_won = False
        self._levels_completed = 0
        return self._get_state()

    def get_actions(self) -> List[str]:
        if self._done or self._game_over_flag:
            return ["reset"]
        return ["reset", "up", "down", "left", "right", "select", "click", "undo"]

    def step(self, action: str) -> StepResult:
        e = self._engine

        if action == "reset":
            state = self.reset()
            return StepResult(state=state, reward=0.0, done=False, info={"action": "reset"})

        game_action = self._ACTION_MAP.get(action)
        if game_action is None:
            game_action = GameAction.RESET

        self._last_action_was_reset = False
        self._turn += 1
        total_levels = len(e._levels)
        level_before = e.level_index

        frame = e.perform_action(ActionInput(id=game_action, data={}), raw=True)

        state_name = frame.state.name if frame and frame.state else ""
        game_won = state_name == "WIN"
        game_over = state_name == "GAME_OVER"

        level_reward = 1.0 / total_levels

        if game_won:
            self._done = True
            self._game_won = True
            self._levels_completed += 1
            return StepResult(
                state=self._get_state(),
                reward=level_reward,
                done=True,
                info={"reason": "game_complete"},
            )

        if game_over:
            self._game_over_flag = True
            return StepResult(
                state=self._get_state(),
                reward=0.0,
                done=False,
                info={"reason": "game_over"},
            )

        reward = 0.0
        if e.level_index != level_before:
            reward = level_reward
            self._levels_completed += 1

        return StepResult(
            state=self._get_state(),
            reward=reward,
            done=False,
            info={},
        )

    def _get_state(self) -> GameState:
        valid_actions = self.get_actions()
        text_obs = f"Level {self._engine.level_index + 1}, Turn {self._turn}, Actions: {valid_actions}"
        image_obs = self._render_png()
        return GameState(
            text_observation=text_obs,
            image_observation=image_obs,
            valid_actions=valid_actions,
            turn=self._turn,
            metadata={
                "levels_completed": self._levels_completed,
                "level_index": self._engine.level_index,
                "game_over": getattr(getattr(self._engine, "_state", None), "name", "") == "GAME_OVER",
                "total_levels": self._total_levels,
                "lives": getattr(self._engine, "_lives", 3),
                "done": self._done,
                "info": {},
            },
        )

    def _render_png(self) -> Optional[bytes]:
        import struct
        import zlib
        frame = self.render(mode="rgb_array")
        h, w = frame.shape[:2]
        r, g, b = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
        img_data = b + (g << 8) + (r << 16)
        raw = struct.pack(f"<{h * w}I", *img_data.flat)
        compressed = zlib.compress(raw, 9)
        return struct.pack("<I", h) + struct.pack("<I", w) + compressed

    def is_done(self) -> bool:
        return self._done

    def _build_text_observation(self) -> str:
        return f"Level {self._engine.level_index + 1}, Turn {self._turn}, Actions: {self.get_actions()}"

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        frame = self._engine.camera.render(self._engine.current_level.get_sprites())
        h, w = frame.shape[:2]
        if h != 64 or w != 64:
            row_idx = (np.arange(64) * h // 64).astype(int)
            col_idx = (np.arange(64) * w // 64).astype(int)
            frame = frame[np.ix_(row_idx, col_idx)]
        rgb = np.empty((64, 64, 3), dtype=np.uint8)
        for i in range(64):
            for j in range(64):
                rgb[i, j] = _ARC_PALETTE[frame[i, j]]
        return rgb

    def close(self) -> None:
        self._engine = None


class ArcGameEnv(gym.Env):
    metadata: Dict[str, Any] = {"render_modes": ["rgb_array"], "render_fps": 5}
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

    def __init__(self, seed: int = 0, render_mode: Optional[str] = None) -> None:
        super().__init__()
        self._seed = seed
        self.render_mode = render_mode
        self._action_to_string = {i: a for i, a in enumerate(self.ACTION_LIST)}
        self._string_to_action = {a: i for i, a in enumerate(self.ACTION_LIST)}
        self.observation_space = spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8)
        self.action_space = spaces.Discrete(len(self.ACTION_LIST))
        self._env: Optional[PuzzleEnvironment] = None

    def reset(self, *, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed
        self._env = PuzzleEnvironment(seed=self._seed)
        state = self._env.reset()
        return self._get_obs(), self._build_info(state)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action_str = self._action_to_string[int(action)]
        result = self._env.step(action_str)
        obs = self._get_obs()
        reward = result.reward
        terminated = result.done
        truncated = False
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
        if self._env is None:
            return np.zeros((64, 64, 3), dtype=np.uint8)
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

    def _build_info(self, state: GameState, step_info: Optional[Dict] = None) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "text_observation": state.text_observation,
            "valid_actions": state.valid_actions,
            "turn": state.turn,
            "game_metadata": state.metadata,
        }
        if step_info:
            info["step_info"] = step_info
        return info
