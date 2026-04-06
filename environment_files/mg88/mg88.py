import io
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


BACKGROUND_COLOR = 5
PADDING_COLOR = 4

GRID_MIN = 7
GRID_MAX = 11
SPAWN_CYCLE = 3
MAX_REPLICAS = 4
MAX_LIVES = 3

CLR_PLAYER = 0
CLR_REPLICA_1 = 10
CLR_REPLICA_2 = 15
CLR_REPLICA_3 = 12
CLR_WALL = 4
CLR_OBJECT = 11
CLR_PATTERN = 10
CLR_EXIT = 14
CLR_BAR_LOW = 12
CLR_LIFE = 0
CLR_LIFE_LOST = 4

REPLICA_COLORS = [CLR_REPLICA_1, CLR_REPLICA_2, CLR_REPLICA_3]

sprites = {
    "plr": Sprite(
        pixels=[[CLR_PLAYER]],
        name="plr",
        visible=True,
        collidable=False,
        tags=["plr"],
        layer=5,
    ),
    "wal": Sprite(
        pixels=[[CLR_WALL]],
        name="wal",
        visible=True,
        collidable=True,
        tags=["wal"],
        layer=0,
    ),
    "obj": Sprite(
        pixels=[[CLR_OBJECT]],
        name="obj",
        visible=True,
        collidable=True,
        tags=["obj"],
        layer=3,
    ),
    "ptn": Sprite(
        pixels=[[CLR_PATTERN]],
        name="ptn",
        visible=True,
        collidable=False,
        tags=["ptn"],
        layer=-1,
    ),
    "ext": Sprite(
        pixels=[[CLR_EXIT]],
        name="ext",
        visible=True,
        collidable=False,
        tags=["ext"],
        layer=-1,
    ),
    "rp0": Sprite(
        pixels=[[CLR_REPLICA_1]],
        name="rp0",
        visible=False,
        collidable=False,
        tags=["replica", "rp0"],
        layer=4,
    ),
    "rp1": Sprite(
        pixels=[[CLR_REPLICA_2]],
        name="rp1",
        visible=False,
        collidable=False,
        tags=["replica", "rp1"],
        layer=4,
    ),
    "rp2": Sprite(
        pixels=[[CLR_REPLICA_3]],
        name="rp2",
        visible=False,
        collidable=False,
        tags=["replica", "rp2"],
        layer=4,
    ),
    "rp3": Sprite(
        pixels=[[CLR_REPLICA_1]],
        name="rp3",
        visible=False,
        collidable=False,
        tags=["replica", "rp3"],
        layer=4,
    ),
    "rp4": Sprite(
        pixels=[[CLR_REPLICA_2]],
        name="rp4",
        visible=False,
        collidable=False,
        tags=["replica", "rp4"],
        layer=4,
    ),
    "rp5": Sprite(
        pixels=[[CLR_REPLICA_3]],
        name="rp5",
        visible=False,
        collidable=False,
        tags=["replica", "rp5"],
        layer=4,
    ),
    "rp6": Sprite(
        pixels=[[CLR_REPLICA_1]],
        name="rp6",
        visible=False,
        collidable=False,
        tags=["replica", "rp6"],
        layer=4,
    ),
    "rp7": Sprite(
        pixels=[[CLR_REPLICA_2]],
        name="rp7",
        visible=False,
        collidable=False,
        tags=["replica", "rp7"],
        layer=4,
    ),
}

_TILE_TO_SPRITE = {
    "#": "wal",
    "P": "plr",
    "O": "obj",
    "T": "ptn",
    "E": "ext",
}


_L1 = [
    "#########",
    "#......T#",
    "#.......#",
    "#....T..#",
    "#.......#",
    "#..T....#",
    "#.......#",
    "#P.T...E#",
    "#########",
]

_L2 = [
    "#########",
    "#......T#",
    "#.......#",
    "#..T..T.#",
    "#.......#",
    "#...T...#",
    "#.......#",
    "#PT....E#",
    "#########",
]

_L3 = [
    "###########",
    "#........T#",
    "#.........#",
    "#.....T...#",
    "#.........#",
    "#..T......#",
    "#......T..#",
    "#...T.....#",
    "#PT......E#",
    "###########",
]

_L4 = [
    "###########",
    "#.......TT#",
    "#.........#",
    "#.....T...#",
    "#.......T.#",
    "#..T..T...#",
    "#.........#",
    "#.........#",
    "#PT......E#",
    "###########",
]


def _build_level(map_rows, data):
    sprs = []
    h = len(map_rows)
    w = len(map_rows[0]) if h > 0 else 0
    for y, row in enumerate(map_rows):
        for x, ch in enumerate(row):
            key = _TILE_TO_SPRITE.get(ch)
            if key:
                sprs.append(sprites[key].clone().set_position(x, y))

    for i in range(MAX_REPLICAS):
        sprs.append(sprites[f"rp{i}"].clone().set_position(0, 0))

    return Level(sprites=sprs, grid_size=(w, h), data=data)


levels = [
    _build_level(
        _L1,
        {
            "move_limit": 44,
            "rank_mode": "even",
            "replicas": 2,
            "replica_spawns": [[1, 1], [7, 5], [7, 1]],
        },
    ),
    _build_level(
        _L2,
        {
            "move_limit": 44,
            "rank_mode": "odd",
            "replicas": 2,
            "replica_spawns": [[1, 1], [7, 1], [7, 5], [1, 5]],
        },
    ),
    _build_level(
        _L3,
        {
            "move_limit": 62,
            "rank_mode": "even",
            "replicas": 3,
            "replica_spawns": [[1, 1], [9, 1], [9, 7], [1, 7], [5, 1]],
        },
    ),
    _build_level(
        _L4,
        {
            "move_limit": 40,
            "rank_mode": "odd",
            "replicas": 4,
            "replica_spawns": [[1, 1], [9, 9], [1, 9], [9, 1]],
        },
    ),
]


class HintGridDisplay(RenderableUserDisplay):
    CLR_HINT_BG = 5
    CLR_HINT_BORDER = 3
    CLR_DOT_REQUIRED = 0
    CLR_DOT_DONE = 14
    CLR_DOT_NEXT = 11

    MARGIN = 1

    def __init__(self, game: "Mg88") -> None:
        self.game = game
        self._grid_rows = 0
        self._grid_cols = 0
        self._target_cells: list = []

    def rebuild(self) -> None:
        g = self.game
        self._grid_rows = g.grid_h
        self._grid_cols = g.grid_w

        self._target_cells = []
        for (tx, ty), rank in g.target_rank_by_pos.items():
            if rank in g.required_ranks:
                self._target_cells.append((ty, tx, rank))

    @staticmethod
    def _set_pixel(frame, fy, fx, color):
        fh, fw = frame.shape
        if 0 <= fy < fh and 0 <= fx < fw:
            frame[fy, fx] = color

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        g = self.game
        if self._grid_rows == 0:
            return frame

        ox = self.MARGIN
        oy = self.MARGIN
        w = self._grid_cols
        h = self._grid_rows

        for dy in range(h):
            for dx in range(w):
                self._set_pixel(frame, oy + dy, ox + dx, self.CLR_HINT_BG)

        for dx in range(-1, w + 1):
            self._set_pixel(frame, oy - 1, ox + dx, self.CLR_HINT_BORDER)
            self._set_pixel(frame, oy + h, ox + dx, self.CLR_HINT_BORDER)
        for dy in range(-1, h + 1):
            self._set_pixel(frame, oy + dy, ox - 1, self.CLR_HINT_BORDER)
            self._set_pixel(frame, oy + dy, ox + w, self.CLR_HINT_BORDER)

        done_count = g.visit_progress
        next_rank = (
            g.required_ranks[done_count] if done_count < len(g.required_ranks) else None
        )

        for row, col, rank in self._target_cells:
            req_index = g.required_ranks.index(rank)
            if req_index < done_count:
                color = self.CLR_DOT_DONE
            elif rank == next_rank and not g.exit_ready:
                color = self.CLR_DOT_NEXT
            else:
                color = self.CLR_DOT_REQUIRED
            self._set_pixel(frame, oy + row, ox + col, color)

        return frame


class ProgressDisplay(RenderableUserDisplay):
    def __init__(self, game: "Mg88") -> None:
        self.game = game

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        g = self.game
        fh, fw = frame.shape

        cam_w = GRID_MAX
        cam_h = GRID_MAX
        scale = min(fw // cam_w, fh // cam_h)
        half = max(1, scale // 3)
        x_off = (fw - cam_w * scale) // 2
        y_off = (fh - cam_h * scale) // 2

        def gx(col):
            return slice(x_off + col * scale, x_off + (col + 1) * scale)

        if g.move_limit <= 0:
            return frame

        remaining = max(0, g.move_limit - g.move_count)
        ratio = remaining / g.move_limit
        cells_filled = int(ratio * cam_w)
        bar_y = slice(
            y_off + (cam_h - 1) * scale + (scale - half), y_off + cam_h * scale
        )

        for col in range(cam_w):
            if col < cells_filled:
                frame[bar_y, gx(col)] = CLR_PATTERN if ratio > 0.25 else CLR_BAR_LOW
            else:
                frame[bar_y, gx(col)] = BACKGROUND_COLOR

        life_y = slice(y_off, y_off + half)
        for i in range(MAX_LIVES):
            col = cam_w - MAX_LIVES + i
            if i < g.lives:
                frame[life_y, gx(col)] = CLR_LIFE
            else:
                frame[life_y, gx(col)] = CLR_LIFE_LOST

        return frame


class Mg88(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

        self.player = None
        self.exit_sprite = None
        self.spawn_x = 0
        self.spawn_y = 0
        self.grid_w = GRID_MIN
        self.grid_h = GRID_MIN

        self.walls = set()
        self.push_objects = []

        self.move_count = 0
        self.move_limit = 0
        self.move_history = []
        self.player_history = []
        self.lives = MAX_LIVES
        self.replica_limit = 0
        self.replica_spawn_points = []
        self.replica_spawn_index = 0

        self.replica_pool = []
        self.replicas = []
        self.rank_mode = "odd"
        self.required_ranks = []
        self.target_rank_by_pos = {}
        self.visit_progress = 0
        self.exit_ready = False
        self.sequence_failed = False
        self._history: list = []
        self.reset_press_count = 0
        self.play_since_reset = False
        self._hud = ProgressDisplay(self)
        self._hint = HintGridDisplay(self)

        super().__init__(
            "mg88",
            levels,
            Camera(
                0,
                0,
                GRID_MAX,
                GRID_MAX,
                BACKGROUND_COLOR,
                PADDING_COLOR,
                [self._hint, self._hud],
            ),
            False,
            1,
            [0, 1, 2, 3, 4, 7],
        )

    def on_set_level(self, level: Level) -> None:
        self.lives = MAX_LIVES
        self.player = self.current_level.get_sprites_by_tag("plr")[0]
        self.spawn_x = self.player.x
        self.spawn_y = self.player.y

        self.grid_w, self.grid_h = self.current_level.grid_size

        self.walls = {(s.x, s.y) for s in self.current_level.get_sprites_by_tag("wal")}
        self.push_objects = list(self.current_level.get_sprites_by_tag("obj"))

        exits = self.current_level.get_sprites_by_tag("ext")
        self.exit_sprite = exits[0] if exits else None

        mv = self.current_level.get_data("move_limit")
        self.move_limit = int(mv) if mv is not None else 999

        raw_rank_mode = self.current_level.get_data("rank_mode")
        self.rank_mode = str(raw_rank_mode) if raw_rank_mode is not None else "odd"

        raw_replica_limit = self.current_level.get_data("replicas")
        self.replica_limit = (
            int(raw_replica_limit) if raw_replica_limit is not None else 0
        )
        raw_spawn_points = self.current_level.get_data("replica_spawns")
        self.replica_spawn_points = (
            [tuple(p) for p in raw_spawn_points] if raw_spawn_points is not None else []
        )
        self.replica_pool = self.current_level.get_sprites_by_tag("replica")
        self._reset_level_state()

    def _reset_level_state(self) -> None:
        self.player.set_position(self.spawn_x, self.spawn_y)
        self.move_count = 0
        self.move_history = []
        self.player_history = [(self.spawn_x, self.spawn_y)]
        self.replica_spawn_index = 0
        self.replicas = []
        self.visit_progress = 0
        self.exit_ready = False
        self.sequence_failed = False
        self._history = []
        for spr in self.replica_pool:
            spr.set_visible(False)
            spr.set_position(0, 0)
        self._rebuild_target_ranks()
        self._hint.rebuild()

    def _save_snapshot(self) -> None:
        obj_positions = [(obj.x, obj.y) for obj in self.push_objects]
        replica_snap = [
            {
                "slot": r["slot"],
                "spawn_x": r["spawn_x"],
                "spawn_y": r["spawn_y"],
                "x": r["x"],
                "y": r["y"],
            }
            for r in self.replicas
        ]
        self._history.append(
            {
                "player_x": self.player.x,
                "player_y": self.player.y,
                "move_count": self.move_count,
                "move_history": list(self.move_history),
                "player_history": list(self.player_history),
                "replica_spawn_index": self.replica_spawn_index,
                "replicas": replica_snap,
                "obj_positions": obj_positions,
                "visit_progress": self.visit_progress,
                "exit_ready": self.exit_ready,
                "sequence_failed": self.sequence_failed,
            }
        )

    def _undo(self) -> None:
        if not self._history:
            return
        snap = self._history.pop()
        self.player.set_position(snap["player_x"], snap["player_y"])
        self.move_history = snap["move_history"]
        self.player_history = snap["player_history"]
        self.replica_spawn_index = snap["replica_spawn_index"]
        self.visit_progress = snap["visit_progress"]
        self.exit_ready = snap["exit_ready"]
        self.sequence_failed = snap["sequence_failed"]

        for obj, (ox, oy) in zip(self.push_objects, snap["obj_positions"]):
            obj.set_position(ox, oy)

        for spr in self.replica_pool:
            spr.set_visible(False)
            spr.set_position(0, 0)

        self.replicas = []
        for rs in snap["replicas"]:
            spr = self.replica_pool[rs["slot"]]
            spr.set_position(rs["x"], rs["y"])
            spr.set_visible(True)
            self.replicas.append(
                {
                    "slot": rs["slot"],
                    "sprite": spr,
                    "spawn_x": rs["spawn_x"],
                    "spawn_y": rs["spawn_y"],
                    "x": rs["x"],
                    "y": rs["y"],
                }
            )

    def _lose_life(self) -> None:
        self.lives -= 1
        if self.lives > 0:
            self._reset_level_state()
        else:
            self.lose()

    def handle_reset(self) -> None:
        if not self.play_since_reset:
            self.reset_press_count += 1
        else:
            self.reset_press_count = 1

        if self.reset_press_count >= 2:
            self.lives = MAX_LIVES
            self.reset_press_count = 0
            self.full_reset()
        elif self._action_count > 0 and self.lives > 0:
            self._lose_life()
        else:
            super().handle_reset()

        self.play_since_reset = False
        self._history = []

    def _rebuild_target_ranks(self) -> None:
        targets = self.current_level.get_sprites_by_tag("ptn")
        ranked = sorted(
            (
                (abs(s.x - self.spawn_x) + abs(s.y - self.spawn_y), s.x, s.y, s)
                for s in targets
            )
        )
        self.target_rank_by_pos = {}
        for idx, (_, tx, ty, _spr) in enumerate(ranked, start=1):
            self.target_rank_by_pos[(tx, ty)] = idx

        if self.rank_mode == "even":
            self.required_ranks = [
                idx for idx in range(1, len(ranked) + 1) if idx % 2 == 0
            ]
        else:
            self.required_ranks = [
                idx for idx in range(1, len(ranked) + 1) if idx % 2 == 1
            ]

        required_set = set(self.required_ranks)
        hide_decoys = self._current_level_index < 2
        for idx, (_, tx, ty, spr) in enumerate(ranked, start=1):
            if idx in required_set:
                spr.set_visible(True)
            elif hide_decoys:
                spr.set_visible(False)
                del self.target_rank_by_pos[(tx, ty)]
            else:
                spr.set_visible(True)

    def _inside(self, x: int, y: int) -> bool:
        return 0 <= x < self.grid_w and 0 <= y < self.grid_h

    def _replica_at(self, x: int, y: int, exclude_index=None):
        for i, rep in enumerate(self.replicas):
            if i == exclude_index:
                continue
            if rep["x"] == x and rep["y"] == y:
                return i
        return None

    def _object_at(self, x: int, y: int):
        for obj in self.push_objects:
            if obj.x == x and obj.y == y:
                return obj
        return None

    def _player_blocks(self, x: int, y: int) -> bool:
        return self.player.x == x and self.player.y == y

    def _tile_blocked_for_actor(
        self, x: int, y: int, actor_type: str, actor_index=None
    ) -> bool:
        if not self._inside(x, y):
            return True
        if (x, y) in self.walls:
            return True

        if actor_type == "player":
            return False
        else:
            if self._player_blocks(x, y):
                return True
            if self._replica_at(x, y, exclude_index=actor_index) is not None:
                return True

        return False

    def _push_object_if_needed(
        self, x: int, y: int, dx: int, dy: int, actor_type: str, actor_index=None
    ) -> bool:
        obj = self._object_at(x, y)
        if obj is None:
            return True

        nx, ny = obj.x + dx, obj.y + dy
        if not self._inside(nx, ny):
            return False
        if (nx, ny) in self.walls:
            return False
        if self._object_at(nx, ny) is not None:
            return False

        if actor_type == "player":
            if self._replica_at(nx, ny) is not None:
                return False
        else:
            if self._player_blocks(nx, ny):
                return False
            if self._replica_at(nx, ny, exclude_index=actor_index) is not None:
                return False

        obj.set_position(nx, ny)
        return True

    def _move_player(self, dx: int, dy: int) -> bool:
        tx, ty = self.player.x + dx, self.player.y + dy
        if not self._inside(tx, ty):
            return False
        if (tx, ty) in self.walls:
            return False
        if self._replica_at(tx, ty) is not None:
            return True
        if not self._push_object_if_needed(tx, ty, dx, dy, "player"):
            return False
        self.player.set_position(tx, ty)
        return False

    def _move_replica(self, idx: int, player_dx: int, player_dy: int) -> bool:
        rep = self.replicas[idx]

        if idx % 2 == 0:
            mdx, mdy = player_dx, player_dy
        else:
            mdx, mdy = -player_dx, -player_dy

        tx = rep["x"] + mdx
        ty = rep["y"] + mdy
        if tx == self.player.x and ty == self.player.y:
            return True

        if not self._tile_blocked_for_actor(tx, ty, "replica", actor_index=idx):
            if self._push_object_if_needed(
                tx, ty, mdx, mdy, "replica", actor_index=idx
            ):
                rep["x"] = tx
                rep["y"] = ty
                rep["sprite"].set_position(tx, ty)
        return False

    def _spawn_replica(self) -> None:
        if len(self.replicas) >= min(MAX_REPLICAS, self.replica_limit):
            return
        if not self.replica_spawn_points:
            return

        slot = len(self.replicas)
        spr = self.replica_pool[slot]
        occupied = {(self.player.x, self.player.y)}
        for rep in self.replicas:
            occupied.add((rep["x"], rep["y"]))

        candidates = [
            tuple(p) for p in self.replica_spawn_points if tuple(p) not in occupied
        ]
        if not candidates:
            return

        spawn_x = spawn_y = None
        count = len(self.replica_spawn_points)
        for offset in range(count):
            idx = (self.replica_spawn_index + offset) % count
            candidate_x, candidate_y = self.replica_spawn_points[idx]
            if (candidate_x, candidate_y) in occupied:
                continue
            spawn_x, spawn_y = candidate_x, candidate_y
            self.replica_spawn_index = (idx + 1) % count
            break
        if spawn_x is None or spawn_y is None:
            return

        spr.set_position(spawn_x, spawn_y)
        spr.set_visible(True)

        self.replicas.append(
            {
                "slot": slot,
                "sprite": spr,
                "spawn_x": spawn_x,
                "spawn_y": spawn_y,
                "x": spawn_x,
                "y": spawn_y,
            }
        )

    def _update_visit_progress(self) -> None:
        if self.exit_ready or self.sequence_failed:
            return

        if self.visit_progress >= len(self.required_ranks):
            return

        rank = self.target_rank_by_pos.get((self.player.x, self.player.y))
        if rank is None:
            return

        needed = self.required_ranks[self.visit_progress]
        if rank == needed:
            self.visit_progress += 1
            if self.visit_progress >= len(self.required_ranks):
                self.exit_ready = True
        else:
            self.sequence_failed = True

    def _decode_action(self):
        if self.action.id == GameAction.ACTION1:
            return (0, -1)
        if self.action.id == GameAction.ACTION2:
            return (0, 1)
        if self.action.id == GameAction.ACTION3:
            return (-1, 0)
        if self.action.id == GameAction.ACTION4:
            return (1, 0)
        return (0, 0)

    def step(self) -> None:
        if self.action.id == GameAction.RESET:
            self.complete_action()
            return

        self.play_since_reset = True

        if self.action.id == GameAction.ACTION7:
            self._undo()
            self.move_count += 1
            if self.move_count >= self.move_limit:
                self._lose_life()
            self.complete_action()
            return

        self._save_snapshot()

        dx, dy = self._decode_action()

        armed_before_move = self.exit_ready
        touched_replica = self._move_player(dx, dy)
        if touched_replica:
            self._lose_life()
            self.complete_action()
            return

        self.move_history.append((dx, dy))
        self.move_count += 1
        self.player_history.append((self.player.x, self.player.y))
        self._update_visit_progress()

        if armed_before_move:
            self.next_level()
            self.complete_action()
            return

        if self.move_count % SPAWN_CYCLE == 0:
            self._spawn_replica()

        for i in range(len(self.replicas)):
            if self._move_replica(i, dx, dy):
                self._lose_life()
                self.complete_action()
                return

        if self.move_count >= self.move_limit:
            self._lose_life()
            self.complete_action()
            return

        self.complete_action()


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
        self._engine: Any = Mg88(seed=seed)
        self._done = False
        self._game_won = False
        self._last_action_was_reset = False
        self._total_turns = 0
        self._total_levels = len(levels)

    @staticmethod
    def _frame_to_png(frame: np.ndarray) -> bytes:
        h, w = frame.shape[:2]
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for idx in range(len(ARC_PALETTE)):
            mask = frame == idx
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            rgb[mask] = ARC_PALETTE[idx]

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
        total_targets = len(e.required_ranks)
        visited = e.visit_progress
        active_replicas = len(e.replicas)

        header = (
            f"Level {e._current_level_index + 1}/{len(e._levels)}"
            f" | Moves: {e.move_count}/{e.move_limit}"
            f" | Lives: {e.lives}"
        )
        rank_line = (
            f"Rank mode: {e.rank_mode} | Visited {visited}/{total_targets} targets"
        )
        exit_line = f"Exit ready: {e.exit_ready} | Sequence failed: {e.sequence_failed}"
        replica_line = f"Active replicas: {active_replicas}"
        player_line = f"Player: ({e.player.x}, {e.player.y})"
        required_line = f"Required ranks: {e.required_ranks}"
        grid_line = f"Grid: {e.grid_w}x{e.grid_h}"

        rules = (
            "Visit targets in ranked order (odd/even mode). "
            "Reach exit after all required targets. Avoid replicas."
        )

        return "\n".join(
            [
                header,
                rules,
                rank_line,
                exit_line,
                replica_line,
                player_line,
                required_line,
                grid_line,
            ]
        )

    def _build_game_state(self, done: bool = False) -> GameState:
        e = self._engine
        frame = self._render_frame()
        image_bytes = self._frame_to_png(frame)

        valid_actions = list(self._VALID_ACTIONS) if not done else None

        return GameState(
            text_observation=self._build_text_observation(),
            image_observation=image_bytes,
            valid_actions=valid_actions,
            turn=self._total_turns,
            metadata={
                "total_levels": len(e._levels),
                "level_index": e._current_level_index,
                "levels_completed": getattr(e, "_score", 0),
                "game_over": getattr(getattr(e, "_state", None), "name", "")
                == "GAME_OVER",
                "done": done,
                "info": {},
                "move_count": e.move_count,
                "move_limit": e.move_limit,
                "lives": e.lives,
                "rank_mode": e.rank_mode,
                "visit_progress": e.visit_progress,
                "exit_ready": e.exit_ready,
                "sequence_failed": e.sequence_failed,
                "grid_w": e.grid_w,
                "grid_h": e.grid_h,
                "replicas_active": len(e.replicas),
                "player_x": e.player.x,
                "player_y": e.player.y,
            },
        )

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

        return self._build_game_state()

    def is_done(self) -> bool:
        return self._done

    def get_actions(self) -> List[str]:
        if self._done:
            return ["reset"]
        return list(self._VALID_ACTIONS)

    def step(self, action: str) -> StepResult:
        e = self._engine

        if action == "reset":
            state = self.reset()
            return StepResult(
                state=state, reward=0.0, done=False, info={"action": "reset"}
            )

        if action not in self._ACTION_MAP:
            raise ValueError(
                f"Invalid action '{action}'. Must be one of {list(self._ACTION_MAP.keys())}"
            )

        self._last_action_was_reset = False
        self._total_turns += 1

        game_action = self._ACTION_MAP[action]
        info: Dict[str, Any] = {"action": action}

        level_before = e.level_index

        frame = e.perform_action(ActionInput(id=game_action), raw=True)

        state_name = frame.state.name if frame and frame.state else ""
        game_won = state_name == "WIN"
        game_over = state_name == "GAME_OVER"

        total_levels = len(self._engine._levels)
        level_reward = 1.0 / total_levels

        if game_won:
            self._done = True
            self._game_won = True
            info["reason"] = "game_complete"
            return StepResult(
                state=self._build_game_state(done=True),
                reward=level_reward,
                done=True,
                info=info,
            )

        if game_over:
            self._done = True
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
