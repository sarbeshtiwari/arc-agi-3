from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import struct
import zlib
from random import Random

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
    metadata: Dict = field(default_factory=dict)


@dataclass
class StepResult:
    state: GameState
    reward: float
    done: bool
    info: Dict = field(default_factory=dict)


BACKGROUND_COLOR = 0

PADDING_COLOR = 5

GRID_WIDTH = 16

GRID_HEIGHT = 14

ECHO_CYCLE = 5

MAX_ECHOES = 3

CLR_PLAYER = 4

CLR_WALL = 5

CLR_PLATE_OFF = 9

CLR_PLATE_ON = 3

CLR_DOOR = 1

CLR_EXIT = 14

CLR_HAZARD = 2

ECHO_COLORS = [8, 6, 11]

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
    "plt": Sprite(
        pixels=[[CLR_PLATE_OFF]],
        name="plt",
        visible=True,
        collidable=False,
        tags=["plt"],
        layer=-1,
    ),
    "dor": Sprite(
        pixels=[[CLR_DOOR]],
        name="dor",
        visible=True,
        collidable=True,
        tags=["dor"],
        layer=1,
    ),
    "ext": Sprite(
        pixels=[[CLR_EXIT]],
        name="ext",
        visible=True,
        collidable=False,
        tags=["ext"],
        layer=-1,
    ),
    "haz": Sprite(
        pixels=[[CLR_HAZARD]],
        name="haz",
        visible=True,
        collidable=False,
        tags=["haz"],
        layer=-1,
    ),
    "eg0": Sprite(
        pixels=[[ECHO_COLORS[0]]],
        name="eg0",
        visible=False,
        collidable=False,
        tags=["echo", "eg0"],
        layer=3,
    ),
    "eg1": Sprite(
        pixels=[[ECHO_COLORS[1]]],
        name="eg1",
        visible=False,
        collidable=False,
        tags=["echo", "eg1"],
        layer=3,
    ),
    "eg2": Sprite(
        pixels=[[ECHO_COLORS[2]]],
        name="eg2",
        visible=False,
        collidable=False,
        tags=["echo", "eg2"],
        layer=3,
    ),
    "dov": Sprite(
        pixels=[[CLR_HAZARD]],
        name="dov",
        visible=False,
        collidable=False,
        tags=["dov"],
        layer=10,
    ),
}

_MAP_CHARS = {
    "#": "wal",
    "P": "plr",
    "O": "plt",
    "D": "dor",
    "X": "ext",
    "H": "haz",
}

_L1 = [
    "################",
    "#..............#",
    "#P.............#",
    "#..............#",
    "#...O..........#",
    "########D#######",
    "#..............#",
    "#..............#",
    "#..............#",
    "#..............#",
    "#..............#",
    "#..............#",
    "#.............X#",
    "################",
]

_L2 = [
    "################",
    "#..............#",
    "#P.............#",
    "#..............#",
    "#...O..........#",
    "#..............#",
    "#...O..........#",
    "#..............#",
    "#..............#",
    "###########D####",
    "#..............#",
    "#..............#",
    "#.............X#",
    "################",
]

_L3 = [
    "################",
    "#..............#",
    "#P.............#",
    "#.HHH..........#",
    "#...O..........#",
    "########D#######",
    "#..............#",
    "#..........H...#",
    "#..............#",
    "#..............#",
    "#..............#",
    "#..............#",
    "#.............X#",
    "################",
]

_L4 = [
    "################",
    "#..............#",
    "#P.............#",
    "#.HHH..........#",
    "#...O..........#",
    "#..............#",
    "#...O..........#",
    "#.......H......#",
    "#..............#",
    "###########D####",
    "#..............#",
    "#.......H......#",
    "#.............X#",
    "################",
]

_L5 = [
    "################",
    "#..............#",
    "#P.............#",
    "#.HHH..........#",
    "#...O..........#",
    "#....H.........#",
    "#...H..........#",
    "#....HO........#",
    "#.......#.#....#",
    "#########D######",
    "#..............#",
    "#......H.......#",
    "#.............X#",
    "################",
]


def _build_level(mp, data):

    spr_list = []

    h = len(mp)

    w = len(mp[0]) if h > 0 else 0

    for y, row in enumerate(mp):
        for x, ch in enumerate(row):
            if ch == "P":
                data["origin_x"] = x
                data["origin_y"] = y
            if ch in _MAP_CHARS:
                spr_list.append(sprites[_MAP_CHARS[ch]].clone().set_position(x, y))

    for i in range(MAX_ECHOES):
        spr_list.append(sprites[f"eg{i}"].clone().set_position(0, 0))

    spr_list.append(sprites["dov"].clone().set_position(0, 0))

    return Level(sprites=spr_list, grid_size=(w, h), data=data)


levels = [
    _build_level(
        _L1,
        {
            "pairs": [{"plates": [[4, 4]], "door": [8, 5]}],
            "max_moves": 80,
        },
    ),
    _build_level(
        _L2,
        {
            "pairs": [{"plates": [[4, 4], [4, 6]], "door": [11, 9]}],
            "max_moves": 80,
        },
    ),
    _build_level(
        _L3,
        {
            "pairs": [{"plates": [[4, 4]], "door": [8, 5]}],
            "max_moves": 70,
        },
    ),
    _build_level(
        _L4,
        {
            "pairs": [{"plates": [[4, 4], [4, 6]], "door": [11, 9]}],
            "max_moves": 60,
        },
    ),
    _build_level(
        _L5,
        {
            "pairs": [{"plates": [[4, 4], [6, 7]], "door": [9, 9]}],
            "max_moves": 50,
        },
    ),
]


class HudDisplay(RenderableUserDisplay):
    def __init__(self, game: "Ea99") -> None:

        self.game = game

    def _grid_slicers(self, frame: np.ndarray):
        fh, fw = frame.shape
        cam_w, cam_h = GRID_WIDTH, GRID_HEIGHT
        scale = min(fw // cam_w, fh // cam_h)
        x_off = (fw - cam_w * scale) // 2
        y_off = (fh - cam_h * scale) // 2

        def gy(row):
            return slice(y_off + row * scale, y_off + (row + 1) * scale)

        def gx(col):
            return slice(x_off + col * scale, x_off + (col + 1) * scale)

        return gy, gx, cam_w, cam_h

    def _render_move_bar(self, frame, gy, gx, cam_w, cam_h) -> None:
        g = self.game
        if g.max_moves <= 0:
            return
        remaining = max(0, g.max_moves - g.moves_used)
        ratio = remaining / g.max_moves
        cells_filled = int(ratio * cam_w)
        bar_row = gy(cam_h - 1)
        for col in range(cam_w):
            frame[bar_row, gx(col)] = 3 if col < cells_filled else 0

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        g = self.game
        gy, gx, cam_w, cam_h = self._grid_slicers(frame)

        if g.death_flash > 0:
            for col in range(cam_w):
                frame[gy(0), gx(col)] = CLR_HAZARD
                frame[gy(cam_h - 1), gx(col)] = CLR_HAZARD
            return frame

        for i in range(3):
            frame[gy(0), gx(1 + i)] = CLR_PLAYER if g.lives > i else 0

        for i in range(MAX_ECHOES):
            clr = ECHO_COLORS[i] if i < len(g.echoes) else 0
            frame[gy(0), gx(5 + i)] = clr

        self._render_move_bar(frame, gy, gx, cam_w, cam_h)
        return frame


class Ea99(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:

        self.lives = 3

        self.death_flash = 0

        self.turn = 0

        self.moves_used = 0

        self.max_moves = 80

        self.pos_history: list = []

        self.move_history: list = []

        self.echoes: list = []

        self.spawn_x = 1

        self.spawn_y = 2

        self.walls: list = []

        self.plates_spr: dict = {}

        self.doors_spr: dict = {}

        self.hazards_pos: set = set()

        self.door_pairs: list = []

        self.echo_pool: list = []

        self._hud = HudDisplay(self)

        self._game_over = False

        self._rng = Random(seed)

        self._consecutive_resets = 0

        self._history: List[Dict] = []

        super().__init__(
            "ea99",
            levels,
            Camera(
                x=0,
                y=0,
                width=GRID_WIDTH,
                height=GRID_HEIGHT,
                background=BACKGROUND_COLOR,
                letter_box=PADDING_COLOR,
                interfaces=[self._hud],
            ),
            available_actions=[0, 1, 2, 3, 4, 5, 7],
        )

    def on_set_level(self, level: Level) -> None:

        self.lives = 3

        self.death_flash = 0

        self.turn = 0

        self.moves_used = 0

        self.player = self.current_level.get_sprites_by_tag("plr")[0]

        self.spawn_x = self.player.x

        self.spawn_y = self.player.y

        self.pos_history = [(self.spawn_x, self.spawn_y)]

        self.move_history = []

        self.echoes = []

        self.walls = self.current_level.get_sprites_by_tag("wal")

        self.plates_spr = {
            (s.x, s.y): s for s in self.current_level.get_sprites_by_tag("plt")
        }

        self.doors_spr = {
            (s.x, s.y): s for s in self.current_level.get_sprites_by_tag("dor")
        }

        self.hazards_pos = {
            (s.x, s.y) for s in self.current_level.get_sprites_by_tag("haz")
        }

        self.echo_pool = self.current_level.get_sprites_by_tag("echo")

        for spr in self.echo_pool:
            spr.set_visible(False)

            spr.set_position(0, 0)

        raw = self.current_level.get_data("pairs")

        self.door_pairs = raw if raw is not None else []

        for spr in self.doors_spr.values():
            spr.set_visible(True)

        for spr in self.plates_spr.values():
            spr.color_remap(CLR_PLATE_ON, CLR_PLATE_OFF)

        dov_list = self.current_level.get_sprites_by_tag("dov")

        self.dov_spr = dov_list[0] if dov_list else None

        if self.dov_spr:
            self.dov_spr.set_visible(False)

        mv = self.current_level.get_data("max_moves")

        self.max_moves = int(mv) if mv is not None else 999

        self._game_over = False
        self._history = []

        origin_x = self.current_level.get_data("origin_x")
        origin_y = self.current_level.get_data("origin_y")
        if origin_x is not None and origin_y is not None:
            self.player.set_position(origin_x, origin_y)
            self.spawn_x = origin_x
            self.spawn_y = origin_y
            self.pos_history = [(origin_x, origin_y)]
            self._randomize_spawn(origin_x, origin_y)

    def _get_reachable_empty_tiles(
        self, start_x: int, start_y: int
    ) -> List[Tuple[int, int]]:
        blocked: set[Tuple[int, int]] = set()
        for s in self.walls:
            blocked.add((s.x, s.y))
        for s in self.current_level.get_sprites_by_tag("dor"):
            if s.is_visible:
                blocked.add((s.x, s.y))

        hazard: set[Tuple[int, int]] = set()
        for pos in self.hazards_pos:
            hazard.add(pos)

        visited: set[Tuple[int, int]] = {(start_x, start_y)}
        q: deque[Tuple[int, int]] = deque([(start_x, start_y)])
        reachable: List[Tuple[int, int]] = []

        while q:
            cx, cy = q.popleft()
            if (cx, cy) not in hazard:
                reachable.append((cx, cy))
            for ddx, ddy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                nx, ny = cx + ddx, cy + ddy
                if (nx, ny) not in visited and (nx, ny) not in blocked:
                    visited.add((nx, ny))
                    q.append((nx, ny))

        return reachable

    def _randomize_spawn(self, origin_x: int, origin_y: int) -> None:
        reachable = self._get_reachable_empty_tiles(origin_x, origin_y)
        if not reachable:
            return
        if len(reachable) > 1:
            new_spawn = self._rng.choice(reachable)
            self.spawn_x = new_spawn[0]
            self.spawn_y = new_spawn[1]
            self.player.set_position(self.spawn_x, self.spawn_y)

    def _is_wall(self, x: int, y: int) -> bool:

        for s in self.walls:
            if s.x == x and s.y == y:
                return True

        return False

    def _is_door_closed(self, x: int, y: int) -> bool:

        spr = self.doors_spr.get((x, y))

        return spr is not None and spr.is_visible

    def _echo_at(self, x: int, y: int) -> bool:

        for e in self.echoes:
            if e["x"] == x and e["y"] == y:
                return True

        return False

    def _player_blocked(self, x: int, y: int) -> bool:

        return self._is_wall(x, y) or self._is_door_closed(x, y) or self._echo_at(x, y)

    def _echo_move_blocked(self, x: int, y: int) -> bool:

        return self._is_wall(x, y) or self._is_door_closed(x, y)

    def _occupied(self) -> set:

        occ = {(self.player.x, self.player.y)}

        for e in self.echoes:
            occ.add((e["x"], e["y"]))

        return occ

    def _update_plates_and_doors(self) -> None:

        occ = self._occupied()

        for pos, spr in self.plates_spr.items():
            if pos in occ:
                spr.color_remap(CLR_PLATE_OFF, CLR_PLATE_ON)

            else:
                spr.color_remap(CLR_PLATE_ON, CLR_PLATE_OFF)

        for pair in self.door_pairs:
            door_pos = (pair["door"][0], pair["door"][1])

            if "plates" not in pair:
                continue

            held = all((p[0], p[1]) in occ for p in pair["plates"])

            door_spr = self.doors_spr.get(door_pos)

            if door_spr is not None:
                door_spr.set_visible(not held)

    def _advance_echoes(self) -> None:

        to_destroy = []

        for idx, e in enumerate(self.echoes):
            step = e["step"]

            dx, dy = e["moves"][step]

            nx = e["x"] + dx

            ny = e["y"] + dy

            if not self._echo_move_blocked(nx, ny):
                e["x"] = nx

                e["y"] = ny

            e["step"] = (step + 1) % ECHO_CYCLE

            if e["step"] == 0:
                e["x"] = e["spawn_x"]

                e["y"] = e["spawn_y"]

            e["spr"].set_position(e["x"], e["y"])

            if (e["x"], e["y"]) in self.hazards_pos:
                to_destroy.append(idx)

        for idx in reversed(to_destroy):
            self.echoes[idx]["spr"].set_visible(False)

            self.echoes.pop(idx)

    def _spawn_echo(self) -> None:

        hist_len = len(self.pos_history)

        spawn_idx = hist_len - ECHO_CYCLE - 1

        if spawn_idx < 0:
            spawn_idx = 0

        sx, sy = self.pos_history[spawn_idx]

        echo_moves = list(self.move_history[-ECHO_CYCLE:])

        if len(self.echoes) >= MAX_ECHOES:
            old = self.echoes.pop(0)

            old["spr"].set_visible(False)

        in_use = {e["spr"] for e in self.echoes}

        free_spr = next((s for s in self.echo_pool if s not in in_use), None)

        if free_spr is None:
            return

        free_spr.set_position(sx, sy)

        free_spr.set_visible(True)

        self.echoes.append(
            {
                "spr": free_spr,
                "spawn_x": sx,
                "spawn_y": sy,
                "moves": echo_moves,
                "step": 0,
                "x": sx,
                "y": sy,
            }
        )

    def _die(self) -> None:

        self.lives -= 1

        if self.lives <= 0:
            self.lose()

            self._game_over = True

            return

        self.death_flash = 2

        self.player.set_position(self.spawn_x, self.spawn_y)

        self.turn = 0

        self.moves_used = 0

        self.pos_history = [(self.spawn_x, self.spawn_y)]

        self.move_history = []

        for e in self.echoes:
            e["spr"].set_visible(False)

        self.echoes = []

        for spr in self.doors_spr.values():
            spr.set_visible(True)

        for spr in self.plates_spr.values():
            spr.color_remap(CLR_PLATE_ON, CLR_PLATE_OFF)

    def _save_state(self) -> None:
        echoes_snap = [
            {
                "x": e["x"],
                "y": e["y"],
                "spawn_x": e["spawn_x"],
                "spawn_y": e["spawn_y"],
                "moves": list(e["moves"]),
                "step": e["step"],
                "spr_idx": self.echo_pool.index(e["spr"]),
            }
            for e in self.echoes
        ]
        door_vis = {pos: spr.is_visible for pos, spr in self.doors_spr.items()}
        self._history.append(
            {
                "px": self.player.x,
                "py": self.player.y,
                "turn": self.turn,
                "pos_history": list(self.pos_history),
                "move_history": list(self.move_history),
                "echoes": echoes_snap,
                "door_vis": door_vis,
            }
        )

    def _undo(self) -> None:
        if not self._history:
            return
        snap = self._history.pop()
        self.player.set_position(snap["px"], snap["py"])
        self.turn = snap["turn"]
        self.pos_history = snap["pos_history"]
        self.move_history = snap["move_history"]
        for e in self.echoes:
            e["spr"].set_visible(False)
        self.echoes = []
        for es in snap["echoes"]:
            spr = self.echo_pool[es["spr_idx"]]
            spr.set_position(es["x"], es["y"])
            spr.set_visible(True)
            self.echoes.append(
                {
                    "spr": spr,
                    "x": es["x"],
                    "y": es["y"],
                    "spawn_x": es["spawn_x"],
                    "spawn_y": es["spawn_y"],
                    "moves": es["moves"],
                    "step": es["step"],
                }
            )
        for pos, spr in self.doors_spr.items():
            spr.set_visible(snap["door_vis"].get(pos, True))
        self._update_plates_and_doors()

    def _parse_action(self) -> Optional[Tuple[int, int]]:
        act = self.action.id.value
        if act == 1:
            return 0, -1
        elif act == 2:
            return 0, 1
        elif act == 3:
            return -1, 0
        elif act == 4:
            return 1, 0
        elif act == 5:
            return 0, 0
        return None

    def _check_collisions(self) -> bool:
        if (self.player.x, self.player.y) in self.hazards_pos:
            self._die()
            return True
        for e in self.echoes:
            if e["x"] == self.player.x and e["y"] == self.player.y:
                self._die()
                return True
        for ext_spr in self.current_level.get_sprites_by_tag("ext"):
            if ext_spr.x == self.player.x and ext_spr.y == self.player.y:
                self.next_level()
                return True
        return False

    def step(self) -> None:
        if self.action.id == GameAction.RESET:
            self.complete_action()
            return

        if self.death_flash > 0:
            self.death_flash -= 1
            self.complete_action()
            return

        if self.action.id == GameAction.ACTION7:
            self._consecutive_resets = 0
            self._undo()
            self.moves_used += 1
            self.complete_action()
            return

        direction = self._parse_action()
        if direction is None:
            self.complete_action()
            return

        self._consecutive_resets = 0
        self._save_state()

        dx, dy = direction
        self._advance_echoes()
        self._update_plates_and_doors()

        if dx != 0 or dy != 0:
            nx, ny = self.player.x + dx, self.player.y + dy
            if not self._player_blocked(nx, ny):
                self.player.set_position(nx, ny)

        self.move_history.append((dx, dy))
        self.pos_history.append((self.player.x, self.player.y))
        self.turn += 1
        self.moves_used += 1
        self._update_plates_and_doors()

        if self._check_collisions():
            self.complete_action()
            return

        if self.turn % ECHO_CYCLE == 0:
            self._spawn_echo()

        if self.max_moves > 0 and self.moves_used >= self.max_moves:
            self._die()
            self.complete_action()
            return

        self.complete_action()


class PuzzleEnvironment:
    _ACTION_MAP: Dict[str, GameAction] = {
        "reset": GameAction.RESET,
        "up": GameAction.ACTION1,
        "down": GameAction.ACTION2,
        "left": GameAction.ACTION3,
        "right": GameAction.ACTION4,
        "select": GameAction.ACTION5,
        "undo": GameAction.ACTION7,
    }

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

    _TAG_TO_CHAR: Dict[str, str] = {
        "wal": "#",
        "plr": "@",
        "plt": "O",
        "dor": "D",
        "ext": "X",
        "haz": "H",
        "echo": "E",
        "dov": "",
    }

    _TAG_PRIORITY: Dict[str, int] = {
        "dov": -1,
        "wal": 0,
        "dor": 1,
        "plt": 2,
        "haz": 2,
        "ext": 3,
        "echo": 4,
        "plr": 5,
    }

    def __init__(self, seed: int = 0) -> None:
        self._engine = Ea99(seed=seed)
        self._total_turns = 0
        self._last_action_was_reset = False
        self._game_won = False

    def _build_text_obs(self) -> str:
        e = self._engine
        gs = e.current_level.grid_size
        w, h = (gs[0] if gs else 16), (gs[1] if gs else 14)
        text_grid: List[List[str]] = [["." for _ in range(w)] for _ in range(h)]
        prio_grid: List[List[int]] = [[-2 for _ in range(w)] for _ in range(h)]
        tag_char = self._TAG_TO_CHAR
        tag_prio = self._TAG_PRIORITY
        for sprite in e.current_level.get_sprites():
            if not sprite.is_visible:
                continue
            sx, sy = sprite.x, sprite.y
            if 0 <= sx < w and 0 <= sy < h:
                for tag in sprite.tags or []:
                    ch = tag_char.get(tag, "")
                    pr = tag_prio.get(tag, -1)
                    if ch and pr > prio_grid[sy][sx]:
                        text_grid[sy][sx] = ch
                        prio_grid[sy][sx] = pr
        grid_text = "\n".join("".join(row) for row in text_grid)

        remaining = max(0, e.max_moves - e.moves_used)
        header = (
            f"Level:{e.level_index + 1} Lives:{e.lives} "
            f"Moves:{remaining}/{e.max_moves} Echoes:{len(e.echoes)}"
        )
        return header.strip() + "\n" + grid_text

    @staticmethod
    def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
        chunk = chunk_type + data
        crc = struct.pack(">I", zlib.crc32(chunk) & 0xFFFFFFFF)
        return struct.pack(">I", len(data)) + chunk + crc

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
        raw = b""
        for row in range(h):
            raw += b"\x00" + rgb[row].tobytes()
        ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
        png = b"\x89PNG\r\n\x1a\n"
        png += self._png_chunk(b"IHDR", ihdr)
        png += self._png_chunk(b"IDAT", zlib.compress(raw))
        png += self._png_chunk(b"IEND", b"")
        return png

    def _build_game_state(
        self, done: bool = False, info: Optional[Dict] = None
    ) -> GameState:
        e = self._engine
        valid_actions = self.get_actions() if not done else None
        return GameState(
            text_observation=self._build_text_obs(),
            image_observation=self._build_image_bytes(),
            valid_actions=valid_actions,
            turn=self._total_turns,
            metadata={
                "total_levels": len(self._engine._levels),
                "level_index": e.level_index,
                "lives": e.lives,
                "moves_used": e.moves_used,
                "max_moves": e.max_moves,
                "echoes": len(e.echoes),
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

    def is_done(self) -> bool:
        return self._engine._game_over or self._game_won

    def get_actions(self) -> List[str]:
        if self._engine._game_over:
            return ["reset"]
        return ["up", "down", "left", "right", "select", "undo", "reset"]

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
        info: Dict = {"action": action}

        prev_level = e.level_index
        action_input = ActionInput(id=game_action)
        frame = e.perform_action(action_input, raw=True)

        game_won = frame and frame.state and frame.state.name == "WIN"
        total_levels = len(levels)
        level_completed = game_won or (e.level_index > prev_level)
        reward = (1.0 / total_levels) if level_completed else 0.0

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
                state=self._build_game_state(done=True, info=info),
                reward=0.0,
                done=True,
                info=info,
            )

        return StepResult(
            state=self._build_game_state(done=False, info=info),
            reward=reward,
            done=False,
            info=info,
        )

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        e = self._engine
        index_grid = e.camera.render(e.current_level.get_sprites())
        arr = (
            np.array(index_grid, dtype=np.uint8)
            if not isinstance(index_grid, np.ndarray)
            else index_grid.astype(np.uint8)
        )
        h, w = arr.shape[:2]
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for idx, color in enumerate(self.ARC_PALETTE):
            mask = arr == idx
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

    @staticmethod
    def _resize_nearest(frame: np.ndarray, h: int, w: int) -> np.ndarray:
        src_h, src_w = frame.shape[0], frame.shape[1]
        row_idx = (np.arange(h) * src_h // h).astype(int)
        col_idx = (np.arange(w) * src_w // w).astype(int)
        return frame[np.ix_(row_idx, col_idx)]

    def _get_obs(self) -> np.ndarray:
        frame = self._env.render(mode="rgb_array")
        if frame.shape[0] != self.OBS_HEIGHT or frame.shape[1] != self.OBS_WIDTH:
            frame = self._resize_nearest(frame, self.OBS_HEIGHT, self.OBS_WIDTH)
        return frame

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
