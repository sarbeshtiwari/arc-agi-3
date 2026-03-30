from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from arcengine import (
    ActionInput,
    ARCBaseGame,
    Camera,
    GameAction,
    GameState as EngineGameState,
    Level,
    RenderableUserDisplay,
    Sprite,
)

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

C_BG = 0
C_WALL = 10
C_PLAYER = 14
C_BULB_OFF = 3
C_BULB_ON = 11
C_HOME_OFF = 9
C_HOME_ON_L5 = 8
C_PORTAL = 15
C_CURSOR = 6

BACKGROUND_COLOR = 0
PADDING_COLOR = 0

MAX_LIVES = 3
C_LIFE_ON = 8
C_LIFE_OFF = 5

BASELINE_ACTIONS = [25, 45, 70, 100]
STEP_LIMITS = [int(4.0 * b) for b in BASELINE_ACTIONS]

PORTAL_COLORS = {
    "1": 15,
    "2": 15,
    "3": 15,
    "4": 15,
    "5": 15,
    "6": 15,
    "7": 15,
    "8": 15,
    "9": 15,
}

BAYER_8x8 = np.array(
    [
        [0, 32, 8, 40, 2, 34, 10, 42],
        [48, 16, 56, 24, 50, 18, 58, 26],
        [12, 44, 4, 36, 14, 46, 6, 38],
        [60, 28, 52, 20, 62, 30, 54, 22],
        [3, 35, 11, 43, 1, 33, 9, 41],
        [51, 19, 59, 27, 49, 17, 57, 25],
        [15, 47, 7, 39, 13, 45, 5, 37],
        [63, 31, 55, 23, 61, 29, 53, 21],
    ],
    dtype=np.float32,
)

OPACITY_LEVELS = [0.75, 0.80, 0.85, 0.90]

sprites = {
    "player": Sprite(
        pixels=[[C_PLAYER]],
        name="player",
        visible=True,
        collidable=True,
        tags=["player"],
        layer=3,
    ),
    "wall": Sprite(
        pixels=[[C_WALL]],
        name="wall",
        visible=True,
        collidable=True,
        tags=["wall"],
        layer=1,
    ),
    "bulb_off": Sprite(
        pixels=[[C_BULB_OFF]],
        name="bulb_off",
        visible=True,
        collidable=False,
        tags=["bulb"],
        layer=0,
    ),
    "home": Sprite(
        pixels=[[C_HOME_OFF]],
        name="home",
        visible=True,
        collidable=False,
        tags=["home"],
        layer=0,
    ),
    "portal": Sprite(
        pixels=[[C_PORTAL]],
        name="portal",
        visible=True,
        collidable=False,
        tags=["portal"],
        layer=0,
    ),
    "cursor": Sprite(
        pixels=[[C_CURSOR]],
        name="cursor",
        visible=False,
        collidable=False,
        tags=["cursor"],
        layer=4,
    ),
}


def _row(s, w):
    if len(s) != w:
        raise ValueError(f"Row length {len(s)} != {w}: '{s}'")
    return s


def _grid(rows):
    w = len(rows[0])
    for i, r in enumerate(rows):
        if len(r) != w:
            raise ValueError(f"Row {i} length {len(r)} != {w}: '{r}'")
    return "".join(rows)



W1 = 17
_L1 = [
    _row("#################", W1),
    _row("#...........#..B#", W1),
    _row("#.#########.#...#", W1),
    _row("#.#.......#.#1#.#", W1),
    _row("#.#.#####.#.#####", W1),
    _row("#.#B#...#.#.....#", W1),
    _row("#.#.#.#.#.#####.#", W1),
    _row("#.#...#.#.......#", W1),
    _row("#.#####.#.#####.#", W1),
    _row("#.......#.#B..#.#", W1),
    _row("#.#######.#.#.#.#", W1),
    _row("#.#.......#.#.#.#", W1),
    _row("#.#.#######.#.#.#", W1),
    _row("#.#.1.......#...#", W1),
    _row("#.#.#########.#.#", W1),
    _row("#...............#", W1),
    _row("########P########", W1),
]
LEVEL_1 = _grid(_L1)

W2 = 20
_L2 = [
    _row("####################", W2),
    _row("#...............#.B#", W2),
    _row("#.#############.#..#", W2),
    _row("#.#.B.........#.#1.#", W2),
    _row("#.#.#########.#.####", W2),
    _row("#.#.#...2...#.#....#", W2),
    _row("#.#.#.#####.#.####.#", W2),
    _row("#.#.#.#...#.#.#.B#.#", W2),
    _row("#.#.#.#.#.#.#.#..#.#", W2),
    _row("#.#...#.#.#.#.#.##.#", W2),
    _row("#.#####.#.#.#.#..#.#", W2),
    _row("#.#...#.#.#.#.##.#.#", W2),
    _row("#.#.#.###.#.#....#.#", W2),
    _row("#.#.#.....#.#.2#.#.#", W2),
    _row("#.#.#######.####.#.#", W2),
    _row("#.1...........#.B#.#", W2),
    _row("#.#.###########.##.#", W2),
    _row("#..................#", W2),
    _row("##########P#########", W2),
]
LEVEL_2 = _grid(_L2)

W3 = 22
_L3 = [
    _row("######################", W3),
    _row("#.................#.B#", W3),
    _row("#.###############.#..#", W3),
    _row("#.#B............#.#1.#", W3),
    _row("#.#.###########.#.####", W3),
    _row("#.#.#.........#.#....#", W3),
    _row("#.#.#.#######.#.####.#", W3),
    _row("#.#.#.#...#B#.#.....##", W3),
    _row("#.#.#.#.#.#.#.#####.##", W3),
    _row("#.#.1.#.#.#.#.......##", W3),
    _row("#.#.#.#.#.#.#.#####.##", W3),
    _row("#.#.#.#.#.#.#.#..B#.##", W3),
    _row("#.#.#.#.#.#.#.#.###.##", W3),
    _row("#.#.#.#.#...#.#.....##", W3),
    _row("#.#.#.#.#####.#####.##", W3),
    _row("#.#.#.#B..........#.##", W3),
    _row("#.#.#.###########.#.##", W3),
    _row("#.#.2...........#2####", W3),
    _row("#.#############.#B####", W3),
    _row("#B................####", W3),
    _row("##########P###########", W3),
]
LEVEL_3 = _grid(_L3)

W4 = 26
_L4 = [
    _row("##########################", W4),
    _row("#.....................#.B#", W4),
    _row("#.###################.#..#", W4),
    _row("#.#B................#.#1.#", W4),
    _row("#.#.###############.#.####", W4),
    _row("#.#.#.........#B..#.#....#", W4),
    _row("#.#.#.#######.#.#.#.####.#", W4),
    _row("#.#.#.#.....#.#.#B#......#", W4),
    _row("#.#.#.#.###.#.#.#.######.#", W4),
    _row("#.#.#.#.#B#.#.#.#........#", W4),
    _row("#.#.#.#.#.#.#.#.########.#", W4),
    _row("#.#.1.#.#.#.#.#..........#", W4),
    _row("#.#.#.#.#.#.#.#.#######.##", W4),
    _row("#.#.#.#.#B..#.#..........#", W4),
    _row("#.#.#.#.#####.#.#######.##", W4),
    _row("#.#.#.#.......#.#........#", W4),
    _row("#.#.#.#########.#.#####.##", W4),
    _row("#.#.#...........#...B#3.##", W4),
    _row("#.#.###########.#.####.#.#", W4),
    _row("#.#.#...B.....#.3.....#.##", W4),
    _row("#.#.#.#######.########.#.#", W4),
    _row("#.#.2.........#......2####", W4),
    _row("#.###########.#.....#B####", W4),
    _row("#B....................####", W4),
    _row("############P#############", W4),
]
LEVEL_4 = _grid(_L4)

LEVEL_DATA = [
    {"grid": LEVEL_1, "width": 17, "height": 17, "opacity": OPACITY_LEVELS[0]},
    {"grid": LEVEL_2, "width": 20, "height": 19, "opacity": OPACITY_LEVELS[1]},
    {"grid": LEVEL_3, "width": 22, "height": 21, "opacity": OPACITY_LEVELS[2]},
    {"grid": LEVEL_4, "width": 26, "height": 25, "opacity": OPACITY_LEVELS[3]},
]


def _parse_grid(
    grid_str: str, width: int, height: int, level_idx: int, opacity: float
) -> Level:
    level_sprites: List[Sprite] = []
    home_x, home_y = 0, 0
    portal_map: Dict[str, List[Tuple[int, int]]] = {}

    for y in range(height):
        for x in range(width):
            ch = grid_str[y * width + x]
            if ch == "#":
                level_sprites.append(sprites["wall"].clone().set_position(x, y))
            elif ch == "B":
                level_sprites.append(sprites["bulb_off"].clone().set_position(x, y))
            elif ch == "P":
                home_x, home_y = x, y
                level_sprites.append(sprites["home"].clone().set_position(x, y))
                level_sprites.append(sprites["player"].clone().set_position(x, y))
            elif ch.isdigit():
                portal_color = PORTAL_COLORS.get(ch, C_PORTAL)
                portal_sprite = sprites["portal"].clone().set_position(x, y)
                portal_sprite.color_remap(None, portal_color)
                level_sprites.append(portal_sprite)

                if ch not in portal_map:
                    portal_map[ch] = []
                portal_map[ch].append((x, y))

    teleport_map: Dict[Tuple[int, int], Tuple[int, int]] = {}
    for digit, positions in portal_map.items():
        if len(positions) == 2:
            teleport_map[positions[0]] = positions[1]
            teleport_map[positions[1]] = positions[0]

    cursor_sprite = sprites["cursor"].clone().set_position(home_x, home_y)
    cursor_sprite.visible = False
    level_sprites.append(cursor_sprite)

    return Level(
        sprites=level_sprites,
        grid_size=(width, height),
        data={
            "level_idx": level_idx,
            "home_x": home_x,
            "home_y": home_y,
            "portal_map": teleport_map,
            "opacity": opacity,
        },
        name=f"Level {level_idx + 1}",
    )


def _build_levels() -> List[Level]:
    result: List[Level] = []
    for i, ld in enumerate(LEVEL_DATA):
        result.append(
            _parse_grid(ld["grid"], ld["width"], ld["height"], i, ld["opacity"])
        )
    return result


_CAMERA_SIZES = [
    (17, 17),
    (20, 19),
    (22, 21),
    (26, 25),
]


class OpacityDisplay(RenderableUserDisplay):
    def __init__(self, game: "Ml01") -> None:
        self._g = game

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        opacity = self._g._opacity
        threshold = (1.0 - opacity) * 64.0

        cam_w = self._g.camera.width
        cam_h = self._g.camera.height

        if cam_w == 0 or cam_h == 0:
            return frame

        for bulb in self._g._bulbs:
            bx, by = bulb.x, bulb.y

            bayer_value = BAYER_8x8[by % 8, bx % 8]
            if bayer_value < threshold:
                x0 = int(bx * 64 / cam_w)
                x1 = int((bx + 1) * 64 / cam_w)
                y0 = int(by * 64 / cam_h)
                y1 = int((by + 1) * 64 / cam_h)

                x1 = min(x1, 64)
                y1 = min(y1, 60)

                if y0 < 60:
                    frame[y0:y1, x0:x1] = BACKGROUND_COLOR

        return frame


class HudDisplay(RenderableUserDisplay):
    HUD_Y = 60

    def __init__(self, game: "Ml01") -> None:
        self._g = game

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        g = self._g

        frame[self.HUD_Y, :] = 5

        for i in range(MAX_LIVES):
            lx = 1 + i * 3
            color = C_LIFE_ON if i < g._lives else C_LIFE_OFF
            frame[self.HUD_Y + 1 : self.HUD_Y + 3, lx : lx + 2] = color

        lit = g._lit_count
        total = g._total_bulbs
        pip_start_x = 12
        max_pips = min(total, 13)
        for i in range(max_pips):
            px = pip_start_x + i * 4
            if px + 2 >= 64:
                break
            color = C_BULB_ON if i < lit else C_BULB_OFF
            frame[self.HUD_Y + 1 : self.HUD_Y + 3, px : px + 2] = color

        bar_start = 1
        bar_max_width = 62
        if g._max_steps > 0:
            remaining = max(0, g._max_steps - g._step_count)
            pct = remaining / g._max_steps
            fill_width = int(pct * bar_max_width)
            if pct > 0.5:
                bar_color = 14
            elif pct > 0.25:
                bar_color = 11
            else:
                bar_color = 8
            if fill_width > 0:
                frame[self.HUD_Y + 3, bar_start : bar_start + fill_width] = bar_color

        return frame


class Ml01(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        levels = _build_levels()
        self._variant_idx: int = 0
        self._opacity_display = OpacityDisplay(self)
        self._hud = HudDisplay(self)

        self._player: Optional[Sprite] = None
        self._cursor: Optional[Sprite] = None
        self._walls: List[Sprite] = []
        self._bulbs: List[Sprite] = []
        self._portals: List[Sprite] = []
        self._home: Optional[Sprite] = None
        self._home_x: int = 0
        self._home_y: int = 0
        self._portal_map: Dict[Tuple[int, int], Tuple[int, int]] = {}
        self._bulb_states: Dict[int, bool] = {}
        self._total_bulbs: int = 0
        self._lit_count: int = 0
        self._opacity: float = 0.75
        self._cursor_active: bool = False

        self._lives: int = MAX_LIVES
        self._has_left_home: bool = False
        self._step_count: int = 0
        self._max_steps: int = STEP_LIMITS[0]
        self._history: List[dict] = []

        camera = Camera(
            0,
            0,
            _CAMERA_SIZES[0][0],
            _CAMERA_SIZES[0][1],
            BACKGROUND_COLOR,
            PADDING_COLOR,
            [self._opacity_display, self._hud],
        )

        super().__init__(
            "ml01",
            levels,
            camera,
            available_actions=[0,1, 2, 3, 4, 5, 7],
        )

    def on_set_level(self, level: Level) -> None:
        self._variant_idx = 0
        idx = self.current_level.get_data("level_idx")
        cw, ch = _CAMERA_SIZES[idx]
        self.camera.width = cw
        self.camera.height = ch

        self._player = self.current_level.get_sprites_by_tag("player")[0]
        self._cursor = self.current_level.get_sprites_by_tag("cursor")[0]
        self._walls = list(self.current_level.get_sprites_by_tag("wall"))
        self._bulbs = list(self.current_level.get_sprites_by_tag("bulb"))
        self._portals = list(self.current_level.get_sprites_by_tag("portal"))
        home_sprites = self.current_level.get_sprites_by_tag("home")
        self._home = home_sprites[0] if home_sprites else None
        self._home_x = self.current_level.get_data("home_x")
        self._home_y = self.current_level.get_data("home_y")
        self._portal_map = self.current_level.get_data("portal_map")
        self._opacity = self.current_level.get_data("opacity")

        self._bulb_states = {}
        for b in self._bulbs:
            self._bulb_states[id(b)] = False
            b.color_remap(None, C_BULB_OFF)

        self._total_bulbs = len(self._bulbs)
        self._lit_count = 0
        self._cursor_active = False
        self._cursor.visible = False
        self._cursor.set_position(self._player.x, self._player.y)

        self._lives = MAX_LIVES
        self._has_left_home = False
        self._step_count = 0
        self._max_steps = STEP_LIMITS[idx]
        self._history = []

        self._refresh_home_color()

    def _is_wall(self, x: int, y: int) -> bool:
        for w in self._walls:
            if w.x == x and w.y == y:
                return True
        return False

    def _bulb_at(self, x: int, y: int) -> Optional[Sprite]:
        for b in self._bulbs:
            if b.x == x and b.y == y:
                return b
        return None

    def _is_portal(self, x: int, y: int) -> bool:
        return (x, y) in self._portal_map

    def _recount_lit(self) -> None:
        self._lit_count = sum(1 for v in self._bulb_states.values() if v)

    def _refresh_home_color(self) -> None:
        if self._home is None:
            return
        if self._lit_count >= self._total_bulbs:
            self._home.color_remap(None, C_HOME_ON_L5)
        else:
            self._home.color_remap(None, C_HOME_OFF)

    def _check_win(self) -> bool:
        if self._lit_count < self._total_bulbs:
            return False
        if self._player is None:
            return False
        return self._player.x == self._home_x and self._player.y == self._home_y

    def _lose_life(self) -> None:
        self._lives -= 1
        if self._lives <= 0:
            self.lose()
        else:
            self._reset_level_state()
        self.complete_action()

    def _reset_level_state(self) -> None:
        if self._player is not None:
            self._player.set_position(self._home_x, self._home_y)
        if self._cursor is not None:
            self._cursor.set_position(self._home_x, self._home_y)
            self._cursor.visible = False
        self._cursor_active = False
        for b in self._bulbs:
            self._bulb_states[id(b)] = False
            b.color_remap(None, C_BULB_OFF)
        self._lit_count = 0
        self._has_left_home = False
        self._step_count = 0
        self._history = []
        self._refresh_home_color()

    def _snapshot(self) -> dict:
        return {
            "player": (self._player.x, self._player.y) if self._player else (0, 0),
            "bulb_states": dict(self._bulb_states),
            "lit_count": self._lit_count,
            "step_count": self._step_count,
            "has_left_home": self._has_left_home,
        }

    def _restore_snapshot(self, snapshot: dict) -> None:
        if self._player is not None:
            self._player.set_position(*snapshot["player"])
        if self._cursor is not None and self._player is not None:
            self._cursor.set_position(self._player.x, self._player.y)
        self._bulb_states = dict(snapshot["bulb_states"])
        self._lit_count = snapshot["lit_count"]
        self._step_count = snapshot["step_count"]
        self._has_left_home = snapshot["has_left_home"]
        for bulb in self._bulbs:
            bulb.color_remap(None, C_BULB_ON if self._bulb_states.get(id(bulb), False) else C_BULB_OFF)
        self._refresh_home_color()

    def _is_on_home(self) -> bool:
        if self._player is None:
            return False
        return self._player.x == self._home_x and self._player.y == self._home_y

    def _try_move(self, dx: int, dy: int) -> bool:
        if self._player is None:
            return False

        px, py = self._player.x, self._player.y
        nx, ny = px + dx, py + dy

        width = self.current_level.grid_size[0]
        height = self.current_level.grid_size[1]
        if nx < 0 or ny < 0 or nx >= width or ny >= height:
            return False

        if self._is_wall(nx, ny):
            return False

        self._player.set_position(nx, ny)

        if self._is_portal(nx, ny):
            dest_x, dest_y = self._portal_map[(nx, ny)]
            self._player.set_position(dest_x, dest_y)

        if self._cursor:
            self._cursor.set_position(self._player.x, self._player.y)

        return True

    def step(self) -> None:
        if self._player is None:
            self.complete_action()
            return

        if self.action.id == GameAction.RESET:
            self.handle_reset()
            self.complete_action()
            return

        if self.action.id == GameAction.ACTION7:
            self._step_count += 1
            if self._step_count >= self._max_steps:
                self._lose_life()
                return
            if self._history:
                self._restore_snapshot(self._history.pop())
            self.complete_action()
            return

        dx, dy = 0, 0
        moved = False

        if self.action.id == GameAction.ACTION1:
            dy = -1
            moved = True
            if not self._cursor_active:
                self._cursor_active = True
                if self._cursor:
                    self._cursor.visible = True

        elif self.action.id == GameAction.ACTION2:
            dy = 1
            moved = True
            if not self._cursor_active:
                self._cursor_active = True
                if self._cursor:
                    self._cursor.visible = True

        elif self.action.id == GameAction.ACTION3:
            dx = -1
            moved = True
            if not self._cursor_active:
                self._cursor_active = True
                if self._cursor:
                    self._cursor.visible = True

        elif self.action.id == GameAction.ACTION4:
            dx = 1
            moved = True
            if not self._cursor_active:
                self._cursor_active = True
                if self._cursor:
                    self._cursor.visible = True

        if moved:
            self._history.append(self._snapshot())
            self._step_count += 1
            if self._step_count >= self._max_steps:
                self._lose_life()
                return

            if self._try_move(dx, dy):
                if not self._has_left_home and not self._is_on_home():
                    self._has_left_home = True

                if self._check_win():
                    self.next_level()
                    self.complete_action()
                    return

                if (
                    self._has_left_home
                    and self._is_on_home()
                    and self._lit_count < self._total_bulbs
                ):
                    self._lose_life()
                    return

            self.complete_action()
            return

        if self.action.id == GameAction.ACTION5:
            self._history.append(self._snapshot())
            self._step_count += 1
            if self._step_count >= self._max_steps:
                self._lose_life()
                return

            px, py = self._player.x, self._player.y
            bulb = self._bulb_at(px, py)
            if bulb is not None:
                bid = id(bulb)
                is_on = self._bulb_states.get(bid, False)
                if is_on:
                    self._bulb_states[bid] = False
                    bulb.color_remap(None, C_BULB_OFF)
                else:
                    self._bulb_states[bid] = True
                    bulb.color_remap(None, C_BULB_ON)

                self._recount_lit()
                self._refresh_home_color()

                if self._check_win():
                    self.next_level()
                    self.complete_action()
                    return

            self.complete_action()
            return

        self.complete_action()


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


ACTION_MAP = {
    "reset": 0,
    "up": 1,
    "down": 2,
    "left": 3,
    "right": 4,
    "select": 5,
    "undo": 7,
}


class PuzzleEnvironment:
    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self._engine = Ml01(seed=seed)
        self._level_index = 0
        self._turn = 0
        self._done = False
        self._consecutive_resets = 0
        self._played_since_reset = False

    def reset(self) -> GameState:
        if self._played_since_reset:
            self._consecutive_resets = 0

        target_level = self._level_index
        if self._consecutive_resets >= 1:
            target_level = 0

        self._engine = Ml01(seed=self._seed)
        self._engine.set_level(target_level)
        self._level_index = target_level
        self._turn = 0
        self._done = False
        self._consecutive_resets += 1
        self._played_since_reset = False
        return self._build_state()

    def get_actions(self) -> List[str]:
        if self._done:
            return ["reset"]
        return ["reset", "up", "down", "left", "right", "select", "undo"]

    def is_done(self) -> bool:
        engine = self._require_engine()
        return self._done or engine._state in (
            EngineGameState.GAME_OVER,
            EngineGameState.WIN,
        )

    def _require_engine(self) -> Ml01:
        if self._engine is None:
            raise RuntimeError("PuzzleEnvironment is closed")
        return self._engine

    def _build_text_observation(self) -> str:
        g = self._require_engine()
        width = g.current_level.grid_size[0]
        height = g.current_level.grid_size[1]
        portal_chars: Dict[Tuple[int, int], str] = {}
        for src, dst in g._portal_map.items():
            if src not in portal_chars:
                portal_chars[src] = str((len(portal_chars) // 2) + 1)
            if dst not in portal_chars:
                portal_chars[dst] = portal_chars[src]
        bulb_positions = {(b.x, b.y): ("*" if g._bulb_states.get(id(b), False) else "B") for b in g._bulbs}
        wall_positions = {(w.x, w.y) for w in g._walls}
        rows = []
        for y in range(height):
            chars = []
            for x in range(width):
                if g._player and (x, y) == (g._player.x, g._player.y):
                    chars.append("P")
                elif (x, y) in wall_positions:
                    chars.append("#")
                elif (x, y) in bulb_positions:
                    chars.append(bulb_positions[(x, y)])
                elif (x, y) in portal_chars:
                    chars.append(portal_chars[(x, y)])
                elif (x, y) == (g._home_x, g._home_y):
                    chars.append("H")
                else:
                    chars.append(".")
            rows.append("".join(chars))
        status = f"level={self._level_index + 1} bulbs={g._lit_count}/{g._total_bulbs} lives={g._lives} steps={max(g._max_steps - g._step_count, 0)}"
        return "\n".join(rows + [status])

    def _build_image_observation(self) -> Optional[bytes]:
        return None

    def _build_state(self) -> GameState:
        return GameState(
            text_observation=self._build_text_observation(),
            image_observation=self._build_image_observation(),
            valid_actions=self.get_actions(),
            turn=self._turn,
            metadata={
                "total_levels": len(self._engine._levels),
                "level": self._level_index + 1,
                "levels_completed": getattr(self._engine, "_score", 0),
                "level_index": self._level_index,
                "game_over": getattr(getattr(self._engine, "_state", None), "name", "") == "GAME_OVER",
            },
        )

    def step(self, action: str) -> StepResult:
        action_lower = action.strip().lower()
        if action_lower not in ACTION_MAP:
            return StepResult(self._build_state(), 0.0, False, {"error": f"invalid action: {action}"})
        if action_lower == "reset":
            state = self.reset()
            return StepResult(state, 0.0, False, {"outcome": "reset"})
        if self.is_done():
            self._done = True
            return StepResult(self._build_state(), 0.0, True, {"outcome": "done"})
        self._consecutive_resets = 0
        self._played_since_reset = True
        self._turn += 1
        engine = self._require_engine()
        prev_level = engine._current_level_index
        prev_lives = engine._lives
        engine.perform_action(ActionInput(id=GameAction.from_id(ACTION_MAP[action_lower])))
        self._level_index = engine._current_level_index
        reward = 0.0
        info: dict = {}
        if engine._state == EngineGameState.GAME_OVER:
            self._done = True
            info["outcome"] = "game_over"
        elif engine._state == EngineGameState.WIN:
            reward = 1.0 / len(engine._levels)
            self._done = True
            info["outcome"] = "game_complete"
        elif engine._current_level_index != prev_level:
            reward = 1.0 / len(engine._levels)
            info["outcome"] = "level_complete"
        elif engine._lives < prev_lives:
            info["outcome"] = "death"
        return StepResult(self._build_state(), reward, self._done, info)

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError(f"Unsupported render mode: {mode}")
        engine = self._require_engine()
        index_grid = engine.camera.render(engine.current_level.get_sprites())
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        for idx, color in enumerate(ARC_PALETTE):
            rgb[index_grid == idx] = color
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

