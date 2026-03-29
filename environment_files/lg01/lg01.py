import struct
import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from arcengine import (
    ActionInput,
    ARCBaseGame,
    Camera,
    GameAction,
    Level,
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


C_BACKGROUND = 5
C_WALL = 4
C_PLAYER = 11
C_DOOR_CLOSED = 8
C_DOOR_OPEN = 3
C_BUTTON = 12
C_BTN_PRESSED = 9
C_EXIT = 14
C_BAR_HIGH = 14
C_BAR_MID = 11
C_BAR_LOW = 12
C_BAR_CRIT = 8
C_BAR_EMPTY = 5
C_LIFE_ON = 8
C_LIFE_OFF = 4


def _tile(col: int) -> List[List[int]]:
    return [[col] * 4 for _ in range(4)]


sprites = {
    "player": Sprite(
        pixels=_tile(C_PLAYER),
        name="player",
        visible=True,
        collidable=False,
        layer=3,
        tags=["player"],
    ),
    "wall": Sprite(
        pixels=_tile(C_WALL),
        name="wall",
        visible=True,
        collidable=False,
        layer=1,
        tags=["wall"],
    ),
    "door_closed": Sprite(
        pixels=_tile(C_DOOR_CLOSED),
        name="door_closed",
        visible=True,
        collidable=False,
        layer=1,
        tags=["door_closed"],
    ),
    "door_open": Sprite(
        pixels=_tile(C_DOOR_OPEN),
        name="door_open",
        visible=True,
        collidable=False,
        layer=0,
        tags=["door_open"],
    ),
    "button_unpressed": Sprite(
        pixels=_tile(C_BUTTON),
        name="button_unpressed",
        visible=True,
        collidable=False,
        layer=0,
        tags=["button"],
    ),
    "button_pressed": Sprite(
        pixels=_tile(C_BTN_PRESSED),
        name="button_pressed",
        visible=True,
        collidable=False,
        layer=0,
        tags=["button"],
    ),
    "exit": Sprite(
        pixels=_tile(C_EXIT),
        name="exit",
        visible=True,
        collidable=False,
        layer=0,
        tags=["exit"],
    ),
}

LEVEL_1_GRID = (
    "WWWWWWWW\n"
    "W......W\n"
    "W.P....W\n"
    "W.1.2.3W\n"
    "WW.W...W\n"
    "W......W\n"
    "W..WWWWW\n"
    "W..ABCXW\n"
    "WWWWWWWW"
)

LEVEL_2_GRID = (
    "WWWWWWWWWW\n"
    "W........W\n"
    "W.P......W\n"
    "W.1.2.3.4W\n"
    "WW.W.....W\n"
    "W........W\n"
    "W........W\n"
    "W...WWWWWW\n"
    "W...ABCDXW\n"
    "WWWWWWWWWW"
)

LEVEL_3_GRID = (
    "WWWWWWWWWWWW\n"
    "W..........W\n"
    "W..P.......W\n"
    "W.1.2.3.4.5W\n"
    "WW.W.......W\n"
    "W..........W\n"
    "W..........W\n"
    "W..........W\n"
    "W.....WWWWWW\n"
    "W.....ABCDEX\n"
    "WWWWWWWWWWWW"
)

LEVEL_4_GRID = (
    "WWWWWWWWWWWWWW\n"
    "W............W\n"
    "W...P........W\n"
    "W.1.2.3.4.5.6W\n"
    "WW.W.........W\n"
    "W............W\n"
    "W............W\n"
    "W............W\n"
    "W............W\n"
    "W....WWWWWWWWW\n"
    "W....ABCDEFXWW\n"
    "WWWWWWWWWWWWWW"
)

LEVEL_5_GRID = (
    "WWWWWWWWWWWWWWWW\n"
    "W..............W\n"
    "W.....P........W\n"
    "W.1.2.3.4.5.6.7W\n"
    "WW.W...........W\n"
    "W..............W\n"
    "W..............W\n"
    "W..............W\n"
    "W..............W\n"
    "W..............W\n"
    "W......WWWWWWWWW\n"
    "W......ABCDEFGXW\n"
    "WWWWWWWWWWWWWWWW"
)


def _parse_grid(grid_str: str) -> List[List[str]]:
    rows = grid_str.strip().split("\n")
    return [list(row) for row in rows]


def _build_level(
    grid_str: str,
    viewport_size: Tuple[int, int],
    button_logic: Dict[str, List[Tuple[str, str]]],
    level_name: str,
    baseline: int,
) -> Level:
    grid = _parse_grid(grid_str)
    grid_h = len(grid)
    grid_w = len(grid[0]) if grid_h > 0 else 0

    vp_w, vp_h = viewport_size
    cell_size = 4
    grid_area_h = vp_h - 4
    offset_x = (vp_w - grid_w * cell_size) // 2
    offset_y = (grid_area_h - grid_h * cell_size) // 2

    level_sprites: List[Sprite] = []
    player_pos: Optional[List[int]] = None
    exit_pos: Optional[List[int]] = None
    doors: Dict[str, dict] = {}
    buttons: Dict[str, dict] = {}
    wall_positions: List[List[int]] = []

    for gy in range(grid_h):
        for gx in range(grid_w):
            if gx >= len(grid[gy]):
                continue
            ch = grid[gy][gx]
            px = offset_x + gx * cell_size
            py = offset_y + gy * cell_size

            if ch == "W":
                wall_positions.append([px, py])
                level_sprites.append(sprites["wall"].clone().set_position(px, py))
            elif ch == "P":
                player_pos = [px, py]
            elif ch in "ABCDEFG":
                door_id = ch
                doors[door_id] = {"pos": [px, py], "is_open": False}
                level_sprites.append(
                    sprites["door_closed"].clone().set_position(px, py)
                )
            elif ch in "1234567":
                btn_id = ch
                logic = button_logic.get(btn_id, [])
                buttons[btn_id] = {
                    "pos": [px, py],
                    "logic": logic,
                    "is_pressed_visual": False,
                }
                level_sprites.append(
                    sprites["button_unpressed"].clone().set_position(px, py)
                )
            elif ch == "X":
                exit_pos = [px, py]
                level_sprites.append(sprites["exit"].clone().set_position(px, py))

    if player_pos is not None:
        level_sprites.append(
            sprites["player"].clone().set_position(player_pos[0], player_pos[1])
        )

    return Level(
        sprites=level_sprites,
        grid_size=viewport_size,
        data={
            "player_pos": player_pos if player_pos else [0, 0],
            "exit_pos": exit_pos if exit_pos else [0, 0],
            "doors": doors,
            "buttons": buttons,
            "wall_positions": wall_positions,
            "offset_x": offset_x,
            "offset_y": offset_y,
            "grid_w": grid_w,
            "grid_h": grid_h,
            "cell_size": cell_size,
            "baseline": baseline,
        },
        name=level_name,
    )


BUTTON_LOGIC_L1: Dict[str, List[Tuple[str, str]]] = {
    "1": [("A", "toggle"), ("B", "toggle")],
    "2": [("A", "toggle"), ("C", "toggle")],
    "3": [("C", "toggle")],
}

BUTTON_LOGIC_L2: Dict[str, List[Tuple[str, str]]] = {
    "1": [("A", "toggle"), ("B", "toggle")],
    "2": [("B", "toggle"), ("C", "toggle")],
    "3": [("A", "toggle"), ("D", "toggle")],
    "4": [("C", "toggle"), ("D", "toggle")],
}

BUTTON_LOGIC_L3: Dict[str, List[Tuple[str, str]]] = {
    "1": [("B", "toggle"), ("E", "toggle"), ("C", "toggle")],
    "2": [("E", "toggle"), ("A", "toggle"), ("D", "toggle")],
    "3": [("C", "toggle"), ("A", "toggle")],
    "4": [("B", "toggle"), ("D", "toggle")],
    "5": [("A", "toggle"), ("D", "toggle")],
}

BUTTON_LOGIC_L4: Dict[str, List[Tuple[str, str]]] = {
    "1": [("A", "toggle"), ("D", "toggle")],
    "2": [("A", "toggle"), ("C", "toggle"), ("E", "toggle")],
    "3": [("A", "toggle"), ("E", "toggle")],
    "4": [("F", "toggle"), ("B", "toggle")],
    "5": [("C", "toggle"), ("F", "toggle"), ("D", "toggle")],
    "6": [("B", "toggle"), ("F", "toggle"), ("D", "toggle")],
}

BUTTON_LOGIC_L5: Dict[str, List[Tuple[str, str]]] = {
    "1": [("A", "toggle"), ("D", "toggle")],
    "2": [("B", "toggle"), ("D", "toggle")],
    "3": [("B", "toggle"), ("A", "toggle"), ("G", "toggle")],
    "4": [("A", "toggle"), ("E", "toggle"), ("G", "toggle")],
    "5": [("G", "toggle"), ("F", "toggle"), ("A", "toggle")],
    "6": [("C", "toggle"), ("D", "toggle"), ("E", "toggle")],
    "7": [("C", "toggle"), ("F", "toggle")],
}

levels = [
    _build_level(LEVEL_1_GRID, (64, 40), BUTTON_LOGIC_L1, "Level 1 — 3 Doors", 8),
    _build_level(LEVEL_2_GRID, (64, 44), BUTTON_LOGIC_L2, "Level 2 — 4 Doors", 10),
    _build_level(LEVEL_3_GRID, (64, 48), BUTTON_LOGIC_L3, "Level 3 — 5 Doors", 14),
    _build_level(LEVEL_4_GRID, (64, 52), BUTTON_LOGIC_L4, "Level 4 — 6 Doors", 18),
    _build_level(LEVEL_5_GRID, (64, 56), BUTTON_LOGIC_L5, "Level 5 — 7 Doors", 22),
]

ACTION_LIMITS: Dict[int, int] = {
    0: 80,
    1: 112,
    2: 184,
    3: 180,
    4: 192,
}

LIVES_PER_LEVEL: List[int] = [5, 5, 4, 3, 3]

camera = Camera(
    x=0,
    y=0,
    width=64,
    height=56,
    background=C_BACKGROUND,
    letter_box=C_BACKGROUND,
)


class Lg01(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self.player_x: int = 0
        self.player_y: int = 0
        self.player_sprite: Optional[Sprite] = None
        self.doors: Dict[str, dict] = {}
        self.buttons: Dict[str, dict] = {}
        self.exit_pos: Tuple[int, int] = (0, 0)
        self.wall_cells: Set[Tuple[int, int]] = set()
        self.door_sprites: Dict[str, Sprite] = {}
        self.button_sprites: Dict[str, Sprite] = {}
        self.offset_x: int = 0
        self.offset_y: int = 0
        self.grid_w: int = 8
        self.grid_h: int = 8
        self.cell_size: int = 4
        self.actions_used: int = 0
        self.action_limit: int = 80
        self.bar_sprites: List[Sprite] = []
        self.life_sprites: List[Sprite] = []
        self._lives: int = 5
        self._lives_max: int = 5
        self._undo_stack: List[dict] = []

        super().__init__(
            game_id="lg01",
            levels=levels,
            camera=camera,
            available_actions=[0, 1, 2, 3, 4, 5, 7],
        )

    def on_set_level(self, level: Level) -> None:
        data_player = self.current_level.get_data("player_pos")
        self.player_x = data_player[0]
        self.player_y = data_player[1]

        data_exit = self.current_level.get_data("exit_pos")
        self.exit_pos = (data_exit[0], data_exit[1])

        self.offset_x = self.current_level.get_data("offset_x")
        self.offset_y = self.current_level.get_data("offset_y")
        self.grid_w = self.current_level.get_data("grid_w")
        self.grid_h = self.current_level.get_data("grid_h")
        self.cell_size = self.current_level.get_data("cell_size")

        raw_doors = self.current_level.get_data("doors")
        self.doors = {}
        for door_id, info in raw_doors.items():
            self.doors[door_id] = {
                "pos": list(info["pos"]),
                "is_open": False,
            }

        raw_buttons = self.current_level.get_data("buttons")
        self.buttons = {}
        for btn_id, info in raw_buttons.items():
            self.buttons[btn_id] = {
                "pos": list(info["pos"]),
                "logic": list(info["logic"]),
                "is_pressed_visual": False,
            }

        raw_walls = self.current_level.get_data("wall_positions")
        self.wall_cells = set()
        for w in raw_walls:
            self.wall_cells.add((w[0], w[1]))

        gs = self.current_level.grid_size
        if gs:
            self.camera.width = gs[0]
            self.camera.height = gs[1]

        self.door_sprites = {}
        closed_sprites = self.current_level.get_sprites_by_tag("door_closed")
        for spr in closed_sprites:
            for door_id, info in self.doors.items():
                if spr.x == info["pos"][0] and spr.y == info["pos"][1]:
                    self.door_sprites[door_id] = spr
                    break

        self.button_sprites = {}
        btn_sprites = self.current_level.get_sprites_by_tag("button")
        for spr in btn_sprites:
            for btn_id, info in self.buttons.items():
                if spr.x == info["pos"][0] and spr.y == info["pos"][1]:
                    self.button_sprites[btn_id] = spr
                    break

        player_sprites = self.current_level.get_sprites_by_tag("player")
        if player_sprites:
            self.player_sprite = player_sprites[0]
        else:
            self.player_sprite = None

        self._rebuild_wall_cells()

        self.action_limit = ACTION_LIMITS.get(self._current_level_index, 80)
        self.actions_used = 0
        self._undo_stack = []

        idx = self._current_level_index
        self._lives_max = LIVES_PER_LEVEL[idx] if idx < len(LIVES_PER_LEVEL) else 3
        self._lives = self._lives_max

        self._draw_progress_bar()

    def step(self) -> None:
        action_id = self.action.id

        if action_id == GameAction.RESET:
            self.complete_action()
            return

        if action_id == GameAction.ACTION7:
            self._restore_from_undo()
            self.actions_used += 1
            self._draw_progress_bar()
            if self._bar_is_empty():
                self._lose_life()
            self.complete_action()
            return

        self.actions_used += 1

        self._draw_progress_bar()
        if self._bar_is_empty():
            if self._lose_life():
                self.complete_action()
                return
            self.complete_action()
            return

        if action_id == GameAction.ACTION1:
            self._save_state()
            self._try_move(0, -self.cell_size)
        elif action_id == GameAction.ACTION2:
            self._save_state()
            self._try_move(0, self.cell_size)
        elif action_id == GameAction.ACTION3:
            self._save_state()
            self._try_move(-self.cell_size, 0)
        elif action_id == GameAction.ACTION4:
            self._save_state()
            self._try_move(self.cell_size, 0)
        elif action_id == GameAction.ACTION5:
            self._save_state()
            self._try_interact()

        if self._check_win():
            self.next_level()

        self.complete_action()

    def _try_move(self, dx: int, dy: int) -> None:
        new_x = self.player_x + dx
        new_y = self.player_y + dy

        min_x = self.offset_x
        min_y = self.offset_y
        max_x = self.offset_x + (self.grid_w - 1) * self.cell_size
        max_y = self.offset_y + (self.grid_h - 1) * self.cell_size

        if new_x < min_x or new_x > max_x:
            return
        if new_y < min_y or new_y > max_y:
            return

        if (new_x, new_y) in self.wall_cells:
            return

        self.player_x = new_x
        self.player_y = new_y
        if self.player_sprite is not None:
            self.player_sprite.set_position(self.player_x, self.player_y)

    def _press_button(self, button_id: str) -> None:
        logic = self.buttons[button_id]["logic"]
        for door_id, action in logic:
            if door_id not in self.doors:
                continue
            if action == "open":
                self.doors[door_id]["is_open"] = True
            elif action == "close":
                self.doors[door_id]["is_open"] = False
            elif action == "toggle":
                self.doors[door_id]["is_open"] = not self.doors[door_id]["is_open"]
            self._update_door_sprite(door_id)
        self._rebuild_wall_cells()

    def _update_door_sprite(self, door_id: str) -> None:
        old_sprite = self.door_sprites.get(door_id)
        if old_sprite is not None:
            self.current_level.remove_sprite(old_sprite)

        pos = self.doors[door_id]["pos"]
        is_open = self.doors[door_id]["is_open"]

        if is_open:
            new_sprite = sprites["door_open"].clone().set_position(pos[0], pos[1])
        else:
            new_sprite = sprites["door_closed"].clone().set_position(pos[0], pos[1])

        self.current_level.add_sprite(new_sprite)
        self.door_sprites[door_id] = new_sprite

    def _try_interact(self) -> None:
        for btn_id, info in self.buttons.items():
            if self.player_x == info["pos"][0] and self.player_y == info["pos"][1]:
                self._press_button(btn_id)
                info["is_pressed_visual"] = not info["is_pressed_visual"]
                self._update_button_sprite(btn_id, pressed=info["is_pressed_visual"])
                return

    def _update_button_sprite(self, btn_id: str, pressed: bool) -> None:
        old_sprite = self.button_sprites.get(btn_id)
        if old_sprite is not None:
            self.current_level.remove_sprite(old_sprite)

        pos = self.buttons[btn_id]["pos"]
        self.buttons[btn_id]["is_pressed_visual"] = pressed

        if pressed:
            new_sprite = sprites["button_pressed"].clone().set_position(pos[0], pos[1])
        else:
            new_sprite = (
                sprites["button_unpressed"].clone().set_position(pos[0], pos[1])
            )

        self.current_level.add_sprite(new_sprite)
        self.button_sprites[btn_id] = new_sprite

    def _rebuild_wall_cells(self) -> None:
        raw_walls = self.current_level.get_data("wall_positions")
        self.wall_cells = set()
        for w in raw_walls:
            self.wall_cells.add((w[0], w[1]))
        for door_id, info in self.doors.items():
            if not info["is_open"]:
                pos = info["pos"]
                self.wall_cells.add((pos[0], pos[1]))

    def _draw_progress_bar(self) -> None:
        for spr in self.bar_sprites:
            self.current_level.remove_sprite(spr)
        self.bar_sprites = []

        vp_w = self.camera.width
        vp_h = self.camera.height

        bar_y = vp_h - 4
        bar_total = vp_w

        remaining = max(0, self.action_limit - self.actions_used)
        ratio = remaining / max(1, self.action_limit)
        filled = round(bar_total * ratio)
        self._last_bar_filled = filled

        if ratio > 0.50:
            bar_col = C_BAR_HIGH
        elif ratio > 0.25:
            bar_col = C_BAR_MID
        elif ratio > 0.10:
            bar_col = C_BAR_LOW
        else:
            bar_col = C_BAR_CRIT

        for px in range(filled):
            spr = Sprite(
                pixels=[[bar_col], [bar_col], [bar_col], [bar_col]],
                name="bar_fill",
                visible=True,
                collidable=False,
                layer=10,
                tags=["bar"],
            )
            spr.set_position(px, bar_y)
            self.current_level.add_sprite(spr)
            self.bar_sprites.append(spr)

        for px in range(filled, bar_total):
            spr = Sprite(
                pixels=[[C_BAR_EMPTY], [C_BAR_EMPTY], [C_BAR_EMPTY], [C_BAR_EMPTY]],
                name="bar_empty",
                visible=True,
                collidable=False,
                layer=10,
                tags=["bar"],
            )
            spr.set_position(px, bar_y)
            self.current_level.add_sprite(spr)
            self.bar_sprites.append(spr)

        self._draw_lives()

    def _bar_is_empty(self) -> bool:
        return getattr(self, "_last_bar_filled", 1) == 0

    def _draw_lives(self) -> None:
        for spr in self.life_sprites:
            self.current_level.remove_sprite(spr)
        self.life_sprites = []

        vp_w = self.camera.width
        vp_h = self.camera.height
        bar_y = vp_h - 4

        lost = self._lives_max - self._lives
        for i in range(self._lives_max):
            col = C_LIFE_OFF if i < lost else C_LIFE_ON
            x = vp_w - (self._lives_max - i) * 3 + 1
            spr = Sprite(
                pixels=[[col, col], [col, col]],
                name="life",
                visible=True,
                collidable=False,
                layer=11,
                tags=["life"],
            )
            spr.set_position(x, bar_y + 1)
            self.current_level.add_sprite(spr)
            self.life_sprites.append(spr)

    def _reset_level(self) -> None:
        data_player = self.current_level.get_data("player_pos")
        self.player_x = data_player[0]
        self.player_y = data_player[1]
        if self.player_sprite is not None:
            self.player_sprite.set_position(self.player_x, self.player_y)

        for door_id in self.doors:
            self.doors[door_id]["is_open"] = False
            self._update_door_sprite(door_id)

        for btn_id in self.buttons:
            self.buttons[btn_id]["is_pressed_visual"] = False
            self._update_button_sprite(btn_id, pressed=False)

        self._rebuild_wall_cells()

        self.actions_used = 0
        self._undo_stack = []
        self._draw_progress_bar()

    def _lose_life(self) -> bool:
        self._lives -= 1
        if self._lives <= 0:
            self.lose()
            return True
        self._reset_level()
        return False

    def _check_win(self) -> bool:
        all_open = all(d["is_open"] for d in self.doors.values())
        on_exit = (self.player_x, self.player_y) == self.exit_pos
        return all_open and on_exit

    def _save_state(self) -> None:
        self._undo_stack.append({
            "player_x": self.player_x,
            "player_y": self.player_y,
            "doors": {
                d_id: {"is_open": info["is_open"]}
                for d_id, info in self.doors.items()
            },
            "buttons": {
                b_id: {"is_pressed_visual": info["is_pressed_visual"]}
                for b_id, info in self.buttons.items()
            },
        })

    def _restore_from_undo(self) -> None:
        if not self._undo_stack:
            return
        state = self._undo_stack.pop()
        self.player_x = state["player_x"]
        self.player_y = state["player_y"]
        if self.player_sprite is not None:
            self.player_sprite.set_position(self.player_x, self.player_y)
        for d_id, d_state in state["doors"].items():
            if d_id in self.doors:
                self.doors[d_id]["is_open"] = d_state["is_open"]
                self._update_door_sprite(d_id)
        for b_id, b_state in state["buttons"].items():
            if b_id in self.buttons:
                self.buttons[b_id]["is_pressed_visual"] = b_state["is_pressed_visual"]
                self._update_button_sprite(b_id, pressed=b_state["is_pressed_visual"])
        self._rebuild_wall_cells()
        self._draw_progress_bar()


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

    ACTION_MAP: Dict[str, int] = {
        "up": 1,
        "down": 2,
        "left": 3,
        "right": 4,
        "select": 5,
        "undo": 7,
        "reset": 0,
    }

    def __init__(self, seed: int = 0) -> None:
        self._engine = Lg01(seed=seed)
        self._total_turns = 0
        self._last_action_was_reset = False
        self._prev_score = 0

    def _build_text_obs(self) -> str:
        e = self._engine
        cell = e.cell_size
        ox = e.offset_x
        oy = e.offset_y
        gw = e.grid_w
        gh = e.grid_h

        grid = [["." for _ in range(gw)] for _ in range(gh)]

        raw_walls = e.current_level.get_data("wall_positions")
        for w in raw_walls:
            gx = (w[0] - ox) // cell
            gy = (w[1] - oy) // cell
            if 0 <= gx < gw and 0 <= gy < gh:
                grid[gy][gx] = "W"

        for btn_id, info in e.buttons.items():
            gx = (info["pos"][0] - ox) // cell
            gy = (info["pos"][1] - oy) // cell
            if 0 <= gx < gw and 0 <= gy < gh:
                grid[gy][gx] = btn_id

        for door_id, info in e.doors.items():
            gx = (info["pos"][0] - ox) // cell
            gy = (info["pos"][1] - oy) // cell
            if 0 <= gx < gw and 0 <= gy < gh:
                grid[gy][gx] = door_id.lower() if info["is_open"] else door_id

        ex, ey = e.exit_pos
        egx = (ex - ox) // cell
        egy = (ey - oy) // cell
        if 0 <= egx < gw and 0 <= egy < gh:
            grid[egy][egx] = "X"

        pgx = (e.player_x - ox) // cell
        pgy = (e.player_y - oy) // cell
        if 0 <= pgx < gw and 0 <= pgy < gh:
            grid[pgy][pgx] = "@"

        grid_text = "\n".join("".join(row) for row in grid)

        remaining = max(0, e.action_limit - e.actions_used)
        level_num = e._current_level_index + 1
        door_states = " ".join(
            f"{d}={'open' if info['is_open'] else 'closed'}"
            for d, info in sorted(e.doors.items())
        )
        header = (
            f"Level:{level_num} Lives:{e._lives}/{e._lives_max} "
            f"Actions:{remaining}/{e.action_limit}"
        )
        return header + "\nDoors: " + door_states + "\n" + grid_text

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
        return GameState(
            text_observation=self._build_text_obs(),
            image_observation=self._build_image_bytes(),
            valid_actions=valid_actions,
            turn=self._total_turns,
            metadata={
                "total_levels": len(self._engine._levels),
                "level_index": e.level_index,
                "lives": e._lives,
                "lives_max": e._lives_max,
                "actions_used": e.actions_used,
                "action_limit": e.action_limit,
                "done": done,
                "info": info or {},
            },
        )

    def reset(self) -> GameState:
        e = self._engine
        self._total_turns = 0
        game_won = getattr(e._state, "name", "") == "WIN"
        if game_won or self._last_action_was_reset:
            e.full_reset()
        else:
            e.level_reset()
        self._last_action_was_reset = True
        self._prev_score = e._score
        return self._build_game_state()

    def get_actions(self) -> List[str]:
        if self.is_done():
            return ["reset"]
        return ["reset", "up", "down", "left", "right", "select", "undo"]

    def is_done(self) -> bool:
        state_name = getattr(self._engine._state, "name", "")
        return state_name in ("WIN", "GAME_OVER")

    def step(self, action: str) -> StepResult:
        e = self._engine

        if action == "reset":
            state = self.reset()
            return StepResult(
                state=state, reward=0.0, done=False, info={"action": "reset"}
            )

        if action not in self.ACTION_MAP:
            raise ValueError(
                f"Invalid action '{action}'. Must be one of {list(self.ACTION_MAP.keys())}"
            )

        self._last_action_was_reset = False
        self._total_turns += 1

        prev_score = self._prev_score

        game_action_id = self.ACTION_MAP[action]
        _action_map = {
            1: GameAction.ACTION1,
            2: GameAction.ACTION2,
            3: GameAction.ACTION3,
            4: GameAction.ACTION4,
            5: GameAction.ACTION5,
            7: GameAction.ACTION7,
        }
        game_action = _action_map[game_action_id]
        info: Dict = {"action": action}

        action_input = ActionInput(id=game_action)
        frame = e.perform_action(action_input, raw=True)

        self._prev_score = frame.levels_completed
        levels_advanced = frame.levels_completed - prev_score

        game_won = frame.state and frame.state.name == "WIN"
        game_over = frame.state and frame.state.name == "GAME_OVER"
        done = bool(game_won or game_over)

        reward = levels_advanced * (1.0 / len(e._levels))

        if game_won:
            info["reason"] = "game_complete"
        elif game_over:
            info["reason"] = "game_over"

        return StepResult(
            state=self._build_game_state(done=done, info=info),
            reward=reward,
            done=done,
            info=info,
        )

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError(f"Unsupported render mode: {mode}")
        e = self._engine
        index_grid = e.camera.render(e.current_level.get_sprites())
        arr = np.array(index_grid, dtype=np.uint8)
        h, w = arr.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for idx, color in enumerate(self.ARC_PALETTE):
            mask = arr == idx
            rgb[mask] = color
        return rgb

    def close(self) -> None:
        self._engine = None


class ArcGameEnv(gym.Env):
    metadata: Dict = {
        "render_modes": ["rgb_array"],
        "render_fps": 5,
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
        self._env: Optional[PuzzleEnvironment] = None

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        if seed is not None:
            self._seed = seed

        self._env = PuzzleEnvironment(seed=self._seed)
        state = self._env.reset()

        return self._get_obs(), self._build_info(state)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
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

    def _build_info(self, state: GameState, step_info: Optional[Dict] = None) -> Dict:
        info: Dict = {
            "text_observation": state.text_observation,
            "valid_actions": state.valid_actions,
            "turn": state.turn,
            "game_metadata": state.metadata,
        }
        if step_info:
            info["step_info"] = step_info
        return info
