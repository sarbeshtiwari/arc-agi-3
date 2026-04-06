import io
import struct
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from arcengine import ActionInput, ARCBaseGame, Camera, GameAction, Level, Sprite


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


C_WHITE = 0
C_OFF_WHITE = 1
C_LIGHT_GREY = 2
C_GREY = 3
C_DARK_GREY = 4
C_BLACK = 5
C_MAGENTA = 6
C_PINK = 7
C_RED = 8
C_BLUE = 9
C_LIGHT_BLUE = 10
C_YELLOW = 11
C_ORANGE = 12
C_MAROON = 13
C_GREEN = 14
C_PURPLE = 15


CAMERA_W = 64
CAMERA_H = 64
BORDER = 2
PLAY_W = CAMERA_W - 2 * BORDER
PLAY_H = CAMERA_H - 2 * BORDER
OFFSET = BORDER

OBJ_SIZE = 7
CORE_SIZE = 5
OBJ_HALF = OBJ_SIZE // 2

ROW_SIZE = 4
ROW_SPACING = 9

TRANSPARENT = -1

MAX_LIVES = 3

LIFE_COLOR = 8
LIFE_EMPTY = 4
LIFE_SIZE = 2
LIFE_SPACING = 4

BORDER_TOP_ROWS = list(range(BORDER))
BORDER_BOTTOM_ROWS = list(range(CAMERA_H - BORDER, CAMERA_H))
BORDER_LEFT_COLS = list(range(BORDER))
BORDER_RIGHT_COLS = list(range(CAMERA_W - BORDER, CAMERA_W))

BAR_ROW = CAMERA_H - BORDER
BAR_X0 = BORDER
BAR_WIDTH = PLAY_W


SEPARATOR_GAP = 2


def _compute_layout(nb: int, ns: int) -> Tuple[int, int, int]:
    b_rows = max(1, (nb + ROW_SIZE - 1) // ROW_SIZE)
    s_rows = max(1, (ns + ROW_SIZE - 1) // ROW_SIZE)
    bulb_section_h = b_rows * ROW_SPACING
    sw_section_h = s_rows * ROW_SPACING
    total_needed = bulb_section_h + SEPARATOR_GAP + sw_section_h
    top_margin = max(2, (PLAY_H - total_needed) // 2)
    bulb_row0 = BORDER + top_margin + OBJ_HALF
    separator_y = BORDER + top_margin + bulb_section_h + SEPARATOR_GAP // 2
    sw_row0 = BORDER + top_margin + bulb_section_h + SEPARATOR_GAP + OBJ_HALF
    return bulb_row0, sw_row0, separator_y


def _spacing_for(n: int) -> int:
    if n <= 1:
        return 0
    if n == 2:
        return 24
    if n == 3:
        return 16
    return 14


def _centered_xs(n: int) -> List[int]:
    if n == 1:
        return [PLAY_W // 2]
    sp = _spacing_for(n)
    span = (n - 1) * sp
    x0 = (PLAY_W - span) // 2
    x0 = max(x0, OBJ_HALF + 1)
    return [x0 + i * sp for i in range(n)]


def _row_ys(n: int, row0: int) -> List[int]:
    num_rows = max(1, (n + ROW_SIZE - 1) // ROW_SIZE)
    return [row0 + i * ROW_SPACING for i in range(num_rows)]


def _obj_positions(n: int, row0: int) -> List[Tuple[int, int]]:
    positions = []
    row_ys = _row_ys(n, row0)
    for i in range(n):
        row_idx = i // ROW_SIZE
        col_in_row = i % ROW_SIZE
        objects_in_row = min(ROW_SIZE, n - row_idx * ROW_SIZE)
        xs = _centered_xs(objects_in_row)
        positions.append((xs[col_in_row], row_ys[row_idx]))
    return positions


def _halo7(colour: int) -> List[List[int]]:
    row_solid = [colour] * 7
    row_mid = [colour] + [C_BLACK] * 5 + [colour]
    return [row_solid] + [row_mid] * 5 + [row_solid]


def _cursor7(colour: int) -> List[List[int]]:
    row_solid = [colour] * 7
    row_mid = [colour] + [TRANSPARENT] * 5 + [colour]
    return [row_solid] + [row_mid] * 5 + [row_solid]


def _core5(centre: int, corner: int) -> List[List[int]]:
    c, r = centre, corner
    return [
        [r, r, c, r, r],
        [r, c, c, c, r],
        [c, c, c, c, c],
        [r, c, c, c, r],
        [r, r, c, r, r],
    ]


sprites = {
    "bulb_off": Sprite(
        pixels=_core5(C_DARK_GREY, C_GREY),
        name="bulb_off",
        visible=True,
        collidable=False,
    ),
    "bulb_on": Sprite(
        pixels=_core5(C_YELLOW, C_ORANGE),
        name="bulb_on",
        visible=True,
        collidable=False,
    ),
    "halo_bulb": Sprite(
        pixels=_halo7(C_OFF_WHITE), name="halo_bulb", visible=True, collidable=False
    ),
    "switch_enabled": Sprite(
        pixels=_core5(C_BLUE, C_LIGHT_BLUE),
        name="switch_enabled",
        visible=True,
        collidable=False,
    ),
    "halo_switch": Sprite(
        pixels=_halo7(C_BLUE), name="halo_switch", visible=True, collidable=False
    ),
    "switch_pressed": Sprite(
        pixels=_core5(C_GREY, C_DARK_GREY),
        name="switch_pressed",
        visible=True,
        collidable=False,
    ),
    "halo_pressed": Sprite(
        pixels=_halo7(C_GREY), name="halo_pressed", visible=True, collidable=False
    ),
    "switch_disabled": Sprite(
        pixels=_core5(C_BLUE, C_LIGHT_BLUE),
        name="switch_disabled",
        visible=True,
        collidable=False,
    ),
    "halo_disabled": Sprite(
        pixels=_halo7(C_LIGHT_GREY),
        name="halo_disabled",
        visible=True,
        collidable=False,
    ),
    "player": Sprite(
        pixels=_cursor7(C_PINK), name="player", visible=True, collidable=False
    ),
    "border_normal": Sprite(
        pixels=[[C_GREY]], name="border_normal", visible=True, collidable=True
    ),
    "border_win": Sprite(
        pixels=[[C_GREEN]], name="border_win", visible=True, collidable=True
    ),
    "border_fail": Sprite(
        pixels=[[C_RED]], name="border_fail", visible=True, collidable=True
    ),
    "bar_fill": Sprite(
        pixels=[[C_GREEN]], name="bar_fill", visible=True, collidable=False
    ),
    "bar_used": Sprite(
        pixels=[[C_RED]], name="bar_used", visible=True, collidable=False
    ),
    "separator_dot": Sprite(
        pixels=[[C_DARK_GREY]], name="separator_dot", visible=True, collidable=False
    ),
}


@dataclass
class Bulb:
    id: int
    position: Tuple[int, int]
    state: bool


@dataclass
class Switch:
    id: int
    position: Tuple[int, int]
    enabled: bool
    press_count: int
    is_reset: bool
    bulb_effects: Dict[str, str]
    switch_effects: Dict[str, str]


@dataclass
class Player:
    selected_idx: int
    presses: int


@dataclass
class LevelState:
    bulbs: List[Bulb]
    switches: List[Switch]
    max_presses: int

    initial_bulb_states: Dict[int, bool]
    initial_switch_enabled: Dict[int, bool]

    bulb_row0: int
    sw_row0: int
    separator_y: int


@dataclass(frozen=True)
class UndoState:
    bulb_states: Tuple[bool, ...]
    switch_enabled: Tuple[bool, ...]
    switch_press_counts: Tuple[int, ...]
    cursor_row: int
    cursor_col: int


LEVEL1_CONFIG = {
    "bulbs": [
        {"id": 0, "state": False},
        {"id": 1, "state": False},
        {"id": 2, "state": False},
        {"id": 3, "state": False},
    ],
    "switches": [
        {"id": 0, "bulb_effects": {"0": "toggle", "1": "toggle"}, "switch_effects": {}},
        {"id": 1, "bulb_effects": {"0": "toggle", "2": "toggle"}, "switch_effects": {}},
        {"id": 2, "bulb_effects": {"0": "toggle", "1": "toggle"}, "switch_effects": {}},
        {"id": 3, "reset": True, "bulb_effects": {}, "switch_effects": {}},
        {"id": 4, "bulb_effects": {"2": "toggle"}, "switch_effects": {"5": "toggle"}},
        {
            "id": 5,
            "enabled": False,
            "bulb_effects": {"3": "toggle"},
            "switch_effects": {},
        },
    ],
    "max_presses": 45,
    "cursor_spawns": [(0, 0), (0, 3), (1, 0), (1, 1)],
}

LEVEL2_CONFIG = {
    "bulbs": [
        {"id": 0, "state": False},
        {"id": 1, "state": False},
        {"id": 2, "state": False},
        {"id": 3, "state": False},
        {"id": 4, "state": False},
        {"id": 5, "state": False},
    ],
    "switches": [
        {
            "id": 0,
            "bulb_effects": {"0": "toggle", "2": "toggle", "4": "toggle"},
            "switch_effects": {},
        },
        {
            "id": 1,
            "bulb_effects": {"0": "toggle", "1": "toggle", "2": "toggle"},
            "switch_effects": {},
        },
        {
            "id": 2,
            "bulb_effects": {
                "1": "toggle",
                "2": "toggle",
                "3": "toggle",
                "4": "toggle",
            },
            "switch_effects": {},
        },
        {"id": 3, "bulb_effects": {"0": "toggle", "3": "toggle"}, "switch_effects": {}},
        {"id": 4, "bulb_effects": {}, "switch_effects": {"6": "toggle"}},
        {
            "id": 5,
            "reset": True,
            "bulb_effects": {"0": "toggle", "3": "toggle"},
            "switch_effects": {},
        },
        {
            "id": 6,
            "enabled": False,
            "bulb_effects": {"3": "toggle", "4": "toggle"},
            "switch_effects": {"7": "toggle"},
        },
        {
            "id": 7,
            "enabled": False,
            "bulb_effects": {"5": "toggle"},
            "switch_effects": {},
        },
    ],
    "max_presses": 63,
    "cursor_spawns": [(0, 0), (1, 1), (0, 3), (1, 0)],
}

LEVEL3_CONFIG = {
    "bulbs": [
        {"id": 0, "state": False},
        {"id": 1, "state": False},
        {"id": 2, "state": False},
        {"id": 3, "state": False},
        {"id": 4, "state": False},
        {"id": 5, "state": False},
        {"id": 6, "state": False},
        {"id": 7, "state": False},
    ],
    "switches": [
        {"id": 0, "bulb_effects": {}, "switch_effects": {"7": "toggle"}},
        {
            "id": 1,
            "bulb_effects": {
                "0": "toggle",
                "1": "toggle",
                "2": "toggle",
                "3": "toggle",
            },
            "switch_effects": {},
        },
        {
            "id": 2,
            "bulb_effects": {
                "0": "toggle",
                "1": "toggle",
                "2": "toggle",
                "3": "toggle",
            },
            "switch_effects": {},
        },
        {
            "id": 3,
            "bulb_effects": {
                "3": "toggle",
                "4": "toggle",
                "5": "toggle",
                "6": "toggle",
            },
            "switch_effects": {},
        },
        {"id": 4, "bulb_effects": {"4": "toggle", "5": "toggle"}, "switch_effects": {}},
        {"id": 5, "bulb_effects": {"0": "toggle", "4": "toggle"}, "switch_effects": {}},
        {
            "id": 6,
            "reset": True,
            "bulb_effects": {"0": "toggle", "4": "toggle"},
            "switch_effects": {},
        },
        {
            "id": 7,
            "enabled": False,
            "bulb_effects": {"6": "toggle"},
            "switch_effects": {"8": "toggle"},
        },
        {
            "id": 8,
            "enabled": False,
            "bulb_effects": {"7": "toggle"},
            "switch_effects": {},
        },
    ],
    "max_presses": 72,
    "cursor_spawns": [(0, 2), (1, 3), (0, 0), (1, 1)],
}

LEVEL4_CONFIG = {
    "bulbs": [
        {"id": 0, "state": False},
        {"id": 1, "state": False},
        {"id": 2, "state": False},
        {"id": 3, "state": False},
        {"id": 4, "state": False},
        {"id": 5, "state": False},
        {"id": 6, "state": False},
        {"id": 7, "state": False},
        {"id": 8, "state": False},
    ],
    "switches": [
        {
            "id": 0,
            "bulb_effects": {"0": "toggle", "1": "toggle", "4": "toggle"},
            "switch_effects": {},
        },
        {
            "id": 1,
            "bulb_effects": {"1": "toggle", "2": "toggle", "5": "toggle"},
            "switch_effects": {},
        },
        {
            "id": 2,
            "bulb_effects": {"2": "toggle", "3": "toggle", "6": "toggle"},
            "switch_effects": {},
        },
        {
            "id": 3,
            "bulb_effects": {"3": "toggle", "7": "toggle"},
            "switch_effects": {"5": "toggle"},
        },
        {
            "id": 4,
            "bulb_effects": {},
            "switch_effects": {"6": "toggle", "7": "toggle", "8": "toggle"},
        },
        {
            "id": 5,
            "enabled": False,
            "bulb_effects": {"0": "toggle", "4": "toggle", "5": "toggle"},
            "switch_effects": {},
        },
        {
            "id": 6,
            "enabled": False,
            "bulb_effects": {"1": "toggle", "6": "toggle"},
            "switch_effects": {},
        },
        {
            "id": 7,
            "reset": True,
            "bulb_effects": {"2": "toggle", "7": "toggle"},
            "switch_effects": {},
        },
        {
            "id": 8,
            "enabled": False,
            "bulb_effects": {"8": "toggle"},
            "switch_effects": {},
        },
    ],
    "max_presses": 81,
    "cursor_spawns": [(0, 1), (2, 0), (0, 3), (1, 2)],
}

levels = [
    Level(sprites=[], grid_size=(CAMERA_W, CAMERA_H), data=LEVEL1_CONFIG),
    Level(sprites=[], grid_size=(CAMERA_W, CAMERA_H), data=LEVEL2_CONFIG),
    Level(sprites=[], grid_size=(CAMERA_W, CAMERA_H), data=LEVEL3_CONFIG),
    Level(sprites=[], grid_size=(CAMERA_W, CAMERA_H), data=LEVEL4_CONFIG),
]


class Cl01(ARCBaseGame):
    _player: Player
    _level: Optional[LevelState]
    _game_over: bool
    _cursor_row: int
    _cursor_col: int
    _lives: int
    _transition_pending: str

    def __init__(self, seed: int = 0) -> None:
        self._player = Player(selected_idx=0, presses=0)
        self._level = None
        self._game_over = False
        self._cursor_row = 0
        self._cursor_col = 0
        self._lives = MAX_LIVES
        self._transition_pending = ""
        self._undo_stack: List[UndoState] = []
        self._variation: int = 0
        camera = Camera(
            background=C_BLACK,
            letter_box=C_BLACK,
            width=CAMERA_W,
            height=CAMERA_H,
        )
        super().__init__(
            game_id="cl01",
            levels=levels,
            camera=camera,
            available_actions=[0, 1, 2, 3, 4, 5, 7],
            seed=seed,
        )

    def get_actions(self) -> list[int]:
        return self._available_actions

    @property
    def _current_level(self) -> LevelState:
        if self._level is None:
            raise RuntimeError(
                "Level not initialised — on_set_level has not been called"
            )
        return self._level

    def on_set_level(self, level: Level) -> None:
        self._game_over = False
        self._transition_pending = ""
        self._level = self._build_level(level)
        self._player = Player(selected_idx=0, presses=0)
        spawn_positions = level.get_data("cursor_spawns") or [(0, 0)]
        idx = self._variation % len(spawn_positions)
        start = spawn_positions[idx]
        self._cursor_row = start[0]
        self._cursor_col = start[1]
        self._variation += 1
        self._undo_stack = []
        self._render_all()

    def _build_level(self, level: Level) -> LevelState:
        bulbs_cfg = level.get_data("bulbs") or []
        switches_cfg = level.get_data("switches") or []
        max_presses_raw = level.get_data("max_presses")
        max_presses_cfg = max_presses_raw if max_presses_raw is not None else 72
        nb = len(bulbs_cfg)
        ns = len(switches_cfg)
        computed_bulb_row0, computed_sw_row0, computed_sep_y = _compute_layout(nb, ns)
        bulb_positions = _obj_positions(nb, computed_bulb_row0)
        sw_positions = _obj_positions(ns, computed_sw_row0)
        bulbs = [
            Bulb(id=b["id"], position=bulb_positions[i], state=b["state"])
            for i, b in enumerate(bulbs_cfg)
        ]
        switches = [
            Switch(
                id=s["id"],
                position=sw_positions[i],
                enabled=s.get("enabled", True),
                press_count=0,
                is_reset=s.get("reset", False),
                bulb_effects=s.get("bulb_effects", {}),
                switch_effects=s.get("switch_effects", {}),
            )
            for i, s in enumerate(switches_cfg)
        ]

        initial_bulb_states = {b.id: b.state for b in bulbs}
        initial_switch_enabled = {s.id: s.enabled for s in switches}
        return LevelState(
            bulbs=bulbs,
            switches=switches,
            max_presses=max_presses_cfg,
            initial_bulb_states=initial_bulb_states,
            initial_switch_enabled=initial_switch_enabled,
            bulb_row0=computed_bulb_row0,
            sw_row0=computed_sw_row0,
            separator_y=computed_sep_y,
        )

    def _grid_rows(self) -> List[list]:
        lvl = self._current_level
        nb = len(lvl.bulbs)
        ns = len(lvl.switches)
        rows: List[list] = []
        for r in range((nb + ROW_SIZE - 1) // ROW_SIZE):
            rows.append(list(lvl.bulbs[r * ROW_SIZE : (r + 1) * ROW_SIZE]))
        for r in range((ns + ROW_SIZE - 1) // ROW_SIZE):
            rows.append(list(lvl.switches[r * ROW_SIZE : (r + 1) * ROW_SIZE]))
        return rows

    def _num_bulb_rows(self) -> int:
        nb = len(self._current_level.bulbs)
        return (nb + ROW_SIZE - 1) // ROW_SIZE

    def _cursor_obj(self):
        rows = self._grid_rows()
        row = rows[self._cursor_row]
        col = min(self._cursor_col, len(row) - 1)
        return row[col]

    def _cursor_on_bulb(self) -> bool:
        return self._cursor_row < self._num_bulb_rows()

    def _get_bulb_by_id(self, bulb_id: int) -> Optional[Bulb]:
        for b in self._current_level.bulbs:
            if b.id == bulb_id:
                return b
        return None

    def _get_switch_by_id(self, sw_id: int) -> Optional[Switch]:
        for s in self._current_level.switches:
            if s.id == sw_id:
                return s
        return None

    def _check_win(self) -> bool:
        return all(b.state for b in self._current_level.bulbs)

    def _apply_reset(self) -> None:
        lvl = self._current_level
        for bulb in lvl.bulbs:
            bulb.state = lvl.initial_bulb_states[bulb.id]
        for sw in lvl.switches:
            sw.enabled = lvl.initial_switch_enabled[sw.id]
            sw.press_count = 0

    def _use_move(self) -> bool:
        lvl = self._current_level
        self._player.presses += 1
        if self._player.presses > lvl.max_presses:
            self._lose_life()
            return False
        return True

    def _push_state(self) -> None:
        lvl = self._current_level
        state = UndoState(
            bulb_states=tuple(b.state for b in lvl.bulbs),
            switch_enabled=tuple(s.enabled for s in lvl.switches),
            switch_press_counts=tuple(s.press_count for s in lvl.switches),
            cursor_row=self._cursor_row,
            cursor_col=self._cursor_col,
        )
        self._undo_stack.append(state)
        if len(self._undo_stack) > lvl.max_presses:
            self._undo_stack.pop(0)

    def _pop_state(self) -> bool:
        if not self._undo_stack:
            return False
        s = self._undo_stack.pop()
        lvl = self._current_level
        for i, bulb in enumerate(lvl.bulbs):
            bulb.state = s.bulb_states[i]
        for i, sw in enumerate(lvl.switches):
            sw.enabled = s.switch_enabled[i]
            sw.press_count = s.switch_press_counts[i]
        self._cursor_row = s.cursor_row
        self._cursor_col = s.cursor_col
        return True

    def _move_up(self) -> None:
        rows = self._grid_rows()
        if self._cursor_row > 0:
            self._push_state()
            self._cursor_row -= 1
            self._cursor_col = min(self._cursor_col, len(rows[self._cursor_row]) - 1)
            if self._use_move():
                self._render_all()

    def _move_down(self) -> None:
        rows = self._grid_rows()
        if self._cursor_row < len(rows) - 1:
            self._push_state()
            self._cursor_row += 1
            self._cursor_col = min(self._cursor_col, len(rows[self._cursor_row]) - 1)
            if self._use_move():
                self._render_all()

    def _move_left(self) -> None:
        rows = self._grid_rows()
        old_col = self._cursor_col
        n = len(rows[self._cursor_row])
        new_col = (self._cursor_col - 1) % n
        if new_col != old_col:
            self._push_state()
            self._cursor_col = new_col
            if self._use_move():
                self._render_all()

    def _move_right(self) -> None:
        rows = self._grid_rows()
        old_col = self._cursor_col
        n = len(rows[self._cursor_row])
        new_col = (self._cursor_col + 1) % n
        if new_col != old_col:
            self._push_state()
            self._cursor_col = new_col
            if self._use_move():
                self._render_all()

    def handle_reset(self) -> None:
        from arcengine import GameState as EngineState

        had_progress = (
            self._state == EngineState.GAME_OVER
            or self._player.presses > 0
            or self._lives < MAX_LIVES
            or any(b.state for b in self._current_level.bulbs)
        )
        self._lives = MAX_LIVES
        if self._state == EngineState.WIN:
            self.full_reset()
        elif had_progress:
            self.level_reset()
        else:
            self.full_reset()

    def _lose_life(self) -> None:
        self._lives -= 1
        if self._lives <= 0:
            self._game_over = True
            self._transition_pending = "game_over"
        else:
            self._transition_pending = "level_reset"
        self._render_all()

    def _activate(self) -> None:
        if self._cursor_on_bulb():
            return
        obj = self._cursor_obj()
        if not isinstance(obj, Switch):
            return
        sw = obj
        if not sw.enabled:
            return
        self._push_state()
        if not self._use_move():
            return
        if sw.is_reset:
            self._apply_reset()
        else:
            sw.press_count += 1

            for bulb_id_str, effect in sw.bulb_effects.items():
                bulb = self._get_bulb_by_id(int(bulb_id_str))
                if bulb and effect == "toggle":
                    bulb.state = not bulb.state

            for sw_id_str, effect in sw.switch_effects.items():
                target = self._get_switch_by_id(int(sw_id_str))
                if target and effect == "toggle":
                    target.enabled = not target.enabled
        if self._check_win():
            self._lives = MAX_LIVES
            self._render_all()
            self.next_level()
            return
        self._render_all()

    def _cam(self, gx: int, gy: int) -> Tuple[int, int]:
        return gx + OFFSET, gy + OFFSET

    def _place_7x7(self, sprite_name: str, gx: int, gy: int) -> None:
        cx, cy = self._cam(gx, gy)
        spr = sprites[sprite_name].clone()
        spr.set_position(cx - OBJ_HALF, cy - OBJ_HALF)
        self.current_level.add_sprite(spr)

    def _place_5x5(self, sprite_name: str, gx: int, gy: int) -> None:
        cx, cy = self._cam(gx, gy)
        spr = sprites[sprite_name].clone()
        spr.set_position(cx - (CORE_SIZE // 2), cy - (CORE_SIZE // 2))
        self.current_level.add_sprite(spr)

    def _place_1x1(self, sprite_name: str, cam_x: int, cam_y: int) -> None:
        spr = sprites[sprite_name].clone()
        spr.set_position(cam_x, cam_y)
        self.current_level.add_sprite(spr)

    def _render_border_and_bar(self) -> None:
        won = self._check_win()
        b = (
            "border_win"
            if won
            else ("border_fail" if self._game_over else "border_normal")
        )
        for row in BORDER_TOP_ROWS:
            for cx in range(CAMERA_W):
                self._place_1x1(b, cx, row)
        for row in BORDER_BOTTOM_ROWS:
            if row == BAR_ROW:
                lvl = self._current_level
                remaining = max(0, lvl.max_presses - self._player.presses)
                pip_area = MAX_LIVES * LIFE_SPACING + 1
                bar_w = BAR_WIDTH - pip_area
                fill = int((remaining / lvl.max_presses) * bar_w)
                for bar_row in [BAR_ROW - 1, BAR_ROW]:
                    for i in range(bar_w):
                        self._place_1x1(
                            "bar_fill" if i < fill else "bar_used", BAR_X0 + i, bar_row
                        )
                pip_x0 = BAR_X0 + bar_w + 1
                for i in range(MAX_LIVES):
                    cx = pip_x0 + i * LIFE_SPACING
                    color = LIFE_COLOR if i < self._lives else LIFE_EMPTY
                    pip = Sprite(
                        pixels=[[color, color], [color, color]],
                        name="life_pip",
                        visible=True,
                        collidable=False,
                    )
                    pip.set_position(cx, BAR_ROW - 1)
                    self.current_level.add_sprite(pip)
                for cx in list(range(BORDER)) + list(
                    range(CAMERA_W - BORDER, CAMERA_W)
                ):
                    self._place_1x1(b, cx, row)
            else:
                for cx in range(CAMERA_W):
                    self._place_1x1(b, cx, row)
        for cy in range(BORDER, CAMERA_H - BORDER):
            for cx in BORDER_LEFT_COLS:
                self._place_1x1(b, cx, cy)
            for cx in BORDER_RIGHT_COLS:
                self._place_1x1(b, cx, cy)

    def _render_all(self) -> None:
        self.current_level.remove_all_sprites()
        self._render_border_and_bar()
        lvl = self._current_level
        nb_rows = self._num_bulb_rows()
        grid = self._grid_rows()

        sep_y = lvl.separator_y

        for i, bulb in enumerate(lvl.bulbs):
            gx, gy = bulb.position
            b_row = i // ROW_SIZE
            b_col = i % ROW_SIZE
            clamped_col = min(self._cursor_col, len(grid[b_row]) - 1)
            is_cursor = b_row == self._cursor_row and b_col == clamped_col
            self._place_7x7("halo_bulb", gx, gy)
            self._place_5x5("bulb_on" if bulb.state else "bulb_off", gx, gy)

            if is_cursor:
                self._place_7x7("player", gx, gy)

        for i, sw in enumerate(lvl.switches):
            gx, gy = sw.position
            s_row = nb_rows + (i // ROW_SIZE)
            s_col = i % ROW_SIZE
            clamped_col = min(self._cursor_col, len(grid[s_row]) - 1)
            is_cursor = s_row == self._cursor_row and s_col == clamped_col
            if not sw.enabled:
                halo = "halo_disabled"
                core = "switch_disabled"
            elif sw.press_count % 2 == 1:
                halo = "halo_pressed"
                core = "switch_pressed"
            else:
                halo = "halo_switch"
                core = "switch_enabled"
            self._place_7x7(halo, gx, gy)
            self._place_5x5(core, gx, gy)

            if is_cursor:
                self._place_7x7("player", gx, gy)

    @property
    def extra_state(self) -> dict:
        lvl = self._current_level
        on_bulbs = sum(1 for b in lvl.bulbs if b.state)
        total_bulbs = len(lvl.bulbs)
        obj = self._cursor_obj()
        obj_label = f"Bulb {obj.id}" if self._cursor_on_bulb() else f"Switch {obj.id}"
        disabled_count = sum(
            1 for s in lvl.switches if not s.enabled and not s.is_reset
        )
        return {
            "presses": self._player.presses,
            "max_presses": lvl.max_presses,
            "bulbs_on": on_bulbs,
            "total_bulbs": total_bulbs,
            "cursor": obj_label,
            "switches_disabled": disabled_count,
            "lives": self._lives,
            "level_features": [
                f"Bulbs ON: {on_bulbs}/{total_bulbs}",
                f"Budget: {self._player.presses}/{lvl.max_presses}",
                f"Lives: {self._lives}/{MAX_LIVES}",
                f"Cursor: {obj_label}",
                "Turn ON all bulbs!",
            ],
        }

    def step(self) -> None:
        if self._transition_pending == "level_reset":
            self._transition_pending = ""
            self.level_reset()
            self.complete_action()
            return
        elif self._transition_pending == "game_over":
            self._transition_pending = ""
            self.lose()
            self.complete_action()
            return

        action = self.action.id
        if action == GameAction.ACTION1:
            self._move_up()
        elif action == GameAction.ACTION2:
            self._move_down()
        elif action == GameAction.ACTION3:
            self._move_left()
        elif action == GameAction.ACTION4:
            self._move_right()
        elif action == GameAction.ACTION5:
            self._activate()
        elif action == GameAction.ACTION7:
            self._pop_state()
            self._player.presses += 1
            if self._player.presses > self._current_level.max_presses:
                self._lose_life()
            self._render_all()
        self.complete_action()


class PuzzleEnvironment:
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
    _ACTION_MAP: Dict[str, GameAction] = {
        "reset": GameAction.RESET,
        "up": GameAction.ACTION1,
        "down": GameAction.ACTION2,
        "left": GameAction.ACTION3,
        "right": GameAction.ACTION4,
        "select": GameAction.ACTION5,
        "undo": GameAction.ACTION7,
    }

    _VALID_ACTIONS = list(_ACTION_MAP.keys())

    def __init__(self, seed: int = 0) -> None:
        self._engine = Cl01(seed=seed)
        self._total_turns = 0
        self._done = False
        self._game_won = False
        self._game_over = False
        self._last_action_was_reset = False

    def _frame_to_png(self, frame: np.ndarray) -> bytes:
        h, w = frame.shape[:2]
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for idx, color in enumerate(self.ARC_PALETTE):
            mask = frame == idx
            if mask.ndim == 3:
                mask = mask[:, :, 0]
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
        es = e.extra_state
        total = len(e._levels)

        header = (
            f"Level {e.level_index + 1}/{total}"
            f" | Presses: {es['presses']}/{es['max_presses']}"
            f" | Lives: {es['lives']}/{MAX_LIVES}"
        )
        rules = "Navigate cursor with up/down/left/right. Select to activate switch. Turn ON all bulbs!"

        bulb_parts = []
        for b in e._current_level.bulbs:
            status = "ON" if b.state else "OFF"
            bulb_parts.append(f"B{b.id}:{status}")
        bulb_line = " ".join(bulb_parts)

        switch_parts = []
        for s in e._current_level.switches:
            if not s.enabled:
                state = "DISABLED"
            elif s.is_reset:
                state = "RESET"
            elif s.press_count % 2 == 1:
                state = "PRESSED"
            else:
                state = "READY"
            effects = []
            for bid in s.bulb_effects:
                effects.append(f"B{bid}")
            for sid in s.switch_effects:
                effects.append(f"\u2191S{sid}")
            tag = f"\u2192{','.join(effects)}" if effects else ""
            switch_parts.append(f"S{s.id}:{state}{tag}")
        switch_line = " ".join(switch_parts)

        cursor_label = es["cursor"]
        return (
            header
            + "\n"
            + rules
            + f"\nBulbs: {bulb_line}"
            + f"\n  ON: {es['bulbs_on']}/{es['total_bulbs']}"
            + f"\nSwitches: {switch_line}"
            + f"\nCursor: {cursor_label}"
        )

    def _build_game_state(self, done: bool = False) -> GameState:
        e = self._engine
        frame = self._render_frame()
        image_bytes = self._frame_to_png(frame)

        valid_actions = self.get_actions() if not done else None

        return GameState(
            text_observation=self._build_text_observation(),
            image_observation=image_bytes,
            valid_actions=valid_actions,
            turn=self._total_turns,
            metadata={
                "total_levels": len(e._levels),
                "level_index": e.level_index,
                "levels_completed": getattr(e, "_score", 0),
                "game_over": self._game_over,
                "done": done,
                "info": {},
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
        self._game_over = False

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
                f"Invalid action '{action}'. "
                f"Must be one of {list(self._ACTION_MAP.keys())}"
            )

        self._last_action_was_reset = False
        self._total_turns += 1

        game_action = self._ACTION_MAP[action]
        info: Dict[str, Any] = {"action": action}

        level_before = e.level_index

        action_input = ActionInput(id=game_action)
        frame = e.perform_action(action_input, raw=True)

        state_name = frame.state.name if frame and frame.state else ""
        game_won = state_name == "WIN"
        game_over = state_name == "GAME_OVER"

        total_levels = len(self._engine._levels)
        level_reward = 1.0 / total_levels

        if game_won:
            self._done = True
            self._game_won = True
            self._game_over = False
            info["reason"] = "game_complete"
            return StepResult(
                state=self._build_game_state(done=True),
                reward=level_reward,
                done=True,
                info=info,
            )

        if game_over:
            self._done = True
            self._game_won = False
            self._game_over = True
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
        for idx, color in enumerate(self.ARC_PALETTE):
            mask = index_grid == idx
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            rgb[mask] = color
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
