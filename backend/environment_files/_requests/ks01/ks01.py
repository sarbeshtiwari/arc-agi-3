from typing import List, Tuple

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, RenderableUserDisplay, Sprite

C_BG = 5
C_EMPTY = 3
C_CAPACITY = 12
C_USED = 12
C_TARGET = 14
C_VALUE = 14
C_CURSOR = 11
C_SELECTED = 0
C_UNSELECTED = 4
C_LIFE = 8
C_MOVE = 9
C_MOVE_EMPTY = 3

BACKGROUND_COLOR = 5
PADDING_COLOR = 3
CAM_SIZE = 16

_LEVELS = [
    {
        "items": [(1, 1), (2, 3), (3, 4), (2, 2)],
        "capacity": 5,
        "target": 7,
        "max_moves": 8,
        "solution": [1, 2],
    },
    {
        "items": [(1, 1), (2, 2), (3, 5), (4, 5), (2, 4)],
        "capacity": 6,
        "target": 10,
        "max_moves": 12,
        "solution": [0, 2, 4],
    },
    {
        "items": [(2, 1), (3, 4), (4, 5), (5, 6), (1, 1)],
        "capacity": 7,
        "target": 9,
        "max_moves": 8,
        "solution": [1, 2],
    },
    {
        "items": [(1, 1), (2, 2), (3, 5), (4, 7), (5, 6), (2, 3)],
        "capacity": 8,
        "target": 13,
        "max_moves": 12,
        "solution": [0, 2, 3],
    },
    {
        "items": [(2, 2), (3, 4), (4, 5), (5, 8), (6, 9), (2, 3)],
        "capacity": 10,
        "target": 15,
        "max_moves": 12,
        "solution": [1, 3, 5],
    },
]

def _px(color: int, layer: int = 0, name: str = "px") -> Sprite:
    return Sprite(
        pixels=np.array([[color]], dtype=np.int32),
        name=name,
        visible=True,
        collidable=False,
        tags=[],
        layer=layer,
    )

class KnapsackHUD(RenderableUserDisplay):

    MOVE_W = 5

    def __init__(self, game: "Knapsack01") -> None:
        self._g = game

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape
        cap = getattr(self._g, "_capacity", 0)
        target = getattr(self._g, "_target", 0)
        cur_w = self._g._current_weight() if hasattr(self._g, "_selected") else 0
        cur_v = self._g._current_value() if hasattr(self._g, "_selected") else 0
        moves = getattr(self._g, "_move_count", 0)
        max_moves = getattr(self._g, "_max_moves", 0)
        lives = getattr(self._g, "_lives", 0)

        for x in range(min(cap, w)):
            frame[0, x] = C_CAPACITY

        for i in range(3):
            x = w - 3 + i
            if 0 <= x < w:
                frame[0, x] = C_LIFE if i < lives else C_EMPTY

        if max_moves > 0:
            remain = max(0, max_moves - moves)
            filled = int(round(self.MOVE_W * (remain / max_moves)))
            filled = max(0, min(self.MOVE_W, filled))
            move_x = max(0, w - self.MOVE_W - 6)
            for i in range(self.MOVE_W):
                x = move_x + i
                if 0 <= x < w:
                    frame[0, x] = C_MOVE if i < filled else C_MOVE_EMPTY

        for x in range(min(cap, w)):
            frame[1, x] = C_EMPTY
        for x in range(min(cur_w, cap, w)):
            frame[1, x] = C_USED

        for x in range(min(target, w)):
            frame[2, x] = C_TARGET

        for x in range(min(target, w)):
            frame[3, x] = C_EMPTY
        for x in range(min(cur_v, target, w)):
            frame[3, x] = C_VALUE

        return frame

class Knapsack01(ARCBaseGame):

    def __init__(self) -> None:
        levels = [Level(sprites=[], grid_size=(CAM_SIZE, CAM_SIZE), data=d) for d in _LEVELS]
        self._hud = KnapsackHUD(self)
        self._lives = 3
        cam = Camera(0, 0, CAM_SIZE, CAM_SIZE, BACKGROUND_COLOR, PADDING_COLOR, [self._hud])
        super().__init__("ks01", levels, cam, False, 1, [1, 2, 3, 4, 5, 6])

    def next_level(self) -> None:
        self._lives = 3
        super().next_level()

    def on_set_level(self, level: Level) -> None:
        idx = min(self._current_level_index, len(_LEVELS) - 1)
        data = _LEVELS[idx]
        self._items: List[Tuple[int, int]] = list(data["items"])
        self._capacity = data["capacity"]
        self._target = data["target"]
        self._solution = list(data["solution"])
        self._max_moves = data["max_moves"]
        self._move_count = 0
        self._lives = 3

        self._selected = [False] * len(self._items)
        self._cursor_idx = 0

        self._create_item_sprites()
        self._create_cursor()

    def _create_item_sprites(self) -> None:
        self._item_markers = []
        self._item_weight_sprites = []
        self._item_value_sprites = []

        for idx, (weight, value) in enumerate(self._items):
            y0 = 4 + idx * 2

            left_top = _px(C_UNSELECTED, layer=4, name="mark")
            left_bot = _px(C_UNSELECTED, layer=4, name="mark")
            left_top.set_position(1, y0)
            left_bot.set_position(1, y0 + 1)
            self.current_level.add_sprite(left_top)
            self.current_level.add_sprite(left_bot)
            self._item_markers.append((left_top, left_bot))

            weight_row = []
            for x in range(4, 4 + weight):
                sp = _px(C_CAPACITY, layer=2, name="weight")
                sp.set_position(x, y0)
                self.current_level.add_sprite(sp)
                weight_row.append(sp)
            self._item_weight_sprites.append(weight_row)

            value_row = []
            for x in range(4, 4 + value):
                sp = _px(C_VALUE, layer=2, name="value")
                sp.set_position(x, y0 + 1)
                self.current_level.add_sprite(sp)
                value_row.append(sp)
            self._item_value_sprites.append(value_row)

        self._refresh_selection()

    def _create_cursor(self) -> None:
        self._cursor_top = _px(C_CURSOR, layer=10, name="cursor")
        self._cursor_bot = _px(C_CURSOR, layer=10, name="cursor")
        self.current_level.add_sprite(self._cursor_top)
        self.current_level.add_sprite(self._cursor_bot)
        self._update_cursor()

    def _update_cursor(self) -> None:
        y0 = 4 + self._cursor_idx * 2
        self._cursor_top.set_position(1, y0)
        self._cursor_bot.set_position(1, y0 + 1)

    def _current_weight(self) -> int:
        return sum(self._items[i][0] for i, sel in enumerate(self._selected) if sel)

    def _current_value(self) -> int:
        return sum(self._items[i][1] for i, sel in enumerate(self._selected) if sel)

    def _refresh_selection(self) -> None:
        for idx, selected in enumerate(self._selected):
            color = C_SELECTED if selected else C_UNSELECTED
            left_top, left_bot = self._item_markers[idx]
            left_top.pixels = np.array([[color]], dtype=np.int32)
            left_bot.pixels = np.array([[color]], dtype=np.int32)

    def _check_solution(self) -> bool:
        return self._current_weight() <= self._capacity and self._current_value() == self._target

    def _reset_level(self) -> None:
        self._selected = [False] * len(self._items)
        self._move_count = 0
        self._cursor_idx = 0
        self._refresh_selection()
        self._update_cursor()

    def _trigger_life_loss(self) -> bool:
        self._lives -= 1
        if self._lives <= 0:
            self.lose()
            self.complete_action()
            return True
        self._reset_level()
        return False

    def _toggle_current_item(self) -> bool:
        idx = self._cursor_idx

        if self._selected[idx]:
            self._selected[idx] = False
            self._move_count += 1
            self._refresh_selection()
            return True

        self._selected[idx] = True
        self._move_count += 1
        self._refresh_selection()
        return True

    def _handle_toggle_result(self) -> bool:
        if self._current_weight() > self._capacity or self._current_value() > self._target:
            if self._trigger_life_loss():
                return True
            self.complete_action()
            return True

        if self._check_solution():
            self.next_level()
            self.complete_action()
            return True

        if self._move_count >= self._max_moves:
            if self._trigger_life_loss():
                return True
            self.complete_action()
            return True

        return False

    def _click_item_index(self, gx: int, gy: int) -> int:
        if 4 <= gy < 4 + len(self._items) * 2 and 1 <= gx < CAM_SIZE:
            return (gy - 4) // 2
        return -1

    def step(self) -> None:
        act = self.action.id

        if act in (GameAction.ACTION1, GameAction.ACTION3):
            if self._cursor_idx > 0:
                self._cursor_idx -= 1
                self._update_cursor()
        elif act in (GameAction.ACTION2, GameAction.ACTION4):
            if self._cursor_idx < len(self._items) - 1:
                self._cursor_idx += 1
                self._update_cursor()
        elif act == GameAction.ACTION5:
            changed = self._toggle_current_item()
            if changed and self._handle_toggle_result():
                return
        elif getattr(act, "value", None) == 6:
            px = self.action.data.get("x", 0)
            py = self.action.data.get("y", 0)
            grid_pos = self.camera.display_to_grid(px, py)
            if grid_pos:
                gx, gy = grid_pos
                item_idx = self._click_item_index(gx, gy)
                if 0 <= item_idx < len(self._items):
                    self._cursor_idx = item_idx
                    self._update_cursor()
                    changed = self._toggle_current_item()
                    if changed and self._handle_toggle_result():
                        return

        self.complete_action()
