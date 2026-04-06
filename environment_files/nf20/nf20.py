import random
import struct
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from arcengine import (
    ActionInput,
    ARCBaseGame,
    Camera,
    GameAction,
    Level,
    RenderableUserDisplay,
    Sprite,
)
from gymnasium import spaces


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

STEP = 3
GRID_COLS = 10
GRID_ROWS = 10
MAX_LIVES = 3
MOVE_LIMIT = 90
GOAL_CORNERS = [(1, 1), (8, 1), (1, 8), (8, 8)]


def _px(col: int, row: int) -> Tuple[int, int]:
    return (col * STEP, row * STEP)


sprites = {
    "atom_w": Sprite(
        pixels=[[-1, 0, -1], [0, 0, 0], [-1, 0, -1]],
        name="atom_w",
        visible=True,
        collidable=False,
        layer=4,
        tags=["atom_w"],
    ),
    "atom_c": Sprite(
        pixels=[[-1, 10, -1], [10, 10, 10], [-1, 10, -1]],
        name="atom_c",
        visible=True,
        collidable=False,
        layer=4,
        tags=["atom_c"],
    ),
    "atom_r": Sprite(
        pixels=[[-1, 8, -1], [8, 8, 8], [-1, 8, -1]],
        name="atom_r",
        visible=True,
        collidable=False,
        layer=4,
        tags=["atom_r"],
    ),
    "energy": Sprite(
        pixels=[[11, 11, 11], [11, 11, 11], [11, 11, 11]],
        name="energy",
        visible=True,
        collidable=False,
        layer=2,
        tags=["energy"],
    ),
    "bubble": Sprite(
        pixels=[[10, 10, 10], [10, -1, 10], [10, 10, 10]],
        name="bubble",
        visible=True,
        collidable=False,
        layer=6,
        tags=["bubble"],
    ),
    "toxic": Sprite(
        pixels=[[8, 8, 8], [8, -1, 8], [8, 8, 8]],
        name="toxic",
        visible=True,
        collidable=False,
        layer=6,
        tags=["toxic"],
    ),
    "fizzle": Sprite(
        pixels=[[2, -1, 2], [-1, 2, -1], [2, -1, 2]],
        name="fizzle",
        visible=True,
        collidable=False,
        layer=7,
        tags=["fizzle"],
    ),
    "goal": Sprite(
        pixels=[[14, 14, 14], [14, -1, 14], [14, 14, 14]],
        name="goal",
        visible=True,
        collidable=False,
        layer=2,
        tags=["goal"],
    ),
    "wall": Sprite(
        pixels=[[3, 3, 3], [3, 3, 3], [3, 3, 3]],
        name="wall",
        visible=True,
        collidable=True,
        layer=1,
        tags=["wall"],
    ),
    "active_mark": Sprite(
        pixels=[[14, -1, 14], [-1, -1, -1], [14, -1, 14]],
        name="active_mark",
        visible=True,
        collidable=False,
        layer=8,
        tags=["active_mark"],
    ),
    "slot": Sprite(
        pixels=[[4, 4, 4], [4, -1, 4], [4, 4, 4]],
        name="slot",
        visible=True,
        collidable=False,
        layer=1,
        tags=["slot"],
    ),
    "slot_middle": Sprite(
        pixels=[[4, 4, 4], [4, -1, 4], [4, 4, 4]],
        name="slot_middle",
        visible=True,
        collidable=False,
        layer=1,
        tags=["slot_middle"],
    ),
    "slot_inner": Sprite(
        pixels=[[3, 3, 3], [3, -1, 3], [3, 3, 3]],
        name="slot_inner",
        visible=True,
        collidable=False,
        layer=1,
        tags=["slot_inner"],
    ),
    "glow": Sprite(
        pixels=[[11, 11, 11], [11, 0, 11], [11, 11, 11]],
        name="glow",
        visible=True,
        collidable=False,
        layer=7,
        tags=["glow"],
    ),
}

ATOM_TYPE_MAP = {"w": "atom_w", "c": "atom_c", "r": "atom_r"}

RING_OUTER = [(5, 3), (7, 4), (7, 5), (7, 6), (5, 7), (3, 6), (3, 5), (3, 4)]
RING_MIDDLE = [(4, 4), (6, 4), (6, 6), (4, 6)]
RING_INNER = [(5, 4), (6, 5), (5, 6), (4, 5)]
CENTER = (5, 5)

PHASE_ROTATE = "ROTATE"
PHASE_BUBBLE = "BUBBLE"


def _build_border_walls() -> List[Sprite]:
    wall_sprites: List[Sprite] = []
    for col in range(GRID_COLS):
        for row in range(GRID_ROWS):
            if col == 0 or col == GRID_COLS - 1 or row == 0 or row == GRID_ROWS - 1:
                wall_sprites.append(
                    sprites["wall"].clone().set_position(*_px(col, row))
                )
    return wall_sprites


LEVEL_DATA = [
    {
        "rings": [["r", "w", "r", "w", "r", "w", "r", "w"]],
        "active": [[0, 4]],
        "rule": "ww",
        "moves": {3: (-1, 0, 0), 4: (1, 0, 0)},
        "goal_pos": (8, 1),
    },
    {
        "rings": [["r", "c", "r", "r", "r", "c", "r", "c"]],
        "active": [[0, 4]],
        "rule": "cc",
        "moves": {3: (-1, 0, 0), 4: (1, 0, 0)},
        "goal_pos": (1, 1),
    },
    {
        "rings": [
            ["r", "r", "r", "c", "r", "r", "r", "w"],
            ["r", "r", "c", "w"],
        ],
        "active": [[1, 5], [0, 3]],
        "rule": "wc",
        "moves": {
            1: (1, 0, 0),
            2: (-1, 0, 0),
            3: (0, 0, -1),
            4: (0, 0, 1),
        },
        "goal_pos": (8, 8),
    },
    {
        "rings": [
            ["r", "w", "r", "r", "c", "r", "r", "r"],
            ["w", "r", "r", "c"],
            ["r", "w", "c", "r"],
        ],
        "active": [[3, 6], [1, 2], [0, 1]],
        "rule": "wc_both",
        "moves": {
            1: (1, 3, 0),
            2: (-1, -3, 0),
            3: (0, 0, -1),
            4: (0, 0, 1),
        },
        "goal_pos": (1, 8),
    },
]


def _add_ring_sprites(
    sprite_list: List[Sprite],
    ring_data: List[str],
    ring_positions: List[Tuple[int, int]],
    active_indices: List[int],
    slot_type: str,
) -> None:
    for idx, (col, row) in enumerate(ring_positions):
        sprite_list.append(sprites[slot_type].clone().set_position(*_px(col, row)))
        atom_sprite_name = ATOM_TYPE_MAP[ring_data[idx]]
        sprite_list.append(
            sprites[atom_sprite_name].clone().set_position(*_px(col, row))
        )
        if idx in active_indices:
            sprite_list.append(
                sprites["active_mark"].clone().set_position(*_px(col, row))
            )


def _build_level(level_index: int) -> List[Sprite]:
    level_config = LEVEL_DATA[level_index]
    sprite_list = _build_border_walls()
    center_col, center_row = CENTER
    sprite_list.append(
        sprites["energy"].clone().set_position(*_px(center_col, center_row))
    )
    goal_col, goal_row = level_config["goal_pos"]
    sprite_list.append(sprites["goal"].clone().set_position(*_px(goal_col, goal_row)))

    ring_count = len(level_config["rings"])
    _add_ring_sprites(
        sprite_list,
        level_config["rings"][0],
        RING_OUTER,
        level_config["active"][0],
        "slot",
    )

    if ring_count == 2:
        _add_ring_sprites(
            sprite_list,
            level_config["rings"][1],
            RING_INNER,
            level_config["active"][1],
            "slot_inner",
        )
    elif ring_count == 3:
        _add_ring_sprites(
            sprite_list,
            level_config["rings"][1],
            RING_MIDDLE,
            level_config["active"][1],
            "slot_middle",
        )
        _add_ring_sprites(
            sprite_list,
            level_config["rings"][2],
            RING_INNER,
            level_config["active"][2],
            "slot_inner",
        )

    return sprite_list


levels = [
    Level(
        sprites=_build_level(i),
        grid_size=(32, 32),
        data={"level_idx": i, "max_steps": MOVE_LIMIT},
        name=f"Level {i + 1}",
    )
    for i in range(len(LEVEL_DATA))
]

BACKGROUND_COLOR = 5
PADDING_COLOR = 4


class WheelHUD(RenderableUserDisplay):
    def __init__(self, game: "Nf20", max_steps: int) -> None:
        self.game = game
        self.max_steps = max_steps
        self.steps_left = max_steps

    def set_max(self, value: int) -> None:
        self.max_steps = value
        self.steps_left = value + 1

    def tick(self) -> bool:
        if self.steps_left > 0:
            self.steps_left -= 1
        return self.steps_left > 0

    def reset(self) -> None:
        self.steps_left = self.max_steps

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        bar_row = 61

        bar_length = min(self.max_steps, 40)
        filled_count = min(
            bar_length, (bar_length * self.steps_left) // max(self.max_steps, 1)
        )
        for i in range(bar_length):
            pixel_x = 2 + i
            if pixel_x >= 44:
                break
            color = 11 if i < filled_count else 3
            frame[bar_row, pixel_x] = color
            frame[bar_row + 1, pixel_x] = color

        for life_idx in range(MAX_LIVES):
            pixel_x = 58 - life_idx * 4
            color = 8 if self.game.lives > life_idx else 3
            frame[bar_row, pixel_x] = color
            frame[bar_row + 1, pixel_x] = color
            frame[bar_row, pixel_x + 1] = color
            frame[bar_row + 1, pixel_x + 1] = color

        return frame


class Nf20(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)
        self.hud = WheelHUD(self, MOVE_LIMIT)
        self.total_levels = len(levels)
        self.current_level_index: int = 0
        self.is_game_over: bool = False
        self._game_won: bool = False
        self._moves_on_level: int = 0
        self._last_goal_pos: Optional[Tuple[int, int]] = None
        self._history: List[dict] = []

        super().__init__(
            game_id="nf20",
            levels=levels,
            camera=Camera(
                0,
                0,
                32,
                32,
                BACKGROUND_COLOR,
                PADDING_COLOR,
                [self.hud],
            ),
            available_actions=[0, 1, 2, 3, 4, 5, 7],
            win_score=1,
        )
        self._initialize_level_state()

    def _initialize_level_state(self) -> None:
        level_index = self.current_level.get_data("level_idx") or 0
        config = LEVEL_DATA[level_index]
        self.hud.set_max(MOVE_LIMIT)
        self._level_config = config

        ring_count = len(config["rings"])
        self.outer_ring = list(config["rings"][0])

        if ring_count == 2:
            self.middle_ring: Optional[List[str]] = None
            self.inner_ring: Optional[List[str]] = list(config["rings"][1])
        elif ring_count == 3:
            self.middle_ring = list(config["rings"][1])
            self.inner_ring = list(config["rings"][2])
        else:
            self.middle_ring = None
            self.inner_ring = None

        self.active_outer = config["active"][0]

        if ring_count == 2:
            self.active_middle: List[int] = []
            self.active_inner: List[int] = config["active"][1]
        elif ring_count == 3:
            self.active_middle = config["active"][1]
            self.active_inner = config["active"][2]
        else:
            self.active_middle = []
            self.active_inner = []

        self.reaction_rule = config["rule"]
        self.move_map: Dict[int, Tuple[int, int, int]] = config["moves"]

        self.phase = PHASE_ROTATE
        self.bubble_is_good = True
        self.animation_timer = 0
        self.animation_sprites: List[Sprite] = []
        self.bubbles: List[Sprite] = []
        self._randomize_goal()

    def _randomize_goal(self) -> None:
        goal_sprites = list(self.current_level.get_sprites_by_tag("goal"))
        candidates = [c for c in GOAL_CORNERS if c != self._last_goal_pos]
        new_pos = self._rng.choice(candidates)
        self._last_goal_pos = new_pos
        for g in goal_sprites:
            g.set_position(*_px(*new_pos))

    def on_set_level(self, level: Level) -> None:
        for i, lv in enumerate(self._levels):
            if lv is level:
                self.current_level_index = i
                break
        self.lives = MAX_LIVES
        self.animation_timer = 0
        self.animation_sprites = []
        self.bubbles = []
        self.is_game_over = False
        self._moves_on_level = 0
        self._history = []
        self._initialize_level_state()

    def _refresh_atom_sprites(self) -> None:
        for tag in ["atom_w", "atom_c", "atom_r"]:
            for existing_sprite in list(self.current_level.get_sprites_by_tag(tag)):
                self.current_level.remove_sprite(existing_sprite)

        self._draw_ring_atoms(self.outer_ring, RING_OUTER)

        if self.middle_ring is not None:
            self._draw_ring_atoms(self.middle_ring, RING_MIDDLE)

        if self.inner_ring is not None:
            self._draw_ring_atoms(self.inner_ring, RING_INNER)

    def _draw_ring_atoms(
        self,
        ring_state: List[str],
        ring_positions: List[Tuple[int, int]],
    ) -> None:
        for idx, (col, row) in enumerate(ring_positions):
            if idx < len(ring_state) and ring_state[idx] != "x":
                sprite_name = ATOM_TYPE_MAP.get(ring_state[idx])
                if sprite_name is not None:
                    self.current_level.add_sprite(
                        sprites[sprite_name].clone().set_position(*_px(col, row))
                    )

    @staticmethod
    def _rotate_ring(ring: List[str], steps: int) -> List[str]:
        length = len(ring)
        normalized = steps % length
        if normalized == 0:
            return list(ring)
        return ring[-normalized:] + ring[:-normalized]

    def _execute_rotation(self, action_id: int) -> bool:
        if action_id not in self.move_map:
            return False

        outer_steps, middle_steps, inner_steps = self.move_map[action_id]
        self.outer_ring = self._rotate_ring(self.outer_ring, outer_steps)

        if self.middle_ring is not None and middle_steps != 0:
            self.middle_ring = self._rotate_ring(self.middle_ring, middle_steps)

        if self.inner_ring is not None and inner_steps != 0:
            self.inner_ring = self._rotate_ring(self.inner_ring, inner_steps)

        self._refresh_atom_sprites()
        return True

    def _collect_active_slot_types(self) -> List[str]:
        active_types: List[str] = []
        for idx in self.active_outer:
            if idx < len(self.outer_ring):
                active_types.append(self.outer_ring[idx])
        for idx in self.active_middle:
            if self.middle_ring is not None and idx < len(self.middle_ring):
                active_types.append(self.middle_ring[idx])
        for idx in self.active_inner:
            if self.inner_ring is not None and idx < len(self.inner_ring):
                active_types.append(self.inner_ring[idx])
        return active_types

    def _is_reaction_aligned(self) -> bool:
        active_types = self._collect_active_slot_types()

        if self.reaction_rule == "ww":
            return all(atom_type == "w" for atom_type in active_types)

        if self.reaction_rule == "cc":
            return all(atom_type == "c" for atom_type in active_types)

        if self.reaction_rule == "wc":
            return (
                "w" in active_types and "c" in active_types and "r" not in active_types
            )

        if self.reaction_rule == "wc_both":
            return self._check_each_ring_has_white_and_cyan()

        return False

    def _check_each_ring_has_white_and_cyan(self) -> bool:
        outer_types = [
            self.outer_ring[i] for i in self.active_outer if i < len(self.outer_ring)
        ]
        if "r" in outer_types or "w" not in outer_types or "c" not in outer_types:
            return False

        if self.middle_ring is not None:
            middle_types = [
                self.middle_ring[i]
                for i in self.active_middle
                if i < len(self.middle_ring)
            ]
            if (
                "r" in middle_types
                or "w" not in middle_types
                or "c" not in middle_types
            ):
                return False

        if self.inner_ring is not None:
            inner_types = [
                self.inner_ring[i]
                for i in self.active_inner
                if i < len(self.inner_ring)
            ]
            if "r" in inner_types or "w" not in inner_types or "c" not in inner_types:
                return False

        return True

    def _has_red_in_active_slots(self) -> bool:
        return "r" in self._collect_active_slot_types()

    def _attempt_reaction(self) -> str:
        center_col, center_row = CENTER

        if self._is_reaction_aligned():
            bubble = (
                sprites["bubble"].clone().set_position(*_px(center_col, center_row))
            )
            self.current_level.add_sprite(bubble)
            self.bubbles.append(bubble)
            self.bubble_is_good = True

            glow = sprites["glow"].clone().set_position(*_px(center_col, center_row))
            self.current_level.add_sprite(glow)
            self.animation_sprites.append(glow)
            self.animation_timer = 1

            self._consume_active_atoms()
            self._refresh_atom_sprites()
            return "good"

        if self._has_red_in_active_slots():
            toxic_bubble = (
                sprites["toxic"].clone().set_position(*_px(center_col, center_row))
            )
            self.current_level.add_sprite(toxic_bubble)
            self.bubbles.append(toxic_bubble)
            self.bubble_is_good = False

            self._consume_active_atoms()
            self._refresh_atom_sprites()
            self.animation_timer = 1
            return "toxic"

        fizzle = sprites["fizzle"].clone().set_position(*_px(center_col, center_row))
        self.current_level.add_sprite(fizzle)
        self.animation_sprites.append(fizzle)
        self.animation_timer = 1
        return "fizzle"

    def _consume_active_atoms(self) -> None:
        for idx in self.active_outer:
            if idx < len(self.outer_ring):
                self.outer_ring[idx] = "x"
        for idx in self.active_middle:
            if self.middle_ring is not None and idx < len(self.middle_ring):
                self.middle_ring[idx] = "x"
        for idx in self.active_inner:
            if self.inner_ring is not None and idx < len(self.inner_ring):
                self.inner_ring[idx] = "x"

    def _has_bubble_reached_goal(self) -> bool:
        for goal_sprite in self.current_level.get_sprites_by_tag("goal"):
            for bubble in list(self.bubbles):
                if bubble.x == goal_sprite.x and bubble.y == goal_sprite.y:
                    self.current_level.remove_sprite(bubble)
                    self.bubbles.remove(bubble)
                    return True
        return False

    def _lose_life(self) -> None:
        self.lives -= 1
        if self.lives <= 0:
            self.is_game_over = True
            self.lose()
            return
        self._reset_level_to_initial()
        self._randomize_goal()

    def _reset_level_to_initial(self) -> None:
        level_index = self.current_level.get_data("level_idx") or 0
        config = LEVEL_DATA[level_index]

        ring_count = len(config["rings"])
        self.outer_ring = list(config["rings"][0])

        if ring_count == 2:
            self.middle_ring = None
            self.inner_ring = list(config["rings"][1])
        elif ring_count == 3:
            self.middle_ring = list(config["rings"][1])
            self.inner_ring = list(config["rings"][2])
        else:
            self.middle_ring = None
            self.inner_ring = None

        self._refresh_atom_sprites()

        for bubble in self.bubbles:
            try:
                self.current_level.remove_sprite(bubble)
            except Exception:
                pass
        self.bubbles = []

        for anim_sprite in self.animation_sprites:
            try:
                self.current_level.remove_sprite(anim_sprite)
            except Exception:
                pass
        self.animation_sprites = []
        self.animation_timer = 0

        self.phase = PHASE_ROTATE
        self.bubble_is_good = True
        self.hud.set_max(MOVE_LIMIT)

    def _clear_animation_sprites(self) -> None:
        for animation_sprite in self.animation_sprites:
            try:
                self.current_level.remove_sprite(animation_sprite)
            except Exception:
                pass
        self.animation_sprites.clear()

    def _move_bubble(self, dx: int, dy: int) -> None:
        for bubble in self.bubbles:
            target_x = bubble.x + dx
            target_y = bubble.y + dy
            if 0 <= target_x < GRID_COLS * STEP and 0 <= target_y < GRID_ROWS * STEP:
                is_blocked = any(
                    wall.x == target_x and wall.y == target_y
                    for wall in self.current_level.get_sprites_by_tag("wall")
                )
                if not is_blocked:
                    bubble.set_position(target_x, target_y)

    def _handle_bubble_phase(self) -> None:
        action = self.action.id

        if action == GameAction.ACTION1:
            self._move_bubble(0, -STEP)
        elif action == GameAction.ACTION2:
            self._move_bubble(0, STEP)
        elif action == GameAction.ACTION3:
            self._move_bubble(-STEP, 0)
        elif action == GameAction.ACTION4:
            self._move_bubble(STEP, 0)

        if self._has_bubble_reached_goal():
            if self.bubble_is_good:
                if self.current_level_index >= self.total_levels - 1:
                    self._game_won = True
                self.next_level()
            else:
                self._lose_life()
            self.complete_action()
            return

        if not self.hud.tick():
            self._lose_life()
            self.complete_action()
            return

        self.complete_action()

    def _handle_rotate_phase(self) -> None:
        action = self.action.id

        action_number = action.value if hasattr(action, "value") else int(action)
        if action_number in self.move_map:
            self._execute_rotation(action_number)

        if action == GameAction.ACTION5:
            result = self._attempt_reaction()
            if result == "good" or result == "toxic":
                self.phase = PHASE_BUBBLE

        if not self.hud.tick():
            self._lose_life()
            self.complete_action()
            return

        self.complete_action()

    def _reset_current_level(self) -> None:
        self._reset_level_to_initial()
        self.lives = MAX_LIVES
        self.is_game_over = False
        self._moves_on_level = 0
        self._history = []

    def _save_state(self) -> None:
        snap: dict = {
            "outer_ring": list(self.outer_ring),
            "middle_ring": list(self.middle_ring)
            if self.middle_ring is not None
            else None,
            "inner_ring": list(self.inner_ring)
            if self.inner_ring is not None
            else None,
            "phase": self.phase,
            "bubble_is_good": self.bubble_is_good,
            "animation_timer": self.animation_timer,
            "lives": self.lives,
            "hud_steps_left": self.hud.steps_left,
            "_moves_on_level": self._moves_on_level,
            "bubble_positions": [(b.x, b.y) for b in self.bubbles],
            "bubble_is_toxic": [
                any("toxic" in t for t in b.tags) for b in self.bubbles
            ],
        }
        self._history.append(snap)

    def _undo(self) -> None:
        if not self._history:
            return
        snap = self._history.pop()
        self.outer_ring = list(snap["outer_ring"])
        self.middle_ring = (
            list(snap["middle_ring"]) if snap["middle_ring"] is not None else None
        )
        self.inner_ring = (
            list(snap["inner_ring"]) if snap["inner_ring"] is not None else None
        )
        self.phase = snap["phase"]
        self.bubble_is_good = snap["bubble_is_good"]
        self.animation_timer = snap["animation_timer"]
        self.lives = snap["lives"]
        self._moves_on_level = snap["_moves_on_level"]

        for bubble in self.bubbles:
            try:
                self.current_level.remove_sprite(bubble)
            except Exception:
                pass
        self.bubbles = []

        for anim_sprite in self.animation_sprites:
            try:
                self.current_level.remove_sprite(anim_sprite)
            except Exception:
                pass
        self.animation_sprites = []

        for i, (bx, by) in enumerate(snap["bubble_positions"]):
            is_toxic = snap["bubble_is_toxic"][i]
            bubble_type = "toxic" if is_toxic else "bubble"
            bubble = sprites[bubble_type].clone().set_position(bx, by)
            self.current_level.add_sprite(bubble)
            self.bubbles.append(bubble)

        self._refresh_atom_sprites()

    def step(self) -> None:
        if self.animation_timer > 0:
            self.animation_timer -= 1
            if self.animation_timer == 0:
                self._clear_animation_sprites()
            else:
                self.complete_action()
                return

        if self.action.id == GameAction.RESET:
            self._reset_current_level()
            self.complete_action()
            return

        if self.action.id == GameAction.ACTION6:
            self._undo()
            if not self.hud.tick():
                self._lose_life()
            self.complete_action()
            return

        self._save_state()
        self._moves_on_level += 1

        if self.phase == PHASE_BUBBLE:
            self._handle_bubble_phase()
            return

        self._handle_rotate_phase()


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

    _VALID_ACTIONS = list(_ACTION_MAP.keys())

    def __init__(self, seed: int = 0) -> None:
        self._engine = Nf20(seed)
        self.seed = seed
        self._total_turns = 0
        self._done = False
        self._last_action_was_reset: bool = False

    def _episode_terminal(self) -> bool:
        g = self._engine
        return (
            g._game_won
            or g.is_game_over
            or g.current_level_index >= g.total_levels
            or g.lives <= 0
        )

    def reset(self) -> GameState:
        e = self._engine
        self._total_turns = 0
        self._done = False
        game_won = e._game_won or e.current_level_index >= e.total_levels
        no_moves = e._moves_on_level == 0 and not e.is_game_over
        if game_won or no_moves or self._last_action_was_reset:
            self._engine = Nf20(self.seed)
            e = self._engine
        else:
            e.perform_action(ActionInput(id=GameAction.RESET))
        self._last_action_was_reset = True
        return self._build_game_state()

    def get_actions(self) -> List[str]:
        if self._episode_terminal():
            return ["reset"]
        return list(self._VALID_ACTIONS)

    def is_done(self) -> bool:
        return self._done or self._episode_terminal()

    def _outcome_after_step(
        self, lives_before: int, level_before: int
    ) -> Tuple[float, bool, Dict]:
        g = self._engine
        info: Dict = {
            "lives": g.lives,
            "level": g.current_level_index + 1,
            "phase": g.phase,
        }
        reward = 0.0
        done = False
        if g.lives < lives_before:
            info["event"] = "life_lost"
            if g.lives <= 0:
                info["event"] = "game_over"
                done = True
        elif g._game_won and level_before == g.total_levels - 1:
            reward = 1.0 / g.total_levels
            info["event"] = "game_complete"
            done = True
        elif g.current_level_index != level_before:
            reward = 1.0 / g.total_levels
            info["event"] = "level_complete"
        return reward, done, info

    def step(self, action: str) -> StepResult:
        e = self._engine

        if action == "reset":
            game_won = e._game_won or e.current_level_index >= e.total_levels
            no_moves = e._moves_on_level == 0 and not e.is_game_over
            full_restart = game_won or no_moves
            state = self.reset()
            return StepResult(
                state=state,
                reward=0.0,
                done=False,
                info={"action": "reset", "full_restart": full_restart},
            )

        if action not in self._ACTION_MAP:
            return StepResult(
                state=self._build_game_state(),
                reward=0.0,
                done=self.is_done(),
                info={"error": f"Invalid action: {action}"},
            )

        self._last_action_was_reset = False
        self._total_turns += 1

        lives_before = e.lives
        level_before = e.current_level_index

        ga = self._ACTION_MAP[action]
        action_input = ActionInput(id=ga)
        e.perform_action(action_input)

        reward, done, info = self._outcome_after_step(lives_before, level_before)
        self._done = done or self._episode_terminal()

        return StepResult(
            state=self._build_game_state(),
            reward=reward,
            done=self._done,
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
        h, w, _ = rgb.shape
        raw = bytearray()
        for y in range(h):
            raw.append(0)
            raw.extend(rgb[y].tobytes())
        compressed = zlib.compress(bytes(raw))

        def _chunk(ctype: bytes, data: bytes) -> bytes:
            c = ctype + data
            return (
                struct.pack(">I", len(data))
                + c
                + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
            )

        sig = b"\x89PNG\r\n\x1a\n"
        ihdr = _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        idat = _chunk(b"IDAT", compressed)
        iend = _chunk(b"IEND", b"")
        return sig + ihdr + idat + iend

    def _build_game_state(self) -> GameState:
        g = self._engine
        body = self._build_text_observation()
        text_observation = f"Agent turn: {self._total_turns}\n\n{body}"
        va = self.get_actions()
        return GameState(
            text_observation=text_observation,
            image_observation=self._build_image_bytes(),
            valid_actions=va,
            turn=self._total_turns,
            metadata={
                "total_levels": g.total_levels,
                "level_index": g.current_level_index,
                "levels_completed": getattr(g, "_score", 0),
                "game_over": g.is_game_over,
                "done": self._done,
                "info": {},
            },
        )

    def _build_text_observation(self) -> str:
        g = self._engine
        lines = []
        lines.append(
            f"LEVEL {g.current_level_index + 1}/{g.total_levels} "
            f"LIVES {g.lives}/{MAX_LIVES} "
            f"MOVES {g.hud.steps_left}/{g.hud.max_steps}"
        )
        lines.append(f"PHASE: {g.phase}")
        lines.append(f"RULE: {g.reaction_rule}")

        outer_str = " ".join(g.outer_ring)
        lines.append(f"OUTER: {outer_str}")
        if g.middle_ring is not None:
            middle_str = " ".join(g.middle_ring)
            lines.append(f"MIDDLE: {middle_str}")
        if g.inner_ring is not None:
            inner_str = " ".join(g.inner_ring)
            lines.append(f"INNER: {inner_str}")

        lines.append(f"BUBBLES: {len(g.bubbles)}")

        return "\n".join(lines)


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

    env = ArcGameEnv(seed=0, render_mode="rgb_array")
    check_env(env.unwrapped, skip_render_check=True)

    obs, info = env.reset()
    mask = env.action_mask()
    valid_indices = np.where(mask == 1)[0]
    if len(valid_indices) > 0:
        obs, reward, term, trunc, info = env.step(valid_indices[0])

    env.close()
