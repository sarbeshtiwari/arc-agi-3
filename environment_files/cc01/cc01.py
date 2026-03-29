from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import math
import random

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

import gymnasium as gym
from gymnasium import spaces


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


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
CYAN = 8
MAROON = 9
WHITE = 10

BACKGROUND_COLOR = GREY
PADDING_COLOR = GREY

_SPRITES = {
    "empty": Sprite(
        pixels=[[BLACK]],
        name="empty",
        visible=True,
        collidable=False,
        tags=["empty"],
        layer=-2,
    ),
    "blue": Sprite(
        pixels=[[BLUE]],
        name="blue",
        visible=True,
        collidable=False,
        tags=["block", "block_blue"],
        layer=1,
    ),
    "red": Sprite(
        pixels=[[RED]],
        name="red",
        visible=True,
        collidable=False,
        tags=["block", "block_red"],
        layer=1,
    ),
    "green": Sprite(
        pixels=[[GREEN]],
        name="green",
        visible=True,
        collidable=False,
        tags=["block", "block_green"],
        layer=1,
    ),
    "yellow": Sprite(
        pixels=[[YELLOW]],
        name="yellow",
        visible=True,
        collidable=False,
        tags=["block", "block_yellow"],
        layer=1,
    ),
    "gray": Sprite(
        pixels=[[GREY]],
        name="gray",
        visible=True,
        collidable=False,
        tags=["block", "block_gray"],
        layer=1,
    ),
    "pink": Sprite(
        pixels=[[MAGENTA]],
        name="pink",
        visible=True,
        collidable=False,
        tags=["block", "block_pink"],
        layer=1,
    ),
    "orange": Sprite(
        pixels=[[ORANGE]],
        name="orange",
        visible=True,
        collidable=False,
        tags=["block", "block_orange"],
        layer=1,
    ),
    "cyan": Sprite(
        pixels=[[CYAN]],
        name="cyan",
        visible=True,
        collidable=False,
        tags=["block", "block_cyan"],
        layer=1,
    ),
    "brown": Sprite(
        pixels=[[MAROON]],
        name="brown",
        visible=True,
        collidable=False,
        tags=["block", "block_brown"],
        layer=1,
    ),
    "clicked": Sprite(
        pixels=[[WHITE]],
        name="clicked",
        visible=True,
        collidable=False,
        tags=["clicked"],
        layer=2,
    ),
    "center": Sprite(
        pixels=[[WHITE]],
        name="center",
        visible=True,
        collidable=False,
        tags=["center"],
        layer=0,
    ),
    "selector": Sprite(
        pixels=[[YELLOW]],
        name="selector",
        visible=True,
        collidable=False,
        tags=["selector"],
        layer=5,
    ),
}

_COLOR_TO_SPRITE_KEY = {
    BLUE: "blue",
    RED: "red",
    GREEN: "green",
    YELLOW: "yellow",
    GREY: "gray",
    MAGENTA: "pink",
    ORANGE: "orange",
    CYAN: "cyan",
    MAROON: "brown",
}


def _build_level(data: Dict) -> Level:
    return Level(sprites=[], grid_size=(20, 20), data=data)


levels = [
    _build_level({"num_blocks": 4, "max_moves": 105, "center_x": 10, "center_y": 10}),
    _build_level({"num_blocks": 6, "max_moves": 115, "center_x": 7, "center_y": 10}),
    _build_level({"num_blocks": 8, "max_moves": 125, "center_x": 13, "center_y": 10}),
    _build_level({"num_blocks": 9, "max_moves": 135, "center_x": 10, "center_y": 7}),
    _build_level({"num_blocks": 10, "max_moves": 145, "center_x": 10, "center_y": 13}),
]


class GameHUD(RenderableUserDisplay):
    def __init__(self, game: "Cc01"):
        self._game = game

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        game = self._game

        frame_h, frame_w = frame.shape
        cam_w, cam_h = 20, 20
        scale = min(frame_w // cam_w, frame_h // cam_h)
        x_off = (frame_w - cam_w * scale) // 2
        y_off = (frame_h - cam_h * scale) // 2

        def grid_y(row):
            return slice(y_off + row * scale, y_off + (row + 1) * scale)

        def grid_x(col):
            return slice(x_off + col * scale, x_off + (col + 1) * scale)

        for i in range(3):
            frame[grid_y(0), grid_x(1 + i)] = RED if game._lives > i else BLACK

        return frame


class Cc01(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

        self.selector_x = 10
        self.selector_y = 10
        self.selector_sprite = None

        self._lives = 3
        self._hud = GameHUD(self)

        self._game_over = False
        self._moves_used = 0
        self._max_moves = 0
        self._history: List[Dict] = []

        super().__init__(
            "cc01",
            levels,
            Camera(
                x=0,
                y=0,
                width=20,
                height=20,
                background=BACKGROUND_COLOR,
                letter_box=PADDING_COLOR,
                interfaces=[self._hud],
            ),
            available_actions=[0, 1, 2, 3, 4, 5, 7],
        )
        self.sequence = []
        self.click_index = 0
        self.position_to_index = {}
        self.position_to_distance = {}
        self.move_count = 0
        self.initial_positions = []
        self.initial_distances = []
        self.center_x = 0
        self.center_y = 0

    def _generate_sequence(self, length: int) -> List[int]:
        return [BLUE] * length

    def on_set_level(self, level: Level) -> None:
        self._lives = 3

        for sprite in list(self.current_level.get_sprites()):
            self.current_level.remove_sprite(sprite)

        self._game_over = False
        self._moves_used = 0
        self._history = []
        self._max_moves = int(self.current_level.get_data("max_moves"))

        sequence_length = int(self.current_level.get_data("num_blocks"))
        self.center_x = int(self.current_level.get_data("center_x"))
        self.center_y = int(self.current_level.get_data("center_y"))

        self.sequence = self._generate_sequence(sequence_length)

        center_sprite = (
            _SPRITES["center"].clone().set_position(self.center_x, self.center_y)
        )
        self.current_level.add_sprite(center_sprite)

        self.click_index = 0
        self.move_count = 0

        self._randomize_spawn(self.center_x, self.center_y, sequence_length)

        self.selector_x = self.center_x
        self.selector_y = self.center_y
        self._update_selector()

    def _fallback_place(self, used_positions: set) -> Tuple[int, int]:
        for ox in range(1, 19):
            for oy in range(1, 19):
                if (ox, oy) not in used_positions:
                    return (ox, oy)
        return (1, 1)

    def _sort_and_place_blocks(
        self, positions: List[Tuple[int, int]], center_x: int, center_y: int
    ) -> None:
        positions_with_distance = []
        for pos in positions:
            dist = math.sqrt((pos[0] - center_x) ** 2 + (pos[1] - center_y) ** 2)
            positions_with_distance.append((dist, pos))

        positions_with_distance.sort(key=lambda x: x[0])

        self.initial_positions = [pos for dist, pos in positions_with_distance]
        self.initial_distances = [dist for dist, pos in positions_with_distance]

        self.position_to_index = {}
        self.position_to_distance = {}
        for i, (x, y) in enumerate(self.initial_positions):
            color = self.sequence[i]
            sprite_key = _COLOR_TO_SPRITE_KEY[color]
            sprite = _SPRITES[sprite_key].clone().set_position(x, y)
            self.current_level.add_sprite(sprite)
            self.position_to_index[(x, y)] = i
            self.position_to_distance[(x, y)] = self.initial_distances[i]

    def _randomize_spawn(
        self, center_x: int, center_y: int, sequence_length: int
    ) -> None:
        used_positions: set = set()
        used_positions.add((center_x, center_y))
        positions = []

        for i in range(sequence_length):
            radius = 3.0 + (i * 2.0)
            placed = False

            for _ in range(200):
                angle = self._rng.random() * 2.0 * math.pi
                x = int(center_x + radius * math.cos(angle))
                y = int(center_y + radius * math.sin(angle))
                x = max(1, min(18, x))
                y = max(1, min(18, y))

                if (x, y) not in used_positions:
                    positions.append((x, y))
                    used_positions.add((x, y))
                    placed = True
                    break

            if not placed:
                fallback = self._fallback_place(used_positions)
                positions.append(fallback)
                used_positions.add(fallback)

        self._sort_and_place_blocks(positions, center_x, center_y)

    def _update_selector(self):
        if self.selector_sprite:
            self.current_level.remove_sprite(self.selector_sprite)

        self.selector_sprite = (
            _SPRITES["selector"].clone().set_position(self.selector_x, self.selector_y)
        )
        self.current_level.add_sprite(self.selector_sprite)

    def _save_state(self) -> None:
        unclicked_sprites = []
        for sprite in self.current_level.get_sprites():
            if "block" in (sprite.tags or []):
                unclicked_sprites.append(
                    {"name": sprite.name, "x": sprite.x, "y": sprite.y}
                )
        self._history.append(
            {
                "selector_x": self.selector_x,
                "selector_y": self.selector_y,
                "click_index": self.click_index,
                "move_count": self.move_count,
                "position_to_index": deepcopy(self.position_to_index),
                "position_to_distance": deepcopy(self.position_to_distance),
                "unclicked_sprites": unclicked_sprites,
            }
        )

    def _undo(self) -> None:
        if not self._history:
            return
        snap = self._history.pop()
        self.selector_x = snap["selector_x"]
        self.selector_y = snap["selector_y"]
        self.click_index = snap["click_index"]
        self.move_count = snap["move_count"]
        self.position_to_index = snap["position_to_index"]
        self.position_to_distance = snap["position_to_distance"]

        sprites_to_remove = []
        for sprite in list(self.current_level.get_sprites()):
            tags = sprite.tags or []
            if "center" not in tags and "empty" not in tags:
                sprites_to_remove.append(sprite)
        for sprite in sprites_to_remove:
            self.current_level.remove_sprite(sprite)

        for clicked_pos in self.initial_positions[: self.click_index]:
            clicked_sprite = (
                _SPRITES["clicked"].clone().set_position(clicked_pos[0], clicked_pos[1])
            )
            self.current_level.add_sprite(clicked_sprite)

        for sprite_info in snap["unclicked_sprites"]:
            sprite = (
                _SPRITES[sprite_info["name"]]
                .clone()
                .set_position(sprite_info["x"], sprite_info["y"])
            )
            self.current_level.add_sprite(sprite)

        self._update_selector()

    def _find_closest_positions(
        self, remaining_indices: List[int]
    ) -> List[Tuple[int, int]]:
        remaining_positions = [
            (pos, idx)
            for pos, idx in self.position_to_index.items()
            if idx >= self.click_index
        ]
        min_distance = float("inf")
        min_distance_positions: List[Tuple[int, int]] = []

        for pos, idx in remaining_positions:
            dist = math.sqrt(
                (pos[0] - self.center_x) ** 2 + (pos[1] - self.center_y) ** 2
            )
            if dist < min_distance:
                min_distance = dist
                min_distance_positions = [pos]
            elif abs(dist - min_distance) < 0.3:
                min_distance_positions.append(pos)

        if len(min_distance_positions) >= len(remaining_indices) - 1:
            min_distance_positions = min_distance_positions[:1]

        return min_distance_positions

    def _try_find_position(
        self,
        idx: int,
        new_positions: List[Tuple[int, int]],
        used_positions: Set[Tuple[int, int]],
    ) -> Optional[Tuple[int, int]]:
        for _ in range(150):
            angle = (self._rng.random() * 2.0 * math.pi) + (self.move_count * 0.7)

            if self.level_index == 4:
                base_radius = 4.5 + (idx * 2.8)
                radius = base_radius + self._rng.uniform(-0.5, 0.5)
            else:
                base_radius = 3.5 + (idx * 2.0)
                radius = base_radius + self._rng.uniform(-0.7, 0.7)

            x = int(self.center_x + radius * math.cos(angle))
            y = int(self.center_y + radius * math.sin(angle))
            x = max(1, min(18, x))
            y = max(1, min(18, y))

            if (x, y) in used_positions or (x, y) == (self.center_x, self.center_y):
                continue

            min_separation = 2.5 if self.level_index == 4 else 1.5
            too_close = False
            for existing_pos in new_positions:
                dist_to_existing = math.sqrt(
                    (x - existing_pos[0]) ** 2 + (y - existing_pos[1]) ** 2
                )
                if dist_to_existing < min_separation:
                    too_close = True
                    break

            if not too_close:
                return (x, y)
        return None

    def _compute_fallback_position(self, idx: int, total: int) -> Tuple[int, int]:
        angle = (
            (idx * 2.0 * math.pi) / total
            + (self._rng.random() * math.pi)
            + (self.move_count * 0.7)
        )
        if self.level_index == 4:
            radius = 4.5 + (idx * 2.8) + self._rng.uniform(-0.3, 0.3)
        else:
            radius = 3.5 + (idx * 2.0) + self._rng.uniform(-0.3, 0.3)
        x = int(self.center_x + radius * math.cos(angle))
        y = int(self.center_y + radius * math.sin(angle))
        x = max(1, min(18, x))
        y = max(1, min(18, y))
        return (x, y)

    def _generate_new_positions(
        self, remaining_indices: List[int]
    ) -> List[Tuple[int, int]]:
        new_positions: List[Tuple[int, int]] = []
        used_positions: Set[Tuple[int, int]] = set()

        for idx, i in enumerate(remaining_indices):
            pos = self._try_find_position(idx, new_positions, used_positions)
            if pos is not None:
                new_positions.append(pos)
                used_positions.add(pos)
            else:
                fallback = self._compute_fallback_position(idx, len(remaining_indices))
                new_positions.append(fallback)

        return new_positions

    def _place_block_at(self, pos: Tuple[int, int], idx: int) -> None:
        color = self.sequence[idx]
        sprite_key = _COLOR_TO_SPRITE_KEY[color]
        sprite = _SPRITES[sprite_key].clone().set_position(pos[0], pos[1])
        self.current_level.add_sprite(sprite)
        self.position_to_index[pos] = idx
        dist = math.sqrt((pos[0] - self.center_x) ** 2 + (pos[1] - self.center_y) ** 2)
        self.position_to_distance[pos] = dist

    def _apply_repositioned_blocks(
        self,
        remaining_indices: List[int],
        new_positions: List[Tuple[int, int]],
        min_distance_positions: List[Tuple[int, int]],
    ) -> None:
        blocks_to_keep: Dict[int, Tuple[int, int]] = {}
        for pos, idx in list(self.position_to_index.items()):
            if idx >= self.click_index and pos in min_distance_positions:
                blocks_to_keep[idx] = pos

        sprites_to_remove = []
        for sprite in list(self.current_level.get_sprites()):
            tags = sprite.tags or []
            if (
                "center" not in tags
                and "clicked" not in tags
                and "selector" not in tags
            ):
                sprites_to_remove.append(sprite)

        for sprite in sprites_to_remove:
            self.current_level.remove_sprite(sprite)

        self.position_to_index.clear()
        self.position_to_distance.clear()

        repositioned_count = 0
        for i, idx in enumerate(remaining_indices):
            if idx in blocks_to_keep:
                self._place_block_at(blocks_to_keep[idx], idx)
            else:
                self._place_block_at(new_positions[repositioned_count], idx)
                repositioned_count += 1

    def _reposition_remaining_blocks(self) -> None:
        remaining_indices = [
            i for i in range(len(self.sequence)) if i >= self.click_index
        ]

        if not remaining_indices or len(remaining_indices) == 1:
            return

        min_distance_positions = self._find_closest_positions(remaining_indices)
        new_positions = self._generate_new_positions(remaining_indices)
        self._apply_repositioned_blocks(
            remaining_indices, new_positions, min_distance_positions
        )

    def _check_win(self) -> bool:
        return len(self.position_to_index) == 0 and self.click_index >= len(
            self.sequence
        )

    def _deduct_life(self, reason: str):
        self._lives -= 1

        if self._lives == 0:
            self._game_over = True
            self.lose()
        else:
            self._reset_current_level()

    def _reset_current_level(self):
        self.click_index = 0
        self.move_count = 0

        for sprite in list(self.current_level.get_sprites()):
            self.current_level.remove_sprite(sprite)

        center_sprite = (
            _SPRITES["center"].clone().set_position(self.center_x, self.center_y)
        )
        self.current_level.add_sprite(center_sprite)

        self.position_to_index = {}
        self.position_to_distance = {}
        for i, (x, y) in enumerate(self.initial_positions):
            color = self.sequence[i]
            sprite_key = _COLOR_TO_SPRITE_KEY[color]
            sprite = _SPRITES[sprite_key].clone().set_position(x, y)
            self.current_level.add_sprite(sprite)
            self.position_to_index[(x, y)] = i
            self.position_to_distance[(x, y)] = self.initial_distances[i]

        self._update_selector()

    def _handle_valid_click(self, grid_x: int, grid_y: int) -> None:
        current_sprite = self.current_level.get_sprite_at(grid_x, grid_y)
        if current_sprite:
            self.current_level.remove_sprite(current_sprite)

        clicked_sprite = _SPRITES["clicked"].clone().set_position(grid_x, grid_y)
        self.current_level.add_sprite(clicked_sprite)

        del self.position_to_index[(grid_x, grid_y)]
        if (grid_x, grid_y) in self.position_to_distance:
            del self.position_to_distance[(grid_x, grid_y)]

        self.click_index += 1
        self.move_count += 1

        self._update_selector()

        if self._check_win():
            self.next_level()
        else:
            should_reposition = False
            if self.level_index == 2 and self.move_count % 3 == 0:
                should_reposition = True
            elif self.level_index == 3 and self.move_count % 2 == 0:
                should_reposition = True
            elif self.level_index == 4:
                should_reposition = True

            if should_reposition:
                self._reposition_remaining_blocks()
                self._update_selector()

    def _process_click(self, grid_x: int, grid_y: int) -> None:
        sprite_at_pos = self.current_level.get_sprite_at(grid_x, grid_y)
        if sprite_at_pos and "clicked" in (sprite_at_pos.tags or []):
            self._deduct_life("Already clicked block")
            return

        if sprite_at_pos:
            tags = sprite_at_pos.tags or []
            if "center" in tags or "selector" in tags:
                return

        if (grid_x, grid_y) in self.position_to_index:
            clicked_distance = math.sqrt(
                (grid_x - self.center_x) ** 2 + (grid_y - self.center_y) ** 2
            )

            remaining_distances = []
            for pos in self.position_to_index.keys():
                dist = math.sqrt(
                    (pos[0] - self.center_x) ** 2 + (pos[1] - self.center_y) ** 2
                )
                remaining_distances.append(dist)

            min_distance = min(remaining_distances) if remaining_distances else 0

            if abs(clicked_distance - min_distance) <= 0.3:
                self._handle_valid_click(grid_x, grid_y)
            else:
                self._deduct_life("Wrong block selected")

    def _check_move_limit(self) -> bool:
        self._moves_used += 1
        if self._max_moves > 0 and self._moves_used > self._max_moves:
            self._game_over = True
            self.lose()
            return True
        return False

    def _step_direction(self) -> None:
        self._save_state()
        if self._check_move_limit():
            return

        grid_w, grid_h = self.current_level.grid_size
        if self.action.id == GameAction.ACTION1:
            self.selector_y = max(0, self.selector_y - 1)
        elif self.action.id == GameAction.ACTION2:
            self.selector_y = min(grid_h - 1, self.selector_y + 1)
        elif self.action.id == GameAction.ACTION3:
            self.selector_x = max(0, self.selector_x - 1)
        elif self.action.id == GameAction.ACTION4:
            self.selector_x = min(grid_w - 1, self.selector_x + 1)
        self._update_selector()

    def _step_select(self) -> None:
        self._save_state()
        if self._check_move_limit():
            return
        self._process_click(self.selector_x, self.selector_y)

    def step(self) -> None:
        if self._game_over:
            self.complete_action()
            return

        if self.action.id == GameAction.ACTION7:
            self._undo()
            self._check_move_limit()
            self.complete_action()
            return

        if self.action.id in (
            GameAction.ACTION1,
            GameAction.ACTION2,
            GameAction.ACTION3,
            GameAction.ACTION4,
        ):
            self._step_direction()
            self.complete_action()
            return

        if self.action.id == GameAction.ACTION5:
            self._step_select()
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

    _VALID_ACTIONS = list(_ACTION_MAP.keys())

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
        "block_blue": "B",
        "block_red": "R",
        "block_green": "G",
        "block_yellow": "Y",
        "block_gray": "-",
        "block_pink": "P",
        "block_orange": "O",
        "block_cyan": "C",
        "block_brown": "W",
        "clicked": "X",
        "center": "*",
        "selector": "@",
        "empty": ".",
    }

    _TAG_PRIORITY: Dict[str, int] = {
        "empty": -1,
        "block_blue": 1,
        "block_red": 1,
        "block_green": 1,
        "block_yellow": 1,
        "block_gray": 1,
        "block_pink": 1,
        "block_orange": 1,
        "block_cyan": 1,
        "block_brown": 1,
        "clicked": 2,
        "center": 3,
        "selector": 4,
    }

    def __init__(self, seed: int = 0) -> None:
        self._engine = Cc01(seed=seed)
        self._total_turns = 0
        self._last_action_was_reset = False
        self._game_won = False

    def _build_text_obs(self) -> str:
        e = self._engine
        gs = e.current_level.grid_size
        w, h = (gs[0] if gs else 20), (gs[1] if gs else 20)
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
        remaining_moves = max(0, e._max_moves - e._moves_used)
        blocks_left = len(e.position_to_index)
        header = (
            f"Level:{e.level_index + 1} Lives:{e._lives} "
            f"Moves:{remaining_moves}/{e._max_moves} Blocks:{blocks_left}"
        )
        return header.strip() + "\n" + grid_text

    def _build_game_state(
        self, done: bool = False, info: Optional[Dict] = None
    ) -> GameState:
        e = self._engine
        valid_actions = self.get_actions() if not done else None
        return GameState(
            text_observation=self._build_text_obs(),
            image_observation=None,
            valid_actions=valid_actions,
            turn=self._total_turns,
            metadata={
                "total_levels": len(self._engine._levels),
                "level_index": e.level_index,
                "lives": e._lives,
                "max_moves": e._max_moves,
                "moves_used": e._moves_used,
                "blocks_remaining": len(e.position_to_index),
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
        if mode != "rgb_array":
            raise ValueError(f"Unsupported render mode: {mode}")
        e = self._engine
        index_grid = e.camera.render(e.current_level.get_sprites())
        arr = (
            np.array(index_grid, dtype=np.uint8)
            if not isinstance(index_grid, np.ndarray)
            else index_grid
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
        self._env: Optional[PuzzleEnvironment] = None

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
    def _resize_nearest(frame: np.ndarray, h: int, w: int) -> np.ndarray:
        src_h, src_w = frame.shape[0], frame.shape[1]
        row_idx = (np.arange(h) * src_h // h).astype(int)
        col_idx = (np.arange(w) * src_w // w).astype(int)
        return frame[np.ix_(row_idx, col_idx)]

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
