from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import random
import struct
import zlib

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
    metadata: dict = field(default_factory=dict)


@dataclass
class StepResult:
    state: GameState
    reward: float
    done: bool
    info: dict = field(default_factory=dict)


GRID_SIZE = (64, 64)
CAM_W, CAM_H = GRID_SIZE

BG_COLOR = 5
PAD_COLOR = 5
FRAME_COLOR = 4
PIPE_COLOR = 3
CURRENT_FLUID = 10

TARGET_FLUID = 10
TANK_BG = 4
BEACON_GOOD = 14
BEACON_BAD = 8
VALVE_CENTER = 0
LIFE_ON = 8
LIFE_OFF = 4
MAX_LIVES = 3

MOD_COLORS = {
    2: 9,
    3: 11,
    4: 6,
    5: 15,
}

VALVE_COLORS = [12, 7, 10, 6, 15, 9, 13, 8, 11]
LOCKED_COLOR = 3
COOLDOWN_COLOR = 2
LINK_COLOR = 13

LEVEL_DEFS: List[Dict] = [
    {
        "name": "Level 1",
        "max_clicks": 16,
        "chambers": [
            {"name": "A", "mod": 3, "rect": (6, 20, 10, 16)},
            {"name": "B", "mod": 4, "rect": (27, 20, 10, 16)},
            {"name": "C", "mod": 3, "rect": (48, 20, 10, 16)},
        ],
        "current": [0, 3, 1],
        "target_recipe": [3, 1, 0],
        "valves": [
            {"name": "v1", "pos": (14, 51), "delta": [0, 1, 1]},
            {"name": "v2", "pos": (32, 51), "delta": [2, 3, 2]},
            {"name": "v3", "pos": (50, 51), "delta": [2, 0, 2]},
        ],
        "locked_valves": {},
        "decay_interval": 0,
        "linked_pairs": [],
        "cooldown_valves": [],
    },
    {
        "name": "Level 2",
        "max_clicks": 38,
        "chambers": [
            {"name": "A", "mod": 5, "rect": (2, 20, 8, 15)},
            {"name": "B", "mod": 4, "rect": (12, 20, 8, 15)},
            {"name": "C", "mod": 3, "rect": (22, 20, 8, 15)},
            {"name": "D", "mod": 5, "rect": (32, 20, 8, 15)},
            {"name": "E", "mod": 2, "rect": (42, 20, 8, 15)},
            {"name": "F", "mod": 4, "rect": (52, 20, 8, 15)},
        ],
        "current": [3, 3, 2, 2, 1, 3],
        "target_recipe": [7, 2, 1, 0, 0, 0, 0],
        "valves": [
            {"name": "v1", "pos": (6, 51), "delta": [2, 2, 1, 4, 1, 1]},
            {"name": "v2", "pos": (14, 51), "delta": [2, 1, 1, 3, 1, 1]},
            {"name": "v3", "pos": (22, 51), "delta": [0, 3, 2, 2, 0, 1]},
            {"name": "v4", "pos": (30, 51), "delta": [0, 2, 1, 1, 1, 0]},
            {"name": "v5", "pos": (38, 51), "delta": [3, 0, 1, 0, 1, 1]},
            {"name": "v6", "pos": (46, 51), "delta": [0, 1, 1, 1, 1, 3]},
            {"name": "v7", "pos": (54, 51), "delta": [2, 2, 0, 1, 0, 1]},
        ],
        "locked_valves": {},
        "decay_interval": 0,
        "linked_pairs": [[0, 3], [1, 5]],
        "cooldown_valves": [],
    },
    {
        "name": "Level 3",
        "max_clicks": 48,
        "chambers": [
            {"name": "A", "mod": 5, "rect": (2, 16, 10, 12)},
            {"name": "B", "mod": 4, "rect": (18, 16, 10, 12)},
            {"name": "C", "mod": 3, "rect": (34, 16, 10, 12)},
            {"name": "D", "mod": 5, "rect": (50, 16, 10, 12)},
            {"name": "E", "mod": 4, "rect": (2, 34, 10, 12)},
            {"name": "F", "mod": 2, "rect": (18, 34, 10, 12)},
            {"name": "G", "mod": 3, "rect": (34, 34, 10, 12)},
            {"name": "H", "mod": 5, "rect": (50, 34, 10, 12)},
        ],
        "current": [3, 1, 0, 4, 3, 1, 0, 1],
        "target_recipe": [0, 4, 5, 0, 2, 1, 0, 0, 0],
        "valves": [
            {"name": "v1", "pos": (6, 53), "delta": [3, 1, 1, 2, 1, 0, 0, 3]},
            {"name": "v2", "pos": (18, 53), "delta": [1, 2, 0, 2, 1, 0, 0, 2]},
            {"name": "v3", "pos": (30, 53), "delta": [4, 3, 1, 2, 2, 1, 2, 3]},
            {"name": "v4", "pos": (42, 53), "delta": [2, 0, 1, 2, 3, 1, 0, 2]},
            {"name": "v5", "pos": (54, 53), "delta": [2, 3, 1, 0, 1, 1, 1, 1]},
            {"name": "v6", "pos": (12, 59), "delta": [0, 0, 1, 1, 2, 0, 2, 3]},
            {"name": "v7", "pos": (24, 59), "delta": [0, 2, 0, 4, 3, 1, 1, 1]},
            {"name": "v8", "pos": (36, 59), "delta": [2, 2, 0, 3, 0, 0, 0, 2]},
            {"name": "v9", "pos": (48, 59), "delta": [2, 1, 1, 2, 2, 0, 1, 1]},
        ],
        "locked_valves": {},
        "decay_interval": 0,
        "linked_pairs": [],
        "cooldown_valves": [0, 2, 4, 6, 8],
    },
    {
        "name": "Level 4",
        "max_clicks": 52,
        "chambers": [
            {"name": "A", "mod": 5, "rect": (1, 14, 10, 10)},
            {"name": "B", "mod": 4, "rect": (13, 14, 10, 10)},
            {"name": "C", "mod": 3, "rect": (25, 14, 10, 10)},
            {"name": "D", "mod": 5, "rect": (37, 14, 10, 10)},
            {"name": "E", "mod": 4, "rect": (49, 14, 10, 10)},
            {"name": "F", "mod": 3, "rect": (1, 30, 10, 10)},
            {"name": "G", "mod": 5, "rect": (13, 30, 10, 10)},
            {"name": "H", "mod": 2, "rect": (25, 30, 10, 10)},
            {"name": "I", "mod": 4, "rect": (37, 30, 10, 10)},
            {"name": "J", "mod": 3, "rect": (49, 30, 10, 10)},
        ],
        "current": [4, 2, 1, 3, 0, 2, 1, 1, 3, 2],
        "target_recipe": [2, 3, 4, 0, 1, 2, 0, 0, 1, 0, 1],
        "valves": [
            {"name": "v1", "pos": (5, 48), "delta": [3, 1, 2, 4, 1, 0, 2, 1, 3, 1]},
            {"name": "v2", "pos": (11, 48), "delta": [1, 2, 0, 2, 3, 1, 0, 1, 2, 2]},
            {"name": "v3", "pos": (17, 48), "delta": [4, 3, 1, 1, 2, 2, 3, 0, 1, 1]},
            {"name": "v4", "pos": (23, 48), "delta": [2, 0, 1, 3, 1, 1, 4, 1, 0, 2]},
            {"name": "v5", "pos": (29, 48), "delta": [0, 1, 2, 2, 3, 0, 1, 1, 2, 0]},
            {"name": "v6", "pos": (35, 48), "delta": [1, 3, 0, 1, 0, 2, 2, 0, 3, 1]},
            {"name": "v7", "pos": (41, 48), "delta": [3, 2, 1, 0, 2, 1, 1, 1, 1, 2]},
            {"name": "v8", "pos": (47, 48), "delta": [2, 1, 2, 4, 0, 0, 3, 1, 2, 0]},
            {"name": "v9", "pos": (53, 48), "delta": [0, 3, 1, 2, 1, 2, 0, 0, 1, 1]},
            {"name": "v10", "pos": (17, 55), "delta": [1, 0, 2, 1, 2, 1, 2, 1, 0, 2]},
            {"name": "v11", "pos": (41, 55), "delta": [4, 1, 0, 3, 3, 0, 1, 0, 2, 1]},
        ],
        "locked_valves": {8: 4, 9: 4, 10: 6},
        "decay_interval": 8,
        "linked_pairs": [[0, 5], [4, 9]],
        "cooldown_valves": [2, 4, 6],
    },
]


def _apply_delta(state: List[int], delta: List[int], chambers: List[Dict]) -> List[int]:
    return [
        (value + step) % chamber["mod"]
        for value, step, chamber in zip(state, delta, chambers)
    ]


def _compute_target(start: List[int], level_def: Dict) -> List[int]:
    state = list(start)
    for valve_idx, count in enumerate(level_def["target_recipe"]):
        delta = level_def["valves"][valve_idx]["delta"]
        for _ in range(count):
            state = _apply_delta(state, delta, level_def["chambers"])
    return state


class Cg01(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)
        self._current: List[int] = []
        self._target: List[int] = []
        self._clicks_used = 0
        self._max_clicks = 0
        self._valve_lookup: Dict[Tuple[int, int], int] = {}
        self._out_of_moves = False
        self._selected_valve = 0
        self._history: List[Tuple] = []
        self._level_cleared = False
        self._lives = MAX_LIVES
        self._preserve_lives_on_next_set_level = False
        self._valve_locked: List[bool] = []
        self._valve_cooldown: List[bool] = []
        self._last_valve_used = -1
        self._valve_presses = 0
        self._game_over = False

        game_levels = [
            Level(sprites=[], grid_size=GRID_SIZE, name=ld["name"])
            for ld in LEVEL_DEFS
        ]

        super().__init__(
            "cg01",
            game_levels,
            Camera(
                x=0,
                y=0,
                width=CAM_W,
                height=CAM_H,
                background=BG_COLOR,
                letter_box=PAD_COLOR,
                interfaces=[],
            ),
            available_actions=[0, 1, 2, 3, 4, 5, 6, 7],
        )

    def on_set_level(self, level: Level) -> None:
        level_def = LEVEL_DEFS[self.level_index]
        if not self._preserve_lives_on_next_set_level:
            self._lives = MAX_LIVES
        self._preserve_lives_on_next_set_level = False
        while True:
            start = [
                self._rng.randint(0, chamber["mod"] - 1)
                for chamber in level_def["chambers"]
            ]
            target = _compute_target(start, level_def)
            if start != target:
                break
        self._current = start
        self._target = target
        self._clicks_used = 0
        self._max_clicks = level_def["max_clicks"]
        self._out_of_moves = False
        self._selected_valve = self._rng.randint(0, len(level_def["valves"]) - 1)
        self._history = []
        self._level_cleared = False
        self._valve_presses = 0
        self._last_valve_used = -1
        self._game_over = False

        valve_count = len(level_def["valves"])
        locked_valves = level_def.get("locked_valves", {})
        self._valve_locked = [
            idx in locked_valves for idx in range(valve_count)
        ]
        self._valve_cooldown = [False] * valve_count

        self._valve_lookup = {}
        for idx, valve in enumerate(level_def["valves"]):
            vx, vy = valve["pos"]
            for py in range(vy - 1, vy + 2):
                for px in range(vx - 1, vx + 2):
                    if 0 <= px < CAM_W and 0 <= py < CAM_H:
                        self._valve_lookup[(px, py)] = idx
        self._rebuild(level)

    def _rebuild(self, level: Level) -> None:
        level.remove_all_sprites()
        self._draw_frame(level)
        self._draw_chambers(level)
        self._draw_linked_indicators(level)
        self._draw_valves(level)
        self._draw_cursor(level)
        self._draw_lives(level)
        self._draw_progress(level)
        self._draw_budget(level)

    def _draw_frame(self, level: Level) -> None:
        frame_color = BEACON_GOOD if self._level_cleared else FRAME_COLOR
        level.add_sprite(Sprite(pixels=[[frame_color] * CAM_W], name="top", x=0, y=0, layer=0))
        level.add_sprite(
            Sprite(pixels=[[frame_color] * CAM_W], name="bottom", x=0, y=CAM_H - 1, layer=0)
        )
        level.add_sprite(
            Sprite(pixels=[[frame_color] for _ in range(CAM_H)], name="left", x=0, y=0, layer=0)
        )
        level.add_sprite(
            Sprite(
                pixels=[[frame_color] for _ in range(CAM_H)],
                name="right",
                x=CAM_W - 1,
                y=0,
                layer=0,
            )
        )

    def _tank_border_pixels(self, width: int, height: int, color: int, matched: bool) -> List[List[int]]:
        pixels = [[color] * width for _ in range(height)]
        if matched and height > 0:
            pixels[-1] = [BEACON_GOOD] * width
        return pixels

    def _segment_pixels(self, width: int, height: int, modulus: int, value: int, color: int) -> List[List[int]]:
        pixels = [[TANK_BG] * width for _ in range(height)]
        if height <= 0 or width <= 0:
            return pixels
        if modulus <= 1:
            return pixels

        segment_count = modulus - 1
        remaining = height
        heights = []
        for seg in range(segment_count):
            seg_h = remaining // (segment_count - seg)
            heights.append(seg_h)
            remaining -= seg_h

        y = height
        for seg_idx, seg_h in enumerate(reversed(heights), start=1):
            y -= seg_h
            if seg_h <= 0:
                continue
            fill_color = color if seg_idx <= value else TANK_BG
            for row in range(y, y + seg_h):
                for col in range(width):
                    pixels[row][col] = fill_color
            if y > 0:
                for col in range(width):
                    pixels[y - 1][col] = PIPE_COLOR
        return pixels

    def _draw_chambers(self, level: Level) -> None:
        level_def = LEVEL_DEFS[self.level_index]
        for idx, chamber in enumerate(level_def["chambers"]):
            x, y, w, h = chamber["rect"]
            modulus = chamber["mod"]
            current_value = self._current[idx]
            target_value = self._target[idx]
            is_matched = current_value == target_value
            border_color = MOD_COLORS[modulus]

            preview_w = max(4, w - 2)
            preview_h = 6
            preview_x = x + (w - preview_w) // 2
            preview_y = y - 8
            level.add_sprite(
                Sprite(
                    pixels=[[border_color] * preview_w for _ in range(preview_h)],
                    name=f"preview_border_{idx}",
                    x=preview_x,
                    y=preview_y,
                    layer=1,
                    tags=["preview"],
                )
            )
            level.add_sprite(
                Sprite(
                    pixels=self._segment_pixels(preview_w - 2, preview_h - 2, modulus, target_value, TARGET_FLUID),
                    name=f"preview_fill_{idx}",
                    x=preview_x + 1,
                    y=preview_y + 1,
                    layer=2,
                    tags=["preview"],
                )
            )

            pipe_h = max(1, y - (preview_y + preview_h))
            level.add_sprite(
                Sprite(
                    pixels=[[PIPE_COLOR] for _ in range(pipe_h)],
                    name=f"pipe_{idx}",
                    x=x + (w // 2),
                    y=preview_y + preview_h,
                    layer=1,
                    tags=["pipe"],
                )
            )

            level.add_sprite(
                Sprite(
                    pixels=self._tank_border_pixels(w, h, border_color, is_matched),
                    name=f"tank_border_{idx}",
                    x=x,
                    y=y,
                    layer=1,
                    tags=["tank"],
                )
            )
            level.add_sprite(
                Sprite(
                    pixels=self._segment_pixels(w - 2, h - 2, modulus, current_value, CURRENT_FLUID),
                    name=f"tank_fill_{idx}",
                    x=x + 1,
                    y=y + 1,
                    layer=2,
                    tags=["tank"],
                )
            )

            beacon = BEACON_GOOD if is_matched else BEACON_BAD
            level.add_sprite(
                Sprite(
                    pixels=[[beacon] * 3 for _ in range(3)],
                    name=f"beacon_{idx}",
                    x=x + (w // 2) - 1,
                    y=y - 4,
                    layer=3,
                    tags=["beacon"],
                )
            )

    def _draw_linked_indicators(self, level: Level) -> None:
        level_def = LEVEL_DEFS[self.level_index]
        linked_pairs = level_def.get("linked_pairs", [])
        if not linked_pairs:
            return
        for pair_idx, pair in enumerate(linked_pairs):
            chamber_a = level_def["chambers"][pair[0]]
            chamber_b = level_def["chambers"][pair[1]]
            ax, ay, aw, ah = chamber_a["rect"]
            bx, by, bw, bh = chamber_b["rect"]
            mid_x = (ax + aw // 2 + bx + bw // 2) // 2
            mid_y = min(ay, by) - 2
            level.add_sprite(
                Sprite(
                    pixels=[[LINK_COLOR] * 3 for _ in range(2)],
                    name=f"link_{pair_idx}",
                    x=mid_x - 1,
                    y=mid_y,
                    layer=3,
                    tags=["link"],
                )
            )

    def _draw_valves(self, level: Level) -> None:
        level_def = LEVEL_DEFS[self.level_index]
        for idx, valve in enumerate(level_def["valves"]):
            vx, vy = valve["pos"]
            if self._valve_locked[idx]:
                color = LOCKED_COLOR
            elif self._valve_cooldown[idx]:
                color = COOLDOWN_COLOR
            else:
                color = VALVE_COLORS[idx % len(VALVE_COLORS)]
            level.add_sprite(
                Sprite(
                    pixels=[[color] * 3 for _ in range(3)],
                    name=f"valve_{idx}",
                    x=vx - 1,
                    y=vy - 1,
                    layer=3,
                    tags=["valve"],
                )
            )
            center = LOCKED_COLOR if self._valve_locked[idx] else VALVE_CENTER
            level.add_sprite(
                Sprite(
                    pixels=[[center]],
                    name=f"valve_center_{idx}",
                    x=vx,
                    y=vy,
                    layer=4,
                    tags=["valve"],
                )
            )

    def _draw_cursor(self, level: Level) -> None:
        level_def = LEVEL_DEFS[self.level_index]
        if not level_def["valves"]:
            return
        vx, vy = level_def["valves"][self._selected_valve]["pos"]
        pixels = []
        for row in range(5):
            line = []
            for col in range(5):
                if row in (0, 4) or col in (0, 4):
                    line.append(14)
                else:
                    line.append(-1)
            pixels.append(line)
        level.add_sprite(
            Sprite(
                pixels=pixels,
                name="cursor",
                x=vx - 2,
                y=vy - 2,
                layer=5,
                tags=["cursor"],
            )
        )

    def _draw_budget(self, level: Level) -> None:
        width = 56
        x = 4
        y = 62
        remaining = max(0, self._max_clicks - self._clicks_used)
        filled = (remaining * width) // max(1, self._max_clicks)
        if filled > 0:
            level.add_sprite(
                Sprite(
                    pixels=[[12] * filled],
                    name="budget_full",
                    x=x,
                    y=y,
                    layer=2,
                    tags=["budget"],
                )
            )
        if filled < width:
            level.add_sprite(
                Sprite(
                    pixels=[[4] * (width - filled)],
                    name="budget_empty",
                    x=x + filled,
                    y=y,
                    layer=2,
                    tags=["budget"],
                )
            )

    def _draw_progress(self, level: Level) -> None:
        width = 56
        x = 4
        y = 60
        chamber_count = max(1, len(self._current))
        matched = sum(
            1 for current_value, target_value in zip(self._current, self._target)
            if current_value == target_value
        )
        if self._level_cleared:
            matched = chamber_count
        filled = (matched * width) // chamber_count
        if filled > 0:
            level.add_sprite(
                Sprite(
                    pixels=[[BEACON_GOOD] * filled],
                    name="progress_full",
                    x=x,
                    y=y,
                    layer=2,
                    tags=["progress"],
                )
            )
        if filled < width:
            level.add_sprite(
                Sprite(
                    pixels=[[4] * (width - filled)],
                    name="progress_empty",
                    x=x + filled,
                    y=y,
                    layer=2,
                    tags=["progress"],
                )
            )

    def _draw_lives(self, level: Level) -> None:
        life_size = 3
        gap = 2
        total_width = MAX_LIVES * life_size + (MAX_LIVES - 1) * gap
        start_x = CAM_W - total_width - 4
        y = 3

        for idx in range(MAX_LIVES):
            color = LIFE_ON if idx < self._lives else LIFE_OFF
            level.add_sprite(
                Sprite(
                    pixels=[[color] * life_size for _ in range(life_size)],
                    name=f"life_{idx}",
                    x=start_x + idx * (life_size + gap),
                    y=y,
                    layer=6,
                    tags=["life"],
                )
            )

    def _is_valve_usable(self, valve_idx: int) -> bool:
        if self._valve_locked[valve_idx]:
            return False
        if self._valve_cooldown[valve_idx]:
            return False
        return True

    def _save_snapshot(self) -> None:
        self._history.append((
            self._current[:],
            self._clicks_used,
            self._valve_locked[:],
            self._valve_cooldown[:],
            self._valve_presses,
            self._last_valve_used,
            self._selected_valve,
        ))

    def _apply_valve(self, valve_idx: int) -> bool:
        if not self._is_valve_usable(valve_idx):
            return False
        level_def = LEVEL_DEFS[self.level_index]
        valve = level_def["valves"][valve_idx]
        next_state = []
        for value, delta, chamber in zip(self._current, valve["delta"], level_def["chambers"]):
            next_state.append((value + delta) % chamber["mod"])
        if next_state == self._current:
            return False
        self._current = next_state
        self._clicks_used += 1
        self._valve_presses += 1

        self._apply_linked_pairs(level_def, valve_idx)
        self._apply_decay(level_def)
        self._update_locked_valves(level_def)
        self._update_cooldowns(level_def, valve_idx)

        return True

    def _apply_linked_pairs(self, level_def: Dict, valve_idx: int) -> None:
        linked_pairs = level_def.get("linked_pairs", [])
        if not linked_pairs:
            return
        affected = set()
        valve = level_def["valves"][valve_idx]
        for pair in linked_pairs:
            for chamber_idx in pair:
                if valve["delta"][chamber_idx] != 0:
                    affected.update(pair)
        for chamber_idx in affected:
            chamber = level_def["chambers"][chamber_idx]
            self._current[chamber_idx] = (self._current[chamber_idx] + 1) % chamber["mod"]

    def _apply_decay(self, level_def: Dict) -> None:
        decay_interval = level_def.get("decay_interval", 0)
        if decay_interval <= 0:
            return
        if self._valve_presses % decay_interval != 0:
            return
        for idx, chamber in enumerate(level_def["chambers"]):
            self._current[idx] = (self._current[idx] - 1) % chamber["mod"]

    def _update_locked_valves(self, level_def: Dict) -> None:
        locked_valves = level_def.get("locked_valves", {})
        for valve_idx, unlock_after in locked_valves.items():
            if self._valve_presses >= unlock_after:
                self._valve_locked[valve_idx] = False

    def _update_cooldowns(self, level_def: Dict, used_valve_idx: int) -> None:
        cooldown_set = set(level_def.get("cooldown_valves", []))
        if not cooldown_set:
            return
        for idx in range(len(self._valve_cooldown)):
            self._valve_cooldown[idx] = False
        if used_valve_idx in cooldown_set:
            self._valve_cooldown[used_valve_idx] = True
        self._last_valve_used = used_valve_idx

    def _undo(self) -> bool:
        if not self._history:
            return False
        snapshot = self._history.pop()
        self._current = snapshot[0]
        self._clicks_used = self._clicks_used + 1
        self._valve_locked = snapshot[2]
        self._valve_cooldown = snapshot[3]
        self._valve_presses = snapshot[4]
        self._last_valve_used = snapshot[5]
        self._selected_valve = snapshot[6]
        self._out_of_moves = False
        return True

    def _move_selection(self, direction: str) -> bool:
        level_def = LEVEL_DEFS[self.level_index]
        valves = level_def["valves"]
        if not valves:
            return False

        cur_x, cur_y = valves[self._selected_valve]["pos"]
        best_idx = self._selected_valve
        best_key = None

        for idx, valve in enumerate(valves):
            if idx == self._selected_valve:
                continue
            x, y = valve["pos"]
            dx = x - cur_x
            dy = y - cur_y

            if direction == "up" and dy >= 0:
                continue
            if direction == "down" and dy <= 0:
                continue
            if direction == "left" and dx >= 0:
                continue
            if direction == "right" and dx <= 0:
                continue

            if direction in ("up", "down"):
                key = (abs(dy), abs(dx), y, x)
            else:
                key = (abs(dx), abs(dy), x, y)

            if best_key is None or key < best_key:
                best_key = key
                best_idx = idx

        if best_idx != self._selected_valve:
            self._selected_valve = best_idx
            self._clicks_used += 1
            return True
        self._clicks_used += 1
        return True

    def _is_solved(self) -> bool:
        return self._current == self._target

    def _restart_current_level_after_life_loss(self) -> None:
        self._out_of_moves = False
        self._preserve_lives_on_next_set_level = True
        self.set_level(self.level_index)

    def _trigger_failed_attempt(self, level: Level) -> None:
        self._lives -= 1
        if self._lives > 0:
            self._out_of_moves = True
            self._rebuild(level)
            return
        self._game_over = True
        self._rebuild(level)
        self.lose()

    def _reset_level(self) -> None:
        self._out_of_moves = False
        self.set_level(self.level_index)

    def step(self) -> None:
        action = self.action.id

        if action == GameAction.RESET:
            self._reset_level()
            self.complete_action()
            return

        if self._out_of_moves:
            self._restart_current_level_after_life_loss()
            self.complete_action()
            return

        if action in (GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4):
            direction_map = {
                GameAction.ACTION1: "up",
                GameAction.ACTION2: "down",
                GameAction.ACTION3: "left",
                GameAction.ACTION4: "right",
            }
            self._save_snapshot()
            self._move_selection(direction_map[action])
            self._rebuild(self.current_level)
            if self._clicks_used >= self._max_clicks:
                self._trigger_failed_attempt(self.current_level)
            self.complete_action()
            return

        if action == GameAction.ACTION5:
            self._save_snapshot()
            acted = self._apply_valve(self._selected_valve)
            if not acted:
                self._history.pop()
                self.complete_action()
                return
            self._post_valve_check()
            self.complete_action()
            return

        if action == GameAction.ACTION6:
            display_x = self.action.data.get("x")
            display_y = self.action.data.get("y")
            if display_x is None or display_y is None:
                self.complete_action()
                return

            grid_coords = self.camera.display_to_grid(display_x, display_y)
            if grid_coords is None:
                self.complete_action()
                return

            valve_idx = self._valve_lookup.get((grid_coords[0], grid_coords[1]))
            if valve_idx is None:
                self.complete_action()
                return

            self._save_snapshot()
            self._selected_valve = valve_idx
            acted = self._apply_valve(valve_idx)
            if not acted:
                snap = self._history.pop()
                self._selected_valve = snap[6]
                self.complete_action()
                return
            self._post_valve_check()
            self.complete_action()
            return

        if action == GameAction.ACTION7:
            self._undo()
            self._rebuild(self.current_level)
            if self._clicks_used >= self._max_clicks:
                self._trigger_failed_attempt(self.current_level)
            self.complete_action()
            return

        self.complete_action()

    def _post_valve_check(self) -> None:
        level = self.current_level
        self._rebuild(level)

        if self._is_solved():
            self._level_cleared = True
            if self.is_last_level():
                self._rebuild(level)
                self.win()
            else:
                self.next_level()
            return

        if self._clicks_used >= self._max_clicks:
            self._trigger_failed_attempt(level)


ARC_PALETTE = [
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
]


def _png_chunk(chunk_type, data):
    chunk = chunk_type + data
    return (
        struct.pack(">I", len(data))
        + chunk
        + struct.pack(">I", zlib.crc32(chunk) & 0xFFFFFFFF)
    )


def _encode_png(rgb):
    h, w = rgb.shape[0], rgb.shape[1]
    raw = b""
    for y in range(h):
        raw += b"\x00" + rgb[y].tobytes()
    compressed = zlib.compress(raw)
    out = b"\x89PNG\r\n\x1a\n"
    out += _png_chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    out += _png_chunk(b"IDAT", compressed)
    out += _png_chunk(b"IEND", b"")
    return out


class PuzzleEnvironment:
    _ACTION_MAP: Dict[str, GameAction] = {
        "reset": GameAction.RESET,
        "up": GameAction.ACTION1,
        "down": GameAction.ACTION2,
        "left": GameAction.ACTION3,
        "right": GameAction.ACTION4,
        "select": GameAction.ACTION5,
        "click": GameAction.ACTION6,
        "undo": GameAction.ACTION7,
    }

    _VALID_ACTIONS = list(_ACTION_MAP.keys())

    def __init__(self, seed: int = 0) -> None:
        self._engine = Cg01(seed=seed)
        self._seed = seed
        self._total_turns = 0
        self._last_action_was_reset = False
        self._done = False
        self._game_won = False

    def reset(self) -> GameState:
        e = self._engine
        e._rng = random.Random(self._seed)

        if self._game_won or self._last_action_was_reset:
            e.set_level(0)
        else:
            e.set_level(e.level_index)

        self._total_turns = 0
        self._done = False
        self._last_action_was_reset = True
        self._game_won = False
        return self._build_game_state()

    def step(self, action: str) -> StepResult:
        e = self._engine

        if action == "reset":
            state = self.reset()
            return StepResult(
                state=state, reward=0.0, done=False, info={"action": "reset"}
            )

        if action not in self._ACTION_MAP and not action.startswith("click"):
            raise ValueError(
                f"Invalid action '{action}'. Must be one of {list(self._ACTION_MAP.keys())}"
            )

        self._last_action_was_reset = False
        self._total_turns += 1

        click_data: Dict[str, Any] = {}
        base_action = action
        if action.startswith("click"):
            parts = action.split()
            if len(parts) == 3:
                click_data = {"x": int(parts[1]), "y": int(parts[2])}
            else:
                click_data = {"x": CAM_W // 2, "y": CAM_H // 2}
            base_action = "click"

        game_action = self._ACTION_MAP[base_action]
        info: Dict = {"action": action}

        level_before = e.level_index

        if click_data:
            action_input = ActionInput(id=game_action, data=click_data)
        else:
            action_input = ActionInput(id=game_action)
        frame = e.perform_action(action_input, raw=True)

        state_name = frame.state.name if frame and frame.state else ""
        game_won = state_name == "WIN"
        game_over = state_name == "GAME_OVER"

        total_levels = len(e._levels)
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

    def get_actions(self) -> List[str]:
        if self._done:
            return ["reset"]
        return list(self._VALID_ACTIONS)

    def is_done(self) -> bool:
        return self._done

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

    def _build_text_obs(self) -> str:
        e = self._engine
        level_def = LEVEL_DEFS[e.level_index]
        lines = []
        remaining = max(0, e._max_clicks - e._clicks_used)
        lines.append(
            f"Level:{e.level_index + 1}/{len(e._levels)} "
            f"Lives:{e._lives} Budget:{remaining}/{e._max_clicks}"
        )
        for idx, chamber in enumerate(level_def["chambers"]):
            cur = e._current[idx]
            tgt = e._target[idx]
            mod = chamber["mod"]
            match = "OK" if cur == tgt else ".."
            lines.append(f"  {chamber['name']}:cur={cur} tgt={tgt} mod={mod} [{match}]")
        valve_parts = []
        for idx, valve in enumerate(level_def["valves"]):
            locked = "L" if e._valve_locked[idx] else ""
            cool = "C" if e._valve_cooldown[idx] else ""
            sel = "*" if idx == e._selected_valve else ""
            valve_parts.append(f"{valve['name']}{sel}{locked}{cool}")
        lines.append("Valves: " + " ".join(valve_parts))
        if e._out_of_moves:
            lines.append("[OUT OF MOVES - press any key to restart]")
        if e._game_over:
            lines.append("[GAME OVER]")
        return "\n".join(lines)

    def _build_game_state(self, done: bool = False) -> GameState:
        valid_actions = self.get_actions() if not done else None

        try:
            rgb = self.render()
            image_bytes = _encode_png(rgb)
        except Exception:
            image_bytes = None

        return GameState(
            text_observation=self._build_text_obs(),
            image_observation=image_bytes,
            valid_actions=valid_actions,
            turn=self._total_turns,
            metadata={
                "total_levels": len(self._engine._levels),
                "level_index": self._engine.level_index,
                "levels_completed": getattr(self._engine, "_score", 0),
                "game_over": getattr(
                    getattr(self._engine, "_state", None), "name", ""
                )
                == "GAME_OVER",
                "done": done,
                "info": {},
            },
        )


class ArcGameEnv(gym.Env):
    metadata: Dict[str, Any] = {
        "render_modes": ["rgb_array"],
        "render_fps": 1,
    }

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
