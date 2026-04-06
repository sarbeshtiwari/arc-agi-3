import copy
import zlib
import struct
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import random

import gymnasium as gym
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


CAM_W = 64
CAM_H = 64

WHITE = 0
LIGHT = 2
GRAY = 3
BLACK = 5
RED = 8
BLUE = 9
YELLOW = 11
ORANGE = 12
GREEN = 14
MAGENTA = 6
CYAN = 15
PURPLE = 7

COLOR_MAP = {
    'red': RED,
    'blue': BLUE,
    'yellow': YELLOW,
    'orange': ORANGE,
    'green': GREEN,
    'purple': PURPLE,
    'cyan': CYAN,
    'pink': MAGENTA,
    'key': CYAN,
    'lock': BLACK
}

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


LEVEL_DATA = [
    {
        "level": 1,
        "name": "Introduction",
        "description": "Learn the basics - sort the colors!",
        "buckets": [
            {"capacity": 8, "balls": ["pink", "blue", "pink", "blue", "pink", "blue", "pink", "blue"]},
            {"capacity": 8, "balls": ["blue", "pink", "blue", "pink", "blue", "pink", "blue", "pink"]},
            {"capacity": 8, "balls": ["pink", "blue", "pink", "blue", "pink", "blue", "pink", "blue"]},
            {"capacity": 8, "balls": ["blue", "pink", "blue", "pink", "blue", "pink", "blue", "pink"]},
            {"capacity": 8, "balls": []}
        ],
        "special_rules": {
            "target_colors": {0: "blue", 1: "pink", 2: "blue", 3: "pink"},
            "must_be_empty": [4]
        },
        "move_limit": 600
    },
    {
        "level": 2,
        "name": "Mirror Match",
        "description": "Reverse the top 4 to mirror the bottom 4! Move unused colors to C4",
        "buckets": [
            {"capacity": 8, "balls": ["blue", "red", "yellow", "pink",  "cyan", "yellow", "pink", "red"], "fixed_balls": [0, 1, 2, 3]},
            {"capacity": 8, "balls": ["red", "yellow", "pink", "blue",  "green", "blue", "yellow", "pink"], "fixed_balls": [0, 1, 2, 3]},
            {"capacity": 8, "balls": ["yellow", "pink", "blue", "red",  "orange", "red", "blue", "pink"], "fixed_balls": [0, 1, 2, 3]},
            {"capacity": 8, "balls": ["pink", "blue", "red", "yellow",  "purple", "yellow", "red", "blue"], "fixed_balls": [0, 1, 2, 3]},
            {"capacity": 8, "balls": []},
            {"capacity": 8, "balls": ["blue", "pink", "yellow", "red"]}
        ],
        "special_rules": {
            "mirror_matching": True,
            "match_containers": [0, 1, 2, 3],
            "mirror_line_position": 3,
            "unused_colors": ["cyan", "green", "orange", "purple"],
            "unused_container": 4
        },
        "move_limit": 700
    },
    {
        "level": 3,
        "name": "Joint Blocks",
        "description": "Same colors stick together and move as one!",
        "buckets": [
            {"capacity": 10, "balls": ["red", "yellow", "blue", "yellow", "red", "blue", "red", "yellow"]},
            {"capacity": 10, "balls": ["blue", "red", "yellow", "red", "blue", "yellow", "blue", "red"]},
            {"capacity": 10, "balls": ["yellow", "blue", "red", "blue", "yellow", "red", "yellow", "blue"]},
            {"capacity": 10, "balls": ["red", "blue", "yellow", "blue", "red", "yellow"]},
        ],
        "special_rules": {
            "joint_blocks": True
        },
        "move_limit": 800
    },
    {
        "level": 4,
        "name": "Key & Lock",
        "description": "Find hidden cyan keys! Drop key on purple lock to unlock! Sort colors!",
        "buckets": [
            {"capacity": 6, "balls": ["red", "blue", "key", "green", "red", "lock"]},
            {"capacity": 6, "balls": ["blue", "red", "key", "green", "blue", "lock"]},
            {"capacity": 6, "balls": ["green", "key", "red", "blue", "green", "green"]},
            {"capacity": 6, "balls": ["red", "blue", "red", "blue", "green", "lock"]},
            {"capacity": 6, "balls": []},
            {"capacity": 6, "balls": []},
        ],
        "special_rules": {
            "key_lock": True,
        },
        "move_limit": 800
    }
]

LEVELS = [
    Level(
        sprites=[], 
        grid_size=(CAM_W, CAM_H), 
        name=level_data["name"], 
        data=level_data
    )
    for level_data in LEVEL_DATA
]

class Bb77(ARCBaseGame):
    
    def __init__(self, seed: int = 0) -> None:
        self.MAX_LIVES = 3
        self._current_level_idx = 0
        self.buckets = []
        self.moves = 0
        self.locked_buckets = set()
        self._game_over = False
        self._level_done = False
        self.won = False
        self.special_rules = {}
        self.move_limit = None
        self._lives = self.MAX_LIVES
        self.keys_collected = 0
        self.cursor_pos = 0
        self.holding_ball = None
        self.gravity_normal = True
        self._rng = random.Random(seed)
        self._history = []
        self._last_level_won = False
        self._consecutive_resets = 0
        self._actions_since_reset = 0
        
        camera = Camera(
            x=0,
            y=0,
            width=CAM_W,
            height=CAM_H,
            background=LIGHT,
            letter_box=WHITE,
            interfaces=[],
        )
        
        super().__init__("bb77", LEVELS, camera, available_actions=[0, 2, 3, 4, 5, 6, 7])
    
    def on_set_level(self, level: Level):
        self._history = []
        self._level_done = False
        self._last_level_won = False
        for i, lv in enumerate(LEVELS):
            if lv.name == level.name:
                self._current_level_idx = i
                self.load_level_data(LEVEL_DATA[i])
                break
    
    def load_level_data(self, level_data: dict, restore_lives: bool = True):
        self.buckets = []
        for bucket_data in level_data["buckets"]:
            bucket = {
                "capacity": bucket_data["capacity"],
                "balls": bucket_data["balls"].copy(),
                "lock_after_receive": bucket_data.get("lock_after_receive", False),
                "hidden_balls": bucket_data.get("hidden_balls", []).copy() if "hidden_balls" in bucket_data else [],
                "chain_reaction": bucket_data.get("chain_reaction", False),
                "fixed_balls": bucket_data.get("fixed_balls", []).copy() if "fixed_balls" in bucket_data else []
            }
            self.buckets.append(bucket)
        
        self.special_rules = level_data["special_rules"]
        self.move_limit = level_data["move_limit"]
        self.moves = 0

        if restore_lives:
            self._lives = self.MAX_LIVES

        self.keys_collected = 0
        self.locked_buckets = set()
        self._game_over = False
        self.won = False
        self.cursor_pos = 0
        self.holding_ball = None
        self.gravity_normal = True
        self._history = []
        self._actions_since_reset = 0
        self._randomize_positions()
        self._rebuild()
    
    def _randomize_positions(self) -> None:
        special_balls = {"key", "lock"}
        movable_balls = []
        slots = []
        for bi, bucket in enumerate(self.buckets):
            fixed = set(bucket.get("fixed_balls", []))
            for si, ball in enumerate(bucket["balls"]):
                if si in fixed:
                    continue
                if ball in special_balls:
                    continue
                movable_balls.append(ball)
                slots.append((bi, si))
        if not movable_balls:
            return
        self._rng.shuffle(movable_balls)
        for idx, (bi, si) in enumerate(slots):
            self.buckets[bi]["balls"][si] = movable_balls[idx]
    
    def step(self) -> None:
        if self.action.id == GameAction.RESET:
            self.complete_action()
            return

        if self._game_over:
            self.lose()
            self.complete_action()
            return
        
        aid = self.action.id

        if aid == GameAction.ACTION7:
            if self._history:
                snapshot = self._history.pop()
                self.buckets = snapshot["buckets"]
                self.cursor_pos = snapshot["cursor_pos"]
                self.holding_ball = snapshot["holding_ball"]
                self.locked_buckets = snapshot["locked_buckets"]
                self.keys_collected = snapshot["keys_collected"]
                self.gravity_normal = snapshot["gravity_normal"]
                self._rebuild()
            self.moves += 1
            self._actions_since_reset += 1
            if self.moves >= (self.move_limit or float("inf")):
                self._handle_death()
            if self._game_over:
                self.lose()
            self.complete_action()
            return

        if aid in (GameAction.ACTION2, GameAction.ACTION3,
                   GameAction.ACTION4, GameAction.ACTION5, GameAction.ACTION6):
            self._history.append(self._take_snapshot())

        if aid == GameAction.ACTION2:
            self._handle_place()
        elif aid == GameAction.ACTION3:
            self.cursor_pos = max(0, self.cursor_pos - 1)
        elif aid == GameAction.ACTION4:
            self.cursor_pos = min(len(self.buckets) - 1, self.cursor_pos + 1)
        elif aid == GameAction.ACTION5:
            self._handle_pick()
        elif aid == GameAction.ACTION6:
            if self.holding_ball:
                self._handle_place()

        if aid in (GameAction.ACTION2, GameAction.ACTION3,
                   GameAction.ACTION4, GameAction.ACTION5, GameAction.ACTION6):
            self.moves += 1
            self._actions_since_reset += 1
            self._rebuild()

        if self._check_win():
            self.won = True
            self._game_over = True
            self._level_done = True
            self._history = []
            if self._current_level_idx >= len(LEVEL_DATA) - 1:
                self._last_level_won = True
            self.next_level()
            self.complete_action()
            return

        if self.moves >= (self.move_limit or float("inf")):
            self._handle_death()

        if self._game_over:
            self.lose()
        
        self.complete_action()

    def _take_snapshot(self) -> Dict:
        return {
            "buckets": copy.deepcopy(self.buckets),
            "cursor_pos": self.cursor_pos,
            "holding_ball": copy.deepcopy(self.holding_ball),
            "locked_buckets": set(self.locked_buckets),
            "keys_collected": self.keys_collected,
            "gravity_normal": self.gravity_normal,
            "moves_used": self.moves,
        }
    
    def _handle_pick(self):
        if self.holding_ball is not None:
            return
        
        bucket = self.buckets[self.cursor_pos]
        is_inverted = self.special_rules.get("gravity_inversion") and not self.gravity_normal

        if not bucket["balls"]:
            return

        top_idx = 0 if is_inverted else len(bucket["balls"]) - 1
        top_ball = bucket["balls"][top_idx]

        if self.special_rules.get("key_lock"):
            if top_ball == "lock":
                return
            if top_ball == "key":
                bucket["balls"].pop(top_idx)
                self.keys_collected += 1
                self.holding_ball = {
                    "color": "key",
                    "from_bucket": self.cursor_pos
                }
                return

        if is_inverted:
            if not bucket["balls"]:
                return
            ball_idx = 0
            if ball_idx in bucket.get("fixed_balls", []):
                return
            self.holding_ball = {
                "color": bucket["balls"].pop(0),
                "from_bucket": self.cursor_pos
            }
        else:
            if bucket["balls"] and not self._is_ball_hidden(self.cursor_pos, len(bucket["balls"]) - 1):
                ball_idx = len(bucket["balls"]) - 1
                if ball_idx in bucket.get("fixed_balls", []):
                    return
                
                if self.special_rules.get("joint_blocks"):
                    joint_size = self._get_joint_block_size(self.cursor_pos)
                    colors = []
                    for _ in range(joint_size):
                        colors.append(bucket["balls"].pop())
                    colors.reverse()
                    self.holding_ball = {
                        "color": colors[0],
                        "colors": colors,
                        "from_bucket": self.cursor_pos,
                        "joint_size": joint_size
                    }
                else:
                    self.holding_ball = {
                        "color": bucket["balls"].pop(),
                        "from_bucket": self.cursor_pos
                    }
    
    def _handle_place(self):
        if self.holding_ball is None:
            return
        
        bucket = self.buckets[self.cursor_pos]
        is_inverted = self.special_rules.get("gravity_inversion") and not self.gravity_normal

        if self.special_rules.get("key_lock") and self.holding_ball["color"] == "key":
            if bucket["balls"]:
                top_idx = 0 if is_inverted else len(bucket["balls"]) - 1
                if bucket["balls"][top_idx] == "lock":
                    bucket["balls"].pop(top_idx)
                    self.keys_collected -= 1
                    self.holding_ball = None
                    return
            if self._can_place_at_cursor():
                bucket["balls"].append(self.holding_ball["color"])
                self.keys_collected -= 1
                self.holding_ball = None
                return
            return
        
        bucket = self.buckets[self.cursor_pos]
        is_inverted = self.special_rules.get("gravity_inversion") and not self.gravity_normal
        
        if self._can_place_at_cursor():
            if is_inverted:
                bucket["balls"].insert(0, self.holding_ball["color"])
            else:
                if "colors" in self.holding_ball:
                    for color in self.holding_ball["colors"]:
                        bucket["balls"].append(color)
                else:
                    bucket["balls"].append(self.holding_ball["color"])
            
            if bucket.get("lock_after_receive", False):
                self.locked_buckets.add(self.cursor_pos)
            
            self._handle_chain_reaction(self.cursor_pos, self.holding_ball["color"])
            self.holding_ball = None
            
            if self.special_rules.get("gravity_inversion"):
                self.gravity_normal = not self.gravity_normal
    
    def _can_place_at_cursor(self) -> bool:
        if not self.holding_ball:
            return False
        
        if self.cursor_pos == self.holding_ball.get("from_bucket"):
            return True
        
        if self.cursor_pos in self.locked_buckets:
            return False
        
        bucket = self.buckets[self.cursor_pos]
        
        joint_size = self.holding_ball.get("joint_size", 1)
        if len(bucket["balls"]) + joint_size > bucket["capacity"]:
            return False
        
        return True
    
    def _get_joint_block_size(self, bucket_idx: int) -> int:
        if not self.special_rules.get("joint_blocks"):
            return 1
        
        bucket = self.buckets[bucket_idx]
        if not bucket["balls"]:
            return 0
        
        top_color = bucket["balls"][-1]
        count = 1
        
        for i in range(len(bucket["balls"]) - 2, -1, -1):
            if bucket["balls"][i] == top_color:
                count += 1
            else:
                break
        
        return count
    
    def _is_ball_hidden(self, bucket_idx: int, ball_idx: int) -> bool:
        bucket = self.buckets[bucket_idx]
        return ball_idx in bucket.get("hidden_balls", [])
    
    def _handle_chain_reaction(self, bucket_idx: int, ball_color: str):
        if "chain_reactions" not in self.special_rules:
            return
        
        reactions = self.special_rules["chain_reactions"]
        if bucket_idx not in reactions:
            return
        
        reaction = reactions[bucket_idx]
        if reaction["trigger_color"] == ball_color and reaction["effect"] == "move_top_ball":
            target_idx = reaction["target_bucket"]
            if target_idx < len(self.buckets) and self.buckets[target_idx]["balls"]:
                ball = self.buckets[target_idx]["balls"].pop()
                for i in range(len(self.buckets)):
                    if i != target_idx and len(self.buckets[i]["balls"]) < self.buckets[i]["capacity"]:
                        self.buckets[i]["balls"].append(ball)
                        break
    
    def _handle_death(self) -> None:
        self._lives -= 1

        if self._lives <= 0:
            self._game_over = True
            self._rebuild()
            return

        self.load_level_data(LEVEL_DATA[self._current_level_idx], restore_lives=False)

    def handle_reset(self) -> None:
        self._game_over = False
        if self._last_level_won or (self._consecutive_resets >= 1 and self._actions_since_reset == 0):
            self._consecutive_resets = 0
            self._actions_since_reset = 0
            self._last_level_won = False
            self._lives = self.MAX_LIVES
            self.full_reset()
        else:
            self._consecutive_resets += 1
            self._actions_since_reset = 0
            self._lives = self.MAX_LIVES
            self.load_level_data(LEVEL_DATA[self._current_level_idx])

    def _check_win(self) -> bool:

        if self.special_rules.get("key_lock"):
            for bucket in self.buckets:
                if "key" in bucket["balls"] or "lock" in bucket["balls"]:
                    return False
            if self.holding_ball and self.holding_ball.get("color") == "key":
                return False
            colors_in_buckets = {}
            for bucket in self.buckets:
                regular = [b for b in bucket["balls"] if b not in ("key", "lock")]
                if not regular:
                    continue
                if len(bucket["balls"]) != bucket["capacity"]:
                    return False
                if not all(b == regular[0] for b in regular):
                    return False
                color = regular[0]
                if color in colors_in_buckets:
                    return False
                colors_in_buckets[color] = True
            return True
        if self.special_rules.get("mirror_matching"):
            match_containers = self.special_rules.get("match_containers", [])
            unused_colors = self.special_rules.get("unused_colors", [])
            unused_container = self.special_rules.get("unused_container")
            
            for i in match_containers:
                bucket = self.buckets[i]
                if len(bucket["balls"]) != 8:
                    return False
                bottom_4 = bucket["balls"][0:4]
                top_4 = bucket["balls"][4:8]
                if top_4 != list(reversed(bottom_4)):
                    return False
            
            if unused_container is not None and unused_colors:
                unused_bucket = self.buckets[unused_container]
                if len(unused_bucket["balls"]) != len(unused_colors):
                    return False
                for ball in unused_bucket["balls"]:
                    if ball not in unused_colors:
                        return False
            
            return True
        
        if self.special_rules.get("pattern_matching"):
            match_containers = self.special_rules.get("match_containers", [])
            unused_colors_container = self.special_rules.get("unused_colors_container")
            unused_colors = self.special_rules.get("unused_colors", [])
            
            for i in match_containers:
                bucket = self.buckets[i]
                if len(bucket["balls"]) != 8:
                    return False
                
                bottom_4 = bucket["balls"][0:4]
                top_4 = bucket["balls"][4:8]
                
                if bottom_4 != top_4:
                    return False
            
            if unused_colors_container is not None:
                unused_bucket = self.buckets[unused_colors_container]
                if len(unused_bucket["balls"]) != 4:
                    return False
                
                for ball in unused_bucket["balls"]:
                    if ball not in unused_colors:
                        return False
                
                for color in unused_colors:
                    if unused_bucket["balls"].count(color) != 2:
                        return False
            
            return True
        
        if self.special_rules.get("gravity_inversion") and not self.gravity_normal:
            return False
        
        unused_container = self.special_rules.get("unused_container")
        target_colors = self.special_rules.get("target_colors")
        must_be_empty = self.special_rules.get("must_be_empty", [])
        
        for i, bucket in enumerate(self.buckets):
            if i in must_be_empty:
                if bucket["balls"]:
                    return False
                continue
            
            if unused_container is not None and i == unused_container:
                continue
            
            if not bucket["balls"]:
                continue
            
            if len(bucket["balls"]) != bucket["capacity"]:
                return False
            
            first_color = bucket["balls"][0]
            if not all(ball == first_color for ball in bucket["balls"]):
                return False
            
            if target_colors and i in target_colors:
                if first_color != target_colors[i]:
                    return False
        
        return True
    
    def _get_state(self) -> GameState:
        level_info = LEVEL_DATA[self._current_level_idx]
        
        obs = f"Level {self._current_level_idx + 1}: {level_info['name']}\n"
        obs += f"{level_info['description']}\n\n"
        obs += f"Moves: {self.moves}"
        if self.move_limit:
            obs += f"/{self.move_limit}"
        obs += f"  |  Lifelines: {self._lives}"
        if self.special_rules.get("key_lock"):
            total_locks = sum(b["balls"].count("lock") for b in self.buckets)
            obs += f"  |  Keys: {self.keys_collected}  |  Locks: {total_locks}"
        obs += "\n\n"
        
        for i, bucket in enumerate(self.buckets):
            locked = " [LOCKED]" if i in self.locked_buckets else ""
            obs += f"Bucket {i}{locked}: "
            
            if not bucket["balls"]:
                obs += "[Empty]"
            else:
                for j, ball in enumerate(bucket["balls"]):
                    if self._is_ball_hidden(i, j):
                        obs += "[?] "
                    elif j in bucket.get("fixed_balls", []):
                        obs += f"[{ball}*] "
                    else:
                        if self.special_rules.get("joint_blocks") and j > 0 and bucket["balls"][j] == bucket["balls"][j-1]:
                            obs += f"[{ball}=] "
                        else:
                            obs += f"[{ball}] "
            
            obs += f" ({len(bucket['balls'])}/{bucket['capacity']})\n"
        
        if self._game_over:
            if self.won:
                obs += "\n🎉 Level Complete! 🎉\n"
            else:
                obs += "\n❌ Move limit reached! Try again.\n"
        
        return GameState(
            text_observation=obs,
            image_observation=None,
            valid_actions=None,
            turn=self.moves,
            metadata={
                "level": self._current_level_idx + 1,
                "moves": self.moves,
                "won": self.won,
                "game_over": self._game_over,
                "lifelines": self._lives,
            }
        )
    
    def _sprite(self, pixels, name, x, y, layer, tags=None):
        return Sprite(
            pixels=pixels, name=name, x=x, y=y, layer=layer,
            visible=True, collidable=False, tags=tags or [],
        )
    
    def _rebuild(self):
        level = self.current_level
        self.build_level(level)
    
    def build_level(self, level: Level):
        level.remove_all_sprites()
        
        bg = np.full((CAM_H, CAM_W), LIGHT, dtype=np.int32)
        level.add_sprite(self._sprite(bg, "bg", 0, 0, -5, ["bg"]))
        
        title_bar = np.full((4, CAM_W), BLACK, dtype=np.int32)
        level.add_sprite(self._sprite(title_bar, "title_bar", 0, 0, 0, ["ui"]))
        
        num_buckets = len(self.buckets)
        
        if num_buckets >= 6:
            bucket_width = 8
            spacing = 2
            bucket_height = 46
            ball_size = 2
            start_y = 8
        else:
            bucket_width = 10
            spacing = 2
            bucket_height = 48
            ball_size = 3
            start_y = 6
        
        total_width = (num_buckets * bucket_width) + ((num_buckets - 1) * spacing)
        start_x = (CAM_W - total_width) // 2
        
        for i, bucket in enumerate(self.buckets):
            x = start_x + i * (bucket_width + spacing)
            
            if x + bucket_width > CAM_W - 1:
                break
            
            bucket_color = RED if i in self.locked_buckets else GRAY
            bucket_pixels = np.full((bucket_height, bucket_width), bucket_color, dtype=np.int32)
            bucket_pixels[2:-2, 2:-2] = WHITE
            
            if i == self.cursor_pos:
                bucket_pixels[0:2, :] = CYAN
                bucket_pixels[-2:, :] = CYAN
                bucket_pixels[:, 0:2] = CYAN
                bucket_pixels[:, -2:] = CYAN
            
            level.add_sprite(self._sprite(bucket_pixels, f"bucket_{i}", x, start_y, 1, ["bucket"]))
            
            ball_spacing = 1
            is_gravity_inverted = self.special_rules.get("gravity_inversion") and not self.gravity_normal
            
            if num_buckets >= 6:
                ball_padding_x = 3
                ball_padding_bottom = 4
                ball_padding_top = 4
            else:
                ball_padding_x = 3
                ball_padding_bottom = 5
                ball_padding_top = 3
            
            for j, ball in enumerate(bucket["balls"]):
                extra_mirror_gap = 3 if (self.special_rules.get("mirror_matching") and j >= 4) else 0
                
                if is_gravity_inverted:
                    ball_y = start_y + ball_padding_top + j * (ball_size + ball_spacing)
                else:
                    ball_y = start_y + bucket_height - ball_padding_bottom - (j + 1) * (ball_size + ball_spacing) - extra_mirror_gap
                
                ball_x = x + ball_padding_x
                
                if self._is_ball_hidden(i, j):
                    ball_color = GRAY
                    ball_pixels = np.full((ball_size, ball_size), ball_color, dtype=np.int32)
                elif ball == "key":
                    ball_pixels = np.full((ball_size, ball_size), CYAN, dtype=np.int32)
                    if ball_size >= 2:
                        mid = ball_size // 2
                        ball_pixels[mid, mid] = WHITE
                elif ball == "lock":
                    ball_pixels = np.full((ball_size, ball_size), BLACK, dtype=np.int32)
                else:
                    ball_color = COLOR_MAP.get(ball, WHITE)
                    ball_pixels = np.full((ball_size, ball_size), ball_color, dtype=np.int32)
                
                level.add_sprite(self._sprite(ball_pixels, f"ball_{i}_{j}", ball_x, ball_y, 3, ["ball"]))
            
            if self.special_rules.get("mirror_matching") and i in self.special_rules.get("match_containers", []):
                mirror_pos = self.special_rules.get("mirror_line_position", 3)
                mirror_extra_gap = 3
                fixed_top_y = start_y + bucket_height - ball_padding_bottom - (mirror_pos + 1) * (ball_size + ball_spacing)
                movable_bottom_y = fixed_top_y - mirror_extra_gap - ball_size
                line_y = (fixed_top_y + movable_bottom_y) // 2
                line_width = bucket_width - 4
                line_pixels = np.full((2, line_width), BLUE, dtype=np.int32)
                level.add_sprite(self._sprite(line_pixels, f"mirror_line_{i}", x + 2, line_y, 2, ["mirror_line"]))        
        if self.cursor_pos < len(self.buckets):
            is_gravity_inverted = self.special_rules.get("gravity_inversion") and not self.gravity_normal
            cursor_x = start_x + self.cursor_pos * (bucket_width + spacing) + bucket_width // 2 - 2
            
            if is_gravity_inverted:
                cursor_y = start_y + bucket_height + 1
                cursor_pixels = np.full((3, 5), RED, dtype=np.int32)
                cursor_pixels[0:2, 1:4] = ORANGE
            else:
                cursor_y = start_y - 4
                if self._current_level_idx == 0:
                    _INTRO_HINT = {0: BLUE, 1: MAGENTA, 2: BLUE, 3: MAGENTA, 4: YELLOW}
                    hint_col = _INTRO_HINT.get(self.cursor_pos, YELLOW)
                    cursor_pixels = np.full((4, 6), hint_col, dtype=np.int32)
                    cursor_pixels[0, :]  = WHITE
                    cursor_pixels[-1, :] = WHITE
                    cursor_pixels[:, 0]  = WHITE
                    cursor_pixels[:, -1] = WHITE
                else:
                    cursor_pixels = np.full((3, 5), ORANGE, dtype=np.int32)
                    cursor_pixels[1:, 1:4] = YELLOW
            
            level.add_sprite(self._sprite(cursor_pixels, "cursor", cursor_x, cursor_y, 6, ["cursor"]))
            
        if self.holding_ball and self.cursor_pos < len(self.buckets):
            is_gravity_inverted = self.special_rules.get("gravity_inversion") and not self.gravity_normal
            cursor_x_held = start_x + self.cursor_pos * (bucket_width + spacing) + bucket_width // 2 - ball_size // 2
            
            if is_gravity_inverted:
                held_y_base = start_y + bucket_height + 6
            else:
                held_y_base = start_y - ball_size - 2
            
            if "colors" in self.holding_ball:
                for idx, color in enumerate(self.holding_ball["colors"]):
                    held_ball_color = COLOR_MAP.get(color, WHITE)
                    held_ball_pixels = np.full((ball_size, ball_size), held_ball_color, dtype=np.int32)
                    held_y = held_y_base - (idx * (ball_size + 1))
                    level.add_sprite(self._sprite(held_ball_pixels, f"held_ball_{idx}", cursor_x_held, held_y, 8, ["held"]))
            else:
                held_ball_color = COLOR_MAP.get(self.holding_ball["color"], WHITE)
                held_ball_pixels = np.full((ball_size, ball_size), held_ball_color, dtype=np.int32)
                level.add_sprite(self._sprite(held_ball_pixels, "held_ball", cursor_x_held, held_y_base, 8, ["held"]))
        
        if self.special_rules.get("gravity_inversion"):
            ind_color = GREEN if self.gravity_normal else RED
            ind_pixels = np.full((4, 4), ind_color, dtype=np.int32)
            if self.gravity_normal:
                ind_pixels[0:2, 1:3] = ind_color
                ind_pixels[2:4, :] = ind_color
                ind_pixels[2:4, 0] = min(ind_color+1, 15)
                ind_pixels[2:4, 3] = min(ind_color+1, 15)
            else:
                ind_pixels[0:2, :] = ind_color
                ind_pixels[0:2, 0] = min(ind_color+1, 15)
                ind_pixels[0:2, 3] = min(ind_color+1, 15)
                ind_pixels[2:4, 1:3] = ind_color
            level.add_sprite(self._sprite(ind_pixels, "gravity_indicator", CAM_W - 6, 0, 9, ["ui"]))
        
        if self.special_rules.get("key_lock"):
            kx = 1
            ky = CAM_H - 7
            total_keys = sum(b["balls"].count("key") for b in self.buckets) + self.keys_collected
            total_locks = sum(b["balls"].count("lock") for b in self.buckets)
            for ki in range(total_keys):
                collected = ki < self.keys_collected
                kpx = np.full((2, 2), CYAN if collected else GRAY, dtype=np.int32)
                if collected:
                    kpx[0, 0] = WHITE
                level.add_sprite(self._sprite(kpx, f"key_ind_{ki}", kx + ki * 3, ky, 5, ["key_ui"]))
            lock_start_x = kx + total_keys * 3 + 2
            for li in range(total_locks):
                lpx = np.full((2, 2), BLACK, dtype=np.int32)
                level.add_sprite(self._sprite(lpx, f"lock_ind_{li}", lock_start_x + li * 3, ky, 5, ["lock_ui"]))

        pb_y = CAM_H - 3
        pb_width = CAM_W - 14
        pb_height = 2

        pbbg = np.full((pb_height, pb_width), GRAY, dtype=np.int32)
        level.add_sprite(self._sprite(pbbg, "pbbg", 1, pb_y, 3, ["moves"]))

        if self.move_limit and self.move_limit > 0:
            remaining = max(0, self.move_limit - self.moves)
            fill_pct = remaining / self.move_limit
            filled_cells = max(0, min(int(pb_width * fill_pct), pb_width))
            if filled_cells > 0:
                pb_color = RED if remaining <= (self.move_limit // 10) else GREEN
                pbfill = np.full((pb_height, filled_cells), pb_color, dtype=np.int32)
                level.add_sprite(self._sprite(pbfill, "pbfill", 1, pb_y, 4, ["moves"]))

        lf_x = 1 + pb_width + 2
        lf_y = pb_y
        for i in range(3):
            if self._game_over:
                lf_px = np.full((2, 2), GRAY, dtype=np.int32)
            else:
                lf_px = np.full((2, 2), RED if i < self._lives else GRAY, dtype=np.int32)
            level.add_sprite(self._sprite(lf_px, f"lf_{i}", lf_x + i * 3, lf_y, 4, ["lifeline"]))

        if self.won and self._current_level_idx != 3:
            msg_pixels = np.full((4, 24), GREEN, dtype=np.int32)
            msg_x = (CAM_W - 24) // 2
            level.add_sprite(self._sprite(msg_pixels, "win_msg", msg_x, CAM_H - 6, 8, ["message"]))



class PuzzleEnvironment:
    _ACTION_MAP: Dict[str, GameAction] = {
        "reset": GameAction.RESET,
        "up": GameAction.ACTION2,
        "down": GameAction.ACTION6,
        "left": GameAction.ACTION3,
        "right": GameAction.ACTION4,
        "select": GameAction.ACTION5,
        "undo": GameAction.ACTION7,
    }

    _VALID_ACTIONS = list(_ACTION_MAP.keys())

    def __init__(self, seed: int = 0) -> None:
        self._engine = Bb77(seed=seed)
        self._total_turns = 0
        self._done = False
        self._game_won = False
        self._last_action_was_reset = False

    def _build_text_observation(self) -> str:
        state = self._engine._get_state()
        return state.text_observation

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
        for idx in range(len(ARC_PALETTE)):
            mask = arr == idx
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            rgb[mask] = ARC_PALETTE[idx]
        return self._encode_png(rgb)

    def _build_game_state(self, done: bool = False) -> GameState:
        e = self._engine
        valid_actions = self.get_actions() if not done else None
        return GameState(
            text_observation=self._build_text_observation(),
            image_observation=self._build_image_bytes(),
            valid_actions=valid_actions,
            turn=self._total_turns,
            metadata={
                "total_levels": len(e._levels),
                "level_index": e.level_index,
                "levels_completed": getattr(e, "_score", 0),
                "game_over": getattr(getattr(e, "_state", None), "name", "") == "GAME_OVER",
                "done": done,
                "info": {},
            },
        )

    def reset(self) -> GameState:
        if self._game_won or self._last_action_was_reset:
            self._engine.perform_action(ActionInput(id=GameAction.RESET))
            self._engine.perform_action(ActionInput(id=GameAction.RESET))
        else:
            self._engine.perform_action(ActionInput(id=GameAction.RESET))
        self._last_action_was_reset = True
        self._done = False
        self._total_turns = 0
        self._game_won = False
        return self._build_game_state()

    def get_actions(self) -> List[str]:
        if self._done:
            return ["reset"]
        return list(self._VALID_ACTIONS)

    def is_done(self) -> bool:
        return self._done

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
        info: Dict = {"action": action}

        level_before = e.level_index

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
            self._game_won = False
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
        "left",
        "right",
        "select",
        "down",
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

