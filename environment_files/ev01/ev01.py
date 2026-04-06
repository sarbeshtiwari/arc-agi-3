from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

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

C_BG = 5
C_WALL = 4
C_SHAFT = 3
C_FLOOR_LINE = 2
C_ELEV = 9
C_ELEV_FULL = 10
C_DOOR_OPEN = 0
C_REQ_UP = 14
C_REQ_DOWN = 8
C_REQ_HIGH = 11
C_REQ_CRIT = 6
C_DEST = 15
C_EMPTY_IND = 5
C_ELEV_BORDER = 4
C_CURRENT_FLOOR = 10

BACKGROUND_COLOR = 5
PADDING_COLOR = 5

STARVATION_THRESHOLD = 10
STARVATION_GRACE = 30

GRID_WIDTH = 5
CELL_W = 3
CELL_H = 2
VIEWPORT_W = 32
VIEWPORT_H = 32
COL_WALL_L = 0
COL_DOOR_L = 1
COL_SHAFT = 2
COL_DOOR_R = 3
COL_WALL_R = 4

HUD_ROWS = 2
C_GAUGE_HIGH = 14
C_GAUGE_MED = 11
C_GAUGE_LOW = 8
C_HUD_BORDER = 2

MAX_LIVES = 3
C_LIFE = 8

FLOOR_ACCENT_INTERVAL = 5
FLOOR_ACCENT_MIN_FLOORS = 15

F = 64

_anchor_sprite = Sprite(
    pixels=[[C_BG]],
    name="anchor",
    visible=False,
    collidable=False,
    layer=0,
    tags=["anchor"],
)


class ElevatorOverlay(RenderableUserDisplay):
    STATUS_Y = 1
    DIVIDER1_Y = 3
    BLDG_TOP = 4
    BLDG_BOT = 56
    DIVIDER2_Y = 57
    GAUGE_Y = 59
    GAUGE_H = 2

    BLDG_X = 5
    BLDG_RIGHT = 58

    def __init__(self, game: "Ev01") -> None:
        self.g = game

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        g = self.g
        frame[:, :] = C_BG

        total_levels = g.total_levels
        cur_level = g.current_level_index
        for li in range(total_levels):
            lx = 2 + li * 3
            if li < cur_level:
                self._put(frame, self.STATUS_Y, lx, C_GAUGE_HIGH)
                self._put(frame, self.STATUS_Y, lx + 1, C_GAUGE_HIGH)
            elif li == cur_level:
                self._put(frame, self.STATUS_Y, lx, C_ELEV)
                self._put(frame, self.STATUS_Y, lx + 1, C_ELEV)
            else:
                self._put(frame, self.STATUS_Y, lx, C_WALL)
                self._put(frame, self.STATUS_Y, lx + 1, C_WALL)

        lives = g._lives
        for li in range(MAX_LIVES):
            rx = F - 2 - (MAX_LIVES - li) * 3
            if li < lives:
                self._put(frame, self.STATUS_Y, rx, C_LIFE)
                self._put(frame, self.STATUS_Y, rx + 1, C_LIFE)
            else:
                self._put(frame, self.STATUS_Y, rx, C_WALL)
                self._put(frame, self.STATUS_Y, rx + 1, C_WALL)

        self._hline(frame, self.DIVIDER1_Y, 1, F - 2, C_FLOOR_LINE)
        self._hline(frame, self.DIVIDER2_Y, 1, F - 2, C_FLOOR_LINE)

        num_floors = g._num_floors
        avail_h = self.BLDG_BOT - self.BLDG_TOP
        base_h = max(2, avail_h // num_floors)
        extra = avail_h - base_h * num_floors
        floor_heights = [base_h + (1 if fi < extra else 0) for fi in range(num_floors)]
        y_offset = self.BLDG_TOP

        bx_l = self.BLDG_X
        bx_r = self.BLDG_RIGHT
        shaft_l = bx_l + 1
        shaft_r = bx_r - 1
        ind_l = bx_l + 2
        ind_r = bx_r - 2

        fy_top = y_offset
        for y in range(y_offset, y_offset + avail_h):
            self._put(frame, y, bx_l, C_WALL)
            self._put(frame, y, bx_r, C_WALL)

        fy_top = y_offset
        for fi in range(num_floors):
            floor = (num_floors - 1) - fi
            floor_h = floor_heights[fi]

            for y in range(fy_top, fy_top + floor_h):
                shaft_color = C_CURRENT_FLOOR if floor == g._elevator_floor else C_SHAFT
                for x in range(shaft_l, shaft_r + 1):
                    self._put(frame, y, x, shaft_color)

            is_no_stop = floor in g._no_stop_floors
            if is_no_stop:
                for y in range(fy_top, fy_top + floor_h):
                    for x in range(shaft_l, shaft_r + 1):
                        self._put(frame, y, x, C_WALL)
                fy_top += floor_h
                continue

            ind_y_start = fy_top
            ind_y_end = min(fy_top + floor_h, ind_y_start + max(1, floor_h - 1))

            if floor != g._elevator_floor:
                down_color, up_color = g._scheduler.floor_indicator(floor)
                if down_color != C_EMPTY_IND:
                    for y in range(ind_y_start, ind_y_end):
                        self._put(frame, y, ind_l, down_color)
                        self._put(frame, y, ind_l + 1, down_color)
                if up_color != C_EMPTY_IND:
                    for y in range(ind_y_start, ind_y_end):
                        self._put(frame, y, ind_r, up_color)
                        self._put(frame, y, ind_r - 1, up_color)

            dest_reqs = g._scheduler.carried_for_floor(g._passengers, floor)
            if dest_reqs:
                mid_x = (shaft_l + shaft_r) // 2
                for y in range(ind_y_start, ind_y_end):
                    self._put(frame, y, mid_x, C_DEST)
                    self._put(frame, y, mid_x - 1, C_DEST)
                    self._put(frame, y, mid_x + 1, C_DEST)

            if floor == g._elevator_floor:
                car_l = shaft_l + 2
                car_r = shaft_r - 2
                car_y_start = ind_y_start
                car_y_end = fy_top + floor_h

                cab_color = C_ELEV

                if g._doors_open:
                    for y in range(car_y_start, car_y_end):
                        self._put(frame, y, car_l, C_DOOR_OPEN)
                        for x in range(car_l + 1, car_r):
                            self._put(frame, y, x, cab_color)
                        self._put(frame, y, car_r, C_DOOR_OPEN)
                else:
                    door_l_c, center_c, door_r_c = _elevator_cells(
                        g._passengers, g._doors_open
                    )
                    mid_x = (car_l + car_r) // 2
                    for y in range(car_y_start, car_y_end):
                        for x in range(car_l, car_r + 1):
                            if x < car_l + 2:
                                self._put(frame, y, x, door_l_c)
                            elif x > car_r - 2:
                                self._put(frame, y, x, door_r_c)
                            else:
                                self._put(frame, y, x, center_c)

                for x in range(car_l, car_r + 1):
                    self._put(frame, car_y_start, x, C_ELEV_BORDER)
                    if car_y_end - 1 > car_y_start:
                        self._put(frame, car_y_end - 1, x, C_ELEV_BORDER)
                for y in range(car_y_start, car_y_end):
                    self._put(frame, y, car_l, C_ELEV_BORDER)
                    self._put(frame, y, car_r, C_ELEV_BORDER)

            fy_top += floor_h

        gauge_x0 = 4
        gauge_x1 = F - 5
        gauge_w = gauge_x1 - gauge_x0 + 1

        if g._max_actions > 0:
            fill_ratio = max(0.0, min(1.0, g._actions_remaining / g._max_actions))
            filled_px = round(fill_ratio * gauge_w)
            if fill_ratio > 0.4:
                gauge_color = C_GAUGE_HIGH
            elif fill_ratio > 0.2:
                gauge_color = C_GAUGE_MED
            else:
                gauge_color = C_GAUGE_LOW
        else:
            filled_px = gauge_w
            gauge_color = C_GAUGE_HIGH

        for row in range(self.GAUGE_H):
            y = self.GAUGE_Y + row
            self._hline(frame, y, gauge_x0, gauge_x1, C_WALL)
            if filled_px > 0:
                self._hline(frame, y, gauge_x0, gauge_x0 + filled_px - 1, gauge_color)

        return frame

    @staticmethod
    def _put(frame: np.ndarray, y: int, x: int, c: int) -> None:
        if 0 <= y < F and 0 <= x < F:
            frame[y, x] = c

    @staticmethod
    def _hline(frame: np.ndarray, y: int, x0: int, x1: int, c: int) -> None:
        if 0 <= y < F:
            frame[y, max(0, x0) : min(F, x1 + 1)] = c

    @staticmethod
    def _fill_rect(frame: np.ndarray, x: int, y: int, w: int, h: int, c: int) -> None:
        for dy in range(h):
            for dx in range(w):
                px, py = x + dx, y + dy
                if 0 <= px < F and 0 <= py < F:
                    frame[py, px] = c

    @staticmethod
    def _rect_border(frame: np.ndarray, x: int, y: int, w: int, h: int, c: int) -> None:
        for dx in range(w):
            px = x + dx
            if 0 <= px < F:
                if 0 <= y < F:
                    frame[y, px] = c
                if 0 <= y + h - 1 < F:
                    frame[y + h - 1, px] = c
        for dy in range(h):
            py = y + dy
            if 0 <= py < F:
                if 0 <= x < F:
                    frame[py, x] = c
                if 0 <= x + w - 1 < F:
                    frame[py, x + w - 1] = c


@dataclass
class FloorRequest:
    floor: int
    destination: int
    created_turn: int
    priority: str = "normal"
    picked_up: bool = False
    served: bool = False
    high_priority_turn: int = -1

    @property
    def direction(self) -> str:
        if self.destination > self.floor:
            return "up"
        if self.destination < self.floor:
            return "down"
        return "none"


class Scheduler:
    def __init__(
        self,
        starvation_threshold: int = STARVATION_THRESHOLD,
        starvation_grace: int = STARVATION_GRACE,
    ) -> None:
        self.requests: List[FloorRequest] = []
        self.starvation_threshold = starvation_threshold
        self.starvation_grace = starvation_grace

    def add_request(self, req: FloorRequest) -> None:
        self.requests.append(req)

    def update_priorities(self, current_turn: int) -> None:
        for req in self.requests:
            if req.served or req.picked_up:
                continue
            if req.priority == "normal":
                age = current_turn - req.created_turn
                if age >= self.starvation_threshold:
                    req.priority = "high"
                    req.high_priority_turn = current_turn
            elif req.priority == "high" and req.high_priority_turn >= 0:
                grace_elapsed = current_turn - req.high_priority_turn
                if grace_elapsed > self.starvation_grace // 2:
                    req.priority = "critical"

    def pick_up(self, req: FloorRequest) -> None:
        req.picked_up = True

    def serve(self, req: FloorRequest) -> None:
        req.served = True

    def waiting_at_floor(self, floor: int) -> List[FloorRequest]:
        return [
            r
            for r in self.requests
            if r.floor == floor and not r.picked_up and not r.served
        ]

    def carried_for_floor(
        self, passengers: List[FloorRequest], floor: int
    ) -> List[FloorRequest]:
        return [p for p in passengers if p.destination == floor]

    def all_served(self) -> bool:
        return all(r.served for r in self.requests)

    def pending_count(self) -> int:
        return sum(1 for r in self.requests if not r.served)

    def high_priority_count(self) -> int:
        return sum(
            1
            for r in self.requests
            if not r.served and not r.picked_up and r.priority in ("high", "critical")
        )

    def has_expired_starvation(self, current_turn: int) -> bool:
        for req in self.requests:
            if req.served or req.picked_up:
                continue
            if req.high_priority_turn >= 0:
                if (current_turn - req.high_priority_turn) > self.starvation_grace:
                    return True
        return False

    def floor_indicator(self, floor: int) -> Tuple[int, int]:
        waiting = self.waiting_at_floor(floor)
        if not waiting:
            return (C_EMPTY_IND, C_EMPTY_IND)

        down_reqs = [r for r in waiting if r.direction == "down"]
        up_reqs = [r for r in waiting if r.direction == "up"]

        def _color(reqs: List[FloorRequest], normal_color: int) -> int:
            if not reqs:
                return C_EMPTY_IND
            if any(r.priority == "critical" for r in reqs):
                return C_REQ_CRIT
            if any(r.priority == "high" for r in reqs):
                return C_REQ_HIGH
            return normal_color

        return (_color(down_reqs, C_REQ_DOWN), _color(up_reqs, C_REQ_UP))


def _wall_color(floor: int, use_accents: bool) -> int:
    if use_accents and floor % FLOOR_ACCENT_INTERVAL == 0:
        return C_FLOOR_LINE
    return C_WALL


def _elevator_cells(
    passengers: List[FloorRequest],
    doors_open: bool,
) -> Tuple[int, int, int]:
    cab_color = C_ELEV
    if doors_open:
        return (C_DOOR_OPEN, cab_color, C_DOOR_OPEN)
    return (cab_color, cab_color, cab_color)


def _shaft_cells(
    passengers: List[FloorRequest],
    scheduler: Scheduler,
    floor: int,
) -> Tuple[int, int, int]:
    dest_reqs = scheduler.carried_for_floor(passengers, floor)
    center = C_DEST if dest_reqs else C_SHAFT
    return (C_SHAFT, center, C_SHAFT)


def build_grid(
    num_floors: int,
    elevator_floor: int,
    passengers: List[FloorRequest],
    scheduler: Scheduler,
    doors_open: bool = False,
    no_stop_floors: Optional[Set[int]] = None,
) -> List[List[int]]:
    grid: List[List[int]] = []
    use_accents = num_floors >= FLOOR_ACCENT_MIN_FLOORS

    for row in range(num_floors):
        floor = (num_floors - 1) - row
        wall = _wall_color(floor, use_accents)
        is_no_stop = no_stop_floors is not None and floor in no_stop_floors

        if floor == elevator_floor:
            door_l, center, door_r = _elevator_cells(passengers, doors_open)
        elif is_no_stop:
            door_l, center, door_r = C_WALL, C_WALL, C_WALL
        else:
            door_l, center, door_r = _shaft_cells(passengers, scheduler, floor)

        grid.append([wall, door_l, center, door_r, wall])

    for _ in range(HUD_ROWS):
        grid.append([C_BG] * GRID_WIDTH)

    return grid


_sprite_cache: Dict[int, Sprite] = {}


def _get_sprite(color: int) -> Sprite:
    if color not in _sprite_cache:
        _sprite_cache[color] = Sprite(
            pixels=[[color] * CELL_W for _ in range(CELL_H)],
            name=f"cell_{color}",
            visible=True,
            collidable=False,
            layer=1,
            tags=["cell"],
        )
    return _sprite_cache[color]


def grid_to_sprites(grid: List[List[int]]) -> List[Sprite]:
    sprites_list: List[Sprite] = []
    for y, row in enumerate(grid):
        for x, val in enumerate(row):
            sprite = _get_sprite(val).clone().set_position(x * CELL_W, y * CELL_H)
            sprites_list.append(sprite)
    return sprites_list


LEVEL_CONFIGS: List[Dict[str, Any]] = [
    {
        "name": "Express Service",
        "num_floors": 5,
        "elevator_start": 0,
        "max_actions": 100,
        "requests": [
            {"floor": 3, "destination": 1, "spawn_turn": 0},
            {"floor": 1, "destination": 4, "spawn_turn": 0},
            {"floor": 4, "destination": 0, "spawn_turn": 0},
        ],
    },
    {
        "name": "Cluster Run",
        "num_floors": 8,
        "elevator_start": 0,
        "max_actions": 360,
        "door_cost": 2,
        "requests": [
            {"floor": 2, "destination": 6, "spawn_turn": 0},
            {"floor": 3, "destination": 0, "spawn_turn": 0},
            {"floor": 5, "destination": 7, "spawn_turn": 0},
            {"floor": 6, "destination": 1, "spawn_turn": 0},
            {"floor": 7, "destination": 3, "spawn_turn": 0},
            {"floor": 4, "destination": 7, "spawn_turn": 2},
            {"floor": 5, "destination": 0, "spawn_turn": 4},
        ],
    },
    {
        "name": "Skyscraper",
        "num_floors": 13,
        "elevator_start": 6,
        "max_actions": 700,
        "door_cost": 2,
        "starvation_threshold": 20,
        "starvation_grace": 50,
        "requests": [
            {"floor": 0, "destination": 12, "spawn_turn": 0},
            {"floor": 12, "destination": 0, "spawn_turn": 0},
            {"floor": 3, "destination": 10, "spawn_turn": 0},
            {"floor": 9, "destination": 2, "spawn_turn": 0},
            {"floor": 5, "destination": 11, "spawn_turn": 0},
            {"floor": 7, "destination": 1, "spawn_turn": 0},
            {"floor": 1, "destination": 8, "spawn_turn": 2},
            {"floor": 10, "destination": 4, "spawn_turn": 2},
            {"floor": 6, "destination": 12, "spawn_turn": 4},
            {"floor": 4, "destination": 9, "spawn_turn": 6},
            {"floor": 11, "destination": 3, "spawn_turn": 8},
            {"floor": 8, "destination": 0, "spawn_turn": 10},
        ],
    },
    {
        "name": "Grand Central",
        "num_floors": 20,
        "elevator_start": 10,
        "max_actions": 900,
        "door_cost": 2,
        "starvation_threshold": 24,
        "starvation_grace": 60,
        "no_stop_floors": [],
        "requests": [
            {"floor": 0, "destination": 19, "spawn_turn": 0},
            {"floor": 19, "destination": 0, "spawn_turn": 0},
            {"floor": 4, "destination": 15, "spawn_turn": 0},
            {"floor": 14, "destination": 2, "spawn_turn": 0},
            {"floor": 8, "destination": 17, "spawn_turn": 0},
            {"floor": 12, "destination": 1, "spawn_turn": 2},
            {"floor": 2, "destination": 16, "spawn_turn": 2},
            {"floor": 17, "destination": 6, "spawn_turn": 4},
            {"floor": 9, "destination": 3, "spawn_turn": 4},
            {"floor": 5, "destination": 18, "spawn_turn": 6},
            {"floor": 15, "destination": 2, "spawn_turn": 8},
            {"floor": 18, "destination": 8, "spawn_turn": 10},
            {"floor": 11, "destination": 4, "spawn_turn": 12},
            {"floor": 16, "destination": 1, "spawn_turn": 14},
        ],
    },
]


def _build_levels() -> List[Level]:
    result: List[Level] = []
    for cfg in LEVEL_CONFIGS:
        num_floors = cfg["num_floors"]
        no_stop = set(cfg.get("no_stop_floors", []))

        level = Level(
            sprites=[_anchor_sprite.clone().set_position(0, 0)],
            grid_size=(16, 16),
            data={
                "num_floors": num_floors,
                "elevator_start": cfg["elevator_start"],
                "max_actions": cfg["max_actions"],
                "capacity": cfg.get("capacity"),
                "door_cost": cfg.get("door_cost", 1),
                "starvation_threshold": cfg.get(
                    "starvation_threshold", STARVATION_THRESHOLD
                ),
                "starvation_grace": cfg.get("starvation_grace", STARVATION_GRACE),
                "no_stop_floors": list(no_stop),
                "requests": cfg["requests"],
            },
            name=cfg.get("name", ""),
        )
        result.append(level)
    return result


class Ev01(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)
        self._elevator_floor: int = 0
        self._passengers: List[FloorRequest] = []
        self._scheduler = Scheduler()
        self._turn: int = 0
        self._doors_open: bool = False
        self._actions_remaining: int = 0
        self._max_actions: int = 0
        self._num_floors: int = 5
        self._deferred_requests: List[Dict[str, Any]] = []
        self._lives: int = MAX_LIVES
        self._restarting_level: bool = False
        self._elevator_capacity: Optional[int] = None
        self._door_action_cost: int = 1
        self._no_stop_floors: Set[int] = set()
        self._action_count: int = 0
        self._priority_violated: bool = False
        self._history: List[Dict] = []

        game_levels = _build_levels()
        self.total_levels: int = len(game_levels)
        self.current_level_index: int = 0
        self._game_won: bool = False
        self._levels = game_levels

        self.overlay = ElevatorOverlay(self)

        cam = Camera(0, 0, 16, 16, BACKGROUND_COLOR, PADDING_COLOR, [self.overlay])

        super().__init__(
            game_id="ev01",
            levels=self._levels,
            camera=cam,
            available_actions=[0, 1, 2, 5, 7],
            win_score=len(game_levels),
        )

    def on_set_level(self, level: Level) -> None:
        for i, lv in enumerate(self._levels):
            if lv is level:
                self.current_level_index = i
                break

        if not self._restarting_level:
            self._lives = MAX_LIVES

        self._num_floors = level.get_data("num_floors")
        self._max_actions = level.get_data("max_actions")
        self._actions_remaining = self._max_actions
        self._elevator_capacity = level.get_data("capacity")
        self._door_action_cost = level.get_data("door_cost")
        self._passengers = []
        self._doors_open = False
        self._turn = 0
        self._priority_violated = False
        self._history = []

        no_stop_data = level.get_data("no_stop_floors")
        self._no_stop_floors = set(no_stop_data) if no_stop_data else set()

        valid_floors = [
            f for f in range(self._num_floors) if f not in self._no_stop_floors
        ]
        self._elevator_floor = self._rng.choice(valid_floors)

        thresh = level.get_data("starvation_threshold")
        grace = level.get_data("starvation_grace")
        self._scheduler = Scheduler(
            starvation_threshold=thresh if thresh is not None else STARVATION_THRESHOLD,
            starvation_grace=grace if grace is not None else STARVATION_GRACE,
        )
        req_defs = level.get_data("requests")
        randomized = self._randomize_requests(req_defs)
        for rdef in randomized:
            if rdef["spawn_turn"] <= 0:
                self._scheduler.add_request(
                    FloorRequest(
                        floor=rdef["floor"],
                        destination=rdef["destination"],
                        created_turn=0,
                    )
                )
        self._deferred_requests = [
            rdef for rdef in randomized if rdef["spawn_turn"] > 0
        ]

    def _randomize_requests(
        self, req_defs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        valid_floors = [
            f for f in range(self._num_floors) if f not in self._no_stop_floors
        ]
        non_start_floors = [f for f in valid_floors if f != self._elevator_floor]
        if len(non_start_floors) < 2:
            return req_defs
        result: List[Dict[str, Any]] = []
        for rdef in req_defs:
            floor = self._rng.choice(non_start_floors)
            dest = self._rng.choice([f for f in valid_floors if f != floor])
            result.append(
                {
                    "floor": floor,
                    "destination": dest,
                    "spawn_turn": rdef["spawn_turn"],
                }
            )
        return result

    def _rebuild_sprites(self) -> None:
        pass

    def _advance_turn(self) -> None:
        self._turn += 1
        self._actions_remaining -= 1

        still_deferred: List[Dict[str, Any]] = []
        for rdef in self._deferred_requests:
            if rdef["spawn_turn"] <= self._turn:
                self._scheduler.add_request(
                    FloorRequest(
                        floor=rdef["floor"],
                        destination=rdef["destination"],
                        created_turn=self._turn,
                    )
                )
            else:
                still_deferred.append(rdef)
        self._deferred_requests = still_deferred

        self._scheduler.update_priorities(self._turn)

    def _open_doors(self) -> None:
        floor = self._elevator_floor
        self._doors_open = True

        deboarding = self._scheduler.carried_for_floor(self._passengers, floor)
        for passenger in deboarding:
            self._scheduler.serve(passenger)
            self._passengers.remove(passenger)

        waiting = self._scheduler.waiting_at_floor(floor)
        if self._elevator_capacity is not None:
            remaining_capacity = self._elevator_capacity - len(self._passengers)
            priority_order = {"critical": 0, "high": 1, "normal": 2}
            waiting.sort(key=lambda r: priority_order.get(r.priority, 2))
            waiting = waiting[: max(0, remaining_capacity)]
        boarded: List[FloorRequest] = []
        for waiting_req in waiting:
            self._scheduler.pick_up(waiting_req)
            self._passengers.append(waiting_req)
            boarded.append(waiting_req)

        if boarded:
            _RANK = {"normal": 0, "high": 1, "critical": 2}
            min_boarded_rank = min(_RANK.get(r.priority, 0) for r in boarded)
            max_waiting_rank = max(
                (
                    _RANK.get(r.priority, 0)
                    for r in self._scheduler.requests
                    if not r.served and not r.picked_up
                ),
                default=-1,
            )
            if max_waiting_rank > min_boarded_rank:
                self._priority_violated = True

    def _check_win(self) -> bool:
        return self._scheduler.all_served() and len(self._deferred_requests) == 0

    def _check_lose(self) -> bool:
        if self._scheduler.has_expired_starvation(self._turn):
            return True
        return self._actions_remaining <= 0

    def _handle_action(self) -> bool:
        action_id = self.action.id
        doors_opened = False

        if action_id == GameAction.ACTION1:
            if self._elevator_floor < self._num_floors - 1:
                self._elevator_floor += 1

        elif action_id == GameAction.ACTION2:
            if self._elevator_floor > 0:
                self._elevator_floor -= 1

        elif action_id == GameAction.ACTION5:
            if self._elevator_floor not in self._no_stop_floors:
                self._open_doors()
                doors_opened = True

        return doors_opened

    def _resolve_outcomes(self) -> None:
        is_win = self._check_win()

        if is_win:
            self._priority_violated = False
            self.next_level()
            self.complete_action()
            return

        if self._check_lose():
            self._priority_violated = False
            self._lives -= 1
            if self._lives <= 0:
                self.lose()
            else:
                self._restarting_level = True
                self.on_set_level(self.current_level)
                self._restarting_level = False
            self.complete_action()
            return

        if self._priority_violated:
            self._priority_violated = False
            self._lives -= 1
            if self._lives <= 0:
                self.lose()
            else:
                self._restarting_level = True
                self.on_set_level(self.current_level)
                self._restarting_level = False
            self.complete_action()
            return

        self.complete_action()

    def _save_state(self) -> None:
        scheduler_snap = []
        for r in self._scheduler.requests:
            scheduler_snap.append(
                {
                    "floor": r.floor,
                    "destination": r.destination,
                    "created_turn": r.created_turn,
                    "priority": r.priority,
                    "picked_up": r.picked_up,
                    "served": r.served,
                    "high_priority_turn": r.high_priority_turn,
                }
            )
        self._history.append(
            {
                "elevator_floor": self._elevator_floor,
                "doors_open": self._doors_open,
                "turn": self._turn,
                "priority_violated": self._priority_violated,
                "scheduler_requests": scheduler_snap,
                "deferred_requests": [dict(d) for d in self._deferred_requests],
            }
        )

    def _undo(self) -> None:
        if not self._history:
            return
        snap = self._history.pop()
        self._elevator_floor = snap["elevator_floor"]
        self._doors_open = snap["doors_open"]
        self._turn = snap["turn"]
        self._priority_violated = snap["priority_violated"]
        self._deferred_requests = snap["deferred_requests"]
        self._scheduler.requests = [
            FloorRequest(**rd) for rd in snap["scheduler_requests"]
        ]
        self._passengers = [
            r for r in self._scheduler.requests if r.picked_up and not r.served
        ]

    def _reset_current_level(self) -> None:
        self._restarting_level = False
        self._lives = MAX_LIVES
        self.on_set_level(self.current_level)
        self._action_count = 0

    def step(self) -> None:
        if self.action.id == GameAction.RESET:
            self._reset_current_level()
            self.complete_action()
            return

        if self.action.id == GameAction.ACTION7:
            self._action_count += 1
            self._actions_remaining -= 1
            if self._actions_remaining <= 0:
                self._lives -= 1
                if self._lives <= 0:
                    self.lose()
                else:
                    self._restarting_level = True
                    self.on_set_level(self.current_level)
                    self._restarting_level = False
                self.complete_action()
                return
            self._undo()
            self.complete_action()
            return

        self._save_state()
        self._action_count += 1
        self._doors_open = False

        doors_opened = self._handle_action()

        turn_cost = self._door_action_cost if doors_opened else 1
        actual_cost = min(turn_cost, self._actions_remaining)
        for _ in range(actual_cost):
            self._advance_turn()

        self._rebuild_sprites()
        self._resolve_outcomes()


_COLOR_CHARS: Dict[int, str] = {
    0: "□",
    1: "░",
    2: "─",
    3: "│",
    4: "█",
    5: " ",
    6: "M",
    7: "•",
    8: "v",
    9: "E",
    10: "e",
    11: "!",
    12: "X",
    13: "·",
    14: "^",
    15: "*",
}


def _grid_to_text(grid: List[List[int]]) -> str:
    lines: List[str] = []
    for row in grid:
        lines.append("".join(_COLOR_CHARS.get(c, "?") for c in row))
    return "\n".join(lines)


class PuzzleEnvironment:
    _ACTION_MAP: Dict[str, GameAction] = {
        "reset": GameAction.RESET,
        "up": GameAction.ACTION1,
        "down": GameAction.ACTION2,
        "select": GameAction.ACTION5,
        "undo": GameAction.ACTION7,
    }

    _VALID_ACTIONS = list(_ACTION_MAP.keys())

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self._engine = Ev01(seed=seed)
        self._total_turns = 0
        self._done = False
        self._game_over = False
        self._last_action_was_reset = False

    def reset(self) -> GameState:
        e = self._engine
        game_won = e._game_won or e.current_level_index >= e.total_levels
        game_over = self._game_over
        if game_won or e._action_count == 0 or self._last_action_was_reset:
            self._engine = Ev01(seed=self._seed)
            e = self._engine
        else:
            e.perform_action(ActionInput(id=GameAction.RESET))

        self._total_turns = 0
        self._done = False
        self._last_action_was_reset = True
        self._game_over = False

        return self._build_game_state()

    def step(self, action: str) -> StepResult:
        e = self._engine

        if action == "reset":
            state = self.reset()
            return StepResult(
                state=state,
                reward=0.0,
                done=False,
                info={"action": "reset"},
            )

        click_data: Dict[str, Any] = {}
        base_action = action

        if base_action not in self._ACTION_MAP:
            raise ValueError(
                f"Invalid action '{action}'. "
                f"Must be one of {list(self._ACTION_MAP.keys())}"
            )

        self._last_action_was_reset = False
        self._total_turns += 1

        game_action = self._ACTION_MAP[base_action]
        info: Dict[str, Any] = {"action": action}

        level_before = e.current_level_index

        if click_data:
            action_input = ActionInput(id=game_action, data=click_data)
        else:
            action_input = ActionInput(id=game_action)
        frame = e.perform_action(action_input, raw=True)

        state_name = frame.state.name if frame and frame.state else ""
        game_won = state_name == "WIN" or e._game_won
        game_over = state_name == "GAME_OVER"

        level_reward = 1.0 / e.total_levels

        if game_won:
            self._done = True
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
            self._game_over = True
            info["reason"] = "game_over"
            return StepResult(
                state=self._build_game_state(done=True),
                reward=0.0,
                done=True,
                info=info,
            )

        reward = 0.0
        if e.current_level_index != level_before:
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
        actions: List[str] = ["reset"]
        e = self._engine
        can_open = (
            e._elevator_floor not in e._no_stop_floors
            and e._actions_remaining >= e._door_action_cost
        )
        if can_open:
            actions.append("select")
        actions.append("up")
        actions.append("down")
        actions.append("undo")
        return actions

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

    def _build_text_observation(self) -> str:
        e = self._engine
        grid = build_grid(
            num_floors=e._num_floors,
            elevator_floor=e._elevator_floor,
            passengers=e._passengers,
            scheduler=e._scheduler,
            doors_open=e._doors_open,
            no_stop_floors=e._no_stop_floors,
        )
        return _grid_to_text(grid)

    def _build_game_state(self, done: bool = False) -> GameState:
        e = self._engine
        valid_actions = self.get_actions() if not done else None

        return GameState(
            text_observation=self._build_text_observation(),
            image_observation=None,
            valid_actions=valid_actions,
            turn=self._total_turns,
            metadata={
                "total_levels": e.total_levels,
                "level_index": e.current_level_index,
                "levels_completed": getattr(e, "_score", 0),
                "game_over": self._game_over,
                "done": done,
                "info": {},
            },
        )


class ArcGameEnv(gym.Env):
    metadata: Dict[str, Any] = {
        "render_modes": ["rgb_array"],
        "render_fps": 5,
    }

    ACTION_LIST: List[str] = [
        "reset",
        "up",
        "down",
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
        self,
        state: GameState,
        step_info: Optional[Dict] = None,
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
