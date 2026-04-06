import struct
import zlib
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


MAX_LIVES = 3

BACKGROUND_COLOR = 5
PADDING_COLOR = 4

sprites = {
    "plr": Sprite(
        pixels=[[11]],
        name="plr",
        visible=True,
        collidable=True,
        tags=["plr"],
        layer=5,
    ),
    "gol": Sprite(
        pixels=[[14]],
        name="gol",
        visible=True,
        collidable=True,
        tags=["gol"],
        layer=1,
    ),
    "r04": Sprite(
        pixels=[[8, 8, 8, 8]],
        name="r04",
        visible=True,
        collidable=True,
        tags=["obs", "row"],
        layer=2,
    ),
    "r05": Sprite(
        pixels=[[8, 8, 8, 8, 8]],
        name="r05",
        visible=True,
        collidable=True,
        tags=["obs", "row"],
        layer=2,
    ),
    "r06": Sprite(
        pixels=[[8, 8, 8, 8, 8, 8]],
        name="r06",
        visible=True,
        collidable=True,
        tags=["obs", "row"],
        layer=2,
    ),
    "r07": Sprite(
        pixels=[[8, 8, 8, 8, 8, 8, 8]],
        name="r07",
        visible=True,
        collidable=True,
        tags=["obs", "row"],
        layer=2,
    ),
    "r08": Sprite(
        pixels=[[8, 8, 8, 8, 8, 8, 8, 8]],
        name="r08",
        visible=True,
        collidable=True,
        tags=["obs", "row"],
        layer=2,
    ),
    "r10": Sprite(
        pixels=[[8] * 10],
        name="r10",
        visible=True,
        collidable=True,
        tags=["obs", "row"],
        layer=2,
    ),
    "b06": Sprite(
        pixels=[[9, 9, 9, 9, 9, 9]],
        name="b06",
        visible=True,
        collidable=True,
        tags=["obs", "row"],
        layer=2,
    ),
    "b08": Sprite(
        pixels=[[9] * 8],
        name="b08",
        visible=True,
        collidable=True,
        tags=["obs", "row"],
        layer=2,
    ),
    "m06": Sprite(
        pixels=[[6] * 6],
        name="m06",
        visible=True,
        collidable=True,
        tags=["obs", "row"],
        layer=2,
    ),
    "m08": Sprite(
        pixels=[[6] * 8],
        name="m08",
        visible=True,
        collidable=True,
        tags=["obs", "row"],
        layer=2,
    ),
    "m10": Sprite(
        pixels=[[6] * 10],
        name="m10",
        visible=True,
        collidable=True,
        tags=["obs", "row"],
        layer=2,
    ),
    "o06": Sprite(
        pixels=[[12] * 6],
        name="o06",
        visible=True,
        collidable=True,
        tags=["obs", "row"],
        layer=2,
    ),
    "o08": Sprite(
        pixels=[[12] * 8],
        name="o08",
        visible=True,
        collidable=True,
        tags=["obs", "row"],
        layer=2,
    ),
    "o10": Sprite(
        pixels=[[12] * 10],
        name="o10",
        visible=True,
        collidable=True,
        tags=["obs", "row"],
        layer=2,
    ),
    "p05": Sprite(
        pixels=[[15] * 5],
        name="p05",
        visible=True,
        collidable=True,
        tags=["obs", "row"],
        layer=2,
    ),
    "p07": Sprite(
        pixels=[[15] * 7],
        name="p07",
        visible=True,
        collidable=True,
        tags=["obs", "row"],
        layer=2,
    ),
    "d06": Sprite(
        pixels=[[8] * 6, [8] * 6],
        name="d06",
        visible=True,
        collidable=True,
        tags=["obs", "row"],
        layer=2,
    ),
    "d08": Sprite(
        pixels=[[8] * 8, [8] * 8],
        name="d08",
        visible=True,
        collidable=True,
        tags=["obs", "row"],
        layer=2,
    ),
    "d10": Sprite(
        pixels=[[8] * 10, [8] * 10],
        name="d10",
        visible=True,
        collidable=True,
        tags=["obs", "row"],
        layer=2,
    ),
    "d12": Sprite(
        pixels=[[8] * 12, [8] * 12],
        name="d12",
        visible=True,
        collidable=True,
        tags=["obs", "row"],
        layer=2,
    ),
    "db06": Sprite(
        pixels=[[9] * 6, [9] * 6],
        name="db06",
        visible=True,
        collidable=True,
        tags=["obs", "row"],
        layer=2,
    ),
    "db08": Sprite(
        pixels=[[9] * 8, [9] * 8],
        name="db08",
        visible=True,
        collidable=True,
        tags=["obs", "row"],
        layer=2,
    ),
    "dm08": Sprite(
        pixels=[[6] * 8, [6] * 8],
        name="dm08",
        visible=True,
        collidable=True,
        tags=["obs", "row"],
        layer=2,
    ),
    "dm10": Sprite(
        pixels=[[6] * 10, [6] * 10],
        name="dm10",
        visible=True,
        collidable=True,
        tags=["obs", "row"],
        layer=2,
    ),
    "do08": Sprite(
        pixels=[[12] * 8, [12] * 8],
        name="do08",
        visible=True,
        collidable=True,
        tags=["obs", "row"],
        layer=2,
    ),
    "do10": Sprite(
        pixels=[[12] * 10, [12] * 10],
        name="do10",
        visible=True,
        collidable=True,
        tags=["obs", "row"],
        layer=2,
    ),
    "dp06": Sprite(
        pixels=[[15] * 6, [15] * 6],
        name="dp06",
        visible=True,
        collidable=True,
        tags=["obs", "row"],
        layer=2,
    ),
    "dp08": Sprite(
        pixels=[[15] * 8, [15] * 8],
        name="dp08",
        visible=True,
        collidable=True,
        tags=["obs", "row"],
        layer=2,
    ),
    "w31": Sprite(
        pixels=[[3, 3, 3]],
        name="w31",
        visible=True,
        collidable=True,
        tags=["wal"],
        layer=3,
    ),
    "w22": Sprite(
        pixels=[[3, 3], [3, 3]],
        name="w22",
        visible=True,
        collidable=True,
        tags=["wal"],
        layer=3,
    ),
    "w32": Sprite(
        pixels=[[3, 3, 3], [3, 3, 3]],
        name="w32",
        visible=True,
        collidable=True,
        tags=["wal"],
        layer=3,
    ),
    "w42": Sprite(
        pixels=[[4, 4, 4, 4], [4, 4, 4, 4]],
        name="w42",
        visible=True,
        collidable=True,
        tags=["wal"],
        layer=3,
    ),
    "dth": Sprite(
        pixels=[[8]],
        name="dth",
        visible=False,
        collidable=False,
        tags=["dth"],
        layer=10,
    ),
}


levels = [
    Level(
        sprites=[
            sprites["plr"].clone().set_position(8, 0),
            sprites["gol"].clone().set_position(9, 17),
            sprites["r06"].clone().set_position(0, 2),
            sprites["r05"].clone().set_position(9, 2),
            sprites["w31"].clone().set_position(7, 4),
            sprites["b08"].clone().set_position(8, 5),
            sprites["r05"].clone().set_position(0, 8),
            sprites["r06"].clone().set_position(8, 8),
            sprites["w22"].clone().set_position(0, 10),
            sprites["w22"].clone().set_position(14, 10),
            sprites["m08"].clone().set_position(10, 12),
            sprites["p05"].clone().set_position(0, 15),
            sprites["p05"].clone().set_position(10, 15),
        ],
        grid_size=(18, 18),
        data={
            "mvl": 30,
            "rwc": [
                ("r06", 2, 0, 1, 1),
                ("r05", 2, 9, 1, 1),
                ("b08", 5, 8, -1, 1),
                ("r05", 8, 0, 1, 1),
                ("r06", 8, 8, 1, 1),
                ("m08", 12, 10, -1, 2),
                ("p05", 15, 0, 1, 1),
                ("p05", 15, 10, 1, 1),
            ],
            "gw": 18,
            "gh": 18,
            "px": 8,
            "py": 0,
        },
        name="lvl1",
    ),
    Level(
        sprites=[
            sprites["plr"].clone().set_position(10, 0),
            sprites["gol"].clone().set_position(11, 21),
            sprites["r08"].clone().set_position(0, 2),
            sprites["r06"].clone().set_position(11, 2),
            sprites["w32"].clone().set_position(9, 3),
            sprites["b06"].clone().set_position(14, 5),
            sprites["o06"].clone().set_position(2, 5),
            sprites["r10"].clone().set_position(3, 7),
            sprites["w42"].clone().set_position(0, 9),
            sprites["w42"].clone().set_position(16, 9),
            sprites["m06"].clone().set_position(14, 11),
            sprites["m06"].clone().set_position(2, 11),
            sprites["p07"].clone().set_position(0, 13),
            sprites["r05"].clone().set_position(12, 13),
            sprites["w22"].clone().set_position(5, 15),
            sprites["w22"].clone().set_position(13, 15),
            sprites["o10"].clone().set_position(10, 17),
            sprites["r06"].clone().set_position(0, 19),
            sprites["r08"].clone().set_position(10, 19),
        ],
        grid_size=(22, 22),
        data={
            "mvl": 48,
            "rwc": [
                ("r08", 2, 0, 1, 1),
                ("r06", 2, 11, 1, 1),
                ("b06", 5, 14, -1, 2),
                ("o06", 5, 2, -1, 2),
                ("r10", 7, 3, 1, 1),
                ("m06", 11, 14, -1, 2),
                ("m06", 11, 2, -1, 2),
                ("p07", 13, 0, 1, 1),
                ("r05", 13, 12, 1, 1),
                ("o10", 17, 10, -1, 2),
                ("r06", 19, 0, 1, 1),
                ("r08", 19, 10, 1, 1),
            ],
            "gw": 22,
            "gh": 22,
            "px": 10,
            "py": 0,
        },
        name="lvl2",
    ),
    Level(
        sprites=[
            sprites["plr"].clone().set_position(12, 0),
            sprites["gol"].clone().set_position(13, 25),
            sprites["d10"].clone().set_position(4, 2),
            sprites["r05"].clone().set_position(18, 2),
            sprites["w32"].clone().set_position(11, 4),
            sprites["db08"].clone().set_position(16, 5),
            sprites["o08"].clone().set_position(0, 7),
            sprites["o06"].clone().set_position(14, 7),
            sprites["w42"].clone().set_position(0, 9),
            sprites["w22"].clone().set_position(18, 9),
            sprites["dm10"].clone().set_position(12, 10),
            sprites["r06"].clone().set_position(0, 13),
            sprites["r04"].clone().set_position(10, 13),
            sprites["r06"].clone().set_position(18, 13),
            sprites["w31"].clone().set_position(5, 15),
            sprites["w31"].clone().set_position(15, 15),
            sprites["w31"].clone().set_position(22, 15),
            sprites["dp08"].clone().set_position(2, 16),
            sprites["r06"].clone().set_position(16, 16),
            sprites["b08"].clone().set_position(15, 19),
            sprites["r07"].clone().set_position(0, 19),
            sprites["w22"].clone().set_position(8, 21),
            sprites["m10"].clone().set_position(0, 22),
            sprites["m06"].clone().set_position(16, 22),
            sprites["r08"].clone().set_position(16, 24),
            sprites["r08"].clone().set_position(2, 24),
        ],
        grid_size=(26, 26),
        data={
            "mvl": 66,
            "rwc": [
                ("d10", 2, 4, 1, 1),
                ("r05", 2, 18, 1, 1),
                ("db08", 5, 16, -1, 2),
                ("o08", 7, 0, 1, 1),
                ("o06", 7, 14, 1, 1),
                ("dm10", 10, 12, -1, 2),
                ("r06", 13, 0, 1, 1),
                ("r04", 13, 10, 1, 1),
                ("r06", 13, 18, 1, 1),
                ("dp08", 16, 2, 1, 2),
                ("r06", 16, 16, 1, 2),
                ("b08", 19, 15, -1, 2),
                ("r07", 19, 0, -1, 2),
                ("m10", 22, 0, 1, 1),
                ("m06", 22, 16, 1, 1),
                ("r08", 24, 16, -1, 2),
                ("r08", 24, 2, -1, 2),
            ],
            "gw": 26,
            "gh": 26,
            "px": 12,
            "py": 0,
        },
        name="lvl3",
    ),
    Level(
        sprites=[
            sprites["plr"].clone().set_position(14, 0),
            sprites["gol"].clone().set_position(15, 29),
            sprites["d12"].clone().set_position(0, 2),
            sprites["d06"].clone().set_position(16, 2),
            sprites["w32"].clone().set_position(12, 4),
            sprites["w22"].clone().set_position(22, 4),
            sprites["db08"].clone().set_position(20, 5),
            sprites["d06"].clone().set_position(4, 5),
            sprites["d08"].clone().set_position(0, 7),
            sprites["r06"].clone().set_position(12, 7),
            sprites["d06"].clone().set_position(22, 7),
            sprites["w42"].clone().set_position(0, 9),
            sprites["w31"].clone().set_position(10, 9),
            sprites["w42"].clone().set_position(20, 9),
            sprites["do10"].clone().set_position(16, 10),
            sprites["do08"].clone().set_position(0, 10),
            sprites["dm10"].clone().set_position(2, 13),
            sprites["dm08"].clone().set_position(18, 13),
            sprites["w22"].clone().set_position(3, 15),
            sprites["w22"].clone().set_position(11, 15),
            sprites["w22"].clone().set_position(19, 15),
            sprites["w22"].clone().set_position(25, 15),
            sprites["dp08"].clone().set_position(18, 16),
            sprites["dp06"].clone().set_position(4, 16),
            sprites["d10"].clone().set_position(0, 18),
            sprites["d08"].clone().set_position(16, 18),
            sprites["w31"].clone().set_position(6, 20),
            sprites["w31"].clone().set_position(16, 20),
            sprites["db08"].clone().set_position(20, 21),
            sprites["db06"].clone().set_position(6, 21),
            sprites["do10"].clone().set_position(2, 23),
            sprites["do08"].clone().set_position(18, 23),
            sprites["w42"].clone().set_position(0, 25),
            sprites["w22"].clone().set_position(12, 25),
            sprites["w42"].clone().set_position(24, 25),
            sprites["d10"].clone().set_position(16, 26),
            sprites["d08"].clone().set_position(0, 26),
            sprites["r08"].clone().set_position(0, 28),
            sprites["r06"].clone().set_position(12, 28),
            sprites["r08"].clone().set_position(22, 28),
        ],
        grid_size=(30, 30),
        data={
            "mvl": 90,
            "rwc": [
                ("d12", 2, 0, 1, 2),
                ("d06", 2, 16, 1, 2),
                ("db08", 5, 20, -1, 3),
                ("d06", 5, 4, -1, 3),
                ("d08", 7, 0, 1, 2),
                ("r06", 7, 12, 1, 2),
                ("d06", 7, 22, 1, 2),
                ("do10", 10, 16, -1, 2),
                ("do08", 10, 0, -1, 2),
                ("dm10", 13, 2, 1, 3),
                ("dm08", 13, 18, 1, 3),
                ("dp08", 16, 18, -1, 2),
                ("dp06", 16, 4, -1, 2),
                ("d10", 18, 0, 1, 3),
                ("d08", 18, 16, 1, 3),
                ("db08", 21, 20, -1, 2),
                ("db06", 21, 6, -1, 2),
                ("do10", 23, 2, 1, 3),
                ("do08", 23, 18, 1, 3),
                ("d10", 26, 16, -1, 2),
                ("d08", 26, 0, -1, 2),
                ("r08", 28, 0, 1, 3),
                ("r06", 28, 12, 1, 3),
                ("r08", 28, 22, 1, 3),
            ],
            "gw": 30,
            "gh": 30,
            "px": 14,
            "py": 0,
        },
        name="lvl4",
    ),
]


class GameHUD(RenderableUserDisplay):
    def __init__(self, game: "Td59") -> None:
        self._game = game

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        if self._game.xfn:
            return frame

        move_limit = self._game.mvl
        moves_used = self._game.mvu
        if move_limit > 0:
            remaining_ratio = max(0.0, (move_limit - moves_used) / move_limit)
            filled_columns = round(64 * remaining_ratio)
            for col in range(64):
                frame[0, col] = 14 if col < filled_columns else 3

        for i in range(MAX_LIVES):
            pip_col = 58 + i * 2
            frame[1, pip_col] = 8 if i < self._game.lvs else 3

        return frame


class Td59(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self.hud = GameHUD(self)

        self.lvs = MAX_LIVES
        self.mvu = 0
        self.mvl = 0
        self.xfn = False
        self.row_data: List[dict] = []
        self._consecutive_reset_count = 0
        self._undo_stack: List[dict] = []
        self._pit_reset: bool = False

        camera = Camera(
            0,
            0,
            16,
            16,
            BACKGROUND_COLOR,
            PADDING_COLOR,
            [self.hud],
        )

        super().__init__(
            "td59",
            levels,
            camera,
            available_actions=[0, 1, 2, 3, 4, 6, 7],
        )

    def handle_reset(self) -> None:
        self._consecutive_reset_count += 1
        if self._consecutive_reset_count >= 2:
            self.full_reset()
        else:
            self._retry_level()

    def _retry_level(self) -> None:
        self.lvs = MAX_LIVES
        self.xfn = False
        self._available_actions = [0, 1, 2, 3, 4, 6, 7]
        super().level_reset()

    def full_reset(self) -> None:
        self.lvs = MAX_LIVES
        self.xfn = False
        self._consecutive_reset_count = 0
        self._available_actions = [0, 1, 2, 3, 4, 6, 7]
        super().full_reset()

    def level_reset(self) -> None:
        if not self._pit_reset:
            self.lvs = MAX_LIVES
        self.xfn = False
        super().level_reset()

    def on_set_level(self, level: Level) -> None:
        self._cache_level_sprites()
        self._configure_camera()
        self._build_row_data()
        self._cache_spawn()
        self.mvl = self.current_level.get_data("mvl")
        self.mvu = 0
        self.xfn = False
        if not self._pit_reset:
            self.lvs = MAX_LIVES
        self._undo_stack = []

    def _cache_level_sprites(self) -> None:
        self.plr = self.current_level.get_sprites_by_tag("plr")[0]
        self.gol = self.current_level.get_sprites_by_tag("gol")[0]
        self.walls = self.current_level.get_sprites_by_tag("wal")

    def _configure_camera(self) -> None:
        self.camera.width = self.current_level.get_data("gw")
        self.camera.height = self.current_level.get_data("gh")

    def _build_row_data(self) -> None:
        grid_width = self.current_level.get_data("gw")
        obs_sprites = self.current_level.get_sprites_by_tag("row")
        rwc = self.current_level.get_data("rwc")
        unmatched = list(obs_sprites)
        self.row_data = []
        for cfg in rwc:
            _key, row_y, start_x, direction, speed = cfg
            for sp in unmatched:
                if sp.y == row_y and sp.x == start_x:
                    self.row_data.append(
                        {
                            "spr": sp,
                            "dir": direction,
                            "spd": speed,
                            "gw": grid_width,
                        }
                    )
                    unmatched.remove(sp)
                    break

    def _cache_spawn(self) -> None:
        self.spx = self.current_level.get_data("px")
        self.spy = self.current_level.get_data("py")

    def _point_overlaps_rect(
        self,
        point_x: int,
        point_y: int,
        rect_x: int,
        rect_y: int,
        rect_width: int,
        rect_height: int,
    ) -> bool:
        return (
            point_x >= rect_x
            and point_x < rect_x + rect_width
            and point_y >= rect_y
            and point_y < rect_y + rect_height
        )

    def _player_hit_obs(self) -> bool:
        px, py = self.plr.x, self.plr.y
        for rd in self.row_data:
            sp = rd["spr"]

            if sp.x < 0 and sp.x + sp.width <= 1:
                continue
            if self._point_overlaps_rect(px, py, sp.x, sp.y, sp.width, sp.height):
                return True
        return False

    def _player_hit_wall(self, candidate_x: int, candidate_y: int) -> bool:
        for wall in self.walls:
            if self._point_overlaps_rect(
                candidate_x,
                candidate_y,
                wall.x,
                wall.y,
                wall.width,
                wall.height,
            ):
                return True
        return False

    def _reached_goal(self) -> bool:
        return self._point_overlaps_rect(
            self.plr.x,
            self.plr.y,
            self.gol.x,
            self.gol.y,
            self.gol.width,
            self.gol.height,
        )

    def _handle_mouse_click(self) -> None:
        raw_x = int(self.action.data.get("x", 0))
        raw_y = int(self.action.data.get("y", 0))
        coords = self.camera.display_to_grid(raw_x, raw_y)

        dx, dy = 0, 0
        if coords is not None:
            grid_x, grid_y = coords
            delta_x = grid_x - self.plr.x
            delta_y = grid_y - self.plr.y
            if delta_x != 0 or delta_y != 0:
                if abs(delta_x) >= abs(delta_y):
                    dx = 1 if delta_x > 0 else -1
                else:
                    dy = 1 if delta_y > 0 else -1

        self._execute_move(dx, dy)

    def _save_undo_state(self) -> None:
        snapshot = {
            "plr_x": self.plr.x,
            "plr_y": self.plr.y,
            "lvs": self.lvs,
            "row_positions": [(rd["spr"].x, rd["spr"].y) for rd in self.row_data],
            "row_dirs": [rd["dir"] for rd in self.row_data],
        }
        self._undo_stack.append(snapshot)
        if len(self._undo_stack) > 50:
            self._undo_stack.pop(0)

    def _apply_undo(self) -> bool:
        if not self._undo_stack:
            return False
        snapshot = self._undo_stack.pop()

        self.plr.set_position(snapshot["plr_x"], snapshot["plr_y"])
        self.lvs = snapshot["lvs"]

        for i, rd in enumerate(self.row_data):
            rx, ry = snapshot["row_positions"][i]
            rd["spr"].set_position(rx, ry)
            rd["dir"] = snapshot["row_dirs"][i]

        return True

    def _execute_move(self, dx: int, dy: int) -> None:
        grid_width = self.current_level.get_data("gw")
        grid_height = self.current_level.get_data("gh")

        candidate_x = max(0, min(grid_width - 1, self.plr.x + dx))
        candidate_y = max(0, min(grid_height - 1, self.plr.y + dy))

        if self._player_hit_wall(candidate_x, candidate_y):
            candidate_x, candidate_y = self.plr.x, self.plr.y

        self.plr.set_position(candidate_x, candidate_y)
        self._slide_rows()
        self.mvu += 1

        if self._reached_goal():
            self.next_level()
            self.complete_action()
            return

        if self._player_hit_obs():
            self._die()
            self.complete_action()
            return

        if self.mvu >= self.mvl:
            self._die()
            self.complete_action()
            return

        self.complete_action()

    def _slide_rows(self) -> None:
        for rd in self.row_data:
            sp = rd["spr"]
            direction = rd["dir"]
            speed = rd["spd"]
            grid_width = rd["gw"]
            sprite_width = sp.width

            next_x = sp.x + direction * speed

            if direction > 0 and next_x >= grid_width:
                next_x = -sprite_width + (next_x - grid_width)
            elif direction < 0 and next_x + sprite_width <= 0:
                next_x = grid_width + (next_x + sprite_width)

            sp.set_position(next_x, sp.y)

    def _die(self) -> None:
        self.lvs -= 1
        if self.lvs <= 0:
            self._available_actions = [GameAction.RESET.value]
            self.lose()
            return

        self._pit_reset = True
        self.level_reset()
        self._pit_reset = False

    def step(self) -> None:
        if self.xfn:
            self.xfn = False
            for death_sprite in self.current_level.get_sprites_by_tag("dth"):
                self.current_level.remove_sprite(death_sprite)
            self.plr.set_position(self.spx, self.spy)
            self.complete_action()
            return

        if self.action.id == GameAction.RESET:
            self.complete_action()
            return

        if self.action.id == GameAction.ACTION7:
            self._apply_undo()
            self.mvu += 1
            if self.mvu >= self.mvl:
                self._die()
            self.complete_action()
            return

        if self.action.id == GameAction.ACTION6:
            self._consecutive_reset_count = 0
            self._save_undo_state()
            self._handle_mouse_click()
            return

        dx, dy = 0, 0
        aid = self.action.id.value
        if aid == 1:
            dy = -1
            self._consecutive_reset_count = 0
        elif aid == 2:
            dy = 1
            self._consecutive_reset_count = 0
        elif aid == 3:
            dx = -1
            self._consecutive_reset_count = 0
        elif aid == 4:
            dx = 1
            self._consecutive_reset_count = 0

        self._save_undo_state()
        self._execute_move(dx, dy)


class PuzzleEnvironment:
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

    _ACTION_MAP: Dict[str, GameAction] = {
        "reset": GameAction.RESET,
        "up": GameAction.ACTION1,
        "down": GameAction.ACTION2,
        "left": GameAction.ACTION3,
        "right": GameAction.ACTION4,
        "click": GameAction.ACTION6,
        "undo": GameAction.ACTION7,
    }

    _VALID_ACTIONS = list(_ACTION_MAP.keys())

    def __init__(self, seed: int = 0) -> None:
        self._engine = Td59(seed=seed)
        self._total_turns = 0
        self._done = False
        self._game_won = False
        self._game_over = False
        self._last_action_was_reset = False

    def _build_text_obs(self) -> str:
        e = self._engine
        gw = e.current_level.get_data("gw")
        gh = e.current_level.get_data("gh")

        grid = [["." for _ in range(gw)] for _ in range(gh)]

        for wall in e.walls:
            for wy in range(wall.y, wall.y + wall.height):
                for wx in range(wall.x, wall.x + wall.width):
                    if 0 <= wx < gw and 0 <= wy < gh:
                        grid[wy][wx] = "#"

        for rd in e.row_data:
            sp = rd["spr"]
            for ry in range(sp.y, sp.y + sp.height):
                for rx in range(sp.x, sp.x + sp.width):
                    if 0 <= rx < gw and 0 <= ry < gh:
                        grid[ry][rx] = "~"

        gx, gy = e.gol.x, e.gol.y
        if 0 <= gx < gw and 0 <= gy < gh:
            grid[gy][gx] = "G"

        ax, ay = e.plr.x, e.plr.y
        if 0 <= ax < gw and 0 <= ay < gh:
            grid[ay][ax] = "@"

        grid_text = "\n".join("".join(row) for row in grid)

        remaining = max(0, e.mvl - e.mvu)
        header = f"Level:{e.level_index + 1} Lives:{e.lvs} Moves:{remaining}/{e.mvl}"
        return header + "\n" + grid_text

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
        index_grid = e.camera.render(e.current_level.get_sprites())
        if index_grid is None or index_grid.size == 0:
            return None
        h, w = index_grid.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for idx in range(len(self.ARC_PALETTE)):
            mask = index_grid == idx
            rgb[mask] = self.ARC_PALETTE[idx]
        return self._encode_png(rgb)

    def _build_game_state(self, done: bool = False) -> GameState:
        e = self._engine
        valid_actions = self.get_actions() if not done else None
        return GameState(
            text_observation=self._build_text_obs(),
            image_observation=self._build_image_bytes(),
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

        click_data: Dict[str, Any] = {}
        base_action = action
        if action.startswith("click "):
            parts = action.split()
            base_action = "click"
            if len(parts) >= 3:
                try:
                    click_data = {"x": int(parts[1]), "y": int(parts[2])}
                except (ValueError, IndexError):
                    pass

        if base_action not in self._ACTION_MAP:
            raise ValueError(
                f"Invalid action '{action}'. "
                f"Must be one of {list(self._ACTION_MAP.keys())}"
            )

        self._last_action_was_reset = False
        self._total_turns += 1

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
        h, w = index_grid.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for idx in range(len(self.ARC_PALETTE)):
            mask = index_grid == idx
            rgb[mask] = self.ARC_PALETTE[idx]
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

    ACTION_LIST: List[str] = ["reset", "up", "down", "left", "right", "click", "undo"]

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
