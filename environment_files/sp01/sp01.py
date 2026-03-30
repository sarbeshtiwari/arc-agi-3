from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from arcengine import (
    ARCBaseGame,
    ActionInput,
    Camera,
    GameAction,
    GameState as EngineGameState,
    Level,
    RenderableUserDisplay,
    Sprite,
)

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
    (163, 86, 214),
]


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


EMPTY = 0
PLAYER = 1
GOAL = 2
WALL = 3
STICKY = 5
DONE = 6
ARROW = 7
TOGGLE = 8
SWITCH = 9
AGENT2 = 9

BACKGROUND_COLOR = 4
PADDING_COLOR = 4
WALL_RENDER = 0
PROGRESS_FILLED = 3
PROGRESS_EMPTY = 10
LIFE_FULL = 3
LIFE_LOST = 2
GAME_OVER_BG = 0
GAME_OVER_FG = 2

BAR_MAX = 14

MAX_LIVES = 3

_GLYPHS: dict[str, list[list[int]]] = {
    "G": [
        [1, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
    ],
    "A": [
        [0, 1, 0],
        [1, 0, 1],
        [1, 1, 1],
        [1, 0, 1],
        [1, 0, 1],
    ],
    "M": [
        [1, 0, 1],
        [1, 1, 1],
        [1, 0, 1],
        [1, 0, 1],
        [1, 0, 1],
    ],
    "E": [
        [1, 1, 1],
        [1, 0, 0],
        [1, 1, 0],
        [1, 0, 0],
        [1, 1, 1],
    ],
    "O": [
        [0, 1, 0],
        [1, 0, 1],
        [1, 0, 1],
        [1, 0, 1],
        [0, 1, 0],
    ],
    "V": [
        [1, 0, 1],
        [1, 0, 1],
        [1, 0, 1],
        [1, 0, 1],
        [0, 1, 0],
    ],
    "R": [
        [1, 1, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 0, 1],
        [1, 0, 1],
    ],
}

STEP_BUDGET = [4, 6, 12, 8]

DIR_UP = (0, -1)
DIR_DOWN = (0, 1)
DIR_LEFT = (-1, 0)
DIR_RIGHT = (1, 0)

DIR_MAP = {
    "up": DIR_UP,
    "down": DIR_DOWN,
    "left": DIR_LEFT,
    "right": DIR_RIGHT,
}

_tile = Sprite(
    pixels=[[1]],
    name="tile",
    visible=True,
    collidable=True,
)

_MAP_1 = [
    "W W W W W W W W W W W W W W W W",
    "W W W W W W W W W W W W W W W W",
    "W W W W W W W W W W W W W W W W",
    "W W W W W W W W W W W W W W W W",
    "W W W W . . . . . . . . W W W W",
    "W W W W . P . . W . . . W W W W",
    "W W W W . . . . W . . . W W W W",
    "W W W W . . . . . . . G W W W W",
    "W W W W . W W . . . . . W W W W",
    "W W W W . . . . . W . . W W W W",
    "W W W W . . . . . W . . W W W W",
    "W W W W . G . . . . . . W W W W",
    "W W W W W W W W W W W W W W W W",
    "W W W W W W W W W W W W W W W W",
    "W W W W W W W W W W W W W W W W",
    "W W W W W W W W W W W W W W W W",
]

_MAP_2 = [
    "W W W W W W W W W W W W W W W W",
    "W W W W W W W W W W W W W W W W",
    "W W . . . . . . . . . W . . W W",
    "W W . P . . . W . . . . . . W W",
    "W W . . . . W . . . . . . . W W",
    "W W . . . . . . . G W . . . W W",
    "W W . . . . . . . . . . . . W W",
    "W W . . . . Au . . . . . W . W W",
    "W W . W W . . . . . . . . . W W",
    "W W . . . . . . . W . . . . W W",
    "W W . . . . . . . . . . . . W W",
    "W W . W . G . . . . . . . . W W",
    "W W . . . W . . W . . . . . W W",
    "W W . . . . . . . . . . . . W W",
    "W W W W W W W W W W W W W W W W",
    "W W W W W W W W W W W W W W W W",
]

_MAP_3 = [
    "W W W W W W W W W W W W W W W W",
    "W . . . . . . . . . . . . . . W",
    "W . P . . W . . . . . . . . . W",
    "W . . . . . . . . . . . W . . W",
    "W . . . . . . . . T . . G . . W",
    "W . . . . . W . . T . . . . . W",
    "W . . . . . . . . T . . . W . W",
    "W . . . . . . . W . . . . . . W",
    "W . . . . . . W . . . . . . . W",
    "W W . . W . . . . . . . . . . W",
    "W . W . . . . . . . . . . . . W",
    "W . . X . . . . . . . . . . . W",
    "W G . . . . . . . . . . . . . W",
    "W . . W . . . . . . . . . . . W",
    "W . . . . . . . . . . . . . . W",
    "W W W W W W W W W W W W W W W W",
]

_MAP_4 = [
    "W W W W W W W W W W W W W W W W",
    "W . . . . . . . . . . . . . . W",
    "W . P . . . . . . . 2 . W . . W",
    "W . . . . W . W . . . . . . . W",
    "W . . . . . . . . . . W . . . W",
    "W . W . . . . . . . . . . . . W",
    "W . . . . . . . . W . . . . . W",
    "W . . W . . . . . . . . . W . W",
    "W . . . . . . . W . . . . . . W",
    "W . . . . . . . . G W . . . . W",
    "W . . . W . . . . . . . . . . W",
    "W . . . . . W . . . . . . . . W",
    "W . . . . . . . . . . . . . . W",
    "W . . . . . . . . . . . . . . W",
    "W . . . . . . . . . . . . . G W",
    "W W W W W W W W W W W W W W W W",
]

ALL_MAPS = [_MAP_1, _MAP_2, _MAP_3, _MAP_4]

_MULTI_AGENT_LEVELS = {3}


def _parse_map(
    rows: list[str],
) -> tuple[
    dict[tuple[int, int], int],
    dict[tuple[int, int], str],
    list[tuple[int, int]],
    list[tuple[int, int]],
    int,
    int,
]:
    grid: dict[tuple[int, int], int] = {}
    arrow_dirs: dict[tuple[int, int], str] = {}
    players: list[tuple[int, int]] = []
    agents: list[tuple[int, int]] = []

    token_rows = [r.split() for r in rows]
    h = len(token_rows)
    w = max(len(tr) for tr in token_rows)

    for y, trow in enumerate(token_rows):
        for x, tok in enumerate(trow):
            t = tok.strip()
            if t == ".":
                grid[(x, y)] = EMPTY
            elif t == "W":
                grid[(x, y)] = WALL
            elif t == "P":
                grid[(x, y)] = EMPTY
                players.append((x, y))
            elif t == "G":
                grid[(x, y)] = GOAL
            elif t == "S":
                grid[(x, y)] = STICKY
            elif t == "T":
                grid[(x, y)] = TOGGLE
            elif t == "X":
                grid[(x, y)] = SWITCH
            elif t == "2":
                grid[(x, y)] = EMPTY
                agents.append((x, y))
            elif t.startswith("A") and len(t) == 2:
                direction_char = t[1]
                dmap = {"u": "up", "d": "down", "l": "left", "r": "right"}
                dname = dmap.get(direction_char, "up")
                grid[(x, y)] = ARROW
                arrow_dirs[(x, y)] = dname
            else:
                grid[(x, y)] = EMPTY

    return grid, arrow_dirs, players, agents, w, h


class GameHUD(RenderableUserDisplay):
    def __init__(self, game: "Sp01") -> None:
        self.game = game

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        g = self.game

        lives_lost = MAX_LIVES - g._lives
        box_xs = [width // 2 - 3, width // 2, width // 2 + 3]
        for i, bx in enumerate(box_xs):
            if 0 <= bx < width:
                frame[0, bx] = LIFE_LOST if i < lives_lost else LIFE_FULL

        budget = g._get_step_budget()
        total = min(3 * budget, BAR_MAX)
        filled = min(g._moves, total)
        bar_y = height - 1
        bar_start_x = (width - total) // 2
        if bar_start_x < 1:
            bar_start_x = 1
        for i in range(total):
            bx = bar_start_x + i
            if bx >= width - 1:
                break
            frame[bar_y, bx] = PROGRESS_FILLED if i < filled else PROGRESS_EMPTY

        return frame


class Sp01(ARCBaseGame):

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self._level_idx: int = 0

        self._grid: dict[tuple[int, int], int] = {}
        self._arrow_dirs: dict[tuple[int, int], str] = {}
        self._sprite_at: dict[tuple[int, int], object] = {}

        self._agents: list[tuple[int, int]] = []
        self._agent_colors: list[int] = []

        self._goals: set[tuple[int, int]] = set()
        self._collected: set[tuple[int, int]] = set()

        self._toggles_solid: bool = True

        self._moves: int = 0

        self._lives: int = MAX_LIVES
        self._game_over: bool = False

        self.hud = GameHUD(self)

        _levels = [
            Level(sprites=[], grid_size=(16, 16), data={"idx": i})
            for i in range(len(ALL_MAPS))
        ]

        camera = Camera(0, 0, 16, 16, BACKGROUND_COLOR, PADDING_COLOR,
                        [self.hud])

        super().__init__(
            game_id="sp01",
            levels=_levels,
            camera=camera,
            available_actions=[0, 1, 2, 3, 4],
            seed=seed,
        )

    def _place(self, x: int, y: int, color: int) -> None:
        sprite = _tile.clone().color_remap(None, color).set_position(x, y)
        self.current_level.add_sprite(sprite)
        self._sprite_at[(x, y)] = sprite

    def _remove_sprite(self, x: int, y: int) -> None:
        pos = (x, y)
        if pos in self._sprite_at:
            self.current_level.remove_sprite(self._sprite_at[pos])
            del self._sprite_at[pos]

    def _get_step_budget(self) -> int:
        if 0 <= self._level_idx < len(STEP_BUDGET):
            return STEP_BUDGET[self._level_idx]
        return 1

    def _render_all(self) -> None:
        for s in list(self.current_level.get_sprites()):
            self.current_level.remove_sprite(s)
        self._sprite_at.clear()

        w, h = self.current_level.grid_size

        for y in range(h):
            for x in range(w):
                pos = (x, y)
                cell = self._grid.get(pos, EMPTY)

                if cell == WALL:
                    self._place(x, y, WALL_RENDER)
                elif cell == GOAL:
                    if pos in self._collected:
                        self._place(x, y, DONE)
                    else:
                        self._place(x, y, GOAL)
                elif cell == ARROW:
                    self._place(x, y, ARROW)
                elif cell == TOGGLE:
                    if self._toggles_solid:
                        self._place(x, y, WALL_RENDER)

        for i, (ax, ay) in enumerate(self._agents):
            color = self._agent_colors[i] if i < len(self._agent_colors) else PLAYER
            self._remove_sprite(ax, ay)
            self._place(ax, ay, color)

    def _is_blocking(self, x: int, y: int) -> bool:
        w, h = self.current_level.grid_size
        if x < 0 or x >= w or y < 0 or y >= h:
            return True
        cell = self._grid.get((x, y), EMPTY)
        if cell == WALL:
            return True
        if cell == TOGGLE and self._toggles_solid:
            return True
        return False

    def _is_stopping(self, x: int, y: int) -> bool:
        return self._grid.get((x, y), EMPTY) == STICKY

    def _get_arrow_dir(self, x: int, y: int) -> Optional[str]:
        if self._grid.get((x, y)) == ARROW:
            return self._arrow_dirs.get((x, y))
        return None

    def _is_switch(self, x: int, y: int) -> bool:
        return self._grid.get((x, y)) == SWITCH

    def _is_goal(self, x: int, y: int) -> bool:
        return self._grid.get((x, y)) == GOAL

    def _slide_agent(self, idx: int, dx: int, dy: int) -> None:
        ax, ay = self._agents[idx]
        cur_dx, cur_dy = dx, dy

        while True:
            nx, ny = ax + cur_dx, ay + cur_dy

            if self._is_blocking(nx, ny):
                break

            occupied_by_other = False
            for j, (ox, oy) in enumerate(self._agents):
                if j != idx and ox == nx and oy == ny:
                    occupied_by_other = True
                    break
            if occupied_by_other:
                break

            ax, ay = nx, ny

            if self._level_idx not in _MULTI_AGENT_LEVELS:
                if self._is_goal(ax, ay) and (ax, ay) not in self._collected:
                    self._collected.add((ax, ay))

            if self._is_switch(ax, ay):
                self._toggles_solid = not self._toggles_solid

            if self._is_stopping(ax, ay):
                break

            arrow_dir = self._get_arrow_dir(ax, ay)
            if arrow_dir is not None:
                ndx, ndy = DIR_MAP.get(arrow_dir, (cur_dx, cur_dy))
                cur_dx, cur_dy = ndx, ndy
                continue

        self._agents[idx] = (ax, ay)

    def _check_win(self) -> bool:
        if self._level_idx in _MULTI_AGENT_LEVELS:
            agent_positions = set(self._agents)
            for g in self._goals:
                if g not in agent_positions:
                    return False
            for a in self._agents:
                if a not in self._goals:
                    return False
            return True
        else:
            return self._goals.issubset(self._collected)

    def _build_level(self, idx: int) -> None:
        raw_map = ALL_MAPS[idx]
        grid, arrow_dirs, players, extra_agents, mw, mh = _parse_map(raw_map)

        self._grid = grid
        self._arrow_dirs = arrow_dirs
        self._toggles_solid = True
        self._moves = 0
        self._collected = set()

        self._agents = list(players)
        self._agent_colors = [PLAYER] * len(players)
        for ea in extra_agents:
            self._agents.append(ea)
            self._agent_colors.append(AGENT2)

        self._goals = set()
        for pos, cell in grid.items():
            if cell == GOAL:
                self._goals.add(pos)

        self._render_all()

    def on_set_level(self, level: Level) -> None:
        for s in list(level.get_sprites()):
            level.remove_sprite(s)
        self._sprite_at.clear()

        idx = level.get_data("idx")
        if idx is None:
            idx = 0
        self._level_idx = idx

        if self._level_idx >= len(ALL_MAPS):
            return

        self._build_level(self._level_idx)

    def _show_game_over_screen(self) -> None:
        w, h = self.current_level.grid_size

        for y in range(1, h - 1):
            for x in range(1, w - 1):
                self._remove_sprite(x, y)
                self._place(x, y, GAME_OVER_BG)

        def _draw_word(word: str, row_start: int) -> None:
            col_start = (w - len(word) * 3) // 2
            for li, ch in enumerate(word):
                glyph = _GLYPHS.get(ch, _GLYPHS["E"])
                lx = col_start + li * 3
                for gy, row in enumerate(glyph):
                    for gx, pixel in enumerate(row):
                        if pixel:
                            px = lx + gx
                            py = row_start + gy
                            if 0 < px < w - 1 and 0 < py < h - 1:
                                self._remove_sprite(px, py)
                                self._place(px, py, GAME_OVER_FG)

        _draw_word("GAME", 3)
        _draw_word("OVER", 9)

    def _handle_move(self, direction: str) -> None:
        if self._game_over:
            return

        dx, dy = DIR_MAP.get(direction, (0, 0))
        if dx == 0 and dy == 0:
            return

        old_positions = list(self._agents)

        for i in range(len(self._agents)):
            self._slide_agent(i, dx, dy)

        if self._agents == old_positions:
            return

        self._moves += 1
        self._render_all()

        if self._check_win():
            self.next_level()
            return

        budget = self._get_step_budget()
        total = min(3 * budget, BAR_MAX)
        if self._moves >= total:
            self._lives -= 1
            if self._lives == 0:
                self._game_over = True
                self._show_game_over_screen()
                self.lose()
            else:
                self._build_level(self._level_idx)

    def step(self) -> None:
        action_id = self.action.id

        if action_id == GameAction.ACTION1:
            self._handle_move("up")
        elif action_id == GameAction.ACTION2:
            self._handle_move("down")
        elif action_id == GameAction.ACTION3:
            self._handle_move("left")
        elif action_id == GameAction.ACTION4:
            self._handle_move("right")

        self.complete_action()


_ACTION_MAP: dict[str, GameAction] = {
    "up": GameAction.ACTION1,
    "down": GameAction.ACTION2,
    "left": GameAction.ACTION3,
    "right": GameAction.ACTION4,
}

_VALID_ACTIONS: list[str] = ["reset", "up", "down", "left", "right"]


class PuzzleEnvironment:

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self._engine = Sp01(seed=seed)
        self._total_turns = 0
        self._level_index = 0
        self._done = False
        self._last_outcome = "active"
        self._consecutive_resets = 0
        self._played_since_reset = False

    def reset(self) -> GameState:
        if self._played_since_reset:
            self._consecutive_resets = 0

        target_level = self._level_index
        if self._consecutive_resets >= 1:
            target_level = 0

        self._engine = Sp01(seed=self._seed)
        self._engine.set_level(target_level)
        self._level_index = target_level
        self._total_turns = 0
        self._done = False
        self._last_outcome = "active"
        self._consecutive_resets += 1
        self._played_since_reset = False
        return self._build_game_state()

    def step(self, action: str) -> StepResult:
        action = action.strip().lower()
        if action == "reset":
            state = self.reset()
            return StepResult(state=state, reward=0.0, done=False, info={"outcome": "reset"})

        if self._done:
            return StepResult(
                state=self._build_game_state(),
                reward=0.0,
                done=True,
                info={"error": "Episode finished. Call reset()."},
            )

        if action not in _ACTION_MAP:
            raise ValueError(f"Unknown action '{action}'. Valid: {_VALID_ACTIONS}")

        self._consecutive_resets = 0
        self._played_since_reset = True

        prev_level = self._engine._level_idx
        total_levels = len(self._engine._levels)

        game_action = _ACTION_MAP[action]
        self._engine.perform_action(ActionInput(id=game_action))
        self._total_turns += 1

        self._level_index = self._engine._level_idx

        engine_state = self._engine._state
        reward = 0.0
        done = False
        info: dict = {"action": action, "outcome": "active"}

        if engine_state == EngineGameState.WIN:
            reward = 1.0 / total_levels
            done = True
            info["outcome"] = "game_complete"
        elif engine_state == EngineGameState.GAME_OVER:
            done = True
            info["outcome"] = "game_over"
        elif self._engine._level_idx != prev_level:
            reward = 1.0 / total_levels
            info["outcome"] = "level_complete"

        self._done = done
        self._last_outcome = info["outcome"]
        state = self._build_game_state()
        return StepResult(state=state, reward=reward, done=done, info=info)

    def get_actions(self) -> List[str]:
        if self._done:
            return ["reset"]
        return list(_VALID_ACTIONS)

    def is_done(self) -> bool:
        return self._done

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError(f"Unsupported render mode: {mode}")
        index_grid = self._engine.camera.render(self._engine.current_level.get_sprites())
        h, w = index_grid.shape[0], index_grid.shape[1]
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for idx, color in enumerate(ARC_PALETTE):
            rgb[index_grid == idx] = color
        return rgb

    def close(self) -> None:
        self._engine = None

    def _build_text_observation(self) -> str:
        g = self._engine
        w, h = g.current_level.grid_size
        total_levels = len(g._levels)
        tile_map = {0: '.', 2: 'G', 3: 'W', 5: 'S', 6: 'D', 7: 'A', 9: 'X'}
        lines = []
        lines.append(f"Level:{g._level_idx+1}/{total_levels} Moves:{g._moves} Lives:{g._lives}/{MAX_LIVES}")
        if g._game_over:
            lines.append("GAME_OVER")
        for y in range(h):
            row = ""
            for x in range(w):
                is_agent = False
                for i, (ax, ay) in enumerate(g._agents):
                    if ax == x and ay == y:
                        row += 'P' if i == 0 else '2'
                        is_agent = True
                        break
                if not is_agent:
                    cell = g._grid.get((x, y), 0)
                    if cell == 2 and (x, y) in g._collected:
                        row += 'D'
                    elif cell == 8:
                        row += 'W' if g._toggles_solid else '.'
                    else:
                        row += tile_map.get(cell, '.')
            lines.append(row)
        return "\n".join(lines)

    def _build_game_state(self) -> GameState:
        return GameState(
            text_observation=self._build_text_observation(),
            image_observation=None,
            valid_actions=self.get_actions(),
            turn=self._total_turns,
            metadata={
                "total_levels": len(self._engine._levels),
                "level": self._engine._level_idx + 1,
                "moves_used": self._engine._moves,
                "lives": self._engine._lives,
                "game_over": self._engine._game_over,
                "levels_completed": getattr(self._engine, "_score", 0),
                "level_index": self._engine._level_idx,
            },
        )


class ArcGameEnv(gym.Env):

    metadata: Dict[str, Any] = {
        "render_modes": ["rgb_array"],
        "render_fps": 10,
    }

    ACTION_LIST: List[str] = [
        "reset",
        "up",
        "down",
        "left",
        "right",
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
