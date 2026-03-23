"""
Game Engine Service - Manages game instances and executes game logic.

Two modes:
1. Native mode: If arcengine is installed, use it directly via arc_agi.Arcade
2. Direct mode: Dynamically load and execute the game .py file using arcengine
   classes that must be installed (pip install arc-agi)

If arcengine is not installed at all, falls back to a minimal stub that
attempts to load the game module anyway (will fail if game imports arcengine).
"""

import importlib.util
import os
import re
import sys
import time
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)
from pathlib import Path
from typing import Any

# Try to import arcengine
try:
    import arc_agi
    from arcengine import GameAction
    HAS_ARCENGINE = True
except ImportError:
    HAS_ARCENGINE = False


# ARC-AGI-3 color palette (0-15) as RGB tuples
# Source: app.py -> _PALETTE_HEX
ARC_COLORS = {
    0: (255, 255, 255),  # White
    1: (204, 204, 204),  # Off-white
    2: (153, 153, 153),  # Light Grey
    3: (102, 102, 102),  # Grey
    4: (51, 51, 51),     # Dark Grey
    5: (0, 0, 0),        # Black
    6: (229, 58, 163),   # Magenta
    7: (255, 123, 204),  # Pink
    8: (249, 60, 49),    # Red
    9: (30, 147, 255),   # Blue
    10: (136, 216, 241), # Light Blue
    11: (255, 220, 0),   # Yellow
    12: (255, 133, 27),  # Orange
    13: (146, 18, 49),   # Maroon
    14: (79, 204, 48),   # Green
    15: (163, 86, 214),  # Purple
}


@dataclass
class GameFrame:
    grid: list[list[int]]
    width: int
    height: int
    state: str  # NOT_FINISHED, WIN, GAME_OVER
    score: float
    level: int
    total_actions: int
    available_actions: list[str]
    metadata: dict | None = None


@dataclass
class LevelStats:
    """Stats for a single level attempt."""
    level: int = 0
    actions: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    completed: bool = False
    lives_used: int = 0
    game_overs: int = 0
    resets: int = 0

    @property
    def elapsed_seconds(self) -> float:
        end = self.end_time if self.end_time > 0 else time.time()
        return round(end - self.start_time, 2) if self.start_time > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "actions": self.actions,
            "time": self.elapsed_seconds,
            "completed": self.completed,
            "lives_used": self.lives_used,
            "game_overs": self.game_overs,
            "resets": self.resets,
        }


@dataclass
class GameInstance:
    session_id: str
    game_id: str
    game_code: str
    game_dir: str
    seed: int = 0
    current_level: int = 0
    state: str = "NOT_FINISHED"
    score: float = 0.0
    total_actions: int = 0
    actions_since_reset: int = 0  # track moves since last reset
    action_log: list = field(default_factory=list)
    # Per-level tracking
    _current_level_stats: Any = None
    _completed_levels: list = field(default_factory=list)  # list of LevelStats
    _game_start_time: float = 0.0
    _game_end_time: float = 0.0
    # The loaded game object (ARCBaseGame subclass instance)
    _game_obj: Any = None
    # Arcade instance (for native mode)
    _engine_env: Any = None
    _engine_arcade: Any = None
    # Fallback grid when game can't be loaded
    _last_grid: list = field(default_factory=lambda: [[0] * 8 for _ in range(8)])
    _available_actions: list = field(
        default_factory=lambda: [
            "ACTION1", "ACTION2", "ACTION3", "ACTION4",
            "ACTION5", "ACTION6", "ACTION7",
        ]
    )
    _mode: str = "none"  # "arcade", "direct", "none"
    _last_activity: float = 0.0


MAX_CONCURRENT_SESSIONS = 600
SESSION_IDLE_TIMEOUT = 1800
MAX_ACTION_LOG_SIZE = 500


class GameEngineService:

    def __init__(self, environments_dir: str):
        self.environments_dir = environments_dir
        self._instances: dict[str, GameInstance] = {}
        self._lock = __import__('threading').Lock()

    def _cleanup_idle(self):
        now = time.time()
        with self._lock:
            stale = [
                sid for sid, inst in self._instances.items()
                if inst._last_activity > 0 and (now - inst._last_activity) > SESSION_IDLE_TIMEOUT
            ]
        for sid in stale:
            logger.debug(f"[ENGINE] Cleaning up idle session: {sid}")
            try:
                self.destroy_instance(sid)
            except Exception:
                self._instances.pop(sid, None)
        if stale:
            logger.debug(f"[ENGINE] Cleaned up {len(stale)} idle session(s). Active: {len(self._instances)}")

    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._instances)

    def create_instance(
        self, session_id: str, game_id: str, game_code: str,
        game_dir: str, seed: int = 0,
    ) -> GameInstance:
        """Create a new game instance and initialize it."""
        self._cleanup_idle()

        if len(self._instances) >= MAX_CONCURRENT_SESSIONS:
            raise RuntimeError(
                f"Server at capacity ({MAX_CONCURRENT_SESSIONS} concurrent games). Try again later."
            )

        instance = GameInstance(
            session_id=session_id,
            game_id=game_id,
            game_code=game_code,
            game_dir=game_dir,
            seed=seed,
        )

        # Strategy 1: Try our direct load first (fixes @dataclass bug in arc_agi)
        # This is more reliable than arc_agi.Arcade for games using @dataclass
        if HAS_ARCENGINE:
            self._try_direct_load(instance)
            if instance._mode == "direct":
                with self._lock:
                    self._instances[session_id] = instance
                now = time.time()
                instance._last_activity = now
                instance._game_start_time = now
                instance._current_level_stats = LevelStats(
                    level=instance.current_level, start_time=now
                )
                return instance

            # Strategy 2: Fall back to arc_agi.Arcade
            try:
                instance._engine_arcade = arc_agi.Arcade(
                    environments_dir=self.environments_dir
                )
                instance._engine_env = instance._engine_arcade.make(
                    game_id, seed=seed
                )
                test_obs = instance._engine_env.step(GameAction.ACTION1)
                if test_obs is not None:
                    instance._last_grid = self._extract_grid_from_frame(test_obs)
                    if hasattr(instance._engine_env, "action_space"):
                        instance._available_actions = [
                            str(a) for a in instance._engine_env.action_space
                        ]
                    instance._mode = "arcade"
                    grid = instance._last_grid
                    h = len(grid)
                    w = len(grid[0]) if h > 0 else 0
                    logger.debug(f"[ENGINE] {game_id}: loaded via Arcade grid={w}x{h}")
                else:
                    raise RuntimeError("Arcade returned None - game didn't load")
            except Exception as e:
                logger.debug(f"[ENGINE] Arcade also failed for {game_id}: {e}")
                if instance._mode == "none":
                    self._fallback_parse(instance)
        else:
            # Strategy 2: Try loading the game module directly
            self._try_direct_load(instance)

        with self._lock:
            self._instances[session_id] = instance
        instance._last_activity = time.time()

        now = time.time()
        instance._game_start_time = now
        instance._current_level_stats = LevelStats(
            level=instance.current_level, start_time=now
        )

        return instance

    def _try_direct_load(self, instance: GameInstance):
        """
        Try to dynamically import and instantiate the game class directly.
        This fixes the arc_agi exec() bug where @dataclass fails because
        the module isn't registered in sys.modules.
        """
        game_py = os.path.join(instance.game_dir, f"{instance.game_code}.py")
        if not os.path.exists(game_py):
            logger.debug(f"[ENGINE] Game file not found: {game_py}")
            self._fallback_parse(instance)
            return

        try:
            # Add game dir to sys.path so imports within the game work
            game_dir = str(Path(instance.game_dir).resolve())
            if game_dir not in sys.path:
                sys.path.insert(0, game_dir)

            # Create a unique module name and register it in sys.modules
            # BEFORE exec - this fixes @dataclass which needs sys.modules lookup
            module_name = f"_arcgame_{instance.game_code}_{instance.session_id[:8]}"
            
            spec = importlib.util.spec_from_file_location(module_name, game_py)
            module = importlib.util.module_from_spec(spec)
            
            # Register in sys.modules BEFORE exec (fixes @dataclass)
            sys.modules[module_name] = module
            module.__module__ = module_name
            
            try:
                spec.loader.exec_module(module)
            except Exception as load_err:
                # Clean up on failure
                sys.modules.pop(module_name, None)
                raise load_err

            game_class = None
            expected_name = instance.game_code.capitalize()

            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, type) and attr_name != "ARCBaseGame":
                    if hasattr(attr, "step") and hasattr(attr, "complete_action"):
                        game_class = attr
                        break

            if game_class is None:
                logger.debug(f"[ENGINE] No game class found in {game_py}")
                self._fallback_parse(instance)
                return

            try:
                game_obj = game_class(seed=instance.seed)
            except TypeError:
                game_obj = game_class()

            instance._game_obj = game_obj
            instance._mode = "direct"

            try:
                from arcengine import ActionInput
                reset_action = ActionInput(id=GameAction.RESET)
                frame_data = game_obj.perform_action(reset_action, raw=True)
                if frame_data is not None:
                    instance._last_grid = self._extract_grid_from_frame(frame_data)
                    self._update_state_from_frame(instance, frame_data)
                else:
                    self._extract_grid_from_game(instance)
            except Exception as e:
                logger.debug(f"[ENGINE] Initial reset failed, falling back to grid extraction: {e}")
                self._extract_grid_from_game(instance)

            self._extract_actions_from_file(game_py, instance)

            grid = instance._last_grid
            h = len(grid)
            w = len(grid[0]) if h > 0 else 0
            logger.debug(f"[ENGINE] {instance.game_id}: loaded directly ({game_class.__name__}) grid={w}x{h}")

        except Exception as e:
            logger.debug(f"[ENGINE] Direct load failed for {instance.game_id}: {e}")
            logger.debug("", exc_info=True)
            self._fallback_parse(instance)

    def _extract_grid_from_game(self, instance: GameInstance):
        game = instance._game_obj
        if game is None:
            return

        try:
            # Try common ways to get the grid from an ARCBaseGame
            if hasattr(game, "camera") and hasattr(game.camera, "render"):
                grid = game.camera.render()
                if isinstance(grid, list) and len(grid) > 0:
                    instance._last_grid = grid
                    return

            if hasattr(game, "render"):
                grid = game.render()
                if isinstance(grid, list) and len(grid) > 0:
                    instance._last_grid = grid
                    return

            if hasattr(game, "current_level") and hasattr(game.current_level, "grid_size"):
                gs = game.current_level.grid_size
                w, h = gs[0], gs[1]
                # Build grid from sprites
                grid = [[0] * w for _ in range(h)]
                if hasattr(game.current_level, "_sprites"):
                    for sprite in game.current_level._sprites:
                        if hasattr(sprite, "x") and hasattr(sprite, "y") and hasattr(sprite, "pixels"):
                            sx, sy = int(sprite.x), int(sprite.y)
                            for dy, row in enumerate(sprite.pixels):
                                for dx, val in enumerate(row):
                                    gx, gy = sx + dx, sy + dy
                                    if 0 <= gx < w and 0 <= gy < h and val > 0:
                                        grid[gy][gx] = val
                instance._last_grid = grid
                return

            # Last resort: check grid_size from levels
            if hasattr(game, "levels") and len(game.levels) > 0:
                level = game.levels[0]
                if hasattr(level, "grid_size"):
                    w, h = level.grid_size
                    instance._last_grid = [[0] * w for _ in range(h)]

        except Exception as e:
            logger.debug(f"[ENGINE] Grid extraction failed: {e}")

    def _extract_actions_from_file(self, game_py: str, instance: GameInstance):
        try:
            with open(game_py, "r") as f:
                content = f.read()
            actions_found = []
            for i in range(1, 8):
                if f"ACTION{i}" in content:
                    actions_found.append(f"ACTION{i}")
            if actions_found:
                instance._available_actions = actions_found
        except Exception:
            pass

    def _fallback_parse(self, instance: GameInstance):
        """
        Last resort: parse the .py file as text to extract grid sizes and
        sprite pixel data. No execution, just text parsing.
        """
        game_py = os.path.join(instance.game_dir, f"{instance.game_code}.py")
        if not os.path.exists(game_py):
            instance._mode = "none"
            return

        try:
            with open(game_py, "r") as f:
                content = f.read()

            grid_sizes = re.findall(r"grid_size\s*=\s*\((\d+)\s*,\s*(\d+)\)", content)
            if grid_sizes:
                w, h = int(grid_sizes[0][0]), int(grid_sizes[0][1])
                instance._last_grid = [[0] * w for _ in range(h)]

            # Extract pixel data from sprite definitions
            # Look for patterns like: pixels=[ [1, 2], [3, 4] ]
            pixel_blocks = re.findall(
                r"pixels\s*=\s*\[((?:\s*\[[\d\s,\-]+\]\s*,?\s*)+)\]",
                content,
            )
            if pixel_blocks and grid_sizes:
                w, h = int(grid_sizes[0][0]), int(grid_sizes[0][1])
                grid = [[0] * w for _ in range(h)]

                place_y = 1
                for block in pixel_blocks:
                    rows = re.findall(r"\[([\d\s,\-]+)\]", block)
                    sprite_pixels = []
                    for row_str in rows:
                        vals = [int(v.strip()) for v in row_str.split(",") if v.strip().lstrip("-").isdigit()]
                        sprite_pixels.append(vals)

                    place_x = 1
                    for dy, row in enumerate(sprite_pixels):
                        for dx, val in enumerate(row):
                            gy, gx = place_y + dy, place_x + dx
                            if 0 <= gy < h and 0 <= gx < w and val > 0:
                                grid[gy][gx] = val
                    place_y += len(sprite_pixels) + 1
                    if place_y >= h:
                        break

                instance._last_grid = grid

            self._extract_actions_from_file(game_py, instance)

            instance._mode = "parsed"
            logger.debug(f"[ENGINE] {instance.game_id}: loaded via file parsing (read-only)")

        except Exception as e:
            logger.debug(f"[ENGINE] Fallback parse failed: {e}")
            instance._mode = "none"

    # ──── Step ────

    def step(
        self, session_id: str, action: str,
        x: int | None = None, y: int | None = None,
    ) -> GameFrame:
        """Execute an action on a game instance and return the new frame."""
        instance = self._instances.get(session_id)
        if not instance:
            raise ValueError(f"No game instance found for session {session_id}")

        if instance.state in ("WIN", "GAME_OVER"):
            if action == "RESET" and instance.state == "GAME_OVER":
                pass
            else:
                return self._make_frame(instance)

        if len(instance.action_log) < MAX_ACTION_LOG_SIZE:
            instance.action_log.append({
                "action": action, "x": x, "y": y,
                "timestamp": time.time(), "action_idx": instance.total_actions,
            })
        instance.total_actions += 1
        instance._last_activity = time.time()

        if action == "RESET":
            return self.reset(session_id)

        instance.actions_since_reset += 1

        if instance._current_level_stats:
            instance._current_level_stats.actions += 1

        prev_level = instance.current_level

        # Dispatch based on mode
        if instance._mode == "arcade" and instance._engine_env is not None:
            self._step_arcade(instance, action, x, y)
        elif instance._mode == "direct" and instance._game_obj is not None:
            self._step_direct(instance, action, x, y)
        else:
            # Parsed / none mode - basic simulation
            self._step_simulation(instance, action, x, y)

        if instance.current_level != prev_level and instance._current_level_stats:
            now = time.time()
            instance._current_level_stats.end_time = now
            instance._current_level_stats.completed = True
            instance._completed_levels.append(instance._current_level_stats)
            # Start new level timer
            instance._current_level_stats = LevelStats(
                level=instance.current_level, start_time=now
            )

        if instance.state in ("WIN", "GAME_OVER"):
            now = time.time()
            instance._game_end_time = now
            if instance._current_level_stats:
                instance._current_level_stats.end_time = now
                instance._current_level_stats.completed = (instance.state == "WIN")
                if instance.state == "GAME_OVER":
                    instance._current_level_stats.game_overs += 1
                    instance._current_level_stats.lives_used += 1
                # Only add if it has actions
                if instance._current_level_stats.actions > 0:
                    instance._completed_levels.append(instance._current_level_stats)
                    instance._current_level_stats = None
            # On WIN, set current_level to total completed levels
            if instance.state == "WIN":
                instance.current_level = len(instance._completed_levels)

        return self._make_frame(instance)

    def _step_arcade(self, instance: GameInstance, action: str, x, y):
        try:
            action_enum = getattr(GameAction, action)
            if action == "ACTION6" and x is not None and y is not None:
                obs = instance._engine_env.step(action_enum, data={"x": x, "y": y})
            else:
                obs = instance._engine_env.step(action_enum)

            instance._last_grid = self._get_grid_from_env(instance._engine_env)
            if hasattr(obs, "state"):
                instance.state = obs.state
            if hasattr(obs, "score"):
                instance.score = obs.score
        except Exception as e:
            logger.debug(f"[ENGINE] Arcade step error: {e}")

    def _step_direct(self, instance: GameInstance, action: str, x, y):
        game = instance._game_obj
        try:
            from arcengine import ActionInput
            action_enum = getattr(GameAction, action)
            data = {}
            if action == "ACTION6" and x is not None and y is not None:
                data = {"x": x, "y": y}

            action_input = ActionInput(id=action_enum, data=data)
            frame_data = game.perform_action(action_input, raw=True)

            if frame_data is not None:
                instance._last_grid = self._extract_grid_from_frame(frame_data)
                self._update_state_from_frame(instance, frame_data)
            else:
                # perform_action returned None - try direct step
                self._step_direct_fallback(instance, action, x, y)

        except ImportError:
            self._step_direct_fallback(instance, action, x, y)
        except Exception as e:
            logger.debug(f"[ENGINE] Direct step error: {e}")
            logger.debug("", exc_info=True)

    def _step_direct_fallback(self, instance: GameInstance, action: str, x, y):
        game = instance._game_obj
        try:
            if HAS_ARCENGINE:
                action_enum = getattr(GameAction, action)
                if hasattr(game, "set_action"):
                    data = {"x": x or 0, "y": y or 0} if action == "ACTION6" else {}
                    game.set_action(action_enum, data=data)

            game.step()
            self._extract_grid_from_game(instance)

            if hasattr(game, "state"):
                state = str(game.state).upper()
                if "WIN" in state:
                    instance.state = "WIN"
                elif "OVER" in state or "FAIL" in state:
                    instance.state = "GAME_OVER"
        except Exception as e:
            logger.debug(f"[ENGINE] Direct fallback step error: {e}")

    def _step_simulation(self, instance: GameInstance, action: str, x, y):
        import random
        grid = instance._last_grid
        h = len(grid)
        w = len(grid[0]) if h > 0 else 0

        if action == "ACTION6" and x is not None and y is not None:
            if 0 <= y < h and 0 <= x < w:
                current = grid[y][x]
                if current != 0:
                    grid[y][x] = 0
                    instance.score += 1.0
                else:
                    grid[y][x] = random.randint(1, 15)
        elif action in ("ACTION1", "ACTION2", "ACTION3", "ACTION4"):
            direction_map = {
                "ACTION1": (0, -1),
                "ACTION2": (0, 1),
                "ACTION3": (-1, 0),
                "ACTION4": (1, 0),
            }
            dx, dy = direction_map[action]
            new_grid = [[0] * w for _ in range(h)]
            for r in range(h):
                for c in range(w):
                    if grid[r][c] != 0:
                        nr, nc = r + dy, c + dx
                        if 0 <= nr < h and 0 <= nc < w:
                            new_grid[nr][nc] = grid[r][c]
            instance._last_grid = new_grid

        has_colors = any(cell != 0 for row in instance._last_grid for cell in row)
        if not has_colors and instance.total_actions > 0:
            instance.state = "WIN"
            instance.score = max(instance.score, 100.0)

    # ──── Reset ────

    def reset(self, session_id: str) -> GameFrame:
        """
        Smart reset logic:
        - If user has taken actions since last reset -> reset SAME level
        - If user has NOT moved (0 actions since reset) -> reset to LEVEL 1
        """
        instance = self._instances.get(session_id)
        if not instance:
            raise ValueError(f"No game instance found for session {session_id}")

        had_moves = instance.actions_since_reset > 0
        was_game_over = instance.state == "GAME_OVER"
        instance.actions_since_reset = 0
        instance.state = "NOT_FINISHED"

        if had_moves or was_game_over:
            logger.debug(f"[ENGINE] Reset same level (was_game_over={was_game_over})")

            if was_game_over:
                self._do_level_reset_after_gameover(instance)
            else:
                self._do_engine_reset(instance)

            resets_so_far = 0
            game_overs_so_far = 0
            lives_used_so_far = 0

            if instance._current_level_stats:
                resets_so_far = instance._current_level_stats.resets + 1
                game_overs_so_far = instance._current_level_stats.game_overs
                lives_used_so_far = instance._current_level_stats.lives_used
            elif was_game_over and instance._completed_levels:
                last = instance._completed_levels[-1]
                if not last.completed:
                    instance._completed_levels.pop()
                    resets_so_far = last.resets + 1
                    game_overs_so_far = last.game_overs
                    lives_used_so_far = last.lives_used

            instance._current_level_stats = LevelStats(
                level=instance.current_level, start_time=time.time(),
                resets=resets_so_far, game_overs=game_overs_so_far,
                lives_used=lives_used_so_far,
            )
        else:
            # No moves since last reset -> full restart from level 1
            logger.debug(f"[ENGINE] Reset to level 1 (no moves taken)")
            instance.score = 0.0
            instance.total_actions = 0
            instance.current_level = 0
            instance._completed_levels = []
            instance._game_end_time = 0.0
            now = time.time()
            instance._game_start_time = now
            self._do_full_restart(instance)
            instance._current_level_stats = LevelStats(
                level=instance.current_level, start_time=now
            )

        return self._make_frame(instance)

    def _do_level_reset_after_gameover(self, instance: GameInstance):
        game_obj = instance._game_obj
        if game_obj is None:
            self._do_engine_reset(instance)
            return

        target_level = instance.current_level
        try:
            saved_score = game_obj._score
            game_obj._levels[target_level] = game_obj._clean_levels[target_level].clone()
            game_obj.set_level(target_level)
            game_obj._score = saved_score
            game_obj._action_count = 1

            from arcengine import ActionInput
            frame_data = game_obj.perform_action(ActionInput(id=GameAction.RESET), raw=True)
            if frame_data is not None:
                instance._last_grid = self._extract_grid_from_frame(frame_data)

            instance.state = "NOT_FINISHED"
            instance.current_level = target_level
        except Exception as e:
            logger.debug(f"[ENGINE] level_reset after gameover failed: {e}")

    def _do_engine_reset(self, instance: GameInstance):
        if instance._mode == "direct" and instance._game_obj is not None:
            try:
                from arcengine import ActionInput
                reset_action = ActionInput(id=GameAction.RESET)
                frame_data = instance._game_obj.perform_action(reset_action, raw=True)
                if frame_data is not None:
                    instance._last_grid = self._extract_grid_from_frame(frame_data)
                    self._update_state_from_frame(instance, frame_data)
                    return
            except Exception as e:
                logger.debug(f"[ENGINE] Engine reset failed: {e}")
        elif instance._mode == "arcade" and instance._engine_env is not None:
            try:
                obs = instance._engine_env.step(GameAction.RESET)
                if obs is not None:
                    instance._last_grid = self._extract_grid_from_frame(obs)
                    self._update_state_from_frame(instance, obs)
                    return
            except Exception as e:
                logger.debug(f"[ENGINE] Arcade reset failed: {e}")

        # Fallback: full restart
        self._do_full_restart(instance)

    def skip_to_level(self, session_id: str, target_level: int):
        """
        Skip to a target level using the game's set_level + level_reset API.
        After set_level(), we must NOT send a RESET action because handle_reset()
        will see _action_count==0 and call full_reset() which goes back to level 0.
        Instead we call level_reset() directly to initialize the level properly.
        """
        instance = self._instances.get(session_id)
        if not instance or target_level <= 0:
            return

        game_obj = instance._game_obj
        if game_obj is None:
            return

        try:
            # set_level sets the index and calls on_set_level
            game_obj.set_level(target_level)

            # level_reset clones the clean level, re-calls set_level at same index,
            # and sets state to NOT_FINISHED -- without going back to level 0
            game_obj.level_reset()

            # Bump action_count so a subsequent RESET from user does level_reset not full_reset
            game_obj._action_count = 1

            # Get the initial grid by sending a RESET action now that action_count > 0
            # handle_reset() will see _action_count != 0 and call level_reset() (same level)
            from arcengine import ActionInput
            frame_data = game_obj.perform_action(
                ActionInput(id=GameAction.RESET), raw=True
            )
            if frame_data is not None:
                instance._last_grid = self._extract_grid_from_frame(frame_data)
                self._update_state_from_frame(instance, frame_data)

            instance.current_level = target_level
            # Reset timing for the new level
            now = time.time()
            instance._game_start_time = now
            instance._completed_levels = []
            instance._current_level_stats = LevelStats(
                level=target_level, start_time=now
            )
            logger.debug(f"[ENGINE] Skipped to level {target_level} via set_level + level_reset")
        except Exception as e:
            logger.debug(f"[ENGINE] skip_to_level failed: {e}")
            # If skip fails, game stays at whatever level it loaded at

    def _do_full_restart(self, instance: GameInstance):
        if instance._mode == "direct":
            self._try_direct_load(instance)
        elif instance._mode == "arcade":
            try:
                instance._engine_arcade = arc_agi.Arcade(
                    environments_dir=self.environments_dir
                )
                instance._engine_env = instance._engine_arcade.make(
                    instance.game_id, seed=instance.seed
                )
                instance._last_grid = self._get_grid_from_env(instance._engine_env)
            except Exception:
                self._try_direct_load(instance)
        else:
            self._fallback_parse(instance)

    # ──── Read ────

    def get_frame(self, session_id: str) -> GameFrame:
        instance = self._instances.get(session_id)
        if not instance:
            raise ValueError(f"No game instance found for session {session_id}")
        return self._make_frame(instance)

    def destroy_instance(self, session_id: str):
        with self._lock:
            instance = self._instances.pop(session_id, None)
        if instance:
            game_dir = str(Path(instance.game_dir).resolve())
            if game_dir in sys.path:
                sys.path.remove(game_dir)

    def _extract_grid_from_frame(self, frame_data) -> list[list[int]]:
        """Extract a 2D grid from a FrameDataRaw object.
        
        FrameDataRaw structure:
          .frame  = [numpy.ndarray(64,64, dtype=int8)]  (list of frames, usually 1)
          .state  = GameState.NOT_FINISHED / WIN / GAME_OVER
          .available_actions = [1, 2, 3, 4, 5]  (list of int action ids)
        """
        try:
            # FrameDataRaw.frame is a list of numpy arrays
            if hasattr(frame_data, "frame") and frame_data.frame:
                arr = frame_data.frame[-1]  # last frame
                # Convert numpy array to list of lists
                if hasattr(arr, "tolist"):
                    return arr.tolist()
                if isinstance(arr, list):
                    return arr

            # Fallback: try .grid directly
            if hasattr(frame_data, "grid"):
                grid = frame_data.grid
                if hasattr(grid, "tolist"):
                    return grid.tolist()
                if isinstance(grid, list):
                    return grid

        except Exception as e:
            logger.debug(f"[ENGINE] Grid extraction from frame failed: {e}")

        return [[0] * 8 for _ in range(8)]

    def _update_state_from_frame(self, instance: GameInstance, frame_data):
        try:
            if hasattr(frame_data, "state"):
                state_str = str(frame_data.state).upper()
                if "WIN" in state_str:
                    instance.state = "WIN"
                elif "GAME_OVER" in state_str or "OVER" in state_str:
                    instance.state = "GAME_OVER"
                else:
                    instance.state = "NOT_FINISHED"

            if hasattr(frame_data, "available_actions") and frame_data.available_actions:
                instance._available_actions = [
                    f"ACTION{a}" for a in frame_data.available_actions
                ]

            if hasattr(frame_data, "levels_completed"):
                instance.current_level = frame_data.levels_completed or 0
        except Exception as e:
            logger.debug(f"[ENGINE] State extraction failed: {e}")

    def _get_grid_from_env(self, env) -> list[list[int]]:
        try:
            if hasattr(env, "render"):
                frame = env.render()
                if isinstance(frame, list) and len(frame) > 0:
                    return frame
            if hasattr(env, "grid"):
                return env.grid
            if hasattr(env, "observation"):
                obs = env.observation
                if isinstance(obs, dict) and "grid" in obs:
                    return obs["grid"]
        except Exception:
            pass
        return [[0] * 8 for _ in range(8)]

    def _make_frame(self, instance: GameInstance) -> GameFrame:
        grid = instance._last_grid
        h = len(grid)
        w = len(grid[0]) if h > 0 else 0

        completed_levels = [ls.to_dict() for ls in instance._completed_levels]
        current_level_stats = (
            instance._current_level_stats.to_dict()
            if instance._current_level_stats
            else None
        )

        if instance._game_start_time > 0:
            end = instance._game_end_time if instance._game_end_time > 0 else time.time()
            total_time = round(end - instance._game_start_time, 2)
        else:
            total_time = 0.0

        return GameFrame(
            grid=grid,
            width=w,
            height=h,
            state=instance.state,
            score=instance.score,
            level=instance.current_level,
            total_actions=instance.total_actions,
            available_actions=instance._available_actions,
            metadata={
                "seed": instance.seed,
                "game_id": instance.game_id,
                "mode": instance._mode,
                "total_time": total_time,
                "current_level_stats": current_level_stats,
                "completed_levels": completed_levels,
                "total_game_overs": sum(ls.get("game_overs", 0) for ls in completed_levels) + (current_level_stats or {}).get("game_overs", 0),
                "total_resets": sum(ls.get("resets", 0) for ls in completed_levels) + (current_level_stats or {}).get("resets", 0),
            },
        )

    @staticmethod
    def get_color_palette() -> dict[int, tuple[int, int, int]]:
        return ARC_COLORS
