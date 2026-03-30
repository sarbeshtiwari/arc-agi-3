"""
Python bridge for PuzzleEnvironment game execution.
Reads NDJSON commands from stdin, executes game actions, writes NDJSON responses to stdout.
Designed to be spawned as a long-lived subprocess by the Node.js server.

Protocol:
  stdin  -> one JSON object per line (commands)
  stdout <- one JSON object per line (responses)

Commands:
  {"command": "init", "game_id": "...", "game_path": "...", "seed": 42}
  {"command": "action", "action": "up"}
  {"command": "reset"}
  {"command": "quit"}

Responses:
  {"type": "ready", "game_id": "...", "frame": {...}}
  {"type": "frame", "grid": [...], "state": "...", ...}
  {"type": "error", "code": "...", "message": "..."}
"""

import sys
import json
import base64
import importlib.util
import traceback
import time
from pathlib import Path

ARCENGINE_PATH = Path(__file__).resolve().parent.parent.parent / "external" / "ARCEngine"
sys.path.insert(0, str(ARCENGINE_PATH))


def _log(level: str, msg: str, **extra):
    payload = {"type": "log", "level": level, "message": msg}
    if extra:
        payload["metadata"] = extra
    print(json.dumps(payload), flush=True)


# ---------------------------------------------------------------------------
# Game loading — finds PuzzleEnvironment class in the given file
# ---------------------------------------------------------------------------

def load_puzzle_env_class(file_path: str):
    path = Path(file_path)
    _log("debug", f"Loading game file: {file_path}", file_path=str(path), exists=path.exists())
    if not path.exists():
        raise FileNotFoundError(f"Game file not found: {file_path}")

    module_name = f"game_{path.stem}"
    _log("debug", f"Creating module spec: {module_name}", module_name=module_name)
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create module spec from: {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    t0 = time.time()
    spec.loader.exec_module(module)
    load_time = round(time.time() - t0, 3)
    _log("debug", f"Module loaded in {load_time}s", module_name=module_name, load_time_s=load_time)

    all_attrs = [a for a in dir(module) if not a.startswith("_")]
    _log("debug", f"Module exports: {all_attrs}", module_name=module_name, exports=all_attrs)

    puzzle_env_class = getattr(module, "PuzzleEnvironment", None)
    if puzzle_env_class is None:
        raise ValueError(
            f"No PuzzleEnvironment class found in {file_path}. "
            f"All games must export a PuzzleEnvironment class."
        )

    bases = [b.__name__ for b in puzzle_env_class.__mro__]
    _log("info", f"Found PuzzleEnvironment class", class_name=puzzle_env_class.__name__, mro=bases)

    return puzzle_env_class


# ---------------------------------------------------------------------------
# Action translation — maps legacy ACTION1..7/RESET names to PuzzleEnvironment action names
# ---------------------------------------------------------------------------

# arcengine GameAction int values: RESET=0, ACTION1=1, ..., ACTION7=7
_LEGACY_TO_INT = {
    "RESET": 0,
    "ACTION1": 1, "ACTION2": 2, "ACTION3": 3, "ACTION4": 4,
    "ACTION5": 5, "ACTION6": 6, "ACTION7": 7,
}


_INT_TO_LEGACY = {v: k for k, v in _LEGACY_TO_INT.items()}


# Common action name → legacy int mapping for games without ACTION_MAP
_WELL_KNOWN_ACTIONS = {
    "up": 1, "down": 2, "left": 3, "right": 4,
    "select": 5, "click": 6, "undo": 7,
}


def build_action_translator(env):
    action_map = getattr(env, "ACTION_MAP", None) or {}

    # Fallback: if no ACTION_MAP, infer from get_actions() or VALID_ACTIONS
    inferred = False
    if not action_map:
        inferred = True
        actions = []
        if hasattr(env, "get_actions"):
            try:
                actions = env.get_actions() or []
            except Exception:
                actions = []
        if not actions:
            actions = getattr(env, "VALID_ACTIONS", []) or []

        _log("info", "No ACTION_MAP found, building from available actions", available=actions)
        # Build a synthetic ACTION_MAP: well-known names get standard slots,
        # unknown names get next available slot
        used_ints = {0}  # reserve 0 for reset
        for name in actions:
            low = name.lower()
            if low == "reset":
                action_map["reset"] = 0
                continue
            if low in _WELL_KNOWN_ACTIONS:
                int_val = _WELL_KNOWN_ACTIONS[low]
                action_map[low] = int_val
                used_ints.add(int_val)

        # Assign remaining unknown actions to next free slots
        next_slot = 1
        for name in actions:
            low = name.lower()
            if low in action_map:
                continue
            while next_slot in used_ints:
                next_slot += 1
            action_map[low] = next_slot
            used_ints.add(next_slot)
            next_slot += 1

        _log("info", "Inferred ACTION_MAP", inferred_map=action_map)

    _log("debug", f"Building action translator", action_map=action_map)
    int_to_name = {}
    name_to_legacy = {}
    for name, int_val in action_map.items():
        int_to_name[int_val] = name
        if int_val in _INT_TO_LEGACY:
            name_to_legacy[name] = _INT_TO_LEGACY[int_val]

    _log("debug", f"Action mapping built", int_to_name=int_to_name, name_to_legacy=name_to_legacy)

    def translate(legacy_action_str):
        upper = legacy_action_str.upper()
        if upper in _LEGACY_TO_INT:
            int_val = _LEGACY_TO_INT[upper]
            resolved = int_to_name.get(int_val, legacy_action_str)
            _log("debug", f"Translated action: {legacy_action_str} -> {resolved}", legacy=legacy_action_str, resolved=resolved, int_val=int_val)
            return resolved
        lower = legacy_action_str.lower()
        if lower in action_map:
            _log("debug", f"Action passed through: {lower}")
            return lower
        _log("warn", f"Unknown action, passing through: {legacy_action_str}")
        return legacy_action_str

    def to_legacy_list(puzzle_action_names):
        result = []
        for n in puzzle_action_names:
            key = n.lower() if isinstance(n, str) else (n.name.lower() if hasattr(n, 'name') else str(n).lower())
            legacy = name_to_legacy.get(key, _INT_TO_LEGACY.get(n.value if hasattr(n, 'value') else n, str(n)))
            result.append(legacy)
        return result

    return translate, to_legacy_list, inferred


# ---------------------------------------------------------------------------
# Frame extraction from PuzzleEnvironment GameState / StepResult
# ---------------------------------------------------------------------------

def build_frame_from_game_state(env, game_state, total_actions, to_legacy_list, reward=0.0, done=False, info=None):
    grid = _extract_grid(env)

    metadata = game_state.metadata or {}

    # WIN detection: check all known info keys, then fall back to metadata.game_over
    if done:
        is_win = (
            info and any(info.get(k) == "game_complete" for k in ("reason", "event", "outcome"))
        )
        if not is_win and "game_over" in metadata:
            # game_over=True means player lost, False means player won
            is_win = not metadata["game_over"]
        state_str = "WIN" if is_win else "GAME_OVER"
    else:
        state_str = "NOT_FINISHED"

    level = metadata.get("level_index", 0) or 0

    image_b64 = None
    if game_state.image_observation:
        image_b64 = base64.b64encode(game_state.image_observation).decode("ascii")

    valid_actions = game_state.valid_actions or []
    legacy_actions = to_legacy_list(valid_actions)

    _log("debug", "Frame built", 
         grid_size=f"{len(grid)}x{len(grid[0]) if grid and len(grid) > 0 else 0}",
         state=state_str, reward=reward, done=done,
         game_level=level,
         total_actions=total_actions,
         valid_actions=valid_actions,
         legacy_actions=legacy_actions,
         has_image=image_b64 is not None,
         has_text_obs=bool(game_state.text_observation),
         info=info)

    return {
        "type": "frame",
        "grid": grid,
        "width": len(grid[0]) if grid and len(grid) > 0 else 0,
        "height": len(grid) if grid else 0,
        "state": state_str,
        "level": level,
        "levels_completed": level,
        "total_levels": metadata.get("total_levels") or 1,
        "total_actions": total_actions,
        "available_actions": legacy_actions,
        "reward": reward,
        "done": done,
        "text_observation": game_state.text_observation,
        "image_observation_b64": image_b64,
        "metadata": metadata,
    }


def _extract_grid(env):
    try:
        engine = env._engine
        frame = engine.camera.render(engine.current_level.get_sprites())
        if hasattr(frame, "tolist"):
            return frame.tolist()
        return frame
    except Exception:
        return [[0]]


# ---------------------------------------------------------------------------
# NDJSON helpers
# ---------------------------------------------------------------------------

def emit(obj: dict):
    print(json.dumps(obj), flush=True)


def emit_error(message: str, code: str = "GAME_ERROR"):
    emit({"type": "error", "code": code, "message": message})


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    _log("info", "game_runner.py started", python_version=sys.version, arcengine_path=str(ARCENGINE_PATH))
    env = None
    translate_action = None
    to_legacy_list = None
    action_map_inferred = False
    total_actions = 0

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            cmd = json.loads(line)
        except json.JSONDecodeError as e:
            emit_error(f"Invalid JSON: {e}", "INVALID_JSON")
            continue

        command = cmd.get("command", "")
        _log("debug", f"Received command: {command}", raw_command=cmd)

        try:
            if command == "init":
                game_id = cmd.get("game_id", "unknown")
                game_path = cmd.get("game_path")
                seed = cmd.get("seed")

                _log("info", f"Initializing game", game_id=game_id, game_path=game_path, seed=seed)

                if not game_path:
                    emit_error("game_path is required for init", "MISSING_GAME_PATH")
                    continue

                t0 = time.time()
                PuzzleEnvClass = load_puzzle_env_class(game_path)
                _log("debug", f"Instantiating PuzzleEnvironment", class_name=PuzzleEnvClass.__name__, seed=seed)

                env = PuzzleEnvClass() if seed is None else PuzzleEnvClass(seed=seed)
                _log("debug", f"PuzzleEnvironment created", 
                     has_action_map=hasattr(env, "ACTION_MAP"),
                     has_engine=hasattr(env, "_engine"),
                     action_map=getattr(env, "ACTION_MAP", None))

                translate_action, to_legacy_list, action_map_inferred = build_action_translator(env)
                total_actions = 0

                _log("debug", "Calling env.reset() for initial state")
                game_state = env.reset()
                _log("debug", f"Initial reset complete",
                     valid_actions=game_state.valid_actions,
                     turn=game_state.turn,
                     metadata=game_state.metadata)

                frame = build_frame_from_game_state(env, game_state, total_actions, to_legacy_list)
                frame["action_map_inferred"] = action_map_inferred

                metadata = game_state.metadata or {}
                init_time = round(time.time() - t0, 3)
                _log("info", f"Game initialized in {init_time}s", 
                     game_id=game_id, init_time_s=init_time,
                     total_levels=metadata.get("total_levels", 1),
                     grid_size=f"{frame['height']}x{frame['width']}",
                     available_actions=frame["available_actions"])

                emit({
                    "type": "ready",
                    "game_id": game_id,
                    "frame": frame,
                    "metadata": {
                        "game_id": game_id,
                        "level_count": metadata.get("total_levels", 1),
                        "total_levels": metadata.get("total_levels", 1),
                        "action_map_inferred": action_map_inferred,
                    },
                })

            elif command == "action":
                if env is None:
                    emit_error("No game loaded. Send 'init' first.", "NO_GAME")
                    continue

                raw_action = cmd.get("action", "")
                action_str = translate_action(raw_action)

                total_actions += 1
                _log("debug", f"Calling env.step('{action_str}')", action_num=total_actions, raw=raw_action, translated=action_str)

                t0 = time.time()
                step_result = env.step(action_str)
                step_time = round(time.time() - t0, 4)

                _log("debug", f"Step completed in {step_time}s",
                     step_time_s=step_time,
                     reward=step_result.reward,
                     done=step_result.done,
                     info=step_result.info,
                     new_valid_actions=step_result.state.valid_actions,
                     turn=step_result.state.turn)

                frame = build_frame_from_game_state(
                    env,
                    step_result.state,
                    total_actions,
                    to_legacy_list,
                    reward=step_result.reward,
                    done=step_result.done,
                    info=step_result.info,
                )
                frame["action_map_inferred"] = action_map_inferred
                emit(frame)

            elif command == "reset":
                if env is None:
                    emit_error("No game loaded. Send 'init' first.", "NO_GAME")
                    continue

                _log("info", "Resetting environment", total_actions_before_reset=total_actions)
                total_actions = 0
                game_state = env.reset()
                _log("debug", "Reset complete",
                     valid_actions=game_state.valid_actions,
                     turn=game_state.turn,
                     metadata=game_state.metadata)

                frame = build_frame_from_game_state(env, game_state, total_actions, to_legacy_list)
                frame["action_map_inferred"] = action_map_inferred
                emit(frame)

            elif command == "quit":
                _log("info", "Quit received, shutting down")
                emit({"type": "quit", "message": "goodbye"})
                break

            else:
                _log("warn", f"Unknown command: {command}")
                emit_error(f"Unknown command: {command}", "UNKNOWN_COMMAND")

        except FileNotFoundError as e:
            _log("error", f"File not found: {e}", traceback=traceback.format_exc())
            emit_error(str(e), "FILE_NOT_FOUND")
        except ImportError as e:
            _log("error", f"Import error: {e}", traceback=traceback.format_exc())
            emit_error(f"Failed to import game: {e}", "IMPORT_ERROR")
        except ValueError as e:
            _log("error", f"Value error: {e}", traceback=traceback.format_exc())
            emit_error(str(e), "INVALID_GAME")
        except Exception as e:
            _log("error", f"Unexpected error: {e}", traceback=traceback.format_exc())
            emit_error(f"Unexpected error: {e}", "UNEXPECTED_ERROR")

    _log("info", "game_runner.py exiting")
    return 0


if __name__ == "__main__":
    sys.exit(main())
