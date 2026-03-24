"""
Python bridge for ARCEngine game execution.
Reads NDJSON commands from stdin, executes game actions, writes NDJSON responses to stdout.
Designed to be spawned as a long-lived subprocess by the Node.js server.

Protocol:
  stdin  -> one JSON object per line (commands)
  stdout <- one JSON object per line (responses)

Commands:
  {"command": "init", "game_id": "...", "game_path": "...", "seed": 42}
  {"command": "action", "action": "ACTION1", "x": 0, "y": 0}
  {"command": "reset"}
  {"command": "quit"}

Responses:
  {"type": "ready", "game_id": "...", "frame": {...}}
  {"type": "frame", "grid": [...], "state": "...", ...}
  {"type": "error", "code": "...", "message": "..."}
"""

import sys
import json
import importlib.util
import traceback
from pathlib import Path

# Add ARCEngine to path — expected at external/ARCEngine relative to project root
ARCENGINE_PATH = Path(__file__).resolve().parent.parent.parent / "external" / "ARCEngine"
sys.path.insert(0, str(ARCENGINE_PATH))

try:
    from arcengine import ARCBaseGame, ActionInput, GameAction
except ImportError as e:
    print(json.dumps({
        "type": "error",
        "code": "ARCENGINE_NOT_FOUND",
        "message": f"Failed to import ARCEngine: {e}"
    }), flush=True)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Game loading
# ---------------------------------------------------------------------------

def load_game_from_file(file_path: str):
    """
    Dynamically load a game class from a Python file.
    Returns an instance of the first ARCBaseGame subclass found in the module.

    The module is registered in sys.modules so that @dataclass and other
    decorators that rely on the module lookup work correctly.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Game file not found: {file_path}")

    module_name = f"game_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create module spec from: {file_path}")

    module = importlib.util.module_from_spec(spec)
    # Register in sys.modules BEFORE exec so @dataclass can resolve the module
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    # Find the game class (first subclass of ARCBaseGame that isn't ARCBaseGame itself)
    game_class = None
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if (isinstance(attr, type)
                and issubclass(attr, ARCBaseGame)
                and attr is not ARCBaseGame):
            game_class = attr
            break

    if game_class is None:
        raise ValueError(f"No ARCBaseGame subclass found in {file_path}")

    return game_class


# ---------------------------------------------------------------------------
# Action mapping
# ---------------------------------------------------------------------------

ACTION_MAP = {
    "RESET": GameAction.RESET,
    "ACTION1": GameAction.ACTION1,
    "ACTION2": GameAction.ACTION2,
    "ACTION3": GameAction.ACTION3,
    "ACTION4": GameAction.ACTION4,
    "ACTION5": GameAction.ACTION5,
    "ACTION6": GameAction.ACTION6,
    "ACTION7": GameAction.ACTION7,
}


def parse_action(action_str: str) -> GameAction:
    """Convert an action string to a GameAction enum value."""
    result = ACTION_MAP.get(action_str.upper())
    if result is None:
        raise ValueError(f"Unknown action: {action_str}")
    return result


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def extract_frame(game, frame_data, total_actions: int) -> dict:
    """
    Build a frame response dict from ARCEngine FrameData.
    frame_data.frame is a list of animation frames (each a 2D grid).
    We take the first frame as the current grid.
    """
    frame_list = frame_data.frame
    # Normalise: ensure we have a plain Python list of lists
    if isinstance(frame_list, list):
        frame_list = [
            f.tolist() if hasattr(f, "tolist") else f
            for f in frame_list
        ]
    elif hasattr(frame_list, "tolist"):
        frame_list = [frame_list.tolist()]

    # Primary grid is the LAST animation frame (contains the current/new level state).
    # On level clear, arcengine returns multiple frames: [victory_frame, new_level_frame].
    # We always want the final frame which shows the current game state.
    grid = frame_list[-1] if frame_list else []

    state_val = frame_data.state.value if hasattr(frame_data.state, "value") else str(frame_data.state)

    return {
        "type": "frame",
        "grid": grid,
        "width": len(grid[0]) if grid and len(grid) > 0 else 0,
        "height": len(grid) if grid else 0,
        "state": state_val,
        "level": frame_data.levels_completed,
        "levels_completed": frame_data.levels_completed,
        "total_actions": total_actions,
        "max_actions": getattr(game, "max_actions", 100),
        "available_actions": [f"ACTION{a}" for a in frame_data.available_actions] if frame_data.available_actions else [],
        "win_levels": frame_data.win_levels,
    }


# ---------------------------------------------------------------------------
# NDJSON helpers
# ---------------------------------------------------------------------------

def emit(obj: dict):
    """Write a JSON object as a single line to stdout and flush."""
    print(json.dumps(obj), flush=True)


def emit_error(message: str, code: str = "GAME_ERROR"):
    emit({"type": "error", "code": code, "message": message})


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    game = None
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

        try:
            # ----------------------------------------------------------
            # INIT — load a game from file and emit ready + initial frame
            # ----------------------------------------------------------
            if command == "init":
                game_id = cmd.get("game_id", "unknown")
                game_path = cmd.get("game_path")
                seed = cmd.get("seed")

                if not game_path:
                    emit_error("game_path is required for init", "MISSING_GAME_PATH")
                    continue

                GameClass = load_game_from_file(game_path)
                game = GameClass() if seed is None else GameClass(seed=seed)
                total_actions = 0

                # Get initial frame via RESET action
                initial_frame_data = game.perform_action(ActionInput(id=GameAction.RESET))
                frame = extract_frame(game, initial_frame_data, total_actions)

                emit({
                    "type": "ready",
                    "game_id": game_id,
                    "frame": frame,
                    "metadata": {
                        "game_id": game_id,
                        "level_count": len(getattr(game, "_levels", [])),
                        "win_score": getattr(game, "win_score", 1),
                        "max_actions": getattr(game, "max_actions", 100),
                    },
                })

            # ----------------------------------------------------------
            # ACTION — perform a game action and return the new frame
            # ----------------------------------------------------------
            elif command == "action":
                if game is None:
                    emit_error("No game loaded. Send 'init' first.", "NO_GAME")
                    continue

                action_str = cmd.get("action", "ACTION1")
                action_id = parse_action(action_str)
                action_input = ActionInput(id=action_id)

                # ACTION6 supports coordinates (x, y)
                if action_str.upper() == "ACTION6":
                    if "x" in cmd and "y" in cmd:
                        action_input.x = cmd["x"]
                        action_input.y = cmd["y"]

                total_actions += 1
                frame_data = game.perform_action(action_input)
                emit(extract_frame(game, frame_data, total_actions))

            # ----------------------------------------------------------
            # RESET — reset the game to initial state
            # ----------------------------------------------------------
            elif command == "reset":
                if game is None:
                    emit_error("No game loaded. Send 'init' first.", "NO_GAME")
                    continue

                total_actions = 0
                frame_data = game.perform_action(ActionInput(id=GameAction.RESET))
                emit(extract_frame(game, frame_data, total_actions))

            # ----------------------------------------------------------
            # QUIT — exit cleanly
            # ----------------------------------------------------------
            elif command == "quit":
                emit({"type": "quit", "message": "goodbye"})
                break

            else:
                emit_error(f"Unknown command: {command}", "UNKNOWN_COMMAND")

        except FileNotFoundError as e:
            emit_error(str(e), "FILE_NOT_FOUND")
        except ImportError as e:
            emit_error(f"Failed to import game: {e}", "IMPORT_ERROR")
        except ValueError as e:
            emit_error(str(e), "INVALID_GAME")
        except Exception as e:
            emit_error(f"Unexpected error: {e}", "UNEXPECTED_ERROR")
            traceback.print_exc(file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
