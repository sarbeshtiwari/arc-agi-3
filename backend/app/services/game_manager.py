import ast
import json
import os
import re
import shutil
import tokenize
import io
from pathlib import Path
from typing import BinaryIO


class GameValidationError(Exception):
    pass


ALLOWED_METADATA_KEYS = {"game_id", "default_fps", "baseline_actions", "tags", "local_dir"}


class GameManagerService:
    def __init__(self, environments_dir: str):
        self.environments_dir = environments_dir
        os.makedirs(environments_dir, exist_ok=True)

    def _check_no_comments_or_docstrings(self, content: str) -> list[str]:
        errors = []

        # Check for single-line comments (#)
        for i, line in enumerate(content.splitlines(), 1):
            stripped = line.strip()
            # Skip shebang
            if i == 1 and stripped.startswith("#!"):
                continue
            # Check for comments (# not inside strings)
            code_part = stripped.split('#', 1)
            if len(code_part) > 1 and stripped and not stripped.startswith(('"""', "'''")):
                # Verify it's a real comment using tokenizer
                pass

        try:
            tokens = list(tokenize.generate_tokens(io.StringIO(content).readline))
            for tok in tokens:
                if tok.type == tokenize.COMMENT:
                    errors.append(f"Line {tok.start[0]}: Comment found: {tok.string.strip()[:50]}")
        except tokenize.TokenizeError:
            pass

        # Check for docstrings (triple-quoted strings as first statement in class/function/module)
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
                    body = node.body
                    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, (ast.Constant, ast.Str)):
                        val = body[0].value
                        if isinstance(val, ast.Constant) and isinstance(val.value, str):
                            line = body[0].lineno
                            snippet = val.value.strip()[:40].replace('\n', ' ')
                            errors.append(f"Line {line}: Docstring found: \"{snippet}...\"")
        except SyntaxError:
            pass

        return errors

    def validate_game_file(self, content: str, game_code: str) -> dict:
        info = {
            "class_name": None,
            "grid_sizes": [],
            "actions_used": [],
            "has_levels": False,
            "has_sprites": False,
            "has_step": False,
            "has_complete_action": False,
        }

        # Check for comments and docstrings
        issues = self._check_no_comments_or_docstrings(content)
        if issues:
            has_comments = any("Comment found" in i for i in issues)
            has_docstrings = any("Docstring found" in i for i in issues)
            parts = []
            if has_comments:
                parts.append("comments (#)")
            if has_docstrings:
                parts.append("docstrings (\"\"\")")
            raise GameValidationError(
                f"Game file must not contain {' or '.join(parts)}. Please remove them and re-upload."
            )

        class_pattern = r'class\s+(\w+)\s*\(\s*ARCBaseGame\s*\)'
        class_match = re.search(class_pattern, content)
        if class_match:
            info["class_name"] = class_match.group(1)

        if re.search(r'def\s+step\s*\(\s*self', content):
            info["has_step"] = True
        if 'complete_action' in content:
            info["has_complete_action"] = True
        if re.search(r'levels\s*=\s*\[', content):
            info["has_levels"] = True
        if re.search(r'sprites\s*=\s*\{', content):
            info["has_sprites"] = True
        grid_sizes = re.findall(r'grid_size\s*=\s*\((\d+)\s*,\s*(\d+)\)', content)
        info["grid_sizes"] = [(int(w), int(h)) for w, h in grid_sizes]
        for i in range(1, 8):
            if f'ACTION{i}' in content:
                info["actions_used"].append(f'ACTION{i}')

        return info

    def validate_metadata(self, content: str) -> dict:
        try:
            metadata = json.loads(content)
        except json.JSONDecodeError as e:
            raise GameValidationError(f"Invalid JSON in metadata.json: {e}")

        if not isinstance(metadata, dict):
            raise GameValidationError("metadata.json must be a JSON object")

        # Check for required field
        if "game_id" not in metadata:
            raise GameValidationError("metadata.json must contain 'game_id' field")

        # Check for unknown keys
        unknown_keys = set(metadata.keys()) - ALLOWED_METADATA_KEYS
        if unknown_keys:
            raise GameValidationError(
                f"metadata.json contains unknown fields: {', '.join(sorted(unknown_keys))}. "
                f"Allowed fields: {', '.join(sorted(ALLOWED_METADATA_KEYS))}"
            )

        # Validate types
        game_id = metadata["game_id"]
        if not isinstance(game_id, str) or not game_id:
            raise GameValidationError("game_id must be a non-empty string")

        if not re.match(r'^[a-z]{2,6}\d{1,4}$', game_id) and not re.match(r'^[a-z0-9]{2,10}-v\d+$', game_id):
            raise GameValidationError(
                f"game_id '{game_id}' must match format '<code><number>' (e.g. 'ls20', 'ab12') "
                f"or '<code>-v<number>' (e.g. 'ls20-v1')"
            )

        if "default_fps" in metadata:
            if not isinstance(metadata["default_fps"], int) or metadata["default_fps"] < 1:
                raise GameValidationError("default_fps must be a positive integer")

        if "baseline_actions" in metadata:
            ba = metadata["baseline_actions"]
            if not isinstance(ba, list) or not all(isinstance(x, int) for x in ba):
                raise GameValidationError("baseline_actions must be a list of integers")

        if "tags" in metadata:
            tags = metadata["tags"]
            if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
                raise GameValidationError("tags must be a list of strings")

        if "local_dir" in metadata:
            if not isinstance(metadata["local_dir"], str):
                raise GameValidationError("local_dir must be a string")

        return metadata
    
    def upload_game(
        self,
        game_py_content: bytes,
        metadata_content: bytes,
        game_code: str | None = None,
        version: str | None = None,
    ) -> dict:
        metadata_str = metadata_content.decode("utf-8")
        metadata = self.validate_metadata(metadata_str)
        
        full_game_id = metadata["game_id"]
        if "-" in full_game_id:
            parts = full_game_id.rsplit("-", 1)
            if not game_code:
                game_code = parts[0]
            if not version:
                version = parts[1] if len(parts) > 1 else "v1"
        else:
            if not game_code:
                game_code = full_game_id
            if not version:
                version = "v1"
        
        game_py_str = game_py_content.decode("utf-8")
        validation_info = self.validate_game_file(game_py_str, game_code)
        
        game_dir = os.path.join(self.environments_dir, game_code, version)
        os.makedirs(game_dir, exist_ok=True)
        
        game_py_path = os.path.join(game_dir, f"{game_code}.py")
        metadata_path = os.path.join(game_dir, "metadata.json")
        
        with open(game_py_path, "w", encoding="utf-8") as f:
            f.write(game_py_str)
        
        metadata["local_dir"] = os.path.join("environment_files", game_code, version)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        
        return {
            "game_id": full_game_id,
            "game_code": game_code,
            "version": version,
            "local_dir": game_dir,
            "game_file_path": game_py_path,
            "metadata_file_path": metadata_path,
            "metadata": metadata,
            "validation_info": validation_info,
        }
    
    def delete_game_files(self, game_code: str, version: str) -> bool:
        game_dir = os.path.join(self.environments_dir, game_code, version)
        if os.path.exists(game_dir):
            shutil.rmtree(game_dir, ignore_errors=True)
            parent = os.path.join(self.environments_dir, game_code)
            if os.path.exists(parent) and not os.listdir(parent):
                os.rmdir(parent)
            return True
        return False
    
    def get_game_file_content(self, game_code: str, version: str) -> str | None:
        game_py_path = os.path.join(self.environments_dir, game_code, version, f"{game_code}.py")
        if os.path.exists(game_py_path):
            with open(game_py_path, "r", encoding="utf-8") as f:
                return f.read()
        return None
    
    def get_metadata_content(self, game_code: str, version: str) -> dict | None:
        metadata_path = os.path.join(self.environments_dir, game_code, version, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None
    
    def list_local_games(self) -> list[dict]:
        games = []
        if not os.path.exists(self.environments_dir):
            return games
        
        for game_code_dir in sorted(os.listdir(self.environments_dir)):
            game_code_path = os.path.join(self.environments_dir, game_code_dir)
            if not os.path.isdir(game_code_path):
                continue
            
            for version_dir in sorted(os.listdir(game_code_path)):
                version_path = os.path.join(game_code_path, version_dir)
                if not os.path.isdir(version_path):
                    continue
                
                metadata_path = os.path.join(version_path, "metadata.json")
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, "r") as f:
                            metadata = json.load(f)
                        games.append({
                            "game_code": game_code_dir,
                            "version": version_dir,
                            "game_id": metadata.get("game_id", f"{game_code_dir}-{version_dir}"),
                            "metadata": metadata,
                            "local_dir": version_path,
                        })
                    except Exception:
                        pass
        
        return games
    
    def update_game_file(self, game_code: str, version: str, new_content: str) -> bool:
        game_py_path = os.path.join(self.environments_dir, game_code, version, f"{game_code}.py")
        if os.path.exists(game_py_path):
            with open(game_py_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            return True
        return False
