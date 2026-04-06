from __future__ import annotations
import io
import random
import struct
import zlib
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces

from arcengine import (
    ARCBaseGame,
    Camera,
    GameAction,
    GameState as _EngineState,
    Level,
    ActionInput,
)
from arcengine.interfaces import RenderableUserDisplay

CANVAS = 64
MAX_LIVES = 3
BAR_W = 50
BAR_Y = 59

C_BG = 0
C_EDGE = 3
C_NODE = 10
C_SEL = 6
C_SEL_NODE = 2
C_LOCK = 9
C_UNBAL = 8
C_RING = 2
C_BAR_FILL = 14
C_BAR_EMPTY = 3
C_LIFE_ON = 14
C_LIFE_OFF = 8
C_DOT_DONE = 14
C_DOT_CUR = 6
C_DOT_TODO = 3
C_GOAL_NODE = 4
C_GOAL_EDGE = 5

NODE_PALETTE = [6, 4, 14, 7, 10, 8, 1, 9]

REF_X1 = 40
REF_Y1 = 0
REF_X2 = 63
REF_Y2 = 13

TREE_Y_TOP = 18
TREE_Y_BOT = 50
TREE_X_MIN = 6
TREE_X_MAX = 58

N_LEVELS = 4


class Node:
    __slots__ = ("val", "left", "right", "locked")

    def __init__(self, val, left=None, right=None, locked=False):
        self.val = val
        self.left = left
        self.right = right
        self.locked = locked

    def copy(self):
        n = Node(self.val, locked=self.locked)
        if self.left:
            n.left = self.left.copy()
        if self.right:
            n.right = self.right.copy()
        return n


def _height(n):
    if n is None:
        return 0
    return 1 + max(_height(n.left), _height(n.right))


def _bf(n):
    if n is None:
        return 0
    return _height(n.left) - _height(n.right)


def _avl_ok(n):
    if n is None:
        return True
    if abs(_bf(n)) > 1:
        return False
    return _avl_ok(n.left) and _avl_ok(n.right)


def _find_parent(root, target):
    stack = [root]
    while stack:
        n = stack.pop()
        if n.left is target or n.right is target:
            return n
        if n.right:
            stack.append(n.right)
        if n.left:
            stack.append(n.left)
    return None


def _right_rotate(holder, node):
    if node is None or node.left is None:
        return False
    parent = _find_parent(holder[0], node)
    pivot = node.left
    node.left = pivot.right
    pivot.right = node
    if parent is None:
        holder[0] = pivot
    elif parent.left is node:
        parent.left = pivot
    else:
        parent.right = pivot
    return True


def _find_path(root, target):
    if root is None:
        return None
    if root is target:
        return []
    left = _find_path(root.left, target)
    if left is not None:
        return ["L"] + left
    right = _find_path(root.right, target)
    if right is not None:
        return ["R"] + right
    return None


def _follow_mirror(root, path):
    n = root
    for step in path:
        if n is None:
            return None
        n = n.right if step == "L" else n.left
    return n


def _layout(
    root, x_min=TREE_X_MIN, x_max=TREE_X_MAX, y_top=TREE_Y_TOP, y_bot=TREE_Y_BOT
):
    if root is None:
        return [], [], []

    positions = []
    edges = []
    refs = []

    x_order = {}
    counter = [0]

    def _inorder(n):
        if n is None:
            return
        _inorder(n.left)
        x_order[id(n)] = counter[0]
        counter[0] += 1
        _inorder(n.right)

    _inorder(root)
    n_nodes = counter[0]

    tree_depth = _height(root)

    usable_w = x_max - x_min
    x_step = usable_w / max(n_nodes - 1, 1) if n_nodes > 1 else 0
    y_step = (y_bot - y_top) / max(tree_depth - 1, 1) if tree_depth > 1 else 0

    def _dfs(n, depth):
        if n is None:
            return None
        idx = len(positions)
        if n_nodes == 1:
            x = (x_min + x_max) // 2
        else:
            x = int(x_min + x_order[id(n)] * x_step)
        y = int(y_top + depth * y_step)
        x = max(4, min(CANVAS - 5, x))
        y = max(4, min(CANVAS - 10, y))
        positions.append((x, y))
        refs.append(n)
        if n.left:
            ci = _dfs(n.left, depth + 1)
            edges.append((idx, ci))
        if n.right:
            ci = _dfs(n.right, depth + 1)
            edges.append((idx, ci))
        return idx

    _dfs(root, 0)
    return positions, edges, refs


def _trees_equal(a, b):
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if a.val != b.val:
        return False
    return _trees_equal(a.left, b.left) and _trees_equal(a.right, b.right)


def _collect_vals(n):
    if n is None:
        return []
    return _collect_vals(n.left) + [n.val] + _collect_vals(n.right)


def _balanced_bst(vals):
    if not vals:
        return None
    mid = len(vals) // 2
    node = Node(vals[mid])
    node.left = _balanced_bst(vals[:mid])
    node.right = _balanced_bst(vals[mid + 1 :])
    return node


def _lvl1():
    return Node(30, Node(20, Node(10))), 2, "Left Chain"


def _lvl2():
    return (
        Node(50, Node(40, Node(30, Node(20, Node(10))))),
        2,
        "Cascade",
    )


def _lvl3():
    return (
        Node(
            70,
            Node(40, Node(30, Node(20, Node(10))), Node(60, Node(50))),
        ),
        3,
        "Grove",
    )


def _lvl4():
    return (
        Node(
            80,
            Node(
                70,
                Node(
                    60,
                    Node(50, Node(40, Node(30, Node(20, Node(10))))),
                ),
            ),
            Node(90),
        ),
        4,
        "Ancient Oak",
    )


_BUILDERS = [_lvl1, _lvl2, _lvl3, _lvl4]

_SEL_CANDIDATES: list[list[int]] = [
    [0, 1, 2, 0],
    [0, 2, 4, 5],
    [0, 2, 4, 6],
    [0, 2, 4, 6],
]


class _HUD(RenderableUserDisplay):
    def __init__(self):
        self.positions = []
        self.edges = []
        self.refs = []
        self.sel = 0
        self.lives = MAX_LIVES
        self.progress = 1
        self.progress_max = 1
        self.lvl = 0
        self.ref_positions = []
        self.ref_edges = []
        self.ref_refs = []
        self.val_to_color = {}

    @staticmethod
    def _px(f, x, y, c):
        if 0 <= x < CANVAS and 0 <= y < CANVAS:
            f[y, x] = c

    def _line(self, f, x0, y0, x1, y1, c):
        steps = max(abs(x1 - x0), abs(y1 - y0), 1)
        for i in range(steps + 1):
            x = int(x0 + (x1 - x0) * i / steps)
            y = int(y0 + (y1 - y0) * i / steps)
            self._px(f, x, y, c)

    def _draw_tree(
        self,
        f,
        positions,
        edges,
        refs,
        node_color,
        edge_color,
        sel_idx=-1,
        node_r=2,
        val_to_color=None,
    ):
        for a, b in edges:
            x0, y0 = positions[a]
            x1, y1 = positions[b]
            self._line(f, x0, y0, x1, y1, edge_color)

        for i, (x, y) in enumerate(positions):
            node = refs[i]
            unbal = abs(_bf(node)) > 1

            if val_to_color and node.val in val_to_color:
                col = val_to_color[node.val]
            elif node_color is not None:
                col = node_color
            else:
                col = C_NODE

            if sel_idx >= 0 and i == sel_idx:
                col = C_SEL_NODE

            if node_r <= 0:
                self._px(f, x, y, col)
            else:
                for dy in range(-node_r, node_r):
                    for dx in range(-node_r, node_r):
                        self._px(f, x + dx, y + dy, col)

            if node_r >= 2:
                if unbal:
                    for d in range(-3, 3):
                        self._px(f, x + d, y - 3, C_UNBAL)
                        self._px(f, x + d, y + 2, C_UNBAL)
                        self._px(f, x - 3, y + d, C_UNBAL)
                        self._px(f, x + 2, y + d, C_UNBAL)
                if sel_idx >= 0 and i == sel_idx:
                    for d in range(-3, 3):
                        self._px(f, x + d, y - 3, C_SEL_NODE)
                        self._px(f, x + d, y + 2, C_SEL_NODE)
                        self._px(f, x - 3, y + d, C_SEL_NODE)
                        self._px(f, x + 2, y + d, C_SEL_NODE)

    def _draw_ref_box(self, f):
        if not self.ref_positions:
            return
        for y in range(REF_Y1 + 1, REF_Y2):
            for x in range(REF_X1 + 1, REF_X2):
                self._px(f, x, y, C_BG)
        self._draw_tree(
            f,
            self.ref_positions,
            self.ref_edges,
            self.ref_refs,
            C_GOAL_NODE,
            C_GOAL_EDGE,
            node_r=1,
            val_to_color=self.val_to_color,
        )
        for x in range(REF_X1, REF_X2 + 1):
            self._px(f, x, REF_Y1, C_EDGE)
            self._px(f, x, REF_Y2, C_EDGE)
        for y in range(REF_Y1, REF_Y2 + 1):
            self._px(f, REF_X1, y, C_EDGE)
            self._px(f, REF_X2, y, C_EDGE)

    def render_interface(self, f):
        self._draw_tree(
            f,
            self.positions,
            self.edges,
            self.refs,
            C_NODE,
            C_EDGE,
            sel_idx=self.sel,
            node_r=2,
            val_to_color=self.val_to_color,
        )

        self._draw_ref_box(f)

        for i in range(MAX_LIVES):
            c = C_LIFE_ON if i < self.lives else C_LIFE_OFF
            bx = 2 + i * 4
            for dy in range(2):
                for dx in range(2):
                    self._px(f, bx + dx, 1 + dy, c)

        if self.progress_max > 0:
            ratio = self.progress / self.progress_max
            fill = int(BAR_W * ratio)
        else:
            fill = 0
        fill = max(0, min(BAR_W, fill))
        for i in range(BAR_W):
            c = C_BAR_FILL if i < fill else C_BAR_EMPTY
            self._px(f, 7 + i, BAR_Y, c)
            self._px(f, 7 + i, BAR_Y + 1, c)

        return f


class Tb01(ARCBaseGame):
    def __init__(self, seed: int = 0):
        self.hud = _HUD()
        self._rng = random.Random(seed)

        self.tree = None
        self.positions = []
        self.refs = []
        self.sel = 0

        self.progress = 1
        self.progress_max = 1
        self._lives = MAX_LIVES

        self.game_won = False
        self._force_full_reset = False
        self._goal_tree = None

        self._undo_stack = []

        levels = [
            Level(sprites=[], grid_size=(CANVAS, CANVAS), name=f"Level {i + 1}")
            for i in range(N_LEVELS)
        ]

        camera = Camera(0, 0, CANVAS, CANVAS, C_BG, C_BG, [self.hud])

        super().__init__(
            game_id="tb01",
            levels=levels,
            camera=camera,
            available_actions=[0, 1, 2, 3, 4, 5, 7],
        )

    def on_set_level(self, level):
        data = _BUILDERS[self.level_index]()
        self.tree = data[0]
        opt = data[1]

        self.progress_max = opt * 8
        self.progress = self.progress_max + 1
        self._undo_stack = []

        vals = sorted(_collect_vals(self.tree))

        val_to_color = {}
        for i, v in enumerate(vals):
            val_to_color[v] = NODE_PALETTE[i % len(NODE_PALETTE)]
        self.hud.val_to_color = val_to_color

        goal_tree = _balanced_bst(vals)
        self._goal_tree = goal_tree
        rp, re, rr = _layout(
            goal_tree,
            x_min=REF_X1 + 2,
            x_max=REF_X2 - 4,
            y_top=REF_Y1 + 4,
            y_bot=REF_Y2 - 2,
        )
        self.hud.ref_positions = rp
        self.hud.ref_edges = re
        self.hud.ref_refs = rr

        self._sync_layout()

        candidates = _SEL_CANDIDATES[self.level_index]
        chosen = self._rng.choice(candidates)
        self.sel = chosen % len(self.refs) if self.refs else 0

        self._sync_hud()

    def _sync_layout(self):
        self.positions, edges, self.refs = _layout(self.tree)
        self.hud.positions = self.positions
        self.hud.edges = edges
        self.hud.refs = self.refs
        if self.sel >= len(self.refs):
            self.sel = 0
        self.hud.sel = self.sel

    def _sync_hud(self):
        self.hud.lives = self._lives
        self.hud.progress = self.progress
        self.hud.progress_max = self.progress_max
        self.hud.lvl = self.level_index
        self.hud.sel = self.sel

    def handle_reset(self):
        self.game_won = False
        self._lives = MAX_LIVES
        if self._force_full_reset:
            self._force_full_reset = False
            self.full_reset()
        else:
            super().handle_reset()

    def _level_reset(self):
        self.on_set_level(self.current_level)

    def _lose_life(self):
        self._lives -= 1
        self.hud.lives = self._lives
        if self._lives <= 0:
            self.lose()
        else:
            self._level_reset()

    def _move_sel(self, dx, dy):
        if not self.refs:
            return
        node = self.refs[self.sel]
        target = None

        if dy == -1:
            target = _find_parent(self.tree, node)
        elif dy == 1:
            has_l = node.left is not None
            has_r = node.right is not None
            if has_l and not has_r:
                target = node.left
            elif has_r and not has_l:
                target = node.right
        elif dx == -1:
            target = node.left
        elif dx == 1:
            target = node.right

        if target is not None:
            for i, n in enumerate(self.refs):
                if n is target:
                    self.sel = i
                    self.hud.sel = i
                    return

    def _try_rotate(self, direction):
        if not self.refs:
            return
        node = self.refs[self.sel]
        if node.locked:
            return

        holder = [self.tree]
        ok = False
        if direction == "right":
            ok = _right_rotate(holder, node)

        if ok:
            self.tree = holder[0]

            target = node
            self._sync_layout()
            for i, n in enumerate(self.refs):
                if n is target:
                    self.sel = i
                    break
            self._sync_hud()

    def _save_snapshot(self):
        return {
            "tree": self.tree.copy() if self.tree else None,
            "sel": self.sel,
        }

    def _restore_snapshot(self, snap):
        self.tree = snap["tree"]
        self.sel = snap["sel"]
        self._sync_layout()
        self._sync_hud()

    def step(self):
        aid = self.action.id

        if aid == GameAction.ACTION7:
            if self._undo_stack:
                snap = self._undo_stack.pop()
                self._restore_snapshot(snap)
                self.progress -= 1
                self._sync_hud()
                if self.progress <= 0:
                    self._lose_life()
            self.complete_action()
            return

        self._undo_stack.append(self._save_snapshot())

        if aid == GameAction.ACTION5:
            self._try_rotate("right")
        elif aid == GameAction.ACTION1:
            self._move_sel(0, -1)
        elif aid == GameAction.ACTION2:
            self._move_sel(0, 1)
        elif aid == GameAction.ACTION3:
            self._move_sel(-1, 0)
        elif aid == GameAction.ACTION4:
            self._move_sel(1, 0)
        self.progress -= 1
        self._sync_hud()

        if _trees_equal(self.tree, self._goal_tree):
            if self.is_last_level():
                self.game_won = True
                self.win()
            else:
                self._lives = MAX_LIVES
                self.hud.lives = MAX_LIVES
                self.next_level()
            self.complete_action()
            return

        if self.progress <= 0:
            self._lose_life()

        self.complete_action()


@dataclass
class GameState:
    text_observation: str
    image_observation: bytes | None
    valid_actions: list | None
    turn: int
    metadata: dict = field(default_factory=dict)


@dataclass
class StepResult:
    state: GameState
    reward: float
    done: bool
    info: dict = field(default_factory=dict)


def _tree_text(n, prefix="", is_right=False, sel=None):
    if n is None:
        return ""
    marker = "R── " if is_right else "L── "
    b = _bf(n)
    fl = ""
    if n.locked:
        fl += " [LOCK]"
    if sel is n:
        fl += " <"
    ub = " !" if abs(b) > 1 else ""
    line = f"{prefix}{marker}{n.val} (bf={b:+d}){ub}{fl}"
    child_prefix = prefix + ("    " if not is_right else "    ")
    parts = [line]
    if n.left:
        parts.append(_tree_text(n.left, child_prefix, False, sel))
    if n.right:
        parts.append(_tree_text(n.right, child_prefix, True, sel))
    return "\n".join(p for p in parts if p)


class PuzzleEnvironment:
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

    _ACTION_MAP: Dict[str, GameAction] = {
        "reset": GameAction.RESET,
        "up": GameAction.ACTION1,
        "down": GameAction.ACTION2,
        "left": GameAction.ACTION3,
        "right": GameAction.ACTION4,
        "select": GameAction.ACTION5,
        "undo": GameAction.ACTION7,
    }

    _VALID_ACTIONS: List[str] = [
        "reset",
        "up",
        "down",
        "left",
        "right",
        "select",
        "undo",
    ]

    def __init__(self, seed: int = 0) -> None:
        self._engine = Tb01(seed=seed)
        self._done = False
        self._game_over = False
        self._last_action_was_reset = False
        self._game_won = False
        self._total_turns = 0

    def _render_frame(self) -> np.ndarray:
        e = self._engine
        return e.camera.render(e.current_level.get_sprites())

    @staticmethod
    def _frame_to_png(frame: np.ndarray) -> bytes:
        h, w = frame.shape[:2]
        channels = frame.shape[2] if frame.ndim == 3 else 1
        color_type = 2 if channels == 3 else 0
        buf = io.BytesIO()
        buf.write(b"\x89PNG\r\n\x1a\n")

        def _chunk(ctype: bytes, data: bytes) -> None:
            buf.write(struct.pack(">I", len(data)))
            buf.write(ctype)
            buf.write(data)
            buf.write(struct.pack(">I", zlib.crc32(ctype + data) & 0xFFFFFFFF))

        _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, color_type, 0, 0, 0))
        raw = b""
        for y in range(h):
            raw += b"\x00" + frame[y].astype(np.uint8).tobytes()
        _chunk(b"IDAT", zlib.compress(raw))
        _chunk(b"IEND", b"")
        return buf.getvalue()

    def _build_text_observation(self) -> str:
        g = self._engine
        sel_node = g.refs[g.sel] if g.refs else None
        lvl_data = _BUILDERS[g.level_index]()
        lvl_name = lvl_data[2]

        tree_txt = _tree_text(g.tree, "", False, sel_node) if g.tree else "(empty)"

        lines = [
            f"=== Tree Balancing Puzzle | Level {g.level_index + 1}: {lvl_name} ===",
            f"Lives: {g._lives}/{MAX_LIVES}  Moves: {g.progress}/{g.progress_max}",
            f"Selected: node {sel_node.val if sel_node else '?'}",
            "",
            tree_txt,
            "",
            "select = right-rotate  (left child rises)",
            "Actions: up / down / left / right / select / undo",
        ]

        return "\n".join(lines)

    def _build_image_observation(self) -> Optional[bytes]:
        try:
            frame = self._render_frame()
            if frame is not None:
                rgb = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
                for idx, color in enumerate(self.ARC_PALETTE):
                    mask = frame == idx
                    rgb[mask] = color
                return self._frame_to_png(rgb)
        except (ValueError, TypeError, OSError):
            pass
        return None

    def _build_state(self) -> GameState:
        g = self._engine
        lvl_data = _BUILDERS[g.level_index]()
        lvl_name = lvl_data[2]

        return GameState(
            text_observation=self._build_text_observation(),
            image_observation=self._build_image_observation(),
            valid_actions=self.get_actions() if not self._done else None,
            turn=self._total_turns,
            metadata={
                "total_levels": len(self._engine._levels),
                "level_index": g.level_index,
                "levels_completed": getattr(self._engine, "_score", 0),
                "game_over": self._game_over,
                "done": self._done,
                "info": {},
                "level": g.level_index + 1,
                "level_name": lvl_name,
                "lives": g._lives,
                "moves": g.progress_max - g.progress,
                "max_moves": g.progress_max,
                "balanced": _avl_ok(g.tree) if g.tree else True,
            },
        )

    def reset(self) -> GameState:
        e = self._engine
        reset_input = ActionInput(id=GameAction.RESET)
        if self._game_won or self._last_action_was_reset:
            e._force_full_reset = True
        e.perform_action(reset_input)
        self._done = False
        self._game_over = False
        self._total_turns = 0
        self._last_action_was_reset = True
        self._game_won = False
        return self._build_state()

    def get_actions(self) -> List[str]:
        if self._done or self._game_over:
            return ["reset"]
        return list(self._VALID_ACTIONS)

    def step(self, action: str) -> StepResult:
        parts = action.split()
        action_key = parts[0] if parts else action

        if action_key not in self._ACTION_MAP:
            return StepResult(
                state=self._build_state(),
                reward=0.0,
                done=False,
                info={"action": action, "error": "invalid_action"},
            )

        if action_key == "reset":
            state = self.reset()
            return StepResult(
                state=state, reward=0.0, done=False, info={"action": "reset"}
            )

        if self._done or self._game_over:
            return StepResult(
                state=self._build_state(),
                reward=0.0,
                done=self._done,
                info={"action": action, "error": "only_reset_allowed"},
            )

        self._last_action_was_reset = False
        e = self._engine
        game_action = self._ACTION_MAP[action_key]
        action_input = ActionInput(id=game_action)

        total_levels = len(e._levels)
        level_before = e.level_index
        result = e.perform_action(action_input)
        self._total_turns += 1

        reward = 0.0
        done = False
        info: Dict[str, Any] = {"action": action}

        if result.state == _EngineState.WIN:
            done = True
            self._done = True
            self._game_won = True
            reward = 1.0 / total_levels
            info["reason"] = "game_complete"
        elif result.state == _EngineState.GAME_OVER:
            self._game_over = True
            info["reason"] = "game_over"
        elif e.level_index > level_before:
            reward = 1.0 / total_levels

        return StepResult(
            state=self._build_state(), reward=reward, done=done, info=info
        )

    def is_done(self) -> bool:
        return self._done

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError(f"Unsupported render mode: {mode}")
        frame = self._render_frame()
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        for idx, color in enumerate(self.ARC_PALETTE):
            mask = frame == idx
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
