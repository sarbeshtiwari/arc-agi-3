# task3.py  –  Chain Weaver puzzle for ARC-AGI-3  (game-id: cweav-v1)
# ─────────────────────────────────────────────────────────────────────────────
# DIFFICULTY PROGRESSION  (6 levels, strictly increasing order)
# ─────────────────────────────────────────────────────────────────────────────
#  Level 1 – Easy       :  9×9  grid | 1 chain  | no junctions  | 60 steps | 3 lives
#  Level 2 – Moderate   : 11×11 grid | 2 chains | 1 junction     | 144 steps | 3 lives
#  Level 3 – Hard       : 13×13 grid | 3 chains | 2 junctions    | 260 steps | 3 lives
#  Level 4 – Very Hard  : 15×15 grid | 4 chains | 3 junctions    | 328 steps | 3 lives
#  Level 5 – Expert     : 17×17 grid | 5 chains | 4 junctions    | 536 steps | 3 lives
#  Level 6 – Master     : 19×19 grid | 6 chains | 5 junctions    | 760 steps | 3 lives
#
# CONCEPT  (Linked-List + Binary-Tree — Knight L-Shape connections!)
#   Coloured chains (linked lists) are laid out spatially on the grid.  Each
#   chain is a sequence of NODE cells where consecutive nodes form a chess
#   KNIGHT'S L-SHAPE (2 cells in one direction + 1 cell perpendicular).
#
#   JUNCTION NODES (binary tree branch-points) appear where chains split.
#   A junction has TWO outgoing L-shape connections (left-child, right-child)
#   and the player must choose the CORRECT branch.
#
#   INTERACTION MODEL:
#     1. Player moves YELLOW cursor freely on the grid (arrow keys/WASD)
#     2. Press SPACE on a node to select it
#     3. First: select the RED HEAD node (press SPACE) → turns GREEN
#     4. BEFORE head is selected: ALL magenta nodes are DECOYS (-1 life)
#     5. AFTER head: only magenta nodes that form an L-shape (knight move)
#        with the last GREEN node are valid selections
#     6. Non-L-shape magenta nodes become DECOYS when pressed (-1 life)
#     7. Pre-placed DECOY nodes also exist (always -1 life)
#     8. At JUNCTIONS: pick the correct L-shape branch
#     9. Reach the TAIL to complete the chain, repeat for all chains
#
# ── OBSTACLE LAYERS ──────────────────────────────────────────────────────────
#   1. ARROW FADING   – L3-L5: arrow connectors are INVISIBLE.
#   2. DECOY NODES    – Pre-placed orange fake nodes. SPACE = -1 life.
#   3. L-SHAPE DECOYS – Any magenta node NOT forming L-shape from last green.
#   4. WALL MAZE      – Grey walls blocking cursor movement paths.
#   5. TRAP CELLS     – Orange deadly floor cells.  Step on = -1 life.
#   6. GHOST NODES    – L4-L5: some real chain nodes start INVISIBLE (black).
#                       They only reveal when the previous node is visited.
#   7. TIGHT BUDGETS  – Limited steps; every move counts.
#
# COLOUR LEGEND  (ARC-AGI-3 palette, indices 0-15)
#  0  White       – (unused)
#  1  Off-white   – (unused)
#  2  Light Grey  – (unused)
#  3  Grey        – Wall (impassable — perimeter AND interior maze walls)
#  4  Dark Grey   – Exposed DECOY (after player steps on fake node)
#  5  Black       – Background floor (passable) / GHOST NODE (camouflaged!)
#  6  Magenta     – Chain NODE: unvisited / wrong-branch decoy (same appearance!)
#  7  Pink        – (unused)
#  8  Red         – HEAD node (chain start — select FIRST with SPACE)
#  9  Blue        – Step-bar HUD colour (right column)
#  10 Light Blue  – (unused in L3+, arrow hint in L1-L2)
#  11 Yellow      – Player CURSOR (the piece you move freely)
#  12 Orange      – TRAP CELL (step on = -1 life) / DECOY NODE (SPACE = -1 life)
#  13 Maroon      – TAIL node (chain end — reach this to complete a chain)
#  14 Green       – Chain NODE: correctly visited (turned green on SPACE)
#  15 Purple      – JUNCTION node (binary-tree split — choose left or right!)
#
# KNIGHT L-SHAPE RULE
#   Consecutive chain nodes are ALWAYS a chess knight's move apart:
#   (±1, ±2) or (±2, ±1) cells offset.  This is the core DSA pattern
#   that replaces arrow-following from the original design.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from arcengine import (
    ARCBaseGame,
    Camera,
    GameAction,
    Level,
    Sprite,
)

# ── Colour palette (ARC-AGI-3 standard, indices 0-15) ─────────────────────────
BG_COLOR        = 5    # Black       – background floor
FLOOR_COLOR     = 5    # Black       – passable floor tile
WALL_COLOR      = 3    # Grey        – wall (impassable)
CURSOR_COLOR    = 11   # Yellow      – player cursor
NODE_COLOR      = 6    # Magenta     – unvisited chain node (also decoy colour!)
VISITED_COLOR   = 14   # Green       – correctly visited node
HEAD_COLOR      = 8    # Red         – head of chain (entry point)
TAIL_COLOR      = 13   # Maroon      – tail of chain (exit point)
JUNCTION_COLOR  = 15   # Purple      – junction node (binary tree split)
TRAP_COLOR      = 12   # Orange      – deadly trap cell
DECOY_COLOR     = 12   # Orange      – decoy node (visibly dangerous!)
DECOY_DEAD      = 4    # Dark Grey   – exposed decoy
GHOST_COLOR     = 5    # Black       – ghost node (camouflaged as floor)

# ── HUD palette ───────────────────────────────────────────────────────────────
HUD_LIFE_ALIVE  = 6    # Magenta  – life pip: remaining
HUD_LIFE_LOST   = 3    # Grey     – life pip: lost
HUD_STEP_EMPTY  = 3    # Grey     – step bar: used

# ── Knight move offsets ───────────────────────────────────────────────────────
KNIGHT_OFFSETS = [
    (1, 2), (2, 1), (2, -1), (1, -2),
    (-1, 2), (-2, 1), (-2, -1), (-1, -2),
]

def _is_knight_move(a: tuple[int, int], b: tuple[int, int]) -> bool:
    dx, dy = abs(b[0] - a[0]), abs(b[1] - a[1])
    return (dx == 1 and dy == 2) or (dx == 2 and dy == 1)

# ── Sprite templates (1×1 pixel per grid cell) ───────────────────────────────
_SPRITES = {
    "wall": Sprite(
        pixels=[[WALL_COLOR]],
        name="wall", visible=True, collidable=True,
    ),
    "floor": Sprite(
        pixels=[[FLOOR_COLOR]],
        name="floor", visible=True, collidable=False,
    ),
    "node": Sprite(
        pixels=[[NODE_COLOR]],
        name="node", visible=True, collidable=False,
    ),
    "head_node": Sprite(
        pixels=[[HEAD_COLOR]],
        name="head_node", visible=True, collidable=False,
    ),
    "tail_node": Sprite(
        pixels=[[TAIL_COLOR]],
        name="tail_node", visible=True, collidable=False,
    ),
    "junction_node": Sprite(
        pixels=[[JUNCTION_COLOR]],
        name="junction_node", visible=True, collidable=False,
    ),
    "visited": Sprite(
        pixels=[[VISITED_COLOR]],
        name="visited", visible=True, collidable=False,
    ),
    "cursor": Sprite(
        pixels=[[CURSOR_COLOR]],
        name="cursor", visible=True, collidable=False,
    ),
    "trap": Sprite(
        pixels=[[TRAP_COLOR]],
        name="trap", visible=True, collidable=False,
    ),
    "decoy": Sprite(
        pixels=[[DECOY_COLOR]],
        name="decoy", visible=True, collidable=False,
    ),
    "decoy_dead": Sprite(
        pixels=[[DECOY_DEAD]],
        name="decoy_dead", visible=True, collidable=False,
    ),
    "ghost_node": Sprite(
        pixels=[[GHOST_COLOR]],
        name="ghost_node", visible=True, collidable=False,
    ),
}


# ══════════════════════════════════════════════════════════════════════════════
#  Linked-List + Binary-Tree data structures  (knight L-shape connections)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ChainNode:
    """
    A node in a linked-list chain placed at grid position (gx, gy).
    Consecutive nodes MUST form a chess knight's L-shape (2+1 cells).
    Junction nodes have TWO outgoing L-shape connections (left, right).
    """
    gx: int
    gy: int
    next_node: Optional["ChainNode"]   = None
    is_junction: bool                   = False
    left_branch: Optional["ChainNode"] = None
    right_branch: Optional["ChainNode"]= None
    correct_branch: str                 = ""     # "left" or "right"
    is_head: bool                       = False
    is_tail: bool                       = False
    visited: bool                       = False
    node_id: int                        = 0
    chain_id: int                       = 0
    ghost: bool                         = False
    revealed: bool                      = True


def flatten_chain(head: ChainNode) -> list[ChainNode]:
    """Flatten chain following correct branches. All links are knight L-shapes."""
    result: list[ChainNode] = []
    node: Optional[ChainNode] = head
    visited_ids: set[int] = set()
    while node is not None and node.node_id not in visited_ids:
        visited_ids.add(node.node_id)
        result.append(node)
        if node.is_junction:
            if node.correct_branch == "left":
                node = node.left_branch
            else:
                node = node.right_branch
        else:
            node = node.next_node
    return result


def get_all_chain_nodes(head: ChainNode) -> list[ChainNode]:
    """Get ALL nodes including both junction branches."""
    result: list[ChainNode] = []
    stack: list[ChainNode] = [head]
    seen: set[int] = set()
    while stack:
        n = stack.pop()
        if n.node_id in seen:
            continue
        seen.add(n.node_id)
        result.append(n)
        if n.is_junction:
            if n.left_branch:
                stack.append(n.left_branch)
            if n.right_branch:
                stack.append(n.right_branch)
        if n.next_node and n.next_node.node_id not in seen:
            stack.append(n.next_node)
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  Level definitions — knight L-shape chains, junctions, obstacles
# ══════════════════════════════════════════════════════════════════════════════

BuildResult = tuple[
    list[ChainNode],                    # chain heads
    list[tuple[int, int]],              # decoy positions
    list[tuple[int, int]],              # trap positions
    list[tuple[int, int]],              # interior wall positions
    set[int],                           # ghost node ids
]


def _build_level1() -> BuildResult:
    """Level 1: 9×9, 1 chain, 5 nodes, no junctions. All L-shape links."""
    nid = 0
    n0 = ChainNode(gx=2, gy=2, is_head=True, node_id=nid, chain_id=0); nid += 1
    n1 = ChainNode(gx=3, gy=4, node_id=nid, chain_id=0); nid += 1
    n2 = ChainNode(gx=4, gy=6, node_id=nid, chain_id=0); nid += 1
    n3 = ChainNode(gx=6, gy=5, node_id=nid, chain_id=0); nid += 1
    n4 = ChainNode(gx=7, gy=3, is_tail=True, node_id=nid, chain_id=0); nid += 1
    n0.next_node = n1; n1.next_node = n2; n2.next_node = n3; n3.next_node = n4
    decoys = [(5, 3), (4, 4)]
    traps  = [(3, 6)]
    walls  = [(3, 3), (5, 2), (6, 7)]
    return [n0], decoys, traps, walls, set()


def _build_level2() -> BuildResult:
    """Level 2: 11×11, 2 chains, 1 junction."""
    nid = 0
    a0 = ChainNode(gx=2, gy=2, is_head=True, node_id=nid, chain_id=0); nid += 1
    a1 = ChainNode(gx=4, gy=3, node_id=nid, chain_id=0); nid += 1
    a2 = ChainNode(gx=5, gy=5, is_junction=True, node_id=nid, chain_id=0); nid += 1
    a3_l = ChainNode(gx=4, gy=7, node_id=nid, chain_id=0); nid += 1
    a3_r = ChainNode(gx=7, gy=6, node_id=nid, chain_id=0); nid += 1
    a4 = ChainNode(gx=5, gy=9, is_tail=True, node_id=nid, chain_id=0); nid += 1
    a0.next_node = a1; a1.next_node = a2
    a2.left_branch = a3_l; a2.right_branch = a3_r; a2.correct_branch = "left"
    a3_l.next_node = a4
    b0 = ChainNode(gx=8, gy=2, is_head=True, node_id=nid, chain_id=1); nid += 1
    b1 = ChainNode(gx=9, gy=4, node_id=nid, chain_id=1); nid += 1
    b2 = ChainNode(gx=8, gy=6, is_tail=True, node_id=nid, chain_id=1); nid += 1
    b0.next_node = b1; b1.next_node = b2
    decoys = [(3, 5), (6, 8), (7, 4)]
    traps  = [(4, 5), (6, 3)]
    walls  = [(3, 3), (6, 2), (2, 5), (9, 7)]
    return [a0, b0], decoys, traps, walls, set()


def _build_level3() -> BuildResult:
    """Level 3: 13×13, 3 chains, 2 junctions. Arrows faded."""
    nid = 0
    a0 = ChainNode(gx=2, gy=2, is_head=True, node_id=nid, chain_id=0); nid += 1
    a1 = ChainNode(gx=4, gy=3, node_id=nid, chain_id=0); nid += 1
    a2 = ChainNode(gx=6, gy=2, node_id=nid, chain_id=0); nid += 1
    a3 = ChainNode(gx=7, gy=4, is_junction=True, node_id=nid, chain_id=0); nid += 1
    a4_l = ChainNode(gx=5, gy=5, node_id=nid, chain_id=0); nid += 1
    a4_r = ChainNode(gx=9, gy=5, node_id=nid, chain_id=0); nid += 1
    a5 = ChainNode(gx=10, gy=7, is_tail=True, node_id=nid, chain_id=0); nid += 1
    a0.next_node = a1; a1.next_node = a2; a2.next_node = a3
    a3.left_branch = a4_l; a3.right_branch = a4_r; a3.correct_branch = "right"
    a4_r.next_node = a5
    b0 = ChainNode(gx=2, gy=6, is_head=True, node_id=nid, chain_id=1); nid += 1
    b1 = ChainNode(gx=3, gy=8, is_junction=True, node_id=nid, chain_id=1); nid += 1
    b2_l = ChainNode(gx=1, gy=9, node_id=nid, chain_id=1); nid += 1
    b2_r = ChainNode(gx=5, gy=9, node_id=nid, chain_id=1); nid += 1
    b3 = ChainNode(gx=4, gy=11, is_tail=True, node_id=nid, chain_id=1); nid += 1
    b0.next_node = b1
    b1.left_branch = b2_l; b1.right_branch = b2_r; b1.correct_branch = "right"
    b2_r.next_node = b3
    c0 = ChainNode(gx=10, gy=2, is_head=True, node_id=nid, chain_id=2); nid += 1
    c1 = ChainNode(gx=11, gy=4, node_id=nid, chain_id=2); nid += 1
    c2 = ChainNode(gx=10, gy=6, node_id=nid, chain_id=2); nid += 1
    c3 = ChainNode(gx=11, gy=8, is_tail=True, node_id=nid, chain_id=2); nid += 1
    c0.next_node = c1; c1.next_node = c2; c2.next_node = c3
    decoys = [(6, 6), (4, 8), (8, 10), (3, 4)]
    traps  = [(5, 3), (7, 7), (9, 3)]
    walls  = [(3, 3), (8, 2), (1, 5),
              (4, 9), (6, 9), (11, 10)]
    return [a0, b0, c0], decoys, traps, walls, set()


def _build_level4() -> BuildResult:
    """Level 4: 15×15, 4 chains, 3 junctions. Ghost nodes."""
    nid = 0
    a0 = ChainNode(gx=2, gy=2, is_head=True, node_id=nid, chain_id=0); nid += 1
    a1 = ChainNode(gx=4, gy=3, node_id=nid, chain_id=0); nid += 1
    a2 = ChainNode(gx=3, gy=5, is_junction=True, node_id=nid, chain_id=0); nid += 1
    a3_l = ChainNode(gx=2, gy=7, node_id=nid, chain_id=0, ghost=True, revealed=False); nid += 1
    a3_r = ChainNode(gx=5, gy=6, node_id=nid, chain_id=0); nid += 1
    a4 = ChainNode(gx=3, gy=9, is_tail=True, node_id=nid, chain_id=0); nid += 1
    a0.next_node = a1; a1.next_node = a2
    a2.left_branch = a3_l; a2.right_branch = a3_r; a2.correct_branch = "left"
    a3_l.next_node = a4
    b0 = ChainNode(gx=8, gy=2, is_head=True, node_id=nid, chain_id=1); nid += 1
    b1 = ChainNode(gx=10, gy=3, node_id=nid, chain_id=1); nid += 1
    b2 = ChainNode(gx=12, gy=2, node_id=nid, chain_id=1); nid += 1
    b3 = ChainNode(gx=11, gy=4, node_id=nid, chain_id=1); nid += 1
    b4 = ChainNode(gx=12, gy=6, is_junction=True, node_id=nid, chain_id=1); nid += 1
    b5_l = ChainNode(gx=10, gy=7, node_id=nid, chain_id=1); nid += 1
    b5_r = ChainNode(gx=13, gy=8, node_id=nid, chain_id=1, ghost=True, revealed=False); nid += 1
    b6 = ChainNode(gx=12, gy=10, is_tail=True, node_id=nid, chain_id=1); nid += 1
    b0.next_node = b1; b1.next_node = b2; b2.next_node = b3; b3.next_node = b4
    b4.left_branch = b5_l; b4.right_branch = b5_r; b4.correct_branch = "right"
    b5_r.next_node = b6
    c0 = ChainNode(gx=6, gy=8, is_head=True, node_id=nid, chain_id=2); nid += 1
    c1 = ChainNode(gx=7, gy=10, node_id=nid, chain_id=2); nid += 1
    c2 = ChainNode(gx=6, gy=12, is_junction=True, node_id=nid, chain_id=2); nid += 1
    c3_l = ChainNode(gx=4, gy=13, is_tail=True, node_id=nid, chain_id=2); nid += 1
    c3_r = ChainNode(gx=8, gy=13, node_id=nid, chain_id=2); nid += 1
    c0.next_node = c1; c1.next_node = c2
    c2.left_branch = c3_l; c2.right_branch = c3_r; c2.correct_branch = "left"
    d0 = ChainNode(gx=2, gy=12, is_head=True, node_id=nid, chain_id=3); nid += 1
    d1 = ChainNode(gx=3, gy=10, is_tail=True, node_id=nid, chain_id=3); nid += 1
    d0.next_node = d1
    ghost_ids = {a3_l.node_id, b5_r.node_id}
    decoys = [(4, 6), (8, 6), (10, 6), (5, 4), (11, 10)]
    traps  = [(3, 3), (9, 3), (7, 11)]
    walls  = [(5, 2), (7, 4), (9, 4), (1, 6), (7, 5), (13, 4),
              (1, 9), (5, 9), (9, 9), (11, 9),
              (4, 11), (8, 11), (10, 11)]
    return [a0, b0, c0, d0], decoys, traps, walls, ghost_ids


# def _build_level5() -> BuildResult:
#     """Level 5: 17×17, 5 chains, 4 junctions. Ghost nodes. Maximum difficulty."""
#     nid = 0
#     a0 = ChainNode(gx=2, gy=2, is_head=True, node_id=nid, chain_id=0); nid += 1
#     a1 = ChainNode(gx=4, gy=3, node_id=nid, chain_id=0); nid += 1
#     a2 = ChainNode(gx=6, gy=2, node_id=nid, chain_id=0); nid += 1
#     a3 = ChainNode(gx=7, gy=4, is_junction=True, node_id=nid, chain_id=0); nid += 1
#     a4_l = ChainNode(gx=5, gy=5, node_id=nid, chain_id=0); nid += 1
#     a4_r = ChainNode(gx=9, gy=5, node_id=nid, chain_id=0); nid += 1
#     a5 = ChainNode(gx=8, gy=7, node_id=nid, chain_id=0, ghost=True, revealed=False); nid += 1
#     a6 = ChainNode(gx=9, gy=9, is_tail=True, node_id=nid, chain_id=0); nid += 1
#     a0.next_node = a1; a1.next_node = a2; a2.next_node = a3
#     a3.left_branch = a4_l; a3.right_branch = a4_r; a3.correct_branch = "right"
#     a4_r.next_node = a5; a5.next_node = a6
#     b0 = ChainNode(gx=12, gy=2, is_head=True, node_id=nid, chain_id=1); nid += 1
#     b1 = ChainNode(gx=14, gy=3, node_id=nid, chain_id=1); nid += 1
#     b2 = ChainNode(gx=13, gy=5, node_id=nid, chain_id=1); nid += 1
#     b3 = ChainNode(gx=14, gy=7, is_junction=True, node_id=nid, chain_id=1); nid += 1
#     b4_l = ChainNode(gx=12, gy=8, node_id=nid, chain_id=1); nid += 1
#     b4_r = ChainNode(gx=15, gy=9, node_id=nid, chain_id=1); nid += 1
#     b5 = ChainNode(gx=13, gy=10, node_id=nid, chain_id=1, ghost=True, revealed=False); nid += 1
#     b6 = ChainNode(gx=12, gy=12, is_tail=True, node_id=nid, chain_id=1); nid += 1
#     b0.next_node = b1; b1.next_node = b2; b2.next_node = b3
#     b3.left_branch = b4_l; b3.right_branch = b4_r; b3.correct_branch = "left"
#     b4_l.next_node = b5; b5.next_node = b6
#     c0 = ChainNode(gx=2, gy=8, is_head=True, node_id=nid, chain_id=2); nid += 1
#     c1 = ChainNode(gx=3, gy=10, node_id=nid, chain_id=2); nid += 1
#     c2 = ChainNode(gx=2, gy=12, is_junction=True, node_id=nid, chain_id=2); nid += 1
#     c3_l = ChainNode(gx=1, gy=14, is_tail=True, node_id=nid, chain_id=2); nid += 1
#     c3_r = ChainNode(gx=4, gy=13, node_id=nid, chain_id=2); nid += 1
#     c0.next_node = c1; c1.next_node = c2
#     c2.left_branch = c3_l; c2.right_branch = c3_r; c2.correct_branch = "left"
#     d0 = ChainNode(gx=6, gy=10, is_head=True, node_id=nid, chain_id=3); nid += 1
#     d1 = ChainNode(gx=7, gy=12, node_id=nid, chain_id=3, ghost=True, revealed=False); nid += 1
#     d2 = ChainNode(gx=6, gy=14, is_junction=True, node_id=nid, chain_id=3); nid += 1
#     d3_l = ChainNode(gx=5, gy=12, node_id=nid, chain_id=3); nid += 1
#     d3_r = ChainNode(gx=8, gy=15, is_tail=True, node_id=nid, chain_id=3); nid += 1
#     d0.next_node = d1; d1.next_node = d2
#     d2.left_branch = d3_l; d2.right_branch = d3_r; d2.correct_branch = "right"
#     e0 = ChainNode(gx=10, gy=10, is_head=True, node_id=nid, chain_id=4); nid += 1
#     e1 = ChainNode(gx=11, gy=12, node_id=nid, chain_id=4); nid += 1
#     e2 = ChainNode(gx=10, gy=14, node_id=nid, chain_id=4); nid += 1
#     e3 = ChainNode(gx=12, gy=15, is_tail=True, node_id=nid, chain_id=4); nid += 1
#     e0.next_node = e1; e1.next_node = e2; e2.next_node = e3
#     ghost_ids = {a5.node_id, b5.node_id, d1.node_id}
#     decoys = [(4, 4), (10, 4), (8, 4), (14, 10), (4, 12), (3, 8), (11, 14)]
#     traps  = [(3, 3), (5, 3), (13, 3), (15, 5), (3, 11), (7, 13)]
#     walls  = [(3, 2), (5, 4), (7, 2), (11, 2), (13, 4),
#               (1, 5), (3, 5), (11, 5), (15, 4),
#               (1, 7), (3, 7), (5, 7), (11, 7), (13, 7), (15, 7),
#               (1, 9), (5, 9), (7, 9), (11, 9), (13, 9),
#               (4, 11), (8, 11), (14, 11),
#               (3, 13), (5, 13), (9, 13), (11, 13), (13, 13)]
#     return [a0, b0, c0, d0, e0], decoys, traps, walls, ghost_ids


# def _build_level6() -> BuildResult:
#     """
#     Level 6: 19×19, 6 chains, 5 junctions, 4 ghost nodes. Ultimate difficulty.
#     Chain A: HEAD(2,2) -L-> (4,3) -L-> (6,2) -L-> (7,4) -L-> JUNCTION(8,6)
#              left: (6,7)                                       [WRONG]
#              right: GHOST(10,7) -L-> (11,9) -L-> TAIL(12,7)   [CORRECT]
#     Chain B: HEAD(14,2) -L-> (16,3) -L-> JUNCTION(15,5)
#              left: GHOST(14,7) -L-> TAIL(15,9)                [CORRECT]
#              right: (17,6)                                      [WRONG]
#     Chain C: HEAD(2,10) -L-> (3,12) -L-> JUNCTION(2,14)
#              left: TAIL(1,16)                                   [CORRECT]
#              right: (4,15)                                      [WRONG]
#     Chain D: HEAD(7,10) -L-> GHOST(8,12) -L-> JUNCTION(7,14)
#              left: (5,15)                                       [WRONG]
#              right: (9,15) -L-> TAIL(10,17)                    [CORRECT]
#     Chain E: HEAD(12,10) -L-> (13,12) -L-> JUNCTION(12,14)
#              left: (11,16)                                      [WRONG]
#              right: GHOST(14,15) -L-> TAIL(15,17)              [CORRECT]
#     Chain F: HEAD(16,10) -L-> (17,12) -L-> (16,14) -L-> TAIL(17,16)
#     """
#     nid = 0
#     # Chain A: 8 correct-path nodes, 1 junction, 1 ghost
#     a0 = ChainNode(gx=2, gy=2, is_head=True, node_id=nid, chain_id=0); nid += 1
#     a1 = ChainNode(gx=4, gy=3, node_id=nid, chain_id=0); nid += 1
#     a2 = ChainNode(gx=6, gy=2, node_id=nid, chain_id=0); nid += 1
#     a3 = ChainNode(gx=7, gy=4, node_id=nid, chain_id=0); nid += 1
#     a4 = ChainNode(gx=8, gy=6, is_junction=True, node_id=nid, chain_id=0); nid += 1
#     a5_l = ChainNode(gx=6, gy=7, node_id=nid, chain_id=0); nid += 1  # WRONG
#     a5_r = ChainNode(gx=10, gy=7, node_id=nid, chain_id=0, ghost=True, revealed=False); nid += 1  # CORRECT
#     a6 = ChainNode(gx=11, gy=9, node_id=nid, chain_id=0); nid += 1
#     a7 = ChainNode(gx=12, gy=7, is_tail=True, node_id=nid, chain_id=0); nid += 1
#     a0.next_node = a1; a1.next_node = a2; a2.next_node = a3; a3.next_node = a4
#     a4.left_branch = a5_l; a4.right_branch = a5_r; a4.correct_branch = "right"
#     a5_r.next_node = a6; a6.next_node = a7

#     # Chain B: 5 correct-path nodes, 1 junction, 1 ghost
#     b0 = ChainNode(gx=14, gy=2, is_head=True, node_id=nid, chain_id=1); nid += 1
#     b1 = ChainNode(gx=16, gy=3, node_id=nid, chain_id=1); nid += 1
#     b2 = ChainNode(gx=15, gy=5, is_junction=True, node_id=nid, chain_id=1); nid += 1
#     b3_l = ChainNode(gx=14, gy=7, node_id=nid, chain_id=1, ghost=True, revealed=False); nid += 1  # CORRECT
#     b3_r = ChainNode(gx=17, gy=6, node_id=nid, chain_id=1); nid += 1  # WRONG
#     b4 = ChainNode(gx=15, gy=9, is_tail=True, node_id=nid, chain_id=1); nid += 1
#     b0.next_node = b1; b1.next_node = b2
#     b2.left_branch = b3_l; b2.right_branch = b3_r; b2.correct_branch = "left"
#     b3_l.next_node = b4

#     # Chain C: 4 correct-path nodes, 1 junction, 0 ghost
#     c0 = ChainNode(gx=2, gy=10, is_head=True, node_id=nid, chain_id=2); nid += 1
#     c1 = ChainNode(gx=3, gy=12, node_id=nid, chain_id=2); nid += 1
#     c2 = ChainNode(gx=2, gy=14, is_junction=True, node_id=nid, chain_id=2); nid += 1
#     c3_l = ChainNode(gx=1, gy=16, is_tail=True, node_id=nid, chain_id=2); nid += 1  # CORRECT
#     c3_r = ChainNode(gx=4, gy=15, node_id=nid, chain_id=2); nid += 1  # WRONG
#     c0.next_node = c1; c1.next_node = c2
#     c2.left_branch = c3_l; c2.right_branch = c3_r; c2.correct_branch = "left"

#     # Chain D: 5 correct-path nodes, 1 junction, 1 ghost
#     d0 = ChainNode(gx=7, gy=10, is_head=True, node_id=nid, chain_id=3); nid += 1
#     d1 = ChainNode(gx=8, gy=12, node_id=nid, chain_id=3, ghost=True, revealed=False); nid += 1
#     d2 = ChainNode(gx=7, gy=14, is_junction=True, node_id=nid, chain_id=3); nid += 1
#     d3_l = ChainNode(gx=5, gy=15, node_id=nid, chain_id=3); nid += 1  # WRONG
#     d3_r = ChainNode(gx=9, gy=15, node_id=nid, chain_id=3); nid += 1  # CORRECT
#     d4 = ChainNode(gx=10, gy=17, is_tail=True, node_id=nid, chain_id=3); nid += 1
#     d0.next_node = d1; d1.next_node = d2
#     d2.left_branch = d3_l; d2.right_branch = d3_r; d2.correct_branch = "right"
#     d3_r.next_node = d4

#     # Chain E: 5 correct-path nodes, 1 junction, 1 ghost
#     e0 = ChainNode(gx=12, gy=10, is_head=True, node_id=nid, chain_id=4); nid += 1
#     e1 = ChainNode(gx=13, gy=12, node_id=nid, chain_id=4); nid += 1
#     e2 = ChainNode(gx=12, gy=14, is_junction=True, node_id=nid, chain_id=4); nid += 1
#     e3_l = ChainNode(gx=11, gy=16, node_id=nid, chain_id=4); nid += 1  # WRONG
#     e3_r = ChainNode(gx=14, gy=15, node_id=nid, chain_id=4, ghost=True, revealed=False); nid += 1  # CORRECT
#     e4 = ChainNode(gx=15, gy=17, is_tail=True, node_id=nid, chain_id=4); nid += 1
#     e0.next_node = e1; e1.next_node = e2
#     e2.left_branch = e3_l; e2.right_branch = e3_r; e2.correct_branch = "right"
#     e3_r.next_node = e4

#     # Chain F: 4 nodes, 0 junctions, 0 ghost (simple)
#     f0 = ChainNode(gx=16, gy=10, is_head=True, node_id=nid, chain_id=5); nid += 1
#     f1 = ChainNode(gx=17, gy=12, node_id=nid, chain_id=5); nid += 1
#     f2 = ChainNode(gx=16, gy=14, node_id=nid, chain_id=5); nid += 1
#     f3 = ChainNode(gx=17, gy=16, is_tail=True, node_id=nid, chain_id=5); nid += 1
#     f0.next_node = f1; f1.next_node = f2; f2.next_node = f3

#     ghost_ids = {a5_r.node_id, b3_l.node_id, d1.node_id, e3_r.node_id}
#     decoys = [(5, 4), (9, 4), (3, 6), (13, 4), (16, 8),
#               (6, 11), (10, 12), (3, 15), (8, 16), (14, 12)]
#     traps  = [(4, 5), (11, 3), (15, 4), (1, 11), (6, 13),
#               (10, 13), (16, 16), (13, 17)]
#     walls  = [(3, 2), (5, 2), (7, 2), (11, 2), (13, 2), (15, 2), (17, 2),
#               (1, 4), (3, 4), (5, 6), (9, 6), (11, 4), (13, 6), (17, 4),
#               (1, 8), (3, 8), (5, 8), (9, 8), (11, 8), (13, 8),
#               (1, 10), (5, 10), (9, 10), (11, 10), (15, 10), (17, 10),
#               (4, 12), (6, 12), (10, 14), (14, 12), (16, 12),
#               (2, 16), (4, 16), (6, 16), (8, 14), (12, 16),
#               (3, 17), (7, 17), (13, 16)]
#     return [a0, b0, c0, d0, e0, f0], decoys, traps, walls, ghost_ids


# ── Level descriptors ─────────────────────────────────────────────────────────
LEVEL_INFO = [
    {
        "desc":        "9x9 - Knight's First Link",
        "difficulty":  "Easy [1/6]",
        "max_steps":   60,
        "lives":       3,
        "grid":        (9, 9),
        "builder":     _build_level1,
        "show_arrows": True,
    },
    {
        "desc":        "11x11 - Fork in the L-Shape",
        "difficulty":  "Moderate [2/6]",
        "max_steps":   144,
        "lives":       3,
        "grid":        (11, 11),
        "builder":     _build_level2,
        "show_arrows": True,
    },
    {
        "desc":        "13x13 - Blind L-Links",
        "difficulty":  "Hard [3/6]",
        "max_steps":   260,
        "lives":       3,
        "grid":        (13, 13),
        "builder":     _build_level3,
        "show_arrows": False,
    },
    {
        "desc":        "15x15 - Ghost Knights",
        "difficulty":  "Very Hard [4/6]",
        "max_steps":   328,
        "lives":       3,
        "grid":        (15, 15),
        "builder":     _build_level4,
        "show_arrows": False,
    },
    # {
    #     "desc":        "17x17 - The Grand L-Weave",
    #     "difficulty":  "Expert [5/6]",
    #     "max_steps":   536,
    #     "lives":       3,
    #     "grid":        (17, 17),
    #     "builder":     _build_level5,
    #     "show_arrows": False,
    # },
    # {
    #     "desc":        "19x19 - The Phantom Lattice",
    #     "difficulty":  "Master [6/6]",
    #     "max_steps":   760,
    #     "lives":       3,
    #     "grid":        (19, 19),
    #     "builder":     _build_level6,
    #     "show_arrows": False,
    # },
]

LEVEL_THEMES = [
    {"background": 5, "letter_box": 14, "name": "Thread Garden"},
    {"background": 5, "letter_box": 9,  "name": "Tangled Web"},
    {"background": 5, "letter_box": 13, "name": "Broken Links"},
    {"background": 5, "letter_box": 10, "name": "Ghost Circuit"},
    {"background": 5, "letter_box": 8,  "name": "Grand Loom"},
    {"background": 5, "letter_box": 15, "name": "Phantom Lattice"},
]

# ── Build Level objects ───────────────────────────────────────────────────────
_levels: list[Level] = []
for _i, _info in enumerate(LEVEL_INFO):
    _c, _r = _info["grid"]
    _lvl = Level(sprites=[], grid_size=(_c, _r))
    _lvl.map_index = _i
    _levels.append(_lvl)


# ══════════════════════════════════════════════════════════════════════════════
#  Game class
# ══════════════════════════════════════════════════════════════════════════════

class Cw45(ARCBaseGame):
    """
    Chain Weaver puzzle  (cweav-v1).

    Knight L-Shape chain traversal game.  Linked-list chains of nodes are
    connected by chess knight moves (2+1 cells).  Player moves cursor
    freely and presses SPACE to select nodes.

    ── DSA CONCEPTS ─────────────────────────────────────────────────────
    LINKED LIST : Each chain is a singly-linked list with knight-move links.
    BINARY TREE : Junction nodes split into left/right knight-move branches.

    ── INTERACTION ──────────────────────────────────────────────────────
    ACTION1-4 : Move cursor (up/down/left/right)
    ACTION5   : SPACE — select node under cursor

    ── RULES ───────────────────────────────────────────────────────────
    1. Before HEAD selected: ALL magenta nodes are decoys
    2. Press SPACE on RED head → turns green, chain starts
    3. After head: only L-shape (knight move) magenta from last green = valid
    4. Non-L-shape press = DECOY (-1 life)
    5. Pre-placed decoy nodes always cost -1 life
    6. Trap cells cost -1 life on movement (not selection)
    """

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

        camera = Camera(
            background=BG_COLOR,
            letter_box=WALL_COLOR,
            width=19,
            height=19,
        )

        super().__init__(
            game_id="cweav-v1",
            levels=_levels,
            camera=camera,
            available_actions=[1, 2, 3, 4, 5],
        )

        self.cursor_pos: tuple[int, int] = (0, 0)
        self._cursor_sprite: Sprite | None = None

        self._chain_heads: list[ChainNode] = []
        self._current_chain_idx: int = 0
        self._current_chain_order: list[ChainNode] = []
        self._visit_index: int = 0
        self._all_chain_nodes: list[ChainNode] = []
        self._node_sprites: dict[int, Sprite] = {}

        self._lives:     int = 3
        self._steps:     int = 0
        self._max_steps: int = 30

        self._node_positions: dict[tuple[int, int], ChainNode] = {}
        self._decoy_positions: set[tuple[int, int]] = set()
        self._trap_positions: set[tuple[int, int]] = set()
        self._ghost_node_ids: set[int] = set()
        self._chains_completed: int = 0
        self._total_chains: int = 0
        self._head_selected: bool = False

    # ── arcengine callbacks ────────────────────────────────────────────────────

    def on_set_level(self, level: Level) -> None:
        if not hasattr(self, "_rng"):
            return

        idx  = max(0, min(self._current_level_index, len(LEVEL_INFO) - 1))
        info = LEVEL_INFO[idx]
        cols, rows = info["grid"]

        self._steps     = 0
        self._max_steps = info["max_steps"]
        self._lives     = info["lives"]
        self._visit_index = 0
        self._chains_completed = 0
        self._head_selected = False

        self.camera.width  = cols
        self.camera.height = rows

        theme = LEVEL_THEMES[idx % len(LEVEL_THEMES)]
        self.camera.background = theme["background"]
        self.camera.letter_box = theme["letter_box"]

        self.current_level.remove_all_sprites()

        # ── Place perimeter walls ─────────────────────────────────────────────
        for y in range(rows):
            for x in range(cols):
                if x == 0 or y == 0 or x == cols - 1 or y == rows - 1:
                    sp = _SPRITES["wall"].clone().set_position(x, y)
                    self.current_level.add_sprite(sp)

        # ── Build chains + obstacles ──────────────────────────────────────────
        (self._chain_heads, decoys, traps, walls,
         self._ghost_node_ids) = info["builder"]()

        # Reset visited state
        for head in self._chain_heads:
            self._reset_chain_visited(head)

        self._total_chains = len(self._chain_heads)
        self._current_chain_idx = 0

        self._all_chain_nodes = []
        for head in self._chain_heads:
            self._all_chain_nodes.extend(get_all_chain_nodes(head))
        self._node_positions = {(n.gx, n.gy): n for n in self._all_chain_nodes}

        if self._chain_heads:
            self._current_chain_order = flatten_chain(self._chain_heads[0])
        else:
            self._current_chain_order = []
        self._visit_index = 0

        # ── Place interior walls ──────────────────────────────────────────────
        occupied = set()
        occupied.update((n.gx, n.gy) for n in self._all_chain_nodes)
        for (wx, wy) in walls:
            if (0 < wx < cols - 1 and 0 < wy < rows - 1
                    and (wx, wy) not in occupied):
                sp = _SPRITES["wall"].clone().set_position(wx, wy)
                self.current_level.add_sprite(sp)
                occupied.add((wx, wy))

        # ── Place trap cells ──────────────────────────────────────────────────
        self._trap_positions = set()
        for (tx, ty) in traps:
            if (0 < tx < cols - 1 and 0 < ty < rows - 1
                    and (tx, ty) not in occupied):
                sp = _SPRITES["trap"].clone().set_position(tx, ty)
                self.current_level.add_sprite(sp)
                self._trap_positions.add((tx, ty))
                occupied.add((tx, ty))

        # ── Place decoy nodes ─────────────────────────────────────────────────
        self._decoy_positions = set()
        for (dx, dy) in decoys:
            if (0 < dx < cols - 1 and 0 < dy < rows - 1
                    and (dx, dy) not in occupied):
                sp = _SPRITES["decoy"].clone().set_position(dx, dy)
                self.current_level.add_sprite(sp)
                self._decoy_positions.add((dx, dy))
                occupied.add((dx, dy))

        # ── Draw chain nodes (progressive reveal: only active chain shows colours) ─
        self._node_sprites = {}
        for node in self._all_chain_nodes:
            if node.ghost and not node.revealed:
                sp = _SPRITES["ghost_node"].clone().set_position(node.gx, node.gy)
            elif node.chain_id == self._current_chain_idx:
                # Active chain: show full colour coding
                if node.is_head:
                    sp = _SPRITES["head_node"].clone().set_position(node.gx, node.gy)
                elif node.is_tail:
                    sp = _SPRITES["tail_node"].clone().set_position(node.gx, node.gy)
                elif node.is_junction:
                    sp = _SPRITES["junction_node"].clone().set_position(node.gx, node.gy)
                else:
                    sp = _SPRITES["node"].clone().set_position(node.gx, node.gy)
            else:
                # Inactive chain: all nodes look like plain magenta
                sp = _SPRITES["node"].clone().set_position(node.gx, node.gy)
            self._node_sprites[node.node_id] = sp
            self.current_level.add_sprite(sp)

        # ── Spawn cursor ─────────────────────────────────────────────────────
        cx = cols // 2
        cy = rows - 2
        while self._is_wall_at(cx, cy) and cy > 1:
            cy -= 1
        self.cursor_pos = (cx, cy)
        self._cursor_sprite = _SPRITES["cursor"].clone().set_position(cx, cy)
        self.current_level.add_sprite(self._cursor_sprite)

        self._refresh_hud()

        # Console banner
        print(f"\n{'='*64}")
        print(f"  LEVEL {idx+1}/{len(LEVEL_INFO)}  |  {info['desc']}")
        print(f"  Knight L-Shape Chain Traversal")
        print(f"  Chains: {self._total_chains}  Lives: {self._lives}  "
              f"Budget: {self._max_steps}")
        print(f"  SPACE to select nodes (L-shape rule)")
        print(f"{'='*64}\n")

    # ── Main step ──────────────────────────────────────────────────────────────

    def step(self) -> None:
        act = self.action.id
        idx  = self._current_level_index
        info = LEVEL_INFO[idx]
        cols, rows = info["grid"]

        # ACTION5 = SPACE (select node)
        if act == GameAction.ACTION5:
            self._handle_space_press(info)
            self._refresh_hud()
            self.complete_action()
            return

        # ACTION1-4 = movement
        dx, dy = 0, 0
        if act == GameAction.ACTION1:   dx, dy =  0, -1
        elif act == GameAction.ACTION2: dx, dy =  0,  1
        elif act == GameAction.ACTION3: dx, dy = -1,  0
        elif act == GameAction.ACTION4: dx, dy =  1,  0

        if dx == 0 and dy == 0:
            self.complete_action()
            return

        nx, ny = self.cursor_pos[0] + dx, self.cursor_pos[1] + dy

        if nx <= 0 or ny <= 0 or nx >= cols - 1 or ny >= rows - 1:
            self.complete_action()
            return

        if self._is_wall_at(nx, ny):
            self.complete_action()
            return

        self._steps += 1
        self.cursor_pos = (nx, ny)
        if self._cursor_sprite:
            self._cursor_sprite.set_position(nx, ny)
        self._bring_cursor_to_top()

        # Trap check on movement
        if (nx, ny) in self._trap_positions:
            self._lives -= 1
            print(f"  [TRAP!] at ({nx},{ny})  Lives: {self._lives}")
            self._refresh_hud()
            if self._lives <= 0:
                self.lose()
            else:
                self._reset_level()
            self.complete_action()
            return

        # Decoy check on movement (orange decoys also trigger on contact)
        if (nx, ny) in self._decoy_positions:
            self._lives -= 1
            self._expose_decoy(nx, ny)
            self._decoy_positions.discard((nx, ny))
            print(f"  [DECOY TOUCH!] at ({nx},{ny})  Lives: {self._lives}")
            self._refresh_hud()
            if self._lives <= 0:
                self.lose()
            else:
                self._reset_level()
            self.complete_action()
            return

        # Step budget
        if self._steps >= self._max_steps:
            self.lose()
            self.complete_action()
            return

        self._refresh_hud()
        self.complete_action()

    def _handle_space_press(self, info: dict) -> None:
        """Handle SPACE press — node selection with L-shape rule."""
        cx, cy = self.cursor_pos

        # Pre-placed decoy
        if (cx, cy) in self._decoy_positions:
            self._lives -= 1
            self._expose_decoy(cx, cy)
            self._decoy_positions.discard((cx, cy))
            print(f"  [DECOY!] at ({cx},{cy})  Lives: {self._lives}")
            if self._lives <= 0:
                self.lose()
            else:
                self._reset_level()
            return

        # Trap cell — pressing SPACE on orange also costs a life
        if (cx, cy) in self._trap_positions:
            self._lives -= 1
            print(f"  [TRAP SPACE!] at ({cx},{cy})  Lives: {self._lives}")
            if self._lives <= 0:
                self.lose()
            else:
                self._reset_level()
            return

        # Check chain node
        if (cx, cy) not in self._node_positions:
            return

        node = self._node_positions[(cx, cy)]

        if node.ghost and not node.revealed:
            return

        if node.visited:
            return

        # HEAD NOT YET SELECTED
        if not self._head_selected:
            if (self._visit_index < len(self._current_chain_order)
                    and node is self._current_chain_order[self._visit_index]
                    and node.is_head):
                node.visited = True
                self._visit_index += 1
                self._head_selected = True
                self._update_node_sprite(node)
                self._reveal_next_ghost(node)
                print(f"  [HEAD] Selected at ({cx},{cy})")
            else:
                self._lives -= 1
                print(f"  [NOT HEAD!] at ({cx},{cy})  Lives: {self._lives}")
                if self._lives <= 0:
                    self.lose()
                else:
                    self._reset_level()
            return

        # HEAD SELECTED — L-shape validation
        if self._visit_index >= len(self._current_chain_order):
            return

        if self._visit_index > 0:
            last = self._current_chain_order[self._visit_index - 1]
            last_pos = (last.gx, last.gy)
        else:
            return

        node_pos = (node.gx, node.gy)
        if not _is_knight_move(last_pos, node_pos):
            self._lives -= 1
            print(f"  [NOT L-SHAPE!] {last_pos}->{node_pos}  Lives: {self._lives}")
            if self._lives <= 0:
                self.lose()
            else:
                self._reset_level()
            return

        expected = self._current_chain_order[self._visit_index]
        if node is expected:
            node.visited = True
            self._visit_index += 1
            self._update_node_sprite(node)
            self._reveal_next_ghost(node)
            print(f"  [VISIT] ({cx},{cy}) correct  "
                  f"{self._visit_index}/{len(self._current_chain_order)}")

            if self._visit_index >= len(self._current_chain_order):
                self._chains_completed += 1
                print(f"  [CHAIN COMPLETE] {self._chains_completed}/{self._total_chains}")
                if self._chains_completed >= self._total_chains:
                    if self._current_level_index < len(self._levels) - 1:
                        self.next_level()
                    else:
                        self.win()
                else:
                    self._current_chain_idx += 1
                    self._current_chain_order = flatten_chain(
                        self._chain_heads[self._current_chain_idx])
                    self._visit_index = 0
                    self._head_selected = False
                    self._refresh_chain_sprites()  # Reveal new chain colours
        elif node.chain_id != self._current_chain_idx:
            self._lives -= 1
            print(f"  [WRONG CHAIN!] at ({cx},{cy})  Lives: {self._lives}")
            if self._lives <= 0:
                self.lose()
            else:
                self._reset_level()
        else:
            self._lives -= 1
            print(f"  [WRONG ORDER!] at ({cx},{cy})  Lives: {self._lives}")
            if self._lives <= 0:
                self.lose()
            else:
                self._reset_level()

    # ── Ghost reveal ──────────────────────────────────────────────────────────

    def _reveal_next_ghost(self, visited_node: ChainNode) -> None:
        successors = []
        if visited_node.is_junction:
            if visited_node.left_branch:
                successors.append(visited_node.left_branch)
            if visited_node.right_branch:
                successors.append(visited_node.right_branch)
        elif visited_node.next_node:
            successors.append(visited_node.next_node)

        for succ in successors:
            if succ.ghost and not succ.revealed:
                succ.revealed = True
                old_sp = self._node_sprites.get(succ.node_id)
                if old_sp:
                    self.current_level.remove_sprite(old_sp)
                new_sp = _SPRITES["node"].clone().set_position(succ.gx, succ.gy)
                self._node_sprites[succ.node_id] = new_sp
                self.current_level.add_sprite(new_sp)
                print(f"  [REVEAL] Ghost at ({succ.gx},{succ.gy})")
        self._bring_cursor_to_top()

    def _update_node_sprite(self, node: ChainNode) -> None:
        old_sp = self._node_sprites.get(node.node_id)
        if old_sp:
            self.current_level.remove_sprite(old_sp)
        new_sp = _SPRITES["visited"].clone().set_position(node.gx, node.gy)
        self._node_sprites[node.node_id] = new_sp
        self.current_level.add_sprite(new_sp)
        self._bring_cursor_to_top()

    def _refresh_chain_sprites(self) -> None:
        """
        Update all non-visited chain node sprites for progressive reveal.
        Called when the active chain changes so the new chain's head/tail/junction
        get their special colours, while all other chains remain plain magenta.
        """
        for node in self._all_chain_nodes:
            if node.visited:
                continue  # Already green, don't touch
            if node.ghost and not node.revealed:
                continue  # Still invisible, don't touch

            old_sp = self._node_sprites.get(node.node_id)
            if old_sp:
                self.current_level.remove_sprite(old_sp)

            if node.chain_id == self._current_chain_idx:
                # Active chain: show full colour coding
                if node.is_head:
                    sp = _SPRITES["head_node"].clone().set_position(node.gx, node.gy)
                elif node.is_tail:
                    sp = _SPRITES["tail_node"].clone().set_position(node.gx, node.gy)
                elif node.is_junction:
                    sp = _SPRITES["junction_node"].clone().set_position(node.gx, node.gy)
                else:
                    sp = _SPRITES["node"].clone().set_position(node.gx, node.gy)
            else:
                # Inactive chain: plain magenta
                sp = _SPRITES["node"].clone().set_position(node.gx, node.gy)

            self._node_sprites[node.node_id] = sp
            self.current_level.add_sprite(sp)

        # Re-add cursor on top so it's not hidden behind new sprites
        if self._cursor_sprite:
            self.current_level.remove_sprite(self._cursor_sprite)
            self.current_level.add_sprite(self._cursor_sprite)

        print(f"  [REVEAL] New active chain {self._current_chain_idx + 1} colours shown")

    def _bring_cursor_to_top(self) -> None:
        """Re-add cursor sprite so it renders on top of all other sprites."""
        if self._cursor_sprite:
            self.current_level.remove_sprite(self._cursor_sprite)
            self.current_level.add_sprite(self._cursor_sprite)

    def _expose_decoy(self, x: int, y: int) -> None:
        to_remove = [
            sp for sp in self.current_level.get_sprites()
            if sp.x == x and sp.y == y and sp.name == "decoy"
        ]
        for sp in to_remove:
            self.current_level.remove_sprite(sp)
        dead_sp = _SPRITES["decoy_dead"].clone().set_position(x, y)
        self.current_level.add_sprite(dead_sp)
        self._bring_cursor_to_top()

    def _is_wall_at(self, x: int, y: int) -> bool:
        for sp in self.current_level.get_sprites():
            if sp.x == x and sp.y == y and sp.name == "wall":
                return True
        return False

    # ── HUD ───────────────────────────────────────────────────────────────────

    def _refresh_hud(self) -> None:
        idx        = self._current_level_index
        info       = LEVEL_INFO[idx]
        cols, rows = info["grid"]
        theme      = LEVEL_THEMES[idx % len(LEVEL_THEMES)]

        to_remove = [
            sp for sp in self.current_level.get_sprites()
            if sp.name in ("hud_pip", "hud_bar", "hud_wall")
            and 0 < sp.y < rows - 1
            and (sp.x == 0 or sp.x == cols - 1)
        ]
        for sp in to_remove:
            self.current_level.remove_sprite(sp)

        to_remove_walls = [
            sp for sp in self.current_level.get_sprites()
            if sp.name == "wall"
            and 0 < sp.y < rows - 1
            and (sp.x == 0 or sp.x == cols - 1)
        ]
        for sp in to_remove_walls:
            self.current_level.remove_sprite(sp)

        interior_h = rows - 2

        # LEFT: life pips
        max_l = info["lives"]
        pip_ys = [1 + interior_h * (i + 1) // (max_l + 1) for i in range(max_l)]
        pip_set = set(pip_ys)
        for i, py in enumerate(pip_ys):
            color = HUD_LIFE_ALIVE if i < self._lives else HUD_LIFE_LOST
            sp = Sprite(pixels=[[color]], name="hud_pip",
                        visible=True, collidable=True)
            sp.set_position(0, py)
            self.current_level.add_sprite(sp)
        for wy in range(1, rows - 1):
            if wy not in pip_set:
                sp = Sprite(pixels=[[WALL_COLOR]], name="hud_wall",
                            visible=True, collidable=True)
                sp.set_position(0, wy)
                self.current_level.add_sprite(sp)

        # RIGHT: step bar
        steps_left = max(0, self._max_steps - self._steps)
        fraction   = steps_left / self._max_steps if self._max_steps > 0 else 0.0
        filled     = int(round(fraction * interior_h))
        step_full_color = theme["letter_box"]
        for i in range(interior_h):
            gy    = rows - 2 - i
            color = step_full_color if i < filled else HUD_STEP_EMPTY
            sp = Sprite(pixels=[[color]], name="hud_bar",
                        visible=True, collidable=True)
            sp.set_position(cols - 1, gy)
            self.current_level.add_sprite(sp)

        # Border warning
        if self._lives >= info["lives"]:
            self.camera.letter_box = theme["letter_box"]
        elif self._lives == 2:
            self.camera.letter_box = 11
        elif self._lives == 1:
            self.camera.letter_box = 8
        else:
            self.camera.letter_box = 3

        # Ensure cursor renders on top after HUD rebuild
        self._bring_cursor_to_top()

    def _reset_chain_visited(self, node: Optional[ChainNode]) -> None:
        if node is None:
            return
        stack: list[ChainNode] = [node]
        seen: set[int] = set()
        while stack:
            n = stack.pop()
            if n.node_id in seen:
                continue
            seen.add(n.node_id)
            n.visited = False
            if n.ghost:
                n.revealed = False
            if n.next_node:
                stack.append(n.next_node)
            if n.left_branch:
                stack.append(n.left_branch)
            if n.right_branch:
                stack.append(n.right_branch)

    def _reset_level(self) -> None:
        """Reset current level but preserve remaining lives."""
        saved_lives = self._lives
        try:
            self.set_level(self._current_level_index)
        except Exception:
            pass
        self._lives = saved_lives
        self._refresh_hud()

    @property
    def extra_state(self) -> dict:
        idx   = self._current_level_index
        info  = LEVEL_INFO[idx]
        theme = LEVEL_THEMES[idx % len(LEVEL_THEMES)]

        steps_left  = max(0, self._max_steps - self._steps)
        lives_left  = max(0, self._lives)
        step_pct    = (round(steps_left / self._max_steps * 100)
                       if self._max_steps else 0)
        lives_icons = "O" * lives_left + "X" * (info["lives"] - lives_left)

        return {
            "level":              idx + 1,
            "total_levels":       len(LEVEL_INFO),
            "level_desc":         info["desc"],
            "difficulty":         info["difficulty"],
            "lives":              lives_left,
            "max_lives":          info["lives"],
            "lives_icons":        lives_icons,
            "steps":              self._steps,
            "max_steps":          self._max_steps,
            "steps_left":         steps_left,
            "steps_pct":          step_pct,
            "total_chains":       self._total_chains,
            "chains_completed":   self._chains_completed,
            "current_chain":      self._current_chain_idx + 1,
            "chain_nodes_total":  len(self._current_chain_order),
            "chain_nodes_visited": self._visit_index,
            "head_selected":      self._head_selected,
            "ghost_remaining":    sum(
                1 for n in self._all_chain_nodes
                if n.ghost and not n.revealed
            ),
            "decoys_active":      len(self._decoy_positions),
            "traps_on_grid":      len(self._trap_positions),
            "cursor":             list(self.cursor_pos),
            "grid_size":          list(info["grid"]),
            "mechanic":           "Knight L-Shape (2+1 cells)",
            "ui_theme_name":      theme["name"],
        }
