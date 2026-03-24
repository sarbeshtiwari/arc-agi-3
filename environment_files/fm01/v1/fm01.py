import random
from math import gcd
from typing import Dict, List, Optional, Tuple

from arcengine import (
    ARCBaseGame,
    Camera,
    GameAction,
    Level,
    Sprite,
)

BG = 0
BORDER = 5
LIFE_COL = 8
LIFE_LOST_COL = 4
BAR_OUTLINE = 3

MOVE_BAR_COL = 2
CURSOR_COL = 13
CURSOR_ROW = 11
MATCHED = 14

LIVES_PER_LEVEL = [5, 5, 5, 4, 3, 3]

BUDGET_PER_LEVEL = [24, 32, 40, 48, 62, 70]

BAR_COLORS = [9, 6, 7, 8, 10, 15, 12]

GRID_W = 32
GRID_H = 32
BORDER_COL_L = 0
BORDER_COL_R = 31
BORDER_ROW_T = 0
BORDER_ROW_B = 31
PTR_X = 1
BAR_X = 2
BAR_MAX_LEN = 28
INNER_TOP = 1
INNER_BOTTOM = 30

levels = [Level(sprites=[], grid_size=(GRID_W, GRID_H)) for _ in range(6)]

def _bar_len(num: int, den: int) -> int:
    return max(1, round(num / den * BAR_MAX_LEN))

def _build_pool() -> List[Tuple[int, int, int]]:
    seen_lengths: Dict[int, Tuple[int, int]] = {}
    for den in range(2, 16):
        for num in range(1, den):
            g = gcd(num, den)
            rn, rd = num // g, den // g
            bl = _bar_len(rn, rd)
            if bl not in seen_lengths:
                seen_lengths[bl] = (rn, rd)
    return sorted((bl, n, d) for bl, (n, d) in seen_lengths.items())

_POOL = _build_pool()

_LEVEL_PAIR_INDICES = [
    [1, 3, 6],
    [1, 3, 5, 7, 9],
    [1, 3, 5, 7, 9, 11],
    [0, 2, 4, 6, 8, 10, 11],
    [0, 1, 3, 5, 7, 9, 10, 11],
    list(range(12)),
]

_LEVEL_N_PAIRS = [2, 3, 4, 5, 6, 7]

def _level_fractions(level_idx: int) -> List[Tuple[int, int, int]]:
    n = _LEVEL_N_PAIRS[level_idx]
    candidates = [_POOL[i % len(_POOL)] for i in _LEVEL_PAIR_INDICES[level_idx]]
    seen: Dict[int, Tuple[int, int]] = {}
    for bl, num, den in candidates:
        if bl not in seen:
            seen[bl] = (num, den)
    unique = [(bl, n2, d2) for bl, (n2, d2) in seen.items()]
    unique.sort(key=lambda t: t[0])
    return unique[:n]

class Fm01(ARCBaseGame):

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)
        camera = Camera(
            background=BG,
            letter_box=5,
            width=min(GRID_W, 64),
            height=min(GRID_H, 64),
        )
        super().__init__(
            game_id="fm01",
            levels=levels,
            camera=camera,
            available_actions=[1, 2, 3, 4, 5],
        )

        self._bars: List[dict] = []
        self._cursor_row: int = 0
        self._selected_row: Optional[int] = None
        self._matches: int = 0
        self._n_pairs: int = 0
        self._ready: bool = False
        self._active_pair_id: int = 0
        self._lives: int = 5
        self._lives_max: int = 5
        self._budget: int = 0
        self._budget_max: int = 0

    def on_set_level(self, level: Level) -> None:
        if not hasattr(self, "_ready"):
            return

        self.current_level.remove_all_sprites()

        idx = self._current_level_index

        self._lives_max = LIVES_PER_LEVEL[idx] if idx < len(LIVES_PER_LEVEL) else 3
        self._lives = self._lives_max

        self._budget_max = BUDGET_PER_LEVEL[idx] if idx < len(BUDGET_PER_LEVEL) else 64
        self._budget = self._budget_max
        fractions = _level_fractions(idx)
        n_pairs = len(fractions)
        self._n_pairs = n_pairs
        self._matches = 0
        self._selected_row = None

        bars: List[dict] = []
        for pair_id, (bl, num, den) in enumerate(fractions):
            colour = BAR_COLORS[pair_id % len(BAR_COLORS)]
            for _ in range(2):
                bars.append(
                    {
                        "bar_len": bl,
                        "num": num,
                        "den": den,
                        "colour": colour,
                        "pair_id": pair_id,
                        "matched": False,
                    }
                )

        self._rng.shuffle(bars)

        for row_idx, bar in enumerate(bars):
            bar["row"] = row_idx

        self._bars = bars
        self._cursor_row = 0
        self._active_pair_id = 0
        self._ready = True
        self._render()

        print(
            f"[Fm01] L{idx + 1}/6 | "
            f"{n_pairs} pairs (ascending bar_len) | "
            f"budget={self._budget_max} | lives={self._lives}/{self._lives_max} | "
            f"fractions: {[(b['num'], b['den']) for b in bars[::2]]}"
        )

    def step(self) -> None:
        if not self._ready:
            self.complete_action()
            return

        action = self.action
        n_rows = len(self._bars)
        life_lost_this_step = False

        self._budget = max(0, self._budget - 1)

        if action.id == GameAction.ACTION1:
            self._cursor_row = max(0, self._cursor_row - 1)

        elif action.id == GameAction.ACTION2:
            self._cursor_row = min(n_rows - 1, self._cursor_row + 1)

        elif action.id == GameAction.ACTION3:
            for r in range(self._cursor_row - 1, -1, -1):
                if not self._bars[r]["matched"]:
                    self._cursor_row = r
                    break

        elif action.id == GameAction.ACTION4:
            for r in range(self._cursor_row + 1, n_rows):
                if not self._bars[r]["matched"]:
                    self._cursor_row = r
                    break

        elif action.id == GameAction.ACTION5:
            bar_here = self._bars[self._cursor_row]
            is_locked = (
                not bar_here["matched"] and bar_here["pair_id"] != self._active_pair_id
            )

            if bar_here["matched"]:
                pass

            elif is_locked:
                self._selected_row = None
                self._lives -= 1
                life_lost_this_step = True
                if self._lives <= 0:
                    self._lives = 0
                    self._render()
                    self.lose()
                    self.complete_action()
                    return

            elif self._selected_row is None:
                self._selected_row = self._cursor_row

            elif self._selected_row == self._cursor_row:
                self._selected_row = None

            else:
                bar_sel = self._bars[self._selected_row]
                if bar_here["pair_id"] == bar_sel["pair_id"]:
                    bar_here["matched"] = True
                    bar_sel["matched"] = True
                    self._matches += 1
                    self._selected_row = None
                    self._active_pair_id = self._next_active_pair_id()
                else:
                    self._selected_row = None
                    self._lives -= 1
                    life_lost_this_step = True
                    if self._lives <= 0:
                        self._lives = 0
                        self._render()
                        self.lose()
                        self.complete_action()
                        return

        self._render()

        if self._matches >= self._n_pairs:
            if self._current_level_index < len(self._levels) - 1:
                self.next_level()
            else:
                self.win()
            self.complete_action()
            return

        if self._budget <= 0 and not life_lost_this_step:
            if self._lose_life():
                self.complete_action()
                return
            self.complete_action()
            return

        self.complete_action()

    def _next_active_pair_id(self) -> int:
        matched_ids = {bar["pair_id"] for bar in self._bars if bar["matched"]}
        for pid in range(self._n_pairs):
            if pid not in matched_ids:
                return pid
        return self._n_pairs

    def _lose_life(self) -> bool:
        self._lives -= 1
        if self._lives <= 0:
            self._lives = 0
            self._render()
            self.lose()
            return True
        self._reset_level()
        print(
            f"[Fm01] Lost a life -- {self._lives}/{self._lives_max} remaining "
            f"(L{self._current_level_index + 1})"
        )
        return False

    def _reset_level(self) -> None:
        idx = self._current_level_index
        fractions = _level_fractions(idx)
        n_pairs = len(fractions)
        self._n_pairs = n_pairs
        self._matches = 0
        self._selected_row = None

        bars: List[dict] = []
        for pair_id, (bl, num, den) in enumerate(fractions):
            colour = BAR_COLORS[pair_id % len(BAR_COLORS)]
            for _ in range(2):
                bars.append(
                    {
                        "bar_len": bl,
                        "num": num,
                        "den": den,
                        "colour": colour,
                        "pair_id": pair_id,
                        "matched": False,
                    }
                )

        self._rng.shuffle(bars)
        for row_idx, bar in enumerate(bars):
            bar["row"] = row_idx

        self._bars = bars
        self._cursor_row = 0
        self._active_pair_id = 0
        self._budget = self._budget_max
        self._render()
        print(f"[Fm01] Reset -- L{idx + 1}")

    def _render(self) -> None:
        self.current_level.remove_all_sprites()

        n_rows = len(self._bars)

        usable_rows = INNER_BOTTOM - INNER_TOP + 1
        spacing = max(1, usable_rows // max(n_rows, 1))

        for bar in self._bars:
            row_idx = bar["row"]
            gy = INNER_TOP + min(row_idx * spacing, usable_rows - 1)
            bar["_gy"] = gy

            bl = bar["bar_len"]
            is_matched = bar["matched"]
            is_locked = not is_matched and bar["pair_id"] != self._active_pair_id
            is_selected = (
                self._selected_row is not None and self._selected_row == row_idx
            )
            is_cursor = self._cursor_row == row_idx

            if is_matched:
                fill = MATCHED
            elif is_selected:
                fill = CURSOR_ROW
            else:
                fill = bar["colour"]

            if is_cursor and is_selected:
                self._px(PTR_X, gy, CURSOR_ROW, f"ptr_{row_idx}", layer=4)
            elif is_cursor:
                self._px(PTR_X, gy, CURSOR_COL, f"ptr_{row_idx}", layer=4)
            elif is_selected:
                self._px(PTR_X, gy, CURSOR_ROW, f"ptr_{row_idx}", layer=3)

            for x in range(BAR_X, BAR_X + BAR_MAX_LEN):
                self._px(x, gy, BAR_OUTLINE, f"track_{row_idx}_{x}", layer=0)

            for x in range(BAR_X, BAR_X + bl):
                self._px(x, gy, fill, f"bar_{row_idx}_{x}", layer=1)

        self._draw_border()

    def _draw_border(self) -> None:
        for x in range(GRID_W):
            self._px(x, BORDER_ROW_T, BORDER, f"brt_{x}", layer=5)
        for x in range(GRID_W):
            self._px(x, BORDER_ROW_B, BORDER, f"brb_{x}", layer=5)
        for y in range(1, GRID_H - 1):
            self._px(BORDER_COL_L, y, BORDER, f"brl_{y}", layer=5)
            self._px(BORDER_COL_R, y, BORDER, f"brr_{y}", layer=5)

        lives_max = self._lives_max
        dot_spacing = 2
        total_dot_width = lives_max + (lives_max - 1) * (dot_spacing - 1)
        inner_w = GRID_W - 2
        start_x = 1 + (inner_w - total_dot_width) // 2
        lost = lives_max - self._lives
        for i in range(lives_max):
            col = LIFE_LOST_COL if i < lost else LIFE_COL
            self._px(start_x + i * dot_spacing, BORDER_ROW_T, col, f"life_{i}", layer=6)

        bar_slots = GRID_W - 2
        if self._budget_max > 0:
            filled = round((self._budget / self._budget_max) * bar_slots)
        else:
            filled = 0
        for slot in range(bar_slots):
            col = MOVE_BAR_COL if slot < filled else LIFE_LOST_COL
            self._px(1 + slot, BORDER_ROW_B, col, f"bar_{slot}", layer=6)

    def _px(self, x: int, y: int, color: int, name: str, layer: int = 0) -> None:
        if not (0 <= x < GRID_W and 0 <= y < GRID_H):
            return
        sp = Sprite(
            pixels=[[color]],
            name=name,
            visible=True,
            collidable=False,
            layer=layer,
        )
        sp.set_position(x, y)
        self.current_level.add_sprite(sp)

    @property
    def extra_state(self) -> dict:
        n_pairs = self._n_pairs
        matches = self._matches
        remaining = n_pairs - matches

        bar_info = []
        for bar in sorted(self._bars, key=lambda b: (b["pair_id"], b["row"])):
            bar_info.append(
                {
                    "row": bar["row"],
                    "num": bar["num"],
                    "den": bar["den"],
                    "bar_len": bar["bar_len"],
                    "pair_id": bar["pair_id"],
                    "matched": bar["matched"],
                    "selected": self._selected_row == bar["row"],
                    "cursor": self._cursor_row == bar["row"],
                }
            )

        selected_info = None
        if self._selected_row is not None:
            sb = self._bars[self._selected_row]
            selected_info = {
                "row": sb["row"],
                "num": sb["num"],
                "den": sb["den"],
                "bar_len": sb["bar_len"],
                "fraction_str": f"{sb['num']}/{sb['den']}",
            }

        hints = []
        pair_rows: Dict[int, List[int]] = {}
        for bar in self._bars:
            if not bar["matched"]:
                pid = bar["pair_id"]
                pair_rows.setdefault(pid, []).append(bar["row"])
        for pid in sorted(pair_rows.keys()):
            rows = pair_rows[pid]
            if len(rows) == 2:
                bar_ref = next(b for b in self._bars if b["pair_id"] == pid)
                hints.append(
                    {
                        "pair_id": pid,
                        "fraction": f"{bar_ref['num']}/{bar_ref['den']}",
                        "bar_len": bar_ref["bar_len"],
                        "rows": sorted(rows),
                    }
                )

        return {
            "snake_length": matches,
            "target_length": n_pairs,
            "speed_cells_s": 0,
            "speed_interval_ms": 0,
            "remaining_cells": remaining,
            "circuit_title": f"Fraction Matching – Level {self._current_level_index + 1}",
            "pairs_matched": matches,
            "pairs_total": n_pairs,
            "pairs_remaining": remaining,
            "cursor_row": self._cursor_row,
            "selected": selected_info,
            "unmatched_pairs": hints,
            "bars": bar_info,
            "active_pair_id": self._active_pair_id,
            "active_fraction": (f"{hints[0]['fraction']}" if hints else "none"),
            "lives": self._lives,
            "lives_max": self._lives_max,
            "budget_remaining": self._budget,
            "budget_max": self._budget_max,
            "level_features": [
                f"Lives: {self._lives}/{self._lives_max}",
                f"Budget: {self._budget}/{self._budget_max}",
                f"Pairs: {matches}/{n_pairs} matched",
                f"Remaining: {remaining}",
                f"Active (locked others cost a life): {hints[0]['fraction'] if hints else 'none'}",
                f"Selected: {selected_info['fraction_str'] if selected_info else 'none'}",
                "Match order: ascending | locked bars dimmed red pointer",
            ],
        }
