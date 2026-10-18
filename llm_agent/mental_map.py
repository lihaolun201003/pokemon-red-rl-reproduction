"""Self-recorded mental map built only from the agent's own movement attempts.

No RAM, no map data, no external knowledge: the map is pure dead reckoning from
what the agent itself observed:

* a direction press whose screen changed a lot     -> a successful step
* a direction press whose screen changed almost 0   -> blocked (tried, cannot go)
* a small pixel change (character turning)         -> ambiguous, ignore
* a map change / fade / new area                   -> start a new local area

The rendered ASCII map is injected into every prompt so the model can route
around walls/fences it already discovered instead of retrying them forever.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from llm_agent.contracts import DIRECTIONS, ScreenEvent

#: changed-pixel ratio above this means the camera scrolled => a real one-tile move
MOVE_MIN_RATIO: float = 0.08
#: below this the screen is effectively identical => the direction is blocked
BLOCK_MAX_RATIO: float = 0.004
#: a whole-screen change (map transition) resets the local area
AREA_CHANGE_RATIO: float = 0.55

_DELTAS: dict[str, tuple[int, int]] = {
    "up": (0, 1),
    "down": (0, -1),
    "left": (-1, 0),
    "right": (1, 0),
}


@dataclass
class AreaMap:
    """One explored area (until a map transition resets the frame)."""

    area_id: int
    visited: set[tuple[int, int]] = field(default_factory=lambda: {(0, 0)})
    blocked: set[tuple[int, int]] = field(default_factory=set)
    edges: dict[tuple[int, int], dict[str, str]] = field(default_factory=dict)


class MentalMap:
    """Dead-reckoning grid map of the current area and past area summaries."""

    def __init__(
        self,
        *,
        move_min_ratio: float = MOVE_MIN_RATIO,
        block_max_ratio: float = BLOCK_MAX_RATIO,
        area_change_ratio: float = AREA_CHANGE_RATIO,
    ) -> None:
        self.move_min_ratio = float(move_min_ratio)
        self.block_max_ratio = float(block_max_ratio)
        self.area_change_ratio = float(area_change_ratio)
        self.pos: tuple[int, int] = (0, 0)
        self.area_id = 1
        self.areas: dict[int, AreaMap] = {1: AreaMap(1)}
        self.moves = 0
        self.blocked_attempts = 0
        self.area_switches = 0
        self.last_transition = "start"

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    @property
    def area(self) -> AreaMap:
        return self.areas.setdefault(self.area_id, AreaMap(self.area_id))

    @property
    def explored_cells(self) -> int:
        return sum(len(area.visited) for area in self.areas.values())

    # ------------------------------------------------------------------ #
    # Recording
    # ------------------------------------------------------------------ #
    def reset_area(self, reason: str = "transition") -> None:
        """Start a fresh local frame (map transition / fade / new area)."""
        self.area_id += 1
        self.pos = (0, 0)
        self.areas[self.area_id] = AreaMap(self.area_id)
        self.area_switches += 1
        self.last_transition = reason

    def update(
        self,
        steps: Iterable[Any],
        *,
        final_event: ScreenEvent | None = None,
        area_changed: bool = False,
    ) -> None:
        """Consume the steps of one action batch (duck-typed ActionStep objects)."""
        for step in steps:
            button = str(getattr(step, "button", "") or "").lower()
            if button not in DIRECTIONS:
                continue
            ratio = float(getattr(step, "changed_ratio", 0.0) or 0.0)
            if ratio >= self.move_min_ratio:
                self._mark_open(button)
                self._move(button)
            elif ratio <= self.block_max_ratio:
                self._mark_blocked(button)
        transition = area_changed or final_event in (
            ScreenEvent.MAP_CHANGE,
            ScreenEvent.TRANSITION,
        )
        if transition:
            self.reset_area(reason=f"event:{getattr(final_event, 'value', final_event)}")

    def observe_change_ratio(self, ratio: float, event: ScreenEvent | None = None) -> None:
        """Opportunistic area-change detection when no steps were involved."""
        if ratio >= self.area_change_ratio or event in (ScreenEvent.MAP_CHANGE, ScreenEvent.TRANSITION):
            self.reset_area(reason=f"event:{getattr(event, 'value', event)}")

    def _mark_open(self, direction: str) -> None:
        self.area.edges.setdefault(self.pos, {})[direction] = "open"

    def _mark_blocked(self, direction: str) -> None:
        dx, dy = _DELTAS[direction]
        target = (self.pos[0] + dx, self.pos[1] + dy)
        self.area.blocked.add(target)
        self.area.edges.setdefault(self.pos, {})[direction] = "blocked"
        self.blocked_attempts += 1
        if target in self.area.visited and len(self.area.visited) > 1:
            # We walked over this cell before, so it cannot be a permanent wall;
            # forget the stale blocked marker to keep the map honest.
            self.area.blocked.discard(target)

    def _move(self, direction: str) -> None:
        dx, dy = _DELTAS[direction]
        self.pos = (self.pos[0] + dx, self.pos[1] + dy)
        self.area.visited.add(self.pos)
        self.area.blocked.discard(self.pos)
        self.moves += 1

    # ------------------------------------------------------------------ #
    # Rendering / prompt context
    # ------------------------------------------------------------------ #
    def blocked_directions(self) -> list[str]:
        """Directions that are known-blocked from the current tile."""
        edges = self.area.edges.get(self.pos, {})
        return [direction for direction in DIRECTIONS if edges.get(direction) == "blocked"]

    def render_map(self, *, radius: int = 4) -> str:
        """ASCII map centred on the agent (north = up)."""
        radius = max(2, min(8, int(radius)))
        area = self.area
        if self.moves == 0 and self.blocked_attempts == 0:
            return ""
        rows: list[str] = []
        for dy in range(radius, -radius - 1, -1):
            row_chars: list[str] = []
            for dx in range(-radius, radius + 1):
                cell = (self.pos[0] + dx, self.pos[1] + dy)
                if dx == 0 and dy == 0:
                    row_chars.append("@")
                elif cell in area.blocked:
                    row_chars.append("#")
                elif cell in area.visited:
                    row_chars.append("·")
                else:
                    row_chars.append("?")
            rows.append("".join(row_chars))
        blocked = ", ".join({"up": "上", "down": "下", "left": "左", "right": "右"}[d] for d in self.blocked_directions())
        lines = [
            f"区域 area {self.area_id}（本区域走过 {len(area.visited)} 格，共探索 {self.explored_cells} 格）",
            "图例: @=你  ·=走过的路  #=你试过走不通（别再撞）  ?=未探索；上方=北",
        ]
        lines.extend(rows)
        lines.append(f"当前位置已知走不通的方向: {blocked if blocked else '无（都可以试）'}")
        return "\n".join(lines)

    def render_context(self) -> str:
        """Compact prompt block (empty when nothing has been explored yet)."""
        if self.moves == 0 and self.blocked_attempts == 0 and self.area_switches == 0:
            return ""
        return self.render_map()

    def summary(self) -> dict[str, Any]:
        return {
            "area_id": self.area_id,
            "pos": list(self.pos),
            "moves": self.moves,
            "blocked_attempts": self.blocked_attempts,
            "area_switches": self.area_switches,
            "explored_cells": self.explored_cells,
            "current_area_cells": len(self.area.visited),
            "blocked_now": self.blocked_directions(),
        }

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def snapshot(self) -> dict[str, Any]:
        areas: dict[str, Any] = {}
        for area_id, area in self.areas.items():
            areas[str(area_id)] = {
                "visited": [list(cell) for cell in sorted(area.visited)],
                "blocked": [list(cell) for cell in sorted(area.blocked)],
                "edges": {
                    f"{cell[0]},{cell[1]}": dict(directions)
                    for cell, directions in area.edges.items()
                },
            }
        return {
            "version": 1,
            "pos": list(self.pos),
            "area_id": self.area_id,
            "moves": self.moves,
            "blocked_attempts": self.blocked_attempts,
            "area_switches": self.area_switches,
            "last_transition": self.last_transition,
            "areas": areas,
        }

    def restore(self, snapshot: Mapping[str, Any] | None) -> None:
        snapshot = snapshot or {}
        if not snapshot or int(snapshot.get("version") or 0) != 1:
            return
        try:
            self.pos = tuple(int(value) for value in snapshot.get("pos") or (0, 0))  # type: ignore[assignment]
            self.area_id = int(snapshot.get("area_id") or 1)
            self.moves = int(snapshot.get("moves") or 0)
            self.blocked_attempts = int(snapshot.get("blocked_attempts") or 0)
            self.area_switches = int(snapshot.get("area_switches") or 0)
            self.last_transition = str(snapshot.get("last_transition") or "")
            self.areas = {}
            for area_key, area_data in (snapshot.get("areas") or {}).items():
                area_id = int(area_key)
                area = AreaMap(area_id)
                area.visited = {tuple(int(v) for v in cell) for cell in area_data.get("visited", [])}
                area.blocked = {tuple(int(v) for v in cell) for cell in area_data.get("blocked", [])}
                edges: dict[tuple[int, int], dict[str, str]] = {}
                for cell_key, directions in (area_data.get("edges") or {}).items():
                    x_text, y_text = str(cell_key).split(",", 1)
                    edges[(int(x_text), int(y_text))] = {
                        str(direction): str(state) for direction, state in dict(directions).items()
                    }
                area.edges = edges
                self.areas[area_id] = area
            self.areas.setdefault(self.area_id, AreaMap(self.area_id))
        except Exception:
            # A malformed snapshot must never break a run; fall back to a fresh map.
            self.pos = (0, 0)
            self.area_id = 1
            self.areas = {1: AreaMap(1)}
