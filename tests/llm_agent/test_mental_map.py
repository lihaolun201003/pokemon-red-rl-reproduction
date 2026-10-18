"""Tests for the self-recorded dead-reckoning mental map (observation-only)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from llm_agent.contracts import ScreenEvent
from llm_agent.mental_map import MentalMap


def step(button: str, ratio: float) -> SimpleNamespace:
    return SimpleNamespace(button=button, changed_ratio=ratio)


def test_successful_moves_update_position_and_visited() -> None:
    mental_map = MentalMap()
    mental_map.update([step("up", 0.3), step("up", 0.3), step("right", 0.25)])
    assert mental_map.pos == (1, 2)
    assert mental_map.moves == 3
    assert (0, 0) in mental_map.area.visited
    assert (0, 1) in mental_map.area.visited
    assert (0, 2) in mental_map.area.visited
    assert (1, 2) in mental_map.area.visited
    assert mental_map.blocked_directions() == []
    assert mental_map.area.edges[(0, 0)]["up"] == "open"


def test_blocked_direction_is_marked_and_rendered() -> None:
    mental_map = MentalMap()
    mental_map.update([step("up", 0.0)])  # tried to walk into something
    assert mental_map.pos == (0, 0)
    assert mental_map.blocked_attempts == 1
    assert (0, 1) in mental_map.area.blocked
    assert mental_map.blocked_directions() == ["up"]
    rendered = mental_map.render_map(radius=3)
    assert "@" in rendered
    assert "#" in rendered
    assert "走不通" in rendered


def test_ambiguous_small_change_is_not_a_move_and_not_a_block() -> None:
    mental_map = MentalMap()
    mental_map.update([step("left", 0.02)])  # character turned but did not move
    assert mental_map.pos == (0, 0)
    assert mental_map.moves == 0
    assert mental_map.blocked_attempts == 0
    assert mental_map.blocked_directions() == []


def test_non_direction_buttons_are_ignored() -> None:
    mental_map = MentalMap()
    mental_map.update([step("a", 0.4), step("start", 0.4), step("b", 0.0)])
    assert mental_map.pos == (0, 0)
    assert mental_map.moves == 0
    assert mental_map.blocked_attempts == 0


def test_stale_blocked_marker_forgotten_when_cell_walked_later() -> None:
    mental_map = MentalMap()
    mental_map.update([step("right", 0.0)])  # blocked into (1,0)
    assert (1, 0) in mental_map.area.blocked
    mental_map.update([step("up", 0.3), step("right", 0.3), step("down", 0.3)])
    assert mental_map.pos == (1, 0)
    assert (1, 0) not in mental_map.area.blocked  # walked onto the cell from another side


def test_map_transition_resets_area() -> None:
    mental_map = MentalMap()
    mental_map.update([step("up", 0.3), step("right", 0.3)])
    old_area = mental_map.area_id
    mental_map.update([step("down", 0.3)], final_event=ScreenEvent.MAP_CHANGE)
    assert mental_map.area_id == old_area + 1
    assert mental_map.pos == (0, 0)
    assert mental_map.area_switches == 1
    assert mental_map.explored_cells >= 3  # old area is retained in history


def test_transition_event_flag_resets_even_without_event_enum() -> None:
    mental_map = MentalMap()
    mental_map.update([step("up", 0.9)], area_changed=True)
    assert mental_map.area_switches == 1


def test_snapshot_restore_roundtrip() -> None:
    mental_map = MentalMap()
    mental_map.update([step("up", 0.3), step("up", 0.3), step("right", 0.0), step("down", 0.3)])
    snapshot = mental_map.snapshot()
    restored = MentalMap()
    restored.restore(snapshot)
    assert restored.pos == mental_map.pos
    assert restored.moves == mental_map.moves
    assert restored.blocked_attempts == mental_map.blocked_attempts
    assert restored.area.visited == mental_map.area.visited
    assert restored.area.blocked == mental_map.area.blocked
    assert restored.render_map() == mental_map.render_map()


def test_malformed_snapshot_falls_back_quietly() -> None:
    mental_map = MentalMap()
    mental_map.restore({"version": 1, "pos": "bad", "areas": {"x": {}}})
    assert mental_map.pos == (0, 0)
    assert mental_map.area_id == 1


def test_empty_map_renders_nothing() -> None:
    assert MentalMap().render_context() == ""
    assert MentalMap().render_map() == ""


def test_summary_and_legend() -> None:
    mental_map = MentalMap()
    mental_map.update([step("down", 0.4)])
    summary = mental_map.summary()
    assert summary["explored_cells"] == 2
    assert summary["moves"] == 1
    rendered = mental_map.render_map(radius=2)
    assert "区域 area" in rendered
    assert "北" in rendered


@pytest.mark.parametrize("direction", ["up", "down", "left", "right"])
def test_each_direction_moves_one_cell(direction: str) -> None:
    mental_map = MentalMap()
    mental_map.update([step(direction, 0.3)])
    expected = {"up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0)}[direction]
    assert mental_map.pos == expected
