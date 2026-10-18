"""Unit tests for the high-level persistent navigation facade (nav-control).

Uses the small fakes from ``test_controller.py`` (fake emulator/localizer/grid/
world/planner/explorer).  The last test is an optional real-integration smoke
test for persistence + relocalization that runs once the nav-map modules exist.
"""

from __future__ import annotations

import numpy as np
import pytest

from llm_agent.contracts import ActionOutcome, ActionStep, Frame, ScreenEvent
from llm_agent.navigation.navigation_agent import (
    NavigationAgent,
    NavigationAgentConfig,
)
from llm_agent.navigation.types import (
    CellState,
    ExploreStatus,
    FrameContext,
    MotionResult,
    PathPlan,
)

from tests.navigation.test_controller import (
    FakeEmulator,
    FakeExplorer,
    FakeLocalizer,
    FakePathfinder,
    FakeWorld,
    blocked,
    interrupted,
    moved,
    transition,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def make_frame(index: int) -> Frame:
    return Frame(
        frame_index=int(index),
        rgba=np.zeros((144, 160, 4), dtype=np.uint8),
        sha256=f"sha{index}",
        captured_at=0.0,
        is_black=False,
        luma_mean=0.0,
    )


def make_outcome(
    steps: list[tuple[str, int]],
    *,
    events: list[ScreenEvent] | None = None,
    stopped_reason: str = "completed",
    start_frame: int = 0,
) -> ActionOutcome:
    action_steps: list[ActionStep] = []
    for button, frame_index in steps:
        action_steps.append(
            ActionStep(
                button=button,
                frame=make_frame(frame_index),
                screen_changed=True,
                changed_ratio=0.1,
                event=(events[0] if events else ScreenEvent.NONE),
            )
        )
    return ActionOutcome(
        buttons=[button for button, _ in steps],
        steps=action_steps,
        start_frame=int(start_frame),
        end_frame=int(steps[-1][1]) if steps else int(start_frame),
        stopped_reason=stopped_reason,
        events=list(events or []),
        summary="test outcome",
    )


class AgentHarness:
    """Bundle of NavigationAgent + fakes for one test."""

    def __init__(
        self,
        *,
        script: list | None = None,
        pose: tuple[int, int] = (5, 5),
        world: FakeWorld | None = None,
        world_root=None,
        config: NavigationAgentConfig | None = None,
        targets: list | None = None,
        finish_sequence: list | None = None,
        make_area: bool = True,
    ) -> None:
        self.emulator = FakeEmulator()
        self.world = world if world is not None else FakeWorld()
        if make_area and not self.world.areas:
            area = self.world.create_area()
            self.world.set_current(area.area_id, pose)
        self.localizer = FakeLocalizer(script=script, pose=pose)
        self.localizer.area_id = self.world.current_area
        self.pathfinder = FakePathfinder()
        self.explorer = FakeExplorer(targets=targets, finish_sequence=finish_sequence)
        self.agent = NavigationAgent(
            emulator=self.emulator,
            world_root=world_root,
            config=config,
            world=self.world,
            localizer=self.localizer,
            pathfinder=self.pathfinder,
            explorer=self.explorer,
        )

    @property
    def grid(self):
        return self.world.area().grid


# --------------------------------------------------------------------------- #
# Attach
# --------------------------------------------------------------------------- #


def test_attach_creates_first_area_at_origin():
    harness = AgentHarness(make_area=False)
    harness.agent.attach()
    status = harness.agent.status()
    assert status["area_id"] == "area_0"
    assert status["pose"]["x"] == 0 and status["pose"]["y"] == 0
    assert status["areas_total"] == 1
    assert harness.agent.last_attach is not None
    assert harness.agent.last_attach["new_area"] is True
    # one observation recorded into the fresh grid
    assert len(harness.world.area("area_0").grid.observe_calls) == 1
    assert (0, 0) in harness.world.area("area_0").entry_cells


def test_attach_uses_strong_relocalization_match():
    world = FakeWorld()
    area = world.create_area(area_id="area_0")
    world.set_current("area_0", (5, 5))
    from tests.navigation.test_controller import _match

    world.match_results = [_match("area_0", (2, 3))]
    harness = AgentHarness(world=world, make_area=False)
    harness.agent.attach()
    status = harness.agent.status()
    assert status["area_id"] == "area_0"
    assert status["pose"]["x"] == 2 and status["pose"]["y"] == 3
    assert harness.agent.last_attach is not None
    assert harness.agent.last_attach["matched"] is True
    assert status["areas_total"] == 1  # no duplicate area created


def test_attach_weak_evidence_never_merges_areas():
    world = FakeWorld()
    world.create_area(area_id="area_0")
    world.set_current("area_0", (5, 5))
    from tests.navigation.test_controller import _match

    world.match_results = [
        _match("area_0", (0, 0), score=0.9, margin=0.01, second_score=0.89)
    ]
    harness = AgentHarness(world=world, make_area=False)
    harness.agent.attach()
    status = harness.agent.status()
    assert status["areas_total"] == 2
    assert status["area_id"] != "area_0"
    assert harness.agent.last_attach["matched"] is False


# --------------------------------------------------------------------------- #
# Status / map / world reporting
# --------------------------------------------------------------------------- #


def test_status_has_the_contract_field_names():
    harness = AgentHarness()
    status = harness.agent.status()
    for key in (
        "area_id",
        "area_kind",
        "pose",
        "cells_total",
        "walkable",
        "blocked",
        "unknown_frontier",
        "coverage_ratio",
        "areas_total",
        "last_event",
        "paused",
        "world",
    ):
        assert key in status, key
    for pose_key in ("x", "y", "facing", "confidence", "valid"):
        assert pose_key in status["pose"]
    assert set(status["world"]) >= {"areas", "connections"}
    assert status["world"]["areas"] == [] or isinstance(status["world"]["areas"], list)


def test_status_before_attach_is_safe():
    harness = AgentHarness(make_area=False)
    status = harness.agent.status()
    assert status["area_id"] == ""
    assert status["areas_total"] == 0
    assert status["paused"] is False


def test_status_counts_follow_grid_state():
    harness = AgentHarness(script=[moved("right"), blocked("up")], pose=(5, 5))
    harness.agent.controller.observe()
    harness.agent.controller.move_once("right")
    harness.agent.controller.move_once("up")
    status = harness.agent.status()
    assert status["walkable"] >= 1
    assert status["cells_total"] >= 2
    assert status["blocked"] >= 1
    assert status["world"]["areas"][0][0] == "area_0"
    assert status["world"]["areas"][0][3] >= 1  # walkable count of the area


def test_local_map_shape_and_window():
    harness = AgentHarness(script=[moved("right")], pose=(5, 5))
    harness.agent.controller.observe()
    harness.agent.controller.move_once("right")
    local = harness.agent.local_map(radius=4)
    assert set(local) >= {"area_id", "pose", "ascii", "cells", "coverage"}
    assert local["area_id"] == "area_0"
    assert isinstance(local["ascii"], str)
    for cell in local["cells"]:
        assert {"x", "y", "state", "terrain", "visits", "confidence"} <= set(cell)
        assert abs(cell["x"] - 6) <= 4 and abs(cell["y"] - 5) <= 4
    assert (6, 5) in {(cell["x"], cell["y"]) for cell in local["cells"]}


def test_world_summary_shape():
    harness = AgentHarness(pose=(5, 5))
    summary = harness.agent.world_summary()
    assert set(summary) >= {"current_area", "current_pose", "areas_total"}
    assert summary["current_area"] == "area_0"
    assert summary["current_pose"] == [5, 5]


# --------------------------------------------------------------------------- #
# Pause / resume
# --------------------------------------------------------------------------- #


def test_pause_blocks_exploration_and_resume_reenables():
    harness = AgentHarness(finish_sequence=[(True, "done", {})])
    harness.agent.pause()
    assert harness.agent.paused is True
    paused = harness.agent.explore_current_area(max_steps=10)
    assert paused.status is ExploreStatus.PAUSED
    paused_nav = harness.agent.navigate_to({"cell": [6, 5]})
    assert paused_nav.status is ExploreStatus.PAUSED
    harness.agent.resume()
    assert harness.agent.paused is False
    report = harness.agent.explore_current_area(max_steps=10)
    assert report.status is ExploreStatus.COMPLETED


# --------------------------------------------------------------------------- #
# Exploration / navigation delegation
# --------------------------------------------------------------------------- #


def test_explore_current_area_delegates_and_autosaves():
    harness = AgentHarness(finish_sequence=[(True, "done", {})])
    report = harness.agent.explore_current_area(max_steps=10)
    assert report.status is ExploreStatus.COMPLETED
    assert harness.agent.last_report is report
    assert harness.world.save_count >= 1  # autosave after a mutating call
    assert harness.agent.status()["paused"] is False


def test_explore_and_report_returns_serialisable_dict():
    harness = AgentHarness(finish_sequence=[(True, "done", {})])
    data = harness.agent.explore_and_report(max_steps=5)
    assert set(data) >= {"report", "status", "world", "map"}
    assert data["report"]["status"] == "completed"
    assert data["status"]["area_id"] == "area_0"
    assert isinstance(data["world"], dict) and isinstance(data["map"], dict)


def test_navigate_to_cell_via_agent():
    harness = AgentHarness(script=[moved("right")], pose=(5, 5))
    report = harness.agent.navigate_to({"cell": [6, 5]}, max_steps=10)
    assert report.status is ExploreStatus.COMPLETED
    assert report.reached is True
    assert report.pose.cell == (6, 5)
    assert harness.world.save_count >= 1


def test_navigate_to_blocked_is_reported_not_raised():
    harness = AgentHarness(pose=(5, 5))
    harness.pathfinder.plans.append(PathPlan(reachable=False, reason="walled off"))
    report = harness.agent.navigate_to({"cell": [9, 9]}, max_steps=10)
    assert report.status is ExploreStatus.BLOCKED
    assert report.reached is False


# --------------------------------------------------------------------------- #
# External raw-tool outcomes: exactly-once dedupe
# --------------------------------------------------------------------------- #


def test_external_outcome_updates_map_exactly_once():
    harness = AgentHarness(script=[moved("right")], pose=(5, 5))
    harness.agent.controller.observe()  # baseline view
    outcome = make_outcome([("right", 24)])
    harness.agent.observe_external_outcome(outcome)
    assert [call for call in harness.grid.calls if call[0] == "move"] == [
        ("move", (5, 5), "right", 1)
    ]
    assert harness.localizer.pose.cell == (6, 5)
    # replaying the exact same outcome must not double count
    harness.agent.observe_external_outcome(outcome)
    assert [call for call in harness.grid.calls if call[0] == "move"] == [
        ("move", (5, 5), "right", 1)
    ]
    # a *new* outcome object describing the same completed batch is deduped too
    replay = make_outcome([("right", 24)])
    harness.agent.observe_external_outcome(replay)
    assert [call for call in harness.grid.calls if call[0] == "move"] == [
        ("move", (5, 5), "right", 1)
    ]
    assert harness.localizer.pose.cell == (6, 5)


def test_external_outcome_non_direction_press_does_not_move():
    harness = AgentHarness(pose=(5, 5))
    harness.agent.controller.observe()
    outcome = make_outcome([("a", 24)])
    harness.agent.observe_external_outcome(outcome)
    assert harness.grid.calls == []
    assert harness.localizer.pose.cell == (5, 5)


def test_external_outcome_transition_relocalizes():
    harness = AgentHarness(script=[transition("right")], pose=(5, 5))
    harness.agent.controller.observe()
    harness.world.create_area(area_id="area_1")
    from tests.navigation.test_controller import _match

    harness.world.match_results = [_match("area_1", (0, 0))]
    harness.agent.observe_external_outcome(make_outcome([("right", 24)]))
    assert harness.world.current_area == "area_1"
    assert harness.agent.last_event is not None
    assert harness.agent.last_event.kind.value == "transition_event"
    assert harness.grid.calls == []  # transitions never confirm a move


def test_external_outcome_dialog_event_pauses_navigation():
    harness = AgentHarness(
        script=[interrupted("right", FrameContext.DIALOG)],
        pose=(5, 5),
    )
    harness.agent.controller.observe()
    harness.agent.observe_external_outcome(make_outcome([("right", 24)]))
    assert harness.agent.last_event is not None
    assert harness.agent.last_event.kind.value == "dialog_event"
    assert harness.agent.paused is True
    assert harness.grid.calls == []


def test_external_outcome_a_press_dialog_event_pauses_navigation():
    harness = AgentHarness(pose=(5, 5))
    harness.agent.controller.observe()
    harness.agent.observe_external_outcome(
        make_outcome([("a", 24)], events=[ScreenEvent.DIALOG], stopped_reason="event")
    )
    assert harness.agent.last_event is not None
    assert harness.agent.last_event.kind.value == "dialog_event"
    assert harness.agent.paused is True


def test_external_outcome_event_does_not_pause_when_configured_off():
    config = NavigationAgentConfig()
    config.controller.pause_on_event = False
    harness = AgentHarness(pose=(5, 5), config=config)
    harness.agent.controller.observe()
    harness.agent.observe_external_outcome(
        make_outcome([("a", 24)], events=[ScreenEvent.DIALOG], stopped_reason="event")
    )
    assert harness.agent.last_event is not None
    assert harness.agent.paused is False


# --------------------------------------------------------------------------- #
# Persistence hooks
# --------------------------------------------------------------------------- #


def test_save_calls_world_save(tmp_path):
    harness = AgentHarness(pose=(5, 5), world_root=tmp_path / "run")
    before = harness.world.save_count
    harness.agent.save()
    assert harness.world.save_count >= before + 1
    assert harness.agent.last_save_error is None


def test_real_persistence_and_relocalization_roundtrip(tmp_path, open_room):
    """Real WorldModel + real Localizer + real GridMap: save -> reload -> attach."""
    pytest.importorskip("llm_agent.navigation.world_model")
    pytest.importorskip("llm_agent.navigation.grid_map")
    from llm_agent.navigation.world_model import WorldModel

    from tests.navigation.conftest import SyntheticEmulator

    root = tmp_path / "run"
    emulator = SyntheticEmulator(open_room, pose=(5, 4), facing="down")
    agent = NavigationAgent(emulator=emulator, world_root=root)
    agent.attach()
    first = agent.status()
    assert first["areas_total"] == 1
    assert first["area_id"]
    assert first["cells_total"] > 0
    agent.save()

    reloaded = WorldModel(root=root)
    assert reloaded.load() is True
    assert list(reloaded.areas) == [first["area_id"]]
    assert reloaded.current_area in reloaded.areas

    # A second agent loading the same world root must relocalize, not duplicate.
    agent2 = NavigationAgent(emulator=emulator, world_root=root)
    agent2.attach()
    second = agent2.status()
    assert second["areas_total"] == 1
    assert second["area_id"] == first["area_id"]
    assert agent2.last_attach is not None
    assert agent2.last_attach["matched"] is True
    # the area-local origin is where the area was first entered: (0, 0)
    assert second["pose"]["x"] == first["pose"]["x"]
    assert second["pose"]["y"] == first["pose"]["y"]
    assert second["pose"]["valid"] is True


def test_explore_with_real_modules_is_safe(tmp_path, open_room):
    """One small bounded real explore run (Localizer+GridMap+WorldModel+Planner)."""
    pytest.importorskip("llm_agent.navigation.world_model")
    pytest.importorskip("llm_agent.navigation.grid_map")
    pytest.importorskip("llm_agent.navigation.explorer")

    from tests.navigation.conftest import SyntheticEmulator

    emulator = SyntheticEmulator(open_room, pose=(5, 4), facing="down")
    agent = NavigationAgent(emulator=emulator, world_root=tmp_path / "run")
    agent.attach()
    report = agent.explore_current_area(max_steps=4, max_seconds=10.0)
    assert report.status in {
        ExploreStatus.COMPLETED,
        ExploreStatus.STEP_LIMIT,
        ExploreStatus.TIME_LIMIT,
        ExploreStatus.NO_PROGRESS,
        ExploreStatus.BLOCKED,
        ExploreStatus.TRANSITION,
        ExploreStatus.INTERRUPTED_EVENT,
    }
    assert report.steps <= 8  # max_steps plus one frontier probe per cycle
    status = agent.status()
    assert status["cells_total"] > 0
    assert status["areas_total"] >= 1
