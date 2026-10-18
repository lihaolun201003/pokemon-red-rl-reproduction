"""Unit tests for the multi-area world model + visual relocalization (no ROM).

The relocalization tests build synthetic rooms and store their tile hashes through
``GridMap.observe_view`` exactly the way the real controller does, then verify:

* the correct (area, pose) wins with a large margin;
* the player's own sprite tile is ignored;
* cells without stored hashes are not compared;
* identical areas are reported as ambiguous and rejected by ``locate`` unless an
  expected area resolves the tie;
* persistence round-trips through ``<root>/navigation/world.json`` +
  ``areas/<id>.json`` and tolerates damaged files.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_agent.navigation import vision
from llm_agent.navigation.grid_map import GridMap
from llm_agent.navigation.types import (
    VIEW_H,
    VIEW_PLAYER,
    VIEW_W,
    CellState,
    FrameView,
    PlayerPose,
    TileObs,
)
from llm_agent.navigation.world_model import (
    DEFAULT_MATCH_MIN_MARGIN,
    MIN_COMPARABLE_TILES,
    WORLD_MODEL_FORMAT,
    WORLD_MODEL_FORMAT_VERSION,
    AreaRecord,
    WorldConnection,
    WorldModel,
)

from tests.navigation.conftest import SyntheticRoom, luma_from_rgba


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

#: Views centred here are fully inside a 14x11 synthetic room.
_BIG_POSES = [(7, 5), (8, 5), (7, 4), (8, 6), (6, 5), (9, 5), (7, 6), (8, 4)]


def _pose(x: int, y: int, *, valid: bool = True) -> PlayerPose:
    return PlayerPose(x=int(x), y=int(y), facing="down", confidence=1.0, valid=valid)


def _view(room: SyntheticRoom, pose: tuple[int, int], *, facing: str = "down") -> FrameView:
    luma = luma_from_rgba(room.render(pose, facing))
    tiles = vision.extract_tiles(luma, (0, 0))
    return FrameView(frame_index=0, signature="synthetic", grid_offset=(0, 0), tiles=tiles)


def _all_tile_view(tile_hash: str = "aaa") -> FrameView:
    tiles = [
        TileObs(tx=tx, ty=ty, tile_hash=tile_hash)
        for ty in range(VIEW_H)
        for tx in range(VIEW_W)
    ]
    return FrameView(frame_index=0, signature="synthetic", grid_offset=(0, 0), tiles=tiles)


def _observe_area(
    world: WorldModel,
    room: SyntheticRoom,
    poses: list[tuple[int, int]],
    *,
    area_id: str = "room_a",
    kind: str = "interior",
    frame_index: int = 0,
) -> AreaRecord:
    record = world.create_area(area_id=area_id, kind=kind, frame_index=frame_index)
    frame = frame_index
    for pose in poses:
        frame += 1
        record.grid.observe_view(
            _view(room, pose), _pose(*pose), frame_index=frame
        )
    return record


# --------------------------------------------------------------------------- #
# AreaRecord / WorldConnection records
# --------------------------------------------------------------------------- #


def test_area_record_roundtrip() -> None:
    grid = GridMap("area_001", kind="interior")
    grid.mark_state((0, 0), CellState.WALKABLE)
    record = AreaRecord(
        area_id="area_001",
        kind="interior",
        grid=grid,
        visits=2,
        first_frame=5,
        last_frame=9,
        entry_cells=[(0, 0), (1, 2)],
        explored=True,
        notes=["first room"],
    )
    payload = record.to_dict()
    assert set(payload) == {
        "area_id",
        "kind",
        "visits",
        "first_frame",
        "last_frame",
        "entry_cells",
        "explored",
        "notes",
    }
    assert payload["entry_cells"] == [[0, 0], [1, 2]]
    clone = AreaRecord.from_dict(payload)
    assert clone.area_id == "area_001"
    assert clone.kind == "interior"
    assert clone.visits == 2
    assert clone.first_frame == 5
    assert clone.last_frame == 9
    assert clone.entry_cells == [(0, 0), (1, 2)]
    assert clone.explored is True
    assert clone.notes == ["first room"]
    assert clone.grid.size() == 0  # grids are persisted separately

    attached = AreaRecord.from_dict(payload, grid=grid)
    assert attached.grid is grid

    tolerant = AreaRecord.from_dict({"area_id": "a"})
    assert tolerant.kind == "unknown"
    assert tolerant.visits == 0
    assert tolerant.grid.area_id == "a"

    with pytest.raises(ValueError):
        AreaRecord.from_dict({})
    with pytest.raises(ValueError):
        AreaRecord.from_dict([])  # type: ignore[arg-type]


def test_world_connection_roundtrip_merge_and_tolerance() -> None:
    connection = WorldConnection(
        from_area="a",
        from_cell=(3, 4),
        direction="down",
        to_area="b",
        to_pose=(2, 2),
        kind="warp",
        confidence=0.75,
        observations=1,
    )
    assert WorldConnection.from_dict(connection.to_dict()) == connection
    assert connection.key == ("a", (3, 4), "down", "b")

    tolerant = WorldConnection.from_dict({"from_area": "a", "to_area": "b"})
    assert tolerant.from_cell == (0, 0)
    assert tolerant.kind == "unknown"
    assert tolerant.confidence == pytest.approx(0.5)
    assert tolerant.observations == 1
    with pytest.raises(ValueError):
        WorldConnection.from_dict({"from_area": "", "to_area": ""})

    world = WorldModel()
    world.add_connection(connection)
    world.add_connection(
        WorldConnection(
            from_area="a",
            from_cell=(3, 4),
            direction="down",
            to_area="b",
            to_pose=(2, 2),
            kind="warp",
            confidence=0.9,
        )
    )
    assert len(world.connections) == 1
    merged = world.connections[0]
    assert merged.observations == 2
    assert merged.confidence == pytest.approx(0.9)
    assert merged.to_pose == (2, 2)

    world.add_connection(WorldConnection("b", (2, 2), "up", "a"))
    assert len(world.connections) == 2
    with pytest.raises(TypeError):
        world.add_connection({"from_area": "a"})  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Area registry / current state
# --------------------------------------------------------------------------- #


def test_create_area_ids_kind_and_idempotence() -> None:
    world = WorldModel()
    first = world.create_area(kind="interior", frame_index=3)
    second = world.create_area(kind="exterior")
    third = world.create_area()
    assert [first.area_id, second.area_id, third.area_id] == [
        "area_001",
        "area_002",
        "area_003",
    ]
    assert first.kind == "interior"
    assert first.grid.kind == "interior"
    assert first.first_frame == 3
    assert first.last_frame == 3
    assert first.visits == 0
    assert first.entry_cells == []  # entry pose is recorded by set_current()
    assert set(world.areas) == {"area_001", "area_002", "area_003"}

    same = world.create_area(
        area_id="area_001", kind="interior", frame_index=8, entry_cell=(2, 2)
    )
    assert same is first
    assert first.last_frame == 8
    assert (2, 2) in first.entry_cells

    custom = world.area("custom")
    assert custom.area_id == "custom"
    assert world.areas["custom"] is custom


def test_area_and_set_current_and_visits() -> None:
    world = WorldModel()
    # area() with no current area creates one and makes it current.
    created = world.area()
    assert created.area_id == "area_001"
    assert world.current_area == "area_001"
    assert world.current_pose == (0, 0)
    assert world.area() is created

    world.set_current("room", (2, 3))
    assert world.current_area == "room"
    assert world.current_pose == (2, 3)
    assert world.current_record is not None
    assert world.current_grid is world.areas["room"].grid
    assert world.areas["room"].entry_cells == [(2, 3)]

    world.set_current("room", (2, 3))  # entry cells are deduplicated
    assert world.areas["room"].entry_cells == [(2, 3)]
    world.set_current("room", (5, 5))
    assert world.areas["room"].entry_cells == [(2, 3), (5, 5)]

    world.record_visit("room", frame_index=10)
    record = world.areas["room"]
    assert (record.visits, record.first_frame, record.last_frame) == (1, 10, 10)
    world.record_visit("room", frame_index=12)
    assert (record.visits, record.first_frame, record.last_frame) == (2, 10, 12)
    world.record_visit("room", frame_index=9)  # out-of-order frame never rewinds
    assert record.last_frame == 12


def test_set_area_kind_and_mark_explored() -> None:
    world = WorldModel()
    record = world.create_area(area_id="a")
    assert record.kind == "unknown"
    world.set_area_kind("a", "exterior")
    assert record.kind == "exterior"
    assert record.grid.kind == "exterior"
    world.set_area_kind("a", "garbage")  # tolerated -> unknown
    assert record.kind == "unknown"

    assert world.unexplored_areas() == ["a"]
    world.mark_area_explored("a")
    assert world.unexplored_areas() == []
    assert world.summary()["explored_areas"] == 1


# --------------------------------------------------------------------------- #
# Relocalization
# --------------------------------------------------------------------------- #


def test_match_view_relocalizes_known_pose() -> None:
    room = SyntheticRoom(width=14, height=11, seed=7)
    world = WorldModel()
    _observe_area(world, room, _BIG_POSES)
    query = _view(room, (7, 5))

    matches = world.match_view(query)
    assert matches
    best = matches[0]
    assert best.area_id == "room_a"
    assert best.pose == (7, 5)
    assert best.score == pytest.approx(1.0)
    assert best.compared == 89  # 90 visible tiles minus the ignored player tile
    assert best.matched == best.compared
    assert best.margin > DEFAULT_MATCH_MIN_MARGIN
    assert best.evidence["candidate_count"] >= 1
    assert best.margin == pytest.approx(best.score - best.second_score, abs=1e-4)

    # Deterministic output.
    assert [match.to_dict() for match in world.match_view(query)] == [
        match.to_dict() for match in matches
    ]

    located = world.locate(query)
    assert located is not None
    assert located.area_id == "room_a"
    assert located.pose == (7, 5)
    assert located.evidence["accepted_by"] == "margin"
    # locate() is a pure query: it must not move the world state.
    assert world.current_area == ""
    assert world.current_pose == (0, 0)


def test_match_view_ignores_player_tile_and_cells_without_hashes() -> None:
    room = SyntheticRoom(width=14, height=11, seed=11)
    world = WorldModel()
    record = _observe_area(world, room, _BIG_POSES)
    query = _view(room, (7, 5))
    baseline = world.match_view(query)[0]
    assert (baseline.pose, baseline.score) == ((7, 5), pytest.approx(1.0))

    # 1) Corrupting the player's own tile hash must not change the result.
    mutated: list[TileObs] = []
    for tile in query.tiles:
        if (tile.tx, tile.ty) == VIEW_PLAYER:
            mutated.append(TileObs(tx=tile.tx, ty=tile.ty, tile_hash="f" * 16))
        else:
            mutated.append(tile)
    sprite_view = FrameView(
        frame_index=0, signature="synthetic", grid_offset=query.grid_offset, tiles=mutated
    )
    sprite_matches = world.match_view(sprite_view)
    assert sprite_matches[0].pose == baseline.pose
    assert sprite_matches[0].score == pytest.approx(baseline.score)

    # 2) A cell without stored hashes is not "comparable".
    target_cell = (9, 6)  # (7,5) + rel (2,1)
    cell = record.grid.get_cell(*target_cell)
    assert cell.tile_hashes
    saved_hashes = list(cell.tile_hashes)
    cell.tile_hashes.clear()
    reduced = world.match_view(query)[0]
    assert reduced.pose == baseline.pose
    assert reduced.compared == baseline.compared - 1
    assert reduced.score == pytest.approx(1.0)
    cell.tile_hashes = saved_hashes


def test_match_view_ambiguous_identical_areas_kept_and_rejected() -> None:
    room = SyntheticRoom(width=14, height=11, seed=21)
    world = WorldModel()
    _observe_area(world, room, _BIG_POSES, area_id="room_a")
    _observe_area(world, room, _BIG_POSES, area_id="room_b")  # identical rendering
    query = _view(room, (7, 5))

    matches = world.match_view(query)
    assert len(matches) >= 2
    assert matches[0].score == pytest.approx(1.0)
    assert matches[1].score == pytest.approx(1.0)
    assert matches[0].margin == pytest.approx(0.0)
    assert {match.area_id for match in matches[:2]} == {"room_a", "room_b"}

    # Ambiguous: the controller must treat the location as unknown...
    assert world.locate(query) is None
    # ... unless an expected area (from a just-recorded connection) resolves it.
    resolved = world.locate(query, expected_area="room_b")
    assert resolved is not None
    assert resolved.area_id == "room_b"
    assert resolved.pose == (7, 5)
    assert resolved.evidence["accepted_by"] == "expected_area"


def test_locate_accepts_other_clear_winner_even_with_expected_area() -> None:
    world = WorldModel()
    room_a = SyntheticRoom(width=14, height=11, seed=31)
    room_b = SyntheticRoom(width=14, height=11, seed=32)
    _observe_area(world, room_a, _BIG_POSES, area_id="room_a")
    _observe_area(world, room_b, _BIG_POSES, area_id="room_b")

    query = _view(room_b, (7, 5))
    match = world.locate(query, expected_area="room_a")
    assert match is not None
    assert match.area_id == "room_b"  # clear margin wins over a stale expectation
    assert match.evidence["accepted_by"] == "margin"


def test_match_view_candidate_filter_and_edge_cases() -> None:
    world = WorldModel()
    room_a = SyntheticRoom(width=14, height=11, seed=41)
    room_b = SyntheticRoom(width=14, height=11, seed=42)
    _observe_area(world, room_a, _BIG_POSES, area_id="room_a")
    _observe_area(world, room_b, _BIG_POSES, area_id="room_b")
    query = _view(room_b, (7, 5))

    only_b = world.match_view(query, candidate_area_ids=["room_b"])
    assert only_b and all(match.area_id == "room_b" for match in only_b)
    # Different random tile patterns -> no candidate pose in room_a.
    assert world.match_view(query, candidate_area_ids=["room_a"]) == []
    assert world.match_view(query, candidate_area_ids=["does-not-exist"]) == []

    # Empty / tiny views are never matched.
    empty = FrameView(frame_index=0, signature="s", grid_offset=(0, 0), tiles=[])
    assert world.match_view(empty) == []
    assert world.locate(empty) is None
    tiny = FrameView(
        frame_index=0, signature="s", grid_offset=(0, 0), tiles=list(query.tiles)[:3]
    )
    assert world.match_view(tiny) == []
    assert MIN_COMPARABLE_TILES > 3


def test_match_view_ignores_cells_without_stored_hashes() -> None:
    world = WorldModel()
    record = world.create_area(area_id="sparse")
    for index in range(MIN_COMPARABLE_TILES - 1):
        record.grid.get_cell(index, 0).add_hash("aaa")
    assert world.match_view(_all_tile_view("aaa")) == []
    record.grid.get_cell(MIN_COMPARABLE_TILES - 1, 0).add_hash("aaa")
    matches = world.match_view(_all_tile_view("aaa"))
    assert matches
    assert matches[0].score == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def _saved_world(tmp_path: Path) -> tuple[WorldModel, SyntheticRoom, str]:
    """Build + save a two-area world; returns (world, query_room, query_area_id)."""
    room_a = SyntheticRoom(width=14, height=11, seed=51)
    room_b = SyntheticRoom(width=10, height=9, seed=52)
    world = WorldModel(root=tmp_path)
    _observe_area(world, room_a, [(7, 5), (8, 5), (7, 4)], area_id="room_a", kind="interior")
    record_b = _observe_area(world, room_b, [(4, 4)], area_id="room_b", kind="exterior")

    world.areas["room_a"].grid.mark_state((7, 5), CellState.WALKABLE)
    world.areas["room_a"].grid.confirm_move((7, 5), "right")
    world.set_current("room_a", (8, 5))
    world.record_visit("room_a", frame_index=5)
    world.record_visit("room_a", frame_index=9)
    world.record_visit("room_b", frame_index=3)
    world.add_connection(
        WorldConnection(
            from_area="room_a",
            from_cell=(8, 5),
            direction="right",
            to_area="room_b",
            to_pose=(4, 4),
            kind="walk",
            confidence=0.8,
        )
    )
    world.mark_area_explored("room_a")
    world.save()
    return world, room_b, record_b.area_id


def test_save_load_multi_area_roundtrip(tmp_path: Path) -> None:
    world, query_room, query_area_id = _saved_world(tmp_path)

    world_file = tmp_path / "navigation" / "world.json"
    areas_dir = tmp_path / "navigation" / "areas"
    assert world.world_path == world_file
    assert world_file.is_file()
    assert (areas_dir / "room_a.json").is_file()
    assert (areas_dir / "room_b.json").is_file()
    assert not list((tmp_path / "navigation").glob("*.tmp"))
    assert not list(areas_dir.glob("*.tmp"))

    payload = json.loads(world_file.read_text(encoding="utf-8"))
    assert payload["format"] == WORLD_MODEL_FORMAT
    assert payload["version"] == WORLD_MODEL_FORMAT_VERSION
    assert payload["current_area"] == "room_a"
    assert payload["current_pose"] == [8, 5]
    combined = world_file.read_text(encoding="utf-8") + "".join(
        path.read_text(encoding="utf-8") for path in areas_dir.glob("*.json")
    )
    assert str(tmp_path) not in combined  # no absolute paths inside the data

    # Saving twice produces byte-identical files (deterministic).
    first_snapshot = world_file.read_bytes()
    world.save()
    assert world_file.read_bytes() == first_snapshot

    restored = WorldModel(root=tmp_path)
    assert restored.load() is True
    assert restored.load_warnings == []
    assert set(restored.areas) == {"room_a", "room_b"}
    assert restored.current_area == "room_a"
    assert restored.current_pose == (8, 5)
    assert restored.areas["room_a"].kind == "interior"
    assert restored.areas["room_b"].kind == "exterior"
    assert restored.areas["room_a"].visits == 2
    assert restored.areas["room_a"].explored is True
    assert restored.areas["room_b"].visits == 1
    assert restored.areas["room_a"].entry_cells == [(8, 5)]
    assert restored.connections == world.connections

    # Stored grids are restored cell-for-cell.
    original_grid = world.areas["room_a"].grid
    restored_grid = restored.areas["room_a"].grid
    assert restored_grid.to_dict() == original_grid.to_dict()
    assert restored_grid.count_state(CellState.WALKABLE) == original_grid.count_state(
        CellState.WALKABLE
    )

    # Relocalization still works after the round-trip.
    query = _view(query_room, (4, 4))
    matches = restored.match_view(query)
    assert matches and matches[0].area_id == "room_b"
    assert matches[0].pose == (4, 4)
    assert matches[0].score == pytest.approx(1.0)


def test_save_load_when_root_is_navigation_dir(tmp_path: Path) -> None:
    nav_dir = tmp_path / "navigation"
    nav_dir.mkdir()
    world = WorldModel(root=nav_dir)
    world.create_area(area_id="a", kind="interior")
    world.set_current("a", (1, 2))
    world.save()

    assert (nav_dir / "world.json").is_file()
    assert (nav_dir / "areas" / "a.json").is_file()
    assert not (nav_dir / "navigation").exists()

    restored = WorldModel(root=nav_dir)
    assert restored.load() is True
    assert restored.current_area == "a"
    assert restored.current_pose == (1, 2)
    # A re-save stays in the same layout.
    restored.save()
    assert (nav_dir / "world.json").is_file()


def test_persistence_tolerant_of_corrupt_world(tmp_path: Path) -> None:
    nav_dir = tmp_path / "navigation"
    nav_dir.mkdir()
    (nav_dir / "world.json").write_text("{definitely not json", encoding="utf-8")
    world = WorldModel(root=tmp_path)
    assert world.load() is False
    assert any("corrupt world.json" in warning for warning in world.load_warnings)

    # Unsupported future version is refused, not guessed at.
    (nav_dir / "world.json").write_text(
        json.dumps({"format": WORLD_MODEL_FORMAT, "version": 99, "areas": [], "connections": []}),
        encoding="utf-8",
    )
    future = WorldModel(root=tmp_path)
    assert future.load() is False
    assert any("version 99" in warning for warning in future.load_warnings)

    # Missing world.json is simply a fresh world.
    fresh = WorldModel(root=tmp_path / "elsewhere")
    assert fresh.load() is False
    assert fresh.load_warnings == []


def test_persistence_tolerant_of_corrupt_area_grid(tmp_path: Path) -> None:
    world, _query_room, _ = _saved_world(tmp_path)
    grid_file = tmp_path / "navigation" / "areas" / "room_a.json"
    grid_file.write_text("not json at all", encoding="utf-8")

    restored = WorldModel(root=tmp_path)
    assert restored.load() is True
    assert "room_a" in restored.areas
    assert restored.areas["room_a"].grid.size() == 0  # empty grid fallback
    assert restored.areas["room_a"].visits == world.areas["room_a"].visits
    assert any("unreadable grid file" in warning for warning in restored.load_warnings)


def test_load_unknown_current_area_is_reset(tmp_path: Path) -> None:
    world, _query_room, _ = _saved_world(tmp_path)
    world_file = tmp_path / "navigation" / "world.json"
    payload = json.loads(world_file.read_text(encoding="utf-8"))
    payload["current_area"] = "ghost"
    payload["current_pose"] = [9, 9]
    world_file.write_text(json.dumps(payload), encoding="utf-8")

    restored = WorldModel(root=tmp_path)
    assert restored.load() is True
    assert restored.current_area == ""
    assert restored.current_pose == (0, 0)
    assert any("ghost" in warning for warning in restored.load_warnings)


def test_world_without_root_is_memory_only() -> None:
    world = WorldModel()
    assert world.root is None
    assert world.world_path is None
    assert world.load() is False
    with pytest.raises(ValueError):
        world.save()


def test_persistence_handles_unsafe_area_ids(tmp_path: Path) -> None:
    world = WorldModel(root=tmp_path)
    area_id = "Pallet Town/Route 1"
    world.create_area(area_id=area_id, kind="exterior")
    world.set_current(area_id, (1, 1))
    world.areas[area_id].grid.mark_state((1, 1), CellState.WALKABLE)
    world.save()

    area_files = list((tmp_path / "navigation" / "areas").iterdir())
    assert len(area_files) == 1
    assert area_files[0].suffix == ".json"
    assert "/" not in area_files[0].name and "\\" not in area_files[0].name

    restored = WorldModel(root=tmp_path)
    assert restored.load() is True
    assert set(restored.areas) == {area_id}
    assert restored.current_area == area_id
    assert restored.areas[area_id].grid.count_state(CellState.WALKABLE) == 1


# --------------------------------------------------------------------------- #
# Summaries / graph
# --------------------------------------------------------------------------- #


def test_summary_render_and_graph_dict(tmp_path: Path) -> None:
    world, _query_room, _ = _saved_world(tmp_path)

    summary = world.summary()
    assert summary["areas_total"] == 2
    assert summary["connections"] == 1
    assert summary["current_area"] == "room_a"
    assert summary["current_pose"] == [8, 5]
    assert summary["areas_compact"] == [
        ["room_a", "interior", 2, summary["areas"][0]["walkable"]],
        ["room_b", "exterior", 1, summary["areas"][1]["walkable"]],
    ]
    assert summary["explored_areas"] == 1
    assert summary["unexplored"] == ["room_b"]
    assert summary["cells_total"] > 0
    assert summary["walkable_total"] >= 2
    assert isinstance(summary["current_coverage"], dict)

    text = world.render_summary()
    assert text.startswith("World: 2 areas | 1 connections")
    assert "room_a" in text and "room_b" in text
    assert "Connections:" in text
    assert "* room_a" in text  # current marker
    assert "(explored)" in text

    graph = world.graph_dict()
    assert graph["current_area"] == "room_a"
    assert len(graph["areas"]) == 2
    assert graph["nodes"] is graph["areas"]
    assert len(graph["connections"]) == 1
    assert graph["edges"] is graph["connections"]
    current_node = [node for node in graph["areas"] if node["current"]][0]
    assert current_node["id"] == "room_a"
    assert len(current_node["bounds"]) == 4
    assert current_node["coverage"] > 0.0
