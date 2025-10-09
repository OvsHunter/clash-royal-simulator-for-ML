from __future__ import annotations

from pathlib import Path

from core.arena import Arena
from core.simulation import Engine


def _arena_path() -> str:
    root = Path(__file__).resolve().parents[1]
    return str(root / "data" / "arena.json")


def _prep_engine() -> Engine:
    arena = Arena(_arena_path())
    engine = Engine(arena)
    engine.players[1].elixir = 10.0
    engine.players[2].elixir = 10.0
    return engine


def test_auto_reason_and_threat_interactions_for_troops() -> None:
    engine = _prep_engine()

    # Spawn an enemy Hog Rider so there is an active threat on the board.
    engine.deploy(2, "Hog Rider", 7, 10)
    engine.tick()
    engine.poll_decisions()

    # Advance the simulation so the Hog Rider crosses the river and poses a threat.
    for _ in range(200):
        engine.tick()

    engine.deploy(1, "Knight", 7, 24)
    engine.tick()

    decisions = engine.poll_decisions()
    assert len(decisions) == 1
    record = decisions[0]

    assert record.player == 1
    assert record.card_key == "Knight"
    assert record.reason.startswith("Deployed Knight (troop)")
    assert "Hog Rider" in record.reason
    assert "structure" not in record.reason.lower()

    metadata = record.metadata
    assert metadata["card_role"] == "troop"
    assert metadata["card_targets"]

    threats = metadata["interactions"]["threats"]
    hog_entries = [entry for entry in threats if entry["card_key"] == "Hog Rider"]
    assert hog_entries, "expected Hog Rider to be reported as a threat"
    assert hog_entries[0]["engageable"] is True

    supports = metadata["interactions"]["supports"]
    assert isinstance(supports, list)


def test_auto_reason_mentions_support_when_no_enemy_threats() -> None:
    engine = _prep_engine()

    engine.deploy(1, "Musketeer", 6, 24)
    engine.tick()
    engine.poll_decisions()

    engine.deploy(1, "Knight", 7, 24)
    engine.tick()

    decisions = engine.poll_decisions()
    assert len(decisions) == 1
    record = decisions[0]

    assert "support" in record.reason.lower()

    supports = record.metadata["interactions"]["supports"]
    assert any(entry["card_key"] == "Musketeer" for entry in supports)

    threats = record.metadata["interactions"]["threats"]
    assert threats == []
