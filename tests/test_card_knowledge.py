from __future__ import annotations

import json
from pathlib import Path

from ai import card_knowledge


ROOT = Path(__file__).resolve().parents[1]


def test_catalogue_covers_all_troops():
    troops = json.loads((ROOT / "data" / "troops.json").read_text())
    behaviours = card_knowledge.load_catalogue()
    missing = [name for name, info in troops.items() if info.get("type") == "troop" and name.lower() not in behaviours]
    assert not missing, f"missing troop behaviours: {missing}"


def test_recommend_wizard_for_swarms():
    deck = ["Knight", "Wizard", "Fireball", "Baby Dragon"]
    recommendations = card_knowledge.recommend_cards(
        deck,
        threat_tags=["ground_swarm"],
        desired_roles=["ranged_support"],
        max_cost=5,
    )
    assert recommendations, "expected wizard recommendation"
    top_card = recommendations[0][0].name
    assert top_card == "Wizard"


def test_describe_card_returns_string():
    desc = card_knowledge.describe("Baby Dragon")
    assert isinstance(desc, str)
    assert "Baby Dragon" in desc
