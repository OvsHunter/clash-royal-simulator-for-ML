from __future__ import annotations

from typing import Iterable

from core.arena import Arena
from core.simulation import Engine, Unit
from core.troop_data import get_card


BUILDING_TARGET_UNITS = [
    "Giant",
    "Royal Giant",
    "Goblin Giant",
    "Electro Giant",
    "Golem",
    "Elixir Golem",
    "Battle Ram",
    "Hog Rider",
    "Ram Rider",
    "Miner",
    "Wall Breakers",
    "Balloon",
]


def _make_engine() -> Engine:
    arena = Arena("data/arena.json")
    engine = Engine(arena)
    engine.players[1].elixir = 10.0
    engine.players[2].elixir = 10.0
    return engine


def _find_unit(units: Iterable[Unit], name: str, owner: int) -> Unit:
    for unit in units:
        if unit.name == name and unit.owner == owner:
            return unit
    raise AssertionError(f"Unit {name} for owner {owner} not found")


def test_building_target_cards_marked_for_buildings() -> None:
    for name in BUILDING_TARGET_UNITS:
        card = get_card(name, 11)
        assert card["targets"] == ["buildings"], f"{name} targets {card['targets']}"
        assert card.get("target_only_buildings") is True


def test_building_target_unit_ignores_distant_troops() -> None:
    engine = _make_engine()
    engine.deploy(1, "Hog Rider", 7, 24)
    engine.deploy(2, "Knight", 7, 10)
    engine.tick()

    hog = _find_unit(engine.units, "Hog Rider", 1)
    knight = _find_unit(engine.units, "Knight", 2)

    # Place the knight within aggro range but outside the engagement radius.
    knight.x = hog.x + 3.0
    knight.y = hog.y

    enemies = [u for u in engine.units if u.owner == 2]
    target = engine._choose_target_object(hog, enemies)
    assert "x0" in target and "y0" in target  # structure target
    assert target["owner"] == 2


def test_building_target_unit_engages_blockers_when_touching() -> None:
    engine = _make_engine()
    engine.deploy(1, "Hog Rider", 7, 24)
    engine.deploy(2, "Knight", 7, 10)
    engine.tick()

    hog = _find_unit(engine.units, "Hog Rider", 1)
    knight = _find_unit(engine.units, "Knight", 2)

    knight.x = hog.x
    knight.y = hog.y

    enemies = [u for u in engine.units if u.owner == 2]
    target = engine._choose_target_object(hog, enemies)
    assert target.get("id") == knight.id
    assert "x" in target and "y" in target


def test_building_target_unit_pushes_past_blockers() -> None:
    engine = _make_engine()
    engine.deploy(1, "Hog Rider", 7, 24)
    engine.deploy(2, "Knight", 7, 10)
    engine.tick()

    hog = _find_unit(engine.units, "Hog Rider", 1)
    knight = _find_unit(engine.units, "Knight", 2)

    # Keep the blocker stationary and durable so it remains in place.
    knight.hp = 1e9
    knight.speed = 0.0
    knight._path = []
    knight._path_goal_id = None
    knight.x = hog.x
    knight.y = hog.y - 1.0
    knight.damage = 0.0

    initial_y = hog.y
    for _ in range(40):
        engine.tick()

    assert hog.y + 1e-3 < knight.y
    assert hog.y < initial_y


def test_building_target_unit_ignores_stuns_and_slows() -> None:
    engine = _make_engine()
    engine.deploy(1, "Hog Rider", 7, 24)
    engine.tick()

    hog = _find_unit(engine.units, "Hog Rider", 1)
    start_y = hog.y

    # Enemy freeze/stun should have no impact.
    engine.apply_stun_area(owner=2, x=hog.x, y=hog.y, radius=2.0, duration=3.0)
    for _ in range(5):
        engine.tick()

    assert getattr(hog, "_stun_timer") == 0.0
    assert hog.y < start_y

    # Direct slow also ignored.
    hog.apply_slow(0.2, 5.0)
    assert getattr(hog, "_slow_timer") == 0.0
    assert getattr(hog, "_slow_multiplier") >= 1.0
