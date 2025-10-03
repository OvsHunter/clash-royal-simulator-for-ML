from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .rules import (
    DEFAULT_ARROWS_PROJECTILE,
    DEFAULT_FIREBALL_DAMAGE,
    DEFAULT_FIREBALL_RADIUS,
    DEFAULT_FIREBALL_SPEED,
    DEFAULT_POISON_DPS,
    DEFAULT_POISON_DURATION,
    DEFAULT_RAGE_DURATION,
    DEFAULT_RAGE_LINGER,
    DEFAULT_RAGE_SPEED_MULTIPLIER,
    DEFAULT_SMALL_RADIUS,
    DEFAULT_TANK_RADIUS,
    DEFAULT_ZAP_STUN,
    TANK_HP_THRESHOLD,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OVERRIDE_PATH = ROOT / "data" / "troop_overrides.json"

FALLBACK_SPELLS = {
    "Fireball": {
        "card_type": "spell",
        "cost": 4,
        "projectile": "FireballSpell",
        "radius": DEFAULT_FIREBALL_RADIUS,
        "dmg": DEFAULT_FIREBALL_DAMAGE,
        "speed": DEFAULT_FIREBALL_SPEED,
    },
    "Arrows": {
        "card_type": "spell",
        "cost": 3,
        "projectile": DEFAULT_ARROWS_PROJECTILE,
        "radius": 2.0,
        "dmg": 144.0,
    },
    "Zap": {
        "card_type": "spell",
        "cost": 2,
        "radius": 2.5,
        "dmg": 192.0,
        "stun_duration": DEFAULT_ZAP_STUN,
    },
    "Poison": {
        "card_type": "spell",
        "cost": 4,
        "radius": 3.5,
        "duration": DEFAULT_POISON_DURATION,
        "tick_dps": DEFAULT_POISON_DPS,
    },
    "Rage": {
        "card_type": "spell",
        "cost": 2,
        "radius": 5.0,
        "duration": DEFAULT_RAGE_DURATION,
        "speed_multiplier": DEFAULT_RAGE_SPEED_MULTIPLIER,
        "linger": DEFAULT_RAGE_LINGER,
    },
}

DEFAULT_TOURNAMENT_LEVELS = {
    "common": 11,
    "rare": 9,
    "epic": 6,
    "legendary": 5,
    "champion": 5,
}

_cache_cards: Dict[str, Dict[str, Any]] = {}
_loaded = False


def _normalize_targets(raw: Any) -> List[str]:
    if isinstance(raw, str):
        raw = raw.lower()
        if raw == "both":
            return ["ground", "air"]
        if raw in ("ground", "air", "buildings"):
            return [raw]
        return ["ground", "air"]
    if isinstance(raw, (list, tuple)):
        out: List[str] = []
        for entry in raw:
            entry_str = str(entry).lower()
            if entry_str == "both":
                out.extend(["ground", "air"])
            elif entry_str in ("ground", "air", "buildings"):
                out.append(entry_str)
        return out or ["ground", "air"]
    return ["ground", "air"]


def _level_value(values: Any, level: int) -> Optional[float]:
    if not values:
        return None
    try:
        lvl = max(1, int(level))
    except (TypeError, ValueError):
        lvl = 1
    if isinstance(values, dict):
        # Some datasets store per-level dicts keyed by string levels.
        if str(lvl) in values:
            return float(values[str(lvl)])
        try_levels = sorted(int(k) for k in values.keys())
        if not try_levels:
            return None
        idx = min(max(lvl, try_levels[0]), try_levels[-1])
        return float(values[str(idx)])
    seq = list(values)
    if not seq:
        return None
    idx = max(0, min(len(seq) - 1, lvl - 1))
    try:
        return float(seq[idx])
    except (TypeError, ValueError):
        return None


def _default_level(card: Dict[str, Any]) -> int:
    rarity = str(card.get("rarity", "")).lower()
    return DEFAULT_TOURNAMENT_LEVELS.get(rarity, 11)


def _coerce_float(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _normalize_card(name: str, raw: Dict[str, Any]) -> Dict[str, Any]:
    card = raw.copy()
    card.setdefault("name", name)
    card.setdefault("card_type", "troop")
    card.setdefault("cost", 3)
    card.setdefault("hp", 300)
    card.setdefault("dmg", 100)
    card.setdefault("hit_speed", 1.0)
    card.setdefault("speed", 1.0)
    card.setdefault("range", 1.0)
    card.setdefault("count", 1)
    card.setdefault("targets", "both")
    card.setdefault("is_air", False)

    card["targets"] = _normalize_targets(card.get("targets", "both"))
    card["flying"] = bool(card.get("is_air", card.get("flying", False)))

    # Convert key numeric fields to float for consistency.
    card["hp"] = _coerce_float(card.get("hp", 300.0), 300.0)
    card["dmg"] = _coerce_float(card.get("dmg", 100.0), 100.0)
    card["hit_speed"] = _coerce_float(card.get("hit_speed", 1.0), 1.0)
    card["speed"] = _coerce_float(card.get("speed", 1.0), 1.0)
    card["range"] = max(0.1, _coerce_float(card.get("range", 1.0), 1.0))

    radius = card.get("radius", 0.0) or card.get("radius_collision") or card.get("collision_radius")
    radius = _coerce_float(radius, 0.0)
    if radius <= 0.0:
        hp_estimate = _coerce_float(card.get("hp", 0.0), 0.0)
        radius = DEFAULT_TANK_RADIUS if hp_estimate >= TANK_HP_THRESHOLD else DEFAULT_SMALL_RADIUS
    card["radius"] = radius

    # Normalise per-level arrays to lists of floats where present.
    for key in ("hp_levels", "dmg_levels"):
        levels = card.get(key)
        if isinstance(levels, list):
            card[key] = [ _coerce_float(v, 0.0) for v in levels ]

    card["_default_level"] = _default_level(card)
    return card


def _normalize_spell(name: str, raw: Dict[str, Any]) -> Dict[str, Any]:
    spell = raw.copy()
    spell.setdefault("card_type", "spell")
    spell.setdefault("cost", 2)
    spell.setdefault("radius", 2.5)
    spell.setdefault("duration", 0.0)
    spell.setdefault("targets", ["ground", "air"])
    spell.setdefault("dmg", 0.0)
    spell.setdefault("tick_dps", spell.get("damage_per_second", 0.0))
    spell["targets"] = _normalize_targets(spell.get("targets", ["ground", "air"]))
    spell["_default_level"] = _default_level(spell)
    return spell


def _inject_projectile_metadata(card: Dict[str, Any]) -> None:
    projectile_data = card.get("projectile_data") or {}
    card["projectile_speed"] = _coerce_float(
        projectile_data.get("speed", card.get("projectile_speed", 0.0)), 0.0
    )
    card["splash_radius"] = _coerce_float(
        projectile_data.get("radius", card.get("splash_radius", 0.0)), 0.0
    )
    # Target coverage sourced from card targeting rules.
    targets = card.get("targets", ["ground", "air"])
    if isinstance(targets, str):
        targets = _normalize_targets(targets)
    hits_air = any(t in ("air",) for t in targets)
    hits_ground = any(t in ("ground", "buildings") for t in targets)
    card["projectile_hits_air"] = hits_air
    card["projectile_hits_ground"] = hits_ground




def _merge_overrides(cards: Dict[str, Dict[str, Any]]) -> None:
    if not OVERRIDE_PATH.exists():
        return
    with OVERRIDE_PATH.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, dict):
        return
    for name, override in raw.items():
        base = cards.get(name)
        if not base or not isinstance(override, dict):
            continue
        for key, value in override.items():
            base[key] = value
def _load_cards() -> None:
    global _loaded, _cache_cards
    if _loaded:
        return

    cards: Dict[str, Dict[str, Any]] = {}
    troops_path = DATA_DIR / "troops.json"
    if troops_path.exists():
        with troops_path.open("r", encoding="utf-8") as fh:
            raw_cards = json.load(fh)
        for name, raw in raw_cards.items():
            if isinstance(raw, dict):
                cards[name] = _normalize_card(name, raw)

    spells_path = DATA_DIR / "spells.json"
    if spells_path.exists():
        with spells_path.open("r", encoding="utf-8") as sfh:
            raw_spells = json.load(sfh)
        for name, raw in raw_spells.items():
            if isinstance(raw, dict):
                cards[name] = _normalize_spell(name, raw)

    for spell_name, defaults in FALLBACK_SPELLS.items():
        cards.setdefault(spell_name, _normalize_spell(spell_name, defaults))

    _merge_overrides(cards)
    _cache_cards = cards
    _loaded = True


def get_card(name: str, level: Optional[int] = None) -> Optional[Dict[str, Any]]:
    _load_cards()
    base = _cache_cards.get(name)
    if base is None:
        return None

    card = copy.deepcopy(base)
    target_level = level if level is not None else card.get("_default_level", 11)
    try:
        target_level = int(target_level)
    except (TypeError, ValueError):
        target_level = int(card.get("_default_level", 11))
    card["level"] = target_level

    if card.get("card_type") != "troop":
        return card

    hp_value = _level_value(card.get("hp_levels"), target_level)
    if hp_value is not None:
        card["hp"] = float(hp_value)
    card["hp_max"] = float(card.get("hp", 0.0))

    dmg_value = _level_value(card.get("dmg_levels"), target_level)
    if dmg_value is not None:
        card["dmg"] = float(dmg_value)

    projectile_data = card.get("projectile_data") or {}
    projectile_levels = projectile_data.get("dmg_levels")
    projectile_damage = _level_value(projectile_levels, target_level)
    if projectile_damage is None:
        projectile_damage = projectile_data.get("dmg")
    if _coerce_float(card.get("dmg"), 0.0) <= 0.0 and projectile_damage is not None:
        card["dmg"] = _coerce_float(projectile_damage, 0.0)

    _inject_projectile_metadata(card)
    return card


def list_cards() -> List[str]:
    _load_cards()
    return sorted(_cache_cards.keys())
