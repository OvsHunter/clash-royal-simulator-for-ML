#!/usr/bin/env python3
"""Build a consolidated Clash Royale unit catalogue for ML workflows.

The script pulls raw exports from `data/raw/` (or the legacy `raw_data/`
fallback), filters out non-playable entries, infers coarse-grained roles and
counter-strategies, and writes the processed dataset to CSV/JSON/Markdown.

It is safe to re-run at any time; the output is deterministic so downstream
diffs stay quiet when the raw inputs are unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

RAW_FILENAMES = {
    "card_index": "cards_all.json",
    "troop": "cards_stats_troop.json",
    "building": "cards_stats_building.json",
    "spell": "cards_stats_spell.json",
    "character": "cards_stats_characters.json",
}

EXCLUDED_NAME_SUBSTRINGS = (
    "boss ",
    "tutorial",
    "test",
    "draft",
    "challenge",
    "event",
    "limited",
    "debug",
)

ARENA_TOWER_NAMES = {
    "kingtower",
    "princesstower",
    "archertower",
    "king tower",
    "princess tower",
    "archer tower",
}

ROLE_COUNTER_PRESETS: Dict[str, Sequence[str]] = {
    "Champion - Long-Range Burst DPS": (
        "Spell pressure (Fireball/Lightning)",
        "Assassins or Miner on top",
        "Reset/disable ability timing",
    ),
    "Champion - Dash DPS": (
        "Cheap swarms to stop dash lanes",
        "Reset/knockback effects",
        "Kite into defending buildings",
    ),
    "Champion - Melee Tank Drill": (
        "Air units or swarms",
        "Reset effects (Lightning/Fisherman)",
        "Pull with buildings or Tombstone",
    ),
    "Champion - Defensive Reflection": (
        "Wait out reflection window",
        "High DPS ground units",
        "Opposite-lane pressure",
    ),
    "Champion - Swarm Amplifier": (
        "Spell deny (Poison/Fireball)",
        "Eliminate soul generators quickly",
        "Bridge pressure to split resources",
    ),
    "Win Condition - Building Targeting (Ground)": (
        "Pull with defensive buildings",
        "High single-target DPS (Mini P.E.K.K.A/Infern0)",
        "Cheap swarms for distraction",
    ),
    "Win Condition - Building Targeting (Air)": (
        "Air-targeting buildings (Tesla/Inferno)",
        "Long-range air DPS (Musketeer/Archers)",
        "Swarm or stun to delay hits",
    ),
    "Siege Building / Win Condition": (
        "Tank at the bridge",
        "Spell pressure (Fireball/Rocket)",
        "Earthquake or Miner to break anchor",
    ),
    "Heavy Ground Tank": (
        "Inferno Tower/Dragon",
        "Ground-kiting buildings",
        "High DPS support troops",
    ),
    "Air Tank / Win Condition": (
        "Air-targeting Infernos",
        "Ground tanks to soak pups",
        "Timed spell pressure",
    ),
    "Melee Tank / Bridge Pressure": (
        "Kiting buildings",
        "High DPS melee (Mini P.E.K.K.A)",
        "Air units or swarms",
    ),
    "Melee DPS / Assassin": (
        "Swarm distractions",
        "Air units out of reach",
        "Control placement (tornado/kiting)",
    ),
    "Ranged Support / Anti-Air": (
        "Spell value (Fireball/Poison)",
        "Assassins (Miner/Bandit)",
        "Split-lane pressure",
    ),
    "Air Splash Support": (
        "Long-range air DPS",
        "Spell pressure (Fireball/Lightning)",
        "Keep spacing to dodge splash",
    ),
    "Splash Control": (
        "Snipe with ranged units",
        "Lightning/Fireball",
        "Spread deployments",
    ),
    "Swarm Cycle Unit": (
        "Small spells (Log/Arrows)",
        "Splash damage troops",
        "Force quick cycling",
    ),
    "Support / Healer": (
        "Burst damage to delete support",
        "Crowd control (stuns/knockback)",
        "Separate from tanks before engaging",
    ),
    "Support / Utility": (
        "Trade efficiently",
        "Spell chip when clumped",
        "Opposite lane pressure",
    ),
    "Spawner Building": (
        "Spell value (Poison/Fireball)",
        "Opposite-lane pressure",
        "Bridge blocks to soak spawns",
    ),
    "Defensive Building / Anti-Air": (
        "Spell removal (Fireball/Lightning)",
        "Earthquake pressure",
        "Overwhelm with split attacks",
    ),
    "Defensive Building / Ground Pull": (
        "Earthquake or Lightning",
        "Air offense",
        "Split pressure to drain lifetime",
    ),
    "Resource Building": (
        "Rocket/Earthquake",
        "Miner on placement",
        "Immediate bridge pressure",
    ),
    "Trap Building / Pressure": (
        "Prediction spells",
        "Surround on spawn",
        "Pull with alternate buildings",
    ),
    "Spell - Direct Damage": (
        "Spread troops",
        "Spell bait interchange",
        "Keep key troops out of range",
    ),
    "Spell - Area Control": (
        "Avoid stacking",
        "Pressure opposite lane",
        "Clean up with fast units",
    ),
    "Spell - Buff / Debuff": (
        "Reset effects / knockback",
        "Disengage during buff window",
        "Spell denial on key troops",
    ),
    "Spell - Summon": (
        "Timed spells (Poison/Log)",
        "Splash defenders",
        "Pressure elsewhere to split elixir",
    ),
    "Spell - Support Utility": (
        "Track cycle",
        "Punish mirrored deployments",
        "Force awkward timing",
    ),
}

ROLE_OVERRIDES: Dict[str, Tuple[str, Sequence[str]]] = {
    # Manual touches for cards whose behaviour is easier to express directly.
    "miner": (
        "Melee DPS / Assassin",
        ROLE_COUNTER_PRESETS["Melee DPS / Assassin"],
    ),
    "graveyard": (
        "Spell - Summon",
        ROLE_COUNTER_PRESETS["Spell - Summon"],
    ),
}

CHAMPION_META: Dict[str, Tuple[str, Sequence[str]]] = {
    "archerqueen": (
        "Champion - Long-Range Burst DPS",
        ROLE_COUNTER_PRESETS["Champion - Long-Range Burst DPS"],
    ),
    "goldenknight": (
        "Champion - Dash DPS",
        ROLE_COUNTER_PRESETS["Champion - Dash DPS"],
    ),
    "mightyminer": (
        "Champion - Melee Tank Drill",
        ROLE_COUNTER_PRESETS["Champion - Melee Tank Drill"],
    ),
    "monk": (
        "Champion - Defensive Reflection",
        ROLE_COUNTER_PRESETS["Champion - Defensive Reflection"],
    ),
    "skeletonking": (
        "Champion - Swarm Amplifier",
        ROLE_COUNTER_PRESETS["Champion - Swarm Amplifier"],
    ),
}

SWARM_THRESHOLD = 5
HEAVY_HP_THRESHOLD = 2600
MINI_TANK_HP_THRESHOLD = 1500
RANGED_RANGE_THRESHOLD = 5200
MELEE_SPEED_THRESHOLD = 80

TYPE_ORDER = {
    "Champion": 0,
    "Troop": 1,
    "Building": 2,
    "Spell": 3,
}


def normalize(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def load_json(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def build_lookup(entries: Iterable[dict]) -> Dict[str, dict]:
    mapping: Dict[str, dict] = {}
    for entry in entries:
        for key in ("name_en", "name", "key"):
            value = entry.get(key)
            if not value:
                continue
            mapping.setdefault(normalize(str(value)), entry)
    return mapping


def should_exclude_name(name: str) -> bool:
    lowered = name.lower()
    if any(token in lowered for token in EXCLUDED_NAME_SUBSTRINGS):
        return True
    return normalize(name) in ARENA_TOWER_NAMES


def should_exclude_entry(entry: dict) -> bool:
    if entry.get("not_in_use") or entry.get("not_visible"):
        return True
    if entry.get("type_of_spell") == "Tutorial":
        return True
    return False


def resolve_raw_dir(preferred: Path) -> Path:
    if preferred.exists():
        return preferred
    fallback = Path("raw_data")
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        f"Could not find raw data directory. Checked {preferred} and {fallback}."
    )


def coerce_float(value: Optional[float]) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


@dataclass
class CardFeatures:
    name: str
    norm_name: str
    card_type: str
    rarity: Optional[str]
    elixir: float
    raw_entry: dict
    char_entry: Optional[dict]
    is_flying: bool
    targets_air: bool
    targets_ground: bool
    is_building_targeting: bool
    summon_count: int
    effective_hp: float
    area_radius: float
    attack_range: float
    speed: float


def extract_features(
    name: str,
    card_type: str,
    entry: dict,
    char_entry: Optional[dict],
    rarity: Optional[str],
) -> CardFeatures:
    norm_name = normalize(name)
    char = char_entry or {}
    is_flying = (char.get("flying_height") or 0) > 0
    targets_air = bool(char.get("attacks_air"))
    targets_ground = bool(char.get("attacks_ground", True))
    is_building_targeting = bool(
        char.get("target_only_buildings") or entry.get("target_only_buildings")
    )
    summon_numbers = [
        entry.get("summon_number"),
        entry.get("summon_character_second_count"),
        entry.get("summon_character_second_count2"),
        entry.get("summon_character_second_count3"),
    ]
    summon_count = max([count for count in summon_numbers if count] + [0])
    hitpoints = coerce_float(char.get("hitpoints"))
    effective_hp = hitpoints * max(1, summon_count or 1)
    area_radius = coerce_float(char.get("area_damage_radius") or entry.get("area_damage_radius"))
    attack_range = coerce_float(
        char.get("range") or char.get("sight_range") or entry.get("range")
    )
    speed = coerce_float(char.get("speed"))
    return CardFeatures(
        name=name,
        norm_name=norm_name,
        card_type=card_type,
        rarity=rarity,
        elixir=coerce_float(entry.get("mana_cost") or entry.get("elixir")),
        raw_entry=entry,
        char_entry=char_entry,
        is_flying=is_flying,
        targets_air=targets_air,
        targets_ground=targets_ground,
        is_building_targeting=is_building_targeting,
        summon_count=summon_count,
        effective_hp=effective_hp,
        area_radius=area_radius,
        attack_range=attack_range,
        speed=speed,
    )


SPLASH_NAMES = {
    "wizard",
    "icewizard",
    "electrowizard",
    "magicarcher",
    "executioner",
    "bowler",
    "valkyrie",
    "babydragon",
    "megaknight",
    "electrodragon",
    "motherwitch",
    "witch",
    "darkprince",
    "bomber",
}

HEALER_NAMES = {
    "battlehealer",
    "healspirit",
    "monk",
}

MELEE_DPS_NAMES = {
    "minipekka",
    "bandit",
    "berserker",
    "prince",
    "ramrider",
    "lumberjack",
    "royalghost",
}


def infer_troop_role(features: CardFeatures) -> str:
    name = features.norm_name
    if name in ROLE_OVERRIDES:
        return ROLE_OVERRIDES[name][0]
    if features.rarity == "Champion":
        if name in CHAMPION_META:
            return CHAMPION_META[name][0]
        return "Champion - Defensive Reflection"
    if features.is_building_targeting:
        return (
            "Win Condition - Building Targeting (Air)"
            if features.is_flying
            else "Win Condition - Building Targeting (Ground)"
        )
    if features.is_flying and features.area_radius > 0:
        return "Air Splash Support"
    if name in HEALER_NAMES or features.raw_entry.get("heal_per_second"):
        return "Support / Healer"
    if features.summon_count >= SWARM_THRESHOLD or features.raw_entry.get("is_a_group"):
        return "Swarm Cycle Unit"
    if features.area_radius > 0 or name in SPLASH_NAMES:
        return "Splash Control"
    if features.effective_hp >= HEAVY_HP_THRESHOLD * 1.25 and not features.is_flying:
        return "Heavy Ground Tank"
    if features.is_flying and features.effective_hp >= HEAVY_HP_THRESHOLD:
        return "Air Tank / Win Condition"
    if features.effective_hp >= MINI_TANK_HP_THRESHOLD:
        return "Melee Tank / Bridge Pressure"
    if (
        name in MELEE_DPS_NAMES
        or (features.speed >= MELEE_SPEED_THRESHOLD and features.attack_range <= 2000)
    ):
        return "Melee DPS / Assassin"
    if features.attack_range >= RANGED_RANGE_THRESHOLD and features.targets_air:
        return "Ranged Support / Anti-Air"
    return "Support / Utility"


def infer_building_role(features: CardFeatures) -> str:
    raw = features.raw_entry
    name = features.norm_name
    if name in ROLE_OVERRIDES:
        return ROLE_OVERRIDES[name][0]
    if raw.get("spawn_interval", 0):
        return "Spawner Building"
    if "elixir" in name or "collector" in name:
        return "Resource Building"
    if raw.get("range", 0) >= 6500:
        return "Siege Building / Win Condition"
    if features.is_building_targeting:
        return "Trap Building / Pressure"
    if raw.get("attacks_air") and raw.get("attacks_ground"):
        return "Defensive Building / Anti-Air"
    return "Defensive Building / Ground Pull"


def infer_spell_role(features: CardFeatures) -> str:
    raw = features.raw_entry
    name = features.norm_name
    if name in ROLE_OVERRIDES:
        return ROLE_OVERRIDES[name][0]
    if raw.get("spawn_character") or raw.get("spawn_interval"):
        return "Spell - Summon"
    if raw.get("heal_per_second") or raw.get("buff_time") or raw.get("buff"):
        return "Spell - Buff / Debuff"
    if raw.get("duration_seconds") or raw.get("life_duration"):
        return "Spell - Area Control"
    if "mirror" in name or "clone" in name or "rage" in name:
        return "Spell - Support Utility"
    return "Spell - Direct Damage"


def infer_role(features: CardFeatures) -> Tuple[str, Sequence[str]]:
    name = features.norm_name
    if name in ROLE_OVERRIDES:
        return ROLE_OVERRIDES[name]
    if features.card_type == "Champion":
        meta = CHAMPION_META.get(name)
        if meta:
            return meta
        role = "Champion - Defensive Reflection"
    elif features.card_type == "Troop":
        role = infer_troop_role(features)
    elif features.card_type == "Building":
        role = infer_building_role(features)
    elif features.card_type == "Spell":
        role = infer_spell_role(features)
    else:
        role = "Support / Utility"
    counters = ROLE_COUNTER_PRESETS.get(role, ROLE_COUNTER_PRESETS["Support / Utility"])
    return role, counters


def format_elixir(value: float) -> float:
    return float(int(value)) if abs(value - round(value)) < 1e-6 else round(value, 2)


def update_role_overrides_from_champions() -> None:
    # Normalize champion meta keys once at import time.
    normalized: Dict[str, Tuple[str, Sequence[str]]] = {}
    for key, value in CHAMPION_META.items():
        normalized[normalize(key)] = value
    CHAMPION_META.clear()
    CHAMPION_META.update(normalized)


def generate_records(raw_dir: Path) -> Tuple[List[dict], List[str]]:
    card_index = list(load_json(raw_dir / RAW_FILENAMES["card_index"]))
    troop_lookup = build_lookup(load_json(raw_dir / RAW_FILENAMES["troop"]))
    building_lookup = build_lookup(load_json(raw_dir / RAW_FILENAMES["building"]))
    spell_lookup = build_lookup(load_json(raw_dir / RAW_FILENAMES["spell"]))
    character_lookup = build_lookup(load_json(raw_dir / RAW_FILENAMES["character"]))

    processed: Dict[str, dict] = {}
    skipped: List[str] = []

    for card in card_index:
        name = card["name"]
        if should_exclude_name(name):
            continue
        normalized_name = normalize(name)
        raw_entry: Optional[dict] = None
        rarity: Optional[str] = None
        card_type = card["type"].lower()

        if card_type == "troop":
            raw_entry = troop_lookup.get(normalized_name)
        elif card_type == "building":
            raw_entry = building_lookup.get(normalized_name)
        elif card_type == "spell":
            raw_entry = spell_lookup.get(normalized_name)

        if not raw_entry or should_exclude_entry(raw_entry):
            skipped.append(name)
            continue

        rarity = raw_entry.get("rarity")
        elixir_cost = coerce_float(raw_entry.get("mana_cost") or raw_entry.get("elixir"))
        if elixir_cost <= 0:
            skipped.append(name)
            continue

        char_entry: Optional[dict] = None
        if card_type == "troop":
            candidates = [
                raw_entry.get("summon_character"),
                raw_entry.get("name_en"),
                raw_entry.get("name"),
                raw_entry.get("key"),
            ]
            for candidate in candidates:
                if not candidate:
                    continue
                char_entry = character_lookup.get(normalize(str(candidate)))
                if char_entry:
                    break

        output_type = "Troop"
        if rarity == "Champion":
            output_type = "Champion"
        elif card_type == "building":
            output_type = "Building"
        elif card_type == "spell":
            output_type = "Spell"

        features = extract_features(name, output_type, raw_entry, char_entry, rarity)
        role, counters = infer_role(features)
        processed[name] = {
            "Name": name,
            "Type": output_type,
            "Role": role,
            "Counters": list(counters),
            "Elixir Cost": format_elixir(features.elixir),
        }

    ordered = sorted(
        processed.values(),
        key=lambda record: (TYPE_ORDER.get(record["Type"], 9), record["Name"]),
    )
    return ordered, skipped


def write_json(path: Path, records: Sequence[dict]) -> None:
    path.write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def write_csv(path: Path, records: Sequence[dict]) -> None:
    fieldnames = ["Name", "Type", "Elixir Cost", "Role", "Counters"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = record.copy()
            row["Counters"] = json.dumps(record["Counters"], ensure_ascii=False)
            writer.writerow(row)


def write_markdown(path: Path, records: Sequence[dict]) -> None:
    lines = [
        "| Name | Type | Elixir | Role | Counters |",
        "| --- | --- | --- | --- | --- |",
    ]
    for record in records:
        counters_inline = json.dumps(record["Counters"], ensure_ascii=False)
        lines.append(
            f"| {record['Name']} | {record['Type']} | {record['Elixir Cost']} | "
            f"{record['Role']} | `{counters_inline}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate consolidated Clash Royale unit dataset."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Folder containing raw card exports (default: data/raw, fallback: raw_data).",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
        help="Folder where processed outputs are written (default: data/processed).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    update_role_overrides_from_champions()
    raw_dir = resolve_raw_dir(args.raw_dir)
    processed_dir = args.processed_dir
    processed_dir.mkdir(parents=True, exist_ok=True)

    records, skipped = generate_records(raw_dir)
    if not records:
        print("No records generated.", file=sys.stderr)
        return 1

    write_json(processed_dir / "cr_units_roles_counters_master.json", records)
    write_csv(processed_dir / "cr_units_roles_counters_master.csv", records)
    write_markdown(processed_dir / "cr_units_roles_counters_master.md", records)

    print(f"Wrote {len(records)} cards to {processed_dir}.")
    if skipped:
        missing = ", ".join(sorted(set(skipped)))
        print(
            "Skipped entries without complete data: " + missing,
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
