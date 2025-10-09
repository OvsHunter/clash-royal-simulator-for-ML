#!/usr/bin/env python3
"""
Build a consolidated Clash Royale card knowledge base for AI usage.

The script ingests raw exports from ``raw_data/``, enriches them with projectile
stats and AI metadata, and materialises a suite of processed artefacts under
``data/processed/`` (master JSONs, per-card logic files, matrices, summaries).

Idempotent: safe to rerun whenever the raw data is refreshed.
"""

from __future__ import annotations

import csv
import json
import os
import re
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT_DIR, "raw_data")
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
SUMMARY_DIR = os.path.join(PROCESSED_DIR, "summary")
LOGIC_DIRS = {
    "troop": os.path.join(PROCESSED_DIR, "troop_logic"),
    "building": os.path.join(PROCESSED_DIR, "building_logic"),
    "spell": os.path.join(PROCESSED_DIR, "spell_logic"),
}

ROLES_DATA_CANDIDATES = [
    os.path.join(PROCESSED_DIR, "cr_units_roles_counters_master.json"),
    os.path.join(RAW_DIR, "cr_units_roles_counters_master.json"),
]

INACTIVE_TOKENS = ("event", "test", "tutorial", "debug", "limited", "evolved")
TIMESTAMP = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def load_json(path: str) -> Any:
    """Load a JSON document if it exists, returning ``None`` otherwise."""
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_dirs() -> None:
    """Ensure output directories exist."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    for path in LOGIC_DIRS.values():
        os.makedirs(path, exist_ok=True)


def normalise_key(value: Any) -> Optional[str]:
    """Normalise keys (id/key/name) for matching across datasets."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return str(int(value))
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def index_by_keys(entries: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Index raw stat entries by normalised ``id``/``key``/``name``."""
    index: Dict[str, Dict[str, Any]] = {}
    for entry in entries:
        keys = {
            normalise_key(entry.get("id")),
            normalise_key(entry.get("key")),
            normalise_key(entry.get("name")),
            normalise_key(entry.get("name_en")),
        }
        for key in filter(None, keys):
            current = index.get(key)
            if current is None or len(entry) > len(current):
                index[key] = entry
    return index


def detect_inactive(card: Dict[str, Any]) -> bool:
    """Return True if the card should be considered inactive/event-only."""
    name = (card.get("name") or "").lower()
    key = (card.get("key") or "").lower()
    return any(token in name or token in key for token in INACTIVE_TOKENS)


def attach_projectile_data(stats: Dict[str, Any], projectile_lookup: Dict[str, Dict[str, Any]]) -> None:
    """Embed projectile metadata for keys referencing projectile ids."""
    if not stats:
        return
    projectile_keys: List[str] = []
    for key, value in stats.items():
        if not isinstance(value, str):
            continue
        if "projectile" in key.lower():
            projectile_keys.append(value)
    if not projectile_keys:
        return
    enriched: Dict[str, Dict[str, Any]] = {}
    for proj_key in projectile_keys:
        norm = normalise_key(proj_key)
        if not norm:
            continue
        projectile = projectile_lookup.get(norm)
        if projectile:
            enriched[proj_key] = projectile
    if enriched:
        stats.setdefault("projectile_data", enriched)


def load_roles_dataset() -> Dict[str, Dict[str, Any]]:
    """Load optional roles/counters dataset for AI metadata enrichment."""
    for candidate in ROLES_DATA_CANDIDATES:
        data = load_json(candidate)
        if isinstance(data, list):
            lookup: Dict[str, Dict[str, Any]] = {}
            for row in data:
                key = normalise_key(row.get("Name"))
                if key:
                    lookup[key] = row
            return lookup
    return {}


def infer_category(card_type: str, role: str) -> str:
    """Derive a coarse category for AI usage."""
    role_lower = role.lower()
    if "defense" in role_lower:
        return "Defense"
    if "spell" in role_lower:
        return "Spell"
    if "support" in role_lower:
        return "Support"
    if card_type == "spell":
        return "Spell"
    if card_type == "building":
        return "Defense"
    return "Offense"


def default_placement(card_type: str, role: str) -> str:
    """Return a default placement strategy text."""
    if card_type == "spell":
        return "cast based on predicted value (tower pressure or defense)"
    if "tank" in role.lower() or "win condition" in role.lower():
        return "place behind king tower or at bridge when elixir > 8"
    if card_type == "building":
        return "place in middle tiles for optimal pull coverage"
    return "deploy behind tank or in safe tiles to support pushes"


def default_usage(role: str) -> List[str]:
    """Provide default usage hints derived from role text."""
    role_lower = role.lower()
    usage: List[str] = []
    if "support" in role_lower:
        usage.append("support push")
    if "win condition" in role_lower or "tank" in role_lower:
        usage.append("tower pressure")
    if "defense" in role_lower or "building" in role_lower:
        usage.append("bridge defense")
    if not usage:
        usage.append("cycle value")
    return usage


def determine_combo_core(role: str, rarity: str) -> bool:
    """Decide if a card is typically core to combos."""
    role_lower = role.lower()
    return any(token in role_lower for token in ("win condition", "champion", "tank")) or rarity.lower() == "champion"


def extract_stat_value(stats: Dict[str, Any], key: str) -> Optional[float]:
    """Fetch a numeric stat if present."""
    if not stats:
        return None
    value = stats.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def sanitise_filename(name: str) -> str:
    """Create a filesystem-safe filename for per-card logic outputs."""
    safe = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    return safe or "card"


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------

def merge_card_stats(
    card: Dict[str, Any],
    troop_index: Dict[str, Dict[str, Any]],
    building_index: Dict[str, Dict[str, Any]],
    spell_index: Dict[str, Dict[str, Any]],
    projectile_index: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Merge mechanical stats into the base card record."""
    norm_key = normalise_key(card.get("id")) or normalise_key(card.get("key")) or normalise_key(card.get("name"))
    stats: Optional[Dict[str, Any]] = None
    source_file = None
    card_type = card.get("type", "").lower()
    if card_type in ("troop", "champion"):
        stats = troop_index.get(norm_key)
        source_file = "cards_stats_characters.json"
    elif card_type == "building":
        stats = building_index.get(norm_key)
        source_file = "cards_stats_building.json"
    elif card_type == "spell":
        stats = spell_index.get(norm_key)
        source_file = "cards_stats_spell.json"

    stats = stats.copy() if isinstance(stats, dict) else {}
    attach_projectile_data(stats, projectile_index)

    merged = dict(card)
    merged["stats"] = stats
    merged["source_file"] = source_file or "cards_all.json"
    merged["last_updated"] = TIMESTAMP

    # Promote key identifiers and economical metadata from stats when missing.
    fallback_mapping = {
        "id": stats.get("id"),
        "key": stats.get("key") or stats.get("sc_key"),
        "rarity": stats.get("rarity"),
        "elixir": stats.get("elixir") or stats.get("mana_cost"),
        "description": stats.get("description"),
    }
    # Buildings/spells often use mana_cost instead of elixir.
    if fallback_mapping["elixir"] is None:
        fallback_mapping["elixir"] = stats.get("mana_cost")

    for field, value in fallback_mapping.items():
        if value is not None and not merged.get(field):
            merged[field] = value

    return merged


def attach_ai_profile(card: Dict[str, Any], roles_lookup: Dict[str, Dict[str, Any]]) -> None:
    """Attach AI metadata to a card record."""
    norm_key = normalise_key(card.get("name")) or normalise_key(card.get("key"))
    role_entry = roles_lookup.get(norm_key, {})
    role = role_entry.get("Role") or "Unknown Role"
    counters = role_entry.get("Counters") or []
    if isinstance(counters, str):
        counters = [item.strip() for item in counters.split(";") if item.strip()]
    synergy = role_entry.get("Synergies") or role_entry.get("Synergy") or []
    if isinstance(synergy, str):
        synergy = [item.strip() for item in synergy.split(";") if item.strip()]

    card_type = card.get("type", "").lower()
    rarity = card.get("rarity", "Common")

    profile = {
        "role": role,
        "category": infer_category(card_type, role),
        "placement": default_placement(card_type, role),
        "usage": default_usage(role),
        "synergy_with": synergy,
        "counters": counters,
        "countered_by": role_entry.get("CounteredBy", []),
        "elixir_efficiency": role_entry.get("ElixirEfficiency", "unknown"),
        "cycle_value": role_entry.get("CycleValue", card.get("elixir", 0)),
        "is_combo_core": determine_combo_core(role, rarity),
    }

    if isinstance(profile["countered_by"], str):
        profile["countered_by"] = [item.strip() for item in profile["countered_by"].split(";") if item.strip()]
    if not isinstance(profile["usage"], list):
        profile["usage"] = [profile["usage"]]
    if not isinstance(profile["counters"], list):
        profile["counters"] = []
    if not isinstance(profile["synergy_with"], list):
        profile["synergy_with"] = []
    if not isinstance(profile["countered_by"], list):
        profile["countered_by"] = []

    card["ai_profile"] = profile


def write_logic_file(card: Dict[str, Any], directory: str) -> None:
    """Emit per-card logic JSON summarising AI hints."""
    name = card.get("name") or card.get("key") or "Unknown"
    profile = card.get("ai_profile", {})
    logic_payload = {
        "id": card.get("id"),
        "name": name,
        "role": profile.get("role"),
        "placement_strategy": profile.get("placement"),
        "best_synergies": profile.get("synergy_with", []),
        "common_counters": profile.get("counters", []),
        "attack_plan": profile.get("usage", []),
        "defense_usage": profile.get("defense_usage", ["Absorb hits for support troops"] if "tank" in (profile.get("role") or "").lower() else []),
        "punish_condition": profile.get("punish_condition", ["When opponent overcommits on offense"]),
        "ai_considerations": {
            "synergy_weight": 0.8 if profile.get("is_combo_core") else 0.5,
            "defense_weight": 0.6 if profile.get("category") == "Defense" else 0.4,
            "cycle_priority": 0.7 if card.get("elixir", 0) <= 3 else 0.5,
        },
    }
    filepath = os.path.join(directory, f"{sanitise_filename(name)}.json")
    with open(filepath, "w", encoding="utf-8") as handle:
        json.dump(logic_payload, handle, indent=2, ensure_ascii=False)


def build_matrix(cards: List[Dict[str, Any]], field: str) -> Dict[str, List[str]]:
    """Create a matrix mapping card name to a list from ai_profile[field]."""
    matrix: Dict[str, List[str]] = {}
    for card in cards:
        profile = card.get("ai_profile", {})
        values = profile.get(field, [])
        if not isinstance(values, list):
            continue
        matrix[card.get("name", card.get("key", "Unknown"))] = values
    return matrix


def build_roles_summary(cards: List[Dict[str, Any]]) -> List[Tuple[str, int]]:
    """Aggregate card counts per AI role."""
    counts: Dict[str, int] = {}
    for card in cards:
        role = card.get("ai_profile", {}).get("role", "Unknown Role")
        counts[role] = counts.get(role, 0) + 1
    return sorted(counts.items(), key=lambda item: item[0])


def extract_summary_row(card: Dict[str, Any]) -> Dict[str, Any]:
    """Extract minimal fields for summary exports."""
    stats = card.get("stats", {})
    return {
        "id": card.get("id"),
        "name": card.get("name"),
        "type": card.get("type"),
        "rarity": card.get("rarity"),
        "elixir": card.get("elixir"),
        "hitpoints": extract_stat_value(stats, "hitpoints"),
        "damage": extract_stat_value(stats, "damage"),
        "range": extract_stat_value(stats, "range"),
        "speed": extract_stat_value(stats, "speed"),
        "role": card.get("ai_profile", {}).get("role"),
        "active": card.get("active"),
    }


def write_csv(path: str, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    """Write rows out to CSV."""
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown_table(path: str, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    """Emit a simple Markdown table for the provided rows."""
    header = "| " + " | ".join(fieldnames) + " |\n"
    separator = "| " + " | ".join("---" for _ in fieldnames) + " |\n"
    lines = [header, separator]
    for row in rows:
        values = [str(row.get(field, "")) if row.get(field) is not None else "" for field in fieldnames]
        lines.append("| " + " | ".join(values) + " |\n")
    with open(path, "w", encoding="utf-8") as handle:
        handle.writelines(lines)


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

def main() -> None:
    ensure_dirs()

    cards_all = load_json(os.path.join(RAW_DIR, "cards_all.json")) or []
    troops_data = load_json(os.path.join(RAW_DIR, "cards_stats_characters.json")) or []
    buildings_data = load_json(os.path.join(RAW_DIR, "cards_stats_building.json")) or []
    spells_data = load_json(os.path.join(RAW_DIR, "cards_stats_spell.json")) or []
    projectiles_data = load_json(os.path.join(RAW_DIR, "cards_stats_projectile.json")) or []

    troop_index = index_by_keys(troops_data)
    building_index = index_by_keys(buildings_data)
    spell_index = index_by_keys(spells_data)
    projectile_index = index_by_keys(projectiles_data)
    roles_lookup = load_roles_dataset()

    active_cards: List[Dict[str, Any]] = []
    inactive_cards: List[Dict[str, Any]] = []
    seen_keys: set[str] = set()

    for base_card in cards_all:
        merged = merge_card_stats(base_card, troop_index, building_index, spell_index, projectile_index)
        norm_key = normalise_key(merged.get("id")) or normalise_key(merged.get("key")) or normalise_key(merged.get("name"))
        if norm_key in seen_keys:
            continue
        seen_keys.add(norm_key)

        is_active = not detect_inactive(merged)
        merged["active"] = is_active

        if is_active:
            attach_ai_profile(merged, roles_lookup)
            active_cards.append(merged)
        else:
            merged["ai_profile"] = {}
            inactive_cards.append(merged)

    # Write master JSON outputs
    with open(os.path.join(PROCESSED_DIR, "cards_master.json"), "w", encoding="utf-8") as handle:
        json.dump(active_cards, handle, indent=2, ensure_ascii=False)
    with open(os.path.join(PROCESSED_DIR, "cards_inactive.json"), "w", encoding="utf-8") as handle:
        json.dump(inactive_cards, handle, indent=2, ensure_ascii=False)

    # Per-card logic files
    troop_count = 0
    building_count = 0
    spell_count = 0
    for card in active_cards:
        card_type = card.get("type", "").lower()
        if card_type in ("troop", "champion"):
            write_logic_file(card, LOGIC_DIRS["troop"])
            troop_count += 1
        elif card_type == "building":
            write_logic_file(card, LOGIC_DIRS["building"])
            building_count += 1
        elif card_type == "spell":
            write_logic_file(card, LOGIC_DIRS["spell"])
            spell_count += 1

    # Matrices
    counter_matrix = build_matrix(active_cards, "counters")
    synergy_matrix = build_matrix(active_cards, "synergy_with")
    with open(os.path.join(PROCESSED_DIR, "counter_matrix.json"), "w", encoding="utf-8") as handle:
        json.dump(counter_matrix, handle, indent=2, ensure_ascii=False)
    with open(os.path.join(PROCESSED_DIR, "synergy_matrix.json"), "w", encoding="utf-8") as handle:
        json.dump(synergy_matrix, handle, indent=2, ensure_ascii=False)

    # Summary outputs
    summary_rows = [extract_summary_row(card) for card in active_cards]
    cards_fieldnames = ["id", "name", "type", "rarity", "elixir", "hitpoints", "damage", "range", "speed", "role", "active"]
    write_csv(os.path.join(SUMMARY_DIR, "cards_master.csv"), summary_rows, cards_fieldnames)
    write_markdown_table(os.path.join(SUMMARY_DIR, "cards_master.md"), summary_rows, cards_fieldnames)

    roles_summary = [{"role": role, "count": count} for role, count in build_roles_summary(active_cards)]
    roles_fieldnames = ["role", "count"]
    write_csv(os.path.join(SUMMARY_DIR, "roles_summary.csv"), roles_summary, roles_fieldnames)
    write_markdown_table(os.path.join(SUMMARY_DIR, "roles_summary.md"), roles_summary, roles_fieldnames)

    # Console summary
    print(f"Exported {len(active_cards)} playable cards")
    print(f"Stored {len(inactive_cards)} inactive cards")
    print(f"Generated {troop_count} troop logic files")
    print(f"Generated {building_count} building logic files")
    print(f"Generated {spell_count} spell logic files")
    print("Built counter and synergy matrices")
    print("CSV + Markdown summaries complete")


if __name__ == "__main__":
    main()
