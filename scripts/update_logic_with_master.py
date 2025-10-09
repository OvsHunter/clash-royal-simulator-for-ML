#!/usr/bin/env python3
"""
update_logic_with_master.py

Synchronizes all troop, building, and spell logic files with up-to-date mechanical
stats from cards_master.json. Ensures each logic file contains the correct:
- hitpoints, damage, speed, range, elixir, and targeting behavior
- projectile data (if applicable)
- strategic metadata (role, synergy, counters)
Keeps AI considerations intact.

Output:
- Updated files in data/processed/troop_logic, building_logic, spell_logic
- Summary report in console

Dependencies: json, os, datetime
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

DATA_ROOT = os.path.join("data", "processed")
MASTER_PATH = os.path.join(DATA_ROOT, "cards_master.json")
LOGIC_FOLDERS = {
    "troop": os.path.join(DATA_ROOT, "troop_logic"),
    "building": os.path.join(DATA_ROOT, "building_logic"),
    "spell": os.path.join(DATA_ROOT, "spell_logic"),
}


# --------------------------------------------------------------------------- #
# Utility helpers                                                            #
# --------------------------------------------------------------------------- #

def load_json(path: str) -> Optional[Dict]:
    """Load JSON from disk."""
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: str, payload: Dict) -> None:
    """Persist JSON to disk atomically."""
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(tmp_path, path)


def normalise_key(value: Optional[str]) -> Optional[str]:
    """Normalise identifiers for lookups."""
    if value is None:
        return None
    return "".join(ch.lower() for ch in str(value) if ch.isalnum())


def build_master_index(master_cards: List[Dict]) -> Dict[str, Dict]:
    """Index master data by id/name/key for quick access."""
    index: Dict[str, Dict] = {}
    for card in master_cards:
        keys = {
            normalise_key(card.get("id")),
            normalise_key(card.get("key")),
            normalise_key(card.get("name")),
        }
        for key in filter(None, keys):
            # Prefer richer entries (more keys).
            existing = index.get(key)
            if existing is None or len(card.keys()) > len(existing.keys()):
                index[key] = card
    return index


def extract_mechanics(card: Dict) -> Dict:
    """Create a condensed mechanics payload from master stats."""
    stats = card.get("stats") or {}
    mechanics = {
        "elixir": card.get("elixir"),
        "hitpoints": stats.get("hitpoints"),
        "damage": stats.get("damage"),
        "hit_speed": stats.get("hit_speed"),
        "speed": stats.get("speed"),
        "range": stats.get("range"),
        "targets_ground": bool(stats.get("attacks_ground", True)),
        "targets_air": bool(stats.get("attacks_air", False)),
        "target_only_buildings": bool(stats.get("target_only_buildings")),
        "projectile": stats.get("projectile") or stats.get("projectile_key"),
        "projectile_speed": None,
        "projectile_radius": None,
        "projectile_data": None,
        "per_level": {
            "hitpoints": stats.get("hitpoints_per_level"),
            "damage": stats.get("damage_per_level"),
            "dps": stats.get("dps_per_level"),
        },
    }

    proj_bundle = stats.get("projectile_data")
    if isinstance(proj_bundle, dict) and mechanics["projectile"]:
        projectile_info = proj_bundle.get(mechanics["projectile"])
        if isinstance(projectile_info, dict):
            mechanics["projectile_speed"] = projectile_info.get("speed")
            mechanics["projectile_radius"] = projectile_info.get("radius")
            mechanics["projectile_data"] = projectile_info
    return mechanics


def merge_mechanics(logic_data: Dict, master_card: Dict) -> Tuple[Dict, bool]:
    """Merge canonical mechanics into a logic file."""
    mechanics = extract_mechanics(master_card)
    ai_profile = master_card.get("ai_profile") or {}

    updated = dict(logic_data)
    updated["id"] = master_card.get("id", updated.get("id"))
    updated["name"] = master_card.get("name", updated.get("name"))
    updated["role"] = ai_profile.get("role", updated.get("role"))
    updated["placement_strategy"] = ai_profile.get("placement", updated.get("placement_strategy"))
    updated["best_synergies"] = ai_profile.get("synergy_with", updated.get("best_synergies", []))
    updated["common_counters"] = ai_profile.get("counters", updated.get("common_counters", []))
    updated["attack_plan"] = ai_profile.get("usage", updated.get("attack_plan", []))
    updated["ai_profile_last_synced"] = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    considerations = updated.get("ai_considerations") or {}
    updated["ai_considerations"] = {
        "synergy_weight": considerations.get("synergy_weight", 0.5),
        "defense_weight": considerations.get("defense_weight", 0.4),
        "cycle_priority": considerations.get("cycle_priority", 0.5),
    }

    mechanics_payload = updated.get("mechanics", {})
    mechanics_payload.update(mechanics)
    updated["mechanics"] = mechanics_payload

    projectile_attached = mechanics_payload.get("projectile_data") is not None
    return updated, projectile_attached


def update_logic_folder(folder_path: str, master_index: Dict[str, Dict]) -> Tuple[int, int]:
    """Synchronize one logic directory."""
    if not os.path.isdir(folder_path):
        return (0, 0)

    updated_files = 0
    projectile_links = 0

    for filename in os.listdir(folder_path):
        if not filename.endswith(".json"):
            continue
        path = os.path.join(folder_path, filename)
        logic_data = load_json(path)
        if not isinstance(logic_data, dict):
            continue

        keys = [
            normalise_key(logic_data.get("id")),
            normalise_key(logic_data.get("name")),
        ]

        master_card = None
        for key in filter(None, keys):
            master_card = master_index.get(key)
            if master_card:
                break
        if master_card is None:
            continue

        merged, projectile_attached = merge_mechanics(logic_data, master_card)
        save_json(path, merged)
        updated_files += 1
        if projectile_attached:
            projectile_links += 1

    return updated_files, projectile_links


def print_sync_summary(total_files: int, total_projectiles: int) -> None:
    """Print a concise sync summary."""
    print(f"Synced {total_files} logic files with master stats")
    print(f"Updated {total_projectiles} projectile links")
    print("AI metadata preserved")


# --------------------------------------------------------------------------- #
# Entrypoint                                                                 #
# --------------------------------------------------------------------------- #

def main() -> None:
    master_data = load_json(MASTER_PATH)
    if not isinstance(master_data, list):
        raise SystemExit(f"Missing or invalid master data at {MASTER_PATH}")

    master_index = build_master_index(master_data)

    total_files = 0
    total_projectiles = 0
    for folder in LOGIC_FOLDERS.values():
        updated, projectile_links = update_logic_folder(folder, master_index)
        total_files += updated
        total_projectiles += projectile_links

    print_sync_summary(total_files, total_projectiles)


if __name__ == "__main__":
    main()
