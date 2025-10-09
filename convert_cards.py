import json, sys
from pathlib import Path

TOURNY_LEVEL = {
    "Common": 11,
    "Rare": 9,
    "Epic": 7,
    "Legendary": 5,
    "Champion": 5,
}

BUILDING_ONLY_UNITS = {
    "giant",
    "royal giant",
    "goblin giant",
    "electro giant",
    "golem",
    "elixir golem",
    "battle ram",
    "hog rider",
    "ram rider",
    "miner",
    "wall breakers",
    "balloon",
}

def load_json(path: Path):
    if not path.exists():
        print(f"Missing: {path}")
        return []
    return json.loads(path.read_text(encoding="utf-8"))

def save_json(path: Path, data: dict):
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Saved {path} ({len(data)} entries)")

def pick_level(arr, rarity):
    if not arr: return 0
    lvl = TOURNY_LEVEL.get(rarity, 11) - 1
    lvl = min(lvl, len(arr)-1)
    return arr[lvl]

def simplify_projectile(card):
    name = card.get("name") or card.get("key")
    return {
        "speed": float(card.get("speed",0))/100.0,
        "dmg": card.get("damage",0),
        "radius": float(card.get("radius",0))/1000.0,
        "aoe_to_air": card.get("aoe_to_air",False),
        "aoe_to_ground": card.get("aoe_to_ground",True),
        "pushback": card.get("pushback",0),
    }

def simplify_troop(card, characters, projectiles):
    if card.get("not_in_use") or not card.get("unlock_arena"):
        return None
    rarity = card.get("rarity","Common")
    name = card.get("name_en") or card.get("name") or card.get("key")

    # Find linked character (either summon_character or self)
    char_name = card.get("summon_character") or card.get("name")
    char = characters.get(char_name, {})

    # Pull per-level stats (prefer troop, fallback to character)
    hp_levels = card.get("hitpoints_per_level") or char.get("hitpoints_per_level") or []
    dmg_levels = card.get("damage_per_level") or char.get("damage_per_level") or []

    hp = pick_level(hp_levels, rarity) or card.get("hitpoints") or char.get("hitpoints", 0)
    dmg = pick_level(dmg_levels, rarity) or card.get("damage") or char.get("damage", 0)

    # Movement + range (fallback to character if missing)
    hit_speed = float(card.get("hit_speed") or char.get("hit_speed", 1000)) / 1000.0
    move_speed = float(card.get("speed") or char.get("speed", 0)) / 100.0
    attack_range = float(card.get("range") or char.get("range", char.get("sight_range", 0))) / 1000.0

    # Projectile (check troop, then character)
    proj_name = card.get("projectile") or char.get("projectile")
    proj_data = projectiles.get(proj_name) if proj_name else None

    name_lower = (name or "").lower()
    char_building_only = bool(char.get("target_only_buildings"))
    card_building_only = bool(card.get("target_only_buildings"))
    building_only = bool(char_building_only or card_building_only or name_lower in BUILDING_ONLY_UNITS)

    if building_only:
        targets = ["buildings"]
    else:
        targets = (
            (["ground"] if (card.get("attacks_ground") or char.get("attacks_ground")) else []) +
            (["air"] if (card.get("attacks_air") or char.get("attacks_air")) else [])
        ) or ["ground"]

    return {
        "type": "troop",
        "rarity": rarity,
        "cost": card.get("mana_cost", 0),
        "hp": hp,
        "hp_levels": hp_levels,
        "dmg": dmg,
        "dmg_levels": dmg_levels,
        "hit_speed": hit_speed,
        "speed": move_speed,
        "range": attack_range,
        "count": int(card.get("summon_number") or 1),
        "targets": targets,
        "is_air": bool(card.get("flying_height",0) or char.get("flying_height",0)),
        "summon_character": card.get("summon_character"),
        "projectile": proj_name,
        "projectile_data": proj_data,
        "target_only_buildings": building_only,
    }

def simplify_building(card):
    if card.get("not_in_use") or not card.get("unlock_arena"):
        return None
    rarity = card.get("rarity","Common")
    hp_levels = card.get("hitpoints_per_level") or []
    dmg_levels = card.get("damage_per_level") or []
    return {
        "type": "building",
        "rarity": rarity,
        "cost": card.get("mana_cost", 0),
        "hp": pick_level(hp_levels, rarity) or card.get("hitpoints",0),
        "hp_levels": hp_levels,
        "dmg": pick_level(dmg_levels, rarity) or card.get("damage",0),
        "dmg_levels": dmg_levels,
        "hit_speed": float(card.get("hit_speed", 1000))/1000.0,
        "range": float(card.get("range", 0))/1000.0,
        "lifetime": float(card.get("life_time",0))/1000.0,
        "attacks_ground": card.get("attacks_ground", True),
        "attacks_air": card.get("attacks_air", False),
        "spawns": card.get("spawn_character"),
        "spawn_interval": float(card.get("spawn_interval",0))/1000.0,
    }

def simplify_spell(card):
    if card.get("not_in_use") or not card.get("unlock_arena"):
        return None
    rarity = card.get("rarity","Common")
    dmg_levels = card.get("damage_per_level") or []
    return {
        "type": "spell",
        "rarity": rarity,
        "cost": card.get("mana_cost",0),
        "dmg": pick_level(dmg_levels, rarity) or card.get("damage",0) or card.get("instant_damage",0),
        "dmg_levels": dmg_levels,
        "radius": float(card.get("radius",0))/1000.0,
        "duration": float(card.get("duration_seconds",0)),
        "heal_per_sec": float(card.get("heal_per_second",0)),
        "targets": ["ground","air"] if card.get("aoe_to_air") else ["ground"],
    }

def main():
    if len(sys.argv) < 3:
        print("Usage: python convert_cards.py <input_folder> <output_folder>")
        return
    input_folder = Path(sys.argv[1])
    output_folder = Path(sys.argv[2])
    output_folder.mkdir(parents=True, exist_ok=True)

    troops_raw = load_json(input_folder / "cards_stats_troop.json")
    builds_raw = load_json(input_folder / "cards_stats_building.json")
    spells_raw = load_json(input_folder / "cards_stats_spell.json")
    projs_raw  = load_json(input_folder / "cards_stats_projectile.json")
    chars_raw  = load_json(input_folder / "cards_stats_characters.json")

    characters = {c.get("name") or c.get("key"): c for c in chars_raw}
    projectiles = {c.get("name") or c.get("key"): simplify_projectile(c) for c in projs_raw}

    troops = { (c.get("name_en") or c.get("name")): simplify_troop(c, characters, projectiles)
               for c in troops_raw if simplify_troop(c, characters, projectiles)}
    builds = { (c.get("name_en") or c.get("name")): simplify_building(c)
               for c in builds_raw if simplify_building(c)}
    spells = { (c.get("name_en") or c.get("name")): simplify_spell(c)
               for c in spells_raw if simplify_spell(c)}

    save_json(output_folder / "troops.json", troops)
    save_json(output_folder / "buildings.json", builds)
    save_json(output_folder / "spells.json", spells)
    save_json(output_folder / "projectiles.json", projectiles)

    # 🔎 Debug check
    debug = {}
    for name, troop in troops.items():
        if troop["hp"] <= 0 or troop["dmg"] <= 0:
            debug[name] = troop
    for name, bld in builds.items():
        if bld["hp"] <= 0:
            debug[name] = bld
    for name, spell in spells.items():
        if spell["dmg"] == 0 and spell["heal_per_sec"] == 0:
            debug[name] = spell

    if debug:
        dbg_path = output_folder / "debug_missing.json"
        save_json(dbg_path, debug)
        print(f"[!] Some cards missing values -> see {dbg_path}")
    else:
        print("✅ All cards populated with base stats.")

if __name__ == "__main__":
    main()
