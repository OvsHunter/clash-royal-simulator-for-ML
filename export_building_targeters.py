import json, os

# ---------- CONFIG ----------
RAW_TROOPS = os.path.join("raw_data", "cards_stats_troop.json")
MASTER_PATH = os.path.join("data", "processed", "cr_units_roles_counters_master.json")
OUT_BASE = os.path.join("data", "processed", "building_targeting_troops.json")
OUT_LEVELS = os.path.join("data", "processed", "building_targeting_troops_levels.json")

# ---------- LOAD ----------
with open(RAW_TROOPS, "r", encoding="utf-8") as f:
    troops = json.load(f)

try:
    with open(MASTER_PATH, "r", encoding="utf-8") as f:
        master_data = {entry["Name"].lower(): entry for entry in json.load(f)}
except FileNotFoundError:
    master_data = {}

# ---------- FILTER HELPERS ----------
def is_building_targeter(t):
    """Return True if troop only targets buildings."""
    target = str(t.get("target", "")).strip().lower()
    return (
        t.get("target_only_buildings") is True
        or target in ["building", "buildings"]
        or str(t.get("target_type", "")).strip().lower() in ["building", "buildings"]
        or t.get("targetsBuildings") is True
    )

def skip_invalid(t):
    """Skip debug, event, or hidden troops."""
    name = (t.get("name_en") or t.get("name") or "").lower()
    if not name:
        return True
    if t.get("not_in_use") or t.get("not_visible"):
        return True
    if any(tok in name for tok in ["event","challenge","test","tutorial","debug","evolved","limited"]):
        return True
    return False

# ---------- BUILD BASE TROOPS ----------
building_targeters = []
seen = set()

for t in troops:
    name = (t.get("name_en") or t.get("name") or "").strip()
    lname = name.lower()
    if skip_invalid(t) or lname in seen:
        continue
    if not is_building_targeter(t):
        continue
    seen.add(lname)

    troop_data = {
        "name": name,
        "rarity": t.get("rarity"),
        "elixir": t.get("mana_cost") or t.get("elixir"),
        "hitpoints": t.get("hitpoints"),
        "damage": t.get("damage"),
        "attack_speed": t.get("hit_speed"),
        "speed": t.get("speed")
    }

    # Attach role + counters if present in master dataset
    if lname in master_data:
        troop_data["role"] = master_data[lname].get("Role")
        troop_data["counters"] = master_data[lname].get("Counters")

    building_targeters.append(troop_data)

# ---------- SAVE BASE ----------
os.makedirs(os.path.dirname(OUT_BASE), exist_ok=True)
with open(OUT_BASE, "w", encoding="utf-8") as f:
    json.dump(building_targeters, f, indent=2, ensure_ascii=False)

print(f"✅ Exported {len(building_targeters)} building-targeting troops to {OUT_BASE}")

# ---------- BUILD LEVEL STATS ----------
levels_data = {}
for t in troops:
    name = (t.get("name_en") or t.get("name") or "").strip()
    lname = name.lower()
    if lname not in seen:
        continue

    # Extract per-level arrays if present
    hp_levels = t.get("hitpoints_per_level") or []
    dmg_levels = t.get("damage_per_level") or []
    speed = t.get("speed")
    elixir = t.get("mana_cost") or t.get("elixir")

    levels = {}
    for i in range(max(len(hp_levels), len(dmg_levels))):
        # Use actual Clash Royale level range (9–15)
        level = i + 9
        hp = hp_levels[i] if i < len(hp_levels) else t.get("hitpoints")
        dmg = dmg_levels[i] if i < len(dmg_levels) else t.get("damage")
        levels[str(level)] = {"hp": hp, "damage": dmg}

    levels_data[name] = {
        "base_elixir": elixir,
        "speed": speed,
        "levels": levels
    }

# ---------- SAVE LEVEL STATS ----------
with open(OUT_LEVELS, "w", encoding="utf-8") as f:
    json.dump(levels_data, f, indent=2, ensure_ascii=False)

print(f"✅ Exported level stats for {len(levels_data)} troops to {OUT_LEVELS}")

# ---------- PRINT SUMMARY ----------
print("\n🧱 Base Troops:")
for t in building_targeters:
    print(" -", t["name"])
