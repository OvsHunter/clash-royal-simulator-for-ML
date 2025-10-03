"""Generate heuristic troop behavior metadata.

This utility stitches together card metadata coming from two sources:

* ``raw_data/cards_all.json`` – authoritative list of every card with its
  Clash Royale wiki link.
* ``data/troops.json`` – normalised combat statistics used by the simulator.

Using a set of deterministic heuristics we infer high level roles, counters
and placement recommendations for **every** troop card.  The result is written
to ``data/card_behaviors.json`` and consumed by the AI helpers in
``ai/card_knowledge.py``.

The heuristic layer is intentionally lightweight – it does not attempt to
perfectly model every interaction, but it provides enough structure for a
rule-based agent (or a learning algorithm) to reason about which cards answer
common threats such as "swarm ground units" or "support a tank push".

The script can be re-run whenever troop statistics change.  Behaviour tags are
stable, deterministic and derived exclusively from the checked-in datasets, so
the output is reproducible without external network calls.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
RAW_CARDS = ROOT / "raw_data" / "cards_all.json"
TROOPS_PATH = ROOT / "data" / "troops.json"
OUTPUT_PATH = ROOT / "data" / "card_behaviors.json"


# --- Helper data structures -------------------------------------------------


@dataclass
class CardStats:
    """Reduced view of troop statistics required for heuristics."""

    name: str
    cost: float
    hp: float
    dmg: float
    hit_speed: float
    speed: float
    range: float
    count: int
    targets: Sequence[str]
    flying: bool
    projectile_radius: float


@dataclass
class BehaviorSummary:
    """High level behaviour and usage hints for a troop."""

    name: str
    wiki_url: Optional[str]
    cost: float
    targets: Sequence[str]
    flying: bool
    primary_roles: List[str] = field(default_factory=list)
    counter_tags: List[str] = field(default_factory=list)
    synergy_tags: List[str] = field(default_factory=list)
    placement_hints: List[str] = field(default_factory=list)
    elixir_profile: str = "cycle"


# --- Classification helpers -------------------------------------------------


WIN_CONDITIONS = {
    "Balloon",
    "Battle Ram",
    "Electro Giant",
    "Elixir Golem",
    "Giant",
    "Goblin Drill",
    "Goblin Giant",
    "Golem",
    "Hog Rider",
    "Lava Hound",
    "Miner",
    "Ram Rider",
    "Royal Giant",
    "Royal Hogs",
    "Skeleton Barrel",
    "Wall Breakers",
}


TANKISH_UNITS = {
    "Dark Prince",
    "Golden Knight",
    "Ice Golem",
    "Knight",
    "Mega Knight",
    "Mini P.E.K.K.A",
    "Prince",
    "Ram Rider",
    "Royal Ghost",
    "Valkyrie",
}


SWARM_THRESHOLDS = {
    "swarm": 5,
    "squad": 3,
}


def load_troop_stats() -> Dict[str, CardStats]:
    raw = json.loads(TROOPS_PATH.read_text())
    stats: Dict[str, CardStats] = {}
    for name, info in raw.items():
        if info.get("type") != "troop":
            continue
        projectile = info.get("projectile_data") or {}
        stats[name] = CardStats(
            name=name,
            cost=float(info.get("cost", 0.0) or 0.0),
            hp=float(info.get("hp", 0.0) or 0.0),
            dmg=float(info.get("dmg", 0.0) or projectile.get("dmg", 0.0) or 0.0),
            hit_speed=float(info.get("hit_speed", 0.0) or 0.0),
            speed=float(info.get("speed", 0.0) or 0.0),
            range=float(info.get("range", 0.0) or 0.0),
            count=int(info.get("count", 0) or 0),
            targets=tuple(info.get("targets", [])),
            flying=bool(info.get("is_air") or info.get("flying")),
            projectile_radius=float(projectile.get("radius", 0.0) or 0.0),
        )
    return stats


def load_card_index() -> Dict[str, Dict[str, str]]:
    cards = json.loads(RAW_CARDS.read_text())
    index: Dict[str, Dict[str, str]] = {}
    for entry in cards:
        if entry.get("type") != "troop":
            continue
        index[entry["name"]] = entry
    return index


def _is_tank(stats: CardStats) -> bool:
    return stats.hp >= 3000 or stats.name in TANKISH_UNITS


def _is_mini_tank(stats: CardStats) -> bool:
    return 1600 <= stats.hp < 3000 and stats.name not in TANKISH_UNITS


def _is_splash(stats: CardStats) -> bool:
    if stats.projectile_radius >= 0.9:
        return True
    if stats.count > SWARM_THRESHOLDS["squad"] and stats.range >= 2.5:
        return True
    if "Wizard" in stats.name or stats.name in {"Valkyrie", "Executioner", "Bowler"}:
        return True
    return False


def _is_ranged(stats: CardStats) -> bool:
    return stats.range >= 4.0 or stats.projectile_radius > 0.0


def _is_cycle(stats: CardStats) -> bool:
    return stats.cost <= 3


def _is_heavy(stats: CardStats) -> bool:
    return stats.cost >= 5


def _swarm_grade(stats: CardStats) -> Optional[str]:
    if stats.count >= SWARM_THRESHOLDS["swarm"]:
        return "swarm"
    if stats.count >= SWARM_THRESHOLDS["squad"]:
        return "squad"
    return None


def _has_anti_air(stats: CardStats) -> bool:
    return "air" in {t.lower() for t in stats.targets}


def _infer_roles(stats: CardStats) -> Tuple[List[str], List[str], List[str], List[str], str]:
    primary: List[str] = []
    counters: List[str] = []
    synergies: List[str] = []
    placements: List[str] = []

    if _is_tank(stats):
        primary.append("frontline_tank")
        synergies.extend(["enable_support_push", "absorb_tower_shots"])
        placements.append("lead_lane_push")
    elif _is_mini_tank(stats):
        primary.append("mini_tank")
        placements.append("bridge_blocker")

    swarm_grade = _swarm_grade(stats)
    if swarm_grade:
        primary.append(f"{swarm_grade}_troop")
        synergies.append("overwhelm_single_target")
        placements.append("split_lane_pressure")

    if _is_ranged(stats):
        primary.append("ranged_support")
        placements.append("place_behind_tank")
        synergies.append("protects_from_distance")

    if _is_splash(stats):
        primary.append("splash_damage")
        counters.extend(["counters_ground_swarm"])
        if stats.flying:
            counters.append("counters_air_swarm")
        placements.append("defend_from_safe_tiles")
        synergies.append("clears_support_swarm")

    if stats.name in WIN_CONDITIONS:
        primary.append("win_condition")
        synergies.append("pair_with_support")
        placements.append("deploy_at_bridge_for_pressure")

    if _has_anti_air(stats):
        counters.append("counters_air_threats")

    if stats.hit_speed and stats.hit_speed <= 1.1 and stats.dmg >= 250:
        primary.append("burst_dps")
        counters.append("melts_tanks")

    if not primary:
        primary.append("support")

    if stats.flying:
        primary.append("air_unit")
        placements.append("kite_over_river")
        synergies.append("requires_air_support")

    if not counters:
        counters.append("flex_counter")

    placements = sorted(dict.fromkeys(placements))
    primary = sorted(dict.fromkeys(primary))
    counters = sorted(dict.fromkeys(counters))
    synergies = sorted(dict.fromkeys(synergies))

    if _is_heavy(stats):
        elixir_profile = "heavy"
    elif _is_cycle(stats):
        elixir_profile = "cycle"
    else:
        elixir_profile = "standard"

    return primary, counters, synergies, placements, elixir_profile


def build_behaviors() -> List[BehaviorSummary]:
    stats_map = load_troop_stats()
    card_index = load_card_index()

    summaries: List[BehaviorSummary] = []
    for name, stats in stats_map.items():
        meta = card_index.get(name, {})
        primary, counters, synergies, placements, profile = _infer_roles(stats)
        summary = BehaviorSummary(
            name=name,
            wiki_url=meta.get("wiki_url"),
            cost=stats.cost,
            targets=list(stats.targets),
            flying=stats.flying,
            primary_roles=primary,
            counter_tags=counters,
            synergy_tags=synergies,
            placement_hints=placements,
            elixir_profile=profile,
        )
        summaries.append(summary)

    summaries.sort(key=lambda s: s.name.lower())
    return summaries


def serialise(summaries: Iterable[BehaviorSummary]) -> List[Dict[str, object]]:
    return [
        {
            "name": s.name,
            "wiki_url": s.wiki_url,
            "cost": s.cost,
            "targets": list(s.targets),
            "flying": s.flying,
            "primary_roles": s.primary_roles,
            "counter_tags": s.counter_tags,
            "synergy_tags": s.synergy_tags,
            "placement_hints": s.placement_hints,
            "elixir_profile": s.elixir_profile,
        }
        for s in summaries
    ]


def main() -> None:
    summaries = build_behaviors()
    data = serialise(summaries)
    OUTPUT_PATH.write_text(json.dumps(data, indent=2) + "\n")
    print(f"Wrote {len(data)} troop behaviour entries -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

