#!/usr/bin/env python3
"""validate_simulation_balance.py

Compare simulated DPS vs. stat-sheet DPS for troop cards. Results are written to
data/processed/summary/simulation_validation.csv and summarized on stdout.
"""

from __future__ import annotations

import csv
import json
import os
import statistics
import sys
from typing import Dict, List

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.arena import Arena  # type: ignore
from core.simulation import Engine  # type: ignore

MASTER_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "cards_master.json")
ARENA_PATH = os.path.join(PROJECT_ROOT, "data", "arena.json")
SUMMARY_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "summary")
SUMMARY_FILE = os.path.join(SUMMARY_DIR, "simulation_validation.csv")

ALLOWED_TYPES = {"troop", "champion"}
TARGET_DELTA = 5.0  # percent
WARNING_DELTA = 10.0  # percent


def load_master_cards() -> List[Dict]:
    if not os.path.exists(MASTER_PATH):
        raise SystemExit(f"Missing master file at {MASTER_PATH}")
    with open(MASTER_PATH, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise SystemExit("cards_master.json is not a list")
    return payload


def evaluate_card(card_name: str, *, level: int | None = None) -> Dict[str, object]:
    arena = Arena(ARENA_PATH)
    engine = Engine(arena)
    try:
        result = engine.validate_dps(card_name, level=level)
        result["status"] = "ok"
        result["notes"] = ""
        return result
    except Exception as exc:  # pragma: no cover - safety net
        return {
            "card": card_name,
            "level": level or "",
            "simulated_dps": None,
            "expected_dps": None,
            "delta_pct": None,
            "status": "error",
            "notes": str(exc),
        }


def write_results(rows: List[Dict[str, object]]) -> None:
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    fieldnames = ["card", "level", "status", "expected_dps", "simulated_dps", "delta_pct", "notes"]
    with open(SUMMARY_FILE, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize(rows: List[Dict[str, object]]) -> None:
    valid = [row for row in rows if isinstance(row.get("delta_pct"), (int, float, float))]
    within = sum(1 for row in valid if abs(float(row["delta_pct"])) <= TARGET_DELTA)
    over_warn = sum(1 for row in valid if abs(float(row["delta_pct"])) > WARNING_DELTA)
    deltas = [float(row["delta_pct"]) for row in valid]
    avg = statistics.mean(deltas) if deltas else 0.0

    print(f"{within}/{len(valid)} cards within ±{TARGET_DELTA:.0f}% of sheet DPS")
    print(f"{over_warn} cards deviate by more than {WARNING_DELTA:.0f}%")
    print(f"Average delta {avg:.2f}% ({len(valid)} cards measured)")
    print(f"Detailed results written to {SUMMARY_FILE}")


def main() -> None:
    master_cards = load_master_cards()
    rows: List[Dict[str, object]] = []

    for card in master_cards:
        name = card.get("name")
        ctype = (card.get("type") or "").lower()
        if not name or ctype not in ALLOWED_TYPES:
            continue
        result = evaluate_card(name, level=card.get("level"))
        # pretty formatting
        for key in ("simulated_dps", "expected_dps", "delta_pct"):
            if isinstance(result.get(key), (int, float)):
                result[key] = round(float(result[key]), 2)
        rows.append(result)

    write_results(rows)
    summarize(rows)


if __name__ == "__main__":
    main()
