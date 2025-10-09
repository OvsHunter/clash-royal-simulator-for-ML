#!/usr/bin/env python3
"""
auto_retrain_and_update.py

Automatically manages the full data pipeline:
- Detects changes in raw_data or processed files
- Rebuilds cards_master.json if needed
- Synchronizes logic files
- Validates mechanical parity
- Retrains AI policy

Outputs:
- data/processed/cards_master.json (refreshed)
- Updated logic and AI weights
- Console status summary

Dependencies: os, time, subprocess, json, hashlib
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from typing import Dict, Iterable, List, Tuple

STATE_PATH = os.path.join("data", "processed", ".auto_pipeline_state.json")
WATCH_PATHS = [
    "raw_data",
    os.path.join("data", "processed", "cards_master.json"),
    os.path.join("docs", "training_reward_guidelines.md"),
]

PIPELINE_STEPS: List[Tuple[str, List[str]]] = [
    ("Rebuilding master data", ["python", "tools/build_cards_knowledge_base.py"]),
    ("Synchronizing logic files", ["python", "scripts/update_logic_with_master.py"]),
    ("Validating mechanics", ["python", "scripts/validate_simulation_balance.py"]),
    ("Training AI policy", ["python", "scripts/train_ai_from_data.py"]),
]


def compute_digest(paths: Iterable[str]) -> str:
    digest = hashlib.md5()
    for path in paths:
        if not os.path.exists(path):
            continue
        if os.path.isdir(path):
            for root, _, files in os.walk(path):
                for filename in sorted(files):
                    file_path = os.path.join(root, filename)
                    try:
                        stat = os.stat(file_path)
                    except FileNotFoundError:
                        continue
                    digest.update(file_path.encode("utf-8"))
                    digest.update(str(int(stat.st_mtime)).encode("utf-8"))
                    digest.update(str(stat.st_size).encode("utf-8"))
        else:
            stat = os.stat(path)
            digest.update(path.encode("utf-8"))
            digest.update(str(int(stat.st_mtime)).encode("utf-8"))
            digest.update(str(stat.st_size).encode("utf-8"))
    return digest.hexdigest()


def load_state() -> Dict[str, str]:
    if not os.path.exists(STATE_PATH):
        return {}
    with open(STATE_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_state(digest: str) -> None:
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w", encoding="utf-8") as handle:
        json.dump({"digest": digest}, handle, indent=2)


def run_pipeline_step(step_name: str, command: List[str]) -> None:
    print(f"{step_name}...")
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise SystemExit(f"Step failed: {step_name}")


def print_pipeline_summary() -> None:
    print("Pipeline completed successfully")


def main() -> None:
    parser = argparse.ArgumentParser(description="Automated retraining pipeline.")
    parser.add_argument("--watch", action="store_true", help="Continuously watch for changes.")
    parser.add_argument("--interval", type=int, default=60, help="Polling interval in seconds.")
    parser.add_argument("--force", action="store_true", help="Run pipeline regardless of detected changes.")
    args = parser.parse_args()

    def run_once(force: bool) -> None:
        current_digest = compute_digest(WATCH_PATHS)
        previous = load_state().get("digest")
        if not force and previous == current_digest:
            print("No changes detected")
            return

        print("Detected data changes")
        for step_name, command in PIPELINE_STEPS:
            run_pipeline_step(step_name, command)
        print_pipeline_summary()
        save_state(current_digest)

    if args.watch:
        try:
            while True:
                run_once(force=args.force)
                args.force = False
                time.sleep(max(args.interval, 5))
        except KeyboardInterrupt:
            sys.exit(0)
    else:
        run_once(force=args.force)


if __name__ == "__main__":
    main()
