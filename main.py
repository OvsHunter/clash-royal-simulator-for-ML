"""Project entry point providing a small CLI menu.

Option 1 attempts to run the training pipeline (expects core.train.main).
Option 2 launches the GUI after surfacing the latest two model artifacts so
users can pit them against each other manually inside the arena.
"""
from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List


DEFAULT_MODEL_DIRS: tuple[str, ...] = (
    "runs/",
    "checkpoints/",
    "models/",
    "artifacts/",
)
MODEL_EXTENSIONS: tuple[str, ...] = ("*.pt", "*.pth", "*.ckpt")


def _gather_candidate_models(base_dirs: Iterable[str]) -> List[Path]:
    candidates: List[Path] = []
    seen: set[Path] = set()
    for raw_dir in base_dirs:
        directory = Path(raw_dir)
        if not directory.exists():
            continue
        for pattern in MODEL_EXTENSIONS:
            for path in directory.glob(pattern):
                resolved = path.resolve()
                if resolved not in seen:
                    candidates.append(resolved)
                    seen.add(resolved)
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates


def train_ai() -> None:
    try:
        train_module = importlib.import_module("core.train")
    except ImportError:
        print("[!] Training module core.train is not available yet.")
        print("    Please implement core/train.py with a main() entry point.")
        return

    entry = getattr(train_module, "main", None)
    if callable(entry):
        entry()
    else:
        print("[!] core.train found, but it does not expose a callable main().")


def launch_gui_with_latest_models() -> None:
    models = _gather_candidate_models(DEFAULT_MODEL_DIRS)
    if len(models) < 2:
        print("[!] Could not locate at least two trained model files.")
        print("    Expected to find files with extensions", MODEL_EXTENSIONS)
        print("    inside one of:")
        for path in DEFAULT_MODEL_DIRS:
            print("      -", path)
        print("    Launching the GUI anyway so you can inspect the arena manually.")
        selected = []
    else:
        selected = models[:2]
        print("Launching GUI with the latest models:")
        for idx, path in enumerate(selected, start=1):
            print(f"  Model {idx}: {path}")

    env = os.environ.copy()
    if len(selected) == 2:
        env["CR_MODEL_A"] = str(selected[0])
        env["CR_MODEL_B"] = str(selected[1])

    try:
        subprocess.run([sys.executable, "-m", "gui.gui"], env=env, check=False)
    except FileNotFoundError:
        print("[!] Unable to execute gui/gui.py. Ensure the file exists and dependencies are installed.")


def main() -> None:
    MENU = (
        "\n=== Clash Royale Simulator Launcher ===\n"
        "1) Train AI\n"
        "2) Watch latest trained models in GUI\n"
        "3) Exit\n"
    )

    while True:
        print(MENU, end="")
        try:
            choice = input("Select an option [1-3]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if choice == "1":
            train_ai()
        elif choice == "2":
            launch_gui_with_latest_models()
        elif choice == "3":
            print("Goodbye!")
            break
        else:
            print("Invalid selection. Please choose 1, 2, or 3.")


if __name__ == "__main__":
    main()
