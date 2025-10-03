"""Loads projectile definitions from data/projectiles.json."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "projectiles.json"


@dataclass(frozen=True)
class ProjectileDef:
    name: str
    speed: float
    damage: float
    radius: float
    hits_air: bool
    hits_ground: bool
    pushback: float

    @property
    def has_aoe(self) -> bool:
        return self.radius > 0.0


_CACHE: Dict[str, ProjectileDef] = {}
_LOADED = False


def _load() -> None:
    global _LOADED, _CACHE
    if _LOADED:
        return
    if not DATA_PATH.exists():
        _CACHE = {}
        _LOADED = True
        return
    with DATA_PATH.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    projectiles: Dict[str, ProjectileDef] = {}
    for name, data in raw.items():
        if not isinstance(data, dict):
            continue
        try:
            projectiles[name] = ProjectileDef(
                name=name,
                speed=float(data.get("speed", 6.0)),
                damage=float(data.get("dmg", 0.0)),
                radius=float(data.get("radius", 0.0)),
                hits_air=bool(data.get("aoe_to_air", False)),
                hits_ground=bool(data.get("aoe_to_ground", True)),
                pushback=float(data.get("pushback", 0.0)),
            )
        except (TypeError, ValueError):
            continue
    _CACHE = projectiles
    _LOADED = True


def get_projectile(name: Optional[str]) -> Optional[ProjectileDef]:
    if not name:
        return None
    _load()
    return _CACHE.get(name)


def list_projectiles() -> Dict[str, ProjectileDef]:
    _load()
    return dict(_CACHE)
