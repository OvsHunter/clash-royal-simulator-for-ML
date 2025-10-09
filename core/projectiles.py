"""Loads projectile definitions from data/projectiles.json."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "projectiles.json"
MASTER_CARDS_PATH = ROOT / "data" / "processed" / "cards_master.json"


def _scale_distance(value):
    if value in (None, False):
        return 0.0
    try:
        val = float(value)
    except (TypeError, ValueError):
        return 0.0
    if abs(val) <= 10.0:
        return val
    return val / 1000.0


def _scale_speed(value):
    if value in (None, False):
        return 0.0
    try:
        val = float(value)
    except (TypeError, ValueError):
        return 0.0
    # Master data stores projectile speed in cm/s; convert to tiles/s (1 tile = 1m).
    if val > 50.0:
        return val / 100.0
    return val


@dataclass(frozen=True)
class ProjectileDef:
    name: str
    speed: float
    damage: float
    hit_radius: float
    area_radius: float
    hits_air: bool
    hits_ground: bool
    pushback: float
    lifetime: float = 0.0
    homing: bool = True

    @property
    def radius(self) -> float:
        # Backwards compatibility
        return self.area_radius

    @property
    def has_aoe(self) -> bool:
        return self.area_radius > 0.0


_CACHE: Dict[str, ProjectileDef] = {}
_LOADED = False


def _load() -> None:
    global _LOADED, _CACHE
    if _LOADED:
        return
    projectiles: Dict[str, ProjectileDef] = {}
    if DATA_PATH.exists():
        with DATA_PATH.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        for name, data in raw.items():
            if not isinstance(data, dict):
                continue
            try:
                speed = float(data.get("speed", 6.0))
                damage = float(data.get("dmg", 0.0))
                radius = float(data.get("radius", 0.0))
                projectiles[name] = ProjectileDef(
                    name=name,
                    speed=speed,
                    damage=damage,
                    hit_radius=radius,
                    area_radius=radius,
                    hits_air=bool(data.get("aoe_to_air", False)),
                    hits_ground=bool(data.get("aoe_to_ground", True)),
                    pushback=float(data.get("pushback", 0.0)),
                    lifetime=float(data.get("lifetime", 0.0)),
                    homing=bool(data.get("homing", True)),
                )
            except (TypeError, ValueError):
                continue
    if MASTER_CARDS_PATH.exists():
        try:
            with MASTER_CARDS_PATH.open("r", encoding="utf-8") as mf:
                master_cards = json.load(mf)
        except (OSError, json.JSONDecodeError):
            master_cards = []
        if isinstance(master_cards, list):
            for card in master_cards:
                stats = card.get("stats") if isinstance(card, dict) else None
                if not isinstance(stats, dict):
                    continue
                pdata = stats.get("projectile_data")
                if not isinstance(pdata, dict):
                    continue
                for proj_name, payload in pdata.items():
                    if not isinstance(payload, dict):
                        continue
                    base = projectiles.get(proj_name)
                    speed = _scale_speed(payload.get("speed", base.speed if base else 0.0))
                    damage = float(payload.get("damage", base.damage if base else 0.0) or 0.0)
                    hit_radius = _scale_distance(payload.get("radius", base.hit_radius if base else 0.0))
                    area_radius = _scale_distance(
                        payload.get("area_damage_radius", payload.get("radius", base.area_radius if base else 0.0))
                    )
                    hits_ground = bool(payload.get("aoe_to_ground", base.hits_ground if base else True))
                    hits_air = bool(payload.get("aoe_to_air", base.hits_air if base else False))
                    pushback = float(payload.get("pushback", base.pushback if base else 0.0) or 0.0)
                    lifetime_raw = payload.get("lifetime")
                    lifetime = 0.0
                    if lifetime_raw:
                        lifetime = float(lifetime_raw) / 1000.0 if float(lifetime_raw) > 10 else float(lifetime_raw)
                    homing = bool(payload.get("homing", base.homing if base else True))
                    projectiles[proj_name] = ProjectileDef(
                        name=proj_name,
                        speed=speed if speed > 0.0 else (base.speed if base else 0.0),
                        damage=damage if damage > 0.0 else (base.damage if base else 0.0),
                        hit_radius=hit_radius if hit_radius > 0.0 else (base.hit_radius if base else 0.0),
                        area_radius=area_radius if area_radius > 0.0 else (base.area_radius if base else 0.0),
                        hits_air=hits_air,
                        hits_ground=hits_ground,
                        pushback=pushback,
                        lifetime=lifetime if lifetime > 0.0 else (base.lifetime if base else 0.0),
                        homing=homing,
                    )
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
