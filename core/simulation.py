from __future__ import annotations

"""Core Clash Royale simulation loop: troops, towers, projectiles, spells."""

import copy
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

from .rules import (
    MATCH_TIME,
    DOUBLE_ELIXIR_START,
    TRIPLE_ELIXIR_START,
    DEFAULT_SMALL_RADIUS,
    DEFAULT_POISON_DURATION,
    DEFAULT_POISON_DPS,
    DEFAULT_RAGE_DURATION,
    DEFAULT_RAGE_SPEED_MULTIPLIER,
    DEFAULT_RAGE_LINGER,
    DEFAULT_ZAP_STUN,
    DEFAULT_ARROWS_PROJECTILE,
)
from .pathfinding import compute_path, find_target, world_to_tile, distance_tiles
from .troop_data import get_card
from .projectiles import get_projectile, ProjectileDef

ELIXIR_MAX = 10.0
ELIXIR_REGEN_BASE = 1.0 / 2.8
MIN_HIT_SPEED = 0.2
MELEE_RANGE = 0.5
AGGRO_SIGHT = 6.0

TICK_RATE: float = 15.0
TICK_DT: float = 1.0 / TICK_RATE
DEFAULT_RETARGET_COOLDOWN = 0.3

DEFAULT_TOWER_STATS: Dict[str, Dict[str, float]] = {
    "king": {"hp": 4300.0, "damage": 240.0, "range": 7.0, "hit_speed": 1.0, "projectile_speed": 6.0, "damage_modifier": 1.0},
    "princess": {"hp": 2534.0, "damage": 130.0, "range": 8.0, "hit_speed": 0.8, "projectile_speed": 6.0, "damage_modifier": 1.0},
}


def _normalize_key(value: Any) -> Optional[str]:
    if value is None:
        return None
    return "".join(ch.lower() for ch in str(value) if ch.isalnum())


def _load_master_cards() -> Dict[str, Dict[str, Any]]:
    master_path = Path(__file__).resolve().parents[1] / "data" / "processed" / "cards_master.json"
    if not master_path.exists():
        return {}
    try:
        with master_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    index: Dict[str, Dict[str, Any]] = {}
    if not isinstance(payload, list):
        return index
    for card in payload:
        if not isinstance(card, dict):
            continue
        keys = {
            _normalize_key(card.get("name")),
            _normalize_key(card.get("key")),
            _normalize_key(card.get("id")),
        }
        for key in filter(None, keys):
            existing = index.get(key)
            if existing is None or len(card.keys()) > len(existing.keys()):
                index[key] = card
    return index


MASTER_CARD_INDEX = _load_master_cards()


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _ms_to_seconds(value: Any) -> float:
    if not value:
        return 0.0
    try:
        return float(value) / 1000.0
    except (TypeError, ValueError):
        return 0.0


def _scale_distance(value: Any) -> float:
    if not value:
        return 0.0
    try:
        val = float(value)
    except (TypeError, ValueError):
        return 0.0
    if abs(val) <= 10.0:
        return val
    return val / 1000.0


def _scale_speed(value: Any) -> float:
    if not value:
        return 0.0
    try:
        val = float(value)
    except (TypeError, ValueError):
        return 0.0
    if abs(val) > 50.0:
        return val / 100.0
    return val


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        stripped = value.strip().lower()
        if stripped in {"true", "t", "1", "yes", "y", "on"}:
            return True
        if stripped in {"false", "f", "0", "no", "n", "off"}:
            return False
    return None


def _mechanic_level_value(mechanics: Dict[str, Any], key: str, level: int) -> Any:
    if key not in mechanics:
        return None
    value = mechanics[key]
    if value is None:
        return None
    if isinstance(value, list):
        if not value:
            return None
        idx = max(0, min(len(value) - 1, max(1, int(level)) - 1))
        return value[idx]
    if isinstance(value, dict):
        if not value:
            return None
        str_level = str(level)
        if str_level in value:
            return value[str_level]
        numeric: List[Tuple[int, Any]] = []
        for raw_key, entry in value.items():
            try:
                numeric.append((int(raw_key), entry))
            except (TypeError, ValueError):
                continue
        if numeric:
            numeric.sort()
            for lvl, entry in numeric:
                if lvl >= level:
                    return entry
            return numeric[-1][1]
        return next(iter(value.values()))
    return value


def _derive_target_types(base_flags: Sequence[str], mechanics: Dict[str, Any]) -> Tuple[str, ...]:
    flags: Set[str] = {str(flag).lower() for flag in base_flags if flag}
    if not flags:
        flags = {"ground"}

    if mechanics:
        only_buildings = _coerce_bool(mechanics.get("target_only_buildings"))
        if only_buildings:
            return ("buildings",)
        targets_raw = mechanics.get("targets")
        if targets_raw:
            if isinstance(targets_raw, str):
                targets_seq = [targets_raw]
            else:
                targets_seq = targets_raw
            new_flags: Set[str] = set()
            for entry in targets_seq:
                label = str(entry).lower()
                if label == "both":
                    new_flags.update({"ground", "air"})
                elif label in {"ground", "air", "buildings"}:
                    new_flags.add(label)
            if new_flags:
                flags = new_flags

        only_air = _coerce_bool(mechanics.get("target_only_air"))
        if only_air:
            flags = {"air"}

        attacks_air = _coerce_bool(mechanics.get("attacks_air"))
        if attacks_air is False:
            flags.discard("air")

        attacks_ground = _coerce_bool(mechanics.get("attacks_ground"))
        if attacks_ground is False:
            flags.discard("ground")
            flags.discard("buildings")

    if not flags:
        flags = {"ground"}
    if "buildings" in flags and len(flags) > 1:
        flags.discard("ground")
    ordered = []
    for option in ("ground", "air", "buildings"):
        if option in flags:
            ordered.append(option)
    if not ordered:
        ordered = list(flags)
    return tuple(ordered)

@dataclass
class Unit:
    id: int
    owner: int
    name: str
    x: float
    y: float
    hp: float
    hp_max: float
    alive: bool = True

    speed: float = 1.0
    range: float = 1.0
    hit_speed: float = 1.0
    damage: float = 50.0
    targeting: str = "ground"
    flying: bool = False
    radius: float = 0.45
    aggro_range: float = AGGRO_SIGHT
    level: int = 1
    projectile_speed: float = 0.0
    projectile_radius: float = 0.0
    projectile_hits_air: bool = True
    projectile_hits_ground: bool = True
    projectile_key: Optional[str] = None
    card_key: str = ""
    base_speed: float = 1.0
    splash_radius: float = 0.0
    projectile_hit_radius: float = 0.2
    projectile_area_radius: float = 0.0
    projectile_lifetime: float = 0.0
    charge_windup: float = 0.0
    charge_speed_mult: float = 1.0
    charge_damage_mult: float = 1.0
    charge_reset_on_hit: bool = True
    spawn_effect: Optional[Dict[str, Any]] = None
    slow_effect: Optional[Dict[str, Any]] = None
    chain_config: Optional[Dict[str, Any]] = None
    support_units: Optional[List[Dict[str, Any]]] = None
    death_spawn_config: Optional[Dict[str, Any]] = None
    target_types: Tuple[str, ...] = field(default_factory=lambda: ("ground", "air"))
    sight_range: float = AGGRO_SIGHT
    retarget_cooldown: float = DEFAULT_RETARGET_COOLDOWN
    tower_damage_multiplier: float = 1.0

    _attack_cd: float = 0.0
    active: bool = True
    deploy_cooldown: float = 0.0
    _deploy_complete_fired: bool = False
    _deploy_total_time: float = 0.0
    _windup_timer: float = 0.0
    _first_attack_pending: bool = False
    _load_time: float = 0.0
    _load_after_retarget: bool = False
    _last_target_signature: Optional[str] = None
    _path_refresh_timer: float = 0.0
    _needs_detour: bool = False
    _path: List[Tuple[int, int]] = field(default_factory=list)
    _path_goal_id: Optional[int] = None
    _stun_timer: float = 0.0
    _slow_timer: float = 0.0
    _slow_multiplier: float = 1.0
    _rage_multiplier: float = 1.0
    _rage_timer: float = 0.0
    _rage_linger: float = 0.0
    _charge_progress: float = 0.0
    _charge_ready: bool = False
    _charge_active: bool = False
    _attached_to: Optional["Unit"] = None
    _attach_offset: Tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))
    _death_handled: bool = False
    _support_children: List[int] = field(default_factory=list)
    _retarget_timer: float = 0.0
    _current_target_id: Optional[int] = None
    _current_target_label: Optional[str] = None
    _pending_attack_signature: Optional[str] = None
    _stuck_timer: float = 0.0
    _last_progress_pos: Tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))
    _death_spawn_entries: List[Dict[str, Any]] = field(default_factory=list)

    def center(self) -> Tuple[float, float]:
        return self.x, self.y

    def within_attack_range(self, tx: float, ty: float) -> bool:
        return distance_tiles((self.x, self.y), (tx, ty)) <= self.range + 1e-6

    def within_melee(self, tx: float, ty: float) -> bool:
        return distance_tiles((self.x, self.y), (tx, ty)) <= MELEE_RANGE + 1e-6

    def can_attack_air(self) -> bool:
        return self.targeting in ("air", "both")

    def can_attack_ground(self) -> bool:
        return self.targeting in ("ground", "both", "buildings")


@dataclass
class PendingDeploy:
    '''Action requested by a player that will be resolved on the next tick.'''

    card_key: str
    tile_x: int
    tile_y: int
    level_override: Optional[int] = None
    reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_tuple(self) -> Tuple[str, int, int, Optional[int]]:
        return (self.card_key, self.tile_x, self.tile_y, self.level_override)


@dataclass
class DecisionRecord:
    '''Snapshot of a resolved deployment with optional human-readable context.'''

    time: float
    player: int
    card_key: str
    tile_x: int
    tile_y: int
    level: int
    cost: int
    elixir_before: float
    elixir_after: float
    reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Player:
    idx: int
    elixir: float = 5.0
    crowns: int = 0
    pending_spawns: List[PendingDeploy] = field(default_factory=list)
    card_levels: Dict[str, int] = field(default_factory=dict)

class SpellEffect:
    alive: bool = True

    def tick(self, engine: "Engine", dt: float) -> None:
        raise NotImplementedError


class PoisonCloud(SpellEffect):
    def __init__(self, owner: int, x: float, y: float, radius: float, duration: float, dps: float, label: str, tower_multiplier: float = 1.0) -> None:
        self.owner = owner
        self.x = x
        self.y = y
        self.radius = radius
        self.remaining = duration
        self.dps = dps
        self.label = label
        self.tower_multiplier = tower_multiplier

    def tick(self, engine: "Engine", dt: float) -> None:
        if not self.alive:
            return
        self.remaining -= dt
        damage = self.dps * dt
        if damage > 0.0:
            engine.deal_area(
                self.owner,
                self.x,
                self.y,
                self.radius,
                damage,
                True,
                True,
                ("spell", self.label),
                tower_multiplier=self.tower_multiplier,
            )
        if self.remaining <= 0.0:
            self.alive = False


class RageZone(SpellEffect):
    def __init__(self, owner: int, x: float, y: float, radius: float, duration: float, multiplier: float, linger: float, label: str) -> None:
        self.owner = owner
        self.x = x
        self.y = y
        self.radius = radius
        self.remaining = duration
        self.multiplier = multiplier
        self.linger = linger
        self.label = label

    def tick(self, engine: "Engine", dt: float) -> None:
        if not self.alive:
            return
        self.remaining -= dt
        for unit in engine.units:
            if not unit.alive or unit.owner != self.owner:
                continue
            if distance_tiles((unit.x, unit.y), (self.x, self.y)) <= self.radius + unit.radius:
                unit.apply_rage(self.multiplier, dt + self.linger, self.linger)
        if self.remaining <= 0.0:
            self.alive = False


class Projectile:
    __slots__ = (
        "owner",
        "x",
        "y",
        "z",
        "speed",
        "damage",
        "hit_radius",
        "area_radius",
        "hits_air",
        "hits_ground",
        "source",
        "target_unit",
        "target_structure",
        "target_pos",
        "pushback",
        "alive",
        "max_lifetime",
        "age",
        "homing",
    )

    def __init__(
        self,
        owner: int,
        x: float,
        y: float,
        speed: float,
        damage: float,
        hit_radius: float = 0.25,
        area_radius: float = 0.0,
        hits_air: bool = True,
        hits_ground: bool = True,
        source: Tuple[str, str] = ("unit", ""),
        target_unit: Optional[Unit] = None,
        target_structure: Optional[Dict[str, Any]] = None,
        target_pos: Optional[Tuple[float, float]] = None,
        pushback: float = 0.0,
        z: float = 0.0,
        lifetime: float = 0.0,
        homing: bool = True,
    ) -> None:
        self.owner = owner
        self.x = x
        self.y = y
        self.z = z
        self.speed = max(0.01, speed)
        self.damage = damage
        self.hit_radius = max(0.0, hit_radius)
        self.area_radius = max(0.0, area_radius)
        self.hits_air = hits_air
        self.hits_ground = hits_ground
        self.source = source
        self.target_unit = target_unit
        self.target_structure = target_structure
        self.target_pos = target_pos
        self.pushback = pushback
        self.alive = True
        self.max_lifetime = max(0.0, lifetime)
        self.age = 0.0
        self.homing = homing

    @property
    def radius(self) -> float:
        return self.area_radius

    def tick(self, engine: "Engine", dt: float) -> None:
        if not self.alive:
            return
        self.age += dt
        if self.max_lifetime > 0.0 and self.age >= self.max_lifetime:
            self._on_hit(engine, None)
            self.alive = False
            return

        point, direct_target, target_radius = self._target_state()
        if point is None:
            self._on_hit(engine, None)
            self.alive = False
            return

        dx = point[0] - self.x
        dy = point[1] - self.y
        dist = math.hypot(dx, dy)
        collision_threshold = max(self.hit_radius + target_radius, self.hit_radius, 1e-6)
        step = self.speed * dt

        if dist <= collision_threshold or dist <= step:
            self.x, self.y = point
            self._on_hit(engine, direct_target)
            self.alive = False
            return

        if dist > 1e-6:
            self.x += dx / dist * step
            self.y += dy / dist * step
        else:
            self.x, self.y = point
            self._on_hit(engine, direct_target)
            self.alive = False

    def _target_state(self) -> Tuple[Optional[Tuple[float, float]], Optional[Union[Unit, Dict[str, Any]]], float]:
        if self.target_unit is not None and not self.target_unit.alive:
            self.target_unit = None
        if self.target_structure is not None and not self.target_structure.get("alive", True):
            self.target_structure = None

        if self.target_unit is not None and self.target_unit.alive:
            return (self.target_unit.x, self.target_unit.y), self.target_unit, getattr(self.target_unit, "radius", 0.4)

        if self.target_structure is not None and self.target_structure.get("alive", True):
            cx = self.target_structure.get("cx")
            cy = self.target_structure.get("cy")
            if cx is None or cy is None:
                cx, cy = self._structure_center(self.target_structure)
            radius = float(self.target_structure.get("radius", 0.5) or 0.5)
            return (float(cx), float(cy)), self.target_structure, radius

        if self.target_pos is not None:
            return self.target_pos, None, 0.0

        return None, None, 0.0

    def _structure_center(self, structure: Dict[str, Any]) -> Tuple[float, float]:
        x0 = float(structure.get("x0", 0.0))
        y0 = float(structure.get("y0", 0.0))
        w = float(structure.get("width", 1.0))
        h = float(structure.get("height", 1.0))
        return (x0 + w / 2.0, y0 + h / 2.0)

    def _on_hit(self, engine: "Engine", direct_target: Optional[Union[Unit, Dict[str, Any]]]) -> None:
        if self.damage <= 0.0:
            return
        if self.area_radius > 0.0:
            engine.deal_area(self.owner, self.x, self.y, self.area_radius, self.damage, self.hits_air, self.hits_ground, self.source)
            return
        if isinstance(direct_target, Unit) and direct_target.alive:
            if (direct_target.flying and self.hits_air) or ((not direct_target.flying) and self.hits_ground):
                direct_target.take_damage(self.damage, self.owner, self.source)
            return
        if isinstance(direct_target, dict) and direct_target.get("alive", True):
            engine.damage_structure(direct_target, self.damage, self.owner, self.source)
            return
        engine.deal_area(
            self.owner,
            self.x,
            self.y,
            max(self.hit_radius, DEFAULT_SMALL_RADIUS),
            self.damage,
            self.hits_air,
            self.hits_ground,
            self.source,
        )

def _unit_apply_stun(self: Unit, duration: float) -> None:
    if duration <= 0.0:
        return
    if getattr(self, "targeting", "ground") == "buildings":
        return
    self._stun_timer = max(getattr(self, '_stun_timer', 0.0), duration)
    self._attack_cd = max(self._attack_cd, duration)


def _unit_apply_slow(self: Unit, multiplier: float, duration: float) -> None:
    if duration <= 0.0:
        return
    if getattr(self, "targeting", "ground") == "buildings":
        return
    try:
        mult = float(multiplier)
    except (TypeError, ValueError):
        mult = 1.0
    mult = max(0.1, mult)
    self._slow_multiplier = min(getattr(self, '_slow_multiplier', 1.0), mult)
    self._slow_timer = max(getattr(self, '_slow_timer', 0.0), float(duration))


def _unit_apply_rage(self: Unit, multiplier: float, duration: float, linger: float) -> None:
    self._rage_multiplier = max(getattr(self, '_rage_multiplier', 1.0), multiplier)
    self._rage_timer = max(getattr(self, '_rage_timer', 0.0), duration)
    self._rage_linger = max(getattr(self, '_rage_linger', 0.0), linger)


def _unit_take_damage(self: Unit, amount: float, source_owner: Optional[int], source: Tuple[str, str]) -> None:
    if not self.alive:
        return
    self.hp -= amount
    if self.hp <= 0.0 and self.alive:
        self.alive = False


Unit.apply_stun = _unit_apply_stun
Unit.apply_slow = _unit_apply_slow
Unit.apply_rage = _unit_apply_rage
Unit.take_damage = _unit_take_damage


class Engine:
    """Core battle loop for the arena."""

    def __init__(self, arena, seed: Optional[int] = None) -> None:
        self.arena = arena
        self.time = 0.0
        self.over = False
        self.winner: Optional[int] = None

        self.units: List[Unit] = []
        self.projectiles: List[Projectile] = []
        self.effects: List[SpellEffect] = []
        self._unit_id_seq = 1

        self.p1 = Player(1)
        self.p2 = Player(2)
        self.players: Dict[int, Player] = {1: self.p1, 2: self.p2}

        self.decision_log: List[DecisionRecord] = []
        self._recent_decisions: List[DecisionRecord] = []

        self.event_log: List[Tuple[str, Dict[str, Any]]] = []
        self._recent_events: List[Tuple[str, Dict[str, Any]]] = []
        self._event_listeners: Dict[str, List[Callable[[Dict[str, Any]], None]]] = defaultdict(list)

        self.towers = self.arena.towers
        self.tick_duration = TICK_DT
        self.dt = self.tick_duration
        if seed is not None:
            random.seed(seed)

        self._prepare_towers()
        self.tower_level_p1 = 11
        self.tower_level_p2 = 11
        self.set_tower_levels(11, 11)
        self._river_band = self.arena.river_band()
    def tick(self) -> None:
        if self.over:
            return
        self._regen_elixir()
        self._consume_pending_spawns(self.p1)
        self._consume_pending_spawns(self.p2)
        self._update_towers()
        self._update_units()
        self._update_projectiles()
        self._update_effects()
        self.time += self.dt
        if self.time >= MATCH_TIME:
            self._finalize_match()

    def deploy(
        self,
        player_idx: int,
        card_key: str,
        tile_x: int,
        tile_y: int,
        *,
        level: Optional[int] = None,
        reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Queue a deployment request for the next simulation tick."""

        if player_idx not in self.players:
            raise ValueError(f'invalid player index: {player_idx}')

        pending = self.players[player_idx].pending_spawns
        entry = PendingDeploy(
            card_key=card_key,
            tile_x=tile_x,
            tile_y=tile_y,
            level_override=level,
            reason=reason,
            metadata=dict(metadata) if metadata else {},
        )
        pending.append(entry)
        return True

    def list_units(self) -> List[Unit]:
        return [u for u in self.units if u.alive]

    # ------------------------------------------------------------------
    # Decision history helpers
    # ------------------------------------------------------------------

    def poll_decisions(self, *, clear: bool = True) -> List[DecisionRecord]:
        '''Return decisions resolved since the last poll.'''

        recent = list(self._recent_decisions)
        if clear:
            self._recent_decisions.clear()
        return recent

    def clear_decision_log(self) -> None:
        '''Remove all stored decision history.'''

        self.decision_log.clear()
        self._recent_decisions.clear()

    def _push_decision(self, record: DecisionRecord) -> None:
        self.decision_log.append(record)
        self._recent_decisions.append(record)

    # ------------------------------------------------------------------
    # Event helpers
    # ------------------------------------------------------------------

    def add_event_listener(self, event_name: str, callback: Callable[[Dict[str, Any]], None]) -> None:
        self._event_listeners[event_name].append(callback)

    def remove_event_listener(self, event_name: str, callback: Callable[[Dict[str, Any]], None]) -> None:
        listeners = self._event_listeners.get(event_name)
        if not listeners:
            return
        try:
            listeners.remove(callback)
        except ValueError:
            return
        if not listeners:
            self._event_listeners.pop(event_name, None)

    def poll_events(self, *, clear: bool = True) -> List[Tuple[str, Dict[str, Any]]]:
        events = list(self._recent_events)
        if clear:
            self._recent_events.clear()
        return events

    def clear_event_log(self) -> None:
        self.event_log.clear()
        self._recent_events.clear()

    def _dispatch_event(self, event_name: str, payload: Dict[str, Any]) -> None:
        event_payload = dict(payload)
        event_payload.setdefault("time", self.time)
        record = (event_name, event_payload)
        self.event_log.append(record)
        self._recent_events.append(record)
        for callback in tuple(self._event_listeners.get(event_name, ())):
            try:
                callback(dict(event_payload))
            except Exception:
                continue

    # ------------------------------------------------------------------
    # Decision context helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_card_role(card: Dict[str, Any]) -> str:
        raw_role = str(card.get('card_type', 'troop')).strip().lower()
        if raw_role in ('structure', 'tower', 'building', 'defense', 'defence'):
            return 'building'
        if raw_role in ('troop', 'unit', 'character'):
            return 'troop'
        if raw_role == 'spell':
            return 'spell'
        if raw_role == 'trap':
            return 'trap'
        return raw_role or 'troop'

    @staticmethod
    def _normalize_targets_metadata(card: Dict[str, Any]) -> List[str]:
        targets = card.get('targets', [])
        if isinstance(targets, str):
            targets_list = [t.strip().lower() for t in targets.split(',') if t.strip()]
        else:
            targets_list = [str(t).strip().lower() for t in targets]
        normalized: List[str] = []
        for target in targets_list:
            if target == 'both':
                normalized.extend(['ground', 'air'])
            elif target in ('ground', 'air', 'buildings'):
                normalized.append(target)
        return normalized or ['ground', 'air']

    def _card_can_target(self, card: Dict[str, Any], unit: Unit, role: str) -> bool:
        if role == 'spell':
            return True
        targets = self._normalize_targets_metadata(card)
        if unit.flying:
            return 'air' in targets
        return 'ground' in targets or 'buildings' in targets

    def _collect_interactions(
        self,
        owner: int,
        card: Dict[str, Any],
        tile_x: int,
        tile_y: int,
        role: str,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        origin_x = tile_x + 0.5
        origin_y = tile_y + 0.5
        detection_radius = max(AGGRO_SIGHT, float(card.get('range', 0.0)) + 6.0)
        enemy: List[Dict[str, Any]] = []
        allies: List[Dict[str, Any]] = []
        for unit in self.units:
            if not unit.alive:
                continue
            distance = math.hypot(unit.x - origin_x, unit.y - origin_y)
            if distance > detection_radius + unit.radius:
                continue
            entry = {
                'unit_id': unit.id,
                'card_key': unit.card_key or unit.name,
                'name': unit.name,
                'distance': round(distance, 3),
                'hp': round(unit.hp, 2),
                'hp_max': round(unit.hp_max, 2),
                'flying': bool(unit.flying),
                'targeting': unit.targeting,
            }
            if unit.owner == owner:
                allies.append(entry)
            else:
                engageable = self._card_can_target(card, unit, role)
                entry.update(
                    {
                        'engageable': engageable,
                        'threat_type': 'air' if unit.flying else 'ground',
                    }
                )
                enemy.append(entry)
        enemy.sort(key=lambda ent: (not ent.get('engageable', False), ent['distance']))
        allies.sort(key=lambda ent: ent['distance'])
        return enemy, allies

    def _compose_auto_reason(
        self,
        card_key: str,
        role: str,
        tile_x: int,
        tile_y: int,
        threats: List[Dict[str, Any]],
        supports: List[Dict[str, Any]],
    ) -> str:
        location = f'({tile_x}, {tile_y})'
        subject = f'{card_key} ({role})'
        if threats:
            engaged = [t for t in threats if t.get('engageable')]
            focus = engaged or threats
            names = [t.get('card_key') or t.get('name') or 'enemy' for t in focus][:3]
            targets = ', '.join(names)
            return f'Deployed {subject} at {location} to respond to {targets}'
        if supports:
            names = [s.get('card_key') or s.get('name') or 'ally' for s in supports][:3]
            allies = ', '.join(names)
            return f'Deployed {subject} at {location} to support {allies}'
        return f'Deployed {subject} at {location}'

    def _build_decision_context(
        self,
        owner: int,
        card_key: str,
        card: Dict[str, Any],
        tile_x: int,
        tile_y: int,
        base_metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        metadata = dict(base_metadata) if base_metadata else {}
        role = self._normalize_card_role(card)
        metadata.setdefault('card_role', role)
        metadata.setdefault('card_name', card.get('name', card_key))
        metadata.setdefault('card_targets', self._normalize_targets_metadata(card))
        metadata.setdefault('flying', bool(card.get('flying', False)))
        metadata.setdefault('range', float(card.get('range', 0.0)))
        metadata.setdefault('speed', float(card.get('speed', 0.0)))
        threats, supports = self._collect_interactions(owner, card, tile_x, tile_y, role)
        metadata['interactions'] = {
            'threats': threats,
            'supports': supports,
        }
        metadata.setdefault('detection_radius', max(AGGRO_SIGHT, float(card.get('range', 0.0)) + 4.0))
        reason = self._compose_auto_reason(card_key, role, tile_x, tile_y, threats, supports)
        return reason, metadata

    def _prepare_towers(self) -> None:
        for tower in self.towers:
            label = tower.get("label", "")
            owner = tower.get("owner")
            if owner not in (1, 2):
                owner = 1 if label.startswith("p1_") else 2
            tower["owner"] = owner
            tower_type = "king" if "king" in label else "princess"
            defaults = DEFAULT_TOWER_STATS.get(tower_type, DEFAULT_TOWER_STATS["princess"])
            tower["hp"] = float(tower.get("hp", defaults["hp"]))
            tower["hp_max"] = float(tower.get("hp_max", tower["hp"]))
            damage = float(tower.get("damage", tower.get("dmg", defaults["damage"])))
            tower["damage"] = damage
            tower["dmg"] = damage
            tower["range"] = float(tower.get("range", defaults["range"]))
            tower["hit_speed"] = max(float(tower.get("hit_speed", defaults["hit_speed"])), MIN_HIT_SPEED)
            tower["alive"] = bool(tower.get("alive", True)) and tower["hp"] > 0
            tower["projectile_speed"] = float(tower.get("projectile_speed", defaults.get("projectile_speed", 6.0)))
            tower["damage_modifier"] = float(tower.get("damage_modifier", tower.get("crown_damage_modifier", defaults.get("damage_modifier", 1.0))))
            tower["cx"] = tower["x0"] + tower["width"] / 2.0
            tower["cy"] = tower["y0"] + tower["height"] / 2.0
            tower["radius"] = max(tower["width"], tower["height"]) * 0.5
            tower["cooldown"] = 0.0
            tower["stun_timer"] = 0.0
            tower["slow_timer"] = 0.0
            tower["slow_multiplier"] = 1.0
            tower["target"] = None
            tower["type"] = tower_type
            tower.setdefault("active", tower_type != "king")
            tower["base_hp"] = tower["hp"]
            tower["base_damage"] = damage
            tower["level"] = 11
            self._initialize_tower_combat_state(tower)

    def _initialize_tower_combat_state(self, tower: Dict[str, Any]) -> None:
        tower_type = tower.get("type", "princess")
        owner = tower.get("owner", 1)
        level = tower.get("level", 11)
        card_name = "Princess" if tower_type == "princess" else None
        if card_name:
            card = get_card(card_name, level=level)
            master_card = self._lookup_master_card(card_name, card) if card else None
            mechanics = (master_card or {}).get("stats", {})
            spec = self._extract_unit_stats(card, card_name) if card else {}
            projectile_key = card.get("projectile") if card else None
            projectile_speed = spec.get("projectile_speed", tower.get("projectile_speed", 6.0))
            projectile_hit_radius = spec.get("projectile_hit_radius", 0.25)
            projectile_area_radius = spec.get("projectile_area_radius", 0.0)
            target_types = _derive_target_types(spec.get("target_flags", []), mechanics)
            sight_range = _scale_distance(mechanics.get("sight_range")) or spec.get("aggro", tower.get("range", 7.0))
            load_time = _ms_to_seconds(mechanics.get("load_time")) if mechanics.get("load_time") else 0.0
            load_first_hit = bool(mechanics.get("load_first_hit")) and load_time > 0.0
            load_after_retarget = bool(mechanics.get("load_after_retarget"))
            retarget_raw = mechanics.get("retarget_cooldown", mechanics.get("retarget_time"))
            retarget_cooldown = _coerce_float(retarget_raw)
            if retarget_cooldown and retarget_cooldown > 10.0:
                retarget_cooldown = _ms_to_seconds(retarget_cooldown)
        else:
            projectile_key = None
            projectile_speed = tower.get("projectile_speed", 6.0)
            projectile_hit_radius = 0.3
            projectile_area_radius = 0.0
            target_types = ("ground", "air")
            sight_range = tower.get("range", 7.0)
            load_time = 0.0
            load_first_hit = False
            load_after_retarget = False
            retarget_cooldown = None

        tower_identity = int(tower.get("id", 0) or 0)
        if tower_identity == 0:
            tower_identity = abs(hash(tower.get("label", "tower"))) % 1000000 + 1000
        unit = Unit(
            id=-tower_identity,
            owner=owner,
            name=tower.get("label", f"{tower_type}_tower"),
            x=float(tower.get("cx", 0.0)),
            y=float(tower.get("cy", 0.0)),
            hp=float(tower.get("hp", 0.0)),
            hp_max=float(tower.get("hp_max", tower.get("hp", 0.0))),
            alive=bool(tower.get("alive", True)),
            speed=0.0,
            range=float(tower.get("range", DEFAULT_TOWER_STATS.get(tower_type, DEFAULT_TOWER_STATS["princess"])["range"])),
            hit_speed=float(tower.get("hit_speed", DEFAULT_TOWER_STATS.get(tower_type, DEFAULT_TOWER_STATS["princess"])["hit_speed"])),
            damage=float(tower.get("damage", DEFAULT_TOWER_STATS.get(tower_type, DEFAULT_TOWER_STATS["princess"])["damage"])),
            targeting="both",
            flying=False,
            radius=max(float(tower.get("radius", 0.5)), 0.5),
            aggro_range=float(sight_range),
            level=level,
            projectile_speed=float(projectile_speed),
            projectile_radius=float(projectile_area_radius),
            projectile_hit_radius=float(projectile_hit_radius),
            projectile_area_radius=float(projectile_area_radius),
            projectile_hits_air="air" in target_types,
            projectile_hits_ground=any(flag in ("ground", "buildings") for flag in target_types),
            projectile_key=projectile_key,
            card_key=tower.get("label", f"{tower_type}_tower"),
            base_speed=0.0,
            splash_radius=0.0,
        )
        unit.target_types = target_types if target_types else ("ground",)
        unit.sight_range = float(sight_range or unit.range)
        unit.retarget_cooldown = retarget_cooldown if retarget_cooldown is not None else DEFAULT_RETARGET_COOLDOWN
        unit._load_time = load_time
        unit._first_attack_pending = load_first_hit
        unit._load_after_retarget = load_after_retarget
        unit._windup_timer = 0.0
        unit._attack_cd = 0.0
        unit._retarget_timer = 0.0
        unit._deploy_complete_fired = True
        unit.deploy_cooldown = 0.0
        unit.active = bool(tower.get("active", tower_type != "king"))
        unit.tower_damage_multiplier = float(tower.get("damage_modifier", 1.0))
        unit._projectile_offset = 0.0
        unit._projectile_start_z = 0.0
        unit._last_progress_pos = (unit.x, unit.y)

        tower["_unit"] = unit
        tower["_load_time"] = load_time
        tower["_load_after_retarget"] = load_after_retarget
        tower["_first_attack_pending"] = load_first_hit
        tower["_retarget_cooldown"] = unit.retarget_cooldown
        tower["_sight_range"] = unit.sight_range
        tower["_projectile_key"] = projectile_key
        tower["target_types"] = unit.target_types
        tower["target"] = None
        tower["hits_air"] = "air" in unit.target_types
        tower["hits_ground"] = any(flag in ("ground", "buildings") for flag in unit.target_types)
        tower["projectile_hit_radius"] = projectile_hit_radius
        tower["projectile_area_radius"] = projectile_area_radius
        self._sync_tower_unit_stats(tower)

    def _sync_tower_unit_stats(self, tower: Dict[str, Any]) -> None:
        unit = tower.get("_unit")
        if not unit:
            return
        unit.hp_max = float(tower.get("hp_max", unit.hp_max))
        unit.hp = float(tower.get("hp", unit.hp))
        unit.damage = float(tower.get("damage", unit.damage))
        unit.hit_speed = float(tower.get("hit_speed", unit.hit_speed))
        unit.range = float(tower.get("range", unit.range))
        unit.aggro_range = max(unit.range, float(tower.get("_sight_range", unit.sight_range)))
        unit.sight_range = float(tower.get("_sight_range", unit.aggro_range))
        unit.projectile_speed = float(tower.get("projectile_speed", unit.projectile_speed))
        unit.projectile_key = tower.get("_projectile_key", unit.projectile_key)
        unit.projectile_hit_radius = float(tower.get("projectile_hit_radius", unit.projectile_hit_radius))
        unit.projectile_area_radius = float(tower.get("projectile_area_radius", unit.projectile_area_radius))
        unit.level = int(tower.get("level", unit.level))
        unit.tower_damage_multiplier = float(tower.get("damage_modifier", unit.tower_damage_multiplier))
        unit.active = bool(tower.get("active", unit.active))
        unit.alive = bool(tower.get("alive", unit.alive)) and unit.hp > 0.0
        tower["hits_air"] = "air" in unit.target_types
        tower["hits_ground"] = any(flag in ("ground", "buildings") for flag in unit.target_types)

    def _sync_tower_unit_state(self, tower: Dict[str, Any]) -> None:
        unit = tower.get("_unit")
        if not unit:
            return
        unit.hp = float(tower.get("hp", unit.hp))
        unit.hp_max = float(tower.get("hp_max", unit.hp_max))
        unit.alive = bool(tower.get("alive", True)) and unit.hp > 0.0
        unit.active = bool(tower.get("active", True))
        unit.x = float(tower.get("cx", unit.x))
        unit.y = float(tower.get("cy", unit.y))
    @staticmethod
    def _tower_level_multiplier(level: int, base_level: int = 11) -> float:
        lvl = Engine._coerce_tower_level(level, base_level)
        return 1.035 ** (lvl - base_level)

    @staticmethod
    def _coerce_tower_level(value: Optional[int], default: int = 11) -> int:
        try:
            lvl = int(value)
        except (TypeError, ValueError):
            lvl = default
        return max(1, min(15, lvl))

    def _apply_tower_level(self, tower: Dict[str, Any], level: int) -> None:
        base_hp = float(tower.get("base_hp", tower.get("hp_max", tower.get("hp", 0.0))))
        base_damage = float(tower.get("base_damage", tower.get("damage", 0.0)))
        prev_max = float(tower.get("hp_max", base_hp)) or base_hp or 1.0
        prev_frac = tower.get("hp", prev_max) / prev_max if prev_max > 0 else 1.0
        prev_frac = max(0.0, min(1.0, prev_frac))
        factor = self._tower_level_multiplier(level)
        new_hp_max = base_hp * factor
        tower["hp_max"] = new_hp_max
        tower["hp"] = max(0.0, min(new_hp_max, new_hp_max * prev_frac))
        new_damage = base_damage * factor
        tower["damage"] = new_damage
        tower["dmg"] = new_damage
        tower["level"] = Engine._coerce_tower_level(level)
        self._sync_tower_unit_stats(tower)

    def set_tower_levels(self, p1_level: int, p2_level: int) -> None:
        self.tower_level_p1 = self._coerce_tower_level(p1_level, 11)
        self.tower_level_p2 = self._coerce_tower_level(p2_level, 11)
        for tower in self.towers:
            owner = tower.get("owner", 1)
            level = self.tower_level_p1 if owner == 1 else self.tower_level_p2
            self._apply_tower_level(tower, level)

    def _regen_elixir(self) -> None:
        # Double/triple elixir windows accelerate regen pace.
        multiplier = 1.0
        if self.time >= TRIPLE_ELIXIR_START:
            multiplier = 3.0
        elif self.time >= DOUBLE_ELIXIR_START:
            multiplier = 2.0

        gain = ELIXIR_REGEN_BASE * multiplier * self.dt
        if gain <= 0.0:
            return
        for player in (self.p1, self.p2):
            player.elixir = clamp(player.elixir + gain, 0.0, ELIXIR_MAX)

    def _consume_pending_spawns(self, player: Player) -> None:
        if not player.pending_spawns:
            return
        kept: List[PendingDeploy] = []
        for entry in player.pending_spawns:
            if isinstance(entry, PendingDeploy):
                pending = entry
            else:
                card_key, tx, ty, *maybe_level = entry
                level_override = maybe_level[0] if maybe_level else None
                pending = PendingDeploy(card_key, tx, ty, level_override)
            card_key = pending.card_key
            tx, ty = pending.tile_x, pending.tile_y

            if not self.arena.is_deploy_legal(player.idx, tx, ty):
                continue

            desired_level = (
                pending.level_override
                if pending.level_override is not None
                else player.card_levels.get(card_key)
            )
            card = get_card(card_key, level=desired_level)
            if not card:
                continue

            cost = int(card.get('cost', 0))
            actual_level = int(card.get('level', desired_level or 1))
            if player.elixir + 1e-6 < cost:
                pending.level_override = actual_level
                kept.append(pending)
                continue

            auto_reason, metadata = self._build_decision_context(
                player.idx,
                card_key,
                card,
                tx,
                ty,
                pending.metadata,
            )
            reason_text = pending.reason or auto_reason

            elixir_before = player.elixir
            master_card = self._lookup_master_card(card_key, card)
            card_type = (card.get("type") or card.get("card_type") or "troop").lower()
            if card_type == "spell":
                if not self._cast_spell(player.idx, card_key, card, tx, ty, actual_level, master_card):
                    continue
            else:
                self._spawn_card(player.idx, card_key, card, tx, ty, actual_level, master_card=master_card)
            player.elixir -= cost
            player.card_levels[card_key] = actual_level

            record = DecisionRecord(
                time=self.time,
                player=player.idx,
                card_key=card_key,
                tile_x=tx,
                tile_y=ty,
                level=actual_level,
                cost=cost,
                elixir_before=elixir_before,
                elixir_after=player.elixir,
                reason=reason_text,
                metadata=metadata,
            )
            self._push_decision(record)
        player.pending_spawns = kept

    def _lookup_master_card(self, card_key: str, card: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not MASTER_CARD_INDEX:
            return None
        keys = {
            _normalize_key(card_key),
            _normalize_key(card.get("name")),
            _normalize_key(card.get("display_name")),
            _normalize_key(card.get("key")),
            _normalize_key(card.get("id")),
        }
        for key in filter(None, keys):
            master = MASTER_CARD_INDEX.get(key)
            if master:
                return master
        return None


    def _spawn_card(
        self,
        owner: int,
        card_key: str,
        card: Dict[str, Any],
        tx: int,
        ty: int,
        level: int,
        master_card: Optional[Dict[str, Any]] = None,
    ) -> None:
        name = (
            card.get("display_name")
            or card.get("name")
            or card.get("key")
            or card_key
        )
        spec = self._extract_unit_stats(card, name)
        count = int(card.get("count", 1))
        offsets = self._generate_spawn_offsets(count)
        placed = 0
        for dx, dy in offsets:
            if placed >= count:
                break
            px = tx + 0.5 + dx
            py = ty + 0.5 + dy
            gx = int(round(px - 0.5))
            gy = int(round(py - 0.5))
            if not self.arena.is_walkable(owner, gx, gy, flying=spec["flying"]):
                continue
            if self._collides_with_tower(gx, gy) or self._collides_with_units(px, py, spec["radius"], owner, spec["flying"]):
                continue
            unit = self._create_unit(
                owner=owner,
                card_key=card_key,
                name=name,
                level=level,
                card=card,
                spec=spec,
                x=px,
                y=py,
                master_card=master_card,
            )
            if unit is None:
                continue
            unit._attach_offset = (dx, dy)
            self.units.append(unit)
            self._unit_id_seq += 1
            self._on_unit_spawn(unit, card)
            self._initialize_deploy_state(unit)
            placed += 1

    def _cast_spell(
        self,
        owner: int,
        card_key: str,
        card: Dict[str, Any],
        tx: int,
        ty: int,
        level: int,
        master_card: Optional[Dict[str, Any]],
    ) -> bool:
        mechanics = (master_card or {}).get("stats", {})
        level = max(1, int(level))

        def _resolve_level_value(value: Any) -> Any:
            if value is None:
                return None
            if isinstance(value, list):
                if not value:
                    return None
                idx = max(0, min(len(value) - 1, level - 1))
                return value[idx]
            if isinstance(value, dict):
                if str(level) in value:
                    return value[str(level)]
                numeric_keys = [k for k in value.keys() if str(k).isdigit()]
                if numeric_keys:
                    numeric_keys.sort(key=lambda k: int(k))
                    for key in numeric_keys:
                        if int(key) >= level:
                            return value[key]
                    return value[numeric_keys[-1]]
                return next(iter(value.values()), None)
            return value

        def _stat_value(key: str) -> Any:
            val = _mechanic_level_value(mechanics, key, level)
            if val is None:
                val = mechanics.get(key)
            return _resolve_level_value(val)

        def _stat_distance(key: str) -> Optional[float]:
            value = _stat_value(key)
            if value is None:
                return None
            return _scale_distance(value)

        def _stat_seconds(key: str) -> Optional[float]:
            value = _stat_value(key)
            if value is None:
                return None
            try:
                val = float(value)
            except (TypeError, ValueError):
                return None
            if abs(val) > 10.0:
                return _ms_to_seconds(val)
            return val

        radius_candidates = [
            _stat_distance("area_damage_radius"),
            _stat_distance("radius"),
            _stat_distance("summon_radius"),
        ]
        card_radius = card.get("radius")
        if card_radius:
            radius_candidates.append(float(card_radius))
        radius = max([r for r in radius_candidates if r and r > 0.0], default=0.0)

        damage_val = _stat_value("instant_damage")
        if damage_val is None:
            damage_val = _stat_value("damage")
        if damage_val is None:
            damage_val = card.get("dmg", card.get("damage", 0.0))
        try:
            damage = float(damage_val or 0.0)
        except (TypeError, ValueError):
            damage = 0.0

        duration = _stat_seconds("life_duration")
        if duration is None:
            duration = _stat_seconds("duration")
        if duration is None:
            raw_duration = card.get("duration")
            duration = float(raw_duration or 0.0)

        stun_duration = _stat_seconds("stun_time")
        if stun_duration is None:
            stun_duration = _stat_seconds("stun_duration")
        if stun_duration is None:
            raw_stun = card.get("stun_duration")
            stun_duration = float(raw_stun or 0.0)

        buff_data = mechanics.get("buff_data")
        tick_dps = 0.0
        dot_tower_multiplier = 1.0
        if isinstance(buff_data, dict):
            dps_val = _resolve_level_value(buff_data.get("damage_per_second"))
            if dps_val is None:
                dps_val = _resolve_level_value(buff_data.get("damage"))
            if dps_val is not None:
                try:
                    tick_dps = float(dps_val)
                except (TypeError, ValueError):
                    tick_dps = 0.0
            tower_pct = _resolve_level_value(buff_data.get("crown_tower_damage_percent"))
            if tower_pct is not None:
                try:
                    dot_tower_multiplier = (100.0 + float(tower_pct)) / 100.0
                except (TypeError, ValueError):
                    dot_tower_multiplier = 1.0
            if stun_duration <= 0.0:
                buff_time = _resolve_level_value(buff_data.get("buff_time"))
                if buff_time:
                    stun_duration = _ms_to_seconds(float(buff_time)) if float(buff_time) > 10.0 else float(buff_time)

        if tick_dps <= 0.0:
            raw_tick = card.get("tick_dps", card.get("damage_per_second", 0.0))
            tick_dps = float(raw_tick or 0.0)

        tower_pct_direct = _stat_value("crown_tower_damage_percent")
        if tower_pct_direct is None and isinstance(buff_data, dict):
            tower_pct_direct = _resolve_level_value(buff_data.get("crown_tower_damage_percent"))
        if tower_pct_direct is None:
            tower_pct_direct = card.get("crown_tower_damage_percent")
        try:
            tower_multiplier = (100.0 + float(tower_pct_direct)) / 100.0 if tower_pct_direct is not None else 1.0
        except (TypeError, ValueError):
            tower_multiplier = 1.0

        speed_mult_val = _stat_value("speed_multiplier")
        if speed_mult_val is None and isinstance(buff_data, dict):
            speed_mult_val = _resolve_level_value(buff_data.get("speed_multiplier"))
        if speed_mult_val is None:
            speed_mult_val = card.get("speed_multiplier")
        try:
            raw_speed_mult = float(speed_mult_val or 0.0)
        except (TypeError, ValueError):
            raw_speed_mult = 0.0
        if raw_speed_mult > 0.0 and raw_speed_mult < 5.0:
            speed_mult = raw_speed_mult
        elif raw_speed_mult > 5.0:
            speed_mult = 1.0 + raw_speed_mult / 100.0
        else:
            speed_mult = raw_speed_mult

        linger_val = _stat_seconds("linger")
        if linger_val is None:
            raw_linger = card.get("linger", card.get("speed_linger", 0.0))
            linger_val = float(raw_linger or 0.0)
        linger = linger_val

        projectile_key = _stat_value("projectile")
        if not projectile_key:
            projectile_key = card.get("projectile")
        if not projectile_key and card_key.lower() == "arrows":
            projectile_key = DEFAULT_ARROWS_PROJECTILE
        proj_def: Optional[ProjectileDef] = get_projectile(projectile_key)

        proj_speed_val = _stat_value("projectile_speed")
        if proj_speed_val is None:
            proj_speed_val = card.get("speed")
        projectile_speed = 0.0
        if proj_speed_val is not None:
            projectile_speed = _scale_speed(proj_speed_val)
        elif proj_def:
            projectile_speed = proj_def.speed
        else:
            projectile_speed = 0.0

        hits_air = True
        hits_ground = True
        stat_hits_air = _stat_value("hits_air")
        stat_hits_ground = _stat_value("hits_ground")
        if stat_hits_air is not None:
            hits_air = bool(stat_hits_air)
        if stat_hits_ground is not None:
            hits_ground = bool(stat_hits_ground)
        else:
            targets = card.get("targets") or ["ground", "air"]
            hits_air = any(str(t).lower() in ("air", "both") for t in targets)
            hits_ground = any(str(t).lower() in ("ground", "both") for t in targets)

        pushback_val = _stat_value("pushback")
        if pushback_val is None:
            pushback_val = card.get("pushback", 0.0)
        try:
            pushback = float(pushback_val or 0.0)
        except (TypeError, ValueError):
            pushback = 0.0

        projectile_start_z = _stat_distance("projectile_start_z")
        if projectile_start_z is None:
            projectile_start_z = _scale_distance(card.get("projectile_start_z"))

        x = tx + 0.5
        y = ty + 0.5
        uses_projectile = bool(projectile_key) or projectile_speed > 0.0

        if damage > 0.0 and not uses_projectile:
            radius_for_damage = radius if radius > 0.0 else DEFAULT_SMALL_RADIUS
            self.deal_area(
                owner,
                x,
                y,
                radius_for_damage,
                damage,
                hits_air,
                hits_ground,
                ("spell", card_key),
                tower_multiplier=tower_multiplier,
            )

        if tick_dps > 0.0 and duration > 0.0:
            radius_for_effect = radius if radius > 0.0 else DEFAULT_SMALL_RADIUS
            self.effects.append(
                PoisonCloud(owner, x, y, radius_for_effect, duration, tick_dps, card_key, tower_multiplier=dot_tower_multiplier)
            )

        if stun_duration > 0.0:
            radius_for_stun = radius if radius > 0.0 else DEFAULT_SMALL_RADIUS
            self.apply_stun_area(owner, x, y, radius_for_stun, stun_duration)

        if projectile_key or projectile_speed > 0.0:
            start_y = y + (6.0 if owner == 1 else -6.0)
            projectile = Projectile(
                owner=owner,
                x=x,
                y=start_y,
                speed=max(0.1, projectile_speed if projectile_speed > 0.0 else 8.0),
                damage=damage if damage > 0.0 else (proj_def.damage if proj_def else 0.0),
                hit_radius=proj_def.hit_radius if proj_def else 0.25,
                area_radius=radius if radius > 0.0 else (proj_def.area_radius if proj_def else 0.0),
                hits_air=hits_air if projectile_key else (proj_def.hits_air if proj_def else True),
                hits_ground=hits_ground if projectile_key else (proj_def.hits_ground if proj_def else True),
                source=("spell", card_key),
                target_pos=(x, y),
                pushback=proj_def.pushback if proj_def else pushback,
                lifetime=proj_def.lifetime if proj_def else 0.0,
                homing=proj_def.homing if proj_def else True,
                z=projectile_start_z or 0.0,
            )
            self.projectiles.append(projectile)

        if speed_mult > 1.0 and duration > 0.0:
            radius_for_effect = radius if radius > 0.0 else 5.0
            effective_linger = linger if linger is not None else DEFAULT_RAGE_LINGER
            self.effects.append(RageZone(owner, x, y, radius_for_effect, duration, speed_mult, effective_linger, card_key))

        return True

    @staticmethod
    def _generate_spawn_offsets(count: int) -> List[Tuple[float, float]]:
        offsets: List[Tuple[float, float]] = [(0.0, 0.0)]
        if count <= 1:
            return offsets
        ring = 1
        while len(offsets) < count:
            for dy in range(-ring, ring + 1):
                for dx in range(-ring, ring + 1):
                    if dx == 0 and dy == 0:
                        continue
                    if max(abs(dx), abs(dy)) == ring:
                        offsets.append((float(dx), float(dy)))
                        if len(offsets) >= count:
                            return offsets
            ring += 1
        return offsets[:count]

    def _extract_unit_stats(self, card: Dict[str, Any], name: str) -> Dict[str, Any]:
        name_lower = name.lower()
        hp = float(card.get("hp", 100.0))
        damage = float(card.get("dmg", 50.0))
        rng = float(card.get("range", 1.0))
        hit_speed = max(float(card.get("hit_speed", 1.0)), MIN_HIT_SPEED)
        speed = float(card.get("speed", 1.0))
        targets = card.get("targets", ["ground"])
        if isinstance(targets, str):
            target_flags = [targets.lower()]
        else:
            target_flags = [str(t).lower() for t in targets]
        targeting = "ground"
        if "buildings" in target_flags and len(target_flags) == 1:
            targeting = "buildings"
        elif "air" in target_flags and "ground" in target_flags:
            targeting = "both"
        elif "air" in target_flags:
            targeting = "air"
        flying = bool(card.get("is_air", card.get("flying", False)))
        radius = float(card.get("radius", 0.45))
        if name_lower in {"giant", "golem", "electro giant", "royal giant"}:
            radius = max(radius, 0.7)
        if name_lower in {"balloon", "lava hound"}:
            flying = True
        aggro = float(card.get("aggro_range", card.get("sight_range", AGGRO_SIGHT)))
        projectile_speed = float(card.get("projectile_speed", 0.0) or 0.0)
        projectile_area_radius = float(card.get("splash_radius", 0.0) or 0.0)
        projectile_hit_radius = float(card.get("projectile_radius", projectile_area_radius if projectile_area_radius > 0 else 0.2) or 0.2)
        projectile_hits_air = bool(card.get("projectile_hits_air", "air" in target_flags))
        projectile_hits_ground = bool(
            card.get("projectile_hits_ground", any(t in ("ground", "buildings") for t in target_flags))
        )
        splash_radius = projectile_area_radius
        return {
            "hp": hp,
            "damage": damage,
            "range": rng,
            "hit_speed": hit_speed,
            "speed": speed,
            "targeting": targeting,
            "flying": flying,
            "radius": radius,
            "aggro": aggro,
            "projectile_speed": projectile_speed,
            "projectile_radius": projectile_area_radius,
            "projectile_hit_radius": projectile_hit_radius,
            "projectile_area_radius": projectile_area_radius,
            "projectile_hits_air": projectile_hits_air,
            "projectile_hits_ground": projectile_hits_ground,
            "target_flags": target_flags,
            "splash_radius": splash_radius,
            "projectile_lifetime": float(card.get("projectile_lifetime", 0.0) or 0.0),
        }

    def _create_unit(
        self,
        *,
        owner: int,
        card_key: str,
        name: str,
        level: int,
        card: Dict[str, Any],
        spec: Dict[str, Any],
        x: float,
        y: float,
        attached_to: Optional[Unit] = None,
        master_card: Optional[Dict[str, Any]] = None,
    ) -> Optional[Unit]:
        mechanics = (master_card or {}).get("stats", {})
        unit = Unit(
            id=self._unit_id_seq,
            owner=owner,
            name=name,
            x=x,
            y=y,
            hp=spec["hp"],
            hp_max=spec["hp"],
            speed=spec["speed"],
            range=spec["range"],
            hit_speed=spec["hit_speed"],
            damage=spec["damage"],
            targeting=spec["targeting"],
            flying=spec["flying"],
            radius=spec["radius"],
            aggro_range=spec["aggro"],
            level=level,
            projectile_speed=spec["projectile_speed"],
            projectile_radius=spec["projectile_radius"],
            projectile_hit_radius=spec["projectile_hit_radius"],
            projectile_area_radius=spec["projectile_area_radius"],
            projectile_lifetime=spec.get("projectile_lifetime", 0.0),
            projectile_hits_air=spec["projectile_hits_air"],
            projectile_hits_ground=spec["projectile_hits_ground"],
            projectile_key=card.get("projectile"),
            card_key=card_key,
            base_speed=spec["speed"],
            splash_radius=spec.get("splash_radius", 0.0),
        )
        if attached_to is not None:
            unit._attached_to = attached_to
        unit._mechanics = mechanics
        unit._master_card = master_card
        self._apply_mechanics_overrides(unit, spec=spec, mechanics=mechanics)
        deploy_time = max(0.0, _ms_to_seconds(mechanics.get("deploy_time")))
        unit.deploy_cooldown = deploy_time
        unit._deploy_total_time = deploy_time
        unit.active = False
        unit._deploy_complete_fired = False
        if getattr(unit, "_load_time", 0.0) <= 0.0:
            load_time_raw = mechanics.get("load_time")
            if load_time_raw is not None:
                unit._load_time = max(0.0, _ms_to_seconds(load_time_raw))
            else:
                fallback_load = card.get("load_time")
                if fallback_load is not None:
                    unit._load_time = max(0.0, float(fallback_load))
        load_after_retarget = _coerce_bool(mechanics.get("load_after_retarget"))
        if load_after_retarget is not None:
            unit._load_after_retarget = load_after_retarget
        elif "load_after_retarget" in card:
            fallback_flag = _coerce_bool(card.get("load_after_retarget"))
            if fallback_flag is not None:
                unit._load_after_retarget = fallback_flag
        first_hit_flag = _coerce_bool(mechanics.get("load_first_hit"))
        if first_hit_flag is None and "load_first_hit" in card:
            first_hit_flag = _coerce_bool(card.get("load_first_hit"))
        unit._first_attack_pending = bool(first_hit_flag) and unit._load_time > 0.0
        unit._windup_timer = 0.0
        unit._path_refresh_timer = 0.0
        unit._needs_detour = False
        unit._projectile_offset = _scale_distance(mechanics.get("projectile_start_radius"))
        unit._projectile_start_z = _scale_distance(mechanics.get("projectile_start_z"))
        unit._death_spawn_character = mechanics.get("death_spawn_character")
        unit._death_spawn_count = int(mechanics.get("death_spawn_count", 0) or mechanics.get("death_spawn_number", 0))
        unit._death_spawn_level_offset = int(mechanics.get("death_spawn_level_offset", 0))
        unit._death_damage = float(mechanics.get("death_damage") or 0.0)
        unit._death_damage_radius = _scale_distance(mechanics.get("death_damage_radius"))
        unit._death_hits_air = bool(mechanics.get("death_damage_hits_air", False))
        unit._last_progress_pos = (x, y)
        return unit

    def _on_unit_spawn(self, unit: Unit, card: Dict[str, Any]) -> None:
        spawn_cfg = copy.deepcopy(card.get("spawn_effect"))
        if spawn_cfg:
            self._perform_spawn_effect(unit, spawn_cfg)
        unit.spawn_effect = spawn_cfg
        unit.slow_effect = copy.deepcopy(card.get("slow_effect"))
        unit.chain_config = copy.deepcopy(card.get("attack_chain") or card.get("chain"))
        unit.support_units = copy.deepcopy(card.get("support_units"))
        unit.death_spawn_config = copy.deepcopy(card.get("death_spawn"))
        charge_cfg = card.get("charge")
        if charge_cfg:
            unit.charge_windup = float(charge_cfg.get("windup_time", unit.charge_windup))
            unit.charge_speed_mult = float(charge_cfg.get("speed_multiplier", unit.charge_speed_mult))
            unit.charge_damage_mult = float(charge_cfg.get("damage_multiplier", unit.charge_damage_mult))
            unit.charge_reset_on_hit = bool(charge_cfg.get("reset_on_hit", unit.charge_reset_on_hit))
        unit._stun_timer = 0.0
        unit._slow_timer = 0.0
        unit._slow_multiplier = 1.0
        unit._charge_progress = 0.0
        unit._charge_ready = False
        unit._charge_active = False
        if unit.support_units:
            self._spawn_support_units(unit, unit.support_units)
            unit.support_units = None

    def _apply_mechanics_overrides(self, unit: Unit, *, spec: Dict[str, Any], mechanics: Dict[str, Any]) -> None:
        level = max(1, int(getattr(unit, "level", 1)))
        base_sight = float(spec.get("aggro", unit.aggro_range))
        unit.sight_range = base_sight
        unit.aggro_range = base_sight
        unit.retarget_cooldown = DEFAULT_RETARGET_COOLDOWN
        unit.tower_damage_multiplier = 1.0

        target_types = _derive_target_types(spec.get("target_flags", []), mechanics)
        unit.target_types = target_types
        if target_types == ("buildings",):
            unit.targeting = "buildings"
        elif "ground" in target_types and "air" in target_types:
            unit.targeting = "both"
        elif "air" in target_types:
            unit.targeting = "air"
        else:
            unit.targeting = "ground"
        unit.projectile_hits_air = "air" in target_types
        unit.projectile_hits_ground = any(flag in ("ground", "buildings") for flag in target_types)

        if not mechanics:
            return

        range_val = _mechanic_level_value(mechanics, "range", level)
        if range_val is not None:
            rng = _scale_distance(range_val)
            if rng > 0.0:
                unit.range = rng

        sight_val = _mechanic_level_value(mechanics, "sight_range", level)
        if sight_val is not None:
            sight = _scale_distance(sight_val)
            if sight > 0.0:
                unit.sight_range = sight
                unit.aggro_range = sight

        hit_speed_val = _mechanic_level_value(mechanics, "hit_speed", level)
        if hit_speed_val is not None:
            hit_speed = _ms_to_seconds(hit_speed_val)
            if hit_speed > 0.0:
                unit.hit_speed = max(MIN_HIT_SPEED, hit_speed)

        load_val = _mechanic_level_value(mechanics, "load_time", level)
        if load_val is not None:
            unit._load_time = max(0.0, _ms_to_seconds(load_val))

        proj_speed_val = _mechanic_level_value(mechanics, "projectile_speed", level)
        if proj_speed_val is not None:
            scaled_speed = _scale_speed(proj_speed_val)
            if scaled_speed > 0.0:
                unit.projectile_speed = scaled_speed

        area_val = _mechanic_level_value(mechanics, "area_damage_radius", level)
        if area_val is None:
            area_val = mechanics.get("area_damage_radius")
        if area_val is not None:
            area_radius = _scale_distance(area_val)
            if area_radius > 0.0:
                unit.projectile_area_radius = area_radius
                unit.projectile_radius = area_radius
                unit.splash_radius = area_radius

        hit_radius_val = mechanics.get("projectile_radius")
        if hit_radius_val is not None:
            hit_radius = _scale_distance(hit_radius_val)
            if hit_radius > 0.0:
                unit.projectile_hit_radius = hit_radius

        lifetime_val = _mechanic_level_value(mechanics, "projectile_lifetime", level)
        if lifetime_val is None:
            lifetime_val = mechanics.get("projectile_lifetime")
        if lifetime_val:
            lifetime = float(lifetime_val)
            if lifetime > 10.0:
                lifetime = lifetime / 1000.0
            if lifetime > 0.0:
                unit.projectile_lifetime = lifetime

        load_after_retarget = _coerce_bool(mechanics.get("load_after_retarget"))
        if load_after_retarget is not None:
            unit._load_after_retarget = load_after_retarget

        first_hit_raw = _mechanic_level_value(mechanics, "load_first_hit", level)
        if first_hit_raw is None:
            first_hit_raw = mechanics.get("load_first_hit")
        first_hit_flag = _coerce_bool(first_hit_raw)
        unit._first_attack_pending = bool(first_hit_flag) and unit._load_time > 0.0

        damage_val = _mechanic_level_value(mechanics, "damage_per_level", level)
        if damage_val is None:
            damage_val = _mechanic_level_value(mechanics, "damage", level)
        damage_float = _coerce_float(damage_val)
        if damage_float is not None and damage_float > 0.0:
            unit.damage = damage_float

        collision_val = _mechanic_level_value(mechanics, "collision_radius", level)
        if collision_val is not None:
            radius = _scale_distance(collision_val)
            if radius > 0.0:
                unit.radius = radius

        tower_pct = _mechanic_level_value(mechanics, "crown_tower_damage_percent", level)
        if tower_pct is None:
            tower_pct = mechanics.get("crown_tower_damage_percent")
        pct_float = _coerce_float(tower_pct)
        if pct_float is not None:
            unit.tower_damage_multiplier = max(0.0, (100.0 + pct_float) / 100.0)

        death_damage_val = _mechanic_level_value(mechanics, "death_damage", level)
        if death_damage_val is None:
            death_damage_val = mechanics.get("death_damage")
        dd_float = _coerce_float(death_damage_val)
        if dd_float is not None:
            unit._death_damage = dd_float

        death_radius_val = _mechanic_level_value(mechanics, "death_damage_radius", level)
        if death_radius_val is None:
            death_radius_val = mechanics.get("death_damage_radius")
        dr_float = _coerce_float(death_radius_val)
        if dr_float is not None:
            unit._death_damage_radius = _scale_distance(dr_float)

        death_hits_air = _coerce_bool(mechanics.get("death_damage_hits_air"))
        if death_hits_air is not None:
            unit._death_hits_air = death_hits_air

        projectile_data = mechanics.get("projectile_data")
        if isinstance(projectile_data, dict):
            def _projectile_level_value(entry: Dict[str, Any], key: str) -> Optional[float]:
                if entry is None:
                    return None
                per_level = entry.get(f"{key}_per_level")
                if isinstance(per_level, list) and per_level:
                    idx = max(0, min(len(per_level) - 1, level - 1))
                    return _coerce_float(per_level[idx])
                if isinstance(per_level, dict) and per_level:
                    if str(level) in per_level:
                        return _coerce_float(per_level[str(level)])
                    numeric = sorted((int(k), v) for k, v in per_level.items() if str(k).isdigit())
                    for lvl, value in numeric:
                        if lvl >= level:
                            return _coerce_float(value)
                    if numeric:
                        return _coerce_float(numeric[-1][1])
                return _coerce_float(entry.get(key))

            selected_key = unit.projectile_key
            entry = projectile_data.get(selected_key) if selected_key else None
            dmg_val = _projectile_level_value(entry, "damage") if entry else None
            if (dmg_val or 0.0) <= 0.0:
                for key, candidate in projectile_data.items():
                    dmg_candidate = _projectile_level_value(candidate, "damage")
                    if dmg_candidate and dmg_candidate > 0.0:
                        selected_key = key
                        entry = candidate
                        dmg_val = dmg_candidate
                        break

            if entry:
                if dmg_val and dmg_val > 0.0:
                    unit.damage = dmg_val
                speed_val = _projectile_level_value(entry, "speed")
                if speed_val and speed_val > 0.0:
                    unit.projectile_speed = _scale_speed(speed_val)
                radius_val = _projectile_level_value(entry, "radius")
                if radius_val and radius_val > 0.0:
                    scaled_radius = _scale_distance(radius_val)
                    unit.projectile_area_radius = scaled_radius
                    unit.projectile_hit_radius = max(0.05, scaled_radius)
                hits_air_flag = entry.get("aoe_to_air")
                if hits_air_flag is not None:
                    unit.projectile_hits_air = bool(hits_air_flag)
                hits_ground_flag = entry.get("aoe_to_ground")
                if hits_ground_flag is not None:
                    unit.projectile_hits_ground = bool(hits_ground_flag)
                lifetime_val = entry.get("lifetime")
                if lifetime_val:
                    lifetime = float(lifetime_val)
                    if lifetime > 10.0:
                        lifetime = lifetime / 1000.0
                    unit.projectile_lifetime = max(unit.projectile_lifetime, lifetime)
                unit.projectile_key = selected_key or unit.projectile_key

        death_entries: List[Dict[str, Any]] = []
        for suffix in ("", "2", "3"):
            key_suffix = suffix
            character = mechanics.get(f"death_spawn_character{key_suffix}") or mechanics.get(
                f"death_spawn_character{key_suffix}".rstrip("2").rstrip("3")
            )
            if not character:
                continue
            count_val = _mechanic_level_value(mechanics, f"death_spawn_count{key_suffix}", level)
            if count_val is None:
                count_val = mechanics.get(f"death_spawn_count{key_suffix}", mechanics.get("death_spawn_count"))
            count_int = int(_coerce_float(count_val) or 0)
            if count_int <= 0:
                continue
            radius_val = mechanics.get(f"death_spawn_radius{key_suffix}", mechanics.get("death_spawn_radius"))
            min_radius_val = mechanics.get(
                f"death_spawn_min_radius{key_suffix}", mechanics.get("death_spawn_min_radius")
            )
            entry = {
                "name": character,
                "count": count_int,
                "radius": _scale_distance(radius_val) if radius_val is not None else 0.0,
                "min_radius": _scale_distance(min_radius_val) if min_radius_val is not None else 0.0,
                "level_offset": int(
                    _coerce_float(mechanics.get(f"death_spawn_level_offset{key_suffix}", 0)) or 0
                ),
                "deploy_time": _ms_to_seconds(
                    mechanics.get(f"death_spawn_deploy_time{key_suffix}", mechanics.get("death_spawn_deploy_time"))
                ),
                "randomize": bool(mechanics.get("randomize_death_spawn", False)),
                "allow_over_river": bool(
                    mechanics.get(f"death_spawn_allow_over_river{key_suffix}", False)
                ),
            }
            death_entries.append(entry)

        if death_entries:
            unit._death_spawn_entries = death_entries
            primary = death_entries[0]
            unit._death_spawn_character = primary["name"]
            unit._death_spawn_count = primary["count"]
            unit._death_spawn_level_offset = primary.get("level_offset", 0)

        retarget_keys = (
            "retarget_cooldown",
            "retarget_time",
            "retarget_delay",
            "retarget_speed",
        )
        for key in retarget_keys:
            val = mechanics.get(key)
            if val is None:
                continue
            retarget = _coerce_float(_mechanic_level_value(mechanics, key, level))
            if retarget is None:
                retarget = _coerce_float(val)
            if retarget is None:
                continue
            if retarget > 10.0:
                retarget = _ms_to_seconds(retarget)
            unit.retarget_cooldown = max(0.0, retarget)
            break
        else:
            if _coerce_bool(mechanics.get("retarget_each_tick")):
                unit.retarget_cooldown = 0.0

    def _complete_unit_deploy(self, unit: Unit, *, immediate: bool = False) -> None:
        if getattr(unit, "_deploy_complete_fired", False):
            unit.active = True
            unit.deploy_cooldown = 0.0
            return
        unit.deploy_cooldown = 0.0
        unit.active = True
        unit._deploy_complete_fired = True
        payload = {
            "unit_id": unit.id,
            "owner": unit.owner,
            "card_key": unit.card_key,
            "name": unit.name,
            "position": {"x": unit.x, "y": unit.y},
            "deploy_time": float(getattr(unit, "_deploy_total_time", 0.0)),
            "immediate": bool(immediate),
        }
        self._dispatch_event("on_deploy_complete", payload)

    def _initialize_deploy_state(self, unit: Unit, *, skip_delay: bool = False) -> None:
        cooldown = max(0.0, float(getattr(unit, "deploy_cooldown", 0.0)))
        if skip_delay:
            cooldown = 0.0
            unit._deploy_total_time = 0.0
        elif cooldown > 0.0:
            unit._deploy_total_time = max(float(getattr(unit, "_deploy_total_time", 0.0)), cooldown)
        unit.deploy_cooldown = cooldown
        if cooldown <= 1e-6:
            self._complete_unit_deploy(unit, immediate=True)
        else:
            unit.active = False
            unit._deploy_complete_fired = False

    def _perform_spawn_effect(self, unit: Unit, cfg: Dict[str, Any]) -> None:
        damage = float(cfg.get("damage", 0.0))
        radius = float(cfg.get("radius", 0.0))
        stun = float(cfg.get("stun", 0.0))
        if radius <= 0.0:
            return
        if damage > 0.0:
            self.deal_area(unit.owner, unit.x, unit.y, radius, damage, True, True, ("spawn", unit.name))
        if stun > 0.0:
            self.apply_stun_area(unit.owner, unit.x, unit.y, radius, stun)

    def _spawn_support_units(self, parent: Unit, configs: List[Dict[str, Any]]) -> None:
        for cfg in configs or []:
            name = cfg.get("name")
            if not name:
                continue
            count = int(cfg.get("count", 1))
            level_offset = int(cfg.get("level_offset", 0))
            spread = float(cfg.get("spread", 0.3))
            self._spawn_auxiliary_units(
                owner=parent.owner,
                name=name,
                level=parent.level + level_offset,
                origin_x=parent.x,
                origin_y=parent.y,
                count=count,
                spread=spread,
                attached_to=parent,
                follow_parent=True,
            )

    def _spawn_auxiliary_units(
        self,
        *,
        owner: int,
        name: str,
        level: int,
        origin_x: float,
        origin_y: float,
        count: int,
        spread: float,
        attached_to: Optional[Unit] = None,
        follow_parent: bool = False,
        min_spread: float = 0.0,
        randomize: bool = True,
    ) -> None:
        card = get_card(name, level=level)
        if not card:
            return
        spec = self._extract_unit_stats(card, name)
        master_card = self._lookup_master_card(name, card)
        for idx in range(max(1, count)):
            if follow_parent and attached_to is not None and count > 1:
                offset_x = (idx - (count - 1) / 2.0) * spread
                offset_y = 0.0
            elif follow_parent and attached_to is not None:
                offset_x = 0.0
                offset_y = 0.0
            else:
                if randomize:
                    angle = random.random() * 2.0 * math.pi
                else:
                    angle = (idx / max(1, count)) * 2.0 * math.pi
                radius = max(0.0, spread)
                if radius > 0.0 and min_spread > 0.0:
                    radius = random.uniform(min_spread, max(min_spread, radius))
                offset_x = math.cos(angle) * radius
                offset_y = math.sin(angle) * radius
            px = origin_x + offset_x
            py = origin_y + offset_y
            unit = self._create_unit(
                owner=owner,
                card_key=name,
                name=name,
                level=level,
                card=card,
                spec=spec,
                x=px,
                y=py,
                attached_to=attached_to if follow_parent else None,
                master_card=master_card,
            )
            if unit is None:
                continue
            unit._windup_timer = 0.0
            if follow_parent and attached_to is not None:
                unit._attached_to = attached_to
                unit._attach_offset = (offset_x, offset_y)
                unit.radius = min(unit.radius, 0.2)
            self.units.append(unit)
            if follow_parent and attached_to is not None:
                attached_to._support_children.append(unit.id)
            self._unit_id_seq += 1
            self._on_unit_spawn(unit, card)
            self._initialize_deploy_state(unit, skip_delay=True)

    def _tower_select_target(self, tower: Dict[str, Any], tower_unit: Unit, enemies: Sequence[Unit]) -> Optional[Unit]:
        best_unit: Optional[Unit] = None
        best_score = float("inf")
        for enemy in enemies:
            if not enemy.alive:
                continue
            if enemy.flying and "air" not in tower_unit.target_types:
                continue
            if (not enemy.flying) and not any(flag in ("ground", "buildings") for flag in tower_unit.target_types):
                continue
            if not self._tower_can_target(tower, enemy):
                continue
            dist = distance_tiles((tower_unit.x, tower_unit.y), (enemy.x, enemy.y))
            if dist > tower_unit.sight_range + enemy.radius:
                continue
            score = dist
            if getattr(enemy, "targeting", "") == "buildings":
                score -= 0.25
            if score < best_score:
                best_score = score
                best_unit = enemy
        return best_unit
    def _tower_can_target(self, tower: Dict[str, Any], unit: Unit) -> bool:
        """Return True if `tower` is allowed to fire at `unit`.

        Princess towers stay in their lane: we gate vertical reach with the
        river band and horizontal reach with the arena midline so each tower
        only covers its half.
        """
        if not unit.alive:
            return False
        if distance_tiles((tower["cx"], tower["cy"]), (unit.x, unit.y)) > tower.get("range", 7.0) + unit.radius:
            return False

        tower_type = tower.get("type")
        if tower_type == "princess":
            if self._river_band is not None:
                river_min, river_max = self._river_band
                if tower["owner"] == 1 and unit.y < river_min - 1e-6:
                    return False
                if tower["owner"] == 2 and unit.y > river_max + 1e-6:
                    return False

            label = tower.get("label", "")
            mid_x = self.arena.width / 2.0
            # Left/right lanes: left towers ignore targets past mid_x and vice versa.
            if "left" in label and unit.x > mid_x + 0.5:
                return False
            if "right" in label and unit.x < mid_x - 0.5:
                return False
        return True

    def _update_towers(self) -> None:
        enemies_by_owner: Dict[int, List[Unit]] = {
            1: [u for u in self.units if u.alive and u.owner == 2],
            2: [u for u in self.units if u.alive and u.owner == 1],
        }
        for tower in self.towers:
            if not tower.get("alive", True):
                continue
            unit = tower.get("_unit")
            if unit is None:
                continue
            self._sync_tower_unit_state(tower)
            unit._retarget_timer = max(0.0, getattr(unit, "_retarget_timer", 0.0) - self.dt)
            if tower.get("stun_timer", 0.0) > 0.0:
                tower["stun_timer"] = max(0.0, tower["stun_timer"] - self.dt)
                unit._stun_timer = tower["stun_timer"]
            if tower.get("slow_timer", 0.0) > 0.0:
                tower["slow_timer"] = max(0.0, tower["slow_timer"] - self.dt)
                if tower["slow_timer"] <= 0.0:
                    tower["slow_multiplier"] = 1.0
            unit._slow_timer = tower.get("slow_timer", 0.0)
            unit._slow_multiplier = tower.get("slow_multiplier", 1.0)
            if unit._windup_timer > 0.0:
                unit._windup_timer = max(0.0, unit._windup_timer - self.dt)
            tower["slow_multiplier"] = unit._slow_multiplier
            if not unit.active:
                continue
            target = tower.get("target")
            if target is not None:
                if isinstance(target, Unit):
                    if not target.alive or not self._tower_can_target(tower, target):
                        target = None
                else:
                    target = None
            if target is None or unit._retarget_timer <= 1e-6:
                candidate = self._tower_select_target(tower, unit, enemies_by_owner.get(unit.owner, []))
                tower["target"] = candidate
                target = candidate
                unit._pending_attack_signature = None
                unit._retarget_timer = unit.retarget_cooldown
            if target is None:
                tower["cooldown"] = getattr(unit, "_attack_cd", 0.0)
                continue
            if self._try_attack(unit, target):
                tower["target"] = target
            tower["cooldown"] = getattr(unit, "_attack_cd", 0.0)
    def _update_units(self) -> None:
        alive_units = [u for u in self.units if u.alive]
        for unit in alive_units:
            self._update_unit_status(unit)
        self._maintain_support_positions(alive_units)

        buckets = self._build_spatial_buckets()
        for unit in alive_units:
            self._refresh_unit_target(unit, buckets)

        for unit in alive_units:
            if not unit.alive:
                continue
            if getattr(unit, "_attached_to", None) is not None:
                unit._path = []
                continue
            path_target: Optional[Union[Unit, Dict[str, Any]]] = self._get_unit_target(unit)
            if unit.targeting == "buildings":
                structure = self._nearest_enemy_structure(unit)
                if structure is not None:
                    path_target = structure
            self._update_unit_path(unit, path_target)

        buckets = self._build_spatial_buckets()
        for unit in alive_units:
            if not unit.alive:
                continue
            target = self._get_unit_target(unit)
            if unit._stun_timer > 0.0:
                self._resolve_unit_collisions(unit, buckets)
                continue
            if self._try_attack(unit, target):
                self._resolve_unit_collisions(unit, buckets)
                continue
            if self._should_hold_position(unit, target):
                unit._path = []
                self._resolve_unit_collisions(unit, buckets)
                continue
            self._move_along_path(unit)
            self._resolve_unit_collisions(unit, buckets)

        self._handle_unit_deaths()
        self.units = [u for u in self.units if u.alive]

        if not any(tw.get("alive", True) and tw.get("owner") == 1 for tw in self.towers):
            self.over, self.winner = True, 2
        if not any(tw.get("alive", True) and tw.get("owner") == 2 for tw in self.towers):
            self.over, self.winner = True, 1

    def _update_unit_status(self, unit: Unit) -> None:
        deploy_cd = max(0.0, float(getattr(unit, "deploy_cooldown", 0.0)))
        if deploy_cd > 0.0:
            deploy_cd = max(0.0, deploy_cd - self.dt)
            unit.deploy_cooldown = deploy_cd
            if deploy_cd > 0.0:
                unit.active = False
            else:
                self._complete_unit_deploy(unit)
        else:
            if not getattr(unit, "_deploy_complete_fired", False):
                immediate = float(getattr(unit, "_deploy_total_time", 0.0)) <= 1e-6
                self._complete_unit_deploy(unit, immediate=immediate)
            else:
                unit.active = True
        if getattr(unit, "_stun_timer", 0.0) > 0.0:
            unit._stun_timer = max(0.0, unit._stun_timer - self.dt)
            if unit._stun_timer <= 0.0:
                unit._attack_cd = max(0.0, unit._attack_cd)
        if getattr(unit, "_slow_timer", 0.0) > 0.0:
            unit._slow_timer = max(0.0, unit._slow_timer - self.dt)
            if unit._slow_timer <= 0.0:
                unit._slow_multiplier = 1.0
        if getattr(unit, "_rage_timer", 0.0) > 0.0:
            unit._rage_timer = max(0.0, unit._rage_timer - self.dt)
            if unit._rage_timer <= 0.0 and getattr(unit, "_rage_linger", 0.0) <= 0.0:
                unit._rage_multiplier = 1.0
        if getattr(unit, "_rage_timer", 0.0) <= 0.0 and getattr(unit, "_rage_linger", 0.0) > 0.0:
            unit._rage_linger = max(0.0, unit._rage_linger - self.dt)
            if unit._rage_linger <= 0.0:
                unit._rage_multiplier = 1.0
        if getattr(unit, "_stun_timer", 0.0) > 0.0:
            self._reset_charge(unit)
        if getattr(unit, "_path_refresh_timer", 0.0) > 0.0:
            unit._path_refresh_timer = max(0.0, unit._path_refresh_timer - self.dt)
        if getattr(unit, "_windup_timer", 0.0) > 0.0:
            unit._windup_timer = max(0.0, unit._windup_timer - self.dt)

    def _maintain_support_positions(self, units: List[Unit]) -> None:
        for unit in units:
            parent = getattr(unit, "_attached_to", None)
            if parent is None:
                continue
            if not parent.alive:
                unit._attached_to = None
                unit._attach_offset = (0.0, 0.0)
                continue
            unit.x = parent.x + unit._attach_offset[0]
            unit.y = parent.y + unit._attach_offset[1]
            unit._path = []

    def _current_move_speed(self, unit: Unit) -> float:
        base = getattr(unit, "base_speed", unit.speed)
        speed = base
        speed *= getattr(unit, "_rage_multiplier", 1.0)
        speed *= getattr(unit, "_slow_multiplier", 1.0)
        if getattr(unit, "_charge_active", False):
            speed *= unit.charge_speed_mult
        return max(0.0, speed)

    def _reset_charge(self, unit: Unit) -> None:
        if getattr(unit, "charge_windup", 0.0) <= 0.0:
            return
        unit._charge_progress = 0.0
        unit._charge_ready = False
        unit._charge_active = False

    def _handle_unit_deaths(self) -> None:
        for unit in self.units:
            if unit.alive or getattr(unit, "_death_handled", False):
                continue
            self._process_unit_death(unit)
            unit._death_handled = True

    def _process_unit_death(self, unit: Unit) -> None:
        if unit._support_children:
            for child_id in unit._support_children:
                child = self._find_unit_by_id(child_id)
                if child and child.alive:
                    child._attached_to = None
                    child._attach_offset = (0.0, 0.0)
        cfg = getattr(unit, "death_spawn_config", None)
        if not cfg:
            cfg = {}
        for entry in cfg.get("units", []):
            name = entry.get("name")
            if not name:
                continue
            count = int(entry.get("count", 1))
            spread = float(entry.get("spread", 0.6))
            level_offset = int(entry.get("level_offset", 0))
            self._spawn_auxiliary_units(
                owner=unit.owner,
                name=name,
                level=unit.level + level_offset,
                origin_x=unit.x,
                origin_y=unit.y,
                count=count,
                spread=spread,
                attached_to=None,
                follow_parent=False,
            )
        spawn_handled = False
        entries = getattr(unit, "_death_spawn_entries", []) or []
        if entries:
            spawn_handled = True
            for entry in entries:
                name = entry.get("name")
                if not name:
                    continue
                count = int(entry.get("count", 0))
                if count <= 0:
                    continue
                spread = float(entry.get("radius", getattr(unit, "_death_damage_radius", 0.6)) or 0.0)
                if spread <= 0.0:
                    spread = max(0.4, getattr(unit, "_death_damage_radius", 0.6))
                min_radius = float(entry.get("min_radius", 0.0) or 0.0)
                level_offset = int(entry.get("level_offset", 0) or 0)
                self._spawn_auxiliary_units(
                    owner=unit.owner,
                    name=name,
                    level=unit.level + level_offset,
                    origin_x=unit.x,
                    origin_y=unit.y,
                    count=count,
                    spread=spread,
                    attached_to=None,
                    follow_parent=False,
                    min_spread=min_radius,
                    randomize=bool(entry.get("randomize", True)),
                )

        if not spawn_handled:
            spawn_name = getattr(unit, "_death_spawn_character", None)
            spawn_count = int(getattr(unit, "_death_spawn_count", 0))
            if spawn_name and spawn_count > 0:
                spread = max(0.4, min(0.8, getattr(unit, "_death_damage_radius", 0.6)))
                self._spawn_auxiliary_units(
                    owner=unit.owner,
                    name=spawn_name,
                    level=unit.level + int(getattr(unit, "_death_spawn_level_offset", 0)),
                    origin_x=unit.x,
                    origin_y=unit.y,
                    count=spawn_count,
                    spread=spread,
                    attached_to=None,
                    follow_parent=False,
                )
        death_damage = float(getattr(unit, "_death_damage", 0.0))
        if death_damage > 0.0:
            radius = max(float(getattr(unit, "_death_damage_radius", 0.0) or 0.0), DEFAULT_SMALL_RADIUS)
            hits_air = bool(getattr(unit, "_death_hits_air", False))
            self.deal_area(unit.owner, unit.x, unit.y, radius, death_damage, hits_air, True, ('death', unit.name))

    def _apply_melee_damage(self, attacker: Unit, target: Union[Unit, Dict[str, Any]], damage: float) -> None:
        splash = getattr(attacker, 'splash_radius', 0.0)
        if splash > 0.0:
            if isinstance(target, Unit):
                cx, cy = target.x, target.y
            else:
                cx, cy = self._structure_center(target)
            self.deal_area(attacker.owner, cx, cy, splash, damage, attacker.can_attack_air(), True, ('unit', attacker.name))
        else:
            if isinstance(target, Unit):
                target.take_damage(damage, attacker.owner, ('unit', attacker.name))
            else:
                self.damage_structure(target, damage, attacker.owner, ('unit', attacker.name))

    def _apply_slow_effect(self, attacker: Unit, target: Union[Unit, Dict[str, Any]], effect: Optional[Dict[str, Any]]) -> None:
        if not effect:
            return
        duration = float(effect.get('duration', 0.0))
        if duration <= 0.0:
            return
        multiplier = float(effect.get('speed_multiplier', 1.0))
        if isinstance(target, Unit):
            target.apply_slow(multiplier, duration)
            self._reset_charge(target)
        else:
            target['slow_timer'] = max(duration, target.get('slow_timer', 0.0))
            target['slow_multiplier'] = min(multiplier, target.get('slow_multiplier', 1.0))

    def _apply_chain_damage(self, attacker: Unit, primary_target: Union[Unit, Dict[str, Any]], config: Optional[Dict[str, Any]], base_damage: float) -> None:
        if not config:
            return
        max_hits = int(config.get('targets', 1))
        if max_hits <= 1:
            return
        radius = float(config.get('jump_radius', 2.0))
        stun = float(config.get('stun', 0.0))
        chain_damage = float(config.get('damage', base_damage))
        visited_units: Set[int] = set()
        visited_structures: Set[str] = set()
        source: Union[Unit, Dict[str, Any]] = primary_target
        hits = 1
        while hits < max_hits:
            candidate = self._find_chain_target(attacker, source, radius, visited_units, visited_structures)
            if candidate is None:
                break
            if isinstance(candidate, Unit):
                candidate.take_damage(chain_damage, attacker.owner, ('unit', attacker.name))
                if stun > 0.0:
                    candidate.apply_stun(stun)
            else:
                self.damage_structure(candidate, chain_damage, attacker.owner, ('unit', attacker.name))
                if stun > 0.0:
                    candidate['stun_timer'] = max(candidate.get('stun_timer', 0.0), stun)
            source = candidate
            hits += 1

    def _find_chain_target(self, attacker: Unit, source: Union[Unit, Dict[str, Any]], radius: float, visited_units: Set[int], visited_structures: Set[str]) -> Optional[Union[Unit, Dict[str, Any]]]:
        if isinstance(source, Unit):
            visited_units.add(source.id)
            src_pos = (source.x, source.y)
        else:
            label = source.get('label')
            if label:
                visited_structures.add(label)
            src_pos = self._structure_center(source)
        best_unit = None
        best_dist = float('inf')
        for unit in self.units:
            if not unit.alive or unit.owner == attacker.owner:
                continue
            if unit.id in visited_units:
                continue
            dist = math.hypot(unit.x - src_pos[0], unit.y - src_pos[1])
            if dist <= radius and dist < best_dist:
                best_dist = dist
                best_unit = unit
        if best_unit is not None:
            visited_units.add(best_unit.id)
            return best_unit
        best_tower = None
        best_dist = float('inf')
        for tower in self.towers:
            if not tower.get('alive', True) or tower.get('owner') == attacker.owner:
                continue
            label = tower.get('label')
            if label and label in visited_structures:
                continue
            cx = tower.get('cx')
            cy = tower.get('cy')
            if cx is None or cy is None:
                cx, cy = self._structure_center(tower)
            dist = math.hypot(cx - src_pos[0], cy - src_pos[1])
            if dist <= radius and dist < best_dist:
                best_dist = dist
                best_tower = tower
        if best_tower is not None:
            label = best_tower.get('label')
            if label:
                visited_structures.add(label)
            return best_tower
        return None

    def _apply_stun_to_unit(self, target: Unit, duration: float) -> None:
        if duration <= 0.0:
            return
        target.apply_stun(duration)
        self._reset_charge(target)

    def _update_projectiles(self) -> None:
        if not self.projectiles:
            return
        alive: List[Projectile] = []
        for projectile in self.projectiles:
            projectile.tick(self, self.dt)
            if projectile.alive:
                alive.append(projectile)
        self.projectiles = alive

    def _update_effects(self) -> None:
        if not self.effects:
            return
        alive_effects: List[SpellEffect] = []
        for effect in self.effects:
            effect.tick(self, self.dt)
            if getattr(effect, "alive", True):
                alive_effects.append(effect)
        self.effects = alive_effects

    def _collides_with_tower(self, gx: int, gy: int) -> bool:
        for tower in self.towers:
            if not tower.get("alive", True):
                continue
            if tower["x0"] <= gx < tower["x0"] + tower["width"] and tower["y0"] <= gy < tower["y0"] + tower["height"]:
                return True
        return False

    def _collides_with_units(self, x: float, y: float, radius: float, owner: int, flying: bool) -> bool:
        for unit in self.units:
            if not unit.alive:
                continue
            if getattr(unit, "_attached_to", None) is not None:
                continue
            if unit.flying != flying:
                continue
            dist = distance_tiles((unit.x, unit.y), (x, y))
            if unit.owner != owner and dist < (unit.radius + radius):
                return True
            if unit.owner == owner and dist < (unit.radius + radius) * 0.7:
                return True
        return False

    def _nearest_enemy_structure(self, unit: Unit) -> Optional[Dict[str, Any]]:
        best = None
        best_dist = float("inf")
        for tower in self.towers:
            if not tower.get("alive", True) or tower.get("owner") == unit.owner:
                continue
            cx = tower["x0"] + tower["width"] / 2.0
            cy = tower["y0"] + tower["height"] / 2.0
            d = distance_tiles((unit.x, unit.y), (cx, cy))
            if d < best_dist:
                best_dist = d
                best = tower
        return best

    def _update_unit_path(self, unit: Unit, target: Optional[Union[Unit, Dict[str, Any]]]) -> None:
        if target is None:
            unit._path = []
            unit._path_goal_id = None
            return
        if getattr(unit, '_attached_to', None) is not None:
            unit._path = []
            unit._path_goal_id = None
            return
        goal_id = id(target)
        needs_new = False
        if not unit._path:
            needs_new = True
        elif unit._path_goal_id != goal_id:
            needs_new = True
        elif not self._path_head_walkable(unit):
            needs_new = True
        building_only = unit.targeting == "buildings"
        blocked_tiles: Set[Tuple[int, int]] = set()
        if building_only:
            friendly_tiles: Set[Tuple[int, int]] = set()
            enemy_tiles: Set[Tuple[int, int]] = set()
            for other in self.units:
                if other is unit or not other.alive:
                    continue
                if getattr(other, "_attached_to", None) is not None:
                    continue
                tile = (int(math.floor(other.x)), int(math.floor(other.y)))
                dist = math.hypot(unit.x - other.x, unit.y - other.y)
                if other.owner == unit.owner:
                    if dist <= unit.radius + other.radius + 0.5:
                        friendly_tiles.add(tile)
                else:
                    if dist <= unit.radius + other.radius + 0.5:
                        enemy_tiles.add(tile)
            blocked_tiles = friendly_tiles | enemy_tiles
            start_tile = (int(math.floor(unit.x)), int(math.floor(unit.y)))
            if start_tile in friendly_tiles:
                blocked_tiles.discard(start_tile)
            if getattr(unit, "_path_refresh_timer", 0.0) <= 0.0:
                needs_new = True
            if getattr(unit, "_needs_detour", False):
                needs_new = True
        if needs_new:
            unit._path = compute_path(
                self.arena,
                unit.__dict__,
                target,
                blocked_tiles=blocked_tiles if blocked_tiles else None,
            )
            unit._path_goal_id = goal_id
            self._reset_charge(unit)
            unit._needs_detour = False
            if building_only:
                unit._path_refresh_timer = 0.3

    def _path_head_walkable(self, unit: Unit) -> bool:
        if not unit._path:
            return True
        nx, ny = unit._path[0]
        return self.arena.is_walkable(unit.owner, nx, ny, flying=unit.flying)

    def _move_along_path(self, unit: Unit) -> None:
        if not getattr(unit, "active", True):
            return
        if getattr(unit, '_attached_to', None) is not None:
            return
        if getattr(unit, '_stun_timer', 0.0) > 0.0:
            self._reset_charge(unit)
            return
        if not unit._path and unit.targeting == "buildings":
            structure = self._nearest_enemy_structure(unit)
            if structure:
                self._update_unit_path(unit, structure)
        if not unit._path:
            if not getattr(unit, '_charge_ready', False):
                self._reset_charge(unit)
            return
        speed = self._current_move_speed(unit)
        if speed <= 1e-6:
            if not getattr(unit, '_charge_ready', False):
                self._reset_charge(unit)
            return
        nx, ny = unit._path[0]
        tx, ty = nx + 0.5, ny + 0.5
        dx = tx - unit.x
        dy = ty - unit.y
        dist = math.hypot(dx, dy)
        step = speed * self.dt
        moved = False
        if dist <= step:
            unit.x, unit.y = tx, ty
            unit._path.pop(0)
            moved = dist > 0.0
        else:
            unit.x += dx / (dist + 1e-9) * step
            unit.y += dy / (dist + 1e-9) * step
            moved = step > 0.0
        if getattr(unit, 'charge_windup', 0.0) > 0.0:
            if moved and getattr(unit, '_stun_timer', 0.0) <= 0.0:
                unit._charge_progress += self.dt
                if not unit._charge_ready and unit._charge_progress >= unit.charge_windup:
                    unit._charge_ready = True
                    unit._charge_active = True
            elif not unit._charge_ready:
                self._reset_charge(unit)
        if unit._path:
            last_x, last_y = unit._last_progress_pos
            displacement = math.hypot(unit.x - last_x, unit.y - last_y)
            if displacement > 0.04:
                unit._stuck_timer = 0.0
                unit._last_progress_pos = (unit.x, unit.y)
            else:
                unit._stuck_timer += self.dt
                if unit._stuck_timer >= 0.6:
                    unit._path = []
                    unit._needs_detour = True
                    unit._path_refresh_timer = 0.0
                    unit._stuck_timer = 0.0
                    unit._last_progress_pos = (unit.x, unit.y)
        else:
            unit._stuck_timer = 0.0
            unit._last_progress_pos = (unit.x, unit.y)

    def _clear_unit_target(self, unit: Unit) -> None:
        unit._current_target_id = None
        unit._current_target_label = None
        unit._pending_attack_signature = None

    def _set_unit_target(self, unit: Unit, target: Union[Unit, Dict[str, Any]]) -> None:
        if isinstance(target, Unit):
            unit._current_target_id = target.id
            unit._current_target_label = None
        else:
            unit._current_target_label = target.get("label")
            unit._current_target_id = None
        unit._pending_attack_signature = None

    def _get_unit_target(self, unit: Unit) -> Optional[Union[Unit, Dict[str, Any]]]:
        if unit._current_target_id is not None:
            candidate = self._find_unit_by_id(unit._current_target_id)
            if candidate and candidate.alive:
                return candidate
            unit._current_target_id = None
        if unit._current_target_label:
            structure = self._find_structure_by_label(unit._current_target_label)
            if structure and structure.get("alive", True):
                return structure
            unit._current_target_label = None
        return None

    def _find_structure_by_label(self, label: Optional[str]) -> Optional[Dict[str, Any]]:
        if not label:
            return None
        for tower in self.towers:
            if tower.get("label") == label:
                return tower
        for building in getattr(self.arena, "buildings", ()):
            if building.get("label") == label:
                return building
        return None

    def _target_signature(self, target: Union[Unit, Dict[str, Any]]) -> str:
        if isinstance(target, Unit):
            return f"unit:{target.id}"
        if isinstance(target, dict) and "x" in target and "y" in target:
            return f"unitdict:{target.get('id', 'anon')}:{target['x']}:{target['y']}"
        label = target.get("label")
        if label:
            return f"struct:{label}"
        x0 = target.get("x0", 0)
        y0 = target.get("y0", 0)
        width = target.get("width", 1)
        height = target.get("height", 1)
        return f"struct:{x0}:{y0}:{width}:{height}"

    def _target_point_category(self, unit: Unit, target: Union[Unit, Dict[str, Any]]) -> Tuple[Tuple[float, float], str, float]:
        if isinstance(target, Unit):
            radius = getattr(target, "radius", 0.4)
            category = "air" if target.flying else "ground"
            return (target.x, target.y), category, radius
        if isinstance(target, dict):
            if "x" in target and "y" in target:
                radius = float(target.get("radius", 0.4) or 0.4)
                category = "air" if bool(target.get("flying", False)) else "ground"
                return (float(target["x"]), float(target["y"])), category, radius
            if "x0" in target and "y0" in target and "width" in target and "height" in target:
                cx, cy = self._closest_point_on_rect(unit.x, unit.y, target)
                return (cx, cy), "buildings", 0.0
        cx, cy = (float(target.get("x", unit.x)), float(target.get("y", unit.y))) if isinstance(target, dict) else (unit.x, unit.y)
        return (cx, cy), "ground", 0.0

    def _target_within_sight(self, unit: Unit, target: Union[Unit, Dict[str, Any]]) -> bool:
        point, _category, radius = self._target_point_category(unit, target)
        dist = math.hypot(unit.x - point[0], unit.y - point[1]) - radius
        dist = max(0.0, dist)
        return dist <= unit.sight_range + unit.radius + 1e-6

    def _target_in_attack_range(self, unit: Unit, point: Tuple[float, float], target_radius: float) -> bool:
        distance = math.hypot(unit.x - point[0], unit.y - point[1]) - (unit.radius + target_radius)
        distance = max(0.0, distance)
        if unit.range <= MELEE_RANGE + 1e-6:
            return distance <= MELEE_RANGE
        return distance <= unit.range + 1e-6

    def _unit_can_target_enemy(self, unit: Unit, target: Union[Unit, Dict[str, Any]]) -> bool:
        _point, category, _radius = self._target_point_category(unit, target)
        if category == "air":
            return "air" in unit.target_types
        if category == "ground":
            return any(flag in ("ground", "buildings") for flag in unit.target_types)
        if category == "buildings":
            return "buildings" in unit.target_types or "ground" in unit.target_types
        return False

    def _find_best_target(self, unit: Unit, buckets: Dict[Tuple[int, int], List[Unit]], cell: float = 2.0) -> Optional[Union[Unit, Dict[str, Any]]]:
        best_unit = self._find_best_unit_target(unit, buckets, cell)
        if best_unit is not None:
            return best_unit
        if any(flag in ("ground", "buildings") for flag in unit.target_types):
            structure = self._nearest_enemy_structure(unit)
            if structure:
                return structure
        return None

    def _find_best_unit_target(self, unit: Unit, buckets: Dict[Tuple[int, int], List[Unit]], cell: float) -> Optional[Unit]:
        if unit.sight_range <= 0.0:
            return None
        origin_x = int(unit.x // cell)
        origin_y = int(unit.y // cell)
        reach = unit.sight_range + unit.radius + 1.0
        cell_radius = max(1, int(math.ceil(reach / cell)))
        best_unit = None
        best_dist = float("inf")
        for gx in range(origin_x - cell_radius, origin_x + cell_radius + 1):
            for gy in range(origin_y - cell_radius, origin_y + cell_radius + 1):
                for enemy in buckets.get((gx, gy), ()):
                    if enemy.owner == unit.owner or not enemy.alive:
                        continue
                    if not self._unit_can_target_enemy(unit, enemy):
                        continue
                    dist = math.hypot(unit.x - enemy.x, unit.y - enemy.y) - enemy.radius
                    dist = max(0.0, dist)
                    if dist <= unit.sight_range + unit.radius and dist < best_dist - 1e-6:
                        best_dist = dist
                        best_unit = enemy
        return best_unit

    def _refresh_unit_target(self, unit: Unit, buckets: Dict[Tuple[int, int], List[Unit]], cell: float = 2.0) -> None:
        if not getattr(unit, "active", True):
            self._clear_unit_target(unit)
            unit._retarget_timer = unit.retarget_cooldown
            return
        unit._retarget_timer = max(0.0, getattr(unit, "_retarget_timer", 0.0) - self.dt)
        current = self._get_unit_target(unit)
        if current is not None and not self._target_within_sight(unit, current):
            self._clear_unit_target(unit)
            current = None
            unit._retarget_timer = 0.0
        if unit._retarget_timer > 0.0 and current is not None:
            return
        new_target = self._find_best_target(unit, buckets, cell)
        if new_target is None:
            self._clear_unit_target(unit)
            unit._retarget_timer = unit.retarget_cooldown
            return
        self._set_unit_target(unit, new_target)
        unit._retarget_timer = unit.retarget_cooldown

    def _build_spatial_buckets(self, cell: float = 2.0) -> Dict[Tuple[int, int], List[Unit]]:
        buckets: Dict[Tuple[int, int], List[Unit]] = {}
        for unit in self.units:
            if not unit.alive:
                continue
            key = (int(unit.x // cell), int(unit.y // cell))
            buckets.setdefault(key, []).append(unit)
        return buckets

    def _resolve_unit_collisions(self, unit: Unit, buckets: Dict[Tuple[int, int], List[Unit]], cell: float = 2.0) -> None:
        building_only = unit.targeting == "buildings"
        key_x = int(unit.x // cell)
        key_y = int(unit.y // cell)
        for ny in range(key_y - 1, key_y + 2):
            for nx in range(key_x - 1, key_x + 2):
                for other in buckets.get((nx, ny), ()):
                    if other is unit or not other.alive:
                        continue
                    if getattr(other, '_attached_to', None) is not None or getattr(unit, '_attached_to', None) is not None:
                        continue
                    if other.flying != unit.flying:
                        continue
                    dist = math.hypot(unit.x - other.x, unit.y - other.y)
                    min_sep = unit.radius + other.radius
                    if other.owner == unit.owner:
                        overlap = min_sep - dist
                        if overlap > 0.0:
                            unit._needs_detour = True
                            other._needs_detour = True
                            if building_only:
                                unit._path_refresh_timer = 0.0
                            if dist <= 1e-6:
                                nx, ny = 0.0, 0.0
                            else:
                                nx = (unit.x - other.x) / dist
                                ny = (unit.y - other.y) / dist
                            if nx == 0.0 and ny == 0.0:
                                nx, ny = 1.0, 0.0
                            push_factor = max(unit.radius, other.radius)
                            separation = (overlap + 0.05) * (0.35 + 0.45 * push_factor)
                            unit.x += nx * separation
                            unit.y += ny * separation
                            if getattr(other, '_attached_to', None) is None:
                                other.x -= nx * separation * 0.6
                                other.y -= ny * separation * 0.6
                        continue
                    if building_only:
                        if dist < min_sep:
                            unit._needs_detour = True
                            unit._path_refresh_timer = 0.0
                        continue
                    if dist < min_sep:
                        unit._path = []
                        return
    def _unit_can_fire_projectile(self, unit: Unit) -> bool:
        if unit.projectile_speed > 0.0 or unit.projectile_radius > 0.0:
            return True
        if unit.projectile_key and unit.range > MELEE_RANGE + 0.1:
            return True
        return False

    def _projectile_launch_position(self, unit: Unit, target_x: float, target_y: float) -> Tuple[float, float]:
        offset = float(getattr(unit, "_projectile_offset", 0.0) or 0.0)
        if offset <= 0.0:
            return unit.x, unit.y
        dx = target_x - unit.x
        dy = target_y - unit.y
        dist = math.hypot(dx, dy)
        if dist <= 1e-6:
            return unit.x, unit.y
        return unit.x + dx / dist * offset, unit.y + dy / dist * offset

    def _should_hold_position(self, unit: Unit, target: Optional[Union[Unit, Dict[str, Any]]]) -> bool:
        if target is None or not self._unit_can_fire_projectile(unit):
            return False
        if not self._unit_can_target_enemy(unit, target):
            return False
        point, _category, radius = self._target_point_category(unit, target)
        return self._target_in_attack_range(unit, point, radius)

    def _try_attack(self, unit: Unit, target: Optional[Union[Unit, Dict[str, Any]]]) -> bool:
        if not getattr(unit, "active", True):
            unit._pending_attack_signature = None
            return False
        if target is None:
            unit._pending_attack_signature = None
            return False
        if getattr(unit, "_stun_timer", 0.0) > 0.0:
            return False
        if not self._unit_can_target_enemy(unit, target):
            return False

        point, category, target_radius = self._target_point_category(unit, target)
        if not self._target_in_attack_range(unit, point, target_radius):
            return False

        speed_factor = max(0.05, getattr(unit, "_rage_multiplier", 1.0) * getattr(unit, "_slow_multiplier", 1.0))
        base_hit_speed = max(unit.hit_speed, MIN_HIT_SPEED)
        base_load_time = max(unit._load_time, 0.0)
        load_time = base_load_time / speed_factor
        cycle_time = base_hit_speed / speed_factor
        recovery_time = max(cycle_time - load_time, 0.0)

        unit._attack_cd = max(0.0, unit._attack_cd - self.dt)
        signature = self._target_signature(target)
        pending_sig = unit._pending_attack_signature

        if pending_sig is not None:
            if signature != pending_sig:
                if unit._load_after_retarget and load_time > 0.0:
                    unit._windup_timer = load_time
                unit._pending_attack_signature = signature
            if unit._windup_timer > 1e-6:
                return False
            return self._complete_attack(unit, target, signature, point, category, recovery_time, load_time)

        if unit._windup_timer > 1e-6:
            return False
        if unit._attack_cd > 1e-6:
            return False

        unit._pending_attack_signature = signature
        if unit._first_attack_pending:
            unit._first_attack_pending = False
        if load_time > 1e-6:
            unit._windup_timer = load_time
            return False
        return self._complete_attack(unit, target, signature, point, category, recovery_time, load_time)

    def _complete_attack(
        self,
        unit: Unit,
        target: Union[Unit, Dict[str, Any]],
        signature: str,
        point: Tuple[float, float],
        category: str,
        recovery_time: float,
        load_time: float,
    ) -> bool:
        if isinstance(target, Unit) and not target.alive:
            unit._pending_attack_signature = None
            unit._attack_cd = 0.0
            return False

        projectile_capable = self._unit_can_fire_projectile(unit)
        proj_def = get_projectile(unit.projectile_key) if unit.projectile_key else None
        proj_speed = unit.projectile_speed if unit.projectile_speed > 0.0 else (
            proj_def.speed if proj_def and proj_def.speed > 0.0 else 8.0
        )
        proj_hit_radius = unit.projectile_hit_radius if unit.projectile_hit_radius > 0.0 else (
            proj_def.hit_radius if proj_def and proj_def.hit_radius > 0.0 else 0.25
        )
        proj_area_radius = unit.projectile_area_radius if unit.projectile_area_radius > 0.0 else unit.projectile_radius
        if proj_def and proj_def.area_radius > 0.0:
            proj_area_radius = max(proj_area_radius, proj_def.area_radius)
        hits_air = proj_def.hits_air if proj_def else unit.projectile_hits_air
        hits_ground = proj_def.hits_ground if proj_def else unit.projectile_hits_ground
        pushback = proj_def.pushback if proj_def else 0.0
        proj_lifetime = unit.projectile_lifetime if unit.projectile_lifetime > 0.0 else (
            proj_def.lifetime if proj_def and proj_def.lifetime > 0.0 else 0.0
        )
        proj_homing = proj_def.homing if proj_def else True

        damage = unit.damage
        if getattr(unit, "_charge_active", False):
            damage *= unit.charge_damage_mult
        if category == "buildings":
            damage *= unit.tower_damage_multiplier

        start_z = float(getattr(unit, "_projectile_start_z", 0.0))

        if projectile_capable:
            start_x, start_y = self._projectile_launch_position(unit, point[0], point[1])
            projectile = Projectile(
                owner=unit.owner,
                x=start_x,
                y=start_y,
                speed=max(0.01, proj_speed),
                damage=damage,
                hit_radius=proj_hit_radius,
                area_radius=proj_area_radius,
                hits_air=hits_air,
                hits_ground=hits_ground,
                source=("unit", unit.name),
                target_unit=target if isinstance(target, Unit) else None,
                target_structure=target if isinstance(target, dict) else None,
                pushback=pushback,
                z=start_z,
                lifetime=proj_lifetime,
                homing=proj_homing,
            )
            self.projectiles.append(projectile)
        else:
            self._apply_melee_damage(unit, target, damage)

        if unit.slow_effect:
            self._apply_slow_effect(unit, target, unit.slow_effect)
        if unit.chain_config:
            self._apply_chain_damage(unit, target, unit.chain_config, damage)
        if getattr(unit, "charge_reset_on_hit", True):
            self._reset_charge(unit)

        unit._attack_cd = recovery_time
        unit._pending_attack_signature = None
        unit._last_target_signature = signature
        unit._windup_timer = 0.0
        return True

    def _closest_point_on_rect(self, x: float, y: float, rect: Dict[str, Any]) -> Tuple[float, float]:
        x0, y0, w, h = rect["x0"], rect["y0"], rect["width"], rect["height"]
        rx0, rx1 = x0, x0 + w
        ry0, ry1 = y0, y0 + h
        cx = clamp(x, rx0, rx1)
        cy = clamp(y, ry0, ry1)
        if rx0 < cx < rx1:
            cy = ry0 if abs(y - ry0) < abs(y - ry1) else ry1
        elif ry0 < cy < ry1:
            cx = rx0 if abs(x - rx0) < abs(x - rx1) else rx1
        return cx, cy

    def _find_unit_by_id(self, uid: Optional[int]) -> Optional[Unit]:
        if uid is None:
            return None
        for unit in self.units:
            if unit.id == uid:
                return unit
        return None

    def _find_tower_by_label(self, label: Optional[str]) -> Optional[Dict[str, Any]]:
        if not label:
            return None
        for tower in self.towers:
            if tower.get("label") == label:
                return tower
        return None

    def deal_area(
        self,
        owner: int,
        x: float,
        y: float,
        radius: float,
        damage: float,
        hits_air: bool,
        hits_ground: bool,
        source: Tuple[str, str],
        *,
        tower_multiplier: float = 1.0,
    ) -> None:
        if damage <= 0.0:
            return
        for unit in self.units:
            if not unit.alive or unit.owner == owner:
                continue
            if unit.flying and not hits_air:
                continue
            if (not unit.flying) and not hits_ground:
                continue
            if distance_tiles((unit.x, unit.y), (x, y)) <= radius + unit.radius:
                unit.take_damage(damage, owner, source)
        for tower in self.towers:
            if not tower.get("alive", True) or tower.get("owner") == owner:
                continue
            if distance_tiles((tower["cx"], tower["cy"]), (x, y)) <= radius + tower["radius"]:
                self.damage_structure(tower, damage * tower_multiplier, owner, source)

    def apply_stun_area(self, owner: int, x: float, y: float, radius: float, duration: float) -> None:
        area = max(radius, DEFAULT_SMALL_RADIUS)
        for unit in self.units:
            if not unit.alive or unit.owner == owner:
                continue
            if distance_tiles((unit.x, unit.y), (x, y)) <= area + unit.radius:
                unit.apply_stun(duration)
        for tower in self.towers:
            if not tower.get("alive", True) or tower.get("owner") == owner:
                continue
            if distance_tiles((tower["cx"], tower["cy"]), (x, y)) <= area + tower["radius"]:
                tower["stun_timer"] = max(tower.get("stun_timer", 0.0), duration)

    def damage_structure(self, structure: Dict[str, Any], amount: float, attacker_owner: Optional[int], source: Tuple[str, str]) -> None:
        if structure is None or not structure.get("alive", True):
            return
        structure["hp"] = max(0.0, structure.get("hp", 0.0) - amount)
        if structure.get("type") == "king" and not structure.get("active", True):
            self.arena.activate_king_tower(structure)
            structure["active"] = True
        self._sync_tower_unit_state(structure)
        if structure["hp"] <= 0 and structure.get("alive", True):
            structure["alive"] = False
            structure["target"] = None
            self.arena.on_tower_destroyed(structure)
            self._handle_tower_destroyed(structure, attacker_owner)

    def _handle_tower_destroyed(self, tower: Dict[str, Any], attacker_owner: Optional[int]) -> None:
        if attacker_owner in (1, 2):
            (self.p1 if attacker_owner == 1 else self.p2).crowns += 1
        if tower.get("type") == "king":
            self.over = True
            self.winner = attacker_owner
        tower["alive"] = False
        self._sync_tower_unit_state(tower)

    def cast_spell(self, owner: int, card_name: str, card: Dict[str, Any], tile_x: int, tile_y: int) -> None:
        wx = clamp(tile_x + 0.5, 0.0, self.arena.width - 1e-3)
        wy = clamp(tile_y + 0.5, 0.0, self.arena.height - 1e-3)
        radius = float(card.get("radius", 0.0) or 0.0)
        damage = float(card.get("dmg", card.get("damage", 0.0)) or 0.0)
        duration = float(card.get("duration", 0.0) or 0.0)
        tick_dps = float(card.get("tick_dps", card.get("damage_per_second", 0.0)) or 0.0)
        speed_mult = float(card.get("speed_multiplier", 0.0) or 0.0)
        linger = float(card.get("linger", card.get("speed_linger", 0.0)) or 0.0)
        stun = float(card.get("stun_duration", 0.0) or 0.0)

        cname = card_name.lower()
        if tick_dps <= 0.0 and cname == "poison":
            tick_dps = DEFAULT_POISON_DPS
        if duration <= 0.0 and cname == "poison":
            duration = DEFAULT_POISON_DURATION
        if speed_mult <= 0.0 and cname == "rage":
            speed_mult = DEFAULT_RAGE_SPEED_MULTIPLIER
        if duration <= 0.0 and speed_mult > 1.0:
            duration = DEFAULT_RAGE_DURATION
        if linger <= 0.0 and speed_mult > 1.0:
            linger = DEFAULT_RAGE_LINGER
        if stun <= 0.0 and cname == "zap":
            stun = DEFAULT_ZAP_STUN

        projectile_key = card.get("projectile")
        if not projectile_key and cname == "arrows":
            projectile_key = DEFAULT_ARROWS_PROJECTILE
        proj_def: Optional[ProjectileDef] = get_projectile(projectile_key)

        if proj_def:
            radius = max(radius, proj_def.area_radius)
            if damage <= 0.0:
                damage = proj_def.damage
            proj_speed = float(card.get("speed", 0.0) or 0.0)
            if proj_speed <= 0.0:
                proj_speed = proj_def.speed
            hits_ground = proj_def.hits_ground
            hits_air = proj_def.hits_air
            pushback = proj_def.pushback
            hit_radius = proj_def.hit_radius if proj_def.hit_radius > 0.0 else max(radius * 0.6, 0.3)
            proj_lifetime = proj_def.lifetime
            proj_homing = proj_def.homing
        else:
            proj_speed = float(card.get("speed", 0.0) or 0.0)
            hits_ground = True
            hits_air = True
            pushback = 0.0
            hit_radius = radius * 0.6 if radius > 0.0 else 0.3
            proj_lifetime = 0.0
            proj_homing = False

        area_radius = radius if radius > 0.0 else (proj_def.area_radius if proj_def else 0.0)
        if proj_def or proj_speed > 0.0:
            start_y = wy + (6.0 if owner == 1 else -6.0)
            projectile = Projectile(
                owner=owner,
                x=wx,
                y=start_y,
                speed=max(0.1, proj_speed if proj_speed > 0.0 else 8.0),
                damage=damage,
                hit_radius=hit_radius,
                area_radius=area_radius,
                hits_air=hits_air,
                hits_ground=hits_ground,
                source=("spell", card_name),
                target_pos=(wx, wy),
                pushback=pushback,
                lifetime=proj_lifetime,
                homing=proj_homing,
            )
            self.projectiles.append(projectile)
        else:
            fallback_radius = area_radius if area_radius > 0.0 else DEFAULT_SMALL_RADIUS
            if damage > 0.0:
                self.deal_area(owner, wx, wy, fallback_radius, damage, True, True, ("spell", card_name))

        if stun > 0.0:
            stun_radius = area_radius if area_radius > 0.0 else DEFAULT_SMALL_RADIUS
            self.apply_stun_area(owner, wx, wy, stun_radius, stun)
        if tick_dps > 0.0 and duration > 0.0:
            poison_radius = area_radius if area_radius > 0.0 else 3.5
            self.effects.append(PoisonCloud(owner, wx, wy, poison_radius, duration, tick_dps, card_name))
        if speed_mult > 1.0 and duration > 0.0:
            rage_radius = area_radius if area_radius > 0.0 else 5.0
            self.effects.append(RageZone(owner, wx, wy, rage_radius, duration, speed_mult, linger, card_name))

    def validate_dps(self, card_name: str, *, level: Optional[int] = None, duration: float = 6.0) -> Dict[str, float]:
        card = get_card(card_name, level=level)
        if not card:
            raise ValueError(f"unknown card: {card_name}")
        master_card = self._lookup_master_card(card_name, card)
        spec = self._extract_unit_stats(card, card_name)
        unit = self._create_unit(
            owner=1,
            card_key=card_name,
            name=card.get("name", card_name),
            level=card.get("level", level or card.get("_default_level", 11)),
            card=card,
            spec=spec,
            x=8.5,
            y=16.0,
            master_card=master_card,
        )
        if unit is None:
            raise ValueError(f"unable to build unit for {card_name}")
        unit.active = True
        unit.deploy_cooldown = 0.0
        unit._deploy_complete_fired = True
        unit._retarget_timer = 0.0
        unit._attack_cd = 0.0
        unit._windup_timer = 0.0
        unit._stun_timer = 0.0
        unit._slow_timer = 0.0
        unit._slow_multiplier = 1.0
        unit._rage_multiplier = 1.0
        unit._last_progress_pos = (unit.x, unit.y)

        target_radius = 0.45
        min_gap = unit.radius + target_radius + 0.05
        close_distance = max(min_gap, 0.6)
        if unit.range > close_distance + 0.1:
            target_distance = close_distance
        else:
            target_distance = max(min_gap, unit.range - 0.1)
        target = Unit(
            id=-999999,
            owner=2,
            name="DPS Dummy",
            x=unit.x + max(target_distance, unit.radius + 0.4),
            y=unit.y,
            hp=1_000_000.0,
            hp_max=1_000_000.0,
            speed=0.0,
            range=0.0,
            hit_speed=1.0,
            damage=0.0,
            targeting="ground",
            flying=False,
            radius=0.45,
            aggro_range=0.0,
        )
        target.active = True
        target.deploy_cooldown = 0.0
        target._deploy_complete_fired = True

        original_units = self.units
        original_projectiles = self.projectiles
        original_effects = self.effects
        original_towers = self.towers
        original_time = self.time
        original_unit_id = self._unit_id_seq
        attack_times: List[float] = []
        damage_before = target.hp
        try:
            self.units = [unit, target]
            self.projectiles = []
            self.effects = []
            self.towers = []
            min_duration = unit._load_time + max(unit.hit_speed * 3.0, 1.0)
            ticks = max(1, int(max(duration, min_duration) / self.dt))
            for step in range(ticks):
                self._update_unit_status(unit)
                self._update_unit_status(target)
                if self._try_attack(unit, target):
                    attack_times.append(step * self.dt)
                self._update_projectiles()
                if not target.alive:
                    break
            # Allow outstanding projectiles to resolve.
            if self.projectiles:
                flush_steps = int(1.0 / self.dt)
                for _ in range(flush_steps):
                    self._update_projectiles()
                    if not self.projectiles:
                        break
            elapsed = ticks * self.dt
            damage_done = damage_before - target.hp
            if len(attack_times) >= 2 and damage_done > 0:
                intervals = [
                    attack_times[i + 1] - attack_times[i] for i in range(len(attack_times) - 1)
                ]
                avg_interval = sum(intervals) / len(intervals) if intervals else unit.hit_speed
                avg_interval = max(avg_interval, 1e-6)
                avg_damage = damage_done / len(attack_times)
                simulated_dps = avg_damage / avg_interval
            else:
                simulated_dps = damage_done / elapsed if elapsed > 0 else 0.0
        finally:
            self.units = original_units
            self.projectiles = original_projectiles
            self.effects = original_effects
            self.towers = original_towers
            self.time = original_time
            self._unit_id_seq = original_unit_id

        expected_damage = float(getattr(unit, "damage", 0.0) or 0.0)
        expected_hit_speed = float(getattr(unit, "hit_speed", 1.0) or 1.0)
        expected_dps = expected_damage / expected_hit_speed if expected_hit_speed > 0 else 0.0
        delta_pct = None
        if expected_dps:
            delta_pct = (simulated_dps - expected_dps) / expected_dps * 100.0
        return {
            "card": card_name,
            "level": card.get("level", level or card.get("_default_level", 11)),
            "simulated_dps": simulated_dps,
            "expected_dps": expected_dps,
            "delta_pct": delta_pct,
        }
    def _finalize_match(self) -> None:
        if self.p1.crowns > self.p2.crowns:
            self.winner = 1
        elif self.p2.crowns > self.p1.crowns:
            self.winner = 2
        else:
            p1_damage = sum((tw["hp_max"] - tw.get("hp", 0)) for tw in self.towers if tw.get("owner") == 2)
            p2_damage = sum((tw["hp_max"] - tw.get("hp", 0)) for tw in self.towers if tw.get("owner") == 1)
            if p1_damage > p2_damage:
                self.winner = 1
            elif p2_damage > p1_damage:
                self.winner = 2
            else:
                self.winner = 0
        self.over = True


__all__ = ["Engine", "Unit", "Player"]
