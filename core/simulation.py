from __future__ import annotations

"""Core Clash Royale simulation loop: troops, towers, projectiles, spells."""

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

from .rules import (
    TIME_STEP,
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

DEFAULT_TOWER_STATS: Dict[str, Dict[str, float]] = {
    "king": {"hp": 4300.0, "damage": 240.0, "range": 7.0, "hit_speed": 1.0, "projectile_speed": 6.0, "damage_modifier": 1.0},
    "princess": {"hp": 2534.0, "damage": 130.0, "range": 8.0, "hit_speed": 0.8, "projectile_speed": 6.0, "damage_modifier": 1.0},
}

def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))

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
    charge_windup: float = 0.0
    charge_speed_mult: float = 1.0
    charge_damage_mult: float = 1.0
    charge_reset_on_hit: bool = True
    spawn_effect: Optional[Dict[str, Any]] = None
    slow_effect: Optional[Dict[str, Any]] = None
    chain_config: Optional[Dict[str, Any]] = None
    support_units: Optional[List[Dict[str, Any]]] = None
    death_spawn_config: Optional[Dict[str, Any]] = None

    _attack_cd: float = 0.0
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
class Player:
    idx: int
    elixir: float = 5.0
    crowns: int = 0
    pending_spawns: List[Tuple[str, int, int, Optional[int]]] = field(default_factory=list)
    card_levels: Dict[str, int] = field(default_factory=dict)

class SpellEffect:
    alive: bool = True

    def tick(self, engine: "Engine", dt: float) -> None:
        raise NotImplementedError


class PoisonCloud(SpellEffect):
    def __init__(self, owner: int, x: float, y: float, radius: float, duration: float, dps: float, label: str) -> None:
        self.owner = owner
        self.x = x
        self.y = y
        self.radius = radius
        self.remaining = duration
        self.dps = dps
        self.label = label

    def tick(self, engine: "Engine", dt: float) -> None:
        if not self.alive:
            return
        self.remaining -= dt
        damage = self.dps * dt
        if damage > 0.0:
            engine.deal_area(self.owner, self.x, self.y, self.radius, damage, True, True, ("spell", self.label))
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
        "speed",
        "damage",
        "radius",
        "hits_air",
        "hits_ground",
        "source",
        "target_unit",
        "target_structure",
        "target_pos",
        "pushback",
        "alive",
    )

    def __init__(
        self,
        owner: int,
        x: float,
        y: float,
        speed: float,
        damage: float,
        radius: float,
        hits_air: bool,
        hits_ground: bool,
        source: Tuple[str, str],
        target_unit: Optional[Unit] = None,
        target_structure: Optional[Dict[str, Any]] = None,
        target_pos: Optional[Tuple[float, float]] = None,
        pushback: float = 0.0,
    ) -> None:
        self.owner = owner
        self.x = x
        self.y = y
        self.speed = max(0.01, speed)
        self.damage = damage
        self.radius = max(0.0, radius)
        self.hits_air = hits_air
        self.hits_ground = hits_ground
        self.source = source
        self.target_unit = target_unit
        self.target_structure = target_structure
        self.target_pos = target_pos
        self.pushback = pushback
        self.alive = True

    def tick(self, engine: "Engine", dt: float) -> None:
        if not self.alive:
            return
        pos = self._target_position()
        if pos is None:
            self.alive = False
            return
        dx = pos[0] - self.x
        dy = pos[1] - self.y
        dist = math.hypot(dx, dy)
        if dist <= 1e-6 or dist <= self.speed * dt:
            self.x, self.y = pos
            self._on_hit(engine)
            self.alive = False
            return
        step = self.speed * dt
        self.x += dx / dist * step
        self.y += dy / dist * step

    def _target_position(self) -> Optional[Tuple[float, float]]:
        if self.target_unit is not None and self.target_unit.alive:
            return (self.target_unit.x, self.target_unit.y)
        if self.target_structure is not None and self.target_structure.get("alive", True):
            cx = self.target_structure.get("cx")
            cy = self.target_structure.get("cy")
            if cx is None or cy is None:
                cx, cy = self._structure_center(self.target_structure)
            return (float(cx), float(cy))
        return self.target_pos

    def _structure_center(self, structure: Dict[str, Any]) -> Tuple[float, float]:
        x0 = float(structure.get("x0", 0.0))
        y0 = float(structure.get("y0", 0.0))
        w = float(structure.get("width", 1.0))
        h = float(structure.get("height", 1.0))
        return (x0 + w / 2.0, y0 + h / 2.0)

    def _on_hit(self, engine: "Engine") -> None:
        if self.radius > 0.0:
            engine.deal_area(self.owner, self.x, self.y, self.radius, self.damage, self.hits_air, self.hits_ground, self.source)
            return
        if self.target_unit is not None and self.target_unit.alive:
            if (self.target_unit.flying and self.hits_air) or ((not self.target_unit.flying) and self.hits_ground):
                self.target_unit.take_damage(self.damage, self.owner, self.source)
            return
        if self.target_structure is not None and self.target_structure.get("alive", True):
            engine.damage_structure(self.target_structure, self.damage, self.owner, self.source)
            return
        engine.deal_area(self.owner, self.x, self.y, DEFAULT_SMALL_RADIUS, self.damage, self.hits_air, self.hits_ground, self.source)

def _unit_apply_stun(self: Unit, duration: float) -> None:
    if duration <= 0.0:
        return
    self._stun_timer = max(getattr(self, "_stun_timer", 0.0), duration)
    self._attack_cd = max(self._attack_cd, duration)


Unit.apply_stun = _unit_apply_stun


def _unit_apply_rage(self: Unit, multiplier: float, duration: float, linger: float) -> None:
    self._rage_multiplier = max(getattr(self, "_rage_multiplier", 1.0), multiplier)
    self._rage_timer = max(getattr(self, "_rage_timer", 0.0), duration)
    self._rage_linger = max(getattr(self, "_rage_linger", 0.0), linger)


def _unit_take_damage(self: Unit, amount: float, source_owner: Optional[int], source: Tuple[str, str]) -> None:
    if not self.alive:
        return
    self.hp -= amount
    if self.hp <= 0.0 and self.alive:
        self.alive = False


Unit.apply_rage = _unit_apply_rage
Unit.apply_slow = _unit_apply_slow
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

        self.towers = self.arena.towers
        self.dt = TIME_STEP
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
    ) -> bool:
        pending = self.p1.pending_spawns if player_idx == 1 else self.p2.pending_spawns
        pending.append((card_key, tile_x, tile_y, level))
        return True

    def list_units(self) -> List[Unit]:
        return [u for u in self.units if u.alive]

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
        kept: List[Tuple[str, int, int, Optional[int]]] = []
        for entry in player.pending_spawns:
            card_key, tx, ty, *maybe_level = entry
            level_override = maybe_level[0] if maybe_level else None
            if not self.arena.is_deploy_legal(player.idx, tx, ty):
                continue
            desired_level = level_override if level_override is not None else player.card_levels.get(card_key)
            card = get_card(card_key, level=desired_level)
            if not card:
                continue
            cost = int(card.get("cost", 0))
            actual_level = int(card.get("level", desired_level or 1))
            if player.elixir + 1e-6 < cost:
                kept.append((card_key, tx, ty, actual_level))
                continue
            self._spawn_card(player.idx, card_key, card, tx, ty, actual_level)
            player.elixir -= cost
            player.card_levels[card_key] = actual_level
        player.pending_spawns = kept


    def _spawn_card(
        self,
        owner: int,
        card_key: str,
        card: Dict[str, Any],
        tx: int,
        ty: int,
        level: int,
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
            )
            if unit is None:
                continue
            unit._attach_offset = (dx, dy)
            self.units.append(unit)
            self._unit_id_seq += 1
            self._on_unit_spawn(unit, card)
            placed += 1

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
        projectile_speed = float(card.get("projectile_speed", 0.0))
        projectile_radius = float(card.get("splash_radius", 0.0))
        projectile_hits_air = bool(card.get("projectile_hits_air", "air" in target_flags))
        projectile_hits_ground = bool(
            card.get("projectile_hits_ground", any(t in ("ground", "buildings") for t in target_flags))
        )
        splash_radius = float(card.get("splash_radius", 0.0))
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
            "projectile_radius": projectile_radius,
            "projectile_hits_air": projectile_hits_air,
            "projectile_hits_ground": projectile_hits_ground,
            "target_flags": target_flags,
            "splash_radius": splash_radius,
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
    ) -> Optional[Unit]:
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
            projectile_hits_air=spec["projectile_hits_air"],
            projectile_hits_ground=spec["projectile_hits_ground"],
            projectile_key=card.get("projectile"),
            card_key=card_key,
            base_speed=spec["speed"],
            splash_radius=spec.get("splash_radius", 0.0),
        )
        if attached_to is not None:
            unit._attached_to = attached_to
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
    ) -> None:
        card = get_card(name, level=level)
        if not card:
            return
        spec = self._extract_unit_stats(card, name)
        for idx in range(max(1, count)):
            if follow_parent and attached_to is not None and count > 1:
                offset_x = (idx - (count - 1) / 2.0) * spread
                offset_y = 0.0
            elif follow_parent and attached_to is not None:
                offset_x = 0.0
                offset_y = 0.0
            else:
                angle = random.random() * 2.0 * math.pi
                offset_x = math.cos(angle) * spread
                offset_y = math.sin(angle) * spread
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
            )
            if unit is None:
                continue
            if follow_parent and attached_to is not None:
                unit._attached_to = attached_to
                unit._attach_offset = (offset_x, offset_y)
                unit.radius = min(unit.radius, 0.2)
            self.units.append(unit)
            if follow_parent and attached_to is not None:
                attached_to._support_children.append(unit.id)
            self._unit_id_seq += 1
            self._on_unit_spawn(unit, card)
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

    def _tower_acquire_target(self, tower: Dict[str, Any]) -> Optional[Unit]:
        owner = tower.get("owner")
        best_unit = None
        best_score = float("inf")
        for unit in self.units:
            if unit.owner == owner or not unit.alive:
                continue
            if not self._tower_can_target(tower, unit):
                continue
            dist = distance_tiles((tower["cx"], tower["cy"]), (unit.x, unit.y))
            bias = -0.25 if unit.targeting == "buildings" else 0.0
            score = dist + bias
            if score < best_score:
                best_score = score
                best_unit = unit
        return best_unit

    def _fire_tower(self, tower: Dict[str, Any], target: Unit) -> None:
        damage = tower.get("damage", 0.0) * tower.get("damage_modifier", 1.0)
        projectile_speed = tower.get("projectile_speed", 6.0)
        if projectile_speed <= 0.0:
            target.take_damage(damage, tower.get("owner"), ("tower", tower.get("label", "tower")))
            return
        projectile = Projectile(
            owner=tower.get("owner", 1),
            x=tower["cx"],
            y=tower["cy"],
            speed=projectile_speed,
            damage=damage,
            radius=0.0,
            hits_air=True,
            hits_ground=True,
            source=("tower", tower.get("label", "tower")),
            target_unit=target,
        )
        self.projectiles.append(projectile)

    def _update_towers(self) -> None:
        for tower in self.towers:
            if not tower.get('alive', True):
                continue
            if tower.get('stun_timer', 0.0) > 0.0:
                tower['stun_timer'] = max(0.0, tower['stun_timer'] - self.dt)
                continue
            slow_timer = tower.get('slow_timer', 0.0)
            if slow_timer > 0.0:
                slow_timer = max(0.0, slow_timer - self.dt)
                tower['slow_timer'] = slow_timer
                if slow_timer <= 0.0:
                    tower['slow_multiplier'] = 1.0
            slow_multiplier = max(0.05, tower.get('slow_multiplier', 1.0))
            if not tower.get('active', True):
                continue
            tower['cooldown'] = max(0.0, tower.get('cooldown', 0.0) - self.dt * slow_multiplier)
            target = tower.get('target')
            if target is not None and (not target.alive or not self._tower_can_target(tower, target)):
                target = None
            if target is None:
                target = self._tower_acquire_target(tower)
                tower['target'] = target
            if target is None:
                continue
            if tower['cooldown'] > 1e-6:
                continue
            self._fire_tower(tower, target)
            effective_hit_speed = tower.get('hit_speed', 1.0) / slow_multiplier
            tower['cooldown'] = max(0.0, effective_hit_speed)
    def _update_units(self) -> None:
        alive_units = [u for u in self.units if u.alive]
        for unit in alive_units:
            self._update_unit_status(unit)
        self._maintain_support_positions(alive_units)

        u1 = [u for u in alive_units if u.owner == 1]
        u2 = [u for u in alive_units if u.owner == 2]

        for unit in alive_units:
            enemies = u2 if unit.owner == 1 else u1
            target_dict = self._choose_target_object(unit, enemies)
            if getattr(unit, "_attached_to", None) is None:
                self._update_unit_path(unit, target_dict)
            else:
                unit._path = []

        buckets = self._build_spatial_buckets()
        for unit in alive_units:
            if not unit.alive:
                continue
            target = self._choose_target_object(unit, u2 if unit.owner == 1 else u1)
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
            return
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

    def _nearest_enemy_structure(self, unit: Unit) -> Dict[str, Any]:
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
        return best or {"pos": (unit.x, unit.y)}

    def _update_unit_path(self, unit: Unit, target: Dict[str, Any]) -> None:
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
        if needs_new:
            unit._path = compute_path(self.arena, unit.__dict__, target)
            unit._path_goal_id = goal_id
            self._reset_charge(unit)

    def _path_head_walkable(self, unit: Unit) -> bool:
        if not unit._path:
            return True
        nx, ny = unit._path[0]
        return self.arena.is_walkable(unit.owner, nx, ny, flying=unit.flying)

    def _move_along_path(self, unit: Unit) -> None:
        if getattr(unit, '_attached_to', None) is not None:
            return
        if getattr(unit, '_stun_timer', 0.0) > 0.0:
            self._reset_charge(unit)
            return
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

    def _build_spatial_buckets(self, cell: float = 2.0) -> Dict[Tuple[int, int], List[Unit]]:
        buckets: Dict[Tuple[int, int], List[Unit]] = {}
        for unit in self.units:
            if not unit.alive:
                continue
            key = (int(unit.x // cell), int(unit.y // cell))
            buckets.setdefault(key, []).append(unit)
        return buckets

    def _resolve_unit_collisions(self, unit: Unit, buckets: Dict[Tuple[int, int], List[Unit]], cell: float = 2.0) -> None:
        key_x = int(unit.x // cell)
        key_y = int(unit.y // cell)
        for ny in range(key_y - 1, key_y + 2):
            for nx in range(key_x - 1, key_x + 2):
                for other in buckets.get((nx, ny), ()):
                    if other is unit or not other.alive:
                        continue
                    if getattr(other, '_attached_to', None) is not None or getattr(unit, '_attached_to', None) is not None:
                        continue
                    if other.owner == unit.owner:
                        continue
                    if other.flying != unit.flying:
                        continue
                    if distance_tiles((unit.x, unit.y), (other.x, other.y)) < (unit.radius + other.radius):
                        unit._path = []
                        return
    def _choose_target_object(self, unit: Unit, enemies: List[Unit]) -> Dict[str, Any]:
        best = None
        best_dist = float("inf")
        for enemy in enemies:
            if not enemy.alive:
                continue
            dist = distance_tiles((unit.x, unit.y), (enemy.x, enemy.y))
            if dist > unit.aggro_range:
                continue
            if enemy.flying and not unit.can_attack_air():
                continue
            if (not enemy.flying) and not unit.can_attack_ground():
                continue
            if dist < best_dist:
                best_dist = dist
                best = enemy
        if best is not None:
            return best.__dict__
        return self._nearest_enemy_structure(unit)

    def _unit_can_fire_projectile(self, unit: Unit) -> bool:
        if unit.projectile_speed > 0.0 or unit.projectile_radius > 0.0:
            return True
        if unit.projectile_key and unit.range > MELEE_RANGE + 0.1:
            return True
        return False

    def _should_hold_position(self, unit: Unit, target: Dict[str, Any]) -> bool:
        if not target or not self._unit_can_fire_projectile(unit):
            return False
        if "x" in target and "y" in target:
            tx, ty = float(target["x"]), float(target["y"])
            is_air = bool(target.get("flying", False))
            if (is_air and not unit.can_attack_air()) or ((not is_air) and not unit.can_attack_ground()):
                return False
            return unit.within_attack_range(tx, ty)
        if "x0" in target and "y0" in target:
            cx, cy = self._closest_point_on_rect(unit.x, unit.y, target)
            return unit.within_attack_range(cx, cy)
        return False

    def _try_attack(self, unit: Unit, target: Dict[str, Any]) -> bool:
        unit._attack_cd = max(0.0, unit._attack_cd - self.dt)
        if unit._attack_cd > 1e-6 or getattr(unit, '_stun_timer', 0.0) > 0.0:
            return False
        projectile_capable = self._unit_can_fire_projectile(unit)
        damage = unit.damage
        if getattr(unit, '_charge_active', False):
            damage *= unit.charge_damage_mult
        if "x" in target and "y" in target:
            tx, ty = float(target["x"]), float(target["y"])
            is_air = bool(target.get('flying', False))
            if (is_air and not unit.can_attack_air()) or ((not is_air) and not unit.can_attack_ground()):
                return False
            if not (unit.within_melee(tx, ty) or unit.within_attack_range(tx, ty)):
                return False
            victim = self._find_unit_by_id(target.get('id'))
            if not victim or not victim.alive:
                return False
            if projectile_capable:
                speed = unit.projectile_speed if unit.projectile_speed > 0.0 else 8.0
                projectile = Projectile(
                    owner=unit.owner,
                    x=unit.x,
                    y=unit.y,
                    speed=speed,
                    damage=damage,
                    radius=unit.projectile_radius,
                    hits_air=unit.projectile_hits_air,
                    hits_ground=unit.projectile_hits_ground,
                    source=('unit', unit.name),
                    target_unit=victim,
                )
                self.projectiles.append(projectile)
            else:
                self._apply_melee_damage(unit, victim, damage)
            if unit.slow_effect:
                self._apply_slow_effect(unit, victim, unit.slow_effect)
            if unit.chain_config:
                self._apply_chain_damage(unit, victim, unit.chain_config, damage)
            if getattr(unit, 'charge_reset_on_hit', True):
                self._reset_charge(unit)
            unit._attack_cd = unit.hit_speed
            return True
        if "x0" in target and "y0" in target:
            cx, cy = self._closest_point_on_rect(unit.x, unit.y, target)
            if not (unit.within_melee(cx, cy) or unit.within_attack_range(cx, cy)):
                return False
            tower = self._find_tower_by_label(target.get('label'))
            if tower and tower.get('alive', True):
                if projectile_capable:
                    speed = unit.projectile_speed if unit.projectile_speed > 0.0 else 8.0
                    projectile = Projectile(
                        owner=unit.owner,
                        x=unit.x,
                        y=unit.y,
                        speed=speed,
                        damage=damage,
                        radius=unit.projectile_radius,
                        hits_air=unit.projectile_hits_air,
                        hits_ground=unit.projectile_hits_ground,
                        source=('unit', unit.name),
                        target_structure=tower,
                    )
                    self.projectiles.append(projectile)
                else:
                    self._apply_melee_damage(unit, tower, damage)
                if unit.slow_effect:
                    self._apply_slow_effect(unit, tower, unit.slow_effect)
            if getattr(unit, 'charge_reset_on_hit', True):
                self._reset_charge(unit)
            unit._attack_cd = unit.hit_speed
            return True
        return False

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

    def deal_area(self, owner: int, x: float, y: float, radius: float, damage: float, hits_air: bool, hits_ground: bool, source: Tuple[str, str]) -> None:
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
                self.damage_structure(tower, damage, owner, source)

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
            radius = max(radius, proj_def.radius)
            if damage <= 0.0:
                damage = proj_def.damage
            proj_speed = float(card.get("speed", proj_def.speed))
            hits_ground = proj_def.hits_ground
            hits_air = proj_def.hits_air
            pushback = proj_def.pushback
        else:
            proj_speed = float(card.get("speed", 0.0) or 0.0)
            hits_ground = True
            hits_air = True
            pushback = 0.0

        if proj_def or proj_speed > 0.0:
            start_y = wy + (6.0 if owner == 1 else -6.0)
            projectile = Projectile(
                owner=owner,
                x=wx,
                y=start_y,
                speed=max(0.1, proj_speed if proj_speed > 0.0 else 8.0),
                damage=damage,
                radius=radius,
                hits_air=hits_air,
                hits_ground=hits_ground,
                source=("spell", card_name),
                target_pos=(wx, wy),
                pushback=pushback,
            )
            self.projectiles.append(projectile)
        else:
            area_radius = radius if radius > 0.0 else DEFAULT_SMALL_RADIUS
            if damage > 0.0:
                self.deal_area(owner, wx, wy, area_radius, damage, True, True, ("spell", card_name))

        if stun > 0.0:
            area_radius = radius if radius > 0.0 else DEFAULT_SMALL_RADIUS
            self.apply_stun_area(owner, wx, wy, area_radius, stun)
        if tick_dps > 0.0 and duration > 0.0:
            area_radius = radius if radius > 0.0 else 3.5
            self.effects.append(PoisonCloud(owner, wx, wy, area_radius, duration, tick_dps, card_name))
        if speed_mult > 1.0 and duration > 0.0:
            area_radius = radius if radius > 0.0 else 5.0
            self.effects.append(RageZone(owner, wx, wy, area_radius, duration, speed_mult, linger, card_name))
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
