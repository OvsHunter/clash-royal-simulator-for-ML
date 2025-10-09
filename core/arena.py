from __future__ import annotations

"""Arena grid, tower state, and deploy rules for the Clash-style simulation."""

import json
from itertools import chain
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

DEFAULT_TOWER_STATS: Dict[str, Dict[str, float]] = {
    "king": {"hp": 4824.0, "damage": 209.0, "range": 7.0, "hit_speed": 1.0},
    "princess": {"hp": 2534.0, "damage": 159.0, "range": 8.0, "hit_speed": 0.8},
}

ZONE_UNLOCK_MAP: Dict[str, Sequence[str]] = {
    "p1_princess_left": ("deploy_p1_cond_left",),
    "p1_princess_right": ("deploy_p1_cond_right",),
    "p2_princess_left": ("deploy_p2_cond_left",),
    "p2_princess_right": ("deploy_p2_cond_right",),
}


class ArenaError(Exception):
    """Raised when arena data is invalid or inconsistent."""


class StructureFootprint:
    __slots__ = ("label", "owner", "tiles", "tile_set")

    def __init__(self, label: str, owner: int, tiles: Iterable[Tuple[int, int]]) -> None:
        tile_tuple = tuple(tiles)
        self.label = label
        self.owner = owner
        self.tiles = tile_tuple
        self.tile_set = frozenset(tile_tuple)


class Arena:
    """In-memory representation of the arena grid and tower state."""

    def __init__(self, path: str) -> None:
        data = self._load_file(Path(path))
        self._path = Path(path)

        grid_size = data.get("grid_size", {})
        self.width = int(grid_size.get("width", 0))
        self.height = int(grid_size.get("height", 0))
        if (self.width, self.height) != (18, 32):
            raise ArenaError(f"arena dimensions expected to be 18x32, got {self.width}x{self.height}")

        tilemap = data.get("tilemap")
        if not isinstance(tilemap, list) or len(tilemap) != self.height:
            raise ArenaError("tilemap height mismatch")
        self.grid: List[List[str]] = []
        for row in tilemap:
            if not isinstance(row, list) or len(row) != self.width:
                raise ArenaError("tilemap width mismatch")
            self.grid.append([str(cell) for cell in row])

        self.towers: List[Dict[str, Any]] = []
        self._tower_lookup: Dict[str, Dict[str, Any]] = {}
        self._tower_footprints: Dict[str, StructureFootprint] = {}
        self.buildings: List[Dict[str, Any]] = []

        self._shared_unlocked_zones: set[str] = set()
        all_conditional = set(chain.from_iterable(ZONE_UNLOCK_MAP.values()))
        self._owner_conditional_access: Dict[str, bool] = {zone: True for zone in all_conditional}

        self._load_towers(data.get("towers", []))
        self._river_band = self._compute_river_band()

    # ------------------------------------------------------------------
    @staticmethod
    def _load_file(path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def _load_towers(self, entries: Iterable[Dict[str, Any]]) -> None:
        for idx, raw in enumerate(entries):
            label = str(raw.get("label", ""))
            if not label:
                raise ArenaError("tower entry missing label")
            owner = 1 if label.startswith("p1_") else 2
            tower_type = "king" if "king" in label else "princess"
            x0 = int(raw.get("x0", 0))
            y0 = int(raw.get("y0", 0))
            width = int(raw.get("width", 1))
            height = int(raw.get("height", 1))

            stats = DEFAULT_TOWER_STATS[tower_type]
            hp = float(raw.get("hp", stats["hp"]))
            damage = float(raw.get("damage", raw.get("dmg", stats["damage"])))
            rng = float(raw.get("range", stats["range"]))
            hit_speed = float(raw.get("hit_speed", stats["hit_speed"]))

            tower = {
                "id": idx + 1,
                "label": label,
                "owner": owner,
                "type": tower_type,
                "x0": x0,
                "y0": y0,
                "width": width,
                "height": height,
                "hp": hp,
                "hp_max": hp,
                "damage": damage,
                "range": rng,
                "hit_speed": hit_speed,
                "active": bool(raw.get("active", tower_type != "king")),
                "alive": True,
            }

            footprint_tiles = ((tx, ty) for tx in range(x0, x0 + width) for ty in range(y0, y0 + height))
            self._tower_footprints[label] = StructureFootprint(label, owner, footprint_tiles)

            self.towers.append(tower)
            self._tower_lookup[label] = tower

    # ------------------------------------------------------------------
    # Grid queries & deployment
    # ------------------------------------------------------------------

    def tile(self, x: int, y: int) -> str:
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y][x]
        return "unplayable"

    def is_deploy_legal(self, player_id: int, x: int, y: int) -> bool:
        if not (0 <= x < self.width and 0 <= y < self.height):
            return False
        tile = self.tile(x, y)
        if tile in ("unplayable", "river"):
            return False
        if self.structure_at(x, y, alive_only=True) is not None:
            return False

        base_label = f"deploy_p{player_id}"
        if tile == base_label:
            return True

        if tile.startswith("deploy_p"):
            zone_owner = 1 if "deploy_p1" in tile else 2
            owner_access = self._owner_conditional_access.get(tile, True)
            if "_cond_" in tile:
                if player_id == zone_owner and owner_access:
                    return True
                return tile in self._shared_unlocked_zones
            return False

        return False

    def can_deploy(self, player_id: int, x: int, y: int) -> bool:
        return self.is_deploy_legal(player_id, x, y)

    def set_owner_conditional_access(self, zone_label: str, enabled: bool) -> None:
        if zone_label not in self._owner_conditional_access:
            raise ArenaError(f"unknown conditional zone {zone_label}")
        self._owner_conditional_access[zone_label] = bool(enabled)

    # ------------------------------------------------------------------
    # Structures & walkability
    # ------------------------------------------------------------------

    def _iter_alive_structures(self) -> Iterable[Dict[str, Any]]:
        yield from (t for t in self.towers if t.get("alive", True))
        yield from (b for b in self.buildings if b.get("alive", True))

    def structure_at(self, x: int, y: int, alive_only: bool = True) -> Optional[Dict[str, Any]]:
        for tower in self.towers:
            if alive_only and not tower.get("alive", True):
                continue
            if (x, y) in self._tower_footprints[tower["label"]].tile_set:
                return tower
        for building in self.buildings:
            if alive_only and not building.get("alive", True):
                continue
            x0 = building.get("x0", 0)
            y0 = building.get("y0", 0)
            width = building.get("width", 1)
            height = building.get("height", 1)
            if x0 <= x < x0 + width and y0 <= y < y0 + height:
                return building
        return None

    def is_walkable(self, owner: int, x: int, y: int, *, flying: bool) -> bool:
        if not (0 <= x < self.width and 0 <= y < self.height):
            return False
        tile = self.tile(x, y)
        if tile == "unplayable":
            return False
        if tile == "river" and not flying:
            return False
        return self.structure_at(x, y, alive_only=True) is None

    # ------------------------------------------------------------------
    # Tower state utilities
    # ------------------------------------------------------------------

    def activate_king_tower(self, tower: Dict[str, Any]) -> None:
        tower_ref = self._resolve_tower(tower)
        tower_ref["active"] = True

    def on_tower_destroyed(self, tower: Dict[str, Any]) -> None:
        tower_ref = self._resolve_tower(tower)
        label = tower_ref.get("label", "")
        self._unlock_zones_for_tower(label)
        if tower_ref.get("type") == "princess":
            self._activate_owner_king(tower_ref.get("owner"))

    def _unlock_zones_for_tower(self, tower_label: str) -> None:
        for zone in ZONE_UNLOCK_MAP.get(tower_label, ()):  # shared zones become deployable for both
            self._shared_unlocked_zones.add(zone)

    def _activate_owner_king(self, owner: Optional[int]) -> None:
        if owner not in (1, 2):
            return
        for tower in self.towers:
            if tower.get("type") != "king" or tower.get("owner") != owner:
                continue
            if tower.get("alive", True):
                self.activate_king_tower(tower)
                tower["active"] = True
            break

    def _resolve_tower(self, tower: Dict[str, Any]) -> Dict[str, Any]:
        label = tower.get("label") if isinstance(tower, dict) else tower
        if not label or label not in self._tower_lookup:
            raise ArenaError(f"unknown tower reference: {tower}")
        return self._tower_lookup[label]

    # ------------------------------------------------------------------
    # Diagnostics & helpers
    # ------------------------------------------------------------------

    def alive_structures(self) -> List[Dict[str, Any]]:
        return list(self._iter_alive_structures())

    def tower_footprint(self, label: str) -> Tuple[Tuple[int, int], ...]:
        return self._tower_footprints[label].tiles

    def river_band(self) -> Optional[Tuple[int, int]]:
        return self._river_band

    def summary(self) -> str:
        lines = [f"Arena {self.width}x{self.height} from {self._path.name}"]
        for tower in self.towers:
            status = "alive" if tower.get("alive", True) else "destroyed"
            active = "active" if tower.get("active", False) else "inactive"
            lines.append(
                f"  {tower['label']} (owner {tower['owner']}): {status}, {active}, hp={tower['hp']:.0f}/{tower['hp_max']:.0f}"
            )
        if self._shared_unlocked_zones:
            lines.append(f"  shared unlocked zones: {sorted(self._shared_unlocked_zones)}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _compute_river_band(self) -> Optional[Tuple[int, int]]:
        rows = [y for y in range(self.height) if any(self.tile(x, y) == "river" for x in range(self.width))]
        if not rows:
            return None
        return min(rows), max(rows)


if __name__ == "__main__":
    arena_path = Path(__file__).resolve().parents[1] / "data" / "arena.json"
    arena = Arena(str(arena_path))
    print(arena.summary())
    probes = [(8, 15), (4, 16), (4, 23), (8, 26)]
    for x, y in probes:
        print(
            f"tile({x},{y})={arena.tile(x,y):>18} walk_ground={arena.is_walkable(1,x,y,flying=False)} walk_air={arena.is_walkable(1,x,y,flying=True)}"
        )
    print("River band:", arena.river_band())
