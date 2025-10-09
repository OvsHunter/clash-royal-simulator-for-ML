from __future__ import annotations

import heapq
import math
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

SQRT2 = math.sqrt(2)
DIRECTIONS: Tuple[Tuple[int, int, float], ...] = (
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (-1, -1, SQRT2),
    (1, -1, SQRT2),
    (-1, 1, SQRT2),
    (1, 1, SQRT2),
)
MAX_FLOW_DISTANCE = 9999.0
BRIDGE_CAPACITY = 4
BRIDGE_RESERVE_STEPS = 5
MAX_NODES_DEFAULT = 6000


def _get_attr(obj: Union[Dict[str, Any], object], key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def world_to_tile(x: float, y: float) -> Tuple[int, int]:
    return int(round(x)), int(round(y))


def distance_tiles(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return (max(dx, dy) - min(dx, dy)) + SQRT2 * min(dx, dy)


def _structure_perimeter(x0: int, y0: int, width: int, height: int) -> List[Tuple[int, int]]:
    tiles: List[Tuple[int, int]] = []
    left = x0 - 1
    right = x0 + width
    top = y0 - 1
    bottom = y0 + height
    for x in range(x0, x0 + width):
        tiles.append((x, top))
        tiles.append((x, bottom))
    for y in range(y0, y0 + height):
        tiles.append((left, y))
        tiles.append((right, y))
    seen: set[Tuple[int, int]] = set()
    out: List[Tuple[int, int]] = []
    for t in tiles:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _tile_is_clear(
    arena,
    owner: int,
    flying: bool,
    x: int,
    y: int,
    radius: float,
    blocked_tiles: Optional[Set[Tuple[int, int]]] = None,
) -> bool:
    if blocked_tiles and (x, y) in blocked_tiles:
        return False
    if not (0 <= x < arena.width and 0 <= y < arena.height):
        return False
    if not arena.is_walkable(owner, x, y, flying=flying):
        return False
    effective = max(0.0, radius - 0.3)
    if effective <= 0.0:
        return True
    limit = max(1, int(math.ceil(effective)))
    threshold = effective + 1e-3
    for dx in range(-limit, limit + 1):
        for dy in range(-limit, limit + 1):
            if dx == 0 and dy == 0:
                continue
            if math.hypot(dx, dy) > threshold:
                continue
            tx = x + dx
            ty = y + dy
            if blocked_tiles and (tx, ty) in blocked_tiles:
                return False
            if not (0 <= tx < arena.width and 0 <= ty < arena.height):
                return False
            if not arena.is_walkable(owner, tx, ty, flying=flying):
                return False
    return True


def _goal_tiles_for_target(arena, target: Union[Dict[str, Any], object]) -> List[Tuple[int, int]]:
    if target is None:
        return []
    if isinstance(target, dict):
        if target.get("alive") is False:
            return []
        if "x0" in target and "y0" in target:
            x0 = int(target.get("x0", 0))
            y0 = int(target.get("y0", 0))
            width = int(target.get("width", 1))
            height = int(target.get("height", 1))
            candidate = _structure_perimeter(x0, y0, width, height)
            return [t for t in candidate if 0 <= t[0] < arena.width and 0 <= t[1] < arena.height]
        if "x" in target and "y" in target:
            return [world_to_tile(float(target["x"]), float(target["y"]))]
    else:
        pos = getattr(target, "pos", None)
        if pos is not None:
            return [world_to_tile(float(pos[0]), float(pos[1]))]
        if hasattr(target, "x") and hasattr(target, "y"):
            return [world_to_tile(float(target.x), float(target.y))]
    return []


def _clamp_tile(arena, tile: Tuple[int, int]) -> Tuple[int, int]:
    return (max(0, min(arena.width - 1, tile[0])), max(0, min(arena.height - 1, tile[1])))


def _build_passable_fn(
    arena,
    owner: int,
    flying: bool,
    goal_tiles: Sequence[Tuple[int, int]],
    start: Tuple[int, int],
    blocked_tiles: Optional[Set[Tuple[int, int]]] = None,
    unit_radius: float = 0.0,
) -> Any:
    goal_set = set(goal_tiles)
    blocked = set(blocked_tiles or [])

    def passable(x: int, y: int) -> bool:
        if (x, y) == start or (x, y) in goal_set:
            return True
        return _tile_is_clear(arena, owner, flying, x, y, unit_radius, blocked_tiles=blocked)

    return passable


def _reconstruct_path(came_from: Dict[Tuple[int, int], Tuple[int, int]], current: Tuple[int, int]) -> List[Tuple[int, int]]:
    path: List[Tuple[int, int]] = []
    while current in came_from:
        path.append(current)
        current = came_from[current]
    path.reverse()
    return path


def _astar(
    arena,
    start: Tuple[int, int],
    goal_tiles: Sequence[Tuple[int, int]],
    passable_fn,
    *,
    max_nodes: int,
    allow_diagonal_cutting: bool = False,
) -> List[Tuple[int, int]]:
    goal_set = set(goal_tiles)
    if not goal_set:
        return []

    def heuristic(tile: Tuple[int, int]) -> float:
        return min(distance_tiles(tile, goal) for goal in goal_set)

    open_heap: List[Tuple[float, Tuple[int, int]]] = [(heuristic(start), start)]
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    g_score: Dict[Tuple[int, int], float] = {start: 0.0}
    closed: Set[Tuple[int, int]] = set()
    nodes_expanded = 0
    width, height = arena.width, arena.height

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        closed.add(current)
        if current in goal_set:
            return _reconstruct_path(came_from, current)
        nodes_expanded += 1
        if nodes_expanded > max_nodes:
            break
        cx, cy = current
        for dx, dy, cost in DIRECTIONS:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            if not allow_diagonal_cutting and dx != 0 and dy != 0:
                if not (passable_fn(cx + dx, cy) and passable_fn(cx, cy + dy)):
                    continue
            if not passable_fn(nx, ny):
                continue
            neighbor = (nx, ny)
            tentative_g = g_score[current] + cost
            if tentative_g + 1e-9 < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor)
                heapq.heappush(open_heap, (f_score, neighbor))

    return []


def _river_band(arena) -> Optional[Tuple[int, int]]:
    rows = [y for y in range(arena.height) if any(arena.tile(x, y) == "river" for x in range(arena.width))]
    if not rows:
        return None
    return min(rows), max(rows)


def _bridge_tiles(arena, side: int, river_min: int, river_max: int) -> List[Tuple[int, int]]:
    tiles: List[Tuple[int, int]] = []
    for y in range(arena.height):
        for x in range(arena.width):
            if arena.tile(x, y) == "bridge":
                if side > 0 and y > river_max:
                    tiles.append((x, y))
                elif side < 0 and y < river_min:
                    tiles.append((x, y))
    return tiles


def _select_bridge_tile(arena, start: Tuple[int, int], side: int, river_min: int, river_max: int) -> Optional[Tuple[int, int]]:
    candidates = _bridge_tiles(arena, side, river_min, river_max)
    if not candidates:
        return None
    return min(candidates, key=lambda pos: distance_tiles(start, pos))


def _legacy_path(
    arena,
    owner: int,
    flying: bool,
    start: Tuple[int, int],
    goal_tiles: Sequence[Tuple[int, int]],
    blocked_tiles: Optional[Set[Tuple[int, int]]],
    max_nodes: int,
    unit_radius: float,
) -> List[Tuple[int, int]]:
    def direct() -> List[Tuple[int, int]]:
        passable = _build_passable_fn(arena, owner, flying, goal_tiles, start, blocked_tiles, unit_radius)
        return _astar(
            arena,
            start,
            goal_tiles,
            passable,
            max_nodes=max_nodes,
            allow_diagonal_cutting=False,
        )

    if flying:
        return direct()

    river_band = _river_band(arena)
    if river_band is None:
        return direct()

    river_min, river_max = river_band
    start_side = 1 if start[1] > river_max else -1 if start[1] < river_min else 0
    goal_requires_other_side = any(
        (1 if gy > river_max else -1 if gy < river_min else 0) != start_side for gx, gy in goal_tiles
    )

    if start_side != 0 and goal_requires_other_side:
        lane_tile = _select_bridge_tile(arena, start, start_side, river_min, river_max)
        if lane_tile is not None:
            passable_lane = _build_passable_fn(arena, owner, flying, (lane_tile,), start, blocked_tiles, unit_radius)
            lane_path = _astar(
                arena,
                start,
                (lane_tile,),
                passable_lane,
                max_nodes=max_nodes,
                allow_diagonal_cutting=False,
            )
            if lane_path:
                passable_final = _build_passable_fn(
                    arena, owner, flying, goal_tiles, lane_tile, blocked_tiles, unit_radius
                )
                final_path = _astar(
                    arena,
                    lane_tile,
                    goal_tiles,
                    passable_final,
                    max_nodes=max_nodes,
                    allow_diagonal_cutting=False,
                )
                if final_path:
                    return lane_path + final_path

    return direct()


class FlowField:
    """Precomputed potential field that directs units toward a goal."""
    __slots__ = ("arena", "owner", "flying", "width", "height", "costs")

    def __init__(self, arena, owner: int, flying: bool, goal_tiles: Sequence[Tuple[int, int]]) -> None:
        self.arena = arena
        self.owner = owner
        self.flying = flying
        self.width = arena.width
        self.height = arena.height
        self.costs: List[List[float]] = [[MAX_FLOW_DISTANCE for _ in range(self.width)] for _ in range(self.height)]
        self._build(goal_tiles)

    def _build(self, goal_tiles: Sequence[Tuple[int, int]]) -> None:
        if not goal_tiles:
            return
        heap: List[Tuple[float, int, int]] = []
        goal_set: Set[Tuple[int, int]] = set()
        for gx, gy in goal_tiles:
            if not self._in_bounds(gx, gy):
                continue
            if (gx, gy) in goal_set:
                continue
            goal_set.add((gx, gy))
            self.costs[gy][gx] = 0.0
            heapq.heappush(heap, (0.0, gx, gy))

        while heap:
            cost, x, y = heapq.heappop(heap)
            if cost > self.costs[y][x] + 1e-6:
                continue
            for dx, dy, step_cost in DIRECTIONS:
                nx = x + dx
                ny = y + dy
                if not self._in_bounds(nx, ny):
                    continue
                if not self._walkable(nx, ny) and (nx, ny) not in goal_set:
                    continue
                if dx != 0 and dy != 0:
                    if not (self._walkable(x, ny) and self._walkable(nx, y)):
                        continue
                next_cost = cost + step_cost
                if next_cost + 1e-6 < self.costs[ny][nx]:
                    self.costs[ny][nx] = next_cost
                    heapq.heappush(heap, (next_cost, nx, ny))

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def _walkable(self, x: int, y: int, *, radius: float = 0.0, blocked: Optional[Set[Tuple[int, int]]] = None) -> bool:
        return _tile_is_clear(self.arena, self.owner, self.flying, x, y, radius, blocked_tiles=blocked)

    def cost_at(self, x: int, y: int) -> float:
        if not self._in_bounds(x, y):
            return MAX_FLOW_DISTANCE
        return self.costs[y][x]


class FlowFieldManager:
    def __init__(self) -> None:
        self._cache: Dict[Tuple[int, int, bool, Tuple[Tuple[int, int], ...], Tuple[Any, ...]], FlowField] = {}

    @staticmethod
    def _arena_signature(arena) -> Tuple[Any, ...]:
        signature: List[Any] = []
        for tower in arena.towers:
            signature.append((tower.get("label"), bool(tower.get("alive", True))))
        for building in getattr(arena, "buildings", ()):
            signature.append(
                (
                    building.get("label"),
                    bool(building.get("alive", True)),
                    int(building.get("x0", 0)),
                    int(building.get("y0", 0)),
                )
            )
        return tuple(signature)

    def get_field(
        self,
        arena,
        owner: int,
        flying: bool,
        goal_tiles: Sequence[Tuple[int, int]],
    ) -> FlowField:
        goals_tuple = tuple(sorted(set(_clamp_tile(arena, tile) for tile in goal_tiles)))
        key = (id(arena), owner, flying, goals_tuple, self._arena_signature(arena))
        field = self._cache.get(key)
        if field is None:
            field = FlowField(arena, owner, flying, goals_tuple)
            self._cache[key] = field
        return field

    def select_step(
        self,
        arena,
        field: FlowField,
        current: Tuple[int, int],
        unit_radius: float,
        blocked_tiles: Optional[Set[Tuple[int, int]]],
        occupancy: "BridgeOccupancy",
        goal_tiles: Sequence[Tuple[int, int]],
    ) -> Optional[Tuple[int, int]]:
        cx, cy = current
        current_cost = field.cost_at(cx, cy)
        if current_cost <= 1e-6:
            return None

        best_tile: Optional[Tuple[int, int]] = None
        best_score = float("inf")
        blocked = blocked_tiles or set()
        goal_set = set(goal_tiles)

        for dx, dy, step_cost in DIRECTIONS:
            nx = cx + dx
            ny = cy + dy
            if not field._in_bounds(nx, ny):
                continue
            if not field._walkable(nx, ny, radius=unit_radius, blocked=blocked) and (nx, ny) not in goal_set:
                continue
            if dx != 0 and dy != 0:
                if not (
                    field._walkable(cx, ny, radius=unit_radius, blocked=blocked)
                    and field._walkable(nx, cy, radius=unit_radius, blocked=blocked)
                ):
                    continue
            neighbor_cost = field.cost_at(nx, ny)
            if neighbor_cost >= MAX_FLOW_DISTANCE:
                continue
            if neighbor_cost > current_cost + 1e-3 and (nx, ny) not in goal_set:
                continue

            penalty = 0.0
            if (nx, ny) in blocked and (nx, ny) not in goal_set:
                penalty += 5.0
            penalty += occupancy.penalty(arena, (nx, ny), unit_radius)
            directional_bias = 0.05 if dx != 0 and dy != 0 else 0.0
            step_penalty = step_cost * 0.01

            score = neighbor_cost + penalty + directional_bias + step_penalty
            if score + 1e-6 < best_score:
                best_score = score
                best_tile = (nx, ny)

        return best_tile


class BridgeOccupancy:
    """Tracks short-lived congestion on bridge tiles to throttle traffic."""
    def __init__(self) -> None:
        self._arena_id: Optional[int] = None
        self._queues: Dict[Tuple[int, int], deque[Tuple[int, float]]] = {}
        self._tick: int = 0

    def _ensure_arena(self, arena) -> None:
        arena_id = id(arena)
        if self._arena_id == arena_id:
            return
        self._arena_id = arena_id
        self._queues.clear()
        for y, row in enumerate(arena.grid):
            for x, cell in enumerate(row):
                if str(cell).startswith("bridge"):
                    self._queues[(x, y)] = deque(maxlen=BRIDGE_CAPACITY * 4)

    def reserve_path(self, arena, path: Sequence[Tuple[int, int]], unit_radius: float) -> None:
        if not path:
            return
        self._ensure_arena(arena)
        self._tick += 1
        weight = max(1.0, unit_radius / 0.4)
        for tile in path[:BRIDGE_RESERVE_STEPS]:
            queue = self._queues.get(tile)
            if queue is None:
                continue
            queue.append((self._tick, weight))
            self._prune(queue)

    def _prune(self, queue: deque[Tuple[int, float]]) -> None:
        while queue and self._tick - queue[0][0] > BRIDGE_RESERVE_STEPS:
            queue.popleft()

    def penalty(self, arena, tile: Tuple[int, int], unit_radius: float) -> float:
        self._ensure_arena(arena)
        queue = self._queues.get(tile)
        if not queue:
            return 0.0
        self._prune(queue)
        density = sum(weight for _, weight in queue)
        if density == 0:
            return 0.0
        radius_weight = max(0.2, unit_radius)
        return density * radius_weight


_FLOW_MANAGER = FlowFieldManager()
_BRIDGE_OCCUPANCY = BridgeOccupancy()


def compute_path(
    arena,
    unit_dict: Union[Dict[str, Any], object],
    target: Union[Dict[str, Any], object],
    *,
    blocked_tiles: Optional[Set[Tuple[int, int]]] = None,
    max_steps: int = 48,
) -> List[Tuple[int, int]]:
    owner = int(_get_attr(unit_dict, "owner", 1) or 1)
    flying = bool(_get_attr(unit_dict, "flying", False))
    unit_radius = float(_get_attr(unit_dict, "radius", 0.5) or 0.5)
    start_x = float(_get_attr(unit_dict, "x", 0.0))
    start_y = float(_get_attr(unit_dict, "y", 0.0))
    start_tile = world_to_tile(start_x, start_y)

    goal_tiles = _goal_tiles_for_target(arena, target)
    if not goal_tiles:
        return []

    field = _FLOW_MANAGER.get_field(arena, owner, flying, goal_tiles)

    path: List[Tuple[int, int]] = []
    current = start_tile
    visited: Set[Tuple[int, int]] = set()
    fallback_needed = False

    for _ in range(max_steps):
        next_tile = _FLOW_MANAGER.select_step(
            arena=arena,
            field=field,
            current=current,
            unit_radius=unit_radius,
            blocked_tiles=blocked_tiles,
            occupancy=_BRIDGE_OCCUPANCY,
            goal_tiles=goal_tiles,
        )
        if next_tile is None or next_tile == current or next_tile in visited:
            fallback_needed = True
            break
        path.append(next_tile)
        visited.add(next_tile)
        current = next_tile
        if field.cost_at(current[0], current[1]) <= 1e-6:
            break

    if not path or fallback_needed:
        fallback_path = _legacy_path(
            arena=arena,
            owner=owner,
            flying=flying,
            start=start_tile,
            goal_tiles=goal_tiles,
            blocked_tiles=blocked_tiles,
            max_nodes=MAX_NODES_DEFAULT,
            unit_radius=unit_radius,
        )
        if fallback_path:
            _BRIDGE_OCCUPANCY.reserve_path(arena, fallback_path, unit_radius)
            return fallback_path
        if not path:
            return []

    _BRIDGE_OCCUPANCY.reserve_path(arena, path, unit_radius)
    return path


def find_target(
    arena,
    unit_dict: Union[Dict[str, Any], object],
    enemy_units: Sequence[Union[Dict[str, Any], object]],
) -> Optional[Union[Dict[str, Any], object]]:
    ux = float(_get_attr(unit_dict, "x", 0.0))
    uy = float(_get_attr(unit_dict, "y", 0.0))
    aggro = float(_get_attr(unit_dict, "aggro_range", 6.0))
    targeting_raw = _get_attr(unit_dict, "targeting", ["both"]) or ["both"]
    targeting = [str(t).lower() for t in (targeting_raw if isinstance(targeting_raw, (list, tuple)) else [targeting_raw])]

    building_only = "buildings" in targeting and all(tag not in ("ground", "air", "both") for tag in targeting)

    best_unit = None
    best_unit_dist = float("inf")
    if not building_only:
        for enemy in enemy_units:
            if not _get_attr(enemy, "alive", True):
                continue
            enemy_flying = bool(_get_attr(enemy, "flying", False))
            if enemy_flying and all(tag not in ("air", "both") for tag in targeting):
                continue
            if not enemy_flying and all(tag not in ("ground", "both") for tag in targeting):
                continue
            ex = float(_get_attr(enemy, "x", 0.0))
            ey = float(_get_attr(enemy, "y", 0.0))
            d = math.hypot(ux - ex, uy - ey)
            if d <= aggro and d < best_unit_dist:
                best_unit_dist = d
                best_unit = enemy
    if best_unit is not None:
        return best_unit

    best_structure = None
    best_structure_dist = float("inf")
    for structure in arena.alive_structures():
        if structure.get("owner") == _get_attr(unit_dict, "owner"):
            continue
        if all(tag not in ("buildings", "both", "ground") for tag in targeting):
            continue
        cx = structure.get("x0", 0) + structure.get("width", 1) / 2.0
        cy = structure.get("y0", 0) + structure.get("height", 1) / 2.0
        d = math.hypot(ux - cx, uy - cy)
        if d < best_structure_dist:
            best_structure_dist = d
            best_structure = structure

    return best_structure


def visualize_path(arena, path: Sequence[Tuple[int, int]], *, mark: str = "*") -> str:
    grid = [row[:] for row in arena.grid]
    for x, y in path:
        if 0 <= x < arena.width and 0 <= y < arena.height:
            grid[y][x] = mark
    return "\n".join(" ".join(row) for row in grid)


if __name__ == "__main__":
    import sys

    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from core.arena import Arena

    arena_path = ROOT / "data" / "arena.json"
    arena = Arena(str(arena_path))
    unit = {"x": 8.5, "y": 23.5, "owner": 1, "flying": False}
    target = next(t for t in arena.towers if t["label"] == "p2_king")
    path = compute_path(arena, unit, target)
    print("Computed path (first 12 tiles):", path[:12])
    print("Path length:", len(path))
