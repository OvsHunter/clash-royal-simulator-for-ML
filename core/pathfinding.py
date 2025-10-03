from __future__ import annotations

import heapq
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

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


def _build_passable_fn(arena, owner: int, flying: bool, goal_tiles: Sequence[Tuple[int, int]], start: Tuple[int, int]):
    goal_set = set(goal_tiles)

    def passable(x: int, y: int) -> bool:
        if (x, y) == start:
            return True
        if (x, y) in goal_set:
            return True
        return arena.is_walkable(owner, x, y, flying=flying)

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

    open_heap: List[Tuple[float, Tuple[int, int]]] = []
    heapq.heappush(open_heap, (heuristic(start), start))
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    g_score: Dict[Tuple[int, int], float] = {start: 0.0}
    closed: set[Tuple[int, int]] = set()

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
            if tentative_g < g_score.get(neighbor, float("inf")):
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
    out: List[Tuple[int, int]] = []
    for y in range(arena.height):
        for x in range(arena.width):
            if arena.tile(x, y) == "bridge":
                if side > 0 and y > river_max:
                    out.append((x, y))
                elif side < 0 and y < river_min:
                    out.append((x, y))
    return out


def _select_bridge_tile(arena, start: Tuple[int, int], side: int, river_min: int, river_max: int) -> Optional[Tuple[int, int]]:
    candidates = _bridge_tiles(arena, side, river_min, river_max)
    if not candidates:
        return None
    sx, sy = start
    return min(candidates, key=lambda t: distance_tiles(start, t))


def compute_path(
    arena,
    unit_dict: Union[Dict[str, Any], object],
    target: Union[Dict[str, Any], object],
    *,
    allow_diagonal_cutting: bool = False,
    max_nodes: Optional[int] = None,
) -> List[Tuple[int, int]]:
    start = world_to_tile(float(_get_attr(unit_dict, "x", 0.0)), float(_get_attr(unit_dict, "y", 0.0)))
    owner = int(_get_attr(unit_dict, "owner", 1))
    flying = bool(_get_attr(unit_dict, "flying", False))

    goals = _goal_tiles_for_target(arena, target)
    if not goals:
        return []

    limit = MAX_NODES_DEFAULT if max_nodes is None else max(1, max_nodes)

    def direct_path() -> List[Tuple[int, int]]:
        passable_fn = _build_passable_fn(arena, owner, flying, goals, start)
        return _astar(
            arena,
            start,
            goals,
            passable_fn,
            max_nodes=limit,
            allow_diagonal_cutting=allow_diagonal_cutting,
        )

    if flying:
        return direct_path()

    river_band = _river_band(arena)
    if river_band is None:
        return direct_path()

    river_min, river_max = river_band
    start_side = 1 if start[1] > river_max else -1 if start[1] < river_min else 0
    goal_requires_other_side = False
    for gx, gy in goals:
        goal_side = 1 if gy > river_max else -1 if gy < river_min else 0
        if goal_side != start_side:
            goal_requires_other_side = True
            break

    if start_side != 0 and goal_requires_other_side:
        lane_tile = _select_bridge_tile(arena, start, start_side, river_min, river_max)
        if lane_tile is not None:
            passable_lane = _build_passable_fn(arena, owner, flying, (lane_tile,), start)
            lane_path = _astar(
                arena,
                start,
                (lane_tile,),
                passable_lane,
                max_nodes=limit,
                allow_diagonal_cutting=allow_diagonal_cutting,
            )
            if lane_path:
                passable_final = _build_passable_fn(arena, owner, flying, goals, lane_tile)
                final_path = _astar(
                    arena,
                    lane_tile,
                    goals,
                    passable_final,
                    max_nodes=limit,
                    allow_diagonal_cutting=allow_diagonal_cutting,
                )
                if final_path:
                    return lane_path + final_path

    return direct_path()


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
