"""Statistics helpers for edge partition results."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Set, Tuple

import networkx as nx

from cne.utils.edge import edge_load


def _resolve_node_positions(graph: nx.Graph) -> Dict[Any, Tuple[float, float]]:
    positions: Dict[Any, Tuple[float, float]] = {}
    has_any_pos_attr = False

    for node, data in graph.nodes(data=True):
        pos = data.get("pos")
        if isinstance(pos, (tuple, list)) and len(pos) >= 2:
            positions[node] = (float(pos[0]), float(pos[1]))
            has_any_pos_attr = True
        elif isinstance(node, (tuple, list)) and len(node) >= 2:
            positions[node] = (float(node[0]), float(node[1]))

    if len(positions) == graph.number_of_nodes() and (has_any_pos_attr or graph.number_of_nodes() > 0):
        return positions

    fallback = nx.spring_layout(graph, seed=0)
    for node, p in fallback.items():
        positions.setdefault(node, (float(p[0]), float(p[1])))
    return positions


def _edge_center(edge: frozenset, positions: Dict[Any, Tuple[float, float]]) -> Tuple[float, float]:
    u, v = tuple(edge)
    pu = positions[u]
    pv = positions[v]
    return ((pu[0] + pv[0]) * 0.5, (pu[1] + pv[1]) * 0.5)


def _convex_hull(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    pts = sorted(set(points))
    if len(pts) <= 1:
        return pts

    def cross(o, a, b) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List[Tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: List[Tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


def _polygon_area_perimeter(poly: List[Tuple[float, float]]) -> Tuple[float, float]:
    n = len(poly)
    if n < 3:
        return 0.0, 0.0

    area2 = 0.0
    perim = 0.0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        area2 += x1 * y2 - x2 * y1
        perim += math.hypot(x2 - x1, y2 - y1)
    return abs(area2) * 0.5, perim


def _compactness_for_points(points: List[Tuple[float, float]]) -> Tuple[float, str]:
    if len(points) < 3:
        return 0.0, "insufficient_points"

    # Preferred method: concave hull (if shapely is available), fallback to convex hull.
    try:
        import shapely  # type: ignore
        from shapely.geometry import MultiPoint  # type: ignore

        mp = MultiPoint(points)
        if hasattr(shapely, "concave_hull"):
            poly = shapely.concave_hull(mp, ratio=0.35, allow_holes=False)
        else:
            poly = mp.convex_hull

        area = float(getattr(poly, "area", 0.0))
        perim = float(getattr(poly, "length", 0.0))
        if perim <= 0.0:
            return 0.0, "concave_hull"
        c = (4.0 * math.pi * area) / (perim * perim)
        return max(0.0, min(1.0, c)), "concave_hull"
    except Exception:
        hull = _convex_hull(points)
        area, perim = _polygon_area_perimeter(hull)
        if perim <= 0.0:
            return 0.0, "convex_hull_fallback"
        c = (4.0 * math.pi * area) / (perim * perim)
        return max(0.0, min(1.0, c)), "convex_hull_fallback"


def partition_stats(
    graph: nx.Graph,
    partitions: List[Set[frozenset]],
    weight: str = "weight",
) -> Dict:
    """Return core statistics for edge-partition results."""
    positions = _resolve_node_positions(graph)

    loads = []
    edge_counts = []
    compactness_per_partition = []
    compactness_methods = []
    for part in partitions:
        load = sum(edge_load(graph, *e, weight) for e in part)
        loads.append(load)
        edge_counts.append(len(part))

        centers = [_edge_center(e, positions) for e in part]
        c, method = _compactness_for_points(centers)
        compactness_per_partition.append(c)
        compactness_methods.append(method)

    mean_load = sum(loads) / len(loads) if loads else 0
    max_l = max(loads) if loads else 0
    min_l = min(loads) if loads else 0
    imbalance = (max_l - min_l) / mean_load if mean_load > 0 else 0

    part_nodes = []
    for part in partitions:
        nodes = set()
        for e in part:
            nodes.update(e)
        part_nodes.append(nodes)

    all_nodes = set()
    shared = set()
    for ns in part_nodes:
        shared |= all_nodes & ns
        all_nodes |= ns

    compactness_mean = (
        sum(compactness_per_partition) / len(compactness_per_partition)
        if compactness_per_partition
        else 0.0
    )

    return {
        "loads": loads,
        "edge_counts": edge_counts,
        "mean_load": mean_load,
        "max_imbalance": imbalance,
        "shared_nodes": len(shared),
        "partition_nodes": [sorted(ns) for ns in part_nodes],
        "compactness_per_partition": compactness_per_partition,
        "compactness_mean": compactness_mean,
        "compactness_method_per_partition": compactness_methods,
        "compactness_method": compactness_methods[0] if compactness_methods else "insufficient_points",
    }
