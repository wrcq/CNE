"""Pure k-medoids edge partitioning by edge-center distance only.

This algorithm ignores graph connectivity constraints and partitions edges
purely by geometric clustering of edge centers.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Set, Tuple

import networkx as nx

from cne.utils.edge import edge_id


def _resolve_node_positions(graph: nx.Graph) -> Dict[Any, Tuple[float, float]]:
    """Resolve node coordinates from node attribute `pos` or node tuple fallback."""
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


def _euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _edge_sort_key(edge: frozenset) -> Tuple[Any, ...]:
    return tuple(sorted(edge))


def _kmedoids_partition_core(
    graph: nx.Graph,
    k: int,
    max_iter: int,
) -> Tuple[List[Set[frozenset]], List[frozenset]]:
    """Internal k-medoids routine returning both partitions and final medoid edges."""
    n_edges = graph.number_of_edges()
    if n_edges == 0:
        return [set() for _ in range(k)], []

    k = min(k, n_edges)
    positions = _resolve_node_positions(graph)
    edges = sorted((edge_id(u, v) for u, v in graph.edges()), key=_edge_sort_key)
    edge_centers = {e: _edge_center(e, positions) for e in edges}

    def dist(e1: frozenset, e2: frozenset) -> float:
        return _euclidean(edge_centers[e1], edge_centers[e2])

    # Deterministic farthest-point initialization.
    global_cx = sum(edge_centers[e][0] for e in edges) / len(edges)
    global_cy = sum(edge_centers[e][1] for e in edges) / len(edges)
    first = min(edges, key=lambda e: (_euclidean(edge_centers[e], (global_cx, global_cy)), _edge_sort_key(e)))
    medoids: List[frozenset] = [first]
    while len(medoids) < k:
        candidates = [e for e in edges if e not in medoids]
        best = max(candidates, key=lambda e: min(dist(e, m) for m in medoids))
        medoids.append(best)

    for _ in range(max_iter):
        clusters: Dict[frozenset, List[frozenset]] = {m: [] for m in medoids}
        for e in edges:
            winner = min(medoids, key=lambda m: (dist(e, m), _edge_sort_key(m)))
            clusters[winner].append(e)

        updated: List[frozenset] = []
        used = set()
        for m in medoids:
            members = clusters.get(m, [])
            if not members:
                continue
            best = min(
                members,
                key=lambda c: (sum(dist(c, other) for other in members), _edge_sort_key(c)),
            )
            if best in used:
                alt = [x for x in members if x not in used]
                if alt:
                    best = min(
                        alt,
                        key=lambda c: (sum(dist(c, other) for other in members), _edge_sort_key(c)),
                    )
            updated.append(best)
            used.add(best)

        if len(updated) < k:
            remain = [e for e in edges if e not in used]
            while len(updated) < k and remain:
                refill = max(remain, key=lambda e: min(dist(e, m) for m in updated) if updated else 0.0)
                updated.append(refill)
                used.add(refill)
                remain = [e for e in edges if e not in used]

        updated = sorted(updated[:k], key=_edge_sort_key)
        if updated == sorted(medoids, key=_edge_sort_key):
            medoids = updated
            break
        medoids = updated

    partitions: List[Set[frozenset]] = [set() for _ in range(k)]
    for e in edges:
        idx = min(range(k), key=lambda i: (dist(e, medoids[i]), _edge_sort_key(medoids[i])))
        partitions[idx].add(e)

    return partitions, medoids


def kmedoids_partition(
    graph: nx.Graph,
    k: int,
    max_iter: int = 50,
) -> List[Set[frozenset]]:
    """Partition edges with pure k-medoids on edge-center Euclidean distance."""
    partitions, _ = _kmedoids_partition_core(graph, k, max_iter)
    return partitions


def kmedoids_partition_with_medoids(
    graph: nx.Graph,
    k: int,
    max_iter: int = 50,
) -> Tuple[List[Set[frozenset]], List[frozenset]]:
    """Return both k-medoids edge partitions and final medoid edges."""
    return _kmedoids_partition_core(graph, k, max_iter)
