"""METIS baseline: treat each original edge as a node and partition by distance graph.

Workflow:
1) Build edge-center points from original graph edges.
2) Build a kNN proximity graph among these points.
3) Run METIS node partition on the proximity graph.
4) Map node partitions back to original edge partitions.

Connectivity of the original graph is NOT enforced.
"""

from __future__ import annotations

import math
import os
import site
from typing import Any, Dict, List, Set, Tuple

import networkx as nx

from cne.utils.edge import edge_id


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


def _euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _edge_sort_key(edge: frozenset) -> Tuple[Any, ...]:
    return tuple(sorted(edge))


def _import_metis():
    """Import python-metis lazily so non-METIS workflows are unaffected."""
    if not os.environ.get("METIS_DLL"):
        # Common local setup: metis.dll placed in active environment's site-packages.
        candidates: List[str] = []
        try:
            candidates.extend(site.getsitepackages())
        except Exception:
            pass
        try:
            user_site = site.getusersitepackages()
            if user_site:
                candidates.append(user_site)
        except Exception:
            pass

        for base in candidates:
            dll_path = os.path.join(base, "metis.dll")
            if os.path.isfile(dll_path):
                os.environ["METIS_DLL"] = dll_path
                break

    try:
        import metis  # type: ignore

        return metis
    except Exception as exc:  # pragma: no cover - depends on local METIS runtime
        raise RuntimeError(
            "METIS backend unavailable. Install python package 'metis' and set METIS_DLL to the "
            "full path of metis.dll on Windows."
        ) from exc


def metis_edge_node_partition(
    graph: nx.Graph,
    k: int,
    knn: int = 8,
) -> List[Set[frozenset]]:
    """Partition edges by METIS node partition on edge-center kNN graph."""
    n_edges = graph.number_of_edges()
    if n_edges == 0:
        return [set() for _ in range(k)]

    k = min(k, n_edges)
    positions = _resolve_node_positions(graph)
    edges = sorted((edge_id(u, v) for u, v in graph.edges()), key=_edge_sort_key)
    centers = [_edge_center(e, positions) for e in edges]

    # Build an undirected weighted proximity graph over edge-centers.
    n = len(edges)
    knn = max(1, min(knn, max(1, n - 1)))
    point_graph = nx.Graph()
    point_graph.add_nodes_from(range(n))

    for i in range(n):
        dists: List[Tuple[float, int]] = []
        for j in range(n):
            if i == j:
                continue
            dists.append((_euclidean(centers[i], centers[j]), j))
        dists.sort(key=lambda x: x[0])

        for d, j in dists[:knn]:
            # Higher weight means stronger affinity in METIS cut objective.
            w = max(1, int(round(1000.0 / (1.0 + d))))
            if point_graph.has_edge(i, j):
                if w > point_graph[i][j].get("weight", 1):
                    point_graph[i][j]["weight"] = w
            else:
                point_graph.add_edge(i, j, weight=w)

    metis = _import_metis()
    _, parts = metis.part_graph(point_graph, nparts=k)

    partitions: List[Set[frozenset]] = [set() for _ in range(k)]
    for idx, p in enumerate(parts):
        p_int = int(p)
        if p_int < 0 or p_int >= k:
            continue
        partitions[p_int].add(edges[idx])

    return partitions
