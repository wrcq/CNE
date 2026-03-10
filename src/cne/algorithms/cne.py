"""Constrained Neighbor Expansion (CNE) edge partitioning algorithm.

This module implements a competitive multi-source synchronized expansion strategy.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

from cne.utils.edge import edge_id, edge_load


def _build_edge_adjacency(graph: nx.Graph) -> Dict[frozenset, Set[frozenset]]:
    """Build edge adjacency where two edges are adjacent if they share a node."""
    adj: Dict[frozenset, Set[frozenset]] = {}
    for u, v in graph.edges():
        eid = edge_id(u, v)
        adj[eid] = set()

    for node in graph.nodes():
        incident = [edge_id(node, nb) for nb in graph.neighbors(node)]
        for i, e1 in enumerate(incident):
            for e2 in incident[i + 1 :]:
                adj[e1].add(e2)
                adj[e2].add(e1)
    return adj


def _select_seed_edges(
    graph: nx.Graph,
    edge_adj: Dict[frozenset, Set[frozenset]],
    k: int,
    weight: str,
) -> List[frozenset]:
    """Greedily pick k dispersed seed edges using BFS distance in edge-adjacency graph."""
    all_edges = list(edge_adj.keys())
    if k >= len(all_edges):
        return all_edges[:k]

    def bfs_dist(start: frozenset, target: frozenset) -> int:
        if start == target:
            return 0
        visited = {start}
        queue = [start]
        dist = 0
        while queue:
            dist += 1
            next_queue = []
            for e in queue:
                for nb in edge_adj[e]:
                    if nb == target:
                        return dist
                    if nb not in visited:
                        visited.add(nb)
                        next_queue.append(nb)
            queue = next_queue
        return float("inf")

    seeds: List[frozenset] = [max(all_edges, key=lambda e: edge_load(graph, *e, weight))]

    while len(seeds) < k:
        best_edge = None
        best_min_dist = -1
        for e in all_edges:
            if e in seeds:
                continue
            min_d = min(bfs_dist(e, s) for s in seeds)
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_edge = e
        if best_edge is None:
            break
        seeds.append(best_edge)
    return seeds


def _edges_connected(edge_set: Set[frozenset], edge_adj: Dict[frozenset, Set[frozenset]]) -> bool:
    """Check edge-connectivity (connectivity in edge-adjacency graph)."""
    if len(edge_set) <= 1:
        return True
    start = next(iter(edge_set))
    visited = {start}
    queue = [start]
    while queue:
        cur = queue.pop()
        for nb in edge_adj[cur]:
            if nb in edge_set and nb not in visited:
                visited.add(nb)
                queue.append(nb)
    return len(visited) == len(edge_set)


def _resolve_node_positions(graph: nx.Graph) -> Dict[Any, Tuple[float, float]]:
    """Resolve 2D node positions for distance-based competitive cost.

    Priority: existing node attribute `pos` -> tuple/list node labels -> spring layout fallback.
    """
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
    """Compute center point of an edge from endpoint positions."""
    u, v = tuple(edge)
    pu = positions[u]
    pv = positions[v]
    return ((pu[0] + pv[0]) * 0.5, (pu[1] + pv[1]) * 0.5)


def _euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Euclidean distance in 2D."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def cne_partition(
    graph: nx.Graph,
    k: int,
    weight: str = "weight",
    seed_edges: Optional[List[Tuple]] = None,
    refine_iterations: int = 50,
    alpha: float = 1.0,
    beta: float = 1.0,
    overload_threshold: float = 1.2,
) -> List[Set[frozenset]]:
    """Partition edges into k balanced connected groups using competitive CNE expansion.

    Competitive cost for assigning edge ``e`` to partition ``G_i``:
    ``J(e, G_i) = alpha * J_scale_hat(G_i) + beta * J_dist_hat(e, G_i)``.

    Admission gating by relative load intensity:
    ``sigma_i = S(G_i) / mean(S)``, and partitions with ``sigma_i > overload_threshold``
    are temporarily excluded from competition when non-overloaded competitors exist.
    """
    n_edges = graph.number_of_edges()
    if n_edges == 0:
        return [set() for _ in range(k)]

    k = min(k, n_edges)
    edge_adj = _build_edge_adjacency(graph)

    if seed_edges is not None:
        seeds = [edge_id(u, v) for u, v in seed_edges]
    else:
        seeds = _select_seed_edges(graph, edge_adj, k, weight)
    assert len(seeds) == k, f"seed edge count ({len(seeds)}) must equal k ({k})"

    partitions: List[Set[frozenset]] = [set() for _ in range(k)]
    assigned: Set[frozenset] = set()

    for i, se in enumerate(seeds):
        partitions[i].add(se)
        assigned.add(se)

    positions = _resolve_node_positions(graph)
    edge_centers = {eid: _edge_center(eid, positions) for eid in edge_adj}
    seed_centers = {i: edge_centers[seeds[i]] for i in range(k)}

    total_load = sum(d.get(weight, 1.0) for _, _, d in graph.edges(data=True))
    target_load = total_load / k
    active = [True] * k

    while len(assigned) < n_edges and any(active):
        loads = [sum(edge_load(graph, *e, weight) for e in partitions[i]) for i in range(k)]
        expanded_this_round = False

        current_total = sum(loads)
        mean_scale = (current_total / k) if k > 0 else 0.0
        scale_raw = [(ld / mean_scale) if mean_scale > 0 else 0.0 for ld in loads]
        sigma = scale_raw
        max_scale = max(scale_raw) if scale_raw else 0.0
        scale_norm = [(s / max_scale) if max_scale > 0 else 0.0 for s in scale_raw]

        frontiers: List[Set[frozenset]] = [set() for _ in range(k)]
        edge_competitors: Dict[frozenset, Set[int]] = {}
        for idx in range(k):
            if not active[idx]:
                continue
            for e in partitions[idx]:
                for nb_e in edge_adj[e]:
                    if nb_e in assigned:
                        continue
                    frontiers[idx].add(nb_e)
                    edge_competitors.setdefault(nb_e, set()).add(idx)

            if not frontiers[idx]:
                active[idx] = False

        proposals: Dict[int, frozenset] = {}
        for idx in range(k):
            if not active[idx]:
                continue

            best_edge = None
            best_cost = float("inf")
            for candidate in frontiers[idx]:
                competitors = edge_competitors.get(candidate, set()) or {idx}
                dist_values = [
                    _euclidean(edge_centers[candidate], seed_centers[c]) for c in competitors
                ]
                max_dist = max(dist_values) if dist_values else 0.0
                dist = _euclidean(edge_centers[candidate], seed_centers[idx])
                dist_norm = (dist / max_dist) if max_dist > 0 else 0.0
                cost = alpha * scale_norm[idx] + beta * dist_norm

                if cost < best_cost:
                    best_cost = cost
                    best_edge = candidate

            if best_edge is not None:
                proposals[idx] = best_edge

        if not proposals:
            break

        edge_to_proposers: Dict[frozenset, List[int]] = {}
        for idx, edge in proposals.items():
            edge_to_proposers.setdefault(edge, []).append(idx)

        for edge, proposers in edge_to_proposers.items():
            competitors = edge_competitors.get(edge, set())
            if len(competitors) >= 2:
                candidates = competitors
            else:
                candidates = set(proposers)

            non_overloaded = {i for i in candidates if sigma[i] <= overload_threshold}
            gated_candidates = non_overloaded if non_overloaded else candidates

            dist_by_candidate = {
                i: _euclidean(edge_centers[edge], seed_centers[i]) for i in gated_candidates
            }
            max_dist = max(dist_by_candidate.values()) if dist_by_candidate else 0.0

            def _cost(i: int) -> Tuple[float, float, int]:
                dist_norm = (dist_by_candidate[i] / max_dist) if max_dist > 0 else 0.0
                value = alpha * scale_norm[i] + beta * dist_norm
                return (value, loads[i], i)

            if non_overloaded:
                winner = min(gated_candidates, key=_cost)
            else:
                # Fallback when all adjacent candidates are overloaded.
                winner = min(gated_candidates, key=lambda i: (loads[i], i))
            if edge not in assigned:
                partitions[winner].add(edge)
                assigned.add(edge)
                expanded_this_round = True

        if not expanded_this_round:
            break

    for u, v in graph.edges():
        eid = edge_id(u, v)
        if eid not in assigned:
            min_idx = min(range(k), key=lambda i: sum(edge_load(graph, *e, weight) for e in partitions[i]))
            partitions[min_idx].add(eid)
            assigned.add(eid)

    for _ in range(refine_iterations):
        loads = [sum(edge_load(graph, *e, weight) for e in partitions[i]) for i in range(k)]
        max_idx = max(range(k), key=lambda i: loads[i])
        min_idx = min(range(k), key=lambda i: loads[i])

        if max_idx == min_idx:
            break
        imbalance = loads[max_idx] - loads[min_idx]
        if imbalance < target_load * 0.01:
            break

        improved = False
        for edge in list(partitions[max_idx]):
            has_adj_min = any(nb in partitions[min_idx] for nb in edge_adj[edge])
            if not has_adj_min:
                continue

            trial = partitions[max_idx] - {edge}
            if not trial or not _edges_connected(trial, edge_adj):
                continue

            trial_min = partitions[min_idx] | {edge}
            if not _edges_connected(trial_min, edge_adj):
                continue

            e_load = edge_load(graph, *edge, weight)
            new_max_load = loads[max_idx] - e_load
            new_min_load = loads[min_idx] + e_load
            if abs(new_max_load - new_min_load) < imbalance:
                partitions[max_idx].remove(edge)
                partitions[min_idx].add(edge)
                improved = True
                break

        if not improved:
            break

    return partitions


# Backward-compatible alias.
neighbor_expansion_partition = cne_partition

