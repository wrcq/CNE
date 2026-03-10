"""Constrained Neighbor Expansion (CNE) edge partitioning algorithm.

This module implements the existing multi-source synchronized expansion strategy.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

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


def cne_partition(
    graph: nx.Graph,
    k: int,
    weight: str = "weight",
    seed_edges: Optional[List[Tuple]] = None,
    refine_iterations: int = 50,
) -> List[Set[frozenset]]:
    """Partition edges into k balanced connected groups using CNE multi-source expansion."""
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

    total_load = sum(d.get(weight, 1.0) for _, _, d in graph.edges(data=True))
    target_load = total_load / k
    active = [True] * k

    while len(assigned) < n_edges and any(active):
        loads = [sum(edge_load(graph, *e, weight) for e in partitions[i]) for i in range(k)]
        order = sorted(range(k), key=lambda i: loads[i])
        expanded_this_round = False

        for idx in order:
            if not active[idx]:
                continue

            frontier: Set[frozenset] = set()
            for e in partitions[idx]:
                for nb_e in edge_adj[e]:
                    if nb_e not in assigned:
                        frontier.add(nb_e)

            if not frontier:
                active[idx] = False
                continue

            current_load = loads[idx]
            best_edge = None
            best_diff = float("inf")

            for candidate in frontier:
                new_load = current_load + edge_load(graph, *candidate, weight)
                diff = abs(new_load - target_load)
                if diff < best_diff:
                    best_diff = diff
                    best_edge = candidate

            if best_edge is not None:
                partitions[idx].add(best_edge)
                assigned.add(best_edge)
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

