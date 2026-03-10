"""Neighbor Expansion (NE) single-source sequential edge partitioning algorithm."""

from __future__ import annotations

from typing import List, Optional, Set, Tuple

import networkx as nx

from cne.algorithms.cne import _build_edge_adjacency, _edges_connected, _select_seed_edges
from cne.utils.edge import edge_id, edge_load


def ne_partition(
    graph: nx.Graph,
    k: int,
    weight: str = "weight",
    seed_edges: Optional[List[Tuple]] = None,
    refine_iterations: int = 50,
) -> List[Set[frozenset]]:
    """Partition edges with single-source sequential expansion.

    Different from CNE multi-source synchronized expansion, each source expands
    continuously until its load is close to target, then the next source expands.
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

    total_load = sum(d.get(weight, 1.0) for _, _, d in graph.edges(data=True))
    target_load = total_load / k
    tolerance = target_load * 0.05

    active = [True] * k
    while len(assigned) < n_edges and any(active):
        expanded_any = False

        for idx in range(k):
            if not active[idx]:
                continue

            while True:
                current_load = sum(edge_load(graph, *e, weight) for e in partitions[idx])

                frontier: Set[frozenset] = set()
                for e in partitions[idx]:
                    for nb_e in edge_adj[e]:
                        if nb_e not in assigned:
                            frontier.add(nb_e)

                if not frontier:
                    active[idx] = False
                    break

                current_diff = abs(current_load - target_load)
                best_edge = None
                best_new_diff = float("inf")

                for candidate in frontier:
                    new_load = current_load + edge_load(graph, *candidate, weight)
                    new_diff = abs(new_load - target_load)
                    if new_diff < best_new_diff:
                        best_new_diff = new_diff
                        best_edge = candidate

                if best_edge is None:
                    active[idx] = False
                    break

                if current_diff <= tolerance:
                    break
                if current_load >= target_load and best_new_diff >= current_diff:
                    break

                partitions[idx].add(best_edge)
                assigned.add(best_edge)
                expanded_any = True

                if abs(
                    sum(edge_load(graph, *e, weight) for e in partitions[idx]) - target_load
                ) <= tolerance:
                    break

        if not expanded_any:
            break

    for u, v in graph.edges():
        eid = edge_id(u, v)
        if eid in assigned:
            continue

        # Prefer attaching to an adjacent partition so connectivity is preserved.
        adjacent_indices = [
            i for i in range(k) if partitions[i] and any(nb in partitions[i] for nb in edge_adj[eid])
        ]
        if adjacent_indices:
            target_idx = min(
                adjacent_indices,
                key=lambda i: sum(edge_load(graph, *e, weight) for e in partitions[i]),
            )
        else:
            target_idx = min(
                range(k),
                key=lambda i: sum(edge_load(graph, *e, weight) for e in partitions[i]),
            )

        partitions[target_idx].add(eid)
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
