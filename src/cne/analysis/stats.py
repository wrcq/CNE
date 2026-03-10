"""Statistics helpers for edge partition results."""

from __future__ import annotations

from typing import Dict, List, Set

import networkx as nx

from cne.utils.edge import edge_load


def partition_stats(
    graph: nx.Graph,
    partitions: List[Set[frozenset]],
    weight: str = "weight",
) -> Dict:
    """Return core statistics for edge-partition results."""
    loads = []
    edge_counts = []
    for part in partitions:
        load = sum(edge_load(graph, *e, weight) for e in part)
        loads.append(load)
        edge_counts.append(len(part))

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

    return {
        "loads": loads,
        "edge_counts": edge_counts,
        "mean_load": mean_load,
        "max_imbalance": imbalance,
        "shared_nodes": len(shared),
        "partition_nodes": [sorted(ns) for ns in part_nodes],
    }
