from __future__ import annotations

from cne.algorithms.neighbor_expansion import (
    _build_edge_adjacency,
    _edges_connected,
    neighbor_expansion_partition,
)
from cne.graph.generators import build_grid_road_network
from cne.utils.edge import edge_id


def test_partition_covers_all_edges_without_overlap():
    graph = build_grid_road_network(rows=4, cols=5, seed=1)
    partitions = neighbor_expansion_partition(graph, k=3)

    assigned = set()
    for part in partitions:
        for e in part:
            assert e not in assigned
            assigned.add(e)

    all_edges = {edge_id(u, v) for u, v in graph.edges()}
    assert assigned == all_edges


def test_each_partition_is_edge_connected_when_nonempty():
    graph = build_grid_road_network(rows=5, cols=6, seed=2)
    partitions = neighbor_expansion_partition(graph, k=4)
    edge_adj = _build_edge_adjacency(graph)

    for part in partitions:
        if part:
            assert _edges_connected(part, edge_adj)
