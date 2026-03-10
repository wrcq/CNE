from __future__ import annotations

import networkx as nx

from cne.algorithms.cne import (
    _build_edge_adjacency,
    _edges_connected,
    cne_partition,
)
from cne.algorithms.ne import ne_partition
from cne.graph.generators import build_grid_road_network
from cne.utils.edge import edge_id


def test_partition_covers_all_edges_without_overlap():
    graph = build_grid_road_network(rows=4, cols=5, seed=1)
    partitions = cne_partition(graph, k=3)

    assigned = set()
    for part in partitions:
        for e in part:
            assert e not in assigned
            assigned.add(e)

    all_edges = {edge_id(u, v) for u, v in graph.edges()}
    assert assigned == all_edges


def test_each_partition_is_edge_connected_when_nonempty():
    graph = build_grid_road_network(rows=5, cols=6, seed=2)
    partitions = cne_partition(graph, k=4)
    edge_adj = _build_edge_adjacency(graph)

    for part in partitions:
        if part:
            assert _edges_connected(part, edge_adj)


def test_ne_partition_covers_all_edges_and_preserves_connectivity():
    graph = build_grid_road_network(rows=5, cols=6, seed=4)
    partitions = ne_partition(graph, k=4)
    edge_adj = _build_edge_adjacency(graph)

    assigned = set()
    for part in partitions:
        for e in part:
            assert e not in assigned
            assigned.add(e)
        if part:
            assert _edges_connected(part, edge_adj)

    all_edges = {edge_id(u, v) for u, v in graph.edges()}
    assert assigned == all_edges


def test_competitive_assignment_prefers_closer_seed_when_distance_dominates():
    graph = nx.Graph()
    graph.add_edge(0, 1, weight=1.0)
    graph.add_edge(2, 3, weight=1.0)
    graph.add_edge(1, 2, weight=1.0)
    nx.set_node_attributes(
        graph,
        {
            0: (0.0, 0.0),
            1: (1.0, 0.0),
            2: (1.2, 0.0),
            3: (5.0, 0.0),
        },
        "pos",
    )

    partitions = cne_partition(
        graph,
        k=2,
        seed_edges=[(0, 1), (2, 3)],
        refine_iterations=0,
        alpha=0.0,
        beta=1.0,
    )

    contested = edge_id(1, 2)
    assert contested in partitions[0]
    assert contested not in partitions[1]


def test_competitive_assignment_prefers_lighter_partition_when_scale_dominates():
    graph = nx.Graph()
    graph.add_edge(0, 1, weight=5.0)
    graph.add_edge(2, 3, weight=1.0)
    graph.add_edge(1, 2, weight=1.0)
    nx.set_node_attributes(
        graph,
        {
            0: (0.0, 0.0),
            1: (1.0, 0.0),
            2: (2.0, 0.0),
            3: (3.0, 0.0),
        },
        "pos",
    )

    partitions = cne_partition(
        graph,
        k=2,
        seed_edges=[(0, 1), (2, 3)],
        refine_iterations=0,
        alpha=1.0,
        beta=0.0,
    )

    contested = edge_id(1, 2)
    assert contested in partitions[1]
    assert contested not in partitions[0]


def test_overload_gating_blocks_significantly_overloaded_partition():
    graph = nx.Graph()
    graph.add_edge(0, 1, weight=10.0)
    graph.add_edge(2, 3, weight=1.0)
    graph.add_edge(1, 2, weight=1.0)
    nx.set_node_attributes(
        graph,
        {
            0: (0.0, 0.0),
            1: (1.0, 0.0),
            2: (2.0, 0.0),
            3: (3.0, 0.0),
        },
        "pos",
    )

    partitions = cne_partition(
        graph,
        k=2,
        seed_edges=[(0, 1), (2, 3)],
        refine_iterations=0,
        alpha=0.0,
        beta=1.0,
        overload_threshold=1.2,
    )

    contested = edge_id(1, 2)
    assert contested in partitions[1]
    assert contested not in partitions[0]
