from __future__ import annotations

from cne.algorithms import cne_partition
from cne.analysis import partition_stats
from cne.graph import build_grid_road_network


def test_partition_stats_shape_and_ranges():
    graph = build_grid_road_network(rows=4, cols=4, seed=3)
    partitions = cne_partition(graph, k=3)
    stats = partition_stats(graph, partitions)

    assert len(stats["loads"]) == 3
    assert len(stats["edge_counts"]) == 3
    assert len(stats["partition_nodes"]) == 3
    assert stats["mean_load"] > 0
    assert stats["max_imbalance"] >= 0
    assert stats["shared_nodes"] >= 0

    assert sum(stats["edge_counts"]) == graph.number_of_edges()
