from __future__ import annotations

from cne.graph.generators import build_grid_road_network, build_random_road_network


def test_grid_generator_sizes():
    graph = build_grid_road_network(rows=5, cols=6, seed=42)
    assert graph.number_of_nodes() == 30
    assert graph.number_of_edges() == 49
    assert all("weight" in graph[u][v] for u, v in graph.edges())


def test_random_generator_is_connected():
    graph = build_random_road_network(n=20, extra_edges=10, seed=7)
    assert graph.number_of_nodes() == 20
    assert graph.number_of_edges() == 29
    assert all("weight" in graph[u][v] for u, v in graph.edges())
