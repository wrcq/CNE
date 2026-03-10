"""Sample road-network generators."""

from __future__ import annotations

import random

import networkx as nx


def build_grid_road_network(rows: int = 5, cols: int = 6, seed: int = 42) -> nx.Graph:
    """Build a grid-like road network with random edge loads in [1, 10]."""
    rng = random.Random(seed)
    graph = nx.grid_2d_graph(rows, cols)
    mapping = {(r, c): r * cols + c for r, c in graph.nodes()}
    graph = nx.relabel_nodes(graph, mapping)
    for u, v in graph.edges():
        graph[u][v]["weight"] = round(rng.uniform(1.0, 10.0), 1)
    return graph


def build_random_road_network(n: int = 30, extra_edges: int = 15, seed: int = 42) -> nx.Graph:
    """Build a connected random road network using tree + additional edges."""
    rng = random.Random(seed)
    positions = {i: (rng.uniform(0, 10), rng.uniform(0, 10)) for i in range(n)}

    graph = nx.Graph()
    graph.add_nodes_from(range(n))

    try:
        tree = nx.random_labeled_tree(n, seed=seed)
    except AttributeError:
        tree = nx.random_tree(n, seed=seed)
    for u, v in tree.edges():
        graph.add_edge(u, v, weight=round(rng.uniform(1.0, 10.0), 1))

    all_possible = [
        (i, j)
        for i in range(n)
        for j in range(i + 1, n)
        if not graph.has_edge(i, j)
    ]
    rng.shuffle(all_possible)
    for u, v in all_possible[:extra_edges]:
        graph.add_edge(u, v, weight=round(rng.uniform(1.0, 10.0), 1))

    nx.set_node_attributes(graph, positions, "pos")
    return graph
