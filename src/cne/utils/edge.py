"""Edge utilities shared across modules."""

from __future__ import annotations

import networkx as nx


def edge_id(u, v) -> frozenset:
    """Return an unordered edge identifier for undirected graphs."""
    return frozenset((u, v))


def edge_load(graph: nx.Graph, u, v, weight: str = "weight") -> float:
    """Read edge load from graph, defaulting to 1.0 when absent."""
    return graph[u][v].get(weight, 1.0)


# Backward-compatible aliases
_edge_id = edge_id
_edge_load = edge_load
