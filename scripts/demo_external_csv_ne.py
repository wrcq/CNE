"""Run NE partitioning from external CSV road-network files.

Expected files:
- nodes.csv: node_id,x,y
- edges.csv: u,v,weight (support_count column is optional and ignored)
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cne.algorithms import ne_partition
from cne.algorithms.cne import _build_edge_adjacency, _select_seed_edges
from cne.analysis import partition_stats
from cne.viz import draw_partitioned_graph


def load_graph_from_csv(nodes_csv: Path, edges_csv: Path) -> tuple[nx.Graph, dict]:
    graph = nx.Graph()
    pos = {}

    with nodes_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            node_id = int(row["node_id"])
            x = float(row["x"])
            y = float(row["y"])
            graph.add_node(node_id, pos=(x, y))
            pos[node_id] = (x, y)

    with edges_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            u = int(row["u"])
            v = int(row["v"])
            w = float(row.get("weight", 1.0))
            graph.add_edge(u, v, weight=w)

    return graph, pos


def select_seed_edges(
    graph: nx.Graph,
    k: int,
    weight: str = "weight",
) -> list[tuple[int, int]]:
    n_edges = graph.number_of_edges()
    if n_edges == 0:
        return []

    k_eff = min(k, n_edges)
    edge_adj = _build_edge_adjacency(graph)
    seed_edges = _select_seed_edges(graph, edge_adj, k_eff, weight)
    return [tuple(sorted(e)) for e in seed_edges]


def seed_centers_from_edges(
    graph: nx.Graph,
    seeds: list[tuple[int, int]],
) -> list[tuple[int, int, float, float]]:
    out = []
    for u, v in seeds:
        pu = graph.nodes[u].get("pos")
        pv = graph.nodes[v].get("pos")
        if pu is None or pv is None:
            continue
        cx = (float(pu[0]) + float(pv[0])) * 0.5
        cy = (float(pu[1]) + float(pv[1])) * 0.5
        out.append((u, v, cx, cy))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NE from external CSV road data")
    parser.add_argument("--nodes", default="dataset/processed/nodes.csv", help="Path to nodes.csv")
    parser.add_argument("--edges", default="dataset/processed/edges.csv", help="Path to edges.csv")
    parser.add_argument("-k", type=int, default=4, help="Number of partitions")
    parser.add_argument(
        "--show-edge-labels",
        action="store_true",
        help="show edge weight labels on plot",
    )
    args = parser.parse_args()

    nodes_csv = ROOT / args.nodes
    edges_csv = ROOT / args.edges

    graph, pos = load_graph_from_csv(nodes_csv, edges_csv)
    seed_edges = select_seed_edges(graph, args.k)
    seed_centers = seed_centers_from_edges(graph, seed_edges)

    partitions = ne_partition(
        graph,
        k=args.k,
        seed_edges=seed_edges,
    )

    stats = partition_stats(graph, partitions)
    print("=" * 65)
    print("外部 CSV 路网 NE 分区结果")
    print("=" * 65)
    print(f"节点数: {graph.number_of_nodes()}, 边数: {graph.number_of_edges()}")
    print(f"参数: k={args.k}, seed=bfs-dispersed")
    print(f"各子图边数: {stats['edge_counts']}")
    print(f"各子图负载: {[f'{x:.1f}' for x in stats['loads']]}")
    print(f"不均衡度: {stats['max_imbalance']:.2%}, 共享节点: {stats['shared_nodes']}")
    if seed_centers:
        print("种子点(由 seed edge 中点计算):")
        for i, (u, v, cx, cy) in enumerate(seed_centers):
            print(f"  seed[{i}] edge=({u},{v}) center=({cx:.8f},{cy:.8f})")

    out_img = OUTPUT_DIR / f"external_ne_k{args.k}.png"
    draw_partitioned_graph(
        graph,
        partitions,
        pos=pos,
        seed_centers=[(cx, cy) for _, _, cx, cy in seed_centers],
        title=f"外部路网 NE 分区 (K={args.k}, seed=bfs-dispersed)",
        show_edge_labels=args.show_edge_labels,
        save_path=os.path.join(str(OUTPUT_DIR), out_img.name),
    )


if __name__ == "__main__":
    main()
