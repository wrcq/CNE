"""Run pure k-medoids edge partitioning from external CSV road-network files.

Expected files:
- nodes.csv: node_id,x,y
- edges.csv: u,v,weight
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
import time

import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cne.algorithms import kmedoids_partition_with_medoids
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


def centers_from_edges(
    graph: nx.Graph,
    edges: list[frozenset],
) -> list[tuple[float, float]]:
    """Return center points for actual representative edges."""
    out: list[tuple[float, float]] = []
    for e in edges:
        u, v = tuple(e)
        pu = graph.nodes[u]["pos"]
        pv = graph.nodes[v]["pos"]
        cx = (float(pu[0]) + float(pv[0])) * 0.5
        cy = (float(pu[1]) + float(pv[1])) * 0.5
        out.append((cx, cy))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pure k-medoids from external CSV road data")
    parser.add_argument("--nodes", default="dataset/processed/nodes.csv", help="Path to nodes.csv")
    parser.add_argument("--edges", default="dataset/processed/edges.csv", help="Path to edges.csv")
    parser.add_argument("-k", type=int, default=3, help="Number of clusters")
    parser.add_argument("--max-iter", type=int, default=20, help="k-medoids max iterations")
    parser.add_argument(
        "--show-edge-labels",
        action="store_true",
        help="show edge weight labels on plot",
    )
    args = parser.parse_args()

    nodes_csv = ROOT / args.nodes
    edges_csv = ROOT / args.edges


    start = time.time()

    graph, pos = load_graph_from_csv(nodes_csv, edges_csv)
    partitions, medoid_edges = kmedoids_partition_with_medoids(
        graph,
        k=args.k,
        max_iter=args.max_iter,
    )

    stats = partition_stats(graph, partitions)
    centers = centers_from_edges(graph, medoid_edges)

    print("=" * 65)
    print("外部 CSV 路网 K-Medoids 边聚类结果")
    print("=" * 65)
    print(f"节点数: {graph.number_of_nodes()}, 边数: {graph.number_of_edges()}")
    print(f"参数: k={args.k}, max_iter={args.max_iter}")
    print(f"各子图边数: {stats['edge_counts']}")
    print(f"各子图负载: {[f'{x:.1f}' for x in stats['loads']]}")
    print(f"不均衡度: {stats['max_imbalance']:.2%}, 共享节点: {stats['shared_nodes']}")
    if medoid_edges:
        print("代表边(最终 medoids):")
        for i, e in enumerate(medoid_edges):
            u, v = sorted(tuple(e))
            print(f"  medoid[{i}] edge=({u},{v})")

    out_img = OUTPUT_DIR / f"external_kmedoids_k{args.k}.png"
    draw_partitioned_graph(
        graph,
        partitions,
        pos=pos,
        seed_centers=centers,
        title=f"外部路网 K-Medoids 边聚类 (K={args.k})",
        show_edge_labels=args.show_edge_labels,
        save_path=os.path.join(str(OUTPUT_DIR), out_img.name),
    )
    end = time.time()
    print(f"运行时间: {end - start:.2f} 秒")

if __name__ == "__main__":
    main()
