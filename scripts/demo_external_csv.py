"""Run CNE partitioning from external CSV road-network files.

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

from cne.algorithms import cne_partition
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CNE from external CSV road data")
    parser.add_argument("--nodes", default="dataset/processed/nodes.csv", help="Path to nodes.csv")
    parser.add_argument("--edges", default="dataset/processed/edges.csv", help="Path to edges.csv")
    parser.add_argument("-k", type=int, default=4, help="Number of partitions")
    parser.add_argument("--alpha", type=float, default=1.0, help="weight for scale/load term")
    parser.add_argument("--beta", type=float, default=1.0, help="weight for distance term")
    parser.add_argument(
        "--overload-threshold",
        type=float,
        default=1.2,
        help="relative load intensity threshold for gating overloaded partitions",
    )
    parser.add_argument(
        "--show-edge-labels",
        action="store_true",
        help="show edge weight labels on plot",
    )
    args = parser.parse_args()

    nodes_csv = ROOT / args.nodes
    edges_csv = ROOT / args.edges

    graph, pos = load_graph_from_csv(nodes_csv, edges_csv)
    partitions = cne_partition(
        graph,
        k=args.k,
        alpha=args.alpha,
        beta=args.beta,
        overload_threshold=args.overload_threshold,
    )

    stats = partition_stats(graph, partitions)
    print("=" * 65)
    print("外部 CSV 路网 CNE 分区结果")
    print("=" * 65)
    print(f"节点数: {graph.number_of_nodes()}, 边数: {graph.number_of_edges()}")
    print(
        f"参数: k={args.k}, alpha={args.alpha}, beta={args.beta}, "
        f"lambda={args.overload_threshold}"
    )
    print(f"各子图边数: {stats['edge_counts']}")
    print(f"各子图负载: {[f'{x:.1f}' for x in stats['loads']]}")
    print(f"不均衡度: {stats['max_imbalance']:.2%}, 共享节点: {stats['shared_nodes']}")

    out_img = OUTPUT_DIR / f"external_cne_k{args.k}.png"
    draw_partitioned_graph(
        graph,
        partitions,
        pos=pos,
        title=(
            f"外部路网 CNE 分区 (K={args.k}, a={args.alpha}, "
            f"b={args.beta}, l={args.overload_threshold})"
        ),
        show_edge_labels=args.show_edge_labels,
        save_path=os.path.join(str(OUTPUT_DIR), out_img.name),
    )


if __name__ == "__main__":
    main()
