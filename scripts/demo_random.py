"""Random-network demo for NE partitioning."""

from __future__ import annotations

import argparse
import networkx as nx
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cne.algorithms import cne_partition
from cne.analysis import partition_stats
from cne.graph import build_random_road_network
from cne.viz import draw_partitioned_graph


def main():
    parser = argparse.ArgumentParser(description="Random-network demo for competitive CNE")
    parser.add_argument("--alpha", type=float, default=1.0, help="weight for scale/load term")
    parser.add_argument("--beta", type=float, default=1.0, help="weight for distance term")
    parser.add_argument(
        "--overload-threshold",
        type=float,
        default=1.2,
        help="relative load intensity threshold for gating overloaded partitions",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("示例 2: 随机路网 (30 节点) -> 边分割为 4 个子图")
    print("=" * 60)

    graph = build_random_road_network(n=30, extra_edges=15)
    total_load = sum(d["weight"] for _, _, d in graph.edges(data=True))
    print(f"节点数: {graph.number_of_nodes()}, 边数: {graph.number_of_edges()}")
    print(f"总负载: {total_load:.1f}")

    partitions = cne_partition(
        graph,
        k=4,
        alpha=args.alpha,
        beta=args.beta,
        overload_threshold=args.overload_threshold,
    )
    stats = partition_stats(graph, partitions)
    print(
        f"  参数: alpha={args.alpha}, beta={args.beta}, "
        f"lambda={args.overload_threshold}"
    )
    print(f"  各子图负载: {[f'{l:.1f}' for l in stats['loads']]}")
    print(f"  不均衡度: {stats['max_imbalance']:.2%}, 共享节点: {stats['shared_nodes']}")

    pos = nx.get_node_attributes(graph, "pos")
    draw_partitioned_graph(
        graph,
        partitions,
        pos=pos,
        title="随机路网 NE 边负载均衡分割 (K=4)",
        save_path=os.path.join(str(OUTPUT_DIR), "result_random.png"),
    )


if __name__ == "__main__":
    main()
