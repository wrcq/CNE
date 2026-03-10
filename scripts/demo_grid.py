"""Grid-network demo for NE partitioning."""

from __future__ import annotations

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
from cne.graph import build_grid_road_network
from cne.viz import draw_partitioned_graph


def print_stats(graph, partitions, label=""):
    stats = partition_stats(graph, partitions)
    print(f"\n{'-' * 55}")
    print(f"边分割统计 {label}")
    print(f"{'-' * 55}")
    print(f"  子图数量:     {len(partitions)}")
    print(f"  总边数:       {graph.number_of_edges()}")
    print(f"  各子图边数:   {stats['edge_counts']}")
    print(f"  各子图负载:   {[f'{l:.1f}' for l in stats['loads']]}")
    print(f"  平均负载:     {stats['mean_load']:.1f}")
    print(f"  不均衡度:     {stats['max_imbalance']:.2%}")
    print(f"  共享节点数:   {stats['shared_nodes']}")


def main():
    print("=" * 60)
    print("示例 1: 5x6 网格路网 -> 边分割为 3 个子图")
    print("=" * 60)

    graph = build_grid_road_network(5, 6)
    total_load = sum(d["weight"] for _, _, d in graph.edges(data=True))
    print(f"节点数: {graph.number_of_nodes()}, 边数: {graph.number_of_edges()}")
    print(f"总负载: {total_load:.1f}")

    partitions = cne_partition(graph, k=3)
    print_stats(graph, partitions, "(网格路网, K=3)")

    pos = {i: (i % 6, -(i // 6)) for i in graph.nodes()}
    draw_partitioned_graph(
        graph,
        partitions,
        pos=pos,
        title="网格路网 NE 边负载均衡分割 (K=3)",
        save_path=os.path.join(str(OUTPUT_DIR), "result_grid.png"),
    )


if __name__ == "__main__":
    main()
