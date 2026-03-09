"""
NE 路网边负载均衡分割 — 演示脚本
==================================
生成示例路网，运行 NE 边分割算法，并绘制可视化结果。
边是分配对象，节点可被多个子图共享。
"""

import networkx as nx
import random
import os
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

from ne_algorithm import neighbor_expansion_partition, partition_stats, _edge_id
from visualize import draw_partitioned_graph


def build_grid_road_network(rows: int = 5, cols: int = 6,
                            seed: int = 42) -> nx.Graph:
    """生成一个网格状路网，边权重（负载）随机在 [1, 10] 之间"""
    rng = random.Random(seed)
    G = nx.grid_2d_graph(rows, cols)
    mapping = {(r, c): r * cols + c for r, c in G.nodes()}
    G = nx.relabel_nodes(G, mapping)
    for u, v in G.edges():
        G[u][v]["weight"] = round(rng.uniform(1.0, 10.0), 1)
    return G


def build_random_road_network(n: int = 30, extra_edges: int = 15,
                              seed: int = 42) -> nx.Graph:
    """生成一个随机路网：先用随机树保证连通，再加一些额外边"""
    rng = random.Random(seed)
    positions = {i: (rng.uniform(0, 10), rng.uniform(0, 10)) for i in range(n)}

    G = nx.Graph()
    G.add_nodes_from(range(n))

    tree = nx.random_tree(n, seed=seed)
    for u, v in tree.edges():
        G.add_edge(u, v, weight=round(rng.uniform(1.0, 10.0), 1))

    all_possible = [(i, j) for i in range(n) for j in range(i + 1, n)
                    if not G.has_edge(i, j)]
    rng.shuffle(all_possible)
    for u, v in all_possible[:extra_edges]:
        G.add_edge(u, v, weight=round(rng.uniform(1.0, 10.0), 1))

    nx.set_node_attributes(G, positions, "pos")
    return G


def print_stats(graph, partitions, label=""):
    """打印边分割统计信息"""
    stats = partition_stats(graph, partitions)
    print(f"\n{'─' * 55}")
    print(f"边分割统计 {label}")
    print(f"{'─' * 55}")
    print(f"  子图数量:     {len(partitions)}")
    print(f"  总边数:       {graph.number_of_edges()}")
    print(f"  各子图边数:   {stats['edge_counts']}")
    print(f"  各子图负载:   {[f'{l:.1f}' for l in stats['loads']]}")
    print(f"  平均负载:     {stats['mean_load']:.1f}")
    print(f"  不均衡度:     {stats['max_imbalance']:.2%}")
    print(f"  共享节点数:   {stats['shared_nodes']}")
    for i, part in enumerate(partitions):
        edges_str = [f"({min(e)},{max(e)})" for e in sorted(part, key=lambda e: tuple(sorted(e)))]
        print(f"  子图 {i}: {len(part)} 条边, 节点 {stats['partition_nodes'][i]}")


def demo_grid():
    """网格路网边分割示例"""
    print("=" * 60)
    print("示例 1: 5×6 网格路网 -> 边分割为 3 个子图")
    print("=" * 60)

    G = build_grid_road_network(5, 6)
    print(f"节点数: {G.number_of_nodes()}, 边数: {G.number_of_edges()}")
    total_load = sum(d["weight"] for _, _, d in G.edges(data=True))
    print(f"总负载: {total_load:.1f}")

    partitions = neighbor_expansion_partition(G, k=3)
    print_stats(G, partitions, "(网格路网, K=3)")

    pos = {i: (i % 6, -(i // 6)) for i in G.nodes()}
    save_path = os.path.join(OUTPUT_DIR, "result_grid.png")
    draw_partitioned_graph(
        G, partitions, pos=pos,
        title="网格路网 NE 边负载均衡分割 (K=3)",
        save_path=save_path,
    )


def demo_random():
    """随机路网边分割示例"""
    print("\n" + "=" * 60)
    print("示例 2: 随机路网 (30 节点) -> 边分割为 4 个子图")
    print("=" * 60)

    G = build_random_road_network(n=30, extra_edges=15)
    print(f"节点数: {G.number_of_nodes()}, 边数: {G.number_of_edges()}")
    total_load = sum(d["weight"] for _, _, d in G.edges(data=True))
    print(f"总负载: {total_load:.1f}")

    partitions = neighbor_expansion_partition(G, k=4)
    print_stats(G, partitions, "(随机路网, K=4)")

    pos = nx.get_node_attributes(G, "pos")
    save_path = os.path.join(OUTPUT_DIR, "result_random.png")
    draw_partitioned_graph(
        G, partitions, pos=pos,
        title="随机路网 NE 边负载均衡分割 (K=4)",
        save_path=save_path,
    )


def demo_k_comparison():
    """不同 K 值对比"""
    print("\n" + "=" * 60)
    print("示例 3: 不同 K 值对比 (网格路网)")
    print("=" * 60)

    G = build_grid_road_network(5, 6)

    for k in [2, 3, 4, 5]:
        partitions = neighbor_expansion_partition(G, k=k)
        stats = partition_stats(G, partitions)
        loads_str = ", ".join(f"{l:.1f}" for l in stats["loads"])
        print(f"  K={k}: 负载=[{loads_str}], "
              f"不均衡度={stats['max_imbalance']:.2%}, "
              f"共享节点={stats['shared_nodes']}")


if __name__ == "__main__":
    demo_grid()
    demo_random()
    demo_k_comparison()
    print("\n✓ 完成！图片已保存到当前目录")
