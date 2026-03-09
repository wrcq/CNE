"""
邻居扩展算法 (Neighbor Expansion, NE) — 路网边负载均衡分割
============================================================
分配对象是 **边**（而非节点）。节点可被多个子图共享，负载全在边上。
目标：将路网的所有边分割为 K 个组，使各组的边负载总和尽可能相近，
且每个组的边构成连通子图。

算法流程:
  1. 选取 K 条尽量分散的种子边
  2. 多源同步扩展：每轮让负载最小的分区优先，从当前边集的
     相邻边（共享端点）中选择一条加入
  3. 局部调整 (refinement)：在保持连通性的前提下，将边界边
     从重负载分区移到轻负载分区
"""

from __future__ import annotations

import networkx as nx
from typing import List, Set, Dict, Tuple, Optional


# ──────────────────────────────────────────────
#  边的表示：统一用 frozenset({u, v}) 作为边标识
# ──────────────────────────────────────────────

def _edge_id(u, v) -> frozenset:
    return frozenset((u, v))


def _edge_load(graph: nx.Graph, u, v, weight: str = "weight") -> float:
    return graph[u][v].get(weight, 1.0)


# ──────────────────────────────────────────────
#  边邻接关系：两条边相邻 ⇔ 共享一个端点
# ──────────────────────────────────────────────

def _build_edge_adjacency(graph: nx.Graph) -> Dict[frozenset, Set[frozenset]]:
    """构建边邻接表：每条边 -> 与其共享端点的所有边"""
    adj: Dict[frozenset, Set[frozenset]] = {}
    for u, v in graph.edges():
        eid = _edge_id(u, v)
        adj[eid] = set()

    for node in graph.nodes():
        incident = [_edge_id(node, nb) for nb in graph.neighbors(node)]
        for i, e1 in enumerate(incident):
            for e2 in incident[i + 1:]:
                adj[e1].add(e2)
                adj[e2].add(e1)
    return adj


# ──────────────────────────────────────────────
#  种子选择：选取 K 条尽量分散的种子边
# ──────────────────────────────────────────────

def _select_seed_edges(
    graph: nx.Graph,
    edge_adj: Dict[frozenset, Set[frozenset]],
    k: int,
) -> List[frozenset]:
    """贪心选取 K 条相互尽量远的种子边（基于边邻接图 BFS 距离）"""
    all_edges = list(edge_adj.keys())
    if k >= len(all_edges):
        return all_edges[:k]

    def bfs_dist(start: frozenset, target: frozenset) -> int:
        if start == target:
            return 0
        visited = {start}
        queue = [start]
        dist = 0
        while queue:
            dist += 1
            next_queue = []
            for e in queue:
                for nb in edge_adj[e]:
                    if nb == target:
                        return dist
                    if nb not in visited:
                        visited.add(nb)
                        next_queue.append(nb)
            queue = next_queue
        return float("inf")

    # 第一条种子边：负载最大的边（通常是 "热点"）
    seeds: List[frozenset] = [
        max(all_edges, key=lambda e: _edge_load(graph, *e))
    ]

    while len(seeds) < k:
        best_edge = None
        best_min_dist = -1
        for e in all_edges:
            if e in seeds:
                continue
            min_d = min(bfs_dist(e, s) for s in seeds)
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_edge = e
        if best_edge is not None:
            seeds.append(best_edge)
        else:
            break
    return seeds


# ──────────────────────────────────────────────
#  检查边集的连通性
# ──────────────────────────────────────────────

def _edges_connected(edge_set: Set[frozenset],
                     edge_adj: Dict[frozenset, Set[frozenset]]) -> bool:
    """检查一组边是否通过共享端点形成连通结构"""
    if len(edge_set) <= 1:
        return True
    start = next(iter(edge_set))
    visited = {start}
    queue = [start]
    while queue:
        cur = queue.pop()
        for nb in edge_adj[cur]:
            if nb in edge_set and nb not in visited:
                visited.add(nb)
                queue.append(nb)
    return len(visited) == len(edge_set)


# ──────────────────────────────────────────────
#  核心: 邻居扩展边分割
# ──────────────────────────────────────────────

def neighbor_expansion_partition(
    graph: nx.Graph,
    k: int,
    weight: str = "weight",
    seed_edges: Optional[List[Tuple]] = None,
    refine_iterations: int = 50,
) -> List[Set[frozenset]]:
    """
    邻居扩展 (NE) 路网边负载均衡分割算法。

    Parameters
    ----------
    graph : nx.Graph
        带边权重的无向路网图。边属性 `weight` 表示负载。
    k : int
        分割子图数量。
    weight : str, default "weight"
        边权重属性名。
    seed_edges : list of (u, v) tuples, optional
        种子边列表 (长度 = k)。为 None 时自动选取。
    refine_iterations : int, default 50
        局部调整最大迭代次数。

    Returns
    -------
    list of set[frozenset]
        长度为 k 的列表，每个元素是该子图的边集合。
        每条边用 frozenset({u, v}) 表示。
    """
    n_edges = graph.number_of_edges()
    if n_edges == 0:
        return [set() for _ in range(k)]

    k = min(k, n_edges)

    # 构建边邻接关系
    edge_adj = _build_edge_adjacency(graph)

    # ── 1. 选取种子边 ──
    if seed_edges is not None:
        seeds = [_edge_id(u, v) for u, v in seed_edges]
    else:
        seeds = _select_seed_edges(graph, edge_adj, k)
    assert len(seeds) == k, f"种子边数 ({len(seeds)}) 必须等于 k ({k})"

    partitions: List[Set[frozenset]] = [set() for _ in range(k)]
    edge_to_part: Dict[frozenset, int] = {}
    assigned: Set[frozenset] = set()

    for i, se in enumerate(seeds):
        partitions[i].add(se)
        edge_to_part[se] = i
        assigned.add(se)

    # ── 2. 多源同步扩展 ──
    total_load = sum(d.get(weight, 1.0) for _, _, d in graph.edges(data=True))
    target_load = total_load / k

    active = [True] * k

    while len(assigned) < n_edges and any(active):
        # 各分区当前负载
        loads = [
            sum(_edge_load(graph, *e, weight) for e in partitions[i])
            for i in range(k)
        ]

        # 负载从小到大，优先扩展
        order = sorted(range(k), key=lambda i: loads[i])

        expanded_this_round = False

        for idx in order:
            if not active[idx]:
                continue

            # 当前分区的所有相邻未分配边
            frontier: Set[frozenset] = set()
            for e in partitions[idx]:
                for nb_e in edge_adj[e]:
                    if nb_e not in assigned:
                        frontier.add(nb_e)

            if not frontier:
                active[idx] = False
                continue

            # 选择使该分区负载最接近目标的边
            current_load = loads[idx]
            best_edge = None
            best_diff = float("inf")

            for candidate in frontier:
                new_load = current_load + _edge_load(graph, *candidate, weight)
                diff = abs(new_load - target_load)
                if diff < best_diff:
                    best_diff = diff
                    best_edge = candidate

            if best_edge is not None:
                partitions[idx].add(best_edge)
                edge_to_part[best_edge] = idx
                assigned.add(best_edge)
                expanded_this_round = True

        if not expanded_this_round:
            break

    # ── 3. 处理未分配的边（不连通的子图等） ──
    for u, v in graph.edges():
        eid = _edge_id(u, v)
        if eid not in assigned:
            # 分配到负载最小的分区
            min_idx = min(
                range(k),
                key=lambda i: sum(_edge_load(graph, *e, weight) for e in partitions[i])
            )
            partitions[min_idx].add(eid)
            edge_to_part[eid] = min_idx
            assigned.add(eid)

    # ── 4. 局部调整 (refinement) ──
    for _ in range(refine_iterations):
        loads = [
            sum(_edge_load(graph, *e, weight) for e in partitions[i])
            for i in range(k)
        ]
        max_idx = max(range(k), key=lambda i: loads[i])
        min_idx = min(range(k), key=lambda i: loads[i])

        if max_idx == min_idx:
            break
        imbalance = loads[max_idx] - loads[min_idx]
        if imbalance < target_load * 0.01:
            break

        improved = False
        # 尝试将重分区边界的一条边移到轻分区
        for edge in list(partitions[max_idx]):
            # 检查该边是否与轻分区相邻（共享端点）
            has_adj_min = any(
                nb in partitions[min_idx] for nb in edge_adj[edge]
            )
            if not has_adj_min:
                continue

            # 移除后连通性检查
            trial = partitions[max_idx] - {edge}
            if not trial:
                continue
            if not _edges_connected(trial, edge_adj):
                continue

            # 加入后连通性检查
            trial_min = partitions[min_idx] | {edge}
            if not _edges_connected(trial_min, edge_adj):
                continue

            # 检查移动是否减少不均衡
            e_load = _edge_load(graph, *edge, weight)
            new_max_load = loads[max_idx] - e_load
            new_min_load = loads[min_idx] + e_load
            if abs(new_max_load - new_min_load) < imbalance:
                partitions[max_idx].remove(edge)
                partitions[min_idx].add(edge)
                edge_to_part[edge] = min_idx
                improved = True
                break

        if not improved:
            break

    return partitions


# ──────────────────────────────────────────────
#  统计信息
# ──────────────────────────────────────────────

def partition_stats(
    graph: nx.Graph,
    partitions: List[Set[frozenset]],
    weight: str = "weight",
) -> Dict:
    """
    返回边分割统计信息。

    Returns
    -------
    dict with keys:
        - loads: 各子图边负载总和
        - edge_counts: 各子图边数
        - mean_load: 平均负载
        - max_imbalance: 不均衡度 (max-min)/mean
        - shared_nodes: 被多个子图共享的节点数
    """
    loads = []
    edge_counts = []
    for part in partitions:
        load = sum(_edge_load(graph, *e, weight) for e in part)
        loads.append(load)
        edge_counts.append(len(part))

    mean_load = sum(loads) / len(loads) if loads else 0
    max_l = max(loads) if loads else 0
    min_l = min(loads) if loads else 0
    imbalance = (max_l - min_l) / mean_load if mean_load > 0 else 0

    # 计算共享节点
    part_nodes = []
    for part in partitions:
        nodes = set()
        for e in part:
            nodes.update(e)
        part_nodes.append(nodes)

    all_nodes = set()
    shared = set()
    for ns in part_nodes:
        shared |= (all_nodes & ns)
        all_nodes |= ns

    return {
        "loads": loads,
        "edge_counts": edge_counts,
        "mean_load": mean_load,
        "max_imbalance": imbalance,
        "shared_nodes": len(shared),
        "partition_nodes": [sorted(ns) for ns in part_nodes],
    }
