"""
路网边分割可视化模块
====================
将 NE 边分割结果绘制为彩色路网图，不同子图的边用不同颜色显示。
节点若被多个子图共享，用分饼图标记。
"""

from __future__ import annotations

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
import numpy as np
from typing import List, Set, Dict, Optional
from collections import defaultdict

# 设置中文字体
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


# 预定义一组对比度较高的颜色
_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9A6324", "#800000", "#aaffc3", "#808000",
    "#000075", "#a9a9a9",
]


def _edge_id(u, v) -> frozenset:
    return frozenset((u, v))


def draw_partitioned_graph(
    graph: nx.Graph,
    partitions: List[Set[frozenset]],
    weight: str = "weight",
    pos: Optional[Dict] = None,
    title: str = "路网边负载均衡分割结果 (NE 算法)",
    figsize: tuple = (14, 10),
    node_size: int = 250,
    edge_width_scale: float = 2.0,
    show_edge_labels: bool = True,
    save_path: Optional[str] = None,
):
    """
    绘制边分割后的路网图。

    不同子图的边用不同颜色绘制。
    共享节点（属于多个子图）用多色环标记。

    Parameters
    ----------
    graph : nx.Graph
    partitions : list of set[frozenset]
        边分割结果。
    weight : str
    pos : dict, optional
    title : str
    figsize, node_size, edge_width_scale, show_edge_labels, save_path
    """
    k = len(partitions)
    colors = _COLORS[:k] if k <= len(_COLORS) else \
        [plt.cm.tab20(i / k) for i in range(k)]

    # 边 -> 分区映射
    edge_to_part: Dict[frozenset, int] = {}
    for i, part in enumerate(partitions):
        for e in part:
            edge_to_part[e] = i

    # 节点 -> 所属分区集合
    node_parts: Dict[int, Set[int]] = defaultdict(set)
    for i, part in enumerate(partitions):
        for e in part:
            for n in e:
                node_parts[n].add(i)

    # 布局
    if pos is None:
        pos = nx.spring_layout(graph, seed=42,
                               k=1.5 / (graph.number_of_nodes() ** 0.5))

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_title(title, fontsize=16, fontweight="bold")

    # ── 绘制边 ──
    for u, v, data in graph.edges(data=True):
        w = data.get(weight, 1.0)
        eid = _edge_id(u, v)
        part_idx = edge_to_part.get(eid, -1)

        x = [pos[u][0], pos[v][0]]
        y = [pos[u][1], pos[v][1]]

        if part_idx >= 0:
            color = colors[part_idx]
            ax.plot(x, y, color=color, linewidth=max(1.5, w * edge_width_scale),
                    alpha=0.8, solid_capstyle="round", zorder=1)
        else:
            ax.plot(x, y, color="#aaaaaa", linewidth=1,
                    alpha=0.3, linestyle=":", zorder=1)

        # 边权重标签
        if show_edge_labels:
            mid_x = (pos[u][0] + pos[v][0]) / 2
            mid_y = (pos[u][1] + pos[v][1]) / 2
            ax.text(mid_x, mid_y, f"{w:.1f}", fontsize=7,
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                              edgecolor="none", alpha=0.75), zorder=4)

    # ── 绘制节点 ──
    for node in graph.nodes():
        x, y = pos[node]
        parts = sorted(node_parts.get(node, set()))

        if len(parts) == 0:
            ax.scatter(x, y, s=node_size, c="#cccccc", edgecolors="white",
                       linewidths=1.5, zorder=3)
        elif len(parts) == 1:
            ax.scatter(x, y, s=node_size, c=colors[parts[0]],
                       edgecolors="white", linewidths=1.5, zorder=3)
        else:
            # 共享节点：画多色分饼
            n_parts = len(parts)
            theta_start = 90  # 从顶部开始
            for idx, p in enumerate(parts):
                wedge = mpatches.Wedge(
                    (x, y), _node_radius(ax, node_size, figsize),
                    theta_start + idx * 360 / n_parts,
                    theta_start + (idx + 1) * 360 / n_parts,
                    facecolor=colors[p], edgecolor="white", linewidth=1.5,
                    zorder=3, transform=ax.transData,
                )
                ax.add_patch(wedge)

    # 节点标签
    nx.draw_networkx_labels(graph, pos, font_size=8, font_color="white",
                            font_weight="bold", ax=ax)

    # ── 图例 ──
    from ne_algorithm import _edge_load
    legend_handles = []
    for i, part in enumerate(partitions):
        load = sum(_edge_load(graph, *e, weight) for e in part)
        nodes_in_part = set()
        for e in part:
            nodes_in_part.update(e)
        label = f"子图 {i}  |  {len(part)} 条边  |  负载 {load:.1f}"
        patch = mpatches.Patch(color=colors[i], label=label)
        legend_handles.append(patch)

    # 共享节点图例
    shared_count = sum(1 for n in graph.nodes() if len(node_parts.get(n, set())) > 1)
    if shared_count > 0:
        legend_handles.append(
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor="#888888", markeredgecolor="black",
                       markersize=8,
                       label=f"共享节点: {shared_count} 个")
        )

    ax.legend(handles=legend_handles, loc="upper left", fontsize=9,
              framealpha=0.9, edgecolor="#cccccc")

    ax.axis("off")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"图片已保存: {save_path}")
    else:
        plt.show()

    plt.close(fig)


def _node_radius(ax, node_size, figsize):
    """估算节点在数据坐标中的半径（用于绘制分饼楔形）"""
    # 粗略从 scatter size (点^2 单位) 转到数据坐标
    xlim = ax.get_xlim()
    data_range = xlim[1] - xlim[0] if xlim[1] != xlim[0] else 1.0
    fig_dpi = 100
    fig_width_px = figsize[0] * fig_dpi
    pts_per_data = fig_width_px / data_range
    radius_pts = (node_size ** 0.5) / 2
    return radius_pts / pts_per_data
