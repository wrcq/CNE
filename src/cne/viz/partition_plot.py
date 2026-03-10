"""Visualization utilities for NE edge partition results."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx

from cne.utils.edge import edge_id, edge_load

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

_COLORS = [
    "#e6194b",
    "#3cb44b",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#42d4f4",
    "#f032e6",
    "#bfef45",
    "#fabed4",
    "#469990",
    "#dcbeff",
    "#9A6324",
    "#800000",
    "#aaffc3",
    "#808000",
    "#000075",
    "#a9a9a9",
]


def draw_partitioned_graph(
    graph: nx.Graph,
    partitions: List[Set[frozenset]],
    weight: str = "weight",
    pos: Optional[Dict] = None,
    title: str = "路网边负载均衡分割结果 (NE 算法)",
    figsize: tuple = (14, 10),
    node_size: int = 250,
    edge_width_scale: float = 2.0,
    show_edge_labels: bool = False,
    show_node_labels: bool = False,
    show_nodes: bool = False,
    seed_centers: Optional[List[Tuple[float, float]]] = None,
    save_path: Optional[str] = None,
):
    """Draw partitioned graph where each partition has a distinct edge color."""
    k = len(partitions)
    colors = _COLORS[:k] if k <= len(_COLORS) else [plt.cm.tab20(i / k) for i in range(k)]

    edge_to_part: Dict[frozenset, int] = {}
    for i, part in enumerate(partitions):
        for e in part:
            edge_to_part[e] = i

    node_parts: Dict[int, Set[int]] = defaultdict(set)
    for i, part in enumerate(partitions):
        for e in part:
            for node in e:
                node_parts[node].add(i)

    if pos is None:
        pos = nx.spring_layout(graph, seed=42, k=1.5 / (graph.number_of_nodes() ** 0.5))

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_title(title, fontsize=16, fontweight="bold")

    for u, v, data in graph.edges(data=True):
        w = data.get(weight, 1.0)
        eid = edge_id(u, v)
        part_idx = edge_to_part.get(eid, -1)

        x = [pos[u][0], pos[v][0]]
        y = [pos[u][1], pos[v][1]]

        if part_idx >= 0:
            color = colors[part_idx]
            ax.plot(x, y, color=color, linewidth=max(1.5, edge_width_scale), alpha=0.8, solid_capstyle="round", zorder=1)
        else:
            ax.plot(x, y, color="#aaaaaa", linewidth=1, alpha=0.3, linestyle=":", zorder=1)

        if show_edge_labels:
            mid_x = (pos[u][0] + pos[v][0]) / 2
            mid_y = (pos[u][1] + pos[v][1]) / 2
            ax.text(
                mid_x,
                mid_y,
                f"{w:.1f}",
                fontsize=7,
                ha="center",
                va="center",
                bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.75},
                zorder=4,
            )

    if show_nodes:
        for node in graph.nodes():
            x, y = pos[node]
            parts = sorted(node_parts.get(node, set()))

            if len(parts) == 0:
                ax.scatter(x, y, s=node_size, c="#cccccc", edgecolors="white", linewidths=1.5, zorder=3)
            elif len(parts) == 1:
                ax.scatter(x, y, s=node_size, c=colors[parts[0]], edgecolors="white", linewidths=1.5, zorder=3)
            else:
                n_parts = len(parts)
                theta_start = 90
                for idx, p in enumerate(parts):
                    wedge = mpatches.Wedge(
                        (x, y),
                        _node_radius(ax, node_size, figsize),
                        theta_start + idx * 360 / n_parts,
                        theta_start + (idx + 1) * 360 / n_parts,
                        facecolor=colors[p],
                        edgecolor="white",
                        linewidth=1.5,
                        zorder=3,
                        transform=ax.transData,
                    )
                    ax.add_patch(wedge)

    if show_node_labels:
        nx.draw_networkx_labels(graph, pos, font_size=8, font_color="white", font_weight="bold", ax=ax)

    if seed_centers:
        sx = [p[0] for p in seed_centers]
        sy = [p[1] for p in seed_centers]
        ax.scatter(
            sx,
            sy,
            s=30,
            marker="s",
            c="black",
            edgecolors="white",
            linewidths=0.6,
            zorder=6,
            label="种子边中心",
        )

    legend_handles = []
    for i, _ in enumerate(partitions):
        legend_handles.append(mpatches.Patch(color=colors[i], label=f"子图 {i}"))
    if seed_centers:
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                markerfacecolor="black",
                markeredgecolor="white",
                markeredgewidth=0.6,
                markersize=6,
                linestyle="None",
                label="种子边中心",
            )
        )

    ax.legend(handles=legend_handles, loc="upper left", fontsize=9, framealpha=0.9, edgecolor="#cccccc")
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"图片已保存: {save_path}")
    else:
        plt.show()

    plt.close(fig)


def _node_radius(ax, node_size, figsize):
    """Estimate node radius in data coordinates for pie-like shared-node wedges."""
    xlim = ax.get_xlim()
    data_range = xlim[1] - xlim[0] if xlim[1] != xlim[0] else 1.0
    fig_dpi = 100
    fig_width_px = figsize[0] * fig_dpi
    pts_per_data = fig_width_px / data_range
    radius_pts = (node_size ** 0.5) / 2
    return radius_pts / pts_per_data
