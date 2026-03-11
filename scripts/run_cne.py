"""Run CNE partitioning from external CSV road-network files.

Expected files:
- nodes.csv: node_id,x,y
- edges.csv: u,v,weight (support_count column is optional and ignored)
"""

from __future__ import annotations

import argparse
import csv
import math
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

from cne.algorithms import cne_partition
from cne.algorithms.cne import _build_edge_adjacency, _select_seed_edges_by_strategy
from cne.analysis import partition_stats
from cne.utils.edge import edge_id, edge_load
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


def compute_seed_centers(
    graph: nx.Graph,
    k: int,
    seed_strategy: str,
    seed_max_iter: int,
    weight: str = "weight",
) -> list[tuple[int, int, float, float]]:
    """Compute seed edges used by CNE and return their center coordinates."""
    n_edges = graph.number_of_edges()
    if n_edges == 0:
        return []

    k_eff = min(k, n_edges)
    edge_adj = _build_edge_adjacency(graph)
    pos = {n: tuple(graph.nodes[n]["pos"]) for n in graph.nodes()}
    seed_edges = _select_seed_edges_by_strategy(
        graph,
        edge_adj,
        k_eff,
        weight,
        strategy=seed_strategy,
        positions=pos,
        seed_max_iter=seed_max_iter,
    )

    out: list[tuple[int, int, float, float]] = []
    for e in seed_edges:
        u, v = sorted(e)
        pu = graph.nodes[u].get("pos")
        pv = graph.nodes[v].get("pos")
        if pu is None or pv is None:
            continue
        cx = (float(pu[0]) + float(pv[0])) * 0.5
        cy = (float(pu[1]) + float(pv[1])) * 0.5
        out.append((u, v, cx, cy))
    return out


def _edge_center_from_pos(graph: nx.Graph, edge: frozenset) -> tuple[float, float]:
    u, v = tuple(edge)
    pu = graph.nodes[u]["pos"]
    pv = graph.nodes[v]["pos"]
    return ((float(pu[0]) + float(pv[0])) * 0.5, (float(pu[1]) + float(pv[1])) * 0.5)


def compute_partition_cost_terms(
    graph: nx.Graph,
    partitions: list[set[frozenset]],
    k: int,
    alpha: float,
    beta: float,
    seed_strategy: str,
    seed_max_iter: int,
    weight: str = "weight",
) -> dict:
    """Compute post-hoc CNE-like scale/dist/total cost terms per partition."""
    n_edges = graph.number_of_edges()
    if n_edges == 0:
        return {
            "scale_hat": [],
            "dist_hat": [],
            "total_cost": [],
            "mean_total_cost": 0.0,
        }

    k_eff = min(k, n_edges)
    edge_adj = _build_edge_adjacency(graph)
    pos = {n: tuple(graph.nodes[n]["pos"]) for n in graph.nodes()}
    seed_edges = _select_seed_edges_by_strategy(
        graph,
        edge_adj,
        k_eff,
        weight,
        strategy=seed_strategy,
        positions=pos,
        seed_max_iter=seed_max_iter,
    )
    seed_centers = [_edge_center_from_pos(graph, se) for se in seed_edges]

    # Map each partition to the seed edge it contains; fallback by index when needed.
    seed_by_part: list[frozenset | None] = [None] * len(partitions)
    for i, se in enumerate(seed_edges):
        found = False
        for p_idx, part in enumerate(partitions):
            if se in part:
                seed_by_part[p_idx] = se
                found = True
                break
        if not found and i < len(partitions):
            seed_by_part[i] = se

    loads = [sum(edge_load(graph, *e, weight) for e in part) for part in partitions]
    mean_load = (sum(loads) / len(loads)) if loads else 0.0
    scale_raw = [(ld / mean_load) if mean_load > 0 else 0.0 for ld in loads]
    max_scale = max(scale_raw) if scale_raw else 0.0
    scale_hat = [(x / max_scale) if max_scale > 0 else 0.0 for x in scale_raw]

    mean_dists = []
    for p_idx, part in enumerate(partitions):
        se = seed_by_part[p_idx]
        if se is None:
            mean_dists.append(0.0)
            continue

        seed_center = _edge_center_from_pos(graph, se)
        if not part:
            mean_dists.append(0.0)
            continue

        dvals = []
        for e in part:
            c = _edge_center_from_pos(graph, e)
            dvals.append(math.hypot(c[0] - seed_center[0], c[1] - seed_center[1]))
        mean_dists.append(sum(dvals) / len(dvals))

    max_dist = max(mean_dists) if mean_dists else 0.0
    dist_hat = [(d / max_dist) if max_dist > 0 else 0.0 for d in mean_dists]

    total_cost = [alpha * scale_hat[i] + beta * dist_hat[i] for i in range(len(partitions))]
    mean_total_cost = (sum(total_cost) / len(total_cost)) if total_cost else 0.0

    return {
        "scale_hat": scale_hat,
        "dist_hat": dist_hat,
        "total_cost": total_cost,
        "mean_total_cost": mean_total_cost,
        "seed_centers": seed_centers,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CNE from external CSV road data")
    parser.add_argument("--nodes", default="dataset/processed/nodes.csv", help="Path to nodes.csv")
    parser.add_argument("--edges", default="dataset/processed/edges.csv", help="Path to edges.csv")
    parser.add_argument("-k", type=int, default=3, help="Number of partitions")
    parser.add_argument("--alpha", type=float, default=0.5, help="weight for scale/load term")
    parser.add_argument("--beta", type=float, default=0.5, help="weight for distance term")
    parser.add_argument(
        "--overload-threshold",
        type=float,
        default=1.05,
        help="relative load intensity threshold for gating overloaded partitions",
    )
    parser.add_argument(
        "--show-edge-labels",
        action="store_true",
        help="show edge weight labels on plot",
    )
    parser.add_argument(
        "--seed-strategy",
        choices=["bfs-dispersed", "k-medoids"],
        default="k-medoids",
        help="Seed edge selection strategy",
    )
    parser.add_argument(
        "--seed-max-iter",
        type=int,
        default=50,
        help="Max k-medoids iterations for seed selection",
    )
    parser.add_argument(
        "--refine-iterations",
        type=int,
        default=5,
        help="Number of refinement iterations",
    )
    args = parser.parse_args()

    nodes_csv = ROOT / args.nodes
    edges_csv = ROOT / args.edges

    start = time.time()
    graph, pos = load_graph_from_csv(nodes_csv, edges_csv)
    seed_centers = compute_seed_centers(
        graph,
        args.k,
        seed_strategy=args.seed_strategy,
        seed_max_iter=args.seed_max_iter,
    )
    partitions = cne_partition(
        graph,
        k=args.k,
        alpha=args.alpha,
        beta=args.beta,
        overload_threshold=args.overload_threshold,
        seed_strategy=args.seed_strategy,
        seed_max_iter=args.seed_max_iter,
    )

    stats = partition_stats(graph, partitions)
    cost_terms = compute_partition_cost_terms(
        graph,
        partitions,
        k=args.k,
        alpha=args.alpha,
        beta=args.beta,
        seed_strategy=args.seed_strategy,
        seed_max_iter=args.seed_max_iter,
    )
    
    

    print("=" * 65)
    print("外部 CSV 路网 CNE 分区结果")
    print("=" * 65)
    print(f"节点数: {graph.number_of_nodes()}, 边数: {graph.number_of_edges()}")
    print(
        f"参数: k={args.k}, alpha={args.alpha}, beta={args.beta}, "
        f"lambda={args.overload_threshold}, seed={args.seed_strategy}, seed_iter={args.seed_max_iter}"
    )
    print(f"各子图边数: {stats['edge_counts']}")
    print(f"各子图负载: {[f'{x:.1f}' for x in stats['loads']]}")
    print(f"各子图距离项 J_dist_hat: {[f'{x:.3f}' for x in cost_terms['dist_hat']]}")
    print(f"各子图负载项 J_scale_hat: {[f'{x:.3f}' for x in cost_terms['scale_hat']]}")
    print(f"各子图总代价 J_total: {[f'{x:.3f}' for x in cost_terms['total_cost']]}")
    print(f"平均总代价: {cost_terms['mean_total_cost']:.3f}")
    print(f"不均衡度: {stats['max_imbalance']:.2%}, 共享节点: {stats['shared_nodes']}")
    if seed_centers:
        print("种子点(由 seed edge 中点计算):")
        for i, (u, v, cx, cy) in enumerate(seed_centers):
            print(f"  seed[{i}] edge=({u},{v}) center=({cx:.8f},{cy:.8f})")

    out_img = OUTPUT_DIR / f"external_cne_k{args.k}.png"
    draw_partitioned_graph(
        graph,
        partitions,
        pos=pos,
        seed_centers=[(cx, cy) for _, _, cx, cy in seed_centers],
        title=(
            f"外部路网 CNE 分区 (K={args.k}, a={args.alpha}, "
            f"b={args.beta}, l={args.overload_threshold})"
        ),
        show_edge_labels=args.show_edge_labels,
        save_path=os.path.join(str(OUTPUT_DIR), out_img.name),
    )
    end = time.time()
    print(f"运行时间: {end - start:.2f} 秒")

if __name__ == "__main__":
    main()
