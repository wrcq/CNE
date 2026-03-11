"""Run METIS baseline from external CSV road-network files.

Baseline design:
- Treat each original edge as a node (represented by edge center).
- Build a kNN proximity graph on edge centers.
- Run METIS node partition and map labels back to edge partitions.

This method does NOT enforce original-graph connectivity constraints.
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

from cne.algorithms import metis_edge_node_partition
from cne.analysis import partition_stats
from cne.viz import draw_partitioned_graph


def ensure_metis_dll() -> None:
    """Best-effort METIS DLL discovery for Windows local runs."""
    existing = os.environ.get("METIS_DLL")
    if existing and Path(existing).is_file():
        return

    candidates = [
        ROOT / ".venv" / "Lib" / "site-packages" / "metis.dll",
        ROOT / "metis.dll",
    ]
    for dll in candidates:
        if dll.is_file():
            os.environ["METIS_DLL"] = str(dll)
            return


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


def representative_edges(
    graph: nx.Graph,
    partitions: list[set[frozenset]],
) -> list[frozenset]:
    """Choose one actual representative edge per partition by intra-part center-distance."""
    def edge_center(e: frozenset) -> tuple[float, float]:
        u, v = tuple(e)
        pu = graph.nodes[u]["pos"]
        pv = graph.nodes[v]["pos"]
        return ((float(pu[0]) + float(pv[0])) * 0.5, (float(pu[1]) + float(pv[1])) * 0.5)

    out: list[frozenset] = []
    for part in partitions:
        if not part:
            continue
        centers = {e: edge_center(e) for e in part}
        best = min(
            part,
            key=lambda e: sum(
                ((centers[e][0] - centers[o][0]) ** 2 + (centers[e][1] - centers[o][1]) ** 2) ** 0.5
                for o in part
            ),
        )
        out.append(best)
    return out


def centers_from_edges(
    graph: nx.Graph,
    edges: list[frozenset],
) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for e in edges:
        u, v = tuple(e)
        pu = graph.nodes[u]["pos"]
        pv = graph.nodes[v]["pos"]
        out.append(((float(pu[0]) + float(pv[0])) * 0.5, (float(pu[1]) + float(pv[1])) * 0.5))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run METIS edge-as-node baseline from external CSV road data")
    parser.add_argument("--nodes", default="dataset/processed/nodes.csv", help="Path to nodes.csv")
    parser.add_argument("--edges", default="dataset/processed/edges.csv", help="Path to edges.csv")
    parser.add_argument("-k", type=int, default=3, help="Number of partitions")
    parser.add_argument("--knn", type=int, default=8, help="k-NN size for edge-center proximity graph")
    parser.add_argument(
        "--show-edge-labels",
        action="store_true",
        help="show edge weight labels on plot",
    )
    args = parser.parse_args()

    ensure_metis_dll()

    nodes_csv = ROOT / args.nodes
    edges_csv = ROOT / args.edges

    start = time.time()
    graph, pos = load_graph_from_csv(nodes_csv, edges_csv)

    try:
        partitions = metis_edge_node_partition(
            graph,
            k=args.k,
            knn=args.knn,
        )
    except RuntimeError as exc:
        print("METIS 运行失败:")
        print(exc)
        print("Windows 提示: 需要安装 METIS 原生库，并设置环境变量 METIS_DLL 指向 metis.dll。")
        return

    stats = partition_stats(graph, partitions)
    reps = representative_edges(graph, partitions)
    centers = centers_from_edges(graph, reps)

    print("=" * 65)
    print("外部 CSV 路网 METIS 基准结果 (edge-as-node)")
    print("=" * 65)
    print(f"节点数: {graph.number_of_nodes()}, 边数: {graph.number_of_edges()}")
    print(f"参数: k={args.k}, knn={args.knn}")
    print(f"各子图边数: {stats['edge_counts']}")
    print(f"各子图负载: {[f'{x:.1f}' for x in stats['loads']]}")
    print(f"不均衡度: {stats['max_imbalance']:.2%}, 共享节点: {stats['shared_nodes']}")
    print(f"各子图紧凑度: {[f'{x:.4f}' for x in stats['compactness_per_partition']]}")
    print(f"平均紧凑度: {stats['compactness_mean']:.4f} (method={stats['compactness_method']})")
    if reps:
        print("代表边(每个分区一条实际边):")
        for i, e in enumerate(reps):
            u, v = sorted(tuple(e))
            print(f"  rep[{i}] edge=({u},{v})")

    out_img = OUTPUT_DIR / f"external_metis_k{args.k}.png"
    draw_partitioned_graph(
        graph,
        partitions,
        pos=pos,
        seed_centers=centers,
        title=f"Case1: METIS",
        show_edge_labels=args.show_edge_labels,
        save_path=os.path.join(str(OUTPUT_DIR), out_img.name),
    )
    end = time.time()
    print(f"运行时间: {end - start:.2f} 秒")

if __name__ == "__main__":
    main()
