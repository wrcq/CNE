"""Compare NE partition balance across different K values."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cne.algorithms import cne_partition
from cne.analysis import partition_stats
from cne.graph import build_grid_road_network


def main():
    parser = argparse.ArgumentParser(description="Compare competitive CNE balance across K")
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
    print("示例 3: 不同 K 值对比 (网格路网)")
    print("=" * 60)
    print(
        f"参数: alpha={args.alpha}, beta={args.beta}, "
        f"lambda={args.overload_threshold}"
    )

    graph = build_grid_road_network(5, 6)
    for k in [2, 3, 4, 5]:
        partitions = cne_partition(
            graph,
            k=k,
            alpha=args.alpha,
            beta=args.beta,
            overload_threshold=args.overload_threshold,
        )
        stats = partition_stats(graph, partitions)
        loads_str = ", ".join(f"{l:.1f}" for l in stats["loads"])
        print(
            f"  K={k}: 负载=[{loads_str}], "
            f"不均衡度={stats['max_imbalance']:.2%}, "
            f"共享节点={stats['shared_nodes']}"
        )


if __name__ == "__main__":
    main()
