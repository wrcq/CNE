# CNE

Road-network edge partitioning toolkit based on neighbor expansion ideas.

## Project Layout

```text
CNE/
  main.py
  scripts/
    demo_grid.py
    demo_random.py
    compare_k.py
  src/cne/
    algorithms/
      cne.py               # Competitive multi-source synchronized expansion
      ne.py                # Single-source sequential expansion
    analysis/
    graph/
    utils/
    viz/
  tests/
```

## Install

```bash
pip install -r requirements.txt
```

## Run

Run all demos:

```bash
python main.py
```

Run a single demo:

```bash
python scripts/demo_grid.py
python scripts/demo_random.py
python scripts/compare_k.py
```

Run external CSV demos:

```bash
python scripts/demo_external_csv.py -k 4 --alpha 0.8 --beta 0.2 --seed-strategy k-medoids --overload-threshold 1.0
python scripts/demo_external_csv_ne.py -k 4
python scripts/demo_external_csv_kmedoids.py -k 4 --max-iter 50
```

Notes:

- `demo_external_csv_kmedoids.py` is a pure edge clustering baseline.
- It clusters by edge-center Euclidean distance only.
- It does not enforce partition connectivity.

## Core API

```python
from cne.algorithms import cne_partition, ne_partition
from cne.analysis import partition_stats
```

Available partition APIs:

- `cne_partition(graph, k, weight="weight", seed_edges=None, refine_iterations=50, alpha=1.0, beta=1.0, overload_threshold=1.2)`
- `cne_partition(graph, k, weight="weight", seed_edges=None, refine_iterations=50, alpha=1.0, beta=1.0, overload_threshold=1.2, seed_strategy="k-medoids")`
- `ne_partition(graph, k, weight="weight", seed_edges=None, refine_iterations=50)`
- `kmedoids_partition(graph, k, max_iter=50)`

Behavior:

1. `cne_partition`: competitive multi-source synchronized expansion.
2. `ne_partition`: single-source sequential expansion (one source grows near target load, then moves to next source).

Competitive CNE cost for assigning candidate edge `e` to subgraph `G_i`:

```math
J(e,G_i) = \alpha \hat{J}_{\mathrm{scale}}(G_i) + \beta \hat{J}_{\mathrm{dist}}(e,G_i)
```

- `\hat{J}_{\mathrm{scale}}`: normalized scale/load term (larger-load partitions are penalized)
- `\hat{J}_{\mathrm{dist}}`: normalized Euclidean distance between edge center and seed-edge center
- `alpha`, `beta`: trade-off between load balancing and spatial compactness

Load-gating operator for suppressing top-heavy growth:

```math
\sigma_i = \frac{S(G_i)}{\bar{S}_t},\quad \bar{S}_t=\frac{1}{K}\sum_{j=1}^{K}S(G_j)
```

If `\sigma_i > overload_threshold` (default `1.2`), partition `G_i` is treated as temporarily overloaded and excluded from candidate competition when non-overloaded alternatives exist. If all adjacent candidates are overloaded, the edge is assigned to the currently lightest adjacent partition.

When an edge is adjacent to multiple subgraphs at the same expansion round, it is assigned to the subgraph with minimum competitive cost.

Demo parameters:

```bash
python scripts/demo_grid.py --alpha 1.0 --beta 1.0 --overload-threshold 1.2
python scripts/demo_random.py --alpha 0.8 --beta 1.2 --overload-threshold 1.2
python scripts/compare_k.py --alpha 1.5 --beta 0.5 --overload-threshold 1.2
```

## Migration Notes

Old module `src/cne/algorithms/neighbor_expansion.py` has been migrated to `src/cne/algorithms/cne.py`.

Compatibility alias is still provided:

- `neighbor_expansion_partition` -> `cne_partition`

Recommended new imports:

```python
from cne.algorithms import cne_partition
from cne.algorithms import ne_partition
```
