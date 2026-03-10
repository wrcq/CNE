# CNE Project

Edge-based road-network partitioning toolkit with a modular project layout.

## Project Structure

```text
CNE_project/
  algorithms/
    ne.py                  # Multi-source synchronized NE algorithm
    cne.py                 # Placeholder for future constrained variants
  utils/
    graph_utils.py         # Edge ID/load/adjacency/connectivity helpers
    metrics.py             # Partition statistics
  visualization/
    plot_partition.py      # Partition plotting utilities
  experiments/
    run_ne_demo.py         # Demo entrypoint
  data/                    # Reserved for datasets
```

Legacy entrypoints are kept for compatibility:

- `ne_algorithm.py` -> re-exports from new modules
- `visualize.py` -> re-exports `draw_partitioned_graph`
- `demo.py` -> calls new demo entrypoint

## Install

```bash
pip install -r requirements.txt
```

## Run Demo

Recommended:

```bash
python -m CNE_project.experiments.run_demo
```

Switch algorithm by argument:

```bash
python -m CNE_project.experiments.run_cne_demo --algo cne
python -m CNE_project.experiments.run_cne_demo --algo ne
python -m CNE_project.experiments.run_demo --algo cne
python -m CNE_project.experiments.run_demo --algo ne
```

Compatibility (still works):

```bash
python demo.py
```

Outputs are saved to:

- `output/result_grid_<algo>.png`
- `output/result_random_<algo>.png`

## Core API

```python
from CNE_project.algorithms.ne import neighbor_expansion_partition
from CNE_project.utils.metrics import partition_stats
```

Main function:

- `neighbor_expansion_partition(graph, k, weight="weight", seed_edges=None, refine_iterations=50)`

Algorithm behavior:

1. Select edge seeds.
2. Perform multi-source synchronized expansion (low-load partition expands first).
3. Refine by moving boundary edges while preserving connectivity.

## Add New Algorithms

1. Add a new file in `CNE_project/algorithms/` (for example `my_algo.py`).
2. Reuse helpers from `CNE_project/utils/graph_utils.py`.
3. Add evaluation logic using `CNE_project/utils/metrics.py`.
4. Add an experiment script in `CNE_project/experiments/`.

`cne.py` already exists as an extension point for constrained NE variants.
