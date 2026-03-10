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
      cne.py               # Multi-source synchronized expansion
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

## Core API

```python
from cne.algorithms import cne_partition, ne_partition
from cne.analysis import partition_stats
```

Available partition APIs:

- `cne_partition(graph, k, weight="weight", seed_edges=None, refine_iterations=50)`
- `ne_partition(graph, k, weight="weight", seed_edges=None, refine_iterations=50)`

Behavior:

1. `cne_partition`: multi-source synchronized expansion (lighter partitions expand first).
2. `ne_partition`: single-source sequential expansion (one source grows near target load, then moves to next source).

## Migration Notes

Old module `src/cne/algorithms/neighbor_expansion.py` has been migrated to `src/cne/algorithms/cne.py`.

Compatibility alias is still provided:

- `neighbor_expansion_partition` -> `cne_partition`

Recommended new imports:

```python
from cne.algorithms import cne_partition
from cne.algorithms import ne_partition
```
