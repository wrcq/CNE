"""Preprocess GeoJSON road network for CNE input.

Pipeline:
1) Read LineString road features from GeoJSON.
2) Merge nearby nodes using radius-based clustering in meters.
3) Collapse parallel/opposite carriageways by deduplicating edges on merged node pairs.
4) Export nodes.csv and edges.csv.

The script is dependency-light and uses only Python stdlib.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def lonlat_to_local_xy(lon: float, lat: float, lon0: float, lat0: float) -> Tuple[float, float]:
    """Approximate lon/lat to local metric coordinates (meters)."""
    r = 6371000.0
    x = math.radians(lon - lon0) * r * math.cos(math.radians(lat0))
    y = math.radians(lat - lat0) * r
    return x, y


def local_xy_to_lonlat(x: float, y: float, lon0: float, lat0: float) -> Tuple[float, float]:
    """Inverse transform of lonlat_to_local_xy."""
    r = 6371000.0
    lon = lon0 + math.degrees(x / (r * math.cos(math.radians(lat0))))
    lat = lat0 + math.degrees(y / r)
    return lon, lat


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def cell_key(x: float, y: float, cell_size: float) -> Tuple[int, int]:
    return (math.floor(x / cell_size), math.floor(y / cell_size))


def iter_neighbor_cells(cx: int, cy: int) -> Iterable[Tuple[int, int]]:
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            yield (cx + dx, cy + dy)


def read_linestring_features(geojson_path: Path) -> List[dict]:
    data = json.loads(geojson_path.read_text(encoding="utf-8"))
    features = data.get("features", [])
    out = []
    for feat in features:
        geom = feat.get("geometry") or {}
        if geom.get("type") != "LineString":
            continue
        coords = geom.get("coordinates") or []
        if len(coords) < 2:
            continue
        out.append(feat)
    if not out:
        raise ValueError("No valid LineString features found in input GeoJSON")
    return out


def build_point_table(features: List[dict]) -> Tuple[List[Tuple[float, float]], List[List[int]], float, float]:
    lon_sum = 0.0
    lat_sum = 0.0
    cnt = 0
    for feat in features:
        for lon, lat in feat["geometry"]["coordinates"]:
            lon_sum += float(lon)
            lat_sum += float(lat)
            cnt += 1
    lon0 = lon_sum / cnt
    lat0 = lat_sum / cnt

    points_xy: List[Tuple[float, float]] = []
    line_point_indices: List[List[int]] = []
    for feat in features:
        idxs: List[int] = []
        for lon, lat in feat["geometry"]["coordinates"]:
            x, y = lonlat_to_local_xy(float(lon), float(lat), lon0, lat0)
            idxs.append(len(points_xy))
            points_xy.append((x, y))
        line_point_indices.append(idxs)

    return points_xy, line_point_indices, lon0, lat0


def cluster_points(points_xy: List[Tuple[float, float]], eps: float) -> Dict[int, int]:
    n = len(points_xy)
    uf = UnionFind(n)
    grid: Dict[Tuple[int, int], List[int]] = defaultdict(list)

    for i, (x, y) in enumerate(points_xy):
        cx, cy = cell_key(x, y, eps)
        for nx, ny in iter_neighbor_cells(cx, cy):
            for j in grid.get((nx, ny), []):
                x2, y2 = points_xy[j]
                if (x - x2) * (x - x2) + (y - y2) * (y - y2) <= eps * eps:
                    uf.union(i, j)
        grid[(cx, cy)].append(i)

    root_to_cluster: Dict[int, int] = {}
    point_to_cluster: Dict[int, int] = {}
    next_cluster = 0
    for i in range(n):
        r = uf.find(i)
        if r not in root_to_cluster:
            root_to_cluster[r] = next_cluster
            next_cluster += 1
        point_to_cluster[i] = root_to_cluster[r]

    return point_to_cluster


def build_cluster_centroids(
    points_xy: List[Tuple[float, float]],
    point_to_cluster: Dict[int, int],
    lon0: float,
    lat0: float,
) -> Dict[int, Tuple[float, float]]:
    acc: Dict[int, Tuple[float, float, int]] = {}
    for i, (x, y) in enumerate(points_xy):
        cid = point_to_cluster[i]
        sx, sy, c = acc.get(cid, (0.0, 0.0, 0))
        acc[cid] = (sx + x, sy + y, c + 1)

    out: Dict[int, Tuple[float, float]] = {}
    for cid, (sx, sy, c) in acc.items():
        mx = sx / c
        my = sy / c
        lon, lat = local_xy_to_lonlat(mx, my, lon0, lat0)
        out[cid] = (lon, lat)
    return out


def segment_length_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _angle_deg(v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
    """Return angle in degrees between vectors v1 and v2 in [0, 180]."""
    n1 = math.hypot(v1[0], v1[1])
    n2 = math.hypot(v2[0], v2[1])
    if n1 == 0.0 or n2 == 0.0:
        return 0.0
    c = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    c = max(-1.0, min(1.0, c))
    return math.degrees(math.acos(c))


def build_collapsed_edges(
    points_xy: List[Tuple[float, float]],
    line_point_indices: List[List[int]],
    point_to_cluster: Dict[int, int],
) -> Dict[Tuple[int, int], Tuple[float, int]]:
    edge_acc: Dict[Tuple[int, int], Tuple[float, int]] = {}

    for idxs in line_point_indices:
        for i in range(len(idxs) - 1):
            p1 = idxs[i]
            p2 = idxs[i + 1]
            c1 = point_to_cluster[p1]
            c2 = point_to_cluster[p2]
            if c1 == c2:
                continue

            u, v = (c1, c2) if c1 < c2 else (c2, c1)
            seg_len = segment_length_m(points_xy[p1], points_xy[p2])

            total, count = edge_acc.get((u, v), (0.0, 0))
            edge_acc[(u, v)] = (total + seg_len, count + 1)

    return edge_acc


def simplify_collinear_degree2_nodes(
    node_xy: Dict[int, Tuple[float, float]],
    edge_pairs: List[Tuple[int, int]],
    angle_threshold_deg: float,
) -> List[Tuple[int, int]]:
    """Iteratively merge removable nodes with degree=2 and near-collinear incident edges.

    A node n is removable iff:
    1) deg(n) == 2
    2) abs(180 - angle(nei1-n-nei2)) <= angle_threshold_deg
    """
    edge_set = {tuple(sorted((u, v))) for u, v in edge_pairs if u != v}
    adj: Dict[int, set] = defaultdict(set)
    for u, v in edge_set:
        adj[u].add(v)
        adj[v].add(u)

    changed = True
    while changed:
        changed = False
        for n in list(adj.keys()):
            neigh = list(adj.get(n, set()))
            if len(neigh) != 2:
                continue

            a, b = neigh
            if a == b or a not in node_xy or b not in node_xy or n not in node_xy:
                continue

            pn = node_xy[n]
            pa = node_xy[a]
            pb = node_xy[b]
            v1 = (pa[0] - pn[0], pa[1] - pn[1])
            v2 = (pb[0] - pn[0], pb[1] - pn[1])
            ang = _angle_deg(v1, v2)

            if abs(180.0 - ang) > angle_threshold_deg:
                continue

            # Remove n-a and n-b, then add a-b to keep topology connected.
            e1 = tuple(sorted((n, a)))
            e2 = tuple(sorted((n, b)))
            edge_set.discard(e1)
            edge_set.discard(e2)
            adj[a].discard(n)
            adj[b].discard(n)
            adj[n].clear()

            e3 = tuple(sorted((a, b)))
            edge_set.add(e3)
            adj[a].add(b)
            adj[b].add(a)

            changed = True
            break

    return sorted(edge_set)


def write_nodes_csv(path: Path, cluster_centroids: Dict[int, Tuple[float, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["node_id", "x", "y"])
        for cid in sorted(cluster_centroids):
            lon, lat = cluster_centroids[cid]
            writer.writerow([cid, f"{lon:.8f}", f"{lat:.8f}"])


def write_edges_csv(
    path: Path,
    edge_pairs: List[Tuple[int, int]],
    node_xy: Dict[int, Tuple[float, float]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["u", "v", "weight"])
        for u, v in edge_pairs:
            w = segment_length_m(node_xy[u], node_xy[v])
            writer.writerow([u, v, f"{w:.3f}"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess GeoJSON roads to CNE CSV graph")
    parser.add_argument("--input", required=True, help="Path to input GeoJSON")
    parser.add_argument(
        "--output-dir",
        default="dataset/processed",
        help="Directory for output nodes.csv and edges.csv",
    )
    parser.add_argument(
        "--eps-node",
        type=float,
        default=60.0,
        help="Node merge radius in meters (default 60 for very aggressive consolidation)",
    )
    parser.add_argument(
        "--angle-threshold",
        type=float,
        default=30.0,
        help="Keep-node angle threshold in degrees for collinear simplification (180 +/- threshold)",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    features = read_linestring_features(in_path)
    points_xy, line_point_indices, lon0, lat0 = build_point_table(features)
    point_to_cluster = cluster_points(points_xy, args.eps_node)
    cluster_centroids = build_cluster_centroids(points_xy, point_to_cluster, lon0, lat0)
    edge_acc = build_collapsed_edges(points_xy, line_point_indices, point_to_cluster)

    cluster_xy = {
        cid: lonlat_to_local_xy(lon, lat, lon0, lat0)
        for cid, (lon, lat) in cluster_centroids.items()
    }
    collapsed_edge_pairs = sorted(edge_acc.keys())
    simplified_edge_pairs = simplify_collinear_degree2_nodes(
        cluster_xy,
        collapsed_edge_pairs,
        angle_threshold_deg=args.angle_threshold,
    )

    nodes_csv = out_dir / "nodes.csv"
    edges_csv = out_dir / "edges.csv"
    write_nodes_csv(nodes_csv, cluster_centroids)
    write_edges_csv(edges_csv, simplified_edge_pairs, cluster_xy)

    raw_points = len(points_xy)
    merged_nodes = len(cluster_centroids)
    raw_segments = sum(max(0, len(x) - 1) for x in line_point_indices)
    collapsed_edges = len(collapsed_edge_pairs)
    merged_edges = len(simplified_edge_pairs)

    print("Preprocess done")
    print(f"input: {in_path}")
    print(f"output nodes: {nodes_csv}")
    print(f"output edges: {edges_csv}")
    print(f"raw points: {raw_points} -> merged nodes: {merged_nodes}")
    print(f"raw segments: {raw_segments} -> collapsed edges: {collapsed_edges}")
    print(f"after collinear simplify (deg=2): {merged_edges}")


if __name__ == "__main__":
    main()
