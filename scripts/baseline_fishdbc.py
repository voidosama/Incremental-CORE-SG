#!/usr/bin/env python3
"""FISHDBC baseline, emitted into the same CSV schema as coresg_exp.

FISHDBC (Dell'Amico, arXiv:1910.07283) is the closest published approach: a
flexible, incremental, scalable approximation of HDBSCAN* that keeps an
incremental MST fed by HNSW-derived candidate edges. It is the natural external
comparison, and it is the *opposite* design decision on the same problem --
it maintains an MST, this project avoids maintaining one.

What to compare, and what not to
--------------------------------
FISHDBC is single-threaded Python; this project is multi-threaded C++/CUDA.
Racing them on wall-clock measures the languages, not the methods, and any such
number is indefensible in a thesis. Two things are worth comparing:

  * quality  -- ARI of each method's clustering against the same reference
                (exact HDBSCAN*), which is language-independent;
  * shape    -- how per-batch cost grows as the stream lengthens. The slope is
                the property under test, not the intercept.

Scope difference worth stating explicitly in the write-up: FISHDBC approximates
HDBSCAN* for a *single* mpts. CORE-SG exists to serve a whole range of mpts
from one graph, and this project's contribution is making that incremental. So
FISHDBC is a baseline for the incremental-clustering claim, not for the
multiple-mpts claim, where it has no equivalent.

Install (not done automatically -- it is a third-party package and not on PyPI):

    pip install git+https://github.com/matteodellamico/flexible-clustering

Usage:

    python scripts/baseline_fishdbc.py --data data/manifold_mid.fvecs \\
        --name manifold_mid --batches 8 --limit 20000 --out results.csv
"""

import argparse
import csv
import os
import sys
import time

import numpy as np


def read_fvecs(path, limit=0):
    raw = np.fromfile(path, dtype=np.int32)
    if raw.size == 0:
        raise SystemExit(f"{path} is empty")
    d = int(raw[0])
    stride = d + 1
    n = raw.size // stride
    if limit:
        n = min(n, limit)
    return raw[: n * stride].reshape(n, stride)[:, 1:].view(np.float32)


class CsvSink:
    """Same long-format schema coresg_exp writes, so results concatenate."""

    HEADER = ["experiment", "dataset", "n", "dim", "k", "config", "value",
              "seed", "batch", "metric", "measurement"]

    def __init__(self, path):
        exists = os.path.exists(path) and os.path.getsize(path) > 0
        self.f = open(path, "a", newline="")
        self.w = csv.writer(self.f)
        if not exists:
            self.w.writerow(self.HEADER)

    def row(self, experiment, dataset, n, dim, k, config, value, seed, batch,
            metric, measurement):
        self.w.writerow([experiment, dataset, n, dim, k, config, value, seed,
                         batch, metric, measurement])
        self.f.flush()


def load_fishdbc():
    try:
        from flexible_clustering import FISHDBC
        return FISHDBC
    except ImportError:
        raise SystemExit(
            "FISHDBC is not installed. It is not on PyPI; install from source:\n"
            "    pip install git+https://github.com/matteodellamico/flexible-clustering")


def reference_labels(x, min_cluster_size):
    """Exact HDBSCAN* labels, the common yardstick both methods are scored on."""
    try:
        from sklearn.cluster import HDBSCAN
        return HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(x)
    except ImportError:
        pass
    try:
        import hdbscan
        return hdbscan.HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(x)
    except ImportError:
        raise SystemExit("need scikit-learn >= 1.3 (HDBSCAN) or the hdbscan package "
                         "for the reference clustering")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--limit", type=int, default=20000,
                    help="cap rows; FISHDBC is single-threaded Python, keep this modest")
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--base-fraction", type=float, default=0.5)
    ap.add_argument("--min-cluster-size", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="results.csv")
    args = ap.parse_args()

    FISHDBC = load_fishdbc()
    from sklearn.metrics import adjusted_rand_score

    x = read_fvecs(args.data, args.limit)
    n, dim = x.shape
    name = args.name or os.path.basename(args.data)
    print(f"{name}: {n} x {dim}")

    sink = CsvSink(args.out)
    base = int(args.base_fraction * n)
    per_batch = max(1, (n - base) // max(1, args.batches))

    def euclidean(a, b):
        return float(np.linalg.norm(a - b))

    for s in range(args.seeds):
        seed = 9000 + s
        rng = np.random.default_rng(seed)
        order = rng.permutation(n)
        xs = x[order]

        clusterer = FISHDBC(euclidean)

        t0 = time.perf_counter()
        for row in xs[:base]:
            clusterer.add(row)
        build_s = time.perf_counter() - t0
        sink.row("baseline_fishdbc", name, base, dim, args.min_cluster_size,
                 "method", "fishdbc", seed, 0, "build_seconds", build_s)
        print(f"  seed {seed}: base of {base} in {build_s:.2f}s")

        cursor = base
        cumulative = build_s
        for b in range(args.batches):
            if cursor >= n:
                break
            count = min(per_batch, n - cursor)
            t0 = time.perf_counter()
            for row in xs[cursor:cursor + count]:
                clusterer.add(row)
            insert_s = time.perf_counter() - t0
            cursor += count
            cumulative += insert_s

            batch = b + 1
            sink.row("baseline_fishdbc", name, cursor, dim, args.min_cluster_size,
                     "method", "fishdbc", seed, batch, "insert_seconds", insert_s)
            sink.row("baseline_fishdbc", name, cursor, dim, args.min_cluster_size,
                     "method", "fishdbc", seed, batch, "cumulative_seconds", cumulative)
            sink.row("baseline_fishdbc", name, cursor, dim, args.min_cluster_size,
                     "method", "fishdbc", seed, batch, "throughput_points_per_sec",
                     count / max(insert_s, 1e-9))

            t0 = time.perf_counter()
            labels = clusterer.cluster(min_cluster_size=args.min_cluster_size)[0]
            cluster_s = time.perf_counter() - t0
            labels = np.asarray(labels)[:cursor]

            # The reference has to be recomputed on the points that exist *now*.
            # Slicing a clustering derived from the full file would score
            # FISHDBC against structure it has not been shown yet, which shows
            # up as an artificially low early ARI that climbs as the stream
            # catches up -- handicapping the baseline through a measurement
            # artifact rather than on its merits.
            ref_now = reference_labels(xs[:cursor], args.min_cluster_size)
            ari = adjusted_rand_score(ref_now, labels)
            sink.row("baseline_fishdbc", name, cursor, dim, args.min_cluster_size,
                     "method", "fishdbc", seed, batch, "ari_vs_exact_hdbscan", ari)
            sink.row("baseline_fishdbc", name, cursor, dim, args.min_cluster_size,
                     "method", "fishdbc", seed, batch, "cluster_extract_seconds", cluster_s)
            n_ref = len(set(ref_now.tolist())) - (1 if -1 in ref_now else 0)
            sink.row("baseline_fishdbc", name, cursor, dim, args.min_cluster_size,
                     "method", "fishdbc", seed, batch, "reference_clusters", float(n_ref))
            print(f"    batch {batch}: +{count} in {insert_s:.2f}s, ARI {ari:.4f} "
                  f"(reference has {n_ref} clusters at this point)")

    print(f"\nwrote {args.out}")
    print("To compare: score this project's clustering against the SAME reference\n"
          "(exact HDBSCAN* on the same points) and join on dataset + batch.\n"
          "Compare ARI and the per-batch cost *slope*, not raw wall-clock -- see the\n"
          "module docstring.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
