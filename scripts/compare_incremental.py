#!/usr/bin/env python3
"""CORE-SG against FISHDBC, on one yardstick.

Why this script exists
----------------------
`baseline_fishdbc.py` scores FISHDBC's HDBSCAN* clustering against sklearn's
exact HDBSCAN*. Every experiment in `coresg_exp` scores a DBSCAN* clustering
(a flat cut of the MST at one threshold) against the exact CORE-SG. Those are
two different clusterings measured against two different references, so the two
ARI columns could not be put in the same table however similar they looked --
and the external baseline is the most visible gap in the evaluation.

The fix is to compare the one artefact both methods actually produce: the MST
in mutual reachability space. This script

  1. fixes an arrival order and hands the *same* order to both methods, so at
     every batch they have been shown exactly the same points;
  2. runs `coresg_exp --experiment stream`, which exports the CORE-SG MST after
     each batch;
  3. turns that MST into HDBSCAN* labels using `hdbscan.hdbscan_`'s
     label -> condense_tree -> compute_stability -> get_clusters, which is
     literally the code FISHDBC's own `cluster()` calls on its MST;
  4. runs FISHDBC over the same stream;
  5. scores both against one reference -- exact HDBSCAN* recomputed on the
     points present at that batch.

The only thing that differs between the two rows is therefore the MST each
method produced, which is the thing under test.

Two traps this script is written to avoid
-----------------------------------------
*The reference must be recomputed per batch.* Computing it once on the whole
file and slicing scores a method against structure it has not been shown, which
depresses early ARI and then climbs as the stream catches up. That is a
measurement artifact, and inflicting it on a competitor is worse than inflicting
it on yourself.

*mpts must match on all three sides.* FISHDBC's constructor defaults to
`min_samples=5` while sklearn's HDBSCAN defaults `min_samples` to
`min_cluster_size`. Left alone, the baseline runs FISHDBC at mpts=5 and scores
it against a reference at mpts=10 -- a handicap that has nothing to do with
FISHDBC. Every side here is pinned to `--mpts` explicitly.

*Do not race the wall-clocks.* FISHDBC is single-threaded Python and this
project is multi-threaded C++/CUDA; a timing comparison measures the languages.
Both sides' per-batch seconds are emitted so the cost *slope* can be compared,
which is language-independent. The intercept is not a result.

Usage:

    python scripts/compare_incremental.py --data data/manifold_mid.fvecs \\
        --name manifold_mid --limit 20000 --batches 8 --mpts 10 \\
        --out results.csv
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def find_binary(name):
    for candidate in (
        os.path.join(ROOT, "build", "bin", "Release", name + ".exe"),
        os.path.join(ROOT, "build", "bin", name),
        os.path.join(ROOT, "build", "bin", "Release", name),
    ):
        if os.path.exists(candidate):
            return candidate
    found = shutil.which(name)
    if found:
        return found
    raise SystemExit(f"cannot find {name}; build the project first")


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
    """The long-format schema coresg_exp writes, so everything concatenates."""

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


# --------------------------------------------------------------------------
# The shared extraction path
# --------------------------------------------------------------------------

def hdbscan_labels_from_mst(edges, n, min_cluster_size):
    """HDBSCAN* labels from an MST, via the exact path FISHDBC's cluster() uses.

    `edges` is an (m, 3) float64 array of [u, v, weight] in mutual reachability
    space. hdbscan's `label` expects the edges sorted by weight -- FISHDBC gets
    that for free because it keeps `_mst_edges` sorted, so it never sorts
    explicitly and neither does the reference implementation. An MST arriving
    from elsewhere has to be sorted here or the single-linkage tree is wrong.
    """
    from hdbscan import hdbscan_

    if edges.shape[0] != n - 1:
        # A forest, not a tree: the CORE-SG's connectivity repair is what
        # normally prevents this. Extraction on a forest silently produces a
        # malformed dendrogram, so refuse rather than emit a number.
        raise SystemExit(
            f"MST has {edges.shape[0]} edges for {n} points; expected {n - 1}. "
            "The graph is disconnected -- check that ensure_connected is on.")
    ordered = edges[np.argsort(edges[:, 2]), :]
    slt = hdbscan_.label(ordered)
    condensed = hdbscan_.condense_tree(slt, min_cluster_size)
    stability = hdbscan_.compute_stability(condensed)
    labels = hdbscan_.get_clusters(condensed, stability, "eom", False, False)[0]
    return np.asarray(labels)


def read_mst(path):
    """(n, edges) from the binary written by exp_stream_mst."""
    with open(path, "rb") as f:
        header = np.fromfile(f, dtype=np.uint64, count=2)
        n, m = int(header[0]), int(header[1])
        rec = np.dtype([("u", np.uint32), ("v", np.uint32), ("w", np.float32)])
        raw = np.fromfile(f, dtype=rec, count=m)
    edges = np.empty((m, 3), dtype=np.float64)
    edges[:, 0] = raw["u"]
    edges[:, 1] = raw["v"]
    edges[:, 2] = raw["w"]
    return n, edges


def reference_labels(x, mpts):
    """Exact HDBSCAN*, the common yardstick. `min_samples` is pinned to mpts so
    all three sides use the same density definition."""
    try:
        from sklearn.cluster import HDBSCAN
        return HDBSCAN(min_cluster_size=mpts, min_samples=mpts).fit_predict(x)
    except ImportError:
        pass
    try:
        import hdbscan
        return hdbscan.HDBSCAN(min_cluster_size=mpts,
                               min_samples=mpts).fit_predict(x)
    except ImportError:
        raise SystemExit("need scikit-learn >= 1.3 (HDBSCAN) or the hdbscan "
                         "package for the reference clustering")


# --------------------------------------------------------------------------

def run_coresg(args, exe, xs, order, seed, work, sink, dim, ref_cache):
    from sklearn.metrics import adjusted_rand_score

    n = len(order)
    order_path = os.path.join(work, f"order_{seed}.i32")
    order.astype(np.int32).tofile(order_path)
    mst_dir = os.path.join(work, f"mst_{seed}")
    os.makedirs(mst_dir, exist_ok=True)

    cmd = [exe, "--experiment", "stream",
           "--data", args.data, "--name", args.name,
           "--k", str(args.k), "--metric", args.metric,
           "--batches", str(args.batches),
           "--base-fraction", str(args.base_fraction),
           "--export-mpts", str(args.mpts),
           "--order-file", order_path, "--mst-dir", mst_dir,
           "--seed-base", str(seed), "--seeds", "1",
           "--out", args.out, "--quiet"]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    if not args.gpu:
        cmd += ["--gpu", "0"]
    print("  coresg:", " ".join(cmd[1:]), flush=True)
    subprocess.run(cmd, check=True)

    base = int(args.base_fraction * n)
    per_batch = (n - base) // max(1, args.batches)
    cursor = base
    for b in range(1, args.batches + 1):
        if cursor >= n:
            break
        cursor = min(cursor + per_batch, n)
        path = os.path.join(mst_dir, f"{args.name}_seed{seed}_batch{b}.mst")
        if not os.path.exists(path):
            print(f"    batch {b}: no MST written, stopping")
            break
        mst_n, edges = read_mst(path)
        labels = hdbscan_labels_from_mst(edges, mst_n, args.mpts)

        ref = ref_cache.get(cursor)
        if ref is None:
            ref = reference_labels(xs[:cursor], args.mpts)
            ref_cache[cursor] = ref
        ari = adjusted_rand_score(ref, labels[:cursor])
        n_ref = len(set(ref.tolist())) - (1 if -1 in ref else 0)
        n_found = len(set(labels.tolist())) - (1 if -1 in labels else 0)
        sink.row("stream_coresg", args.name, cursor, dim, args.mpts, "method",
                 "coresg", seed, b, "ari_vs_exact_hdbscan", ari)
        sink.row("stream_coresg", args.name, cursor, dim, args.mpts, "method",
                 "coresg", seed, b, "reference_clusters", float(n_ref))
        sink.row("stream_coresg", args.name, cursor, dim, args.mpts, "method",
                 "coresg", seed, b, "clusters_found", float(n_found))
        print(f"    batch {b}: n={cursor} ARI {ari:.4f} "
              f"({n_found} clusters vs reference {n_ref})", flush=True)


def run_fishdbc(args, xs, seed, sink, dim, ref_cache):
    try:
        from flexible_clustering import FISHDBC
    except ImportError:
        raise SystemExit(
            "FISHDBC is not installed. It is not on PyPI, and its setup.py "
            "imports Cython without declaring it, so pip's isolated build "
            "fails:\n"
            "    conda activate coresg\n"
            "    pip install --no-build-isolation "
            "git+https://github.com/matteodellamico/flexible-clustering")
    from sklearn.metrics import adjusted_rand_score

    n = len(xs)
    base = int(args.base_fraction * n)
    per_batch = (n - base) // max(1, args.batches)

    def euclidean(a, b):
        return float(np.linalg.norm(a - b))

    # min_samples pinned to mpts: the constructor defaults to 5, which would
    # score FISHDBC's mpts=5 clustering against an mpts=10 reference.
    clusterer = FISHDBC(euclidean, min_samples=args.mpts)

    t0 = time.perf_counter()
    for row in xs[:base]:
        clusterer.add(row)
    build_s = time.perf_counter() - t0
    sink.row("stream_fishdbc", args.name, base, dim, args.mpts, "method",
             "fishdbc", seed, 0, "build_seconds", build_s)
    print(f"    fishdbc base of {base} in {build_s:.2f}s", flush=True)

    cursor = base
    cumulative = build_s
    for b in range(1, args.batches + 1):
        if cursor >= n:
            break
        count = min(per_batch, n - cursor)
        t0 = time.perf_counter()
        for row in xs[cursor:cursor + count]:
            clusterer.add(row)
        insert_s = time.perf_counter() - t0
        cursor += count
        cumulative += insert_s

        t0 = time.perf_counter()
        labels = np.asarray(clusterer.cluster(min_cluster_size=args.mpts)[0])
        cluster_s = time.perf_counter() - t0

        ref = ref_cache.get(cursor)
        if ref is None:
            ref = reference_labels(xs[:cursor], args.mpts)
            ref_cache[cursor] = ref
        ari = adjusted_rand_score(ref, labels[:cursor])
        n_found = len(set(labels.tolist())) - (1 if -1 in labels else 0)

        for metric, value in (("insert_seconds", insert_s),
                              ("cumulative_seconds", cumulative),
                              ("throughput_points_per_sec",
                               count / max(insert_s, 1e-9)),
                              ("cluster_extract_seconds", cluster_s),
                              ("ari_vs_exact_hdbscan", ari),
                              ("clusters_found", float(n_found))):
            sink.row("stream_fishdbc", args.name, cursor, dim, args.mpts,
                     "method", "fishdbc", seed, b, metric, value)
        print(f"    batch {b}: n={cursor} ARI {ari:.4f} "
              f"({n_found} clusters), +{count} in {insert_s:.2f}s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--limit", type=int, default=20000,
                    help="cap rows; FISHDBC is single-threaded Python and the "
                         "per-batch reference is O(n^2), so keep this modest")
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--base-fraction", type=float, default=0.5)
    ap.add_argument("--mpts", type=int, default=10,
                    help="min_cluster_size AND min_samples on every side")
    ap.add_argument("--k", type=int, default=32, help="k_max for the CORE-SG")
    ap.add_argument("--metric", default="l2")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("--skip-fishdbc", action="store_true",
                    help="run only the CORE-SG side (FISHDBC not installed)")
    ap.add_argument("--skip-coresg", action="store_true")
    args = ap.parse_args()
    args.name = args.name or os.path.splitext(os.path.basename(args.data))[0]

    exe = find_binary("coresg_exp")
    x = read_fvecs(args.data, args.limit)
    n, dim = x.shape
    print(f"{args.name}: {n} x {dim}, mpts={args.mpts}, {args.batches} batches")

    sink = CsvSink(args.out)
    work = tempfile.mkdtemp(prefix="coresg_stream_")
    try:
        for s in range(args.seeds):
            seed = 9000 + s
            print(f"seed {seed}:", flush=True)
            # One order, both methods. This is what makes the per-batch rows
            # joinable on (dataset, seed, batch).
            order = np.random.default_rng(seed).permutation(n)
            xs = x[order]
            # The reference is O(n^2) and identical for both methods at a given
            # batch, so it is computed once and shared.
            ref_cache = {}
            if not args.skip_coresg:
                run_coresg(args, exe, xs, order, seed, work, sink, dim, ref_cache)
            if not args.skip_fishdbc:
                run_fishdbc(args, xs, seed, sink, dim, ref_cache)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    print(f"\nwrote {args.out}")
    print("Join stream_coresg and stream_fishdbc on (dataset, seed, batch).\n"
          "Compare ari_vs_exact_hdbscan directly, and per-batch cost as a\n"
          "*slope* only -- the intercept is a language difference, not a result.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
