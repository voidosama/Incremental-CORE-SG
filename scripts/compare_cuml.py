#!/usr/bin/env python3
"""CORE-SG against NVIDIA cuML's GPU HDBSCAN, on one yardstick.

Why this baseline, and why it is different from the FISHDBC one
---------------------------------------------------------------
`compare_incremental.py` scores CORE-SG against FISHDBC and deliberately
*refuses* to race the wall-clocks: FISHDBC is single-threaded Python and this
project is C++/CUDA, so a timing comparison there measures the languages.

cuML HDBSCAN is the opposite situation and the comparison this project actually
needs. It is CUDA, it runs on the same card, and it is what a practitioner
would reach for. So here **wall-clock is the result**, not a caveat -- and the
question is sharp:

    when a batch of points arrives, is updating the CORE-SG and re-extracting
    MST_mpts cheaper than letting cuML rebuild HDBSCAN* on the new prefix?

cuML has no incremental path, so rebuilding is the only thing it can do. That
is not a strawman -- it is the state of the art for GPU density clustering, and
"you must rebuild" is precisely the cost this project sets out to remove.

Two claims are measured, and they are different claims
------------------------------------------------------
1. **Per-batch cost.** CORE-SG pays `insert_seconds + cluster_extract_seconds`;
   cuML pays a full `fit` on the points present. Both produce HDBSCAN* labels
   at the same mpts. The *slope* matters more than the intercept: CORE-SG's
   update is O(batch) and cuML's rebuild is O(n), so the ratio must widen as
   the stream lengthens. A single (n, ratio) point proves nothing.

2. **Multiple mpts from one graph.** This is the CORE-SG paper's own reason to
   exist, and cuML has no equivalent: every mpts is a separate `fit`. Measured
   at the final batch over `--mpts-sweep`, CORE-SG re-extracts from the graph
   it already has while cuML refits per value. Reported separately because it
   is a different contribution from incrementality and should not be silently
   folded into claim 1.

Quality is not assumed away
---------------------------
Both sides are scored by ARI against the *same* reference -- exact HDBSCAN*
recomputed on the points present at that batch -- and the two are also scored
against each other. A speed claim with no agreement column is worthless, and a
per-batch reference is required: computing it once on the whole file and
slicing scores a method against structure it has not been shown.

`min_samples` is pinned to `--mpts` on every side. cuML defaults
`min_samples` to `min_cluster_size`, sklearn does the same, and FISHDBC
defaults to 5; left alone, the three sides silently cluster at three different
densities.

Install (Linux / WSL2 only -- RAPIDS does not ship for native Windows):

    conda install -c rapidsai -c conda-forge -c nvidia cuml=25.* python=3.12 cuda-version=12.*

Usage:

    python scripts/compare_cuml.py --data data/sift/sift_base.fvecs \\
        --name sift1m --limit 200000 --k 32 --batches 8 --mpts 10 \\
        --out results.csv

    python scripts/compare_cuml.py --mode capacity --data data/capacity10m.fvecs \\
        --name capacity10m --k 32 --mpts 10 --out results.csv
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


# --------------------------------------------------------------------------
# Shared plumbing (kept byte-compatible with compare_incremental.py so the
# rows concatenate into one results.csv)
# --------------------------------------------------------------------------

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


def hdbscan_labels_from_mst(edges, n, min_cluster_size):
    """HDBSCAN* labels from an MST, via the same path FISHDBC's cluster() uses.

    `hdbscan.label` requires the edges sorted by weight; an MST arriving from
    elsewhere has to be sorted here or the single-linkage tree is wrong.
    """
    from hdbscan import hdbscan_

    if edges.shape[0] != n - 1:
        raise SystemExit(
            f"MST has {edges.shape[0]} edges for {n} points; expected {n - 1}. "
            "The graph is disconnected -- check that ensure_connected is on.")
    ordered = edges[np.argsort(edges[:, 2]), :]
    slt = hdbscan_.label(ordered)
    condensed = hdbscan_.condense_tree(slt, min_cluster_size)
    stability = hdbscan_.compute_stability(condensed)
    return np.asarray(hdbscan_.get_clusters(condensed, stability, "eom", False, False)[0])


def reference_labels(x, mpts):
    """Exact HDBSCAN*, the common yardstick, on the CPU."""
    try:
        from sklearn.cluster import HDBSCAN
        return HDBSCAN(min_cluster_size=mpts, min_samples=mpts).fit_predict(x)
    except ImportError:
        pass
    try:
        import hdbscan
        return hdbscan.HDBSCAN(min_cluster_size=mpts, min_samples=mpts).fit_predict(x)
    except ImportError:
        raise SystemExit("need scikit-learn >= 1.3 (HDBSCAN) or the hdbscan package")


# --------------------------------------------------------------------------
# cuML
# --------------------------------------------------------------------------

class OutOfMemory(Exception):
    """cuML could not fit the problem on the device. A *result*, not a crash."""


def import_cuml():
    try:
        from cuml.cluster import HDBSCAN as cuHDBSCAN
    except ImportError as exc:
        raise SystemExit(
            "cuML is not importable (%s).\n"
            "RAPIDS is Linux/WSL2 only; install with:\n"
            "  conda install -c rapidsai -c conda-forge -c nvidia cuml "
            "python=3.12 cuda-version=12.*" % exc)
    return cuHDBSCAN


def gpu_free_bytes():
    try:
        import cupy
        free, total = cupy.cuda.runtime.memGetInfo()
        return int(free), int(total)
    except Exception:
        return None, None


def cuml_fit(x, mpts, cuHDBSCAN):
    """One cuML HDBSCAN fit. Returns (labels, seconds, peak_device_bytes).

    Out-of-memory is translated into `OutOfMemory` rather than allowed to
    propagate: on the capacity scenario the OOM *is* the measurement, and a
    traceback in the middle of a sweep loses every row after it.
    """
    import cupy

    pool = cupy.get_default_memory_pool()
    pool.free_all_blocks()
    before = pool.used_bytes()
    try:
        cupy.cuda.runtime.deviceSynchronize()
        t0 = time.perf_counter()
        model = cuHDBSCAN(min_cluster_size=mpts, min_samples=mpts)
        labels = model.fit_predict(x)
        cupy.cuda.runtime.deviceSynchronize()
        seconds = time.perf_counter() - t0
    except (cupy.cuda.memory.OutOfMemoryError, MemoryError) as exc:
        raise OutOfMemory(str(exc))
    except RuntimeError as exc:
        # RAFT/RMM surface allocation failures as RuntimeError with a message.
        msg = str(exc).lower()
        if "out of memory" in msg or "bad_alloc" in msg or "rmm" in msg:
            raise OutOfMemory(str(exc))
        raise
    peak = max(0, pool.total_bytes() - before)
    return np.asarray(labels.get() if hasattr(labels, "get") else labels), seconds, peak


# --------------------------------------------------------------------------
# Mode: stream -- per-batch head to head
# --------------------------------------------------------------------------

def run_stream(args, sink):
    from sklearn.metrics import adjusted_rand_score

    cuHDBSCAN = import_cuml()
    exe = find_binary("coresg_exp")
    xs = read_fvecs(args.data, args.limit)
    n, dim = xs.shape
    print(f"loaded {n} x {dim} from {args.data}")

    base = int(args.base_fraction * n)
    per_batch = (n - base) // args.batches
    if per_batch == 0:
        raise SystemExit("too few points for that many batches")

    for seed in range(args.seed_base, args.seed_base + args.seeds):
        rng = np.random.default_rng(seed)
        order = rng.permutation(n).astype(np.int32)

        work = tempfile.mkdtemp(prefix="coresg_cuml_")
        try:
            order_path = os.path.join(work, f"order_{seed}.i32")
            order.tofile(order_path)
            mst_dir = os.path.join(work, f"mst_{seed}")
            os.makedirs(mst_dir, exist_ok=True)

            # --- CORE-SG: one incremental pass, MST exported per batch -----
            cmd = [exe, "--experiment", "stream",
                   "--data", args.data, "--name", args.name,
                   "--k", str(args.k), "--metric", args.metric,
                   "--batches", str(args.batches),
                   "--base-fraction", str(args.base_fraction),
                   "--export-mpts", str(args.mpts),
                   "--order-file", order_path, "--mst-dir", mst_dir,
                   "--seed-base", str(seed), "--seeds", "1",
                   "--out", args.out]
            if args.limit:
                cmd += ["--limit", str(args.limit)]
            if not args.gpu:
                cmd += ["--gpu", "0"]
            print("  " + " ".join(cmd))
            subprocess.run(cmd, check=True)

            # --- cuML: a full rebuild at every batch ------------------------
            cursor = base
            for b in range(1, args.batches + 1):
                cursor = min(base + b * per_batch, n)
                prefix = np.ascontiguousarray(xs[order[:cursor]], dtype=np.float32)

                try:
                    cu_labels, cu_seconds, cu_peak = cuml_fit(prefix, args.mpts, cuHDBSCAN)
                except OutOfMemory as exc:
                    print(f"  batch {b}: cuML OOM at n={cursor} -- {exc}")
                    sink.row("stream_cuml", args.name, cursor, dim, args.k, "method",
                             "cuml", seed, b, "out_of_memory", 1.0)
                    break

                sink.row("stream_cuml", args.name, cursor, dim, args.k, "method",
                         "cuml", seed, b, "rebuild_seconds", cu_seconds)
                sink.row("stream_cuml", args.name, cursor, dim, args.k, "method",
                         "cuml", seed, b, "device_bytes", float(cu_peak))
                sink.row("stream_cuml", args.name, cursor, dim, args.k, "method",
                         "cuml", seed, b, "out_of_memory", 0.0)

                # Agreement. The reference is exact HDBSCAN* on exactly the
                # points both methods have been shown at this batch.
                if args.reference_limit and cursor <= args.reference_limit:
                    ref = reference_labels(prefix, args.mpts)
                    sink.row("stream_cuml", args.name, cursor, dim, args.k, "method",
                             "cuml", seed, b, "ari_vs_reference",
                             adjusted_rand_score(ref, cu_labels))

                    mst_path = os.path.join(
                        mst_dir, f"{args.name}_seed{seed}_batch{b}.mst")
                    if os.path.exists(mst_path):
                        mn, edges = read_mst(mst_path)
                        cs_labels = hdbscan_labels_from_mst(edges, mn, args.mpts)
                        sink.row("stream_coresg", args.name, cursor, dim, args.k, "method",
                                 "coresg", seed, b, "ari_vs_reference",
                                 adjusted_rand_score(ref, cs_labels))
                        sink.row("stream_cuml", args.name, cursor, dim, args.k, "method",
                                 "cuml", seed, b, "ari_coresg_vs_cuml",
                                 adjusted_rand_score(cs_labels, cu_labels))
                print(f"  batch {b}: n={cursor} cuML rebuild {cu_seconds:.3f}s")

            # --- Claim 2: several mpts from one graph ----------------------
            if args.mpts_sweep:
                values = [int(v) for v in args.mpts_sweep.split(",") if v]
                full = np.ascontiguousarray(xs[order[:cursor]], dtype=np.float32)
                total = 0.0
                for m in values:
                    try:
                        _, s, _ = cuml_fit(full, m, cuHDBSCAN)
                    except OutOfMemory as exc:
                        print(f"  mpts={m}: cuML OOM -- {exc}")
                        break
                    total += s
                    sink.row("mpts_cuml", args.name, cursor, dim, args.k, "mpts",
                             str(m), seed, -1, "rebuild_seconds", s)
                sink.row("mpts_cuml", args.name, cursor, dim, args.k, "mpts",
                         "all", seed, -1, "rebuild_seconds_total", total)
                print(f"  cuML refit for {len(values)} mpts values: {total:.3f}s total "
                      f"(CORE-SG re-extracts these from one graph; see the `mpts` "
                      f"experiment's extract_seconds)")
        finally:
            shutil.rmtree(work, ignore_errors=True)


# --------------------------------------------------------------------------
# Mode: capacity -- where cuML runs out of device memory
# --------------------------------------------------------------------------

def run_capacity(args, sink):
    """Walk n upward until cuML cannot fit the problem on one GPU.

    The point is not that cuML is badly implemented -- it is that a rebuild
    baseline must hold the whole problem on one device, while this project's
    instances are sized by the *batch*, so the ceiling moves with the number of
    devices instead of the data set. The OOM is recorded as an outcome so the
    sweep produces a row rather than a traceback.
    """
    cuHDBSCAN = import_cuml()
    xs = read_fvecs(args.data, args.limit)
    n_total, dim = xs.shape
    free, total = gpu_free_bytes()
    print(f"loaded {n_total} x {dim}; device free {free} / {total} bytes")

    sizes = [int(v) for v in args.capacity_sizes.split(",") if v]
    sizes = [s for s in sizes if s <= n_total]
    if not sizes:
        raise SystemExit("no capacity sizes within the data set")

    for m in sizes:
        prefix = np.ascontiguousarray(xs[:m], dtype=np.float32)
        try:
            _, seconds, peak = cuml_fit(prefix, args.mpts, cuHDBSCAN)
        except OutOfMemory as exc:
            print(f"  n={m}: cuML OUT OF MEMORY -- {exc}")
            sink.row("capacity_cuml", args.name, m, dim, args.k, "method", "cuml",
                     args.seed_base, -1, "out_of_memory", 1.0)
            # Everything larger will also fail; stop rather than burn hours.
            break
        print(f"  n={m}: cuML {seconds:.3f}s, peak {peak / 2**30:.2f} GiB")
        sink.row("capacity_cuml", args.name, m, dim, args.k, "method", "cuml",
                 args.seed_base, -1, "rebuild_seconds", seconds)
        sink.row("capacity_cuml", args.name, m, dim, args.k, "method", "cuml",
                 args.seed_base, -1, "device_bytes", float(peak))
        sink.row("capacity_cuml", args.name, m, dim, args.k, "method", "cuml",
                 args.seed_base, -1, "out_of_memory", 0.0)

    print("\nNow run the CORE-SG side at the same sizes, which reports the per-device\n"
          "footprint that makes the comparison (largest_instance_bytes):\n"
          f"  {find_binary('coresg_exp')} --experiment capacity --data {args.data} "
          f"--name {args.name} --k {args.k} --scaling-instances 1,4,8,16 "
          f"--oom-fallback 0 --out {args.out}")


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["stream", "capacity"], default="stream")
    ap.add_argument("--data", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--metric", default="l2")
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--base-fraction", type=float, default=0.5)
    ap.add_argument("--mpts", type=int, default=10,
                    help="pinned on every side: CORE-SG export, cuML, reference")
    ap.add_argument("--mpts-sweep", default="",
                    help="comma-separated mpts values for the multiple-mpts claim")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--reference-limit", type=int, default=50000,
                    help="skip the exact HDBSCAN* reference above this n (it is CPU "
                         "and O(n^2)-ish); 0 disables quality scoring entirely")
    ap.add_argument("--capacity-sizes", default="1000000,2000000,4000000,8000000,10000000")
    ap.add_argument("--out", default="results.csv")
    args = ap.parse_args()

    sink = CsvSink(args.out)
    if args.mode == "stream":
        run_stream(args, sink)
    else:
        run_capacity(args, sink)


if __name__ == "__main__":
    main()
