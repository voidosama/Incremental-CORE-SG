#!/usr/bin/env python
"""cuML HDBSCAN with its own partitioning turned on.

scripts/compare_cuml.py constructs HDBSCAN(min_cluster_size, min_samples) and
nothing else, which is the brute-force, single-partition path. The banca is
right that this is not the configuration to compare against: since the knn
partitioning parameters landed, cuML can itself split the data and process
sets larger than device memory.

The parameters live inside build_kwds, not at the top level:

    HDBSCAN(min_cluster_size=..., min_samples=...,
            build_algo="nn_descent",
            build_kwds={"knn_n_clusters": C, "knn_overlap_factor": V})

with knn_overlap_factor strictly less than knn_n_clusters, so the smallest
partitioned setting is C=3, V=2. The docs give device residency as roughly
(knn_overlap_factor / knn_n_clusters) * n rows, which is the same shape as
this work's (1 + omega) * n / m: at omega = 1 and m = C the two coincide.
That correspondence is the point of the comparison.

This script probes what the installed cuML actually accepts before running,
so an older build that lacks the parameters reports that instead of silently
measuring the unpartitioned path again.

    python scripts/cuml_partitioned.py --out results_parecer.csv \
        --data data/sift/sift_base.fvecs --n 500000,1000000 \
        --mpts 10 --clusters 1,4,8,16 --overlap 2 --seeds 42 43 44
"""
import argparse
import csv
import gc
import inspect
import os
import subprocess
import sys
import threading
import time

HEADER = ["experiment", "dataset", "n", "dim", "k", "config", "value",
          "seed", "batch", "metric", "measurement"]


def read_fvecs(path, limit):
    import numpy as np
    raw = np.fromfile(path, dtype="int32")
    if raw.size == 0:
        raise SystemExit("empty file: %s" % path)
    d = int(raw[0])
    row = d + 1
    total = raw.size // row
    take = min(total, limit) if limit else total
    v = raw.reshape(total, row)[:take, 1:]
    return v.view("float32").astype("float32"), d


def capabilities():
    """What this cuML understands. Reported, not assumed."""
    try:
        import cuml
        from cuml.cluster import HDBSCAN
    except ImportError as e:
        raise SystemExit("cuML not importable: %s" % e)
    sig = inspect.signature(HDBSCAN.__init__)
    names = set(sig.parameters)
    return cuml.__version__, HDBSCAN, ("build_algo" in names), ("build_kwds" in names), names


class DevicePeak:
    """Peak device memory attributable to this run, sampled while it runs.

    The first attempt read RMM statistics after the fact, which returns
    nothing unless statistics were enabled, and would have missed a
    transient peak even if it had.

    Two sources, in order. Per-process memory from nvidia-smi is the one
    to want, but some drivers report it as [N/A] (Windows WDDM does), so
    the fallback is total device memory with the pre-run level subtracted.
    The fallback is only honest on a quiet machine: anything else
    allocating on the same card lands in the number. Which source was
    used is recorded alongside the measurement.
    """

    def __init__(self, interval=0.1):
        self.interval = interval
        self.peak = -1
        self.mode = "none"
        self._base = None
        self._stop = threading.Event()
        self._t = None

    @staticmethod
    def _smi(query):
        try:
            out = subprocess.run(["nvidia-smi", "--query-" + query,
                                  "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=10)
            return out.stdout if out.returncode == 0 else ""
        except Exception:
            return ""

    def _per_process(self):
        me = str(os.getpid())
        total = None
        for line in self._smi("compute-apps=pid,used_memory").splitlines():
            parts = [q.strip() for q in line.split(",")]
            if len(parts) >= 2 and parts[0] == me:
                try:
                    total = (total or 0) + float(parts[1])
                except ValueError:
                    return None          # driver reports [N/A]
        return total

    def _device_total(self):
        vals = []
        for line in self._smi("gpu=memory.used").splitlines():
            try:
                vals.append(float(line.strip()))
            except ValueError:
                pass
        return sum(vals) if vals else None

    def _sample_once(self):
        mb = self._per_process()
        if mb is not None:
            self.mode = "per_process"
            self.peak = max(self.peak, int(mb * 1024 * 1024))
            return
        mb = self._device_total()
        if mb is None:
            return
        if self._base is None:
            self._base = mb
        self.mode = "device_total_delta"
        self.peak = max(self.peak, int(max(0.0, mb - self._base) * 1024 * 1024))

    def _loop(self):
        while not self._stop.is_set():
            self._sample_once()
            self._stop.wait(self.interval)

    def __enter__(self):
        # baseline before any allocation, for the fallback path
        mb = self._device_total()
        if mb is not None:
            self._base = mb
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._t:
            self._t.join(timeout=5)
        self._sample_once()
        return False


def release_pool():
    """Best effort between runs, so an OOM does not poison the next line."""
    try:
        import rmm
        rmm.reinitialize()
    except Exception:
        pass
    try:
        import cupy
        cupy.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass
    gc.collect()


def one_fit(HDBSCAN, x, mpts, clusters, overlap, build_algo, supports_kwds):
    kwargs = dict(min_cluster_size=mpts, min_samples=mpts)
    if build_algo:
        kwargs["build_algo"] = build_algo
    if supports_kwds and clusters and clusters > 1:
        # knn_overlap_factor must be < knn_n_clusters; clamp rather than fail,
        # and record what was actually asked for.
        v = min(overlap, clusters - 1)
        kwargs["build_kwds"] = {"knn_n_clusters": clusters, "knn_overlap_factor": v}
    with DevicePeak() as peak:
        t0 = time.perf_counter()
        model = HDBSCAN(**kwargs)
        labels = model.fit_predict(x)
        secs = time.perf_counter() - t0
    try:
        labels = labels.get()
    except AttributeError:
        pass
    return labels, secs, peak.peak, kwargs, peak.mode


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--name", default="")
    ap.add_argument("--n", default="500000,1000000")
    ap.add_argument("--mpts", type=int, default=10)
    ap.add_argument("--clusters", default="1,4,8,16",
                    help="knn_n_clusters values; 1 is the unpartitioned baseline")
    ap.add_argument("--overlap", type=int, default=2, help="knn_overlap_factor")
    ap.add_argument("--build-algo", default="nn_descent",
                    help="brute_force or nn_descent; empty leaves the default")
    ap.add_argument("--seeds", nargs="*", type=int, default=[42])
    args = ap.parse_args()

    version, HDBSCAN, has_algo, has_kwds, names = capabilities()
    label = args.name or os.path.splitext(os.path.basename(args.data))[0]
    print("cuML %s" % version)
    print("  build_algo supported: %s" % has_algo)
    print("  build_kwds supported: %s" % has_kwds)
    if not has_kwds:
        print("  !! this build has no build_kwds, so knn_n_clusters cannot be set.")
        print("     Partitioned rows will be skipped; the C=1 baseline still runs,")
        print("     and that result is itself the answer to parecer item 3.3.")
    print("  constructor parameters: %s" % ", ".join(sorted(names - {"self"})))

    sizes = [int(s) for s in args.n.split(",") if s.strip()]
    clusters = [int(c) for c in args.clusters.split(",") if c.strip()]

    new_file = not os.path.exists(args.out)
    written = 0
    with open(args.out, "a", newline="") as fh:
        w = csv.writer(fh)
        if new_file:
            w.writerow(HEADER)
        for n in sizes:
            x, dim = read_fvecs(args.data, n)
            if x.shape[0] < n:
                print("-- only %d rows available, skipping n=%d" % (x.shape[0], n))
                continue
            for c in clusters:
                if c > 1 and not has_kwds:
                    continue
                for seed in args.seeds:
                    tag = "C%d" % c
                    try:
                        labels, secs, peak, kwargs, peak_src = one_fit(
                            HDBSCAN, x, args.mpts, c, args.overlap,
                            args.build_algo or None, has_kwds)
                    except Exception as e:
                        # Out of memory at C=1 is a result, not a failure: it is
                        # the number the thesis quotes for where cuML stops.
                        print("-- n=%d %s seed=%d FAILED: %s" % (n, tag, seed, e))
                        w.writerow(["cuml_partitioned", label, n, dim, args.mpts,
                                    "clusters", c, seed, -1, "failed", 1])
                        written += 1
                        release_pool()
                        continue
                    nclust = len(set(int(v) for v in labels) - {-1})
                    for metric, value in (("seconds", secs),
                                          ("peak_device_bytes", peak),
                                          ("peak_device_source", peak_src),
                                          ("clusters_found", nclust)):
                        w.writerow(["cuml_partitioned", label, n, dim, args.mpts,
                                    "clusters", c, seed, -1, metric, value])
                        written += 1
                    fh.flush()
                    print("n=%-9d %-5s seed=%-4d %8.2fs  peak=%s  clusters=%d  %s"
                          % (n, tag, seed, secs,
                             ("%.2f GB" % (peak / 1e9)) if peak > 0 else "n/a",
                             nclust, kwargs.get("build_kwds", "{}")))
                    release_pool()
            del x

    print("\nwrote %d rows to %s" % (written, args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
