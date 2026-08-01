#!/usr/bin/env python3
"""Run the thesis experiment suite.

Encodes the whole matrix -- which experiment runs on which data set, at what
size, with how many seeds -- in one place, so a run is reproducible and the
cluster run is the same command as the laptop run. Everything appends to a
single long-format CSV.

    python scripts/run_suite.py --generate          # build the synthetic sets
    python scripts/run_suite.py --list              # show the matrix
    python scripts/run_suite.py --dry-run
    python scripts/run_suite.py --out results.csv
    python scripts/run_suite.py --only bridges,budget --families manifold_exact

Three data set families, and the split matters:

  real_paper   the CORE-SG paper's own sets (FMA, 20 Newsgroups). Comparability
               with the baseline.
  real_scale   SIFT1M / GIST1M / GloVe. Scale, high dimensionality, and the
               regime where the k-NN graph is already connected -- which means
               the bridges have nothing to do there, so these sets exercise
               cost but NOT bridge quality.
  manifold     synthetic: low intrinsic dimensionality rotated into a high
               ambient one, many well-separated clusters. This is the family
               where the k-NN graph genuinely fragments, so it is the only one
               that tests what the anti-hub bridges are for. Sized so that the
               fragmentation survives the O(n^2) exact-MST reference: the graph
               only breaks apart when points-per-cluster greatly exceeds k, so
               small-n configurations must keep the cluster count low.
"""

import argparse
import csv
import os
import subprocess
import sys
import shutil

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


def data(*parts):
    return os.path.join(ROOT, "data", *parts)


# name -> dict(path, k, metric, limit, exact_limit, generate=?)
FAMILIES = {
    # ---- the CORE-SG paper's own data -------------------------------------
    "fma_chroma":   dict(group="real_paper", path=data("fma_chroma_cens.fvecs"), k=32),
    "fma_mfcc":     dict(group="real_paper", path=data("fma_mfcc.fvecs"), k=32),
    "20ng_tfidf":   dict(group="real_paper", path=data("20newsgroups_tfidf1000.fvecs"), k=24),
    "20ng_pca500":  dict(group="real_paper", path=data("20newsgroups_pca500.fvecs"), k=24),

    # ---- scale and high dimensionality ------------------------------------
    "sift1m":       dict(group="real_scale", path=data("sift", "sift_base.fvecs"),
                         k=32, limit=200000),
    "gist1m":       dict(group="real_scale", path=data("gist", "gist_base.fvecs"),
                         k=32, limit=100000),
    "glove100":     dict(group="real_scale", path=data("glove100.fvecs"),
                         k=32, limit=200000, metric="cosine"),

    # ---- where the bridges actually matter --------------------------------
    # Fragmentation needs points-per-cluster >> k. At n=4000 with k=24 that
    # caps the cluster count around 16, which still gives 16 components.
    "manifold_exact": dict(group="manifold", path=data("manifold_exact.fvecs"), k=24,
                           generate=dict(n=4000, dim=128, intrinsic=6, clusters=16,
                                         separation=30)),
    "manifold_mid":   dict(group="manifold", path=data("manifold_mid.fvecs"), k=24,
                           generate=dict(n=100000, dim=128, intrinsic=6, clusters=150,
                                         separation=25)),
    "manifold_wide":  dict(group="manifold", path=data("manifold_wide.fvecs"), k=24,
                           generate=dict(n=100000, dim=256, intrinsic=8, clusters=300,
                                         separation=40)),

    # Extra fragmenting configurations at exact-reference scale. Claim C rested
    # on manifold_exact alone, because every real data set turns out to have a
    # connected k-NN graph and therefore cannot exercise the bridges at all.
    # These vary the cluster count (hence the component count) at n=4000, which
    # is cheap and multiplies the evidence from one data set to four.
    "manifold_c8":   dict(group="manifold", path=data("manifold_c8.fvecs"), k=24,
                          generate=dict(n=4000, dim=128, intrinsic=6, clusters=8,
                                        separation=25)),
    "manifold_c24":  dict(group="manifold", path=data("manifold_c24.fvecs"), k=24,
                          generate=dict(n=4000, dim=64, intrinsic=4, clusters=24,
                                        separation=30)),
    "manifold_c32":  dict(group="manifold", path=data("manifold_c32.fvecs"), k=16,
                          generate=dict(n=4000, dim=128, intrinsic=6, clusters=32,
                                        separation=35)),

    # ---- capacity: does not fit on one 8 GB card --------------------------
    # k=64 on purpose: at ~848 B/point a single instance needs ~8.5 GB, which
    # does NOT fit on an 8 GB card. That is the point -- one instance OOMs,
    # four instances hold ~2.1 GB each and complete. The capacity argument is
    # about what becomes *possible*, not about speed.
    "capacity10m":  dict(group="capacity", path=data("capacity10m.fvecs"), k=64,
                         generate=dict(n=10000000, dim=32, intrinsic=8, clusters=200,
                                       separation=20)),
}

# experiment -> which families, plus per-experiment overrides
MATRIX = [
    # Quality of the bridge estimate. Needs the O(n^2) exact reference, so it
    # runs small -- and only `manifold` actually stresses it.
    # seeds=2 on purpose: these set brute_force_limit = n, so the k-NN half is
    # exact and the whole pipeline is deterministic -- every quality figure came
    # back with a standard deviation of exactly 0.00000 across 5 seeds. Extra
    # seeds buy timing jitter and nothing else.
    ("bridges", ["manifold_exact", "manifold_c8", "manifold_c24", "manifold_c32",
                 "20ng_pca500", "20ng_tfidf", "fma_chroma", "fma_mfcc",
                 "sift1m", "gist1m", "glove100"], dict(exact_limit=4000, seeds=2)),

    # Ambient vs intrinsic dimensionality in the budget.
    # 20ng_tfidf carries the extreme case and has to be here: at d=1000 the
    # ambient rule asks for more anti-hubs than there are points, so it selects
    # every one and the "approximation" silently becomes the exact MST it is
    # meant to replace. No other set shows that as starkly.
    ("budget", ["manifold_exact", "manifold_c8", "manifold_c24", "manifold_c32",
                "20ng_pca500", "20ng_tfidf", "fma_mfcc", "sift1m",
                "gist1m", "glove100"], dict(exact_limit=4000, seeds=2)),

    # Incremental vs full rebuild, and arrival order.
    # manifold_wide belongs in the full-scale experiments, not in bridges or
    # budget: those cap n at 4000, where 300 clusters leave ~13 points each and
    # the graph stops fragmenting -- destroying the 273-component structure that
    # makes it worth running. Here components_before_repair answers the coverage
    # question at full scale, with no quadratic reference needed.
    ("incremental", ["manifold_mid", "manifold_wide", "fma_chroma", "fma_mfcc",
                     "20ng_pca500", "sift1m", "gist1m", "glove100"], dict(batches=8)),
    ("arrival", ["manifold_mid", "fma_mfcc", "sift1m"], dict(batches=8)),

    # The point of CORE-SG: many MSTs from one graph.
    ("mpts", ["manifold_exact", "fma_mfcc", "20ng_pca500", "sift1m"],
     dict(exact_limit=5000)),

    # Parameter sensitivity.
    ("sensitivity", ["manifold_mid", "fma_mfcc", "sift1m"], dict()),

    # Long-horizon drift: many small batches, quality sampled at every one.
    ("drift", ["manifold_mid", "manifold_wide", "fma_mfcc", "sift1m"], dict(batches=25)),

    # Scaling. GPU rows only produce anything on a multi-GPU machine.
    ("scaling", ["manifold_mid", "sift1m", "gist1m"], dict()),

    # capacity10m's full scaling sweep already ran (36 configs on 10M points,
    # ~2 days) and its rows are in results.csv -- do NOT schedule it again.
    # What that sweep never tested is the configuration it existed for: the
    # instance counts were {4,8,16}, all of which fit, so instances=1 (7.90 GiB,
    # over an 8 GB card) never ran. This probe adds exactly that, with the OOM
    # fallback disabled so a refused job is recorded in seconds instead of
    # turning into a multi-hour CPU descent over 10M points.
    ("scaling", ["capacity10m"],
     dict(tag="oom", scaling_instances="1,2", seeds=1, no_oom_fallback=True,
          skip_cpu_scaling=True)),
]


def generate(only=None):
    prep = os.path.join(HERE, "prepare_datasets.py")
    for name, spec in FAMILIES.items():
        if only and name not in only:
            continue
        gen = spec.get("generate")
        if not gen:
            continue
        if os.path.exists(spec["path"]):
            print(f"  {name}: already present, skipping")
            continue
        cmd = [sys.executable, prep, "synthetic",
               "--n", str(gen["n"]), "--dim", str(gen["dim"]),
               "--intrinsic", str(gen["intrinsic"]),
               "--clusters", str(gen["clusters"]),
               "--separation", str(gen["separation"]),
               "--out", spec["path"]]
        print(f"  {name}: {' '.join(cmd[2:])}")
        subprocess.run(cmd, check=True)


def already_done(out_path):
    """(experiment, dataset) pairs already present in the CSV.

    A long sweep on a shared cluster will be interrupted -- a walltime limit, a
    preempted job, a bug in run 31 of 37. Results are appended, so completed
    work already survives; this is what stops a re-run from *duplicating* it.
    """
    done = set()
    if not os.path.exists(out_path):
        return done
    try:
        with open(out_path, newline="") as f:
            for row in csv.DictReader(f):
                done.add((row["experiment"], row["dataset"]))
    except (OSError, KeyError):
        pass
    return done


# coresg_exp writes rows under its own experiment names, which do not always
# match the matrix key (incremental/arrival share one function; drift and
# budget have their own).
EXPERIMENT_ROW_NAME = {
    "incremental": "incremental",
    "arrival": "arrival_order",
    "bridges": "bridges",
    "budget": "budget_dim",
    "mpts": "mpts_sweep",
    "scaling": "scaling",
    "sensitivity": "sensitivity",
    "drift": "drift",
}


def build_commands(args):
    exe = find_binary("coresg_exp")
    done = already_done(args.out) if args.resume else set()
    cmds = []
    for experiment, families, extra in MATRIX:
        if args.only and experiment not in args.only:
            continue
        for fam in families:
            if args.families and fam not in args.families:
                continue
            spec = FAMILIES[fam]
            if args.groups and spec["group"] not in args.groups:
                continue
            if not os.path.exists(spec["path"]):
                cmds.append((None, f"{experiment:<12} {fam:<16} MISSING {spec['path']}"))
                continue

            tag = extra.get("tag")
            label = fam if not tag else f"{fam}_{tag}"
            if args.resume:
                row_name = EXPERIMENT_ROW_NAME.get(experiment, experiment)
                if (row_name, label) in done:
                    cmds.append((None, f"{experiment:<12} {label:<24} done, skipping"))
                    continue
            # Per-row seeds override the global default: bridges/budget are
            # deterministic, capacity is expensive.
            seeds = extra.get("seeds", args.seeds)
            cmd = [exe, "--experiment", experiment,
                   "--data", spec["path"], "--name", label,
                   "--k", str(spec.get("k", 32)),
                   "--metric", spec.get("metric", "l2"),
                   "--seeds", str(seeds),
                   "--out", args.out, "--quiet"]
            if extra.get("scaling_instances"):
                cmd += ["--scaling-instances", extra["scaling_instances"]]
            if extra.get("no_oom_fallback"):
                cmd += ["--no-oom-fallback"]
            if extra.get("skip_cpu_scaling"):
                cmd += ["--skip-cpu-scaling"]
            if spec.get("limit"):
                cmd += ["--limit", str(spec["limit"])]
            if extra.get("exact_limit"):
                cmd += ["--exact-limit", str(extra["exact_limit"])]
            if extra.get("batches"):
                cmd += ["--batches", str(extra["batches"])]
            if not args.gpu:
                cmd += ["--gpu", "0"]
            cmds.append((cmd, f"{experiment:<12} {label:<24} k={spec.get('k', 32)}"))
    return cmds


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--only", help="comma-separated experiment names")
    ap.add_argument("--families", help="comma-separated family names")
    ap.add_argument("--groups", help="comma-separated: real_paper,real_scale,manifold,capacity")
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--generate", action="store_true", help="build the synthetic sets and exit")
    ap.add_argument("--list", action="store_true", help="print the matrix and exit")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--keep-going", action="store_true",
                    help="continue after a failing run instead of stopping")
    ap.add_argument("--resume", action="store_true",
                    help="skip runs already present in --out (safe to re-run after a "
                         "crash or a walltime kill without duplicating rows)")
    args = ap.parse_args()
    for key in ("only", "families", "groups"):
        v = getattr(args, key)
        setattr(args, key, set(v.split(",")) if v else None)

    if args.generate:
        print("generating synthetic data sets:")
        generate(args.families)
        return 0

    cmds = build_commands(args)
    if args.list or args.dry_run:
        for cmd, label in cmds:
            print(f"  {label}")
            if args.dry_run and cmd:
                print(f"      {' '.join(cmd)}")
        missing = sum(1 for c, _ in cmds if c is None)
        print(f"\n{len(cmds) - missing} runs, {missing} skipped for missing data")
        return 0

    failures = []
    total = sum(1 for c, _ in cmds if c)
    done = 0
    for cmd, label in cmds:
        if cmd is None:
            print(f"[skip] {label}")
            continue
        done += 1
        print(f"[{done}/{total}] {label}", flush=True)
        result = subprocess.run(cmd)
        if result.returncode != 0:
            failures.append(label)
            print(f"    FAILED (exit {result.returncode})")
            if not args.keep_going:
                print("stopping; pass --keep-going to continue past failures")
                break

    print(f"\nwrote {args.out}")
    if failures:
        print(f"{len(failures)} failed:")
        for f in failures:
            print(f"  {f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
