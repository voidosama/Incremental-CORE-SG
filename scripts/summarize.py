#!/usr/bin/env python3
"""Headline tables from results.csv.

Safe to run mid-sweep: the harness flushes every row as it is produced, so
whatever has finished is already readable. Incomplete experiments simply show
fewer rows.

    python scripts/summarize.py                      # everything available
    python scripts/summarize.py --only bridges       # one section
    python scripts/summarize.py --csv summary.csv    # also write a tidy file

Every figure is mean +/- standard deviation over seeds. A blank cell means that
combination has not finished yet.
"""

import argparse
import os
import sys

try:
    import pandas as pd
except ImportError:
    raise SystemExit("needs pandas:  conda activate coresg")

# The tables print "+/-" as U+00B1, which the Windows console's default code
# page cannot encode -- every spread came out as a replacement character.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def agg(df, experiment, config=None, metric=None, batch=None):
    """Mean/std over seeds, indexed by dataset and config value.

    `batch` matters wherever an experiment emits the same metric name both
    per-batch and as a run-level total -- the refinement cap sweep emits
    `insert_seconds` for every batch *and* summed at batch=-1, and pooling the
    two produces a mean of neither.
    """
    d = df[df.experiment == experiment]
    if config is not None:
        d = d[d.config == config]
    if metric is not None:
        d = d[d.metric == metric]
    if batch is not None:
        d = d[d.batch == batch]
    if d.empty:
        return None
    return d.groupby(["dataset", "value"])["measurement"].agg(["mean", "std", "count"])


def show(title, table, fmt="{:.5f}", note=None):
    print(f"\n### {title}")
    if note:
        print(f"    {note}")
    if table is None or table.empty:
        print("    (no rows yet)")
        return
    for (ds, val), row in table.iterrows():
        sd = "" if pd.isna(row["std"]) else f" ± {fmt.format(row['std'])}"
        print(f"    {ds:<18} {str(val):<16} {fmt.format(row['mean'])}{sd}"
              f"   (n={int(row['count'])})")


def section_bridges(df):
    print("\n" + "=" * 78)
    print("BRIDGES -- anti-hub estimate vs the exact MST_kmax")
    print("=" * 78)
    show("MST total weight ratio (approx / exact; 1.0 == identical cost)",
         agg(df, "bridges", "mpts", "weight_ratio"), "{:.5f}",
         "The primary quality number. Edge overlap is NOT reported: two optimal "
         "MSTs\n    of the same graph can share few edges and still induce the "
         "identical clustering.")
    show("DBSCAN* ARI vs exact (1.0 == identical clustering)",
         agg(df, "bridges", "mpts", "ari_vs_exact"), "{:.4f}")
    show("Components left by the k-NN graph alone (the no-bridges ablation)",
         agg(df, "bridges", "mode", "components_before_repair"), "{:.1f}",
         "Above 1 means the bridges have real work to do on that data set.")
    show("Bridge construction time (s)",
         agg(df, "bridges", "mode", "bridge_seconds"), "{:.4f}")
    show("Anti-hub count sweep -- weight ratio vs how many were selected",
         agg(df, "bridges", "max_antihubs", "weight_ratio"), "{:.5f}")


def section_budget(df):
    print("\n" + "=" * 78)
    print("BUDGET -- which rule to adopt for d * ln(|E|)")
    print("=" * 78)
    show("Anti-hubs selected", agg(df, "budget_dim", "mode", "antihubs_selected"), "{:.0f}")
    show("As a fraction of n (1.0 means it selected every point)",
         agg(df, "budget_dim", "mode", "antihub_fraction_of_n"), "{:.3f}")
    show("Bridge time (s)", agg(df, "budget_dim", "mode", "bridge_seconds"), "{:.4f}")
    show("Effective dimensionality used",
         agg(df, "budget_dim", "mode", "effective_dim"), "{:.1f}")
    show("MST weight ratio by rule and mpts",
         agg(df, "budget_dim", "mode_mpts", "weight_ratio"), "{:.5f}")
    show("ARI vs exact by rule and mpts",
         agg(df, "budget_dim", "mode_mpts", "ari_vs_exact"), "{:.4f}")


def section_incremental(df, experiment="incremental"):
    print("\n" + "=" * 78)
    print(f"{experiment.upper()} -- incremental insertion vs full rebuild")
    print("=" * 78)
    d = df[df.experiment == experiment]
    if d.empty:
        print("    (no rows yet)")
        return
    for metric, fmt in [("recall_incremental", "{:.4f}"), ("recall_rebuild", "{:.4f}"),
                        ("incremental_total_seconds", "{:.2f}"),
                        ("rebuild_seconds", "{:.2f}"),
                        ("ari_vs_rebuild", "{:.4f}")]:
        t = d[d.metric == metric].groupby(["dataset", "value"])["measurement"].agg(
            ["mean", "std", "count"])
        show(metric, t if not t.empty else None, fmt)

    # The equal-work control. A bare rebuild is a cheaper baseline, not an
    # equal one -- the incremental path re-descends the base once per batch --
    # so the honest comparison reads the incremental point against the
    # rebuild's (cost, recall) frontier rather than against a single point.
    eff = d[d.config == "rebuild_effort"]
    if not eff.empty:
        piv = eff.pivot_table(index=["dataset", "value"], columns="metric",
                              values="measurement", aggfunc="mean")
        cols = [c for c in ("rebuild_seconds", "recall_rebuild", "refine_rounds",
                            "distance_evals") if c in piv.columns]
        print("\n### equal-work frontier: rebuild effort vs the incremental point")
        print("    A flat frontier with unchanged refine_rounds/distance_evals means the")
        print("    descent had already converged, i.e. the rebuild cannot be given more")
        print("    useful work -- which is a different claim from 'incremental is better'.")
        print(piv[cols].round(5).to_string())

    # Per-batch view: cost and quality as the stream lengthens.
    pb = d[d.metric.isin(["recall_new", "insert_seconds", "refine_nodes_fraction"])]
    if not pb.empty:
        print("\n### per batch (mean over seeds)")
        piv = pb.pivot_table(index=["dataset", "batch"], columns="metric",
                             values="measurement", aggfunc="mean")
        piv = piv[piv.index.get_level_values("batch") >= 0]
        with pd.option_context("display.width", 120):
            print(piv.round(4).to_string())


def section_drift(df):
    print("\n" + "=" * 78)
    print("DRIFT -- does quality decay over a long stream?")
    print("=" * 78)
    d = df[df.experiment == "drift"]
    if d.empty:
        print("    (no rows yet)")
        return
    piv = d[d.metric.isin(["recall_all", "recall_delta_vs_first", "insert_seconds"])]
    piv = piv.pivot_table(index=["dataset", "batch"], columns="metric",
                          values="measurement", aggfunc="mean")
    print(piv.round(4).to_string())
    print("\n    recall_delta_vs_first staying flat (or positive) means the graph is "
          "not rotting.")


def section_mpts(df):
    print("\n" + "=" * 78)
    print("MPTS SWEEP -- the reason CORE-SG exists")
    print("=" * 78)
    show("Seconds for the whole sweep, by phase",
         agg(df, "mpts_sweep", "phase", "seconds"), "{:.3f}")
    show("Complete-graph equivalent (extrapolated)",
         agg(df, "mpts_sweep", "phase", "seconds_extrapolated"), "{:.3f}")
    show("Speed-up", agg(df, "mpts_sweep", "phase", "ratio"), "{:.1f}")


def section_scaling(df):
    print("\n" + "=" * 78)
    print("SCALING -- Claim B")
    print("=" * 78)
    show("Insert seconds by CPU threads",
         agg(df, "scaling", "cpu_threads", "insert_seconds"), "{:.3f}")
    show("Insert seconds by GPUs x instances",
         agg(df, "scaling", "gpus_x_instances", "insert_seconds"), "{:.3f}",
         "Instance count is fixed across device counts, so 1xN vs 4xN is a real\n"
         "    strong-scaling comparison.")
    show("Parallel fraction f (only the instance phase spreads across devices)",
         agg(df, "scaling", "gpus_x_instances", "parallel_fraction"), "{:.3f}")
    show("Largest instance resident bytes (should stay flat as devices are added)",
         agg(df, "scaling", "gpus_x_instances", "largest_instance_bytes"), "{:.0f}")

    f = df[(df.experiment == "scaling") & (df.metric == "parallel_fraction")]
    if not f.empty:
        fm = f["measurement"].mean()
        print(f"\n    Amdahl ceiling at f={fm:.3f}:  "
              + "  ".join(f"{p}x: {1/((1-fm)+fm/p):.2f}x" for p in (2, 4, 8)))


def section_sensitivity(df):
    print("\n" + "=" * 78)
    print("SENSITIVITY")
    print("=" * 78)
    d = df[df.experiment == "sensitivity"]
    if d.empty:
        print("    (no rows yet)")
        return
    # The reference row first: without it a flat column of 1s below is
    # ambiguous between "the quota worked" and "nothing was ever disconnected".
    show("Components in the k-NN half alone (the reference the sweep is read against)",
         agg(df, "sensitivity", "reference", "components_knn_only"), "{:.1f}",
         "1 means this data set does not fragment, so the bridge-coverage knobs\n"
         "    below cannot move anything on it -- that is a property of the data,\n"
         "    not of the parameters.")
    for cfg in sorted(d.config.unique()):
        if cfg == "reference":
            continue
        t = d[(d.config == cfg) & (d.metric == "recall")].groupby(
            ["dataset", "value"])["measurement"].agg(["mean", "std", "count"])
        show(f"recall vs {cfg}", t if not t.empty else None, "{:.4f}")
    # Bridge coverage: the metric the anti-hub knobs can actually move.
    for cfg in ("antihub_multiplier", "min_per_component"):
        show(f"components after bridges vs {cfg}",
             agg(df, "sensitivity", cfg, "components_before_repair"), "{:.1f}")
    # Refinement cap policy: fractional caps grow with n, absolute ones do not.
    show("recall by refinement cap policy",
         agg(df, "sensitivity", "refine_cap", "recall", batch=-1), "{:.5f}",
         "frac_* caps at a fraction of n; nodes_* is an absolute ceiling.")
    show("total insert seconds by refinement cap policy",
         agg(df, "sensitivity", "refine_cap", "insert_seconds", batch=-1), "{:.4f}")
    show("refinement set as a fraction of the graph (is the cap binding?)",
         agg(df, "sensitivity", "refine_cap", "refine_nodes_fraction", batch=-1), "{:.4f}")

    # The slope, which is the whole reason the absolute arms exist. Within one
    # run the graph doubles, so a fractional cap should roughly double the
    # per-batch cost while an absolute one holds it flat.
    rc = df[(df.experiment == "sensitivity") & (df.config == "refine_cap") &
            (df.batch >= 0) & (df.metric == "insert_seconds")]
    if not rc.empty:
        piv = rc.pivot_table(index=["dataset", "value"], columns="batch",
                             values="measurement", aggfunc="mean")
        first, last = piv.columns.min(), piv.columns.max()
        piv = piv[[first, last]].copy()
        piv.columns = [f"batch_{first}_s", f"batch_{last}_s"]
        piv["growth"] = piv.iloc[:, 1] / piv.iloc[:, 0]
        print("\n### per-batch cost growth as the graph doubles")
        print("    growth ~1.0 => an update costs Theta(batch); ~2.0 => Theta(n).")
        print("    This is the comparison against an HNSW-based competitor whose")
        print("    per-batch cost is flat by construction.")
        print(piv.round(4).to_string())


def section_connectivity(df):
    d = df[df.experiment == "connectivity"]
    if d.empty:
        return
    print("\n" + "=" * 78)
    print("CONNECTIVITY -- how broken is the k-NN half, and what does the gate cost?")
    print("=" * 78)
    print("    components_before_repair is recorded AFTER the bridges are folded in, so")
    print("    it reads 1 on every other experiment. These are the k-NN half alone.")
    show("Components of the k-NN graph, by arm and k",
         agg(df, "connectivity", "half_k", "components_knn_only"), "{:.1f}",
         "approx_full: production scale, NN-Descent half.\n"
         "    approx_matched / exact_matched: same n, so the pair isolates the effect\n"
         "    of approximating the half on connectivity.")
    show("Recall of the approximate half against the exact one, same n",
         agg(df, "connectivity", "half_k", "knn_recall_of_approx_half"), "{:.5f}")
    show("Cost of deriving the component labelling (the gate), seconds",
         agg(df, "connectivity", "half_k", "components_seconds"), "{:.5f}")
    show("What the gate would skip: hubness + bridge seconds",
         agg(df, "connectivity", "gate", "bridge_seconds"), "{:.5f}")
    show("...as a fraction of build time",
         agg(df, "connectivity", "gate", "bridge_fraction_of_build"), "{:.4f}")


def section_batchsize(df):
    d = df[df.experiment == "batchsize"]
    if d.empty:
        return
    print("\n" + "=" * 78)
    print("BATCH SIZE -- where the fixed per-update cost stops being amortised")
    print("=" * 78)
    show("Seconds per inserted point", agg(df, "batchsize", "batch_fraction",
                                           "seconds_per_inserted_point"), "{:.6f}")
    show("Total insert seconds (the `rebuild` row is the line to beat)",
         agg(df, "batchsize", "batch_fraction", "insert_total_seconds"), "{:.3f}")
    show("Fraction of update time in phases that do not shrink with the batch",
         agg(df, "batchsize", "batch_fraction", "fixed_phase_fraction"), "{:.4f}",
         "hubness + bridges + connectivity, all of which run once per insert\n"
         "    however few points arrived.")
    show("Final recall", agg(df, "batchsize", "batch_fraction", "recall_final"), "{:.5f}")


def section_parity(df):
    d = df[df.experiment == "parity"]
    if d.empty:
        return
    print("\n" + "=" * 78)
    print("BACKEND PARITY -- do the CPU and CUDA paths produce the same graph?")
    print("=" * 78)
    show("Jobs that really ran on a device (0 on the cpu arm is expected;\n"
         "    0 on the cuda arm means a silent host demotion and voids the comparison)",
         agg(df, "parity", "backend", "jobs_on_device"), "{:.0f}")
    show("k-NN recall against one brute-force reference",
         agg(df, "parity", "backend", "recall"), "{:.5f}")
    show("DBSCAN* ARI between the two backends",
         agg(df, "parity", "backend", "ari_gpu_vs_cpu"), "{:.5f}")
    show("MST weight ratio, gpu/cpu -- SYMMETRIC here, either side of 1.0 is fine",
         agg(df, "parity", "backend", "mst_weight_ratio_symmetric"), "{:.5f}")


def section_headtohead(df):
    d = df[df.experiment.isin(["stream_coresg", "stream_fishdbc"])]
    if d.empty:
        return
    print("\n" + "=" * 78)
    print("HEAD TO HEAD -- CORE-SG vs FISHDBC, one arrival order, one reference")
    print("=" * 78)
    print("    Both scored as HDBSCAN* against exact HDBSCAN* on the points present at")
    print("    that batch, via the same extraction code. Compare ARI directly; compare")
    print("    cost as a SLOPE only -- FISHDBC is single-threaded Python.")
    ari = d[d.metric == "ari_vs_exact_hdbscan"].pivot_table(
        index=["dataset", "batch"], columns="experiment", values="measurement",
        aggfunc="mean")
    ref = d[d.metric == "reference_clusters"].groupby(
        ["dataset", "batch"])["measurement"].mean()
    if not ari.empty:
        ari = ari.join(ref.rename("reference_clusters"))
        print("\n### ARI vs exact HDBSCAN*")
        print(ari.round(4).to_string())
    cost = d[d.metric == "insert_seconds"].pivot_table(
        index=["dataset", "batch"], columns="experiment", values="measurement",
        aggfunc="mean")
    if not cost.empty:
        print("\n### per-batch insert seconds (read the slope, not the level)")
        print(cost.round(4).to_string())


def section_fishdbc(df):
    d = df[df.experiment == "baseline_fishdbc"]
    if d.empty:
        return
    print("\n" + "=" * 78)
    print("FISHDBC BASELINE")
    print("=" * 78)
    print("    Compare ARI and the per-batch cost SLOPE, never raw wall-clock:")
    print("    FISHDBC is single-threaded Python, this project is C++/CUDA.")
    piv = d[d.metric.isin(["ari_vs_exact_hdbscan", "insert_seconds"])].pivot_table(
        index=["dataset", "batch"], columns="metric", values="measurement", aggfunc="mean")
    print(piv.round(4).to_string())


SECTIONS = {
    "bridges": section_bridges,
    "connectivity": section_connectivity,
    "budget": section_budget,
    "incremental": lambda d: section_incremental(d, "incremental"),
    "arrival": lambda d: section_incremental(d, "arrival_order"),
    "drift": section_drift,
    "mpts": section_mpts,
    "scaling": section_scaling,
    "sensitivity": section_sensitivity,
    "batchsize": section_batchsize,
    "parity": section_parity,
    "headtohead": section_headtohead,
    "fishdbc": section_fishdbc,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results.csv")
    ap.add_argument("--only", help="comma-separated: " + ",".join(SECTIONS))
    ap.add_argument("--csv", help="also write a tidy mean/std table here")
    args = ap.parse_args()

    if not os.path.exists(args.results):
        raise SystemExit(f"{args.results} not found")
    df = pd.read_csv(args.results)
    # Seeds are counted per experiment: each uses its own base (bridges 2000+,
    # budget 7000+, ...) so a global unique count would overstate the repeats.
    per_exp = df.groupby("experiment")["seed"].nunique()
    print(f"{args.results}: {len(df):,} rows, {df.experiment.nunique()} experiments, "
          f"{df.dataset.nunique()} data sets, "
          f"{per_exp.min()}-{per_exp.max()} seeds per experiment")

    want = set(args.only.split(",")) if args.only else set(SECTIONS)
    for name, fn in SECTIONS.items():
        if name in want:
            fn(df)

    if args.csv:
        tidy = df.groupby(["experiment", "dataset", "config", "value", "metric"])[
            "measurement"].agg(["mean", "std", "count"]).reset_index()
        tidy.to_csv(args.csv, index=False)
        print(f"\nwrote {args.csv} ({len(tidy):,} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
