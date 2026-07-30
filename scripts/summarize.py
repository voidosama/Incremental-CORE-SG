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


def agg(df, experiment, config=None, metric=None):
    """Mean/std over seeds, indexed by dataset and config value."""
    d = df[df.experiment == experiment]
    if config is not None:
        d = d[d.config == config]
    if metric is not None:
        d = d[d.metric == metric]
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
    for cfg in sorted(d.config.unique()):
        t = d[(d.config == cfg) & (d.metric == "recall")].groupby(
            ["dataset", "value"])["measurement"].agg(["mean", "std", "count"])
        show(f"recall vs {cfg}", t if not t.empty else None, "{:.4f}")


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
    "budget": section_budget,
    "incremental": lambda d: section_incremental(d, "incremental"),
    "arrival": lambda d: section_incremental(d, "arrival_order"),
    "drift": section_drift,
    "mpts": section_mpts,
    "scaling": section_scaling,
    "sensitivity": section_sensitivity,
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
