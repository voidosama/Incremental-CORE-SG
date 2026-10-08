#!/usr/bin/env python
"""ARI between approximate and exact MSTs under FOSC/EOM selection.

The thesis says the clustering it compares is FOSC's, but the ARI it reports
comes from a single DBSCAN* cut at the mean edge weight, and admits that that
threshold is insensitive to what the bridges did. This closes that gap without
touching the C++: FOSC is excess-of-mass selection over the condensed tree,
the hdbscan package implements it, and it needs only an MST.

Input is the pair of trees `coresg_cli extract` writes, one per (dataset,
seed, mpts):

    <dir>/<dataset>_seed<S>_antihub_mpts<M>.tsv
    <dir>/<dataset>_seed<S>_exact_mpts<M>.tsv

Output is appended to the same CSV the C++ writes, with the same eleven
columns, so summarize.py and the thesis scripts read it unchanged.

    python scripts/fosc_ari.py --mst-dir data/mst_fosc --out results_parecer.csv
"""
import argparse
import csv
import glob
import os
import re
import sys

HEADER = ["experiment", "dataset", "n", "dim", "k", "config", "value",
          "seed", "batch", "metric", "measurement"]

NAME = re.compile(r"^(?P<ds>.+)_seed(?P<seed>\d+)_(?P<arm>antihub|exact)_mpts(?P<mpts>\d+)\.tsv$")


def read_edges(path):
    import numpy as np
    rows = []
    with open(path, newline="") as fh:
        rd = csv.reader(fh, delimiter="\t")
        head = next(rd)
        if head[:3] != ["u", "v", "weight"]:
            raise SystemExit("unexpected header in %s: %r" % (path, head))
        for r in rd:
            if len(r) >= 3:
                rows.append((float(r[0]), float(r[1]), float(r[2])))
    return np.asarray(rows, dtype="float64")


def labels_from_mst(edges, n, min_cluster_size):
    """FOSC: condensed tree, stability, excess of mass.

    Identical to the path scripts/compare_cuml.py already uses, kept separate
    so this script has no import-time dependency on that one.
    """
    import numpy as np
    from hdbscan import hdbscan_

    if edges.shape[0] != n - 1:
        raise ValueError("MST has %d edges for n=%d; expected %d"
                         % (edges.shape[0], n, n - 1))
    ordered = edges[np.argsort(edges[:, 2]), :]
    slt = hdbscan_.label(ordered)
    condensed = hdbscan_.condense_tree(slt, min_cluster_size)
    stability = hdbscan_.compute_stability(condensed)
    return np.asarray(hdbscan_.get_clusters(condensed, stability, "eom", False, False)[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mst-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mpts", default="8,16,24",
                    help="comma-separated mpts values to score")
    ap.add_argument("--dim", type=int, default=0,
                    help="recorded in the dim column only; 0 leaves it blank")
    args = ap.parse_args()

    try:
        from sklearn.metrics import adjusted_rand_score
    except ImportError:
        raise SystemExit("needs scikit-learn for adjusted_rand_score")

    wanted = {int(x) for x in args.mpts.split(",") if x.strip()}

    # Index every tsv by (dataset, seed, mpts, arm) so the two arms can be paired.
    found = {}
    for path in sorted(glob.glob(os.path.join(args.mst_dir, "*.tsv"))):
        m = NAME.match(os.path.basename(path))
        if not m:
            continue
        mpts = int(m.group("mpts"))
        if mpts not in wanted:
            continue
        found[(m.group("ds"), int(m.group("seed")), mpts, m.group("arm"))] = path

    pairs = sorted({(ds, s, mp) for (ds, s, mp, arm) in found})
    if not pairs:
        raise SystemExit("no MST pairs under %s" % args.mst_dir)

    new_file = not os.path.exists(args.out)
    written = skipped = 0
    with open(args.out, "a", newline="") as fh:
        w = csv.writer(fh)
        if new_file:
            w.writerow(HEADER)
        for ds, seed, mpts in pairs:
            pa = found.get((ds, seed, mpts, "antihub"))
            pe = found.get((ds, seed, mpts, "exact"))
            if not pa or not pe:
                print("-- %s seed=%d mpts=%d: missing one arm, skipping" % (ds, seed, mpts))
                skipped += 1
                continue
            ea, ee = read_edges(pa), read_edges(pe)
            n = ea.shape[0] + 1
            if ee.shape[0] + 1 != n:
                print("-- %s seed=%d mpts=%d: arms disagree on n (%d vs %d), skipping"
                      % (ds, seed, mpts, n, ee.shape[0] + 1))
                skipped += 1
                continue
            try:
                la = labels_from_mst(ea, n, mpts)
                le = labels_from_mst(ee, n, mpts)
            except ValueError as e:
                print("-- %s seed=%d mpts=%d: %s" % (ds, seed, mpts, e))
                skipped += 1
                continue
            ari = adjusted_rand_score(le, la)
            # Cluster counts travel with the ARI: an ARI of 1 over two labellings
            # that both found a single cluster says much less than one over two
            # labellings that found forty, and the parecer asks for exactly that
            # distinction.
            ka = len(set(la.tolist()) - {-1})
            ke = len(set(le.tolist()) - {-1})
            for metric, value in (("ari_fosc", ari),
                                  ("fosc_clusters_approx", ka),
                                  ("fosc_clusters_exact", ke)):
                w.writerow(["fosc", ds, n, args.dim or "", mpts,
                            "mpts", mpts, seed, -1, metric, value])
                written += 1
            print("%-18s seed=%-4d mpts=%-3d  ARI=%.6f  clusters %d vs %d"
                  % (ds, seed, mpts, ari, ka, ke))

    print("\nwrote %d rows to %s (%d pairs skipped)" % (written, args.out, skipped))
    return 0


if __name__ == "__main__":
    sys.exit(main())
