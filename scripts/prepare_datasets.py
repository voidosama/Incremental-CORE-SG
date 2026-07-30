#!/usr/bin/env python3
"""Prepare evaluation data sets in .fvecs, the format coresg reads natively.

Sub-commands
------------
  newsgroups   20 Newsgroups, following the CORE-SG paper's recipe exactly:
               the 1000 terms with the highest TF-IDF, then PCA to the 500
               principal components. Needs no manual download -- scikit-learn
               fetches the corpus itself -- so this data set is reproducible
               from the paper's text alone.
  hdf5         Convert an ANN-benchmarks .hdf5 (GloVe, SIFT, ...) to .fvecs.
  synthetic    Gaussian blobs with per-cluster densities spanning an order of
               magnitude, for the scale tests that no public set covers.
  inspect      Print shape/dtype/statistics of a .fvecs file.

Everything writes float32 .fvecs: per row, an int32 dimension followed by that
many float32 values.
"""

import argparse
import os
import struct
import sys

import numpy as np


def write_fvecs(path, arr):
    """Write an (n, d) float32 array as .fvecs."""
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    n, d = arr.shape
    with open(path, "wb") as f:
        header = struct.pack("<i", d)
        for i in range(n):
            f.write(header)
            f.write(arr[i].tobytes())
    print(f"wrote {path}: {n} x {d} ({os.path.getsize(path) / 1e6:.1f} MB)")


def read_fvecs(path, limit=None):
    raw = np.fromfile(path, dtype=np.int32)
    if raw.size == 0:
        raise SystemExit(f"{path} is empty")
    d = int(raw[0])
    stride = d + 1
    n = raw.size // stride
    if limit:
        n = min(n, limit)
    view = raw[: n * stride].reshape(n, stride)
    return view[:, 1:].view(np.float32)


def cmd_newsgroups(args):
    try:
        from sklearn.datasets import fetch_20newsgroups
        from sklearn.decomposition import PCA
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        raise SystemExit("needs scikit-learn:  pip install scikit-learn")

    print("fetching 20 Newsgroups (scikit-learn caches it under ~/scikit_learn_data)")
    # The CORE-SG paper reports 11,314 points, which is the *train* split
    # (train 11,314 + test 7,532 = 18,846). Default to train so the row count
    # matches the baseline paper exactly.
    bunch = fetch_20newsgroups(
        subset=args.subset, remove=("headers", "footers", "quotes"), random_state=0
    )
    print(f"  {len(bunch.data)} documents (subset={args.subset}; "
          f"the CORE-SG paper reports 11,314)")

    # The paper: "selected the top 1000 terms with the highest TF-IDF values".
    vec = TfidfVectorizer(max_features=args.terms, sublinear_tf=True)
    tfidf = vec.fit_transform(bunch.data).toarray().astype(np.float32)
    print(f"  TF-IDF: {tfidf.shape}")

    # Restricting to 1000 terms leaves some documents with an all-zero vector.
    # They are not harmless: TfidfVectorizer L2-normalises, so every real
    # document sits at distance exactly 1.0 from every empty one -- closer than
    # its genuine neighbours, which are further apart than that. The k nearest
    # neighbours of a typical document then become k empty documents at
    # identical distance. Measured on this corpus, keeping them makes the 1st
    # and 20th neighbour exactly equidistant for 86.8% of points; dropping them
    # takes that to 0.0%. That flattening destroys the core-distance variation
    # the whole density-based construction depends on, so they go.
    keep = np.abs(tfidf).sum(axis=1) > 0
    dropped = int((~keep).sum())
    if dropped:
        print(f"  dropping {dropped} documents containing none of the top-{args.terms} "
              f"terms ({100.0 * dropped / len(keep):.1f}%)")
        tfidf = tfidf[keep]
    target = bunch.target[keep]
    write_fvecs(os.path.join(args.out_dir, "20newsgroups_tfidf1000.fvecs"), tfidf)

    # "...applied PCA to reduce to the 500 principal components".
    comps = min(args.components, min(tfidf.shape) - 1)
    pca = PCA(n_components=comps, random_state=0)
    reduced = pca.fit_transform(tfidf).astype(np.float32)
    var = float(pca.explained_variance_ratio_.sum())
    print(f"  PCA: {reduced.shape}, explains {var:.1%} of the variance "
          f"(the paper reports ~75%)")
    write_fvecs(
        os.path.join(args.out_dir, f"20newsgroups_pca{comps}.fvecs"), reduced
    )

    np.savetxt(
        os.path.join(args.out_dir, "20newsgroups_labels.txt"),
        target, fmt="%d",
    )
    print("  wrote labels (for reference only -- quality claims compare against "
          "the exact CORE-SG, not these)")


def cmd_fma(args):
    """FMA audio features, the two variants the CORE-SG paper used.

    features.csv carries a three-level column header (feature, statistic,
    number). Selecting the top level reproduces the paper's dimensionalities
    exactly: chroma_cens is 12 bins x 7 statistics = 84, mfcc is 20
    coefficients x 7 statistics = 140.
    """
    try:
        import pandas as pd
    except ImportError:
        raise SystemExit("needs pandas:  pip install pandas")
    import zipfile

    print(f"reading {args.zip} (features.csv is ~900 MB uncompressed; this takes a minute)")
    with zipfile.ZipFile(args.zip) as z:
        with z.open("fma_metadata/features.csv") as f:
            df = pd.read_csv(f, index_col=0, header=[0, 1, 2])

    for feature, expected in (("chroma_cens", 84), ("mfcc", 140)):
        if feature not in df.columns.get_level_values(0):
            print(f"  {feature}: not present, skipped")
            continue
        sub = df[feature]
        arr = sub.to_numpy(dtype=np.float32)
        # A handful of tracks have missing features; they would otherwise
        # become NaN rows and poison every distance they take part in.
        finite = np.isfinite(arr).all(axis=1)
        dropped = int((~finite).sum())
        if dropped:
            print(f"  {feature}: dropping {dropped} rows with non-finite values")
            arr = arr[finite]
        if arr.shape[1] != expected:
            print(f"  {feature}: got {arr.shape[1]} dims, the paper reports {expected}")
        write_fvecs(os.path.join(args.out_dir, f"fma_{feature}.fvecs"), arr)


def cmd_hdf5(args):
    try:
        import h5py
    except ImportError:
        raise SystemExit("needs h5py:  pip install h5py")
    with h5py.File(args.path, "r") as f:
        if args.key not in f:
            raise SystemExit(f"key '{args.key}' not in {args.path}; keys: {list(f.keys())}")
        data = np.asarray(f[args.key], dtype=np.float32)
    if args.limit:
        data = data[: args.limit]
    out = args.out or os.path.splitext(args.path)[0] + ".fvecs"
    write_fvecs(out, data)


def cmd_synthetic(args):
    rng = np.random.default_rng(args.seed)

    # Data is generated in `gen_dim` dimensions and then, if an embedding is
    # requested, rotated into `dim` by an orthonormal map. A rotation preserves
    # all pairwise distances exactly, so the cluster structure and separation
    # are untouched while the *ambient* dimensionality rises and the intrinsic
    # one stays put. That gives a data set with known ground-truth intrinsic
    # dimensionality -- the only way to check both the estimator and the budget
    # that depends on it against something other than a guess.
    gen_dim = args.intrinsic if args.intrinsic else args.dim
    if gen_dim > args.dim:
        raise SystemExit("--intrinsic cannot exceed --dim")

    centres = rng.normal(0.0, args.separation, size=(args.clusters, gen_dim))
    # Densities spanning ~10x: the regime where one mpts cannot describe every
    # region, which is the whole reason CORE-SG exists.
    sigma = 0.15 + 1.35 * rng.random(args.clusters)
    assign = rng.integers(0, args.clusters, size=args.n)

    basis = None
    if gen_dim < args.dim:
        q, _ = np.linalg.qr(rng.normal(size=(args.dim, gen_dim)))
        basis = q.T.astype(np.float32)  # (gen_dim, dim), orthonormal rows
        print(f"  embedding {gen_dim}-d data into {args.dim}-d by random rotation "
              f"(distances preserved; intrinsic dimensionality stays {gen_dim})")

    out = np.empty((args.n, args.dim), dtype=np.float32)
    chunk = 200_000
    for lo in range(0, args.n, chunk):
        hi = min(lo + chunk, args.n)
        a = assign[lo:hi]
        block = centres[a] + sigma[a][:, None] * rng.normal(size=(hi - lo, gen_dim))
        out[lo:hi] = (block @ basis if basis is not None else block).astype(np.float32)
    write_fvecs(args.out, out)
    np.savetxt(args.out.replace(".fvecs", "_labels.txt"), assign, fmt="%d")


def cmd_inspect(args):
    arr = read_fvecs(args.path, args.limit)
    print(f"{args.path}: {arr.shape[0]} x {arr.shape[1]} float32")
    print(f"  min {arr.min():.4f}  max {arr.max():.4f}  mean {arr.mean():.4f}")
    norms = np.linalg.norm(arr[: min(len(arr), 10000)], axis=1)
    print(f"  row norm: mean {norms.mean():.4f}  std {norms.std():.4f}")
    if np.allclose(norms, 1.0, atol=1e-3):
        print("  rows look L2-normalised -- use --metric cosine")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("newsgroups", help="20 Newsgroups via scikit-learn")
    p.add_argument("--out-dir", default="data")
    p.add_argument("--terms", type=int, default=1000)
    p.add_argument("--components", type=int, default=500)
    p.add_argument("--subset", default="train", choices=["train", "test", "all"],
                   help="train (default) reproduces the CORE-SG paper's 11,314 points")
    p.set_defaults(func=cmd_newsgroups)

    p = sub.add_parser("fma", help="FMA chroma_cens (84d) and mfcc (140d)")
    p.add_argument("zip", help="path to fma_metadata.zip")
    p.add_argument("--out-dir", default="data")
    p.set_defaults(func=cmd_fma)

    p = sub.add_parser("hdf5", help="ANN-benchmarks .hdf5 -> .fvecs")
    p.add_argument("path")
    p.add_argument("--key", default="train")
    p.add_argument("--out")
    p.add_argument("--limit", type=int, default=0)
    p.set_defaults(func=cmd_hdf5)

    p = sub.add_parser("synthetic", help="Gaussian blobs of varying density")
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--dim", type=int, default=32, help="ambient dimensionality")
    p.add_argument("--intrinsic", type=int, default=0,
                   help="generate in this many dims then rotate into --dim "
                        "(0 = no embedding, intrinsic == ambient)")
    p.add_argument("--clusters", type=int, default=25)
    p.add_argument("--separation", type=float, default=8.0)
    p.add_argument("--seed", type=int, default=20260729)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_synthetic)

    p = sub.add_parser("inspect", help="describe a .fvecs file")
    p.add_argument("path")
    p.add_argument("--limit", type=int, default=0)
    p.set_defaults(func=cmd_inspect)

    args = ap.parse_args()
    if getattr(args, "out_dir", None):
        os.makedirs(args.out_dir, exist_ok=True)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
