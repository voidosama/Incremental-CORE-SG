#!/usr/bin/env bash
# Everything that has to be (re)measured after the anchored-insert change.
#
# Ordered so the two runs that can kill the story come first: if `strategy`
# does not show a flat update and `incremental` does not hold ARI, nothing
# after them is worth the cluster time. Read STAGE 1 and STAGE 2 before letting
# the rest run.
#
# Appends to results_v2.csv, NOT results.csv. The old file stays intact -- you
# need both to show what changed, and `run_suite.py --resume` would otherwise
# see the old rows and skip almost everything.
#
# Safe to re-run: every stage is skipped if its rows are already present.
# Resume after a walltime kill by running the same command again.
#
#   bash scripts/rerun_v2.sh              # everything
#   bash scripts/rerun_v2.sh decisive     # stages 1-2 only, ~2 hours
#   bash scripts/rerun_v2.sh baselines    # FISHDBC + cuML only
#   bash scripts/rerun_v2.sh gaps         # the three holes the v2 sweep left
#
# Under SLURM, wrap it exactly as CLUSTER.md §5 shows and resubmit on timeout.

set -u -o pipefail

cd "$(dirname "$0")/.."
OUT=results_v2.csv
LOG=rerun_v2.log
EXP=./build/bin/coresg_exp
WHAT=${1:-all}

exec > >(tee -a "$LOG") 2>&1
echo "=== rerun_v2 started $(date -Is), target=$WHAT, out=$OUT ==="

[ -x "$EXP" ] || { echo "FATAL: $EXP not built. cmake --build build -j"; exit 1; }

# Skip a direct coresg_exp run whose rows are already in the CSV. run_suite.py
# does this itself via --resume; these one-off invocations need their own guard.
done_already() {  # done_already <experiment> <dataset>
    [ -f "$OUT" ] && grep -q "^$1,$2," "$OUT"
}

run_exp() {  # run_exp <experiment> <dataset-name> <args...>
    local e=$1 n=$2; shift 2
    if done_already "$e" "$n"; then echo "-- $e/$n already present, skipping"; return 0; fi
    echo "-- $e/$n"
    "$EXP" --experiment "$e" --name "$n" --out "$OUT" "$@" || echo "!! $e/$n FAILED (continuing)"
}

# ---------------------------------------------------------------------------
# STAGE 0 -- the checks that fail silently rather than loudly
# ---------------------------------------------------------------------------
if [ "$WHAT" = all ] || [ "$WHAT" = decisive ]; then
    echo; echo "### STAGE 0: preflight"
    nvidia-smi || true
    ./build/bin/coresg_cli preflight --n 50000 --dim 32 --kmax 24 || {
        echo "FATAL: preflight failed. A sweep started now would produce rows"
        echo "labelled '4 GPUs' that silently ran on the host."; exit 1; }
fi

# ---------------------------------------------------------------------------
# STAGE 1 -- does an update cost O(batch)?  THE headline
# ---------------------------------------------------------------------------
if [ "$WHAT" = all ] || [ "$WHAT" = decisive ]; then
    echo; echo "### STAGE 1: strategy A/B (GPU + CPU)"
    python scripts/run_suite.py --only strategy --out "$OUT" --resume --keep-going

    # The slope figure. One `strategy` run only varies n by 2x, which is not
    # enough to show a trend; these hold the batch at 12,500 and vary n by 10x.
    # batches = limit/25000 because per_batch = 0.5*limit/batches.
    echo; echo "### STAGE 1b: fixed-batch slope across n (sift1m)"
    for L in 100000 200000 400000 1000000; do
        run_exp strategy "sift1m_n$L" --data data/sift/sift_base.fvecs \
            --k 32 --limit "$L" --batches $((L / 25000)) --seeds 3
    done
fi

# ---------------------------------------------------------------------------
# STAGE 2 -- did the quality claims survive?  The other one that can kill it
# ---------------------------------------------------------------------------
if [ "$WHAT" = all ] || [ "$WHAT" = decisive ]; then
    echo; echo "### STAGE 2: incremental (ARI + MST weight ratio vs rebuild)"
    python scripts/run_suite.py --only incremental --out "$OUT" --resume --keep-going
    echo
    echo ">>> STOP AND READ before continuing."
    echo ">>> STAGE 1: instance_nodes flat in n, rebuild_speedup climbing?"
    echo ">>> STAGE 2: ari_vs_rebuild and mst_weight_ratio_vs_rebuild still near 1?"
    echo ">>> If either is no, the rest of the sweep is not worth the time."
fi
[ "$WHAT" = decisive ] && { echo "=== decisive stages done $(date -Is) ==="; exit 0; }

# ---------------------------------------------------------------------------
# STAGE 3 -- the rest of the invalidated experiments
# ---------------------------------------------------------------------------
if [ "$WHAT" = all ]; then
    echo; echo "### STAGE 3: arrival, batchsize, drift, sensitivity, parity"
    # arrival is the adversarial case for the seeding search: grouped arrival
    # means the base graph covers only part of the space early on.
    # sensitivity now sweeps search_ef, which is the answer to "is the recall
    # difference a fixed loss or a dial".
    python scripts/run_suite.py --only arrival,batchsize,drift,sensitivity,parity \
        --out "$OUT" --resume --keep-going
fi

# ---------------------------------------------------------------------------
# STAGE 4 -- multi-GPU.  Expect this to come down; it is not a regression
# ---------------------------------------------------------------------------
if [ "$WHAT" = all ]; then
    echo; echo "### STAGE 4: scaling (instances are batch-sized now, so f will drop)"
    python scripts/run_suite.py --only scaling --families manifold_mid,sift1m,gist1m \
        --out "$OUT" --resume --keep-going
fi

# ---------------------------------------------------------------------------
# STAGE 5 -- external baselines.  The competitive claim
# ---------------------------------------------------------------------------
if [ "$WHAT" = all ] || [ "$WHAT" = baselines ]; then
    echo; echo "### STAGE 5a: FISHDBC"
    # Easy to miss: it drives --experiment stream, so it goes through
    # base_params and is now measuring the anchored path.
    if python -c "import flexible_clustering" 2>/dev/null; then
        for f in manifold_mid fma_mfcc sift1m; do
            case $f in
                manifold_mid) D=data/manifold_mid.fvecs; K=24;;
                fma_mfcc)     D=data/fma_mfcc.fvecs;     K=32;;
                sift1m)       D=data/sift/sift_base.fvecs; K=32;;
            esac
            python scripts/compare_incremental.py --data "$D" --name "$f" --k "$K" \
                --limit 20000 --batches 8 --mpts 10 --out "$OUT" \
                || echo "!! FISHDBC/$f FAILED (continuing)"
        done
    else
        echo "-- flexible_clustering not installed, skipping FISHDBC."
        echo "   pip install --no-build-isolation git+https://github.com/matteodellamico/flexible-clustering"
    fi

    echo; echo "### STAGE 5b: NVIDIA cuML HDBSCAN"
    if python -c "import cuml" 2>/dev/null; then
        # Three sizes: the slope is the claim, one n proves nothing.
        for L in 100000 200000 400000; do
            python scripts/compare_cuml.py --data data/sift/sift_base.fvecs \
                --name "sift1m_n$L" --limit "$L" --k 32 --batches $((L / 25000)) \
                --mpts 10 --mpts-sweep 4,8,16,24,32 --out "$OUT" \
                || echo "!! cuML/sift1m_n$L FAILED (continuing)"
        done
        # A second corpus so it is not a single-dataset claim.
        python scripts/compare_cuml.py --data data/fma_mfcc.fvecs --name fma_mfcc \
            --k 32 --batches 8 --mpts 10 --mpts-sweep 4,8,16,24,32 --out "$OUT" \
            || echo "!! cuML/fma_mfcc FAILED (continuing)"

        # Where cuML runs out of device memory. The OOM is recorded as a row.
        echo; echo "### STAGE 5c: cuML capacity"
        python scripts/compare_cuml.py --mode capacity --data data/capacity10m.fvecs \
            --name capacity10m --k 32 --mpts 10 \
            --capacity-sizes 1000000,2000000,4000000,8000000,10000000 --out "$OUT" \
            || echo "!! cuML capacity FAILED (continuing)"
        # The CORE-SG side at the same sizes: per-device footprint is the
        # comparison, not speed.
        run_exp scaling capacity10m_cuml --data data/capacity10m.fvecs --k 32 \
            --scaling-instances 1,4,8,16 --no-oom-fallback --skip-cpu-scaling --seeds 1
    else
        echo "-- cuML not installed, skipping the GPU baseline. This is the"
        echo "   competitive claim, so do not ship without it:"
        echo "   conda install -c rapidsai -c conda-forge -c nvidia cuml python=3.12 cuda-version=12.*"
    fi
fi

# ---------------------------------------------------------------------------
# STAGE 6 -- the three holes the first v2 sweep left
# ---------------------------------------------------------------------------
if [ "$WHAT" = gaps ]; then
    # 6a. Is the recall deficit a fixed loss or a dial?
    #
    # The v2 sweep put three data sets below the 0.85 floor -- gist1m 0.770,
    # glove100 0.749, 20ng_pca500 0.792 -- and the `sensitivity` sweep could not
    # have caught it: it inserts the whole second half as ONE batch, where recall
    # saturates at 0.998, and it does not include any of these three. The loss is
    # cumulative over batches, so it has to be measured on a multi-batch run.
    #
    # search_ef drives recall on the *new* points. It is the binding constraint
    # where recall_new is itself low: gist1m 0.918, glove100 0.809.
    echo; echo "### STAGE 6a: search_ef on the data sets where the search struggles"
    for E in 64 128 256 512; do
        run_exp strategy "gist1m_ef$E"      --data data/gist/gist_base.fvecs \
            --k 32 --limit 100000 --batches 8 --seeds 2 --search-ef "$E"
        run_exp strategy "glove100_ef$E"    --data data/glove100.fvecs --metric cosine \
            --k 32 --limit 200000 --batches 8 --seeds 2 --search-ef "$E"
        run_exp strategy "20ng_pca500_ef$E" --data data/20newsgroups_pca500.fvecs \
            --k 24 --batches 8 --seeds 2 --search-ef "$E"
    done

    # 6b. The other half of the mechanism. recall_new stays flat across batches
    # while recall_all decays, so points inserted early are going stale: on
    # sift1m recall_new held at 0.982 while recall_all fell 0.947 -> 0.915.
    # search_ef cannot fix that -- the refinement rings are what revisit old
    # points, and refine_per_new is their budget. It is per *new* point, so
    # raising it keeps the update O(batch).
    echo; echo "### STAGE 6b: refine_per_new -- does repairing old points recover recall_all?"
    for RP in 2 4 8 16; do
        run_exp strategy "sift1m_rp$RP" --data data/sift/sift_base.fvecs \
            --k 32 --limit 200000 --batches 8 --seeds 2 --refine-per-new "$RP"
        run_exp strategy "gist1m_rp$RP" --data data/gist/gist_base.fvecs \
            --k 32 --limit 100000 --batches 8 --seeds 2 --refine-per-new "$RP"
    done

    # 6c. The cuML comparison has no agreement column: --reference-limit
    # defaults to 50000 and every batch of the first run was above it, because
    # the base alone is half the file. A speed claim with no ARI next to it is
    # the first thing a reviewer will reject. Keep n under the limit so the
    # exact HDBSCAN* reference actually gets computed, and use one seed --
    # sklearn's HDBSCAN on 40k points is the slow part, not cuML.
    echo; echo "### STAGE 6c: cuML ARI agreement at a size the exact reference can reach"
    if python -c "import cuml" 2>/dev/null; then
        for f in fma_mfcc manifold_mid; do
            case $f in
                fma_mfcc)     D=data/fma_mfcc.fvecs;    K=32;;
                manifold_mid) D=data/manifold_mid.fvecs; K=24;;
            esac
            python scripts/compare_cuml.py --data "$D" --name "${f}_ari" --k "$K" \
                --limit 40000 --batches 8 --mpts 10 --seeds 1 \
                --reference-limit 50000 --out "$OUT" \
                || echo "!! cuML ARI/$f FAILED (continuing)"
        done
    else
        echo "-- cuML not installed, skipping."
    fi

    # 6d. cuML's device footprint came back as 0.0: the probe watches cupy's
    # memory pool and cuML allocates through RMM, which it never sees. Sampling
    # nvidia-smi alongside the run is the reliable way to get the number.
    echo
    echo ">>> For cuML's peak device memory, run this in a second shell during 6c:"
    echo ">>>   nvidia-smi --query-gpu=memory.used --format=csv --loop-ms=200 > cuml_mem.csv"
fi

echo; echo "=== rerun_v2 finished $(date -Is) ==="
echo "Rows: $(wc -l < "$OUT" 2>/dev/null || echo 0) in $OUT"
echo
echo "Headline tables:"
echo "  python scripts/summarize.py --results $OUT --only strategy"
echo "  python scripts/summarize.py --results $OUT --only incremental,cuml"
