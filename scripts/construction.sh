#!/usr/bin/env bash
# Evidence for the construction claim: decomposed, multi-instance CORE-SG
# construction of a fixed data set.
#
# Four things are measured here that no existing sweep records.
#
# STAGE 1 (confirm) -- chunked against monolithic construction, on GPU, with a
# phase breakdown of BOTH arms. The comparison itself already exists inside
# `exp_incremental` (build_seconds, cumulative_seconds, rebuild_seconds), and
# results_v2.csv already holds it at n <= 200k. What it did not record is where
# a *build's* wall clock goes, so the two arms could not be compared on the one
# axis the construction claim rests on: how much of each is device parallel.
# `emit_build_phases` now writes that for the initial build and the rebuild, so
# a `*_parallel_fraction` exists for both arms and the Amdahl argument stops
# being an estimate.
#
# Run it at sizes the existing rows do not cover. Below n = 200k the O(n^2)
# recall reference switches on and dominates the run; above it the experiment
# skips the reference by itself, which is what makes 1M affordable.
#
# STAGE 2 (scale) -- how far the decomposition goes on this hardware. A pure
# monolithic build at increasing n, reading the ceiling off the machine rather
# than off an extrapolation. It uses coresg_cli rather than coresg_exp on
# purpose: `exp_incremental` keeps two CoreSG objects resident at once (the
# chunked one and the rebuild), which roughly doubles peak host memory and
# would hit the 126 GB limit well before the algorithm does.
#
# STAGE 3 (divide) -- the decomposition itself, on its own two axes: instances
# within one build, and pieces of the data set. This is the stage the scope
# actually rests on; stages 1 and 2 inherited their shape from the streaming
# question and answer it only sideways. Its instance axis runs on one GPU, so
# only the device axis needs the cluster.
#
# STAGE 3c (knobs) -- search_ef and refine-iters swept *downward*. Both
# defaults were set for the streaming path; the phase breakdown puts both on
# the dominant cost of a build, and neither has ever been tested below its
# default. Local, one GPU, no cluster needed.
#
#   bash scripts/construction.sh            # all three stages
#   bash scripts/construction.sh confirm    # stage 1 only, ~30-60 min on 4 GPUs
#   bash scripts/construction.sh scale      # stage 2 only, ~30 min + generation
#   bash scripts/construction.sh divide     # stage 3 only, ~30 min on one GPU
#   bash scripts/construction.sh knobs      # stage 3c only, ~40 min on one GPU
#
# Every stage appends to results_construction.csv and every stage is safe to
# re-run: stages 1 and 3b skip a (experiment, dataset) pair already in the CSV,
# stage 2 skips a size already recorded, stage 3a skips a (label, instances)
# pair. Everything is also echoed to construction.log.
#
# Under SLURM, wrap exactly as CLUSTER.md section 5 shows and resubmit on
# timeout.

set -u -o pipefail

cd "$(dirname "$0")/.."
OUT=results_construction.csv
LOG=construction.log
PREP=scripts/prepare_datasets.py
WHAT=${1:-all}

# The cluster builds to build/bin; a Windows/MSVC tree builds to
# build/bin/Release with a .exe suffix. Probing beats hardcoding either, so the
# same script runs in both places and a local single-GPU confirmation run does
# not need the cluster queue.
pick() {  # pick <basename>
    local b=$1
    for c in "./build/bin/$b" "./build/bin/Release/$b.exe" "./build/bin/$b.exe" "./build/Release/$b.exe"; do
        [ -x "$c" ] && { echo "$c"; return 0; }
    done
    return 1
}
EXP=$(pick coresg_exp) || EXP=""
CLI=$(pick coresg_cli) || CLI=""
PY=python
command -v python >/dev/null 2>&1 || PY=python3

exec > >(tee -a "$LOG") 2>&1
echo "=== construction started $(date -Is), target=$WHAT, out=$OUT ==="

[ -n "$EXP" ] || { echo "FATAL: coresg_exp not found. cmake --build build -j"; exit 1; }
[ -n "$CLI" ] || { echo "FATAL: coresg_cli not found. cmake --build build -j"; exit 1; }
echo "exp=$EXP  cli=$CLI"

done_already() {  # done_already <experiment> <dataset>
    [ -f "$OUT" ] && grep -q "^$1,$2," "$OUT"
}

run_exp() {  # run_exp <label> <args...>
    local label=$1; shift
    if done_already incremental "$label"; then
        echo "-- incremental/$label already present, skipping"; return 0
    fi
    echo "-- incremental/$label"
    "$EXP" --experiment incremental --name "$label" --out "$OUT" "$@" \
        || echo "!! incremental/$label FAILED (continuing)"
}

# ---------------------------------------------------------------------------
# STAGE 0 -- refuse to produce rows labelled GPU that quietly ran on the host
# ---------------------------------------------------------------------------
if [ "$WHAT" = all ] || [ "$WHAT" = confirm ]; then
    echo; echo "### STAGE 0: preflight"
    nvidia-smi || true
    "$CLI" preflight --n 50000 --dim 32 --kmax 24 || {
        echo "FATAL: preflight failed. Rows produced now would say '4 GPUs' and"
        echo "mean nothing. Fix the device path before continuing."; exit 1; }
fi

# ---------------------------------------------------------------------------
# STAGE 1 -- chunked vs monolithic construction, with both phase breakdowns
#
# Three sizes per data set so the ratio can be read as a trend rather than a
# point. gist1m stops at 500k: at d=960 the experiment holds three copies of
# the vectors (the gathered order, the chunked CoreSG, the rebuilt one), which
# is 11.5 GB of vectors alone at 1M before either graph is counted.
# ---------------------------------------------------------------------------
if [ "$WHAT" = all ] || [ "$WHAT" = confirm ]; then
    echo; echo "### STAGE 1: chunked vs monolithic construction (GPU)"

    for N in 200000 500000 1000000; do
        run_exp "sift1m_c${N}" --data data/sift/sift_base.fvecs --limit "$N" \
            --k 32 --seeds 2 --batches 8 --gpu 1
    done

    for N in 200000 500000 1000000; do
        run_exp "glove100_c${N}" --data data/glove100.fvecs --limit "$N" \
            --k 32 --metric cosine --seeds 2 --batches 8 --gpu 1
    done

    for N in 200000 500000; do
        run_exp "gist1m_c${N}" --data data/gist/gist_base.fvecs --limit "$N" \
            --k 32 --seeds 2 --batches 8 --gpu 1
    done

    echo
    echo "Stage 1 done. The four numbers the paper needs, per row:"
    echo "  incremental_total_seconds / rebuild_seconds   -> the decomposition cost"
    echo "  build_parallel_fraction                       -> f for a monolithic build"
    echo "  rebuild_parallel_fraction                     -> same, at final n"
    echo "  build_largest_instance_bytes                  -> per-device residency"
    echo "Compare rebuild_parallel_fraction against the scaling experiment's"
    echo "parallel_fraction for an update. The gap is the argument."
fi

# ---------------------------------------------------------------------------
# STAGE 2 -- how far the decomposition goes on this hardware
#
# 126 GB of host RAM and four 8 GB cards (CLUSTER.md). The 10M capacity run
# peaked at 17.18 GB, so the host ceiling lands somewhere near 60M at d=32 and
# nothing near a billion points is reachable here -- the arithmetic is in the
# echo below so nobody has to rediscover it.
#
# Instance count is raised with n on purpose. That is the decomposition claim
# in operational form: a build that will not fit as four instances fits as
# sixty-four, on the same cards, because an instance never has to hold the
# whole data set.
# ---------------------------------------------------------------------------
if [ "$WHAT" = all ] || [ "$WHAT" = scale ]; then
    echo; echo "### STAGE 2: scale ceiling"
    echo "host budget: 126 GB. Measured 1718 B/point at d=32 (capacity10m, 10M)."
    echo "  10M -> ~17 GB    25M -> ~43 GB    50M -> ~86 GB    1B -> ~1.7 TB"
    echo "1B is out by more than an order of magnitude on this node."
    echo

    for N in 10000000 20000000 30000000 40000000 50000000; do
        # Resume check first: generation is minutes and gigabytes, so a
        # re-run must not pay for a size it already recorded.
        if done_already construction "scale${N}"; then
            echo "-- scale${N} already present, skipping"; continue
        fi

        F="data/scale${N}.fvecs"
        if [ ! -f "$F" ]; then
            echo "-- generating $F ($(( N * 132 / 1000000000 )) GB on disk)"
            "$PY" "$PREP" synthetic --n "$N" --dim 32 --intrinsic 8 \
                --clusters 200 --separation 6 --out "$F" \
                || { echo "!! generation failed at N=$N, stopping stage 2"; break; }
        fi

        # Instances scale with n so no single instance outgrows an 8 GB card.
        INST=$(( N / 1000000 * 2 ))
        [ "$INST" -lt 4 ] && INST=4

        echo "-- build n=$N, instances=$INST"
        if "$CLI" build --data "$F" --kmax 32 --instances "$INST" --gpu 1 --csv "$OUT" --name "scale${N}"; then
            echo "-- n=$N OK"
        else
            echo "!! n=$N FAILED -- this is the ceiling; report the last size that passed"
            break
        fi
    done

    echo
    echo "Stage 2 done. Rows land in $OUT under experiment=construction,"
    echo "dataset=scale<N>. The numbers for the paper:"
    echo "  largest_instance_bytes  -> per-device residency at that n"
    echo "  jobs_on_device / jobs_out_of_memory -> that it really ran on GPUs"
    echo "  peak_host_bytes         -> where the 126 GB ceiling actually bites"
    echo "  parallel_fraction       -> f for a monolithic build at scale"
    echo "The largest n that completed is the ceiling on this hardware."
fi

# ---------------------------------------------------------------------------
# STAGE 3 -- the decomposition itself
#
# Stages 1 and 2 inherited their shape from the streaming question: build on
# half the data, then add batches, and ask whether each addition beats a
# rebuild. That is the wrong axis for a construction claim. Decomposing a build
# is not a speed optimisation and never was -- k pieces pay the fixed per-update
# costs k times, so a chunked construction costs more wall clock than one
# monolithic build, by a measured 2.2 times at k=9. What it buys is that no
# single piece has to fit anywhere. This stage sweeps the two knobs that make
# that trade, instead of the batch index.
#
# 3a -- instances within one build. The primary axis: a monolithic build split
#   into 1, 2, ... 64 self-contained NN-Descent instances, queued onto whatever
#   devices exist. The claim is that largest_instance_bytes falls roughly as
#   1/instances while total time does not rise much, which is what lets a build
#   that will not fit as one instance fit as sixty-four on the same card.
#
#   The instance axis needs no second GPU: with one device the instances queue
#   onto one slot, and the memory result is the same result. The device axis is
#   what needs the cluster, so the sweep sizes itself to whatever nvidia-smi
#   reports and the same invocation does what the hardware allows.
#
# 3b -- pieces of the data set. The orthogonal axis, and the one stage 1 got
#   wrong by fixing base_fraction at 0.5. Building in k pieces means building
#   on n/k and adding k-1 more of the same size, which the existing flags
#   already express exactly. k=1 costs nothing extra to measure: rebuild_seconds
#   is already the monolithic build at the same final n, emitted by every run.
# ---------------------------------------------------------------------------
if [ "$WHAT" = all ] || [ "$WHAT" = divide ]; then
    echo; echo "### STAGE 3: divide and conquer"

    NDEV=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || true)
    [ -z "$NDEV" ] && NDEV=0
    [ "$NDEV" -lt 1 ] && NDEV=1
    echo "visible devices: $NDEV (instance axis runs on any count; device axis needs the cluster)"

    # ---- 3a: instances within one monolithic build ------------------------
    # A row already present is skipped on (label, instances), not on label
    # alone, because one label covers the whole instance sweep.
    for SPEC in "sift1m:data/sift/sift_base.fvecs:l2" "glove100:data/glove100.fvecs:cosine"; do
        DSNAME=${SPEC%%:*}
        REST=${SPEC#*:}
        DSFILE=${REST%%:*}
        DSMETRIC=${REST#*:}
        [ -f "$DSFILE" ] || { echo "-- $DSFILE missing, skipping $DSNAME"; continue; }

        for G in $(seq 1 "$NDEV"); do
            DEVLIST=$(seq -s, 0 $((G - 1)))
            for I in 1 2 4 8 16 32 64; do
                LABEL="build_${DSNAME}_g${G}"
                if [ -f "$OUT" ] && grep -q "^construction,${LABEL},.*,instances,${I}," "$OUT"; then
                    echo "-- $LABEL instances=$I already present, skipping"
                    continue
                fi
                echo "-- $LABEL instances=$I devices=$DEVLIST"
                "$CLI" build --data "$DSFILE" --limit 500000 --kmax 32 --metric "$DSMETRIC" --instances "$I" --gpu 1 --devices "$DEVLIST" --csv "$OUT" --name "$LABEL" || echo "!! $LABEL i=$I FAILED (continuing)"
            done
        done
    done

    # ---- 3b: k pieces of the data set -------------------------------------
    # base_fraction = 1/k and batches = k-1 gives k equal pieces. Each run also
    # emits rebuild_seconds, which is k=1 on the same data and seed, so every
    # row carries its own baseline.
    for KSPEC in "2:0.5" "4:0.25" "8:0.125" "16:0.0625" "32:0.03125"; do
        K=${KSPEC%%:*}
        BF=${KSPEC#*:}
        run_exp "sift1m_k${K}" --data data/sift/sift_base.fvecs --limit 500000 --k 32 --base-fraction "$BF" --batches $((K - 1)) --seeds 2 --gpu 1
    done

    echo
    echo "Stage 3 done."
    echo "  3a: experiment=construction, dataset=build_<set>_g<devices>, config=instances"
    echo "      plot largest_instance_bytes and total_seconds against instances."
    echo "      jobs_out_of_memory > 0 marks where one instance stopped fitting."
    echo "  3b: experiment=incremental, dataset=sift1m_k<k>"
    echo "      cumulative_seconds is construction in k pieces,"
    echo "      rebuild_seconds is the same data set built at once (k=1)."
fi

# ---------------------------------------------------------------------------
# STAGE 3c -- the two knobs that were never swept downward
#
# Both defaults were chosen for the streaming path and never questioned for a
# build, and the phase breakdown says both sit on the dominant cost.
#
# search_ef defaults to 2k = 64. The existing sensitivity sweep went 64, 128,
# 256, 512 -- upward only, asking whether a wider beam buys recall. Nobody
# asked what a narrower one costs, and the seeding search is 43 % of a chunk
# at n = 1M, roughly linear in ef. If 32 holds recall it halves the largest
# single cost in chunked construction.
#
# refine-iters defaults to 5. After the instance blocks merge, a monolithic
# build runs an unrestricted NN-Descent over every node on the host: 7.1 s of a
# 13.8 s build at n = 1M, 51 %, and the reason the build's parallel fraction is
# 0.298 while an update's is 0.93. What that pass repairs is cross-block edges,
# which is a bounded problem being solved with an unbounded tool. Capping the
# rounds is the one-flag version of bounding it; if recall survives 2 rounds
# the build gets materially cheaper and materially more parallel.
#
# Recall is sampled, not exhaustive -- an O(n^2) reference does not exist at
# 500k. Every arm shares one query set (the seed is fixed), so the arms are
# comparable to each other even though each carries sampling error.
# ---------------------------------------------------------------------------
if [ "$WHAT" = all ] || [ "$WHAT" = knobs ]; then
    echo; echo "### STAGE 3c: search_ef and refine-iters, swept downward"

    # ---- ef: cheaper seeding, and what it costs ---------------------------
    # n = 100,000 is deliberate and it is the whole point of this arm. A
    # narrower beam is unconditionally faster, so the only number that decides
    # anything is what it does to recall -- and exp_incremental computes recall
    # only where it can afford the exhaustive reference, `d.n <= 200000`. Run
    # this at 500k and it silently produces timings with no quality column,
    # which is an argument for whatever is cheapest.
    #
    # The search cost per point depends on ef, k and d, none of which move with
    # n, so the recall-against-ef curve measured here is the one that applies
    # at scale. Absolute seconds are not; read those from the 500k rows.
    for EF in 24 32 48 64; do
        run_exp "sift1m_ef${EF}_n100000" --data data/sift/sift_base.fvecs --limit 100000 --k 32 --search-ef "$EF" --seeds 2 --batches 8 --gpu 1
    done

    # ---- refine rounds: cheaper merge repair? ------------------------------
    for R in 1 2 3 5; do
        LABEL="refine_sift1m"
        if [ -f "$OUT" ] && grep -q "^construction,${LABEL},.*,refine_iters,${R}," "$OUT"; then
            echo "-- $LABEL refine-iters=$R already present, skipping"
        else
            echo "-- $LABEL refine-iters=$R"
            "$CLI" build --data data/sift/sift_base.fvecs --limit 500000 --kmax 32 --instances 8 --gpu 1 --refine-iters "$R" --recall-sample 2000 --csv "$OUT" --name "$LABEL" --config refine_iters --config-value "$R" || echo "!! $LABEL r=$R FAILED (continuing)"
        fi
    done

    echo
    echo "Stage 3c done."
    echo "  ef:     experiment=incremental, dataset=sift1m_ef<ef>_n100000"
    echo "          plot recall_new against ef; that is the axis that decides."
    echo "          search cost is knn_seconds - instance - merge - refine."
    echo "  refine: experiment=construction, dataset=refine_sift1m, config=refine_iters"
    echo "          plot total_seconds, refine_seconds, parallel_fraction, recall."
    echo "          r=5 is today's default and the baseline for the others."
fi

echo; echo "=== construction finished $(date -Is) ==="
