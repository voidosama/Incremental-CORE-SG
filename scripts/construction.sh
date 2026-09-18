#!/usr/bin/env bash
# Evidence for the construction claim: decomposed, multi-instance CORE-SG
# construction of a fixed data set.
#
# Two things are measured here that no existing sweep records.
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
#   bash scripts/construction.sh            # both stages
#   bash scripts/construction.sh confirm    # stage 1 only, ~30-60 min on 4 GPUs
#   bash scripts/construction.sh scale      # stage 2 only, ~30 min + generation
#
# Stage 1 appends to results_construction.csv and is safe to re-run: a
# (experiment, dataset) pair already in the CSV is skipped. Stage 2 is
# log-only; its numbers are the phase line coresg_cli prints.
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
        if "$CLI" build --data "$F" --kmax 32 --instances "$INST" --gpu 1; then
            echo "-- n=$N OK"
        else
            echo "!! n=$N FAILED -- this is the ceiling; report the last size that passed"
            break
        fi
    done

    echo
    echo "Stage 2 done. Read the '[coresg] build' line for each n:"
    echo "  knn / hubness / missing / repair split, and the instance share"
    echo "The largest n that completed is the number for the paper, together"
    echo "with the per-device instance size that made it fit."
fi

echo; echo "=== construction finished $(date -Is) ==="
