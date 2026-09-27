#!/usr/bin/env bash
# Everything the paper needs, in the order the paper needs it.
#
# Why re-run at all: the reverse-list fill was made parallel (commit 478514f),
# which took the hubness phase from 8.63 s to 1.50 s at n = 4e6 and should take
# a 50M build from roughly 1342 s to 1068 s. Every build timing and every
# parallel_fraction recorded before that commit is stale. Quality numbers are
# not -- the change is a reordering of work, not of results -- but they are
# cheap to reproduce alongside the timings, so they are here too.
#
# Ordered by what the paper claims, not by what is quick. A walltime kill after
# TIER 1 still leaves the contribution measured; a kill after TIER 2 leaves it
# measured and defended. TIER 3 is the material that moved to future work when
# the scope narrowed to construction, kept because it is the evidence for that
# section.
#
#   bash scripts/final_suite.sh          # everything, in order
#   bash scripts/final_suite.sh 1        # tier 1 only
#   bash scripts/final_suite.sh 2        # tiers 1-2
#   bash scripts/final_suite.sh overlap  # the 1.4 sweep alone, ~25 min on 4 GPUs
#
# Appends to results_final.csv and skips any (experiment, dataset) pair already
# there, so re-running after an interruption resumes rather than repeats.
# Under SLURM, wrap as CLUSTER.md section 5 shows and resubmit on timeout.

set -u -o pipefail

cd "$(dirname "$0")/.."
# Overridable so a sweep added after a full run can be appended straight into
# the file that run produced, instead of landing in a second file that then has
# to be merged by hand:  OUT=results_final_v4.csv bash scripts/final_suite.sh extract
OUT=${OUT:-results_final.csv}
LOG=${LOG:-final_suite.log}
TIER=${1:-3}
# A numeric argument selects tiers 1..N. A word selects one named block and
# nothing else, so a sweep added after a full run can be fired on its own.
case "$TIER" in
    ""|*[!0-9]*) TARGET="$TIER"; TIER=0 ;;
    *) TARGET=all ;;
esac

pick() {
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
echo "=== final_suite started $(date -Is), tiers 1..$TIER, out=$OUT ==="
[ -n "$EXP" ] || { echo "FATAL: coresg_exp not built. cmake --build build -j"; exit 1; }
[ -n "$CLI" ] || { echo "FATAL: coresg_cli not built. cmake --build build -j"; exit 1; }
echo "exp=$EXP  cli=$CLI"

have() { [ -f "$OUT" ] && grep -q "^$1,$2," "$OUT"; }

# run_exp <experiment> <label> <args...>
run_exp() {
    local e=$1 label=$2; shift 2
    if have "$e" "$label"; then echo "-- $e/$label present, skipping"; return 0; fi
    echo; echo "-- $e/$label"
    "$EXP" --experiment "$e" --name "$label" --out "$OUT" "$@" \
        || echo "!! $e/$label FAILED (continuing)"
}

# run_build <label> <config> <value> <args...>
run_build() {
    local label=$1 cfg=$2 val=$3; shift 3
    if [ -f "$OUT" ] && grep -q "^construction,${label},.*,${cfg},${val}," "$OUT"; then
        echo "-- construction/$label $cfg=$val present, skipping"; return 0
    fi
    echo; echo "-- construction/$label $cfg=$val"
    "$CLI" build --csv "$OUT" --name "$label" --config "$cfg" --config-value "$val" "$@" \
        || echo "!! construction/$label $cfg=$val FAILED (continuing)"
}

echo; echo "### PREFLIGHT"
nvidia-smi || true
"$CLI" preflight --n 50000 --dim 32 --kmax 24 || {
    echo "FATAL: preflight failed. Rows produced now would say GPU and mean nothing."
    exit 1; }

# ===========================================================================
# TIER 1 -- the contribution
#
# Three claims, in the order the abstract makes them: per-device memory does
# not grow with the data, the work spreads across devices, and the whole thing
# reaches sizes a single-device tool cannot.
# ===========================================================================
echo; echo "############ TIER 1 ############"

NDEV=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || true)
[ -z "$NDEV" ] && NDEV=0
[ "$NDEV" -lt 1 ] && NDEV=1
echo; echo "visible devices: $NDEV"

if [ "$TIER" -ge 1 ]; then
    # --- 1.1 the scale ceiling -------------------------------------------------
    # The headline. Per-device residency stayed at 293 MB from 10M to 50M because
    # the instance count rises with n; that is the claim in one table. Generation
    # is minutes and gigabytes, so the resume check comes before it.
    for N in 10000000 20000000 30000000 40000000 50000000; do
        if have construction "scale${N}"; then
            echo "-- scale${N} present, skipping"; continue
        fi
        F="data/scale${N}.fvecs"
        if [ ! -f "$F" ]; then
            echo "-- generating $F (~$(( N * 132 / 1000000000 )) GB)"
            "$PY" scripts/prepare_datasets.py synthetic --n "$N" --dim 32 --intrinsic 8 \
                --clusters 200 --separation 6 --out "$F" \
                || { echo "!! generation failed at $N, stopping the sweep here"; break; }
        fi
        INST=$(( N / 500000 ))
        [ "$INST" -lt 4 ] && INST=4
        # Recall as well as seconds. Without it the claim is that a 50M graph can
        # be built, with no statement that the result is any good -- the first
        # thing a reader will ask. Sampled, so it costs O(q*n*d) instead of the
        # O(n^2 d) an exact reference needs: a few minutes at 50M against a
        # build of roughly eighteen.
        run_build "scale${N}" instances "$INST" --data "$F" --kmax 32 --instances "$INST" --gpu 1 --recall-sample 2000
    done

    # --- 1.2 instances against devices -----------------------------------------
    # The mechanism. largest_instance_bytes against instance count is the memory
    # claim; instance_seconds against device count is the scaling claim. Both
    # datasets, every device count the node has.

    for SPEC in "sift1m:data/sift/sift_base.fvecs:l2" "glove100:data/glove100.fvecs:cosine"; do
        NAME=${SPEC%%:*}; REST=${SPEC#*:}; FILE=${REST%%:*}; MET=${REST#*:}
        [ -f "$FILE" ] || { echo "-- $FILE missing, skipping $NAME"; continue; }
        for G in $(seq 1 "$NDEV"); do
            DEVS=$(seq -s, 0 $((G - 1)))
            for I in 1 2 4 8 16 32 64; do
                run_build "build_${NAME}_g${G}" instances "$I" --data "$FILE" --limit 500000 \
                    --kmax 32 --metric "$MET" --instances "$I" --gpu 1 --devices "$DEVS" \
                    --recall-sample 2000
            done
        done
    done

    # --- 1.3 where a single instance stops fitting -----------------------------
    # The claim is that decomposition makes graphs buildable that otherwise would
    # not fit on a card. Every run so far scaled the instance count with n, so
    # nothing ever failed and the claim rested on arithmetic. Here the instance
    # count is pinned low and n is raised until a device refuses, with the host
    # fallback switched off so the refusal is loud instead of silent.
    # The first version stopped at 10M because that is how large capacity10m is,
    # and nothing failed: at one instance and d=32 a device needs 560 B per
    # point, so 10M is 5.6 GB and an 8 GB card takes it comfortably. The probe
    # ran out of data, not out of memory. The synthetic sets generated in 1.1
    # reach 50M, which is past the ceiling for one or two instances.
    PROBE=data/scale50000000.fvecs
    [ -f "$PROBE" ] || PROBE=data/capacity10m.fvecs
    PROBE_MAX=50000000
    [ "$PROBE" = data/capacity10m.fvecs ] && PROBE_MAX=10000000
    echo; echo "-- oom probe data: $PROBE (up to $PROBE_MAX points)"
    for I in 1 2 4; do
        for N in 10000000 15000000 20000000 25000000 30000000 40000000 50000000; do
            LABEL="oomprobe_i${I}"
            if [ -f "$OUT" ] && grep -q "^construction,${LABEL},${N},.*,instances,${I}," "$OUT"; then
                echo "-- $LABEL n=$N present, skipping"; continue
            fi
            if [ "$N" -gt "$PROBE_MAX" ]; then
                echo "!! instances=$I never failed within $PROBE -- the ceiling is above"
                echo "   $PROBE_MAX points; generate a larger set to find it"
                break
            fi
            echo; echo "-- oom probe: instances=$I n=$N"
            if "$CLI" build --data "$PROBE" --limit "$N" --kmax 32 \
                    --instances "$I" --gpu 1 --oom-fallback 0 \
                    --csv "$OUT" --name "$LABEL" --config instances --config-value "$I"; then
                echo "-- instances=$I n=$N fits"
            else
                echo "!! instances=$I n=$N does NOT fit -- this is the number for the paper"
                break
            fi
        done
    done

    # ===========================================================================
    # TIER 2 -- what defends it
    # ===========================================================================
fi

# --- 1.4 overlap against instance count ------------------------------------
# Instances currently share 5 percent of a block, drawn uniformly. Raising that
# moves work out of the repair and into the instance phase: at n = 5e5 with
# eight instances on one GPU, overlap 1.0 takes the repair from 3.94 s to
# 2.72 s and the device-parallel fraction from 0.306 to 0.534. The instance
# phase grows by more than the repair shrinks, so on one device the total gets
# worse; the question this sweep answers is what happens when that phase is
# divided across four.
#
# The second question is the frontier. Doubling the instance count alongside
# the overlap lands at the same per-device residency with higher recall --
# 28.9 MB at 0.9673 against 30.9 MB at 0.9600, and 14.6 MB at 0.9638 against
# 15.5 MB at 0.9498, both measured on one GPU. Those pairs are what decides
# whether overlap is an option or a better default.
if [ "$TIER" -ge 1 ] || [ "$TARGET" = overlap ]; then
    echo; echo "### 1.4 overlap sweep"
    for G in $(seq 1 "$NDEV"); do
        DEVS=$(seq -s, 0 $((G - 1)))
        for O in 0.05 0.5 1.0; do
            # The full instance range, so each overlap yields a complete
            # memory-against-recall frontier rather than its lower half. At
            # m=1 there is no partition and therefore no repair, so the row is
            # identical across overlaps; it is measured under each anyway so
            # that a single overlap's rows stand on their own.
            for I in 1 2 4 8 16 32 64; do
                run_build "ovl_sift1m_g${G}_o${O}" instances "$I" \
                    --data data/sift/sift_base.fvecs --limit 500000 --kmax 32 \
                    --instances "$I" --overlap "$O" --gpu 1 --devices "$DEVS" \
                    --recall-sample 2000
            done
        done
        # A second data set, at the two overlaps the frontier compares, so the
        # result is not a property of sift1m alone. glove100 gains the most
        # from overlap: +0.044 recall at less memory on four devices.
        for O in 0.05 1.0; do
            for I in 1 2 4 8 16 32 64; do
                run_build "ovl_glove100_g${G}_o${O}" instances "$I" \
                    --data data/glove100.fvecs --limit 500000 --kmax 32 --metric cosine \
                    --instances "$I" --overlap "$O" --gpu 1 --devices "$DEVS" \
                    --recall-sample 2000
            done
        done
    done
    echo
    echo "1.4 done. dataset=ovl_<set>_g<devices>_o<overlap>, config=instances."
    echo "  Compare at matched largest_instance_bytes across the o0.05 and o1.0"
    echo "  rows, on recall and on total_seconds."
fi

# --- 2.0 what one mpts query costs -----------------------------------------
# Sits above the tier gate so it can be fired on its own, the same way 1.4 is.
#
# Section V-H claims a CORE-SG serves every mpts below kmax and charges the
# alternative one full fit per value. The extraction half of that was the one
# number in the paper with no row behind it in the current results file: the
# old figure came from the streaming experiment, which is out of scope now.
#
# `mpts` now times each extraction separately instead of only the sweep total,
# so what is reported is the cost of answering one query. At n small enough for
# an O(n^2 d) spanning tree it also times the complete-graph alternative, which
# is the only size where the comparison is measured rather than extrapolated
# from nothing.
if [ "$TIER" -ge 2 ] || [ "$TARGET" = extract ]; then
    echo; echo "### 2.0 MST extraction cost"

    # The one size where the naive alternative is affordable. Three exact
    # spanning trees over 20,000 points at d=128 is the bulk of this run.
    run_exp mpts sift1m_extract_n20000 --data data/sift/sift_base.fvecs --limit 20000 \
        --k 32 --seeds 2 --gpu 1 --devices 0

    # The sizes Table 8 uses, on one device, so the extraction and the build it
    # is charged against come off the same card.
    for N in 200000 500000 1000000; do
        run_exp mpts "sift1m_extract_n${N}" --data data/sift/sift_base.fvecs --limit "$N" \
            --k 32 --seeds 3 --gpu 1 --devices 0
    done

    # A second data set and a non-Euclidean metric.
    run_exp mpts glove100_extract_n500000 --data data/glove100.fvecs --limit 500000 \
        --k 32 --metric cosine --seeds 3 --gpu 1 --devices 0

    # And one at the scale the memory claim reaches, where the stored edge list
    # is a few hundred million entries. One seed and every device: the build is
    # minutes and the quantity under test is the query, not the build.
    if [ -f data/scale10000000.fvecs ]; then
        run_exp mpts scale10000000_extract --data data/scale10000000.fvecs \
            --k 32 --seeds 1 --gpu 1 --instances 20
    fi

    echo
    echo "2.0 done. experiment=mpts_sweep, dataset=<set>_extract_n<N>."
    echo "  config=extract_mpts, one row per mpts: seconds, mst_edges, components."
    echo "  config=phase: build seconds and knn_edges, first_extraction,"
    echo "  coresg_sweep, sweep_over_build, and at n=20000 the complete-graph"
    echo "  speedup. Quote the per-query range from extract_mpts, excluding"
    echo "  mpts=1, which pays the first touch of the arrays."
fi

[ "$TIER" -ge 2 ] || { echo; echo "=== stopping after tier 1 ==="; exit 0; }
echo; echo "############ TIER 2 ############"

# --- 2.1 against cuML, build against build ---------------------------------
# The existing comparison scores our *extension* against cuML's *build*, which
# is a fair claim about a different question. Under a construction scope the
# like-for-like number is build against build, and the two curves currently
# overlap at one point only. This puts cuML where our builds already are.
if [ -f scripts/compare_cuml.py ]; then
    for N in 500000 1000000; do
        if have capacity_cuml "sift1m_build_n${N}"; then
            echo "-- cuML n=$N present, skipping"; continue
        fi
        echo; echo "-- cuML build at n=$N"
        "$PY" scripts/compare_cuml.py --mode capacity --data data/sift/sift_base.fvecs \
            --name "sift1m_build_n${N}" --k 32 --mpts 10 --capacity-sizes "$N" --out "$OUT" \
            || echo "!! cuML n=$N FAILED (continuing)"
    done
fi

# --- 2.2 the two backends agree --------------------------------------------
run_exp parity sift1m --data data/sift/sift_base.fvecs --limit 50000 --k 32 --seeds 3
run_exp parity manifold_mid --data data/manifold_mid.fvecs --k 24 --seeds 3

# --- 2.3 the repair cannot be cheapened ------------------------------------
# A negative result, and one worth having: it is why the build's parallel
# fraction stays where it is. Recall against rounds, nothing else.
for R in 1 2 3 5; do
    run_build refine_sift1m refine_iters "$R" --data data/sift/sift_base.fvecs \
        --limit 500000 --kmax 32 --instances 8 --gpu 1 --refine-iters "$R" \
        --recall-sample 2000
done

# --- 2.4 the bridges ------------------------------------------------------
# Quality, not timing, so the hubness change does not touch the result. Here
# because the anti-hub substitution is half the method and the table should
# come from the same binary as everything else.
run_exp bridges manifold_exact --data data/manifold_exact.fvecs --k 16 --seeds 3
run_exp bridges manifold_c24 --data data/manifold_c24.fvecs --k 24 --seeds 3
run_exp connectivity sift1m --data data/sift/sift_base.fvecs --limit 100000 --k 32 \
    --k-sweep 4,8,16,24,32 --seeds 2
run_exp connectivity manifold_mid --data data/manifold_mid.fvecs --k 24 \
    --k-sweep 4,8,16,24 --seeds 2

# ===========================================================================
# TIER 3 -- the future-work section
#
# Everything here is about extending a graph rather than building one, which
# is the part the scope narrowing moved out of the contribution. It is not
# dead: it is the evidence that the closing section is an opening rather than
# a guess. Run it last because nothing in the paper's argument waits on it.
# ===========================================================================
[ "$TIER" -ge 3 ] || { echo; echo "=== stopping after tier 2 ==="; exit 0; }
echo; echo "############ TIER 3 ############"

for N in 200000 500000 1000000; do
    run_exp incremental "sift1m_c${N}" --data data/sift/sift_base.fvecs --limit "$N" \
        --k 32 --seeds 2 --batches 8 --gpu 1
done
for KSPEC in "2:0.5" "4:0.25" "8:0.125" "16:0.0625" "32:0.03125"; do
    K=${KSPEC%%:*}; BF=${KSPEC#*:}
    run_exp incremental "sift1m_k${K}" --data data/sift/sift_base.fvecs --limit 500000 \
        --k 32 --base-fraction "$BF" --batches $((K - 1)) --seeds 2 --gpu 1
done
for EF in 24 32 48 64; do
    run_exp incremental "sift1m_ef${EF}_n100000" --data data/sift/sift_base.fvecs \
        --limit 100000 --k 32 --search-ef "$EF" --seeds 2 --batches 8 --gpu 1
done

echo; echo "=== final_suite finished $(date -Is) ==="
echo "Everything is in $OUT. The three numbers the contribution rests on:"
echo "  construction / scale<N>          largest_instance_bytes, flat against n"
echo "  construction / build_<set>_g<D>  instance_seconds against D, memory against instances"
echo "  construction / oomprobe_i<I>     the largest n that fits without decomposition"
