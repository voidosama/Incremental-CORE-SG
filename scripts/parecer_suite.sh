#!/usr/bin/env bash
# The runs the banca asked for, in one unattended pass.
#
# Every item here answers a numbered point in the parecer of Oct 2026. The
# order is by execution cost, ascending, with one exception: the CPU-only
# build runs first because it is the one whose answer most changes what the
# text may claim, and it is also the longest, so starting it first lets it
# overlap with everything else being written.
#
#   bash scripts/parecer_suite.sh           # everything, in order
#   bash scripts/parecer_suite.sh cpu       # one named block only
#   OUT=results_parecer.csv bash scripts/parecer_suite.sh
#
# Appends to results_parecer.csv and skips any (experiment,label) pair already
# present, so a kill and a resubmit resume rather than repeat.
#
# Blocks, with the parecer item each one closes:
#   cpu       3.1  CPU-only build; how much do the GPUs actually buy
#   premerge  3.1  recall after merge, before repair, over m and omega
#   budget    3.4  m=1 with the iteration budget m>1 gets for free
#   fosc      3.7  ARI under FOSC/EOM instead of a single DBSCAN* cut
#   cuml      3.3  cuML with its own partitioning turned on
#   seeds     3.8  the construction grid, repeated over seeds

set -u -o pipefail
cd "$(dirname "$0")/.."

OUT=${OUT:-results_parecer.csv}
LOG=${LOG:-parecer_suite.log}
TARGET=${1:-all}
SEEDS=${SEEDS:-"42 43 44"}

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
echo "=== parecer_suite started $(date -Is), target=$TARGET, out=$OUT ==="
[ -n "$CLI" ] || { echo "FATAL: coresg_cli not built. cmake --build build -j"; exit 1; }
echo "cli=$CLI  exp=${EXP:-<none>}"

want() { [ "$TARGET" = all ] || [ "$TARGET" = "$1" ]; }
have() { [ -f "$OUT" ] && grep -q "^construction,$1," "$OUT"; }

# run <label> <config> <value> <args...>
run() {
    local label=$1 cfg=$2 val=$3; shift 3
    if [ -f "$OUT" ] && grep -q "^construction,${label},.*,${cfg},${val}," "$OUT"; then
        echo "-- $label $cfg=$val present, skipping"; return 0
    fi
    echo; echo "-- $label $cfg=$val  ($(date -Is))"
    "$CLI" build --csv "$OUT" --name "$label" --config "$cfg" --config-value "$val" "$@" \
        || echo "!! $label $cfg=$val FAILED (continuing)"
}

SIFT=data/sift/sift_base.fvecs
GLOVE=data/glove100.fvecs

echo; echo "### PREFLIGHT"
nvidia-smi || true
"$CLI" preflight --n 50000 --dim 32 --kmax 24 || echo "!! preflight failed (continuing; CPU blocks still valid)"

# ===========================================================================
# 1. CPU ONLY  -- parecer 3.1, defence question 2
#
# The claim under test is that the GPUs matter. The honest way to show it is
# the same code path with the devices switched off, on the same data. --gpu 0
# sends every instance to the host backend; everything else is unchanged, so
# the ratio isolates the devices and nothing else.
#
# Cost: the decomposed 50M build is ~1090 s, of which ~863 s is already CPU.
# The instance phase (123 s across 4 devices) is what moves to the host, so
# expect hours, not minutes. It is first so it can run overnight.
# ===========================================================================
if want cpu; then
echo; echo "############ 1. CPU-ONLY BUILDS ############"
for N in 10000000 50000000; do
    F="data/scale${N}.fvecs"
    if [ ! -f "$F" ]; then
        echo "-- generating $F"
        "$PY" scripts/prepare_datasets.py synthetic --n "$N" --dim 32 --intrinsic 8 \
            --clusters 200 --separation 6 --out "$F" \
            || { echo "!! generation failed at $N, skipping"; continue; }
    fi
    INST=$(( N / 500000 ))
    # Same instance count as the GPU run it is compared against, so the only
    # difference between the two rows is which backend ran the instances.
    run "cpuonly_scale${N}" backend cpu \
        --data "$F" --kmax 32 --instances "$INST" --overlap 0.05 \
        --gpu 0 --recall-sample 2000
done
# The same pair on real data, where the GPU numbers already exist.
[ -f "$SIFT" ] && run "cpuonly_sift1m" backend cpu \
    --data "$SIFT" --limit 500000 --kmax 32 --instances 8 --overlap 0.05 \
    --gpu 0 --recall-sample 2000
fi

# ===========================================================================
# 2. RECALL BEFORE THE REPAIR  -- parecer 3.1, defence question 1
#
# "Qual e a revocacao do grafo logo apos a mesclagem, antes do reparo?"
# --refine-iters 0 leaves the repair loop with zero rounds, so the recorded
# recall is exactly the state the devices produced. Cheaper than a normal
# build, because the phase being skipped is the dominant one.
# ===========================================================================
if want premerge; then
echo; echo "############ 2. RECALL BEFORE REPAIR ############"
for SPEC in "sift1m:$SIFT:l2" "glove100:$GLOVE:cosine"; do
    NAME=${SPEC%%:*}; REST=${SPEC#*:}; FILE=${REST%%:*}; MET=${REST#*:}
    [ -f "$FILE" ] || { echo "-- $FILE missing, skipping $NAME"; continue; }
    for W in 0.05 0.5 1.0; do
        for I in 1 2 4 8 16 32 64; do
            run "premerge_${NAME}_o${W}" instances "$I" \
                --data "$FILE" --limit 500000 --kmax 32 --metric "$MET" \
                --instances "$I" --overlap "$W" --refine-iters 0 \
                --gpu 1 --recall-sample 2000
        done
    done
done
fi

# ===========================================================================
# 3. EQUAL ITERATION BUDGET  -- parecer 3.4
#
# m=1 runs no repair, so the m=1 vs m=2 recall rise confounds decomposition
# with iteration budget. A partitioned build gets --iters rounds inside the
# instance plus --refine-iters rounds of repair; m=1 gets only the first.
# Matching the total is one flag: --iters 15 against the default 10 + 5.
# The sweep around it gives the curve rather than a single point.
# ===========================================================================
if want budget; then
echo; echo "############ 3. m=1 AT MATCHED BUDGET ############"
for SPEC in "sift1m:$SIFT:l2" "glove100:$GLOVE:cosine"; do
    NAME=${SPEC%%:*}; REST=${SPEC#*:}; FILE=${REST%%:*}; MET=${REST#*:}
    [ -f "$FILE" ] || { echo "-- $FILE missing, skipping $NAME"; continue; }
    for IT in 10 12 15 20 25; do
        run "budget_${NAME}_m1" iters "$IT" \
            --data "$FILE" --limit 500000 --kmax 32 --metric "$MET" \
            --instances 1 --overlap 0.05 --iters "$IT" \
            --gpu 1 --recall-sample 2000
    done
    # The partitioned arms it is being compared against, same seed, same data.
    for I in 2 8; do
        run "budget_${NAME}_m${I}" iters 10 \
            --data "$FILE" --limit 500000 --kmax 32 --metric "$MET" \
            --instances "$I" --overlap 0.05 --iters 10 --refine-iters 5 \
            --gpu 1 --recall-sample 2000
    done
done
fi

# ===========================================================================
# 4. FOSC / EOM AGREEMENT  -- parecer 3.7
#
# The thesis says the compared clustering is FOSC's, but the ARI it reports
# comes from a single DBSCAN* cut at the mean edge weight. There is no FOSC
# in the C++; the selection lives in the hdbscan package, and it only needs
# an MST. So: extract the approximate and the exact tree with `extract`, then
# label both through condense_tree + compute_stability + EOM in Python.
#
# n is small because the exact arm is the O(n^2 d) reference, same as Tab. 2.
# ===========================================================================
if want fosc; then
echo; echo "############ 4. ARI UNDER FOSC ############"
mkdir -p data/mst_fosc
for DS in manifold_c32 manifold_c24 manifold_exact manifold_c8; do
    F="data/${DS}.fvecs"
    [ -f "$F" ] || { echo "-- $F missing, skipping $DS"; continue; }
    for S in $SEEDS; do
        A="data/mst_fosc/${DS}_seed${S}_antihub"
        E="data/mst_fosc/${DS}_seed${S}_exact"
        [ -f "${A}_mpts24.tsv" ] || "$CLI" extract --data "$F" --kmax 24 --limit 4000 \
            --missing antihub --seed "$S" --mpts 8,16,24 --out-prefix "$A" \
            || echo "!! extract antihub $DS seed $S FAILED"
        [ -f "${E}_mpts24.tsv" ] || "$CLI" extract --data "$F" --kmax 24 --limit 4000 \
            --missing exact --seed "$S" --mpts 8,16,24 --out-prefix "$E" \
            || echo "!! extract exact $DS seed $S FAILED"
    done
done
"$PY" scripts/fosc_ari.py --mst-dir data/mst_fosc --out "$OUT" --mpts 8,16,24 \
    || echo "!! fosc_ari.py FAILED (continuing)"
fi

# ===========================================================================
# 5. cuML WITH ITS OWN PARTITIONING  -- parecer 3.3
#
# scripts/compare_cuml.py builds HDBSCAN(min_cluster_size, min_samples) and
# nothing else, which is the brute-force unpartitioned path. The parecer is
# right that this is not the comparable configuration. knn_n_clusters and
# knn_overlap_factor live inside build_kwds, and knn_overlap_factor must be
# strictly less than knn_n_clusters, so the smallest partitioned setting is
# 3 clusters with overlap 2.
#
# Note (1+omega)/m against overlap/n_clusters: at omega=1 and m=C the two
# residency formulas coincide, which is the comparison worth reporting.
# ===========================================================================
if want cuml; then
echo; echo "############ 5. cuML, PARTITIONED ############"
"$PY" scripts/cuml_partitioned.py --out "$OUT" --data "$SIFT" \
    --n 500000,1000000,2000000,4000000,8000000 --mpts 10 \
    --clusters 1,4,8,16 --overlap 2 --seeds $SEEDS \
    || echo "!! cuml_partitioned.py FAILED (continuing)"
fi

# ===========================================================================
# 6. THE CONSTRUCTION GRID, OVER SEEDS  -- parecer 3.8
#
# Every row behind the memory, overlap, scale and scaling tables carries
# seed=0. The text claims multiple seeds. This repeats the grid so the
# tables can carry dispersion, or so the claim can be corrected with
# evidence. Last because it is the most expensive and the least surprising.
# ===========================================================================
if want seeds; then
echo; echo "############ 6. CONSTRUCTION OVER SEEDS ############"
for S in $SEEDS; do
    for SPEC in "sift1m:$SIFT:l2" "glove100:$GLOVE:cosine"; do
        NAME=${SPEC%%:*}; REST=${SPEC#*:}; FILE=${REST%%:*}; MET=${REST#*:}
        [ -f "$FILE" ] || continue
        for W in 0.05 1.0; do
            for I in 1 2 4 8 16 32 64; do
                run "seeded_${NAME}_o${W}_s${S}" instances "$I" \
                    --data "$FILE" --limit 500000 --kmax 32 --metric "$MET" \
                    --instances "$I" --overlap "$W" --seed "$S" \
                    --gpu 1 --recall-sample 2000
            done
        done
    done
    # The scale table too, at the two ends of its range.
    for N in 10000000 50000000; do
        F="data/scale${N}.fvecs"
        [ -f "$F" ] || continue
        run "seeded_scale${N}_s${S}" instances "$(( N / 500000 ))" \
            --data "$F" --kmax 32 --instances "$(( N / 500000 ))" \
            --overlap 0.05 --seed "$S" --gpu 1 --recall-sample 2000
    done
done
fi

echo; echo "=== parecer_suite finished $(date -Is) ==="
echo "rows now in $OUT: $( [ -f "$OUT" ] && wc -l < "$OUT" || echo 0 )"
