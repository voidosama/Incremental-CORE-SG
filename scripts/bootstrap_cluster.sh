#!/usr/bin/env bash
# One-shot cluster setup: build, fetch data, verify. Steps 1-4 of CLUSTER.md.
#
#   bash scripts/bootstrap_cluster.sh
#   bash scripts/bootstrap_cluster.sh --arch 75 --skip-data
#
# Idempotent -- re-running skips what is already done. It deliberately stops at
# the point where the actual experiments would start, so you can inspect the
# preflight output before committing the cluster time.

set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

ARCH=75          # RTX 2080 SUPER. sm_70 is a hard floor (warp-level spin locks).
SKIP_DATA=0
JOBS="$(nproc 2>/dev/null || echo 4)"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --arch) ARCH="$2"; shift 2 ;;
        --skip-data) SKIP_DATA=1; shift ;;
        --jobs) JOBS="$2"; shift 2 ;;
        *) echo "unknown option $1"; exit 2 ;;
    esac
done

step() { echo; echo "=== $* ==="; }
die()  { echo "FAILED: $*" >&2; exit 1; }

step "environment"
command -v nvcc >/dev/null || echo "  ! nvcc not on PATH -- module load cuda/... first"
command -v cmake >/dev/null || die "cmake not found (conda activate coresg?)"
python -c "import numpy, sklearn" 2>/dev/null || die "python deps missing (conda activate coresg?)"
nvcc --version 2>/dev/null | tail -1
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv 2>/dev/null \
    || echo "  ! nvidia-smi unavailable"

step "build (arch sm_$ARCH)"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
    $(command -v ninja >/dev/null && echo "-G Ninja") || die "cmake configure"
cmake --build build -j "$JOBS" || die "build"

step "tests"
ctest --test-dir build --output-on-failure || die "tests"

if [[ $SKIP_DATA -eq 0 ]]; then
    step "data (~3.5 GB download, ~14 GB after preparation)"
    bash scripts/fetch_data.sh || die "data preparation"
fi

step "suite inventory"
python scripts/run_suite.py --list | tail -3

step "preflight"
BIN="build/bin/coresg_cli"
[[ -x "$BIN" ]] || BIN="build/bin/Release/coresg_cli"
"$BIN" preflight --n 50000 --dim 32 --kmax 24
PRE=$?

echo
if [[ $PRE -eq 0 ]]; then
    cat <<'EOF'
Bootstrap complete. To run the suite:

    python scripts/run_suite.py --seeds 5 --out results.csv --resume --keep-going 2>&1 | tee -a run.log

Under SLURM, use the sbatch template in CLUSTER.md instead of running on a login node.
EOF
else
    echo "PREFLIGHT FAILED -- fix this before starting a long run (see CLUSTER.md §4)."
    exit 1
fi
