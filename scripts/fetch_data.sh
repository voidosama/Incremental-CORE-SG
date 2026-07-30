#!/usr/bin/env bash
# Fetch and prepare every public data set, directly on whatever machine runs it.
#
# This exists so the ~11.5 GB of data never has to be copied from a laptop over
# a flaky SSH link. All of it is public and reproducible: the archives come from
# their original hosts, the derived .fvecs files are built locally, and the
# synthetic families are generated. The code repository stays at ~500 KB.
#
#   bash scripts/fetch_data.sh              # everything
#   bash scripts/fetch_data.sh sift glove   # only these
#
# Safe to re-run: downloads resume (curl -C -), and anything already present at
# the right size is skipped. Roughly 3.5 GB downloaded, ~14 GB after extraction
# and preparation.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="$ROOT/data"
mkdir -p "$DATA"
cd "$DATA" || exit 1

WANT="${*:-sift gist glove fma newsgroups synthetic}"
fail=0

want() { [[ " $WANT " == *" $1 "* ]]; }

# fetch <url> <file> <expected-bytes>
fetch() {
    local url="$1" file="$2" expect="$3"
    if [[ -f "$file" ]]; then
        local have
        have=$(stat -c%s "$file" 2>/dev/null || stat -f%z "$file" 2>/dev/null || echo 0)
        if [[ "$have" == "$expect" ]]; then
            echo "  $file: already complete ($(numfmt --to=iec "$have" 2>/dev/null || echo "$have B"))"
            return 0
        fi
        echo "  $file: resuming (have $have of $expect)"
    fi
    echo "  fetching $file ..."
    if ! curl -L -C - --retry 5 --retry-delay 5 --progress-bar -o "$file" "$url"; then
        echo "  !! download failed: $file"
        return 1
    fi
    local got
    got=$(stat -c%s "$file" 2>/dev/null || stat -f%z "$file" 2>/dev/null || echo 0)
    if [[ -n "$expect" && "$got" != "$expect" ]]; then
        echo "  !! size mismatch for $file: got $got, expected $expect"
        return 1
    fi
}

py() { python "$ROOT/scripts/prepare_datasets.py" "$@"; }

echo "=== data directory: $DATA ==="

if want sift; then
    echo "--- SIFT1M (1M x 128) ---"
    fetch "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz" sift.tar.gz 168280445 \
        && { [[ -f sift/sift_base.fvecs ]] || tar -xzf sift.tar.gz; } || fail=1
fi

if want gist; then
    echo "--- GIST1M (1M x 960; the ImageNet substitute, ~2.6 GB) ---"
    fetch "ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz" gist.tar.gz 2740172684 \
        && { [[ -f gist/gist_base.fvecs ]] || tar -xzf gist.tar.gz; } || fail=1
fi

if want glove; then
    echo "--- GloVe-100 (angular; use --metric cosine) ---"
    fetch "http://ann-benchmarks.com/glove-100-angular.hdf5" glove-100-angular.hdf5 485413888 \
        && { [[ -f glove100.fvecs ]] || py hdf5 "$DATA/glove-100-angular.hdf5" --key train \
             --out "$DATA/glove100.fvecs"; } || fail=1
fi

if want fma; then
    echo "--- FMA (chroma_cens 84d, mfcc 140d) ---"
    fetch "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip" fma_metadata.zip 358412441 \
        && { [[ -f fma_mfcc.fvecs ]] || py fma "$DATA/fma_metadata.zip" --out-dir "$DATA"; } \
        || fail=1
fi

if want newsgroups; then
    echo "--- 20 Newsgroups (scikit-learn fetches it; no manual download) ---"
    if [[ -f 20newsgroups_pca500.fvecs ]]; then
        echo "  already prepared"
    else
        py newsgroups --out-dir "$DATA" || fail=1
    fi
fi

if want synthetic; then
    echo "--- synthetic families (manifold + 10M capacity set) ---"
    python "$ROOT/scripts/run_suite.py" --generate || fail=1
fi

echo
echo "=== inventory ==="
find "$DATA" -name '*.fvecs' -size +1M -printf '  %-46p %8.1f MB\n' \
    2>/dev/null | sed "s|$DATA/||" || ls -lh "$DATA"/*.fvecs 2>/dev/null

echo
if [[ $fail -eq 0 ]]; then
    echo "all requested data ready. Next:  python scripts/run_suite.py --list"
else
    echo "SOME STEPS FAILED -- re-run this script; completed downloads are skipped."
    exit 1
fi
