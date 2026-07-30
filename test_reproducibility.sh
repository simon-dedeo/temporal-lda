#!/bin/bash
set -euo pipefail

BIN=${1:-./temporal_lda_omp}
THREADS=${OMP_TEST_THREADS:-4}
REPEATS=${OMP_TEST_REPEATS:-3}
ITERS=${OMP_TEST_ITERATIONS:-120}
TMP=$(mktemp -d "${TMPDIR:-/tmp}/temporal-lda-repro.XXXXXX")
trap 'rm -rf "$TMP"' EXIT

for run in $(seq 1 "$REPEATS"); do
    out="$TMP/run_$run"
    mkdir -p "$out"
    OMP_NUM_THREADS=$THREADS "$BIN" \
        --docs test_data/documents.txt \
        --vocab test_data/vocab.txt \
        --metadata test_data/metadata.txt \
        --output "$out" \
        --K 6 --iterations "$ITERS" --seed 42 --sigma 10 \
        --optimize-interval 50 --converge 1e-12 \
        --local-alpha "$out/alpha.txt" > "$out/run.log" 2>&1
done

for run in $(seq 2 "$REPEATS"); do
    for name in beta.txt theta.txt weights.txt alpha.txt; do
        if ! cmp -s "$TMP/run_1/$name" "$TMP/run_$run/$name"; then
            echo "FAIL: run 1 and run $run differ in $name" >&2
            exit 1
        fi
    done
done

echo "PASS: $REPEATS runs are byte-identical at seed=42 threads=$THREADS iterations=$ITERS"
