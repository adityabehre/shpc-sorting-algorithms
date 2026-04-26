#!/usr/bin/env bash
# run_large_bench.sh — Benchmark only the 5M and 10M element files.
# Usage: bash run_large_bench.sh
set -e

source venv/bin/activate

TMPDIR_LARGE=$(mktemp -d)
# Create a temporary dataset dir containing only the large files
mkdir -p "$TMPDIR_LARGE/nearly_sorted" "$TMPDIR_LARGE/ordered" \
         "$TMPDIR_LARGE/repeated_values" "$TMPDIR_LARGE/reverse_ordered" \
         "$TMPDIR_LARGE/same_value" "$TMPDIR_LARGE/uniform"

for f in $(find benchmark-dataset-for-sorting-algorithms -name "*-input-5000000-*" -o -name "*-input-10000000-*"); do
    type=$(basename "$(dirname "$f")")
    cp "$f" "$TMPDIR_LARGE/$type/"
done

echo "Running benchmark on large files in $TMPDIR_LARGE ..."
python3 benchmark.py \
  --dataset_dir "$TMPDIR_LARGE" \
  --output large_only_results.csv \
  --runs 3 \
  --warmup 1

rm -rf "$TMPDIR_LARGE"
echo "Done. Merging into benchmark_results.csv ..."

python3 -c "
import pandas as pd
base = pd.read_csv('benchmark_results.csv')
new  = pd.read_csv('large_only_results.csv')
print(f'Base rows: {len(base)}   New rows: {len(new)}')
combined = pd.concat([base, new], ignore_index=True)
combined.to_csv('benchmark_results.csv', index=False)
print(f'Combined: {len(combined)} rows   Max size: {combined[\"input_size\"].max():,}')
"
echo "Merged. Re-run graph_results.py to regenerate plots."
