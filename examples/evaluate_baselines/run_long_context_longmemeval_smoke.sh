#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

config_path="examples/evaluate_baselines/configs/long_context_longmemeval.json"
dataset_path="data/longmemeval_s_cleaned.json"
output_dir="examples/evaluate_baselines/outputs/longmemeval/long_context"

python memory_construction.py \
  --memory-type "Long-Context" \
  --dataset-type "LongMemEval" \
  --dataset-path "$dataset_path" \
  --config-path "$config_path" \
  --num-workers 1 \
  --start-idx 0 \
  --end-idx 1 \
  --token-cost-save-filename "$output_dir/token_cost_smoke"

python memory_search.py \
  --memory-type "Long-Context" \
  --dataset-type "LongMemEval" \
  --dataset-path "$dataset_path" \
  --config-path "$config_path" \
  --num-workers 1 \
  --top-k 1 \
  --start-idx 0 \
  --end-idx 1

