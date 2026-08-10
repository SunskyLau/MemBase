#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/scripts/common.sh"
method="$(normalise_method "${1:-}")"
memory_type="$(memory_type_for "$method")"
config_path="$(config_path_for "$method")"
num_workers="${NUM_WORKERS:-4}"
start_idx="${START_IDX:-0}"

cd "$repo_root"
require_file "$config_path"
save_dir="$(json_value "$config_path" save_dir)"
dataset_path="${save_dir}/${dataset_type}_stage_1.json"
require_file "$dataset_path"
dataset_size="$(standardized_dataset_size "$dataset_path")"
end_idx="${END_IDX:-$dataset_size}"
if [[ "$end_idx" =~ ^[0-9]+$ ]] && (( end_idx > dataset_size )); then
  end_idx="$dataset_size"
fi
top_k="${TOP_K:-$(default_top_k_for "$method")}"
validate_positive_integer "top_k" "$top_k"
validate_range "$start_idx" "$end_idx"
configure_method_environment "$method"

python memory_search.py \
  --memory-type "$memory_type" \
  --dataset-type "$dataset_type" \
  --dataset-path "$dataset_path" \
  --dataset-standardized \
  --config-path "$config_path" \
  --num-workers "$num_workers" \
  --top-k "$top_k" \
  --start-idx "$start_idx" \
  --end-idx "$end_idx"

echo "Search results: ${save_dir}/${top_k}_${start_idx}_${end_idx}.json"
