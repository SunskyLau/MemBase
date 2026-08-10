#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/scripts/common.sh"
method="$(normalise_method "${1:-}")"
memory_type="$(memory_type_for "$method")"
config_path="$(config_path_for "$method")"
dataset_path="${DATASET_PATH:-${example_dir}/data/mobilemem_omni_locomo.json}"
num_workers="${NUM_WORKERS:-4}"
seed="${SEED:-42}"

cd "$repo_root"
require_file "$config_path"
require_file "$dataset_path"
sample_size="${SAMPLE_SIZE:-$(raw_dataset_size "$dataset_path")}"
validate_positive_integer "sample_size" "$sample_size"
configure_method_environment "$method"

save_dir="$(json_value "$config_path" save_dir)"
mkdir -p "$save_dir"
args=(
  --memory-type "$memory_type"
  --dataset-type "$dataset_type"
  --dataset-path "$dataset_path"
  --config-path "$config_path"
  --num-workers "$num_workers"
  --sample-size "$sample_size"
  --seed "$seed"
  --token-cost-save-filename "${save_dir}/token_cost_${method}"
)
if [[ "${RERUN:-1}" == "1" ]]; then
  args+=(--rerun)
fi

python memory_construction.py "${args[@]}"
echo "Stage-1 dataset: ${save_dir}/${dataset_type}_stage_1.json"
