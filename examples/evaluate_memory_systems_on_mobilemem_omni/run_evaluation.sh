#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/scripts/common.sh"
method="$(normalise_method "${1:-}")"
config_path="$(config_path_for "$method")"
start_idx="${START_IDX:-0}"
qa_model="${QA_MODEL:-gpt-5.4-mini}"
judge_model="${JUDGE_MODEL:-Qwen3-14B}"
qa_batch_size="${QA_BATCH_SIZE:-4}"
judge_batch_size="${JUDGE_BATCH_SIZE:-4}"
api_config_path="${API_CONFIG_PATH:-${example_dir}/configs/api_config.json}"
prompt_template="${example_dir}/qa_prompt.py:get_mobilemem_omni_qa_prompt"

cd "$repo_root"
require_file "$config_path"
require_file "$api_config_path"
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
search_results_path="${save_dir}/${top_k}_${start_idx}_${end_idx}.json"
require_file "$search_results_path"

python memory_evaluation.py \
  --search-results-path "$search_results_path" \
  --dataset-type "$dataset_type" \
  --qa-model "$qa_model" \
  --judge-model "$judge_model" \
  --qa-batch-size "$qa_batch_size" \
  --judge-batch-size "$judge_batch_size" \
  --api-config-path "$api_config_path" \
  --prompt-template "$prompt_template"

evaluation_results_path="${search_results_path%.json}_evaluation.json"
python "${example_dir}/summarize_results.py" "$evaluation_results_path"
