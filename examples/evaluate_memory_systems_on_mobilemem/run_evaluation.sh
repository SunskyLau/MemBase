#!/usr/bin/env bash
# ================================================================
#  Unified Stage 3: Question Answering and Evaluation on MobileMem
#
#  METHOD: long_context | mem0 | mem0_graph | rag | amem |
#          memos | evermemos | langmem | hipporag
#
#  Keep top_k/start_idx/end_idx aligned with run_search.sh.
# ================================================================
METHOD="amem"

top_k=""
start_idx=0
end_idx=""                    # empty means the end of the Stage-1 dataset

qa_model="gpt-4.1-mini"
judge_model="gpt-4.1-mini"
qa_batch_size=4
judge_batch_size=4
# ================================================================

set -euo pipefail
cd "$(dirname "$0")/../.."

example_dir="examples/evaluate_memory_systems_on_mobilemem"
dataset_type="MobileMem"
api_config_path="${example_dir}/configs/api_config.json"

case "$METHOD" in
  long_context) default_top_k=1  ;;
  mem0)         default_top_k=30 ;;
  mem0_graph)   default_top_k=30 ;;
  rag)          default_top_k=30 ;;
  amem)         default_top_k=30 ;;
  memos)        default_top_k=30 ;;
  evermemos)    default_top_k=30 ;;
  langmem)      default_top_k=30 ;;
  hipporag)     default_top_k=30 ;;
  *) echo "Unknown METHOD: '$METHOD'" >&2; exit 1 ;;
esac

resolved_top_k="${top_k:-$default_top_k}"
config_path="${example_dir}/configs/${METHOD}_config.json"
if [[ ! -f "$config_path" ]]; then
    echo "Config not found: $config_path" >&2
    exit 1
fi
if [[ ! -f "$api_config_path" ]]; then
    echo "API config not found: $api_config_path" >&2
    exit 1
fi

save_dir=$(python -c \
    "import json,sys; print(json.load(open(sys.argv[1], encoding='utf-8'))['save_dir'])" \
    "$config_path")
dataset_path="${save_dir}/${dataset_type}_stage_1.json"
if [[ ! -f "$dataset_path" ]]; then
    echo "Stage-1 dataset not found: $dataset_path" >&2
    echo "Run run_construction.sh first with the same METHOD." >&2
    exit 1
fi

dataset_size=$(python -c \
    "import json,sys; print(len(json.load(open(sys.argv[1], encoding='utf-8'))['trajectories']))" \
    "$dataset_path")
resolved_end_idx="${end_idx:-$dataset_size}"
if [[ "$resolved_end_idx" =~ ^[0-9]+$ ]] && (( resolved_end_idx > dataset_size )); then
    resolved_end_idx="$dataset_size"
fi

if ! [[ "$resolved_top_k" =~ ^[1-9][0-9]*$ ]]; then
    echo "top_k must be a positive integer, got: '$resolved_top_k'" >&2
    exit 1
fi
if ! [[ "$start_idx" =~ ^[0-9]+$ && "$resolved_end_idx" =~ ^[0-9]+$ ]]; then
    echo "start_idx and end_idx must be non-negative integers." >&2
    exit 1
fi
if (( start_idx >= resolved_end_idx )); then
    echo "start_idx must be smaller than end_idx (${start_idx} >= ${resolved_end_idx})." >&2
    exit 1
fi

search_results_path="${save_dir}/${resolved_top_k}_${start_idx}_${resolved_end_idx}.json"
if [[ ! -f "$search_results_path" ]]; then
    echo "Search results not found: $search_results_path" >&2
    echo "Run run_search.sh first and keep METHOD/top_k/start_idx/end_idx aligned." >&2
    exit 1
fi

log_dir="${example_dir}/logs/${METHOD}"
mkdir -p "$log_dir"
log_file="${log_dir}/evaluation.log"

nohup python memory_evaluation.py \
    --search-results-path "$search_results_path" \
    --dataset-type "$dataset_type" \
    --qa-model "$qa_model" \
    --judge-model "$judge_model" \
    --qa-batch-size "$qa_batch_size" \
    --judge-batch-size "$judge_batch_size" \
    --api-config-path "$api_config_path" \
    > "$log_file" 2>&1 &

echo $! > "${log_dir}/evaluation.pid"
echo "Evaluation started. PID: $(cat "${log_dir}/evaluation.pid")"
echo "Method  : $METHOD"
echo "Log     : $log_file"
echo "Input   : $search_results_path"
echo "Results : ${search_results_path%.json}_evaluation.json"
