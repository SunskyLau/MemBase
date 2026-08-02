#!/usr/bin/env bash
# ================================================================
#  Unified Stage 2: Memory Retrieval on MobileMem
#
#  METHOD: long_context | mem0 | mem0_graph | rag | amem |
#          memos | evermemos | langmem | hipporag
#
#  Long-Context defaults to top_k=1. Other methods default to 30.
#  Set top_k to a positive integer to override the method default.
# ================================================================
METHOD="amem"

num_workers=4
top_k=""
start_idx=0
end_idx=""                    # empty means the end of the Stage-1 dataset
# ================================================================

set -euo pipefail
cd "$(dirname "$0")/../.."

example_dir="examples/evaluate_memory_systems_on_mobilemem"
dataset_type="MobileMem"

case "$METHOD" in
  long_context) memory_type="Long-Context"; default_top_k=1  ;;
  mem0)         memory_type="Mem0";         default_top_k=30 ;;
  mem0_graph)   memory_type="Mem0";         default_top_k=30 ;;
  rag)          memory_type="NaiveRAG";     default_top_k=30 ;;
  amem)         memory_type="A-MEM";        default_top_k=30 ;;
  memos)        memory_type="MemOS";        default_top_k=30 ;;
  evermemos)    memory_type="EverMemOS";    default_top_k=30 ;;
  langmem)      memory_type="LangMem";      default_top_k=30 ;;
  hipporag)     memory_type="HippoRAG2";    default_top_k=30 ;;
  *) echo "Unknown METHOD: '$METHOD'" >&2; exit 1 ;;
esac

resolved_top_k="${top_k:-$default_top_k}"
config_path="${example_dir}/configs/${METHOD}_config.json"
api_config_path="${example_dir}/configs/api_config.json"
if [[ ! -f "$config_path" ]]; then
    echo "Config not found: $config_path" >&2
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

case "$METHOD" in
  memos)
    export MOS_EMBEDDER_TIMEOUT="${MOS_EMBEDDER_TIMEOUT:-120}"
    ;;
  evermemos)
    export MEMORY_LANGUAGE="${MEMORY_LANGUAGE:-zh}"
    ;;
  hipporag)
    if [[ -z "${OPENAI_API_KEY:-}" ]]; then
        if [[ ! -f "$api_config_path" ]]; then
            echo "API config not found: $api_config_path" >&2
            exit 1
        fi
        OPENAI_API_KEY=$(python -c \
            "import json,sys; print(json.load(open(sys.argv[1], encoding='utf-8'))['api_keys'][0])" \
            "$api_config_path")
        export OPENAI_API_KEY
    fi
    ;;
esac

log_dir="${example_dir}/logs/${METHOD}"
mkdir -p "$log_dir"
log_file="${log_dir}/search.log"
search_results_path="${save_dir}/${resolved_top_k}_${start_idx}_${resolved_end_idx}.json"

nohup python memory_search.py \
    --memory-type "$memory_type" \
    --dataset-type "$dataset_type" \
    --dataset-path "$dataset_path" \
    --dataset-standardized \
    --config-path "$config_path" \
    --num-workers "$num_workers" \
    --top-k "$resolved_top_k" \
    --start-idx "$start_idx" \
    --end-idx "$resolved_end_idx" \
    > "$log_file" 2>&1 &

echo $! > "${log_dir}/search.pid"
echo "Search started. PID: $(cat "${log_dir}/search.pid")"
echo "Method  : $METHOD"
echo "Range   : ${start_idx}-${resolved_end_idx}"
echo "Top-k   : $resolved_top_k"
echo "Log     : $log_file"
echo "Results : $search_results_path"
