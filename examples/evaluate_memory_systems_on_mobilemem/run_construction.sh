#!/usr/bin/env bash
# ================================================================
#  Unified Stage 1: Memory Construction on MobileMem
#
#  METHOD: long_context | mem0 | mem0_graph | rag | amem |
#          memos | evermemos | langmem | hipporag
#
#  An empty sample_size processes every trajectory while still
#  writing the standardized MobileMem_stage_1.json used by Stage 2.
# ================================================================
METHOD="amem"

dataset_path="examples/evaluate_memory_systems_on_mobilemem/data/MobileMem/mobilemem_data.json"
num_workers=4
sample_size=""               # empty means all MobileMem trajectories
seed=42
# ================================================================

set -euo pipefail
cd "$(dirname "$0")/../.."

example_dir="examples/evaluate_memory_systems_on_mobilemem"
dataset_type="MobileMem"

case "$METHOD" in
  long_context) memory_type="Long-Context" ;;
  mem0)         memory_type="Mem0"         ;;
  mem0_graph)   memory_type="Mem0"         ;;
  rag)          memory_type="NaiveRAG"     ;;
  amem)         memory_type="A-MEM"        ;;
  memos)        memory_type="MemOS"        ;;
  evermemos)    memory_type="EverMemOS"    ;;
  langmem)      memory_type="LangMem"      ;;
  hipporag)     memory_type="HippoRAG2"    ;;
  *) echo "Unknown METHOD: '$METHOD'" >&2; exit 1 ;;
esac

config_path="${example_dir}/configs/${METHOD}_config.json"
api_config_path="${example_dir}/configs/api_config.json"
if [[ ! -f "$config_path" ]]; then
    echo "Config not found: $config_path" >&2
    exit 1
fi
if [[ ! -f "$dataset_path" ]]; then
    echo "MobileMem data not found: $dataset_path" >&2
    echo "Place the dataset there or update dataset_path in this script." >&2
    exit 1
fi

save_dir=$(python -c \
    "import json,sys; print(json.load(open(sys.argv[1], encoding='utf-8'))['save_dir'])" \
    "$config_path")

if [[ -n "$sample_size" ]]; then
    resolved_sample_size="$sample_size"
else
    resolved_sample_size=$(python -c \
        "import json,sys; print(len(json.load(open(sys.argv[1], encoding='utf-8'))))" \
        "$dataset_path")
fi

if ! [[ "$resolved_sample_size" =~ ^[1-9][0-9]*$ ]]; then
    echo "sample_size must resolve to a positive integer, got: '$resolved_sample_size'" >&2
    exit 1
fi

extra_args=()
case "$METHOD" in
  memos)
    export MOS_EMBEDDER_TIMEOUT="${MOS_EMBEDDER_TIMEOUT:-120}"
    tokenizer_path=$(python -c \
        "import json,sys; print(json.load(open(sys.argv[1], encoding='utf-8'))['extractor_config']['config']['model_name_or_path'])" \
        "$config_path")
    extra_args+=(--tokenizer-path "$tokenizer_path")
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
token_cost_file="${save_dir}/token_cost_${METHOD}"
mkdir -p "$log_dir" "$save_dir"
log_file="${log_dir}/construction.log"

nohup python memory_construction.py \
    --memory-type "$memory_type" \
    --dataset-type "$dataset_type" \
    --dataset-path "$dataset_path" \
    --config-path "$config_path" \
    --num-workers "$num_workers" \
    --sample-size "$resolved_sample_size" \
    --seed "$seed" \
    --token-cost-save-filename "$token_cost_file" \
    --rerun \
    "${extra_args[@]}" \
    > "$log_file" 2>&1 &

echo $! > "${log_dir}/construction.pid"
echo "Construction started. PID: $(cat "${log_dir}/construction.pid")"
echo "Method  : $METHOD"
echo "Samples : $resolved_sample_size"
echo "Log     : $log_file"
echo "Output  : $save_dir"
echo "Stage-1 : ${save_dir}/${dataset_type}_stage_1.json"
