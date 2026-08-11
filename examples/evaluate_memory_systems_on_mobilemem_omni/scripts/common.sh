#!/usr/bin/env bash

example_dir_abs="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_root="$(cd "${example_dir_abs}/../.." && pwd)"
example_dir="examples/evaluate_memory_systems_on_mobilemem_omni"
dataset_type="MobileMemOmni"

normalise_method() {
  case "${1:-}" in
    long_context|long-context) echo "long_context" ;;
    rag|naive_rag|naive-rag) echo "rag" ;;
    mem0) echo "mem0" ;;
    langmem) echo "langmem" ;;
    evermemos) echo "evermemos" ;;
    *)
      echo "Unknown method: '${1:-}'. Expected long_context, rag, mem0, langmem, or evermemos." >&2
      return 1
      ;;
  esac
}

memory_type_for() {
  case "$1" in
    long_context) echo "Long-Context" ;;
    rag) echo "NaiveRAG" ;;
    mem0) echo "Mem0" ;;
    langmem) echo "LangMem" ;;
    evermemos) echo "EverMemOS" ;;
  esac
}

default_top_k_for() {
  case "$1" in
    long_context) echo "1" ;;
    *) echo "15" ;;
  esac
}

config_path_for() {
  echo "${example_dir}/configs/$1_config.json"
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Required file not found: $1" >&2
    return 1
  fi
}

json_value() {
  python -c "import json,sys; print(json.load(open(sys.argv[1], encoding='utf-8'))[sys.argv[2]])" "$1" "$2"
}

raw_dataset_size() {
  python -c "import json,sys; d=json.load(open(sys.argv[1], encoding='utf-8')); print(len(d) if isinstance(d,list) else 1)" "$1"
}

standardized_dataset_size() {
  python -c "import json,sys; print(len(json.load(open(sys.argv[1], encoding='utf-8'))['trajectories']))" "$1"
}

validate_positive_integer() {
  if ! [[ "$2" =~ ^[1-9][0-9]*$ ]]; then
    echo "$1 must be a positive integer, got '$2'." >&2
    return 1
  fi
}

validate_range() {
  if ! [[ "$1" =~ ^[0-9]+$ && "$2" =~ ^[0-9]+$ ]]; then
    echo "start_idx and end_idx must be non-negative integers." >&2
    return 1
  fi
  if (( $1 >= $2 )); then
    echo "start_idx must be smaller than end_idx ($1 >= $2)." >&2
    return 1
  fi
}

configure_method_environment() {
  if [[ "$1" == "evermemos" ]]; then
    export MEMORY_LANGUAGE="${MEMORY_LANGUAGE:-zh}"
  fi
}
