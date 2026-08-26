#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is not set; refusing to start paid QA/judge calls." >&2
  exit 2
fi

dataset="${1:-}"
qa_model="${QA_MODEL:-gpt-4.1-mini}"
judge_model="${JUDGE_MODEL:-gpt-4.1-mini}"

case "$dataset" in
  locomo)
    dataset_type="LoCoMo"
    search_results_path="examples/evaluate_baselines/outputs/locomo/long_context/1_0_1.json"
    timestamp_args=()
    ;;
  longmemeval)
    dataset_type="LongMemEval"
    search_results_path="examples/evaluate_baselines/outputs/longmemeval/long_context/1_0_1.json"
    timestamp_args=(--add-question-timestamp)
    ;;
  *)
    echo "Usage: $0 {locomo|longmemeval}" >&2
    exit 2
    ;;
esac

python memory_evaluation.py \
  --search-results-path "$search_results_path" \
  --dataset-type "$dataset_type" \
  --qa-model "$qa_model" \
  --judge-model "$judge_model" \
  --qa-batch-size 1 \
  --judge-batch-size 1 \
  "${timestamp_args[@]}"
