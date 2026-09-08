#!/usr/bin/env bash
set -euo pipefail

# 密钥与地址统一读取 envs/.env；模型名称在这里独立调整。
experiment_model="gpt"                     # gpt / qwen：构建与回答一起切换
JUDGE_PROFILE="gpt"                     # 两套实验固定同一评判模型
EMBEDDING_PROFILE="gpt"                 # 不把 OpenAI 嵌入发往百炼

MODE="core"
CONDA_ENV="membase-amem"
TOP_K=10
WORKERS=2
JUDGE_WORKERS=4
CHECK_WORKERS=8
PARALLEL_JOBS=1
SEED=0
PROGRESS_INTERVAL=10
RUN_ID="mab_6k_amem_4o_mini_01"
MEMORY_CONFIG=""
MAX_LLM_REQUESTS=""
MAX_EMBEDDING_REQUESTS=""
BUDGET_LEDGER=""
LOCOMO_JUDGE=0
EMBEDDING_MODEL="text-embedding-3-small"
DRY_RUN=0

STAGE="all"
extra_args=()
for argument in "$@"; do
  case "$argument" in
    all|construction|search|evaluation) STAGE="$argument" ;;
    --dry-run) DRY_RUN=1 ;;
    *) extra_args+=("$argument") ;;
  esac
done

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
python_command=(conda run --no-capture-output -n "$CONDA_ENV" python)
if [[ "$DRY_RUN" == "1" ]]; then
  python_command=(python)
  extra_args+=(--dry-run)
fi
[[ -z "$MEMORY_CONFIG" ]] || extra_args+=(--memory-config "$MEMORY_CONFIG")
[[ -z "$MAX_LLM_REQUESTS" ]] || extra_args+=(--max-llm-requests "$MAX_LLM_REQUESTS")
[[ -z "$MAX_EMBEDDING_REQUESTS" ]] || extra_args+=(--max-embedding-requests "$MAX_EMBEDDING_REQUESTS")
[[ -z "$BUDGET_LEDGER" ]] || extra_args+=(--budget-ledger "$BUDGET_LEDGER")
[[ "$LOCOMO_JUDGE" != "1" ]] || extra_args+=(--locomo-judge)

exec "${python_command[@]}" "${repo_root}/scripts/run_with_progress.py" \
  --entry "$STAGE" --progress-interval "$PROGRESS_INTERVAL" \
  --benchmark memoryagentbench --baseline amem --mode "$MODE" \
  --output-dir "${repo_root}/experiments/amem_memoryagentbench/runs" --run-id "$RUN_ID" \
  --model-profile "$experiment_model" --gpt-model gpt-4o-mini --qwen-model qwen3-30b-a3b-instruct-2507 \
  --judge-profile "$JUDGE_PROFILE" --judge-model gpt-4o-2024-11-20 --embedding-profile "$EMBEDDING_PROFILE" \
  --embedding-model "$EMBEDDING_MODEL" --top-k "$TOP_K" --seed "$SEED" \
  --workers "$WORKERS" --judge-workers "$JUDGE_WORKERS" --check-workers "$CHECK_WORKERS" \
  --parallel-jobs "$PARALLEL_JOBS" "${extra_args[@]}"
