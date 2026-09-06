#!/usr/bin/env bash
set -euo pipefail

PROGRESS_INTERVAL=10                     # 进度刷新间隔（秒），不影响实验配置

# 在这里填写普通配置和密钥，然后直接运行 ./run.sh。
MODE="core"
CONDA_ENV="membase-ourmem"
OPENAI_API_KEY="${OPENAI_API_KEY:-}"
OPENAI_BASE_URL="https://api.n1n.ai/v1"
MEMORY_MODEL="gpt-4.1-mini"
ANSWER_MODEL="gpt-4.1-mini"
JUDGE_MODEL="gpt-4.1-mini"
EMBEDDING_MODEL="text-embedding-3-small"
WORKERS=1
SEED=0
CHECK_WORKERS=8
RUN_ID="v5_core_01"
DRY_RUN=0
MEMORY_CONFIG=""                       # 可选：OurMem 参数 JSON，不含密钥
MAX_LLM_REQUESTS=""                     # 留空为全量；受限验证可填 100
MAX_EMBEDDING_REQUESTS=""               # 留空为全量；受限验证可填 20
BUDGET_LEDGER=""                        # 可选：跨实验共用的请求计数 SQLite
LOCOMO_JUDGE=0                          # 额外模型评判，与官方 F1 分开

case "${1:-}" in
  "") ;;
  --dry-run) DRY_RUN=1 ;;
  *) echo "Usage: $0 [--dry-run]; edit configuration at the top." >&2; exit 2 ;;
esac
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
python_command=(conda run --no-capture-output -n "$CONDA_ENV" python)
extra_args=()
if [[ "$DRY_RUN" == "1" ]]; then
  python_command=(python)
  extra_args+=(--dry-run)
fi
[[ -z "$MEMORY_CONFIG" ]] || extra_args+=(--memory-config "$MEMORY_CONFIG")
[[ -z "$MAX_LLM_REQUESTS" ]] || extra_args+=(--max-llm-requests "$MAX_LLM_REQUESTS")
[[ -z "$MAX_EMBEDDING_REQUESTS" ]] || extra_args+=(--max-embedding-requests "$MAX_EMBEDDING_REQUESTS")
[[ -z "$BUDGET_LEDGER" ]] || extra_args+=(--budget-ledger "$BUDGET_LEDGER")
[[ "$LOCOMO_JUDGE" != "1" ]] || extra_args+=(--locomo-judge)
export OPENAI_API_KEY

exec "${python_command[@]}" "${repo_root}/scripts/run_with_progress.py" --progress-interval "$PROGRESS_INTERVAL" \
  --benchmark locomo --baseline ourmem --mode "$MODE" \
  --output-dir "${script_dir}/runs" --run-id "$RUN_ID" \
  --base-url "$OPENAI_BASE_URL" --internal-model "$MEMORY_MODEL" \
  --answer-model "$ANSWER_MODEL" --judge-model "$JUDGE_MODEL" \
  --embedding-model "$EMBEDDING_MODEL" --workers "$WORKERS" --seed "$SEED" \
  --check-workers "$CHECK_WORKERS" "${extra_args[@]}"
