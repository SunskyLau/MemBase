#!/usr/bin/env bash
set -euo pipefail

# 6k 单跳、多跳各 100 题；历史输入保持完整。
CONDA_ENV="membase-ourmem"
RUN_ID="mab_6k_ourmem_04"
WORKERS=2
MODEL="gpt-4.1-mini"
EMBEDDING_MODEL="text-embedding-3-small"
PROGRESS_INTERVAL=10

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
STAGE=all
extra_args=()
python_command=(conda run --no-capture-output -n "$CONDA_ENV" python)
for argument in "$@"; do
  case "$argument" in
    all|construction|search|evaluation) STAGE="$argument" ;;
    --dry-run) extra_args+=(--dry-run); python_command=(python) ;;
    *) echo "Usage: $0 [all|construction|search|evaluation] [--dry-run]" >&2; exit 2 ;;
  esac
done

# 凭据沿用本地配置，不复制到脚本或命令日志中。
if [[ -f "${repo_root}/envs/.env" ]]; then
  set -a
  source "${repo_root}/envs/.env"
  set +a
fi
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.n1n.ai/v1}"

exec "${python_command[@]}" "${repo_root}/scripts/run_with_progress.py" \
  --progress-interval "$PROGRESS_INTERVAL" --entry "$STAGE" \
  --benchmark memoryagentbench --baseline ourmem --mode 6k \
  --output-dir "${repo_root}/experiments/ourmem_memoryagentbench/runs" --run-id "$RUN_ID" \
  --base-url "$OPENAI_BASE_URL" --internal-model "$MODEL" \
  --answer-model "$MODEL" --judge-model "$MODEL" \
  --embedding-model "$EMBEDDING_MODEL" --workers "$WORKERS" \
  --temperature 0.7 --seed 0 --check-workers 8 "${extra_args[@]}"
