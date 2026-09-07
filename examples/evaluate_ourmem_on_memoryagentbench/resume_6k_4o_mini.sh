#!/usr/bin/env bash
set -euo pipefail

# 只重跑检索、回答和评分，配置从已完成构建的运行目录读取。
CONDA_ENV="membase-ourmem"
RUN_ID="mab_6k_ourmem_4o_mini_01"
PROGRESS_INTERVAL=10

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
run_dir="${repo_root}/experiments/ourmem_memoryagentbench/runs/${RUN_ID}"
extra_args=()
python_command=(conda run --no-capture-output -n "$CONDA_ENV" python)
if [[ "$#" == 1 && "$1" == "--dry-run" ]]; then
  extra_args+=(--dry-run)
elif [[ "$#" != 0 ]]; then
  echo "Usage: $0 [--dry-run]" >&2
  exit 2
fi
if [[ "${#extra_args[@]}" == 0 ]]; then
  set -a
  source "${repo_root}/envs/.env"
  set +a
fi

for stage in search evaluation; do
  "${python_command[@]}" "${repo_root}/scripts/run_with_progress.py" \
    --entry "$stage" --progress-interval "$PROGRESS_INTERVAL" \
    --run-dir "$run_dir" "${extra_args[@]}"
done
