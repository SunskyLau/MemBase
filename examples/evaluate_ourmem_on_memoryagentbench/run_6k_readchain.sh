#!/usr/bin/env bash
set -euo pipefail

# 原记忆复制到独立对照目录；不重建、不改动原结果。普通重跑复用本次已完成题目。
CONDA_ENV="membase-ourmem"
SOURCE_RUN="mab_6k_ourmem_4o_mini_01"
RUN_ID="mab_6k_ourmem_4o_mini_readchain_01"
PROGRESS_INTERVAL=10

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
parent="${repo_root}/experiments/ourmem_memoryagentbench/runs"
run_dir="${parent}/${RUN_ID}"
python_command=(conda run --no-capture-output -n "$CONDA_ENV" python)
extra=()
if [[ "$#" == 1 && "$1" == "--dry-run" ]]; then
  extra+=(--dry-run)
elif [[ "$#" != 0 ]]; then
  echo "Usage: $0 [--dry-run]" >&2
  exit 2
fi
if [[ ! -f "${run_dir}/config.json" ]]; then
  "${python_command[@]}" "${repo_root}/scripts/prepare_read_resume.py" \
    --from-run "${parent}/${SOURCE_RUN}" --run-dir "$run_dir" "${extra[@]}"
  if [[ "${#extra[@]}" != 0 ]]; then exit 0; fi
fi
if [[ "${#extra[@]}" == 0 ]]; then
  set -a
  source "${repo_root}/envs/.env"
  set +a
fi
for stage in search evaluation; do
  "${python_command[@]}" "${repo_root}/scripts/run_with_progress.py" \
    --entry "$stage" --progress-interval "$PROGRESS_INTERVAL" \
    --run-dir "$run_dir" "${extra[@]}"
done
