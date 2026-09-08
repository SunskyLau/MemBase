#!/usr/bin/env bash
set -euo pipefail

# 只在这里调整实验配置；密钥和服务地址自动读取 envs/.env。
MODEL_PROFILE="gpt"
MODE="core"                              # smoke / core / full
RUN_ID="meme_${MODE}_md_flat_official_02"
CONDA_ENV="membase-meme-mdflat"
ANSWER_MODEL="gpt-4o-mini"
JUDGE_MODEL="gpt-4o-2024-11-20"
WORKERS=4
JUDGE_WORKERS=4
CHECK_WORKERS=8
PROGRESS_INTERVAL=10
DRY_RUN=0

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
extra_args=()
for argument in "$@"; do
  case "$argument" in
    --dry-run) DRY_RUN=1 ;;
    *) extra_args+=("$argument") ;;
  esac
done
python_command=(conda run --no-capture-output -n "$CONDA_ENV" python)
if [[ "$DRY_RUN" == "1" ]]; then
  python_command=(python)
  extra_args+=(--dry-run)
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec "${python_command[@]}" "${repo_root}/scripts/run_with_progress.py" \
  --entry all --progress-interval "$PROGRESS_INTERVAL" \
  --benchmark meme --baseline md_flat --mode "$MODE" \
  --output-dir "${repo_root}/experiments/meme_md_flat/runs" --run-id "$RUN_ID" \
  --model-profile "$MODEL_PROFILE" --gpt-model "$ANSWER_MODEL" \
  --judge-profile gpt --judge-model "$JUDGE_MODEL" --embedding-profile gpt \
  --workers "$WORKERS" --judge-workers "$JUDGE_WORKERS" --check-workers "$CHECK_WORKERS" \
  "${extra_args[@]}"
