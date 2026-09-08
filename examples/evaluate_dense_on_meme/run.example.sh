#!/usr/bin/env bash
set -euo pipefail

# 密钥与地址统一读取 envs/.env；模型名称在这里独立调整。
experiment_model="gpt"                     # gpt / qwen：构建与回答一起切换
JUDGE_PROFILE="gpt"                     # 两套实验固定同一评判模型
EMBEDDING_PROFILE="gpt"                 # 不把 OpenAI 嵌入发往百炼

MODE="core"                              # smoke / core / full
CONDA_ENV="membase-meme-dense"            # 已安装依赖的 Conda 环境名
TOP_K=5                                  # 每个问题召回的文本块数量
WORKERS=4                                # 同时处理的样本数量
JUDGE_WORKERS=4                           # 同时评判的样本数量
CHECK_WORKERS=8                           # 每个样本内的评判并发数
PARALLEL_JOBS=1
SEED=0
PROGRESS_INTERVAL=10                     # 进度刷新间隔（秒），不影响实验配置
RUN_ID="membase_core_01"                         # 相同配置续跑保留此名称；新实验改名
MEMORY_CONFIG=""
MAX_LLM_REQUESTS=""
MAX_EMBEDDING_REQUESTS=""
BUDGET_LEDGER=""
LOCOMO_JUDGE=0
EMBEDDING_MODEL="text-embedding-3-small"
DRY_RUN=0                                # 1：仅预览；0：运行实验

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
[[ "$STAGE" == "all" ]] || { echo "官方原生基线仅支持 all 入口" >&2; exit 2; }

exec "${python_command[@]}" "${repo_root}/scripts/run_with_progress.py" \
  --entry "$STAGE" --progress-interval "$PROGRESS_INTERVAL" \
  --benchmark meme --baseline dense --mode "$MODE" \
  --output-dir "${repo_root}/experiments/meme_dense/runs" --run-id "$RUN_ID" \
  --model-profile "$experiment_model" --gpt-model gpt-4o-mini --qwen-model qwen3-30b-a3b-instruct-2507 \
  --judge-profile "$JUDGE_PROFILE" --judge-model gpt-4o-2024-11-20 --embedding-profile "$EMBEDDING_PROFILE" \
  --embedding-model "$EMBEDDING_MODEL" --top-k "$TOP_K" --seed "$SEED" \
  --workers "$WORKERS" --judge-workers "$JUDGE_WORKERS" --check-workers "$CHECK_WORKERS" \
  --parallel-jobs "$PARALLEL_JOBS" "${extra_args[@]}"
