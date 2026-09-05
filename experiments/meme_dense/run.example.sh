#!/usr/bin/env bash
set -euo pipefail

# 复制为 run.sh 并填写本段配置后，直接执行 ./run.sh。
# 官方稠密检索使用 text-embedding-3-small。
MODE="core"                              # smoke / core / full
CONDA_ENV="membase-meme-dense"            # 已安装依赖的 Conda 环境名
OPENAI_API_KEY=""                         # 在这里填写密钥
OPENAI_BASE_URL="https://api.n1n.ai/v1"
ANSWER_MODEL="gpt-4.1-mini"
JUDGE_MODEL="gpt-4.1-mini"
TOP_K=5                                  # 每个问题召回的文本块数量
WORKERS=4                                # 同时处理的样本数量
JUDGE_WORKERS=4                           # 同时评判的样本数量
CHECK_WORKERS=8                           # 每个样本内的评判并发数
RUN_ID="core_01"                         # 相同配置续跑保留此名称；新实验改名
DRY_RUN=0                                # 1：仅预览；0：运行实验

# 可选：./run.sh --dry-run 只预览；所有实验逻辑由 Python 组件完成。
case "${1:-}" in
  "") ;;
  --dry-run) DRY_RUN=1 ;;
  *) echo "Usage: $0 [--dry-run]; edit MODE at the top of this file." >&2; exit 2 ;;
esac

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
python_command=(conda run --no-capture-output -n "$CONDA_ENV" python)
extra_args=()
if [[ "$DRY_RUN" == "1" ]]; then
  python_command=(python)
  extra_args=(--dry-run)
fi
export OPENAI_API_KEY

exec "${python_command[@]}" "${repo_root}/scripts/run_benchmark.py" \
  --benchmark meme \
  --baseline dense \
  --mode "$MODE" \
  --output-dir "${script_dir}/runs" \
  --run-id "$RUN_ID" \
  --base-url "$OPENAI_BASE_URL" \
  --answer-model "$ANSWER_MODEL" \
  --judge-model "$JUDGE_MODEL" \
  --workers "$WORKERS" \
  --judge-workers "$JUDGE_WORKERS" \
  --check-workers "$CHECK_WORKERS" \
  --top-k "$TOP_K" \
  "${extra_args[@]}"
