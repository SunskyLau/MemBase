#!/usr/bin/env bash
set -euo pipefail

# 模型与运行名称在这里确定；其余参数沿用同目录的官方配置模板。
MODEL="qwen3-30b-a3b-instruct-2507"
MODE="core"                              # smoke / core / full
RUN_ID="${MODEL}_${MODE}_01"             # 新实验改编号；相同配置续跑保留名称

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${script_dir}/run.example.sh" \
  --model-profile qwen --answer-model "$MODEL" --internal-model "$MODEL" \
  --mode "$MODE" --run-id "$RUN_ID" "$@"
