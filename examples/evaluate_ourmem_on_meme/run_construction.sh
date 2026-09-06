#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 三个阶段读取同一份本地配置；未复制 run.sh 时可用环境变量配合示例。
entry="${script_dir}/run.sh"
[[ -f "$entry" ]] || entry="${script_dir}/run.example.sh"
exec bash "$entry" construction "$@"
