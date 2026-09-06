#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
entry="${script_dir}/run.sh"
[[ -f "$entry" ]] || entry="${script_dir}/run.example.sh"
exec bash "$entry" evaluation "$@"
