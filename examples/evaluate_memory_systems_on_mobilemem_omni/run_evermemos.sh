#!/usr/bin/env bash
set -euo pipefail
exec bash "$(dirname "$0")/scripts/run_method.sh" evermemos "${1:-all}"
