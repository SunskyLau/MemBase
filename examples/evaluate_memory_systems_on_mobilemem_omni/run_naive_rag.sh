#!/usr/bin/env bash
set -euo pipefail
exec bash "$(dirname "$0")/scripts/run_method.sh" rag "${1:-all}"
