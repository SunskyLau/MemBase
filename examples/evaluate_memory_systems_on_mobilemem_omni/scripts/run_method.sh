#!/usr/bin/env bash
set -euo pipefail

method="${1:?A method name is required.}"
stage="${2:-all}"
example_dir_abs="$(cd "$(dirname "$0")/.." && pwd)"

run_stage() {
  bash "${example_dir_abs}/run_$1.sh" "$method"
}

case "$stage" in
  construction|search|evaluation)
    run_stage "$stage"
    ;;
  all)
    run_stage construction
    run_stage search
    run_stage evaluation
    ;;
  *)
    echo "Usage: $0 METHOD {construction|search|evaluation|all}" >&2
    exit 1
    ;;
esac
