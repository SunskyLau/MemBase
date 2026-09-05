"""下载、校验官方数据并准备固定版本代码。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=["all", "memoryagentbench", "meme"], default="all")
    parser.add_argument("--check-only", action="store_true", help="只校验已准备的数据，不下载、不修改文件")
    args = parser.parse_args()
    from membase.datasets import memoryagentbench, meme
    selected = [memoryagentbench, meme] if args.benchmark == "all" else [
        memoryagentbench if args.benchmark == "memoryagentbench" else meme]
    try:
        for module in selected:
            result = module.check() if args.check_only else module.prepare()
            print(json.dumps(result, ensure_ascii=False, indent=2))
    except (OSError, ValueError, ImportError, subprocess.CalledProcessError) as exc:
        print(f"数据准备失败：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
