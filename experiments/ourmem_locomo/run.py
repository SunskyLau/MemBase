"""兼容原入口的位置；实际实验统一交给共享 Python 运行器。"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main() -> int:
    from scripts.run_with_progress import main as run
    legacy = {"--construction-concurrency", "--evaluation-concurrency", "--api-config", "--resume",
              "--stage", "--rescore", "--analysis-only", "--config"}
    if any(arg.split("=", 1)[0] in legacy for arg in sys.argv[1:]):
        print("旧版运行参数不适用于 V5；请使用同目录 run.sh。旧实验结果保持原样，不会被 V5 复用。", file=sys.stderr)
        return 2
    arguments = sys.argv[1:]
    if not any(arg.startswith("--output-dir") for arg in arguments):
        arguments += ["--output-dir", str(Path(__file__).resolve().parent / "runs")]
    sys.argv = [sys.argv[0], "--benchmark", "locomo", "--baseline", "ourmem", *arguments]
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
