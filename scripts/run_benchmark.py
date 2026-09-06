"""从参数启动一个独立基线实验；--dry-run 不调用模型。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None, *, stages=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=["locomo", "longmemeval", "memoryagentbench", "meme"], required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--mode", choices=["smoke", "core", "full"], default="core")
    parser.add_argument("--output-dir", type=Path, required=True, help="本实验的 runs 目录")
    parser.add_argument("--run-id", default="", help="留空时以 UTC 时间命名；相同配置可续跑")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--upstream-dir", type=Path)
    parser.add_argument("--answer-model", default="gpt-4.1-mini")
    parser.add_argument("--internal-model", default="gpt-4.1-mini")
    parser.add_argument("--judge-model", default="gpt-4.1-mini")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--parallel-jobs", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--judge-workers", type=int, default=4)
    parser.add_argument("--check-workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--memory-config", type=Path, help="OurMem 可选参数 JSON；不包含凭据")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-llm-requests", type=int)
    parser.add_argument("--max-embedding-requests", type=int)
    parser.add_argument("--budget-ledger", type=Path, help="多个受限验证共享的 SQLite 请求计数")
    parser.add_argument("--locomo-judge", action="store_true", help="额外报告 LoCoMo 模型评判；不替代官方 F1")
    args = parser.parse_args(argv)

    from membase.datasets import memoryagentbench, meme
    from membase.runners.benchmark import BenchmarkRunConfig
    allowed = {"locomo": {"ourmem"}, "longmemeval": {"ourmem"},
               "memoryagentbench": {"long_context", "bm25", "ourmem"},
               "meme": {"in_context", "bm25", "dense", "md_flat", "ourmem"}}
    if args.baseline not in allowed[args.benchmark]:
        parser.error(f"{args.benchmark} 支持的基线：{sorted(allowed[args.benchmark])}")
    if stages is not None and args.baseline != "ourmem":
        parser.error("官方原生基线使用一键入口，不支持独立三阶段执行")
    if args.baseline == "ourmem":
        from membase.datasets.official import default_paths
        default_data, default_upstream = default_paths(args.benchmark)
        if args.top_k is not None:
            parser.error("OurMem 使用分阶段候选预算，不接受单一 --top-k；请使用 --memory-config")
        if any(value is not None and value < 0 for value in (args.max_llm_requests, args.max_embedding_requests)):
            parser.error("请求上限必须非负；不设置表示不限制")
        from membase.runners.protocol import OfficialRunConfig
        config_type = OfficialRunConfig
        extra = dict(embedding_model=args.embedding_model, memory_config=args.memory_config,
                     seed=args.seed, max_llm_requests=args.max_llm_requests,
                     max_embedding_requests=args.max_embedding_requests,
                     budget_ledger=args.budget_ledger, locomo_judge=args.locomo_judge)
    else:
        module = memoryagentbench if args.benchmark == "memoryagentbench" else meme
        default_data, default_upstream = module.DEFAULT_DATA_ROOT, module.DEFAULT_UPSTREAM
        config_type, extra = BenchmarkRunConfig, {}
    top_k = args.top_k if args.top_k is not None else (10 if args.benchmark == "memoryagentbench" else 5)
    if min(top_k, args.parallel_jobs, args.workers, args.judge_workers, args.check_workers) < 1:
        parser.error("检索数量和并发数必须为正整数")
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    if Path(run_id).name != run_id or run_id in {".", ".."}:
        parser.error("run-id 必须是单个目录名")
    config = config_type(
        benchmark=args.benchmark, baseline=args.baseline, mode=args.mode,
        data_root=(args.data_root or default_data).resolve(),
        upstream_dir=(args.upstream_dir or default_upstream).resolve(),
        run_dir=(args.output_dir / run_id).resolve(), answer_model=args.answer_model,
        internal_model=args.internal_model, judge_model=args.judge_model, base_url=args.base_url,
        top_k=top_k, temperature=args.temperature, parallel_jobs=args.parallel_jobs,
        workers=args.workers, judge_workers=args.judge_workers, check_workers=args.check_workers,
        dry_run=args.dry_run, **extra,
    )
    return execute_config(config, stages=stages)


def execute_config(config, *, stages=None):
    from membase.runners import memoryagentbench as mab_runner, meme as meme_runner
    runner = mab_runner if config.benchmark == "memoryagentbench" else meme_runner
    if config.baseline == "ourmem":
        from membase.runners import protocol as runner
    try:
        if config.baseline == "ourmem":
            runner.run(config, stages=stages or ("construction", "search", "evaluation"))
        else:
            runner.run(config)
    except (OSError, ValueError, RuntimeError, ImportError, KeyError, TypeError) as exc:
        # start_run 拒绝冲突配置时不能改写已有运行的状态。
        from membase.utils.benchmark_files import read_json
        from membase.utils.experiment import fail_run
        from membase.utils.ourmem_version import ImplementationMismatchError
        manifest = config.run_dir / "config.json"
        if not config.dry_run and manifest.exists() and not isinstance(exc, ImplementationMismatchError):
            try:
                if read_json(manifest).get("config") == config.saved_config():
                    fail_run(config.run_dir, exc)
            except (OSError, ValueError):
                pass
        text = str(exc)
        if os.environ.get("OPENAI_API_KEY"):
            text = text.replace(os.environ["OPENAI_API_KEY"], "[REDACTED]")
        print(f"实验未完成：{text}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
