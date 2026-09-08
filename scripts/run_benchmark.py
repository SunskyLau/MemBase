"""从参数启动一个独立基线实验；--dry-run 不调用模型。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from membase.utils.benchmark_files import read_json


def main(argv=None, *, stages=None) -> int:
    from membase.configs.model_profiles import load_environment, add_profile_arguments, apply_profile
    load_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    add_profile_arguments(parser)
    parser.add_argument("--benchmark", choices=["locomo", "longmemeval", "memoryagentbench", "meme"], required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--mode", choices=["smoke", "core", "full", "6k"], default="core",
                        help="6k 仅用于 MAB：6k 单跳和多跳全部问题")
    parser.add_argument("--output-dir", type=Path, required=True, help="本实验的 runs 目录")
    parser.add_argument("--run-id", default="", help="留空时以 UTC 时间命名；相同配置可续跑")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--upstream-dir", type=Path)
    parser.add_argument("--answer-model")
    parser.add_argument("--internal-model")
    parser.add_argument("--judge-model")
    parser.add_argument("--base-url")
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--parallel-jobs", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--judge-workers", type=int, default=4)
    parser.add_argument("--check-workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--memory-config", type=Path, help="记忆方法可选参数 JSON；不包含凭据")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-llm-requests", type=int)
    parser.add_argument("--max-embedding-requests", type=int)
    parser.add_argument("--budget-ledger", type=Path, help="多个受限验证共享的 SQLite 请求计数")
    parser.add_argument("--locomo-judge", action="store_true", help="额外报告 LoCoMo 模型评判；不替代官方 F1")
    args = parser.parse_args(argv)
    try:
        routing = apply_profile(args)
    except ValueError as error:
        parser.error(str(error))
    if args.mode == "6k" and args.benchmark != "memoryagentbench":
        parser.error("6k 模式仅适用于 MemoryAgentBench")

    from membase.datasets import memoryagentbench, meme
    from membase.runners.benchmark import BenchmarkRunConfig
    allowed = {"locomo": {"ourmem"}, "longmemeval": {"ourmem"},
               "memoryagentbench": {"long_context", "bm25", "ourmem", "amem"},
               "meme": {"in_context", "bm25", "dense", "md_flat", "ourmem"}}
    if args.baseline not in allowed[args.benchmark]:
        parser.error(f"{args.benchmark} 支持的基线：{sorted(allowed[args.benchmark])}")
    if stages is not None and args.baseline not in {"ourmem", "amem"}:
        parser.error("官方原生基线使用一键入口，不支持独立三阶段执行")
    if args.baseline in {"ourmem", "amem"}:
        from membase.datasets.official import default_paths
        default_data, default_upstream = default_paths(args.benchmark)
        if any(value is not None and value < 0 for value in (args.max_llm_requests, args.max_embedding_requests)):
            parser.error("请求上限必须非负；不设置表示不限制")
        from membase.runners.protocol import OfficialRunConfig
        config_type = OfficialRunConfig
        extra = dict(memory_config=args.memory_config,
                     seed=args.seed if args.seed is not None else 0, max_llm_requests=args.max_llm_requests,
                     max_embedding_requests=args.max_embedding_requests,
                     budget_ledger=args.budget_ledger, locomo_judge=args.locomo_judge)
    else:
        module = memoryagentbench if args.benchmark == "memoryagentbench" else meme
        default_data, default_upstream = module.DEFAULT_DATA_ROOT, module.DEFAULT_UPSTREAM
        config_type, extra = BenchmarkRunConfig, {}
        if args.benchmark == "meme" and args.baseline in {"in_context", "dense", "md_flat"}:
            from membase.runners.meme import MemeRunConfig
            config_type, extra = MemeRunConfig, {"seed":args.seed}
    if args.temperature is None:
        args.temperature = 0 if config_type.__name__ == "MemeRunConfig" else 0.7
    default_k = 10 if args.benchmark == "memoryagentbench" else 5
    if args.baseline == "ourmem":
        settings = read_json(args.memory_config) if args.memory_config else {}
        default_k = settings.get("top_k", 20)
    top_k = args.top_k if args.top_k is not None else default_k
    if min(top_k, args.parallel_jobs, args.workers, args.judge_workers, args.check_workers) < 1:
        parser.error("检索数量和并发数必须为正整数")
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    if args.model_profile:
        run_id += f"_{args.model_profile}"
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
        dry_run=args.dry_run, embedding_model=args.embedding_model, **routing, **extra,
    )
    return execute_config(config, stages=stages)


def execute_config(config, *, stages=None):
    from membase.configs.model_profiles import load_environment, redact
    load_environment()
    from membase.runners import memoryagentbench as mab_runner, meme as meme_runner
    runner = mab_runner if config.benchmark == "memoryagentbench" else meme_runner
    if config.baseline in {"ourmem", "amem"}:
        from membase.runners import protocol as runner
    try:
        if config.baseline in {"ourmem", "amem"}:
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
        text = redact(str(exc))
        print(f"实验未完成：{text}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
