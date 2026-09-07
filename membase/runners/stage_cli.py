"""原三阶段命令的官方协议选项；只解析和转交同一组运行参数。"""

from __future__ import annotations

import argparse
from pathlib import Path


def official_stage(stage, argv):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--protocol", default="membase")
    parser.add_argument("--run-dir", type=Path)
    options, remaining = parser.parse_known_args(argv)
    if options.protocol != "official":
        return None
    if options.run_dir is not None:
        from ..utils.benchmark_files import read_json
        from .protocol import OfficialRunConfig
        from scripts.run_benchmark import execute_config
        allowed = {"--dry-run"}
        if set(remaining) - allowed:
            parser.error("--run-dir 读取已冻结的运行配置，不与新的实验参数混用")
        saved = read_json(options.run_dir / "config.json")["config"]
        if saved["baseline"] == "ourmem":
            execution = read_json(options.run_dir / "execution.json")
            saved.update({key: execution[key] for key in ("workers", "check_workers")})
        for name in ("data_root", "upstream_dir", "run_dir", "memory_config", "budget_ledger"):
            if saved.get(name) is not None:
                saved[name] = Path(saved[name])
        if saved["run_dir"].resolve() != options.run_dir.resolve():
            parser.error("不支持移动运行目录后隐式迁移配置")
        return execute_config(OfficialRunConfig(**{**saved, "dry_run": "--dry-run" in remaining}), stages=(stage,))
    from scripts.run_benchmark import main
    aliases = {"--dataset-path": "--data-root", "--config-path": "--memory-config",
               "--num-workers": "--workers", "--qa-model": "--answer-model"}
    datasets = {"LoCoMo": "locomo", "LongMemEval": "longmemeval", "MemoryAgentBench": "memoryagentbench", "MEME": "meme"}
    expanded = []
    for value in remaining:
        flag, separator, argument = value.partition("=")
        expanded.extend([flag, argument] if separator and flag in {*aliases, "--memory-type", "--dataset-type"} else [value])
    remaining = expanded
    translated, index = [], 0
    while index < len(remaining):
        value = remaining[index]
        if value in {"--memory-type", "--dataset-type"} and index + 1 < len(remaining):
            argument = remaining[index + 1]
            if value == "--memory-type":
                if argument not in {"OurMem", "A-MEM"}:
                    parser.error("官方三阶段入口当前接入 OurMem 和 A-MEM；官方原生基线使用其一键入口")
                translated += ["--baseline", {"OurMem": "ourmem", "A-MEM": "amem"}[argument]]
            else:
                translated += ["--benchmark", datasets.get(argument, argument)]
            index += 2
            continue
        translated.append(aliases.get(value, value))
        index += 1
    if "--baseline" not in translated:
        translated += ["--baseline", "ourmem"]
    return main(translated, stages=(stage,))
