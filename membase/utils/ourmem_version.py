"""记录实际参与 OurMem 运行的源码内容，不把未提交工作树误当作 Git HEAD。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .benchmark_files import REPO_ROOT, sha256_file


class ImplementationMismatchError(ValueError):
    """拒绝恢复时不应把已有成功运行的状态改成失败。"""


_LOCAL_FILES = (
    "membase/configs/ourmem.py", "membase/configs/base.py", "membase/layers/ourmem.py",
    "membase/datasets/official.py", "membase/datasets/memoryagentbench.py",
    "membase/datasets/meme.py", "membase/evaluation/official.py", "membase/evaluation/meme.py",
    "membase/evaluation/memoryagentbench.py", "membase/runners/protocol.py",
    "membase/runners/benchmark.py", "membase/utils/benchmark_files.py",
    "membase/utils/experiment.py", "membase/utils/ourmem_version.py",
    "scripts/run_benchmark.py", "scripts/prepare_benchmarks.py", "envs/ourmem_requirements.txt",
    "memory_construction.py", "memory_search.py", "memory_evaluation.py",
    "membase/runners/construction.py", "membase/runners/search.py", "membase/runners/evaluation.py",
    "membase/runners/stage_cli.py", "membase/model_types/dataset.py", "membase/layers/base.py",
    "membase/runners/question_outcomes.py",
    "membase/datasets/base.py", "membase/datasets/locomo.py", "membase/datasets/longmemeval.py",
    "membase/datasets/__init__.py", "membase/inference_utils/base_operator.py",
    "membase/inference_utils/operators.py", "membase/inference_utils/backends.py",
    "membase/inference_utils/prompts.py",
    "membase/inference_utils/model_client.py", "membase/utils/tokenization.py",
)

_OFFICIAL_FILES = {
    "locomo": ("task_eval/evaluation.py", "task_eval/gpt_utils.py", "global_methods.py"),
    "longmemeval": ("src/evaluation/evaluate_qa.py", "src/generation/run_generation.py"),
    "memoryagentbench": ("utils/templates.py", "utils/eval_other_utils.py", "utils/eval_data_utils.py", "agent.py"),
    "meme": ("code/agents/base.py", "code/eval/judge.py", "code/eval/run_agent.py"),
}


def ourmem_fingerprint(repo_root: Path = REPO_ROOT, *, official_roots: dict[str, Path] | None = None) -> dict:
    return method_fingerprint("ourmem", repo_root, official_roots=official_roots)


def method_fingerprint(method: str, repo_root: Path = REPO_ROOT, *, official_roots: dict[str, Path] | None = None) -> dict:
    """只读、纯标准库；验证脚本和正式运行器共享同一份指纹定义。"""
    if method not in {"ourmem", "amem"}:
        raise ValueError(f"Unknown memory method: {method}")
    specific = {"membase/configs/ourmem.py", "membase/layers/ourmem.py", "envs/ourmem_requirements.txt"}
    common = set(_LOCAL_FILES) - specific
    if method == "ourmem":
        specific.update(path.relative_to(repo_root).as_posix() for path in (repo_root / "membase/ourmem").glob("*.py"))
    else:
        specific = {"membase/configs/amem.py", "membase/layers/amem.py", "envs/amem_mab_requirements.txt",
                    "membase/baselines/amem/UPSTREAM.md"}
        specific.update(path.relative_to(repo_root).as_posix() for path in (repo_root / "membase/baselines/amem").glob("*.py"))
    paths = {name: repo_root / name for name in common | specific}
    for benchmark, upstream in (official_roots or {}).items():
        for name in _OFFICIAL_FILES[benchmark]:
            paths[f"official/{benchmark}/{name}"] = upstream / name
        if benchmark == "memoryagentbench":
            for path in (upstream / "configs/data_conf/Conflict_Resolution").glob("*.yaml"):
                paths[f"official/{benchmark}/{path.relative_to(upstream).as_posix()}"] = path
    files = {name: sha256_file(path) for name, path in sorted(paths.items())}
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {"schema_version": 1, "sha256": hashlib.sha256(encoded).hexdigest(), "files": files}
