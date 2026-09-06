"""三阶段共用的运行上下文：配置、产物、并发及恢复；不实现记忆方法。"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path

from .benchmark import BenchmarkRunConfig
from ..datasets.official import episode_manifest, fingerprint
from ..evaluation.official import Scorer, summarize, summarize_run_costs
from ..utils.benchmark_files import read_json, write_json, sha256_file
from ..utils.experiment import start_run, fail_run, finish_run
from ..utils.ourmem_version import ImplementationMismatchError, ourmem_fingerprint

PROTOCOL_VERSION = "membase-three-stage-v1"
DATASET_NAMES = {"locomo": "LoCoMo", "longmemeval": "LongMemEval", "memoryagentbench": "MemoryAgentBench", "meme": "MEME"}


@dataclass(frozen=True)
class OfficialRunConfig(BenchmarkRunConfig):
    embedding_model: str = "text-embedding-3-small"
    memory_config: Path | None = None
    seed: int = 0
    max_llm_requests: int | None = None
    max_embedding_requests: int | None = None
    budget_ledger: Path | None = None
    locomo_judge: bool = False

    def saved_config(self):
        values = super().saved_config()
        for key in ("top_k", "parallel_jobs", "judge_workers"):
            values.pop(key)
        values["temperature"] = self.temperature if self.benchmark == "memoryagentbench" else 0
        return values


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def checked(path: Path, expected: dict, label: str):
    value = read_json(path)
    if not isinstance(value, dict) or any(value.get(key) != item for key, item in expected.items()):
        raise ValueError(f"{label} 与本次输入或快照不一致：{path}")
    return value


def question_file(directory: Path, question_id: str):
    return directory / f"{hashlib.sha256(question_id.encode()).hexdigest()[:24]}.json"


def preview(config, stages=("construction", "search", "evaluation")):
    # 原始输入保持完整；不导入运行器、模型或创建运行目录。
    from ..datasets.official import load_episodes
    from ..datasets import memoryagentbench
    if config.benchmark == "memoryagentbench":
        questions = memoryagentbench.load_manifest(config.data_root)["questions"]
        matrix = [{"sample": key, "phases": [{"name": "final", "questions": 4 if config.mode == "smoke" else len(questions[key])}],
                   "history": "complete selected subset"} for key in memoryagentbench.subsets(config.mode)]
    else:
        matrix = [{"sample": s.key, "messages": sum(map(len, s.sessions)),
                   "phases": [{"name": p.name, "session_end": p.session_end, "questions": len(p.questions)} for p in s.phases]}
                  for s in load_episodes(config.benchmark, config.data_root, config.mode)]
    overrides = read_json(config.memory_config) if config.memory_config else {}
    result = {"dry_run": True, "config": config.saved_config(), "matrix": matrix,
              "stages": list(stages), "protocol": PROTOCOL_VERSION,
              "memory": {"storage": "independent SQLite per sample", "max_claim_depth": overrides.get("max_claim_depth", 5)},
              "scoring": "pinned official protocol"}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


class RunContext:
    def __init__(self, config, *, client=None, layer_factory=None):
        self.config, self.layer_factory = config, layer_factory
        protocol = {**fingerprint(config.benchmark, config.data_root, config.upstream_dir),
                    "workflow": PROTOCOL_VERSION,
                    "implementation": ourmem_fingerprint(official_roots={config.benchmark: config.upstream_dir})}
        self.overrides = read_json(config.memory_config) if config.memory_config else {}
        if set(self.overrides) & {"api_key", "api_keys", "base_url", "base_urls"}:
            raise ValueError("方法配置不得包含接口凭据")
        protocol["memory_overrides"] = self.overrides
        path = config.run_dir / "config.json"
        if path.exists():
            saved = read_json(path)
            previous = saved.get("protocol", {})
            if previous.get("workflow") != PROTOCOL_VERSION:
                raise ImplementationMismatchError("旧运行保持只读；三阶段流程需要新的 RUN_ID")
            if previous.get("implementation") != protocol["implementation"]:
                raise ImplementationMismatchError("源码内容发生变化，请使用新的 RUN_ID；旧产物不变")
            if saved.get("config") != config.saved_config() or previous != protocol:
                raise ImplementationMismatchError("RUN_ID 对应的配置不同，请使用新的 RUN_ID；旧产物不变")
        from ..datasets import DATASET_MAPPING
        from ..configs import CONFIG_MAPPING
        from ..ourmem.llm import ModelClient, RequestBudget
        self.dataset_cls = DATASET_MAPPING[DATASET_NAMES[config.benchmark]]
        self.memory_config = CONFIG_MAPPING["OurMem"](**{
            **self.overrides, "model_name": config.internal_model, "answer_model": config.answer_model,
            "judge_model": config.judge_model, "seed": config.seed,
            "embedding_model_name": config.embedding_model, "api_key": os.environ.get("OPENAI_API_KEY", ""), "base_url": config.base_url})
        self.owns_client = client is None
        if self.owns_client and not self.memory_config.api_key:
            raise ValueError("请提供 OPENAI_API_KEY")
        start_run(config.run_dir, config.saved_config(), protocol)
        self.client = client or ModelClient(self.memory_config, budget=RequestBudget(
            config.max_llm_requests, config.max_embedding_requests, config.budget_ledger or config.run_dir / "budget.sqlite"),
            log_path=config.run_dir / "requests.jsonl")

    def samples(self):
        return self.dataset_cls.iter_official(self.config.data_root, self.config.mode)

    def directory(self, sample):
        return self.config.run_dir / "samples" / sample.key

    def prepare_sample(self, sample):
        directory, signature = self.directory(sample), episode_manifest(sample)
        path = directory / "manifest.json"
        if path.exists() and read_json(path) != signature:
            raise ValueError(f"样本输入改变，不能续跑：{sample.key}")
        write_json(path, signature)
        write_json(directory / "source_mapping.json", sample.source_mapping)
        return directory

    @contextmanager
    def sample_scope(self, sample, *, load=False):
        from ..layers import MEMORY_LAYERS_MAPPING
        from ..ourmem.llm import ModelClient
        directory = self.prepare_sample(sample)
        client = ModelClient(self.memory_config, budget=self.client.budget, log_path=directory / "requests.jsonl") if self.owns_client else self.client
        config = self.memory_config.model_copy(update={"user_id": sample.namespace, "save_dir": str(directory / "memory")})
        factory = self.layer_factory or MEMORY_LAYERS_MAPPING["OurMem"].from_config
        layer = factory(config, client=client)
        try:
            if load and not layer.load_memory(sample.namespace):
                raise ValueError(f"缺少已构建记忆：{sample.key}")
            yield layer, client
        finally:
            layer.cleanup()
            if self.owns_client:
                client.close()

    def scorer(self, sample, client):
        return Scorer(self.config.benchmark, self.config.upstream_dir, client, self.config.judge_model,
                      self.directory(sample) / "judge_calls", locomo_judge=self.config.locomo_judge)

    def checkpoint(self, sample):
        path = self.directory(sample) / "checkpoint.json"
        return read_json(path) if path.exists() else {"sessions_ingested": 0, "phases": {}}

    def require_snapshot(self, sample, phase):
        point = self.checkpoint(sample)["phases"][phase.name]
        path = self.directory(sample) / phase.name / "memory_snapshot.json"
        if not path.is_file() or point.get("snapshot_sha256") != sha256_file(path):
            raise ValueError(f"观察点快照缺失或损坏：{sample.key}/{phase.name}")
        return point["snapshot_id"]

    def require_stage(self, sample, stage):
        return checked(self.directory(sample) / f"{stage}.json", {"complete": True, "manifest": episode_manifest(sample)}, stage)

    def finish_stage(self, sample, stage):
        write_json(self.directory(sample) / f"{stage}.json", {"complete": True, "manifest": episode_manifest(sample)})

    def execute(self, stage, worker):
        from ..ourmem.llm import BudgetExceeded
        write_json(self.config.run_dir / "status.json", {"status": "running", "stage": stage})
        completed, failures, count, exhausted = [], [], 0, False
        iterator = iter(self.samples())
        with ThreadPoolExecutor(max_workers=self.config.workers) as pool:
            pending = {}
            while pending or not exhausted:
                while not exhausted and len(pending) < self.config.workers:
                    sample = next(iterator, None)
                    if sample is None:
                        exhausted = True
                    else:
                        print(f"[{sample.key}] {stage} 开始/续跑", flush=True)
                        pending[pool.submit(self.process_sample, stage, worker, sample)] = sample
                        count += 1
                if not pending:
                    break
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    sample = pending.pop(future)
                    try:
                        result = future.result()
                        completed.append((replace(sample, sessions=(), reference={"domain": sample.reference.get("domain")}, source_mapping={}), result))
                        print(f"[{sample.key}] {stage} 完成", flush=True)
                    except Exception as error:
                        failures.append({"sample": sample.key, "stage": stage, "error_type": type(error).__name__})
                        if isinstance(error, BudgetExceeded):
                            exhausted = True
                            write_json(self.config.run_dir / "budget_stop.json", {"status": "stopped", "reason": "request_budget_exhausted"})
        if failures or len(completed) != count:
            write_json(self.config.run_dir / "failures.json", failures)
            raise RuntimeError(f"{stage} 阶段有 {len(failures)} 个样本未完成")
        if not completed:
            raise ValueError("没有选中任何样本")
        if stage == "evaluation":
            result = summarize(self.config.benchmark, completed)
            result.update(selected_mode=self.config.mode, request_costs=self.costs())
            finish_run(self.config.run_dir, result)
            return result
        write_json(self.config.run_dir / "status.json", {"status": "stage_complete", "completed_stage": stage})
        return completed

    def process_sample(self, stage, worker, sample):
        try:
            return worker(sample)
        except BaseException as error:
            write_json(self.directory(sample) / "status.json", {"status": "incomplete", "stage": stage, "error_type": type(error).__name__})
            raise

    def costs(self):
        return summarize_run_costs(self.config.run_dir, self.client.budget.summary(),
                                   exclusive_ledger=self.owns_client and self.config.budget_ledger is None)

    def close(self):
        write_json(self.config.run_dir / "costs.json", self.costs())
        if self.owns_client:
            self.client.close()
            self.client.budget.close()


def run(config, *, stages=("construction", "search", "evaluation"), client=None, layer_factory=None):
    if config.dry_run:
        return preview(config, stages)
    from .construction import ConstructionRunner
    from .search import SearchRunner
    from .evaluation import EvaluationRunner
    context = RunContext(config, client=client, layer_factory=layer_factory)
    runners = {"construction": ConstructionRunner, "search": SearchRunner, "evaluation": EvaluationRunner}
    try:
        result = None
        for stage in stages:
            result = runners[stage](None, runtime=context).run()
        return result
    except BaseException as error:
        fail_run(config.run_dir, error)
        raise
    finally:
        context.close()
