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
from ..utils.ourmem_version import ImplementationMismatchError, ourmem_fingerprint, method_fingerprint

PROTOCOL_VERSION = "membase-three-stage-v3"
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
        for key in ("parallel_jobs", "judge_workers"):
            values.pop(key)
        if self.baseline == "ourmem":
            values.pop("workers")
            values.pop("check_workers")
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


def protocol_for(config, overrides):
    return {**fingerprint(config.benchmark, config.data_root, config.upstream_dir),
            "workflow": PROTOCOL_VERSION,
            "implementation": (ourmem_fingerprint(official_roots={config.benchmark: config.upstream_dir})
                               if config.baseline == "ourmem" else method_fingerprint("amem", official_roots={config.benchmark: config.upstream_dir})),
            "memory_overrides": ({k: v for k, v in overrides.items() if k not in {"request_timeout", "transport_retry_window"}}
                                 if config.baseline == "ourmem" else overrides)}


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
              "memory": ({"storage": "independent SQLite per sample", "max_claim_depth": overrides.get("max_claim_depth", 5),
                          "top_k": config.top_k, "reader": "single hybrid retrieval with grouped versions and support expansion"}
                         if config.baseline == "ourmem" else {"storage": "isolated Chroma + atomic state checkpoint",
                             "checkpoint_interval": overrides.get("checkpoint_interval", 8), "evo_threshold": overrides.get("evo_threshold", 100),
                             "top_k": config.top_k, "max_evidence_tokens": overrides.get("max_evidence_tokens"),
                             "llm_temperature": overrides.get("llm_temperature", 0.7),
                             "llm_max_output_tokens": overrides.get("llm_max_output_tokens"),
                             "llm_max_input_tokens": overrides.get("llm_max_input_tokens")}),
              "scoring": "pinned official protocol",
              "execution": {"workers": config.workers, "check_workers": config.check_workers}}
    if (config.run_dir / "read_revision.json").exists():
        from ..utils.read_revision import validate_revision
        saved = read_json(config.run_dir / "config.json")
        if saved["config"] != config.saved_config():
            raise ImplementationMismatchError("续跑配置与原运行不一致")
        revision = validate_revision(config.run_dir, saved, protocol_for(config, overrides), stages)
        result["reuse_construction"] = {"revision": revision["id"], "samples": sorted(revision["construction"]),
                                       "memory_unchanged": True, "construction_will_run": False}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


class RunContext:
    def __init__(self, config, *, client=None, layer_factory=None, stages=("construction", "search", "evaluation")):
        self.config, self.layer_factory = config, layer_factory
        self.layer_name = {"ourmem": "OurMem", "amem": "A-MEM"}[config.baseline]
        if config.baseline == "amem" and config.benchmark != "memoryagentbench":
            raise ValueError("A-MEM official three-stage adapter currently supports MAB only")
        self.blocked_samples = set()
        self.stage_failures = []
        self.stop_requested = False
        self.overrides = read_json(config.memory_config) if config.memory_config else {}
        protocol = protocol_for(config, self.overrides)
        self.read_revision = None
        if set(self.overrides) & {"api_key", "api_keys", "base_url", "base_urls", "llm_api_key", "embedding_api_key",
                                 "llm_base_url", "embedding_base_url"}:
            raise ValueError("方法配置不得包含接口凭据")
        path = config.run_dir / "config.json"
        if path.exists():
            saved = read_json(path)
            previous = saved.get("protocol", {})
            if previous.get("workflow") != PROTOCOL_VERSION:
                raise ImplementationMismatchError("旧运行保持只读；三阶段流程需要新的 RUN_ID")
            if (config.run_dir / "read_revision.json").exists():
                from ..utils.read_revision import validate_revision
                self.read_revision = validate_revision(config.run_dir, saved, protocol, stages)
                # 原 config.json 继续描述真实的构建版本；读取版本单独记录，不能冒充重建。
                protocol = {**protocol, "implementation": previous["implementation"]}
            elif previous.get("implementation") != protocol["implementation"]:
                raise ImplementationMismatchError("源码内容发生变化，请使用新的 RUN_ID；旧产物不变")
            if saved.get("config") != config.saved_config() or previous != protocol:
                raise ImplementationMismatchError("RUN_ID 对应的配置不同，请使用新的 RUN_ID；旧产物不变")
        from ..datasets import DATASET_MAPPING
        from ..configs import CONFIG_MAPPING
        from ..inference_utils.model_client import ModelClient, ModelClientConfig, RequestBudget
        self.dataset_cls = DATASET_MAPPING[DATASET_NAMES[config.benchmark]]
        api_key = os.environ.get(config.api_key_env, "")
        service_fields = {"embedding_base_url": config.embedding_base_url,
                          "embedding_api_key": os.environ.get(config.embedding_api_key_env or config.api_key_env, ""),
                          "judge_base_url": config.judge_base_url,
                          "judge_api_key": os.environ.get(config.judge_api_key_env or config.api_key_env, "")}
        if client is None:
            from ..configs.model_profiles import credential
            credential(config.api_key_env)
            credential(config.embedding_api_key_env or config.api_key_env)
            if config.benchmark in {"meme", "longmemeval"} or config.locomo_judge:
                credential(config.judge_api_key_env or config.api_key_env)
        self.call_config = ModelClientConfig(model_name=config.internal_model, answer_model=config.answer_model,
                                             judge_model=config.judge_model, seed=config.seed,
                                             embedding_model_name=config.embedding_model, api_key=api_key, base_url=config.base_url,
                                             **service_fields)
        if config.baseline == "ourmem":
            self.memory_config = CONFIG_MAPPING["OurMem"](**{
                **self.overrides, "top_k": config.top_k, "model_name": config.internal_model, "answer_model": config.answer_model,
                "judge_model": config.judge_model, "seed": config.seed,
                "embedding_model_name": config.embedding_model, "api_key": api_key, "base_url": config.base_url,
                **service_fields})
            self.call_config = self.memory_config
        else:
            self.memory_config = CONFIG_MAPPING["A-MEM"](**{
                **self.overrides, "user_id": "default", "llm_backend": "openai", "llm_model": config.internal_model,
                "llm_api_key": api_key, "llm_base_url": config.base_url, "embedding_provider": "openai",
                "retriever_name_or_path": config.embedding_model, "embedding_api_key": service_fields["embedding_api_key"],
                "embedding_base_url": config.embedding_base_url or config.base_url, "preserve_unknown_time": True})
            self.call_config = self.call_config.model_copy(update={
                "max_context_tokens": self.memory_config.llm_max_input_tokens})
        self.owns_client = client is None
        start_run(config.run_dir, config.saved_config(), protocol)
        if config.baseline == "ourmem":
            write_json(config.run_dir / "execution.json", {"workers": config.workers, "check_workers": config.check_workers,
                "request_timeout": self.call_config.request_timeout,
                "transport_retry_window": self.call_config.transport_retry_window})
        self.client = client or ModelClient(self.call_config, budget=RequestBudget(
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
        from ..inference_utils.model_client import ModelClient
        directory = self.prepare_sample(sample)
        client = ModelClient(self.call_config, budget=self.client.budget, log_path=directory / "requests.jsonl") if self.owns_client else self.client
        config = self.memory_config.model_copy(update={"user_id": sample.namespace, "save_dir": str(directory / "memory")})
        factory = self.layer_factory or MEMORY_LAYERS_MAPPING[self.layer_name].from_config
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
        from ..inference_utils.model_client import BudgetExceeded, TransportUnavailable
        from openai import APIStatusError
        write_json(self.config.run_dir / "status.json", {"status": "running", "stage": stage})
        completed, exhausted = [], False
        iterator = iter(self.samples())
        with ThreadPoolExecutor(max_workers=self.config.workers) as pool:
            pending = {}
            while pending or not exhausted:
                while not exhausted and len(pending) < self.config.workers:
                    sample = next(iterator, None)
                    if sample is None:
                        exhausted = True
                    elif sample.key in self.blocked_samples:
                        continue
                    else:
                        print(f"[{sample.key}] {stage} 开始/续跑", flush=True)
                        pending[pool.submit(self.process_sample, stage, worker, sample)] = sample
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
                        self.blocked_samples.add(sample.key)
                        detail = str(error)
                        if os.environ.get("OPENAI_API_KEY"):
                            detail = detail.replace(os.environ["OPENAI_API_KEY"], "[REDACTED]")
                        self.stage_failures.append({"sample": sample.key, "stage": stage,
                                                    "error_type": type(error).__name__, "reason": detail})
                        print(f"[{sample.key}] {stage} 未完成：{detail}", flush=True)
                        if isinstance(error, (BudgetExceeded, TransportUnavailable, APIStatusError, OSError, ImportError)):
                            exhausted = True
                            self.stop_requested = True
                            if isinstance(error, TransportUnavailable):
                                write_json(self.config.run_dir / "paused.json", {"reason": "transport_unavailable", "sample": sample.key})
                            if isinstance(error, BudgetExceeded):
                                write_json(self.config.run_dir / "budget_stop.json", {"status": "stopped", "reason": "request_budget_exhausted"})
        if self.stage_failures:
            write_json(self.config.run_dir / "failures.json", self.stage_failures)
        if not completed and not self.stage_failures:
            raise ValueError("没有选中任何样本")
        if stage == "evaluation" and not self.stage_failures:
            result = summarize(self.config.benchmark, completed)
            if self.read_revision:
                result["read_revision"] = {"id": self.read_revision["id"], "construction_reused": True,
                                           "prior_query_costs_included": True}
            result.update(selected_mode=self.config.mode, request_costs=self.costs())
            finish_run(self.config.run_dir, result)
            return result
        write_json(self.config.run_dir / "status.json", {"status": "stage_incomplete" if self.stage_failures else "stage_complete",
                                                         "completed_stage": stage})
        return completed

    def record_incomplete(self):
        """只列覆盖情况，不把未完成样本排除后计算貌似完整的成绩。"""
        coverage = []
        for sample in self.samples():
            for phase in sample.phases:
                folder = self.directory(sample) / phase.name
                # 缺项清单是文件级进度；完整成绩仍须通过正式校验。
                answered, failed, unprocessed, missing_retrievals, missing_scores = [], [], [], [], []
                for question in phase.questions:
                    if question_file(folder / "failures", question.id).is_file():
                        failed.append(question.id)
                        continue
                    if question_file(folder / "answers", question.id).is_file():
                        answered.append(question.id)
                    else:
                        unprocessed.append(question.id)
                    if not question_file(folder / "retrievals", question.id).is_file():
                        missing_retrievals.append(question.id)
                    score_exists = ((self.directory(sample) / "official_judge.json").is_file() if self.config.benchmark == "meme"
                                    else question_file(folder / "scores", question.id).is_file())
                    if not score_exists:
                        missing_scores.append(question.id)
                coverage.append({"sample": sample.key, "phase": phase.name,
                                 "expected_question_ids": [q.id for q in phase.questions],
                                 "answer_artifacts": answered, "failure_artifacts": failed,
                                 "missing_answer_ids": unprocessed,
                                 "missing_retrieval_ids": missing_retrievals,
                                 "missing_score_ids": missing_scores,
                                 "evaluation_complete": (self.directory(sample) / "evaluation.json").is_file()})
        write_json(self.config.run_dir / "partial_summary.json", {"status": "incomplete", "score": None,
                                                                  "coverage": coverage, "failures": self.stage_failures})

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
    context = RunContext(config, client=client, layer_factory=layer_factory, stages=stages)
    runners = {"construction": ConstructionRunner, "search": SearchRunner, "evaluation": EvaluationRunner}
    try:
        result = None
        for stage in stages:
            result = runners[stage](None, runtime=context).run()
            if context.stop_requested:
                break
        if context.stage_failures:
            raise RuntimeError(f"实验仍有 {len(context.blocked_samples)} 个样本未完成；详见 failures.json 和 partial_summary.json")
        return result
    except BaseException as error:
        context.record_incomplete()
        fail_run(config.run_dir, error)
        raise
    finally:
        context.close()
