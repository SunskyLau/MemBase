"""OurMem 的官方协议运行：样本隔离、阶段续跑、逐题保存和完整性验收。"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import time

from .benchmark import BenchmarkRunConfig
from ..datasets import memoryagentbench as mab, meme
from ..datasets.ourmem_benchmarks import (QA_DATA, Episode, episode_manifest,
                                         fingerprint, load_episodes)
from ..evaluation.ourmem import Scorer, answer_prompt, answer_system, summarize, summarize_run_costs, validate_answers, validate_scores
from ..utils.benchmark_files import read_json, sha256_file, write_json
from ..utils.experiment import start_run, fail_run, finish_run
from ..utils.ourmem_version import ImplementationMismatchError, ourmem_fingerprint


@dataclass(frozen=True)
class OurMemRunConfig(BenchmarkRunConfig):
    embedding_model: str = "text-embedding-3-small"
    memory_config: Path | None = None
    seed: int = 0
    max_llm_requests: int | None = None
    max_embedding_requests: int | None = None
    budget_ledger: Path | None = None
    locomo_judge: bool = False

    def saved_config(self) -> dict:
        values = super().saved_config()
        # 通用基线的单一 Top-k 等参数不控制 OurMem，避免配置记录造成误解。
        for unused in ("top_k", "parallel_jobs", "judge_workers"):
            values.pop(unused)
        values["temperature"] = self.temperature if self.benchmark == "memoryagentbench" else 0
        return values


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _jsonable(value):
    return value.model_dump(mode="json") if hasattr(value, "model_dump") else value


def preview(config: OurMemRunConfig) -> dict:
    """干运行只列执行矩阵，不导入模型、创建运行目录或 SQLite 文件。"""
    if config.benchmark == "memoryagentbench":
        manifest = mab.load_manifest(config.data_root)
        entries = [{"sample": key, "phases": [{"name": "final", "questions": 4 if config.mode == "smoke" else len(manifest["questions"][key])}],
                    "history": "complete selected subset"} for key in mab.subsets(config.mode)]
    else:
        entries = [{"sample": ep.key, "messages": sum(map(len, ep.sessions)),
                    "phases": [{"name": p.name, "session_end": p.session_end, "questions": len(p.questions)} for p in ep.phases]}
                   for ep in load_episodes(config.benchmark, config.data_root, config.mode)]
    overrides = read_json(config.memory_config) if config.memory_config else {}
    result = {"dry_run": True, "config": config.saved_config(), "matrix": entries,
              "memory": {"storage": "independent SQLite per sample", "max_claim_depth": overrides.get("max_claim_depth", 5),
                         "config_override": str(config.memory_config) if config.memory_config else None},
              "scoring": "pinned official protocol; LoCoMo optional extra judge is separate"}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def _checked(path: Path, expected: dict, label: str) -> dict:
    value = read_json(path)
    if any(value.get(key) != data for key, data in expected.items()):
        raise ValueError(f"{label} 与本次输入或快照不一致：{path}")
    return value


def _question_file(folder: Path, question_id: str) -> Path:
    return folder / f"{hashlib.sha256(question_id.encode()).hexdigest()[:24]}.json"


def _run_episode(config: OurMemRunConfig, episode: Episode, client, memory_config,
                 system_factory=None) -> dict:
    from ..ourmem.models import InputMessage, InputPolicy
    if system_factory is None:
        from ..ourmem.system import OurMemSystem
        system_factory = OurMemSystem
    directory = config.run_dir / "samples" / episode.key
    signature = episode_manifest(episode)
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        if read_json(manifest_path) != signature:
            raise ValueError(f"样本输入改变，不能续跑：{episode.key}")
    else:
        write_json(manifest_path, signature)
    write_json(directory / "source_mapping.json", episode.source_mapping)
    checkpoint_path = directory / "checkpoint.json"
    checkpoint = read_json(checkpoint_path) if checkpoint_path.exists() else {"sessions_ingested": 0, "phases": {}}
    system = system_factory(config=memory_config, storage_dir=directory / "memory", client=client)
    scorer = Scorer(config.benchmark, config.upstream_dir, client, config.judge_model,
                    directory / "judge_calls", locomo_judge=config.locomo_judge)
    all_answers, phase_rows = [], {}
    try:
        for phase in episode.phases:
            phase_dir = directory / phase.name
            if checkpoint["sessions_ingested"] > phase.session_end and phase.name not in checkpoint["phases"]:
                raise ValueError(f"缺少 {phase.name} 快照但已经摄入后续信息；请保留旧产物并新建运行")
            while checkpoint["sessions_ingested"] < phase.session_end:
                index = checkpoint["sessions_ingested"]
                started = time.monotonic()
                messages = [InputMessage(**message) for message in episode.sessions[index]]
                system.ingest(messages, namespace=episode.namespace, input_policy=InputPolicy(**episode.input_policy))
                cutoff = sum(len(s) for s in episode.sessions[:index + 1]) - 1
                snapshot_id = system.flush(namespace=episode.namespace, source_cutoff=cutoff)
                checkpoint["sessions_ingested"] = index + 1
                checkpoint["last_snapshot_id"] = snapshot_id
                write_json(checkpoint_path, checkpoint)
                write_json(directory / "ingest" / f"{index:06d}.json",
                           {"session_index": index, "snapshot_id": snapshot_id,
                            "seconds": time.monotonic() - started})
            if phase.name not in checkpoint["phases"]:
                checkpoint["phases"][phase.name] = {"snapshot_id": checkpoint["last_snapshot_id"]}
                write_json(checkpoint_path, checkpoint)
            snapshot_id = checkpoint["phases"][phase.name]["snapshot_id"]
            snapshot_path = phase_dir / "memory_snapshot.json"
            snapshot = system.get_memory_snapshot(episode.namespace, snapshot_id)
            if not snapshot_path.exists():
                write_json(snapshot_path, snapshot)
            rows = []
            for question in phase.questions:
                path = _question_file(phase_dir / "answers", question.id)
                identity = {"question_id": question.id, "question": question.text, "snapshot_id": snapshot_id}
                if path.exists():
                    record = _checked(path, identity, "回答")
                    validate_answers([record], (question,))
                else:
                    # 不能在变化后的状态首次生成变化前答案。
                    if checkpoint["sessions_ingested"] > phase.session_end:
                        raise ValueError(f"缺少早期回答 {question.id}，但已经摄入未来会话；需使用新运行目录")
                    started = time.monotonic()
                    prepared = system.prepare_evidence(question.text, namespace=episode.namespace,
                                                       snapshot_id=snapshot_id, query_time=question.query_time)
                    prepared_data = _jsonable(prepared)
                    prompt, answer_limit = answer_prompt(config.benchmark, config.upstream_dir,
                                                          episode, question, prepared_data["context"])
                    answer = client.text(prompt, stage="answer", model=config.answer_model,
                                         temperature=config.temperature if config.benchmark == "memoryagentbench" else 0,
                                         max_tokens=answer_limit, allow_truncated=True,
                                         system=answer_system(config.benchmark, config.upstream_dir))
                    record = {**identity, **prepared_data, "answer_text": answer,
                              "answer_finish_reason": getattr(answer, "finish_reason", "stop"),
                              "answer_request_id": getattr(answer, "request_id", None),
                              "answer_seconds": time.monotonic() - started}
                    if record["answer_finish_reason"] == "length":
                        record.update(resolution_status="incomplete", reason="answer_output_truncated")
                    validate_answers([record], (question,))
                    write_json(path, record)
                rows.append(record)
            validate_answers(rows, phase.questions)
            phase_rows[phase.name] = rows
            # 全部问题写盘后才允许摄入下一阶段，不让评分失败丢失昂贵回答。
            write_json(phase_dir / "answers.json", rows)
            for question, record in zip(phase.questions, rows):
                if config.benchmark == "meme":
                    continue
                score_path = _question_file(phase_dir / "scores", question.id)
                identity = {"answer_sha256": _digest(record), "judge_model": config.judge_model}
                already_saved = score_path.exists()
                if already_saved:
                    score = _checked(score_path, identity, "评分")
                else:
                    score = {**identity, "scores": scorer.score(question, record["answer_text"])}
                validate_scores(config.benchmark, score["scores"], locomo_judge=config.locomo_judge)
                if not already_saved:
                    write_json(score_path, score)
                all_answers.append({**record, "scores": score["scores"]})
        output = {"sample": episode.key, "manifest": signature, "answers": all_answers}
        if config.benchmark == "meme":
            raw = episode.reference
            official = {"episode_id": raw["episode_id"], "domain": raw["domain"], "root": raw.get("root", ""),
                        "config": {"agent_type": "ourmem", "agent_model": config.answer_model,
                                   "internal_model": config.internal_model},
                        "memory_snapshots": {f"{phase.name}_questions": read_json(directory / phase.name / "memory_snapshot.json")["text"] for phase in episode.phases}}
            for phase in episode.phases:
                official[f"{phase.name}_answers"] = [
                    {**question.reference, "agent_answer": row["answer_text"], "retrieved_context": row["context"],
                     "read_audit": {k: row[k] for k in ("resolution_status", "reason", "coverage", "read_trace")}}
                    for question, row in zip(phase.questions, phase_rows[phase.name])]
            write_json(directory / "official_answers.json", official)
            judge_path = directory / "official_judge.json"
            receipt_path = directory / "judge_receipt.json"
            identity = {"answer_sha256": _digest(official), "judge_model": config.judge_model}
            if judge_path.exists() and receipt_path.exists() and read_json(receipt_path) == {
                    **identity, "judge_sha256": sha256_file(judge_path)}:
                judged = read_json(judge_path)
            else:
                judged = scorer.score_meme(official, config.check_workers)
                write_json(judge_path, judged)
                write_json(receipt_path, {**identity, "judge_sha256": sha256_file(judge_path)})
            from ..evaluation.meme import validate_judge
            validate_judge(judge_path, official, config.judge_model)
            output["judge"] = judged
        write_json(directory / "result.json", output)
        write_json(directory / "status.json", {"status": "complete"})
        return output
    except BaseException as exc:
        write_json(directory / "status.json", {"status": "incomplete", "error_type": type(exc).__name__})
        raise
    finally:
        system.close()


def run(config: OurMemRunConfig, *, client=None, system_factory=None) -> dict:
    if config.dry_run:
        return preview(config)
    protocol = fingerprint(config.benchmark, config.data_root, config.upstream_dir)
    overrides = read_json(config.memory_config) if config.memory_config else {}
    if set(overrides) & {"api_key", "api_keys", "base_url", "base_urls"}:
        raise ValueError("memory-config 只允许方法参数，接口凭据应由环境和运行参数提供")
    # 运行恢复同时固定提示和实现内容，防止修改代码后复用旧方法产物。
    protocol["implementation"] = ourmem_fingerprint(official_roots={config.benchmark: config.upstream_dir})
    protocol["memory_overrides"] = overrides
    protocol["official_import_adaptations"] = {
        "locomo": "Skip unused bert_score import for native F1; official function bodies unchanged",
        "memoryagentbench": "Load official normalization, substring and max-over-golds functions only",
        "longmemeval": "Load official judge prompt function and plain-history answer string only",
        "meme": "Reuse task judge functions and trivial-pass filtering; shared request transport",
    }
    saved_manifest = config.run_dir / "config.json"
    if saved_manifest.exists():
        previous = read_json(saved_manifest).get("protocol", {}).get("implementation")
        if previous != protocol["implementation"]:
            raise ImplementationMismatchError("OurMem 或官方适配源码内容发生变化，不能复用旧运行；请使用新的 RUN_ID，旧产物保持不变")
    start_run(config.run_dir, config.saved_config(), protocol)
    from ..configs.ourmem import OurMemConfig
    from ..ourmem.llm import BudgetExceeded, ModelClient, RequestBudget

    memory_config = OurMemConfig(**{**overrides, "model_name": config.internal_model,
                                   "answer_model": config.answer_model,
                                   "judge_model": config.judge_model,
                                   "seed": config.seed,
                                   "api_key": os.environ.get("OPENAI_API_KEY", ""), "base_url": config.base_url,
                                   "embedding_model_name": config.embedding_model})
    owns_client = client is None
    if owns_client:
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("请提供 OPENAI_API_KEY；凭据不会写入配置")
        budget = RequestBudget(max_llm_requests=config.max_llm_requests,
                               max_embedding_requests=config.max_embedding_requests,
                               ledger_path=config.budget_ledger or config.run_dir / "budget.sqlite")
        client = ModelClient(config=memory_config, budget=budget, log_path=config.run_dir / "requests.jsonl")
    episodes = iter(load_episodes(config.benchmark, config.data_root, config.mode))
    completed, failures = [], []
    def execute(ep):
        print(f"[{ep.key}] 开始/续跑", flush=True)
        sample_client = (ModelClient(config=memory_config, budget=client.budget,
                                     log_path=config.run_dir / "samples" / ep.key / "requests.jsonl")
                         if owns_client else client)
        result = _run_episode(config, ep, sample_client, memory_config, system_factory)
        print(f"[{ep.key}] 完成", flush=True)
        # 汇总只需要问题标签；不让已处理的长对话常驻内存。
        compact = replace(ep, sessions=(), reference={"domain": ep.reference.get("domain")}, source_mapping={})
        return compact, result
    try:
        selected_count = 0
        with ThreadPoolExecutor(max_workers=config.workers) as pool:
            futures = {}
            exhausted = False
            while futures or not exhausted:
                while not exhausted and len(futures) < config.workers:
                    ep = next(episodes, None)
                    if ep is None:
                        exhausted = True
                    else:
                        futures[pool.submit(execute, ep)] = ep.key
                        selected_count += 1
                if not futures:
                    break
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    key = futures.pop(future)
                    try:
                        completed.append(future.result())
                    except Exception as exc:
                        failures.append({"sample": key, "error_type": type(exc).__name__})
                        if isinstance(exc, BudgetExceeded):
                            exhausted = True
                            write_json(config.run_dir / "budget_stop.json", {
                                "status": "stopped", "reason": "request_budget_exhausted",
                                "remaining_samples_not_started": True})
        if failures or len(completed) != selected_count:
            write_json(config.run_dir / "failures.json", failures)
            raise RuntimeError(f"{len(failures)} 个样本未完成；不计算删减分母后的全量结果")
        result = summarize(config.benchmark, completed)
        result["selected_mode"] = config.mode
        result["request_costs"] = summarize_run_costs(
            config.run_dir, client.budget.summary(), exclusive_ledger=owns_client and config.budget_ledger is None)
        finish_run(config.run_dir, result)
        return result
    except BaseException as exc:
        fail_run(config.run_dir, exc)
        raise
    finally:
        write_json(config.run_dir / "costs.json", summarize_run_costs(
            config.run_dir, client.budget.summary(), exclusive_ledger=owns_client and config.budget_ledger is None))
        if owns_client:
            client.budget.close()
