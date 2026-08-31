"""Run the complete, resumable OurMem evaluation on LoCoMo."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from string import Template
from typing import Any, TextIO


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from openai import OpenAI
from smartcomment import disable_tracing

from membase.datasets.locomo import LoCoMo, _parse_evidence_ids
from membase.evaluation.bleu import BLEU
from membase.model_types.dataset import Message, QuestionAnswerPair
from membase.model_types.memory import MemoryEntry
from membase.ourmem.models import FactStatus
from membase.runners.construction import ConstructionRunner, ConstructionRunnerConfig
from membase.runners.evaluation import evaluate_memory
from membase.runners.search import memory_search

DEFAULT_DATASET_PATH = REPOSITORY_ROOT / "data" / "locomo10.json"
DEFAULT_API_CONFIG_PATH = REPOSITORY_ROOT / "examples" / "ourmem" / "api_config.json"
DEFAULT_RUNS_DIR = Path(__file__).resolve().parent / "runs"


class TeeStream:
    """Write console output to both the terminal and a persistent log."""

    def __init__(self, terminal: TextIO, log: TextIO) -> None:
        self.terminal = terminal
        self.log = log

    def write(self, text: str) -> int:
        self.terminal.write(text)
        self.log.write(text)
        self.log.flush()
        return len(text)

    def flush(self) -> None:
        self.terminal.flush()
        self.log.flush()

    def isatty(self) -> bool:
        return self.terminal.isatty()


def include_image_caption(message: Message) -> Message:
    """Append LoCoMo's released image description to the source message."""

    caption = message.metadata.get("blip_caption")
    if not caption:
        return message
    query = message.metadata.get("query")
    image_text = f"[Image shared by {message.name}: {caption}]"
    if query:
        image_text = f"{image_text} [Image query: {query}]"
    message.content = f"{message.content}\n{image_text}"
    return message


def keep_standard_questions(qa_pair: QuestionAnswerPair) -> bool:
    """Keep LoCoMo categories 1–4 and exclude adversarial questions."""

    return qa_pair.metadata.get("question_type") != "adversarial"


def get_ourmem_locomo_qa_prompt() -> Template:
    """Return a method-neutral LoCoMo answer prompt for structured memory."""

    return Template(
        "You answer questions using retrieved memories from a long conversation between "
        "two people. The memories may contain atomic facts and derived claims with their "
        "supporting evidence.\n\n"
        "Instructions:\n"
        "1. Use only the supplied memories.\n"
        "2. Keep speaker identities distinct.\n"
        "3. Pay close attention to mention times and event times.\n"
        "4. Resolve relative dates from the timestamp of the supporting memory.\n"
        "5. When information conflicts, use the currently valid and most recent evidence.\n"
        "6. Return only the concise answer, normally no more than 5–6 words.\n"
        "7. If the memories do not support an answer, say that you do not know.\n\n"
        "Memories:\n$context\n\nQuestion: $question\n\nAnswer:"
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_api_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    return {
        "api_key": config["api_keys"][0],
        "base_url": config["base_urls"][0],
        "model_name": config["model_name"],
        "embedding_model_name": config["embedding_model_name"],
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def git_status() -> list[str]:
    output = subprocess.run(
        ["git", "status", "--short"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [line for line in output.splitlines() if line]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def filtered_dataset(dataset_path: Path) -> LoCoMo:
    return LoCoMo.read_raw_data(str(dataset_path)).sample(
        question_filter=keep_standard_questions
    )


def expected_users(dataset: LoCoMo, start: int, end: int) -> list[str]:
    return [trajectory.id for trajectory, _ in zip(*dataset[start:end])]


def memory_config(
    api: dict[str, Any],
    run_dir: Path,
    message_batch_size: int,
) -> dict[str, Any]:
    return {
        "user_id": "guest",
        "save_dir": str(run_dir / "memory"),
        "model_name": api["model_name"],
        "embedding_model_name": api["embedding_model_name"],
        "api_key": api["api_key"],
        "base_url": api["base_url"],
        "fact_candidate_k": 8,
        "support_candidate_k": 12,
        "claim_candidate_k": 8,
        "fact_similarity_threshold": 0.45,
        "claim_similarity_threshold": 0.25,
        "message_batch_size": message_batch_size,
        "embedding_batch_size": 128,
        "max_induced_claims_per_fact": 4,
    }


def write_manifest(
    args: argparse.Namespace,
    api: dict[str, Any],
    run_dir: Path,
) -> None:
    manifest = {
        "created_at": utc_now(),
        "git_commit": git_commit(),
        "git_status": git_status(),
        "run_script_sha256": sha256(Path(__file__)),
        "dataset_path": str(args.dataset_path),
        "dataset_sha256": sha256(args.dataset_path),
        "scope": {
            "start_index": args.start_index,
            "end_index": args.end_index,
            "question_categories": [1, 2, 3, 4],
        },
        "models": {
            "memory_construction": api["model_name"],
            "answer_generation": api["model_name"],
            "llm_judge": api["model_name"],
            "embedding": api["embedding_model_name"],
        },
        "protocol": {
            "shared_memory_per_conversation": True,
            "include_blip_caption": True,
            "top_k": args.top_k,
            "message_batch_size": args.message_batch_size,
            "construction_workers": args.workers,
            "search_workers": args.workers,
            "evaluation_concurrency": args.evaluation_concurrency,
            "temperature": 0.0,
            "metrics": ["llm_judge", "f1", "bleu"],
        },
        "api_base_url": api["base_url"],
        "run_dir": str(run_dir),
    }
    write_json(run_dir / "manifest.json", manifest)


def update_stage_time(run_dir: Path, stage: str, started: float) -> None:
    path = run_dir / "stage_times.json"
    times = read_json(path) if path.exists() else {}
    times[stage] = {
        "completed_at": utc_now(),
        "wall_seconds": time.monotonic() - started,
    }
    write_json(path, times)


def run_preflight(
    api: dict[str, Any],
    dataset_path: Path,
    run_dir: Path,
) -> None:
    started = time.monotonic()
    raw_dataset = LoCoMo.read_raw_data(str(dataset_path))
    dataset = filtered_dataset(dataset_path)
    question_count = sum(len(questions) for _, questions in dataset)
    image_question_count = 0
    unresolved_evidence_ids = set()
    raw = read_json(dataset_path)
    for sample in raw:
        messages = {
            turn["dia_id"]: turn
            for key, turns in sample["conversation"].items()
            if key.startswith("session_") and key.removeprefix("session_").isdigit()
            for turn in turns
        }
        for qa in sample["qa"]:
            if qa["category"] == 5:
                continue
            evidence_ids = [
                evidence
                for group in qa["evidence"]
                for evidence in _parse_evidence_ids(group)
            ]
            unresolved_evidence_ids.update(
                item for item in evidence_ids if item not in messages
            )
            if any(
                messages[item].get("blip_caption")
                for item in evidence_ids
                if item in messages
            ):
                image_question_count += 1

    client = OpenAI(api_key=api["api_key"], base_url=api["base_url"])
    response = client.chat.completions.create(
        model=api["model_name"],
        messages=[{"role": "user", "content": "Reply with exactly: OK"}],
        temperature=0,
    )
    embedding = client.embeddings.create(
        model=api["embedding_model_name"],
        input=["agent memory"],
    )
    bleu_witness = BLEU().compute(["memory answer"], [["memory answer"]])[0][
        "value"
    ]
    result = {
        "completed_at": utc_now(),
        "conversations": len(raw_dataset),
        "questions_categories_1_4": question_count,
        "questions_with_image_evidence": image_question_count,
        "unresolved_evidence_ids": sorted(unresolved_evidence_ids),
        "chat_model": api["model_name"],
        "chat_witness": response.choices[0].message.content.strip(),
        "embedding_model": api["embedding_model_name"],
        "embedding_dimension": len(embedding.data[0].embedding),
        "bleu_dependency_witness": bleu_witness,
    }
    if result["conversations"] != 10 or question_count != 1540:
        raise RuntimeError(f"Unexpected LoCoMo scope: {result}")
    if (
        result["chat_witness"] != "OK"
        or result["embedding_dimension"] != 1536
        or bleu_witness != 1.0
    ):
        raise RuntimeError(f"API witness failed: {result}")
    write_json(run_dir / "preflight.json", result)
    update_stage_time(run_dir, "preflight", started)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def memory_path(run_dir: Path, user_id: str) -> Path:
    return run_dir / "memory" / user_id / f"{user_id}.json"


def snapshot_statistics(path: Path) -> dict[str, int]:
    snapshot = read_json(path)
    facts = snapshot["fact_store"]["facts"]
    claims = snapshot["claim_memory"]["claim_versions"]
    current_ids = set(snapshot["claim_memory"]["current_claim_version_id"].values())
    current_claims = [claim for claim in claims if claim["id"] in current_ids]
    fact_status = Counter(fact["status"] for fact in facts)
    claim_status = Counter(claim["status"] for claim in current_claims)
    return {
        "evidence_quotes": len(snapshot["fact_store"]["evidence_quotes"]),
        "facts_total": len(facts),
        "facts_active": fact_status[FactStatus.ACTIVE.value],
        "facts_superseded": fact_status[FactStatus.SUPERSEDED.value],
        "facts_retracted": fact_status[FactStatus.RETRACTED.value],
        "justifications_total": len(snapshot["claim_memory"]["justifications"]),
        "claim_versions_total": len(claims),
        "claims_current": len(current_claims),
        "claims_valid": claim_status["valid"],
        "claims_invalid": claim_status["invalid"],
    }


def run_construction(
    args: argparse.Namespace,
    config: dict[str, Any],
    dataset: LoCoMo,
    run_dir: Path,
) -> None:
    started = time.monotonic()
    runner = ConstructionRunner(
        ConstructionRunnerConfig(
            memory_type="OurMem",
            dataset_type="LoCoMo",
            dataset_path=str(args.dataset_path),
            memory_config=config,
            num_workers=args.workers,
            start_idx=args.start_index,
            end_idx=args.end_index,
            rerun=False,
            strict=True,
            token_cost_save_filename=str(run_dir / "token_cost_construction"),
            message_preprocessor=include_image_caption,
            tracing=False,
        )
    )
    runner.run()
    users = expected_users(dataset, args.start_index, args.end_index)
    missing = [user for user in users if not memory_path(run_dir, user).exists()]
    if missing:
        raise RuntimeError(f"Construction did not produce snapshots for: {missing}")
    stats = {
        user: snapshot_statistics(memory_path(run_dir, user))
        for user in users
    }
    write_json(run_dir / "construction_summary.json", stats)
    update_stage_time(run_dir, "construction", started)


def valid_shard(path: Path, expected_count: int) -> bool:
    if not path.exists():
        return False
    try:
        data = read_json(path)
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(data, list) and len(data) == expected_count


def run_search(
    args: argparse.Namespace,
    config: dict[str, Any],
    dataset: LoCoMo,
    run_dir: Path,
) -> Path:
    started = time.monotonic()
    shards_dir = run_dir / "retrieval_shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    trajectories, question_lists = dataset[
        args.start_index : args.end_index
    ]

    pending = []
    for trajectory, questions in zip(trajectories, question_lists):
        shard = shards_dir / f"{trajectory.id}.json"
        if not valid_shard(shard, len(questions)):
            pending.append((trajectory.id, questions, shard))

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                memory_search,
                "OurMem",
                user_id,
                questions,
                config,
                args.top_k,
                True,
                None,
            ): (user_id, shard)
            for user_id, questions, shard in pending
        }
        for future in as_completed(futures):
            user_id, shard = futures[future]
            results = future.result()
            write_json(shard, results)
            print(f"Saved retrieval shard: {user_id} ({len(results)} questions)")

    merged = []
    for trajectory, questions in zip(trajectories, question_lists):
        shard = shards_dir / f"{trajectory.id}.json"
        if not valid_shard(shard, len(questions)):
            raise RuntimeError(f"Invalid retrieval shard: {shard}")
        merged.extend(read_json(shard))
    output_path = run_dir / f"retrievals_top{args.top_k}.json"
    write_json(output_path, merged)
    update_stage_time(run_dir, "search", started)
    return output_path


def deserialize_retrievals(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deserialized = []
    for item in items:
        deserialized.append(
            {
                **item,
                "qa_pair": QuestionAnswerPair(**item["qa_pair"]),
                "retrieved_memories": [
                    MemoryEntry(**memory) for memory in item["retrieved_memories"]
                ],
            }
        )
    return deserialized


def run_evaluation(
    args: argparse.Namespace,
    api: dict[str, Any],
    dataset: LoCoMo,
    run_dir: Path,
) -> Path:
    started = time.monotonic()
    disable_tracing()
    retrieval_shards = run_dir / "retrieval_shards"
    evaluation_shards = run_dir / "evaluation_shards"
    evaluation_shards.mkdir(parents=True, exist_ok=True)
    api_keys = [api["api_key"]] * args.evaluation_concurrency
    base_urls = [api["base_url"]] * args.evaluation_concurrency
    trajectories, question_lists = dataset[
        args.start_index : args.end_index
    ]

    for trajectory, questions in zip(trajectories, question_lists):
        output_path = evaluation_shards / f"{trajectory.id}.json"
        if valid_shard(output_path, len(questions)):
            print(f"Reusing evaluation shard: {trajectory.id}")
            continue
        retrieval_path = retrieval_shards / f"{trajectory.id}.json"
        retrievals = deserialize_retrievals(read_json(retrieval_path))
        print(f"Evaluating {trajectory.id}: {len(retrievals)} questions")
        results = evaluate_memory(
            retrievals=retrievals,
            qa_model=api["model_name"],
            judge_model=api["model_name"],
            dataset_cls=LoCoMo,
            qa_batch_size=args.evaluation_concurrency,
            judge_batch_size=args.evaluation_concurrency,
            prompt_template=get_ourmem_locomo_qa_prompt,
            interface_kwargs={
                "api_keys": api_keys,
                "base_urls": base_urls,
            },
            metrics=["f1", "bleu", "llm_judge"],
        )
        write_json(output_path, results)
        print(f"Saved evaluation shard: {trajectory.id}")

    merged = []
    for trajectory, questions in zip(trajectories, question_lists):
        shard = evaluation_shards / f"{trajectory.id}.json"
        if not valid_shard(shard, len(questions)):
            raise RuntimeError(f"Invalid evaluation shard: {shard}")
        merged.extend(read_json(shard))
    output_path = run_dir / f"evaluation_top{args.top_k}.json"
    write_json(output_path, merged)
    update_stage_time(run_dir, "evaluation", started)
    return output_path


def wilson_interval(correct: int, total: int) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    z = 1.959963984540054
    p = correct / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(
        p * (1 - p) / total + z * z / (4 * total * total)
    ) / denominator
    return center - margin, center + margin


def aggregate_metric(
    items: list[dict[str, Any]],
    metric: str,
) -> float:
    return sum(item["metrics"][metric]["value"] for item in items) / len(items)


def retrieved_source_ids(item: dict[str, Any]) -> set[str]:
    return {
        source_id
        for memory in item["retrieved_memories"]
        for source_id in memory.get("metadata", {}).get("source_message_ids", [])
    }


def analyze(
    args: argparse.Namespace,
    dataset: LoCoMo,
    run_dir: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    evaluation_path = run_dir / f"evaluation_top{args.top_k}.json"
    items = read_json(evaluation_path)
    expected = sum(
        len(questions)
        for _, questions in dataset[args.start_index : args.end_index]
    )
    if len(items) != expected:
        raise RuntimeError(f"Expected {expected} evaluation items, found {len(items)}")

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        groups[item["qa_pair"]["metadata"]["question_type"]].append(item)

    def summarize(group: list[dict[str, Any]]) -> dict[str, Any]:
        correct = sum(item["metrics"]["llm_judge"]["value"] for item in group)
        low, high = wilson_interval(int(correct), len(group))
        return {
            "questions": len(group),
            "correct": int(correct),
            "accuracy": correct / len(group),
            "accuracy_ci95": [low, high],
            "f1": aggregate_metric(group, "f1"),
            "bleu1": aggregate_metric(group, "bleu"),
        }

    overall = summarize(items)
    by_category = {
        name: summarize(group)
        for name, group in sorted(groups.items())
    }

    evidence_items = [
        item for item in items if item["qa_pair"]["metadata"].get("evidence")
    ]
    evidence_hit = 0
    evidence_recall_sum = 0.0
    for item in evidence_items:
        gold = set(item["qa_pair"]["metadata"]["evidence"])
        retrieved = retrieved_source_ids(item)
        overlap = gold & retrieved
        evidence_hit += bool(overlap)
        evidence_recall_sum += len(overlap) / len(gold)

    with_claim = [
        item
        for item in items
        if any(
            memory.get("metadata", {}).get("kind") == "claim"
            for memory in item["retrieved_memories"]
        )
    ]
    claim_accuracy = (
        aggregate_metric(with_claim, "llm_judge") if with_claim else None
    )
    snapshots = read_json(run_dir / "construction_summary.json")
    totals = Counter()
    for stats in snapshots.values():
        totals.update(stats)

    failures = []
    failure_counts = Counter()
    for item in items:
        if item["metrics"]["llm_judge"]["value"] == 1.0:
            continue
        category = item["qa_pair"]["metadata"]["question_type"]
        if failure_counts[category] >= 5:
            continue
        failure_counts[category] += 1
        failures.append(
            {
                "category": category,
                "question": item["qa_pair"]["question"],
                "gold": item["qa_pair"]["golden_answers"],
                "prediction": item["prediction"],
                "gold_evidence": item["qa_pair"]["metadata"].get("evidence", []),
                "retrieved_source_ids": sorted(retrieved_source_ids(item)),
                "retrieved": [
                    memory.get("formatted_content") or memory["content"]
                    for memory in item["retrieved_memories"][:3]
                ],
            }
        )

    analysis = {
        "completed_at": utc_now(),
        "protocol": {
            "top_k": args.top_k,
            "categories": [1, 2, 3, 4],
            "start_index": args.start_index,
            "end_index": args.end_index,
        },
        "overall": overall,
        "by_category": by_category,
        "retrieval": {
            "questions_with_gold_evidence": len(evidence_items),
            "evidence_hit_rate": evidence_hit / len(evidence_items),
            "mean_evidence_recall": evidence_recall_sum / len(evidence_items),
            "questions_retrieving_claim": len(with_claim),
            "claim_retrieval_rate": len(with_claim) / len(items),
            "accuracy_when_claim_retrieved": claim_accuracy,
            "known_unresolved_gold_evidence_ids": [
                "D",
                "D10:19",
                "D4:36",
            ],
        },
        "memory": dict(totals),
        "failure_examples": failures,
    }
    update_stage_time(run_dir, "analysis", started)
    analysis["stage_times"] = read_json(run_dir / "stage_times.json")
    write_json(run_dir / "analysis.json", analysis)
    write_report(run_dir / "REPORT.md", analysis)
    return analysis


def percent(value: float | None) -> str:
    return "N/A" if value is None else f"{100 * value:.2f}%"


def write_report(path: Path, analysis: dict[str, Any]) -> None:
    overall = analysis["overall"]
    lines = [
        "# OurMem 在 LoCoMo 上的实验结果",
        "",
        "## 核心结果",
        "",
        (
            f"- 模型评判准确率：{overall['correct']}/{overall['questions']} "
            f"= {percent(overall['accuracy'])}"
        ),
        (
            "- 95% 置信区间："
            f"[{percent(overall['accuracy_ci95'][0])}, "
            f"{percent(overall['accuracy_ci95'][1])}]"
        ),
        f"- 词元 F1：{overall['f1']:.4f}",
        f"- BLEU-1：{overall['bleu1']:.4f}",
        "",
        "## 分类结果",
        "",
        "| 类别 | 正确/总数 | 准确率 | F1 | BLEU-1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for category, result in analysis["by_category"].items():
        lines.append(
            f"| {category} | {result['correct']}/{result['questions']} | "
            f"{percent(result['accuracy'])} | {result['f1']:.4f} | "
            f"{result['bleu1']:.4f} |"
        )
    retrieval = analysis["retrieval"]
    lines.extend(
        [
            "",
            "## 检索与主张使用",
            "",
            f"- 证据命中率：{percent(retrieval['evidence_hit_rate'])}",
            f"- 平均证据召回率：{percent(retrieval['mean_evidence_recall'])}",
            f"- 派生主张检索率：{percent(retrieval['claim_retrieval_rate'])}",
            (
                "- 检索到派生主张时的准确率："
                f"{percent(retrieval['accuracy_when_claim_retrieved'])}"
            ),
            (
                "- 原始数据中无法解析的证据编号："
                + ", ".join(retrieval["known_unresolved_gold_evidence_ids"])
            ),
            "",
            "## 记忆规模",
            "",
        ]
    )
    for key, value in sorted(analysis["memory"].items()):
        lines.append(f"- `{key}`：{value}")
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "该结果说明当前系统在固定 LoCoMo 协议下的端到端表现。"
            "它不能单独证明成立依据机制优于其他记忆系统；该结论仍需相同协议下的"
            "基线和消融实验支持。派生主张检索率与准确率之间的关系也只是相关性，"
            "不能解释为因果增益。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["all", "preflight", "construction", "search", "evaluation", "analysis"],
        default="all",
    )
    parser.add_argument("--run-name", default="full_v1")
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--api-config-path", type=Path, default=DEFAULT_API_CONFIG_PATH)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--evaluation-concurrency", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--message-batch-size", type=int, default=8)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.dataset_path = args.dataset_path.resolve()
    args.api_config_path = args.api_config_path.resolve()
    run_dir = (args.runs_dir / args.run_name).resolve()
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{args.stage}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_file = log_path.open("a", encoding="utf-8")
    sys.stdout = TeeStream(sys.__stdout__, log_file)
    sys.stderr = TeeStream(sys.__stderr__, log_file)

    api = load_api_config(args.api_config_path)
    dataset = filtered_dataset(args.dataset_path)
    if not 0 <= args.start_index < args.end_index <= len(dataset):
        raise ValueError("The requested trajectory range is invalid")
    config = memory_config(api, run_dir, args.message_batch_size)
    write_manifest(args, api, run_dir)

    print(f"Run directory: {run_dir}")
    print(f"Log file: {log_path}")
    print(f"API config: {args.api_config_path}")
    print(f"Stage: {args.stage}")

    if args.stage in {"all", "preflight"}:
        run_preflight(api, args.dataset_path, run_dir)
    if args.stage in {"all", "construction"}:
        run_construction(args, config, dataset, run_dir)
    if args.stage in {"all", "search"}:
        run_search(args, config, dataset, run_dir)
    if args.stage in {"all", "evaluation"}:
        run_evaluation(args, api, dataset, run_dir)
    if args.stage in {"all", "analysis"}:
        result = analyze(args, dataset, run_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
