"""MEME 官方回答/评分的完整性检查与平凡通过过滤汇总。"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from ..datasets.meme import episode_key, question_key
from ..utils.benchmark_files import read_json, sha256_file


def output_name(episode: dict, baseline: str, model: str) -> str:
    return f"agent_{episode_key(episode)}_{baseline}_{model.replace('/', '-')}.json"


def judge_name(episode: dict, baseline: str, model: str, judge_model: str) -> str:
    return f"eval_{episode_key(episode)}_{baseline}_{model.replace('/', '-')}_{judge_model.replace('/', '-')}.json"


def validate_questions(records: list[dict], questions: list[dict], label: str, *, allow_failures=False) -> None:
    actual = Counter(question_key(q) for q in records)
    expected = Counter(question_key(q) for q in questions)
    if actual != expected:
        missing = [key[1] for key in (expected - actual)]
        extra = [key[1] for key in (actual - expected)]
        raise ValueError(f"{label} 问题不完整：缺失 {missing[:5]}；多余/重复 {extra[:5]}")
    for record in records:
        if allow_failures and record.get("technical_failure"):
            failure = record["technical_failure"]
            if not isinstance(failure.get("reason"), str) or not isinstance(failure.get("request_ids"), list):
                raise ValueError(f"{label} 技术失败缺少审计记录")
            continue
        if not isinstance(record.get("agent_answer"), str) or not record["agent_answer"].strip():
            raise ValueError(f"{label} 缺少模型回答")


def validate_answer(path: Path, episode: dict, baseline: str, model: str,
                    internal_model: str, top_k: int, *, audited=False) -> dict:
    output = read_json(path)
    if output["episode_id"] != episode["episode_id"] or output["domain"] != episode["domain"]:
        raise ValueError(f"MEME 样本身份不匹配：{path}")
    config = output["config"]
    if config.get("agent_type") != baseline or config.get("agent_model") != model:
        raise ValueError(f"MEME 基线或模型不匹配：{path}")
    if baseline != "in_context" and config.get("internal_model") != internal_model:
        raise ValueError(f"MEME 内部模型不匹配：{path}")
    if baseline in {"bm25", "dense"} and config.get("top_k") != top_k:
        raise ValueError(f"MEME 检索数量不匹配：{path}")
    if audited and output.get("instrumentation", {}).get("protocol") != "meme-native-audit-v1":
        raise ValueError(f"MEME 缺少当前执行审计：{path}")
    for phase in ("before", "after"):
        validate_questions(output[f"{phase}_answers"], episode[f"{phase}_questions"]["questions"],
                           f"{episode_key(episode)}/{phase}", allow_failures=audited)
        if audited:
            for row in output[f"{phase}_answers"]:
                if not isinstance(row.get("retrieved_context"), str) or "context_tokens" not in row or "stage_times" not in row:
                    raise ValueError(f"MEME 缺少上下文或耗时记录：{path}")
    return output


def validate_judge(path: Path, answer: dict, judge_model: str, *, allow_failures=False) -> dict:
    result = read_json(path)
    if answer.get("instrumentation", {}).get("protocol") == "meme-native-audit-v1":
        if result.get("instrumentation", {}).get("protocol") != "meme-native-audit-v1":
            raise ValueError(f"MEME 评分缺少执行审计：{path}")
        if result.get("domain") != answer["domain"]:
            raise ValueError(f"MEME 评分领域不匹配：{path}")
    if result["episode_id"] != answer["episode_id"] or result["agent_config"] != answer["config"]:
        raise ValueError(f"MEME 评分身份或配置不匹配：{path}")
    if result["judge_config"]["judge_model"] != judge_model:
        raise ValueError(f"MEME 评判模型不匹配：{path}")
    for phase in ("before", "after"):
        rows = result[f"{phase}_answers"]
        validate_questions(rows, answer[f"{phase}_answers"], f"{path.name}/{phase}", allow_failures=allow_failures)
        original = {question_key(q): q["agent_answer"] for q in answer[f"{phase}_answers"]}
        for row in rows:
            if row["agent_answer"] != original[question_key(row)] or type(row.get("u_pass")) is not bool:
                raise ValueError(f"MEME 评分不是当前回答的完整结果：{path}")
            if row.get("u_reason") == "missing":
                raise ValueError(f"MEME 评分缺失：{path}")
            if allow_failures and row.get("technical_failure") and row["u_pass"] is not False:
                raise ValueError("技术失败不能成为正确回答")
    # 与官方 judge_episode 的前后状态过滤保持一致。
    before = {next(iter(q["entity_values"])): q["u_pass"] for q in result["before_answers"]}
    for row in result["after_answers"]:
        task = row["task_type"].split(" (")[0]
        if task in {"Cas", "Abs", "Del"}:
            b_pass = before.get(next(iter(row["entity_values"])), False)
            label = ("real" if b_pass else "trivial") if row["u_pass"] else (
                "knew_but_failed" if b_pass else "never_knew")
            if row.get("pass_type") != label:
                raise ValueError(f"MEME 平凡通过过滤不完整：{path}")
    return result


def receipt(answer_path: Path, judge_path: Path, judge_model: str) -> dict:
    return {"answer_sha256": sha256_file(answer_path), "judge_sha256": sha256_file(judge_path),
            "judge_model": judge_model}


def judge_is_reusable(answer_path: Path, judge_path: Path, judge_model: str,
                      saved: dict | None, answer: dict, *, allow_failures=False) -> bool:
    if saved is None or not judge_path.is_file():
        return False
    try:
        if saved != receipt(answer_path, judge_path, judge_model):
            return False
        validate_judge(judge_path, answer, judge_model, allow_failures=allow_failures)
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return True


def summarize(results: list[tuple[str, dict]]) -> dict:
    """主准确率按问题计算；Cas/Abs/Del 只计官方 real pass。"""
    by_task: dict[str, dict] = {}
    by_domain: dict[str, dict] = {}
    by_hop: dict[str, dict] = {}
    failures = Counter()
    before_total = before_correct = answered = scored = 0
    latencies = {"retrieve_seconds": [], "answer_seconds": []}
    contexts = []
    for domain, result in results:
        for phase in ("before", "after"):
            for row in result[f"{phase}_answers"]:
                failure = row.get("technical_failure")
                if failure:
                    failures[f"{phase}/{failure['stage']}"] += 1
                answered += int(isinstance(row.get("agent_answer"), str) and bool(row["agent_answer"].strip()))
                scored += int(not failure)
                if phase == "before":
                    before_total += 1
                    before_correct += int(row["u_pass"])
                for key in latencies:
                    if key in row.get("stage_times", {}):
                        latencies[key].append(row["stage_times"][key])
                if "context_tokens" in row:
                    contexts.append(row["context_tokens"])
        for row in result["after_answers"]:
            task = row["task_type"].split(" (")[0]
            correct = row["pass_type"] == "real" if task in {"Cas", "Abs", "Del"} else row["u_pass"]
            groups = [(task, by_task), (domain, by_domain)]
            if "hop" in row:
                groups.append((f"{task}/{row['hop']}", by_hop))
            for key, group in groups:
                entry = group.setdefault(key, {"total": 0, "correct": 0, "unfiltered_correct": 0})
                entry["total"] += 1
                entry["correct"] += int(correct)
                entry["unfiltered_correct"] += int(row["u_pass"])
    for group in (by_task, by_domain, by_hop):
        for entry in group.values():
            entry["accuracy"] = entry["correct"] / entry["total"]
    total = sum(row["total"] for row in by_task.values())
    correct = sum(row["correct"] for row in by_task.values())
    return {"episodes": len(results), "total": total, "correct": correct,
            "accuracy": correct / total, "by_task": by_task, "by_domain": by_domain,
            "by_task_hop": by_hop,
            "before": {"total": before_total, "correct": before_correct,
                       "accuracy": before_correct / before_total if before_total else None},
            "expected_questions": before_total + total,
            "actual_answers": answered, "actual_scores": scored,
            "technical_failure_questions": sum(failures.values()), "failures_by_phase_stage": dict(failures),
            "timing": {key: {"observations":len(values), "total":sum(values),
                              "mean":sum(values)/len(values) if values else None} for key,values in latencies.items()},
            "context_tokens": {"observations":len(contexts), "mean":sum(contexts)/len(contexts) if contexts else None,
                               "max":max(contexts) if contexts else None},
            "filter": "Cas/Abs/Del require correct before and after answers"}
