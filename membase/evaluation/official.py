"""方法无关的官方提示、评分与结果校验；模型接口由调用方注入。"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from types import ModuleType

from ..datasets.official import Episode, Question
from ..utils.benchmark_files import read_json, write_json


def official_module(path: Path, *, definitions: set[str] | None = None,
                    exclude_imports: set[str] = frozenset()) -> ModuleType:
    """只延迟无关导入或提取指定定义；官方函数体和提示词保持原样。"""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    if definitions is not None:
        tree.body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                     and node.name in definitions]
    elif exclude_imports:
        tree.body = [node for node in tree.body if not (
            isinstance(node, ast.ImportFrom) and node.module in exclude_imports)]
    module = ModuleType(f"official_{hashlib.sha256(str(path).encode()).hexdigest()[:12]}")
    module.__file__ = str(path)
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module


def literal_assignment(path: Path, name: str):
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f"固定官方版本中没有 {name}：{path}")


def answer_prompt(benchmark: str, upstream: Path, episode: Episode,
                  question: Question, context: str) -> tuple[str, int]:
    if benchmark == "meme":
        template = literal_assignment(upstream / "code/agents/base.py", "UNIFIED_ANSWER_PROMPT")
        return template.format(context=context, question=question.text), 500
    if benchmark == "memoryagentbench":
        module = official_module(upstream / "utils/templates.py")
        template = module.BASE_TEMPLATES["factconsolidation"]["query"]["rag_agent"]
        source = question.reference["source"]
        setting = upstream / "configs/data_conf/Conflict_Resolution" / f"{source.capitalize()}.yaml"
        limit = next(int(line.split(":", 1)[1].strip()) for line in setting.read_text().splitlines()
                     if line.startswith("generation_max_length:"))
        return "Memory 1:\n" + context + "\n\n" + template.format(question=question.text), limit
    if benchmark == "locomo":
        path = upstream / "task_eval/gpt_utils.py"
        template = literal_assignment(path, "QA_PROMPT")
        intro = literal_assignment(path, "CONV_START_PROMPT")
        conv = episode.reference["conversation"]
        return intro.format(conv["speaker_a"], conv["speaker_b"]) + context + "\n\n" + template.format(question.text), 32
    # 读取官方普通历史回答分支的原字符串，不导入 transformers 等生成后端。
    tree = ast.parse((upstream / "src/generation/run_generation.py").read_text(encoding="utf-8"))
    candidates = [node.value for node in ast.walk(tree) if isinstance(node, ast.Constant)
                  and isinstance(node.value, str) and node.value.startswith(
                      "I will give you several history chats between you and a user. Please answer the question based on the relevant chat history.\n")
                  and node.value.endswith("\nAnswer:")]
    if len(set(candidates)) != 1:
        raise ValueError("LongMemEval 官方普通回答模板与固定版本不符")
    return candidates[0].format(context, question.reference["question_date"], question.text), 500


def answer_system(benchmark: str, upstream: Path) -> str | None:
    if benchmark == "memoryagentbench":
        return literal_assignment(upstream / "utils/templates.py", "SYSTEM_MESSAGE")
    return None


class Scorer:
    def __init__(self, benchmark: str, upstream: Path, client, judge_model: str,
                 cache_dir: Path, *, locomo_judge: bool = False):
        self.benchmark, self.upstream = benchmark, upstream
        self.client, self.judge_model, self.cache_dir = client, judge_model, cache_dir
        self.locomo_judge = locomo_judge
        self.module = None
        if benchmark == "locomo":
            self.module = official_module(upstream / "task_eval/evaluation.py", exclude_imports={"bert_score"})
        elif benchmark == "memoryagentbench":
            path = upstream / "utils/eval_other_utils.py"
            self.module = official_module(path, definitions={"normalize_answer", "substring_exact_match_score", "drqa_metric_max_over_ground_truths"})
            import re
            import string
            self.module.__dict__.update(re=re, string=string)
        elif benchmark == "longmemeval":
            self.module = official_module(upstream / "src/evaluation/evaluate_qa.py", definitions={"get_anscheck_prompt"})

    def _json(self, prompt: str, expected_items: int | None = None) -> dict:
        key = hashlib.sha256((self.judge_model + "\n" + prompt).encode()).hexdigest()
        path = self.cache_dir / f"{key}.json"
        failure_path = self.cache_dir / f"{key}.failure.json"
        def validate(value):
            if not isinstance(value, dict):
                raise ValueError("官方评判必须返回 JSON 对象")
            if expected_items is not None:
                if not isinstance(value.get("results"), list):
                    raise ValueError("官方聚合评判缺少 results 列表")
                if len(value["results"]) != expected_items:
                    raise ValueError("官方聚合评判的逐项结果数量不完整")
                if any(not isinstance(v, dict) or type(v.get("present")) is not bool for v in value["results"]):
                    raise ValueError("官方评判缺少逐项布尔结果")
            elif type(value.get("correct")) is not bool:
                raise ValueError("官方评判缺少 correct 布尔值")
            return value

        if path.exists():
            return validate(read_json(path))
        from ..inference_utils.model_client import RecoverableModelError, failure_details
        if failure_path.exists():
            failure = read_json(failure_path)
            raise RecoverableModelError(failure["reason"], request_ids=failure["request_ids"])
        try:
            value = self.client.request_json("judge", prompt, None, validator=validate, model=self.judge_model)
        except RecoverableModelError as error:
            write_json(failure_path, failure_details(error))
            raise
        write_json(path, value)
        return value

    def score(self, question: Question, answer: str) -> dict:
        if self.benchmark == "memoryagentbench":
            value = self.module.drqa_metric_max_over_ground_truths(
                self.module.substring_exact_match_score, answer, question.reference["answer"])
            return {"substring_exact_match": bool(value)}
        if self.benchmark == "locomo":
            values, _, _ = self.module.eval_question_answering([{**question.reference, "prediction": answer}])
            result = {"official_f1": float(values[0])}
            if self.locomo_judge:
                prompt = ("Evaluate whether the candidate answers the question correctly. Equivalent wording is allowed; "
                          "missing required information is incorrect. Return JSON with boolean correct and string reason.\n"
                          f"Question: {question.text}\nReference: {question.reference['answer']}\nCandidate: {answer}")
                result["additional_judge"] = self._json(prompt)
            return result
        if self.benchmark == "longmemeval":
            prompt = self.module.get_anscheck_prompt(question.reference["question_type"], question.text,
                                                     question.reference["answer"], answer,
                                                     abstention="_abs" in question.id)
            response = self.client.text(prompt, stage="judge", model=self.judge_model,
                                        temperature=0, max_tokens=10)
            return {"official_accuracy": "yes" in response.lower(), "judge_response": response}
        raise ValueError("MEME 使用完整 before/after 样本评分")

    def score_meme(self, output: dict, check_workers: int = 8) -> dict:
        official = official_module(self.upstream / "code/eval/judge.py")
        owner = self
        from threading import local
        from ..inference_utils.model_client import RecoverableModelError, failure_details
        blocked = {}
        for phase in ("before", "after"):
            for row in output[f"{phase}_answers"]:
                if not row.get("technical_failure"):
                    continue
                ev = row["entity_values"]
                entity = next(iter(ev))
                key = (("multi", phase, row["question"], json.dumps(ev, sort_keys=True))
                       if phase == "after" and row["task_type"].split(" (")[0] == "Agg"
                       else ("single", phase, row["question"], entity, ev[entity]))
                blocked[key] = row["technical_failure"]

        def failed_check(details):
            return {"u_pass": False, "u_reason": "technical_failure_assigned_zero",
                    "official_u_pass": None, "score_origin": "technical_failure", "technical_failure": details}

        class SharedJudge(official.LLMJudge):
            context = local()

            def _call(self, prompt):
                return owner._json(prompt, getattr(self.context, "expected_items", None))

            def u_check(self, question, entity, gold_value, agent_answer, task_type, phase="after"):
                key = ("single", phase, question, entity, gold_value)
                if key in blocked:
                    return failed_check(blocked[key])
                try:
                    return super().u_check(question, entity, gold_value, agent_answer, task_type, phase)
                except RecoverableModelError as error:
                    return failed_check(failure_details(error))

            def u_check_multi(self, question, entity_values, agent_answer, phase="after"):
                key = ("multi", phase, question, json.dumps(entity_values, sort_keys=True))
                if key in blocked:
                    return failed_check(blocked[key])
                self.context.expected_items = len(entity_values)
                try:
                    return super().u_check_multi(question, entity_values, agent_answer, phase)
                except RecoverableModelError as error:
                    return failed_check(failure_details(error))
                finally:
                    self.context.expected_items = None

        # 仅替换传输和重复请求缓存，任务判断与平凡通过过滤由官方执行。
        judge = SharedJudge(None, model=self.judge_model, max_retries=1)
        result = official.judge_episode(output, judge, max_workers=check_workers)
        result["judge_config"] = {"judge_model": self.judge_model}
        return result


def validate_answers(records: list[dict], questions: tuple[Question, ...], *, complete: bool = True, allow_failures: bool = False) -> None:
    expected = {q.id: q for q in questions}
    actual = [row.get("question_id") for row in records]
    if len(set(actual)) != len(actual) or set(actual) - set(expected):
        raise ValueError("回答中有重复或意外问题标识")
    if complete and set(actual) != set(expected):
        raise ValueError(f"缺少回答：{sorted(set(expected) - set(actual))[:10]}")
    for row in records:
        if row.get("question") != expected[row["question_id"]].text:
            raise ValueError("回答问题与输入不一致")
        if allow_failures and row.get("status") == "technical_failure":
            from ..runners.question_outcomes import validate_failure
            validate_failure(row)
            continue
        if not isinstance(row.get("answer_text"), str) or not row["answer_text"].strip():
            raise ValueError("缺少有效回答文本")
        if row.get("resolution_status") not in {"resolved", "unknown", "conflict", "deleted", "incomplete", "not_assessed"}:
            raise ValueError("回答缺少处理状态")


def validate_scores(benchmark: str, scores: dict, *, locomo_judge: bool = False) -> None:
    if not isinstance(scores, dict):
        raise ValueError("评分记录必须是对象")
    if benchmark == "locomo":
        value = scores.get("official_f1")
        if type(value) not in {int, float} or not 0 <= value <= 1:
            raise ValueError("LoCoMo 缺少合法官方 F1")
        if locomo_judge:
            judge = scores.get("additional_judge")
            if not isinstance(judge, dict) or type(judge.get("correct")) is not bool:
                raise ValueError("LoCoMo 缺少额外评判结果")
    else:
        metric = "substring_exact_match" if benchmark == "memoryagentbench" else "official_accuracy"
        if type(scores.get(metric)) is not bool:
            raise ValueError(f"缺少合法官方指标：{metric}")
        if benchmark == "longmemeval" and not isinstance(scores.get("judge_response"), str):
            raise ValueError("LongMemEval 缺少官方评判原文")


def summarize(benchmark: str, episodes: list[tuple[Episode, dict]]) -> dict:
    from collections import Counter
    expected = sum(len(p.questions) for ep, _ in episodes for p in ep.phases)
    warning_samples = sum(bool(out.get("memory_warnings")) for _, out in episodes)
    if benchmark == "meme":
        from .meme import summarize as official_summary
        result = official_summary([(ep.reference["domain"], output["judge"]) for ep, output in episodes])
        rows = [row for _, out in episodes for phase in ("before", "after") for row in out["judge"][f"{phase}_answers"]]
        failures = Counter(row["technical_failure"].get("failed_stage", "judge") for row in rows if row.get("technical_failure"))
        # 原生 before/ER 不调用评判，不把它计成一次实际评分。
        scored = sum(not row.get("technical_failure") and row.get("u_reason") != "skipped (ER)" for row in rows)
        result.update(expected_questions=expected, answered_questions=sum(bool(row.get("agent_answer")) for row in rows),
                      scored_questions=scored, technical_failures_by_stage=dict(failures),
                      technical_failure_questions=sum(failures.values()), memory_warning_samples=warning_samples,
                      scoring_policy="official_task_rules_with_technical_failures_assigned_zero")
        return result
    groups = {}
    metric = {"locomo": "official_f1", "longmemeval": "official_accuracy",
              "memoryagentbench": "substring_exact_match"}[benchmark]
    statuses, failures = {}, Counter()
    answered = scored = 0
    for ep, output in episodes:
        refs = {q.id: q for p in ep.phases for q in p.questions}
        for row in output["answers"]:
            ref = refs[row["question_id"]].reference
            group = str(ref["category"]) if benchmark == "locomo" else (
                ref["question_type"] if benchmark == "longmemeval" else ref["source"])
            values = groups.setdefault(group, [])
            answered += bool(row.get("answer_text"))
            if row.get("status") == "technical_failure":
                from ..runners.question_outcomes import validate_failure
                validate_failure(row)
                values.append(0.0)
                failures[row["failed_stage"]] += 1
            else:
                values.append(float(row["scores"][metric]))
                scored += 1
            status = row["resolution_status"]
            statuses[status] = statuses.get(status, 0) + 1
    values = [v for group in groups.values() for v in group]
    if not values:
        raise ValueError("没有可汇总的完整回答")
    if len(values) != expected:
        raise ValueError("存在未处理问题，不能用缩小的分母汇总")
    result = {"metric": metric, "episodes": len(episodes), "questions": len(values),
              "score": sum(values) / len(values), "by_group": {
                  name: {"questions": len(v), "score": sum(v) / len(v)} for name, v in groups.items()},
              "resolution_status_counts": statuses, "expected_questions": expected,
              "answered_questions": answered, "scored_questions": scored,
              "technical_failures_by_stage": dict(failures), "technical_failure_questions": sum(failures.values()),
              "memory_warning_samples": warning_samples,
              "scoring_policy": "official_metric_with_technical_failures_assigned_zero"}
    if benchmark == "locomo" and any(out.get("locomo_judge_enabled") or
            any("additional_judge" in (row.get("scores") or {}) for row in out["answers"]) for _, out in episodes):
        additional = [float(row["scores"]["additional_judge"]["correct"]) if row.get("status") != "technical_failure" else 0.0
                      for _, output in episodes for row in output["answers"]
                      if row.get("status") == "technical_failure" or "additional_judge" in row["scores"]]
        if additional:
            result["additional_judge_accuracy"] = sum(additional) / len(additional)
    return result


def summarize_run_costs(run_dir: Path, shared_budget_totals: dict, *, exclusive_ledger: bool = False) -> dict:
    """运行内日志负责归属，共享账本负责额度；不能把其他实验的调用算到本次。"""
    def empty():
        return {"llm_requests": 0, "embedding_requests": 0, "failed_requests": 0,
                "requests_without_usage": 0, "usage": {}}

    def add(group: dict, record: dict) -> None:
        group[f"{record['kind']}_requests"] += 1
        group["failed_requests"] += int("error" in record)
        usage = record.get("usage")
        if not isinstance(usage, dict):
            group["requests_without_usage"] += 1
        else:
            for key, value in usage.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    group["usage"][key] = group["usage"].get(key, 0) + value

    totals, by_stage, by_model, by_log = empty(), {}, {}, {}
    seen, validation_events, unknown_events, malformed = set(), 0, 0, []
    for path in sorted(run_dir.rglob("requests.jsonl")):
        label = str(path.relative_to(run_dir))
        with path.open(encoding="utf-8", errors="replace") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    # 崩溃留下半行日志时仍保存其余已确认成本，不编造缺失调用。
                    malformed.append({"file": label, "line": line_number})
                    continue
                if not isinstance(record, dict):
                    unknown_events += 1
                    continue
                if "validation_error" in record:
                    validation_events += 1
                    continue  # 同一请求的校验事件不是第二次外发请求。
                if record.get("kind") not in {"llm", "embedding"} or type(record.get("id")) is not int:
                    unknown_events += 1
                    continue
                identity = (record["kind"], record["id"])
                if identity in seen:
                    continue
                seen.add(identity)
                stage = record.get("stage", "embedding" if record["kind"] == "embedding" else "unattributed")
                model = record.get("model", "unattributed")
                for group in (totals, by_stage.setdefault(stage, empty()),
                              by_model.setdefault(model, empty()), by_log.setdefault(label, empty())):
                    add(group, record)
    outside = {kind: max(0, shared_budget_totals.get(kind, 0) - totals[kind])
               for kind in ("llm_requests", "embedding_requests")}
    complete = (exclusive_ledger and all(shared_budget_totals.get(kind, 0) == totals[kind] for kind in outside)
                and not malformed and not unknown_events)
    return {"shared_budget_totals": shared_budget_totals,
            "run_log_totals": {**totals, "by_stage": by_stage, "by_model": by_model, "by_log": by_log},
            "attribution": {
                "ledger_scope": "this_run_only" if exclusive_ledger else "shared_or_external",
                "coverage": "complete" if complete else "logged_requests_only",
                "unattributed_global_requests": outside,
                "note": ("Unattributed requests belong to this run but lack a complete local request log."
                         if exclusive_ledger else
                         "Unattributed global requests may belong to other runs or be absent from local logs; they are not assigned to this run."),
                "validation_events_not_requests": validation_events,
                "unrecognized_log_events": unknown_events, "malformed_log_lines": malformed,
            }}
