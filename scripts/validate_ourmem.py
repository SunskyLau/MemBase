"""受限真实验证：独立短场景或真实输入抽取，不运行正式评测集问答。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-config", type=Path, default=ROOT / "examples/ourmem/api_config.json")
    parser.add_argument("--attempt", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S"))
    parser.add_argument("--real-data-extraction", action="store_true", help="只验证四个真实样本首批消息的抽取，不执行基准问答")
    parser.add_argument("--resume-attempt", help="复制一次失败短场景的V5检查点到新尝试；不覆盖旧产物")
    args = parser.parse_args()
    if Path(args.attempt).name != args.attempt or args.attempt in {".", ".."}:
        parser.error("attempt 必须是单个目录名")

    from membase.configs.ourmem import OurMemConfig
    from membase.ourmem.llm import ModelClient, RequestBudget
    from membase.ourmem.models import InputMessage, InputPolicy
    from membase.ourmem.system import OurMemSystem
    from membase.utils.benchmark_files import read_json, write_json
    from membase.utils.ourmem_version import ourmem_fingerprint

    credentials = read_json(args.api_config)
    config = OurMemConfig(api_key=credentials["api_keys"][0], base_url=credentials["base_urls"][0])
    parent = ROOT / "experiments/ourmem_validation/runs/v5_acceptance"
    folder = parent / args.attempt
    if folder.exists():
        parser.error("为保留既有验证产物，attempt 名称必须未使用；所有attempt仍共享总预算")
    folder.mkdir(parents=True)
    previous = None
    if args.resume_attempt:
        if args.real_data_extraction or Path(args.resume_attempt).name != args.resume_attempt or args.resume_attempt in {".", ".."}:
            parser.error("resume-attempt 仅适用于同目录的短场景失败记录")
        prior_folder = parent / args.resume_attempt
        previous = read_json(prior_folder / "validation.json")
        names = [case["name"] for case in previous["cases"]]
        if (previous.get("config") != config.model_dump(mode="json") or
                previous.get("complete") or names != ["before", "after_update"]):
            parser.error("该继续入口要求同配置、前两步已完成且删除阶段失败的短场景")
        shutil.copytree(prior_folder / "memory", folder / "memory")
    budget = RequestBudget(100, 20, parent / "request_budget.sqlite3")
    client = ModelClient(config, budget=budget, log_path=folder / "requests.jsonl")
    system = OurMemSystem(config=config, storage_dir=folder / "memory", client=client)
    namespace = "synthetic-trip"
    cases = list(previous["cases"]) if previous else []
    result = {"scope": ("first message batch of each real dataset; extraction only; not benchmark accuracy" if args.real_data_extraction else "synthetic validation only; not benchmark accuracy"), "config": config.model_dump(mode="json"),
              "implementation": ourmem_fingerprint(),
              "cases": cases, "complete": False, "budget_before": budget.summary()}
    if previous:
        result["continuation"] = {"from_attempt": args.resume_attempt,
                                  "previous_implementation": previous.get("implementation"),
                                  "reason": "explicit technical-failure continuation; completed earlier answers are reused, not regenerated"}
    write_json(folder / "validation.json", result)

    def ask(name, question, expected, date):
        snapshot = system.flush(namespace=namespace)
        answer = system.answer(question, namespace=namespace, snapshot_id=snapshot, query_time=date)
        prompt = ("Evaluate this independent synthetic test. Return JSON with boolean correct and a brief reason.\n"
                  f"Question: {question}\nExpected behavior: {expected}\nActual answer: {answer.answer_text}")
        def validate(raw):
            if type(raw.get("correct")) is not bool:
                raise ValueError("Judge must return boolean correct")
            return raw
        judged = client.request_json("judge", prompt, None, validator=validate, model=config.judge_model)
        case = {"name": name, "snapshot_id": snapshot, "answer": answer.model_dump(mode="json"), "check": judged}
        cases.append(case)
        write_json(folder / "validation.json", {**result, "budget": budget.summary()})
        print(json.dumps({"case": name, "answer": answer.answer_text, "check": judged,
                          "resolution": answer.resolution_status, "budget": budget.summary()}, ensure_ascii=False))
        return answer

    try:
        if args.real_data_extraction:
            from membase.datasets.official import default_paths, fingerprint, load_episodes
            from membase.ourmem.extractor import FactExtractor
            from membase.ourmem.models import Source
            for benchmark in ("locomo", "longmemeval", "memoryagentbench", "meme"):
                data_root, upstream = default_paths(benchmark)
                data_version = fingerprint(benchmark, data_root, upstream)
                episode = next(load_episodes(benchmark, data_root, "smoke"))
                # 明确的片段接入验证，不冒充保留完整历史的正式冒烟评测。
                batch = episode.sessions[0][:config.b_memory]
                sources = [Source(namespace="input-validation-" + benchmark, source_order=i,
                                  **InputMessage.model_validate(message).model_dump(exclude={"source_order"}))
                           for i, message in enumerate(batch)]
                extraction = FactExtractor(client, config).extract(
                    sources, [], InputPolicy.model_validate(episode.input_policy))
                artifact = {"benchmark": benchmark, "data_version": data_version,
                            "input_scope": "up to eight first-session messages; no evaluation questions or answers",
                            "sources": [source.model_dump(mode="json") for source in sources],
                            "extraction": extraction.model_dump(mode="json")}
                write_json(folder / (benchmark + "_extraction.json"), artifact)
                case = {"name": benchmark, "source_count": len(sources),
                        "draft_count": len(extraction.drafts), "unresolved_count": len(extraction.unresolved),
                        "structural_validation_passed": True}
                cases.append(case)
                write_json(folder / "validation.json", {**result, "budget": budget.summary()})
                print(json.dumps(case, ensure_ascii=False), flush=True)
            result["complete"] = len(cases) == 4
            code = 0 if result["complete"] else 1
        else:
            if previous is None:
                system.ingest([
                    InputMessage(message_id="initial-limit", conversation_id="trip", speaker="Alex",
                                 content="My father's maximum continuous walking distance is 300 meters.",
                                 mention_time="2026-08-01T12:00:00"),
                    InputMessage(message_id="initial-hotel", conversation_id="trip", speaker="Alex",
                                 content="For our trip with my father, Hotel Alder is 200 meters walking distance from the station.",
                                 mention_time="2026-08-01T12:01:00"),
                ], namespace=namespace, input_policy=InputPolicy())
                ask("before", "Is the walk from Hotel Alder to the station within my father's walking limit?",
                    "Yes. The 200-meter walk is within his 300-meter limit; do not make broader claims about the hotel.",
                    "2026-08-01T13:00:00")

                system.ingest([InputMessage(message_id="distance-update", conversation_id="trip", speaker="Alex",
                                            content="The station has moved. Hotel Alder's walking distance to it has changed from 200 meters to 2000 meters today.",
                                            mention_time="2026-08-02T12:00:00")], namespace=namespace, input_policy=InputPolicy())
                ask("after_update", "Is the current walk from Hotel Alder to the station within my father's walking limit?",
                    "No. The new 2000-meter distance exceeds the unchanged 300-meter limit.", "2026-08-02T13:00:00")

            system.ingest([InputMessage(message_id="forget-limit", conversation_id="trip", speaker="Alex",
                                        content="Please forget the maximum distance my father can walk and do not use that information again.",
                                        mention_time="2026-08-03T12:00:00")], namespace=namespace, input_policy=InputPolicy())
            deletion_snapshot = system.flush(namespace=namespace)
            deletion_request_boundary = budget.summary()["last_request_id"]
            after_delete = ask("after_delete", "What is my father's maximum continuous walking distance?",
                               "The information was deleted and must not be supplied, guessed, or restated.",
                               "2026-08-03T13:00:00")
            leak = bool(re.search(r"\b300\b", after_delete.answer_text))
            after_requests = []
            for line in (folder / "requests.jsonl").read_text().splitlines():
                entry = json.loads(line)
                if entry.get("kind") == "llm" and entry.get("id", 0) > deletion_request_boundary and entry.get("stage") != "judge":
                    after_requests.append(entry.get("messages", []))
            leaked_context = bool(re.search(r"\b300\b", json.dumps(after_requests)))
            snapshot = system.get_memory_snapshot(namespace, deletion_snapshot)
            write_json(folder / "final_memory.json", snapshot)
            result["deleted_value_in_answer"] = leak
            result["deleted_value_in_post_delete_model_input"] = leaked_context
            result["complete"] = len(cases) == 3 and all(case["check"]["correct"] for case in cases) and not leak and not leaked_context
            code = 0 if result["complete"] else 1
    except Exception as error:
        message = str(error).replace(config.api_key, "[REDACTED]")
        result["error"] = {"type": type(error).__name__, "message": message}
        print(f"受限验证未完成：{type(error).__name__}: {message}", file=sys.stderr)
        code = 1
    finally:
        result["budget"] = budget.summary()
        write_json(folder / "validation.json", result)
        system.close()
        client.close()
        budget.close()
    print(f"验证记录：{folder / 'validation.json'}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
