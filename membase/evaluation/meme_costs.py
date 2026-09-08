"""按用途汇总持久账本；未知用量不当作零，评判费用不混入方法费用。"""
from collections import Counter
from contextlib import closing
import json
from pathlib import Path
import sqlite3


def summarize_costs(stage_dir: Path) -> dict:
    groups = {}
    for path in sorted((stage_dir / "audit").glob("*/*/requests.sqlite")):
        with closing(sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)) as db:
            for kind, stage, model, status, raw, elapsed in db.execute(
                    "SELECT kind,stage,model,status,usage,elapsed FROM requests"):
                key = (stage, kind, model)
                row = groups.setdefault(key, {
                    "stage": stage, "kind": kind, "model": model, "requests": 0,
                    "statuses": Counter(), "requests_without_usage": 0,
                    "known_input_tokens": 0, "known_output_tokens": 0,
                    "known_cached_input_tokens": 0, "request_seconds": 0.0,
                })
                row["requests"] += 1
                row["statuses"][status] += 1
                row["request_seconds"] += elapsed or 0
                usage = json.loads(raw) if raw is not None else {}
                input_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
                output_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
                if input_tokens is None or (kind == "llm" and output_tokens is None):
                    row["requests_without_usage"] += 1
                row["known_input_tokens"] += input_tokens or 0
                row["known_output_tokens"] += output_tokens or 0
                row["known_cached_input_tokens"] += (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
    rows = list(groups.values())
    return {
        "by_stage_model": rows,
        "method_requests": sum(r["requests"] for r in rows if r["stage"] != "judge"),
        "judge_requests": sum(r["requests"] for r in rows if r["stage"] == "judge"),
        "requests_without_usage": sum(r["requests_without_usage"] for r in rows),
        "method_known_tokens": sum(r["known_input_tokens"] + r["known_output_tokens"] for r in rows if r["stage"] != "judge"),
        "judge_known_tokens": sum(r["known_input_tokens"] + r["known_output_tokens"] for r in rows if r["stage"] == "judge"),
        "monetary_cost": None,
        "note": "All attempts included. Token totals are lower bounds when usage is unknown. "
                "Apply the actual provider price sheet separately; cached tokens are a subset of input tokens. "
                "Request seconds are summed latency, not parallel wall time.",
    }
