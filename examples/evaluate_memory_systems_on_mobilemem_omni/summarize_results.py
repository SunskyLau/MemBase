#!/usr/bin/env python3
"""Aggregate MemBase evaluation results overall and by question type."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _metric_value(result: dict[str, Any], name: str) -> float | None:
    metric = result.get("metrics", {}).get(name)
    if not isinstance(metric, dict) or "value" not in metric:
        return None
    return float(metric["value"])


def _summarise(items: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = sorted(
        {
            name
            for item in items
            for name in item.get("metrics", {})
        }
    )
    metrics = {}
    for name in metric_names:
        values = [
            value
            for item in items
            if (value := _metric_value(item, name)) is not None
        ]
        metrics[name] = sum(values) / len(values) if values else 0.0
    return {"count": len(items), "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evaluation_results", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    results = json.loads(args.evaluation_results.read_text(encoding="utf-8"))
    if not isinstance(results, list):
        raise ValueError("Evaluation results must be a JSON array.")

    grouped = defaultdict(list)
    for result in results:
        question_type = (
            result.get("qa_pair", {})
            .get("metadata", {})
            .get("question_type", "unknown")
        )
        grouped[question_type].append(result)

    summary = {
        "overall": _summarise(results),
        "by_question_type": {
            name: _summarise(items)
            for name, items in sorted(grouped.items())
        },
    }
    output_path = args.output or args.evaluation_results.with_name(
        f"{args.evaluation_results.stem}_summary.json"
    )
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Summary saved to: {output_path}")


if __name__ == "__main__":
    main()
