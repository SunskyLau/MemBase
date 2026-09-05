"""核对官方 MAB 结果的问题标识，汇总其原始子串匹配分数。"""

from __future__ import annotations

from pathlib import Path

from ..utils.benchmark_files import read_json


def validate_result(path: Path, expected: list[dict], source: str,
                    model: str, *, allow_partial: bool = False) -> dict:
    result = read_json(path)
    if result["dataset_config"]["sub_dataset"] != source or result["agent_config"]["model"] != model:
        raise ValueError(f"MAB 结果配置不匹配：{path}")
    records = result["data"]
    actual_ids = [row["qa_pair_id"] for row in records]
    expected_ids = [q["id"] for q in expected]
    # 官方续跑按照已有问题数量恢复，所以只能接受正确的连续前缀。
    if actual_ids != expected_ids[:len(actual_ids)]:
        raise ValueError(f"MAB 问题标识缺失、重复或顺序错误：{path}")
    for index, row in enumerate(records):
        if row.get("query_id") != index or not isinstance(row.get("output"), str):
            raise ValueError(f"MAB 回答记录不完整：{path}/{index}")
        if not row["output"].strip() or expected[index]["question"] not in row.get("query", ""):
            raise ValueError(f"MAB 回答或问题内容不匹配：{path}/{index}")
        if row.get("substring_exact_match") not in (0, 1):
            raise ValueError(f"MAB 缺少官方评分：{path}/{index}")
    if not allow_partial and len(records) != len(expected):
        missing = expected_ids[len(records):]
        raise ValueError(f"MAB {source} 只完成 {len(records)}/{len(expected)}；缺失：{missing[:8]}")
    return result


def summarize(results: dict[str, dict]) -> dict:
    by_subset = {}
    for source, result in results.items():
        rows = result["data"]
        passed = sum(row["substring_exact_match"] for row in rows)
        by_subset[source] = {"total": len(rows), "correct": passed,
                             "accuracy": passed / len(rows)}
    total = sum(item["total"] for item in by_subset.values())
    correct = sum(item["correct"] for item in by_subset.values())
    return {"metric": "substring_exact_match", "by_subset": by_subset,
            "total": total, "correct": correct, "accuracy": correct / total}
