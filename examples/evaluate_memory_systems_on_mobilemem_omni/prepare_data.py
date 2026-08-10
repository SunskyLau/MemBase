#!/usr/bin/env python3
"""Merge and validate per-user MobileMem-Omni LoCoMo JSON files."""

import argparse
import json
import re
from pathlib import Path
from typing import Any


VALID_CATEGORIES = set(range(1, 8))


def _natural_key(path: Path) -> tuple[Any, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    )


def _load_samples(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON object or array.")
    if not all(isinstance(item, dict) for item in data):
        raise ValueError(f"{path} contains a non-object sample.")
    return data


def _validate_sample(sample: dict[str, Any], source: Path, index: int) -> str:
    sample_id = str(sample.get("sample_id") or "").strip()
    if not sample_id:
        raise ValueError(f"{source} sample {index} has no sample_id.")

    conversation = sample.get("conversation")
    if not isinstance(conversation, dict):
        raise ValueError(f"{source} sample {sample_id} has no conversation object.")
    if not conversation.get("speaker_a") or not conversation.get("speaker_b"):
        raise ValueError(f"{source} sample {sample_id} has incomplete speaker names.")
    session_keys = [
        key
        for key, value in conversation.items()
        if key.startswith("session_")
        and not key.endswith("_date_time")
        and isinstance(value, list)
    ]
    if not session_keys:
        raise ValueError(f"{source} sample {sample_id} has no sessions.")
    for key in session_keys:
        if not conversation.get(f"{key}_date_time"):
            raise ValueError(f"{source} sample {sample_id} {key} has no timestamp.")

    questions = sample.get("qa")
    if not isinstance(questions, list):
        raise ValueError(f"{source} sample {sample_id} has no qa array.")
    for question_index, question in enumerate(questions, start=1):
        category = question.get("category")
        if category not in VALID_CATEGORIES:
            raise ValueError(
                f"{source} sample {sample_id} question {question_index} has category "
                f"{category!r}; expected 1-7. This often means an unknown question type "
                "was converted to category 0."
            )
        if not str(question.get("question") or "").strip():
            raise ValueError(
                f"{source} sample {sample_id} question {question_index} is empty."
            )
    return sample_id


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge locomo_u*.json files into one validated MobileMem-Omni file."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="A locomo_u*.json file or a directory containing per-user files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "examples/evaluate_memory_systems_on_mobilemem_omni/data/"
            "mobilemem_omni_locomo.json"
        ),
        help="Combined output JSON path.",
    )
    args = parser.parse_args()

    input_path = args.input
    if input_path.is_dir():
        files = sorted(input_path.glob("locomo_u*.json"), key=_natural_key)
    elif input_path.is_file():
        files = [input_path]
    else:
        raise FileNotFoundError(f"Input does not exist: {input_path}")
    if not files:
        raise FileNotFoundError(f"No locomo_u*.json files found under {input_path}")

    combined = []
    seen_sample_ids = set()
    for file_path in files:
        for index, sample in enumerate(_load_samples(file_path), start=1):
            sample_id = _validate_sample(sample, file_path, index)
            if sample_id in seen_sample_ids:
                raise ValueError(f"Duplicate sample_id across input files: {sample_id}")
            seen_sample_ids.add(sample_id)
            combined.append(sample)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(combined, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    question_count = sum(len(sample["qa"]) for sample in combined)
    print(
        f"Prepared {len(combined)} users and {question_count} questions from "
        f"{len(files)} file(s): {args.output}"
    )


if __name__ == "__main__":
    main()
