"""逐题技术失败是持久化终态，不是假回答，也不在续跑时反复抽奖。"""

from ..utils.benchmark_files import read_json, write_json
from ..inference_utils.model_client import failure_details
from .protocol import checked, digest, question_file


def identity(question, snapshot_id):
    return {"question_id": question.id, "question": question.text, "snapshot_id": snapshot_id,
            "query_time": question.query_time}


def read_failure(runtime, sample, phase, question, snapshot_id):
    folder = runtime.directory(sample) / phase.name
    path = question_file(folder / "failures", question.id)
    if not path.exists():
        return None
    record = checked(path, identity(question, snapshot_id), "单题失败终态")
    validate_failure(record)
    stage = record["failed_stage"]
    if stage == "retrieval":
        payload = identity(question, snapshot_id)
    else:
        kind = "retrievals" if stage == "answer" else "answers"
        payload = read_json(question_file(folder / kind, question.id))
        if stage == "judge":
            retrieval = read_json(question_file(folder / "retrievals", question.id))
            if payload.get("retrieval_sha256") != digest(retrieval):
                raise ValueError("失败评分对应的回答与检索产物不一致")
    if record["input_sha256"] != digest(payload):
        raise ValueError("失败记录对应的输入已改变，不能复用计零结果")
    return record


def validate_failure(record):
    if (record.get("status") != "technical_failure" or record.get("score_origin") != "technical_failure"
            or record.get("assigned_score") != 0 or record.get("failed_stage") not in {"retrieval", "answer", "judge"}
            or not isinstance(record.get("reason"), str) or not isinstance(record.get("error_type"), str)
            or not isinstance(record.get("request_ids"), list)):
        raise ValueError("无效的单题技术失败记录")


def save_failure(runtime, sample, phase, question, snapshot_id, stage, error, *, payload=None):
    binding = identity(question, snapshot_id)
    record = {**binding, "status": "technical_failure", "failed_stage": stage,
              **failure_details(error), "assigned_score": 0, "score_origin": "technical_failure",
              "input_sha256": digest(binding if payload is None else payload)}
    write_json(question_file(runtime.directory(sample) / phase.name / "failures", question.id), record)
    print(f"[{sample.key}/{phase.name}/{question.id}] {stage} 技术失败，计 0 分；继续其他题", flush=True)
    return record


def failure_result(record, answer=None):
    # 没有模型回答时明确为 null，不生成可在拒答题中得分的占位答案。
    return {**(answer or {}), **record, "answer_text": (answer or {}).get("answer_text"),
            "resolution_status": "incomplete", "scores": None}


def stored_failure_result(runtime, sample, phase, question, record):
    answer = None
    if record["failed_stage"] == "judge":
        answer = read_json(question_file(runtime.directory(sample) / phase.name / "answers", question.id))
    return failure_result(record, answer)
