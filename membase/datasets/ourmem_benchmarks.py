"""官方格式到 OurMem 的白名单输入；答案及任务标签只保留在评测侧。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Iterator

from . import memoryagentbench as mab, meme
from ..utils.benchmark_files import REPO_ROOT, git_output, read_json, verify_hash, verify_repository


QA_DATA = {
    "locomo": {
        "filename": "locomo10.json", "count": 10,
        "sha256": "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4",
        "url": "https://github.com/snap-research/locomo.git",
        "commit": "3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376", "directory": "locomo",
    },
    "longmemeval": {
        "filename": "longmemeval_s_cleaned.json", "count": 500,
        "sha256": "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442",
        "url": "https://github.com/xiaowu0162/LongMemEval.git",
        "commit": "9e0b455f4ef0e2ab8f2e582289761153549043fc", "directory": "LongMemEval",
    },
}


@dataclass(frozen=True)
class Question:
    id: str
    text: str
    query_time: str | None
    reference: dict  # 绝不传给记忆系统。


@dataclass(frozen=True)
class Phase:
    name: str
    session_end: int
    questions: tuple[Question, ...]


@dataclass(frozen=True)
class Episode:
    key: str
    namespace: str
    sessions: tuple[tuple[dict, ...], ...]
    phases: tuple[Phase, ...]
    input_policy: dict
    reference: dict
    source_mapping: dict


def default_paths(benchmark: str) -> tuple[Path, Path]:
    if benchmark in QA_DATA:
        return REPO_ROOT / "data", REPO_ROOT / "external" / QA_DATA[benchmark]["directory"]
    module = mab if benchmark == "memoryagentbench" else meme
    return module.DEFAULT_DATA_ROOT, module.DEFAULT_UPSTREAM


def prepare_qa(benchmark: str, data_root: Path | None = None,
               upstream: Path | None = None, *, check_only: bool = False) -> dict:
    data_default, upstream_default = default_paths(benchmark)
    data_root, upstream = data_root or data_default, upstream or upstream_default
    spec = QA_DATA[benchmark]
    verify_hash(data_root / spec["filename"], spec["sha256"])
    if not upstream.exists() and not check_only:
        # 新副本自带 Git 对象；不移动旧项目，也不依赖旧路径运行。
        local = REPO_ROOT.parent / "ours/external" / spec["directory"]
        origin = str(local) if (local / ".git").is_dir() else spec["url"]
        upstream.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--quiet", "--no-hardlinks", origin, str(upstream)], check=True)
        subprocess.run(["git", "-C", str(upstream), "checkout", "--quiet", "--detach", spec["commit"]], check=True)
        subprocess.run(["git", "-C", str(upstream), "remote", "set-url", "origin", spec["url"]], check=True)
    verify_repository(upstream, spec["commit"])
    if git_output(upstream, "diff", "HEAD", "--name-only"):
        raise ValueError(f"官方副本有未记录修改：{upstream}")
    rows = read_json(data_root / spec["filename"])
    if len(rows) != spec["count"]:
        raise ValueError(f"{benchmark} 样本数量不符")
    return {"benchmark": benchmark, "episodes": len(rows), "sha256": spec["sha256"],
            "upstream_commit": spec["commit"]}


def fingerprint(benchmark: str, data_root: Path, upstream: Path) -> dict:
    if benchmark in QA_DATA:
        return prepare_qa(benchmark, data_root, upstream, check_only=True)
    return (mab if benchmark == "memoryagentbench" else meme).check(data_root, upstream)


def _opaque(kind: str, value: str) -> str:
    return f"{kind}-{hashlib.sha256(value.encode()).hexdigest()[:24]}"


def _iso(value: str | None, style: str) -> str | None:
    return datetime.strptime(value, style).isoformat() if value else None


def _messages(raw: list[dict], namespace: str, session_index: int,
              timestamp: str | None, *, locomo: bool = False) -> tuple[tuple[dict, ...], dict]:
    conversation_id = _opaque("conversation", f"{namespace}:{session_index}")
    output, mapping = [], {}
    for i, message in enumerate(raw):
        message_id = _opaque("message", f"{conversation_id}:{i}")
        content = message["text"] if locomo else message["content"]
        if locomo and message.get("blip_caption"):
            content += f"\n[Image caption: {message['blip_caption']}]"
        role = "user" if locomo else message["role"]
        output.append({"message_id": message_id, "conversation_id": conversation_id,
                       "speaker": message["speaker"] if locomo else role,
                       "role": role, "content": content, "mention_time": timestamp})
        mapping[message_id] = {"session_index": session_index, "message_index": i,
                               "original_id": message.get("dia_id")}
    return tuple(output), mapping


def load_episodes(benchmark: str, data_root: Path, mode: str) -> Iterator[Episode]:
    """问题标签仅存在 Question.reference；sessions 是唯一的写入输入。"""
    if benchmark == "meme":
        for variant, smoke in meme.stages(mode):
            for raw in meme.select_episodes(data_root, variant, smoke):
                key = f"{variant}/{meme.episode_key(raw)}"
                namespace = _opaque("memory", key)
                sessions, mapping = [], {}
                for i, session in enumerate(raw["sessions"]):
                    messages, ids = _messages(session["conversation"], namespace, i, session.get("timestamp"))
                    sessions.append(messages)
                    mapping.update(ids)
                phases = []
                for phase in ("before", "after"):
                    info = raw[f"{phase}_questions"]
                    questions = tuple(Question(f"{phase}-{i}", q["question"], None, q)
                                      for i, q in enumerate(info["questions"]))
                    phases.append(Phase(phase, info["position_after_session"] + 1, questions))
                yield Episode(key, namespace, tuple(sessions), tuple(phases),
                              {"description": "Messages retain original user and assistant roles; explicit user corrections and deletion requests are authoritative.",
                               "control_roles": ["user"]}, raw, mapping)
        return
    if benchmark == "memoryagentbench":
        selected = set(mab.subsets(mode))
        for raw in mab.load_raw(data_root):
            key = raw["metadata"]["source"]
            if key not in selected:
                continue
            namespace = _opaque("memory", key)
            # 保留每行公开序号和完整文本；未编号的引言也不丢弃。
            lines = raw["context"].splitlines(keepends=True)
            session, mapping = _messages([{"role": "user", "content": text} for text in lines if text.strip()], namespace, 0, None)
            limit = 4 if mode == "smoke" else len(raw["questions"])
            questions = tuple(Question(raw["metadata"]["qa_pair_ids"][i], q, None,
                                       {"answer": raw["answers"][i], "source": key})
                              for i, q in enumerate(raw["questions"][:limit]))
            policy = {"description": "Each fact has a public serial number. Larger serial numbers override conflicting older facts, even when real-world knowledge differs. This is knowledge precedence, not proof of a real-world change.",
                      "control_roles": ["user"], "update_priority": "newer_source"}
            yield Episode(key, namespace, (session,), (Phase("final", 1, questions),), policy, raw, mapping)
        return
    rows = read_json(data_root / QA_DATA[benchmark]["filename"])
    for index, raw in enumerate(rows[:1] if mode == "smoke" else rows):
        key = f"sample_{index:04d}"
        namespace = _opaque("memory", f"{benchmark}:{index}")
        sessions, mapping = [], {}
        if benchmark == "locomo":
            conv = raw["conversation"]
            for number in sorted(int(k.split("_")[1]) for k in conv if re.fullmatch(r"session_\d+", k)):
                timestamp = _iso(conv[f"session_{number}_date_time"], "%I:%M %p on %d %B, %Y")
                messages, ids = _messages(conv[f"session_{number}"], namespace, number, timestamp, locomo=True)
                sessions.append(messages)
                mapping.update(ids)
            selected = [(i, q) for i, q in enumerate(raw["qa"]) if q["category"] in {1, 2, 3, 4}]
            questions = tuple(Question(f"{key}-{i}", q["question"], None, q)
                              for i, q in (selected[:4] if mode == "smoke" else selected))
        else:
            if not len(raw["haystack_sessions"]) == len(raw["haystack_dates"]) == len(raw["haystack_session_ids"]):
                raise ValueError(f"LongMemEval 会话和时间数量不符：{key}")
            for i, (session, timestamp) in enumerate(zip(raw["haystack_sessions"], raw["haystack_dates"])):
                messages, ids = _messages(session, namespace, i, _iso(timestamp, "%Y/%m/%d (%a) %H:%M"))
                sessions.append(messages)
                mapping.update(ids)
            questions = (Question(raw["question_id"], raw["question"],
                                  _iso(raw["question_date"], "%Y/%m/%d (%a) %H:%M"),
                                  {key: raw[key] for key in ("question_type", "question_date", "answer", "answer_session_ids")}),)
        policy = {"description": ("Both named speakers are humans, not assistant outputs." if benchmark == "locomo" else
                                   "Preserve user and assistant roles; assistant statements describe advice or prior assistant outputs, not completed user actions."),
                  "control_roles": ["user"]}
        yield Episode(key, namespace, tuple(sessions), (Phase("final", len(sessions), questions),), policy, raw, mapping)


def episode_manifest(episode: Episode) -> dict:
    """评测侧身份、输入摘要与问题清单；不作为模型上下文。"""
    payload = {"sessions": episode.sessions, "policy": episode.input_policy,
               "phases": [{"name": p.name, "session_end": p.session_end,
                           "questions": [{"id": q.id, "text": q.text, "query_time": q.query_time,
                                          "reference": q.reference} for q in p.questions]} for p in episode.phases]}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return {"key": episode.key, "input_sha256": digest, "namespace": episode.namespace,
            "sessions": len(episode.sessions), "messages": sum(map(len, episode.sessions)),
            "phases": [{"name": p.name, "session_end": p.session_end,
                        "question_ids": [q.id for q in p.questions]} for p in episode.phases]}
