"""MEME 官方数据的版本、解包和变化前后问题定位。"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import subprocess
import sys

from ..utils.benchmark_files import (
    REPO_ROOT, download_verified, git_output, prepare_repository, read_json,
    verify_hash, verify_repository,
)

UPSTREAM_URL = "https://github.com/SeokwonJung-Jay/MEME-public.git"
UPSTREAM_COMMIT = "0271ad85389a963cbc4892a36391f868ba4d18d1"
DATASET_REVISION = "03932fd33a08debf182ad01a47504024201d86f2"
DATA_HASHES = {
    "nofiller": "1687d028cc1638986df9f58f0c3f072f614cf13d72a791541b4439fddf701636",
    "filler32k": "a88d28374a002b3e5b1683fb7201d06a1ce739d2ebf94c971c37bb65cf6ebdd3",
    "filler128k": "eb861f866dde067e6c7ea6db9bdf17b83a56cdc578dcca57d9d8809dc22747fe",
}
DOMAIN_PREFIX = {"personal_life": "pl", "software_project": "sw"}
TASKS = {"ER", "Agg", "Tr", "Del", "Cas", "Abs"}
DEFAULT_DATA_ROOT = REPO_ROOT / "data/meme"
DEFAULT_UPSTREAM = REPO_ROOT / "external/MEME-public"


def question_key(question: dict) -> tuple:
    """官方问题没有统一 id，以类型、原始问题和目标实体共同定位。"""
    return (question["task_type"], question["question"],
            tuple(sorted(question["entity_values"])))


def episode_key(episode: dict) -> str:
    return f"{DOMAIN_PREFIX[episode['domain']]}_{episode['episode_id']:03d}"


def raw_path(data_root: Path, variant: str) -> Path:
    return data_root / "raw" / f"meme_{variant}.json"


def episode_path(data_root: Path, variant: str, episode: dict) -> Path:
    return (data_root / "unpacked" / f"{variant}_{DOMAIN_PREFIX[episode['domain']]}"
            / f"episode_{episode['episode_id']:03d}.json")


def load_raw(data_root: Path, variant: str) -> list[dict]:
    rows = read_json(raw_path(data_root, variant))
    per_domain = 20 if variant == "filler128k" else 50
    if Counter(ep["domain"] for ep in rows) != Counter({domain: per_domain for domain in DOMAIN_PREFIX}):
        raise ValueError(f"MEME {variant} 的领域/样本数量不正确")
    if len({ep["episode_id"] for ep in rows}) != len(rows):
        raise ValueError(f"MEME {variant} 含重复样本")
    if {task["type"] for ep in rows for task in ep["tasks"]} != TASKS:
        raise ValueError(f"MEME {variant} 的任务类型不完整")
    episodes = []
    for row in rows:
        prefix, number = row["episode_id"].split("_")
        if prefix != DOMAIN_PREFIX[row["domain"]]:
            raise ValueError("MEME 样本标识与领域不一致")
        episode = {**row, "episode_id": int(number)}
        before = row["before_questions"]["position_after_session"]
        after = row["after_questions"]["position_after_session"]
        if not 0 <= before < after < len(row["sessions"]):
            raise ValueError(f"MEME 提问位置异常：{row['episode_id']}")
        for phase in ("before_questions", "after_questions"):
            questions = row[phase]["questions"]
            if not questions or len({question_key(q) for q in questions}) != len(questions):
                raise ValueError(f"MEME 问题缺失或重复：{row['episode_id']}/{phase}")
        episodes.append(episode)
    return episodes


def stages(mode: str) -> list[tuple[str, bool]]:
    if mode == "smoke":
        return [("nofiller", True)]
    if mode == "core":
        return [("nofiller", True), ("filler32k", False)]
    if mode == "full":
        return [(variant, False) for variant in DATA_HASHES]
    raise ValueError(f"未知实验模式：{mode}")


def select_episodes(data_root: Path, variant: str, smoke: bool = False) -> list[dict]:
    episodes = load_raw(data_root, variant)
    if smoke:
        return [next(ep for ep in episodes if ep["domain"] == domain) for domain in DOMAIN_PREFIX]
    return episodes


def check(data_root: Path = DEFAULT_DATA_ROOT, upstream: Path = DEFAULT_UPSTREAM) -> dict:
    verify_repository(upstream, UPSTREAM_COMMIT)
    if git_output(upstream, "diff", "HEAD", "--name-only"):
        raise ValueError("MEME 官方代码存在未记录的修改")
    counts = {}
    for variant, digest in DATA_HASHES.items():
        verify_hash(raw_path(data_root, variant), digest)
        episodes = load_raw(data_root, variant)
        expected = {episode_path(data_root, variant, ep) for ep in episodes}
        actual = set()
        for prefix in DOMAIN_PREFIX.values():
            actual.update((data_root / "unpacked" / f"{variant}_{prefix}").glob("episode_*.json"))
        if actual != expected:
            raise ValueError(f"MEME 解包文件缺失或多余：{variant}；请执行数据准备")
        for ep in episodes:
            if read_json(episode_path(data_root, variant, ep)) != ep:
                raise ValueError(f"MEME 解包内容与原始文件不一致：{episode_key(ep)}")
        counts[variant] = len(episodes)
    return {"benchmark": "meme", "episodes": counts, "raw_sha256": DATA_HASHES}


def prepare(data_root: Path = DEFAULT_DATA_ROOT, upstream: Path = DEFAULT_UPSTREAM) -> dict:
    prepare_repository(upstream, UPSTREAM_URL, UPSTREAM_COMMIT,
                       data_root / "upstream/MEME-public")
    for variant, digest in DATA_HASHES.items():
        url = f"https://huggingface.co/datasets/meme-benchmark/MEME/resolve/{DATASET_REVISION}/meme_{variant}.json"
        download_verified(url, raw_path(data_root, variant), digest)
        episodes = load_raw(data_root, variant)
        missing = False
        for ep in episodes:
            path = episode_path(data_root, variant, ep)
            if path.exists() and read_json(path) != ep:
                raise ValueError(f"已有解包文件被修改，保留原文件：{path}")
            missing |= not path.exists()
        if missing:
            subprocess.run([sys.executable, str(upstream / "code/dataset_tools/unpack_dataset.py"),
                            "--input", str(raw_path(data_root, variant)),
                            "--output", str(data_root / "unpacked")], check=True)
    return check(data_root, upstream)
