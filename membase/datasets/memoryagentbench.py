"""MemoryAgentBench 官方冲突消解数据：准备、读取和问题清单。"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from ..utils.benchmark_files import (
    REPO_ROOT, download_verified, git_output, prepare_repository, read_json,
    verify_hash, verify_repository, write_json,
)

UPSTREAM_URL = "https://github.com/HUST-AI-HYZ/MemoryAgentBench.git"
UPSTREAM_COMMIT = "fe1735de8cf8b9908e1e3d3b5612afc815698062"
DATASET_REVISION = "7ea066982b140a19337e17e60d45d4076e042faf"
DATA_SHA256 = "24d5c3f09ce0ce15625cb9f8a98f44f0d864ca6c94d7b4ad04eb697ca3a5ff45"
FILENAME = "Conflict_Resolution-00000-of-00001.parquet"
DEFAULT_DATA_ROOT = REPO_ROOT / "data/memoryagentbench"
DEFAULT_UPSTREAM = REPO_ROOT / "external/MemoryAgentBench"
LOADER_FILE = "utils/eval_data_utils.py"
LOADER_NEEDLE = '        raw_data = load_dataset(dataset_name, split=split_name, revision="main")'
LOADER_REPLACEMENT = '''        # CLAIMMEM_LOCAL_CONFLICT_DATA: use the pinned local Parquet when provided.
        local_conflict_path = os.environ.get("MAB_CONFLICT_PARQUET")
        if local_conflict_path and split_name == "Conflict_Resolution":
            raw_data = load_dataset(
                "parquet",
                data_files={split_name: local_conflict_path},
                split=split_name,
            )
        else:
            raw_data = load_dataset(dataset_name, split=split_name, revision="main")'''


def subsets(mode: str) -> list[str]:
    if mode == "smoke":
        return ["factconsolidation_sh_6k"]
    if mode not in {"core", "full"}:
        raise ValueError(f"未知实验模式：{mode}")
    lengths = ("6k", "32k") if mode == "core" else ("6k", "32k", "64k", "262k")
    return [f"factconsolidation_{hop}_{length}" for length in lengths for hop in ("sh", "mh")]


def raw_path(data_root: Path) -> Path:
    return data_root / "raw" / FILENAME


def expected_loader(upstream: Path) -> str:
    original = git_output(upstream, "show", f"{UPSTREAM_COMMIT}:{LOADER_FILE}") + "\n"
    if original.count(LOADER_NEEDLE) != 1:
        raise ValueError("固定版本的官方加载器与预期不符")
    return original.replace(LOADER_NEEDLE, LOADER_REPLACEMENT)


def verify_source(upstream: Path) -> None:
    verify_repository(upstream, UPSTREAM_COMMIT)
    if (upstream / LOADER_FILE).read_text(encoding="utf-8").rstrip() != expected_loader(upstream).rstrip():
        raise ValueError("官方加载器缺少或修改了已约定的本地 Parquet 补丁")
    changed = git_output(upstream, "diff", "HEAD", "--name-only").splitlines()
    if set(changed) - {LOADER_FILE}:
        raise ValueError(f"官方代码有未记录的改动：{changed}")


def load_raw(data_root: Path) -> list[dict]:
    """仅实际解析 Parquet 时导入 pyarrow；帮助和干运行无需该包。"""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("首次生成 MAB 问题清单需要在已有含 pyarrow 的环境执行数据准备") from exc
    return pq.read_table(raw_path(data_root)).to_pylist()


def build_manifest(data_root: Path) -> dict:
    questions = {}
    for row in load_raw(data_root):
        source = row["metadata"]["source"]
        ids = row["metadata"]["qa_pair_ids"]
        if source in questions or len(ids) != 100 or len(set(ids)) != 100:
            raise ValueError(f"MAB 子集或问题标识重复/缺失：{source}")
        if len(row["questions"]) != 100 or len(row["answers"]) != 100 or not row["context"]:
            raise ValueError(f"MAB 子集内容不完整：{source}")
        questions[source] = [
            {"id": qid, "question": question}
            for qid, question in zip(ids, row["questions"])
        ]
    if set(questions) != set(subsets("full")):
        raise ValueError("MAB 必须包含八个 FactConsolidation 子集")
    manifest = {"dataset_revision": DATASET_REVISION, "raw_sha256": DATA_SHA256,
                "questions": questions}
    write_json(data_root / "manifest.json", manifest)
    return manifest


def load_manifest(data_root: Path) -> dict:
    manifest = read_json(data_root / "manifest.json")
    if manifest["dataset_revision"] != DATASET_REVISION or manifest["raw_sha256"] != DATA_SHA256:
        raise ValueError("MAB 问题清单与固定数据版本不一致")
    if set(manifest["questions"]) != set(subsets("full")):
        raise ValueError("MAB 问题清单缺少子集")
    for source, questions in manifest["questions"].items():
        if len(questions) != 100 or len({q["id"] for q in questions}) != 100:
            raise ValueError(f"MAB 问题清单不完整：{source}")
    return manifest


def check(data_root: Path = DEFAULT_DATA_ROOT, upstream: Path = DEFAULT_UPSTREAM) -> dict:
    verify_hash(raw_path(data_root), DATA_SHA256)
    verify_source(upstream)
    manifest = load_manifest(data_root)
    return {"benchmark": "memoryagentbench", "subsets": len(manifest["questions"]),
            "questions": sum(map(len, manifest["questions"].values())), "raw_sha256": DATA_SHA256}


def prepare(data_root: Path = DEFAULT_DATA_ROOT, upstream: Path = DEFAULT_UPSTREAM) -> dict:
    prepare_repository(upstream, UPSTREAM_URL, UPSTREAM_COMMIT,
                       data_root / "upstream/MemoryAgentBench")
    target = upstream / LOADER_FILE
    patched = expected_loader(upstream)
    current = target.read_text(encoding="utf-8")
    original = git_output(upstream, "show", f"{UPSTREAM_COMMIT}:{LOADER_FILE}")
    if current.rstrip() == original.rstrip():
        target.write_text(patched, encoding="utf-8")
    elif current.rstrip() != patched.rstrip():
        raise ValueError("已有加载器改动无法安全迁移，保留原文件")
    url = f"https://huggingface.co/datasets/ai-hyz/MemoryAgentBench/resolve/{DATASET_REVISION}/data/{FILENAME}"
    download_verified(url, raw_path(data_root), DATA_SHA256)
    if not (data_root / "manifest.json").exists() or importlib.util.find_spec("pyarrow"):
        build_manifest(data_root)
    return check(data_root, upstream)
