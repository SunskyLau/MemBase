"""显式复用已完成构建：只重新执行静态问答的读取阶段，不改写原始构建指纹。"""
from __future__ import annotations

from contextlib import closing
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3

from .benchmark_files import read_json, write_json, sha256_file
from .ourmem_version import ImplementationMismatchError, ourmem_fingerprint


# 后三项是引入读取修订所需的调度接入，不在续跑中执行构建。
READ_REVISION_FILES = {
    "membase/ourmem/reader.py", "membase/ourmem/evidence_format.py", "membase/ourmem/calculation.py",
    "membase/runners/protocol.py", "membase/utils/ourmem_version.py", "membase/utils/read_revision.py",
}
_TABLES = ("metadata", "sources", "versions", "dependencies", "refs", "operations", "snapshots", "progress")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def changed_files(previous, current):
    a, b = previous["files"], current["files"]
    return sorted(key for key in a.keys() | b.keys() if a.get(key) != b.get(key))


def _supported(saved):
    config = saved["config"]
    if config["baseline"] != "ourmem" or config["benchmark"] not in {"locomo", "longmemeval", "memoryagentbench"}:
        raise ValueError("读取修订仅支持已完成构建的 OurMem 静态问答；MEME 观察点不允许这样回填")


def construction_state(run_dir):
    """不调用模型、不打开可写存储；不把读取向量缓存算成构建内容变化。"""
    from ..datasets.official import episode_manifest, load_episodes
    run_dir = Path(run_dir)
    saved = read_json(run_dir / "config.json")
    _supported(saved)
    cfg = saved["config"]
    result = {}
    for sample in load_episodes(cfg["benchmark"], Path(cfg["data_root"]), cfg["mode"]):
        folder = run_dir / "samples" / sample.key
        manifest = episode_manifest(sample)
        if read_json(folder / "manifest.json") != manifest:
            raise ValueError(f"样本输入发生变化：{sample.key}")
        receipt = read_json(folder / "construction.json")
        if not receipt.get("complete") or receipt.get("manifest") != manifest:
            raise ValueError(f"构建未完成：{sample.key}")
        checkpoint = read_json(folder / "checkpoint.json")
        if checkpoint["sessions_ingested"] != len(sample.sessions):
            raise ValueError(f"输入尚未完整摄入：{sample.key}")
        databases = list((folder / "memory").glob("*.sqlite"))
        if len(databases) != 1 or databases[0].is_symlink():
            raise ValueError(f"需要一个独立的本地记忆数据库：{sample.key}")
        db = databases[0]
        with closing(sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
            namespace = connection.execute("SELECT value FROM metadata WHERE key='namespace'").fetchone()[0]
            if namespace != sample.namespace:
                raise ValueError(f"记忆命名空间不一致：{sample.key}")
            tables = {name: _digest(connection.execute(f'SELECT * FROM "{name}" ORDER BY rowid').fetchall())
                      for name in _TABLES}
            phases = {}
            for phase in sample.phases:
                point = checkpoint["phases"][phase.name]
                prefix = "snapshot-" + hashlib.sha256(json.dumps([namespace], ensure_ascii=False, sort_keys=True,
                                                                  separators=(",", ":")).encode()).hexdigest()[:24] + ":"
                if not point["snapshot_id"].startswith(prefix):
                    raise ValueError(f"快照命名空间不一致：{sample.key}")
                path = folder / phase.name / "memory_snapshot.json"
                if sha256_file(path) != point["snapshot_sha256"]:
                    raise ValueError(f"构建快照损坏：{sample.key}/{phase.name}")
                snapshot = connection.execute("SELECT id FROM snapshots WHERE id=?", (int(point["snapshot_id"].rsplit(":",1)[1]),)).fetchone()
                if snapshot is None:
                    raise ValueError(f"数据库缺少观察点快照：{sample.key}")
                phases[phase.name] = dict(point)
        result[sample.key] = {"database": db.relative_to(run_dir).as_posix(), "tables": tables, "phases": phases,
                              "files": {name: sha256_file(folder / name) for name in
                                        ("manifest.json", "construction.json", "checkpoint.json", "source_mapping.json")}}
    if not result:
        raise ValueError("没有可复用的构建样本")
    return result


def validate_revision(run_dir, saved, current_protocol, stages):
    run_dir = Path(run_dir)
    _supported(saved)
    if not stages or set(stages) - {"search", "evaluation"}:
        raise ImplementationMismatchError("此运行已绑定读取修订；请使用续跑脚本，仅执行 search/evaluation，不能重新构建")
    record = read_json(run_dir / "read_revision.json")
    if record.get("state") != "ready" or record.get("original_manifest_sha256") != sha256_file(run_dir / "config.json"):
        raise ImplementationMismatchError("读取修订记录尚未准备好，或原始运行配置已改变")
    old_protocol = saved["protocol"]
    for key in current_protocol.keys() | old_protocol.keys():
        if key != "implementation" and current_protocol.get(key) != old_protocol.get(key):
            raise ImplementationMismatchError("数据或方法配置变化，不能复用原构建")
    if record["read_implementation"] != current_protocol["implementation"]:
        raise ImplementationMismatchError("读取代码再次变化；需显式准备新的读取修订并清理旧读取产物")
    changes = changed_files(old_protocol["implementation"], current_protocol["implementation"])
    if set(changes) - READ_REVISION_FILES:
        raise ImplementationMismatchError("构建相关代码发生变化，禁止用读取修订跳过重建")
    if record["construction"] != construction_state(run_dir):
        raise ImplementationMismatchError("构建内容、输入或快照发生变化，不能继续复用")
    return record


def _ensure_stopped(run_dir):
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            args = (process / "cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        runner = any(arg.endswith((b"/run_benchmark.py", b"/memory_search.py", b"/memory_evaluation.py", b"/memory_construction.py")) for arg in args)
        if runner and (str(run_dir).encode() in args or run_dir.name.encode() in args):
            raise ValueError("实验进程仍在运行；先停止后才能清理读取产物")


def prepare_revision(run_dir, *, clear=False):
    """清理仅在显式准备时执行；普通续跑不会反复删除新结果。"""
    from ..datasets.official import fingerprint
    run_dir = Path(run_dir).resolve()
    _ensure_stopped(run_dir)
    saved = read_json(run_dir / "config.json")
    _supported(saved)
    cfg = saved["config"]
    if Path(cfg["run_dir"]).resolve() != run_dir:
        raise ValueError("不允许用移动后的运行目录隐式迁移构建")
    data = fingerprint(cfg["benchmark"], Path(cfg["data_root"]), Path(cfg["upstream_dir"]))
    if any(saved["protocol"].get(key) != value for key, value in data.items()):
        raise ValueError("官方数据或协议发生变化")
    current = ourmem_fingerprint(official_roots={cfg["benchmark"]: Path(cfg["upstream_dir"])})
    changes = changed_files(saved["protocol"]["implementation"], current)
    if set(changes) - READ_REVISION_FILES:
        raise ValueError(f"存在构建相关修改，不能复用：{sorted(set(changes)-READ_REVISION_FILES)}")
    construction = construction_state(run_dir)
    discard = []
    for sample, item in construction.items():
        folder = run_dir / "samples" / sample
        for phase in item["phases"]:
            discard.extend(folder / phase / name for name in ("retrievals", "answers", "scores", "failures"))
        discard.extend(folder / name for name in ("search.json", "search_results.json", "evaluation.json", "status.json"))
    discard.extend(run_dir / name for name in ("search.json", "evaluation.json", "summary.json", "partial_summary.json", "failures.json"))
    discard = [path for path in discard if path.exists()]
    if any(path.is_symlink() or not path.resolve().is_relative_to(run_dir) for path in discard):
        raise ValueError("读取产物包含越界路径或符号链接；不执行清理")
    count = sum(len(list(path.rglob('*.json'))) if path.is_dir() else 1 for path in discard)
    result = {"changed_files": changes, "samples": sorted(construction), "discarded_json_count": count,
              "stages": ["search", "evaluation"], "dry_run": not clear}
    if not clear:
        return result
    history = run_dir / "read_revisions"
    revision = f"{1 + max((int(p.name) for p in history.glob('*') if p.is_dir() and p.name.isdigit()), default=0):03d}"
    archive = history / revision
    record = {"state": "preparing", "id": revision, "original_manifest_sha256": sha256_file(run_dir / "config.json"),
              "construction_implementation": saved["protocol"]["implementation"], "read_implementation": current,
              "construction": construction, "discarded": [p.relative_to(run_dir).as_posix() for p in discard],
              "previous_status": read_json(run_dir / "status.json"), "prior_query_costs_retained": True}
    record["request_log_offsets"] = {p.relative_to(run_dir).as_posix(): p.stat().st_size
                                     for p in (run_dir / "samples").glob("*/requests.jsonl")}
    write_json(run_dir / "read_revision.json", record)
    for path in discard:
        destination = archive / "discarded" / path.relative_to(run_dir)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(destination))
    if construction_state(run_dir) != construction:
        raise RuntimeError("清理后构建内容不一致；停止，不签发复用记录")
    record["state"] = "ready"
    write_json(archive / "receipt.json", record)
    write_json(run_dir / "read_revision.json", record)
    write_json(run_dir / "status.json", {"status": "ready", "stage": "search", "read_revision": revision})
    return {**result, "revision": revision, "archive": str(archive)}


def fork_read_revision(source_dir, run_dir):
    """显式创建读取对照；复制构建及费用历史，不移动原实验的任何产物。"""
    source_dir, run_dir = Path(source_dir).resolve(), Path(run_dir).resolve()
    prepare_revision(source_dir)  # 只读检查：仅允许读取侧代码变化。
    if run_dir.exists() or run_dir.is_relative_to(source_dir):
        raise ValueError("读取对照需要独立、尚不存在的运行目录")
    saved = read_json(source_dir / "config.json")
    state = construction_state(source_dir)
    run_dir.mkdir(parents=True)
    copied = {**saved, "config": {**saved["config"], "run_dir": str(run_dir)}}
    write_json(run_dir / "config.json", copied)
    shutil.copy2(source_dir / "execution.json", run_dir / "execution.json")
    # 预算与日志一起继承；新旧请求编号不会冲突，已有费用不被隐藏。
    with closing(sqlite3.connect((source_dir / "budget.sqlite").as_uri() + "?mode=ro", uri=True)) as original, \
         closing(sqlite3.connect(run_dir / "budget.sqlite")) as destination:
        original.backup(destination)
    for key, item in state.items():
        original, destination = source_dir / "samples" / key, run_dir / "samples" / key
        destination.mkdir(parents=True)
        for name in (*item["files"], "requests.jsonl"):
            shutil.copy2(original / name, destination / name)
        shutil.copytree(original / "memory", destination / "memory")
        for phase in item["phases"]:
            (destination / phase).mkdir()
            shutil.copy2(original / phase / "memory_snapshot.json", destination / phase / "memory_snapshot.json")
    assert construction_state(run_dir) == state
    write_json(run_dir / "status.json", {"status": "ready", "stage": "search"})
    result = prepare_revision(run_dir, clear=True)
    provenance = {"source_run": str(source_dir), "source_config_sha256": sha256_file(source_dir / "config.json"),
                  "source_summary_sha256": sha256_file(source_dir / "summary.json"),
                  "construction_reused": True, "prior_query_costs_inherited": True,
                  "budget_before": read_json(source_dir / "summary.json")["request_costs"]["shared_budget_totals"]}
    write_json(run_dir / "comparison_source.json", provenance)
    return {**result, "source_run": str(source_dir)}
