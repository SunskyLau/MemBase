"""只读实验进度：观察已有产物，不介入记忆、评分或续跑判定。"""

from __future__ import annotations

from contextlib import closing
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sqlite3


STAGES = {"extract": "抽取", "reconcile": "协调", "generate": "派生/修复",
          "verify": "依赖验证", "verify_control": "控制复核", "embedding": "嵌入",
          "read_plan": "问题分解", "read_assess": "证据判断", "answer": "回答", "judge": "评分"}


def read_record(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


def readonly_database(path: Path):
    # mode=ro 禁止观察器创建数据库；短超时不阻塞实验写入。
    return sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=0.2)


class ExperimentProgress:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.previous: dict[str, str] = {}
        self.log_offsets: dict[Path, int] = {}

    def _requests(self, config: dict) -> str:
        ledger = Path(config.get("budget_ledger") or self.run_dir / "budget.sqlite")
        if not ledger.is_file():
            return ""
        with closing(readonly_database(ledger)) as db:
            counts = {(kind, status): n for kind, status, n in db.execute(
                "SELECT kind,status,count(*) FROM requests GROUP BY kind,status")}
            pending = db.execute("SELECT stage,started_at FROM requests WHERE status='reserved' ORDER BY id DESC LIMIT 1").fetchone()
        values = []
        for kind, label in (("llm", "模型"), ("embedding", "嵌入")):
            total = sum(n for (category, _), n in counts.items() if category == kind)
            failed = counts.get((kind, "failed"), 0)
            values.append(f"{label}请求 {total}（失败 {failed}）")
        if pending:
            started = datetime.fromisoformat(pending[1])
            if started.tzinfo is None:
                started = started.replace(tzinfo=timezone.utc)
            age = (datetime.now(timezone.utc) - started).total_seconds()
            values.append(f"最近未结束记录：{STAGES.get(pending[0], pending[0])}，{duration(age)}")
        prefix = "共用账本：" if config.get("budget_ledger") else ""
        return prefix + "；".join(values)

    def _ourmem(self) -> list[tuple[str, str]]:
        result = []
        for path in sorted((self.run_dir / "samples").rglob("manifest.json")):
            folder, manifest = path.parent, read_record(path)
            label = folder.relative_to(self.run_dir / "samples").as_posix()
            status = read_record(folder / "status.json").get("status")
            if status == "complete":
                result.append((label, f"[{label}] 已完成并评分"))
                continue
            parts = []
            for database in sorted((folder / "memory").glob("*.sqlite")):
                try:
                    with closing(readonly_database(database)) as db:
                        row = db.execute("SELECT payload FROM progress WHERE key='system:ingest'").fetchone()
                        ingest = json.loads(row[0]) if row else {}
                        done = ingest.get("processed_through", -1) + 1
                        memories = db.execute("SELECT count(*) FROM versions").fetchone()[0]
                        total = manifest.get("messages", "?")
                        parts.append(f"输入处理 {done}/{total}；记忆版本 {memories}（含历史）")
                        batch = ingest.get("pending_batch")
                        if batch:
                            row = db.execute("SELECT payload FROM progress WHERE key=?", ("batch:" + batch["id"],)).fetchone()
                            progress = json.loads(row[0]) if row else {}
                            if "extraction" in progress:
                                count = len(progress["extraction"].get("drafts", []))
                                parts.append(f"当前批事实 {progress.get('completed_drafts', 0)}/{count}")
                            else:
                                parts.append("当前批正在抽取")
                except (sqlite3.Error, ValueError, KeyError, TypeError):
                    parts.append("数据库进度暂不可读，下一次重试")
            for phase in manifest.get("phases", []):
                name, count = phase["name"], len(phase["question_ids"])
                answered = sum(1 for _ in (folder / name / "answers").glob("*.json"))
                scored = sum(1 for _ in (folder / name / "scores").glob("*.json"))
                parts.append(f"{name} 回答文件 {answered}/{count}" + (f"，评分文件 {scored}/{count}" if scored else ""))
            if (folder / "official_judge.json").is_file():
                parts.append("已生成官方评判文件，完成状态以校验为准")
            if status == "incomplete":
                error = read_record(folder / "status.json").get("error_type", "未记录原因")
                parts.append(f"未完成（{error}）；续跑时按原检查点处理")
            result.append((label, f"[{label}] " + ("；".join(parts) or "正在初始化")))
        return result

    def _baselines(self, config: dict, protocol: dict) -> list[tuple[str, str]]:
        result = []
        if config.get("benchmark") == "memoryagentbench":
            expected = 4 if config.get("mode") == "smoke" else 100
            for folder in sorted(self.run_dir.glob("factconsolidation_*")):
                if not folder.is_dir():
                    continue
                files = list(folder.glob("outputs/Conflict_Resolution/*_results.json"))
                counts = [len(read_record(path).get("data", [])) for path in files]
                result.append((folder.name, f"[{folder.name}] 回答结果文件记录 {max(counts, default=0)}/{expected}；完整性以官方结果校验为准"))
        else:
            totals = protocol.get("data", {}).get("episodes", {})
            for parent in ("smoke", "full"):
                for folder in sorted((self.run_dir / parent).glob("*")):
                    if not folder.is_dir():
                        continue
                    expected = 2 if parent == "smoke" else totals.get(folder.name, "?")
                    answers = sum(1 for _ in (folder / "outputs").glob("*.json"))
                    judges = sum(1 for _ in (folder / "judge").glob("*.json"))
                    label = f"{parent}/{folder.name}"
                    result.append((label, f"[{label}] 样本回答文件 {answers}/{expected}；评判文件 {judges}/{expected}"))
        return result

    def _official_logs(self) -> list[str]:
        result = []
        paths = [*self.run_dir.glob("factconsolidation_*/run.log"), *self.run_dir.glob("*/*/logs/*.log")]
        for path in sorted(paths):
            try:
                size = path.stat().st_size
                previous = self.log_offsets.get(path, 0)
                if previous == size:
                    continue
                start = previous if previous <= size else 0
                with path.open("rb") as stream:
                    stream.seek(max(start, size - 65536))
                    content = stream.read(65536).decode("utf-8", errors="replace")
                self.log_offsets[path] = size
                lines = [line.strip() for line in content.splitlines() if line.strip()]
                if lines:
                    latest = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", lines[-1])
                    result.append(f"[{path.relative_to(self.run_dir)}] {latest[:320]}")
            except OSError:
                continue
        return result

    def _model_errors(self) -> list[str]:
        result = []
        for path in (self.run_dir / "samples").rglob("requests.jsonl"):
            try:
                size = path.stat().st_size
                previous = self.log_offsets.get(path, 0)
                if size == previous:
                    continue
                with path.open("rb") as stream:
                    stream.seek(max(previous if previous <= size else 0, size - 65536))
                    lines = stream.read(65536).decode("utf-8", errors="replace").splitlines()
                self.log_offsets[path] = size
                for line in lines:
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue  # 长日志从尾部读取时首行可能是不完整的。
                    if not isinstance(row, dict):
                        continue
                    error = row.get("validation_error") or row.get("error")
                    if error:
                        label = path.parent.relative_to(self.run_dir / "samples")
                        result.append(f"[{label}] 请求 {row.get('id')} {STAGES.get(row.get('stage'), row.get('stage', ''))}：{str(error)[:320]}")
            except OSError:
                continue
        return result

    def poll(self) -> list[str]:
        manifest = read_record(self.run_dir / "config.json")
        status = read_record(self.run_dir / "status.json").get("status", "等待初始化")
        header = f"[{datetime.now(timezone.utc):%H:%M:%S} UTC] 记录状态：{status}"
        if not manifest:
            return [header + "；正在检查环境、数据或运行配置"]
        config = manifest.get("config", {})
        try:
            requests = self._requests(config)
            if requests:
                header += "；" + requests
        except (sqlite3.Error, ValueError, TypeError):
            header += "；请求账本暂不可读"
        result = [header]
        rows = self._ourmem() if config.get("baseline") == "ourmem" else self._baselines(config, manifest.get("protocol", {}))
        for key, line in rows:
            if self.previous.get(key) != line:
                result.append(line)
                self.previous[key] = line
        result.extend(self._official_logs())
        result.extend(self._model_errors())
        return result
