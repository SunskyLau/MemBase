"""统一来源、内容版本、依赖和操作的追加式 SQLite 存储。"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Iterable

from .models import (
    ControlOperation, DependencyLink, MemoryVersion, PremiseRef, Snapshot, Source, utc_now,
)
from .persistence import canonical_json, open_database


@dataclass(frozen=True)
class MemoryView:
    """固定提交与来源前缀；对象作为只读值使用，不原地修改缓存状态。"""

    namespace: str
    sequence: int
    source_cutoff: int
    sources: dict[str, Source]
    versions: dict[str, MemoryVersion]
    dependencies: dict[str, DependencyLink]
    operations: list[ControlOperation]
    version_sequences: dict[str, int]
    dependency_sequences: dict[str, int]
    operation_sequences: dict[str, int]


def reference_ids(ref: PremiseRef) -> Iterable[tuple[str, str]]:
    yield (ref.type.value, ref.id)
    for span in ref.context_refs:
        yield ("SOURCE", span.source_id)


def dependency_signature(dependency: DependencyLink) -> str:
    data = dependency.model_dump(mode="json", exclude={"id", "created_at"})
    refs = []
    for ref in dependency.premise_refs:
        value = ref.model_dump(mode="json")
        value["context_refs"] = sorted(value["context_refs"], key=canonical_json)
        refs.append(canonical_json(value))
    data["premise_refs"] = sorted(
        set(refs)
    )
    return sha256(canonical_json(data).encode()).hexdigest()


class OurMemStore:
    """一个实例只拥有一个命名空间；写入组要么全部提交，要么全部回滚。"""

    def __init__(
        self, path: str | Path, namespace: str, *,
        max_claim_depth: int = 5, max_premises_per_dependency: int = 8,
    ) -> None:
        self.path, self.namespace = str(path), namespace
        self.max_claim_depth = max_claim_depth
        self.max_premises_per_dependency = max_premises_per_dependency
        self.connection = open_database(path, namespace)
        self._views: OrderedDict[tuple[int, int, int], MemoryView] = OrderedDict()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> OurMemStore:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    @property
    def current_seq(self) -> int:
        return self.connection.execute("SELECT COALESCE(MAX(seq), 0) FROM commits").fetchone()[0]

    @property
    def source_cutoff(self) -> int:
        return self.connection.execute(
            "SELECT COALESCE(MAX(source_order), -1) FROM sources"
        ).fetchone()[0]

    def _namespace(self, records: Iterable[object]) -> None:
        for record in records:
            if record.namespace != self.namespace:
                raise ValueError("Records cannot cross namespaces")

    def add_sources(self, sources: list[Source]) -> list[Source]:
        self._namespace(sources)
        result: list[Source] = []
        new: list[Source] = []
        seen: dict[tuple[str, str], Source] = {}
        for source in sources:
            identity = (source.conversation_id or "", source.message_id)
            row = self.connection.execute(
                "SELECT payload FROM sources WHERE conversation_id=? AND message_id=?", identity
            ).fetchone()
            old = Source.model_validate_json(row[0]) if row else seen.get(identity)
            if old is not None:
                fields = {"id", "created_at"}
                if source.model_dump(exclude=fields) != old.model_dump(exclude=fields):
                    raise ValueError("Replayed message differs from its stored source")
                result.append(old)
            else:
                seen[identity] = source
                new.append(source)
                result.append(source)
        if not new:
            return result
        visible = self.view()
        combined_sources = visible.sources | {item.id: item for item in new}
        for source in new:
            for ref in source.generation_refs:
                self._validate_ref(ref, combined_sources, visible.versions)
        from .maintenance import validate_graph
        validate_graph(combined_sources, visible.versions, visible.dependencies, self.max_claim_depth)
        with self.connection:
            seq = self.connection.execute("INSERT INTO commits DEFAULT VALUES").lastrowid
            for source in new:
                self.connection.execute(
                    "INSERT INTO sources VALUES (?, ?, ?, ?, ?, ?)",
                    (source.id, source.message_id, source.conversation_id or "",
                     source.source_order, canonical_json(source), seq),
                )
                self._index_refs("source", source.id, source.generation_refs)
        self._views.clear()
        return result

    @staticmethod
    def _validate_ref(
        ref: PremiseRef, sources: dict[str, Source], versions: dict[str, MemoryVersion],
    ) -> None:
        if ref.type == "SOURCE":
            source = sources.get(ref.id)
            if source is None or ref.span is None or ref.span.source_id != ref.id:
                raise ValueError("SOURCE requires a visible source and matching span")
            spans = [ref.span, *ref.context_refs]
        else:
            if ref.id not in versions:
                raise ValueError(f"Missing memory premise: {ref.id}")
            if ref.type == "HISTORICAL" and ref.at_time is None:
                raise ValueError("HISTORICAL requires its evaluation time")
            spans = ref.context_refs
        for span in spans:
            source = sources.get(span.source_id)
            if source is None or not (0 <= span.start < span.end <= len(source.content)):
                raise ValueError("Evidence span must address nonempty original source text")

    def _index_refs(self, owner_type: str, owner_id: str, refs: Iterable[PremiseRef]) -> None:
        rows = [(owner_type, owner_id, kind, ref_id) for ref in refs for kind, ref_id in reference_ids(ref)]
        self.connection.executemany("INSERT OR IGNORE INTO refs VALUES (?, ?, ?, ?)", rows)

    def commit(
        self, versions: list[MemoryVersion] | None = None,
        dependencies: list[DependencyLink] | None = None,
        operations: list[ControlOperation] | None = None, *,
        expected_seq: int | None = None, source_cutoff: int | None = None,
        progress: dict[str, dict] | None = None,
    ) -> int:
        """这里只校验可确定的边界；自然语言充分性已由写入器验证。"""
        from .maintenance import validate_graph

        if expected_seq is not None and expected_seq != self.current_seq:
            raise ValueError("The memory changed after the proposal snapshot")
        cutoff = self.source_cutoff if source_cutoff is None else source_cutoff
        current = self.view(source_cutoff=cutoff)
        versions, dependencies, operations = list(versions or []), list(dependencies or []), list(operations or [])
        self._namespace([*versions, *dependencies, *operations])
        # 重试可重复提交同一不可变对象，但绝不能改写已有标识的内容。
        versions = self._new_records("versions", versions)
        all_versions = current.versions | {version.id: version for version in versions}
        known_dependencies = {
            row["signature"] for row in self.connection.execute("SELECT signature FROM dependencies")
        }
        unique_dependencies: list[DependencyLink] = []
        for dep in dependencies:
            if dep.target_version_id not in all_versions:
                raise ValueError(f"Missing dependency target: {dep.target_version_id}")
            if not 1 <= len(dep.premise_refs) <= self.max_premises_per_dependency:
                raise ValueError("Dependency premise count exceeds its configured bound")
            for ref in dep.premise_refs:
                self._validate_ref(ref, current.sources, all_versions)
            signature = dependency_signature(dep)
            if signature not in known_dependencies:
                known_dependencies.add(signature)
                unique_dependencies.append(dep)
        dependencies = self._new_records("dependencies", unique_dependencies)
        all_dependencies = current.dependencies | {dep.id: dep for dep in dependencies}
        for version in versions:
            revision = version.revision
            if revision is None:
                continue
            previous = all_versions.get(revision.previous_version_id)
            if previous is None or previous.memory_key != version.memory_key:
                raise ValueError("A revision must reference the same visible memory family")
            if not revision.evidence_refs:
                raise ValueError("A revision requires explicit evidence")
            for ref in revision.evidence_refs:
                self._validate_ref(ref, current.sources, all_versions)
            for dep in all_dependencies.values():
                if dep.target_version_id == version.id and any(
                    ref.type == "CURRENT" and ref.id == previous.id for ref in dep.premise_refs
                ):
                    raise ValueError("A new version cannot depend on CURRENT of its replaced version")
            operations.append(ControlOperation(
                id=f"revision:{version.id}", namespace=self.namespace,
                kind="correct" if revision.reason == "correction" else "supersede",
                scope="version", target_id=previous.id,
                effective_time=revision.effective_time,
                evidence_refs=revision.evidence_refs, replacement_id=version.id,
                source_cutoff=cutoff, reason=revision.reason,
            ))
        for dep in dependencies:
            if dep.effect == "INVALIDATE":
                operations.append(ControlOperation(
                    id=f"invalidate:{dep.id}", namespace=self.namespace,
                    kind="close", scope="version", target_id=dep.target_version_id,
                    effective_time=dep.effective_time, evidence_refs=dep.premise_refs,
                    source_cutoff=cutoff, reason=f"dependency:{dep.id}",
                ))
        # 先移除已提交操作，重放旧删除不能重新扩大到后来主动提供的新来源。
        operations = self._new_records("operations", operations)
        # 在删除时固定直接来源范围。历史快照可能尚未抽取这些事实，不能靠它的旧图猜范围。
        deleted_spans = []
        for op in operations:
            if op.kind != "delete" or op.scope not in {"version", "key"}:
                continue
            target_ids = {v.id for v in all_versions.values()
                          if (op.scope == "version" and v.id == op.target_id)
                          or (op.scope == "key" and v.memory_key == op.target_id)}
            refs = [ref for dep in all_dependencies.values()
                    if dep.effect == "SUPPORT" and dep.target_version_id in target_ids
                    for ref in dep.premise_refs if ref.type == "SOURCE"]
            refs.extend(ref for ref in op.evidence_refs if ref.type == "SOURCE")
            for ref in refs:
                span = ref.span
                deleted_spans.append(ControlOperation(
                    id=f"delete-span:{op.id}:{span.source_id}:{span.start}:{span.end}",
                    namespace=self.namespace, kind="delete", scope="span", target_id=span.source_id,
                    span=span, source_cutoff=cutoff, reason="Explicit deletion source scope", topic=op.topic,
                ))
        operations.extend(deleted_spans)
        operations = self._new_records("operations", operations)
        operation_ids = {op.id for op in current.operations} | {op.id for op in operations}
        for op in operations:
            if op.source_cutoff is not None and op.source_cutoff > cutoff:
                raise ValueError("Control operation cannot use future sources")
            targets = {
                "version": all_versions, "key": {v.memory_key for v in all_versions.values()},
                "source": current.sources, "span": current.sources,
                "dependency": all_dependencies, "operation": operation_ids,
            }
            if op.target_id not in targets[op.scope]:
                raise ValueError(f"Missing control target: {op.scope}/{op.target_id}")
            if op.scope == "span" and (op.span is None or op.span.source_id != op.target_id):
                raise ValueError("A span control must identify the same target source")
            if op.replacement_id is not None:
                replacement = all_versions.get(op.replacement_id)
                if replacement is None:
                    raise ValueError("Missing replacement version")
                if op.scope == "version" and replacement.memory_key != all_versions[op.target_id].memory_key:
                    raise ValueError("A replacement must preserve the target memory identity")
            for ref in op.evidence_refs:
                self._validate_ref(ref, current.sources, all_versions)
            if op.span is not None:
                self._validate_ref(PremiseRef(type="SOURCE", id=op.span.source_id, span=op.span), current.sources, all_versions)
        validate_graph(current.sources, all_versions, all_dependencies, self.max_claim_depth)
        # 用提交后的控制状态校验前提，防止新值经中间结论偷偷依赖被自己关闭的旧值。
        if dependencies or any(op.kind == "delete" for op in operations):
            from .maintenance import _Evaluation
            proposed_sequence = self.current_seq + 1
            proposed = MemoryView(
                namespace=self.namespace, sequence=proposed_sequence, source_cutoff=cutoff,
                sources=current.sources, versions=all_versions, dependencies=all_dependencies,
                operations=[*current.operations, *operations],
                version_sequences=current.version_sequences | {v.id: proposed_sequence for v in versions},
                dependency_sequences=current.dependency_sequences | {d.id: proposed_sequence for d in dependencies},
                operation_sequences=current.operation_sequences | {op.id: proposed_sequence for op in operations},
            )
            evaluation = _Evaluation(proposed)
            if any(op.kind == "delete" for op in operations):
                baseline = _Evaluation(current)
                deletion_id = sha256(canonical_json(sorted(op.id for op in operations if op.kind == "delete")).encode()).hexdigest()[:20]
                propagated = []
                for source_id in current.sources:
                    old_ranges = baseline.source_ranges(source_id, include_retracted=False)
                    for start, end in evaluation.source_ranges(source_id, include_retracted=False):
                        if any(a <= start and b >= end for a, b in old_ranges):
                            continue
                        propagated.append(ControlOperation(
                            id=f"delete-provenance:{deletion_id}:{source_id}:{start}:{end}",
                            namespace=self.namespace, kind="delete", scope="span", target_id=source_id,
                            span={"source_id": source_id, "start": start, "end": end},
                            source_cutoff=cutoff, reason="Recorded provenance deletion",
                        ))
                operations.extend(self._new_records("operations", propagated))
            for dep in dependencies:
                point = dep.effective_time or evaluation.now
                if not all(evaluation.ref_available(ref, point) for ref in dep.premise_refs):
                    raise ValueError("Dependency premises are not usable in the committed state")
        if not (versions or dependencies or operations or progress):
            return self.current_seq
        with self.connection:
            seq = self.connection.execute("INSERT INTO commits DEFAULT VALUES").lastrowid
            for version in versions:
                self.connection.execute(
                    "INSERT INTO versions VALUES (?, ?, ?, ?, ?)",
                    (version.id, version.memory_key, canonical_json(version), cutoff, seq),
                )
                if version.revision:
                    self._index_refs("version", version.id, version.revision.evidence_refs)
            for dep in dependencies:
                self.connection.execute(
                    "INSERT INTO dependencies VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (dep.id, dep.target_version_id, dep.effect.value, dependency_signature(dep),
                     canonical_json(dep), cutoff, seq),
                )
                self._index_refs("dependency", dep.id, dep.premise_refs)
            for op in operations:
                self.connection.execute(
                    "INSERT INTO operations VALUES (?, ?, ?, ?, ?)",
                    (op.id, op.target_id, canonical_json(op), cutoff, seq),
                )
                self._index_refs("operation", op.id, op.evidence_refs)
            for key, value in (progress or {}).items():
                self.connection.execute(
                    "INSERT OR REPLACE INTO progress VALUES (?, ?)", (key, canonical_json(value))
                )
        self._views.clear()
        return seq

    def _new_records(self, table: str, records: list) -> list:
        unique: dict[str, object] = {}
        for record in records:
            row = self.connection.execute(f"SELECT payload FROM {table} WHERE id=?", (record.id,)).fetchone()
            old = json.loads(row[0]) if row else (
                unique[record.id].model_dump(mode="json") if record.id in unique else None
            )
            value = record.model_dump(mode="json")
            if old is not None:
                # 时间戳不是语义；一次重试可以重新构造相同操作。
                if {k: v for k, v in old.items() if k != "created_at"} != {
                    k: v for k, v in value.items() if k != "created_at"
                }:
                    raise ValueError(f"Immutable record changed: {record.id}")
            else:
                unique[record.id] = record
        return list(unique.values())

    def snapshot(self, snapshot_id: int) -> Snapshot:
        row = self.connection.execute("SELECT * FROM snapshots WHERE id=?", (snapshot_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown snapshot: {snapshot_id}")
        return Snapshot(id=row["id"], namespace=self.namespace,
                        source_cutoff=row["source_cutoff"], sequence=row["commit_seq"],
                        maintenance_incomplete=bool(row["maintenance_incomplete"]), created_at=row["created_at"])

    def _pending_at(self, sequence: int, cutoff: int) -> bool:
        from .maintenance import _Evaluation
        evaluation = _Evaluation(self.view(sequence=sequence, source_cutoff=cutoff))
        if any(evaluation.evaluate(version_id).reason == "maintenance_incomplete"
               for version_id in evaluation.view.versions):
            return True
        for row in self.connection.execute("SELECT payload FROM progress"):
            progress = json.loads(row[0])
            batch_cutoff = progress.get("source_cutoff", cutoff)
            if batch_cutoff <= cutoff and progress.get("technical_failed"):
                return True
            for scope in progress.get("pending_scopes", []):
                if scope.get("source_cutoff", batch_cutoff) <= cutoff:
                    return True
        return False

    def publish(self, source_cutoff: int | None = None) -> Snapshot:
        cutoff = self.source_cutoff if source_cutoff is None else source_cutoff
        if cutoff > self.source_cutoff:
            raise ValueError("Cannot publish unseen sources")
        seq = self.current_seq
        incomplete = self._pending_at(seq, cutoff)
        with self.connection:
            self.connection.execute(
                "INSERT OR IGNORE INTO snapshots(commit_seq, source_cutoff, maintenance_incomplete, created_at) VALUES (?, ?, ?, ?)",
                (seq, cutoff, int(incomplete), utc_now())
            )
        row = self.connection.execute(
            "SELECT id FROM snapshots WHERE commit_seq=? AND source_cutoff=?", (seq, cutoff)
        ).fetchone()
        return self.snapshot(row[0])

    def view(
        self, snapshot: Snapshot | int | None = None, *, source_cutoff: int | None = None,
        sequence: int | None = None,
    ) -> MemoryView:
        if isinstance(snapshot, int):
            snapshot = self.snapshot(snapshot)
        if snapshot is not None:
            if snapshot.namespace != self.namespace:
                raise ValueError("Snapshot namespace mismatch")
            seq, cutoff = snapshot.sequence, snapshot.source_cutoff
        else:
            seq, cutoff = self.current_seq if sequence is None else sequence, self.source_cutoff
        if source_cutoff is not None:
            cutoff = min(cutoff, source_cutoff)
        key = (seq, cutoff, self.current_seq)
        if key in self._views:
            return self._views[key]
        def rows(table: str) -> list:
            return self.connection.execute(
                f"SELECT * FROM {table} WHERE seq <= ? AND source_order <= ? ORDER BY seq, rowid",
                (seq, cutoff),
            ).fetchall()
        source_rows, version_rows, dependency_rows, operation_rows = (
            rows("sources"), rows("versions"), rows("dependencies"), rows("operations")
        )
        # 遗忘权限覆盖历史查询；仅叠加当前删除，不让未来事实、更正或其他控制泄入旧快照。
        future_rows = self.connection.execute(
            "SELECT * FROM operations WHERE seq > ? OR source_order > ? ORDER BY seq, rowid", (seq, cutoff)
        ).fetchall()
        operation_rows.extend(row for row in future_rows if json.loads(row["payload"])["kind"] == "delete")
        view = MemoryView(
            namespace=self.namespace, sequence=seq, source_cutoff=cutoff,
            sources={r["id"]: Source.model_validate_json(r["payload"]) for r in source_rows},
            versions={r["id"]: MemoryVersion.model_validate_json(r["payload"]) for r in version_rows},
            dependencies={r["id"]: DependencyLink.model_validate_json(r["payload"]) for r in dependency_rows},
            operations=[ControlOperation.model_validate_json(r["payload"]) for r in operation_rows],
            version_sequences={r["id"]: r["seq"] for r in version_rows},
            dependency_sequences={r["id"]: r["seq"] for r in dependency_rows},
            operation_sequences={r["id"]: r["seq"] for r in operation_rows},
        )
        self._views[key] = view
        if len(self._views) > 4:
            self._views.popitem(last=False)
        return view

    def get_source(self, source_id: str, snapshot: Snapshot | int | None = None) -> Source:
        return self.view(snapshot).sources[source_id]

    def get_version(self, version_id: str, snapshot: Snapshot | int | None = None) -> MemoryVersion:
        return self.view(snapshot).versions[version_id]

    def get_dependency(self, dependency_id: str, snapshot: Snapshot | int | None = None) -> DependencyLink:
        return self.view(snapshot).dependencies[dependency_id]

    def sources(self, snapshot: Snapshot | int | None = None) -> list[Source]:
        return sorted(self.view(snapshot).sources.values(), key=lambda source: source.source_order)

    def versions(self, snapshot: Snapshot | int | None = None, *, memory_key: str | None = None) -> list[MemoryVersion]:
        return [v for v in self.view(snapshot).versions.values() if memory_key is None or v.memory_key == memory_key]

    def dependencies(self, target_id: str | None = None, snapshot: Snapshot | int | None = None) -> list[DependencyLink]:
        return [d for d in self.view(snapshot).dependencies.values() if target_id is None or d.target_version_id == target_id]

    def operations(self, snapshot: Snapshot | int | None = None) -> list[ControlOperation]:
        return self.view(snapshot).operations

    def get_embedding(self, record_id: str, model: str, text_hash: str):
        import numpy as np
        row = self.connection.execute(
            "SELECT vector, dimensions FROM embeddings WHERE record_id=? AND model=? AND text_hash=?",
            (record_id, model, text_hash),
        ).fetchone()
        return None if row is None else np.frombuffer(row["vector"], dtype=np.float32).copy()

    def get_progress(self, key: str) -> dict | None:
        row = self.connection.execute("SELECT payload FROM progress WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def set_progress(self, key: str, payload: dict) -> None:
        encoded = canonical_json(payload)
        row = self.connection.execute("SELECT payload FROM progress WHERE key=?", (key,)).fetchone()
        if row and row[0] == encoded:
            return
        with self.connection:
            self.connection.execute("INSERT INTO commits DEFAULT VALUES")
            self.connection.execute("INSERT OR REPLACE INTO progress VALUES (?, ?)", (key, encoded))
        self._views.clear()

    def put_embedding(self, record_id: str, model: str, text_hash: str, vector) -> None:
        import numpy as np
        vector = np.asarray(vector, dtype=np.float32)
        if vector.ndim != 1 or not np.isfinite(vector).all():
            raise ValueError("Embedding must be a finite one-dimensional vector")
        norm = np.linalg.norm(vector)
        vector = vector / norm if norm else vector
        with self.connection:
            self.connection.execute(
                "INSERT OR REPLACE INTO embeddings VALUES (?, ?, ?, ?, ?)",
                (record_id, model, text_hash, vector.tobytes(), len(vector)),
            )

    def rebuild_indexes(self) -> None:
        """从不可变记录重建反向引用；状态求值不依赖旧缓存。"""
        view = self.view()
        with self.connection:
            self.connection.execute("DELETE FROM refs")
            self.connection.execute("DELETE FROM state_cache")
            for source in view.sources.values():
                self._index_refs("source", source.id, source.generation_refs)
            for version in view.versions.values():
                if version.revision:
                    self._index_refs("version", version.id, version.revision.evidence_refs)
            for dep in view.dependencies.values():
                self._index_refs("dependency", dep.id, dep.premise_refs)
            for op in view.operations:
                self._index_refs("operation", op.id, op.evidence_refs)
