"""确定性依赖维护：时间、关闭、删除与支持展开使用同一套判断。"""

from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timezone
import json
from typing import TYPE_CHECKING

from .models import (
    ControlOperation, DependencyLink, EvidenceBundle, MaintenanceReport, MemoryVersion,
    PremiseRef, Resolution, Snapshot, Source, SourceSpan, TimePoint,
)
from .persistence import canonical_json

if TYPE_CHECKING:
    from .store import MemoryView, OurMemStore


def compare_time(left: TimePoint | None, right: TimePoint | None) -> int | None:
    """未知日期不补造；有共同时间轴时才比较，偏序不足时返回 None。"""
    if left is None or right is None:
        return None
    sides = {"before": -1, "at": 0, "after": 1}
    if left.date is not None and right.date is not None:
        def normalized(value: str) -> datetime:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
        a, b = normalized(left.date), normalized(right.date)
        if a != b:
            return (a > b) - (a < b)
        # 同一日期没有消息序号的查询，at 指该日期本身，不假设日内相对顺序。
        if left.order is None or right.order is None:
            return (sides[left.side] > sides[right.side]) - (sides[left.side] < sides[right.side])
    if left.order is not None and right.order is not None:
        a = (left.order, left.offset, sides[left.side])
        b = (right.order, right.offset, sides[right.side])
        return (a > b) - (a < b)
    return None


def validate_graph(
    sources: dict[str, Source], versions: dict[str, MemoryVersion],
    dependencies: dict[str, DependencyLink], max_depth: int,
) -> dict[str, int]:
    """检查全图而非仅新后缀；增加一条旧节点的路径也可能加深其所有后继。"""
    edges: dict[str, set[str]] = {record_id: set() for record_id in sources | versions}
    support_edges: dict[str, list[list[str]]] = defaultdict(list)
    for source in sources.values():
        for ref in source.generation_refs:
            edges[source.id].add(ref.id)
    for version in versions.values():
        if version.revision:
            for ref in version.revision.evidence_refs:
                edges[version.id].add(ref.id)
    for dependency in dependencies.values():
        if dependency.target_version_id not in versions:
            raise ValueError("Unknown dependency target")
        refs = []
        for ref in dependency.premise_refs:
            edges[dependency.target_version_id].add(ref.id)
            refs.append(ref.id)
        if dependency.effect == "SUPPORT":
            support_edges[dependency.target_version_id].append(refs)
    visiting, done = set(), set()
    def check_cycle(node: str) -> None:
        if node in visiting:
            raise ValueError("Dependency, source provenance or revision creates a cycle")
        if node in done:
            return
        if node not in edges:
            raise ValueError(f"Unknown graph reference: {node}")
        visiting.add(node)
        for premise in edges[node]:
            check_cycle(premise)
        visiting.remove(node)
        done.add(node)
    for node in edges:
        check_cycle(node)
    depths: dict[str, int] = {}
    def depth(node: str) -> int:
        if node in depths:
            return depths[node]
        if node in sources:
            refs = sources[node].generation_refs
            value = max((depth(ref.id) for ref in refs), default=-1)
        else:
            paths = support_edges[node]
            value = max((1 + max(depth(ref) for ref in path) for path in paths), default=0)
        depths[node] = value
        return value
    for version_id in versions:
        if depth(version_id) > max_depth:
            raise ValueError(f"Complete claim path exceeds depth {max_depth}: {version_id}")
    return {version_id: depths[version_id] for version_id in versions}


class _Evaluation:
    """单次求值共享视图及备忘录，不把时间相关结果当作永久状态。"""

    def __init__(self, view: MemoryView, query_time: TimePoint | str | None = None,
                 source_span_cutoff: SourceSpan | None = None) -> None:
        self.view = view
        self.span_cutoff = source_span_cutoff
        self.now = self._point(query_time)
        revoked = {op.target_id for op in view.operations if op.kind == "revoke_control"}
        self.operations = [op for op in view.operations if op.id not in revoked and op.kind != "revoke_control"]
        families = defaultdict(list)
        for version in view.versions.values():
            families[version.memory_key].append(version)
        self._version_ops = defaultdict(list)
        self._disabled_dependencies, self._deleted_dependencies = set(), set()
        for op in self.operations:
            if op.scope == "version" and op.target_id in view.versions:
                self._version_ops[op.target_id].append(op)
            elif op.scope == "key":
                for version in families[op.target_id]:
                    if self._applies(op, version):
                        self._version_ops[version.id].append(op)
            elif op.scope == "dependency" and op.kind in {"retract", "delete", "close"}:
                self._disabled_dependencies.add(op.target_id)
                if op.kind == "delete":
                    self._deleted_dependencies.add(op.target_id)
        self.supports: dict[str, list[DependencyLink]] = defaultdict(list)
        for dep in view.dependencies.values():
            if dep.effect == "SUPPORT":
                self.supports[dep.target_version_id].append(dep)
        self.memo: dict[tuple[str, str], Resolution] = {}
        self.source_memo: dict[tuple[str, int, int, str, bool], bool] = {}
        self.content_memo: dict[str, bool] = {}
        self._deleted_ranges: dict[str, list[tuple[int, int]]] | None = None
        self._retracted_ranges: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for op in self.operations:
            if op.kind == "retract" and op.scope in {"source", "span"} and op.target_id in view.sources:
                interval = (op.span.start, op.span.end) if op.span else (0, len(view.sources[op.target_id].content))
                self._retracted_ranges[op.target_id].append(interval)

    def _point(self, value: TimePoint | str | None) -> TimePoint:
        if isinstance(value, str):
            return TimePoint(date=value)
        if value is not None:
            return value
        sources = list(self.view.sources.values())
        if not sources:
            return TimePoint()
        last = max(sources, key=lambda item: item.source_order)
        date = last.mention_time
        if date:
            try:
                datetime.fromisoformat(date.replace("Z", "+00:00"))
            except ValueError:
                date = None
        return TimePoint(date=date, order=last.source_order,
                         offset=self.span_cutoff.end if self.span_cutoff else len(last.content), side="after")

    def _applies(self, op: ControlOperation, version: MemoryVersion) -> bool:
        if op.scope == "version":
            return op.target_id == version.id
        return (op.scope == "key" and op.target_id == version.memory_key
                and self.view.version_sequences[version.id] <= self.view.operation_sequences[op.id])

    def _effective(self, op: ControlOperation, point: TimePoint) -> bool:
        if op.kind in {"correct", "delete", "retract"}:
            return True
        effective = op.effective_time
        if effective is None and op.source_cutoff is not None and op.source_cutoff >= 0:
            effective = TimePoint(order=op.source_cutoff)
        comparison = compare_time(point, effective)
        return comparison is None or comparison >= 0

    def control_evidence_valid(self, op: ControlOperation) -> bool:
        point = op.effective_time
        if point is None and op.source_cutoff is not None and op.source_cutoff >= 0:
            point = TimePoint(order=op.source_cutoff, side="after")
        return all(self.ref_available(ref, point or self.now) for ref in op.evidence_refs)

    def version_deleted(self, version_id: str) -> bool:
        return any(op.kind == "delete" for op in self._version_ops[version_id])

    def _deletion_closure(self) -> dict[str, list[tuple[int, int]]]:
        """只沿显式出处传播遮蔽，不靠相似度猜测哪些原文可能包含被删值。"""
        ranges = {source_id: set() for source_id in self.view.sources}
        deletes = [op for op in self.operations if op.kind == "delete"]
        if not deletes:
            return {source_id: [] for source_id in ranges}
        families = defaultdict(list)
        for version in self.view.versions.values():
            families[version.memory_key].append(version)
        deleted_versions, deleted_dependencies = set(), set()
        for op in deletes:
            if op.scope in {"source", "span"} and op.target_id in ranges:
                source = self.view.sources[op.target_id]
                ranges[source.id].add((op.span.start, op.span.end) if op.span else (0, len(source.content)))
            if op.scope == "dependency":
                deleted_dependencies.add(op.target_id)
            targets = families[op.target_id] if op.scope == "key" else ([self.view.versions[op.target_id]] if op.scope == "version" and op.target_id in self.view.versions else [])
            refs = list(op.evidence_refs)
            for version in targets:
                if not self._applies(op, version):
                    continue
                deleted_versions.add(version.id)
                for dep in self.supports[version.id]:
                    if self.view.dependency_sequences[dep.id] <= self.view.operation_sequences[op.id]:
                        refs.extend(dep.premise_refs)
            for ref in refs:
                if ref.type == "SOURCE" and ref.id in ranges:
                    ranges[ref.id].add((ref.span.start, ref.span.end))
        def hidden(span):
            return any(span.start < end and start < span.end for start, end in ranges.get(span.source_id, ()))
        def ref_deleted(ref):
            if ref.type != "SOURCE":
                return ref.id in deleted_versions
            return hidden(ref.span) or any(hidden(span) for span in ref.context_refs)
        all_refs = [ref for dep in self.view.dependencies.values() for ref in dep.premise_refs]
        all_refs.extend(ref for source in self.view.sources.values() for ref in source.generation_refs)
        all_refs.extend(ref for version in self.view.versions.values() if version.revision for ref in version.revision.evidence_refs)
        changed = True
        while changed:
            changed = False
            for ref in all_refs:
                if ref.type == "SOURCE" and any(hidden(span) for span in ref.context_refs):
                    interval = (ref.span.start, ref.span.end)
                    if interval not in ranges[ref.id]:
                        ranges[ref.id].add(interval)
                        changed = True
            for version_id, paths in self.supports.items():
                if version_id not in deleted_versions and paths and all(
                    dep.id in deleted_dependencies or any(ref_deleted(ref) for ref in dep.premise_refs) for dep in paths
                ):
                    deleted_versions.add(version_id)
                    changed = True
            for source in self.view.sources.values():
                if source.generation_refs and any(ref_deleted(ref) for ref in source.generation_refs):
                    # generation_refs描述整条自有输出；有逐句关联时，上面的span传播只遮对应片段。
                    interval = (0, len(source.content))
                    if interval not in ranges[source.id]:
                        ranges[source.id].add(interval)
                        changed = True
        return {source_id: list(spans) for source_id, spans in ranges.items()}

    def source_ranges(self, source_id: str, *, include_retracted: bool = True) -> list[tuple[int, int]]:
        """原文、索引和支持校验共享遮蔽；普通撤回不删除陈述历史。"""
        if self._deleted_ranges is None:
            self._deleted_ranges = self._deletion_closure()
        source = self.view.sources[source_id]
        hidden = list(self._deleted_ranges[source_id])
        if include_retracted:
            hidden.extend(self._retracted_ranges[source_id])
        if self.span_cutoff and self.span_cutoff.source_id == source_id:
            hidden.append((self.span_cutoff.end, len(source.content)))
        intervals: list[tuple[int, int]] = []
        for start, end in sorted(hidden):
            if start >= end:
                continue
            if intervals and start <= intervals[-1][1]:
                intervals[-1] = (intervals[-1][0], max(end, intervals[-1][1]))
            else:
                intervals.append((start, end))
        return intervals

    def source_allowed(self, span: SourceSpan, point: TimePoint, *, follow_generation: bool = True) -> bool:
        key = (span.source_id, span.start, span.end, canonical_json(point), follow_generation)
        if key in self.source_memo:
            return self.source_memo[key]
        source = self.view.sources.get(span.source_id)
        if source is None:
            return False
        result = not any(span.start < end and start < span.end for start, end in self.source_ranges(source.id))
        if result and follow_generation and source.generation_refs:
            result = all(self.ref_available(ref, point) for ref in source.generation_refs)
        self.source_memo[key] = result
        return result

    def content_visible(self, version_id: str) -> bool:
        """失去删除证据的派生文字也不可回传；历史过期本身不等于删除。"""
        if version_id in self.content_memo:
            return self.content_memo[version_id]
        if self.version_deleted(version_id):
            return False
        def ref_visible(ref: PremiseRef) -> bool:
            if ref.type != "SOURCE":
                return self.content_visible(ref.id)
            spans = [ref.span, *ref.context_refs]
            for span in spans:
                if any(span.start < end and start < span.end
                       for start, end in self.source_ranges(span.source_id, include_retracted=False)):
                    return False
            source = self.view.sources[ref.id]
            return all(ref_visible(original) for original in source.generation_refs)
        paths = self.supports[version_id]
        allowed = not paths or any(
            all(ref_visible(ref) for ref in dep.premise_refs)
            and dep.id not in self._deleted_dependencies
            for dep in paths
        )
        self.content_memo[version_id] = allowed
        return allowed

    def ref_available(self, ref: PremiseRef, point: TimePoint) -> bool:
        if ref.type == "SOURCE":
            return self.source_allowed(ref.span, point) and all(
                # 消歧要求这段话仍可引用，不要求助手当时说的结论现在仍为真。
                self.source_allowed(span, point, follow_generation=False) for span in ref.context_refs
            )
        return self.evaluate(ref.id, ref.at_time if ref.type == "HISTORICAL" else point).usable

    def _dependency_allowed(self, dependency: DependencyLink) -> bool:
        return dependency.id not in self._disabled_dependencies

    def evaluate(self, version_id: str, point: TimePoint | None = None) -> Resolution:
        point = point or self.now
        key = (version_id, canonical_json(point))
        if key in self.memo:
            return self.memo[key]
        version = self.view.versions.get(version_id)
        if version is None:
            return Resolution(version_id=version_id, usable=False, status="stale", reason="not_visible")
        def result(usable: bool, status: str, reason: str = "", support_ids: list[str] | None = None) -> Resolution:
            resolution = Resolution(version_id=version_id, usable=usable, status=status,
                                    reason=reason, support_ids=support_ids or [])
            self.memo[key] = resolution
            return resolution
        if self.version_deleted(version_id):
            return result(False, "deleted", "deleted")
        if not self.content_visible(version_id):
            return result(False, "deleted", "deleted_evidence")
        scope = version.valid_time
        if scope.kind == "state":
            start, end = compare_time(point, scope.start), compare_time(point, scope.end)
            if start is not None and start < 0:
                return result(False, "stale", "not_yet_applicable")
            if end is not None and end >= 0:
                return result(False, "stale", "expired")
        matching = [op for op in self._version_ops[version_id] if self._effective(op, point)]
        for op in matching:
            if op.kind in {"close", "supersede", "correct", "retract"}:
                # 关闭是已提交决定，不跟随触发前提反复开关；复核须追加 revoke_control。
                return result(False, "superseded", "corrected" if op.kind == "correct" else "closed")
        flags = {}
        for op in matching:
            if op.kind in {"pending", "resolve_pending"}:
                flags["pending"] = op.kind == "pending"
            if op.kind in {"conflict", "resolve_conflict"}:
                flags["conflict"] = op.kind == "conflict"
        if flags.get("conflict"):
            return result(False, "stale", "conflict")
        if flags.get("pending"):
            return result(False, "stale", "maintenance_incomplete")
        support_ids = [dep.id for dep in self.supports[version_id]
                       if self._dependency_allowed(dep)
                       and all(self.ref_available(ref, point) for ref in dep.premise_refs)]
        if not support_ids:
            return result(False, "stale", "unsupported")
        return result(True, "active", support_ids=support_ids)


class MaintenanceEngine:
    def __init__(self, store: OurMemStore) -> None:
        self.store = store
        self._cached_key = None
        self._cached_evaluation = None

    def _evaluation(self, snapshot=None, query_time=None, source_cutoff=None, source_span_cutoff=None) -> _Evaluation:
        view = self.store.view(snapshot, source_cutoff=source_cutoff)
        key = (min(view.sequence, self.store.data_seq), view.source_cutoff, self.store.data_seq,
               canonical_json(query_time), canonical_json(source_span_cutoff))
        if key != self._cached_key:
            self._cached_evaluation = _Evaluation(view, query_time, source_span_cutoff)
            self._cached_key = key
        return self._cached_evaluation

    def evaluate(self, version_id: str, query_time: TimePoint | str | None = None,
                 snapshot: Snapshot | int | None = None, source_cutoff: int | None = None,
                 source_span_cutoff: SourceSpan | None = None) -> Resolution:
        return self._evaluation(snapshot, query_time, source_cutoff, source_span_cutoff).evaluate(version_id)

    def evaluate_all(self, snapshot=None, query_time=None, source_cutoff=None) -> dict[str, Resolution]:
        evaluation = self._evaluation(snapshot, query_time, source_cutoff)
        return {version_id: evaluation.evaluate(version_id) for version_id in evaluation.view.versions}

    def reading_references(self, evaluation: _Evaluation) -> dict[str, PremiseRef]:
        """历史记录要在其实际依据成立时可用；纠错和删除不能被旧时间绕过。"""
        points = {}
        def origin(version_id):
            if version_id in points:
                return points[version_id]
            version = evaluation.view.versions[version_id]
            positions = []
            for dep in evaluation.supports[version_id]:
                for ref in dep.premise_refs:
                    if ref.type == "SOURCE":
                        source = evaluation.view.sources[ref.id]
                        positions.append(TimePoint(date=source.mention_time, order=source.source_order, offset=ref.span.end))
                    else:
                        positions.extend([ref.at_time] if ref.type == "HISTORICAL" else origin(ref.id))
            if version.valid_time.start is not None:
                positions.append(version.valid_time.start)
            # 检查已知依据边界，不枚举前提组合；晚到的另一条或路径不能抹去较早历史。
            points[version_id] = list({canonical_json(p): p for p in positions}.values())
            return points[version_id]
        result = {}
        for version in evaluation.view.versions.values():
            if not evaluation.content_visible(version.id):
                continue
            current = evaluation.evaluate(version.id)
            if current.usable:
                result[version.id] = PremiseRef(type="CURRENT", id=version.id)
            elif current.reason not in {"deleted", "deleted_evidence", "corrected", "not_yet_applicable", "conflict"}:
                for point in sorted(origin(version.id), key=lambda p: (p.order if p.order is not None else -1, p.offset), reverse=True):
                    if compare_time(point, evaluation.now) != 1 and evaluation.evaluate(version.id, point).usable:
                        result[version.id] = PremiseRef(type="HISTORICAL", id=version.id, at_time=point)
                        break
        return result

    def depth(self, version_id: str, snapshot=None, source_cutoff=None) -> int:
        view = self.store.view(snapshot, source_cutoff=source_cutoff)
        return validate_graph(view.sources, view.versions, view.dependencies, self.store.max_claim_depth)[version_id]

    def affected(self, changed_ids: list[str], snapshot=None, source_cutoff=None) -> list[str]:
        """完整传递闭包；失效依据、消歧原文与修订依据同样进入反向目录。"""
        view = self.store.view(snapshot, source_cutoff=source_cutoff)
        families = defaultdict(list)
        for version in view.versions.values():
            families[version.memory_key].append(version.id)
        operations = {op.id: op for op in view.operations}
        queue = deque(changed_ids)
        seen: set[str] = set()
        while queue:
            node = queue.popleft()
            if node in seen:
                continue
            seen.add(node)
            following = set(families.get(node, ()))
            if node in view.dependencies:
                following.add(view.dependencies[node].target_version_id)
            for owner_type, owner_id in self.store.connection.execute(
                    "SELECT owner_type, owner_id FROM refs WHERE ref_id=?", (node,)):
                if owner_type == "dependency" and owner_id in view.dependencies:
                    following.add(view.dependencies[owner_id].target_version_id)
                elif owner_type == "version" and owner_id in view.versions:
                    following.add(owner_id)
                elif owner_type == "source" and owner_id in view.sources:
                    following.add(owner_id)
                elif owner_type == "operation" and owner_id in operations:
                    op = operations[owner_id]
                    if op.scope == "version":
                        following.add(op.target_id)
                    elif op.scope == "key":
                        following.update(families[op.target_id])
            queue.extend(sorted(following - seen))
        # 直接比较拓扑深度，保证原子依赖先于上层目标，不截断传播层数。
        dependencies = defaultdict(set)
        for dep in view.dependencies.values():
            dependencies[dep.target_version_id].update(ref.id for ref in dep.premise_refs if ref.type != "SOURCE")
        ranks = {}
        def rank(node: str) -> int:
            if node not in ranks:
                ranks[node] = 1 + max((rank(ref) for ref in dependencies[node]), default=-1)
            return ranks[node]
        return sorted((node for node in seen if node in view.versions), key=lambda node: (rank(node), node))

    def recompute(self, changed_ids: list[str], snapshot=None, query_time=None,
                  source_cutoff=None) -> MaintenanceReport:
        evaluation = self._evaluation(snapshot, query_time, source_cutoff)
        affected = self.affected(changed_ids, snapshot, source_cutoff)
        results = [evaluation.evaluate(version_id) for version_id in affected]
        # 缓存仅便于检查；历史查询始终按自己的时间和快照重算。
        if snapshot is None:
            with self.store.connection:
                for result in results:
                    self.store.connection.execute(
                        "INSERT OR REPLACE INTO state_cache VALUES (?, ?, ?)",
                        (result.version_id, evaluation.view.sequence, canonical_json(result)),
                    )
        pending = []
        for result in results:
            if result.status == "deleted":
                continue
            version = evaluation.view.versions[result.version_id]
            has_replacement = any(
                op.replacement_id is not None for op in evaluation._version_ops[result.version_id]
            )
            control_review = any(
                op.kind in {"close", "supersede", "correct"} and not evaluation.control_evidence_valid(op)
                for op in evaluation._version_ops[result.version_id]
            )
            if (control_review or result.reason in {"unsupported", "maintenance_incomplete"}
                    or result.reason == "closed" and not has_replacement):
                decisions = [op for op in evaluation._version_ops[result.version_id]
                             if op.kind in {"pending", "resolve_pending"}]
                if control_review or not (decisions and decisions[-1].kind == "resolve_pending" and decisions[-1].reason == "UNKNOWN"):
                    pending.append(result.version_id)
        return MaintenanceReport(changed_ids=affected, pending_ids=pending,
                                 results=[item.model_dump(mode="json") for item in results], incomplete=bool(pending))

    def pending(self, snapshot=None, source_cutoff=None) -> list[str]:
        evaluation = self._evaluation(snapshot, source_cutoff=source_cutoff)
        return [version_id for version_id in evaluation.view.versions
                if evaluation.evaluate(version_id).reason == "maintenance_incomplete"]

    def content_visible(self, version_id: str, snapshot=None, source_cutoff=None) -> bool:
        return self._evaluation(snapshot, source_cutoff=source_cutoff).content_visible(version_id)

    def source_text(self, source_id: str, snapshot=None, source_cutoff=None,
                    source_span_cutoff=None, *, include_retracted: bool = False) -> str:
        evaluation = self._evaluation(snapshot, source_cutoff=source_cutoff, source_span_cutoff=source_span_cutoff)
        content = evaluation.view.sources[source_id].content
        chars = list(content)
        for start, end in evaluation.source_ranges(source_id, include_retracted=include_retracted):
            # 保持字符坐标不变，后续引用仍指向完整原文而非重新编号的摘要。
            chars[start:end] = [" "] * (end - start)
        return "".join(chars)

    def visible_spans(self, source_id: str, start: int = 0, end: int | None = None,
                      snapshot=None, source_cutoff=None, source_span_cutoff=None) -> list[SourceSpan]:
        """将请求片段减去删除空洞，不能把安全展示文本误当成跨洞的原文引用。"""
        evaluation = self._evaluation(snapshot, source_cutoff=source_cutoff, source_span_cutoff=source_span_cutoff)
        length = len(evaluation.view.sources[source_id].content)
        end = length if end is None else end
        if not 0 <= start <= end <= length:
            raise ValueError("Requested source range is outside the original message")
        spans, cursor = [], start
        for left, right in evaluation.source_ranges(source_id, include_retracted=False):
            if right <= cursor or left >= end:
                continue
            if left > cursor:
                spans.append(SourceSpan(source_id=source_id, start=cursor, end=min(left, end)))
            cursor = max(cursor, min(right, end))
        if cursor < end:
            spans.append(SourceSpan(source_id=source_id, start=cursor, end=end))
        return spans

    def source_context(self, source_id: str, snapshot=None, query_time=None,
                       source_cutoff=None, source_span_cutoff=None) -> dict:
        """原文可以回看，但必须同时显示已知更新、纠错和控制状态。"""
        evaluation = self._evaluation(snapshot, query_time, source_cutoff, source_span_cutoff)
        affected = self.affected([source_id], snapshot, source_cutoff)
        families = {evaluation.view.versions[item].memory_key for item in affected}
        versions = [v for v in evaluation.view.versions.values()
                    if v.memory_key in families and evaluation.content_visible(v.id)]
        source = evaluation.view.sources[source_id]
        controls = []
        for op in evaluation.operations:
            if op.kind == "delete":
                if (op.scope in {"source", "span"} and op.target_id == source_id
                        or any(evaluation._applies(op, evaluation.view.versions[item]) for item in affected)):
                    controls.append({"id": op.id, "kind": "delete", "topic": op.topic})
                continue
            if any(evaluation._applies(op, version) for version in versions):
                controls.append({"kind": op.kind, "target_id": op.target_id,
                                 "effective_time": op.effective_time.model_dump() if op.effective_time else None,
                                 "replacement_id": op.replacement_id,
                                 "evidence_valid": evaluation.control_evidence_valid(op)})
        return {"source_id": source.id, "speaker": source.speaker, "role": source.role,
                "mention_time": source.mention_time, "source_order": source.source_order,
                "content": self.source_text(source_id, snapshot, source_cutoff, source_span_cutoff),
                "versions": [{"id": v.id, "memory_key": v.memory_key, "content": v.content,
                              "valid_time": v.valid_time.model_dump(mode="json"),
                              "resolution": evaluation.evaluate(v.id).model_dump(mode="json")}
                             for v in versions], "controls": controls}

    def evidence(self, version_id: str, query_time: TimePoint | str | None = None,
                 snapshot: Snapshot | int | None = None, source_cutoff: int | None = None,
                 max_tokens: int | None = None, token_counter=None, bundle_cost=None) -> EvidenceBundle:
        evaluation = self._evaluation(snapshot, query_time, source_cutoff)
        path_cache: dict[tuple[str, str], EvidenceBundle] = {}
        if token_counter is None:
            from .tokenization import count_tokens
            token_counter = count_tokens
        def source_evidence(ref: PremiseRef, point: TimePoint) -> EvidenceBundle:
            if not evaluation.ref_available(ref, point):
                return EvidenceBundle(complete=False, reason="source_unavailable")
            spans = [ref.span, *ref.context_refs]
            texts = []
            for span in spans:
                source = evaluation.view.sources[span.source_id]
                texts.append(f"SOURCE {source.id}[{span.start}:{span.end}] ({source.speaker}, {source.mention_time or 'date unknown'}): "
                             + source.content[span.start:span.end])
            refs = [ref]
            version_ids = []
            nodes, links = [], []
            source = evaluation.view.sources[ref.id]
            for original in source.generation_refs:
                child = expand_ref(original, point)
                if not child.complete:
                    return child
                texts.append(child.text)
                refs.extend(child.refs)
                version_ids.extend(child.version_ids)
                nodes.extend(child.nodes)
                links.extend(child.links)
            return EvidenceBundle(text="\n".join(texts), refs=refs, version_ids=version_ids, nodes=nodes, links=links)
        def expand_ref(ref: PremiseRef, point: TimePoint) -> EvidenceBundle:
            if ref.type == "SOURCE":
                return source_evidence(ref, point)
            child = expand(ref.id, ref.at_time if ref.type == "HISTORICAL" else point)
            return child.model_copy(update={"refs": [ref, *child.refs]})
        def expand(target_id: str, point: TimePoint) -> EvidenceBundle:
            key = (target_id, canonical_json(point))
            if key in path_cache:
                return path_cache[key]
            result = evaluation.evaluate(target_id, point)
            if not result.usable:
                return EvidenceBundle(complete=False, reason=result.reason)
            choices = []
            for dep_id in result.support_ids:
                dep = evaluation.view.dependencies[dep_id]
                parts = [expand_ref(ref, point) for ref in dep.premise_refs]
                if not all(part.complete for part in parts):
                    continue
                target = evaluation.view.versions[target_id]
                text = f"MEMORY {target.id}: {target.content}\nTIME {canonical_json(target.valid_time)}\n"
                text += f"SUPPORT {dep.id}: " + ", ".join(ref.id for ref in dep.premise_refs) + "\n"
                text += "\n".join(dict.fromkeys(part.text for part in parts))
                refs = [ref for part in parts for ref in part.refs]
                version_ids = [target_id, *(node for part in parts for node in part.version_ids)]
                nodes = [{"version_id": target_id, "at_time": point.model_dump(mode="json")},
                         *(node for part in parts for node in part.nodes)]
                links = [{"dependency_id": dep.id, "at_time": point.model_dump(mode="json")},
                         *(link for part in parts for link in part.links)]
                if target.revision:
                    revision_point = target.revision.effective_time or point
                    evidence_parts = [expand_ref(ref, revision_point) for ref in target.revision.evidence_refs]
                    revision_control = next((op for op in evaluation.operations if op.id == f"revision:{target.id}"), None)
                    if all(part.complete for part in evidence_parts) and revision_control is not None:
                        text += "\nREVISION " + canonical_json(target.revision)
                        for evidence in evidence_parts:
                            text += "\n" + evidence.text
                            refs.extend(evidence.refs)
                            version_ids.extend(evidence.version_ids)
                            nodes.extend(evidence.nodes)
                            links.extend(evidence.links)
                    else:
                        # 新值有直接依据时可继续成立，但不把已被推翻的变化解释当作证明。
                        text += "\nRevision interpretation unavailable or requires review."
                choices.append(EvidenceBundle(text=text, refs=list({canonical_json(ref): ref for ref in refs}.values()),
                                              version_ids=list(dict.fromkeys(version_ids)),
                                              nodes=list({canonical_json(n): n for n in nodes}.values()),
                                              links=list({canonical_json(n): n for n in links}.values())))
            bundle = min(choices, key=lambda item: (bundle_cost(item) if bundle_cost else token_counter(item.text), item.text)) if choices else EvidenceBundle(complete=False, reason="no_complete_path")
            path_cache[key] = bundle
            return bundle
        bundle = expand(version_id, evaluation.now)
        if max_tokens is not None:
            if token_counter(bundle.text) > max_tokens:
                return EvidenceBundle(complete=False, reason="evidence_budget_exceeded")
        return bundle
