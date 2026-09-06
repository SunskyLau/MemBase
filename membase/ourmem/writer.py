"""按来源顺序协调内容，提交已验证局部图，并消费有界的修复队列。"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json

from .extractor import ExtractionResult, FactDraft, FactExtractor
from .inducer import DependencyInducer, DependencyProposal, LocalGraphProposal
from .llm import ContextLimitError, OutputLimitError
from .models import (
    ControlOperation, DependencyLink, InputPolicy, MaintenanceReport, MemoryVersion,
    PremiseRef, Revision, Source, SourceSpan, TimePoint,
)
from .reconciler import CoordinationDecision, MemoryReconciler


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=lambda item: item.model_dump(mode="json"))


def _id(prefix: str, *parts: object) -> str:
    return f"{prefix}-{sha256(_json(parts).encode()).hexdigest()[:24]}"


@dataclass
class StagedWrite:
    versions: list[MemoryVersion] = field(default_factory=list)
    dependencies: list[DependencyLink] = field(default_factory=list)
    operations: list[ControlOperation] = field(default_factory=list)
    target_id: str | None = None
    changed_ids: list[str] = field(default_factory=list)


class MemoryWriter:
    def __init__(self, store, retriever, evaluator, llm, config) -> None:
        self.store, self.retriever, self.evaluator = store, retriever, evaluator
        self.llm, self.config = llm, config
        self.extractor = FactExtractor(llm, config)
        self.reconciler = MemoryReconciler(llm, config)
        self.inducer = DependencyInducer(llm, config)

    def process_batch(
        self, sources: list[Source], input_policy: InputPolicy, batch_id: str,
    ) -> MaintenanceReport:
        """重跑同一批次不重置生成预算；每条修改只看当时的来源前缀。"""
        key = f"batch:{batch_id}"
        progress = self.store.get_progress(key) or {
            "generation_calls": 0, "completed_drafts": 0, "fingerprints": [], "report": {},
        }
        progress["source_cutoff"] = max(source.source_order for source in sources)
        progress.setdefault("pending_scopes", [])
        if progress.get("complete"):
            return MaintenanceReport.model_validate(progress["report"])
        report = MaintenanceReport.model_validate(progress.get("report", {}))
        source_map = {source.id: source for source in sources}
        if "extraction" not in progress:
            first = sources[0]
            context = [source for source in self.store.sources()
                       if source.conversation_id == first.conversation_id
                       and source.source_order < first.source_order]
            context = [source.model_copy(update={"content": self.evaluator.source_text(source.id)})
                       for source in context]
            try:
                extraction = self.extractor.extract(sources, context, input_policy)
            except Exception:
                progress["technical_failed"] = True
                self.store.set_progress(key, progress)
                raise
            progress["extraction"] = extraction.model_dump(mode="json")
            self.store.set_progress(key, progress)
        extraction = ExtractionResult.model_validate(progress["extraction"])
        for index, draft in enumerate(extraction.drafts):
            if index < progress["completed_drafts"]:
                continue
            source = source_map[draft.source_id]
            prefix = draft.evidence_refs[0].span
            token = f"{batch_id}:fact:{index}"
            try:
                stored_write = progress.get("writes", {}).get(str(index))
                if stored_write is None:
                    spans = [span for ref in draft.evidence_refs if ref.type == "SOURCE"
                             for span in [ref.span, *ref.context_refs]]
                    if any(self.store.get_source(span.source_id).content[span.start:span.end]
                           != self.evaluator.source_text(span.source_id, include_retracted=True)[span.start:span.end]
                           for span in spans):
                        progress["completed_drafts"] = index + 1
                        report.results.append({"stage": "extract", "source_id": source.id,
                                               "outcome": "UNAVAILABLE", "reason": "cached_evidence_no_longer_available"})
                        progress["report"] = report.model_dump(mode="json")
                        self.store.set_progress(key, progress)
                        continue
                    try:
                        stage, decision = self._coordinate(
                            draft, input_policy, source.source_order, prefix, token,
                        )
                    except (ContextLimitError, OutputLimitError) as error:
                        stage = StagedWrite()
                        decision = CoordinationDecision(identity="NEW", identity_description=draft.content,
                                                        action="DEFER", reason=str(error))
                    stored_write = {"changed_ids": stage.changed_ids, "action": decision.action,
                                    "reason": decision.reason}
                    saved = {**progress, "writes": {**progress.get("writes", {}), str(index): stored_write}}
                    self.store.commit(stage.versions, stage.dependencies, stage.operations,
                                      source_cutoff=source.source_order, progress={key: saved})
                    progress.update(saved)
                if stored_write["action"] == "DEFER":
                    report.incomplete = True
                    report.results.append({"stage": "reconcile", "source_id": source.id,
                                           "outcome": "DEFERRED", "reason": stored_write["reason"]})
                    self._remember_scope(progress, source, prefix, "reconcile", stored_write["reason"])
                else:
                    report.changed_ids.extend(stored_write["changed_ids"])
                    self._maintain(stored_write["changed_ids"], source, prefix, input_policy,
                                   key, progress, report)
                progress["completed_drafts"] = index + 1
                progress["report"] = report.model_dump(mode="json")
                self.store.set_progress(key, progress)
            except Exception:
                # 技术失败必须让调用方失败，不能以旧快照冒充已经完成新批次。
                progress["report"] = report.model_dump(mode="json")
                progress["technical_failed"] = True
                self.store.set_progress(key, progress)
                raise
        for unresolved in extraction.unresolved:
            report.results.append({"stage": "extract", "outcome": "DEFERRED",
                                   **unresolved.model_dump(mode="json")})
            report.incomplete = True
            source = source_map[unresolved.source_id]
            self._remember_scope(progress, source,
                                 SourceSpan(source_id=source.id, start=0, end=len(source.content)),
                                 "extract", unresolved.reason)
        report.changed_ids = list(dict.fromkeys(report.changed_ids))
        report.pending_ids = list(dict.fromkeys(report.pending_ids))
        report.generation_calls = progress["generation_calls"]
        progress.update(complete=True, technical_failed=False, report=report.model_dump(mode="json"))
        self.store.set_progress(key, progress)
        return report

    def _coordinate(
        self, draft: FactDraft, policy: InputPolicy, cutoff: int, prefix: SourceSpan,
        token: str, staged: StagedWrite | None = None,
    ) -> tuple[StagedWrite, CoordinationDecision]:
        staged = staged or StagedWrite()
        mandatory = [draft.source_id, *[ref.id for ref in draft.evidence_refs]]
        mandatory.extend(version.id for version in staged.versions)
        candidates = self._candidates([draft.content], "reconcile", cutoff, prefix,
                                      mandatory=mandatory, staged=staged)
        decision = self.reconciler.reconcile(draft, candidates["versions"], policy,
                                             evidence_context=candidates)
        if decision.identity == "NEW" and decision.action != "DEFER":
            checked = self._candidates([draft.content, decision.identity_description],
                                       "reconcile", cutoff, prefix, mandatory=mandatory, staged=staged)
            # NEW 只再查一次；第二次依然找不到即可由程序分配身份。
            if set(checked["version_ids"]) - set(candidates["version_ids"]):
                decision = self.reconciler.reconcile(draft, checked["versions"], policy,
                                                     identity_recheck=True, evidence_context=checked)
        return self._materialize(draft, decision, cutoff, token, staged), decision

    def maintain_change(self, changed_ids, source, input_policy, batch_id) -> MaintenanceReport:
        """可信控制接口已提交明确操作后，复用同一修复队列而不重新抽取命令。"""
        key = f"batch:{batch_id}"
        progress = self.store.get_progress(key) or {
            "source_cutoff": source.source_order, "generation_calls": 0, "fingerprints": [], "report": {},
        }
        progress.setdefault("pending_scopes", [])
        if progress.get("complete"):
            return MaintenanceReport.model_validate(progress["report"])
        report = MaintenanceReport.model_validate(progress["report"])
        report.changed_ids.extend(changed_ids)
        prefix = SourceSpan(source_id=source.id, start=0, end=len(source.content))
        try:
            self._maintain(changed_ids, source, prefix, input_policy, key, progress, report)
        except Exception:
            progress.update(technical_failed=True, report=report.model_dump(mode="json"))
            self.store.set_progress(key, progress)
            raise
        report.generation_calls = progress["generation_calls"]
        report.changed_ids = list(dict.fromkeys(report.changed_ids))
        report.pending_ids = list(dict.fromkeys(report.pending_ids))
        progress.update(complete=True, technical_failed=False, report=report.model_dump(mode="json"))
        self.store.set_progress(key, progress)
        return report

    def _materialize(
        self, draft: FactDraft, decision: CoordinationDecision, cutoff: int,
        token: str, staged: StagedWrite,
    ) -> StagedWrite:
        result = StagedWrite()
        if decision.action == "DEFER":
            return result
        view = self.store.view(source_cutoff=cutoff)
        versions = view.versions | {version.id: version for version in staged.versions}
        target = versions.get(decision.target_id)
        if decision.action in {"DELETE", "RETRACT"}:
            scope = decision.scope
            target_id = target.memory_key if scope == "key" and target else decision.target_id
            if scope in {"source", "span"}:
                if decision.target_span:
                    target_id = decision.target_span.source_id
                if target_id not in view.sources:
                    raise ValueError("Control refers to a source outside the visible prefix")
            op = ControlOperation(
                id=_id("op", token, decision.action), namespace=self.store.namespace,
                kind=decision.action.lower(), target_id=target_id, scope=scope,
                span=decision.target_span, effective_time=decision.effective_time,
                evidence_refs=draft.evidence_refs, reason=decision.reason, source_cutoff=cutoff,
                topic=decision.topic,
            )
            result.operations.append(op)
            result.changed_ids.append(target_id)
            return result
        if decision.action == "SUPPLEMENT":
            if target is None:
                raise ValueError("Supplement target is missing")
            state = self.evaluator.evaluate(target.id, source_cutoff=cutoff) if target.id in view.versions else None
            if state and state.status in {"superseded", "deleted"}:
                raise ValueError("Closed versions require a new reconfirmation period")
            result.target_id = target.id
        else:
            memory_key = decision.memory_key or _id("key", token)
            revision = None
            if decision.action == "REVISE":
                effective = decision.effective_time or draft.valid_time.start
                if effective is None:
                    ref = next((ref for ref in draft.evidence_refs if ref.type == "SOURCE"), None)
                    effective = TimePoint(order=cutoff, offset=ref.span.end if ref else 0)
                revision = Revision(previous_version_id=target.id,
                                    reason=decision.revision_reason,
                                    effective_time=effective, evidence_refs=draft.evidence_refs)
                result.changed_ids.append(target.id)
            version = MemoryVersion(
                id=_id("mem", token), namespace=self.store.namespace, memory_key=memory_key,
                content=draft.content, valid_time=draft.valid_time, modality=draft.modality,
                revision=revision,
            )
            if (revision is not None and revision.reason != "correction"
                    and version.valid_time.kind == "state" and version.valid_time.start is None):
                version = version.model_copy(update={"valid_time": version.valid_time.model_copy(
                    update={"start": revision.effective_time, "precision": "order"
                            if revision.effective_time.date is None else version.valid_time.precision})})
            result.versions.append(version)
            result.target_id = version.id
            if decision.action == "CONFLICT":
                for alternative in (target.id, version.id):
                    result.operations.append(ControlOperation(
                        id=_id("op", token, "conflict", alternative), namespace=self.store.namespace,
                        kind="conflict", target_id=alternative,
                        evidence_refs=draft.evidence_refs, reason=decision.reason, source_cutoff=cutoff,
                    ))
                result.changed_ids.append(target.id)
        result.dependencies.append(DependencyLink(
            id=_id("dep", token, result.target_id), namespace=self.store.namespace,
            target_version_id=result.target_id, premise_refs=draft.evidence_refs,
        ))
        result.changed_ids.append(result.target_id)
        for resolution in decision.resolved_conflicts:
            result.operations.append(ControlOperation(
                id=_id("op", token, "resolve_conflict", resolution.target_id), namespace=self.store.namespace,
                kind="resolve_conflict", target_id=resolution.target_id, evidence_refs=draft.evidence_refs,
                reason=decision.reason, source_cutoff=cutoff,
            ))
            if resolution.outcome == "close":
                result.operations.append(ControlOperation(
                    id=_id("op", token, "close_conflict", resolution.target_id), namespace=self.store.namespace,
                    kind="close", target_id=resolution.target_id, evidence_refs=draft.evidence_refs,
                    effective_time=decision.effective_time or TimePoint(order=cutoff),
                    reason=decision.reason, source_cutoff=cutoff, replacement_id=result.target_id,
                ))
            result.changed_ids.append(resolution.target_id)
        return result

    def _candidates(
        self, queries: list[str], mode: str, cutoff: int, prefix: SourceSpan,
        mandatory: list[str], staged: StagedWrite | None = None,
    ) -> dict:
        matches = self.retriever.retrieve(
            queries, mode=mode, mandatory_context=mandatory,
            source_cutoff=cutoff, source_span_cutoff=prefix,
        )
        ids = list(dict.fromkeys([*mandatory, *[
            match.source_id if match.kind == "source" else match.id for match in matches
        ]]))
        source_spans: dict[str, list[SourceSpan]] = {}
        for match in matches:
            if match.kind == "source":
                source_spans.setdefault(match.source_id, []).append(match.span)
        context = self._context(ids, cutoff, prefix, staged, source_spans)
        # 独立语义候选可以少取；已知结构证据不能截成半条路径。
        while self.llm.count_tokens(_json(context)) > self.config.max_context_tokens - 4000:
            removable = [item for item in ids if item not in mandatory]
            if not removable:
                raise ContextLimitError("A necessary proof exceeds max_context_tokens")
            ids.remove(removable[-1])
            context = self._context(ids, cutoff, prefix, staged, source_spans)
        return context

    def _context(
        self, ids: list[str], cutoff: int, prefix: SourceSpan,
        staged: StagedWrite | None = None,
        source_spans: dict[str, list[SourceSpan]] | None = None,
    ) -> dict:
        view = self.store.view(source_cutoff=cutoff)
        staged = staged or StagedWrite()
        versions = view.versions | {version.id: version for version in staged.versions}
        dependencies = view.dependencies | {dep.id: dep for dep in staged.dependencies}
        chosen_versions: set[str] = set()
        chosen_sources: set[str] = set()
        chosen_deps: set[str] = set()
        spans = {source_id: list(items) for source_id, items in (source_spans or {}).items()}
        spans.setdefault(prefix.source_id, []).append(prefix)
        stack = list(ids)
        while stack:
            item = stack.pop()
            if item in view.sources:
                chosen_sources.add(item)
                continue
            if item not in versions or item in chosen_versions:
                continue
            version = versions[item]
            if item in view.versions and not self.evaluator.content_visible(item, source_cutoff=cutoff):
                continue
            chosen_versions.add(item)
            # 同版本族用于判断历史/替代关系，而不是自动重绑支持引用。
            stack.extend(other.id for other in versions.values() if other.memory_key == version.memory_key)
            refs = list(version.revision.evidence_refs) if version.revision else []
            for dep in dependencies.values():
                if dep.target_version_id == item:
                    chosen_deps.add(dep.id)
                    refs.extend(dep.premise_refs)
            for operation in view.operations:
                if (operation.kind in {"close", "supersede", "correct"}
                        and (operation.target_id == item or operation.scope == "key" and operation.target_id == version.memory_key)):
                    refs.extend(operation.evidence_refs)
            for ref in refs:
                stack.append(ref.id)
                if ref.type == "SOURCE":
                    spans.setdefault(ref.id, []).append(ref.span)
                for span in ref.context_refs:
                    spans.setdefault(span.source_id, []).append(span)
                stack.extend(span.source_id for span in ref.context_refs)
        serialized_versions = []
        for version_id in sorted(chosen_versions):
            version = versions[version_id]
            data = version.model_dump(mode="json", exclude={"namespace", "created_at", "status"})
            if version_id in view.versions:
                data["resolution"] = self.evaluator.evaluate(version_id, source_cutoff=cutoff).model_dump(mode="json")
            else:
                data["verification_state"] = "staged_not_committed"
            serialized_versions.append(data)
        serialized_sources = []
        for source_id in sorted(chosen_sources, key=lambda item: view.sources[item].source_order):
            source = view.sources[source_id]
            # 这里要求检索器统一遮蔽删除片段；相同消息再额外裁到当前事实末尾。
            text = self.evaluator.source_text(source.id, source_cutoff=cutoff, source_span_cutoff=prefix)
            if not text.strip():
                continue
            data = source.model_dump(mode="json", exclude={"namespace", "message_id", "created_at"})
            ranges = spans.get(source_id) or [SourceSpan(source_id=source_id, start=0, end=len(text))]
            intervals = []
            for span in sorted(ranges, key=lambda item: item.start):
                end = min(span.end, prefix.end) if source_id == prefix.source_id else span.end
                if span.start >= end:
                    continue
                if intervals and span.start <= intervals[-1][1]:
                    intervals[-1] = (intervals[-1][0], max(end, intervals[-1][1]))
                else:
                    intervals.append((span.start, end))
            data["fragments"] = [{"span": {"source_id": source_id, "start": start, "end": end},
                                  "text": text[start:end]} for start, end in intervals if text[start:end].strip()]
            data["content"] = "\n".join(fragment["text"] for fragment in data["fragments"])
            data["original_length"] = len(source.content)
            if not data["content"].strip():
                continue
            serialized_sources.append(data)
        source_ids = [source["id"] for source in serialized_sources]
        operations = [op.model_dump(mode="json", exclude={"reason", "namespace", "created_at"}
                                    | ({"evidence_refs"} if op.kind == "delete" else set())) for op in view.operations
                      if op.target_id in chosen_versions or op.target_id in chosen_sources
                      or op.target_id in {versions[item].memory_key for item in chosen_versions}]
        return {"versions": serialized_versions, "sources": serialized_sources,
                "dependencies": [dependencies[item].model_dump(mode="json", exclude={"namespace", "created_at"})
                                 for item in sorted(chosen_deps)],
                "operations": operations, "version_ids": sorted(chosen_versions), "source_ids": source_ids,
                "evaluation_time": {"order": cutoff, "offset": prefix.end},
                "source_cutoff": cutoff}

    def _mark_pending(self, target_ids: list[str], cutoff: int, reason: str) -> None:
        ops = []
        view = self.store.view(source_cutoff=cutoff)
        flags = {}
        for operation in view.operations:
            if operation.kind in {"pending", "resolve_pending"}:
                flags[operation.target_id] = operation.kind
        for target_id in target_ids:
            if target_id not in view.versions:
                continue
            if flags.get(target_id) == "pending":
                continue
            ops.append(ControlOperation(namespace=self.store.namespace, kind="pending",
                                        target_id=target_id, reason=reason, source_cutoff=cutoff))
        if ops:
            self.store.commit(operations=ops, source_cutoff=cutoff)

    def _check_premises(self, refs, context, cutoff, prefix, staged) -> None:
        """只检查可确定的引用与时点；自然语言充分性仍由验证器负责。"""
        supplied = {source["id"]: source for source in context["sources"]}
        staged_ids = {version.id for version in staged.versions}
        for ref in refs:
            if ref.type == "SOURCE":
                for span in [ref.span, *ref.context_refs]:
                    source = supplied.get(span.source_id)
                    if source is None:
                        raise ValueError("A premise source was not supplied to verification")
                    visible_ranges = [fragment["span"] for fragment in source["fragments"]]
                    if not any(item["start"] <= span.start < span.end <= item["end"] for item in visible_ranges):
                        raise ValueError("A premise span lies outside the supplied evidence")
                    raw = self.store.get_source(span.source_id).content[span.start:span.end]
                    allowed = self.evaluator.source_text(span.source_id, source_cutoff=cutoff,
                                                         source_span_cutoff=prefix,
                                                         include_retracted=True)[span.start:span.end]
                    if raw != allowed:
                        raise ValueError("A premise crosses deleted, withdrawn, or future source text")
            elif ref.id not in staged_ids:
                result = self.evaluator.evaluate(ref.id, query_time=ref.at_time if ref.type == "HISTORICAL" else None,
                                                 source_cutoff=cutoff)
                if not result.usable:
                    raise ValueError(f"The premise is not applicable: {result.reason}")

    def _maintain(
        self, changed: list[str], source: Source, prefix: SourceSpan, policy: InputPolicy,
        progress_key: str, progress: dict, report: MaintenanceReport,
    ) -> None:
        cutoff = source.source_order
        maintenance = self.evaluator.recompute(changed, source_cutoff=cutoff)
        pending = list(maintenance.pending_ids)
        self._mark_pending(pending, cutoff, "dependency_changed")
        queue: list[tuple[list[str], list[str]]] = [(changed, pending)]
        def defer(targets, reason):
            self._deferred(report, targets, reason)
            self._remember_scope(progress, source, prefix, "discovery", reason, trigger_ids=triggers)
        while queue:
            triggers, targets = queue.pop(0)
            reserved = any(fingerprint not in progress["fingerprints"]
                           for fingerprint in progress.get("generation_tasks", {}))
            if (progress["generation_calls"] >= self.config.max_generation_calls_per_update and not reserved):
                defer(targets, "generation_budget")
                continue
            if len(targets) > self.config.max_repair_targets_per_call:
                size = self.config.max_repair_targets_per_call
                queue[0:0] = [(triggers, targets[start:start + size]) for start in range(0, len(targets), size)]
                continue
            view = self.store.view(source_cutoff=cutoff)
            queries = [view.versions[item].content for item in triggers if item in view.versions
                       and self.evaluator.content_visible(item, source_cutoff=cutoff)]
            visible_source = self.evaluator.source_text(source.id, source_cutoff=cutoff,
                                                        source_span_cutoff=prefix)
            queries.append(visible_source[prefix.start:prefix.end].strip() or "Changed memory controls")
            try:
                context = self._candidates(queries, "derive", cutoff, prefix, [*triggers, *targets])
            except ContextLimitError as error:
                if len(targets) > 1:
                    midpoint = len(targets) // 2
                    queue[0:0] = [(triggers, targets[:midpoint]), (triggers, targets[midpoint:])]
                else:
                    defer(targets, str(error))
                continue
            context["input_policy"] = policy.model_dump(mode="json")
            fingerprint = sha256(_json((context, targets, self.config.model_dump(exclude={"api_key"}))).encode()).hexdigest()
            if fingerprint in progress["fingerprints"]:
                continue
            tasks = progress.setdefault("generation_tasks", {})
            if fingerprint not in tasks:
                if progress["generation_calls"] >= self.config.max_generation_calls_per_update:
                    defer(targets, "generation_budget")
                    continue
                progress["generation_calls"] += 1
                tasks[fingerprint] = {"ordinal": progress["generation_calls"]}
                self.store.set_progress(progress_key, progress)
            task = tasks[fingerprint]
            if "proposal" not in task:
                try:
                    proposal = self.inducer.propose(context, targets)
                except (ContextLimitError, OutputLimitError) as error:
                    progress["fingerprints"].append(fingerprint)
                    self.store.set_progress(progress_key, progress)
                    if len(targets) > 1:
                        midpoint = len(targets) // 2
                        queue[0:0] = [(triggers, targets[:midpoint]), (triggers, targets[midpoint:])]
                    else:
                        defer(targets, str(error))
                    continue
                task["proposal"] = proposal.model_dump(mode="json")
                self.store.set_progress(progress_key, progress)
            proposal = LocalGraphProposal.model_validate(task["proposal"])
            if proposal.gap_queries or proposal.open_queries:
                # 补检只在本次局部调用范围内执行一次，不递归发散。
                try:
                    extra = self._candidates([*queries, *proposal.gap_queries, *proposal.open_queries],
                                             "derive", cutoff, prefix, [*triggers, *targets])
                    extra["input_policy"] = policy.model_dump(mode="json")
                    if (task.get("gap_reserved") or progress["generation_calls"] < self.config.max_generation_calls_per_update):
                        if not task.get("gap_reserved"):
                            progress["generation_calls"] += 1
                            task["gap_reserved"] = True
                        self.store.set_progress(progress_key, progress)
                        if "gap_proposal" not in task:
                            expanded_proposal = self.inducer.propose(extra, targets)
                            task["gap_proposal"] = expanded_proposal.model_dump(mode="json")
                            self.store.set_progress(progress_key, progress)
                        proposal = LocalGraphProposal.model_validate(task["gap_proposal"])
                        context = extra
                    else:
                        task["gap_incomplete"] = True
                        defer(targets, "gap_generation_budget")
                except (ContextLimitError, OutputLimitError) as error:
                    task["gap_incomplete"] = True
                    defer(targets, str(error))
            token = f"{progress_key}:generation:{task['ordinal']}"
            accepted, failed = self._commit_graph(proposal, context, policy, cutoff, prefix, token)
            progress["fingerprints"].append(fingerprint)
            self.store.set_progress(progress_key, progress)
            report.results.extend(failed)
            report.incomplete |= bool(failed)
            if not failed and not task.get("gap_incomplete"):
                progress["pending_scopes"] = [scope for scope in progress["pending_scopes"]
                                               if not (scope["source_id"] == source.id
                                                       and scope["span"] == prefix.model_dump(mode="json")
                                                       and scope.get("trigger_ids", []) == sorted(triggers)
                                                       and scope["stage"] == "discovery")]
            if accepted:
                report.changed_ids.extend(accepted)
                result = self.evaluator.recompute(accepted, source_cutoff=cutoff)
                self._mark_pending(result.pending_ids, cutoff, "dependency_changed")
                # 新版本需要继续发现未知下游，不能只重算已有边就停止级联。
                queue.append((accepted, result.pending_ids))
            for repair in proposal.repairs:
                if repair.outcome == "DEFERRED" or any(item.get("target_id") == repair.target_id for item in failed):
                    report.pending_ids.append(repair.target_id)
                    report.incomplete = True

    @staticmethod
    def _deferred(report: MaintenanceReport, targets: list[str], reason: str) -> None:
        report.incomplete = True
        report.pending_ids.extend(targets)
        report.results.append({"stage": "generate", "outcome": "DEFERRED",
                               "reason": reason, "targets": targets})

    @staticmethod
    def _remember_scope(progress, source, span, stage, reason, trigger_ids=()):
        scope = {"source_id": source.id, "source_cutoff": source.source_order,
                 "span": span.model_dump(mode="json"), "stage": stage, "reason": reason,
                 "trigger_ids": sorted(trigger_ids)}
        scopes = progress.setdefault("pending_scopes", [])
        scopes[:] = [item for item in scopes if not (item["source_id"] == source.id
                    and item["span"] == scope["span"] and item["stage"] == stage
                    and item.get("trigger_ids", []) == scope["trigger_ids"])]
        scopes.append(scope)

    def _commit_graph(
        self, graph: LocalGraphProposal, context: dict, policy: InputPolicy,
        cutoff: int, prefix: SourceSpan, token: str,
    ) -> tuple[list[str], list[dict]]:
        claims = {claim.temporary_id: claim for claim in graph.claims}
        expected_seq = self.store.current_seq
        remaining = list(graph.dependencies)
        staged = StagedWrite()
        mapping: dict[str, str] = {}
        failed: list[dict] = []
        completed_targets: set[str] = set()
        while remaining:
            ready_targets = []
            for dep in remaining:
                target_deps = [item for item in remaining if item.target_id == dep.target_id]
                if all(ref.id not in claims or ref.id in mapping for item in target_deps for ref in item.premise_refs):
                    ready_targets.append(dep.target_id)
            if not ready_targets:
                failed.extend({"target_id": dep.target_id, "stage": "verify", "reason": "cyclic_or_rejected_premise"} for dep in remaining)
                break
            for target_id in dict.fromkeys(ready_targets):
                group = [dep for dep in remaining if dep.target_id == target_id]
                remaining = [dep for dep in remaining if dep.target_id != target_id]
                verified_links: list[DependencyLink] = []
                target_stage = StagedWrite()
                target_content = claims[target_id].content if target_id in claims else self.store.get_version(target_id).content
                for dep in group:
                    if (target_id not in claims and dep.effect == "SUPPORT"
                            and self.evaluator.evaluate(target_id, source_cutoff=cutoff).status in {"superseded", "deleted"}):
                        failed.append({"target_id": target_id, "stage": "verify",
                                       "reason": "Closed versions require reconfirmation as a new period"})
                        continue
                    refs = [ref.model_copy(update={"id": mapping.get(ref.id, ref.id)}) for ref in dep.premise_refs]
                    resolved_dep = dep.model_copy(update={"premise_refs": refs})
                    premise_ids = [ref.id for ref in refs]
                    try:
                        verify_context = self._candidates([target_content], "validate", cutoff, prefix,
                                                           premise_ids + ([target_id] if target_id not in claims else []), staged)
                    except ContextLimitError as error:
                        failed.append({"target_id": target_id, "stage": "verify", "reason": str(error)})
                        continue
                    verify_context.update(target=claims[target_id].model_dump(mode="json") if target_id in claims
                                          else {**self.store.get_version(target_id).model_dump(mode="json", exclude={"namespace", "created_at", "status"}),
                                                "resolution": self.evaluator.evaluate(target_id, source_cutoff=cutoff).model_dump(mode="json")},
                                          input_policy=policy.model_dump(mode="json"))
                    try:
                        self._check_premises(refs, verify_context, cutoff, prefix, staged)
                    except ValueError as error:
                        failed.append({"target_id": target_id, "stage": "verify", "reason": str(error)})
                        continue
                    try:
                        verification = self.inducer.verify(resolved_dep, verify_context)
                    except (ContextLimitError, OutputLimitError) as error:
                        failed.append({"target_id": target_id, "stage": "verify", "reason": str(error)})
                        continue
                    if not verification.accepted:
                        failed.append({"target_id": target_id, "stage": "verify", "reason": verification.reason})
                        continue
                    for path in verification.sufficient_paths:
                        verified_links.append(DependencyLink(
                            id=_id("dep", token, dep.temporary_id, path), namespace=self.store.namespace,
                            target_version_id=target_id, premise_refs=[refs[index] for index in path],
                            effect=dep.effect, effective_time=dep.effective_time,
                        ))
                if not verified_links:
                    continue
                if target_id in claims:
                    claim = claims[target_id]
                    support_refs = next(link.premise_refs for link in verified_links if link.effect == "SUPPORT")
                    draft = FactDraft(content=claim.content, valid_time=claim.valid_time,
                                      modality=claim.modality, evidence_refs=support_refs,
                                      source_id=prefix.source_id)
                    try:
                        target_stage, decision = self._coordinate(draft, policy, cutoff, prefix,
                                                                   f"{token}:{target_id}", staged)
                    except (ContextLimitError, OutputLimitError) as error:
                        failed.append({"target_id": target_id, "stage": "reconcile", "reason": str(error)})
                        continue
                    if decision.action in {"DEFER", "CONFLICT"}:
                        failed.append({"target_id": target_id, "stage": "reconcile", "reason": decision.reason})
                        continue
                    real_id = target_stage.target_id
                    mapping[target_id] = real_id
                    # _materialize 提供的单路径由逐条验证后的完整 OR 路径替换。
                    target_stage.dependencies = []
                else:
                    real_id = target_id
                for link in verified_links:
                    target_stage.dependencies.append(link.model_copy(update={"target_version_id": real_id}))
                staged.versions.extend(target_stage.versions)
                staged.dependencies.extend(target_stage.dependencies)
                staged.operations.extend(target_stage.operations)
                staged.changed_ids.extend(target_stage.changed_ids or [real_id])
                completed_targets.add(target_id)
        replaced = {version.revision.previous_version_id for version in staged.versions if version.revision}
        staged.dependencies = [dep for dep in staged.dependencies
                               if not (dep.effect == "INVALIDATE" and dep.target_version_id in replaced)]
        for dep in staged.dependencies:
            if dep.effect == "INVALIDATE":
                staged.operations.append(ControlOperation(
                    id=_id("op", token, "known_unknown", dep.target_version_id), namespace=self.store.namespace,
                    kind="resolve_pending", target_id=dep.target_version_id,
                    reason="UNKNOWN", source_cutoff=cutoff,
                ))
        for repair in graph.repairs:
            proof_target = repair.proposal_id if repair.outcome == "UPDATED" else repair.target_id
            if repair.outcome != "DEFERRED" and proof_target in completed_targets:
                if repair.outcome == "UPDATED":
                    new_id = mapping[repair.proposal_id]
                    new_version = next((version for version in staged.versions if version.id == new_id), None)
                    new_version = new_version or self.store.get_version(new_id)
                    old_version = self.store.get_version(repair.target_id)
                    if new_version.memory_key != old_version.memory_key:
                        failed.append({"target_id": repair.target_id, "stage": "repair",
                                       "reason": "Replacement did not retain the repaired memory identity"})
                        continue
                    if new_id != old_version.id:
                        revision_matches = new_version.revision and new_version.revision.previous_version_id == old_version.id
                        prior_replacement = any(op.target_id == old_version.id and op.replacement_id == new_id
                                                for op in self.store.operations())
                        if not revision_matches and not prior_replacement:
                            failed.append({"target_id": repair.target_id, "stage": "repair",
                                           "reason": "Proposed new value did not replace the repair target"})
                            continue
                staged.operations.append(ControlOperation(
                    id=_id("op", token, "resolved", repair.target_id), namespace=self.store.namespace,
                    kind="resolve_pending", target_id=repair.target_id,
                    reason=repair.outcome, source_cutoff=cutoff,
                ))
            else:
                failed.append({"target_id": repair.target_id, "stage": "repair", "reason": repair.reason})
        operation_map = {op.id: op for op in self.store.view(source_cutoff=cutoff).operations}
        for review in graph.control_reviews:
            operation = operation_map[review.operation_id]
            if review.decision == "DEFERRED":
                failed.append({"target_id": operation.target_id, "stage": "control_review", "reason": review.reason})
                continue
            try:
                review_context = self._candidates([review.reason], "validate", cutoff, prefix,
                                                   [operation.target_id, *[ref.id for ref in review.evidence_refs]], staged)
                self._check_premises(review.evidence_refs, review_context, cutoff, prefix, staged)
                verification = self.inducer.verify_control(review, operation.model_dump(mode="json", exclude={"namespace", "created_at"}), review_context)
            except (ValueError, ContextLimitError, OutputLimitError) as error:
                failed.append({"target_id": operation.target_id, "stage": "control_review", "reason": str(error)})
                continue
            if not verification.accepted:
                failed.append({"target_id": operation.target_id, "stage": "control_review", "reason": verification.reason})
                continue
            selected = verification.sufficient_paths[0]
            refs = [review.evidence_refs[index] for index in selected]
            if review.decision == "REVOKE":
                staged.operations.append(ControlOperation(
                    id=_id("op", token, "revoke", operation.id), namespace=self.store.namespace,
                    kind="revoke_control", target_id=operation.id, scope="operation",
                    evidence_refs=refs, reason=review.reason, source_cutoff=cutoff,
                ))
            if operation.scope == "version":
                staged.operations.append(ControlOperation(
                    id=_id("op", token, "control_reviewed", operation.id), namespace=self.store.namespace,
                    kind="resolve_pending", target_id=operation.target_id,
                    evidence_refs=refs, reason="control_reviewed", source_cutoff=cutoff,
                ))
                staged.changed_ids.append(operation.target_id)
        if staged.dependencies or staged.operations:
            try:
                sequence = self.store.commit(staged.versions, staged.dependencies, staged.operations,
                                             expected_seq=expected_seq, source_cutoff=cutoff)
                if sequence == expected_seq:
                    return [], failed
            except ValueError as error:
                # 全局环、深度、引用错误使整组回滚；不留下引用未提交前提的上层。
                failed.append({"stage": "commit", "reason": str(error)})
                return [], failed
        return list(dict.fromkeys(staged.changed_ids)), failed
