"""按来源顺序协调内容，提交已验证局部图，并消费有界的修复队列。"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json

from .extractor import ExtractionResult, FactDraft, FactExtractor
from .inducer import (DependencyInducer, DependencyProposal, LocalGraphProposal,
                      VERIFICATION_PROMPT, CONTROL_VERIFICATION_PROMPT)
from .llm import ContextLimitError, OutputLimitError, RecoverableModelError, StructuredOutputError, failure_details
from .models import (
    ControlOperation, DependencyLink, InputPolicy, MaintenanceReport, MemoryVersion,
    PremiseRef, Revision, Source, SourceSpan, TimePoint,
)
from .reconciler import CoordinationDecision, MemoryReconciler
from .structured_output import fits_request
from .store import dependency_signature


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
        progress.setdefault("unresolved_drafts", {})
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
                        stage, decision = self._coordinate(draft, input_policy, source.source_order, prefix, token)
                    except RecoverableModelError as error:
                        # 原文已保存；未决关系不猜测、不关闭旧值，也不阻挡无关输入。
                        progress["unresolved_drafts"][str(index)] = {"draft": draft.model_dump(mode="json"),
                            "source_cutoff": source.source_order, **failure_details(error)}
                        self._remember_scope(progress, source, prefix, "reconcile", str(error), error=error)
                        progress["completed_drafts"] = index + 1
                        report.incomplete = True
                        progress["report"] = report.model_dump(mode="json")
                        self.store.set_progress(key, progress)
                        continue
                    stored_write = {"changed_ids": stage.changed_ids, "action": decision.action,
                                    "reason": decision.reason}
                    saved = {**progress, "writes": {**progress.get("writes", {}), str(index): stored_write}}
                    self.store.commit(stage.versions, stage.dependencies, stage.operations,
                                      source_cutoff=source.source_order, progress={key: saved})
                    progress.update(saved)
                if stored_write["action"] == "DEFER":
                    progress["unresolved_drafts"][str(index)] = {"draft": draft.model_dump(mode="json"),
                        "source_cutoff": source.source_order, "reason": stored_write["reason"]}
                    report.incomplete = True
                    report.results.append({"stage": "reconcile", "source_id": source.id,
                                           "outcome": "DEFERRED", "reason": stored_write["reason"]})
                    self._remember_scope(progress, source, prefix, "reconcile", stored_write["reason"])
                else:
                    report.changed_ids.extend(stored_write["changed_ids"])
                    self._maintain(stored_write["changed_ids"], source, prefix, input_policy,
                                   key, progress, report, discover=False)
                progress["completed_drafts"] = index + 1
                progress["report"] = report.model_dump(mode="json")
                self.store.set_progress(key, progress)
            except Exception:
                # 技术失败必须让调用方失败，不能以旧快照冒充已经完成新批次。
                progress["report"] = report.model_dump(mode="json")
                progress["technical_failed"] = True
                self.store.set_progress(key, progress)
                raise
        # 可选归纳在批末执行一次；批内必要维护仍按各自的来源前缀处理。
        if not progress.get("discovery_complete"):
            last = sources[-1]
            span = SourceSpan(source_id=last.id, start=0, end=len(last.content))
            self._maintain(list(dict.fromkeys(report.changed_ids)), last, span, input_policy,
                           key, progress, report, discover=True)
            progress["discovery_complete"] = True
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
        mandatory.extend(span.source_id for ref in draft.evidence_refs for span in ref.context_refs)
        mandatory.extend(version.id for version in staged.versions)
        def request(context, recheck=False):
            return self.reconciler.prompt(draft), self.reconciler.payload(
                draft, context["versions"], policy, recheck, context)
        candidates = self._candidates([draft.content], "reconcile", cutoff, prefix,
                                      mandatory=mandatory, staged=staged, request_builder=request)
        try:
            decision = self.reconciler.reconcile(draft, candidates["versions"], policy, evidence_context=candidates)
        except StructuredOutputError:
            # 一次针对关系的独立补检；不在同一组错误候选上无限纠错。
            checked = self._candidates([draft.content, f"Existing subject and attribute for: {draft.content}"],
                                       "reconcile", cutoff, prefix, mandatory=mandatory, staged=staged,
                                       request_builder=lambda context: request(context, True))
            decision = self.reconciler.reconcile(draft, checked["versions"], policy,
                                                 identity_recheck=True, evidence_context=checked)
        if decision.identity == "NEW" and decision.action != "DEFER":
            checked = self._candidates([draft.content, decision.identity_description],
                                       "reconcile", cutoff, prefix, mandatory=mandatory, staged=staged,
                                       request_builder=lambda context: request(context, True))
            # 候选没变化不代表 NEW 正确；复核只看简短陈述，不再搬入全部原文。
            if checked["versions"]:
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
        *, request_builder,
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
        def fits(value):
            return fits_request(self.llm, self.config, *request_builder(value))
        if not fits(context):
            optional = [item for item in ids if item not in mandatory]
            context = self._context(mandatory, cutoff, prefix, staged, source_spans)
            if not fits(context):
                raise ContextLimitError("A necessary proof exceeds max_context_tokens")
            low, high = 0, len(optional)
            while low < high:
                middle = (low + high + 1) // 2
                trial = self._context([*mandatory, *optional[:middle]], cutoff, prefix, staged, source_spans)
                if fits(trial):
                    low, context = middle, trial
                else:
                    high = middle - 1
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
        direct_version_ids = set(ids) & versions.keys()
        source_facts = {}
        for dependency in dependencies.values():
            if dependency.effect == "SUPPORT" and all(ref.type == "SOURCE" for ref in dependency.premise_refs):
                for ref in dependency.premise_refs:
                    source_facts.setdefault(ref.id, []).append((dependency.target_version_id, ref.span))
        while stack:
            item = stack.pop()
            if item in view.sources:
                if item in chosen_sources:
                    continue
                chosen_sources.add(item)
                if item not in ids:
                    continue
                ranges = spans.get(item)
                for version_id, quote in source_facts.get(item, []):
                    if item == prefix.source_id and quote.end > prefix.end:
                        continue
                    if ranges and not any(quote.start < span.end and span.start < quote.end for span in ranges):
                        continue
                    stack.append(version_id)
                    direct_version_ids.add(version_id)
                    family = versions[version_id].memory_key
                    stack.extend(v.id for v in versions.values() if v.memory_key == family
                                 and v.id in view.versions and self.evaluator.evaluate(v.id, source_cutoff=cutoff).usable)
                continue
            if item not in versions or item in chosen_versions:
                continue
            version = versions[item]
            if item in view.versions and not self.evaluator.content_visible(item, source_cutoff=cutoff):
                continue
            chosen_versions.add(item)
            # 只展开直接修订与选中的完整路径，不把整个版本族搬入每个请求。
            if version.revision and item in direct_version_ids:
                stack.append(version.revision.previous_version_id)
            refs = list(version.revision.evidence_refs) if version.revision else []
            paths = [dep for dep in dependencies.values() if dep.target_version_id == item and dep.effect == "SUPPORT"]
            if item in view.versions:
                usable = set(self.evaluator.evaluate(item, source_cutoff=cutoff).support_ids)
                paths = [dep for dep in paths if dep.id in usable] or paths
            if paths:
                dep = min(paths, key=lambda value: len(_json(value.premise_refs)))
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
        for version_id in dict.fromkeys([*(item for item in ids if item in chosen_versions), *sorted(chosen_versions)]):
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

    def _semantic_state(self, ids, cutoff):
        view = self.store.view(source_cutoff=cutoff)
        result = {}
        for key in sorted(set(ids)):
            if key in view.versions:
                version = view.versions[key]
                state = self.evaluator.evaluate(key, source_cutoff=cutoff)
                result[key] = (version.content, version.valid_time.model_dump(mode="json"), version.modality,
                    state.usable, state.reason, sorted(dependency_signature(d) for d in view.dependencies.values()
                                                     if d.id in state.support_ids))
            elif key in view.sources:
                result[key] = self.evaluator.source_text(key, source_cutoff=cutoff)
        return result

    def _reviewable_operations(self, changed, cutoff):
        evaluation = self.evaluator._evaluation(source_cutoff=cutoff)
        changed = set(changed)
        return [op.id for op in evaluation.operations if op.kind in {"close", "supersede", "correct"}
                and any(ref.id in changed or any(span.source_id in changed for span in ref.context_refs)
                        for ref in op.evidence_refs) and not evaluation.control_evidence_valid(op)]

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
        progress_key: str, progress: dict, report: MaintenanceReport, *, discover=True,
    ) -> None:
        cutoff = source.source_order
        maintenance = self.evaluator.recompute(changed, source_cutoff=cutoff)
        pending = list(maintenance.pending_ids)
        if not changed and not pending:
            return
        self._mark_pending(pending, cutoff, "dependency_changed")
        queue: list[tuple[list[str], list[str]]] = [(changed, pending)] if (discover or pending) else []
        def defer(targets, reason, error=None):
            if not targets:
                progress.setdefault("discovery_outcomes", []).append({"reason": reason,
                    **(failure_details(error) if error else {})})
                return
            self._deferred(report, targets, reason)
            if error is not None:
                report.results[-1].update(failure_details(error))
            self._remember_scope(progress, source, prefix, "discovery", reason, trigger_ids=triggers, error=error)
        while queue:
            triggers, targets = queue.pop(0)
            reserved = any(fingerprint not in progress["fingerprints"]
                           for fingerprint in progress.get("generation_tasks", {}))
            if (not targets and progress["generation_calls"] >= self.config.max_generation_calls_per_update and not reserved):
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
                reviewable = self._reviewable_operations(triggers, cutoff)
                build_request = lambda value: self.inducer.generation_request(
                    {**value, "input_policy": policy.model_dump(mode="json"), "reviewable_operation_ids": reviewable,
                     "trigger_ids": [*triggers, source.id]}, targets)
                context = self._candidates(queries, "derive", cutoff, prefix, [*triggers, *targets],
                                           request_builder=build_request)
            except ContextLimitError as error:
                if len(targets) > 1:
                    midpoint = len(targets) // 2
                    queue[0:0] = [(triggers, targets[:midpoint]), (triggers, targets[midpoint:])]
                else:
                    defer(targets, str(error), error)
                continue
            except RecoverableModelError as error:
                defer(targets, str(error), error)
                continue
            context["input_policy"] = policy.model_dump(mode="json")
            context.update(reviewable_operation_ids=reviewable, trigger_ids=[*triggers, source.id])
            # 指纹只依赖语义条件；进度记录、临时查询和重复的解除暂停不引起重试。
            fingerprint = sha256(_json((sorted(targets), sorted(triggers),
                self._semantic_state([*triggers, *targets], cutoff), reviewable)).encode()).hexdigest()
            if fingerprint in progress["fingerprints"]:
                continue
            tasks = progress.setdefault("generation_tasks", {})
            if fingerprint not in tasks:
                if not targets and progress["generation_calls"] >= self.config.max_generation_calls_per_update:
                    defer(targets, "generation_budget")
                    continue
                if targets:
                    progress["repair_calls"] = progress.get("repair_calls", 0) + 1
                else:
                    progress["generation_calls"] += 1
                tasks[fingerprint] = {"ordinal": len(tasks) + 1, "repair": bool(targets)}
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
                        defer(targets, str(error), error)
                    continue
                except RecoverableModelError as error:
                    # 已用完本次调用的重试，不在刷新或重启时重新抽奖。
                    task["failure"] = failure_details(error)
                    progress["fingerprints"].append(fingerprint)
                    defer(targets, str(error), error)
                    self.store.set_progress(progress_key, progress)
                    continue
                task["proposal"] = proposal.model_dump(mode="json")
                self.store.set_progress(progress_key, progress)
            proposal = LocalGraphProposal.model_validate(task["proposal"])
            if not (proposal.claims or proposal.dependencies or proposal.repairs or proposal.control_reviews
                    or proposal.gap_queries or proposal.open_queries):
                progress.setdefault("discovery_outcomes", []).append({"reason": proposal.no_op_reason or "model_returned_no_proposal",
                                                                       "kind": "no_new_memory"})
            if proposal.gap_queries or proposal.open_queries:
                # 补检只在本次局部调用范围内执行一次，不递归发散。
                try:
                    extra = self._candidates([*queries, *proposal.gap_queries, *proposal.open_queries],
                                             "derive", cutoff, prefix, [*triggers, *targets], request_builder=build_request)
                    extra["input_policy"] = policy.model_dump(mode="json")
                    extra.update(reviewable_operation_ids=reviewable, trigger_ids=[*triggers, source.id])
                    if (targets or task.get("gap_reserved") or progress["generation_calls"] < self.config.max_generation_calls_per_update):
                        if not task.get("gap_reserved"):
                            if targets:
                                progress["repair_calls"] = progress.get("repair_calls", 0) + 1
                            else:
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
                    defer(targets, str(error), error)
                except RecoverableModelError as error:
                    task["gap_incomplete"] = True
                    task["gap_failure"] = failure_details(error)
                    defer(targets, str(error), error)
            token = f"{progress_key}:generation:{task['ordinal']}"
            accepted, failed = self._commit_graph(proposal, context, policy, cutoff, prefix, token)
            progress["fingerprints"].append(fingerprint)
            self.store.set_progress(progress_key, progress)
            report.results.extend(failed)
            failed_repairs = [item for item in failed if item.get("target_id") in targets]
            report.incomplete |= bool(failed_repairs)
            if failed_repairs:
                self._remember_scope(progress, source, prefix, "discovery", "dependency_verification_incomplete",
                                     trigger_ids=triggers)
            if not failed_repairs and not task.get("gap_incomplete"):
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
                if discover or result.pending_ids:
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
    def _remember_scope(progress, source, span, stage, reason, trigger_ids=(), error=None):
        scope = {"source_id": source.id, "source_cutoff": source.source_order,
                 "span": span.model_dump(mode="json"), "stage": stage, "reason": reason,
                 "trigger_ids": sorted(trigger_ids)}
        if error is not None:
            scope.update(failure_details(error))
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
        receipt_key = f"graph:{token}"
        receipt = self.store.get_progress(receipt_key) or {"mapping": {}, "done": [], "claims": {}, "changed_ids": []}
        for key, value in receipt["claims"].items():
            claims[key] = type(claims[key]).model_validate(value)
        remaining = [dep for dep in graph.dependencies if dep.temporary_id not in receipt["done"]]
        staged = StagedWrite()
        mapping: dict[str, str] = dict(receipt["mapping"])
        # 精确复述直接复用已有版本；新的独立支持路径仍走验证，不能制造自依赖。
        visible = self.store.view(source_cutoff=cutoff)
        def same_statement(claim, version):
            unspecified = (claim.valid_time.start is None and claim.valid_time.end is None and claim.valid_time.text is None
                           and claim.valid_time.kind in {"unknown", version.valid_time.kind})
            return (" ".join(claim.content.split()) == " ".join(version.content.split())
                    and claim.modality == version.modality
                    and (unspecified or claim.valid_time == version.valid_time))
        for name, claim in claims.items():
            if name not in mapping:
                match = next((v for v in visible.versions.values() if same_statement(claim, v)
                              and self.evaluator.evaluate(v.id, source_cutoff=cutoff).usable), None)
                if match:
                    mapping[name] = match.id
        failed: list[dict] = []
        failed.extend(graph.issues)
        completed_targets: set[str] = set(mapping)
        changed_ids = list(receipt["changed_ids"])
        while remaining:
            ready_targets = []
            for dep in remaining:
                target_deps = [item for item in remaining if item.target_id == dep.target_id]
                if any(all(ref.id not in claims or ref.id in mapping for ref in item.premise_refs) for item in target_deps):
                    ready_targets.append(dep.target_id)
            if not ready_targets:
                failed.extend({"target_id": dep.target_id, "stage": "verify", "reason": "cyclic_or_rejected_premise"} for dep in remaining)
                break
            for target_id in list(dict.fromkeys(ready_targets))[:self.config.max_claims_per_call]:
                group = [dep for dep in remaining if dep.target_id == target_id
                         and all(ref.id not in claims or ref.id in mapping for ref in dep.premise_refs)]
                # 每条或路径独立就绪；失败的兄弟路径不阻挡已经完整的路径。
                group = group[:self.config.max_dependencies_per_call]
                consumed = {dep.temporary_id for dep in group}
                remaining = [dep for dep in remaining if dep.temporary_id not in consumed]
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
                    if target_id in claims and all(ref.type == "SOURCE" for ref in refs):
                        refs = self._bind_atomic_premises(refs, cutoff)
                        if not refs:
                            failed.append({"target_id": target_id, "stage": "grounding", "reason": "atomic_premise_unresolved"})
                            continue
                    resolved_dep = dep.model_copy(update={"premise_refs": refs})
                    real_target = mapping.get(target_id, target_id)
                    if any(ref.type == "CURRENT" and ref.id == real_target for ref in refs):
                        receipt["done"].append(dep.temporary_id)
                        continue
                    known = {dependency_signature(d) for d in self.store.dependencies()}
                    signature = dependency_signature(DependencyLink(namespace=self.store.namespace,
                        target_version_id=real_target, premise_refs=refs, effect=dep.effect, effective_time=dep.effective_time))
                    if signature in known:
                        receipt["done"].append(dep.temporary_id)
                        continue
                    premise_ids = [ref.id for ref in refs]
                    extra_context = {"target": claims[target_id].model_dump(mode="json") if target_id in claims
                                     else {**self.store.get_version(target_id).model_dump(mode="json", exclude={"namespace", "created_at", "status"}),
                                           "resolution": self.evaluator.evaluate(target_id, source_cutoff=cutoff).model_dump(mode="json")},
                                     "input_policy": policy.model_dump(mode="json")}
                    try:
                        self._check_premises(refs, self._context(premise_ids, cutoff, prefix, staged), cutoff, prefix, staged)
                    except ValueError as error:
                        failed.append({"target_id": target_id, "stage": "verify", "reason": str(error)})
                        continue
                    premise_evidence = []
                    for ref in refs:
                        if ref.type == "SOURCE":
                            source = self.store.get_source(ref.id)
                            text = self.evaluator.source_text(ref.id, source_cutoff=cutoff)[ref.span.start:ref.span.end]
                            premise_evidence.append({"reference": ref.model_dump(mode="json"), "text": text,
                                "speaker": source.speaker, "source_order": source.source_order})
                        else:
                            proof = self.evaluator.evidence(ref.id, ref.at_time if ref.type == "HISTORICAL" else None,
                                                            source_cutoff=cutoff, token_counter=self.llm.count_tokens)
                            premise_evidence.append({"reference": ref.model_dump(mode="json"),
                                "content": self.store.get_version(ref.id).content, "evidence": proof.text})
                    extra_context["premise_evidence"] = premise_evidence
                    try:
                        # 容量检查与实际发送使用同一份逐路径证据，不再预装无关历史。
                        verify_context = self._candidates([target_content], "validate", cutoff, prefix,
                                                           premise_ids + ([target_id] if target_id not in claims else []), staged,
                                                           request_builder=lambda value: (VERIFICATION_PROMPT, self.inducer.verification_payload(
                                                               {**value, **extra_context, "dependency": resolved_dep.model_dump(mode="json")})))
                    except RecoverableModelError as error:
                        failed.append({"target_id": target_id, "stage": "verify", **failure_details(error)})
                        continue
                    verify_context.update(extra_context)
                    try:
                        verification = self.inducer.verify(resolved_dep, verify_context)
                    except RecoverableModelError as error:
                        failed.append({"target_id": target_id, "stage": "verify", **failure_details(error)})
                        continue
                    if not verification.accepted:
                        failed.append({"target_id": target_id, "stage": "verify", "reason": verification.reason})
                        continue
                    if verification.revised_claim is not None:
                        if target_id not in claims or target_id in mapping:
                            failed.append({"target_id": target_id, "stage": "verify", "reason": "Cannot rewrite an already bound target"})
                            continue
                        if verified_links:
                            failed.append({"target_id": target_id, "stage": "verify", "reason": "Earlier path validated a different formulation"})
                            continue
                        revised = verification.revised_claim.model_copy(update={"temporary_id": target_id})
                        claims[target_id] = revised
                        target_content = revised.content
                    for path in verification.sufficient_paths:
                        if target_id in claims and all(refs[index].type == "SOURCE" for index in path):
                            failed.append({"target_id": target_id, "stage": "verify",
                                           "reason": "Verifier selected source-only support for a derived claim"})
                            continue
                        verified_links.append(DependencyLink(
                            id=_id("dep", token, dep.temporary_id, path), namespace=self.store.namespace,
                            target_version_id=target_id, premise_refs=[refs[index] for index in path],
                            effect=dep.effect, effective_time=dep.effective_time,
                        ))
                if not verified_links:
                    continue
                new_target = target_id in claims and target_id not in mapping
                if new_target:
                    claim = claims[target_id]
                    support_refs = next(link.premise_refs for link in verified_links if link.effect == "SUPPORT")
                    draft = FactDraft(content=claim.content, valid_time=claim.valid_time,
                                      modality=claim.modality, evidence_refs=support_refs,
                                      source_id=prefix.source_id)
                    try:
                        target_stage, decision = self._coordinate(draft, policy, cutoff, prefix,
                                                                   f"{token}:{target_id}", staged)
                    except RecoverableModelError as error:
                        failed.append({"target_id": target_id, "stage": "reconcile", **failure_details(error)})
                        continue
                    if decision.action in {"DEFER", "CONFLICT"}:
                        failed.append({"target_id": target_id, "stage": "reconcile", "reason": decision.reason})
                        continue
                    real_id = target_stage.target_id
                    mapping[target_id] = real_id
                    # _materialize 提供的单路径由逐条验证后的完整 OR 路径替换。
                    target_stage.dependencies = []
                else:
                    real_id = mapping.get(target_id, target_id)
                for link in verified_links:
                    target_stage.dependencies.append(link.model_copy(update={"target_version_id": real_id}))
                    if link.effect == "INVALIDATE":
                        target_stage.operations.append(ControlOperation(
                            id=_id("op", token, "known_unknown", real_id), namespace=self.store.namespace,
                            kind="resolve_pending", target_id=real_id, reason="UNKNOWN", source_cutoff=cutoff))
                affected = list(dict.fromkeys([real_id, *target_stage.changed_ids]))
                before = self._semantic_state(affected, cutoff)
                committed = False
                for link in target_stage.dependencies:
                    versions = target_stage.versions if not committed else []
                    if versions and versions[0].revision:
                        version = versions[0]
                        versions = [version.model_copy(update={"revision": version.revision.model_copy(
                            update={"evidence_refs": link.premise_refs})})]
                    try:
                        # 每条充分路径可独立提交，超深或成环的另一条不能使它回滚。
                        self.store.commit(versions, [link], target_stage.operations if not committed else [],
                            source_cutoff=cutoff, progress={receipt_key: {
                                "mapping": dict(mapping), "done": list(set(receipt["done"]) | consumed),
                                "claims": {key: value.model_dump(mode="json") for key, value in claims.items() if key in mapping},
                                "changed_ids": list(dict.fromkeys([*changed_ids, *affected]))}})
                        committed = True
                    except ValueError as error:
                        failed.append({"target_id": target_id, "stage": "commit", "reason": str(error)})
                if not committed:
                    if new_target:
                        mapping.pop(target_id, None)
                    continue
                after = self._semantic_state(affected, cutoff)
                changed_ids.extend(key for key in affected if before.get(key) != after.get(key))
                receipt = self.store.get_progress(receipt_key)
                receipt["changed_ids"] = list(dict.fromkeys(changed_ids))
                self.store.set_progress(receipt_key, receipt)
                completed_targets.add(target_id)
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
                staged.changed_ids.append(repair.target_id)
            else:
                failed.append({"target_id": repair.target_id, "stage": "repair", "reason": repair.reason})
        operation_map = {op.id: op for op in self.store.view(source_cutoff=cutoff).operations}
        for review in graph.control_reviews:
            operation = operation_map[review.operation_id]
            if review.decision == "DEFERRED":
                failed.append({"target_id": operation.target_id, "stage": "control_review", "reason": review.reason})
                continue
            if review.decision == "REVOKE" and policy.update_priority == "newer_source":
                view = self.store.view(source_cutoff=cutoff)
                def newest(refs):
                    leaves = list(refs)
                    for ref in refs:
                        if ref.type != "SOURCE":
                            leaves.extend(self.evaluator.evidence(ref.id, ref.at_time if ref.type == "HISTORICAL" else None,
                                                                  source_cutoff=cutoff).refs)
                    return max((view.sources[ref.id].source_order for ref in leaves if ref.type == "SOURCE"), default=-1)
                if (self.evaluator._evaluation(source_cutoff=cutoff).control_evidence_valid(operation)
                        and newest(review.evidence_refs) <= newest(operation.evidence_refs)):
                    failed.append({"target_id": operation.target_id, "stage": "control_review", "reason": "Older evidence cannot undo public source precedence"})
                    continue
            try:
                operation_data = operation.model_dump(mode="json", exclude={"namespace", "created_at"})
                review_context = self._candidates([review.reason], "validate", cutoff, prefix,
                                                   [operation.target_id, *[ref.id for ref in review.evidence_refs]], staged,
                                                   request_builder=lambda value: (CONTROL_VERIFICATION_PROMPT, self.inducer.verification_payload(
                                                       {**value, "input_policy": policy.model_dump(mode="json"),
                                                        "review": review.model_dump(mode="json"), "operation": operation_data})))
                # 控制复核也必须知道来源优先规则，否则会将合法更新误判为错误。
                review_context["input_policy"] = policy.model_dump(mode="json")
                self._check_premises(review.evidence_refs, review_context, cutoff, prefix, staged)
                verification = self.inducer.verify_control(review, operation_data, review_context)
            except RecoverableModelError as error:
                failed.append({"target_id": operation.target_id, "stage": "control_review", **failure_details(error)})
                continue
            except ValueError as error:
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
            pending = self.evaluator.evaluate(operation.target_id, source_cutoff=cutoff).reason == "maintenance_incomplete" if operation.scope == "version" else False
            if operation.scope == "version" and pending:
                staged.operations.append(ControlOperation(
                    id=_id("op", token, "control_reviewed", operation.id), namespace=self.store.namespace,
                    kind="resolve_pending", target_id=operation.target_id,
                    evidence_refs=refs, reason="control_reviewed", source_cutoff=cutoff,
                ))
            if review.decision == "REVOKE" or pending:
                staged.changed_ids.append(operation.target_id)
        if staged.dependencies or staged.operations:
            before = self._semantic_state(staged.changed_ids, cutoff)
            try:
                self.store.commit(staged.versions, staged.dependencies, staged.operations, source_cutoff=cutoff)
            except ValueError as error:
                # 全局环、深度、引用错误使整组回滚；不留下引用未提交前提的上层。
                failed.append({"stage": "commit", "reason": str(error)})
                return list(dict.fromkeys(changed_ids)), failed
            after = self._semantic_state(staged.changed_ids, cutoff)
            changed_ids.extend(key for key in staged.changed_ids if before.get(key) != after.get(key))
        receipt["mapping"] = mapping
        self.store.set_progress(receipt_key, receipt)
        return list(dict.fromkeys(changed_ids)), failed

    def _bind_atomic_premises(self, refs, cutoff):
        """仅绑定来源片段明确对应的唯一事实；数字前缀差异不再导致误拒绝。"""
        view = self.store.view(source_cutoff=cutoff)
        bound = []
        for ref in refs:
            matches = set()
            for dep in view.dependencies.values():
                if dep.effect != "SUPPORT" or len(dep.premise_refs) != 1:
                    continue
                saved = dep.premise_refs[0]
                if saved.type != "SOURCE" or saved.id != ref.id:
                    continue
                a, b = saved.span, ref.span
                contained = (a.start <= b.start < b.end <= a.end or b.start <= a.start < a.end <= b.end)
                if (contained and all(ctx in saved.context_refs for ctx in ref.context_refs)
                        and self.evaluator.evaluate(dep.target_version_id, source_cutoff=cutoff).usable):
                    matches.add(dep.target_version_id)
            if len(matches) != 1:
                return []
            bound.append(PremiseRef(type="CURRENT", id=next(iter(matches))))
        return bound
