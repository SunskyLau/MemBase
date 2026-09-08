"""串联 V5 各组件：来源先落盘，批次可恢复，快照只在同步维护后发布。"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from ..configs.ourmem import OurMemConfig
from .llm import ModelClient
from .maintenance import MaintenanceEngine
from .models import (AnswerResult, ControlOperation, InputMessage, InputPolicy,
                     MaintenanceReport, PremiseRef, PreparedContext, Source,
                     SourceSpan, TimePoint, new_id)
from .persistence import canonical_json
from .retriever import MemoryCandidateRetriever
from .store import OurMemStore
from .writer import MemoryWriter


@dataclass
class _Namespace:
    store: OurMemStore
    maintenance: MaintenanceEngine
    retriever: MemoryCandidateRetriever
    writer: MemoryWriter


def _opaque(prefix: str, *values: str) -> str:
    return prefix + "-" + sha256(canonical_json(values).encode()).hexdigest()[:24]


class OurMemSystem:
    """一个对象可管理多个独立样本；每个样本按调用顺序由单一写者处理。"""

    def __init__(self, config: OurMemConfig, storage_dir: str | Path | None = None,
                 client: ModelClient | None = None) -> None:
        self.config = config
        self.storage_dir = Path(storage_dir if storage_dir is not None else config.save_dir)
        self._owns_client = client is None
        self.client = client or ModelClient(config)
        self._namespaces: dict[str, _Namespace] = {}

    def database_path(self, namespace: str) -> Path:
        return self.storage_dir / f"{_opaque('memory', namespace)}.sqlite"

    def _state(self, namespace: str) -> _Namespace:
        if not namespace:
            raise ValueError("A namespace must not be empty")
        if namespace in self._namespaces:
            return self._namespaces[namespace]
        store = OurMemStore(self.database_path(namespace), namespace,
                            max_claim_depth=self.config.max_claim_depth)
        # 接口密钥不落盘；路径和调用者名称不改变记忆方法。
        config_data = self.config.model_dump(mode="json", exclude={"api_key", "user_id", "save_dir", "request_timeout", "transport_retry_window"})
        expected = {"sha256": sha256(canonical_json(config_data).encode()).hexdigest(), "config": config_data}
        prior = store.get_progress("system:configuration")
        if prior is not None and prior != expected:
            store.close()
            raise ValueError("Memory configuration changed; rebuild in a new storage directory")
        if prior is None:
            store.set_progress("system:configuration", expected)
        maintenance = MaintenanceEngine(store)
        retriever = MemoryCandidateRetriever(store, self.client.embed, self.config,
                                             maintenance=maintenance, token_counter=self.client.count_tokens)
        writer = MemoryWriter(store, retriever, maintenance, self.client, self.config)
        state = _Namespace(store, maintenance, retriever, writer)
        self._namespaces[namespace] = state
        return state

    def get_store(self, namespace: str) -> OurMemStore:
        """用于本地审计和受控接口；正常写入仍经 ingest/flush。"""
        return self._state(namespace).store

    @staticmethod
    def _progress(state: _Namespace) -> dict:
        return state.store.get_progress("system:ingest") or {"processed_through": -1, "pending_batch": None}

    def ingest(self, messages: list[InputMessage], namespace: str,
               input_policy: InputPolicy | None = None) -> list[MaintenanceReport]:
        state = self._state(namespace)
        self._resume_api_delete(state)
        old_policy = state.store.get_progress("system:input_policy")
        policy = InputPolicy.model_validate(input_policy or old_policy or {})
        policy_data = policy.model_dump(mode="json")
        if old_policy is not None and old_policy != policy_data:
            raise ValueError("A namespace cannot silently change its public input policy")
        if old_policy is None:
            state.store.set_progress("system:input_policy", policy_data)
        existing = {(s.conversation_id, s.message_id): s for s in state.store.sources()}
        next_order = state.store.source_cutoff + 1
        sources = []
        for value in messages:
            message = InputMessage.model_validate(value)
            conversation = _opaque("conversation", namespace, message.conversation_id or "")
            identity = _opaque("message", conversation, message.message_id)
            old = existing.get((conversation, identity))
            order = old.source_order if old else next_order
            if message.source_order is not None and message.source_order != order:
                raise ValueError("Source order must match stable ingestion order")
            source = Source(id=old.id if old else new_id("src"), namespace=namespace,
                            message_id=identity, conversation_id=conversation,
                            source_order=order, content=message.content, speaker=message.speaker,
                            role=message.role, mention_time=message.mention_time,
                            generation_refs=message.generation_refs)
            sources.append(source)
            if old is None:
                next_order += 1
                existing[(conversation, identity)] = source
        # 一次来源组完整写入；中途模型失败不会丢失任何原文。
        state.store.add_sources(sources)
        return self._drain(state, policy, state.store.source_cutoff, force=False)

    def _drain(self, state: _Namespace, policy: InputPolicy, cutoff: int, *, force: bool) -> list[MaintenanceReport]:
        reports = []
        progress = self._progress(state)
        while True:
            all_sources = {source.id: source for source in state.store.sources()}
            waiting = [s for s in all_sources.values() if progress["processed_through"] < s.source_order <= cutoff]
            if not waiting:
                return reports
            pending = progress["pending_batch"]
            if pending:
                batch = [all_sources[sid] for sid in pending["source_ids"]]
                if batch[-1].source_order > cutoff:
                    raise ValueError("Cannot split an unfinished transaction at a new observation point")
            else:
                batch, token_count = [], 0
                for source in waiting:
                    count = self.client.count_tokens(source.content)
                    if batch and (source.conversation_id != batch[0].conversation_id
                                  or len(batch) >= self.config.b_memory
                                  or token_count + count > self.config.max_batch_tokens):
                        break
                    batch.append(source)
                    token_count += count
                    if len(batch) >= self.config.b_memory or token_count >= self.config.max_batch_tokens:
                        break
                bounded = len(batch) < len(waiting) or len(batch) >= self.config.b_memory or token_count >= self.config.max_batch_tokens
                if not force and not bounded:
                    return reports
                pending = {"id": _opaque("batch", *[s.id for s in batch]), "source_ids": [s.id for s in batch]}
                progress["pending_batch"] = pending
                state.store.set_progress("system:ingest", progress)
            report = state.writer.process_batch(batch, policy, pending["id"])
            reports.append(report)
            progress = {"processed_through": batch[-1].source_order, "pending_batch": None}
            state.store.set_progress("system:ingest", progress)

    def _snapshot_token(self, namespace: str, snapshot_id: int) -> str:
        return f"{_opaque('snapshot', namespace)}:{snapshot_id}"

    def _snapshot_number(self, namespace: str, token: str) -> int:
        prefix, number = token.rsplit(":", 1)
        if prefix != _opaque("snapshot", namespace):
            raise ValueError("Snapshot belongs to a different namespace")
        return int(number)

    def flush(self, namespace: str, source_cutoff: int | None = None) -> str:
        state = self._state(namespace)
        self._resume_api_delete(state)
        cutoff = state.store.source_cutoff if source_cutoff is None else source_cutoff
        if cutoff < -1 or cutoff > state.store.source_cutoff:
            raise ValueError("Requested source cutoff is not visible")
        policy = InputPolicy.model_validate(state.store.get_progress("system:input_policy") or {})
        try:
            self._drain(state, policy, cutoff, force=True)
            versions = list(state.store.view(source_cutoff=cutoff).versions)
            report = state.maintenance.recompute(versions, source_cutoff=cutoff)
            state.retriever.sync(source_cutoff=cutoff)
        except Exception:
            state.store.set_progress("system:flush", {"source_cutoff": cutoff, "technical_failed": True})
            raise
        state.store.set_progress("system:flush", {"source_cutoff": cutoff, "technical_failed": False,
                                                   "report": {"incomplete": report.incomplete}})
        snapshot = state.store.publish(cutoff)
        return self._snapshot_token(namespace, snapshot.id)

    def prepare_evidence(self, query: str, namespace: str, snapshot_id: str,
                         query_time: str | None = None, *, top_k: int | None = None) -> PreparedContext:
        from .reader import MemoryReader
        state = self._state(namespace)
        number = self._snapshot_number(namespace, snapshot_id)
        state.store.snapshot(number)
        reader = MemoryReader(state.store, state.retriever, state.maintenance, self.client, self.config)
        return reader.prepare(query, number, query_time, top_k=top_k)

    def answer(self, query: str, namespace: str, snapshot_id: str,
               query_time: str | None = None, answer_template: str | None = None, *, top_k: int | None = None) -> AnswerResult:
        prepared = self.prepare_evidence(query, namespace, snapshot_id, query_time, top_k=top_k)
        if prepared.resolution_status == "deleted":
            return AnswerResult(**prepared.model_dump(), answer_text="I don't have that information.")
        template = answer_template or (
            "Answer the question concisely using the allowed evidence below. Respect CURRENT and HISTORICAL "
            "labels, explicit corrections, uncertainty and unavailable evidence. Do not invent user facts or "
            "repeat deleted information. If required evidence is missing, say so.\n\n"
            "Evidence:\n{context}\n\nQuestion: {question}\nAnswer:")
        answer = self.client.text(template.format(context=prepared.context, question=query),
                                  stage="answer", model=self.config.answer_model, temperature=0)
        return AnswerResult(**prepared.model_dump(), answer_text=answer)

    def get_memory_snapshot(self, namespace: str, snapshot_id: str) -> dict:
        state = self._state(namespace)
        number = self._snapshot_number(namespace, snapshot_id)
        snapshot = state.store.snapshot(number)
        view = state.store.view(snapshot)
        evaluation = state.maintenance._evaluation(snapshot)
        lines, current, history = [], [], []
        for version in view.versions.values():
            if not evaluation.content_visible(version.id):
                continue
            resolution = evaluation.evaluate(version.id)
            label = "CURRENT" if resolution.usable else "NOT_CURRENT"
            support_sources = sorted({ref.id for dep in evaluation.supports[version.id] for ref in dep.premise_refs})
            text = (f"{label} {version.id} ({resolution.reason or resolution.status.value}; "
                    f"modality={version.modality}; direct premise ids={support_sources}; "
                    f"time={canonical_json(version.valid_time)}): {version.content}")
            (current if resolution.usable else history).append(text)
        lines += ["CURRENT MEMORY", *current, "HISTORICAL OR UNRESOLVED RECORDS (not unconditional current facts)", *history]
        lines.append("ORIGINAL UTTERANCES (consult their associated version/control status)")
        for source in state.store.sources(snapshot):
            chars = list(source.content)
            for start, end in evaluation.source_ranges(source.id, include_retracted=False):
                chars[start:end] = [" "] * (end - start)
            content = "".join(chars)
            if content.strip():
                lines.append(canonical_json({"source_id": source.id, "speaker": source.speaker,
                                             "role": source.role, "source_order": source.source_order,
                                             "mention_time": source.mention_time, "content": content}))
        lines.append("CONTROL RECORDS (do not treat closed versions as current facts)")
        for operation in evaluation.operations:
            lines.append(canonical_json({"kind": operation.kind, "target_id": operation.target_id,
                                         "scope": operation.scope, "replacement_id": operation.replacement_id,
                                         "effective_time": operation.effective_time.model_dump() if operation.effective_time else None}))
        return {"text": "\n".join(lines), "snapshot_id": snapshot_id,
                "source_cutoff": snapshot.source_cutoff, "sequence": snapshot.sequence,
                "maintenance_incomplete": snapshot.maintenance_incomplete,
                "database": str(self.database_path(namespace))}

    def delete(self, memory_id: str, namespace: str) -> str:
        """可信程序接口按指定版本删除；操作来源不伪装成用户说过的事实。"""
        self.flush(namespace)
        state = self._state(namespace)
        state.store.get_version(memory_id)
        if state.maintenance.evaluate(memory_id).status == "deleted":
            return self.flush(namespace)
        source = Source(namespace=namespace, message_id=new_id("control"),
                        conversation_id="memory_control_api", speaker="memory_control_api", role="system",
                        content="Delete the requested memory record.", source_order=state.store.source_cutoff + 1)
        reference = PremiseRef(type="SOURCE", id=source.id,
                               span=SourceSpan(source_id=source.id, start=0, end=len(source.content)))
        operation = ControlOperation(id=_opaque("delete", memory_id), namespace=namespace,
                                     kind="delete", scope="version", target_id=memory_id,
                                     evidence_refs=[reference], source_cutoff=source.source_order,
                                     effective_time=TimePoint(order=source.source_order), reason="trusted_api_request")
        # 先记录意图；中断发生在来源写入与控制提交之间也可准确恢复。
        state.store.set_progress("system:api_delete", {"source": source.model_dump(mode="json"),
                                 "operation": operation.model_dump(mode="json"), "complete": False})
        return self.flush(namespace)

    def _resume_api_delete(self, state: _Namespace) -> None:
        progress = state.store.get_progress("system:api_delete")
        if progress is None or progress["complete"]:
            return
        source = Source.model_validate(progress["source"])
        operation = ControlOperation.model_validate(progress["operation"])
        state.store.add_sources([source])
        state.store.commit(operations=[operation], source_cutoff=source.source_order,
                           progress={"system:ingest": {"processed_through": source.source_order, "pending_batch": None}})
        policy = InputPolicy.model_validate(state.store.get_progress("system:input_policy") or {})
        state.writer.maintain_change([operation.target_id, source.id], source, policy, _opaque("maintenance", source.id))
        state.store.set_progress("system:api_delete", {**progress, "complete": True})

    def close(self) -> None:
        for state in self._namespaces.values():
            state.store.close()
        self._namespaces.clear()
        if self._owns_client:
            self.client.budget.close()
