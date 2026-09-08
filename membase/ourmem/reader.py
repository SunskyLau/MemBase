"""原始问题的一次混合检索与完整支持展开；不调用规划或判断模型。"""
from __future__ import annotations
from .evidence_format import render_evidence
from .models import PreparedContext
from .persistence import canonical_json


class MemoryReader:
    def __init__(self, store, retriever, evaluator, llm, config):
        self.store, self.retriever, self.evaluator = store, retriever, evaluator
        self.llm, self.config = llm, config

    def prepare(self, query: str, snapshot_id: int, query_time=None, top_k=None) -> PreparedContext:
        snapshot = self.store.snapshot(snapshot_id)
        view = self.store.view(snapshot)
        evaluation = self.evaluator._evaluation(snapshot, query_time)
        groups = self.retriever.search_memories(query, snapshot, query_time, top_k)
        bundles, roots, optional, omitted = [], [], [], []
        seen = set()
        policy = self.store.get_progress("system:input_policy") or {}
        def bundle_for(ref):
            return self.evaluator.evidence(ref.id, ref.at_time if ref.type == "HISTORICAL" else query_time,
                snapshot, token_counter=self.llm.count_tokens,
                bundle_cost=lambda part: self.llm.count_tokens(render_evidence([part], view, evaluation, policy)))
        def fits(parts):
            return self.llm.count_tokens(render_evidence(parts, view, evaluation, policy)) <= self.config.max_evidence_tokens
        for group in groups:
            # 当前值和最相关命中一起提供；不能把被命中的旧值包装成当前值。
            required = list({ref.id: ref for ref in [*group.members[:1],
                            *(r for r in group.members if r.type == "CURRENT")]}.values())
            parts = [bundle_for(ref) for ref in required if ref.id not in seen]
            if any(not part.complete for part in parts) or not fits([*bundles, *parts]):
                omitted.append({"memory_key": group.id, "reason": "complete_evidence_exceeds_budget"})
                continue
            bundles.extend(parts)
            roots.append(group.id)
            seen.update(ref.id for ref in required)
            optional.extend(ref for ref in group.members if ref.id not in seen)
        # 额外历史不抢占其他条目的必要证明预算。
        for ref in optional:
            if ref.id in seen:
                continue
            part = bundle_for(ref)
            if part.complete and fits([*bundles, part]):
                bundles.append(part)
                seen.add(ref.id)
            else:
                omitted.append({"version_id": ref.id, "reason": "optional_history_not_included"})
        context = render_evidence(bundles, view, evaluation, policy)
        if not bundles:
            context = "No usable memory evidence was retrieved. Do not invent missing facts."
        refs = {canonical_json(ref): ref for bundle in bundles for ref in bundle.refs}
        return PreparedContext(context=context, evidence_refs=list(refs.values()),
            resolution_status="incomplete" if not bundles or omitted else "not_assessed",
            reason="no_usable_evidence" if not bundles else "evidence_budget_limited" if omitted else "retrieval_only",
            coverage={"top_k": self.config.top_k if top_k is None else top_k,
                      "retrieved_memory_keys": [g.id for g in groups], "selected_memory_keys": roots,
                      "version_ids": sorted({v for bundle in bundles for v in bundle.version_ids}),
                      "omitted": omitted, "exhaustive": False, "answer_sufficiency_assessed": False,
                      "maintenance_incomplete": snapshot.maintenance_incomplete,
                      "unresolved_input_count": len(self.store.unresolved(snapshot)),
                      "evidence_tokens": self.llm.count_tokens(context)},
            read_trace=[{"phase": "retrieve", **self.retriever.last_trace},
                        {"phase": "expand", "memory_groups": len(roots), "proofs": len(bundles), "omitted": omitted}])
