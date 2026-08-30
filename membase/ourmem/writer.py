"""连接事实检索、协调、依据归纳和级联维护的写入流程。"""

from __future__ import annotations

from dataclasses import dataclass, field

from .inducer import JustificationInducer
from .maintenance import ClaimMemory, MaintenanceReport
from .models import (
    AtomicFact,
    ClaimKind,
    ClaimStatus,
    ClaimVersion,
    EvidenceQuote,
    FactUpdate,
    FactUpdateAction,
    Justification,
)
from .reconciler import FactReconciler
from .retriever import MemoryCandidateRetriever, SemanticCandidate
from .store import OurMemStore


@dataclass
class FactWriteResult:
    """一次事实写入产生的版本操作与派生主张变化。"""

    update: FactUpdate
    changed_claim_keys: list[str] = field(default_factory=list)
    induced_claim_keys: list[str] = field(default_factory=list)


class MemoryWriter:
    """执行一条新事实的端到端写入和局部维护。"""

    def __init__(
        self,
        fact_store: OurMemStore,
        claim_memory: ClaimMemory,
        retriever: MemoryCandidateRetriever,
        reconciler: FactReconciler,
        inducer: JustificationInducer,
        fact_candidate_k: int = 8,
        support_candidate_k: int = 12,
        claim_candidate_k: int = 8,
        fact_similarity_threshold: float = 0.45,
        claim_similarity_threshold: float = 0.25,
        max_induced_claims_per_fact: int = 4,
    ) -> None:
        self.fact_store = fact_store
        self.claim_memory = claim_memory
        self.retriever = retriever
        self.reconciler = reconciler
        self.inducer = inducer
        self.fact_candidate_k = fact_candidate_k
        self.support_candidate_k = support_candidate_k
        self.claim_candidate_k = claim_candidate_k
        self.fact_similarity_threshold = fact_similarity_threshold
        self.claim_similarity_threshold = claim_similarity_threshold
        self.max_induced_claims_per_fact = max_induced_claims_per_fact

    def write(
        self,
        evidence_quote: EvidenceQuote,
        new_fact: AtomicFact,
    ) -> FactWriteResult:
        """协调、执行并派生一条新原子事实（atomic fact）。"""

        active_facts = self.fact_store.get_active_facts()
        fact_matches = self.retriever.retrieve_facts(
            new_fact.content,
            active_facts,
            self.fact_candidate_k,
        )
        fact_by_id = {fact.id: fact for fact in active_facts}
        candidates = [
            fact_by_id[item.id]
            for item in fact_matches
            if item.score >= self.fact_similarity_threshold
        ]
        update = self.reconciler.reconcile(
            new_fact,
            evidence_quote,
            candidates,
        )

        if update.action is FactUpdateAction.DUPLICATE:
            return FactWriteResult(update=update)

        self.fact_store.add_evidence_quote(evidence_quote)
        result = FactWriteResult(update=update)
        if update.action is FactUpdateAction.ADD:
            self.fact_store.add_fact(new_fact)
            self._apply_new_defeaters(
                new_fact,
                evidence_quote,
                result,
                include_invalid_claims=False,
                hard_changed_version_id=None,
            )
        elif update.action is FactUpdateAction.SUPERSEDE:
            self.fact_store.supersede_fact(update.target_fact_id, new_fact)
            self._apply_new_defeaters(
                new_fact,
                evidence_quote,
                result,
                include_invalid_claims=True,
                hard_changed_version_id=update.target_fact_id,
            )
            report = self.claim_memory.on_version_changed(update.target_fact_id)
            result.changed_claim_keys.extend(report.changed_claim_keys)
        elif update.action is FactUpdateAction.RETRACT:
            report = self.claim_memory.retract_fact(
                update.target_fact_id,
                evidence_quote.id,
            )
            result.changed_claim_keys.extend(report.changed_claim_keys)

        if update.action in {FactUpdateAction.ADD, FactUpdateAction.SUPERSEDE}:
            self._induce_claims(new_fact, result)
        result.changed_claim_keys = list(dict.fromkeys(result.changed_claim_keys))
        return result

    def _apply_new_defeaters(
        self,
        new_fact: AtomicFact,
        evidence_quote: EvidenceQuote,
        result: FactWriteResult,
        include_invalid_claims: bool,
        hard_changed_version_id: str | None,
    ) -> None:
        current_claims = [
            claim
            for claim in self.claim_memory.get_current_claims()
            if include_invalid_claims or claim.status is ClaimStatus.VALID
            if hard_changed_version_id is None
            or hard_changed_version_id
            not in self.claim_memory.get_current_supporting_fact_ids(
                claim.claim_key
            )
        ]
        support_facts_by_claim = {
            claim.claim_key: [
                self.fact_store.get_fact(fact_id)
                for fact_id in self.claim_memory.get_current_supporting_fact_ids(
                    claim.claim_key
                )
            ]
            for claim in current_claims
        }
        matches = self.retriever.retrieve_items(
            new_fact.content,
            [
                (
                    claim.id,
                    "claim",
                    (
                        f"{claim.clause.value.proposition} Supported by: "
                        + "; ".join(
                            fact.content
                            for fact in support_facts_by_claim[claim.claim_key]
                        )
                    ),
                )
                for claim in current_claims
            ],
            self.claim_candidate_k,
        )
        claim_by_version_id = {claim.id: claim for claim in current_claims}
        candidates = [
            claim_by_version_id[item.id]
            for item in matches
            if item.score >= self.claim_similarity_threshold
        ]
        defeated_keys = self.inducer.detect_defeated_claim_keys(
            new_fact,
            evidence_quote,
            candidates,
            {
                claim.claim_key: support_facts_by_claim[claim.claim_key]
                for claim in candidates
            },
        )
        for claim_key in defeated_keys:
            current = self.claim_memory.get_current_justification(claim_key)
            replacement = Justification(
                conclusion_key=current.conclusion_key,
                conclusion_kind=current.conclusion_kind,
                support_version_ids=current.support_version_ids,
                explicit_defeater_version_ids=[
                    *current.explicit_defeater_version_ids,
                    new_fact.id,
                ],
                clause=current.clause,
            )
            report = self.claim_memory.replace_justification(
                current.id,
                replacement,
            )
            result.changed_claim_keys.extend(report.changed_claim_keys)

    def _induce_claims(
        self,
        new_fact: AtomicFact,
        result: FactWriteResult,
    ) -> None:
        queue: list[tuple[str, str, str, int]] = [
            (new_fact.id, "fact", new_fact.content, 0)
        ]
        while queue:
            if len(result.induced_claim_keys) >= self.max_induced_claims_per_fact:
                break
            anchor_id, anchor_kind, anchor_text, depth = queue.pop(0)
            if depth >= 2:
                continue
            support_candidates = self._support_candidates(
                anchor_id,
                anchor_kind,
                anchor_text,
            )
            existing_claims = [
                claim
                for claim in self.claim_memory.get_current_claims()
                if claim.status is ClaimStatus.VALID
            ]
            justifications = self.inducer.induce(
                anchor_id,
                support_candidates,
                self._existing_claim_candidates(anchor_text, existing_claims),
            )
            expected_kind = ClaimKind.STATE if depth == 0 else ClaimKind.DECISION
            for justification in justifications:
                if len(result.induced_claim_keys) >= self.max_induced_claims_per_fact:
                    break
                if justification.conclusion_kind is not expected_kind:
                    continue
                current = self.claim_memory.get_current_justification(
                    justification.conclusion_key
                )
                report = (
                    self.claim_memory.add_justification(justification)
                    if current is None
                    else self.claim_memory.replace_justification(
                        current.id,
                        justification,
                    )
                )
                result.changed_claim_keys.extend(report.changed_claim_keys)
                if justification.conclusion_key not in result.induced_claim_keys:
                    result.induced_claim_keys.append(justification.conclusion_key)
                claim = self.claim_memory.get_current_claim_version(
                    justification.conclusion_key
                )
                if (
                    report.changed_claim_keys
                    and claim.status is ClaimStatus.VALID
                ):
                    queue.append(
                        (
                            claim.id,
                            "claim",
                            claim.clause.value.proposition,
                            depth + 1,
                        )
                    )

    def _support_candidates(
        self,
        anchor_id: str,
        anchor_kind: str,
        anchor_text: str,
    ) -> list[SemanticCandidate]:
        active_facts = self.fact_store.get_active_facts()
        active_claims = [
            claim
            for claim in self.claim_memory.get_current_claims()
            if claim.status is ClaimStatus.VALID
        ]
        matches = self.retriever.retrieve_mixed(
            anchor_text,
            active_facts,
            active_claims,
            self.support_candidate_k,
        )
        claim_by_id = {claim.id: claim for claim in active_claims}
        candidates = []
        for item in matches:
            if item.id == anchor_id:
                continue
            if item.kind == "fact":
                candidates.append(item)
                continue
            claim = claim_by_id[item.id]
            support_text = "; ".join(
                self.fact_store.get_fact(fact_id).content
                for fact_id in self.claim_memory.get_current_supporting_fact_ids(
                    claim.claim_key
                )
            )
            candidates.append(
                SemanticCandidate(
                    id=item.id,
                    kind=item.kind,
                    text=(
                        f"{item.text} Supported by: {support_text}"
                    ),
                    score=item.score,
                )
            )
        return [
            SemanticCandidate(
                id=anchor_id,
                kind=anchor_kind,
                text=anchor_text,
                score=1.0,
            ),
            *candidates,
        ]

    def _existing_claim_candidates(
        self,
        query: str,
        claims: list[ClaimVersion],
    ) -> list[ClaimVersion]:
        matches = self.retriever.retrieve_claims(
            query,
            claims,
            self.claim_candidate_k,
        )
        claim_by_id = {claim.id: claim for claim in claims}
        return [claim_by_id[item.id] for item in matches]
