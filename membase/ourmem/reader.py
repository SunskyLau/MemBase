"""联合检索原子事实与派生主张，并展开原始证据。"""

from __future__ import annotations

from ..model_types.memory import MemoryEntry
from .maintenance import ClaimMemory
from .models import AtomicFact, ClaimStatus, EvidenceQuote
from .retriever import MemoryCandidateRetriever
from .store import OurMemStore


class MemoryReader:
    """把当前有效事实和主张转换为 MemBase 检索结果。"""

    def __init__(
        self,
        fact_store: OurMemStore,
        claim_memory: ClaimMemory,
        retriever: MemoryCandidateRetriever,
    ) -> None:
        self.fact_store = fact_store
        self.claim_memory = claim_memory
        self.retriever = retriever

    def retrieve(self, query: str, k: int = 10) -> list[MemoryEntry]:
        active_facts = self.fact_store.get_active_facts()
        active_claims = [
            claim
            for claim in self.claim_memory.get_current_claims()
            if claim.status is ClaimStatus.VALID
        ]
        matches = self.retriever.retrieve_mixed(
            query,
            active_facts,
            active_claims,
            k,
        )
        fact_by_id = {fact.id: fact for fact in active_facts}
        claim_by_id = {claim.id: claim for claim in active_claims}

        entries = []
        for match in matches:
            if match.kind == "fact":
                fact = fact_by_id[match.id]
                quote = self.fact_store.get_evidence_quote(
                    fact.evidence_quote_id
                )
                entries.append(
                    MemoryEntry(
                        content=fact.content,
                        formatted_content=self._format_fact(fact, quote),
                        metadata={
                            "id": fact.id,
                            "kind": "fact",
                            "score": match.score,
                            "mention_time": fact.mention_time,
                            "event_time": fact.event_time,
                            "evidence_quote_ids": [quote.id],
                            "source_message_ids": [quote.message_id],
                        },
                    )
                )
                continue

            claim = claim_by_id[match.id]
            supporting_fact_ids = self.claim_memory.get_current_supporting_fact_ids(
                claim.claim_key
            )
            support_lines = []
            evidence_quote_ids = []
            source_message_ids = []
            for fact_id in supporting_fact_ids:
                fact = self.fact_store.get_fact(fact_id)
                quote = self.fact_store.get_evidence_quote(
                    fact.evidence_quote_id
                )
                support_lines.append(
                    "- " + self._format_fact(fact, quote).replace("\n", "\n  ")
                )
                evidence_quote_ids.append(quote.id)
                source_message_ids.append(quote.message_id)
            entries.append(
                MemoryEntry(
                    content=claim.clause.value.proposition,
                    formatted_content=(
                        f"Claim: {claim.clause.value.proposition}\n"
                        f"Polarity: {claim.clause.value.polarity.value}\n"
                        "Evidence:\n"
                        + "\n".join(support_lines)
                    ),
                    metadata={
                        "id": claim.id,
                        "claim_key": claim.claim_key,
                        "kind": "claim",
                        "polarity": claim.clause.value.polarity.value,
                        "score": match.score,
                        "supporting_fact_ids": supporting_fact_ids,
                        "evidence_quote_ids": evidence_quote_ids,
                        "source_message_ids": list(
                            dict.fromkeys(source_message_ids)
                        ),
                    },
                )
            )
        return entries

    @staticmethod
    def _format_fact(fact: AtomicFact, quote: EvidenceQuote) -> str:
        lines = [
            f"Fact: {fact.content}",
            f"Mention time: {fact.mention_time}",
        ]
        if fact.event_time is not None:
            lines.append(f"Event time: {fact.event_time}")
        lines.append(f"Source: {quote.quote}")
        return "\n".join(lines)
