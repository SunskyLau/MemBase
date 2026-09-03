"""OurMem 第一版端到端记忆系统。"""

from __future__ import annotations

from pathlib import Path

from ..model_types.dataset import Message
from ..model_types.memory import MemoryEntry
from .extractor import FactExtractor
from .inducer import JustificationInducer
from .maintenance import ClaimMemory
from .models import AtomicFact
from .persistence import OurMemPersistence
from .reader import MemoryReader
from .reconciler import FactReconciler
from .retriever import MemoryCandidateRetriever, OpenAIEmbedder
from .store import OurMemStore
from .writer import FactWriteResult, MemoryWriter


class OurMemSystem:
    """组合事实抽取、更新、派生、持久化和联合检索。"""

    def __init__(
        self,
        model_name: str,
        embedding_model_name: str,
        api_key: str,
        base_url: str,
        fact_candidate_k: int = 8,
        support_candidate_k: int = 12,
        claim_candidate_k: int = 8,
        fact_similarity_threshold: float = 0.45,
        claim_similarity_threshold: float = 0.25,
        embedding_batch_size: int = 128,
        max_induced_claims_per_fact: int = 4,
        fact_store: OurMemStore | None = None,
        claim_memory: ClaimMemory | None = None,
    ) -> None:
        self.fact_store = fact_store or OurMemStore()
        self.claim_memory = claim_memory or ClaimMemory(self.fact_store)
        self.extractor = FactExtractor(
            model_name=model_name,
            api_keys=api_key,
            base_urls=base_url,
        )
        self.reconciler = FactReconciler(
            model_name=model_name,
            api_keys=api_key,
            base_urls=base_url,
        )
        self.inducer = JustificationInducer(
            model_name=model_name,
            api_keys=api_key,
            base_urls=base_url,
        )
        self.retriever = MemoryCandidateRetriever(
            OpenAIEmbedder(
                model_name=embedding_model_name,
                api_key=api_key,
                base_url=base_url,
                batch_size=embedding_batch_size,
            )
        )
        self.writer = MemoryWriter(
            self.fact_store,
            self.claim_memory,
            self.retriever,
            self.reconciler,
            self.inducer,
            fact_candidate_k=fact_candidate_k,
            support_candidate_k=support_candidate_k,
            claim_candidate_k=claim_candidate_k,
            fact_similarity_threshold=fact_similarity_threshold,
            claim_similarity_threshold=claim_similarity_threshold,
            max_induced_claims_per_fact=max_induced_claims_per_fact,
        )
        self.reader = MemoryReader(
            self.fact_store,
            self.claim_memory,
            self.retriever,
        )

    def ingest(
        self,
        messages: list[Message],
        session_id: str | None = None,
        message_offset: int = 0,
    ) -> list[FactWriteResult]:
        """从消息中抽取事实并依次写入记忆。"""

        extracted = self.extractor.extract(
            messages,
            session_id=session_id,
            message_offset=message_offset,
        )
        return [self.writer.write(fact) for fact in extracted]

    def write_fact(
        self,
        fact: AtomicFact,
    ) -> FactWriteResult:
        """写入已经抽取好的原子事实（atomic fact）。"""

        return self.writer.write(fact)

    def retrieve(self, query: str, k: int = 10) -> list[MemoryEntry]:
        """联合检索事实和派生主张（derived claim）。"""

        return self.reader.retrieve(query, k)

    def save(self, path: str | Path) -> None:
        """保存全部权威记忆状态；嵌入缓存将在加载后重建。"""

        OurMemPersistence.save(path, self.fact_store, self.claim_memory)

    @classmethod
    def load(
        cls,
        path: str | Path,
        model_name: str,
        embedding_model_name: str,
        api_key: str,
        base_url: str,
        fact_candidate_k: int = 8,
        support_candidate_k: int = 12,
        claim_candidate_k: int = 8,
        fact_similarity_threshold: float = 0.45,
        claim_similarity_threshold: float = 0.25,
        embedding_batch_size: int = 128,
        max_induced_claims_per_fact: int = 4,
    ) -> OurMemSystem:
        """恢复记忆状态并重新连接推理与嵌入接口。"""

        fact_store, claim_memory = OurMemPersistence.load(path)
        return cls(
            model_name=model_name,
            embedding_model_name=embedding_model_name,
            api_key=api_key,
            base_url=base_url,
            fact_candidate_k=fact_candidate_k,
            support_candidate_k=support_candidate_k,
            claim_candidate_k=claim_candidate_k,
            fact_similarity_threshold=fact_similarity_threshold,
            claim_similarity_threshold=claim_similarity_threshold,
            embedding_batch_size=embedding_batch_size,
            max_induced_claims_per_fact=max_induced_claims_per_fact,
            fact_store=fact_store,
            claim_memory=claim_memory,
        )
