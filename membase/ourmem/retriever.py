"""OurMem 的轻量语义候选检索（semantic candidate retrieval）。"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Callable

from openai import OpenAI

from .models import AtomicFact, ClaimVersion


@dataclass(frozen=True)
class SemanticCandidate:
    """一条带相似度分数的事实或派生主张候选。"""

    id: str
    kind: str
    text: str
    score: float


class OpenAIEmbedder:
    """调用 OpenAI 兼容接口生成文本嵌入（embedding）。"""

    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: str,
        batch_size: int = 128,
    ) -> None:
        self.model_name = model_name
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self.batch_size = batch_size

    def __call__(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        for start in range(0, len(texts), self.batch_size):
            response = self._client.embeddings.create(
                model=self.model_name,
                input=texts[start : start + self.batch_size],
            )
            embeddings.extend(
                item.embedding
                for item in sorted(response.data, key=lambda item: item.index)
            )
        return embeddings


class MemoryCandidateRetriever:
    """在当前事实和主张集合上执行内存语义检索。"""

    def __init__(
        self,
        embedder: Callable[[list[str]], list[list[float]]],
    ) -> None:
        self._embedder = embedder
        self._embedding_cache: dict[str, tuple[str, list[float]]] = {}

    def retrieve_facts(
        self,
        query: str,
        facts: list[AtomicFact],
        k: int,
    ) -> list[SemanticCandidate]:
        return self._retrieve(
            query,
            [(fact.id, "fact", fact.content) for fact in facts],
            k,
        )

    def retrieve_claims(
        self,
        query: str,
        claims: list[ClaimVersion],
        k: int,
    ) -> list[SemanticCandidate]:
        return self._retrieve(
            query,
            [
                (claim.id, "claim", claim.clause.value.proposition)
                for claim in claims
            ],
            k,
        )

    def retrieve_mixed(
        self,
        query: str,
        facts: list[AtomicFact],
        claims: list[ClaimVersion],
        k: int,
    ) -> list[SemanticCandidate]:
        items = [(fact.id, "fact", fact.content) for fact in facts]
        items.extend(
            (claim.id, "claim", claim.clause.value.proposition)
            for claim in claims
        )
        return self._retrieve(query, items, k)

    def retrieve_items(
        self,
        query: str,
        items: list[tuple[str, str, str]],
        k: int,
    ) -> list[SemanticCandidate]:
        """检索调用方构造的带标识文本条目。"""

        return self._retrieve(query, items, k)

    def _retrieve(
        self,
        query: str,
        items: list[tuple[str, str, str]],
        k: int,
    ) -> list[SemanticCandidate]:
        if not items:
            return []

        missing = [
            (item_id, text)
            for item_id, _, text in items
            if self._embedding_cache.get(item_id, (None, None))[0] != text
        ]
        if missing:
            vectors = self._embedder([text for _, text in missing])
            for (item_id, text), vector in zip(missing, vectors):
                self._embedding_cache[item_id] = (text, vector)

        query_vector = self._embedder([query])[0]
        candidates = [
            SemanticCandidate(
                id=item_id,
                kind=kind,
                text=text,
                score=self._cosine(
                    query_vector,
                    self._embedding_cache[item_id][1],
                ),
            )
            for item_id, kind, text in items
        ]
        candidates.sort(key=lambda item: (-item.score, item.id))
        return candidates[:k]

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        dot_product = sum(a * b for a, b in zip(left, right))
        left_norm = sqrt(sum(value * value for value in left))
        right_norm = sqrt(sum(value * value for value in right))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return dot_product / (left_norm * right_norm)
