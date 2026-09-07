"""统一候选检索：先过滤可见内容，再进行精确向量、BM25 与排名融合。"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
import re
from typing import Callable

import numpy as np

from .maintenance import MaintenanceEngine, _Evaluation, compare_time
from .models import MemoryVersion, Snapshot, Source, SourceSpan, TimePoint
from .store import OurMemStore
from .tokenization import count_tokens, get_encoder


@dataclass(frozen=True)
class RetrievalCandidate:
    id: str
    kind: str
    text: str
    score: float = 0.0
    record: Source | MemoryVersion | None = None
    source_id: str | None = None
    span: SourceSpan | None = None
    mandatory: bool = False


def _tokens(text: str) -> list[str]:
    # 保留单字符、数字和标识符；中文按字检索，不要求另建实体词表。
    return re.findall(r"[\u3400-\u9fff]|[^\W_]+(?:[_'-][^\W_]+)*", text.lower())


class MemoryCandidateRetriever:
    def __init__(self, store: OurMemStore, embedder: Callable, config, *,
                 maintenance: MaintenanceEngine | None = None, token_counter=None) -> None:
        self.store, self.embedder, self.config = store, embedder, config
        self.maintenance = maintenance or MaintenanceEngine(store)
        self.model_name = config.embedding_model_name
        self._query_cache: dict[str, np.ndarray] = {}
        self._token_counter = token_counter
        self.last_trace: dict = {}
        self._index_key = None
        self._index = None
        self._record_vectors = {}

    def _count(self, text: str) -> int:
        if self._token_counter:
            return self._token_counter(text)
        return count_tokens(text)

    def source_chunks(self, source: Source, content: str | None = None) -> list[RetrievalCandidate]:
        """切片仍使用完整消息的字符位置；不会因抽取遗漏而丢掉原文。"""
        content = source.content if content is None else content
        presented_source = source.model_copy(update={"content": content})
        encoder = get_encoder()
        encoded = encoder.encode(content, disallowed_special=())
        if not encoded:
            return []
        _, offsets = encoder.decode_with_offsets(encoded)
        limit = self.config.max_source_chunk_tokens
        chunks = []
        start_token = 0
        while start_token < len(encoded):
            end_token = min(start_token + limit, len(encoded))
            start = offsets[start_token]
            end = offsets[end_token] if end_token < len(encoded) else len(content)
            while end_token > start_token + 1 and self._count(content[start:end]) > limit:
                end_token -= 1
                end = offsets[end_token]
            # 极小预算可能装不下一个Unicode字符，不能输出空片段然后声称覆盖。
            if end <= start or self._count(content[start:end]) > limit:
                raise ValueError("Source chunk token budget cannot fit a complete character")
            if content[start:end].strip():
                span = SourceSpan(source_id=source.id, start=start, end=end)
                chunks.append(RetrievalCandidate(id=f"{source.id}:{start}:{end}", kind="source",
                                                text=content[start:end], record=presented_source,
                                                source_id=source.id, span=span))
            start_token = end_token
        return chunks

    def _items(self, mode: str, evaluation: _Evaluation) -> list[RetrievalCandidate]:
        items = []
        if mode != "source":
            for version in evaluation.view.versions.values():
                if not evaluation.content_visible(version.id):
                    continue
                state = evaluation.evaluate(version.id)
                if mode in {"derive", "read", "aggregate"} and state.status in {"superseded", "deleted"}:
                    continue
                if mode in {"derive", "read", "aggregate"} and state.reason in {"expired", "not_yet_applicable"}:
                    continue
                items.append(RetrievalCandidate(id=version.id, kind="memory", text=version.content, record=version))
        for source in sorted(evaluation.view.sources.values(), key=lambda item: item.source_order):
            chars = list(source.content)
            for start, end in evaluation.source_ranges(source.id, include_retracted=False):
                chars[start:end] = [" "] * (end - start)
            items.extend(self.source_chunks(source, "".join(chars)))
        return items

    @staticmethod
    def _normalized(vector) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float32)
        if vector.ndim != 1 or not np.isfinite(vector).all():
            raise ValueError("Embedding result must be a finite vector")
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    def _prepare_vectors(self, items: list[RetrievalCandidate], queries: list[str] | None = None) -> tuple[np.ndarray, list[np.ndarray]]:
        """候选与查询合并补算；只有真实记忆记录的向量写入数据库。"""
        queries = queries or []
        vectors, records = {}, []
        by_text = dict(self._query_cache)
        for item in items:
            digest = sha256(item.text.encode()).hexdigest()
            vector = self._record_vectors.get((item.id, digest))
            if vector is None:
                vector = self.store.get_embedding(item.id, self.model_name, digest)
            records.append((item, digest, vector))
            if vector is not None:
                self._record_vectors[item.id, digest] = vector
                vectors[item.id] = vector
                by_text.setdefault(item.text, vector)
        missing_texts = list(dict.fromkeys(
            [item.text for item, _, vector in records if vector is None and item.text not in by_text]
            + [query for query in queries if query not in by_text]
        ))
        if missing_texts:
            # ModelClient负责按embedding_batch_size切真实请求，统一逐次占用额度。
            generated = self.embedder(missing_texts)
            if len(generated) != len(missing_texts):
                raise ValueError("Embedding response omitted requested texts")
            by_text.update((text, self._normalized(vector)) for text, vector in zip(missing_texts, generated))
        for item, digest, vector in records:
            if vector is None:
                vector = by_text[item.text]
                self.store.put_embedding(item.id, self.model_name, digest, vector)
                vectors[item.id] = vector
                self._record_vectors[item.id, digest] = vector
        for query in queries:
            self._query_cache.setdefault(query, by_text[query])
        return np.stack([vectors[item.id] for item in items]), [self._query_cache[query] for query in queries]

    def sync(self, snapshot=None, source_cutoff=None) -> None:
        """发布前确保当前来源与记录都有对应向量；检索本身不依赖外部服务索引。"""
        evaluation = _Evaluation(self.store.view(snapshot, source_cutoff=source_cutoff))
        items = self._items("reconcile", evaluation)
        if items:
            self._prepare_vectors(items)

    def retrieve(self, queries: list[str] | str, mode: str = "derive", mandatory_context=None,
                 snapshot: Snapshot | int | None = None, query_time: TimePoint | str | None = None,
                 source_cutoff: int | None = None, source_span_cutoff: SourceSpan | None = None,
                 budget: int | None = None) -> list[RetrievalCandidate]:
        import bm25s

        queries = [queries] if isinstance(queries, str) else list(dict.fromkeys(queries))
        if mode not in self.config.max_candidates:
            raise ValueError(f"Unknown retrieval mode: {mode}")
        evaluation = _Evaluation(self.store.view(snapshot, source_cutoff=source_cutoff), query_time, source_span_cutoff)
        index_key = (min(evaluation.view.sequence, self.store.data_seq), self.store.data_seq,
                     evaluation.view.source_cutoff, str(evaluation.now), str(source_span_cutoff), mode)
        if self._index_key == index_key:
            items, sparse, corpus = self._index
        else:
            items = self._items(mode, evaluation)
            corpus = [_tokens(item.text) for item in items]
            sparse = bm25s.BM25(method="lucene", idf_method="lucene", k1=1.5, b=0.75, backend="numpy")
            if items:
                sparse.index(corpus, show_progress=False)
            self._index_key, self._index = index_key, (items, sparse, corpus)
        by_id = {item.id: item for item in items}
        mandatory = []
        for record in mandatory_context or []:
            record_id = record if isinstance(record, str) else record.id
            if record_id in evaluation.view.versions:
                if evaluation.content_visible(record_id):
                    value = evaluation.view.versions[record_id]
                    mandatory.append(RetrievalCandidate(id=value.id, kind="memory", text=value.content, record=value, mandatory=True))
            elif record_id in evaluation.view.sources:
                mandatory.extend(replace(item, mandatory=True) for item in items if item.source_id == record_id)
            elif record_id in by_id:
                mandatory.append(replace(by_id[record_id], mandatory=True))
        if not items or not queries:
            self.last_trace = {"mode": mode, "queries": queries, "eligible": len(items), "returned": len(mandatory)}
            return mandatory
        matrix, vectors = (self._prepare_vectors(items, queries) if self.config.k_dense[mode]
                           else (None, [None] * len(queries)))
        scores: dict[str, float] = defaultdict(float)
        per_query: list[list[str]] = []
        for query, vector in zip(queries, vectors):
            ranks: dict[str, float] = defaultdict(float)
            if matrix is not None:
                similarities = matrix @ vector
                indices = sorted(range(len(items)), key=lambda i: (-float(similarities[i]), items[i].id))[:self.config.k_dense[mode]]
                for rank, index in enumerate(indices, 1):
                    ranks[items[index].id] += 1.0 / (self.config.rrf_c + rank)
            sparse_k = min(self.config.k_bm25[mode], len(items))
            tokens = _tokens(query)
            if sparse_k and tokens:
                indices, values = sparse.retrieve([tokens], k=sparse_k, show_progress=False)
                for rank, (index, value) in enumerate(zip(indices[0], values[0]), 1):
                    if value > 0:
                        ranks[items[int(index)].id] += 1.0 / (self.config.rrf_c + rank)
            # 时间候选来自词义已命中且有时间轴的内容，可补回被语义Top-k挤出的相邻记录。
            timed = []
            for index, item in enumerate(items):
                if not set(tokens).intersection(corpus[index]):
                    continue
                if item.kind == "memory":
                    point = item.record.valid_time.start
                else:
                    point = TimePoint(order=item.record.source_order)
                    if item.record.mention_time:
                        try:
                            point = TimePoint(date=item.record.mention_time, order=item.record.source_order)
                        except ValueError:
                            pass
                if point is None:
                    continue
                distance = None
                if point.date is not None and evaluation.now.date is not None:
                    def timestamp(text):
                        value = datetime.fromisoformat(text.replace("Z", "+00:00"))
                        return value.replace(tzinfo=timezone.utc).timestamp() if value.tzinfo is None else value.timestamp()
                    distance = abs(timestamp(point.date) - timestamp(evaluation.now.date))
                elif point.order is not None and evaluation.now.order is not None:
                    distance = abs(point.order - evaluation.now.order)
                if distance is not None:
                    timed.append((distance, -ranks.get(item.id, 0), item.id))
            timed.sort()
            for rank, (_, _, item_id) in enumerate(timed[:self.config.k_time[mode]], 1):
                ranks[item_id] += 1.0 / (self.config.rrf_c + rank)
            per_query.append(sorted(ranks, key=lambda item_id: (-ranks[item_id], item_id)))
            for item_id, value in ranks.items():
                scores[item_id] += value
        limit = self.config.max_candidates[mode]
        if budget is not None:
            limit = min(limit, budget)
        selected = list(dict.fromkeys(item.id for item in mandatory))
        semantic = []
        # 每个需求先拿一个结果，再按融合分数补齐，避免一个查询吞掉全部候选页。
        for ranking in per_query:
            candidate = next((item_id for item_id in ranking if item_id not in selected and item_id not in semantic), None)
            if candidate and len(semantic) < limit:
                semantic.append(candidate)
        for item_id in sorted(scores, key=lambda value: (-scores[value], value)):
            if len(semantic) >= limit:
                break
            if item_id not in selected and item_id not in semantic:
                semantic.append(item_id)
        result = list({item.id: item for item in mandatory}.values())
        result.extend(replace(by_id[item_id], score=scores[item_id]) for item_id in semantic)
        self.last_trace = {"mode": mode, "queries": queries, "eligible": len(items),
                           "returned": len(result), "mandatory": len(selected),
                           "source_tokens": sum(self._count(item.text) for item in result if item.kind == "source")}
        return result

    def source_page(self, cursor: int = 0, page_tokens: int | None = None, *, snapshot=None,
                    source_cutoff=None, conversation_id: str | None = None,
                    start_time: TimePoint | None = None, end_time: TimePoint | None = None) -> tuple[list[RetrievalCandidate], int | None]:
        """真实来源分页；游标遍历完整范围，与 Top-k 和语义查询轮数无关。"""
        evaluation = _Evaluation(self.store.view(snapshot, source_cutoff=source_cutoff))
        chunks = self._items("source", evaluation)
        chunks = [item for item in chunks if conversation_id is None or item.record.conversation_id == conversation_id]
        if start_time or end_time:
            def in_range(item):
                point = TimePoint(order=item.record.source_order)
                if item.record.mention_time:
                    try:
                        point = TimePoint(date=item.record.mention_time, order=item.record.source_order)
                    except ValueError:
                        pass
                left, right = compare_time(point, start_time), compare_time(point, end_time)
                return not (left is not None and left < 0 or right is not None and right >= 0)
            chunks = [item for item in chunks if in_range(item)]
        limit = page_tokens if page_tokens is not None else self.config.max_context_tokens
        page, consumed = [], 0
        index = cursor
        while index < len(chunks):
            size = self._count(chunks[index].text)
            if consumed + size > limit:
                if not page:
                    raise ValueError("Source page budget cannot fit its next complete chunk")
                break
            page.append(chunks[index])
            consumed += size
            index += 1
        return page, index if index < len(chunks) else None
