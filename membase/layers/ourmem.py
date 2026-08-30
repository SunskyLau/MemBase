from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from .base import MemBaseLayer
from ..configs.ourmem import OurMemConfig
from ..model_types.dataset import Message
from ..model_types.memory import MemoryEntry
from ..ourmem import OurMemSystem


class OurMemLayer(MemBaseLayer):
    """将 OurMemSystem 适配到 MemBase 的统一评测接口。"""

    layer_type: ClassVar[str] = "OurMem"

    def __init__(self, config: OurMemConfig) -> None:
        self.config = config
        self.system = self._new_system()
        self._current_session_id: str | None = None
        self._message_index = 0
        self._message_buffer: list[Message] = []

    def _new_system(self) -> OurMemSystem:
        return OurMemSystem(
            model_name=self.config.model_name,
            embedding_model_name=self.config.embedding_model_name,
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            fact_candidate_k=self.config.fact_candidate_k,
            support_candidate_k=self.config.support_candidate_k,
            claim_candidate_k=self.config.claim_candidate_k,
            fact_similarity_threshold=self.config.fact_similarity_threshold,
            claim_similarity_threshold=self.config.claim_similarity_threshold,
            embedding_batch_size=self.config.embedding_batch_size,
            max_induced_claims_per_fact=self.config.max_induced_claims_per_fact,
        )

    def add_message(self, message: Message, **kwargs: Any) -> None:
        session_id = kwargs.get("session_id")
        if session_id != self._current_session_id:
            self.flush()
            self._current_session_id = session_id
            self._message_index = 0
        self._message_buffer.append(message)
        if len(self._message_buffer) >= self.config.message_batch_size:
            self.flush()

    def add_messages(self, messages: list[Message], **kwargs: Any) -> None:
        for message in messages:
            self.add_message(message, **kwargs)

    def retrieve(self, query: str, k: int = 10, **kwargs: Any) -> list[MemoryEntry]:
        self.flush()
        return self.system.retrieve(query, k)

    def delete(self, memory_id: str) -> bool:
        fact = self.system.fact_store.get_fact(memory_id)
        self.system.claim_memory.retract_fact(fact.id)
        return True

    def update(self, memory_id: str, **kwargs: Any) -> bool:
        return False

    def save_memory(self) -> None:
        self.flush()
        self.system.save(self._memory_path())

    def flush(self) -> None:
        if not self._message_buffer:
            return
        self.system.ingest(
            self._message_buffer,
            session_id=self._current_session_id,
            message_offset=self._message_index,
        )
        self._message_index += len(self._message_buffer)
        self._message_buffer.clear()

    def load_memory(self, user_id: str | None = None) -> bool:
        user_id = user_id or self.config.user_id
        path = self._memory_path(user_id)
        if not path.exists():
            return False
        self.system = OurMemSystem.load(
            path=path,
            model_name=self.config.model_name,
            embedding_model_name=self.config.embedding_model_name,
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            fact_candidate_k=self.config.fact_candidate_k,
            support_candidate_k=self.config.support_candidate_k,
            claim_candidate_k=self.config.claim_candidate_k,
            fact_similarity_threshold=self.config.fact_similarity_threshold,
            claim_similarity_threshold=self.config.claim_similarity_threshold,
            embedding_batch_size=self.config.embedding_batch_size,
            max_induced_claims_per_fact=self.config.max_induced_claims_per_fact,
        )
        self._current_session_id = None
        self._message_index = 0
        return True

    def _memory_path(self, user_id: str | None = None) -> Path:
        return Path(self.config.save_dir) / f"{user_id or self.config.user_id}.json"
