"""MemBase 的薄适配层；写入、控制与读取语义均由 OurMemSystem 负责。"""

from __future__ import annotations

from typing import Any, ClassVar

from .base import MemBaseLayer
from ..configs.ourmem import OurMemConfig
from ..model_types.dataset import Message
from ..model_types.memory import MemoryEntry
from ..ourmem.models import InputMessage, InputPolicy
from ..ourmem.system import OurMemSystem, _opaque


class OurMemLayer(MemBaseLayer):
    layer_type: ClassVar[str] = "OurMem"

    def __init__(self, config: OurMemConfig, client=None) -> None:
        self.config = config
        self.system = OurMemSystem(config=config, storage_dir=config.save_dir, client=client)

    @staticmethod
    def _message(message: Message, session_id: str | None) -> InputMessage:
        # metadata 不整体转发；图像描述是唯一明确允许的附加输入。
        text = message.content
        if message.metadata.get("blip_caption"):
            text += f"\n[Image caption: {message.metadata['blip_caption']}]"
        return InputMessage(message_id=message.id, conversation_id=session_id,
                            content=text, speaker=message.name, role=message.role,
                            mention_time=message.timestamp, source_order=message.source_order)

    def add_message(self, message: Message, **kwargs: Any) -> None:
        self.add_messages([message], **kwargs)

    def add_messages(self, messages: list[Message], **kwargs: Any) -> None:
        policy = InputPolicy.model_validate(kwargs["input_policy"]) if kwargs.get("input_policy") is not None else None
        self.system.ingest([self._message(m, kwargs.get("session_id")) for m in messages],
                           namespace=self.config.user_id, input_policy=policy)

    def flush(self) -> str:
        return self.system.flush(namespace=self.config.user_id)

    def get_memory_snapshot(self, snapshot_id: str) -> dict:
        return self.system.get_memory_snapshot(self.config.user_id, snapshot_id)

    @classmethod
    def from_config(cls, config, *, client=None):
        return cls(config, client=client)

    def retrieve(self, query: str, k: int | None = None, **kwargs: Any) -> list[MemoryEntry]:
        if k is not None and k < 1:
            raise ValueError("k must be positive")
        snapshot_id = kwargs.get("snapshot_id") or self.flush()
        prepared = self.system.prepare_evidence(query, namespace=self.config.user_id,
                                                snapshot_id=snapshot_id,
                                                query_time=kwargs.get("query_time"), top_k=k)
        # k 限制入口记忆条目，必要支持路径作为完整上下文返回。
        return [MemoryEntry(content=prepared.context, formatted_content=prepared.context,
                            metadata={"snapshot_id": snapshot_id,
                                      **prepared.model_dump(mode="json", exclude={"context"})})]

    def delete(self, memory_id: str) -> bool:
        self.system.delete(memory_id, namespace=self.config.user_id)
        return True

    def update(self, memory_id: str, **kwargs: Any) -> bool:
        """修改必须带真实来源；不允许凭一个新字符串直接改写已保存事实。"""
        message = kwargs.get("source_message")
        if not isinstance(message, Message):
            raise ValueError("update requires source_message: Message with the actual correction/update")
        namespace = self.config.user_id
        store = self.system.get_store(namespace)
        target = store.get_version(memory_id)
        session_id = kwargs.get("session_id")
        self.add_message(message, session_id=session_id, input_policy=kwargs.get("input_policy"))
        self.flush()
        conversation = _opaque("conversation", namespace, session_id or "")
        message_id = _opaque("message", conversation, message.id)
        source = next(s for s in store.sources() if s.conversation_id == conversation and s.message_id == message_id)
        family = {v.id for v in store.versions(memory_key=target.memory_key)}
        if not any(d.target_version_id in family and any(ref.type == "SOURCE" and ref.id == source.id for ref in d.premise_refs)
                   for d in store.dependencies()):
            raise ValueError("The supplied source was not successfully coordinated into the requested memory family")
        return True

    def save_memory(self) -> None:
        self.flush()  # SQLite 已持续写入；这里只完成剩余批次和可读取快照。

    def load_memory(self, user_id: str | None = None) -> bool:
        namespace = user_id or self.config.user_id
        if not self.system.database_path(namespace).is_file():
            return False
        self.system.get_store(namespace)  # 同时核对配置和数据库版本。
        self.config = self.config.model_copy(update={"user_id": namespace})
        return True

    def close(self) -> None:
        self.system.close()

    def cleanup(self) -> None:
        self.close()
