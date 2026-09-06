"""A-MEM 薄适配：原生笔记演化、独立集合和完整状态检查点。"""

from hashlib import sha256
import json
import os
from pathlib import Path
import pickle
import tempfile
import numpy as np
from typing import Any, ClassVar

from .base import MemBaseLayer
from ..baselines.amem.memory_system import AgenticMemorySystem, MemoryNote
from ..configs.amem import AMEMConfig
from ..model_types.dataset import Message
from ..model_types.memory import MemoryEntry
from ..utils import PatchSpec, make_attr_patch, token_monitor


class AMEMLayer(MemBaseLayer):
    layer_type: ClassVar[str] = "A-MEM"

    def __init__(self, config: AMEMConfig, *, client=None) -> None:
        self.config, self.client = config, client
        self.processed_messages = 0
        self.input_policy = {}
        self.snapshot_id = None
        self._saved_digest = None
        namespace = sha256((str(Path(config.save_dir).resolve()) + ":" + config.user_id).encode()).hexdigest()[:32]
        self.memory_layer = AgenticMemorySystem(
            model_name=config.retriever_name_or_path, llm_backend=config.llm_backend,
            llm_model=config.llm_model, evo_threshold=config.evo_threshold,
            api_key=config.llm_api_key, base_url=config.llm_base_url,
            embedder_provider=config.embedding_provider,
            embedding_api_key=config.embedding_api_key, embedding_base_url=config.embedding_base_url,
            user_id=config.user_id, shared_client=client, collection_name="amem_" + namespace,
            preserve_unknown_time=config.preserve_unknown_time)

    @classmethod
    def from_config(cls, config, *, client=None):
        return cls(config, client=client)

    @property
    def checkpoint_path(self):
        return Path(self.config.save_dir) / "amem_state.pkl"

    def add_message(self, message: Message, **kwargs: Any) -> None:
        text = f"Speaker {message.name} (role: {message.role}) says: {message.content}"
        note_id = sha256((self.config.user_id + ":" + str(kwargs.get("session_id")) + ":" + message.id).encode()).hexdigest()
        if note_id in self.memory_layer.memories:
            if self.memory_layer.memories[note_id].content != text:
                raise ValueError("Existing A-MEM input changed")
            return
        if message.source_order is not None and message.source_order != self.processed_messages:
            raise ValueError("A-MEM input order differs from its checkpoint")
        self.memory_layer.retriever.operation_stage = "amem_build_embedding"
        self.memory_layer.add_note(text, time=message.timestamp, id=note_id)
        self.processed_messages += 1
        self.snapshot_id = None

    def add_messages(self, messages: list[Message], **kwargs: Any) -> None:
        if "input_policy" in kwargs:
            policy = dict(kwargs["input_policy"])
            if self.input_policy and policy != self.input_policy:
                raise ValueError("A-MEM public input policy changed")
            self.input_policy = policy
            self.memory_layer.llm_controller.llm.input_policy = policy
        for message in messages:
            self.add_message(message, **kwargs)

    def _state(self):
        notes = list(self.memory_layer.memories.values())
        index = self.memory_layer.retriever.collection.get(
            ids=[note.id for note in notes], include=["documents", "metadatas", "embeddings"]) if notes else {
                "ids": [], "documents": [], "metadatas": [], "embeddings": []}
        positions = {value: i for i, value in enumerate(index["ids"])}
        # 笔记可能已经演化，而索引尚未到刷新时机：两者分别保存，不能在加载时重新推理。
        rows = [{"id": n.id, "document": index["documents"][positions[n.id]],
                 "metadata": index["metadatas"][positions[n.id]],
                 "embedding": index["embeddings"][positions[n.id]]} for n in notes]
        return {"schema_version": 1, "config": self.config.model_dump(),
                "notes": [vars(n).copy() for n in notes], "index": rows,
                "evo_cnt": self.memory_layer.evo_cnt, "dirty": self.memory_layer.dirty,
                "warnings": self.memory_layer.warnings, "processed_messages": self.processed_messages,
                "input_policy": self.input_policy}

    def save_memory(self) -> None:
        payload = self._state()
        digest = self._state_digest(payload)
        if self._saved_digest == digest and self.checkpoint_path.is_file():
            return
        encoded = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        # 数据与输入位置在同一个文件原子提交；中断最多重做尚未提交的检查点批次。
        with tempfile.NamedTemporaryFile(dir=self.checkpoint_path.parent, prefix=".amem-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.checkpoint_path)
        self._saved_digest = digest
        from ..utils.benchmark_files import write_json
        write_json(self.checkpoint_path.parent / "amem_progress.json", {
            "processed_messages": self.processed_messages, "memories": len(payload["notes"]),
            "evolution_count": payload["evo_cnt"], "warnings": len(payload["warnings"])})

    @staticmethod
    def _state_digest(payload):
        # pickle 的对象共享关系不稳定；快照按逻辑内容和向量字节绑定。
        def vector(value):
            return {"vector_sha256": sha256(np.asarray(value, dtype=np.float32).tobytes()).hexdigest()}
        return sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False, default=vector).encode()).hexdigest()

    def load_memory(self, user_id: str | None = None) -> bool:
        if user_id is not None and user_id != self.config.user_id:
            raise ValueError("A-MEM checkpoint belongs to another namespace")
        if not self.checkpoint_path.is_file():
            return False
        encoded = self.checkpoint_path.read_bytes()
        payload = pickle.loads(encoded)  # 仅加载本运行生成并通过配置校验的本地检查点。
        if payload["schema_version"] != 1 or payload["config"] != self.config.model_dump():
            raise ValueError("A-MEM checkpoint configuration differs from this run")
        self.memory_layer.memories = {
            row["id"]: MemoryNote(**row, preserve_unknown_time=self.config.preserve_unknown_time)
            for row in payload["notes"]}
        index = payload["index"]
        collection = self.memory_layer.retriever.collection
        existing = collection.get()["ids"]
        if existing:
            collection.delete(ids=existing)
        if index:
            collection.add(ids=[r["id"] for r in index], documents=[r["document"] for r in index],
                           metadatas=[r["metadata"] for r in index], embeddings=[r["embedding"] for r in index])
        self.memory_layer.evo_cnt = payload["evo_cnt"]
        self.memory_layer.dirty = payload["dirty"]
        self.memory_layer.warnings = payload["warnings"]
        self.processed_messages = payload["processed_messages"]
        self.input_policy = payload["input_policy"]
        self.memory_layer.llm_controller.llm.input_policy = self.input_policy
        self._saved_digest = self._state_digest(payload)
        self.snapshot_id = None
        return True

    def flush(self):
        if self.snapshot_id is not None and not self.memory_layer.dirty:
            return self.snapshot_id
        self.memory_layer.retriever.operation_stage = "amem_build_embedding"
        self.memory_layer.consolidate_memories()
        self.save_memory()
        self.snapshot_id = "amem:" + self._saved_digest
        return self.snapshot_id

    def get_memory_snapshot(self, snapshot_id):
        if self.memory_layer.dirty or snapshot_id != "amem:" + str(self._saved_digest):
            raise ValueError("A-MEM only exposes its matching final observation snapshot")
        return {"snapshot_id": snapshot_id, "text": "\n".join(
                    note.content for note in self.memory_layer.memories.values()),
                "source_cutoff": self.processed_messages - 1,
                "maintenance_incomplete": bool(self.memory_layer.warnings),
                "warnings": self.memory_layer.warnings,
                "checkpoint": str(self.checkpoint_path)}

    def retrieve(self, query: str, k: int = 10, **kwargs: Any) -> list[MemoryEntry]:
        if kwargs.get("query_time") is not None and self.config.preserve_unknown_time:
            raise ValueError("MAB has no public query date")
        if kwargs.get("snapshot_id") is not None:
            if self.memory_layer.dirty or kwargs["snapshot_id"] != "amem:" + str(self._saved_digest):
                raise ValueError("A-MEM retrieval snapshot mismatch")
        self.memory_layer.retriever.operation_stage = "amem_search_embedding"
        memories = self.memory_layer.search_agentic(query, k=k)
        outputs = []
        for memory in memories:
            fields = {"memory content": memory["content"], "memory context": memory["context"],
                      "memory keywords": str(memory["keywords"]), "memory tags": str(memory["tags"])}
            if memory.get("timestamp"):
                fields["talk start time"] = memory["timestamp"]
            formatted = "\n".join(f"{key}: {value}" for key, value in fields.items())
            outputs.append(MemoryEntry(content=memory["content"], formatted_content=formatted,
                                       metadata={key: value for key, value in memory.items() if key != "content"}))
        return outputs

    def delete(self, memory_id: str) -> bool:
        self.snapshot_id = None
        return self.memory_layer.delete(memory_id)

    def update(self, memory_id: str, **kwargs: Any) -> bool:
        self.snapshot_id = None
        return self.memory_layer.update(memory_id, **kwargs)

    def cleanup(self):
        # Chroma 是本地临时索引；真正的恢复依据是已原子保存的检查点。
        self.memory_layer.retriever.client.delete_collection(self.memory_layer.collection_name)

    def get_patch_specs(self) -> list[PatchSpec]:
        if self.client is not None:
            return []  # 已由共享账本计数，不叠加旧监控包装。
        # In this case, we modify an instance's method. 
        # Other instances are not affected. 
        # Note that there is no need to check `response_format` parameter. 
        getter, setter = make_attr_patch(self.memory_layer.llm_controller.llm, "get_completion")
        spec = PatchSpec(
            name=f"{self.memory_layer.llm_controller.llm.__class__.__name__}.get_completion",
            getter=getter,
            setter=setter,
            wrapper=token_monitor(
                extract_model_name=lambda *args, **kwargs: (self.config.llm_model, {}),
                extract_input_dict=lambda *args, **kwargs: {
                    "messages": [
                        {
                            "role": "system", 
                            "content": "You must respond with a JSON object."
                        }, 
                        {
                            "role": "user",
                            "content": kwargs.get("prompt", args[0] if len(args) > 0 else "") 
                        }
                    ],
                    "metadata": {
                        "op_type": (
                            "generation"
                            if kwargs.get("prompt", args[0] if len(args) > 0 else "").startswith(
                                "Generate a structured analysis"
                            ) 
                            else "update"
                        )
                    }
                },
                extract_output_dict=lambda result: {
                    "messages": result
                },
            ),
        )
        return [spec]
