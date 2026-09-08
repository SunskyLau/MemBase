from .base import MemBaseConfig 
from pydantic import Field
from typing import Literal


class AMEMConfig(MemBaseConfig):
    """The default configuration for A-MEM."""

    llm_backend: Literal["openai", "ollama"] = Field(
        default="openai",
        description="The backend to use for the LLM. Currently, only openai and ollama are supported.",
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        description="The base backbone model to use.",
    )
    llm_api_key: str | None = Field(
        default=None,
        repr=False, exclude=True,
        description=(
            "The API key to use for the LLM. It is used for openai backend. "
            "If not provided, the API key will be loaded from the environment variable."
        ),
    )
    llm_base_url: str | None = Field(
        default=None,
        description=(
            "The base URL to use for the LLM. It is used for openai backend. "
            "If not provided, the base URL will be loaded from the environment variable."
        ),
    )

    embedding_provider: Literal["sentence-transformers", "openai"] = Field(
        default="sentence-transformers",
        description="The provider for the embedding model.",
    )
    retriever_name_or_path: str = Field(
        default="all-MiniLM-L6-v2",
        description="The name or path of the retriever model to use.",
    )
    embedding_api_key: str | None = Field(
        default=None,
        repr=False, exclude=True,
        description=(
            "The API key to use for the embedding model. It is used for openai backend. "
            "If not provided, the API key will be loaded from the environment variable."
        ),
    )
    embedding_base_url: str | None = Field(
        default=None,
        description=(
            "The base URL to use for the embedding model. It is used for openai backend. "
            "If not provided, the base URL will be loaded from the environment variable."
        ),
    )

    # In A-MEM, each memory evolution operation modifies the keywords, tags, and context of notes. 
    # However, the corresponding embeddings are not updated. 
    # If the embeddings were updated every time a note is added, the overhead would be substantial. 
    # Therefore, A-MEM introduces a hyperparameter `evo_threshold`
    # where after adding `evo_threshold` notes, all note embeddings are updated.
    evo_threshold: int = Field(
        default=100,
        description="The threshold for the number of memories to trigger evolution.",
        gt=0,
    )

    preserve_unknown_time: bool = False
    llm_temperature: float = Field(default=0.7, ge=0, le=2)
    llm_max_output_tokens: int | None = Field(default=None, ge=1,
        description="官方 OpenAI 路径默认不指定；如服务商默认截断，显式配置并记录适配值。")
    llm_max_input_tokens: int | None = Field(default=None, ge=1,
        description="不额外限制方法输入；仍受模型服务的真实上下文容量约束。")
    checkpoint_interval: int = Field(default=8, ge=1)
    max_evidence_tokens: int | None = Field(default=None, ge=1,
        description="默认完整保留检索结果，不额外施加证据词元预算。")

    def get_llm_models(self) -> list[str]:
        return [self.llm_model]
