from pydantic import Field

from .base import MemBaseConfig


class OurMemConfig(MemBaseConfig):
    """OurMem 第一版端到端配置。"""

    model_name: str = "gpt-4.1-mini"
    embedding_model_name: str = "text-embedding-3-small"
    api_key: str = Field(repr=False)
    base_url: str = "https://api.openai.com/v1"
    fact_candidate_k: int = Field(default=8, ge=1)
    support_candidate_k: int = Field(default=12, ge=2)
    claim_candidate_k: int = Field(default=8, ge=1)
    fact_similarity_threshold: float = Field(default=0.45, ge=-1, le=1)
    claim_similarity_threshold: float = Field(default=0.25, ge=-1, le=1)
    message_batch_size: int = Field(default=8, ge=1)
    embedding_batch_size: int = Field(default=128, ge=1)
    max_induced_claims_per_fact: int = Field(default=4, ge=1)

    def get_llm_models(self) -> list[str]:
        return [self.model_name]
