"""V5 启动参数。上限不是每次必须用满的配额。"""

from pydantic import Field

from .base import MemBaseConfig


class OurMemConfig(MemBaseConfig):
    user_id: str = "default"
    model_name: str = "gpt-4.1-mini"
    embedding_model_name: str = "text-embedding-3-small"
    answer_model: str = "gpt-4.1-mini"
    judge_model: str = "gpt-4.1-mini"
    api_key: str = Field(default="", repr=False, exclude=True)
    base_url: str = "https://api.openai.com/v1"
    b_memory: int = Field(default=8, ge=1)
    max_batch_tokens: int = Field(default=6000, ge=1)
    w_context: int = Field(default=8, ge=0)
    w_context_max: int = Field(default=64, ge=0)
    max_source_chunk_tokens: int = Field(default=1000, ge=1)
    max_context_tokens: int = Field(default=16000, ge=1)
    max_model_output_tokens: int = Field(default=6000, ge=1)
    rrf_c: int = Field(default=60, ge=1)
    q_max: int = Field(default=0, ge=0)
    max_gap_queries_per_call: int = Field(default=2, ge=0)
    max_claim_depth: int = Field(default=5, ge=1)
    max_claims_per_call: int = Field(default=8, ge=1)
    max_dependencies_per_call: int = Field(default=24, ge=1)
    max_premises_per_dependency: int = Field(default=8, ge=1)
    max_repair_targets_per_call: int = Field(default=8, ge=1)
    max_generation_calls_per_update: int = Field(default=8, ge=1)
    max_read_rounds: int = Field(default=4, ge=1)
    max_read_queries_per_round: int = Field(default=4, ge=1)
    max_aggregate_scan_tokens: int = Field(default=64000, ge=0)
    max_evidence_tokens: int = Field(default=8000, ge=1)
    max_llm_retries: int = Field(default=2, ge=0)
    memory_temperature: float = 0.0
    seed: int = 0
    embedding_batch_size: int = Field(default=128, ge=1)
    request_timeout: float = Field(default=120.0, gt=0)
    k_dense: dict[str, int] = Field(default_factory=lambda: {
        "reconcile": 8, "derive": 16, "validate": 8,
        "read": 20, "history": 20, "aggregate": 20, "source": 8})
    k_bm25: dict[str, int] = Field(default_factory=lambda: {
        "reconcile": 8, "derive": 16, "validate": 8,
        "read": 20, "history": 20, "aggregate": 20, "source": 8})
    k_time: dict[str, int] = Field(default_factory=lambda: {
        "reconcile": 2, "derive": 4, "validate": 2,
        "read": 4, "history": 4, "aggregate": 4, "source": 0})
    max_candidates: dict[str, int] = Field(default_factory=lambda: {
        "reconcile": 16, "derive": 32, "validate": 16,
        "read": 40, "history": 40, "aggregate": 40, "source": 8})

    def get_llm_models(self) -> list[str]:
        return list(dict.fromkeys([self.model_name, self.answer_model, self.judge_model]))
