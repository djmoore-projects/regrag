"""Runtime configuration, read from environment variables / .env."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://regrag:regrag@localhost:5432/regrag"
    data_dir: str = "data"
    ecfr_date: str = "2026-10-01"

    embed_model: str = "BAAI/bge-base-en-v1.5"
    embed_dim: int = 768
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-12-v2"

    # Retrieval knobs
    candidates_per_retriever: int = 50
    rrf_k: int = 60
    rerank_pool: int = 30
    final_k: int = 8
    # Structure-aware boosts: linked rule text in the rerank pool, rule-text guarantee, citation pinning.
    structural_boosts: bool = True
    link_expansion_max: int = 4
    # Below this top rerank score the system abstains without calling the LLM.
    abstain_threshold: float = 0.0

    # Generation
    anthropic_model: str = "claude-opus-5-5"
    anthropic_effort: str = "low"
    max_answer_tokens: int = 4000

    # Public-demo cost guards
    rate_limit_per_minute: int = 6
    daily_llm_budget: int = 300

    phoenix_collector_endpoint: str | None = None
    phoenix_project: str = "regrag"


@lru_cache
def get_settings() -> Settings:
    return Settings()
