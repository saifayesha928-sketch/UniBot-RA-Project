from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, cast
from urllib.parse import urlparse

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_GROUNDING_MODEL = str(_PROJECT_ROOT / "models" / "tinylettuce-ettin-68m-en")

from pydantic import AnyUrl, Field, ValidationInfo, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def validate_postgres_direct_dsn(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme != "postgresql+psycopg":
        raise ValueError("must use the postgresql+psycopg SQLAlchemy dialect")
    if parsed.hostname is None or "pooler" in parsed.hostname:
        raise ValueError("must target the direct Neon host, not the pooler")
    if "sslmode=require" not in value:
        raise ValueError("must require SSL for Neon connections")
    return value


def validate_postgres_pooled_dsn(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme != "postgresql+psycopg":
        raise ValueError("must use the postgresql+psycopg SQLAlchemy dialect")
    if parsed.hostname is None or "pooler" not in parsed.hostname:
        raise ValueError("must target the pooled Neon host")
    if "sslmode=require" not in value:
        raise ValueError("must require SSL for Neon connections")
    return value


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
    env_file=".env",
    env_prefix="UNIBOT_",
    case_sensitive=False,
    extra="ignore",
)

    postgres_direct_dsn: str = Field(..., min_length=1)
    postgres_pooled_dsn: str = Field(..., min_length=1)
    qdrant_url: AnyUrl
    qdrant_api_key: str | None = None
    raw_storage_backend: Literal["local", "object_store"] = "local"
    raw_object_store_bucket: str | None = None
    raw_object_store_endpoint_url: AnyUrl | None = None
    raw_object_store_region: str | None = None
    raw_object_store_prefix: str = ""
    environment: Literal["development", "test", "production"] = "development"
    document_parser_backend: Literal["ade", "docling"] = "ade"
    document_cache_dir: str = ".unibot/cache/document_parser"
    ade_model: str = "dpt-2-latest"
    admin_api_key: str | None = None
    retrieval_runtime_mode: Literal["local_smoke", "production_like"] | None = None
    embedding_dense_backend: Literal["hashing", "cohere"] | None = None
    embedding_sparse_backend: Literal["hashing", "token", "fastembed"] | None = None
    cohere_api_key: str | None = None
    cohere_embed_model: str = "embed-v4.0"
    cohere_embed_base_url: str = "https://api.cohere.com/v2/embed"
    reranker_backend: Literal["token_overlap", "cohere", "cross_encoder"] | None = None
    cohere_rerank_model: str = "rerank-v4.0-fast"
    cohere_rerank_base_url: str = "https://api.cohere.com/v2/rerank"
    answer_model_backend: Literal["deterministic", "cohere", "openrouter"] | None = None
    answer_model_fallback_backend: Literal["cohere", "openrouter"] | None = None
    cohere_chat_model: str = "command-a-03-2025"
    cohere_chat_base_url: str = "https://api.cohere.com/v2/chat"
    cohere_timeout_seconds: float = Field(default=30.0, gt=0)
    openrouter_api_key: str | None = None
    openrouter_model: str = "anthropic/claude-sonnet-4"
    openrouter_base_url: str = "https://openrouter.ai/api/v1/chat/completions"
    openrouter_timeout_seconds: float = Field(default=60.0, gt=0)
    openrouter_app_name: str = "UniBot"
    grounding_verifier_backend: Literal["passthrough", "lettucedetect"] | None = None
    grounding_model: str = _DEFAULT_GROUNDING_MODEL
    grounding_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    grounding_span_confidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    grounding_skip_low_risk: bool = True
    grounding_confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    query_classifier_backend: Literal["keyword", "semantic"] = "semantic"
    query_classifier_shadow_mode: bool = False
    semantic_classifier_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    contextual_retrieval_enabled: bool = False
    contextual_retrieval_model: str = "anthropic/claude-haiku-4-5-20251001"
    contextual_retrieval_max_concurrency: int = Field(default=50, ge=1, le=500)
    contextual_retrieval_cache_ttl: Literal["5m", "1h"] = "5m"
    contextual_retrieval_max_retries: int = Field(default=3, ge=0, le=10)
    retrieval_min_relevance_score: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum reranker score for evidence to pass the relevance gate. "
            "Set to 0.0 to disable gating and let the answering model "
            "handle noise filtering. Recommended production range: 0.1-0.3."
        ),
    )
    retrieval_candidate_multiplier: int = Field(default=5, ge=1, le=100)
    retrieval_candidate_floor: int = Field(default=50, ge=1, le=500)
    retrieval_route_planning_enabled: bool = True
    query_rewriter_enabled: bool = False
    query_rewriter_model: str = "anthropic/claude-haiku-4-5-20251001"
    query_rewriter_timeout_seconds: float = Field(default=10.0, gt=0)
    query_rewriter_provider_order: str = "cerebras,groq"

    @field_validator("postgres_direct_dsn")
    @classmethod
    def validate_postgres_direct_dsn(cls, value: str) -> str:
        return validate_postgres_direct_dsn(value)

    @field_validator("postgres_pooled_dsn")
    @classmethod
    def validate_postgres_pooled_dsn(cls, value: str) -> str:
        return validate_postgres_pooled_dsn(value)

    @field_validator("raw_object_store_bucket")
    @classmethod
    def validate_object_store_bucket(
        cls, value: str | None, info: ValidationInfo
    ) -> str | None:
        backend = info.data.get("raw_storage_backend")
        if backend == "object_store" and not value:
            raise ValueError(
                "UNIBOT_RAW_OBJECT_STORE_BUCKET is required when using object storage"
            )
        return value

    @model_validator(mode="after")
    def validate_runtime_provider_settings(self) -> "Settings":
        dense_backend = self.resolve_embedding_dense_backend()
        sparse_backend = self.resolve_embedding_sparse_backend()

        if dense_backend == "hashing" and sparse_backend != "hashing":
            raise ValueError(
                "Hashing dense embeddings require hashing sparse embeddings"
            )
        if dense_backend == "cohere" and sparse_backend == "hashing":
            raise ValueError("Hashing sparse embeddings are development/test only")
        if dense_backend == "hashing" and sparse_backend == "fastembed":
            raise ValueError("FastEmbed sparse requires Cohere dense embeddings")
        if self.environment == "production" and (
            dense_backend == "hashing" or sparse_backend == "hashing"
        ):
            raise ValueError("Hashing embedding providers are development/test only")
        if self.environment == "production" and sparse_backend == "token":
            raise ValueError(
                "Token sparse embeddings are not supported in production; use fastembed."
            )
        if dense_backend == "cohere" and not self.cohere_api_key:
            raise ValueError(
                "UNIBOT_COHERE_API_KEY is required when using Cohere embeddings"
            )
        reranker_backend = self.resolve_reranker_backend()
        answer_model_backend = self.resolve_answer_model_backend()
        if self.environment == "production" and reranker_backend == "token_overlap":
            raise ValueError("Deterministic rerankers are development/test only")
        if self.environment == "production" and answer_model_backend == "deterministic":
            raise ValueError("Deterministic answer models are development/test only")
        if answer_model_backend == "openrouter" and not self.openrouter_api_key:
            raise ValueError(
                "UNIBOT_OPENROUTER_API_KEY is required when using OpenRouter answering"
            )
        if (
            reranker_backend == "cohere" or answer_model_backend == "cohere"
        ) and not self.cohere_api_key:
            raise ValueError(
                "UNIBOT_COHERE_API_KEY is required when using Cohere reranking or answering"
            )
        fallback_backend = self.answer_model_fallback_backend
        if fallback_backend == "openrouter" and not self.openrouter_api_key:
            raise ValueError(
                "UNIBOT_OPENROUTER_API_KEY is required when answer_model_fallback_backend is openrouter"
            )
        if fallback_backend == "cohere" and not self.cohere_api_key:
            raise ValueError(
                "UNIBOT_COHERE_API_KEY is required when answer_model_fallback_backend is cohere"
            )
        return self

    def resolve_retrieval_runtime_mode(
        self,
    ) -> Literal["local_smoke", "production_like"]:
        return resolve_retrieval_runtime_mode(self)

    def resolve_embedding_dense_backend(self) -> Literal["hashing", "cohere"]:
        return resolve_embedding_dense_backend(self)

    def resolve_embedding_sparse_backend(
        self,
    ) -> Literal["hashing", "token", "fastembed"]:
        return resolve_embedding_sparse_backend(self)

    def resolve_reranker_backend(
        self,
    ) -> Literal["token_overlap", "cohere", "cross_encoder"]:
        return resolve_reranker_backend(self)

    def resolve_answer_model_backend(
        self,
    ) -> Literal["deterministic", "cohere", "openrouter"]:
        return resolve_answer_model_backend(self)

    def resolve_grounding_verifier_backend(
        self,
    ) -> Literal["passthrough", "lettucedetect"]:
        return resolve_grounding_verifier_backend(self)

    def retrieval_quality_warning(self) -> str | None:
        return retrieval_quality_warning(self)


def resolve_retrieval_runtime_mode(
    settings: object,
) -> Literal["local_smoke", "production_like"]:
    mode = getattr(settings, "retrieval_runtime_mode", None)
    if mode is not None:
        return cast(Literal["local_smoke", "production_like"], mode)
    environment = getattr(settings, "environment", "development")
    return "production_like" if environment == "production" else "local_smoke"


def resolve_embedding_dense_backend(settings: object) -> Literal["hashing", "cohere"]:
    backend = getattr(settings, "embedding_dense_backend", None)
    if backend is not None:
        return cast(Literal["hashing", "cohere"], backend)
    return (
        "cohere"
        if resolve_retrieval_runtime_mode(settings) == "production_like"
        else "hashing"
    )


def resolve_embedding_sparse_backend(
    settings: object,
) -> Literal["hashing", "token", "fastembed"]:
    backend = getattr(settings, "embedding_sparse_backend", None)
    if backend is not None:
        return cast(Literal["hashing", "token", "fastembed"], backend)
    return (
        "fastembed"
        if resolve_retrieval_runtime_mode(settings) == "production_like"
        else "hashing"
    )


def resolve_reranker_backend(
    settings: object,
) -> Literal["token_overlap", "cohere", "cross_encoder"]:
    backend = getattr(settings, "reranker_backend", None)
    if backend is not None:
        return cast(Literal["token_overlap", "cohere", "cross_encoder"], backend)
    return (
        "cohere"
        if resolve_retrieval_runtime_mode(settings) == "production_like"
        else "token_overlap"
    )


def resolve_answer_model_backend(
    settings: object,
) -> Literal["deterministic", "cohere", "openrouter"]:
    backend = getattr(settings, "answer_model_backend", None)
    if backend is not None:
        return cast(Literal["deterministic", "cohere", "openrouter"], backend)
    return (
        "cohere"
        if resolve_retrieval_runtime_mode(settings) == "production_like"
        else "deterministic"
    )


def resolve_grounding_verifier_backend(
    settings: object,
) -> Literal["passthrough", "lettucedetect"]:
    backend = getattr(settings, "grounding_verifier_backend", None)
    if backend is not None:
        return cast(Literal["passthrough", "lettucedetect"], backend)
    return (
        "lettucedetect"
        if resolve_retrieval_runtime_mode(settings) == "production_like"
        else "passthrough"
    )


def retrieval_quality_warning(settings: object) -> str | None:
    if resolve_retrieval_runtime_mode(settings) == "production_like":
        return None
    return (
        "UniBot is running in local_smoke retrieval mode "
        "(hashing embeddings, token-overlap reranking, deterministic answers, "
        "passthrough grounding). "
        "Use UNIBOT_RETRIEVAL_RUNTIME_MODE=production_like for production-grade "
        "retrieval (Cohere dense + SPLADE++ sparse, neural reranking, LettuceDetect grounding)."
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
