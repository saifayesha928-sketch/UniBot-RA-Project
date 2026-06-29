from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from unibot.settings import (
    Settings,
    resolve_embedding_dense_backend,
    resolve_embedding_sparse_backend,
)

DenseBackend = Literal["hashing", "cohere"]
SparseBackend = Literal["hashing", "token", "fastembed"]


@dataclass(frozen=True, slots=True)
class EmbeddingProviderSettings:
    environment: Literal["development", "test", "production"]
    dense_backend: DenseBackend
    sparse_backend: SparseBackend
    cohere_api_key: str | None
    cohere_embed_model: str
    cohere_embed_base_url: str
    cohere_timeout_seconds: float


def resolve_embedding_provider_settings(
    settings: Settings,
) -> EmbeddingProviderSettings:
    return EmbeddingProviderSettings(
        environment=settings.environment,
        dense_backend=resolve_embedding_dense_backend(settings),
        sparse_backend=resolve_embedding_sparse_backend(settings),
        cohere_api_key=settings.cohere_api_key,
        cohere_embed_model=settings.cohere_embed_model,
        cohere_embed_base_url=settings.cohere_embed_base_url,
        cohere_timeout_seconds=settings.cohere_timeout_seconds,
    )
