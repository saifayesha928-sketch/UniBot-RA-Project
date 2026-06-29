from __future__ import annotations

import re
from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import httpx

from unibot.indexing.composite_embeddings import CompositeDenseSparseEmbeddingProvider
from unibot.indexing.embeddings import (
    DenseSparseEmbeddingProvider,
    EmbeddingVector,
    SparseVector,
)
from unibot.indexing.provider_settings import resolve_embedding_provider_settings
from unibot.indexing.providers.real_dense import CohereDenseEmbeddingProvider
from unibot.indexing.providers.fastembed_sparse import FastEmbedSparseEmbeddingProvider
from unibot.indexing.providers.real_sparse import TokenSparseEmbeddingProvider
from unibot.settings import Settings


class HashingEmbeddingProvider:
    def embed(self, text: str) -> EmbeddingVector:
        return self.embed_document(text)

    def embed_document(self, text: str) -> EmbeddingVector:
        return self._embed(text)

    def embed_query(self, text: str) -> EmbeddingVector:
        return self._embed(text)

    def _embed(self, text: str) -> EmbeddingVector:
        token_counts = Counter(re.findall(r"[a-z0-9]+", text.lower()))
        ordered_tokens = sorted(token_counts.items())
        dense_seed = sum(ord(char) for char in text[:32])
        dense_vector = tuple(
            ((dense_seed + offset) % 100) / 100.0 for offset in (11, 37, 61)
        )
        sparse_indices = tuple(index + 1 for index, _ in enumerate(ordered_tokens[:16]))
        sparse_values = tuple(float(count) for _, count in ordered_tokens[:16])
        if not sparse_indices:
            sparse_indices = (1,)
            sparse_values = (1.0,)
        return EmbeddingVector(
            dense_vector=dense_vector,
            sparse_vector=SparseVector(
                indices=sparse_indices,
                values=sparse_values,
            ),
        )


def create_embedding_provider(
    *,
    settings: Settings,
    client: httpx.Client | None = None,
) -> DenseSparseEmbeddingProvider:
    provider_settings = resolve_embedding_provider_settings(settings)

    if (
        provider_settings.dense_backend == "hashing"
        and provider_settings.sparse_backend == "hashing"
    ):
        return HashingEmbeddingProvider()

    if (
        provider_settings.dense_backend == "cohere"
        and provider_settings.sparse_backend == "token"
        and provider_settings.cohere_api_key is not None
    ):
        return CompositeDenseSparseEmbeddingProvider(
            dense_provider=CohereDenseEmbeddingProvider(
                api_key=provider_settings.cohere_api_key,
                model=provider_settings.cohere_embed_model,
                base_url=provider_settings.cohere_embed_base_url,
                timeout=provider_settings.cohere_timeout_seconds,
                client=client,
            ),
            sparse_provider=TokenSparseEmbeddingProvider(),
        )

    if (
        provider_settings.dense_backend == "cohere"
        and provider_settings.sparse_backend == "fastembed"
        and provider_settings.cohere_api_key is not None
    ):
        return CompositeDenseSparseEmbeddingProvider(
            dense_provider=CohereDenseEmbeddingProvider(
                api_key=provider_settings.cohere_api_key,
                model=provider_settings.cohere_embed_model,
                base_url=provider_settings.cohere_embed_base_url,
                timeout=provider_settings.cohere_timeout_seconds,
                client=client,
            ),
            sparse_provider=FastEmbedSparseEmbeddingProvider(),
        )

    raise ValueError(
        "Unsupported embedding provider configuration. "
        "Use hashing/hashing for development/test, cohere/fastembed for production, "
        "or cohere/token as a local-only fallback (not supported in production)."
    )
