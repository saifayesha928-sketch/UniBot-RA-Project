from __future__ import annotations

from typing import cast

from unibot.indexing.dense_providers import DenseEmbeddingProvider
from unibot.indexing.embeddings import EmbeddingVector, SparseVector
from unibot.indexing.sparse_providers import SparseEmbeddingProvider


class CompositeDenseSparseEmbeddingProvider:
    def __init__(
        self,
        *,
        dense_provider: DenseEmbeddingProvider,
        sparse_provider: SparseEmbeddingProvider,
    ) -> None:
        self._dense_provider = dense_provider
        self._sparse_provider = sparse_provider

    def embed(self, text: str) -> EmbeddingVector:
        return self.embed_document(text)

    def embed_document(self, text: str) -> EmbeddingVector:
        return EmbeddingVector(
            dense_vector=_embed_dense(self._dense_provider, text, mode="document"),
            sparse_vector=_embed_sparse(self._sparse_provider, text, mode="document"),
        )

    def embed_document_batch(
        self, texts: list[str] | tuple[str, ...]
    ) -> list[EmbeddingVector]:
        dense_batch_method = getattr(self._dense_provider, "embed_dense_document_batch", None)
        if callable(dense_batch_method):
            dense_vectors = dense_batch_method(list(texts))
        else:
            dense_vectors = [
                _embed_dense(self._dense_provider, t, mode="document") for t in texts
            ]
        sparse_batch_method = getattr(self._sparse_provider, "embed_sparse_document_batch", None)
        if callable(sparse_batch_method):
            sparse_vectors = sparse_batch_method(list(texts))
        else:
            sparse_vectors = [
                _embed_sparse(self._sparse_provider, t, mode="document") for t in texts
            ]
        return [
            EmbeddingVector(dense_vector=d, sparse_vector=s)
            for d, s in zip(dense_vectors, sparse_vectors, strict=True)
        ]

    def embed_query(self, text: str) -> EmbeddingVector:
        return EmbeddingVector(
            dense_vector=_embed_dense(self._dense_provider, text, mode="query"),
            sparse_vector=_embed_sparse(self._sparse_provider, text, mode="query"),
        )

    def embed_query_batch(
        self, texts: list[str] | tuple[str, ...]
    ) -> list[EmbeddingVector]:
        dense_batch_method = getattr(self._dense_provider, "embed_dense_query_batch", None)
        if callable(dense_batch_method):
            dense_vectors = dense_batch_method(list(texts))
        else:
            dense_vectors = [
                _embed_dense(self._dense_provider, t, mode="query") for t in texts
            ]
        sparse_batch_method = getattr(self._sparse_provider, "embed_sparse_query_batch", None)
        if callable(sparse_batch_method):
            sparse_vectors = sparse_batch_method(list(texts))
        else:
            sparse_vectors = [
                _embed_sparse(self._sparse_provider, t, mode="query") for t in texts
            ]
        return [
            EmbeddingVector(dense_vector=d, sparse_vector=s)
            for d, s in zip(dense_vectors, sparse_vectors, strict=True)
        ]


def _embed_dense(provider: DenseEmbeddingProvider, text: str, *, mode: str) -> tuple[float, ...]:
    method_name = "embed_dense_query" if mode == "query" else "embed_dense_document"
    method = getattr(provider, method_name, None)
    if callable(method):
        return tuple(method(text))
    return tuple(provider.embed_dense(text))


def _embed_sparse(provider: SparseEmbeddingProvider, text: str, *, mode: str) -> SparseVector:
    method_name = "embed_sparse_query" if mode == "query" else "embed_sparse_document"
    method = getattr(provider, method_name, None)
    if callable(method):
        return cast(SparseVector, method(text))
    return provider.embed_sparse(text)
