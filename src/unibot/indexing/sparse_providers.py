from __future__ import annotations

from typing import Protocol

from unibot.indexing.embeddings import SparseVector


class SparseEmbeddingProvider(Protocol):
    def embed_sparse(self, text: str) -> SparseVector: ...
