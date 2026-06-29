from __future__ import annotations

from typing import Protocol


class DenseEmbeddingProvider(Protocol):
    def embed_dense(self, text: str) -> tuple[float, ...]: ...
