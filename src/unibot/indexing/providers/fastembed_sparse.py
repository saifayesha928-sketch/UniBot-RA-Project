from __future__ import annotations

from pathlib import Path

import structlog
from fastembed import SparseTextEmbedding

from unibot.indexing.embeddings import SparseVector

logger = structlog.get_logger(__name__)

_DEFAULT_MODEL = "prithivida/Splade_PP_en_v1"
_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "fastembed"


class FastEmbedSparseEmbeddingProvider:
    """SPLADE++ sparse embeddings via FastEmbed (ONNX Runtime, no GPU required)."""

    def __init__(self, *, model: str = _DEFAULT_MODEL, cache_dir: Path | None = None) -> None:
        self._model_name = model
        self._cache_dir = str(cache_dir or _DEFAULT_CACHE_DIR)
        self._model: SparseTextEmbedding | None = None

    def _ensure_model(self) -> SparseTextEmbedding:
        if self._model is None:
            logger.info("fastembed.loading_model", model=self._model_name, cache_dir=self._cache_dir)
            self._model = SparseTextEmbedding(model_name=self._model_name, cache_dir=self._cache_dir)
        return self._model

    def embed_sparse(self, text: str) -> SparseVector:
        return self.embed_sparse_document(text)

    def embed_sparse_document(self, text: str) -> SparseVector:
        return self._embed(text)

    def embed_sparse_query(self, text: str) -> SparseVector:
        model = self._ensure_model()
        results = list(model.query_embed(text))
        if not results:
            return SparseVector(indices=(1,), values=(1.0,))
        embedding = results[0]
        return SparseVector(
            indices=tuple(int(idx) for idx in embedding.indices),
            values=tuple(float(val) for val in embedding.values),
        )

    def embed_sparse_query_batch(
        self, texts: list[str] | tuple[str, ...],
    ) -> list[SparseVector]:
        model = self._ensure_model()
        results = list(model.query_embed(list(texts)))
        return [
            SparseVector(
                indices=tuple(int(idx) for idx in emb.indices),
                values=tuple(float(val) for val in emb.values),
            )
            for emb in results
        ]

    def embed_sparse_document_batch(
        self, texts: list[str] | tuple[str, ...],
    ) -> list[SparseVector]:
        model = self._ensure_model()
        results = list(model.embed(list(texts)))
        return [
            SparseVector(
                indices=tuple(int(idx) for idx in emb.indices),
                values=tuple(float(val) for val in emb.values),
            )
            for emb in results
        ]

    def _embed(self, text: str) -> SparseVector:
        model = self._ensure_model()
        results = list(model.embed([text]))
        if not results:
            return SparseVector(indices=(1,), values=(1.0,))
        embedding = results[0]
        return SparseVector(
            indices=tuple(int(idx) for idx in embedding.indices),
            values=tuple(float(val) for val in embedding.values),
        )
