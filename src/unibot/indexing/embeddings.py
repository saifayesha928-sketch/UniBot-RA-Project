from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, cast

import structlog

from unibot.indexing.chunks import IndexChunk

logger = structlog.get_logger(__name__)

_EMBED_BATCH_SIZE = 96


@dataclass(frozen=True, slots=True)
class SparseVector:
    indices: tuple[int, ...]
    values: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class EmbeddingVector:
    dense_vector: tuple[float, ...]
    sparse_vector: SparseVector


@dataclass(frozen=True, slots=True)
class EmbeddedChunk:
    chunk: IndexChunk
    record_version_id: str
    vectors: EmbeddingVector


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    embedded_chunks: tuple[EmbeddedChunk, ...]
    failed_record_version_ids: tuple[str, ...]


class DenseSparseEmbeddingProvider(Protocol):
    def embed(self, text: str) -> EmbeddingVector: ...


def embed_document_text(
    provider: DenseSparseEmbeddingProvider,
    text: str,
) -> EmbeddingVector:
    embed_document = getattr(provider, "embed_document", None)
    if callable(embed_document):
        return cast(EmbeddingVector, embed_document(text))
    return provider.embed(text)


def embed_query_text(
    provider: DenseSparseEmbeddingProvider,
    text: str,
) -> EmbeddingVector:
    embed_query = getattr(provider, "embed_query", None)
    if callable(embed_query):
        return cast(EmbeddingVector, embed_query(text))
    return provider.embed(text)


def embed_query_texts_batch(
    provider: DenseSparseEmbeddingProvider,
    texts: list[str] | tuple[str, ...],
) -> dict[str, EmbeddingVector]:
    """Embed multiple query texts, returning a text→vector map.

    Uses the provider's batch method when available, falling back to
    sequential embed_query_text calls.
    """
    unique_texts = list(dict.fromkeys(texts))  # preserve order, dedupe
    batch_fn = getattr(provider, "embed_query_batch", None)
    if callable(batch_fn):
        vectors = cast(list[EmbeddingVector], batch_fn(unique_texts))
    else:
        vectors = [embed_query_text(provider, t) for t in unique_texts]
    return dict(zip(unique_texts, vectors, strict=True))


def embed_chunks(
    chunks: list[IndexChunk] | tuple[IndexChunk, ...],
    provider: DenseSparseEmbeddingProvider,
    *,
    progress: Callable[[int, int], None] | None = None,
) -> EmbeddingResult:
    active_chunks = [c for c in chunks if c.is_active]

    embed_batch = getattr(provider, "embed_document_batch", None)
    if callable(embed_batch):
        return _embed_chunks_batched(active_chunks, embed_batch, progress=progress)
    return _embed_chunks_sequential(active_chunks, provider, progress=progress)


def _embed_chunks_batched(
    chunks: list[IndexChunk],
    embed_batch_fn: Callable[[list[str]], list[EmbeddingVector]],
    *,
    progress: Callable[[int, int], None] | None = None,
) -> EmbeddingResult:
    embedded_chunks: list[EmbeddedChunk] = []
    failed_record_version_ids: list[str] = []
    total = len(chunks)

    for batch_start in range(0, total, _EMBED_BATCH_SIZE):
        batch = chunks[batch_start : batch_start + _EMBED_BATCH_SIZE]
        batch_end = batch_start + len(batch)
        logger.info(
            "embedding.batch",
            progress=f"{batch_end}/{total}",
            batch_size=len(batch),
        )

        try:
            vectors_list = embed_batch_fn([c.text for c in batch])
        except Exception:
            logger.warning(
                "embedding.batch_failed_falling_back_to_sequential",
                batch_start=batch_start,
                batch_size=len(batch),
                exc_info=True,
            )
            for chunk in batch:
                try:
                    vectors = embed_batch_fn([chunk.text])[0]
                except Exception:
                    logger.warning(
                        "embedding.chunk_failed",
                        record_version_id=chunk.record_version_id,
                        chunk_id=chunk.chunk_id,
                        exc_info=True,
                    )
                    failed_record_version_ids.append(chunk.record_version_id)
                    continue
                embedded_chunks.append(
                    EmbeddedChunk(
                        chunk=chunk,
                        record_version_id=chunk.record_version_id,
                        vectors=vectors,
                    )
                )
            continue

        for chunk, vectors in zip(batch, vectors_list, strict=True):
            embedded_chunks.append(
                EmbeddedChunk(
                    chunk=chunk,
                    record_version_id=chunk.record_version_id,
                    vectors=vectors,
                )
            )

        if progress is not None:
            progress(batch_end, total)

    return EmbeddingResult(
        embedded_chunks=tuple(embedded_chunks),
        failed_record_version_ids=tuple(failed_record_version_ids),
    )


def _embed_chunks_sequential(
    chunks: list[IndexChunk],
    provider: DenseSparseEmbeddingProvider,
    *,
    progress: Callable[[int, int], None] | None = None,
) -> EmbeddingResult:
    embedded_chunks: list[EmbeddedChunk] = []
    failed_record_version_ids: list[str] = []
    total = len(chunks)

    for i, chunk in enumerate(chunks, 1):
        try:
            vectors = embed_document_text(provider, chunk.text)
        except Exception:
            logger.warning(
                "embedding.chunk_failed",
                record_version_id=chunk.record_version_id,
                chunk_id=chunk.chunk_id,
                exc_info=True,
            )
            failed_record_version_ids.append(chunk.record_version_id)
        else:
            embedded_chunks.append(
                EmbeddedChunk(
                    chunk=chunk,
                    record_version_id=chunk.record_version_id,
                    vectors=vectors,
                )
            )
        if progress is not None:
            progress(i, total)

    return EmbeddingResult(
        embedded_chunks=tuple(embedded_chunks),
        failed_record_version_ids=tuple(failed_record_version_ids),
    )
