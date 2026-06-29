from __future__ import annotations

import asyncio
import hashlib
import random
from collections import defaultdict
from typing import Literal

import httpx
import structlog

from unibot.crawl.async_runner import run_sync
from unibot.db.models import ContextualChunkCache
from unibot.db.repositories.contextual_chunk_cache import ContextualChunkCacheRepository
from unibot.domain.record_payloads import extract_primary_content
from unibot.indexing.chunks import IndexChunk
from unibot.indexing.contextual_retrieval import (
    call_openrouter_context_async,
    context_prompt_hash,
)
from unibot.verify.rules import VerificationDecision

logger = structlog.get_logger(__name__)


def _chunk_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cache_key(
    record_version_id: str,
    chunk_id: str,
    chunk_text: str,
    model_name: str,
    prompt_hash: str,
) -> str:
    text_hash = _chunk_text_hash(chunk_text)
    raw = "|".join([record_version_id, chunk_id, text_hash, model_name, prompt_hash])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


_DEFAULT_BATCH_SIZE = 100
_DEFAULT_INTER_BATCH_DELAY_MIN = 1.0
_DEFAULT_INTER_BATCH_DELAY_MAX = 5.0

# Retry configuration for transient OpenRouter errors.
_TRANSIENT_STATUS_CODES = frozenset({429, 408, 502, 503})
_BASE_RETRY_DELAY = 1.0
_MAX_RETRY_DELAY = 8.0
_RETRY_JITTER_MAX = 0.5


def _retry_delay(attempt: int, response: httpx.Response | None = None) -> float:
    """Exponential backoff with jitter.  Respects ``Retry-After`` header."""
    if response is not None:
        retry_after = response.headers.get("retry-after")
        if retry_after:
            try:
                return min(float(retry_after), _MAX_RETRY_DELAY)
            except ValueError:
                pass
    delay = min(_BASE_RETRY_DELAY * (2 ** attempt), _MAX_RETRY_DELAY)
    return delay + random.uniform(0, _RETRY_JITTER_MAX)


class ContextualizationService:
    def __init__(
        self,
        *,
        cache_repository: ContextualChunkCacheRepository,
        model_name: str,
        max_concurrency: int,
        cache_ttl: Literal["5m", "1h"] = "5m",
        timeout: float,
        run_sync_timeout: float = 600.0,
        base_url: str,
        api_key: str,
        app_name: str,
        max_retries: int = 3,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        inter_batch_delay_min: float = _DEFAULT_INTER_BATCH_DELAY_MIN,
        inter_batch_delay_max: float = _DEFAULT_INTER_BATCH_DELAY_MAX,
    ) -> None:
        self._cache_repository = cache_repository
        self._model_name = model_name
        self._max_concurrency = max_concurrency
        self._cache_ttl = cache_ttl
        self._timeout = timeout
        self._run_sync_timeout = run_sync_timeout
        self._base_url = base_url
        self._api_key = api_key
        self._app_name = app_name
        self._max_retries = max_retries
        self._batch_size = batch_size
        self._inter_batch_delay_min = inter_batch_delay_min
        self._inter_batch_delay_max = inter_batch_delay_max
        self._prompt_hash = context_prompt_hash(cache_ttl=cache_ttl)

    def contextualize(
        self,
        *,
        chunks: tuple[IndexChunk, ...],
        decisions: tuple[VerificationDecision, ...],
    ) -> tuple[IndexChunk, ...]:
        if not chunks:
            return chunks

        decisions_by_rvid = {
            d.candidate.record_version_id: d for d in decisions
        }

        # Build cache keys for all chunks
        chunk_cache_keys: dict[int, str] = {}
        for idx, chunk in enumerate(chunks):
            key = _cache_key(
                chunk.record_version_id,
                chunk.chunk_id,
                chunk.text,
                self._model_name,
                self._prompt_hash,
            )
            chunk_cache_keys[idx] = key

        # Synchronous cache lookup
        all_keys = list(chunk_cache_keys.values())
        cached = self._cache_repository.get_many(all_keys)

        # Separate hits from misses
        results: dict[int, str] = {}  # idx -> context_text or original chunk text
        miss_indices: list[int] = []

        for idx, chunk in enumerate(chunks):
            key = chunk_cache_keys[idx]
            if key in cached:
                results[idx] = cached[key].context_text
            elif chunk.record_version_id not in decisions_by_rvid:
                # No decision available — keep original text
                results[idx] = chunk.text
            else:
                miss_indices.append(idx)

        # Process misses via async concurrency
        if miss_indices:
            # Group misses by record_version_id for batched concurrent processing
            groups: dict[str, list[int]] = defaultdict(list)
            for idx in miss_indices:
                groups[chunks[idx].record_version_id].append(idx)

            # Sort within each group by chunk_index to preserve ordering
            for rvid in groups:
                groups[rvid].sort(key=lambda i: chunks[i].metadata.get("chunk_index", 0))

            # Build document text lookup
            doc_texts: dict[str, str] = {}
            for rvid, indices in groups.items():
                decision = decisions_by_rvid[rvid]
                doc_texts[rvid] = extract_primary_content(
                    decision.candidate.record_payload,
                    record_type=decision.candidate.record_type,
                )

            # Run async orchestration
            async_results, cacheable_indices = run_sync(
                lambda: self._process_groups_async(groups, chunks, doc_texts),
                timeout=self._run_sync_timeout,
            )
            results.update(async_results)

            # Persist new cache entries — best-effort.  If the cache write
            # fails (e.g. dead session), log and continue.  The contextualized
            # text is already in ``results`` and the pipeline can proceed.
            new_entries: list[ContextualChunkCache] = []
            for idx in miss_indices:
                if idx in cacheable_indices:
                    chunk = chunks[idx]
                    new_entries.append(ContextualChunkCache(
                        cache_key=chunk_cache_keys[idx],
                        record_version_id=chunk.record_version_id,
                        chunk_id=chunk.chunk_id,
                        chunk_text_hash=_chunk_text_hash(chunk.text),
                        model_name=self._model_name,
                        prompt_hash=self._prompt_hash,
                        context_text=results[idx],
                    ))
            if new_entries:
                try:
                    self._cache_repository.save_many(new_entries)
                except Exception:
                    logger.warning(
                        "contextualization_service.cache_write_failed",
                        entry_count=len(new_entries),
                        exc_info=True,
                    )

        # Build output preserving original order
        output: list[IndexChunk] = []
        for idx, chunk in enumerate(chunks):
            context_text = results.get(idx)
            if context_text is not None and context_text != chunk.text:
                # Build contextualized text: prepend context
                contextualized_text = f"{context_text.strip()}\n\n{chunk.text}"
                output.append(IndexChunk(
                    chunk_id=chunk.chunk_id,
                    record_version_id=chunk.record_version_id,
                    text=contextualized_text,
                    metadata=chunk.metadata,
                    is_active=chunk.is_active,
                    original_text=chunk.text,
                ))
            else:
                output.append(chunk)

        return tuple(output)

    async def _process_groups_async(
        self,
        groups: dict[str, list[int]],
        chunks: tuple[IndexChunk, ...],
        doc_texts: dict[str, str],
    ) -> tuple[dict[int, str], set[int]]:
        results: dict[int, str] = {}
        cacheable_indices: set[int] = set()

        # Flatten all work items from groups into a single ordered list.
        all_items: list[tuple[str, int]] = [
            (doc_texts[rvid], idx)
            for rvid, indices in groups.items()
            for idx in indices
        ]

        total_chunks = len(all_items)
        total_failed = 0
        semaphore = asyncio.Semaphore(self._max_concurrency)
        client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=self._max_concurrency,
                max_keepalive_connections=self._max_concurrency,
            ),
        )

        async def process_chunk(document_text: str, idx: int) -> bool:
            """Process a single chunk with retries.  Returns True on success."""
            chunk = chunks[idx]
            async with semaphore:
                last_exc: BaseException | None = None
                attempts = 0
                for attempt in range(1 + self._max_retries):
                    attempts = attempt + 1
                    try:
                        context = await call_openrouter_context_async(
                            document_text=document_text,
                            chunk_text=chunk.text,
                            model=self._model_name,
                            base_url=self._base_url,
                            api_key=self._api_key,
                            app_name=self._app_name,
                            timeout=self._timeout,
                            cache_ttl=self._cache_ttl,
                            client=client,
                        )
                        if context and context.strip():
                            results[idx] = context.strip()
                            cacheable_indices.add(idx)
                        else:
                            results[idx] = chunk.text
                        return True
                    except httpx.HTTPStatusError as exc:
                        last_exc = exc
                        if (
                            exc.response.status_code in _TRANSIENT_STATUS_CODES
                            and attempt < self._max_retries
                        ):
                            wait = _retry_delay(attempt, exc.response)
                            logger.debug(
                                "contextualization_service.retrying",
                                chunk_id=chunk.chunk_id,
                                attempt=attempts,
                                status_code=exc.response.status_code,
                                delay=round(wait, 2),
                            )
                            await asyncio.sleep(wait)
                            continue
                        break
                    except (httpx.TimeoutException, httpx.NetworkError) as exc:
                        last_exc = exc
                        if attempt < self._max_retries:
                            wait = _retry_delay(attempt)
                            logger.debug(
                                "contextualization_service.retrying",
                                chunk_id=chunk.chunk_id,
                                attempt=attempts,
                                delay=round(wait, 2),
                            )
                            await asyncio.sleep(wait)
                            continue
                        break
                    except Exception as exc:
                        last_exc = exc
                        break

                logger.warning(
                    "contextualization_service.model_call_failed",
                    chunk_id=chunk.chunk_id,
                    attempts=attempts,
                    error=str(last_exc),
                )
                results[idx] = chunk.text
                return False

        try:
            batch_size = self._batch_size
            # Warm-up: first batch is smaller to avoid a TCP connection storm
            # on a cold connection pool hitting Cloudflare's DDoS protection.
            warm_up_size = max(1, self._max_concurrency // 5)
            batches: list[list[tuple[str, int]]] = []
            if total_chunks > warm_up_size:
                batches.append(all_items[:warm_up_size])
                rest = all_items[warm_up_size:]
                for i in range(0, len(rest), batch_size):
                    batches.append(rest[i : i + batch_size])
            else:
                batches.append(all_items)
            total_batches = len(batches)

            for batch_num, batch in enumerate(batches, start=1):
                batch_tasks = [
                    process_chunk(doc_text, idx) for doc_text, idx in batch
                ]
                outcomes = await asyncio.gather(*batch_tasks)
                batch_failed = sum(1 for ok in outcomes if not ok)
                total_failed += batch_failed

                logger.info(
                    "contextualization_service.batch_complete",
                    batch=batch_num,
                    total_batches=total_batches,
                    batch_size=len(batch),
                    batch_failed=batch_failed,
                    total_failed=total_failed,
                    total_chunks=total_chunks,
                )

                # Randomised pause between batches to avoid provider-side throttling.
                if batch_num < total_batches:
                    delay = random.uniform(
                        self._inter_batch_delay_min, self._inter_batch_delay_max,
                    )
                    await asyncio.sleep(delay)
        finally:
            await client.aclose()

        return results, cacheable_indices
