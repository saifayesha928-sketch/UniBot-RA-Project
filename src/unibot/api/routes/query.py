from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, cast

import httpx
import structlog
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

from unibot.api.dependencies import (
    get_qdrant_client,
    get_session,
    serialize_generation_slim,
)
from unibot.answering.service import (
    AnswerResult,
    QueryService,
)
from unibot.db.models import ServingGeneration
from unibot.db.repositories.serving_generations import ServingGenerationRepository
from unibot.retrieval.query_classification import (
    classify_query,
    extract_secondary_hints,
    reconcile_classification,
    resolve_effective_source_class_hint,
)
from unibot.retrieval.query_cache import QueryResultCache
from unibot.retrieval.service import RetrievalService

logger = structlog.get_logger(__name__)

router = APIRouter()
_query_cache = QueryResultCache(maxsize=2048, ttl=3600)


class QueryRequest(BaseModel):
    query_text: str = Field(min_length=1, max_length=1024)
    source_class_hint: str | None = None
    limit: int = Field(default=5, ge=1, le=10)


def _serialize_answer(
    result: AnswerResult,
    *,
    generation: ServingGeneration,
) -> dict[str, Any]:
    return {
        "status": result.status,
        "answer_text": result.answer_text,
        "claims": [
            {"text": claim.text, "citation_ids": list(claim.citation_ids)}
            for claim in result.claims
        ],
        "citations": [
            {
                "citation_id": citation.citation_id,
                "chunk_id": citation.chunk_id,
                "chunk_index": citation.chunk_index,
                "chunk_count": citation.chunk_count,
                "record_version_id": citation.record_version_id,
                "source_url": citation.source_url,
                "source_locator": citation.source_locator,
            }
            for citation in result.citations
        ],
        "warnings": list(result.warnings),
        "generation": serialize_generation_slim(generation),
    }


def _out_of_scope_result(reason: str) -> AnswerResult:
    return AnswerResult(
        status="abstained",
        answer_text=(
            "I cannot answer that from the knowledge base because the request is out of scope."
        ),
        claims=(),
        citations=(),
        warnings=(reason,),
        prompt="",
    )


@dataclass(slots=True)
class _QueryContext:
    """Shared context produced by classification + rewriting orchestration."""

    classification: Any
    keyword_classification: Any
    raw_semantic: Any
    effective_source_class_hint: str | None
    retrieval_query: str
    secondary_hints: tuple[tuple[str | None, str | None], ...]
    hint_is_user_provided: bool
    live_backend: str
    rewrite_result: Any
    t_classify_start: float = 0.0
    t_classify_end: float = 0.0


async def _orchestrate_classification_and_rewrite(
    payload: QueryRequest,
    request: Request,
) -> _QueryContext:
    """Run classification + rewriting in parallel and resolve hints.

    Shared by both ``/query`` and ``/query/stream``.  The caller is
    responsible for checking ``ctx.classification.abstain_immediately``
    and handling it appropriately.
    """
    hint_is_user_provided = payload.source_class_hint is not None

    settings = getattr(request.app.state, "settings", None)
    classifier_backend = getattr(settings, "query_classifier_backend", "keyword")
    semantic_classifier = getattr(request.app.state, "semantic_classifier", None)
    query_rewriter = getattr(request.app.state, "query_rewriter", None)

    keyword_classification = classify_query(payload.query_text)

    need_semantic = classifier_backend == "semantic" and semantic_classifier is not None
    need_rewrite = query_rewriter is not None

    async def _classify_async() -> tuple[
        Any, tuple[tuple[str | None, str | None], ...]
    ]:
        if not need_semantic or semantic_classifier is None:
            return None, ()
        classify_multi = getattr(semantic_classifier, "classify_with_secondary", None)
        if callable(classify_multi):
            return cast(
                tuple[Any, tuple[tuple[str | None, str | None], ...]],
                await asyncio.to_thread(classify_multi, payload.query_text),
            )
        result = await asyncio.to_thread(
            semantic_classifier.classify, payload.query_text
        )
        return result, extract_secondary_hints(payload.query_text, primary=result)

    async def _rewrite_async() -> Any:
        if not need_rewrite or query_rewriter is None:
            return None
        async_rw = getattr(query_rewriter, "async_rewrite", None)
        if async_rw is not None:
            return await async_rw(payload.query_text)
        return await asyncio.to_thread(query_rewriter.rewrite, payload.query_text)

    t_classify_start = time.monotonic()
    (raw_semantic, semantic_secondary_hints), rewrite_result = await asyncio.gather(
        _classify_async(), _rewrite_async()
    )

    secondary_hints: tuple[tuple[str | None, str | None], ...] = ()
    if raw_semantic is not None:
        secondary_hints = semantic_secondary_hints
        classification = reconcile_classification(keyword_classification, raw_semantic)
        live_backend = "semantic"
    else:
        classification = keyword_classification
        live_backend = "keyword"
    t_classify_end = time.monotonic()

    # Resolve effective hints and secondary intents.
    effective_source_class_hint = resolve_effective_source_class_hint(
        classified_hint=classification.source_class_hint,
        requested_hint=payload.source_class_hint,
    )

    if live_backend == "keyword":
        secondary_hints = extract_secondary_hints(
            payload.query_text, primary=classification
        )

    # Apply rewrite result.
    retrieval_query = payload.query_text
    if rewrite_result is not None:
        if rewrite_result.rewritten_query != rewrite_result.original_query:
            retrieval_query = rewrite_result.rewritten_query

    return _QueryContext(
        classification=classification,
        keyword_classification=keyword_classification,
        raw_semantic=raw_semantic,
        effective_source_class_hint=effective_source_class_hint,
        retrieval_query=retrieval_query,
        secondary_hints=secondary_hints,
        hint_is_user_provided=hint_is_user_provided,
        live_backend=live_backend,
        rewrite_result=rewrite_result,
        t_classify_start=t_classify_start,
        t_classify_end=t_classify_end,
    )


@router.post("/query")
async def query_records(payload: QueryRequest, request: Request) -> dict[str, Any]:
    t_start = time.monotonic()
    session, close_session = get_session(request)
    try:
        t_gen_lookup_start = time.monotonic()
        generation = ServingGenerationRepository(session).get_active_generation()
        t_gen_lookup_end = time.monotonic()
        if generation is None:
            raise HTTPException(
                status_code=503,
                detail="No active serving generation is available.",
            )

        # Early cache check — avoids remote classifier and rewriter calls on repeat queries.
        cached_early = _query_cache.get_early(
            query_text=payload.query_text,
            generation_id=generation.generation_id,
            limit=payload.limit,
            user_hint=payload.source_class_hint,
        )
        if cached_early is not None:
            t_cache_hit = time.monotonic()
            logger.info(
                "query.cache_hit_early",
                generation_id=generation.generation_id,
            )
            logger.info(
                "query.cache_hit",
                generation_id=generation.generation_id,
                source_class_hint=payload.source_class_hint,
                record_type_hint=None,
                early=True,
            )
            logger.info(
                "query.latency_ms",
                classify_ms=0.0,
                rewrite_ms=0.0,
                cache_hit=True,
                total_ms=round((t_cache_hit - t_start) * 1000, 1),
            )
            return cached_early  # type: ignore[no-any-return]

        ctx = await _orchestrate_classification_and_rewrite(payload, request)
        classification = ctx.classification
        t_pre_classify = ctx.t_classify_start
        t_post_classify = ctx.t_classify_end
        # Both classify and rewrite start concurrently via gather.
        t_pre_rewrite = t_pre_classify

        logger.info(
            "query.classified",
            backend=ctx.live_backend,
            keyword_source_hint=ctx.keyword_classification.source_class_hint,
            keyword_record_type_hint=ctx.keyword_classification.record_type_hint,
            semantic_source_hint=ctx.raw_semantic.source_class_hint
            if ctx.raw_semantic is not None
            else None,
            semantic_record_type_hint=ctx.raw_semantic.record_type_hint
            if ctx.raw_semantic is not None
            else None,
            final_source_hint=classification.source_class_hint,
            final_record_type_hint=classification.record_type_hint,
            guardrail_active=ctx.raw_semantic is not None
            and classification is not ctx.raw_semantic,
            abstain_immediately=classification.abstain_immediately,
            parallel_rewrite=ctx.rewrite_result is not None,
        )

        settings = getattr(request.app.state, "settings", None)
        shadow_mode = getattr(settings, "query_classifier_shadow_mode", False)
        semantic_classifier = getattr(request.app.state, "semantic_classifier", None)
        if shadow_mode and semantic_classifier is not None:
            try:
                if ctx.live_backend == "keyword":
                    shadow_result = await asyncio.to_thread(
                        semantic_classifier.classify, payload.query_text
                    )
                    shadow_backend = "semantic"
                else:
                    shadow_result = ctx.keyword_classification
                    shadow_backend = "keyword"
                diverged = (
                    shadow_result.query_class != classification.query_class
                    or shadow_result.source_class_hint
                    != classification.source_class_hint
                    or shadow_result.record_type_hint != classification.record_type_hint
                )
                logger.info(
                    "query.shadow_classification",
                    live_backend=ctx.live_backend,
                    shadow_backend=shadow_backend,
                    live_query_class=str(classification.query_class),
                    shadow_query_class=str(shadow_result.query_class),
                    live_source_class_hint=classification.source_class_hint,
                    shadow_source_class_hint=shadow_result.source_class_hint,
                    live_record_type_hint=classification.record_type_hint,
                    shadow_record_type_hint=shadow_result.record_type_hint,
                    diverged=diverged,
                )
            except Exception:
                logger.warning(
                    "query.shadow_classification_failed",
                    exc_info=True,
                )

        if classification.abstain_immediately:
            serialized = _serialize_answer(
                _out_of_scope_result(classification.reason),
                generation=generation,
            )
            _query_cache.put_dual(
                query_text=payload.query_text,
                generation_id=generation.generation_id,
                limit=payload.limit,
                user_hint=payload.source_class_hint,
                source_class_hint=payload.source_class_hint,
                record_type_hint=None,
                hint_is_user_provided=ctx.hint_is_user_provided,
                retrieval_query=None,
                result=serialized,
            )
            return serialized

        logger.info(
            "query.hints_resolved",
            effective_source_class_hint=ctx.effective_source_class_hint,
            record_type_hint=classification.record_type_hint,
        )

        if ctx.secondary_hints:
            logger.info(
                "query.secondary_intents",
                count=len(ctx.secondary_hints),
                hints=ctx.secondary_hints,
                backend=ctx.live_backend,
            )

        if ctx.rewrite_result is not None and ctx.retrieval_query != payload.query_text:
            logger.debug(
                "query.rewritten",
                rewritten_differs=True,
            )
        t_post_rewrite = time.monotonic()

        # Check cache before expensive retrieval
        cached_result = _query_cache.get(
            query_text=payload.query_text,
            retrieval_query=ctx.retrieval_query,
            generation_id=generation.generation_id,
            source_class_hint=ctx.effective_source_class_hint,
            record_type_hint=classification.record_type_hint,
            hint_is_user_provided=ctx.hint_is_user_provided,
            limit=payload.limit,
        )
        t_post_cache = time.monotonic()
        if cached_result is not None:
            logger.info(
                "query.cache_hit",
                generation_id=generation.generation_id,
                source_class_hint=ctx.effective_source_class_hint,
                record_type_hint=classification.record_type_hint,
            )
            logger.info(
                "query.latency_ms",
                classify_ms=round((t_post_classify - t_pre_classify) * 1000, 1),
                rewrite_ms=round((t_post_rewrite - t_pre_rewrite) * 1000, 1),
                cache_hit=True,
                total_ms=round((t_post_cache - t_start) * 1000, 1),
            )
            return cached_result  # type: ignore[no-any-return]

        logger.info(
            "query.cache_miss",
            generation_id=generation.generation_id,
            source_class_hint=ctx.effective_source_class_hint,
            record_type_hint=classification.record_type_hint,
        )

        query_service = QueryService(
            retrieval_service=_build_retrieval_service(request, session),
            answering_service=request.app.state.answering_service,
        )
        try:
            retrieval_query = (
                ctx.retrieval_query
                if ctx.retrieval_query != payload.query_text
                else None
            )
            async_method = getattr(query_service, "async_answer_query", None)
            if async_method is not None:
                answer_result = await async_method(
                    payload.query_text,
                    retrieval_query=retrieval_query,
                    active_generation=generation,
                    source_class_hint=ctx.effective_source_class_hint,
                    record_type_hint=classification.record_type_hint,
                    hint_is_user_provided=ctx.hint_is_user_provided,
                    limit=payload.limit,
                    secondary_hints=ctx.secondary_hints,
                )
            else:
                               answer_result = await asyncio.to_thread(
                    query_service.answer_query,
                    payload.query_text,
                    retrieval_query=retrieval_query,
                    active_generation=generation,
                    source_class_hint=ctx.effective_source_class_hint,
                    record_type_hint=classification.record_type_hint,
                    hint_is_user_provided=ctx.hint_is_user_provided,
                    limit=payload.limit,
                    secondary_hints=ctx.secondary_hints,
                )
        except (httpx.HTTPError, RuntimeError):
            raise HTTPException(
                status_code=503,
                detail="A provider dependency is temporarily unavailable. Please try again later.",
            )
    
        t_post_answer = time.monotonic()
        logger.info(
            "query.latency_ms",
            gen_lookup_ms=round((t_gen_lookup_end - t_gen_lookup_start) * 1000, 1),
            classify_ms=round((t_post_classify - t_pre_classify) * 1000, 1),
            rewrite_ms=round((t_post_rewrite - t_pre_rewrite) * 1000, 1),
            cache_hit=False,
            retrieval_and_answer_ms=round((t_post_answer - t_post_cache) * 1000, 1),
            total_ms=round((t_post_answer - t_start) * 1000, 1),
        )
        serialized = _serialize_answer(answer_result, generation=generation)
        _query_cache.put_dual(
            query_text=payload.query_text,
            generation_id=generation.generation_id,
            limit=payload.limit,
            user_hint=payload.source_class_hint,
            source_class_hint=ctx.effective_source_class_hint,
            record_type_hint=classification.record_type_hint,
            hint_is_user_provided=ctx.hint_is_user_provided,
            retrieval_query=ctx.retrieval_query,
            result=serialized,
        )
        return serialized
    finally:
        if close_session:
            session.close()


def _sse_event(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _build_retrieval_service(
    request: Request,
    session: Any,
) -> RetrievalService:
    settings = getattr(request.app.state, "settings", None)
    return RetrievalService(
        session=session,
        qdrant_client=get_qdrant_client(request),
        embedding_provider=request.app.state.embedding_provider,
        reranker=request.app.state.reranker,
        min_relevance_score=float(
            getattr(settings, "retrieval_min_relevance_score", 0.0)
        ),
        candidate_multiplier=int(
            getattr(settings, "retrieval_candidate_multiplier", 5)
        ),
        candidate_floor=int(getattr(settings, "retrieval_candidate_floor", 50)),
        route_planning_enabled=bool(
            getattr(settings, "retrieval_route_planning_enabled", True)
        ),
    )


@router.post("/query/stream")
async def query_records_stream(
    payload: QueryRequest, request: Request
) -> StreamingResponse:
    async def _generate() -> AsyncIterator[str]:
        session, close_session = get_session(request)
        try:
            generation = ServingGenerationRepository(session).get_active_generation()
            if generation is None:
                yield _sse_event("error", {"detail": "No active serving generation."})
                return

            # Early cache check — return cached result immediately if available.
            cached_early = _query_cache.get_early(
                query_text=payload.query_text,
                generation_id=generation.generation_id,
                limit=payload.limit,
                user_hint=payload.source_class_hint,
            )
            if cached_early is not None:
                yield _sse_event("result", cached_early)
                return

            yield _sse_event("progress", {"stage": "classifying"})

            ctx = await _orchestrate_classification_and_rewrite(payload, request)

            if ctx.classification.abstain_immediately:
                serialized = _serialize_answer(
                    _out_of_scope_result(ctx.classification.reason),
                    generation=generation,
                )
                _query_cache.put_dual(
                    query_text=payload.query_text,
                    generation_id=generation.generation_id,
                    limit=payload.limit,
                    user_hint=payload.source_class_hint,
                    source_class_hint=payload.source_class_hint,
                    record_type_hint=None,
                    hint_is_user_provided=ctx.hint_is_user_provided,
                    retrieval_query=None,
                    result=serialized,
                )
                yield _sse_event(
                    "result",
                    serialized,
                )
                return

            yield _sse_event("progress", {"stage": "retrieving"})

            retrieval_service = _build_retrieval_service(request, session)
            evidence = retrieval_service.retrieve(
                ctx.retrieval_query,
                active_generation=generation,
                source_class_hint=ctx.effective_source_class_hint,
                record_type_hint=ctx.classification.record_type_hint,
                hint_is_user_provided=ctx.hint_is_user_provided,
                limit=payload.limit,
                secondary_hints=ctx.secondary_hints,
            )

            yield _sse_event("progress", {"stage": "answering"})

            answering_service = request.app.state.answering_service
            answer_result = await answering_service.async_answer(
                payload.query_text, evidence
            )

            serialized = _serialize_answer(answer_result, generation=generation)
            _query_cache.put_dual(
                query_text=payload.query_text,
                generation_id=generation.generation_id,
                limit=payload.limit,
                user_hint=payload.source_class_hint,
                source_class_hint=ctx.effective_source_class_hint,
                record_type_hint=ctx.classification.record_type_hint,
                hint_is_user_provided=ctx.hint_is_user_provided,
                retrieval_query=ctx.retrieval_query,
                result=serialized,
            )
            yield _sse_event("result", serialized)

        except (httpx.HTTPError, RuntimeError):
            yield _sse_event(
                "error",
                {"detail": "A provider dependency is temporarily unavailable."},
            )
        finally:
            if close_session:
                session.close()

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
