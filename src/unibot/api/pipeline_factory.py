"""Shared query runtime and pipeline construction for API and eval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import structlog

from unibot.answering.model_adapter import create_answer_model
from unibot.answering.service import AnsweringService, QueryService
from unibot.indexing.provider_factory import create_embedding_provider
from unibot.retrieval.reranker import create_reranker
from unibot.retrieval.service import RetrievalService
from unibot.settings import Settings

logger = structlog.get_logger(__name__)


@dataclass
class QueryRuntimeDeps:
    embedding_provider: Any
    reranker: Any
    answering_service: AnsweringService
    semantic_classifier: Any
    query_rewriter: Any
    classifier_backend: str
    cleanup: Any
    async_cleanup: Any = None


@dataclass
class QueryPipelineDeps:
    query_service: QueryService
    semantic_classifier: Any
    query_rewriter: Any
    classifier_backend: str
    cleanup: Any


def build_query_runtime(settings: Settings) -> QueryRuntimeDeps:
    """Build shared long-lived runtime dependencies for query serving/eval."""
    from unibot.answering.grounding import create_grounding_verifier
    from unibot.http_clients import build_provider_http_clients
    from unibot.retrieval.query_rewriter import create_query_rewriter
    from unibot.settings import (
        resolve_answer_model_backend,
        resolve_embedding_dense_backend,
    )

    shared_clients = build_provider_http_clients(settings)

    embedding_provider = create_embedding_provider(
        settings=settings,
        client=shared_clients.cohere,
    )
    reranker = create_reranker(
        settings=settings,
        client=shared_clients.cohere,
    )
    primary_backend = resolve_answer_model_backend(settings)
    fallback_backend = getattr(settings, "answer_model_fallback_backend", None)

    # Resolve which HTTP client to pass for primary and fallback.
    primary_client = (
        shared_clients.openrouter
        if primary_backend == "openrouter"
        else shared_clients.cohere
    )
    async_primary_client = (
        shared_clients.async_openrouter
        if primary_backend == "openrouter"
        else shared_clients.async_cohere
    )
    fallback_client = None
    async_fallback_client = None
    if fallback_backend is not None:
        fallback_client = (
            shared_clients.openrouter
            if fallback_backend == "openrouter"
            else shared_clients.cohere
        )
        async_fallback_client = (
            shared_clients.async_openrouter
            if fallback_backend == "openrouter"
            else shared_clients.async_cohere
        )

    answer_model = create_answer_model(
        settings=settings,
        client=primary_client,
        fallback_client=fallback_client,
        async_client=async_primary_client,
        async_fallback_client=async_fallback_client,
    )
    grounding_verifier = create_grounding_verifier(settings=settings)
    answering_service = AnsweringService(
        answer_model=answer_model,
        grounding_verifier=grounding_verifier,
        grounding_skip_low_risk=bool(
            getattr(settings, "grounding_skip_low_risk", False)
        ),
        grounding_confidence_threshold=float(
            getattr(settings, "grounding_confidence_threshold", 0.5)
        ),
    )

    classifier_backend = getattr(settings, "query_classifier_backend", "keyword")
    shadow_mode = getattr(settings, "query_classifier_shadow_mode", False)
    semantic_classifier = None
    if classifier_backend == "semantic" or shadow_mode:
        if resolve_embedding_dense_backend(settings) != "cohere":
            logger.warning(
                "semantic classifier requires Cohere dense embeddings; "
                "falling back to keyword classifier"
            )
        else:
            from unibot.retrieval.semantic_classifier import SemanticQueryClassifier

            dense_provider = embedding_provider._dense_provider  # type: ignore[attr-defined]
            semantic_classifier = SemanticQueryClassifier(
                dense_embedding_provider=dense_provider,
                threshold=getattr(settings, "semantic_classifier_threshold", 0.6),
            )

    query_rewriter = create_query_rewriter(
        settings=settings,
        client=shared_clients.openrouter,
    )

    return QueryRuntimeDeps(
        embedding_provider=embedding_provider,
        reranker=reranker,
        answering_service=answering_service,
        semantic_classifier=semantic_classifier,
        query_rewriter=query_rewriter,
        classifier_backend=classifier_backend,
        cleanup=shared_clients.close,
        async_cleanup=shared_clients.aclose,
    )


def build_query_pipeline(
    settings: Settings,
    session: Any,
    qdrant_client: Any,
) -> QueryPipelineDeps:
    """Build QueryService and supporting pipeline deps for eval/CLI paths."""
    runtime = build_query_runtime(settings)

    retrieval_service = RetrievalService(
        session=session,
        qdrant_client=qdrant_client,
        embedding_provider=runtime.embedding_provider,
        reranker=runtime.reranker,
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

    query_service = QueryService(
        retrieval_service=retrieval_service,
        answering_service=runtime.answering_service,
    )

    return QueryPipelineDeps(
        query_service=query_service,
        semantic_classifier=runtime.semantic_classifier,
        query_rewriter=runtime.query_rewriter,
        classifier_backend=runtime.classifier_backend,
        cleanup=runtime.cleanup,
    )
