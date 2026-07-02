from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import structlog
from qdrant_client import QdrantClient, models
from sqlalchemy import select
from sqlalchemy.orm import Session

from unibot.db.models import CanonicalRecord, ServingGeneration, SourceRegistry
from unibot.domain.record_payloads import extract_primary_content
from unibot.verify.value_identity import value_hash_for_stored_record
from unibot.indexing.embeddings import (
    DenseSparseEmbeddingProvider,
    EmbeddingVector,
    embed_query_text,
    embed_query_texts_batch,
)
from unibot.retrieval.filters import (
    enforce_source_diversity,
    expand_source_classes,
    payload_matches_active_generation,
    record_is_current_only,
    select_primary_hits,
    select_with_type_reservation,
    source_class_matches,
    source_is_query_allowed,
)
from unibot.retrieval.query_classification import normalize_source_class_hint
from unibot.retrieval.query_expansion import expand_query_with_synonyms
from unibot.retrieval.reranker import RerankCandidate, Reranker, TokenOverlapReranker

logger = structlog.get_logger(__name__)

DEFAULT_MAX_EVIDENCE_PER_SOURCE = 1
DEFAULT_RESERVED_TYPE_SLOTS = 3


@dataclass(frozen=True, slots=True)
class RetrievedEvidence:
    record_version_id: str
    record_id: str
    record_type: str
    source_class: str
    source_url: str
    source_locator: str
    source_authority_tier: int
    conflict_scope_id: str
    dedupe_key: str
    value_hash: str
    freshness_status: str
    year_confidence: str
    cycle_label: str | None
    content: str
    score: float
    chunk_id: str = ""
    chunk_index: int = 0
    chunk_count: int = 1
    context_window: str = ""  # Adjacent chunk text for LLM context; NOT included in citations.
    contextualized_text: str = ""  # Enriched text from contextual indexing; used for reranking.


@dataclass(frozen=True, slots=True)
class _RetrievalRoute:
    label: str
    source_class_hint: str | None = None
    record_type_hint: str | None = None


DEFAULT_MIN_RELEVANCE_SCORE = 0.0  # Safe fallback for direct callers; production uses settings.py (0.1).
MAX_SECONDARY_ROUTES = 2  # Cap secondary intent routes to bound Qdrant query fan-out.


def filter_below_min_score(
    evidence: tuple[RetrievedEvidence, ...],
    *,
    min_score: float,
) -> tuple[RetrievedEvidence, ...]:
    if min_score <= 0.0:
        return evidence
    return tuple(item for item in evidence if item.score >= min_score)


_AUTHORITY_DECAY: dict[int, float] = {1: 1.0, 2: 0.99, 3: 0.97, 4: 0.95, 5: 0.92}


def apply_authority_decay(
    evidence: tuple[RetrievedEvidence, ...],
) -> tuple[RetrievedEvidence, ...]:
    """Apply a soft score penalty based on source authority tier.

    Higher-authority sources (lower tier number) keep more of their rerank
    score.  The decay is conservative: a tier-5 source needs ~8% higher
    raw relevance to outrank a tier-1 source with the same content.
    """
    adjusted = tuple(
        replace(
            item,
            score=item.score * _AUTHORITY_DECAY.get(item.source_authority_tier, 0.92),
        )
        for item in evidence
    )
    return tuple(sorted(adjusted, key=lambda e: (-e.score, e.record_version_id)))


def expand_chunk_context(
    evidence: RetrievedEvidence,
    sibling_texts: dict[int, str],
) -> RetrievedEvidence:
    """Attach adjacent chunk texts as supplementary context.

    Sibling text is stored in ``context_window`` — a field that the prompt
    builder renders as surrounding context but that is NOT included in
    ``Citation.content``.  This preserves citation integrity: claims are
    grounded against the directly-cited chunk only.
    """
    if evidence.chunk_count <= 1 or not sibling_texts:
        return evidence

    parts: list[str] = []
    for idx in sorted(sibling_texts.keys()):
        parts.append(sibling_texts[idx])

    return replace(evidence, context_window="\n\n".join(parts))


class RetrievalService:
    def __init__(
        self,
        *,
        session: Session,
        qdrant_client: QdrantClient,
        embedding_provider: DenseSparseEmbeddingProvider | None = None,
        reranker: Reranker | None = None,
        max_evidence_per_source: int = DEFAULT_MAX_EVIDENCE_PER_SOURCE,
        reserved_type_slots: int = DEFAULT_RESERVED_TYPE_SLOTS,
        min_relevance_score: float = DEFAULT_MIN_RELEVANCE_SCORE,
        candidate_multiplier: int = 5,
        candidate_floor: int = 50,
        route_planning_enabled: bool = True,
    ) -> None:
        self._session = session
        self._qdrant_client = qdrant_client
        self._embedding_provider = embedding_provider
        self._reranker = reranker or TokenOverlapReranker()
        self._max_evidence_per_source = max(0, max_evidence_per_source)
        self._reserved_type_slots = max(0, reserved_type_slots)
        self._min_relevance_score = max(0.0, min_relevance_score)
        self._candidate_multiplier = max(1, candidate_multiplier)
        self._candidate_floor = max(1, candidate_floor)
        self._route_planning_enabled = route_planning_enabled

    def retrieve(
        self,
        query_text: str,
        *,
        active_generation: ServingGeneration,
        source_class_hint: str | None = None,
        record_type_hint: str | None = None,
        hint_is_user_provided: bool = False,
        limit: int = 3,
        secondary_hints: tuple[tuple[str | None, str | None], ...] = (),
    ) -> tuple[RetrievedEvidence, ...]:
        source_class_hint = normalize_source_class_hint(source_class_hint)

        can_relax_classifier_hint = (
            source_class_hint is not None
            and not hint_is_user_provided
        )

        if self._route_planning_enabled and (
            record_type_hint is not None or can_relax_classifier_hint or secondary_hints
        ):
            return self._retrieve_route_planned(
                query_text,
                active_generation=active_generation,
                source_class_hint=source_class_hint,
                record_type_hint=record_type_hint,
                hint_is_user_provided=hint_is_user_provided,
                limit=limit,
                secondary_hints=secondary_hints,
            )

        return self._retrieve_adaptive(
            query_text,
            active_generation=active_generation,
            source_class_hint=source_class_hint,
            record_type_hint=record_type_hint,
            can_relax_classifier_hint=can_relax_classifier_hint,
            limit=limit,
        )

    def _retrieve_route_planned(
        self,
        query_text: str,
        *,
        active_generation: ServingGeneration,
        source_class_hint: str | None,
        record_type_hint: str | None,
        hint_is_user_provided: bool,
        limit: int,
        secondary_hints: tuple[tuple[str | None, str | None], ...] = (),
    ) -> tuple[RetrievedEvidence, ...]:
        """Query planned routes, merge, then rerank once over the unified set."""
        routes = _build_route_plan(
            source_class_hint=source_class_hint,
            record_type_hint=record_type_hint,
            hint_is_user_provided=hint_is_user_provided,
        )

        # Inject secondary intent routes for multi-intent queries.
        if secondary_hints:
            extended = list(routes)
            for src_hint, rec_hint in secondary_hints[:MAX_SECONDARY_ROUTES]:
                extended.append(
                    _RetrievalRoute(
                        label="secondary_intent",
                        source_class_hint=src_hint,
                        record_type_hint=rec_hint,
                    )
                )
            routes = _dedupe_routes(extended)

        # Precompute embeddings for all expanded query variants once,
        # then reuse across routes to avoid redundant API calls.
        t_embed_start = time.monotonic()
        precomputed_vectors: dict[str, EmbeddingVector] | None = None
        if self._embedding_provider is not None:
            expanded_queries = expand_query_with_synonyms(query_text)
            precomputed_vectors = embed_query_texts_batch(
                self._embedding_provider, list(expanded_queries)
            )
        t_embed_end = time.monotonic()

        t_search_start = time.monotonic()
        route_counts, merged_points = self._batch_query_routes(
            query_text,
            routes,
            active_generation=active_generation,
            limit=limit,
            precomputed_vectors=precomputed_vectors,
        )
        batch_search_ms = round((time.monotonic() - t_search_start) * 1000, 1)

        # Single DB hydration pass over merged points from all routes.
        # Build a union of allowed source classes from all routes to preserve
        # the DB-side safety filter. If any route has no hint (global route),
        # all source classes are allowed.
        route_source_hints = [r.source_class_hint for r in routes]
        allowed_source_classes: frozenset[str] | None = None
        if all(h is not None for h in route_source_hints):
            combined: set[str] = set()
            for h in route_source_hints:
                combined.update(expand_source_classes(h))  # type: ignore[arg-type]
            allowed_source_classes = frozenset(combined)

        t_hydrate_start = time.monotonic()
        evidence = self._hydrate_and_filter(
            tuple(merged_points),
            active_generation=active_generation,
            source_class_hint=None,
            allowed_source_classes=allowed_source_classes,
        )
        t_hydrate_end = time.monotonic()

        logger.info(
            "retrieval.route_plan",
            route_labels=[route.label for route in routes],
            route_counts=route_counts,
            batch_search_ms=batch_search_ms,
            merged_count=len(evidence),
            embedding_ms=round((t_embed_end - t_embed_start) * 1000, 1),
            hydrate_ms=round((t_hydrate_end - t_hydrate_start) * 1000, 1),
            source_class_hint=source_class_hint,
            record_type_hint=record_type_hint,
            hint_is_user_provided=hint_is_user_provided,
        )

        if not evidence:
            logger.info(
                "retrieval.completed",
                final_hit_count=0,
                reason="no_evidence_after_route_planning",
                source_class_hint=source_class_hint,
                record_type_hint=record_type_hint,
            )
            return ()

        t_rerank_start = time.monotonic()
        final_hits, empty_reason = self._rerank_select_and_expand(
            query_text,
            evidence=evidence,
            active_generation=active_generation,
            record_type_hint=record_type_hint,
            source_class_hint=source_class_hint,
            limit=limit,
        )
        t_rerank_end = time.monotonic()

        if not final_hits:
            logger.info(
                "retrieval.completed",
                final_hit_count=0,
                reason=empty_reason or "no_results_after_selection",
                min_relevance_score=self._min_relevance_score,
                source_class_hint=source_class_hint,
                record_type_hint=record_type_hint,
            )
            return ()

        top_scores = sorted((item.score for item in final_hits), reverse=True)[:5]
        logger.info(
            "retrieval.completed",
            final_hit_count=len(final_hits),
            top_scores=top_scores,
            rerank_and_expand_ms=round((t_rerank_end - t_rerank_start) * 1000, 1),
            source_class_hint=source_class_hint,
            record_type_hint=record_type_hint,
            route_planned=True,
        )
        return final_hits

    def _retrieve_adaptive(
        self,
        query_text: str,
        *,
        active_generation: ServingGeneration,
        source_class_hint: str | None,
        record_type_hint: str | None,
        can_relax_classifier_hint: bool,
        limit: int,
    ) -> tuple[RetrievedEvidence, ...]:
        """Original adaptive retry logic: retry unhinted on empty/low results."""
        hint_relaxed = False

        evidence = self._query_and_filter(
            query_text,
            active_generation=active_generation,
            source_class_hint=source_class_hint,
            record_type_hint=record_type_hint,
            limit=limit,
        )

        if not evidence and can_relax_classifier_hint:
            logger.info(
                "retrieval.adaptive_retry",
                reason="empty_results_with_classifier_hint",
                original_hint=source_class_hint,
            )
            hint_relaxed = True
            evidence = self._query_and_filter(
                query_text,
                active_generation=active_generation,
                source_class_hint=None,
                record_type_hint=record_type_hint,
                limit=limit,
            )

        if not evidence:
            logger.info(
                "retrieval.completed",
                final_hit_count=0,
                reason="no_evidence_after_filtering",
                source_class_hint=source_class_hint,
                record_type_hint=record_type_hint,
            )
            return ()

        final_hits, empty_reason = self._rerank_select_and_expand(
            query_text,
            evidence=evidence,
            active_generation=active_generation,
            record_type_hint=record_type_hint,
            source_class_hint=source_class_hint,
            limit=limit,
        )
        if (
            not final_hits
            and empty_reason == "all_below_min_relevance_score"
            and can_relax_classifier_hint
            and not hint_relaxed
        ):
            logger.info(
                "retrieval.adaptive_retry",
                reason="low_relevance_with_classifier_hint",
                original_hint=source_class_hint,
                min_relevance_score=self._min_relevance_score,
            )
            hint_relaxed = True
            relaxed_evidence = self._query_and_filter(
                query_text,
                active_generation=active_generation,
                source_class_hint=None,
                record_type_hint=record_type_hint,
                limit=limit,
            )
            if relaxed_evidence:
                final_hits, empty_reason = self._rerank_select_and_expand(
                    query_text,
                    evidence=relaxed_evidence,
                    active_generation=active_generation,
                    record_type_hint=record_type_hint,
                    source_class_hint=source_class_hint,
                    limit=limit,
                )

        if not final_hits:
            logger.info(
                "retrieval.completed",
                final_hit_count=0,
                reason=empty_reason or "no_results_after_selection",
                min_relevance_score=self._min_relevance_score,
                source_class_hint=source_class_hint,
                record_type_hint=record_type_hint,
                hint_relaxed=hint_relaxed,
            )
            return ()

        top_scores = sorted((item.score for item in final_hits), reverse=True)[:5]

        logger.info(
            "retrieval.completed",
            final_hit_count=len(final_hits),
            top_scores=top_scores,
            source_class_hint=source_class_hint,
            record_type_hint=record_type_hint,
            hint_relaxed=hint_relaxed,
        )
        return final_hits

    def _rerank_select_and_expand(
        self,
        query_text: str,
        *,
        evidence: tuple[RetrievedEvidence, ...],
        active_generation: ServingGeneration,
        record_type_hint: str | None,
        source_class_hint: str | None = None,
        limit: int,
    ) -> tuple[tuple[RetrievedEvidence, ...], str | None]:
        # Use the most-specific synonym-expanded query for reranking.
        # Synonym expansion can produce variants that are closer to the
        # source text than the original query (e.g. "attendance" →
        # "class attendance"), improving reranker precision.
        expanded = expand_query_with_synonyms(query_text)
        rerank_query = max(expanded, key=len)

        # Qdrant RRF is candidate generation only. Final ranking comes from the reranker.
        score_map = {
            scored_item.record_version_id: scored_item.score
            for scored_item in self._reranker.rerank(
                rerank_query,
                [
                    RerankCandidate(
                        record_version_id=item.chunk_id,
                        text="\n".join(
                            part for part in (
                                item.contextualized_text or item.content,
                                item.cycle_label or "",
                            ) if part
                        ),
                    )
                    for item in evidence
                ],
            )
        }
        reranked = tuple(
            replace(item, score=score_map.get(item.chunk_id, 0.0))
            for item in evidence
        )

        # Query-time authority adjustment for contextually authoritative sources.
        if source_class_hint:
            from unibot.retrieval.authority_overrides import adjusted_authority_tier
            reranked = tuple(
                replace(
                    item,
                    source_authority_tier=adjusted_authority_tier(
                        item.source_authority_tier, item.source_url, source_class_hint,
                    ),
                )
                for item in reranked
            )

        # Apply authority decay before selection so canonical sources rank higher.
        reranked = apply_authority_decay(reranked)

        # Relevance gate: remove evidence below minimum score threshold.
        reranked = filter_below_min_score(reranked, min_score=self._min_relevance_score)
        if not reranked:
            return (), "all_below_min_relevance_score"

        primary_hits = select_primary_hits(reranked)
        diverse_hits = enforce_source_diversity(
            primary_hits,
            max_per_source=self._max_evidence_per_source,
            target_size=max(limit, self._reserved_type_slots),
        )
        final_hits = select_with_type_reservation(
            diverse_hits,
            record_type_hint=record_type_hint,
            limit=limit,
            reserved_slots=self._reserved_type_slots,
        )

        # Expand multi-chunk evidence with adjacent chunks for richer generation context.
        if any(item.chunk_count > 1 for item in final_hits):
            final_hits = self._expand_evidence_context(
                final_hits,
                collection_name=active_generation.qdrant_collection,
            )

        return final_hits, None

    def _batch_query_routes(
        self,
        query_text: str,
        routes: tuple[_RetrievalRoute, ...],
        *,
        active_generation: ServingGeneration,
        limit: int,
        precomputed_vectors: dict[str, EmbeddingVector] | None,
    ) -> tuple[dict[str, int], list[Any]]:
        """Batch vector search for all routes in one network round-trip.

        Builds one ``QueryRequest`` per (route, expanded_query) pair and
        submits them all via ``query_batch_points()``.  Responses are
        regrouped by route and RRF-merged per route, then deduped across
        routes.

        Returns ``(route_counts, merged_points)`` where merged_points
        are deduped across routes by point ID.
        """
        if self._embedding_provider is None or precomputed_vectors is None:
            raise RuntimeError(
                "Batch route search requires precomputed embedding vectors."
            )

        expanded_queries = expand_query_with_synonyms(query_text)
        query_limit = max(limit * self._candidate_multiplier, self._candidate_floor)

        # Build one QueryRequest per (route, expanded_query) pair.
        requests: list[models.QueryRequest] = []
        request_tags: list[tuple[int, int]] = []  # (route_idx, query_idx)

        for route_idx, route in enumerate(routes):
            route_filter = _build_query_filter(
                active_generation_id=active_generation.generation_id,
                source_class_hint=route.source_class_hint,
                record_type_hint=route.record_type_hint,
            )
            for query_idx, eq in enumerate(expanded_queries):
                vectors = precomputed_vectors[eq]
                requests.append(
                    models.QueryRequest(
                        prefetch=[
                            models.Prefetch(
                                using="dense",
                                query=list(vectors.dense_vector),
                                limit=query_limit,
                            ),
                            models.Prefetch(
                                using="sparse",
                                query=models.SparseVector(
                                    indices=list(vectors.sparse_vector.indices),
                                    values=list(vectors.sparse_vector.values),
                                ),
                                limit=query_limit,
                            ),
                        ],
                        query=models.FusionQuery(fusion=models.Fusion.RRF),
                        filter=route_filter,
                        with_payload=True,
                        with_vector=False,
                        limit=query_limit,
                    )
                )
                request_tags.append((route_idx, query_idx))

        # Single network round-trip for all routes × expanded queries.
        try:
            batch_responses = self._qdrant_client.query_batch_points(
                collection_name=active_generation.qdrant_collection,
                requests=requests,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Batch vector search failed for {len(requests)} requests; "
                "embedding provider or Qdrant may be unavailable."
            ) from exc

        # Regroup responses by route_idx.
        per_route: dict[int, list[tuple[int, list[Any]]]] = {}
        for (route_idx, query_idx), response in zip(
            request_tags, batch_responses, strict=True
        ):
            per_route.setdefault(route_idx, []).append(
                (query_idx, list(response.points))
            )

        # Per-route RRF merge across expanded queries, then dedup across routes.
        _RRF_K = 60
        route_counts: dict[str, int] = {}
        seen_point_ids: set[str] = set()
        merged_points: list[Any] = []

        for route_idx, route in enumerate(routes):
            responses = per_route.get(route_idx, [])
            route_points: dict[str, Any] = {}

            if len(responses) == 1:
                for point in responses[0][1]:
                    route_points[str(point.id)] = point
            else:
                point_rrf_scores: dict[str, float] = {}
                for _query_idx, points in responses:
                    for rank, point in enumerate(points):
                        pid = str(point.id)
                        if pid not in route_points:
                            route_points[pid] = point
                        point_rrf_scores[pid] = (
                            point_rrf_scores.get(pid, 0.0)
                            + 1.0 / (_RRF_K + rank + 1)
                        )
                sorted_ids = sorted(
                    route_points.keys(),
                    key=lambda pid: point_rrf_scores.get(pid, 0.0),
                    reverse=True,
                )
                route_points = {pid: route_points[pid] for pid in sorted_ids}

            route_counts[route.label] = len(route_points)
            for point in route_points.values():
                pid = str(point.id)
                if pid not in seen_point_ids:
                    seen_point_ids.add(pid)
                    merged_points.append(point)

        logger.info(
            "retrieval.batch_query",
            request_count=len(requests),
            route_count=len(routes),
            expanded_query_count=len(expanded_queries),
            route_counts=route_counts,
        )
        return route_counts, merged_points

    def _query_and_filter(
        self,
        query_text: str,
        *,
        active_generation: ServingGeneration,
        source_class_hint: str | None,
        record_type_hint: str | None = None,
        limit: int,
        precomputed_vectors: dict[str, EmbeddingVector] | None = None,
    ) -> tuple[RetrievedEvidence, ...]:
        """Run vector search, DB lookup, and safety filters. Returns unranked evidence."""
        t_qdrant_start = time.monotonic()
        payload_points = self._query_generation(
            query_text,
            active_generation=active_generation,
            source_class_hint=source_class_hint,
            record_type_hint=record_type_hint,
            limit=limit,
            precomputed_vectors=precomputed_vectors,
        )
        qdrant_search_ms = round((time.monotonic() - t_qdrant_start) * 1000, 1)
        return self._hydrate_and_filter(
            payload_points,
            active_generation=active_generation,
            source_class_hint=source_class_hint,
            qdrant_search_ms=qdrant_search_ms,
        )

    def _hydrate_and_filter(
        self,
        payload_points: tuple[Any, ...],
        *,
        active_generation: ServingGeneration,
        source_class_hint: str | None,
        allowed_source_classes: frozenset[str] | None = None,
        qdrant_search_ms: float | None = None,
    ) -> tuple[RetrievedEvidence, ...]:
        """DB lookup and safety filters for raw Qdrant points."""
        candidate_count_before_db_filter = len(payload_points)
        active_points = [
            point
            for point in payload_points
            if payload_matches_active_generation(
                (point.payload or {}).get("serving_generation_id"),
                active_generation_id=active_generation.generation_id,
            )
            and "record_version_id" in (point.payload or {})
        ]
        if not active_points:
            logger.info(
                "retrieval.filtering_complete",
                candidate_count_before_db_filter=candidate_count_before_db_filter,
                surviving_evidence_count=0,
            )
            return ()

        record_version_ids = {
            str(point.payload["record_version_id"]) for point in active_points
        }
        t_db_start = time.monotonic()
        rows = self._session.execute(
            select(CanonicalRecord, SourceRegistry.source_class, SourceRegistry.legal_status)
            .join(
                SourceRegistry,
                CanonicalRecord.source_id == SourceRegistry.source_id,
                isouter=True,
            )
            .where(CanonicalRecord.record_version_id.in_(tuple(record_version_ids)))
        ).all()
        t_db_end = time.monotonic()
        rows_by_record_version_id = {
            record.record_version_id: (record, source_class, legal_status)
            for record, source_class, legal_status in rows
        }

        evidence: list[RetrievedEvidence] = []
        for point in active_points:
            payload = point.payload or {}
            record_tuple = rows_by_record_version_id.get(str(payload["record_version_id"]))
            if record_tuple is None:
                continue
            record, source_class, legal_status = record_tuple
            if not record_is_current_only(record):
                continue
            if not source_is_query_allowed(legal_status):
                continue
            if allowed_source_classes is not None:
                if (source_class or "general") not in allowed_source_classes:
                    continue
            elif not source_class_matches(source_class, source_class_hint=source_class_hint):
                continue

            original = str(payload.get("original_text", "")).strip()
            contextualized = str(payload.get("text", "")).strip()
            content = (
                original
                or contextualized
                or extract_primary_content(
                    record.record_payload,
                    record_type=record.record_type,
                )
            )
            evidence.append(
                RetrievedEvidence(
                    chunk_id=str(payload.get("chunk_id") or record.record_version_id),
                    chunk_index=int(payload.get("chunk_index") or 0),
                    chunk_count=int(payload.get("chunk_count") or 1),
                    record_version_id=record.record_version_id,
                    record_id=record.record_id,
                    record_type=record.record_type,
                    source_class=source_class or "general",
                    source_url=record.source_url,
                    source_locator=str(payload.get("source_locator") or record.source_locator),
                    source_authority_tier=record.source_authority_tier,
                    conflict_scope_id=record.conflict_scope_id,
                    dedupe_key=record.dedupe_key,
                    value_hash=value_hash_for_stored_record(
                        record.record_type,
                        record.record_payload or {},
                        record.source_text_hash,
                    ),
                    freshness_status=record.freshness_status,
                    year_confidence=record.year_confidence,
                    cycle_label=record.cycle_label,
                    content=content,
                    score=0.0,
                    contextualized_text=(
                        contextualized
                        if original and contextualized and contextualized != original
                        else ""
                    ),
                )
            )

        logger.info(
            "retrieval.filtering_complete",
            candidate_count_before_db_filter=candidate_count_before_db_filter,
            surviving_evidence_count=len(evidence),
            record_types=sorted({item.record_type for item in evidence}),
            qdrant_search_ms=qdrant_search_ms,
            db_hydration_ms=round((t_db_end - t_db_start) * 1000, 1),
            db_row_count=len(rows),
        )
        return tuple(evidence)

    def _expand_evidence_context(
        self,
        evidence: tuple[RetrievedEvidence, ...],
        *,
        collection_name: str,
    ) -> tuple[RetrievedEvidence, ...]:
        """Fetch adjacent chunks for multi-chunk evidence and set context_window.

        Collects all sibling chunk IDs across evidence items and fetches them
        in a single batched ``qdrant_client.retrieve()`` call to avoid
        sequential round-trips.
        """
        t_expand_start = time.monotonic()

        # First pass: collect all sibling IDs we need to fetch.
        # Map UUID string → list of (evidence_index, chunk_index) for routing results back.
        all_sibling_ids: list[str] = []
        uuid_to_targets: dict[str, list[tuple[int, int]]] = {}

        for ev_idx, item in enumerate(evidence):
            if item.chunk_count <= 1:
                continue

            adjacent_indices: list[int] = []
            if item.chunk_index > 0:
                adjacent_indices.append(item.chunk_index - 1)
            if item.chunk_index < item.chunk_count - 1:
                adjacent_indices.append(item.chunk_index + 1)

            for adj_idx in adjacent_indices:
                adj_chunk_id = f"{item.record_version_id}:chunk:{adj_idx}"
                uuid_str = str(uuid5(NAMESPACE_URL, adj_chunk_id))
                if uuid_str not in uuid_to_targets:
                    uuid_to_targets[uuid_str] = []
                    all_sibling_ids.append(uuid_str)
                uuid_to_targets[uuid_str].append((ev_idx, adj_idx))

        sibling_fetch_count = len(all_sibling_ids)

        # Single batched retrieve for all sibling chunks.
        fetched_by_uuid: dict[str, dict[str, Any]] = {}
        if all_sibling_ids:
            try:
                points = self._qdrant_client.retrieve(
                    collection_name,
                    ids=all_sibling_ids,
                    with_payload=["original_text", "text"],
                    with_vectors=False,
                )
                for point in points:
                    fetched_by_uuid[str(point.id)] = point.payload or {}
            except Exception:
                logger.debug(
                    "retrieval.sibling_chunk_batch_fetch_failed",
                    sibling_count=sibling_fetch_count,
                    exc_info=True,
                )

        # Second pass: distribute fetched texts back to evidence items.
        sibling_texts_by_ev: dict[int, dict[int, str]] = {}
        for uuid_str, targets in uuid_to_targets.items():
            payload = fetched_by_uuid.get(uuid_str)
            if payload is None:
                continue
            text = (
                str(payload.get("original_text", "")).strip()
                or str(payload.get("text", "")).strip()
            )
            if not text:
                continue
            for ev_idx, adj_idx in targets:
                if ev_idx not in sibling_texts_by_ev:
                    sibling_texts_by_ev[ev_idx] = {}
                sibling_texts_by_ev[ev_idx][adj_idx] = text

        expanded: list[RetrievedEvidence] = []
        for ev_idx, item in enumerate(evidence):
            sibling_texts = sibling_texts_by_ev.get(ev_idx)
            if sibling_texts:
                expanded.append(expand_chunk_context(item, sibling_texts))
            else:
                expanded.append(item)

        logger.info(
            "retrieval.context_expansion",
            sibling_fetch_count=sibling_fetch_count,
            expansion_ms=round((time.monotonic() - t_expand_start) * 1000, 1),
            multi_chunk_items=sum(1 for e in evidence if e.chunk_count > 1),
        )
        return tuple(expanded)

    def _query_generation(
        self,
        query_text: str,
        *,
        active_generation: ServingGeneration,
        source_class_hint: str | None,
        record_type_hint: str | None = None,
        limit: int,
        precomputed_vectors: dict[str, EmbeddingVector] | None = None,
    ) -> tuple[Any, ...]:
        if self._embedding_provider is None:
            raise RuntimeError(
                "Retrieval requires an explicit embedding provider; no runtime fallback is available."
            )

        expanded_queries = expand_query_with_synonyms(query_text)
        query_limit = max(limit * self._candidate_multiplier, self._candidate_floor)
        query_filter = _build_query_filter(
            active_generation_id=active_generation.generation_id,
            source_class_hint=source_class_hint,
            record_type_hint=record_type_hint,
        )

        def _search_single(query: str) -> list[Any]:
            if precomputed_vectors is not None and query in precomputed_vectors:
                query_vectors = precomputed_vectors[query]
            else:
                query_vectors = embed_query_text(self._embedding_provider, query)  # type: ignore[arg-type]
            response = self._qdrant_client.query_points(
                collection_name=active_generation.qdrant_collection,
                prefetch=[
                    models.Prefetch(
                        using="dense",
                        query=list(query_vectors.dense_vector),
                        limit=query_limit,
                    ),
                    models.Prefetch(
                        using="sparse",
                        query=models.SparseVector(
                            indices=list(query_vectors.sparse_vector.indices),
                            values=list(query_vectors.sparse_vector.values),
                        ),
                        limit=query_limit,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                query_filter=query_filter,
                with_payload=True,
                with_vectors=False,
                limit=query_limit,
            )
            return list(response.points)

        all_points: dict[str, Any] = {}
        _RRF_K = 60  # Standard constant from Cormack et al. (2009)

        if len(expanded_queries) == 1:
            # Single query — no RRF scoring needed; ordering from Qdrant is sufficient.
            for point in _search_single(expanded_queries[0]):
                all_points[str(point.id)] = point
        else:
            point_rrf_scores: dict[str, float] = {}
            last_exc: Exception | None = None

            with ThreadPoolExecutor(max_workers=min(len(expanded_queries), 4)) as pool:
                futures = [pool.submit(_search_single, q) for q in expanded_queries]
                failed_count = 0
                for future in futures:
                    try:
                        for rank, point in enumerate(future.result()):
                            pid = str(point.id)
                            if pid not in all_points:
                                all_points[pid] = point
                            point_rrf_scores[pid] = (
                                point_rrf_scores.get(pid, 0.0)
                                + 1.0 / (_RRF_K + rank + 1)
                            )
                    except Exception as exc:
                        failed_count += 1
                        last_exc = exc
                        logger.warning(
                            "retrieval.subquery_failed",
                            exc_info=True,
                        )
                if failed_count == len(futures) and not all_points:
                    raise RuntimeError(
                        f"All {failed_count} expanded subqueries failed; "
                        "embedding provider may be unavailable."
                    ) from last_exc

            # Sort by fused RRF score so best candidates come first.
            sorted_ids = sorted(
                all_points.keys(),
                key=lambda pid: point_rrf_scores.get(pid, 0.0),
                reverse=True,
            )
            all_points = {pid: all_points[pid] for pid in sorted_ids}

        logger.info(
            "retrieval.query_generation",
            expanded_query_count=len(expanded_queries),
            candidate_count=len(all_points),
            source_class_hint=source_class_hint,
        )
        return tuple(all_points.values())



def _build_query_filter(
    *,
    active_generation_id: str,
    source_class_hint: str | None,
    record_type_hint: str | None = None,
) -> models.Filter:
    must_conditions: list[
        models.FieldCondition | models.IsEmptyCondition | models.IsNullCondition
        | models.HasIdCondition | models.HasVectorCondition | models.NestedCondition
        | models.Filter
    ] = [
        models.FieldCondition(
            key="serving_generation_id",
            match=models.MatchValue(value=active_generation_id),
        ),
        models.FieldCondition(
            key="freshness_status",
            match=models.MatchValue(value="current"),
        ),
    ]
    if source_class_hint is not None:
        expanded = expand_source_classes(source_class_hint)
        if len(expanded) == 1:
            must_conditions.append(
                models.FieldCondition(
                    key="source_class",
                    match=models.MatchValue(value=expanded[0]),
                )
            )
        else:
            must_conditions.append(
                models.FieldCondition(
                    key="source_class",
                    match=models.MatchAny(any=list(expanded)),
                )
            )
    if record_type_hint is not None:
        must_conditions.append(
            models.FieldCondition(
                key="record_type",
                match=models.MatchValue(value=record_type_hint),
            )
        )
    return models.Filter(must=must_conditions)


def _build_route_plan(
    *,
    source_class_hint: str | None,
    record_type_hint: str | None,
    hint_is_user_provided: bool,
) -> tuple[_RetrievalRoute, ...]:
    routes: list[_RetrievalRoute] = []

    if hint_is_user_provided:
        if source_class_hint is not None and record_type_hint is not None:
            routes.append(
                _RetrievalRoute(
                    label="strict_source_and_type",
                    source_class_hint=source_class_hint,
                    record_type_hint=record_type_hint,
                )
            )
        elif source_class_hint is not None:
            routes.append(
                _RetrievalRoute(
                    label="strict_source",
                    source_class_hint=source_class_hint,
                )
            )
        elif record_type_hint is not None:
            routes.append(
                _RetrievalRoute(
                    label="strict_record_type",
                    record_type_hint=record_type_hint,
                )
            )
        return tuple(routes)

    if source_class_hint is not None:
        routes.append(
            _RetrievalRoute(
                label="source_hint",
                source_class_hint=source_class_hint,
            )
        )

    routes.append(_RetrievalRoute(label="global"))

    if record_type_hint is not None:
        routes.append(
            _RetrievalRoute(
                label="record_type",
                record_type_hint=record_type_hint,
            )
        )

    return _dedupe_routes(routes)


def _dedupe_routes(
    routes: list[_RetrievalRoute] | tuple[_RetrievalRoute, ...],
) -> tuple[_RetrievalRoute, ...]:
    deduped: list[_RetrievalRoute] = []
    seen: set[tuple[str | None, str | None]] = set()
    for route in routes:
        key = (route.source_class_hint, route.record_type_hint)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(route)
    return tuple(deduped)
