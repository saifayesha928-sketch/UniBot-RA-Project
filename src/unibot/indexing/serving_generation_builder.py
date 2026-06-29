from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import structlog
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from unibot.db.models import CanonicalRecord, ServingGeneration, SourceRegistry
from unibot.db.repositories.serving_generations import ServingGenerationRepository
from unibot.extract.text import stable_slug_with_hash
from unibot.verify.value_identity import value_hash_for_stored_record
from unibot.indexing.chunks import IndexChunk, build_chunks
from unibot.db.repositories.contextual_chunk_cache import ContextualChunkCacheRepository
from unibot.indexing.contextualization_service import ContextualizationService
from unibot.indexing.embeddings import DenseSparseEmbeddingProvider, EmbeddingResult, embed_chunks
from unibot.indexing.qdrant_writer import QdrantWriter
from unibot.pipeline.contracts import CycleProgressCallback
from unibot.retrieval.filters import QUERYABLE_SERVING_STATUSES as _SERVING_ELIGIBLE_STATUSES
from unibot.verify.deduplication import resolve_duplicates
from unibot.verify.rules import DedupeResult, VerificationCandidate, VerificationDecision

logger = structlog.get_logger()


@dataclass(frozen=True, slots=True)
class ServingGenerationBuildResult:
    generation: ServingGeneration
    record_version_ids: tuple[str, ...]
    failed_record_version_ids: tuple[str, ...]


class ServingGenerationBuilder:
    def __init__(
        self,
        *,
        session: Session | None = None,
        session_factory: Callable[[], Session] | None = None,
        generation_repository: ServingGenerationRepository,
        qdrant_writer: QdrantWriter,
        embedding_provider: DenseSparseEmbeddingProvider,
        alias_name: str = "unibot-active",
        collection_prefix: str = "unibot-generation",
        progress: CycleProgressCallback | None = None,
    ) -> None:
        if session is None and session_factory is None:
            raise ValueError("Provide either session or session_factory")
        # When a bare session is provided, wrap it in a factory that returns
        # the same session.  The _caller_owns_session flag ensures we never
        # close or commit a session we don't own — the caller controls its
        # lifecycle (important for tests and the orchestrator path).
        if session_factory is not None:
            self._session_factory = session_factory
            self._caller_owns_session = False
        else:
            assert session is not None
            self._session_factory = lambda: session
            self._caller_owns_session = True
        self._generation_repository = generation_repository
        self._qdrant_writer = qdrant_writer
        self._embedding_provider = embedding_provider
        self._alias_name = alias_name
        self._collection_prefix = collection_prefix
        self._progress = progress

    def _close_if_owned(self, session: Session) -> None:
        """Close the session only if we own it (factory path).
        Never close a caller-provided session — the caller controls its lifecycle.
        """
        if not self._caller_owns_session:
            session.close()

    def build_and_activate(self, *, generation_label: str) -> ServingGenerationBuildResult:
        # --- Phase 1: Plan (short session) ---
        if self._progress:
            self._progress.on_phase_start("plan")
        session = self._session_factory()
        try:
            active_generation = self._generation_repository.get_active_generation()
            previous_collection = (
                active_generation.qdrant_collection if active_generation is not None else None
            )
            previous_record_ids = set(
                active_generation.generation_metadata.get("record_version_ids", [])
                if active_generation is not None
                else []
            )
            if self._progress:
                self._progress.on_generation_step("loading", "Loading authoritative records")
            decisions = self._load_current_authoritative_records(session)
            dedupe_result = resolve_duplicates(decisions)
            collection_name = self._build_collection_name(generation_label)
            generation = self._generation_repository.create_staged_generation(
                generation_label=generation_label,
                qdrant_collection=collection_name,
                generation_metadata={},
            )
            session.commit()
        finally:
            self._close_if_owned(session)
        if self._progress:
            self._progress.on_phase_done("plan")

        # --- Phase 2: Build chunks (NO session held) ---
        if self._progress:
            self._progress.on_phase_start("build")
            self._progress.on_generation_step("chunking", f"Building chunks from {len(dedupe_result.primary_records)} records")
        chunks = build_chunks(
            list(dedupe_result.primary_records),
            serving_generation_id=generation.generation_id,
        )
        if self._progress:
            self._progress.on_phase_done("build")

        # --- Phase 3: Contextualize chunks ---
        if self._progress:
            self._progress.on_phase_start("contextualize")
            self._progress.on_generation_step("contextualizing", f"Contextualizing {len(chunks)} chunks")
        chunks = self._contextualize_chunks(
            chunks,
            decisions=tuple(dedupe_result.primary_records),
        )
        if self._progress:
            self._progress.on_phase_done("contextualize")

        # --- Phase 4: Embed chunks ---
        if self._progress:
            self._progress.on_phase_start("embed", total=len(chunks))
            self._progress.on_generation_step("embedding", f"Embedding {len(chunks)} chunks")
        eligible_chunk_count = len(chunks)

        def _embed_progress(completed: int, total: int) -> None:
            if self._progress:
                self._progress.on_generation_progress(completed, total)

        embeddings = embed_chunks(chunks, self._embedding_provider, progress=_embed_progress)

        if eligible_chunk_count > 0 and not embeddings.embedded_chunks:
            raise RuntimeError(
                "Serving generation build produced zero successful chunks; alias activation aborted."
            )
        if self._progress:
            self._progress.on_phase_done("embed")

        # --- Phase 5: Qdrant writes (NO session) ---
        if self._progress:
            self._progress.on_phase_start("index")
            self._progress.on_generation_step("indexing", f"Upserting {len(embeddings.embedded_chunks)} vectors to Qdrant")
        if embeddings.embedded_chunks:
            dense_vector_size = len(embeddings.embedded_chunks[0].vectors.dense_vector)
            self._qdrant_writer.ensure_collection(
                collection_name,
                dense_vector_size=dense_vector_size,
                fail_if_exists=True,
            )
            self._qdrant_writer.upsert_records(
                collection_name,
                [
                    self._qdrant_writer.point_from_embedded_chunk(chunk)
                    for chunk in embeddings.embedded_chunks
                ],
            )
        else:
            self._qdrant_writer.ensure_collection(collection_name, dense_vector_size=1)

        new_record_ids = {chunk.record_version_id for chunk in embeddings.embedded_chunks}
        to_deindex = tuple(sorted(previous_record_ids - new_record_ids))
        to_activate = tuple(sorted(new_record_ids))
        if self._progress:
            self._progress.on_phase_done("index")

        # --- Phase 6: Alias switch + DB activation (short session) ---
        if self._progress:
            self._progress.on_phase_start("activate")
            self._progress.on_generation_step("activating", f"Switching alias and activating {len(to_activate)} records")
        try:
            self._qdrant_writer.switch_alias(self._alias_name, collection_name)
        except Exception:
            logger.exception(
                "serving_generation.alias_switch_failed",
                collection_name=collection_name,
            )
            raise

        activation_session = self._session_factory()
        try:
            # Re-fetch the generation into this session
            merged_generation = activation_session.merge(generation)
            merged_generation.generation_metadata = self._build_generation_metadata(
                session=activation_session,
                embeddings=embeddings,
                dedupe_result=dedupe_result,
            )
            # For transactional atomicity, retire-old + activate-new +
            # mark-records must all happen in ONE session/transaction.
            #
            # In the legacy path (caller_owns_session), self._generation_repository
            # is already bound to this session, so we use it directly.  This
            # preserves test wrapper interception (_FailingActivationRepository,
            # _RecordingGenerationRepository, etc.).
            #
            # In the factory path, self._generation_repository was built with
            # session_factory= and would open its OWN session for activate_generation,
            # breaking atomicity.  So we create a temporary repo bound to the
            # activation session.
            if self._caller_owns_session:
                self._generation_repository.activate_generation(merged_generation)
            else:
                activation_repo = ServingGenerationRepository(session=activation_session)
                activation_repo.activate_generation(merged_generation)
            self._mark_records_indexed_active(activation_session, to_activate)
            self._mark_records_deindexed(activation_session, to_deindex)
            activation_session.commit()
            # Update the generation object for the return value
            generation = merged_generation
        except Exception:
            activation_session.rollback()
            logger.exception(
                "serving_generation.db_activation_failed",
                collection_name=collection_name,
            )
            if previous_collection is not None:
                try:
                    self._qdrant_writer.switch_alias(self._alias_name, previous_collection)
                except Exception:
                    logger.exception(
                        "serving_generation.alias_rollback_failed",
                        previous_collection=previous_collection,
                    )
            raise
        finally:
            self._close_if_owned(activation_session)

        if self._progress:
            self._progress.on_phase_done("activate")

        # --- Clean up retired collection (best-effort) ---
        if previous_collection is not None and previous_collection != collection_name:
            try:
                self._qdrant_writer.delete_collection(previous_collection)
            except Exception:
                logger.exception(
                    "serving_generation.retired_collection_cleanup_failed",
                    collection_name=previous_collection,
                )

        return ServingGenerationBuildResult(
            generation=generation,
            record_version_ids=tuple(
                str(v) for v in cast(list[str], generation.generation_metadata["record_version_ids"])
            ),
            failed_record_version_ids=embeddings.failed_record_version_ids,
        )

    def recover_on_startup(self) -> None:
        """Reconcile DB and Qdrant alias state after a crash."""
        session = self._session_factory()
        try:
            active_generation = self._generation_repository.get_active_generation()
            alias_collection = self._qdrant_writer.resolve_alias(self._alias_name)

            if active_generation is not None:
                if alias_collection != active_generation.qdrant_collection:
                    logger.warning(
                        "serving_generation.recovery_repoint_alias",
                        expected=active_generation.qdrant_collection,
                        actual=alias_collection,
                    )
                    self._qdrant_writer.switch_alias(
                        self._alias_name, active_generation.qdrant_collection
                    )
                return

            staged_rows = session.execute(
                select(ServingGeneration).where(ServingGeneration.status == "staged")
            ).scalars().all()
            for staged in staged_rows:
                logger.warning(
                    "serving_generation.recovery_mark_failed",
                    generation_id=staged.generation_id,
                )
                staged.status = "failed"
            session.commit()
        finally:
            self._close_if_owned(session)

    def _contextualize_chunks(
        self,
        chunks: tuple[IndexChunk, ...],
        *,
        decisions: tuple[VerificationDecision, ...],
    ) -> tuple[IndexChunk, ...]:
        from unibot.settings import get_settings

        try:
            settings = get_settings()
        except Exception:
            logger.debug("contextual_retrieval.settings_unavailable")
            return chunks
        if not getattr(settings, "contextual_retrieval_enabled", False):
            return chunks

        api_key = getattr(settings, "openrouter_api_key", None)
        if not api_key:
            logger.warning("contextual_retrieval.no_openrouter_api_key_configured")
            return chunks

        # On the legacy (session=) path, pass the session directly so the
        # cache repo does NOT close it — the caller owns it.
        # On the factory path, pass the factory so the cache repo opens
        # and closes its own short-lived sessions.
        if self._caller_owns_session:
            cache_repository = ContextualChunkCacheRepository(
                session=self._session_factory(),
            )
        else:
            cache_repository = ContextualChunkCacheRepository(
                session_factory=self._session_factory,
            )
        service = ContextualizationService(
            cache_repository=cache_repository,
            model_name=getattr(
                settings,
                "contextual_retrieval_model",
                "anthropic/claude-haiku-4-5-20251001",
            ),
            max_concurrency=getattr(settings, "contextual_retrieval_max_concurrency", 50),
            cache_ttl=getattr(settings, "contextual_retrieval_cache_ttl", "5m"),
            timeout=float(getattr(settings, "openrouter_timeout_seconds", 60.0)),
            max_retries=getattr(settings, "contextual_retrieval_max_retries", 3),
            run_sync_timeout=600.0,
            base_url=getattr(
                settings,
                "openrouter_base_url",
                "https://openrouter.ai/api/v1/chat/completions",
            ),
            api_key=api_key,
            app_name=getattr(settings, "openrouter_app_name", "UniBot"),
        )

        return service.contextualize(chunks=chunks, decisions=decisions)

    def _build_generation_metadata(
        self, *, session: Session, embeddings: EmbeddingResult, dedupe_result: DedupeResult
    ) -> dict[str, object]:
        persisted_conflicts: dict[str, list[str]] = defaultdict(list)
        contradictory_rows = session.execute(
            select(
                CanonicalRecord.dedupe_key,
                CanonicalRecord.record_version_id,
            ).where(CanonicalRecord.freshness_status == "contradictory")
        ).all()
        for row in contradictory_rows:
            persisted_conflicts[row.dedupe_key].append(row.record_version_id)

        return {
            "record_version_ids": sorted(
                {chunk.record_version_id for chunk in embeddings.embedded_chunks}
            ),
            "failed_record_version_ids": list(embeddings.failed_record_version_ids),
            "duplicate_conflicts": [
                {
                    "dedupe_key": dedupe_key,
                    "record_ids": sorted(record_ids),
                }
                for dedupe_key, record_ids in sorted(persisted_conflicts.items())
            ],
        }

    def _load_current_authoritative_records(
        self, session: Session,
    ) -> tuple[VerificationDecision, ...]:
        rows = session.execute(
            select(
                CanonicalRecord,
                SourceRegistry.legal_status,
                SourceRegistry.source_class,
            )
            .join(
                SourceRegistry,
                CanonicalRecord.source_id == SourceRegistry.source_id,
                isouter=True,
            )
                        .where(
                or_(
                    SourceRegistry.legal_status.is_(None),
                    SourceRegistry.legal_status == "allowed",
                )
            )
        ).all()

        decisions: list[VerificationDecision] = []
        for record, _legal_status, source_class in rows:
            record_payload = dict(record.record_payload or {})
            if source_class is not None:
                record_payload.setdefault("source_class", source_class)
            candidate = VerificationCandidate(
                record_id=record.record_id,
                record_version_id=record.record_version_id,
                record_type=record.record_type,
                conflict_scope_id=record.conflict_scope_id,
                dedupe_key=record.dedupe_key,
                value_hash=value_hash_for_stored_record(
                    record.record_type,
                    record.record_payload or {},
                    record.source_text_hash,
                ),
                source_authority_tier=record.source_authority_tier,
                source_url=record.source_url,
                source_locator=record.source_locator,
                cycle_label=record.cycle_label,
                effective_from=record.effective_from,
                effective_to=record.effective_to,
                year_confidence=record.year_confidence,
                record_payload=record_payload,
            )
            decisions.append(
                VerificationDecision(
                    candidate=candidate,
                    verification_status="verified",
                    freshness_status="current",
                    serving_status="eligible",
                    is_current_candidate=record.is_current_candidate,
                    is_current_authoritative=record.is_current_authoritative,
                )
            )

        return tuple(decisions)

    def _mark_records_indexed_active(
        self, session: Session, record_version_ids: tuple[str, ...]
    ) -> None:
        if not record_version_ids:
            return
        rows = session.execute(
            select(CanonicalRecord).where(
                CanonicalRecord.record_version_id.in_(record_version_ids)
            )
        ).scalars()
        for row in rows:
            row.serving_status = "indexed_active"
        session.flush()

    def _mark_records_deindexed(
        self, session: Session, record_version_ids: tuple[str, ...]
    ) -> None:
        if not record_version_ids:
            return
        rows = session.execute(
            select(CanonicalRecord).where(
                CanonicalRecord.record_version_id.in_(record_version_ids)
            )
        ).scalars()
        for row in rows:
            row.serving_status = "deindexed"
        session.flush()

    def _build_collection_name(self, generation_label: str) -> str:
        suffix_budget = 255 - len(self._collection_prefix) - 1
        slug = stable_slug_with_hash(generation_label, max_length=suffix_budget)
        return f"{self._collection_prefix}-{slug}"
