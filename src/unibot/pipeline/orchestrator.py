from __future__ import annotations

import tempfile
from collections import defaultdict
from dataclasses import replace
from importlib import import_module
from pathlib import Path

from sqlalchemy.orm import Session

from unibot.crawl.fetchers import FetchedArtifact
from unibot.crawl.jobs import CrawlJob, select_sources_for_crawl
from unibot.crawl.snapshot_store import RawSnapshotStore
from unibot.db.models import CanonicalRecord, SourceSection as SourceSectionModel, VerificationEvent
from unibot.db.repositories.serving_generations import ServingGenerationRepository
from unibot.db.repositories.source_registry import SourceRegistryEntry, SourceRegistryRepository
from unibot.indexing.embeddings import DenseSparseEmbeddingProvider
from unibot.indexing.qdrant_writer import QdrantWriter
from unibot.pipeline.contracts import (
    ArtifactFetcher,
    CandidateExtractor,
    CycleProgressCallback,
    SourceDiscoverer,
    UpdateCycleAudit,
    UpdateCycleResult,
    _CrawlPhaseResult,
    _ScopeProcessingResult,
)
from unibot.storage.object_store import LocalObjectStore, ObjectStore
from unibot.verify.rules import VerificationCandidate, VerificationDecision


def _compat_facade():
    """Resolve the scheduler.jobs facade at call time to avoid circular imports.

    The orchestrator must read ``logger`` and ``ServingGenerationBuilder`` from
    the facade module so that tests which monkeypatch those names on
    ``unibot.scheduler.jobs`` still intercept the orchestrator's calls.
    """
    return import_module("unibot.scheduler.jobs")


class UpdateCycleOrchestrator:
    def __init__(
        self,
        *,
        session: Session,
        qdrant_writer: QdrantWriter,
        embedding_provider: DenseSparseEmbeddingProvider,
        source_entries: tuple[SourceRegistryEntry, ...],
        fetch_artifact: ArtifactFetcher,
        extract_candidates: CandidateExtractor,
        discover_sources: SourceDiscoverer | None = None,
        object_store: ObjectStore | None = None,
        auto_activate: bool = True,
        alias_name: str = "unibot-active",
        collection_prefix: str = "unibot-generation",
        progress: CycleProgressCallback | None = None,
    ) -> None:
        self._session = session
        self._qdrant_writer = qdrant_writer
        self._embedding_provider = embedding_provider
        self._source_entries = source_entries
        self._fetch_artifact = fetch_artifact
        self._extract_candidates = extract_candidates
        self._discover_sources = discover_sources or (lambda _job, _artifact: ())
        self._object_store = object_store or LocalObjectStore(
            base_path=Path(tempfile.mkdtemp(prefix="unibot-raw-")),
        )
        self._auto_activate = auto_activate
        self._alias_name = alias_name
        self._collection_prefix = collection_prefix
        self._source_registry_repository = SourceRegistryRepository(session)
        self._generation_repository = ServingGenerationRepository(session)
        self._progress = progress
        self._crawled_parent_urls: set[str] = set()
        self._discovered_child_urls: dict[str, set[str]] = defaultdict(set)
        self._source_id_cache: dict[str, str | None] = {}

    def _reload_crawl_timestamps(
        self,
        entries: tuple[SourceRegistryEntry, ...],
    ) -> tuple[SourceRegistryEntry, ...]:
        from unibot.pipeline import source_lifecycle as _sl
        return _sl.reload_crawl_timestamps(self._session, entries)

    def run(self, *, generation_label: str, limit: int | None = None) -> UpdateCycleResult:
        self._source_registry_repository.upsert_entries(self._source_entries)
        self._session.flush()

        crawl = self._run_crawl_phase(limit=limit)
        self._run_reconciliation_phase(crawl.pending_deindex_record_version_ids)
        active_record_version_ids = self._run_generation_phase(
            generation_label=generation_label,
        )
        audit = self._run_audit_phase(
            fetched_source_urls=tuple(crawl.fetched_source_urls),
        )

        return UpdateCycleResult(
            selected_source_urls=tuple(crawl.processed_source_urls),
            fetched_source_urls=tuple(crawl.fetched_source_urls),
            pending_deindex_record_version_ids=tuple(
                sorted(set(crawl.pending_deindex_record_version_ids))
            ),
            verification_event_ids=tuple(sorted(crawl.verification_event_ids)),
            active_record_version_ids=active_record_version_ids,
            audit=audit,
            generation_label=generation_label,
        )

    def _run_crawl_phase(self, *, limit: int | None) -> _CrawlPhaseResult:
        from unibot.pipeline.source_lifecycle import normalized_crawl_status as _normalized_crawl_status

        logger = _compat_facade().logger
        selected_jobs = list(select_sources_for_crawl(self._source_entries, limit=limit))
        logger.info("update_cycle.sources_selected", count=len(selected_jobs))
        if self._progress:
            self._progress.on_phase_start("crawling", total=len(selected_jobs))
        processed_source_urls: list[str] = []
        queued_source_urls = {job.source_url for job in selected_jobs}
        fetched_source_urls: list[str] = []
        pending_deindex_record_version_ids: list[str] = []
        verification_event_ids: list[str] = []

        while selected_jobs:
            job = selected_jobs.pop(0)
            queued_source_urls.discard(job.source_url)
            processed_source_urls.append(job.source_url)
            if self._progress:
                self._progress.on_source_start(job.source_url, "fetching")
            try:
                artifact = self._fetch_artifact(job)
            except Exception:
                logger.exception(
                    "update_cycle.fetch_failed",
                    source_url=job.source_url,
                    source_class=job.source_class,
                )
                self._record_source_error(job, "fetch_failed")
                if self._progress:
                    self._progress.on_source_failed(job.source_url, "fetch_failed")
                continue
            effective_job = job
            if isinstance(artifact, FetchedArtifact) and artifact.source_url != job.source_url:
                self._source_registry_repository.rebind_source_url(
                    old_url=job.source_url,
                    new_url=artifact.source_url,
                )
                effective_job = replace(job, source_url=artifact.source_url)
            logger.info("update_cycle.source_fetched", source_url=effective_job.source_url)
            if self._progress:
                self._progress.on_source_start(effective_job.source_url, "extracting")
            fetched_source_urls.append(effective_job.source_url)
            self._crawled_parent_urls.add(effective_job.source_url)
            discovered_entries = tuple(self._discover_sources(effective_job, artifact))
            if discovered_entries:
                self._source_registry_repository.upsert_entries(discovered_entries)
                self._session.flush()
                for entry in discovered_entries:
                    if entry.parent_source_url:
                        self._discovered_child_urls[entry.parent_source_url].add(
                            entry.source_url
                        )
                if self._progress:
                    self._progress.on_sources_discovered(len(discovered_entries))
                discovered_entries = self._reload_crawl_timestamps(discovered_entries)
                for discovered_job in select_sources_for_crawl(discovered_entries):
                    if (
                        discovered_job.source_url in processed_source_urls
                        or discovered_job.source_url in queued_source_urls
                    ):
                        continue
                    selected_jobs.append(discovered_job)
                    queued_source_urls.add(discovered_job.source_url)
            try:
                candidates = self._extract_candidates_with_provenance(effective_job, artifact)
            except Exception:
                logger.exception(
                    "update_cycle.extraction_failed",
                    source_url=effective_job.source_url,
                    source_class=effective_job.source_class,
                )
                self._record_source_error(effective_job, "extraction_failed")
                if self._progress:
                    self._progress.on_source_failed(effective_job.source_url, "extraction_failed")
                continue
            self._record_source_success(
                effective_job,
                crawl_status=_normalized_crawl_status(effective_job, candidates),
            )
            if self._progress:
                self._progress.on_source_done(effective_job.source_url, len(candidates))
            if not candidates:
                continue

            try:
                processing = self._process_scope_updates(candidates)
            except Exception:
                logger.exception(
                    "update_cycle.scope_processing_failed",
                    source_url=effective_job.source_url,
                    source_class=effective_job.source_class,
                )
                self._record_source_error(effective_job, "scope_processing_failed")
                if self._progress:
                    self._progress.on_source_failed(
                        effective_job.source_url, "scope_processing_failed"
                    )
                continue
            pending_deindex_record_version_ids.extend(
                processing.pending_deindex_record_version_ids
            )
            verification_event_ids.extend(processing.verification_event_ids)

        self._session.commit()
        logger.info(
            "update_cycle.crawl_checkpoint_committed",
            fetched_count=len(fetched_source_urls),
        )

        if self._progress:
            self._progress.on_phase_done("crawling")

        return _CrawlPhaseResult(
            processed_source_urls=processed_source_urls,
            fetched_source_urls=fetched_source_urls,
            pending_deindex_record_version_ids=pending_deindex_record_version_ids,
            verification_event_ids=verification_event_ids,
        )

    def _run_reconciliation_phase(
        self, pending_deindex_record_version_ids: list[str],
    ) -> None:
        if self._progress:
            self._progress.on_phase_start("reconciliation")
        self._reconcile_disappeared_sources()
        pending_deindex_record_version_ids.extend(self._downgrade_expired_sources())
        self._persist_supporting_links()
        self._run_cross_source_enrichment()
        # Checkpoint: commit reconciliation work before the generation phase
        # starts.  The generation phase may run for minutes (LLM + embedding).
        # Without this commit, the reconciliation transaction would be
        # idle-in-transaction and killed by Neon's timeout.
        self._session.commit()
        if self._progress:
            self._progress.on_phase_done("reconciliation")

    def _run_generation_phase(
        self, *, generation_label: str,
    ) -> tuple[str, ...]:
        facade = _compat_facade()
        logger = facade.logger
        ServingGenerationBuilder = facade.ServingGenerationBuilder

        active_record_version_ids: tuple[str, ...] = ()
        if self._progress:
            self._progress.on_phase_start("generation_build")
        if self._auto_activate:
            # Derive a session factory from the orchestrator's own engine
            # so the builder can use short-lived sessions for each phase.
            # We derive from self._session.get_bind() rather than using
            # get_direct_session_factory() to preserve test injection —
            # tests pass in-memory SQLite sessions via db_session.
            from sqlalchemy.orm import sessionmaker as _sessionmaker
            _builder_factory = _sessionmaker(
                bind=self._session.get_bind(),
                autoflush=False,
                expire_on_commit=False,
            )
            builder = ServingGenerationBuilder(
                session_factory=_builder_factory,
                generation_repository=ServingGenerationRepository(
                    session_factory=_builder_factory,
                ),
                qdrant_writer=self._qdrant_writer,
                embedding_provider=self._embedding_provider,
                alias_name=self._alias_name,
                collection_prefix=self._collection_prefix,
                progress=self._progress,
            )
            build_result = builder.build_and_activate(generation_label=generation_label)
            active_record_version_ids = tuple(sorted(build_result.record_version_ids))
            logger.info(
                "update_cycle.generation_built",
                generation_label=generation_label,
                active_count=len(build_result.record_version_ids),
                failed_count=len(build_result.failed_record_version_ids),
            )
        else:
            logger.info(
                "update_cycle.activation_skipped",
                generation_label=generation_label,
                reason="manual_review_mode",
            )
        if self._progress:
            self._progress.on_phase_done("generation_build")
        return active_record_version_ids

    def _run_audit_phase(
        self, *, fetched_source_urls: tuple[str, ...],
    ) -> UpdateCycleAudit:
        from unibot.pipeline.audits import assert_audit_invariants as _assert_audit_invariants

        logger = _compat_facade().logger
        if self._progress:
            self._progress.on_phase_start("audit")
        audit = self._run_audits(fetched_source_urls=fetched_source_urls)
        _assert_audit_invariants(audit, require_active_alias_checks=self._auto_activate)
        logger.info(
            "update_cycle.audit_complete",
            active_alias_checked=self._auto_activate,
            active_index_count=audit.active_index_count,
            canonical_eligible_count=audit.canonical_eligible_count,
            contradictory_scopes=len(audit.contradictory_scope_ids),
            blocked_records=len(audit.blocked_or_inactive_record_version_ids),
        )
        if self._progress:
            self._progress.on_phase_done("audit")
        return audit

    def _extract_candidates_with_provenance(
        self,
        job: CrawlJob,
        artifact: object,
    ) -> tuple[VerificationCandidate, ...]:
        candidates = tuple(self._extract_candidates(job, artifact))
        if not isinstance(artifact, FetchedArtifact):
            return candidates

        source_id = self._source_id_for_url(job.source_url)
        if source_id is None:
            return candidates

        snapshot = RawSnapshotStore(
            session=self._session,
            object_store=self._object_store,
        ).store_snapshot(source_id=source_id, artifact=artifact)
        persisted_sections = self._persist_source_sections(
            source_id=source_id,
            snapshot_id=snapshot.snapshot_id,
            artifact=artifact,
        )
        return tuple(
            self._attach_provenance(
                candidate,
                page_content_hash=snapshot.page_content_hash,
                persisted_sections=persisted_sections,
            )
            for candidate in candidates
        )

    def _persist_source_sections(
        self,
        *,
        source_id: str,
        snapshot_id: str,
        artifact: FetchedArtifact,
    ) -> dict[str, SourceSectionModel]:
        from unibot.pipeline import provenance as _prov

        return _prov.persist_source_sections(
            self._session,
            source_id=source_id,
            snapshot_id=snapshot_id,
            artifact=artifact,
        )

    def _attach_provenance(
        self,
        candidate: VerificationCandidate,
        *,
        page_content_hash: str,
        persisted_sections: dict[str, SourceSectionModel],
    ) -> VerificationCandidate:
        from unibot.pipeline import provenance as _prov

        return _prov.attach_provenance(
            candidate,
            page_content_hash=page_content_hash,
            persisted_sections=persisted_sections,
        )

    def _process_scope_updates(
        self,
        incoming_candidates: tuple[VerificationCandidate, ...],
    ) -> _ScopeProcessingResult:
        from unibot.pipeline import canonical_upsert as _cu
        return _cu.process_scope_updates(
            self._session,
            incoming_candidates,
            source_id_resolver=self._source_id_for_url,
            create_verification_event=lambda session, decision: _cu.create_verification_event(session, decision),
        )

    def _create_record(
        self,
        candidate: VerificationCandidate,
        decision: VerificationDecision,
    ) -> CanonicalRecord | None:
        from unibot.pipeline import canonical_upsert as _cu
        return _cu.create_record(
            self._session, candidate, decision, self._source_id_for_url,
        )

    def _update_record(
        self,
        row: CanonicalRecord,
        candidate: VerificationCandidate,
        decision: VerificationDecision,
    ) -> None:
        from unibot.pipeline import canonical_upsert as _cu
        _cu.update_record(
            self._session, row, candidate, decision, self._source_id_for_url,
        )

    def _apply_decision(
        self,
        row: CanonicalRecord,
        decision: VerificationDecision,
    ) -> None:
        from unibot.pipeline import canonical_upsert as _cu
        _cu.apply_decision(row, decision)

    def _link_superseded_records(
        self,
        existing_rows: list[CanonicalRecord],
        incoming_candidates: list[VerificationCandidate],
        decisions: dict[str, VerificationDecision],
    ) -> tuple[str, ...]:
        from unibot.pipeline import canonical_upsert as _cu
        return _cu.link_superseded_records(
            self._session, existing_rows, incoming_candidates, decisions,
        )

    def _create_verification_event(
        self,
        decision: VerificationDecision,
    ) -> VerificationEvent:
        from unibot.pipeline import canonical_upsert as _cu
        return _cu.create_verification_event(self._session, decision)

    def _reconcile_disappeared_sources(self) -> None:
        from unibot.pipeline import source_lifecycle as _sl
        _sl.reconcile_disappeared_sources(
            self._session, self._crawled_parent_urls, self._discovered_child_urls,
        )

    def _persist_supporting_links(self) -> None:
        from unibot.pipeline import supporting_evidence as _se
        _se.persist_supporting_links(self._session, self._create_verification_event)

    def _run_cross_source_enrichment(self) -> None:
        from unibot.pipeline import enrichment as _enr
        _enr.run_cross_source_enrichment(self._session)

    def _load_current_authoritative_decisions(self) -> tuple[VerificationDecision, ...]:
        from unibot.pipeline import supporting_evidence as _se
        return _se.load_current_authoritative_decisions(self._session)

    def _run_audits(
        self,
        *,
        fetched_source_urls: tuple[str, ...] = (),
    ) -> UpdateCycleAudit:
        from unibot.pipeline import audits as _aud
        return _aud.run_audits(
            self._session,
            fetched_source_urls=fetched_source_urls,
            auto_activate=self._auto_activate,
            qdrant_writer=self._qdrant_writer,
            alias_name=self._alias_name,
        )

    def _record_source_error(self, job: CrawlJob, error_type: str) -> None:
        from unibot.pipeline import source_lifecycle as _sl
        _sl.record_source_error(
            self._session, job, error_type, self._source_id_for_url(job.source_url),
        )

    def _record_source_success(self, job: CrawlJob, *, crawl_status: str | None = None) -> None:
        from unibot.pipeline import source_lifecycle as _sl
        _sl.record_source_success(self._session, job, crawl_status=crawl_status)

    def _downgrade_expired_sources(self) -> tuple[str, ...]:
        from unibot.pipeline import source_lifecycle as _sl
        return _sl.downgrade_expired_sources(self._session)

    def _source_id_for_url(self, source_url: str) -> str | None:
        from unibot.pipeline import source_lifecycle as _sl
        return _sl.source_id_for_url(self._session, source_url, self._source_id_cache)
