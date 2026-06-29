from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from unibot.db.models import IndexJob
from unibot.db.repositories.serving_generations import ServingGenerationRepository
from unibot.indexing.embeddings import DenseSparseEmbeddingProvider
from unibot.indexing.qdrant_writer import QdrantWriter
from unibot.indexing.serving_generation_builder import ServingGenerationBuilder
from unibot.utils import utc_now as _utc_now


@dataclass(frozen=True, slots=True)
class IndexJobExecutionResult:
    processed_count: int
    succeeded_count: int
    failed_count: int
    processed_job_ids: tuple[str, ...]
    succeeded_job_ids: tuple[str, ...]
    failed_job_ids: tuple[str, ...]
    generation_label: str | None = None


class IndexJobExecutor:
    def __init__(
        self,
        *,
        session: Session | None = None,
        session_factory: Callable[[], Session] | None = None,
        generation_repository: ServingGenerationRepository | None = None,
        qdrant_writer: QdrantWriter,
        embedding_provider: DenseSparseEmbeddingProvider,
        alias_name: str = "unibot-active",
        collection_prefix: str = "unibot-generation",
    ) -> None:
        if session is None and session_factory is None:
            raise ValueError("Provide either session or session_factory")
        if session_factory is not None:
            self._session_factory = session_factory
            self._caller_owns_session = False
        else:
            assert session is not None
            self._session_factory = lambda: session
            self._caller_owns_session = True
        self._generation_repository = generation_repository or ServingGenerationRepository(
            session_factory=self._session_factory,
        )
        self._qdrant_writer = qdrant_writer
        self._embedding_provider = embedding_provider
        self._alias_name = alias_name
        self._collection_prefix = collection_prefix

    def _close_if_owned(self, session: Session) -> None:
        if not self._caller_owns_session:
            session.close()

    def run_pending_jobs(self, *, limit: int | None = None) -> IndexJobExecutionResult:
        # Phase 1: Claim pending jobs (short session)
        session = self._session_factory()
        try:
            query = (
                select(IndexJob)
                .where(IndexJob.status == "pending")
                .order_by(IndexJob.scheduled_at, IndexJob.job_id)
            )
            if limit is not None:
                query = query.limit(limit)

            pending_jobs = list(session.execute(query).scalars().all())
            if not pending_jobs:
                return IndexJobExecutionResult(
                    processed_count=0,
                    succeeded_count=0,
                    failed_count=0,
                    processed_job_ids=(),
                    succeeded_job_ids=(),
                    failed_job_ids=(),
                    generation_label=None,
                )

            started_at = _utc_now()
            for job in pending_jobs:
                job.status = "running"
                job.started_at = started_at
                job.finished_at = None
                job.error_message = None

            processed_job_ids = tuple(job.job_id for job in pending_jobs)
            session.commit()
        finally:
            self._close_if_owned(session)

        # Phase 2: Build generation (uses its own sessions internally)
        generation_label = _build_generation_label(started_at)
        try:
            # Propagate the ownership mode to the builder so the
            # _caller_owns_session flag is set correctly downstream.
            if self._caller_owns_session:
                builder = ServingGenerationBuilder(
                    session=self._session_factory(),
                    generation_repository=self._generation_repository,
                    qdrant_writer=self._qdrant_writer,
                    embedding_provider=self._embedding_provider,
                    alias_name=self._alias_name,
                    collection_prefix=self._collection_prefix,
                )
            else:
                builder = ServingGenerationBuilder(
                    session_factory=self._session_factory,
                    generation_repository=self._generation_repository,
                    qdrant_writer=self._qdrant_writer,
                    embedding_provider=self._embedding_provider,
                    alias_name=self._alias_name,
                    collection_prefix=self._collection_prefix,
                )
            build_result = builder.build_and_activate(generation_label=generation_label)
        except Exception as exc:
            # Phase 2 failed: mark jobs as failed (short session)
            fail_session = self._session_factory()
            try:
                finished_at = _utc_now()
                failed_jobs = list(
                    fail_session.execute(
                        select(IndexJob).where(IndexJob.job_id.in_(processed_job_ids))
                    ).scalars().all()
                )
                for job in failed_jobs:
                    job.status = "failed"
                    job.error_message = str(exc)
                    job.finished_at = finished_at
                fail_session.commit()
            finally:
                self._close_if_owned(fail_session)
            failed_job_ids = tuple(job.job_id for job in failed_jobs)
            return IndexJobExecutionResult(
                processed_count=len(processed_job_ids),
                succeeded_count=0,
                failed_count=len(failed_job_ids),
                processed_job_ids=processed_job_ids,
                succeeded_job_ids=(),
                failed_job_ids=failed_job_ids,
                generation_label=generation_label,
            )

        # Phase 3: Mark jobs succeeded (short session)
        success_session = self._session_factory()
        try:
            finished_at = _utc_now()
            succeeded_jobs = list(
                success_session.execute(
                    select(IndexJob).where(IndexJob.job_id.in_(processed_job_ids))
                ).scalars().all()
            )
            for job in succeeded_jobs:
                job.status = "succeeded"
                job.generation_id = build_result.generation.generation_id
                job.error_message = None
                job.finished_at = finished_at
            success_session.commit()
        finally:
            self._close_if_owned(success_session)

        succeeded_job_ids = tuple(job.job_id for job in succeeded_jobs)
        return IndexJobExecutionResult(
            processed_count=len(processed_job_ids),
            succeeded_count=len(succeeded_job_ids),
            failed_count=0,
            processed_job_ids=processed_job_ids,
            succeeded_job_ids=succeeded_job_ids,
            failed_job_ids=(),
            generation_label=generation_label,
        )


def _build_generation_label(started_at: datetime) -> str:
    return started_at.astimezone(timezone.utc).strftime("reindex-%Y%m%d-%H%M%S")
