from __future__ import annotations

from collections.abc import Callable

from sqlalchemy import select
from sqlalchemy.orm import Session

from unibot.db.models import ServingGeneration
from unibot.utils import utc_now as _utc_now


class ServingGenerationRepository:
    def __init__(
        self,
        session: Session | None = None,
        *,
        session_factory: Callable[[], Session] | None = None,
    ) -> None:
        if session is None and session_factory is None:
            raise ValueError("Provide either session or session_factory")
        self._session = session
        self._session_factory = session_factory

    def _open_session(self) -> tuple[Session, bool]:
        if self._session_factory is not None:
            return self._session_factory(), True
        assert self._session is not None
        return self._session, False

    def create_staged_generation(
        self,
        *,
        generation_label: str,
        qdrant_collection: str,
        generation_metadata: dict,
    ) -> ServingGeneration:
        session, should_close = self._open_session()
        try:
            generation = ServingGeneration(
                generation_label=generation_label,
                status="staged",
                qdrant_collection=qdrant_collection,
                generation_metadata=generation_metadata,
            )
            session.add(generation)
            if should_close:
                session.commit()
                session.expunge(generation)
            else:
                session.flush()
            return generation
        finally:
            if should_close:
                session.close()

    def get_active_generation(self) -> ServingGeneration | None:
        session, should_close = self._open_session()
        try:
            result = session.execute(
                select(ServingGeneration).where(ServingGeneration.status == "active")
            ).scalar_one_or_none()
            if result is not None and should_close:
                session.expunge(result)
            return result
        finally:
            if should_close:
                session.close()

    def activate_generation(self, generation: ServingGeneration) -> ServingGeneration:
        session, should_close = self._open_session()
        try:
            merged_gen = session.merge(generation)

            active_generation = session.execute(
                select(ServingGeneration).where(ServingGeneration.status == "active")
            ).scalar_one_or_none()
            if active_generation is not None and active_generation.generation_id != merged_gen.generation_id:
                active_generation.status = "retired"

            merged_gen.status = "active"
            merged_gen.activated_at = _utc_now()
            if should_close:
                session.commit()
                session.expunge(merged_gen)
            else:
                session.flush()
            return merged_gen
        finally:
            if should_close:
                session.close()
