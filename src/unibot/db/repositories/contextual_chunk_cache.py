from __future__ import annotations

from collections.abc import Callable

from sqlalchemy import select
from sqlalchemy.orm import Session

from unibot.db.models import ContextualChunkCache


class ContextualChunkCacheRepository:
    """Cache repository that manages its own session lifecycle.

    Accepts either a session_factory (preferred — opens/closes sessions per
    operation) or a legacy session instance for backward compatibility in tests.
    """

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
        """Return (session, should_close).

        If constructed with a factory, opens a new session (caller must close).
        If constructed with a bare session, returns it (caller must NOT close).
        """
        if self._session_factory is not None:
            return self._session_factory(), True
        assert self._session is not None
        return self._session, False

    def get_many(
        self, cache_keys: list[str] | tuple[str, ...]
    ) -> dict[str, ContextualChunkCache]:
        if not cache_keys:
            return {}
        session, should_close = self._open_session()
        try:
            rows = session.execute(
                select(ContextualChunkCache).where(
                    ContextualChunkCache.cache_key.in_(cache_keys)
                )
            ).scalars().all()
            # Detach from session so caller can use them freely
            if should_close:
                session.expunge_all()
            return {row.cache_key: row for row in rows}
        finally:
            if should_close:
                session.close()

    def save_many(
        self, entries: list[ContextualChunkCache] | tuple[ContextualChunkCache, ...]
    ) -> None:
        if not entries:
            return
        session, should_close = self._open_session()
        try:
            for entry in entries:
                session.merge(entry)
            if should_close:
                session.commit()
            else:
                session.flush()
        except Exception:
            if should_close:
                session.rollback()
            raise
        finally:
            if should_close:
                session.close()
