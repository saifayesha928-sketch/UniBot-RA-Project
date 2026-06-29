from __future__ import annotations

import hashlib
import re
import threading
from typing import Any

from cachetools import TTLCache


_PUNCTUATION = re.compile(r"[^\w\s]", re.UNICODE)


def _normalize_near_duplicate(text: str) -> str:
    """Aggressive normalization: lowercase, strip punctuation, sort tokens."""
    lowered = text.strip().lower()
    stripped = _PUNCTUATION.sub("", lowered)
    tokens = sorted(stripped.split())
    return " ".join(tokens)


class QueryResultCache:
    """Thread-safe exact-match query result cache with TTL invalidation."""

    def __init__(self, *, maxsize: int = 2048, ttl: float = 3600) -> None:
        self._cache: TTLCache[str, Any] = TTLCache(maxsize=maxsize, ttl=ttl)
        self._lock = threading.Lock()

    def get(
        self,
        *,
        query_text: str,
        generation_id: str,
        source_class_hint: str | None,
        record_type_hint: str | None,
        hint_is_user_provided: bool = False,
        limit: int = 5,
        retrieval_query: str | None = None,
    ) -> Any | None:
        key = self._build_key(
            query_text, generation_id, source_class_hint, record_type_hint,
            hint_is_user_provided, limit, retrieval_query,
        )
        with self._lock:
            return self._cache.get(key)

    def put(
        self,
        *,
        query_text: str,
        generation_id: str,
        source_class_hint: str | None,
        record_type_hint: str | None,
        hint_is_user_provided: bool = False,
        limit: int = 5,
        retrieval_query: str | None = None,
        result: Any,
    ) -> None:
        key = self._build_key(
            query_text, generation_id, source_class_hint, record_type_hint,
            hint_is_user_provided, limit, retrieval_query,
        )
        with self._lock:
            self._cache[key] = result

    def get_early(
        self,
        *,
        query_text: str,
        generation_id: str,
        limit: int,
        user_hint: str | None,
    ) -> Any | None:
        """Pre-remote cache check using only locally-available fields.

        Tries exact match first, then near-duplicate (punctuation-stripped,
        token-sorted) to catch trivial reformulations.
        """
        key = self._build_early_key(query_text, generation_id, limit, user_hint)
        with self._lock:
            result = self._cache.get(key)
            if result is not None:
                return result
            nd_key = self._build_near_duplicate_key(
                query_text, generation_id, limit, user_hint
            )
            return self._cache.get(nd_key)

    def put_dual(
        self,
        *,
        query_text: str,
        generation_id: str,
        limit: int,
        user_hint: str | None,
        source_class_hint: str | None,
        record_type_hint: str | None,
        hint_is_user_provided: bool,
        retrieval_query: str | None,
        result: Any,
    ) -> None:
        """Store result under early, full, and near-duplicate keys."""
        early_key = self._build_early_key(query_text, generation_id, limit, user_hint)
        full_key = self._build_key(
            query_text, generation_id, source_class_hint, record_type_hint,
            hint_is_user_provided, limit, retrieval_query,
        )
        nd_key = self._build_near_duplicate_key(
            query_text, generation_id, limit, user_hint
        )
        with self._lock:
            self._cache[early_key] = result
            if full_key != early_key:
                self._cache[full_key] = result
            if nd_key != early_key:
                self._cache[nd_key] = result

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    @staticmethod
    def _build_early_key(
        query_text: str,
        generation_id: str,
        limit: int,
        user_hint: str | None,
    ) -> str:
        normalized_query = re.sub(r"\s+", " ", query_text.strip().lower())
        raw = repr((normalized_query, generation_id, limit, user_hint))
        return hashlib.sha256(raw.encode()).hexdigest()

    @staticmethod
    def _build_near_duplicate_key(
        query_text: str,
        generation_id: str,
        limit: int,
        user_hint: str | None,
    ) -> str:
        normalized_query = _normalize_near_duplicate(query_text)
        raw = repr(("nd", normalized_query, generation_id, limit, user_hint))
        return hashlib.sha256(raw.encode()).hexdigest()

    @staticmethod
    def _build_key(
        query_text: str,
        generation_id: str,
        source_class_hint: str | None,
        record_type_hint: str | None,
        hint_is_user_provided: bool,
        limit: int,
        retrieval_query: str | None = None,
    ) -> str:
        normalized_query = re.sub(r"\s+", " ", query_text.strip().lower())
        normalized_retrieval = (
            re.sub(r"\s+", " ", retrieval_query.strip().lower())
            if retrieval_query is not None
            else normalized_query
        )
        raw = repr((
            normalized_query,
            normalized_retrieval,
            generation_id,
            source_class_hint,
            record_type_hint,
            hint_is_user_provided,
            limit,
        ))
        return hashlib.sha256(raw.encode()).hexdigest()
