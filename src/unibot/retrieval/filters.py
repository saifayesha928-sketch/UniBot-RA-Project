from __future__ import annotations

from collections import defaultdict
from types import MappingProxyType
from typing import Protocol, TypeVar

from unibot.db.models import CanonicalRecord

QUERYABLE_SERVING_STATUSES = frozenset({"eligible", "pending_index", "indexed_active"})
BLOCKED_LEGAL_STATUSES = frozenset({"blocked", "restricted"})
AMBIGUOUS_YEAR_CONFIDENCE = frozenset({"low", "unknown"})


class PrimaryHitCandidate(Protocol):
    @property
    def record_version_id(self) -> str: ...
    @property
    def record_type(self) -> str: ...
    @property
    def source_url(self) -> str: ...
    @property
    def dedupe_key(self) -> str: ...
    @property
    def value_hash(self) -> str: ...
    @property
    def source_authority_tier(self) -> int: ...
    @property
    def score(self) -> float: ...


T = TypeVar("T", bound=PrimaryHitCandidate)


def payload_matches_active_generation(
    payload_generation_id: object,
    *,
    active_generation_id: str,
) -> bool:
    return str(payload_generation_id or "") == active_generation_id


def record_is_current_only(record: CanonicalRecord) -> bool:
    return (
        record.verification_status == "verified"
        and record.freshness_status == "current"
        and record.is_current_authoritative
        and record.serving_status in QUERYABLE_SERVING_STATUSES
    )


def source_is_query_allowed(legal_status: str | None) -> bool:
    return (legal_status or "allowed") not in BLOCKED_LEGAL_STATUSES


# When a source_class_hint is given, these additional source classes are
# also considered matching.  This handles content that logically belongs
# to one class but is stored under another (e.g. regulatory PDFs stored
# as "document_asset" that contain policy rules).
SOURCE_CLASS_EXPANSIONS: MappingProxyType[str, tuple[str, ...]] = MappingProxyType({
    "policy": ("policy", "document_asset"),
})


def expand_source_classes(source_class_hint: str) -> tuple[str, ...]:
    """Return source classes to query for the given hint.

    The expansion always includes the hint itself.  Additional classes are
    added when content logically belonging to one class is stored under
    another (e.g. regulatory PDFs stored as ``document_asset``).
    """
    return SOURCE_CLASS_EXPANSIONS.get(source_class_hint, (source_class_hint,))


def source_class_matches(source_class: str | None, *, source_class_hint: str | None) -> bool:
    if source_class_hint is None:
        return True
    return source_class in expand_source_classes(source_class_hint)


def record_type_matches(record_type: str | None, *, record_type_hint: str | None) -> bool:
    if record_type_hint is None:
        return True
    return record_type == record_type_hint


def enforce_source_diversity(
    candidates: tuple[T, ...] | list[T],
    *,
    max_per_source: int = 2,
    target_size: int | None = None,
) -> tuple[T, ...]:
    """Cap the number of candidates from any single ``source_url``.

    When a page is chunked into many small records (e.g. one ``<li>`` per
    record), sibling chunks with near-identical embeddings can dominate the
    top-K results.  This function keeps at most *max_per_source* candidates
    per ``source_url``, preferring higher-scored ones. If capping would leave
    fewer than *target_size* results and no alternative sources are available,
    overflow candidates are added back in score order so retrieval does not
    underfill. The output is sorted by descending score.
    """
    by_score = sorted(candidates, key=lambda c: (-c.score, c.record_version_id))
    if target_size is not None and target_size <= 0:
        return ()
    if max_per_source <= 0:
        if target_size is None:
            return tuple(by_score)
        return tuple(by_score[:target_size])

    seen_records: dict[str, set[str]] = {}  # source_url -> set of record_version_ids
    kept: list[T] = []
    overflow: list[T] = []
    for candidate in by_score:
        url = candidate.source_url or candidate.record_version_id
        rvid = candidate.record_version_id
        url_records = seen_records.setdefault(url, set())
        if rvid in url_records:
            # Already admitted this record — allow all its chunks through
            kept.append(candidate)
            continue
        if len(url_records) < max_per_source:
            url_records.add(rvid)
            kept.append(candidate)
            continue
        overflow.append(candidate)

    if target_size is not None:
        desired_size = min(target_size, len(by_score))
        if len(kept) < desired_size:
            kept.extend(overflow[: desired_size - len(kept)])
        kept = kept[:desired_size]

    return tuple(sorted(kept, key=lambda c: (-c.score, c.record_version_id)))


def select_with_type_reservation(
    candidates: tuple[T, ...] | list[T],
    *,
    record_type_hint: str | None,
    limit: int = 5,
    reserved_slots: int = 2,
) -> tuple[T, ...]:
    """Select top-*limit* candidates, reserving slots for the hinted type.

    When *record_type_hint* is set, up to *reserved_slots* of the final
    results are guaranteed to come from that record type (if enough exist).
    The remaining slots are filled by the highest-scoring candidates of any
    type.  The final output is always sorted by descending score.

    When *record_type_hint* is ``None``, this falls back to pure
    score-based selection (identical to taking the top-*limit* by score).
    """
    by_score = sorted(candidates, key=lambda c: (-c.score, c.record_version_id))

    if limit <= 0:
        return ()

    if not record_type_hint:
        return tuple(by_score[:limit])

    reserved_limit = min(max(reserved_slots, 0), limit)
    hinted = [c for c in by_score if c.record_type == record_type_hint]
    reserved = hinted[:reserved_limit]
    reserved_ids = {c.record_version_id for c in reserved}

    remaining = [c for c in by_score if c.record_version_id not in reserved_ids]
    remaining_slots = max(limit - len(reserved), 0)
    rest = remaining[:remaining_slots]

    return tuple(sorted(reserved + rest, key=lambda c: (-c.score, c.record_version_id)))


def select_primary_hits(candidates: tuple[T, ...] | list[T]) -> tuple[T, ...]:
    grouped: dict[str, list[T]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.dedupe_key].append(candidate)

    primary_hits: list[T] = []
    for grouped_candidates in grouped.values():
        winning_tier = min(
            candidate.source_authority_tier for candidate in grouped_candidates
        )
        winning_candidates = [
            candidate
            for candidate in grouped_candidates
            if candidate.source_authority_tier == winning_tier
        ]
        winning_candidates.sort(
            key=lambda candidate: (-candidate.score, candidate.record_version_id)
        )

        distinct_values = {candidate.value_hash for candidate in winning_candidates}
        if len(distinct_values) > 1:
            primary_hits.extend(winning_candidates)
            continue

        # Single value_hash: keep all chunks from the best-scoring record only.
        # This preserves multi-chunk coverage while still deduplicating across
        # records from different sources that carry identical content.
        best = winning_candidates[0]
        primary_hits.extend(
            candidate
            for candidate in winning_candidates
            if candidate.record_version_id == best.record_version_id
        )

    return tuple(
        sorted(
            primary_hits,
            key=lambda candidate: (
                -candidate.score,
                candidate.source_authority_tier,
                candidate.record_version_id,
            ),
        )
    )
