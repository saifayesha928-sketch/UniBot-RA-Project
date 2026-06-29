from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from unibot.crawl.jobs import CrawlJob
from unibot.db.repositories.source_registry import SourceRegistryEntry
from unibot.retrieval.filters import QUERYABLE_SERVING_STATUSES as _SERVING_ELIGIBLE_STATUSES
from unibot.verify.rules import VerificationCandidate
from unibot.verify.source_class_currentness import (
    CONTEXT_DEPENDENT_TYPES,
    PARENT_DEPENDENT_TYPES,
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_REMOVABLE_RECORD_TYPES = {
    "faculty_profile",
    "faculty_publication",
    "faculty_award",
    "faculty_affiliation",
    "merit_list",
    "document_asset",
}

_ZERO_CANONICAL_AUDIT_EXEMPT_SOURCE_CLASSES = {"document_landing", "research_main"}

_EXPLICIT_PARENT_STATE_TYPES = PARENT_DEPENDENT_TYPES | CONTEXT_DEPENDENT_TYPES

# Re-export for downstream modules
SERVING_ELIGIBLE_STATUSES = _SERVING_ELIGIBLE_STATUSES

# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


class ArtifactFetcher(Protocol):
    def __call__(self, job: CrawlJob) -> object:
        pass


class CandidateExtractor(Protocol):
    def __call__(
        self,
        job: CrawlJob,
        artifact: object,
    ) -> tuple[VerificationCandidate, ...] | list[VerificationCandidate]:
        pass


class SourceDiscoverer(Protocol):
    def __call__(
        self,
        job: CrawlJob,
        artifact: object,
    ) -> tuple[SourceRegistryEntry, ...] | list[SourceRegistryEntry]:
        pass


class CycleProgressCallback(Protocol):
    def on_phase_start(self, phase: str, total: int | None = None) -> None: ...
    def on_source_start(self, source_url: str, step: str) -> None: ...
    def on_source_done(self, source_url: str, records: int) -> None: ...
    def on_source_failed(self, source_url: str, error: str) -> None: ...
    def on_sources_discovered(self, count: int) -> None: ...
    def on_phase_done(self, phase: str) -> None: ...
    def on_generation_step(self, step: str, detail: str) -> None: ...
    def on_generation_progress(self, completed: int, total: int) -> None: ...


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class UpdateCycleAudit:
    active_index_count: int
    canonical_eligible_count: int
    blocked_or_inactive_record_version_ids: tuple[str, ...]
    contradictory_scope_ids: tuple[str, ...]
    duplicate_rule_violations: tuple[str, ...]
    sources_with_sections_but_no_canonical_records: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _CrawlPhaseResult:
    processed_source_urls: list[str]
    fetched_source_urls: list[str]
    pending_deindex_record_version_ids: list[str]
    verification_event_ids: list[str]


@dataclass(frozen=True, slots=True)
class UpdateCycleResult:
    selected_source_urls: tuple[str, ...]
    fetched_source_urls: tuple[str, ...]
    pending_deindex_record_version_ids: tuple[str, ...]
    verification_event_ids: tuple[str, ...]
    active_record_version_ids: tuple[str, ...]
    audit: UpdateCycleAudit
    generation_label: str


@dataclass(frozen=True, slots=True)
class _ScopeProcessingResult:
    pending_deindex_record_version_ids: tuple[str, ...]
    verification_event_ids: tuple[str, ...]
