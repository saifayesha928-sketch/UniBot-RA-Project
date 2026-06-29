"""Compatibility facade for the update-cycle pipeline.

All business logic now lives under ``unibot.pipeline.*``.  This module
re-exports the public surface so that existing imports, monkeypatch targets,
and string-path attribute resolution (e.g.
``"unibot.scheduler.jobs.UpdateCycleOrchestrator"``) continue to work.
"""
from __future__ import annotations

import structlog

from unibot.indexing.serving_generation_builder import ServingGenerationBuilder  # noqa: F401

# ---------------------------------------------------------------------------
# Re-exports — contracts, result types, shared constants
# ---------------------------------------------------------------------------
from unibot.pipeline.contracts import (  # noqa: F401
    ArtifactFetcher,
    CandidateExtractor,
    CycleProgressCallback,
    SourceDiscoverer,
    UpdateCycleAudit,
    UpdateCycleResult,
    _CrawlPhaseResult,
    _ScopeProcessingResult,
    _EXPLICIT_PARENT_STATE_TYPES,
    _REMOVABLE_RECORD_TYPES,
    _SERVING_ELIGIBLE_STATUSES,
    _ZERO_CANONICAL_AUDIT_EXEMPT_SOURCE_CLASSES,
)

# ---------------------------------------------------------------------------
# Re-exports — orchestrator
# ---------------------------------------------------------------------------
from unibot.pipeline.orchestrator import UpdateCycleOrchestrator  # noqa: F401

# ---------------------------------------------------------------------------
# Re-exports — review resolution (used by admin API)
# ---------------------------------------------------------------------------
from unibot.pipeline.review_resolution import (  # noqa: F401
    recompute_scope_state,
    load_recompute_rows as _load_recompute_rows,
)

# ---------------------------------------------------------------------------
# Re-exports — source lifecycle
# ---------------------------------------------------------------------------
from unibot.pipeline.source_lifecycle import normalized_crawl_status as _normalized_crawl_status  # noqa: F401

# ---------------------------------------------------------------------------
# Re-exports — decision engine
# ---------------------------------------------------------------------------
from unibot.pipeline.decision_engine import (  # noqa: F401
    ParentStateLookup as _ParentStateLookup,
    candidate_from_row as _candidate_from_row,
    build_parent_state_lookup as _build_parent_state_lookup,
    classify_candidates_for_parent_resolution as _classify_candidates_for_parent_resolution,
    normalize_fetched_at as _normalize_fetched_at,
)

# ---------------------------------------------------------------------------
# Re-exports — provenance
# ---------------------------------------------------------------------------
from unibot.pipeline.provenance import provenance_locator_candidates as _provenance_locator_candidates  # noqa: F401

# ---------------------------------------------------------------------------
# Re-exports — audits
# ---------------------------------------------------------------------------
from unibot.pipeline.audits import (  # noqa: F401
    assert_audit_invariants as _assert_audit_invariants,
    find_sources_with_sections_but_no_canonical_records as _find_sources_with_sections_but_no_canonical_records,
    find_contradictory_scopes as _find_contradictory_scopes,
    find_active_duplicate_violations as _find_active_duplicate_violations,
)

# ---------------------------------------------------------------------------
# Module-level names that tests monkeypatch
# ---------------------------------------------------------------------------
# ``logger`` and ``ServingGenerationBuilder`` MUST remain on this module
# because existing tests patch them here and the orchestrator resolves them
# at call time via ``importlib.import_module("unibot.scheduler.jobs")``.
logger = structlog.get_logger()
