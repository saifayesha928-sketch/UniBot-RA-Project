from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

AccessLevel = Literal["allowed", "restricted", "blocked"]
SourceClass = Literal[
    "general",
    "admissions_cycle",
    "program",
    "program_fee_schedule",
    "merit_list",
    "faculty",
    "research_main",
    "research_subdomain",
    "library",
    "application_portal",
    "scholarship",
    "news_event",
    "policy",
    "document_landing",
    "document_asset",
    "student_service",
    "university_info",
    "org_unit",
    "institutional_page",
    "department_directory",
    "qec_page",
    "qec_directory",
    "qec_document_landing",
    "qec_report_matrix",
    "unesco_page",
    "unesco_directory",
    "unesco_report_index",
    "oric_page",
    "career_page",
    "office_page",
    "tender_landing",
    "legal_act_rule",
    "rti_disclosure",
    "scholarship_notice",
]
AuthorityDecisionStatus = Literal["authoritative", "contradictory", "insufficient"]
PageKind = Literal[
    "dedicated_child",
    "dedicated_section",
    "official_document",
    "overview",
    "navigation",
]
ContentKind = Literal[
    "page_body",
    "structured_section",
    "table",
    "official_document",
    "summary",
    "menu",
    "footer",
    "card",
]


@dataclass(frozen=True, slots=True)
class SourcePolicy:
    source_url: str
    canonical_url: str
    hostname: str
    source_class: SourceClass
    access_level: AccessLevel
    allows_automated_crawling: bool
    is_high_risk: bool


@dataclass(frozen=True, slots=True)
class SourceContext:
    source_url: str
    page_kind: PageKind
    content_kind: ContentKind


@dataclass(frozen=True, slots=True)
class AuthorityRecord:
    record_id: str
    conflict_scope_id: str
    dedupe_key: str
    value_hash: str
    source_authority_tier: int
    is_current: bool
    is_verified: bool = True


@dataclass(frozen=True, slots=True)
class AuthorityDecision:
    status: AuthorityDecisionStatus
    primary_record: AuthorityRecord | None
    supporting_records: tuple[AuthorityRecord, ...]
    conflicting_records: tuple[AuthorityRecord, ...]
