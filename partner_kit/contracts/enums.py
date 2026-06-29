"""Enum literals and runtime allow-sets for the data contract.

Use the `Literal` aliases for static typing and the frozenset constants for
runtime validation of the values your records use.
"""

from __future__ import annotations

from typing import Literal

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

RecordType = Literal[
    "general",
    "admissions_cycle",
    "program",
    "program_curriculum",
    "program_fee_schedule",
    "merit_list",
    "faculty_profile",
    "faculty_publication",
    "faculty_award",
    "faculty_affiliation",
    "research_entity",
    "scholarship",
    "news_event",
    "policy_rule",
    "document_landing",
    "document_asset",
    "student_service",
    "university_info",
    "org_unit",
]

CrawlMethod = Literal["html_static", "browser", "wordpress_api"]
LegalStatus = Literal["allowed", "restricted", "blocked"]
CrawlStatus = Literal["unverified", "verified_live", "verified_placeholder", "blocked"]
ParserTarget = Literal["html", "document"]
YearConfidence = Literal["high", "medium", "low", "unknown"]

RefreshPolicy = Literal[
    "every_6_hours_while_active_daily_otherwise",
    "daily_during_season_weekly_otherwise",
    "weekly",
    "weekly_or_monthly_based_on_change_rate",
    "monthly",
]

SOURCE_CLASSES: frozenset[str] = frozenset(SourceClass.__args__)
RECORD_TYPES: frozenset[str] = frozenset(RecordType.__args__)
CRAWL_METHODS: frozenset[str] = frozenset(CrawlMethod.__args__)
LEGAL_STATUSES: frozenset[str] = frozenset(LegalStatus.__args__)
CRAWL_STATUSES: frozenset[str] = frozenset(CrawlStatus.__args__)
PARSER_TARGETS: frozenset[str] = frozenset(ParserTarget.__args__)
YEAR_CONFIDENCES: frozenset[str] = frozenset(YearConfidence.__args__)
REFRESH_POLICIES: frozenset[str] = frozenset(RefreshPolicy.__args__)

AUTHORITY_TIERS: frozenset[int] = frozenset({1, 2, 3, 4, 5})
