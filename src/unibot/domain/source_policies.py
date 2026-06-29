from __future__ import annotations

import re
from pathlib import PurePosixPath
from urllib.parse import urlsplit, urlunsplit

from unibot.domain.types import AccessLevel, SourceClass, SourcePolicy

HIGH_RISK_SOURCE_CLASSES = {
    "admissions_cycle",
    "program_fee_schedule",
    "merit_list",
}

SOURCE_FAMILIES: dict[str, str] = {
    "admissions_cycle": "admissions",
    "program": "admissions",
    "program_fee_schedule": "admissions",
    "merit_list": "admissions",
}


def get_source_family(source_class: str) -> str | None:
    """Return the reconciliation family for a source class, or None."""
    return SOURCE_FAMILIES.get(source_class)


_DOCUMENT_SUFFIXES = {
    ".pdf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".csv",
    ".ppt",
    ".pptx",
}

_ORG_UNIT_PATHS: set[str] = set()

_URL_ALIAS_MAP: dict[str, str] = {}


def _canonicalize_url(url: str) -> tuple[str, str, str]:
    parsed = urlsplit(url)
    hostname = (parsed.hostname or "").lower()
    path = parsed.path or "/"
    canonical_url = urlunsplit((parsed.scheme.lower(), hostname, path, "", ""))
    return hostname, path.lower(), canonical_url


def canonicalize_known_source_alias(url: str) -> str:
    parsed = urlsplit(url)
    hostname = (parsed.hostname or "").lower()
    path = parsed.path or "/"
    canonical_path = _URL_ALIAS_MAP.get(path.lower())
    if canonical_path is None:
        return url
    return urlunsplit((parsed.scheme.lower(), hostname, canonical_path, "", ""))


def legacy_source_url_aliases(url: str) -> tuple[str, ...]:
    parsed = urlsplit(url)
    hostname = (parsed.hostname or "").lower()
    path = parsed.path or "/"
    legacy_paths = tuple(
        legacy_path
        for legacy_path, canonical_path in _URL_ALIAS_MAP.items()
        if canonical_path == path.lower()
    )
    return tuple(
        urlunsplit((parsed.scheme.lower(), hostname, legacy_path, "", ""))
        for legacy_path in legacy_paths
    )


def _path_segment_contains(path: str, keyword: str) -> bool:
    """Check if keyword appears as a whole word in any path segment.

    Uses word-boundary matching to prevent substring false positives
    (e.g., 'fee' matching inside 'kafeel' in /faculty/dr-kafeel-sarwar/).
    """
    return bool(re.search(rf"\b{re.escape(keyword)}\b", path))


def _classify_source(hostname: str, path: str) -> SourceClass:
    """Best-effort source classification from a URL path.

    Generic, domain-agnostic heuristics. ``ExtractedRecord`` already carries an
    explicit ``record_type`` for ingested records, so this only acts as a
    fallback for URLs lacking one. Partners may extend these rules for their
    own site structure.
    """
    if PurePosixPath(path).suffix.lower() in _DOCUMENT_SUFFIXES:
        return "document_asset"
    if _path_segment_contains(path, "fee") or _path_segment_contains(path, "tuition"):
        return "program_fee_schedule"
    if "merit" in path:
        return "merit_list"
    if "scholarship" in path or "financial-assistance" in path or "financial-aid" in path:
        return "scholarship"
    if path.startswith("/admissions/") and path != "/admissions/":
        if not any(
            fragment in path
            for fragment in (
                "eligibility", "apply", "criteria", "calendar", "schedule",
                "faqs", "application-process", "fee-refund",
            )
        ):
            return "program"
    if "admission" in path:
        return "admissions_cycle"
    if "faculty" in path:
        return "faculty"
    if path == "/research/" or path.startswith("/research/"):
        return "research_main"
    if "news" in path or "event" in path:
        return "news_event"
    if "policy" in path or "regulation" in path:
        return "policy"
    if "student-service" in path or "placement" in path:
        return "student_service"
    if "library" in path:
        return "library"
    if (
        "about" in path
        or "information" in path
        or "academic-calendar" in path
        or "vision" in path
    ):
        return "university_info"
    return "general"


def _resolve_access_level(hostname: str) -> tuple[AccessLevel, bool]:
    """Default access policy: allow automated crawling.

    Partners operating the optional crawl path can special-case hostnames
    (e.g. application portals or library systems) that must be restricted or
    blocked.
    """
    return "allowed", True


def get_source_policy(url: str) -> SourcePolicy:
    hostname, path, canonical_url = _canonicalize_url(url)
    source_class = _classify_source(hostname, path)
    access_level, allows_automated_crawling = _resolve_access_level(hostname)

    return SourcePolicy(
        source_url=url,
        canonical_url=canonical_url,
        hostname=hostname,
        source_class=source_class,
        access_level=access_level,
        allows_automated_crawling=allows_automated_crawling,
        is_high_risk=source_class in HIGH_RISK_SOURCE_CLASSES,
    )
