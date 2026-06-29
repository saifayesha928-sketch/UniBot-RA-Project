"""Dataclasses for the files you deliver to the backend.

- `SourceRegistryEntry`: one per URL or document, written to sources.json.
- `ExtractedRecord`: one per fact, written as one line in records.jsonl.
- `DocumentAssetRecord`: helper for `document_asset` records.

Serialize with `dataclasses.asdict(record)` to produce the JSON the ingester reads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .enums import (
    CrawlMethod,
    CrawlStatus,
    LegalStatus,
    ParserTarget,
    SourceClass,
    YearConfidence,
)


@dataclass(frozen=True, slots=True)
class SourceRegistryEntry:
    source_url: str
    canonical_url: str
    source_class: SourceClass
    crawl_method: CrawlMethod
    legal_status: LegalStatus
    default_authority_tier: int
    refresh_policy: str
    parser_target: ParserTarget = "html"
    crawl_status: CrawlStatus | None = None
    parent_source_url: str | None = None
    link_text: str | None = None
    is_active: bool = True
    last_crawled_at: datetime | None = None
    last_successful_crawl_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class ExtractedRecord:
    record_id: str
    record_type: str
    source_url: str
    source_section_id: str
    source_section_label: str
    source_locator: str
    source_authority_tier: int
    conflict_scope_id: str
    dedupe_key: str
    record_payload: dict[str, Any] = field(default_factory=dict)
    cycle_label: str | None = None
    year_confidence: YearConfidence = "unknown"


@dataclass(frozen=True, slots=True)
class DocumentAssetRecord:
    record_id: str
    document_url: str
    parent_page_url: str
    source_section_id: str
    source_locator: str
    parser_backend: str
    filename: str
    page_count: int
    source_authority_tier: int
    media_type: str = "application/octet-stream"
    document_kind: str = "unknown"
    parser_metadata: dict[str, Any] = field(default_factory=dict)

    def to_extracted_record(self) -> ExtractedRecord:
        payload: dict[str, Any] = {
            "document_url": self.document_url,
            "parent_page_url": self.parent_page_url,
            "parser_backend": self.parser_backend,
            "filename": self.filename,
            "page_count": self.page_count,
            "media_type": self.media_type,
            "document_kind": self.document_kind,
        }
        if self.parser_metadata:
            payload["parser_metadata"] = self.parser_metadata
        return ExtractedRecord(
            record_id=self.record_id,
            record_type="document_asset",
            source_url=self.document_url,
            source_section_id=self.source_section_id,
            source_section_label=self.filename,
            source_locator=self.source_locator,
            source_authority_tier=self.source_authority_tier,
            conflict_scope_id=self.record_id,
            dedupe_key=self.document_url,
            record_payload=payload,
        )
