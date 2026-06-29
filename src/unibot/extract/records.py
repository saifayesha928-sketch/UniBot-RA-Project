from __future__ import annotations

import hashlib
import mimetypes
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Literal

from unibot.extract.documents import (
    BoundingBox,
    ParsedDocument,
    ParsedDocumentSection,
    TableCellGrounding,
)

YearConfidence = Literal["high", "medium", "low", "unknown"]
_SKIP = object()


@dataclass(frozen=True, slots=True)
class SourceSection:
    section_id: str
    section_label: str
    section_type: str
    source_locator: str
    source_url: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    content_hash: str | None = None
    element: Any = None


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
class EvidenceRecord:
    record_id: str
    source_url: str
    source_section_id: str
    source_section_label: str
    source_locator: str
    source_authority_tier: int
    conflict_scope_id: str
    dedupe_key: str
    value_text: str
    record_type: str = "evidence"
    record_payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    records: tuple[ExtractedRecord, ...] = ()
    evidence_records: tuple[EvidenceRecord, ...] = ()
    parsed_document: ParsedDocument | None = None


@dataclass(frozen=True, slots=True)
class ExtractionContext:
    source_class: str
    source_url: str
    html: str | None = None
    markdown: str | None = None
    parser_target: str = "html"
    default_authority_tier: int = 1
    parent_source_url: str | None = None
    link_text: str | None = None
    verification_state_by_url: dict[str, str] = field(default_factory=dict)
    document_bytes: bytes | None = None
    document_content_type: str | None = None
    parsed_document: ParsedDocument | None = None
    fetch_metadata: dict[str, Any] = field(default_factory=dict)
    artifact_identity_status: str | None = None


@dataclass(frozen=True, slots=True)
class DocumentAssetRecord:
    record_id: str
    record_type: str
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
        payload = {
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
            record_type=self.record_type,
            source_url=self.document_url,
            source_section_id=self.source_section_id,
            source_section_label=self.filename,
            source_locator=self.source_locator,
            source_authority_tier=self.source_authority_tier,
            conflict_scope_id=self.record_id,
            dedupe_key=self.document_url,
            record_payload=payload,
        )


def _derive_media_type(filename: str) -> str:
    guessed, _ = mimetypes.guess_type(filename)
    return guessed or "application/octet-stream"


_MEDIA_TYPE_TO_DOCUMENT_KIND: dict[str, str] = {
    "application/pdf": "pdf",
    "application/msword": "word_document",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "word_document",
    "application/vnd.ms-excel": "spreadsheet",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "spreadsheet",
    "application/vnd.ms-powerpoint": "presentation",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "presentation",
}


def _derive_document_kind(media_type: str) -> str:
    if media_type in _MEDIA_TYPE_TO_DOCUMENT_KIND:
        return _MEDIA_TYPE_TO_DOCUMENT_KIND[media_type]
    if media_type.startswith("image/"):
        return "image"
    return "unknown"


def build_document_asset_record(
    *,
    document_url: str,
    parent_page_url: str,
    parsed_document: ParsedDocument,
    source_authority_tier: int,
) -> DocumentAssetRecord:
    primary_section = parsed_document.sections[0] if parsed_document.sections else None
    record_id = hashlib.sha256(document_url.encode("utf-8")).hexdigest()[:16]
    media_type = _derive_media_type(parsed_document.filename)
    document_kind = _derive_document_kind(media_type)
    return DocumentAssetRecord(
        record_id=f"document_asset:{record_id}",
        record_type="document_asset",
        document_url=document_url,
        parent_page_url=parent_page_url,
        source_section_id=primary_section.section_id if primary_section else "document-root",
        source_locator=primary_section.source_locator if primary_section else "Document Root",
        parser_backend=parsed_document.parser_backend,
        filename=parsed_document.filename,
        page_count=parsed_document.page_count,
        source_authority_tier=source_authority_tier,
        media_type=media_type,
        document_kind=document_kind,
        parser_metadata=serialize_for_storage(parsed_document.metadata),
    )


def resolve_content_hash(content: str, explicit_hash: str | None = None) -> str:
    return explicit_hash or hashlib.sha256(content.encode("utf-8")).hexdigest()


def serialize_bounding_box(box: BoundingBox | None) -> dict[str, float] | None:
    if box is None:
        return None
    return {
        "left": float(box.left),
        "top": float(box.top),
        "right": float(box.right),
        "bottom": float(box.bottom),
    }


def serialize_table_cells(
    table_cells: tuple[TableCellGrounding, ...],
) -> list[dict[str, Any]]:
    return [
        {
            "row": cell.row,
            "column": cell.column,
            "rowspan": cell.rowspan,
            "colspan": cell.colspan,
            "source_locator": cell.source_locator,
            "bounding_box": serialize_bounding_box(cell.bounding_box),
        }
        for cell in table_cells
    ]


def build_document_grounding(section: ParsedDocumentSection) -> dict[str, Any]:
    return {
        "section_id": section.section_id,
        "segment_id": section.segment_id,
        "page_number": section.page_number,
        "sheet_name": section.sheet_name,
        "bounding_box": serialize_bounding_box(section.bounding_box),
        "table_cells": serialize_table_cells(section.table_cells),
        "section_metadata": serialize_for_storage(section.metadata),
    }


def build_document_evidence_payload(
    *,
    parent_page_url: str,
    parsed_document: ParsedDocument,
    section: ParsedDocumentSection,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "parent_page_url": parent_page_url,
        "parser_backend": parsed_document.parser_backend,
        "page_number": section.page_number,
        "chunk_type": section.chunk_type,
        "sheet_name": section.sheet_name,
        "section_content_hash": resolve_content_hash(section.content, section.content_hash),
        "document_grounding": build_document_grounding(section),
    }
    return payload


def serialize_for_storage(value: Any) -> Any:
    serialized = _serialize_for_storage(value)
    return {} if serialized is _SKIP else serialized


def _serialize_for_storage(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            serialized = _serialize_for_storage(item)
            if serialized is _SKIP:
                continue
            result[str(key)] = serialized
        return result
    if isinstance(value, (list, tuple, set)):
        result_list: list[Any] = []
        for item in value:
            serialized = _serialize_for_storage(item)
            if serialized is _SKIP:
                continue
            result_list.append(serialized)
        return result_list
    if is_dataclass(value) and not isinstance(value, type):
        return _serialize_for_storage(asdict(value))
    try:
        public_attrs = {
            key: item
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    except TypeError:
        public_attrs = {}
    if public_attrs:
        return _serialize_for_storage(public_attrs)
    return str(value)


def resolve_extraction_context(
    *,
    source_class: str,
    context: ExtractionContext | None = None,
    html: str | None = None,
    markdown: str | None = None,
    source_url: str | None = None,
    parser_target: str = "html",
    default_authority_tier: int = 1,
    parent_source_url: str | None = None,
    link_text: str | None = None,
    verification_state_by_url: dict[str, str] | None = None,
    fetch_metadata: dict[str, Any] | None = None,
    artifact_identity_status: str | None = None,
) -> ExtractionContext:
    if context is not None:
        return context
    if source_url is None:
        raise ValueError("source_url is required when context is not provided")
    return ExtractionContext(
        source_class=source_class,
        source_url=source_url,
        html=html,
        markdown=markdown,
        parser_target=parser_target,
        default_authority_tier=default_authority_tier,
        parent_source_url=parent_source_url,
        link_text=link_text,
        verification_state_by_url=verification_state_by_url or {},
        fetch_metadata=fetch_metadata or {},
        artifact_identity_status=artifact_identity_status,
    )
