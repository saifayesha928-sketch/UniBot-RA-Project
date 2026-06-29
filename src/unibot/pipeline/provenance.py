from __future__ import annotations

from dataclasses import replace

from sqlalchemy.orm import Session

from unibot.crawl.fetchers import FetchedArtifact
from unibot.db.models import SourceSection as SourceSectionModel
from unibot.extract.documents import ParsedDocument
from unibot.extract.records import (
    build_document_grounding,
    resolve_content_hash,
    serialize_for_storage,
)
from unibot.extract.sectionizer import sectionize_html
from unibot.verify.rules import VerificationCandidate


def provenance_locator_candidates(source_locator: str | None) -> tuple[str, ...]:
    if not source_locator:
        return ()

    candidates: list[str] = []
    current = source_locator
    while current and current not in candidates:
        candidates.append(current)
        if " " in current:
            current = current.rsplit(" ", 1)[0].strip()
            continue
        head, sep, tail = current.rpartition("/")
        if sep and ":" in tail:
            current = head
            continue
        break
    return tuple(candidates)


def attach_provenance(
    candidate: VerificationCandidate,
    *,
    page_content_hash: str,
    persisted_sections: dict[str, SourceSectionModel],
) -> VerificationCandidate:
    section = None
    for locator in provenance_locator_candidates(candidate.source_locator):
        section = persisted_sections.get(locator)
        if section is not None:
            break
    if section is None and persisted_sections:
        section = min(
            persisted_sections.values(),
            key=lambda s: s.section_order,
        )
    return replace(
        candidate,
        source_section_id=(
            section.source_section_id if section is not None else None
        ),
        source_section_label=(
            candidate.source_section_label
            or (section.section_label if section is not None else None)
        ),
        page_content_hash=page_content_hash,
    )


def persist_source_sections(
    session: Session,
    *,
    source_id: str,
    snapshot_id: str,
    artifact: FetchedArtifact,
) -> dict[str, SourceSectionModel]:
    parsed_document = artifact.metadata.get("parsed_document")
    persisted_sections: dict[str, SourceSectionModel] = {}

    if isinstance(parsed_document, ParsedDocument):
        for order, doc_section in enumerate(parsed_document.sections, start=1):
            content_hash = resolve_content_hash(doc_section.content, doc_section.content_hash)
            document_grounding = build_document_grounding(doc_section)
            row = SourceSectionModel(
                snapshot_id=snapshot_id,
                source_id=source_id,
                section_label=doc_section.source_locator,
                section_type=doc_section.chunk_type,
                source_locator=doc_section.source_locator,
                section_order=order,
                text_content=doc_section.content,
                source_text_hash=content_hash,
                parser_backend=parsed_document.parser_backend,
                page_number=doc_section.page_number,
                grounding_data={
                    "chunk_type": doc_section.chunk_type,
                    "section_id": doc_section.section_id,
                    "section_content_hash": content_hash,
                    "bounding_box": document_grounding["bounding_box"],
                    "table_cells": document_grounding["table_cells"],
                    "section_metadata": document_grounding["section_metadata"],
                    "sheet_name": doc_section.sheet_name,
                    "segment_id": doc_section.segment_id,
                },
            )
            session.add(row)
            persisted_sections[doc_section.source_locator] = row
        return persisted_sections

    if "html" not in artifact.content_type.lower():
        return {}

    html_text = artifact.content.decode("utf-8", errors="ignore")
    html_sections = sectionize_html(html=html_text, source_url=artifact.source_url)
    for order, section in enumerate(html_sections, start=1):
        content_hash = resolve_content_hash(section.content, section.content_hash)
        row = SourceSectionModel(
            snapshot_id=snapshot_id,
            source_id=source_id,
            section_label=section.section_label,
            section_type=section.section_type,
            source_locator=section.source_locator,
            section_order=order,
            text_content=section.content,
            source_text_hash=content_hash,
            parser_backend="html_sectionizer",
            grounding_data={
                **serialize_for_storage(section.metadata),
                "section_content_hash": content_hash,
            },
        )
        session.add(row)
        persisted_sections[section.source_locator] = row

    if persisted_sections:
        session.flush()
    return persisted_sections
