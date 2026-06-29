from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit

from unibot.settings import Settings


@dataclass(frozen=True, slots=True)
class BoundingBox:
    left: float
    top: float
    right: float
    bottom: float


@dataclass(frozen=True, slots=True)
class TableCellGrounding:
    row: int
    column: int
    rowspan: int = 1
    colspan: int = 1
    bounding_box: BoundingBox | None = None
    source_locator: str | None = None


@dataclass(frozen=True, slots=True)
class ParsedDocumentSection:
    section_id: str
    chunk_type: str
    content: str
    source_locator: str
    page_number: int | None = None
    sheet_name: str | None = None
    segment_id: str | None = None
    bounding_box: BoundingBox | None = None
    table_cells: tuple[TableCellGrounding, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    content_hash: str | None = None


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    markdown: str
    parser_backend: str
    filename: str
    page_count: int
    sections: tuple[ParsedDocumentSection, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


class DocumentParser(Protocol):
    def parse(self, document_path: Path) -> ParsedDocument: ...


def document_suffix_for_url(document_url: str) -> str:
    return Path(urlsplit(document_url).path).suffix or ".bin"


def get_document_parser(settings: Settings) -> DocumentParser:
    if settings.document_parser_backend == "docling":
        from unibot.extract.parsers.docling_parser import DoclingDocumentParser

        inner: DocumentParser = DoclingDocumentParser()
    else:
        import os

        if not os.environ.get("VISION_AGENT_API_KEY"):
            raise ValueError(
                "VISION_AGENT_API_KEY must be set when using the ADE document parser backend"
            )

        from unibot.extract.parsers.ade import AdeDocumentParser

        inner = AdeDocumentParser(ade_model=settings.ade_model)

    from unibot.extract.parsers.cached import CachedDocumentParser

    return CachedDocumentParser(inner=inner, cache_dir=settings.document_cache_dir)
