from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from unibot.extract.documents import (
    BoundingBox,
    DocumentParser,
    ParsedDocument,
    ParsedDocumentSection,
    TableCellGrounding,
)
from unibot.extract.records import serialize_bounding_box

logger = logging.getLogger(__name__)


class CachedDocumentParser:
    """Caching wrapper around any DocumentParser.

    Cache key is SHA-256 of document bytes (content-addressable).
    """

    def __init__(self, *, inner: DocumentParser, cache_dir: str | Path) -> None:
        self._inner = inner
        self._cache_dir = Path(cache_dir)

    def parse(self, document_path: Path) -> ParsedDocument:
        content_bytes = document_path.read_bytes()
        content_hash = hashlib.sha256(content_bytes).hexdigest()
        cache_file = self._cache_dir / f"{content_hash}.json"

        if cache_file.exists():
            try:
                cached = _load_from_cache(cache_file)
                logger.debug("document parser cache hit: %s", content_hash[:12])
                return cached
            except Exception:
                logger.warning(
                    "document parser cache corrupted, re-parsing: %s",
                    content_hash[:12],
                )

        result = self._inner.parse(document_path)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        _save_to_cache(cache_file, result)
        logger.debug("document parser cache stored: %s", content_hash[:12])
        return result

    def purge(self) -> int:
        """Delete all cached entries. Returns the number of files removed."""
        if not self._cache_dir.exists():
            return 0
        count = 0
        for f in self._cache_dir.glob("*.json"):
            f.unlink()
            count += 1
        return count


def _save_to_cache(path: Path, doc: ParsedDocument) -> None:
    data = {
        "markdown": doc.markdown,
        "parser_backend": doc.parser_backend,
        "filename": doc.filename,
        "page_count": doc.page_count,
        "metadata": doc.metadata,
        "sections": [_section_to_dict(s) for s in doc.sections],
    }
    path.write_text(json.dumps(data, ensure_ascii=False))


def _load_from_cache(path: Path) -> ParsedDocument:
    data = json.loads(path.read_text())
    sections = tuple(_section_from_dict(s) for s in data["sections"])
    return ParsedDocument(
        markdown=data["markdown"],
        parser_backend=data["parser_backend"],
        filename=data["filename"],
        page_count=data["page_count"],
        metadata=data.get("metadata", {}),
        sections=sections,
    )


def _section_to_dict(section: ParsedDocumentSection) -> dict[str, Any]:
    d: dict[str, Any] = {
        "section_id": section.section_id,
        "chunk_type": section.chunk_type,
        "content": section.content,
        "source_locator": section.source_locator,
        "page_number": section.page_number,
        "sheet_name": section.sheet_name,
        "segment_id": section.segment_id,
        "content_hash": section.content_hash,
        "metadata": section.metadata,
        "bounding_box": serialize_bounding_box(section.bounding_box),
        "table_cells": [_cell_to_dict(c) for c in section.table_cells],
    }
    return d


def _section_from_dict(d: dict[str, Any]) -> ParsedDocumentSection:
    return ParsedDocumentSection(
        section_id=d["section_id"],
        chunk_type=d["chunk_type"],
        content=d["content"],
        source_locator=d["source_locator"],
        page_number=d.get("page_number"),
        sheet_name=d.get("sheet_name"),
        segment_id=d.get("segment_id"),
        content_hash=d.get("content_hash"),
        metadata=d.get("metadata", {}),
        bounding_box=_bbox_from_dict(d.get("bounding_box")),
        table_cells=tuple(_cell_from_dict(c) for c in d.get("table_cells", [])),
    )


def _bbox_from_dict(d: dict[str, float] | None) -> BoundingBox | None:
    if d is None:
        return None
    return BoundingBox(left=d["left"], top=d["top"], right=d["right"], bottom=d["bottom"])


def _cell_to_dict(cell: TableCellGrounding) -> dict[str, Any]:
    return {
        "row": cell.row,
        "column": cell.column,
        "rowspan": cell.rowspan,
        "colspan": cell.colspan,
        "bounding_box": serialize_bounding_box(cell.bounding_box),
        "source_locator": cell.source_locator,
    }


def _cell_from_dict(d: dict[str, Any]) -> TableCellGrounding:
    return TableCellGrounding(
        row=d["row"],
        column=d["column"],
        rowspan=d.get("rowspan", 1),
        colspan=d.get("colspan", 1),
        bounding_box=_bbox_from_dict(d.get("bounding_box")),
        source_locator=d.get("source_locator"),
    )
