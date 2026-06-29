from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from unibot.extract._geometry import as_bounding_box as _as_bounding_box
from unibot.extract._geometry import column_name as _column_name
from unibot.extract.documents import (
    BoundingBox,
    ParsedDocument,
    ParsedDocumentSection,
    TableCellGrounding,
)


class DoclingDocumentParser:
    def parse(self, document_path: Path) -> ParsedDocument:
        from docling.document_converter import DocumentConverter  # type: ignore[import-not-found]

        converter = DocumentConverter()
        conversion = converter.convert(document_path)
        document = conversion.document

        sections = _iter_rich_sections(document)
        if not sections:
            sections = _iter_legacy_sections(document)

        if not sections:
            markdown = ""
            if hasattr(document, "export_to_markdown"):
                markdown = str(document.export_to_markdown())
            sections.append(
                ParsedDocumentSection(
                    section_id="section-1",
                    chunk_type="section",
                    content=markdown,
                    source_locator="Section: Document",
                    segment_id="section-1",
                    content_hash=_hash_text(markdown),
                )
            )

        raw_page_count = getattr(document, "page_count", None)
        if raw_page_count is None:
            pages = getattr(document, "pages", None)
            raw_page_count = len(pages) if pages else 0
        page_count = int(raw_page_count)
        markdown = ""
        export_to_markdown = getattr(document, "export_to_markdown", None)
        if callable(export_to_markdown):
            markdown = str(export_to_markdown())

        return ParsedDocument(
            markdown=markdown,
            parser_backend="docling",
            filename=document_path.name,
            page_count=page_count,
            sections=tuple(sections),
            metadata=_document_metadata(document),
        )


def _iter_rich_sections(document: Any) -> list[ParsedDocumentSection]:
    iterate_items = getattr(document, "iterate_items", None)
    if not callable(iterate_items):
        return []

    sections: list[ParsedDocumentSection] = []
    table_index_by_page: dict[int, int] = {}
    for raw_index, item_info in enumerate(iterate_items(), start=1):
        item = item_info[0] if isinstance(item_info, tuple) else item_info
        self_ref = getattr(item, "self_ref", None)
        if isinstance(self_ref, str) and "/table_cells/" in self_ref:
            continue

        content = _item_markdown(item).strip()
        if not content:
            continue

        chunk_type = _item_chunk_type(item)
        page_number = _item_page_number(item)
        source_locator = _item_source_locator(
            item=item,
            chunk_type=chunk_type,
            page_number=page_number,
            default_index=raw_index,
            table_index_by_page=table_index_by_page,
        )
        sections.append(
            ParsedDocumentSection(
                section_id=str(self_ref or f"section-{raw_index}"),
                chunk_type=chunk_type,
                content=content,
                source_locator=source_locator,
                page_number=page_number,
                sheet_name=getattr(item, "sheet_name", None),
                segment_id=str(self_ref) if self_ref is not None else None,
                bounding_box=_item_bounding_box(item),
                table_cells=tuple(_item_table_cells(item, source_locator)),
                metadata=_item_metadata(item),
                content_hash=_hash_text(content),
            )
        )

    return sections


def _iter_legacy_sections(document: Any) -> list[ParsedDocumentSection]:
    sections: list[ParsedDocumentSection] = []

    for index, section in enumerate(getattr(document, "sections", []), start=1):
        label = getattr(section, "heading", None) or f"Section {index}"
        content = str(getattr(section, "markdown", ""))
        sections.append(
            ParsedDocumentSection(
                section_id=f"section-{index}",
                chunk_type="section",
                content=content,
                source_locator=f"Section: {label}",
                page_number=getattr(section, "page_number", None),
                segment_id=f"section-{index}",
                content_hash=_hash_text(content),
            )
        )

    for index, sheet in enumerate(getattr(document, "sheets", []), start=1):
        name = getattr(sheet, "name", f"Sheet{index}")
        content = str(getattr(sheet, "markdown", ""))
        sections.append(
            ParsedDocumentSection(
                section_id=f"sheet-{index}",
                chunk_type="sheet",
                content=content,
                source_locator=f"Sheet: {name}",
                sheet_name=name,
                segment_id=f"sheet-{index}",
                content_hash=_hash_text(content),
            )
        )

    return sections


def _item_markdown(item: Any) -> str:
    markdown = getattr(item, "markdown", None)
    if markdown:
        return str(markdown)
    export_to_markdown = getattr(item, "export_to_markdown", None)
    if callable(export_to_markdown):
        return str(export_to_markdown())
    text = getattr(item, "text", None)
    if text:
        return str(text)
    return ""


def _item_chunk_type(item: Any) -> str:
    label = str(getattr(item, "label", "")).lower()
    if "table" in label:
        return "table"
    if "sheet" in label:
        return "sheet"
    return label or "section"


def _item_page_number(item: Any) -> int | None:
    prov = getattr(item, "prov", None) or []
    if not prov:
        return None
    page_no = getattr(prov[0], "page_no", None)
    if page_no is None:
        return None
    return max(int(page_no) - 1, 0)


def _item_source_locator(
    *,
    item: Any,
    chunk_type: str,
    page_number: int | None,
    default_index: int,
    table_index_by_page: dict[int, int],
) -> str:
    if chunk_type == "table" and page_number is not None:
        table_index_by_page[page_number] = table_index_by_page.get(page_number, 0) + 1
        return f"Page {page_number + 1}, Table {table_index_by_page[page_number]}"

    sheet_name = getattr(item, "sheet_name", None)
    if sheet_name:
        return f"Sheet: {sheet_name}"

    heading = str(getattr(item, "text", "")).strip().splitlines()[0:1]
    if heading and heading[0]:
        return f"Section: {heading[0]}"
    return f"Section: Item {default_index}"


def _item_bounding_box(item: Any) -> BoundingBox | None:
    prov = getattr(item, "prov", None) or []
    if not prov:
        return None
    return _as_bounding_box(getattr(prov[0], "bbox", None))


def _item_table_cells(item: Any, table_locator: str) -> list[TableCellGrounding]:
    raw_cells = getattr(getattr(item, "data", None), "table_cells", None)
    if not raw_cells:
        raw_cells = getattr(item, "table_cells", None) or []

    table_cells: list[TableCellGrounding] = []
    for raw_cell in raw_cells:
        row = _row_or_column_index(
            getattr(raw_cell, "row", None),
            getattr(raw_cell, "start_row_offset_idx", None),
        )
        column = _row_or_column_index(
            getattr(raw_cell, "column", None),
            getattr(raw_cell, "start_col_offset_idx", None),
        )
        locator = None
        if row is not None and column is not None:
            locator = f"{table_locator}, Row {row} Column {_column_name(column)}"
        table_cells.append(
            TableCellGrounding(
                row=row or 1,
                column=column or 1,
                rowspan=_span(
                    getattr(raw_cell, "rowspan", None),
                    getattr(raw_cell, "row_span", None),
                    getattr(raw_cell, "start_row_offset_idx", None),
                    getattr(raw_cell, "end_row_offset_idx", None),
                ),
                colspan=_span(
                    getattr(raw_cell, "colspan", None),
                    getattr(raw_cell, "col_span", None),
                    getattr(raw_cell, "start_col_offset_idx", None),
                    getattr(raw_cell, "end_col_offset_idx", None),
                ),
                bounding_box=_as_bounding_box(getattr(raw_cell, "bbox", None)),
                source_locator=locator,
            )
        )

    return table_cells


def _row_or_column_index(explicit: Any, offset_index: Any) -> int | None:
    if explicit is not None:
        return int(explicit)
    if offset_index is not None:
        return int(offset_index) + 1
    return None


def _span(explicit: Any, alternate: Any, start: Any, end: Any) -> int:
    if explicit is not None:
        return int(explicit)
    if alternate is not None:
        return int(alternate)
    if start is not None and end is not None:
        return int(end) - int(start) + 1
    return 1


def _item_metadata(item: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    self_ref = getattr(item, "self_ref", None)
    if self_ref is not None:
        metadata["docling_self_ref"] = str(self_ref)
    label = getattr(item, "label", None)
    if label is not None:
        metadata["docling_label"] = str(label)
    prov = getattr(item, "prov", None) or []
    if prov:
        metadata["docling_provenance"] = [
            _serialize_provenance_item(raw_prov) for raw_prov in prov
        ]
    return metadata


def _serialize_provenance_item(raw_prov: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    page_no = getattr(raw_prov, "page_no", None)
    if page_no is not None:
        payload["page_number"] = max(int(page_no) - 1, 0)
    bbox = _as_bounding_box(getattr(raw_prov, "bbox", None))
    if bbox is not None:
        payload["bounding_box"] = {
            "left": bbox.left,
            "top": bbox.top,
            "right": bbox.right,
            "bottom": bbox.bottom,
        }
    return payload


def _document_metadata(document: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {"schema_name": type(document).__name__}
    origin = getattr(document, "origin", None)
    if origin is not None:
        origin_payload = {
            key: value
            for key, value in vars(origin).items()
            if not key.startswith("_")
        }
        if origin_payload:
            metadata["origin"] = origin_payload
    return metadata



def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
