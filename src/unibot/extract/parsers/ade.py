from __future__ import annotations

from pathlib import Path

from landingai_ade import Client as AdeClient

from unibot.extract._geometry import as_bounding_box as _as_bounding_box
from unibot.extract._geometry import column_name as _column_name
from unibot.extract.documents import (
    ParsedDocument,
    ParsedDocumentSection,
    TableCellGrounding,
)


class AdeDocumentParser:
    def __init__(self, *, ade_model: str) -> None:
        self._ade_model = ade_model
        self._client = AdeClient()

    def parse(self, document_path: Path) -> ParsedDocument:
        response = self._client.parse(document=document_path, model=self._ade_model)
        grounding_map = getattr(response, "grounding", {}) or {}
        sections: list[ParsedDocumentSection] = []
        table_index_by_page: dict[int, int] = {}

        for chunk in getattr(response, "chunks", []):
            grounding = grounding_map.get(chunk.id) or getattr(chunk, "grounding", None)
            page_number = getattr(grounding, "page", None)
            box = _as_bounding_box(getattr(grounding, "box", None))
            chunk_type = str(getattr(chunk, "type", "text"))
            is_table = "table" in chunk_type.lower()

            table_cells: list[TableCellGrounding] = []
            source_locator = "Document Segment"
            if page_number is not None:
                source_locator = f"Page {page_number + 1}"

            if is_table and page_number is not None:
                table_index_by_page[page_number] = table_index_by_page.get(page_number, 0) + 1
                table_index = table_index_by_page[page_number]
                source_locator = f"Page {page_number + 1}, Table {table_index}"
                for grounded_id, grounded_value in grounding_map.items():
                    if getattr(grounded_value, "type", None) != "tableCell":
                        continue
                    if getattr(grounded_value, "page", None) != page_number:
                        continue
                    position = getattr(grounded_value, "position", None)
                    if position is None:
                        continue
                    row = int(getattr(position, "row", 0) or 0)
                    column = int(getattr(position, "col", 0) or 0)
                    table_cells.append(
                        TableCellGrounding(
                            row=row,
                            column=column,
                            rowspan=int(getattr(position, "rowspan", 1) or 1),
                            colspan=int(getattr(position, "colspan", 1) or 1),
                            bounding_box=_as_bounding_box(getattr(grounded_value, "box", None)),
                            source_locator=(
                                f"{source_locator}, Row {row} Column {_column_name(column)}"
                            ),
                        )
                    )

            sections.append(
                ParsedDocumentSection(
                    section_id=str(getattr(chunk, "id", f"chunk-{len(sections) + 1}")),
                    chunk_type=chunk_type,
                    content=str(getattr(chunk, "markdown", "")),
                    source_locator=source_locator,
                    page_number=page_number,
                    bounding_box=box,
                    table_cells=tuple(table_cells),
                )
            )

        metadata = getattr(response, "metadata", None)
        return ParsedDocument(
            markdown=str(getattr(response, "markdown", "")),
            parser_backend="ade",
            filename=str(getattr(metadata, "filename", document_path.name)),
            page_count=int(getattr(metadata, "page_count", 0) or 0),
            sections=tuple(sections),
            metadata={
                "job_id": getattr(metadata, "job_id", None),
                "duration_ms": getattr(metadata, "duration_ms", None),
            },
        )
