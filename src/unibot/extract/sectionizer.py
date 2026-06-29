from __future__ import annotations

import re
from collections.abc import Iterable
from collections import defaultdict

from bs4 import BeautifulSoup, Tag

from unibot.extract.records import SourceSection
from unibot.extract.text import slugify

_SECTION_SELECTOR = "nav, section, article, table"
_FUSION_ROW_SELECTOR = "main .fusion-fullwidth"


def _slugify(value: str) -> str:
    return slugify(value) or "section"


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _iter_selected_tags(soup: BeautifulSoup) -> Iterable[Tag]:
    seen: set[int] = set()
    for tag in soup.select(_SECTION_SELECTOR):
        tag_id = id(tag)
        if tag_id in seen:
            continue
        seen.add(tag_id)
        yield tag


def _iter_fusion_rows(soup: BeautifulSoup) -> Iterable[Tag]:
    seen: set[int] = set()
    for tag in soup.select(_FUSION_ROW_SELECTOR):
        tag_id = id(tag)
        if tag_id in seen:
            continue
        seen.add(tag_id)
        yield tag


def _build_dom_path(tag: Tag) -> str:
    if tag.get("id"):
        return f"#{tag['id']}"

    segments: list[str] = []
    current: Tag | None = tag
    while current is not None and current.name not in {"[document]", "html"}:
        parent = current.parent if isinstance(current.parent, Tag) else None
        if parent is None:
            segments.append(current.name)
            break

        siblings = [child for child in parent.find_all(current.name, recursive=False)]
        index = siblings.index(current) + 1
        segments.append(f"{current.name}:nth-of-type({index})")
        current = parent

    return " > ".join(reversed(segments))


def _label_for_tag(tag: Tag) -> str:
    if tag.name == "table":
        caption = tag.find("caption")
        if caption is not None and _normalize_text(caption.get_text(" ", strip=True)):
            return _normalize_text(caption.get_text(" ", strip=True))

    heading = tag.find(re.compile("^h[1-6]$"))
    if heading is not None and _normalize_text(heading.get_text(" ", strip=True)):
        return _normalize_text(heading.get_text(" ", strip=True))

    if tag.name == "nav":
        return "Navigation"

    if tag.get("aria-label"):
        return _normalize_text(str(tag["aria-label"]))

    if tag.get("id"):
        return _slugify(str(tag["id"])).replace("-", " ").title()

    return tag.name.title()


def _nearest_parent_context(tag: Tag) -> Tag | None:
    current = tag.parent if isinstance(tag.parent, Tag) else None
    while current is not None and current.name not in {"[document]", "html", "body"}:
        if current.name in {"section", "article"}:
            return current
        if current.get("id") or current.get("aria-label"):
            return current
        current = current.parent if isinstance(current.parent, Tag) else None
    return None


def _fusion_title_text(tag: Tag) -> str | None:
    title = tag.select_one(".fusion-title")
    if title is None:
        return None
    text = _normalize_text(title.get_text(" ", strip=True))
    return text or None


def _table_label_within_row(table: Tag, row: Tag, row_label: str) -> str:
    caption = table.find("caption")
    if caption is not None:
        text = _normalize_text(caption.get_text(" ", strip=True))
        if text:
            return text

    headings: list[str] = []
    for node in row.find_all(True):
        if node is table:
            break
        if node.name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            text = _normalize_text(node.get_text(" ", strip=True))
            if text:
                headings.append(text)
                continue
        if "fusion-title" in (node.get("class") or []):
            text = _normalize_text(node.get_text(" ", strip=True))
            if text:
                headings.append(text)

    if headings:
        return headings[-1]
    return row_label


def _normalize_content(tag: Tag) -> str:
    raw_content = tag.get_text("\n", strip=True)
    content_lines = [
        normalized_line
        for line in raw_content.splitlines()
        if (normalized_line := _normalize_text(line))
    ]
    return "\n".join(content_lines)


def sectionize_html(*, html: str, source_url: str) -> tuple[SourceSection, ...]:
    soup = BeautifulSoup(html, "lxml")
    sections: list[SourceSection] = []
    semantic_counts: dict[str, int] = defaultdict(int)

    fusion_rows = tuple(_iter_fusion_rows(soup))
    if fusion_rows:
        for row in fusion_rows:
            content = _normalize_content(row)
            if not content:
                continue
            label = _fusion_title_text(row) or _label_for_tag(row)
            slug = _slugify(label)
            semantic_counts[slug] += 1
            suffix = f"-{semantic_counts[slug]}" if semantic_counts[slug] > 1 else ""
            locator = f"main://{slug}{suffix}"
            sections.append(
                SourceSection(
                    section_id=f"{slug}{suffix}",
                    section_label=label,
                    section_type="fusion_row",
                    source_locator=locator,
                    source_url=source_url,
                    content=content,
                    metadata={
                        "dom_path": _build_dom_path(row),
                        "semantic_locator": locator,
                    },
                )
            )

            table_counts: dict[str, int] = defaultdict(int)
            for table in row.select("table"):
                table_content = _normalize_content(table)
                if not table_content:
                    continue
                table_label = _table_label_within_row(table, row, label)
                table_slug = _slugify(table_label)
                table_counts[table_slug] += 1
                table_suffix = (
                    f"-{table_counts[table_slug]}"
                    if table_counts[table_slug] > 1
                    else ""
                )
                table_locator = f"{locator}/table:{table_slug}{table_suffix}"
                sections.append(
                    SourceSection(
                        section_id=f"{slug}{suffix}-table-{table_slug}{table_suffix}",
                        section_label=table_label,
                        section_type="table",
                        source_locator=table_locator,
                        source_url=source_url,
                        content=table_content,
                        metadata={
                            "dom_path": _build_dom_path(table),
                            "parent_locator": locator,
                        },
                        element=table,
                    )
                )

        return tuple(sections)

    for order, tag in enumerate(_iter_selected_tags(soup), start=1):
        content = _normalize_content(tag)
        if not content:
            continue

        locator = _build_dom_path(tag)
        label = _label_for_tag(tag)
        section_id = str(tag.get("id") or f"{_slugify(label)}-{order}")
        section_type = "table" if tag.name == "table" else tag.name
        metadata = {"dom_path": locator}
        if section_type == "table":
            parent_tag = _nearest_parent_context(tag)
            if parent_tag is not None:
                metadata["parent_locator"] = _build_dom_path(parent_tag)
                metadata["parent_label"] = _label_for_tag(parent_tag)
        sections.append(
            SourceSection(
                section_id=section_id,
                section_label=label,
                section_type=section_type,
                source_locator=locator,
                source_url=source_url,
                content=content,
                metadata=metadata,
                element=tag if section_type == "table" else None,
            )
        )

    if sections:
        return tuple(sections)

    body = soup.body or soup
    content = "\n".join(
        normalized_line
        for line in body.get_text("\n", strip=True).splitlines()
        if (normalized_line := _normalize_text(line))
    )
    if not content:
        return ()

    return (
        SourceSection(
            section_id="document-body",
            section_label="Document Body",
            section_type="body",
            source_locator="body",
            source_url=source_url,
            content=content,
            metadata={"dom_path": "body"},
        ),
    )
