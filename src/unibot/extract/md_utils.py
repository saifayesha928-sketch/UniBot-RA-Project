"""Shared markdown parsing utilities for the markdown-first extraction pipeline.

All markdown parsers depend on this module for boilerplate removal, section
splitting, pipe-table parsing, and link extraction.
"""
from __future__ import annotations

import re
from urllib.parse import urlsplit

from unibot.extract.text import slugify

_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_NOISE_LINES = re.compile(
    r"^(?:"
    r"[\uffe9\uffeb\u00d7x\s]*$"              # modal artifacts: ￩ ￫ × x
    r"|Previous\s+Next\s*$"                     # carousel nav
    r"|\[Page load link\].*$"                   # footer link
    r"|\[Go to Top\].*$"                        # scroll link
    r"|\[\s*Go to Top\s*\].*$"                  # variant
    r"|!\[\]\([^)]*Logo[^)]*\).*$"              # logo image
    r"|\[Skip to content\].*$"                  # skip nav link
    r")",
    re.MULTILINE | re.IGNORECASE,
)
_BOLD_MARKER = re.compile(r"\*\*([^*]+)\*\*")
ITALIC_MARKER = re.compile(r"(?<!\w)_([^_]+)_(?!\w)")
NAV_ONLY_LIST_ITEM = re.compile(r"^\[.+\]\(.+\)$")
_PIPE_TABLE_SEPARATOR = re.compile(r"^\s*\|[\s\-:|]+\|\s*$")
_MD_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_FOOTER_BOUNDARY = re.compile(r"^#{5}\s+Students\s*$", re.MULTILINE)
_FIRST_H1 = re.compile(r"^#\s+.+$", re.MULTILINE)

# WordPress author byline: "BS Computer Science[siteadmin](author_url)2026-03-26T08:05:31+00:00"
# Anchored to position 0 of section body (NO re.MULTILINE — intentional).
_WP_BYLINE_RE = re.compile(
    r"^[^\n]*"                            # heading text repeat (any chars on first line)
    r"(?:By\s+)?"                         # optional "By " prefix
    r"\[[^\]]+\]"                         # [author_name]
    r"\(https?://[^)]+/author/[^)]+\)"   # (author_profile_url) — also matches title attrs
    r"[|]?"                               # optional pipe separator
    r"\d{4}-\d{2}-\d{2}T[\d:+]+\s*"     # ISO 8601 timestamp
    r"\n?",
)
# Banner image: [![](thumbnail.jpg)](video_url) — matches at position 0 only.
_BANNER_IMG_RE = re.compile(
    r"^\s*"
    r"\[?!\[[^\]]*\]\([^)]+\)\]?"        # image markdown, optionally link-wrapped
    r"(?:\([^)]+\))?"                     # optional link target
    r"\s*\n?",
)


def strip_boilerplate(raw_markdown: str) -> str:
    """Remove site-wide header/footer boilerplate from raw Crawl4AI markdown.

    Strategy:
    1. Find the first ``# `` heading — everything before it is header boilerplate.
    2. Find ``##### Students`` — everything from it onward is footer boilerplate.
    3. Remove remaining noise lines (modal artifacts, carousel nav, etc.).
    """
    text = raw_markdown

    # Cut header: everything before the first H1 heading
    h1_match = _FIRST_H1.search(text)
    if h1_match is not None:
        text = text[h1_match.start():]

    # Cut footer: everything from ##### Students onward
    footer_match = _FOOTER_BOUNDARY.search(text)
    if footer_match is not None:
        text = text[:footer_match.start()]

    # Remove noise lines
    text = _NOISE_LINES.sub("", text)

    # Collapse runs of 3+ blank lines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def strip_wp_artifacts(body: str) -> str:
    """Remove WP author byline and leading banner images from a section body.

    Call on section bodies returned by split_sections(), where the byline
    is at position 0 (the heading has been split off).

    Safe to call on bodies that don't contain these artifacts — returns unchanged.
    """
    body = _WP_BYLINE_RE.sub("", body, count=1)
    # ^ without MULTILINE only matches position 0, so sub() replaces the
    # leading banner each iteration until the body no longer starts with one.
    prev = None
    while prev != body and _BANNER_IMG_RE.match(body):
        prev = body
        body = _BANNER_IMG_RE.sub("", body, count=1)
    return body.lstrip()


_MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")


def strip_md_images(text: str) -> str:
    """Remove markdown image tags ![alt](url) from text."""
    return _MD_IMAGE_RE.sub("", text).strip()


def split_sections(
    markdown: str,
) -> list[tuple[str, int, str]]:
    """Split markdown into sections by heading boundaries.

    Returns a list of ``(heading_text, heading_level, body_content)`` tuples.
    Content before the first heading is returned with heading_text="" and level=0.
    """
    sections: list[tuple[str, int, str]] = []
    matches = list(_HEADING_PATTERN.finditer(markdown))

    if not matches:
        content = markdown.strip()
        if content:
            sections.append(("", 0, content))
        return sections

    # Content before the first heading
    pre_content = markdown[:matches[0].start()].strip()
    if pre_content:
        sections.append(("", 0, pre_content))

    for i, match in enumerate(matches):
        level = len(match.group(1))
        heading_text = match.group(2).strip()
        # Strip anchor links from heading text: "#### [Tab Name](#anchor)" → "Tab Name"
        anchor_match = re.match(r"\[([^\]]+)\]\([^)]*\)", heading_text)
        if anchor_match:
            heading_text = anchor_match.group(1)
        # Strip bold from heading text
        heading_text = _BOLD_MARKER.sub(r"\1", heading_text).strip()

        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        body = markdown[start:end].strip()
        sections.append((heading_text, level, body))

    return sections


def find_sections_by_heading(
    sections: list[tuple[str, int, str]],
    *keywords: str,
) -> list[tuple[str, int, str]]:
    """Find sections whose heading text contains any of the given keywords (case-insensitive)."""
    lowered_keywords = [kw.casefold() for kw in keywords]
    return [
        (heading, level, body)
        for heading, level, body in sections
        if any(kw in heading.casefold() for kw in lowered_keywords)
    ]


def collect_section_with_children(
    sections: list[tuple[str, int, str]],
    index: int,
) -> str:
    """Collect the body of a section plus all immediate child section bodies.

    When a section has an empty body because a sub-heading immediately
    follows, this collects the child sections' bodies up to the next
    same-or-lower level heading.

    Args:
        sections: list of (heading, level, body) tuples from split_sections
        index: index into sections; must satisfy 0 <= index < len(sections)
    """
    if not (0 <= index < len(sections)):
        return ""
    _heading, level, body = sections[index]
    # Level-0 sections (content before first heading) have no children
    if level == 0:
        return body
    parts = [body] if body.strip() else []
    for j in range(index + 1, len(sections)):
        _child_heading, child_level, child_body = sections[j]
        if child_level <= level:
            break
        if child_body.strip():
            parts.append(child_body.strip())
    return "\n\n".join(parts)


_MAX_MULTILINE_JOIN_LOOKAHEAD = 5


def _join_multiline_pipe_cells(text: str) -> str:
    """Pre-process markdown to join multi-line pipe table cells.

    A multi-line cell occurs when a line starts with ``|`` but does not
    end with ``|`` — the cell content wraps to the next line(s) until a
    line ending with ``|`` closes the row.

    Safety: lookahead is capped at ``_MAX_MULTILINE_JOIN_LOOKAHEAD`` lines
    to prevent runaway joins when a ``|`` appears in non-table context.
    """
    lines = text.split("\n")
    result: list[str] = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if (
            stripped.startswith("|")
            and not stripped.endswith("|")
            and not _PIPE_TABLE_SEPARATOR.match(stripped)
        ):
            joined = lines[i].rstrip()
            j = i + 1
            found_closing = False
            while j < len(lines) and (j - i) <= _MAX_MULTILINE_JOIN_LOOKAHEAD:
                next_stripped = lines[j].strip()
                if next_stripped.startswith("|") and next_stripped.endswith("|"):
                    found_closing = False
                    break
                joined = joined + " " + next_stripped
                if next_stripped.endswith("|"):
                    found_closing = True
                    break
                j += 1
            if found_closing:
                result.append(joined)
                i = j + 1
            else:
                result.append(lines[i])
                i += 1
        else:
            result.append(lines[i])
            i += 1
    return "\n".join(result)


def find_tables_in_text(text: str) -> list[tuple[str, str]]:
    """Extract all pipe-delimited table blocks from a markdown text block.

    Returns a list of ``(table_text, trailing_text)`` tuples.  ``trailing_text``
    contains the non-table content between the end of the current table and the
    start of the next table (or end of text), capped at 500 characters.
    """
    text = _join_multiline_pipe_cells(text)
    lines = text.split("\n")
    tables: list[tuple[str, str]] = []
    current_table_lines: list[str] = []
    trailing_lines: list[str] = []
    in_table = False
    _MAX_TRAILING = 500

    def _flush_table() -> None:
        if current_table_lines and len(current_table_lines) >= 2:
            tables.append(("\n".join(current_table_lines), ""))

    def _attach_trailing() -> None:
        if tables and trailing_lines:
            raw = "\n".join(trailing_lines).strip()
            if raw:
                capped = raw[:_MAX_TRAILING]
                tables[-1] = (tables[-1][0], capped)

    for line in lines:
        stripped = line.strip()
        is_pipe_line = stripped.startswith("|") and stripped.endswith("|")
        is_separator = bool(_PIPE_TABLE_SEPARATOR.match(stripped))

        if is_pipe_line or is_separator:
            if not in_table and trailing_lines:
                # New table starting — attach collected trailing to previous table
                _attach_trailing()
                trailing_lines = []
            current_table_lines.append(line)
            in_table = True
        elif in_table:
            # End of table block
            _flush_table()
            current_table_lines = []
            in_table = False
            trailing_lines = [line]
        else:
            trailing_lines.append(line)

    # Flush remaining table
    _flush_table()
    current_table_lines = []
    # Attach any remaining trailing content
    _attach_trailing()

    return tables


def _strip_bold(text: str) -> str:
    """Remove ``**bold**`` markers from text."""
    return _BOLD_MARKER.sub(r"\1", text).strip()


def parse_pipe_table(
    table_text: str,
) -> tuple[list[str], list[dict[str, str]]]:
    """Parse a pipe-delimited markdown table into headers and row dicts.

    Returns ``(headers, rows)`` where headers is a list of column names
    and rows is a list of dicts mapping header→cell value.

    Handles:
    - Bold markers in headers and cells (``**text**`` → ``text``)
    - Leading/trailing whitespace in cells
    - Inconsistent column counts (colspan artifacts) — short rows get empty values
    - Single-cell rows (category headers in scholarship tables)
    """
    lines = [line.strip() for line in table_text.strip().split("\n") if line.strip()]
    if len(lines) < 2:
        return [], []

    # If the first line is a separator (| --- | --- |), the table has no
    # explicit header row.  Drop the separator, infer ``col_0, col_1, …``
    # headers from the data rows, and process every remaining line as data.
    if _PIPE_TABLE_SEPARATOR.match(lines[0]):
        data_lines = [ln for ln in lines[1:] if not _PIPE_TABLE_SEPARATOR.match(ln)]
        if not data_lines:
            return [], []
        max_cols = 0
        for line in data_lines:
            if line.startswith("|") and line.endswith("|"):
                col_count = len(line.split("|")) - 2
            else:
                col_count = len(line.split("|"))
            max_cols = max(max_cols, col_count)
        if max_cols == 0:
            return [], []
        raw_headers = [f"col_{i}" for i in range(max_cols)]
        headerless_rows: list[dict[str, str]] = []
        for line in data_lines:
            raw_parts = line.split("|")
            headerless_cells: list[str] = []
            for part in raw_parts[1:-1] if line.startswith("|") and line.endswith("|") else raw_parts:
                headerless_cells.append(_strip_bold(part.strip()))
            headerless_row: dict[str, str] = {}
            for i, header in enumerate(raw_headers):
                headerless_row[header] = headerless_cells[i] if i < len(headerless_cells) else ""
            headerless_rows.append(headerless_row)
        return raw_headers, headerless_rows

    # Parse header row
    raw_headers = [_strip_bold(cell.strip()) for cell in lines[0].split("|") if cell.strip()]
    if not raw_headers:
        # Headerless table — infer column count from data rows.
        # Use TOTAL cells per row (including empty), not just non-empty,
        # because real headerless tables have misaligned rows where some
        # cells are empty but the column structure must be preserved.
        max_cols = 0
        for line in lines[1:]:
            if _PIPE_TABLE_SEPARATOR.match(line):
                continue
            if line.startswith("|") and line.endswith("|"):
                col_count = len(line.split("|")) - 2
            else:
                col_count = len(line.split("|"))
            max_cols = max(max_cols, col_count)
        if max_cols == 0:
            return [], []
        raw_headers = [f"col_{i}" for i in range(max_cols)]

    rows: list[dict[str, str]] = []
    for line in lines[1:]:
        # Skip separator lines (| --- | --- |)
        if _PIPE_TABLE_SEPARATOR.match(line):
            continue
        # Extract cell values: strip leading/trailing pipe delimiters, then split
        raw_parts = line.split("|")
        cells_cleaned: list[str] = []
        for part in raw_parts[1:-1] if line.startswith("|") and line.endswith("|") else raw_parts:
            cells_cleaned.append(_strip_bold(part.strip()))

        row: dict[str, str] = {}
        for i, header in enumerate(raw_headers):
            row[header] = cells_cleaned[i] if i < len(cells_cleaned) else ""
        rows.append(row)

    return raw_headers, rows


def extract_links(text: str) -> list[tuple[str, str]]:
    """Extract all ``[text](url)`` markdown links from text.

    Returns a list of ``(link_text, url)`` tuples.
    Image links (``![alt](url)``) are excluded.

    Handles Crawl4AI line continuations where link text contains ``\\n``
    (e.g., ``[Block Chain Lab  \\n](url)``).
    """
    # Normalize line continuations inside markdown links: join [text\n](url)
    normalized = re.sub(r"\[([^\]]*?)\s*\n\s*\]", lambda m: "[" + m.group(1).strip() + "]", text)
    results: list[tuple[str, str]] = []
    for match in _MD_LINK_PATTERN.finditer(normalized):
        # Skip image links: preceded by !
        start = match.start()
        if start > 0 and normalized[start - 1] == "!":
            continue
        link_text = match.group(1).strip()
        results.append((link_text, match.group(2)))
    return results


def extract_email_addresses(text: str) -> list[str]:
    """Extract email addresses from plain text (mailto: links are stripped in markdown)."""
    pattern = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b")
    return pattern.findall(text)


def strip_inline_markup(text: str) -> str:
    """Strip ``**bold**`` and ``_italic_`` markdown markers from text."""
    text = _BOLD_MARKER.sub(r"\1", text)
    text = ITALIC_MARKER.sub(r"\1", text)
    return text.strip()


_MD_HEADING_PREFIX = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_MD_LINK_FRAGMENT = re.compile(r"\]\(https?://[^)]*\)")
_MD_ANCHOR_PREFIX = re.compile(r"^\[", re.MULTILINE)


def strip_inline_noise(text: str) -> str:
    """Remove markdown structural markers that interfere with regex matching.

    This is intentionally aggressive — use only on text destined for regex
    extraction, not for display.  Handles crawl4ai quirks:
    - ``**Answer** :`` → ``Answer :``
    - ``#### [Question 1: text](url)`` → ``Question 1: text``

    Stripping order matters:
    1. Heading prefixes (``####``) — exposes leading ``[`` for step 4
    2. Bold/italic markers — delegates to existing ``strip_inline_markup()``
    3. Link URL fragments ``](url)`` — leaves orphan ``[`` at line start
    4. Orphan leading ``[`` at start of line — final cleanup
    """
    text = _MD_HEADING_PREFIX.sub("", text)        # #### text → text
    text = strip_inline_markup(text)               # **bold** → bold, _italic_ → italic
    text = _MD_LINK_FRAGMENT.sub("", text)         # ](url) → ""
    text = _MD_ANCHOR_PREFIX.sub("", text)         # leading [ → ""
    return text


def strip_nav_list_items(body: str) -> str:
    """Remove markdown list items that are purely navigation links."""
    lines = body.split("\n")
    result: list[str] = []
    for line in lines:
        stripped = line.strip()
        item_match = re.match(r"^\s*(?:\*|-|\d+\.)\s+(.+)$", stripped)
        if item_match and NAV_ONLY_LIST_ITEM.match(item_match.group(1).strip()):
            continue
        result.append(line)
    return "\n".join(result)


def page_slug(source_url: str) -> str:
    """Extract the last path segment of a URL as a slug."""
    return slugify(urlsplit(source_url).path.rstrip("/").rsplit("/", maxsplit=1)[-1])
