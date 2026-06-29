from __future__ import annotations

import re

from bs4 import Tag

_SUMMARY_LABEL_RE = re.compile(
    r"(?i)\b(?:grand\s+total|sub[\s-]?total|total(?:\s+degree)?\s+fee|"
    r"net\s+payable|overall\s+total)\b|"
    r"^(?:\**)?\s*total\s*(?:\**)?$",
)


def is_summary_row_label(text: str) -> bool:
    """Detect summary/total row labels by keyword matching."""
    cleaned = text.strip().replace("**", "")
    if not cleaned:
        return False
    return _SUMMARY_LABEL_RE.search(cleaned) is not None

_CYCLE_PATTERN = re.compile(r"\b(20\d{2}(?:\s*[-–]\s*20\d{2})?)\b")

# Matches suspicious truncated amounts like 173,00 (comma followed by exactly two digits
# at end of the number, not part of a longer valid group chain ending in 3 digits)
_SUSPICIOUS_AMOUNT_RE = re.compile(r"\b\d+,\d{1,2}\b(?!,|\d)")
# Well-formed amounts always end with a 3-digit group: 157,000 or 6,73,775
_VALID_TRAILING_RE = re.compile(r",\d{3}\b")
# Narrow repair: exactly comma followed by two digits (truncated trailing zero)
_REPAIRABLE_AMOUNT_RE = re.compile(r"\b(\d{1,3}),(\d{2})\b(?!,|\d)")


def _repair_amount(text: str) -> tuple[str, bool]:
    """Repair a truncated comma-grouped amount if unambiguous.

    Only repairs amounts matching N,DD (one comma followed by exactly two digits)
    where the leading group is 1-3 digits — consistent with a missing trailing zero.
    """
    match = _REPAIRABLE_AMOUNT_RE.fullmatch(text.strip())
    if match:
        return f"{match.group(1)},{match.group(2)}0", True
    return text, False


def _repair_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], int]:
    """Apply narrow amount repair to all cell values. Returns repaired rows and repair count."""
    repaired_rows: list[dict[str, str]] = []
    repair_count = 0
    for row in rows:
        repaired_row: dict[str, str] = {}
        for key, value in row.items():
            repaired_value, was_repaired = _repair_amount(value)
            repaired_row[key] = repaired_value
            if was_repaired:
                repair_count += 1
        repaired_rows.append(repaired_row)
    return repaired_rows, repair_count


_TOTAL_DEGREE_FEE_RE = re.compile(
    r"(?:total|overall).*?(?:fee|cost).*?([\d,]+)", re.IGNORECASE
)
_REPEAT_FEE_RE = re.compile(
    r"(?:repeat|re-?take).*?([\d,]+).*?(?:credit\s*hour)", re.IGNORECASE
)
_ANNUAL_COST_RE = re.compile(
    r"(?:annual|yearly).*?(?:cost|fee).*?([\d,]+)", re.IGNORECASE
)


def _extract_structured_fee_fields(text: str) -> dict[str, str | None]:
    """Extract named scalar fee fields from raw table text including footnotes."""
    total_match = _TOTAL_DEGREE_FEE_RE.search(text)
    repeat_match = _REPEAT_FEE_RE.search(text)
    annual_match = _ANNUAL_COST_RE.search(text)
    return {
        "total_degree_fee": total_match.group(1) if total_match else None,
        "repeat_fee_per_credit_hour": repeat_match.group(1) if repeat_match else None,
        "annual_cost_per_student": annual_match.group(1) if annual_match else None,
    }


def normalize_fee_payload(
    *,
    program_name: str,
    table_label: str,
    raw_table_text: str,
    rows: list[dict[str, str]],
) -> dict[str, object]:
    cycle_label = infer_cycle_label(table_label) or infer_cycle_label(raw_table_text)
    repaired_rows, _ = _repair_rows(rows)
    # Separate summary rows (totals) from data rows
    data_rows = [r for r in repaired_rows if r.get("_row_kind") != "summary"]
    warnings = _detect_parse_warnings(data_rows)
    result: dict[str, object] = {
        "program_name": program_name,
        "currency": infer_currency(raw_table_text),
        "audience": infer_audience(table_label, raw_table_text),
        "table_kind": infer_table_kind(table_label),
        "cycle_label": cycle_label,
        "rows": data_rows,
        "raw_table_text": raw_table_text,
    }
    # Structured scalar fee fields extracted from raw text (including footnotes)
    structured = _extract_structured_fee_fields(raw_table_text)
    for key, value in structured.items():
        if value is not None:
            result[key] = value
    if warnings:
        result["parse_warnings"] = warnings
    return result


def _detect_parse_warnings(rows: list[dict[str, str]]) -> list[str]:
    warnings: list[str] = []
    for row in rows:
        for value in row.values():
            if value and _SUSPICIOUS_AMOUNT_RE.search(value):
                if not _VALID_TRAILING_RE.search(value):
                    warnings.append(f"invalid_amount_format:{value}")
    return warnings


def _flatten_cell_text(cell: Tag) -> str:
    return cell.get_text(" ", strip=True)


def _direct_cells(row: Tag) -> list[Tag]:
    return row.find_all(["th", "td"], recursive=False)


def _looks_like_key_value_table(rows: list[Tag]) -> bool:
    if not rows:
        return False
    data_rows = rows
    # Skip a single-cell title row at the top
    first_cells = _direct_cells(data_rows[0])
    if len(first_cells) == 1:
        data_rows = data_rows[1:]
    # If the first multi-cell row is all <th> and subsequent rows have <td>,
    # it's a matrix header row, not a key-value table
    if (
        len(data_rows) >= 2
        and all(cell.name == "th" for cell in _direct_cells(data_rows[0]))
        and any(cell.name == "td" for cell in _direct_cells(data_rows[1]))
    ):
        return False
    if not data_rows:
        return False
    direct_rows = [_direct_cells(row) for row in data_rows]
    if not all(len(cells) == 2 for cells in direct_rows):
        return False
    amount_like_rows = sum(
        1 for cells in direct_rows if re.search(r"\d", _flatten_cell_text(cells[1]))
    )
    return amount_like_rows >= max(1, len(direct_rows) // 2)


def normalize_fee_table_rows(table: Tag) -> list[dict[str, str]]:
    container = table.find("tbody", recursive=False) or table
    rows = container.find_all("tr", recursive=False)
    if not rows:
        return []

    if _looks_like_key_value_table(rows):
        return _normalize_key_value_rows(rows)

    return _normalize_matrix_rows(rows)


def _normalize_key_value_rows(rows: list[Tag]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for row in rows:
        cells = _direct_cells(row)
        if not cells:
            continue
        # Skip single-cell title/spanning rows
        if len(cells) == 1:
            continue
        label = _flatten_cell_text(cells[0])
        value = _flatten_cell_text(cells[1])
        if label or value:
            normalized.append({"label": label, "value": value})
    return normalized


def _normalize_matrix_rows(rows: list[Tag]) -> list[dict[str, str]]:
    header_cells = _direct_cells(rows[0])
    headers = [_header_key(_flatten_cell_text(cell)) for cell in header_cells]
    if not headers:
        return []

    normalized_rows: list[dict[str, str]] = []
    for row in rows[1:]:
        cells = _direct_cells(row)
        if not cells:
            continue
        entry: dict[str, str] = {}
        for index, header in enumerate(headers):
            if not header or index >= len(cells):
                continue
            entry[header] = _flatten_cell_text(cells[index])
        # Detect summary rows: first column blank OR contains total/summary keywords
        if entry and headers and headers[0] in entry:
            first_val = entry[headers[0]].strip()
            if not first_val:
                remaining_values = [
                    v for k, v in entry.items() if k != headers[0] and v.strip()
                ]
                if remaining_values:
                    entry["_row_kind"] = "summary"
            elif is_summary_row_label(first_val):
                entry["_row_kind"] = "summary"
        normalized_rows.append(entry)
    return normalized_rows


def normalize_fee_table_from_markdown(table_text: str) -> list[dict[str, str]]:
    """Parse a pipe-delimited markdown fee table into normalized row dicts.

    Detects key-value (2-column, label+amount) vs. matrix (multi-column with headers)
    layouts using the same heuristics as the HTML version.
    """
    from unibot.extract.md_utils import parse_pipe_table

    headers, rows = parse_pipe_table(table_text)
    if not rows:
        return []

    # Detect when "headers" are actually data: in key-value tables rendered by
    # Crawl4AI from HTML without <thead>, the first data row becomes the header.
    # Re-inject it as the first data row when the "header" looks like data.
    if len(headers) == 2 and rows:
        h0, h1 = headers
        if re.search(r"\d", h1):
            injected_row = {h0: h0, h1: h1}
            rows = [injected_row, *rows]

    # Normalize header keys to snake_case
    normalized_headers = [_header_key(h) for h in headers]

    # Detect key-value layout: exactly 2 columns, second column has numbers
    if len(headers) == 2:
        amount_count = sum(
            1 for row in rows
            if re.search(r"\d", row.get(headers[1], ""))
        )
        if amount_count >= max(1, len(rows) // 2):
            return [
                {"label": row.get(headers[0], ""), "value": row.get(headers[1], "")}
                for row in rows
                if row.get(headers[0], "").strip() or row.get(headers[1], "").strip()
            ]

    # Matrix layout: use normalized headers as keys
    result: list[dict[str, str]] = []
    for row in rows:
        entry: dict[str, str] = {}
        for raw_header, norm_header in zip(headers, normalized_headers):
            if norm_header:
                entry[norm_header] = row.get(raw_header, "")
        # Detect summary rows: first column blank OR contains total/summary keywords
        if normalized_headers and normalized_headers[0] in entry:
            first_val = entry[normalized_headers[0]].strip()
            if not first_val:
                remaining = [v for k, v in entry.items() if k != normalized_headers[0] and v.strip()]
                if remaining:
                    entry["_row_kind"] = "summary"
            elif is_summary_row_label(first_val):
                entry["_row_kind"] = "summary"
        result.append(entry)
    return result


def normalize_fee_paragraph_rows(raw_table_text: str) -> list[dict[str, str]]:
    rows = [line.strip() for line in raw_table_text.splitlines() if line.strip()]
    return [{"line": line} for line in rows]


def infer_cycle_label(*values: str) -> str | None:
    for value in values:
        match = _CYCLE_PATTERN.search(value)
        if match:
            return match.group(1)
    return None


def infer_table_kind(label: str) -> str:
    lowered = label.casefold()
    if "new intake" in lowered:
        return "new_intake"
    if "continuing" in lowered:
        return "continuing"
    if "international" in lowered:
        return "international"
    return "general"


def infer_audience(*values: str) -> str:
    lowered = " ".join(values).casefold()
    if "international" in lowered or "usd" in lowered:
        return "international"
    return "domestic"


def infer_currency(*values: str) -> str:
    lowered = " ".join(values).casefold()
    has_dollar = "usd" in lowered or "$" in lowered
    has_pkr = "pkr" in lowered or "rs." in lowered or "rs " in lowered
    # Large comma-separated numbers (e.g. 157,000) are PKR-style amounts
    pkr_amounts = len(re.findall(r"\b\d{2,3},\d{3}\b", lowered))
    if has_pkr and has_dollar or (has_dollar and pkr_amounts > 0):
        # Mixed content: count occurrences to determine dominant currency
        usd_amounts = len(re.findall(r"\$\s*\d+|\d+\$|usd\s*\d+", lowered))
        if pkr_amounts > usd_amounts:
            return "PKR"
        return "USD"
    if has_dollar:
        return "USD"
    return "PKR"


def _header_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.strip().casefold()).strip("_")
