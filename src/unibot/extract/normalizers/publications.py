from __future__ import annotations

import re

# Double quotes: standard "..." and curly \u201c...\u201d
_DOUBLE_QUOTED_PATTERN = re.compile(r'["\u201c](?P<title>.+?)["\u201d]')
# Single quotes: '...' and curly \u2018...\u2019
_SINGLE_QUOTED_PATTERN = re.compile(r"['\u2018](?P<title>.+?)['\u2019]")
# Cross-quote: any opening quote matched with any closing quote
_CROSS_QUOTED_PATTERN = re.compile(
    r"['\"\u2018\u201c](?P<title>.+?)['\"\u2019\u201d]"
)
_YEAR_PATTERN = re.compile(r"\b(20\d{2})\b")
_WHITESPACE_PATTERN = re.compile(r"\s+")
_NUMBERED_PREFIX = re.compile(r"^(?:\[\d+\]|\d+[.)]\s)\s*")
_HEADING_PREFIXES = ("selected publications", "journal papers", "conference papers")
_PERSON_TOKEN_PATTERN = re.compile(r"(?:[A-Z][A-Za-z.'-]*|[A-Z]\.)$")
# APA: Author(s) (Year). Title. Venue  OR  Author(s), (Year), Title. Venue
_APA_YEAR_BOUNDARY = re.compile(
    r"[,\s]\s*\((\d{4})\)[.,]\s*"
)
_AUTHOR_YEAR_ARTIFACT = re.compile(r"\s*\(\d{4}\)\s*\.?\s*$")
_YEAR_ONLY_LINE = re.compile(r"^\s*(20\d{2})\s*[.,;]?\s*$")
_TRAILING_METRICS_SUFFIX = re.compile(
    r"\s*\((?=[^)]*\b(?:snip|sjr|impact factor|percentile in scopus|ajg|jcr)\b)[^)]*\)\s*$",
    re.IGNORECASE,
)
_VOLUME_COMPONENT_PATTERN = re.compile(
    r"^(?:vol\.?\s*\d+(?:\.\d+)?|no\.?\s*\d+(?:\.\d+)?|\d+(?:\.\d+)?)$",
    re.IGNORECASE,
)


def _clean_author_list(authors: list) -> list[str]:
    """Remove year artifacts that leaked into author name segments."""
    cleaned = []
    for a in authors:
        if not isinstance(a, str):
            cleaned.append(a)
            continue
        a = _AUTHOR_YEAR_ARTIFACT.sub("", a).strip().rstrip(".,;")
        if a:
            cleaned.append(a)
    return cleaned


def _normalize_inline_spacing(text: str) -> str:
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r"\s+\.", ".", text)
    return _WHITESPACE_PATTERN.sub(" ", text).strip()


def _strip_trailing_metrics(text: str) -> str:
    return _TRAILING_METRICS_SUFFIX.sub("", text).strip().rstrip(".,;")


def _clean_embedded_title_quotes(text: str) -> str:
    return re.sub(r"['\u2018\u2019\u201c\u201d]([^'\u2018\u2019\u201c\u201d]{1,80})['\u2018\u2019\u201c\u201d]", r"\1", text)


def _split_structured_lines(citation: str) -> list[str]:
    lines: list[str] = []
    for raw_line in citation.splitlines():
        line = _normalize_inline_spacing(raw_line)
        if not line:
            continue
        if (
            lines
            and line[:1] in {",", ";", ":"}
            and _YEAR_ONLY_LINE.fullmatch(lines[-1]) is None
        ):
            lines[-1] = _normalize_inline_spacing(f"{lines[-1]} {line}")
            continue
        lines.append(line)
    return lines


def _split_authors_text(text: str) -> list[str]:
    normalized = _normalize_inline_spacing(text).rstrip(".")
    normalized = re.sub(r"\s+\band\b\s+", ", ", normalized, flags=re.IGNORECASE)
    return [part.strip() for part in normalized.split(",") if part.strip()]


def _should_use_quoted_match(normalized: str, title_match: re.Match) -> bool:
    prefix = normalized[: title_match.start()].rstrip(" ,;")
    if not prefix:
        return True
    # Strip numbered prefix (e.g. "[1]") before evaluating the author segment
    prefix = _NUMBERED_PREFIX.sub("", prefix).strip(" ,;")
    if not prefix:
        return True
    last_segment = prefix.rsplit(",", maxsplit=1)[-1].strip()
    if not last_segment:
        return True
    author_candidate = re.sub(r"^and\s+", "", last_segment, flags=re.IGNORECASE)
    if _looks_like_author_segment(author_candidate):
        return True
    if re.search(r"\band\b", author_candidate, re.IGNORECASE):
        and_parts = re.split(r"\s+and\s+", author_candidate, flags=re.IGNORECASE)
        if all(_looks_like_author_segment(p.strip()) for p in and_parts if p.strip()):
            return True
    if re.search(r"\(\d{4}\)\s*$", last_segment):
        return True
    if last_segment.startswith("(") and last_segment.endswith(")"):
        return True
    # Recognize "et al." as a valid author-adjacent pattern
    if re.search(r"\bet\s+al\b\.?", last_segment, re.IGNORECASE):
        return True
    if any(char.islower() for char in last_segment):
        return False
    return True


_VENUE_KEYWORDS = (
    "journal",
    "conference",
    "proceedings",
    "transactions",
    "letters",
    "symposium",
    "workshop",
    "review",
    "magazine",
    "ieee",
    "acm",
)

_VENUE_BOUNDARY_WORDS = frozenset({
    "accepted", "published", "submitted", "appeared", "presented",
    "in press", "to appear",
})
_PUBLISHER_WORDS = (
    "press",
    "springer",
    "elsevier",
    "wiley",
    "taylor",
    "oxford",
    "cambridge",
)


def _split_suffix_authors_venue(text: str) -> tuple[list[str], str | None]:
    """Split 'Author1, Author2, ..., Venue info' into authors and venue."""
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        return [], None

    authors: list[str] = []
    venue_start = None
    for i, part in enumerate(parts):
        lowered = part.casefold()
        if any(word in lowered for word in _VENUE_BOUNDARY_WORDS):
            venue_start = i
            break
        if any(kw in lowered for kw in _VENUE_KEYWORDS):
            venue_start = i
            break
        if lowered.startswith("and ") and not _looks_like_author_segment(
            part[4:].strip()
        ):
            venue_start = i
            break
        clean = part.strip()
        if clean.casefold().startswith("and "):
            clean = clean[4:].strip()
        if _looks_like_author_segment(clean):
            authors.append(clean)
        elif authors:
            venue_start = i
            break
        else:
            venue_start = i
            break

    venue = None
    if venue_start is not None:
        venue_text = ", ".join(parts[venue_start:]).strip(" ,.;")
        venue_text = re.sub(
            r"^(?:accepted|published|submitted|appeared|presented)\s+(?:for\s+publication\s+)?in\s+",
            "", venue_text, flags=re.IGNORECASE,
        ).strip()
        if venue_text:
            venue = venue_text

    return authors, venue


def _looks_like_venue_or_meta(text: str) -> bool:
    lowered = text.casefold()
    return (
        _looks_like_venue(text)
        or any(word in lowered for word in _PUBLISHER_WORDS)
        or any(word in lowered for word in _VENUE_BOUNDARY_WORDS)
        or "vol." in lowered
        or lowered.startswith("vol ")
        or "pp." in lowered
        or _VOLUME_COMPONENT_PATTERN.fullmatch(text.strip()) is not None
    )


def _strip_heading_prefix(text: str) -> str:
    lowered = text.lower()
    for prefix in _HEADING_PREFIXES:
        if lowered.startswith(prefix):
            return text[len(prefix):].strip(" :-")
    return text


def _try_structured_lines(
    lines: list[str],
    payload: dict[str, object],
) -> dict[str, object] | None:
    if len(lines) < 3:
        return None

    year_index = next(
        (index for index, line in enumerate(lines) if _YEAR_ONLY_LINE.fullmatch(line)),
        None,
    )
    if year_index is None or year_index < 2:
        return None

    title_index = year_index - 2
    venue_index = year_index - 1
    title = lines[title_index].rstrip(".,;")
    venue = lines[venue_index].rstrip(".,;")
    authors_text = " ".join(lines[:title_index]).strip()
    if not title or not venue or not authors_text:
        return None
    if _looks_like_author_segment(title) or _looks_like_author_segment(venue):
        return None

    payload["authors"] = _split_authors_text(authors_text)
    payload["title"] = title
    payload["venue"] = venue
    year_match = _YEAR_ONLY_LINE.fullmatch(lines[year_index])
    assert year_match is not None  # guaranteed by the search above
    payload["year"] = int(year_match.group(1))
    return payload


def _try_author_prefix_title_venue_year(
    normalized: str,
    payload: dict[str, object],
) -> dict[str, object] | None:
    year_match = None
    for candidate in _YEAR_PATTERN.finditer(normalized):
        year_match = candidate
    if year_match is None:
        return None

    prefix = _strip_trailing_metrics(normalized[: year_match.start()].rstrip(" ,("))
    parts = [part.strip(" ,.;") for part in prefix.split(",") if part.strip(" ,.;")]
    if len(parts) < 3:
        return None

    author_parts: list[str] = []
    index = 0
    while index < len(parts) and _looks_like_author_segment(parts[index]):
        author_parts.append(parts[index])
        index += 1

    remaining = parts[index:]
    if len(author_parts) < 2 or len(remaining) < 2:
        return None

    title, venue = _split_title_first_title_and_venue(remaining)
    if not title or not venue:
        return None

    title = _clean_embedded_title_quotes(title)

    payload["authors"] = author_parts
    payload["title"] = title
    payload["venue"] = venue
    payload["year"] = int(year_match.group(1))
    return payload


def _split_title_first_title_and_venue(parts: list[str]) -> tuple[str | None, str | None]:
    if len(parts) < 2:
        return None, None
    split_index = len(parts) - 1
    venue_parts = [parts[-1]]
    while split_index > 0:
        candidate = parts[split_index - 1]
        if _looks_like_venue_or_meta(candidate):
            venue_parts.insert(0, candidate)
            split_index -= 1
            continue
        if _VOLUME_COMPONENT_PATTERN.fullmatch(venue_parts[0].strip()) is not None:
            venue_parts.insert(0, candidate)
            split_index -= 1
            continue
        break
    title_parts = parts[:split_index]

    if not title_parts or not venue_parts:
        return None, None

    title = ", ".join(title_parts).strip(" ,.;")
    venue = ", ".join(venue_parts).strip(" ,.;")
    return title or None, venue or None


def _try_title_first_venue_year(
    normalized: str,
    payload: dict[str, object],
) -> dict[str, object] | None:
    year_match = None
    for candidate in _YEAR_PATTERN.finditer(normalized):
        year_match = candidate
    if year_match is None:
        return None

    prefix = _strip_trailing_metrics(normalized[: year_match.start()].rstrip(" ,("))
    if re.search(r"\.\s+[A-Z]", prefix):
        return None
    parts = [part.strip(" ,.;") for part in prefix.split(",") if part.strip(" ,.;")]
    if len(parts) < 2 or _looks_like_author_segment(parts[0]):
        return None

    title, venue = _split_title_first_title_and_venue(parts)
    if not title or not venue:
        return None

    payload["title"] = _clean_embedded_title_quotes(title)
    payload["venue"] = venue
    payload["year"] = int(year_match.group(1))
    return payload


def parse_publication_citation(citation: str) -> dict[str, object]:
    result = _parse_publication_citation_inner(citation)
    authors_val = result.get("authors")
    if isinstance(authors_val, list):
        result["authors"] = _clean_author_list(authors_val)
    # Build citation_text from parsed components, fall back to raw_citation
    parts: list[str] = []
    authors_val = result.get("authors")
    if isinstance(authors_val, list):
        parts.append(", ".join(str(a) for a in authors_val))
    if result.get("year"):
        parts.append(f"({result['year']})")
    if result.get("title"):
        parts.append(str(result["title"]))
    if result.get("venue"):
        parts.append(str(result["venue"]))
    result["citation_text"] = ". ".join(parts) if parts else result.get("raw_citation")
    return result


def _parse_publication_citation_inner(citation: str) -> dict[str, object]:
    structured_lines = _split_structured_lines(citation)
    normalized = _normalize_inline_spacing(citation)
    normalized = _strip_heading_prefix(normalized)
    payload: dict[str, object] = {
        "raw_citation": normalized,
        "title": None,
        "authors": [],
        "venue": None,
        "year": None,
    }

    structured_result = _try_structured_lines(structured_lines, payload.copy())
    if structured_result is not None:
        structured_result["raw_citation"] = normalized
        return structured_result

    # Try double-quoted title first (most common academic format)
    title_match = _DOUBLE_QUOTED_PATTERN.search(normalized)
    if title_match is not None and _should_use_quoted_match(normalized, title_match):
        return _extract_from_quoted_match(title_match, normalized, payload)

    # Try single-quoted title (social science / humanities style)
    title_match = _SINGLE_QUOTED_PATTERN.search(normalized)
    if title_match is not None and _should_use_quoted_match(normalized, title_match):
        return _extract_from_quoted_match(title_match, normalized, payload)

    # Cross-quote fallback: mismatched opening/closing quote characters
    title_match = _CROSS_QUOTED_PATTERN.search(normalized)
    if (
        title_match is not None
        and len(title_match.group("title")) > 15
        and _should_use_quoted_match(normalized, title_match)
    ):
        return _extract_from_quoted_match(title_match, normalized, payload)

    # APA format: Author(s) (Year). Title. Venue, ...
    apa_result = _try_apa_format(normalized, payload)
    if apa_result is not None:
        return apa_result

    # Strip numbered prefix for fallback branches too
    normalized = _NUMBERED_PREFIX.sub("", normalized).strip()
    payload["raw_citation"] = _strip_heading_prefix(
        _normalize_inline_spacing(citation)
    )

    prefix_author_result = _try_author_prefix_title_venue_year(normalized, payload.copy())
    if prefix_author_result is not None:
        prefix_author_result["raw_citation"] = payload["raw_citation"]
        return prefix_author_result

    # Branch: Author . Title. Venue , Year.
    dot_split = re.split(r"\s*\.\s+", normalized)
    if len(dot_split) >= 3:
        result = _try_author_dot_title_dot_venue(dot_split, payload)
        if result is not None:
            return result

    # Branch: Title, Venue (Year) [optional suffix authors]
    paren_year = re.search(r"\((\d{4})\)\s*", normalized)
    if paren_year is not None:
        result = _try_title_comma_venue_paren_year(normalized, paren_year, payload)
        if result is not None:
            return result

    # Generic fallback: try to extract year, then infer title from text before venue
    year_match = None
    for candidate in _YEAR_PATTERN.finditer(normalized):
        year_match = candidate
    if year_match is not None:
        payload["year"] = int(year_match.group(1))
        collapsed_result = _try_collapsed_title_authors_venue(normalized, year_match, payload)
        if collapsed_result is not None:
            return collapsed_result
        # Heuristic: text before the last comma preceding the year is likely the title
        prefix = normalized[: year_match.start()].rstrip(" ,(")
        last_comma = prefix.rfind(",")
        if last_comma > 20:
            candidate_title = prefix[:last_comma].strip(" ,;")
            # Strip author prefix: anything before the first comma-separated
            # segment that looks like a title (longer than 30 chars after authors)
            # Simple heuristic: if there's a (year) pattern early, authors precede it
            author_year = re.search(r"\(\d{4}\)[,.]?\s*", candidate_title)
            if author_year:
                candidate_title = candidate_title[author_year.end():].strip(" ,;")
            if len(candidate_title) > 15:
                payload["title"] = candidate_title
        elif last_comma < 0 and len(prefix) > 20:
            # No comma — look for "in Venue" boundary
            # Handles "Title in VenueAcronym Year: FullVenueName"
            in_pos = prefix.rfind(" in ")
            if in_pos > 10:
                payload["title"] = prefix[:in_pos].strip()
                payload["venue"] = prefix[in_pos + 4:].strip()

    title_first_result = _try_title_first_venue_year(normalized, payload.copy())
    if title_first_result is not None:
        title_first_result["raw_citation"] = payload["raw_citation"]
        return title_first_result

    return payload


def _try_collapsed_title_authors_venue(
    normalized: str,
    year_match: re.Match,
    payload: dict[str, object],
) -> dict[str, object] | None:
    prefix = normalized[: year_match.start()].rstrip(" ,(")
    title_and_authors, venue = _split_venue_suffix(prefix)
    if venue is None:
        return None

    title, authors = _split_title_and_authors(title_and_authors)
    if not title:
        return None

    payload["title"] = title
    payload["venue"] = venue
    if authors:
        payload["authors"] = authors
    return payload


def _split_venue_suffix(text: str) -> tuple[str, str | None]:
    lowered = text.lower()
    start = -1
    for keyword in _VENUE_KEYWORDS:
        keyword_pos = lowered.rfind(keyword)
        if keyword_pos > start:
            start = keyword_pos
    if start > 0:
        return text[:start].rstrip(" ,"), text[start:].strip(" ,")

    parts = [part.strip() for part in text.split(",") if part.strip()]
    if len(parts) >= 2 and _looks_like_venue(parts[-1]):
        return ", ".join(parts[:-1]), parts[-1]

    return text, None


def _split_title_and_authors(text: str) -> tuple[str | None, list[str]]:
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if not parts:
        return None, []

    authors: list[str] = []
    while len(parts) > 1 and _looks_like_author_segment(parts[-1]):
        authors.insert(0, parts.pop())

    if authors and parts:
        trailing_author, remaining = _split_trailing_author_segment(
            parts[-1],
            expected_size=len(authors[0].split()),
        )
        if trailing_author is not None:
            authors.insert(0, trailing_author)
            if remaining:
                parts[-1] = remaining
            else:
                parts.pop()

    title = ", ".join(parts).strip(" ,;")
    return (title or None), authors


def _split_trailing_author_segment(
    text: str,
    *,
    expected_size: int | None = None,
) -> tuple[str | None, str]:
    words = [word for word in text.split() if word]
    sizes = (expected_size,) if expected_size is not None else (3, 2)
    for size in sizes:
        if size is None:
            continue
        if len(words) <= size:
            continue
        candidate = " ".join(words[-size:])
        if not _looks_like_author_segment(candidate):
            continue
        remaining = " ".join(words[:-size]).strip(" ,;")
        if len(remaining.split()) < 2:
            continue
        if remaining:
            return candidate, remaining
    return None, text


def _looks_like_author_segment(text: str) -> bool:
    words = [word for word in text.split() if word]
    if not (1 < len(words) <= 4):
        return False
    return all(_PERSON_TOKEN_PATTERN.fullmatch(word) for word in words)


def _looks_like_venue(text: str) -> bool:
    lowered = text.casefold()
    return any(keyword in lowered for keyword in _VENUE_KEYWORDS)


def _try_title_comma_venue_paren_year(
    normalized: str, paren_year: re.Match, payload: dict[str, object]
) -> dict[str, object] | None:
    """Parse: Title, Venue (Year)"""
    year = int(paren_year.group(1))
    prefix = normalized[: paren_year.start()].rstrip(" ,")
    last_comma = prefix.rfind(",")
    if last_comma < 5:
        return None
    title = prefix[:last_comma].strip()
    venue = prefix[last_comma + 1 :].strip()
    if not title or not venue:
        return None
    payload["title"] = title
    payload["venue"] = venue
    payload["year"] = year
    suffix = normalized[paren_year.end():].strip(" ,.;")
    if suffix:
        suffix_authors, _ = _split_suffix_authors_venue(suffix)
        if suffix_authors:
            payload["authors"] = suffix_authors
    return payload


def _try_apa_format(
    normalized: str, payload: dict[str, object]
) -> dict[str, object] | None:
    """Parse APA-style: Author(s) (Year). Title. Venue, ... or Author(s), (Year), Title. Venue."""
    apa_match = _APA_YEAR_BOUNDARY.search(normalized)
    if apa_match is None:
        return None
    year = int(apa_match.group(1))
    remainder = normalized[apa_match.end():].strip()
    if not remainder:
        return None
    # Split remainder by sentence-ending periods to separate title from venue.
    # Pattern: Title. Venue, Vol(Issue), Pages.
    # Find the first period followed by a space and uppercase letter (sentence boundary)
    sentence_split = re.split(r"\.\s+(?=[A-Z])", remainder, maxsplit=1)
    title = sentence_split[0].strip().rstrip(".")
    if not title or len(title) < 10:
        return None
    payload["year"] = year
    payload["title"] = title
    if len(sentence_split) > 1:
        venue = sentence_split[1].strip().rstrip(".")
        if venue:
            payload["venue"] = venue
    # Authors are everything before the year boundary
    author_text = normalized[: apa_match.start()].strip().rstrip(",")
    if author_text:
        payload["authors"] = [a.strip() for a in author_text.split(",") if a.strip()]
    return payload


def _try_author_dot_title_dot_venue(
    dot_split: list[str], payload: dict[str, object]
) -> dict[str, object] | None:
    """Parse: Author(s) . Title. Venue , Year."""
    # Last segment may contain "Venue , Year." or just "Year"
    last = dot_split[-1].strip().rstrip(".")
    year_match = _YEAR_PATTERN.search(last)
    if year_match is None:
        return None
    year = int(year_match.group(1))
    venue_part = last[: year_match.start()].strip().rstrip(" ,")
    venue: str | None
    # If venue is in the last segment, title is the second-to-last
    if venue_part:
        title = dot_split[-2].strip().rstrip(".")
        venue = venue_part
    elif len(dot_split) >= 3:
        title = dot_split[-3].strip().rstrip(".") if len(dot_split) >= 4 else dot_split[-2].strip().rstrip(".")
        venue = dot_split[-2].strip().rstrip(".") if len(dot_split) >= 4 else None
    else:
        return None
    if not title or len(title) < 10:
        return None
    payload["title"] = title
    if venue:
        payload["venue"] = venue
    payload["year"] = year
    # Authors are everything before the title segment
    title_index = dot_split.index(title + ".") if (title + ".") in dot_split else None
    if title_index is None:
        # Find the segment that matches the title
        for i, seg in enumerate(dot_split):
            if seg.strip().rstrip(".") == title:
                title_index = i
                break
    if title_index is not None and title_index > 0:
        author_text = ". ".join(dot_split[:title_index]).strip().rstrip(".")
        if author_text:
            payload["authors"] = [
                a.strip() for a in author_text.split(",") if a.strip()
            ]
    return payload


def _extract_from_quoted_match(
    title_match: re.Match,
    normalized: str,
    payload: dict[str, object],
) -> dict[str, object]:
    payload["title"] = title_match.group("title").strip().rstrip(",")
    authors_text = normalized[: title_match.start()].strip(" ,;")
    # Strip numbered prefix like [1]
    authors_text = _NUMBERED_PREFIX.sub("", authors_text).strip(" ,;")
    if authors_text:
        payload["authors"] = _split_authors_text(authors_text)

    suffix = normalized[title_match.end() :].strip(" ,.;")
    year_match = None
    for candidate in _YEAR_PATTERN.finditer(suffix):
        year_match = candidate
    if year_match is not None:
        payload["year"] = int(year_match.group(1))
        venue = suffix[: year_match.start()].strip(" ,.;")
        if venue:
            payload["venue"] = venue
    elif suffix:
        payload["venue"] = suffix

    if not payload["authors"]:
        suffix_text = normalized[title_match.end():].strip(" ,.;")
        year_in_suffix = None
        for candidate in _YEAR_PATTERN.finditer(suffix_text):
            year_in_suffix = candidate
        if year_in_suffix is not None:
            payload["year"] = int(year_in_suffix.group(1))
            suffix_text = suffix_text[:year_in_suffix.start()].rstrip(" ,.;")

        suffix_authors, suffix_venue = _split_suffix_authors_venue(suffix_text)
        if suffix_authors:
            payload["authors"] = suffix_authors
        if suffix_venue:
            payload["venue"] = suffix_venue

    # If no year found in suffix, check the authors prefix (e.g. "Author (2020), 'Title'")
    if payload["year"] is None:
        authors_text = normalized[: title_match.start()]
        year_in_prefix = None
        for candidate in _YEAR_PATTERN.finditer(authors_text):
            year_in_prefix = candidate
        if year_in_prefix is not None:
            payload["year"] = int(year_in_prefix.group(1))

    return payload
