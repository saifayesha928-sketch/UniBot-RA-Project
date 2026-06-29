"""Ingest a partner data bundle (``sources.json`` + ``records.jsonl``) into the
canonical store, so the records become eligible for serving-generation builds.

This is the entry point a partner runs *instead of* the crawl/extract pipeline:
they have already collected and extracted their content, so this command loads
their delivered artifacts straight into the database.

Pipeline position::

    data/sources.json + data/records.jsonl
        -> ingest_records       (this script, writes canonical records)
        -> build_serving_generation
        -> API

Usage::

    python -m scripts.ingest_records \
        --sources data/sources.json \
        --records data/records.jsonl

    # Parse + validate only, no database writes:
    python -m scripts.ingest_records --records data/records.jsonl --dry-run

Record format
-------------
``--records`` accepts any of the following (the loader is deliberately
tolerant, since partner exports vary):

* JSON Lines, one ``ExtractedRecord`` object per line (the documented format),
* a single pretty-printed JSON array of records,
* several concatenated JSON arrays/objects, optionally separated by ``//`` or
  ``/* ... */`` comments.

Each record must contain the required ``ExtractedRecord`` fields; non-conforming
records are reported and skipped (or cause a non-zero exit under ``--strict``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REQUIRED_RECORD_FIELDS = (
    "record_id",
    "record_type",
    "source_url",
    "source_section_id",
    "source_section_label",
    "source_locator",
    "source_authority_tier",
    "conflict_scope_id",
    "dedupe_key",
)

_LINE_COMMENT = re.compile(r"^\s*//.*$", re.MULTILINE)
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)


def _strip_comments(text: str) -> str:
    text = _BLOCK_COMMENT.sub("", text)
    text = _LINE_COMMENT.sub("", text)
    return text


def load_json_records(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Load records from a file in JSONL / array / concatenated-JSON form.

    Returns ``(records, warnings)``. Never raises on malformed regions. It
    records a warning and resumes at the next decodable value.
    """
    raw = path.read_text(encoding="utf-8")
    warnings: list[str] = []

    # Fast path: strict JSON Lines (one object per non-empty line).
    line_objs: list[dict[str, Any]] = []
    jsonl_ok = True
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            jsonl_ok = False
            break
        if isinstance(obj, dict):
            line_objs.append(obj)
        else:
            jsonl_ok = False
            break
    if jsonl_ok and line_objs:
        return line_objs, warnings

    # Tolerant path: strip comments, stream-decode every top-level JSON value,
    # flatten arrays, and skip past any malformed region.
    text = _strip_comments(raw)
    decoder = json.JSONDecoder()
    records: list[dict[str, Any]] = []
    i, n = 0, len(text)
    while i < n:
        while i < n and text[i] in " \t\r\n,":
            i += 1
        if i >= n:
            break
        try:
            obj, end = decoder.raw_decode(text, i)
        except json.JSONDecodeError as exc:
            # Skip to the next plausible JSON start and keep going.
            nxt = min(
                (p for p in (text.find("{", i + 1), text.find("[", i + 1)) if p != -1),
                default=-1,
            )
            warnings.append(f"skipped malformed region at offset {i} ({exc.msg})")
            if nxt == -1:
                break
            i = nxt
            continue
        if isinstance(obj, list):
            records.extend(x for x in obj if isinstance(x, dict))
        elif isinstance(obj, dict):
            records.append(obj)
        i = end
    return records, warnings


def validate_record(rec: dict[str, Any]) -> str | None:
    """Return an error string if the record is invalid, else None."""
    missing = [f for f in REQUIRED_RECORD_FIELDS if f not in rec]
    if missing:
        return f"missing fields: {', '.join(missing)}"
    if not isinstance(rec.get("source_authority_tier"), int):
        return "source_authority_tier must be an integer"
    return None


def _canonical_payload_hash(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def build_candidate(rec: dict[str, Any]):  # -> VerificationCandidate
    """Convert one ``ExtractedRecord`` dict into a ``VerificationCandidate``."""
    from unibot.verify.rules import VerificationCandidate
    from unibot.verify.value_identity import value_hash_for_stored_record

    payload = dict(rec.get("record_payload") or {})
    record_type = rec["record_type"]
    value_hash = value_hash_for_stored_record(
        record_type, payload, source_text_hash=_canonical_payload_hash(payload)
    )
    record_version_id = hashlib.sha256(
        f"{rec['record_id']}|{value_hash}".encode("utf-8")
    ).hexdigest()[:32]
    return VerificationCandidate(
        record_id=rec["record_id"],
        record_version_id=record_version_id,
        record_type=record_type,
        conflict_scope_id=rec["conflict_scope_id"],
        dedupe_key=rec["dedupe_key"],
        value_hash=value_hash,
        source_authority_tier=int(rec["source_authority_tier"]),
        source_url=rec["source_url"],
        source_locator=rec.get("source_locator", "body"),
        source_section_id=rec.get("source_section_id"),
        source_section_label=rec.get("source_section_label"),
        cycle_label=rec.get("cycle_label"),
        year_confidence=rec.get("year_confidence", "unknown"),
        record_payload=payload,
    )


def _source_entry_from_dict(d: dict[str, Any]):  # -> SourceRegistryEntry
    from unibot.db.repositories.source_registry import SourceRegistryEntry

    return SourceRegistryEntry(
        source_url=d["source_url"],
        canonical_url=d.get("canonical_url", d["source_url"]),
        source_class=d["source_class"],
        crawl_method=d.get("crawl_method", "html_static"),
        legal_status=d.get("legal_status", "allowed"),
        crawl_status=d.get("crawl_status"),
        default_authority_tier=int(d.get("default_authority_tier", 1)),
        refresh_policy=d.get("refresh_policy", "manual_only"),
        parser_target=d.get("parser_target", "html"),
        parent_source_url=d.get("parent_source_url"),
        link_text=d.get("link_text"),
        is_active=bool(d.get("is_active", True)),
    )


def load_sources(session, path: Path) -> int:
    """Upsert ``sources.json`` entries into the source registry."""
    from unibot.db.repositories.source_registry import SourceRegistryRepository

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("sources", [])
    entries = tuple(_source_entry_from_dict(d) for d in data)
    SourceRegistryRepository(session).upsert_entries(entries)
    session.flush()
    return len(entries)


def ingest_candidates(session, candidates: tuple) -> int:
    """Run candidates through the canonical-upsert verification pipeline."""
    from unibot.pipeline import canonical_upsert as cu
    from unibot.pipeline import source_lifecycle as sl

    cache: dict[str, str | None] = {}

    def source_id_resolver(url: str) -> str | None:
        return sl.source_id_for_url(session, url, cache)

    cu.process_scope_updates(
    session,
    candidates,
    source_id_resolver=source_id_resolver,
    create_verification_event=lambda s, decision: None,
)
    return len(candidates)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ingest_records")
    parser.add_argument("--records", default="data/records.jsonl", type=Path)
    parser.add_argument("--sources", default="data/sources.json", type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and validate only; do not touch the database.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any record fails validation.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)

    records, warnings = load_json_records(args.records)
    print(f"parsed {len(records)} record(s) from {args.records}")
    for w in warnings:
        print(f"  warning: {w}")

    valid: list[dict[str, Any]] = []
    invalid = 0
    for rec in records:
        err = validate_record(rec)
        if err:
            invalid += 1
            if invalid <= 20:
                print(f"  invalid record {rec.get('record_id', '?')!r}: {err}")
        else:
            valid.append(rec)
    print(f"valid: {len(valid)}  invalid: {invalid}")
    if args.strict and invalid:
        print("strict mode: aborting due to invalid records.")
        return 1

    if args.dry_run:
        # Surface what would be ingested without importing DB machinery.
        candidates = tuple(build_candidate(r) for r in valid)
        print(f"dry-run OK: {len(candidates)} candidate(s) built (no DB writes).")
        return 0

    from unibot.db.session import direct_session_scope

    seen = set()
    unique_candidates = []

    for r in valid:
        c = build_candidate(r)
        if c.record_version_id in seen:
            print(f"Skipping duplicate: {c.record_id}")
            continue
        seen.add(c.record_version_id)
        unique_candidates.append(c)

    candidates = tuple(unique_candidates)

    with direct_session_scope() as session:
        if args.sources.exists():
            n_sources = load_sources(session, args.sources)
            print(f"loaded {n_sources} source(s) from {args.sources}")
        else:
            print(f"no sources file at {args.sources}; skipping source registry load")

        ingest_candidates(session, candidates)
        session.commit()

    print(f"ingested {len(candidates)} record(s) into canonical store.")
    print("next: python -m scripts.build_serving_generation --generation-label <label>")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())