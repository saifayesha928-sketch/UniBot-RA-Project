from __future__ import annotations

import structlog
from sqlalchemy import or_
from sqlalchemy.orm import Session

from unibot.db.models import CanonicalRecord, SourceSection as SourceSectionModel, SourceRegistry
from unibot.enrich.apply import fuzzy_program_lookup

logger = structlog.get_logger()


def run_cross_source_enrichment(session: Session) -> None:
    """Fill faculty_label and department_label from org_unit listing pages."""
    # Deferred import to avoid circular dependencies
    from unibot.enrich.faculty_school_map import (
        LISTING_SLUG_TO_LABEL,
        build_faculty_department_map,
        build_faculty_school_map,
        parse_program_department_map,
        parse_program_faculty_map,
    )

    # Load text content from faculty listing org_unit pages
    listing_rows = (
        session.query(
            SourceRegistry.source_url,
            SourceSectionModel.text_content,
        )
        .join(
            SourceSectionModel,
            SourceSectionModel.source_id == SourceRegistry.source_id,
        )
        .filter(SourceRegistry.source_url.like("%/faculty/faculty-of-%"))
        .all()
    )
    if not listing_rows:
        return

    listings: dict[str, str] = {}
    for listing_row in listing_rows:
        url = listing_row.source_url
        for slug, label in LISTING_SLUG_TO_LABEL.items():
            if slug in url:
                existing = listings.get(label, "")
                listings[label] = existing + "\n" + listing_row.text_content
                break

    if not listings:
        return

    # --- Faculty label enrichment ---
    school_map = build_faculty_school_map(listings)
    if school_map:
        profile_rows = (
            session.query(CanonicalRecord)
            .filter(
                CanonicalRecord.record_type == "faculty_profile",
                or_(
                    CanonicalRecord.record_payload["faculty_label"].as_string() == "null",
                    CanonicalRecord.record_payload["faculty_label"].as_string().is_(None),
                ),
            )
            .all()
        )
        enriched_count = 0
        for row in profile_rows:
            name = row.record_payload.get("name", "")
            faculty_label = school_map.get(name)
            if faculty_label:
                updated_payload = dict(row.record_payload)
                updated_payload["faculty_label"] = faculty_label
                row.record_payload = updated_payload
                enriched_count += 1

        if enriched_count:
            session.flush()
            logger.info(
                "update_cycle.faculty_label_enrichment",
                enriched_count=enriched_count,
                school_map_size=len(school_map),
            )

    # --- Faculty department_label enrichment ---
    faculty_dept_map = build_faculty_department_map(listings)
    if faculty_dept_map:
        faculty_no_dept = (
            session.query(CanonicalRecord)
            .filter(
                CanonicalRecord.record_type == "faculty_profile",
                or_(
                    CanonicalRecord.record_payload["department_label"].as_string() == "null",
                    CanonicalRecord.record_payload["department_label"].as_string().is_(None),
                ),
            )
            .all()
        )
        dept_enriched = 0
        for row in faculty_no_dept:
            name = row.record_payload.get("name", "")
            dept = faculty_dept_map.get(name)
            if dept:
                updated_payload = dict(row.record_payload)
                updated_payload["department_label"] = dept
                row.record_payload = updated_payload
                dept_enriched += 1

        if dept_enriched:
            session.flush()
            logger.info(
                "update_cycle.faculty_department_enrichment",
                enriched_count=dept_enriched,
                map_size=len(faculty_dept_map),
            )

    # --- Program department_label enrichment ---
    program_dept_map = parse_program_department_map(listings)
    if program_dept_map:
        program_rows = (
            session.query(CanonicalRecord)
            .filter(
                CanonicalRecord.record_type == "program",
                or_(
                    CanonicalRecord.record_payload["department_label"].as_string() == "null",
                    CanonicalRecord.record_payload["department_label"].as_string().is_(None),
                ),
            )
            .all()
        )
        dept_enriched = 0
        for row in program_rows:
            program_name = row.record_payload.get("program_name", "")
            entry = fuzzy_program_lookup(program_name, program_dept_map)
            if entry:
                dept_label, _faculty_label = entry
                updated_payload = dict(row.record_payload)
                updated_payload["department_label"] = dept_label
                row.record_payload = updated_payload
                dept_enriched += 1

        if dept_enriched:
            session.flush()
            logger.info(
                "update_cycle.program_department_enrichment",
                enriched_count=dept_enriched,
                map_size=len(program_dept_map),
            )

    # --- Program faculty_label enrichment via faculty map (superset of dept_map) ---
    program_faculty_map = parse_program_faculty_map(listings, dept_map=program_dept_map)
    if program_faculty_map:
        unfilled_program_rows = (
            session.query(CanonicalRecord)
            .filter(
                CanonicalRecord.record_type == "program",
                or_(
                    CanonicalRecord.record_payload["faculty_label"].as_string() == "null",
                    CanonicalRecord.record_payload["faculty_label"].as_string().is_(None),
                ),
            )
            .all()
        )
        fac_map_enriched = 0
        for row in unfilled_program_rows:
            program_name = row.record_payload.get("program_name", "")
            faculty_label = fuzzy_program_lookup(program_name, program_faculty_map)
            if faculty_label:
                updated_payload = dict(row.record_payload)
                updated_payload["faculty_label"] = faculty_label
                row.record_payload = updated_payload
                fac_map_enriched += 1

        if fac_map_enriched:
            session.flush()
            logger.info(
                "update_cycle.program_faculty_map_enrichment",
                enriched_count=fac_map_enriched,
            )

    # --- Document title enrichment ---
    landing_rows = (
        session.query(CanonicalRecord)
        .filter(CanonicalRecord.record_type == "document_landing")
        .all()
    )
    url_to_title: dict[str, str] = {}
    for row in landing_rows:
        payload = row.record_payload or {}
        urls = payload.get("linked_document_urls", [])
        labels = payload.get("linked_document_labels", [])
        for url, label in zip(urls, labels):
            if url and label:
                url_to_title[url] = label

    asset_rows = (
        session.query(CanonicalRecord)
        .filter(CanonicalRecord.record_type == "document_asset")
        .all()
    )
    title_enriched = 0
    for row in asset_rows:
        if row.record_payload.get("document_title"):
            continue
        doc_url = row.record_payload.get("document_url", "")
        title = url_to_title.get(doc_url)
        if not title:
            filename = row.record_payload.get("filename", "")
            if filename:
                stem = filename.rsplit(".", 1)[0]
                derived = stem.replace("-", " ").replace("_", " ").strip()
                if derived:
                    title = derived
        if title:
            updated = dict(row.record_payload)
            updated["document_title"] = title
            row.record_payload = updated
            title_enriched += 1

    if title_enriched:
        session.flush()
        logger.info(
            "update_cycle.document_title_enrichment",
            enriched_count=title_enriched,
        )
