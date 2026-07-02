from __future__ import annotations

import argparse
import sys
from collections import deque
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

from rich.console import Console

if TYPE_CHECKING:
    from unibot.db.repositories.source_registry import SourceRegistryEntry
from rich.live import Live
import structlog

from unibot.cli.display import RichProgressDisplay

logger = structlog.get_logger(__name__)


def _ensure_project_root_on_path() -> None:
    """Add the project root to sys.path so scripts/ is importable."""
    # Walk up from src/unibot/cli/commands.py → project root
    project_root = str(Path(__file__).resolve().parents[3])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def run_update_cycle(args: argparse.Namespace) -> int:
    verbose = getattr(args, "verbose", False)
    quiet = getattr(args, "quiet", False)
    console = Console(quiet=quiet) if quiet else Console()
    is_tty = console.is_terminal and not quiet
    exclude_research_subdomains = getattr(
        args,
        "exclude_research_subdomains",
        False,
    )
    exclude_news = getattr(args, "exclude_news", False)

    if not is_tty or verbose:
        import sys as _sys
        from unibot.logging import configure_logging

        configure_logging(file=_sys.stderr)

    from qdrant_client import QdrantClient

    from unibot.crawl.fetchers import RawArtifactFetcher
    from unibot.db.repositories.source_registry import (
        SourceRegistryRepository,
        build_seed_source_registry_entries,
    )
    from unibot.db.session import direct_session_scope
    from unibot.http_clients import build_provider_http_clients
    from unibot.indexing.provider_factory import create_embedding_provider
    from unibot.indexing.qdrant_writer import QdrantWriter
    from unibot.scheduler.jobs import UpdateCycleOrchestrator
    from unibot.settings import get_settings, retrieval_quality_warning
    from unibot.storage.object_store import LocalObjectStore, S3ObjectStore

    _ensure_project_root_on_path()
    from scripts.run_update_cycle import (
        _discover_source_entries,
        _extract_candidates,
        _fetch_artifact_for_job,
    )

    settings = get_settings()
    if warning := retrieval_quality_warning(settings):
        logger.warning("runtime.retrieval_quality", warning=warning)
    qdrant_url = str(settings.qdrant_url)
    fetcher = RawArtifactFetcher()
    seed_entries = build_seed_source_registry_entries()

    object_store: LocalObjectStore | S3ObjectStore
    if settings.raw_storage_backend == "local":
        object_store = LocalObjectStore(
            base_path=Path(".unibot") / "raw",
        )
    else:
        object_store = S3ObjectStore(
            bucket=settings.raw_object_store_bucket or "",
            prefix=settings.raw_object_store_prefix,
            endpoint_url=(
                str(settings.raw_object_store_endpoint_url)
                if settings.raw_object_store_endpoint_url is not None
                else None
            ),
            region_name=settings.raw_object_store_region,
        )

    with ExitStack() as stack:
        shared_http_clients = build_provider_http_clients(settings)
        stack.callback(shared_http_clients.close)
        qdrant_client = QdrantClient(url=qdrant_url, api_key=settings.qdrant_api_key, timeout=120,)
        stack.callback(qdrant_client.close)
        with direct_session_scope() as session:
            registry_repo = SourceRegistryRepository(session)
            registry_repo.upsert_entries(seed_entries)
            session.flush()
            source_entries = registry_repo.get_active_entries(
                exclude_research_subdomains=exclude_research_subdomains,
                exclude_news=exclude_news,
            )

            if getattr(args, "dry_run", False):
                from unibot.crawl.jobs import select_sources_for_crawl

                planned_jobs = select_sources_for_crawl(source_entries, limit=args.limit)
                if is_tty:
                    from rich.table import Table as RichTable

                    table = RichTable(
                        title="Dry Run: Sources to Crawl",
                        border_style="yellow",
                    )
                    table.add_column("#", style="dim", justify="right")
                    table.add_column("Source URL", style="cyan")
                    table.add_column("Class", style="green")
                    for i, job in enumerate(planned_jobs, 1):
                        table.add_row(
                            str(i),
                            job.source_url,
                            str(job.source_class),
                        )
                    console.print(table)
                    console.print(
                        f"\nWould crawl [bold]{len(planned_jobs)}[/bold] sources. "
                        f"Run without --dry-run to execute."
                    )
                else:
                    for job in planned_jobs:
                        print(f"{job.source_url}\t{job.source_class}")
                    print(f"total: {len(planned_jobs)}")
                return 0

            progress_display: RichProgressDisplay | None = None
            if is_tty:
                progress_display = RichProgressDisplay(
                    console=console, generation_label=args.generation_label
                )

            orchestrator = UpdateCycleOrchestrator(
                session=session,
                qdrant_writer=QdrantWriter(qdrant_client),
                embedding_provider=create_embedding_provider(
                    settings=settings,
                    client=shared_http_clients.cohere,
                ),
                source_entries=source_entries,
                fetch_artifact=lambda job: _fetch_artifact_for_job(fetcher, job),
                extract_candidates=lambda job, artifact: _extract_candidates(
                    job, artifact, source_entries=source_entries
                ),
                discover_sources=lambda job, artifact: _discover_source_entries(
                    job,
                    artifact,
                    exclude_research_subdomains=exclude_research_subdomains,
                ),
                object_store=object_store,
                auto_activate=not args.no_auto_activate,
                progress=progress_display,
            )

            if is_tty and progress_display is not None:
                try:
                    with Live(
                        progress_display.build_display(),
                        console=console,
                        refresh_per_second=4,
                    ) as live:
                        _bind_live_refresh(progress_display, live)
                        result = orchestrator.run(
                            generation_label=args.generation_label,
                            limit=args.limit,
                        )
                except KeyboardInterrupt:
                    console.print()
                    console.print(
                        progress_display.build_summary_table(
                            title="Update Cycle Interrupted"
                        )
                    )
                    if progress_display._all_sources:
                        console.print()
                        console.print(progress_display.build_source_details_table())
                    return 130
                console.print()
                console.print(progress_display.build_summary_table())
                console.print()
                console.print(progress_display.build_source_details_table())
            else:
                result = orchestrator.run(
                    generation_label=args.generation_label,
                    limit=args.limit,
                )
                if args.no_auto_activate:
                    print(
                        f"update cycle completed without activation; "
                        f"pending index jobs remain for generation {result.generation_label}"
                    )
                else:
                    print(
                        f"built and activated generation "
                        f"{result.generation_label} with "
                        f"{len(result.active_record_version_ids)} active records"
                    )

            if getattr(args, "json", False):
                import dataclasses
                import json

                print(json.dumps(dataclasses.asdict(result), default=str))

    return 0


def run_index_jobs(args: argparse.Namespace) -> int:
    console = Console()
    is_tty = console.is_terminal

    from unibot.commands import run_index_jobs_main

    forwarded: list[str] = [
        "--alias-name",
        args.alias_name,
        "--collection-prefix",
        args.collection_prefix,
    ]
    if args.limit is not None:
        forwarded.extend(["--limit", str(args.limit)])

    if is_tty:
        from rich.status import Status

        with Status("[bold cyan]Processing index jobs...", console=console):
            rc = run_index_jobs_main(forwarded)
        if rc == 0:
            console.print("[green]\u2713[/green] Index jobs completed successfully")
        else:
            console.print("[red]\u2717[/red] Some index jobs failed")
        return rc
    else:
        return run_index_jobs_main(forwarded)


def run_build_generation(args: argparse.Namespace) -> int:
    console = Console()
    is_tty = console.is_terminal

    from qdrant_client import QdrantClient

    from unibot.db.repositories.serving_generations import ServingGenerationRepository
    from unibot.db.session import get_direct_session_factory
    from unibot.indexing.provider_factory import create_embedding_provider
    from unibot.indexing.qdrant_writer import QdrantWriter
    from unibot.indexing.serving_generation_builder import ServingGenerationBuilder
    from unibot.settings import get_settings

    settings = get_settings()
    qdrant_client = QdrantClient(
        url=str(settings.qdrant_url), api_key=settings.qdrant_api_key, timeout=6000,
    )

    try:
        session_factory = get_direct_session_factory()
        from unibot.cli.display import _BUILD_GENERATION_PHASES

        progress_display: RichProgressDisplay | None = None
        if is_tty:
            progress_display = RichProgressDisplay(
                console=console,
                generation_label=args.generation_label,
                title="Build Generation",
                phase_order=_BUILD_GENERATION_PHASES,
            )

        builder = ServingGenerationBuilder(
            session_factory=session_factory,
            generation_repository=ServingGenerationRepository(session_factory=session_factory),
            qdrant_writer=QdrantWriter(qdrant_client),
            embedding_provider=create_embedding_provider(settings=settings),
            alias_name=args.alias_name,
            collection_prefix=args.collection_prefix,
            progress=progress_display,
        )

        if is_tty and progress_display is not None:
            with Live(
                progress_display.build_display(),
                console=console,
                refresh_per_second=4,
            ) as live:
                _bind_live_refresh(progress_display, live)
                result = builder.build_and_activate(
                    generation_label=args.generation_label
                )
            console.print()
            console.print(progress_display.build_summary_table())
        else:
            result = builder.build_and_activate(
                generation_label=args.generation_label
            )
            print(
                f"built serving generation "
                f"{result.generation.generation_label} with "
                f"{len(result.record_version_ids)} records"
            )
    finally:
        qdrant_client.close()

    return 0


def run_bootstrap(args: argparse.Namespace) -> int:
    console = Console()
    is_tty = console.is_terminal

    from unibot.db.repositories.source_registry import (
        SourceRegistryRepository,
        build_seed_source_registry_entries,
    )
    from unibot.db.session import direct_session_scope
    from unibot.logging import configure_logging

    configure_logging()
    seed_entries = build_seed_source_registry_entries()
    with direct_session_scope() as session:
        repo = SourceRegistryRepository(session)
        repo.upsert_entries(seed_entries)
    if is_tty:
        console.print(
            f"[green]\u2713[/green] Bootstrapped {len(seed_entries)} source entries"
        )
    else:
        print(f"bootstrapped {len(seed_entries)} source entries")
    return 0


def run_eval(args: argparse.Namespace) -> int:
    _ensure_project_root_on_path()
    from scripts.run_eval import main as eval_main

    return eval_main()


def _filter_seed_entries(
    entries: tuple[SourceRegistryEntry, ...],
    *,
    exclude_research_subdomains: bool = False,
    exclude_news: bool = False,
) -> tuple[SourceRegistryEntry, ...]:
    """Apply source_class filters to seed entries (in-memory equivalent of get_active_entries)."""
    filtered = entries
    if exclude_research_subdomains:
        filtered = tuple(e for e in filtered if e.source_class != "research_subdomain")
    if exclude_news:
        filtered = tuple(e for e in filtered if e.source_class != "news_event")
    return filtered


def _build_local_school_map(
    output_dir: Path,
) -> tuple[dict[str, str], dict[str, str]]:
    """Build faculty school map from already-crawled org_unit HTML files.

    Returns (school_map, listings) where school_map is
    {faculty_name -> faculty_label} and listings is
    {faculty_label -> page_text} for downstream map builders.
    """
    from bs4 import BeautifulSoup

    from unibot.enrich.faculty_school_map import (
        LISTING_SLUG_TO_LABEL,
        build_faculty_school_map,
    )

    org_unit_dir = output_dir / "org_unit"
    if not org_unit_dir.is_dir():
        return {}, {}

    # Sort longest-first so "business-management-science" matches before "science"
    slugs_by_length = sorted(
        LISTING_SLUG_TO_LABEL.items(), key=lambda x: len(x[0]), reverse=True,
    )

    listing_parts: dict[str, list[str]] = {}
    for entry in org_unit_dir.iterdir():
        if not entry.is_dir():
            continue
        html_path = entry / "page.html"
        if not html_path.is_file():
            continue
        for slug, label in slugs_by_length:
            slug_normalised = slug.replace("faculty-of-", "")
            if slug_normalised in entry.name:
                html = html_path.read_text(encoding="utf-8", errors="replace")
                soup = BeautifulSoup(html, "lxml")
                # Strip nav, footer, header, aside to prevent program
                # listings in menus/footers from contaminating enrichment.
                for tag in soup.find_all(["nav", "footer", "header", "aside"]):
                    tag.decompose()
                text = soup.get_text("\n", strip=True)
                listing_parts.setdefault(label, []).append(text)
                break

    listings = {label: "\n".join(parts) for label, parts in listing_parts.items()}
    if not listings:
        return {}, {}
    return build_faculty_school_map(listings), listings


def _apply_local_enrichment(
    output_dir: Path,
) -> None:
    """Apply cross-source enrichment to extract-only JSON output files.

    Enriches:
    1. faculty_profile records with faculty_label (from org_unit listings)
    2. program records with department_label and faculty_label (from org_unit listings)
    3. document_asset records with document_title (from document_landing labels, fallback: filename)
    """
    import json

    from unibot.enrich.apply import (
        enrich_faculty_department_labels,
        enrich_faculty_labels,
        enrich_program_departments,
        enrich_program_faculty_labels,
    )
    from unibot.enrich.faculty_school_map import (
        build_faculty_department_map,
        parse_program_department_map,
        parse_program_faculty_map,
    )

    school_map, listings = _build_local_school_map(output_dir)

    # Build program enrichment maps from the same listings
    program_dept_map = parse_program_department_map(listings) if listings else {}
    program_faculty_map = parse_program_faculty_map(listings) if listings else {}
    faculty_dept_map = build_faculty_department_map(listings) if listings else {}

    # --- Enrich faculty profiles (existing logic, unchanged) ---
    if school_map:
        faculty_dir = output_dir / "faculty"
        if faculty_dir.is_dir():
            enriched_count = 0
            for entry in faculty_dir.iterdir():
                json_path = entry / "extraction.json"
                if not json_path.is_file():
                    continue
                try:
                    data = json.loads(json_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError) as exc:
                    logger.warning(
                        "enrich.json_read_failed", path=str(json_path), error=str(exc),
                    )
                    continue
                records = data.get("records", [])
                before_labels = [
                    (
                        r.get("record_payload", {}).get("faculty_label"),
                        r.get("record_payload", {}).get("department_label"),
                    )
                    for r in records
                    if r.get("record_type") == "faculty_profile"
                ]
                enrich_faculty_labels(records, school_map)
                enrich_faculty_department_labels(records, faculty_dept_map)
                after_labels = [
                    (
                        r.get("record_payload", {}).get("faculty_label"),
                        r.get("record_payload", {}).get("department_label"),
                    )
                    for r in records
                    if r.get("record_type") == "faculty_profile"
                ]
                if before_labels != after_labels:
                    enriched_count += sum(
                        1 for b, a in zip(before_labels, after_labels) if b != a
                    )
                    tmp_path = json_path.with_suffix(".tmp")
                    tmp_path.write_text(
                        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8",
                    )
                    tmp_path.replace(json_path)

            if enriched_count:
                logger.info(
                    "extract_only.faculty_label_enrichment",
                    enriched_count=enriched_count,
                    school_map_size=len(school_map),
                )

    # --- Enrich program records (new) ---
    if program_dept_map or program_faculty_map:
        program_dir = output_dir / "program"
        if program_dir.is_dir():
            enriched_count = 0
            for entry in program_dir.iterdir():
                json_path = entry / "extraction.json"
                if not json_path.is_file():
                    continue
                try:
                    data = json.loads(json_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError) as exc:
                    logger.warning(
                        "enrich.json_read_failed", path=str(json_path), error=str(exc),
                    )
                    continue
                records = data.get("records", [])
                before_labels = [
                    (
                        r.get("record_payload", {}).get("department_label"),
                        r.get("record_payload", {}).get("faculty_label"),
                    )
                    for r in records
                    if r.get("record_type") == "program"
                ]
                enrich_program_departments(records, program_dept_map)
                enrich_program_faculty_labels(records, program_faculty_map)
                after_labels = [
                    (
                        r.get("record_payload", {}).get("department_label"),
                        r.get("record_payload", {}).get("faculty_label"),
                    )
                    for r in records
                    if r.get("record_type") == "program"
                ]
                if before_labels != after_labels:
                    enriched_count += sum(
                        1 for b, a in zip(before_labels, after_labels) if b != a
                    )
                    tmp_path = json_path.with_suffix(".tmp")
                    tmp_path.write_text(
                        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8",
                    )
                    tmp_path.replace(json_path)

            if enriched_count:
                logger.info(
                    "extract_only.program_enrichment",
                    enriched_count=enriched_count,
                    dept_map_size=len(program_dept_map),
                    faculty_map_size=len(program_faculty_map),
                )

    # --- Enrich document_asset records with document_title ---
    landing_dir = output_dir / "document_landing"
    asset_dir = output_dir / "document_asset"
    if landing_dir.is_dir() and asset_dir.is_dir():
        # Build url→title map from document_landing records
        url_to_title: dict[str, str] = {}
        for entry in landing_dir.iterdir():
            json_path = entry / "extraction.json"
            if not json_path.is_file():
                continue
            try:
                landing_data = json.loads(json_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            for rec in landing_data.get("records", []):
                payload = rec.get("record_payload", {})
                urls = payload.get("linked_document_urls", [])
                labels = payload.get("linked_document_labels", [])
                for url, label in zip(urls, labels):
                    if url and label:
                        url_to_title[url] = label

        # Enrich asset records
        title_enriched = 0
        for entry in asset_dir.iterdir():
            json_path = entry / "extraction.json"
            if not json_path.is_file():
                continue
            try:
                asset_data = json.loads(json_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            changed = False
            for rec in asset_data.get("records", []):
                if rec.get("record_type") != "document_asset":
                    continue
                payload = rec.get("record_payload", {})
                if payload.get("document_title"):
                    continue
                doc_url = payload.get("document_url", "")
                title = url_to_title.get(doc_url)
                if not title:
                    filename = payload.get("filename", "")
                    if filename:
                        stem = filename.rsplit(".", 1)[0]
                        derived = stem.replace("-", " ").replace("_", " ").strip()
                        if derived:
                            title = derived
                if title:
                    payload["document_title"] = title
                    title_enriched += 1
                    changed = True
            if changed:
                tmp_path = json_path.with_suffix(".tmp")
                tmp_path.write_text(
                    json.dumps(asset_data, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                tmp_path.replace(json_path)

        if title_enriched:
            logger.info(
                "extract_only.document_title_enrichment",
                enriched_count=title_enriched,
                url_map_size=len(url_to_title),
            )


def run_extract_only(args: argparse.Namespace) -> int:
    import json

    verbose = getattr(args, "verbose", False)
    quiet = getattr(args, "quiet", False)
    console = Console(quiet=quiet) if quiet else Console()
    is_tty = console.is_terminal and not quiet
    exclude_research_subdomains = getattr(
        args, "exclude_research_subdomains", False,
    )
    exclude_news = getattr(args, "exclude_news", False)
    output_dir = Path(getattr(args, "output", ".unibot/extract-only"))

    if not is_tty or verbose:
        import sys as _sys
        from unibot.logging import configure_logging

        configure_logging(file=_sys.stderr)

    from unibot.crawl.fetchers import FetchedArtifact, RawArtifactFetcher
    from unibot.crawl.jobs import select_sources_for_crawl
    from unibot.db.repositories.source_registry import (
        build_seed_source_registry_entries,
    )
    from unibot.extract.dispatch import dispatch_extraction

    _ensure_project_root_on_path()
    from scripts.run_update_cycle import (
        _build_extraction_context,
        _discover_source_entries,
        _fetch_artifact_for_job,
    )

    fetcher = RawArtifactFetcher()
    seed_entries = build_seed_source_registry_entries()

    # In-memory filtering — no DB needed. Seed entries already have
    # is_active=True and last_crawled_at=None (force=True bypasses timestamps).
    source_entries = _filter_seed_entries(
        seed_entries,
        exclude_research_subdomains=exclude_research_subdomains,
        exclude_news=exclude_news,
    )
    # Mutable collection of all known entries (seeds + discovered) for
    # _build_extraction_context's verification_state_by_url mapping.
    all_known_entries: list[SourceRegistryEntry] = list(source_entries)

    selected_jobs = list(select_sources_for_crawl(
        source_entries, limit=args.limit, force=True,
    ))

    sources_ok = 0
    sources_failed = 0

    if is_tty:
        console.print(
            f"[bold cyan]Extract Only[/bold cyan]: "
            f"{len(selected_jobs)} sources selected"
        )

    processed_urls: set[str] = set()
    queued_urls = {job.source_url for job in selected_jobs}
    job_queue = deque(selected_jobs)

    while job_queue:
        job = job_queue.popleft()
        queued_urls.discard(job.source_url)
        if job.source_url in processed_urls:
            continue
        processed_urls.add(job.source_url)

        if is_tty:
            console.print(
                f"  [dim]\u2192[/dim] {job.source_url} "
                f"[dim]({job.source_class})[/dim]"
            )

        try:
            artifact = _fetch_artifact_for_job(fetcher, job)
        except Exception as exc:
            sources_failed += 1
            if is_tty:
                console.print(f"    [red]\u2717[/red] fetch failed: {exc}")
            continue

        if not isinstance(artifact, FetchedArtifact):
            sources_failed += 1
            continue

        # Discover child sources for queue expansion (purely in-memory)
        try:
            discovered = _discover_source_entries(
                job, artifact,
                exclude_research_subdomains=exclude_research_subdomains,
            )
            if exclude_news:
                discovered = tuple(
                    e for e in discovered if e.source_class != "news_event"
                )
            if discovered:
                all_known_entries.extend(discovered)
                # force=True: extract everything regardless of crawl timestamps
                for child_job in select_sources_for_crawl(discovered, force=True):
                    if child_job.source_url not in processed_urls and child_job.source_url not in queued_urls:
                        job_queue.append(child_job)
                        queued_urls.add(child_job.source_url)
        except Exception:
            logger.debug(
                "extract_only.discovery_failed",
                source_url=job.source_url,
                exc_info=True,
            )

        try:
            context = _build_extraction_context(
                job, artifact, source_entries=tuple(all_known_entries),
            )
            extraction = dispatch_extraction(context=context)
        except Exception as exc:
            sources_failed += 1
            if is_tty:
                console.print(f"    [red]\u2717[/red] extraction failed: {exc}")
            continue

        _write_extraction_result(
            output_dir, job, artifact, extraction, json_dumps=json.dumps,
        )
        sources_ok += 1

        if is_tty:
            n_records = len(extraction.records)
            n_evidence = len(extraction.evidence_records)
            console.print(
                f"    [green]\u2713[/green] "
                f"{n_records} records, {n_evidence} evidence"
            )

    # Apply cross-source enrichment (faculty_label from org_unit listing pages)
    _apply_local_enrichment(output_dir)

    if is_tty:
        console.print()
        console.print(
            f"[bold green]\u2713[/bold green] "
            f"Extracted {sources_ok} sources "
            f"({sources_failed} failed) "
            f"\u2192 [cyan]{output_dir}/[/cyan]"
        )
    else:
        print(f"wrote {sources_ok} extraction results to {output_dir}/")

    return 0


def _write_extraction_result(
    output_dir: Path,
    job: Any,
    artifact: Any,
    extraction: Any,
    *,
    json_dumps: Any = None,
) -> Path:
    """Write extraction results for a single source into a per-source subfolder.

    Layout::

        <output_dir>/
          <source_class>/
            <slug>/
              extraction.json   — records + evidence
              page.html         — raw HTML (when available)
              page.md           — crawl4ai markdown (when available)
    """
    import json as _json

    from unibot.extract.records import serialize_for_storage

    _dumps = json_dumps or _json.dumps

    from urllib.parse import urlsplit

    import hashlib as _hashlib

    path = urlsplit(job.source_url).path.strip("/") or "index"
    slug = path.replace("/", "_")
    if len(slug) > 200:
        slug_hash = _hashlib.sha256(path.encode()).hexdigest()[:12]
        slug = slug[:187] + "_" + slug_hash

    source_dir = output_dir / str(job.source_class) / slug
    source_dir.mkdir(parents=True, exist_ok=True)

    # -- raw HTML
    html_content = getattr(artifact, "content", None)
    if html_content and "html" in getattr(artifact, "content_type", "").lower():
        (source_dir / "page.html").write_bytes(html_content)

    # -- raw markdown
    metadata = getattr(artifact, "metadata", None) or {}
    raw_markdown = metadata.get("markdown")
    if isinstance(raw_markdown, str) and raw_markdown.strip():
        (source_dir / "page.md").write_text(raw_markdown, encoding="utf-8")

    # -- extraction results
    records = [
        {
            "record_id": r.record_id,
            "record_type": r.record_type,
            "source_url": r.source_url,
            "source_section_id": r.source_section_id,
            "source_section_label": r.source_section_label,
            "source_locator": r.source_locator,
            "source_authority_tier": r.source_authority_tier,
            "conflict_scope_id": r.conflict_scope_id,
            "dedupe_key": r.dedupe_key,
            "cycle_label": r.cycle_label,
            "year_confidence": r.year_confidence,
            "record_payload": serialize_for_storage(r.record_payload),
        }
        for r in extraction.records
    ]
    evidence = [
        {
            "record_id": e.record_id,
            "record_type": e.record_type,
            "source_url": e.source_url,
            "source_section_id": e.source_section_id,
            "source_section_label": e.source_section_label,
            "source_locator": e.source_locator,
            "source_authority_tier": e.source_authority_tier,
            "conflict_scope_id": e.conflict_scope_id,
            "dedupe_key": e.dedupe_key,
            "value_text": e.value_text,
            "record_payload": serialize_for_storage(e.record_payload),
        }
        for e in extraction.evidence_records
    ]
    result = {
        "source_url": job.source_url,
        "source_class": str(job.source_class),
        "records": records,
        "evidence_records": evidence,
    }

    extraction_path = source_dir / "extraction.json"
    extraction_path.write_text(
        _dumps(result, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )
    return extraction_path


def run_cache_purge(args: argparse.Namespace) -> int:
    import os

    cache_dir = Path(
        os.environ.get("UNIBOT_DOCUMENT_CACHE_DIR", ".unibot/cache/document_parser")
    )
    if not cache_dir.exists():
        print("document parser cache: nothing to purge (directory does not exist)")
        return 0
    count = 0
    for f in cache_dir.glob("*.json"):
        f.unlink()
        count += 1
    print(f"document parser cache: purged {count} entries")
    return 0


class _CommandEntry(TypedDict):
    run: Callable[[argparse.Namespace], int]
    description: str


COMMAND_REGISTRY: dict[str, _CommandEntry] = {
    "update-cycle": {
        "run": run_update_cycle,
        "description": "Crawl, extract, verify, and optionally activate",
    },
    "extract-only": {
        "run": run_extract_only,
        "description": "Run parsers/extractors and save results locally as JSON",
    },
    "index-jobs": {
        "run": run_index_jobs,
        "description": "Process pending embedding/indexing jobs",
    },
    "build-generation": {
        "run": run_build_generation,
        "description": "Build and activate a serving generation",
    },
    "bootstrap": {
        "run": run_bootstrap,
        "description": "Initialize source registry with seed entries",
    },
    "eval": {
        "run": run_eval,
        "description": "Evaluate against the active generation",
    },
    "cache-purge": {
        "run": run_cache_purge,
        "description": "Purge the document parser cache",
    },
}


def _bind_live_refresh(
    display: RichProgressDisplay, live: Live
) -> None:
    """Wrap all callback methods to auto-refresh the Live display."""
    original_methods = {
        "on_source_done": display.on_source_done,
        "on_source_failed": display.on_source_failed,
        "on_source_start": display.on_source_start,
        "on_phase_start": display.on_phase_start,
        "on_phase_done": display.on_phase_done,
        "on_sources_discovered": display.on_sources_discovered,
        "on_generation_step": display.on_generation_step,
        "on_generation_progress": display.on_generation_progress,
    }

    def _make_wrapper(fn):
        def wrapper(*a, **kw):
            fn(*a, **kw)
            live.update(display.build_display())

        return wrapper

    for name, fn in original_methods.items():
        setattr(display, name, _make_wrapper(fn))
