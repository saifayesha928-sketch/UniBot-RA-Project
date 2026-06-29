from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="unibot-cli",
        description="UniBot unified CLI",
    )
    subcommands = parser.add_subparsers(dest="command")

    # update-cycle
    uc = subcommands.add_parser(
        "update-cycle",
        help="Crawl, extract, verify, and optionally activate a generation.",
    )
    uc.add_argument("--generation-label")
    uc.add_argument("--limit", type=int)
    uc.add_argument(
        "--exclude-research-subdomains",
        action="store_true",
        default=None,
        help="Exclude research subdomains from discovery, crawl selection, and extraction context.",
    )
    uc.add_argument(
        "--exclude-news",
        action="store_true",
        default=None,
        help="Exclude news/events sources from crawl and extraction.",
    )
    uc.add_argument(
        "--no-auto-activate",
        action="store_true",
        default=None,
        help="Stop after crawl/extract/verify; leave activation to run-index-jobs.",
    )
    uc.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Show which sources would be crawled without executing.",
    )
    verbosity = uc.add_mutually_exclusive_group()
    verbosity.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Show detailed per-source output alongside progress display.",
    )
    verbosity.add_argument(
        "--quiet", "-q",
        action="store_true",
        default=False,
        help="Output only JSON result to stdout (for scripting).",
    )
    uc.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output result as JSON to stdout.",
    )

    # extract-only
    eo = subcommands.add_parser(
        "extract-only",
        help="Run parsers/extractors and save results locally as JSON (no DB upsert or verification).",
    )
    eo.add_argument("--limit", type=int)
    eo.add_argument(
        "--exclude-research-subdomains",
        action="store_true",
        default=None,
        help="Exclude research subdomains from discovery and extraction context.",
    )
    eo.add_argument(
        "--exclude-news",
        action="store_true",
        default=None,
        help="Exclude news/events sources from crawl and extraction.",
    )
    eo.add_argument(
        "--output", "-o",
        default=None,
        help="Output directory (default: .unibot/extract-only/).",
    )
    eo_verbosity = eo.add_mutually_exclusive_group()
    eo_verbosity.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Show detailed per-source output.",
    )
    eo_verbosity.add_argument(
        "--quiet", "-q",
        action="store_true",
        default=False,
        help="Suppress interactive output.",
    )

    # index-jobs (replaces old run-index-jobs)
    ij = subcommands.add_parser(
        "index-jobs",
        help="Process pending embedding/indexing jobs.",
    )
    ij.add_argument("--alias-name", default="unibot-active")
    ij.add_argument("--collection-prefix", default="unibot-generation")
    ij.add_argument("--limit", type=int)

    # build-generation
    bg = subcommands.add_parser(
        "build-generation",
        help="Build and activate a serving generation.",
    )
    bg.add_argument("--generation-label")
    bg.add_argument("--alias-name", default="unibot-active")
    bg.add_argument("--collection-prefix", default="unibot-generation")

    # bootstrap
    subcommands.add_parser(
        "bootstrap",
        help="Initialize source registry with seed entries.",
    )

    # eval
    subcommands.add_parser(
        "eval",
        help="Evaluate against the active generation.",
    )

    # Legacy commands (kept for backward compat)
    subcommands.add_parser(
        "check-neon-connection",
        help="Verify both direct and pooled Neon connections.",
    )

    # run-index-jobs (legacy alias)
    rij = subcommands.add_parser(
        "run-index-jobs",
        help="(Legacy) Rebuild and activate a serving generation from pending index jobs.",
    )
    rij.add_argument("--alias-name", default="unibot-active")
    rij.add_argument("--collection-prefix", default="unibot-generation")
    rij.add_argument("--limit", type=int)

    sa = subcommands.add_parser("serve-api", help="Run the UniBot HTTP API.")
    sa.add_argument("--host", default="127.0.0.1")
    sa.add_argument("--port", type=int, default=8000)

    subcommands.add_parser(
        "cache-purge",
        help="Purge the document parser cache.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    # No subcommand: interactive mode (TTY only)
    if args.command is None:
        if not sys.stdout.isatty():
            parser.print_help()
            return 2
        from unibot.cli.interactive import prompt_job_selection

        args.command = prompt_job_selection()
        return _dispatch(args)

    return _dispatch(args)


def _dispatch(args: argparse.Namespace) -> int:
    command = args.command

    # Legacy commands
    if command == "check-neon-connection":
        from unibot.commands import check_neon_connection_main

        return check_neon_connection_main()
    if command == "serve-api":
        import uvicorn

        uvicorn.run(
            "main:create_app", host=args.host, port=args.port, factory=True
        )
        return 0
    if command == "run-index-jobs":
        # Legacy alias → forward to index-jobs handler
        command = "index-jobs"

    # Interactive prompt for missing required args (TTY only)
    if sys.stdout.isatty():
        from unibot.cli.interactive import (
            prompt_build_generation_args,
            prompt_extract_only_args,
            prompt_index_jobs_args,
            prompt_update_cycle_args,
        )

        if command == "update-cycle":
            args = prompt_update_cycle_args(args)
        elif command == "extract-only":
            args = prompt_extract_only_args(args)
        elif command == "index-jobs":
            args = prompt_index_jobs_args(args)
        elif command == "build-generation":
            args = prompt_build_generation_args(args)

    # Ensure required args have values for non-TTY
    if command == "update-cycle" and not getattr(args, "generation_label", None):
        print("error: --generation-label is required", file=sys.stderr)
        return 2
    if command == "build-generation" and not getattr(
        args, "generation_label", None
    ):
        print("error: --generation-label is required", file=sys.stderr)
        return 2

    # Set defaults for missing optional flags
    if command == "update-cycle" and args.no_auto_activate is None:
        args.no_auto_activate = False
    if command == "update-cycle" and args.exclude_research_subdomains is None:
        args.exclude_research_subdomains = False
    if command == "update-cycle" and args.exclude_news is None:
        args.exclude_news = False
    if command == "extract-only" and args.exclude_research_subdomains is None:
        args.exclude_research_subdomains = False
    if command == "extract-only" and args.exclude_news is None:
        args.exclude_news = False
    if command == "extract-only" and not args.output:
        args.output = ".unibot/extract-only"

    from unibot.cli.commands import COMMAND_REGISTRY

    entry = COMMAND_REGISTRY.get(command)
    if entry is None:
        print(f"error: unknown command '{command}'", file=sys.stderr)
        return 2
    return entry["run"](args)
