from __future__ import annotations

import argparse
from datetime import date

from InquirerPy import inquirer
from InquirerPy.base.control import Choice

JOB_CHOICES: list[dict[str, str]] = [
    {"value": "update-cycle", "name": "Update Cycle     \u2014 Crawl, extract, verify, and optionally activate"},
    {"value": "extract-only", "name": "Extract Only     \u2014 Run parsers/extractors and save results locally as JSON"},
    {"value": "index-jobs", "name": "Index Jobs       \u2014 Process pending embedding/indexing jobs"},
    {"value": "build-generation", "name": "Build Generation \u2014 Build and activate a serving generation"},
    {"value": "bootstrap", "name": "Bootstrap        \u2014 Initialize source registry with seed entries"},
    {"value": "eval", "name": "Run Eval         \u2014 Evaluate against the active generation"},
]


def prompt_job_selection() -> str:
    choices = [Choice(value=c["value"], name=c["name"]) for c in JOB_CHOICES]
    result: str = inquirer.fuzzy(
        message="Select a job to run:",
        choices=choices,
        border=True,
    ).execute()
    return result


def prompt_update_cycle_args(args: argparse.Namespace) -> argparse.Namespace:
    if not getattr(args, "generation_label", None):
        args.generation_label = inquirer.text(
            message="Generation label:",
            default=date.today().isoformat(),
        ).execute()
    if not getattr(args, "limit", None):
        limit_str = inquirer.text(
            message="Source limit (press Enter to skip):",
            default="",
        ).execute()
        args.limit = int(limit_str) if limit_str.strip() else None
    if not hasattr(args, "no_auto_activate") or args.no_auto_activate is None:
        activate = inquirer.confirm(
            message="Activate the new generation automatically?",
            default=False,
        ).execute()
        args.no_auto_activate = not activate
    if (
        not hasattr(args, "exclude_research_subdomains")
        or args.exclude_research_subdomains is None
    ):
        args.exclude_research_subdomains = inquirer.confirm(
            message="Exclude research subdomains?",
            default=False,
        ).execute()
    if (
        not hasattr(args, "exclude_news")
        or args.exclude_news is None
    ):
        args.exclude_news = inquirer.confirm(
            message="Exclude news/events?",
            default=False,
        ).execute()
    return args


def prompt_extract_only_args(args: argparse.Namespace) -> argparse.Namespace:
    if not getattr(args, "limit", None):
        limit_str = inquirer.text(
            message="Source limit (press Enter to skip):",
            default="",
        ).execute()
        args.limit = int(limit_str) if limit_str.strip() else None
    if (
        not hasattr(args, "exclude_research_subdomains")
        or args.exclude_research_subdomains is None
    ):
        args.exclude_research_subdomains = inquirer.confirm(
            message="Exclude research subdomains?",
            default=False,
        ).execute()
    if (
        not hasattr(args, "exclude_news")
        or args.exclude_news is None
    ):
        args.exclude_news = inquirer.confirm(
            message="Exclude news/events?",
            default=False,
        ).execute()
    if not getattr(args, "output", None):
        args.output = inquirer.text(
            message="Output directory:",
            default=".unibot/extract-only",
        ).execute()
    return args


def prompt_index_jobs_args(args: argparse.Namespace) -> argparse.Namespace:
    if not getattr(args, "alias_name", None):
        args.alias_name = inquirer.text(
            message="Alias name:",
            default="unibot-active",
        ).execute()
    if not getattr(args, "collection_prefix", None):
        args.collection_prefix = inquirer.text(
            message="Collection prefix:",
            default="unibot-generation",
        ).execute()
    if not getattr(args, "limit", None):
        limit_str = inquirer.text(
            message="Job limit (press Enter to skip):",
            default="",
        ).execute()
        args.limit = int(limit_str) if limit_str.strip() else None
    return args


def prompt_build_generation_args(args: argparse.Namespace) -> argparse.Namespace:
    if not getattr(args, "generation_label", None):
        args.generation_label = inquirer.text(
            message="Generation label:",
            default=date.today().isoformat(),
        ).execute()
    if not getattr(args, "alias_name", None):
        args.alias_name = inquirer.text(
            message="Alias name:",
            default="unibot-active",
        ).execute()
    if not getattr(args, "collection_prefix", None):
        args.collection_prefix = inquirer.text(
            message="Collection prefix:",
            default="unibot-generation",
        ).execute()
    return args
