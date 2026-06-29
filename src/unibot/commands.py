from __future__ import annotations

import argparse
from collections.abc import Sequence
from contextlib import ExitStack

import structlog
from qdrant_client import QdrantClient
from sqlalchemy import create_engine, text

from unibot.db.session import get_database_settings, get_direct_session_factory
from unibot.http_clients import build_provider_http_clients
from unibot.indexing.index_job_executor import IndexJobExecutor
from unibot.indexing.provider_factory import create_embedding_provider
from unibot.indexing.qdrant_writer import QdrantWriter
from unibot.logging import configure_logging
from unibot.settings import get_settings, retrieval_quality_warning

logger = structlog.get_logger(__name__)


def build_run_index_jobs_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run_index_jobs")
    parser.add_argument("--alias-name", default="unibot-active")
    parser.add_argument("--collection-prefix", default="unibot-generation")
    parser.add_argument("--limit", type=int)
    return parser


def run_index_jobs_main(argv: Sequence[str] | None = None) -> int:
    parser = build_run_index_jobs_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    settings = get_settings()
    if warning := retrieval_quality_warning(settings):
        logger.warning("runtime.retrieval_quality", warning=warning)
    qdrant_url = str(settings.qdrant_url)

    with ExitStack() as stack:
        shared_http_clients = build_provider_http_clients(settings)
        stack.callback(shared_http_clients.close)
        qdrant_client = QdrantClient(url=qdrant_url, api_key=settings.qdrant_api_key, timeout=120)
        stack.callback(qdrant_client.close)

        session_factory = get_direct_session_factory()
        executor = IndexJobExecutor(
            session_factory=session_factory,
            qdrant_writer=QdrantWriter(qdrant_client),
            embedding_provider=create_embedding_provider(
                settings=settings,
                client=shared_http_clients.cohere,
            ),
            alias_name=args.alias_name,
            collection_prefix=args.collection_prefix,
        )
        result = executor.run_pending_jobs(limit=args.limit)

    print(
        "processed index jobs: "
        f"processed={result.processed_count} "
        f"succeeded={result.succeeded_count} "
        f"failed={result.failed_count}"
    )
    return 1 if result.failed_count else 0


def _check_connection(name: str, dsn: str) -> None:
    engine = create_engine(dsn, pool_pre_ping=True)
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        print(f"{name}: ok")
    finally:
        engine.dispose()


def check_neon_connection_main() -> int:
    configure_logging()
    settings = get_database_settings()
    _check_connection("direct", settings.postgres_direct_dsn)
    _check_connection("pooled", settings.postgres_pooled_dsn)
    return 0
