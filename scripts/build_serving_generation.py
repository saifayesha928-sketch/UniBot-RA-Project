from __future__ import annotations

import argparse
from collections.abc import Sequence

from qdrant_client import QdrantClient

from unibot.db.repositories.serving_generations import ServingGenerationRepository
from unibot.db.session import get_direct_session_factory
from unibot.indexing.provider_factory import (
    HashingEmbeddingProvider as _HashingEmbeddingProvider,
    create_embedding_provider,
)
from unibot.indexing.serving_generation_builder import ServingGenerationBuilder
from unibot.indexing.qdrant_writer import QdrantWriter
from unibot.settings import get_settings

HashingEmbeddingProvider = _HashingEmbeddingProvider


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="build_serving_generation")
    parser.add_argument("--generation-label", required=True)
    parser.add_argument("--alias-name", default="unibot-active")
    parser.add_argument("--collection-prefix", default="unibot-generation")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    settings = get_settings()

    session_factory = get_direct_session_factory()
    builder = ServingGenerationBuilder(
        session_factory=session_factory,
        generation_repository=ServingGenerationRepository(session_factory=session_factory),
        qdrant_writer=QdrantWriter(QdrantClient(url=str(settings.qdrant_url), api_key=settings.qdrant_api_key, timeout=120)),
        embedding_provider=create_embedding_provider(settings=settings),
        alias_name=args.alias_name,
        collection_prefix=args.collection_prefix,
    )
    result = builder.build_and_activate(generation_label=args.generation_label)

    print(
        "built serving generation "
        f"{result.generation.generation_label} with {len(result.record_version_ids)} records"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
