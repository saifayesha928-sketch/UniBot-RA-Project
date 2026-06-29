"""Quick verification: query Neon + Qdrant to check serving generation state."""
from __future__ import annotations

import json
import os
import sys
from dotenv import load_dotenv

load_dotenv()
from qdrant_client import QdrantClient
from sqlalchemy import create_engine, text


def main() -> None:
    # --- Neon (Postgres) ---
    dsn = os.environ.get("UNIBOT_POSTGRES_DIRECT_DSN")
    if not dsn:
        print("ERROR: UNIBOT_POSTGRES_DIRECT_DSN not set")
        sys.exit(1)

    engine = create_engine(dsn)
    print("=== Neon: serving_generations ===")
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT generation_id, generation_label, status, qdrant_collection, "
                "activated_at, created_at, "
                "generation_metadata "
                "FROM serving_generations ORDER BY created_at DESC LIMIT 5"
            )
        ).fetchall()
        for row in rows:
            metadata = row[6] or {}
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            record_ids = metadata.get("record_version_ids", [])
            failed_ids = metadata.get("failed_record_version_ids", [])
            print(
                f"  id={row[0]} label={row[1]} status={row[2]} "
                f"collection={row[3]} activated={row[4]} created={row[5]} "
                f"records={len(record_ids)} failed={len(failed_ids)}"
            )

    # --- Qdrant ---
    qdrant_url = os.environ.get("UNIBOT_QDRANT_URL")
    qdrant_api_key = os.environ.get("UNIBOT_QDRANT_API_KEY")
    if not qdrant_url:
        print("\nERROR: UNIBOT_QDRANT_URL not set")
        sys.exit(1)

    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    print("\n=== Qdrant: aliases ===")
    aliases = client.get_aliases().aliases
    for alias in aliases:
        print(f"  alias={alias.alias_name} -> collection={alias.collection_name}")

    print("\n=== Qdrant: collections ===")
    collections = client.get_collections().collections
    for coll in collections:
        info = client.get_collection(coll.name)
        print(f"  collection={coll.name} points={info.points_count} status={info.status}")
        # Show all available count fields
        for attr in ("vectors_count", "indexed_vectors_count"):
            if hasattr(info, attr):
                print(f"    {attr}={getattr(info, attr)}")


if __name__ == "__main__":
    main()
