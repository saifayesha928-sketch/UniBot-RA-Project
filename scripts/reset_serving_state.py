"""Wipe all Qdrant generation collections and reset DB serving state.

Usage:
    uv run python scripts/reset_serving_state.py
"""

from __future__ import annotations

from qdrant_client import QdrantClient, models
from sqlalchemy import delete, update

from unibot.db.models import CanonicalRecord, ServingGeneration
from unibot.db.session import direct_session_scope
from unibot.settings import get_settings


def main() -> int:
    settings = get_settings()
    client = QdrantClient(url=str(settings.qdrant_url), api_key=settings.qdrant_api_key)

    aliases = client.get_aliases().aliases
    collections = [
        c.name
        for c in client.get_collections().collections
        if c.name.startswith("unibot-generation")
    ]

    with direct_session_scope() as session:
        gen_count = session.query(ServingGeneration).count()
        indexed_count = (
            session.query(CanonicalRecord)
            .filter(CanonicalRecord.serving_status == "indexed_active")
            .count()
        )
        eligible_count = (
            session.query(CanonicalRecord)
            .filter(CanonicalRecord.serving_status == "eligible")
            .count()
        )

    print(f"Aliases: {[a.alias_name for a in aliases]}")
    print(f"Generation collections: {collections}")
    print(f"ServingGeneration rows: {gen_count}")
    print(f"CanonicalRecord indexed_active: {indexed_count}")
    print(f"CanonicalRecord eligible: {eligible_count}")
    print()

    # 1. Remove alias
    delete_ops = [
        models.DeleteAliasOperation(
            delete_alias=models.DeleteAlias(alias_name=alias.alias_name)
        )
        for alias in aliases
        if alias.alias_name == "unibot-active"
    ]
    if delete_ops:
        client.update_collection_aliases(delete_ops)
        print("Removed unibot-active alias")
    else:
        print("No unibot-active alias to remove")

    # 2. Delete generation collections
    for name in collections:
        client.delete_collection(name)
        print(f"Deleted collection: {name}")

    # 3. Clean DB state
    with direct_session_scope() as session:
        session.execute(delete(ServingGeneration))
        result = session.execute(
            update(CanonicalRecord)
            .where(CanonicalRecord.serving_status == "indexed_active")
            .values(serving_status="eligible")
        )
        session.commit()
        print("Deleted all ServingGeneration rows")
        print(f"Reset {result.rowcount} records from indexed_active -> eligible")  # type: ignore[attr-defined]

    # 4. Verify
    with direct_session_scope() as session:
        gen_count = session.query(ServingGeneration).count()
        indexed_count = (
            session.query(CanonicalRecord)
            .filter(CanonicalRecord.serving_status == "indexed_active")
            .count()
        )
        eligible_count = (
            session.query(CanonicalRecord)
            .filter(CanonicalRecord.serving_status == "eligible")
            .count()
        )

    remaining_aliases = client.get_aliases().aliases
    remaining_collections = [
        c.name
        for c in client.get_collections().collections
        if c.name.startswith("unibot-generation")
    ]

    print("\n--- Post-wipe state ---")
    print(f"ServingGeneration rows: {gen_count}")
    print(f"CanonicalRecord indexed_active: {indexed_count}")
    print(f"CanonicalRecord eligible: {eligible_count}")
    print(f"Remaining aliases: {[a.alias_name for a in remaining_aliases]}")
    print(f"Remaining generation collections: {remaining_collections}")
    print("\nDone. Ready to rebuild generations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
