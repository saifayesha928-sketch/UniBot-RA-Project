from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Mapping
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from unibot.indexing.embeddings import EmbeddedChunk


@dataclass(frozen=True, slots=True)
class QdrantRecordPoint:
    point_id: str
    record_version_id: str
    text: str
    metadata: dict[str, Any]
    dense_vector: tuple[float, ...]
    sparse_indices: tuple[int, ...]
    sparse_values: tuple[float, ...]
    original_text: str = ""  # Clean text for citations; empty = use text


class QdrantWriter:
    def __init__(self, client: QdrantClient) -> None:
        self._client = client

    def ensure_collection(
        self,
        collection_name: str,
        *,
        dense_vector_size: int,
        fail_if_exists: bool = False,
    ) -> None:
        if self._client.collection_exists(collection_name):
            if fail_if_exists:
                raise ValueError(f"Collection already exists: {collection_name}")
            return

        self._client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": models.VectorParams(
                    size=dense_vector_size,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(),
            },
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Payload indexes have no effect in the local Qdrant",
                category=UserWarning,
            )
            for field_name in (
                "serving_generation_id",
                "freshness_status",
                "source_class",
                "record_type",
            ):
                self._client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            self._client.create_payload_index(
                collection_name=collection_name,
                field_name="source_authority_tier",
                field_schema=models.PayloadSchemaType.INTEGER,
            )

    def build_point(
        self,
        *,
        point_id: str | None = None,
        record_version_id: str,
        text: str,
        metadata: Mapping[str, Any],
        dense_vector: tuple[float, ...],
        sparse_indices: tuple[int, ...],
        sparse_values: tuple[float, ...],
        original_text: str = "",
    ) -> QdrantRecordPoint:
        return QdrantRecordPoint(
            point_id=point_id or record_version_id,
            record_version_id=record_version_id,
            text=text,
            metadata=dict(metadata),
            dense_vector=dense_vector,
            sparse_indices=sparse_indices,
            sparse_values=sparse_values,
            original_text=original_text,
        )

    def point_from_embedded_chunk(self, embedded_chunk: EmbeddedChunk) -> QdrantRecordPoint:
        return self.build_point(
            point_id=embedded_chunk.chunk.chunk_id,
            record_version_id=embedded_chunk.record_version_id,
            text=embedded_chunk.chunk.text,
            original_text=embedded_chunk.chunk.original_text,
            metadata={
                **embedded_chunk.chunk.metadata,
                "chunk_id": embedded_chunk.chunk.chunk_id,
            },
            dense_vector=embedded_chunk.vectors.dense_vector,
            sparse_indices=embedded_chunk.vectors.sparse_vector.indices,
            sparse_values=embedded_chunk.vectors.sparse_vector.values,
        )

    def upsert_records(
        self,
        collection_name: str,
        points: list[QdrantRecordPoint] | tuple[QdrantRecordPoint, ...],
        *,
        batch_size: int = 16,
    ) -> None:
        point_list = list(points)
        for batch_start in range(0, len(point_list), batch_size):
            batch = point_list[batch_start : batch_start + batch_size]
            self._client.upsert(
                collection_name=collection_name,
                points=[
                    models.PointStruct(
                        id=_point_id(point.point_id),
                        vector={
                            "dense": list(point.dense_vector),
                            "sparse": models.SparseVector(
                                indices=list(point.sparse_indices),
                                values=list(point.sparse_values),
                            ),
                        },
                        payload={
                            **point.metadata,
                            "record_version_id": point.record_version_id,
                            "text": point.text,
                            "original_text": point.original_text,
                        },
                    )
                    for point in batch
                ],
                wait=True,
            )

    def delete_point_ids(
        self,
        collection_name: str,
        point_ids: list[str] | tuple[str, ...],
    ) -> None:
        if not point_ids:
            return
        self._client.delete(
            collection_name,
            [_point_id(point_id) for point_id in point_ids],
            wait=True,
        )

    def delete_by_filter(
        self,
        collection_name: str,
        payload_filter: Mapping[str, str | int | bool],
    ) -> None:
        if not payload_filter:
            return
        must_conditions: list[models.FieldCondition | models.IsEmptyCondition | models.IsNullCondition | models.HasIdCondition | models.HasVectorCondition | models.NestedCondition | models.Filter] = [
            models.FieldCondition(
                key=key,
                match=models.MatchValue(value=value),
            )
            for key, value in payload_filter.items()
        ]
        self._client.delete(
            collection_name,
            models.Filter(must=must_conditions),
            wait=True,
        )

    def switch_alias(self, alias_name: str, collection_name: str) -> None:
        operations: list[models.CreateAliasOperation | models.DeleteAliasOperation] = []
        current_collection = self.resolve_alias(alias_name)
        if current_collection is not None:
            operations.append(
                models.DeleteAliasOperation(
                    delete_alias=models.DeleteAlias(alias_name=alias_name)
                )
            )
        operations.append(
            models.CreateAliasOperation(
                create_alias=models.CreateAlias(
                    collection_name=collection_name,
                    alias_name=alias_name,
                )
            )
        )
        self._client.update_collection_aliases(operations)

    def resolve_alias(self, alias_name: str) -> str | None:
        aliases = self._client.get_aliases().aliases
        for alias in aliases:
            if alias.alias_name == alias_name:
                return alias.collection_name
        return None

    def delete_collection(self, collection_name: str) -> bool:
        """Delete a Qdrant collection. Returns True if the collection existed and was deleted."""
        if not self._client.collection_exists(collection_name):
            return False
        return self._client.delete_collection(collection_name)

    def list_record_version_ids(self, collection_or_alias: str) -> tuple[str, ...]:
        records: list[models.Record] = []
        offset = None
        while True:
            batch, offset = self._client.scroll(
                collection_or_alias,
                limit=1000,
                offset=offset,
                with_payload=["record_version_id"],
                with_vectors=False,
            )
            records.extend(batch)
            if offset is None:
                break
        return tuple(
            sorted({
                str(record.payload["record_version_id"])
                for record in records
                if record.payload is not None
            })
        )


def _point_id(point_id: str) -> str:
    return str(uuid5(NAMESPACE_URL, point_id))
