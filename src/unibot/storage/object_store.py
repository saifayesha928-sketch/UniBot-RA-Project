from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

try:
    import boto3  # type: ignore[import-untyped]
except ModuleNotFoundError:
    class _MissingBoto3Module:
        def client(self, *_: object, **__: object) -> object:
            raise ModuleNotFoundError("boto3 is required for S3ObjectStore")

    boto3 = _MissingBoto3Module()


@dataclass(frozen=True, slots=True)
class StoredObject:
    storage_uri: str
    backend: str


class ObjectStore(Protocol):
    def put(self, key: str, content: bytes) -> StoredObject:
        """Persist raw artifact bytes and return its storage reference."""

    def get(self, key: str) -> bytes | None:
        """Retrieve raw artifact bytes by key, or None if not found."""


class LocalObjectStore:
    def __init__(self, *, base_path: str | Path) -> None:

        self._base_path = Path(base_path)
        self._base_path.mkdir(parents=True, exist_ok=True)
        self._resolved_base_path = self._base_path.resolve()

    def put(self, key: str, content: bytes) -> StoredObject:
        destination = self._resolve_key_path(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)

        return StoredObject(
            storage_uri=str(destination),
            backend="local",
        )

    def get(self, key: str) -> bytes | None:
        path = self._resolve_key_path(key)
        if not path.exists():
            return None
        return path.read_bytes()

    def _resolve_key_path(self, key: str) -> Path:
        resolved = (self._base_path / key).resolve()
        if (
            resolved != self._resolved_base_path
            and self._resolved_base_path not in resolved.parents
        ):
            raise ValueError("object key escapes base path")
        return resolved


class S3ObjectStore:
    def __init__(
        self,
        *,
        bucket: str,
        prefix: str = "",
        endpoint_url: str | None = None,
        region_name: str | None = None,
    ) -> None:
        self._bucket = bucket
        self._prefix = prefix.strip("/")
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            region_name=region_name,
        )

    def put(self, key: str, content: bytes) -> StoredObject:
        object_key = "/".join(part for part in (self._prefix, key) if part)
        self._client.put_object(Bucket=self._bucket, Key=object_key, Body=content)
        return StoredObject(
            storage_uri=f"s3://{self._bucket}/{object_key}",
            backend="s3",
        )

    def get(self, key: str) -> bytes | None:
        object_key = "/".join(part for part in (self._prefix, key) if part)
        no_such_key = getattr(getattr(self._client, "exceptions", None), "NoSuchKey", None)
        try:
            response = self._client.get_object(Bucket=self._bucket, Key=object_key)
        except Exception as exc:
            if no_such_key is not None and isinstance(exc, no_such_key):
                return None
            raise
        body: bytes = response["Body"].read()
        return body
