from __future__ import annotations

import hashlib
from pathlib import PurePosixPath
from urllib.parse import urlsplit

from sqlalchemy.orm import Session

import structlog

from unibot.crawl.fetchers import FetchedArtifact, RawArtifactFetcher
from unibot.db.models import RawSnapshot
from unibot.storage.object_store import ObjectStore

logger = structlog.get_logger(__name__)


def _content_extension(content_type: str) -> str:
    normalized = content_type.lower()
    if "html" in normalized:
        return ".html"
    if "markdown" in normalized:
        return ".md"
    if "pdf" in normalized:
        return ".pdf"
    return ".bin"


def _storage_key(source_url: str, content_hash: str, content_type: str) -> str:
    parsed = urlsplit(source_url)
    hostname = parsed.hostname or "unknown-host"
    path = parsed.path.strip("/") or "index"
    safe_path = PurePosixPath(path)
    filename = f"{safe_path.name}-{content_hash[:12]}{_content_extension(content_type)}"
    parent = str(safe_path.parent)
    if parent == ".":
        parent = ""
    segments = [segment for segment in (hostname, parent, filename) if segment]
    return str(PurePosixPath(*segments))


class RawSnapshotStore:
    def __init__(self, *, session: Session, object_store: ObjectStore) -> None:
        self._session = session
        self._object_store = object_store

    def fetch_and_store(
        self,
        *,
        source_id: str,
        source_url: str,
        fetcher: RawArtifactFetcher,
        requires_browser: bool = False,
    ) -> RawSnapshot:
        artifact = fetcher.fetch(source_url, requires_browser=requires_browser)
        return self.store_snapshot(source_id=source_id, artifact=artifact)

    def store_snapshot(
        self,
        *,
        source_id: str,
        artifact: FetchedArtifact,
    ) -> RawSnapshot:
        content_hash = hashlib.sha256(artifact.content).hexdigest()
        stored_object = self._object_store.put(
            _storage_key(artifact.source_url, content_hash, artifact.content_type),
            artifact.content,
        )

        # Persist the crawl4ai raw markdown alongside the HTML snapshot.
        # This is supplementary — a failure here must not abort the primary snapshot.
        raw_markdown = artifact.metadata.get("markdown")
        if isinstance(raw_markdown, str) and raw_markdown.strip():
            try:
                md_key = _storage_key(artifact.source_url, content_hash, "text/markdown")
                self._object_store.put(md_key, raw_markdown.encode("utf-8"))
            except Exception:
                logger.warning(
                    "snapshot.markdown_sidecar_failed",
                    source_url=artifact.source_url,
                    exc_info=True,
                )

        snapshot = RawSnapshot(
            source_id=source_id,
            source_url=artifact.source_url,
            content_type=artifact.content_type,
            storage_uri=stored_object.storage_uri,
            storage_backend=stored_object.backend,
            page_content_hash=content_hash,
            http_status=artifact.http_status,
            etag=artifact.etag,
            last_modified=artifact.last_modified,
            fetch_metadata={
                "fetch_method": artifact.fetch_method,
                "requires_browser": artifact.requires_browser,
                "content_length": len(artifact.content),
                **{
                    key: value
                    for key, value in artifact.metadata.items()
                    if key != "parsed_document"
                },
            },
        )
        self._session.add(snapshot)
        self._session.flush()
        return snapshot
