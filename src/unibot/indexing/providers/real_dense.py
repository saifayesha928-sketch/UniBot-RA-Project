from __future__ import annotations

import time

import httpx
import structlog

logger = structlog.get_logger(__name__)

_MAX_RETRIES = 5
_RATE_LIMIT_BACKOFF_SECONDS = 65.0
_SERVER_ERROR_BACKOFF_SECONDS = 5.0
_BACKOFF_MULTIPLIER = 1.5
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_MAX_TEXTS_PER_REQUEST = 96
_INTER_BATCH_DELAY_SECONDS = 3.0


class CohereDenseEmbeddingProvider:
    def __init__(
        self,
        *,
        api_key: str,
        model: str = "embed-v4.0",
        base_url: str = "https://api.cohere.com/v2/embed",
        timeout: float = 90.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        self._timeout = timeout
        self._client = client

    def embed_dense(self, text: str) -> tuple[float, ...]:
        return self.embed_dense_document(text)

    def embed_dense_document(self, text: str) -> tuple[float, ...]:
        return self._embed(text, input_type="search_document")

    def embed_dense_query(self, text: str) -> tuple[float, ...]:
        return self._embed(text, input_type="search_query")

    def embed_dense_document_batch(
        self, texts: list[str] | tuple[str, ...]
    ) -> list[tuple[float, ...]]:
        return self._embed_batch(list(texts), input_type="search_document")

    def embed_dense_query_batch(
        self, texts: list[str] | tuple[str, ...]
    ) -> list[tuple[float, ...]]:
        return self._embed_batch(list(texts), input_type="search_query")

    def _embed(self, text: str, *, input_type: str) -> tuple[float, ...]:
        results = self._embed_batch([text], input_type=input_type)
        return results[0]

    def _embed_batch(
        self, texts: list[str], *, input_type: str
    ) -> list[tuple[float, ...]]:
        all_embeddings: list[tuple[float, ...]] = []

        for i, batch_start in enumerate(range(0, len(texts), _MAX_TEXTS_PER_REQUEST)):
            if i > 0:
                time.sleep(_INTER_BATCH_DELAY_SECONDS)
            batch = texts[batch_start : batch_start + _MAX_TEXTS_PER_REQUEST]
            batch_embeddings = self._call_api(batch, input_type=input_type)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def _call_api(
        self, texts: list[str], *, input_type: str
    ) -> list[tuple[float, ...]]:
        last_exception: Exception | None = None

        post = self._client.post if self._client is not None else httpx.post
        for attempt in range(_MAX_RETRIES + 1):
            response = post(
                self._base_url,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._model,
                    "texts": texts,
                    "input_type": input_type,
                    "embedding_types": ["float"],
                },
                timeout=self._timeout,
            )

            if response.status_code not in _RETRYABLE_STATUS_CODES:
                response.raise_for_status()
                break

            last_exception = httpx.HTTPStatusError(
                f"{response.status_code} {response.reason_phrase}",
                request=response.request,
                response=response,
            )

            if attempt == _MAX_RETRIES:
                raise last_exception

            retry_after = response.headers.get("retry-after")
            if retry_after:
                wait = float(retry_after)
            elif response.status_code == 429:
                wait = _RATE_LIMIT_BACKOFF_SECONDS * (_BACKOFF_MULTIPLIER ** attempt)
            else:
                wait = _SERVER_ERROR_BACKOFF_SECONDS * (_BACKOFF_MULTIPLIER ** attempt)

            logger.info(
                "cohere.rate_limited",
                status=response.status_code,
                attempt=attempt + 1,
                wait_seconds=round(wait, 1),
                batch_size=len(texts),
            )
            time.sleep(wait)
        else:
            if last_exception is not None:
                raise last_exception

        payload = response.json()
        embeddings = payload.get("embeddings", {})
        float_embeddings = embeddings.get("float", [])
        if not float_embeddings or len(float_embeddings) != len(texts):
            raise ValueError(
                f"Cohere embed response returned {len(float_embeddings)} embeddings "
                f"for {len(texts)} texts"
            )
        return [tuple(float(v) for v in emb) for emb in float_embeddings]
