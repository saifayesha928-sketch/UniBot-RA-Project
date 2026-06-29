from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

import httpx

from unibot.settings import resolve_reranker_backend

logger = logging.getLogger(__name__)

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True, slots=True)
class RerankCandidate:
    record_version_id: str
    text: str


@dataclass(frozen=True, slots=True)
class RerankScore:
    record_version_id: str
    score: float


class Reranker(Protocol):
    def rerank(
        self,
        query_text: str,
        candidates: tuple[RerankCandidate, ...] | list[RerankCandidate],
    ) -> tuple[RerankScore, ...]:
        pass


class TokenOverlapReranker:
    def rerank(
        self,
        query_text: str,
        candidates: tuple[RerankCandidate, ...] | list[RerankCandidate],
    ) -> tuple[RerankScore, ...]:
        query_tokens = _tokenize(query_text)
        scores = []
        for candidate in candidates:
            candidate_tokens = _tokenize(candidate.text)
            overlap = len(query_tokens & candidate_tokens)
            normalization = len(query_tokens) or 1
            score = overlap / normalization
            if query_text.strip() and query_text.lower() in candidate.text.lower():
                score += 0.25
            scores.append(
                RerankScore(
                    record_version_id=candidate.record_version_id,
                    score=score,
                )
            )

        return tuple(
            sorted(scores, key=lambda item: (-item.score, item.record_version_id))
        )


def _tokenize(text: str) -> set[str]:
    return {token for token in _TOKEN_PATTERN.findall(text.lower()) if token}


class CohereReranker:
    def __init__(
        self,
        *,
        api_key: str,
        model: str = "rerank-v4.0-fast",
        base_url: str = "https://api.cohere.com/v2/rerank",
        timeout: float = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        self._timeout = timeout
        self._client = client

    def rerank(
        self,
        query_text: str,
        candidates: tuple[RerankCandidate, ...] | list[RerankCandidate],
    ) -> tuple[RerankScore, ...]:
        candidate_items = tuple(candidates)
        if not candidate_items:
            return ()

        try:
            return self._rerank_via_api(query_text, candidate_items)
        except (httpx.HTTPError, KeyError, IndexError, ValueError, TypeError):
            logger.warning(
                "cohere_reranker.api_failed, falling back to cross_encoder",
                exc_info=True,
            )
            return CrossEncoderReranker().rerank(query_text, candidate_items)

    def _rerank_via_api(
        self,
        query_text: str,
        candidate_items: tuple[RerankCandidate, ...],
    ) -> tuple[RerankScore, ...]:
        post = self._client.post if self._client is not None else httpx.post
        response = post(
            self._base_url,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self._model,
                "query": query_text,
                "documents": [candidate.text for candidate in candidate_items],
            },
            timeout=self._timeout,
        )
        response.raise_for_status()
        payload = response.json()
        return tuple(
            RerankScore(
                record_version_id=candidate_items[int(item["index"])].record_version_id,
                score=float(item["relevance_score"]),
            )
            for item in payload.get("results", [])
        )


@lru_cache(maxsize=1)
def _get_cross_encoder(model_name: str):  # type: ignore[no-untyped-def]
    from sentence_transformers import CrossEncoder

    logger.info("Loading cross-encoder model: %s", model_name)
    return CrossEncoder(model_name)


_DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    def __init__(self, *, model_name: str = _DEFAULT_CROSS_ENCODER_MODEL) -> None:
        self._model_name = model_name

    def rerank(
        self,
        query_text: str,
        candidates: tuple[RerankCandidate, ...] | list[RerankCandidate],
    ) -> tuple[RerankScore, ...]:
        candidate_items = tuple(candidates)
        if not candidate_items:
            return ()

        try:
            model = _get_cross_encoder(self._model_name)
        except Exception:
            logger.warning(
                "cross_encoder.load_failed, falling back to token overlap",
                exc_info=True,
            )
            return TokenOverlapReranker().rerank(query_text, candidate_items)

        pairs = [(query_text, c.text) for c in candidate_items]
        raw_scores = model.predict(pairs)

        scores = [
            RerankScore(
                record_version_id=candidate_items[i].record_version_id,
                score=float(raw_scores[i]),
            )
            for i in range(len(candidate_items))
        ]
        return tuple(sorted(scores, key=lambda s: (-s.score, s.record_version_id)))


def create_reranker(*, settings: object | None = None, client: httpx.Client | None = None) -> Reranker:
    if settings is None:
        return TokenOverlapReranker()

    backend = resolve_reranker_backend(settings)
    cohere_api_key: str | None = getattr(settings, "cohere_api_key", None)
    if backend == "cohere" and cohere_api_key is not None:
        return CohereReranker(
            api_key=cohere_api_key,
            model=getattr(settings, "cohere_rerank_model", "rerank-v4.0"),
            base_url=getattr(
                settings,
                "cohere_rerank_base_url",
                "https://api.cohere.com/v2/rerank",
            ),
            timeout=float(getattr(settings, "cohere_timeout_seconds", 30.0)),
            client=client,
        )
    if backend == "cohere" and cohere_api_key is None:
        # Unreachable via Settings (validator catches this at startup),
        # but reachable via plain object settings in tests/manual wiring.
        logger.warning(
            "Cohere reranker requested but no API key configured; "
            "falling back to cross_encoder reranker."
        )
        return CrossEncoderReranker()
    if backend == "cross_encoder":
        return CrossEncoderReranker(
            model_name=getattr(
                settings,
                "cross_encoder_model",
                _DEFAULT_CROSS_ENCODER_MODEL,
            ),
        )
    return TokenOverlapReranker()
