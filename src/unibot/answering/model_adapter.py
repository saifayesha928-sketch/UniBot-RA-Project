from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

import httpx

from unibot.settings import resolve_answer_model_backend

if TYPE_CHECKING:
    from unibot.answering.service import Citation
    from unibot.retrieval.service import RetrievedEvidence


@dataclass(frozen=True, slots=True)
class GeneratedClaim:
    text: str
    citation_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CitationAnswerDraft:
    status: Literal["answered", "abstained"]
    answer_text: str
    claims: tuple[GeneratedClaim, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CitationAnswerRequest:
    query_text: str
    evidence: tuple["RetrievedEvidence", ...]
    citations: tuple["Citation", ...]
    prompt: str


class CitationAnswerModel(Protocol):
    def generate(self, request: CitationAnswerRequest) -> CitationAnswerDraft: ...


class FallbackCitationAnswerModel:
    """Tries the primary answer model, falls back on transient errors."""

    def __init__(
        self,
        *,
        primary: CitationAnswerModel,
        fallback: CitationAnswerModel,
    ) -> None:
        self._primary = primary
        self._fallback = fallback

    def generate(self, request: CitationAnswerRequest) -> CitationAnswerDraft:
        import structlog

        logger = structlog.get_logger(__name__)
        try:
            result = self._primary.generate(request)
            logger.info("answer_model.provider_used", provider="primary")
            return result
        except (httpx.TimeoutException, httpx.HTTPError, RuntimeError) as exc:
            logger.warning(
                "answer_model.primary_failed_using_fallback",
                exc_type=type(exc).__name__,
            )
            result = self._fallback.generate(request)
            logger.info("answer_model.provider_used", provider="fallback")
            return result

    async def async_generate(
        self, request: CitationAnswerRequest
    ) -> CitationAnswerDraft:
        import structlog

        logger = structlog.get_logger(__name__)
        primary_async = getattr(self._primary, "async_generate", None)
        try:
            if primary_async is not None:
                result: CitationAnswerDraft = await primary_async(request)
            else:
                result = await asyncio.to_thread(self._primary.generate, request)
            logger.info("answer_model.provider_used", provider="primary")
            return result
        except (httpx.TimeoutException, httpx.HTTPError, RuntimeError) as exc:
            logger.warning(
                "answer_model.primary_failed_using_fallback",
                exc_type=type(exc).__name__,
            )
            fallback_async = getattr(self._fallback, "async_generate", None)
            if fallback_async is not None:
                result = await fallback_async(request)
            else:
                result = await asyncio.to_thread(self._fallback.generate, request)
            logger.info("answer_model.provider_used", provider="fallback")
            return result


class CohereCitationAnswerModel:
    def __init__(
        self,
        *,
        api_key: str,
        model: str = "command-a-03-2025",
        base_url: str = "https://api.cohere.com/v2/chat",
        timeout: float = 30.0,
        client: httpx.Client | None = None,
        async_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        self._timeout = timeout
        self._client = client
        self._async_client = async_client

    def _build_request_kwargs(self, request: CitationAnswerRequest) -> dict:
        return {
            "headers": {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            "json": {
                "model": self._model,
                "messages": [{"role": "user", "content": request.prompt}],
                "response_format": {
                    "type": "json_object",
                    "json_schema": {
                        "type": "object",
                        "required": ["status", "answer_text", "claims", "warnings"],
                        "properties": {
                            "status": {"type": "string"},
                            "answer_text": {"type": "string"},
                            "claims": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "required": ["text", "citation_ids"],
                                    "properties": {
                                        "text": {"type": "string"},
                                        "citation_ids": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                    },
                                },
                            },
                            "warnings": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                    },
                },
            },
            "timeout": self._timeout,
        }

    @staticmethod
    def _parse_response(payload: dict) -> CitationAnswerDraft:
        draft_payload = json.loads(_extract_text_response(payload))
        raw_status = str(draft_payload["status"])
        if raw_status not in {"answered", "abstained"}:
            raise ValueError(f"Unsupported answer status: {raw_status}")
        status: Literal["answered", "abstained"] = (
            "answered" if raw_status == "answered" else "abstained"
        )
        return CitationAnswerDraft(
            status=status,
            answer_text=str(draft_payload["answer_text"]),
            claims=tuple(
                GeneratedClaim(
                    text=str(claim["text"]),
                    citation_ids=tuple(
                        str(item) for item in claim.get("citation_ids", [])
                    ),
                )
                for claim in draft_payload.get("claims", [])
            ),
            warnings=tuple(str(item) for item in draft_payload.get("warnings", [])),
        )

    def generate(self, request: CitationAnswerRequest) -> CitationAnswerDraft:
        post = self._client.post if self._client is not None else httpx.post
        response = post(self._base_url, **self._build_request_kwargs(request))
        response.raise_for_status()
        return self._parse_response(response.json())

    async def async_generate(
        self, request: CitationAnswerRequest
    ) -> CitationAnswerDraft:
        if self._async_client is not None:
            response = await self._async_client.post(
                self._base_url, **self._build_request_kwargs(request)
            )
        else:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self._base_url, **self._build_request_kwargs(request)
                )
        response.raise_for_status()
        return self._parse_response(response.json())


def create_answer_model(
    *,
    settings: object | None = None,
    client: httpx.Client | None = None,
    fallback_client: httpx.Client | None = None,
    async_client: httpx.AsyncClient | None = None,
    async_fallback_client: httpx.AsyncClient | None = None,
) -> CitationAnswerModel:
    if settings is None:
        from unibot.answering.service import DeterministicCitationAnswerModel

        return DeterministicCitationAnswerModel()

    primary = _build_single_answer_model(
        backend=resolve_answer_model_backend(settings),
        settings=settings,
        client=client,
        async_client=async_client,
    )

    fallback_backend: str | None = getattr(
        settings, "answer_model_fallback_backend", None
    )
    if fallback_backend is not None:
        fallback = _build_single_answer_model(
            backend=fallback_backend,
            settings=settings,
            client=fallback_client,
            async_client=async_fallback_client,
        )
        return FallbackCitationAnswerModel(primary=primary, fallback=fallback)

    return primary


def _build_single_answer_model(
    *,
    backend: str,
    settings: object,
    client: httpx.Client | None = None,
    async_client: httpx.AsyncClient | None = None,
) -> CitationAnswerModel:
    openrouter_api_key: str | None = getattr(settings, "openrouter_api_key", None)
    if backend == "openrouter":
        if openrouter_api_key is None:
            raise ValueError(
                "UNIBOT_OPENROUTER_API_KEY is required when using OpenRouter answering"
            )
        from unibot.answering.openrouter_adapter import OpenRouterCitationAnswerModel

        return OpenRouterCitationAnswerModel(
            api_key=openrouter_api_key,
            model=getattr(
                settings, "openrouter_model", "anthropic/claude-sonnet-4-20250514"
            ),
            base_url=getattr(
                settings,
                "openrouter_base_url",
                "https://openrouter.ai/api/v1/chat/completions",
            ),
            timeout=float(getattr(settings, "openrouter_timeout_seconds", 30.0)),
            app_name=getattr(settings, "openrouter_app_name", "UniBot"),
            client=client,
            async_client=async_client,
        )

    cohere_api_key: str | None = getattr(settings, "cohere_api_key", None)
    if backend == "cohere":
        if cohere_api_key is None:
            raise ValueError(
                "UNIBOT_COHERE_API_KEY is required when using Cohere answering"
            )
        return CohereCitationAnswerModel(
            api_key=cohere_api_key,
            model=getattr(settings, "cohere_chat_model", "command-a-03-2025"),
            base_url=getattr(
                settings,
                "cohere_chat_base_url",
                "https://api.cohere.com/v2/chat",
            ),
            timeout=float(getattr(settings, "cohere_timeout_seconds", 30.0)),
            client=client,
            async_client=async_client,
        )

    from unibot.answering.service import DeterministicCitationAnswerModel

    return DeterministicCitationAnswerModel()


def _extract_text_response(payload: dict[str, object]) -> str:
    message = payload.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text")
                    if isinstance(text, str):
                        return text
    text = payload.get("text")
    if isinstance(text, str):
        return text
    raise ValueError("Cohere chat response did not include text content")
