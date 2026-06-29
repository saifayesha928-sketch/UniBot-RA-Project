from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Protocol

import httpx
import pybreaker

logger = logging.getLogger(__name__)

_REWRITE_PROMPT = """\
You are a query rewriter for the university knowledge base.

Rewrite the following user query into clear, specific university domain language.
Rules:
- Preserve the user's intent exactly.
- Use domain-specific terms: "tuition fee" not "cost", "admission" not "getting in", \
"faculty" not "teachers", "CGPA" not "grades", "scholarship" not "financial help".
- Keep it as a single concise query.
- If the query is already well-formed, return it unchanged.
- Return ONLY the rewritten query, nothing else.

User query: {query}"""


@dataclass(frozen=True, slots=True)
class RewriteResult:
    original_query: str
    rewritten_query: str


class QueryRewriter(Protocol):
    def rewrite(self, query_text: str) -> RewriteResult: ...


class PassthroughQueryRewriter:
    def rewrite(self, query_text: str) -> RewriteResult:
        return RewriteResult(
            original_query=query_text,
            rewritten_query=query_text,
        )


class DomainQueryRewriter:
    def __init__(
        self,
        *,
        api_key: str,
        model: str = "anthropic/claude-haiku-4-5-20251001",
        base_url: str = "https://openrouter.ai/api/v1/chat/completions",
        timeout: float = 10.0,
        app_name: str = "UniBot",
        client: httpx.Client | None = None,
        min_length_ratio: float = 0.3,
        max_length_ratio: float = 3.0,
        provider_order: tuple[str, ...] = (),
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        self._timeout = timeout
        self._app_name = app_name
        self._client = client
        self._min_length_ratio = min_length_ratio
        self._max_length_ratio = max_length_ratio
        self._provider_order = provider_order
        self._breaker = pybreaker.CircuitBreaker(
            fail_max=3,
            reset_timeout=30,
            exclude=[ValueError],
            name="query_rewriter",
        )

    def rewrite(self, query_text: str) -> RewriteResult:
        try:
            rewritten = self._breaker.call(self._call_api, query_text).strip()
            if not self._is_valid_rewrite(query_text, rewritten):
                logger.info(
                    "query_rewriter.rewrite_rejected",
                    extra={
                        "reason": "failed_quality_gate",
                        "original_len": len(query_text),
                        "rewritten_len": len(rewritten),
                    },
                )
                return RewriteResult(
                    original_query=query_text,
                    rewritten_query=query_text,
                )
            return RewriteResult(
                original_query=query_text,
                rewritten_query=rewritten,
            )
        except pybreaker.CircuitBreakerError:
            logger.info("query_rewriter.circuit_open")
            return RewriteResult(
                original_query=query_text,
                rewritten_query=query_text,
            )
        except Exception:
            logger.warning(
                "query_rewriter.api_failed",
                exc_info=True,
            )
            return RewriteResult(
                original_query=query_text,
                rewritten_query=query_text,
            )

    def _is_valid_rewrite(self, original: str, rewritten: str) -> bool:
        if not rewritten:
            return False
        if "\n" in rewritten:
            return False
        ratio = len(rewritten) / max(len(original), 1)
        if ratio < self._min_length_ratio or ratio > self._max_length_ratio:
            return False
        return True

    async def async_rewrite(self, query_text: str) -> RewriteResult:
        try:
            rewritten = (
                await asyncio.to_thread(self._breaker.call, self._call_api, query_text)
            ).strip()
            if not self._is_valid_rewrite(query_text, rewritten):
                logger.info(
                    "query_rewriter.rewrite_rejected",
                    extra={
                        "reason": "failed_quality_gate",
                        "original_len": len(query_text),
                        "rewritten_len": len(rewritten),
                    },
                )
                return RewriteResult(
                    original_query=query_text,
                    rewritten_query=query_text,
                )
            return RewriteResult(
                original_query=query_text,
                rewritten_query=rewritten,
            )
        except pybreaker.CircuitBreakerError:
            logger.info("query_rewriter.circuit_open")
            return RewriteResult(
                original_query=query_text,
                rewritten_query=query_text,
            )
        except Exception:
            logger.warning(
                "query_rewriter.async_api_failed",
                exc_info=True,
            )
            return RewriteResult(
                original_query=query_text,
                rewritten_query=query_text,
            )

    def _build_request_kwargs(self, query_text: str) -> dict:
        json_body: dict = {
            "model": self._model,
            "messages": [
                {
                    "role": "user",
                    "content": _REWRITE_PROMPT.format(query=query_text),
                }
            ],
            "temperature": 0,
            "max_tokens": 100,
        }
        if self._provider_order:
            json_body["provider"] = {
                "order": list(self._provider_order),
                "allow_fallbacks": True,
            }
        return {
            "headers": {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "X-OpenRouter-Title": self._app_name,
            },
            "json": json_body,
            "timeout": self._timeout,
        }

    @staticmethod
    def _parse_rewrite_response(payload: dict, query_text: str) -> str:
        choices = payload.get("choices", [])
        if not choices:
            return query_text
        return str(choices[0].get("message", {}).get("content", query_text))

    def _call_api(self, query_text: str) -> str:
        post = self._client.post if self._client is not None else httpx.post
        response = post(self._base_url, **self._build_request_kwargs(query_text))
        response.raise_for_status()
        return self._parse_rewrite_response(response.json(), query_text)

def create_query_rewriter(
    *,
    settings: object | None = None,
    client: httpx.Client | None = None,
) -> QueryRewriter:
    if settings is None:
        return PassthroughQueryRewriter()

    enabled = getattr(settings, "query_rewriter_enabled", False)
    if not enabled:
        return PassthroughQueryRewriter()

    api_key = getattr(settings, "openrouter_api_key", None)
    if api_key is None:
        logger.warning("Query rewriter enabled but no OpenRouter API key; using passthrough.")
        return PassthroughQueryRewriter()

    raw_order = getattr(settings, "query_rewriter_provider_order", "")
    provider_order = tuple(
        s.strip() for s in str(raw_order).split(",") if s.strip()
    )

    return DomainQueryRewriter(
        api_key=api_key,
        model=getattr(settings, "query_rewriter_model", "anthropic/claude-haiku-4-5-20251001"),
        base_url=getattr(
            settings,
            "openrouter_base_url",
            "https://openrouter.ai/api/v1/chat/completions",
        ),
        timeout=float(getattr(settings, "query_rewriter_timeout_seconds", 10.0)),
        app_name=getattr(settings, "openrouter_app_name", "UniBot"),
        client=client,
        provider_order=provider_order,
    )
