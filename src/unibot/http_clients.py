from __future__ import annotations

from dataclasses import dataclass

import httpx

from unibot.settings import (
    resolve_answer_model_backend,
    resolve_embedding_dense_backend,
    resolve_reranker_backend,
)


@dataclass(slots=True)
class ProviderHttpClients:
    cohere: httpx.Client | None = None
    openrouter: httpx.Client | None = None
    async_cohere: httpx.AsyncClient | None = None
    async_openrouter: httpx.AsyncClient | None = None

    def close(self) -> None:
        if self.cohere is not None:
            self.cohere.close()
        if self.openrouter is not None:
            self.openrouter.close()

    async def aclose(self) -> None:
        if self.async_cohere is not None:
            await self.async_cohere.aclose()
        if self.async_openrouter is not None:
            await self.async_openrouter.aclose()
        self.close()


def build_provider_http_clients(settings: object) -> ProviderHttpClients:
    fallback_backend = getattr(settings, "answer_model_fallback_backend", None)
    use_cohere = (
        resolve_embedding_dense_backend(settings) == "cohere"
        or resolve_reranker_backend(settings) == "cohere"
        or resolve_answer_model_backend(settings) == "cohere"
        or fallback_backend == "cohere"
    )
    rewriter_needs_openrouter = bool(
        getattr(settings, "query_rewriter_enabled", False)
    )
    use_openrouter = (
        resolve_answer_model_backend(settings) == "openrouter"
        or fallback_backend == "openrouter"
        or rewriter_needs_openrouter
    )
    return ProviderHttpClients(
        cohere=httpx.Client() if use_cohere else None,
        openrouter=httpx.Client() if use_openrouter else None,
        async_cohere=httpx.AsyncClient() if use_cohere else None,
        async_openrouter=httpx.AsyncClient() if use_openrouter else None,
    )
