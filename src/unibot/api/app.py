from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI
from qdrant_client import QdrantClient
from sqlalchemy.orm import Session
import structlog

from unibot.api.pipeline_factory import build_query_runtime
from unibot.settings import (
    Settings,
    get_settings,
    resolve_grounding_verifier_backend,
    retrieval_quality_warning,
)

logger = structlog.get_logger(__name__)


def create_app(
    *,
    session_factory: Callable[[], Session] | None = None,
    qdrant_client: QdrantClient | None = None,
    close_sessions: bool = True,
    admin_api_key: str | None = None,
    enable_admin_auth: bool = True,
    settings: Settings | None = None,
) -> FastAPI:
    resolved_settings = settings if settings is not None else get_settings()

    owns_qdrant_client = False
    if qdrant_client is None:
        qdrant_url = getattr(resolved_settings, "qdrant_url", None)
        if qdrant_url is not None:
            qdrant_client = QdrantClient(
                url=str(qdrant_url),
                api_key=getattr(resolved_settings, "qdrant_api_key", None),
            )
            owns_qdrant_client = True

    runtime = build_query_runtime(resolved_settings)

    if resolve_grounding_verifier_backend(resolved_settings) == "lettucedetect":
        from unibot.answering.grounding import warm_detector

        from unibot.answering.grounding import _DEFAULT_MODEL_PATH

        grounding_model = getattr(
            resolved_settings, "grounding_model", _DEFAULT_MODEL_PATH,
        )
        warm_detector(grounding_model)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        yield
        if runtime.async_cleanup is not None:
            await runtime.async_cleanup()
        else:
            runtime.cleanup()
        if owns_qdrant_client and qdrant_client is not None:
            qdrant_client.close()

    app = FastAPI(title="UniBot API", lifespan=lifespan)
    app.state.session_factory = session_factory
    app.state.qdrant_client = qdrant_client
    app.state.close_sessions = close_sessions
    app.state.enable_admin_auth = enable_admin_auth
    if admin_api_key is None:
        admin_api_key = getattr(resolved_settings, "admin_api_key", None)
    app.state.admin_api_key = admin_api_key
    app.state.settings = resolved_settings
    app.state.embedding_provider = runtime.embedding_provider
    app.state.reranker = runtime.reranker
    app.state.answering_service = runtime.answering_service
    app.state.semantic_classifier = runtime.semantic_classifier
    app.state.query_rewriter = runtime.query_rewriter

    if warning := retrieval_quality_warning(resolved_settings):
        logger.warning("runtime.retrieval_quality", warning=warning)

    from unibot.api.routes.admin import router as admin_router
    from unibot.api.routes.query import router as query_router

    app.include_router(query_router)
    app.include_router(admin_router)
    return app

app = create_app()