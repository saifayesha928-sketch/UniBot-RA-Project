from __future__ import annotations

import hashlib
import json
from typing import Callable, Literal

import httpx
import structlog

from unibot.answering._openrouter_utils import extract_content as _extract_content

logger = structlog.get_logger(__name__)

# Prompt used to situate each chunk within its parent document.
_CONTEXT_PROMPT = """\
<document>
{document}
</document>
Here is the chunk we want to situate within the whole document
<chunk>
{chunk}
</chunk>
Please give a short succinct context to situate this chunk within the overall document \
for the purposes of improving search retrieval of the chunk. \
Answer only with the succinct context and nothing else."""

# Stable instruction text used before and after the document block.
# Kept separate so that the document block can carry cache_control independently.
_INSTRUCTION_PREFIX = (
    "You will be given a document and a chunk from that document. "
    "Give a short succinct context to situate the chunk within the overall document "
    "for the purposes of improving search retrieval of the chunk. "
    "Answer only with the succinct context and nothing else."
)

_CHUNK_INSTRUCTION = (
    "Here is the chunk we want to situate within the whole document:"
)

ModelFunc = Callable[[str, str], str]


def context_prompt_hash(*, cache_ttl: Literal["5m", "1h"] = "5m") -> str:
    """Return a stable hash of the prompt template used for contextual retrieval.

    Changes when the static prompt/request shape changes, which should invalidate
    cached results.
    """
    template_request = build_openrouter_context_request(
        model="__MODEL__",
        document_text="__DOCUMENT__",
        chunk_text="__CHUNK__",
        cache_ttl=cache_ttl,
    )
    template_text = json.dumps(template_request, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(template_text.encode("utf-8")).hexdigest()


def build_openrouter_context_request(
    *,
    model: str,
    document_text: str,
    chunk_text: str,
    cache_ttl: Literal["5m", "1h"] = "5m",
) -> dict[str, object]:
    """Build an OpenRouter chat completion request with explicit cache breakpoints.

    Uses a content-block array so the document block carries a cache_control
    annotation while the chunk text remains outside the cached prefix.
    """
    cache_control: dict[str, str] = {"type": "ephemeral"}
    if cache_ttl == "1h":
        cache_control["ttl"] = "1h"

    content_blocks: list[dict[str, object]] = [
        {
            "type": "text",
            "text": _INSTRUCTION_PREFIX,
        },
        {
            "type": "text",
            "text": f"<document>\n{document_text}\n</document>",
            "cache_control": cache_control,
        },
        {
            "type": "text",
            "text": (
                f"{_CHUNK_INSTRUCTION}\n"
                f"<chunk>\n{chunk_text}\n</chunk>"
            ),
        },
    ]

    return {
        "model": model,
        "messages": [{"role": "user", "content": content_blocks}],
        "temperature": 0,
        "max_tokens": 200,
    }


# ---------------------------------------------------------------------------
# Existing sync helpers (preserved for backward compatibility)
# ---------------------------------------------------------------------------


def generate_chunk_context(
    document_text: str,
    chunk_text: str,
    *,
    model_func: ModelFunc,
) -> str:
    return model_func(document_text, chunk_text)


def contextualize_chunk(
    document_text: str,
    chunk_text: str,
    *,
    model_func: ModelFunc,
) -> str:
    try:
        context = generate_chunk_context(
            document_text, chunk_text, model_func=model_func,
        )
        if context and context.strip():
            return f"{context.strip()}\n\n{chunk_text}"
    except Exception:
        logger.warning(
            "contextual_retrieval.model_failed",
            exc_info=True,
        )
    return chunk_text


def contextualize_chunks_batch(
    chunks_data: list[tuple[str, str]],
    *,
    model_func: ModelFunc,
) -> list[str]:
    return [
        contextualize_chunk(doc, chunk, model_func=model_func)
        for doc, chunk in chunks_data
    ]


def build_context_prompt(document_text: str, chunk_text: str) -> str:
    return _CONTEXT_PROMPT.format(document=document_text, chunk=chunk_text)


def create_openrouter_model_func(
    *,
    api_key: str,
    model: str = "anthropic/claude-haiku-4-5-20251001",
    base_url: str = "https://openrouter.ai/api/v1/chat/completions",
    timeout: float = 30.0,
    app_name: str = "UniBot",
    client: httpx.Client | None = None,
) -> ModelFunc:
    """Create a model function using OpenRouter API.

    Same HTTP pattern as the OpenRouter answer model adapter
    (openrouter_adapter.py:74-119).
    """

    def _call(document_text: str, chunk_text: str) -> str:
        prompt = build_context_prompt(document_text, chunk_text)
        post = client.post if client is not None else httpx.post
        response = post(
            base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "X-OpenRouter-Title": app_name,
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 200,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        return _extract_content(response.json())

    return _call


# ---------------------------------------------------------------------------
# Async transport for the new contextualization service
# ---------------------------------------------------------------------------


async def call_openrouter_context_async(
    *,
    document_text: str,
    chunk_text: str,
    model: str,
    base_url: str,
    api_key: str,
    app_name: str,
    timeout: float = 30.0,
    cache_ttl: Literal["5m", "1h"] = "5m",
    client: httpx.AsyncClient | None = None,
) -> str:
    """Call OpenRouter asynchronously with explicit cache breakpoints.

    When *client* is provided it is used directly (caller owns its lifecycle).
    Otherwise a fresh ``AsyncClient`` is created and closed per call.
    """
    request_body = build_openrouter_context_request(
        model=model,
        document_text=document_text,
        chunk_text=chunk_text,
        cache_ttl=cache_ttl,
    )
    # Provider routing for resilient batch processing — kept outside the
    # request builder so it does not affect the prompt hash used for caching.
    request_body["provider"] = {"allow_fallbacks": True}
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-OpenRouter-Title": app_name,
    }

    if client is not None:
        response = await client.post(
            base_url, headers=headers, json=request_body, timeout=timeout,
        )
        response.raise_for_status()
        return _extract_content(response.json())

    async with httpx.AsyncClient() as internal_client:
        response = await internal_client.post(
            base_url, headers=headers, json=request_body, timeout=timeout,
        )
        response.raise_for_status()
        return _extract_content(response.json())
