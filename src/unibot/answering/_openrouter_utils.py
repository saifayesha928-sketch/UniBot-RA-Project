from __future__ import annotations


def extract_content(payload: dict[str, object]) -> str:
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message", {})
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content
    raise ValueError(
        "OpenRouter response did not include message content in choices[0]"
    )
