from __future__ import annotations

import json
from typing import TYPE_CHECKING, Literal

import httpx

from unibot.answering._openrouter_utils import extract_content as _extract_content

if TYPE_CHECKING:
    from unibot.answering.model_adapter import CitationAnswerDraft, CitationAnswerRequest

_CITATION_ANSWER_SCHEMA = {
    "type": "object",
    "required": ["status", "answer_text", "claims", "warnings"],
    "properties": {
        "status": {"type":"string","enum":["answered","abstained"],"description":"Whether the query was answered or abstained."},
        "answer_text": {"type":"string","description":"The full answer in Markdown (headings, bold, bullets) with inline citation markers such as [1]."},
        "claims": {
            "type":"array",
            "items":{
                "type":"object",
                "required":["text","citation_ids"],
                "properties":{
                    "text":{"type":"string"},
                    "citation_ids":{"type":"array","items":{"type":"string"}}
                },
                "additionalProperties":False,
            },
        },
        "warnings":{"type":"array","items":{"type":"string"}},
    },
    "additionalProperties":False,
}

class OpenRouterCitationAnswerModel:
    def __init__(
        self, *,
        api_key:str,
        model: str = "anthropic/claude-sonnet-4",
        base_url:str="https://openrouter.ai/api/v1/chat/completions",
        timeout:float=30.0,
        app_name:str="UniBot",
        client:httpx.Client|None=None,
        async_client:httpx.AsyncClient|None=None,
    )->None:
        self._api_key=api_key
        self._model=model
        self._base_url=base_url
        self._timeout=timeout
        self._app_name=app_name
        self._client=client
        self._async_client=async_client

    def _build_request_kwargs(self, request:"CitationAnswerRequest")->dict:
        return {
            "headers":{
                "Authorization":f"Bearer {self._api_key}",
                "Content-Type":"application/json",
                "X-OpenRouter-Title":self._app_name,
            },
            "json":{
                "model":self._model,
                "messages":[{"role":"user","content":request.prompt}],
                "temperature":0.3,
                "max_tokens":300,
                "response_format":{
                    "type":"json_schema",
                    "json_schema":{
                        "name":"citation_answer",
                        "strict":True,
                        "schema":_CITATION_ANSWER_SCHEMA,
                    },
                },
            },
            "timeout":self._timeout,
        }

    @staticmethod
    def _parse_response(payload:dict)->"CitationAnswerDraft":
        from unibot.answering.model_adapter import CitationAnswerDraft, GeneratedClaim
        print("="*100)
        print("FULL OPENROUTER RESPONSE")
        print(json.dumps(payload, indent=2))
        print("="*100)
        draft_text=_extract_content(payload)
        draft_text = draft_text.strip()

        if draft_text.startswith("```json"):
         draft_text = draft_text[7:]

        if draft_text.endswith("```"):
         draft_text = draft_text[:-3]

        draft_text = draft_text.strip()
        print("RAW CONTENT")
        print(repr(draft_text))
        print("="*100)
        draft_payload=json.loads(draft_text)
        raw_status=str(draft_payload["status"])
        if raw_status not in {"answered","abstained"}:
            raise ValueError(f"Unsupported answer status: {raw_status}")
        status:Literal["answered","abstained"]="answered" if raw_status=="answered" else "abstained"
        return CitationAnswerDraft(
            status=status,
            answer_text=str(draft_payload["answer_text"]),
            claims=tuple(
                GeneratedClaim(
                    text=str(claim["text"]),
                    citation_ids=tuple(str(cid) for cid in claim.get("citation_ids", [])),
                ) for claim in draft_payload.get("claims", [])
            ),
            warnings=tuple(str(w) for w in draft_payload.get("warnings", [])),
        )

    def generate(self, request:"CitationAnswerRequest")->"CitationAnswerDraft":
        post=self._client.post if self._client is not None else httpx.post
        response=post(self._base_url, **self._build_request_kwargs(request))
        response.raise_for_status()
        return self._parse_response(response.json())

    async def async_generate(self, request:"CitationAnswerRequest")->"CitationAnswerDraft":
        print(">>> async_generate ENTERED <<<")
        kwargs=self._build_request_kwargs(request)
        print("="*80)
        print(json.dumps(kwargs["json"], indent=2))
        print("="*80)
        if self._async_client is not None:
            response=await self._async_client.post(self._base_url, **kwargs)
        else:
            async with httpx.AsyncClient() as client:
                response=await client.post(self._base_url, **kwargs)
        print("POST COMPLETED")
        if response.status_code>=400:
            print("="*80)
            print("STATUS:", response.status_code)
            print("BODY:")
            print(response.text)
            print("="*80)
        response.raise_for_status()
        return self._parse_response(response.json())
