from __future__ import annotations

import asyncio
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal, Protocol

import structlog

if TYPE_CHECKING:
    from unibot.db.models import ServingGeneration

from unibot.answering.citation_validation import validate_claim_citations
from unibot.answering.grounding import (
    GroundingResult,
    GroundingVerifier,
    PassthroughGroundingVerifier,
)
from unibot.answering.model_adapter import (
    CitationAnswerDraft,
    CitationAnswerModel,
    CitationAnswerRequest,
    GeneratedClaim,
)
from unibot.answering.prompting import build_citation_answer_prompt
from unibot.retrieval.filters import AMBIGUOUS_YEAR_CONFIDENCE
from unibot.retrieval.service import RetrievedEvidence
from unibot.verify.source_class_currentness import CYCLE_AWARE_TYPES

logger = structlog.get_logger(__name__)

_DEGREE_PREFIXES = re.compile(r"^(bs|ms|phd|mba|emba|be|me)[-_]", re.IGNORECASE)
_FEE_QUERY_TERMS = re.compile(
    r"\b(fee|fees|tuition|cost|costs|charge|charges|price|prices)\b",
    re.IGNORECASE,
)

_NUMERIC_QUERY = re.compile(
    r"\b(rate|percentage|ratio|number of"
    r"|placement rate|employment rate|acceptance rate"
    r"|average|median|total number)\b",
    re.IGNORECASE,
)

_MONETARY_AMOUNT_QUERY = re.compile(r"\bhow much\b", re.IGNORECASE)

_NUMERIC_CONTENT = re.compile(r"\d")


def _query_expects_numeric_answer(query_text: str) -> bool:
    return _NUMERIC_QUERY.search(query_text) is not None or (
        _MONETARY_AMOUNT_QUERY.search(query_text) is not None
        and _FEE_QUERY_TERMS.search(query_text) is not None
    )


def _evidence_has_numeric_content(
    evidence: tuple["RetrievedEvidence", ...],
) -> bool:
    return any(_NUMERIC_CONTENT.search(item.content) for item in evidence)


_HIGH_RISK_RECORD_TYPES: frozenset[str] = frozenset(
    {
        "program_fee_schedule",
        "admissions_cycle",
        "scholarship",
        "merit_list",
    }
)

_HIGH_RISK_SOURCE_CLASSES: frozenset[str] = frozenset(
    {
        "policy",
        "admissions_cycle",
        "scholarship",
        "merit_list",
    }
)


def _is_program_fee_query(query_text: str) -> bool:
    return _FEE_QUERY_TERMS.search(query_text) is not None


def _program_ambiguity_levels(evidence: tuple | list) -> tuple[str, ...]:
    fee_items = [
        item
        for item in evidence
        if getattr(item, "record_type", None) == "program_fee_schedule"
    ]
    if len(fee_items) < 2:
        return ()

    programs: dict[str, set[str]] = defaultdict(set)
    for item in fee_items:
        key = item.dedupe_key
        parts = key.split(":")
        if len(parts) < 3:
            continue
        slug = parts[-2]
        match = _DEGREE_PREFIXES.match(slug)
        if not match:
            continue
        prefix = match.group(1).lower()
        base = _DEGREE_PREFIXES.sub("", slug)
        programs[base].add(prefix)

    mixed_levels: set[str] = set()
    for prefixes in programs.values():
        if len(prefixes) > 1:
            mixed_levels.update(prefixes)
    return tuple(sorted(level.upper() for level in mixed_levels))


def _has_program_ambiguity(evidence: tuple | list) -> bool:
    """Detect when fee evidence mixes different degree levels for similar program names."""
    return bool(_program_ambiguity_levels(evidence))


@dataclass(frozen=True, slots=True)
class Citation:
    citation_id: str
    record_version_id: str
    source_url: str
    source_locator: str
    chunk_id: str = ""
    chunk_index: int = 0
    chunk_count: int = 1
    content: str = ""


@dataclass(frozen=True, slots=True)
class MaterialClaim:
    text: str
    citation_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AnswerResult:
    status: Literal["answered", "abstained"]
    answer_text: str
    claims: tuple[MaterialClaim, ...]
    citations: tuple[Citation, ...]
    warnings: tuple[str, ...]
    prompt: str


@dataclass(frozen=True, slots=True)
class _ValidatedContext:
    evidence_items: tuple[RetrievedEvidence, ...]
    freshness_warnings: tuple[str, ...]
    grounding_mode: Literal["full", "skip"]
    prompt: str
    citations: tuple[Citation, ...]
    prompt_build_ms: float


class AnsweringService:
    def __init__(
        self,
        *,
        answer_model: CitationAnswerModel | None = None,
        grounding_verifier: GroundingVerifier | None = None,
        grounding_skip_low_risk: bool = False,
        grounding_confidence_threshold: float = 0.5,
    ) -> None:
        self._answer_model = answer_model or DeterministicCitationAnswerModel()
        self._grounding_verifier = grounding_verifier or PassthroughGroundingVerifier()
        self._grounding_skip_low_risk = grounding_skip_low_risk
        self._grounding_confidence_threshold = grounding_confidence_threshold

    @staticmethod
    def _log_and_return(
        result: AnswerResult,
        *,
        evidence_items: tuple[RetrievedEvidence, ...],
        grounding_score: float | None = None,
    ) -> AnswerResult:
        logger.info(
            "answering.result",
            status=result.status,
            evidence_count=len(evidence_items),
            evidence_record_types=sorted({item.record_type for item in evidence_items})
            if evidence_items
            else [],
            evidence_scores=sorted(
                (item.score for item in evidence_items), reverse=True
            )[:5]
            if evidence_items
            else [],
            citation_count=len(result.citations),
            warning_count=len(result.warnings),
            grounding_score=grounding_score,
        )
        return result

    def _resolve_grounding_mode(
        self,
        evidence: tuple[RetrievedEvidence, ...],
    ) -> Literal["full", "skip"]:
        if not self._grounding_skip_low_risk:
            return "full"
        if not evidence:
            return "full"
        for item in evidence:
            if item.record_type in _HIGH_RISK_RECORD_TYPES:
                return "full"
            if item.source_class in _HIGH_RISK_SOURCE_CLASSES:
                return "full"
        if min(item.score for item in evidence) < self._grounding_confidence_threshold:
            return "full"
        return "skip"

    def _validate_pre_generation(
        self,
        query_text: str,
        evidence: tuple[RetrievedEvidence, ...] | list[RetrievedEvidence],
    ) -> AnswerResult | _ValidatedContext:
        all_evidence = tuple(evidence)

        if not all_evidence:
            prompt = build_citation_answer_prompt(query_text, all_evidence)
            return self._log_and_return(
                AnswerResult(
                    status="abstained",
                    answer_text="I cannot answer from current evidence because no eligible evidence was retrieved.",
                    claims=(),
                    citations=(),
                    warnings=("No evidence returned.",),
                    prompt=prompt,
                ),
                evidence_items=all_evidence,
            )

        clean_evidence, freshness_warnings = _filter_freshness_uncertain(all_evidence)

        if not clean_evidence:
            prompt = build_citation_answer_prompt(query_text, all_evidence)
            return self._log_and_return(
                AnswerResult(
                    status="abstained",
                    answer_text="I could not confirm that this information is current enough to answer safely from the available evidence.",
                    claims=(),
                    citations=_build_citations(all_evidence),
                    warnings=("Freshness uncertainty requires abstention.",)
                    + freshness_warnings,
                    prompt=prompt,
                ),
                evidence_items=all_evidence,
            )

        evidence_items = clean_evidence

        if _query_expects_numeric_answer(
            query_text
        ) and not _evidence_has_numeric_content(evidence_items):
            prompt = build_citation_answer_prompt(query_text, evidence_items)
            return self._log_and_return(
                AnswerResult(
                    status="abstained",
                    answer_text=(
                        "I could not find specific numeric data for this question "
                        "in the current evidence."
                    ),
                    claims=(),
                    citations=_build_citations(evidence_items),
                    warnings=("Numeric query with no numeric evidence.",)
                    + freshness_warnings,
                    prompt=prompt,
                ),
                evidence_items=evidence_items,
            )

        grounding_mode = self._resolve_grounding_mode(evidence_items)
        t_prompt_start = time.monotonic()
        prompt = build_citation_answer_prompt(
            query_text,
            evidence_items,
            strip_context_window=grounding_mode == "skip",
        )
        citations = _build_citations(evidence_items)
        t_prompt_end = time.monotonic()

        if len(citations) != len(evidence_items):
            return self._log_and_return(
                AnswerResult(
                    status="abstained",
                    answer_text="I cannot answer because the retrieved evidence is missing a source URL or exact locator.",
                    claims=(),
                    citations=citations,
                    warnings=("Evidence missing citation fields.",),
                    prompt=prompt,
                ),
                evidence_items=evidence_items,
            )

        if _has_contradiction(evidence_items):
            return self._log_and_return(
                AnswerResult(
                    status="abstained",
                    answer_text="I found conflicting evidence for this question, so I cannot synthesize a single answer safely.",
                    claims=(),
                    citations=citations,
                    warnings=("Conflicting evidence requires abstention.",),
                    prompt=prompt,
                ),
                evidence_items=evidence_items,
            )

        ambiguity_levels = (
            _program_ambiguity_levels(evidence_items)
            if _is_program_fee_query(query_text)
            else ()
        )
        if ambiguity_levels:
            degree_levels = ", ".join(ambiguity_levels)
            return self._log_and_return(
                AnswerResult(
                    status="abstained",
                    answer_text=(
                        "I found fee information for multiple degree levels "
                        f"({degree_levels}) of a similar program. Could you specify which "
                        "degree level you are asking about?"
                    ),
                    claims=(),
                    citations=citations,
                    warnings=(
                        "Mixed degree-level fee evidence requires clarification.",
                    ),
                    prompt=prompt,
                ),
                evidence_items=evidence_items,
            )

        return _ValidatedContext(
            evidence_items=evidence_items,
            freshness_warnings=freshness_warnings,
            grounding_mode=grounding_mode,
            prompt=prompt,
            citations=citations,
            prompt_build_ms=round((t_prompt_end - t_prompt_start) * 1000, 1),
        )

    def _finalize_post_generation(
        self,
        draft: CitationAnswerDraft,
        ctx: _ValidatedContext,
        *,
        generation_ms: float,
        ground_fn: Callable[
            [tuple[GeneratedClaim, ...], tuple[Citation, ...]],
            GroundingResult,
        ],
        grounding_override: tuple[GroundingResult, float] | None = None,
    ) -> AnswerResult:
        t_cite_start = time.monotonic()
        citations_valid, citation_warnings = validate_claim_citations(
            draft.claims, ctx.citations
        )
        t_cite_end = time.monotonic()
        if not citations_valid:
            return self._log_and_return(
                AnswerResult(
                    status="abstained",
                    answer_text="I cannot answer because the generated response referenced evidence that was not retrieved.",
                    claims=(),
                    citations=ctx.citations,
                    warnings=citation_warnings,
                    prompt=ctx.prompt,
                ),
                evidence_items=ctx.evidence_items,
            )

        grounding_mode = ctx.grounding_mode
        if grounding_mode == "skip":
            t_ground_start = time.monotonic()
            grounding = PassthroughGroundingVerifier().verify(
                draft.claims, ctx.citations
            )
            grounding_ms = round((time.monotonic() - t_ground_start) * 1000, 1)
        elif grounding_override is not None:
            grounding, grounding_ms = grounding_override
        else:
            t_ground_start = time.monotonic()
            grounding = ground_fn(draft.claims, ctx.citations)
            grounding_ms = round((time.monotonic() - t_ground_start) * 1000, 1)
        logger.info(
            "answering.grounding_mode",
            mode=grounding_mode,
            grounding_ms=grounding_ms,
        )
        if not grounding.supported:
            grounding_warnings = grounding.warnings or (
                "Grounding verification failed: claims are not sufficiently "
                "supported by cited evidence.",
            )
            return self._log_and_return(
                AnswerResult(
                    status="abstained",
                    answer_text="I cannot answer because the generated claims could not be verified against the cited evidence.",
                    claims=(),
                    citations=ctx.citations,
                    warnings=grounding_warnings,
                    prompt=ctx.prompt,
                ),
                evidence_items=ctx.evidence_items,
                grounding_score=grounding.score,
            )

        logger.info(
            "answering.stage_latency",
            prompt_build_ms=ctx.prompt_build_ms,
            generation_ms=generation_ms,
            citation_validation_ms=round((t_cite_end - t_cite_start) * 1000, 1),
            grounding_ms=grounding_ms,
            claim_count=len(draft.claims),
        )

        claims = tuple(
            MaterialClaim(text=claim.text, citation_ids=claim.citation_ids)
            for claim in draft.claims
        )
        return self._log_and_return(
            AnswerResult(
                status=draft.status,
                answer_text=draft.answer_text,
                claims=claims,
                citations=ctx.citations,
                warnings=draft.warnings + ctx.freshness_warnings,
                prompt=ctx.prompt,
            ),
            evidence_items=ctx.evidence_items,
            grounding_score=grounding.score,
        )

    def answer(
        self,
        query_text: str,
        evidence: tuple[RetrievedEvidence, ...] | list[RetrievedEvidence],
    ) -> AnswerResult:
        pre = self._validate_pre_generation(query_text, evidence)
        if isinstance(pre, AnswerResult):
            return pre
        ctx = pre

        t_gen_start = time.monotonic()
        draft = self._answer_model.generate(
            CitationAnswerRequest(
                query_text=query_text,
                evidence=ctx.evidence_items,
                citations=ctx.citations,
                prompt=ctx.prompt,
            )
        )
        t_gen_end = time.monotonic()

        return self._finalize_post_generation(
            draft,
            ctx,
            generation_ms=round((t_gen_end - t_gen_start) * 1000, 1),
            ground_fn=self._grounding_verifier.verify,
        )

    async def async_answer(
        self,
        query_text: str,
        evidence: tuple[RetrievedEvidence, ...] | list[RetrievedEvidence],
    ) -> AnswerResult:
        """Async variant of answer(). Uses async LLM generation and
        runs CPU-bound grounding in a thread."""
        pre = self._validate_pre_generation(query_text, evidence)
        if isinstance(pre, AnswerResult):
            return pre
        ctx = pre

        t_gen_start = time.monotonic()
        answer_request = CitationAnswerRequest(
            query_text=query_text,
            evidence=ctx.evidence_items,
            citations=ctx.citations,
            prompt=ctx.prompt,
        )
        async_gen = getattr(self._answer_model, "async_generate", None)
        if async_gen is not None:
            draft = await async_gen(answer_request)
        else:
            draft = await asyncio.to_thread(self._answer_model.generate, answer_request)
        t_gen_end = time.monotonic()

        if ctx.grounding_mode == "skip":
            ground_fn = self._grounding_verifier.verify
            grounding_override = None
        else:
            t_ground_start = time.monotonic()
            grounding_result = await asyncio.to_thread(
                self._grounding_verifier.verify, draft.claims, ctx.citations
            )
            grounding_override = (
                grounding_result,
                round((time.monotonic() - t_ground_start) * 1000, 1),
            )
            ground_fn = lambda claims, citations: grounding_result  # noqa: E731

        return self._finalize_post_generation(
            draft,
            ctx,
            generation_ms=round((t_gen_end - t_gen_start) * 1000, 1),
            ground_fn=ground_fn,
            grounding_override=grounding_override,
        )


class RetrievalServiceProtocol(Protocol):
    def retrieve(
        self,
        query_text: str,
        *,
        active_generation: ServingGeneration,
        source_class_hint: str | None = None,
        record_type_hint: str | None = None,
        hint_is_user_provided: bool = False,
        limit: int = 5,
        secondary_hints: tuple[tuple[str | None, str | None], ...] = (),
    ) -> tuple[RetrievedEvidence, ...]: ...


class QueryService:
    def __init__(
        self,
        *,
        retrieval_service: RetrievalServiceProtocol,
        answering_service: AnsweringService,
    ) -> None:
        self._retrieval_service = retrieval_service
        self._answering_service = answering_service

    def answer_query(
        self,
        query_text: str,
        *,
        retrieval_query: str | None = None,
        active_generation: ServingGeneration,
        source_class_hint: str | None = None,
        record_type_hint: str | None = None,
        hint_is_user_provided: bool = False,
        limit: int = 5,
        secondary_hints: tuple[tuple[str | None, str | None], ...] = (),
    ) -> AnswerResult:
        evidence = self._retrieval_service.retrieve(
            retrieval_query or query_text,
            active_generation=active_generation,
            source_class_hint=source_class_hint,
            record_type_hint=record_type_hint,
            hint_is_user_provided=hint_is_user_provided,
            limit=limit,
            secondary_hints=secondary_hints,
        )
        return self._answering_service.answer(query_text, evidence)

    async def async_answer_query(
        self,
        query_text: str,
        *,
        retrieval_query: str | None = None,
        active_generation: ServingGeneration,
        source_class_hint: str | None = None,
        record_type_hint: str | None = None,
        hint_is_user_provided: bool = False,
        limit: int = 5,
        secondary_hints: tuple[tuple[str | None, str | None], ...] = (),
    ) -> AnswerResult:
        """Async variant: keeps sync retrieval on the calling thread and uses
        async answering for provider I/O."""
        evidence = self._retrieval_service.retrieve(
            retrieval_query or query_text,
            active_generation=active_generation,
            source_class_hint=source_class_hint,
            record_type_hint=record_type_hint,
            hint_is_user_provided=hint_is_user_provided,
            limit=limit,
            secondary_hints=secondary_hints,
        )
        return await self._answering_service.async_answer(query_text, evidence)


def _build_citations(
    evidence: tuple[RetrievedEvidence, ...],
) -> tuple[Citation, ...]:
    citations: list[Citation] = []
    for index, item in enumerate(evidence, start=1):
        if not item.source_url or not item.source_locator:
            continue
        citations.append(
            Citation(
                citation_id=f"[{index}]",
                chunk_id=item.chunk_id or item.record_version_id,
                chunk_index=item.chunk_index,
                chunk_count=item.chunk_count,
                record_version_id=item.record_version_id,
                source_url=item.source_url,
                source_locator=item.source_locator,
                content=item.content,
            )
        )
    return tuple(citations)


def _filter_freshness_uncertain(
    evidence: tuple[RetrievedEvidence, ...],
) -> tuple[tuple[RetrievedEvidence, ...], tuple[str, ...]]:
    """Remove items with freshness uncertainty, return clean items and warnings."""
    clean: list[RetrievedEvidence] = []
    warnings: list[str] = []

    for item in evidence:
        if item.freshness_status != "current":
            warnings.append(
                f"Dropped evidence {item.record_version_id}: "
                f"freshness_status={item.freshness_status}"
            )
            continue
        if (
            item.record_type in CYCLE_AWARE_TYPES
            and item.year_confidence in AMBIGUOUS_YEAR_CONFIDENCE
        ):
            warnings.append(
                f"Dropped evidence {item.record_version_id}: "
                f"ambiguous year_confidence for {item.record_type}"
            )
            continue
        clean.append(item)

    return tuple(clean), tuple(warnings)


def _has_contradiction(evidence: tuple[RetrievedEvidence, ...]) -> bool:
    grouped: dict[tuple[str, str], list[RetrievedEvidence]] = defaultdict(list)
    for item in evidence:
        grouped[(item.conflict_scope_id, item.dedupe_key)].append(item)

    for group in grouped.values():
        winning_tier = min(item.source_authority_tier for item in group)
        winning_items = [
            item for item in group if item.source_authority_tier == winning_tier
        ]
        if len({item.value_hash for item in winning_items}) > 1:
            return True
    return False


class DeterministicCitationAnswerModel:
    def generate(self, request: CitationAnswerRequest) -> CitationAnswerDraft:
        claims = tuple(
            GeneratedClaim(
                text=item.content,
                citation_ids=(f"[{index}]",),
            )
            for index, item in enumerate(request.evidence, start=1)
        )
        answer_lines = [f"{claim.text} {claim.citation_ids[0]}" for claim in claims]
        return CitationAnswerDraft(
            status="answered",
            answer_text="\n".join(answer_lines),
            claims=claims,
        )
