from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from lettucedetect.models.inference import HallucinationDetector

    from unibot.answering.model_adapter import GeneratedClaim
    from unibot.answering.service import Citation

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_PATH = str(Path(__file__).resolve().parents[2].parent / "models" / "tinylettuce-ettin-68m-en")


@dataclass(frozen=True, slots=True)
class ClaimVerdict:
    claim_text: str
    supported: bool
    reasoning: str


@dataclass(frozen=True, slots=True)
class GroundingResult:
    supported: bool
    score: float
    verdicts: tuple[ClaimVerdict, ...]
    warnings: tuple[str, ...] = ()


class GroundingVerifier(Protocol):
    def verify(
        self,
        claims: tuple["GeneratedClaim", ...],
        citations: tuple["Citation", ...],
    ) -> GroundingResult: ...


class PassthroughGroundingVerifier:
    def verify(
        self,
        claims: tuple["GeneratedClaim", ...],
        citations: tuple["Citation", ...],
    ) -> GroundingResult:
        verdicts = tuple(
            ClaimVerdict(
                claim_text=claim.text,
                supported=True,
                reasoning="passthrough",
            )
            for claim in claims
        )
        return GroundingResult(
            supported=True,
            score=1.0,
            verdicts=verdicts,
        )


@lru_cache(maxsize=1)
def _get_detector(model_path: str) -> "HallucinationDetector":
    from lettucedetect.models.inference import HallucinationDetector

    logger.info("Loading LettuceDetect model: %s", model_path)
    return HallucinationDetector(
        method="transformer",
        model_path=model_path,
    )


class LettuceDetectGroundingVerifier:
    def __init__(
        self,
        *,
        model_path: str = _DEFAULT_MODEL_PATH,
        threshold: float = 0.5,
        span_confidence_threshold: float = 0.0,
    ) -> None:
        self._model_path = model_path
        self._threshold = threshold
        self._span_confidence_threshold = span_confidence_threshold

    def verify(
        self,
        claims: tuple["GeneratedClaim", ...],
        citations: tuple["Citation", ...],
    ) -> GroundingResult:
        if not claims:
            return GroundingResult(supported=True, score=1.0, verdicts=())

        citations_by_id = {c.citation_id: c for c in citations}

        try:
            detector = _get_detector(self._model_path)
        except Exception as exc:
            logger.error("Failed to load LettuceDetect model: %s", exc)
            error_verdicts = tuple(
                ClaimVerdict(
                    claim_text=c.text, supported=False, reasoning="detector unavailable",
                )
                for c in claims
            )
            return GroundingResult(
                supported=False,
                score=0.0,
                verdicts=error_verdicts,
                warnings=(f"Grounding verifier failed to load: {exc}",),
            )

        verdicts: list[ClaimVerdict] = []
        for claim in claims:
            evidence_texts = [
                citations_by_id[cid].content
                for cid in claim.citation_ids
                if cid in citations_by_id and citations_by_id[cid].content
            ]

            if not evidence_texts:
                verdicts.append(
                    ClaimVerdict(
                        claim_text=claim.text,
                        supported=False,
                        reasoning="No cited evidence found for this claim.",
                    )
                )
                continue

            try:
                spans = detector.predict(
                    context=evidence_texts,
                    answer=claim.text,
                    question=None,
                    output_format="spans",
                )
            except Exception as exc:
                logger.warning("LettuceDetect prediction failed for claim: %s", exc)
                verdicts.append(
                    ClaimVerdict(
                        claim_text=claim.text,
                        supported=False,
                        reasoning=f"Prediction error: {exc}",
                    )
                )
                continue

            confident_spans = [
                s for s in spans
                if s.get("confidence", 1.0) >= self._span_confidence_threshold
            ]

            if confident_spans:
                flagged = "; ".join(s["text"] for s in confident_spans if "text" in s)
                verdicts.append(
                    ClaimVerdict(
                        claim_text=claim.text,
                        supported=False,
                        reasoning=f"Hallucinated spans detected: {flagged}",
                    )
                )
            else:
                verdicts.append(
                    ClaimVerdict(
                        claim_text=claim.text,
                        supported=True,
                        reasoning="Claim is supported by cited evidence.",
                    )
                )

        verdict_tuple = tuple(verdicts)
        supported_count = sum(1 for v in verdict_tuple if v.supported)
        score = supported_count / len(verdict_tuple) if verdict_tuple else 0.0

        return GroundingResult(
            supported=score >= self._threshold,
            score=score,
            verdicts=verdict_tuple,
        )


def warm_detector(model_path: str = _DEFAULT_MODEL_PATH) -> None:
    """Eagerly load the LettuceDetect model so first query doesn't pay the cost."""
    logger.info("Warming LettuceDetect grounding model: %s", model_path)
    _get_detector(model_path)
    logger.info("LettuceDetect grounding model ready")


def create_grounding_verifier(
    *,
    settings: object | None = None,
) -> GroundingVerifier:
    if settings is None:
        return PassthroughGroundingVerifier()

    from unibot.settings import resolve_grounding_verifier_backend

    backend = resolve_grounding_verifier_backend(settings)
    if backend == "lettucedetect":
        return LettuceDetectGroundingVerifier(
            model_path=str(
                getattr(settings, "grounding_model", _DEFAULT_MODEL_PATH)
            ),
            threshold=float(getattr(settings, "grounding_threshold", 0.5)),
            span_confidence_threshold=float(
                getattr(settings, "grounding_span_confidence_threshold", 0.6)
            ),
        )

    return PassthroughGroundingVerifier()
