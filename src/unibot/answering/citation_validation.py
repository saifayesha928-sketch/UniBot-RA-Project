from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from unibot.answering.model_adapter import GeneratedClaim
    from unibot.answering.service import Citation


def validate_claim_citations(
    claims: tuple["GeneratedClaim", ...],
    citations: tuple["Citation", ...],
) -> tuple[bool, tuple[str, ...]]:
    """Check that every citation_id maps to a known evidence chunk.

    Returns ``(valid, warnings)`` where *valid* is ``False`` when any claim
    references an unknown or empty citation id.
    """
    citations_by_id = {citation.citation_id: citation for citation in citations}
    warnings: list[str] = []

    for claim in claims:
        if not claim.citation_ids:
            warnings.append(
                "Claim is missing citations; forcing abstention for safety."
            )
            continue
        for cid in claim.citation_ids:
            if not cid or cid not in citations_by_id:
                warnings.append(
                    f"Claim references unknown citation '{cid}'; "
                    "forcing abstention for safety."
                )

    return (len(warnings) == 0, tuple(warnings))
