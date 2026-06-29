from __future__ import annotations

from unibot.verify.rules import ManualReviewEvent, VerificationDecision


class ManualReviewQueue:
    def __init__(self) -> None:
        self._events: list[ManualReviewEvent] = []

    @property
    def events(self) -> tuple[ManualReviewEvent, ...]:
        return tuple(self._events)

    def enqueue(self, decision: VerificationDecision) -> ManualReviewEvent:
        if not decision.requires_manual_review or decision.manual_review_reason is None:
            raise ValueError("decision does not require manual review")

        event = ManualReviewEvent(
            record_version_id=decision.candidate.record_version_id,
            event_type="manual_review_required",
            verification_status="pending",
            event_payload={
                "reason": decision.manual_review_reason,
                "record_id": decision.candidate.record_id,
                "record_type": decision.candidate.record_type,
                "source_url": decision.candidate.source_url,
                "conflicting_record_ids": list(decision.conflicting_record_ids),
            },
            notes=decision.notes,
        )
        self._events.append(event)
        return event
