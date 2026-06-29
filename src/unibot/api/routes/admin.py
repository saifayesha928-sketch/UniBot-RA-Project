from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, cast
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import select

from unibot.api.dependencies import (
    get_session,
    require_admin_key,
    serialize_generation,
)
from unibot.api.serialization import serialize_datetime
from unibot.db.models import CanonicalRecord, ServingGeneration, VerificationEvent
from unibot.verify.rules import VerificationStatus
from unibot.db.repositories.serving_generations import ServingGenerationRepository
from unibot.scheduler.jobs import recompute_scope_state

router = APIRouter(prefix="/admin", dependencies=[Depends(require_admin_key)])


class ReviewResolutionRequest(BaseModel):
    reviewer: str
    notes: str | None = None


@router.get("/generations")
def list_generations(request: Request) -> dict[str, Any]:
    session, close_session = get_session(request)
    try:
        items = sorted(
            session.execute(select(ServingGeneration)).scalars().all(),
            key=lambda generation: (
                generation.status != "active",
                generation.created_at,
                generation.generation_id,
            ),
        )
        active_generation = ServingGenerationRepository(session).get_active_generation()
        return {
            "active_generation": (
                serialize_generation(active_generation)
                if active_generation is not None
                else None
            ),
            "items": [serialize_generation(item) for item in items],
        }
    finally:
        if close_session:
            session.close()


@router.get("/review-queue")
def review_queue(request: Request) -> dict[str, Any]:
    session, close_session = get_session(request)
    try:
        rows = session.execute(
            select(VerificationEvent, CanonicalRecord)
            .join(
                CanonicalRecord,
                VerificationEvent.record_version_id == CanonicalRecord.record_version_id,
            )
            .where(VerificationEvent.event_type == "manual_review_required")
            .where(VerificationEvent.verification_status == "pending")
            .order_by(CanonicalRecord.record_version_id)
        ).all()
        items = []
        for event, record in rows:
            payload = event.event_payload or {}
            items.append(
                {
                    "event_id": event.event_id,
                    "record_version_id": record.record_version_id,
                    "record_id": record.record_id,
                    "record_type": record.record_type,
                    "source_url": record.source_url,
                    "source_locator": record.source_locator,
                    "freshness_status": record.freshness_status,
                    "verification_status": record.verification_status,
                    "serving_status": record.serving_status,
                    "reason": payload.get("reason"),
                    "conflicting_record_ids": payload.get("conflicting_record_ids", []),
                    "notes": event.notes,
                    "created_at": serialize_datetime(event.created_at),
                }
            )
        return {"items": items}
    finally:
        if close_session:
            session.close()


@router.post("/review-queue/{event_id}/approve")
def approve_review(
    event_id: UUID,
    payload: ReviewResolutionRequest,
    request: Request,
) -> dict[str, Any]:
    return _resolve_review(str(event_id), payload, request, resolution="verified")


@router.post("/review-queue/{event_id}/reject")
def reject_review(
    event_id: UUID,
    payload: ReviewResolutionRequest,
    request: Request,
) -> dict[str, Any]:
    return _resolve_review(str(event_id), payload, request, resolution="rejected")


def _resolve_review(
    event_id: str,
    payload: ReviewResolutionRequest,
    request: Request,
    *,
    resolution: str,
) -> dict[str, Any]:
    session, close_session = get_session(request)
    try:
        event = session.get(VerificationEvent, event_id)
        if event is None:
            raise HTTPException(status_code=404, detail="Review event not found.")

        resolved_at = datetime.now(timezone.utc)
        event.verification_status = cast(VerificationStatus, resolution)
        event.reviewer = payload.reviewer
        event.notes = payload.notes
        event.resolved_at = resolved_at

        if event.record_version_id is not None:
            record = session.get(CanonicalRecord, event.record_version_id)
            if record is not None:
                record.verification_status = cast(VerificationStatus, resolution)

                recompute_scope_state(session, record.record_version_id)

        session.commit()

        return {
            "event": {
                "event_id": event.event_id,
                "verification_status": event.verification_status,
                "reviewer": event.reviewer,
                "notes": event.notes,
                "resolved_at": serialize_datetime(event.resolved_at),
            },
        }
    finally:
        if close_session:
            session.close()
