"""Repository helpers for persisting FetchedDocument passages with per-lead deduplication."""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from leadqual.evidence.fetcher import FetchedDocument
from leadqual.storage.models import Evidence


def normalize_passage(text: str) -> str:
    """Normalize internal whitespace so trivial formatting differences hash identically."""
    return " ".join(text.split())


def compute_content_hash(passage: str) -> str:
    """Return a 64-character SHA-256 hex digest for a passage."""
    return hashlib.sha256(passage.encode("utf-8")).hexdigest()


def _ensure_utc(dt: datetime) -> datetime:
    """Ensure a datetime is timezone-aware in UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


@dataclass(frozen=True)
class StoreEvidenceOutcome:
    """Result summary after attempting to store a fetched document's passage."""

    inserted: bool
    content_hash: str
    evidence_id: uuid.UUID | None
    retrieved_at: datetime


def store_fetched_document(
    session: Session,
    *,
    lead_id: uuid.UUID,
    document: FetchedDocument,
) -> StoreEvidenceOutcome:
    """Persist a FetchedDocument into evidence with explicit fetch timestamp and per-lead dedupe."""
    passage = normalize_passage(document.text)
    if not passage:
        raise ValueError("Cannot store an empty passage.")

    content_hash = compute_content_hash(passage)
    fetch_timestamp = _ensure_utc(document.fetched_at)
    new_id = uuid.uuid4()

    stmt = (
        insert(Evidence)
        .values(
            id=new_id,
            lead_id=lead_id,
            source_url=document.final_url,
            retrieved_at=fetch_timestamp,
            passage=passage,
            content_hash=content_hash,
            embedding=None,
        )
        .on_conflict_do_nothing(constraint="uq_evidence_lead_content_hash")
        .returning(Evidence.id)
    )

    inserted_id = session.execute(stmt).scalar_one_or_none()
    if inserted_id is not None:
        return StoreEvidenceOutcome(
            inserted=True,
            content_hash=content_hash,
            evidence_id=inserted_id,
            retrieved_at=fetch_timestamp,
        )

    existing_row = session.execute(
        select(Evidence.id, Evidence.retrieved_at).where(
            Evidence.lead_id == lead_id,
            Evidence.content_hash == content_hash,
        )
    ).one()

    return StoreEvidenceOutcome(
        inserted=False,
        content_hash=content_hash,
        evidence_id=existing_row.id,
        retrieved_at=_ensure_utc(existing_row.retrieved_at),
    )