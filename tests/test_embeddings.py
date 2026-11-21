"""Unit tests for EvidenceBatchEmbedder querying NULL embeddings and updating rows."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from leadqual.evidence.embeddings import (
    EmbeddingDimensionError,
    EmbeddingError,
    EvidenceBatchEmbedder,
)
from leadqual.storage.models import Base, Evidence, Lead


class DeterministicEmbeddingProvider:
    """Produces repeatable dummy vectors based on passage length and index."""

    def __init__(self, dim: int = 4, *, fail_on_call: int | None = None, bad_dim: bool = False) -> None:
        self.dim = dim
        self.fail_on_call = fail_on_call
        self.bad_dim = bad_dim
        self.calls: list[list[str]] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        call_index = len(self.calls)
        self.calls.append(list(texts))

        if self.fail_on_call is not None and call_index == self.fail_on_call:
            raise RuntimeError("Simulated provider timeout")

        out: list[list[float]] = []
        for idx, text in enumerate(texts):
            vec_len = (self.dim + 1) if self.bad_dim else self.dim
            base_val = float(len(text) % 10) + float(idx)
            out.append([base_val + float(i) * 0.1 for i in range(vec_len)])
        return out


@pytest.fixture()
def db_session() -> Session:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, _connection_record) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    SessionFactory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    session = SessionFactory()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


def _seed_evidence(
    session: Session,
    *,
    lead_id: uuid.UUID,
    count: int,
    start_index: int = 0,
    already_embedded: bool = False,
) -> list[Evidence]:
    created: list[Evidence] = []
    ts = datetime(2026, 11, 21, 9, 0, 0, tzinfo=timezone.utc)
    for idx in range(start_index, start_index + count):
        ev = Evidence(
            lead_id=lead_id,
            source_url=f"https://acme.example.com/page-{idx}",
            retrieved_at=ts,
            passage=f"Evidence passage number {idx} for enterprise qualification.",
            content_hash=f"{idx:064x}",
            embedding=[0.1, 0.2, 0.3, 0.4] if already_embedded else None,
        )
        session.add(ev)
        created.append(ev)
    session.commit()
    return created


def test_embeds_only_null_rows_in_batches_and_skips_already_embedded(db_session: Session) -> None:
    lead = Lead(company_name="Acme Corp", target_domain="acme.example.com")
    db_session.add(lead)
    db_session.commit()

    already_done = _seed_evidence(db_session, lead_id=lead.id, count=2, already_embedded=True)
    pending = _seed_evidence(db_session, lead_id=lead.id, count=5, start_index=2, already_embedded=False)

    provider = DeterministicEmbeddingProvider(dim=4)
    embedder = EvidenceBatchEmbedder(provider=provider, batch_size=2, expected_dimension=4)

    report = embedder.embed_pending_evidence(db_session, lead_id=lead.id)
    db_session.commit()

    # 5 pending rows with batch_size=2 -> batches of 2, 2, 1
    assert report.batches_processed == 3
    assert report.embedded_count == 5
    assert report.failures == []
    assert [len(b) for b in provider.calls] == [2, 2, 1]

    # Previously embedded rows retain their original vector
    refreshed_done = db_session.execute(
        select(Evidence).where(Evidence.id == already_done[0].id)
    ).scalar_one()
    assert refreshed_done.embedding == [0.1, 0.2, 0.3, 0.4]

    # Pending rows now have non-null 4-dim embeddings
    refreshed_pending = db_session.execute(
        select(Evidence).where(Evidence.id.in_([e.id for e in pending]))
    ).scalars().all()
    assert all(row.embedding is not None and len(row.embedding) == 4 for row in refreshed_pending)


def test_dimension_mismatch_records_failure_without_corrupting_other_batches(db_session: Session) -> None:
    lead = Lead(company_name="Beta Inc", target_domain="beta.example.com")
    db_session.add(lead)
    db_session.commit()

    _seed_evidence(db_session, lead_id=lead.id, count=2, already_embedded=False)

    class MismatchedProvider:
        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 2.0] for _ in texts]

    embedder = EvidenceBatchEmbedder(
        provider=MismatchedProvider(),
        batch_size=2,
        expected_dimension=384,
    )
    report = embedder.embed_pending_evidence(db_session, lead_id=lead.id)

    assert report.embedded_count == 0
    assert report.batches_processed == 0
    assert len(report.failures) == 1
    assert report.failures[0].error_type == EmbeddingDimensionError.__name__


def test_provider_failure_is_isolated_per_batch(db_session: Session) -> None:
    lead = Lead(company_name="Gamma Ltd", target_domain="gamma.example.com")
    db_session.add(lead)
    db_session.commit()

    _seed_evidence(db_session, lead_id=lead.id, count=4, already_embedded=False)

    # Fail on batch index 0, succeed on batch index 1
    provider = DeterministicEmbeddingProvider(dim=4, fail_on_call=0)
    embedder = EvidenceBatchEmbedder(provider=provider, batch_size=2, expected_dimension=4)

    report = embedder.embed_pending_evidence(db_session, lead_id=lead.id)
    db_session.commit()

    assert report.batches_processed == 1
    assert report.embedded_count == 2
    assert len(report.failures) == 1
    assert report.failures[0].error_type == "RuntimeError"