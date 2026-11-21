"""Unit tests for storing FetchedDocument passages with explicit timestamps and per-lead dedupe."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, event, select
from sqlalchemy.dialects.postgresql import Insert
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from leadqual.evidence.fetcher import FetchedDocument
from leadqual.storage.evidence_repo import (
    compute_content_hash,
    normalize_passage,
    store_fetched_document,
)
from leadqual.storage.models import Base, Evidence, Lead


@compiles(Insert, "sqlite")
def _compile_pg_insert_on_conflict_do_nothing_for_sqlite(element: Insert, compiler, **kw) -> str:
    """Allow testing PostgreSQL ON CONFLICT DO NOTHING semantics on SQLite in unit tests."""
    text = compiler.visit_insert(element, **kw)
    if "ON CONFLICT" not in text and (
        element._post_values_clause is not None or getattr(element, "_on_conflict_do_nothing", False)
    ):
        return f"{text} ON CONFLICT DO NOTHING"
    return text


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


def _create_lead(session: Session, company_name: str = "Acme Corp") -> Lead:
    lead = Lead(company_name=company_name, target_domain="acme.example.com")
    session.add(lead)
    session.commit()
    return lead


def test_stores_fetched_document_with_explicit_utc_timestamp(db_session: Session) -> None:
    lead = _create_lead(db_session)
    explicit_ts = datetime(2026, 11, 21, 7, 55, 0, tzinfo=timezone.utc)
    doc = FetchedDocument(
        url="https://acme.example.com/start",
        final_url="https://acme.example.com/security",
        status_code=200,
        text="SOC2 Type II\n\n  certified with SSO and RBAC.",
        fetched_at=explicit_ts,
    )

    outcome = store_fetched_document(db_session, lead_id=lead.id, document=doc)
    db_session.commit()

    expected_passage = "SOC2 Type II certified with SSO and RBAC."
    assert outcome.inserted is True
    assert outcome.evidence_id is not None
    assert outcome.content_hash == compute_content_hash(expected_passage)
    assert outcome.retrieved_at == explicit_ts
    assert normalize_passage(doc.text) == expected_passage

    stored = db_session.execute(
        select(Evidence).where(Evidence.id == outcome.evidence_id)
    ).scalar_one()
    assert stored.source_url == "https://acme.example.com/security"
    assert stored.retrieved_at.replace(tzinfo=timezone.utc) == explicit_ts


def test_running_same_page_twice_stores_passage_once_for_same_lead(db_session: Session) -> None:
    lead = _create_lead(db_session)
    ts1 = datetime(2026, 11, 21, 7, 55, 0, tzinfo=timezone.utc)
    ts2 = datetime(2026, 11, 21, 8, 5, 0, tzinfo=timezone.utc)

    doc1 = FetchedDocument(
        url="https://acme.example.com/about",
        final_url="https://acme.example.com/about",
        status_code=200,
        text="Acme builds B2B workflow software.",
        fetched_at=ts1,
    )
    doc2 = FetchedDocument(
        url="https://acme.example.com/about?utm_source=test",
        final_url="https://acme.example.com/about",
        status_code=200,
        text="Acme   builds\nB2B workflow software.",
        fetched_at=ts2,
    )

    first = store_fetched_document(db_session, lead_id=lead.id, document=doc1)
    second = store_fetched_document(db_session, lead_id=lead.id, document=doc2)
    db_session.commit()

    assert first.inserted is True
    assert second.inserted is False
    assert second.content_hash == first.content_hash
    assert second.evidence_id == first.evidence_id
    assert second.retrieved_at == ts1

    rows = db_session.execute(
        select(Evidence).where(Evidence.lead_id == lead.id)
    ).scalars().all()
    assert len(rows) == 1


def test_same_passage_for_different_leads_is_allowed(db_session: Session) -> None:
    lead_a = _create_lead(db_session, "Lead A")
    lead_b = _create_lead(db_session, "Lead B")
    shared_text = "Shared industry benchmark report text."
    ts = datetime(2026, 11, 21, 7, 55, 0, tzinfo=timezone.utc)

    doc_a = FetchedDocument(
        url="https://report.example.com/2026",
        final_url="https://report.example.com/2026",
        status_code=200,
        text=shared_text,
        fetched_at=ts,
    )
    doc_b = FetchedDocument(
        url="https://report.example.com/2026",
        final_url="https://report.example.com/2026",
        status_code=200,
        text=shared_text,
        fetched_at=ts,
    )

    res_a = store_fetched_document(db_session, lead_id=lead_a.id, document=doc_a)
    res_b = store_fetched_document(db_session, lead_id=lead_b.id, document=doc_b)
    db_session.commit()

    assert res_a.inserted is True
    assert res_b.inserted is True
    assert res_a.content_hash == res_b.content_hash
    assert res_a.evidence_id != res_b.evidence_id


def test_empty_passage_raises_value_error(db_session: Session) -> None:
    lead = _create_lead(db_session)
    blank_doc = FetchedDocument(
        url="https://acme.example.com/blank",
        final_url="https://acme.example.com/blank",
        status_code=200,
        text="   \n\t  ",
        fetched_at=datetime(2026, 11, 21, 7, 55, 0, tzinfo=timezone.utc),
    )

    with pytest.raises(ValueError, match="empty passage"):
        store_fetched_document(db_session, lead_id=lead.id, document=blank_doc)