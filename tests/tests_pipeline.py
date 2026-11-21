"""Unit tests for EvidenceResearchPipeline connecting search, safe fetch, and storage."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx
import pytest
from sqlalchemy import create_engine, event, select
from sqlalchemy.dialects.postgresql import Insert
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from leadqual.evidence.fetcher import SafePageFetcher
from leadqual.evidence.pipeline import EvidenceResearchPipeline
from leadqual.evidence.search_client import TavilySearchClient
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


class StubSearchProvider:
    """Deterministic search provider returning preconfigured hits."""

    def __init__(self, results: list[dict[str, Any]]) -> None:
        self._results = results

    def search(
        self,
        query: str,
        *,
        max_results: int = 5,
        search_depth: str = "basic",
        include_raw_content: bool = False,
    ) -> dict[str, Any]:
        return {"results": self._results[:max_results]}


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


def test_pipeline_collects_skips_unsafe_urls_and_deduplicates_reruns(db_session: Session) -> None:
    lead = Lead(company_name="Acme Cloud", target_domain="acme.example.com")
    db_session.add(lead)
    db_session.commit()

    pages = {
        "https://acme.example.com/security": (
            "<html><body><article><h1>Security</h1>"
            "<p>Acme Cloud provides SOC2 Type II controls and SAML SSO for enterprise customers.</p>"
            "</article></body></html>"
        ),
        "https://acme.example.com/pricing": (
            "<html><body><article><h1>Pricing</h1>"
            "<p>Annual enterprise contracts include dedicated onboarding and audit logging.</p>"
            "</article></body></html>"
        ),
    }

    def handler(request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        if url_str in pages:
            return httpx.Response(status_code=200, text=pages[url_str], headers={"content-type": "text/html"})
        return httpx.Response(status_code=404, text="Not found")

    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=transport, follow_redirects=False)

    search_client = TavilySearchClient(
        api_key="tvly-test-key",
        client=StubSearchProvider(
            [
                {"url": "https://acme.example.com/security", "title": "Security"},
                {"url": "http://127.0.0.1/internal-admin", "title": "SSRF Target"},
                {"url": "https://acme.example.com/pricing", "title": "Pricing"},
                {"url": "https://acme.example.com/security", "title": "Duplicate URL"},
            ]
        ),
    )
    fetcher = SafePageFetcher(http_client=http_client, allow_dns_lookup=False)
    pipeline = EvidenceResearchPipeline(search_client=search_client, page_fetcher=fetcher)

    # First run: 2 valid pages stored, 1 unsafe URL blocked into failures, duplicate URL skipped
    first_run = pipeline.collect_for_lead(
        db_session,
        lead_id=lead.id,
        query="Acme Cloud security pricing",
        max_results=5,
    )
    db_session.commit()

    assert first_run.inserted_count == 2
    assert first_run.deduplicated_count == 0
    assert len(first_run.failures) == 1
    assert first_run.failures[0].url == "http://127.0.0.1/internal-admin"
    assert first_run.failures[0].error_type == "UnsafeUrlError"

    # Second run with same content: 0 new insertions, 2 deduplicated
    second_run = pipeline.collect_for_lead(
        db_session,
        lead_id=lead.id,
        query="Acme Cloud security pricing",
        max_results=5,
    )
    db_session.commit()

    assert second_run.inserted_count == 0
    assert second_run.deduplicated_count == 2

    rows = db_session.execute(select(Evidence).where(Evidence.lead_id == lead.id)).scalars().all()
    assert len(rows) == 2
    assert all(row.retrieved_at is not None for row in rows)
    for row in rows:
        if row.retrieved_at.tzinfo is None:
            assert row.retrieved_at.replace(tzinfo=timezone.utc) <= datetime.now(timezone.utc)
        else:
            assert row.retrieved_at <= datetime.now(timezone.utc)

    http_client.close()