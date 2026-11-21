"""Tests for evidence-ID validation/retry, YAML ICP loading, and full pipeline runner."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import pytest
from sqlalchemy import create_engine, event, select
from sqlalchemy.dialects.postgresql import Insert
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from leadqual.evidence.embeddings import EvidenceBatchEmbedder
from leadqual.evidence.fetcher import SafePageFetcher
from leadqual.evidence.pipeline import EvidenceResearchPipeline
from leadqual.evidence.search_client import TavilySearchClient
from leadqual.qualification.icp_loader import load_icp_from_yaml
from leadqual.qualification.reranker import EvidenceCrossEncoderReranker
from leadqual.qualification.runner import LeadQualificationRunner
from leadqual.qualification.scorer import ICPLeadScorer, QualificationParseError
from leadqual.storage.models import Base, Evidence, Lead


@compiles(Insert, "sqlite")
def _compile_pg_insert_on_conflict_do_nothing_for_sqlite(element: Insert, compiler, **kw) -> str:
    text = compiler.visit_insert(element, **kw)
    if "ON CONFLICT" not in text and (
        element._post_values_clause is not None or getattr(element, "_on_conflict_do_nothing", False)
    ):
        return f"{text} ON CONFLICT DO NOTHING"
    return text


class ScriptedQualificationProvider:
    """Returns scripted responses across retry attempts and records calls."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[str] = []

    def complete_json(self, *, model: str, system_prompt: str, user_prompt: str) -> str:
        self.calls.append(user_prompt)
        if not self._responses:
            raise AssertionError("No more scripted responses configured.")
        return self._responses.pop(0)


class StubSearchProvider:
    def __init__(self, results: list[dict[str, Any]]) -> None:
        self._results = results

    def search(self, query: str, *, max_results: int = 5, search_depth: str = "basic", include_raw_content: bool = False) -> dict[str, Any]:
        return {"results": self._results[:max_results]}


class DummyEmbedderProvider:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.5, 0.25] for _ in texts]


class DummyRerankerModel:
    def predict(self, pairs):
        return [1.0 - idx * 0.1 for idx, _ in enumerate(pairs)]


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


def test_made_up_evidence_id_triggers_retry_and_succeeds_on_valid_id(db_session: Session) -> None:
    lead = Lead(company_name="Acme Corp", target_domain="acme.example.com")
    db_session.add(lead)
    db_session.commit()

    real_ev = Evidence(
        lead_id=lead.id,
        source_url="https://acme.example.com/security",
        retrieved_at=datetime(2026, 11, 21, 10, 0, 0, tzinfo=timezone.utc),
        passage="Acme Corp has SOC2 Type II and SAML SSO.",
        content_hash="a" * 64,
    )
    db_session.add(real_ev)
    db_session.commit()

    made_up_id = str(uuid.uuid4())
    valid_id = str(real_ev.id)

    bad_payload = json.dumps(
        {
            "score": 85,
            "verdict": "QUALIFIED",
            "summary": "Claimed evidence that wasn't retrieved.",
            "decisions": [
                {
                    "criterion": "SOC2 Type II",
                    "status": "met",
                    "evidence_ids": [made_up_id],
                    "rationale": "Invented ID.",
                }
            ],
            "cited_urls": ["https://acme.example.com/security"],
        }
    )
    good_payload = json.dumps(
        {
            "score": 85,
            "verdict": "QUALIFIED",
            "summary": "Grounded in retrieved evidence ID.",
            "decisions": [
                {
                    "criterion": "SOC2 Type II",
                    "status": "met",
                    "evidence_ids": [valid_id],
                    "rationale": "Confirmed in retrieved security passage.",
                }
            ],
            "cited_urls": ["https://acme.example.com/security"],
        }
    )

    provider = ScriptedQualificationProvider([bad_payload, good_payload])
    scorer = ICPLeadScorer(provider=provider, model="test-model", max_retries=2)

    from leadqual.qualification.scorer import ICPProfile

    icp = ICPProfile(name="B2B SaaS", required_signals=["SOC2 Type II"])
    assessment = scorer.score_lead(db_session, lead_id=lead.id, icp=icp)

    assert len(provider.calls) == 2
    assert "RETRY REQUIRED" in provider.calls[1]
    assert assessment.cited_evidence_ids == [valid_id]
    assert assessment.decisions[0].status == "MET"


def test_made_up_evidence_id_exhausting_retries_raises_parse_error(db_session: Session) -> None:
    lead = Lead(company_name="Acme Corp", target_domain="acme.example.com")
    db_session.add(lead)
    db_session.commit()

    ev = Evidence(
        lead_id=lead.id,
        source_url="https://acme.example.com/about",
        retrieved_at=datetime(2026, 11, 21, 10, 0, 0, tzinfo=timezone.utc),
        passage="Acme builds B2B tools.",
        content_hash="b" * 64,
    )
    db_session.add(ev)
    db_session.commit()

    bogus_id = str(uuid.uuid4())
    always_bad = json.dumps(
        {
            "score": 50,
            "verdict": "REVIEW",
            "summary": "Always cites fake ID.",
            "decisions": [
                {
                    "criterion": "SOC2",
                    "status": "not_met",
                    "evidence_ids": [bogus_id],
                    "rationale": "Fake ID.",
                }
            ],
        }
    )

    provider = ScriptedQualificationProvider([always_bad, always_bad])
    scorer = ICPLeadScorer(provider=provider, model="test-model", max_retries=1)

    from leadqual.qualification.scorer import ICPProfile

    icp = ICPProfile(name="B2B SaaS", required_signals=["SOC2"])
    with pytest.raises(QualificationParseError, match="Invented or unretrieved evidence_id"):
        scorer.score_lead(db_session, lead_id=lead.id, icp=icp)


def test_yaml_icp_loader_and_full_runner_end_to_end(tmp_path: Path, db_session: Session) -> None:
    yaml_file = tmp_path / "icp.yaml"
    yaml_file.write_text(
        """
name: Enterprise B2B ICP
target_industries:
  - B2B SaaS
  - FinTech
required_signals:
  - SOC2
  - SAML SSO
disqualifiers:
  - Consumer-only
min_score_threshold: 75
""".strip(),
        encoding="utf-8",
    )

    icp = load_icp_from_yaml(yaml_file)
    assert icp.name == "Enterprise B2B ICP"
    assert icp.min_score_threshold == 75

    lead = Lead(company_name="FinPulse", target_domain="finpulse.example.com")
    db_session.add(lead)
    db_session.commit()

    ev_rows = list(
        db_session.execute(select(Evidence).where(Evidence.lead_id == lead.id)).scalars().all()
    )
    assert ev_rows == []

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            text="<html><body><article><h1>FinPulse Security</h1><p>SOC2 certified and SAML SSO enabled.</p></article></body></html>",
            headers={"content-type": "text/html"},
        )

    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=transport, follow_redirects=False)

    pipeline = EvidenceResearchPipeline(
        search_client=TavilySearchClient(
            api_key="tvly-test",
            client=StubSearchProvider([{"url": "https://finpulse.example.com/security", "title": "Security"}]),
        ),
        page_fetcher=SafePageFetcher(http_client=http_client, allow_dns_lookup=False),
    )
    embedder = EvidenceBatchEmbedder(provider=DummyEmbedderProvider(), batch_size=8, expected_dimension=2)
    reranker = EvidenceCrossEncoderReranker(model=DummyRerankerModel())

    # Pre-seed an evidence row so we can grab its real ID for the scripted LLM response
    pre_ev = Evidence(
        lead_id=lead.id,
        source_url="https://finpulse.example.com/security",
        retrieved_at=datetime(2026, 11, 21, 10, 0, 0, tzinfo=timezone.utc),
        passage="FinPulse Security SOC2 certified and SAML SSO enabled.",
        content_hash="c" * 64,
        embedding=[0.5, 0.25],
    )
    db_session.add(pre_ev)
    db_session.commit()

    llm_response = json.dumps(
        {
            "score": 92,
            "verdict": "QUALIFIED",
            "summary": "FinPulse meets SOC2 and SAML SSO requirements.",
            "decisions": [
                {
                    "criterion": "SOC2",
                    "status": "met",
                    "evidence_ids": [str(pre_ev.id)],
                    "rationale": "Explicitly stated in security evidence.",
                }
            ],
            "cited_urls": ["https://finpulse.example.com/security"],
        }
    )
    scorer = ICPLeadScorer(
        provider=ScriptedQualificationProvider([llm_response]),
        model="free-tier-model",
    )

    runner = LeadQualificationRunner(
        research_pipeline=pipeline,
        scorer=scorer,
        embedder=embedder,
        reranker=reranker,
    )
    result = runner.run_for_lead(
        db_session,
        lead_id=lead.id,
        query="FinPulse SOC2 SAML SSO",
        icp_yaml_path=yaml_file,
    )
    db_session.commit()

    assert result.assessment.verdict == "QUALIFIED"
    assert result.assessment.cited_evidence_ids == [str(pre_ev.id)]
    assert result.embedding_report is not None
    assert result.embedding_report.embedded_count == 1
    http_client.close()