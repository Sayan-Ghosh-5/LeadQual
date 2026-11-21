"""Unit tests for ICPLeadScorer with model swappability and structured JSON validation."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from leadqual.qualification.scorer import (
    ICPLeadScorer,
    ICPProfile,
    QualificationAssessment,
    QualificationParseError,
)
from leadqual.storage.models import Base, Evidence, Lead


class FakeQualificationProvider:
    """Returns canned JSON responses and records the model name requested."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.last_model: str | None = None
        self.last_user_prompt: str | None = None

    def complete_json(self, *, model: str, system_prompt: str, user_prompt: str) -> str:
        self.last_model = model
        self.last_user_prompt = user_prompt
        return self.content


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


def test_scores_lead_using_configured_free_or_custom_model(db_session: Session) -> None:
    lead = Lead(company_name="Acme Security", target_domain="acme.example.com")
    db_session.add(lead)
    db_session.commit()

    ev = Evidence(
        lead_id=lead.id,
        source_url="https://acme.example.com/compliance",
        retrieved_at=datetime(2026, 11, 21, 10, 0, 0, tzinfo=timezone.utc),
        passage="Acme Security provides SOC2 Type II compliance and SAML SSO for B2B SaaS teams.",
        content_hash="a" * 64,
    )
    db_session.add(ev)
    db_session.commit()

    icp = ICPProfile(
        name="Mid-Market B2B SaaS",
        target_industries=["B2B SaaS", "Cloud Infrastructure"],
        required_signals=["SOC2 Type II", "SAML SSO"],
        disqualifiers=["Agency-only"],
        min_score_threshold=70,
    )

    canned = json.dumps(
        {
            "score": 88,
            "verdict": "QUALIFIED",
            "summary": "Evidence confirms SOC2 Type II and SAML SSO alignment.",
            "matched_criteria": ["SOC2 Type II", "SAML SSO"],
            "missing_or_risk_signals": [],
            "cited_urls": ["https://acme.example.com/compliance"],
        }
    )
    provider = FakeQualificationProvider(content=canned)
    scorer = ICPLeadScorer(provider=provider, model="gemini-2.5-flash")

    result = scorer.score_lead(db_session, lead_id=lead.id, icp=icp)

    assert isinstance(result, QualificationAssessment)
    assert provider.last_model == "gemini-2.5-flash"
    assert result.model_used == "gemini-2.5-flash"
    assert result.score == 88
    assert result.verdict == "QUALIFIED"
    assert result.cited_urls == ["https://acme.example.com/compliance"]


def test_malformed_model_json_raises_qualification_parse_error(db_session: Session) -> None:
    lead = Lead(company_name="Beta Corp", target_domain="beta.example.com")
    db_session.add(lead)
    db_session.commit()

    icp = ICPProfile(name="Enterprise")
    provider = FakeQualificationProvider(content="not valid json at all")
    scorer = ICPLeadScorer(provider=provider, model="free-test-model")

    with pytest.raises(QualificationParseError):
        scorer.score_lead(db_session, lead_id=lead.id, icp=icp)