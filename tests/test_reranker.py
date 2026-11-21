"""Unit tests for EvidenceCrossEncoderReranker and score_lead_with_reranking."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from leadqual.qualification.reranker import (
    EvidenceCrossEncoderReranker,
    RerankerError,
    build_icp_rerank_query,
    score_lead_with_reranking,
)
from leadqual.qualification.scorer import ICPLeadScorer, ICPProfile
from leadqual.storage.models import Base, Evidence, Lead


class KeywordCrossEncoder:
    """Deterministic test cross-encoder that scores passages by keyword overlap."""

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.seen_pairs: list[tuple[str, str]] = []

    def predict(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        if self.fail:
            raise RuntimeError("Encoder inference failed")
        self.seen_pairs.extend(pairs)
        scores: list[float] = []
        for _query, passage in pairs:
            lowered = passage.lower()
            score = 0.0
            if "soc2" in lowered:
                score += 5.0
            if "saml" in lowered or "sso" in lowered:
                score += 4.0
            if "blog" in lowered or "holiday" in lowered:
                score -= 2.0
            scores.append(score)
        return scores


class StubQualificationProvider:
    """Captures the exact user prompt sent to the LLM scorer."""

    def __init__(self, response_json: str) -> None:
        self.response_json = response_json
        self.last_user_prompt: str | None = None

    def complete_json(self, *, model: str, system_prompt: str, user_prompt: str) -> str:
        self.last_user_prompt = user_prompt
        return self.response_json


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


def test_reranks_passages_and_passes_top_k_to_scorer(db_session: Session) -> None:
    lead = Lead(company_name="Acme Corp", target_domain="acme.example.com")
    db_session.add(lead)
    db_session.commit()

    ts = datetime(2026, 11, 21, 10, 0, 0, tzinfo=timezone.utc)
    ev_irrelevant = Evidence(
        lead_id=lead.id,
        source_url="https://acme.example.com/blog/holiday-party",
        retrieved_at=ts,
        passage="Holiday blog post with team photos and generic updates.",
        content_hash="1" * 64,
    )
    ev_high_signal = Evidence(
        lead_id=lead.id,
        source_url="https://acme.example.com/trust/soc2-sso",
        retrieved_at=ts,
        passage="Acme Corp holds SOC2 Type II attestation and enterprise SAML SSO.",
        content_hash="2" * 64,
    )
    ev_medium_signal = Evidence(
        lead_id=lead.id,
        source_url="https://acme.example.com/security/sso",
        retrieved_at=ts,
        passage="Supports SAML SSO for enterprise workspaces.",
        content_hash="3" * 64,
    )
    db_session.add_all([ev_irrelevant, ev_high_signal, ev_medium_signal])
    db_session.commit()

    icp = ICPProfile(
        name="Enterprise Security ICP",
        target_industries=["B2B SaaS"],
        required_signals=["SOC2", "SAML SSO"],
        min_score_threshold=75,
    )

    query_str = build_icp_rerank_query(lead, icp)
    assert "Enterprise Security ICP" in query_str
    assert "SOC2" in query_str

    reranker = EvidenceCrossEncoderReranker(model=KeywordCrossEncoder())
    llm_provider = StubQualificationProvider(
        json.dumps(
            {
                "score": 90,
                "verdict": "QUALIFIED",
                "summary": "Strong SOC2 and SAML SSO evidence found.",
                "matched_criteria": ["SOC2", "SAML SSO"],
                "missing_or_risk_signals": [],
                "cited_urls": ["https://acme.example.com/trust/soc2-sso"],
            }
        )
    )
    scorer = ICPLeadScorer(provider=llm_provider, model="free-model")

    assessment, ranked = score_lead_with_reranking(
        db_session,
        lead_id=lead.id,
        icp=icp,
        reranker=reranker,
        scorer=scorer,
        top_k_for_llm=2,
    )

    assert len(ranked) == 2
    assert ranked[0].evidence.id == ev_high_signal.id
    assert ranked[0].score > ranked[1].score
    assert ranked[1].evidence.id == ev_medium_signal.id
    assert assessment.verdict == "QUALIFIED"

    assert llm_provider.last_user_prompt is not None
    assert "https://acme.example.com/trust/soc2-sso" in llm_provider.last_user_prompt
    assert "https://acme.example.com/blog/holiday-party" not in llm_provider.last_user_prompt


def test_reranker_wraps_model_failure_in_reranker_error() -> None:
    lead = Lead(company_name="Acme Corp", target_domain="acme.example.com")
    icp = ICPProfile(name="Enterprise ICP", required_signals=["SOC2"])
    ev = Evidence(
        lead_id=lead.id,
        source_url="https://acme.example.com/docs",
        passage="Some text",
        content_hash="4" * 64,
    )

    reranker = EvidenceCrossEncoderReranker(model=KeywordCrossEncoder(fail=True))
    with pytest.raises(RerankerError):
        reranker.rank_evidence(lead, icp, [ev])