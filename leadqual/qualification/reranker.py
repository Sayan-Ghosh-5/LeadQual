"""Hugging Face cross-encoder reranker for ordering Evidence passages against ICP criteria."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session

from leadqual.qualification.scorer import (
    ICPLeadScorer,
    ICPProfile,
    QualificationAssessment,
    QualificationError,
)
from leadqual.storage.models import Evidence, Lead


class RerankerError(QualificationError):
    """Raised when passage reranking fails or returns invalid scores."""


class CrossEncoderScorer(Protocol):
    """Protocol for cross-encoder models that score (query, passage) pairs."""

    def predict(self, pairs: Sequence[tuple[str, str]]) -> Sequence[float]: ...


@dataclass(frozen=True)
class RankedEvidence:
    """An Evidence record paired with its cross-encoder relevance score."""

    evidence: Evidence
    score: float


def build_icp_rerank_query(lead: Lead, icp: ICPProfile) -> str:
    """Build a concise query string summarizing ICP criteria and target lead context."""
    parts: list[str] = [f"ICP: {icp.name}", f"Company: {lead.company_name}"]
    if lead.target_domain:
        parts.append(f"Domain: {lead.target_domain}")
    if icp.target_industries:
        parts.append(f"Industries: {', '.join(icp.target_industries)}")
    if icp.required_signals:
        parts.append(f"Required signals: {', '.join(icp.required_signals)}")
    if icp.disqualifiers:
        parts.append(f"Disqualifiers: {', '.join(icp.disqualifiers)}")
    return " | ".join(parts)


def load_hf_cross_encoder(model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2") -> CrossEncoderScorer:
    """Load a Hugging Face CrossEncoder model via sentence-transformers."""
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RerankerError(
            "sentence-transformers is required to load a Hugging Face cross-encoder model."
        ) from exc

    try:
        return CrossEncoder(model_name)
    except Exception as exc:
        raise RerankerError(f"Failed to load cross-encoder model '{model_name}': {exc}") from exc


@dataclass
class EvidenceCrossEncoderReranker:
    """Ranks stored Evidence passages for a lead against an ICPProfile using a cross-encoder."""

    model: CrossEncoderScorer
    max_chars_per_passage: int = 1200

    def rank_evidence(
        self,
        lead: Lead,
        icp: ICPProfile,
        evidence_rows: Sequence[Evidence],
        *,
        top_k: int | None = None,
    ) -> list[RankedEvidence]:
        """Score (icp_query, passage) pairs and return passages sorted descending by score."""
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive when provided.")
        if not evidence_rows:
            return []

        query = build_icp_rerank_query(lead, icp)
        pairs: list[tuple[str, str]] = [
            (query, row.passage[: self.max_chars_per_passage]) for row in evidence_rows
        ]

        try:
            raw_scores = list(self.model.predict(pairs))
        except Exception as exc:
            raise RerankerError(f"Cross-encoder prediction failed: {exc}") from exc

        if len(raw_scores) != len(evidence_rows):
            raise RerankerError(
                f"Cross-encoder returned {len(raw_scores)} scores for {len(evidence_rows)} passages."
            )

        ranked = [
            RankedEvidence(evidence=row, score=float(score))
            for row, score in zip(evidence_rows, raw_scores)
        ]
        ranked.sort(key=lambda item: item.score, reverse=True)

        if top_k is not None:
            return ranked[:top_k]
        return ranked


def score_lead_with_reranking(
    session: Session,
    *,
    lead_id: uuid.UUID,
    icp: ICPProfile,
    reranker: EvidenceCrossEncoderReranker,
    scorer: ICPLeadScorer,
    candidate_limit: int = 40,
    top_k_for_llm: int = 8,
) -> tuple[QualificationAssessment, list[RankedEvidence]]:
    """Fetch candidate Evidence rows, rerank against ICP, and pass top passages to ICPLeadScorer."""
    if candidate_limit <= 0 or top_k_for_llm <= 0:
        raise ValueError("candidate_limit and top_k_for_llm must be positive.")

    lead = session.execute(select(Lead).where(Lead.id == lead_id)).scalar_one_or_none()
    if lead is None:
        raise QualificationError(f"Lead '{lead_id}' not found.")

    candidates = list(
        session.execute(
            select(Evidence)
            .where(Evidence.lead_id == lead_id)
            .order_by(Evidence.retrieved_at.desc())
            .limit(candidate_limit)
        )
        .scalars()
        .all()
    )

    ranked = reranker.rank_evidence(lead, icp, candidates, top_k=top_k_for_llm)
    top_passages = [item.evidence for item in ranked]

    # Temporarily pass only the top-ranked passages into scorer via prompt formatting
    system_prompt, user_prompt = scorer._build_prompts(lead, icp, top_passages)
    raw_output = scorer.provider.complete_json(
        model=scorer.model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
    from leadqual.qualification.scorer import QualificationParseError, _extract_json_object

    payload = _extract_json_object(raw_output)
    try:
        assessment = QualificationAssessment(
            lead_id=lead_id,
            icp_name=icp.name,
            model_used=scorer.model,
            score=payload["score"],
            verdict=payload["verdict"],
            summary=payload["summary"],
            matched_criteria=payload.get("matched_criteria", []),
            missing_or_risk_signals=payload.get("missing_or_risk_signals", []),
            cited_urls=payload.get("cited_urls", []),
        )
    except Exception as exc:
        raise QualificationParseError(f"Invalid qualification fields: {exc}") from exc

    return assessment, ranked