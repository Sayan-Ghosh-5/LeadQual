"""End-to-end runner wiring research -> embedding -> reranking -> evidence-ID ICP scoring."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from leadqual.evidence.embeddings import BatchEmbeddingReport, EvidenceBatchEmbedder
from leadqual.evidence.pipeline import EvidenceResearchPipeline, ResearchRunReport
from leadqual.qualification.icp_loader import load_icp_from_yaml
from leadqual.qualification.reranker import EvidenceCrossEncoderReranker, RankedEvidence
from leadqual.qualification.scorer import (
    ICPLeadScorer,
    ICPProfile,
    QualificationAssessment,
    QualificationError,
)
from leadqual.storage.models import Evidence, Lead


@dataclass
class FullQualificationRunResult:
    """Complete output of running research, optional embedding/reranking, and ICP scoring."""

    lead_id: uuid.UUID
    icp: ICPProfile
    research_report: ResearchRunReport
    embedding_report: BatchEmbeddingReport | None
    ranked_evidence: list[RankedEvidence]
    assessment: QualificationAssessment


@dataclass
class LeadQualificationRunner:
    """Orchestrates evidence research, optional batch embedding, reranking, and ICP scoring."""

    research_pipeline: EvidenceResearchPipeline
    scorer: ICPLeadScorer
    embedder: EvidenceBatchEmbedder | None = None
    reranker: EvidenceCrossEncoderReranker | None = None
    candidate_limit: int = 40
    top_k_passages: int = 8

    def run_for_lead(
        self,
        session: Session,
        *,
        lead_id: uuid.UUID,
        query: str,
        icp: ICPProfile | None = None,
        icp_yaml_path: str | Path | None = None,
        max_results: int = 5,
        search_depth: str = "basic",
    ) -> FullQualificationRunResult:
        if icp is None:
            if icp_yaml_path is None:
                raise QualificationError("Provide either 'icp' or 'icp_yaml_path'.")
            resolved_icp = load_icp_from_yaml(icp_yaml_path)
        else:
            resolved_icp = icp

        lead = session.execute(select(Lead).where(Lead.id == lead_id)).scalar_one_or_none()
        if lead is None:
            raise QualificationError(f"Lead '{lead_id}' not found.")

        # 1. Search -> SSRF-safe fetch/extract -> deduplicated store
        research_report = self.research_pipeline.collect_for_lead(
            session,
            lead_id=lead_id,
            query=query,
            max_results=max_results,
            search_depth=search_depth,
        )

        # 2. Optional batch embedding for newly stored Evidence rows
        embedding_report: BatchEmbeddingReport | None = None
        if self.embedder is not None:
            embedding_report = self.embedder.embed_pending_evidence(
                session,
                lead_id=lead_id,
            )

        # 3. Load candidate Evidence rows for this lead
        candidates = list(
            session.execute(
                select(Evidence)
                .where(Evidence.lead_id == lead_id)
                .order_by(Evidence.retrieved_at.desc())
                .limit(self.candidate_limit)
            )
            .scalars()
            .all()
        )

        # 4. Optional cross-encoder reranking against ICP criteria
        if self.reranker is not None:
            ranked = self.reranker.rank_evidence(
                lead,
                resolved_icp,
                candidates,
                top_k=self.top_k_passages,
            )
            selected_rows = [item.evidence for item in ranked]
        else:
            selected_rows = candidates[: self.top_k_passages]
            ranked = [RankedEvidence(evidence=row, score=0.0) for row in selected_rows]

        # 5. Qualify using only retrieved evidence IDs + retry guard
        assessment = self.scorer.score_passages(lead, resolved_icp, selected_rows)

        return FullQualificationRunResult(
            lead_id=lead_id,
            icp=resolved_icp,
            research_report=research_report,
            embedding_report=embedding_report,
            ranked_evidence=ranked,
            assessment=assessment,
        )