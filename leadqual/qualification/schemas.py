from enum import Enum

from pydantic import BaseModel, Field


# ── Enum ────────────────────────────────────────────────────────────
class CriterionStatus(str, Enum):  # noqa: UP042
    """How a single ICP criterion was evaluated."""

    MET = "MET"
    NOT_MET = "NOT_MET"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


# ── Per-criterion result ────────────────────────────────────────────
class CriterionResult(BaseModel):
    """The LLM's judgment on one ICP criterion."""

    criterion_id: str = Field(
        description="Must match an id from the ICP YAML, e.g. 'crit_industry'."
    )
    status: CriterionStatus = Field(
        description="Whether the lead satisfies this criterion."
    )
    reasoning: str = Field(
        description="One-sentence explanation of why this status was chosen."
    )
    evidence_ids: list[str] = Field(
        default_factory=list,
        description="IDs of supporting evidence (empty in Phase 1).",
    )


# ── Top-level verdict ──────────────────────────────────────────────
class QualificationVerdict(BaseModel):
    """Complete qualification output for a single lead."""

    company_name: str = Field(
        description="Name of the company being evaluated."
    )
    overall_fit: bool = Field(
        description="True if the lead is a good fit for the ICP."
    )
    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="How confident the model is in its verdict (0.0–1.0).",
    )
    criteria_results: list[CriterionResult] = Field(
        description="One result per ICP criterion."
    )
    summary: str = Field(
        description="Two-to-three sentence human-readable summary."
    )