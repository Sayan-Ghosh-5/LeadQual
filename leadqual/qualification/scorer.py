"""ICP lead qualification scorer with mandatory evidence IDs and retry validation."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from leadqual.storage.models import Evidence, Lead


class QualificationError(Exception):
    """Base exception for ICP qualification errors."""


class QualificationParseError(QualificationError):
    """Raised when model output cannot be parsed or fails schema/evidence-ID validation."""


class ICPProfile(BaseModel):
    """Ideal Customer Profile criteria used to score a lead."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., min_length=1)
    target_industries: list[str] = Field(default_factory=list)
    required_signals: list[str] = Field(default_factory=list)
    disqualifiers: list[str] = Field(default_factory=list)
    min_score_threshold: int = Field(default=70, ge=0, le=100)


class CriterionDecision(BaseModel):
    """A single evaluated criterion with mandatory grounded evidence IDs."""

    model_config = ConfigDict(frozen=True)

    criterion: str = Field(..., min_length=1)
    status: Literal["MET", "NOT_MET"]
    evidence_ids: list[str] = Field(..., min_length=1)
    rationale: str = Field(..., min_length=1)

    @field_validator("status", mode="before")
    @classmethod
    def normalize_status(cls, value: Any) -> str:
        if isinstance(value, str):
            normalized = value.strip().upper()
            if normalized in {"MET", "NOT_MET"}:
                return normalized
        raise ValueError("status must be 'met' or 'not_met'")


class QualificationAssessment(BaseModel):
    """Structured qualification output grounded in retrieved evidence IDs."""

    model_config = ConfigDict(frozen=True)

    lead_id: uuid.UUID
    icp_name: str
    model_used: str
    score: int = Field(..., ge=0, le=100)
    verdict: Literal["QUALIFIED", "REVIEW", "NOT_QUALIFIED"]
    summary: str = Field(..., min_length=1)
    decisions: list[CriterionDecision] = Field(default_factory=list)
    matched_criteria: list[str] = Field(default_factory=list)
    missing_or_risk_signals: list[str] = Field(default_factory=list)
    cited_evidence_ids: list[str] = Field(default_factory=list)
    cited_urls: list[str] = Field(default_factory=list)

    @field_validator("verdict", mode="before")
    @classmethod
    def normalize_verdict(cls, value: Any) -> str:
        if isinstance(value, str):
            return value.strip().upper()
        return value


class QualificationProvider(Protocol):
    """Protocol for chat/completion backends used in ICP scoring."""

    def complete_json(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
    ) -> str: ...


def _extract_json_object(raw_text: str) -> dict[str, Any]:
    """Extract a JSON object even if wrapped in ```json fences."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise QualificationParseError(f"Failed to parse JSON from model output: {exc}") from exc

    if not isinstance(parsed, dict):
        raise QualificationParseError("Top-level model output must be a JSON object.")
    return parsed


def _collect_and_validate_evidence_ids(
    payload: dict[str, Any],
    allowed_ids: set[str],
) -> tuple[list[CriterionDecision], list[str]]:
    """Parse criterion decisions and verify every cited ID was retrieved in this run."""
    raw_decisions = payload.get("decisions")
    if not isinstance(raw_decisions, list):
        if "matched_criteria" in payload or "missing_or_risk_signals" in payload:
            return [], []
        raise QualificationParseError("Missing or invalid 'decisions' list in model response.")

    decisions: list[CriterionDecision] = []
    seen_ids: list[str] = []

    for entry in raw_decisions:
        if not isinstance(entry, dict):
            raise QualificationParseError("Each item in 'decisions' must be an object.")
        try:
            decision = CriterionDecision(**entry)
        except Exception as exc:
            raise QualificationParseError(f"Invalid criterion decision: {exc}") from exc

        for ev_id in decision.evidence_ids:
            if ev_id not in allowed_ids:
                raise QualificationParseError(
                    f"Invented or unretrieved evidence_id '{ev_id}' cited in decision '{decision.criterion}'."
                )
            if ev_id not in seen_ids:
                seen_ids.append(ev_id)

        decisions.append(decision)

    extra_cited = payload.get("cited_evidence_ids")
    if extra_cited is not None:
        if not isinstance(extra_cited, list) or not all(isinstance(x, str) for x in extra_cited):
            raise QualificationParseError("'cited_evidence_ids' must be a list of strings.")
        for ev_id in extra_cited:
            if ev_id not in allowed_ids:
                raise QualificationParseError(
                    f"Invented or unretrieved evidence_id '{ev_id}' in 'cited_evidence_ids'."
                )
            if ev_id not in seen_ids:
                seen_ids.append(ev_id)

    return decisions, seen_ids


@dataclass
class ICPLeadScorer:
    """Evaluates stored Evidence against an ICPProfile with strict evidence-ID verification and retry."""

    provider: QualificationProvider
    model: str
    max_evidence_items: int = 12
    max_chars_per_passage: int = 1200
    max_retries: int = 2

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative.")

    def build_prompts(
        self,
        lead: Lead,
        icp: ICPProfile,
        evidence_rows: list[Evidence],
        *,
        retry_reason: str | None = None,
    ) -> tuple[str, str]:
        system_prompt = (
            "You are a strict B2B lead qualification analyst. "
            "Evaluate the lead strictly using the provided evidence passages. "
            "Every criterion decision MUST have status 'met' or 'not_met' and MUST include at least one "
            "exact evidence_id from the retrieved passages provided in the prompt. "
            "NEVER invent or guess evidence IDs. "
            "Respond ONLY with a valid JSON object containing keys: "
            "score (integer 0-100), verdict ('QUALIFIED', 'REVIEW', or 'NOT_QUALIFIED'), "
            "summary (string), "
            "decisions (list of objects with 'criterion', 'status' ['met'|'not_met'], 'evidence_ids' [non-empty list of IDs], 'rationale'), "
            "cited_urls (list of strings)."
        )

        id_to_url: dict[str, str] = {}
        formatted_blocks: list[str] = []
        for row in evidence_rows:
            ev_id = str(row.id)
            id_to_url[ev_id] = row.source_url
            clipped = row.passage[: self.max_chars_per_passage]
            formatted_blocks.append(
                f"[evidence_id: {ev_id}]\nURL: {row.source_url}\nPassage: {clipped}"
            )

        evidence_section = (
            "\n\n".join(formatted_blocks) if formatted_blocks else "(No stored evidence available.)"
        )
        allowed_list = ", ".join(str(r.id) for r in evidence_rows) or "NONE"

        retry_banner = ""
        if retry_reason:
            retry_banner = (
                f"RETRY REQUIRED: Your previous output failed validation ({retry_reason}). "
                f"Use ONLY retrieved evidence_ids from this exact set: [{allowed_list}].\n\n"
            )

        user_prompt = (
            f"{retry_banner}"
            f"ICP Name: {icp.name}\n"
            f"Target Industries: {', '.join(icp.target_industries) or 'Any'}\n"
            f"Required Signals: {', '.join(icp.required_signals) or 'None specified'}\n"
            f"Disqualifiers: {', '.join(icp.disqualifiers) or 'None specified'}\n"
            f"Minimum Qualification Score: {icp.min_score_threshold}\n"
            f"Allowed Evidence IDs: [{allowed_list}]\n\n"
            f"Lead Company: {lead.company_name}\n"
            f"Lead Domain: {lead.target_domain}\n\n"
            f"Retrieved Evidence:\n{evidence_section}\n"
        )
        return system_prompt, user_prompt

    _build_prompts = build_prompts

    def score_passages(
        self,
        lead: Lead,
        icp: ICPProfile,
        evidence_rows: list[Evidence],
    ) -> QualificationAssessment:
        """Score a pre-selected/reranked list of Evidence rows with retry on invalid/invented IDs."""
        allowed_ids: set[str] = {str(row.id) for row in evidence_rows}
        id_to_url: dict[str, str] = {str(row.id): row.source_url for row in evidence_rows}

        last_error: QualificationParseError | None = None
        retry_reason: str | None = None

        for _attempt in range(self.max_retries + 1):
            system_prompt, user_prompt = self.build_prompts(
                lead,
                icp,
                evidence_rows,
                retry_reason=retry_reason,
            )
            raw_output = self.provider.complete_json(
                model=self.model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )

            try:
                payload = _extract_json_object(raw_output)
                decisions, cited_ids = _collect_and_validate_evidence_ids(payload, allowed_ids)

                derived_urls: list[str] = []
                for ev_id in cited_ids:
                    url = id_to_url.get(ev_id)
                    if url and url not in derived_urls:
                        derived_urls.append(url)
                for url in payload.get("cited_urls", []):
                    if isinstance(url, str) and url not in derived_urls:
                        derived_urls.append(url)

                matched = payload.get("matched_criteria", [d.criterion for d in decisions if d.status == "MET"])
                missing = payload.get(
                    "missing_or_risk_signals", [d.criterion for d in decisions if d.status == "NOT_MET"]
                )

                return QualificationAssessment(
                    lead_id=lead.id,
                    icp_name=icp.name,
                    model_used=self.model,
                    score=payload["score"],
                    verdict=payload["verdict"],
                    summary=payload["summary"],
                    decisions=decisions,
                    matched_criteria=matched,
                    missing_or_risk_signals=missing,
                    cited_evidence_ids=cited_ids,
                    cited_urls=derived_urls,
                )
            except QualificationParseError as exc:
                last_error = exc
                retry_reason = str(exc)

        assert last_error is not None
        raise last_error

    def score_lead(
        self,
        session: Session,
        *,
        lead_id: uuid.UUID,
        icp: ICPProfile,
    ) -> QualificationAssessment:
        """Load stored Evidence for a lead, score against ICP, and validate evidence IDs."""
        lead = session.execute(select(Lead).where(Lead.id == lead_id)).scalar_one_or_none()
        if lead is None:
            raise QualificationError(f"Lead '{lead_id}' not found.")

        evidence_rows = list(
            session.execute(
                select(Evidence)
                .where(Evidence.lead_id == lead_id)
                .order_by(Evidence.retrieved_at.desc())
                .limit(self.max_evidence_items)
            )
            .scalars()
            .all()
        )
        return self.score_passages(lead, icp, evidence_rows)