"""ICP lead qualification scorer backed by stored Evidence rows and a swappable LLM provider."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from leadqual.storage.models import Evidence, Lead


class QualificationError(Exception):
    """Base exception for ICP qualification errors."""


class QualificationParseError(QualificationError):
    """Raised when the model response cannot be parsed or validated."""


class ICPProfile(BaseModel):
    """Ideal Customer Profile criteria used to score a lead."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., min_length=1)
    target_industries: list[str] = Field(default_factory=list)
    required_signals: list[str] = Field(default_factory=list)
    disqualifiers: list[str] = Field(default_factory=list)
    min_score_threshold: int = Field(default=70, ge=0, le=100)


class QualificationAssessment(BaseModel):
    """Structured qualification output grounded in stored evidence."""

    model_config = ConfigDict(frozen=True)

    lead_id: uuid.UUID
    icp_name: str
    model_used: str
    score: int = Field(..., ge=0, le=100)
    verdict: Literal["QUALIFIED", "REVIEW", "NOT_QUALIFIED"]
    summary: str = Field(..., min_length=1)
    matched_criteria: list[str] = Field(default_factory=list)
    missing_or_risk_signals: list[str] = Field(default_factory=list)
    cited_urls: list[str] = Field(default_factory=list)

    @field_validator("verdict", mode="before")
    @classmethod
    def normalize_verdict(cls, v: Any) -> str:
        if isinstance(v, str):
            return v.strip().upper()
        return v


class QualificationProvider(Protocol):
    """Protocol for chat/completion backends used in ICP scoring."""

    def complete_json(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
    ) -> str: ...


@dataclass
class OpenAICompatibleQualificationProvider:
    """Calls an OpenAI-compatible /chat/completions endpoint (free-tier API or local server)."""

    base_url: str
    api_key: str | None = None
    timeout: float = 30.0
    temperature: float = 0.1
    http_client: httpx.Client | None = None

    def complete_json(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        endpoint = f"{self.base_url.rstrip('/')}/chat/completions"
        headers: dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload: dict[str, Any] = {
            "model": model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        try:
            if self.http_client is not None:
                resp = self.http_client.post(endpoint, json=payload, headers=headers, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
            else:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(endpoint, json=payload, headers=headers)
                    resp.raise_for_status()
                    data = resp.json()
        except httpx.HTTPError as exc:
            raise QualificationError(f"Qualification LLM request failed: {exc}") from exc

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise QualificationParseError("Unexpected chat completion response structure.") from exc

        if not isinstance(content, str) or not content.strip():
            raise QualificationParseError("Model returned empty content.")
        return content


def _extract_json_object(raw_text: str) -> dict[str, Any]:
    """Extract a JSON object even if wrapped in ```json fences."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        # Drop opening and closing fence lines if present
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


@dataclass
class ICPLeadScorer:
    """Evaluates a Lead's stored Evidence against an ICPProfile using a configurable model."""

    provider: QualificationProvider
    model: str
    max_evidence_items: int = 12
    max_chars_per_passage: int = 1200

    def _build_prompts(self, lead: Lead, icp: ICPProfile, evidence_rows: list[Evidence]) -> tuple[str, str]:
        system_prompt = (
            "You are a strict B2B lead qualification analyst. "
            "Evaluate the lead strictly using the provided evidence passages. "
            "Do not invent facts not supported by the evidence. "
            "Respond ONLY with a valid JSON object containing keys: "
            "score (integer 0-100), verdict ('QUALIFIED', 'REVIEW', or 'NOT_QUALIFIED'), "
            "summary (string), matched_criteria (list of strings), "
            "missing_or_risk_signals (list of strings), cited_urls (list of strings)."
        )

        formatted_blocks: list[str] = []
        for idx, row in enumerate(evidence_rows, start=1):
            clipped = row.passage[: self.max_chars_per_passage]
            formatted_blocks.append(f"[{idx}] URL: {row.source_url}\nPassage: {clipped}")

        evidence_section = "\n\n".join(formatted_blocks) if formatted_blocks else "(No stored evidence available.)"

        user_prompt = (
            f"ICP Name: {icp.name}\n"
            f"Target Industries: {', '.join(icp.target_industries) or 'Any'}\n"
            f"Required Signals: {', '.join(icp.required_signals) or 'None specified'}\n"
            f"Disqualifiers: {', '.join(icp.disqualifiers) or 'None specified'}\n"
            f"Minimum Qualification Score: {icp.min_score_threshold}\n\n"
            f"Lead Company: {lead.company_name}\n"
            f"Lead Domain: {lead.target_domain}\n\n"
            f"Stored Evidence:\n{evidence_section}\n"
        )
        return system_prompt, user_prompt

    def score_lead(
        self,
        session: Session,
        *,
        lead_id: uuid.UUID,
        icp: ICPProfile,
    ) -> QualificationAssessment:
        """Load stored Evidence for a lead, score against ICP, and return a validated assessment."""
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

        system_prompt, user_prompt = self._build_prompts(lead, icp, evidence_rows)
        raw_output = self.provider.complete_json(
            model=self.model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        payload = _extract_json_object(raw_output)

        try:
            return QualificationAssessment(
                lead_id=lead_id,
                icp_name=icp.name,
                model_used=self.model,
                score=payload["score"],
                verdict=payload["verdict"],
                summary=payload["summary"],
                matched_criteria=payload.get("matched_criteria", []),
                missing_or_risk_signals=payload.get("missing_or_risk_signals", []),
                cited_urls=payload.get("cited_urls", []),
            )
        except Exception as exc:
            raise QualificationParseError(f"Invalid qualification fields: {exc}") from exc