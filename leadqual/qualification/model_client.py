"""
Groq SDK wrapper — the single point of contact with the LLM provider.

Handles:
  • Client initialisation from config
  • Timeout enforcement
  • Structured-output parsing into QualificationVerdict
  • Exactly one retry on malformed output or transient errors
  • Token-usage tracking
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from groq import APIError, Groq
from pydantic import ValidationError

from leadqual.config import settings
from leadqual.qualification.schemas import QualificationVerdict

logger = logging.getLogger(__name__)

# ── Custom exception ────────────────────────────────────────────────

class ModelExecutionError(Exception):
    """Raised when the model call fails after all retries."""


# ── Response container ──────────────────────────────────────────────

@dataclass
class ModelResponse:
    """Bundles the parsed verdict with token-usage metadata."""

    verdict: QualificationVerdict
    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


# ── Internal helpers ────────────────────────────────────────────────

def _build_client() -> Groq:
    """Create a Groq client from the centralised settings."""
    return Groq(
        api_key=settings.MODEL_API_KEY,
        timeout=settings.MODEL_TIMEOUT_SECONDS,
        max_retries=0,  # we handle retries ourselves
    )


def _build_messages(lead_context: str, criteria_yaml: str) -> list[dict[str, str]]:
    """Construct the system + user messages for the qualification prompt."""

    schema_json = json.dumps(QualificationVerdict.model_json_schema(), indent=2)

    system_prompt = (
        "You are a lead-qualification analyst. "
        "Evaluate the provided company against the ICP criteria and "
        "return a JSON object that strictly conforms to the following schema:\n\n"
        f"{schema_json}\n\n"
        "Return ONLY valid JSON — no markdown, no commentary."
    )

    user_prompt = (
        "## ICP Criteria\n"
        f"{criteria_yaml}\n\n"
        "## Lead Context\n"
        f"{lead_context}\n\n"
        "Evaluate this lead against every criterion above."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _call_and_parse(client: Groq, messages: list[dict[str, str]]) -> ModelResponse:
    """
    Make a single API call and parse the result.

    Raises ModelExecutionError on any failure so the caller can retry.
    """
    try:
        response = client.chat.completions.create(
            model=settings.MODEL_NAME,
            messages=messages,  # ty: ignore[invalid-argument-type]
            response_format={"type": "json_object"},
            temperature=0.0,
        )
    except APIError as exc:
        raise ModelExecutionError(f"Groq API error: {exc}") from exc

    # ── Extract raw text ────────────────────────────────────────
    raw_content = response.choices[0].message.content
    if not raw_content:
        raise ModelExecutionError("Model returned empty content.")

    # ── Parse into Pydantic model ───────────────────────────────
    try:
        verdict = QualificationVerdict.model_validate_json(raw_content)
    except (ValidationError, json.JSONDecodeError) as exc:
        raise ModelExecutionError(
            f"Failed to parse model output into QualificationVerdict: {exc}"
        ) from exc

    # ── Token usage ─────────────────────────────────────────────
    usage = response.usage
    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0

    return ModelResponse(
        verdict=verdict,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


# ── Public API ──────────────────────────────────────────────────────

def call_model_with_retry(
    lead_context: str,
    criteria_yaml: str,
) -> ModelResponse:
    """
    Call the LLM, parse the output, and retry **exactly once** on failure.

    Parameters
    ----------
    lead_context : str
        Free-text description of the company / lead to evaluate.
    criteria_yaml : str
        Raw YAML string of the ICP criteria (loaded from icp/v1.yaml).

    Returns
    -------
    ModelResponse
        Parsed verdict + token counts.

    Raises
    ------
    ModelExecutionError
        If both the initial call and the single retry fail.
    """
    client = _build_client()
    messages = _build_messages(lead_context, criteria_yaml)

    max_attempts = 1 + settings.MODEL_MAX_RETRIES  # 1 initial + N retries

    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            logger.info("Model call attempt %d/%d", attempt, max_attempts)
            return _call_and_parse(client, messages)
        except ModelExecutionError as exc:
            last_error = exc
            logger.warning("Attempt %d failed: %s", attempt, exc)

    # All attempts exhausted
    raise ModelExecutionError(
        f"Model call failed after {max_attempts} attempt(s)."
    ) from last_error