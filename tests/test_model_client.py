"""
Unit tests for leadqual.qualification.model_client

All tests mock the Groq SDK — no live network calls are made.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from leadqual.qualification.model_client import (
    ModelExecutionError,
    ModelResponse,
    call_model_with_retry,
)
from leadqual.qualification.schemas import QualificationVerdict

# ── Helpers ─────────────────────────────────────────────────────────

VALID_VERDICT_JSON = json.dumps(
    {
        "company_name": "Test Corp",
        "overall_fit": True,
        "confidence_score": 0.92,
        "criteria_results": [
            {
                "criterion_id": "crit_industry",
                "status": "MET",
                "reasoning": "Test reasoning.",
                "evidence_ids": [],
            }
        ],
        "summary": "Test summary.",
    }
)

INVALID_JSON = "this is definitely not valid json {{{"

DUMMY_CRITERIA_YAML = "criteria:\n  - id: crit_industry\n    weight: 1.0"
DUMMY_LEAD_CONTEXT = "Test Corp is a health-tech startup."


def _make_mock_response(
    content: str,
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
) -> MagicMock:
    """Build a fake Groq ChatCompletion response object."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    response.usage.prompt_tokens = prompt_tokens
    response.usage.completion_tokens = completion_tokens
    return response


# ── Tests ───────────────────────────────────────────────────────────


class TestCallModelWithRetry:
    """Tests for the call_model_with_retry() public function."""

    # ── 1. Happy Path ───────────────────────────────────────────

    @patch("leadqual.qualification.model_client._build_client")
    def test_happy_path_returns_valid_verdict(
        self, mock_build_client: MagicMock
    ) -> None:
        """A valid JSON response on the first call should parse cleanly."""

        # Arrange
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_mock_response(
            VALID_VERDICT_JSON,
            prompt_tokens=200,
            completion_tokens=80,
        )
        mock_build_client.return_value = mock_client

        # Act
        result = call_model_with_retry(
            lead_context=DUMMY_LEAD_CONTEXT,
            criteria_yaml=DUMMY_CRITERIA_YAML,
        )

        # Assert — structure
        assert isinstance(result, ModelResponse)
        assert isinstance(result.verdict, QualificationVerdict)

        # Assert — content
        assert result.verdict.company_name == "Test Corp"
        assert result.verdict.overall_fit is True
        assert result.verdict.confidence_score == 0.92
        assert len(result.verdict.criteria_results) == 1
        assert result.verdict.criteria_results[0].status.value == "MET"

        # Assert — token tracking
        assert result.prompt_tokens == 200
        assert result.completion_tokens == 80
        assert result.total_tokens == 280

        # Assert — only one API call was made (no retry needed)
        assert mock_client.chat.completions.create.call_count == 1

    # ── 2. Single Retry Path ────────────────────────────────────

    @patch("leadqual.qualification.model_client._build_client")
    def test_retry_succeeds_on_second_attempt(
        self, mock_build_client: MagicMock
    ) -> None:
        """Invalid JSON on call #1, valid JSON on call #2 → should succeed."""

        # Arrange
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            _make_mock_response(INVALID_JSON),   # call 1: garbage
            _make_mock_response(VALID_VERDICT_JSON),  # call 2: good
        ]
        mock_build_client.return_value = mock_client

        # Act
        result = call_model_with_retry(
            lead_context=DUMMY_LEAD_CONTEXT,
            criteria_yaml=DUMMY_CRITERIA_YAML,
        )

        # Assert — succeeded despite first failure
        assert isinstance(result.verdict, QualificationVerdict)
        assert result.verdict.company_name == "Test Corp"

        # Assert — exactly two API calls were made
        assert mock_client.chat.completions.create.call_count == 2

    # ── 3. Fatal Failure Path ───────────────────────────────────

    @patch("leadqual.qualification.model_client._build_client")
    def test_fatal_failure_after_all_retries(
        self, mock_build_client: MagicMock
    ) -> None:
        """Invalid JSON on both calls → should raise ModelExecutionError."""

        # Arrange
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            _make_mock_response(INVALID_JSON),   # call 1: garbage
            _make_mock_response(INVALID_JSON),   # call 2: still garbage
        ]
        mock_build_client.return_value = mock_client

        # Act & Assert
        with pytest.raises(ModelExecutionError, match="failed after 2 attempt"):
            call_model_with_retry(
                lead_context=DUMMY_LEAD_CONTEXT,
                criteria_yaml=DUMMY_CRITERIA_YAML,
            )

        # Assert — exactly two calls, then stopped (no infinite loop)
        assert mock_client.chat.completions.create.call_count == 2