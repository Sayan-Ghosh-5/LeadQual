"""
Qualification engine — orchestrates ICP loading, prompt assembly,
and model invocation.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from leadqual.qualification.model_client import ModelResponse, call_model_with_retry

logger = logging.getLogger(__name__)

# Default path relative to the project root
DEFAULT_ICP_PATH = Path("icp/v1.yaml")


class ICPQualifyEngine:
    """Loads an ICP definition and qualifies leads against it."""

    def __init__(self, icp_path: str | Path | None = None) -> None:
        self.icp_path = Path(icp_path) if icp_path else DEFAULT_ICP_PATH
        self._icp_yaml: str | None = None

    # ── ICP loading ─────────────────────────────────────────────

    def load_icp(self) -> str:
        """Read and validate the ICP YAML file. Returns the raw YAML string."""
        if not self.icp_path.exists():
            raise FileNotFoundError(
                f"ICP file not found: {self.icp_path.resolve()}"
            )

        raw = self.icp_path.read_text(encoding="utf-8")

        # Quick sanity check — make sure it parses
        data = yaml.safe_load(raw)
        if not data or "criteria" not in data:
            raise ValueError(
                f"ICP file missing 'criteria' key: {self.icp_path}"
            )

        self._icp_yaml = raw
        logger.info(
            "Loaded ICP '%s' with %d criteria",
            data.get("name", "unnamed"),
            len(data["criteria"]),
        )
        return raw

    @property
    def icp_yaml(self) -> str:
        """Lazy-load the ICP on first access."""
        if self._icp_yaml is None:
            self.load_icp()
        return self._icp_yaml  # type: ignore[return-value]  # ty: ignore[invalid-return-type]

    # ── Qualification ───────────────────────────────────────────

    def qualify(
        self,
        company_name: str,
        lead_text: str,
    ) -> ModelResponse:
        """
        Evaluate a lead against the loaded ICP.

        Parameters
        ----------
        company_name : str
            Human-readable company name (used for display / logging).
        lead_text : str
            Free-text description of the company to evaluate.

        Returns
        -------
        ModelResponse
            Parsed verdict + token usage.
        """
        logger.info("Qualifying lead: %s", company_name)

        # Build a rich context block for the LLM
        lead_context = (
            f"Company Name: {company_name}\n\n"
            f"Company Details:\n{lead_text}"
        )

        response = call_model_with_retry(
            lead_context=lead_context,
            criteria_yaml=self.icp_yaml,
        )

        logger.info(
            "Result for %s: fit=%s confidence=%.2f tokens=%d",
            company_name,
            response.verdict.overall_fit,
            response.verdict.confidence_score,
            response.total_tokens,
        )
        return response