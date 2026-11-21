"""YAML configuration loader for ICPProfile definitions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from leadqual.qualification.scorer import ICPProfile, QualificationError


class ICPConfigError(QualificationError, ValueError):
    """Raised when an ICP YAML file is missing, unreadable, or fails schema validation."""


def _parse_yaml_mapping(raw_text: str) -> dict[str, Any]:
    """Parse YAML using PyYAML when installed, with a lightweight fallback for simple configs."""
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError:
        yaml = None

    if yaml is not None:
        try:
            data = yaml.safe_load(raw_text)
        except Exception as exc:
            raise ICPConfigError(f"Invalid YAML syntax: {exc}") from exc
    else:
        data = _minimal_yaml_fallback(raw_text)

    if not isinstance(data, dict):
        raise ICPConfigError("Top-level ICP YAML document must be a mapping/dictionary.")
    return data


def _minimal_yaml_fallback(raw_text: str) -> dict[str, Any]:
    """Minimal fallback parser for flat keys and simple lists if PyYAML is not installed."""
    result: dict[str, Any] = {}
    current_list_key: str | None = None

    for raw_line in raw_text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if stripped.startswith("- ") and current_list_key is not None:
            item = stripped[2:].strip().strip('"').strip("'")
            result[current_list_key].append(item)
            continue

        if ":" in stripped:
            key, val = stripped.split(":", 1)
            key = key.strip()
            val = val.strip()
            if not val:
                result[key] = []
                current_list_key = key
            else:
                current_list_key = None
                unquoted = val.strip('"').strip("'")
                if unquoted.isdigit():
                    result[key] = int(unquoted)
                else:
                    result[key] = unquoted

    return result


def load_icp_from_yaml(path: str | Path) -> ICPProfile:
    """Load and validate an ICPProfile from a YAML file."""
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        raise ICPConfigError(f"ICP YAML file not found: '{file_path}'")

    try:
        raw_text = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ICPConfigError(f"Failed to read ICP YAML file '{file_path}': {exc}") from exc

    raw_data = _parse_yaml_mapping(raw_text)

    # Support optional wrapping under an `icp:` root key
    mapping = raw_data["icp"] if "icp" in raw_data and isinstance(raw_data["icp"], dict) else raw_data

    try:
        return ICPProfile(**mapping)
    except Exception as exc:
        raise ICPConfigError(f"Invalid ICPProfile fields in '{file_path}': {exc}") from exc