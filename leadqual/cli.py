"""
Command-line interface for the LeadQual qualification engine.

Usage:
    python -m leadqual qualify --company "Acme Health" --text-file sample.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from leadqual.qualification.model_client import ModelExecutionError
from leadqual.qualification.qualifier import ICPQualifyEngine


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="leadqual",
        description="AI-powered lead qualification against your ICP.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # ── qualify subcommand ──────────────────────────────────────
    qual = subparsers.add_parser(
        "qualify",
        help="Qualify a single lead against the ICP.",
    )
    qual.add_argument(
        "--company",
        required=True,
        help="Company name (e.g. 'Acme Health').",
    )
    qual.add_argument(
        "--text-file",
        required=True,
        type=Path,
        help="Path to a text file containing the lead description.",
    )
    qual.add_argument(
        "--icp",
        default=None,
        help="Path to ICP YAML (default: icp/v1.yaml).",
    )
    return parser


def _cmd_qualify(args: argparse.Namespace) -> None:
    """Execute the qualify subcommand."""

    # ── Read lead text ──────────────────────────────────────────
    text_path: Path = args.text_file
    if not text_path.exists():
        print(f"Error: text file not found: {text_path.resolve()}", file=sys.stderr)
        sys.exit(1)

    lead_text = text_path.read_text(encoding="utf-8").strip()
    if not lead_text:
        print("Error: text file is empty.", file=sys.stderr)
        sys.exit(1)

    # ── Run qualification ───────────────────────────────────────
    engine = ICPQualifyEngine(icp_path=args.icp)

    try:
        response = engine.qualify(
            company_name=args.company,
            lead_text=lead_text,
        )
    except ModelExecutionError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        sys.exit(1)

    # ── Output ──────────────────────────────────────────────────
    output = {
        **response.verdict.model_dump(),
        "token_usage": {
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "total_tokens": response.total_tokens,
        },
    }

    print(json.dumps(output, indent=2))


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "qualify":
        _cmd_qualify(args)


if __name__ == "__main__":
    main()