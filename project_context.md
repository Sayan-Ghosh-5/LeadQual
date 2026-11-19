# LeadQual: Phase 1 — Model-Powered Qualifier

This document outlines the architecture, components, and design decisions completed during Phase 1 of the LeadQual project.

---

## 1. Executive Summary

LeadQual is an automated sales intelligence tool designed to evaluate inbound leads against an organization's Ideal Customer Profile (ICP). Rather than relying on subjective manual reviews or brittle keyword searches, LeadQual leverages an LLM to evaluate unstructured company descriptions against structured, weighted rubric criteria and produce deterministic, validated JSON decisions.

---

## 2. Phase 1 Deliverables (What We Built)

### 1. Project Scaffold & Dependencies (`pyproject.toml`)
- Initialized a Python 3.12+ project with a clean, modular package structure following modern PEP 621 standards.
- Configured Hatchling as the lightweight build backend.
- Configured core runtime dependencies (`pydantic>=2.0`, `pydantic-settings`, `httpx`, `pyyaml`, `groq`) alongside developer extras (`pytest`, `pytest-mock`, `ruff`).

### 2. Centralized Configuration (`leadqual/config.py`)
- Implemented a `Settings` class using `pydantic-settings` to safely load environment variables (`MODEL_API_KEY`, `MODEL_NAME`, `MODEL_TIMEOUT_SECONDS`, and `MODEL_MAX_RETRIES`) from a `.env` file and system environment.
- Configured fail-fast behavior for the mandatory API key and safe defaults for timeouts and retries, while ignoring extraneous variables (`extra="ignore"`).

### 3. Externalized ICP Definition (`icp/v1.yaml`)
- Decoupled target customer criteria from the codebase into a versioned YAML specification.
- Defines clear evaluation criteria (industry, size, tech stack, geography), each assigned a unique identifier, description, and numeric weight (0.0 to 1.0).

### 4. Strict Data Contracts (`leadqual/qualification/schemas.py`)
- Built robust Pydantic v2 data models defining the exact interface for LLM output:
  - `CriterionStatus` Enum: `MET`, `NOT_MET`, `INSUFFICIENT_EVIDENCE`.
  - `CriterionResult`: Individual criterion evaluations with one-sentence reasoning and reserved evidence slots.
  - `QualificationVerdict`: Top-level structured verdict including company name, binary overall fit, confidence score (clamped between 0.0 and 1.0), list of per-criterion evaluations, and an executive summary.

### 5. Encapsulated Model Wrapper (`leadqual/qualification/model_client.py`)
- Created an isolated wrapper around the Groq SDK (`call_model_with_retry`).
- Enforces strict JSON response mode (`response_format={"type": "json_object"}`), zero temperature for determinism, and timeout handling.
- Embeds the auto-generated JSON schema into the prompt to guarantee structural conformance.
- Implements custom exception wrapping (`ModelExecutionError`) and precise single-retry logic to handle transient failures cleanly.
- Captures token usage metadata (`prompt_tokens`, `completion_tokens`, `total_tokens`).

### 6. Qualification Engine (`leadqual/qualification/qualifier.py`)
- Created `ICPQualifyEngine` to orchestrate ICP file loading, validation, context prompt construction, model invocation, and response parsing.
- Features lazy-loading and caching of the ICP specification file to avoid redundant disk I/O.

### 7. Command-Line Interface (`leadqual/cli.py` & `leadqual/__main__.py`)
- Implemented an `argparse`-powered CLI exposing the `qualify` subcommand with `--company`, `--text-file`, and optional `--icp` arguments.
- Formats and outputs the complete qualification verdict and token usage as formatted JSON to stdout.

### 8. Mock-First Unit Testing (`tests/test_model_client.py` & `tests/conftest.py`)
- Implemented test suites using `pytest` and `unittest.mock.patch` covering:
  - Happy Path (valid JSON on first attempt)
  - Single Retry Path (malformed JSON on attempt 1, valid JSON on attempt 2)
  - Fatal Failure Path (invalid JSON on all retries, verifying loop termination)
- Guaranteed offline, zero-cost, and deterministic test runs without live API calls.

---

## 3. Key Architectural Decisions (Why We Built It This Way)

### Externalizing the ICP (YAML)
- **Why:** Hardcoding target profiles into source code makes updating criteria or pivoting markets rigid. By loading criteria from an external YAML file, the system can pivot to a completely new industry (e.g., healthcare software to EdTech) with zero code changes.
- **Versioning:** Files like `icp/v1.yaml` and `icp/v2.yaml` allow testing revisions side-by-side and enable instant rollback.

### Strict Pydantic Schemas & JSON Mode
- **Why:** LLMs are inherently unstructured text generators. Without strong constraints, model drift introduces unexpected text or inconsistent keys. Generating a JSON schema from Pydantic and enforcing JSON mode ensures type safety and predictable downstream integration.

### Encapsulating the Model SDK
- **Why:** Direct SDK calls scattered across business logic lead to tight coupling. Isolating all Groq client setup, calls, and conversions into `model_client.py` ensures switching providers (e.g., to Anthropic, OpenAI, or Google) requires modifying only one file.

### Controlled Retry Logic
- **Why:** While transient network or formatting issues can occur, uncontrolled retries can lead to infinite loops and budget exhaustion. Restricting retry attempts to exactly one balances operational resilience with cost control.

### Mock-First Unit Testing
- **Why:** Relying on live network calls during CI/CD introduces flakiness, rate limits, latency, and monetary cost. Mocking the SDK client guarantees sub-second, deterministic test runs anywhere.

---

## 4. End-to-End Component Flow

```
User (CLI Command)
      │
      ▼
leadqual/cli.py ───────────────────► Reads lead text file
      │
      ▼
leadqual/qualification/qualifier.py ──► Reads & validates icp/v1.yaml
      │
      ▼
leadqual/qualification/model_client.py
      │   ├── Loads settings from leadqual/config.py
      │   ├── Generates schema from leadqual/qualification/schemas.py
      │   ├── Sends prompt to Groq API (JSON mode, temperature 0.0)
      │   └── Handles retries & validates response
      │
      ▼
Returns validated ModelResponse
      │
      ▼
CLI prints formatted JSON verdict to stdout
```

