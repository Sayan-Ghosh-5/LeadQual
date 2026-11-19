# LeadQual — AI-Powered Lead Qualification Engine

LeadQual is a command-line tool that uses large language models to automatically score sales leads against your **Ideal Customer Profile (ICP)**. You define your criteria in a simple YAML file, feed in a company description, and get back a structured JSON verdict with per-criterion scores, reasoning, and token usage.

---

## Quick Start

```bash
# 1. Clone and install
pip install -e ".[dev]"

# 2. Configure your API key
cp .env.example .env
# Edit .env -> add your Groq API key from https://console.groq.com/keys

# 3. Qualify a lead
python -m leadqual qualify --company "Acme Health" --text-file sample.txt
```

---

## Project Structure

```text
LEADqual/
├── pyproject.toml              # Project metadata & dependencies
├── .env.example                # Template for environment variables
├── .env                        # Your local secrets (gitignored)
├── .gitignore
├── sample.txt                  # Example lead description (positive match)
├── bad_lead.txt                # Example lead description (negative match)
├── icp/
│   └── v1.yaml                 # ICP criteria (editable, versioned)
├── leadqual/
│   ├── __init__.py
│   ├── __main__.py             # Enables `python -m leadqual`
│   ├── cli.py                  # CLI argument parsing & output
│   ├── config.py               # Centralized settings (pydantic-settings)
│   └── qualification/
│       ├── __init__.py
│       ├── schemas.py          # Pydantic output contracts
│       ├── model_client.py     # LLM provider wrapper
│       └── qualifier.py        # Orchestration engine
└── tests/
    ├── __init__.py
    ├── conftest.py             # Shared test fixtures
    └── test_model_client.py    # Unit tests (no live API calls)
```

---

## Phase 1 — What We Built and Why

### Step 1: Project Foundation (`pyproject.toml`)

- **What:** Created a PEP 621-compliant `pyproject.toml` with Hatchling as the build backend, core runtime dependencies, and optional dependency groups (`dev`, `openai`, `google`).
- **Why:**
  - **Single source of truth:** Replaces the legacy mess of `setup.py`, `setup.cfg`, and `requirements.txt` with one declarative file.
  - **Hatchling:** Chosen over setuptools because it is lightweight, PEP 621-native, and requires zero boilerplate configuration.
  - **Optional dependency groups:** (`pip install -e ".[dev]"`) keep production installs lean while giving developers everything they need in one command.
  - **Provider extras:** (`[openai]`, `[google]`) let us swap LLM backends without forcing every user to install every SDK.

### Step 2: Configuration (`config.py` + `.env`)

- **What:** Built a `Settings` class using `pydantic-settings` that reads `MODEL_API_KEY`, `MODEL_NAME`, `MODEL_TIMEOUT_SECONDS`, and `MODEL_MAX_RETRIES` from environment variables and a `.env` file.
- **Why:**
  - **Security:** API keys live in `.env` (gitignored), never in source code.
  - **Fail-fast validation:** `MODEL_API_KEY` has no default value, so Pydantic raises a clear `ValidationError` at startup if it is missing — no silent failures 200 lines deep in the code.
  - **Type safety:** `MODEL_TIMEOUT_SECONDS` is guaranteed to be a float, `MODEL_MAX_RETRIES` an int. No runtime surprises.
  - **Environment switching:** Different `.env` files for dev/staging/prod with zero code changes.

### Step 3: ICP Definition (`icp/v1.yaml`)

- **What:** Created a versioned YAML file defining four scoring criteria (industry, company size, tech stack, geography), each with an explicit ID, human-readable description, and numeric weight.
- **Why:**
  - **Separation of concerns:** The scoring rubric lives outside the code. Sales and marketing teams can edit criteria without opening a Python file.
  - **Versioning:** The file is named `v1.yaml`, not `icp.yaml`. When criteria change, we create `v2.yaml` and test it side-by-side before switching. The old version is preserved for rollback.
  - **Retargeting without code changes:** To pivot from healthcare to fintech, you edit four lines of YAML. The Python codebase is completely generic — it reads whatever criteria the file contains.
  - **Weights allow nuanced scoring:** Industry match (1.0) matters more than geography (0.4), and the LLM respects that hierarchy.

### Step 4: Pydantic Schemas (`schemas.py`)

- **What:** Defined three strict data models:
  - `CriterionStatus` enum: `MET`, `NOT_MET`, `INSUFFICIENT_EVIDENCE`
  - `CriterionResult`: per-criterion evaluation with reasoning
  - `QualificationVerdict`: final result with confidence score (`0.0`–`1.0`)
- **Why:**
  - **LLMs are unreliable narrators:** Without schemas, the model might return "yes", "maybe", "probably a good fit", or free-text paragraphs. The enum forces exactly three outcomes — nothing else.
  - **Downstream safety:** Any code that consumes a `QualificationVerdict` can trust that `confidence_score` is between 0.0 and 1.0, that `criteria_results` is a list, and that every field exists. Pydantic guarantees this at parse time.
  - **Validation:** Strict field constraints catch model hallucinations early.
  - **Forward compatibility:** `evidence_ids` is an empty list in Phase 1, reserving the slot for Phase 2 evidence collection without breaking the schema later.

### Step 5: Model SDK Wrapper (`model_client.py`)

- **What:** Built `call_model_with_retry()` — the single function that initializes the Groq client, enforces timeouts, requests structured JSON output, parses the response into `QualificationVerdict`, retries once on failure, and tracks token usage.
- **Why:**
  - **Single point of contact:** No other file in the codebase imports the Groq SDK directly. If we switch to OpenAI or Gemini, we change one file.
  - **Timeout enforcement:** Prevents the app from hanging indefinitely when the API is slow. The default 10-second timeout is configurable via `.env`.
  - **Structured output:** `response_format={"type": "json_object"}` tells Groq to constrain its output to valid JSON, dramatically reducing parse failures.
  - **Resilient retry policy:** Exactly one retry balances resilience against cost. Retrying on transient failures (malformed JSON, network blips) is smart; retrying five times on a bad prompt wastes tokens and money.
  - **Explicit control:** `max_retries=0` on the Groq client disables the SDK's built-in retry so we control the retry logic ourselves (including parse failures, which the SDK doesn't know about).
  - **Clean exceptions:** `ModelExecutionError` is a custom exception that gives callers a clean, predictable error type instead of leaking Groq SDK internals.
  - **Token tracking:** (`ModelResponse.prompt_tokens`, `completion_tokens`) enables cost monitoring from day one.

> **Decision — Groq as the LLM provider:**
> - Free tier with no credit card required.
> - Extremely fast inference (Groq's LPU hardware).
> - OpenAI-compatible API, making future migration trivial.
> - **Model selection:** We initially targeted `llama-3.1-70b-versatile`, but Groq decommissioned it during development. After querying the available model catalog via `client.models.list()`, we selected `qwen/qwen3.8-27b` — a capable 27B-parameter model available on the free tier. This reinforced the value of storing the model name in `.env` rather than hardcoding it: the switch was a one-line config change.

### Step 6: Qualification Engine & CLI (`qualifier.py` + `cli.py`)

- **What:** Built `ICPQualifyEngine` (loads YAML, assembles the prompt, calls the model wrapper) and a CLI using `argparse` that exposes:
  ```bash
  python -m leadqual qualify --company "Acme Health" --text-file sample.txt
  ```
- **Why:**
  - **Reusable engine:** `ICPQualifyEngine` encapsulates the full qualification flow so it can be reused from a CLI, a web API, a batch processor, or a test — without duplicating logic.
  - **Lazy ICP loading:** (`@property`) means the YAML is read once and cached, even if `qualify()` is called multiple times.
  - **Standard library CLI:** `argparse` was chosen over `typer` to avoid an extra dependency for a simple CLI. If the CLI grows complex (multiple subcommands, rich output), migrating to typer later is straightforward.
  - **Standard package invocation:** `__main__.py` enables the `python -m leadqual` invocation pattern, which is the Pythonic standard for runnable packages.
  - **Composability:** JSON output to stdout makes the tool composable — pipe it to `jq`, redirect to a file, or consume it from another program.

### Step 7: Unit Tests (`test_model_client.py`)

- **What:** Wrote three pytest tests using `unittest.mock.patch` to simulate:

| Test | Scenario | Verified |
| :--- | :--- | :--- |
| **Happy Path** | Valid JSON on first call | Correct parsing, token tracking, 1 API call |
| **Retry Path** | Invalid JSON -> valid JSON | Retry fires once, succeeds, 2 API calls |
| **Fatal Path** | Invalid JSON twice | `ModelExecutionError` raised, exactly 2 calls, no infinite loop |

- **Why:**
  - **Zero cost & offline:** Tests use `@patch` to replace the Groq client with mocks. They run in ~0.33 seconds, cost \$0, and work offline.
  - **CI-safe:** `conftest.py` sets a dummy `MODEL_API_KEY` so tests pass even on machines without a `.env` file (e.g., CI runners).
  - **Deterministic:** `side_effect` lists simulate multi-call scenarios (fail-then-succeed) without complex state management.
  - **Loop prevention:** Testing the retry boundary (exactly 2 calls, not 3) prevents the most dangerous bug in retry logic: infinite loops that burn through API quotas.

---

## Key Design Decisions Summary

| Decision | Choice | Rationale |
| :--- | :--- | :--- |
| **Config format** | `.env` + `pydantic-settings` | Type-safe, fail-fast, gitignore-friendly |
| **ICP storage** | Versioned YAML (`v1.yaml`) | Editable by non-developers, rollback-safe |
| **Output validation** | Pydantic models with field constraints | Catches LLM hallucinations at parse time |
| **LLM provider** | Groq (free tier) | Fast, free, OpenAI-compatible |
| **LLM model** | `qwen/qwen3.8-27b` | High capability on free tier, easily swapped in `.env` |
| **JSON mode** | `response_format={"type": "json_object"}` | Native Groq support, fewer parse failures |
| **Retry strategy** | Exactly 1 retry | Resilient to transient errors, cost-conscious |
| **CLI framework** | `argparse` (stdlib) | Zero extra dependencies |
| **Test strategy** | Mocked Groq client | Fast, free, deterministic, CI-safe |
| **Build backend** | Hatchling | Lightweight, PEP 621-native |

---

## Running Tests

```bash
pytest tests/ -v
```

Expected output:
```text
tests/test_model_client.py::TestCallModelWithRetry::test_happy_path_returns_valid_verdict PASSED
tests/test_model_client.py::TestCallModelWithRetry::test_retry_succeeds_on_second_attempt PASSED
tests/test_model_client.py::TestCallModelWithRetry::test_fatal_failure_after_all_retries PASSED

============================== 3 passed in 0.33s ===============================
```

---

## Changing Your ICP (No Code Required)

Edit `icp/v1.yaml` — or create `icp/v2.yaml` and pass it via `--icp`:

```bash
python -m leadqual qualify --company "New Lead" --text-file lead.txt --icp icp/v2.yaml
```

