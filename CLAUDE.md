# Buyer Landscape Analysis — multi-agent system

Portfolio project: given a target company, produce a ranked buyer landscape
(strategic acquirers + financial sponsors) as validated JSON + a markdown memo,
written back to a mock CRM. Built with Pydantic AI. The point is to demonstrate
production agentic patterns: typed tool use, dynamic LLM routing, structured
output validation, logging/observability, retries, and error handling.

## Architecture

- `schemas.py` — all Pydantic data contracts. Every agent output validates
  against one of these. Change schemas here first; agents follow.
- `supervisor.py` — router agent emits a `RouterDecision`; plain Python
  dispatches on the enum. Guardrails (iteration caps, deepen-research cap)
  are deterministic code, never LLM-decided.
- `tools/` (Day 2) — typed tool functions: EDGAR full-text search, web search,
  yfinance comps, mock-CRM writer (SQLite tables shaped like Salesforce
  Account/Opportunity/Note objects).
- `evals/` (Day 5) — golden set of 5–10 targets; deterministic structural
  checks + LLM-as-judge on rationale quality.

## Conventions

- Python 3.12+, `uv` for env/deps. Run: `uv run python supervisor.py`.
- Always check current Pydantic AI docs before writing agent/tool code —
  the API moves fast. Docs index: https://ai.pydantic.dev/llms.txt
- Every external call (LLM, HTTP) gets: timeout, tenacity retry with
  exponential backoff, and a log line carrying `run_id`.
- Failures degrade, never crash: append to `RunState.errors`, let the
  router see it and re-route.
- All agent outputs are Pydantic models with real constraints (ranges,
  enums, min lengths). No bare `str` outputs.
- Tests with pytest; mock LLM calls in unit tests (use Pydantic AI's
  TestModel if current docs still support it).
- No secrets in code. `ANTHROPIC_API_KEY` from env only.

## Style

- Type hints everywhere; small functions; docstrings explain *why*.
- Prefer boring, readable code over cleverness — this repo will be read
  by interviewers.
