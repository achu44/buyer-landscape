# Buyer Landscape Analysis — multi-agent system

Given a target company, produce a ranked buyer landscape (strategic acquirers
and financial sponsors) with per-buyer rationale and fit scores, as validated
JSON plus a banker-readable memo, written to a mock CRM.

Demonstrates: LLM tool use with typed schemas, dynamic routing (LLM output
determines the next step), structured output validation with retry-on-failure,
and production hygiene — logging with run IDs, bounded loops, retries with
backoff, graceful degradation.

## Setup (macOS)

```bash
# 1. Project env (uv: https://docs.astral.sh/uv/)
uv init --python 3.12
uv add pydantic pydantic-ai tenacity httpx yfinance

# 2. API key
export ANTHROPIC_API_KEY=...   # put in ~/.zshrc or a .env you don't commit

# 3. Identify yourself to SEC EDGAR (it throttles anonymous traffic)
export EDGAR_USER_AGENT="your-project-name your.email@example.com"

# 4. Web search key (optional; free tier at
#    https://api-dashboard.search.brave.com/). Without it the run still
#    works, sourcing evidence from EDGAR alone.
export BRAVE_SEARCH_API_KEY=...

# 5. Claude Code (if not installed)
npm install -g @anthropic-ai/claude-code   # or: brew install claude-code

# 6. Start coding
git init && claude
```

Optional — give Claude Code searchable Pydantic AI docs via MCP
(run in your terminal, not inside a claude session):

```bash
claude mcp add --transport http <name> <docs-mcp-url>
claude mcp list   # verify
```

Or skip MCP and just point Claude Code at https://ai.pydantic.dev/llms.txt
(already referenced in CLAUDE.md).

## Viewing past runs

Every run leaves two kinds of history, both gitignored:

- **`runs/<run_id>.json`** — the run's final state: steps taken, errors,
  profile, buyer lists, landscape, `started_at` / `finished_at`. Written for
  every run, including one that fails before reaching the CRM. Override the
  directory with `RUNS_DIR`.
- **`crm.db`** — the mock CRM: one `accounts` row per completed run (dated by
  `created_at`), one `opportunities` row per buyer, and `notes` holding each
  buyer's rationale and the landscape summary. Override with `CRM_DB_PATH`.

```bash
# Every run, newest first: when, how it ended, how many errors
for f in $(ls -t runs/*.json); do
  jq -r '[.run_id, .started_at, (.steps_taken | last // "none"), (.errors | length)] | @tsv' "$f"
done

# One run's route and errors
jq '{steps_taken, errors, started_at, finished_at}' runs/<run_id>.json

# One run's ranked buyers
jq -r '.landscape | (.strategic_buyers + .sponsor_buyers)[] | [.buyer_type, .fit_score, .confidence, .name] | @tsv' runs/<run_id>.json

# CRM: completed runs, then one run's opportunities and summary
sqlite3 -column -header crm.db "SELECT run_id, name, created_at FROM accounts ORDER BY id DESC"
sqlite3 -column -header crm.db "SELECT buyer_name, buyer_type, fit_score, stage FROM opportunities WHERE run_id='<run_id>' ORDER BY fit_score DESC"
sqlite3 crm.db "SELECT body FROM notes WHERE parent_type='account' AND run_id='<run_id>'"
```

## Layout

```
schemas.py      # Pydantic data contracts (Day 1)
supervisor.py   # router agent + deterministic dispatch loop (Day 1)
tools/          # EDGAR, web search, comps, mock-CRM writer (Day 2)
evals/          # golden set + judges (Day 5)
CLAUDE.md       # project conventions for Claude Code
```

## Week plan

1. Schemas, mock CRM tables, repo scaffolding
2. Tools with retries, timeouts, structured logging
3. Specialist agents (profiler, strategic, sponsor)
4. Supervisor loop end-to-end on one target
5. Eval harness + README architecture diagram
6. Observability polish (Logfire traces), failure-mode demos
7. Write-up of design decisions
