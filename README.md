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

# 4. Claude Code (if not installed)
npm install -g @anthropic-ai/claude-code   # or: brew install claude-code

# 5. Start coding
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
