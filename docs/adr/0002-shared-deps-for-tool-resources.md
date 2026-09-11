# Tools receive shared resources via deps_type, not module globals

Every research agent (`profiler_agent`, `strategic_agent`, `sponsor_agent`)
needs its tools to share one `httpx.AsyncClient` (configured once with a
timeout and tenacity retry) and to log with the run's `run_id` — a
requirement from CLAUDE.md's production-hygiene conventions. We chose
Pydantic AI's `deps_type`/`RunContext` dependency-injection mechanism over a
module-level global client: a single `Deps` dataclass is passed to each
agent's `run_sync`, and every tool function reaches shared resources via
`ctx.deps` instead of importing a global. This is the pattern the whole
`tools/` layer will be written against, so changing it later means touching
every tool function's signature.

Tool functions live in `tools/` as plain typed functions and get attached to
each agent via constructor `tools=[...]`, not inline `@agent.tool`
decorators — this keeps a tool like `edgar_search` reusable across
`profiler_agent` and the specialist agents rather than bound to one.
