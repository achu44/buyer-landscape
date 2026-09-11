# Tool failures surface to the model as ToolFailed or ModelRetry

A research tool can fail in two materially different ways, and the pattern
set here is the one every tool in `tools/` follows. The shared HTTP client
already retries transient failures (docs/adr/0002-shared-deps-for-tool-resources.md),
so a tool only ever sees failures that retrying has not fixed:

- **The model caused it and can fix it** — a malformed query EDGAR rejects
  with a 400. Raise `ModelRetry`: the model gets a correction prompt and
  rephrases.
- **The model cannot fix it** — the source is down after the retry budget is
  spent, blocked (403), or answering with something unparseable. Raise
  `ToolFailed`: the model sees the failure, does not burn its retry budget
  re-issuing the same call, and sources evidence elsewhere.

Neither case crashes the run, which keeps CLAUDE.md's "failures degrade,
never crash" rule intact one level lower than the supervisor: a dead source
costs the landscape some citations, not the analysis.

A malformed *record within* a successful response is a third case, handled
without an exception at all — it is logged and skipped, and the next result
takes its place, so one odd filing costs one source rather than the search.

## Considered options

- **Return an empty result set on failure.** Rejected: the model cannot tell
  "EDGAR is down" from "no company matches", and would wrongly conclude the
  evidence does not exist.
- **Let `httpx` exceptions propagate to the supervisor's `except` block.**
  Rejected: it aborts the whole agent run — including the buyers it had
  already sourced — over one unavailable source, and the router sees only a
  failed step rather than a partial answer.
- **`ModelRetry` for everything.** Rejected: re-issuing a call that just
  failed four times at the transport layer spends the retry budget on a
  source that is down.
