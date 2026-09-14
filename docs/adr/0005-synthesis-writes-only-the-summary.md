# Synthesis writes only the summary; the landscape is assembled from run state

By the time the router asks for synthesis, everything a `BuyerLandscape` holds
except its summary already exists and has already been validated: the
`TargetProfile` from profiling, and both buyer lists from sourcing and any
deepen-research rounds. The synthesis agent used to hand all of it back — its
output type was the whole `BuyerLandscape` — so the one step meant to *add* a
summary re-typed the profile and every `BuyerCandidate` with its rationale,
signals and sources.

The second live run showed the cost. With twelve candidates in run state, that
answer took longer to generate than the 120-second request timeout allowed,
and every retry was cut off the same way. It was also the step's weakest
guarantee: "do not invent buyers" lived in the prompt, and nothing checked that
the buyers coming back were the buyers that went in.

**The synthesis agent's output is a `LandscapeSummary` — the summary and
nothing else.** `supervisor.py` assembles the `BuyerLandscape` from `RunState`:
the profile as profiled, each buyer list ranked by `fit_score` (highest first,
ties in sourced order), and the model's summary. The model is shown the buyers
in that same ranked order, so the summary is written against the landscape it
will sit on top of.

**Synthesis refuses to run on a state that cannot become a landscape.** No
profile, or fewer buyers than `MIN_LANDSCAPE_BUYERS`, raises before any model
call. The minimum is one constant shared with `BuyerLandscape`'s own validator,
so the pre-check and the schema cannot drift apart.

## Considered options

- **Keep the full-landscape output and stream the request, or raise the
  timeout.** Rejected: it makes the slow answer finish rather than removing
  it, keeps paying output tokens to copy data the run already holds, and still
  trusts the prompt to leave the buyers alone.
- **Have the model return a ranking — buyer names in order — alongside the
  summary.** Rejected for now: the instruction was always "rank by fit_score",
  which code applies exactly, and a name list reintroduces the matching and
  renaming failures that deepen-research had to guard against
  (docs/adr/0004-deepen-research-returns-verdicts.md). If ranking ever needs
  judgment beyond `fit_score`, that is the place to revisit.
- **Keep the full-landscape output and validate that its buyers match run
  state.** Rejected: it would catch an altered buyer but not the time and
  tokens spent producing one.
