# A deepen-research round returns verdicts, and covers every list at once

Re-researching low-confidence `BuyerCandidate`s needs an answer to two
questions: what the pass hands back, and how much of a run one round buys.

**The pass returns a verdict per candidate, not a replacement list.** A
`DeepenedBatch` carries one `DeepenedCandidate` per name it was asked about,
each stating `confirmed` or `refuted` plus the evidence behind it, and the
rewritten `BuyerCandidate` only when confirmed. Model validators tie the two
together: a confirmation without a buyer, a refutation carrying one, or a
confirmation that renames the buyer all fail validation and go back to the
model. The supervisor holds a pass to the candidates it was handed — findings
about names nobody asked about, or a candidate left unanswered, fail the whole
pass rather than being merged — and then drops any low-confidence candidate
the pass confirmed nothing for, so the list that survives a round carries no
unevidenced names. Dropping an unreported name is the last line of defense
behind that check, not the everyday path.

**One round deepens every list that needs it.** `_deepen_research` collects
the buyer types with low-confidence candidates and runs each list's pass
inside a single round, incrementing `deepen_rounds_used` once. `RunState`
reports the combined low-confidence count to the router so the decision is
made on the number that matches what the step does. A pass that fails is
recorded in `RunState.errors` and stepped over, so one dead source does not
leave the other list unresolved.

## Considered options

- **Reuse `BuyerCandidateBatch`, treating absence as "dropped".** Rejected:
  "we could not substantiate this" and "the model forgot to mention it" would
  be the same response, and the batch's `min_length=1` would make a round
  that refutes everything — a real and useful answer — fail validation.
- **A three-way verdict with "still uncertain".** Rejected: the round exists
  to decide whether a name stays, and a third value hands the decision back
  to the next round, which the cap may never grant.
- **One round per buyer list.** Rejected: the cap would then buy a different
  amount of research depending on how many lists happened to come back shaky,
  and a run with both lists low-confidence would spend its whole budget on
  one deepen apiece rather than getting two attempts at each.
