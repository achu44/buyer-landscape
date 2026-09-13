# Buyer Landscape Analysis

Given a target company, produces a ranked buyer landscape (strategic acquirers
and financial sponsors) and writes it back to a mock CRM shaped like
Salesforce's Account/Opportunity/Note objects.

## Language

**Account**:
The target company being profiled — one per run, built from `TargetProfile`.

**Opportunity**:
A potential transaction between the target `Account` and one specific buyer.
One `Opportunity` per `BuyerCandidate`, not one per run — `fit_score` maps to
probability, `confidence` to stage.
_Avoid_: "engagement" (that's the run as a whole, not an Opportunity)

**Note**:
Free text attached to an `Opportunity`, holding that buyer's `rationale` and
`signals`. The overall banker-readable summary is a `Note` on the `Account`.

**Deepen-research round**:
One bounded pass over *every* buyer list that currently holds low-confidence
`BuyerCandidate`s, asking a deepen agent to confirm or refute each of them
against fresh evidence. A confirmed candidate replaces its low-confidence
entry in place — deepening never appends duplicates. A refuted candidate, or
one the pass never reported on, is dropped. The cap counts rounds, not agent
runs: a round covers both lists.
_Avoid_: "retry" (that's a failed step run again, not a shaky answer firmed up)
