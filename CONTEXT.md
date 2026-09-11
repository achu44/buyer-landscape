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
A bounded re-run of a specialist agent against only its current low-confidence
`BuyerCandidate`s, asking it to confirm or refute each with new evidence. The
agent's returned candidates replace the corresponding low-confidence entries
in place — deepening never appends duplicates.
