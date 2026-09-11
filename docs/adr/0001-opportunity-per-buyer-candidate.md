# Opportunity maps to one BuyerCandidate, not one engagement

The mock CRM's `Opportunity` table could plausibly represent either the whole
sell-side engagement (one row per run) or a single potential deal with one
buyer (one row per `BuyerCandidate`). We chose the latter: it matches how
banks actually use Salesforce — an Opportunity tracks a deal with a specific
counterparty — and gives `WRITE_TO_CRM` a natural one-row-per-candidate write
target instead of a single record that has to hold a whole buyer list.

## Considered options

- One `Opportunity` per run, buyers recorded as child `Note`s. Rejected:
  doesn't match real CRM usage, and buries per-buyer fields (`fit_score`,
  `confidence`) as note text instead of structured columns.
