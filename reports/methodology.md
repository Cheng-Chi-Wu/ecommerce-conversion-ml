# ECC MLE iteration compact and observation ledger

Workflow applied: `ecc:mle-workflow` from the installed Everything Claude Code
plugin. The local portfolio uses data contracts, reusable transformations,
reproducible training, artifact provenance, evaluation slices and explicit
product limitations. No production infrastructure or global hooks were installed.

## Iteration compact

**Goal:** rank recently active visitors for next-day purchase propensity and
explain observed journey drop-off. **Who cares:** a product/growth analyst with
a limited overnight contact budget. **Success:** validation AP above a constant
prevalence baseline, supported by held-out AP and Precision@1%. **Guardrails:**
precision, recall, false contacts, known/new visitor errors and metadata coverage.
**Mistake budget:** contact cost and harm are unavailable, so F1 is an explicit
offline default, not an economically optimal threshold. **Unacceptable mistakes:**
future-filled features, random splitting, test-tuned thresholds or invented results.
**Assumptions:** UTC calendar batch policy, 30-minute session gap, complete
capture between observed endpoints. **Baseline:** training prevalence, with a
fixed 0.5 threshold. **Signals:** engagement, intent, recency, breadth and purchase
history. **Fallback:** retain descriptive SQL/baseline if learned ranking fails
offline checks. **Status:** portfolio prototype, no deployed intervention.

## Observation ledger

1. **Audit:** exact duplicates and structural nulls must be distinguished.
   **Decision:** remove only exact event duplicates; preserve source fingerprints
   and original property rows. **Evidence:** `data_quality.json`.
2. **Temporal metadata:** category changes begin after the event capture starts.
   **Decision:** preserve missing historical categories rather than joining later
   snapshots backward. **Evidence:** event-time coverage and metadata boundary test.
3. **Rare target:** many purchases occur before the next midnight. **Decision:**
   explicitly frame this as next-day re-engagement; avoid comparing its prevalence
   to full-period conversion as though denominators were equal.
4. **Native dependency:** the local XGBoost wheel lacks a working OpenMP runtime.
   **Decision:** capture optional import errors and retain the executable baseline
   comparison. Optional fit failures are not swallowed.
5. **Model choice:** use validation AP and validation-F1 thresholds, then freeze
   the decision for test reporting. **Evidence:** `model_results.json` and
   `run_manifest.json`. **Tradeoff:** raw class-weighted scores are uncalibrated.
6. **Error analysis:** compare historical buyers, cart activity, unknown metadata
   and visitors seen/not seen in train. **Evidence:** `error_slices.csv`, calibration
   bins and approximate day-block AP uncertainty. Slices overlap.

## Next falsifiable experiment

Build an hourly or event-triggered snapshot policy. Hypothesis: fresher behavior
captures more impending purchases than midnight-only observations at the same
contact capacity. Measure validation AP and precision at daily capacity, then
evaluate on a **new forward holdout**. Purge the full 24h label horizon and handle
overlapping hourly labels explicitly. If ranking improves, test incremental
purchase outcomes with a randomized contact/no-contact experiment. Do not infer
uplift from offline propensity or assume readable revenue fields exist.
