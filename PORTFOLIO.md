# Resume and interview notes

## Recommended positioning

**E-commerce Conversion Prediction & Customer Journey Analytics**

Python · SQL/SQLite · pandas · scikit-learn · pytest · GitHub Actions

This project demonstrates end-to-end data science: defining a product question,
auditing event data, writing SQL, building point-in-time features, evaluating rare
outcomes and proposing an experiment. Lead with scale, methodology and measured
offline performance. All numbers below refer to the saved local run.

## Resume bullets: Data Scientist / Product Data Scientist

- Built a reproducible Python/SQL pipeline analyzing 2.76M e-commerce events and
  generating 1.63M visitor-day snapshots with point-in-time category joins and
  purged chronological train/validation/test splits.
- Compared classification baselines for next-day purchases; achieved 0.239 test
  average precision and 6.66% precision in the top 1% of ranked snapshots,
  representing 48.6× the 0.137% held-out purchase prevalence.
- Quantified 69.87% cart-event abandonment using SQL and proposed a randomized
  re-engagement experiment; implemented 17 local tests covering temporal
  boundaries, SQL definitions, evaluation and scoring.

For a two-bullet resume, use the first two and discuss the experiment in an
interview. Link the project title to the GitHub repository.

## Alternative emphasis: Data Analyst

- Analyzed 2.76M shopping events with Python and SQL to quantify an ordered
  visitor funnel, 69.87% cart-event abandonment and category-level performance,
  with explicit denominators and follow-up eligibility rules.
- Delivered executed notebooks, visualizations and reproducible audit reports;
  translated observational findings into hypotheses for randomized retention
  and re-engagement experiments.

## Alternative emphasis: Machine Learning Engineer

- Implemented modular ingestion, feature generation, training and batch-scoring
  stages for 1.63M visitor-day snapshots, with versioned model metadata, input
  fingerprints and point-in-time feature validation.
- Added 17 local regression tests and a GitHub Actions workflow; validated saved
  model/prediction parity and chronological holdout metrics for rare-event ranking.

The pipeline is an offline prototype. Do not claim deployed infrastructure,
production traffic, real-time inference or operational reliability measurements.

## Interview talking points

1. **Why this target?** Midnight snapshots rank visitors active in the previous
   day for purchases in the next 24 hours. This evaluates overnight
   re-engagement, and misses purchases completed before the cutoff.
2. **How is leakage prevented?** Features end at the observation cutoff,
   category metadata is joined at event time, and a one-day purge separates
   label horizons across chronological partitions.
3. **Why Logistic Regression?** It had the highest validation average precision.
   Random Forest's slightly higher test AP was not used to change the selection.
4. **What does 48.6× mean?** Top-1% test precision divided by overall test
   prevalence. It is ranking enrichment, not a causal treatment effect or an
   increase in revenue. The top cohort still has many non-purchasers.
5. **What would you do next?** Evaluate fresher observation cutoffs on a new
   forward holdout, then test incremental purchases with randomized contact
   and no-contact groups. Calibrate probabilities using validation data.

## Evidence

- [Model metrics and parameters](reports/model_results.json)
- [Feature and split audit](reports/feature_audit.json)
- [SQL validation](reports/sql_validation.json)
- [Artifact verification from the original run](reports/verification.json)
- [Tests](tests/test_features.py)

Keep the resume consistent with these artifacts. No measured revenue lift,
causal retention effect or deployed experiment is available in this dataset.
