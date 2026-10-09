# E-commerce Conversion Prediction and Customer Journey Analytics

An end-to-end analysis of **2.76 million shopping events**, combining SQL customer
journey analytics with leakage-aware prediction of next-day purchases.
Built by a UCLA MEng Data Science student using **Python, SQL/SQLite, pandas,
scikit-learn, matplotlib, pytest and GitHub Actions**.

[![Tests](https://github.com/Cheng-Chi-Wu/ecommerce-conversion-ml/actions/workflows/tests.yml/badge.svg)](https://github.com/Cheng-Chi-Wu/ecommerce-conversion-ml/actions/workflows/tests.yml)

## Project at a glance

- **Data engineering:** audited 2,756,101 raw events and built 1,634,383 visitor-day
  snapshots with historical category joins and explicit observation/label windows.
- **Machine learning:** compared prevalence, Logistic Regression and Random Forest
  baselines on chronological splits with a one-day purge. The validation-selected
  Logistic Regression achieved **0.2392 test average precision** and **6.66%
  Precision@1%**, versus **0.1371%** test prevalence (**48.6× ranking lift**).
- **Product analytics:** quantified an ordered shopping funnel and **69.87%**
  cart-event abandonment; proposed a randomized re-engagement experiment to
  measure incremental purchases.
- **Reproducibility:** modular pipeline, executed notebooks, input fingerprints,
  persisted evaluation artifacts and **17 passing local tests**.

The lift measures offline concentration of future purchases in the ranked cohort;
it does not measure conversion improvement from contacting customers.

### Explore the evidence

| Interest | Start here |
| --- | --- |
| Business findings and proposed experiments | [Results](reports/results.md) |
| SQL and customer journey analysis | [Executed analytics notebook](notebooks/01_customer_journey.ipynb) · [Funnel SQL](sql/01_funnel.sql) |
| Modeling, validation and error analysis | [Executed modeling notebook](notebooks/02_conversion_model.ipynb) · [Recorded model results](reports/model_results.json) |
| Leakage prevention and engineering | [Feature construction](src/features.py) · [Boundary tests](tests/test_features.py) · [Data contract](reports/data_contract.md) |
| Run the project | [Reproduction](#reproduction) |
| Resume wording and interview preparation | [Portfolio notes](PORTFOLIO.md) |

## Business problem

Where do shoppers drop out of the observed journey, and which recently active
visitors are most likely to purchase in the next 24 hours? The intended model
decision is an **overnight batch ranking for a re-engagement experiment**, with a
limited contact budget. SQL describes the customer journey; ML prioritizes future
outcomes. Propensity is not incremental uplift, so business impact needs a
randomized experiment.

## Dataset and attribution

The [Retailrocket dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)
contains timestamped shopping interactions, item-property changes and a category
tree. It was supplied by Retailrocket, not collected by this project's author.
The source listing displays [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
See [DATA_SOURCE.md](DATA_SOURCE.md) for attribution, licensing scope and local
input fingerprints. No Kaggle notebook was copied. Raw data is not committed.

## Results

<!-- RESULTS:START -->

The audited source contains **2,756,101 events** from
**2015-05-03 to 2015-09-18 UTC**, across **1,407,580 visitors**.
Removing 460 exact duplicate rows leaves
**2,755,641 events** for analysis.

- Full-period visitor conversion: **0.83%** (any transaction / any visitor).
- Ordered view → cart → transaction visitors: **1,404,179 → 32,934 → 9,952**.
- 24h cart-event abandonment: **69.87%** across 68,710 eligible cart events.
- Retrospective returning-visitor conversion: **3.50%**, versus **0.53%** for single-day visitors.
- Event-time category coverage: **76.16%**; future metadata is never backfilled.

**1,634,383 visitor-day snapshots** were generated. The held-out test has
**313,534 snapshots**, **430 purchases** and **0.1371% prevalence**.
These snapshot metrics answer a different question from full-period visitor conversion.

| Model | PR-AUC (AP) | ROC-AUC | Precision | Recall | F1 | Precision@1% |
| --- | --- | --- | --- | --- | --- | --- |
| Naive prevalence baseline | 0.0014 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0006 |
| Logistic Regression | 0.2392 | 0.8858 | 0.6357 | 0.1907 | 0.2934 | 0.0666 |
| Random Forest | 0.2405 | 0.8594 | 0.6406 | 0.1907 | 0.2939 | 0.0625 |

**Logistic Regression** was chosen using validation AP, with its threshold
selected on validation F1. Test Precision@1% is **6.66%**
at **K=3,136**, or **48.6×** test prevalence.
Selecting the top 1% separately each day yields **6.70%** pooled precision.
The threshold produces **82 true positives**, **47 false positives**,
and **348 false negatives**. The approximate 95% day-block bootstrap
interval for selected-model AP is **[0.1996, 0.2890]** (100 replicates).

Boosting status: **Skipped: no usable XGBoost/LightGBM runtime**.
The installed XGBoost wheel could not load its missing macOS OpenMP runtime (`libomp`). No boosting metrics are invented. Complete parameters, import status and fit times are in
[model_results.json](reports/model_results.json); cohort errors are in
[error_slices.csv](reports/error_slices.csv).

<!-- RESULTS:END -->

![Ordered visitor funnel](reports/figures/funnel.png)
![Daily event activity](reports/figures/event_trend.png)
![Held-out precision-recall curves](reports/figures/precision_recall_curve.png)
![Confusion matrix](reports/figures/confusion_matrix.png)
![Predictive feature importance](reports/figures/feature_importance.png)

## Repository structure

```text
data/raw/                 Original CSVs; ignored by Git
data/processed/           SQLite, snapshots, model, predictions; ignored by Git
notebooks/                Executed walkthrough and reproducible exploration
sql/                      Schema and five standalone analysis queries
src/                      Ingest, SQL, features, modeling, scoring and reporting
reports/figures/           README figures
reports/                  Audits, aggregate query outputs and evaluation evidence
tests/                    Window boundaries, temporal joins, SQL and metrics
requirements*.txt         Dependencies and tested environment lock
```

## Database schema

All timestamps are integer Unix **milliseconds in UTC**. Nullable transaction
IDs on views/carts and parent IDs on root categories are structural, not
imputation targets. An event transaction row represents an item interaction;
several rows can belong to one transaction ID. Event counts are not order counts.

| Table | Grain / key | Main fields |
| --- | --- | --- |
| `events` | Deduplicated event; `event_id` | timestamp, visitorid, event, itemid, nullable transactionid |
| `item_properties` | Original property-change row | timestamp, itemid, property, value, source_file |
| `categories` | Category; `categoryid` | nullable parentid |
| `category_history` | Item + effective timestamp | categoryid |
| `event_categories` | Event; `event_id` | event fields, nullable categoryid and category_timestamp |

See [schema.sql](sql/schema.sql). Catalog relationships are logical rather than
enforced foreign keys because coverage is incomplete. Exact event duplicates
are removed; raw property rows are retained. Category changes are deduplicated,
and conflicting category values at the same item/time stop ingestion. The
database is replaced only after row-count and SQLite integrity checks pass.

## SQL analysis

| Query | Definition |
| --- | --- |
| [Funnel](sql/01_funnel.sql) | Unique visitors with a view, then a strictly later cart, then a strictly later transaction in the capture period; can span sessions and items |
| [Conversion and abandonment](sql/02_conversion.sql) | Any purchasing visitor / any visitor; eligible cart events without a strictly later same-item purchase within 24h / eligible cart events |
| [Returning visitors](sql/03_returning_visitors.sql) | Active on at least two observed UTC dates; retrospective segment, not an ML feature |
| [Product/category performance](sql/04_product_category.sql) | Views, carts, transaction item events and purchasing visitor rates; categories joined at event time |
| [Daily/weekly trends](sql/05_trends.sql) | UTC dates and Monday-start weeks, event counts and within-period visitor conversion |

Abandonment excludes carts without a full 24h follow-up at the capture boundary.
There is no cart ID; repeated cart additions are separate events and can share
one future purchase. Unknown categories remain explicit. Visitor denominators
overlap across products/categories and should not be summed. First/last daily
and weekly buckets can be partial; interpret boundary trends accordingly.

## ML methodology

**Unit:** one visitor at a fixed UTC midnight `t`, eligible only if active in the
prior 24h. The prediction is whether any transaction occurs in `(t, t+24h]`.
No future activity is required for eligibility. This policy primarily captures
next-day returning purchases; it excludes first-visit purchases completed before
the next cutoff. Its positive rate is therefore lower than full-period conversion.

| Feature | Definition and business hypothesis |
| --- | --- |
| `views_1h`, `views_24h` | Views in `(t-1h,t]` and `(t-24h,t]`; recent engagement |
| `unique_items_viewed_24h` | Distinct viewed items in prior 24h; breadth of exploration |
| `repeat_views_24h` | Views minus unique viewed items; repeated consideration |
| `cart_additions_24h` | Cart event count in prior 24h; purchase intent |
| `last_session_duration_minutes` | First-to-last observed event in latest session; engagement depth |
| `category_diversity_24h` | Distinct known event-time categories viewed; breadth of interests |
| `unknown_category_views_24h` | Views with unavailable event-time metadata; missingness signal |
| `previous_transactions` | All transaction item events observed through `t`; historical buying |
| `hours_since_last_event` | Time from most recent observed event to `t`; recency |

Sessions use a gap **greater than 30 minutes** to start a new session. The last
session can start before the 24h lookback, but only events through `t` contribute
to its duration. Unknown categories do not increase diversity. Zero is a real
absence of activity, not a missing numeric feature. Outliers are retained; the
logistic pipeline applies `log1p` then train-fitted standardization. Correlated
counts and unfiltered abnormal traffic are limitations, not causal signals.

Distinct observation dates are divided chronologically into roughly **60% train,
20% validation, 20% test**. A one-day boundary purge requires every earlier
partition's label horizon to end before the next partition's first observation.
Exact dates and sizes are in [feature_audit.json](reports/feature_audit.json).
The same visitor may legitimately appear later; IDs are not model features.
This evaluates future traffic, rather than a visitor-disjoint cold-start task.

Models are trained in order: a constant training-prevalence baseline (threshold
0.5), weighted Logistic Regression, weighted Random Forest, then XGBoost or
LightGBM when its runtime loads. Hyperparameters are fixed and recorded, with
seed 42. There is no SMOTE or random split. The model with highest **validation
average precision** is selected before inspecting test scores. Candidate
thresholds maximize validation F1; no model is refit on test data.

**Primary metric:** PR-AUC, reported as non-interpolated **average precision
(AP)**. Also report ROC-AUC, precision, recall, F1 and Precision@K. K is the
ceiling of 1% of held-out snapshots, with seeded, label-independent tie breaking.
A separate daily top-1% analysis reflects batch capacity. F1 is an offline
default; operational thresholds require real contact costs. Class weighting
does not produce calibrated purchase probabilities. Cohort errors, calibration
bins and an approximate day-block bootstrap AP interval accompany the results.

## Leakage prevention

- Recent-feature windows are `(t-window,t]`; labels are `(t,t+24h]`. A transaction
  exactly at `t` is history, never a future label.
- Candidate snapshots are fixed calendar cutoffs selected from past activity,
  not last events chosen using a visitor's future session completion.
- Item categories use backward as-of joins, with metadata timestamp ≤ event
  timestamp ≤ observation time. Missing past metadata is never future-filled.
- Observation windows must have complete global lookback and follow-up.
- Label horizons are purged across chronological partition boundaries.
- Preprocessing fits on train only; thresholds and model selection use validation.
- Only the ten documented features enter models. Visitor/item IDs, target window
  fields and retrospective returning-visitor flags are excluded.

Assertions and tests check these rules, including exact millisecond boundaries
and mutation of future events. See [data_contract.md](reports/data_contract.md).

## Business recommendations

1. Audit cart abandonment and test a consent-based reminder against a randomized
   holdout. The observed rate suggests a question to investigate, not a proven
   checkout failure.
2. Test model-ranked overnight re-engagement at a limited capacity. Measure
   incremental transactions per eligible visitor and per contact; guard against
   fatigue. Avoid automatically discounting visitors who would buy anyway.
3. Inspect high-volume category IDs alongside their metadata coverage before
   proposing navigation changes. Use descriptive rates to form hypotheses.
4. Validate recurring-customer strategies experimentally. The retrospective
   returning-visitor association does not establish causal retention effects.

Further interpretation and category tables: [results.md](reports/results.md).

## Limitations

This historical, anonymized dataset lacks readable prices, revenue/margin,
demographics, marketing exposures, consent status and causal treatment labels.
Visitor IDs need not correspond to people across devices. Partial instrumentation,
weekly metadata snapshots, unknown categories, traffic anomalies and right
censoring can affect findings. A global last timestamp is a capture-boundary
proxy, not proof of perfect tracking for every visitor. Returning status is
descriptive over the full period. Impurity importance or coefficients can be
unstable with correlated features and do not prove causality. Rare test positives
limit precision; day-block bootstrapping leaves cross-day visitor dependence.
The model is an offline portfolio prototype, not a production deployment or
evidence of incremental revenue. A new forward holdout is needed for further tuning.

## Reproduction

Run from the repository root with **Python 3.9+**. Download the four original CSVs
from Kaggle and put them in `data/raw/` with their original filenames. Missing
files or unexpected headers stop the pipeline; no substitute data is generated.
Allow several GB of disk space (the database alone is about 2.1 GB), additional
space for an atomic rebuild, and several GB of RAM. Properties are streamed;
events/snapshots are processed in memory. CPU runtime varies by hardware.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python -m src.pipeline all
```

Optional boosting: `python -m pip install xgboost` or `lightgbm`. The installed
package must also load its native runtime; import errors are recorded without
blocking the baseline models. See the [official XGBoost installation guide](https://xgboost.readthedocs.io/en/stable/install.html).
The captured environment is [requirements-lock.txt](requirements-lock.txt),
recorded on macOS ARM64/Python 3.9; use `requirements.txt` on other platforms.

Stages can be rerun independently:

```bash
python -m src.pipeline ingest
python -m src.pipeline sql
python -m src.pipeline features
python -m src.pipeline train
python -m src.pipeline report
sqlite3 data/processed/retailrocket.sqlite < sql/02_conversion.sql
```

To explore the executed notebook:

```bash
python -m pip install -r requirements-notebooks.txt
python -m ipykernel install --prefix .venv --name retailrocket --display-name 'Retailrocket (.venv)'
python -m src.notebooks build
python -m src.notebooks execute
python -m src.verify
# Optional interactive UI: python -m pip install jupyterlab; jupyter lab
```

The notebook reads audited outputs and executes SQLite queries; model training
lives in `src/`, with no hidden notebook state. Its committed outputs contain
aggregate summaries only. To score your own locally computed feature CSV:

```bash
python -m src.predict data/processed/features_to_score.csv --output data/processed/scores.csv
```

Compute input features with the same documented point-in-time policy. The batch
scorer checks required columns, numeric types, nonnegative finite values, and
uses preprocessing bundled with the locally generated model. Only load your
own trusted model bundle. [run_manifest.json](reports/run_manifest.json) records
input hashes, source-code hash, model/config version, packages and offline
evidence checks. [verification.json](reports/verification.json) records final
checks. GitHub CI runs data-free boundary/SQL/metric tests; full-data runs are
local because raw data is excluded.

## Learning and next iteration

The most useful lessons are explicit SQL denominators, temporal metadata joins,
rare-event ranking and separating prediction from causal product impact.
[methodology.md](reports/methodology.md) records the ECC MLE design choices and
the next falsifiable experiment. Extend this project with hourly observations,
rolling forward backtests and validation-only probability calibration, evaluated
on a fresh future holdout.
