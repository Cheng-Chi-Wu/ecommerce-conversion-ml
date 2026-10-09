# How the analysis is set up

These are the main choices behind the current code and results, along with
what they leave unanswered.

## What the model predicts

Each row is a visitor active in the 24 hours before a fixed UTC midnight. The
label records any transaction in the following 24 hours. A visitor can appear
on more than one date. The model uses ten behavior features, not visitor or
item IDs.

This target is useful for thinking about reaching visitors the next day. It
also leaves something out: purchases made before midnight are already history.
The model does not cover those same-visit purchases. Its positive rate and the
full-period visitor conversion rate count different things, so they should not
be compared as though they were the same metric.

See [feature construction](../src/features.py) and the
[observation/split audit](feature_audit.json).

## Category data is available at different times

Item properties arrive as timestamped updates. The ingestion code assigns the
latest category available at the event timestamp. Events before an item's first
known category remain unknown. This yields 76.16% event-time category coverage;
filling the rest from later records would introduce future information.

Exact duplicate events are removed. Property history is retained, and
conflicting category values at the same item/time stop ingestion.

See [ingestion](../src/ingest.py), [data audit](data_quality.json) and the
[future-metadata test](../tests/test_features.py).

## Why Logistic Regression is the selected model

| Model | Validation AP | Test AP |
| --- | --- | --- |
| Logistic Regression | 0.2492 | 0.2392 |
| Random Forest | 0.2349 | 0.2405 |

The implementation selects the highest validation average precision before
computing test metrics. Random Forest's slightly higher test AP does not change
that selection. Neither model has undergone an extensive hyperparameter search.

Dates are split into train, validation and test, with a one-day gap at each
boundary. The same visitor can appear in more than one period. That matches the
question of predicting future activity, including return visits. It does not
answer how well the model works only on completely new visitors. The error
tables report visitors seen and not seen in training separately.

The yes/no prediction threshold uses the best validation F1. Without contact
costs, there is no basis for calling that the best business threshold. Taking
the top 1% is a separate way to use the scores; the two precision values measure
different selections.

See [model selection](../src/modeling.py), [parameters and metrics](model_results.json)
and [error slices](error_slices.csv).

## What limits the results

The test set contains 430 positive snapshots. A 100-replicate day-block bootstrap
gives an approximate AP interval of [0.1996, 0.2890], but repeated visitors across
days remain dependent. Class weighting also makes raw model scores unsuitable
as calibrated purchase probabilities; [calibration bins](calibration_bins.csv)
show the discrepancy.

XGBoost was installed locally but could not load its OpenMP runtime (`libomp`).
No boosting model was evaluated in the recorded run. The import error is kept
in [model_results.json](model_results.json).

## Next comparison

Compare daily and fresher observation cutoffs on validation data at the same
contact capacity. Hourly labels overlap, so their full 24-hour horizons must
remain separated across partitions. Evaluate the chosen approach on a new
forward holdout rather than reusing the existing test set for tuning.

If the ranking is useful, a later contact/no-contact experiment could measure
incremental purchases. Neither that intervention nor the cutoff comparison has
been run here.
