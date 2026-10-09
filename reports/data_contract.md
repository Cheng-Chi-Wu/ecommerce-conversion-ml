# Data and prediction contract

- **Decision:** an overnight re-engagement ranking for an experimental contact
  budget; the product/growth team would own a future intervention.
- **Entity/key:** `(visitorid, observation_time)`; cutoff is a fixed UTC midnight.
- **Eligibility:** at least one deduplicated event in `(t-24h,t]`; eligibility
  cannot depend on returning or purchasing later.
- **Label:** 1 if any transaction item event is in `(t,t+24h]`, otherwise 0.
  The label becomes available after its 24h horizon, not at prediction time.
- **Capture:** require `t-24h >= global_min_event_time` and
  `t+24h <= global_max_event_time`. This avoids boundary censoring but assumes
  observed capture is meaningful between the dataset endpoints.
- **Features:** the ten named numeric features in `src/features.py`; all must be
  finite and nonnegative. IDs and diagnostic time fields are excluded.
- **History:** transaction counts include all events observed through `t`;
  session duration uses only observed events in the latest 30-minute-gap session.
- **Metadata:** latest category update at or before the event. No future fill.
  Unknowns contribute to a missingness count, not a known category count.
- **Input schema:** exact CSV headers are checked. Required event/property fields
  cannot be null. Transaction IDs may be null for nontransactions, root parents
  may be null. Event types are view/addtocart/transaction. Timestamps are UTC ms.
- **Deduplication:** remove exact duplicate event rows across all five source
  fields; retain raw property rows and deduplicate derived category changes.
  Conflicting category values for one item/time fail closed.
- **Partitioning:** chronological 60/20/20 observation dates, with earlier label
  horizon strictly before later observation start. Boundary rows are purged.
- **Visitor overlap:** allowed because the goal is future-traffic evaluation;
  IDs are excluded, and known/new visitor slices are reported separately.
- **Selection:** best validation AP; validation-F1 threshold; test is reporting
  only. Fixed hyperparameters and seeds. No random split or test tuning.
- **Output:** score, thresholded prediction, optional visitor/time keys and
  model version. Scores from class-weighted training are uncalibrated.
- **Offline checks:** AP and top-1% precision must exceed the naive reference to
  claim useful offline ranking. Passing these checks does not authorize deployment.
- **Provenance:** raw file SHA-256 values in `data_quality.json`, code/config
  hashes and dependency versions in `run_manifest.json`.
