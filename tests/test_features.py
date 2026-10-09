import numpy as np
import pandas as pd

from src.common import DAY, HOUR
from src.features import FEATURES, make_snapshots, assign_time_splits
from src.ingest import attach_categories


def events_frame(records):
    frame = pd.DataFrame(records, columns=['timestamp', 'visitorid', 'event', 'itemid'])
    frame['event_id'] = np.arange(len(frame)) + 1
    frame['categoryid'] = 7
    return frame


def test_exact_window_boundaries_and_historical_transactions():
    t = 10 * DAY
    events = events_frame([
        (t-DAY, 1, 'view', 100),       # excluded at lower lookback boundary
        (t-HOUR, 1, 'view', 1),        # excluded from 1h, included in 24h
        (t-HOUR+1, 1, 'view', 1),
        (t, 1, 'view', 1),
        (t, 1, 'transaction', 1),      # features/history, not label at t
        (t+1, 1, 'transaction', 1),    # label only
        (t+DAY, 1, 'transaction', 1),  # included at label upper boundary
        (t+DAY+1, 1, 'transaction', 1),
    ])
    row = make_snapshots(events, 0, 30*DAY).query('observation_time == @t').iloc[0]
    assert row.views_24h == 3
    assert row.views_1h == 2
    assert row.unique_items_viewed_24h == 1
    assert row.repeat_views_24h == 2
    assert row.previous_transactions == 1
    assert row.label_first_time == t+1
    assert row.label_last_time == t+DAY


def test_event_at_cutoff_is_not_a_future_label():
    t = 10*DAY
    frame = events_frame([(t-1, 1, 'view', 1), (t, 1, 'transaction', 1)])
    row = make_snapshots(frame, 0, 30*DAY).query('observation_time == @t').iloc[0]
    assert row.label == 0
    assert row.previous_transactions == 1


def test_future_behavior_cannot_change_features_or_eligibility():
    t = 10*DAY
    past = [(t-1000, 1, 'view', 1), (t-500, 2, 'view', 2)]
    before = make_snapshots(events_frame(past), 0, 30*DAY).query('observation_time == @t')
    after = make_snapshots(events_frame(past + [(t+1, 1, 'transaction', 1),
                                              (t+2, 1, 'addtocart', 3)]), 0, 30*DAY)
    after = after.query('observation_time == @t')
    pd.testing.assert_frame_equal(before[['visitorid']+FEATURES].reset_index(drop=True),
                                  after[['visitorid']+FEATURES].reset_index(drop=True))
    assert list(before.label) == [0, 0]  # No future events is a legitimate negative.
    assert list(after.label) == [1, 0]


def test_future_metadata_never_backfills_a_category():
    frame = events_frame([(10, 1, 'view', 1), (20, 1, 'view', 2), (30, 1, 'view', 1)])
    frame = frame.drop(columns='categoryid')
    history = pd.DataFrame([(1, 5, 7), (1, 25, 9), (2, 21, 8)],
                           columns=['itemid', 'timestamp', 'categoryid'])
    result = attach_categories(frame, history).set_index('timestamp')
    assert result.loc[10, 'categoryid'] == 7
    assert pd.isna(result.loc[20, 'categoryid'])
    assert result.loc[30, 'categoryid'] == 9


def test_capture_edges_and_temporal_purge():
    t = 10*DAY
    frame = events_frame([(t-1, 1, 'view', 1), (t+DAY, 2, 'view', 2), (t+DAY+1, 2, 'view', 2)])
    snapshots = make_snapshots(frame, t-DAY, t+2*DAY)
    assert snapshots.observation_time.min()-DAY >= t-DAY
    assert snapshots.label_end.max() <= t+2*DAY
    artificial = pd.DataFrame({'observation_time': np.arange(2, 42)*DAY})
    artificial['label_end'] = artificial.observation_time+DAY
    split = assign_time_splits(artificial)
    assert split.split.eq('purged').sum() == 2
    for first, second in [('train', 'validation'), ('validation', 'test')]:
        assert split.loc[split.split.eq(first), 'label_end'].max() < split.loc[
            split.split.eq(second), 'observation_time'].min()
