"""Point-in-time visitor snapshots at fixed UTC midnights, using vectorized groups."""
import sqlite3

import numpy as np
import pandas as pd

from .common import DB, DAY, HOUR, PROCESSED, REPORTS, save_json

FEATURES = [
    'views_1h', 'views_24h', 'unique_items_viewed_24h', 'repeat_views_24h',
    'cart_additions_24h', 'last_session_duration_minutes', 'category_diversity_24h',
    'unknown_category_views_24h', 'previous_transactions', 'hours_since_last_event',
]


def make_snapshots(events, capture_start=None, capture_end=None):
    """Features use (t-24h,t]; labels use (t,t+24h]. No event is in both.

    One row per visitor active in the preceding 24h, at fixed daily UTC cutoffs.
    A row's existence never depends on that visitor returning in the label window.
    Global capture edges ensure complete lookback and label follow-up.
    """
    start = int(events.timestamp.min()) if capture_start is None else capture_start
    end = int(events.timestamp.max()) if capture_end is None else capture_end
    frame = events.sort_values(['visitorid', 'timestamp', 'event_id'], kind='stable').copy()
    assert frame.event_id.is_unique
    # Integer ceiling places events exactly at t into features at t, never at t+24h.
    frame['observation_time'] = ((frame.timestamp - 1) // DAY + 1) * DAY
    frame['view'] = frame.event.eq('view').astype('int64')
    frame['cart'] = frame.event.eq('addtocart').astype('int64')
    frame['transaction'] = frame.event.eq('transaction').astype('int64')
    frame['view_1h'] = (frame.view.eq(1) &
                        frame.timestamp.gt(frame.observation_time - HOUR)).astype('int64')
    frame['unknown_category_view'] = (frame.view.eq(1) & frame.categoryid.isna()).astype('int64')
    frame['previous_transactions'] = frame.groupby('visitorid', sort=False).transaction.cumsum()
    gap = frame.groupby('visitorid', sort=False).timestamp.diff()
    frame['new_session'] = (gap.isna() | gap.gt(HOUR / 2)).astype('int64')
    frame['session_id'] = frame.groupby('visitorid', sort=False).new_session.cumsum()
    session_start = frame.groupby(['visitorid', 'session_id'], sort=False).timestamp.transform('min')
    frame['last_session_duration_minutes'] = (frame.timestamp - session_start) / 60_000
    keys = ['visitorid', 'observation_time']
    grouped = frame.groupby(keys, sort=False)
    snapshots = grouped.agg(
        views_1h=('view_1h', 'sum'), views_24h=('view', 'sum'),
        cart_additions_24h=('cart', 'sum'),
        unknown_category_views_24h=('unknown_category_view', 'sum'),
        feature_min_time=('timestamp', 'min'), feature_max_time=('timestamp', 'max'))
    views = frame.loc[frame.view.eq(1)]
    snapshots['unique_items_viewed_24h'] = views.groupby(keys, sort=False).itemid.nunique()
    snapshots['category_diversity_24h'] = views.groupby(keys, sort=False).categoryid.nunique()
    snapshots[['unique_items_viewed_24h', 'category_diversity_24h']] = snapshots[
        ['unique_items_viewed_24h', 'category_diversity_24h']].fillna(0).astype('int64')
    snapshots['repeat_views_24h'] = snapshots.views_24h - snapshots.unique_items_viewed_24h
    latest = grouped.tail(1).set_index(keys)
    snapshots['previous_transactions'] = latest.previous_transactions
    snapshots['last_session_duration_minutes'] = latest.last_session_duration_minutes
    snapshots = snapshots.reset_index()
    snapshots['hours_since_last_event'] = (snapshots.observation_time - snapshots.feature_max_time) / HOUR
    snapshots = snapshots.loc[(snapshots.observation_time - DAY >= start) &
                               (snapshots.observation_time + DAY <= end)].copy()

    transactions = frame.loc[frame.transaction.eq(1), ['visitorid', 'observation_time', 'timestamp']].copy()
    transactions['observation_time'] -= DAY
    outcomes = transactions.groupby(keys, sort=False).agg(
        label_first_time=('timestamp', 'min'), label_last_time=('timestamp', 'max'))
    snapshots = snapshots.merge(outcomes, on=keys, how='left', validate='one_to_one')
    snapshots['label'] = snapshots.label_first_time.notna().astype('int64')
    snapshots['label_end'] = snapshots.observation_time + DAY
    validate_snapshots(snapshots)
    return snapshots.sort_values(['observation_time', 'visitorid']).reset_index(drop=True)


def validate_snapshots(frame):
    assert len(frame) > 0, 'No complete observation windows'
    assert not frame.duplicated(['visitorid', 'observation_time']).any()
    assert (frame.feature_min_time > frame.observation_time - DAY).all()
    assert (frame.feature_max_time <= frame.observation_time).all()
    positive = frame.label.eq(1)
    assert (frame.loc[positive, 'label_first_time'] > frame.loc[positive, 'observation_time']).all()
    assert (frame.loc[positive, 'label_last_time'] <= frame.loc[positive, 'label_end']).all()
    assert np.isfinite(frame[FEATURES].to_numpy()).all()
    assert (frame[FEATURES] >= 0).all().all()


def assign_time_splits(frame):
    """60/20/20 of distinct observation dates; purge overlapping target horizons."""
    times = np.sort(frame.observation_time.unique())
    if len(times) < 10:
        raise ValueError('Need at least 10 observation days for chronological evaluation')
    validation_start = int(times[int(len(times) * .6)])
    test_start = int(times[int(len(times) * .8)])
    result = frame.copy()
    result['split'] = 'purged'
    result.loc[result.label_end.lt(validation_start), 'split'] = 'train'
    result.loc[result.observation_time.ge(validation_start) &
               result.label_end.lt(test_start), 'split'] = 'validation'
    result.loc[result.observation_time.ge(test_start), 'split'] = 'test'
    for earlier, later in [('train', 'validation'), ('validation', 'test')]:
        a, b = result.loc[result.split.eq(earlier)], result.loc[result.split.eq(later)]
        assert len(a) and len(b)
        assert a.label_end.max() < b.observation_time.min()
    return result


def build_features():
    with sqlite3.connect(DB) as con:
        events = pd.read_sql_query('SELECT * FROM event_categories', con)
    snapshots = assign_time_splits(make_snapshots(events))
    snapshots.to_pickle(PROCESSED / 'visitor_snapshots.pkl')
    summary = {'features': FEATURES, 'rows': len(snapshots),
               'positive_rate': float(snapshots.label.mean()),
               'observation_policy': 'Fixed UTC midnight; active in preceding 24h',
               'feature_window': '(t-24h,t]; historical transaction counts and last-session history through t',
               'label_window': '(t,t+24h]', 'splits': {}}
    for name, subset in snapshots.groupby('split'):
        summary['splits'][name] = {
            'rows': len(subset), 'positives': int(subset.label.sum()),
            'positive_rate': float(subset.label.mean()),
            'observation_start': str(pd.to_datetime(subset.observation_time.min(), unit='ms', utc=True)),
            'observation_end': str(pd.to_datetime(subset.observation_time.max(), unit='ms', utc=True)),
            'label_end_max': str(pd.to_datetime(subset.label_end.max(), unit='ms', utc=True)),
        }
    summary['validation'] = {'point_in_time_checks_passed': True,
                             'train_validation_visitor_overlap': int(len(set(snapshots.loc[
                                 snapshots.split.eq('train'), 'visitorid']) & set(snapshots.loc[
                                 snapshots.split.eq('validation'), 'visitorid']))),
                             'train_test_visitor_overlap': int(len(set(snapshots.loc[
                                 snapshots.split.eq('train'), 'visitorid']) & set(snapshots.loc[
                                 snapshots.split.eq('test'), 'visitorid'])))}
    save_json(REPORTS / 'feature_audit.json', summary)
    print(f'Validated snapshots: {len(snapshots):,}; positive rate={snapshots.label.mean():.4%}', flush=True)
    return snapshots
