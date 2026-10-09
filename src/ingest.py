"""Validate source files, stream item properties, and build SQLite from scratch."""
import hashlib
import sqlite3

import numpy as np
import pandas as pd

from .common import DB, RAW, REPORTS, ROOT, save_json, markdown_table

FILES = {
    'events.csv': ['timestamp', 'visitorid', 'event', 'itemid', 'transactionid'],
    'item_properties_part1.csv': ['timestamp', 'itemid', 'property', 'value'],
    'item_properties_part2.csv': ['timestamp', 'itemid', 'property', 'value'],
    'category_tree.csv': ['categoryid', 'parentid'],
}


def source_fingerprint(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return {'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def summarize(frame):
    result = {'rows': len(frame), 'columns': list(frame.columns),
              'dtypes': {k: str(v) for k, v in frame.dtypes.items()},
              'missing': {k: int(v) for k, v in frame.isna().sum().items()},
              'exact_duplicate_rows': int(frame.duplicated().sum())}
    if 'timestamp' in frame:
        result['timestamp_min'] = int(frame.timestamp.min())
        result['timestamp_max'] = int(frame.timestamp.max())
        result['utc_start'] = str(pd.to_datetime(frame.timestamp.min(), unit='ms', utc=True))
        result['utc_end'] = str(pd.to_datetime(frame.timestamp.max(), unit='ms', utc=True))
    return result


def attach_categories(events, history):
    """Use only the latest category change at or before each event timestamp."""
    left = events.sort_values('timestamp', kind='stable')
    right = history.rename(columns={'timestamp': 'category_timestamp'})
    right = right.sort_values('category_timestamp', kind='stable')
    if right.empty:
        result = left.assign(categoryid=np.nan, category_timestamp=np.nan)
    else:
        result = pd.merge_asof(left, right, left_on='timestamp',
                               right_on='category_timestamp', by='itemid',
                               direction='backward', allow_exact_matches=True)
    known = result.category_timestamp.notna()
    assert (result.loc[known, 'category_timestamp'] <= result.loc[known, 'timestamp']).all()
    assert len(result) == len(events)
    return result


def build_database():
    missing = [name for name in FILES if not (RAW / name).exists()]
    if missing:
        raise FileNotFoundError(f'Missing {missing}. Place the original files in {RAW}.')
    for name, expected in FILES.items():
        actual = list(pd.read_csv(RAW / name, nrows=0).columns)
        if actual != expected:
            raise ValueError(f'{name}: expected {expected}; found {actual}')
    profiles = {'files': {}, 'cleaning': {}, 'validation': {}}
    for name in FILES:
        profiles['files'][name] = source_fingerprint(RAW / name)

    events = pd.read_csv(RAW / 'events.csv', dtype={'transactionid': 'Int64'})
    profiles['files']['events.csv'].update(summarize(events))
    profiles['files']['events.csv']['event_distribution'] = {
        k: int(v) for k, v in events.event.value_counts().items()}
    profiles['files']['events.csv']['unique_visitors'] = int(events.visitorid.nunique())
    profiles['files']['events.csv']['unique_items'] = int(events.itemid.nunique())
    profiles['files']['events.csv']['unique_transaction_ids'] = int(events.transactionid.nunique())
    required = ['timestamp', 'visitorid', 'event', 'itemid']
    assert not events[required].isna().any().any(), 'Missing required event field'
    assert set(events.event) <= {'view', 'addtocart', 'transaction'}, 'Unknown event type'
    profiles['validation']['transactions_missing_id'] = int(
        events.loc[events.event.eq('transaction'), 'transactionid'].isna().sum())
    profiles['validation']['nontransactions_with_id'] = int(
        events.loc[events.event.ne('transaction'), 'transactionid'].notna().sum())
    events = events.drop_duplicates().copy()
    events.insert(0, 'event_id', events.index.astype('int64') + 1)
    profiles['cleaning']['events_rows_after_exact_deduplication'] = len(events)

    categories = pd.read_csv(RAW / 'category_tree.csv', dtype={'parentid': 'Int64'})
    profiles['files']['category_tree.csv'].update(summarize(categories))
    assert not categories.categoryid.isna().any()
    assert categories.categoryid.is_unique, 'Conflicting category keys'
    parents = dict(zip(categories.categoryid, categories.parentid))
    for node in parents:
        seen = set()
        current = node
        while current in parents and not pd.isna(parents[current]):
            assert current not in seen, 'Category cycle detected'
            seen.add(current)
            current = parents[current]
    profiles['validation']['category_orphan_parents'] = int(
        (categories.parentid.notna() & ~categories.parentid.isin(categories.categoryid)).sum())

    # Build a sibling temporary database; only replace a previous successful DB
    # once every validation passes. A failed ingestion cannot silently leave a
    # half-loaded database at the public output path.
    pending = DB.with_suffix('.building.sqlite')
    pending.unlink(missing_ok=True)
    with sqlite3.connect(pending) as con:
        con.execute('PRAGMA journal_mode=OFF')
        con.execute('PRAGMA synchronous=OFF')
        con.execute('PRAGMA temp_store=FILE')
        con.executescript((ROOT / 'sql/schema.sql').read_text())
        events.to_sql('events', con, if_exists='append', index=False, chunksize=20_000)
        categories.to_sql('categories', con, if_exists='append', index=False)
        for name in ['item_properties_part1.csv', 'item_properties_part2.csv']:
            print(f'Loading {name} in chunks...', flush=True)
            stats = {'rows': 0, 'columns': FILES[name], 'missing': {c: 0 for c in FILES[name]},
                     'dtypes': {}, 'timestamp_min': None, 'timestamp_max': None}
            for chunk in pd.read_csv(RAW / name, chunksize=250_000,
                                     dtype={'property': str, 'value': str}):
                assert not chunk.isna().any().any(), f'Missing required property field in {name}'
                stats['rows'] += len(chunk)
                stats['dtypes'] = {k: str(v) for k, v in chunk.dtypes.items()}
                for col, n in chunk.isna().sum().items():
                    stats['missing'][col] += int(n)
                low, high = int(chunk.timestamp.min()), int(chunk.timestamp.max())
                stats['timestamp_min'] = min(low, stats['timestamp_min'] or low)
                stats['timestamp_max'] = max(high, stats['timestamp_max'] or high)
                chunk['source_file'] = name
                chunk.to_sql('item_properties', con, if_exists='append', index=False, chunksize=20_000)
            for edge in ['min', 'max']:
                stats[f'utc_{"start" if edge == "min" else "end"}'] = str(
                    pd.to_datetime(stats[f'timestamp_{edge}'], unit='ms', utc=True))
            profiles['files'][name].update(stats)
        print('Auditing global property duplicates and category history...', flush=True)
        duplicates = con.execute('''SELECT COALESCE(SUM(n-1),0) FROM (
            SELECT COUNT(*) AS n FROM item_properties
            GROUP BY timestamp,itemid,property,value HAVING COUNT(*)>1)''').fetchone()[0]
        per_file_duplicates = con.execute('''SELECT source_file,SUM(n-1) FROM (
            SELECT source_file,COUNT(*) AS n FROM item_properties
            GROUP BY source_file,timestamp,itemid,property,value HAVING COUNT(*)>1)
            GROUP BY source_file''').fetchall()
        by_file = dict(per_file_duplicates)
        for name in ['item_properties_part1.csv', 'item_properties_part2.csv']:
            profiles['files'][name]['exact_duplicate_rows'] = int(by_file.get(name, 0))
        profiles['cleaning']['property_global_exact_duplicates_retained'] = int(duplicates)
        invalid_categories = con.execute('''SELECT COUNT(*) FROM item_properties
            WHERE property='categoryid' AND (value='' OR value GLOB '*[^0-9]*')''').fetchone()[0]
        assert invalid_categories == 0, 'Unparseable category identifiers'
        conflicts = con.execute('''SELECT COUNT(*) FROM (
            SELECT itemid,timestamp FROM item_properties WHERE property='categoryid'
            GROUP BY itemid,timestamp HAVING COUNT(DISTINCT value)>1)''').fetchone()[0]
        profiles['validation']['conflicting_category_changes'] = int(conflicts)
        assert conflicts == 0, 'Ambiguous same-time categories; inspect source before continuing'
        con.execute('''INSERT INTO category_history SELECT DISTINCT itemid,timestamp,
            CAST(value AS INTEGER) FROM item_properties WHERE property='categoryid' ''')
        con.execute('CREATE INDEX events_visitor_time ON events(visitorid,timestamp,event,itemid)')
        con.execute('CREATE INDEX events_time ON events(timestamp)')
        con.execute('CREATE INDEX properties_item_time ON item_properties(itemid,timestamp)')
        history = pd.read_sql_query('SELECT * FROM category_history', con)
        enriched = attach_categories(events.drop(columns='transactionid'), history)
        enriched.to_sql('event_categories', con, if_exists='append', index=False, chunksize=20_000)
        con.execute('CREATE INDEX event_categories_time ON event_categories(timestamp)')
        profiles['validation']['event_category_coverage_asof'] = float(enriched.categoryid.notna().mean())
        profiles['validation']['category_history_unknown_tree_ids'] = int(
            (~history.categoryid.isin(categories.categoryid)).sum())
        counts = {table: con.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]
                  for table in ['events', 'item_properties', 'categories', 'category_history', 'event_categories']}
        assert counts['events'] == len(events) == counts['event_categories']
        assert counts['item_properties'] == sum(profiles['files'][n]['rows'] for n in
                                               ['item_properties_part1.csv', 'item_properties_part2.csv'])
        assert con.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
        assert not con.execute('PRAGMA foreign_key_check').fetchall()
        profiles['validation']['sqlite_row_counts'] = counts
        con.execute('ANALYZE')
        con.commit()
    pending.replace(DB)
    save_json(REPORTS / 'data_quality.json', profiles)
    rows = []
    for name, stats in profiles['files'].items():
        rows.append({'file': name, 'rows': stats['rows'], 'exact duplicates': stats['exact_duplicate_rows'],
                     'missing values': sum(stats['missing'].values()),
                     'UTC start': stats.get('utc_start', '—'), 'UTC end': stats.get('utc_end', '—')})
    (REPORTS / 'data_quality.md').write_text(
        '# Data quality audit\n\n' + markdown_table(pd.DataFrame(rows)) +
        '\n\nExact event duplicates are removed from analytics and modeling. All property rows are '
        'retained for provenance; category history uses distinct updates. Missing transaction IDs '
        'on views/carts and missing parent IDs on root categories are structural.\n\n'
        'Full column types, null counts, event distribution, SHA-256 fingerprints, temporal '
        'coverage and integrity checks are in [data_quality.json](data_quality.json).\n')
    print(f'Validated database: {DB}; counts={counts}', flush=True)
    return profiles
