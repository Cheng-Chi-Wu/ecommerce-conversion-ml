import sqlite3

import pandas as pd
import pytest

from src.common import ROOT, DAY


@pytest.fixture
def database():
    with sqlite3.connect(':memory:') as con:
        con.executescript((ROOT/'sql/schema.sql').read_text())
        rows = [
            (0, 1, 'view', 1), (1, 1, 'addtocart', 1), (2, 1, 'transaction', 1),
            (0, 2, 'addtocart', 1), (1, 2, 'view', 1), (2, 2, 'transaction', 1),
            (0, 3, 'view', 1), (1, 3, 'addtocart', 1), (1, 3, 'transaction', 1),
            (0, 4, 'view', 1), (1, 4, 'addtocart', 1), (DAY+2, 4, 'transaction', 1),
            (2*DAY, 5, 'view', 1),
        ]
        con.executemany('INSERT INTO events(timestamp,visitorid,event,itemid) VALUES(?,?,?,?)', rows)
        con.execute('''INSERT INTO event_categories(event_id,timestamp,visitorid,event,itemid)
                       SELECT event_id,timestamp,visitorid,event,itemid FROM events''')
        yield con


def test_funnel_requires_strict_event_order(database):
    result = pd.read_sql_query((ROOT/'sql/01_funnel.sql').read_text(), database)
    assert list(result.visitors) == [5, 3, 2]


def test_abandonment_has_same_item_and_24h_followup(database):
    result = pd.read_sql_query((ROOT/'sql/02_conversion.sql').read_text(), database).set_index('metric')
    assert result.loc['visitor_conversion_rate', 'value'] == .8
    assert result.loc['24h_cart_event_abandonment_rate', 'value'] == .5
    assert result.loc['eligible_cart_events', 'value'] == 4


def test_every_analytics_query_executes_and_trends_reconcile(database):
    for path in sorted((ROOT/'sql').glob('0*.sql')):
        assert not pd.read_sql_query(path.read_text(), database).empty
    trends = pd.read_sql_query((ROOT/'sql/05_trends.sql').read_text(), database)
    assert trends.query("grain == 'daily'").events.sum() == 13
    assert trends.query("grain == 'weekly'").events.sum() == 13
