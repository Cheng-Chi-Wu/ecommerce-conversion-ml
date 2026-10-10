import pandas as pd

from src.analytics import category_summary


def test_top_twenty_is_fixed_with_numeric_ties_and_unknowns():
    records = []
    for grain in ['category', 'product']:
        for entity in range(1, 23):
            records.append([grain, str(entity), 100, 10, 7,
                            999 if entity == 1 else 2000, .01])
    records.append(['category', 'unknown', 300, 20, 99, 100, .04])
    full = pd.DataFrame(records, columns=['grain', 'entity', 'views', 'carts',
                                         'transaction_events', 'visitors', 'visitor_conversion_rate'])
    summary = category_summary(full)
    shuffled = category_summary(full.sample(frac=1, random_state=7))
    pd.testing.assert_frame_equal(summary, shuffled)
    for group in ['top_categories', 'top_items']:
        rows = summary.loc[summary.summary_group.eq(group)]
        assert rows.entity.tolist() == [str(i) for i in range(1, 21)]
        assert rows['rank'].tolist() == list(range(1, 21))
    eligible = summary.loc[summary.summary_group.eq('categories_1000_visitors')]
    assert eligible.entity.tolist() == [str(i) for i in range(2, 22)]
    unknown = summary.loc[summary.summary_group.eq('unknown_category')]
    assert len(unknown) == 1
    assert unknown.iloc[0].transaction_events == 99
    assert pd.isna(unknown.iloc[0]['rank'])


def test_transaction_count_precedes_id_and_values_are_preserved():
    full = pd.DataFrame([
        ['category', '5', 10, 2, 3, 10, .2],
        ['category', '9', 20, 3, 4, 20, .15],
        ['product', '8', 20, 3, 4, 20, .15],
    ], columns=['grain', 'entity', 'views', 'carts', 'transaction_events',
                'visitors', 'visitor_conversion_rate'])
    summary = category_summary(full)
    known = summary.loc[summary.summary_group.eq('top_categories')]
    assert known.entity.tolist() == ['9', '5']
    pd.testing.assert_frame_equal(known[list(full.columns)].reset_index(drop=True),
                                  full.iloc[[1, 0]].reset_index(drop=True))
