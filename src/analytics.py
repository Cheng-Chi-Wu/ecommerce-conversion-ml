"""Execute versioned SQL analyses and generate descriptive figures."""
import sqlite3

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from .common import DB, FIGURES, REPORTS, ROOT, PROCESSED, save_json

plt.rcParams.update({'figure.dpi': 140, 'axes.spines.top': False,
                     'axes.spines.right': False, 'font.size': 10})


def category_summary(frame):
    """Fixed top-20 lists, descending transactions then ascending numeric ID."""
    categories = frame.loc[frame.grain.eq('category') & frame.entity.ne('unknown')]
    items = frame.loc[frame.grain.eq('product')]
    groups = [('top_categories', categories), ('top_items', items),
              ('categories_1000_visitors', categories.loc[categories.visitors.ge(1000)])]
    selected = []
    for name, candidates in groups:
        ranked = candidates.assign(entity_id=pd.to_numeric(candidates.entity))
        ranked = ranked.sort_values(['transaction_events', 'entity_id'],
                                    ascending=[False, True]).head(20).drop(columns='entity_id')
        ranked = ranked.assign(summary_group=name, rank=range(1, len(ranked) + 1))
        selected.append(ranked)
    unknown = frame.loc[frame.grain.eq('category') & frame.entity.eq('unknown')].copy()
    selected.append(unknown.assign(summary_group='unknown_category', rank=pd.NA))
    summary = pd.concat(selected, ignore_index=True)
    summary['rank'] = summary['rank'].astype('Int64')
    return summary[['summary_group', 'rank'] + list(frame.columns)]


def run_analytics():
    outputs = {}
    with sqlite3.connect(DB) as con:
        for path in sorted((ROOT / 'sql').glob('0*.sql')):
            print(f'Executing {path.name}...', flush=True)
            frame = pd.read_sql_query(path.read_text(), con)
            assert not frame.empty, f'No output for {path.name}'
            if path.stem == '04_product_category':
                PROCESSED.mkdir(parents=True, exist_ok=True)
                frame.to_csv(PROCESSED / '04_product_category_full.csv', index=False)
                summary = category_summary(frame)
                summary.to_csv(REPORTS / f'{path.stem}.csv', index=False)
                save_json(REPORTS / 'category_summary.json', {
                    'full_rows': len(frame), 'summary_rows': len(summary),
                    'full_output': 'data/processed/04_product_category_full.csv',
                    'ranking': 'transaction_events descending, numeric entity ID ascending',
                    'groups': {'top_categories': 'Top 20 known categories; no visitor filter',
                               'top_items': 'Top 20 items; no visitor filter',
                               'categories_1000_visitors': 'Top 20 known categories with at least 1000 visitors',
                               'unknown_category': 'All events without a historical category; unranked'},
                    'overlap': 'Groups overlap; do not add their counts together'})
            else:
                frame.to_csv(REPORTS / f'{path.stem}.csv', index=False)
            outputs[path.stem] = frame
    funnel = outputs['01_funnel']
    assert funnel.visitors.is_monotonic_decreasing
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(funnel.stage, funnel.visitors, color=['#4361ee', '#4cc9f0', '#f72585'])
    ax.invert_yaxis()
    for bar, count in zip(bars, funnel.visitors):
        ax.text(count, bar.get_y() + bar.get_height()/2, f' {count:,}', va='center')
    ax.set(xlabel='Unique visitors (ordered events)', title='Observed visitor funnel: view → cart → purchase')
    ax.set_xlim(0, funnel.visitors.max()*1.18)
    fig.tight_layout()
    fig.savefig(FIGURES / 'funnel.png')
    plt.close(fig)
    daily = outputs['05_trends'].query("grain == 'daily'").copy()
    daily['period'] = pd.to_datetime(daily.period)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(daily.period, daily.views, color='#4361ee', label='Views')
    axes[1].plot(daily.period, daily.carts, label='Cart events', color='#4cc9f0')
    axes[1].plot(daily.period, daily.transaction_events, label='Transaction item events', color='#f72585')
    for ax in axes:
        ax.set_ylabel('Events / UTC day')
        ax.legend()
    axes[0].set_title('Daily customer activity (separate scales for visibility)')
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIGURES / 'event_trend.png')
    plt.close(fig)
    save_json(REPORTS / 'sql_validation.json', {'nonempty_queries': list(outputs),
                                               'ordered_funnel_monotonic': True,
                                               'daily_rows': len(daily)})
    return outputs
