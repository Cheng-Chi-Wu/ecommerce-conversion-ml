"""Execute versioned SQL analyses and generate descriptive figures."""
import sqlite3

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from .common import DB, FIGURES, REPORTS, ROOT, save_json

plt.rcParams.update({'figure.dpi': 140, 'axes.spines.top': False,
                     'axes.spines.right': False, 'font.size': 10})


def run_analytics():
    outputs = {}
    with sqlite3.connect(DB) as con:
        for path in sorted((ROOT / 'sql').glob('0*.sql')):
            print(f'Executing {path.name}...', flush=True)
            frame = pd.read_sql_query(path.read_text(), con)
            assert not frame.empty, f'No output for {path.name}'
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
