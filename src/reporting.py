"""Build English results and refresh only the generated README results block."""
import json

import pandas as pd

from .common import ROOT, REPORTS, PROCESSED, FIGURES, markdown_table, save_json
from .evaluation import audit_model
from .plots import plot_precision_recall

BEGIN = '<!-- RESULTS:START -->'
END = '<!-- RESULTS:END -->'


def write_results():
    # Refresh existing reports through the same sanitizer without retraining.
    for path in REPORTS.glob('*.json'):
        save_json(path, json.loads(path.read_text()))
    audit_model()
    data = json.loads((REPORTS/'data_quality.json').read_text())
    audit = json.loads((REPORTS/'feature_audit.json').read_text())
    results = json.loads((REPORTS/'model_results.json').read_text())
    predictions = pd.read_csv(PROCESSED/'test_predictions.csv')
    plot_precision_recall(predictions.label, {results['selected_model']: predictions.score},
                          FIGURES/'precision_recall_curve.png')
    errors = json.loads((REPORTS/'error_analysis.json').read_text())
    conversion = pd.read_csv(REPORTS/'02_conversion.csv').set_index('metric').value
    returning = pd.read_csv(REPORTS/'03_returning_visitors.csv').set_index('segment')
    funnel = pd.read_csv(REPORTS/'01_funnel.csv')
    metrics = pd.DataFrame(results['test_metrics'])
    columns = ['model', 'pr_auc_ap', 'roc_auc', 'precision', 'recall', 'f1', 'precision_at_1pct']
    shown = metrics[columns].copy()
    for col in columns[1:]:
        shown[col] = shown[col].map(lambda x: f'{x:.4f}')
    shown.columns = ['Model', 'PR-AUC (AP)', 'ROC-AUC', 'Precision', 'Recall', 'F1', 'Precision@1%']
    best = next(row for row in results['test_metrics'] if row['model'] == results['selected_model'])
    overall = next(row for row in errors['slices'] if row['slice'] == 'all')
    low, high = errors['uncertainty']['pr_auc_ap_95pct_interval']
    import_errors = results['optional_models'].get('import_errors', {})
    boosting_note = ('The installed XGBoost wheel could not load its missing macOS OpenMP runtime '
                     '(`libomp`).'
                     if 'libomp' in ' '.join(import_errors.values()) else
                     'Unavailable optional models are omitted; see the recorded import status.')
    section = f'''The audited source contains **{data['files']['events.csv']['rows']:,} events** from
**2015-05-03 to 2015-09-18 UTC**, across **{data['files']['events.csv']['unique_visitors']:,} visitors**.
Removing {data['files']['events.csv']['exact_duplicate_rows']:,} exact duplicate rows leaves
**{data['cleaning']['events_rows_after_exact_deduplication']:,} events** for analysis.

- Full-period visitor conversion: **{conversion['visitor_conversion_rate']:.2%}** (any transaction / any visitor).
- Ordered view → cart → transaction visitors: **{' → '.join(f'{x:,}' for x in funnel.visitors)}**.
- 24h cart-event abandonment: **{conversion['24h_cart_event_abandonment_rate']:.2%}** across {int(conversion['eligible_cart_events']):,} eligible cart events.
- Retrospective returning-visitor conversion: **{returning.loc['returning', 'visitor_conversion_rate']:.2%}**, versus **{returning.loc['single_day', 'visitor_conversion_rate']:.2%}** for single-day visitors.
- Event-time category coverage: **{data['validation']['event_category_coverage_asof']:.2%}**; future metadata is never backfilled.

**{audit['rows']:,} visitor-day snapshots** were generated. The held-out test has
**{best['rows']:,} snapshots**, **{best['positives']:,} purchases** and **{best['positive_rate']:.4%} prevalence**.
These snapshot metrics answer a different question from full-period visitor conversion.

{markdown_table(shown)}

**{results['selected_model']}** was chosen using validation AP, with its threshold
selected on validation F1. Test Precision@1% is **{best['precision_at_1pct']:.2%}**
at **K={best['k']:,}**, or **{best['lift_at_1pct']:.1f}×** test prevalence.
Selecting the top 1% separately each day yields **{results['daily_capacity_precision_at_1pct']:.2%}** pooled precision.
The threshold produces **{overall['true_positives']:,} true positives**, **{overall['false_positives']:,} false positives**,
and **{overall['false_negatives']:,} false negatives**. The approximate 95% day-block bootstrap
interval for selected-model AP is **[{low:.4f}, {high:.4f}]** (100 replicates).

Boosting status: **{results['optional_models']['boosting']}**.
{boosting_note} Complete parameters, import status and fit times are in
[model_results.json](reports/model_results.json); cohort errors are in
[error_slices.csv](reports/error_slices.csv).
'''
    readme = ROOT/'README.md'
    content = readme.read_text()
    assert content.count(BEGIN) == content.count(END) == 1
    prefix, remainder = content.split(BEGIN)
    _, suffix = remainder.split(END)
    comparison = []
    for row in results['test_metrics']:
        validation = next(item for item in results['validation_metrics']
                          if item['model'] == row['model'])
        comparison.append({'Model': row['model'],
                           'Validation AP': f"{validation['pr_auc_ap']:.4f}",
                           'Test AP': f"{row['pr_auc_ap']:.4f}",
                           'Test precision, top 1%': f"{row['precision_at_1pct']:.2%}"})
    summary = f'''The analysis starts with **{data['files']['events.csv']['rows']:,} events** from
**{data['files']['events.csv']['unique_visitors']:,} visitors**. SQL finds **{conversion['24h_cart_event_abandonment_rate']:.2%}**
cart-event abandonment: no later same-item purchase within 24 hours, among carts
with complete follow-up. Repeated additions count as separate cart events.

The prediction dataset has **{audit['rows']:,} visitor-day snapshots**. The test
period has **{best['rows']:,} snapshots**, of which **{best['positives']:,}** include a
next-day purchase (**{best['positive_rate']:.4%}** prevalence).

{markdown_table(pd.DataFrame(comparison))}

**{results['selected_model']}** was selected using validation AP. Its top
**{best['k']:,}** test snapshots contain **{best['precision_at_1pct']:.2%}** purchasers.
AP means average precision; the top-1% column measures ranking at a fixed capacity.
'''
    readme.write_text(prefix + BEGIN + '\n\n' + summary + '\n' + END + suffix)
    split_rows = []
    for name in ['train', 'validation', 'test', 'purged']:
        row = audit['splits'][name]
        split_rows.append({'split': name, 'rows': row['rows'], 'positives': row['positives'],
                           'first observation UTC': row['observation_start'],
                           'last observation UTC': row['observation_end']})
    category_performance = pd.read_csv(REPORTS/'04_product_category.csv', dtype={'entity': str})
    leaders = category_performance.loc[
        category_performance.summary_group.eq('categories_1000_visitors')].sort_values('rank').head(5)
    leaders = leaders[['entity', 'views', 'carts', 'transaction_events', 'visitors', 'visitor_conversion_rate']]
    leaders['visitor_conversion_rate'] = leaders.visitor_conversion_rate.map(lambda x: f'{x:.2%}')
    (REPORTS/'results.md').write_text('# Results and interpretation\n\n' +
        section.replace('(reports/', '(') +
        '\n## Figures\n\n'
        '![Ordered visitor funnel](figures/funnel.png)\n\n'
        '![Daily event activity](figures/event_trend.png)\n\n'
        '![Selected-model precision-recall curve](figures/precision_recall_curve.png)\n\n'
        '![Validation-threshold confusion matrix](figures/confusion_matrix.png)\n\n'
        '![Model coefficients](figures/feature_importance.png)\n' +
        '\n## Chronological partitions\n\n' + markdown_table(pd.DataFrame(split_rows)) +
        '\n\nPurged rows span two separate boundary dates; their min/max are not one continuous excluded interval.\n'
        '\n## Category activity\n\nTop categories by transaction item events, with at least 1,000 observed visitors.\n\n' +
        markdown_table(leaders) +
        '\n\nCategory IDs are anonymized. Category rates are descriptive, with overlapping visitor denominators.\n'
        '\n## Business recommendations\n\n'
        '1. Audit the high cart abandonment rate, then test a consent-based reminder against a randomized '
        'no-reminder holdout. A cart event is not a unique cart; repeated additions can share a later purchase.\n'
        '2. Use the overnight model to prioritize a small batch for a re-engagement experiment. '
        'Compare incremental transactions per eligible visitor and per contact; track unsubscribe or fatigue '
        'guardrails if such data can be collected. High propensity alone does not imply persuasion.\n'
        '3. Inspect high-volume anonymous category IDs and unknown metadata coverage before proposing '
        'navigation or product changes. The data does not establish checkout causes or revenue impact.\n'
        '4. Treat returning-visitor differences as descriptive. Future observed active days cannot be used '
        'to define a targeting cohort at an earlier observation time.\n'
        '\n## Error analysis and next experiment\n\n'
        'See [error_slices.csv](error_slices.csv) for known/new visitors, cart activity, previous purchases '
        'and unknown categories. Slices overlap. False positives represent unnecessary contacts under a '
        'threshold policy; false negatives represent missed next-day purchasers. F1 is only a transparent '
        'offline default until contact costs are known. A next experiment should compare event-triggered '
        'or hourly observations with the daily policy in a fresh forward holdout, without retuning on this test.\n'
        '\nClass weighting makes raw scores unsuitable as calibrated purchase probabilities. '
        '[calibration_bins.csv](calibration_bins.csv) documents the discrepancy. The day-block interval '
        'is approximate and does not remove all within-visitor correlation across days.\n')
    print('Validated and generated English README/results.', flush=True)
