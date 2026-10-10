"""Build and execute the customer journey and model analysis notebooks."""
import argparse
import os

from .common import ROOT, PROCESSED
from .publication import sanitize_text


SETUP = '''from pathlib import Path
import sys
import json
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Image

ROOT = Path.cwd()
if not (ROOT / 'src').exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
assert (ROOT / 'reports/data_quality.json').exists(), 'Run python -m src.pipeline all first'
REPORTS = ROOT / 'reports'
'''


def build_notebooks():
    import nbformat as nbf
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    definitions = {
        '01_customer_journey.ipynb': [
            md('# Customer journey analytics\n\n'
               'Business question: where do observed visitors leave the shopping journey? '
               'Use audited Retailrocket data, explicit SQL denominators and temporal metadata. '
               '[Source and attribution](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset): '
               'Retailrocket, CC BY-NC-SA 4.0. Data was not collected by the project author.'),
            code(SETUP),
            md('## 1. Audit before analysis\n\nStructural nulls differ from missing required fields. '
               'Exact duplicate event rows are removed. The JSON contains input hashes and validations.'),
            code("""quality = json.loads((REPORTS / 'data_quality.json').read_text())
audit_rows = []
for name, stats in quality['files'].items():
    audit_rows.append({'file': name, 'rows': stats['rows'],
                       'duplicates': stats['exact_duplicate_rows'],
                       'missing_values': sum(stats['missing'].values()),
                       'UTC_start': stats.get('utc_start', 'not timestamped'),
                       'UTC_end': stats.get('utc_end', 'not timestamped')})
display(pd.DataFrame(audit_rows))
display(pd.Series(quality['files']['events.csv']['event_distribution'], name='raw_events'))
print('Event-time category coverage:', f"{quality['validation']['event_category_coverage_asof']:.2%}")
"""),
            md('## 2. SQL funnel\n\nRequire strictly increasing timestamps. This is a visitor '
               'funnel across the capture period; it can span sessions and items.'),
            code("""with sqlite3.connect(ROOT / 'data/processed/retailrocket.sqlite') as con:
    funnel = pd.read_sql_query((ROOT / 'sql/01_funnel.sql').read_text(), con)
display(funnel)
display(Image(filename=str(REPORTS / 'figures/funnel.png')))
"""),
            md('## 3. Conversion, abandonment and returning visitors\n\n'
               'Cart abandonment is at the cart-event level, with 24h same-item follow-up. '
               'Returning visitors are a retrospective segment, not a model feature.'),
            code("""with sqlite3.connect(ROOT / 'data/processed/retailrocket.sqlite') as con:
    display(pd.read_sql_query((ROOT / 'sql/02_conversion.sql').read_text(), con))
    display(pd.read_sql_query((ROOT / 'sql/03_returning_visitors.sql').read_text(), con))
"""),
            md('## 4. Category performance and trends\n\n'
               'Unknown categories are retained. Restrict a descriptive category table to '
               'at least 1,000 observed visitors; rate denominators overlap between categories.'),
            code("""performance = pd.read_csv(REPORTS / '04_product_category.csv', dtype={'entity': str})
eligible = performance.query("summary_group == 'categories_1000_visitors'").sort_values('rank')
display(eligible.head(5).drop(columns=['summary_group', 'rank']))
display(Image(filename=str(REPORTS / 'figures/event_trend.png')))
"""),
            md('## 5. Product interpretation\n\n'
               'Form hypotheses about abandonment and recurring visits, then test a reminder '
               'against a randomized no-contact holdout. High propensity does not prove uplift. '
               'Anonymous categories and missing prices cannot support product names or revenue estimates. '
               'Review `reports/results.md` for the measured results and proposed experiments.'),
        ],
        '02_conversion_model.ipynb': [
            md('# Next-day purchase prediction\n\n'
               'A fixed-midnight ranking experiment with chronological evaluation. '
               'Training code is reproducible in `src/`; this notebook inspects evidence, '
               'rather than requiring hidden training state. Dataset: Retailrocket, CC BY-NC-SA 4.0.'),
            code(SETUP),
            md('## 1. Prediction contract and leakage\n\n'
               'Active visitors in `(t-24h,t]`; purchase label in `(t,t+24h]`. '
               'A transaction at exactly t is past history. Category joins are backward in time. '
               'Purge target horizons at chronological split boundaries. Daily observations '
               'predict returning purchases, not purchases completed earlier the same day.'),
            code("""feature_audit = json.loads((REPORTS / 'feature_audit.json').read_text())
display(pd.DataFrame(feature_audit['splits']).T)
print('Features:', ', '.join(feature_audit['features']))
print('Point-in-time checks:', feature_audit['validation'])
"""),
            md('## 2. Baselines and evaluation\n\n'
               'Compare constant prevalence, weighted logistic regression, random forest and '
               'usable optional boosting. Select model by validation AP and threshold by '
               'validation F1. Test results do not change that decision. AP is the reported '
               'PR-AUC convention, and Precision@K uses a 1% capacity.'),
            code("""results = json.loads((REPORTS / 'model_results.json').read_text())
print('Selected using validation AP:', results['selected_model'])
print('Optional model status:', results['optional_models']['boosting'])
metrics = ['model', 'pr_auc_ap', 'roc_auc', 'precision', 'recall', 'f1', 'precision_at_1pct']
display(pd.read_csv(REPORTS / 'validation_metrics.csv')[metrics])
display(pd.read_csv(REPORTS / 'test_metrics.csv')[metrics])
"""),
            code("""display(Image(filename=str(REPORTS / 'figures/precision_recall_curve.png')))
display(Image(filename=str(REPORTS / 'figures/confusion_matrix.png')))
display(Image(filename=str(REPORTS / 'figures/feature_importance.png')))
"""),
            md('## 3. Error slices, calibration and uncertainty\n\n'
               'Slices overlap and cannot be summed. Weighted-model scores are not calibrated '
               'purchase probabilities. A day-block AP interval estimates only approximate uncertainty.'),
            code("""display(pd.read_csv(REPORTS / 'error_slices.csv'))
display(pd.read_csv(REPORTS / 'calibration_bins.csv'))
errors = json.loads((REPORTS / 'error_analysis.json').read_text())
print('AP uncertainty:', errors['uncertainty'])
print('Daily top-1% pooled precision:', results['daily_capacity_precision_at_1pct'])
"""),
            md('## 4. Saved model and scoring parity\n\n'
               'The model bundle keeps preprocessing and threshold together. Test locally '
               'computed features against the same predictions without publishing visitor records.'),
            code("""import joblib
import numpy as np
from src.predict import score_features
from src.features import FEATURES

bundle = joblib.load(ROOT / 'data/processed/selected_model.joblib')
snapshots = pd.read_pickle(ROOT / 'data/processed/visitor_snapshots.pkl')
test_sample = snapshots.loc[snapshots.split.eq('test')].head(100)
scored = score_features(bundle, test_sample)
stored = pd.read_csv(ROOT / 'data/processed/test_predictions.csv').head(100)
np.testing.assert_allclose(scored.score, stored.score, rtol=1e-10, atol=1e-12)
assert np.array_equal(scored.prediction, stored.prediction)
print('Scoring parity passed on 100 held-out snapshots.')
print('Model version:', bundle['metadata']['model_version'])
"""),
            md('## 5. Next experiment\n\n'
               'Compare hourly or event-triggered observations on a fresh forward holdout. '
               'Preserve the full 24h horizon purge and avoid repeated test tuning. '
               'For a product intervention, evaluate incremental transactions in a randomized '
               'holdout, not only offline propensity.'),
        ],
    }
    for name, cells in definitions.items():
        notebook = nbf.v4.new_notebook(cells=cells)
        notebook.metadata['kernelspec'] = {'display_name': 'Retailrocket (.venv)',
                                            'language': 'python', 'name': 'retailrocket'}
        notebook.metadata['language_info'] = {'name': 'python'}
        nbf.validate(notebook)
        nbf.write(notebook, ROOT/'notebooks'/name)
    print('Built customer journey and model notebooks.', flush=True)


def execute_notebooks(name=None):
    import nbformat
    from nbclient import NotebookClient
    targets = [path for path in sorted((ROOT/'notebooks').glob('*.ipynb'))
               if name is None or path.name == name]
    if not targets:
        raise ValueError(f'No matching notebook: {name}')
    for environment_name, folder in [('IPYTHONDIR', 'ipython'), ('JUPYTER_RUNTIME_DIR', 'jupyter_runtime'),
                                     ('MPLCONFIGDIR', 'matplotlib')]:
        location = PROCESSED/folder
        location.mkdir(exist_ok=True)
        os.environ[environment_name] = str(location)
    for path in targets:
        notebook = nbformat.read(path, as_version=4)
        client = NotebookClient(notebook, timeout=180, kernel_name='retailrocket',
                                resources={'metadata': {'path': str(ROOT)}})
        client.execute()
        # Only saved text outputs are sanitized; code and numeric results stay intact.
        for cell in notebook.cells:
            for output in cell.get('outputs', []):
                if 'text' in output:
                    output['text'] = sanitize_text(output['text'])
                for mime, value in output.get('data', {}).items():
                    if mime.startswith('text/') and isinstance(value, str):
                        output['data'][mime] = sanitize_text(value)
                if 'traceback' in output:
                    output['traceback'] = [sanitize_text(line) for line in output['traceback']]
        nbformat.write(notebook, path)
        assert all(cell.execution_count is not None for cell in notebook.cells if cell.cell_type == 'code')
        assert all(output.output_type != 'error' for cell in notebook.cells if cell.cell_type == 'code'
                   for output in cell.get('outputs', []))
        print(f'Executed and validated {path.name}.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['build', 'execute'])
    parser.add_argument('--name', help='Execute only this notebook filename')
    args = parser.parse_args()
    if args.action == 'build':
        build_notebooks()
    else:
        execute_notebooks(args.name)


if __name__ == '__main__':
    main()
