"""Read-only artifact checks plus a persisted final verification record.

Run after the full pipeline: python -m src.verify
No data ingestion or model training is performed.
"""
import hashlib
import json
import re
import sqlite3
import subprocess
import sys
import tempfile
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from scipy.special import expit

from .common import ROOT, DB, RAW, PROCESSED, REPORTS, FIGURES, DAY, save_json
from .features import FEATURES, validate_snapshots, assign_time_splits
from .ingest import source_fingerprint
from .modeling import evaluate
from .predict import score_features


def verify():
    results = {'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'checks': {}}
    data = json.loads((REPORTS/'data_quality.json').read_text())
    incomplete = list(PROCESSED.glob('*.building*'))
    assert not incomplete, f'Incomplete ingestion outputs: {incomplete}'
    print('Checking SQLite integrity and persisted row counts...', flush=True)
    with sqlite3.connect(DB.as_uri()+'?mode=ro', uri=True) as con:
        integrity = [row[0] for row in con.execute('PRAGMA integrity_check')]
        assert integrity == ['ok'], integrity
        assert not con.execute('PRAGMA foreign_key_check').fetchall()
        counts = {table: con.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]
                  for table in data['validation']['sqlite_row_counts']}
        assert counts == data['validation']['sqlite_row_counts']
        assert con.execute('''SELECT COUNT(*) FROM event_categories
            WHERE category_timestamp>timestamp''').fetchone()[0] == 0
    results['checks']['sqlite'] = {'integrity': 'ok', 'foreign_key_violations': 0,
                                   'row_counts': counts, 'future_category_assignments': 0}
    for name, expected in data['files'].items():
        assert source_fingerprint(RAW/name)['sha256'] == expected['sha256'], f'Input changed: {name}'
    results['checks']['raw_fingerprints_match'] = True
    print('Checking readable reports, images, snapshots and model...', flush=True)
    json_files = []
    for path in REPORTS.glob('*.json'):
        json.loads(path.read_text())
        json_files.append(path.name)
    csv_rows = {path.name: len(pd.read_csv(path, low_memory=False)) for path in REPORTS.glob('*.csv')}
    image_sizes = {}
    for name in ['funnel', 'event_trend', 'confusion_matrix', 'precision_recall_curve', 'feature_importance']:
        with Image.open(FIGURES/f'{name}.png') as im:
            image_sizes[name] = list(im.size)
            im.verify()
    snapshots = pd.read_pickle(PROCESSED/'visitor_snapshots.pkl')
    validate_snapshots(snapshots)
    np.testing.assert_array_equal(assign_time_splits(snapshots).split, snapshots.split)
    assert len(snapshots) == json.loads((REPORTS/'feature_audit.json').read_text())['rows']
    assert (snapshots.observation_time-DAY >= data['files']['events.csv']['timestamp_min']).all()
    assert (snapshots.label_end <= data['files']['events.csv']['timestamp_max']).all()
    bundle = joblib.load(PROCESSED/'selected_model.joblib')
    assert bundle['features'] == FEATURES
    assert np.isfinite(bundle['threshold']) and 0 <= bundle['threshold'] <= 1
    manifest = json.loads((REPORTS/'run_manifest.json').read_text())
    assert bundle['metadata'] == manifest
    code_hash = hashlib.sha256()
    paths = sorted(list((ROOT/'src').glob('*.py')) + list((ROOT/'sql').glob('*.sql')))
    for path in paths:
        code_hash.update(str(path.relative_to(ROOT)).encode())
        code_hash.update(path.read_bytes())
    assert code_hash.hexdigest() == manifest['code_sha256'], 'Refresh metadata with python -m src.pipeline report'
    test = snapshots.loc[snapshots.split.eq('test')]
    stored = pd.read_csv(PROCESSED/'test_predictions.csv')
    np.testing.assert_array_equal(test[['visitorid', 'observation_time', 'label']].to_numpy(),
                                  stored[['visitorid', 'observation_time', 'label']].to_numpy())
    with warnings.catch_warnings(record=True) as numeric_warnings:
        warnings.simplefilter('always', RuntimeWarning)
        actual = score_features(bundle, test)
    np.testing.assert_allclose(actual.score, stored.score, rtol=1e-10, atol=1e-12)
    np.testing.assert_array_equal(actual.prediction, stored.prediction)
    numerical = {'all_test_scores_finite': bool(np.isfinite(actual.score).all()),
                 'persisted_prediction_parity': True,
                 'runtime_warnings': sorted(set(str(item.message) for item in numeric_warnings))}
    model = bundle['model']
    if hasattr(model, 'named_steps') and 'logisticregression' in model.named_steps:
        transformed = model[:-1].transform(test[FEATURES])
        logistic = model.named_steps['logisticregression']
        independent_logits = np.einsum('ij,j->i', transformed, logistic.coef_[0])+logistic.intercept_[0]
        independent_probabilities = expit(independent_logits)
        np.testing.assert_allclose(actual.score, independent_probabilities, rtol=1e-10, atol=1e-12)
        numerical['independent_einsum_probability_max_difference'] = float(
            np.max(np.abs(actual.score.to_numpy()-independent_probabilities)))
        numerical['solver_iterations'] = int(logistic.n_iter_[0])
        assert logistic.n_iter_[0] < logistic.max_iter
        assert np.isfinite(logistic.coef_).all()
    summary = json.loads((REPORTS/'model_results.json').read_text())
    expected = next(row for row in summary['test_metrics'] if row['model'] == summary['selected_model'])
    recomputed = evaluate(test.label, actual.score.to_numpy(), bundle['threshold'])
    for metric in ['pr_auc_ap', 'roc_auc', 'precision', 'recall', 'f1', 'precision_at_1pct']:
        np.testing.assert_allclose(recomputed[metric], expected[metric], rtol=1e-10, atol=1e-12)
    results['checks']['readable_artifacts'] = {'json': sorted(json_files), 'csv_rows': csv_rows,
                                              'figure_dimensions': image_sizes, 'snapshots': len(snapshots)}
    results['checks']['model'] = {'version': manifest['model_version'], 'features': FEATURES,
                                 'test_rows_checked': len(test), 'metrics_reconcile': True,
                                 'numerical_verification': numerical}
    # Notebook checks are optional for a lightweight environment, but required
    # if notebook artifacts have been delivered.
    import nbformat
    notebooks = {}
    for path in (ROOT/'notebooks').glob('*.ipynb'):
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        cells = [cell for cell in notebook.cells if cell.cell_type == 'code']
        assert all(cell.execution_count is not None for cell in cells), f'Unexecuted notebook: {path.name}'
        assert not [output for cell in cells for output in cell.outputs if output.output_type == 'error']
        notebooks[path.name] = {'executed_code_cells': len(cells), 'errors': 0}
    assert len(notebooks) == 2
    results['checks']['notebooks'] = notebooks
    readme = (ROOT/'README.md').read_text()
    local_links = re.findall(r'\]\(([^)]+)\)', readme)
    for target in local_links:
        if not target.startswith(('https:', 'http:', '#')) and target != 'reports/verification.json':
            assert (ROOT/target.split('#')[0]).exists(), f'Broken README link: {target}'
    assert 'Run the pipeline to generate verified results here.' not in readme
    results['checks']['readme_links'] = 'ok'
    print('Checking Git exclusions and regression tests...', flush=True)
    with tempfile.TemporaryDirectory(prefix='retailrocket-ignore-') as directory:
        location = Path(directory)
        subprocess.run(['git', 'init', '-q', str(location)], check=True)
        (location/'.gitignore').write_text((ROOT/'.gitignore').read_text())
        ignored = ['data/raw/events.csv', 'data/processed/retailrocket.sqlite', '.venv/pyvenv.cfg',
                   'src/__pycache__/features.pyc', '.env', 'credentials.json', 'private.key', 'archive/events.csv']
        included = ['README.md', 'src/features.py', 'reports/test_metrics.csv',
                    'reports/figures/funnel.png', 'data/raw/.gitkeep', 'notebooks/01_customer_journey.ipynb']
        for path in ignored+included:
            code = subprocess.run(['git', '-C', directory, 'check-ignore', '-q', '--no-index', path]).returncode
            assert code == (0 if path in ignored else 1), f'Unexpected Git exclusion: {path}'
    status = subprocess.run(['git', 'status', '--short'], cwd=ROOT, capture_output=True, text=True)
    results['checks']['git'] = {'repository_initialized': status.returncode == 0,
                               'exclusion_rules_verified': True,
                               'status': status.stdout.strip() if status.returncode == 0 else 'Not a Git repository'}
    tests = subprocess.run([sys.executable, '-m', 'pytest', '-q'], cwd=ROOT, capture_output=True, text=True)
    assert tests.returncode == 0, tests.stdout+tests.stderr
    results['checks']['pytest'] = {'returncode': tests.returncode, 'output': tests.stdout.strip()}
    lock = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True, check=True)
    (ROOT/'requirements-lock.txt').write_text(
        '# Captured local environment: macOS ARM64, Python 3.9.6.\n'
        '# Optional XGBoost is installed but lacks libomp in this run.\n'+lock.stdout)
    results['all_checks_passed'] = True
    save_json(REPORTS/'verification.json', results)
    print(f'All checks passed; saved {REPORTS / "verification.json"}.', flush=True)
    return results


if __name__ == '__main__':
    verify()
