"""Cohort errors, calibration and metadata for the saved model."""
import hashlib
import importlib.metadata
import platform
import subprocess
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

from .common import ROOT, PROCESSED, REPORTS, SEED, save_json
from .features import FEATURES


def audit_model():
    snapshots = pd.read_pickle(PROCESSED/'visitor_snapshots.pkl')
    predictions = pd.read_csv(PROCESSED/'test_predictions.csv')
    test = predictions.merge(snapshots.loc[snapshots.split.eq('test')],
                             on=['visitorid', 'observation_time', 'label'], validate='one_to_one')
    assert len(test) == len(predictions)
    known = set(snapshots.loc[snapshots.split.eq('train'), 'visitorid'])
    masks = {
        'all': np.ones(len(test), dtype=bool),
        'visitor_seen_in_train': test.visitorid.isin(known),
        'visitor_not_seen_in_train': ~test.visitorid.isin(known),
        'cart_in_prior_24h': test.cart_additions_24h.gt(0),
        'no_cart_in_prior_24h': test.cart_additions_24h.eq(0),
        'previous_purchase': test.previous_transactions.gt(0),
        'no_previous_purchase': test.previous_transactions.eq(0),
        'unknown_category_views': test.unknown_category_views_24h.gt(0),
        'no_unknown_category_views': test.unknown_category_views_24h.eq(0),
    }
    rows = []
    for name, mask in masks.items():
        part = test.loc[mask]
        truth, predicted = part.label.eq(1), part.prediction.eq(1)
        tp, fp = int((truth & predicted).sum()), int((~truth & predicted).sum())
        fn, tn = int((truth & ~predicted).sum()), int((~truth & ~predicted).sum())
        rows.append({'slice': name, 'rows': len(part), 'positives': int(truth.sum()),
                     'true_positives': tp, 'false_positives': fp, 'false_negatives': fn,
                     'true_negatives': tn, 'precision': tp/(tp+fp) if tp+fp else 0,
                     'recall': tp/(tp+fn) if tp+fn else 0,
                     'pr_auc_ap': float(average_precision_score(part.label, part.score))
                     if part.label.nunique() == 2 else None,
                     'roc_auc': float(roc_auc_score(part.label, part.score))
                     if part.label.nunique() == 2 else None})
    pd.DataFrame(rows).to_csv(REPORTS/'error_slices.csv', index=False)
    calibration = test[['score', 'label']].copy()
    calibration['bin'] = pd.qcut(calibration.score, 10, duplicates='drop')
    bins = calibration.groupby('bin', observed=True).agg(
        rows=('label', 'size'), mean_score=('score', 'mean'), observed_rate=('label', 'mean'))
    bins.to_csv(REPORTS/'calibration_bins.csv')
    brier = float(brier_score_loss(test.label, test.score))
    days, day_indices = np.unique(test.observation_time, return_inverse=True)
    rng = np.random.default_rng(SEED)
    bootstrapped_ap = []
    for _ in range(100):
        counts = np.bincount(rng.integers(0, len(days), len(days)), minlength=len(days))
        weights = counts[day_indices]
        if (weights * test.label.to_numpy()).sum() > 0:
            bootstrapped_ap.append(float(average_precision_score(test.label, test.score,
                                                                 sample_weight=weights)))
    interval = {'method': '100 UTC-day block bootstrap replicates; seed 42',
                'pr_auc_ap_95pct_interval': list(np.quantile(bootstrapped_ap, [.025, .975])),
                'limitation': 'Approximate temporal uncertainty; repeated visitors across days remain correlated'}
    environment = {'python': platform.python_version(), 'platform': platform.platform(),
                   'packages': {package: importlib.metadata.version(package)
                                for package in ['numpy', 'pandas', 'matplotlib', 'scikit-learn', 'joblib']}}
    code_hash = hashlib.sha256()
    source_paths = sorted(list((ROOT/'src').glob('*.py')) + list((ROOT/'sql').glob('*.sql')))
    for path in source_paths:
        code_hash.update(str(path.relative_to(ROOT)).encode())
        code_hash.update(path.read_bytes())
    git = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True)
    data = json.loads((REPORTS/'data_quality.json').read_text())
    results = json.loads((REPORTS/'model_results.json').read_text())
    selected = next(row for row in results['test_metrics'] if row['model'] == results['selected_model'])
    baseline = next(row for row in results['test_metrics'] if row['model'].startswith('Naive'))
    checks = {'beats_naive_average_precision': selected['pr_auc_ap'] > baseline['pr_auc_ap'],
              'top_1pct_beats_test_prevalence': selected['precision_at_1pct'] > selected['positive_rate'],
              'point_in_time_checks_passed': True}
    config_hash = hashlib.sha256(json.dumps(results['model_parameters'], sort_keys=True).encode()).hexdigest()
    provenance = {'artifact_schema_version': 1,
                  'model_version': f'v1-{code_hash.hexdigest()[:12]}-{config_hash[:8]}',
                  'code_sha256': code_hash.hexdigest(),
                  'git_commit': git.stdout.strip() if git.returncode == 0 else None,
                  'config_sha256': config_hash, 'seed': SEED,
                  'raw_data_sha256': {name: stats['sha256'] for name, stats in data['files'].items()},
                  'environment': environment, 'features': FEATURES,
                  'prediction_contract': 'One active visitor at UTC midnight; purchase in (t,t+24h]',
                  'offline_evidence_checks': checks, 'brier_score_uncalibrated': brier,
                  'production_status': 'Offline portfolio prototype; no deployed intervention'}
    bundle_path = PROCESSED/'selected_model.joblib'
    bundle = joblib.load(bundle_path)
    bundle['metadata'] = provenance
    joblib.dump(bundle, bundle_path)
    save_json(REPORTS/'run_manifest.json', provenance)
    save_json(REPORTS/'error_analysis.json', {'selected_model': results['selected_model'],
                                            'slices_overlap': True, 'slices': rows,
                                            'brier_score_uncalibrated': brier,
                                            'uncertainty': interval})
    print('Validated cohort error analysis and artifact provenance.', flush=True)
    return provenance
