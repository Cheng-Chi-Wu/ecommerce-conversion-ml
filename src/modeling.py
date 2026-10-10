"""Chronological model comparison, validation-only selection and held-out metrics."""
import importlib.util
import time

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             precision_recall_curve, precision_score, recall_score,
                             f1_score, ConfusionMatrixDisplay)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .common import FIGURES, PROCESSED, REPORTS, SEED, save_json
from .features import FEATURES, validate_snapshots
from .plots import plot_precision_recall
from .publication import sanitize_text


def select_threshold(y, scores):
    """Choose maximum validation F1; prefer higher threshold when F1 ties."""
    precision, recall, thresholds = precision_recall_curve(y, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-15)
    if not len(thresholds):
        return 0.5
    best = np.flatnonzero(np.isclose(f1, f1.max(), rtol=1e-12, atol=1e-12))[-1]
    return float(thresholds[best])


def precision_at_k(y, scores, fraction=.01):
    """Top ceil(fraction*n) rows; reproducible, label-independent tie breaking."""
    target = np.asarray(y)
    k = max(1, int(np.ceil(len(target) * fraction)))
    tie_breaker = np.random.default_rng(SEED).random(len(target))
    order = np.lexsort((tie_breaker, -np.asarray(scores)))
    return float(target[order[:k]].mean()), k


def evaluate(y, scores, threshold):
    assert np.isfinite(scores).all()
    assert set(np.unique(y)) == {0, 1}, 'Both classes required in evaluation split'
    predictions = np.asarray(scores) >= threshold
    p_at_k, k = precision_at_k(y, scores)
    prevalence = float(np.mean(y))
    return {
        'pr_auc_ap': float(average_precision_score(y, scores)),
        'roc_auc': float(roc_auc_score(y, scores)),
        'precision': float(precision_score(y, predictions, zero_division=0)),
        'recall': float(recall_score(y, predictions, zero_division=0)),
        'f1': float(f1_score(y, predictions, zero_division=0)),
        'precision_at_1pct': p_at_k, 'k': k,
        'lift_at_1pct': p_at_k / prevalence, 'positive_rate': prevalence,
        'threshold': float(threshold), 'rows': len(y), 'positives': int(np.sum(y)),
    }


def train_models():
    frame = pd.read_pickle(PROCESSED / 'visitor_snapshots.pkl')
    validate_snapshots(frame)
    partitions = {name: frame.loc[frame.split.eq(name)] for name in ['train', 'validation', 'test']}
    for part in partitions.values():
        assert set(part.label.unique()) == {0, 1}
    train, validation, test = [partitions[k] for k in ['train', 'validation', 'test']]
    assert train.label_end.max() < validation.observation_time.min()
    assert validation.label_end.max() < test.observation_time.min()
    models = {
        'Naive prevalence baseline': DummyClassifier(strategy='prior'),
        'Logistic Regression': make_pipeline(FunctionTransformer(np.log1p), StandardScaler(),
                                             LogisticRegression(class_weight='balanced',
                                                                max_iter=1000, random_state=SEED)),
        'Random Forest': RandomForestClassifier(n_estimators=150, max_depth=12,
                                                min_samples_leaf=30, max_features='sqrt',
                                                class_weight='balanced_subsample',
                                                n_jobs=4, random_state=SEED),
    }
    optional_status = {}
    optional_import_errors = {}
    XGBClassifier = LGBMClassifier = None
    for module, class_name in [('xgboost', 'XGBClassifier'), ('lightgbm', 'LGBMClassifier')]:
        if importlib.util.find_spec(module) is not None:
            try:
                imported = __import__(module, fromlist=[class_name])
                if module == 'xgboost':
                    XGBClassifier = getattr(imported, class_name)
                else:
                    LGBMClassifier = getattr(imported, class_name)
            except Exception as error:
                # Restrict this catch to optional import/load failures, never fit
                # failures. Installed wheels can still lack a native runtime.
                optional_import_errors[module] = sanitize_text(
                    f'{type(error).__name__}: {str(error).strip()}')
    if XGBClassifier is not None:
        models['XGBoost'] = XGBClassifier(
            n_estimators=250, max_depth=4, learning_rate=.06,
            subsample=.8, colsample_bytree=.9, tree_method='hist',
            scale_pos_weight=float((train.label == 0).sum() / train.label.sum()),
            eval_metric='aucpr', n_jobs=4, random_state=SEED)
        optional_status['boosting'] = 'XGBoost available and trained'
    elif LGBMClassifier is not None:
        models['LightGBM'] = LGBMClassifier(n_estimators=250, max_depth=4, num_leaves=15,
                                           learning_rate=.06, class_weight='balanced',
                                           n_jobs=4, random_state=SEED, verbosity=-1)
        optional_status['boosting'] = 'LightGBM available and trained'
    else:
        optional_status['boosting'] = 'Skipped: no usable XGBoost/LightGBM runtime'
    optional_status['import_errors'] = optional_import_errors
    validation_rows, test_rows, fit_seconds, thresholds = [], [], {}, {}
    print(f'Train={len(train):,}; validation={len(validation):,}; test={len(test):,}', flush=True)
    # Complete model choice and threshold choice before evaluating any test score.
    for name, model in models.items():
        print(f'Training {name}...', flush=True)
        started = time.monotonic()
        model.fit(train[FEATURES], train.label)
        fit_seconds[name] = round(time.monotonic() - started, 2)
        scores = model.predict_proba(validation[FEATURES])[:, 1]
        threshold = .5 if name.startswith('Naive') else select_threshold(validation.label, scores)
        thresholds[name] = threshold
        validation_rows.append({'model': name, **evaluate(validation.label, scores, threshold)})
    selected = max(validation_rows, key=lambda row: row['pr_auc_ap'])['model']
    test_scores = {}
    for name, model in models.items():
        scores = model.predict_proba(test[FEATURES])[:, 1]
        test_scores[name] = scores
        test_rows.append({'model': name, **evaluate(test.label, scores, thresholds[name])})
    pd.DataFrame(validation_rows).to_csv(REPORTS / 'validation_metrics.csv', index=False)
    pd.DataFrame(test_rows).to_csv(REPORTS / 'test_metrics.csv', index=False)
    best = models[selected]
    joblib.dump({'model': best, 'features': FEATURES, 'threshold': thresholds[selected],
                 'selected_by': 'validation average precision', 'seed': SEED},
                PROCESSED / 'selected_model.joblib')
    selected_scores = test_scores[selected]
    predictions = selected_scores >= thresholds[selected]
    prediction_frame = test[['visitorid', 'observation_time', 'label']].copy()
    prediction_frame['score'] = selected_scores
    prediction_frame['prediction'] = predictions.astype('int64')
    prediction_frame.to_csv(PROCESSED / 'test_predictions.csv', index=False)
    daily = []
    for cutoff, part in prediction_frame.groupby('observation_time'):
        p, k = precision_at_k(part.label, part.score)
        daily.append({'observation_time': cutoff, 'rows': len(part), 'k': k,
                      'selected_true_positives': int(round(p*k)), 'precision_at_1pct': p,
                      'positive_rate': float(part.label.mean())})
    daily = pd.DataFrame(daily)
    daily.to_csv(REPORTS / 'daily_topk.csv', index=False)
    daily_precision = float(daily.selected_true_positives.sum() / daily.k.sum())
    summary = {'selected_model': selected, 'selection_metric': 'validation average precision',
               'threshold_rule': 'Maximum validation F1 (naive threshold fixed at 0.5)',
               'precision_at_k_rule': 'Global top ceil(1% of test snapshots); seeded tie breaking',
               'daily_capacity_precision_at_1pct': daily_precision,
               'fit_seconds': fit_seconds, 'optional_models': optional_status,
               'features': FEATURES, 'validation_metrics': validation_rows,
               'test_metrics': test_rows, 'model_parameters': {
                   name: {k: str(v) for k, v in model.get_params().items()}
                   for name, model in models.items()}}
    save_json(REPORTS / 'model_results.json', summary)
    plot_precision_recall(test.label, test_scores, FIGURES / 'precision_recall_curve.png')
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(test.label, predictions, display_labels=['No purchase', 'Purchase'],
                                             ax=ax, colorbar=False, values_format='d', cmap='Blues')
    ax.set_title(f'{selected}: validation-selected threshold')
    fig.tight_layout()
    fig.savefig(FIGURES / 'confusion_matrix.png')
    plt.close(fig)
    if hasattr(best, 'feature_importances_'):
        importance = best.feature_importances_
        importance_type = 'Model feature importance (association, not causation)'
    elif hasattr(best, 'named_steps'):
        importance = np.abs(best.named_steps['logisticregression'].coef_[0])
        importance_type = 'Absolute standardized log-feature coefficient'
    else:
        importance = np.zeros(len(FEATURES))
        importance_type = 'Constant baseline: no feature importance'
    feature_importance = pd.DataFrame({'feature': FEATURES, 'importance': importance}).sort_values('importance')
    feature_importance.to_csv(REPORTS / 'feature_importance.csv', index=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(feature_importance.feature, feature_importance.importance, color='#4361ee')
    ax.set(xlabel=importance_type, title=f'{selected}: predictive signals')
    fig.tight_layout()
    fig.savefig(FIGURES / 'feature_importance.png')
    plt.close(fig)
    print(f'Validated model comparison; selected {selected} using validation AP.', flush=True)
    return summary
