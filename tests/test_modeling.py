import numpy as np
import pytest

from src.modeling import evaluate, precision_at_k, select_threshold


def test_precision_at_k_uses_ceiling_and_deterministic_ties():
    y = np.array([1, 0, 1, 0, 0])
    scores = np.array([.9, .8, .7, .1, .1])
    assert precision_at_k(y, scores, .21) == (.5, 2)
    assert precision_at_k(y, np.ones(5), .4) == precision_at_k(y, np.ones(5), .4)


def test_threshold_selection_and_imbalanced_metrics():
    y = np.array([0, 0, 1, 1])
    scores = np.array([.1, .2, .7, .8])
    threshold = select_threshold(y, scores)
    result = evaluate(y, scores, threshold)
    assert threshold == .7
    assert result['pr_auc_ap'] == 1
    assert result['roc_auc'] == 1
    assert result['precision'] == result['recall'] == result['f1'] == 1


def test_constant_naive_baseline_reports_prevalence_ap():
    y = np.array([0]*99+[1])
    result = evaluate(y, np.full(100, .01), .5)
    assert result['pr_auc_ap'] == pytest.approx(.01)
    assert result['roc_auc'] == .5
    assert result['f1'] == 0
