"""Precision–recall rendering without misleading interpolation of constant scores."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


def plot_precision_recall(labels, scores_by_model, output):
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, scores in scores_by_model.items():
        scores = np.asarray(scores)
        ap = average_precision_score(labels, scores)
        if np.unique(scores).size == 1:
            # A constant scorer has one operating point. Connecting it to the
            # zero-recall endpoint would draw an unjustified diagonal.
            continue
        precision, recall, _ = precision_recall_curve(labels, scores)
        ax.step(recall[::-1], precision[::-1], where='pre', label=f'{name} (AP={ap:.3f})')
    ax.axhline(np.mean(labels), color='gray', linestyle='--',
               label=f'Constant-score reference (prevalence={np.mean(labels):.4%})')
    ax.set(xlabel='Recall', ylabel='Precision', title='Held-out precision–recall curve',
           xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
