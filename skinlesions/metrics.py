"""Evaluation metrics for multi-class classification."""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def compute_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_score: np.ndarray,
    classes: List[str],
    average_types: Sequence[str] = ("macro", "weighted"),
) -> Dict:
    """Compute a comprehensive set of classification metrics.

    Parameters
    ----------
    y_true:
        Ground-truth class indices.
    y_pred:
        Predicted class indices.
    y_score:
        Softmax probability matrix of shape (N, C).
    classes:
        Class name list (must be in the same order as indices).
    average_types:
        Which averaging strategies to include (``macro``, ``weighted``).

    Returns
    -------
    Nested dict containing accuracy, per-average precision/recall/F1,
    per-class metrics, confusion matrix, and one-vs-rest ROC-AUC scores.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    num_classes = len(classes)
    y_bin = label_binarize(y_true, classes=list(range(num_classes)))

    metrics: Dict = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, target_names=classes, output_dict=True
        ),
    }

    for avg in average_types:
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average=avg, zero_division=0)
        metrics[f"precision_{avg}"] = float(p)
        metrics[f"recall_{avg}"] = float(r)
        metrics[f"f1_{avg}"] = float(f)

    # One-vs-rest ROC-AUC per class and macro average
    try:
        roc_auc_per_class = {}
        for i, cls in enumerate(classes):
            roc_auc_per_class[cls] = float(
                roc_auc_score(y_bin[:, i], y_score[:, i])
            )
        metrics["roc_auc_ovr"] = roc_auc_per_class
        metrics["roc_auc_macro"] = float(
            roc_auc_score(y_bin, y_score, multi_class="ovr", average="macro")
        )
    except ValueError:
        # Can occur with very small/unbalanced test sets
        metrics["roc_auc_ovr"] = {}
        metrics["roc_auc_macro"] = None

    return metrics
