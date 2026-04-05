"""Plotting helpers: confusion matrix and ROC curves."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.preprocessing import label_binarize

# Small constant for numerical stability in normalisation
_EPSILON: float = 1e-12


def save_confusion_matrix(
    cm: Sequence[Sequence[int]],
    classes: List[str],
    outpath: Path,
    title: str = "Confusion Matrix",
    normalize: bool = True,
) -> None:
    """Save a confusion-matrix heatmap to *outpath*.

    Parameters
    ----------
    cm:
        Integer confusion matrix (rows = true, cols = predicted).
    classes:
        Class name list.
    outpath:
        Destination file path (PNG).
    normalize:
        Whether to show row-normalised fractions alongside counts.
    """
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    cm_arr = np.array(cm, dtype=float)
    if normalize:
        row_sums = cm_arr.sum(axis=1, keepdims=True)
        cm_norm = np.where(row_sums > 0, cm_arr / np.maximum(row_sums, _EPSILON), 0.0)
    else:
        cm_norm = cm_arr

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)

    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)

    thresh = cm_norm.max() / 2.0
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            raw = int(cm_arr[i, j])
            frac = f"{cm_norm[i, j]:.2f}" if normalize else ""
            label = f"{raw}\n({frac})" if normalize else str(raw)
            ax.text(
                j, i, label,
                ha="center", va="center",
                color="white" if cm_norm[i, j] > thresh else "black",
                fontsize=8,
            )

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def save_roc_curves(
    y_true: Sequence[int],
    y_score: np.ndarray,
    classes: List[str],
    outpath: Path,
    title: str = "ROC Curves (One-vs-Rest)",
) -> None:
    """Save one-vs-rest ROC curves for each class to *outpath*.

    Parameters
    ----------
    y_true:
        Ground-truth class indices.
    y_score:
        Softmax probability matrix of shape (N, C).
    classes:
        Class name list.
    outpath:
        Destination file path (PNG).
    """
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    y_true = np.asarray(y_true)
    num_classes = len(classes)
    y_bin = label_binarize(y_true, classes=list(range(num_classes)))

    fig, ax = plt.subplots(figsize=(8, 6))

    for i, cls in enumerate(classes):
        try:
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
            from sklearn.metrics import auc
            auc_val = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=1.5, label=f"{cls} (AUC={auc_val:.3f})")
        except ValueError:
            continue

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def save_training_curves(
    history: dict,
    outpath: Path,
    title: str = "Training Curves",
) -> None:
    """Save training/validation loss and accuracy curves.

    Parameters
    ----------
    history:
        Dict with keys ``train_loss``, ``val_loss``, ``train_acc``, ``val_acc``
        each mapping to a list of per-epoch values.
    outpath:
        Destination file path (PNG).
    """
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    epochs = list(range(1, len(history.get("train_loss", [])) + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    if "train_loss" in history:
        ax1.plot(epochs, history["train_loss"], label="Train")
    if "val_loss" in history:
        ax1.plot(epochs, history["val_loss"], label="Val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss")
    ax1.legend()

    if "train_acc" in history:
        ax2.plot(epochs, history["train_acc"], label="Train")
    if "val_acc" in history:
        ax2.plot(epochs, history["val_acc"], label="Val")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy")
    ax2.legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
