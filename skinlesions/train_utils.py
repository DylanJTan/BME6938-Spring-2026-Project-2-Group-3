"""Training utilities: seed, loops, AMP, early stopping, checkpointing."""

from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: int = 42) -> None:
    """Set all relevant random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Stop training when a monitored metric stops improving.

    Parameters
    ----------
    patience:
        Number of epochs with no improvement before stopping.
    mode:
        ``'min'`` for loss-style metrics, ``'max'`` for accuracy/F1.
    min_delta:
        Minimum change to qualify as improvement.
    """

    def __init__(self, patience: int = 10, mode: str = "min", min_delta: float = 0.0) -> None:
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best: Optional[float] = None
        self.counter: int = 0
        self.should_stop: bool = False

    def __call__(self, metric: float) -> bool:
        """Return ``True`` if training should be stopped."""
        if self.best is None:
            self.best = metric
            return False

        if self.mode == "min":
            improved = metric < self.best - self.min_delta
        else:
            improved = metric > self.best + self.min_delta

        if improved:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


# ---------------------------------------------------------------------------
# Training / validation loops
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
) -> Dict[str, float]:
    """Run one training epoch.

    Returns
    -------
    dict with keys ``loss`` and ``acc`` (top-1 accuracy).
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        if scaler is not None:
            with autocast(device_type="cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += preds.eq(labels).sum().item()
        total += images.size(0)

    return {"loss": running_loss / total, "acc": correct / total}


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Run one validation epoch (no gradient computation).

    Returns
    -------
    dict with keys ``loss`` and ``acc``.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += preds.eq(labels).sum().item()
        total += images.size(0)

    return {"loss": running_loss / total, "acc": correct / total}


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    metrics: dict,
    path: Path,
    extra: Optional[dict] = None,
) -> None:
    """Save a training checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(
    model: nn.Module,
    path: Path,
    device: torch.device,
    optimizer: Optional[Optimizer] = None,
) -> Tuple[int, dict]:
    """Load a checkpoint into *model* (and optionally *optimizer*).

    Returns
    -------
    Tuple of (epoch, metrics_dict).
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint.get("epoch", 0), checkpoint.get("metrics", {})
