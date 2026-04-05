"""CLI: train a model on the skin lesion dataset.

Usage
-----
    python -m skinlesions.scripts.train \\
        --config configs/config.yaml \\
        --model cnn_baseline

    python -m skinlesions.scripts.train \\
        --config configs/config.yaml \\
        --model resnet18_pretrained \\
        --experiment-name resnet18_run
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import yaml
from torch.cuda.amp import GradScaler

from skinlesions.data.loader import build_dataloaders
from skinlesions.models import build_model
from skinlesions.train_utils import (
    EarlyStopping,
    load_checkpoint,
    save_checkpoint,
    seed_everything,
    train_one_epoch,
    validate_one_epoch,
)
from skinlesions.plots import save_training_curves


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_path(p: str | Path, base: Optional[Path] = None) -> Path:
    """Make *p* absolute, optionally relative to *base*."""
    p = Path(p)
    if not p.is_absolute() and base is not None:
        p = base / p
    return p.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a skin lesion classifier")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--model", type=str, default=None,
                        help="Model name (overrides config training.model_name)")
    parser.add_argument("--experiment-name", type=str, default=None,
                        help="Experiment name (overrides config project.experiment_name)")
    parser.add_argument("--resume", type=Path, default=None,
                        help="Path to a checkpoint to resume training from")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise SystemExit(f"Config not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    repo_root = cfg_path.parent.resolve()

    # --- experiment identity -----------------------------------------------
    proj_cfg = cfg.get("project", {})
    exp_name: str = args.experiment_name or proj_cfg.get("experiment_name", "run")
    model_name: str = args.model or cfg.get("training", {}).get("model_name", "cnn_baseline")
    seed: int = int(proj_cfg.get("seed", 42))

    run_name = f"{model_name}_{exp_name}"

    # --- paths ---------------------------------------------------------------
    paths_cfg = cfg.get("paths", {})
    split_dir = _resolve_path(paths_cfg.get("split_manifest_dir", "results/logs/splits"), repo_root)
    model_dir = _resolve_path(paths_cfg.get("model_dir", "results/models"), repo_root)
    log_dir = _resolve_path(paths_cfg.get("log_dir", "results/logs"), repo_root)
    figure_dir = _resolve_path(paths_cfg.get("figure_dir", "results/figures"), repo_root)

    for d in (model_dir, log_dir, figure_dir):
        d.mkdir(parents=True, exist_ok=True)

    # --- reproducibility -----------------------------------------------------
    seed_everything(seed)

    # --- device --------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device}  model={model_name}  experiment={exp_name}")

    # --- data ----------------------------------------------------------------
    classes = cfg.get("data", {}).get("classes", [])
    if not classes:
        raise SystemExit("No classes defined in config data.classes")

    train_cfg = cfg.get("training", {})
    loaders = build_dataloaders(split_dir, classes, cfg)

    # --- model ---------------------------------------------------------------
    num_classes = len(classes)
    model = build_model(model_name, num_classes=num_classes)
    model.to(device)

    # --- loss ----------------------------------------------------------------
    label_smoothing: float = float(train_cfg.get("label_smoothing", 0.0))
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # --- optimiser -----------------------------------------------------------
    lr: float = float(train_cfg.get("learning_rate", 1e-3))
    wd: float = float(train_cfg.get("weight_decay", 1e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(train_cfg.get("max_epochs", 50)),
    )

    # --- AMP -----------------------------------------------------------------
    use_amp: bool = bool(train_cfg.get("use_amp", True)) and device.type == "cuda"
    scaler: Optional[GradScaler] = GradScaler() if use_amp else None

    # --- early stopping & checkpointing -------------------------------------
    patience: int = int(train_cfg.get("early_stopping_patience", 10))
    early_stop = EarlyStopping(patience=patience, mode="min")

    best_ckpt = model_dir / f"best_{run_name}.pt"
    last_ckpt = model_dir / f"last_{run_name}.pt"

    start_epoch = 0
    if args.resume and args.resume.exists():
        start_epoch, _ = load_checkpoint(model, args.resume, device, optimizer)
        print(f"[train] Resumed from {args.resume} (epoch {start_epoch})")

    # --- log file ------------------------------------------------------------
    log_csv = log_dir / f"metrics_{run_name}.csv"
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    with log_csv.open("w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "elapsed_s"])

        max_epochs: int = int(train_cfg.get("max_epochs", 50))
        best_val_loss = float("inf")

        # Initialise loop variables to guard against an empty range
        epoch: int = start_epoch
        train_metrics: dict = {"loss": float("nan"), "acc": float("nan")}
        val_metrics: dict = {"loss": float("nan"), "acc": float("nan")}

        for epoch in range(start_epoch + 1, max_epochs + 1):
            t0 = time.time()
            train_metrics = train_one_epoch(
                model, loaders["train"], criterion, optimizer, device, scaler
            )
            val_metrics = validate_one_epoch(
                model, loaders["val"], criterion, device
            )
            scheduler.step()

            elapsed = time.time() - t0

            # Log
            writer.writerow([
                epoch,
                f"{train_metrics['loss']:.6f}",
                f"{train_metrics['acc']:.6f}",
                f"{val_metrics['loss']:.6f}",
                f"{val_metrics['acc']:.6f}",
                f"{elapsed:.1f}",
            ])
            fcsv.flush()

            history["train_loss"].append(train_metrics["loss"])
            history["val_loss"].append(val_metrics["loss"])
            history["train_acc"].append(train_metrics["acc"])
            history["val_acc"].append(val_metrics["acc"])

            print(
                f"Epoch {epoch:3d}/{max_epochs} | "
                f"train loss={train_metrics['loss']:.4f} acc={train_metrics['acc']:.4f} | "
                f"val loss={val_metrics['loss']:.4f} acc={val_metrics['acc']:.4f} | "
                f"{elapsed:.1f}s"
            )

            # Checkpoint best
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                save_checkpoint(
                    model, optimizer, epoch,
                    {**train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}},
                    best_ckpt,
                    extra={"model_name": model_name, "classes": classes, "run_name": run_name},
                )
                print(f"  → saved best checkpoint (val_loss={best_val_loss:.4f})")

            # Early stopping
            if early_stop(val_metrics["loss"]):
                print(f"[train] Early stopping at epoch {epoch}")
                break

        # Save last checkpoint
        save_checkpoint(
            model, optimizer, epoch,
            {**train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}},
            last_ckpt,
            extra={"model_name": model_name, "classes": classes, "run_name": run_name},
        )

    # --- training curves -----------------------------------------------------
    save_training_curves(
        history,
        outpath=figure_dir / f"training_curves_{run_name}.png",
        title=f"Training Curves – {run_name}",
    )

    print(f"\n[train] Done.  Best checkpoint: {best_ckpt}")
    print(f"[train] Metrics log: {log_csv}")


if __name__ == "__main__":
    main()
