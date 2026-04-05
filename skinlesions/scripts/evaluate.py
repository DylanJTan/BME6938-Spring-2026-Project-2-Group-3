"""CLI: evaluate a trained checkpoint on the test split.

Usage
-----
    python -m skinlesions.scripts.evaluate \\
        --config configs/config.yaml \\
        --checkpoint results/models/best_cnn_baseline_baseline_run.pt

    # Compare two checkpoints and produce a summary table
    python -m skinlesions.scripts.evaluate \\
        --config configs/config.yaml \\
        --checkpoint results/models/best_cnn_baseline_baseline_run.pt \\
                     results/models/best_resnet18_pretrained_resnet18_run.pt \\
        --summary-out results/logs/comparison.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import torch
import yaml
from tqdm import tqdm

from skinlesions.data.loader import build_dataloaders
from skinlesions.models import build_model
from skinlesions.train_utils import load_checkpoint
from skinlesions.metrics import compute_metrics
from skinlesions.plots import save_confusion_matrix, save_roc_curves


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_path(p, base=None):
    p = Path(p)
    if not p.is_absolute() and base is not None:
        p = base / p
    return p.resolve()


def _resolve_project_root(cfg_path: Path) -> Path:
    """Infer repository root from config file location."""
    if (cfg_path.parent / "setup.py").exists():
        return cfg_path.parent.resolve()
    if (cfg_path.parent.parent / "setup.py").exists():
        return cfg_path.parent.parent.resolve()
    return cfg_path.parent.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate skin lesion classifiers")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument(
        "--checkpoint", type=Path, nargs="+", required=True,
        help="One or more checkpoint paths to evaluate"
    )
    parser.add_argument(
        "--summary-out", type=Path, default=None,
        help="Path to write comparison JSON/CSV summary (default: results/logs/comparison.json)"
    )
    parser.add_argument(
        "--split", type=str, default="test",
        choices=("train", "val", "test"),
        help="Which split to evaluate on (default: test)"
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Single-checkpoint evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_checkpoint(
    ckpt_path: Path,
    cfg: dict,
    cfg_base: Path,
    split: str = "test",
) -> dict:
    """Run inference with a single checkpoint and return metrics dict."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint metadata
    ckpt = torch.load(ckpt_path, map_location=device)
    model_name: str = ckpt.get("model_name", cfg.get("training", {}).get("model_name", "cnn_baseline"))
    classes: List[str] = ckpt.get("classes", cfg.get("data", {}).get("classes", []))

    if not classes:
        raise ValueError("No classes found in checkpoint or config.")

    num_classes = len(classes)
    model = build_model(model_name, num_classes=num_classes)
    load_checkpoint(model, ckpt_path, device)
    model.to(device)
    model.eval()

    paths_cfg = cfg.get("paths", {})
    split_dir = _resolve_path(paths_cfg.get("split_manifest_dir", "results/logs/splits"), cfg_base)

    class_to_idx = {c: i for i, c in enumerate(classes)}
    from skinlesions.data.dataset import SkinLesionDataset
    from skinlesions import transforms as T
    from torch.utils.data import DataLoader

    data_cfg = cfg.get("data", {})
    manifest = split_dir / f"{split}.csv"
    dataset = SkinLesionDataset(
        manifest_path=manifest,
        class_to_idx=class_to_idx,
        transform=T.from_config(cfg, split=split),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.get("training", {}).get("batch_size", 32)),
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 4)),
        pin_memory=bool(data_cfg.get("pin_memory", True)),
    )

    all_labels: List[int] = []
    all_preds: List[int] = []
    all_scores: List[np.ndarray] = []

    for images, labels in tqdm(loader, desc=f"Evaluating {model_name}", leave=False):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().tolist()
        all_labels.extend(labels.tolist())
        all_preds.extend(preds)
        all_scores.append(probs)

    y_score = np.concatenate(all_scores, axis=0)
    avg_types = cfg.get("evaluation", {}).get("average_types", ["macro", "weighted"])

    metrics = compute_metrics(
        y_true=all_labels,
        y_pred=all_preds,
        y_score=y_score,
        classes=classes,
        average_types=avg_types,
    )
    metrics["model_name"] = model_name
    metrics["checkpoint"] = str(ckpt_path)
    metrics["split"] = split

    return metrics, all_labels, all_preds, y_score, classes


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

    repo_root = _resolve_project_root(cfg_path)
    paths_cfg = cfg.get("paths", {})
    figure_dir = _resolve_path(paths_cfg.get("figure_dir", "results/figures"), repo_root)
    log_dir = _resolve_path(paths_cfg.get("log_dir", "results/logs"), repo_root)
    figure_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    eval_cfg = cfg.get("evaluation", {})

    all_results = []

    for ckpt_path in args.checkpoint:
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            print(f"[evaluate] Checkpoint not found, skipping: {ckpt_path}")
            continue

        print(f"\n[evaluate] Checkpoint: {ckpt_path.name}")
        metrics, y_true, y_pred, y_score, classes = evaluate_checkpoint(
            ckpt_path=ckpt_path,
            cfg=cfg,
            cfg_base=repo_root,
            split=args.split,
        )

        model_name = metrics["model_name"]
        tag = ckpt_path.stem

        print(f"  Accuracy : {metrics['accuracy']:.4f}")
        for avg in cfg.get("evaluation", {}).get("average_types", ["macro", "weighted"]):
            print(f"  F1 ({avg:8s}): {metrics.get(f'f1_{avg}', 'N/A'):.4f}")

        # Confusion matrix
        if eval_cfg.get("save_confusion_matrix", True):
            cm_path = figure_dir / f"confusion_matrix_{tag}.png"
            save_confusion_matrix(
                cm=metrics["confusion_matrix"],
                classes=classes,
                outpath=cm_path,
                title=f"Confusion Matrix – {model_name}",
            )
            print(f"  Confusion matrix → {cm_path}")

        # ROC curves
        if eval_cfg.get("save_roc_curve", True):
            roc_path = figure_dir / f"roc_ovr_{tag}.png"
            save_roc_curves(
                y_true=y_true,
                y_score=y_score,
                classes=classes,
                outpath=roc_path,
                title=f"ROC Curves (OvR) – {model_name}",
            )
            print(f"  ROC curves       → {roc_path}")

        # Per-checkpoint JSON
        metrics_path = log_dir / f"metrics_{tag}.json"
        # confusion_matrix is already a list; safe to serialise
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"  Metrics JSON     → {metrics_path}")

        all_results.append(metrics)

    # --- Comparison summary --------------------------------------------------
    if len(all_results) > 1 or args.summary_out:
        summary_out = args.summary_out or (log_dir / "comparison.json")
        summary_out = Path(summary_out)
        summary_out.parent.mkdir(parents=True, exist_ok=True)

        with summary_out.open("w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n[evaluate] Comparison JSON → {summary_out}")

        # Markdown table
        md_path = summary_out.with_suffix(".md")
        avg_types = cfg.get("evaluation", {}).get("average_types", ["macro", "weighted"])
        header_cols = ["model", "accuracy"] + [f"f1_{a}" for a in avg_types] + ["roc_auc_macro"]
        rows_md = []
        for r in all_results:
            row = [
                r.get("model_name", ""),
                f"{r.get('accuracy', 0):.4f}",
            ]
            for a in avg_types:
                row.append(f"{r.get(f'f1_{a}', 0):.4f}")
            row.append(f"{r.get('roc_auc_macro') or 0:.4f}")
            rows_md.append(row)

        with md_path.open("w", encoding="utf-8") as f:
            f.write("# Model Comparison\n\n")
            f.write("| " + " | ".join(header_cols) + " |\n")
            f.write("| " + " | ".join(["---"] * len(header_cols)) + " |\n")
            for row in rows_md:
                f.write("| " + " | ".join(row) + " |\n")
        print(f"[evaluate] Comparison table → {md_path}")


if __name__ == "__main__":
    main()
