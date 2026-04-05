"""Run the full reproducible pipeline: splits, training, and comparison eval.

This script is intended as the single default command for end-to-end runs.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


DEFAULT_MODEL_RUNS = {
    "cnn_baseline": "baseline_run",
    "resnet18_pretrained": "resnet18_run",
    "resnet50_pretrained": "resnet50_run",
    "efficientnet_b0": "effb0_run",
}


def _resolve_project_root(cfg_path: Path) -> Path:
    if (cfg_path.parent / "setup.py").exists():
        return cfg_path.parent.resolve()
    if (cfg_path.parent.parent / "setup.py").exists():
        return cfg_path.parent.parent.resolve()
    return cfg_path.parent.resolve()


def _resolve_path(p: str | Path, base: Path) -> Path:
    p = Path(p)
    if not p.is_absolute():
        p = base / p
    return p.resolve()


def _run(cmd: list[str], cwd: Path) -> None:
    print("[run_full]", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full skin lesion pipeline")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument(
        "--models",
        nargs="*",
        default=list(DEFAULT_MODEL_RUNS.keys()),
        choices=list(DEFAULT_MODEL_RUNS.keys()),
        help="Models to train/evaluate (default: all supported)",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path("results/logs/comparison.json"),
        help="Comparison output JSON path",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=("train", "val", "test"),
        help="Split for comparison evaluation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise SystemExit(f"Config not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    repo_root = _resolve_project_root(cfg_path)
    model_dir = _resolve_path(cfg.get("paths", {}).get("model_dir", "results/models"), repo_root)
    summary_out = _resolve_path(args.summary_out, repo_root)

    # 1) Deterministic split generation.
    _run(
        [
            sys.executable,
            "-m",
            "skinlesions.scripts.make_splits",
            "--config",
            str(cfg_path),
        ],
        cwd=repo_root,
    )

    # 2) Train all requested models.
    checkpoints: list[Path] = []
    for model in args.models:
        exp_name = DEFAULT_MODEL_RUNS[model]
        _run(
            [
                sys.executable,
                "-m",
                "skinlesions.scripts.train",
                "--config",
                str(cfg_path),
                "--model",
                model,
                "--experiment-name",
                exp_name,
            ],
            cwd=repo_root,
        )
        checkpoints.append(model_dir / f"best_{model}_{exp_name}.pt")

    # 3) Evaluate all best checkpoints and generate figures/tables.
    eval_cmd = [
        sys.executable,
        "-m",
        "skinlesions.scripts.evaluate",
        "--config",
        str(cfg_path),
        "--split",
        args.split,
        "--checkpoint",
        *[str(p) for p in checkpoints],
        "--summary-out",
        str(summary_out),
    ]
    _run(eval_cmd, cwd=repo_root)

    print("[run_full] Complete.")


if __name__ == "__main__":
    main()
