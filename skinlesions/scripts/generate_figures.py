"""Generate evaluation figures/logs from existing checkpoints (no training)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate figures from saved checkpoints")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument(
        "--checkpoint-pattern",
        type=str,
        default="best_*.pt",
        help="Glob pattern to discover checkpoints in model_dir",
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
        help="Split for evaluation",
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

    checkpoints = sorted(model_dir.glob(args.checkpoint_pattern))
    if not checkpoints:
        raise SystemExit(
            f"No checkpoints found in {model_dir} matching pattern {args.checkpoint_pattern!r}."
        )

    cmd = [
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

    print("[generate_figures]", " ".join(cmd))
    subprocess.run(cmd, cwd=repo_root, check=True)
    print("[generate_figures] Complete.")


if __name__ == "__main__":
    main()
