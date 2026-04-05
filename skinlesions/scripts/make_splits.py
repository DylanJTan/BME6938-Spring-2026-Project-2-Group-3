"""CLI: create train/val/test split manifests from the dataset root.

Usage
-----
    python -m skinlesions.scripts.make_splits --config configs/config.yaml
    python -m skinlesions.scripts.make_splits --config configs/config.yaml \\
        --outdir data/splits --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from skinlesions.data.splits import collect_images, stratified_split, write_manifest


def _resolve_project_root(cfg_path: Path) -> Path:
    """Infer repository root from config file location."""
    if (cfg_path.parent / "setup.py").exists():
        return cfg_path.parent.resolve()
    if (cfg_path.parent.parent / "setup.py").exists():
        return cfg_path.parent.parent.resolve()
    return cfg_path.parent.resolve()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create train/val/test split manifests")
    p.add_argument("--config", type=Path, default=Path("configs/config.yaml"),
                   help="Path to YAML config file")
    p.add_argument("--outdir", type=Path, default=None,
                   help="Override output directory for manifests")
    p.add_argument("--train-frac", type=float, default=None)
    p.add_argument("--val-frac", type=float, default=None)
    p.add_argument("--test-frac", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        # Try relative to CWD, then search upward
        if not cfg_path.exists():
            for parent in Path.cwd().parents:
                candidate = parent / cfg_path
                if candidate.exists():
                    cfg_path = candidate
                    break

    if not cfg_path.exists():
        raise SystemExit(f"Config not found: {args.config}")

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    repo_root = _resolve_project_root(cfg_path)

    # Resolve paths from config (CLI args take precedence)
    dataset_root = Path(cfg["paths"]["dataset_root"])
    if not dataset_root.is_absolute():
        dataset_root = (repo_root / dataset_root).resolve()

    outdir = args.outdir
    if outdir is None:
        outdir = Path(cfg["paths"].get("split_manifest_dir", "results/logs/splits"))
    if not outdir.is_absolute():
        outdir = (repo_root / outdir).resolve()

    split_cfg = cfg.get("data", {}).get("split", {})
    train_frac = args.train_frac if args.train_frac is not None else float(split_cfg.get("train", 0.70))
    val_frac = args.val_frac if args.val_frac is not None else float(split_cfg.get("val", 0.15))
    test_frac = args.test_frac if args.test_frac is not None else float(split_cfg.get("test", 0.15))
    seed = args.seed if args.seed is not None else int(cfg.get("project", {}).get("seed", 42))

    classes = cfg.get("data", {}).get("classes") or None

    if not dataset_root.exists():
        raise SystemExit(f"dataset_root does not exist: {dataset_root}")

    print(f"Dataset root : {dataset_root}")
    print(f"Output dir   : {outdir}")
    print(f"Fractions    : train={train_frac}, val={val_frac}, test={test_frac}")
    print(f"Seed         : {seed}")

    rows = collect_images(dataset_root, classes=classes)
    print(f"Total images : {len(rows)}")

    # Print per-class counts
    from collections import Counter
    counts = Counter(r["class"] for r in rows)
    for cls, n in sorted(counts.items()):
        print(f"  {cls}: {n}")

    train, val, test = stratified_split(
        rows,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        seed=seed,
    )

    write_manifest(train, outdir / "train.csv")
    write_manifest(val,   outdir / "val.csv")
    write_manifest(test,  outdir / "test.csv")

    print(f"\nManifests written to {outdir}/")
    print(f"  train.csv : {len(train)} samples")
    print(f"  val.csv   : {len(val)} samples")
    print(f"  test.csv  : {len(test)} samples")


if __name__ == "__main__":
    main()
