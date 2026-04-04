"""Stratified train/val/test split utilities."""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def collect_images(dataset_root: Path, classes: Optional[List[str]] = None) -> List[Dict[str, str]]:
    """Enumerate all images under *dataset_root* grouped by class folder.

    Parameters
    ----------
    dataset_root:
        Root directory containing one sub-folder per class.
    classes:
        If provided, only enumerate the listed class names (in that order).
        Unknown folders are silently skipped.

    Returns
    -------
    List of ``{"path": str, "class": str}`` dicts.
    """
    dataset_root = Path(dataset_root)
    if classes:
        dirs = [dataset_root / c for c in classes if (dataset_root / c).is_dir()]
    else:
        dirs = sorted([p for p in dataset_root.iterdir() if p.is_dir()])

    rows: List[Dict[str, str]] = []
    for cls_dir in dirs:
        cls_name = cls_dir.name
        imgs = sorted([
            p for p in cls_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ])
        for p in imgs:
            rows.append({"path": str(p), "class": cls_name})
    return rows


def stratified_split(
    rows: List[Dict[str, str]],
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Perform deterministic stratified split by class.

    Returns
    -------
    Three lists of ``(path, class)`` tuples: train, val, test.
    """
    rng = random.Random(seed)
    by_class: Dict[str, List[str]] = {}
    for r in rows:
        by_class.setdefault(r["class"], []).append(r["path"])

    train, val, test = [], [], []
    for cls, items in sorted(by_class.items()):
        items_copy = list(items)
        rng.shuffle(items_copy)
        n = len(items_copy)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        train.extend((p, cls) for p in items_copy[:n_train])
        val.extend((p, cls) for p in items_copy[n_train:n_train + n_val])
        test.extend((p, cls) for p in items_copy[n_train + n_val:])

    return train, val, test


def write_manifest(rows: List[Tuple[str, str]], outpath: Path) -> None:
    """Write a list of (path, class) tuples to a CSV manifest file."""
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with outpath.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "class"])
        for path, cls in rows:
            writer.writerow([path, cls])


def read_manifest(path: Path) -> List[Dict[str, str]]:
    """Read a CSV manifest written by :func:`write_manifest`.

    Returns
    -------
    List of ``{"path": str, "class": str}`` dicts.
    """
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({"path": row["path"], "class": row["class"]})
    return rows
