"""DataLoader builder from config + split manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from torch.utils.data import DataLoader

from skinlesions.data.dataset import SkinLesionDataset
from skinlesions import transforms as T


def build_dataloaders(
    manifest_dir: Path,
    classes: List[str],
    cfg: dict,
    splits: Tuple[str, ...] = ("train", "val", "test"),
) -> Dict[str, DataLoader]:
    """Build DataLoaders for each requested split.

    Parameters
    ----------
    manifest_dir:
        Directory containing ``train.csv``, ``val.csv``, ``test.csv``.
    classes:
        Ordered list of class names (defines class-to-index mapping).
    cfg:
        Top-level config dict (reads ``data.*`` and ``training.batch_size``).
    splits:
        Which splits to load.

    Returns
    -------
    Dict mapping split name to its DataLoader.
    """
    manifest_dir = Path(manifest_dir)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})
    batch_size: int = int(train_cfg.get("batch_size", 32))
    num_workers: int = int(data_cfg.get("num_workers", 4))
    pin_memory: bool = bool(data_cfg.get("pin_memory", True))

    loaders: Dict[str, DataLoader] = {}
    for split in splits:
        manifest = manifest_dir / f"{split}.csv"
        tfm = T.from_config(cfg, split=split)
        dataset = SkinLesionDataset(
            manifest_path=manifest,
            class_to_idx=class_to_idx,
            transform=tfm,
        )
        shuffle = split == "train"
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
    return loaders
