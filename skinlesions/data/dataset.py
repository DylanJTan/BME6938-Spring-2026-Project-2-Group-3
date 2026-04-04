"""PyTorch Dataset backed by a CSV split manifest."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset

from skinlesions.data.splits import read_manifest


class SkinLesionDataset(Dataset):
    """Image dataset that reads from a CSV manifest (path, class columns).

    Parameters
    ----------
    manifest_path:
        Path to a CSV file with ``path`` and ``class`` columns.
    class_to_idx:
        Mapping from class name to integer index.  If ``None``, classes are
        derived (alphabetically) from the manifest itself.
    transform:
        Optional torchvision transform applied to each PIL image.
    """

    def __init__(
        self,
        manifest_path: Path,
        class_to_idx: Optional[Dict[str, int]] = None,
        transform: Optional[Callable] = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.transform = transform

        rows = read_manifest(self.manifest_path)
        if not rows:
            raise ValueError(f"Empty manifest: {manifest_path}")

        if class_to_idx is None:
            classes = sorted({r["class"] for r in rows})
            class_to_idx = {c: i for i, c in enumerate(classes)}

        self.class_to_idx: Dict[str, int] = class_to_idx
        self.idx_to_class: Dict[int, str] = {v: k for k, v in class_to_idx.items()}
        self.classes: List[str] = list(class_to_idx.keys())

        self.samples: List[Tuple[str, int]] = []
        for r in rows:
            cls = r["class"]
            if cls not in class_to_idx:
                raise ValueError(
                    f"Class {cls!r} in manifest not found in class_to_idx."
                )
            self.samples.append((r["path"], class_to_idx[cls]))

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple:
        img_path, label = self.samples[index]
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label
