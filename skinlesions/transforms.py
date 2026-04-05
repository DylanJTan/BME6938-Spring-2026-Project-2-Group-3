"""Image transform pipelines for train / val / test.

This module provides torchvision-based transforms with sensible defaults
and a small helper to build transforms from a config dictionary.
"""

from __future__ import annotations

from typing import Tuple

from torchvision import transforms

# Default normalization (ImageNet) — replace with dataset-specific values if known
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


def get_train_transforms(
    image_size: int = 224,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD,
    horizontal_flip_prob: float = 0.5,
    vertical_flip_prob: float = 0.2,
    rotation_degrees: int = 20,
    brightness: float = 0.2,
    contrast: float = 0.2,
    saturation: float = 0.2,
    hue: float = 0.02,
) -> transforms.Compose:
    """Return a torchvision Compose for training augmentations.

    Augmentations: random resized crop, flips, small rotation, color jitter,
    then normalize to tensors.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=horizontal_flip_prob),
        transforms.RandomVerticalFlip(p=vertical_flip_prob),
        transforms.RandomRotation(degrees=rotation_degrees),
        transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def get_val_transforms(
    image_size: int = 224,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD,
) -> transforms.Compose:
    """Return transforms for validation (deterministic).

    Resize then center-crop to preserve aspect ratio before normalization.
    """
    resize_size = int(image_size * 256 / 224)
    return transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def get_test_transforms(
    image_size: int = 224,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD,
) -> transforms.Compose:
    """Test transforms (same as validation by default)."""
    return get_val_transforms(image_size=image_size, mean=mean, std=std)


def from_config(cfg: dict, split: str = "train") -> transforms.Compose:
    """Build transforms from a config dictionary.

    Reads from ``cfg['data']`` for image_size/img_size, mean, std, and
    augmentation sub-keys.
    """
    data_cfg = cfg.get("data", {})
    image_size = int(data_cfg.get("img_size", data_cfg.get("image_size", 224)))
    mean = tuple(data_cfg.get("mean", IMAGENET_MEAN))
    std = tuple(data_cfg.get("std", IMAGENET_STD))

    if split == "train":
        aug = data_cfg.get("augmentation", {})
        cj = aug.get("color_jitter", {})
        return get_train_transforms(
            image_size=image_size,
            mean=mean,
            std=std,
            horizontal_flip_prob=float(aug.get("horizontal_flip_prob", 0.5)),
            vertical_flip_prob=float(aug.get("vertical_flip_prob", 0.2)),
            rotation_degrees=int(aug.get("rotation_degrees", 20)),
            brightness=float(cj.get("brightness", 0.2)),
            contrast=float(cj.get("contrast", 0.2)),
            saturation=float(cj.get("saturation", 0.2)),
            hue=float(cj.get("hue", 0.02)),
        )
    elif split in ("val", "test"):
        return get_val_transforms(image_size=image_size, mean=mean, std=std)
    else:
        raise ValueError(f"Unknown split: {split!r}")
