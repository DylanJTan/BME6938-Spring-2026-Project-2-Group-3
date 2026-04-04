"""Model definitions for skin lesion classification.

Provides:
  - ``CNNBaseline``:  a modest CNN trained from scratch
  - ``build_model``:  factory function keyed by model_name string

Supported model_name values
---------------------------
``cnn_baseline``
    Custom CNN trained from scratch.
``resnet18_pretrained``
    Torchvision ResNet-18 with ImageNet weights; final FC replaced.
``resnet50_pretrained``
    Torchvision ResNet-50 with ImageNet weights; final FC replaced.
``efficientnet_b0``
    Torchvision EfficientNet-B0 with ImageNet weights; classifier replaced.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    EfficientNet_B0_Weights,
    ResNet18_Weights,
    ResNet50_Weights,
)


class CNNBaseline(nn.Module):
    """A moderately deep CNN designed for 224×224 RGB images.

    Architecture (each block: Conv→BN→ReLU→MaxPool):
        Block 1 : 3  → 32  channels
        Block 2 : 32 → 64  channels
        Block 3 : 64 → 128 channels
        Block 4 : 128 → 256 channels
    Followed by global average pooling and a two-layer MLP head.
    """

    def __init__(self, num_classes: int = 5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 112
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 56
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 28
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 14
            # Block 5
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 7
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def build_model(model_name: str, num_classes: int = 5) -> nn.Module:
    """Return a model instance for the given *model_name*.

    Parameters
    ----------
    model_name:
        One of ``cnn_baseline``, ``resnet18_pretrained``,
        ``resnet50_pretrained``, ``efficientnet_b0``.
    num_classes:
        Number of output classes.

    Raises
    ------
    ValueError
        If *model_name* is not recognised.
    """
    name = model_name.lower()

    if name == "cnn_baseline":
        return CNNBaseline(num_classes=num_classes)

    if name == "resnet18_pretrained":
        model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if name == "resnet50_pretrained":
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, num_classes),
        )
        return model

    raise ValueError(
        f"Unknown model_name {model_name!r}. "
        "Choose from: cnn_baseline, resnet18_pretrained, resnet50_pretrained, efficientnet_b0."
    )
