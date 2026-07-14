"""Model definitions.

Three architectures are provided on purpose:

  simple_cnn      — from-scratch baseline. You need this. Without a baseline
                    you cannot claim transfer learning "helped"; you can only
                    claim it "worked".
  resnet18        — the workhorse. Fast, well understood, good Grad-CAM target.
  efficientnet_b0 — stronger, similar parameter count, slower to converge.

Being able to say "ResNet18 beat my from-scratch CNN by X points and here is
why" is worth more in an interview than a single high accuracy number.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

NUM_CLASSES = 2


class SimpleCNN(nn.Module):
    """Deliberately plain 4-block CNN. The control condition, not the product."""

    def __init__(self, dropout: float = 0.2) -> None:
        super().__init__()
        def block(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(3, 32), block(32, 64), block(64, 128), block(128, 256)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(256, NUM_CLASSES)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)))


def build_model(cfg: dict) -> nn.Module:
    arch = cfg["model"]["arch"]
    pretrained = cfg["model"]["pretrained"]
    dropout = cfg["model"]["dropout"]
    freeze = cfg["model"]["freeze_backbone"]

    if arch == "simple_cnn":
        return SimpleCNN(dropout)

    if arch == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        if freeze:
            for p in model.parameters():
                p.requires_grad = False
        model.fc = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(model.fc.in_features, NUM_CLASSES)
        )
        return model

    if arch == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        if freeze:
            for p in model.parameters():
                p.requires_grad = False
        in_f = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, NUM_CLASSES))
        return model

    raise ValueError(f"Unknown arch: {arch!r}")


def get_target_layer(model: nn.Module, arch: str) -> nn.Module:
    """The last conv layer — where Grad-CAM hooks in.

    Rule of thumb: the deepest layer that still has spatial extent. Go deeper
    and you lose all localisation; go shallower and the heatmap shows edges
    rather than semantics.
    """
    if arch == "resnet18":
        return model.layer4[-1]
    if arch == "efficientnet_b0":
        return model.features[-1]
    if arch == "simple_cnn":
        return model.features[-1]
    raise ValueError(f"Unknown arch: {arch!r}")
