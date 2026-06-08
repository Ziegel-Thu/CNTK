"""Small PyTorch models for feature-dynamics experiments."""

from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class MLP(nn.Module):
    """Small ReLU MLP exposing its learned feature representation."""

    def __init__(self, input_dim: int, width: int = 128, depth: int = 2) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1")

        layers: list[nn.Module] = []
        last_dim = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(last_dim, width))
            layers.append(nn.ReLU())
            last_dim = width
        self.feature_net = nn.Sequential(*layers)
        self.head = nn.Linear(width, 1)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_net(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x)).squeeze(-1)

    def freeze_features(self) -> None:
        for param in self.feature_net.parameters():
            param.requires_grad_(False)


class SmallCNN(nn.Module):
    """Small CIFAR-style CNN exposing penultimate features."""

    def __init__(self, width: int = 32, feature_dim: int = 128) -> None:
        super().__init__()
        self.feature_net = nn.Sequential(
            nn.Conv2d(3, width, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(width, 2 * width, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(2 * width, feature_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.head = nn.Linear(feature_dim, 1)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_net(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x)).squeeze(-1)

    def freeze_features(self) -> None:
        for param in self.feature_net.parameters():
            param.requires_grad_(False)


def binary_accuracy(logits: torch.Tensor, targets01: torch.Tensor) -> float:
    preds = (torch.sigmoid(logits) >= 0.5).to(targets01.dtype)
    return float((preds == targets01).float().mean().item())
