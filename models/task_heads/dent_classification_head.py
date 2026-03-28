from __future__ import annotations

import torch
import torch.nn as nn


class DentClassificationHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden_channels: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden = int(hidden_channels) if hidden_channels is not None else max(int(in_channels) // 2, 32)
        drop = float(dropout)

        layers: list[nn.Module] = [
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(int(in_channels), hidden),
            nn.ReLU(inplace=True),
        ]
        if drop > 0.0:
            layers.append(nn.Dropout(p=drop))
        layers.append(nn.Linear(hidden, int(num_classes)))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)
