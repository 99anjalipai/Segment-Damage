from __future__ import annotations

import torch
import torch.nn as nn


class FeatureProjectionNetwork(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int | None = None,
        num_layers: int = 2,
        dropout: float = 0.0,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        hidden = int(hidden_channels) if hidden_channels is not None else int(in_channels)
        channels = [int(in_channels)]
        if num_layers > 1:
            channels.extend([hidden] * (num_layers - 1))
        channels.append(int(out_channels))

        blocks = []
        for i in range(len(channels) - 1):
            in_c = channels[i]
            out_c = channels[i + 1]
            blocks.append(nn.Conv2d(in_c, out_c, kernel_size=1, bias=False))
            if i < len(channels) - 2:
                blocks.append(nn.BatchNorm2d(out_c))
                blocks.append(nn.ReLU(inplace=True))
                if dropout > 0:
                    blocks.append(nn.Dropout2d(p=float(dropout)))

        self.projector = nn.Sequential(*blocks)
        self.use_residual = bool(use_residual) and int(in_channels) == int(out_channels)
        self.out_channels = int(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.projector(x)
        if self.use_residual:
            projected = projected + x
        return projected
