"""Configurable CNN binary classifier for tornado input fields.

Input batches are accepted as [batch, nx, ny, nc] (channels last) or
[batch, nc, nx, ny] (channels first). The model returns unnormalized logits.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 pool_size: int = 2, dropout: float = 0.0):
        super().__init__()
        padding = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        ]
        if pool_size and pool_size > 1:
            layers.append(nn.MaxPool2d(pool_size, ceil_mode=True))
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TornadoCNN(nn.Module):
    """A stack of configurable 2D convolutional blocks and a dense head."""

    def __init__(self, input_channels: int, channels: list[int] | tuple[int, ...],
                 kernel_size: int = 3, pool_size: int = 2,
                 dropout: float = 0.0, fc_nodes: list[int] | tuple[int, ...] = (128,)):
        super().__init__()
        if input_channels < 1 or not channels or any(c < 1 for c in channels):
            raise ValueError("input_channels and every convolution channel count must be positive")
        blocks = []
        current = input_channels
        for count in channels:
            blocks.append(ConvBlock(current, int(count), kernel_size, pool_size, dropout))
            current = int(count)
        self.features = nn.Sequential(*blocks)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        head: list[nn.Module] = []
        current = int(channels[-1])
        for width in fc_nodes:
            if int(width) < 1:
                raise ValueError("fully connected layer widths must be positive")
            head.extend([nn.Linear(current, int(width)), nn.GELU()])
            if dropout > 0:
                head.append(nn.Dropout(dropout))
            current = int(width)
        head.append(nn.Linear(current, 1))
        self.classifier = nn.Sequential(*head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected a 4D input [B,nx,ny,nc] or [B,nc,nx,ny], got {tuple(x.shape)}")
        # YAML's input_channels makes the expected channel axis unambiguous.
        if x.shape[-1] == self.features[0].layers[0].in_channels:
            x = x.permute(0, 3, 1, 2).contiguous()
        elif x.shape[1] == self.features[0].layers[0].in_channels:
            pass  # already channels first
        else:
            raise ValueError(f"Neither input channel axis matches configured input_channels; got {tuple(x.shape)}")
        return self.classifier(self.global_pool(self.features(x)).flatten(1)).squeeze(-1)


MODEL_REGISTRY = {"TornadoCNN": TornadoCNN}


def build_model(config: Mapping[str, Any]) -> nn.Module:
    """Build the configured model. Add replacement architectures to MODEL_REGISTRY."""
    name = config.get("name", "TornadoCNN")
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model {name!r}; registered models: {sorted(MODEL_REGISTRY)}")
    params = dict(config.get("params", {}))
    if "input_channels" not in params:
        params["input_channels"] = config.get("input_channels")
    return MODEL_REGISTRY[name](**params)
