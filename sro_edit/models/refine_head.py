from __future__ import annotations

# Adapted and narrowed to the paper's 3D residual refiner from
# code/svis/interactive_heads/models/refine_head.py. Changes: removed unused
# 2D and gated-residual options. The upstream project is Apache-2.0.

from dataclasses import dataclass

import torch
import torch.nn as nn


class _ResidualBlock3D(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv3d(channels, channels, kernel_size, padding=padding, bias=False)
        self.norm1 = nn.InstanceNorm3d(channels, affine=True)
        self.activation = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size, padding=padding, bias=False)
        self.norm2 = nn.InstanceNorm3d(channels, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.activation(self.norm1(self.conv1(x)))
        residual = self.norm2(self.conv2(residual))
        return self.activation(x + residual)


@dataclass
class RefineHeadConfig:
    in_channels: int
    out_channels: int = 1
    width: int = 32
    num_blocks: int = 4


class RefineHead(nn.Module):
    """Training-only 3D residual-logit refiner from the paper's B1/B2 stages."""

    def __init__(self, cfg: RefineHeadConfig):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(int(cfg.in_channels), int(cfg.width), kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(int(cfg.width), affine=True),
            nn.LeakyReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            *[_ResidualBlock3D(int(cfg.width)) for _ in range(int(cfg.num_blocks))]
        )
        self.delta_head = nn.Conv3d(int(cfg.width), int(cfg.out_channels), kernel_size=1)

    def predict_delta(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError("expected x=(B,C,D,H,W)")
        return self.delta_head(self.blocks(self.stem(x)))

    def forward(self, x: torch.Tensor, base_logits: torch.Tensor) -> torch.Tensor:
        delta = self.predict_delta(x)
        if delta.shape != base_logits.shape:
            raise ValueError(f"delta/base shape mismatch: {delta.shape} vs {base_logits.shape}")
        return base_logits + delta
