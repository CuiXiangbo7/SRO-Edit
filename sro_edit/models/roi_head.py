from __future__ import annotations

# Adapted and narrowed to the paper's 3D path from
# code/svis/interactive_heads/models/roi_head.py. Changes: fixed 3D operations
# and the paper's single-score 32-64-1 head. The upstream project is Apache-2.0.

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn


class _ConvBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


@dataclass
class ROIHeadConfig:
    in_channels: int
    feat_channels: int = 32
    num_blocks: int = 3
    mlp_hidden: int = 64


class SupervoxelROIHead(nn.Module):
    """3D click-conditioned ROI scorer with mean-pooled supervoxel features.

    ``x`` is [image, accumulated positive clicks, accumulated negative clicks]
    with shape (B,C,D,H,W). ``supervoxels`` has shape (B,D,H,W).
    """

    def __init__(self, cfg: ROIHeadConfig):
        super().__init__()
        self.cfg = cfg
        blocks = []
        channels = int(cfg.in_channels)
        for _ in range(int(cfg.num_blocks)):
            blocks.append(_ConvBlock3D(channels, int(cfg.feat_channels)))
            channels = int(cfg.feat_channels)
        self.encoder = nn.Sequential(*blocks)
        self.scorer = nn.Sequential(
            nn.Linear(int(cfg.feat_channels), int(cfg.mlp_hidden)),
            nn.LeakyReLU(inplace=True),
            nn.Linear(int(cfg.mlp_hidden), 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        supervoxels: torch.Tensor,
    ) -> tuple[List[torch.Tensor], List[torch.Tensor]]:
        if x.ndim != 5 or supervoxels.ndim != 4:
            raise ValueError("expected x=(B,C,D,H,W), supervoxels=(B,D,H,W)")
        if x.shape[0] != supervoxels.shape[0] or x.shape[2:] != supervoxels.shape[1:]:
            raise ValueError("x and supervoxels must share batch and spatial dimensions")

        features = self.encoder(x)
        logits_by_case: List[torch.Tensor] = []
        ids_by_case: List[torch.Tensor] = []
        for batch_index in range(x.shape[0]):
            labels = supervoxels[batch_index].long()
            ids, inverse = torch.unique(labels, sorted=True, return_inverse=True)
            flattened_features = features[batch_index].reshape(features.shape[1], -1).transpose(0, 1)
            inverse_flat = inverse.reshape(-1)
            pooled_sum = torch.zeros(
                (ids.numel(), features.shape[1]), device=features.device, dtype=features.dtype
            )
            pooled_sum.index_add_(0, inverse_flat, flattened_features)
            counts = torch.bincount(inverse_flat, minlength=ids.numel()).to(features.dtype).unsqueeze(1)
            pooled_mean = pooled_sum / counts.clamp_min(1.0)
            logits_by_case.append(self.scorer(pooled_mean))
            ids_by_case.append(ids)
        return logits_by_case, ids_by_case
