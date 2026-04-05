"""Anomaly baseline model: lightweight dual-branch convolutional autoencoder."""
from __future__ import annotations

import torch
import torch.nn as nn


class _ShallowBranchAE(nn.Module):
    """A shallow autoencoder branch used as anomaly baseline."""

    def __init__(self, in_channels: int = 3, hidden_channels: int = 12):
        super().__init__()
        h = hidden_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, h, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(h),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(h, h * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(h * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(h * 2, h, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(h),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(h, in_channels, kernel_size=3, padding=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z_map = self.encoder(x)
        z = self.pool(z_map).flatten(1)
        y = self.decoder(z_map)
        if y.shape[-2:] != x.shape[-2:]:
            y = nn.functional.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return y, z


class DualBranchAEBaseline(nn.Module):
    """Simple baseline: independent shallow AEs for oper and wave branches."""

    def __init__(self, oper_channels: int = 3, wave_channels: int = 3, base_channels: int = 12):
        super().__init__()
        self.oper_branch = _ShallowBranchAE(in_channels=oper_channels, hidden_channels=base_channels)
        self.wave_branch = _ShallowBranchAE(in_channels=wave_channels, hidden_channels=base_channels)

    def forward(self, oper_x: torch.Tensor, wave_x: torch.Tensor) -> dict[str, torch.Tensor]:
        oper_recon, oper_z = self.oper_branch(oper_x)
        wave_recon, wave_z = self.wave_branch(wave_x)
        return {
            "oper_recon": oper_recon,
            "wave_recon": wave_recon,
            "oper_z": oper_z,
            "wave_z": wave_z,
        }
