"""风-浪异常检测模型：双分支卷积自编码器。"""
from __future__ import annotations

import torch
import torch.nn as nn


class _ConvBlock(nn.Module):
	def __init__(self, in_ch: int, out_ch: int):
		super().__init__()
		self.net = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.net(x)


class _BranchAE(nn.Module):
	"""单分支卷积 AE，输入输出同尺寸。"""

	def __init__(self, in_channels: int = 3, base_channels: int = 24):
		super().__init__()
		c = base_channels
		self.enc1 = _ConvBlock(in_channels, c)
		self.down1 = nn.MaxPool2d(2)
		self.enc2 = _ConvBlock(c, c * 2)
		self.down2 = nn.MaxPool2d(2)
		self.bottleneck = _ConvBlock(c * 2, c * 4)

		self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
		self.dec1 = _ConvBlock(c * 4, c * 2)
		self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
		self.dec2 = _ConvBlock(c * 2, c)
		self.out_conv = nn.Conv2d(c, in_channels, kernel_size=1)

		self.pool = nn.AdaptiveAvgPool2d(1)

	def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		h1 = self.enc1(x)
		h2 = self.enc2(self.down1(h1))
		hb = self.bottleneck(self.down2(h2))

		z = self.pool(hb).flatten(1)

		y = self.up1(hb)
		y = self.dec1(y)
		y = self.up2(y)
		y = self.dec2(y)
		y = self.out_conv(y)

		if y.shape[-2:] != x.shape[-2:]:
			y = nn.functional.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)
		return y, z


class DualBranchAutoEncoder(nn.Module):
	"""风分支 + 浪分支双自编码器。"""

	def __init__(
		self,
		oper_channels: int = 3,
		wave_channels: int = 3,
		base_channels: int = 24,
	):
		super().__init__()
		latent_dim = base_channels * 4
		self.oper_branch = _BranchAE(in_channels=oper_channels, base_channels=base_channels)
		self.wave_branch = _BranchAE(in_channels=wave_channels, base_channels=base_channels)
		self.fusion_gate = nn.Sequential(
			nn.Linear(latent_dim * 2, latent_dim),
			nn.ReLU(inplace=True),
			nn.Linear(latent_dim, latent_dim),
			nn.Sigmoid(),
		)
		# Cross heads inject the other branch's latent signal into reconstruction.
		self.oper_from_wave = nn.Linear(latent_dim, oper_channels)
		self.wave_from_oper = nn.Linear(latent_dim, wave_channels)

	def forward(self, oper_x: torch.Tensor, wave_x: torch.Tensor) -> dict[str, torch.Tensor]:
		oper_recon, oper_z = self.oper_branch(oper_x)
		wave_recon, wave_z = self.wave_branch(wave_x)
		gate = self.fusion_gate(torch.cat([oper_z, wave_z], dim=1))
		fused_z = gate * oper_z + (1.0 - gate) * wave_z

		oper_bias = self.oper_from_wave(wave_z).unsqueeze(-1).unsqueeze(-1)
		wave_bias = self.wave_from_oper(oper_z).unsqueeze(-1).unsqueeze(-1)
		oper_cross_recon = oper_recon + oper_bias
		wave_cross_recon = wave_recon + wave_bias
		return {
			"oper_recon": oper_recon,
			"wave_recon": wave_recon,
			"oper_cross_recon": oper_cross_recon,
			"wave_cross_recon": wave_cross_recon,
			"oper_z": oper_z,
			"wave_z": wave_z,
			"fused_z": fused_z,
			"fusion_gate": gate,
		}
