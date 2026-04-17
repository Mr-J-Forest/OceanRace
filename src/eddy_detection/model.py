"""中尺度涡旋分割模型（U-Net baseline）。"""
from __future__ import annotations

import torch
import torch.nn as nn


class _DoubleConv(nn.Module):
	def __init__(self, in_ch: int, out_ch: int) -> None:
		super().__init__()
		self.block = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.block(x)


class EddyUNet(nn.Module):
	"""U-Net 三分类分割网络。"""

	def __init__(
		self,
		in_channels: int = 3,
		num_classes: int = 3,
		base_channels: int = 32,
	) -> None:
		super().__init__()
		b = base_channels
		self.enc1 = _DoubleConv(in_channels, b)
		self.enc2 = _DoubleConv(b, b * 2)
		self.enc3 = _DoubleConv(b * 2, b * 4)
		self.enc4 = _DoubleConv(b * 4, b * 8)
		self.pool = nn.MaxPool2d(2)

		self.bottleneck = _DoubleConv(b * 8, b * 16)

		self.up4 = nn.ConvTranspose2d(b * 16, b * 8, kernel_size=2, stride=2)
		self.dec4 = _DoubleConv(b * 16, b * 8)
		self.up3 = nn.ConvTranspose2d(b * 8, b * 4, kernel_size=2, stride=2)
		self.dec3 = _DoubleConv(b * 8, b * 4)
		self.up2 = nn.ConvTranspose2d(b * 4, b * 2, kernel_size=2, stride=2)
		self.dec2 = _DoubleConv(b * 4, b * 2)
		self.up1 = nn.ConvTranspose2d(b * 2, b, kernel_size=2, stride=2)
		self.dec1 = _DoubleConv(b * 2, b)

		self.head = nn.Conv2d(b, num_classes, kernel_size=1)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		e1 = self.enc1(x)
		e2 = self.enc2(self.pool(e1))
		e3 = self.enc3(self.pool(e2))
		e4 = self.enc4(self.pool(e3))
		b = self.bottleneck(self.pool(e4))

		d4 = self.up4(b)
		d4 = torch.cat([d4, e4], dim=1)
		d4 = self.dec4(d4)

		d3 = self.up3(d4)
		d3 = torch.cat([d3, e3], dim=1)
		d3 = self.dec3(d3)

		d2 = self.up2(d3)
		d2 = torch.cat([d2, e2], dim=1)
		d2 = self.dec2(d2)

		d1 = self.up1(d2)
		d1 = torch.cat([d1, e1], dim=1)
		d1 = self.dec1(d1)
		return self.head(d1)

