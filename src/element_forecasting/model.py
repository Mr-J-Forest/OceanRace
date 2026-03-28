"""要素长期预测模型：Transformer + Block Attn-Res。"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
	"""轻量 RMSNorm，用于稳定 block-level attention 聚合。"""

	def __init__(self, dim: int, eps: float = 1e-6) -> None:
		super().__init__()
		self.eps = eps
		self.weight = nn.Parameter(torch.ones(dim))

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
		return (x / rms) * self.weight


def block_attn_res(
	blocks: list[torch.Tensor],
	partial_block: torch.Tensor,
	proj: nn.Linear,
	norm: RMSNorm,
) -> torch.Tensor:
	"""Inter-block attention: 聚合历史 block 与当前 partial block 的上下文。"""

	v = torch.stack(blocks + [partial_block], dim=0)
	k = norm(v)
	logits = torch.einsum("d, n b t d -> n b t", proj.weight.squeeze(0), k)
	weights = logits.softmax(dim=0)
	h = torch.einsum("n b t, n b t d -> b t d", weights, v)
	return h


class BlockResidualTransformerLayer(nn.Module):
	"""在 Attention 与 MLP 之前都注入一次 block_attn_res。"""

	def __init__(self, d_model: int, nhead: int, mlp_ratio: float = 4.0, dropout: float = 0.1) -> None:
		super().__init__()
		self.norm1 = RMSNorm(d_model)
		self.norm2 = RMSNorm(d_model)
		self.self_attn = nn.MultiheadAttention(
			embed_dim=d_model,
			num_heads=nhead,
			dropout=dropout,
			batch_first=True,
		)
		hidden = int(d_model * mlp_ratio)
		self.mlp = nn.Sequential(
			nn.Linear(d_model, hidden),
			nn.GELU(),
			nn.Dropout(dropout),
			nn.Linear(hidden, d_model),
			nn.Dropout(dropout),
		)
		self.block_proj_attn = nn.Linear(d_model, 1, bias=False)
		self.block_proj_mlp = nn.Linear(d_model, 1, bias=False)
		self.block_norm_attn = RMSNorm(d_model)
		self.block_norm_mlp = RMSNorm(d_model)

	def forward(
		self,
		x: torch.Tensor,
		blocks: list[torch.Tensor],
		partial_block: torch.Tensor | None,
	) -> torch.Tensor:
		partial = x if partial_block is None else partial_block
		x = x + block_attn_res(blocks, partial, self.block_proj_attn, self.block_norm_attn)
		attn_out, _ = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)
		x = x + attn_out
		x = x + block_attn_res(blocks, partial, self.block_proj_mlp, self.block_norm_mlp)
		x = x + self.mlp(self.norm2(x))
		return x


class BlockResidualTransformerEncoder(nn.Module):
	"""维护 ``blocks`` 与 ``partial_block`` 的 Transformer 编码器。"""

	def __init__(self, num_layers: int, d_model: int, nhead: int, block_size: int, dropout: float = 0.1) -> None:
		super().__init__()
		self.layers = nn.ModuleList(
			[
				BlockResidualTransformerLayer(d_model=d_model, nhead=nhead, dropout=dropout)
				for _ in range(num_layers)
			]
		)
		self.block_size = max(2, int(block_size))
		self.block_stride = max(1, self.block_size // 2)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		blocks: list[torch.Tensor] = []
		partial_block: torch.Tensor | None = None
		for layer_number, layer in enumerate(self.layers, start=1):
			x = layer(x, blocks=blocks, partial_block=partial_block)
			partial_block = x if partial_block is None else (partial_block + x)
			if layer_number % self.block_stride == 0:
				blocks.append(partial_block)
				partial_block = None
		return x


class TransformerForecastBranch(nn.Module):
	"""将每个网格点作为 token 序列建模并输出多步预测。"""

	def __init__(
		self,
		in_channels: int,
		input_steps: int,
		output_steps: int,
		d_model: int = 128,
		nhead: int = 4,
		num_layers: int = 6,
		block_size: int = 4,
		dropout: float = 0.1,
		spatial_downsample: int = 4,
	) -> None:
		super().__init__()
		self.in_channels = in_channels
		self.input_steps = input_steps
		self.output_steps = output_steps
		self.spatial_downsample = max(1, int(spatial_downsample))
		self.in_proj = nn.Linear(in_channels, d_model)
		self.pos_emb = nn.Parameter(torch.zeros(1, input_steps, d_model))
		# Time-of-Day Embedding: Sin and Cos projection
		self.tod_emb = nn.Linear(2, d_model)
		self.encoder = BlockResidualTransformerEncoder(
			num_layers=num_layers,
			d_model=d_model,
			nhead=nhead,
			block_size=block_size,
			dropout=dropout,
		)
		self.out_proj = nn.Linear(d_model, output_steps * in_channels)

	def forward(self, x: torch.Tensor, t0: torch.Tensor | None = None) -> torch.Tensor:
		if x.dim() != 5:
			raise ValueError(f"expected x with shape (B,T,C,H,W), got {tuple(x.shape)}")
		bsz, t_in, ch, h, w = x.shape
		if t_in != self.input_steps:
			raise ValueError(f"expected input_steps={self.input_steps}, got {t_in}")
		if ch != self.in_channels:
			raise ValueError(f"expected in_channels={self.in_channels}, got {ch}")

		if self.spatial_downsample > 1:
			xr = x.reshape(bsz * t_in, ch, h, w)
			xr = F.avg_pool2d(xr, kernel_size=self.spatial_downsample, stride=self.spatial_downsample)
			h_r, w_r = xr.shape[-2], xr.shape[-1]
			x_work = xr.reshape(bsz, t_in, ch, h_r, w_r)
		else:
			h_r, w_r = h, w
			x_work = x

		# Time-of-Day Embedding (using t0 if provided, else relative sequence)
		device = x.device
		seq_idx = torch.arange(t_in, device=device).unsqueeze(0).expand(bsz, t_in)
		if t0 is not None:
			# t0 shape: (B,)
			abs_time = seq_idx + t0.unsqueeze(1)
		else:
			abs_time = seq_idx
		
		# 假设 1 step = 1 小时周期
		sin_tod = torch.sin(2 * torch.pi * abs_time / 24.0).unsqueeze(-1)
		cos_tod = torch.cos(2 * torch.pi * abs_time / 24.0).unsqueeze(-1)
		tod_feats = torch.cat([sin_tod, cos_tod], dim=-1) # (B, T_in, 2)
		tod_embeds = self.tod_emb(tod_feats) # (B, T_in, d_model)

		# 每个空间位置独立建模时间序列，统一批处理提升吞吐。
		token = x_work.permute(0, 3, 4, 1, 2).reshape(bsz * h_r * w_r, t_in, ch)
		# 维度对齐: token 是 (B*H*W, T, C), tod_embeds 是 (B, T, D)
		# 需要把 tod 膨胀到各个网格位置 (B, 1, 1, T, D) -> (B*H*W, T, D)
		tod_embeds_expanded = tod_embeds.view(bsz, 1, 1, t_in, -1).expand(bsz, h_r, w_r, t_in, -1)
		tod_embeds_expanded = tod_embeds_expanded.reshape(bsz * h_r * w_r, t_in, -1)

		h_tok = self.in_proj(token) + self.pos_emb[:, :t_in, :] + tod_embeds_expanded
		h_tok = self.encoder(h_tok)
		tail = h_tok[:, -1, :]
		pred_low = self.out_proj(tail).view(bsz, h_r, w_r, self.output_steps, ch)
		pred_low = pred_low.permute(0, 3, 4, 1, 2).contiguous()

		if self.spatial_downsample > 1:
			up = pred_low.reshape(bsz * self.output_steps, ch, h_r, w_r)
			up = F.interpolate(up, size=(h, w), mode="bilinear", align_corners=False)
			return up.reshape(bsz, self.output_steps, ch, h, w).contiguous()
		return pred_low


class HybridElementForecastModel(nn.Module):
	"""长时序预测主模型（仅 Transformer 分支）。"""

	def __init__(
		self,
		in_channels: int,
		input_steps: int,
		output_steps: int,
		d_model: int = 128,
		nhead: int = 4,
		num_layers: int = 6,
		block_size: int = 4,
		dropout: float = 0.1,
		spatial_downsample: int = 4,
	) -> None:
		super().__init__()
		self.output_steps = output_steps
		self.transformer = TransformerForecastBranch(
			in_channels=in_channels,
			input_steps=input_steps,
			output_steps=output_steps,
			d_model=d_model,
			nhead=nhead,
			num_layers=num_layers,
			block_size=block_size,
			dropout=dropout,
			spatial_downsample=spatial_downsample,
		)

	def forward(self, x: torch.Tensor, t0: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
		pred_transformer = self.transformer(x, t0=t0)
		pred = pred_transformer
		return {
			"pred": pred,
			"pred_transformer": pred_transformer,
		}
