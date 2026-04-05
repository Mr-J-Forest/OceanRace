"""要素长期预测模型：Transformer + Block Attn-Res。"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableRefineHead(nn.Module):
	"""轻量空间细化头：使用 depthwise + pointwise 结构恢复局地细节。"""

	def __init__(
		self,
		channels: int,
		num_layers: int = 2,
		hidden_ratio: float = 1.0,
		residual: bool = True,
	) -> None:
		super().__init__()
		self.residual = residual
		num_layers = max(1, int(num_layers))
		hidden_channels = max(1, int(round(channels * float(hidden_ratio))))
		blocks: list[nn.Module] = []
		for _ in range(num_layers):
			blocks.append(
				nn.Sequential(
					nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
					nn.GELU(),
					nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=True),
					nn.GELU(),
					nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True),
				)
			)
		self.blocks = nn.ModuleList(blocks)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		out = x
		for block in self.blocks:
			delta = block(out)
			out = out + delta if self.residual else delta
		return out


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
		n1 = self.norm1(x)
		attn_out, _ = self.self_attn(n1, n1, n1, need_weights=False)
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


class SpatioTemporalTransformerBranch(nn.Module):
	"""时空双轨 Transformer：先在每帧内进行空间 Attention，再沿时间轴进行 Temporal Attention。"""

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
		multi_scale_enabled: bool = False,
		aux_spatial_downsample: int = 8,
		multi_scale_fusion: str = "residual_add",
		multi_scale_aux_weight: float = 0.35,
		periodic_periods: tuple[float, ...] | list[float] | None = None,
		periodic_harmonics: int = 1,
		refine_head_enabled: bool = False,
		refine_head_hidden_ratio: float = 1.0,
		refine_head_num_layers: int = 2,
		refine_head_residual: bool = True,
	) -> None:
		super().__init__()
		self.in_channels = in_channels
		self.input_steps = input_steps
		self.output_steps = output_steps
		self.patch_size = max(1, int(spatial_downsample))
		self.multi_scale_enabled = bool(multi_scale_enabled)
		self.aux_patch_size = max(1, int(aux_spatial_downsample))
		self.multi_scale_fusion = str(multi_scale_fusion).strip().lower()
		self.multi_scale_aux_weight = float(multi_scale_aux_weight)
		if periodic_periods is None:
			periodic_periods = (24.0,)
		periods = [float(p) for p in periodic_periods if float(p) > 0]
		if not periods:
			periods = [24.0]
		self.periodic_harmonics = max(1, int(periodic_harmonics))
		period_tensor = torch.tensor(periods, dtype=torch.float32)
		self.register_buffer("periodic_periods", period_tensor, persistent=False)
		self.time_feature_dim = int(2 * len(periods) * self.periodic_harmonics)

		self.patch_embed = nn.Conv2d(
			in_channels, d_model, kernel_size=self.patch_size, stride=self.patch_size
		)
		self.patch_embed_aux = None
		if self.multi_scale_enabled:
			self.patch_embed_aux = nn.Conv2d(
				in_channels,
				d_model,
				kernel_size=self.aux_patch_size,
				stride=self.aux_patch_size,
			)
		
		# 预设最大网格数量
		self.max_spatial_tokens = 32768
		self.spa_pos_emb = nn.Parameter(torch.zeros(1, self.max_spatial_tokens, d_model))

		num_spa_layers = max(1, num_layers // 2)
		spa_layer = nn.TransformerEncoderLayer(
			d_model=d_model, nhead=nhead, dim_feedforward=int(d_model * 4), 
			dropout=dropout, batch_first=True, norm_first=True
		)
		self.spatial_encoder = nn.TransformerEncoder(spa_layer, num_layers=num_spa_layers)

		self.tem_pos_emb = nn.Parameter(torch.zeros(1, input_steps, d_model))
		self.tod_emb = nn.Linear(self.time_feature_dim, d_model)

		num_tem_layers = max(1, num_layers - num_spa_layers)
		self.temporal_encoder = BlockResidualTransformerEncoder(
			num_layers=num_tem_layers,
			d_model=d_model,
			nhead=nhead,
			block_size=block_size,
			dropout=dropout,
		)

		self.out_proj = nn.Linear(d_model, output_steps * in_channels * self.patch_size * self.patch_size)
		self.refine_head = None
		if bool(refine_head_enabled):
			self.refine_head = DepthwiseSeparableRefineHead(
				channels=in_channels,
				num_layers=refine_head_num_layers,
				hidden_ratio=refine_head_hidden_ratio,
				residual=refine_head_residual,
			)

	def _build_periodic_time_features(self, abs_time: torch.Tensor) -> torch.Tensor:
		"""Build multi-period harmonic features with shape (B, T, 2 * periods * harmonics)."""

		t = abs_time.to(dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
		periods = self.periodic_periods.to(device=abs_time.device, dtype=torch.float32).view(1, 1, -1, 1)
		harmonics = torch.arange(
			1,
			self.periodic_harmonics + 1,
			device=abs_time.device,
			dtype=torch.float32,
		).view(1, 1, 1, -1)
		angles = 2.0 * torch.pi * harmonics * t / periods
		sin_feat = torch.sin(angles)
		cos_feat = torch.cos(angles)
		return torch.cat([sin_feat, cos_feat], dim=-1).flatten(start_dim=2)

	def forward(self, x: torch.Tensor, t0: torch.Tensor | None = None) -> torch.Tensor:
		if x.dim() != 5:
			raise ValueError(f"expected x with shape (B,T,C,H,W), got {tuple(x.shape)}")
		bsz, t_in, ch, h, w = x.shape
		
		# 1. Patch Embedding & 空间 Transformer
		x_reshaped = x.reshape(bsz * t_in, ch, h, w)
		patches = self.patch_embed(x_reshaped)
		if self.multi_scale_enabled and self.patch_embed_aux is not None:
			aux_patches = self.patch_embed_aux(x_reshaped)
			if aux_patches.shape[-2:] != patches.shape[-2:]:
				aux_patches = F.interpolate(aux_patches, size=patches.shape[-2:], mode="bilinear", align_corners=False)
			if self.multi_scale_fusion == "residual_add":
				patches = patches + (self.multi_scale_aux_weight * aux_patches)
			else:
				patches = 0.5 * patches + 0.5 * aux_patches
		_, d, h_r, w_r = patches.shape
		num_spatial_tokens = h_r * w_r
		
		spa_tokens = patches.reshape(bsz * t_in, d, num_spatial_tokens).permute(0, 2, 1)
		spa_tokens = spa_tokens + self.spa_pos_emb[:, :num_spatial_tokens, :]
		spa_out = self.spatial_encoder(spa_tokens)

		# 2. 时空转换与时间 Transformer (B, Grid, T_in, D)
		tem_tokens = spa_out.view(bsz, t_in, num_spatial_tokens, d).permute(0, 2, 1, 3)
		
		device = x.device
		seq_idx = torch.arange(t_in, device=device).unsqueeze(0).expand(bsz, t_in)
		abs_time = (seq_idx + t0.unsqueeze(1)) if t0 is not None else seq_idx
			
		periodic_feats = self._build_periodic_time_features(abs_time)
		tod_embeds = self.tod_emb(periodic_feats)
		
		tem_tokens = tem_tokens + self.tem_pos_emb[:, :t_in, :].unsqueeze(1) + tod_embeds.unsqueeze(1)
		tem_in = tem_tokens.reshape(bsz * num_spatial_tokens, t_in, d)
		tem_out = self.temporal_encoder(tem_in) 
		
		# 3. 解码头
		tail = tem_out[:, -1, :]
		pred_flat = self.out_proj(tail)
		
		pred = pred_flat.view(bsz, h_r, w_r, self.output_steps, ch, self.patch_size, self.patch_size)
		pred = pred.permute(0, 3, 4, 1, 5, 2, 6).contiguous()
		pred = pred.reshape(bsz, self.output_steps, ch, h_r * self.patch_size, w_r * self.patch_size)
		
		if pred.shape[-2:] != (h, w):
			pred = F.interpolate(pred.flatten(0,1), size=(h, w), mode="bilinear", align_corners=False)
			pred = pred.view(bsz, self.output_steps, ch, h, w)

		if self.refine_head is not None:
			pred_2d = pred.flatten(0, 1)
			pred_2d = self.refine_head(pred_2d)
			pred = pred_2d.view(bsz, self.output_steps, ch, h, w)
			
		return pred


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
		multi_scale_enabled: bool = False,
		aux_spatial_downsample: int = 8,
		multi_scale_fusion: str = "residual_add",
		multi_scale_aux_weight: float = 0.35,
		periodic_periods: tuple[float, ...] | list[float] | None = None,
		periodic_harmonics: int = 1,
		refine_head_enabled: bool = False,
		refine_head_hidden_ratio: float = 1.0,
		refine_head_num_layers: int = 2,
		refine_head_residual: bool = True,
	) -> None:
		super().__init__()
		self.output_steps = output_steps
		self.transformer = SpatioTemporalTransformerBranch(
			in_channels=in_channels,
			input_steps=input_steps,
			output_steps=output_steps,
			d_model=d_model,
			nhead=nhead,
			num_layers=num_layers,
			block_size=block_size,
			dropout=dropout,
			spatial_downsample=spatial_downsample,
			multi_scale_enabled=multi_scale_enabled,
			aux_spatial_downsample=aux_spatial_downsample,
			multi_scale_fusion=multi_scale_fusion,
			multi_scale_aux_weight=multi_scale_aux_weight,
			periodic_periods=periodic_periods,
			periodic_harmonics=periodic_harmonics,
			refine_head_enabled=refine_head_enabled,
			refine_head_hidden_ratio=refine_head_hidden_ratio,
			refine_head_num_layers=refine_head_num_layers,
			refine_head_residual=refine_head_residual,
		)

	def forward(self, x: torch.Tensor, t0: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
		pred_transformer = self.transformer(x, t0=t0)
		pred = pred_transformer
		return {
			"pred": pred,
			"pred_transformer": pred_transformer,
		}
