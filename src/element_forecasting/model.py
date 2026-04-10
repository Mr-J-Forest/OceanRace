"""要素长期预测模型：ViT Transformer + 多专家融合。"""
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
	"""时空双轨 ViT Transformer：空间 ViT 编码后进行时间 ViT 编码。"""

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
		tem_layer = nn.TransformerEncoderLayer(
			d_model=d_model,
			nhead=nhead,
			dim_feedforward=int(d_model * 4),
			dropout=dropout,
			batch_first=True,
			norm_first=True,
		)
		self.temporal_encoder = nn.TransformerEncoder(tem_layer, num_layers=num_tem_layers)
		self.temporal_norm = nn.LayerNorm(d_model)
		self.shared_head = nn.Sequential(
			nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
			nn.GELU(),
			nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
			nn.GELU(),
		)
		self.var_heads = nn.ModuleList(
			[
				nn.Sequential(
					nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
					nn.GELU(),
					nn.Conv2d(d_model, output_steps, kernel_size=1),
				)
				for _ in range(in_channels)
			]
		)
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

		tem_tokens = spa_out.view(bsz, t_in, num_spatial_tokens, d).permute(0, 2, 1, 3)
		
		device = x.device
		seq_idx = torch.arange(t_in, device=device).unsqueeze(0).expand(bsz, t_in)
		abs_time = (seq_idx + t0.unsqueeze(1)) if t0 is not None else seq_idx
			
		periodic_feats = self._build_periodic_time_features(abs_time)
		tod_embeds = self.tod_emb(periodic_feats)
		
		tem_tokens = tem_tokens + self.tem_pos_emb[:, :t_in, :].unsqueeze(1) + tod_embeds.unsqueeze(1)
		tem_in = tem_tokens.reshape(bsz * num_spatial_tokens, t_in, d)
		tem_out = self.temporal_encoder(tem_in)
		tem_out = self.temporal_norm(tem_out)
		tail = tem_out[:, -1, :]
		tail_map = tail.view(bsz, h_r, w_r, d).permute(0, 3, 1, 2).contiguous()
		if tail_map.shape[-2:] != (h, w):
			tail_map = F.interpolate(tail_map, size=(h, w), mode="bilinear", align_corners=False)
		shared = self.shared_head(tail_map)
		pred_channels = [head(shared) for head in self.var_heads]
		pred = torch.stack(pred_channels, dim=2)
		
		if pred.shape[-2:] != (h, w):
			pred = F.interpolate(pred.flatten(0,1), size=(h, w), mode="bilinear", align_corners=False)
			pred = pred.view(bsz, self.output_steps, ch, h, w)

		if self.refine_head is not None:
			pred_2d = pred.flatten(0, 1)
			pred_2d = self.refine_head(pred_2d)
			pred = pred_2d.view(bsz, self.output_steps, ch, h, w)
			
		return pred


class _DoubleConv(nn.Module):
	"""UNet基础卷积块。"""

	def __init__(self, in_channels: int, out_channels: int) -> None:
		super().__init__()
		self.net = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.GELU(),
			nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.GELU(),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.net(x)


class UNetExpert(nn.Module):
	"""将输入时间窗展平到通道维度的轻量UNet专家。"""

	def __init__(
		self,
		in_channels: int,
		input_steps: int,
		output_steps: int,
		base_channels: int = 48,
	) -> None:
		super().__init__()
		flat_in = int(in_channels * input_steps)
		self.output_steps = int(output_steps)
		self.in_channels = int(in_channels)

		c1 = max(16, int(base_channels))
		c2 = c1 * 2
		c3 = c2 * 2

		self.enc1 = _DoubleConv(flat_in, c1)
		self.enc2 = _DoubleConv(c1, c2)
		self.enc3 = _DoubleConv(c2, c3)
		self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

		self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
		self.dec2 = _DoubleConv(c2 + c2, c2)
		self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
		self.dec1 = _DoubleConv(c1 + c1, c1)
		self.head = nn.Conv2d(c1, output_steps * in_channels, kernel_size=1)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		bsz, t_in, ch, h, w = x.shape
		x2d = x.reshape(bsz, t_in * ch, h, w)

		e1 = self.enc1(x2d)
		e2 = self.enc2(self.pool(e1))
		e3 = self.enc3(self.pool(e2))

		d2 = self.up2(e3)
		if d2.shape[-2:] != e2.shape[-2:]:
			d2 = F.interpolate(d2, size=e2.shape[-2:], mode="bilinear", align_corners=False)
		d2 = self.dec2(torch.cat([d2, e2], dim=1))

		d1 = self.up1(d2)
		if d1.shape[-2:] != e1.shape[-2:]:
			d1 = F.interpolate(d1, size=e1.shape[-2:], mode="bilinear", align_corners=False)
		d1 = self.dec1(torch.cat([d1, e1], dim=1))

		out = self.head(d1)
		return out.view(bsz, self.output_steps, self.in_channels, h, w)


class TrajGRUCell(nn.Module):
	"""Trajectory GRU 单元。"""

	def __init__(
		self,
		input_channels: int,
		hidden_channels: int,
		kernel_size: int = 3,
		num_links: int = 9,
		flow_hidden_channels: int = 32,
		flow_clip: float = 4.0,
	) -> None:
		super().__init__()
		padding = kernel_size // 2
		self.hidden_channels = int(hidden_channels)
		self.num_links = max(1, int(num_links))
		self.flow_clip = float(max(0.0, flow_clip))
		flow_mid = max(8, int(flow_hidden_channels))
		self.flow_generator = nn.Sequential(
			nn.Conv2d(input_channels + hidden_channels, flow_mid, kernel_size=3, padding=1),
			nn.GELU(),
			nn.Conv2d(flow_mid, 2 * self.num_links, kernel_size=3, padding=1),
		)
		self.conv_zr = nn.Conv2d(
			input_channels + hidden_channels,
			2 * hidden_channels,
			kernel_size=kernel_size,
			padding=padding,
		)
		self.conv_xn = nn.Conv2d(input_channels, hidden_channels, kernel_size=kernel_size, padding=padding)
		self.conv_hn = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding=padding)

	def _warp(self, h: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
		bsz, _, hgt, wid = h.shape
		yy = torch.linspace(-1.0, 1.0, steps=hgt, device=h.device, dtype=h.dtype)
		xx = torch.linspace(-1.0, 1.0, steps=wid, device=h.device, dtype=h.dtype)
		grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
		base_grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).expand(bsz, hgt, wid, 2)
		norm_x = flow[:, 0] * (2.0 / max(1, wid - 1))
		norm_y = flow[:, 1] * (2.0 / max(1, hgt - 1))
		flow_grid = torch.stack((norm_x, norm_y), dim=-1)
		sample_grid = base_grid + flow_grid
		return F.grid_sample(h, sample_grid, mode="bilinear", padding_mode="border", align_corners=True)

	def forward(
		self,
		x: torch.Tensor,
		h_prev: torch.Tensor,
	) -> torch.Tensor:
		flow_raw = self.flow_generator(torch.cat([x, h_prev], dim=1))
		flows = flow_raw.view(flow_raw.shape[0], self.num_links, 2, flow_raw.shape[-2], flow_raw.shape[-1])
		if self.flow_clip > 0.0:
			flows = torch.tanh(flows) * self.flow_clip
		warped_states: list[torch.Tensor] = []
		for i in range(self.num_links):
			warped_states.append(self._warp(h_prev, flows[:, i]))
		h_agg = torch.stack(warped_states, dim=0).mean(dim=0)
		zr = torch.sigmoid(self.conv_zr(torch.cat([x, h_agg], dim=1)))
		z, r = torch.chunk(zr, 2, dim=1)
		n = torch.tanh(self.conv_xn(x) + self.conv_hn(r * h_agg))
		return (1.0 - z) * n + z * h_prev


class TrajGRUExpert(nn.Module):
	"""以 TrajGRU 编码输入序列，再解码为多步输出的专家分支。"""

	def __init__(
		self,
		in_channels: int,
		output_steps: int,
		hidden_channels: int = 64,
		kernel_size: int = 3,
		num_links: int = 9,
		flow_hidden_channels: int = 32,
		flow_clip: float = 4.0,
	) -> None:
		super().__init__()
		self.in_channels = int(in_channels)
		self.output_steps = int(output_steps)
		hid = max(16, int(hidden_channels))
		self.cell = TrajGRUCell(
			input_channels=in_channels,
			hidden_channels=hid,
			kernel_size=kernel_size,
			num_links=num_links,
			flow_hidden_channels=flow_hidden_channels,
			flow_clip=flow_clip,
		)
		self.head = nn.Sequential(
			nn.Conv2d(hid, hid, kernel_size=3, padding=1),
			nn.GELU(),
			nn.Conv2d(hid, output_steps * in_channels, kernel_size=1),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		bsz, _, _, h, w = x.shape
		h_t = x.new_zeros((bsz, self.cell.hidden_channels, h, w))
		for t in range(x.shape[1]):
			h_t = self.cell(x[:, t], h_t)
		out = self.head(h_t)
		return out.view(bsz, self.output_steps, self.in_channels, h, w)


class HybridElementForecastModel(nn.Module):
	"""长时序预测主模型（Transformer + UNet/TrajGRU 多专家）。"""

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
		moe_enabled: bool = False,
		moe_unet_base_channels: int = 48,
		moe_convlstm_hidden_channels: int = 64,
		moe_convlstm_kernel_size: int = 3,
		moe_trajgru_hidden_channels: int | None = None,
		moe_trajgru_kernel_size: int | None = None,
		moe_trajgru_num_links: int = 9,
		moe_trajgru_flow_hidden_channels: int = 32,
		moe_trajgru_flow_clip: float = 4.0,
		moe_transformer_fusion_alpha: float = 0.45,
		moe_residual_beta: float = 0.3,
		moe_focus_channel_indices: tuple[int, ...] | list[int] | None = None,
		moe_focus_boost: float = 0.25,
	) -> None:
		super().__init__()
		self.output_steps = output_steps
		self.in_channels = in_channels
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
		self.moe_enabled = bool(moe_enabled)
		self.moe_transformer_fusion_alpha = float(max(0.0, min(1.0, moe_transformer_fusion_alpha)))
		self.moe_residual_beta = float(moe_residual_beta)
		self.moe_focus_boost = float(max(0.0, moe_focus_boost))
		focus_indices = [
			int(i)
			for i in (moe_focus_channel_indices or [])
			if 0 <= int(i) < int(in_channels)
		]
		self.moe_focus_channel_indices = tuple(sorted(set(focus_indices)))

		self.unet_expert = None
		self.convlstm_expert = None
		self.gate_mlp = None
		if self.moe_enabled:
			self.unet_expert = UNetExpert(
				in_channels=in_channels,
				input_steps=input_steps,
				output_steps=output_steps,
				base_channels=moe_unet_base_channels,
			)
			traj_hidden_channels = int(
				moe_trajgru_hidden_channels
				if moe_trajgru_hidden_channels is not None
				else moe_convlstm_hidden_channels
			)
			traj_kernel_size = int(
				moe_trajgru_kernel_size
				if moe_trajgru_kernel_size is not None
				else moe_convlstm_kernel_size
			)
			self.convlstm_expert = TrajGRUExpert(
				in_channels=in_channels,
				output_steps=output_steps,
				hidden_channels=traj_hidden_channels,
				kernel_size=traj_kernel_size,
				num_links=moe_trajgru_num_links,
				flow_hidden_channels=moe_trajgru_flow_hidden_channels,
				flow_clip=moe_trajgru_flow_clip,
			)
			self.gate_mlp = nn.Sequential(
				nn.Linear(in_channels, max(16, in_channels * 2)),
				nn.GELU(),
				nn.Linear(max(16, in_channels * 2), 2),
			)

	def forward(self, x: torch.Tensor, t0: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
		pred_transformer = self.transformer(x, t0=t0)
		pred_unet = pred_transformer
		pred_convlstm = pred_transformer
		gate_weights = pred_transformer.new_zeros((pred_transformer.shape[0], 2))
		if self.moe_enabled and self.unet_expert is not None and self.convlstm_expert is not None and self.gate_mlp is not None:
			pred_unet = self.unet_expert(x)
			pred_convlstm = self.convlstm_expert(x)
			gate_feat = x[:, -1].mean(dim=(-2, -1))
			gate_logits = self.gate_mlp(gate_feat)
			gate_weights = torch.softmax(gate_logits, dim=-1)
			w_unet = gate_weights[:, 0].view(-1, 1, 1, 1, 1)
			w_convlstm = gate_weights[:, 1].view(-1, 1, 1, 1, 1)
			pred_experts = (w_unet * pred_unet) + (w_convlstm * pred_convlstm)
			pred = pred_transformer + (self.moe_residual_beta * pred_experts)
			if self.moe_focus_channel_indices and self.moe_focus_boost > 0.0:
				idx = list(self.moe_focus_channel_indices)
				delta_focus = (pred_experts - pred_transformer)[:, :, idx]
				pred[:, :, idx] = pred[:, :, idx] + self.moe_focus_boost * delta_focus
		else:
			pred = pred_transformer
		return {
			"pred": pred,
			"pred_transformer": pred_transformer,
			"pred_unet": pred_unet,
			"pred_convlstm": pred_convlstm,
			"moe_gate_weights": gate_weights,
		}
