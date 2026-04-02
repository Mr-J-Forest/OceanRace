"""要素长期预测推理器。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from element_forecasting.model import HybridElementForecastModel
from utils.dataset_utils import destandardize_tensor, load_norm_stats, project_root


class ElementForecastPredictor:
	def __init__(
		self,
		checkpoint_path: str | Path,
		device: str = "auto",
		norm_stats_path: str | Path | None = None,
	) -> None:
		ckpt = torch.load(Path(checkpoint_path), map_location="cpu")
		model_cfg = ckpt.get("model_config", {})
		in_channels = int(ckpt["in_channels"])
		input_steps = int(ckpt["input_steps"])
		output_steps = int(ckpt["output_steps"])

		if device == "auto":
			device = "cuda" if torch.cuda.is_available() else "cpu"
		self.device = torch.device(device)

		self.model = HybridElementForecastModel(
			in_channels=in_channels,
			input_steps=input_steps,
			output_steps=output_steps,
			d_model=int(model_cfg.get("d_model", 128)),
			nhead=int(model_cfg.get("nhead", 4)),
			num_layers=int(model_cfg.get("num_layers", 6)),
			block_size=int(model_cfg.get("block_size", 4)),
			dropout=float(model_cfg.get("dropout", 0.1)),
			spatial_downsample=int(model_cfg.get("spatial_downsample", 4)),
			periodic_periods=model_cfg.get("periodic_periods", [24.0]),
			periodic_harmonics=int(model_cfg.get("periodic_harmonics", 1)),
		)
		self.model.load_state_dict(ckpt["model_state"])
		self.model.to(self.device)
		self.model.eval()
		self.var_names = tuple(ckpt.get("var_names", []))
		self.input_steps = input_steps
		self.output_steps = output_steps
		if norm_stats_path is None:
			norm_stats_path = project_root() / "data/processed/normalization/element_forecasting_norm.json"
		norm_path = Path(norm_stats_path)
		self._norm = load_norm_stats(norm_path) if norm_path.is_file() else None

	def _destandardize_pred(self, pred: torch.Tensor) -> torch.Tensor:
		# shape: [B, T_out, C, H, W]
		if self._norm is None or not self.var_names:
			return pred
		out = pred.clone()
		n_channels = int(out.shape[2])
		for c, key in enumerate(self.var_names[:n_channels]):
			out[:, :, c, :, :] = destandardize_tensor(out[:, :, c, :, :], key, self._norm)
		return out

	@torch.no_grad()
	def predict(
		self,
		x: torch.Tensor,
		denormalize: bool = True,
		return_cpu: bool = True,
	) -> dict[str, Any]:
		"""x shape: ``(B, input_steps, C, H, W)``。"""

		x = x.to(self.device)
		out = self.model(x)
		pred = out["pred"]
		pred_transformer = out["pred_transformer"]
		if denormalize:
			pred = self._destandardize_pred(pred)
			pred_transformer = self._destandardize_pred(pred_transformer)
		if return_cpu:
			pred = pred.cpu()
			pred_transformer = pred_transformer.cpu()
		return {
			"pred": pred,
			"pred_transformer": pred_transformer,
			"var_names": self.var_names,
			"denormalized": denormalize,
		}

	@torch.no_grad()
	def predict_long_horizon(
		self,
		x: torch.Tensor,
		target_steps: int,
		overlap_steps: int = 4,
		enable_overlap_blend: bool = True,
		denormalize: bool = True,
		return_cpu: bool = True,
	) -> dict[str, Any]:
		"""长时滚动预测，支持重叠线性融合以减小块边界台阶跳变。"""

		if target_steps <= 0:
			raise ValueError("target_steps must be > 0")
		model_steps = int(self.output_steps)
		if model_steps <= 0:
			raise ValueError("model output_steps must be > 0")

		overlap = int(max(0, overlap_steps if enable_overlap_blend else 0))
		if overlap >= model_steps:
			overlap = max(0, model_steps - 1)
		stride = max(1, model_steps - overlap)

		cur_x = x.to(self.device)
		out_seq: torch.Tensor | None = None
		cursor_start = 0

		while out_seq is None or out_seq.shape[1] < target_steps:
			out = self.predict(cur_x, denormalize=False, return_cpu=False)
			pred_chunk = out["pred"].float()

			if out_seq is None:
				out_seq = pred_chunk
			else:
				overlap_len = max(0, out_seq.shape[1] - cursor_start)
				overlap_len = min(overlap_len, pred_chunk.shape[1])
				if overlap_len > 0:
					if enable_overlap_blend:
						alpha = torch.linspace(0.0, 1.0, steps=overlap_len, device=pred_chunk.device)
						alpha = alpha.view(1, overlap_len, 1, 1, 1)
						old = out_seq[:, cursor_start:cursor_start + overlap_len]
						new = pred_chunk[:, :overlap_len]
						out_seq[:, cursor_start:cursor_start + overlap_len] = old * (1.0 - alpha) + new * alpha
					else:
						out_seq[:, cursor_start:cursor_start + overlap_len] = pred_chunk[:, :overlap_len]

				tail = pred_chunk[:, overlap_len:]
				if tail.shape[1] > 0:
					out_seq = torch.cat([out_seq, tail], dim=1)

			feed_steps = min(stride, pred_chunk.shape[1])
			feed = pred_chunk[:, :feed_steps]
			cur_x = torch.cat([cur_x.float(), feed], dim=1)[:, -self.input_steps:].contiguous()
			cursor_start += stride

		assert out_seq is not None
		pred = out_seq[:, :target_steps]
		pred_transformer = pred

		if denormalize:
			pred = self._destandardize_pred(pred)
			pred_transformer = self._destandardize_pred(pred_transformer)
		if return_cpu:
			pred = pred.cpu()
			pred_transformer = pred_transformer.cpu()
		return {
			"pred": pred,
			"pred_transformer": pred_transformer,
			"var_names": self.var_names,
			"denormalized": denormalize,
		}
