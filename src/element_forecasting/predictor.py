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
