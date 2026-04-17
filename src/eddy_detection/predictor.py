"""中尺度涡旋分割推理与目标提取。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from eddy_detection.postprocess import extract_eddy_objects, remove_small_components


@torch.no_grad()
def predict_logits(model: torch.nn.Module, x: torch.Tensor, device: str) -> torch.Tensor:
	model.eval()
	xx = x.to(device)
	logits = model(xx)
	return logits.cpu()


def postprocess_mask(mask: np.ndarray, min_region_pixels: int = 16) -> np.ndarray:
	"""对 1/2 两类前景分别做连通域过滤。"""
	out = np.zeros_like(mask, dtype=np.uint8)
	for cls in (1, 2):
		cleaned = remove_small_components((mask == cls).astype(np.uint8), min_region_pixels)
		out[cleaned > 0] = cls
	return out


def infer_batch_to_objects(
	model: torch.nn.Module,
	x: torch.Tensor,
	device: str,
	min_region_pixels: int = 16,
) -> list[dict[str, Any]]:
	"""输入 batch `(B,C,H,W)`，输出每张图像的目标列表。"""
	logits = predict_logits(model, x, device)
	pred = torch.argmax(logits, dim=1).numpy().astype(np.uint8)
	out: list[dict[str, Any]] = []
	for i in range(pred.shape[0]):
		m = postprocess_mask(pred[i], min_region_pixels=min_region_pixels)
		cyc = extract_eddy_objects(m, class_id=1)
		anti = extract_eddy_objects(m, class_id=2)
		out.append(
			{
				"mask": m,
				"objects": cyc + anti,
				"cyclonic_count": len(cyc),
				"anticyclonic_count": len(anti),
			}
		)
	return out


def load_checkpoint(model: torch.nn.Module, ckpt: Path, map_location: str = "cpu") -> dict[str, Any]:
	state = torch.load(ckpt, map_location=map_location)
	model.load_state_dict(state["model_state_dict"])
	return state

