"""风-浪异常检测器：基于重构误差输出异常分数与等级。"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from anomaly_detection.trainer import batch_recon_error


@torch.no_grad()
def infer_errors(
	model: torch.nn.Module,
	loader: torch.utils.data.DataLoader,
	device: torch.device | str,
) -> np.ndarray:
	model.eval()
	all_err: list[np.ndarray] = []
	d = torch.device(device)
	for batch in loader:
		oper_x = batch["oper_x"].to(d).float().nan_to_num(0.0)
		wave_x = batch["wave_x"].to(d).float().nan_to_num(0.0)
		oper_v = batch["oper_valid"].to(d).float()
		wave_v = batch["wave_valid"].to(d).float()

		out = model(oper_x, wave_x)
		errs = batch_recon_error(oper_x, wave_x, out["oper_recon"], out["wave_recon"], oper_v, wave_v)
		all_err.append(errs)
	return np.concatenate(all_err, axis=0) if all_err else np.array([], dtype=np.float32)


def classify_levels(errors: np.ndarray, threshold: float) -> list[str]:
	"""按阈值与倍率划分预警等级。"""

	levels: list[str] = []
	thr = max(1e-9, float(threshold))
	for e in errors.tolist():
		r = float(e) / thr
		if r < 1.0:
			levels.append("normal")
		elif r < 1.5:
			levels.append("warning")
		else:
			levels.append("critical")
	return levels


def build_detection_report(errors: np.ndarray, threshold: float) -> dict[str, Any]:
	flags = (errors >= float(threshold)).astype(np.int64)
	levels = classify_levels(errors, threshold)
	return {
		"threshold": float(threshold),
		"num_samples": int(errors.shape[0]),
		"num_anomaly": int(flags.sum()),
		"anomaly_ratio": float(flags.mean()) if errors.size else 0.0,
		"levels": levels,
		"flags": flags.tolist(),
	}


def associate_events(
	timestamps: list[int | float | None],
	flags: np.ndarray,
	events: list[dict[str, Any]],
) -> dict[str, Any]:
	"""将异常时刻与事件时间窗（如台风）做重叠关联。"""

	if len(timestamps) != int(flags.shape[0]):
		raise ValueError("timestamps and flags length mismatch")

	anom_ts = [int(t) for t, f in zip(timestamps, flags.tolist()) if t is not None and int(f) == 1]
	hits: list[dict[str, Any]] = []
	for ev in events:
		start = int(ev["start"])
		end = int(ev["end"])
		cnt = sum(1 for t in anom_ts if start <= t <= end)
		if cnt > 0:
			hits.append(
				{
					"name": ev.get("name", "event"),
					"start": start,
					"end": end,
					"anomaly_points": int(cnt),
				}
			)

	return {
		"num_events": int(len(events)),
		"num_matched_events": int(len(hits)),
		"matched_events": hits,
	}
