"""风-浪异常训练器：双分支 AE + 重构误差阈值标定。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from anomaly_detection.evaluator import calibrate_threshold


def masked_mse(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
	diff = (pred - target) ** 2
	w = valid.float()
	num = (diff * w).sum()
	den = w.sum() + eps
	return num / den


@torch.no_grad()
def batch_recon_error(
	oper_x: torch.Tensor,
	wave_x: torch.Tensor,
	oper_recon: torch.Tensor,
	wave_recon: torch.Tensor,
	oper_valid: torch.Tensor,
	wave_valid: torch.Tensor,
) -> np.ndarray:
	"""按样本输出重构误差，用于阈值与检测。"""

	b = oper_x.shape[0]
	out = np.zeros(b, dtype=np.float32)
	for i in range(b):
		o = masked_mse(oper_recon[i : i + 1], oper_x[i : i + 1], oper_valid[i : i + 1]).item()
		w = masked_mse(wave_recon[i : i + 1], wave_x[i : i + 1], wave_valid[i : i + 1]).item()
		out[i] = float(0.5 * o + 0.5 * w)
	return out


def train_one_epoch(
	model: nn.Module,
	loader: torch.utils.data.DataLoader,
	optimizer: torch.optim.Optimizer,
	device: torch.device | str,
) -> dict[str, float]:
	model.train()
	loss_sum = 0.0
	cnt = 0
	for batch in loader:
		oper_x = batch["oper_x"].to(device).float().nan_to_num(0.0)
		wave_x = batch["wave_x"].to(device).float().nan_to_num(0.0)
		oper_v = batch["oper_valid"].to(device).float()
		wave_v = batch["wave_valid"].to(device).float()

		optimizer.zero_grad(set_to_none=True)
		out = model(oper_x, wave_x)
		loss_oper = masked_mse(out["oper_recon"], oper_x, oper_v)
		loss_wave = masked_mse(out["wave_recon"], wave_x, wave_v)
		loss = 0.5 * loss_oper + 0.5 * loss_wave
		if not torch.isfinite(loss):
			continue
		loss.backward()
		optimizer.step()

		loss_sum += float(loss.item())
		cnt += 1
	return {"loss": loss_sum / max(1, cnt)}


@torch.no_grad()
def validate_one_epoch(
	model: nn.Module,
	loader: torch.utils.data.DataLoader,
	device: torch.device | str,
) -> dict[str, Any]:
	model.eval()
	loss_sum = 0.0
	cnt = 0
	all_err: list[np.ndarray] = []

	for batch in loader:
		oper_x = batch["oper_x"].to(device).float().nan_to_num(0.0)
		wave_x = batch["wave_x"].to(device).float().nan_to_num(0.0)
		oper_v = batch["oper_valid"].to(device).float()
		wave_v = batch["wave_valid"].to(device).float()

		out = model(oper_x, wave_x)
		loss_oper = masked_mse(out["oper_recon"], oper_x, oper_v)
		loss_wave = masked_mse(out["wave_recon"], wave_x, wave_v)
		loss = 0.5 * loss_oper + 0.5 * loss_wave
		if not torch.isfinite(loss):
			continue

		errs = batch_recon_error(oper_x, wave_x, out["oper_recon"], out["wave_recon"], oper_v, wave_v)
		all_err.append(errs)
		loss_sum += float(loss.item())
		cnt += 1

	errors = np.concatenate(all_err, axis=0) if all_err else np.array([], dtype=np.float32)
	return {
		"loss": loss_sum / max(1, cnt),
		"errors": errors,
	}


@dataclass
class AnomalyTrainConfig:
	lr: float = 1e-3
	epochs: int = 10
	batch_size: int = 8
	num_workers: int = 0
	device: str = "cuda" if torch.cuda.is_available() else "cpu"
	threshold_quantile: float = 0.95
	output_dir: Path = Path("outputs/anomaly_detection")
	save_name: str = "anomaly_ae_best.pt"


def fit(
	model: nn.Module,
	train_loader: torch.utils.data.DataLoader,
	val_loader: torch.utils.data.DataLoader,
	cfg: AnomalyTrainConfig,
) -> dict[str, Any]:
	device = torch.device(cfg.device)
	model = model.to(device)
	optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

	cfg.output_dir.mkdir(parents=True, exist_ok=True)
	best_path = cfg.output_dir / cfg.save_name

	best_val = float("inf")
	best_threshold = 0.0
	history: list[dict[str, float]] = []

	for epoch in range(1, cfg.epochs + 1):
		tr = train_one_epoch(model, train_loader, optimizer, device)
		va = validate_one_epoch(model, val_loader, device)
		threshold = calibrate_threshold(va["errors"], cfg.threshold_quantile) if va["errors"].size else 0.0
		row = {
			"epoch": float(epoch),
			"train_loss": float(tr["loss"]),
			"val_loss": float(va["loss"]),
			"threshold": float(threshold),
		}
		history.append(row)

		if va["loss"] < best_val:
			best_val = float(va["loss"])
			best_threshold = float(threshold)
			torch.save(
				{
					"epoch": epoch,
					"model_state_dict": model.state_dict(),
					"optimizer_state_dict": optimizer.state_dict(),
					"val_loss": best_val,
					"threshold": best_threshold,
				},
				best_path,
			)

	return {
		"history": history,
		"best_val_loss": float(best_val),
		"best_threshold": float(best_threshold),
		"checkpoint": str(best_path),
	}
