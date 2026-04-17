"""中尺度涡旋分割训练器。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from eddy_detection.evaluator import confusion_matrix, segmentation_metrics
from utils.logger import get_logger, tqdm, tqdm_logging

_log = get_logger(__name__)


@dataclass
class EddyTrainConfig:
	epochs: int = 10
	batch_size: int = 4
	lr: float = 1e-3
	num_workers: int = 0
	weight_decay: float = 1e-4
	dice_weight: float = 0.5
	ce_weight: float = 0.5
	boundary_weight: float = 0.0
	log_interval: int = 20
	device: str = "cpu"
	batch_sleep_ms: int = 0


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
	x = torch.stack([b["x"] for b in batch], dim=0)
	y = torch.stack([b["y"] for b in batch], dim=0)
	return {"x": x, "y": y, "meta": batch}


def _dice_loss(logits: torch.Tensor, target: torch.Tensor, num_classes: int = 3) -> torch.Tensor:
	probs = torch.softmax(logits, dim=1)
	target_oh = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
	dims = (0, 2, 3)
	inter = torch.sum(probs * target_oh, dim=dims)
	den = torch.sum(probs + target_oh, dim=dims)
	dice = (2.0 * inter + 1.0) / (den + 1.0)
	return 1.0 - dice.mean()


def _target_boundary_map(target: torch.Tensor) -> torch.Tensor:
	"""从离散标签生成 4 邻域边界图，shape=(B,1,H,W)。"""
	b, h, w = target.shape
	edge = torch.zeros((b, h, w), dtype=torch.float32, device=target.device)
	edge[:, 1:, :] = torch.maximum(edge[:, 1:, :], (target[:, 1:, :] != target[:, :-1, :]).float())
	edge[:, :-1, :] = torch.maximum(edge[:, :-1, :], (target[:, 1:, :] != target[:, :-1, :]).float())
	edge[:, :, 1:] = torch.maximum(edge[:, :, 1:], (target[:, :, 1:] != target[:, :, :-1]).float())
	edge[:, :, :-1] = torch.maximum(edge[:, :, :-1], (target[:, :, 1:] != target[:, :, :-1]).float())
	return edge.unsqueeze(1)


def _pred_boundary_strength(logits: torch.Tensor) -> torch.Tensor:
	"""由前景概率估计边界强度，shape=(B,1,H,W)。"""
	probs = torch.softmax(logits, dim=1)
	fg = probs[:, 1:, :, :].sum(dim=1, keepdim=True)

	kx = torch.tensor(
		[[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
		dtype=fg.dtype,
		device=fg.device,
	).unsqueeze(1)
	ky = torch.tensor(
		[[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
		dtype=fg.dtype,
		device=fg.device,
	).unsqueeze(1)
	gx = F.conv2d(fg, kx, padding=1)
	gy = F.conv2d(fg, ky, padding=1)
	mag = torch.sqrt(gx * gx + gy * gy + 1e-6)
	den = torch.amax(mag, dim=(2, 3), keepdim=True).clamp_min(1e-6)
	return mag / den


def _run_epoch(
	model: torch.nn.Module,
	loader: DataLoader,
	device: torch.device,
	optimizer: torch.optim.Optimizer | None,
	cfg: EddyTrainConfig,
	*,
	phase: str,
	epoch: int,
	total_epochs: int,
) -> dict[str, float]:
	is_train = optimizer is not None
	model.train(is_train)

	total_loss = 0.0
	total_n = 0
	cm = torch.zeros((3, 3), dtype=torch.int64)
	t0 = time.time()
	n_batches = max(1, len(loader))

	iter_loader = tqdm(
		loader,
		total=n_batches,
		desc=f"epoch {epoch}/{total_epochs} {phase}",
		unit="batch",
		leave=True,
	)
	for bi, batch in enumerate(iter_loader, start=1):
		x = batch["x"].to(device)
		y = batch["y"].to(device)
		logits = model(x)
		ce = F.cross_entropy(logits, y)
		dice = _dice_loss(logits, y, num_classes=3)
		loss = cfg.ce_weight * ce + cfg.dice_weight * dice
		if cfg.boundary_weight > 0.0:
			target_edge = _target_boundary_map(y)
			pred_edge = _pred_boundary_strength(logits)
			bd = F.l1_loss(pred_edge, target_edge)
			loss = loss + cfg.boundary_weight * bd

		if is_train:
			optimizer.zero_grad(set_to_none=True)
			loss.backward()
			optimizer.step()

		pred = torch.argmax(logits, dim=1)
		cm += confusion_matrix(pred.cpu(), y.cpu(), num_classes=3)
		total_loss += float(loss.item()) * x.size(0)
		total_n += int(x.size(0))
		iter_loader.set_postfix(loss=f"{float(loss.item()):.4f}")

		if bi % max(1, cfg.log_interval) == 0 or bi == n_batches:
			elapsed = max(1e-6, time.time() - t0)
			avg_loss = total_loss / max(total_n, 1)
			pct = 100.0 * bi / n_batches
			batches_per_sec = bi / elapsed
			remain_batches = max(0, n_batches - bi)
			eta_sec = remain_batches / max(1e-6, batches_per_sec)
			_log.info(
				"epoch=%s/%s phase=%s batch=%s/%s (%.1f%%) avg_loss=%.4f elapsed=%.1fs eta=%.1fs",
				epoch,
				total_epochs,
				phase,
				bi,
				n_batches,
				pct,
				avg_loss,
				elapsed,
				eta_sec,
			)

		if is_train and cfg.batch_sleep_ms > 0:
			time.sleep(cfg.batch_sleep_ms / 1000.0)

	out = segmentation_metrics(cm)
	out["loss"] = total_loss / max(total_n, 1)
	return out


def train_eddy_segmentation(
	model: torch.nn.Module,
	train_ds: torch.utils.data.Dataset,
	val_ds: torch.utils.data.Dataset,
	cfg: EddyTrainConfig,
	output_dir: Path,
) -> Path:
	"""训练并返回 best checkpoint 路径。"""
	output_dir.mkdir(parents=True, exist_ok=True)
	ckpt_dir = output_dir / "checkpoints"
	ckpt_dir.mkdir(parents=True, exist_ok=True)

	device = torch.device(cfg.device)
	model = model.to(device)

	tr_loader = DataLoader(
		train_ds,
		batch_size=cfg.batch_size,
		shuffle=True,
		num_workers=cfg.num_workers,
		collate_fn=_collate,
	)
	va_loader = DataLoader(
		val_ds,
		batch_size=cfg.batch_size,
		shuffle=False,
		num_workers=cfg.num_workers,
		collate_fn=_collate,
	)

	opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
	best_key = -1.0
	best_path = ckpt_dir / "best.pt"

	for epoch in range(cfg.epochs):
		with tqdm_logging():
			train_m = _run_epoch(
				model,
				tr_loader,
				device,
				opt,
				cfg,
				phase="train",
				epoch=epoch + 1,
				total_epochs=cfg.epochs,
			)
			val_m = _run_epoch(
				model,
				va_loader,
				device,
				None,
				cfg,
				phase="val",
				epoch=epoch + 1,
				total_epochs=cfg.epochs,
			)

		_log.info(
			"epoch=%s train_loss=%.4f val_loss=%.4f val_mIoU=%.4f val_eddyIoU=%.4f val_f1=%.4f",
			epoch + 1,
			train_m["loss"],
			val_m["loss"],
			val_m["miou"],
			val_m.get("eddy_iou", 0.0),
			val_m["macro_f1"],
		)

		# 目标导向：仅看涡旋（气旋+反气旋合并）的 IoU 选优。
		score = float(val_m.get("eddy_iou", val_m["miou"]))
		state = {
			"epoch": epoch + 1,
			"model_state_dict": model.state_dict(),
			"optimizer_state_dict": opt.state_dict(),
			"train_metrics": train_m,
			"val_metrics": val_m,
			"config": cfg.__dict__,
		}
		torch.save(state, ckpt_dir / "last.pt")
		if score > best_key:
			best_key = score
			torch.save(state, best_path)

	_log.info("best checkpoint: %s", best_path)
	return best_path

