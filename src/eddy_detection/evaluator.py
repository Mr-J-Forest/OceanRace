"""涡旋分割评估指标。"""
from __future__ import annotations

import torch


@torch.no_grad()
def confusion_matrix(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 3) -> torch.Tensor:
	"""计算混淆矩阵（C x C）。"""
	p = pred.view(-1).to(torch.int64)
	t = target.view(-1).to(torch.int64)
	valid = (t >= 0) & (t < num_classes)
	p = p[valid]
	t = t[valid]
	idx = t * num_classes + p
	cm = torch.bincount(idx, minlength=num_classes * num_classes)
	return cm.reshape(num_classes, num_classes)


def _safe_div(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
	return x / torch.clamp(y, min=1e-8)


def _binary_eddy_metrics(cm: torch.Tensor) -> dict[str, float]:
	"""将 1/2 类合并为 eddy，计算二分类指标。"""
	t = cm.to(torch.float32)

	# 正类: eddy(1/2), 负类: background(0)
	tp = t[1:, 1:].sum()
	fp = t[0, 1:].sum()
	fn = t[1:, 0].sum()
	tn = t[0, 0]

	precision = _safe_div(tp, tp + fp)
	recall = _safe_div(tp, tp + fn)
	f1 = _safe_div(2.0 * precision * recall, precision + recall)
	iou = _safe_div(tp, tp + fp + fn)
	acc = _safe_div(tp + tn, tp + tn + fp + fn)

	return {
		"eddy_precision": float(precision.item()),
		"eddy_recall": float(recall.item()),
		"eddy_f1": float(f1.item()),
		"eddy_iou": float(iou.item()),
		"eddy_acc": float(acc.item()),
	}


def segmentation_metrics(cm: torch.Tensor) -> dict[str, float]:
	"""从混淆矩阵计算像素级指标。"""
	tp = torch.diag(cm).float()
	fp = cm.sum(dim=0).float() - tp
	fn = cm.sum(dim=1).float() - tp
	tn = cm.sum().float() - tp - fp - fn

	precision = _safe_div(tp, tp + fp)
	recall = _safe_div(tp, tp + fn)
	f1 = _safe_div(2 * precision * recall, precision + recall)
	iou = _safe_div(tp, tp + fp + fn)
	acc = _safe_div(tp + tn, tp + tn + fp + fn)

	return {
		"pixel_acc": float(_safe_div(tp.sum(), cm.sum().float()).item()),
		"macro_precision": float(precision.mean().item()),
		"macro_recall": float(recall.mean().item()),
		"macro_f1": float(f1.mean().item()),
		"miou": float(iou.mean().item()),
		"macro_acc": float(acc.mean().item()),
		"f1_bg": float(f1[0].item()) if f1.numel() > 0 else 0.0,
		"f1_cyclonic": float(f1[1].item()) if f1.numel() > 1 else 0.0,
		"f1_anticyclonic": float(f1[2].item()) if f1.numel() > 2 else 0.0,
		**_binary_eddy_metrics(cm),
	}

