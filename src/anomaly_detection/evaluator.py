"""风-浪异常评估：阈值划分与监督指标。"""
from __future__ import annotations

from typing import Any

import numpy as np


def calibrate_threshold(errors: np.ndarray, quantile: float = 0.95) -> float:
	if errors.size == 0:
		raise ValueError("empty errors")
	q = float(np.clip(quantile, 0.5, 0.999))
	return float(np.quantile(errors, q))


def classify_by_threshold(errors: np.ndarray, threshold: float) -> np.ndarray:
	return (errors >= float(threshold)).astype(np.int64)


def summarize_errors(errors: np.ndarray) -> dict[str, float]:
	if errors.size == 0:
		return {"count": 0.0}
	return {
		"count": float(errors.size),
		"mean": float(np.mean(errors)),
		"std": float(np.std(errors)),
		"p50": float(np.quantile(errors, 0.50)),
		"p90": float(np.quantile(errors, 0.90)),
		"p95": float(np.quantile(errors, 0.95)),
		"p99": float(np.quantile(errors, 0.99)),
		"max": float(np.max(errors)),
	}


def evaluate_with_labels(errors: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, Any]:
	"""labels: 0=normal, 1=anomaly。"""

	if errors.shape[0] != labels.shape[0]:
		raise ValueError("errors and labels length mismatch")

	y_pred = classify_by_threshold(errors, threshold)
	y_true = labels.astype(np.int64)

	tp = int(np.sum((y_true == 1) & (y_pred == 1)))
	tn = int(np.sum((y_true == 0) & (y_pred == 0)))
	fp = int(np.sum((y_true == 0) & (y_pred == 1)))
	fn = int(np.sum((y_true == 1) & (y_pred == 0)))

	eps = 1e-9
	acc = (tp + tn) / max(1, len(y_true))
	prec = tp / (tp + fp + eps)
	rec = tp / (tp + fn + eps)
	f1 = (2.0 * prec * rec) / (prec + rec + eps)

	return {
		"accuracy": float(acc),
		"precision": float(prec),
		"recall": float(rec),
		"f1": float(f1),
		"far": float(fp / (fp + tn + eps)),
		"tp": tp,
		"tn": tn,
		"fp": fp,
		"fn": fn,
		"threshold": float(threshold),
	}


def roc_auc_from_scores(labels: np.ndarray, scores: np.ndarray) -> float:
	"""无 sklearn 依赖的二分类 ROC-AUC 计算。"""

	y = labels.astype(np.int64)
	s = scores.astype(np.float64)
	if y.shape[0] != s.shape[0]:
		raise ValueError("labels and scores length mismatch")
	n_pos = int((y == 1).sum())
	n_neg = int((y == 0).sum())
	if n_pos == 0 or n_neg == 0:
		return float("nan")

	order = np.argsort(s)
	ranks = np.empty_like(order, dtype=np.float64)
	ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)
	sum_pos = float(ranks[y == 1].sum())
	auc = (sum_pos - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg)
	return float(auc)
