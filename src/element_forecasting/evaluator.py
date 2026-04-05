"""要素预测评估指标。"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
	return torch.mean((pred - target) ** 2)


def rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
	return torch.sqrt(mse(pred, target) + 1e-12)


def mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
	return torch.mean(torch.abs(pred - target))


def nse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
	"""Nash-Sutcliffe Efficiency。"""

	num = torch.sum((pred - target) ** 2)
	denom = torch.sum((target - torch.mean(target)) ** 2) + 1e-12
	return 1.0 - num / denom


def compute_regression_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
	return {
		"mse": float(mse(pred, target).item()),
		"rmse": float(rmse(pred, target).item()),
		"mae": float(mae(pred, target).item()),
		"nse": float(nse(pred, target).item()),
	}


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
	"""
	掩膜版 MSE：仅在 mask==1 的位置计算。

	pred/target/mask 形状需可广播到一致；mask 为 0/1 或任意非负权重都可。
	"""
	p = pred.float()
	t = target.float()
	m = mask.float()
	diff2 = (p - t).pow(2)
	num = torch.sum(diff2 * m)
	den = torch.sum(m).clamp_min(eps)
	return num / den


def masked_nrmse_percent(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
	"""掩膜版 NRMSE 百分比：NRMSE = RMSE / (target_max - target_min) * 100。"""
	p = pred.float()
	t = target.float()
	m = mask.float()
	valid = m > 0
	if not torch.any(valid):
		return torch.tensor(float("inf"), device=p.device)
	rmse_val = masked_rmse(p, t, m, eps=eps)
	t_valid = t[valid]
	t_range = (torch.max(t_valid) - torch.min(t_valid)).clamp_min(eps)
	return (rmse_val / t_range) * 100.0

def masked_weighted_mse(
	pred: torch.Tensor,
	target: torch.Tensor,
	mask: torch.Tensor,
	channel_weights: torch.Tensor | None = None,
	spatial_weights: torch.Tensor | None = None,
	eps: float = 1e-12,
) -> torch.Tensor:
	"""按通道加权的掩膜 MSE。"""
	p = pred.float()
	t = target.float()
	m = mask.float()
	w = torch.ones_like(p) if channel_weights is None else channel_weights.float()
	sw = torch.ones_like(p) if spatial_weights is None else spatial_weights.float()
	diff2 = (p - t).pow(2)
	num = torch.sum(diff2 * m * w * sw)
	den = torch.sum(m * w * sw).clamp_min(eps)
	return num / den


def masked_spatial_mean_mse(
	pred: torch.Tensor,
	target: torch.Tensor,
	mask: torch.Tensor,
	channel_weights: torch.Tensor | None = None,
	eps: float = 1e-12,
) -> torch.Tensor:
	"""空间均值约束：比较每个样本/步长/变量的空间均值误差，并支持通道加权。"""
	p = pred.float()
	t = target.float()
	m = mask.float()

	den_hw = torch.sum(m, dim=(-2, -1), keepdim=True).clamp_min(eps)
	p_mean = torch.sum(p * m, dim=(-2, -1), keepdim=True) / den_hw
	t_mean = torch.sum(t * m, dim=(-2, -1), keepdim=True) / den_hw
	mean_diff2 = (p_mean - t_mean).pow(2)

	valid_bt = (den_hw > eps).float()
	w = torch.ones_like(mean_diff2) if channel_weights is None else channel_weights.float()
	num = torch.sum(mean_diff2 * valid_bt * w)
	den = torch.sum(valid_bt * w).clamp_min(eps)
	return num / den


def masked_rmse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
	return torch.sqrt(masked_mse(pred, target, mask, eps=eps) + eps)


def masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
	p = pred.float()
	t = target.float()
	m = mask.float()
	absdiff = torch.abs(p - t)
	num = torch.sum(absdiff * m)
	den = torch.sum(m).clamp_min(eps)
	return num / den


def masked_nse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
	"""
	掩膜版 NSE：
	mean 取 mask 覆盖的有效点；分子/分母也只在有效点统计。
	"""
	p = pred.float()
	t = target.float()
	m = mask.float()
	den = torch.sum(m).clamp_min(eps)

	mean_t = torch.sum(t * m) / den
	ss_res = torch.sum(((p - t) ** 2) * m)
	ss_tot = torch.sum(((t - mean_t) ** 2) * m) + eps
	return 1.0 - ss_res / ss_tot


def _spatial_gradients(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
	"""返回张量在 H/W 两个方向的一阶差分，输出形状与输入一致。"""
	dh = x[..., 1:, :] - x[..., :-1, :]
	dw = x[..., :, 1:] - x[..., :, :-1]
	dh = F.pad(dh, (0, 0, 0, 1))
	dw = F.pad(dw, (0, 1, 0, 0))
	return dh, dw


def _gradient_magnitude(x: torch.Tensor) -> torch.Tensor:
	dh, dw = _spatial_gradients(x)
	return torch.sqrt(dh.pow(2) + dw.pow(2) + 1e-12)


def masked_gradient_l1(
	pred: torch.Tensor,
	target: torch.Tensor,
	mask: torch.Tensor,
	eps: float = 1e-12,
) -> torch.Tensor:
	"""梯度一致性损失：比较预测与目标的空间梯度（L1）。"""
	p = pred.float()
	t = target.float()
	m = mask.float()
	p_dh, p_dw = _spatial_gradients(p)
	t_dh, t_dw = _spatial_gradients(t)
	grad_l1 = torch.abs(p_dh - t_dh) + torch.abs(p_dw - t_dw)
	num = torch.sum(grad_l1 * m)
	den = torch.sum(m).clamp_min(eps)
	return num / den


def masked_edge_l1(
	pred: torch.Tensor,
	target: torch.Tensor,
	mask: torch.Tensor,
	edge_type: str = "sobel",
	eps: float = 1e-12,
) -> torch.Tensor:
	"""可选边缘一致性损失（sobel/laplacian）。"""
	p = pred.float()
	t = target.float()
	m = mask.float()

	edge_type = str(edge_type).strip().lower()
	if edge_type == "laplacian":
		kernel = torch.tensor(
			[[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
			dtype=p.dtype,
			device=p.device,
		)
		kernel = kernel.view(1, 1, 3, 3)
		p2 = p.flatten(0, 2).unsqueeze(1)
		t2 = t.flatten(0, 2).unsqueeze(1)
		p_edge = F.conv2d(p2, kernel, padding=1).squeeze(1).view_as(p)
		t_edge = F.conv2d(t2, kernel, padding=1).squeeze(1).view_as(t)
		absdiff = torch.abs(p_edge - t_edge)
	else:
		sobel_x = torch.tensor(
			[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
			dtype=p.dtype,
			device=p.device,
		).view(1, 1, 3, 3)
		sobel_y = torch.tensor(
			[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
			dtype=p.dtype,
			device=p.device,
		).view(1, 1, 3, 3)
		p2 = p.flatten(0, 2).unsqueeze(1)
		t2 = t.flatten(0, 2).unsqueeze(1)
		p_gx = F.conv2d(p2, sobel_x, padding=1)
		p_gy = F.conv2d(p2, sobel_y, padding=1)
		t_gx = F.conv2d(t2, sobel_x, padding=1)
		t_gy = F.conv2d(t2, sobel_y, padding=1)
		p_edge = torch.sqrt(p_gx.pow(2) + p_gy.pow(2) + 1e-12).squeeze(1).view_as(p)
		t_edge = torch.sqrt(t_gx.pow(2) + t_gy.pow(2) + 1e-12).squeeze(1).view_as(t)
		absdiff = torch.abs(p_edge - t_edge)

	num = torch.sum(absdiff * m)
	den = torch.sum(m).clamp_min(eps)
	return num / den


def build_online_region_weights(
	target: torch.Tensor,
	mask: torch.Tensor,
	base_weight: float = 1.0,
	strength: float = 1.0,
	quantile: float = 0.8,
) -> torch.Tensor:
	"""基于目标场梯度在线生成区域权重，强调锋面等高梯度区域。"""
	t = target.float()
	m = mask.float()
	g = _gradient_magnitude(t)
	valid = m > 0
	if torch.any(valid):
		q = float(max(0.0, min(1.0, quantile)))
		thr = torch.quantile(g[valid], q)
		focus = (g >= thr).float()
	else:
		focus = torch.zeros_like(g)
	weights = float(base_weight) + float(strength) * focus
	return weights * m


def masked_gradient_rmse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
	p = pred.float()
	t = target.float()
	m = mask.float()
	p_grad = _gradient_magnitude(p)
	t_grad = _gradient_magnitude(t)
	diff2 = (p_grad - t_grad).pow(2)
	num = torch.sum(diff2 * m)
	den = torch.sum(m).clamp_min(eps)
	return torch.sqrt(num / den + eps)


def masked_local_extreme_error(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
	"""局地极值误差：按每个(B,T,C)比较有效区域内 max/min 偏差。"""
	p = pred.float()
	t = target.float()
	m = mask.float()
	valid = m > 0

	neg_inf = torch.full_like(p, float("-inf"))
	pos_inf = torch.full_like(p, float("inf"))
	p_max = torch.where(valid, p, neg_inf).amax(dim=(-2, -1))
	t_max = torch.where(valid, t, neg_inf).amax(dim=(-2, -1))
	p_min = torch.where(valid, p, pos_inf).amin(dim=(-2, -1))
	t_min = torch.where(valid, t, pos_inf).amin(dim=(-2, -1))

	finite = torch.isfinite(p_max) & torch.isfinite(t_max) & torch.isfinite(p_min) & torch.isfinite(t_min)
	err = (torch.abs(p_max - t_max) + torch.abs(p_min - t_min)) * finite.float()
	den = finite.float().sum().clamp_min(eps)
	return err.sum() / den


def masked_edge_region_rmse(
	pred: torch.Tensor,
	target: torch.Tensor,
	mask: torch.Tensor,
	quantile: float = 0.8,
	eps: float = 1e-12,
) -> torch.Tensor:
	"""高梯度区域 RMSE（以目标梯度分位数选边缘区域）。"""
	p = pred.float()
	t = target.float()
	m = mask.float()
	g = _gradient_magnitude(t)
	valid = m > 0
	if torch.any(valid):
		q = float(max(0.0, min(1.0, quantile)))
		thr = torch.quantile(g[valid], q)
		edge_mask = (g >= thr).float() * m
	else:
		edge_mask = torch.zeros_like(m)
	return torch.sqrt(masked_mse(p, t, edge_mask, eps=eps) + eps)


def compute_regression_metrics_masked(
	pred: torch.Tensor,
	target: torch.Tensor,
	mask: torch.Tensor,
	edge_quantile: float = 0.8,
) -> dict[str, float]:
	return {
		"mse": float(masked_mse(pred, target, mask).item()),
		"rmse": float(masked_rmse(pred, target, mask).item()),
		"nrmse_percent": float(masked_nrmse_percent(pred, target, mask).item()),
		"mae": float(masked_mae(pred, target, mask).item()),
		"nse": float(masked_nse(pred, target, mask).item()),
		"grad_rmse": float(masked_gradient_rmse(pred, target, mask).item()),
		"extreme_error": float(masked_local_extreme_error(pred, target, mask).item()),
		"edge_rmse": float(masked_edge_region_rmse(pred, target, mask, quantile=edge_quantile).item()),
	}
