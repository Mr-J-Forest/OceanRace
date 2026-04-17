from __future__ import annotations

import torch
import torch.nn.functional as F


def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
	p = pred.float()
	t = target.float()
	m = mask.float()
	num = torch.sum(((p - t) ** 2) * m)
	den = torch.sum(m).clamp_min(eps)
	return num / den


def _grad_x(field: torch.Tensor) -> torch.Tensor:
	g = field[..., 1:] - field[..., :-1]
	return F.pad(g, (0, 1, 0, 0), mode="replicate")


def _grad_y(field: torch.Tensor) -> torch.Tensor:
	g = field[..., 1:, :] - field[..., :-1, :]
	return F.pad(g, (0, 0, 0, 1), mode="replicate")


def warp_scalar_with_flow(
	scalar: torch.Tensor,
	u_flow: torch.Tensor,
	v_flow: torch.Tensor,
	padding_mode: str = "border",
) -> torch.Tensor:
	if scalar.dim() != 3:
		raise ValueError(f"expected scalar shape (B,H,W), got {tuple(scalar.shape)}")
	if u_flow.shape != scalar.shape or v_flow.shape != scalar.shape:
		raise ValueError("u_flow/v_flow shape must match scalar shape")
	bsz, hgt, wid = scalar.shape
	dtype = scalar.dtype
	device = scalar.device
	yy = torch.linspace(-1.0, 1.0, steps=hgt, device=device, dtype=dtype)
	xx = torch.linspace(-1.0, 1.0, steps=wid, device=device, dtype=dtype)
	grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
	base_grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).expand(bsz, hgt, wid, 2)
	norm_u = u_flow * (2.0 / max(1, wid - 1))
	norm_v = v_flow * (2.0 / max(1, hgt - 1))
	flow_grid = torch.stack((norm_u, norm_v), dim=-1)
	sample_grid = base_grid + flow_grid
	out = F.grid_sample(
		scalar.unsqueeze(1),
		sample_grid,
		mode="bilinear",
		padding_mode=padding_mode,
		align_corners=True,
	)
	return out[:, 0]


def velocity_scalar_composite_loss(
	*,
	pred: torch.Tensor,
	target: torch.Tensor,
	mask: torch.Tensor,
	input_last_frame: torch.Tensor,
	var_to_idx: dict[str, int],
	lambda_div: float = 0.1,
	lambda_smooth: float = 0.2,
	lambda_mag: float = 0.1,
	lambda_anom: float = 0.5,
	lambda_adv: float = 1.0,
	lambda_grad: float = 0.2,
	div_target_zero: bool = False,
	eps: float = 1e-12,
) -> tuple[torch.Tensor, dict[str, float]]:
	def _idx(name: str) -> int | None:
		i = var_to_idx.get(name, None)
		return int(i) if i is not None else None

	i_sst = _idx("sst")
	i_sss = _idx("sss")
	i_ssu = _idx("ssu")
	i_ssv = _idx("ssv")
	loss_zero = pred.new_zeros((), dtype=torch.float32)

	if i_ssu is None or i_ssv is None:
		l_uv = loss_zero
		log_uv = {"l_uv_mse": 0.0, "l_div": 0.0, "l_smooth": 0.0, "l_mag": 0.0}
	else:
		u_pred = pred[:, :, i_ssu]
		v_pred = pred[:, :, i_ssv]
		u_gt = target[:, :, i_ssu]
		v_gt = target[:, :, i_ssv]
		m_u = mask[:, :, i_ssu]
		m_v = mask[:, :, i_ssv]
		m_uv = torch.minimum(m_u, m_v)

		l_uv_mse = _masked_mse(u_pred, u_gt, m_u, eps=eps) + _masked_mse(v_pred, v_gt, m_v, eps=eps)

		du_dx_pred = _grad_x(u_pred)
		du_dy_pred = _grad_y(u_pred)
		dv_dx_pred = _grad_x(v_pred)
		dv_dy_pred = _grad_y(v_pred)
		du_dx_gt = _grad_x(u_gt)
		du_dy_gt = _grad_y(u_gt)
		dv_dx_gt = _grad_x(v_gt)
		dv_dy_gt = _grad_y(v_gt)

		div_pred = du_dx_pred + dv_dy_pred
		div_gt = torch.zeros_like(div_pred) if div_target_zero else (du_dx_gt + dv_dy_gt)
		l_div = _masked_mse(div_pred, div_gt, m_uv, eps=eps)

		l_smooth = (
			_masked_mse(du_dx_pred, du_dx_gt, m_u, eps=eps)
			+ _masked_mse(du_dy_pred, du_dy_gt, m_u, eps=eps)
			+ _masked_mse(dv_dx_pred, dv_dx_gt, m_v, eps=eps)
			+ _masked_mse(dv_dy_pred, dv_dy_gt, m_v, eps=eps)
		)

		mag_pred = torch.sqrt(torch.clamp(u_pred ** 2 + v_pred ** 2, min=eps))
		mag_gt = torch.sqrt(torch.clamp(u_gt ** 2 + v_gt ** 2, min=eps))
		l_mag = _masked_mse(mag_pred, mag_gt, m_uv, eps=eps)

		l_uv = l_uv_mse + float(lambda_div) * l_div + float(lambda_smooth) * l_smooth + float(lambda_mag) * l_mag
		log_uv = {
			"l_uv_mse": float(l_uv_mse.item()),
			"l_div": float(l_div.item()),
			"l_smooth": float(l_smooth.item()),
			"l_mag": float(l_mag.item()),
		}

	def _scalar_loss(idx_scalar: int | None, idx_name: str) -> tuple[torch.Tensor, dict[str, float]]:
		if idx_scalar is None:
			return loss_zero, {"l_mse": 0.0, "l_anom": 0.0, "l_adv": 0.0, "l_grad": 0.0}

		s_pred = pred[:, :, idx_scalar]
		s_gt = target[:, :, idx_scalar]
		s_mask = mask[:, :, idx_scalar]
		l_mse = _masked_mse(s_pred, s_gt, s_mask, eps=eps)

		den = torch.sum(s_mask, dim=(-2, -1), keepdim=True).clamp_min(eps)
		mu_gt = torch.sum(s_gt * s_mask, dim=(-2, -1), keepdim=True) / den
		s_pred_anom = s_pred - mu_gt
		s_gt_anom = s_gt - mu_gt
		l_anom = _masked_mse(s_pred_anom, s_gt_anom, s_mask, eps=eps)

		if i_ssu is None or i_ssv is None:
			l_adv = loss_zero
		else:
			u_pred = pred[:, :, i_ssu]
			v_pred = pred[:, :, i_ssv]
			s_prev = input_last_frame[:, idx_scalar]
			l_adv_total = pred.new_zeros((), dtype=torch.float32)
			for t in range(s_pred.shape[1]):
				adv = warp_scalar_with_flow(s_prev, u_pred[:, t], v_pred[:, t])
				l_adv_total = l_adv_total + _masked_mse(adv, s_gt[:, t], s_mask[:, t], eps=eps)
				s_prev = s_pred[:, t]
			l_adv = l_adv_total / max(1, int(s_pred.shape[1]))

		s_dx_pred = _grad_x(s_pred)
		s_dy_pred = _grad_y(s_pred)
		s_dx_gt = _grad_x(s_gt)
		s_dy_gt = _grad_y(s_gt)
		l_grad = _masked_mse(s_dx_pred, s_dx_gt, s_mask, eps=eps) + _masked_mse(s_dy_pred, s_dy_gt, s_mask, eps=eps)

		l_scalar = l_mse + float(lambda_anom) * l_anom + float(lambda_adv) * l_adv + float(lambda_grad) * l_grad
		return l_scalar, {
			f"{idx_name}_l_mse": float(l_mse.item()),
			f"{idx_name}_l_anom": float(l_anom.item()),
			f"{idx_name}_l_adv": float(l_adv.item()),
			f"{idx_name}_l_grad": float(l_grad.item()),
		}

	l_sst, log_sst = _scalar_loss(i_sst, "sst")
	l_sss, log_sss = _scalar_loss(i_sss, "sss")
	l_total = l_sst + l_sss + 2.0 * l_uv
	logs: dict[str, float] = {"l_total_phys": float(l_total.item()), "l_uv": float(l_uv.item()), "l_sst": float(l_sst.item()), "l_sss": float(l_sss.item())}
	logs.update(log_uv)
	logs.update(log_sst)
	logs.update(log_sss)
	return l_total, logs
