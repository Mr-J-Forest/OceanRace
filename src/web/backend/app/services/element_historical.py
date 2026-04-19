"""要素预报：从 NetCDF 按「约一年前最近时刻」读取历史同期场（物理量纲）。"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import xarray as xr


def load_historical_same_period(
    data_path: str,
    t0: int,
    input_steps: int,
    output_steps: int,
    nc_var_names: tuple[str, ...],
    max_time_delta_hours: float = 72.0,
) -> np.ndarray | None:
    """
    对每个预报时效 k，取验证时刻 time[t0+input_steps+k] 减约一年，
    在整卷 time 轴上找最近观测切片，堆成 (output_steps, C, H, W)。
    找不到足够近的时刻则该步为 NaN。
    """
    try:
        ds = xr.open_dataset(data_path)
    except Exception:
        return None

    try:
        if "time" not in ds.coords and "time" not in ds:
            return None
        times = pd.to_datetime(ds["time"].values)
        n_time = int(len(times))
        if n_time <= 0:
            return None

        need = t0 + input_steps + output_steps
        if need > n_time:
            return None

        first = str(nc_var_names[0])
        if first not in ds:
            return None
        ref0 = np.asarray(ds[first].isel(time=t0 + input_steps).values, dtype=np.float32)
        if ref0.ndim != 2:
            return None
        h, w = int(ref0.shape[0]), int(ref0.shape[1])
        out = np.full((output_steps, len(nc_var_names), h, w), np.nan, dtype=np.float32)
        tx = pd.DatetimeIndex(pd.to_datetime(times))

        for k in range(output_steps):
            ti = t0 + input_steps + k
            if ti < 0 or ti >= n_time:
                continue
            target_ts = pd.Timestamp(times[ti]) - pd.DateOffset(years=1)
            j = int(tx.get_indexer([target_ts], method="nearest")[0])
            dt_h = abs((tx[j] - target_ts).total_seconds()) / 3600.0
            if dt_h > max_time_delta_hours:
                continue
            for ci, vn in enumerate(nc_var_names):
                if vn not in ds:
                    continue
                raw = np.asarray(ds[vn].isel(time=j).values, dtype=np.float32)
                if raw.shape == (h, w):
                    out[k, ci] = raw
        return out
    finally:
        try:
            ds.close()
        except Exception:
            pass


def element_metrics_masked(
    pred: np.ndarray,
    true: np.ndarray,
    mask: np.ndarray | None,
) -> dict[str, Any]:
    """
    pred/true: (T, C, H, W)。mask 与现有 extract_mask 语义一致（可为 4D）。
    返回每变量 MSE/MAE/R2（仅在有效格点展平后计算）。
    """
    t_steps, n_ch, h, w = pred.shape[0], pred.shape[1], pred.shape[2], pred.shape[3]
    per_var: dict[str, dict[str, float]] = {}
    per_step_mse: list[float] = []

    for step in range(t_steps):
        se_sum = 0.0
        ct = 0
        for c in range(min(n_ch, true.shape[1])):
            ps = pred[step, c]
            ts = true[step, c]
            ms = mask
            if ms is not None:
                if ms.ndim == 4:
                    m2 = ms[min(step, ms.shape[0] - 1), min(c, ms.shape[1] - 1)]
                elif ms.ndim == 3:
                    m2 = ms[min(step, ms.shape[0] - 1)]
                elif ms.shape == (h, w):
                    m2 = ms
                else:
                    m2 = None
            else:
                m2 = None
            if m2 is not None and m2.shape == (h, w):
                valid = np.isfinite(ps) & np.isfinite(ts) & (m2 >= 0.5)
            else:
                valid = np.isfinite(ps) & np.isfinite(ts)
            if not np.any(valid):
                continue
            d = ps[valid] - ts[valid]
            se_sum += float(np.sum(d * d))
            ct += int(d.size)
        per_step_mse.append(float(se_sum / max(ct, 1)))

    for c in range(min(n_ch, true.shape[1])):
        pv = pred[:, c].reshape(-1)
        tv = true[:, c].reshape(-1)
        if mask is not None and mask.ndim == 4:
            mv = mask[:, c].reshape(-1)
            valid = np.isfinite(pv) & np.isfinite(tv) & (mv >= 0.5)
        elif mask is not None and mask.ndim == 3 and mask.shape[0] == t_steps:
            mv = np.stack(
                [mask[min(s, mask.shape[0] - 1)] for s in range(t_steps)], axis=0
            ).reshape(-1)
            valid = np.isfinite(pv) & np.isfinite(tv) & (mv >= 0.5)
        else:
            valid = np.isfinite(pv) & np.isfinite(tv)
        if not np.any(valid):
            per_var[str(c)] = {"mse": float("nan"), "mae": float("nan"), "r2": float("nan"), "n": 0}
            continue
        p = pv[valid].astype(np.float64)
        t = tv[valid].astype(np.float64)
        err = p - t
        mse = float(np.mean(err**2))
        mae = float(np.mean(np.abs(err)))
        t_mean = float(np.mean(t))
        ss_tot = float(np.sum((t - t_mean) ** 2))
        ss_res = float(np.sum(err**2))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")
        per_var[str(c)] = {"mse": mse, "mae": mae, "r2": r2, "n": int(valid.sum())}

    return {"per_var_index": per_var, "per_step_mse": per_step_mse}
