"""异常检测 inspect：快照网格与统计（供 routers/anomaly 使用）。"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch

from src.anomaly_detection.dataset import AnomalyFrameDataset


def masked_mean_and_p95(field: np.ndarray, valid: np.ndarray) -> tuple[float, float]:
    data = np.asarray(field, dtype=np.float32)
    mask = np.asarray(valid, dtype=np.float32) >= 0.5
    if data.shape != mask.shape:
        mask = np.ones_like(data, dtype=bool)
    selected = data[mask]
    if selected.size == 0:
        selected = data[np.isfinite(data)]
    if selected.size == 0:
        return (0.0, 0.0)
    selected = selected[np.isfinite(selected)]
    if selected.size == 0:
        return (0.0, 0.0)
    return (float(np.mean(selected)), float(np.percentile(selected, 95.0)))


def extract_anomaly_sample_fields(sample: dict[str, Any]) -> dict[str, Any] | None:
    oper_x = sample.get("oper_x")
    wave_x = sample.get("wave_x")
    oper_valid = sample.get("oper_valid")
    wave_valid = sample.get("wave_valid")

    if not (isinstance(oper_x, torch.Tensor) and isinstance(wave_x, torch.Tensor)):
        return None

    wind_grid = oper_x[2].detach().cpu().numpy().astype(np.float32)
    wave_grid = wave_x[0].detach().cpu().numpy().astype(np.float32)

    wind_valid = (
        oper_valid[2].detach().cpu().numpy().astype(np.float32)
        if isinstance(oper_valid, torch.Tensor)
        else np.ones_like(wind_grid, dtype=np.float32)
    )
    wave_valid_arr = (
        wave_valid[0].detach().cpu().numpy().astype(np.float32)
        if isinstance(wave_valid, torch.Tensor)
        else np.ones_like(wave_grid, dtype=np.float32)
    )

    wind_mean, wind_p95 = masked_mean_and_p95(wind_grid, wind_valid)
    wave_mean, wave_p95 = masked_mean_and_p95(wave_grid, wave_valid_arr)

    return {
        "wind_grid": wind_grid,
        "wave_grid": wave_grid,
        "wind_valid": wind_valid,
        "wave_valid": wave_valid_arr,
        "wind_mean": wind_mean,
        "wind_p95": wind_p95,
        "wave_mean": wave_mean,
        "wave_p95": wave_p95,
    }


def get_anomaly_snapshot_dataset(
    *,
    cache: dict[tuple[Any, ...], dict[str, Any]],
    processed_dir: str,
    manifest_path: str,
    split: str,
    norm_stats_path: str | None,
    open_file_lru_size: int,
) -> AnomalyFrameDataset:
    key = (processed_dir, manifest_path, split, norm_stats_path)
    signature = (
        float(os.path.getmtime(manifest_path)) if os.path.exists(manifest_path) else -1.0,
        float(os.path.getmtime(norm_stats_path)) if norm_stats_path and os.path.exists(norm_stats_path) else -1.0,
        int(open_file_lru_size),
    )
    cached = cache.get(key)
    if cached is not None and cached.get("signature") == signature and isinstance(cached.get("dataset"), AnomalyFrameDataset):
        return cached["dataset"]

    if cached is not None and isinstance(cached.get("dataset"), AnomalyFrameDataset):
        try:
            cached["dataset"].close()
        except Exception:
            pass

    ds = AnomalyFrameDataset(
        processed_anomaly_dir=processed_dir,
        split=split,
        manifest_path=manifest_path,
        norm_stats_path=norm_stats_path,
        open_file_lru_size=max(0, int(open_file_lru_size)),
    )
    cache[key] = {"signature": signature, "dataset": ds}
    return ds
