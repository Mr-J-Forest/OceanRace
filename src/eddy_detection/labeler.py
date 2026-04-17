"""涡旋伪标签生成（OW + 涡度符号）。

标签编码：0=背景，1=气旋，2=反气旋。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import tempfile

import numpy as np
import xarray as xr

from data_preprocessing.io import open_nc
from eddy_detection.postprocess import remove_small_components


@dataclass
class EddyLabelConfig:
    """伪标签参数。"""

    ow_std_scale: float = 1.0
    threshold_mode: str = "global"
    min_region_pixels: int = 16
    chunk_size: int = 64
    gradient_mode: str = "metric"
    polarity_mode: str = "by_lat"


_EARTH_RADIUS_M = 6_371_000.0


def _ow_and_vorticity_index(ugos: np.ndarray, vgos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """索引坐标梯度（兼容旧逻辑，物理量纲不严格）。"""
    du_dx = np.gradient(ugos, axis=2)
    du_dy = np.gradient(ugos, axis=1)
    dv_dx = np.gradient(vgos, axis=2)
    dv_dy = np.gradient(vgos, axis=1)

    sn = du_dx - dv_dy
    ss = dv_dx + du_dy
    zeta = dv_dx - du_dy
    ow = sn * sn + ss * ss - zeta * zeta
    return ow, zeta


def _ow_and_vorticity_metric(
    ugos: np.ndarray,
    vgos: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """基于经纬度物理距离计算梯度（单位: m）。"""
    lat_rad = np.deg2rad(np.asarray(latitude, dtype=np.float64))
    lon_rad = np.deg2rad(np.asarray(longitude, dtype=np.float64))

    # d/dy = (1/R) * d/d(lat_rad)
    du_dlat = np.gradient(ugos, lat_rad, axis=1)
    dv_dlat = np.gradient(vgos, lat_rad, axis=1)
    du_dy = du_dlat / _EARTH_RADIUS_M
    dv_dy = dv_dlat / _EARTH_RADIUS_M

    # d/dx = 1/(R*cos(lat)) * d/d(lon_rad)
    du_dlon = np.gradient(ugos, lon_rad, axis=2)
    dv_dlon = np.gradient(vgos, lon_rad, axis=2)
    cos_lat = np.cos(lat_rad)
    cos_lat = np.where(np.abs(cos_lat) < 1e-6, 1e-6, cos_lat)
    meter_per_lon_rad = (_EARTH_RADIUS_M * cos_lat)[None, :, None]
    du_dx = du_dlon / meter_per_lon_rad
    dv_dx = dv_dlon / meter_per_lon_rad

    sn = du_dx - dv_dy
    ss = dv_dx + du_dy
    zeta = dv_dx - du_dy
    ow = sn * sn + ss * ss - zeta * zeta
    return ow, zeta


def _ow_and_vorticity(
    ugos: np.ndarray,
    vgos: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    gradient_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    mode = str(gradient_mode).strip().lower()
    if mode == "index":
        return _ow_and_vorticity_index(ugos, vgos)
    return _ow_and_vorticity_metric(ugos, vgos, latitude, longitude)


def _cyclonic_mask(zeta_t: np.ndarray, latitude: np.ndarray, polarity_mode: str) -> np.ndarray:
    mode = str(polarity_mode).strip().lower()
    if mode == "zeta_sign":
        return zeta_t > 0

    lat = np.asarray(latitude, dtype=np.float64)
    lat2d = lat[:, None]
    north = np.logical_and(lat2d >= 0.0, zeta_t > 0)
    south = np.logical_and(lat2d < 0.0, zeta_t < 0)
    return np.logical_or(north, south)


def build_eddy_mask(
    ugos: np.ndarray,
    vgos: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    cfg: EddyLabelConfig,
) -> np.ndarray:
    """根据 OW 与涡度构建三分类标签。"""
    ow, zeta = _ow_and_vorticity(
        ugos,
        vgos,
        latitude,
        longitude,
        gradient_mode=cfg.gradient_mode,
    )
    mode = str(cfg.threshold_mode).strip().lower()

    if mode == "daily":
        core = np.zeros_like(ow, dtype=np.uint8)
        for t in range(ow.shape[0]):
            sigma_t = float(np.nanstd(ow[t]))
            if not np.isfinite(sigma_t) or sigma_t <= 1e-12:
                sigma_t = 1.0
            core[t] = (ow[t] < (-cfg.ow_std_scale * sigma_t)).astype(np.uint8)
    else:
        sigma = float(np.nanstd(ow))
        if not np.isfinite(sigma) or sigma <= 1e-12:
            sigma = 1.0
        core = (ow < (-cfg.ow_std_scale * sigma)).astype(np.uint8)

    labels = np.zeros_like(core, dtype=np.uint8)

    for t in range(core.shape[0]):
        core_t = remove_small_components(core[t], cfg.min_region_pixels)
        cyc = np.logical_and(core_t > 0, _cyclonic_mask(zeta[t], latitude, cfg.polarity_mode))
        anti = np.logical_and(core_t > 0, zeta[t] <= 0)
        labels[t][cyc] = 1
        labels[t][anti] = 2

    return labels


def _compute_global_sigma_from_ds(
    ds: xr.Dataset,
    chunk_size: int,
    *,
    latitude: np.ndarray,
    longitude: np.ndarray,
    gradient_mode: str,
) -> float:
    """按时间分块统计 OW 全局标准差，避免一次性占用大量内存。"""
    tlen = int(ds["ugos"].shape[0])
    count = 0
    sum_ow = 0.0
    sumsq_ow = 0.0

    for t0 in range(0, tlen, chunk_size):
        t1 = min(tlen, t0 + chunk_size)
        ug = np.asarray(ds["ugos"].values[t0:t1], dtype=np.float32)
        vg = np.asarray(ds["vgos"].values[t0:t1], dtype=np.float32)
        ow, _ = _ow_and_vorticity(
            ug,
            vg,
            latitude,
            longitude,
            gradient_mode=gradient_mode,
        )
        ow64 = np.asarray(ow, dtype=np.float64)
        valid = np.isfinite(ow64)
        if np.any(valid):
            vals = ow64[valid]
            count += int(vals.size)
            sum_ow += float(vals.sum())
            sumsq_ow += float((vals * vals).sum())

    if count <= 1:
        return 1.0

    mean = sum_ow / count
    var = max(0.0, (sumsq_ow / count) - (mean * mean))
    sigma = float(np.sqrt(var))
    if not np.isfinite(sigma) or sigma <= 1e-12:
        sigma = 1.0
    return sigma


def generate_labels_for_clean_file(
    clean_nc: Path,
    out_nc: Path,
    cfg: EddyLabelConfig,
) -> tuple[Path, int]:
    """为单个 `*_clean.nc` 生成 `eddy_mask` 标签文件。"""
    ds = open_nc(clean_nc)
    try:
        if "ugos" not in ds or "vgos" not in ds:
            raise KeyError(f"missing ugos/vgos in {clean_nc}")
        tlen = int(ds["ugos"].shape[0])
        h = int(ds["ugos"].shape[1])
        w = int(ds["ugos"].shape[2])
        latitude = np.asarray(ds["latitude"].values, dtype=np.float64)
        longitude = np.asarray(ds["longitude"].values, dtype=np.float64)
        chunk_size = max(1, int(cfg.chunk_size))
        mode = str(cfg.threshold_mode).strip().lower()
        gradient_mode = str(cfg.gradient_mode).strip().lower()

        sigma_global = None
        if mode == "global":
            sigma_global = _compute_global_sigma_from_ds(
                ds,
                chunk_size=chunk_size,
                latitude=latitude,
                longitude=longitude,
                gradient_mode=gradient_mode,
            )

        labels = np.zeros((tlen, h, w), dtype=np.uint8)
        for t0 in range(0, tlen, chunk_size):
            t1 = min(tlen, t0 + chunk_size)
            ug_chunk = np.asarray(ds["ugos"].values[t0:t1], dtype=np.float32)
            vg_chunk = np.asarray(ds["vgos"].values[t0:t1], dtype=np.float32)
            ow_chunk, zeta_chunk = _ow_and_vorticity(
                ug_chunk,
                vg_chunk,
                latitude,
                longitude,
                gradient_mode=gradient_mode,
            )

            for i in range(t1 - t0):
                if mode == "daily":
                    sigma_i = float(np.nanstd(ow_chunk[i]))
                else:
                    sigma_i = float(sigma_global if sigma_global is not None else 1.0)
                if not np.isfinite(sigma_i) or sigma_i <= 1e-12:
                    sigma_i = 1.0

                core_t = (ow_chunk[i] < (-cfg.ow_std_scale * sigma_i)).astype(np.uint8)
                core_t = remove_small_components(core_t, cfg.min_region_pixels)
                cyc = np.logical_and(core_t > 0, _cyclonic_mask(zeta_chunk[i], latitude, cfg.polarity_mode))
                anti = np.logical_and(core_t > 0, zeta_chunk[i] <= 0)
                labels[t0 + i][cyc] = 1
                labels[t0 + i][anti] = 2

        coords = {
            "time": ds["time"],
            "latitude": ds["latitude"],
            "longitude": ds["longitude"],
        }
        out_ds = xr.Dataset(
            data_vars={
                "eddy_mask": (("time", "latitude", "longitude"), labels),
            },
            coords=coords,
            attrs={
                "source_clean_file": str(clean_nc),
                "label_definition": "0=background,1=cyclonic,2=anticyclonic",
                "method": "Okubo-Weiss + vorticity-sign",
                "ow_std_scale": float(cfg.ow_std_scale),
                "threshold_mode": str(cfg.threshold_mode),
                "gradient_mode": str(cfg.gradient_mode),
                "polarity_mode": str(cfg.polarity_mode),
                "min_region_pixels": int(cfg.min_region_pixels),
                "chunk_size": int(chunk_size),
            },
        )
        out_nc.parent.mkdir(parents=True, exist_ok=True)
        if out_nc.exists():
            out_nc.unlink()
        comp = {"zlib": True, "complevel": 4}
        tmp_nc = Path(tempfile.gettempdir()) / f"eddy_label_{clean_nc.stem}.nc"
        if tmp_nc.exists():
            tmp_nc.unlink()
        out_ds.to_netcdf(tmp_nc, mode="w", encoding={"eddy_mask": comp})
        shutil.move(str(tmp_nc), str(out_nc))
    finally:
        ds.close()

    return out_nc, int(labels.shape[0])
