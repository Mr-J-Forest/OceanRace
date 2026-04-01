"""共享夹具：在临时目录写入迷你 NetCDF，供 splitter / validator 回归测试。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr


def write_eddy_clean_nc(
    path: Path,
    *,
    time_coord: list[float] | None = None,
) -> None:
    """与 ``validate_eddy_nc`` 期望一致的涡旋清洗样例（小网格）。"""
    if time_coord is None:
        time_coord = [0.0, 1.0]
    t = len(time_coord)
    la, lo = 2, 2
    shape = (t, la, lo)
    adt = np.ones(shape, dtype=np.float32)
    ug = np.zeros(shape, dtype=np.float32)
    vg = np.zeros(shape, dtype=np.float32)
    ones = np.ones(shape, dtype=np.float32)
    ds = xr.Dataset(
        data_vars={
            "adt": (("time", "latitude", "longitude"), adt),
            "ugos": (("time", "latitude", "longitude"), ug),
            "vgos": (("time", "latitude", "longitude"), vg),
            "adt_valid": (("time", "latitude", "longitude"), ones),
            "ugos_valid": (("time", "latitude", "longitude"), ones),
            "vgos_valid": (("time", "latitude", "longitude"), ones),
        },
        coords={
            "time": time_coord,
            "latitude": np.arange(la, dtype=np.float32),
            "longitude": np.arange(lo, dtype=np.float32),
        },
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)


def write_element_clean_nc(
    path: Path,
    *,
    t: int = 2,
    lat: int = 2,
    lon: int = 2,
    base: float = 1.0,
    step: float = 1.0,
    vars_names: tuple[str, ...] = ("sst", "sss", "ssu", "ssv"),
) -> None:
    """写入 element clean 样例；每个变量随时间线性变化，便于断言窗口内容。"""

    la, lo = lat, lon
    shape = (t, la, lo)
    ones = np.ones(shape, dtype=np.float32)
    data_vars = {}
    for i, n in enumerate(vars_names):
        # 每个变量使用不同起点，防止多变量时互相混淆。
        offset = base + i * 100.0
        seq = np.arange(t, dtype=np.float32) * step + offset
        arr = np.broadcast_to(seq[:, None, None], shape).astype(np.float32)
        data_vars[n] = (("time", "lat", "lon"), arr)
        data_vars[f"{n}_valid"] = (("time", "lat", "lon"), ones)
    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": np.arange(t, dtype=np.float32),
            "lat": np.arange(la, dtype=np.float32),
            "lon": np.arange(lo, dtype=np.float32),
        },
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)


def write_anomaly_year_dir(year_dir: Path) -> None:
    """某年目录下 ``oper_clean.nc`` + ``wave_clean.nc``。"""
    year_dir.mkdir(parents=True, exist_ok=True)
    vt = [0.0, 1.0]
    la, lo = 2, 2
    shape = (len(vt), la, lo)
    ones = np.ones(shape, dtype=np.float32)
    u = np.ones(shape, dtype=np.float32)
    v = np.zeros(shape, dtype=np.float32)
    oper = xr.Dataset(
        data_vars={
            "u10": (("valid_time", "latitude", "longitude"), u),
            "v10": (("valid_time", "latitude", "longitude"), v),
            "u10_valid": (("valid_time", "latitude", "longitude"), ones),
            "v10_valid": (("valid_time", "latitude", "longitude"), ones),
        },
        coords={
            "valid_time": vt,
            "latitude": np.arange(la, dtype=np.float32),
            "longitude": np.arange(lo, dtype=np.float32),
        },
    )
    swh = np.ones(shape, dtype=np.float32)
    mwp = np.ones(shape, dtype=np.float32)
    mwd = np.ones(shape, dtype=np.float32)
    wave = xr.Dataset(
        data_vars={
            "swh": (("valid_time", "latitude", "longitude"), swh),
            "mwp": (("valid_time", "latitude", "longitude"), mwp),
            "mwd": (("valid_time", "latitude", "longitude"), mwd),
            "swh_valid": (("valid_time", "latitude", "longitude"), ones),
            "mwp_valid": (("valid_time", "latitude", "longitude"), ones),
            "mwd_valid": (("valid_time", "latitude", "longitude"), ones),
        },
        coords={
            "valid_time": vt,
            "latitude": np.arange(la, dtype=np.float32),
            "longitude": np.arange(lo, dtype=np.float32),
        },
    )
    oper.to_netcdf(year_dir / "oper_clean.nc")
    wave.to_netcdf(year_dir / "wave_clean.nc")


def minimal_split_cfg(root_eddy: str, root_el: str, root_an: str) -> dict:
    """与 ``splitter.list_processed_samples`` 兼容的最小 ``paths.processed`` 配置。"""
    return {
        "paths": {
            "processed": {
                "eddy": root_eddy,
                "element_forecasting": root_el,
                "anomaly": root_an,
            },
            "splits": "data/processed/splits",
            "normalization": "data/processed/normalization",
        }
    }
