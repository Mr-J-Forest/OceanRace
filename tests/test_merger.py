"""data_preprocessing.merger：按时序合并清洗后样本。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from data_preprocessing.merger import run_merge_for_task
from data_preprocessing.splitter import TASK_ANOMALY, TASK_EDDY, TASK_ELEMENT
from tests.conftest import minimal_split_cfg


def _write_eddy_clean(path: Path, times: list[float]) -> None:
    t = len(times)
    la, lo = 2, 2
    shape = (t, la, lo)
    ones = np.ones(shape, dtype=np.float32)
    ds = xr.Dataset(
        data_vars={
            "adt": (("time", "latitude", "longitude"), np.ones(shape, dtype=np.float32)),
            "ugos": (("time", "latitude", "longitude"), np.ones(shape, dtype=np.float32)),
            "vgos": (("time", "latitude", "longitude"), np.ones(shape, dtype=np.float32)),
            "adt_valid": (("time", "latitude", "longitude"), ones),
            "ugos_valid": (("time", "latitude", "longitude"), ones),
            "vgos_valid": (("time", "latitude", "longitude"), ones),
        },
        coords={
            "time": np.asarray(times, dtype=np.float64),
            "latitude": np.arange(la, dtype=np.float32),
            "longitude": np.arange(lo, dtype=np.float32),
        },
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)


def _write_element_clean(path: Path, times: list[float]) -> None:
    t = len(times)
    la, lo = 2, 2
    shape = (t, la, lo)
    ones = np.ones(shape, dtype=np.float32)
    vars_ = ("sst", "sss", "ssu", "ssv")
    data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
    for n in vars_:
        data_vars[n] = (("time", "lat", "lon"), np.ones(shape, dtype=np.float32))
        data_vars[f"{n}_valid"] = (("time", "lat", "lon"), ones)
    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": np.asarray(times, dtype=np.float64),
            "lat": np.arange(la, dtype=np.float32),
            "lon": np.arange(lo, dtype=np.float32),
        },
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)


def _write_anomaly_clean(path: Path, times: list[float], *, kind: str) -> None:
    t = len(times)
    la, lo = 2, 2
    shape = (t, la, lo)
    ones = np.ones(shape, dtype=np.float32)

    if kind == "oper":
        ds = xr.Dataset(
            data_vars={
                "u10": (("valid_time", "latitude", "longitude"), np.ones(shape, dtype=np.float32)),
                "v10": (("valid_time", "latitude", "longitude"), np.ones(shape, dtype=np.float32)),
                "wind_speed": (("valid_time", "latitude", "longitude"), np.ones(shape, dtype=np.float32)),
                "u10_valid": (("valid_time", "latitude", "longitude"), ones),
                "v10_valid": (("valid_time", "latitude", "longitude"), ones),
                "wind_speed_valid": (("valid_time", "latitude", "longitude"), ones),
            },
            coords={
                "valid_time": np.asarray(times, dtype=np.float64),
                "latitude": np.arange(la, dtype=np.float32),
                "longitude": np.arange(lo, dtype=np.float32),
            },
        )
    else:
        ds = xr.Dataset(
            data_vars={
                "swh": (("valid_time", "latitude", "longitude"), np.ones(shape, dtype=np.float32)),
                "mwp": (("valid_time", "latitude", "longitude"), np.ones(shape, dtype=np.float32)),
                "mwd": (("valid_time", "latitude", "longitude"), np.ones(shape, dtype=np.float32)),
                "swh_valid": (("valid_time", "latitude", "longitude"), ones),
                "mwp_valid": (("valid_time", "latitude", "longitude"), ones),
                "mwd_valid": (("valid_time", "latitude", "longitude"), ones),
            },
            coords={
                "valid_time": np.asarray(times, dtype=np.float64),
                "latitude": np.arange(la, dtype=np.float32),
                "longitude": np.arange(lo, dtype=np.float32),
            },
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)


def test_merge_eddy_files_sorted_by_time(tmp_path: Path) -> None:
    cfg = minimal_split_cfg(
        "data/processed/eddy_detection",
        "data/processed/element_forecasting",
        "data/processed/anomaly_detection",
    )
    eddy_dir = tmp_path / "data/processed/eddy_detection"
    # 文件名逆序，验证按 time 坐标而非文件名顺序合并。
    _write_eddy_clean(eddy_dir / "b_clean.nc", [10.0, 11.0])
    _write_eddy_clean(eddy_dir / "a_clean.nc", [1.0, 2.0])

    outs = run_merge_for_task(TASK_EDDY, cfg, tmp_path)
    assert len(outs) == 1
    merged = xr.open_dataset(outs[0])
    try:
        assert merged.sizes["time"] == 4
        assert np.all(np.diff(np.asarray(merged["time"].values)) >= 0)
    finally:
        merged.close()


def test_merge_element_outputs_single_file(tmp_path: Path) -> None:
    cfg = minimal_split_cfg(
        "data/processed/eddy_detection",
        "data/processed/element_forecasting",
        "data/processed/anomaly_detection",
    )
    el_dir = tmp_path / "data/processed/element_forecasting"
    _write_element_clean(el_dir / "x_clean.nc", [2.0, 3.0])
    _write_element_clean(el_dir / "y_clean.nc", [0.0, 1.0])

    outs = run_merge_for_task(TASK_ELEMENT, cfg, tmp_path)
    assert [p.name for p in outs] == ["all_clean_merged.nc"]
    assert outs[0].is_file()


def test_merge_element_skips_empty_time_file(tmp_path: Path) -> None:
    cfg = minimal_split_cfg(
        "data/processed/eddy_detection",
        "data/processed/element_forecasting",
        "data/processed/anomaly_detection",
    )
    el_dir = tmp_path / "data/processed/element_forecasting"
    _write_element_clean(el_dir / "ok_clean.nc", [0.0, 1.0])
    _write_element_clean(el_dir / "empty_clean.nc", [])

    outs = run_merge_for_task(TASK_ELEMENT, cfg, tmp_path)
    merged = xr.open_dataset(outs[0])
    try:
        assert merged.sizes["time"] == 2
    finally:
        merged.close()


def test_merge_anomaly_outputs_oper_and_wave(tmp_path: Path) -> None:
    cfg = minimal_split_cfg(
        "data/processed/eddy_detection",
        "data/processed/element_forecasting",
        "data/processed/anomaly_detection",
    )
    an_dir = tmp_path / "data/processed/anomaly_detection"
    y1 = an_dir / "2020"
    y2 = an_dir / "2021"
    _write_anomaly_clean(y2 / "oper_clean.nc", [10.0, 11.0], kind="oper")
    _write_anomaly_clean(y1 / "oper_clean.nc", [1.0, 2.0], kind="oper")
    _write_anomaly_clean(y2 / "wave_clean.nc", [10.0, 11.0], kind="wave")
    _write_anomaly_clean(y1 / "wave_clean.nc", [1.0, 2.0], kind="wave")

    outs = run_merge_for_task(TASK_ANOMALY, cfg, tmp_path)
    names = sorted(p.name for p in outs)
    assert names == ["oper_merged.nc", "wave_merged.nc"]
    for p in outs:
        assert p.is_file()


def test_merge_element_skips_empty_time_file(tmp_path: Path) -> None:
    cfg = minimal_split_cfg(
        "data/processed/eddy_detection",
        "data/processed/element_forecasting",
        "data/processed/anomaly_detection",
    )
    el_dir = tmp_path / "data/processed/element_forecasting"
    _write_element_clean(el_dir / "empty_clean.nc", [])
    _write_element_clean(el_dir / "ok_clean.nc", [1.0, 2.0])

    outs = run_merge_for_task(TASK_ELEMENT, cfg, tmp_path)
    merged = xr.open_dataset(outs[0])
    try:
        assert merged.sizes["time"] == 2
        assert np.all(np.asarray(merged["time"].values) == np.asarray([1.0, 2.0]))
    finally:
        merged.close()
