"""按时间坐标将清洗后的样本合并为任务级总文件。"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

from netCDF4 import Dataset as NcDataset
import numpy as np

from data_preprocessing.io import open_nc
from data_preprocessing.splitter import (
    TASK_ANOMALY,
    TASK_EDDY,
    TASK_ELEMENT,
    list_processed_anomaly,
    list_processed_eddy,
    list_processed_element,
)
from utils.logger import tqdm

_log = logging.getLogger(__name__)


def _time_coord_name(ds: Any) -> str:
    for name in ("time", "valid_time"):
        if name in ds.coords:
            return name
    raise ValueError("dataset has no time coord: expected one of time/valid_time")


def _first_time_value(path: Path) -> tuple[str, float | None]:
    ds = open_nc(path, decode_times=False)
    try:
        coord = _time_coord_name(ds)
        t = np.asarray(ds.coords[coord].values).ravel()
        if t.size == 0:
            return coord, None
        return coord, float(t[0])
    finally:
        ds.close()


def _sort_paths_by_time(paths: list[Path]) -> tuple[str, list[Path], list[Path]]:
    if not paths:
        raise ValueError("no input files to merge")

    keyed: list[tuple[float, int, Path]] = []
    skipped: list[Path] = []
    coord_name: str | None = None
    for i, p in enumerate(paths):
        coord, t0 = _first_time_value(p)
        if coord_name is None:
            coord_name = coord
        elif coord_name != coord:
            raise ValueError(
                f"inconsistent time coordinate: expected {coord_name}, got {coord} in {p.name}"
            )
        if t0 is None:
            skipped.append(p)
            continue
        keyed.append((t0, i, p))

    if not keyed:
        raise ValueError("all input files have empty time coordinate")

    keyed.sort(key=lambda x: (x[0], x[1]))
    return coord_name or "time", [p for _, _, p in keyed], skipped


def _create_output_schema_from_template(
    template_path: Path,
    out_file: Path,
    time_coord: str,
    complevel: int,
) -> None:
    ds = open_nc(template_path, decode_times=False)
    try:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        if out_file.exists():
            out_file.unlink()
        # 先写一个 0 帧模板文件，保持 xarray 对 time/attrs/encoding 的兼容处理。
        template = ds.isel({time_coord: slice(0, 0)})
        enc: dict[str, dict[str, Any]] = {}
        for var_name in template.data_vars:
            enc[var_name] = {"zlib": True, "complevel": int(complevel)}
        template.to_netcdf(
            out_file,
            encoding=enc,
            unlimited_dims=[time_coord],
        )
    finally:
        ds.close()


def _append_one_file(path: Path, out_file: Path, time_coord: str, offset: int) -> int:
    ds = open_nc(path, decode_times=False)
    try:
        time_values = np.asarray(ds.coords[time_coord].values)
        if time_values.size == 0:
            return offset

        n_time = int(time_values.shape[0])
        with NcDataset(out_file, mode="a") as nc:
            if time_coord in nc.variables:
                nc.variables[time_coord][offset : offset + n_time] = time_values

            for var_name, da in ds.data_vars.items():
                if var_name not in nc.variables:
                    _log.warning("skip var missing in output schema: %s (%s)", var_name, path.name)
                    continue
                arr = np.asarray(da.values)
                dims = tuple(da.dims)
                if time_coord in dims:
                    axis = dims.index(time_coord)
                    idx = [slice(None)] * arr.ndim
                    idx[axis] = slice(offset, offset + n_time)
                    nc.variables[var_name][tuple(idx)] = arr
                elif offset == 0:
                    nc.variables[var_name][:] = arr
        return offset + n_time
    finally:
        ds.close()


def _merge_files(in_files: list[Path], out_file: Path, complevel: int) -> Path:
    if not in_files:
        raise ValueError("no input files to merge")

    time_coord, ordered, skipped = _sort_paths_by_time(in_files)
    if skipped:
        _log.warning(
            "skip %s files with empty %s coordinate: %s",
            len(skipped),
            time_coord,
            ", ".join(p.name for p in skipped[:10]),
        )
    _create_output_schema_from_template(ordered[0], out_file, time_coord, complevel)
    offset = 0
    for p in tqdm(
        ordered,
        total=len(ordered),
        desc=f"merge files ({out_file.name})",
        unit="file",
    ):
        offset = _append_one_file(p, out_file, time_coord, offset)

    _log.info("merged %s files (%s frames) -> %s", len(ordered), offset, out_file)
    return out_file


def run_merge_for_task(
    task: str,
    cfg: Mapping[str, Any],
    root: Path,
    *,
    limit: int | None = None,
) -> list[Path]:
    """按任务合并清洗结果，返回产出文件路径列表。"""
    complevel = int(cfg.get("output", {}).get("complevel", 4))

    if task == TASK_EDDY:
        files = list_processed_eddy(cfg, root)
        if limit is not None:
            files = files[: max(0, limit)]
        out = root / cfg["paths"]["processed"]["eddy"] / "all_clean_merged.nc"
        return [_merge_files(files, out, complevel)]

    if task == TASK_ELEMENT:
        files = list_processed_element(cfg, root)
        if limit is not None:
            files = files[: max(0, limit)]
        out = root / cfg["paths"]["processed"]["element_forecasting"] / "all_clean_merged.nc"
        result = _merge_files(files, out, complevel)
        path_txt = root / cfg["paths"]["processed"]["element_forecasting"] / "path.txt"
        path_txt.write_text(str(result.resolve()), encoding="utf-8")
        return [result]

    if task == TASK_ANOMALY:
        year_dirs = list_processed_anomaly(cfg, root)
        if limit is not None:
            year_dirs = year_dirs[: max(0, limit)]
        oper_files = [d / "oper_clean.nc" for d in year_dirs if (d / "oper_clean.nc").is_file()]
        wave_files = [d / "wave_clean.nc" for d in year_dirs if (d / "wave_clean.nc").is_file()]
        out_dir = root / cfg["paths"]["processed"]["anomaly"]
        outs: list[Path] = []
        if oper_files:
            outs.append(_merge_files(oper_files, out_dir / "oper_merged.nc", complevel))
        if wave_files:
            outs.append(_merge_files(wave_files, out_dir / "wave_merged.nc", complevel))
        if not outs:
            raise ValueError("no anomaly oper/wave files to merge")
        return outs

    raise ValueError(f"unknown task: {task}")
