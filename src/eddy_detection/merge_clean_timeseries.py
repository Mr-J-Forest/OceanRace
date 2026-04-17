"""将涡旋清洗后的 5 个 NetCDF 文件按时间序列拼接为一个总文件。"""
from __future__ import annotations

import argparse
import logging
import shutil
import tempfile
import sys
from pathlib import Path
from typing import Any

import numpy as np
from netCDF4 import Dataset as NcDataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data_preprocessing.io import open_nc

_log = logging.getLogger(__name__)


def _setup_logging(log_file: Path | None = None) -> None:
    root = logging.getLogger()
    if root.handlers:
        return
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    root.addHandler(sh)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)


def _project_root() -> Path:
    return ROOT


def _default_input_files(input_dir: Path) -> list[Path]:
    names = [
        "19930101_20021231_clean.nc",
        "20030101_20121231_clean.nc",
        "20130101_20221231_clean.nc",
        "20230101_20231231_clean.nc",
        "20240101_20241231_clean.nc",
    ]
    return [input_dir / n for n in names]


def _first_last_time(path: Path, time_coord: str = "time") -> tuple[float | None, float | None]:
    ds = open_nc(path, decode_times=False)
    try:
        if time_coord not in ds.coords:
            raise KeyError(f"missing time coord {time_coord!r} in {path}")
        vals = np.asarray(ds.coords[time_coord].values).ravel()
        if vals.size == 0:
            return None, None
        return float(vals[0]), float(vals[-1])
    finally:
        ds.close()


def _validate_inputs(paths: list[Path], time_coord: str = "time") -> None:
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("missing input files:\n" + "\n".join(str(p) for p in missing))

    prev_end: float | None = None
    for p in paths:
        start, end = _first_last_time(p, time_coord=time_coord)
        if start is None or end is None:
            _log.warning("skip empty time file: %s", p)
            continue
        if prev_end is not None and start <= prev_end:
            raise ValueError(
                f"time overlap or disorder detected: prev_end={prev_end}, current_start={start}, file={p.name}"
            )
        # 允许 1 天间隔，若不是则给 warning（不强制中断）。
        if prev_end is not None and abs((start - prev_end) - 1.0) > 1e-6:
            _log.warning(
                "time not strictly day-by-day between files: prev_end=%s current_start=%s file=%s",
                prev_end,
                start,
                p.name,
            )
        prev_end = end


def _create_schema(template_path: Path, out_tmp: Path, time_coord: str, complevel: int) -> None:
    ds = open_nc(template_path, decode_times=False)
    try:
        if out_tmp.exists():
            out_tmp.unlink()
        tmpl = ds.isel({time_coord: slice(0, 0)})
        enc: dict[str, dict[str, Any]] = {
            str(var): {"zlib": True, "complevel": int(complevel)} for var in tmpl.data_vars
        }
        tmpl.to_netcdf(out_tmp, encoding=enc, unlimited_dims=[time_coord])
    finally:
        ds.close()


def _append_file(path: Path, out_tmp: Path, time_coord: str, offset: int) -> int:
    ds = open_nc(path, decode_times=False)
    try:
        if time_coord not in ds.coords:
            raise KeyError(f"missing coord {time_coord!r} in {path}")
        tvals = np.asarray(ds.coords[time_coord].values)
        if tvals.size == 0:
            return offset

        n_time = int(tvals.shape[0])
        tkey = str(time_coord)
        with NcDataset(out_tmp, mode="a") as nc:
            nc.variables[tkey][offset : offset + n_time] = tvals
            for var_name, da in ds.data_vars.items():
                vname = str(var_name)
                if vname not in nc.variables:
                    _log.warning("skip variable not in output schema: %s", vname)
                    continue
                arr = np.asarray(da.values)
                dims = tuple(str(d) for d in da.dims)
                if tkey in dims:
                    axis = dims.index(tkey)
                    idx = [slice(None)] * arr.ndim
                    idx[axis] = slice(offset, offset + n_time)
                    nc.variables[vname][tuple(idx)] = arr
                elif offset == 0:
                    nc.variables[vname][:] = arr
        return offset + n_time
    finally:
        ds.close()


def merge_eddy_clean_files(
    input_files: list[Path],
    output_file: Path,
    *,
    time_coord: str = "time",
    complevel: int = 4,
    overwrite: bool = False,
) -> Path:
    _validate_inputs(input_files, time_coord=time_coord)

    if output_file.exists() and not overwrite:
        raise FileExistsError(f"output exists, use --overwrite: {output_file}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = Path(tempfile.gettempdir()) / output_file.name

    _create_schema(input_files[0], tmp_out, time_coord=time_coord, complevel=complevel)

    offset = 0
    for p in tqdm(input_files, desc="merge eddy clean files", unit="file"):
        offset = _append_file(p, tmp_out, time_coord=time_coord, offset=offset)
        _log.info("appended %s -> total time steps=%s", p.name, offset)

    if output_file.exists():
        output_file.unlink()
    shutil.move(str(tmp_out), str(output_file))
    _log.info("merged %s files (%s time steps) -> %s", len(input_files), offset, output_file)
    return output_file


def main() -> None:
    root = _project_root()
    default_in = root / "data/processed/eddy_detection"
    default_out = default_in / "19930101_20241231_clean.nc"

    ap = argparse.ArgumentParser(description="拼接 5 个涡旋 clean.nc 文件为一个时间序列总文件")
    ap.add_argument("--input-dir", type=Path, default=default_in)
    ap.add_argument("--output", type=Path, default=default_out)
    ap.add_argument("--time-coord", type=str, default="time")
    ap.add_argument("--complevel", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    _setup_logging(log_file=root / "outputs/eddy_detection/merge_eddy.log")

    files = _default_input_files(args.input_dir)
    merge_eddy_clean_files(
        input_files=files,
        output_file=args.output,
        time_coord=args.time_coord,
        complevel=args.complevel,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
