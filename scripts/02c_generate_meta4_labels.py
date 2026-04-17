"""基于 META4.0 方法链路生成涡旋标签（单一 clean.nc）。

输出变量：
- eddy_mask(time, latitude, longitude): 0=背景, 1=气旋, 2=反气旋
- eddy_instance_id(time, latitude, longitude): 每日实例编号（跨日不连续）
- obs_count(time): 每个时间片检测到的涡旋个数
- processed(time): 该时间片是否已完成写入

说明：
- 本脚本使用 py-eddy-tracker 的闭合轮廓检测与筛选逻辑，参数默认按 META4.0 handbook 对齐。
- 默认仅做 detection 标注，不做跨日 network/segment 关联编号。
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

import netCDF4 as nc4
import numpy as np
import xarray as xr
from matplotlib.path import Path as MplPath

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _default_pet_src() -> Path:
    candidates = [
        ROOT / "py-eddy-tracker-master" / "py-eddy-tracker-master" / "src",
        ROOT / "py-eddy-tracker-master" / "src",
        ROOT.parent / "py-eddy-tracker-master" / "py-eddy-tracker-master" / "src",
        ROOT.parent / "py-eddy-tracker-master" / "src",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]

from utils.logger import get_logger, setup_logging, tqdm, tqdm_logging  # noqa: E402

_log = get_logger(__name__)


def _to_py_datetime(t: Any) -> datetime:
    if isinstance(t, datetime):
        return t
    if hasattr(t, "year") and hasattr(t, "month") and hasattr(t, "day"):
        return datetime(int(t.year), int(t.month), int(t.day))
    if isinstance(t, np.datetime64):
        sec = int((t - np.datetime64("1970-01-01T00:00:00")) / np.timedelta64(1, "s"))
        return datetime.fromtimestamp(sec, timezone.utc).replace(tzinfo=None)
    return datetime.fromtimestamp(0, timezone.utc).replace(tzinfo=None)


def _ensure_pet_import(pet_src: Path | None) -> Any:
    if pet_src is not None and str(pet_src) not in sys.path:
        sys.path.insert(0, str(pet_src))
    from py_eddy_tracker.dataset.grid import RegularGridDataset  # type: ignore

    return RegularGridDataset


def _rasterize_polygon(
    lon: np.ndarray,
    lat: np.ndarray,
    poly_lon: np.ndarray,
    poly_lat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    x = np.asarray(poly_lon, dtype=np.float64)
    y = np.asarray(poly_lat, dtype=np.float64)
    good = np.isfinite(x) & np.isfinite(y)
    if good.sum() < 3:
        return None
    x = x[good]
    y = y[good]

    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))

    lat_idx = np.where((lat >= y_min) & (lat <= y_max))[0]
    lon_idx = np.where((lon >= x_min) & (lon <= x_max))[0]

    if lat_idx.size == 0 or lon_idx.size == 0:
        return None

    lon_sub = lon[lon_idx]
    lat_sub = lat[lat_idx]
    xx, yy = np.meshgrid(lon_sub, lat_sub)
    points = np.column_stack([xx.ravel(), yy.ravel()])

    path = MplPath(np.column_stack([x, y]), closed=True)
    inside = path.contains_points(points).reshape(lat_sub.size, lon_sub.size)
    if not np.any(inside):
        return None

    return lat_idx, lon_idx, inside


def _init_output(
    out_nc: Path,
    in_ds: xr.Dataset,
    overwrite: bool,
) -> None:
    if out_nc.exists() and overwrite:
        out_nc.unlink()
    if out_nc.exists():
        return

    out_nc.parent.mkdir(parents=True, exist_ok=True)

    time_raw = np.asarray(in_ds["time"].values)
    lat = np.asarray(in_ds["latitude"].values)
    lon = np.asarray(in_ds["longitude"].values)

    with nc4.Dataset(out_nc, "w", format="NETCDF4") as dst:
        dst.createDimension("time", time_raw.shape[0])
        dst.createDimension("latitude", lat.shape[0])
        dst.createDimension("longitude", lon.shape[0])

        tvar = dst.createVariable("time", time_raw.dtype, ("time",))
        yvar = dst.createVariable("latitude", lat.dtype, ("latitude",))
        xvar = dst.createVariable("longitude", lon.dtype, ("longitude",))

        tvar[:] = time_raw
        yvar[:] = lat
        xvar[:] = lon

        for k, v in in_ds["time"].attrs.items():
            setattr(tvar, k, v)
        for k, v in in_ds["latitude"].attrs.items():
            setattr(yvar, k, v)
        for k, v in in_ds["longitude"].attrs.items():
            setattr(xvar, k, v)

        comp = dict(zlib=True, complevel=4)
        dst.createVariable("eddy_mask", "u1", ("time", "latitude", "longitude"), fill_value=0, **comp)
        dst.createVariable("eddy_instance_id", "i4", ("time", "latitude", "longitude"), fill_value=0, **comp)
        dst.createVariable("obs_count", "i4", ("time",), fill_value=0)
        dst.createVariable("processed", "u1", ("time",), fill_value=0)

        dst.setncattr("label_definition", "eddy_mask: 0=background,1=cyclonic,2=anticyclonic")
        dst.setncattr("instance_definition", "eddy_instance_id: per-time-step object id, 0=background")
        dst.setncattr("method", "META4.0-style contour detection via py-eddy-tracker")
        dst.setncattr("note", "Detection labels only; no cross-day network/segment ids in this file")


def main() -> None:
    ap = argparse.ArgumentParser(description="META4.0 风格涡旋标签生成（单文件）")
    ap.add_argument(
        "--clean-nc",
        type=Path,
        default=ROOT / "data/processed/eddy_detection/19930101_20241231_clean.nc",
    )
    ap.add_argument(
        "--out-nc",
        type=Path,
        default=ROOT / "data/processed/eddy_detection/labels/19930101_20241231_label_meta4.nc",
    )
    ap.add_argument(
        "--pet-src",
        type=Path,
        default=_default_pet_src(),
        help="py-eddy-tracker 的 src 路径",
    )
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--end-index", type=int, default=-1, help="-1 表示到最后")
    ap.add_argument("--overwrite", action="store_true")

    # META4.0 handbook 对齐参数（可按需微调）
    ap.add_argument("--step-m", type=float, default=0.002, help="等值线步长，单位 m（0.2 cm）")
    ap.add_argument("--shape-error", type=float, default=70.0)
    ap.add_argument("--pixel-min", type=int, default=5)
    ap.add_argument("--pixel-max", type=int, default=1000)
    ap.add_argument("--presampling-multiplier", type=int, default=10)
    ap.add_argument("--sampling", type=int, default=20)
    ap.add_argument("--sampling-method", type=str, default="visvalingam", choices=["visvalingam", "uniform"])
    ap.add_argument("--mle", type=int, default=1, help="轮廓内局地极值允许个数")
    ap.add_argument("--nb-step-min", type=int, default=2, help="振幅阈值步数，2 对应 0.4cm")
    ap.add_argument("--nb-step-to-be-mle", type=int, default=2)
    ap.add_argument(
        "--sync-every",
        type=int,
        default=10,
        help="每处理多少个时间片强制同步一次到磁盘，降低长任务中断风险",
    )

    args = ap.parse_args()

    setup_logging(log_file=ROOT / "outputs/logs/eddy_label_meta4.log")

    if not args.clean_nc.is_file():
        raise FileNotFoundError(f"clean file not found: {args.clean_nc}")

    RegularGridDataset = _ensure_pet_import(args.pet_src)

    in_ds = xr.open_dataset(args.clean_nc, decode_times=False)
    in_ds_dt = xr.open_dataset(args.clean_nc, decode_times=True)
    try:
        if "time" not in in_ds.coords or "latitude" not in in_ds.coords or "longitude" not in in_ds.coords:
            raise KeyError("clean.nc must contain time/latitude/longitude coords")

        if not {"adt", "ugos", "vgos"}.issubset(set(in_ds.variables)):
            raise KeyError("clean.nc must contain adt/ugos/vgos")

        _init_output(args.out_nc, in_ds, overwrite=bool(args.overwrite))

        time_vals = np.asarray(in_ds_dt["time"].values)
        lat = np.asarray(in_ds["latitude"].values, dtype=np.float64)
        lon = np.asarray(in_ds["longitude"].values, dtype=np.float64)
        tlen = int(time_vals.shape[0])

        start = max(0, int(args.start_index))
        end = tlen - 1 if int(args.end_index) < 0 else min(tlen - 1, int(args.end_index))
        if end < start:
            raise ValueError(f"invalid range: start={start}, end={end}")

        with nc4.Dataset(args.out_nc, "r+") as out_ds, tqdm_logging():
            v_mask = out_ds.variables["eddy_mask"]
            v_id = out_ds.variables["eddy_instance_id"]
            v_cnt = out_ds.variables["obs_count"]
            v_done = out_ds.variables["processed"]

            for ti in tqdm(range(start, end + 1), desc="meta4 label", unit="time"):
                done_val = v_done[ti]
                if np.ma.is_masked(done_val):
                    done_val = 0
                if int(done_val) == 1 and not args.overwrite:
                    continue

                dt = _to_py_datetime(time_vals[ti])

                grid = RegularGridDataset(
                    str(args.clean_nc),
                    "longitude",
                    "latitude",
                    indexs={"time": int(ti)},
                )

                anti, cyc = grid.eddy_identification(
                    "adt",
                    "ugos",
                    "vgos",
                    dt,
                    step=float(args.step_m),
                    shape_error=float(args.shape_error),
                    presampling_multiplier=int(args.presampling_multiplier),
                    sampling=int(args.sampling),
                    sampling_method=str(args.sampling_method),
                    pixel_limit=(int(args.pixel_min), int(args.pixel_max)),
                    mle=int(args.mle),
                    nb_step_min=int(args.nb_step_min),
                    nb_step_to_be_mle=int(args.nb_step_to_be_mle),
                )

                mask = np.zeros((lat.shape[0], lon.shape[0]), dtype=np.uint8)
                inst = np.zeros((lat.shape[0], lon.shape[0]), dtype=np.int32)
                obs_id = 0

                # cyclonic -> 1
                for oi in range(len(cyc)):
                    rr = _rasterize_polygon(lon, lat, cyc.contour_lon_e[oi], cyc.contour_lat_e[oi])
                    if rr is None:
                        continue
                    lat_idx, lon_idx, inside = rr
                    sub_m = mask[np.ix_(lat_idx, lon_idx)]
                    sub_i = inst[np.ix_(lat_idx, lon_idx)]
                    write = np.logical_and(inside, sub_m == 0)
                    if np.any(write):
                        obs_id += 1
                        sub_m[write] = 1
                        sub_i[write] = obs_id
                        mask[np.ix_(lat_idx, lon_idx)] = sub_m
                        inst[np.ix_(lat_idx, lon_idx)] = sub_i

                # anticyclonic -> 2
                for oi in range(len(anti)):
                    rr = _rasterize_polygon(lon, lat, anti.contour_lon_e[oi], anti.contour_lat_e[oi])
                    if rr is None:
                        continue
                    lat_idx, lon_idx, inside = rr
                    sub_m = mask[np.ix_(lat_idx, lon_idx)]
                    sub_i = inst[np.ix_(lat_idx, lon_idx)]
                    write = np.logical_and(inside, sub_m == 0)
                    if np.any(write):
                        obs_id += 1
                        sub_m[write] = 2
                        sub_i[write] = obs_id
                        mask[np.ix_(lat_idx, lon_idx)] = sub_m
                        inst[np.ix_(lat_idx, lon_idx)] = sub_i

                v_mask[ti, :, :] = mask
                v_id[ti, :, :] = inst
                v_cnt[ti] = int(obs_id)
                v_done[ti] = 1

                if args.sync_every > 0 and ((ti - start + 1) % int(args.sync_every) == 0):
                    out_ds.sync()

            out_ds.sync()
    finally:
        in_ds_dt.close()
        in_ds.close()

    _log.info("META4 label generation done: %s", args.out_nc)


if __name__ == "__main__":
    main()
