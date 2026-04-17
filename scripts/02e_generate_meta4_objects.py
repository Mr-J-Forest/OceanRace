"""基于 META4.0 方法链路生成对象级涡旋标签（单一 clean.nc）。

输出为对象表（obs 维）+ 轮廓点（obs, NbSample）：
- 适合后续目标级训练、关系建模或转换为任意像素标签。
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


def _create_or_open_output(
    out_nc: Path,
    clean_ds: xr.Dataset,
    sampling: int,
    overwrite: bool,
) -> None:
    if out_nc.exists() and overwrite:
        out_nc.unlink()
    if out_nc.exists():
        return

    out_nc.parent.mkdir(parents=True, exist_ok=True)
    time_raw = np.asarray(clean_ds["time"].values)

    with nc4.Dataset(out_nc, "w", format="NETCDF4") as ds:
        ds.createDimension("time", int(time_raw.shape[0]))
        ds.createDimension("obs", None)
        ds.createDimension("NbSample", int(sampling))

        t = ds.createVariable("time", time_raw.dtype, ("time",))
        t[:] = time_raw
        for k, v in clean_ds["time"].attrs.items():
            setattr(t, k, v)

        ds.createVariable("processed", "u1", ("time",), fill_value=0)
        ds.createVariable("obs_count", "i4", ("time",), fill_value=0)

        comp = dict(zlib=True, complevel=4)
        ds.createVariable("time_index", "i4", ("obs",), **comp)
        ds.createVariable("polarity", "u1", ("obs",), **comp)
        ds.createVariable("center_longitude", "f4", ("obs",), fill_value=np.nan, **comp)
        ds.createVariable("center_latitude", "f4", ("obs",), fill_value=np.nan, **comp)
        ds.createVariable("amplitude", "f4", ("obs",), fill_value=np.nan, **comp)
        ds.createVariable("speed_radius", "f4", ("obs",), fill_value=np.nan, **comp)
        ds.createVariable("effective_radius", "f4", ("obs",), fill_value=np.nan, **comp)
        ds.createVariable("speed_average", "f4", ("obs",), fill_value=np.nan, **comp)
        ds.createVariable("shape_error_speed", "f4", ("obs",), fill_value=np.nan, **comp)
        ds.createVariable("shape_error_effective", "f4", ("obs",), fill_value=np.nan, **comp)

        ds.createVariable(
            "effective_contour_longitude",
            "f4",
            ("obs", "NbSample"),
            fill_value=np.nan,
            **comp,
        )
        ds.createVariable(
            "effective_contour_latitude",
            "f4",
            ("obs", "NbSample"),
            fill_value=np.nan,
            **comp,
        )
        ds.createVariable(
            "speed_contour_longitude",
            "f4",
            ("obs", "NbSample"),
            fill_value=np.nan,
            **comp,
        )
        ds.createVariable(
            "speed_contour_latitude",
            "f4",
            ("obs", "NbSample"),
            fill_value=np.nan,
            **comp,
        )

        ds.setncattr("method", "META4.0-style contour/object labels via py-eddy-tracker")
        ds.setncattr("note", "Object-level observations only, one row per detected eddy per time step")


def _fit_sample(arr: np.ndarray, n: int) -> np.ndarray:
    out = np.full((n,), np.nan, dtype=np.float32)
    if arr.size == 0:
        return out
    m = min(n, arr.size)
    out[:m] = arr[:m].astype(np.float32)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="META4 对象级标签生成（单文件）")
    ap.add_argument(
        "--clean-nc",
        type=Path,
        default=ROOT / "data/processed/eddy_detection/19930101_20241231_clean.nc",
    )
    ap.add_argument(
        "--out-nc",
        type=Path,
        default=ROOT / "data/processed/eddy_detection/labels/19930101_20241231_objects_meta4.nc",
    )
    ap.add_argument(
        "--pet-src",
        type=Path,
        default=_default_pet_src(),
    )
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--end-index", type=int, default=-1)
    ap.add_argument("--overwrite", action="store_true")

    # META4 对齐参数
    ap.add_argument("--step-m", type=float, default=0.002)
    ap.add_argument("--shape-error", type=float, default=70.0)
    ap.add_argument("--pixel-min", type=int, default=5)
    ap.add_argument("--pixel-max", type=int, default=1000)
    ap.add_argument("--presampling-multiplier", type=int, default=10)
    ap.add_argument("--sampling", type=int, default=20)
    ap.add_argument("--sampling-method", type=str, default="visvalingam", choices=["visvalingam", "uniform"])
    ap.add_argument("--mle", type=int, default=1)
    ap.add_argument("--nb-step-min", type=int, default=2)
    ap.add_argument("--nb-step-to-be-mle", type=int, default=2)
    ap.add_argument("--sync-every", type=int, default=10)

    args = ap.parse_args()

    setup_logging(log_file=ROOT / "outputs/logs/eddy_object_meta4.log")

    if not args.clean_nc.is_file():
        raise FileNotFoundError(f"clean file not found: {args.clean_nc}")

    RegularGridDataset = _ensure_pet_import(args.pet_src)

    clean_ds = xr.open_dataset(args.clean_nc, decode_times=False)
    clean_ds_dt = xr.open_dataset(args.clean_nc, decode_times=True)
    try:
        if "time" not in clean_ds.coords:
            raise KeyError("clean.nc must contain time coord")
        if not {"adt", "ugos", "vgos"}.issubset(set(clean_ds.variables)):
            raise KeyError("clean.nc must contain adt/ugos/vgos")

        _create_or_open_output(args.out_nc, clean_ds, int(args.sampling), bool(args.overwrite))

        tvals = np.asarray(clean_ds_dt["time"].values)
        tlen = int(tvals.shape[0])
        start = max(0, int(args.start_index))
        end = tlen - 1 if int(args.end_index) < 0 else min(tlen - 1, int(args.end_index))
        if end < start:
            raise ValueError(f"invalid range: start={start}, end={end}")

        with nc4.Dataset(args.out_nc, "r+") as out_ds, tqdm_logging():
            v_processed = out_ds.variables["processed"]
            v_count = out_ds.variables["obs_count"]

            v_ti = out_ds.variables["time_index"]
            v_pol = out_ds.variables["polarity"]
            v_lon = out_ds.variables["center_longitude"]
            v_lat = out_ds.variables["center_latitude"]
            v_amp = out_ds.variables["amplitude"]
            v_rs = out_ds.variables["speed_radius"]
            v_re = out_ds.variables["effective_radius"]
            v_us = out_ds.variables["speed_average"]
            v_ses = out_ds.variables["shape_error_speed"]
            v_see = out_ds.variables["shape_error_effective"]
            v_ce_lon = out_ds.variables["effective_contour_longitude"]
            v_ce_lat = out_ds.variables["effective_contour_latitude"]
            v_cs_lon = out_ds.variables["speed_contour_longitude"]
            v_cs_lat = out_ds.variables["speed_contour_latitude"]

            v_count.set_auto_mask(False)
            v_processed.set_auto_mask(False)

            # obs write pointer by completed timesteps
            done_mask = v_processed[:] == 1
            obs_ptr = int(v_count[done_mask].sum()) if done_mask.size else 0

            for ti in tqdm(range(start, end + 1), desc="meta4 objects", unit="time"):
                if int(v_processed[ti]) == 1 and not args.overwrite:
                    continue

                dt = _to_py_datetime(tvals[ti])
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

                rows: list[dict[str, Any]] = []

                for i in range(len(cyc)):
                    rows.append(
                        dict(
                            pol=1,
                            lon=float(cyc.lon[i]),
                            lat=float(cyc.lat[i]),
                            amp=float(cyc.amplitude[i]),
                            rs=float(cyc.radius_s[i]),
                            re=float(cyc.radius_e[i]),
                            us=float(cyc.speed_average[i]),
                            ses=float(cyc.shape_error_s[i]),
                            see=float(cyc.shape_error_e[i]),
                            ce_lon=_fit_sample(np.asarray(cyc.contour_lon_e[i]), int(args.sampling)),
                            ce_lat=_fit_sample(np.asarray(cyc.contour_lat_e[i]), int(args.sampling)),
                            cs_lon=_fit_sample(np.asarray(cyc.contour_lon_s[i]), int(args.sampling)),
                            cs_lat=_fit_sample(np.asarray(cyc.contour_lat_s[i]), int(args.sampling)),
                        )
                    )

                for i in range(len(anti)):
                    rows.append(
                        dict(
                            pol=2,
                            lon=float(anti.lon[i]),
                            lat=float(anti.lat[i]),
                            amp=float(anti.amplitude[i]),
                            rs=float(anti.radius_s[i]),
                            re=float(anti.radius_e[i]),
                            us=float(anti.speed_average[i]),
                            ses=float(anti.shape_error_s[i]),
                            see=float(anti.shape_error_e[i]),
                            ce_lon=_fit_sample(np.asarray(anti.contour_lon_e[i]), int(args.sampling)),
                            ce_lat=_fit_sample(np.asarray(anti.contour_lat_e[i]), int(args.sampling)),
                            cs_lon=_fit_sample(np.asarray(anti.contour_lon_s[i]), int(args.sampling)),
                            cs_lat=_fit_sample(np.asarray(anti.contour_lat_s[i]), int(args.sampling)),
                        )
                    )

                n = len(rows)
                if n > 0:
                    sl = slice(obs_ptr, obs_ptr + n)
                    v_ti[sl] = int(ti)
                    v_pol[sl] = np.asarray([r["pol"] for r in rows], dtype=np.uint8)
                    v_lon[sl] = np.asarray([r["lon"] for r in rows], dtype=np.float32)
                    v_lat[sl] = np.asarray([r["lat"] for r in rows], dtype=np.float32)
                    v_amp[sl] = np.asarray([r["amp"] for r in rows], dtype=np.float32)
                    v_rs[sl] = np.asarray([r["rs"] for r in rows], dtype=np.float32)
                    v_re[sl] = np.asarray([r["re"] for r in rows], dtype=np.float32)
                    v_us[sl] = np.asarray([r["us"] for r in rows], dtype=np.float32)
                    v_ses[sl] = np.asarray([r["ses"] for r in rows], dtype=np.float32)
                    v_see[sl] = np.asarray([r["see"] for r in rows], dtype=np.float32)
                    v_ce_lon[sl, :] = np.stack([r["ce_lon"] for r in rows], axis=0)
                    v_ce_lat[sl, :] = np.stack([r["ce_lat"] for r in rows], axis=0)
                    v_cs_lon[sl, :] = np.stack([r["cs_lon"] for r in rows], axis=0)
                    v_cs_lat[sl, :] = np.stack([r["cs_lat"] for r in rows], axis=0)
                    obs_ptr += n

                v_count[ti] = int(n)
                v_processed[ti] = 1

                if int(args.sync_every) > 0 and ((ti - start + 1) % int(args.sync_every) == 0):
                    out_ds.sync()

            out_ds.sync()
    finally:
        clean_ds_dt.close()
        clean_ds.close()

    _log.info("META4 object labels done: %s", args.out_nc)


if __name__ == "__main__":
    main()
