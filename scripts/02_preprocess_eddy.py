"""中尺度涡旋一键预处理（单脚本自包含）。

流程：
1) clean.nc -> META4 对象级标签
2) 对象级标签 -> 像素级 mask
3) mask 背景 NaN -> 显式 0
4) 生成赛题口径时间切分 manifest（可选）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import netCDF4 as nc4
import numpy as np
from matplotlib.path import Path as MplPath
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.logger import get_logger, setup_logging, tqdm, tqdm_logging

_log = get_logger(__name__)


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


def _ensure_pet_import(pet_src: Path | None) -> Any:
    if pet_src is not None and str(pet_src) not in sys.path:
        sys.path.insert(0, str(pet_src))
    from py_eddy_tracker.dataset.grid import RegularGridDataset  # type: ignore

    return RegularGridDataset


def _to_py_datetime(t: Any) -> datetime:
    if isinstance(t, datetime):
        return t
    if hasattr(t, "year") and hasattr(t, "month") and hasattr(t, "day"):
        return datetime(int(t.year), int(t.month), int(t.day))
    if isinstance(t, np.datetime64):
        sec = int((t - np.datetime64("1970-01-01T00:00:00")) / np.timedelta64(1, "s"))
        return datetime.fromtimestamp(sec, timezone.utc).replace(tzinfo=None)
    return datetime.fromtimestamp(0, timezone.utc).replace(tzinfo=None)


def _fit_sample(arr: np.ndarray, n: int) -> np.ndarray:
    out = np.full((n,), np.nan, dtype=np.float32)
    if arr.size == 0:
        return out
    m = min(n, arr.size)
    out[:m] = arr[:m].astype(np.float32)
    return out


def _create_or_open_objects_output(out_nc: Path, clean_ds: xr.Dataset, sampling: int, overwrite: bool) -> None:
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
        ds.createVariable("effective_contour_longitude", "f4", ("obs", "NbSample"), fill_value=np.nan, **comp)
        ds.createVariable("effective_contour_latitude", "f4", ("obs", "NbSample"), fill_value=np.nan, **comp)
        ds.createVariable("speed_contour_longitude", "f4", ("obs", "NbSample"), fill_value=np.nan, **comp)
        ds.createVariable("speed_contour_latitude", "f4", ("obs", "NbSample"), fill_value=np.nan, **comp)

        ds.setncattr("method", "META4.0-style contour/object labels via py-eddy-tracker")
        ds.setncattr("note", "Object-level observations only, one row per detected eddy per time step")


def _generate_objects(
    *,
    clean_nc: Path,
    out_nc: Path,
    pet_src: Path,
    start_index: int,
    end_index: int,
    overwrite: bool,
    sync_every: int,
    step_m: float,
    shape_error: float,
    pixel_min: int,
    pixel_max: int,
    presampling_multiplier: int,
    sampling: int,
    sampling_method: str,
    mle: int,
    nb_step_min: int,
    nb_step_to_be_mle: int,
) -> None:
    RegularGridDataset = _ensure_pet_import(pet_src)

    clean_ds = xr.open_dataset(clean_nc, decode_times=False)
    clean_ds_dt = xr.open_dataset(clean_nc, decode_times=True)
    try:
        if "time" not in clean_ds.coords:
            raise KeyError("clean.nc must contain time coord")
        if not {"adt", "ugos", "vgos"}.issubset(set(clean_ds.variables)):
            raise KeyError("clean.nc must contain adt/ugos/vgos")

        _create_or_open_objects_output(out_nc, clean_ds, int(sampling), bool(overwrite))

        tvals = np.asarray(clean_ds_dt["time"].values)
        tlen = int(tvals.shape[0])
        start = max(0, int(start_index))
        end = tlen - 1 if int(end_index) < 0 else min(tlen - 1, int(end_index))
        if end < start:
            raise ValueError(f"invalid range: start={start}, end={end}")

        with nc4.Dataset(out_nc, "r+") as out_ds, tqdm_logging():
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

            done_mask = v_processed[:] == 1
            obs_ptr = int(v_count[done_mask].sum()) if done_mask.size else 0

            for ti in tqdm(range(start, end + 1), desc="meta4 objects", unit="time"):
                if int(v_processed[ti]) == 1 and not overwrite:
                    continue

                dt = _to_py_datetime(tvals[ti])
                grid = RegularGridDataset(str(clean_nc), "longitude", "latitude", indexs={"time": int(ti)})

                anti, cyc = grid.eddy_identification(
                    "adt",
                    "ugos",
                    "vgos",
                    dt,
                    step=float(step_m),
                    shape_error=float(shape_error),
                    presampling_multiplier=int(presampling_multiplier),
                    sampling=int(sampling),
                    sampling_method=str(sampling_method),
                    pixel_limit=(int(pixel_min), int(pixel_max)),
                    mle=int(mle),
                    nb_step_min=int(nb_step_min),
                    nb_step_to_be_mle=int(nb_step_to_be_mle),
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
                            ce_lon=_fit_sample(np.asarray(cyc.contour_lon_e[i]), int(sampling)),
                            ce_lat=_fit_sample(np.asarray(cyc.contour_lat_e[i]), int(sampling)),
                            cs_lon=_fit_sample(np.asarray(cyc.contour_lon_s[i]), int(sampling)),
                            cs_lat=_fit_sample(np.asarray(cyc.contour_lat_s[i]), int(sampling)),
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
                            ce_lon=_fit_sample(np.asarray(anti.contour_lon_e[i]), int(sampling)),
                            ce_lat=_fit_sample(np.asarray(anti.contour_lat_e[i]), int(sampling)),
                            cs_lon=_fit_sample(np.asarray(anti.contour_lon_s[i]), int(sampling)),
                            cs_lat=_fit_sample(np.asarray(anti.contour_lat_s[i]), int(sampling)),
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

                if int(sync_every) > 0 and ((ti - start + 1) % int(sync_every) == 0):
                    out_ds.sync()

            out_ds.sync()
    finally:
        clean_ds_dt.close()
        clean_ds.close()


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


def _init_mask_output(out_nc: Path, clean_nc: Path, overwrite: bool) -> None:
    if out_nc.exists() and overwrite:
        out_nc.unlink()
    if out_nc.exists():
        return

    ds = xr.open_dataset(clean_nc, decode_times=False)
    try:
        time_raw = np.asarray(ds["time"].values)
        lat = np.asarray(ds["latitude"].values)
        lon = np.asarray(ds["longitude"].values)
        time_attrs = dict(ds["time"].attrs)
        lat_attrs = dict(ds["latitude"].attrs)
        lon_attrs = dict(ds["longitude"].attrs)
    finally:
        ds.close()

    out_nc.parent.mkdir(parents=True, exist_ok=True)
    with nc4.Dataset(out_nc, "w", format="NETCDF4") as out:
        out.createDimension("time", int(time_raw.shape[0]))
        out.createDimension("latitude", int(lat.shape[0]))
        out.createDimension("longitude", int(lon.shape[0]))

        t = out.createVariable("time", time_raw.dtype, ("time",))
        y = out.createVariable("latitude", lat.dtype, ("latitude",))
        x = out.createVariable("longitude", lon.dtype, ("longitude",))
        t[:] = time_raw
        y[:] = lat
        x[:] = lon
        for k, v in time_attrs.items():
            setattr(t, k, v)
        for k, v in lat_attrs.items():
            setattr(y, k, v)
        for k, v in lon_attrs.items():
            setattr(x, k, v)

        comp = dict(zlib=True, complevel=4)
        out.createVariable("eddy_mask", "u1", ("time", "latitude", "longitude"), fill_value=0, **comp)
        out.createVariable("eddy_instance_id", "i4", ("time", "latitude", "longitude"), fill_value=0, **comp)
        out.createVariable("processed", "u1", ("time",), fill_value=0)
        out.createVariable("obs_count", "i4", ("time",), fill_value=0)


def _objects_to_mask(
    *,
    clean_nc: Path,
    objects_nc: Path,
    out_nc: Path,
    start_index: int,
    end_index: int,
    overwrite: bool,
    sync_every: int,
) -> None:
    _init_mask_output(out_nc, clean_nc, overwrite)

    clean = xr.open_dataset(clean_nc, decode_times=False)
    try:
        lat = np.asarray(clean["latitude"].values, dtype=np.float64)
        lon = np.asarray(clean["longitude"].values, dtype=np.float64)
        tlen = int(clean["time"].shape[0])
    finally:
        clean.close()

    start = max(0, int(start_index))
    end = tlen - 1 if int(end_index) < 0 else min(tlen - 1, int(end_index))
    if end < start:
        raise ValueError(f"invalid range: {start}-{end}")

    with nc4.Dataset(objects_nc, "r") as obj, nc4.Dataset(out_nc, "r+") as out, tqdm_logging():
        v_ti = obj.variables["time_index"]
        v_pol = obj.variables["polarity"]
        v_lon = obj.variables["effective_contour_longitude"]
        v_lat = obj.variables["effective_contour_latitude"]
        ti = np.asarray(v_ti[:], dtype=np.int32)
        pol = np.asarray(v_pol[:], dtype=np.uint8)

        order = np.argsort(ti, kind="mergesort")
        ti_sorted = ti[order]
        counts = np.bincount(ti_sorted, minlength=tlen)
        csum = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))

        vm = out.variables["eddy_mask"]
        vi = out.variables["eddy_instance_id"]
        vp = out.variables["processed"]
        vc = out.variables["obs_count"]
        vp.set_auto_mask(False)

        for t in tqdm(range(start, end + 1), desc="objects->mask", unit="time"):
            if int(vp[t]) == 1 and not overwrite:
                continue

            mask = np.zeros((lat.shape[0], lon.shape[0]), dtype=np.uint8)
            inst = np.zeros((lat.shape[0], lon.shape[0]), dtype=np.int32)
            i0, i1 = int(csum[t]), int(csum[t + 1])
            idxs = order[i0:i1]

            obs_id = 0
            for oi in idxs:
                rr = _rasterize_polygon(lon, lat, v_lon[oi, :], v_lat[oi, :])
                if rr is None:
                    continue
                lat_idx, lon_idx, inside = rr
                sub_m = mask[np.ix_(lat_idx, lon_idx)]
                sub_i = inst[np.ix_(lat_idx, lon_idx)]
                write = np.logical_and(inside, sub_m == 0)
                if np.any(write):
                    obs_id += 1
                    cls = 1 if int(pol[oi]) == 1 else 2
                    sub_m[write] = cls
                    sub_i[write] = obs_id
                    mask[np.ix_(lat_idx, lon_idx)] = sub_m
                    inst[np.ix_(lat_idx, lon_idx)] = sub_i

            vm[t, :, :] = mask
            vi[t, :, :] = inst
            vc[t] = int(obs_id)
            vp[t] = 1

            if int(sync_every) > 0 and ((t - start + 1) % int(sync_every) == 0):
                out.sync()

        out.sync()


def _fix_bg0(input_nc: Path, output_nc: Path, chunk_size: int, overwrite: bool) -> None:
    if output_nc.exists() and not overwrite:
        raise FileExistsError(f"output exists (use --overwrite): {output_nc}")

    ds = xr.open_dataset(input_nc, decode_times=True)
    try:
        if "eddy_mask" not in ds:
            raise KeyError(f"missing variable 'eddy_mask' in {input_nc}")

        tlen = int(ds["eddy_mask"].shape[0])
        lat = np.asarray(ds["latitude"].values)
        lon = np.asarray(ds["longitude"].values)
        time = np.asarray(ds["time"].values)
        chunk = max(1, int(chunk_size))

        mask_out = np.zeros((tlen, lat.shape[0], lon.shape[0]), dtype=np.uint8)
        with tqdm_logging():
            for t0 in tqdm(range(0, tlen, chunk), desc="fix bg NaN->0", unit="chunk"):
                t1 = min(tlen, t0 + chunk)
                arr = np.asarray(ds["eddy_mask"].values[t0:t1], dtype=np.float32)
                arr = np.where(np.isfinite(arr), arr, 0.0)
                arr = np.clip(arr, 0.0, 2.0)
                mask_out[t0:t1] = arr.astype(np.uint8)

        out_ds = xr.Dataset(
            data_vars={"eddy_mask": (("time", "latitude", "longitude"), mask_out)},
            coords={"time": time, "latitude": lat, "longitude": lon},
            attrs={
                **dict(ds.attrs),
                "label_definition": "0=background,1=cyclonic,2=anticyclonic",
                "background_encoding": "explicit_zero",
                "converted_from": str(input_nc),
            },
        )

        output_nc.parent.mkdir(parents=True, exist_ok=True)
        if output_nc.exists():
            output_nc.unlink()

        tmp_nc = Path(tempfile.gettempdir()) / f"{output_nc.stem}.tmp.nc"
        if tmp_nc.exists():
            tmp_nc.unlink()
        out_ds.to_netcdf(tmp_nc, mode="w", encoding={"eddy_mask": {"zlib": True, "complevel": 4, "dtype": "u1"}})
        os.replace(tmp_nc, output_nc)
    finally:
        ds.close()


def _extract_years(vals: np.ndarray) -> np.ndarray:
    if np.issubdtype(vals.dtype, np.datetime64):
        return vals.astype("datetime64[Y]").astype(np.int32) + 1970
    return np.asarray([int(getattr(v, "year")) for v in vals], dtype=np.int32)


def _range(mask: np.ndarray, name: str) -> tuple[int, int]:
    idx = np.where(mask)[0]
    if idx.size == 0:
        raise ValueError(f"{name} split is empty")
    return int(idx[0]), int(idx[-1])


def _write_competition_split(clean_nc: Path, label_nc: Path, out_path: Path) -> None:
    def _to_manifest_path(p: Path) -> str:
        # Prefer project-relative paths, but allow external datasets outside ROOT.
        if not p.is_absolute():
            return p.as_posix()
        p_abs = p.resolve()
        try:
            return p_abs.relative_to(ROOT.resolve()).as_posix()
        except ValueError:
            return p_abs.as_posix()

    ds = xr.open_dataset(clean_nc, decode_times=True)
    try:
        years = _extract_years(np.asarray(ds["time"].values))
    finally:
        ds.close()

    train_mask = years <= 2022
    test_mask = years == 2023
    val_mask = years == 2024
    tr_s, tr_e = _range(train_mask, "train")
    te_s, te_e = _range(test_mask, "test")
    va_s, va_e = _range(val_mask, "val")

    payload = {
        "task": "eddy",
        "mode": "merged_time",
        "time_coord": "time",
        "clean_nc": _to_manifest_path(clean_nc),
        "label_nc": _to_manifest_path(label_nc),
        "year_range": {"min": int(years.min()), "max": int(years.max())},
        "train": {"start": tr_s, "end": tr_e, "year_start": int(years[tr_s]), "year_end": int(years[tr_e])},
        "test": {"start": te_s, "end": te_e, "year_start": int(years[te_s]), "year_end": int(years[te_e])},
        "val": {"start": va_s, "end": va_e, "year_start": int(years[va_s]), "year_end": int(years[va_e])},
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Eddy one-shot preprocess: clean metadata -> META4 objects -> mask -> bg0 "
            "(optional competition split)"
        )
    )
    ap.add_argument(
        "--clean-nc",
        type=Path,
        default=ROOT / "data/processed/eddy_detection/19930101_20241231_clean.nc",
        help="输入 clean.nc（元数据主文件）",
    )
    ap.add_argument(
        "--objects-nc",
        type=Path,
        default=ROOT / "data/processed/eddy_detection/labels/19930101_20241231_objects_meta4.nc",
        help="对象级标签输出",
    )
    ap.add_argument(
        "--mask-nc",
        type=Path,
        default=ROOT / "data/processed/eddy_detection/labels/19930101_20241231_label_meta4_mask.nc",
        help="像素级 mask 输出",
    )
    ap.add_argument(
        "--mask-bg0-nc",
        type=Path,
        default=ROOT / "data/processed/eddy_detection/labels/19930101_20241231_label_meta4_mask_bg0.nc",
        help="背景置 0 后的最终标签输出",
    )
    ap.add_argument(
        "--split-out",
        type=Path,
        default=ROOT / "data/processed/splits/eddy_merged_time_competition.json",
        help="赛题口径时间切分 manifest 输出",
    )
    ap.add_argument(
        "--pet-src",
        type=Path,
        default=_default_pet_src(),
        help="py-eddy-tracker 源码 src 目录（默认自动探测）",
    )
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--end-index", type=int, default=-1)
    ap.add_argument("--sync-every", type=int, default=10)
    ap.add_argument("--chunk-size", type=int, default=256)
    ap.add_argument("--skip-split", action="store_true", help="跳过赛题口径时间切分")
    ap.add_argument("--overwrite", action="store_true", help="各阶段输出存在时覆盖")
    ap.add_argument("--sampling", type=int, default=20)
    ap.add_argument("--step-m", type=float, default=0.002)
    ap.add_argument("--shape-error", type=float, default=70.0)
    ap.add_argument("--pixel-min", type=int, default=5)
    ap.add_argument("--pixel-max", type=int, default=1000)
    ap.add_argument("--presampling-multiplier", type=int, default=10)
    ap.add_argument("--sampling-method", type=str, default="visvalingam", choices=["visvalingam", "uniform"])
    ap.add_argument("--mle", type=int, default=1)
    ap.add_argument("--nb-step-min", type=int, default=2)
    ap.add_argument("--nb-step-to-be-mle", type=int, default=2)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(log_file=ROOT / "outputs/logs/preprocess_eddy.log")

    if not args.clean_nc.is_file():
        raise FileNotFoundError(f"clean file not found: {args.clean_nc}")

    _log.info("[objects] generating object-level labels -> %s", args.objects_nc)
    _generate_objects(
        clean_nc=args.clean_nc,
        out_nc=args.objects_nc,
        pet_src=args.pet_src,
        start_index=int(args.start_index),
        end_index=int(args.end_index),
        overwrite=bool(args.overwrite),
        sync_every=max(1, int(args.sync_every)),
        step_m=float(args.step_m),
        shape_error=float(args.shape_error),
        pixel_min=int(args.pixel_min),
        pixel_max=int(args.pixel_max),
        presampling_multiplier=int(args.presampling_multiplier),
        sampling=int(args.sampling),
        sampling_method=str(args.sampling_method),
        mle=int(args.mle),
        nb_step_min=int(args.nb_step_min),
        nb_step_to_be_mle=int(args.nb_step_to_be_mle),
    )

    _log.info("[mask] rasterizing objects -> %s", args.mask_nc)
    _objects_to_mask(
        clean_nc=args.clean_nc,
        objects_nc=args.objects_nc,
        out_nc=args.mask_nc,
        start_index=int(args.start_index),
        end_index=int(args.end_index),
        overwrite=bool(args.overwrite),
        sync_every=max(1, int(args.sync_every)),
    )

    _log.info("[bg0] fixing background to explicit 0 -> %s", args.mask_bg0_nc)
    _fix_bg0(
        input_nc=args.mask_nc,
        output_nc=args.mask_bg0_nc,
        chunk_size=max(1, int(args.chunk_size)),
        overwrite=bool(args.overwrite),
    )

    if not args.skip_split:
        _log.info("[split] writing competition manifest -> %s", args.split_out)
        _write_competition_split(args.clean_nc, args.mask_bg0_nc, args.split_out)

    _log.info(
        "eddy preprocess done: objects=%s mask=%s bg0=%s split=%s",
        args.objects_nc,
        args.mask_nc,
        args.mask_bg0_nc,
        "skipped" if args.skip_split else args.split_out,
    )


if __name__ == "__main__":
    main()
