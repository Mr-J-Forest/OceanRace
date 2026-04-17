"""将 META4 对象级标签转换为像素级训练 mask。"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import netCDF4 as nc4
import numpy as np
from matplotlib.path import Path as MplPath
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.logger import get_logger, setup_logging, tqdm, tqdm_logging  # noqa: E402

_log = get_logger(__name__)


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


def _init_out(out_nc: Path, clean_nc: Path, overwrite: bool) -> None:
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

        out.setncattr("method", "META4.0 object-to-pixel rasterization")
        out.setncattr("label_definition", "eddy_mask:0=background,1=cyclonic,2=anticyclonic")


def main() -> None:
    ap = argparse.ArgumentParser(description="对象级标签转换像素级 mask")
    ap.add_argument(
        "--clean-nc",
        type=Path,
        default=ROOT / "data/processed/eddy_detection/19930101_20241231_clean.nc",
    )
    ap.add_argument(
        "--objects-nc",
        type=Path,
        default=ROOT / "data/processed/eddy_detection/labels/19930101_20241231_objects_meta4.nc",
    )
    ap.add_argument(
        "--out-nc",
        type=Path,
        default=ROOT / "data/processed/eddy_detection/labels/19930101_20241231_label_meta4_mask.nc",
    )
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--end-index", type=int, default=-1)
    ap.add_argument("--sync-every", type=int, default=10)
    args = ap.parse_args()

    setup_logging(log_file=ROOT / "outputs/logs/meta4_objects_to_mask.log")

    if not args.clean_nc.is_file():
        raise FileNotFoundError(args.clean_nc)
    if not args.objects_nc.is_file():
        raise FileNotFoundError(args.objects_nc)

    _init_out(args.out_nc, args.clean_nc, bool(args.overwrite))

    clean = xr.open_dataset(args.clean_nc, decode_times=False)
    try:
        lat = np.asarray(clean["latitude"].values, dtype=np.float64)
        lon = np.asarray(clean["longitude"].values, dtype=np.float64)
        tlen = int(clean["time"].shape[0])
    finally:
        clean.close()

    start = max(0, int(args.start_index))
    end = tlen - 1 if int(args.end_index) < 0 else min(tlen - 1, int(args.end_index))
    if end < start:
        raise ValueError(f"invalid range: {start}-{end}")

    with nc4.Dataset(args.objects_nc, "r") as obj, nc4.Dataset(args.out_nc, "r+") as out, tqdm_logging():
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
            if int(vp[t]) == 1 and not args.overwrite:
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

            if int(args.sync_every) > 0 and ((t - start + 1) % int(args.sync_every) == 0):
                out.sync()

        out.sync()

    _log.info("done: %s", args.out_nc)


if __name__ == "__main__":
    main()
