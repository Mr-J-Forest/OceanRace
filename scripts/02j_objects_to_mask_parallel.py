"""对象级标签并行转换为像素级 mask：分时间段并行 + 最后合并。"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import netCDF4 as nc4
import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]


def _build_ranges(total: int, workers: int) -> list[tuple[int, int]]:
    workers = max(1, min(workers, total))
    base = total // workers
    rem = total % workers
    out: list[tuple[int, int]] = []
    s = 0
    for i in range(workers):
        n = base + (1 if i < rem else 0)
        e = s + n - 1
        out.append((s, e))
        s = e + 1
    return out


def _run_workers(
    clean_nc: Path,
    objects_nc: Path,
    workers: int,
    shard_dir: Path,
    sync_every: int,
    start_index: int,
    end_index: int,
) -> list[Path]:
    ds = xr.open_dataset(clean_nc, decode_times=False)
    try:
        total = int(ds["time"].shape[0])
    finally:
        ds.close()

    s0 = max(0, int(start_index))
    e0 = total - 1 if int(end_index) < 0 else min(total - 1, int(end_index))
    if e0 < s0:
        raise ValueError(f"invalid range: start={s0}, end={e0}, total={total}")

    ranges_local = _build_ranges(e0 - s0 + 1, workers)
    ranges = [(s0 + a, s0 + b) for a, b in ranges_local]

    shard_dir.mkdir(parents=True, exist_ok=True)

    py = ROOT / ".venv-meta4/bin/python"
    if not py.exists():
        py = Path(sys.executable)

    env = os.environ.copy()
    env["HDF5_USE_FILE_LOCKING"] = "FALSE"

    procs: list[tuple[subprocess.Popen[str], object, Path]] = []
    shards: list[Path] = []
    for i, (s, e) in enumerate(ranges):
        shard = shard_dir / f"meta4_mask_shard_{i:02d}.nc"
        logf = shard_dir / f"meta4_mask_shard_{i:02d}.log"
        shards.append(shard)
        cmd = [
            str(py),
            str(ROOT / "scripts/02h_objects_to_pixel_mask.py"),
            "--clean-nc",
            str(clean_nc),
            "--objects-nc",
            str(objects_nc),
            "--out-nc",
            str(shard),
            "--start-index",
            str(s),
            "--end-index",
            str(e),
            "--sync-every",
            str(sync_every),
            "--overwrite",
        ]
        h = open(logf, "w", encoding="utf-8")
        p = subprocess.Popen(cmd, stdout=h, stderr=subprocess.STDOUT, text=True, env=env)
        procs.append((p, h, logf))

    rc = 0
    for p, h, _ in procs:
        r = p.wait()
        h.close()
        if r != 0:
            rc = r

    if rc != 0:
        bad = [str(logf) for p, _, logf in procs if p.returncode != 0]
        raise SystemExit(f"worker failed, logs: {bad}")

    return shards


def _create_var_like(dst: nc4.Dataset, src_var: nc4.Variable, name: str) -> nc4.Variable:
    kwargs = {}
    if hasattr(src_var, "filters"):
        f = src_var.filters()
        if isinstance(f, dict) and f.get("zlib"):
            kwargs["zlib"] = True
            kwargs["complevel"] = int(f.get("complevel", 4))
    fillv = getattr(src_var, "_FillValue", None)
    if fillv is not None:
        var = dst.createVariable(name, src_var.datatype, src_var.dimensions, fill_value=fillv, **kwargs)
    else:
        var = dst.createVariable(name, src_var.datatype, src_var.dimensions, **kwargs)
    for a in src_var.ncattrs():
        if a == "_FillValue":
            continue
        var.setncattr(a, src_var.getncattr(a))
    return var


def _merge_shards(shards: list[Path], out_nc: Path) -> None:
    if not shards:
        raise ValueError("empty shard list")
    shards = sorted(shards)

    if out_nc.exists():
        out_nc.unlink()
    out_nc.parent.mkdir(parents=True, exist_ok=True)

    with nc4.Dataset(shards[0], "r") as s0, nc4.Dataset(out_nc, "w", format="NETCDF4") as dst:
        for name, dim in s0.dimensions.items():
            dst.createDimension(name, len(dim) if not dim.isunlimited() else None)

        for name, var in s0.variables.items():
            dv = _create_var_like(dst, var, name)
            dv[:] = var[:]

        for a in s0.ncattrs():
            dst.setncattr(a, s0.getncattr(a))

    with nc4.Dataset(out_nc, "r+") as dst:
        vp = dst.variables["processed"]
        vc = dst.variables["obs_count"]
        vm = dst.variables["eddy_mask"]
        vi = dst.variables["eddy_instance_id"]
        vp.set_auto_mask(False)

        for sp in shards[1:]:
            with nc4.Dataset(sp, "r") as src:
                spv = src.variables["processed"]
                spv.set_auto_mask(False)
                idx = np.where(spv[:] == 1)[0]
                if idx.size == 0:
                    continue
                vp[idx] = 1
                vc[idx] = src.variables["obs_count"][idx]
                vm[idx, :, :] = src.variables["eddy_mask"][idx, :, :]
                vi[idx, :, :] = src.variables["eddy_instance_id"][idx, :, :]
                dst.sync()


def main() -> None:
    ap = argparse.ArgumentParser(description="并行对象->像素mask")
    ap.add_argument("--clean-nc", type=Path, default=ROOT / "data/processed/eddy_detection/19930101_20241231_clean.nc")
    ap.add_argument("--objects-nc", type=Path, default=ROOT / "data/processed/eddy_detection/labels/19930101_20241231_objects_meta4.nc")
    ap.add_argument("--out-nc", type=Path, default=ROOT / "data/processed/eddy_detection/labels/19930101_20241231_label_meta4_mask.nc")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--end-index", type=int, default=-1)
    ap.add_argument("--sync-every", type=int, default=20)
    ap.add_argument("--shard-dir", type=Path, default=ROOT / "data/processed/eddy_detection/labels/meta4_mask_shards")
    args = ap.parse_args()

    shards = _run_workers(
        clean_nc=args.clean_nc,
        objects_nc=args.objects_nc,
        workers=int(args.workers),
        shard_dir=args.shard_dir,
        sync_every=int(args.sync_every),
        start_index=int(args.start_index),
        end_index=int(args.end_index),
    )
    _merge_shards(shards, args.out_nc)
    print(f"done: {args.out_nc}")


if __name__ == "__main__":
    main()
