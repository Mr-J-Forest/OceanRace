"""META4 对象级标签并行调度：分时间段并行生成 + 最后合并。"""
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
    pet_src: Path,
    workers: int,
    shard_dir: Path,
    common_args: list[str],
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

    procs: list[tuple[subprocess.Popen[str], object, Path]] = []
    shards: list[Path] = []

    env = os.environ.copy()
    env["HDF5_USE_FILE_LOCKING"] = "FALSE"

    for i, (s, e) in enumerate(ranges):
        shard = shard_dir / f"meta4_objects_shard_{i:02d}.nc"
        logf = shard_dir / f"meta4_objects_shard_{i:02d}.log"
        shards.append(shard)
        cmd = [
            str(py),
            str(ROOT / "scripts/02e_generate_meta4_objects.py"),
            "--clean-nc",
            str(clean_nc),
            "--out-nc",
            str(shard),
            "--pet-src",
            str(pet_src),
            "--start-index",
            str(s),
            "--end-index",
            str(e),
            "--overwrite",
            *common_args,
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
        # dimensions
        dst.createDimension("time", len(s0.dimensions["time"]))
        dst.createDimension("NbSample", len(s0.dimensions["NbSample"]))
        dst.createDimension("obs", None)

        # variables
        _create_var_like(dst, s0.variables["time"], "time")
        dst.variables["time"][:] = s0.variables["time"][:]

        _create_var_like(dst, s0.variables["processed"], "processed")
        _create_var_like(dst, s0.variables["obs_count"], "obs_count")
        dst.variables["processed"][:] = 0
        dst.variables["obs_count"][:] = 0

        obs_vars = [k for k in s0.variables.keys() if k not in ("time", "processed", "obs_count")]
        for k in obs_vars:
            _create_var_like(dst, s0.variables[k], k)

        for a in s0.ncattrs():
            dst.setncattr(a, s0.getncattr(a))

    obs_ptr = 0
    with nc4.Dataset(out_nc, "r+") as dst:
        vp = dst.variables["processed"]
        vc = dst.variables["obs_count"]
        vp.set_auto_mask(False)
        vc.set_auto_mask(False)

        obs_vars = [k for k in dst.variables.keys() if k not in ("time", "processed", "obs_count")]

        for sp in shards:
            with nc4.Dataset(sp, "r") as src:
                spv = src.variables["processed"]
                scv = src.variables["obs_count"]
                spv.set_auto_mask(False)
                scv.set_auto_mask(False)
                m = spv[:] == 1
                idx = np.where(m)[0]
                if idx.size:
                    vp[idx] = 1
                    vc[idx] = scv[idx]

                n_obs = len(src.dimensions["obs"])
                if n_obs > 0:
                    sl = slice(obs_ptr, obs_ptr + n_obs)
                    for k in obs_vars:
                        if src.variables[k].ndim == 1:
                            dst.variables[k][sl] = src.variables[k][:]
                        else:
                            dst.variables[k][sl, :] = src.variables[k][:, :]
                    obs_ptr += n_obs
                dst.sync()


def main() -> None:
    ap = argparse.ArgumentParser(description="META4 对象级并行标注")
    ap.add_argument("--clean-nc", type=Path, default=ROOT / "data/processed/eddy_detection/19930101_20241231_clean.nc")
    ap.add_argument("--out-nc", type=Path, default=ROOT / "data/processed/eddy_detection/labels/19930101_20241231_objects_meta4.nc")
    ap.add_argument("--pet-src", type=Path, default=_default_pet_src())
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--end-index", type=int, default=-1)
    ap.add_argument("--shard-dir", type=Path, default=ROOT / "data/processed/eddy_detection/labels/meta4_objects_shards")

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
    ap.add_argument("--sync-every", type=int, default=20)

    args = ap.parse_args()

    common_args = [
        "--step-m", str(args.step_m),
        "--shape-error", str(args.shape_error),
        "--pixel-min", str(args.pixel_min),
        "--pixel-max", str(args.pixel_max),
        "--presampling-multiplier", str(args.presampling_multiplier),
        "--sampling", str(args.sampling),
        "--sampling-method", str(args.sampling_method),
        "--mle", str(args.mle),
        "--nb-step-min", str(args.nb_step_min),
        "--nb-step-to-be-mle", str(args.nb_step_to_be_mle),
        "--sync-every", str(args.sync_every),
    ]

    shards = _run_workers(
        clean_nc=args.clean_nc,
        pet_src=args.pet_src,
        workers=int(args.workers),
        shard_dir=args.shard_dir,
        common_args=common_args,
        start_index=int(args.start_index),
        end_index=int(args.end_index),
    )
    _merge_shards(shards, args.out_nc)
    print(f"done: {args.out_nc}")


if __name__ == "__main__":
    main()
