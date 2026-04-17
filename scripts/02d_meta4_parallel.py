"""META4 标注并行调度：分时间段并行生成 + 最后合并。

说明：
- 每个 worker 调用 `scripts/02c_generate_meta4_labels.py` 处理一个时间段，写入独立 shard 文件。
- 全部 worker 完成后，按 `processed==1` 将 shard 合并到最终标签文件。
"""
from __future__ import annotations

import argparse
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

    local_total = e0 - s0 + 1
    ranges_local = _build_ranges(local_total, workers)
    ranges = [(s0 + a, s0 + b) for a, b in ranges_local]
    shard_dir.mkdir(parents=True, exist_ok=True)

    py = ROOT / ".venv-meta4/bin/python"
    if not py.exists():
        py = Path(sys.executable)

    procs: list[tuple[subprocess.Popen[str], Path, Path]] = []
    shards: list[Path] = []

    for i, (s, e) in enumerate(ranges):
        shard = shard_dir / f"meta4_shard_{i:02d}.nc"
        logf = shard_dir / f"meta4_shard_{i:02d}.log"
        shards.append(shard)
        cmd = [
            str(py),
            str(ROOT / "scripts/02c_generate_meta4_labels.py"),
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
        p = subprocess.Popen(cmd, stdout=h, stderr=subprocess.STDOUT, text=True)
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


def _merge_shards(shards: list[Path], out_nc: Path) -> None:
    if not shards:
        raise ValueError("empty shard list")

    if out_nc.exists():
        out_nc.unlink()
    out_nc.parent.mkdir(parents=True, exist_ok=True)

    # Use first shard as template
    with nc4.Dataset(shards[0], "r") as src0, nc4.Dataset(out_nc, "w", format="NETCDF4") as dst:
        for name, dim in src0.dimensions.items():
            dst.createDimension(name, len(dim) if not dim.isunlimited() else None)

        for name, var in src0.variables.items():
            comp = var.filters() if hasattr(var, "filters") else None
            kwargs = {}
            if isinstance(comp, dict):
                # Preserve zlib settings when available.
                if comp.get("zlib"):
                    kwargs["zlib"] = True
                    kwargs["complevel"] = int(comp.get("complevel", 4))
            fillv = getattr(var, "_FillValue", None)
            if fillv is not None:
                v = dst.createVariable(name, var.datatype, var.dimensions, fill_value=fillv, **kwargs)
            else:
                v = dst.createVariable(name, var.datatype, var.dimensions, **kwargs)
            for a in var.ncattrs():
                if a == "_FillValue":
                    continue
                v.setncattr(a, var.getncattr(a))
            v[:] = var[:]

        for a in src0.ncattrs():
            dst.setncattr(a, src0.getncattr(a))

    # Merge from remaining shards where processed==1
    with nc4.Dataset(out_nc, "r+") as dst:
        vp = dst.variables["processed"]
        vp.set_auto_mask(False)
        vm = dst.variables["eddy_mask"]
        vi = dst.variables["eddy_instance_id"]
        vc = dst.variables["obs_count"]

        for s in shards[1:]:
            with nc4.Dataset(s, "r") as src:
                sp = src.variables["processed"]
                sp.set_auto_mask(False)
                m = sp[:] == 1
                idx = np.where(m)[0]
                if idx.size == 0:
                    continue
                vm[idx, :, :] = src.variables["eddy_mask"][idx, :, :]
                vi[idx, :, :] = src.variables["eddy_instance_id"][idx, :, :]
                vc[idx] = src.variables["obs_count"][idx]
                vp[idx] = 1
                dst.sync()


def main() -> None:
    ap = argparse.ArgumentParser(description="META4 并行标注调度")
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
    )
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--end-index", type=int, default=-1)
    ap.add_argument(
        "--shard-dir",
        type=Path,
        default=ROOT / "data/processed/eddy_detection/labels/meta4_shards",
    )

    # pass-through params for 02c
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
