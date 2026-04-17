"""将 META4 像素标签中的背景 NaN 修复为显式 0。

用途：
- 历史文件 `*_label_meta4_mask.nc` 可能将背景存为 NaN。
- 该脚本导出新的紧凑标签文件：仅保留 `eddy_mask`，并保证标签为 0/1/2。

示例：
  python scripts/02h_fix_meta4_mask_background.py
  python scripts/02h_fix_meta4_mask_background.py \
    --input-nc data/processed/eddy_detection/labels/19930101_20241231_label_meta4_mask.nc \
    --output-nc data/processed/eddy_detection/labels/19930101_20241231_label_meta4_mask_bg0.nc
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from utils.logger import get_logger, setup_logging, tqdm, tqdm_logging  # noqa: E402

_log = get_logger(__name__)


def _default_output(input_nc: Path) -> Path:
    stem = input_nc.stem
    if stem.endswith("_meta4_mask"):
        stem = stem + "_bg0"
    else:
        stem = stem + "_bg0"
    return input_nc.with_name(stem + input_nc.suffix)


def main() -> None:
    ap = argparse.ArgumentParser(description="修复 META4 标签背景：NaN -> 0")
    ap.add_argument(
        "--input-nc",
        type=Path,
        default=ROOT / "data/processed/eddy_detection/labels/19930101_20241231_label_meta4_mask.nc",
    )
    ap.add_argument("--output-nc", type=Path, default=None)
    ap.add_argument("--chunk-size", type=int, default=256, help="按时间分块处理，降低内存占用")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    setup_logging(log_file=ROOT / "outputs/logs/fix_meta4_mask_background.log")

    input_nc = args.input_nc
    output_nc = args.output_nc or _default_output(input_nc)

    if not input_nc.is_file():
        raise FileNotFoundError(f"input not found: {input_nc}")
    if output_nc.exists() and not args.overwrite:
        raise FileExistsError(f"output exists (use --overwrite): {output_nc}")

    ds = xr.open_dataset(input_nc, decode_times=True)
    try:
        if "eddy_mask" not in ds:
            raise KeyError(f"missing variable 'eddy_mask' in {input_nc}")

        tlen = int(ds["eddy_mask"].shape[0])
        lat = np.asarray(ds["latitude"].values)
        lon = np.asarray(ds["longitude"].values)
        time = np.asarray(ds["time"].values)
        chunk = max(1, int(args.chunk_size))

        mask_out = np.zeros((tlen, lat.shape[0], lon.shape[0]), dtype=np.uint8)
        with tqdm_logging():
            for t0 in tqdm(range(0, tlen, chunk), desc="fix bg NaN->0", unit="chunk"):
                t1 = min(tlen, t0 + chunk)
                arr = np.asarray(ds["eddy_mask"].values[t0:t1], dtype=np.float32)
                arr = np.where(np.isfinite(arr), arr, 0.0)
                arr = np.clip(arr, 0.0, 2.0)
                mask_out[t0:t1] = arr.astype(np.uint8)

        out_ds = xr.Dataset(
            data_vars={
                "eddy_mask": (("time", "latitude", "longitude"), mask_out),
            },
            coords={
                "time": time,
                "latitude": lat,
                "longitude": lon,
            },
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
        comp = {"zlib": True, "complevel": 4, "dtype": "u1"}
        out_ds.to_netcdf(tmp_nc, mode="w", encoding={"eddy_mask": comp})
        os.replace(tmp_nc, output_nc)

        _log.info("saved fixed mask: %s", output_nc)
        _log.info("shape=(%s,%s,%s)", tlen, lat.shape[0], lon.shape[0])
    finally:
        ds.close()


if __name__ == "__main__":
    main()
