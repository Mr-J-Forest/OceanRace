"""按赛题口径划分时间索引：
- train: 1993-01-01 ~ 2022-12-31
- test : 2023-01-01 ~ 2023-12-31
- val  : 2024-01-01 ~ 2024-12-31
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]


def _extract_years(vals: np.ndarray) -> np.ndarray:
    if np.issubdtype(vals.dtype, np.datetime64):
        return vals.astype("datetime64[Y]").astype(np.int32) + 1970
    return np.asarray([int(getattr(v, "year")) for v in vals], dtype=np.int32)


def _range(mask: np.ndarray, name: str) -> tuple[int, int]:
    idx = np.where(mask)[0]
    if idx.size == 0:
        raise ValueError(f"{name} split is empty")
    return int(idx[0]), int(idx[-1])


def main() -> None:
    ap = argparse.ArgumentParser(description="生成赛题口径时间划分 manifest")
    ap.add_argument(
        "--clean-nc",
        type=Path,
        default=ROOT / "data/processed/eddy_detection/19930101_20241231_clean.nc",
    )
    ap.add_argument(
        "--label-nc",
        type=Path,
        default=ROOT / "data/processed/eddy_detection/labels/19930101_20241231_label_meta4_mask.nc",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data/processed/splits/eddy_merged_time_competition.json",
    )
    args = ap.parse_args()

    ds = xr.open_dataset(args.clean_nc, decode_times=True)
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
        "clean_nc": args.clean_nc.resolve().relative_to(ROOT.resolve()).as_posix(),
        "label_nc": args.label_nc.resolve().relative_to(ROOT.resolve()).as_posix(),
        "year_range": {"min": int(years.min()), "max": int(years.max())},
        "train": {"start": tr_s, "end": tr_e, "year_start": int(years[tr_s]), "year_end": int(years[tr_e])},
        "test": {"start": te_s, "end": te_e, "year_start": int(years[te_s]), "year_end": int(years[te_e])},
        "val": {"start": va_s, "end": va_e, "year_start": int(years[va_s]), "year_end": int(years[va_e])},
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
