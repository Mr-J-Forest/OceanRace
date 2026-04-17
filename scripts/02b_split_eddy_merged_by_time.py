"""基于单一合并文件按时间划分涡旋 train/val/test。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data_preprocessing.io import open_nc  # noqa: E402
from utils.logger import get_logger, setup_logging  # noqa: E402

_log = get_logger(__name__)


def _extract_years(time_values: np.ndarray) -> np.ndarray:
    if np.issubdtype(time_values.dtype, np.datetime64):
        return time_values.astype("datetime64[Y]").astype(np.int32) + 1970

    # cftime objects fallback
    try:
        years = np.asarray([int(getattr(t, "year")) for t in time_values], dtype=np.int32)
        return years
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(
            "unable to parse years from time coordinate; please ensure decode_times works"
        ) from exc


def _range_from_mask(mask: np.ndarray, split_name: str) -> tuple[int, int]:
    idx = np.where(mask)[0]
    if idx.size == 0:
        raise ValueError(f"{split_name} split is empty, please adjust year boundaries")
    return int(idx[0]), int(idx[-1])


def main() -> None:
    ap = argparse.ArgumentParser(description="按年份对合并涡旋数据进行时间划分")
    ap.add_argument(
        "--clean-nc",
        type=Path,
        default=ROOT / "data/processed/eddy_detection/19930101_20241231_clean.nc",
        help="合并后的 clean.nc",
    )
    ap.add_argument(
        "--label-nc",
        type=Path,
        default=ROOT / "data/processed/eddy_detection/labels/19930101_20241231_label.nc",
        help="对应像素级标签文件",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data/processed/splits/eddy_merged_time.json",
        help="输出时间划分配置",
    )
    ap.add_argument("--train-end-year", type=int, default=2017)
    ap.add_argument("--val-end-year", type=int, default=2020)
    ap.add_argument("--time-coord", type=str, default="time")
    args = ap.parse_args()

    setup_logging(log_file=ROOT / "outputs/eddy_detection/split_eddy_merged.log")

    if not args.clean_nc.is_file():
        raise FileNotFoundError(f"clean file not found: {args.clean_nc}")
    if not args.label_nc.is_file():
        _log.warning("label file not found yet (you should generate it before training): %s", args.label_nc)

    ds = open_nc(args.clean_nc, decode_times=True)
    try:
        if args.time_coord not in ds.coords:
            raise KeyError(f"missing time coord {args.time_coord!r} in {args.clean_nc}")
        tvals = np.asarray(ds[args.time_coord].values)
    finally:
        ds.close()

    years = _extract_years(tvals)
    y_min, y_max = int(years.min()), int(years.max())
    if args.train_end_year >= args.val_end_year:
        raise ValueError("train-end-year must be < val-end-year")

    train_mask = years <= args.train_end_year
    val_mask = np.logical_and(years > args.train_end_year, years <= args.val_end_year)
    test_mask = years > args.val_end_year

    tr_s, tr_e = _range_from_mask(train_mask, "train")
    va_s, va_e = _range_from_mask(val_mask, "val")
    te_s, te_e = _range_from_mask(test_mask, "test")

    payload = {
        "task": "eddy",
        "mode": "merged_time",
        "time_coord": args.time_coord,
        "clean_nc": args.clean_nc.resolve().relative_to(ROOT.resolve()).as_posix(),
        "label_nc": args.label_nc.resolve().relative_to(ROOT.resolve()).as_posix(),
        "year_range": {"min": y_min, "max": y_max},
        "train": {"start": tr_s, "end": tr_e, "year_start": int(years[tr_s]), "year_end": int(years[tr_e])},
        "val": {"start": va_s, "end": va_e, "year_start": int(years[va_s]), "year_end": int(years[va_e])},
        "test": {"start": te_s, "end": te_e, "year_start": int(years[te_s]), "year_end": int(years[te_e])},
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _log.info("split saved to %s", args.out)
    _log.info(
        "train=[%s,%s] val=[%s,%s] test=[%s,%s] years=%s-%s",
        tr_s,
        tr_e,
        va_s,
        va_e,
        te_s,
        te_e,
        y_min,
        y_max,
    )


if __name__ == "__main__":
    main()
