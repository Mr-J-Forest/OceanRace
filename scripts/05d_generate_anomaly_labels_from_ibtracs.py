"""Generate anomaly labels/events from IBTrACS tracks for current dataset splits."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from anomaly_detection.dataset import AnomalyFrameDataset
from utils.logger import get_logger, setup_logging

_log = get_logger(__name__)


def _resolve_path(path_like: str | Path | None, default: Path) -> Path:
    if path_like is None:
        return default
    p = Path(path_like)
    return p if p.is_absolute() else (ROOT / p)


def _to_float(v: str | None) -> float | None:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _to_ts(iso_time: str | None) -> int | None:
    if iso_time is None:
        return None
    s = str(iso_time).strip()
    if not s:
        return None
    try:
        dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except Exception:
        return None
    return int(dt.timestamp())


def _to_epoch_seconds(ts_raw: int) -> int:
    """Normalize integer epoch timestamp to seconds.

    Dataset timestamps may be stored as ns/us/ms; IBTrACS timestamps are seconds.
    """

    t = int(ts_raw)
    at = abs(t)
    if at >= 10**17:  # likely nanoseconds
        return t // 10**9
    if at >= 10**14:  # likely microseconds
        return t // 10**6
    if at >= 10**11:  # likely milliseconds
        return t // 10**3
    return t


def _norm_lon(lon: float) -> float:
    return lon + 360.0 if lon < 0 else lon


def _best_wind_kts(row: dict[str, str]) -> float | None:
    for key in ("WMO_WIND", "USA_WIND", "TOKYO_WIND", "CMA_WIND", "HKO_WIND", "KMA_WIND"):
        v = _to_float(row.get(key))
        if v is not None:
            return v
    return None


def _collect_split_timestamps(args: argparse.Namespace) -> dict[str, list[int]]:
    processed_dir = _resolve_path(args.processed_dir, ROOT / "data/processed/anomaly_detection")
    manifest = _resolve_path(args.manifest, ROOT / "data/processed/splits/anomaly_detection.json")
    norm_stats = _resolve_path(args.norm_stats, ROOT / "data/processed/normalization/anomaly_detection_norm.json")

    out: dict[str, list[int]] = {}
    for split in ("train", "val", "test"):
        ds = AnomalyFrameDataset(
            processed_anomaly_dir=processed_dir,
            split=split,
            manifest_path=manifest,
            norm_stats_path=norm_stats if norm_stats.is_file() else None,
            root=ROOT,
            open_file_lru_size=int(args.open_file_lru_size),
        )
        ts_list: list[int] = []
        for i in range(len(ds)):
            sample = ds[i]
            try:
                ts = int(sample.get("timestamp", -1))
            except Exception:
                ts = -1
            ts_list.append(_to_epoch_seconds(ts))
        ds.close()
        out[split] = ts_list
    return out


def _build_events(
    ibtracs_csv: Path,
    ts_min: int,
    ts_max: int,
    *,
    basin: str,
    min_wind_kts: float,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    padding_hours: int,
) -> list[dict[str, Any]]:
    tracks: dict[str, list[int]] = defaultdict(list)

    with ibtracs_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = _to_ts(row.get("ISO_TIME"))
            if ts is None:
                continue
            if ts < ts_min - 86400 or ts > ts_max + 86400:
                continue

            b = str(row.get("BASIN", "")).strip()
            if basin and b and b != basin:
                continue

            lat = _to_float(row.get("LAT"))
            lon = _to_float(row.get("LON"))
            if lat is None or lon is None:
                continue

            lon = _norm_lon(lon)
            if not (lat_min <= lat <= lat_max and lon_min <= lon <= lon_max):
                continue

            wind = _best_wind_kts(row)
            if wind is None or wind < float(min_wind_kts):
                continue

            sid = str(row.get("SID", "")).strip()
            if not sid:
                continue
            tracks[sid].append(ts)

    pad = int(padding_hours) * 3600
    events: list[dict[str, Any]] = []
    for sid, ts_list in tracks.items():
        if not ts_list:
            continue
        start = max(ts_min, min(ts_list) - pad)
        end = min(ts_max, max(ts_list) + pad)
        if end < start:
            continue
        events.append({"name": sid, "start": int(start), "end": int(end)})

    events.sort(key=lambda x: (x["start"], x["end"], x["name"]))

    merged: list[dict[str, Any]] = []
    for ev in events:
        if not merged or ev["start"] > merged[-1]["end"]:
            merged.append({**ev})
            continue
        merged[-1]["end"] = max(merged[-1]["end"], ev["end"])
        merged[-1]["name"] = f"{merged[-1]['name']}|{ev['name']}"
    return merged


def _label_by_events(ts_list: list[int], events: list[dict[str, Any]]) -> list[int]:
    labels: list[int] = []
    for ts in ts_list:
        if ts < 0:
            labels.append(0)
            continue
        flag = 0
        for ev in events:
            if ev["start"] <= ts <= ev["end"]:
                flag = 1
                break
        labels.append(flag)
    return labels


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate anomaly labels/events from IBTrACS")
    ap.add_argument(
        "--ibtracs-csv",
        default="data/raw/external/ibtracs/ibtracs.WP.list.v04r01.csv",
        help="IBTrACS CSV path",
    )
    ap.add_argument("--processed-dir", default=None)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--norm-stats", default=None)
    ap.add_argument("--open-file-lru-size", type=int, default=32)

    ap.add_argument("--basin", default="WP")
    ap.add_argument("--min-wind-kts", type=float, default=34.0)
    ap.add_argument("--lat-min", type=float, default=0.0)
    ap.add_argument("--lat-max", type=float, default=60.0)
    ap.add_argument("--lon-min", type=float, default=100.0)
    ap.add_argument("--lon-max", type=float, default=180.0)
    ap.add_argument("--padding-hours", type=int, default=0, help="expand each event window by +/- hours")

    ap.add_argument("--labels-out", default="outputs/anomaly_detection/labels.json")
    ap.add_argument("--events-out", default="outputs/anomaly_detection/events.json")
    ap.add_argument("--meta-out", default="outputs/anomaly_detection/ibtracs_label_meta.json")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(log_file=ROOT / "outputs/logs/anomaly_generate_ibtracs_labels.log")

    ibtracs_csv = _resolve_path(args.ibtracs_csv, ROOT / "data/raw/external/ibtracs/ibtracs.WP.list.v04r01.csv")
    if not ibtracs_csv.is_file():
        raise SystemExit(f"IBTrACS CSV not found: {ibtracs_csv}")

    _log.info("Collecting split timestamps from processed anomaly dataset...")
    split_ts = _collect_split_timestamps(args)

    all_ts = [t for vals in split_ts.values() for t in vals if t >= 0]
    if not all_ts:
        raise SystemExit("No valid timestamps found in anomaly dataset")
    ts_min, ts_max = int(min(all_ts)), int(max(all_ts))

    _log.info("Building event windows from IBTrACS CSV...")
    events = _build_events(
        ibtracs_csv,
        ts_min,
        ts_max,
        basin=str(args.basin).strip(),
        min_wind_kts=float(args.min_wind_kts),
        lat_min=float(args.lat_min),
        lat_max=float(args.lat_max),
        lon_min=float(args.lon_min),
        lon_max=float(args.lon_max),
        padding_hours=int(args.padding_hours),
    )

    labels: dict[str, list[int]] = {}
    split_stats: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        y = _label_by_events(split_ts[split], events)
        labels[split] = y
        split_stats[split] = {
            "num_samples": int(len(y)),
            "num_positive": int(sum(y)),
            "positive_ratio": float(sum(y) / len(y)) if y else 0.0,
        }

    labels_out = _resolve_path(args.labels_out, ROOT / "outputs/anomaly_detection/labels.json")
    events_out = _resolve_path(args.events_out, ROOT / "outputs/anomaly_detection/events.json")
    meta_out = _resolve_path(args.meta_out, ROOT / "outputs/anomaly_detection/ibtracs_label_meta.json")
    labels_out.parent.mkdir(parents=True, exist_ok=True)

    labels_out.write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
    events_out.write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")
    meta_out.write_text(
        json.dumps(
            {
                "ibtracs_csv": str(ibtracs_csv),
                "time_range": {"min": ts_min, "max": ts_max},
                "config": {
                    "basin": args.basin,
                    "min_wind_kts": float(args.min_wind_kts),
                    "lat_min": float(args.lat_min),
                    "lat_max": float(args.lat_max),
                    "lon_min": float(args.lon_min),
                    "lon_max": float(args.lon_max),
                    "padding_hours": int(args.padding_hours),
                },
                "num_events": int(len(events)),
                "split_stats": split_stats,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    _log.info("Wrote labels: %s", labels_out)
    _log.info("Wrote events: %s", events_out)
    _log.info("Wrote meta: %s", meta_out)


if __name__ == "__main__":
    main()
