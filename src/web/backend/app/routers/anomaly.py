"""异常检测：标签/事件与 manifest 对齐检查；recent_window 与快照格点。"""
from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException

from src.anomaly_detection.dataset import AnomalyFrameDataset

from .. import state
from ..anomaly_inspect import extract_anomaly_sample_fields, get_anomaly_snapshot_dataset
from ..paths import read_path_txt, resolve_path
from ..schemas import AnomalyInspectRequest
from ..time_utils import to_epoch_seconds

router = APIRouter(prefix="/api/anomaly", tags=["anomaly"])


@router.post("/inspect")
def inspect_anomaly(req: AnomalyInspectRequest):
    split = str(req.split).strip().lower()
    if split not in {"train", "val", "test"}:
        raise HTTPException(status_code=400, detail="split must be one of train/val/test")

    labels_path = resolve_path(req.labels_json.strip('"\''))
    events_path = resolve_path(req.events_json.strip('"\''))
    manifest_path = resolve_path(req.manifest_path.strip('"\''))
    processed_input = req.processed_dir.strip('"\'')
    if processed_input in {"", "data/processed/anomaly_detection", "data/processed/anomaly_detection/path.txt"}:
        processed_input = read_path_txt("data/processed/anomaly_detection/path.txt") or processed_input
    processed_dir = resolve_path(processed_input)
    norm_stats_path = resolve_path(req.norm_stats_path.strip('"\''))

    if not os.path.exists(manifest_path):
        raise HTTPException(status_code=404, detail=f"manifest file not found: {manifest_path}")

    labels: list[int] = []
    split_labels_raw: list[Any] = []
    events: list[dict] = []
    labels_mtime = 0.0
    events_mtime = 0.0
    manifest_mtime = float(os.path.getmtime(manifest_path))

    if not bool(req.snapshot_only):
        if not os.path.exists(labels_path):
            raise HTTPException(status_code=404, detail=f"labels file not found: {labels_path}")
        if not os.path.exists(events_path):
            raise HTTPException(status_code=404, detail=f"events file not found: {events_path}")
        labels_mtime = float(os.path.getmtime(labels_path))
        events_mtime = float(os.path.getmtime(events_path))
        try:
            labels_obj = json.loads(open(labels_path, "r", encoding="utf-8").read())
            events_obj = json.loads(open(events_path, "r", encoding="utf-8").read())
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"invalid json: {e}")

        if not isinstance(labels_obj, dict):
            raise HTTPException(status_code=400, detail="labels_json must be an object with split keys")

        split_labels_raw = labels_obj.get(split)
        if not isinstance(split_labels_raw, list):
            raise HTTPException(status_code=400, detail=f"labels_json missing list for split={split}")

        labels = [1 if int(v) == 1 else 0 for v in split_labels_raw]

        if isinstance(events_obj, list):
            for ev in events_obj:
                if not isinstance(ev, dict):
                    continue
                if "start" not in ev or "end" not in ev:
                    continue
                try:
                    start = int(ev["start"])
                    end = int(ev["end"])
                except Exception:
                    continue
                if end < start:
                    continue
                events.append({"name": str(ev.get("name", "event")), "start": start, "end": end})

    cache_key = (processed_dir, manifest_path, split)
    cached_timestamps = state.anomaly_timestamps_cache.get(cache_key)
    if cached_timestamps is None:
        ds = AnomalyFrameDataset(
            processed_anomaly_dir=processed_dir,
            split=split,
            manifest_path=manifest_path,
            norm_stats_path=norm_stats_path if os.path.exists(norm_stats_path) else None,
            open_file_lru_size=max(0, int(req.open_file_lru_size)),
        )
        try:
            timestamps = [to_epoch_seconds(ts) for ts in ds.get_timestamps()]
        finally:
            ds.close()
        state.anomaly_timestamps_cache[cache_key] = timestamps
    else:
        timestamps = cached_timestamps

    if bool(req.snapshot_only):
        labels = [0 for _ in timestamps]
        split_labels_raw = labels

    n_pair = min(len(labels), len(timestamps))
    if n_pair == 0:
        raise HTTPException(status_code=400, detail="empty split after loading labels/timestamps")

    if len(labels) != len(timestamps):
        labels = labels[:n_pair]
        timestamps = timestamps[:n_pair]

    def _hit_events(ts: int) -> list[str]:
        if ts < 0:
            return []
        return [ev["name"] for ev in events if ev["start"] <= ts <= ev["end"]]

    max_points = max(1, int(req.max_points))
    window_hours = max(1, int(req.recent_window_hours))
    latest_timestamp = int(timestamps[-1]) if timestamps else -1

    if bool(req.snapshot_only):
        base_response: dict[str, Any] = {
            "split": split,
            "num_samples": n_pair,
            "num_positive": 0,
            "positive_ratio": 0.0,
            "num_events": 0,
            "matched_positive": 0,
            "matched_positive_ratio": 0.0,
            "matched_event_count": 0,
            "points": [],
            "latest_timestamp": latest_timestamp,
            "recent_window_hours": window_hours,
            "recent_window": [],
            "truncated": False,
            "labels_timestamps_aligned": True,
        }
    else:
        overview_cache_key = (
            state.ANOMALY_OVERVIEW_CACHE_SCHEMA,
            labels_path,
            events_path,
            manifest_path,
            processed_dir,
            norm_stats_path if os.path.exists(norm_stats_path) else None,
            split,
            n_pair,
            max_points,
            window_hours,
            labels_mtime,
            events_mtime,
            manifest_mtime,
            int(timestamps[-1]) if timestamps else -1,
        )
        cached_overview = state.anomaly_overview_cache.get(overview_cache_key)
        if cached_overview is not None:
            base_response = dict(cached_overview)
        else:
            positive_points: list[dict] = []
            matched_positive = 0
            matched_event_names: set[str] = set()

            for idx, (y, ts) in enumerate(zip(labels, timestamps)):
                if y != 1:
                    continue
                hits = _hit_events(ts)
                if hits:
                    matched_positive += 1
                    matched_event_names.update(hits)
                positive_points.append(
                    {
                        "index": idx,
                        "timestamp": ts,
                        "event_hits": hits,
                        "matched": bool(hits),
                    }
                )

            preview = positive_points[:max_points]
            window_start = latest_timestamp - window_hours * 3600
            recent_window: list[dict] = []
            for idx in range(n_pair - 1, -1, -1):
                ts = int(timestamps[idx])
                if ts < window_start and recent_window:
                    break
                hits = _hit_events(ts)
                recent_window.append(
                    {
                        "index": idx,
                        "timestamp": ts,
                        "label": int(labels[idx]),
                        "event_hits": hits,
                        "matched": bool(hits),
                    }
                )
            recent_window.reverse()

            if recent_window:
                ds_recent = get_anomaly_snapshot_dataset(
                    cache=state.anomaly_snapshot_dataset_cache,
                    processed_dir=processed_dir,
                    manifest_path=manifest_path,
                    split=split,
                    norm_stats_path=norm_stats_path if os.path.exists(norm_stats_path) else None,
                    open_file_lru_size=max(8, int(req.open_file_lru_size)),
                )
                for row in recent_window:
                    try:
                        idx = int(row.get("index", -1))
                        if idx < 0:
                            continue
                        sample = ds_recent[idx]
                        fields = extract_anomaly_sample_fields(sample)
                        if not fields:
                            continue
                        row["wind_mean"] = float(fields["wind_mean"])
                        row["wind_p95"] = float(fields["wind_p95"])
                        row["wave_mean"] = float(fields["wave_mean"])
                        row["wave_p95"] = float(fields["wave_p95"])
                    except Exception:
                        continue

            base_response = {
                "split": split,
                "num_samples": n_pair,
                "num_positive": int(sum(labels)),
                "positive_ratio": float(sum(labels) / n_pair),
                "num_events": len(events),
                "matched_positive": matched_positive,
                "matched_positive_ratio": float(matched_positive / max(1, sum(labels))),
                "matched_event_count": len(matched_event_names),
                "points": preview,
                "latest_timestamp": latest_timestamp,
                "recent_window_hours": window_hours,
                "recent_window": recent_window,
                "truncated": len(positive_points) > len(preview),
                "labels_timestamps_aligned": len(split_labels_raw) == len(timestamps),
            }
            state.anomaly_overview_cache[overview_cache_key] = dict(base_response)

    snapshot = None
    if bool(req.include_snapshot) and n_pair > 0:
        if req.snapshot_index is not None:
            snap_idx = max(0, min(int(req.snapshot_index), n_pair - 1))
        else:
            snap_idx = next((i for i, y in enumerate(labels) if int(y) == 1), n_pair // 2)

        snapshot_cache_key = (
            processed_dir,
            manifest_path,
            split,
            norm_stats_path if os.path.exists(norm_stats_path) else None,
            manifest_mtime,
            int(snap_idx),
        )
        cached_snapshot = state.anomaly_snapshot_cache.get(snapshot_cache_key)
        if cached_snapshot is not None:
            snapshot = cached_snapshot
        else:
            ds_snap = get_anomaly_snapshot_dataset(
                cache=state.anomaly_snapshot_dataset_cache,
                processed_dir=processed_dir,
                manifest_path=manifest_path,
                split=split,
                norm_stats_path=norm_stats_path if os.path.exists(norm_stats_path) else None,
                open_file_lru_size=max(8, int(req.open_file_lru_size)),
            )
            try:
                sample = ds_snap[snap_idx]
                fields = extract_anomaly_sample_fields(sample)
                if fields:
                    snapshot = {
                        "index": int(snap_idx),
                        "timestamp": int(timestamps[snap_idx]) if snap_idx < len(timestamps) else -1,
                        "wind_speed": np.nan_to_num(fields["wind_grid"], nan=0.0).tolist(),
                        "wave_swh": np.nan_to_num(fields["wave_grid"], nan=0.0).tolist(),
                        "wind_valid": np.nan_to_num(fields["wind_valid"], nan=0.0).tolist(),
                        "wave_valid": np.nan_to_num(fields["wave_valid"], nan=0.0).tolist(),
                        "wind_mean": float(fields["wind_mean"]),
                        "wind_p95": float(fields["wind_p95"]),
                        "wave_mean": float(fields["wave_mean"]),
                        "wave_p95": float(fields["wave_p95"]),
                    }
            except Exception:
                snapshot = None

            if snapshot is not None:
                state.anomaly_snapshot_cache[snapshot_cache_key] = snapshot
                if len(state.anomaly_snapshot_cache) > state.ANOMALY_SNAPSHOT_CACHE_MAX:
                    oldest_key = next(iter(state.anomaly_snapshot_cache))
                    state.anomaly_snapshot_cache.pop(oldest_key, None)

    response = dict(base_response)
    response["snapshot"] = snapshot
    return response
