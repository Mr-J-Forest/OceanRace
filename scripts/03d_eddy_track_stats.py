"""基于识别结果做轨迹级统计：生命周期、半径、数量时序。"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.logger import get_logger, setup_logging  # noqa: E402
from utils.visualization_defaults import apply_matplotlib_defaults, standard_savefig_kwargs  # noqa: E402

_log = get_logger(__name__)


@dataclass
class TrackPoint:
    time_index: int
    center_y: float
    center_x: float
    area: float
    radius_px: float


@dataclass
class Track:
    track_id: int
    rotation: str
    points: list[TrackPoint]


def _radius_from_area(area: float) -> float:
    return float(math.sqrt(max(area, 0.0) / math.pi))


def _distance(a: TrackPoint, b: TrackPoint) -> float:
    return float(math.hypot(a.center_y - b.center_y, a.center_x - b.center_x))


def _area_ratio_ok(prev_area: float, curr_area: float, max_area_ratio: float) -> bool:
    pa = max(float(prev_area), 1.0)
    ca = max(float(curr_area), 1.0)
    ratio = ca / pa
    inv = 1.0 / max(float(max_area_ratio), 1.0)
    return inv <= ratio <= float(max_area_ratio)


def _mutual_nearest_match(
    *,
    active_tracks: list[Track],
    points: list[TrackPoint],
    rots: list[str],
    max_link_dist_px: float,
    max_area_ratio: float,
) -> tuple[list[tuple[int, int]], list[bool]]:
    assigned = [False] * len(points)
    matches: list[tuple[int, int]] = []

    for rotation in ("cyclonic", "anticyclonic"):
        track_candidates: list[int] = [
            i for i, tr in enumerate(active_tracks) if tr.rotation == rotation
        ]
        point_candidates: list[int] = [
            j for j, r in enumerate(rots) if r == rotation and not assigned[j]
        ]
        if not track_candidates or not point_candidates:
            continue

        track_best: dict[int, tuple[int, float]] = {}
        point_best: dict[int, tuple[int, float]] = {}

        for ti in track_candidates:
            last = active_tracks[ti].points[-1]
            best_j = -1
            best_d = 1e18
            for pj in point_candidates:
                p = points[pj]
                if not _area_ratio_ok(last.area, p.area, max_area_ratio):
                    continue
                d = _distance(last, p)
                if d > max_link_dist_px:
                    continue
                if d < best_d:
                    best_d = d
                    best_j = pj
            if best_j >= 0:
                track_best[ti] = (best_j, best_d)

        for pj in point_candidates:
            p = points[pj]
            best_t = -1
            best_d = 1e18
            for ti in track_candidates:
                last = active_tracks[ti].points[-1]
                if not _area_ratio_ok(last.area, p.area, max_area_ratio):
                    continue
                d = _distance(last, p)
                if d > max_link_dist_px:
                    continue
                if d < best_d:
                    best_d = d
                    best_t = ti
            if best_t >= 0:
                point_best[pj] = (best_t, best_d)

        for ti, (pj, _d1) in track_best.items():
            rb = point_best.get(pj)
            if rb is None:
                continue
            best_t_for_point, _d2 = rb
            if best_t_for_point == ti and not assigned[pj]:
                assigned[pj] = True
                matches.append((ti, pj))

    return matches, assigned


def _build_tracks(
    records: list[dict],
    *,
    max_link_dist_px: float,
    max_gap_steps: int,
    index_step: int,
    max_area_ratio: float,
) -> list[Track]:
    by_time: dict[int, list[TrackPoint]] = defaultdict(list)
    rot_by_time: dict[int, list[str]] = defaultdict(list)

    for rec in records:
        t = int(rec["time_index"])
        for ed in rec.get("eddies", []):
            area = float(ed.get("area", 0.0))
            cy, cx = ed["center_yx"]
            by_time[t].append(
                TrackPoint(
                    time_index=t,
                    center_y=float(cy),
                    center_x=float(cx),
                    area=area,
                    radius_px=_radius_from_area(area),
                )
            )
            rot_by_time[t].append(str(ed.get("rotation", "unknown")))

    times = sorted(by_time.keys())
    tracks: list[Track] = []
    active: list[Track] = []
    next_id = 1

    for t in times:
        points = by_time[t]
        rots = rot_by_time[t]
        # 先剔除断轨，再在剩余活动轨迹上做双向最近邻匹配。
        active = [
            tr
            for tr in active
            if t - tr.points[-1].time_index <= max_gap_steps * max(index_step, 1)
        ]

        matches, assigned = _mutual_nearest_match(
            active_tracks=active,
            points=points,
            rots=rots,
            max_link_dist_px=max_link_dist_px,
            max_area_ratio=max_area_ratio,
        )
        for ti, pj in matches:
            active[ti].points.append(points[pj])

        # 新建未匹配目标
        for j, p in enumerate(points):
            if assigned[j]:
                continue
            tr = Track(track_id=next_id, rotation=rots[j], points=[p])
            next_id += 1
            tracks.append(tr)
            active.append(tr)

    return tracks


def _plot_counts(times: list[int], cyc: list[int], anti: list[int], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(times, cyc, label="cyclonic count")
    ax.plot(times, anti, label="anticyclonic count")
    ax.set_title("Eddy Count Time Series")
    ax.set_xlabel("time_index")
    ax.set_ylabel("count")
    ax.legend()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, **standard_savefig_kwargs())
    plt.close(fig)


def _plot_hist(values: Sequence[float], title: str, xlabel: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    if values:
        ax.hist(values, bins=30, alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("frequency")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, **standard_savefig_kwargs())
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="涡旋轨迹统计")
    ap.add_argument(
        "--objects-json",
        type=Path,
        default=ROOT / "outputs/final_results/eddy_detection/test_eddy_objects.json",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/final_results/eddy_detection/tracks",
    )
    ap.add_argument("--step-days", type=int, default=8, help="time_index 步长对应天数")
    ap.add_argument("--max-link-dist-px", type=float, default=18.0)
    ap.add_argument("--max-gap-steps", type=int, default=1)
    ap.add_argument("--max-area-ratio", type=float, default=2.5)
    ap.add_argument("--min-track-steps", type=int, default=2)
    args = ap.parse_args()

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    setup_logging(log_file=ROOT / "outputs/eddy_detection/track_stats.log")
    apply_matplotlib_defaults()

    if not args.objects_json.is_file():
        raise FileNotFoundError(f"missing objects json: {args.objects_json}")

    records = json.loads(args.objects_json.read_text(encoding="utf-8"))
    time_values = sorted({int(r["time_index"]) for r in records})
    if len(time_values) >= 2:
        diffs = [b - a for a, b in zip(time_values[:-1], time_values[1:]) if b > a]
        index_step = int(np.median(diffs)) if diffs else 1
    else:
        index_step = 1

    tracks = _build_tracks(
        records,
        max_link_dist_px=float(args.max_link_dist_px),
        max_gap_steps=int(args.max_gap_steps),
        index_step=index_step,
        max_area_ratio=float(args.max_area_ratio),
    )

    # 过滤过短轨迹
    tracks = [t for t in tracks if len(t.points) >= int(args.min_track_steps)]

    life_steps = [len(t.points) for t in tracks]
    life_days = [int(args.step_days) * s for s in life_steps]
    mean_radius = [float(np.mean([p.radius_px for p in t.points])) for t in tracks]

    # 数量时序
    times = sorted({int(r["time_index"]) for r in records})
    cyc_count = []
    anti_count = []
    rec_by_t = {int(r["time_index"]): r for r in records}
    for t in times:
        rec = rec_by_t[t]
        cyc_count.append(int(rec.get("cyclonic_count", 0)))
        anti_count.append(int(rec.get("anticyclonic_count", 0)))

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "num_tracks": len(tracks),
        "mean_lifecycle_steps": float(np.mean(life_steps)) if life_steps else 0.0,
        "median_lifecycle_steps": float(np.median(life_steps)) if life_steps else 0.0,
        "max_lifecycle_steps": int(np.max(life_steps)) if life_steps else 0,
        "mean_lifecycle_days": float(np.mean(life_days)) if life_days else 0.0,
        "mean_radius_px": float(np.mean(mean_radius)) if mean_radius else 0.0,
        "median_radius_px": float(np.median(mean_radius)) if mean_radius else 0.0,
        "max_radius_px": float(np.max(mean_radius)) if mean_radius else 0.0,
        "step_days": int(args.step_days),
        "tracking": {
            "max_link_dist_px": float(args.max_link_dist_px),
            "max_gap_steps": int(args.max_gap_steps),
            "index_step": int(index_step),
            "max_area_ratio": float(args.max_area_ratio),
            "min_track_steps": int(args.min_track_steps),
        },
    }

    tracks_json = []
    for t in tracks:
        tracks_json.append(
            {
                "track_id": t.track_id,
                "rotation": t.rotation,
                "lifecycle_steps": len(t.points),
                "lifecycle_days": len(t.points) * int(args.step_days),
                "mean_radius_px": float(np.mean([p.radius_px for p in t.points])),
                "points": [
                    {
                        "time_index": p.time_index,
                        "center_yx": [p.center_y, p.center_x],
                        "area": p.area,
                        "radius_px": p.radius_px,
                    }
                    for p in t.points
                ],
            }
        )

    (out_dir / "track_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "tracks.json").write_text(json.dumps(tracks_json, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "count_timeseries.json").write_text(
        json.dumps(
            {
                "time_index": times,
                "cyclonic_count": cyc_count,
                "anticyclonic_count": anti_count,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    _plot_counts(times, cyc_count, anti_count, out_dir / "count_timeseries.png")
    _plot_hist(life_days, "Track Lifecycle Distribution", "lifecycle (days)", out_dir / "lifecycle_hist.png")
    _plot_hist(mean_radius, "Track Radius Distribution", "mean radius (pixel)", out_dir / "radius_hist.png")

    _log.info("track summary saved: %s", out_dir / "track_summary.json")
    _log.info("tracks saved: %s", out_dir / "tracks.json")
    _log.info("figures saved under: %s", out_dir)


if __name__ == "__main__":
    main()
