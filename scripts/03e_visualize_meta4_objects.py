"""Visualize META4 object-level contours on the raw ADT field."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import netCDF4 as nc4

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data_preprocessing.io import open_nc  # noqa: E402
from utils.logger import setup_logging  # noqa: E402
from utils.visualization_defaults import apply_matplotlib_defaults, standard_savefig_kwargs  # noqa: E402


def _choose_times(counts: np.ndarray, n: int) -> list[int]:
    idx = np.argsort(counts)[::-1]
    return [int(i) for i in idx[:n]]


def _plot_time(clean_nc: Path, objects_nc: Path, t: int, out_path: Path) -> None:
    clean = open_nc(clean_nc)
    obj = nc4.Dataset(objects_nc, "r")
    try:
        adt = np.asarray(clean["adt"][t].values, dtype=np.float32)
        lon = np.asarray(clean["longitude"].values, dtype=np.float32)
        lat = np.asarray(clean["latitude"].values, dtype=np.float32)
        ti = np.asarray(obj.variables["time_index"][:], dtype=np.int64)
        sel = np.where(ti == t)[0]
        vlon = obj.variables["effective_contour_longitude"]
        vlat = obj.variables["effective_contour_latitude"]
        cs_lon = obj.variables["speed_contour_longitude"]
        cs_lat = obj.variables["speed_contour_latitude"]
        center_lon = np.asarray(obj.variables["center_longitude"][:], dtype=np.float32)
        center_lat = np.asarray(obj.variables["center_latitude"][:], dtype=np.float32)
        polarity = np.asarray(obj.variables["polarity"][:], dtype=np.int64)

        fig, ax = plt.subplots(1, 1, figsize=(9.5, 7.0))
        finite = np.isfinite(adt)
        if np.any(finite):
            vmin = float(np.nanpercentile(adt, 2.0))
            vmax = float(np.nanpercentile(adt, 98.0))
            vcenter = float(np.nanmedian(adt))
            if not (vmin < vcenter < vmax):
                vcenter = 0.5 * (vmin + vmax)
        else:
            vmin, vmax, vcenter = -1.0, 1.0, 0.0
        norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
        im = ax.imshow(
            adt,
            cmap="RdBu_r",
            norm=norm,
            origin="lower",
            extent=[float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())],
            aspect="auto",
        )
        seen_cyc = False
        seen_anti = False
        for oi in sel:
            pol = int(polarity[oi])
            color = "#2C7FB8" if pol == 1 else "#D95F5F"
            lbl = "cyclonic" if pol == 1 else "anticyclonic"
            label = None
            if pol == 1 and not seen_cyc:
                label = lbl
                seen_cyc = True
            if pol != 1 and not seen_anti:
                label = lbl
                seen_anti = True
            ax.plot(vlon[oi, :], vlat[oi, :], color=color, lw=1.1, alpha=0.95, label=label)
            ax.plot(cs_lon[oi, :], cs_lat[oi, :], color=color, lw=0.6, ls="--", alpha=0.7)
            ax.scatter([center_lon[oi]], [center_lat[oi]], c=color, s=12, marker="x", linewidths=0.8)

        ax.set_title(f"META4 contours at time_index={t}, objects={len(sel)}")
        ax.set_xlim(float(lon.min()), float(lon.max()))
        ax.set_ylim(float(lat.min()), float(lat.max()))
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
        if seen_cyc or seen_anti:
            ax.legend(loc="upper right", framealpha=0.85)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="ADT")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, **standard_savefig_kwargs())
        plt.close(fig)
    finally:
        obj.close()
        clean.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="visualize META4 object contours")
    ap.add_argument("--clean-nc", type=Path, default=ROOT / "data/processed/eddy_detection/19930101_20241231_clean.nc")
    ap.add_argument("--objects-nc", type=Path, default=ROOT / "data/processed/eddy_detection/labels/19930101_20241231_objects_meta4.nc")
    ap.add_argument("--time-indices", type=int, nargs="*", default=None)
    ap.add_argument("--top-n", type=int, default=4)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "outputs/final_results/eddy_detection/meta4_object_visuals")
    args = ap.parse_args()

    setup_logging(log_file=ROOT / "outputs/logs/meta4_object_visualize.log")
    apply_matplotlib_defaults()

    with nc4.Dataset(args.objects_nc, "r") as ds:
        counts = np.asarray(ds.variables["obs_count"][:], dtype=np.int64)
        times = [int(t) for t in args.time_indices] if args.time_indices else _choose_times(counts, max(1, int(args.top_n)))

    for t in times:
        out = args.out_dir / f"meta4_object_time_{t:05d}.png"
        _plot_time(args.clean_nc, args.objects_nc, t, out)
        print(out)


if __name__ == "__main__":
    main()