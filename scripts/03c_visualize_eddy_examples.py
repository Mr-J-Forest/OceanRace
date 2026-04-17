"""生成涡旋识别对照图：原场 + mask + 边界 + 中心。"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data_preprocessing.io import open_nc  # noqa: E402
from eddy_detection.dataset import EddySegmentationDataset  # noqa: E402
from eddy_detection.model import EddyUNet  # noqa: E402
from eddy_detection.predictor import infer_batch_to_objects, load_checkpoint  # noqa: E402
from utils.logger import get_logger, setup_logging  # noqa: E402
from utils.visualization_defaults import (  # noqa: E402
    apply_matplotlib_defaults,
    standard_savefig_kwargs,
)

_log = get_logger(__name__)


def _plot_one(
    *,
    adt: np.ndarray,
    pred_mask: np.ndarray,
    objects: list[dict],
    out_path: Path,
    title: str,
    boundary_linewidth: float,
    center_marker_size: float,
) -> None:
    fig, axes2d = plt.subplots(2, 2, figsize=(13.8, 10.2))
    axes = axes2d.flatten()

    # 使用分位数和中位数中心化，避免全图偏红导致蓝色区域不明显。
    finite_mask = np.isfinite(adt)
    if np.any(finite_mask):
        vmin = float(np.nanpercentile(adt, 2.0))
        vmax = float(np.nanpercentile(adt, 98.0))
        vcenter = float(np.nanmedian(adt))
        if not (vmin < vcenter < vmax):
            vcenter = 0.5 * (vmin + vmax)
    else:
        vmin, vmax, vcenter = -1.0, 1.0, 0.0
    norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

    im0 = axes[0].imshow(adt, cmap="RdBu_r", norm=norm, alpha=0.88)
    axes[0].set_title("Raw ADT")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    mask_cmap = ListedColormap(["#f7f7f7", "#b9d3ff", "#f5b6b6"])
    im1 = axes[1].imshow(pred_mask, vmin=0, vmax=2, cmap=mask_cmap)
    axes[1].set_title("Predicted Mask")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(adt, cmap="RdBu_r", norm=norm, alpha=0.86)
    axes[2].set_title("Overlay: Boundary + Center + Rotation")

    # 用掩膜等值线绘制边界，可得到更细、更连续的边界线条。
    cyc = (pred_mask == 1).astype(np.uint8)
    anti = (pred_mask == 2).astype(np.uint8)
    cyc_color = "#2C7FB8"
    anti_color = "#D95F5F"
    if np.any(cyc):
        axes[2].contour(cyc, levels=[0.5], colors=[cyc_color], linewidths=boundary_linewidth)
    if np.any(anti):
        axes[2].contour(anti, levels=[0.5], colors=[anti_color], linewidths=boundary_linewidth)

    for obj in objects:
        cy, cx = obj["center_yx"]
        rot = obj.get("rotation", "unknown")
        c = cyc_color if rot == "cyclonic" else anti_color
        axes[2].scatter([cx], [cy], s=center_marker_size, c=c, marker="x", linewidths=0.8)

    legend_handles = [
        Line2D([0], [0], color=cyc_color, lw=max(1.0, boundary_linewidth + 0.2), label="cyclonic"),
        Line2D([0], [0], color=anti_color, lw=max(1.0, boundary_linewidth + 0.2), label="anticyclonic"),
    ]
    axes[2].legend(handles=legend_handles, loc="upper right", framealpha=0.85)

    im3 = axes[3].imshow(adt, cmap="RdBu_r", norm=norm, alpha=0.9)
    axes[3].set_title("ADT Height (Comparison)")
    levels = np.linspace(float(np.nanmin(adt)), float(np.nanmax(adt)), 10)
    if np.isfinite(levels).all() and (levels[-1] - levels[0] > 1e-8):
        axes[3].contour(adt, levels=levels, colors="white", linewidths=0.35, alpha=0.65)
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xlabel("longitude index")
        ax.set_ylabel("latitude index")

    fig.suptitle(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, **standard_savefig_kwargs())
    plt.close(fig)


def _plot_adt_compare_only(*, adt: np.ndarray, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5.5))
    finite_mask = np.isfinite(adt)
    if np.any(finite_mask):
        vmin = float(np.nanpercentile(adt, 2.0))
        vmax = float(np.nanpercentile(adt, 98.0))
        vcenter = float(np.nanmedian(adt))
        if not (vmin < vcenter < vmax):
            vcenter = 0.5 * (vmin + vmax)
    else:
        vmin, vmax, vcenter = -1.0, 1.0, 0.0
    norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    im = ax.imshow(adt, cmap="RdBu_r", norm=norm, alpha=0.9)
    levels = np.linspace(float(np.nanmin(adt)), float(np.nanmax(adt)), 14)
    if np.isfinite(levels).all() and (levels[-1] - levels[0] > 1e-8):
        ax.contour(adt, levels=levels, colors="white", linewidths=0.45, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("longitude index")
    ax.set_ylabel("latitude index")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="ADT")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, **standard_savefig_kwargs())
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="导出涡旋识别对照图")
    ap.add_argument(
        "--time-split-manifest",
        type=Path,
        default=ROOT / "data/processed/splits/eddy_merged_time.json",
    )
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "outputs/eddy_detection/checkpoints/best.pt",
    )
    ap.add_argument(
        "--norm",
        type=Path,
        default=ROOT / "data/processed/normalization/eddy_norm.json",
    )
    ap.add_argument("--input-steps", type=int, default=1)
    ap.add_argument("--step-stride", type=int, default=8)
    ap.add_argument("--base-channels", type=int, default=32)
    ap.add_argument("--num-samples", type=int, default=12)
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--boundary-linewidth", type=float, default=0.55)
    ap.add_argument("--center-marker-size", type=float, default=14.0)
    ap.add_argument("--save-adt-only", action="store_true", help="额外导出纯ADT高度对比图")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/final_results/eddy_detection/figures",
    )
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    setup_logging(log_file=ROOT / "outputs/eddy_detection/visualize_eddy.log")
    apply_matplotlib_defaults()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")

    norm_path = args.norm if args.norm.is_file() else None
    ds = EddySegmentationDataset(
        split="test",
        input_steps=args.input_steps,
        step_stride=args.step_stride,
        time_split_manifest_path=args.time_split_manifest,
        norm_stats_path=norm_path,
        root=ROOT,
    )
    model = EddyUNet(in_channels=args.input_steps * 3, num_classes=3, base_channels=args.base_channels)
    load_checkpoint(model, args.checkpoint, map_location=args.device)
    model = model.to(args.device)

    n = min(args.num_samples, max(0, len(ds) - args.start_index))
    if n <= 0:
        raise ValueError("no samples to visualize; check --start-index and --num-samples")

    for i in range(args.start_index, args.start_index + n):
        item = ds[i]
        x = item["x"].unsqueeze(0)
        pred = infer_batch_to_objects(model, x, args.device, min_region_pixels=16)[0]

        # 读取原始 ADT 帧用于展示（不标准化）
        p = Path(item["path"])
        t = int(item["time_index"])
        xds = open_nc(p)
        try:
            adt = np.asarray(xds["adt"].values[t], dtype=np.float32)
        finally:
            xds.close()

        objs: list[dict] = []
        for o in pred["objects"]:
            cls = int(o["class_id"])
            rot = "cyclonic" if cls == 1 else "anticyclonic"
            oo = dict(o)
            oo["rotation"] = rot
            objs.append(oo)

        out = args.out_dir / f"sample_{i:05d}_t{t:05d}.png"
        title = f"test sample={i} time_index={t} cyc={pred['cyclonic_count']} anti={pred['anticyclonic_count']}"
        _plot_one(
            adt=adt,
            pred_mask=pred["mask"],
            objects=objs,
            out_path=out,
            title=title,
            boundary_linewidth=max(0.2, float(args.boundary_linewidth)),
            center_marker_size=max(6.0, float(args.center_marker_size)),
        )
        if args.save_adt_only:
            adt_only = args.out_dir / "adt_compare" / f"sample_{i:05d}_t{t:05d}_adt.png"
            _plot_adt_compare_only(adt=adt, out_path=adt_only, title=f"ADT Height Comparison | sample={i} t={t}")
        _log.info("saved figure: %s", out)


if __name__ == "__main__":
    main()
