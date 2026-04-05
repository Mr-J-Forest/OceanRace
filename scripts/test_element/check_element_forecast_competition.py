"""要素预测比赛检查（单文件模式）。

功能：
1. 仅使用单个 NetCDF 文件构建窗口样本（不走 manifest）。
2. 单次前向输出若不足 72h，自动循环滚动回灌预测到目标时长。
3. 统计整体指标 + 各预测步指标，并与比赛阈值对比。
4. 输出 JSON/CSV/Markdown 报告与多张可视化图片。

示例：
  python scripts/test_element/check_element_forecast_competition.py \
    --data-file data/processed/element_forecasting/path.txt \
    --split test --time-step-hours 1 --eval-horizon-hours 72
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from element_forecasting.dataset import ElementForecastWindowDataset
from element_forecasting.predictor import ElementForecastPredictor
from utils.dataset_utils import destandardize_tensor, load_norm_stats
from utils.logger import get_logger, setup_logging, tqdm, tqdm_logging
from utils.visualization_defaults import (
    DEFAULT_CMAP_DIVERGING,
    apply_matplotlib_defaults,
    standard_savefig_kwargs,
)

_log = get_logger(__name__)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return cfg if isinstance(cfg, dict) else {}


def _resolve_path(value: str | Path | None, *, default: str | Path | None = None) -> Path | None:
    raw = value if value is not None else default
    if raw is None:
        return None
    p = Path(raw)
    if not p.is_absolute():
        p = ROOT / p
    return p


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "x": torch.stack([b["x"] for b in batch], dim=0),
        "y": torch.stack([b["y"] for b in batch], dim=0),
        "y_valid": torch.stack([b["y_valid"] for b in batch], dim=0),
    }


def _destandardize_batch(
    y_std: torch.Tensor,
    var_names: tuple[str, ...],
    norm: dict[str, tuple[float, float]] | None,
) -> torch.Tensor:
    # y_std: [B, T_out, C, H, W]
    if norm is None:
        return y_std
    out = y_std.clone()
    n_channels = int(out.shape[2])
    for c, key in enumerate(var_names[:n_channels]):
        out[:, :, c, :, :] = destandardize_tensor(out[:, :, c, :, :], key, norm)
    return out


def _roll_forecast_std(
    predictor: ElementForecastPredictor,
    x_std: torch.Tensor,
    target_steps: int,
    overlap_steps: int,
    enable_overlap_blend: bool,
) -> torch.Tensor:
    """滚动预测（标准化空间）：支持重叠融合以降低块边界跳变。"""
    out = predictor.predict_long_horizon(
        x=x_std,
        target_steps=target_steps,
        overlap_steps=overlap_steps,
        enable_overlap_blend=enable_overlap_blend,
        denormalize=False,
        return_cpu=False,
    )
    pred_std = out["pred"].float()
    if pred_std.ndim != 5:
        raise RuntimeError("predictor output shape is invalid")
    return pred_std


def _build_single_file_subset(
    full_ds: ElementForecastWindowDataset,
    split: str,
    train_ratio: float,
    val_ratio: float,
) -> tuple[torch.utils.data.Dataset, dict[str, int]]:
    n_total = len(full_ds)
    if n_total <= 0:
        raise SystemExit("single-file dataset is empty")

    if split == "all":
        return full_ds, {"total": n_total, "train": n_total, "val": 0, "test": 0, "selected": n_total}

    if not (0.0 < train_ratio < 1.0):
        raise SystemExit("train_ratio (or single_file_train_ratio) must be in (0,1)")
    if not (0.0 <= val_ratio < 1.0):
        raise SystemExit("val_ratio (or single_file_val_ratio) must be in [0,1)")
    if train_ratio + val_ratio >= 1.0:
        raise SystemExit("train_ratio + val_ratio must be < 1")

    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val

    if split == "train":
        indices = list(range(0, n_train))
    elif split == "val":
        indices = list(range(n_train, n_train + n_val))
    elif split == "test":
        indices = list(range(n_train + n_val, n_total))
    else:
        raise SystemExit(f"unsupported split: {split}")

    if not indices:
        raise SystemExit(
            f"split={split} has 0 windows under current ratios; total={n_total}, train={n_train}, val={n_val}, test={n_test}"
        )

    return Subset(full_ds, indices), {
        "total": n_total,
        "train": n_train,
        "val": n_val,
        "test": n_test,
        "selected": len(indices),
    }


def _write_per_horizon_csv(
    path: Path,
    horizon_hours: np.ndarray,
    mse_h: np.ndarray,
    rmse_h: np.ndarray,
    mae_h: np.ndarray,
    rel_mse_pct_h: np.ndarray,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "horizon_hours", "mse", "rmse", "mae", "relative_mse_percent"])
        for i in range(horizon_hours.shape[0]):
            w.writerow(
                [
                    i + 1,
                    float(horizon_hours[i]),
                    float(mse_h[i]),
                    float(rmse_h[i]),
                    float(mae_h[i]),
                    float(rel_mse_pct_h[i]),
                ]
            )


def _plot_horizon_metrics(
    fig_path: Path,
    horizon_hours: np.ndarray,
    rmse_h: np.ndarray,
    mae_h: np.ndarray,
    rel_mse_pct_h: np.ndarray,
    mse_percent_threshold: float,
) -> None:
    import matplotlib.pyplot as plt

    apply_matplotlib_defaults()
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    ax0 = axes[0]
    ax0.plot(horizon_hours, rmse_h, marker="o", label="RMSE")
    ax0.plot(horizon_hours, mae_h, marker="s", label="MAE")
    ax0.set_ylabel("Error")
    ax0.set_title("Per-Horizon Error Curves")
    ax0.legend(loc="best")

    ax1 = axes[1]
    ax1.plot(horizon_hours, rel_mse_pct_h, marker="o", label="Relative MSE(%)")
    ax1.axhline(float(mse_percent_threshold), color="tab:red", linestyle="--", label="Threshold")
    ax1.set_xlabel("Forecast Horizon (hours)")
    ax1.set_ylabel("Relative MSE (%)")
    ax1.set_title("Per-Horizon Relative MSE vs Threshold")
    ax1.legend(loc="best")

    fig.savefig(fig_path, **standard_savefig_kwargs())
    plt.close(fig)


def _plot_competition_comparison(
    fig_path: Path,
    horizon_hours: float,
    horizon_threshold: float,
    rel_mse_percent: float,
    mse_percent_threshold: float,
) -> None:
    import matplotlib.pyplot as plt

    apply_matplotlib_defaults()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    ax0 = axes[0]
    ax0.bar(["actual", "threshold"], [horizon_hours, horizon_threshold], color=["#4C72B0", "#55A868"])
    ax0.set_title("Horizon Requirement")
    ax0.set_ylabel("Hours")

    ax1 = axes[1]
    ax1.bar(["actual", "threshold"], [rel_mse_percent, mse_percent_threshold], color=["#DD8452", "#C44E52"])
    ax1.set_title("Relative MSE Requirement")
    ax1.set_ylabel("Percent")

    fig.savefig(fig_path, **standard_savefig_kwargs())
    plt.close(fig)


def _plot_sample_timeseries(
    fig_path: Path,
    sample_pred: torch.Tensor,
    sample_target: torch.Tensor,
    sample_mask: torch.Tensor,
    var_names: tuple[str, ...],
    time_step_hours: float,
    max_plot_vars: int,
) -> None:
    import matplotlib.pyplot as plt

    apply_matplotlib_defaults()
    eps = 1e-12
    pred = sample_pred.float()
    target = sample_target.float()
    mask = sample_mask.float()

    valid_hw = torch.sum(mask, dim=(2, 3)).clamp_min(eps)
    pred_mean = torch.sum(pred * mask, dim=(2, 3)) / valid_hw
    target_mean = torch.sum(target * mask, dim=(2, 3)) / valid_hw

    t_steps = int(pred.shape[0])
    hours = (np.arange(t_steps, dtype=np.float64) + 1.0) * float(time_step_hours)
    n_channels = int(pred.shape[1])
    n_plot = max(1, min(max_plot_vars, n_channels))

    fig, axes = plt.subplots(n_plot, 1, figsize=(12, 3.2 * n_plot), sharex=True)
    if n_plot == 1:
        axes = [axes]

    for c in range(n_plot):
        name = var_names[c] if c < len(var_names) else f"var_{c}"
        ax = axes[c]
        ax.plot(hours, target_mean[:, c].cpu().numpy(), label=f"target-{name}")
        ax.plot(hours, pred_mean[:, c].cpu().numpy(), label=f"pred-{name}")
        ax.set_ylabel("Spatial Mean")
        ax.legend(loc="best")

    axes[-1].set_xlabel("Forecast Horizon (hours)")
    fig.suptitle("Sample Forecast vs Target (Spatial Mean)")
    fig.savefig(fig_path, **standard_savefig_kwargs())
    plt.close(fig)


def _plot_sample_map(
    fig_path: Path,
    sample_pred: torch.Tensor,
    sample_target: torch.Tensor,
    sample_mask: torch.Tensor,
    var_name: str,
    time_step_hours: float,
    channel_idx: int = 0,
) -> None:
    import matplotlib.pyplot as plt

    apply_matplotlib_defaults()

    pred = sample_pred[-1, channel_idx].float().cpu().numpy()
    target = sample_target[-1, channel_idx].float().cpu().numpy()
    mask_c = channel_idx if sample_mask.shape[1] > channel_idx else 0
    mask = sample_mask[-1, mask_c].float().cpu().numpy() > 0.5

    pred = np.where(mask, pred, np.nan)
    target = np.where(mask, target, np.nan)
    err = np.where(mask, pred - target, np.nan)

    finite_vals = np.concatenate([pred[np.isfinite(pred)], target[np.isfinite(target)]])
    if finite_vals.size > 0:
        vmin = float(np.min(finite_vals))
        vmax = float(np.max(finite_vals))
    else:
        vmin, vmax = -1.0, 1.0

    finite_err = err[np.isfinite(err)]
    if finite_err.size > 0:
        emax = float(np.max(np.abs(finite_err)))
    else:
        emax = 1.0

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    hour = sample_pred.shape[0] * float(time_step_hours)

    im0 = axes[0].imshow(target, vmin=vmin, vmax=vmax)
    axes[0].set_title(f"Target {var_name} (t+{hour:.1f}h)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(pred, vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Pred {var_name} (t+{hour:.1f}h)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(err, cmap=DEFAULT_CMAP_DIVERGING, vmin=-emax, vmax=emax)
    axes[2].set_title("Pred - Target")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.savefig(fig_path, **standard_savefig_kwargs())
    plt.close(fig)


def _write_markdown_summary(
    path: Path,
    report: dict[str, Any],
    figure_files: list[str],
    per_horizon_csv: str,
) -> None:
    lines: list[str] = []
    lines.append("# Element Forecast Competition Check (Single File)")
    lines.append("")
    lines.append(f"- Verdict: {'PASS' if report['pass']['overall'] else 'FAIL'}")
    lines.append(f"- Horizon: {report['model']['horizon_hours']:.2f} h (threshold >= {report['competition_thresholds']['horizon_hours_min']:.2f} h)")
    lines.append(
        f"- Relative MSE: {report['metrics']['relative_mse_percent']:.4f}% "
        f"(threshold <= {report['competition_thresholds']['mse_percent_max']:.4f}%)"
    )
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append(f"- MSE: {report['metrics']['mse']:.6f}")
    lines.append(f"- RMSE: {report['metrics']['rmse']:.6f}")
    lines.append(f"- MAE: {report['metrics']['mae']:.6f}")
    lines.append(f"- NSE: {report['metrics']['nse']:.6f}")
    lines.append(f"- Relative MSE(%): {report['metrics']['relative_mse_percent']:.6f}")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append(f"- Per-horizon metrics CSV: {per_horizon_csv}")
    for f in figure_files:
        lines.append(f"- Figure: {f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Element forecasting competition check (single file mode)")
    ap.add_argument("--data-config", type=Path, default=ROOT / "configs/data_config.yaml")
    ap.add_argument("--train-config", type=Path, default=ROOT / "configs/element_forecasting/train.yaml")
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "outputs/element_forecasting/checkpoints/hybrid_best.pt")
    ap.add_argument("--data-file", type=str, default=None, help="单一 NetCDF 文件路径（必需；默认读取 train.yaml 的 data_file）")
    ap.add_argument("--processed-dir", type=str, default=None)
    ap.add_argument("--norm", type=str, default=None)
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test", "all"])
    ap.add_argument("--single-file-train-ratio", type=float, default=None)
    ap.add_argument("--single-file-val-ratio", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--time-step-hours", type=float, default=1.0, help="每个时间步代表多少小时")
    ap.add_argument("--eval-horizon-hours", type=float, default=None, help="滚动预测目标时长（小时）。默认自动取模型自身输出长度与比赛目标之间的最大值")
    ap.add_argument("--horizon-hours-threshold", type=float, default=72.0)
    ap.add_argument("--mse-percent-threshold", type=float, default=15.0)
    ap.add_argument("--open-file-lru-size", type=int, default=16)
    ap.add_argument("--overlap-steps", type=int, default=None, help="滚动推理重叠步数（默认读取 train.yaml，缺省 4）")
    ap.add_argument(
        "--enable-overlap-blend",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="是否启用重叠线性融合（默认读取 train.yaml，缺省开启）",
    )
    ap.add_argument("--max-batches", type=int, default=0, help="仅评估前 N 个 batch；0 表示全部")
    ap.add_argument("--max-plot-vars", type=int, default=4)
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/final_results/element_forecasting/competition_single_file",
    )
    args = ap.parse_args()

    log_file = ROOT / "outputs/logs/element_forecasting_competition_check.log"
    setup_logging(log_file=log_file)

    data_cfg = _load_yaml(args.data_config)
    train_cfg = _load_yaml(args.train_config)

    ckpt = _resolve_path(args.checkpoint)
    if ckpt is None or not ckpt.is_file():
        raise SystemExit(f"checkpoint not found: {ckpt}")

    output_dir = _resolve_path(args.output_dir)
    if output_dir is None:
        raise SystemExit("output_dir is invalid")
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    norm_path = _resolve_path(
        args.norm,
        default=(
            train_cfg.get("norm_stats_path")
            or data_cfg.get("artifacts", {})
            .get("normalization_files", {})
            .get("element_forecasting", "data/processed/normalization/element_forecasting_norm.json")
        ),
    )
    norm = load_norm_stats(norm_path) if norm_path is not None and norm_path.is_file() else None

    predictor = ElementForecastPredictor(
        checkpoint_path=ckpt,
        device=args.device,
        norm_stats_path=norm_path,
    )
    var_names = predictor.var_names
    if not var_names:
        cfg_names = tuple(train_cfg.get("var_names", []))
        if not cfg_names:
            raise SystemExit("var_names missing in checkpoint and train config")
        var_names = cfg_names

    input_steps = int(predictor.input_steps)
    model_output_steps = int(predictor.output_steps)

    if args.eval_horizon_hours is not None and args.eval_horizon_hours > 0:
        eval_steps = max(1, int(round(float(args.eval_horizon_hours) / float(args.time_step_hours))))
    else:
        req_steps = max(1, int(math.ceil(float(args.horizon_hours_threshold) / float(args.time_step_hours))))
        eval_steps = max(model_output_steps, req_steps)

    rolling_iters = int(math.ceil(eval_steps / max(model_output_steps, 1)))
    overlap_steps = int(args.overlap_steps if args.overlap_steps is not None else train_cfg.get("overlap_steps", 4))
    overlap_steps = max(0, overlap_steps)
    if args.enable_overlap_blend is None:
        enable_overlap_blend = bool(train_cfg.get("overlap_blend_enabled", True))
    else:
        enable_overlap_blend = bool(args.enable_overlap_blend)

    data_file = _resolve_path(args.data_file, default=train_cfg.get("data_file"))
    if data_file is None:
        raise SystemExit("single-file mode requires --data-file or train.yaml:data_file")
    if not data_file.is_file():
        raise SystemExit(f"single data file not found: {data_file}")

    train_ratio = float(
        args.single_file_train_ratio
        if args.single_file_train_ratio is not None
        else train_cfg.get("train_ratio", train_cfg.get("single_file_train_ratio", 0.8))
    )
    val_ratio = float(
        args.single_file_val_ratio
        if args.single_file_val_ratio is not None
        else train_cfg.get("val_ratio", train_cfg.get("single_file_val_ratio", 0.2))
    )

    full_ds = ElementForecastWindowDataset(
        data_file=data_file,
        var_names=var_names,
        input_steps=input_steps,
        output_steps=eval_steps,
        window_stride=int(train_cfg.get("window_stride", 1)),
        open_file_lru_size=max(1, int(args.open_file_lru_size)),
        split=None,
        norm_stats_path=norm_path,
        root=ROOT,
    )
    ds, split_stats = _build_single_file_subset(full_ds, args.split, train_ratio, val_ratio)

    batch_size = int(args.batch_size if args.batch_size is not None else train_cfg.get("batch_size", 8))
    num_workers = int(args.num_workers if args.num_workers is not None else train_cfg.get("num_workers", 0))
    loader = DataLoader(
        ds,
        batch_size=max(1, batch_size),
        shuffle=False,
        num_workers=max(0, num_workers),
        collate_fn=_collate,
    )

    _log.info("=" * 60)
    _log.info("🕵️ START COMPETITION CHECK (Single File)")
    _log.info("=" * 60)
    _log.info(f"📂 Data     : {data_file.name}")
    _log.info(f"🔢 Windows  : {split_stats['selected']} selected (Split: {args.split})")
    _log.info(f"⏱️  Steps    : {input_steps} (in) -> {eval_steps} (out total)")
    _log.info(f"🌀 Rollout  : Approx {rolling_iters} autoregressive loops per window")
    _log.info(f"🧩 Blend    : enabled={enable_overlap_blend} overlap_steps={overlap_steps}")
    _log.info("=" * 60)

    eps = 1e-12
    compute_device = predictor.device
    total_samples = 0
    ss_res_total = 0.0
    abs_err_total = 0.0
    mask_total = 0.0
    target_sum_total = 0.0
    target_sq_total = 0.0
    ss_res_total_std = 0.0
    target_sq_total_std = 0.0

    ss_res_h = torch.zeros(eval_steps, dtype=torch.float64, device=compute_device)
    abs_err_h = torch.zeros(eval_steps, dtype=torch.float64, device=compute_device)
    mask_h = torch.zeros(eval_steps, dtype=torch.float64, device=compute_device)
    target_sq_h = torch.zeros(eval_steps, dtype=torch.float64, device=compute_device)
    ss_res_h_std = torch.zeros(eval_steps, dtype=torch.float64, device=compute_device)
    target_sq_h_std = torch.zeros(eval_steps, dtype=torch.float64, device=compute_device)

    sample_pred: torch.Tensor | None = None
    sample_target: torch.Tensor | None = None
    sample_mask: torch.Tensor | None = None

    with tqdm_logging():
        pbar = tqdm(loader, desc="Comp-Check", ncols=100)
        for bi, batch in enumerate(pbar, start=1):
            x = batch["x"].float().to(compute_device, non_blocking=True)
            y_std = batch["y"].float().to(compute_device, non_blocking=True)
            y_valid = batch["y_valid"].float().to(compute_device, non_blocking=True)

            pred_std = _roll_forecast_std(
                predictor,
                x_std=x,
                target_steps=eval_steps,
                overlap_steps=overlap_steps,
                enable_overlap_blend=enable_overlap_blend,
            )
            pred = _destandardize_batch(pred_std, var_names=var_names, norm=norm)
            target = _destandardize_batch(y_std, var_names=var_names, norm=norm)
            mask = y_valid

            diff = (pred - target).float()
            diff2 = diff.pow(2)
            absdiff = diff.abs()

            ss_res_total += float(torch.sum(diff2 * mask).item())
            abs_err_total += float(torch.sum(absdiff * mask).item())
            mask_total += float(torch.sum(mask).item())
            target_sum_total += float(torch.sum(target * mask).item())
            target_sq_total += float(torch.sum(target.pow(2) * mask).item())
            target_sq_total_std += float(torch.sum(y_std.pow(2) * mask).item())
            ss_res_total_std += float(torch.sum((pred_std - y_std).pow(2) * mask).item())

            ss_res_h += torch.sum(diff2 * mask, dim=(0, 2, 3, 4)).to(dtype=torch.float64)
            abs_err_h += torch.sum(absdiff * mask, dim=(0, 2, 3, 4)).to(dtype=torch.float64)
            mask_h += torch.sum(mask, dim=(0, 2, 3, 4)).to(dtype=torch.float64)
            target_sq_h += torch.sum(target.pow(2) * mask, dim=(0, 2, 3, 4)).to(dtype=torch.float64)
            target_sq_h_std += torch.sum(y_std.pow(2) * mask, dim=(0, 2, 3, 4)).to(dtype=torch.float64)
            ss_res_h_std += torch.sum((pred_std - y_std).pow(2) * mask, dim=(0, 2, 3, 4)).to(dtype=torch.float64)

            total_samples += int(x.shape[0])

            if sample_pred is None and int(x.shape[0]) > 0:
                sample_pred = pred[0].detach().cpu()
                sample_target = target[0].detach().cpu()
                sample_mask = mask[0].detach().cpu()

            if args.max_batches > 0 and bi >= args.max_batches:
                _log.warning("stop early by --max-batches=%d", args.max_batches)
                break

    if total_samples <= 0 or mask_total <= eps:
        raise SystemExit("no valid evaluation samples consumed")

    mse = ss_res_total / max(mask_total, eps)
    rmse = math.sqrt(max(mse, 0.0))
    mae = abs_err_total / max(mask_total, eps)

    ss_tot = target_sq_total - (target_sum_total * target_sum_total) / max(mask_total, eps)
    nse = 1.0 - (ss_res_total / max(ss_tot, eps))
    rel_mse_pct = (ss_res_total_std / max(target_sq_total_std, eps)) * 100.0

    mse_h = (ss_res_h / torch.clamp_min(mask_h, eps)).cpu().numpy()
    rmse_h = np.sqrt(np.maximum(mse_h, 0.0))
    mae_h = (abs_err_h / torch.clamp_min(mask_h, eps)).cpu().numpy()
    rel_mse_pct_h = ((ss_res_h_std / torch.clamp_min(target_sq_h_std, eps)) * 100.0).cpu().numpy()
    horizon_hours_axis = (np.arange(eval_steps, dtype=np.float64) + 1.0) * float(args.time_step_hours)

    horizon_hours = float(eval_steps) * float(args.time_step_hours)
    pass_horizon = horizon_hours >= float(args.horizon_hours_threshold)
    pass_mse_pct = rel_mse_pct <= float(args.mse_percent_threshold)
    pass_all = bool(pass_horizon and pass_mse_pct)

    per_horizon_csv = output_dir / "per_horizon_metrics.csv"
    _write_per_horizon_csv(per_horizon_csv, horizon_hours_axis, mse_h, rmse_h, mae_h, rel_mse_pct_h)

    fig1 = figures_dir / "horizon_metrics.png"
    _plot_horizon_metrics(fig1, horizon_hours_axis, rmse_h, mae_h, rel_mse_pct_h, float(args.mse_percent_threshold))

    fig2 = figures_dir / "competition_threshold_comparison.png"
    _plot_competition_comparison(
        fig2,
        horizon_hours=horizon_hours,
        horizon_threshold=float(args.horizon_hours_threshold),
        rel_mse_percent=rel_mse_pct,
        mse_percent_threshold=float(args.mse_percent_threshold),
    )

    figure_files = [str(fig1), str(fig2)]

    if sample_pred is not None and sample_target is not None and sample_mask is not None:
        fig3 = figures_dir / "sample_timeseries_spatial_mean.png"
        _plot_sample_timeseries(
            fig3,
            sample_pred=sample_pred,
            sample_target=sample_target,
            sample_mask=sample_mask,
            var_names=var_names,
            time_step_hours=float(args.time_step_hours),
            max_plot_vars=max(1, int(args.max_plot_vars)),
        )
        figure_files.append(str(fig3))

        n_plot = max(1, min(int(args.max_plot_vars), len(var_names)))
        for c in range(n_plot):
            vname = var_names[c] if c < len(var_names) else f"var_{c}"
            fig_map = figures_dir / f"sample_last_horizon_map_{vname}.png"
            _plot_sample_map(
                fig_map,
                sample_pred=sample_pred,
                sample_target=sample_target,
                sample_mask=sample_mask,
                var_name=vname,
                time_step_hours=float(args.time_step_hours),
                channel_idx=c,
            )
            figure_files.append(str(fig_map))

    report = {
        "task": "element_forecasting",
        "mode": "single_file",
        "checkpoint": str(ckpt),
        "dataset": {
            "data_file": str(data_file),
            "split": args.split,
            "total_windows": split_stats["total"],
            "selected_windows": split_stats["selected"],
            "single_file_train_ratio": train_ratio,
            "single_file_val_ratio": val_ratio,
        },
        "model": {
            "var_names": list(var_names),
            "input_steps": input_steps,
            "single_shot_output_steps": model_output_steps,
            "eval_output_steps": eval_steps,
            "rolling_iterations_estimate": rolling_iters,
            "time_step_hours": float(args.time_step_hours),
            "horizon_hours": horizon_hours,
        },
        "competition_thresholds": {
            "horizon_hours_min": float(args.horizon_hours_threshold),
            "mse_percent_max": float(args.mse_percent_threshold),
        },
        "metrics": {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "nse": float(nse),
            "relative_mse_percent": float(rel_mse_pct),
        },
        "pass": {
            "horizon": bool(pass_horizon),
            "mse_percent": bool(pass_mse_pct),
            "overall": bool(pass_all),
        },
        "artifacts": {
            "per_horizon_csv": str(per_horizon_csv),
            "figures": figure_files,
        },
        "notes": [
            "evaluation is single-file only and uses rolling autoregressive forecasting",
            "relative_mse_percent = sum((pred_std-y_std)^2)/sum(y_std^2)*100 on valid mask",
            "if time_step_hours is not 1, pass --time-step-hours explicitly",
        ],
    }

    report_json = output_dir / "competition_report.json"
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_md = output_dir / "competition_summary.md"
    _write_markdown_summary(summary_md, report, figure_files, str(per_horizon_csv))

    verdict_str = "✅ PASS" if pass_all else "❌ FAIL"
    _log.info("=" * 60)
    _log.info("🚀 COMPETITION EVALUATION REPORT 🚀")
    _log.info("=" * 60)
    _log.info(f"🎯 Verdict         : {verdict_str}")
    _log.info(f"⏱️  Horizon         : {horizon_hours:.2f} h (Target: >= {args.horizon_hours_threshold} h) -> {'✅' if pass_horizon else '❌'}")
    _log.info(f"📉 Relative MSE    : {rel_mse_pct:.4f} % (Target: <= {args.mse_percent_threshold} %) -> {'✅' if pass_mse_pct else '❌'}")
    _log.info("-" * 60)
    _log.info("🏅 Detailed Metrics:")
    _log.info(f"  - MSE            : {mse:.6f}")
    _log.info(f"  - RMSE           : {rmse:.6f}")
    _log.info(f"  - MAE            : {mae:.6f}")
    _log.info(f"  - NSE            : {nse:.6f}")
    _log.info("-" * 60)
    _log.info("📁 Generated Artifacts:")
    _log.info(f"  - JSON Report    : {report_json}")
    _log.info(f"  - Markdown       : {summary_md}")
    _log.info(f"  - Metrics CSV    : {per_horizon_csv}")
    _log.info(f"  - Figures ({len(figure_files)})   : {figures_dir}")
    _log.info("=" * 60)


if __name__ == "__main__":
    main()
