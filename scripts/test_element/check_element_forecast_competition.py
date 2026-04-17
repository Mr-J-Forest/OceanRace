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

默认按 train.yaml 的 split_mode=competition_years 与 split_years 切分。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
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
    nrmse_pct_h: np.ndarray,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "horizon_hours", "mse", "rmse", "mae", "nrmse_percent"])
        for i in range(horizon_hours.shape[0]):
            w.writerow(
                [
                    i + 1,
                    float(horizon_hours[i]),
                    float(mse_h[i]),
                    float(rmse_h[i]),
                    float(mae_h[i]),
                    float(nrmse_pct_h[i]),
                ]
            )


def _plot_horizon_metrics(
    fig_path: Path,
    horizon_hours: np.ndarray,
    rmse_h: np.ndarray,
    mae_h: np.ndarray,
    nrmse_pct_h: np.ndarray,
    nrmse_percent_threshold: float,
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
    ax1.plot(horizon_hours, nrmse_pct_h, marker="o", label="NRMSE(%)")
    ax1.axhline(float(nrmse_percent_threshold), color="tab:red", linestyle="--", label="Threshold")
    ax1.set_xlabel("Forecast Horizon (hours)")
    ax1.set_ylabel("NRMSE (%)")
    ax1.set_title("Per-Horizon NRMSE vs Threshold")
    ax1.legend(loc="best")

    fig.savefig(fig_path, **standard_savefig_kwargs())
    plt.close(fig)


def _plot_competition_comparison(
    fig_path: Path,
    horizon_hours: float,
    horizon_threshold: float,
    nrmse_percent: float,
    nrmse_percent_threshold: float,
) -> None:
    import matplotlib.pyplot as plt

    apply_matplotlib_defaults()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    ax0 = axes[0]
    ax0.bar(["actual", "threshold"], [horizon_hours, horizon_threshold], color=["#4C72B0", "#55A868"])
    ax0.set_title("Horizon Requirement")
    ax0.set_ylabel("Hours")

    ax1 = axes[1]
    ax1.bar(["actual", "threshold"], [nrmse_percent, nrmse_percent_threshold], color=["#DD8452", "#C44E52"])
    ax1.set_title("NRMSE Requirement")
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


def _update_minmax_per_channel(
    current_min: torch.Tensor,
    current_max: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid = mask > 0
    masked_min = torch.where(valid, target, torch.full_like(target, float("inf")))
    masked_max = torch.where(valid, target, torch.full_like(target, float("-inf")))
    batch_min = torch.amin(masked_min, dim=(0, 1, 3, 4)).to(dtype=torch.float64)
    batch_max = torch.amax(masked_max, dim=(0, 1, 3, 4)).to(dtype=torch.float64)
    has_valid = torch.any(valid, dim=(0, 1, 3, 4))
    new_min = torch.where(has_valid, torch.minimum(current_min, batch_min), current_min)
    new_max = torch.where(has_valid, torch.maximum(current_max, batch_max), current_max)
    return new_min, new_max


def _safe_metrics_from_sums(
    *,
    ss_res: float,
    abs_err: float,
    sum_err: float,
    mask_sum: float,
    target_min: float,
    target_max: float,
    eps: float,
) -> dict[str, float]:
    mse = ss_res / max(mask_sum, eps)
    rmse = math.sqrt(max(mse, 0.0))
    mae = abs_err / max(mask_sum, eps)
    bias = sum_err / max(mask_sum, eps)
    target_range = max(target_max - target_min, eps)
    nrmse_pct = (rmse / target_range) * 100.0
    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "bias": float(bias),
        "nrmse_percent": float(nrmse_pct),
    }


def _sample_last_step_rmse(diff: torch.Tensor, mask: torch.Tensor, eps: float) -> torch.Tensor:
    diff_last = diff[:, -1]
    mask_last = mask[:, -1]
    num = torch.sum((diff_last ** 2) * mask_last, dim=(1, 2, 3))
    den = torch.sum(mask_last, dim=(1, 2, 3)).clamp_min(eps)
    return torch.sqrt(num / den + eps)


def _write_segment_bias_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    fields = ["scope", "mse", "rmse", "mae", "bias", "nrmse_percent"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fields})


def _write_per_variable_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    fields = [
        "var",
        "bias",
        "mae",
        "rmse",
        "nrmse_percent",
        "rmse_first_24h",
        "rmse_last_24h",
        "rmse_growth_last_vs_first",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fields})


def _write_markdown_summary(
    path: Path,
    report: dict[str, Any],
    figure_files: list[str],
    per_horizon_csv: str,
    segment_bias_csv: str,
    per_variable_csv: str,
) -> None:
    lines: list[str] = []
    lines.append("# Element Forecast Competition Check (Single File)")
    lines.append("")
    lines.append(f"- Verdict: {'PASS' if report['pass']['overall'] else 'FAIL'}")
    lines.append(f"- Horizon: {report['model']['horizon_hours']:.2f} h (threshold >= {report['competition_thresholds']['horizon_hours_min']:.2f} h)")
    lines.append(
        f"- NRMSE: {report['metrics']['nrmse_percent']:.4f}% "
        f"(threshold <= {report['competition_thresholds']['nrmse_percent_max']:.4f}%)"
    )
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append(f"- MSE: {report['metrics']['mse']:.6f}")
    lines.append(f"- RMSE: {report['metrics']['rmse']:.6f}")
    lines.append(f"- MAE: {report['metrics']['mae']:.6f}")
    lines.append(f"- NSE: {report['metrics']['nse']:.6f}")
    lines.append(f"- NRMSE(%): {report['metrics']['nrmse_percent']:.6f}")
    seg = report.get("segmented_metrics", {})
    first_seg = seg.get("first_24h", {})
    last_seg = seg.get("last_24h", {})
    delta_seg = seg.get("delta_last_minus_first", {})
    if first_seg and last_seg:
        lines.append("")
        lines.append("## Segmented Robustness")
        lines.append("")
        lines.append(
            f"- First 24h RMSE/NRMSE/Bias: {float(first_seg.get('rmse', float('nan'))):.6f} / "
            f"{float(first_seg.get('nrmse_percent', float('nan'))):.6f}% / {float(first_seg.get('bias', float('nan'))):.6f}"
        )
        lines.append(
            f"- Last 24h RMSE/NRMSE/Bias: {float(last_seg.get('rmse', float('nan'))):.6f} / "
            f"{float(last_seg.get('nrmse_percent', float('nan'))):.6f}% / {float(last_seg.get('bias', float('nan'))):.6f}"
        )
        lines.append(
            f"- Last-First Delta (RMSE/NRMSE/Bias): {float(delta_seg.get('rmse', float('nan'))):+.6f} / "
            f"{float(delta_seg.get('nrmse_percent', float('nan'))):+.6f}% / {float(delta_seg.get('bias', float('nan'))):+.6f}"
        )

    top_var = report.get("per_variable_summary", {}).get("worst_nrmse_var", None)
    if isinstance(top_var, dict) and top_var:
        lines.append("")
        lines.append("## Variable Risk")
        lines.append("")
        lines.append(
            f"- Worst variable by NRMSE: {top_var.get('var', 'unknown')} "
            f"(NRMSE={float(top_var.get('nrmse_percent', float('nan'))):.6f}%, "
            f"Bias={float(top_var.get('bias', float('nan'))):.6f})"
        )
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append(f"- Per-horizon metrics CSV: {per_horizon_csv}")
    lines.append(f"- Segment bias CSV: {segment_bias_csv}")
    lines.append(f"- Per-variable summary CSV: {per_variable_csv}")
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
    ap.add_argument("--nrmse-percent-threshold", type=float, default=15.0)
    ap.add_argument("--mse-percent-threshold", type=float, default=None, help="[deprecated] use --nrmse-percent-threshold")
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
    ap.add_argument("--plot-samples", type=int, default=4, help="可视化样本数量")
    ap.add_argument(
        "--sample-selection",
        type=str,
        default="mixed",
        choices=["mixed", "hard", "random"],
        help="样本选择策略：hard=高误差，random=随机，mixed=混合",
    )
    ap.add_argument("--hard-sample-ratio", type=float, default=0.5, help="mixed 模式下高误差样本占比")
    ap.add_argument("--sample-seed", type=int, default=42, help="随机采样种子")
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
    test_ratio = float(train_cfg.get("test_ratio", max(0.0, 1.0 - train_ratio - val_ratio)))

    split_mode = str(train_cfg.get("split_mode", "competition_years")).strip().lower()
    if split_mode not in ("competition_years", "ratio"):
        _log.warning("unsupported split_mode=%s, fallback to competition_years", split_mode)
        split_mode = "competition_years"

    split_years = train_cfg.get("split_years")

    ds_common_kwargs = {
        "data_file": data_file,
        "var_names": var_names,
        "input_steps": input_steps,
        "output_steps": eval_steps,
        "window_stride": int(train_cfg.get("window_stride", 1)),
        "open_file_lru_size": max(1, int(args.open_file_lru_size)),
        "split_mode": split_mode,
        "split_years": split_years,
        "split_ratios": (train_ratio, val_ratio, test_ratio),
        "norm_stats_path": norm_path,
        "root": ROOT,
    }

    if split_mode == "competition_years":
        ds_split = None if args.split == "all" else args.split
        ds = ElementForecastWindowDataset(split=ds_split, **ds_common_kwargs)
        full_ds = ElementForecastWindowDataset(split=None, **ds_common_kwargs)
        train_ds = ElementForecastWindowDataset(split="train", **ds_common_kwargs)
        val_ds = ElementForecastWindowDataset(split="val", **ds_common_kwargs)
        test_ds = ElementForecastWindowDataset(split="test", **ds_common_kwargs)
        split_stats = {
            "total": len(full_ds),
            "train": len(train_ds),
            "val": len(val_ds),
            "test": len(test_ds),
            "selected": len(ds),
        }
    else:
        full_ds = ElementForecastWindowDataset(split=None, **ds_common_kwargs)
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
    _log.info(f"🔢 Windows  : {split_stats['selected']} selected (Split: {args.split}, mode={split_mode})")
    _log.info(f"⏱️  Steps    : {input_steps} (in) -> {eval_steps} (out total)")
    _log.info(f"🌀 Rollout  : Approx {rolling_iters} autoregressive loops per window")
    _log.info(f"🧩 Blend    : enabled={enable_overlap_blend} overlap_steps={overlap_steps}")
    _log.info("=" * 60)

    eps = 1e-12
    random.seed(int(args.sample_seed))
    compute_device = predictor.device
    total_samples = 0
    ss_res_total = 0.0
    abs_err_total = 0.0
    sum_err_total = 0.0
    mask_total = 0.0
    target_sum_total = 0.0
    target_sq_total = 0.0
    target_min_total = float("inf")
    target_max_total = float("-inf")

    first_steps = min(24, eval_steps)
    last_start = max(0, eval_steps - 24)
    ss_res_first = 0.0
    abs_err_first = 0.0
    sum_err_first = 0.0
    mask_first_total = 0.0
    target_min_first = float("inf")
    target_max_first = float("-inf")
    ss_res_last = 0.0
    abs_err_last = 0.0
    sum_err_last = 0.0
    mask_last_total = 0.0
    target_min_last = float("inf")
    target_max_last = float("-inf")

    n_vars = len(var_names)
    var_ss_res = torch.zeros(n_vars, dtype=torch.float64, device=compute_device)
    var_abs_err = torch.zeros(n_vars, dtype=torch.float64, device=compute_device)
    var_sum_err = torch.zeros(n_vars, dtype=torch.float64, device=compute_device)
    var_mask = torch.zeros(n_vars, dtype=torch.float64, device=compute_device)
    var_min = torch.full((n_vars,), float("inf"), dtype=torch.float64, device=compute_device)
    var_max = torch.full((n_vars,), float("-inf"), dtype=torch.float64, device=compute_device)
    var_ss_res_first = torch.zeros(n_vars, dtype=torch.float64, device=compute_device)
    var_mask_first = torch.zeros(n_vars, dtype=torch.float64, device=compute_device)
    var_ss_res_last = torch.zeros(n_vars, dtype=torch.float64, device=compute_device)
    var_mask_last = torch.zeros(n_vars, dtype=torch.float64, device=compute_device)

    ss_res_h = torch.zeros(eval_steps, dtype=torch.float64, device=compute_device)
    abs_err_h = torch.zeros(eval_steps, dtype=torch.float64, device=compute_device)
    mask_h = torch.zeros(eval_steps, dtype=torch.float64, device=compute_device)
    target_min_h = torch.full((eval_steps,), float("inf"), dtype=torch.float64, device=compute_device)
    target_max_h = torch.full((eval_steps,), float("-inf"), dtype=torch.float64, device=compute_device)

    hard_candidates: list[dict[str, Any]] = []
    random_samples: list[dict[str, Any]] = []
    seen_windows = 0
    sample_pool_cap = max(1, int(args.plot_samples))

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
            sum_err_total += float(torch.sum(diff * mask).item())
            mask_total += float(torch.sum(mask).item())
            target_sum_total += float(torch.sum(target * mask).item())
            target_sq_total += float(torch.sum(target.pow(2) * mask).item())

            var_ss_res += torch.sum(diff2 * mask, dim=(0, 1, 3, 4)).to(dtype=torch.float64)
            var_abs_err += torch.sum(absdiff * mask, dim=(0, 1, 3, 4)).to(dtype=torch.float64)
            var_sum_err += torch.sum(diff * mask, dim=(0, 1, 3, 4)).to(dtype=torch.float64)
            var_mask += torch.sum(mask, dim=(0, 1, 3, 4)).to(dtype=torch.float64)
            var_min, var_max = _update_minmax_per_channel(var_min, var_max, target=target, mask=mask)

            diff_first = diff[:, :first_steps]
            mask_first = mask[:, :first_steps]
            target_first = target[:, :first_steps]
            ss_res_first += float(torch.sum((diff_first ** 2) * mask_first).item())
            abs_err_first += float(torch.sum(torch.abs(diff_first) * mask_first).item())
            sum_err_first += float(torch.sum(diff_first * mask_first).item())
            mask_first_total += float(torch.sum(mask_first).item())
            if torch.any(mask_first > 0):
                target_first_valid = target_first[mask_first > 0]
                target_min_first = min(target_min_first, float(torch.min(target_first_valid).item()))
                target_max_first = max(target_max_first, float(torch.max(target_first_valid).item()))
            var_ss_res_first += torch.sum((diff_first ** 2) * mask_first, dim=(0, 1, 3, 4)).to(dtype=torch.float64)
            var_mask_first += torch.sum(mask_first, dim=(0, 1, 3, 4)).to(dtype=torch.float64)

            diff_last = diff[:, last_start:eval_steps]
            mask_last = mask[:, last_start:eval_steps]
            target_last = target[:, last_start:eval_steps]
            ss_res_last += float(torch.sum((diff_last ** 2) * mask_last).item())
            abs_err_last += float(torch.sum(torch.abs(diff_last) * mask_last).item())
            sum_err_last += float(torch.sum(diff_last * mask_last).item())
            mask_last_total += float(torch.sum(mask_last).item())
            if torch.any(mask_last > 0):
                target_last_valid = target_last[mask_last > 0]
                target_min_last = min(target_min_last, float(torch.min(target_last_valid).item()))
                target_max_last = max(target_max_last, float(torch.max(target_last_valid).item()))
            var_ss_res_last += torch.sum((diff_last ** 2) * mask_last, dim=(0, 1, 3, 4)).to(dtype=torch.float64)
            var_mask_last += torch.sum(mask_last, dim=(0, 1, 3, 4)).to(dtype=torch.float64)

            valid = mask > 0
            if torch.any(valid):
                t_valid = target[valid]
                target_min_total = min(target_min_total, float(torch.min(t_valid).item()))
                target_max_total = max(target_max_total, float(torch.max(t_valid).item()))

            ss_res_h += torch.sum(diff2 * mask, dim=(0, 2, 3, 4)).to(dtype=torch.float64)
            abs_err_h += torch.sum(absdiff * mask, dim=(0, 2, 3, 4)).to(dtype=torch.float64)
            mask_h += torch.sum(mask, dim=(0, 2, 3, 4)).to(dtype=torch.float64)
            valid_h = mask > 0
            target_masked_min = torch.where(valid_h, target, torch.full_like(target, float("inf")))
            target_masked_max = torch.where(valid_h, target, torch.full_like(target, float("-inf")))
            tmin_h = torch.amin(target_masked_min, dim=(0, 2, 3, 4)).to(dtype=torch.float64)
            tmax_h = torch.amax(target_masked_max, dim=(0, 2, 3, 4)).to(dtype=torch.float64)
            has_valid_h = torch.any(valid_h, dim=(0, 2, 3, 4))
            target_min_h = torch.where(has_valid_h, torch.minimum(target_min_h, tmin_h), target_min_h)
            target_max_h = torch.where(has_valid_h, torch.maximum(target_max_h, tmax_h), target_max_h)

            total_samples += int(x.shape[0])

            per_sample_last_rmse = _sample_last_step_rmse(diff, mask, eps=eps)
            for si in range(int(x.shape[0])):
                sample_item = {
                    "sample_id": int(seen_windows),
                    "score": float(per_sample_last_rmse[si].item()),
                    "pred": pred[si].detach().cpu(),
                    "target": target[si].detach().cpu(),
                    "mask": mask[si].detach().cpu(),
                }
                seen_windows += 1

                hard_candidates.append(sample_item)
                hard_candidates.sort(key=lambda item: float(item["score"]), reverse=True)
                if len(hard_candidates) > sample_pool_cap:
                    hard_candidates = hard_candidates[:sample_pool_cap]

                if len(random_samples) < sample_pool_cap:
                    random_samples.append(sample_item)
                else:
                    j = random.randint(0, seen_windows - 1)
                    if j < sample_pool_cap:
                        random_samples[j] = sample_item

            if args.max_batches > 0 and bi >= args.max_batches:
                _log.warning("stop early by --max-batches=%d", args.max_batches)
                break

    if total_samples <= 0 or mask_total <= eps:
        raise SystemExit("no valid evaluation samples consumed")

    mse = ss_res_total / max(mask_total, eps)
    rmse = math.sqrt(max(mse, 0.0))
    mae = abs_err_total / max(mask_total, eps)
    bias = sum_err_total / max(mask_total, eps)

    ss_tot = target_sq_total - (target_sum_total * target_sum_total) / max(mask_total, eps)
    nse = 1.0 - (ss_res_total / max(ss_tot, eps))
    target_range = max(target_max_total - target_min_total, eps)
    nrmse_pct = (rmse / target_range) * 100.0

    first_24h_metrics = _safe_metrics_from_sums(
        ss_res=ss_res_first,
        abs_err=abs_err_first,
        sum_err=sum_err_first,
        mask_sum=mask_first_total,
        target_min=target_min_first,
        target_max=target_max_first,
        eps=eps,
    )
    last_24h_metrics = _safe_metrics_from_sums(
        ss_res=ss_res_last,
        abs_err=abs_err_last,
        sum_err=sum_err_last,
        mask_sum=mask_last_total,
        target_min=target_min_last,
        target_max=target_max_last,
        eps=eps,
    )

    var_rows: list[dict[str, float | str]] = []
    worst_var: dict[str, float | str] | None = None
    for c, vname in enumerate(var_names):
        den = float(var_mask[c].item())
        if den <= eps:
            row = {
                "var": str(vname),
                "bias": float("nan"),
                "mae": float("nan"),
                "rmse": float("nan"),
                "nrmse_percent": float("nan"),
                "rmse_first_24h": float("nan"),
                "rmse_last_24h": float("nan"),
                "rmse_growth_last_vs_first": float("nan"),
            }
            var_rows.append(row)
            continue
        mse_v = float(var_ss_res[c].item() / den)
        rmse_v = float(math.sqrt(max(mse_v, 0.0)))
        mae_v = float(var_abs_err[c].item() / den)
        bias_v = float(var_sum_err[c].item() / den)
        range_v = max(float(var_max[c].item() - var_min[c].item()), eps)
        nrmse_v = float((rmse_v / range_v) * 100.0)
        rmse_first_v = float(math.sqrt(max(float(var_ss_res_first[c].item() / max(float(var_mask_first[c].item()), eps)), 0.0)))
        rmse_last_v = float(math.sqrt(max(float(var_ss_res_last[c].item() / max(float(var_mask_last[c].item()), eps)), 0.0)))
        growth_v = float(rmse_last_v / max(rmse_first_v, eps))
        row = {
            "var": str(vname),
            "bias": bias_v,
            "mae": mae_v,
            "rmse": rmse_v,
            "nrmse_percent": nrmse_v,
            "rmse_first_24h": rmse_first_v,
            "rmse_last_24h": rmse_last_v,
            "rmse_growth_last_vs_first": growth_v,
        }
        if worst_var is None or float(row["nrmse_percent"]) > float(worst_var["nrmse_percent"]):
            worst_var = dict(row)
        var_rows.append(row)

    mse_h = (ss_res_h / torch.clamp_min(mask_h, eps)).cpu().numpy()
    rmse_h = np.sqrt(np.maximum(mse_h, 0.0))
    mae_h = (abs_err_h / torch.clamp_min(mask_h, eps)).cpu().numpy()
    range_h = torch.clamp_min(target_max_h - target_min_h, eps)
    nrmse_pct_h = (torch.from_numpy(rmse_h).to(device=compute_device, dtype=torch.float64) / range_h) * 100.0
    nrmse_pct_h = nrmse_pct_h.cpu().numpy()
    horizon_hours_axis = (np.arange(eval_steps, dtype=np.float64) + 1.0) * float(args.time_step_hours)

    horizon_hours = float(eval_steps) * float(args.time_step_hours)
    pass_horizon = horizon_hours >= float(args.horizon_hours_threshold)
    nrmse_threshold = float(args.nrmse_percent_threshold if args.nrmse_percent_threshold is not None else 15.0)
    if args.mse_percent_threshold is not None:
        nrmse_threshold = float(args.mse_percent_threshold)
    pass_nrmse_pct = nrmse_pct <= nrmse_threshold
    pass_all = bool(pass_horizon and pass_nrmse_pct)

    per_horizon_csv = output_dir / "per_horizon_metrics.csv"
    _write_per_horizon_csv(per_horizon_csv, horizon_hours_axis, mse_h, rmse_h, mae_h, nrmse_pct_h)

    segment_bias_csv = output_dir / "segment_bias_metrics.csv"
    _write_segment_bias_csv(
        segment_bias_csv,
        rows=[
            {"scope": "overall", "mse": mse, "rmse": rmse, "mae": mae, "bias": bias, "nrmse_percent": nrmse_pct},
            {"scope": "first_24h", **first_24h_metrics},
            {"scope": "last_24h", **last_24h_metrics},
        ],
    )

    per_variable_csv = output_dir / "per_variable_summary.csv"
    _write_per_variable_csv(per_variable_csv, var_rows)

    fig1 = figures_dir / "horizon_metrics.png"
    _plot_horizon_metrics(fig1, horizon_hours_axis, rmse_h, mae_h, nrmse_pct_h, nrmse_threshold)

    fig2 = figures_dir / "competition_threshold_comparison.png"
    _plot_competition_comparison(
        fig2,
        horizon_hours=horizon_hours,
        horizon_threshold=float(args.horizon_hours_threshold),
        nrmse_percent=nrmse_pct,
        nrmse_percent_threshold=nrmse_threshold,
    )

    figure_files = [str(fig1), str(fig2)]

    selected_samples: list[dict[str, Any]] = []
    strategy = str(args.sample_selection).strip().lower()
    n_plot_samples = max(1, int(args.plot_samples))
    if strategy == "hard":
        selected_samples = hard_candidates[:n_plot_samples]
    elif strategy == "random":
        selected_samples = random_samples[:n_plot_samples]
    else:
        hard_ratio = min(1.0, max(0.0, float(args.hard_sample_ratio)))
        n_hard = int(round(n_plot_samples * hard_ratio))
        n_hard = max(1, min(n_plot_samples, n_hard)) if n_plot_samples > 1 else 1
        n_random = max(0, n_plot_samples - n_hard)
        selected_samples = hard_candidates[:n_hard] + random_samples[:n_random]
        dedup: list[dict[str, Any]] = []
        seen_ids: set[int] = set()
        for item in selected_samples:
            sid = int(item.get("sample_id", -1))
            if sid in seen_ids:
                continue
            seen_ids.add(sid)
            dedup.append(item)
        selected_samples = dedup[:n_plot_samples]

    n_plot = max(1, min(int(args.max_plot_vars), len(var_names)))
    for idx, item in enumerate(selected_samples, start=1):
        fig_ts = figures_dir / f"sample_{idx:02d}_timeseries_spatial_mean.png"
        _plot_sample_timeseries(
            fig_ts,
            sample_pred=item["pred"],
            sample_target=item["target"],
            sample_mask=item["mask"],
            var_names=var_names,
            time_step_hours=float(args.time_step_hours),
            max_plot_vars=max(1, int(args.max_plot_vars)),
        )
        figure_files.append(str(fig_ts))

        for c in range(n_plot):
            vname = var_names[c] if c < len(var_names) else f"var_{c}"
            fig_map = figures_dir / f"sample_{idx:02d}_last_horizon_map_{vname}.png"
            _plot_sample_map(
                fig_map,
                sample_pred=item["pred"],
                sample_target=item["target"],
                sample_mask=item["mask"],
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
            "nrmse_percent_max": nrmse_threshold,
        },
        "metrics": {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "bias": float(bias),
            "nse": float(nse),
            "nrmse_percent": float(nrmse_pct),
        },
        "segmented_metrics": {
            "first_24h": first_24h_metrics,
            "last_24h": last_24h_metrics,
            "delta_last_minus_first": {
                "rmse": float(last_24h_metrics["rmse"] - first_24h_metrics["rmse"]),
                "mae": float(last_24h_metrics["mae"] - first_24h_metrics["mae"]),
                "bias": float(last_24h_metrics["bias"] - first_24h_metrics["bias"]),
                "nrmse_percent": float(last_24h_metrics["nrmse_percent"] - first_24h_metrics["nrmse_percent"]),
            },
        },
        "per_variable_summary": {
            "rows": var_rows,
            "worst_nrmse_var": worst_var,
        },
        "pass": {
            "horizon": bool(pass_horizon),
            "nrmse_percent": bool(pass_nrmse_pct),
            "overall": bool(pass_all),
        },
        "artifacts": {
            "per_horizon_csv": str(per_horizon_csv),
            "segment_bias_csv": str(segment_bias_csv),
            "per_variable_csv": str(per_variable_csv),
            "figures": figure_files,
        },
        "sample_selection": {
            "strategy": strategy,
            "plot_samples": n_plot_samples,
            "hard_sample_ratio": float(args.hard_sample_ratio),
            "selected_count": len(selected_samples),
        },
        "notes": [
            "evaluation is single-file only and uses rolling autoregressive forecasting",
            "nrmse_percent = RMSE / (target_max - target_min) * 100 on valid mask",
            "if time_step_hours is not 1, pass --time-step-hours explicitly",
        ],
    }

    report_json = output_dir / "competition_report.json"
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_md = output_dir / "competition_summary.md"
    _write_markdown_summary(
        summary_md,
        report,
        figure_files,
        str(per_horizon_csv),
        str(segment_bias_csv),
        str(per_variable_csv),
    )

    verdict_str = "✅ PASS" if pass_all else "❌ FAIL"
    _log.info("=" * 60)
    _log.info("🚀 COMPETITION EVALUATION REPORT 🚀")
    _log.info("=" * 60)
    _log.info(f"🎯 Verdict         : {verdict_str}")
    _log.info(f"⏱️  Horizon         : {horizon_hours:.2f} h (Target: >= {args.horizon_hours_threshold} h) -> {'✅' if pass_horizon else '❌'}")
    _log.info(f"📉 NRMSE           : {nrmse_pct:.4f} % (Target: <= {nrmse_threshold} %) -> {'✅' if pass_nrmse_pct else '❌'}")
    _log.info("-" * 60)
    _log.info("🏅 Detailed Metrics:")
    _log.info(f"  - MSE            : {mse:.6f}")
    _log.info(f"  - RMSE           : {rmse:.6f}")
    _log.info(f"  - MAE            : {mae:.6f}")
    _log.info(f"  - Bias           : {bias:.6f}")
    _log.info(f"  - NSE            : {nse:.6f}")
    _log.info(
        "  - Segment RMSE   : first24h=%.6f last24h=%.6f (delta=%+.6f)",
        float(first_24h_metrics["rmse"]),
        float(last_24h_metrics["rmse"]),
        float(last_24h_metrics["rmse"] - first_24h_metrics["rmse"]),
    )
    _log.info("-" * 60)
    _log.info("📁 Generated Artifacts:")
    _log.info(f"  - JSON Report    : {report_json}")
    _log.info(f"  - Markdown       : {summary_md}")
    _log.info(f"  - Metrics CSV    : {per_horizon_csv}")
    _log.info(f"  - Segment CSV    : {segment_bias_csv}")
    _log.info(f"  - Per-Var CSV    : {per_variable_csv}")
    _log.info(f"  - Figures ({len(figure_files)})   : {figures_dir}")
    _log.info("=" * 60)


if __name__ == "__main__":
    main()
