"""
要素预测模型比赛达标检查脚本。

用法（项目根目录）：

  python scripts/08_check_element_forecast_competition.py

默认检查两项：
1) 预测时长是否 >= 72 小时（支持 12h 滚动到 72h）
2) 相对 MSE(%) 是否 <= 15
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from element_forecasting.dataset import ElementForecastWindowDataset
from element_forecasting.evaluator import compute_regression_metrics_masked
from element_forecasting.predictor import ElementForecastPredictor
from utils.dataset_utils import destandardize_tensor, load_norm_stats
from utils.logger import get_logger, setup_logging

_log = get_logger(__name__)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return cfg if isinstance(cfg, dict) else {}


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


def _relative_mse_percent(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-12) -> float:
    p = pred.float()
    t = target.float()
    m = mask.float()
    num = torch.sum(((p - t) ** 2) * m)
    den = torch.sum((t**2) * m).clamp_min(eps)
    return float((num / den * 100.0).item())


def _roll_forecast_std(
    predictor: ElementForecastPredictor,
    x_std: torch.Tensor,
    target_steps: int,
) -> torch.Tensor:
    """滚动预测（标准化空间）：单次输出12h，循环直到 target_steps。"""
    if target_steps <= 0:
        raise ValueError("target_steps must be > 0")

    cur_x = x_std.clone()
    chunks: list[torch.Tensor] = []
    got = 0
    in_steps = int(predictor.input_steps)

    while got < target_steps:
        out = predictor.predict(cur_x, denormalize=False)
        pred_std = out["pred"].float()
        if pred_std.ndim != 5:
            raise RuntimeError("predictor output shape is invalid")

        take = min(target_steps - got, int(pred_std.shape[1]))
        chunks.append(pred_std[:, :take])
        got += take
        if got >= target_steps:
            break

        # 自回归回灌：保留最近 input_steps 个时间步作为下一轮输入。
        cat = torch.cat([cur_x.float(), pred_std], dim=1)
        cur_x = cat[:, -in_steps:].contiguous()

    return torch.cat(chunks, dim=1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Check element forecasting model against competition requirements")
    ap.add_argument("--data-config", type=Path, default=ROOT / "configs/data_config.yaml")
    ap.add_argument("--train-config", type=Path, default=ROOT / "configs/element_forecasting/train.yaml")
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "outputs/element_forecasting/checkpoints/hybrid_best.pt")
    ap.add_argument("--processed-dir", type=str, default=None)
    ap.add_argument("--data-file", type=str, default=None, help="单一 NetCDF 文件；设置后忽略 split/manifest")
    ap.add_argument("--manifest", type=str, default=None)
    ap.add_argument("--norm", type=str, default=None)
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--time-step-hours", type=float, default=1.0, help="每个时间步代表多少小时")
    ap.add_argument("--eval-horizon-hours", type=float, default=72.0, help="滚动评估总时长（小时）")
    ap.add_argument("--horizon-hours-threshold", type=float, default=72.0)
    ap.add_argument("--mse-percent-threshold", type=float, default=15.0)
    ap.add_argument("--open-file-lru-size", type=int, default=16)
    ap.add_argument("--max-batches", type=int, default=0, help="仅评估前 N 个 batch；0 表示全部")
    ap.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/final_results/element_forecasting/competition_check.json",
    )
    args = ap.parse_args()

    log_file = ROOT / "outputs/logs/element_forecasting_competition_check.log"
    setup_logging(log_file=log_file)

    data_cfg = _load_yaml(args.data_config)
    train_cfg = _load_yaml(args.train_config)

    ckpt = args.checkpoint
    if not ckpt.is_absolute():
        ckpt = ROOT / ckpt
    if not ckpt.is_file():
        raise SystemExit(f"checkpoint not found: {ckpt}")

    norm_path_str = (
        args.norm
        or train_cfg.get("norm_stats_path")
        or data_cfg.get("artifacts", {}).get("normalization_files", {}).get("element_forecasting", "data/processed/normalization/element_forecasting_norm.json")
    )
    norm_path = Path(norm_path_str)
    if not norm_path.is_absolute():
        norm_path = ROOT / norm_path
    norm = load_norm_stats(norm_path) if norm_path.is_file() else None

    predictor = ElementForecastPredictor(
        checkpoint_path=ckpt,
        device=args.device,
        norm_stats_path=norm_path,
    )
    var_names = predictor.var_names
    input_steps = predictor.input_steps
    model_output_steps = predictor.output_steps
    eval_steps = max(1, int(round(float(args.eval_horizon_hours) / float(args.time_step_hours))))

    processed_dir = Path(
        args.processed_dir
        or train_cfg.get("processed_dir")
        or data_cfg.get("paths", {}).get("processed", {}).get("element_forecasting", "data/processed/element_forecasting")
    )
    if not processed_dir.is_absolute():
        processed_dir = ROOT / processed_dir

    manifest_path = Path(
        args.manifest
        or train_cfg.get("manifest_path")
        or data_cfg.get("artifacts", {}).get("split_manifests", {}).get("element_forecasting", "data/processed/splits/element_forecasting.json")
    )
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path

    data_file: Path | None = None
    data_file_str = args.data_file
    if data_file_str:
        data_file = Path(data_file_str)
        if not data_file.is_absolute():
            data_file = ROOT / data_file

    data_source_mode = "manifest_split" if data_file is None else "single_file"
    try:
        ds = ElementForecastWindowDataset(
            processed_dir=processed_dir,
            data_file=data_file,
            var_names=var_names,
            input_steps=input_steps,
            output_steps=eval_steps,
            window_stride=int(train_cfg.get("window_stride", 1)),
            stitch_across_files=bool(train_cfg.get("stitch_across_files", True)),
            open_file_lru_size=args.open_file_lru_size,
            split=None if data_file is not None else args.split,
            manifest_path=manifest_path,
            norm_stats_path=norm_path,
            root=ROOT,
        )
    except FileNotFoundError as ex:
        fallback = train_cfg.get("data_file")
        if data_file is not None or not fallback:
            raise
        fb = Path(fallback)
        if not fb.is_absolute():
            fb = ROOT / fb
        _log.warning("manifest split contains missing file, fallback to single data_file: %s | err=%s", fb, ex)
        ds = ElementForecastWindowDataset(
            processed_dir=processed_dir,
            data_file=fb,
            var_names=var_names,
            input_steps=input_steps,
            output_steps=eval_steps,
            window_stride=int(train_cfg.get("window_stride", 1)),
            stitch_across_files=bool(train_cfg.get("stitch_across_files", True)),
            open_file_lru_size=args.open_file_lru_size,
            split=None,
            manifest_path=manifest_path,
            norm_stats_path=norm_path,
            root=ROOT,
        )
        data_file = fb
        data_source_mode = "single_file_fallback"
    if len(ds) == 0:
        raise SystemExit("evaluation dataset is empty")

    loader = DataLoader(
        ds,
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
        num_workers=max(0, int(args.num_workers)),
        collate_fn=_collate,
    )

    _log.info("start competition check | windows=%d | split=%s | data_file=%s", len(ds), args.split, str(data_file or ""))

    total = 0
    sum_mse = 0.0
    sum_rmse = 0.0
    sum_mae = 0.0
    sum_nse = 0.0
    sum_rel_mse_pct = 0.0

    for bi, batch in enumerate(loader, start=1):
        x = batch["x"]
        y_std = batch["y"]
        y_valid = batch["y_valid"]

        pred_std = _roll_forecast_std(predictor, x_std=x.float(), target_steps=eval_steps)
        pred = _destandardize_batch(pred_std, var_names=var_names, norm=norm)
        y = _destandardize_batch(y_std.float(), var_names=var_names, norm=norm)

        bs = int(x.size(0))
        m = compute_regression_metrics_masked(pred, y, y_valid)
        rel_mse_pct = _relative_mse_percent(pred, y, y_valid)

        total += bs
        sum_mse += m["mse"] * bs
        sum_rmse += m["rmse"] * bs
        sum_mae += m["mae"] * bs
        sum_nse += m["nse"] * bs
        sum_rel_mse_pct += rel_mse_pct * bs

        if args.max_batches > 0 and bi >= args.max_batches:
            break

    if total <= 0:
        raise SystemExit("no evaluation samples consumed")

    metrics = {
        "mse": sum_mse / total,
        "rmse": sum_rmse / total,
        "mae": sum_mae / total,
        "nse": sum_nse / total,
        "relative_mse_percent": sum_rel_mse_pct / total,
    }

    horizon_hours = float(eval_steps) * float(args.time_step_hours)
    pass_horizon = horizon_hours >= float(args.horizon_hours_threshold)
    pass_mse_pct = metrics["relative_mse_percent"] <= float(args.mse_percent_threshold)
    pass_all = bool(pass_horizon and pass_mse_pct)

    report = {
        "task": "element_forecasting",
        "checkpoint": str(ckpt),
        "dataset": {
            "source_mode": data_source_mode,
            "split": args.split,
            "data_file": str(data_file) if data_file is not None else None,
            "num_windows": len(ds),
            "num_eval_samples": total,
        },
        "model": {
            "var_names": list(var_names),
            "input_steps": input_steps,
            "single_shot_output_steps": model_output_steps,
            "eval_output_steps": eval_steps,
            "time_step_hours": float(args.time_step_hours),
            "horizon_hours": horizon_hours,
        },
        "competition_thresholds": {
            "horizon_hours_min": float(args.horizon_hours_threshold),
            "mse_percent_max": float(args.mse_percent_threshold),
        },
        "metrics": {k: float(v) for k, v in metrics.items()},
        "pass": {
            "horizon": pass_horizon,
            "mse_percent": pass_mse_pct,
            "overall": pass_all,
        },
        "notes": [
            "relative_mse_percent = sum((pred-target)^2)/sum(target^2)*100（mask有效点）",
            "若 time_step_hours 不是 1 小时，请通过参数覆盖",
            "评估采用滚动预测：单次输出窗口循环回灌，直到 eval_horizon_hours",
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    verdict = "PASS" if pass_all else "FAIL"
    _log.info("competition check done | verdict=%s | horizon=%.1fh | rel_mse=%.4f%% | report=%s", verdict, horizon_hours, metrics["relative_mse_percent"], args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
