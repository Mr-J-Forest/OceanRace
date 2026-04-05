"""训练历史可视化：读取 train_history.json 并输出趋势图。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
	sys.path.insert(0, str(SRC))

from utils.logger import get_logger, setup_logging
from utils.visualization_defaults import apply_matplotlib_defaults, standard_savefig_kwargs


logger = get_logger(__name__)


def _load_history(path: Path) -> list[dict[str, Any]]:
	if not path.is_file():
		raise FileNotFoundError(f"history file not found: {path}")
	data = json.loads(path.read_text(encoding="utf-8"))
	if not isinstance(data, list) or not data:
		raise ValueError("history json must be a non-empty list")
	return [item for item in data if isinstance(item, dict)]


def _collect_series(history: list[dict[str, Any]], key: str) -> list[float]:
	out: list[float] = []
	for row in history:
		v = row.get(key)
		try:
			out.append(float(v))
		except (TypeError, ValueError):
			out.append(float("nan"))
	return out


def _save_loss_figure(history: list[dict[str, Any]], out_dir: Path) -> Path:
	epochs = _collect_series(history, "epoch")
	train_loss = _collect_series(history, "train_loss")
	val_loss = _collect_series(history, "val_loss")

	fig, ax = plt.subplots(figsize=(10, 5.5))
	ax.plot(epochs, train_loss, marker="o", label="train_loss")
	ax.plot(epochs, val_loss, marker="s", label="val_loss")
	ax.set_title("Training vs Validation Loss")
	ax.set_xlabel("Epoch")
	ax.set_ylabel("Loss")
	ax.legend()

	out_path = out_dir / "traintrend_loss_curve.png"
	fig.savefig(out_path, **standard_savefig_kwargs())
	plt.close(fig)
	return out_path


def _save_metrics_figure(history: list[dict[str, Any]], out_dir: Path) -> Path:
	epochs = _collect_series(history, "epoch")
	rmse = _collect_series(history, "val_rmse")
	mae = _collect_series(history, "val_mae")
	nse = _collect_series(history, "val_nse")
	nrmse = _collect_series(history, "val_nrmse_percent")

	fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
	ax1, ax2, ax3, ax4 = axes.flatten()

	ax1.plot(epochs, rmse, marker="o")
	ax1.set_title("Validation RMSE")
	ax1.set_ylabel("RMSE")

	ax2.plot(epochs, mae, marker="o")
	ax2.set_title("Validation MAE")
	ax2.set_ylabel("MAE")

	ax3.plot(epochs, nse, marker="o")
	ax3.set_title("Validation NSE")
	ax3.set_xlabel("Epoch")
	ax3.set_ylabel("NSE")

	has_nrmse = any(v == v for v in nrmse)
	if has_nrmse:
		ax4.plot(epochs, nrmse, marker="o", label="val_nrmse_percent")
		ax4.axhline(15.0, color="#C44E52", linestyle="--", label="threshold=15%")
		ax4.legend()
	else:
		ax4.text(0.5, 0.5, "val_nrmse_percent not found", ha="center", va="center", transform=ax4.transAxes)
	ax4.set_title("Validation NRMSE (%)")
	ax4.set_xlabel("Epoch")
	ax4.set_ylabel("Percent")

	fig.suptitle("Element Forecast Training Metrics", y=1.02)
	fig.tight_layout()

	out_path = out_dir / "traintrend_val_metrics.png"
	fig.savefig(out_path, **standard_savefig_kwargs())
	plt.close(fig)
	return out_path


def main() -> None:
	ap = argparse.ArgumentParser(description="Visualize element forecasting training history")
	ap.add_argument(
		"--history",
		type=Path,
		default=ROOT / "outputs/element_forecasting/metrics/train_history.json",
		help="path to train_history.json",
	)
	ap.add_argument(
		"--out-dir",
		type=Path,
		default=ROOT / "outputs/element_forecasting/figures",
		help="directory to save generated figures",
	)
	args = ap.parse_args()

	setup_logging(log_file=ROOT / "outputs/logs/traintrend_visualization.log", force=True)
	apply_matplotlib_defaults()

	out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
	out_dir.mkdir(parents=True, exist_ok=True)
	history_path = args.history if args.history.is_absolute() else ROOT / args.history

	history = _load_history(history_path)
	p1 = _save_loss_figure(history, out_dir)
	p2 = _save_metrics_figure(history, out_dir)

	logger.info("history file: %s", history_path)
	logger.info("saved figure: %s", p1)
	logger.info("saved figure: %s", p2)


if __name__ == "__main__":
	main()
