"""风-浪异常训练入口（双分支 AE）。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
	sys.path.insert(0, str(SRC))

from anomaly_detection.dataset import AnomalyFrameDataset
from anomaly_detection.detector import associate_events, build_detection_report, infer_errors
from anomaly_detection.evaluator import evaluate_with_labels, roc_auc_from_scores, summarize_errors
from anomaly_detection.model import DualBranchAutoEncoder
from anomaly_detection.trainer import AnomalyTrainConfig, fit
from utils.logger import get_logger, setup_logging

_log = get_logger(__name__)


def _load_yaml(path: Path) -> dict[str, Any]:
	if not path.is_file():
		return {}
	data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
	return data if isinstance(data, dict) else {}


def _resolve_path(path_like: str | Path | None, *, default: Path) -> Path:
	if path_like is None:
		return default
	p = Path(path_like)
	return p if p.is_absolute() else (ROOT / p)


def _resolve_device(v: str) -> str:
	if v == "auto":
		return "cuda" if torch.cuda.is_available() else "cpu"
	return v


def parse_args() -> argparse.Namespace:
	ap = argparse.ArgumentParser(description="Train anomaly detector (dual-branch AE)")
	ap.add_argument("--data-config", default="configs/data_config.yaml")
	ap.add_argument("--model-config", default="configs/anomaly_detection/model.yaml")
	ap.add_argument("--train-config", default="configs/anomaly_detection/train.yaml")

	ap.add_argument("--epochs", type=int, default=None)
	ap.add_argument("--batch-size", type=int, default=None)
	ap.add_argument("--num-workers", type=int, default=None)
	ap.add_argument("--lr", type=float, default=None)
	ap.add_argument("--device", default=None)

	ap.add_argument("--processed-dir", default=None)
	ap.add_argument("--manifest", default=None)
	ap.add_argument("--norm-stats", default=None)

	ap.add_argument("--max-train-samples", type=int, default=None)
	ap.add_argument("--max-val-samples", type=int, default=None)
	ap.add_argument("--max-test-samples", type=int, default=None)
	ap.add_argument("--open-file-lru-size", type=int, default=None)
	ap.add_argument("--output-dir", default=None)
	ap.add_argument("--labels-json", default=None, help="optional labels json with train/val/test arrays")
	ap.add_argument("--events-json", default=None, help="optional event windows json for association")
	ap.add_argument(
		"--report-splits",
		default="val",
		help="comma-separated splits for post-train report, e.g. val or train,val,test",
	)
	ap.add_argument("--skip-split-report", action="store_true", help="skip post-train split report generation")
	return ap.parse_args()



def _build_split_dataset(
	*,
	split: str,
	processed_dir: Path,
	manifest: Path,
	norm_stats: Path,
	open_file_lru_size: int,
) -> AnomalyFrameDataset:
	return AnomalyFrameDataset(
		processed_anomaly_dir=processed_dir,
		split=split,
		manifest_path=manifest,
		norm_stats_path=norm_stats if norm_stats.is_file() else None,
		root=ROOT,
		open_file_lru_size=open_file_lru_size,
	)


def _subset_if_needed(ds: Any, max_n: int | None) -> Any:
	if max_n is None:
		return ds
	n = int(max_n)
	if n > 0 and len(ds) > n:
		return Subset(ds, list(range(n)))
	return ds


def _collect_timestamps(loader: DataLoader) -> list[int | float | None]:
	vals: list[int | float | None] = []
	for b in loader:
		ts = b.get("timestamp")
		if ts is None:
			vals.extend([None] * int(b["oper_x"].shape[0]))
		elif isinstance(ts, list):
			vals.extend(ts)
		else:
			vals.extend(ts.tolist())
	return vals


def main() -> None:
	args = parse_args()
	setup_logging(log_file=ROOT / "outputs/logs/anomaly_detection_train.log")

	data_cfg = _load_yaml(_resolve_path(args.data_config, default=ROOT / "configs/data_config.yaml"))
	model_cfg = _load_yaml(_resolve_path(args.model_config, default=ROOT / "configs/anomaly_detection/model.yaml"))
	train_cfg = _load_yaml(_resolve_path(args.train_config, default=ROOT / "configs/anomaly_detection/train.yaml"))

	processed_dir = _resolve_path(
		args.processed_dir or train_cfg.get("processed_dir") or data_cfg.get("paths", {}).get("processed", {}).get("anomaly"),
		default=ROOT / "data/processed/anomaly_detection",
	)
	manifest = _resolve_path(
		args.manifest or train_cfg.get("manifest_path") or data_cfg.get("artifacts", {}).get("split_manifests", {}).get("anomaly_detection"),
		default=ROOT / "data/processed/splits/anomaly_detection.json",
	)
	norm_stats = _resolve_path(
		args.norm_stats or train_cfg.get("norm_stats_path") or data_cfg.get("artifacts", {}).get("normalization_files", {}).get("anomaly_detection"),
		default=ROOT / "data/processed/normalization/anomaly_detection_norm.json",
	)

	open_file_lru_size = int(
		args.open_file_lru_size
		if args.open_file_lru_size is not None
		else train_cfg.get("open_file_lru_size", 6)
	)
	if sys.platform == "win32" and open_file_lru_size > 0:
		_log.warning("Windows detected: forcing open_file_lru_size=0 to avoid NetCDF handle crash")
		open_file_lru_size = 0

	train_ds = _build_split_dataset(
		split="train",
		processed_dir=processed_dir,
		manifest=manifest,
		norm_stats=norm_stats,
		open_file_lru_size=open_file_lru_size,
	)
	val_ds = _build_split_dataset(
		split="val",
		processed_dir=processed_dir,
		manifest=manifest,
		norm_stats=norm_stats,
		open_file_lru_size=open_file_lru_size,
	)

	train_ds = _subset_if_needed(train_ds, args.max_train_samples)
	val_ds = _subset_if_needed(val_ds, args.max_val_samples)

	if len(train_ds) == 0:
		raise SystemExit("train dataset is empty")
	if len(val_ds) == 0:
		raise SystemExit("val dataset is empty")

	batch_size = int(args.batch_size if args.batch_size is not None else train_cfg.get("batch_size", 8))
	num_workers = int(args.num_workers if args.num_workers is not None else train_cfg.get("num_workers", 0))
	train_loader = DataLoader(
		train_ds,
		batch_size=batch_size,
		shuffle=True,
		num_workers=num_workers,
		pin_memory=torch.cuda.is_available(),
	)
	val_loader = DataLoader(
		val_ds,
		batch_size=batch_size,
		shuffle=False,
		num_workers=num_workers,
		pin_memory=torch.cuda.is_available(),
	)

	model = DualBranchAutoEncoder(
		oper_channels=int(model_cfg.get("oper_channels", 3)),
		wave_channels=int(model_cfg.get("wave_channels", 3)),
		base_channels=int(model_cfg.get("base_channels", 24)),
	)

	output_dir = _resolve_path(args.output_dir or train_cfg.get("output_dir"), default=ROOT / "outputs/anomaly_detection")
	cfg = AnomalyTrainConfig(
		lr=float(args.lr if args.lr is not None else train_cfg.get("lr", 1e-3)),
		epochs=int(args.epochs if args.epochs is not None else train_cfg.get("epochs", 10)),
		batch_size=batch_size,
		num_workers=num_workers,
		device=_resolve_device(str(args.device or train_cfg.get("device", "auto"))),
		threshold_quantile=float(train_cfg.get("threshold_quantile", 0.95)),
		output_dir=output_dir,
		save_name=str(train_cfg.get("save_name", "anomaly_ae_best.pt")),
	)

	_log.info("Start anomaly training: train=%d val=%d", len(train_ds), len(val_ds))
	_log.info("Train config: epochs=%d batch_size=%d device=%s", cfg.epochs, batch_size, cfg.device)

	result = fit(model, train_loader, val_loader, cfg)

	# 载入最佳权重后导出 split 级别报告
	ckpt = torch.load(result["checkpoint"], map_location=cfg.device)
	model.load_state_dict(ckpt["model_state_dict"])
	model.to(cfg.device)
	threshold = float(result["best_threshold"])

	split_reports: dict[str, Any] = {}
	labels_map: dict[str, list[int]] = {}
	if args.labels_json:
		lp = _resolve_path(args.labels_json, default=ROOT / "outputs/anomaly_detection/labels.json")
		if lp.is_file():
			obj = json.loads(lp.read_text(encoding="utf-8"))
			if isinstance(obj, dict):
				labels_map = {k: [int(v) for v in vals] for k, vals in obj.items() if isinstance(vals, list)}

	events: list[dict[str, Any]] = []
	if args.events_json:
		ep = _resolve_path(args.events_json, default=ROOT / "outputs/anomaly_detection/events.json")
		if ep.is_file():
			obj = json.loads(ep.read_text(encoding="utf-8"))
			if isinstance(obj, list):
				events = [x for x in obj if isinstance(x, dict) and "start" in x and "end" in x]

	report_splits = [s.strip() for s in str(args.report_splits).split(",") if s.strip()]
	if not report_splits:
		report_splits = ["val"]

	if not args.skip_split_report:
		for split_name in report_splits:
			ds = _build_split_dataset(
				split=split_name,
				processed_dir=processed_dir,
				manifest=manifest,
				norm_stats=norm_stats,
				open_file_lru_size=open_file_lru_size,
			)
			if split_name == "train" and args.max_train_samples is not None:
				ds = _subset_if_needed(ds, args.max_train_samples)
			if split_name == "val" and args.max_val_samples is not None:
				ds = _subset_if_needed(ds, args.max_val_samples)
			if split_name == "test" and args.max_test_samples is not None:
				ds = _subset_if_needed(ds, args.max_test_samples)

			loader = DataLoader(
				ds,
				batch_size=batch_size,
				shuffle=False,
				num_workers=num_workers,
				pin_memory=torch.cuda.is_available(),
			)
			errors = infer_errors(model, loader, cfg.device)
			report = {
				"error_summary": summarize_errors(errors),
				"detection": build_detection_report(errors, threshold),
			}

			if split_name in labels_map:
				lb = np.asarray(labels_map[split_name], dtype=np.int64)
				if lb.shape[0] == errors.shape[0]:
					metrics = evaluate_with_labels(errors, lb, threshold)
					metrics["auc"] = roc_auc_from_scores(lb, errors)
					report["labeled_metrics"] = metrics

			if events:
				ts = _collect_timestamps(loader)
				flags = np.asarray(report["detection"]["flags"], dtype=np.int64)
				report["event_association"] = associate_events(ts, flags, events)

			split_reports[split_name] = report

	output_dir.mkdir(parents=True, exist_ok=True)
	(output_dir / "history.json").write_text(
		json.dumps(result["history"], ensure_ascii=False, indent=2),
		encoding="utf-8",
	)
	(output_dir / "summary.json").write_text(
		json.dumps({k: v for k, v in result.items() if k != "history"}, ensure_ascii=False, indent=2),
		encoding="utf-8",
	)
	(output_dir / "split_reports.json").write_text(
		json.dumps(split_reports, ensure_ascii=False, indent=2),
		encoding="utf-8",
	)

	final_dir = ROOT / "outputs/final_results/anomaly_detection"
	final_dir.mkdir(parents=True, exist_ok=True)
	(final_dir / "anomaly_summary.json").write_text(
		json.dumps(
			{
				"best_val_loss": result["best_val_loss"],
				"best_threshold": result["best_threshold"],
				"checkpoint": result["checkpoint"],
				"split_reports_file": str(output_dir / "split_reports.json"),
			},
			ensure_ascii=False,
			indent=2,
		),
		encoding="utf-8",
	)
	_log.info("Done. checkpoint=%s", result["checkpoint"])


if __name__ == "__main__":
	main()
