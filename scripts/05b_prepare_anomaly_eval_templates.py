"""生成 anomaly 评估模板：labels.json / events.json。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
	sys.path.insert(0, str(SRC))

from anomaly_detection.dataset import AnomalyFrameDataset
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


def _parse_splits(text: str) -> list[str]:
	splits = [s.strip() for s in str(text).split(",") if s.strip()]
	if not splits:
		raise ValueError("splits is empty")
	for s in splits:
		if s not in {"train", "val", "test"}:
			raise ValueError(f"unsupported split: {s}")
	return splits


def parse_args() -> argparse.Namespace:
	ap = argparse.ArgumentParser(description="Prepare anomaly labels/events json templates")
	ap.add_argument("--data-config", default="configs/data_config.yaml")
	ap.add_argument("--train-config", default="configs/anomaly_detection/train.yaml")
	ap.add_argument("--processed-dir", default=None)
	ap.add_argument("--manifest", default=None)
	ap.add_argument("--norm-stats", default=None)
	ap.add_argument("--open-file-lru-size", type=int, default=None)
	ap.add_argument("--splits", default="val,test", help="comma-separated splits from train,val,test")
	ap.add_argument("--max-samples-per-split", type=int, default=None)
	ap.add_argument("--label-init-value", type=int, default=-1, help="placeholder label value, usually -1")
	ap.add_argument("--output-dir", default="outputs/anomaly_detection/templates")
	ap.add_argument("--force", action="store_true", help="overwrite existing template files")
	return ap.parse_args()


def _subset_len(n: int, max_n: int | None) -> int:
	if max_n is None:
		return n
	if max_n <= 0:
		return n
	return min(n, int(max_n))


def main() -> None:
	args = parse_args()
	setup_logging(log_file=ROOT / "outputs/logs/anomaly_prepare_templates.log")

	data_cfg = _load_yaml(_resolve_path(args.data_config, default=ROOT / "configs/data_config.yaml"))
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

	splits = _parse_splits(args.splits)
	out_dir = _resolve_path(args.output_dir, default=ROOT / "outputs/anomaly_detection/templates")
	out_dir.mkdir(parents=True, exist_ok=True)

	labels_template: dict[str, list[int]] = {}
	split_meta: dict[str, dict[str, Any]] = {}
	global_ts: list[int] = []

	for split in splits:
		ds = AnomalyFrameDataset(
			processed_anomaly_dir=processed_dir,
			split=split,
			manifest_path=manifest,
			norm_stats_path=norm_stats if norm_stats.is_file() else None,
			root=ROOT,
			open_file_lru_size=open_file_lru_size,
		)
		n_total = len(ds)
		n_use = _subset_len(n_total, args.max_samples_per_split)
		labels_template[split] = [int(args.label_init_value)] * n_use

		ts_min: int | None = None
		ts_max: int | None = None
		for i in range(n_use):
			sample = ds[i]
			ts = sample.get("timestamp")
			if ts is None:
				continue
			try:
				tsi = int(ts)
			except Exception:
				continue
			if tsi < 0:
				continue
			global_ts.append(tsi)
			if ts_min is None or tsi < ts_min:
				ts_min = tsi
			if ts_max is None or tsi > ts_max:
				ts_max = tsi

		split_meta[split] = {
			"num_samples": int(n_use),
			"source_total_samples": int(n_total),
			"timestamp_min": ts_min,
			"timestamp_max": ts_max,
		}
		ds.close()

	if global_ts:
		ev_start = int(min(global_ts))
		ev_end = int(max(global_ts))
	else:
		ev_start, ev_end = 0, 0

	events_template: list[dict[str, Any]] = [
		{
			"name": "typhoon_example",
			"start": ev_start,
			"end": ev_end,
			"note": "replace start/end with real typhoon time window",
		}
	]

	labels_path = out_dir / "labels.template.json"
	events_path = out_dir / "events.template.json"
	meta_path = out_dir / "template_meta.json"

	for p in (labels_path, events_path, meta_path):
		if p.exists() and not args.force:
			raise SystemExit(f"{p} already exists; use --force to overwrite")

	labels_path.write_text(json.dumps(labels_template, ensure_ascii=False, indent=2), encoding="utf-8")
	events_path.write_text(json.dumps(events_template, ensure_ascii=False, indent=2), encoding="utf-8")
	meta_path.write_text(
		json.dumps(
			{
				"processed_dir": str(processed_dir),
				"manifest": str(manifest),
				"splits": splits,
				"label_init_value": int(args.label_init_value),
				"split_meta": split_meta,
			},
			ensure_ascii=False,
			indent=2,
		),
		encoding="utf-8",
	)

	_log.info("Wrote labels template: %s", labels_path)
	_log.info("Wrote events template: %s", events_path)
	_log.info("Wrote template meta: %s", meta_path)


if __name__ == "__main__":
	main()
