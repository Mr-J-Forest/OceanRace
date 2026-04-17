"""中尺度涡旋分割训练入口。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
	sys.path.insert(0, str(SRC))

from eddy_detection.dataset import EddySegmentationDataset  # noqa: E402
from eddy_detection.model import EddyUNet  # noqa: E402
from eddy_detection.trainer import EddyTrainConfig, train_eddy_segmentation  # noqa: E402
from utils.logger import get_logger, setup_logging  # noqa: E402

_log = get_logger(__name__)


def _load_yaml(path: Path) -> dict[str, Any]:
	if not path.is_file():
		return {}
	data = yaml.safe_load(path.read_text(encoding="utf-8"))
	return data if isinstance(data, dict) else {}


def main() -> None:
	default_model_cfg = ROOT / "configs/eddy_detection/model.yaml"
	default_train_cfg = ROOT / "configs/eddy_detection/train.yaml"

	pre = argparse.ArgumentParser(add_help=False)
	pre.add_argument("--model-config", type=Path, default=default_model_cfg)
	pre.add_argument("--train-config", type=Path, default=default_train_cfg)
	pre_args, rest = pre.parse_known_args(sys.argv[1:])

	model_cfg = _load_yaml(pre_args.model_config)
	train_cfg = _load_yaml(pre_args.train_config)

	def _m(key: str, default: Any) -> Any:
		return model_cfg.get(key, default)

	def _t(key: str, default: Any) -> Any:
		return train_cfg.get(key, default)

	paths_cfg = train_cfg.get("paths", {}) if isinstance(train_cfg.get("paths", {}), dict) else {}

	ap = argparse.ArgumentParser(description="中尺度涡旋分割训练", parents=[pre])
	ap.add_argument("--epochs", type=int, default=int(_t("epochs", 8)))
	ap.add_argument("--batch-size", type=int, default=int(_t("batch_size", 4)))
	ap.add_argument("--lr", type=float, default=float(_t("lr", 1e-3)))
	ap.add_argument("--num-workers", type=int, default=int(_t("num_workers", 0)))
	ap.add_argument("--cpu-max-threads", type=int, default=int(_t("cpu_max_threads", 4)))
	ap.add_argument("--batch-sleep-ms", type=int, default=int(_t("batch_sleep_ms", 8)))
	ap.add_argument("--log-interval", type=int, default=int(_t("log_interval", 20)))
	ap.add_argument("--weight-decay", type=float, default=float(_t("weight_decay", 1e-4)))
	ap.add_argument("--dice-weight", type=float, default=float(_t("dice_weight", 0.5)))
	ap.add_argument("--ce-weight", type=float, default=float(_t("ce_weight", 0.5)))
	ap.add_argument("--boundary-weight", type=float, default=float(_t("boundary_weight", 0.1)))
	ap.add_argument("--input-steps", type=int, default=int(_m("input_steps", 1)))
	ap.add_argument("--step-stride", type=int, default=int(_m("step_stride", 1)))
	ap.add_argument("--max-train-samples", type=int, default=None)
	ap.add_argument("--max-val-samples", type=int, default=None)
	ap.add_argument("--base-channels", type=int, default=int(_m("base_channels", 32)))
	ap.add_argument("--num-classes", type=int, default=int(_m("num_classes", 3)))
	ap.add_argument(
		"--norm",
		type=Path,
		default=ROOT / "data/processed/normalization/eddy_norm.json",
	)
	ap.add_argument(
		"--manifest",
		type=Path,
		default=ROOT / "data/processed/splits/eddy.json",
	)
	ap.add_argument(
		"--time-split-manifest",
		type=Path,
		default=(ROOT / paths_cfg["time_split_manifest"]) if "time_split_manifest" in paths_cfg else None,
		help="单文件时间划分配置（scripts/02b_split_eddy_merged_by_time.py 产出）",
	)
	ap.add_argument(
		"--clean-nc",
		type=Path,
		default=(ROOT / paths_cfg["merged_clean_nc"]) if "merged_clean_nc" in paths_cfg else None,
		help="单文件模式：合并 clean.nc",
	)
	ap.add_argument(
		"--label-nc",
		type=Path,
		default=(ROOT / paths_cfg["merged_label_nc"]) if "merged_label_nc" in paths_cfg else None,
		help="单文件模式：对应 label.nc",
	)
	ap.add_argument(
		"--labels-dir",
		type=Path,
		default=ROOT / "data/processed/eddy_detection/labels",
	)
	ap.add_argument(
		"--out-dir",
		type=Path,
		default=ROOT / "outputs/eddy_detection",
	)
	ap.add_argument(
		"--log-file",
		type=Path,
		default=Path("outputs/logs/eddy_train.log"),
		help="训练日志路径（相对项目根目录或绝对路径）",
	)
	dev_default = _t("device", "auto")
	if dev_default == "auto":
		dev_default = "cuda" if torch.cuda.is_available() else "cpu"
	ap.add_argument("--device", type=str, default=dev_default)
	args = ap.parse_args(rest)

	log_file = args.log_file if args.log_file.is_absolute() else (ROOT / args.log_file)
	setup_logging(log_file=log_file)
	if args.device.startswith("cuda") and torch.cuda.is_available():
		_log.info("train device: %s (%s)", args.device, torch.cuda.get_device_name(0))
	else:
		_log.info("train device: %s", args.device)

	if args.cpu_max_threads > 0:
		torch.set_num_threads(int(args.cpu_max_threads))
		try:
			torch.set_num_interop_threads(max(1, int(args.cpu_max_threads // 2)))
		except RuntimeError:
			# set_num_interop_threads may only be called once per process; ignore safely.
			pass
		_log.info(
			"cpu thread limits: num_threads=%s interop_threads~=%s",
			int(args.cpu_max_threads),
			max(1, int(args.cpu_max_threads // 2)),
		)

	norm_path = args.norm if args.norm.is_file() else None
	if norm_path is None:
		_log.warning("norm file missing, train without standardization: %s", args.norm)

	use_merged_mode = args.time_split_manifest is not None or args.clean_nc is not None
	if use_merged_mode:
		train_ds = EddySegmentationDataset(
			split="train",
			input_steps=args.input_steps,
			step_stride=args.step_stride,
			max_samples=args.max_train_samples,
			time_split_manifest_path=args.time_split_manifest,
			clean_nc_path=args.clean_nc,
			label_nc_path=args.label_nc,
			norm_stats_path=norm_path,
			labels_dir=args.labels_dir,
			root=ROOT,
		)
		val_ds = EddySegmentationDataset(
			split="val",
			input_steps=args.input_steps,
			step_stride=args.step_stride,
			max_samples=args.max_val_samples,
			time_split_manifest_path=args.time_split_manifest,
			clean_nc_path=args.clean_nc,
			label_nc_path=args.label_nc,
			norm_stats_path=norm_path,
			labels_dir=args.labels_dir,
			root=ROOT,
		)
	else:
		train_ds = EddySegmentationDataset(
			split="train",
			input_steps=args.input_steps,
			step_stride=args.step_stride,
			max_samples=args.max_train_samples,
			manifest_path=args.manifest,
			norm_stats_path=norm_path,
			labels_dir=args.labels_dir,
			root=ROOT,
		)
		val_ds = EddySegmentationDataset(
			split="val",
			input_steps=args.input_steps,
			step_stride=args.step_stride,
			max_samples=args.max_val_samples,
			manifest_path=args.manifest,
			norm_stats_path=norm_path,
			labels_dir=args.labels_dir,
			root=ROOT,
		)

	if len(train_ds) == 0:
		raise SystemExit(
			"train dataset empty. Please run:\n"
			"1) python scripts/02c_generate_meta4_labels.py --clean-nc data/processed/eddy_detection/19930101_20241231_clean.nc --out-nc data/processed/eddy_detection/labels/19930101_20241231_label_meta4_mask.nc\n"
			"2) python scripts/02h_fix_meta4_mask_background.py --input-nc data/processed/eddy_detection/labels/19930101_20241231_label_meta4_mask.nc --output-nc data/processed/eddy_detection/labels/19930101_20241231_label_meta4_mask_bg0.nc\n"
			"3) python scripts/02b_split_eddy_merged_by_time.py --clean-nc data/processed/eddy_detection/19930101_20241231_clean.nc --label-nc data/processed/eddy_detection/labels/19930101_20241231_label_meta4_mask_bg0.nc"
		)

	in_channels = args.input_steps * 3
	model = EddyUNet(
		in_channels=in_channels,
		num_classes=args.num_classes,
		base_channels=args.base_channels,
	)
	cfg = EddyTrainConfig(
		epochs=args.epochs,
		batch_size=args.batch_size,
		lr=args.lr,
		num_workers=args.num_workers,
		weight_decay=args.weight_decay,
		dice_weight=args.dice_weight,
		ce_weight=args.ce_weight,
		boundary_weight=max(0.0, float(args.boundary_weight)),
		log_interval=args.log_interval,
		device=args.device,
		batch_sleep_ms=max(0, int(args.batch_sleep_ms)),
	)
	best = train_eddy_segmentation(model, train_ds, val_ds, cfg, args.out_dir)
	_log.info("training complete, best=%s", best)


if __name__ == "__main__":
	main()
