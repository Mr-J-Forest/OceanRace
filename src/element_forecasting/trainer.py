"""要素长期预测训练入口。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Subset

from element_forecasting.dataset import ElementForecastWindowDataset
from element_forecasting.evaluator import compute_regression_metrics_masked, masked_mse
from element_forecasting.model import HybridElementForecastModel
from utils.logger import get_logger, setup_logging, tqdm, tqdm_logging
import torch.nn.functional as F

_log = get_logger(__name__)

def masked_fft_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
	"""计算时序上幅频特性的 L1 差异，迫使模型保留高频分布特征，防止曲线随时间步无限平滑。"""
	# shape (B, T, C, H, W)，沿 T 维度 (dim=1) 做 FFT
	p = pred * mask
	t = target * mask
	p_fft = torch.fft.rfft(p, dim=1)
	t_fft = torch.fft.rfft(t, dim=1)
	return F.l1_loss(torch.abs(p_fft), torch.abs(t_fft))


def _load_yaml(path: Path) -> dict[str, Any]:
	if not path.is_file():
		return {}
	cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
	return cfg if isinstance(cfg, dict) else {}


def _as_var_names(value: Any) -> tuple[str, ...]:
	if isinstance(value, (list, tuple)):
		items = [str(v).strip() for v in value if str(v).strip()]
	elif isinstance(value, str):
		items = [v.strip() for v in value.split(",") if v.strip()]
	else:
		items = []
	if not items:
		raise ValueError("var_names is empty; configure variables in train.yaml/model.yaml")
	return tuple(items)


def resolve_core_config(
	*,
	args_var_names: str | None,
	args_input_steps: int | None,
	args_output_steps: int | None,
	args_window_stride: int | None,
	args_stitch_across_files: bool | None,
	train_cfg: dict[str, Any],
	model_cfg: dict[str, Any],
) -> dict[str, Any]:
	var_names = _as_var_names(args_var_names or train_cfg.get("var_names") or model_cfg.get("var_names"))
	input_steps = int(args_input_steps or train_cfg.get("input_steps") or model_cfg.get("input_steps", 12))
	output_steps = int(args_output_steps or train_cfg.get("output_steps") or model_cfg.get("output_steps", 12))
	window_stride = int(args_window_stride or train_cfg.get("window_stride", 1))
	if args_stitch_across_files is None:
		stitch_across_files = bool(train_cfg.get("stitch_across_files", True))
	else:
		stitch_across_files = bool(args_stitch_across_files)
	return {
		"var_names": var_names,
		"input_steps": input_steps,
		"output_steps": output_steps,
		"window_stride": window_stride,
		"stitch_across_files": stitch_across_files,
	}


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
	x = torch.stack([b["x"] for b in batch], dim=0)
	y = torch.stack([b["y"] for b in batch], dim=0)
	y_valid = torch.stack([b["y_valid"] for b in batch], dim=0)
	return {
		"x": x,
		"y": y,
		"y_valid": y_valid,
		"t0": torch.tensor([b["t0"] for b in batch], dtype=torch.long),
		"paths": [b["path"] for b in batch],
	}


def run_training(args: argparse.Namespace) -> None:
	root = Path(__file__).resolve().parents[2]
	data_cfg = _load_yaml(args.data_config)
	model_cfg = _load_yaml(args.model_config)
	train_cfg = _load_yaml(args.train_config)

	core = resolve_core_config(
		args_var_names=args.var_names,
		args_input_steps=args.input_steps,
		args_output_steps=args.output_steps,
		args_window_stride=args.window_stride,
		args_stitch_across_files=args.stitch_across_files,
		train_cfg=train_cfg,
		model_cfg=model_cfg,
	)
	var_names = core["var_names"]
	input_steps = core["input_steps"]
	output_steps = core["output_steps"]
	window_stride = core["window_stride"]
	stitch_across_files = core["stitch_across_files"]
	open_file_lru_size = int(
		args.open_file_lru_size
		if args.open_file_lru_size is not None
		else train_cfg.get("open_file_lru_size", 16)
	)
	single_file_train_ratio = float(
		train_cfg.get("single_file_train_ratio", data_cfg.get("split", {}).get("train_ratio", 0.7))
	)
	single_file_val_ratio = float(
		train_cfg.get("single_file_val_ratio", data_cfg.get("split", {}).get("val_ratio", 0.15))
	)

	processed_dir = Path(
		args.processed_dir
		or train_cfg.get("processed_dir")
		or data_cfg.get("paths", {}).get("processed", {}).get("element_forecasting", "data/processed/element_forecasting")
	)
	if not processed_dir.is_absolute():
		processed_dir = root / processed_dir

	data_file: Path | None = None
	data_file_str = args.data_file or train_cfg.get("data_file")
	if data_file_str:
		data_file = Path(data_file_str)
		if not data_file.is_absolute():
			data_file = root / data_file

	manifest_path = Path(
		args.manifest
		or train_cfg.get("manifest_path")
		or data_cfg.get("artifacts", {}).get("split_manifests", {}).get("element_forecasting", "data/processed/splits/element_forecasting.json")
	)
	if not manifest_path.is_absolute():
		manifest_path = root / manifest_path

	norm_path_str = (
		args.norm
		or train_cfg.get("norm_stats_path")
		or data_cfg.get("artifacts", {}).get("normalization_files", {}).get("element_forecasting", "data/processed/normalization/element_forecasting_norm.json")
	)
	norm_path = Path(norm_path_str)
	if not norm_path.is_absolute():
		norm_path = root / norm_path
	if not norm_path.is_file():
		norm_path = None

	out_dir = Path(args.output_dir or train_cfg.get("output_dir", "outputs/element_forecasting"))
	if not out_dir.is_absolute():
		out_dir = root / out_dir
	ckpt_dir = out_dir / "checkpoints"
	metrics_dir = out_dir / "metrics"
	ckpt_dir.mkdir(parents=True, exist_ok=True)
	metrics_dir.mkdir(parents=True, exist_ok=True)

	log_file = root / "outputs/logs/element_forecasting_train.log"
	setup_logging(log_file=log_file)

	if data_file is not None:
		full_ds = ElementForecastWindowDataset(
			data_file=data_file,
			var_names=var_names,
			input_steps=input_steps,
			output_steps=output_steps,
			window_stride=window_stride,
			stitch_across_files=stitch_across_files,
			open_file_lru_size=open_file_lru_size,
			split=None,
			norm_stats_path=norm_path,
			root=root,
		)
		n_total = len(full_ds)
		if n_total < 2:
			raise SystemExit("single data file has too few windows (<2); adjust input/output steps or provide longer series")

		n_train = int(n_total * single_file_train_ratio)
		n_val = int(n_total * single_file_val_ratio)
		n_train = min(max(1, n_train), n_total - 1)
		n_val = max(1, n_val)
		if n_train + n_val > n_total:
			n_val = n_total - n_train
		if n_val <= 0:
			raise SystemExit("single-file split produced empty val set; lower single_file_train_ratio or increase sequence length")

		train_indices = list(range(0, n_train))
		val_indices = list(range(n_total - n_val, n_total))
		train_ds = Subset(full_ds, train_indices)
		val_ds = Subset(full_ds, val_indices)
	else:
		train_ds = ElementForecastWindowDataset(
			processed_dir=processed_dir,
			var_names=var_names,
			input_steps=input_steps,
			output_steps=output_steps,
			window_stride=window_stride,
			stitch_across_files=stitch_across_files,
			open_file_lru_size=open_file_lru_size,
			split="train",
			manifest_path=manifest_path,
			norm_stats_path=norm_path,
			root=root,
		)
		val_ds = ElementForecastWindowDataset(
			processed_dir=processed_dir,
			var_names=var_names,
			input_steps=input_steps,
			output_steps=output_steps,
			window_stride=window_stride,
			stitch_across_files=stitch_across_files,
			open_file_lru_size=open_file_lru_size,
			split="val",
			manifest_path=manifest_path,
			norm_stats_path=norm_path,
			root=root,
		)
	if len(train_ds) == 0:
		raise SystemExit("train dataset is empty; check split manifest or time window settings")

	batch_size = int(args.batch_size if args.batch_size is not None else train_cfg.get("batch_size", 2))
	num_workers = int(args.num_workers if args.num_workers is not None else train_cfg.get("num_workers", 0))
	device = args.device or train_cfg.get("device", "auto")
	if device == "auto":
		device = "cuda" if torch.cuda.is_available() else "cpu"
	if str(device).startswith("cuda"):
		torch.backends.cuda.matmul.allow_tf32 = True
		torch.backends.cudnn.allow_tf32 = True
		torch.backends.cudnn.benchmark = True
		torch.set_float32_matmul_precision("high")
	pin_memory = str(device).startswith("cuda")
	persistent_workers = num_workers > 0
	loader_kwargs = {
		"num_workers": num_workers,
		"pin_memory": pin_memory,
		"persistent_workers": persistent_workers,
	}
	if num_workers > 0:
		loader_kwargs["prefetch_factor"] = 2
	train_loader = DataLoader(
		train_ds,
		batch_size=batch_size,
		shuffle=True,
		collate_fn=_collate,
		**loader_kwargs,
	)
	val_loader = DataLoader(
		val_ds,
		batch_size=batch_size,
		shuffle=False,
		collate_fn=_collate,
		**loader_kwargs,
	)

	sample = train_ds[0]
	in_channels = int(sample["x"].shape[1])

	model = HybridElementForecastModel(
		in_channels=in_channels,
		input_steps=input_steps,
		output_steps=output_steps,
		d_model=int(model_cfg.get("d_model", 128)),
		nhead=int(model_cfg.get("nhead", 4)),
		num_layers=int(model_cfg.get("num_layers", 6)),
		block_size=int(model_cfg.get("block_size", 4)),
		dropout=float(model_cfg.get("dropout", 0.1)),
		spatial_downsample=int(model_cfg.get("spatial_downsample", 4)),
	).to(device)

	lr = float(args.lr or train_cfg.get("lr", 1e-4))
	epochs = int(args.epochs or train_cfg.get("epochs", 10))
	loss_main_weight = float(train_cfg.get("loss_main_weight", 1.0))
	loss_aux_transformer_weight = float(train_cfg.get("loss_aux_transformer_weight", 0.0))
	loss_fft_weight = float(train_cfg.get("loss_fft_weight", 0.1))
	grad_accum_steps = max(1, int(train_cfg.get("grad_accum_steps", 1)))
	amp_enabled = bool(train_cfg.get("amp", True)) and str(device).startswith("cuda")
	amp_device_type = "cuda" if str(device).startswith("cuda") else "cpu"
	optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
	scaler = torch.amp.GradScaler(amp_device_type, enabled=amp_enabled)

	best_val = float("inf")
	best_path = ckpt_dir / "hybrid_best.pt"
	history: list[dict[str, float]] = []

	_log.info("=" * 60)
	_log.info("🚀 START TRAINING: Element Forecasting")
	_log.info("=" * 60)
	_log.info(f"⚙️  Model    : {in_channels} channels, vars={list(var_names)}")
	_log.info(f"⏱️  Steps    : {input_steps} (in) -> {output_steps} (out)")
	_log.info(f"📦 Data     : {len(train_ds)} train windows | {len(val_ds)} val windows")
	_log.info(f"🚀 Training : amp={amp_enabled}, grad_accum={grad_accum_steps}, bs={train_loader.batch_size}")
	_log.info("=" * 60)

	for epoch in range(1, epochs + 1):
		model.train()
		train_loss_sum = 0.0
		train_count = 0
		optimizer.zero_grad(set_to_none=True)
		with tqdm_logging():
			train_pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs:02d} [Train]", ncols=100, leave=False)
			for step, batch in enumerate(train_pbar, start=1):
				x = batch["x"].to(device).nan_to_num(0.0)
				y = batch["y"].to(device).nan_to_num(0.0)
				y_valid = batch["y_valid"].to(device)
				t0 = batch["t0"].to(device)
				try:
					with torch.amp.autocast(amp_device_type, enabled=amp_enabled):
						out = model(x, t0=t0)
						pred = out["pred"]
						loss_main = masked_mse(pred, y, y_valid)
						loss_fft = masked_fft_loss(pred, y, y_valid)
						loss = loss_main_weight * loss_main + loss_fft_weight * loss_fft
						if loss_aux_transformer_weight != 0.0:
							loss_aux = masked_mse(out["pred_transformer"], y, y_valid)
							loss = loss + loss_aux_transformer_weight * loss_aux
				except torch.OutOfMemoryError as ex:
					if str(device).startswith("cuda"):
						torch.cuda.empty_cache()
					raise RuntimeError(
						"CUDA OOM during forward. Try reducing batch_size, increasing spatial_downsample, reducing d_model/num_layers, or disabling CUDA by --device cpu."
					) from ex

				scaled_loss = loss / grad_accum_steps
				scaler.scale(scaled_loss).backward()

				if step % grad_accum_steps == 0:
					scaler.unscale_(optimizer)
					torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
					scaler.step(optimizer)
					scaler.update()
					optimizer.zero_grad(set_to_none=True)
					
				if torch.isnan(loss):
					_log.error(f"NaN loss detected at step {step}! Input has NaN: {torch.isnan(x).any()}, Target has NaN: {torch.isnan(y).any()}")
					# stop early for debug
					return

				train_loss_sum += float(loss.item()) * x.size(0)
				train_count += x.size(0)
				train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

		if train_count > 0 and (len(train_loader) % grad_accum_steps != 0):
			scaler.unscale_(optimizer)
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
			scaler.step(optimizer)
			scaler.update()
			optimizer.zero_grad(set_to_none=True)

		train_loss = train_loss_sum / max(train_count, 1)

		model.eval()
		val_loss_sum = 0.0
		val_count = 0
		val_metrics_sum = {"mse": 0.0, "mae": 0.0, "nse": 0.0}

		with torch.no_grad():
			with tqdm_logging():
				val_pbar = tqdm(val_loader, desc=f"Epoch {epoch:02d}/{epochs:02d} [Val  ]", ncols=100, leave=False)
				for batch in val_pbar:
					x = batch["x"].to(device)
					y = batch["y"].to(device)
					y_valid = batch["y_valid"].to(device)
					t0 = batch["t0"].to(device)
					pred = model(x, t0=t0)["pred"]
					loss = masked_mse(pred, y, y_valid)
					bs = x.size(0)
					val_loss_sum += float(loss.item()) * bs
					val_count += bs

					# 逐批次计算并累加，防止全部 concat 导致 OOM
					batch_metrics = compute_regression_metrics_masked(pred, y, y_valid)
					for k in val_metrics_sum:
						val_metrics_sum[k] += batch_metrics[k] * bs
					
					val_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

		val_loss = val_loss_sum / max(val_count, 1)
		metrics = {k: v / max(val_count, 1) for k, v in val_metrics_sum.items()}
		metrics["rmse"] = metrics["mse"] ** 0.5

		epoch_record = {
			"epoch": float(epoch),
			"train_loss": float(train_loss),
			"val_loss": float(val_loss),
			"val_mse": float(metrics["mse"]),
			"val_rmse": float(metrics["rmse"]),
			"val_mae": float(metrics["mae"]),
			"val_nse": float(metrics["nse"]),
		}
		history.append(epoch_record)
		
		# 精简的单行输出，取消各种花哨标签，只留核心数据
		_log.info(
			f"Epoch {epoch:02d}/{epochs:02d} | "
			f"Train Loss: {train_loss:.4f} | "
			f"Val Loss: {val_loss:.4f} | "
			f"RMSE: {metrics['rmse']:.4f} | "
			f"MAE: {metrics['mae']:.4f}"
		)

		if val_loss < best_val:
			best_val = val_loss
			torch.save(
				{
					"model_state": model.state_dict(),
					"var_names": list(var_names),
					"input_steps": input_steps,
					"output_steps": output_steps,
					"in_channels": in_channels,
					"model_config": model_cfg,
				},
				best_path,
			)
			_log.info("new best checkpoint saved: %s", best_path)

	(metrics_dir / "train_history.json").write_text(
		json.dumps(history, ensure_ascii=False, indent=2),
		encoding="utf-8",
	)
	
	_log.info("=" * 60)
	_log.info("🎉 TRAINING COMPLETED 🎉")
	_log.info("=" * 60)
	_log.info(f"🏆 Best Validation Loss : {best_val:.6f}")
	_log.info(f"💾 Checkpoint Saved To  : {best_path}")
	_log.info(f"📊 Training History     : {metrics_dir / 'train_history.json'}")
	_log.info("=" * 60)


def main() -> None:
	root = Path(__file__).resolve().parents[2]
	ap = argparse.ArgumentParser(description="Hybrid long-term forecasting trainer")
	ap.add_argument("--data-config", type=Path, default=root / "configs/data_config.yaml")
	ap.add_argument("--model-config", type=Path, default=root / "configs/element_forecasting/model.yaml")
	ap.add_argument("--train-config", type=Path, default=root / "configs/element_forecasting/train.yaml")
	ap.add_argument("--processed-dir", type=str, default=None)
	ap.add_argument("--data-file", type=str, default=None, help="单一 NetCDF 文件路径；设置后将忽略 manifest/split")
	ap.add_argument("--manifest", type=str, default=None)
	ap.add_argument("--norm", type=str, default=None)
	ap.add_argument("--var-names", type=str, default=None, help="逗号分隔变量名，输入变量=输出变量")
	ap.add_argument("--input-steps", type=int, default=None)
	ap.add_argument("--output-steps", type=int, default=None)
	ap.add_argument("--window-stride", type=int, default=None)
	ap.add_argument("--open-file-lru-size", type=int, default=None, help="文件句柄 LRU 大小")
	ap.add_argument(
		"--stitch-across-files",
		action=argparse.BooleanOptionalAction,
		default=None,
		help="是否跨文件拼接时间轴后再切窗（默认读取 train.yaml）",
	)
	ap.add_argument("--epochs", type=int, default=None)
	ap.add_argument("--batch-size", type=int, default=None)
	ap.add_argument("--lr", type=float, default=None)
	ap.add_argument("--num-workers", type=int, default=None)
	ap.add_argument("--device", type=str, default=None)
	ap.add_argument("--output-dir", type=str, default=None)
	args = ap.parse_args()
	run_training(args)


if __name__ == "__main__":
	main()
