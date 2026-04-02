"""要素长期预测训练入口。"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from element_forecasting.dataset import ElementForecastWindowDataset
from element_forecasting.evaluator import (
	compute_regression_metrics_masked,
	masked_mse,
	masked_spatial_mean_mse,
	masked_weighted_mse,
)
from element_forecasting.model import HybridElementForecastModel
from utils.logger import get_logger, setup_logging, tqdm, tqdm_logging

_log = get_logger(__name__)


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
	train_cfg: dict[str, Any],
	model_cfg: dict[str, Any],
) -> dict[str, Any]:
	var_names = _as_var_names(args_var_names or train_cfg.get("var_names") or model_cfg.get("var_names"))
	input_steps = int(args_input_steps or train_cfg.get("input_steps") or model_cfg.get("input_steps", 12))
	output_steps = int(args_output_steps or train_cfg.get("output_steps") or model_cfg.get("output_steps", 12))
	window_stride = int(args_window_stride or train_cfg.get("window_stride", 1))
	return {
		"var_names": var_names,
		"input_steps": input_steps,
		"output_steps": output_steps,
		"window_stride": window_stride,
	}


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
	x = torch.stack([b["x"] for b in batch], dim=0)
	y = torch.stack([b["y"] for b in batch], dim=0)
	y_valid = torch.stack([b["y_valid"] for b in batch], dim=0)
	return {
		"x": x,
		"y": y,
		"y_valid": y_valid,
		"paths": [b["path"] for b in batch],
	}


def _scheduled_sampling_epsilon(
	*,
	epoch: int,
	total_epochs: int,
	enabled: bool,
	start_epoch: int,
	epsilon_start: float,
	epsilon_min: float,
	decay_type: str,
) -> float:
	if not enabled:
		return 1.0
	if epoch < start_epoch:
		return float(epsilon_start)
	epsilon_start = float(max(0.0, min(1.0, epsilon_start)))
	epsilon_min = float(max(0.0, min(1.0, epsilon_min)))
	if epsilon_start <= epsilon_min:
		return epsilon_min

	span = max(1, total_epochs - start_epoch)
	progress = min(1.0, max(0.0, (epoch - start_epoch + 1) / span))
	if decay_type == "cosine":
		factor = 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi)).item())
		epsilon = epsilon_min + (epsilon_start - epsilon_min) * factor
	else:
		epsilon = epsilon_start - (epsilon_start - epsilon_min) * progress
	return float(max(epsilon_min, min(epsilon_start, epsilon)))


def _mix_rollout_input(
	*,
	pred_chunk: torch.Tensor,
	target_chunk: torch.Tensor,
	epsilon: float,
	enabled: bool,
) -> torch.Tensor:
	if not enabled:
		return target_chunk
	if epsilon >= 1.0:
		return target_chunk
	if epsilon <= 0.0:
		return pred_chunk
	bsz = pred_chunk.shape[0]
	gate = (torch.rand(bsz, 1, 1, 1, 1, device=pred_chunk.device) < float(epsilon)).float()
	return gate * target_chunk + (1.0 - gate) * pred_chunk


def run_training(args: argparse.Namespace) -> None:
	root = Path(__file__).resolve().parents[2]
	data_cfg = _load_yaml(args.data_config)
	model_cfg = _load_yaml(args.model_config)
	train_cfg = _load_yaml(args.train_config)
	# Suppress a known PyTorch transformer warning that is harmless for this model and clutters console output.
	warnings.filterwarnings(
		"ignore",
		message="enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True",
		category=UserWarning,
	)

	core = resolve_core_config(
		args_var_names=args.var_names,
		args_input_steps=args.input_steps,
		args_output_steps=args.output_steps,
		args_window_stride=args.window_stride,
		train_cfg=train_cfg,
		model_cfg=model_cfg,
	)
	var_names = core["var_names"]
	input_steps = core["input_steps"]
	model_output_steps = core["output_steps"]
	rollout_steps = max(1, int(train_cfg.get("rollout_steps", 1)))
	train_target_steps = int(model_output_steps * rollout_steps)
	window_stride = core["window_stride"]
	open_file_lru_size = int(train_cfg.get("open_file_lru_size", train_cfg.get("dataset_cache_max_files", 16)))
	open_file_lru_size = max(1, open_file_lru_size)
	split_cfg = data_cfg.get("split", {})
	train_ratio = float(train_cfg.get("train_ratio", split_cfg.get("train_ratio", 0.7)))
	val_ratio = float(train_cfg.get("val_ratio", split_cfg.get("val_ratio", 0.15)))
	test_ratio = float(train_cfg.get("test_ratio", split_cfg.get("test_ratio", 0.15)))

	data_file = Path(
		args.data_file
		or train_cfg.get("data_file")
		or data_cfg.get("paths", {}).get("processed", {}).get("element_forecasting", "data/processed/element_forecasting/all_clean_merged.nc")
	)
	if not data_file.is_absolute():
		data_file = root / data_file

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

	train_ds = ElementForecastWindowDataset(
		data_file=data_file,
		var_names=var_names,
		input_steps=input_steps,
		output_steps=train_target_steps,
		window_stride=window_stride,
		open_file_lru_size=open_file_lru_size,
		split="train",
		split_ratios=(train_ratio, val_ratio, test_ratio),
		norm_stats_path=norm_path,
		root=root,
	)
	val_ds = ElementForecastWindowDataset(
		data_file=data_file,
		var_names=var_names,
		input_steps=input_steps,
		output_steps=model_output_steps,
		window_stride=window_stride,
		open_file_lru_size=open_file_lru_size,
		split="val",
		split_ratios=(train_ratio, val_ratio, test_ratio),
		norm_stats_path=norm_path,
		root=root,
	)
	if len(train_ds) == 0:
		raise SystemExit("train dataset is empty; check data file and split/time-window settings")

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
	rollout_gamma = float(train_cfg.get("rollout_gamma", 1.0))
	rollout_gamma = max(0.0, min(1.0, rollout_gamma))
	rollout_detach_between_steps = bool(train_cfg.get("rollout_detach_between_steps", False))
	ss_enabled = bool(train_cfg.get("scheduled_sampling_enabled", False))
	ss_start_epoch = max(1, int(train_cfg.get("scheduled_sampling_start_epoch", 1)))
	ss_epsilon_start = float(train_cfg.get("scheduled_sampling_epsilon_start", 1.0))
	ss_epsilon_min = float(train_cfg.get("scheduled_sampling_epsilon_min", 0.3))
	ss_decay_type = str(train_cfg.get("scheduled_sampling_decay_type", "linear")).strip().lower()
	progress_bar_enabled = bool(train_cfg.get("progress_bar_enabled", True))
	progress_bar_mininterval = float(train_cfg.get("progress_bar_mininterval", 1.5))
	progress_bar_mininterval = max(0.2, progress_bar_mininterval)
	progress_bar_leave = bool(train_cfg.get("progress_bar_leave", False))
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
		output_steps=model_output_steps,
		d_model=int(model_cfg.get("d_model", 128)),
		nhead=int(model_cfg.get("nhead", 4)),
		num_layers=int(model_cfg.get("num_layers", 6)),
		block_size=int(model_cfg.get("block_size", 4)),
		dropout=float(model_cfg.get("dropout", 0.1)),
		spatial_downsample=int(model_cfg.get("spatial_downsample", 4)),
		periodic_periods=model_cfg.get("periodic_periods", [24.0]),
		periodic_harmonics=int(model_cfg.get("periodic_harmonics", 1)),
	).to(device)

	lr = float(args.lr or train_cfg.get("lr", 1e-4))
	epochs = int(args.epochs or train_cfg.get("epochs", 10))
	loss_main_weight = float(train_cfg.get("loss_main_weight", 1.0))
	loss_aux_transformer_weight = float(train_cfg.get("loss_aux_transformer_weight", 0.2))
	loss_spatial_mean_weight = float(train_cfg.get("loss_spatial_mean_weight", 0.2))
	loss_aux_spatial_mean_weight = float(train_cfg.get("loss_aux_spatial_mean_weight", 0.05))
	var_loss_weights_cfg = train_cfg.get("var_loss_weights", {})
	if not isinstance(var_loss_weights_cfg, dict):
		var_loss_weights_cfg = {}
	channel_weights = torch.tensor(
		[float(var_loss_weights_cfg.get(v, 1.0)) for v in var_names],
		dtype=torch.float32,
		device=device,
	).view(1, 1, -1, 1, 1)
	grad_accum_steps = max(1, int(train_cfg.get("grad_accum_steps", 1)))
	amp_enabled = bool(train_cfg.get("amp", True)) and str(device).startswith("cuda")
	amp_device_type = "cuda" if str(device).startswith("cuda") else "cpu"
	optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
	scaler = torch.amp.GradScaler(amp_device_type, enabled=amp_enabled)

	best_val = float("inf")
	best_path = ckpt_dir / "hybrid_best.pt"
	history: list[dict[str, float]] = []

	_log.info(
		"start training | vars=%s | in/out=%d/%d | rollout=%d(gamma=%.2f) | ss=%s(eps %.2f->%.2f) | amp=%s",
		list(var_names),
		input_steps,
		model_output_steps,
		rollout_steps,
		rollout_gamma,
		ss_enabled,
		ss_epsilon_start,
		ss_epsilon_min,
		amp_enabled,
	)
	_log.info(
		"data=%s | split=(%.2f,%.2f,%.2f) | windows train/val=%d/%d | batch=%d accum=%d workers=%d",
		str(data_file),
		train_ratio,
		val_ratio,
		test_ratio,
		len(train_ds),
		len(val_ds),
		batch_size,
		grad_accum_steps,
		num_workers,
	)
	_log.debug(
		"details | open_file_lru_size=%d | var_loss_weights=%s | loss_spatial_mean_weight=%.3f | loss_aux_spatial_mean_weight=%.3f | rollout_detach_between_steps=%s | ss_start_epoch=%d | progress_bar_enabled=%s mininterval=%.2f leave=%s",
		open_file_lru_size,
		{v: float(var_loss_weights_cfg.get(v, 1.0)) for v in var_names},
		loss_spatial_mean_weight,
		loss_aux_spatial_mean_weight,
		rollout_detach_between_steps,
		ss_start_epoch,
		progress_bar_enabled,
		progress_bar_mininterval,
		progress_bar_leave,
	)

	for epoch in range(1, epochs + 1):
		model.train()
		train_loss_sum = 0.0
		train_count = 0
		epsilon = _scheduled_sampling_epsilon(
			epoch=epoch,
			total_epochs=epochs,
			enabled=ss_enabled,
			start_epoch=ss_start_epoch,
			epsilon_start=ss_epsilon_start,
			epsilon_min=ss_epsilon_min,
			decay_type=ss_decay_type,
		)
		optimizer.zero_grad(set_to_none=True)
		pbar_disable = (not progress_bar_enabled)
		desc = f"epoch {epoch}/{epochs}"
		with tqdm_logging():
			for step, batch in enumerate(
				tqdm(
					train_loader,
					desc=desc,
					disable=pbar_disable,
					leave=progress_bar_leave,
					mininterval=progress_bar_mininterval,
					dynamic_ncols=True,
				),
				start=1,
			):
				x = batch["x"].to(device).nan_to_num(0.0)
				y = batch["y"].to(device).nan_to_num(0.0)
				y_valid = batch["y_valid"].to(device)
				try:
					with torch.amp.autocast(amp_device_type, enabled=amp_enabled):
						cur_x = x
						rollout_loss = x.new_zeros((), dtype=torch.float32)
						weight_sum = 0.0
						for rollout_idx in range(rollout_steps):
							t0 = rollout_idx * model_output_steps
							t1 = t0 + model_output_steps
							y_chunk = y[:, t0:t1]
							y_valid_chunk = y_valid[:, t0:t1]

							out = model(cur_x)
							pred = out["pred"]
							pred_transformer = out["pred_transformer"]
							loss_main = masked_weighted_mse(pred, y_chunk, y_valid_chunk, channel_weights=channel_weights)
							loss_aux = masked_weighted_mse(pred_transformer, y_chunk, y_valid_chunk, channel_weights=channel_weights)
							loss_main_mean = masked_spatial_mean_mse(pred, y_chunk, y_valid_chunk, channel_weights=channel_weights)
							loss_aux_mean = masked_spatial_mean_mse(pred_transformer, y_chunk, y_valid_chunk, channel_weights=channel_weights)
							step_loss = (
								loss_main_weight * loss_main
								+ loss_aux_transformer_weight * loss_aux
								+ loss_spatial_mean_weight * loss_main_mean
								+ loss_aux_spatial_mean_weight * loss_aux_mean
							)
							w = rollout_gamma ** rollout_idx
							rollout_loss = rollout_loss + (w * step_loss)
							weight_sum += w

							if rollout_idx < rollout_steps - 1:
								next_pred = pred.detach() if rollout_detach_between_steps else pred
								mixed = _mix_rollout_input(
									pred_chunk=next_pred,
									target_chunk=y_chunk,
									epsilon=epsilon,
									enabled=ss_enabled and epoch >= ss_start_epoch,
								)
								cur_x = torch.cat([cur_x, mixed], dim=1)[:, -input_steps:].contiguous()

						loss = rollout_loss / max(weight_sum, 1e-12)
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
			for batch in val_loader:
				x = batch["x"].to(device)
				y = batch["y"].to(device)
				y_valid = batch["y_valid"].to(device)
				pred = model(x)["pred"]
				loss = masked_mse(pred, y, y_valid)
				bs = x.size(0)
				val_loss_sum += float(loss.item()) * bs
				val_count += bs

				# 逐批次计算并累加，防止全部 concat 导致 OOM
				batch_metrics = compute_regression_metrics_masked(pred, y, y_valid)
				for k in val_metrics_sum:
					val_metrics_sum[k] += batch_metrics[k] * bs

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
		_log.info(
			"epoch=%d train_loss=%.6f val_loss=%.6f val_rmse=%.6f val_mae=%.6f val_nse=%.6f ss_epsilon=%.4f",
			epoch,
			train_loss,
			val_loss,
			metrics["rmse"],
			metrics["mae"],
			metrics["nse"],
			epsilon,
		)

		if val_loss < best_val:
			best_val = val_loss
			torch.save(
				{
					"model_state": model.state_dict(),
					"var_names": list(var_names),
					"input_steps": input_steps,
					"output_steps": model_output_steps,
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
	_log.info("training done | best_val=%.6f | checkpoint=%s", best_val, best_path)


def main() -> None:
	root = Path(__file__).resolve().parents[2]
	ap = argparse.ArgumentParser(description="Hybrid long-term forecasting trainer")
	ap.add_argument("--data-config", type=Path, default=root / "configs/data_config.yaml")
	ap.add_argument("--model-config", type=Path, default=root / "configs/element_forecasting/model.yaml")
	ap.add_argument("--train-config", type=Path, default=root / "configs/element_forecasting/train.yaml")
	ap.add_argument("--data-file", type=str, default=None)
	ap.add_argument("--norm", type=str, default=None)
	ap.add_argument("--var-names", type=str, default=None, help="逗号分隔变量名，输入变量=输出变量")
	ap.add_argument("--input-steps", type=int, default=None)
	ap.add_argument("--output-steps", type=int, default=None)
	ap.add_argument("--window-stride", type=int, default=None)
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
