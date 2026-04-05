"""要素预测模块单测：数据窗、模型前向、配置覆盖。"""
from __future__ import annotations

import json

import torch

from element_forecasting.dataset import ElementForecastWindowDataset
from element_forecasting.evaluator import masked_spatial_mean_mse, masked_weighted_mse
from element_forecasting.model import HybridElementForecastModel
from element_forecasting.predictor import ElementForecastPredictor
from element_forecasting.trainer import (
    _mix_rollout_input,
    _scheduled_sampling_epsilon,
    resolve_core_config,
)
from tests.conftest import write_element_clean_nc


def test_dataset_single_file_split_windows(tmp_path) -> None:
    data_file = tmp_path / "data/processed/element_forecasting/all_clean_merged.nc"
    data_file.parent.mkdir(parents=True)
    write_element_clean_nc(data_file, t=10, base=0.0, vars_names=("sst",))

    # t=10, input+output=4 -> total windows=7
    # split ratios 0.5/0.25/0.25 => train=3, val=1, test=3
    ds_train = ElementForecastWindowDataset(
        data_file=data_file,
        var_names=("sst",),
        input_steps=2,
        output_steps=2,
        window_stride=1,
        split="train",
        split_ratios=(0.5, 0.25, 0.25),
        root=tmp_path,
    )
    ds_val = ElementForecastWindowDataset(
        data_file=data_file,
        var_names=("sst",),
        input_steps=2,
        output_steps=2,
        window_stride=1,
        split="val",
        split_ratios=(0.5, 0.25, 0.25),
        root=tmp_path,
    )
    ds_test = ElementForecastWindowDataset(
        data_file=data_file,
        var_names=("sst",),
        input_steps=2,
        output_steps=2,
        window_stride=1,
        split="test",
        split_ratios=(0.5, 0.25, 0.25),
        root=tmp_path,
    )

    assert len(ds_train) == 3
    assert len(ds_val) == 1
    assert len(ds_test) == 3

    s = ds_train[1]
    x = s["x"][:, 0, 0, 0]
    y = s["y"][:, 0, 0, 0]
    # t0=1: x=[1,2], y=[3,4]
    assert torch.allclose(x, torch.tensor([1.0, 2.0]))
    assert torch.allclose(y, torch.tensor([3.0, 4.0]))


def test_hybrid_model_forward_shape() -> None:
    x = torch.randn(2, 4, 3, 5, 6)
    model = HybridElementForecastModel(
        in_channels=3,
        input_steps=4,
        output_steps=2,
        d_model=16,
        nhead=4,
        num_layers=2,
        block_size=4,
    )
    out = model(x)
    assert out["pred"].shape == (2, 2, 3, 5, 6)
    assert out["pred_transformer"].shape == (2, 2, 3, 5, 6)


def test_dataset_single_data_file_mode(tmp_path) -> None:
    data_file = tmp_path / "data/processed/element_forecasting/all_clean_merged.nc"
    data_file.parent.mkdir(parents=True)
    write_element_clean_nc(data_file, t=10, base=1.0, vars_names=("sst", "sss"))

    ds = ElementForecastWindowDataset(
        data_file=data_file,
        var_names=("sst", "sss"),
        input_steps=3,
        output_steps=2,
        window_stride=1,
        split=None,
        root=tmp_path,
    )
    # t=10, need=5 => windows=6
    assert len(ds) == 6

    s = ds[0]
    assert s["x"].shape == (3, 2, 2, 2)
    assert s["y"].shape == (2, 2, 2, 2)


def test_predictor_denormalize_outputs(tmp_path) -> None:
    model = HybridElementForecastModel(
        in_channels=1,
        input_steps=2,
        output_steps=1,
        d_model=8,
        nhead=2,
        num_layers=1,
        block_size=2,
    )
    ckpt_path = tmp_path / "fake_ckpt.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "var_names": ["sst"],
            "input_steps": 2,
            "output_steps": 1,
            "in_channels": 1,
            "model_config": {
                "d_model": 8,
                "nhead": 2,
                "num_layers": 1,
                "block_size": 2,
                "dropout": 0.0,
                "spatial_downsample": 4,
            },
        },
        ckpt_path,
    )

    norm_path = tmp_path / "norm.json"
    norm_path.write_text(
        json.dumps({"variables": {"sst": {"mean": 10.0, "std": 2.0}}}),
        encoding="utf-8",
    )

    predictor = ElementForecastPredictor(ckpt_path, device="cpu", norm_stats_path=norm_path)
    x = torch.zeros(1, 2, 1, 4, 4)

    out_std = predictor.predict(x, denormalize=False)
    out_denorm = predictor.predict(x, denormalize=True)

    assert out_std["pred"].shape == out_denorm["pred"].shape
    diff = out_denorm["pred"] - out_std["pred"]
    assert torch.isfinite(diff).all()
    # denorm = std * 2 + 10
    assert torch.allclose(out_denorm["pred"], out_std["pred"] * 2.0 + 10.0, atol=1e-5, rtol=1e-5)


def test_resolve_core_config_override_priority() -> None:
    model_cfg = {
        "var_names": ["m1", "m2"],
        "input_steps": 8,
        "output_steps": 4,
    }
    train_cfg = {
        "var_names": ["t1", "t2", "t3"],
        "input_steps": 10,
        "window_stride": 3,
    }

    cfg = resolve_core_config(
        args_var_names="a,b",
        args_input_steps=12,
        args_output_steps=6,
        args_window_stride=1,
        train_cfg=train_cfg,
        model_cfg=model_cfg,
    )

    assert cfg["var_names"] == ("a", "b")
    assert cfg["input_steps"] == 12
    assert cfg["output_steps"] == 6
    assert cfg["window_stride"] == 1

    # 不提供 CLI 时，train_cfg 优先于 model_cfg
    cfg2 = resolve_core_config(
        args_var_names=None,
        args_input_steps=None,
        args_output_steps=None,
        args_window_stride=None,
        train_cfg=train_cfg,
        model_cfg=model_cfg,
    )
    assert cfg2["var_names"] == ("t1", "t2", "t3")
    assert cfg2["input_steps"] == 10
    assert cfg2["output_steps"] == 4
    assert cfg2["window_stride"] == 3


def test_weighted_and_spatial_mean_losses() -> None:
    # shape: [B=1, T=1, C=2, H=1, W=2]
    pred = torch.tensor([[[[[0.0, 0.0]], [[0.0, 0.0]]]]])
    target = torch.tensor([[[[[1.0, 1.0]], [[2.0, 2.0]]]]])
    mask = torch.ones_like(pred)
    channel_weights = torch.tensor([1.0, 2.0]).view(1, 1, 2, 1, 1)

    # ch0 mse=1, ch1 mse=4; weighted mean=(1*1 + 4*2)/(1+2)=3
    loss_point = masked_weighted_mse(pred, target, mask, channel_weights=channel_weights)
    assert torch.allclose(loss_point, torch.tensor(3.0), atol=1e-6)

    # 空间均值与逐点一致（每通道常数场）
    loss_mean = masked_spatial_mean_mse(pred, target, mask, channel_weights=channel_weights)
    assert torch.allclose(loss_mean, torch.tensor(3.0), atol=1e-6)


def test_scheduled_sampling_epsilon_decay() -> None:
    e0 = _scheduled_sampling_epsilon(
        epoch=1,
        total_epochs=10,
        enabled=True,
        start_epoch=3,
        epsilon_start=1.0,
        epsilon_min=0.4,
        decay_type="linear",
    )
    e_mid = _scheduled_sampling_epsilon(
        epoch=6,
        total_epochs=10,
        enabled=True,
        start_epoch=3,
        epsilon_start=1.0,
        epsilon_min=0.4,
        decay_type="linear",
    )
    e_end = _scheduled_sampling_epsilon(
        epoch=10,
        total_epochs=10,
        enabled=True,
        start_epoch=3,
        epsilon_start=1.0,
        epsilon_min=0.4,
        decay_type="linear",
    )
    assert e0 == 1.0
    assert e_end <= e_mid <= e0
    assert e_end >= 0.4


def test_mix_rollout_input_endpoints() -> None:
    pred = torch.zeros(2, 3, 1, 2, 2)
    target = torch.ones(2, 3, 1, 2, 2)

    out_tf = _mix_rollout_input(pred_chunk=pred, target_chunk=target, epsilon=1.0, enabled=True)
    out_ar = _mix_rollout_input(pred_chunk=pred, target_chunk=target, epsilon=0.0, enabled=True)
    out_disabled = _mix_rollout_input(pred_chunk=pred, target_chunk=target, epsilon=0.0, enabled=False)

    assert torch.allclose(out_tf, target)
    assert torch.allclose(out_ar, pred)
    assert torch.allclose(out_disabled, target)


def test_predictor_long_horizon_overlap_blend() -> None:
    predictor = ElementForecastPredictor.__new__(ElementForecastPredictor)
    predictor.device = torch.device("cpu")
    predictor.input_steps = 6
    predictor.output_steps = 6
    predictor.var_names = ("sst",)
    predictor._norm = None

    calls = {"n": 0}

    def fake_predict(x: torch.Tensor, denormalize: bool = False, return_cpu: bool = False):
        base = float(calls["n"] * 10.0)
        calls["n"] += 1
        pred = torch.full((x.shape[0], 6, 1, 1, 1), base, dtype=torch.float32, device=x.device)
        return {
            "pred": pred,
            "pred_transformer": pred,
            "var_names": predictor.var_names,
            "denormalized": denormalize,
        }

    predictor.predict = fake_predict  # type: ignore[method-assign]

    x = torch.zeros(1, 6, 1, 1, 1)
    out = predictor.predict_long_horizon(
        x=x,
        target_steps=10,
        overlap_steps=4,
        enable_overlap_blend=True,
        denormalize=False,
        return_cpu=True,
    )
    pred = out["pred"].squeeze(-1).squeeze(-1).squeeze(-1)
    # 线性融合后，边界处不应出现从 0 直接跳到 10 的硬跳变。
    assert pred.shape[1] == 10
    assert float(pred[0, 2]) < float(pred[0, 5])
    assert float(pred[0, 2]) == 0.0
    assert float(pred[0, 3]) > 0.0
