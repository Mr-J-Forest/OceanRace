# 要素预报主模型配置

本目录为**非基线**主模型的唯一配置来源（与 `configs/baseline/element_forecasting/` 分离）。

| 文件 | 说明 |
|------|------|
| `model.yaml` | 结构超参：`d_model`、`nhead`、`periodic_periods` / `periodic_harmonics`、`spatial_downsample`、`moe_*`（UNet+TrajGRU 多专家，残差融合 `moe_residual_beta`）等 |
| `train.yaml` | 训练过程：`epochs`、`batch_size`、`rollout_*`、`scheduled_sampling_*`、`split_mode` / `split_years`、`loss_aux_unet_weight` / `loss_aux_convlstm_weight` / `loss_focus_vars_weight`，以及速度/标量物理损失权重（`loss_uv_*`、`loss_scalar_*`）等 |

训练入口：`scripts/04_train_forecast.py`（默认加载上述两个文件，可用 CLI 覆盖部分项）。

说明：仓库**不维护** `*.yaml.example`；请直接编辑并提交本目录下的 yaml。敏感项使用 `configs/*.secret` 或环境变量，勿写入可提交配置。
