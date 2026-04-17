# 要素预测模块 (Element Forecasting)

当前模块为纯深度学习方案，主模型名称保留为 HybridElementForecastModel（兼容既有调用），默认预测路径采用 ViT 时空 Transformer + UNet + TrajGRU 多专家融合。

## 模型结构

1. 时空双轨 Transformer（ViT 风格）
   - 空间分支：对每个时刻做 patch embedding 后进行空间注意力编码。
   - 时间分支：将空间 token 重排后沿时间轴做 Transformer 编码。
   - 输出头采用“共享特征 + 按变量独立多头”：`shared_features -> sst/sss/ssu/ssv`，每个变量头为 `Conv -> GELU -> Conv`。

2. 多专家融合（新增）
   - UNet 专家：将输入时间窗展平到通道维度，通过轻量 U-Net 强化局地空间纹理。
   - TrajGRU 专家：通过可学习轨迹偏移进行重采样后门控更新，强化非刚性平流/旋转形变建模。
   - 门控融合：由最后时刻特征生成 `softmax` 门控权重，按样本动态融合 UNet / TrajGRU。
   - 与 Transformer 融合：`pred = pred_transformer + beta * pred_experts`，其中 `beta=moe_residual_beta`（默认 0.3）。
   - ssu/ssv 定向增强：对 `moe_focus_vars` 指定变量（默认 ssu、ssv）额外注入专家残差，降低矢量流场误差。

3. 输出
   - pred：最终融合预测结果。
   - pred_transformer：Transformer 分支预测结果。
   - pred_unet：UNet 专家分支预测结果。
   - pred_convlstm：TrajGRU 专家分支预测结果（为兼容历史接口保留该 key 名称）。
   - moe_gate_weights：每个样本的专家门控权重（2 维，UNet/TrajGRU）。
   - 推理器默认可做反标准化，恢复到物理量纲（可关闭）。

4. 周期性时间编码（已升级）
   - 支持多周期、多谐波时间特征，不再仅限 24 小时单周期。
   - 通过 model.yaml 配置：
     - periodic_periods：周期列表（例如 24、12、168）。
     - periodic_harmonics：每个周期的谐波阶数。
   - 编码后由线性层映射到 d_model，并叠加到 temporal token。

## 数据模式

1. 单文件数据源
   - 要素预测统一使用单个合并文件，路径通过 `data/processed/element_forecasting/path.txt` 读取。
   - 旧的 manifest 多文件/跨文件拼接逻辑已移除。

2. 滑窗采样
   - 采用 input_steps -> output_steps 的时间窗口。
   - 使用 window_stride 控制窗口步长。

3. 训练/验证/测试切分
   - 默认按赛题固定年份切分（`split_mode=competition_years`）：
     - train: 1994-2013
     - test: 2014
     - val: 2015
   - 窗口归属按预测目标段（output window）年份判定，避免跨年泄漏。
   - 仅当 `split_mode=ratio` 时，才使用 `train_ratio/val_ratio/test_ratio`。

4. 变量配置
   - 输入变量与输出变量使用同一组 var_names（如 sst/sss/ssu/ssv）。
   - 通道数由 var_names 自动推断。

## 训练策略

1. 支持 AMP、梯度累积、多进程 DataLoader。
2. 支持 spatial_downsample 以降低空间 token 数和显存压力。
3. 训练损失由“物理主损失 + Transformer/UNet/TrajGRU 辅助损失 + 空间均值约束 + 变量聚焦损失 + 梯度一致性与边缘损失”组成；其中主损失拆分为：
   - 速度场损失：`L_uv_mse + λ1*L_div + λ2*L_smooth + λ3*L_mag`
   - 标量场损失：`L_mse + λ4*L_anom + λ5*L_adv + λ6*L_grad`（`L_adv` 基于 `grid_sample` 的可微分平流 warp）
   - 总主损失：`L_total = L_sst + L_sss + 2.0 * L_uv`
4. 新增 `loss_focus_vars_weight` 对 `moe_focus_vars`（默认 ssu/ssv）做专项加权，优先压低流速分量误差。
5. 模型单段预测默认采用 24→24；为最小化 72 小时滚动误差，训练默认使用 `rollout_steps=3`（24×3=72）。`rollout_gamma` 允许大于 1，用于显式增强后续 rollout 块的惩罚权重。
6. Scheduled Sampling 默认从首个 epoch 开始，`epsilon` 由 `1.0` 按 cosine 衰减到 `0.08`，减轻训练-推理曝光偏差。
7. 训练入口：scripts/04_train_forecast.py（默认走主模型；--baseline 走旧基线）。
8. 当前推荐默认训练资源配置：`epochs=5`、`batch_size=8`、`val_batch_size=1`、`num_workers=12`、`grad_accum_steps=4`（验证默认采用更小 batch 以避免 72→72 OOM）。
9. 支持断点续训：每轮会写 `outputs/element_forecasting/checkpoints/hybrid_last.pt`，可通过 `--resume-from` 或 `--auto-resume-last` 恢复模型、优化器、AMP scaler 与训练历史。

### 验证与最优模型选择（已对齐赛题）

1. 验证阶段不再仅评估单段 24 步前向，而是使用与推理一致的滚动预测（支持 overlap blend）。
2. 默认验证目标为 72 小时（`val_target_steps=72`），与竞赛主指标对齐。
3. `val_loss` 已与训练阶段使用同口径 rollout 组合损失（主损失 + aux + 空间均值 + 梯度 + 可选边缘 + rollout_gamma 权重），可直接与 `train_loss` 比较收敛趋势。
4. best checkpoint 采用分变量鲁棒性优先的复合键：`(max_var_nrmse_percent, mean_var_nrmse_percent, max_abs_var_bias, overall_nrmse_percent)`，避免仅优化总 NRMSE 导致变量失衡。
5. 每轮会记录分变量 `val_nrmse_percent_<var>` 与 `val_bias_<var>` 到训练历史，便于定位单变量退化。

## 关键配置

1. 模型配置：configs/element_forecasting/model.yaml
   - input_steps=24, output_steps=24
   - multi_scale_enabled=false（关闭辅助 patch=8 融合，先抑制插值格纹）
   - refine_head_hidden_ratio=1.5, refine_head_num_layers=4
   - d_model, nhead, num_layers, block_size, dropout, spatial_downsample
   - periodic_periods, periodic_harmonics
   - moe_enabled, moe_unet_base_channels
   - moe_trajgru_hidden_channels, moe_trajgru_kernel_size, moe_trajgru_num_links, moe_trajgru_flow_hidden_channels, moe_trajgru_flow_clip
   - moe_residual_beta, moe_focus_vars, moe_focus_boost

2. 训练配置：configs/element_forecasting/train.yaml
   - data_file, norm_stats_path
   - window_stride, split_mode, split_years
   - train_ratio, val_ratio, test_ratio（仅 split_mode=ratio 时生效）
   - epochs, batch_size, val_batch_size, lr, num_workers, val_num_workers, device, amp, grad_accum_steps
   - resume_from, auto_resume_last
   - rollout_steps, rollout_gamma, val_target_steps
   - scheduled_sampling_start_epoch, scheduled_sampling_epsilon_start, scheduled_sampling_epsilon_min, scheduled_sampling_decay_type
   - overlap_blend_enabled, overlap_steps
   - loss_aux_unet_weight, loss_aux_convlstm_weight, loss_focus_vars_weight
   - moe_focus_vars（用于构建专项通道损失）
   - loss_uv_div_lambda, loss_uv_smooth_lambda, loss_uv_mag_lambda, loss_uv_div_target_zero
   - loss_scalar_anom_lambda, loss_scalar_adv_lambda, loss_scalar_grad_lambda
   - loss_gradient_consistency_weight, loss_edge_weight, edge_loss_type

## 兼容性说明

1. 旧 checkpoint 若未包含 MoE 参数，推理器会按默认值回退（`moe_enabled=false`），可保持兼容加载。
2. 新训练产出的 checkpoint 会写入 `moe_focus_channel_indices`，确保推理端与训练端使用相同的变量聚焦通道。

## 评估指标 (Metrics)

1. 大赛主指标：NRMSE (%)
   - 当前实现采用掩膜区域上的 `NRMSE = RMSE / (target_max - target_min) * 100`。
   - 训练验证日志与竞赛检查脚本均输出 `nrmse_percent`，并用于阈值对比。
2. 偏置与分段鲁棒性指标（新增）
   - 竞赛检查脚本新增 `bias = mean(pred-target)`，并输出 `first_24h`、`last_24h` 分段的 RMSE/MAE/NRMSE/Bias，避免总分掩盖后段漂移。
3. 分变量汇总（新增）
   - 竞赛检查脚本输出每个变量的 `bias/mae/rmse/nrmse_percent` 以及 `rmse_first_24h`、`rmse_last_24h`、`rmse_growth_last_vs_first`。
4. 样本可视化策略（新增）
   - 竞赛检查脚本支持 `hard/random/mixed` 多样本选图，不再只固定展示首样本。
5. 边缘区域指标（edge_rmse）
   - 以目标梯度分位数构造边缘掩膜；当有效点过多时会自动做等步长采样后再计算分位数，避免 `torch.quantile` 在超大输入上的运行时错误。

## 续训命令示例

1. 从指定 checkpoint 继续：`python scripts/04_train_forecast.py --resume-from outputs/element_forecasting/checkpoints/hybrid_last.pt`
2. 自动恢复最近一次 last checkpoint：`python scripts/04_train_forecast.py --auto-resume-last`

## 内存稳定性建议

1. 训练与验证 DataLoader 已分离并发配置：`num_workers` 仅用于训练，`val_num_workers` 仅用于验证。
2. 默认验证侧使用 `val_num_workers=0`，并关闭 `persistent_workers/prefetch`，减少中途被系统 OOM-kill 的风险。
