# 要素预测模块 (Element Forecasting)

当前模块为纯深度学习方案，主模型名称保留为 HybridElementForecastModel（兼容既有调用），实际预测路径仅使用 Transformer 分支。

## 模型结构

1. 时空双轨 Transformer
   - 空间分支：对每个时刻做 patch embedding 后进行空间注意力编码。
   - 时间分支：将空间 token 重排后沿时间轴做 Transformer 编码。
   - 时间分支采用 BlockResidualTransformerEncoder，在每层 Attention 和 MLP 前注入 block-level residual context。

2. 输出
   - pred：最终预测结果。
   - pred_transformer：Transformer 分支预测结果（当前与 pred 一致）。
   - 推理器默认可做反标准化，恢复到物理量纲（可关闭）。

3. 周期性时间编码（已升级）
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
3. 训练损失由主损失、Transformer 辅助损失、空间均值约束、梯度一致性与边缘损失组成；当前默认 `loss_gradient_consistency_weight=0.12`、`loss_edge_weight=0.03(sobel)`。
4. 训练采用 rollout 多段监督；默认 `rollout_steps=3`、`rollout_gamma=1.0`，即 24h x 3 段对齐 72h 目标。
5. Scheduled Sampling 默认从首个 epoch 开始，`epsilon` 由 `1.0` 按 cosine 衰减到 `0.08`，减轻训练-推理曝光偏差。
6. 训练入口：scripts/04_train_forecast.py（默认走主模型；--baseline 走旧基线）。
7. 当前推荐默认训练资源配置：`epochs=30`、`batch_size=8`、`num_workers=12`、`grad_accum_steps=4`（有效 batch 约为 32）。

### 验证与最优模型选择（已对齐赛题）

1. 验证阶段不再仅评估单段 24 步前向，而是使用与推理一致的滚动预测（支持 overlap blend）。
2. 默认验证目标为 72 小时（`val_target_steps=72`），与竞赛主指标对齐。
3. `val_loss` 已与训练阶段使用同口径 rollout 组合损失（主损失 + aux + 空间均值 + 梯度 + 可选边缘 + rollout_gamma 权重），可直接与 `train_loss` 比较收敛趋势。
4. best checkpoint 选择依据为验证集 `nrmse_percent`（越低越好）。

## 关键配置

1. 模型配置：configs/element_forecasting/model.yaml
   - multi_scale_enabled=false（关闭辅助 patch=8 融合，先抑制插值格纹）
   - refine_head_hidden_ratio=1.5, refine_head_num_layers=4
   - d_model, nhead, num_layers, block_size, dropout, spatial_downsample
   - periodic_periods, periodic_harmonics

2. 训练配置：configs/element_forecasting/train.yaml
   - data_file, norm_stats_path
   - window_stride, split_mode, split_years
   - train_ratio, val_ratio, test_ratio（仅 split_mode=ratio 时生效）
   - epochs, batch_size, lr, num_workers, device, amp, grad_accum_steps
   - rollout_steps, rollout_gamma, val_target_steps
   - scheduled_sampling_start_epoch, scheduled_sampling_epsilon_start, scheduled_sampling_epsilon_min, scheduled_sampling_decay_type
   - overlap_blend_enabled, overlap_steps
   - loss_gradient_consistency_weight=0.12, loss_edge_weight=0.03, edge_loss_type=sobel

## 评估指标 (Metrics)

1. 大赛主指标：NRMSE (%)
   - 当前实现采用掩膜区域上的 `NRMSE = RMSE / (target_max - target_min) * 100`。
   - 训练验证日志与竞赛检查脚本均输出 `nrmse_percent`，并作为阈值对比与 best checkpoint 选择依据。

