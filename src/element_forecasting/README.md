# 要素预测模型 (Element Forecasting Model)

当前要素预测模块已调整为纯深度学习架构，不再包含 ARIMA 统计分支。

## 模型结构

主模型封装为 `HybridElementForecastModel`（名称保留以兼容既有调用），内部仅使用 Transformer 分支输出预测结果。

1. Transformer 序列建模主干
   - 将每个空间网格点视为时间序列 token 建模。
   - 在每层中使用 Block-level Residual Attention，在 Attention 前与 MLP 前注入跨 block 上下文。

2. 输出
   - `pred`：最终预测结果。
   - `pred_transformer`：Transformer 分支预测结果（与 `pred` 一致）。
   - 推理器默认对预测结果执行反标准化，恢复到原始物理量量纲（可通过参数关闭）。

3. 周期性时间编码 (Time-of-Day Embedding)
   - 针对预测长波段中常见的随时间周期波动的现象（如昼夜海温变化、潮汐导致的风速波动），模型输入层引入了绝对时间维度的正弦与余弦编码（基于给定时间窗口的相对/绝对步数，假设 `24h` 基础物理循环），并在自注意力机制各层累加该编码以增强周期性拟合。

## 数据与输入输出

1. 按时间窗口采样
   - 使用 `input_steps -> output_steps` 的滑窗方式。
   - 支持单文件训练（如 `all_clean_merged.nc`），也支持多文件并可跨文件拼接时间轴。

2. 变量可配置
   - 输入变量与输出变量使用同一组配置（如 `sst/sss/ssu/ssv`），通道数自动由变量数推断。

3. 小文件时序加载策略
   - 面向大量按时间顺序组织的小 NetCDF 文件，数据集采用跨文件时间窗流式读取。
   - 每个样本按窗口范围定位涉及的 1~N 个文件，单次打开文件后批量读取该窗口所需全部变量，再在内存中拼接为输入/标签。
   - 使用 `open_file_lru_size` 控制文件句柄缓存，避免每个样本反复 open/close。

## 训练策略

1. 支持 AMP、梯度累积与多进程 DataLoader。
2. 支持空间下采样建模后上采样恢复（`spatial_downsample`），用于降低显存压力。
3. 频域损失 (FFT Loss)：为了防止预测曲线在长周期预测（如 48h、72h）阶段因趋向于平均值而变平缓，采用时空域损失叠加频域幅值 `l1_loss` 的策略，强制迫使生成曲线还原具有波动特点的高频成分。
4. 训练入口位于 `trainer.py`，配置位于 `configs/element_forecasting/model.yaml` 与 `configs/element_forecasting/train.yaml`。
