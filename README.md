# 海洋环境智能分析系统

面向海洋环境现象识别与多要素智能分析，基于深度学习处理约 70G 海洋 NetCDF 数据，提供中尺度涡旋识别、水文要素预测与风-浪异常检测等能力。赛题背景、指标与提交材料见 [`docs/赛题A09_面向海洋环境智能分析系统.md`](docs/赛题A09_面向海洋环境智能分析系统.md)。

---

## 目录结构

```
OceanRace/
├── README.md
├── AGENTS.md                  # AI/协作约定与可 @ 引用的编程 Prompt（logger、可视化、outputs、configs）
├── requirements.txt
├── configs/
│   ├── data_config.yaml       # 数据路径、划分、批处理、artifacts 等
│   ├── README.md              # 各子目录配置说明
│   ├── eddy_detection/        # 涡旋主模型 model.yaml / train.yaml
│   ├── element_forecasting/   # 要素预报主模型（非基线）
│   ├── anomaly_detection/     # 异常检测主模型
│   └── baseline/
│       ├── element_forecasting/  # 要素基线（ConvLSTM）
│       └── anomaly_detection/    # 异常检测基线（轻量双分支 AE）
├── data/                      # 通常不提交（.gitignore）
│   ├── __init__.py
│   ├── raw/                   # 原始 NetCDF，见 data/raw/README.md
│   │   ├── eddy_detection/
│   │   ├── element_forecasting/
│   │   └── anomaly_detection/
│   ├── processed/             # 清洗后数据（*_clean.nc、oper/wave_clean 等）
│   │   ├── eddy_detection/
│   │   ├── element_forecasting/
│   │   ├── anomaly_detection/
│   │   ├── normalization/     # 训练集 μ/σ 等 JSON
│   │   └── splits/            # train/val/test 划分清单 JSON
│   └── interim/               # 中间数据（可选）
├── src/
│   ├── data_preprocessing/    # 模块1：数据预处理（cleaner、splitter、io…）
│   ├── eddy_detection/        # 模块2：中尺度涡旋识别（含 dataset.py）
│   ├── element_forecasting/   # 模块3：水文要素预测（含 dataset.py）
│   ├── anomaly_detection/     # 模块4：风-浪异常识别（含 README、dataset.py）
│   ├── baseline/              # 三任务基线（要素/异常已实现）
│   │   ├── eddy_detection/
│   │   ├── element_forecasting/   # convlstm、model、sequence_dataset、train
│   │   └── anomaly_detection/
│   ├── utils/                 # 公共工具（logger、visualization_defaults、dataset_utils、README）
│   └── pipeline.py            # 主流程管道（占位）
├── models/                    # 最佳模型（从 outputs 挑选后放入，可提交）
├── outputs/                   # 中间训练结果（checkpoint、日志等，gitignore）
│   └── final_results/         # 最终最佳输出结果（指标、图表等，可提交）
│       ├── eddy_detection/    # 涡旋识别
│       ├── element_forecasting/   # 要素预报
│       └── anomaly_detection/     # 异常检测
├── scripts/                   # 命令行入口（见 scripts/README.md）
│   ├── README.md
│   ├── 01_data_inspect.py     # 数据探查
│   ├── 02_preprocess.py       # 预处理
│   ├── smoke_element_forecast.py  # 要素基线极少样本冒烟
│   ├── 03_train_eddy.py       # 涡旋训练（占位，待接 src/eddy_detection）
│   ├── 04_train_forecast.py   # 要素预报训练
│   ├── 05_train_anomaly.py    # 异常检测训练（支持 --baseline）
│   ├── 06_run_pipeline.py     # 端到端流水线（占位）
│   └── 07_generate_report.py  # 评估报告（占位）
├── tests/                     # 单元测试（preprocessing、models、pipeline）
└── docs/
    └── 赛题A09_面向海洋环境智能分析系统.md
```

### 脚本与文档

| 路径 | 说明 |
|------|------|
| `AGENTS.md` | **AI 编程 Prompt**（@ 引用）：logger、`visualization_defaults`、`outputs` 分层、`configs` 单一来源 |
| `scripts/README.md` | 脚本说明、日志约定、基线实验衔接、典型流程 |
| `scripts/01_data_inspect.py` | 只读：抽样 `data/raw`，缺测率与最值（可 `--out JSON`） |
| `scripts/02_preprocess.py` | 清洗 → `data/processed/`，可选整合大文件（`merge`）、划分与训练集标准化（`--steps`） |
| `scripts/smoke_element_forecast.py` | 极少样本合成数据，跑 1 epoch 验证要素基线训练链路 |
| `scripts/test_element/check_element_forecast_competition.py` | 要素预测比赛检查（单文件模式），支持滚动预测与重叠融合评估 |
| `scripts/03_train_eddy.py` 等 | 训练/流水线/报告脚本（`03`–`07`）；要素预报用 `python scripts/04_train_forecast.py` |
| `src/baseline/` | 基线实验代码（`PYTHONPATH=src`，如 `python -m baseline.element_forecasting.train`） |
| `src/baseline/element_forecasting/README.md` | 要素 ConvLSTM 基线模块与运行方式 |
| `src/baseline/anomaly_detection/README.md` | 异常检测轻量 AE 基线模块与运行方式 |
| `src/utils/README.md` | 工具说明（logger、dataset_utils、可视化命令行） |
| `data/raw/README.md` | 各任务 NetCDF 维度与坐标说明 |
| `src/anomaly_detection/README.md` | 异常检测方法选型与说明 |
| `docs/赛题A09_面向海洋环境智能分析系统.md` | 赛题 A09 原文整理（指标与提交材料） |

### 产出目录

| 目录 | 说明 |
|------|------|
| `outputs/` | 存放**中间训练结果**（checkpoint、日志等），已 gitignore，不提交 |
| `outputs/eddy_detection/merged_chunks/` | 涡旋清洗小文件整合后的大文件（可分块） |
| `outputs/element_forecasting/merged_chunks/` | 要素清洗小文件整合后的大文件（可分块） |
| `outputs/anomaly_detection/merged_chunks/` | 异常检测 `oper/wave` 清洗文件整合后的大文件（可分块） |
| `outputs/final_results/` | 存放**最终最佳输出结果**（指标、图表等），可提交。下设三模块子目录：`eddy_detection/`、`element_forecasting/`、`anomaly_detection/` |
| `models/` | 存放**最佳模型**。训练完成后从 `outputs/` 挑选最优 checkpoint，复制到 `models/` 后可提交 |

---

## 模块说明

### 模块1：数据预处理 (`src/data_preprocessing/`)

**职责：** 处理原始 NetCDF，为训练提供干净、标准化的输入，并可将清洗后的小文件整合为大文件。

**流程：** `scripts/01_data_inspect.py`（只读统计）→ `cleaner.py`（哨兵/NaN→掩膜）→ `scripts/02_preprocess.py` 落盘 `data/processed/`（可选 `merger` 整合、`splitter` 划分与训练集 \(\mu,\sigma\)）→ 各任务模块 `dataset.py` 中 `Dataset` 供训练。

**文件：** `cleaner.py` · `io.py` · `merger.py` · `splitter.py` · `validator.py`

| 文件 | 要点 |
|------|------|
| `cleaner.py` | 单文件清洗；哨兵与 NaN 掩膜；`*_valid` |
| `io.py` | 打开 NetCDF（含 Windows 中文路径） |
| `merger.py` | 将某任务多个清洗文件按时间维整合为一个或多个大文件（支持分块） |
| `splitter.py` | train/val/test 划分；仅用训练集估计标准化参数并写 JSON |
| `validator.py` | 校验 processed 变量/`*_valid`/时间维单调/有效点比例；`splits/*.json` 路径存在性（`run_validation_for_task` 等） |

**产出：** `data/processed/`、`outputs/*/merged_chunks/`、质量验证报告、划分后的数据集。

---

### 模块2：中尺度涡旋识别 (`src/eddy_detection/`)

**职责：** 识别涡旋结构，输出边界、中心与旋转方向。

**文件：** `dataset.py` · `model.py` · `trainer.py` · `predictor.py` · `evaluator.py`

| 文件 | 要点 |
|------|------|
| `dataset.py` | 从 `data/processed/eddy_detection` 读 `*_clean.nc`；可选 split 与标准化 |
| `model.py` | U-Net 等；气旋/反气旋；边界掩码与中心点 |
| `trainer.py` | 学习率与优化器；数据增强（旋转、翻转）；损失与准确率监控 |
| `predictor.py` | 单张/批量推理；NMS、边界平滑；GeoJSON 输出 |
| `evaluator.py` | 准确率、召回、F1；IoU；混淆矩阵 |

**指标：** 目标准确率 ≥75%；输出边界、中心、旋转方向、强度。

| 类别 | 变量名或内容 | 说明 |
|------|----------------|------|
| **输入（数据）** | `adt`、`ugos`、`vgos` | 动力场，形状 `(time, latitude, longitude)` |
| **输入（坐标）** | `time` | `days since 1950-01-01` |
| **输入（坐标）** | `latitude`、`longitude` | 度 |
| **输出（推理）** | `eddy_mask` 或等价分割结果 | 与空间格网对齐的二值/多类掩膜 |
| **输出（推理）** | 中心坐标、涡旋类型 | 如 `(lon, lat)`、气旋/反气旋标签 |
| **输出（评估）** | IoU、Precision、Recall、F1 等 | 由 `evaluator` 计算 |

---

### 模块3：水文要素预测 (`src/element_forecasting/`)

**职责：** 基于历史数据预测未来 72 小时温度、盐度、流速等。

**文件：** `dataset.py` · `model.py` · `trainer.py` · `predictor.py` · `evaluator.py`

| 文件 | 要点 |
|------|------|
| `dataset.py` | 单文件模式：读取 `all_clean_merged.nc`，按滑窗切样本并按时间比例切分 train/val/test |
| `model.py` | 时空双轨 Transformer + block residual；多周期多谐波时间编码（`periodic_periods`/`periodic_harmonics`） |
| `trainer.py` | 加权损失 + 空间均值约束；rollout 训练；scheduled sampling；梯度累积与 AMP |
| `predictor.py` | 支持长时滚动预测与 overlap 线性融合（缓解 24h/48h 拼接台阶） |
| `evaluator.py` | 掩膜 MSE/RMSE/MAE/NSE；按变量加权 MSE；空间均值约束损失 |

**最新训练策略（已实现）**

1. 变量加权与均值约束
    - 训练损失中引入按变量权重加权的点位误差，提升 `sss`、`ssv` 的学习权重。
    - 新增空间均值约束项，约束 `pred` 与 `target` 在每步每变量上的空间均值一致，抑制整体抬高偏置。

2. Rollout 训练（多步对齐）
    - 训练集标签长度扩展为 `model_output_steps * rollout_steps`。
    - 每个 batch 内做多段前向并按 `rollout_gamma` 聚合段损失，强化多步稳定性。

3. Scheduled Sampling
    - 训练中段间输入按概率在“真实上一段标签 / 上一段预测”间混合。
    - `epsilon` 按 epoch 衰减（支持 linear / cosine），缓解 teacher forcing 与推理不一致。

4. 推理重叠融合
    - 长时预测支持 `overlap_steps` 与线性融合，缓解分段拼接处的硬跳变。
    - 比赛检查脚本已接入该能力，并可通过参数开关。

**关键配置（`configs/element_forecasting/*.yaml`）**

- 模型：`d_model`、`nhead`、`num_layers`、`spatial_downsample`、`periodic_periods`、`periodic_harmonics`
- 训练：`rollout_steps`、`rollout_gamma`、`scheduled_sampling_*`、`overlap_*`、`var_loss_weights`、`loss_spatial_mean_weight`

**指标：** 目标 MSE ≤15%；预测时长 72 小时；要素含温度、盐度、流速。

| 类别 | 变量名或内容 | 说明 |
|------|----------------|------|
| **输入（数据）** | `sst`、`sss`、`ssu`、`ssv` | 海表温盐与流分量，形状 `(time, lat, lon)` |
| **输入（坐标）** | `time` | `hours since 2000-01-01`（单日文件内为多时次） |
| **输入（坐标）** | `lat`、`lon` | 度 |
| **输出（推理）** | `sst`、`sss`、`ssu`、`ssv`（预报） | 未来 ≥72 h 的预报场；实现可增加 `forecast_time` 维或独立写出 `*_fcst.nc` |
| **输出（评估）** | MSE、RMSE、MAE、纳什系数等 | 与真值或验证集对比 |

---

### 模块4：风-浪异常检测 (`src/anomaly_detection/`)

**职责：** 检测风浪异常，关联台风等事件，输出预警信息。

**算法与异常检测方法选型**（分类梳理、AE/iForest 建议等）：[`src/anomaly_detection/README.md`](src/anomaly_detection/README.md)。

**风–浪网格不一致**：原始风场与浪场经纬格网不同（见 `data/raw/README.md`），预处理仍输出分文件的 `oper_clean.nc` / `wave_clean.nc`，**不在预处理阶段强行插值对齐**。建模可采用 **双分支网络**（风、浪各自卷积编码，在全局特征上融合），避免像素级对齐；亦可采用分开建模或先插值再联合，详见 [`src/anomaly_detection/README.md`](src/anomaly_detection/README.md)。

**训练与评估入口脚本：** `scripts/05_train_anomaly.py`（已实现）

**标签与事件模板：** `scripts/05b_prepare_anomaly_eval_templates.py`（已实现）

**文件：** `dataset.py` · `model.py` · `trainer.py` · `detector.py` · `evaluator.py`

| 文件 | 要点 |
|------|------|
| `dataset.py` | 每年 `oper_clean` + `wave_clean`；可选 split 与标准化 |
| `model.py` | 自编码器；可选 One-Class SVM；重构误差 |
| `trainer.py` | 仅用正常样本；学习正常风-浪模式；确定阈值 |
| `detector.py` | 实时检测；台风关联；预警等级 |
| `evaluator.py` | 准确率、精确率、召回；ROC/AUC；误报分析 |

**指标：** 目标准确率 ≥80%；实时检测；输出异常等级、置信度与关联信息。

**重要说明（是否达标的前提）**

- 若仅运行无监督检测（没有标签），可得到 `anomaly_ratio`、阈值与告警等级，但**不能**证明“准确率 ≥80%”。
- 要输出 `Accuracy/Precision/Recall/F1/AUC`，必须提供 `labels.json`（0/1 标签，长度与 split 样本数一致）。
- 要输出台风关联结果，必须提供 `events.json`（含事件 `start/end` 时间窗）。
- 推荐流程：
    1. 运行 `python scripts/05b_prepare_anomaly_eval_templates.py --force` 生成模板；
    2. 填写 `outputs/anomaly_detection/labels.json` 与 `outputs/anomaly_detection/events.json`；
    3. 运行 `python scripts/05_train_anomaly.py ... --labels-json ... --events-json ...` 输出监督指标与事件关联。

| 类别 | 变量名或内容 | 说明 |
|------|----------------|------|
| **输入（风）** | `u10`、`v10` | `data_stream-oper_stepType-instant.nc`，`(valid_time, latitude, longitude)` |
| **输入（浪）** | `swh`、`mwp`、`mwd` | `data_stream-wave_stepType-instant.nc`，浪格网与风可能不一致，需插值或分支处理 |
| **输入（坐标）** | `valid_time` | `seconds since 1970-01-01` |
| **输入（坐标）** | `latitude`、`longitude` | 风、浪各自一套维度 |
| **输入（可选）** | 台风路径/时间索引 | 用于关联分析 |
| **输出（推理）** | `anomaly_score` 或重构误差序列 | 逐时次或逐格点 |
| **输出（推理）** | `alert_level`、是否异常标志 | 离散等级 |
| **输出（关联）** | `typhoon_flag` 或最近台风信息 | 可选 |
| **输出（评估）** | Accuracy、Precision、Recall、F1、AUC 等 | 需提供 0/1 标签 |

---

### 模块5：工具函数 (`src/utils/`)

**职责：** 日志、可视化、`Dataset` 共用辅助（划分清单 / norm JSON / 张量标准化）等；**评估指标**在各自模型模块（如 `evaluator`）中实现。日志统一使用 `utils.logger`（约定见 `AGENTS.md`、`.cursor/rules/logging.mdc`）。用法见 [`src/utils/README.md`](src/utils/README.md)。

**文件：** `logger.py` · `visualization_defaults.py` · `dataset_utils.py`

| 类别 | 内容 | 说明 |
|------|------|------|
| **输入** | 张量、路径、`xarray.Dataset`、配置字典 | 由各业务模块传入 |
| **输出** | 日志文本、图像文件（PNG 等）、标准化后的张量 | 见各函数签名 |

---

### 模块6：主流程 (`src/pipeline.py`)

**职责：** 统一入口：数据预处理、各子模块训练/推理调度、评估与报告生成；**各算法模块彼此独立**，调用顺序与组合方式由实现决定。

**示例（各模块分步调用）：**

```python
data = load_and_preprocess(input_path)
forecast = forecast_model.predict(data)           # 要素预测
eddies = eddy_model.detect(eddy_input)            # 涡旋识别（输入见 data/raw）
anomalies = anomaly_model.evaluate(wind_wave_input)  # 风-浪异常
report = export_results(forecast, eddies, anomalies)  # 可选汇总
```

| 类别 | 内容 | 说明 |
|------|------|------|
| **输入** | 各子模块所需路径、配置、可选 CLI 参数 | 指向 `data/raw` 或 `data/processed`、`models/` |
| **输出** | 汇总报告（JSON/HTML/Markdown）、运行日志 | 可选聚合各模块导出结果；具体字段由 `export_results` 等实现定义 |

各任务 **NetCDF 维度与坐标** 的逐项说明见 [`data/raw/README.md`](data/raw/README.md)。**产物目录：** `outputs/`（预报曲线、涡旋图、异常报告等），以实际代码与 `configs/` 为准。

---

## 开发时间线

| 阶段 | 模块 | 时间 | 产出 |
|------|------|------|------|
| 第1周 | 数据预处理 | 5 天 | 清洗后的数据集 |
| 第2周 | 涡旋识别 | 4 天 | 涡旋检测模型（准确率≥75%） |
| 第3周 | 水文预测 | 4 天 | 时序预测模型（MSE≤15%） |
| 第4周 | 异常检测 | 4 天 | 异常检测模型（准确率≥80%） |
| 第5周 | 系统集成 | 3 天 | 端到端管道 |
| 第6周 | 优化调试 | 2 天 | 性能提升与报告 |

---

## 验收标准

**1. 功能性**

- 涡旋识别准确率 ≥75%
- 水文预测 MSE ≤15%
- 异常检测准确率 ≥80%

**2. 工程性**

- 支持约 70G 级数据处理
- 模块化、接口清晰
- 文档与测试完整

**3. 实用性**

- 72 小时预测能力
- 实时异常预警
- 可视化分析界面

---

通过以上模块协同，形成可扩展的海洋环境智能分析能力，服务于海洋监测、灾害预警与资源开发等场景。
