# 风–浪异常识别模块 (`src/anomaly_detection/`)

本模块对风场（`u10`/`v10`）与海浪（`swh`/`mwp`/`mwd`）进行异常检测，并可与台风等事件关联。原始数据说明见 [`data/raw/README.md`](../../data/raw/README.md)。

---

## 风–浪网格不一致：双分支建模（推荐思路）

风、浪原始 NetCDF 的 **经纬度格网不同**（例如风约 241×321、浪约 121×161，见 `data/raw/README.md`）。当前 **数据预处理**（`cleaner` + `02_preprocess.py`）对每年目录分别写出 **`oper_clean.nc`** 与 **`wave_clean.nc`**，**保留各自网格**，**不在预处理阶段把浪插值到风网格**（或反之）。

若要在**联合建模**时避免「先插值再叠通道」的假设与平滑，可采用 **双分支网络**：

| 环节 | 说明 |
|------|------|
| **风分支** | 输入与风场一致的空间尺寸（如 241×321），卷积编码风变量（`u10`/`v10`/`wind_speed` 等）。 |
| **浪分支** | 输入与浪场一致的空间尺寸（如 121×161），卷积编码浪变量（`swh`/`mwp`/`mwd`）。 |
| **融合** | 两路各自得到 **向量**（或小特征图经全局池化后的向量），在特征上 **concat + MLP** 或 **注意力** 等，再输出异常分数或异常等级。 |

**与「两个完全独立模型」的区别**：双分支仍在 **同一套** 融合头里学习风–浪如何共同解释异常；**分开两个模型**则是风单独出分、浪单独出分，再在分数层做融合，实现更简单。

**与「像素级对齐」的区别**：双分支 **不要求** 风、浪在同一格点一一对应；若希望输出 **逐格点异常图**，需在模型中保留空间分辨率（如各分支自编码器输出与输入同尺寸的重建误差图），或在上采样到同一参考网格后再融合，**与「仅向量融合」是否输出逐点异常** 的设计目标不同，需单独约定。

**与当前仓库的关系**：`dataset.py` 已按年提供 `oper` / `wave` 两组张量，**在现有预处理基础上即可实现双分支**，无需为双分支单独改 cleaner；仅需在模型与 `DataLoader` 中按两路张量分别 `forward`。

---

## 常见异常检测方法分类（原理归纳）

下列方法可按原理分为若干类（便于选型与写技术方案对比），**不必全部实现**。

### 一、基于分布与统计

| 方法 | 核心思想 |
|------|----------|
| 3-Sigma / 拉依达 | 正态假设下，落在均值±3σ 外视为异常 |
| Z-Score | 标准化后按阈值（如 \|Z\|>3）判定 |
| 箱线图（IQR） | 基于四分位距，Q1−1.5×IQR 以下或 Q3+1.5×IQR 以上为异常 |
| 格拉布斯检验（Grubbs） | 正态单变量中检验单个极端值 |

### 二、基于距离与邻近度

| 方法 | 核心思想 |
|------|----------|
| KNN | 与 K 个近邻的平均距离大则异常分高 |
| LOF | 局部密度低于邻居则为离群 |
| COF | 链式距离，改进低密度区域敏感度 |
| SOS | 基于关联概率矩阵，关联度低的点更易判为异常 |

### 三、基于聚类

| 方法 | 核心思想 |
|------|----------|
| DBSCAN | 噪声点（不归任何簇）视为异常 |

### 四、基于树与集成

| 方法 | 核心思想 |
|------|----------|
| 孤立森林（Isolation Forest） | 异常点更易被随机划分「孤立」，路径短则异常分高 |

### 五、基于降维与重构

| 方法 | 核心思想 |
|------|----------|
| PCA | 投影偏差或重构误差大则为异常 |
| 自编码器（Autoencoder） | 仅用正常数据训练，异常样本重构误差大 |

### 六、基于分类边界

| 方法 | 核心思想 |
|------|----------|
| One-Class SVM | 在特征空间包围正常点，界外为异常 |

### 七、基于时间序列

| 方法 | 核心思想 |
|------|----------|
| STL + ESD 等 | 分解趋势/季节后，对残差做广义 ESD 等检验 |

---

## 与本项目数据最适配的选型建议

风–浪数据为 **多变量、时空格点、强非线性耦合**（见 `data/raw/anomaly_detection/`），浪场存在 **较多 NaN**，赛题要求 **深度学习** 与 **识别准确率**、台风关联等。

### 推荐作为主模型：**自编码器（Autoencoder）**

- 与多通道输入（风 + 浪）及「学习正常海况流形」一致，**重构误差** 可直接作为异常分数。
- 风、浪 **格网一致** 时可将多变量叠成多通道、**单编码器**；**格网不一致** 时更宜采用 **双分支**（各自 AE 或各自编码 + 融合解码），见 **「风–浪网格不一致：双分支建模」** 一节，避免预处理阶段强行插值。
- 需在损失与掩膜上 **显式处理浪场 NaN**（如仅对有效格点反传或先插值/分区域建模）。

### 推荐作为基线对照：**孤立森林（Isolation Forest）**

- 不宜对全格点直接展平；应对 **区域统计特征、分位数、能量、风向浪向** 等做特征提取后再送入 iForest，**训练快、易对比**，适合写「对比实验」。

### 可选用（按需）

| 方法 | 适用场景 |
|------|----------|
| Z-score / 3σ / IQR | **单站或区域平均** 的风速、有效波高等 **辅助阈值与可视化**，不宜作为高维格点唯一方案 |
| One-Class SVM | 在 **展平特征向量** 或 **patch** 上作第二基线；注意规模与核函数成本 |
| LOF / KNN | 特征 **已降维**、样本量适中时可试 |
| STL + ESD | 若先聚合为 **单变量时间序列**（某点/某区）并强调季节分解时使用 |

### 小结

- **主技术路线**：自编码器（与 `model.py` 中 AE 设计一致）；网格不一致时优先考虑 **双分支**（见「风–浪网格不一致：双分支建模」）。
- **强基线**：Isolation Forest（在手工或 PCA 特征上）。
- 简单统计方法适合 **解释与监控**，与深度模型 **互补**，不替代主模型。

---

## 与本仓库文件的对应关系（规划）

| 文件 | 职责 |
|------|------|
| `model.py` | 自编码器等异常模型定义 |
| `trainer.py` | 正常样本训练、阈值标定 |
| `detector.py` | 在线检测、台风关联、预警等级 |
| `evaluator.py` | 准确率、AUC、误报率等 |

具体超参数建议放在 `configs/anomaly_detection/model.yaml`（与 `train.yaml`）中维护。

---

## 训练与达标评估（实操）

> 仅有重构误差与阈值统计，**不能**直接证明赛题“异常识别准确率 ≥80%”。
> 必须提供标签（`labels_json`），并输出 `labeled_metrics`（accuracy/F1/AUC）。

### 1) 训练（默认阈值策略）

```bash
python scripts/05_train_anomaly.py \
	--device cuda \
	--epochs 20 \
	--batch-size 32 \
	--num-workers 8 \
	--open-file-lru-size 32 \
	--report-splits val,test
```

### 2) 先生成标签/事件模板（一次性）

```bash
python scripts/05b_prepare_anomaly_eval_templates.py --force
```

将下面两个模板复制并填充真实标注后再用于评估：

- `outputs/anomaly_detection/templates/labels.template.json`
- `outputs/anomaly_detection/templates/events.template.json`

`labels.template.json` 约定：

- 键是 split（如 `val`、`test`）
- 值是与样本数等长的数组，`0=normal`，`1=anomaly`

`events.template.json` 约定：

- 列表元素为 `{"name": ..., "start": int, "end": int}`
- `start/end` 为时间戳整数，供台风事件关联评估使用

### 3) 带标签评估 + 阈值调优（推荐）

```bash
python scripts/05_train_anomaly.py \
	--device cuda \
	--epochs 20 \
	--batch-size 32 \
	--num-workers 8 \
	--open-file-lru-size 32 \
	--report-splits val,test \
	--labels-json outputs/anomaly_detection/labels.json \
	--events-json outputs/anomaly_detection/events.json \
	--threshold-policy val-f1 \
	--threshold-quantiles 0.85,0.90,0.93,0.95,0.97,0.99
```

### 4) 新增阈值策略参数

- `--threshold-policy model`：用训练得到的 best threshold（默认）
- `--threshold-policy val-f1`：用 val 标签调阈值，最大化 F1
- `--threshold-policy val-accuracy`：用 val 标签调阈值，最大化 accuracy
- `--threshold-policy fixed --fixed-threshold <x>`：固定阈值复现实验

### 5) 结果文件解读

- `outputs/anomaly_detection/summary.json`
	- 包含 `best_val_loss`、`best_threshold` 与 `threshold_info`
- `outputs/anomaly_detection/split_reports.json`
	- 无标签时：仅 `error_summary` + `detection`
	- 有标签且长度匹配时：额外有 `labeled_metrics`（accuracy/F1/AUC）
	- 有事件文件时：额外有 `event_association`

