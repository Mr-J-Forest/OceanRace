# 脚本说明

在项目**根目录**下执行（`python scripts/<name>.py`）。配置见 `configs/data_config.yaml`。

---

## 脚本一览

| 脚本 | 作用 |
|------|------|
| `01_data_inspect.py` | 只读抽样统计 `data/raw` 下 NetCDF |
| `02_preprocess.py` | raw → 清洗落盘；可选划分、splits、训练集 norm、回写配置 |
| `smoke_element_forecast.py` | 合成极少样本 + 1 epoch，验证要素 ConvLSTM 基线链路（`outputs/smoke_element_baseline/`，见 `src/baseline/element_forecasting/README.md`） |
| `03_train_eddy.py` | 涡旋识别训练入口（**占位**，待接入 `src/eddy_detection/`） |
| `04_train_forecast.py` | 要素预报训练入口（支持命令行覆盖参数） |
| `run_element_baseline_train.py` | 要素预报基线训练，读取 `configs/baseline/element_forecasting/{model,train}.yaml` |
| `05_train_anomaly.py` | 风-浪异常训练与评估（主模型/基线可切换，支持阈值策略、labels/events） |
| `05b_prepare_anomaly_eval_templates.py` | 生成 anomaly 的 labels/events 模板（用于准确率/AUC与事件关联评估） |
| `05c_compare_anomaly_methods.py` | 统一对比主模型、AE baseline、PCA、IsolationForest |
| `05d_generate_anomaly_labels_from_ibtracs.py` | 从 IBTrACS 轨迹自动生成 labels/events（按样本 timestamp 对齐） |
| `06_run_pipeline.py` | 端到端流水线（**占位**） |
| `07_generate_report.py` | 评估报告生成（**占位**） |
| `08_check_element_forecast_competition.py` | 要素预测比赛门槛检查（支持 12h 滚动到 72h + MSE≤15%），输出 PASS/FAIL JSON |

除 `01`/`02` 外，`04_train_forecast.py`、`05_train_anomaly.py`、`05b_prepare_anomaly_eval_templates.py`、`05c_compare_anomaly_methods.py` 已实现；其余多为预留入口。

---

## 日志（logger）

本仓库脚本在把 `src` 加入 `PYTHONPATH` 后，使用 **`utils.logger`**：入口在 `argparse` 解析完成后调用 `setup_logging()`，运行过程用 `get_logger(__name__).info(...)` 等，**不用 `print` 作为正式输出**。

- **详细约定、参数、`tqdm` 协同、JSON stdout 注意点**：见 **`src/utils/README.md`** 中「`logger.py`：统一日志」一节。
- **`01_data_inspect.py`**：统计结果 JSON 仍写到 **stdout**（纯 JSON，便于管道）；写 `--out` 文件时用 logger 提示路径。
- **调试更啰嗦的日志**：可在运行前设置环境变量，例如 PowerShell：`$env:OCEAN_LOG_LEVEL = "DEBUG"`。

---

## `01_data_inspect.py`

只读：按配置抽样 `data/raw` 下 NetCDF，统计缺测与最值。

```powershell
python scripts/01_data_inspect.py
python scripts/01_data_inspect.py --out outputs/data_stats/summary.json
```

默认结果打印到终端；`--out` 指定 JSON 落盘路径（父目录会自动创建）。

---

## `02_preprocess.py`

从 `data/raw` 清洗写入 `data/processed`，可选 **划分 train/val/test**、**训练集标准化参数**、**回写 `data_config.yaml`**。清洗与 split/stats 阶段会使用 **`tqdm` 进度条**（与日志协同，见 `utils.logger` 的 `tqdm_logging`）。

### 步骤 `--steps` / `--stage`（同义）

逗号分隔，可多选：

| 取值 | 含义 |
|------|------|
| `clean` | 读 raw，清洗后写入 `data/processed`（**默认仅本步**） |
| `split` | 按 `configs/data_config.yaml` 中 `split` 比例划分，写出 `data/processed/splits/*.json` |
| `stats` | 仅用**训练集**估计各变量 mean/std，写出 `data/processed/normalization/*_norm.json`，并合并进配置 |
| `merge` | 将指定任务下清洗后的多个样本按时间坐标拼接成总文件（`eddy/element` 输出 `all_clean_merged.nc`；`anomaly` 输出 `oper_merged.nc` 与 `wave_merged.nc`） |
| `validate` | 仅运行 **validator**：检查 `splits/*.json` 路径与 processed 样本（见 `data_preprocessing.validator`）；也可与 `clean`/`split`/`stats` 组合，或在本命令末尾加 `--validate` |
| `all` | 等价于 `clean,split,stats` 三步（**不含** validate） |

含 `stats` 时会在最后更新 `configs/data_config.yaml` 中的 `artifacts` / `standardization`（首次回写前会备份 `data_config.yaml.bak`）。

### 常用命令

```powershell
python scripts/02_preprocess.py --help

# 只清洗（默认 steps=clean），三类任务全跑
python scripts/02_preprocess.py --task all

# 全流程：清洗 + 划分 + 统计 + 回写配置
python scripts/02_preprocess.py --task all --steps all

# 已有清洗结果，只做划分与统计（PowerShell 建议给逗号参数加引号）
python scripts/02_preprocess.py --task all --steps "split,stats"
# 与 --stage 同义
python scripts/02_preprocess.py --task all --stage "split,stats"

# 只跑某一类任务，例如要素
python scripts/02_preprocess.py --task element --steps "clean,split,stats"

# 将清洗后样本按时序合并（单任务）
python scripts/02_preprocess.py --task element --steps merge

# 清洗后立刻合并
python scripts/02_preprocess.py --task all --steps "clean,merge"

# 调试：每类只处理少量文件/年份
python scripts/02_preprocess.py --task all --limit 2

# 清洗阶段多进程（仅 clean 生效）
python scripts/02_preprocess.py --task all --steps clean -j 4

# 仅根据已有 splits/*.json 与 normalization/*_norm.json 回写配置（不跑清洗/划分/统计）
python scripts/02_preprocess.py --sync-config-only

# 仅校验（manifest + processed 抽检）
python scripts/02_preprocess.py --steps validate

# 全流程后再校验；大数据集可加 --validate-limit 限制每任务检查条数
python scripts/02_preprocess.py --task all --steps all --validate --validate-limit 50
```

### 参数摘要

| 参数 | 说明 |
|------|------|
| `--config` | 默认 `configs/data_config.yaml` |
| `--task` | `all` \| `eddy` \| `element` \| `anomaly` |
| `--steps` / `--stage` | `clean`、`split`、`stats`、`all` 或其逗号组合 |
| `--limit` | 每类最多处理的文件数（eddy/element）或异常年份数；不设则按配置 `batch` 或全部 |
| `-j` / `--workers` | **仅 clean**：并行进程数，默认 `1` |
| `--sync-config-only` | 只合并 artifacts/standardization 到 YAML |
| `--validate` | 在完成其它步骤后执行 `validate_manifest_and_samples` |
| `--validate-limit` | 校验时每任务最多检查的样本数；默认不限制 |

### 数据与配置

- 原始数据路径：`data_config.yaml` → `paths.raw`。
- 输出路径：`paths.processed`、`paths.splits`、`paths.normalization`。
- 划分比例：`split.train_ratio` / `val_ratio` / `test_ratio` / `seed`。

---

## 基线实验（`src/baseline/`，非 `scripts/`）

三任务基线代码在 **`src/baseline/<任务>/`**，与 `scripts/` 并列，需 **`PYTHONPATH=src`** 后以 **模块方式**运行（同样使用 `utils.logger` / `utils.dataset_utils` 等）。

| 入口 | 说明 |
| `python scripts/04_train_forecast.py` | 要素预报训练（主模型或 ConvLSTM 基线） |
| `python -m baseline.element_forecasting.train` | 同上，模块方式（需 `PYTHONPATH=src`） |
| `python scripts/05_train_anomaly.py --baseline` | 异常检测轻量基线训练（双分支浅层 AE） |
| `src/baseline/eddy_detection/` | 占位，见目录 `README.md` |

示例（项目根目录）：

```powershell
python scripts/04_train_forecast.py --epochs 5 --batch-size 2 --help
```

详见 **`src/baseline/element_forecasting/README.md`** 与 **`src/baseline/anomaly_detection/README.md`**。

---

## 典型流程

1. `01_data_inspect.py`：了解 raw 数据质量（可选 `--out` 保存 JSON）。
2. `02_preprocess.py --task all --steps all`：清洗 → 划分 → 标准化参数 → 更新配置。
3. **训练**：要素预报 `python scripts/04_train_forecast.py`；其它任务用 `src/<任务>/dataset.py` 或 `src/baseline/<任务>/`。
| `gui_app.py` | Gradio构建的图形化界面，支持要素预测模型的可视化预测与预览（涡旋检测及异常检测预留入口）|
