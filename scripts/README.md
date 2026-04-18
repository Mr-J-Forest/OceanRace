# 脚本说明

在项目**根目录**下执行（`python scripts/<name>.py`）。配置见 `configs/data_config.yaml`。

---

## 脚本一览

| 脚本 | 作用 |
|------|------|
| `01_data_inspect.py` | 只读抽样统计 `data/raw` 下 NetCDF |
| `02_preprocess_eddy.py` | 涡旋一键预处理（单脚本）：对象识别 -> 像素 mask -> 背景显式 0 -> 赛题切分 |
| `02_preprocess_element.py` | 要素一键预处理：clean + merge + split + stats（可直接接 `04_train_forecast.py`） |
| `02_preprocess_anomaly.py` | 异常一键预处理：clean + split + stats（可直接接 `05_train_anomaly.py`） |
| `sync_data_config.py` | 仅根据已有 `splits/*.json` 与 `normalization/*_norm.json` 回写 `data_config.yaml` |
| `validate_processed.py` | 校验三任务 manifest 路径与 processed 抽检 |
| `03_train_eddy.py` | 涡旋识别训练入口（读取 META4 mask 标签，训练 U-Net 并评估） |
| `06_eddy_assess.py` | 涡旋测试集评估入口（输出 test 指标与对象清单） |
| `04_train_forecast.py` | 要素预报训练入口（支持命令行覆盖参数） |
| `05_train_anomaly.py` | 风-浪异常训练与评估（主模型/基线可切换，支持阈值策略、labels/events） |
| `06_anomaly_assess.py` | 异常评估：`templates`（labels/events 模板）/ `ibtracs`（IBTrACS 标签）/ `compare`（多方法对比）；实现见 `src/anomaly_detection/assess/` |
| `06_element_assess.py` | 要素预报比赛口径验收：单文件、滚动长时预测、阈值 PASS/FAIL，输出 JSON/CSV/Markdown 与图 |
| `07_web_run.py` | 一键启动 Web：FastAPI（uvicorn）+ 可选 Vite 前端；支持 `--backend-only` / `--frontend-only` |

三任务预处理请分别使用 `02_preprocess_eddy.py`、`02_preprocess_element.py`、`02_preprocess_anomaly.py`；核心逻辑在 `src/data_preprocessing/task_pipelines.py`。

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

## 任务预处理（`02_preprocess_*.py`）

每个脚本从 `data/raw` 对应子目录读到 `data/processed`，并写出 **划分清单**、**训练集标准化 JSON**、回写 **`data_config.yaml`**（与 `stats` 等价的最后一步）。实现均在 `src/data_preprocessing/task_pipelines.py`，脚本仅传参。

### 常用命令

```powershell
# 涡旋：clean + META4 标签 + split + stats
python scripts/02_preprocess_eddy.py -j 4 --validate

# 要素：clean + merge（生成 all_clean_merged.nc 与 path.txt）+ split + stats
python scripts/02_preprocess_element.py -j 4 --validate

# 异常：clean + split + stats（可选 --merge 额外生成 oper/wave merged）
python scripts/02_preprocess_anomaly.py -j 4 --validate

# 仅回写配置（已有 splits 与 norm JSON）
python scripts/sync_data_config.py

# 三任务一起校验（manifest + 抽检）
python scripts/validate_processed.py --validate-limit 50
```

各脚本支持 `--limit`（调试）、`-j`（清洗并行）、`--validate`（完成后校验）。详见各文件内 `argparse` 帮助。

---

## 基线实验（`src/baseline/`，非 `scripts/`）

三任务基线代码在 **`src/baseline/<任务>/`**，与 `scripts/` 并列，需 **`PYTHONPATH=src`** 后以 **模块方式**运行（同样使用 `utils.logger` / `utils.dataset_utils` 等）。

| 入口 | 说明 |
| `python scripts/04_train_forecast.py` | 要素预报训练（主模型或 ConvLSTM 基线） |
| `python -m baseline.element_forecasting.train` | 同上，模块方式（需 `PYTHONPATH=src`） |
| `python scripts/05_train_anomaly.py --baseline` | 异常检测轻量基线训练（双分支浅层 AE） |
| `src/baseline/eddy_detection/` | 涡旋基线预留区，见目录 `README.md` |

示例（项目根目录）：

```powershell
python scripts/04_train_forecast.py --epochs 5 --batch-size 2 --help
```

详见 **`src/baseline/element_forecasting/README.md`** 与 **`src/baseline/anomaly_detection/README.md`**。

---

## 典型流程

1. `01_data_inspect.py`：了解 raw 数据质量（可选 `--out` 保存 JSON）。
2. 按任务分别运行 `02_preprocess_eddy.py` / `02_preprocess_element.py` / `02_preprocess_anomaly.py`（或只跑你当前要训练的任务）。
3. **涡旋链路**：`02_preprocess_eddy.py` -> `03_train_eddy.py` -> `06_eddy_assess.py`。
4. **其它任务训练**：要素 `python scripts/04_train_forecast.py`，异常 `python scripts/05_train_anomaly.py`。
