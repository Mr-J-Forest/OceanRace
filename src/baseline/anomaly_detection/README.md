# 异常检测基线（Lightweight Dual-Branch AE）

提供一个可复现实验基线：

- 双分支浅层卷积自编码器（风场/浪场各一支）
- 仅使用重构损失，不启用交叉重构与融合一致性项
- 使用与主模型一致的数据集、阈值、检测报告链路

并提供传统无监督方法用于标准对照：

- PCA（重构误差）
- IsolationForest（异常分数）

## 配置

基线配置与主任务分离：

- `configs/baseline/anomaly_detection/model.yaml`
- `configs/baseline/anomaly_detection/train.yaml`

## 运行

在项目根目录：

```bash
python scripts/05_train_anomaly.py --baseline
```

常见覆盖参数：

```bash
python scripts/05_train_anomaly.py --baseline --epochs 20 --batch-size 16 --num-workers 4
python scripts/05_train_anomaly.py --baseline --labels-json outputs/anomaly_detection/labels.json --threshold-policy val-f1
```

默认中间产物目录：`outputs/baseline/anomaly_detection/`。

## 传统方法统一对比

先分别跑主模型与 AE baseline（得到各自 `summary.json` 与 `split_reports.json`），再运行：

```bash
python scripts/05c_compare_anomaly_methods.py
```

输出：

- `outputs/baseline/anomaly_detection_traditional/summary.json`
- `outputs/baseline/anomaly_detection_traditional/split_reports.json`
- `outputs/baseline/anomaly_detection_traditional/comparison_report.json`
- `outputs/final_results/anomaly_detection/anomaly_methods_comparison.json`
