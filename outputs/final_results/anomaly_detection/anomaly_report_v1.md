# 异常检测对比结果 V1

生成时间：2026-04-04
数据来源：IBTrACS 自动生成标签 + 当前 split（train/val/test）

## 1. 本次对比方法

- main_ae：主模型双分支自编码器
- ae_baseline：轻量 AE 基线
- pca：传统无监督 PCA 重构误差
- iforest：传统无监督 IsolationForest

## 2. 核心结果（Test 集）

| 方法 | Accuracy | Precision | Recall | F1 | AUC | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| main_ae | 0.8669 | 0.0000 | 0.0000 | 0.0000 | 0.8462 | 0 | 1 | 65 | 430 |
| ae_baseline | 0.8589 | 0.0000 | 0.0000 | 0.0000 | 0.8272 | 0 | 5 | 65 | 426 |
| pca | 0.8145 | 0.0909 | 0.0462 | 0.0612 | 0.5435 | 3 | 30 | 62 | 401 |
| iforest | 0.8508 | 0.3448 | 0.1538 | 0.2128 | 0.7064 | 10 | 19 | 55 | 412 |

## 3. 结论（当前版本）

- 以 F1 与召回作为异常检出核心指标时，iforest 在 test 集最优（F1=0.2128, Recall=0.1538）。
- main_ae 与 ae_baseline 的 AUC 较高，但在当前阈值策略下几乎不报异常，导致 test 集 TP=0。
- pca 能检出少量正例，但整体区分度与稳定性低于 iforest（AUC=0.5435）。

## 4. 风险与解释

- main_ae / ae_baseline 的高 Accuracy 主要来自负样本占比高，不能单独代表“识别真实台风能力”。
- 若比赛重点是“真实发生台风有多少被识别出来”，应优先看 Recall/F1，而非 Accuracy。

## 5. 建议下一步

- 对 main_ae 与 ae_baseline 做 test 前仅基于 val 的阈值再校准（目标函数偏 Recall 或 F1）。
- 在相同标签下追加 PR-AUC 与命中台风事件数指标，强化比赛口径解释。
- 保留 iforest 作为传统基线对照，主模型继续优化为高召回版本。

## 6. 产物索引

- 统一对比 JSON：outputs/final_results/anomaly_detection/anomaly_methods_comparison.json
- 主模型报告：outputs/anomaly_detection/split_reports.json
- 基线模型报告：outputs/baseline/anomaly_detection/split_reports.json
- 传统方法报告：outputs/baseline/anomaly_detection_traditional/comparison_report.json
