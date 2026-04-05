# 异常检测对比结果 V2（严格比赛划分）

生成时间：2026-04-04
划分口径：train=2014-2023，test=2024，val=2025
标签来源：IBTrACS v04r01（WP）自动对齐

## 1. 划分与标签统计

- train: 2480 样本，正例 492（19.84%）
- val: 248 样本，正例 3（1.21%）
- test: 248 样本，正例 19（7.66%）

## 2. Test 集关键指标（四模型）

| 方法 | Accuracy | Precision | Recall | F1 | AUC | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| main_ae | 0.9395 | 0.5667 | 0.8947 | 0.6939 | 0.9800 | 17 | 13 | 2 | 216 |
| ae_baseline | 0.8548 | 0.2258 | 0.3684 | 0.2800 | 0.8849 | 7 | 24 | 12 | 205 |
| pca | 0.8750 | 0.0000 | 0.0000 | 0.0000 | 0.4544 | 0 | 12 | 19 | 217 |
| iforest | 0.9113 | 0.4286 | 0.4737 | 0.4500 | 0.7368 | 9 | 12 | 10 | 217 |

## 3. 结论（严格口径）

- 主模型 `main_ae` 在 test 集表现最好（F1=0.6939，Recall=0.8947，AUC=0.9800）。
- `iforest` 作为传统方法在 test 集有可用检出能力（F1=0.4500，Recall=0.4737），可作为强对照组。
- `pca` 在当前阈值下几乎无法检出正例（TP=0），不建议作为最终提交方案。

## 4. 建议

- 以 `main_ae` 作为主提交方向，保留 `iforest` 作为传统无监督对照。
- 继续补充事件级命中统计（命中台风数/漏检台风数）强化“真实台风识别率”解释。
- 对 `main_ae` 再做阈值灵敏度分析，报告 Accuracy/F1/Recall 的折中曲线。

## 5. 产物位置

- 对比 JSON：outputs/final_results/anomaly_detection/anomaly_methods_comparison.json
- 比赛划分 manifest：data/processed/splits/anomaly_detection_competition.json
- 标签元信息：outputs/anomaly_detection/ibtracs_label_meta_competition.json
