# 中尺度涡旋识别模块 — 当前实现说明

本文档描述当前已落地的中尺度涡旋识别链路：**META4 标注预处理（单脚本）→ U-Net 训练 → 测试集评估与对象导出**。赛题要求见 [`docs/赛题A09_面向海洋环境智能分析系统.md`](../../docs/赛题A09_面向海洋环境智能分析系统.md)。

---

## 一、数据概况

| 项目 | 说明 |
|------|------|
| 数据格式 | NetCDF，单文件 365 天 × 160 × 320 格点 |
| 空间范围 | 约 25°N～45°N，140°E～180°E（西北太平洋） |
| 核心变量 | `adt`（动力高度）、`ugos`（纬向流）、`vgos`（经向流） |
| 数据量 | 5 个文件（1993–2002、2003–2012、2013–2022、2023、2024） |
| 划分 | train 4 个、val 1 个 |
| 预处理 | 由 `scripts/02_preprocess_eddy.py` 完成，产出 `*_clean.nc`、`labels/*_objects_meta4.nc`、`labels/*_label_meta4_mask.nc`、`labels/*_label_meta4_mask_bg0.nc` 与赛题时间切分 |

---

## 二、核心难点与应对策略

### 难点 1：对象级标签到像素级训练标签

默认管线用 **META4 风格标签**（py-eddy-tracker 闭合轮廓检测），并在单个脚本内完成对象级检测、像素栅格化与背景修正，供 U-Net 直接读取 `eddy_mask`。

**一键预处理**：`scripts/02_preprocess_eddy.py` 直接产出训练标签 `*_label_meta4_mask_bg0.nc`，无需再串联其它 02 分拆脚本。

### 难点 2：时序 vs 逐帧

当前数据按单日/单时间片组织，训练时通常按时间切分成 train/val/test，并在每个时间片上做分割学习；如果需要更强的时序建模，可在后续版本增加滑动窗口输入。

---

## 三、技术路线

```
原始 adt/ugos/vgos
        │
        ▼
┌──────────────────────────────────────────────┐
│  1. 统一预处理（02_preprocess_eddy.py）      │  对象识别 -> 像素 mask -> 背景 NaN→0 -> split
└────────────────────┬─────────────────────────┘
                     ▼
┌──────────────────────────────────────────────┐
│  2. U-Net 训练（03_train_eddy.py）           │  多通道输入 (adt, ugos, vgos)
└────────────────────┬─────────────────────────┘
                     ▼
┌──────────────────────────────────────────────┐
│  3. 测试评估（06_eddy_assess.py）            │  指标 + 对象清单导出
└──────────────────────────────────────────────┘
```

---

## 四、详细实现要点

### 4.1 标签生成

主线为 **META4 + U-Net**：

- `02_preprocess_eddy.py`：在 ADT 场上做闭合轮廓检测，生成对象级结果并栅格化为 `eddy_mask`，再将背景统一置 0；
- `03_train_eddy.py`：读取 `*_label_meta4_mask_bg0.nc` 进行训练；
- `06_eddy_assess.py`：在测试集输出分割指标与对象级导出。

### 4.2 样本与输入

- 输入张量：`(C, H, W)`，C=3 通道（adt, ugos, vgos）；
- 分辨率：160×320，可按需下采样或裁剪（如 128×256）；
- 标签：二值 mask（涡/非涡）或三分类（背景、气旋、反气旋）。

### 4.3 模型选型

| 模型 | 特点 | 适用 |
|------|------|------|
| U-Net | 经典分割、编码–解码 | 主训练模型 |
| U-Net + SE/ECA | 通道注意力 | 提升精度 |
| ResUNet | 残差 + U-Net | 更深、更稳 |
| FPN / DeepLabV3 | 多尺度 | 大尺度涡旋 |

当前仓库以 U-Net 为主，后续可按需要扩展为 U-Net+Attention 或 ResUNet。

### 4.4 训练配置（建议）

- 损失：Dice + BCE，或 Focal Loss 应对类别不平衡；
- 优化器：AdamW，lr≈1e-3，cosine 衰减；
- batch_size：8–16（视显存）；
- 数据增强：随机旋转 90°、翻转、裁剪、高斯噪声。

### 4.5 后处理流程

1. 连通域分析；
2. 按面积、形状过滤小碎片；
3. 提取涡旋中心（质心或 mask 几何中心）；
4. 保留气旋/反气旋类别与对象属性；
5. 可选 NMS、轮廓平滑，输出 GeoJSON 等。

---

## 五、实现步骤建议

| 阶段 | 任务 | 预计时间 |
|------|------|----------|
| 1 | 运行 `02_preprocess_eddy.py` 产出训练标签与切分清单 | 1 天 |
| 2 | 运行 `03_train_eddy.py` 训练与验证 | 1–2 天 |
| 3 | 运行 `06_eddy_assess.py` 输出测试指标与对象导出 | 0.5–1 天 |

---

## 六、模块与脚本规划

```
src/eddy_detection/
├── README.md           # 本文档
├── dataset.py          # Dataset（manifest / merged_time）
├── model.py            # U-Net 等
├── trainer.py          # 训练循环
├── predictor.py        # 推理 + 后处理
├── evaluator.py        # IoU、Precision、Recall、F1
└── postprocess.py      # 连通域、中心、类型判定

scripts/
├── 02_preprocess_eddy.py  # 涡旋统一预处理入口
├── 03_train_eddy.py       # 训练入口
└── 06_eddy_assess.py      # 测试评估与对象导出入口
```

---

## 七、验收与评估

- **主要指标**：准确率 ≥75%；
- **建议监控**：IoU、Precision、Recall、F1、混淆矩阵；
- **对比基线**：与历史训练结果做 A/B 对比，验证本链路收益。

---

## 八、参考文献

1. Sun X, Zhang M, Dong J, et al. A Deep Framework for Eddy Detection and Tracking From Satellite Sea Surface Height Data. *IEEE TGRS*, 2020.
2. Khachatrian E, Sandalyuk N, Lozou P. Eddy detection in the marginal ice zone with sentinel-1 data using YOLOv5. *Remote Sensing* 15.9 (2023): 2244.
3. Zi N, et al. Ocean eddy detection based on YOLO deep learning algorithm by SAR data. *RSE* 307 (2024): 114139.
4. 张家灏, 邓科峰, 聂腾飞, 等. 基于机器学习的海洋中尺度涡检测识别研究综述. *计算机工程与科学*, 2021, 43(12): 2115–2125.
