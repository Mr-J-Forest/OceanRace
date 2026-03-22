# 中尺度涡旋识别模块 — 预备方案

本文档为涡流（中尺度涡旋）识别赛题的预备技术方案，供开发参考。赛题要求见 [`docs/赛题A09_面向海洋环境智能分析系统.md`](../../docs/赛题A09_面向海洋环境智能分析系统.md)。

---

## 一、数据概况

| 项目 | 说明 |
|------|------|
| 数据格式 | NetCDF，单文件 365 天 × 160 × 320 格点 |
| 空间范围 | 约 25°N～45°N，140°E～180°E（西北太平洋） |
| 核心变量 | `adt`（动力高度）、`ugos`（纬向流）、`vgos`（经向流） |
| 数据量 | 5 个文件（1993–2002、2003–2012、2013–2022、2023、2024） |
| 划分 | train 4 个、val 1 个 |
| 预处理 | 已由 `scripts/02_preprocess.py` 完成，产出 `*_clean.nc`、`eddy_norm.json` |

---

## 二、核心难点与应对策略

### 难点 1：无标注标签

原始 NetCDF 仅含 `adt/ugos/vgos`，**无涡旋边界或中心标注**。

**应对：伪标签 + 深度学习**

- 用 Okubo–Weiss、Winding Angle、SSH 闭合等高线等经典算法生成伪标签；
- 以伪标签训练深度学习模型，提升鲁棒性与精度。

### 难点 2：时序 vs 逐帧

数据为逐日序列，可采取：

- **逐日场**：每天 1 个样本，样本量大；
- **滑动窗口**：多日序列，更利于刻画涡旋演变，工程复杂度更高。

---

## 三、技术路线

```
原始 adt/ugos/vgos
        │
        ▼
┌───────────────────────┐
│  1. 伪标签生成        │  Okubo-Weiss / SSH 闭合等高线
│     (eddy_mask, type) │
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│  2. 样本构建          │  按天切分 or 滑动窗口
│     输入 X + 标签 Y   │
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│  3. 数据增强          │  旋转、翻转、噪声、时间扰动
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│  4. 深度学习模型      │  U-Net / U-Net+Attention / ResUNet
│     语义分割          │  多通道输入 (adt,ugos,vgos)
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│  5. 后处理            │  连通域、中心提取、NMS、边界平滑
│     输出边界+中心+类型│
└───────────────────────┘
```

---

## 四、详细实现要点

### 4.1 伪标签生成

**方案 A：Okubo–Weiss (OW)**

$$OW = \left(\frac{\partial u}{\partial x} - \frac{\partial v}{\partial y}\right)^2 + \left(\frac{\partial v}{\partial x} + \frac{\partial u}{\partial y}\right)^2 - f^2$$

- OW &lt; 0 表示涡旋区；
- 对 `ugos`、`vgos` 求梯度，结合科氏参数 \(f\) 计算 OW；
- 生成二值或多类 mask（背景、气旋、反气旋）。

**方案 B：adt 闭合等高线**

- 在 adt 场中找局部极值（高/低对应反气旋/气旋）；
- 向外扩张闭合等高线作为涡旋边界；
- 结合 `ugos`、`vgos` 流场约束，剔除非涡旋结构。

建议先用 OW 做伪标签，再视效果叠加 SSH 方法。

### 4.2 样本与输入

- 输入张量：`(C, H, W)`，C=3 通道（adt, ugos, vgos）；
- 分辨率：160×320，可按需下采样或裁剪（如 128×256）；
- 标签：二值 mask（涡/非涡）或三分类（背景、气旋、反气旋）。

### 4.3 模型选型

| 模型 | 特点 | 适用 |
|------|------|------|
| U-Net | 经典分割、编码–解码 |  baseline 首选 |
| U-Net + SE/ECA | 通道注意力 | 提升精度 |
| ResUNet | 残差 + U-Net | 更深、更稳 |
| FPN / DeepLabV3 | 多尺度 | 大尺度涡旋 |

建议 baseline 用 U-Net，再尝试 U-Net+ECA 或 ResUNet。

### 4.4 训练配置（建议）

- 损失：Dice + BCE，或 Focal Loss 应对类别不平衡；
- 优化器：AdamW，lr≈1e-3，cosine 衰减；
- batch_size：8–16（视显存）；
- 数据增强：随机旋转 90°、翻转、裁剪、高斯噪声。

### 4.5 后处理流程

1. 连通域分析；
2. 按面积、形状过滤小碎片；
3. 提取涡旋中心（质心或 mask 几何中心）；
4. 判定类型：根据 adt 梯度/曲率或流场旋度区分气旋/反气旋；
5. 可选 NMS、轮廓平滑，输出 GeoJSON 等。

---

## 五、实现步骤建议

| 阶段 | 任务 | 预计时间 |
|------|------|----------|
| 1 | 实现 OW 伪标签生成脚本（基于 `*_clean.nc`） | 2–3 天 |
| 2 | 扩展 `EddyCleanDataset`，支持逐日样本 + mask 标签 | 1 天 |
| 3 | 搭建 U-Net 及 baseline 训练入口 | 2 天 |
| 4 | 训练、调参、数据增强迭代 | 2–3 天 |
| 5 | 实现后处理（中心、类型、GeoJSON）与评估脚本 | 1–2 天 |
| 6 | 对比 OW 与 DL，验证准确率 ≥75% | 1 天 |

---

## 六、模块与脚本规划

```
src/eddy_detection/
├── README.md           # 本文档
├── dataset.py          # 已有，待扩展
├── labeler.py          # 伪标签生成（OW / SSH）
├── model.py            # U-Net 等
├── trainer.py          # 训练循环
├── predictor.py        # 推理 + 后处理
├── evaluator.py        # IoU、Precision、Recall、F1
└── postprocess.py      # 连通域、中心、类型判定

scripts/
├── 02a_generate_eddy_labels.py   # 生成伪标签（新增）
└── 02_train_eddy.py              # 训练入口
```

---

## 七、验收与评估

- **主要指标**：准确率 ≥75%；
- **建议监控**：IoU、Precision、Recall、F1、混淆矩阵；
- **对比基线**：纯 OW 法 vs OW 伪标签 + DL，验证 DL 收益。

---

## 八、参考文献

1. Sun X, Zhang M, Dong J, et al. A Deep Framework for Eddy Detection and Tracking From Satellite Sea Surface Height Data. *IEEE TGRS*, 2020.
2. Khachatrian E, Sandalyuk N, Lozou P. Eddy detection in the marginal ice zone with sentinel-1 data using YOLOv5. *Remote Sensing* 15.9 (2023): 2244.
3. Zi N, et al. Ocean eddy detection based on YOLO deep learning algorithm by SAR data. *RSE* 307 (2024): 114139.
4. 张家灏, 邓科峰, 聂腾飞, 等. 基于机器学习的海洋中尺度涡检测识别研究综述. *计算机工程与科学*, 2021, 43(12): 2115–2125.
