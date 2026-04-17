# META4.0 DT 涡旋识别任务重构方案（先验方法确认 + AI自动识别框架）

## 0. 任务要求重述（按你的要求）

本任务不是“向 META4.0 DT 靠拢”，而是**必须**满足：

1. 源数据集只有一个：`data/processed/eddy_detection/19930101_20241231_clean.nc`。
2. 在该单一源数据集上按 META4.0 DT 方法进行涡旋轮廓标注，生成标准标签。
3. 使用“同一源数据 + 标注结果”训练并实现基于人工智能的中尺度涡旋自动识别模块。
4. 目标能力：边界定位 + 形态分割（实例级或目标级分割）。

因此，流程应改为：
**先严格复现/调用 META4.0 DT 规则链路生成标准标注，再做 AI 学习与推理模块。**

说明：这里的“训练集/验证集/测试集”是**单一源数据集按时间切分后的子集**，不是新的独立数据集。

---

## 1. PDF 内容整理（方法核心）

依据 handbook（META4.0 DT）可提炼为三段流程：

### 1.1 Detection（检测）

- 在 ADT 场上扫描闭合等值线（-100 到 100 cm，0.2 cm 间隔）。
- 对候选轮廓做约束筛选：
  - 形状误差阈值（shape error）；
  - 振幅阈值（amplitude）；
  - 轮廓内极值数量约束（仅一个主极值）；
  - 像素数范围约束；
  - 掩膜像素约束；
  - 气旋/反气旋的符号与高度关系约束。
- 对轮廓进行过采样（点数乘以 10），并以“最大环向平均地转流速轮廓”拟合圆来定义中心与半径。
- 使用 Visvalingam-Whyatt 对轮廓重采样（用于标准化存储）。

### 1.2 Estimation（特征估计）

- 关键特征：中心位置、振幅、速度半径、平均速度、速度剖面、有效轮廓、速度轮廓等。
- 振幅定义：|SSH(local extremum) - SSH(outermost contour)|。

### 1.3 Networking（网络化跟踪）

- 以同极性涡旋轮廓重叠建立关联。
- 主重叠率：intersection/union（并集重叠率），阈值 10%。
- 嵌套场景补充重叠率：intersection/min(area)，阈值 99%。
- 在时间窗内（N=7天）关联，允许短缺测，不强制虚拟重建。
- 显式处理 splitting/merging，并以 segment + network 组织时空演化。

---

## 2. 先确认：py-eddy-tracker 是否实现了 PDF 方法

结论：**实现了同一类核心方法链路（可参数化复现 META4.0 DT 的关键步骤）**，但默认参数不一定就是 handbook 数值，需要按 META4.0 配置化固化。

### 2.1 检测链路对应性

- 闭合轮廓扫描 + step 控制：
  - `src/py_eddy_tracker/dataset/grid.py` 中 `eddy_identification(..., step=...)`。
- 形状误差、像素数、掩膜、极性/高度关系、振幅等筛选：
  - 同文件中 `shape_error`、`pixel_limit`、mask 检查、`has_value`、`get_amplitude` 逻辑。
- 轮廓过采样与圆拟合中心半径：
  - 同文件 `presampling_multiplier`、`_fit_circle_path`、`radius_s`/`lon`/`lat`。
- Visvalingam 重采样：
  - 同文件 `sampling_method="visvalingam"` 和 `resample = visvalingam`。

### 2.2 “单极值/振幅”约束对应性

- `src/py_eddy_tracker/eddy_feature.py` 的 `Amplitude` 类：
  - 通过 `mle`（maximum local extrema）与局地极值检测函数控制轮廓内极值数量；
  - 通过 `within_amplitude_limits()` 和 `nb_step_min` 等控制振幅阈值。

### 2.3 网络化与分裂/合并对应性

- 轮廓重叠代价函数：
  - `src/py_eddy_tracker/poly.py` 的 `vertice_overlap()`，同时支持：
    - intersection/union；
    - minimal-area；
    - hybrid（并集不足时用最小面积判据，且有 >0.99 的嵌套保护逻辑）。
- 网络构建入口：
  - `src/py_eddy_tracker/appli/network.py` 的 `build_network()` / `divide_network()`。
- 观测关联与“最大重叠优先”选择：
  - `src/py_eddy_tracker/observations/tracking.py` 中 `get_next_obs()` / `get_previous_obs()`。
- segment/network 与 splitting/merging 事件模型：
  - `src/py_eddy_tracker/observations/network.py` 中 `splitting_event()` 等接口。

### 2.4 与 handbook 的差异点（必须显式处理）

- 默认参数可能不同（例如默认 `min_overlap=0.2`、`window=1`、`shape_error=55` 等）。
- handbook 给的是 META4.0 DT 产品参数组合（例如 10% + 99% + N=7 等）。

因此：
**py-eddy-tracker 可以作为 META4.0 DT 方法的工程实现底座，但必须通过配置锁定成“META4.0 DT 参数版本”。**

---

## 3. 中尺度涡旋识别任务的“重构后”总框架

## 3.1 总体架构

阶段A（规则标注层）：
- 输入：ADT/SSH、ugos/vgos（或可推导流速场）。
- 引擎：py-eddy-tracker（META4.0 DT 参数模板）。
- 输出：
  - 实例轮廓（有效轮廓 + 速度轮廓）；
  - 类别（气旋/反气旋）；
  - 结构化属性（中心、振幅、半径、速度特征）；
  - 网络关系（track/segment/merge/split）。

阶段B（AI识别层）：
- 输入：阶段A的高质量标签 + 原始网格场（可叠加派生特征）。
- 模型任务拆分：
  - 任务1：边界定位（目标检测/中心回归）；
  - 任务2：形态分割（实例分割或语义分割+后处理实例化）；
  - 任务3：属性回归（振幅、半径、速度半径等，可选多任务学习）。
- 输出：
  - 自动化涡旋识别结果（mask/contour + 属性 + 置信度）。

阶段C（时序关联层）：
- 方案1：直接复用 py-eddy-tracker 的网络化关联。
- 方案2：AI候选先验 + py-eddy-tracker 重叠规则做时序归并。

推荐：优先方案2（工程稳健、可解释性强）。

## 3.2 数据组织定义（单一源数据集）

源数据（唯一）：
- `data/processed/eddy_detection/19930101_20241231_clean.nc`

派生产物A（规则标注结果，META4.0 DT）：
- 与源数据同时间轴的标签文件（例如 `labels/19930101_20241231_label.nc`）；
- 每个时刻包含：轮廓/掩膜标签 + 极性 + 可选属性/关系字段。

派生产物B（训练样本索引）：
- 对同一源数据按年份切分得到 train/val/test 时间索引；
- 训练时读取同一 clean.nc + label.nc，通过索引取样，不复制底层数据。

## 3.3 训练与推理模块建议

- 模型基线：
  - 分割：Mask2Former / YOLO-seg / U-Net(海洋场特化版)
  - 边界回归：CenterNet 或 keypoint + contour head
- 多任务损失：
  - 分类损失 + 分割损失 + 属性回归损失
- 物理约束后处理：
  - 极性一致性、振幅阈值、形状约束、最小像素约束
- 跟踪：
  - 使用 py-eddy-tracker overlap 规则进行网络化（保持与 META4.0 语义一致）。

---

## 4. 实施顺序（可直接执行）

1. 固化 META4.0 参数模板（第一优先级）
- 将 py-eddy-tracker 的 detection/network 参数显式配置为 handbook 版本。

2. 在单一源数据集上生成标准标签
- 生成轮廓标签、属性标签、网络关系标签。

3. 生成训练索引
- 对同一时间序列做样本切片与时间切分，建立 train/val/test。

4. 训练 AI 自动识别模块
- 先边界分割主任务，再叠加属性回归。

5. 构建评估体系
- 分割：IoU、Boundary-F1
- 定位：中心误差、半径误差
- 物理一致性：振幅/极性一致率
- 时序：网络关联准确率（merge/split 识别质量）

---

## 5. 关键结论

- 当前仓库内的 `py-eddy-tracker-master`（若移动到 [OceanRace](../README.md) 根目录下）已具备并实现 handbook 所述核心方法机制。
- 你的任务应改为：
  - 用 py-eddy-tracker（META4.0 参数化）在唯一源数据 `19930101_20241231_clean.nc` 上生成标准标签；
  - 再基于同一源数据 + 标签，进行时间切分训练 AI 自动化识别与形态分割模块。
- 这是“按任务要求直接落地”的路径，不是“概念靠拢”。

补充：当前仓库已经把这一思路工程化为“对象级标签 → mask 背景清零 → mask 化训练 → U-Net 推理/评估 → Web 可视化”的实现链路；更细的执行命令以 [`单一数据集_涡旋标注与训练执行手册.md`](单一数据集_涡旋标注与训练执行手册.md) 为准。
