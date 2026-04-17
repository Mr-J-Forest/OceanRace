# OceanRace 前端应用 (Vue 3 + Vite)

本应用是 **OceanRace 海洋环境智能分析系统** 的前端展示大屏，基于 Vue 3 `<script setup>` 和 TailwindCSS 构建，核心图表由 Plotly.js 驱动。提供“要素预测”与“异常检测”等模块的动态可视化。

---

## 🚀 快速开始

```bash
# 1. 安装依赖
npm install

# 2. 启动开发服务器
npm run dev

# 3. 生产环境构建
npm run build
```

---

## 🌟 核心功能与交互设计

### 1. 🌊 空间演变展示 (Spatial Evolution)
- **高性能 Heatmap 渲染**：利用 Plotly 原生的 `zsmooth: 'best'` 替代了繁重的前端插值逻辑，大幅度降低了 CPU 消耗并消除了播放动画时的卡顿。
- **等比例地图锁定**：配置 `scaleanchor` 与 `scaleratio: 1` 强制 Y 轴 (纬度) 与 X 轴 (经度) 保持 1:1 的真实地理比例，杜绝了自适应窗口大小带来的图像拉伸和变形。
- **洋流流线追踪 (Streamlines)**：在合成流速 (SSUV) 图表中，基于 `U` 和 `V` 速度分量，使用 Euler 法在二维网格中前向追踪计算，利用带端点箭头的平滑样条曲线（`spline`）叠加绘制了连续的洋流走势。
  - **支持独立开关**：地图右上角配备“洋流流线”勾选框，在不重新请求接口的情况下零延迟切换流线显示。
- **沉浸式播放控制条**：播放控制栏（播放/暂停、进度滑块、速度选项 `0.5x, 1x, 2x`）与组件标题栏（Header）进行了无缝整合，避免了悬浮控制条遮挡地图数据的现象。

### 2. 📈 区域变化趋势 (Timeseries Curves)
- **点选联动交互**：监听空间演变地图的 `plotly_click` 事件。用户可点击海洋中的任意网格点，下方折线图将自动向后端请求该坐标的精确数据并更新，标题同步变更为 `点 (Y, X) 变化趋势`。
- **平滑趋势展示**：通过 `mode: 'lines'` 剔除密集点标记，并辅以 `shape: 'spline'` 与 `smoothing: 1.3`，使长达 72 步的时序预测线更加顺滑。
- **Y 轴自适应防抖**：计算当前渲染数据的极值，并为 Y 轴增加 75% 的上下 Padding (`range: [yMin, yMax]`)，有效拉大尺度区间，避免微小波动被过度放大导致视觉上的剧烈跳变。

### 3. ⚠️ 异常检测模块 (Anomaly Inspector)
- 支持加载特定的预测标签和事件，快速查看风浪灾害（台风关联等）的智能识别结果。
- 包含了监测状态、异常分类、台风耦合度和预警简报的完整工作流展示。

---

## 🛠️ 技术栈

- **框架**：Vue 3 (Composition API / script setup)
- **构建工具**：Vite 8
- **样式**：Tailwind CSS (定制了科技风玻璃态组件与渐变色彩)
- **图表**：Plotly.js (高性能网格/等值线/时序渲染)
- **通信**：Axios (结合后端 FastAPI 接口使用)
- **图标**：Lucide Vue Next

## 🔗 依赖后端服务
本应用需要与 OceanRace Python 后端服务协同工作。
- 默认 API Base URL：`http://127.0.0.1:8000/api`
- 开发态也可直接请求 `/api`，由 `vite.config.ts` 代理到后端。
- 请确保已启动位于 `src/web/backend/app/main.py` 的 FastAPI 服务。
