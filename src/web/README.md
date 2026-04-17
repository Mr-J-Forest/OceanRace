# Web 模块 (OceanRace 交互式可视化与控制台)

本模块包含针对 OceanRace 海洋环境智能分析系统开发的全套 Web 交互客户端与服务端程序。主要用于对模型预测产出（涉及涡旋检测、要素预测、异常检测）进行直观的时空动态可视化与交互分析。

## 📁 目录结构

```text
src/web/
├── frontend/       # 前端工程 (Vue 3 + Vite + Tailwind CSS v4)
├── backend/        # 后端 API 服务 (Python + FastAPI)
└── README.md       # 本说明说明
```

---

## 🎨 前端系统 (Frontend)

基于现代前端化技术栈构建的科幻风/暗黑系“玻璃拟物”态仪表盘。

### 1. 技术栈
* **核心框架**: Vue 3 (Composition API) + Vite 构建工具
* **样式方案**: Tailwind CSS v4 (采用全新的扁平化 `@theme` 配置与变量体系)
* **图标组件**: `lucide-vue-next` (风格干练统一的工业级图标库)
* **核心可视化库**: Plotly.js (负责支持复杂的海洋物理场 2D Heatmap 及趋势折线的硬件加速渲染)
* **网络请求**: Axios

### 2. 核心可视化功能
* **空间演变展示区 (Spatial Evolution)**: 依靠 Plotly 动态呈现 SST（海表温度）、SSS（海水盐度）、SSUV（合成流速场）等核心气气象要素。自带双线性插值(`upsampleGridBilinear`)与平滑核填补功能，有效规避由于陆地掩膜造成的截断锯齿。
* **沉浸式播放控制**: 提供时间帧 (Frame) 游标系统，支持按 `$STEP_HOURS` 为周期的动画播放，直观巡览未来时间维度上预测分布。
* **区域变化趋势区 (Regional Trends)**: 横向展开 1xN 阵列时序折线图，联动空间展示，通过垂直游标锚定呈现特定时间的宏观变化走向。

### 3. 开发与运行指令

请确保系统中已安装 Node.js (推荐 v18+)。

```bash
cd src/web/frontend

# 安装依赖包包
npm install

# 启动开发服务器 (默认运行于 http://localhost:5174)
npm run dev

# 生产环境打包构建
npm run build
```

---

## ⚙️ 后端服务 (Backend)

FastAPI 后端已接入，负责承接前端请求并串联涡旋、要素与异常模块的数据接口。

### 运行方式

在项目根目录启动：

```bash
python -m uvicorn src.web.backend.app.main:app --host 0.0.0.0 --port 8000
```

### 通信接口

默认暴露于 `http://127.0.0.1:8000/api`，前端开发服务器通过 Vite 代理转发 `/api` 请求。

当前前端主要依赖以下路由：

* `GET /api/default-data-path`
* `GET /api/eddy/default-paths`
* `GET /api/eddy/default-data-path`
* `POST /api/dataset-info`
* `POST /api/predict`
* `POST /api/eddy/dataset-info`
* `POST /api/eddy/predict-day`
* `POST /api/anomaly/inspect`

* `GET /api/predict/{session_id}/step/{step_idx}`: 获取特定推演时次的 2D 物理空间网格快照 (包含 SST, SSS, SSU, SSV 等通道数据)。
* `GET /api/predict/{session_id}/curve`: 获取整个时间预测跨度的全域/指定区域长效趋势宏观统计数据。

其中涡旋链路配套的对象级标签生成、mask 转换、训练与推理结果会继续写入 `outputs/` 与 `data/processed/` 对应目录。

---

## 🔗 协作说明 (AGENTS.md 遵循)

前端在进行任何渲染工作时，严格遵循主仓库中的 `src/utils/visualization_defaults.py` 色彩方案及 DPI 视觉指导（例如 `Turbo`, `Viridis`, `Cividis` 等严格物理学科用配色调色板已内嵌于 Vue 代码中做双重对齐）。
对于向前端发送产物，主模型应统一回写于 `outputs/final_results/` 或系统缓存映射以供该 Web 模块后端进行序列化分发。