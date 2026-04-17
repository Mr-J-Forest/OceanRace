# 涡旋识别基线（预留）

在此目录下放置涡旋检测/分割基线模型（如 U-Net、轻量 CNN），与 `src/eddy_detection/` 主实现区分。当前项目主线已经落在 `src/eddy_detection/` 的 META4 对象标签到 mask 再到 U-Net 训练/评估链路上，这里保留给后续对照实验使用。

数据：`data/processed/eddy_detection/*_clean.nc`；划分见 `data/processed/splits/eddy.json`。
