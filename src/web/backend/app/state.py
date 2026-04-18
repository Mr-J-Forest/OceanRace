"""进程内缓存（预测会话、涡旋模型与数据集句柄等）。"""
from __future__ import annotations

from threading import Lock
from typing import Any

from src.eddy_detection.model import EddyUNet

prediction_cache: dict[str, Any] = {}
eddy_prediction_cache: dict[str, Any] = {}
anomaly_timestamps_cache: dict[tuple[str, str, str], list[int]] = {}
anomaly_overview_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
anomaly_snapshot_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
anomaly_snapshot_dataset_cache: dict[tuple[str, str, str, str | None], dict[str, Any]] = {}
ANOMALY_SNAPSHOT_CACHE_MAX = 96
ANOMALY_OVERVIEW_CACHE_SCHEMA = 2
eddy_model_cache: dict[tuple[str, int, int, str, int, int], EddyUNet] = {}
eddy_dataset_meta_cache: dict[str, dict[str, Any]] = {}
eddy_norm_cache: dict[str, Any] = {"path": "", "variables": {}}
eddy_dataset_handle_cache: dict[str, dict[str, Any]] = {}
eddy_window_cache: dict[tuple[str, int, int, int, int], dict[str, Any]] = {}
eddy_cache_lock = Lock()

# 要素预报：避免每次 /predict 都 torch.load + 重建 HybridElementForecastModel（主要耗时来源）
element_predictor_cache: dict[tuple[str, str, float], Any] = {}
# 复用 ElementForecastWindowDataset（含 NetCDF LRU 句柄），避免重复扫描与 open_nc
element_forecast_dataset_cache: dict[tuple[Any, ...], Any] = {}
element_predictor_lock = Lock()
