"""FastAPI 请求体验证模型。"""
from __future__ import annotations

from pydantic import BaseModel


class DatasetInfoRequest(BaseModel):
    data_path: str


class PredictRequest(BaseModel):
    model_path: str
    data_path: str
    start_idx: int


class EddyDatasetInfoRequest(BaseModel):
    data_path: str
    input_steps: int = 1


class EddyPredictRequest(BaseModel):
    model_path: str
    data_path: str
    start_idx: int
    input_steps: int = 1
    base_channels: int = 32
    min_region_pixels: int = 16
    horizon_steps: int = 1


class EddyPredictDayRequest(BaseModel):
    model_path: str
    data_path: str
    day_index: int
    input_steps: int = 1
    base_channels: int = 32
    min_region_pixels: int = 16


class EddyDateIndexRequest(BaseModel):
    data_path: str
    date: str
    input_steps: int = 1


class AnomalyInspectRequest(BaseModel):
    labels_json: str
    events_json: str
    manifest_path: str = "data/processed/splits/anomaly_detection_competition.json"
    processed_dir: str = "data/processed/anomaly_detection/path.txt"
    norm_stats_path: str = "data/processed/normalization/anomaly_detection_norm.json"
    split: str = "test"
    open_file_lru_size: int = 32
    max_points: int = 200
    recent_window_hours: int = 24
    snapshot_only: bool = False
    include_snapshot: bool = False
    snapshot_index: int | None = None
