"""涡旋任务数据集。

- `EddyCleanDataset`：保留原始按文件读取接口。
- `EddySegmentationDataset`：用于中尺度涡旋分割训练（逐时刻样本）。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from data_preprocessing.io import open_nc
from utils.dataset_utils import (
    load_norm_stats,
    load_paths_from_manifest,
    project_root,
    standardize_tensor,
)


class EddyCleanDataset(Dataset):
    """eddy_detection 清洗文件：按文件样本，张量形状多为 `(time, lat, lon)`。"""

    def __init__(
        self,
        processed_dir: str | Path | None = None,
        var_names: tuple[str, ...] = ("adt", "ugos", "vgos"),
        split: str | None = None,
        manifest_path: str | Path | None = None,
        norm_stats_path: str | Path | None = None,
        root: Path | None = None,
    ):
        root = root or project_root()
        self.var_names = var_names
        self._norm = load_norm_stats(Path(norm_stats_path)) if norm_stats_path else None
        if split is None:
            self.dir = Path(processed_dir or root / "data/processed/eddy_detection")
            self.paths = sorted(self.dir.glob("*_clean.nc"))
        else:
            man = Path(manifest_path) if manifest_path else root / "data/processed/splits/eddy.json"
            self.paths = load_paths_from_manifest(man, split, root)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        path = self.paths[idx]
        ds = open_nc(path)
        try:
            out: dict[str, Any] = {"path": str(path)}
            for v in self.var_names:
                if v in ds:
                    t = torch.from_numpy(np.asarray(ds[v].values, dtype=np.float32))
                    out[v] = standardize_tensor(t, v, self._norm)
            return out
        finally:
            ds.close()


class EddySegmentationDataset(Dataset):
    """逐时刻样本的涡旋分割数据集。

    返回：
    - `x`: `(C*input_steps, H, W)`
    - `y`: `(H, W)`（0=背景,1=气旋,2=反气旋）
    """

    def __init__(
        self,
        split: str,
        input_steps: int = 1,
        step_stride: int = 1,
        max_samples: int | None = None,
        var_names: tuple[str, ...] = ("adt", "ugos", "vgos"),
        manifest_path: str | Path | None = None,
        time_split_manifest_path: str | Path | None = None,
        clean_nc_path: str | Path | None = None,
        label_nc_path: str | Path | None = None,
        norm_stats_path: str | Path | None = None,
        labels_dir: str | Path | None = None,
        root: Path | None = None,
    ) -> None:
        super().__init__()
        if input_steps <= 0:
            raise ValueError("input_steps must be > 0")
        if step_stride <= 0:
            raise ValueError("step_stride must be > 0")

        root = root or project_root()
        self.root = root
        self.var_names = var_names
        self.input_steps = int(input_steps)
        self.step_stride = int(step_stride)
        self._norm = load_norm_stats(Path(norm_stats_path)) if norm_stats_path else None
        self.max_samples = max_samples
        self.mode = "manifest"

        self.labels_dir = Path(labels_dir) if labels_dir else root / "data/processed/eddy_detection/labels"
        self._samples: list[tuple[Path, Path, int]] = []
        self._merged_x_ds: Any | None = None
        self._merged_y_ds: Any | None = None

        if time_split_manifest_path is not None or clean_nc_path is not None:
            self.mode = "merged_time"
            self._build_index_merged(
                split=split,
                root=root,
                time_split_manifest_path=time_split_manifest_path,
                clean_nc_path=clean_nc_path,
                label_nc_path=label_nc_path,
            )
        else:
            man = Path(manifest_path) if manifest_path else root / "data/processed/splits/eddy.json"
            self.paths = load_paths_from_manifest(man, split, root)
            self._build_index_manifest()

        if self.max_samples is not None and self.max_samples > 0:
            self._samples = self._samples[: self.max_samples]

    def _label_path(self, clean_path: Path) -> Path:
        stem = clean_path.name.replace("_clean.nc", "")
        return self.labels_dir / f"{stem}_label.nc"

    def _build_index_manifest(self) -> None:
        for p in self.paths:
            lp = self._label_path(p)
            if not lp.is_file():
                continue
            ds = open_nc(p)
            try:
                tlen = int(ds[self.var_names[0]].shape[0])
            finally:
                ds.close()
            need = self.input_steps
            if tlen < need:
                continue
            for t in range(need - 1, tlen, self.step_stride):
                self._samples.append((p, lp, t))

    def _build_index_merged(
        self,
        *,
        split: str,
        root: Path,
        time_split_manifest_path: str | Path | None,
        clean_nc_path: str | Path | None,
        label_nc_path: str | Path | None,
    ) -> None:
        if time_split_manifest_path is not None:
            mp = Path(time_split_manifest_path)
            data = json.loads(mp.read_text(encoding="utf-8"))
            if data.get("mode") != "merged_time":
                raise ValueError(f"unsupported split manifest mode: {data.get('mode')}")
            if split not in data:
                raise KeyError(f"split {split!r} not in {mp}")

            clean_p = Path(clean_nc_path) if clean_nc_path is not None else (root / Path(data["clean_nc"]))
            label_p = Path(label_nc_path) if label_nc_path is not None else (root / Path(data["label_nc"]))
            start = int(data[split]["start"])
            end = int(data[split]["end"])
        else:
            if clean_nc_path is None or label_nc_path is None:
                raise ValueError("clean_nc_path and label_nc_path are required in merged mode")
            clean_p = Path(clean_nc_path)
            label_p = Path(label_nc_path)
            ds = open_nc(clean_p)
            try:
                tlen = int(ds[self.var_names[0]].shape[0])
            finally:
                ds.close()
            # 无显式时间划分时默认全量，适合推理或调试。
            start, end = 0, tlen - 1

        if not clean_p.is_file():
            raise FileNotFoundError(f"clean file missing: {clean_p}")
        if not label_p.is_file():
            raise FileNotFoundError(f"label file missing: {label_p}")
        if end < start:
            raise ValueError(f"invalid split range: start={start}, end={end}")

        # 单文件模式下缓存打开句柄，避免每个样本重复打开 NetCDF。
        self._merged_x_ds = open_nc(clean_p)
        self._merged_y_ds = open_nc(label_p)

        begin = max(start, self.input_steps - 1)
        for t in range(begin, end + 1, self.step_stride):
            self._samples.append((clean_p, label_p, t))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        clean_path, label_path, t = self._samples[idx]
        ds_x = self._merged_x_ds if self._merged_x_ds is not None else open_nc(clean_path)
        ds_y = self._merged_y_ds if self._merged_y_ds is not None else open_nc(label_path)
        try:
            x_parts: list[torch.Tensor] = []
            t0 = t - self.input_steps + 1
            for v in self.var_names:
                # Slice on xarray first, then realize values, to avoid loading the full variable.
                arr = np.asarray(ds_x[v][t0 : t + 1].values, dtype=np.float32)
                tv = torch.from_numpy(arr)
                tv = standardize_tensor(tv, v, self._norm)
                x_parts.append(tv)

            # (T,C,H,W) -> (C*T,H,W)
            x = torch.stack(x_parts, dim=1).reshape(-1, x_parts[0].shape[-2], x_parts[0].shape[-1])
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

            y_np = np.asarray(ds_y["eddy_mask"][t].values, dtype=np.int64)
            y = torch.from_numpy(y_np)
            return {
                "x": x,
                "y": y,
                "path": str(clean_path),
                "label_path": str(label_path),
                "time_index": int(t),
            }
        finally:
            if self._merged_x_ds is None:
                ds_x.close()
            if self._merged_y_ds is None:
                ds_y.close()

    def close(self) -> None:
        if self._merged_x_ds is not None:
            self._merged_x_ds.close()
            self._merged_x_ds = None
        if self._merged_y_ds is not None:
            self._merged_y_ds.close()
            self._merged_y_ds = None

    def __del__(self) -> None:
        self.close()
