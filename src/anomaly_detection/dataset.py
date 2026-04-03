"""风-浪异常：从 `data/processed/anomaly_detection` 读取各年 `oper_clean` + `wave_clean`。

使用前请先运行 `python scripts/02_preprocess.py`；可选 `--steps clean,split,stats`。
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.dataset_utils import (
    load_norm_stats,
    load_paths_from_manifest,
    project_root,
    standardize_tensor,
)
from data_preprocessing.io import open_nc


class AnomalyCleanDataset(Dataset):
    """每年目录下 oper_clean + wave_clean，返回 oper / wave 两组张量。"""

    def __init__(
        self,
        processed_anomaly_dir: str | Path | None = None,
        split: str | None = None,
        manifest_path: str | Path | None = None,
        norm_stats_path: str | Path | None = None,
        root: Path | None = None,
    ):
        root = root or project_root()
        self._norm = load_norm_stats(Path(norm_stats_path)) if norm_stats_path else None
        if split is None:
            base = Path(processed_anomaly_dir or root / "data/processed/anomaly_detection")
            self.pairs: list[tuple[Path, Path]] = []
            for ydir in sorted(d for d in base.iterdir() if d.is_dir()):
                op = ydir / "oper_clean.nc"
                wv = ydir / "wave_clean.nc"
                if op.is_file() and wv.is_file():
                    self.pairs.append((op, wv))
        else:
            man = (
                Path(manifest_path)
                if manifest_path
                else root / "data/processed/splits/anomaly_detection.json"
            )
            dirs = load_paths_from_manifest(man, split, root)
            self.pairs = []
            for ydir in dirs:
                op, wv = ydir / "oper_clean.nc", ydir / "wave_clean.nc"
                if op.is_file() and wv.is_file():
                    self.pairs.append((op, wv))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        op, wv = self.pairs[idx]
        ds_o = open_nc(op)
        ds_w = open_nc(wv)
        try:
            oper_keys = ("u10", "v10", "wind_speed")
            oper: dict[str, torch.Tensor] = {}
            for k in oper_keys:
                if k in ds_o:
                    t = torch.from_numpy(np.asarray(ds_o[k].values, dtype=np.float32))
                    oper[k] = standardize_tensor(t, k, self._norm)
            wave: dict[str, torch.Tensor] = {}
            for k in ("swh", "mwp", "mwd"):
                if k in ds_w:
                    t = torch.from_numpy(np.asarray(ds_w[k].values, dtype=np.float32))
                    wave[k] = standardize_tensor(t, k, self._norm)
            return {
                "oper": oper,
                "wave": wave,
                "paths": (str(op), str(wv)),
            }
        finally:
            ds_o.close()
            ds_w.close()


class AnomalyFrameDataset(Dataset):
    """按时间步提供风/浪双分支样本，适配网格不一致建模。"""

    def __init__(
        self,
        processed_anomaly_dir: str | Path | None = None,
        split: str | None = None,
        manifest_path: str | Path | None = None,
        norm_stats_path: str | Path | None = None,
        root: Path | None = None,
        open_file_lru_size: int = 6,
    ):
        root = root or project_root()
        self._norm = load_norm_stats(Path(norm_stats_path)) if norm_stats_path else None
        self.open_file_lru_size = max(0, int(open_file_lru_size))
        self._pair_cache: OrderedDict[tuple[Path, Path], tuple[Any, Any]] = OrderedDict()

        if split is None:
            base = Path(processed_anomaly_dir or root / "data/processed/anomaly_detection")
            self.pairs: list[tuple[Path, Path]] = []
            for ydir in sorted(d for d in base.iterdir() if d.is_dir()):
                op = ydir / "oper_clean.nc"
                wv = ydir / "wave_clean.nc"
                if op.is_file() and wv.is_file():
                    self.pairs.append((op, wv))
        else:
            man = (
                Path(manifest_path)
                if manifest_path
                else root / "data/processed/splits/anomaly_detection.json"
            )
            dirs = load_paths_from_manifest(man, split, root)
            self.pairs = []
            for ydir in dirs:
                op, wv = ydir / "oper_clean.nc", ydir / "wave_clean.nc"
                if op.is_file() and wv.is_file():
                    self.pairs.append((op, wv))

        self.index: list[tuple[int, int]] = []
        for i, (op, wv) in enumerate(self.pairs):
            ds_o = open_nc(op)
            ds_w = open_nc(wv)
            try:
                n_o = int(ds_o["u10"].shape[0]) if "u10" in ds_o else 0
                if "swh" in ds_w:
                    n_w = int(ds_w["swh"].shape[0])
                elif "mwp" in ds_w:
                    n_w = int(ds_w["mwp"].shape[0])
                else:
                    n_w = 0
                n = min(n_o, n_w)
                for t in range(n):
                    self.index.append((i, t))
            finally:
                ds_o.close()
                ds_w.close()

    def __len__(self) -> int:
        return len(self.index)

    def _get_pair_ds(self, pair_idx: int) -> tuple[Any, Any]:
        key = self.pairs[pair_idx]
        if self.open_file_lru_size <= 0:
            return open_nc(key[0]), open_nc(key[1])

        found = self._pair_cache.get(key)
        if found is not None:
            self._pair_cache.move_to_end(key)
            return found

        ds_o, ds_w = open_nc(key[0]), open_nc(key[1])
        self._pair_cache[key] = (ds_o, ds_w)
        self._pair_cache.move_to_end(key)
        while len(self._pair_cache) > self.open_file_lru_size:
            _, old_pair = self._pair_cache.popitem(last=False)
            old_pair[0].close()
            old_pair[1].close()
        return ds_o, ds_w

    def close(self) -> None:
        for ds_o, ds_w in self._pair_cache.values():
            ds_o.close()
            ds_w.close()
        self._pair_cache.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _get_var(
        self,
        ds: Any,
        key: str,
        t: int,
        *,
        default_shape: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """返回标准化后的值与有效掩膜。"""

        if key not in ds:
            z = torch.zeros(default_shape, dtype=torch.float32)
            m = torch.zeros(default_shape, dtype=torch.float32)
            return z, m
        arr = np.asarray(ds[key].values[t], dtype=np.float32)
        valid = np.isfinite(arr)
        filled = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        x = torch.from_numpy(filled)
        x = standardize_tensor(x, key, self._norm)
        m = torch.from_numpy(valid.astype(np.float32))
        return x, m

    def __getitem__(self, idx: int) -> dict[str, Any]:
        pair_idx, t = self.index[idx]
        op, wv = self.pairs[pair_idx]
        ds_o, ds_w = self._get_pair_ds(pair_idx)

        if "u10" in ds_o:
            o_shape = tuple(ds_o["u10"].shape[-2:])
        else:
            raise KeyError(f"u10 not found in {op}")

        if "swh" in ds_w:
            w_shape = tuple(ds_w["swh"].shape[-2:])
        elif "mwp" in ds_w:
            w_shape = tuple(ds_w["mwp"].shape[-2:])
        else:
            raise KeyError(f"swh/mwp not found in {wv}")

        o_u10, m_u10 = self._get_var(ds_o, "u10", t, default_shape=o_shape)
        o_v10, m_v10 = self._get_var(ds_o, "v10", t, default_shape=o_shape)
        o_ws, m_ws = self._get_var(ds_o, "wind_speed", t, default_shape=o_shape)

        w_swh, m_swh = self._get_var(ds_w, "swh", t, default_shape=w_shape)
        w_mwp, m_mwp = self._get_var(ds_w, "mwp", t, default_shape=w_shape)
        w_mwd, m_mwd = self._get_var(ds_w, "mwd", t, default_shape=w_shape)

        oper_x = torch.stack([o_u10, o_v10, o_ws], dim=0)
        wave_x = torch.stack([w_swh, w_mwp, w_mwd], dim=0)

        oper_valid = torch.stack([m_u10, m_v10, m_ws], dim=0)
        wave_valid = torch.stack([m_swh, m_mwp, m_mwd], dim=0)

        timestamp = -1
        if "valid_time" in ds_o:
            timestamp = np.asarray(ds_o["valid_time"].values[t]).item()

        if self.open_file_lru_size <= 0:
            ds_o.close()
            ds_w.close()

        return {
            "oper_x": oper_x.float(),
            "wave_x": wave_x.float(),
            "oper_valid": oper_valid.float(),
            "wave_valid": wave_valid.float(),
            "path_oper": str(op),
            "path_wave": str(wv),
            "time_index": int(t),
            "timestamp": timestamp,
        }
