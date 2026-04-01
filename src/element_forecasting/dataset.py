"""要素长期预测数据集：按时间窗口从 processed NetCDF 采样。"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.dataset_utils import (
    build_cumulative_ends,
    build_global_window_starts,
    discover_clean_paths,
    load_norm_stats,
    load_paths_from_manifest,
    project_root,
    standardize_tensor,
)
from data_preprocessing.io import open_nc


def _iter_data_vars(ds: Any) -> list[str]:
    return [str(k) for k in ds.data_vars.keys()]


def _infer_var_names_from_first_file(path: Path) -> tuple[str, ...]:
    ds = open_nc(path)
    try:
        names = _iter_data_vars(ds)
        if not names:
            raise ValueError(f"no data variables in {path}")
        return tuple(names)
    finally:
        ds.close()


def _time_len(ds: Any, var_names: tuple[str, ...]) -> int:
    for name in var_names:
        if name in ds:
            return int(ds[name].shape[0])
    raise ValueError("cannot infer time dimension from configured variables")


def _to_bool_valid(values: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    finite = np.isfinite(values)
    if valid is None:
        return finite
    vm = np.asarray(valid, dtype=np.float32) > 0.5
    return finite & vm


def _sanitize_values(values: np.ndarray, valid_bool: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    arr = np.where(valid_bool, arr, 0.0)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


class ElementForecastWindowDataset(Dataset):
    """按时间窗构建样本：输入 ``input_steps``，输出 ``output_steps``。"""

    def __init__(
        self,
        processed_dir: str | Path | None = None,
        data_file: str | Path | None = None,
        var_names: tuple[str, ...] | None = ("sst", "sss", "ssu", "ssv"),
        input_steps: int = 12,
        output_steps: int = 12,
        window_stride: int = 1,
        stitch_across_files: bool = True,
        open_file_lru_size: int = 16,
        split: str | None = None,
        manifest_path: str | Path | None = None,
        norm_stats_path: str | Path | None = None,
        root: Path | None = None,
    ):
        if input_steps <= 0 or output_steps <= 0:
            raise ValueError("input_steps and output_steps must be > 0")
        if window_stride <= 0:
            raise ValueError("window_stride must be > 0")

        root = root or project_root()
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.window_stride = window_stride
        self.stitch_across_files = stitch_across_files
        self._norm = load_norm_stats(Path(norm_stats_path)) if norm_stats_path else None
        # 仅复用文件句柄，避免每个样本反复 open/close 小文件造成 I/O 抖动。
        self._open_ds_lru: OrderedDict[Path, Any] = OrderedDict()
        self._max_open_files = max(1, int(open_file_lru_size))

        if data_file is not None:
            p = Path(data_file)
            if not p.is_absolute():
                p = root / p
            self.paths = [p]
        elif split is None:
            self.dir = Path(processed_dir or root / "data/processed/element_forecasting")
            self.paths = discover_clean_paths(self.dir)
        else:
            man = (
                Path(manifest_path)
                if manifest_path
                else root / "data/processed/splits/element_forecasting.json"
            )
            self.paths = load_paths_from_manifest(man, split, root)

        if not self.paths:
            raise ValueError("no clean NetCDF files found")
        for path in self.paths:
            if not Path(path).is_file():
                raise FileNotFoundError(f"dataset file not found: {path}")

        if var_names is None:
            self.var_names = _infer_var_names_from_first_file(self.paths[0])
        else:
            self.var_names = var_names

        self._file_time_lens: list[int] = []
        for path in self.paths:
            ds = open_nc(path)
            try:
                missing = [v for v in self.var_names if v not in ds]
                if missing:
                    raise KeyError(f"missing vars in {path}: {missing}")
                tlen = _time_len(ds, self.var_names)
            finally:
                ds.close()
            self._file_time_lens.append(tlen)

        self._cum_ends = build_cumulative_ends(self._file_time_lens)
        total = self._cum_ends[-1] if self._cum_ends else 0

        self._windows: list[tuple[Path, int]] = []
        self._global_starts: list[int] = []
        need = self.input_steps + self.output_steps

        if self.stitch_across_files:
            self._global_starts = build_global_window_starts(
                total_len=total,
                input_steps=self.input_steps,
                output_steps=self.output_steps,
                stride=self.window_stride,
            )
            return

        for path, tlen in zip(self.paths, self._file_time_lens):
            if tlen < need:
                continue
            for t0 in range(0, tlen - need + 1, self.window_stride):
                self._windows.append((path, t0))

    def __len__(self) -> int:
        if self.stitch_across_files:
            return len(self._global_starts)
        return len(self._windows)

    def _get_open_ds(self, path: Path) -> Any:
        ds = self._open_ds_lru.get(path)
        if ds is not None:
            self._open_ds_lru.move_to_end(path)
            return ds
        ds = open_nc(path)
        self._open_ds_lru[path] = ds
        self._open_ds_lru.move_to_end(path)
        while len(self._open_ds_lru) > self._max_open_files:
            _, evicted = self._open_ds_lru.popitem(last=False)
            evicted.close()
        return ds

    def _close_all_open_ds(self) -> None:
        while self._open_ds_lru:
            _, ds = self._open_ds_lru.popitem(last=False)
            ds.close()

    def __del__(self) -> None:
        try:
            self._close_all_open_ds()
        except Exception:
            # 避免析构阶段异常影响进程退出
            pass

    def _read_multi_var_pairs(self, ds: Any, t0: int, t1: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """一次时间切片读取该片段的全部变量，减少小文件随机访问开销。"""
        chunk = ds.isel(time=slice(t0, t1))
        values: dict[str, np.ndarray] = {}
        valids: dict[str, np.ndarray] = {}
        for var_name in self.var_names:
            raw = np.asarray(chunk[var_name].values, dtype=np.float32)
            valid_name = f"{var_name}_valid"
            valid = np.asarray(chunk[valid_name].values, dtype=np.float32) if valid_name in chunk else None
            valid_bool = _to_bool_valid(raw, valid)
            values[var_name] = _sanitize_values(raw, valid_bool)
            valids[var_name] = valid_bool.astype(np.float32)
        return values, valids

    def _iter_window_spans(self, global_t0: int, length: int) -> list[tuple[int, int, int]]:
        spans: list[tuple[int, int, int]] = []
        remain = int(length)
        cursor = int(global_t0)
        while remain > 0:
            file_idx = int(np.searchsorted(self._cum_ends, cursor, side="right"))
            file_start = 0 if file_idx == 0 else self._cum_ends[file_idx - 1]
            local_t0 = cursor - file_start
            take = min(remain, self._file_time_lens[file_idx] - local_t0)
            spans.append((file_idx, local_t0, local_t0 + take))
            cursor += take
            remain -= take
        return spans

    def _read_concat_window(self, global_t0: int, length: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], str]:
        value_chunks: dict[str, list[np.ndarray]] = {v: [] for v in self.var_names}
        valid_chunks: dict[str, list[np.ndarray]] = {v: [] for v in self.var_names}
        spans = self._iter_window_spans(global_t0, length)
        first_path = str(self.paths[spans[0][0]])

        for file_idx, t0, t1 in spans:
            ds = self._get_open_ds(self.paths[file_idx])
            values, valids = self._read_multi_var_pairs(ds, t0, t1)
            for v in self.var_names:
                value_chunks[v].append(values[v])
                valid_chunks[v].append(valids[v])

        values = {v: np.concatenate(value_chunks[v], axis=0) for v in self.var_names}
        valids = {v: np.concatenate(valid_chunks[v], axis=0) for v in self.var_names}
        return values, valids, first_path

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self.stitch_across_files:
            t0_global = self._global_starts[idx]
            t_in = self.input_steps
            t_out = self.output_steps
            values, valids, first_path = self._read_concat_window(t0_global, t_in + t_out)
            xs: list[torch.Tensor] = []
            ys: list[torch.Tensor] = []
            y_valids: list[torch.Tensor] = []
            for v in self.var_names:
                whole = values[v]
                whole_valid = valids[v]
                x_np = whole[:t_in]
                y_np = whole[t_in:t_in + t_out]
                x_t = standardize_tensor(torch.from_numpy(x_np), v, self._norm)
                y_t = standardize_tensor(torch.from_numpy(y_np), v, self._norm)
                xs.append(x_t)
                ys.append(y_t)

                y_valid_np = whole_valid[t_in:t_in + t_out]
                y_valids.append(torch.from_numpy(y_valid_np).to(dtype=torch.float32))

            x = torch.stack(xs, dim=1)
            y = torch.stack(ys, dim=1)
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            y_valid = torch.stack(y_valids, dim=1)

            return {
                "x": x,
                "y": y,
                "y_valid": y_valid,
                "path": first_path,
                "t0": t0_global,
                "var_names": self.var_names,
            }

        path, t0 = self._windows[idx]
        ds = self._get_open_ds(path)
        values, valids = self._read_multi_var_pairs(ds, t0, t0 + self.input_steps + self.output_steps)
        xs: list[torch.Tensor] = []
        ys: list[torch.Tensor] = []
        y_valids: list[torch.Tensor] = []
        for v in self.var_names:
            whole = values[v]
            whole_valid = valids[v]
            x_np = whole[: self.input_steps]
            y_np = whole[self.input_steps : self.input_steps + self.output_steps]
            x_t = standardize_tensor(torch.from_numpy(x_np), v, self._norm)
            y_t = standardize_tensor(torch.from_numpy(y_np), v, self._norm)
            xs.append(x_t)
            ys.append(y_t)

            y_valid_np = whole_valid[self.input_steps : self.input_steps + self.output_steps]
            y_valids.append(torch.from_numpy(y_valid_np).to(dtype=torch.float32))

        x = torch.stack(xs, dim=1)
        y = torch.stack(ys, dim=1)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        y_valid = torch.stack(y_valids, dim=1)
        return {
            "x": x,
            "y": y,
            "y_valid": y_valid,
            "path": str(path),
            "t0": t0,
            "var_names": self.var_names,
        }


# 兼容旧导出名，避免外部调用断裂。
ElementForecastCleanDataset = ElementForecastWindowDataset
