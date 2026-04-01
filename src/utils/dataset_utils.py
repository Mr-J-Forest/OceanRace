"""供各任务模块 Dataset 复用：划分清单、标准化 JSON、张量标准化。

从本文件导入即可，例如 ``from utils.dataset_utils import project_root, load_paths_from_manifest``。
"""
from __future__ import annotations

import json
from bisect import bisect_right
from pathlib import Path
from typing import Callable

import numpy as np
import torch


def project_root() -> Path:
    """项目根目录（含 `data/`、`src/`）。"""
    return Path(__file__).resolve().parents[2]


def load_paths_from_manifest(manifest_path: Path, split: str, root: Path) -> list[Path]:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if split not in data:
        raise KeyError(f"split {split!r} not in {manifest_path}")
    return [root / Path(r) for r in data[split]]


def load_norm_stats(path: Path) -> dict[str, tuple[float, float]]:
    j = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, tuple[float, float]] = {}
    for k, v in j.get("variables", {}).items():
        out[k] = (float(v["mean"]), float(v["std"]))
    return out


def standardize_tensor(
    t: torch.Tensor,
    key: str,
    norm: dict[str, tuple[float, float]] | None,
) -> torch.Tensor:
    if norm is None or key not in norm:
        return t
    m, s = norm[key]
    return (t - m) / s


def destandardize_tensor(
    t: torch.Tensor,
    key: str,
    norm: dict[str, tuple[float, float]] | None,
) -> torch.Tensor:
    if norm is None or key not in norm:
        return t
    m, s = norm[key]
    return t * s + m


def discover_clean_paths(processed_dir: Path) -> list[Path]:
    """递归发现 processed 目录下的 ``*_clean.nc``。"""

    return sorted(processed_dir.rglob("*_clean.nc"))


def build_cumulative_ends(lengths: list[int]) -> list[int]:
    """将每个文件时间长度转换为累计结束索引（全局时间轴）。"""

    out: list[int] = []
    total = 0
    for n in lengths:
        if n < 0:
            raise ValueError("time length must be >= 0")
        total += n
        out.append(total)
    return out


def build_global_window_starts(
    *,
    total_len: int,
    input_steps: int,
    output_steps: int,
    stride: int,
) -> list[int]:
    """在全局时间轴上生成滑窗起点。"""

    if input_steps <= 0 or output_steps <= 0:
        raise ValueError("input_steps and output_steps must be > 0")
    if stride <= 0:
        raise ValueError("stride must be > 0")
    need = input_steps + output_steps
    if total_len < need:
        return []
    return list(range(0, total_len - need + 1, stride))


def locate_file_index(cumulative_ends: list[int], global_t: int) -> int:
    """给定全局时间索引，返回其所属文件索引。"""

    idx = bisect_right(cumulative_ends, global_t)
    if idx >= len(cumulative_ends):
        raise IndexError("global time index out of bounds")
    return idx


def slice_across_files(
    *,
    paths: list[Path],
    file_lengths: list[int],
    cumulative_ends: list[int],
    global_t0: int,
    length: int,
    read_slice: Callable[[Path, int, int], np.ndarray],
) -> np.ndarray:
    """跨文件切片：在全局时间轴上取长度为 ``length`` 的连续片段。"""

    if length <= 0:
        raise ValueError("length must be > 0")
    left = length
    cur = global_t0
    chunks: list[np.ndarray] = []

    while left > 0:
        file_idx = locate_file_index(cumulative_ends, cur)
        file_start = 0 if file_idx == 0 else cumulative_ends[file_idx - 1]
        offset = cur - file_start
        take = min(left, file_lengths[file_idx] - offset)
        chunks.append(read_slice(paths[file_idx], offset, offset + take))
        cur += take
        left -= take

    return np.concatenate(chunks, axis=0)
