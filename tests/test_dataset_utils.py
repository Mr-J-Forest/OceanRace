"""dataset_utils 通用滑窗工具单测。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from utils.dataset_utils import (
    build_cumulative_ends,
    build_global_window_starts,
    destandardize_tensor,
    discover_clean_paths,
    locate_file_index,
    slice_across_files,
    standardize_tensor,
)


def test_build_cumulative_ends() -> None:
    assert build_cumulative_ends([3, 0, 2]) == [3, 3, 5]
    with pytest.raises(ValueError):
        build_cumulative_ends([1, -1])


def test_build_global_window_starts() -> None:
    starts = build_global_window_starts(total_len=10, input_steps=3, output_steps=2, stride=2)
    assert starts == [0, 2, 4]
    assert build_global_window_starts(total_len=3, input_steps=2, output_steps=2, stride=1) == []


def test_locate_file_index() -> None:
    cum = [3, 8, 10]
    assert locate_file_index(cum, 0) == 0
    assert locate_file_index(cum, 2) == 0
    assert locate_file_index(cum, 3) == 1
    assert locate_file_index(cum, 9) == 2
    with pytest.raises(IndexError):
        locate_file_index(cum, 10)


def test_slice_across_files() -> None:
    paths = [Path("a"), Path("b")]
    file_lengths = [3, 3]
    cum = [3, 6]
    data = {
        "a": np.array([0.0, 1.0, 2.0], dtype=np.float32),
        "b": np.array([10.0, 11.0, 12.0], dtype=np.float32),
    }

    def _read_slice(path: Path, s: int, e: int) -> np.ndarray:
        arr = data[path.as_posix()][s:e]
        return arr[:, None, None]

    got = slice_across_files(
        paths=paths,
        file_lengths=file_lengths,
        cumulative_ends=cum,
        global_t0=1,
        length=4,
        read_slice=_read_slice,
    )
    assert got.shape == (4, 1, 1)
    assert np.allclose(got[:, 0, 0], np.array([1.0, 2.0, 10.0, 11.0], dtype=np.float32))


def test_discover_clean_paths(tmp_path) -> None:
    d = tmp_path / "root"
    (d / "x").mkdir(parents=True)
    (d / "x" / "a_clean.nc").write_bytes(b"1")
    (d / "b_clean.nc").write_bytes(b"2")
    got = discover_clean_paths(d)
    assert sorted(p.name for p in got) == ["a_clean.nc", "b_clean.nc"]


def test_standardize_destandardize_roundtrip() -> None:
    t = np.array([1.0, 3.0, 5.0], dtype=np.float32)
    x = standardize_tensor(
        torch.from_numpy(t),
        key="sst",
        norm={"sst": (2.0, 0.5)},
    )
    y = destandardize_tensor(x, key="sst", norm={"sst": (2.0, 0.5)})
    assert np.allclose(y.numpy(), t)
