"""跨平台打开 NetCDF；Windows 下含中文路径时 netCDF4 可能失败，则复制到临时目录再打开。"""
from __future__ import annotations

import hashlib
import shutil
import sys
import tempfile
from pathlib import Path

import xarray as xr


def _temp_copy(path: Path) -> Path:
    h = hashlib.md5(str(path.resolve()).encode("utf-8")).hexdigest()[:20]
    d = Path(tempfile.gettempdir()) / "ocean_ai_nc"
    d.mkdir(parents=True, exist_ok=True)
    dst = d / f"{h}.nc"
    if not dst.is_file() or dst.stat().st_size != path.stat().st_size:
        shutil.copy2(path, dst)
    return dst


def open_nc(path: str | Path, *, decode_times: bool = True) -> xr.Dataset:
    path = Path(path).resolve()
    if sys.platform == "win32":
        try:
            return xr.open_dataset(path, decode_times=decode_times)
        except (OSError, FileNotFoundError, ValueError):
            return xr.open_dataset(_temp_copy(path), decode_times=decode_times)
    return xr.open_dataset(path, decode_times=decode_times)
