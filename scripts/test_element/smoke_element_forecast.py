"""
极少样本要素预报基线冒烟：在 outputs/smoke_element_baseline/ 写入合成 *_clean.nc、
划分 JSON、norm JSON，然后跑 1 个 epoch 验证训练链路。

项目根目录执行::

  python scripts/smoke_element_forecast.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
OUT = ROOT / "outputs" / "smoke_element_baseline"
REL_NC = "outputs/smoke_element_baseline/fake_clean.nc"


def _write_fake_clean_nc(path: Path) -> None:
    """单文件、时间维 ≥ input_len+forecast_len，与真实 element 变量名一致。"""
    t, la, lo = 24, 4, 4
    rng = np.random.default_rng(0)
    data_vars: dict = {}
    for name in ("sst", "sss", "ssu", "ssv"):
        data_vars[name] = (
            ("time", "lat", "lon"),
            rng.standard_normal((t, la, lo)).astype(np.float32),
        )
    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": np.arange(t, dtype=np.float32),
            "lat": np.arange(la, dtype=np.float32),
            "lon": np.arange(lo, dtype=np.float32),
        },
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)


def _write_manifest(path: Path) -> None:
    rel = REL_NC.replace("\\", "/")
    payload = {
        "task": "element_forecasting",
        "train": [rel],
        "val": [rel],
        "test": [],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_norm(path: Path) -> None:
    payload = {
        "task": "element_forecasting",
        "variables": {
            "sst": {"mean": 0.0, "std": 1.0},
            "sss": {"mean": 0.0, "std": 1.0},
            "ssu": {"mean": 0.0, "std": 1.0},
            "ssv": {"mean": 0.0, "std": 1.0},
        },
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    nc_path = OUT / "fake_clean.nc"
    man_path = OUT / "element_forecasting_smoke.json"
    norm_path = OUT / "element_forecasting_norm.json"

    print("writing synthetic NetCDF and manifests ->", OUT)
    _write_fake_clean_nc(nc_path)
    _write_manifest(man_path)
    _write_norm(norm_path)

    env = os.environ.copy()
    sep = ";" if sys.platform == "win32" else ":"
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(SRC) if not prev else f"{str(SRC)}{sep}{prev}"

    cmd = [
        sys.executable,
        "-m",
        "baseline.element_forecasting.train",
        "--manifest",
        str(man_path),
        "--norm",
        str(norm_path),
        "--epochs",
        "1",
        "--batch-size",
        "1",
        "--input-len",
        "8",
        "--forecast-len",
        "8",
        "--hidden",
        "16",
        "--layers",
        "1",
    ]
    print("running:", " ".join(cmd))
    r = subprocess.run(cmd, cwd=ROOT, env=env)
    raise SystemExit(r.returncode)


if __name__ == "__main__":
    main()
