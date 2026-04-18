"""OceanRace FastAPI 入口：中间件与各任务路由。"""
from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Reduce random native runtime crashes in mixed NumPy/PyTorch OpenMP environments.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

_BACKEND_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _BACKEND_DIR.parent.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import anomaly, eddy, element, health
from .routers.health import print_cuda_startup_banner

try:
    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(1)
except Exception:
    pass


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    print_cuda_startup_banner()
    yield


app = FastAPI(title="OceanRace Backend API", lifespan=_lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(element.router)
app.include_router(eddy.router)
app.include_router(anomaly.router)
