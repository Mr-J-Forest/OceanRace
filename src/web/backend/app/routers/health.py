import torch
from fastapi import APIRouter

router = APIRouter(tags=["meta"])


def torch_cuda_snapshot() -> dict:
    """当前进程 PyTorch 是否链接到 CUDA、能否创建 GPU 张量（与要素/涡旋推理 device 选择一致）。"""
    snap: dict = {
        "torch_version": torch.__version__,
        "cuda_compiled": getattr(torch.version, "cuda", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "current_device": None,
    }
    if snap["cuda_available"] and snap["device_count"] > 0:
        try:
            snap["current_device"] = {
                "index": int(torch.cuda.current_device()),
                "name": torch.cuda.get_device_name(0),
            }
        except Exception as e:
            snap["current_device"] = {"error": str(e)}
    return snap


def print_cuda_startup_banner() -> None:
    """uvicorn 启动时打一行，便于确认是否误装 CPU 版 PyTorch。"""
    s = torch_cuda_snapshot()
    if s["cuda_available"]:
        name = (s.get("current_device") or {}).get("name", "?")
        print(
            f"[OceanRace] PyTorch CUDA 可用 | 编译时 CUDA={s['cuda_compiled']} | GPU: {name}",
            flush=True,
        )
    else:
        print(
            "[OceanRace] PyTorch 未检测到可用 CUDA（推理将走 CPU，会很慢）。"
            " 常见原因：pip 默认装的是 CPU 版 torch。"
            " 请到 https://pytorch.org 按本机环境选择带 CUDA 的安装命令，例如: "
            "pip install torch --index-url https://download.pytorch.org/whl/cu124",
            flush=True,
        )


@router.get("/")
def root_status():
    return {
        "service": "OceanRace Backend API",
        "status": "ok",
        "docs": "/docs",
        "health": "/healthz",
    }


@router.get("/healthz")
def healthz():
    return {"status": "ok", "torch": torch_cuda_snapshot()}
