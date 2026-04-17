"""中尺度涡旋推理导出：边界、中心、旋转方向 + 测试集指标。"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eddy_detection.dataset import EddySegmentationDataset  # noqa: E402
from eddy_detection.evaluator import confusion_matrix, segmentation_metrics  # noqa: E402
from eddy_detection.model import EddyUNet  # noqa: E402
from eddy_detection.predictor import infer_batch_to_objects, load_checkpoint  # noqa: E402
from utils.logger import get_logger, setup_logging, tqdm, tqdm_logging  # noqa: E402

_log = get_logger(__name__)


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    x = torch.stack([b["x"] for b in batch], dim=0)
    y = torch.stack([b["y"] for b in batch], dim=0)
    return {"x": x, "y": y, "meta": batch}


def _rotation_name(class_id: int) -> str:
    if class_id == 1:
        return "cyclonic"
    if class_id == 2:
        return "anticyclonic"
    return "background"


def main() -> None:
    ap = argparse.ArgumentParser(description="涡旋推理与结果导出")
    ap.add_argument(
        "--time-split-manifest",
        type=Path,
        default=ROOT / "data/processed/splits/eddy_merged_time.json",
    )
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "outputs/eddy_detection/checkpoints/best.pt",
    )
    ap.add_argument(
        "--norm",
        type=Path,
        default=ROOT / "data/processed/normalization/eddy_norm.json",
    )
    ap.add_argument("--input-steps", type=int, default=1)
    ap.add_argument("--step-stride", type=int, default=1)
    ap.add_argument("--base-channels", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--max-test-samples", type=int, default=None)
    ap.add_argument("--min-region-pixels", type=int, default=16)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/final_results/eddy_detection",
    )
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    setup_logging(log_file=ROOT / "outputs/eddy_detection/infer_eddy.log")

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")

    norm_path = args.norm if args.norm.is_file() else None
    test_ds = EddySegmentationDataset(
        split="test",
        input_steps=args.input_steps,
        step_stride=args.step_stride,
        max_samples=args.max_test_samples,
        time_split_manifest_path=args.time_split_manifest,
        norm_stats_path=norm_path,
        root=ROOT,
    )
    loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_collate,
    )

    model = EddyUNet(
        in_channels=args.input_steps * 3,
        num_classes=3,
        base_channels=args.base_channels,
    )
    state = load_checkpoint(model, args.checkpoint, map_location=args.device)
    model = model.to(args.device)

    cm = torch.zeros((3, 3), dtype=torch.int64)
    records: list[dict[str, Any]] = []

    with torch.no_grad():
        with tqdm_logging():
            for batch in tqdm(loader, desc="infer eddy test", unit="batch"):
                x = batch["x"]
                y = batch["y"]
                meta = batch["meta"]
                objs = infer_batch_to_objects(
                    model,
                    x,
                    device=args.device,
                    min_region_pixels=args.min_region_pixels,
                )

                logits = model(x.to(args.device)).cpu()
                pred = torch.argmax(logits, dim=1)
                cm += confusion_matrix(pred, y, num_classes=3)

                for i, sample in enumerate(objs):
                    item_meta = meta[i]
                    eddies: list[dict[str, Any]] = []
                    for o in sample["objects"]:
                        eddies.append(
                            {
                                "rotation": _rotation_name(int(o["class_id"])),
                                "center_yx": o["center_yx"],
                                "area": int(o["area"]),
                                "bbox_yx": o["bbox_yx"],
                                "boundary_yx": o["boundary_yx"],
                            }
                        )

                    records.append(
                        {
                            "path": item_meta["path"],
                            "time_index": int(item_meta["time_index"]),
                            "cyclonic_count": int(sample["cyclonic_count"]),
                            "anticyclonic_count": int(sample["anticyclonic_count"]),
                            "eddies": eddies,
                        }
                    )

    metrics = segmentation_metrics(cm)
    if "val_metrics" in state:
        metrics["checkpoint_val_metrics"] = state["val_metrics"]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    obj_path = args.out_dir / "test_eddy_objects.json"
    met_path = args.out_dir / "test_metrics.json"
    obj_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    met_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    _log.info("saved objects -> %s", obj_path)
    _log.info("saved metrics -> %s", met_path)
    _log.info("test pixel_acc=%.4f macro_f1=%.4f miou=%.4f", metrics["pixel_acc"], metrics["macro_f1"], metrics["miou"])


if __name__ == "__main__":
    main()
