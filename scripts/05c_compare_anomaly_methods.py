"""Run and compare anomaly methods: main AE, AE baseline, PCA, IsolationForest."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from torch.utils.data import Subset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from anomaly_detection.dataset import AnomalyFrameDataset
from anomaly_detection.detector import build_detection_report
from anomaly_detection.evaluator import evaluate_with_labels, roc_auc_from_scores, summarize_errors
from baseline.anomaly_detection.traditional import TraditionalAnomalyBaselines, TraditionalConfig
from utils.logger import get_logger, setup_logging

_log = get_logger(__name__)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


def _resolve_path(path_like: str | Path | None, *, default: Path) -> Path:
    if path_like is None:
        return default
    p = Path(path_like)
    return p if p.is_absolute() else (ROOT / p)


def _subset_if_needed(ds: Any, max_n: int | None) -> Any:
    if max_n is None:
        return ds
    n = int(max_n)
    if n > 0 and len(ds) > n:
        return Subset(ds, list(range(n)))
    return ds


def _load_labels(path: Path | None) -> dict[str, np.ndarray]:
    if path is None or (not path.is_file()):
        return {}
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        return {}
    out: dict[str, np.ndarray] = {}
    for k, vals in obj.items():
        if not isinstance(vals, list):
            continue
        arr = np.asarray(vals, dtype=np.int64)
        uniq = set(np.unique(arr).tolist())
        if uniq.issubset({0, 1}):
            out[str(k)] = arr
    return out


def _safe_stats(x: np.ndarray, m: np.ndarray) -> tuple[float, float, float, float]:
    v = x[m > 0.5]
    if v.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    return float(np.mean(v)), float(np.std(v)), float(np.quantile(v, 0.95)), float(np.mean(m > 0.5))


def _sample_features(sample: dict[str, Any]) -> np.ndarray:
    ox = sample["oper_x"].numpy()
    wx = sample["wave_x"].numpy()
    om = sample["oper_valid"].numpy()
    wm = sample["wave_valid"].numpy()

    feats: list[float] = []
    for c in range(ox.shape[0]):
        feats.extend(_safe_stats(ox[c], om[c]))
    for c in range(wx.shape[0]):
        feats.extend(_safe_stats(wx[c], wm[c]))
    return np.asarray(feats, dtype=np.float64)


def _build_dataset(split: str, processed_dir: Path, manifest: Path, norm_stats: Path, lru: int) -> AnomalyFrameDataset:
    return AnomalyFrameDataset(
        processed_anomaly_dir=processed_dir,
        split=split,
        manifest_path=manifest,
        norm_stats_path=norm_stats if norm_stats.is_file() else None,
        root=ROOT,
        open_file_lru_size=lru,
    )


def _extract_features(ds: Any) -> tuple[np.ndarray, list[int]]:
    n = len(ds)
    x = np.zeros((n, 24), dtype=np.float64)
    ts: list[int] = []
    for i in range(n):
        s = ds[i]
        x[i] = _sample_features(s)
        t = s.get("timestamp", -1)
        try:
            ts.append(int(t))
        except Exception:
            ts.append(-1)
    return x, ts


def _threshold_from_train(scores: np.ndarray, q: float) -> float:
    if scores.size == 0:
        return 0.0
    qq = float(np.clip(q, 0.5, 0.999))
    return float(np.quantile(scores, qq))


def _eval_split(scores: np.ndarray, threshold: float, labels: np.ndarray | None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "error_summary": summarize_errors(scores),
        "detection": build_detection_report(scores, threshold),
    }
    if labels is not None and labels.shape[0] == scores.shape[0]:
        m = evaluate_with_labels(scores, labels, threshold)
        m["auc"] = roc_auc_from_scores(labels, scores)
        out["labeled_metrics"] = m
    elif labels is not None:
        out["label_alignment_error"] = {"labels": int(labels.shape[0]), "scores": int(scores.shape[0])}
    return out


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if path.is_file():
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            return obj
    return {}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compare anomaly methods with unified report")
    ap.add_argument("--data-config", default="configs/data_config.yaml")
    ap.add_argument("--train-config", default="configs/anomaly_detection/train.yaml")
    ap.add_argument("--processed-dir", default=None)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--norm-stats", default=None)
    ap.add_argument("--labels-json", default=None)
    ap.add_argument("--open-file-lru-size", type=int, default=None)
    ap.add_argument("--max-train-samples", type=int, default=None)
    ap.add_argument("--max-val-samples", type=int, default=None)
    ap.add_argument("--max-test-samples", type=int, default=None)
    ap.add_argument("--threshold-quantile", type=float, default=0.95)
    ap.add_argument("--pca-components", type=int, default=12)
    ap.add_argument("--iforest-contamination", type=float, default=0.05)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--main-summary", default="outputs/anomaly_detection/summary.json")
    ap.add_argument("--main-split-reports", default="outputs/anomaly_detection/split_reports.json")
    ap.add_argument("--baseline-summary", default="outputs/baseline/anomaly_detection/summary.json")
    ap.add_argument("--baseline-split-reports", default="outputs/baseline/anomaly_detection/split_reports.json")
    ap.add_argument("--output-dir", default="outputs/baseline/anomaly_detection_traditional")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(log_file=ROOT / "outputs/logs/anomaly_compare_methods.log")

    data_cfg = _load_yaml(_resolve_path(args.data_config, default=ROOT / "configs/data_config.yaml"))
    train_cfg = _load_yaml(_resolve_path(args.train_config, default=ROOT / "configs/anomaly_detection/train.yaml"))

    processed_dir = _resolve_path(
        args.processed_dir or train_cfg.get("processed_dir") or data_cfg.get("paths", {}).get("processed", {}).get("anomaly"),
        default=ROOT / "data/processed/anomaly_detection",
    )
    manifest = _resolve_path(
        args.manifest or train_cfg.get("manifest_path") or data_cfg.get("artifacts", {}).get("split_manifests", {}).get("anomaly_detection"),
        default=ROOT / "data/processed/splits/anomaly_detection.json",
    )
    norm_stats = _resolve_path(
        args.norm_stats or train_cfg.get("norm_stats_path") or data_cfg.get("artifacts", {}).get("normalization_files", {}).get("anomaly_detection"),
        default=ROOT / "data/processed/normalization/anomaly_detection_norm.json",
    )
    lru = int(args.open_file_lru_size if args.open_file_lru_size is not None else train_cfg.get("open_file_lru_size", 6))

    labels = _load_labels(_resolve_path(args.labels_json, default=ROOT / "outputs/anomaly_detection/labels.json") if args.labels_json else None)

    train_ds = _subset_if_needed(_build_dataset("train", processed_dir, manifest, norm_stats, lru), args.max_train_samples)
    val_ds = _subset_if_needed(_build_dataset("val", processed_dir, manifest, norm_stats, lru), args.max_val_samples)
    test_ds = _subset_if_needed(_build_dataset("test", processed_dir, manifest, norm_stats, lru), args.max_test_samples)

    _log.info("Extracting tabular features: train=%d val=%d test=%d", len(train_ds), len(val_ds), len(test_ds))
    x_train, _ = _extract_features(train_ds)
    x_val, _ = _extract_features(val_ds)
    x_test, _ = _extract_features(test_ds)

    trad = TraditionalAnomalyBaselines(
        TraditionalConfig(
            pca_components=args.pca_components,
            iforest_contamination=args.iforest_contamination,
            random_state=args.random_state,
        )
    )
    fit_meta = trad.fit(x_train)

    pca_train = trad.pca_scores(x_train)
    pca_val = trad.pca_scores(x_val)
    pca_test = trad.pca_scores(x_test)

    if_train = trad.iforest_scores(x_train)
    if_val = trad.iforest_scores(x_val)
    if_test = trad.iforest_scores(x_test)

    q = float(np.clip(args.threshold_quantile, 0.5, 0.999))
    pca_thr = _threshold_from_train(pca_train, q)
    if_thr = _threshold_from_train(if_train, q)

    split_reports = {
        "pca": {
            "train": _eval_split(pca_train, pca_thr, labels.get("train")),
            "val": _eval_split(pca_val, pca_thr, labels.get("val")),
            "test": _eval_split(pca_test, pca_thr, labels.get("test")),
            "threshold": pca_thr,
        },
        "iforest": {
            "train": _eval_split(if_train, if_thr, labels.get("train")),
            "val": _eval_split(if_val, if_thr, labels.get("val")),
            "test": _eval_split(if_test, if_thr, labels.get("test")),
            "threshold": if_thr,
        },
    }

    out_dir = _resolve_path(args.output_dir, default=ROOT / "outputs/baseline/anomaly_detection_traditional")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "fit_meta": fit_meta,
        "threshold_quantile": q,
        "methods": {
            "pca": {
                "threshold": pca_thr,
                "val_anomaly_ratio": split_reports["pca"]["val"]["detection"]["anomaly_ratio"],
                "test_anomaly_ratio": split_reports["pca"]["test"]["detection"]["anomaly_ratio"],
            },
            "iforest": {
                "threshold": if_thr,
                "val_anomaly_ratio": split_reports["iforest"]["val"]["detection"]["anomaly_ratio"],
                "test_anomaly_ratio": split_reports["iforest"]["test"]["detection"]["anomaly_ratio"],
            },
        },
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "split_reports.json").write_text(json.dumps(split_reports, ensure_ascii=False, indent=2), encoding="utf-8")

    main_summary = _read_json_if_exists(_resolve_path(args.main_summary, default=ROOT / "outputs/anomaly_detection/summary.json"))
    main_reports = _read_json_if_exists(_resolve_path(args.main_split_reports, default=ROOT / "outputs/anomaly_detection/split_reports.json"))
    base_summary = _read_json_if_exists(_resolve_path(args.baseline_summary, default=ROOT / "outputs/baseline/anomaly_detection/summary.json"))
    base_reports = _read_json_if_exists(_resolve_path(args.baseline_split_reports, default=ROOT / "outputs/baseline/anomaly_detection/split_reports.json"))

    def _metrics_block(rep: dict[str, Any], split: str) -> dict[str, Any]:
        x = rep.get(split, {}) if isinstance(rep, dict) else {}
        out = {
            "anomaly_ratio": x.get("detection", {}).get("anomaly_ratio"),
            "num_anomaly": x.get("detection", {}).get("num_anomaly"),
            "count": x.get("error_summary", {}).get("count"),
        }
        if isinstance(x.get("labeled_metrics"), dict):
            out["labeled_metrics"] = x["labeled_metrics"]
        return out

    comparison = {
        "methods": {
            "main_ae": {
                "summary": {
                    "best_val_loss": main_summary.get("best_val_loss"),
                    "best_threshold": main_summary.get("best_threshold"),
                },
                "val": _metrics_block(main_reports, "val"),
                "test": _metrics_block(main_reports, "test"),
            },
            "ae_baseline": {
                "summary": {
                    "best_val_loss": base_summary.get("best_val_loss"),
                    "best_threshold": base_summary.get("best_threshold"),
                },
                "val": _metrics_block(base_reports, "val"),
                "test": _metrics_block(base_reports, "test"),
            },
            "pca": {
                "summary": {
                    "threshold": pca_thr,
                },
                "val": _metrics_block(split_reports["pca"], "val"),
                "test": _metrics_block(split_reports["pca"], "test"),
            },
            "iforest": {
                "summary": {
                    "threshold": if_thr,
                },
                "val": _metrics_block(split_reports["iforest"], "val"),
                "test": _metrics_block(split_reports["iforest"], "test"),
            },
        },
        "notes": [
            "If labeled_metrics are missing, labels are unavailable or not binary/aligned.",
            "Without labels, anomaly_ratio/num_anomaly are unsupervised diagnostics only.",
        ],
    }

    comp_path = out_dir / "comparison_report.json"
    comp_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")

    final_dir = ROOT / "outputs/final_results/anomaly_detection"
    final_dir.mkdir(parents=True, exist_ok=True)
    (final_dir / "anomaly_methods_comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    _log.info("Traditional method results: %s", out_dir)
    _log.info("Unified comparison report: %s", comp_path)


if __name__ == "__main__":
    main()
