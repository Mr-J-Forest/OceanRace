"""
数据预处理入口：raw → 清洗落盘；可选 train/val/test 划分与训练集标准化参数。

在项目根目录执行:
  python scripts/02_preprocess.py --task all
  python scripts/02_preprocess.py --task eddy --limit 2 -j 4
  python scripts/02_preprocess.py --task element --steps clean,split,stats
  python scripts/02_preprocess.py --task all --steps split,stats
  python scripts/02_preprocess.py --stage split,stats
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data_preprocessing.cleaner import load_config  # noqa: E402
from data_preprocessing.config_sync import merge_pipeline_artifacts_into_config  # noqa: E402
from data_preprocessing.merger import run_merge_for_task  # noqa: E402
from data_preprocessing.preprocess_workers import (  # noqa: E402
    clean_anomaly_year_one,
    clean_eddy_one,
    clean_element_one,
)
from data_preprocessing.splitter import (  # noqa: E402
    TASK_ANOMALY,
    TASK_EDDY,
    TASK_ELEMENT,
    run_split_for_task,
    run_standardization_for_task,
)
from data_preprocessing.validator import validate_manifest_and_samples  # noqa: E402
from utils.logger import get_logger, setup_logging, tqdm, tqdm_logging  # noqa: E402

_log = get_logger(__name__)


def run_eddy(cfg: dict, limit: int | None, workers: int) -> None:
    raw = ROOT / cfg["paths"]["raw"]["eddy"]
    out_dir = ROOT / cfg["paths"]["processed"]["eddy"]
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(raw.glob("*.nc"))
    if limit is not None:
        files = files[:limit]
    comp = int(cfg.get("output", {}).get("complevel", 4))
    root_s = str(ROOT)
    jobs = [str(p) for p in files]
    with tqdm_logging():
        if workers <= 1:
            for inp in tqdm(jobs, desc="eddy 清洗", unit="file"):
                _log.info("%s", clean_eddy_one(inp, root_s, cfg, comp))
        else:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(clean_eddy_one, inp, root_s, cfg, comp) for inp in jobs]
                for fut in tqdm(
                    as_completed(futs),
                    total=len(futs),
                    desc="eddy 清洗",
                    unit="file",
                ):
                    _log.info("%s", fut.result())


def run_element(cfg: dict, limit: int | None, workers: int) -> None:
    raw = ROOT / cfg["paths"]["raw"]["element_forecasting"]
    out_dir = ROOT / cfg["paths"]["processed"]["element_forecasting"]
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(raw.glob("*.nc"))
    if limit is not None:
        files = files[:limit]
    comp = int(cfg.get("output", {}).get("complevel", 4))
    root_s = str(ROOT)
    jobs = [str(p) for p in files]
    with tqdm_logging():
        if workers <= 1:
            for inp in tqdm(jobs, desc="element 清洗", unit="file"):
                _log.info("%s", clean_element_one(inp, root_s, cfg, comp))
        else:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(clean_element_one, inp, root_s, cfg, comp) for inp in jobs]
                for fut in tqdm(
                    as_completed(futs),
                    total=len(futs),
                    desc="element 清洗",
                    unit="file",
                ):
                    _log.info("%s", fut.result())


def run_anomaly(cfg: dict, limit: int | None, workers: int) -> None:
    raw = ROOT / cfg["paths"]["raw"]["anomaly"]
    out_root = ROOT / cfg["paths"]["processed"]["anomaly"]
    out_root.mkdir(parents=True, exist_ok=True)
    years = sorted([d for d in raw.iterdir() if d.is_dir()])
    if limit is not None:
        years = years[:limit]
    comp = int(cfg.get("output", {}).get("complevel", 4))
    root_s = str(ROOT)
    jobs = [str(d) for d in years]
    with tqdm_logging():
        if workers <= 1:
            for ydir in tqdm(jobs, desc="anomaly 清洗", unit="year"):
                for line in clean_anomaly_year_one(ydir, root_s, cfg, comp):
                    _log.info("%s", line)
        else:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(clean_anomaly_year_one, yd, root_s, cfg, comp) for yd in jobs]
                for fut in tqdm(
                    as_completed(futs),
                    total=len(futs),
                    desc="anomaly 清洗",
                    unit="year",
                ):
                    for line in fut.result():
                        _log.info("%s", line)


def _steps_set(raw: str) -> set[str]:
    parts = {p.strip().lower() for p in raw.split(",") if p.strip()}
    allowed = {"clean", "split", "stats", "merge", "all", "validate"}
    bad = parts - allowed
    if bad:
        raise SystemExit(f"unknown --steps entries: {bad}; allowed: {allowed}")
    if "all" in parts:
        return {"clean", "split", "stats"}
    return parts


def main() -> None:
    ap = argparse.ArgumentParser(description="清洗 / 划分 / 训练集标准化参数 → data/processed")
    ap.add_argument("--config", type=Path, default=ROOT / "configs/data_config.yaml")
    ap.add_argument("--task", choices=("all", "eddy", "element", "anomaly"), default="all")
    ap.add_argument("--limit", type=int, default=None, help="每任务最多处理文件数（调试用）")
    ap.add_argument(
        "-j",
        "--workers",
        type=int,
        default=1,
        help="清洗阶段并行进程数（1=串行；建议不超过 CPU 核数，大文件注意内存）",
    )
    ap.add_argument(
        "--steps",
        "--stage",
        dest="steps",
        type=str,
        default="clean",
        help="逗号分隔: clean, split, stats, merge, validate, all（默认仅 clean；all=三者全做；merge=按时序合并清洗后的样本；validate=仅校验或与其它步组合）；与 --stage 同义",
    )
    ap.add_argument(
        "--sync-config-only",
        action="store_true",
        help="仅根据已有 splits/*.json 与 normalization/*_norm.json 回写 configs（不跑清洗/划分/统计）",
    )
    ap.add_argument(
        "--validate",
        action="store_true",
        help="在完成本命令其它步骤后执行校验（划分 manifest 路径 + processed 样本抽检）",
    )
    ap.add_argument(
        "--validate-limit",
        type=int,
        default=None,
        dest="validate_limit",
        help="校验时每任务最多检查的样本数；默认不限制（数据量大时建议设小值）",
    )
    args = ap.parse_args()
    setup_logging()
    if args.sync_config_only:
        merge_pipeline_artifacts_into_config(args.config, ROOT)
        return

    steps = _steps_set(args.steps)
    if steps == {"validate"}:
        cfg = load_config(args.config)
        validate_manifest_and_samples(
            cfg,
            ROOT,
            check_splits=True,
            sample_limit=args.validate_limit,
        )
        return

    cfg = load_config(args.config)
    b = cfg.get("batch", {})
    lim_eddy = args.limit if args.limit is not None else b.get("max_files_eddy")
    lim_el = args.limit if args.limit is not None else b.get("max_files_element")
    lim_an = args.limit if args.limit is not None else b.get("max_files_anomaly_years")

    workers = max(1, int(args.workers))

    work_steps = steps - {"validate"}
    run_validate = "validate" in steps or args.validate

    if "clean" in work_steps:
        if args.task in ("all", "eddy"):
            run_eddy(cfg, lim_eddy, workers)
        if args.task in ("all", "element"):
            run_element(cfg, lim_el, workers)
        if args.task in ("all", "anomaly"):
            run_anomaly(cfg, lim_an, workers)

    if "merge" in work_steps:
        tasks: list[str] = ["eddy", "element", "anomaly"] if args.task == "all" else [args.task]
        for tk in tasks:
            task_name = {
                "eddy": TASK_EDDY,
                "element": TASK_ELEMENT,
                "anomaly": TASK_ANOMALY,
            }[tk]
            limit = {
                "eddy": lim_eddy,
                "element": lim_el,
                "anomaly": lim_an,
            }[tk]
            outs = run_merge_for_task(task_name, cfg, ROOT, limit=limit)
            for p in outs:
                _log.info("merged %s -> %s", tk, p.relative_to(ROOT))

    if "split" in work_steps or "stats" in work_steps:
        tasks: list[str] = []
        if args.task == "all":
            tasks = ["eddy", "element", "anomaly"]
        else:
            tasks = [args.task]
        with tqdm_logging():
            for tk in tqdm(tasks, desc="split / stats", unit="task"):
                if "split" in work_steps:
                    if tk == "eddy":
                        run_split_for_task(TASK_EDDY, cfg, ROOT)
                    elif tk == "element":
                        run_split_for_task(TASK_ELEMENT, cfg, ROOT)
                    else:
                        run_split_for_task(TASK_ANOMALY, cfg, ROOT)
                if "stats" in work_steps:
                    if tk == "eddy":
                        run_standardization_for_task(TASK_EDDY, cfg, ROOT)
                    elif tk == "element":
                        run_standardization_for_task(TASK_ELEMENT, cfg, ROOT)
                    else:
                        run_standardization_for_task(TASK_ANOMALY, cfg, ROOT)

    if "stats" in work_steps:
        merge_pipeline_artifacts_into_config(args.config, ROOT)

    if run_validate:
        validate_manifest_and_samples(
            cfg,
            ROOT,
            check_splits=True,
            sample_limit=args.validate_limit,
        )


if __name__ == "__main__":
    main()
