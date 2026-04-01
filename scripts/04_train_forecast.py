"""
要素预报训练入口。项目根目录执行::

  python scripts/04_train_forecast.py --epochs 5 --batch-size 2

默认运行主模块 Hybrid 长期预测框架；如需旧 ConvLSTM 基线可加 ``--baseline``。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> None:
  ap = argparse.ArgumentParser(add_help=False)
  ap.add_argument("--baseline", action="store_true", help="使用旧 ConvLSTM 基线训练器")
  args, _ = ap.parse_known_args(sys.argv[1:])

  if args.baseline:
    from baseline.element_forecasting.train import main as baseline_main  # noqa: E402

    baseline_main()
  else:
    from element_forecasting.trainer import main as hybrid_main  # noqa: E402

    hybrid_main()

if __name__ == "__main__":
    main()
