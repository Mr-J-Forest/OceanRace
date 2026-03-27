"""
要素预报（element）基线模型训练启动脚本。参数从 configs/baseline/element_forecasting/train.yaml、model.yaml 读取，无需命令行传参。

项目根目录执行::

  python scripts/run_element_baseline_train.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from baseline.element_forecasting.train import main  # noqa: E402

if __name__ == "__main__":
    main()
