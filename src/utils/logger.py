"""
统一日志入口 + 与 tqdm 进度条协同。

程序入口尽早调用一次 :func:`setup_logging`；其余模块使用 :func:`get_logger(__name__)`。

使用进度条时，在可能同时打日志的循环外使用 :func:`tqdm_logging` 上下文，避免日志冲乱进度条::

    from utils.logger import get_logger, tqdm, tqdm_logging

    logger = get_logger(__name__)
    with tqdm_logging():
        for x in tqdm(items, desc="处理"):
            logger.info("item %s", x)

环境变量 ``OCEAN_LOG_LEVEL``：``DEBUG`` / ``INFO`` / ``WARNING`` / ``ERROR``（未设置时默认 INFO）。
"""
from __future__ import annotations

import logging
import os
import sys
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

__all__ = [
    "setup_logging",
    "get_logger",
    "reset_logging",
    "tqdm",
    "tqdm_logging",
]

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"

_configured: bool = False


class _ConsoleColorFormatter(logging.Formatter):
    """内部类：控制台彩色日志格式化器，利用 ANSI 转义码提升终端可读性"""
    DIM = "\033[2m"
    RESET = "\033[0m"
    
    COLORS = {
        logging.DEBUG: "\033[36m",       # 青色
        logging.INFO: "\033[32m",        # 绿色
        logging.WARNING: "\033[33m",     # 黄色
        logging.ERROR: "\033[31m",       # 红色
        logging.CRITICAL: "\033[1;31m",  # 粗体红色
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelno, self.RESET)
        
        # 视觉层级优化：精简终端输出，去掉模块名，缩短冗余时间格式，高亮级别
        fmt = (
            f"{self.DIM}%(asctime)s{self.RESET} "
            f"{color}[%(levelname)s]{self.RESET} "
            f"%(message)s"
        )
        formatter = logging.Formatter(fmt, datefmt="%H:%M:%S")
        return formatter.format(record)


def tqdm_logging() -> AbstractContextManager[Any]:
    """
    与 ``tqdm`` 同时使用时包一层，使 ``logging`` 输出重定向，不破坏进度条。

    在 ``with tqdm_logging():`` 块内使用 ``tqdm(...)`` 与 ``logger.info`` 等。
    """
    return logging_redirect_tqdm()


def _parse_level(level: int | str | None) -> int:
    if level is None:
        env = os.environ.get("OCEAN_LOG_LEVEL", "INFO").strip().upper()
        return getattr(logging, env, logging.INFO)
    if isinstance(level, int):
        return level
    s = str(level).strip().upper()
    return getattr(logging, s, logging.INFO)


def setup_logging(
    level: int | str | None = None,
    *,
    log_file: Path | str | None = None,
    force: bool = False,
) -> None:
    """
    配置根 logger：控制台（stderr）+ 可选文件。

    - 默认只配置一次；``force=True`` 时清空已有 handler 再配（测试或二次指定文件时可用）。
    - 建议在 ``if __name__ == "__main__"`` 内、其它业务逻辑之前调用。
    """
    global _configured

    root = logging.getLogger()
    if _configured and not force:
        return

    if force:
        for h in root.handlers[:]:
            root.removeHandler(h)
            h.close()

    lvl = _parse_level(level)
    root.setLevel(lvl)

    # 1. 配置控制台 Handler (彩色, 弱化非核心信息)
    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(lvl)
    sh.setFormatter(_ConsoleColorFormatter())
    root.addHandler(sh)

    # 2. 配置可选的文件 Handler (纯文本，防止 ANSI 乱码)
    if log_file is not None:
        p = Path(log_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(p, encoding="utf-8")
        fh.setLevel(lvl)
        fh.setFormatter(logging.Formatter(_DEFAULT_FORMAT, datefmt=_DATE_FMT))
        root.addHandler(fh)

    _configured = True


def get_logger(name: str | None = None) -> logging.Logger:
    """
    返回命名 logger。若尚未调用过 :func:`setup_logging`，将自动以默认级别安装控制台 handler。
    """
    if not _configured:
        setup_logging()
    return logging.getLogger(name if name is not None else "ocean")


def reset_logging() -> None:
    """清空根 handler 与内部状态，仅用于测试。"""
    global _configured
    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)
        h.close()
    root.setLevel(logging.WARNING)
    _configured = False