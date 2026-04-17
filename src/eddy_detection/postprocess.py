"""涡旋分割后处理：连通域筛选与边界/中心提取。"""
from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np


def _neighbors(y: int, x: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if y > 0:
        out.append((y - 1, x))
    if y + 1 < h:
        out.append((y + 1, x))
    if x > 0:
        out.append((y, x - 1))
    if x + 1 < w:
        out.append((y, x + 1))
    return out


def remove_small_components(mask: np.ndarray, min_pixels: int) -> np.ndarray:
    """删除小连通域。

    参数
    -----
    mask:
        二值 mask，非零区域视为前景。
    min_pixels:
        小于该像素数的连通域会被清除。
    """
    if min_pixels <= 1:
        return (mask > 0).astype(np.uint8)

    fg = (mask > 0).astype(np.uint8)
    h, w = fg.shape
    visited = np.zeros_like(fg, dtype=np.uint8)
    out = np.zeros_like(fg, dtype=np.uint8)

    for y in range(h):
        for x in range(w):
            if fg[y, x] == 0 or visited[y, x] == 1:
                continue

            q: deque[tuple[int, int]] = deque([(y, x)])
            visited[y, x] = 1
            comp: list[tuple[int, int]] = []
            while q:
                cy, cx = q.popleft()
                comp.append((cy, cx))
                for ny, nx in _neighbors(cy, cx, h, w):
                    if fg[ny, nx] == 0 or visited[ny, nx] == 1:
                        continue
                    visited[ny, nx] = 1
                    q.append((ny, nx))

            if len(comp) >= min_pixels:
                for cy, cx in comp:
                    out[cy, cx] = 1

    return out


def extract_eddy_objects(mask: np.ndarray, class_id: int) -> list[dict[str, Any]]:
    """从单类别 mask 提取连通域对象，返回边界与中心。"""
    fg = (mask == class_id).astype(np.uint8)
    h, w = fg.shape
    visited = np.zeros_like(fg, dtype=np.uint8)
    objects: list[dict[str, Any]] = []

    for y in range(h):
        for x in range(w):
            if fg[y, x] == 0 or visited[y, x] == 1:
                continue

            q: deque[tuple[int, int]] = deque([(y, x)])
            visited[y, x] = 1
            comp: list[tuple[int, int]] = []
            while q:
                cy, cx = q.popleft()
                comp.append((cy, cx))
                for ny, nx in _neighbors(cy, cx, h, w):
                    if fg[ny, nx] == 0 or visited[ny, nx] == 1:
                        continue
                    visited[ny, nx] = 1
                    q.append((ny, nx))

            ys = np.asarray([p[0] for p in comp], dtype=np.int32)
            xs = np.asarray([p[1] for p in comp], dtype=np.int32)
            cy = float(np.mean(ys))
            cx = float(np.mean(xs))

            boundary: list[list[int]] = []
            comp_set = set(comp)
            for py, px in comp:
                is_edge = False
                for ny, nx in _neighbors(py, px, h, w):
                    if (ny, nx) not in comp_set:
                        is_edge = True
                        break
                if is_edge:
                    boundary.append([int(py), int(px)])

            objects.append(
                {
                    "class_id": int(class_id),
                    "area": int(len(comp)),
                    "center_yx": [cy, cx],
                    "bbox_yx": [int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())],
                    "boundary_yx": boundary,
                }
            )

    return objects
