"""导出 META4 对象标签文件的变量说明表（CSV + Markdown）。"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import netCDF4 as nc4

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


CH_DESC = {
    "time": "时间轴（天），基准通常为 1950-01-01",
    "processed": "每个时间片是否处理完成（1=完成）",
    "obs_count": "每个时间片检测到的对象数量",
    "time_index": "每个对象对应的时间索引（指向 time 维）",
    "polarity": "对象极性（1=cyclonic, 2=anticyclonic）",
    "center_longitude": "对象中心经度",
    "center_latitude": "对象中心纬度",
    "amplitude": "振幅（极值与外边界高度差）",
    "speed_radius": "速度半径（最大环向平均速度轮廓拟合半径）",
    "effective_radius": "有效半径（有效轮廓拟合半径）",
    "speed_average": "速度轮廓平均地转速度",
    "shape_error_speed": "速度轮廓圆拟合形状误差",
    "shape_error_effective": "有效轮廓圆拟合形状误差",
    "effective_contour_longitude": "有效轮廓经度采样点（obs,NbSample）",
    "effective_contour_latitude": "有效轮廓纬度采样点（obs,NbSample）",
    "speed_contour_longitude": "速度轮廓经度采样点（obs,NbSample）",
    "speed_contour_latitude": "速度轮廓纬度采样点（obs,NbSample）",
}


def _attr_to_text(var: nc4.Variable) -> str:
    attrs = []
    for a in var.ncattrs():
        try:
            v = getattr(var, a)
        except Exception:
            continue
        attrs.append(f"{a}={v}")
    return " | ".join(attrs)


def main() -> None:
    ap = argparse.ArgumentParser(description="导出对象级标签变量说明")
    ap.add_argument(
        "--input-nc",
        type=Path,
        default=ROOT / "data/processed/eddy_detection/labels/19930101_20241231_objects_meta4.nc",
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=ROOT / "docs/meta4_objects_schema.csv",
    )
    ap.add_argument(
        "--out-md",
        type=Path,
        default=ROOT / "docs/meta4_objects_schema.md",
    )
    args = ap.parse_args()

    if not args.input_nc.is_file():
        raise FileNotFoundError(f"input not found: {args.input_nc}")

    rows = []
    with nc4.Dataset(args.input_nc, "r") as ds:
        dims = {k: len(v) for k, v in ds.dimensions.items()}
        for name, var in ds.variables.items():
            rows.append(
                {
                    "name": name,
                    "dtype": str(var.dtype),
                    "dims": "(" + ",".join(var.dimensions) + ")",
                    "shape": "(" + ",".join(str(dims[d]) for d in var.dimensions) + ")",
                    "attrs": _attr_to_text(var),
                    "description_zh": CH_DESC.get(name, ""),
                }
            )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    csv_lines = ["name,dtype,dims,shape,description_zh,attrs"]
    for r in rows:
        vals = [
            r["name"],
            r["dtype"],
            r["dims"],
            r["shape"],
            r["description_zh"],
            r["attrs"],
        ]
        vals = ["\"" + str(v).replace('"', '""') + "\"" for v in vals]
        csv_lines.append(",".join(vals))
    args.out_csv.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    md = []
    md.append("# META4 对象标签变量说明")
    md.append("")
    md.append(f"- 文件: {args.input_nc}")
    md.append("")
    md.append("| 变量 | 类型 | 维度 | 形状 | 中文说明 |")
    md.append("|---|---|---|---|---|")
    for r in rows:
        md.append(
            f"| {r['name']} | {r['dtype']} | {r['dims']} | {r['shape']} | {r['description_zh']} |"
        )

    md.append("")
    md.append("## 备注")
    md.append("")
    md.append("- 详细属性（units、calendar、_FillValue 等）请查看 CSV 的 `attrs` 列。")
    md.append("- `polarity`: 1=cyclonic, 2=anticyclonic。")

    args.out_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"written: {args.out_csv}")
    print(f"written: {args.out_md}")


if __name__ == "__main__":
    main()
