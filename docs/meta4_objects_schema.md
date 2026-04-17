# META4 对象标签变量说明

对象级标签是 META4 识别链路中的中间产物，后续会进一步转换为像素级 mask，并将背景显式修正为 0 供 U-Net 训练。

- 示例文件: data/processed/eddy_detection/labels/19930101_20241231_objects_meta4.nc

| 变量 | 类型 | 维度 | 形状 | 中文说明 |
|---|---|---|---|---|
| time | float32 | (time) | (11688) | 时间轴（天），基准通常为 1950-01-01 |
| processed | uint8 | (time) | (11688) | 每个时间片是否处理完成（1=完成） |
| obs_count | int32 | (time) | (11688) | 每个时间片检测到的对象数量 |
| time_index | int32 | (obs) | (2005768) | 每个对象对应的时间索引（指向 time 维） |
| polarity | uint8 | (obs) | (2005768) | 对象极性（1=cyclonic, 2=anticyclonic） |
| center_longitude | float32 | (obs) | (2005768) | 对象中心经度 |
| center_latitude | float32 | (obs) | (2005768) | 对象中心纬度 |
| amplitude | float32 | (obs) | (2005768) | 振幅（极值与外边界高度差） |
| speed_radius | float32 | (obs) | (2005768) | 速度半径（最大环向平均速度轮廓拟合半径） |
| effective_radius | float32 | (obs) | (2005768) | 有效半径（有效轮廓拟合半径） |
| speed_average | float32 | (obs) | (2005768) | 速度轮廓平均地转速度 |
| shape_error_speed | float32 | (obs) | (2005768) | 速度轮廓圆拟合形状误差 |
| shape_error_effective | float32 | (obs) | (2005768) | 有效轮廓圆拟合形状误差 |
| effective_contour_longitude | float32 | (obs,NbSample) | (2005768,20) | 有效轮廓经度采样点（obs,NbSample） |
| effective_contour_latitude | float32 | (obs,NbSample) | (2005768,20) | 有效轮廓纬度采样点（obs,NbSample） |
| speed_contour_longitude | float32 | (obs,NbSample) | (2005768,20) | 速度轮廓经度采样点（obs,NbSample） |
| speed_contour_latitude | float32 | (obs,NbSample) | (2005768,20) | 速度轮廓纬度采样点（obs,NbSample） |

## 备注

- 详细属性（units、calendar、_FillValue 等）请查看 CSV 的 `attrs` 列。
- `polarity`: 1=cyclonic, 2=anticyclonic。
- 若用于当前仓库的训练链路，请再通过 `02h_fix_meta4_mask_background.py` 和 `02j_objects_to_mask_parallel.py` 转换为训练用 mask。
