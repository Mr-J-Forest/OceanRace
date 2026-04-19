import numpy as np

def calculate_warnings(pred_numpy, mask_numpy, var_names, thresholds, lons, lats):
    """
    计算预报时段内的要素异常预警。
    识别骤升、骤降、突变等异常事件。
    """
    num_steps, num_vars, h, w = pred_numpy.shape
    events = []
    
    # 构造一个形状为 (num_vars, h, w) 的警告级别矩阵：0: 无, 1: 轻度, 2: 中度, 3: 重度
    warning_levels = np.zeros((num_vars, h, w), dtype=int)
    
    for i, var_name in enumerate(var_names):
        thresh = thresholds.get(var_name.upper())
        if thresh is None:
            continue
            
        # 提取当前变量时间序列
        data = pred_numpy[:, i, :, :]  # (T, h, w)
        
        # 提取空间掩膜（取第一个时间步的掩膜）
        # 兼容 mask_numpy 的不同形状
        if mask_numpy.ndim == 4:
            mask_2d = mask_numpy[0, min(i, mask_numpy.shape[1]-1), :, :]
        elif mask_numpy.ndim == 3:
            mask_2d = mask_numpy[0, :, :]
        elif mask_numpy.ndim == 2:
            mask_2d = mask_numpy
        else:
            mask_2d = np.ones((h, w))
            
        valid_mask = (mask_2d >= 0.5)
        
        # 计算时间维度的最大值、最小值、初始值
        data_max = np.nanmax(data, axis=0)
        data_min = np.nanmin(data, axis=0)
        data_initial = data[0, :, :]
        
        # 计算变化幅度
        amplitude = data_max - data_min
        
        # 判断阈值 (轻度 1x, 中度 1.5x, 重度 2x)
        level_1 = thresh
        level_2 = thresh * 1.5
        level_3 = thresh * 2.0
        
        # 找到异常区域
        is_warn = (amplitude >= level_1) & valid_mask
        
        # 填充 warning_levels
        warning_levels[i][(amplitude >= level_1) & valid_mask] = 1
        warning_levels[i][(amplitude >= level_2) & valid_mask] = 2
        warning_levels[i][(amplitude >= level_3) & valid_mask] = 3
        
        # 为了避免事件列表过多，我们通过连通域或者简单的网格降采样聚合事件
        # 这里为了简化，我们按一定步长(例如每隔几个格点)查找极值点作为代表事件
        # 或者直接找全局最严重的几个点
        warn_indices = np.where(is_warn)
        
        # 收集所有的异常点，按幅度降序排序，取前 N 个独立的事件
        # 为了不重叠太近的点，可以加上简单的距离过滤
        event_candidates = []
        for y, x in zip(warn_indices[0], warn_indices[1]):
            amp = amplitude[y, x]
            event_candidates.append({
                "y": int(y), "x": int(x), "amp": float(amp),
                "max_v": float(data_max[y, x]), "min_v": float(data_min[y, x]),
                "init_v": float(data_initial[y, x])
            })
            
        event_candidates.sort(key=lambda e: e["amp"], reverse=True)
        
        # 过滤距离太近的点 (例如距离平方和 < 100)
        filtered_events = []
        for ev in event_candidates:
            too_close = False
            for fev in filtered_events:
                if (ev["y"] - fev["y"])**2 + (ev["x"] - fev["x"])**2 < 100:
                    too_close = True
                    break
            if not too_close:
                filtered_events.append(ev)
            if len(filtered_events) >= 10:  # 每个变量最多10个代表性事件
                break
                
        # 转换为最终格式
        for ev in filtered_events:
            y, x = ev["y"], ev["x"]
            
            lon = float(lons[x]) if len(lons) > x else 0.0
            lat = float(lats[y]) if len(lats) > y else 0.0
            
            # 判断类型：骤升 vs 骤降
            # 找到极值出现的时间索引
            time_series = data[:, y, x]
            t_max = np.nanargmax(time_series)
            t_min = np.nanargmin(time_series)
            
            if t_max > t_min:
                event_type = f"{var_name}骤升"
                desc = f"从 {ev['min_v']:.2f} 上升至 {ev['max_v']:.2f}"
            else:
                event_type = f"{var_name}骤降"
                desc = f"从 {ev['max_v']:.2f} 下降至 {ev['min_v']:.2f}"
                
            if var_name.upper() in ["SSU", "SSV", "SSUV"]:
                event_type = f"{var_name}突变"
                
            amp = ev["amp"]
            lvl = "重度" if amp >= level_3 else ("中度" if amp >= level_2 else "轻度")
            
            events.append({
                "var": var_name,
                "type": event_type,
                "level": lvl,
                "lon": round(lon, 2),
                "lat": round(lat, 2),
                "grid_y": y,
                "grid_x": x,
                "amplitude": round(amp, 2),
                "desc": desc,
                "suggestion": get_suggestion(var_name.upper(), lvl)
            })

    # 将 warning_levels 转换为 JSON 可序列化的列表，处理 NaN
    # warning_levels 里没有 NaN，但是以防万一
    def grid_to_json(g):
        import math
        return [[(None if math.isnan(v) else float(v)) for v in row] for row in g]

    masks_dict = {}
    for i, var_name in enumerate(var_names):
        wl = warning_levels[i].astype(float)
        
        if mask_numpy.ndim == 4:
            mask_2d = mask_numpy[0, min(i, mask_numpy.shape[1]-1), :, :]
        elif mask_numpy.ndim == 3:
            mask_2d = mask_numpy[0, :, :]
        elif mask_numpy.ndim == 2:
            mask_2d = mask_numpy
        else:
            mask_2d = np.ones((h, w))
            
        wl[mask_2d < 0.5] = np.nan
        
        masks_dict[var_name] = grid_to_json(wl)

    return {
        "events": events,
        "masks": masks_dict
    }


def get_suggestion(var_name, level):
    if var_name == "SST":
        return "建议关注周边海域生态环境变化，防范极端水温对养殖业的影响；发布局地水温异常通报。"
    elif var_name == "SSS":
        return "可能由于强降水或洋流平流引起，建议关注盐度剧变对特定海洋生物的影响。"
    elif var_name in ["SSU", "SSV", "SSUV"]:
        return "流速突变可能影响航运安全与海上作业，建议提醒相关海域过往船只注意避险。"
    return "加强动态监测，视情况启动应急预案。"


def calculate_correlation(pred_numpy, mask_numpy, var_names, var_name_1, var_name_2):
    """
    计算两个变量在预报时间窗口内的 Pearson 相关系数矩阵
    """
    try:
        idx1 = [v.upper() for v in var_names].index(var_name_1.upper())
        idx2 = [v.upper() for v in var_names].index(var_name_2.upper())
    except ValueError:
        return None
        
    a = pred_numpy[:, idx1, :, :]
    b = pred_numpy[:, idx2, :, :]
    
    a_mean = np.nanmean(a, axis=0)
    b_mean = np.nanmean(b, axis=0)
    a_dev = a - a_mean
    b_dev = b - b_mean
    
    num = np.nansum(a_dev * b_dev, axis=0)
    den = np.sqrt(np.nansum(a_dev**2, axis=0) * np.nansum(b_dev**2, axis=0))
    
    with np.errstate(divide='ignore', invalid='ignore'):
        corr = num / den
        
    # 处理掩膜
    if mask_numpy.ndim >= 3:
        mask_2d = mask_numpy[0, 0, :, :] if mask_numpy.ndim == 4 else mask_numpy[0, :, :]
        corr[mask_2d < 0.5] = np.nan
        
    corr[np.isinf(corr)] = np.nan
    
    # 计算一些全局统计或敏感性指标
    valid_corr = corr[~np.isnan(corr)]
    if len(valid_corr) > 0:
        mean_corr = float(np.mean(valid_corr))
        mean_abs_corr = float(np.mean(np.abs(valid_corr)))
        high_corr_ratio = float(np.sum(np.abs(valid_corr) > 0.7) / len(valid_corr))
    else:
        mean_corr = 0.0
        mean_abs_corr = 0.0
        high_corr_ratio = 0.0
        
    if high_corr_ratio > 0.4:
        level_text = "极其显著"
    elif high_corr_ratio > 0.2:
        level_text = "较为明显"
    else:
        level_text = "一般或局部"

    if mean_corr > 0.3:
        direction_text = "协同(正相关)主导"
    elif mean_corr < -0.3:
        direction_text = "制约(负相关)主导"
    else:
        direction_text = "协同与制约交织"

    sensitivity_text = (
        f"在预测时段内，{var_name_1} 与 {var_name_2} 的全局平均相关系数为 {mean_corr:.2f}，"
        f"而平均绝对关联强度达到了 {mean_abs_corr:.2f}。有 {(high_corr_ratio*100):.1f}% 的区域表现出高度的时空关联(绝对值>0.7)。"
        f"总体来看，这两个要素在演变过程中具有{level_text}的{direction_text}效应。"
    )

    def grid_to_json(g):
        import math
        return [[(None if math.isnan(v) else float(v)) for v in row] for row in g]
        
    return {
        "var1": var_name_1,
        "var2": var_name_2,
        "correlation_grid": grid_to_json(corr),
        "mean_corr": round(mean_corr, 2),
        "mean_abs_corr": round(mean_abs_corr, 2),
        "high_corr_ratio": round(high_corr_ratio, 3),
        "analysis_text": sensitivity_text
    }
