# -*- coding: utf-8 -*-
"""
06_predict_to_gis_v2.py (v2 实例级 NMS 版)
================================================================
第五阶段:YOLO11 实例分割预测 + 矢量化 (第二版)

核心改进:
  1. 实例级 Polygon NMS (IoU>=0.70), 绝不 dissolve/union 相邻农田
  2. 边缘实例检测与优先级排序(非边缘 > 边缘)
  3. 质量属性: compactness, circularity, vertex_count, is_edge
  4. 双 SHP 输出: all(全量实例) + filtered(保守筛选)
  5. 三种调试叠加图: 原始候选 / NMS 后 / filtered 最终
  6. conf=0.35, iou=0.50

严禁: unary_union(all) / dissolve / buffer+union / OpenCV / Hough
边界来源: ultralytics result.masks.xy
================================================================
"""

import os
import sys
import glob
import time
import math
import traceback

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.windows import Window, transform as win_transform_fn
from rasterio.vrt import WarpedVRT
from rasterio.warp import calculate_default_transform
from rasterio.enums import Resampling
from rasterio.crs import CRS
from shapely.geometry import Polygon, MultiPolygon
from shapely.validation import make_valid

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from ultralytics import YOLO
except ImportError:
    print("[错误] 未安装 ultralytics")
    sys.exit(1)

# =====================================================================
# 配置区
# =====================================================================
TIF_DIR    = r"E:\260827YOLORUN\GEE_Export_2025"
MODEL_PATH = r"E:\260827YOLORUN\runs\segment\ordos_farmland_v1\weights\best.pt"
OUT_DIR    = r"E:\260827YOLORUN\predict_results"

OUT_SHP_ALL     = os.path.join(OUT_DIR, "farmlands_2025_v2_all.shp")
OUT_SHP_FILTERED = os.path.join(OUT_DIR, "farmlands_2025_v2_filtered.shp")

TARGET_CRS = CRS.from_epsg(32648)
TARGET_RESOLUTION_METERS = 10.0

TILE_SIZE = 640
STEP      = 512
BANDS_RGB = [3, 2, 1]

CLIP_PCT     = (2, 98)
BLOCK_STAT   = 4096
HIST_BINS    = 2048
VALID_PIX_RATIO_MIN = 0.20

# YOLO 推理参数
CONF_THRESH  = 0.35
IOU_THRESH   = 0.50
DEVICE       = 0
BATCH_SIZE   = 4
RETINA_MASKS = True

# NMS 去重
DEDUP_IOU_THRESH = 0.70      # 实例级 NMS: IoU>=此值认为是重复
EDGE_DEDUP_IOU   = 0.50      # 边缘实例与非边缘实例的宽松去重阈值

# 质量筛选(仅用于 filtered)
CONF_THRESHOLD   = 0.35
MIN_AREA_M2     = 10000
MAX_AREA_M2     = 600000
MIN_COMPACTNESS = 0.35

# 几何修复
SIMPLIFY_TOLERANCE_M = 0.0   # 0=不简化

# 边缘检测
EDGE_PIXEL_THRESHOLD = 10    # mask 距窗口边缘 < 此像素则标记 edge

# 小规模测试
MAX_TIFS = 2
PRINT_PROGRESS_EVERY = 20

# 调试输出
DEBUG_INPUT_DIR  = os.path.join(OUT_DIR, "debug_inputs_v2")
DEBUG_OVERLAY_DIR = os.path.join(OUT_DIR, "debug_overlays_v2")
SAVE_DEBUG_INPUT_MAX = 30


# =====================================================================
# 工具函数(与 v1 一致)
# =====================================================================
def find_tifs(tif_dir):
    pats = [os.path.join(tif_dir, "*." + e) for e in ("tif", "tiff", "TIF", "TIFF")]
    found = []
    for p in pats:
        found.extend(glob.glob(p))
    return sorted(set(found))


def build_vrt(src):
    if src.crs is None:
        raise ValueError("[CRS 错误] TIF 的 CRS 为 None")
    dst_transform, dst_width, dst_height = calculate_default_transform(
        src.crs, TARGET_CRS, src.width, src.height, *src.bounds,
        resolution=TARGET_RESOLUTION_METERS,
    )
    res = TARGET_RESOLUTION_METERS
    dst_transform = rasterio.transform.from_origin(
        dst_transform.c, dst_transform.f, res, res
    )
    assert abs(abs(dst_transform.a) - abs(dst_transform.e)) < 1e-9
    vrt = WarpedVRT(
        src, crs=TARGET_CRS,
        transform=dst_transform,
        width=dst_width, height=dst_height,
        resampling=Resampling.bilinear,
        nodata=0,
    )
    assert vrt.crs == TARGET_CRS
    return vrt


def compute_global_stretch(vrt, bands, block=BLOCK_STAT, bins=HIST_BINS):
    H, W = vrt.height, vrt.width
    n = len(bands)
    range_min = [None] * n
    range_max = [None] * n

    for r in range(0, H, block):
        for c in range(0, W, block):
            h = min(block, H - r)
            w = min(block, W - c)
            win = Window(c, r, w, h)
            data = vrt.read(bands, window=win, masked=True)
            if data.mask is np.ma.nomask:
                common_valid = np.ones((h, w), dtype=bool)
            else:
                common_valid = ~np.any(data.mask, axis=0)
            if not common_valid.any():
                continue
            for bi in range(n):
                vals = data.data[bi][common_valid]
                if vals.size == 0:
                    continue
                vmin = float(np.min(vals))
                vmax = float(np.max(vals))
                if range_min[bi] is None or vmin < range_min[bi]:
                    range_min[bi] = vmin
                if range_max[bi] is None or vmax > range_max[bi]:
                    range_max[bi] = vmax

    for bi in range(n):
        if range_min[bi] is None:
            raise ValueError(f"波段 {bands[bi]} 无有效像元")

    edges_list = []
    for bi in range(n):
        lo, hi = range_min[bi], range_max[bi]
        if hi - lo < 1e-6:
            hi = lo + 1.0
        edges_list.append(np.linspace(lo, hi, bins + 1))

    hist_acc = [np.zeros(bins, dtype=np.int64) for _ in range(n)]
    for r in range(0, H, block):
        for c in range(0, W, block):
            h = min(block, H - r)
            w = min(block, W - c)
            win = Window(c, r, w, h)
            data = vrt.read(bands, window=win, masked=True)
            if data.mask is np.ma.nomask:
                common_valid = np.ones((h, w), dtype=bool)
            else:
                common_valid = ~np.any(data.mask, axis=0)
            if not common_valid.any():
                continue
            for bi in range(n):
                vals = data.data[bi][common_valid]
                if vals.size == 0:
                    continue
                h_hist, _ = np.histogram(vals, bins=edges_list[bi])
                hist_acc[bi] += h_hist

    stretch = []
    for bi in range(n):
        edges = edges_list[bi]
        hist = hist_acc[bi]
        total = hist.sum()
        if total == 0:
            raise ValueError(f"波段 {bands[bi]} 直方图为空")
        cum = np.cumsum(hist).astype(np.float64) / total
        p_lo_idx = int(np.searchsorted(cum, CLIP_PCT[0] / 100.0))
        p_hi_idx = int(np.searchsorted(cum, CLIP_PCT[1] / 100.0))
        p_lo_idx = min(max(p_lo_idx, 0), bins - 1)
        p_hi_idx = min(max(p_hi_idx, 0), bins - 1)
        if p_hi_idx <= p_lo_idx:
            p_hi_idx = min(p_lo_idx + 1, bins - 1)
        stretch.append((float(edges[p_lo_idx]), float(edges[p_hi_idx])))
    return stretch


def apply_stretch_uint8(data_chw, stretch, valid_mask):
    n, h, w = data_chw.shape
    out = np.zeros((n, h, w), dtype=np.uint8)
    for bi in range(n):
        p_lo, p_hi = stretch[bi]
        if p_hi - p_lo < 1e-6:
            continue
        arr = data_chw[bi].astype(np.float32)
        scaled = (arr - p_lo) / (p_hi - p_lo) * 255.0
        out[bi] = np.where(valid_mask,
                           np.clip(scaled, 0, 255).astype(np.uint8),
                           0)
    return out.transpose(1, 2, 0)


def save_png(arr_hwc_uint8, out_path):
    h, w, _ = arr_hwc_uint8.shape
    with rasterio.open(
        out_path, "w",
        driver="PNG",
        height=h, width=w, count=3, dtype="uint8",
    ) as dst:
        dst.write(arr_hwc_uint8.transpose(2, 0, 1))


# =====================================================================
# 边缘检测:判断 mask 是否接触窗口边缘
# =====================================================================
def is_edge_instance(mask_xy, tile_size=TILE_SIZE, threshold=EDGE_PIXEL_THRESHOLD):
    """
    若 mask bbox 距窗口任一边 < threshold 像素,则标记为边缘实例。
    """
    if len(mask_xy) < 3:
        return True
    xs = mask_xy[:, 0]
    ys = mask_xy[:, 1]
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    if x_min < threshold or x_max > tile_size - threshold:
        return True
    if y_min < threshold or y_max > tile_size - threshold:
        return True
    return False


# =====================================================================
# 几何修复(仅 make_valid + 可选 simplify, 不做形态学操作)
# =====================================================================
def fix_geometry(poly):
    if not poly.is_valid:
        try:
            poly = make_valid(poly)
        except Exception:
            poly = poly.buffer(0)
    if not isinstance(poly, Polygon):
        # make_valid 可能返回 MultiPolygon
        if isinstance(poly, MultiPolygon) and len(poly.geoms) == 1:
            poly = poly.geoms[0]
        elif isinstance(poly, MultiPolygon):
            areas = [g.area for g in poly.geoms]
            poly = poly.geoms[int(np.argmax(areas))]
    if SIMPLIFY_TOLERANCE_M > 0 and isinstance(poly, Polygon) and not poly.is_empty:
        poly = poly.simplify(SIMPLIFY_TOLERANCE_M)
    return poly


# =====================================================================
# 质量属性计算
# =====================================================================
def compute_quality_attrs(poly, confidence, is_edge,
                          source_tif, window_row, window_col,
                          vertex_count):
    area = poly.area
    perim = poly.length
    compactness = 4.0 * math.pi * area / (perim * perim) if perim > 0 else 0.0
    circularity = compactness  # 同一定义
    return {
        "geometry": poly,
        "confidence": round(confidence, 4),
        "area_m2": round(area, 2),
        "perimeter": round(perim, 2),
        "compactness": round(compactness, 4),
        "circularity": round(circularity, 4),
        "vertex_count": vertex_count,
        "is_edge": int(is_edge),
        "source_tif": source_tif,
        "window_row": window_row,
        "window_col": window_col,
        # debug 状态字段
        "status": "kept",        # kept / nms_dropped / filter_excluded
        "drop_reason": "",       # 记录被删除原因
    }


# =====================================================================
# 推理函数
# =====================================================================
def predict_one_tif(vrt, model, stretch_params, tif_name):
    h, w = vrt.height, vrt.width
    r_offsets = list(range(0, h - TILE_SIZE + 1, STEP))
    c_offsets = list(range(0, w - TILE_SIZE + 1, STEP))
    n_total = len(r_offsets) * len(c_offsets)
    print(f"  [滑窗] VRT {w}x{h}, 行偏移 {len(r_offsets)}, 列偏移 {len(c_offsets)}, 总窗口 {n_total}")

    all_candidates = []
    skip_reasons = {
        "low_valid_ratio": 0, "read_error": 0, "size_mismatch": 0,
        "infer_error": 0, "no_masks": 0, "geom_error": 0,
    }
    window_count = 0
    valid_window_count = 0
    debug_saved = 0

    batch_images = []
    batch_windows = []

    def flush_batch():
        nonlocal all_candidates, batch_images, batch_windows
        nonlocal skip_reasons, debug_saved, valid_window_count
        if not batch_images:
            return
        try:
            results = model.predict(
                source=batch_images,
                conf=CONF_THRESH,
                iou=IOU_THRESH,
                imgsz=TILE_SIZE,
                device=DEVICE,
                retina_masks=RETINA_MASKS,
                verbose=False,
            )
            for result, (c_off, r_off), img_hwc in zip(
                results, batch_windows, batch_images
            ):
                n_boxes = 0 if result.boxes is None else len(result.boxes)
                n_masks = 0 if result.masks is None else len(result.masks)
                n_confs = [] if result.boxes is None else result.boxes.conf.tolist()

                if debug_saved < SAVE_DEBUG_INPUT_MAX:
                    win_name = f"w{c_off:05d}_r{r_off:05d}"
                    save_png(img_hwc,
                             os.path.join(DEBUG_INPUT_DIR, f"{tif_name}_{win_name}.png"))
                    debug_saved += 1

                if result.masks is None:
                    skip_reasons["no_masks"] += 1
                    continue

                tile_transform = win_transform_fn(
                    Window(c_off, r_off, TILE_SIZE, TILE_SIZE),
                    vrt.transform
                )

                for mi, poly_px in enumerate(result.masks.xy):
                    if len(poly_px) < 3:
                        continue

                    edge_flag = is_edge_instance(poly_px)

                    utm_coords = [
                        tile_transform * (float(x), float(y))
                        for x, y in poly_px
                    ]
                    # 去闭合点
                    if len(utm_coords) >= 2 and \
                       abs(utm_coords[0][0] - utm_coords[-1][0]) < 1e-6 and \
                       abs(utm_coords[0][1] - utm_coords[-1][1]) < 1e-6:
                        utm_coords = utm_coords[:-1]
                    if len(utm_coords) < 3:
                        continue

                    try:
                        poly = Polygon(utm_coords)
                        poly = fix_geometry(poly)
                        if not isinstance(poly, Polygon) or poly.is_empty:
                            skip_reasons["geom_error"] += 1
                            continue
                        if poly.area < 1.0:
                            skip_reasons["geom_error"] += 1
                            continue

                        conf = float(n_confs[mi]) if mi < len(n_confs) else 0.0
                        vcount = len(poly.exterior.coords) - 1  # 去闭合点

                        attrs = compute_quality_attrs(
                            poly, conf, edge_flag, tif_name,
                            r_off, c_off, vcount
                        )
                        all_candidates.append(attrs)
                    except Exception:
                        skip_reasons["geom_error"] += 1
                        continue

                if n_masks > 0 or n_boxes > 0:
                    print(f"    [推理] win({c_off},{r_off}): "
                          f"boxes={n_boxes}, masks={n_masks}, "
                          f"edge={sum(1 for c in all_candidates[-n_masks:] if c['is_edge'])}")

        except Exception as e:
            print(f"    [推理错误] {e}")
            traceback.print_exc()
            skip_reasons["infer_error"] += 1

        batch_images = []
        batch_windows = []

    for r_off in r_offsets:
        for c_off in c_offsets:
            window_count += 1
            win = Window(c_off, r_off, TILE_SIZE, TILE_SIZE)

            if window_count % PRINT_PROGRESS_EVERY == 0 or window_count == n_total:
                print(f"    窗口 {window_count}/{n_total}  有效窗口 {valid_window_count}  "
                      f"候选 {len(all_candidates)}  跳过 {skip_reasons}")

            try:
                data = vrt.read(BANDS_RGB, window=win, masked=True)
            except Exception:
                skip_reasons["read_error"] += 1
                continue

            if data.shape[1] != TILE_SIZE or data.shape[2] != TILE_SIZE:
                skip_reasons["size_mismatch"] += 1
                continue

            if data.mask is np.ma.nomask:
                valid_mask = np.ones((TILE_SIZE, TILE_SIZE), dtype=bool)
            else:
                valid_mask = ~np.any(data.mask, axis=0)

            if data.dtype.kind == "f":
                valid_mask &= np.all(np.isfinite(data.data), axis=0)

            valid_ratio = float(valid_mask.mean())
            if valid_ratio < VALID_PIX_RATIO_MIN:
                skip_reasons["low_valid_ratio"] += 1
                continue

            valid_window_count += 1
            rgb_uint8 = apply_stretch_uint8(data.data, stretch_params, valid_mask)
            batch_images.append(rgb_uint8)
            batch_windows.append((c_off, r_off))

            if len(batch_images) >= BATCH_SIZE:
                flush_batch()

    flush_batch()

    print(f"  [滑窗完成] {tif_name}:")
    print(f"    总遍历窗口   : {window_count}")
    print(f"    有效窗口     : {valid_window_count}")
    print(f"    跳过原因     : {skip_reasons}")
    print(f"    原始候选数   : {len(all_candidates)}")
    return all_candidates, window_count, valid_window_count, skip_reasons


# =====================================================================
# 实例级 NMS 去重(不 dissolve, 不 union)
# =====================================================================
def nms_dedup(candidates, iou_thresh=DEDUP_IOU_THRESH, edge_iou=EDGE_DEDUP_IOU):
    """
    实例级 Polygon NMS:
    - 按 (非边缘优先, confidence 高优先, 面积大优先) 排序
    - 对每个候选, 与已保留面计算 IoU
    - IoU >= iou_thresh → 丢弃(同一实例重复预测)
    - 若候选是边缘实例, 且与某个非边缘已保留面 IoU >= edge_iou → 丢弃
    - IoU < iou_thresh → 保留(即使相交/相切)
    """
    if not candidates:
        return [], {"nms_dropped": 0, "edge_dropped": 0}

    print(f"  [NMS 去重] 输入 {len(candidates)} 个候选, "
          f"IoU 阈值={iou_thresh}, 边缘 IoU={edge_iou}")
    start = time.time()

    # 排序: 非边缘(0) < 边缘(1); 然后 confidence 降序; 然后面积降序
    sorted_cands = sorted(
        candidates,
        key=lambda c: (c["is_edge"], -c["confidence"], -c["area_m2"])
    )

    kept = []
    nms_dropped = 0
    edge_dropped = 0

    for cand in sorted_cands:
        g = cand["geometry"]
        is_dup = False
        drop_reason = ""

        for k in kept:
            kg = k["geometry"]
            if not g.intersects(kg):
                continue
            inter = g.intersection(kg).area
            union = g.union(kg).area
            if union < 1e-12:
                continue
            iou = inter / union

            if iou >= iou_thresh:
                is_dup = True
                drop_reason = f"nms_iou={iou:.3f}>= {iou_thresh}"
                nms_dropped += 1
                break

            # 边缘实例宽松去重: 若候选是边缘,已保留是非边缘
            if cand["is_edge"] == 1 and k["is_edge"] == 0 and iou >= edge_iou:
                is_dup = True
                drop_reason = f"edge_iou={iou:.3f}>={edge_iou}"
                edge_dropped += 1
                break

        if is_dup:
            cand["status"] = "nms_dropped"
            cand["drop_reason"] = drop_reason
        else:
            cand["status"] = "kept"
            kept.append(cand)

    elapsed = time.time() - start
    print(f"    NMS 前 {len(sorted_cands)} -> NMS 后 {len(kept)} "
          f"(NMS 去除 {nms_dropped}, 边缘去除 {edge_dropped}), 用时 {elapsed:.1f}s")

    return kept, {"nms_dropped": nms_dropped, "edge_dropped": edge_dropped}


# =====================================================================
# 质量筛选(仅用于 filtered, 不改变 all)
# =====================================================================
def apply_quality_filter(poly_dicts,
                         conf_thresh=CONF_THRESHOLD,
                         min_area=MIN_AREA_M2,
                         max_area=MAX_AREA_M2,
                         min_compact=MIN_COMPACTNESS):
    """
    对 NMS 后的实例做保守筛选,生成 filtered 结果。
    统计各种排除原因。
    """
    kept = []
    reasons = {
        "low_conf": 0,
        "too_small": 0,
        "too_large": 0,
        "low_compactness": 0,
    }

    for d in poly_dicts:
        if d["confidence"] < conf_thresh:
            reasons["low_conf"] += 1
            continue
        if d["area_m2"] < min_area:
            reasons["too_small"] += 1
            continue
        if d["area_m2"] > max_area:
            reasons["too_large"] += 1
            continue
        if d["compactness"] < min_compact:
            reasons["low_compactness"] += 1
            continue
        kept.append(d)

    print(f"  [质量筛选] 输入 {len(poly_dicts)} -> 输出 {len(kept)}")
    print(f"    排除原因: {reasons}")
    return kept, reasons


# =====================================================================
# 调试叠加图
# =====================================================================
def save_debug_overlays(tif_name, all_candidates, nms_kept, filtered, img_cache=None):
    """
    为每幅 TIF 生成三种调试叠加图:
    1. 原始候选实例(按窗口来源不同颜色)
    2. NMS 去重后实例
    3. filtered 最终实例
    使用多边形边界,不使用 bbox。
    """
    overlay_dir = os.path.join(DEBUG_OVERLAY_DIR, tif_name.replace(".tif", ""))
    os.makedirs(overlay_dir, exist_ok=True)

    # --- 图1: 原始候选 ---
    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
    colors = plt.cm.tab20(np.linspace(0, 1, min(20, len(all_candidates))))
    for i, c in enumerate(all_candidates):
        g = c["geometry"]
        if g.is_empty:
            continue
        x, y = g.exterior.xy
        ax.plot(x, y, color=colors[i % len(colors)], linewidth=0.5, alpha=0.7)
    ax.set_aspect("equal")
    ax.set_title(f"{tif_name} - Raw candidates ({len(all_candidates)})")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(overlay_dir, "1_raw_candidates.png"), dpi=100)
    plt.close(fig)

    # --- 图2: NMS 后 ---
    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
    for i, c in enumerate(nms_kept):
        g = c["geometry"]
        if g.is_empty:
            continue
        x, y = g.exterior.xy
        color = "blue" if c["is_edge"] == 0 else "orange"
        ax.plot(x, y, color=color, linewidth=0.8, alpha=0.8)
    ax.set_aspect("equal")
    ax.set_title(f"{tif_name} - After NMS ({len(nms_kept)}) "
                 f"[blue=non-edge, orange=edge]")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(overlay_dir, "2_after_nms.png"), dpi=100)
    plt.close(fig)

    # --- 图3: filtered 最终 ---
    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
    for i, c in enumerate(filtered):
        g = c["geometry"]
        if g.is_empty:
            continue
        x, y = g.exterior.xy
        ax.plot(x, y, color="green", linewidth=1.0, alpha=0.8)
        # 标注置信度
        cx, cy = g.centroid.x, g.centroid.y
        ax.text(cx, cy, f"{c['confidence']:.2f}", fontsize=5, ha="center", color="red")
    ax.set_aspect("equal")
    ax.set_title(f"{tif_name} - Filtered ({len(filtered)})")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(overlay_dir, "3_filtered.png"), dpi=100)
    plt.close(fig)

    print(f"    调试叠加图: {overlay_dir}")


# =====================================================================
# 导出 SHP
# =====================================================================
def export_shp(poly_dicts, out_path, include_debug=False):
    cols = {
        "geometry": [d["geometry"] for d in poly_dicts],
        "id": list(range(len(poly_dicts))),
        "confidence": [d["confidence"] for d in poly_dicts],
        "area_m2": [d["area_m2"] for d in poly_dicts],
        "perimeter": [d["perimeter"] for d in poly_dicts],
        "compactnes": [d["compactness"] for d in poly_dicts],
        "circularit": [d["circularity"] for d in poly_dicts],
        "vertex_cnt": [d["vertex_count"] for d in poly_dicts],
        "is_edge": [d["is_edge"] for d in poly_dicts],
        "source_tif": [d["source_tif"] for d in poly_dicts],
        "window_row": [d["window_row"] for d in poly_dicts],
        "window_col": [d["window_col"] for d in poly_dicts],
    }
    if include_debug:
        cols["status"] = [d.get("status", "") for d in poly_dicts]
        cols["drop_reaso"] = [d.get("drop_reason", "") for d in poly_dicts]

    gdf = gpd.GeoDataFrame(cols, crs=TARGET_CRS)
    gdf.to_file(out_path, driver="ESRI Shapefile", encoding="utf-8")
    return gdf


# =====================================================================
# 主流程
# =====================================================================
def main():
    print("=" * 60)
    print("YOLO11 实例分割预测 + 矢量化 (v2 实例级 NMS 版)")
    print("=" * 60)
    print(f"TIF 目录       : {TIF_DIR}")
    print(f"模型权重       : {MODEL_PATH}")
    print(f"输出 SHP (all) : {OUT_SHP_ALL}")
    print(f"输出 SHP (filt): {OUT_SHP_FILTERED}")
    print(f"目标 CRS       : EPSG:32648")
    print(f"像元分辨率     : {TARGET_RESOLUTION_METERS} m")
    print(f"滑窗尺寸       : {TILE_SIZE}x{TILE_SIZE}")
    print(f"滑窗步长       : {STEP}")
    print(f"置信度阈值     : {CONF_THRESH}")
    print(f"IoU 阈值       : {IOU_THRESH}")
    print(f"NMS IoU 阈值   : {DEDUP_IOU_THRESH}")
    print(f"边缘 IoU 阈值  : {EDGE_DEDUP_IOU}")
    print(f"面积筛选       : {MIN_AREA_M2} ~ {MAX_AREA_M2} m2")
    print(f"圆度筛选       : >= {MIN_COMPACTNESS}")
    print(f"MAX_TIFS       : {MAX_TIFS}")
    print()

    if not os.path.isfile(MODEL_PATH):
        print(f"[错误] 模型权重不存在: {MODEL_PATH}")
        sys.exit(1)
    if not os.path.isdir(TIF_DIR):
        print(f"[错误] TIF 目录不存在: {TIF_DIR}")
        sys.exit(1)

    tif_paths = find_tifs(TIF_DIR)
    if not tif_paths:
        print(f"[错误] 在 {TIF_DIR} 找不到任何 .tif 文件")
        sys.exit(1)
    if MAX_TIFS is not None:
        tif_paths = tif_paths[:MAX_TIFS]
    print(f"[1] 待处理 TIF 数量: {len(tif_paths)}")
    for p in tif_paths:
        print(f"    - {os.path.basename(p)}")
    print()

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(DEBUG_INPUT_DIR, exist_ok=True)
    os.makedirs(DEBUG_OVERLAY_DIR, exist_ok=True)

    print(f"[2] 加载 YOLO 模型: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print(f"    model.task = {model.task}")
    assert model.task == "segment", \
        f"[错误] 模型任务类型是 {model.task}, 不是 segment!"
    print(f"    模型加载完成")
    print()

    # 每幅 TIF 独立处理
    tif_results = []  # [(tif_name, all_cands, nms_kept, filtered)]

    for ti, tif_path in enumerate(tif_paths, 1):
        tif_name = os.path.basename(tif_path)
        print(f"[3.{ti}] 处理 {tif_name}")
        print("-" * 60)
        t_start = time.time()

        try:
            with rasterio.open(tif_path) as src:
                print(f"  原始 CRS         : {src.crs}")
                print(f"  原始 size        : {src.width}x{src.height}")
                print(f"  原始 dtype       : {src.dtypes}")
                print(f"  原始 nodata      : {src.nodata}")
                print(f"  原始 descriptions: {src.descriptions}")

                vrt = build_vrt(src)
                print(f"  VRT CRS          : {vrt.crs}")
                print(f"  VRT size         : {vrt.width}x{vrt.height}")
                print(f"  VRT bounds       : {vrt.bounds}")
                print(f"  VRT nodata       : {vrt.nodata}")
                print(f"  VRT res(m)       : {abs(vrt.transform.a):.2f} x "
                      f"{abs(vrt.transform.e):.2f}")

                print(f"  [计算全局拉伸分位数]...")
                stretch_params = compute_global_stretch(vrt, BANDS_RGB)
                band_names = ["R", "G", "B"]
                for i, (lo, hi) in enumerate(stretch_params):
                    print(f"    {band_names[i]}: {CLIP_PCT[0]}%={lo:.2f}, "
                          f"{CLIP_PCT[1]}%={hi:.2f}")

                candidates, n_win, n_valid, skip = predict_one_tif(
                    vrt, model, stretch_params, tif_name
                )
                vrt.close()

            # --- NMS 去重 ---
            print(f"  [NMS] {tif_name}:")
            nms_kept, nms_stats = nms_dedup(candidates)

            # --- 质量筛选 ---
            print(f"  [筛选] {tif_name}:")
            filtered, filter_reasons = apply_quality_filter(nms_kept)

            # --- 调试叠加图 ---
            save_debug_overlays(tif_name, candidates, nms_kept, filtered)

            t_elapsed = time.time() - t_start
            print(f"  [TIF 完成] {tif_name} 用时 {t_elapsed:.1f}s")
            print(f"    原始候选: {len(candidates)}")
            print(f"    NMS 后  : {len(nms_kept)}")
            print(f"    filtered: {len(filtered)}")
            print()

            tif_results.append((tif_name, candidates, nms_kept, filtered))

        except Exception as e:
            print(f"  [TIF 失败] {tif_name}: {e}")
            traceback.print_exc()
            continue

    # =================================================================
    # 汇总
    # =================================================================
    all_candidates = []
    all_nms_kept = []
    all_filtered = []
    for tif_name, cands, nms, filt in tif_results:
        all_candidates.extend(cands)
        all_nms_kept.extend(nms)
        all_filtered.extend(filt)

    print(f"[4] 全部 TIF 推理完成:")
    print(f"    原始候选总数     : {len(all_candidates)}")
    print(f"    NMS 后总数       : {len(all_nms_kept)}")
    print(f"    filtered 总数    : {len(all_filtered)}")
    print()

    # --- 导出 all SHP ---
    if all_nms_kept:
        print(f"[5] 导出 all SHP: {OUT_SHP_ALL}")
        gdf_all = export_shp(all_nms_kept, OUT_SHP_ALL)
        print(f"    导出完成: {len(gdf_all)} 个要素")
        print(f"    CRS: {gdf_all.crs}")

        areas = gdf_all["area_m2"].values
        print(f"    面积 (m2): min={areas.min():.0f}, max={areas.max():.0f}, "
              f"mean={areas.mean():.0f}, median={np.median(areas):.0f}")
        confs = gdf_all["confidence"].values
        print(f"    置信度: min={confs.min():.3f}, max={confs.max():.3f}, "
              f"mean={confs.mean():.3f}")
        compacts = gdf_all["compactnes"].values
        print(f"    圆度: min={compacts.min():.3f}, max={compacts.max():.3f}, "
              f"mean={compacts.mean():.3f}")
        n_edge = int(gdf_all["is_edge"].sum())
        print(f"    边缘实例: {n_edge}/{len(gdf_all)}")
    else:
        print("[警告] NMS 后无剩余多边形,不生成 all SHP")

    # --- 导出 filtered SHP ---
    if all_filtered:
        print(f"\n[6] 导出 filtered SHP: {OUT_SHP_FILTERED}")
        gdf_filt = export_shp(all_filtered, OUT_SHP_FILTERED)
        print(f"    导出完成: {len(gdf_filt)} 个要素")
        print(f"    CRS: {gdf_filt.crs}")
        areas = gdf_filt["area_m2"].values
        print(f"    面积 (m2): min={areas.min():.0f}, max={areas.max():.0f}, "
              f"mean={areas.mean():.0f}")
        confs = gdf_filt["confidence"].values
        print(f"    置信度: min={confs.min():.3f}, max={confs.max():.3f}, "
              f"mean={confs.mean():.3f}")
    else:
        print("\n[警告] filtered 后无剩余多边形,不生成 filtered SHP")

    # --- 汇总报告 ---
    print()
    print("=" * 60)
    print("v2 预测 + 矢量化 完成汇总")
    print("=" * 60)
    print(f"  处理 TIF 数           : {len(tif_paths)}")
    for tif_name, cands, nms, filt in tif_results:
        print(f"  {tif_name}:")
        print(f"    原始候选: {len(cands)}")
        print(f"    NMS 后  : {len(nms)}")
        print(f"    filtered: {len(filt)}")
    print(f"  总原始候选             : {len(all_candidates)}")
    print(f"  总 NMS 后             : {len(all_nms_kept)}")
    print(f"  总 filtered           : {len(all_filtered)}")
    print(f"  输出 SHP (all)        : {OUT_SHP_ALL}")
    print(f"  输出 SHP (filtered)   : {OUT_SHP_FILTERED}")
    print(f"  SHP CRS               : EPSG:32648")
    print(f"  调试叠加图             : {DEBUG_OVERLAY_DIR}")
    print(f"  调试输入图             : {DEBUG_INPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
