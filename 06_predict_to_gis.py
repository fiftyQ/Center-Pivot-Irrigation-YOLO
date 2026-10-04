# -*- coding: utf-8 -*-
"""
06_predict_to_gis.py (v3 修复版)
================================================================
第五阶段:YOLO11 实例分割预测 + 矢量化
目标:对 GEE_Export_2025 下的大幅 TIF 做滑窗推理,输出 EPSG:32648 Shapefile

v2→v3 根因修复:
  原始 TIF nodata=None,但 GEE 实际用 0 作为 NoData 值(约 34% 像元为 0)
  build_vrt 中 nodata=src.nodata=None → WarpedVRT nodata=None
  → vrt.read(masked=True) 返回 nomask → 所有像元被当作有效
  → 边缘窗口中大量 nodata=0 像元混入,valid_ratio 判断失效
  修复:build_vrt 显式设置 nodata=0,让 WarpedVRT 正确标记无效像元

  同时修复:
  - WarpedVRT 不支持 boundless 读取,改为限制窗口在 VRT 范围内
  - 边缘窗口尺寸不足时跳过(因为不需要 boundless 了)
  - 完整复用 data_prepare.py 的 build_vrt / compute_global_stretch / apply_stretch_uint8
  - 百分位统计严格排除 nodata 掩膜像元

纯 YOLO 路线:严禁 OpenCV 边缘检测 / 决策树 / GDAL 矢量化
所有 mask 来自 ultralytics result.masks.xy

运行环境:conda yolotest
Windows 多线程保护:if __name__ == '__main__'
================================================================
"""

import os
import sys
import glob
import time
import traceback
import csv

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.windows import Window, bounds as win_bounds, transform as win_transform_fn
from rasterio.vrt import WarpedVRT
from rasterio.warp import calculate_default_transform
from rasterio.enums import Resampling
from rasterio.crs import CRS
from shapely.geometry import Polygon, MultiPolygon, box
from shapely.ops import unary_union
from shapely.validation import make_valid

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from ultralytics import YOLO
except ImportError:
    print("[错误] 未安装 ultralytics,请先执行: pip install ultralytics")
    sys.exit(1)


# =====================================================================
# 配置区 —— 与 data_prepare.py 严格一致
# =====================================================================
TIF_DIR       = r"E:\260827YOLORUN\GEE_Export_2025"
MODEL_PATH    = r"E:\260827YOLORUN\runs\segment\ordos_farmland_v1\weights\best.pt"
OUT_DIR       = r"E:\260827YOLORUN\predict_results"
OUT_SHP       = os.path.join(OUT_DIR, "farmlands_2025.shp")

TARGET_CRS = CRS.from_epsg(32648)
TARGET_RESOLUTION_METERS = 10.0

TILE_SIZE = 640
STEP      = 512
BANDS_RGB = [3, 2, 1]

CLIP_PCT     = (2, 98)
BLOCK_STAT   = 4096
HIST_BINS    = 2048
VALID_PIX_RATIO_MIN = 0.20

CONF_THRESH  = 0.20
IOU_THRESH   = 0.50
DEVICE       = 0
BATCH_SIZE   = 4
RETINA_MASKS = True

MIN_AREA_M2  = 100.0

DEDUP_IOU_THRESH = 0.5

MAX_TIFS          = 1
MAX_VALID_WINDOWS = 30
PRINT_PROGRESS_EVERY = 20

STOP_WHEN_POLYS  = 30

DEBUG_INPUT_DIR = os.path.join(OUT_DIR, "debug_inputs")
DEBUG_MASK_DIR  = os.path.join(OUT_DIR, "debug_masks")
SAVE_DEBUG_MAX  = 30


# =====================================================================
# 复用 data_prepare.py 的工具函数
# =====================================================================
def find_tifs(tif_dir):
    pats = [os.path.join(tif_dir, "*." + e) for e in ("tif", "tiff", "TIF", "TIFF")]
    found = []
    for p in pats:
        found.extend(glob.glob(p))
    return sorted(set(found))


def build_vrt(src):
    """
    与 data_prepare.py 基本一致,但显式设置 nodata=0。
    根因:GEE 导出的 TIF nodata=None,但实际用 0 表示无效值(约 34% 像元为 0)。
    WarpedVRT 必须设 nodata=0 才能让 masked=True 正确生成掩膜。
    """
    if src.crs is None:
        raise ValueError(f"[CRS 错误] TIF 的 CRS 为 None,无法重投影。")
    dst_transform, dst_width, dst_height = calculate_default_transform(
        src.crs, TARGET_CRS, src.width, src.height, *src.bounds,
        resolution=TARGET_RESOLUTION_METERS,
    )
    res = TARGET_RESOLUTION_METERS
    dst_transform = rasterio.transform.from_origin(
        dst_transform.c, dst_transform.f, res, res
    )
    assert abs(abs(dst_transform.a) - abs(dst_transform.e)) < 1e-9, \
        f"[像元非正方形] X={abs(dst_transform.a)}, Y={abs(dst_transform.e)}"

    vrt = WarpedVRT(
        src, crs=TARGET_CRS,
        transform=dst_transform,
        width=dst_width, height=dst_height,
        resampling=Resampling.bilinear,
        nodata=0,   # 关键修复:GEE 实际用 0 表示 nodata
    )
    assert vrt.crs == TARGET_CRS, f"[VRT CRS 错] {vrt.crs}"
    assert abs(abs(vrt.transform.a) - abs(vrt.transform.e)) < 1e-9, \
        f"[VRT 像元非正方形] X={abs(vrt.transform.a)}, Y={abs(vrt.transform.e)}"
    return vrt


def compute_global_stretch(vrt, bands, block=BLOCK_STAT, bins=HIST_BINS):
    """
    完全复用 data_prepare.py 的 compute_global_stretch。
    分块统计 2%/98% 分位数,使用三波段共同有效掩膜(排除 nodata + mask + NaN)。
    """
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
            raise ValueError(
                f"[波段无有效像元] 第 {bands[bi]} 波段在全图范围内无有效像元。"
            )

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
            raise ValueError(f"[波段无有效像元] 第 {bands[bi]} 波段直方图为空。")
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
    """
    完全复用 data_prepare.py 的 apply_stretch_uint8。
    (3, H, W) → (H, W, 3) uint8,无效像元黑色。
    """
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
    """用 rasterio PNG 驱动保存 (H, W, 3) uint8。"""
    h, w, _ = arr_hwc_uint8.shape
    with rasterio.open(
        out_path, "w",
        driver="PNG",
        height=h, width=w, count=3, dtype="uint8",
    ) as dst:
        dst.write(arr_hwc_uint8.transpose(2, 0, 1))


def save_overlay_png(img_hwc_uint8, mask_px_list, out_path):
    """
    底图 PNG + 红色 mask 边界叠加图(仅调试用)。
    mask_px_list: list of (N, 2) ndarray,像素坐标。
    """
    fig, ax = plt.subplots(figsize=(6.4, 6.4), dpi=100)
    ax.imshow(img_hwc_uint8)
    for poly in mask_px_list:
        if len(poly) < 3:
            continue
        xs = poly[:, 0]
        ys = poly[:, 1]
        xs = np.append(xs, xs[0])
        ys = np.append(ys, ys[0])
        ax.plot(xs, ys, color="red", linewidth=1.0)
    ax.set_xlim(0, TILE_SIZE)
    ax.set_ylim(TILE_SIZE, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


# =====================================================================
# 核心推理函数
# =====================================================================
def predict_one_tif(vrt, model, stretch_params, tif_name, n_valid_windows_limit=None):
    """
    对单幅 VRT 滑窗推理。

    关键设计:
    1. WarpedVRT 不支持 boundless 读取 → 窗口必须限制在 [0, w-TILE_SIZE] 范围内
    2. VRT nodata=0 → masked=True 正确返回掩膜
    3. 有效窗口判断:valid_ratio >= VALID_PIX_RATIO_MIN (20%)
    4. 全局拉伸排除 nodata 掩膜像元
    5. 逐窗口打印诊断信息
    """
    h, w = vrt.height, vrt.width

    r_offsets = list(range(0, h - TILE_SIZE + 1, STEP))
    c_offsets = list(range(0, w - TILE_SIZE + 1, STEP))

    n_total = len(r_offsets) * len(c_offsets)
    print(f"  [滑窗] VRT {w}×{h}, 行偏移 {len(r_offsets)}, 列偏移 {len(c_offsets)}, "
          f"总窗口 {n_total}")

    print(f"  VRT descriptions: {vrt.descriptions}")
    print(f"  VRT dtypes: {vrt.dtypes}")
    print(f"  VRT nodata: {vrt.nodata}")
    print(f"  VRT CRS: {vrt.crs}")

    all_polys = []
    skip_reasons = {
        "low_valid_ratio": 0,
        "read_error": 0,
        "size_mismatch": 0,
        "infer_error": 0,
        "no_masks": 0,
    }
    window_count = 0
    valid_window_count = 0
    debug_saved = 0

    batch_images = []
    batch_windows = []

    def flush_batch():
        nonlocal all_polys, batch_images, batch_windows, skip_reasons, debug_saved
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
                n_confs = 0 if result.boxes is None else result.boxes.conf.tolist()

                if debug_saved < SAVE_DEBUG_MAX:
                    win_name = f"w{c_off:05d}_r{r_off:05d}"
                    save_png(img_hwc,
                             os.path.join(DEBUG_INPUT_DIR, f"{tif_name}_{win_name}.png"))
                    if result.masks is not None and len(result.masks.xy) > 0:
                        save_overlay_png(
                            img_hwc, result.masks.xy,
                            os.path.join(DEBUG_MASK_DIR, f"{tif_name}_{win_name}.png")
                        )
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

                    utm_coords = [
                        tile_transform * (float(x), float(y))
                        for x, y in poly_px
                    ]
                    if len(utm_coords) >= 2 and \
                       abs(utm_coords[0][0] - utm_coords[-1][0]) < 1e-6 and \
                       abs(utm_coords[0][1] - utm_coords[-1][1]) < 1e-6:
                        utm_coords = utm_coords[:-1]
                    if len(utm_coords) < 3:
                        continue

                    try:
                        poly = Polygon(utm_coords)
                        if not poly.is_valid:
                            poly = make_valid(poly)
                        if not isinstance(poly, Polygon) or poly.is_empty:
                            continue
                        if poly.area < MIN_AREA_M2:
                            continue

                        conf = float(n_confs[mi]) if mi < len(n_confs) else 0.0
                        all_polys.append({
                            "geometry": poly,
                            "confidence": conf,
                            "area_m2": round(poly.area, 2),
                            "perimeter": round(poly.length, 2),
                            "source_tif": tif_name,
                            "window_row": r_off,
                            "window_col": c_off,
                        })
                    except Exception:
                        continue

                if valid_window_count <= 30 or n_masks > 0:
                    print(f"    [推理诊断] win({c_off},{r_off}): boxes={n_boxes}, "
                          f"masks={n_masks}, confs={[f'{c:.3f}' for c in n_confs]}")
                    if result.masks is not None:
                        for mi, pp in enumerate(result.masks.xy):
                            print(f"      mask[{mi}]: {len(pp)} pts, "
                                  f"bbox=[{pp[:,0].min():.0f},{pp[:,1].min():.0f},"
                                  f"{pp[:,0].max():.0f},{pp[:,1].max():.0f}]")

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
                      f"已收集多边形 {len(all_polys)}  跳过 {skip_reasons}")

            try:
                data = vrt.read(BANDS_RGB, window=win, masked=True)
            except Exception as e:
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
                finite_valid = np.all(np.isfinite(data.data), axis=0)
                valid_mask &= finite_valid

            valid_ratio = float(valid_mask.mean())

            if window_count <= 30:
                band_stats = []
                for bi in range(3):
                    vals = data.data[bi][valid_mask]
                    if vals.size > 0:
                        band_stats.append(
                            f"R:{vals.min()}-{vals.max()}" if bi == 0 else
                            f"G:{vals.min()}-{vals.max()}" if bi == 1 else
                            f"B:{vals.min()}-{vals.max()}"
                        )
                    else:
                        band_stats.append(f"X:无有效")
                mask_pct = data.mask.mean() if data.mask is not np.ma.nomask else 0.0
                print(f"    [窗口#{window_count}] ({c_off},{r_off}) "
                      f"shape={data.shape} dtype={data.dtype} "
                      f"valid={valid_ratio:.3f} mask_pct={mask_pct:.3f} "
                      f"{' '.join(band_stats)}")

            if valid_ratio < VALID_PIX_RATIO_MIN:
                skip_reasons["low_valid_ratio"] += 1
                continue

            valid_window_count += 1

            rgb_uint8 = apply_stretch_uint8(data.data, stretch_params, valid_mask)

            batch_images.append(rgb_uint8)
            batch_windows.append((c_off, r_off))

            if len(batch_images) >= BATCH_SIZE:
                flush_batch()

            if STOP_WHEN_POLYS is not None and \
               len(all_polys) >= STOP_WHEN_POLYS:
                print(f"  [收集到 {len(all_polys)} 个多边形,提前结束遍历]")
                flush_batch()
                break
        else:
            continue
        break

    flush_batch()

    print(f"  [滑窗完成] {tif_name}:")
    print(f"    总遍历窗口   : {window_count}")
    print(f"    有效窗口     : {valid_window_count}")
    print(f"    跳过原因     : {skip_reasons}")
    print(f"    收集多边形   : {len(all_polys)}")
    return all_polys, window_count, valid_window_count, skip_reasons


# =====================================================================
# IoU 去重(不 dissolve,保留实例)
# =====================================================================
def iou_dedup(poly_dicts, iou_thresh=DEDUP_IOU_THRESH):
    """
    按 confidence 从高到低排序,同类 IoU > 阈值时保留高置信度者。
    不合并、不 dissolve,保留每个独立农田实例。
    """
    if not poly_dicts:
        return []

    print(f"  [IoU 去重] 输入 {len(poly_dicts)} 个候选多边形,阈值={iou_thresh}...")
    start = time.time()

    sorted_dicts = sorted(poly_dicts, key=lambda d: d["confidence"], reverse=True)

    kept = []
    dup_count = 0
    for pd in sorted_dicts:
        g = pd["geometry"]
        is_dup = False
        for kd in kept:
            kg = kd["geometry"]
            if not g.intersects(kg):
                continue
            inter = g.intersection(kg).area
            union = g.union(kg).area
            if union < 1e-12:
                continue
            iou = inter / union
            if iou > iou_thresh:
                is_dup = True
                dup_count += 1
                break
        if not is_dup:
            kept.append(pd)

    print(f"    去重前 {len(sorted_dicts)} → 去重后 {len(kept)} "
          f"(去除 {dup_count}),用时 {time.time() - start:.1f}s")
    return kept


# =====================================================================
# 主流程
# =====================================================================
def main():
    print("=" * 60)
    print("YOLO11 实例分割预测 + 矢量化 (v3 修复版)")
    print("=" * 60)
    print(f"TIF 目录     : {TIF_DIR}")
    print(f"模型权重     : {MODEL_PATH}")
    print(f"输出 SHP     : {OUT_SHP}")
    print(f"目标 CRS     : EPSG:32648")
    print(f"像元分辨率   : {TARGET_RESOLUTION_METERS} m")
    print(f"滑窗尺寸     : {TILE_SIZE}×{TILE_SIZE}")
    print(f"滑窗步长     : {STEP}")
    print(f"置信度阈值   : {CONF_THRESH}")
    print(f"IoU 阈值     : {IOU_THRESH}")
    print(f"去重 IoU     : {DEDUP_IOU_THRESH}")
    print(f"面积过滤     : ≥ {MIN_AREA_M2} m²")
    print(f"MAX_TIFS     : {MAX_TIFS}")
    print(f"MAX_VALID_W  : {MAX_VALID_WINDOWS}")
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
    os.makedirs(DEBUG_MASK_DIR, exist_ok=True)

    print(f"[2] 加载 YOLO 模型: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print(f"    model.task = {model.task}")
    assert model.task == "segment", \
        f"[错误] 模型任务类型是 {model.task},不是 segment!"
    print(f"    模型加载完成")
    print()

    all_poly_dicts = []
    total_windows = 0
    total_valid = 0
    all_skip_reasons = {}

    for ti, tif_path in enumerate(tif_paths, 1):
        tif_name = os.path.basename(tif_path)
        print(f"[3.{ti}] 处理 {tif_name}")
        print("-" * 60)
        t_start = time.time()

        try:
            with rasterio.open(tif_path) as src:
                print(f"  原始 CRS         : {src.crs}")
                print(f"  原始 size        : {src.width}×{src.height}")
                print(f"  原始 dtype       : {src.dtypes}")
                print(f"  原始 nodata      : {src.nodata}")
                print(f"  原始 bounds      : {src.bounds}")
                print(f"  原始 descriptions: {src.descriptions}")

                vrt = build_vrt(src)
                print(f"  VRT CRS          : {vrt.crs}")
                print(f"  VRT size         : {vrt.width}×{vrt.height}")
                print(f"  VRT bounds       : {vrt.bounds}")
                print(f"  VRT nodata       : {vrt.nodata}")
                print(f"  VRT res(m)       : {abs(vrt.transform.a):.2f} × "
                      f"{abs(vrt.transform.e):.2f}")

                print(f"  [计算全局拉伸分位数]...")
                stretch_params = compute_global_stretch(vrt, BANDS_RGB)
                band_names = ["R", "G", "B"]
                for i, (lo, hi) in enumerate(stretch_params):
                    print(f"    {band_names[i]}: {CLIP_PCT[0]}%={lo:.2f}, "
                          f"{CLIP_PCT[1]}%={hi:.2f}")

                polys, n_win, n_valid, skip = predict_one_tif(
                    vrt, model, stretch_params, tif_name,
                    n_valid_windows_limit=MAX_VALID_WINDOWS,
                )
                vrt.close()

            all_poly_dicts.extend(polys)
            total_windows += n_win
            total_valid += n_valid
            for k, v in skip.items():
                all_skip_reasons[k] = all_skip_reasons.get(k, 0) + v

            t_elapsed = time.time() - t_start
            print(f"  [TIF 完成] {tif_name} 用时 {t_elapsed:.1f}s, "
                  f"新增 {len(polys)} 个候选多边形\n")

        except Exception as e:
            print(f"  [TIF 失败] {tif_name}: {e}")
            traceback.print_exc()
            continue

    print(f"[4] 全部 TIF 推理完成:")
    print(f"    总遍历窗口       : {total_windows}")
    print(f"    总有效窗口       : {total_valid}")
    print(f"    跳过原因汇总     : {all_skip_reasons}")
    print(f"    候选多边形总数   : {len(all_poly_dicts)}")
    print()

    if not all_poly_dicts:
        print("[警告] 没有检测到任何农田多边形,不生成 SHP。")
        print("       可能原因:")
        print("       1. 模型置信度过高 → 降低 CONF_THRESH")
        print("       2. 影像颜色分布与训练集差异大 → 检查全局拉伸参数")
        print("       3. 模型权重损坏 → 检查 best.pt 是否正常")
        print("       4. TIF 确实不含农田 → 换一幅有农田的 TIF 测试")
        sys.exit(0)

    print("[5] IoU 去重")
    final_poly_dicts = iou_dedup(all_poly_dicts)
    print()

    if not final_poly_dicts:
        print("[警告] 去重后无剩余多边形,不生成 SHP。")
        sys.exit(0)

    print(f"[6] 导出 Shapefile: {OUT_SHP}")
    gdf_out = gpd.GeoDataFrame(
        {
            "geometry": [d["geometry"] for d in final_poly_dicts],
            "id": list(range(len(final_poly_dicts))),
            "confidence": [d["confidence"] for d in final_poly_dicts],
            "area_m2": [d["area_m2"] for d in final_poly_dicts],
            "perimeter": [d["perimeter"] for d in final_poly_dicts],
            "source_tif": [d["source_tif"] for d in final_poly_dicts],
            "window_row": [d["window_row"] for d in final_poly_dicts],
            "window_col": [d["window_col"] for d in final_poly_dicts],
        },
        crs=TARGET_CRS,
    )
    gdf_out.to_file(OUT_SHP, driver="ESRI Shapefile", encoding="utf-8")
    print(f"    导出完成: {len(gdf_out)} 个要素")
    print(f"    CRS: {gdf_out.crs}")
    print()

    print("=" * 60)
    print("预测 + 矢量化 完成汇总")
    print("=" * 60)
    print(f"  处理 TIF 数           : {len(tif_paths)}")
    print(f"  总遍历窗口           : {total_windows}")
    print(f"  总有效窗口           : {total_valid}")
    print(f"  跳过原因汇总         : {all_skip_reasons}")
    print(f"  候选多边形(去重前)   : {len(all_poly_dicts)}")
    print(f"  最终多边形(去重后)   : {len(final_poly_dicts)}")
    print(f"  输出 SHP              : {OUT_SHP}")
    print(f"  SHP CRS               : EPSG:32648")
    areas = gdf_out["area_m2"].values
    print(f"  面积分布 (m²):")
    print(f"    min={areas.min():.1f}, max={areas.max():.1f}, "
          f"mean={areas.mean():.1f}, median={np.median(areas):.1f}")
    confs = gdf_out["confidence"].values
    print(f"  置信度分布:")
    print(f"    min={confs.min():.3f}, max={confs.max():.3f}, "
          f"mean={confs.mean():.3f}")
    print(f"  调试 PNG:")
    print(f"    输入窗口: {DEBUG_INPUT_DIR}")
    print(f"    mask 叠加: {DEBUG_MASK_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()