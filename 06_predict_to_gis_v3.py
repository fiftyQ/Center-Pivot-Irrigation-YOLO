# -*- coding: utf-8 -*-
"""
06_predict_to_gis_v3.py

阶段 A — 对 yolo_dataset_v2/images/test 做无偏评估(4 个阈值):
  - conf in [0.35, 0.50, 0.65, 0.75]
  - 输出: mask 填充叠加图、红色边界叠加图、预测 TXT、报告
  - 若有真实 test TXT 标签: 匹配每个阈值下的 TP/FP/FN, 计算误检/漏检

阶段 B — 对 1 幅原始 WGS84 TIF 做滑窗预测(小规模测试):
  - WarpedVRT: EPSG:32648, 与 data_prepare.py 完全一致
  - 步长 512, 窗口 640, RGB[3,2,1], 2-98% 拉伸
  - result.masks.xy -> tile_transform (vrt.transform) -> UTM
  - 实例级 Polygon NMS IoU>=0.70, 不 unary_union
  - 输出 all / filtered SHP (CRS=EPSG:32648)
  - MAX_TIFS=1, MAX_VALID_WINDOWS=30, 不全量运行

使用新模型: ordos_farmland_v2_clean_split/best.pt
"""
from __future__ import annotations

import glob
import math
import os
import random
import sys
import time
import traceback

import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import calculate_default_transform
from rasterio.windows import Window, transform as win_transform_fn
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from shapely.validation import make_valid

import geopandas as gpd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection

# Ultralytics + PIL (仅 test 集评估用 PIL 读图, 切片已直接是 uint8)
try:
    from ultralytics import YOLO
    from PIL import Image as PILImage
except ImportError as exc:
    print(f"[错误] 缺少依赖: {exc}")
    sys.exit(1)

# =====================================================================
# 配置
# =====================================================================
TIF_DIR    = r"E:\260827YOLORUN\GEE_Export_2025"
MODEL_PATH = r"E:\260827YOLORUN\runs\segment\ordos_farmland_v2_clean_split\weights\best.pt"

DATASET_V2 = r"E:\260827YOLORUN\yolo_dataset_v2"
TEST_IMG_DIR = os.path.join(DATASET_V2, "images", "test")
TEST_LBL_DIR = os.path.join(DATASET_V2, "labels", "test")
TEST_OUT_ROOT = r"E:\260827YOLORUN\predict_results\v2_test"

PRED_OUT_DIR = r"E:\260827YOLORUN\predict_results"
# 2026-08-30 严格30窗口诊断：使用独立输出路径，避免覆盖之前 v3 结果
OUT_SHP_ALL = os.path.join(PRED_OUT_DIR,
                           "farmlands_2025_v2_strict30_all.shp")
OUT_SHP_FILTERED = os.path.join(PRED_OUT_DIR,
                                "farmlands_2025_v2_strict30_filtered.shp")

TARGET_CRS = CRS.from_epsg(32648)
TARGET_RESOLUTION_METERS = 10.0
TILE_SIZE = 640
STEP = 512
BANDS_RGB = [3, 2, 1]
CLIP_PCT = (2, 98)
BLOCK_STAT = 4096
HIST_BINS = 2048
VALID_PIX_RATIO_MIN = 0.20

# 推理参数
IMG_SIZE = 640
DEVICE = 0
RETINA_MASKS = True
INFER_IOU = 0.50
# 严格诊断模式：INFER_BATCH=1，每窗口推理后立即判断是否达到 windows_with_masks>=MAX_VALID_WINDOWS，
# 防止批量推理造成 overshoot。
INFER_BATCH = 1

CONF_LIST = [0.35, 0.50, 0.65, 0.75]
# 整幅 TIF 预测时的单次推理阈值 (低阈值 -> 后续按阈值过滤),
# 但用户明确要求默认 conf=0.65, 不做再过滤。因此整幅 TIF 直接用 0.65。
TIF_INFER_CONF = 0.65

# NMS / 质量筛选
DEDUP_IOU_THRESH = 0.70
EDGE_DEDUP_IOU = 0.50
FILTER_CONF = 0.65
FILTER_MIN_AREA = 10000
FILTER_MAX_AREA = 600000
FILTER_MIN_COMPACT = 0.35

SIMPLIFY_TOLERANCE_M = 0.0
EDGE_PIXEL_THRESHOLD = 10

# 小规模限制
MAX_TIFS = 1
MAX_VALID_WINDOWS = 30
PRINT_EVERY = 5

DEBUG_DIR = os.path.join(PRED_OUT_DIR, "debug_overlays_v3_strict30")

# =====================================================================
# 2026-08-30 全量模式 (用户确认后开启)
# FULL_MODE=True : 处理全部 6 幅 TIF, 无含mask窗口上限,
#                  输出到 *_full.shp 与 debug_overlays_v3_full,
#                  不覆盖 strict30 与旧 v3 结果 (文件名全部不同)
# =====================================================================
FULL_MODE = True
if FULL_MODE:
    MAX_TIFS = None            # None => 处理 find_tifs 返回的全部 TIF
    MAX_VALID_WINDOWS = None   # None => 不限制含 mask 窗口数量
    INFER_BATCH = 8
    OUT_SHP_ALL = os.path.join(PRED_OUT_DIR,
                               "farmlands_2025_v2_all_full.shp")
    OUT_SHP_FILTERED = os.path.join(PRED_OUT_DIR,
                                    "farmlands_2025_v2_filtered_full.shp")
    DEBUG_DIR = os.path.join(PRED_OUT_DIR, "debug_overlays_v3_full")


def dir_name_for_conf(t):
    return os.path.join(TEST_OUT_ROOT, f"conf_{int(round(t*100)):03d}")


# =====================================================================
# 共用基础函数(与 data_prepare.py / v2 完全一致)
# =====================================================================
def find_tifs(tif_dir):
    pats = [os.path.join(tif_dir, "*." + e) for e in ("tif", "tiff", "TIF", "TIFF")]
    found = []
    for p in pats:
        found.extend(glob.glob(p))
    return sorted(set(found))


def build_vrt(src):
    if src.crs is None:
        raise ValueError("[CRS 错误] TIF CRS 为 None")
    dst_transform, dst_width, dst_height = calculate_default_transform(
        src.crs, TARGET_CRS, src.width, src.height, *src.bounds,
        resolution=TARGET_RESOLUTION_METERS,
    )
    res = TARGET_RESOLUTION_METERS
    dst_transform = rasterio.transform.from_origin(
        dst_transform.c, dst_transform.f, res, res,
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
            common_valid = ~np.any(data.mask, axis=0) \
                if data.mask is not np.ma.nomask else np.ones((h, w), dtype=bool)
            if not common_valid.any():
                continue
            for bi in range(n):
                vals = data.data[bi][common_valid]
                if vals.size == 0:
                    continue
                vmin, vmax = float(vals.min()), float(vals.max())
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
            common_valid = ~np.any(data.mask, axis=0) \
                if data.mask is not np.ma.nomask else np.ones((h, w), dtype=bool)
            if not common_valid.any():
                continue
            for bi in range(n):
                vals = data.data[bi][common_valid]
                if vals.size == 0:
                    continue
                hist_acc[bi] += np.histogram(vals, bins=edges_list[bi])[0]

    stretch = []
    for bi in range(n):
        edges = edges_list[bi]
        hist = hist_acc[bi]
        total = hist.sum()
        cum = np.cumsum(hist).astype(np.float64) / total
        p_lo = int(np.searchsorted(cum, CLIP_PCT[0] / 100.0))
        p_hi = int(np.searchsorted(cum, CLIP_PCT[1] / 100.0))
        p_lo = min(max(p_lo, 0), bins - 1)
        p_hi = min(max(p_hi, 0), bins - 1)
        if p_hi <= p_lo:
            p_hi = min(p_lo + 1, bins - 1)
        stretch.append((float(edges[p_lo]), float(edges[p_hi])))
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
                           np.clip(scaled, 0, 255).astype(np.uint8), 0)
    return out.transpose(1, 2, 0)


def is_edge_instance(poly_px, tile_size=TILE_SIZE, threshold=EDGE_PIXEL_THRESHOLD):
    arr = np.asarray(poly_px, dtype=float)
    if arr.size == 0:
        return False
    xmin, xmax = float(arr[:, 0].min()), float(arr[:, 0].max())
    ymin, ymax = float(arr[:, 1].min()), float(arr[:, 1].max())
    return (xmin < threshold or xmax > tile_size - threshold
            or ymin < threshold or ymax > tile_size - threshold)


def fix_geometry(poly):
    if not poly.is_valid:
        try:
            poly = make_valid(poly)
        except Exception:
            poly = poly.buffer(0)
    if isinstance(poly, Polygon):
        pass
    elif hasattr(poly, "geoms"):
        if len(poly.geoms) == 1:
            poly = poly.geoms[0]
        else:
            areas = [g.area for g in poly.geoms]
            poly = poly.geoms[int(np.argmax(areas))]
    if (SIMPLIFY_TOLERANCE_M > 0 and isinstance(poly, Polygon)
            and not poly.is_empty):
        poly = poly.simplify(SIMPLIFY_TOLERANCE_M)
    return poly


def poly_iou(g1, g2):
    if not g1.intersects(g2):
        return 0.0
    inter = g1.intersection(g2).area
    union = g1.union(g2).area
    return inter / union if union > 1e-12 else 0.0


def nms_dedup(candidates, iou_thresh=DEDUP_IOU_THRESH,
              edge_iou=EDGE_DEDUP_IOU):
    if not candidates:
        return [], {"nms_dropped": 0, "edge_dropped": 0}
    print(f"  [NMS] 输入 {len(candidates)}, IoU>= {iou_thresh}, edge IoU>= {edge_iou}")
    sorted_cands = sorted(
        candidates,
        key=lambda c: (c["is_edge"], -c["confidence"], -c["area_m2"])
    )
    kept = []
    kept_bounds = []
    nms_dropped = 0
    edge_dropped = 0
    for cand in sorted_cands:
        g = cand["geometry"]
        gb = g.bounds
        dup = False
        reason = ""
        for k, kb in zip(kept, kept_bounds):
            # bbox 预过滤: 包围盒不相交 => IoU 必为 0, 跳过精确计算
            # (纯性能优化, 不改变 NMS 语义)
            if gb[2] < kb[0] or kb[2] < gb[0] or gb[3] < kb[1] or kb[3] < gb[1]:
                continue
            kg = k["geometry"]
            iou = poly_iou(g, kg)
            if iou >= iou_thresh:
                dup = True
                reason = f"nms_iou={iou:.3f}>= {iou_thresh}"
                nms_dropped += 1
                break
            if cand["is_edge"] == 1 and k["is_edge"] == 0 and iou >= edge_iou:
                dup = True
                reason = f"edge_iou={iou:.3f}>= {edge_iou}"
                edge_dropped += 1
                break
        if dup:
            cand["status"] = "nms_dropped"
            cand["drop_reason"] = reason
        else:
            cand["status"] = "kept"
            kept.append(cand)
            kept_bounds.append(gb)
    print(f"    NMS 前 {len(sorted_cands)} -> NMS 后 {len(kept)} "
          f"(NMS 去除 {nms_dropped}, 边缘去除 {edge_dropped})")
    return kept, {"nms_dropped": nms_dropped, "edge_dropped": edge_dropped}


def apply_quality_filter(poly_dicts):
    kept = []
    reasons = {"low_conf": 0, "too_small": 0, "too_large": 0, "low_compactness": 0}
    for d in poly_dicts:
        if d["confidence"] < FILTER_CONF:
            reasons["low_conf"] += 1
            continue
        if d["area_m2"] < FILTER_MIN_AREA:
            reasons["too_small"] += 1
            continue
        if d["area_m2"] > FILTER_MAX_AREA:
            reasons["too_large"] += 1
            continue
        if d["compactness"] < FILTER_MIN_COMPACT:
            reasons["low_compactness"] += 1
            continue
        kept.append(d)
    print(f"  [筛选] 输入 {len(poly_dicts)} -> 输出 {len(kept)}  原因: {reasons}")
    return kept, reasons


# =====================================================================
# 阶段 A — test 集评估 (4 阈值)
# =====================================================================
def load_test_images_labels():
    """收集 test 集的 (img_path, lbl_path) 列表, 并验证配对。"""
    if not os.path.isdir(TEST_IMG_DIR):
        raise FileNotFoundError(f"test img 目录不存在: {TEST_IMG_DIR}")
    imgs = sorted(glob.glob(os.path.join(TEST_IMG_DIR, "*.png")))
    print(f"[A.1] test 集图片数: {len(imgs)}")
    pairs = []
    missing = 0
    for ip in imgs:
        stem = os.path.splitext(os.path.basename(ip))[0]
        lp = os.path.join(TEST_LBL_DIR, stem + ".txt")
        if not os.path.isfile(lp):
            missing += 1
            continue
        pairs.append((ip, lp))
    print(f"  PNG/TXT 完整配对: {len(pairs)}, 缺失 TXT: {missing}")
    return pairs


def parse_yolo_segment_txt(txt_path, img_w, img_h):
    """返回 shapely Polygon 列表(像素坐标)。"""
    polys = []
    if not os.path.isfile(txt_path):
        return polys
    text = open(txt_path, "r", encoding="utf-8").read().strip()
    if not text:
        return polys
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        try:
            coords = [float(v) for v in parts[1:]]
        except ValueError:
            continue
        if len(coords) % 2 != 0:
            continue
        arr = np.array(coords, dtype=float).reshape(-1, 2)
        arr_px = arr * [img_w, img_h]
        if len(arr_px) < 3:
            continue
        try:
            p = Polygon(arr_px)
            p = fix_geometry(p)
            if isinstance(p, Polygon) and not p.is_empty and p.area > 0.5:
                polys.append(p)
        except Exception:
            continue
    return polys


def match_preds_gts(pred_polys, gt_polys, iou_thresh=0.50):
    """贪心一对一 IoU 匹配, 返回 (tp, fp, fn, matched_pred_idx, matched_gt_idx)。"""
    # 构建 IoU 矩阵
    n_p, n_g = len(pred_polys), len(gt_polys)
    if n_p == 0 and n_g == 0:
        return 0, 0, 0, set(), set()
    ious = np.zeros((n_p, n_g), dtype=np.float64)
    for i in range(n_p):
        for j in range(n_g):
            ious[i, j] = poly_iou(pred_polys[i], gt_polys[j])
    # 贪心按 IoU 从大到小配对 (一对一)
    pairs = []
    for i in range(n_p):
        for j in range(n_g):
            if ious[i, j] >= iou_thresh:
                pairs.append((ious[i, j], i, j))
    pairs.sort(key=lambda x: -x[0])
    matched_p, matched_g = set(), set()
    for _, i, j in pairs:
        if i in matched_p or j in matched_g:
            continue
        matched_p.add(i)
        matched_g.add(j)
    tp = len(matched_p)
    fp = n_p - tp
    fn = n_g - len(matched_g)
    return tp, fp, fn, matched_p, matched_g


def save_png_pil(arr_hwc_uint8, out_path):
    PILImage.fromarray(arr_hwc_uint8.astype(np.uint8)).save(out_path)


def save_test_outputs(pairs, model):
    """对 test 集单次 low-conf 推理, 再按各阈值过滤输出。"""
    print("[A.2] 推理 test 集...")

    # 收集所有预测结果
    # all_preds_per_img[img_stem] = [(poly_px, confidence)]
    all_preds = {}
    all_imgs_data = {}
    all_gts = {}

    # 低阈值跑一次
    RUN_CONF = 0.05
    n_total = len(pairs)
    for idx, (img_path, lbl_path) in enumerate(pairs, 1):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        img_arr = np.asarray(PILImage.open(img_path).convert("RGB"))
        h, w = img_arr.shape[:2]
        all_imgs_data[stem] = (img_arr, (w, h))
        gts = parse_yolo_segment_txt(lbl_path, w, h)
        all_gts[stem] = gts

        try:
            res = model.predict(
                source=img_arr,
                conf=RUN_CONF, iou=INFER_IOU,
                imgsz=IMG_SIZE, device=DEVICE,
                retina_masks=RETINA_MASKS, verbose=False,
            )[0]
        except Exception as e:
            print(f"  [推理错误] {stem}: {e}")
            all_preds[stem] = []
            continue

        preds = []
        if res.masks is not None:
            confs = res.boxes.conf.tolist() if res.boxes is not None else []
            for mi, xy in enumerate(res.masks.xy):
                if len(xy) < 3:
                    continue
                try:
                    p = fix_geometry(Polygon(xy.tolist()
                                             if hasattr(xy, "tolist") else xy))
                    if (isinstance(p, Polygon) and not p.is_empty
                            and p.area > 0.5):
                        conf = float(confs[mi]) if mi < len(confs) else 0.0
                        preds.append((p, conf))
                except Exception:
                    continue
        all_preds[stem] = preds
        if idx % 10 == 0 or idx == n_total:
            print(f"    {idx}/{n_total}")

    # 按四个阈值分别输出
    summary_rows = []
    for thresh in CONF_LIST:
        out_dir = dir_name_for_conf(thresh)
        os.makedirs(out_dir, exist_ok=True)

        stems = sorted(all_preds.keys())
        tp_tot = fp_tot = fn_tot = 0
        n_with_pred = 0
        n_pred_inst = 0
        vertex_counts = []
        pred_area_stats = []
        per_img_counts = []
        for stem in stems:
            preds = [(p, c) for p, c in all_preds[stem] if c >= thresh]
            img_arr, (w, h) = all_imgs_data[stem]
            gts = all_gts[stem]

            # TXT: class x1 y1 x2 ...
            txt_lines = []
            polys_px = []
            confs_keep = []
            for p, c in preds:
                coords = list(p.exterior.coords)[:-1]
                if len(coords) < 3:
                    continue
                xy = np.asarray(coords, dtype=float) / [w, h]
                if (np.any(xy < 0) or np.any(xy > 1)):
                    # 允许裁剪到 0-1
                    xy = np.clip(xy, 0.0, 1.0)
                flat = [f"{v:.6f}" for v in xy.ravel()]
                txt_lines.append("0 " + " ".join(flat))
                polys_px.append(p)
                confs_keep.append(c)
                vertex_counts.append(len(coords))
                pred_area_stats.append(p.area)

            txt_path = os.path.join(out_dir, stem + ".txt")
            open(txt_path, "w", encoding="utf-8").write("\n".join(txt_lines) + "\n")

            per_img_counts.append(len(polys_px))
            n_pred_inst += len(polys_px)
            if len(polys_px) > 0:
                n_with_pred += 1

            # 匹配真实标签
            if gts:
                tp, fp, fn, _, _ = match_preds_gts(polys_px, gts)
                tp_tot += tp
                fp_tot += fp
                fn_tot += fn
            else:
                # 无 GT: 所有预测都是误检
                fp_tot += len(polys_px)

            # 图1: 填充叠加图
            fig, ax = plt.subplots(figsize=(w/96, h/96), dpi=96)
            ax.imshow(img_arr)
            if polys_px:
                colors = plt.cm.plasma(
                    np.linspace(0, 1, max(len(polys_px), 1)))
                pc = PolyCollection(
                    [list(p.exterior.coords) for p in polys_px],
                    facecolors=colors, edgecolors="none", alpha=0.55)
                ax.add_collection(pc)
            ax.set_title(f"conf>={thresh:.2f}  masks filled (n={len(polys_px)})",
                         fontsize=8)
            ax.axis("off")
            fig.tight_layout(pad=0)
            fig.savefig(os.path.join(out_dir, stem + "_filled.png"), dpi=96)
            plt.close(fig)

            # 图2: 红色边界图
            fig, ax = plt.subplots(figsize=(w/96, h/96), dpi=96)
            ax.imshow(img_arr)
            if polys_px:
                lc = LineCollection(
                    [list(p.exterior.coords) for p in polys_px],
                    colors="red", linewidths=1.5)
                ax.add_collection(lc)
                # 绿色: GT 边界
                if gts:
                    lc_gt = LineCollection(
                        [list(g.exterior.coords) for g in gts],
                        colors="lime", linewidths=0.8, linestyles="--")
                    ax.add_collection(lc_gt)
            ax.set_title(f"conf>={thresh:.2f}  red=pred  green=GT (n={len(polys_px)}/{len(gts)})",
                         fontsize=8)
            ax.axis("off")
            fig.tight_layout(pad=0)
            fig.savefig(os.path.join(out_dir, stem + "_boundary.png"), dpi=96)
            plt.close(fig)

        total_gts = sum(len(v) for v in all_gts.values())
        # FN = gt 中未匹配数量 = 总 GT - TP (但上面在逐图累积时有错: tp+fn!=gt_num 只对逐图成立)
        # 重新精确计算
        tp_tot2 = fp_tot2 = fn_tot2 = 0
        for stem in stems:
            preds = [p for p, c in all_preds[stem] if c >= thresh]
            gts = all_gts[stem]
            tp, fp, fn, _, _ = match_preds_gts(preds, gts)
            tp_tot2 += tp; fp_tot2 += fp; fn_tot2 += fn

        precision = tp_tot2 / max(1, tp_tot2 + fp_tot2)
        recall = tp_tot2 / max(1, tp_tot2 + fn_tot2)
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        avg_pts = float(np.mean(vertex_counts)) if vertex_counts else 0.0
        area_min = float(np.min(pred_area_stats)) if pred_area_stats else 0.0
        area_max = float(np.max(pred_area_stats)) if pred_area_stats else 0.0
        area_mean = float(np.mean(pred_area_stats)) if pred_area_stats else 0.0

        per_img_median = float(np.median(per_img_counts)) if per_img_counts else 0.0
        per_img_mean = float(np.mean(per_img_counts)) if per_img_counts else 0.0

        # 置信度区间统计
        confs = [c for stem in stems for _, c in all_preds[stem] if c >= thresh]
        bins_ct = {}
        for lo, hi in [(0.35, 0.50), (0.50, 0.65), (0.65, 0.75), (0.75, 1.01)]:
            bins_ct[f"{lo:.2f}-{hi:.2f}"] = int(
                sum(1 for c in confs if lo <= c < hi))

        report = []
        report.append(f"conf >= {thresh:.2f}   iou={INFER_IOU}   imgsz={IMG_SIZE}   retina_masks={RETINA_MASKS}")
        report.append(f"图片总数                  : {len(stems)}")
        report.append(f"有预测 mask 的图片数     : {n_with_pred}")
        report.append(f"预测实例总数              : {n_pred_inst}")
        report.append(f"每张图预测数量: mean={per_img_mean:.2f} median={per_img_median:.1f} max={max(per_img_counts, default=0)}")
        report.append(f"mask 多边形平均顶点数     : {avg_pts:.1f}")
        report.append(f"mask 面积(像素^2): min={area_min:.0f} max={area_max:.0f} mean={area_mean:.0f}")
        report.append(f"各置信度区间实例数量: {bins_ct}")
        if total_gts > 0:
            report.append(f"[GT 评估] TP / FP / FN     : {tp_tot2} / {fp_tot2} / {fn_tot2}")
            report.append(f"  Precision               : {precision:.3f}")
            report.append(f"  Recall (漏检=1-Recall)   : {recall:.3f}")
            report.append(f"  F1                      : {f1:.3f}")
            report.append(f"  误检数 (=FP)            : {fp_tot2}")
            report.append(f"  漏检数 (=FN)            : {fn_tot2}")
            report.append(f"  正确识别 TP             : {tp_tot2}")
        else:
            report.append("[GT 评估] test 集无真实标签, 仅报告预测统计。")
        report_text = "\n".join(report)
        print("\n[阈值 {:.2f}]".format(thresh))
        print(report_text)
        with open(os.path.join(out_dir, "report.txt"), "w", encoding="utf-8") as f:
            f.write(report_text + "\n")

        summary_rows.append({
            "threshold": thresh,
            "total_images": len(stems),
            "images_with_pred": n_with_pred,
            "pred_instances": n_pred_inst,
            "TP": tp_tot2, "FP": fp_tot2, "FN": fn_tot2,
            "Precision": precision, "Recall": recall, "F1": f1,
            "mean_vertices": avg_pts, "mean_area_px2": area_mean,
            "per_img_mean": per_img_mean, "per_img_median": per_img_median,
            "conf_bins": bins_ct,
        })

    # 汇总
    print("\n[A.3] test 集 四档阈值对比")
    cols = ["threshold", "pred_instances", "TP", "FP", "FN",
            "Precision", "Recall", "F1", "mean_vertices", "per_img_mean"]
    for s in summary_rows:
        print("  " + " | ".join(f"{k}={s[k]}" for k in cols))
    return summary_rows


# =====================================================================
# 阶段 B — 1 幅 TIF 小规模预测 (MAX_TIFS=1, MAX_VALID_WINDOWS=30)
# =====================================================================
def predict_small_scale(model, tif_paths):
    os.makedirs(DEBUG_DIR, exist_ok=True)
    os.makedirs(PRED_OUT_DIR, exist_ok=True)

    # ===== 独立计数器 (阶段B全局, 避免仅依赖 valid_win - no_mask 推导) =====
    total_windows_seen = 0
    valid_windows_seen = 0
    windows_with_masks = 0
    raw_candidates = 0
    reached_cap = False
    stop_reason = ""
    skip = {"low_valid": 0, "read_err": 0, "size_mismatch": 0,
            "no_mask": 0, "geom_err": 0, "infer_err": 0}
    candidates = []

    tif_path = tif_paths[0]
    tif_name = os.path.basename(tif_path)
    print(f"\n[B.1] 小规模预测: {tif_name}")
    print(f"  MAX_TIFS={MAX_TIFS}, MAX_VALID_WINDOWS={MAX_VALID_WINDOWS}")

    with rasterio.open(tif_path) as src:
        print(f"  src CRS={src.crs}, size={src.width}x{src.height}, "
              f"dtype={src.dtypes}, descriptions={src.descriptions}")
        vrt = build_vrt(src)
        print(f"  vrt={vrt.width}x{vrt.height}, crs={vrt.crs}, "
              f"bounds={vrt.bounds}, res_x={abs(vrt.transform.a):.2f}, "
              f"res_y={abs(vrt.transform.e):.2f}")
        print("  [计算全局拉伸分位数]")
        stretch = compute_global_stretch(vrt, BANDS_RGB)
        for i, (lo, hi) in enumerate(stretch):
            print(f"    {'RGB'[i]}: {CLIP_PCT[0]}%={lo:.2f}, "
                  f"{CLIP_PCT[1]}%={hi:.2f}")

        H, W = vrt.height, vrt.width
        r_offsets = list(range(0, H - TILE_SIZE + 1, STEP))
        c_offsets = list(range(0, W - TILE_SIZE + 1, STEP))

        # 小规模模式：优先扫描 VRT 中央块（前 MAX_TIFS*MAX_VALID_WINDOWS 个窗口
        # 更大概率覆盖真实农田区域，避免扫 30 个左上角纯背景窗口而 mask=0）
        if MAX_TIFS <= 1 and MAX_VALID_WINDOWS <= 30 and len(r_offsets) > 6 \
                and len(c_offsets) > 6:
            r_mid = r_offsets[len(r_offsets) // 2]
            c_mid = c_offsets[len(c_offsets) // 2]
            r_pick = sorted({max(0, r_mid - STEP * 2), r_mid,
                             min(r_offsets[-1], r_mid + STEP * 2)})
            c_pick = sorted({max(0, c_mid - STEP * 2), c_mid,
                             min(c_offsets[-1], c_mid + STEP * 2)})
            # 9 个中心起点向周围扩展 (6x6 = 36 网格, 保证 >=30 窗口)
            def expand(mids, offs, n):
                # mids: 三个起始行/列值; offs: 完整列表; 返回顺序优先中央
                i_mid = min(range(len(offs)),
                            key=lambda i: abs(offs[i] - mids[1]))
                picked = []
                for d in range(0, len(offs)):
                    for si in (i_mid - d, i_mid + d):
                        if 0 <= si < len(offs) and offs[si] not in picked:
                            picked.append(offs[si])
                        if len(picked) >= n:
                            return picked
                return picked
            # 只保留前 ~60 个行/列(共 ~3600 窗口,上限可控); 在 loop 内再
            # 依据 valid_with_mask 停止
            r_offsets = expand(r_pick, r_offsets, max(6, min(60, len(r_offsets))))
            c_offsets = expand(c_pick, c_offsets, max(6, min(60, len(c_offsets))))
            print(f"  [中心优先] r_offsets[:6]={r_offsets[:6]} "
                  f"c_offsets[:6]={c_offsets[:6]}")

        # ===== 2026-08-30 严格计数器(独立整数) =====
        # total_windows_seen : 实际遍历到的滑窗总数 (每进入一次内层循环+1)
        # valid_windows_seen : 通过 combined-valid 掩膜 (VALID_PIX_RATIO_MIN) 的窗口数
        # windows_with_masks : 至少产生 1 个有效 mask 的窗口数 (MAX_VALID_WINDOWS 直接对应该计数)
        # raw_candidates     : 所有 mask 候选 (已校验为 Polygon 非空, geom_err 不计入)

        def process_window_result(res, c_off, r_off, rgb):
            """处理单个窗口推理结果。返回 True 代表该窗口属于 windows_with_masks。
            - 仅使用 result.masks.xy (不使用 bbox 代替 polygon)
            - 局部像素坐标 -> tile_transform (基于 WarpedVRT 的 Window 正向 transform)
            - 输出 EPSG:32648 UTM, 严禁使用原始 WGS84 src.transform
            """
            nonlocal candidates, skip, raw_candidates
            has_valid_mask = False
            if res.masks is None:
                skip["no_mask"] += 1
                return False
            tile_tf = win_transform_fn(
                Window(c_off, r_off, TILE_SIZE, TILE_SIZE),
                vrt.transform)
            confs = res.boxes.conf.tolist() if res.boxes is not None else []
            for mi, xy in enumerate(res.masks.xy):
                if len(xy) < 3:
                    continue
                edge_flag = is_edge_instance(xy)
                try:
                    # 局部 (x, y) = 局部窗口像素坐标 (左上为原点)
                    # tile_tf = rasterio.windows.transform(window, vrt.transform)
                    # 正向乘法：(UTM_x, UTM_y) = tile_tf * (x, y)
                    utm = [tile_tf * (float(x), float(y)) for x, y in xy]
                    if len(utm) >= 2 and \
                       abs(utm[0][0] - utm[-1][0]) < 1e-6 and \
                       abs(utm[0][1] - utm[-1][1]) < 1e-6:
                        utm = utm[:-1]
                    if len(utm) < 3:
                        continue
                    poly = fix_geometry(Polygon(utm))
                    if not isinstance(poly, Polygon) or poly.is_empty \
                       or poly.area < 1.0:
                        skip["geom_err"] += 1
                        continue
                    conf = float(confs[mi]) if mi < len(confs) else 0.0
                    vc = len(poly.exterior.coords) - 1
                    area = poly.area
                    perim = poly.length
                    compact = 4 * math.pi * area / perim ** 2 if perim > 0 else 0.0
                    candidates.append({
                        "geometry": poly,
                        "confidence": round(conf, 4),
                        "area_m2": round(area, 2),
                        "perimeter": round(perim, 2),
                        "compactness": round(compact, 4),
                        "circularity": round(compact, 4),
                        "vertex_count": vc,
                        "is_edge": int(edge_flag),
                        "source_tif": tif_name,
                        "window_row": r_off,
                        "window_col": c_off,
                        "status": "kept", "drop_reason": "",
                    })
                    raw_candidates += 1
                    has_valid_mask = True
                except Exception:
                    skip["geom_err"] += 1
                    continue
            if not has_valid_mask:
                # res.masks 非 None，但 xy 全部 <3 点/几何异常，仍计 no_mask
                skip["no_mask"] += 1
            return has_valid_mask

        stop_flag = False
        # INFER_BATCH == 1: 每窗口推理后立即检查 windows_with_masks,
        # 避免批量推理 overshoot。若 INFER_BATCH>1, 也会在每批后立即判断。
        batch_imgs, batch_meta = [], []

        def flush_batch():
            """推理当前 batch, 返回本 batch 新增的 windows_with_masks 数量。"""
            nonlocal batch_imgs, batch_meta
            added = 0
            if not batch_imgs:
                return 0
            try:
                results = model.predict(
                    source=batch_imgs, conf=TIF_INFER_CONF,
                    iou=INFER_IOU, imgsz=IMG_SIZE,
                    device=DEVICE, retina_masks=RETINA_MASKS,
                    verbose=False)
            except Exception as e:
                skip["infer_err"] += len(batch_imgs)
                print(f"    [推理错误] {e}")
                batch_imgs, batch_meta = [], []
                return 0
            for res, (c_off, r_off, rgb) in zip(results, batch_meta):
                if process_window_result(res, c_off, r_off, rgb):
                    added += 1
            batch_imgs, batch_meta = [], []
            return added

        for r_off in r_offsets:
            if stop_flag:
                break
            for c_off in c_offsets:
                # 严格：达到 MAX_VALID_WINDOWS 个“含 mask 窗口”后，
                # 不再提交新的推理批次（允许当前 batch 已经提交的完成）
                if windows_with_masks >= MAX_VALID_WINDOWS:
                    stop_flag = True
                    stop_reason = "reached_MAX_VALID_WINDOWS_before_window"
                    break
                total_windows_seen += 1
                win = Window(c_off, r_off, TILE_SIZE, TILE_SIZE)
                try:
                    data = vrt.read(BANDS_RGB, window=win, masked=True)
                except Exception:
                    skip["read_err"] += 1
                    continue
                if data.shape[1] != TILE_SIZE or data.shape[2] != TILE_SIZE:
                    skip["size_mismatch"] += 1
                    continue
                # combined-valid mask (与 data_prepare.py/切片训练一致)
                valid_mask = (~np.any(data.mask, axis=0)
                              if data.mask is not np.ma.nomask
                              else np.ones((TILE_SIZE, TILE_SIZE), dtype=bool))
                if data.dtype.kind == "f":
                    valid_mask &= np.all(np.isfinite(data.data), axis=0)
                if float(valid_mask.mean()) < VALID_PIX_RATIO_MIN:
                    skip["low_valid"] += 1
                    continue
                # 合格像素窗口
                valid_windows_seen += 1
                rgb = apply_stretch_uint8(data.data, stretch, valid_mask)
                if valid_windows_seen <= 30:
                    save_png_pil(
                        rgb,
                        os.path.join(DEBUG_DIR,
                                     f"input_r{r_off:05d}_c{c_off:05d}.png"),
                    )
                batch_imgs.append(rgb)
                batch_meta.append((c_off, r_off, rgb))
                if len(batch_imgs) >= max(1, INFER_BATCH):
                    delta = flush_batch()
                    windows_with_masks += delta
                    if windows_with_masks >= MAX_VALID_WINDOWS:
                        stop_flag = True
                        stop_reason = "reached_MAX_VALID_WINDOWS_after_batch"
                        break
        # 尾 batch：若在达到上限之前停止（所有窗口扫描完），处理剩余；
        # 若已达上限则严禁再提交（严格按用户要求：达到30后不再提交新的推理批次）
        if not stop_flag and batch_imgs:
            windows_with_masks += flush_batch()

        vrt.close()

        # 最终上限判断
        reached_cap = (windows_with_masks >= MAX_VALID_WINDOWS) or (stop_flag and "reached" in stop_reason)
        if not stop_reason and reached_cap:
            stop_reason = "reached_MAX_VALID_WINDOWS_after_loop"

        print(f"\n[B.2] 严格计数器统计 (用户要求6项字段):")
        print(f"  target MAX_VALID_WINDOWS (含mask窗口上限) : {MAX_VALID_WINDOWS}")
        print(f"  实际 windows_with_masks                   : {windows_with_masks}")
        print(f"  是否严格达到上限                           : {reached_cap}"
              f"  ({stop_reason if stop_reason else 'exhausted'})")
        print(f"  total_windows_seen   (遍历窗口总数)       : {total_windows_seen}")
        print(f"  valid_windows_seen   (像素合格窗口)       : {valid_windows_seen}")
        print(f"  raw_candidates       (所有有效mask候选)   : {raw_candidates}")
        print(f"  skip 分类计数                             : {skip}")

    # NMS
    nms_kept, nms_stats = nms_dedup(candidates)
    # 质量筛选
    filtered, filter_reasons = apply_quality_filter(nms_kept)

    # 调试叠加图(UTM 空间)
    if candidates:
        g = unary_union([c["geometry"] for c in candidates
                         if isinstance(c["geometry"], Polygon)])
        xmin, ymin, xmax, ymax = g.bounds
    else:
        xmin, ymin, xmax, ymax = 0, 0, 1, 1
    save_tif_debug_overlays(tif_name, candidates, nms_kept, filtered,
                            (xmin, ymin, xmax, ymax))

    # 导出 SHP
    cols = ["geometry", "id", "confidence", "area_m2", "perimeter",
            "compactness", "circularity", "vertex_count", "is_edge",
            "source_tif", "window_row", "window_col"]

    gdf_all = build_gdf(nms_kept)
    gdf_filt = build_gdf(filtered)
    gdf_all.to_file(OUT_SHP_ALL, driver="ESRI Shapefile", encoding="utf-8")
    gdf_filt.to_file(OUT_SHP_FILTERED, driver="ESRI Shapefile", encoding="utf-8")
    print(f"  all SHP : {OUT_SHP_ALL}  (n={len(gdf_all)})")
    print(f"  filt SHP: {OUT_SHP_FILTERED}  (n={len(gdf_filt)})")

    # 最终汇报
    print("\n" + "=" * 60)
    print("[B.3] 小规模预测 最终汇报 (严格MAX_VALID_WINDOWS=windows_with_masks上限)")
    print("=" * 60)
    print(f"  TIF            : {tif_name}")
    print(f"  MAX_VALID_WINDOWS target (含mask窗口上限) : {MAX_VALID_WINDOWS}")
    print(f"  windows_with_masks (实际含mask窗口)       : {windows_with_masks}")
    print(f"  是否达到上限     : {reached_cap}  ({stop_reason if stop_reason else 'scan_exhausted'})")
    print(f"  total_windows_seen : {total_windows_seen}")
    print(f"  valid_windows_seen : {valid_windows_seen}")
    print(f"  raw_candidates     : {raw_candidates}  (=NMS前, len(candidates)={len(candidates)})")
    print(f"  NMS 前 / NMS 后    : {len(candidates)} / {len(nms_kept)}  nms_stats={nms_stats}")
    print(f"  filtered 数量      : {len(filtered)}")
    print(f"  过滤原因           : {filter_reasons}")
    print(f"  跳过窗口原因       : {skip}")
    print(f"  输出 CRS           : {TARGET_CRS}")
    print(f"  all SHP 路径       : {OUT_SHP_ALL}")
    print(f"  filt SHP 路径      : {OUT_SHP_FILTERED}")
    print(f"  debug 图目录       : {DEBUG_DIR}")


def build_gdf(poly_dicts):
    """候选字典列表 -> GeoDataFrame(EPSG:32648)。空列表返回空 GDF(带 geometry 列)。"""
    rows = []
    for idx, d in enumerate(poly_dicts):
        rows.append({
            "geometry": d["geometry"],
            "id": idx,
            "confidence": d["confidence"],
            "area_m2": d["area_m2"],
            "perimeter": d["perimeter"],
            "compactness": d["compactness"],
            "circularity": d["circularity"],
            "vertex_count": d["vertex_count"],
            "is_edge": d["is_edge"],
            "source_tif": d["source_tif"],
            "window_row": d["window_row"],
            "window_col": d["window_col"],
        })
    if not rows:
        rows = [{"geometry": Polygon(), "id": None, "confidence": None,
                 "area_m2": None, "perimeter": None,
                 "compactness": None, "circularity": None,
                 "vertex_count": None, "is_edge": None,
                 "source_tif": "", "window_row": None,
                 "window_col": None}]
        gdf = gpd.GeoDataFrame(rows, crs=TARGET_CRS, geometry="geometry")
        return gdf.iloc[0:0]
    return gpd.GeoDataFrame(rows, crs=TARGET_CRS, geometry="geometry")


def predict_full_scale(model, tif_paths):
    """全量预测: 全部 TIF, 全部窗口, 无含mask窗口上限。

    - 每幅 TIF 独立统计 total_windows_seen / valid_windows_seen /
      windows_with_masks / raw_candidates / skip
    - 6 幅 TIF 空间两两重叠 => NMS 在全部候选上全局执行一次,
      跨 TIF 重复实例被 IoU>=0.70 去重 (不 unary_union / 不 dissolve)
    - filtered 仅按 conf/面积/圆度过滤, 不修改 all 结果
    """
    os.makedirs(DEBUG_DIR, exist_ok=True)
    os.makedirs(PRED_OUT_DIR, exist_ok=True)

    all_candidates = []
    per_tif_reports = []

    for tif_idx, tif_path in enumerate(tif_paths):
        tif_name = os.path.basename(tif_path)
        print(f"\n[FULL {tif_idx + 1}/{len(tif_paths)}] {tif_name}")

        # ---- 每幅 TIF 独立计数器 ----
        total_windows_seen = 0
        valid_windows_seen = 0
        windows_with_masks = 0
        raw_candidates = 0
        skip = {"low_valid": 0, "read_err": 0, "size_mismatch": 0,
                "no_mask": 0, "geom_err": 0, "infer_err": 0}
        tif_candidates = []

        with rasterio.open(tif_path) as src:
            vrt = build_vrt(src)
            print(f"  src CRS={src.crs}, size={src.width}x{src.height}")
            print(f"  vrt={vrt.width}x{vrt.height}, crs={vrt.crs}, "
                  f"res={abs(vrt.transform.a):.2f}m")
            print("  [计算全局拉伸分位数]")
            stretch = compute_global_stretch(vrt, BANDS_RGB)
            for i, (lo, hi) in enumerate(stretch):
                print(f"    {'RGB'[i]}: {CLIP_PCT[0]}%={lo:.2f}, "
                      f"{CLIP_PCT[1]}%={hi:.2f}")

            H, W = vrt.height, vrt.width
            r_offsets = list(range(0, H - TILE_SIZE + 1, STEP))
            c_offsets = list(range(0, W - TILE_SIZE + 1, STEP))
            n_win_grid = len(r_offsets) * len(c_offsets)
            print(f"  窗口网格: {len(r_offsets)} x {len(c_offsets)} "
                  f"= {n_win_grid} (TILE={TILE_SIZE}, STEP={STEP})")

            def process_one(res, c_off, r_off):
                """单窗口结果处理: masks.xy -> tile_transform 正向 -> UTM。"""
                nonlocal windows_with_masks, raw_candidates
                if res.masks is None:
                    skip["no_mask"] += 1
                    return
                tile_tf = win_transform_fn(
                    Window(c_off, r_off, TILE_SIZE, TILE_SIZE),
                    vrt.transform)
                confs = (res.boxes.conf.tolist()
                         if res.boxes is not None else [])
                any_valid = False
                for mi, xy in enumerate(res.masks.xy):
                    if len(xy) < 3:
                        continue
                    edge_flag = is_edge_instance(xy)
                    try:
                        utm = [tile_tf * (float(x), float(y))
                               for x, y in xy]
                        if len(utm) >= 2 and \
                           abs(utm[0][0] - utm[-1][0]) < 1e-6 and \
                           abs(utm[0][1] - utm[-1][1]) < 1e-6:
                            utm = utm[:-1]
                        if len(utm) < 3:
                            continue
                        poly = fix_geometry(Polygon(utm))
                        if not isinstance(poly, Polygon) or poly.is_empty \
                           or poly.area < 1.0:
                            skip["geom_err"] += 1
                            continue
                        conf = float(confs[mi]) if mi < len(confs) else 0.0
                        vc = len(poly.exterior.coords) - 1
                        area = poly.area
                        perim = poly.length
                        compact = (4 * math.pi * area / perim ** 2
                                   if perim > 0 else 0.0)
                        tif_candidates.append({
                            "geometry": poly,
                            "confidence": round(conf, 4),
                            "area_m2": round(area, 2),
                            "perimeter": round(perim, 2),
                            "compactness": round(compact, 4),
                            "circularity": round(compact, 4),
                            "vertex_count": vc,
                            "is_edge": int(edge_flag),
                            "source_tif": tif_name,
                            "window_row": r_off,
                            "window_col": c_off,
                            "status": "kept", "drop_reason": "",
                        })
                        raw_candidates += 1
                        any_valid = True
                    except Exception:
                        skip["geom_err"] += 1
                        continue
                if any_valid:
                    windows_with_masks += 1
                else:
                    skip["no_mask"] += 1

            batch_imgs, batch_meta = [], []
            win_done = 0

            def flush_batch():
                nonlocal batch_imgs, batch_meta
                if not batch_imgs:
                    return
                try:
                    results = model.predict(
                        source=batch_imgs, conf=TIF_INFER_CONF,
                        iou=INFER_IOU, imgsz=IMG_SIZE,
                        device=DEVICE, retina_masks=RETINA_MASKS,
                        verbose=False)
                    for res, (c2, r2) in zip(results, batch_meta):
                        process_one(res, c2, r2)
                except Exception as e:
                    skip["infer_err"] += len(batch_imgs)
                    print(f"    [推理错误] {e}")
                batch_imgs, batch_meta = [], []

            for r_off in r_offsets:
                for c_off in c_offsets:
                    total_windows_seen += 1
                    win = Window(c_off, r_off, TILE_SIZE, TILE_SIZE)
                    try:
                        data = vrt.read(BANDS_RGB, window=win, masked=True)
                    except Exception:
                        skip["read_err"] += 1
                        continue
                    if data.shape[1] != TILE_SIZE or \
                       data.shape[2] != TILE_SIZE:
                        skip["size_mismatch"] += 1
                        continue
                    # combined-valid mask (与 data_prepare.py 一致)
                    valid_mask = (~np.any(data.mask, axis=0)
                                  if data.mask is not np.ma.nomask
                                  else np.ones((TILE_SIZE, TILE_SIZE),
                                               dtype=bool))
                    if data.dtype.kind == "f":
                        valid_mask &= np.all(np.isfinite(data.data), axis=0)
                    if float(valid_mask.mean()) < VALID_PIX_RATIO_MIN:
                        skip["low_valid"] += 1
                        continue
                    valid_windows_seen += 1
                    rgb = apply_stretch_uint8(data.data, stretch, valid_mask)
                    # 每幅 TIF 仅保存前 8 张输入样例 (避免上千 PNG)
                    if valid_windows_seen <= 8:
                        save_png_pil(
                            rgb,
                            os.path.join(
                                DEBUG_DIR,
                                f"input_{tif_idx + 1:02d}_r{r_off:05d}"
                                f"_c{c_off:05d}.png"))
                    batch_imgs.append(rgb)
                    batch_meta.append((c_off, r_off))
                    if len(batch_imgs) >= max(1, INFER_BATCH):
                        flush_batch()
                    win_done += 1
                    if win_done % 200 == 0:
                        print(f"    进度 {win_done}/{n_win_grid} "
                              f"with_mask={windows_with_masks} "
                              f"raw={raw_candidates}")
            flush_batch()
            vrt.close()

        all_candidates.extend(tif_candidates)
        per_tif_reports.append({
            "tif": tif_name,
            "total_windows_seen": total_windows_seen,
            "valid_windows_seen": valid_windows_seen,
            "windows_with_masks": windows_with_masks,
            "raw_candidates": raw_candidates,
            "skip": dict(skip),
        })
        print(f"  [TIF统计] total_windows_seen={total_windows_seen} "
              f"valid_windows_seen={valid_windows_seen} "
              f"windows_with_masks={windows_with_masks} "
              f"raw_candidates={raw_candidates}")
        print(f"  [TIF统计] skip={skip}")

    # ---- 全局 NMS + 质量过滤 (跨 TIF 去重) ----
    nms_kept, nms_stats = nms_dedup(all_candidates)
    filtered, filter_reasons = apply_quality_filter(nms_kept)

    # 每幅 TIF 的 NMS 后 / 过滤后数量 (按 source_tif 分组;
    # 跨 TIF 重复只保留置信度更高一侧, 故归属以保留实例为准)
    per_tif_nms, per_tif_filt = {}, {}
    for c in nms_kept:
        per_tif_nms[c["source_tif"]] = per_tif_nms.get(c["source_tif"], 0) + 1
    for c in filtered:
        per_tif_filt[c["source_tif"]] = per_tif_filt.get(c["source_tif"], 0) + 1

    # ---- debug 叠加图: 每幅 TIF 一套 (raw / NMS 子集 / filtered 子集) ----
    for tif_idx, tif_path in enumerate(tif_paths):
        tif_name = os.path.basename(tif_path)
        raw_sub = [c for c in all_candidates if c["source_tif"] == tif_name]
        nms_sub = [c for c in nms_kept if c["source_tif"] == tif_name]
        filt_sub = [c for c in filtered if c["source_tif"] == tif_name]
        if raw_sub:
            g = unary_union([c["geometry"] for c in raw_sub
                             if isinstance(c["geometry"], Polygon)])
            bounds = g.bounds
        else:
            bounds = (0, 0, 1, 1)
        save_tif_debug_overlays(tif_name, raw_sub, nms_sub, filt_sub, bounds)

    # ---- 导出 SHP ----
    gdf_all = build_gdf(nms_kept)
    gdf_filt = build_gdf(filtered)
    gdf_all.to_file(OUT_SHP_ALL, driver="ESRI Shapefile", encoding="utf-8")
    gdf_filt.to_file(OUT_SHP_FILTERED, driver="ESRI Shapefile",
                     encoding="utf-8")

    # ---- 最终汇报 ----
    print("\n" + "=" * 60)
    print("[FULL] 全量预测 最终汇报")
    print("=" * 60)
    for rep in per_tif_reports:
        t = rep["tif"]
        print(f"  {t}")
        print(f"    total_windows_seen = {rep['total_windows_seen']}")
        print(f"    valid_windows_seen = {rep['valid_windows_seen']}")
        print(f"    windows_with_masks = {rep['windows_with_masks']}")
        print(f"    raw_candidates     = {rep['raw_candidates']}")
        print(f"    nms_kept(该TIF保留) = {per_tif_nms.get(t, 0)}")
        print(f"    filtered(该TIF保留) = {per_tif_filt.get(t, 0)}")
        print(f"    skip = {rep['skip']}")
    tot_tw = sum(r["total_windows_seen"] for r in per_tif_reports)
    tot_vw = sum(r["valid_windows_seen"] for r in per_tif_reports)
    tot_wm = sum(r["windows_with_masks"] for r in per_tif_reports)
    tot_raw = sum(r["raw_candidates"] for r in per_tif_reports)
    tot_skip = {}
    for r in per_tif_reports:
        for k2, v2 in r["skip"].items():
            tot_skip[k2] = tot_skip.get(k2, 0) + v2
    print(f"  [总计] TIF数={len(per_tif_reports)}")
    print(f"    total_windows_seen = {tot_tw}")
    print(f"    valid_windows_seen = {tot_vw}")
    print(f"    windows_with_masks = {tot_wm}")
    print(f"    raw_candidates     = {tot_raw} (NMS 前)")
    print(f"    NMS 前 -> 后       : {len(all_candidates)} -> {len(nms_kept)}"
          f"  stats={nms_stats}")
    print(f"    filtered           : {len(filtered)}  原因={filter_reasons}")
    print(f"    skip 总计          : {tot_skip}")
    print(f"    输出 CRS           : {TARGET_CRS}")
    print(f"    all SHP : {OUT_SHP_ALL}  (n={len(gdf_all)})")
    print(f"    filt SHP: {OUT_SHP_FILTERED}  (n={len(gdf_filt)})")
    print(f"    debug 图: {DEBUG_DIR}")


def save_tif_debug_overlays(tif_name, raw, nms, filt, bounds):
    out_dir = os.path.join(DEBUG_DIR, tif_name.replace(".tif", ""))
    os.makedirs(out_dir, exist_ok=True)
    xmin, ymin, xmax, ymax = bounds

    def draw(ax, polydicts, title, cmap, label_conf=False):
        if polydicts:
            if cmap == "single":
                colors_arr = np.array([plt.cm.tab20(i % 20) for i in range(len(polydicts))])
            elif cmap == "edgecolor":
                colors_arr = np.array([(0, 0, 1, 1) if d["is_edge"] == 0
                                       else (1, 0.65, 0, 1) for d in polydicts])
            else:  # filtered green
                colors_arr = np.array([(0, 0.6, 0.2, 1)] * len(polydicts))
            pc = PolyCollection(
                [list(d["geometry"].exterior.coords) for d in polydicts],
                facecolors="none",
                edgecolors=colors_arr,
                linewidths=0.6,
            )
            ax.add_collection(pc)
            if label_conf:
                for d in polydicts:
                    c = d["geometry"].centroid
                    ax.text(c.x, c.y, f"{d['confidence']:.2f}",
                            fontsize=5, ha="center", color="red")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    fig, ax = plt.subplots(figsize=(10, 10), dpi=110)
    draw(ax, raw, f"Raw candidates (n={len(raw)})", "single")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "1_raw_candidates.png"), dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 10), dpi=110)
    draw(ax, nms, f"After NMS (n={len(nms)})  [blue=non-edge, orange=edge]", "edgecolor")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "2_after_nms.png"), dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 10), dpi=110)
    draw(ax, filt, f"Filtered (n={len(filt)})", "green", label_conf=True)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "3_filtered.png"), dpi=110)
    plt.close(fig)
    print(f"  调试图: {out_dir}")


# =====================================================================
# 入口
# =====================================================================
def main():
    print("=" * 70)
    print("06_predict_to_gis_v3.py — test 集评估 + 1 幅 TIF 小规模预测")
    print("=" * 70)
    print(f"  模型权重       : {MODEL_PATH}")
    print(f"  test 集        : {TEST_IMG_DIR}")
    print(f"  test 输出根目录: {TEST_OUT_ROOT}")
    print(f"  目标 CRS       : EPSG:32648")
    print(f"  测试阈值       : {CONF_LIST}")
    print(f"  TIF 目录       : {TIF_DIR}")
    print(f"  预测 SHP       : {OUT_SHP_ALL} / {OUT_SHP_FILTERED}")
    print(f"  TIF 推理 conf  : {TIF_INFER_CONF}")
    print(f"  NMS IoU        : {DEDUP_IOU_THRESH}")
    print(f"  过滤 conf      : {FILTER_CONF}, 面积 {FILTER_MIN_AREA}~"
          f"{FILTER_MAX_AREA} m2, 圆度 >= {FILTER_MIN_COMPACT}")
    print(f"  MAX_TIFS       : {MAX_TIFS}")
    print(f"  MAX_VALID_WINS : {MAX_VALID_WINDOWS}")
    print()

    # --- 加载模型 ---
    if not os.path.isfile(MODEL_PATH):
        print(f"[错误] 模型不存在: {MODEL_PATH}")
        sys.exit(1)
    print("[模型] 加载 YOLO11...")
    model = YOLO(MODEL_PATH)
    print(f"  model.task = {model.task}")
    assert model.task == "segment", f"模型不是分割任务: {model.task}"

    # --- 阶段 A (strict30 诊断: 若4份阈值report均已存在且含对比行, 直接跳过避免重复重推理) ---
    print("\n==================== 阶段 A: test 集评估 "
          "====================")
    stage_a_skip_ok = True
    for t in CONF_LIST:
        rep = os.path.join(TEST_OUT_ROOT, dir_name_for_conf(t), "report.txt")
        if not os.path.isfile(rep):
            stage_a_skip_ok = False
            break
        with open(rep, "r", encoding="utf-8") as fh:
            content = fh.read()
        if f"threshold={t}" not in content or "TP=" not in content:
            stage_a_skip_ok = False
            break
    if stage_a_skip_ok:
        print("[跳过] 4 份 test 集评估 report 已存在且可解析; 避免重复推理。")
    else:
        pairs = load_test_images_labels()
        _summary = save_test_outputs(pairs, model)

    # --- 阶段 B ---
    print("\n==================== 阶段 B: TIF 滑窗预测 "
          "====================")
    # 最终参数确认 (用户要求执行前打印)
    print("[最终参数确认]")
    print(f"  FULL_MODE         = {FULL_MODE}")
    print(f"  MAX_TIFS          = {MAX_TIFS}")
    print(f"  MAX_VALID_WINDOWS = {MAX_VALID_WINDOWS}")
    print(f"  INFER_BATCH       = {INFER_BATCH}")
    print(f"  TIF_INFER_CONF    = {TIF_INFER_CONF}")
    print(f"  imgsz             = {IMG_SIZE}")
    print(f"  iou               = {INFER_IOU}")
    print(f"  device            = {DEVICE}")
    print(f"  retina_masks      = {RETINA_MASKS}")

    tif_paths = find_tifs(TIF_DIR)
    if not tif_paths:
        print("[错误] 未找到 TIF, 阶段 B 跳过")
        return
    if MAX_TIFS is not None:
        tif_paths = tif_paths[:MAX_TIFS]
    print(f"  待处理 TIF 数量   : {len(tif_paths)}")
    for p in tif_paths:
        print(f"    - {os.path.basename(p)}")
    if FULL_MODE:
        predict_full_scale(model, tif_paths)
    else:
        predict_small_scale(model, tif_paths)

    print("\n[完成] 预测流程结束。")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
