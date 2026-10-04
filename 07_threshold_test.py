# -*- coding: utf-8 -*-
"""
07_threshold_test.py — 置信度阈值对比测试
================================================================
对同一幅 TIF, 分别按 CONF_LIST = [0.35, 0.50, 0.65, 0.75] 输出:
  1. mask 填充图 (UTM 空间)
  2. mask 红色边界图 (UTM 空间)
  3. GeoJSON (CRS=EPSG:32648)
  4. 预测实例数量
  5. 各置信度区间的实例数量

实现说明:
  - 单次以 conf=0.05 推理, 再按各阈值过滤实例。
    依据: NMS 按置信度贪心保留, 高阈值下存活的实例在低阈值下必然同样
    存活, 因此 conf=0.05 结果按 conf>=T 过滤与直接以 conf=T 推理等价。
  - 完全复用 06_predict_to_gis_v2.py 的 build_vrt / 全局拉伸 /
    fix_geometry / tile_transform 正向映射, 不修改任何坐标逻辑。
  - 边界来源: result.masks.xy, 禁止检测框或传统边缘检测。
================================================================
"""

import os
import sys
import time
import importlib.util
import traceback

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.windows import transform as win_transform_fn
from shapely.geometry import Polygon
import geopandas as gpd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection

# -----------------------------------------------------------------
# 导入 v2 模块(保证 VRT/拉伸/几何修复逻辑与 v2 完全一致)
# -----------------------------------------------------------------
V2_PATH = r"E:\260827YOLORUN\06_predict_to_gis_v2.py"
spec = importlib.util.spec_from_file_location("pred_v2", V2_PATH)
v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2)

from ultralytics import YOLO

# -----------------------------------------------------------------
# 配置
# -----------------------------------------------------------------
TIF_DIR    = r"E:\260827YOLORUN\GEE_Export_2025"
MODEL_PATH = r"E:\260827YOLORUN\runs\segment\ordos_farmland_v1\weights\best.pt"
OUT_ROOT   = r"E:\260827YOLORUN\predict_results\threshold_test"

# 测试 TIF; None = 取目录第一幅
TIF_NAME = "Ordos_S2_MultiBand_2025-0000000000-0000000000.tif"

CONF_LIST = [0.35, 0.50, 0.65, 0.75]
RUN_CONF  = 0.05          # 单次低阈值推理, 之后过滤
IOU       = 0.50
IMGSZ     = 640
DEVICE    = 0
RETINA    = True
BATCH     = 4

# 置信度区间统计
INTERVALS = [(0.35, 0.50), (0.50, 0.65), (0.65, 0.75), (0.75, 1.01)]

PRINT_EVERY = 20


def conf_dir_name(t):
    return os.path.join(OUT_ROOT, f"conf_{int(round(t*100)):03d}")


def collect_instances(vrt, model, stretch, tif_name):
    """滑窗推理一次(conf=RUN_CONF), 返回 UTM 实例列表"""
    H, W = vrt.height, vrt.width
    r_offs = list(range(0, H - v2.TILE_SIZE + 1, v2.STEP))
    c_offs = list(range(0, W - v2.TILE_SIZE + 1, v2.STEP))
    n_total = len(r_offs) * len(c_offs)
    print(f"  [滑窗] VRT {W}x{H}, 窗口总数 {n_total}")

    instances = []          # dict: geometry, confidence, window_row, window_col
    stats = {"low_valid": 0, "read_err": 0, "size_err": 0,
             "no_mask": 0, "geom_err": 0, "infer_err": 0}
    n_win = 0
    batch_imgs, batch_meta = [], []

    def flush():
        nonlocal batch_imgs, batch_meta, stats
        if not batch_imgs:
            return
        try:
            results = model.predict(
                source=batch_imgs, conf=RUN_CONF, iou=IOU,
                imgsz=IMGSZ, device=DEVICE, retina_masks=RETINA,
                verbose=False,
            )
            for res, (c_off, r_off) in zip(results, batch_meta):
                if res.masks is None:
                    stats["no_mask"] += 1
                    continue
                tile_tf = win_transform_fn(
                    Window(c_off, r_off, v2.TILE_SIZE, v2.TILE_SIZE),
                    vrt.transform)
                confs = res.boxes.conf.tolist()
                for mi, xy in enumerate(res.masks.xy):
                    if len(xy) < 3:
                        continue
                    utm = [tile_tf * (float(x), float(y)) for x, y in xy]
                    if len(utm) >= 2 and abs(utm[0][0]-utm[-1][0]) < 1e-6 \
                       and abs(utm[0][1]-utm[-1][1]) < 1e-6:
                        utm = utm[:-1]
                    if len(utm) < 3:
                        continue
                    try:
                        poly = v2.fix_geometry(Polygon(utm))
                        if not isinstance(poly, Polygon) or poly.is_empty \
                           or poly.area < 1.0:
                            stats["geom_err"] += 1
                            continue
                        conf = float(confs[mi]) if mi < len(confs) else 0.0
                        instances.append({
                            "geometry": poly,
                            "confidence": round(conf, 4),
                            "window_row": r_off,
                            "window_col": c_off,
                        })
                    except Exception:
                        stats["geom_err"] += 1
        except Exception as e:
            print(f"    [推理错误] {e}")
            traceback.print_exc()
            stats["infer_err"] += 1
        batch_imgs, batch_meta = [], []

    for r_off in r_offs:
        for c_off in c_offs:
            n_win += 1
            if n_win % PRINT_EVERY == 0 or n_win == n_total:
                print(f"    窗口 {n_win}/{n_total}  实例 {len(instances)}  {stats}")
            win = Window(c_off, r_off, v2.TILE_SIZE, v2.TILE_SIZE)
            try:
                data = vrt.read(v2.BANDS_RGB, window=win, masked=True)
            except Exception:
                stats["read_err"] += 1
                continue
            if data.shape[1] != v2.TILE_SIZE or data.shape[2] != v2.TILE_SIZE:
                stats["size_err"] += 1
                continue
            if data.mask is np.ma.nomask:
                valid = np.ones((v2.TILE_SIZE, v2.TILE_SIZE), dtype=bool)
            else:
                valid = ~np.any(data.mask, axis=0)
            if data.dtype.kind == "f":
                valid &= np.all(np.isfinite(data.data), axis=0)
            if float(valid.mean()) < v2.VALID_PIX_RATIO_MIN:
                stats["low_valid"] += 1
                continue
            rgb = v2.apply_stretch_uint8(data.data, stretch, valid)
            batch_imgs.append(rgb)
            batch_meta.append((c_off, r_off))
            if len(batch_imgs) >= BATCH:
                flush()
    flush()

    print(f"  [推理完成] 实例总数(conf>={RUN_CONF}): {len(instances)}, {stats}")
    return instances


def save_threshold_outputs(instances, thresh, tif_base):
    """按阈值过滤并输出填充图/边界图/GeoJSON/统计"""
    out_dir = conf_dir_name(thresh)
    os.makedirs(out_dir, exist_ok=True)

    sel = [d for d in instances if d["confidence"] >= thresh]
    print(f"  [conf>={thresh:.2f}] 实例数: {len(sel)}")

    # --- GeoJSON ---
    if sel:
        gdf = gpd.GeoDataFrame(
            {
                "id": list(range(len(sel))),
                "confidence": [d["confidence"] for d in sel],
                "area_m2": [round(d["geometry"].area, 2) for d in sel],
                "window_row": [d["window_row"] for d in sel],
                "window_col": [d["window_col"] for d in sel],
            },
            geometry=[d["geometry"] for d in sel],
            crs=v2.TARGET_CRS,
        )
        gdf.to_file(os.path.join(out_dir, "instances.geojson"),
                    driver="GeoJSON")
    else:
        gdf = gpd.GeoDataFrame(columns=["id", "confidence", "area_m2",
                                        "window_row", "window_col"],
                               geometry=[], crs=v2.TARGET_CRS)

    # --- 绘图范围 ---
    with rasterio.open(os.path.join(TIF_DIR, tif_base)) as src:
        vrt_tmp = v2.build_vrt(src)
        bounds = vrt_tmp.bounds
        vrt_tmp.close()
    xmin, ymin, xmax, ymax = bounds

    geoms = [d["geometry"] for d in sel]
    confs = [d["confidence"] for d in sel]

    # --- 1. mask 填充图 ---
    fig, ax = plt.subplots(figsize=(14, 12), dpi=140)
    if geoms:
        pc = PolyCollection(
            [g.exterior.coords[:] for g in geoms],
            facecolors=plt.cm.YlGn(
                [(c - 0.35) / 0.65 for c in confs]),
            edgecolors="none", alpha=0.75)
        ax.add_collection(pc)
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_title(f"{tif_base}  conf>={thresh:.2f}  filled masks "
                 f"(n={len(sel)}, color=confidence)")
    ax.set_xlabel("Easting (m)"); ax.set_ylabel("Northing (m)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "1_masks_filled.png"), dpi=140)
    plt.close(fig)

    # --- 2. mask 红色边界图 ---
    fig, ax = plt.subplots(figsize=(14, 12), dpi=140)
    if geoms:
        lc = LineCollection(
            [g.exterior.coords[:] for g in geoms],
            colors="red", linewidths=0.4)
        ax.add_collection(lc)
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_title(f"{tif_base}  conf>={thresh:.2f}  red boundaries (n={len(sel)})")
    ax.set_xlabel("Easting (m)"); ax.set_ylabel("Northing (m)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "2_masks_red_boundary.png"), dpi=140)
    plt.close(fig)

    # --- 4/5. 数量统计 ---
    n_total = len(sel)
    interval_counts = {}
    for lo, hi in INTERVALS:
        interval_counts[f"{lo:.2f}-{hi:.2f}"] = int(
            sum(1 for c in confs if lo <= c < hi))
    below = int(sum(1 for c in confs if c < 0.35))

    areas = [d["geometry"].area for d in sel] if sel else [0.0]
    comps = []
    for d in sel:
        p = d["geometry"]
        comps.append(4*np.pi*p.area/p.length**2 if p.length > 0 else 0.0)

    lines = []
    lines.append(f"TIF: {tif_base}")
    lines.append(f"conf>={thresh:.2f}, iou={IOU}, imgsz={IMGSZ}, retina_masks={RETINA}")
    lines.append(f"预测实例数量: {n_total}")
    lines.append(f"各置信度区间实例数量:")
    lines.append(f"  <0.35          : {below}")
    for k, v_ in interval_counts.items():
        lines.append(f"  {k:<15}: {v_}")
    if sel:
        lines.append(f"面积 m2  : min={min(areas):.0f} max={max(areas):.0f} "
                     f"mean={np.mean(areas):.0f} median={np.median(areas):.0f}")
        lines.append(f"圆度     : min={min(comps):.3f} max={max(comps):.3f} "
                     f"mean={np.mean(comps):.3f}")
    lines.append(f"CRS: EPSG:32648")
    rpt = os.path.join(out_dir, "report.txt")
    with open(rpt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))

    return {"threshold": thresh, "n": n_total,
            "interval_counts": interval_counts, "below": below,
            "mean_area": float(np.mean(areas)) if sel else 0.0,
            "mean_comp": float(np.mean(comps)) if sel else 0.0,
            "geojson": os.path.join(out_dir, "instances.geojson")}


def main():
    print("=" * 64)
    print("07_threshold_test.py — 置信度阈值对比测试")
    print("=" * 64)

    if not os.path.isfile(MODEL_PATH):
        print(f"[错误] 模型不存在: {MODEL_PATH}"); sys.exit(1)

    tif_path = os.path.join(TIF_DIR, TIF_NAME) if TIF_NAME \
        else v2.find_tifs(TIF_DIR)[0]
    tif_base = os.path.basename(tif_path)
    if not os.path.isfile(tif_path):
        print(f"[错误] TIF 不存在: {tif_path}"); sys.exit(1)
    os.makedirs(OUT_ROOT, exist_ok=True)

    print(f"[1] 加载模型: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print(f"    model.task = {model.task}")
    assert model.task == "segment", "模型不是分割模型!"

    print(f"\n[2] 处理 TIF: {tif_base}")
    with rasterio.open(tif_path) as src:
        vrt = v2.build_vrt(src)
        print(f"    VRT: {vrt.width}x{vrt.height}, crs={vrt.crs}")
        print(f"    [计算全局拉伸]")
        stretch = v2.compute_global_stretch(vrt, v2.BANDS_RGB)
        for i, (lo, hi) in enumerate(stretch):
            print(f"      {'RGB'[i]}: {lo:.1f} ~ {hi:.1f}")

        t0 = time.time()
        instances = collect_instances(vrt, model, stretch, tif_base)
        vrt.close()
    print(f"    推理用时 {time.time()-t0:.1f}s")

    print(f"\n[3] 按阈值分别输出")
    summary = []
    for t in CONF_LIST:
        print(f"\n--- conf>={t:.2f} ---")
        summary.append(save_threshold_outputs(instances, t, tif_base))

    print(f"\n[4] 四档阈值对比汇总")
    print(f"{'阈值':>6} | {'实例数':>6} | {'<0.35':>6} | "
          f"{'0.35-0.50':>10} | {'0.50-0.65':>10} | "
          f"{'0.65-0.75':>10} | {'>=0.75':>7} | {'平均面积':>10} | {'平均圆度':>8}")
    for s in summary:
        ic = s["interval_counts"]
        print(f"{s['threshold']:>6.2f} | {s['n']:>6} | {s['below']:>6} | "
              f"{ic['0.35-0.50']:>10} | {ic['0.50-0.65']:>10} | "
              f"{ic['0.65-0.75']:>10} | {ic['0.75-1.01']:>7} | "
              f"{s['mean_area']:>10.0f} | {s['mean_comp']:>8.3f}")

    print(f"\n输出目录: {OUT_ROOT}")
    print("请人工对比各阈值目录下的 1_masks_filled.png / 2_masks_red_boundary.png,")
    print("评估: 误检数量 / 漏检数量 / 正确圆形农田数量 / 边界质量。")
    print("=" * 64)


if __name__ == "__main__":
    main()
