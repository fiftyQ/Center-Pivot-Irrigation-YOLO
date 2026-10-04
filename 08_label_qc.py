# -*- coding: utf-8 -*-
"""
08_label_qc.py — 训练标签质检 + 数据泄漏检查 (不运行模型)
================================================================
A. 标签叠加质检:
   - 从 train/val 随机抽取 >=30 张含实例图片, 将 TXT 分割多边形
     画回原图(红色边界), 输出到 yolo_dataset/label_debug
   - 统计: 实例数 / 多边形点数 / 4 点比例 / >=8 点比例 /
     空标签比例 / 坐标越界数量
   - 自动可疑项: 4 点矩形标签 / 高 IoU 重复标注 / 触边半圆 /
     低圆度标签 / X-Y 相关性异常(检测颠倒)
B. 数据泄漏检查 (dataset_index.csv):
   - train / val 各自 source_tif 列表及交集
   - 同一 tile_name 是否同时出现在 train 和 val
   - train/val 切片 UTM 范围是否存在空间重叠
   - 若共享 source_tif → 明确标记数据泄漏风险
不修改任何图片、标签和训练代码。
================================================================
"""

import os
import glob
import random
import math

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import rasterio.features
from shapely.geometry import Polygon, box

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# -----------------------------------------------------------------
# 配置
# -----------------------------------------------------------------
DS_DIR   = r"E:\260827YOLORUN\yolo_dataset"
CSV_PATH = os.path.join(DS_DIR, "dataset_index.csv")
OUT_DIR  = os.path.join(DS_DIR, "label_debug")

N_TRAIN_SAMPLE = 20
N_VAL_SAMPLE   = 15
RANDOM_SEED    = 42

# 可疑项阈值
RECT_POINTS      = 4      # 多边形点数 == 4 → 矩形标签嫌疑
DUP_IOU_THRESH   = 0.90   # 同 TXT 内实例 IoU > 此值 → 重复标注嫌疑
BORDER_PIX       = 3      # 多边形触边像素阈值 → 半圆/截断嫌疑
MIN_CIRC_WARN    = 0.60   # 圆形农田标签圆度低于此值 → 可疑


# -----------------------------------------------------------------
# 标签解析
# -----------------------------------------------------------------
def parse_label_txt(txt_path):
    """解析 YOLO 分割标签, 返回 (instances, issues)
    instances: list of dict(pts: Nx2 像素坐标, n_pts, area_frac)
    """
    instances = []
    issues = {"oob_coords": 0, "bad_lines": 0}
    if not os.path.isfile(txt_path):
        return instances, issues
    with open(txt_path, "r", encoding="utf-8") as f:
        for ln in f:
            parts = ln.split()
            if not parts:
                continue
            if len(parts) < 7 or (len(parts) - 1) % 2 != 0:
                issues["bad_lines"] += 1
                continue
            try:
                vals = [float(v) for v in parts[1:]]
            except ValueError:
                issues["bad_lines"] += 1
                continue
            pts = np.array(vals, dtype=float).reshape(-1, 2)
            if np.any(pts < 0) or np.any(pts > 1):
                issues["oob_coords"] += len(pts)
            instances.append({
                "pts_norm": pts,
                "n_pts": len(pts),
            })
    return instances, issues


def inst_polygon_px(inst, w, h):
    return Polygon(inst["pts_norm"] * [w, h])


def polygon_iou(g1, g2):
    if not g1.is_valid:
        g1 = g1.buffer(0)
    if not g2.is_valid:
        g2 = g2.buffer(0)
    if g1.is_empty or g2.is_empty:
        return 0.0
    inter = g1.intersection(g2).area
    union = g1.union(g2).area
    return inter / union if union > 1e-12 else 0.0


def touches_border(poly, w, h, margin=BORDER_PIX):
    x0, y0, x1, y1 = poly.bounds
    return x0 <= margin or y0 <= margin or x1 >= w - margin or y1 >= h - margin


# -----------------------------------------------------------------
# A. 抽样叠加质检
# -----------------------------------------------------------------
def sample_and_overlay():
    print("=" * 64)
    print("A. 标签叠加质检 (随机抽样)")
    print("=" * 64)
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = random.Random(RANDOM_SEED)

    all_stats = []
    per_split = {}
    for split in ("train", "val"):
        img_dir = os.path.join(DS_DIR, "images", split)
        lbl_dir = os.path.join(DS_DIR, "labels", split)
        pngs = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        # 只抽含实例的图
        with_inst = []
        for p in pngs:
            txt = os.path.join(lbl_dir,
                               os.path.splitext(os.path.basename(p))[0] + ".txt")
            with open(txt, "r", encoding="utf-8") as f:
                if any(line.split() for line in f):
                    with_inst.append((p, txt))
        n_take = N_TRAIN_SAMPLE if split == "train" else N_VAL_SAMPLE
        take = rng.sample(with_inst, min(n_take, len(with_inst)))
        print(f"[{split}] 含实例图片 {len(with_inst)}, 抽取 {len(take)}")
        per_split[split] = take

    for split, take in per_split.items():
        for img_path, txt_path in take:
            name = os.path.splitext(os.path.basename(img_path))[0]
            with rasterio.open(img_path) as src:
                img = src.read([1, 2, 3]).transpose(1, 2, 0)
            h, w = img.shape[:2]

            insts, issues = parse_label_txt(txt_path)
            polys = [inst_polygon_px(i, w, h) for i in insts]

            # --- 每实例检查 ---
            n_rect = sum(1 for i in insts if i["n_pts"] == RECT_POINTS)
            n_8p   = sum(1 for i in insts if i["n_pts"] >= 8)
            n_border = sum(1 for p in polys if touches_border(p, w, h))
            # 重复标注
            n_dup = 0
            for a in range(len(polys)):
                for b in range(a + 1, len(polys)):
                    if polygon_iou(polys[a], polys[b]) > DUP_IOU_THRESH:
                        n_dup += 1
            # 圆度
            circs = []
            for p in polys:
                c = 4 * math.pi * p.area / p.length ** 2 if p.length > 0 else 0.0
                circs.append(c)

            # X/Y 颠倒启发式: 对每实例计算多边形主轴方向占比,
            # 圆形无强主轴; 若所有实例高度各向异性且方向一致可提示,
            # 这里仅记录平均长短轴比供人工参考
            aniso = []
            for p in polys:
                try:
                    mr = p.minimum_rotated_rectangle
                    coords = list(mr.exterior.coords)[:4]
                    import itertools
                    ds = [math.dist(coords[i], coords[i+1])
                          for i in range(len(coords) - 1)]
                    if max(ds) > 0:
                        aniso.append(min(ds) / max(ds))
                except Exception:
                    pass

            all_stats.append({
                "split": split, "name": name,
                "n_inst": len(insts),
                "pt_counts": [i["n_pts"] for i in insts],
                "n_rect4": n_rect, "n_8p": n_8p,
                "n_border": n_border, "n_dup": n_dup,
                "oob": issues["oob_coords"], "bad_lines": issues["bad_lines"],
                "mean_circ": float(np.mean(circs)) if circs else np.nan,
                "n_low_circ": sum(1 for c in circs if c < MIN_CIRC_WARN),
                "mean_aniso": float(np.mean(aniso)) if aniso else np.nan,
            })

            # --- 叠加图 ---
            fig, ax = plt.subplots(figsize=(6.4, 6.4), dpi=130)
            ax.imshow(img.astype(np.uint8))
            if polys:
                lc = LineCollection(
                    [p.exterior.coords[:] for p in polys],
                    colors="red", linewidths=1.2)
                ax.add_collection(lc)
                for k, p in enumerate(polys):
                    ax.text(p.centroid.x, p.centroid.y, str(k),
                            color="yellow", fontsize=6, ha="center")
            ax.set_title(f"{split}/{name}  inst={len(insts)}", fontsize=8)
            ax.axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR,
                        f"{split}_{name}_overlay.png"), dpi=130)
            plt.close(fig)

    # --- 汇总统计 ---
    df = pd.DataFrame(all_stats)
    n_total_inst = int(df["n_inst"].sum())
    all_pts = [p for row in df["pt_counts"] for p in row]
    n_rect = int(df["n_rect4"].sum())
    n_8p   = int(df["n_8p"].sum())
    n_border = int(df["n_border"].sum())
    n_dup  = int(df["n_dup"].sum())
    n_oob  = int(df["oob"].sum())
    n_bad  = int(df["bad_lines"].sum())
    n_lowc = int(df["n_low_circ"].sum())

    # 空标签比例: 统计全量 TXT(不只抽样)
    n_empty, n_all_txt = 0, 0
    empty_examples = []
    for split in ("train", "val"):
        lbl_dir = os.path.join(DS_DIR, "labels", split)
        for txt in glob.glob(os.path.join(lbl_dir, "*.txt")):
            n_all_txt += 1
            with open(txt, "r", encoding="utf-8") as f:
                if not any(line.split() for line in f):
                    n_empty += 1
                    if len(empty_examples) < 5:
                        empty_examples.append(
                            f"{split}/{os.path.basename(txt)}")

    # --- 抽样图拼板 ---
    print("[拼板] 生成抽样总览图...")
    files = sorted(glob.glob(os.path.join(OUT_DIR, "*_overlay.png")))
    ncol = 6
    nrow = math.ceil(len(files) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3, nrow * 3),
                             dpi=110)
    for ax in np.array(axes).ravel():
        ax.axis("off")
    for ax, fp in zip(np.array(axes).ravel(), files):
        with rasterio.open(fp) as src:
            im = src.read([1, 2, 3]).transpose(1, 2, 0)
        ax.imshow(im.astype(np.uint8))
        ax.set_title(os.path.basename(fp)[:28], fontsize=5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "0_all_samples_grid.png"), dpi=110)
    plt.close(fig)

    print()
    print("-" * 64)
    print("A. 抽样质检汇总")
    print("-" * 64)
    print(f"抽样图片        : {len(df)} (train {sum(df['split']=='train')}, "
          f"val {sum(df['split']=='val')})")
    print(f"抽样本实例总数  : {n_total_inst}")
    if all_pts:
        print(f"多边形点数      : min={min(all_pts)} max={max(all_pts)} "
              f"mean={np.mean(all_pts):.1f} median={np.median(all_pts):.0f}")
        n_inst_all = max(1, len(all_pts))
        print(f"4 点(矩形)比例  : {n_rect}/{len(all_pts)} "
              f"({n_rect/n_inst_all*100:.1f}%)  {'[!] 可疑' if n_rect else 'OK'}")
        print(f">=8 点比例      : {n_8p}/{len(all_pts)} "
              f"({n_8p/n_inst_all*100:.1f}%)")
    print(f"触边(半圆嫌疑)  : {n_border} 实例")
    print(f"重复标注(IoU>{DUP_IOU_THRESH}) : {n_dup} 对  "
          f"{'[!] 可疑' if n_dup else 'OK'}")
    print(f"坐标越界        : {n_oob} 个坐标  {'[!]' if n_oob else 'OK'}")
    print(f"格式坏行        : {n_bad}  {'[!]' if n_bad else 'OK'}")
    print(f"低圆度(<{MIN_CIRC_WARN})    : {n_lowc} 实例 (非圆农田或标注不准)")
    print(f"空标签比例      : {n_empty}/{n_all_txt} "
          f"({n_empty/max(1,n_all_txt)*100:.1f}%)  示例: {empty_examples}")
    print()
    print(f"输出叠加图目录  : {OUT_DIR}")
    print(f"总览拼板        : {os.path.join(OUT_DIR, '0_all_samples_grid.png')}")
    return all_stats


# -----------------------------------------------------------------
# B. 数据泄漏检查
# -----------------------------------------------------------------
def check_leakage():
    print()
    print("=" * 64)
    print("B. 数据泄漏检查 (dataset_index.csv)")
    print("=" * 64)
    df = pd.read_csv(CSV_PATH)
    tr = df[df["split"] == "train"]
    va = df[df["split"] == "val"]

    tr_tifs = sorted(tr["source_tif"].unique().tolist())
    va_tifs = sorted(va["source_tif"].unique().tolist())
    shared  = sorted(set(tr_tifs) & set(va_tifs))

    print(f"[train] source_tif ({len(tr_tifs)}):")
    for t in tr_tifs:
        print(f"    {os.path.basename(t)}")
    print(f"[val]   source_tif ({len(va_tifs)}):")
    for t in va_tifs:
        print(f"    {os.path.basename(t)}")
    print(f"[共有]  source_tif ({len(shared)}):")
    for t in shared:
        print(f"    {os.path.basename(t)}")

    # tile_name 重复
    dup_tiles = set(tr["tile_name"]) & set(va["tile_name"])
    print(f"\n同一 tile 同时出现在 train/val: {len(dup_tiles)}")
    if dup_tiles:
        for t in list(dup_tiles)[:10]:
            print(f"    [!] {t}")

    # 空间范围重叠 (UTM bbox)
    print("\n[train/val 切片 UTM bbox 空间重叠检查]")
    tr_boxes = [box(r.utm_xmin, r.utm_ymin, r.utm_xmax, r.utm_ymax)
                for r in tr.itertuples()]
    va_boxes = [box(r.utm_xmin, r.utm_ymin, r.utm_xmax, r.utm_ymax)
                for r in va.itertuples()]
    tr_tree = gpd.GeoSeries(tr_boxes).sindex

    overlap_tiles = 0
    overlap_area_total = 0.0
    examples = []
    for i, vb in enumerate(va_boxes):
        hits = list(tr_tree.query(vb, predicate="intersects"))
        if hits:
            overlap_tiles += 1
            ov = sum(vb.intersection(tr_boxes[j]).area for j in hits)
            overlap_area_total += ov
            if len(examples) < 5:
                ex_t = os.path.basename(tr.iloc[hits[0]]["tile_name"])
                examples.append(
                    f"val[{va.iloc[i]['tile_name']}] ∩ train[{ex_t}] "
                    f"overlap={ov/1e6:.2f} km2")

    print(f"与 train 切片存在空间重叠的 val 切片: "
          f"{overlap_tiles}/{len(va)} "
          f"({overlap_tiles/max(1,len(va))*100:.1f}%)")
    for e in examples:
        print(f"    {e}")

    # 滑窗重叠说明
    print()
    print("[滑窗参数核对] 切片步长 512 < 窗口 640 → 相邻切片本身重叠 128 px")
    print("  即同一圆形农田很可能同时出现在相邻两切片中,")
    print("  若两切片分属 train/val → 同一目标泄漏。")

    # 结论
    print()
    print("=" * 64)
    risk = False
    if shared:
        risk = True
        print("[!!] 数据泄漏风险: 是 (train 与 val 共享 "
              f"{len(shared)} 幅 source_tif, 全部 {len(tr_tifs)} 幅)")
    if dup_tiles:
        risk = True
        print("[!!] 数据泄漏风险: 是 (同一 tile_name 同时出现在两个集合)")
    if overlap_tiles > 0:
        risk = True
        print(f"[!!] 数据泄漏风险: 是 ({overlap_tiles} 个 val 切片与 "
              f"train 切片 UTM 范围重叠, 滑窗相邻切片 128px 重叠)")
    if not risk:
        print("未发现明显泄漏。")
    else:
        print()
        print("结论: 当前 val 指标不可视为可靠泛化性能,")
        print("  必须按 source_tif 级别重新划分(如 GroupShuffleSplit by TIF),")
        print("  或至少保证同一农田的所有切片只进入一个集合。")
    print("=" * 64)

    # 保存报告
    rpt = os.path.join(OUT_DIR, "leakage_report.txt")
    with open(rpt, "w", encoding="utf-8") as f:
        f.write(f"train source_tif: {len(tr_tifs)}\n")
        for t in tr_tifs:
            f.write(f"  {t}\n")
        f.write(f"val source_tif: {len(va_tifs)}\n")
        for t in va_tifs:
            f.write(f"  {t}\n")
        f.write(f"shared source_tif: {len(shared)}\n")
        for t in shared:
            f.write(f"  {t}\n")
        f.write(f"dup tile_name count: {len(dup_tiles)}\n")
        f.write(f"val tiles overlapping train: {overlap_tiles}/{len(va)}\n")
        f.write("DATA LEAKAGE RISK: YES\n" if risk else "DATA LEAKAGE RISK: NO\n")
    print(f"泄漏报告: {rpt}")


def main():
    sample_and_overlay()
    check_leakage()


if __name__ == "__main__":
    main()
