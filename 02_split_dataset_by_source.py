# -*- coding: utf-8 -*-
"""
02_split_dataset_by_source.py — 按数据源(空间)重新划分数据集, 消除数据泄漏
================================================================
背景:
  旧划分(yolo_dataset)按切片随机 8:2, 导致 train/val 共享全部 6 幅
  source_tif, 且滑窗切片相互重叠 → 同一农田同时进入训练与验证, val
  指标不可靠。

本脚本策略:
  1. 读取 dataset_index.csv, 按 source_tif 统计切片/正负样本/UTM 范围。
  2. 用 rasterio 计算每幅 TIF 的真实 UTM bounds, 检查两两空间重叠,
     并做切片级跨 TIF 重叠检查(判定泄漏的决定性证据)。
  3. 若 TIF 之间无切片级重叠 → 按整幅 TIF 划分 4/1/1 (seed=42),
     同一 TIF 的全部切片进同一集合, 优先保证 val/test 有正样本。
  4. 若存在重叠(本项目实际情形) → 报警, 改用【空间区域划分】:
     - 以切片 UTM 中心 X 排序, 选两个切点将研究区切成 3 个东西向条带
       (train / val / test);
     - 跨越切点的切片(缓冲带)不进入任何集合(原始文件不删除, 只是不复制);
     - 切点搜索目标: val/test 有正样本 + 切片数接近 60/20/20 + 少丢弃;
     - 该方法保证跨集合切片 bbox 零重叠, 从根本上消除泄漏。
     注: 由于 6 幅 TIF 空间连通重叠, 此分支下"整幅 TIF 不拆分"在数学上
     不可能同时满足, 以空间零重叠优先(用户规则 5 优先于规则 3)。

输出:
  E:\\260827YOLORUN\\yolo_dataset_v2\\images\\{train,val,test}
  E:\\260827YOLORUN\\yolo_dataset_v2\\labels\\{train,val,test}
  E:\\260827YOLORUN\\yolo_dataset_v2\\dataset_index.csv
  E:\\260827YOLORUN\\yolo_dataset_v2\\split_report.txt

不删除 / 不修改 旧 yolo_dataset 的任何文件; 不开始训练。
================================================================
"""

import os
import glob
import random
import shutil

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.warp import transform_bounds
from shapely.geometry import box

# -----------------------------------------------------------------
# 配置
# -----------------------------------------------------------------
SRC_DS_DIR = r"E:\260827YOLORUN\yolo_dataset"
DST_DS_DIR = r"E:\260827YOLORUN\yolo_dataset_v2"
CSV_PATH   = os.path.join(SRC_DS_DIR, "dataset_index.csv")
TIF_DIR    = r"E:\260827YOLORUN\GEE_Export_2025"
REPORT_TXT = os.path.join(DST_DS_DIR, "split_report.txt")

COPY_MODE  = True        # True=复制(保留原文件); False=移动
SEED       = 42
TARGET_RATIO = {"train": 0.60, "val": 0.20, "test": 0.20}

SETS = ["train", "val", "test"]


# -----------------------------------------------------------------
# 工具
# -----------------------------------------------------------------
def load_csv():
    df = pd.read_csv(CSV_PATH)
    need = {"tile_name", "source_tif", "split",
            "utm_xmin", "utm_ymin", "utm_xmax", "utm_ymax",
            "instance_count", "is_negative"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"CSV 缺少列: {missing}")
    return df


def tif_utm_bounds(tif_path):
    with rasterio.open(tif_path) as src:
        return transform_bounds(src.crs, "EPSG:32648",
                                *src.bounds, densify_pts=21)


def connected_components(names, box_map):
    """按 bbox 面重叠对 TIF 做并查集连通分量"""
    parent = {n: n for n in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if box(*box_map[a]).intersection(box(*box_map[b])).area > 1.0:
                union(a, b)
    comps = {}
    for n in names:
        comps.setdefault(find(n), []).append(n)
    return list(comps.values())


# -----------------------------------------------------------------
# 第 1 部分: 统计 + 重叠检查
# -----------------------------------------------------------------
def analyze(df):
    print("=" * 70)
    print("[1] source_tif 统计")
    print("=" * 70)
    tif_names = sorted(df["source_tif"].unique())
    stats = {}
    for t in tif_names:
        sub = df[df["source_tif"] == t]
        name = os.path.basename(t)
        full = os.path.join(TIF_DIR, name)
        stats[t] = {
            "name": name,
            "n_tiles": len(sub),
            "n_pos": int((sub["is_negative"] == 0).sum()),
            "n_neg": int((sub["is_negative"] == 1).sum()),
        }
        # CSV 切片范围
        stats[t]["csv_bounds"] = (
            sub["utm_xmin"].min(), sub["utm_ymin"].min(),
            sub["utm_xmax"].max(), sub["utm_ymax"].max())
        # 真实 TIF UTM bounds
        if os.path.isfile(full):
            stats[t]["tif_bounds"] = tif_utm_bounds(full)
        else:
            print(f"  [警告] 找不到 TIF 文件: {full}, 使用 CSV 切片范围代替")
            stats[t]["tif_bounds"] = stats[t]["csv_bounds"]

        b = stats[t]["tif_bounds"]
        print(f"\n  {name}")
        print(f"    切片数 {stats[t]['n_tiles']:>5} | "
              f"正样本 {stats[t]['n_pos']:>4} | 负样本 {stats[t]['n_neg']:>4}")
        print(f"    UTM bounds: X {b[0]:.0f}~{b[2]:.0f}  "
              f"Y {b[1]:.0f}~{b[3]:.0f}")

    # --- TIF 两两重叠 ---
    print("\n" + "=" * 70)
    print("[2] TIF 间 UTM 空间重叠检查")
    print("=" * 70)
    any_overlap = False
    names = [stats[t]["name"] for t in tif_names]
    bmap = {stats[t]["name"]: stats[t]["tif_bounds"] for t in tif_names}
    for i, a in enumerate(names):
        for b_ in names[i + 1:]:
            inter = box(*bmap[a]).intersection(box(*bmap[b_]))
            if inter.area > 1.0:
                any_overlap = True
                print(f"  [OVERLAP] {a} <-> {b_}: {inter.area/1e6:.1f} km2")
    comps = connected_components(names, bmap)
    print(f"\n  空间连通分量数: {len(comps)}")
    for ci, comp in enumerate(comps):
        print(f"    分量{ci}: {comp}")

    # --- 切片级跨 TIF 重叠(泄漏决定性证据) ---
    print("\n  [切片级跨 TIF 重叠检查]")
    gdf = gpd.GeoDataFrame(
        df[["tile_name", "source_tif"]].copy(),
        geometry=[box(r.utm_xmin, r.utm_ymin, r.utm_xmax, r.utm_ymax)
                  for r in df.itertuples()],
        crs="EPSG:32648")
    sidx = gdf.sindex
    cross_pairs = []
    for i, r in enumerate(gdf.itertuples()):
        cand = list(sidx.query(r.geometry, predicate="intersects"))
        for j in cand:
            if j <= i:
                continue
            r2 = gdf.iloc[j]
            if r2["source_tif"] != r.source_tif:
                inter = r.geometry.intersection(r2.geometry).area
                if inter > 1.0:
                    cross_pairs.append((r.tile_name, r2["tile_name"], inter))
    print(f"  跨 TIF 重叠切片对数量: {len(cross_pairs)}")
    for a, b_, ar in cross_pairs[:5]:
        print(f"    例: {a} <-> {b_} overlap={ar/1e6:.2f} km2")

    if any_overlap or cross_pairs:
        print("\n  [!!] 报警: TIF 之间存在真实空间重叠(连通分量 = "
              f"{len(comps)}), 按整幅 TIF 划分无法消除泄漏。")
        print("       → 按用户规则 5, 改用【切片级空间区域划分】。")
        return stats, "SPATIAL", cross_pairs
    else:
        print("\n  TIF 之间无重叠 → 按整幅 TIF 划分 4/1/1。")
        return stats, "BY_TIF", cross_pairs


# -----------------------------------------------------------------
# 第 2a 部分: 按整幅 TIF 4/1/1 划分(无重叠时)
# -----------------------------------------------------------------
def split_by_tif(df, stats):
    print("\n" + "=" * 70)
    print("[3] 按整幅 TIF 划分 (4 train / 1 val / 1 test, seed=42)")
    print("=" * 70)
    tifs = list(stats.keys())
    rng = random.Random(SEED)
    rng.shuffle(tifs)

    # 保证 val/test 有正样本: 优先把有正样本的 TIF 放入 val/test
    pos_tifs = [t for t in tifs if stats[t]["n_pos"] > 0]
    if len(pos_tifs) < 2:
        print("  [警告] 有正样本的 TIF 少于 2 幅, val/test 可能无正样本!")
    val_tif, test_tif = pos_tifs[0], pos_tifs[1]
    train_tifs = [t for t in tifs if t not in (val_tif, test_tif)]

    assign = {}
    for t in train_tifs:
        assign[t] = "train"
    assign[val_tif] = "val"
    assign[test_tif] = "test"

    print(f"  val  <- {os.path.basename(val_tif)} "
          f"(正样本 {stats[val_tif]['n_pos']})")
    print(f"  test <- {os.path.basename(test_tif)} "
          f"(正样本 {stats[test_tif]['n_pos']})")
    for t in train_tifs:
        print(f"  train<- {os.path.basename(t)} "
              f"(正样本 {stats[t]['n_pos']})")
    return {r.tile_name: assign[r.source_tif] for r in df.itertuples()}


# -----------------------------------------------------------------
# 第 2b 部分: 空间区域划分(存在重叠时)
# -----------------------------------------------------------------
def split_spatial(df):
    print("\n" + "=" * 70)
    print("[3] 空间区域划分: X 向三条带 + 缓冲带(跨切点切片不进入任何集合)")
    print("=" * 70)

    xmin = ((df["utm_xmin"] + df["utm_xmax"]) / 2.0).values   # 中心 X
    hw = ((df["utm_xmax"] - df["utm_xmin"]) / 2.0).values     # 半宽 ~3200m
    pos = (df["is_negative"] == 0).values

    centers = np.sort(np.unique(xmin))
    n_c = len(centers)
    print(f"  切片数 {len(df)}, 唯一中心 X 值 {n_c} 个")

    def band_stats(k1, k2):
        """c1 = centers[k1] 切点, c2 = centers[k2]; 返回分配与统计"""
        c1, c2 = centers[k1], centers[k2]
        band = np.where(xmin < c1, 0, np.where(xmin < c2, 1, 2))
        # 跨切点(缓冲带)切片: bbox 跨越 c1 或 c2 → 丢弃
        cross1 = (xmin - hw < c1) & (xmin + hw > c1)
        cross2 = (xmin - hw < c2) & (xmin + hw > c2)
        dropped = cross1 | cross2
        keep = ~dropped
        st = {}
        for bi, s in enumerate(SETS):
            m = keep & (band == bi)
            st[s] = {"n": int(m.sum()), "pos": int(pos[m].sum())}
        st["dropped"] = int(dropped.sum())
        return band, keep, st

    best = None
    rng = random.Random(SEED)
    for k1 in range(1, n_c - 3):
        for k2 in range(k1 + 2, n_c - 1):
            _, _, st = band_stats(k1, k2)
            # 约束: val/test 有正样本, 三集合非空
            if st["val"]["pos"] <= 0 or st["test"]["pos"] <= 0:
                continue
            if min(st[s]["n"] for s in SETS) <= 0:
                continue
            total_keep = sum(st[s]["n"] for s in SETS)
            # 目标: 60/20/20
            imb = sum(abs(st[s]["n"] / total_keep - TARGET_RATIO[s])
                      for s in SETS)
            score = imb + 0.002 * st["dropped"] / len(df)
            if best is None or score < best[0] - 1e-12:
                best = (score, k1, k2, st)

    if best is None:
        raise RuntimeError("未找到满足约束的切点组合(val/test 必须有正样本)!")

    _, k1, k2, st = best
    band, keep, st = band_stats(k1, k2)
    print(f"  切点: c1={centers[k1]:.0f}, c2={centers[k2]:.0f}")
    print(f"  train: {st['train']['n']} 切片 (正样本 {st['train']['pos']})")
    print(f"  val  : {st['val']['n']} 切片 (正样本 {st['val']['pos']})")
    print(f"  test : {st['test']['n']} 切片 (正样本 {st['test']['pos']})")
    print(f"  缓冲带丢弃: {st['dropped']} 切片 (原始文件保留, 仅不进入 v2)")

    assign = {}
    for i, r in enumerate(df.itertuples()):
        if not keep[i]:
            assign[r.tile_name] = "DROPPED_BUFFER"
        else:
            assign[r.tile_name] = SETS[band[i]]
    return assign, st


# -----------------------------------------------------------------
# 第 3 部分: 复制文件
# -----------------------------------------------------------------
def copy_files(df, assign):
    print("\n" + "=" * 70)
    print("[4] 复制 PNG/TXT 成对文件 → yolo_dataset_v2 (copy 模式)")
    print("=" * 70)
    for s in SETS:
        os.makedirs(os.path.join(DST_DS_DIR, "images", s), exist_ok=True)
        os.makedirs(os.path.join(DST_DS_DIR, "labels", s), exist_ok=True)

    n_copied, n_missing = 0, []
    for r in df.itertuples():
        s_new = assign[r.tile_name]
        if s_new == "DROPPED_BUFFER":
            continue
        old_split = r.split
        png_src = os.path.join(SRC_DS_DIR, "images", old_split,
                               r.tile_name + ".png")
        txt_src = os.path.join(SRC_DS_DIR, "labels", old_split,
                               r.tile_name + ".txt")
        if not os.path.isfile(png_src):
            png_src = os.path.join(SRC_DS_DIR, "images",
                                   "val" if old_split == "train" else "train",
                                   r.tile_name + ".png")
        if not os.path.isfile(txt_src):
            txt_src = os.path.join(SRC_DS_DIR, "labels",
                                   "val" if old_split == "train" else "train",
                                   r.tile_name + ".txt")
        if not (os.path.isfile(png_src) and os.path.isfile(txt_src)):
            n_missing.append(r.tile_name)
            continue
        png_dst = os.path.join(DST_DS_DIR, "images", s_new,
                               r.tile_name + ".png")
        txt_dst = os.path.join(DST_DS_DIR, "labels", s_new,
                               r.tile_name + ".txt")
        if COPY_MODE:
            shutil.copy2(png_src, png_dst)
            shutil.copy2(txt_src, txt_dst)
        else:
            shutil.move(png_src, png_dst)
            shutil.move(txt_src, txt_dst)
        n_copied += 1

    print(f"  已复制成对文件: {n_copied}")
    if n_missing:
        print(f"  [警告] 缺失文件(未复制): {len(n_missing)}")
        for m in n_missing[:10]:
            print(f"    {m}")
    return n_copied, n_missing


# -----------------------------------------------------------------
# 第 4 部分: 新 dataset_index.csv + 验证 + 报告
# -----------------------------------------------------------------
def finalize(df, assign, stats, mode):
    # 新 CSV
    df_out = df.copy()
    df_out["split"] = df_out["tile_name"].map(assign)
    csv_out = os.path.join(DST_DS_DIR, "dataset_index.csv")
    df_out.to_csv(csv_out, index=False, encoding="utf-8-sig")
    print(f"\n[5] 新 dataset_index.csv: {csv_out}")

    # --- 验证 ---
    print("\n[6] 验证")
    rep = []
    rep.append("=" * 70)
    rep.append("yolo_dataset_v2 划分报告")
    rep.append("=" * 70)
    rep.append(f"划分模式: {mode}")
    rep.append(f"COPY 模式: {COPY_MODE}, seed={SEED}")
    rep.append("")

    # 每集合统计
    rep.append("[每集合统计]")
    for s in SETS:
        sub = df_out[df_out["split"] == s]
        pngs = glob.glob(os.path.join(DST_DS_DIR, "images", s, "*.png"))
        txts = glob.glob(os.path.join(DST_DS_DIR, "labels", s, "*.txt"))
        png_set = {os.path.splitext(os.path.basename(p))[0] for p in pngs}
        txt_set = {os.path.splitext(os.path.basename(t))[0] for t in txts}
        paired = png_set == txt_set
        rep.append(
            f"  {s:<5}: CSV 记录 {len(sub):>4} | 实际 PNG {len(pngs):>4} | "
            f"实际 TXT {len(txts):>4} | 正样本 "
            f"{int((sub['is_negative'] == 0).sum()):>4} | 负样本 "
            f"{int((sub['is_negative'] == 1).sum()):>4} | "
            f"PNG/TXT 一一对应: {'PASS' if paired else 'FAIL'}")
    dropped = df_out[df_out["split"] == "DROPPED_BUFFER"]
    rep.append(f"  缓冲带(不进入任何集合): {len(dropped)} 切片")
    rep.append("")

    # source_tif 归属
    rep.append("[source_tif 归属]")
    tif_names = sorted(df_out["source_tif"].unique())
    shared = {}
    for t in tif_names:
        sub = df_out[df_out["source_tif"] == t]
        sets_in = sorted(set(sub["split"]) - {"DROPPED_BUFFER"})
        for s in sets_in:
            shared.setdefault(s, set()).add(t)
        rep.append(f"  {os.path.basename(t)}: {sets_in} "
                   f"(切片分布: {sub['split'].value_counts().to_dict()})")
    inter_tv = shared.get("train", set()) & shared.get("val", set())
    inter_tt = shared.get("train", set()) & shared.get("test", set())
    inter_vt = shared.get("val", set()) & shared.get("test", set())
    rep.append(f"  train∩val 共享 TIF: {len(inter_tv)}")
    rep.append(f"  train∩test 共享 TIF: {len(inter_tt)}")
    rep.append(f"  val∩test 共享 TIF: {len(inter_vt)}")
    rep.append("")

    # 跨集合 UTM 重叠(决定性验证)
    rep.append("[跨集合 UTM bbox 重叠验证]")
    leak = 0
    gdf = gpd.GeoDataFrame(
        df_out[["tile_name", "split"]],
        geometry=[box(r.utm_xmin, r.utm_ymin, r.utm_xmax, r.utm_ymax)
                  for r in df_out.itertuples()],
        crs="EPSG:32648")
    sidx = gdf.sindex
    for i, r in enumerate(gdf.itertuples()):
        if r.split == "DROPPED_BUFFER":
            continue
        for j in sidx.query(r.geometry, predicate="intersects"):
            if j <= i:
                continue
            r2 = gdf.iloc[j]
            if r2["split"] in ("DROPPED_BUFFER",) or r2["split"] == r.split:
                continue
            if r.geometry.intersection(r2.geometry).area > 1.0:
                leak += 1
    rep.append(f"  train/val/test 跨集合重叠切片对: {leak}")
    rep.append(f"  结论: {'PASS — 跨集合零空间重叠, 无泄漏' if leak == 0 else 'FAIL — 仍存在泄漏!'}")
    rep.append("")

    # 旧数据保留声明
    rep.append("[旧数据]")
    rep.append(f"  旧 yolo_dataset 未删除未修改: {SRC_DS_DIR}")
    rep.append(f"  缓冲带切片仅未复制到 v2, 原始文件仍在旧目录中")

    txt = "\n".join(rep)
    print(txt)
    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(f"\n划分报告: {REPORT_TXT}")
    return leak


def main():
    df = load_csv()
    print(f"读取 {CSV_PATH}: {len(df)} 行, "
          f"split 分布 {df['split'].value_counts().to_dict()}")

    stats, mode, cross_pairs = analyze(df)

    if mode == "BY_TIF":
        assign = split_by_tif(df, stats)
        st = None
    else:
        assign, st = split_spatial(df)

    copy_files(df, assign)
    leak = finalize(df, assign, stats, mode)

    print("\n" + "=" * 70)
    if leak == 0:
        print("完成: yolo_dataset_v2 已生成, 跨集合零空间重叠。")
        print("在确认之前: 不删除旧 yolo_dataset, 不开始训练。")
    else:
        print("完成但存在泄漏, 请检查报告!")
    print("=" * 70)


if __name__ == "__main__":
    main()
