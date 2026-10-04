# -*- coding: utf-8 -*-
"""
02_split_dataset.py
================================================================
数据集随机划分脚本:把 yolo_dataset/images/train 与 labels/train 中
20% 的配对切片随机抽出来,移动到 images/val 与 labels/val,
并同步更新 dataset_index.csv 的 split 字段。

操作方式:文件剪切(move),不是复制。重复运行会把越来越多文件
挪到 val,因此本脚本应只在数据集生成完成后运行一次。
若需重新划分,请先重新运行 data_prepare.py 重建 train。
================================================================
"""

import os
import csv
import random
import shutil
import sys


# ----------------------------------------------------------------
# 配置区
# ----------------------------------------------------------------
DATASET_ROOT = r"E:\260827YOLORUN\yolo_dataset"

IMG_TRAIN_DIR = os.path.join(DATASET_ROOT, "images", "train")
IMG_VAL_DIR   = os.path.join(DATASET_ROOT, "images", "val")
LBL_TRAIN_DIR = os.path.join(DATASET_ROOT, "labels", "train")
LBL_VAL_DIR   = os.path.join(DATASET_ROOT, "labels", "val")
INDEX_CSV     = os.path.join(DATASET_ROOT, "dataset_index.csv")

VAL_RATIO = 0.2          # 验证集占比 20%
RANDOM_SEED = 42         # 固定随机种子,保证可复现


# ----------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------
def main():
    print("=" * 60)
    print("数据集随机划分:train -> val (20%)")
    print("=" * 60)
    print(f"数据集根目录 : {DATASET_ROOT}")
    print(f"验证集比例   : {VAL_RATIO}")
    print(f"随机种子     : {RANDOM_SEED}")
    print()

    # ---------- 1. 创建 val 目录 ----------
    os.makedirs(IMG_VAL_DIR, exist_ok=True)
    os.makedirs(LBL_VAL_DIR, exist_ok=True)
    print(f"[1] 已确保 val 目录存在:")
    print(f"    {IMG_VAL_DIR}")
    print(f"    {LBL_VAL_DIR}")
    print()

    # ---------- 2. 扫描 train 下所有 PNG,并校验同名 TXT ----------
    if not os.path.isdir(IMG_TRAIN_DIR):
        print(f"[错误] images/train 目录不存在: {IMG_TRAIN_DIR}")
        sys.exit(1)

    png_files = [f for f in os.listdir(IMG_TRAIN_DIR)
                 if f.lower().endswith(".png")]
    if not png_files:
        print(f"[错误] images/train 中没有 PNG 文件: {IMG_TRAIN_DIR}")
        sys.exit(1)

    paired = []        # 配对成功的 (png_basename, png_path, txt_path)
    missing_txt = []   # 缺失 TXT 的 PNG
    for png in png_files:
        base = os.path.splitext(png)[0]
        png_path = os.path.join(IMG_TRAIN_DIR, png)
        txt_path = os.path.join(LBL_TRAIN_DIR, base + ".txt")
        if os.path.isfile(txt_path):
            paired.append((base, png_path, txt_path))
        else:
            missing_txt.append(png)

    print(f"[2] 扫描完成:")
    print(f"    images/train 下 PNG 总数      : {len(png_files)}")
    print(f"    配对成功(PNG + TXT 同名)       : {len(paired)}")
    print(f"    缺失 TXT 的 PNG(将跳过)      : {len(missing_txt)}")
    if missing_txt:
        print("    [警告] 缺失 TXT 的文件示例(前 5 个):")
        for p in missing_txt[:5]:
            print(f"      - {p}")
    if not paired:
        print("[错误] 没有任何配对文件,终止划分。")
        sys.exit(1)
    print()

    # ---------- 3. 随机抽取 20% 作为 val ----------
    random.seed(RANDOM_SEED)
    n_val = int(round(len(paired) * VAL_RATIO))
    # 取下整避免 val 超过总数
    if n_val < 1:
        n_val = 1
    val_set = random.sample(paired, n_val)
    val_bases = {item[0] for item in val_set}  # 用于 CSV 快速查找
    print(f"[3] 随机抽取完成:")
    print(f"    配对总数          : {len(paired)}")
    print(f"    抽取 val 数量     : {n_val}")
    print(f"    剩余 train 数量   : {len(paired) - n_val}")
    print()

    # ---------- 4. 移动文件 ----------
    moved_png = 0
    moved_txt = 0
    move_fail = []
    for base, png_path, txt_path in val_set:
        # PNG
        dst_png = os.path.join(IMG_VAL_DIR, os.path.basename(png_path))
        # TXT
        dst_txt = os.path.join(LBL_VAL_DIR, os.path.basename(txt_path))
        try:
            if os.path.exists(dst_png):
                # 已存在则覆盖(避免残留)
                os.remove(dst_png)
            shutil.move(png_path, dst_png)
            moved_png += 1
        except Exception as e:
            move_fail.append((png_path, str(e)))
        try:
            if os.path.exists(dst_txt):
                os.remove(dst_txt)
            shutil.move(txt_path, dst_txt)
            moved_txt += 1
        except Exception as e:
            move_fail.append((txt_path, str(e)))

    print(f"[4] 文件移动完成:")
    print(f"    成功移动 PNG 数 : {moved_png}")
    print(f"    成功移动 TXT 数 : {moved_txt}")
    if move_fail:
        print(f"    [警告] 移动失败 {len(move_fail)} 个,示例:")
        for p, e in move_fail[:5]:
            print(f"      - {p}: {e}")
    print()

    # ---------- 5. 同步更新 dataset_index.csv ----------
    csv_updated = 0
    if os.path.isfile(INDEX_CSV):
        # 先读取全部
        with open(INDEX_CSV, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            rows = list(reader)

        if "tile_name" not in fieldnames or "split" not in fieldnames:
            print(f"[错误] CSV 缺少 'tile_name' 或 'split' 字段,无法同步更新。")
            print(f"       字段列表: {fieldnames}")
        else:
            for row in rows:
                tn = row.get("tile_name", "")
                # tile_name 通常含 .png 后缀,去掉后缀与文件名比较
                tn_base = os.path.splitext(tn)[0]
                if tn_base in val_bases:
                    row["split"] = "val"
                    csv_updated += 1
            # 覆写原 CSV
            with open(INDEX_CSV, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            print(f"[5] dataset_index.csv 同步更新完成:")
            print(f"    标记为 val 的行数 : {csv_updated}")
            print(f"    CSV 总行数(不含表头): {len(rows)}")
    else:
        print(f"[5] [警告] 未找到 dataset_index.csv: {INDEX_CSV}")
        print("    跳过 CSV 更新步骤。")
    print()

    # ---------- 6. 最终统计 ----------
    final_train_png = len([f for f in os.listdir(IMG_TRAIN_DIR)
                          if f.lower().endswith(".png")])
    final_val_png   = len([f for f in os.listdir(IMG_VAL_DIR)
                          if f.lower().endswith(".png")])
    final_train_txt = len([f for f in os.listdir(LBL_TRAIN_DIR)
                           if f.lower().endswith(".txt")])
    final_val_txt   = len([f for f in os.listdir(LBL_VAL_DIR)
                           if f.lower().endswith(".txt")])

    print("=" * 60)
    print("划分完成统计:")
    print("=" * 60)
    print(f"  train 集 PNG 数 : {final_train_png}")
    print(f"  val   集 PNG 数 : {final_val_png}")
    print(f"  train 集 TXT 数 : {final_train_txt}")
    print(f"  val   集 TXT 数 : {final_val_txt}")
    print()
    # 一致性校验
    ok_png = (final_train_png + final_val_png) == len(png_files)
    ok_txt = (final_train_txt + final_val_txt) == len(paired)
    ok_pair_train = final_train_png == final_train_txt
    ok_pair_val   = final_val_png == final_val_txt
    print(f"  PNG 总数守恒       : {'PASS' if ok_png else 'FAIL'} "
          f"(原始 {len(png_files)} = train {final_train_png} + val {final_val_png})")
    print(f"  TXT 总数守恒       : {'PASS' if ok_txt else 'FAIL'} "
          f"(配对 {len(paired)} = train {final_train_txt} + val {final_val_txt})")
    print(f"  train PNG/TXT 一一对应: {'PASS' if ok_pair_train else 'FAIL'}")
    print(f"  val   PNG/TXT 一一对应: {'PASS' if ok_pair_val else 'FAIL'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
