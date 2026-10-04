# -*- coding: utf-8 -*-
"""
04_train_seg.py

在 Windows + RTX 4060 环境下训练 YOLO11 实例分割模型。
使用无数据泄漏的新数据集: E:\\260827YOLORUN\\yolo_dataset_v2
(按空间区域划分, train/val/test 跨集合零 UTM 重叠)。

默认只生成 dataset.yaml 并执行数据检查, 不自动启动训练:
    & "D:\\Anacondaanzhuang\\envs\\yolotest\\python.exe" "E:\\260827YOLORUN\\04_train_seg.py"

人工确认数据检查通过后, 加 --train 启动训练:
    & "D:\\Anacondaanzhuang\\envs\\yolotest\\python.exe" "E:\\260827YOLORUN\\04_train_seg.py" --train
"""

import math
import multiprocessing
import os
import sys
from pathlib import Path

# ----------------------------------------------------------------
# 配置
# ----------------------------------------------------------------
PROJECT_ROOT = Path(r"E:\260827YOLORUN")
DATASET_ROOT = PROJECT_ROOT / "yolo_dataset_v2"     # 新: 无泄漏数据集
YAML_PATH = DATASET_ROOT / "dataset.yaml"
CSV_PATH = DATASET_ROOT / "dataset_index.csv"

TRAIN_IMAGE_DIR = DATASET_ROOT / "images" / "train"
TRAIN_LABEL_DIR = DATASET_ROOT / "labels" / "train"
VAL_IMAGE_DIR = DATASET_ROOT / "images" / "val"
VAL_LABEL_DIR = DATASET_ROOT / "labels" / "val"
TEST_IMAGE_DIR = DATASET_ROOT / "images" / "test"
TEST_LABEL_DIR = DATASET_ROOT / "labels" / "test"

MODEL_FILENAME = "yolo11n-seg.pt"                   # 必须为实例分割模型
LOCAL_MODEL_PATH = PROJECT_ROOT / MODEL_FILENAME

RUNS_ROOT = PROJECT_ROOT / "runs" / "segment"
EXP_NAME = "ordos_farmland_v2_clean_split"          # 新实验名, 不覆盖 v1
EXP_DIR = RUNS_ROOT / EXP_NAME

EPOCHS = 100
IMGSZ = 640
BATCH = 8
DEVICE = 0
WORKERS = 4
PATIENCE = 30
SEED = 42

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
SETS = ("train", "val", "test")

# 默认 False: 只生成 yaml + 数据检查, 等人工确认;
# 命令行加 --train 或将此值改为 True 才会真正开始训练。
AUTO_TRAIN = False


def path_for_yaml(path):
    """转换为 YAML 和 Ultralytics 兼容的正斜杠路径。"""
    return path.resolve().as_posix()


def build_yaml():
    """生成 Ultralytics 数据集配置文件。"""
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)

    lines = [
        f"path: {path_for_yaml(DATASET_ROOT)}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "nc: 1",
        "names:",
        "  0: farmland",
        "",
    ]

    YAML_PATH.write_text("\n".join(lines), encoding="utf-8")

    print("[1] dataset.yaml 已生成")
    print(f"    路径: {YAML_PATH}")
    print(YAML_PATH.read_text(encoding="utf-8"))


def collect_files(directory, suffixes):
    """按文件名收集指定类型文件。"""
    if not directory.is_dir():
        raise FileNotFoundError(f"目录不存在: {directory}")

    return {
        path.stem: path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in suffixes
    }


def validate_label_file(label_path):
    """验证单个 TXT 是否为 YOLO 实例分割多边形格式。"""
    text = label_path.read_text(encoding="utf-8").strip()

    # 空标签是合法负样本。
    if not text:
        return 0

    instance_count = 0

    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue

        parts = stripped.split()

        try:
            class_id = int(parts[0])
        except (ValueError, IndexError) as exc:
            raise ValueError(
                f"{label_path} 第 {line_number} 行类别无效: {line}"
            ) from exc

        if class_id != 0:
            raise ValueError(
                f"{label_path} 第 {line_number} 行类别必须为 0: {line}"
            )

        coordinate_strings = parts[1:]

        # 分割多边形至少 3 个点, 即至少 6 个坐标。
        if len(coordinate_strings) < 6:
            raise ValueError(
                f"{label_path} 第 {line_number} 行不是有效分割多边形, "
                f"坐标数量为 {len(coordinate_strings)}: {line}"
            )

        if len(coordinate_strings) % 2 != 0:
            raise ValueError(
                f"{label_path} 第 {line_number} 行坐标数量不是偶数: {line}"
            )

        try:
            coordinates = [float(value) for value in coordinate_strings]
        except ValueError as exc:
            raise ValueError(
                f"{label_path} 第 {line_number} 行含非数字坐标: {line}"
            ) from exc

        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError(
                f"{label_path} 第 {line_number} 行包含 NaN 或无穷值: {line}"
            )

        if not all(0.0 <= value <= 1.0 for value in coordinates):
            raise ValueError(
                f"{label_path} 第 {line_number} 行坐标超出 0 到 1: {line}"
            )

        points = list(zip(coordinates[0::2], coordinates[1::2]))
        unique_points = set(points)

        if len(unique_points) < 3:
            raise ValueError(
                f"{label_path} 第 {line_number} 行不足 3 个不同顶点: {line}"
            )

        instance_count += 1

    return instance_count


def validate_split(split_name, image_dir, label_dir):
    """检查单个数据集划分中的图片、标签和分割格式。"""
    images = collect_files(image_dir, IMAGE_SUFFIXES)
    labels = collect_files(label_dir, {".txt"})

    missing_labels = sorted(set(images) - set(labels))
    missing_images = sorted(set(labels) - set(images))

    if missing_labels:
        preview = "\n".join(f"      {name}" for name in missing_labels[:20])
        raise ValueError(
            f"{split_name} 中有 {len(missing_labels)} 张图片缺少 TXT:\n"
            f"{preview}"
        )

    if missing_images:
        preview = "\n".join(f"      {name}" for name in missing_images[:20])
        raise ValueError(
            f"{split_name} 中有 {len(missing_images)} 个 TXT 缺少图片:\n"
            f"{preview}"
        )

    if not images:
        raise ValueError(f"{split_name} 中没有图片: {image_dir}")

    empty_labels = 0
    nonempty_labels = 0
    total_instances = 0

    for label_path in labels.values():
        instance_count = validate_label_file(label_path)
        total_instances += instance_count

        if instance_count == 0:
            empty_labels += 1
        else:
            nonempty_labels += 1

    print(f"[数据检查] {split_name}")
    print(f"    图片数量      : {len(images)}")
    print(f"    TXT 数量      : {len(labels)}")
    print(f"    正样本标签    : {nonempty_labels}")
    print(f"    空标签负样本  : {empty_labels}")
    print(f"    实例总数      : {total_instances}")

    if nonempty_labels == 0:
        raise ValueError(
            f"{split_name} 中没有包含实例的非空标签, 不能开始训练"
        )

    return {
        "images": len(images),
        "labels": len(labels),
        "positive_labels": nonempty_labels,
        "empty_labels": empty_labels,
        "instances": total_instances,
    }


def check_split_overlap_and_source():
    """检查 train/val/test 是否共享 source_tif、是否存在 UTM 空间重叠。"""
    print("[数据检查] source_tif 与 UTM 空间重叠")
    if not CSV_PATH.is_file():
        raise FileNotFoundError(f"找不到数据集索引: {CSV_PATH}")

    import pandas as pd

    df = pd.read_csv(CSV_PATH)
    need = {"tile_name", "source_tif", "split",
            "utm_xmin", "utm_ymin", "utm_xmax", "utm_ymax"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"dataset_index.csv 缺少列: {missing}")

    df = df[df["split"].isin(SETS)]
    tif_by_set = {}
    for s in SETS:
        sub = df[df["split"] == s]
        tif_by_set[s] = set(sub["source_tif"])
        print(f"    {s:<5}: {len(sub)} 条记录, "
              f"source_tif {len(tif_by_set[s])} 幅")

    inter_tv = tif_by_set["train"] & tif_by_set["val"]
    inter_tt = tif_by_set["train"] & tif_by_set["test"]
    inter_vt = tif_by_set["val"] & tif_by_set["test"]
    print(f"    train∩val 共享 source_tif : {len(inter_tv)}")
    print(f"    train∩test 共享 source_tif: {len(inter_tt)}")
    print(f"    val∩test 共享 source_tif  : {len(inter_vt)}")

    # 空间重叠是判定泄漏的决定性标准(共享 TIF 本身允许,
    # 因为 TIF 之间物理重叠, 空间区域划分必然跨越 TIF 边界)
    from shapely.geometry import box
    import geopandas as gpd

    gdf = gpd.GeoDataFrame(
        df[["tile_name", "split"]].copy(),
        geometry=[box(r.utm_xmin, r.utm_ymin, r.utm_xmax, r.utm_ymax)
                  for r in df.itertuples()],
        crs="EPSG:32648",
    )
    sidx = gdf.sindex
    leak_pairs = 0
    for i, r in enumerate(gdf.itertuples()):
        for j in sidx.query(r.geometry, predicate="intersects"):
            if j <= i:
                continue
            other = gdf.iloc[j]
            if other["split"] == r.split:
                continue
            if r.geometry.intersection(other.geometry).area > 1.0:
                leak_pairs += 1

    print(f"    跨集合 UTM bbox 重叠切片对: {leak_pairs}")
    if leak_pairs > 0:
        raise ValueError(
            "检测到跨集合空间重叠, 存在数据泄漏, 禁止训练! "
            "请重新运行 02_split_dataset_by_source.py"
        )
    print("    结论: PASS — 跨集合零空间重叠, 无数据泄漏")


def check_dataset():
    """训练开始前检查 train/val/test 数据。"""
    print("[2] 检查数据集 (yolo_dataset_v2)")
    stats = {}
    for s in SETS:
        stats[s] = validate_split(
            s,
            DATASET_ROOT / "images" / s,
            DATASET_ROOT / "labels" / s,
        )

    check_split_overlap_and_source()

    print()
    print(
        "    train/val/test 图片比例: "
        f"{stats['train']['images']} / {stats['val']['images']} / "
        f"{stats['test']['images']}"
    )
    print(
        "    train/val/test 实例数  : "
        f"{stats['train']['instances']} / {stats['val']['instances']} / "
        f"{stats['test']['instances']}"
    )
    print()


def check_runtime(torch, ultralytics):
    """检查解释器、依赖和 CUDA 环境。"""
    print("[3] 运行环境")
    print(f"    Python 路径       : {sys.executable}")
    print(f"    Python 版本       : {sys.version.split()[0]}")
    print(f"    Ultralytics 版本  : {ultralytics.__version__}")
    print(f"    PyTorch 版本      : {torch.__version__}")
    print(f"    CUDA 可用         : {torch.cuda.is_available()}")
    print(f"    PyTorch CUDA 版本 : {torch.version.cuda}")
    print(f"    GPU 数量          : {torch.cuda.device_count()}")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA 不可用。请确认使用 yolotest 环境并检查 NVIDIA 驱动。"
        )

    if torch.cuda.device_count() <= DEVICE:
        raise RuntimeError(
            f"device={DEVICE} 不存在, GPU 数量为 "
            f"{torch.cuda.device_count()}"
        )

    gpu_name = torch.cuda.get_device_name(DEVICE)
    print(f"    GPU {DEVICE} 名称       : {gpu_name}")
    print()


def resolve_model_weights():
    """优先使用项目目录中的本地权重。"""
    if LOCAL_MODEL_PATH.is_file():
        print(f"[4] 使用本地预训练权重: {LOCAL_MODEL_PATH}")
        return str(LOCAL_MODEL_PATH)

    print(f"[4] 本地权重不存在, 将由 Ultralytics 下载: {MODEL_FILENAME}")
    return MODEL_FILENAME


def check_experiment_directory():
    """避免意外覆盖旧实验 (尤其 ordos_farmland_v1)。"""
    if EXP_DIR.exists():
        raise FileExistsError(
            f"实验目录已存在: {EXP_DIR}\n"
            "请修改 EXP_NAME, 或确认不再需要旧实验后手动处理该目录。"
        )
    old_exp = RUNS_ROOT / "ordos_farmland_v1"
    print(f"[5] 新实验目录: {EXP_DIR} (不存在, 可创建)")
    print(f"    旧实验 {old_exp} 保留不动: {old_exp.exists()}")


def main():
    build_yaml()
    check_dataset()
    check_experiment_directory()

    if not (AUTO_TRAIN or "--train" in sys.argv):
        print("=" * 60)
        print("数据检查全部完成, 未启动训练 (AUTO_TRAIN=False)。")
        print("人工确认无误后, 使用以下命令开始训练:")
        print('  & "D:\\Anacondaanzhuang\\envs\\yolotest\\python.exe" '
              '"E:\\260827YOLORUN\\04_train_seg.py" --train')
        print("=" * 60)
        return

    try:
        import torch
        import ultralytics
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "缺少 torch 或 ultralytics。请使用 yolotest 环境运行。"
        ) from exc

    check_runtime(torch, ultralytics)
    model_weights = resolve_model_weights()

    print("[6] 初始化 YOLO11 实例分割模型")
    model = YOLO(model_weights)
    if model.task != "segment":
        raise RuntimeError(
            f"模型任务类型为 {model.task}, 不是 segment, 禁止训练!"
        )

    print("[7] 开始训练")
    print(f"    dataset.yaml : {YAML_PATH}")
    print(f"    epochs       : {EPOCHS}")
    print(f"    imgsz        : {IMGSZ}")
    print(f"    batch        : {BATCH}")
    print(f"    device       : {DEVICE}")
    print(f"    workers      : {WORKERS}")
    print(f"    patience     : {PATIENCE}")
    print(f"    seed         : {SEED}")
    print(f"    输出目录     : {EXP_DIR}")
    print()

    model.train(
        data=str(YAML_PATH),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        project=str(RUNS_ROOT),
        name=EXP_NAME,
        workers=WORKERS,
        patience=PATIENCE,
        seed=SEED,
        cache=False,
        amp=True,
        plots=True,
        save=True,
        exist_ok=False,
        verbose=True,
    )

    best_path = EXP_DIR / "weights" / "best.pt"
    last_path = EXP_DIR / "weights" / "last.pt"

    print()
    print("=" * 60)
    print("训练完成")
    print(f"结果目录 : {EXP_DIR}")
    print(f"best.pt  : {best_path}, 存在={best_path.is_file()}")
    print(f"last.pt  : {last_path}, 存在={last_path.is_file()}")
    print("=" * 60)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
