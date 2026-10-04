# -*- coding: utf-8 -*-
"""
data_prepare.py  (实例分割版 / UTM WarpedVRT 重构 v4)
==================================================
鄂尔多斯圆形农田 —— Ultralytics YOLO 实例分割(yolo11*-seg.pt)数据准备脚本

================================================================
本版本(v4)与 v3 的关键区别
================================================================
1. SHP 已是 EPSG:32648(UTM 48N),不再无条件 to_crs();
   读取后断言 gdf.crs == EPSG:32648,不满足直接报错,禁止 set_crs。
2. 原始 TIF 是 WGS84 经纬度,像元非正方形(纬度方向拉长),
   直接按像素切片会导致圆形农田变歪/变椭圆。
3. 使用 rasterio.vrt.WarpedVRT 把每幅 TIF 真正重投影到 EPSG:32648,
   并强制 X/Y 分辨率严格相等(TARGET_RESOLUTION_METERS)。
4. 所有后续操作(切片、bounds、window_transform、空间查询、intersection、
   像素映射)全部基于 VRT,不再使用原始 src。
5. 窗口地理边界用 rasterio.windows.bounds(window, vrt.transform),
   第二参数必须是 vrt.transform。
6. 局部像素坐标用 rasterio.windows.transform(window, vrt.transform) 的逆。
7. 生成质检叠加图(底图 PNG + 红线 YOLO 多边形)到 yolo_dataset/debug_overlay。

输出标签格式(每行一个实例,坐标归一化到 0~1):
    0 x1 y1 x2 y2 x3 y3 ... xn yn

依赖:rasterio, geopandas, shapely, numpy, matplotlib + Python 标准库
严禁:gdal, OpenCV, Pillow
运行环境:conda 环境 yolotest
"""

import os
import csv
import glob
import random
import shutil

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.windows import Window, bounds as win_bounds, transform as win_transform_fn
from rasterio.vrt import WarpedVRT
from rasterio.warp import calculate_default_transform
from rasterio.enums import Resampling
from rasterio.crs import CRS
from shapely.geometry import box, Polygon, MultiPolygon, GeometryCollection
from shapely.affinity import affine_transform

# 兼容旧版 Shapely(<2.0 无 make_valid),用 try/except 导入
try:
    from shapely.validation import make_valid   # Shapely 1.8+/2.x
except ImportError:                              # 极旧版本无此函数
    make_valid = None

# matplotlib 仅用于生成质检叠加图(不参与训练数据生成)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =====================================================================
# 配置区
# =====================================================================
TIF_DIR      = r"E:\260827YOLORUN\GEE_Export_2025"   # 遥感影像目录
SHP_DIR      = r"E:\260827YOLORUN\jiaozheng2025"     # 矢量标签目录
OUT_DIR      = r"E:\260827YOLORUN\yolo_dataset"      # 输出根目录

# ---- 目标坐标系(固定 UTM 48N) ----
TARGET_CRS = CRS.from_epsg(32648)

# ---- 强制统一米制分辨率(米/像元) ----
# calculate_default_transform 通常不会产生严格相等的 X/Y 米制分辨率,
# 必须显式指定,否则圆形农田会变椭圆。
TARGET_RESOLUTION_METERS = 10.0   # 10 米/像元(Sentinel-2 常见)

TILE_SIZE    = 640          # 切片像素尺寸
OVERLAP      = 0.20         # 切片重叠率(20%)
STEP         = int(TILE_SIZE * (1 - OVERLAP))   # 步长 = 512
NEG_KEEP     = 0.10         # 背景切片保留比例(10%)

# 波段顺序:探针显示 descriptions=('B2','B3','B4','B8','B11','B12')
# 真彩色 RGB = R:B4, G:B3, B:B2,即读取顺序 [3, 2, 1]
BANDS_RGB    = [3, 2, 1]

CLIP_PCT     = (2, 98)      # 全局拉伸分位数
VIS_RATIO    = 0.70         # 可见比例阈值(交集面积/原面积),<0.7 整片跳过
MIN_PIX_AREA = 4.0          # 最小像素面积过滤(像素²)
MIN_VERTICES = 3            # 多边形最少顶点数
VALID_PIX_RATIO_MIN = 0.20  # 切片有效像元比例下限
BLOCK_STAT   = 4096         # 全局拉伸统计分块尺寸
HIST_BINS    = 2048         # 直方图分箱数
SEED         = 42

# ---- 顶点数控制 ----
# None=不抽稀;数值=超过该顶点数按等间隔抽稀(保留首尾)
MAX_VERTICES = None

# ---- 运行控制 ----
CLEAR_OUTPUT = True         # True=清理旧输出目录
DEBUG_N      = 5            # 每幅 TIF 最多打印的调试切片数
# 第一次小批量验证:
MAX_TIFS          = None    # 最多处理 TIF 数(None=不限)
MAX_SAVED_TILES   = None    # 每幅 TIF 最多保存切片数(None=不限)
# 正样本优先:在配额内优先保留含农田实例的切片,避免全是负样本

IMG_DIR      = os.path.join(OUT_DIR, "images", "train")
LBL_DIR      = os.path.join(OUT_DIR, "labels", "train")
OVERLAY_DIR  = os.path.join(OUT_DIR, "debug_overlay")
INDEX_CSV    = os.path.join(OUT_DIR, "dataset_index.csv")

# CSV 固定字段顺序(按需求 25 项重新设计)
INDEX_FIELDS = [
    "tile_name", "source_tif", "source_crs", "target_crs",
    "utm_xmin", "utm_ymin", "utm_xmax", "utm_ymax",
    "pixel_size_m", "split", "instance_count", "is_negative",
]


# =====================================================================
# 文件查找(glob 兼容大小写)
# =====================================================================
def find_tifs(tif_dir):
    pats = [os.path.join(tif_dir, "*." + e) for e in ("tif", "tiff", "TIF", "TIFF")]
    found = []
    for p in pats:
        found.extend(glob.glob(p))
    return sorted(set(found))


def find_shps(shp_dir):
    pats = [os.path.join(shp_dir, "*." + e) for e in ("shp", "SHP")]
    found = []
    for p in pats:
        found.extend(glob.glob(p))
    return sorted(set(found))


# =====================================================================
# 输出目录清理
# =====================================================================
def _rmtree_onexc(func, path, exc_info):
    """Windows 下文件被占用/只读时的清理回调(Python 3.12+ onexc 签名)。"""
    try:
        os.chmod(path, 0o777)
        os.unlink(path)
    except Exception:
        pass


def _rmtree_onerror(func, path, exc_info):
    """旧版 Python onerror 签名(兼容回退)。"""
    try:
        os.chmod(path, 0o777)
        os.unlink(path)
    except Exception:
        pass


def clean_output():
    for d in (IMG_DIR, LBL_DIR, OVERLAY_DIR):
        if os.path.isdir(d):
            # 兼容新旧 Python:rmtree 优先 onexc,无则降级 onerror
            try:
                shutil.rmtree(d, onexc=_rmtree_onexc)
            except TypeError:
                shutil.rmtree(d, onerror=_rmtree_onerror)
            print(f"  已清理 {d}")
    if os.path.isfile(INDEX_CSV):
        try:
            os.remove(INDEX_CSV)
            print(f"  已清理 {INDEX_CSV}")
        except Exception as e:
            print(f"  [警告] 无法删除 {INDEX_CSV}: {e}")


# =====================================================================
# SHP 读取 + CRS 严格断言
# =====================================================================
def load_all_shps(shp_paths):
    """
    读取所有 SHP。SHP 必须自带 CRS,且必须等于 EPSG:32648。
    不满足立即报错,禁止 set_crs 冒充转换。
    不做任何 to_crs(已经是目标 UTM)。
    """
    if not shp_paths:
        raise FileNotFoundError(f"在 {SHP_DIR} 找不到 .shp 矢量")

    gdfs = []
    for p in shp_paths:
        g = gpd.read_file(p)
        # ---- 严格 CRS 检查 ----
        if g.crs is None:
            raise ValueError(
                f"[CRS 错误] SHP 文件 {p} 的 CRS 为 None。\n"
                f"  请检查同目录下是否存在 .prj 文件;禁止用 set_crs() 冒充转换。"
            )
        # 断言 CRS 必须是 EPSG:32648
        if not (g.crs == TARGET_CRS):
            raise ValueError(
                f"[CRS 不匹配] SHP {p} 的 CRS = {g.crs},\n"
                f"  本项目要求 SHP 必须是 {TARGET_CRS},\n"
                f"  请先在 GIS 中投影转换,或检查 .prj 文件。"
            )
        print(f"  读取 {os.path.basename(p)}: CRS={g.crs}, 几何数={len(g)}")
        gdfs.append(g)

    # 合并(SHP 已统一 EPSG:32648,无需 to_crs)
    import pandas as pd
    merged = pd.concat(gdfs, ignore_index=True)
    merged = gpd.GeoDataFrame(merged, crs=TARGET_CRS).reset_index(drop=True)
    print(f"  合并后总几何数 = {len(merged)}, CRS = {merged.crs}")
    print(f"  合并后矢量边界 = {merged.total_bounds}")

    # 健壮性检查:矢量是否包含有效面几何
    poly_mask = merged.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    n_poly = int(poly_mask.sum())
    print(f"  有效面几何数(Polygon/MultiPolygon) = {n_poly}")
    if n_poly == 0:
        raise ValueError("[矢量错误] SHP 不包含任何 Polygon/MultiPolygon 几何。")
    return merged


# =====================================================================
# 几何清理:make_valid / 去重 / 去共线点
# =====================================================================
def _make_geom_valid(geom):
    if geom.is_valid:
        return geom
    if make_valid is not None:
        try:
            return make_valid(geom)
        except Exception:
            pass
    return geom.buffer(0)


def _remove_collinear_points(coords, eps=1e-9):
    """去除共线点(叉积判别)与连续重复点。"""
    n = len(coords)
    if n <= 2:
        return coords
    dedup = [coords[0]]
    for i in range(1, n):
        x0, y0 = dedup[-1]
        x1, y1 = coords[i]
        if abs(x1 - x0) > eps or abs(y1 - y0) > eps:
            dedup.append(coords[i])
    out = []
    m = len(dedup)
    for i in range(m):
        x0, y0 = dedup[(i - 1) % m]
        x1, y1 = dedup[i]
        x2, y2 = dedup[(i + 1) % m]
        cross = (x1 - x0) * (y2 - y1) - (y1 - y0) * (x2 - x1)
        if abs(cross) > eps:
            out.append((x1, y1))
    return out


def _polygon_to_instance_rings(poly):
    """Polygon -> 外环坐标列表(内洞不写入)。"""
    if not isinstance(poly, Polygon) or poly.is_empty:
        return []
    ext = list(poly.exterior.coords)
    if len(ext) >= 2 and abs(ext[0][0] - ext[-1][0]) < 1e-9 \
            and abs(ext[0][1] - ext[-1][1]) < 1e-9:
        ext = ext[:-1]
    ext = _remove_collinear_points(ext)
    if len(ext) < MIN_VERTICES:
        return []
    return [ext]


def geometry_to_instances(geom):
    """任意几何 -> 多个实例外环列表。"""
    geom = _make_geom_valid(geom)
    instances = []
    if isinstance(geom, Polygon):
        instances.extend(_polygon_to_instance_rings(geom))
    elif isinstance(geom, MultiPolygon):
        for poly in geom.geoms:
            instances.extend(_polygon_to_instance_rings(poly))
    elif isinstance(geom, GeometryCollection):
        for g in geom.geoms:
            instances.extend(geometry_to_instances(g))
    return instances


# =====================================================================
# VRT 创建:把 WGS84 TIF 真正重投影到 EPSG:32648,正方形米制像元
# =====================================================================
def build_vrt(src):
    """
    用 calculate_default_transform + WarpedVRT 把 src 重投影到 TARGET_CRS,
    并强制 X/Y 分辨率严格相等(TARGET_RESOLUTION_METERS)。
    返回:(vrt, dst_transform, dst_width, dst_height, dst_crs)
    """
    if src.crs is None:
        raise ValueError(f"[CRS 错误] TIF 的 CRS 为 None,无法重投影。")

    # 1) 先用 calculate_default_transform 计算默认参数
    dst_transform, dst_width, dst_height = calculate_default_transform(
        src.crs,            # 源 CRS(WGS84)
        TARGET_CRS,         # 目标 CRS(UTM 48N)
        src.width,
        src.height,
        *src.bounds,
        resolution=TARGET_RESOLUTION_METERS,   # 显式指定米/像元
    )

    # 2) 强制 X/Y 分辨率严格相等(calculate_default_transform 可能不严格)
    res = TARGET_RESOLUTION_METERS
    # 重新构造 transform:a=res, e=-res(北朝上,e 为负)
    dst_transform = rasterio.transform.from_origin(
        dst_transform.c,    # 左上角 X
        dst_transform.f,    # 左上角 Y
        res, res            # xsize, ysize(正方形)
    )
    # 重算 width/height(基于 bounds 与分辨率)
    left, top = dst_transform.c, dst_transform.f
    xs = abs(dst_transform.a)
    # 右下角:b=d=0, 右下 X = c + a*width, 下 Y = f + e*height
    # 但 calculate_default_transform 已给出近似 width/height,微调即可
    # 这里直接采用 calculate_default_transform 的 width/height(已是最近整数)

    # 3) 硬性断言:X/Y 米制分辨率必须相等
    assert abs(abs(dst_transform.a) - abs(dst_transform.e)) < 1e-9, \
        f"[像元非正方形] X={abs(dst_transform.a)}, Y={abs(dst_transform.e)}"

    # 4) 创建 WarpedVRT
    vrt = WarpedVRT(
        src,
        crs=TARGET_CRS,
        transform=dst_transform,
        width=dst_width,
        height=dst_height,
        resampling=Resampling.bilinear,   # RGB 用双线性
        nodata=src.nodata,
    )

    # 5) 二次断言:VRT 实际参数
    assert vrt.crs == TARGET_CRS, f"[VRT CRS 错] {vrt.crs}"
    assert abs(abs(vrt.transform.a) - abs(vrt.transform.e)) < 1e-9, \
        f"[VRT 像元非正方形] X={abs(vrt.transform.a)}, Y={abs(vrt.transform.e)}"

    return vrt, vrt.transform, vrt.width, vrt.height, vrt.crs


# =====================================================================
# TIF 元信息打印
# =====================================================================
def print_tif_meta(src, path):
    """打印原始 TIF 与重投影后 VRT 的关键元信息。"""
    print(f"  TIF 路径        : {path}")
    print(f"  原始 size (WxH) : {src.width} x {src.height}")
    print(f"  原始 count      : {src.count}")
    print(f"  原始 descriptions: {src.descriptions}")
    print(f"  原始 dtypes     : {src.dtypes}")
    print(f"  原始 nodata     : {src.nodata}")
    print(f"  原始 CRS        : {src.crs}")
    print(f"  原始 transform  : {src.transform}")
    print(f"  原始 res        : {src.res}")
    print(f"  原始 bounds     : {src.bounds}")
    # 健壮性检查
    if src.count < 3:
        raise ValueError(f"[波段不足] {path} 只有 {src.count} 个波段,至少需要 3。")
    if src.crs is None:
        raise ValueError(f"[CRS 错误] TIF {path} 的 CRS 为 None。")


def print_vrt_meta(vrt, src, gdf_base):
    """打印重投影后 VRT 元信息 + 与 SHP 的相交情况。"""
    print(f"  目标 CRS        : {TARGET_CRS}")
    print(f"  VRT size (WxH)  : {vrt.width} x {vrt.height}")
    print(f"  VRT transform   : {vrt.transform}")
    print(f"  VRT res (m)     : X={abs(vrt.transform.a):.6f}, "
          f"Y={abs(vrt.transform.e):.6f}")
    print(f"  VRT bounds      : {vrt.bounds}")
    print(f"  SHP CRS         : {gdf_base.crs}")
    print(f"  SHP bounds      : {gdf_base.total_bounds}")
    # 相交检查(SHP 已是 UTM,VRT 也是 UTM,直接比较)
    shp_box = box(*gdf_base.total_bounds)
    vrt_box = box(*vrt.bounds)
    intersects = shp_box.intersects(vrt_box)
    print(f"  SHP 与 VRT 相交 : {intersects}")
    if not intersects:
        print("  [警告] SHP 与 VRT 范围完全不相交,按纯背景影像处理(仍生成 10% 负样本)。")


# =====================================================================
# 全局拉伸:基于 VRT 分块统计分位数(排除 NoData/无效)
# =====================================================================
def compute_global_stretch(vrt, bands, block=BLOCK_STAT, bins=HIST_BINS):
    """
    对 VRT 分块统计 2%/98% 分位数。
    使用三波段共同有效掩膜(掩码 + nodata + NaN)。
    返回: [(p_lo, p_hi), ...] 长度 = len(bands)。
    """
    H, W = vrt.height, vrt.width
    n = len(bands)
    range_min = [None] * n
    range_max = [None] * n

    # ---- 第一遍:确定每波段 min/max ----
    for r in range(0, H, block):
        for c in range(0, W, block):
            h = min(block, H - r)
            w = min(block, W - c)
            win = Window(c, r, w, h)
            # masked=True:rasterio 自动用 nodata 与掩码生成 masked array
            data = vrt.read(bands, window=win, masked=True)   # (n, h, w) MaskedArray
            if data.mask is np.ma.nomask:
                common_valid = np.ones((h, w), dtype=bool)
            else:
                # 三波段共同有效(任一波段 mask=True 即无效)
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

    # ---- 严格检查:任一波段无有效像元立即报错 ----
    for bi in range(n):
        if range_min[bi] is None:
            raise ValueError(
                f"[波段无有效像元] 第 {bands[bi]} 波段在全图范围内无有效像元。"
            )

    # ---- 直方图分箱 ----
    edges_list = []
    for bi in range(n):
        lo, hi = range_min[bi], range_max[bi]
        if hi - lo < 1e-6:
            hi = lo + 1.0
        edges_list.append(np.linspace(lo, hi, bins + 1))

    # ---- 第二遍:累加直方图 ----
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

    # ---- 反推分位数 ----
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
    将 (3, H, W) 多波段数据按全局拉伸转为 (H, W, 3) uint8。
    无效像元统一设为黑色 [0,0,0],避免边缘出现绿色/彩色色斑。
    """
    n, h, w = data_chw.shape
    out = np.zeros((n, h, w), dtype=np.uint8)   # 默认全黑
    for bi in range(n):
        p_lo, p_hi = stretch[bi]
        if p_hi - p_lo < 1e-6:
            continue
        arr = data_chw[bi].astype(np.float32)
        scaled = (arr - p_lo) / (p_hi - p_lo) * 255.0
        out[bi] = np.where(valid_mask,
                           np.clip(scaled, 0, 255).astype(np.uint8),
                           0)
    return out.transpose(1, 2, 0)   # HWC


def save_png(arr_hwc_uint8, out_path):
    """用 rasterio PNG 驱动保存 (H, W, 3) uint8。"""
    h, w, _ = arr_hwc_uint8.shape
    with rasterio.open(
        out_path, "w",
        driver="PNG",
        height=h, width=w, count=3, dtype="uint8",
    ) as dst:
        dst.write(arr_hwc_uint8.transpose(2, 0, 1))


def write_empty_txt(path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("")


# =====================================================================
# 质检叠加图(底图 PNG + 红线 YOLO 多边形)
# =====================================================================
def save_overlay_png(img_hwc_uint8, labels, out_path):
    """
    底图为最终 UTM PNG,把生成的 YOLO 分割多边形用红线画回同一 PNG。
    仅用于质检,不改变训练图片。
    使用 matplotlib 保存(不用 Pillow/OpenCV)。
    """
    fig, ax = plt.subplots(figsize=(6.4, 6.4), dpi=100)
    ax.imshow(img_hwc_uint8)
    for ln in labels:
        parts = ln.split()
        if len(parts) < 7:
            continue
        coords = [float(v) for v in parts[1:]]
        xs = [c * TILE_SIZE for c in coords[0::2]]
        ys = [c * TILE_SIZE for c in coords[1::2]]
        # 闭合
        xs.append(xs[0])
        ys.append(ys[0])
        ax.plot(xs, ys, color="red", linewidth=1.2)
    ax.set_xlim(0, TILE_SIZE)
    ax.set_ylim(TILE_SIZE, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


# =====================================================================
# 标签文件验证(运行后)
# =====================================================================
def validate_labels(lbl_dir):
    """
    遍历 labels/train 下所有 .txt,验证每行格式:
      - 类别为 0
      - 坐标数量为偶数
      - 坐标数量 >= 6(即至少 3 个顶点)
      - 坐标都在 0~1
      - 没有 NaN 或 Inf
    返回:(总文件数, 合规文件数, 错误列表)
    """
    txts = sorted(glob.glob(os.path.join(lbl_dir, "*.txt")))
    total = len(txts)
    ok = 0
    errors = []
    for t in txts:
        name = os.path.basename(t)
        try:
            with open(t, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            if not lines:
                ok += 1
                continue
            file_ok = True
            for li, ln in enumerate(lines, 1):
                parts = ln.split()
                if len(parts) < 7:
                    errors.append(f"{name}:L{li} 坐标数<6 ({len(parts)-1})")
                    file_ok = False
                    continue
                if parts[0] != "0":
                    errors.append(f"{name}:L{li} 类别非0 ({parts[0]})")
                    file_ok = False
                    continue
                coords = parts[1:]
                if len(coords) % 2 != 0:
                    errors.append(f"{name}:L{li} 坐标数为奇数 ({len(coords)})")
                    file_ok = False
                    continue
                try:
                    vals = [float(v) for v in coords]
                except ValueError:
                    errors.append(f"{name}:L{li} 存在非数字坐标")
                    file_ok = False
                    continue
                if any(np.isnan(v) or np.isinf(v) for v in vals):
                    errors.append(f"{name}:L{li} 含 NaN/Inf")
                    file_ok = False
                    continue
                if any(v < 0.0 or v > 1.0 for v in vals):
                    errors.append(f"{name}:L{li} 坐标越界[0,1]")
                    file_ok = False
                    continue
            if file_ok:
                ok += 1
        except Exception as e:
            errors.append(f"{name}: 读取异常 {e}")
    return total, ok, errors


# =====================================================================
# 主流程
# =====================================================================
def main():
    random.seed(SEED)

    # ---------- 0. 清理旧输出 ----------
    if CLEAR_OUTPUT:
        print("[步骤 0] 清理旧输出目录...")
        clean_output()

    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs(LBL_DIR, exist_ok=True)
    os.makedirs(OVERLAY_DIR, exist_ok=True)

    # ---------- 1. 找文件 ----------
    tifs = find_tifs(TIF_DIR)
    shps = find_shps(SHP_DIR)
    if not tifs:
        raise FileNotFoundError(f"在 {TIF_DIR} 找不到 .tif/.tiff 影像")
    if not shps:
        raise FileNotFoundError(f"在 {SHP_DIR} 找不到 .shp 矢量")
    print(f"找到 {len(tifs)} 个 TIF 影像, {len(shps)} 个 SHP 矢量")

    # ---------- 2. 读取 SHP(断言 EPSG:32648,不做 to_crs) ----------
    print("\n[步骤 1] 读取并合并所有 SHP 矢量...")
    gdf_base = load_all_shps(shps)
    print(f"  SHP CRS = {gdf_base.crs}")

    # 先过滤空/None,再修复,修复后仍无效才删除
    gdf_base = gdf_base[
        gdf_base.geometry.notna() & ~gdf_base.geometry.is_empty
    ].copy()
    n_before = len(gdf_base)
    n_invalid_before = int((~gdf_base.is_valid).sum())
    gdf_base["geometry"] = gdf_base.geometry.apply(_make_geom_valid)
    gdf_base = gdf_base[
        gdf_base.geometry.notna() &
        ~gdf_base.geometry.is_empty &
        gdf_base.is_valid
    ].copy()
    n_after_valid = len(gdf_base)
    n_before_face = len(gdf_base)
    gdf_base = gdf_base[
        gdf_base.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    ].reset_index(drop=True)
    n_after = len(gdf_base)
    print(f"  几何修复: 输入={n_before}, 修复前无效={n_invalid_before}, "
          f"修复后有效={n_after_valid}, "
          f"面类型保留={n_after}, 丢弃非面={n_before_face - n_after}")

    # 断言:gdf.crs 必须是 EPSG:32648
    assert gdf_base.crs == TARGET_CRS, \
        f"[断言失败] gdf.crs={gdf_base.crs} 不等于 {TARGET_CRS}"

    # ---------- 3. 计数器 ----------
    index_rows = []
    total_pos = 0
    total_neg = 0
    total_skip_partial = 0
    total_skip_small_inst = 0
    total_skip_small_only = 0
    total_skip_bg = 0
    total_skip_nodata = 0
    total_invalid_geom = 0
    total_label_lines = 0
    total_empty_txt = 0

    # ---------- 4. 遍历每幅 TIF ----------
    for tif_idx, tif_path in enumerate(tifs):
        if MAX_TIFS is not None and tif_idx >= MAX_TIFS:
            print(f"\n[限制] 已处理 {tif_idx} 幅 TIF (MAX_TIFS={MAX_TIFS}),停止。")
            break
        stem = os.path.splitext(os.path.basename(tif_path))[0]
        tif_prefix = f"t{tif_idx:02d}_{stem}"
        print(f"\n[步骤 2] 处理影像 #{tif_idx}: {stem}  (前缀={tif_prefix})")

        with rasterio.open(tif_path) as src:
            # ---- 4.1 打印原始元信息 ----
            print_tif_meta(src, tif_path)
            src_crs = src.crs

            # ---- 4.2 构建 WarpedVRT(真正重投影到 UTM 48N) ----
            print("  构建 WarpedVRT(重投影到 EPSG:32648)...")
            vrt, vrt_transform, W, H = (
                None, None, None, None
            )
            vrt_obj, dst_transform, dst_width, dst_height, dst_crs = build_vrt(src)
            vrt = vrt_obj
            vrt_transform = dst_transform
            W, H = dst_width, dst_height

            # 断言:VRT CRS 必须是 EPSG:32648
            assert vrt.crs == TARGET_CRS, \
                f"[断言失败] vrt.crs={vrt.crs}"

            # ---- 4.3 打印 VRT 元信息 + SHP 相交情况 ----
            print_vrt_meta(vrt, src, gdf_base)

            # 影像小于 TILE_SIZE 直接跳过
            if H < TILE_SIZE or W < TILE_SIZE:
                print(f"  [警告] VRT 尺寸 {W}x{H} 小于 {TILE_SIZE},跳过。")
                continue

            # ---- 4.4 SHP 已是 UTM,直接用 gdf_base 查询 VRT 范围 ----
            vrt_west, vrt_south, vrt_east, vrt_north = vrt.bounds
            vrt_box = box(vrt_west, vrt_south, vrt_east, vrt_north)
            # 断言:bounds 在合理 UTM 范围内
            # UTM 48N 标准带 X: 100k-900k,但跨带数据(经度>108°)在 32648 下东界可达 ~1.1M
            # 放宽到 100k-1.5M 兼顾跨区数据,同时拦截 0/负值/异带等严重错误
            assert 100000 < vrt_west < 1500000, f"[断言] vrt_west={vrt_west} 异常(应在 100000-1500000)"
            assert 0 < vrt_south < 10000000, f"[断言] vrt_south={vrt_south} 异常"

            cand_idx = list(gdf_base.sindex.query(vrt_box, predicate="intersects"))
            gdf_local = gdf_base.iloc[cand_idx].reset_index(drop=True)
            sidx_local = gdf_local.sindex
            print(f"  与 VRT 范围相交的目标数量 = {len(gdf_local)}")
            has_vector = len(gdf_local) > 0
            if not has_vector:
                print("  [警告] SHP 与 VRT 完全不相交,按纯背景处理(仍生成 10% 负样本)。")

            # ---- 4.5 全局拉伸参数(基于 VRT,整张一次) ----
            print("  计算 VRT 全局拉伸分位数(排除 NoData)...")
            stretch = compute_global_stretch(vrt, BANDS_RGB)
            print(f"  拉伸参数 = {[(round(a, 2), round(b, 2)) for a, b in stretch]}")

            # ---- 4.6 切片遍历 ----
            n_pos = n_neg = n_skip_partial = n_skip_small_inst = \
                n_skip_small_only = n_skip_bg = n_skip_nodata = 0
            seen = set()
            debug_count = 0
            tile_count = 0              # 已保存切片数(正+负)
            positive_buffer = []        # 暂存正样本(tile_count, info)
            negative_candidates = []     # 暂存负样本候选

            for row in range(0, H, STEP):
                for col in range(0, W, STEP):
                    r = min(row, H - TILE_SIZE)
                    c = min(col, W - TILE_SIZE)
                    if r < 0 or c < 0:
                        continue
                    if (r, c) in seen:
                        continue
                    seen.add((r, c))

                    window = Window(c, r, TILE_SIZE, TILE_SIZE)
                    # 切片级 transform(基于 VRT)
                    tile_transform = win_transform_fn(window, vrt.transform)
                    # 断言:tile_transform 必须来自 vrt.transform
                    assert tile_transform.c == vrt.transform.c + c * vrt.transform.a, \
                        "[断言] tile_transform 与 vrt.transform 不一致"

                    # ---- 4.7 窗口地理边界(用 vrt.transform) ----
                    w_west, w_south, w_east, w_north = win_bounds(
                        window, vrt.transform
                    )
                    win_box_geom = box(w_west, w_south, w_east, w_north)

                    # ---- 4.8 读 3 波段(masked=True) ----
                    data = vrt.read(BANDS_RGB, window=window, masked=True)
                    if data.shape != (len(BANDS_RGB), TILE_SIZE, TILE_SIZE):
                        print(f"    [警告] 切片 ({r},{c}) 尺寸异常 {data.shape},跳过。")
                        continue

                    # 三波段共同有效掩膜
                    if data.mask is np.ma.nomask:
                        valid_mask = np.ones((TILE_SIZE, TILE_SIZE), dtype=bool)
                    else:
                        valid_mask = ~np.any(data.mask, axis=0)
                    valid_ratio = float(valid_mask.mean())
                    if valid_ratio < VALID_PIX_RATIO_MIN:
                        n_skip_nodata += 1
                        continue

                    # 全局拉伸 -> uint8 HWC(无效像元黑色)
                    img_arr = apply_stretch_uint8(data.data, stretch, valid_mask)

                    # ---- 4.9 实例分割标签生成 ----
                    # 局部像素坐标逆仿射(基于 VRT tile_transform)
                    inv = ~tile_transform
                    M_shp = [inv.a, inv.b, inv.d, inv.e, inv.c, inv.f]

                    if has_vector:
                        m_idx = list(sidx_local.query(
                            win_box_geom, predicate="intersects"))
                        matches = gdf_local.iloc[m_idx]
                    else:
                        matches = gdf_local.iloc[0:0]

                    labels = []
                    tile_has_partial = False
                    tile_has_small_only = False

                    for _, feat in matches.iterrows():
                        geom = feat.geometry
                        if geom is None or geom.is_empty:
                            total_invalid_geom += 1
                            continue

                        # 真实几何相交(SHP 与窗口都是 UTM)
                        try:
                            inter = geom.intersection(win_box_geom)
                        except Exception:
                            inter = _make_geom_valid(geom).intersection(win_box_geom)
                        if inter.is_empty or inter.area == 0:
                            continue

                        orig_area = geom.area
                        if orig_area <= 0:
                            total_invalid_geom += 1
                            continue
                        vis_ratio = inter.area / orig_area

                        if vis_ratio < VIS_RATIO:
                            tile_has_partial = True
                            continue

                        # 转到切片局部像素坐标(基于 VRT 逆仿射)
                        try:
                            inter_px = affine_transform(inter, M_shp)
                        except Exception:
                            total_invalid_geom += 1
                            continue
                        if inter_px.area < MIN_PIX_AREA:
                            tile_has_small_only = True
                            n_skip_small_inst += 1
                            continue

                        instances = geometry_to_instances(inter_px)
                        if not instances:
                            tile_has_small_only = True
                            n_skip_small_inst += 1
                            continue

                        for ring in instances:
                            if len(ring) < MIN_VERTICES:
                                tile_has_small_only = True
                                n_skip_small_inst += 1
                                continue
                            coords_norm = []
                            for (x, y) in ring:
                                xn = min(max(x / TILE_SIZE, 0.0), 1.0)
                                yn = min(max(y / TILE_SIZE, 0.0), 1.0)
                                coords_norm.append((xn, yn))
                            if len(coords_norm) < MIN_VERTICES:
                                tile_has_small_only = True
                                n_skip_small_inst += 1
                                continue
                            if MAX_VERTICES is not None and \
                               len(coords_norm) > MAX_VERTICES:
                                idx_arr = np.linspace(
                                    0, len(coords_norm) - 1,
                                    MAX_VERTICES
                                ).astype(int)
                                coords_norm = [coords_norm[i] for i in idx_arr]
                            flat = " ".join(
                                f"{x:.6f} {y:.6f}" for x, y in coords_norm)
                            labels.append(f"0 {flat}")

                    # ---- 4.10 调试打印 ----
                    if debug_count < DEBUG_N:
                        debug_count += 1
                        if labels:
                            all_coords = []
                            for ln in labels:
                                parts = ln.split()[1:]
                                all_coords.extend([float(v) for v in parts])
                            if all_coords:
                                arr_c = np.array(all_coords)
                                xmin = arr_c[0::2].min()
                                xmax = arr_c[0::2].max()
                                ymin = arr_c[1::2].min()
                                ymax = arr_c[1::2].max()
                            else:
                                xmin = xmax = ymin = ymax = -1
                        else:
                            xmin = xmax = ymin = ymax = -1
                        print(f"    [DEBUG] tile=({r},{c}) "
                              f"win_px=({c},{r},{TILE_SIZE},{TILE_SIZE}) "
                              f"utm=({w_west:.1f},{w_south:.1f},"
                              f"{w_east:.1f},{w_north:.1f}) "
                              f"n_match={len(matches)} "
                              f"n_inst={len(labels)} "
                              f"x[{xmin:.3f},{xmax:.3f}] "
                              f"y[{ymin:.3f},{ymax:.3f}] "
                              f"vrt_crs={vrt.crs} "
                              f"gdf_crs={gdf_base.crs}")

                    # ---- 4.11 切片决策 ----
                    tile_name = f"{tif_prefix}_r{r:05d}_c{c:05d}"
                    img_path = os.path.join(IMG_DIR, tile_name + ".png")
                    lbl_path = os.path.join(LBL_DIR, tile_name + ".txt")

                    if tile_has_partial:
                        n_skip_partial += 1
                        continue
                    elif labels:
                        # 正样本:保存图像 + 非空 txt
                        save_png(img_arr, img_path)
                        with open(lbl_path, "w", encoding="utf-8") as f:
                            f.write("\n".join(labels))
                        n_pos += 1
                        total_label_lines += len(labels)
                        idx_status = "positive"
                        idx_is_neg = 0
                        idx_inst = len(labels)
                        # 正样本同时生成质检叠加图(前若干个)
                        if len(positive_buffer) < 5:
                            overlay_path = os.path.join(
                                OVERLAY_DIR, tile_name + "_overlay.png")
                            save_overlay_png(img_arr, labels, overlay_path)
                        positive_buffer.append((
                            tile_count,
                            {
                                "tile_name": tile_name,
                                "source_tif": tif_path,
                                "source_crs": str(src_crs),
                                "target_crs": str(vrt.crs),
                                "utm_xmin": w_west, "utm_ymin": w_south,
                                "utm_xmax": w_east, "utm_ymax": w_north,
                                "pixel_size_m": abs(vrt.transform.a),
                                "split": "train",
                                "instance_count": len(labels),
                                "is_negative": 0,
                            }
                        ))
                        tile_count += 1
                    elif tile_has_small_only:
                        n_skip_small_only += 1
                        continue
                    else:
                        # 纯背景:10% 保留为负样本
                        if random.random() < NEG_KEEP:
                            save_png(img_arr, img_path)
                            write_empty_txt(lbl_path)
                            n_neg += 1
                            total_empty_txt += 1
                            idx_status = "negative"
                            idx_is_neg = 1
                            idx_inst = 0
                            negative_candidates.append({
                                "tile_name": tile_name,
                                "source_tif": tif_path,
                                "source_crs": str(src_crs),
                                "target_crs": str(vrt.crs),
                                "utm_xmin": w_west, "utm_ymin": w_south,
                                "utm_xmax": w_east, "utm_ymax": w_north,
                                "pixel_size_m": abs(vrt.transform.a),
                                "split": "train",
                                "instance_count": 0,
                                "is_negative": 1,
                            })
                            tile_count += 1
                        else:
                            n_skip_bg += 1
                            continue

                    # MAX_SAVED_TILES 检查(正样本优先保留)
                    if MAX_SAVED_TILES is not None and \
                       tile_count >= MAX_SAVED_TILES:
                        # 已达配额:正样本优先(配额内尽量多正样本)
                        # 这里不做截断调整,直接跳出
                        break

                if MAX_SAVED_TILES is not None and \
                   tile_count >= MAX_SAVED_TILES:
                    break

            # ---- 4.12 写入索引(正样本优先,补足负样本到配额) ----
            # 正样本全部写入
            for _, row_dict in positive_buffer:
                index_rows.append(row_dict)
            # 负样本按比例写入(若正样本不足配额,负样本补足)
            n_neg_to_keep = len(negative_candidates)
            if MAX_SAVED_TILES is not None:
                n_neg_to_keep = max(0, MAX_SAVED_TILES - len(positive_buffer))
                n_neg_to_keep = min(n_neg_to_keep, len(negative_candidates))
            for row_dict in negative_candidates[:n_neg_to_keep]:
                index_rows.append(row_dict)

            total_pos += n_pos
            total_neg += n_neg
            total_skip_partial += n_skip_partial
            total_skip_small_inst += n_skip_small_inst
            total_skip_small_only += n_skip_small_only
            total_skip_bg += n_skip_bg
            total_skip_nodata += n_skip_nodata
            print(f"  本 TIF: 正样本={n_pos}, 负样本={n_neg}, "
                  f"严重截断跳过={n_skip_partial}, "
                  f"小目标实例跳过={n_skip_small_inst}, "
                  f"小目标整片跳过={n_skip_small_only}, "
                  f"NoData跳过={n_skip_nodata}, 背景丢弃={n_skip_bg}")

    # ---------- 5. 写索引 CSV ----------
    with open(INDEX_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        writer.writerows(index_rows)

    # ---------- 6. 标签验证 ----------
    print("\n[步骤 3] 验证所有标签文件格式...")
    n_total_lbl, n_ok_lbl, lbl_errors = validate_labels(LBL_DIR)

    # ---------- 7. PNG/TXT 配对检查 ----------
    pngs = set(os.path.splitext(f)[0]
               for f in os.listdir(IMG_DIR)
               if f.lower().endswith(".png"))
    txts_set = set(os.path.splitext(f)[0]
                   for f in os.listdir(LBL_DIR)
                   if f.lower().endswith(".txt"))
    png_only = pngs - txts_set
    txt_only = txts_set - pngs
    pair_check_ok = (len(png_only) == 0 and len(txt_only) == 0)

    # ---------- 8. 汇总 ----------
    print("\n" + "=" * 60)
    print("数据准备完成(实例分割版 v4 / UTM WarpedVRT)")
    print(f"  处理 TIF 数               : {min(len(tifs), MAX_TIFS or len(tifs))}")
    print(f"  SHP 总数                  : {len(shps)}")
    print(f"  正样本切片数              : {total_pos}")
    print(f"  负样本切片数              : {total_neg}")
    print(f"  跳过的严重截断切片数      : {total_skip_partial}")
    print(f"  无效几何数                : {total_invalid_geom}")
    print(f"  标签行数                  : {total_label_lines}")
    print(f"  空 TXT 数                 : {total_empty_txt}")
    n_bad_lbl = n_total_lbl - n_ok_lbl
    print(f"  异常标签数                : {n_bad_lbl}")
    print(f"  小目标实例跳过            : {total_skip_small_inst}")
    print(f"  小目标整片跳过            : {total_skip_small_only}")
    print(f"  背景丢弃数量              : {total_skip_bg}")
    print(f"  NoData过多跳过             : {total_skip_nodata}")
    print(f"  输出 PNG 数量             : {len(pngs)}")
    print(f"  输出 TXT 数量             : {len(txts_set)}")
    print(f"  PNG/TXT 配对检查          : "
          f"{'PASS' if pair_check_ok else 'FAIL'}"
          f"(PNG-only={len(png_only)}, TXT-only={len(txt_only)})")
    print(f"  标签格式验证              : {n_ok_lbl}/{n_total_lbl} 合规")
    if lbl_errors:
        print(f"  标签错误(前 10 条):")
        for e in lbl_errors[:10]:
            print(f"    - {e}")
    print(f"  图像目录                  : {IMG_DIR}")
    print(f"  标签目录                  : {LBL_DIR}")
    print(f"  质检叠加图目录            : {OVERLAY_DIR}")
    print(f"  索引文件                  : {INDEX_CSV}")
    print("=" * 60)


if __name__ == "__main__":
    main()
