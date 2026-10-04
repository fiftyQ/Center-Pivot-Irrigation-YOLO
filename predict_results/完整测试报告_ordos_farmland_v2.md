# 鄂尔多斯圆形农田 YOLO 实例分割 —— 完整测试报告

- **报告日期**：2026-10-04
- **模型实验名**：`ordos_farmland_v2_clean_split`
- **模型权重**：`E:\260827YOLORUN\runs\segment\ordos_farmland_v2_clean_split\weights\best.pt`
- **预测脚本**：`E:\260827YOLORUN\06_predict_to_gis_v3.py`（FULL_MODE=True）
- **Python 解释器**：`D:\Anacondaanzhuang\envs\yolotest\python.exe`

---

## 0. 数据来源与可信度声明

本报告所有数字均来自以下两类**已实际执行并留存**的证据，无推算/脑补值：

1. **脚本实际运行 stdout**（exit code = 0）：阶段 A 评估报告、strict30 诊断、全量预测逐 TIF 与总计统计。
2. **geopandas 独立读取 Shapefile 的核验结果**：CRS、几何有效性、字段、source_tif 分布、面积/置信度/圆度分布。

交叉校验：阶段 A 的四阈值指标在 strict30 运行与全量运行中**两次推理结果完全一致**（实例数/TP/FP/FN 逐项相同），数字可复现。

---

## 1. 运行环境

| 项 | 值 |
|---|---|
| 操作系统 | Windows |
| GPU | RTX 4060 Laptop GPU（8188 MiB） |
| CUDA / PyTorch | CUDA 12.1 / PyTorch 2.5.1+cu121 |
| 系统内存 | 16 GB（推理与训练 workers 均据此调优） |
| 模型任务类型 | `model.task = segment`（实例分割，已断言） |
| 框架 | Ultralytics YOLO11n-seg |

### 推理参数

| 参数 | 值 |
|---|---|
| imgsz | 640 |
| device | 0 |
| retina_masks | True |
| TIF 推理置信度 TIF_INFER_CONF | 0.65 |
| NMS iou（Ultralytics 内部） | 0.50 |
| 滑窗尺寸 / 步长 | 640 × 640 / 512（相邻窗重叠 128 像素） |
| 实例级去重 Polygon-IoU 阈值 | 0.70（边缘候选对非边缘保留面 IoU≥0.50 也去重） |

---

## 2. 数据集与无泄漏划分

### 2.1 背景：旧数据集存在空间泄漏

旧 `yolo_dataset` 采用 8:2 随机切片划分，train/val 共享全部 6 幅 source_tif，**71.8%（145/202）的 val 切片与 train 切片 UTM bbox 重叠**；且 6 幅原始 TIF 两两大面积重叠（1275~1962 km²）形成一个空间连通分量。旧 val 指标不能代表泛化性能。

### 2.2 新数据集 yolo_dataset_v2（切片级空间区域划分）

按 X 向三条带 + 缓冲带划分，跨集合切片 UTM bbox 重叠 = 0：

| 集合 | 切片总数 | 含标注正样本数 |
|---|---:|---:|
| train | 600 | 399 |
| val | 189 | 33 |
| **test（独立，未参与训练/调参）** | **196** | **12** |

---

## 3. 模型训练结果

| 项 | 值 |
|---|---|
| 训练轮次 | 100 epochs（第 77 epoch 因内存中断，workers 4→2 后续跑完成） |
| best epoch | 64 |
| Mask Precision | 0.8425 |
| Mask Recall | 0.6501 |
| Mask mAP50 | 0.6866 |
| Mask mAP50-95 | 0.4892 |

> 注：训练指标来自 val；下文第 4 节的 test 指标来自空间无重叠的独立 test 集，用于无偏泛化评估。

---

## 4. 阶段 A —— 独立 test 集四阈值评估

- **评估集**：`yolo_dataset_v2/images/test`，196 张，全部有对应真实 TXT 标签。
- **匹配规则**：逐图贪心一对一 IoU≥0.50 匹配预测多边形与 GT 多边形，计算 TP/FP/FN。
- **输出**：每档阈值目录下含 mask 填充叠加图、红色预测/绿色 GT 边界图、预测 TXT、report.txt。
- **输出目录**：`predict_results\v2_test\conf_035 | conf_050 | conf_065 | conf_075`

### 4.1 四阈值核心指标

| 指标 | conf≥0.35 | conf≥0.50 | conf≥0.65 | conf≥0.75 |
|---|---:|---:|---:|---:|
| 图片总数 | 196 | 196 | 196 | 196 |
| 有预测 mask 的图片数 | 23 | 19 | 12 | 11 |
| 预测实例总数 | 85 | 70 | 56 | 49 |
| 每图预测数 mean / median / max | 0.43 / 0 / 19 | 0.36 / 0 / 16 | 0.29 / 0 / 14 | 0.25 / 0 / 13 |
| mask 多边形平均顶点数 | 47.4 | 49.2 | 50.2 | 50.9 |
| mask 面积(px²) min / max / mean | 404 / 5274 / 2286 | 404 / 5274 / 2330 | 980 / 5274 / 2339 | 980 / 5274 / 2350 |
| **TP / FP / FN** | **62 / 23 / 19** | **59 / 11 / 22** | **53 / 3 / 28** | **47 / 2 / 34** |
| Precision | 0.729 | 0.843 | **0.946** | 0.959 |
| Recall | 0.765 | 0.728 | 0.654 | 0.580 |
| **F1** | 0.747 | **0.781（最高）** | 0.774 | 0.723 |
| 误检数（=FP） | 23 | 11 | 3 | 2 |
| 漏检数（=FN） | 19 | 22 | 28 | 34 |
| 正确识别数（=TP） | 62 | 59 | 53 | 47 |

### 4.2 置信度区间分布（实例数）

| 阈值 | 0.35–0.50 | 0.50–0.65 | 0.65–0.75 | ≥0.75 |
|---|---:|---:|---:|---:|
| conf≥0.35 | 15 | 14 | 7 | 49 |
| conf≥0.50 | 0 | 14 | 7 | 49 |
| conf≥0.65 | 0 | 0 | 7 | 49 |
| conf≥0.75 | 0 | 0 | 0 | 49 |

### 4.3 结论（基于上表，不作外推）

- **F1 最佳点为 conf=0.50（0.781）**；**生产环境采用 conf=0.65**：Precision=0.946、误检仅 3 个，与全量 TIF 推理阈值一致。
- ≥0.75 高置信实例占 conf≥0.35 总量的 49/85（57.6%），模型对清晰圆形农田预测稳定。
- 196 张 test 切片中仅 11–23 张有预测，多数纯背景切片被正确零预测（test 正样本仅 12 张，存在同一正样本多实例情况，故有预测图数可高于正样本数）。

---

## 5. strict30 小规模诊断（计数逻辑校准）

为修正早期"批量推理 overshoot 导致停止条件失真"的问题，采用 `INFER_BATCH=1`、达到 30 个**含 mask 窗口**后立即停止提交新批次，并使用独立输出路径。

### 5.1 六项独立计数器

| 字段 | 值 |
|---|---:|
| target MAX_VALID_WINDOWS（含 mask 窗口上限） | 30 |
| **windows_with_masks（实际含 mask 窗口）** | **30（严格达到，True）** |
| total_windows_seen | 485 |
| valid_windows_seen | 485 |
| raw_candidates | 63 |
| nms_kept | 42（nms_dropped=19，edge_dropped=2） |
| filtered | 42（low_conf=0 / too_small=0 / too_large=0 / low_compactness=0） |
| skip | no_mask=455，其余 5 类全 0 |

### 5.2 SHP 核验（strict30）

| 项 | 值 |
|---|---|
| all / filtered 要素数 | 42 / 42 |
| CRS | EPSG:32648 |
| source_tif 唯一数 | 1（Ordos_S2_MultiBand_2025-0000000000-0000000000.tif） |
| window_row / col 范围 | row 7680–13312 / col 2560–14336（VRT 中心带） |
| confidence min / mean / max | 0.6507 / 0.8081 / 0.9046 |
| area_m² min / mean / max | 26,400 / 141,690 / 467,200 |
| compactness min / mean / max | 0.6200 / 0.8656 / 0.8979 |
| is_edge | 非边缘 40 / 边缘 2 |
| geometry | 全部 valid，无空几何 |

输出：`farmlands_2025_v2_strict30_all.shp` / `farmlands_2025_v2_strict30_filtered.shp`；调试图目录 `debug_overlays_v3_strict30`。

---

## 6. 全量预测（6 幅 TIF）

### 6.1 执行参数

```
MAX_TIFS = None（全部 6 幅）
MAX_VALID_WINDOWS = None（无窗口上限）
INFER_BATCH = 8
TIF_INFER_CONF = 0.65    imgsz = 640    iou = 0.50
device = 0               retina_masks = True
```

预处理（与 `data_prepare.py` 完全一致）：原始 WGS84 TIF → EPSG:32648 WarpedVRT（10 m 方像元、bilinear、nodata=0）→ RGB 波段 [3,2,1] → combined-valid mask → 仅有效像素 2%–98% 全局直方图拉伸 → (640,640,3) uint8，无效区域填 [0,0,0]。

### 6.2 各 TIF 的 VRT 与全局拉伸分位数

| TIF | 原始尺寸(WGS84) | VRT 尺寸(EPSG:32648) | R 2%/98% | G 2%/98% | B 2%/98% |
|---|---|---|---|---|---|
| ...0000000000-0000000000 | 18944×18944 | 15016×19272 | 1.00 / 2910.67 | 1.00 / 2178.14 | 1.00 / 1515.58 |
| ...0000000000-0000018944 | 18944×18944 | 15382×19555 | 1.00 / 2887.36 | 1.00 / 2132.60 | 1.00 / 1435.15 |
| ...0000000000-0000037888 | 17513×18944 | 14641×19762 | 1.00 / 2173.49 | 1.00 / 1788.71 | 1.00 / 1345.70 |
| ...0000018944-0000000000 | 18944×17383 | 15300×17707 | 1.00 / 2896.99 | 1.00 / 2154.92 | 1.00 / 1493.01 |
| ...0000018944-0000018944 | 18944×17383 | 15626×17988 | 1.00 / 2863.16 | 1.00 / 2125.80 | 1.00 / 1435.67 |
| ...0000018944-0000037888 | 17513×17383 | 14822×18194 | —（该幅无检出，拉伸已计算） |

### 6.3 逐 TIF 窗口与实例统计

| TIF | 窗口网格 | total_windows_seen | valid_windows_seen | windows_with_masks | raw_candidates | nms_kept（归属保留侧） | filtered |
|---|---|---:|---:|---:|---:|---:|---:|
| ...0000000000-0000000000 | 37×29=1073 | 1073 | 1066 | 127 | 459 | 326 | 325 |
| ...0000000000-0000018944 | 37×29=1073 | 1073 | 1054 | 99 | 401 | 284 | 277 |
| ...0000000000-0000037888 | 38×28=1064 | 1064 | 1008 | 46 | 101 | 81 | 80 |
| ...0000018944-0000000000 | 34×29=986 | 986 | 986 | 327 | 1484 | 999 | 998 |
| ...0000018944-0000018944 | 34×30=1020 | 1020 | 999 | 234 | 1049 | 732 | 729 |
| ...0000018944-0000037888 | — | 980 | 940 | **0** | 0 | 0 | 0 |

逐 TIF skip 明细：

| TIF | low_valid | no_mask | geom_err | read_err / size_mismatch / infer_err |
|---|---:|---:|---:|---:|
| ...0000000000-0000000000 | 7 | 939 | 63 | 0 / 0 / 0 |
| ...0000000000-0000018944 | 19 | 955 | 55 | 0 / 0 / 0 |
| ...0000000000-0000037888 | 56 | 962 | 5 | 0 / 0 / 0 |
| ...0000018944-0000000000 | 0 | 659 | 23 | 0 / 0 / 0 |
| ...0000018944-0000018944 | 21 | 765 | 23 | 0 / 0 / 0 |
| ...0000018944-0000037888 | 40 | 940 | 0 | 0 / 0 / 0 |

### 6.4 总计

| 项 | 值 |
|---|---:|
| 处理 TIF 数 | 6 |
| total_windows_seen | **6196** |
| valid_windows_seen | **6053** |
| windows_with_masks | **833** |
| raw_candidates（NMS 前） | **3494** |
| **NMS 前 → NMS 后** | **3494 → 2422**（nms_dropped=966，edge_dropped=106） |
| **filtered** | **2409** |
| 过滤原因 | too_large=13；low_conf=0 / too_small=0 / low_compactness=0 |
| skip 总计 | low_valid=143，no_mask=5220，geom_err=169，read_err=0，size_mismatch=0，infer_err=0 |

> 说明：6 幅 TIF 空间两两重叠，NMS 在全部 3494 个候选上**全局执行一次**；跨 TIF 重复实例只保留置信度更高一侧，因此逐 TIF 的 nms_kept 之和（2422）等于全局 NMS 后总数，raw_candidates 之和（3494）等于全局 NMS 前总数。第 6 幅 TIF（...0000018944-0000037888）940 个有效窗口零检出，表示该景覆盖区域在 conf=0.65 下无圆形农田实例，不等同于数据异常。

---

## 7. Shapefile 独立核验（geopandas 读取）

| 项 | farmlands_2025_v2_all_full.shp | farmlands_2025_v2_filtered_full.shp |
|---|---|---|
| 要素数 | **2422** | **2409** |
| CRS | **EPSG:32648** | **EPSG:32648** |
| geometry 全部 valid | True | True |
| 空几何数 | 0 | 0 |
| area_m² ≤ 0 | 0 | 0 |
| area_m² > 600,000 | 13（保留于 all） | 0（已按 too_large 剔除） |
| area_m² min / median / mean / max | 19,600 / 121,450 / 159,392 / 1,059,700 | 19,600 / 121,050 / 156,192 / 599,800 |
| confidence min / mean / max | 0.6502 / 0.8118 / 0.9711 | 0.6502 / 0.8116 / 0.9367 |
| compactness min / mean / max | 0.3673 / 0.8619 / 0.9235 | 0.3673 / 0.8624 / 0.9235 |
| vertex_count min / max | 10 / 144 | 10 / 119 |
| is_edge | 非边缘 2311 / 边缘 111 | 非边缘 2298 / 边缘 111 |

属性字段（11 项 + geometry）：`id, confidence, area_m2, perimeter, compactness, circularity, vertex_count, is_edge, source_tif, window_row, window_col, geometry`。
（Shapefile 10 字符字段名限制，写入时自动 launder 为 `compactnes / circularit / vertex_cou`，字段含义与数值不变。）

### source_tif 要素分布

| source_tif | all | filtered |
|---|---:|---:|
| ...0000018944-0000000000 | 999 | 998 |
| ...0000018944-0000018944 | 732 | 729 |
| ...0000000000-0000000000 | 326 | 325 |
| ...0000000000-0000018944 | 284 | 277 |
| ...0000000000-0000037888 | 81 | 80 |
| ...0000018944-0000037888 | 0 | 0 |

---

## 8. 方法合规性核对

| 要求 | 落实情况 |
|---|---|
| 使用 `result.masks.xy` 实例分割多边形，禁止检测框替代 | ✅ 全程仅用 masks.xy |
| 禁止 Hough 圆 / OpenCV 轮廓 / 边缘检测 / 决策树 | ✅ 未使用 |
| 坐标转换用 `rasterio.windows.transform(window, vrt.transform)` 正向映射至 EPSG:32648 | ✅ 未用 WGS84 src.transform 或逆变换 |
| 预处理与 data_prepare.py 一致（CRS/10 m 像元/RGB[3,2,1]/nodata/combined-valid/2-98 拉伸/uint8） | ✅ 函数逐行对齐 |
| 实例级 Polygon-IoU NMS，IoU≥0.70 判重 | ✅ |
| 禁止 unary_union / dissolve / 按 intersects·touches 合并相邻农田 | ✅ NMS 仅删低置信重复面，不合并几何（unary_union 仅用于计算调试图 bounds） |
| all 保留 NMS 后全部有效实例；filtered 仅按 conf≥0.65、面积 10000–600000 m²、圆度≥0.35 过滤 | ✅ 过滤原因单独统计，未改 all |
| 输出 CRS EPSG:32648 且属性齐全 | ✅ geopandas 核验通过 |
| 不删除旧模型/旧 SHP/旧 debug，不全量覆盖 | ✅ strict30、_full、旧 v3 文件各自独立 |

---

## 9. 输出文件清单

### 9.1 test 集评估
- `predict_results\v2_test\conf_035\`（填充图 / 边界图 / TXT / report.txt）
- `predict_results\v2_test\conf_050\`
- `predict_results\v2_test\conf_065\`
- `predict_results\v2_test\conf_075\`

### 9.2 strict30 诊断
- `predict_results\farmlands_2025_v2_strict30_all.shp`（n=42）
- `predict_results\farmlands_2025_v2_strict30_filtered.shp`（n=42）
- `predict_results\debug_overlays_v3_strict30\`（3 张调试图 + 输入样例）

### 9.3 全量预测
- `predict_results\farmlands_2025_v2_all_full.shp`（**n=2422，EPSG:32648**）
- `predict_results\farmlands_2025_v2_filtered_full.shp`（**n=2409，EPSG:32648**）
- `predict_results\debug_overlays_v3_full\`：6 个 TIF 子目录 × 3 张叠加图（1_raw_candidates / 2_after_nms / 3_filtered）= 18 张，加 48 张窗口输入样例

---

## 10. 已知局限与建议（仅陈述已观测事实）

1. **test 正样本仅 12 张**，test 指标的置信区间较宽；conf=0.50 与 0.65 的 F1（0.781 vs 0.774）差距小于样本波动量级，阈值选择以"误检代价"为主观依据，生产采用 0.65。
2. **geom_err=169**：masks.xy 转换后几何退化（<3 有效点 / 空面 / 面积<1 m²）被直接丢弃并计数，未计入 raw_candidates。
3. **all_full 中 13 个实例面积 > 600,000 m²**（最大 1,059,700 m²），多为边缘窗口拼接触边的异常大面；filtered 已全部剔除，建议人工在 GIS 中抽查这 13 例（属性 is_edge 可辅助定位）。
4. 第 6 幅 TIF 零检出为单阈值（0.65）结论；若需确认该区域是否存在低置信农田，可用 conf=0.50 对该景单独复扫对比（不改变当前正式结果）。
