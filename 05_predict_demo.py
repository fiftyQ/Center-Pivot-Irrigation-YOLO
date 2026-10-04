import os
from ultralytics import YOLO

def main():
    # ① 先确认权重路径真的存在,别路径错了白跑
    weights = r"E:\260827YOLORUN\runs\segment\ordos_farmland_v1\weights\best.pt"
    if not os.path.exists(weights):
        print(f"[错误] 找不到权重文件: {weights}")
        print("       请检查训练时 project/name 的设置,确认实际输出目录。")
        return

    # ② 加载分割权重
    model = YOLO(weights)

    val_dir = r"E:\260827YOLORUN\yolo_dataset\images\val"
    if not os.path.isdir(val_dir):
        print(f"[错误] 找不到验证集目录: {val_dir}")
        return

    print("开始在验证集切片上进行实例分割预测...")

    # ③ 推理:关掉方框,只画多边形掩码
    results = model.predict(
        source=val_dir,
        imgsz=640,          # 和训练保持一致
        conf=0.25,          # 置信度阈值,太低会出一堆误检
        device=0,           # 用 GPU
        save=True,          # 保存可视化图
        show=False,
        boxes=False,        # 关键:不要方框,只要掩码多边形
        save_txt=True,      # 顺便导出每个地块的多边形坐标(txt)
        save_conf=True,     # txt 里带上置信度
        project="runs/predict",   # 明确输出位置,不依赖当前工作目录
        name="farmland_seg_out",
    )
    print(f"\n预测完成! 结果目录: runs/predict/farmland_seg_out")

if __name__ == "__main__":
    main()
