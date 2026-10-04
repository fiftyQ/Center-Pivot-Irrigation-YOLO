import os

# 替换为你的标签文件夹路径
label_dirs = [
    r"E:\260827YOLORUN\yolo_dataset\labels\train",
    r"E:\260827YOLORUN\yolo_dataset\labels\val"
]

error_files = []

for label_dir in label_dirs:
    for txt_file in os.listdir(label_dir):
        if not txt_file.endswith(".txt"):
            continue
        
        filepath = os.path.join(label_dir, txt_file)
        with open(filepath, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    # 检查后四个数值是否在 0~1 之间
                    for val in parts[1:5]:
                        if float(val) > 1.0 or float(val) < 0.0:
                            error_files.append(filepath)
                            break # 只要发现一行错，就记录该文件

if error_files:
    print(f"⚠️ 发现 {len(error_files)} 个文件的坐标未归一化！")
    print("示例：", error_files[:5])
else:
    print("✅ 完美！所有 txt 文件的坐标均在 0-1 范围内。")