import json
import os
import numpy as np
import cv2

# ================= 配置区域 =================
# 1. 设置您的输入文件夹路径
input_folder = "/mnt/mydata/gww/sj/syndiff/SynDiff/新建文件夹"

# 2. 设置输出掩码的文件夹名称 (会自动创建在 input_folder 里面)
output_folder = os.path.join(input_folder, "masks")

# 3. 定义类别映射 (根据您 JSON 中的 label)
# 这里的数字 1, 2 代表 mask 图片中的像素值。
# 背景默认为 0 (黑色)
class_mapping = {
    "hemorrhage": 1,    # 出血点 -> 像素值 1
    "hard exudate": 2   # 硬性渗出 -> 像素值 2
}
# ===========================================

def main():
    # 如果输出目录不存在，则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"已创建输出目录: {output_folder}")

    # 获取目录下所有文件
    files = os.listdir(input_folder)
    json_files = [f for f in files if f.endswith(".json")]

    if not json_files:
        print(f"错误: 在 {input_folder} 下未找到 JSON 文件。")
        return

    print(f"找到 {len(json_files)} 个 JSON 文件，开始转换...")

    for json_file in json_files:
        json_path = os.path.join(input_folder, json_file)
        
        # 读取 JSON
        try:
            with open(json_path, "r", encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"读取失败 {json_file}: {e}")
            continue

        # 获取图像尺寸
        h = data["imageHeight"]
        w = data["imageWidth"]

        # 创建单通道黑色背景 (uint8)
        mask = np.zeros((h, w), dtype=np.uint8)

        # 遍历该图片中的每一个标注形状
        for shape in data["shapes"]:
            label = shape["label"]
            points = np.array(shape["points"], dtype=np.int32)

            if label in class_mapping:
                class_value = class_mapping[label]
                # fillPoly 用于填充多边形
                # 注意：这会覆盖之前的像素，所以如果有重叠，后画的会覆盖先画的
                cv2.fillPoly(mask, [points], class_value)
            else:
                print(f"警告: 文件 {json_file} 中存在未定义的标签 '{label}'，已跳过。")

        # 保存掩码图片
        # 对应的文件名通常保持一致，后缀改为 .png
        # 例如: 1.json -> 1.png
        filename_no_ext = os.path.splitext(json_file)[0]
        save_path = os.path.join(output_folder, f"{filename_no_ext}.png")
        
        cv2.imwrite(save_path, mask)
        print(f"已转换: {json_file} -> {save_path}")

    print("转换完成！")

if __name__ == "__main__":
    main()