import os
import glob
import json
import numpy as np
import cv2

def batch_generate_he_masks(json_dir, output_dir):
    # 建立输入输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # 提取所有 json 文件路径
    json_pattern = os.path.join(json_dir, '*.json')
    json_files = glob.glob(json_pattern)
    
    if not json_files:
        print("无法推演：指定目录下未找到 .json 文件。")
        return

    # 遍历处理
    for json_path in json_files:
        # 解析文件名，用于生成对应的 mask 文件名
        filename = os.path.basename(json_path)
        base_name, _ = os.path.splitext(filename)
        save_path = os.path.join(output_dir, f"{base_name}_mask.png")
        
        # 载入 JSON 数据
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        h = data['imageHeight']
        w = data['imageWidth']
        # 初始化单通道全黑背景
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # 遍历形状并严格过滤
        for shape in data['shapes']:
            label = shape['label']
            
            # 核心拦截逻辑：只处理硬性渗出
            if label == "hard exudate":
                points = shape['points']
                pts = np.array(points, np.int32)
                # 填充为 255 (白色)
                cv2.fillPoly(mask, [pts], color=255)
                
        # 物理写入磁盘
        cv2.imwrite(save_path, mask)
        print(f"处理完毕: {filename} -> {base_name}_mask.png")

# 物理路径执行参数
input_directory = '/mnt/mydata/gww/sj/syndiff/SynDiff/新建文件夹'
# 建议将 mask 输出到独立目录以防止文件污染
output_directory = '/mnt/mydata/gww/sj/syndiff/SynDiff/he_masks'

batch_generate_he_masks(input_directory, output_directory)