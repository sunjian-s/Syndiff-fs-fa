import cv2
import numpy as np
import os

# === 这里是你指定的图片路径 ===
img_path = "/mnt/mydata/gww/sj/syndiff/SynDiff/test_set/fundus/25.jpg"

def ben_graham_preprocessing(image_path, sigmaX=10):
    # 1. 读取图片
    if not os.path.exists(image_path):
        print(f"错误：找不到文件 {image_path}")
        return None
        
    image = cv2.imread(image_path)
    if image is None:
        print("错误：无法读取图片，可能是格式损坏")
        return None

    # 2. 裁剪去黑边 (保留眼球区域)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = gray > 7
    
    # 防止全黑图报错
    if mask.sum() == 0:
        print("警告：图片几乎全黑")
        return image
        
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    image = image[rmin:rmax, cmin:cmax]

    # 3. 调整尺寸 (这里先统一Resize到512查看效果，你可以按需修改)
    image = cv2.resize(image, (512, 512))
    
    # 4. 核心处理：Ben Graham 方法
    # output = image * 4 + GaussianBlur(image) * (-4) + 128
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX)
    processed = cv2.addWeighted(image, 4, blurred, -4, 128)
    
    return processed

if __name__ == "__main__":
    print(f"正在处理: {img_path} ...")
    
    result = ben_graham_preprocessing(img_path)
    
    if result is not None:
        # 构造保存路径：原文件名_processed.jpg
        dir_name = os.path.dirname(img_path)
        base_name = os.path.basename(img_path)
        file_name, ext = os.path.splitext(base_name)
        
        save_path = os.path.join(dir_name, f"{file_name}_processed{ext}")
        
        cv2.imwrite(save_path, result)
        print(f"处理完成！\n保存路径: {save_path}")
        print("现在你可以下载或打开这张图，观察病灶是否变清晰了。")