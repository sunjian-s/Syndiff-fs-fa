import os
import cv2
import numpy as np
from tqdm import tqdm  # 如果没有装 tqdm，把这行删掉，并把下文的 tqdm(range(...)) 改为 range(...)

# ==========================================
# 1. 核心工具函数
# ==========================================

def get_fov_center(img_shape, fov_mask):
    """
    计算眼球视场的几何中心 (x, y)
    """
    M = cv2.moments(fov_mask)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        # 如果算不出来，默认用图像中心
        cX, cY = img_shape[1] // 2, img_shape[0] // 2
    return (cX, cY)

def remove_fovea(binary_mask, fov_center, safe_radius=80):
    """
    剔除位于视场中心附近的区域 (中心凹误判)
    input: mask, center(x,y), radius(像素距离)
    """
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output_mask = binary_mask.copy()
    
    cX, cY = fov_center
    
    for cnt in contours:
        # 计算该轮廓的质心
        M = cv2.moments(cnt)
        if M["m00"] == 0: continue
        obj_X = int(M["m10"] / M["m00"])
        obj_Y = int(M["m01"] / M["m00"])
        
        # 计算距离中心的距离
        distance = np.sqrt((obj_X - cX)**2 + (obj_Y - cY)**2)
        
        # 如果距离小于安全半径 (比如 80 像素)，且面积适中，认为是中心凹
        # 通常中心凹不会特别大，也不会特别偏
        if distance < safe_radius:
            # 直接涂黑 (剔除)
            cv2.drawContours(output_mask, [cnt], -1, 0, -1)
            
    return output_mask

def filter_by_shape_v5(binary_mask):
    """
    V5 几何筛选 (保持不变)
    """
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output_mask = np.zeros_like(binary_mask)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 30: continue # 过滤极小噪点
            
        rect = cv2.minAreaRect(cnt)
        (center, (w, h), angle) = rect
        if w == 0 or h == 0: continue
        aspect_ratio = max(w, h) / min(w, h)
        
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0: continue
        solidity = float(area) / hull_area
        
        # 大块病灶保护
        if area > 300:
            if aspect_ratio < 4.0:
                cv2.drawContours(output_mask, [cnt], -1, 255, -1)
            continue 
            
        # 中小病灶严管
        else:
            if aspect_ratio > 2.2: continue 
            if solidity < 0.75: continue   
            cv2.drawContours(output_mask, [cnt], -1, 255, -1)
            
    return output_mask

def get_hemorrhage_mask(img_bgr):
    """
    主处理流程 V6: 预处理 -> 阈值 -> 几何筛选 -> 【中心凹剔除】
    """
    # 1. 预处理
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # 2. 提取 FOV 并腐蚀
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    _, fov_mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    
    # 计算 FOV 中心 (用于后续剔除中心凹)
    fov_center = get_fov_center(img_rgb.shape, fov_mask)
    
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    fov_mask = cv2.erode(fov_mask, kernel_erode)
    
    # 3. 增强绿色通道
    g_channel = img_rgb[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g_enhanced = clahe.apply(g_channel)
    
    # 4. 阈值分割
    _, binary = cv2.threshold(g_enhanced, 50, 255, cv2.THRESH_BINARY_INV)
    binary = cv2.bitwise_and(binary, binary, mask=fov_mask)
    
    # 5. 形态学清理
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
    
    # 6. V5 几何筛选 (先选出像出血的东西)
    mask_candidates = filter_by_shape_v5(binary)
    
    # 7. 【V6 新增步骤】中心凹剔除
    # 剔除距离中心 80 像素以内的暗斑
    # 720x576 的图，80像素大概覆盖了中心凹区域
    final_mask = remove_fovea(mask_candidates, fov_center, safe_radius=80)
    
    return final_mask.astype(np.float32) / 255.0

# ==========================================
# 2. 批处理主程序
# ==========================================

def process_batch():
    input_dir = "/mnt/mydata/gww/sj/syndiff/SynDiff/raw_images/fundus"
    output_dir = "/mnt/mydata/gww/sj/syndiff/SynDiff/extracted_masks"
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"📂 输入: {input_dir}")
    print(f"📂 输出: {output_dir}")
    
    valid_extensions = ['.jpg', '.png', '.jpeg', '.bmp', '.tif']
    
    # 这里用 range(1, 25) 遍历 1 到 24
    # 如果你没有安装 tqdm，就把下面这行改成: for i in range(1, 25):
    for i in tqdm(range(1, 25), desc="Processing"):
        file_found = False
        img_path = ""
        filename = ""
        
        for ext in valid_extensions:
            temp_name = f"{i}{ext}"
            temp_path = os.path.join(input_dir, temp_name)
            if os.path.exists(temp_path):
                img_path = temp_path
                filename = temp_name
                file_found = True
                break
        
        if not file_found:
            # print(f"Skipping {i}") # 为了界面清爽可以注释掉
            continue
            
        try:
            img_bgr = cv2.imread(img_path)
            if img_bgr is None: continue
                
            # === 生成 Mask ===
            mask = get_hemorrhage_mask(img_bgr)
            
            # === 保存结果 ===
            # 1. 纯 Mask (PNG)
            mask_uint8 = (mask * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(output_dir, f"{i}_mask.png"), mask_uint8)
            
            # 2. 可视化对比 (JPG)
            vis_img = img_bgr.copy()
            # 红色叠加
            vis_img[mask > 0] = [0, 0, 255] 
            vis_concat = np.hstack((img_bgr, vis_img))
            cv2.imwrite(os.path.join(output_dir, f"{i}_vis.jpg"), vis_concat)
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print("\n✅ 处理完成。检查 extracted_masks 文件夹。")

if __name__ == "__main__":
    process_batch()