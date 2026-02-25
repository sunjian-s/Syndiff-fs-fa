import cv2
import numpy as np
import os
import csv

def process_image(image_path, output_folder, file_name_no_ext):
    # 1. 读取图片 (BGR)
    img = cv2.imread(image_path)
    if img is None:
        print(f"错误: 无法读取图片 {image_path}")
        return 0

    # 2. 预处理：提取绿色通道并增强对比度
    # split 分离通道，索引 1 是绿色 (0=B, 1=G, 2=R)
    b, g, r = cv2.split(img)

    # 增强对比度 (对应原代码 eye_green * 2.5)
    enhanced_g = cv2.convertScaleAbs(g, alpha=2.5, beta=0)

    # 高斯模糊平滑 (对应原代码 smooth)
    blurred_g = cv2.GaussianBlur(enhanced_g, (5, 5), 0)

    # 3. Canny 边缘检测
    edges = cv2.Canny(blurred_g, 35, 70)

    # 4. 形态学闭运算
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # 对应原代码：dilate(2) -> erode(1)
    processed = cv2.dilate(edges, kernel, iterations=2)
    processed = cv2.erode(processed, kernel, iterations=1)

    # 对应原代码：dilate(4) -> erode(3)
    processed = cv2.dilate(processed, kernel, iterations=4)
    processed = cv2.erode(processed, kernel, iterations=3)

    # 5. 定义移除轮廓的辅助函数
    def remove_contours(binary_img, contours_to_remove):
        mask = np.zeros_like(binary_img)
        cv2.drawContours(mask, contours_to_remove, -1, 255, thickness=cv2.FILLED)
        return cv2.bitwise_and(binary_img, cv2.bitwise_not(mask))

    # 6. 第一轮筛选：去除大斑块
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    big_blobs = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 500: 
            big_blobs.append(cnt)
    
    if big_blobs:
        processed = remove_contours(processed, big_blobs)

    processed = cv2.erode(processed, kernel, iterations=1)

    # 7. 第二轮筛选：去除极小噪点
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    small_blobs = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 0 < area <= 10:
            small_blobs.append(cnt)
    
    if small_blobs:
        processed = remove_contours(processed, small_blobs)

    # 8. 第三轮筛选：几何形状分析
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bad_shapes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        width_height_diff = abs(h - w)
        is_elongated = (width_height_diff > 0.2 * h) or (width_height_diff > 0.2 * w)
        is_sparse = (area < 0.45 * w * h)
        is_too_big = (area > 1500)

        if is_elongated or is_sparse or is_too_big:
            bad_shapes.append(cnt)

    if bad_shapes:
        processed = remove_contours(processed, bad_shapes)

    # 9. 生成结果与保存
    white_pixels = cv2.countNonZero(processed)
    
    # 生成红色叠加图
    ma_layer = np.zeros_like(img)
    ma_layer[:, :, 2] = processed 
    overlay = cv2.addWeighted(img, 1.0, ma_layer, 1.0, 0)

# 修改后的代码 (例如改成 png)
    cv2.imwrite(os.path.join(output_folder, file_name_no_ext + '_MA.png'), processed)
    cv2.imwrite(os.path.join(output_folder, file_name_no_ext + '_MAoverlay.png'), overlay)

    return white_pixels

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(current_dir, "input")
    output_folder = os.path.join(current_dir, "output")
    csv_filename = "ma.csv"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if not os.path.exists(input_folder):
        print(f"错误: 找不到 input 文件夹。请在 {current_dir} 下创建 'input' 文件夹并放入图片。")
        exit()

    files_array = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
    # 增加扩展名支持
    valid_extensions = ['.jpg', '.jpeg', '.tif', '.tiff', '.png', '.bmp']
    files_array = [f for f in files_array if os.path.splitext(f)[1].lower() in valid_extensions]

    print(f"找到 {len(files_array)} 张图片，开始处理...")

    with open(csv_filename, 'w', newline='') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow(['filename', 'pixel_count'])

        for file_name in files_array:
            print(f"正在处理: {file_name}")
            full_path = os.path.join(input_folder, file_name)
            file_name_no_ext = os.path.splitext(file_name)[0]
            
            pixel_count = process_image(full_path, output_folder, file_name_no_ext)
            filewriter.writerow([file_name_no_ext + "_microaneurysm.jpg", pixel_count])
            print(f"  -> 完成。检测到 {pixel_count} 个像素。")

    print("\n所有处理已完成！")