import os
import cv2
import glob

# ================= 配置区域 (每次运行前修改这里) =================

# 🔴 任务：请将你需要合并的文件夹路径填入下面的列表
# 注意：列表里的顺序非常重要！Fundus 和 FFA 的合并顺序必须完全一致！

# --- 第 1 步：先运行 Fundus (眼底图) ---
# SOURCE_FOLDERS = [
#     '/mnt/mydata/gww/sj/syndiff/SynDiff/21/CFP/0_Normal',
#     '/mnt/mydata/gww/sj/syndiff/SynDiff/21/CFP/1_DR',
#     '/mnt/mydata/gww/sj/syndiff/SynDiff/raw_images/fundus'
# ]
# OUTPUT_DIR = '/mnt/mydata/gww/sj/syndiff/SynDiff/Merged_Fundus' # 输出路径

# --- 第 2 步：运行完上面后，注释掉上面，把下面这几行取消注释，再运行一次 FFA ---
SOURCE_FOLDERS = [
    '/mnt/mydata/gww/sj/syndiff/SynDiff/21/FFA/0_Normal',
    '/mnt/mydata/gww/sj/syndiff/SynDiff/21/FFA/1_DR',
    '/mnt/mydata/gww/sj/syndiff/SynDiff/raw_images/ffa'
]
OUTPUT_DIR = '/mnt/mydata/gww/sj/syndiff/SynDiff/Merged_FFA' # 输出路径

# -----------------------------------------------------------
NEW_NAME_PREFIX = 'image'  # 结果: image_0001.png
START_INDEX = 1            # 从 1 开始编号
# ===========================================================

def get_image_files(folder_path):
    """获取文件夹内所有图片文件，并排序"""
    if not os.path.exists(folder_path):
        print(f"❌ 错误：路径不存在 -> {folder_path}")
        return []
        
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.JPG', '*.PNG']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(folder_path, ext)))
    
    # 必须排序，保证同文件夹内顺序固定
    return sorted(list(set(files)))

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"📁 创建输出目录: {OUTPUT_DIR}")

    print("🔍 正在扫描所有源文件夹...")
    
    all_files_ordered = []
    
    # 按顺序遍历每一个文件夹
    for i, folder in enumerate(SOURCE_FOLDERS):
        print(f"   正在扫描文件夹 [{i+1}]: {folder}")
        files = get_image_files(folder)
        count = len(files)
        print(f"     -> 找到 {count} 张图片")
        
        # 将这个文件夹的文件追加到总列表中
        all_files_ordered.extend(files)

    total_files = len(all_files_ordered)
    if total_files == 0:
        print("❌ 未找到任何图片，请检查路径是否正确！")
        return

    print(f"\n🚀 开始合并与转换，共 {total_files} 张图片，目标格式: PNG")
    print(f"📂 结果将保存在: {OUTPUT_DIR}")

    current_idx = START_INDEX
    
    for original_path in all_files_ordered:
        # 读取
        img = cv2.imread(original_path)
        
        if img is None:
            print(f"⚠️ 无法读取 (跳过): {original_path}")
            continue
        
        # 构建新文件名
        new_filename = f"{NEW_NAME_PREFIX}_{current_idx:04d}.png"
        save_path = os.path.join(OUTPUT_DIR, new_filename)
        
        # 保存为 PNG
        cv2.imwrite(save_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        
        if current_idx % 100 == 0:
            print(f"   进度 {current_idx}/{total_files}: {os.path.basename(original_path)} -> {new_filename}")
            
        current_idx += 1

    print("\n✅ 处理完成！")

if __name__ == '__main__':
    main()