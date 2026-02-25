import os
import cv2
import numpy as np
import scipy.io
import glob
import random

# ================= 配置区域 (请仔细修改这里) =================

# --- 1. 训练集路径 (你现在的 raw_images 全是训练集) ---
TRAIN_FUNDUS_DIR = './raw_images/fundus'  
TRAIN_FFA_DIR    = './raw_images/ffa'     

# --- 2. 测试集路径 (请在这里填入你“单独放其他地方”的路径) ---
# 例如： 'D:/datasets/test_data/fundus'
TEST_FUNDUS_DIR  = './test_set/fundus'  # <--- 请修改为你的测试集 Fundus 路径
TEST_FFA_DIR     = './test_set/ffa'     # <--- 请修改为你的测试集 FFA 路径

# --- 3. 输出路径 ---
OUTPUT_DIR = './SynDiff_dataset'

# --- 4. 参数设置 ---
IMG_SIZE = 256         # SynDiff 要求的输入尺寸
PATCHES_PER_IMG = 40   # 训练集每张大图切多少个小块
TEST_PATCHES = 10      # 测试集每张图切多少个块 (或者你可以设多一点)
# ==========================================================

def normalize(img):
    """将像素值归一化到 0-1 之间"""
    return img.astype(np.float32) / 255.0

def process_data(fundus_files, ffa_files, mode='train', num_patches=10):
    if len(fundus_files) == 0:
        print(f"警告：{mode} 文件列表为空，跳过处理。")
        return None, None

    print(f"正在处理 {mode} 数据集，共 {len(fundus_files)} 对原始图片...")
    
    data_fundus_list = []
    data_ffa_list = []

    for idx, (f_path, a_path) in enumerate(zip(fundus_files, ffa_files)):
        # 1. 读取图片
        # Fundus 读取为彩色 (RGB)
        img_fundus = cv2.imread(f_path)
        if img_fundus is None:
            print(f"无法读取图片: {f_path}")
            continue
        img_fundus = cv2.cvtColor(img_fundus, cv2.COLOR_BGR2RGB) # 转为 RGB
        
        # FFA 读取为灰度，然后转为伪彩色 (3通道)
        img_ffa = cv2.imread(a_path, cv2.IMREAD_GRAYSCALE)
        if img_ffa is None:
            print(f"无法读取图片: {a_path}")
            continue
        img_ffa = cv2.cvtColor(img_ffa, cv2.COLOR_GRAY2RGB) # 复制通道变成3通道

        h, w, _ = img_fundus.shape

        # 2. 随机切块 (Random Crop)
        for _ in range(num_patches):
            # 随机选择切块的左上角坐标
            # 确保切块不会超出边界
            if h <= IMG_SIZE or w <= IMG_SIZE:
                 # 如果原图比 256 还小，直接 resize (容错处理)
                 patch_fundus = cv2.resize(img_fundus, (IMG_SIZE, IMG_SIZE))
                 patch_ffa = cv2.resize(img_ffa, (IMG_SIZE, IMG_SIZE))
            else:
                top = np.random.randint(0, h - IMG_SIZE)
                left = np.random.randint(0, w - IMG_SIZE)

                # 裁剪 Fundus
                patch_fundus = img_fundus[top:top+IMG_SIZE, left:left+IMG_SIZE, :]
                # 裁剪 FFA (使用相同的坐标，保持解剖结构对应)
                patch_ffa = img_ffa[top:top+IMG_SIZE, left:left+IMG_SIZE, :]

            # 3. 数据增强 (随机翻转) - 仅对训练集做
            if mode == 'train':
                # 水平翻转
                if random.random() > 0.5:
                    patch_fundus = cv2.flip(patch_fundus, 1)
                    patch_ffa = cv2.flip(patch_ffa, 1)
                # 垂直翻转
                if random.random() > 0.5:
                    patch_fundus = cv2.flip(patch_fundus, 0)
                    patch_ffa = cv2.flip(patch_ffa, 0)

            # 4. 归一化并添加到列表
            data_fundus_list.append(normalize(patch_fundus))
            data_ffa_list.append(normalize(patch_ffa))

        if (idx + 1) % 10 == 0:
            print(f"  已处理 {idx + 1} / {len(fundus_files)} 张原始图片")

    # 5. 转换为 numpy 数组
    # 统一转置为 Channel-First: (N, 3, 256, 256)
    arr_fundus = np.array(data_fundus_list).transpose(0, 3, 1, 2)
    arr_ffa = np.array(data_ffa_list).transpose(0, 3, 1, 2)

    print(f"  {mode} 数据集构建完成。Fundus 形状: {arr_fundus.shape}, FFA 形状: {arr_ffa.shape}")
    return arr_fundus, arr_ffa

def get_file_pairs(fundus_dir, ffa_dir):
    """辅助函数：获取文件列表并排序检查"""
    if not os.path.exists(fundus_dir) or not os.path.exists(ffa_dir):
        print(f"路径不存在: {fundus_dir} 或 {ffa_dir}")
        return [], []

    files_fundus = sorted(glob.glob(os.path.join(fundus_dir, '*')))
    files_ffa = sorted(glob.glob(os.path.join(ffa_dir, '*')))
    
    # 简单的数量检查
    if len(files_fundus) != len(files_ffa):
        print(f"警告：文件夹 {fundus_dir} 和 {ffa_dir} 图片数量不一致！")
        print(f"Fundus: {len(files_fundus)}, FFA: {len(files_ffa)}")
        # 取最小值，避免报错，或者直接退出
        min_len = min(len(files_fundus), len(files_ffa))
        files_fundus = files_fundus[:min_len]
        files_ffa = files_ffa[:min_len]
    
    return files_fundus, files_ffa

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # ================= 1. 获取训练集文件列表 =================
    print("\n--- 读取训练集文件 ---")
    train_fundus, train_ffa = get_file_pairs(TRAIN_FUNDUS_DIR, TRAIN_FFA_DIR)
    
    # ================= 2. 获取测试集文件列表 =================
    print("\n--- 读取测试集文件 ---")
    test_fundus, test_ffa = get_file_pairs(TEST_FUNDUS_DIR, TEST_FFA_DIR)

    # ================= 3. 处理并保存训练集 =================
    if len(train_fundus) > 0:
        # 注意：这里我们将 Fundus 设为 contrast1, FFA 设为 contrast2
        train_c1, train_c2 = process_data(train_fundus, train_ffa, mode='train', num_patches=PATCHES_PER_IMG)
        
        print("正在保存训练集 .mat 文件...")
        scipy.io.savemat(os.path.join(OUTPUT_DIR, 'data_train_contrast1.mat'), {'data': train_c1})
        scipy.io.savemat(os.path.join(OUTPUT_DIR, 'data_train_contrast2.mat'), {'data': train_c2})
    else:
        print("未找到训练集文件，跳过训练集生成。")

    # ================= 4. 处理并保存测试集 =================
    if len(test_fundus) > 0:
        test_c1, test_c2 = process_data(test_fundus, test_ffa, mode='test', num_patches=TEST_PATCHES)

        print("正在保存测试/验证集 .mat 文件...")
        scipy.io.savemat(os.path.join(OUTPUT_DIR, 'data_test_contrast1.mat'), {'data': test_c1})
        scipy.io.savemat(os.path.join(OUTPUT_DIR, 'data_test_contrast2.mat'), {'data': test_c2})
        
        # Val 集合直接复制 Test 的 (SynDiff 需要 val 集)
        scipy.io.savemat(os.path.join(OUTPUT_DIR, 'data_val_contrast1.mat'), {'data': test_c1})
        scipy.io.savemat(os.path.join(OUTPUT_DIR, 'data_val_contrast2.mat'), {'data': test_c2})
    else:
        print("未找到测试集文件，跳过测试集生成。")

    print(f"\n全部处理完毕！数据已保存至: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()