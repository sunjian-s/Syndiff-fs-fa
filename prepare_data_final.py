import os
import cv2
import numpy as np
import scipy.io
import glob
import random

# ================= 配置区域 (请仔细核对路径) =================

# --- 1. 训练集路径 ---
TRAIN_FUNDUS_DIR = './raw_images/fundus'  
TRAIN_FFA_DIR    = './raw_images/ffa'     
TRAIN_MASK_DIR   = './raw_images/mask'    # 【新增】训练集 Mask 路径

# --- 2. 测试集路径 ---
TEST_FUNDUS_DIR  = './test_set/fundus'    
TEST_FFA_DIR     = './test_set/ffa'       
TEST_MASK_DIR    = './test_set/mask'      # 【新增】测试集 Mask 路径

# --- 3. 输出路径 ---
OUTPUT_DIR = './SynDiff_dataset_with_mask' # 建议换个新名字，以免和旧的混淆

# --- 4. 参数设置 ---
IMG_SIZE = 256         
PATCHES_PER_IMG = 40   # 训练集扩充倍数
TEST_PATCHES = 10      # 测试集切块数
# ==========================================================

def normalize(img):
    """将像素值归一化到 0-1 之间"""
    return img.astype(np.float32) / 255.0

def get_file_triplets(fundus_dir, ffa_dir, mask_dir):
    """辅助函数：获取 Fundus/FFA/Mask 三个文件夹对应的文件列表"""
    if not os.path.exists(fundus_dir) or not os.path.exists(ffa_dir) or not os.path.exists(mask_dir):
        print(f"❌ 错误：路径不存在!\n  {fundus_dir}\n  {ffa_dir}\n  {mask_dir}")
        return [], [], []

    # 获取所有图片文件
    exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif')
    files_fundus = sorted([f for ext in exts for f in glob.glob(os.path.join(fundus_dir, ext))])
    files_ffa = sorted([f for ext in exts for f in glob.glob(os.path.join(ffa_dir, ext))])
    files_mask = sorted([f for ext in exts for f in glob.glob(os.path.join(mask_dir, ext))])
    
    # 数量检查
    min_len = min(len(files_fundus), len(files_ffa), len(files_mask))
    if len(files_fundus) != len(files_ffa) or len(files_fundus) != len(files_mask):
        print(f"⚠️ 警告：文件数量不一致！将截断到最小数量: {min_len}")
        print(f"  Fundus: {len(files_fundus)}, FFA: {len(files_ffa)}, Mask: {len(files_mask)}")
    
    return files_fundus[:min_len], files_ffa[:min_len], files_mask[:min_len]

def process_data(fundus_files, ffa_files, mask_files, mode='train', num_patches=10):
    if len(fundus_files) == 0:
        print(f"⚠️ 警告：{mode} 文件列表为空，跳过。")
        return None, None

    print(f"🔄 正在处理 {mode} 数据集，共 {len(fundus_files)} 组图片...")
    
    data_fundus_list = []
    data_ffa_list = []

    for idx, (f_path, a_path, m_path) in enumerate(zip(fundus_files, ffa_files, mask_files)):
        # 1. 读取 Fundus (RGB)
        img_fundus = cv2.imread(f_path)
        if img_fundus is None: continue
        img_fundus = cv2.cvtColor(img_fundus, cv2.COLOR_BGR2RGB)
        
        # 2. 读取 FFA (转 RGB)
        img_ffa = cv2.imread(a_path, cv2.IMREAD_GRAYSCALE)
        if img_ffa is None: continue
        img_ffa = cv2.cvtColor(img_ffa, cv2.COLOR_GRAY2RGB) 

        # 3. 读取 Mask (单通道)
        img_mask = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
        if img_mask is None:
            # 如果没有 Mask，生成一个全黑的占位 Mask (防止报错)
            print(f"  [警告] 找不到 Mask: {m_path}，使用全黑代替")
            img_mask = np.zeros((img_fundus.shape[0], img_fundus.shape[1]), dtype=np.uint8)
        
        # 扩展 Mask 维度 (H, W) -> (H, W, 1)
        if img_mask.ndim == 2:
            img_mask = np.expand_dims(img_mask, axis=-1)

        h, w, _ = img_fundus.shape

        # 4. 随机切块
        for _ in range(num_patches):
            # 容错：如果原图太小，直接 resize
            if h <= IMG_SIZE or w <= IMG_SIZE:
                 patch_fundus = cv2.resize(img_fundus, (IMG_SIZE, IMG_SIZE))
                 patch_ffa = cv2.resize(img_ffa, (IMG_SIZE, IMG_SIZE))
                 patch_mask = cv2.resize(img_mask, (IMG_SIZE, IMG_SIZE))
                 # Resize 后 Mask 可能会丢维度，补回来
                 if patch_mask.ndim == 2: patch_mask = np.expand_dims(patch_mask, axis=-1)
            else:
                top = np.random.randint(0, h - IMG_SIZE)
                left = np.random.randint(0, w - IMG_SIZE)

                patch_fundus = img_fundus[top:top+IMG_SIZE, left:left+IMG_SIZE, :]
                patch_ffa = img_ffa[top:top+IMG_SIZE, left:left+IMG_SIZE, :]
                patch_mask = img_mask[top:top+IMG_SIZE, left:left+IMG_SIZE, :]

            # --- 智能筛选策略 (Hard Example Mining) ---
            # 如果是训练集，且 Mask 里有病灶 (亮度>128)，则保留
            # 如果全是背景，有一定概率丢弃，让模型多看病灶
            has_lesion = np.max(patch_mask) > 100
            if mode == 'train' and not has_lesion and random.random() > 0.6:
                continue 

            # 5. 数据增强 (仅训练集)
            if mode == 'train':
                # 水平翻转
                if random.random() > 0.5:
                    patch_fundus = cv2.flip(patch_fundus, 1)
                    patch_ffa = cv2.flip(patch_ffa, 1)
                    patch_mask = cv2.flip(patch_mask, 1)
                    if patch_mask.ndim == 2: patch_mask = np.expand_dims(patch_mask, axis=-1)
                # 垂直翻转
                if random.random() > 0.5:
                    patch_fundus = cv2.flip(patch_fundus, 0)
                    patch_ffa = cv2.flip(patch_ffa, 0)
                    patch_mask = cv2.flip(patch_mask, 0)
                    if patch_mask.ndim == 2: patch_mask = np.expand_dims(patch_mask, axis=-1)

            # 6. 【核心】拼接通道 (3 RGB + 1 Mask = 4 Channels)
            combined_fundus = np.concatenate((patch_fundus, patch_mask), axis=2)
            combined_ffa = np.concatenate((patch_ffa, patch_mask), axis=2)

            data_fundus_list.append(normalize(combined_fundus))
            data_ffa_list.append(normalize(combined_ffa))

        if (idx + 1) % 10 == 0:
            print(f"  已处理 {idx + 1}/{len(fundus_files)}")

    # 7. 转为 PyTorch 格式 (N, 4, 256, 256)
    arr_fundus = np.array(data_fundus_list).transpose(0, 3, 1, 2)
    arr_ffa = np.array(data_ffa_list).transpose(0, 3, 1, 2)

    print(f"  ✅ {mode} 完成。形状: {arr_fundus.shape}")
    return arr_fundus, arr_ffa

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # ================= 1. 准备训练集 =================
    print("\n[1/2] 准备训练集...")
    train_f, train_a, train_m = get_file_triplets(TRAIN_FUNDUS_DIR, TRAIN_FFA_DIR, TRAIN_MASK_DIR)
    
    if len(train_f) > 0:
        train_c1, train_c2 = process_data(train_f, train_a, train_m, mode='train', num_patches=PATCHES_PER_IMG)
        print("  保存训练集 .mat ...")
        scipy.io.savemat(os.path.join(OUTPUT_DIR, 'data_train_contrast1.mat'), {'data': train_c1})
        scipy.io.savemat(os.path.join(OUTPUT_DIR, 'data_train_contrast2.mat'), {'data': train_c2})
    else:
        print("❌ 错误：训练集文件为空，请检查路径！")

    # ================= 2. 准备测试集 =================
    print("\n[2/2] 准备测试集...")
    test_f, test_a, test_m = get_file_triplets(TEST_FUNDUS_DIR, TEST_FFA_DIR, TEST_MASK_DIR)

    if len(test_f) > 0:
        test_c1, test_c2 = process_data(test_f, test_a, test_m, mode='test', num_patches=TEST_PATCHES)
        print("  保存测试集 .mat ...")
        scipy.io.savemat(os.path.join(OUTPUT_DIR, 'data_test_contrast1.mat'), {'data': test_c1})
        scipy.io.savemat(os.path.join(OUTPUT_DIR, 'data_test_contrast2.mat'), {'data': test_c2})
        
        # 同时也作为 Val 集 (训练时用来监控)
        scipy.io.savemat(os.path.join(OUTPUT_DIR, 'data_val_contrast1.mat'), {'data': test_c1})
        scipy.io.savemat(os.path.join(OUTPUT_DIR, 'data_val_contrast2.mat'), {'data': test_c2})
    else:
        print("⚠️ 警告：测试集文件为空。")

    print(f"\n🎉 全部处理完毕！数据已保存至: {OUTPUT_DIR}")
    print("请使用 --num_channels 4 启动训练。")

if __name__ == '__main__':
    main()