import os
import cv2
import numpy as np
import scipy.io
import glob
import random

# ================= 配置区域 =================
TRAIN_FUNDUS_DIR = './raw_images/fundus'  
TRAIN_FFA_DIR    = './raw_images/ffa'     
TEST_FUNDUS_DIR  = './test_set/fundus'  
TEST_FFA_DIR     = './test_set/ffa'     
OUTPUT_DIR = './SynDiff_dataset'

IMG_SIZE = 256         
PATCHES_PER_IMG = 40   # 训练集单图切块数
# ============================================

def normalize(img):
    """
    底层对齐：严格映射至 [-1.0, 1.0]，匹配扩散模型噪声分布
    """
    return (img.astype(np.float32) / 127.5) - 1.0

def filter_image_files(file_list):
    """过滤非图像文件"""
    img_suffixes = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    return [f for f in file_list if f.lower().endswith(img_suffixes)]

def process_data(fundus_files, ffa_files, mode='train', num_patches=40):
    if len(fundus_files) == 0:
        print(f"警告：{mode} 文件列表为空。")
        return None, None

    print(f"开始处理 {mode} 数据集，共 {len(fundus_files)} 对原始图片...")
    
    data_fundus_list = []
    data_ffa_list = []

    for idx, (f_path, a_path) in enumerate(zip(fundus_files, ffa_files)):
        # 1. 严格读取图片并转换通道
        img_fundus = cv2.imread(f_path)
        if img_fundus is None: continue
        img_fundus = cv2.cvtColor(img_fundus, cv2.COLOR_BGR2RGB)
        
        img_ffa = cv2.imread(a_path, cv2.IMREAD_GRAYSCALE)
        if img_ffa is None: continue
        img_ffa = cv2.cvtColor(img_ffa, cv2.COLOR_GRAY2RGB)

        h, w, _ = img_fundus.shape

        # 2. 核心切块逻辑分化
        if h <= IMG_SIZE or w <= IMG_SIZE:
            # 极小图防暴雷：直接 Resize
            patch_fundus = cv2.resize(img_fundus, (IMG_SIZE, IMG_SIZE))
            patch_ffa = cv2.resize(img_ffa, (IMG_SIZE, IMG_SIZE))
            data_fundus_list.append(normalize(patch_fundus))
            data_ffa_list.append(normalize(patch_ffa))
        else:
            if mode == 'train':
                # 训练集：高密度随机裁剪 + 几何增强
                for _ in range(num_patches):
                    top = np.random.randint(0, h - IMG_SIZE)
                    left = np.random.randint(0, w - IMG_SIZE)
                    
                    patch_fundus = img_fundus[top:top+IMG_SIZE, left:left+IMG_SIZE, :]
                    patch_ffa = img_ffa[top:top+IMG_SIZE, left:left+IMG_SIZE, :]

                    if random.random() > 0.5:
                        patch_fundus = cv2.flip(patch_fundus, 1)
                        patch_ffa = cv2.flip(patch_ffa, 1)
                    if random.random() > 0.5:
                        patch_fundus = cv2.flip(patch_fundus, 0)
                        patch_ffa = cv2.flip(patch_ffa, 0)

                    data_fundus_list.append(normalize(patch_fundus))
                    data_ffa_list.append(normalize(patch_ffa))
            
            else:
                # 测试集：确定性 5 宫格裁剪 (不引入随机变量，提供绝对对比基准)
                crops = [
                    (0, 0),                                       # 左上
                    (0, w - IMG_SIZE),                            # 右上
                    (h - IMG_SIZE, 0),                            # 左下
                    (h - IMG_SIZE, w - IMG_SIZE),                 # 右下
                    ((h - IMG_SIZE) // 2, (w - IMG_SIZE) // 2)    # 正中心
                ]
                
                for (top, left) in crops:
                    patch_fundus = img_fundus[top:top+IMG_SIZE, left:left+IMG_SIZE, :]
                    patch_ffa = img_ffa[top:top+IMG_SIZE, left:left+IMG_SIZE, :]
                    data_fundus_list.append(normalize(patch_fundus))
                    data_ffa_list.append(normalize(patch_ffa))

        if (idx + 1) % 10 == 0:
            print(f"  已处理 {idx + 1} / {len(fundus_files)} 张原始图片")

    # 3. 转换为 Channel-First 格式送入网络 (N, C, H, W)
    arr_fundus = np.array(data_fundus_list).transpose(0, 3, 1, 2)
    arr_ffa = np.array(data_ffa_list).transpose(0, 3, 1, 2)

    print(f"  {mode} 构建完成。Fundus: {arr_fundus.shape}, FFA: {arr_ffa.shape}")
    return arr_fundus, arr_ffa

def get_file_pairs(fundus_dir, ffa_dir):
    """文件对齐与过滤"""
    if not os.path.exists(fundus_dir) or not os.path.exists(ffa_dir):
        return [], []

    files_fundus = sorted(glob.glob(os.path.join(fundus_dir, '*')))
    files_ffa = sorted(glob.glob(os.path.join(ffa_dir, '*')))
    
    files_fundus = filter_image_files(files_fundus)
    files_ffa = filter_image_files(files_ffa)
    
    if len(files_fundus) != len(files_ffa):
        print("严重警告：源目录文件数量不对齐，强制截断匹配。")
        min_len = min(len(files_fundus), len(files_ffa))
        files_fundus = files_fundus[:min_len]
        files_ffa = files_ffa[:min_len]
    
    return files_fundus, files_ffa

def main():
    # 锁定全局随机种子，确保流形切分可绝对复现
    random.seed(42)
    np.random.seed(42)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print("\n--- 读取并生成训练集 ---")
    train_fundus, train_ffa = get_file_pairs(TRAIN_FUNDUS_DIR, TRAIN_FFA_DIR)
    if train_fundus:
        train_c1, train_c2 = process_data(train_fundus, train_ffa, mode='train', num_patches=PATCHES_PER_IMG)
        scipy.io.savemat(os.path.join(OUTPUT_DIR, 'data_train_contrast1.mat'), {'data': train_c1})
        scipy.io.savemat(os.path.join(OUTPUT_DIR, 'data_train_contrast2.mat'), {'data': train_c2})

    print("\n--- 读取并生成测试/验证集 ---")
    test_fundus, test_ffa = get_file_pairs(TEST_FUNDUS_DIR, TEST_FFA_DIR)
    if test_fundus:
        # 测试集强制执行内部的 5 宫格逻辑，忽略外部 num_patches 传参
        test_c1, test_c2 = process_data(test_fundus, test_ffa, mode='test')
        scipy.io.savemat(os.path.join(OUTPUT_DIR, 'data_test_contrast1.mat'), {'data': test_c1})
        scipy.io.savemat(os.path.join(OUTPUT_DIR, 'data_test_contrast2.mat'), {'data': test_c2})
        scipy.io.savemat(os.path.join(OUTPUT_DIR, 'data_val_contrast1.mat'), {'data': test_c1})
        scipy.io.savemat(os.path.join(OUTPUT_DIR, 'data_val_contrast2.mat'), {'data': test_c2})

    print(f"\n全部计算与 I/O 完毕，数据锚定于: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()