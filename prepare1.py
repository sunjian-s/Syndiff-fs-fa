import os
import cv2
import numpy as np
import scipy.io
import glob
import random

# ================= 配置区域 (请修改这里) =================

# 1. 输入路径 (指向你刚才“合并成功”的文件夹)
# 请确保路径完全正确！
TRAIN_FUNDUS_DIR = '/mnt/mydata/gww/sj/syndiff/SynDiff/Merged_Fundus'   
TRAIN_FFA_DIR    = '/mnt/mydata/gww/sj/syndiff/SynDiff/Merged_FFA'     

# 2. 测试集路径 (如果你有单独的测试集，请填这里；如果没有，可以暂时指向同一个或者空的)
TEST_FUNDUS_DIR  = './test_set/fundus'  
TEST_FFA_DIR     = './test_set/ffa'     

# 3. 输出路径 (生成的 .mat 文件存放处)
OUTPUT_DIR = './SynDiff_Final_Dataset'

# 4. 参数设置
IMG_SIZE = 256         # 模型输入尺寸
PATCHES_PER_IMG = 18   # 每张图切多少块 (数据量大的话可以适当减少，比如 20-30)
TEST_PATCHES = 10      

# 5. 分辨率对齐参数
TARGET_HEIGHT = 576    # 旧数据的标准高度
RESIZE_THRESHOLD = 800 # 超过这个高度会被判定为“新数据”并进行缩放
# ==========================================================

def normalize(img):
    return img.astype(np.float32) / 255.0

def process_data(fundus_files, ffa_files, mode='train', num_patches=10):
    if len(fundus_files) == 0:
        print(f"警告：{mode} 文件列表为空。")
        return None, None

    print(f"🚀 开始处理 {mode} 数据集，共 {len(fundus_files)} 对图片...")
    
    data_fundus_list = []
    data_ffa_list = []

    for idx, (f_path, a_path) in enumerate(zip(fundus_files, ffa_files)):
        # --- 读取图片 ---
        img_fundus = cv2.imread(f_path)
        img_ffa = cv2.imread(a_path, cv2.IMREAD_GRAYSCALE) # FFA读灰度

        if img_fundus is None or img_ffa is None:
            print(f"无法读取图片 (跳过): {os.path.basename(f_path)}")
            continue

        # 转格式
        img_fundus = cv2.cvtColor(img_fundus, cv2.COLOR_BGR2RGB)
        img_ffa = cv2.cvtColor(img_ffa, cv2.COLOR_GRAY2RGB) # 转伪彩色3通道

        # --- 【核心】分辨率自动对齐 ---
        h_raw, w_raw, _ = img_fundus.shape
        
        # 如果是大图 (1300x991)，缩放到 576 高度
        if h_raw > RESIZE_THRESHOLD:
            scale_ratio = TARGET_HEIGHT / h_raw
            new_w = int(w_raw * scale_ratio)
            new_h = TARGET_HEIGHT 
            
            # 使用区域插值 (INTER_AREA) 保留微小病灶
            img_fundus = cv2.resize(img_fundus, (new_w, new_h), interpolation=cv2.INTER_AREA)
            img_ffa = cv2.resize(img_ffa, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # 更新尺寸
        h, w, _ = img_fundus.shape

        # --- 切块逻辑 ---
        for _ in range(num_patches):
            if h <= IMG_SIZE or w <= IMG_SIZE:
                 # 容错：图太小直接Resize
                 patch_fundus = cv2.resize(img_fundus, (IMG_SIZE, IMG_SIZE))
                 patch_ffa = cv2.resize(img_ffa, (IMG_SIZE, IMG_SIZE))
            else:
                top = np.random.randint(0, h - IMG_SIZE)
                left = np.random.randint(0, w - IMG_SIZE)

                patch_fundus = img_fundus[top:top+IMG_SIZE, left:left+IMG_SIZE, :]
                patch_ffa = img_ffa[top:top+IMG_SIZE, left:left+IMG_SIZE, :]

            # --- 数据增强 (仅训练集) ---
            if mode == 'train':
                # 翻转
                if random.random() > 0.5:
                    patch_fundus = cv2.flip(patch_fundus, 1)
                    patch_ffa = cv2.flip(patch_ffa, 1)
                if random.random() > 0.5:
                    patch_fundus = cv2.flip(patch_fundus, 0)
                    patch_ffa = cv2.flip(patch_ffa, 0)

            data_fundus_list.append(normalize(patch_fundus))
            data_ffa_list.append(normalize(patch_ffa))

        if (idx + 1) % 50 == 0:
            print(f"   已处理 {idx + 1} / {len(fundus_files)}")

    # 转为 PyTorch 格式 (N, 3, H, W)
    arr_fundus = np.array(data_fundus_list).transpose(0, 3, 1, 2)
    arr_ffa = np.array(data_ffa_list).transpose(0, 3, 1, 2)

    print(f"✅ {mode} 构建完成。最终形状: {arr_fundus.shape}")
    return arr_fundus, arr_ffa

def get_file_pairs(fundus_dir, ffa_dir):
    if not os.path.exists(fundus_dir) or not os.path.exists(ffa_dir):
        print(f"❌ 路径不存在: {fundus_dir}")
        return [], []

    # 支持 png (你刚转好的) 和 jpg
    exts = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG']
    files_fundus = sorted([f for ext in exts for f in glob.glob(os.path.join(fundus_dir, ext))])
    files_ffa = sorted([f for ext in exts for f in glob.glob(os.path.join(ffa_dir, ext))])
    
    # 再次确保数量一致
    min_len = min(len(files_fundus), len(files_ffa))
    return files_fundus[:min_len], files_ffa[:min_len]

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. 处理训练集
    train_f, train_a = get_file_pairs(TRAIN_FUNDUS_DIR, TRAIN_FFA_DIR)
    if len(train_f) > 0:
        # 注意：这里调用了 process_data
        train_c1, train_c2 = process_data(train_f, train_a, mode='train', num_patches=PATCHES_PER_IMG)
        
        print("💾 正在保存训练集 .mat ...")
        # 大于 2GB 的话 scipy 可能会慢，耐心等待
        scipy.io.savemat(os.path.join(OUTPUT_DIR, 'data_train_contrast1.mat'), {'data': train_c1})
        scipy.io.savemat(os.path.join(OUTPUT_DIR, 'data_train_contrast2.mat'), {'data': train_c2})
    
    # 2. 处理测试集 (如果有)
    test_f, test_a = get_file_pairs(TEST_FUNDUS_DIR, TEST_FFA_DIR)
    if len(test_f) > 0:
        test_c1, test_c2 = process_data(test_f, test_a, mode='test', num_patches=TEST_PATCHES)
        print("💾 正在保存测试集 .mat ...")
        scipy.io.savemat(os.path.join(OUTPUT_DIR, 'data_test_contrast1.mat'), {'data': test_c1})
        scipy.io.savemat(os.path.join(OUTPUT_DIR, 'data_test_contrast2.mat'), {'data': test_c2})
        # Val集复制Test
        scipy.io.savemat(os.path.join(OUTPUT_DIR, 'data_val_contrast1.mat'), {'data': test_c1})
        scipy.io.savemat(os.path.join(OUTPUT_DIR, 'data_val_contrast2.mat'), {'data': test_c2})

    print(f"\n🎉 恭喜！数据准备完毕。请使用目录下的数据开始训练: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()