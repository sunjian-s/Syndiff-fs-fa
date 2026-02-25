import os
import cv2
import numpy as np
import scipy.io
import glob
import random
import h5py

# ================= 配置区域 (请修改这里) =================

# 1. 测试集路径 (Input)
# 请确保这两个文件夹里分别放着对应的 Fundus 和 FFA 图片
# 这里的 T1 代表 Fundus (输入), T2 代表 FFA (目标)
TEST_FUNDUS_DIR = '/mnt/mydata/gww/sj/syndiff/SynDiff/test_set/fundus' 
TEST_FFA_DIR    = '/mnt/mydata/gww/sj/syndiff/SynDiff/test_set/fundus'

# 2. 输出路径 (Output)
# 生成的 .mat 文件将保存在这里 (通常和 test_set 目录一致)
OUTPUT_DIR = '/mnt/mydata/gww/sj/syndiff/SynDiff/test_set/fundus'

# 3. 图片参数
IMG_SIZE = 256  # 模型要求的输入大小
# =======================================================

def normalize(img):
    """归一化到 0-1"""
    return img.astype(np.float32) / 255.0

def save_mat(filepath, data_dict):
    """兼容大文件的保存函数"""
    try:
        with h5py.File(filepath, 'w') as f:
            for k, v in data_dict.items():
                f.create_dataset(k, data=v)
        print(f"✅ 保存成功 (h5py): {filepath}")
    except Exception as e:
        print(f"⚠️ h5py 保存失败，尝试 scipy: {e}")
        scipy.io.savemat(filepath, data_dict)
        print(f"✅ 保存成功 (scipy): {filepath}")

def process_data(fundus_files, ffa_files, mode='test'):
    if len(fundus_files) == 0:
        print(f"❌ 警告：{mode} 文件列表为空！")
        return None, None

    print(f"🔄 正在处理 {mode} 数据集，共 {len(fundus_files)} 对图片...")
    
    data_fundus_list = []
    data_ffa_list = []

    for idx, (f_path, a_path) in enumerate(zip(fundus_files, ffa_files)):
        # --- 1. 读取图片 ---
        img_fundus = cv2.imread(f_path)
        if img_fundus is None:
            print(f"❌ 无法读取: {f_path}")
            continue
        img_fundus = cv2.cvtColor(img_fundus, cv2.COLOR_BGR2RGB) 
        
        img_ffa = cv2.imread(a_path, cv2.IMREAD_GRAYSCALE)
        if img_ffa is None:
            print(f"❌ 无法读取: {a_path}")
            continue
        img_ffa = cv2.cvtColor(img_ffa, cv2.COLOR_GRAY2RGB) 

        h, w, _ = img_fundus.shape

        # --- 2. 核心逻辑: 中心裁剪 (Center Crop) 256x256 ---
        # 目的：保持原始分辨率，不进行缩放，确保特征尺度与训练时一致
        
        # 计算中心坐标
        center_y = h // 2
        center_x = w // 2
        
        # 计算左上角坐标
        start_x = center_x - (IMG_SIZE // 2)
        start_y = center_y - (IMG_SIZE // 2)
        
        # 边界安全检查 (防止图本身小于 256)
        if start_x < 0 or start_y < 0:
            # 如果原图太小，只能 Resize
            patch_fundus = cv2.resize(img_fundus, (IMG_SIZE, IMG_SIZE))
            patch_ffa    = cv2.resize(img_ffa, (IMG_SIZE, IMG_SIZE))
        else:
            # 执行裁剪
            patch_fundus = img_fundus[start_y:start_y+IMG_SIZE, start_x:start_x+IMG_SIZE]
            patch_ffa    = img_ffa[start_y:start_y+IMG_SIZE, start_x:start_x+IMG_SIZE]

        # --- 3. 加入列表 ---
        data_fundus_list.append(normalize(patch_fundus))
        data_ffa_list.append(normalize(patch_ffa))

        if (idx + 1) % 50 == 0:
            print(f"   已处理 {idx + 1} / {len(fundus_files)}")

    # 转换为 (N, C, H, W)
    arr_fundus = np.array(data_fundus_list).transpose(0, 3, 1, 2)
    arr_ffa = np.array(data_ffa_list).transpose(0, 3, 1, 2)

    print(f"📊 {mode} 最终形状: {arr_fundus.shape}")
    return arr_fundus, arr_ffa

def get_file_pairs(fundus_dir, ffa_dir):
    if not os.path.exists(fundus_dir) or not os.path.exists(ffa_dir):
        print(f"❌ 路径不存在:\n  {fundus_dir}\n  {ffa_dir}")
        return [], []

    exts = ['*.jpg', '*.png', '*.jpeg', '*.bmp']
    files_fundus = []
    files_ffa = []
    
    for ext in exts:
        files_fundus.extend(glob.glob(os.path.join(fundus_dir, ext)))
        files_ffa.extend(glob.glob(os.path.join(ffa_dir, ext)))
    
    files_fundus = sorted(files_fundus)
    files_ffa = sorted(files_ffa)
    
    # 简单的对齐检查
    if len(files_fundus) != len(files_ffa):
        print(f"⚠️ 数量不一致! Fundus:{len(files_fundus)} vs FFA:{len(files_ffa)} (将截断至较短长度)")
        min_len = min(len(files_fundus), len(files_ffa))
        files_fundus = files_fundus[:min_len]
        files_ffa = files_ffa[:min_len]
    
    return files_fundus, files_ffa

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 获取文件列表
    test_fundus, test_ffa = get_file_pairs(TEST_FUNDUS_DIR, TEST_FFA_DIR)

    if len(test_fundus) > 0:
        # 处理数据
        test_c1, test_c2 = process_data(test_fundus, test_ffa, mode='test')

        # 保存 Test 数据
        print("💾 正在保存 Test 数据...")
        save_mat(os.path.join(OUTPUT_DIR, 'data_test_T1.mat'), {'data': test_c1})
        save_mat(os.path.join(OUTPUT_DIR, 'data_test_T2.mat'), {'data': test_c2})
        
        # 保存 Val 数据 (复制一份，防止报错)
        print("💾 正在保存 Val 数据副本...")
        save_mat(os.path.join(OUTPUT_DIR, 'data_val_T1.mat'), {'data': test_c1})
        save_mat(os.path.join(OUTPUT_DIR, 'data_val_T2.mat'), {'data': test_c2})
        
        print(f"\n🎉 全部完成！请检查目录: {OUTPUT_DIR}")
    else:
        print("❌ 未找到图片，请检查路径配置。")

if __name__ == '__main__':
    main()