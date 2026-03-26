import torch.utils.data
import numpy as np
import scipy.io

def CreateDatasetSynthesis(phase, input_path, contrast1='Fundus', contrast2='FFA'):
    # 注意：确保主程序 args.contrast1 和 args.contrast2 与实际生成的文件名后缀完全一致
    target_file_s1 = f"{input_path}/data_{phase}_{contrast1}.mat"
    data_fs_s1 = LoadDataSet(target_file_s1, variable='data')
    
    target_file_s2 = f"{input_path}/data_{phase}_{contrast2}.mat"
    data_fs_s2 = LoadDataSet(target_file_s2, variable='data')

    dataset = torch.utils.data.TensorDataset(torch.from_numpy(data_fs_s1), torch.from_numpy(data_fs_s2))  
    return dataset 

def LoadDataSet(load_dir, variable='data', padding=False):
    """
    自适应防御型数据加载器
    """
    print(f"正在加载数据: {load_dir} ...")
    
    try:
        f = scipy.io.loadmat(load_dir)
        data = f[variable]
    except Exception as e:
        print(f"Scipy 读取失败，尝试 h5py: {e}")
        import h5py
        f = h5py.File(load_dir, 'r')
        data = np.array(f[variable])

    data = data.astype(np.float32)
    
    # 维度对齐检查 (确保 N, C, H, W)
    if data.ndim == 3: 
        data = np.expand_dims(data, axis=1) 
        
    # =======================================================
    # 防御性判定：检测数据分布并自适应归一化至 [-1, 1]
    # =======================================================
    d_min, d_max = data.min(), data.max()
    
    if d_min >= 0.0 and d_max <= 1.0:
        # 数据在 [0, 1] 空间，执行映射
        print(f"检测到数据分布在 [{d_min:.2f}, {d_max:.2f}]，执行 [0,1] 到 [-1,1] 映射。")
        data = (data - 0.5) / 0.5
    elif d_min >= -1.0 and d_max <= 1.0:
        # 数据已经在 [-1, 1] 空间，跳过处理
        print(f"检测到数据分布在 [{d_min:.2f}, {d_max:.2f}]，已符合 [-1, 1] 要求，跳过映射。")
    else:
        # 数据异常，给出强警告
        print(f"警告：数据分布异常 [{d_min:.2f}, {d_max:.2f}]，非预期的 [0,1] 或 [-1,1] 空间！")
        # 强制截断防暴雷
        data = np.clip(data, -1.0, 1.0) 

    # =======================================================
    # 修复的 Padding 算法 (非对称 Padding 防护)
    # =======================================================
    if padding and (data.shape[2] < 256 or data.shape[3] < 256):
        h_diff = 256 - data.shape[2]
        w_diff = 256 - data.shape[3]
        
        # 向上取整和向下取整组合，完美处理奇数差异
        pad_top = h_diff // 2
        pad_bottom = h_diff - pad_top
        pad_left = w_diff // 2
        pad_right = w_diff - pad_left
        
        print(f'Padding applied: H({pad_top},{pad_bottom}), W({pad_left},{pad_right})')
        data = np.pad(data, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), 'constant')   
        
    return data