# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# ---------------------------------------------------------------


"""
用于上采样或下采样图像的层。
许多函数是从 https://github.com/NVlabs/stylegan2 移植过来的。
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from utils.op import upfirdn2d # 【核心】导入 StyleGAN2 的高性能 up/down 算子


# Function ported from StyleGAN2
def get_weight(module,
               shape,
               weight_var='weight',
               kernel_init=None):
    """【辅助】获取/创建一个卷积层或全连接层的权重张量。"""

    return module.param(weight_var, kernel_init, shape)


class Conv2d(nn.Module):
    """
    【模块】自定义的2D卷积层 (移植自 StyleGAN2)
    
    这个模块将“卷积”与“上/下采样”融合在了一起。
    """

    def __init__(self, in_ch, out_ch, kernel, up=False, down=False,
                 resample_kernel=(1, 3, 3, 1), # 用于上/下采样的 FIR 滤波器核
                 use_bias=True,
                 kernel_init=None):
        super().__init__()
        assert not (up and down) # 不能同时上采样和下采样
        assert kernel >= 1 and kernel % 2 == 1 # 卷积核大小必须是奇数
        
        # 1. 初始化权重
        self.weight = nn.Parameter(torch.zeros(out_ch, in_ch, kernel, kernel))
        if kernel_init is not None:
            self.weight.data = kernel_init(self.weight.data.shape)
        # 2. 初始化偏置
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(out_ch))

        self.up = up
        self.down = down
        self.resample_kernel = resample_kernel # FIR 滤波器核
        self.kernel = kernel
        self.use_bias = use_bias

    def forward(self, x):
        # 【关键】根据模式选择不同的操作
        if self.up:
            # 执行“融合的上采样+卷积”
            x = upsample_conv_2d(x, self.weight, k=self.resample_kernel)
        elif self.down:
            # 执行“融合的卷积+下采样”
            x = conv_downsample_2d(x, self.weight, k=self.resample_kernel)
        else:
            # 执行标准的 2D 卷积
            x = F.conv2d(x, self.weight, stride=1, padding=self.kernel // 2)

        if self.use_bias:
            x = x + self.bias.reshape(1, -1, 1, 1)

        return x


def naive_upsample_2d(x, factor=2):
    """【辅助】朴素的上采样 (最近邻插值)"""
    _N, C, H, W = x.shape
    x = torch.reshape(x, (-1, C, H, 1, W, 1))
    x = x.repeat(1, 1, 1, factor, 1, factor) # 将 H 和 W 维度重复 'factor' 次
    return torch.reshape(x, (-1, C, H * factor, W * factor))


def naive_downsample_2d(x, factor=2):
    """【辅助】朴素的下采样 (平均池化)"""
    _N, C, H, W = x.shape
    x = torch.reshape(x, (-1, C, H // factor, factor, W // factor, factor))
    return torch.mean(x, dim=(3, 5)) # 在 2x2 块上取平均值


def upsample_conv_2d(x, w, k=None, factor=2, gain=1):
    """
    【核心】融合的 (上采样 + 卷积)。
    
    它首先执行一个“转置卷积”(ConvTranspose2d) 来实现上采样和卷积，
    然后再通过 upfirdn2d (FIR滤波器) 来平滑结果，以消除伪影。
    
    Args:
        x: 输入张量 [N, C, H, W]
        w: 卷积核权重
        k: FIR 滤波器核 (例如 [1, 3, 3, 1])
        factor: 上采样因子 (例如 2)
        gain: 增益
    """

    assert isinstance(factor, int) and factor >= 1

    # ... (检查权重形状)
    assert len(w.shape) == 4
    convH = w.shape[2]
    convW = w.shape[3]
    inC = w.shape[1]
    outC = w.shape[0]
    assert convW == convH

    # 1. 设置 FIR 滤波器核
    if k is None:
        k = [1] * factor
    k = _setup_kernel(k) * (gain * (factor ** 2)) # 归一化并应用增益
    p = (k.shape[0] - factor) - (convW - 1)

    stride = (factor, factor)

    # ... (计算输出形状和 padding)
    stride = [1, 1, factor, factor]
    output_shape = ((_shape(x, 2) - 1) * factor + convH, (_shape(x, 3) - 1) * factor + convW)
    output_padding = (output_shape[0] - (_shape(x, 2) - 1) * stride[0] - convH,
                      output_shape[1] - (_shape(x, 3) - 1) * stride[1] - convW)
    assert output_padding[0] >= 0 and output_padding[1] >= 0
    num_groups = _shape(x, 1) // inC

    # 2. 【关键】执行“转置卷积” (ConvTranspose2d)
    #    这是一种“学习到的上采样”，同时完成了卷积。
    w = torch.reshape(w, (num_groups, -1, inC, convH, convW))
    w = w[..., ::-1, ::-1].permute(0, 2, 1, 3, 4) # 翻转卷积核 (PyTorch ConvTranspose2d 的要求)
    w = torch.reshape(w, (num_groups * inC, -1, convH, convW))

    x = F.conv_transpose2d(x, w, stride=stride, output_padding=output_padding, padding=0)
    
    # 3. 【关键】使用 FIR 滤波器进行平滑 (抗锯齿)
    return upfirdn2d(x, torch.tensor(k, device=x.device),
                       pad=((p + 1) // 2 + factor - 1, p // 2 + 1))


def conv_downsample_2d(x, w, k=None, factor=2, gain=1):
    """
    【核心】融合的 (卷积 + 下采样)。
    
    它首先使用 upfirdn2d (FIR滤波器) 对输入进行平滑 (抗锯齿)，
    然后再执行一个标准卷积 (stride=factor) 来实现下采样。
    
    Args:
        x: 输入张量 [N, C, H, W]
        w: 卷积核权重
        k: FIR 滤波器核
        factor: 下采样因子
        gain: 增益
    """

    assert isinstance(factor, int) and factor >= 1
    _outC, _inC, convH, convW = w.shape
    assert convW == convH
    if k is None:
        k = [1] * factor
    k = _setup_kernel(k) * gain # 1. 设置 FIR 滤波器核
    p = (k.shape[0] - factor) + (convW - 1)
    s = [factor, factor] # 2. 卷积的步长
    
    # 3. 【关键】先使用 FIR 滤波器对输入 x 进行平滑处理
    #    (注意：这里 up=None, down=None，只做平滑)
    x = upfirdn2d(x, torch.tensor(k, device=x.device),
                   pad=((p + 1) // 2, p // 2))
    # 4. 【关键】再执行标准卷积 (F.conv2d)，使用步长 s (factor) 来实现下采样
    return F.conv2d(x, w, stride=s, padding=0)


def _setup_kernel(k):
    """【辅助】将 1D 滤波器核 (如 [1,3,3,1]) 转换为 2D 滤波器核 (通过外积)"""
    k = np.asarray(k, dtype=np.float32)
    if k.ndim == 1:
        k = np.outer(k, k) # [1,3,3,1] x [1,3,3,1] -> 4x4 矩阵
    k /= np.sum(k) # 归一化 (使其总和为 1)
    assert k.ndim == 2
    assert k.shape[0] == k.shape[1]
    return k


def _shape(x, dim):
    """【辅助】获取张量维度的辅助函数"""
    return x.shape[dim]


def upsample_2d(x, k=None, factor=2, gain=1):
    r"""【辅助】单独的上采样 (不带卷积)"""
    assert isinstance(factor, int) and factor >= 1
    if k is None:
        k = [1] * factor
    k = _setup_kernel(k) * (gain * (factor ** 2))
    p = k.shape[0] - factor
    # 调用 upfirdn2d，只传入 up=factor
    return upfirdn2d(x, torch.tensor(k, device=x.device),
                       up=factor, pad=((p + 1) // 2 + factor - 1, p // 2))


def downsample_2d(x, k=None, factor=2, gain=1):
    r"""【辅助】单独的下采样 (不带卷积)"""
    assert isinstance(factor, int) and factor >= 1
    if k is None:
        k = [1] * factor
    k = _setup_kernel(k) * gain
    p = k.shape[0] - factor
    # 调用 upfirdn2d，只传入 down=factor
    return upfirdn2d(x, torch.tensor(k, device=x.device),
                       down=factor, pad=((p + 1) // 2, p // 2))