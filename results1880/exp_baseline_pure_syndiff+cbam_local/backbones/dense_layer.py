# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This file has been modified from a file released under the MIT License.
# ... (版权和许可信息) ...
# ---------------------------------------------------------------


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import _calculate_fan_in_and_fan_out
import numpy as np
# 不需要看

def _calculate_correct_fan(tensor, mode):
    """
    辅助函数：计算张量的 'fan_in' (输入连接数) 或 'fan_out' (输出连接数)。
    这是 Kaiming 初始化所必需的。
    """
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out', 'fan_avg']
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == 'fan_in' else fan_out


def kaiming_uniform_(tensor, gain=1., mode='fan_in'):
    r"""
    【核心】Kaiming (He) 均匀分布初始化。
    
    这是一个强大的权重初始化方法，旨在帮助深度网络（特别是使用 ReLU 激活的）
    更稳定地收敛。
    
    它根据 'fan_mode' (通常是 'fan_in') 计算出一个边界值 'bound'，
    然后在 U(-bound, bound) 的均匀分布中采样来填充权重张量。
    """
    fan = _calculate_correct_fan(tensor, mode) # 1. 计算 fan_in 或 fan_out
    # gain = calculate_gain(nonlinearity, a)
    var = gain / max(1., fan) # 2. 计算方差
    bound = math.sqrt(3.0 * var)  # 3. 计算均匀分布的边界
    with torch.no_grad():
        return tensor.uniform_(-bound, bound) # 4. 用随机数填充张量


def variance_scaling_init_(tensor, scale):
    """
    【自定义初始化】
    
    一个自定义的“方差缩放”初始化器。
    它调用 kaiming_uniform_，但使用 'fan_avg' (输入和输出的平均值) 
    并将传入的 'scale' (缩放因子) 作为增益 (gain)。
    """
    return kaiming_uniform_(tensor, gain=1e-10 if scale == 0 else scale, mode='fan_avg')


def dense(in_channels, out_channels, init_scale=1.):
    """
    【工厂函数】创建一个“全连接层” (nn.Linear)
    
    这个函数不仅创建了层，还立刻使用我们自定义的 'variance_scaling_init_'
    来初始化权重，并将偏置(bias)初始化为 0。
    """
    lin = nn.Linear(in_channels, out_channels)
    variance_scaling_init_(lin.weight, scale=init_scale) # 自定义权重初始化
    nn.init.zeros_(lin.bias) # 偏置设为 0
    return lin

def conv2d(in_planes, out_planes, kernel_size=(3, 3), stride=1, dilation=1, padding=1, bias=True, padding_mode='zeros',
           init_scale=1.):
    """
    【工厂函数】创建一个“2D 卷积层” (nn.Conv2d)
    
    与 'dense' 函数类似，它创建了一个 Conv2d 层，并立刻使用
    'variance_scaling_init_' 初始化权重，偏置(bias)初始化为 0。
    """
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                     bias=bias, padding_mode=padding_mode)
    variance_scaling_init_(conv.weight, scale=init_scale) # 自定义权重初始化
    if bias:
        nn.init.zeros_(conv.bias) # 偏置设为 0
    return conv