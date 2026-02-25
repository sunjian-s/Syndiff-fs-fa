# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# ... (版权和许可信息) ...
# ---------------------------------------------------------------

# pylint: skip-file
"""
定义 NCSN++ 所需的层 (Layers++)
改编自 https://github.com/yang-song/score_sde_pytorch/blob/main/models/layerspp.py
"""
from . import layers # 导入基础层 (如 ddpm_conv3x3, NIN)
from . import up_or_down_sampling, dense_layer
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

# --- 从基础层文件中导入，方便本文件调用 ---
conv1x1 = layers.ddpm_conv1x1
conv3x3 = layers.ddpm_conv3x3
NIN = layers.NIN
default_init = layers.default_init
dense = dense_layer.dense # 导入自定义的、带初始化的全连接层

class AdaptiveGroupNorm(nn.Module):
    """
    【核心模块】自适应组归一化 (AdaGN)
    
    这是一种强大的“条件注入”方式 (来自 StyleGAN)。
    它不仅归一化特征图 (x)，还使用一个外部的“风格”向量 (style) 
    来动态地学习“缩放” (gamma) 和“偏移” (beta)。
    
    在 SynDiff 中，这个 "style" 将是 风格向量z (latent_z)。
    """
    def __init__(self, num_groups, in_channel, style_dim):
        super().__init__()

        # 1. 标准的组归一化 (GroupNorm)，但 affine=False
        # (affine=False 意味着 PyTorch 不会自己学习 gamma 和 beta)
        self.norm = nn.GroupNorm(num_groups, in_channel, affine=False, eps=1e-6)
        
        # 2. 一个线性层，用于将输入的 'style' 向量 (style_dim) 
        #    转换为 gamma 和 beta (in_channel * 2)
        self.style = dense(style_dim, in_channel * 2)

        # 3. 初始化：gamma (缩放) 默认为 1, beta (偏移) 默认为 0
        self.style.bias.data[:in_channel] = 1 # gamma
        self.style.bias.data[in_channel:] = 0 # beta

    def forward(self, input, style):
        # 1. 将 style 向量 [B, style_dim] 转换为 [B, C*2]
        style = self.style(style).unsqueeze(2).unsqueeze(3) # -> [B, C*2, 1, 1]
        
        # 2. 将 (gamma, beta) 分开
        gamma, beta = style.chunk(2, 1) # -> [B, C, 1, 1]

        # 3. 标准归一化
        out = self.norm(input)
        # 4. 【关键】应用“自适应”的缩放和偏移
        out = gamma * out + beta

        return out

class GaussianFourierProjection(nn.Module):
    """
    【时间步嵌入】高斯傅里叶投影 (Gaussian Fourier embeddings)
    
    这是另一种将标量时间 't' 转换为高维向量的方式。
    它使用固定的 (requires_grad=False) 随机高斯权重 W。
    """
    def __init__(self, embedding_size=256, scale=1.0):
        super().__init__()
        # 创建一个固定的、随机初始化的权重矩阵 W
        self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)

    def forward(self, x):
        # 将 t (x) 投影到高维
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        # 使用 sin 和 cos (傅里叶特征)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Combine(nn.Module):
    """【辅助】用于融合 U-Net 中的跳跃连接 (Skip Connections)"""

    def __init__(self, dim1, dim2, method='cat'):
        super().__init__()
        # 使用 1x1 卷积来匹配通道数
        self.Conv_0 = conv1x1(dim1, dim2)
        self.method = method

    def forward(self, x, y):
        # x 是来自 U-Net 上一层的特征
        # y 是来自 U-Net 编码器（跳跃连接）的特征
        h = self.Conv_0(x) # 1. 适配 x 的通道
        if self.method == 'cat':
            return torch.cat([h, y], dim=1) # 2. 拼接 (Concatenate)
        elif self.method == 'sum':
            return h + y # 2. 相加
        else:
            raise ValueError(f'Method {self.method} not recognized.')


class AttnBlockpp(nn.Module):
    """【模块】(NCSNpp 风格的) 自注意力块 (Self-Attention Block)"""

    def __init__(self, channels, skip_rescale=False, init_scale=0.):
        super().__init__()
        # GroupNorm 的组数 (num_groups) 是动态计算的
        self.GroupNorm_0 = nn.GroupNorm(num_groups=min(channels // 4, 32), num_channels=channels,
                                        eps=1e-6)
        self.NIN_0 = NIN(channels, channels) # Q (Query)
        self.NIN_1 = NIN(channels, channels) # K (Key)
        self.NIN_2 = NIN(channels, channels) # V (Value)
        self.NIN_3 = NIN(channels, channels, init_scale=init_scale) # 最终输出
        self.skip_rescale = skip_rescale # 是否对残差连接进行缩放

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.GroupNorm_0(x)
        q = self.NIN_0(h)
        k = self.NIN_1(h)
        v = self.NIN_2(h)

        # (与 DDPM 的 AttnBlock 几乎相同的计算)
        w = torch.einsum('bchw,bcij->bhwij', q, k) * (int(C) ** (-0.5))
        w = torch.reshape(w, (B, H, W, H * W))
        w = F.softmax(w, dim=-1)
        w = torch.reshape(w, (B, H, W, H, W))
        h = torch.einsum('bhwij,bcij->bchw', w, v)
        h = self.NIN_3(h)
        
        # 残差连接
        if not self.skip_rescale:
            return x + h
        else:
            # 缩放残差 (有助于稳定训练)
            return (x + h) / np.sqrt(2.)


class Upsample(nn.Module):
    """
    【上采样模块】
    
    一个更复杂的上采样模块，它允许选择是否使用：
    1. with_conv: 是否在插值后进行卷积
    2. fir: 是否使用 FIR (有限脉冲响应) 滤波器进行平滑的上采样 (更高质量)
    """
    def __init__(self, in_ch=None, out_ch=None, with_conv=False, fir=False,
                 fir_kernel=(1, 3, 3, 1)):
        super().__init__()
        out_ch = out_ch if out_ch else in_ch
        if not fir: # 如果不使用 FIR
            if with_conv:
                # (标准 DDPM 风格: 插值 + 3x3 卷积)
                self.Conv_0 = conv3x3(in_ch, out_ch)
        else: # 如果使用 FIR
            if with_conv:
                # (使用一个包含 FIR 的特殊卷积层)
                self.Conv2d_0 = up_or_down_sampling.Conv2d(in_ch, out_ch,
                                                           kernel=3, up=True,
                                                           resample_kernel=fir_kernel,
                                                           use_bias=True,
                                                           kernel_init=default_init())
        self.fir = fir
        self.with_conv = with_conv
        self.fir_kernel = fir_kernel
        self.out_ch = out_ch

    def forward(self, x):
        B, C, H, W = x.shape
        if not self.fir:
            # 1. 最近邻插值 (H, W) -> (H*2, W*2)
            h = F.interpolate(x, (H * 2, W * 2), 'nearest')
            if self.with_conv:
                h = self.Conv_0(h) # 2. (可选) 卷积
        else: # 使用 FIR
            if not self.with_conv:
                # 仅 FIR 上采样 (不带卷积)
                h = up_or_down_sampling.upsample_2d(x, self.fir_kernel, factor=2)
            else:
                # FIR 上采样 (带卷积)
                h = self.Conv2d_0(x)

        return h


class Downsample(nn.Module):
    """
    【下采样模块】
    
    一个更复杂的下采样模块，允许选择：
    1. with_conv: 是使用带步长的卷积，还是简单的平均池化
    2. fir: 是否使用 FIR 滤波器进行平滑的下采样
    """
    def __init__(self, in_ch=None, out_ch=None, with_conv=False, fir=False,
                 fir_kernel=(1, 3, 3, 1)):
        super().__init__()
        out_ch = out_ch if out_ch else in_ch
        if not fir:
            if with_conv:
                # (标准 DDPM 风格: 带步长(stride=2)的 3x3 卷积)
                self.Conv_0 = conv3x3(in_ch, out_ch, stride=2, padding=0)
        else: # 使用 FIR
            if with_conv:
                # (使用一个包含 FIR 的特殊卷积层)
                self.Conv2d_0 = up_or_down_sampling.Conv2d(in_ch, out_ch,
                                                           kernel=3, down=True,
                                                           resample_kernel=fir_kernel,
                                                           use_bias=True,
                                                           kernel_init=default_init())
        self.fir = fir
        self.fir_kernel = fir_kernel
        self.with_conv = with_conv
        self.out_ch = out_ch

    def forward(self, x):
        B, C, H, W = x.shape
        if not self.fir:
            if self.with_conv:
                x = F.pad(x, (0, 1, 0, 1)) # 模拟 'SAME' 填充
                x = self.Conv_0(x)
            else:
                x = F.avg_pool2d(x, 2, stride=2) # 简单的平均池化
        else: # 使用 FIR
            if not self.with_conv:
                x = up_or_down_sampling.downsample_2d(x, self.fir_kernel, factor=2)
            else:
                x = self.Conv2d_0(x)

        return x


class ResnetBlockDDPMpp_Adagn(nn.Module):
    """
    【核心模块】(NCSNpp 风格的) 残差块，使用了 AdaGN (自适应组归一化)
    
    这是 SynDiff U-Net 的核心积木。
    它同时接收两种条件：
    1. temb (时间步 t)
    2. zemb (风格向量 z)
    """

    def __init__(self, act, in_ch, out_ch=None, temb_dim=None, zemb_dim=None, conv_shortcut=False,
                 dropout=0.1, skip_rescale=False, init_scale=0.):
        super().__init__()
        out_ch = out_ch if out_ch else in_ch
        
        # 1. 第一个归一化层 (使用 AdaGN)
        # 【关键】: zemb (风格 z) 在这里被注入
        self.GroupNorm_0 = AdaptiveGroupNorm(min(in_ch // 4, 32), in_ch, zemb_dim)
        
        self.Conv_0 = conv3x3(in_ch, out_ch)
        if temb_dim is not None:
            # 【关键】: temb (时间 t) 的全连接层
            self.Dense_0 = nn.Linear(temb_dim, out_ch)
            self.Dense_0.weight.data = default_init()(self.Dense_0.weight.data.shape)
            nn.init.zeros_(self.Dense_0.bias)
            
        
        # 2. 第二个归一化层 (使用 AdaGN)
        self.GroupNorm_1 = AdaptiveGroupNorm(min(out_ch // 4, 32), out_ch, zemb_dim)
        
        self.Dropout_0 = nn.Dropout(dropout)
        self.Conv_1 = conv3x3(out_ch, out_ch, init_scale=init_scale)
        
        # 跳跃连接
        if in_ch != out_ch:
            if conv_shortcut:
                self.Conv_2 = conv3x3(in_ch, out_ch)
            else:
                self.NIN_0 = NIN(in_ch, out_ch) # 1x1 卷积

        self.skip_rescale = skip_rescale
        self.act = act
        self.out_ch = out_ch
        self.conv_shortcut = conv_shortcut

    def forward(self, x, temb=None, zemb=None):
        # x = 图像特征, temb = 时间 t 嵌入, zemb = 风格 z 嵌入
        
        # 1. 预激活 (Pre-activation)
        # 【注入 zemb 1】
        h = self.act(self.GroupNorm_0(x, zemb))
        h = self.Conv_0(h)
        
        # 【注入 temb】
        if temb is not None:
            h += self.Dense_0(self.act(temb))[:, :, None, None]
            
        # 【注入 zemb 2】
        h = self.act(self.GroupNorm_1(h, zemb))
        h = self.Dropout_0(h)
        h = self.Conv_1(h)
        
        # 计算跳跃连接
        if x.shape[1] != self.out_ch:
            if self.conv_shortcut:
                x = self.Conv_2(x)
            else:
                x = self.NIN_0(x)
                
        # 残差相加
        if not self.skip_rescale:
            return x + h
        else:
            return (x + h) / np.sqrt(2.)


class ResnetBlockBigGANpp_Adagn(nn.Module):
    """
    【核心模块】(BigGAN 风格的) 残差块，使用了 AdaGN
    
    这个版本集成了“上采样”和“下采样”功能。
    """
    def __init__(self, act, in_ch, out_ch=None, temb_dim=None, zemb_dim=None, up=False, down=False,
                 dropout=0.1, fir=False, fir_kernel=(1, 3, 3, 1),
                 skip_rescale=True, init_scale=0.):
        super().__init__()

        out_ch = out_ch if out_ch else in_ch
        # 【注入 zemb 1】
        self.GroupNorm_0 = AdaptiveGroupNorm(min(in_ch // 4, 32), in_ch, zemb_dim)
        
        self.up = up
        self.down = down
        self.fir = fir
        self.fir_kernel = fir_kernel

        self.Conv_0 = conv3x3(in_ch, out_ch)
        if temb_dim is not None:
            # 【注入 temb】
            self.Dense_0 = nn.Linear(temb_dim, out_ch)
            self.Dense_0.weight.data = default_init()(self.Dense_0.weight.shape)
            nn.init.zeros_(self.Dense_0.bias)
        
        # 【注入 zemb 2】
        self.GroupNorm_1 = AdaptiveGroupNorm(min(out_ch // 4, 32), out_ch, zemb_dim)
        self.Dropout_0 = nn.Dropout(dropout)
        self.Conv_1 = conv3x3(out_ch, out_ch, init_scale=init_scale)
        if in_ch != out_ch or up or down:
            # 【跳跃连接】
            self.Conv_2 = conv1x1(in_ch, out_ch)

        self.skip_rescale = skip_rescale
        self.act = act
        self.in_ch = in_ch
        self.out_ch = out_ch

    def forward(self, x, temb=None, zemb=None):
        h = self.act(self.GroupNorm_0(x, zemb)) # 预激活 (注入 zemb)

        # 【关键】在卷积之前先执行上/下采样
        if self.up:
            if self.fir:
                h = up_or_down_sampling.upsample_2d(h, self.fir_kernel, factor=2)
                x = up_or_down_sampling.upsample_2d(x, self.fir_kernel, factor=2)
            else:
                h = up_or_down_sampling.naive_upsample_2d(h, factor=2)
                x = up_or_down_sampling.naive_upsample_2d(x, factor=2)
        elif self.down:
            if self.fir:
                h = up_or_down_sampling.downsample_2d(h, self.fir_kernel, factor=2)
                x = up_or_down_sampling.downsample_2d(x, self.fir_kernel, factor=2)
            else:
                h = up_or_down_sampling.naive_downsample_2d(h, factor=2)
                x = up_or_down_sampling.naive_downsample_2d(x, factor=2)

        h = self.Conv_0(h)
        if temb is not None:
            h += self.Dense_0(self.act(temb))[:, :, None, None] # 注入 temb
        h = self.act(self.GroupNorm_1(h, zemb)) # 注入 zemb
        h = self.Dropout_0(h)
        h = self.Conv_1(h)
        
        if self.in_ch != self.out_ch or self.up or self.down:
            x = self.Conv_2(x) # 适配跳跃连接

        if not self.skip_rescale:
            return x + h
        else:
            return (x + h) / np.sqrt(2.) # 残差相加 (带缩放)
    

class ResnetBlockBigGANpp_Adagn_one(nn.Module):
    """
    【模块】(BigGAN 风格的) 残差块，使用了 AdaGN
    
    这个版本与上一个 (ResnetBlockBigGANpp_Adagn) 几乎相同，
    唯一的区别是第二个归一化层 (GroupNorm_1) 是
    【标准】的 GroupNorm，而不是【自适应】的 AdaptiveGroupNorm。
    
    这意味着 'zemb' (风格 z) 只在块的开头被注入一次。
    """
    def __init__(self, act, in_ch, out_ch=None, temb_dim=None, zemb_dim=None, up=False, down=False,
                 dropout=0.1, fir=False, fir_kernel=(1, 3, 3, 1),
                 skip_rescale=True, init_scale=0.):
        super().__init__()

        out_ch = out_ch if out_ch else in_ch
        # 【注入 zemb 1】(使用 AdaGN)
        self.GroupNorm_0 = AdaptiveGroupNorm(min(in_ch // 4, 32), in_ch, zemb_dim)
        
        self.up = up
        self.down = down
        self.fir = fir
        self.fir_kernel = fir_kernel

        self.Conv_0 = conv3x3(in_ch, out_ch)
        if temb_dim is not None:
            # 【注入 temb】
            self.Dense_0 = nn.Linear(temb_dim, out_ch)
            self.Dense_0.weight.data = default_init()(self.Dense_0.weight.shape)
            nn.init.zeros_(self.Dense_0.bias)
        
        # 【区别点】
        # 这里使用的是【标准】GroupNorm，没有 zemb 输入
        self.GroupNorm_1 = nn.GroupNorm(num_groups=min(out_ch // 4, 32), num_channels=out_ch, eps=1e-6)
        
        self.Dropout_0 = nn.Dropout(dropout)
        self.Conv_1 = conv3x3(out_ch, out_ch, init_scale=init_scale)
        if in_ch != out_ch or up or down:
            self.Conv_2 = conv1x1(in_ch, out_ch)

        self.skip_rescale = skip_rescale
        self.act = act
        self.in_ch = in_ch
        self.out_ch = out_ch

    def forward(self, x, temb=None, zemb=None):
        h = self.act(self.GroupNorm_0(x, zemb)) # 预激活 (注入 zemb)

        if self.up:
            if self.fir:
                h = up_or_down_sampling.upsample_2d(h, self.fir_kernel, factor=2)
                x = up_or_down_sampling.upsample_2d(x, self.fir_kernel, factor=2)
            else:
                h = up_or_down_sampling.naive_upsample_2d(h, factor=2)
                x = up_or_down_sampling.naive_upsample_2d(x, factor=2)
        elif self.down:
            if self.fir:
                h = up_or_down_sampling.downsample_2d(h, self.fir_kernel, factor=2)
                x = up_or_down_sampling.downsample_2d(x, self.fir_kernel, factor=2)
            else:
                h = up_or_down_sampling.naive_downsample_2d(h, factor=2)
                x = up_or_down_sampling.naive_downsample_2d(x, factor=2)

        h = self.Conv_0(h)
        if temb is not None:
            h += self.Dense_0(self.act(temb))[:, :, None, None] # 注入 temb
        
        # 【区别点】这里传入 h，而不是 (h, zemb)
        h = self.act(self.GroupNorm_1(h)) 
        h = self.Dropout_0(h)
        h = self.Conv_1(h)
        
        if self.in_ch != self.out_ch or self.up or self.down:
            x = self.Conv_2(x) # 适配跳跃连接

        if not self.skip_rescale:
            return x + h
        else:
            return (x + h) / np.sqrt(2.) # 残差相加