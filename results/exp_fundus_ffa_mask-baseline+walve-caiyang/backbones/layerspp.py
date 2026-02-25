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
import torchvision.ops 
#from .snake_conv import DynamicSnakeConv  # ✅ 确保 snake_conv.py 在同级目录下

# --- 从基础层文件中导入，方便本文件调用 ---
conv1x1 = layers.ddpm_conv1x1
conv3x3 = layers.ddpm_conv3x3
NIN = layers.NIN
default_init = layers.default_init
dense = dense_layer.dense 

# =========================================================================
# 可变形卷积 v2 (DCNv2) - [保留定义以备对比实验]
# =========================================================================
class DeformableConv2d(nn.Module):
    """
    可变形卷积 v2 (DCNv2)
    注：在当前版本中，默认使用下方的 DynamicSnakeConv 替代此模块。
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, init_scale=1.):
        super(DeformableConv2d, self).__init__()
        
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        
        self.offset_conv = nn.Conv2d(in_channels, 
                                     3 * kernel_size * kernel_size, 
                                     kernel_size=kernel_size, 
                                     stride=stride, 
                                     padding=padding, 
                                     bias=True)
        
        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)
        
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.weight.data = default_init(scale=init_scale)(self.weight.data.shape)
        if bias:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        out = self.offset_conv(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return torchvision.ops.deform_conv2d(input=x, 
                                             offset=offset, 
                                             weight=self.weight, 
                                             bias=self.bias, 
                                             stride=self.stride, 
                                             padding=self.padding, 
                                             mask=mask)

# =========================================================================
# 基础层定义 (AdaGN, Fourier, etc.)
# =========================================================================

class AdaptiveGroupNorm(nn.Module):
    """
    【核心模块】自适应组归一化 (AdaGN)
    用于将风格向量 z 注入到网络中。
    """
    def __init__(self, num_groups, in_channel, style_dim):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, in_channel, affine=False, eps=1e-6)
        self.style = dense(style_dim, in_channel * 2)
        self.style.bias.data[:in_channel] = 1 # gamma
        self.style.bias.data[in_channel:] = 0 # beta

    def forward(self, input, style):
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)
        out = self.norm(input)
        out = gamma * out + beta
        return out

class GaussianFourierProjection(nn.Module):
    """时间步嵌入"""
    def __init__(self, embedding_size=256, scale=1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Combine(nn.Module):
    """用于融合 U-Net 中的跳跃连接"""
    def __init__(self, dim1, dim2, method='cat'):
        super().__init__()
        self.Conv_0 = conv1x1(dim1, dim2)
        self.method = method

    def forward(self, x, y):
        h = self.Conv_0(x)
        if self.method == 'cat':
            return torch.cat([h, y], dim=1)
        elif self.method == 'sum':
            return h + y
        else:
            raise ValueError(f'Method {self.method} not recognized.')


class AttnBlockpp(nn.Module):
    """(NCSNpp 风格的) 自注意力块"""
    def __init__(self, channels, skip_rescale=False, init_scale=0.):
        super().__init__()
        self.GroupNorm_0 = nn.GroupNorm(num_groups=min(channels // 4, 32), num_channels=channels, eps=1e-6)
        self.NIN_0 = NIN(channels, channels)
        self.NIN_1 = NIN(channels, channels)
        self.NIN_2 = NIN(channels, channels)
        self.NIN_3 = NIN(channels, channels, init_scale=init_scale)
        self.skip_rescale = skip_rescale

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.GroupNorm_0(x)
        q = self.NIN_0(h)
        k = self.NIN_1(h)
        v = self.NIN_2(h)

        w = torch.einsum('bchw,bcij->bhwij', q, k) * (int(C) ** (-0.5))
        w = torch.reshape(w, (B, H, W, H * W))
        w = F.softmax(w, dim=-1)
        w = torch.reshape(w, (B, H, W, H, W))
        h = torch.einsum('bhwij,bcij->bchw', w, v)
        h = self.NIN_3(h)
        
        if not self.skip_rescale:
            return x + h
        else:
            return (x + h) / np.sqrt(2.)


# =========================================================================
# ✅ 新增：Haar 逆小波变换辅助类
# =========================================================================
class HaarUpsampling(nn.Module):
    """
    使用 Haar 逆小波变换 (IDWT) 进行上采样。
    将 [B, 4*C, H, W] -> [B, C, H*2, W*2]
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x 的通道数必须是 4 的倍数 (对应 LL, HL, LH, HH)
        B, C4, H, W = x.shape
        C = C4 // 4
        
        # 1. 拆分通道
        LL, HL, LH, HH = torch.chunk(x, 4, dim=1)
        
        # 2. Haar 逆变换公式
        # 对应下采样时的 * 0.5，这里进行逆运算恢复数值幅度
        # x00: 左上, x01: 右上, x10: 左下, x11: 右下
        x00 = (LL + HL + LH + HH) * 0.5
        x01 = (LL + HL - LH - HH) * 0.5
        x10 = (LL - HL + LH - HH) * 0.5
        x11 = (LL - HL - LH + HH) * 0.5
        
        # 3. 像素重组 (Interleave)
        # 创建一个 2倍大小的空张量
        out = torch.zeros(B, C, H * 2, W * 2, device=x.device, dtype=x.dtype)
        
        # 填空：像棋盘一样把像素填回去
        out[:, :, 0::2, 0::2] = x00
        out[:, :, 0::2, 1::2] = x01
        out[:, :, 1::2, 0::2] = x10
        out[:, :, 1::2, 1::2] = x11
        
        return out

# =========================================================================
# ✅ 修改后：基于小波的 Upsample 类
# =========================================================================
class Upsample(nn.Module):
    """
    上采样模块 - 改为 Wavelet Upsampling (IDWT)
    优势：从通道中恢复高频细节，而不是通过插值“猜”出模糊的像素。
    """
    def __init__(self, in_ch=None, out_ch=None, with_conv=False, fir=False, fir_kernel=(1, 3, 3, 1)):
        super().__init__()
        out_ch = out_ch if out_ch else in_ch
        self.out_ch = out_ch
        
        # 1. 通道扩展层
        # IDWT 需要 4 个分量 (LL, LH, HL, HH) 才能合成一个像素
        # 所以我们需要把输入通道数映射到 4 * out_ch
        self.conv_expand = nn.Conv2d(in_ch, out_ch * 4, kernel_size=1, stride=1, padding=0, bias=True)
        
        # 2. 核心部件：Haar 逆变换
        self.haar_idwt = HaarUpsampling()

        # 初始化 1x1 卷积
        self.conv_expand.weight.data = default_init()(self.conv_expand.weight.data.shape)
        nn.init.zeros_(self.conv_expand.bias)

    def forward(self, x):
        # 1. 扩展通道: [B, in_ch, H, W] -> [B, 4*out_ch, H, W]
        # 这一步是让网络“学习”如何把低维特征解压成频域分量
        x_expanded = self.conv_expand(x)
        
        # 2. 执行逆小波变换: [B, 4*out_ch, H, W] -> [B, out_ch, H*2, W*2]
        h = self.haar_idwt(x_expanded)
        
        return h

# =========================================================================
# ✅ 新增：Haar 小波变换辅助类 (无需额外安装库)
# =========================================================================
class HaarDownsampling(nn.Module):
    """
    使用 Haar 小波变换进行无损下采样。
    将 [B, C, H, W] -> [B, 4*C, H/2, W/2]
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # 提取四个子图位置
        # x00: 左上, x01: 右上, x10: 左下, x11: 右下
        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]
        
        # Haar 变换公式 (除以2是为了保持数值范围稳定)
        # LL: 低频近似 (平滑部分)
        LL = (x00 + x01 + x10 + x11) * 0.5
        # HL: 水平细节 (检测横向血管)
        HL = (x00 + x01 - x10 - x11) * 0.5
        # LH: 垂直细节 (检测纵向血管)
        LH = (x00 - x01 + x10 - x11) * 0.5
        # HH: 对角细节 (检测斜向血管/噪点)
        HH = (x00 - x01 - x10 + x11) * 0.5
        
        # 在通道维度拼接: [B, C, H/2, W/2] -> [B, 4C, H/2, W/2]
        return torch.cat([LL, HL, LH, HH], dim=1)

# =========================================================================
# ✅ 修改后：基于小波的 Downsample 类
# =========================================================================
class Downsample(nn.Module):
    """
    下采样模块 - 改为 Wavelet Downsampling
    优势：无损保留高频微血管信息，避免 FIR 滤波器的模糊效应。
    """
    def __init__(self, in_ch=None, out_ch=None, with_conv=False, fir=False, fir_kernel=(1, 3, 3, 1)):
        super().__init__()
        # 为了兼容以前的代码调用，保留了 fir 等参数名，但实际逻辑中不再使用它们
        out_ch = out_ch if out_ch else in_ch
        
        self.out_ch = out_ch
        
        # 1. 核心部件：Haar 小波变换
        self.haar_dwt = HaarDownsampling()
        
        # 2. 通道融合层
        # 小波变换会将通道数变成原来的 4 倍 (in_ch * 4)
        # 我们需要用 1x1 卷积将其降维到目标通道数 (out_ch)
        self.conv_1x1 = nn.Conv2d(in_ch * 4, out_ch, kernel_size=1, stride=1, padding=0, bias=True)
        
        # 初始化 1x1 卷积
        self.conv_1x1.weight.data = default_init()(self.conv_1x1.weight.data.shape)
        nn.init.zeros_(self.conv_1x1.bias)

    def forward(self, x):
        # 1. 执行小波变换: [B, C, H, W] -> [B, 4C, H/2, W/2]
        x_dwt = self.haar_dwt(x)
        
        # 2. 融合频域信息: [B, 4C, H/2, W/2] -> [B, out_ch, H/2, W/2]
        out = self.conv_1x1(x_dwt)
        
        return out
# =========================================================================
# ResNet Blocks (已集成 Snake Conv)
# =========================================================================

class ResnetBlockDDPMpp_Adagn(nn.Module):
    """
    (NCSNpp 风格) 残差块 - Baseline 版本
    【修改说明】已移除 DynamicSnakeConv，恢复为标准 conv3x3
    """
    def __init__(self, act, in_ch, out_ch=None, temb_dim=None, zemb_dim=None, conv_shortcut=False,
                 dropout=0.1, skip_rescale=False, init_scale=0.):
        super().__init__()
        out_ch = out_ch if out_ch else in_ch
        
        self.GroupNorm_0 = AdaptiveGroupNorm(min(in_ch // 4, 32), in_ch, zemb_dim)
        
        # =========== 【回滚修改点 START】 ===========
        # 原代码：self.Conv_0 = DynamicSnakeConv(in_ch, out_ch, kernel_size=5)
        # 修改后：使用标准 3x3 卷积
        self.Conv_0 = conv3x3(in_ch, out_ch)
        # =========== 【回滚修改点 END】 =============
        
        if temb_dim is not None:
            self.Dense_0 = nn.Linear(temb_dim, out_ch)
            self.Dense_0.weight.data = default_init()(self.Dense_0.weight.data.shape)
            nn.init.zeros_(self.Dense_0.bias)
            
        self.GroupNorm_1 = AdaptiveGroupNorm(min(out_ch // 4, 32), out_ch, zemb_dim)
        self.Dropout_0 = nn.Dropout(dropout)
        
        self.Conv_1 = conv3x3(out_ch, out_ch, init_scale=init_scale)
        
        if in_ch != out_ch:
            if conv_shortcut:
                self.Conv_2 = conv3x3(in_ch, out_ch)
            else:
                self.NIN_0 = NIN(in_ch, out_ch)

        self.skip_rescale = skip_rescale
        self.act = act
        self.out_ch = out_ch
        self.conv_shortcut = conv_shortcut

    def forward(self, x, temb=None, zemb=None):
        h = self.act(self.GroupNorm_0(x, zemb))
        h = self.Conv_0(h) # 这里现在是普通的卷积了
        
        if temb is not None:
            h += self.Dense_0(self.act(temb))[:, :, None, None]
            
        h = self.act(self.GroupNorm_1(h, zemb))
        h = self.Dropout_0(h)
        h = self.Conv_1(h)
        
        if x.shape[1] != self.out_ch:
            if self.conv_shortcut:
                x = self.Conv_2(x)
            else:
                x = self.NIN_0(x)
                
        if not self.skip_rescale:
            return x + h
        else:
            return (x + h) / np.sqrt(2.)
        
class ResnetBlockBigGANpp_Adagn_one(nn.Module):
    """
    (BigGAN 风格 - 单次注入) 残差块
    保留此定义以防止 ncsnpp_generator_adagn.py 报错
    """
    def __init__(self, act, in_ch, out_ch=None, temb_dim=None, zemb_dim=None, up=False, down=False,
                 dropout=0.1, fir=False, fir_kernel=(1, 3, 3, 1),
                 skip_rescale=True, init_scale=0.):
        super().__init__()

        out_ch = out_ch if out_ch else in_ch
        self.GroupNorm_0 = AdaptiveGroupNorm(min(in_ch // 4, 32), in_ch, zemb_dim)
        
        self.up = up
        self.down = down
        self.fir = fir
        self.fir_kernel = fir_kernel

        # 同样使用我们的小波模块
        if self.up:
            self.wavelet_up = Upsample(in_ch, in_ch)
        
        if self.down:
            self.wavelet_down = Downsample(in_ch, in_ch)

        # ------------------------------------------------------------
        # 注意：这里我们使用普通卷积，因为这个类通常不作为主力使用
        self.Conv_0 = conv3x3(in_ch, out_ch)
        
        if temb_dim is not None:
            self.Dense_0 = nn.Linear(temb_dim, out_ch)
            self.Dense_0.weight.data = default_init()(self.Dense_0.weight.shape)
            nn.init.zeros_(self.Dense_0.bias)
        
        # 注意：这个变体通常使用普通的 GroupNorm 作为第二层，而不是 AdaGN
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
        h = self.act(self.GroupNorm_0(x, zemb))

        if self.up:
            h = self.wavelet_up(h)
            x = self.wavelet_up(x)
        elif self.down:
            h = self.wavelet_down(h)
            x = self.wavelet_down(x)

        h = self.Conv_0(h)
        
        if temb is not None:
            h += self.Dense_0(self.act(temb))[:, :, None, None]
        
        h = self.act(self.GroupNorm_1(h)) # 注意：这里不需要 zemb
        h = self.Dropout_0(h)
        h = self.Conv_1(h)
        
        if self.in_ch != self.out_ch or self.up or self.down:
            x = self.Conv_2(x)

        if not self.skip_rescale:
            return x + h
        else:
            return (x + h) / np.sqrt(2.)



class ResnetBlockBigGANpp_Adagn(nn.Module):
    """
    (BigGAN 风格) 残差块 - ✅ 全面升级为小波变换版本 (All-Wavelet)
    """
    def __init__(self, act, in_ch, out_ch=None, temb_dim=None, zemb_dim=None, up=False, down=False,
                 dropout=0.1, fir=False, fir_kernel=(1, 3, 3, 1),
                 skip_rescale=True, init_scale=0.):
        super().__init__()

        out_ch = out_ch if out_ch else in_ch
        self.GroupNorm_0 = AdaptiveGroupNorm(min(in_ch // 4, 32), in_ch, zemb_dim)
        
        self.up = up
        self.down = down

        # ============================================================
        # ✅ 修改点 1: 初始化小波上下采样模块
        # 注意：这里我们设定输出通道 = 输入通道，只负责变分辨率，不负责变通道
        # 具体的通道变化由后面的 Conv_0 或 Conv_2 负责
        # ============================================================
        if self.up:
            # 使用刚才改写的 Upsample 类 (内部是 Haar IDWT)
            self.wavelet_up = Upsample(in_ch, in_ch)
        
        if self.down:
            # 使用刚才改写的 Downsample 类 (内部是 Haar DWT)
            self.wavelet_down = Downsample(in_ch, in_ch)

        # ------------------------------------------------------------

        self.Conv_0 = conv3x3(in_ch, out_ch)
        
        if temb_dim is not None:
            self.Dense_0 = nn.Linear(temb_dim, out_ch)
            self.Dense_0.weight.data = default_init()(self.Dense_0.weight.shape)
            nn.init.zeros_(self.Dense_0.bias)
        
        self.GroupNorm_1 = AdaptiveGroupNorm(min(out_ch // 4, 32), out_ch, zemb_dim)
        self.Dropout_0 = nn.Dropout(dropout)
        self.Conv_1 = conv3x3(out_ch, out_ch, init_scale=init_scale)
        
        # Shortcut (跳跃连接) 的卷积
        if in_ch != out_ch or up or down:
            self.Conv_2 = conv1x1(in_ch, out_ch)

        self.skip_rescale = skip_rescale
        self.act = act
        self.in_ch = in_ch
        self.out_ch = out_ch

    def forward(self, x, temb=None, zemb=None):
        h = self.act(self.GroupNorm_0(x, zemb))

        # ============================================================
        # ✅ 修改点 2: 使用小波模块替代原来的 up/down 函数
        # ============================================================
        if self.up:
            # 处理主路径 h
            h = self.wavelet_up(h)
            # 处理 shortcut 路径 x (必须同步上采样)
            x = self.wavelet_up(x) 
        elif self.down:
            # 处理主路径 h
            h = self.wavelet_down(h)
            # 处理 shortcut 路径 x (必须同步下采样)
            x = self.wavelet_down(x)
        # ============================================================

        h = self.Conv_0(h)
        
        if temb is not None:
            h += self.Dense_0(self.act(temb))[:, :, None, None]
        
        h = self.act(self.GroupNorm_1(h, zemb))
        h = self.Dropout_0(h)
        h = self.Conv_1(h)
        
        # 处理 Shortcut
        if self.in_ch != self.out_ch or self.up or self.down:
            x = self.Conv_2(x)

        if not self.skip_rescale:
            return x + h
        else:
            return (x + h) / np.sqrt(2.)