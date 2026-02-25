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

# --- 从基础层文件中导入，方便本文件调用 ---
conv1x1 = layers.ddpm_conv1x1
conv3x3 = layers.ddpm_conv3x3
NIN = layers.NIN
default_init = layers.default_init
dense = dense_layer.dense 

# =========================================================================
# ✅ 1. 新增：混合多尺度抗混叠下采样 (HybridDownsample)
# =========================================================================
class HybridDownsample(nn.Module):
    """
    【创新模块】混合多尺度抗混叠下采样 (Hybrid SP-MSD) - 修正版
    结合 FIR 抗混叠 + 多尺度特征提取，用于保留微小病灶和血管结构。
    """
    def __init__(self, in_ch=None, out_ch=None, with_conv=False, fir=False, fir_kernel=(1, 3, 3, 1)):
        super().__init__()
        # 兼容基线参数
        in_ch = in_ch if in_ch is not None else 64
        out_ch = out_ch if out_ch else in_ch
        
        # 1. FIR 抗混叠滤波器 (固定Buffer，物理保真)
        # 即使 fir=False，我们在内部也强制使用抗混叠逻辑来保护病灶
        self.fir_kernel = fir_kernel
        kernel = torch.tensor(self.fir_kernel).float()
        kernel = kernel[:, None] * kernel[None, :]
        kernel = kernel / kernel.sum()
        kernel = kernel[None, None, :, :].repeat(in_ch, 1, 1, 1)
        self.register_buffer('blur_kernel', kernel)
        self.groups = in_ch

        # 2. 多尺度分支 (Inception Style)
        # 计算通道分配：确保拼接后正好等于 out_ch
        mid_c = out_ch // 4
        rem_c = out_ch - mid_c * 3
        
        # 分支A：1x1 Conv → 微动脉瘤（点）
        self.branch_point = nn.Sequential(
            nn.Conv2d(in_ch, mid_c, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # 分支B：3x3 Conv → 血管边缘（线）
        self.branch_edge = nn.Sequential(
            nn.Conv2d(in_ch, mid_c, 3, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # 分支C：3x3 Dilated Conv (d=2) → 出血斑/大结构（面）
        self.branch_area = nn.Sequential(
            nn.Conv2d(in_ch, mid_c, 3, padding=2, dilation=2, bias=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # 分支D：1x1 细节补偿
        self.branch_detail = nn.Sequential(
            nn.Conv2d(in_ch, rem_c, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 3. 下采样操作
        self.pool = nn.AvgPool2d(2, stride=2)

    def forward(self, x):
        # Step 1: FIR抗混叠滤波 (Pre-filtering)
        if x.shape[2] > 2 and x.shape[3] > 2:
            x_pad = F.pad(x, (1, 2, 1, 2), mode='reflect')
            x_anti = F.conv2d(x_pad, self.blur_kernel, groups=self.groups)
        else:
            x_anti = x 

        # Step 2: 多尺度特征提取
        f_point = self.branch_point(x_anti)
        f_edge = self.branch_edge(x_anti)
        f_area = self.branch_area(x_anti)
        f_detail = self.branch_detail(x_anti)
        
        # Step 3: 拼接融合 (Concatenate)
        # [B, mid_c, H, W] * 3 + [B, rem_c, H, W] -> [B, out_ch, H, W]
        out = torch.cat([f_point, f_edge, f_area, f_detail], dim=1)
        
        # Step 4: 下采样
        out = self.pool(out)
        
        return out

# =========================================================================
# 基础层定义
# =========================================================================

class DeformableConv2d(nn.Module):
    """ DCNv2 (保留定义) """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, init_scale=1.):
        super(DeformableConv2d, self).__init__()
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.offset_conv = nn.Conv2d(in_channels, 3 * kernel_size * kernel_size, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
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
        return torchvision.ops.deform_conv2d(input=x, offset=offset, weight=self.weight, bias=self.bias, stride=self.stride, padding=self.padding, mask=mask)

class AdaptiveGroupNorm(nn.Module):
    def __init__(self, num_groups, in_channel, style_dim):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, in_channel, affine=False, eps=1e-6)
        self.style = dense(style_dim, in_channel * 2)
        self.style.bias.data[:in_channel] = 1 
        self.style.bias.data[in_channel:] = 0 

    def forward(self, input, style):
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)
        out = self.norm(input)
        out = gamma * out + beta
        return out

class GaussianFourierProjection(nn.Module):
    def __init__(self, embedding_size=256, scale=1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class Combine(nn.Module):
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
        if not self.skip_rescale: return x + h
        else: return (x + h) / np.sqrt(2.)

class Upsample(nn.Module):
    """
    上采样模块 - DDP 修正版
    确保只初始化当前配置所需的层，避免 unused parameters 报错
    """
    def __init__(self, in_ch=None, out_ch=None, with_conv=False, fir=False, fir_kernel=(1, 3, 3, 1)):
        super().__init__()
        out_ch = out_ch if out_ch else in_ch
        
        self.fir = fir
        self.with_conv = with_conv
        self.fir_kernel = fir_kernel
        self.out_ch = out_ch

        # 逻辑互斥：只定义需要的那一个卷积层
        if not fir:
            if with_conv:
                # 分支 A: 普通插值 + 卷积
                self.Conv_0 = conv3x3(in_ch, out_ch)
        else:
            if with_conv:
                # 分支 B: FIR 滤波 + 卷积
                self.Conv2d_0 = up_or_down_sampling.Conv2d(in_ch, out_ch, kernel=3, up=True,
                                                           resample_kernel=fir_kernel,
                                                           use_bias=True, kernel_init=default_init())

    def forward(self, x):
        B, C, H, W = x.shape
        
        # 逻辑必须与 __init__ 严格对应
        if not self.fir:
            # 分支 A
            h = F.interpolate(x, (H * 2, W * 2), 'nearest')
            if self.with_conv:
                h = self.Conv_0(h)
        else:
            # 分支 B
            if not self.with_conv:
                h = up_or_down_sampling.upsample_2d(x, self.fir_kernel, factor=2)
            else:
                h = self.Conv2d_0(x)
                
        return h

# =========================================================================
# ✅ 2. 修改：下采样模块 (Downsample)
# =========================================================================
class Downsample(nn.Module):
    """下采样模块 - 已升级为使用 HybridDownsample"""
    def __init__(self, in_ch=None, out_ch=None, with_conv=False, fir=False, fir_kernel=(1, 3, 3, 1)):
        super().__init__()
        out_ch = out_ch if out_ch else in_ch
        
        # 标记是否使用新模块
        self.use_new_downsample = False 

        if with_conv:
            # ✅ 【关键修改】: 只要开启卷积下采样，就强制使用我们的混合下采样模块
            # 这替代了原来的 self.Conv_0 = conv3x3(...)
            self.Conv_0 = HybridDownsample(in_ch, out_ch)
            self.use_new_downsample = True
        
        # # 保留旧参数定义，防止旧代码引用报错
        # self.fir = fir
        # self.fir_kernel = fir_kernel
        # self.with_conv = with_conv
        # self.out_ch = out_ch
        
        # # 兼容旧逻辑的定义 (虽然可能不会用到)
        # if not fir and not with_conv:
        #      pass # AvgPool 在 forward 里动态调用
        # if fir:
        #     if with_conv:
        #         self.Conv2d_0 = up_or_down_sampling.Conv2d(in_ch, out_ch, kernel=3, down=True,
        #                                                    resample_kernel=fir_kernel,
        #                                                    use_bias=True, kernel_init=default_init())

    def forward(self, x):
        # ✅ 【关键修改】: 优先使用新模块
        if self.use_new_downsample:
            return self.Conv_0(x)

        # --- 以下是旧代码逻辑 (保留作为 Else 分支) ---
        B, C, H, W = x.shape
        if not self.fir:
            if self.with_conv:
                x = F.pad(x, (0, 1, 0, 1))
                x = self.Conv_0(x)
            else:
                x = F.avg_pool2d(x, 2, stride=2)
        else:
            if not self.with_conv:
                x = up_or_down_sampling.downsample_2d(x, self.fir_kernel, factor=2)
            else:
                x = self.Conv2d_0(x)
        return x

# =========================================================================
# ResNet Blocks (Baseline: 普通卷积)
# =========================================================================

class ResnetBlockDDPMpp_Adagn(nn.Module):
    """ (NCSNpp 风格) 残差块 - Baseline 版本 (标准 Conv3x3) """
    def __init__(self, act, in_ch, out_ch=None, temb_dim=None, zemb_dim=None, conv_shortcut=False,
                 dropout=0.1, skip_rescale=False, init_scale=0.):
        super().__init__()
        out_ch = out_ch if out_ch else in_ch
        
        self.GroupNorm_0 = AdaptiveGroupNorm(min(in_ch // 4, 32), in_ch, zemb_dim)
        
        # ✅ 【确认】使用标准 3x3 卷积 (Baseline配置)
        self.Conv_0 = conv3x3(in_ch, out_ch)
        
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
        h = self.Conv_0(h) 
        
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
                
        if not self.skip_rescale: return x + h
        else: return (x + h) / np.sqrt(2.)

class ResnetBlockBigGANpp_Adagn(nn.Module):
    """ (BigGAN 风格) 残差块 - Baseline 版本 (标准 Conv3x3) """
    def __init__(self, act, in_ch, out_ch=None, temb_dim=None, zemb_dim=None, up=False, down=False,
                 dropout=0.1, fir=False, fir_kernel=(1, 3, 3, 1),
                 skip_rescale=True, init_scale=0.):
        super().__init__()
        out_ch = out_ch if out_ch else in_ch
        self.GroupNorm_0 = AdaptiveGroupNorm(min(in_ch // 4, 32), in_ch, zemb_dim)
        
        self.up = up; self.down = down; self.fir = fir; self.fir_kernel = fir_kernel

        # ✅ 【确认】使用标准 3x3 卷积
        self.Conv_0 = conv3x3(in_ch, out_ch)
        
        if temb_dim is not None:
            self.Dense_0 = nn.Linear(temb_dim, out_ch)
            self.Dense_0.weight.data = default_init()(self.Dense_0.weight.shape)
            nn.init.zeros_(self.Dense_0.bias)
        
        self.GroupNorm_1 = AdaptiveGroupNorm(min(out_ch // 4, 32), out_ch, zemb_dim)
        self.Dropout_0 = nn.Dropout(dropout)
        self.Conv_1 = conv3x3(out_ch, out_ch, init_scale=init_scale)
        if in_ch != out_ch or up or down:
            self.Conv_2 = conv1x1(in_ch, out_ch)

        self.skip_rescale = skip_rescale; self.act = act; self.in_ch = in_ch; self.out_ch = out_ch

    def forward(self, x, temb=None, zemb=None):
        h = self.act(self.GroupNorm_0(x, zemb))

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
        
        if temb is not None: h += self.Dense_0(self.act(temb))[:, :, None, None]
        h = self.act(self.GroupNorm_1(h, zemb))
        h = self.Dropout_0(h)
        h = self.Conv_1(h)
        
        if self.in_ch != self.out_ch or self.up or self.down: x = self.Conv_2(x)

        if not self.skip_rescale: return x + h
        else: return (x + h) / np.sqrt(2.)

class ResnetBlockBigGANpp_Adagn_one(nn.Module):
    """ (BigGAN 风格 - 单次注入) - Baseline 版本 """
    def __init__(self, act, in_ch, out_ch=None, temb_dim=None, zemb_dim=None, up=False, down=False,
                 dropout=0.1, fir=False, fir_kernel=(1, 3, 3, 1),
                 skip_rescale=True, init_scale=0.):
        super().__init__()
        out_ch = out_ch if out_ch else in_ch
        self.GroupNorm_0 = AdaptiveGroupNorm(min(in_ch // 4, 32), in_ch, zemb_dim)
        self.up = up; self.down = down; self.fir = fir; self.fir_kernel = fir_kernel

        # ✅ 【确认】使用标准 3x3 卷积
        self.Conv_0 = conv3x3(in_ch, out_ch)
        
        if temb_dim is not None:
            self.Dense_0 = nn.Linear(temb_dim, out_ch)
            self.Dense_0.weight.data = default_init()(self.Dense_0.weight.shape)
            nn.init.zeros_(self.Dense_0.bias)
        
        self.GroupNorm_1 = nn.GroupNorm(num_groups=min(out_ch // 4, 32), num_channels=out_ch, eps=1e-6)
        self.Dropout_0 = nn.Dropout(dropout)
        self.Conv_1 = conv3x3(out_ch, out_ch, init_scale=init_scale)
        if in_ch != out_ch or up or down: self.Conv_2 = conv1x1(in_ch, out_ch)

        self.skip_rescale = skip_rescale; self.act = act; self.in_ch = in_ch; self.out_ch = out_ch

    def forward(self, x, temb=None, zemb=None):
        h = self.act(self.GroupNorm_0(x, zemb))
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
        
        if temb is not None: h += self.Dense_0(self.act(temb))[:, :, None, None]
        h = self.act(self.GroupNorm_1(h)) 
        h = self.Dropout_0(h)
        h = self.Conv_1(h)
        
        if self.in_ch != self.out_ch or self.up or self.down: x = self.Conv_2(x)

        if not self.skip_rescale: return x + h
        else: return (x + h) / np.sqrt(2.)