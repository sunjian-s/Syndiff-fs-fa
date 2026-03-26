# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# ... (版权和许可信息) ...
# ---------------------------------------------------------------

# pylint: skip-file
"""
定义 NCSN++ 所需的层 (Layers++)
改编自 https://github.com/yang-song/score_sde_pytorch/blob/main/models/layerspp.py
"""
from . import layers  # 导入基础层 (如 ddpm_conv3x3, NIN)
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
# 可变形卷积 v2 (DCNv2) - [保留定义以备对比实验]
# =========================================================================
class DeformableConv2d(nn.Module):
    """
    可变形卷积 v2 (DCNv2)
    注：在当前版本中，默认使用下方的 DynamicSnakeConv 替代此模块。
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 bias=True, init_scale=1.):
        super(DeformableConv2d, self).__init__()

        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size

        self.offset_conv = nn.Conv2d(
            in_channels,
            3 * kernel_size * kernel_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True
        )

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
        return torchvision.ops.deform_conv2d(
            input=x,
            offset=offset,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            mask=mask
        )

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
        self.style.bias.data[:in_channel] = 1  # gamma
        self.style.bias.data[in_channel:] = 0  # beta

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

# =========================================================================
# (NCSNpp 风格) 全局自注意力块
# =========================================================================
class AttnBlockpp(nn.Module):
    """(NCSNpp 风格的) 自注意力块"""
    def __init__(self, channels, skip_rescale=False, init_scale=0.):
        super().__init__()
        self.GroupNorm_0 = nn.GroupNorm(num_groups=min(channels // 4, 32),
                                        num_channels=channels, eps=1e-6)
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
# ✅ 新增：局部注意力（用于高分辨率 256/128/64）
# =========================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MS_CBAMpp(nn.Module):
    """多尺度CBAMpp：适配眼底细粒度特征（血管/小病灶）"""
    def __init__(self, channels, reduction=16, spatial_scales=[3,5,7], skip_rescale=False):
        super().__init__()
        hidden = max(channels // reduction, 4)
        
        # 1. 优化通道注意力：加入批归一化+SiLU，增强稳定性
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),  # 新增：稳定训练
            nn.SiLU(),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        # 通道注意力初始化：让初始输出接近0，避免过度抑制
        nn.init.constant_(self.mlp[-1].weight, 0.)
        nn.init.constant_(self.mlp[-1].bias, 0.) if self.mlp[-1].bias is not None else None

        # 2. 优化空间注意力：多尺度卷积（适配不同粗细的血管/病灶）
        self.spatial_convs = nn.ModuleList([
            nn.Conv2d(2, 1, kernel_size=k, padding=k//2, bias=False) 
            for k in spatial_scales
        ])
        self.spatial_bn = nn.BatchNorm2d(len(spatial_scales))  # 多尺度特征融合
        self.spatial_fuse = nn.Conv2d(len(spatial_scales), 1, kernel_size=1, bias=False)
        # 空间注意力初始化
        for conv in self.spatial_convs:
            nn.init.constant_(conv.weight, 0.)
        nn.init.constant_(self.spatial_fuse.weight, 1./len(spatial_scales))  # 初始均等融合

        self.skip_rescale = skip_rescale

    def forward(self, x):
        B, C, H, W = x.shape
        
        # 通道注意力：保留avg+max，加入BN增强稳定性
        avg = torch.mean(x, dim=(2, 3), keepdim=True)
        mx = torch.amax(x, dim=(2, 3), keepdim=True)
        ca = torch.sigmoid(self.mlp(avg) + self.mlp(mx))  # 通道权重
        h = x * ca

        # 空间注意力：多尺度卷积融合（适配不同尺度的血管/病灶）
        avg_c = torch.mean(h, dim=1, keepdim=True)
        mx_c = torch.amax(h, dim=1, keepdim=True)
        spatial_feat = torch.cat([avg_c, mx_c], dim=1)  # (B,2,H,W)
        
        # 多尺度卷积提取不同尺度空间特征
        multi_scale_feat = [conv(spatial_feat) for conv in self.spatial_convs]
        multi_scale_feat = torch.cat(multi_scale_feat, dim=1)  # (B,3,H,W)
        multi_scale_feat = self.spatial_bn(multi_scale_feat)
        sa = torch.sigmoid(self.spatial_fuse(multi_scale_feat))  # (B,1,H,W)
        h = h * sa

        # 残差连接（保持和原模块一致）
        return (x + h) / np.sqrt(2.) if self.skip_rescale else (x + h)
class SCSALitepp(nn.Module):
    """
    SCSA-Lite: 轻量版“空间(高宽) + 通道”协同注意力
    更偏细节/结构（血管、小病灶）增强，计算量低，适合插在高分辨率。
    """
    def __init__(self, channels, ks=(3, 7), reduction=16, skip_rescale=False):
        super().__init__()
        self.dw_h = nn.ModuleList([
            nn.Conv1d(channels, channels, k, padding=k // 2, groups=channels, bias=True) for k in ks
        ])
        self.dw_w = nn.ModuleList([
            nn.Conv1d(channels, channels, k, padding=k // 2, groups=channels, bias=True) for k in ks
        ])
        hidden = max(channels // reduction, 4)
        self.fc1 = nn.Conv2d(channels, hidden, 1, bias=True)
        self.act = nn.SiLU()
        self.fc2 = nn.Conv2d(hidden, channels, 1, bias=True)
        self.skip_rescale = skip_rescale

    def forward(self, x):
        B, C, H, W = x.shape
        x_h = x.mean(dim=3)  # (B,C,H)
        x_w = x.mean(dim=2)  # (B,C,W)

        h_sum = 0
        w_sum = 0
        for conv in self.dw_h:
            h_sum = h_sum + conv(x_h)
        for conv in self.dw_w:
            w_sum = w_sum + conv(x_w)

        a_h = h_sum.unsqueeze(-1)  # (B,C,H,1)
        a_w = w_sum.unsqueeze(-2)  # (B,C,1,W)
        spatial = torch.sigmoid(a_h + a_w)  # (B,C,H,W)

        y = (x * spatial).mean(dim=(2, 3), keepdim=True)  # (B,C,1,1)
        ch = torch.sigmoid(self.fc2(self.act(self.fc1(y))))

        h = x * spatial * ch
        return (x + h) / np.sqrt(2.) if self.skip_rescale else (x + h)

# =========================================================================
# Upsample / Downsample
# =========================================================================
class Upsample(nn.Module):
    """上采样模块"""
    def __init__(self, in_ch=None, out_ch=None, with_conv=False, fir=False, fir_kernel=(1, 3, 3, 1)):
        super().__init__()
        out_ch = out_ch if out_ch else in_ch
        if not fir:
            if with_conv:
                self.Conv_0 = conv3x3(in_ch, out_ch)
        else:
            if with_conv:
                self.Conv2d_0 = up_or_down_sampling.Conv2d(
                    in_ch, out_ch, kernel=3, up=True,
                    resample_kernel=fir_kernel, use_bias=True, kernel_init=default_init()
                )
        self.fir = fir
        self.with_conv = with_conv
        self.fir_kernel = fir_kernel
        self.out_ch = out_ch

    def forward(self, x):
        B, C, H, W = x.shape
        if not self.fir:
            h = F.interpolate(x, (H * 2, W * 2), 'nearest')
            if self.with_conv:
                h = self.Conv_0(h)
        else:
            if not self.with_conv:
                h = up_or_down_sampling.upsample_2d(x, self.fir_kernel, factor=2)
            else:
                h = self.Conv2d_0(x)
        return h

class Downsample(nn.Module):
    """下采样模块"""
    def __init__(self, in_ch=None, out_ch=None, with_conv=False, fir=False, fir_kernel=(1, 3, 3, 1)):
        super().__init__()
        out_ch = out_ch if out_ch else in_ch
        if not fir:
            if with_conv:
                self.Conv_0 = conv3x3(in_ch, out_ch, stride=2, padding=0)
        else:
            if with_conv:
                self.Conv2d_0 = up_or_down_sampling.Conv2d(
                    in_ch, out_ch, kernel=3, down=True,
                    resample_kernel=fir_kernel, use_bias=True, kernel_init=default_init()
                )
        self.fir = fir
        self.fir_kernel = fir_kernel
        self.with_conv = with_conv
        self.out_ch = out_ch

    def forward(self, x):
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
# ResNet Blocks
# =========================================================================
class ResnetBlockDDPMpp_Adagn(nn.Module):
    """
    (NCSNpp 风格) 残差块 - Baseline 版本
    【修改说明】已移除 DynamicSnakeConv，恢复为标准 conv3x3
    """
    def __init__(self, act, in_ch, out_ch=None, temb_dim=None, zemb_dim=None,
                 conv_shortcut=False, dropout=0.1, skip_rescale=False, init_scale=0.):
        super().__init__()
        out_ch = out_ch if out_ch else in_ch

        self.GroupNorm_0 = AdaptiveGroupNorm(min(in_ch // 4, 32), in_ch, zemb_dim)
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

        if not self.skip_rescale:
            return x + h
        else:
            return (x + h) / np.sqrt(2.)

class ResnetBlockBigGANpp_Adagn(nn.Module):
    """
    (BigGAN 风格) 残差块
    """
    def __init__(self, act, in_ch, out_ch=None, temb_dim=None, zemb_dim=None,
                 up=False, down=False, dropout=0.1, fir=False, fir_kernel=(1, 3, 3, 1),
                 skip_rescale=True, init_scale=0.):
        super().__init__()

        out_ch = out_ch if out_ch else in_ch
        self.GroupNorm_0 = AdaptiveGroupNorm(min(in_ch // 4, 32), in_ch, zemb_dim)

        self.up = up
        self.down = down
        self.fir = fir
        self.fir_kernel = fir_kernel

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

        self.skip_rescale = skip_rescale
        self.act = act
        self.in_ch = in_ch
        self.out_ch = out_ch

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

        if temb is not None:
            h += self.Dense_0(self.act(temb))[:, :, None, None]
        h = self.act(self.GroupNorm_1(h, zemb))
        h = self.Dropout_0(h)
        h = self.Conv_1(h)

        if self.in_ch != self.out_ch or self.up or self.down:
            x = self.Conv_2(x)

        if not self.skip_rescale:
            return x + h
        else:
            return (x + h) / np.sqrt(2.)

class ResnetBlockBigGANpp_Adagn_one(nn.Module):
    """
    (BigGAN 风格 - 单次注入) 残差块
    """
    def __init__(self, act, in_ch, out_ch=None, temb_dim=None, zemb_dim=None,
                 up=False, down=False, dropout=0.1, fir=False, fir_kernel=(1, 3, 3, 1),
                 skip_rescale=True, init_scale=0.):
        super().__init__()

        out_ch = out_ch if out_ch else in_ch
        self.GroupNorm_0 = AdaptiveGroupNorm(min(in_ch // 4, 32), in_ch, zemb_dim)

        self.up = up
        self.down = down
        self.fir = fir
        self.fir_kernel = fir_kernel

        self.Conv_0 = conv3x3(in_ch, out_ch)

        if temb_dim is not None:
            self.Dense_0 = nn.Linear(temb_dim, out_ch)
            self.Dense_0.weight.data = default_init()(self.Dense_0.weight.shape)
            nn.init.zeros_(self.Dense_0.bias)

        self.GroupNorm_1 = nn.GroupNorm(num_groups=min(out_ch // 4, 32),
                                        num_channels=out_ch, eps=1e-6)

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
            h += self.Dense_0(self.act(temb))[:, :, None, None]

        h = self.act(self.GroupNorm_1(h))
        h = self.Dropout_0(h)
        h = self.Conv_1(h)

        if self.in_ch != self.out_ch or self.up or self.down:
            x = self.Conv_2(x)

        if not self.skip_rescale:
            return x + h
        else:
            return (x + h) / np.sqrt(2.)
