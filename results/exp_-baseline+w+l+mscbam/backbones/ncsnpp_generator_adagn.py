# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# ---------------------------------------------------------------

# coding=utf-8
# Copyright 2020 The Google Research Authors.
# ... (Apache 许可信息) ...

# pylint: skip-file
'''
代码改编自: https://github.com/yang-song/score_sde_pytorch/blob/main/models/ncsnpp.py
'''

from . import utils, layers, layerspp, dense_layer
import torch.nn as nn
import functools
import torch
import numpy as np

ResnetBlockDDPM = layerspp.ResnetBlockDDPMpp_Adagn
ResnetBlockBigGAN = layerspp.ResnetBlockBigGANpp_Adagn
ResnetBlockBigGAN_one = layerspp.ResnetBlockBigGANpp_Adagn_one
Combine = layerspp.Combine
conv3x3 = layerspp.conv3x3
conv1x1 = layerspp.conv1x1
get_act = layers.get_act
default_initializer = layers.default_init
dense = dense_layer.dense


# 先补全必要的导入（如果没有的话）
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ========== 粘贴MS_CBAMpp代码 ==========
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
        if self.mlp[-1].bias is not None:
            nn.init.constant_(self.mlp[-1].bias, 0.)

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
# ========== MS_CBAMpp代码结束 ==========

# 原文件的PixelNorm类（接着你的代码）
class PixelNorm(nn.Module):
    """StyleGAN pixel norm"""
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)
    
class PixelNorm(nn.Module):
    """StyleGAN pixel norm"""
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)

def _to_int_set(x):
    """
    Robustly convert config fields like (16,), [16,8], "16", "16,8", etc. into set(int).
    """
    if x is None:
        return set()
    if isinstance(x, set):
        return set(int(v) for v in x)
    if isinstance(x, (list, tuple)):
        out = []
        for v in x:
            if isinstance(v, str):
                v = v.strip()
                if v == '':
                    continue
                out.append(int(v))
            else:
                out.append(int(v))
        return set(out)
    if isinstance(x, str):
        s = x.replace('(', '').replace(')', '').replace('[', '').replace(']', '')
        parts = [p.strip() for p in s.split(',') if p.strip() != '']
        if len(parts) == 0:
            # maybe space separated
            parts = [p.strip() for p in s.split(' ') if p.strip() != '']
        return set(int(p) for p in parts)
    return {int(x)}

@utils.register_model(name='ncsnpp')
class NCSNpp(nn.Module):
    """NCSN++ U-Net (SynDiff generator) with optional local attention."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.not_use_tanh = config.not_use_tanh
        self.act = act = nn.SiLU()
        self.z_emb_dim = z_emb_dim = config.z_emb_dim

        self.nf = nf = config.num_channels_dae
        ch_mult = config.ch_mult
        self.num_res_blocks = num_res_blocks = config.num_res_blocks

        # ✅ robust parsing
        self.attn_resolutions = _to_int_set(getattr(config, 'attn_resolutions', []))
        dropout = config.dropout
        resamp_with_conv = config.resamp_with_conv
        self.num_resolutions = num_resolutions = len(ch_mult)
        self.all_resolutions = all_resolutions = [config.image_size // (2 ** i) for i in range(num_resolutions)]

        self.conditional = conditional = config.conditional
        fir = config.fir
        fir_kernel = config.fir_kernel
        self.skip_rescale = skip_rescale = config.skip_rescale
        self.resblock_type = resblock_type = config.resblock_type.lower()
        self.progressive = progressive = config.progressive.lower()
        self.progressive_input = progressive_input = config.progressive_input.lower()
        self.embedding_type = embedding_type = config.embedding_type.lower()
        init_scale = 0.
        assert progressive in ['none', 'output_skip', 'residual']
        assert progressive_input in ['none', 'input_skip', 'residual']
        assert embedding_type in ['fourier', 'positional']
        combine_method = config.progressive_combine.lower()
        combiner = functools.partial(Combine, method=combine_method)

        modules = []

        # --- 1. time embedding ---
        if embedding_type == 'fourier':
            modules.append(layerspp.GaussianFourierProjection(
                embedding_size=nf, scale=config.fourier_scale
            ))
            embed_dim = 2 * nf
        elif embedding_type == 'positional':
            embed_dim = nf
        else:
            raise ValueError(f'embedding type {embedding_type} unknown.')

        if conditional:
            modules.append(nn.Linear(embed_dim, nf * 4))
            modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)
            modules.append(nn.Linear(nf * 4, nf * 4))
            modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)

        # --- 2. building blocks ---
        AttnBlock = functools.partial(layerspp.AttnBlockpp,
                                      init_scale=init_scale,
                                      skip_rescale=skip_rescale)

        # ✅ local attention config (for ablation)
        self.local_attn_resolutions = _to_int_set(getattr(config, 'local_attn_resolutions', []))
        local_attn_type = getattr(config, 'local_attn_type', 'none')
        local_attn_type = (local_attn_type or 'none').lower()

        # 修改后的代码（第124-130行）
        if local_attn_type == 'cbam':
            # 替换：把layerspp.CBAMpp换成MS_CBAMpp，新增spatial_scales参数
            LocalAttnBlock = functools.partial(MS_CBAMpp, skip_rescale=skip_rescale, spatial_scales=[3,5,7])
        elif local_attn_type == 'scsa':
            LocalAttnBlock = functools.partial(layerspp.SCSALitepp, skip_rescale=skip_rescale)
        elif local_attn_type in ['none', '']:
            LocalAttnBlock = None
        else:
            raise ValueError(f'local_attn_type {local_attn_type} not supported')
        Upsample = functools.partial(layerspp.Upsample,
                                     with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

        if progressive == 'output_skip':
            self.pyramid_upsample = layerspp.Upsample(fir=fir, fir_kernel=fir_kernel, with_conv=False)
        elif progressive == 'residual':
            pyramid_upsample = functools.partial(layerspp.Upsample,
                                                 fir=fir, fir_kernel=fir_kernel, with_conv=True)

        Downsample = functools.partial(layerspp.Downsample,
                                       with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

        if progressive_input == 'input_skip':
            self.pyramid_downsample = layerspp.Downsample(fir=fir, fir_kernel=fir_kernel, with_conv=False)
        elif progressive_input == 'residual':
            pyramid_downsample = functools.partial(layerspp.Downsample,
                                                   fir=fir, fir_kernel=fir_kernel, with_conv=True)

        # --- 3. resblock type ---
        if resblock_type == 'ddpm':
            ResnetBlock = functools.partial(
                ResnetBlockDDPM,
                act=act, dropout=dropout, init_scale=init_scale, skip_rescale=skip_rescale,
                temb_dim=nf * 4, zemb_dim=z_emb_dim
            )
        elif resblock_type == 'biggan':
            ResnetBlock = functools.partial(
                ResnetBlockBigGAN,
                act=act, dropout=dropout, fir=fir, fir_kernel=fir_kernel,
                init_scale=init_scale, skip_rescale=skip_rescale,
                temb_dim=nf * 4, zemb_dim=z_emb_dim
            )
        elif resblock_type == 'biggan_oneadagn':
            ResnetBlock = functools.partial(
                ResnetBlockBigGAN_one,
                act=act, dropout=dropout, fir=fir, fir_kernel=fir_kernel,
                init_scale=init_scale, skip_rescale=skip_rescale,
                temb_dim=nf * 4, zemb_dim=z_emb_dim
            )
        else:
            raise ValueError(f'resblock type {resblock_type} unrecognized.')

        # --- 4. encoder ---
        self.input_channels = 2 * config.num_channels
        self.output_channels = config.num_channels

        if progressive_input != 'none':
            input_pyramid_ch = self.input_channels

        modules.append(conv3x3(self.input_channels, nf))
        hs_c = [nf]

        in_ch = nf
        for i_level in range(num_resolutions):
            for i_block in range(num_res_blocks):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
                in_ch = out_ch

                # ✅ local attention (high-res)
                if (LocalAttnBlock is not None) and (all_resolutions[i_level] in self.local_attn_resolutions):
                    modules.append(LocalAttnBlock(channels=in_ch))

                # global attention (low-res)
                if all_resolutions[i_level] in self.attn_resolutions:
                    modules.append(AttnBlock(channels=in_ch))

                hs_c.append(in_ch)

            if i_level != num_resolutions - 1:
                if resblock_type == 'ddpm':
                    modules.append(Downsample(in_ch=in_ch))
                else:
                    modules.append(ResnetBlock(down=True, in_ch=in_ch))

                if progressive_input == 'input_skip':
                    modules.append(combiner(dim1=input_pyramid_ch, dim2=in_ch))
                    if combine_method == 'cat':
                        in_ch *= 2
                elif progressive_input == 'residual':
                    modules.append(pyramid_downsample(in_ch=input_pyramid_ch, out_ch=in_ch))
                    input_pyramid_ch = in_ch

                hs_c.append(in_ch)

        # --- 5. bottleneck ---
        in_ch = hs_c[-1]
        modules.append(ResnetBlock(in_ch=in_ch))
        modules.append(AttnBlock(channels=in_ch))  # keep bottleneck global attn
        modules.append(ResnetBlock(in_ch=in_ch))

        pyramid_ch = 0

        # --- 6. decoder ---
        for i_level in reversed(range(num_resolutions)):
            for i_block in range(num_res_blocks + 1):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(), out_ch=out_ch))
                in_ch = out_ch

            # ✅ local attention (same resolution as this level)
            if (LocalAttnBlock is not None) and (all_resolutions[i_level] in self.local_attn_resolutions):
                modules.append(LocalAttnBlock(channels=in_ch))

            # global attention
            if all_resolutions[i_level] in self.attn_resolutions:
                modules.append(AttnBlock(channels=in_ch))

            if progressive != 'none':
                if i_level == num_resolutions - 1:
                    if progressive == 'output_skip':
                        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                                    num_channels=in_ch, eps=1e-6))
                        modules.append(conv3x3(in_ch, self.output_channels, init_scale=init_scale))
                        pyramid_ch = self.output_channels
                    elif progressive == 'residual':
                        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                                    num_channels=in_ch, eps=1e-6))
                        modules.append(conv3x3(in_ch, in_ch, bias=True))
                        pyramid_ch = in_ch
                    else:
                        raise ValueError(f'{progressive} is not a valid name.')
                else:
                    if progressive == 'output_skip':
                        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                                    num_channels=in_ch, eps=1e-6))
                        modules.append(conv3x3(in_ch, self.output_channels, bias=True, init_scale=init_scale))
                        pyramid_ch = self.output_channels
                    elif progressive == 'residual':
                        modules.append(pyramid_upsample(in_ch=pyramid_ch, out_ch=in_ch))
                        pyramid_ch = in_ch
                    else:
                        raise ValueError(f'{progressive} is not a valid name')

            if i_level != 0:
                if resblock_type == 'ddpm':
                    modules.append(Upsample(in_ch=in_ch))
                else:
                    modules.append(ResnetBlock(in_ch=in_ch, up=True))

        assert not hs_c

        # --- 7. final output ---
        if progressive != 'output_skip':
            modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                        num_channels=in_ch, eps=1e-6))
            modules.append(conv3x3(in_ch, self.output_channels, init_scale=init_scale))

        self.all_modules = nn.ModuleList(modules)

        # --- 8. z mapping ---
        mapping_layers = [
            PixelNorm(),
            dense(config.nz, z_emb_dim),
            self.act,
        ]
        for _ in range(config.n_mlp):
            mapping_layers.append(dense(z_emb_dim, z_emb_dim))
            mapping_layers.append(self.act)
        self.z_transform = nn.Sequential(*mapping_layers)

    def forward(self, x, time_cond, z):
        zemb = self.z_transform(z)

        modules = self.all_modules
        m_idx = 0

        if self.embedding_type == 'fourier':
            used_sigmas = time_cond
            temb = modules[m_idx](torch.log(used_sigmas))
            m_idx += 1
        elif self.embedding_type == 'positional':
            timesteps = time_cond
            temb = layers.get_timestep_embedding(timesteps, self.nf)
        else:
            raise ValueError(f'embedding type {self.embedding_type} unknown.')

        if self.conditional:
            temb = modules[m_idx](temb); m_idx += 1
            temb = modules[m_idx](self.act(temb)); m_idx += 1
        else:
            temb = None

        if not self.config.centered:
            x = 2 * x - 1.

        # --- encoder ---
        input_pyramid = None
        if self.progressive_input != 'none':
            input_pyramid = x

        hs = [modules[m_idx](x)]
        m_idx += 1

        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = modules[m_idx](hs[-1], temb, zemb); m_idx += 1

                # ✅ local attn
                if (len(self.local_attn_resolutions) > 0) and (h.shape[-1] in self.local_attn_resolutions):
                    h = modules[m_idx](h); m_idx += 1

                # global attn
                if h.shape[-1] in self.attn_resolutions:
                    h = modules[m_idx](h); m_idx += 1

                hs.append(h)

            if i_level != self.num_resolutions - 1:
                if self.resblock_type == 'ddpm':
                    h = modules[m_idx](hs[-1]); m_idx += 1
                else:
                    h = modules[m_idx](hs[-1], temb, zemb); m_idx += 1

                if self.progressive_input == 'input_skip':
                    input_pyramid = self.pyramid_downsample(input_pyramid)
                    h = modules[m_idx](input_pyramid, h); m_idx += 1
                elif self.progressive_input == 'residual':
                    input_pyramid = modules[m_idx](input_pyramid); m_idx += 1
                    if self.skip_rescale:
                        input_pyramid = (input_pyramid + h) / np.sqrt(2.)
                    else:
                        input_pyramid = input_pyramid + h
                    h = input_pyramid

                hs.append(h)

        # --- bottleneck ---
        h = hs[-1]
        h = modules[m_idx](h, temb, zemb); m_idx += 1
        h = modules[m_idx](h); m_idx += 1
        h = modules[m_idx](h, temb, zemb); m_idx += 1

        pyramid = None

        # --- decoder ---
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb, zemb); m_idx += 1

            # ✅ local attn
            if (len(self.local_attn_resolutions) > 0) and (h.shape[-1] in self.local_attn_resolutions):
                h = modules[m_idx](h); m_idx += 1

            # global attn
            if h.shape[-1] in self.attn_resolutions:
                h = modules[m_idx](h); m_idx += 1

            if self.progressive != 'none':
                if i_level == self.num_resolutions - 1:
                    if self.progressive == 'output_skip':
                        pyramid = self.act(modules[m_idx](h)); m_idx += 1
                        pyramid = modules[m_idx](pyramid); m_idx += 1
                    elif self.progressive == 'residual':
                        pyramid = self.act(modules[m_idx](h)); m_idx += 1
                        pyramid = modules[m_idx](pyramid); m_idx += 1
                    else:
                        raise ValueError(f'{self.progressive} is not a valid name.')
                else:
                    if self.progressive == 'output_skip':
                        pyramid = self.pyramid_upsample(pyramid)
                        pyramid_h = self.act(modules[m_idx](h)); m_idx += 1
                        pyramid_h = modules[m_idx](pyramid_h); m_idx += 1
                        pyramid = pyramid + pyramid_h
                    elif self.progressive == 'residual':
                        pyramid = modules[m_idx](pyramid); m_idx += 1
                        if self.skip_rescale:
                            pyramid = (pyramid + h) / np.sqrt(2.)
                        else:
                            pyramid = pyramid + h
                        h = pyramid
                    else:
                        raise ValueError(f'{self.progressive} is not a valid name')

            if i_level != 0:
                if self.resblock_type == 'ddpm':
                    h = modules[m_idx](h); m_idx += 1
                else:
                    h = modules[m_idx](h, temb, zemb); m_idx += 1

        assert not hs

        # --- final ---
        if self.progressive == 'output_skip':
            h = pyramid
        else:
            h = self.act(modules[m_idx](h)); m_idx += 1
            h = modules[m_idx](h); m_idx += 1

        assert m_idx == len(modules)

        if not self.not_use_tanh:
            return torch.tanh(h)
        else:
            return h
