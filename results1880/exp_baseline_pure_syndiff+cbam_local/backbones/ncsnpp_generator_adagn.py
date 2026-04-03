# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# ... (版权和许可信息) ...
# ---------------------------------------------------------------

# coding=utf-8
# Copyright 2020 The Google Research Authors.
# ... (Apache 许可信息) ...

# pylint: skip-file
''' 
代码改编自: https://github.com/yang-song/score_sde_pytorch/blob/main/models/ncsnpp.py
'''

from . import utils, layers, layerspp, dense_layer # 导入所有基础“积木”模块
import torch.nn as nn
import functools
import torch
import numpy as np


# --- 从自定义模块中分配函数(别名)，方便本文件调用 ---
# ResnetBlockDDPM 和 ResnetBlockBigGAN 是【带条件】的残差块 (来自 layerspp.py)
ResnetBlockDDPM = layerspp.ResnetBlockDDPMpp_Adagn
ResnetBlockBigGAN = layerspp.ResnetBlockBigGANpp_Adagn
ResnetBlockBigGAN_one = layerspp.ResnetBlockBigGANpp_Adagn_one
Combine = layerspp.Combine # 用于融合跳跃连接
conv3x3 = layerspp.conv3x3 # 带 DDPM 初始化的 3x3 卷积
conv1x1 = layerspp.conv1x1 # 带 DDPM 初始化的 1x1 卷积
get_act = layers.get_act # 获取激活函数 (如 SiLU)
default_initializer = layers.default_init # DDPM 默认权重初始化器
dense = dense_layer.dense # 带自定义初始化的全连接层


def _to_int_set(value):
    """兼容 argparse 传入的 int / list / tuple / 字符串形式分辨率配置。"""
    if value is None:
        return set()
    if isinstance(value, int):
        return {value}
    if isinstance(value, (list, tuple, set)):
        return {int(v) for v in value}
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return set()
        text = text.replace(',', ' ')
        return {int(v) for v in text.split()}
    raise TypeError(f'Unsupported resolution config type: {type(value)}')

class PixelNorm(nn.Module):
    """
    【模块】像素归一化 (Pixel Normalization)。
    
    来自 StyleGAN 的技术。它在“通道”维度上对每个像素进行归一化。
    用于将输入的风格向量 z 映射到一个更均匀的分布空间。
    """
    def __init__(self):
        super().__init__()

    def forward(self, input):
        # input 形状: [B, C]
        # torch.mean(input ** 2, dim=1, keepdim=True) 计算每个样本的L2范数的平方
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


class MS_CBAMpp(nn.Module):
    """
    多尺度 CBAMpp：更轻量，适合在多个高分辨率层做局部增强。
    """
    def __init__(self, channels, reduction=16, spatial_scales=[3, 5, 7], skip_rescale=False):
        super().__init__()
        hidden = max(channels // reduction, 4)

        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        nn.init.constant_(self.mlp[-1].weight, 0.)
        if self.mlp[-1].bias is not None:
            nn.init.constant_(self.mlp[-1].bias, 0.)

        self.spatial_convs = nn.ModuleList([
            nn.Conv2d(2, 1, kernel_size=k, padding=k // 2, bias=False)
            for k in spatial_scales
        ])
        self.spatial_bn = nn.BatchNorm2d(len(spatial_scales))
        self.spatial_fuse = nn.Conv2d(len(spatial_scales), 1, kernel_size=1, bias=False)
        for conv in self.spatial_convs:
            nn.init.constant_(conv.weight, 0.)
        nn.init.constant_(self.spatial_fuse.weight, 1.0 / len(spatial_scales))

        self.skip_rescale = skip_rescale

    def forward(self, x):
        avg = torch.mean(x, dim=(2, 3), keepdim=True)
        mx = torch.amax(x, dim=(2, 3), keepdim=True)
        ca = torch.sigmoid(self.mlp(avg) + self.mlp(mx))
        h = x * ca

        avg_c = torch.mean(h, dim=1, keepdim=True)
        mx_c = torch.amax(h, dim=1, keepdim=True)
        spatial_feat = torch.cat([avg_c, mx_c], dim=1)

        multi_scale_feat = [conv(spatial_feat) for conv in self.spatial_convs]
        multi_scale_feat = torch.cat(multi_scale_feat, dim=1)
        multi_scale_feat = self.spatial_bn(multi_scale_feat)
        sa = torch.sigmoid(self.spatial_fuse(multi_scale_feat))
        h = h * sa

        return (x + h) / np.sqrt(2.) if self.skip_rescale else (x + h)


@utils.register_model(name='ncsnpp') # 【关键】将这个类注册到全局 _MODELS 字典中，名称为 'ncsnpp'
class NCSNpp(nn.Module):
  """【核心】NCSN++ U-Net 模型架构 (扩散模型的生成器)"""

  def __init__(self, config):
    super().__init__()
    self.config = config
    self.not_use_tanh = config.not_use_tanh # 输出层是否使用 Tanh
    self.act = act = nn.SiLU() # 【关键】使用 SiLU (Swish) 作为全局激活函数
    self.z_emb_dim = z_emb_dim = config.z_emb_dim # 风格/条件 z 向量的维度 (例如 256)
    
    self.nf = nf = config.num_channels_dae # U-Net 的基础通道数 (宽度)
    ch_mult = config.ch_mult # 通道乘数列表，定义了 U-Net 的深度和宽度 (例如 [1, 1, 2, 2, 4, 4])
    self.num_res_blocks = num_res_blocks = config.num_res_blocks # 每个分辨率下的 ResBlock 数量
    self.attn_resolutions = attn_resolutions = _to_int_set(config.attn_resolutions) # 在哪些分辨率下使用“注意力”模块
    dropout = config.dropout
    resamp_with_conv = config.resamp_with_conv # 上/下采样是否带卷积
    self.num_resolutions = num_resolutions = len(ch_mult) # 分辨率层级的总数
    # 计算所有分辨率 (例如 [256, 128, 64, 32, 16, 8])
    self.all_resolutions = all_resolutions = [config.image_size // (2 ** i) for i in range(num_resolutions)]

    self.conditional = conditional = config.conditional  # 是否为“噪声条件” (即是否使用时间 t)
    fir = config.fir # 是否使用 FIR 平滑采样
    fir_kernel = config.fir_kernel
    self.skip_rescale = skip_rescale = config.skip_rescale # 是否缩放残差连接
    self.resblock_type = resblock_type = config.resblock_type.lower() # 残差块的风格 ('ddpm' 或 'biggan')
    self.progressive = progressive = config.progressive.lower() # U-Net 输出模式 (多尺度/渐进式)
    self.progressive_input = progressive_input = config.progressive_input.lower() # U-Net 输入模式
    self.embedding_type = embedding_type = config.embedding_type.lower() # 时间 t 嵌入的类型
    init_scale = 0. # 某些层的初始化缩放因子
    assert progressive in ['none', 'output_skip', 'residual']
    assert progressive_input in ['none', 'input_skip', 'residual']
    assert embedding_type in ['fourier', 'positional']
    combine_method = config.progressive_combine.lower()
    combiner = functools.partial(Combine, method=combine_method) # 跳跃连接的融合方法 (cat 或 sum)

    modules = [] # 这是一个列表，我们将按顺序把 U-Net 的所有层都放进去
    
    # --- 1. 时间步 t 的嵌入 (Embedding) ---
    if embedding_type == 'fourier':
      # 方案 A: 高斯傅里叶投影
      #assert config.training.continuous, "Fourier features are only used for continuous training."

      modules.append(layerspp.GaussianFourierProjection(
        embedding_size=nf, scale=config.fourier_scale
      ))
      embed_dim = 2 * nf # (因为 cat([sin, cos]))

    elif embedding_type == 'positional':
      # 方案 B: 标准的正弦位置编码
      embed_dim = nf

    else:
      raise ValueError(f'embedding type {embedding_type} unknown.')

    if conditional:
      # (b) 如果有时间条件，添加一个 MLP (2层线性层) 来处理 temb
      modules.append(nn.Linear(embed_dim, nf * 4))
      modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
      nn.init.zeros_(modules[-1].bias)
      modules.append(nn.Linear(nf * 4, nf * 4))
      modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
      nn.init.zeros_(modules[-1].bias)

    # --- 2. 准备各种“积木” (带配置参数) ---
    
    # 创建一个 AttnBlock 的“预设” (partial)，锁定参数
    AttnBlock = functools.partial(layerspp.AttnBlockpp,
                                  init_scale=init_scale,
                                  skip_rescale=skip_rescale)

    self.local_attn_resolutions = _to_int_set(getattr(config, 'local_attn_resolutions', []))
    local_attn_type = getattr(config, 'local_attn_type', 'none')
    local_attn_type = (local_attn_type or 'none').lower()
    if local_attn_type == 'cbam':
      LocalAttnBlock = functools.partial(MS_CBAMpp, skip_rescale=skip_rescale, spatial_scales=[3, 5, 7])
    elif local_attn_type == 'scsa':
      LocalAttnBlock = functools.partial(layerspp.SCSALitepp, skip_rescale=skip_rescale)
    elif local_attn_type in ['none', '']:
      LocalAttnBlock = None
    else:
      raise ValueError(f'local_attn_type {local_attn_type} not supported')

    # 创建一个 Upsample 的“预设”
    Upsample = functools.partial(layerspp.Upsample,
                                   with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

    if progressive == 'output_skip':
      self.pyramid_upsample = layerspp.Upsample(fir=fir, fir_kernel=fir_kernel, with_conv=False)
    elif progressive == 'residual':
      pyramid_upsample = functools.partial(layerspp.Upsample,
                                           fir=fir, fir_kernel=fir_kernel, with_conv=True)

    # 创建一个 Downsample 的“预设”
    Downsample = functools.partial(layerspp.Downsample,
                                    with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

    if progressive_input == 'input_skip':
      self.pyramid_downsample = layerspp.Downsample(fir=fir, fir_kernel=fir_kernel, with_conv=False)
    elif progressive_input == 'residual':
      pyramid_downsample = functools.partial(layerspp.Downsample,
                                              fir=fir, fir_kernel=fir_kernel, with_conv=True)

    # --- 3. 根据配置，选择“残差块” (Resnet Block) 的类型 ---
    # 【关键】: SynDiff 使用的是带 AdaGN 的 'biggan' 风格残差块
    if resblock_type == 'ddpm':
      ResnetBlock = functools.partial(ResnetBlockDDPM,
                                      act=act,
                                      dropout=dropout,
                                      init_scale=init_scale,
                                      skip_rescale=skip_rescale,
                                      temb_dim=nf * 4, # 时间 t 嵌入的维度
                                      zemb_dim = z_emb_dim) # 风格 z 嵌入的维度

    elif resblock_type == 'biggan':
      ResnetBlock = functools.partial(ResnetBlockBigGAN,
                                      act=act,
                                      dropout=dropout,
                                      fir=fir,
                                      fir_kernel=fir_kernel,
                                      init_scale=init_scale,
                                      skip_rescale=skip_rescale,
                                      temb_dim=nf * 4,
                                      zemb_dim = z_emb_dim)
    elif resblock_type == 'biggan_oneadagn':
      ResnetBlock = functools.partial(ResnetBlockBigGAN_one,
                                      act=act,
                                      dropout=dropout,
                                      fir=fir,
                                      fir_kernel=fir_kernel,
                                      init_scale=init_scale,
                                      skip_rescale=skip_rescale,
                                      temb_dim=nf * 4,
                                      zemb_dim = z_emb_dim)

    else:
      raise ValueError(f'resblock type {resblock_type} unrecognized.')

    # --- 4. 构建 U-Net 编码器 (Encoder / Downsampling) ---
    
    # 【核心修改】：分离输入和输出通道
    # SynDiff 的输入是 (Noise + Condition)，所以输入通道是 2 * config.num_channels
    # 输出是预测的 Image/Noise，所以输出通道是 config.num_channels
    self.input_channels = 2 * config.num_channels 
    self.output_channels = config.num_channels

    if progressive_input != 'none':
      input_pyramid_ch = self.input_channels

    modules.append(conv3x3(self.input_channels, nf)) # 初始卷积 (修改为 input_channels)
    hs_c = [nf] # hs_c 用于存储所有“跳跃连接” (Skip Connection) 的特征图通道数

    in_ch = nf
    for i_level in range(num_resolutions): # 循环遍历每个分辨率层级 (例如 6 层)
      # -- 在该层级添加 N 个 ResBlock --
      for i_block in range(num_res_blocks):
        out_ch = nf * ch_mult[i_level] # 计算该层级的目标通道数
        modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
        in_ch = out_ch

        if (LocalAttnBlock is not None) and (all_resolutions[i_level] in self.local_attn_resolutions):
          modules.append(LocalAttnBlock(channels=in_ch))

        # -- 在指定分辨率层级添加注意力块 --
        if all_resolutions[i_level] in attn_resolutions:
          modules.append(AttnBlock(channels=in_ch))
        hs_c.append(in_ch) # 存储这个 ResBlock 的输出通道数，供解码器使用

      # -- 添加下采样块 (如果不是最底层) --
      if i_level != num_resolutions - 1:
        if resblock_type == 'ddpm':
          modules.append(Downsample(in_ch=in_ch)) # DDPM 风格下采样
        else:
          modules.append(ResnetBlock(down=True, in_ch=in_ch)) # BigGAN 风格下采样 (集成在 ResBlock 中)

        # (处理“渐进式输入”的跳跃连接，逻辑复杂)
        if progressive_input == 'input_skip':
          modules.append(combiner(dim1=input_pyramid_ch, dim2=in_ch))
          if combine_method == 'cat':
            in_ch *= 2

        elif progressive_input == 'residual':
          modules.append(pyramid_downsample(in_ch=input_pyramid_ch, out_ch=in_ch))
          input_pyramid_ch = in_ch

        hs_c.append(in_ch) # 存储下采样后的通道数

    # --- 5. 构建 U-Net 瓶颈 (Bottleneck) ---
    in_ch = hs_c[-1] # 获取 U-Net 最底层的通道数
    modules.append(ResnetBlock(in_ch=in_ch))
    modules.append(AttnBlock(channels=in_ch)) # 在最底层应用注意力
    modules.append(ResnetBlock(in_ch=in_ch))

    pyramid_ch = 0
    # --- 6. 构建 U-Net 解码器 (Decoder / Upsampling) ---
    for i_level in reversed(range(num_resolutions)): # 反向遍历 (从底层到顶层)
      # (num_res_blocks + 1) 是因为解码器通常比编码器多一个块
      for i_block in range(num_res_blocks + 1):
        out_ch = nf * ch_mult[i_level]
        # 【关键】解码器输入 = 上一层的 (in_ch) + 对应的跳跃连接 (hs_c.pop())
        modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(), 
                                   out_ch=out_ch))
        in_ch = out_ch

      # --- 添加注意力块 ---
      if (LocalAttnBlock is not None) and (all_resolutions[i_level] in self.local_attn_resolutions):
        modules.append(LocalAttnBlock(channels=in_ch))

      if all_resolutions[i_level] in attn_resolutions:
        modules.append(AttnBlock(channels=in_ch))

      # (处理“渐进式输出”，这是一种多尺度输出技术)
      if progressive != 'none':
        if i_level == num_resolutions - 1: # 如果在最底层
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
        else: # 如果在中间层
          if progressive == 'output_skip':
            # ... (为每个中间层级添加一个输出头)
            modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                        num_channels=in_ch, eps=1e-6))
            modules.append(conv3x3(in_ch, self.output_channels, bias=True, init_scale=init_scale))
            pyramid_ch = self.output_channels
          elif progressive == 'residual':
            modules.append(pyramid_upsample(in_ch=pyramid_ch, out_ch=in_ch))
            pyramid_ch = in_ch
          else:
            raise ValueError(f'{progressive} is not a valid name')

      # --- 添加上采样块 (如果不是最顶层) ---
      if i_level != 0:
        if resblock_type == 'ddpm':
          modules.append(Upsample(in_ch=in_ch)) # DDPM 风格上采样
        else:
          modules.append(ResnetBlock(in_ch=in_ch, up=True)) # BigGAN 风格上采样

    assert not hs_c # 确保所有的跳跃连接都已被正确使用 (pop完)

    # --- 7. 最终输出层 ---
    if progressive != 'output_skip':
      # (标准 U-Net 的最终层)
      modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                  num_channels=in_ch, eps=1e-6))
      # 【核心修改】：输出通道改为 self.output_channels (3)
      modules.append(conv3x3(in_ch, self.output_channels, init_scale=init_scale))

    self.all_modules = nn.ModuleList(modules) # 将所有层打包成一个 ModuleList
    
    
    # --- 8. 风格向量 z 的 MLP 映射网络 (Mapping Network) ---
    # (这部分与时间 t 的 MLP 是分开的)
    mapping_layers = [PixelNorm(), # 1. 像素归一化
                      dense(config.nz, z_emb_dim), # 2. 线性层 (nz -> z_emb_dim)
                      self.act,]
    for _ in range(config.n_mlp): # 添加 N 个 MLP 隐藏层
        mapping_layers.append(dense(z_emb_dim, z_emb_dim))
        mapping_layers.append(self.act)
    self.z_transform = nn.Sequential(*mapping_layers) # 打包成 z_transform
    

  def forward(self, x, time_cond, z):
    # --- U-Net 的主前向传播 ---
    # x = 噪点图 x_t (或 x_t 与条件的拼接)
    # time_cond = 时间步 t (或 sigma)
    # z = 风格向量 z (随机噪声)
    
    # --- 1. 计算条件嵌入 ---
    # (a) 计算 z 嵌入 (zemb), [B, nz] -> [B, z_emb_dim]
    zemb = self.z_transform(z)
    
    modules = self.all_modules # 获取所有模块
    m_idx = 0 # 模块索引，用于按顺序调用
    
    # (b) 计算 t 嵌入 (temb)
    if self.embedding_type == 'fourier':
      # (方案 A: 高斯傅里叶)
      used_sigmas = time_cond
      temb = modules[m_idx](torch.log(used_sigmas))
      m_idx += 1
    elif self.embedding_type == 'positional':
      # (方案 B: 正弦位置编码)
      timesteps = time_cond
      temb = layers.get_timestep_embedding(timesteps, self.nf)
    else:
      raise ValueError(f'embedding type {self.embedding_type} unknown.')

    if self.conditional:
      # (c) 通过 MLP 处理 temb
      temb = modules[m_idx](temb)
      m_idx += 1
      temb = modules[m_idx](self.act(temb))
      m_idx += 1
    else:
      temb = None # (如果是非条件模型)

    if not self.config.centered:
      # (如果输入是 [0, 1]，则转换为 [-1, 1])
      x = 2 * x - 1.

    # --- 2. 编码器 (Downsampling) ---
    input_pyramid = None
    if self.progressive_input != 'none':
      input_pyramid = x

    hs = [modules[m_idx](x)] # 运行初始卷积
    m_idx += 1
    for i_level in range(self.num_resolutions):
      for i_block in range(self.num_res_blocks):
        # 【关键】将 temb 和 zemb 注入残差块
        h = modules[m_idx](hs[-1], temb, zemb) 
        m_idx += 1
        if (len(self.local_attn_resolutions) > 0) and (h.shape[-1] in self.local_attn_resolutions):
          h = modules[m_idx](h)
          m_idx += 1
        if h.shape[-1] in self.attn_resolutions: # 如果需要，应用注意力
          h = modules[m_idx](h)
          m_idx += 1
        hs.append(h) # 存储跳跃连接

      if i_level != self.num_resolutions - 1: # 如果不是最底层
        if self.resblock_type == 'ddpm':
          h = modules[m_idx](hs[-1])
          m_idx += 1
        else:
          h = modules[m_idx](hs[-1], temb, zemb) # 下采样块也需要 temb/zemb
          m_idx += 1

        # (处理“渐进式输入”的跳跃连接)
        if self.progressive_input == 'input_skip':
          input_pyramid = self.pyramid_downsample(input_pyramid)
          h = modules[m_idx](input_pyramid, h)
          m_idx += 1
        elif self.progressive_input == 'residual':
          input_pyramid = modules[m_idx](input_pyramid)
          m_idx += 1
          if self.skip_rescale:
            input_pyramid = (input_pyramid + h) / np.sqrt(2.)
          else:
            input_pyramid = input_pyramid + h
          h = input_pyramid
        
        hs.append(h)

    # --- 3. 瓶颈 (Bottleneck) ---
    h = hs[-1] # U-Net 最底层的特征
    h = modules[m_idx](h, temb, zemb); m_idx += 1 # ResBlock
    h = modules[m_idx](h); m_idx += 1 # AttnBlock
    h = modules[m_idx](h, temb, zemb); m_idx += 1 # ResBlock

    pyramid = None

    # --- 4. 解码器 (Upsampling) ---
    for i_level in reversed(range(self.num_resolutions)): # 反向遍历
      for i_block in range(self.num_res_blocks + 1):
        # 【关键】融合 (cat) 上一层的 h 和对应的跳跃连接 hs.pop()
        h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb, zemb)
        m_idx += 1

      if (len(self.local_attn_resolutions) > 0) and (h.shape[-1] in self.local_attn_resolutions):
        h = modules[m_idx](h)
        m_idx += 1

      if h.shape[-1] in self.attn_resolutions:
        h = modules[m_idx](h)
        m_idx += 1

      # (处理“渐进式输出”)
      if self.progressive != 'none':
        if i_level == self.num_resolutions - 1:
          if self.progressive == 'output_skip':
            pyramid = self.act(modules[m_idx](h))
            m_idx += 1
            pyramid = modules[m_idx](pyramid)
            m_idx += 1
          elif self.progressive == 'residual':
            pyramid = self.act(modules[m_idx](h))
            m_idx += 1
            pyramid = modules[m_idx](pyramid)
            m_idx += 1
          else:
            raise ValueError(f'{self.progressive} is not a valid name.')
        else:
          if self.progressive == 'output_skip':
            pyramid = self.pyramid_upsample(pyramid)
            pyramid_h = self.act(modules[m_idx](h))
            m_idx += 1
            pyramid_h = modules[m_idx](pyramid_h)
            m_idx += 1
            pyramid = pyramid + pyramid_h
          elif self.progressive == 'residual':
            pyramid = modules[m_idx](pyramid)
            m_idx += 1
            if self.skip_rescale:
              pyramid = (pyramid + h) / np.sqrt(2.)
            else:
              pyramid = pyramid + h
            h = pyramid
          else:
            raise ValueError(f'{self.progressive} is not a valid name')

      if i_level != 0: # 如果不是最顶层
        if self.resblock_type == 'ddpm':
          h = modules[m_idx](h)
          m_idx += 1
        else:
          h = modules[m_idx](h, temb, zemb) # 上采样块也需要 temb/zemb
          m_idx += 1

    assert not hs # 确保所有跳跃连接都已用完

    # --- 5. 最终输出 ---
    if self.progressive == 'output_skip':
      h = pyramid # (如果使用渐进式输出，则 h 是多尺度融合的结果)
    else:
      # (标准 U-Net 的最终层)
      h = self.act(modules[m_idx](h))
      m_idx += 1
      h = modules[m_idx](h)
      m_idx += 1

    assert m_idx == len(modules) # 确保所有模块都已使用
    
    if not self.not_use_tanh:
      return torch.tanh(h) # 默认使用 Tanh 将输出缩放到 [-1, 1]
    else:
      return h
